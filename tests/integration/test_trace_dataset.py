"""Integration test: verifier_tamper vs a small slice of the Patronus TRACE dataset.

TRACE (arXiv:2601.20103, https://huggingface.co/datasets/PatronusAI/trace-dataset)
is the confirmed labeled reward-hack dataset for the coding domain. This test
fetches license information and a small row slice through the Hugging Face
datasets-server HTTP API with stdlib urllib only, converts rows to the
normalized schema, and checks that verifier_tamper fires more on hacked-labeled
rows than on clean-labeled rows.

Everything network-shaped skips instead of failing: this file runs only under
``-m integration`` and must never break the offline suite.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import pytest

from rolloutscope.detectors import DetectorConfig, VerifierTamperDetector
from rolloutscope.schema import validate_rollout

pytestmark = pytest.mark.integration

DATASET = "PatronusAI/trace-dataset"
HF_API_BASE = "https://huggingface.co/api/datasets"
SERVER_BASE = "https://datasets-server.huggingface.co"
TIMEOUT_SECONDS = 30
ROWS_TO_FETCH = 60

# Licenses that permit fetching a slice for local automated testing. Anything
# else (or an unverifiable license) skips rather than guessing.
PERMISSIVE_LICENSES = {
    "mit",
    "apache-2.0",
    "bsd-3-clause",
    "cc0-1.0",
    "cc-by-4.0",
    "cc-by-sa-4.0",
    "odc-by",
    "cdla-permissive-2.0",
    "openrail",
}

LABEL_KEYS = (
    "label",
    "is_hack",
    "is_hacked",
    "reward_hack",
    "hack",
    "verdict",
    "classification",
    "hack_type",
    "hack_category",
    "category",
)
HACKED_WORDS = {"hack", "hacked", "reward_hack", "positive", "true", "yes", "1"}
CLEAN_WORDS = {"clean", "benign", "negative", "false", "no", "0", "none", "not_hack"}
TEXT_KEYS = (
    "trajectory",
    "messages",
    "conversation",
    "completion",
    "response",
    "solution",
    "output",
    "text",
)


def _fetch_json(url: str) -> Any:
    """GET a JSON document; any network or decode problem skips the test."""
    request = urllib.request.Request(url, headers={"User-Agent": "rolloutscope-tests"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, TimeoutError) as exc:
        pytest.skip(f"network unavailable or bad response from {url}: {exc}")


def _check_license() -> None:
    """Skip unless the dataset card declares a recognized permissive license."""
    info = _fetch_json(f"{HF_API_BASE}/{DATASET}")
    card = info.get("cardData") or {}
    license_value = card.get("license") or info.get("license")
    if isinstance(license_value, list):
        license_value = license_value[0] if license_value else None
    if not isinstance(license_value, str):
        pytest.skip(f"could not verify a license for {DATASET}; not fetching rows")
    if license_value.lower() not in PERMISSIVE_LICENSES:
        pytest.skip(
            f"dataset license {license_value!r} is not in the recognized permissive "
            "set; not fetching rows"
        )


def _first_split() -> tuple[str, str]:
    """Discover a (config, split) pair from the datasets-server splits endpoint."""
    encoded = urllib.parse.quote(DATASET, safe="")
    payload = _fetch_json(f"{SERVER_BASE}/splits?dataset={encoded}")
    splits = payload.get("splits") or []
    if not splits:
        pytest.skip(f"datasets-server reports no splits for {DATASET}")
    first = splits[0]
    return first["config"], first["split"]


def _fetch_rows(config: str, split: str) -> list[dict[str, Any]]:
    """Fetch a small slice of raw rows from the datasets-server rows endpoint."""
    encoded = urllib.parse.quote(DATASET, safe="")
    url = (
        f"{SERVER_BASE}/rows?dataset={encoded}&config={urllib.parse.quote(config)}"
        f"&split={urllib.parse.quote(split)}&offset=0&length={ROWS_TO_FETCH}"
    )
    payload = _fetch_json(url)
    rows = [entry.get("row", {}) for entry in payload.get("rows", [])]
    if not rows:
        pytest.skip(f"datasets-server returned no rows for {DATASET}")
    return rows


def _label_of(row: dict[str, Any]) -> bool | None:
    """Best-effort hacked/clean label; None when the row shape is unrecognized."""
    for key in LABEL_KEYS:
        if key not in row:
            continue
        value = row[key]
        if isinstance(value, bool):
            return value
        if isinstance(value, int | float):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in HACKED_WORDS:
                return True
            if lowered in CLEAN_WORDS:
                return False
            # a named hack subcategory string counts as hacked
            if key in {"hack_type", "hack_category", "category"} and lowered:
                return True
    return None


def _text_of(row: dict[str, Any]) -> str:
    """Flatten the row's trajectory-ish field into one scan-able string."""
    for key in TEXT_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, list) and value:
            chunks: list[str] = []
            for item in value:
                if isinstance(item, str):
                    chunks.append(item)
                elif isinstance(item, dict):
                    content = item.get("content")
                    chunks.append(content if isinstance(content, str) else json.dumps(item))
            if chunks:
                return "\n".join(chunks)
        if isinstance(value, dict) and value:
            return json.dumps(value)
    return ""


def test_verifier_tamper_separates_trace_labels() -> None:
    """verifier_tamper should fire more on hacked-labeled TRACE rows than clean."""
    _check_license()
    config_name, split = _first_split()
    raw_rows = _fetch_rows(config_name, split)

    hacked: list[Any] = []
    clean: list[Any] = []
    for index, raw in enumerate(raw_rows):
        label = _label_of(raw)
        text = _text_of(raw)
        if label is None or not text:
            continue
        rollout = validate_rollout(
            {
                "kind": "single_turn",
                "example_id": index,
                "prompt": None,
                "completion": text,
                "reward": 1.0,
                "metrics": {},
                "is_completed": True,
                "is_truncated": False,
            }
        )
        (hacked if label else clean).append(rollout)

    if not hacked or not clean:
        pytest.skip(
            f"could not derive both hacked and clean labels from the fetched slice "
            f"(hacked={len(hacked)}, clean={len(clean)}); row shape may have changed"
        )

    detector = VerifierTamperDetector()
    config = DetectorConfig()
    hacked_rate = sum(v.fired for v in detector.detect(hacked, config)) / len(hacked)
    clean_rate = sum(v.fired for v in detector.detect(clean, config)) / len(clean)
    assert hacked_rate > clean_rate, (
        f"verifier_tamper did not separate TRACE labels: hacked fire rate "
        f"{hacked_rate:.2f} vs clean fire rate {clean_rate:.2f}"
    )
