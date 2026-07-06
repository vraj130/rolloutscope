"""Reproducible validation of rolloutscope detectors against Patronus TRACE.

TRACE (arXiv:2601.20103, https://huggingface.co/datasets/PatronusAI/trace-dataset)
is a labeled reward-hacking dataset of coding-agent trajectories. Each row carries
a ``conversation`` (a JSON transcript of user and assistant turns, assistant turns
holding tool calls of the form ``{name, parameters}``) and a ``label`` ("0" is
clean, any dotted code such as "1.2.3" is a reward hack). It provides no reward,
no metrics, and no ground-truth answer.

This script fetches a slice through the Hugging Face datasets-server, maps each
row faithfully into the normalized schema (prompt = user turns, completion =
assistant turns with tool calls remapped into the ``{function: {name, arguments}}``
shape the detectors read), and reports the per-row fire rate of ``verifier_tamper``
and ``answer_leakage_echo`` on hacked versus clean rows. Only those two detectors
are exercised: they are the ones this dataset can legitimately test (and, since
TRACE carries no answer or grading criteria, ``answer_leakage_echo`` is expected to
stay silent, which the table makes visible).

Usage:

    uv run python scripts/trace_validation.py

The dataset is gated, so a Hugging Face token is required: put ``HF_TOKEN`` in a
``.env`` file at the repository root, or export it into the environment. The
script skips (prints a reason and exits 0) whenever the network, the license, or
the row shape does not cooperate, so it is safe to run anywhere.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from rolloutscope.detectors import (
    AnswerLeakageEchoDetector,
    DetectorConfig,
    VerifierTamperDetector,
)
from rolloutscope.schema import validate_rollout

REPO_ROOT = Path(__file__).resolve().parent.parent

DATASET = "PatronusAI/trace-dataset"
HF_API_BASE = "https://huggingface.co/api/datasets"
SERVER_BASE = "https://datasets-server.huggingface.co"
TIMEOUT_SECONDS = 30
TARGET_ROWS = 300
PAGE = 100  # datasets-server /rows hard cap per request
OFFSET_CAP = 3000  # safety bound on pagination

# Licenses that permit fetching a slice for local automated validation. Anything
# else (or an unverifiable license) skips rather than guessing. TRACE is
# cc-by-sa-4.0, which is in this set.
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


class SkipSignal(Exception):
    """Raised to skip the run (network, license, or row shape did not cooperate)."""


def _load_env() -> None:
    """Load HF_TOKEN from a repo-root .env when python-dotenv is available.

    python-dotenv is a dev dependency; when it is absent the script falls back to
    whatever HF_TOKEN is already exported in the environment.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(REPO_ROOT / ".env")


def _get(url: str) -> Any:
    """GET a JSON document with optional bearer auth; problems raise SkipSignal."""
    headers = {"User-Agent": "rolloutscope-trace-validation"}
    token = os.environ.get("HF_TOKEN") or None
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            raise SkipSignal(f"HF API returned 401 for {url}; set HF_TOKEN in .env") from exc
        raise SkipSignal(f"HTTP {exc.code} from {url}: {exc}") from exc
    except (urllib.error.URLError, OSError, ValueError, TimeoutError) as exc:
        raise SkipSignal(f"network unavailable or bad response from {url}: {exc}") from exc


def _license() -> str:
    """Return the dataset license, or skip unless it is recognized permissive."""
    info = _get(f"{HF_API_BASE}/{DATASET}")
    card = info.get("cardData") or {}
    value = card.get("license") or info.get("license")
    if isinstance(value, list):
        value = value[0] if value else None
    if not isinstance(value, str):
        raise SkipSignal(f"could not verify a license for {DATASET}")
    if value.lower() not in PERMISSIVE_LICENSES:
        raise SkipSignal(f"license {value!r} is not in the recognized permissive set")
    return value


def _first_split() -> tuple[str, str]:
    """Discover a (config, split) pair from the datasets-server splits endpoint."""
    encoded = urllib.parse.quote(DATASET, safe="")
    payload = _get(f"{SERVER_BASE}/splits?dataset={encoded}")
    splits = payload.get("splits") or []
    if not splits:
        raise SkipSignal(f"datasets-server reports no splits for {DATASET}")
    return splits[0]["config"], splits[0]["split"]


def _page(config: str, split: str, offset: int) -> list[dict[str, Any]]:
    """Fetch one page of raw rows from the datasets-server rows endpoint."""
    encoded = urllib.parse.quote(DATASET, safe="")
    url = (
        f"{SERVER_BASE}/rows?dataset={encoded}"
        f"&config={urllib.parse.quote(config)}"
        f"&split={urllib.parse.quote(split)}"
        f"&offset={offset}&length={PAGE}"
    )
    return [entry.get("row", {}) for entry in _get(url).get("rows", [])]


def _is_hacked(label: Any) -> bool | None:
    """True for a reward-hack label, False for clean, None when unusable."""
    if not isinstance(label, str):
        return None
    stripped = label.strip()
    if not stripped:
        return None
    return stripped != "0"


def _map_row(row: dict[str, Any], index: int) -> Any:
    """Map one TRACE row into a normalized rollout, or None when unusable.

    prompt = user turns; completion = assistant turns with each tool call remapped
    from ``{name, parameters}`` into ``{function: {name, arguments}}``. reward,
    metrics, and answer stay empty because TRACE does not provide them (neither
    detector run here depends on reward or metrics).
    """
    conversation = row.get("conversation")
    if not isinstance(conversation, str):
        return None
    try:
        messages = json.loads(conversation)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(messages, list):
        return None

    prompt_messages: list[dict[str, Any]] = []
    completion_messages: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content") or ""
        if role == "user":
            prompt_messages.append({"role": "user", "content": content})
        elif role == "assistant":
            mapped_calls = []
            for call in message.get("tool_calls") or []:
                if isinstance(call, dict):
                    mapped_calls.append(
                        {
                            "type": "function",
                            "function": {
                                "name": str(call.get("name") or ""),
                                "arguments": json.dumps(call.get("parameters", {})),
                            },
                        }
                    )
            assistant: dict[str, Any] = {"role": "assistant", "content": content}
            if mapped_calls:
                assistant["tool_calls"] = mapped_calls
            completion_messages.append(assistant)

    if not completion_messages:
        return None

    return validate_rollout(
        {
            "kind": "single_turn",
            "example_id": index,
            "rollout_id": f"trace-{index}",
            "prompt": prompt_messages or None,
            "completion": completion_messages,
            "reward": 0.0,
            "metrics": {},
            "answer": None,
            "info": {},
            "is_completed": True,
            "is_truncated": False,
        }
    )


def _fired_count(detector: Any, rollouts: list[Any], cfg: DetectorConfig) -> int:
    """Number of rollouts on which the detector fired at least once."""
    ids = {rollout.rollout_id for rollout in rollouts}
    fired: set[str] = set()
    for verdict in detector.detect(rollouts, cfg):
        if verdict.fired:
            fired.update(rid for rid in verdict.rollout_ids if rid)
    return len(fired & ids)


def main() -> None:
    """Fetch, map, run the two detectors, and print the separation table."""
    _load_env()
    license_value = _license()
    config, split = _first_split()

    hacked: list[Any] = []
    clean: list[Any] = []
    fetched = 0
    index = 0
    offset = 0
    while len(hacked) + len(clean) < TARGET_ROWS and offset < OFFSET_CAP:
        batch = _page(config, split, offset)
        if not batch:
            break
        fetched += len(batch)
        for raw in batch:
            label = _is_hacked(raw.get("label"))
            if label is None:
                index += 1
                continue
            rollout = _map_row(raw, index)
            index += 1
            if rollout is None:
                continue
            (hacked if label else clean).append(rollout)
        offset += PAGE

    if not hacked or not clean:
        raise SkipSignal(f"could not derive both labels (hacked={len(hacked)}, clean={len(clean)})")

    cfg = DetectorConfig()
    detectors = {
        "verifier_tamper": VerifierTamperDetector(),
        "answer_leakage_echo": AnswerLeakageEchoDetector(),
    }

    print("TRACE validation: verifier_tamper and answer_leakage_echo")
    print(f"dataset: {DATASET}  (config={config}, split={split}, license={license_value})")
    print(
        "mapping: prompt=user turns, completion=assistant turns (content plus tool calls); "
        "reward, metrics, and answer are empty because TRACE does not provide them."
    )
    print(f"usable rows: {len(hacked)} hacked, {len(clean)} clean (from {fetched} fetched)")
    print()
    print(f"{'detector':<22}{'hacked fire':>14}{'clean fire':>14}{'gap':>8}")
    for name, detector in detectors.items():
        h_fired = _fired_count(detector, hacked, cfg)
        c_fired = _fired_count(detector, clean, cfg)
        h_rate = h_fired / len(hacked)
        c_rate = c_fired / len(clean)
        h_cell = f"{h_fired}/{len(hacked)} {h_rate:.2f}"
        c_cell = f"{c_fired}/{len(clean)} {c_rate:.2f}"
        print(f"{name:<22}{h_cell:>14}{c_cell:>14}{h_rate - c_rate:>+8.2f}")


if __name__ == "__main__":
    try:
        main()
    except SkipSignal as exc:
        print(f"SKIP: {exc}")
        sys.exit(0)
