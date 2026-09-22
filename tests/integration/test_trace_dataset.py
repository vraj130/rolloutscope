"""Integration test: the detector benchmark against the pinned TRACE revision.

This file used to carry its own row mapping, its own label parsing, and its own
fire-rate arithmetic, all of which disagreed with ``scripts/trace_validation.py``
and produced different numbers for the same dataset. It now calls the shared
library and asserts the properties that mapping is supposed to guarantee.

Everything network-shaped skips instead of failing: this file runs only under
``-m integration`` and must never break the offline suite.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from rolloutscope.benchmark import (
    BenchmarkReport,
    BenchmarkUnavailable,
    evaluate_rows,
    load_dotenv_token,
    load_manifest,
    load_rows,
    report_to_json,
)
from rolloutscope.detectors import DetectorConfig, discover_detectors

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_PATH = REPO_ROOT / "benchmarks" / "trace" / "manifest.json"
SPLIT = "tuning"
"""The holdout stays sealed until the Phase 2 detector repairs land."""


@pytest.fixture(scope="module")
def report() -> BenchmarkReport:
    """Evaluate the tuning split once for the whole module."""
    if not MANIFEST_PATH.is_file():
        pytest.skip(f"no benchmark manifest at {MANIFEST_PATH}")
    manifest = load_manifest(MANIFEST_PATH)
    load_dotenv_token(REPO_ROOT)
    try:
        rows, source, _ = load_rows(revision=manifest.dataset.revision)
    except BenchmarkUnavailable as exc:
        pytest.skip(str(exc))
    return evaluate_rows(
        rows,
        manifest,
        discover_detectors(),
        DetectorConfig(),
        split=SPLIT,
        source=source,
    )


def _metric(report: BenchmarkReport, detector: str):  # type: ignore[no-untyped-def]
    """Return one detector's metrics, failing loudly if it was not scored at all."""
    for metric in report.metrics:
        if metric.detector == detector:
            return metric
    pytest.fail(f"{detector} is missing from the report")


def test_every_manifest_row_in_the_split_was_fetched_unchanged(report: BenchmarkReport) -> None:
    """The revision pin must hold: same rows, same bytes, nothing missing."""
    assert report.manifest_verified, report.verification_notes
    assert report.rows_unusable == 0, report.unusable_by_reason
    assert report.rows_mapped == report.rows_requested


def test_every_selected_detector_appears_with_its_coverage(report: BenchmarkReport) -> None:
    """A detector that could not be scored must say so, not vanish."""
    reported = {metric.detector for metric in report.metrics}
    assert reported == set(discover_detectors())
    for metric in report.metrics:
        assert metric.coverage.denominator == report.rows_mapped


def test_detectors_without_their_inputs_are_insufficient_not_clean(
    report: BenchmarkReport,
) -> None:
    """TRACE ships transcripts only: no reward, no metrics, no reference answer.

    Every detector that gates on one of those must report zero coverage with the
    missing signal named. Counting that silence as true negatives would make the
    clean-side numbers look excellent while measuring nothing at all.
    """
    for name in (
        "reward_saturation_group_collapse",
        "length_inflation",
        "format_only_wins",
        "degenerate_repetition",
        "answer_leakage_echo",
    ):
        metric = _metric(report, name)
        assert metric.counts.units_scored == 0
        assert metric.counts.units_insufficient == report.rows_mapped
        assert metric.counts.true_negatives == 0
        assert metric.recall.value is None
        assert metric.counts.insufficient_by_reason


def test_verifier_tamper_is_scored_on_every_row(report: BenchmarkReport) -> None:
    """verifier_tamper reads assistant text only, which TRACE always provides."""
    metric = _metric(report, "verifier_tamper")
    assert metric.counts.units_scored == report.rows_mapped
    assert metric.counts.units_error == 0
    assert metric.coverage.value == pytest.approx(1.0)


def test_verifier_tamper_separates_the_labels(report: BenchmarkReport) -> None:
    """The hacked-side fire rate must beat the clean-side one.

    This is a separation check, not an accuracy bar. The absolute numbers belong
    in the benchmark report, and any release threshold is a documented product
    decision taken against the sealed holdout, not an assertion smuggled in here.
    """
    metric = _metric(report, "verifier_tamper")
    positive = metric.fire_rate_positive.value
    negative = metric.fire_rate_negative.value
    assert positive is not None and negative is not None
    assert positive > negative, (
        f"verifier_tamper did not separate the TRACE labels on the {SPLIT} split: "
        f"hacked fire rate {positive:.3f} vs clean fire rate {negative:.3f}"
    )


def test_every_scored_positive_and_negative_is_accounted_for(report: BenchmarkReport) -> None:
    """The confusion cells must add up to the scored population, with no gaps."""
    for metric in report.metrics:
        counts = metric.counts
        cells = (
            counts.true_positives
            + counts.false_positives
            + counts.true_negatives
            + counts.false_negatives
        )
        assert cells == counts.units_scored
        assert (
            counts.units_scored + counts.units_insufficient + counts.units_error
            == counts.units_total
        )


def test_script_and_test_produce_the_same_report(report: BenchmarkReport, tmp_path: Path) -> None:
    """The validation script and this test must agree, byte for byte.

    They diverged before because each carried its own mapping and its own
    arithmetic. Running the script as a subprocess and comparing its JSON against
    the report built in this process is the direct check that they no longer can.
    """
    out = tmp_path / "script.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "trace_validation.py"),
            "--split",
            SPLIT,
            "--json",
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert completed.returncode == 0, completed.stderr
    if not out.is_file():
        pytest.skip(f"the script skipped rather than reporting: {completed.stdout.strip()}")
    assert json.loads(out.read_text()) == json.loads(report_to_json(report))
