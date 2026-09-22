"""Counting rules: an unmeasured detector must never look like a clean one."""

from __future__ import annotations

import pytest

from rolloutscope.benchmark import Outcome, Rate, aggregate, wilson_interval


def test_a_rate_with_no_denominator_is_unmeasured_not_zero() -> None:
    """0/0 must render as "not measured", never as 0.0."""
    rate = Rate.of(0, 0)
    assert rate.value is None
    assert rate.interval is None


def test_a_measured_rate_carries_its_interval_and_denominator() -> None:
    """Every reported rate must be checkable against its sample size."""
    rate = Rate.of(3, 10)
    assert rate.value == pytest.approx(0.3)
    assert rate.numerator == 3
    assert rate.denominator == 10
    assert rate.interval is not None
    assert rate.interval.low < 0.3 < rate.interval.high


def test_the_wilson_interval_stays_inside_the_unit_range() -> None:
    """At the boundaries the normal approximation leaves [0, 1]; Wilson does not."""
    assert wilson_interval(0, 0) is None
    low = wilson_interval(0, 8)
    high = wilson_interval(8, 8)
    assert low is not None and high is not None
    assert low.low == 0.0 and low.high < 1.0
    assert high.high == 1.0 and high.low > 0.0


def test_a_wider_sample_narrows_the_interval() -> None:
    """More evidence must mean less uncertainty at the same rate."""
    narrow = wilson_interval(50, 100)
    wide = wilson_interval(5, 10)
    assert narrow is not None and wide is not None
    assert (narrow.high - narrow.low) < (wide.high - wide.low)


def test_insufficient_units_are_excluded_from_every_rate() -> None:
    """The core rule. Silence for lack of data is not a true negative.

    Without this, a detector whose inputs were entirely absent reports a perfect
    false-positive rate and a full complement of true negatives, and the report
    reads as a clean bill of health for a run that measured nothing.
    """
    outcomes = [
        Outcome(unit_id=f"u{i}", scored=False, reason="missing:reward", positive=i < 5)
        for i in range(10)
    ]
    metrics = aggregate("length_inflation", outcomes)
    assert metrics.counts.units_total == 10
    assert metrics.counts.units_scored == 0
    assert metrics.counts.units_insufficient == 10
    assert metrics.counts.true_negatives == 0
    assert metrics.precision.value is None
    assert metrics.recall.value is None
    assert metrics.false_positive_rate.value is None
    assert metrics.coverage.value == pytest.approx(0.0)
    assert metrics.counts.insufficient_by_reason == {"missing:reward": 10}


def test_a_detector_that_raises_is_an_error_not_a_clean_result() -> None:
    """A crash must be visible; scoring it as clean would hide a broken plugin."""
    outcomes = [Outcome(unit_id=f"u{i}", scored=False, error="ValueError: boom") for i in range(3)]
    metrics = aggregate("verifier_tamper", outcomes)
    assert metrics.counts.units_error == 3
    assert metrics.counts.units_insufficient == 0
    assert metrics.counts.units_scored == 0
    assert metrics.counts.error_by_reason == {"ValueError: boom": 3}


def test_the_confusion_cells_partition_the_scored_units() -> None:
    """Scored units must be exactly tp + fp + tn + fn, with nothing unaccounted."""
    outcomes = [
        Outcome(unit_id="a", scored=True, positive=True, fired=True, codes=("1.1.1",)),
        Outcome(unit_id="b", scored=True, positive=True, fired=False, codes=("1.1.1", "1.3.2")),
        Outcome(unit_id="c", scored=True, positive=False, fired=True),
        Outcome(unit_id="d", scored=True, positive=False, fired=False),
        Outcome(unit_id="e", scored=False, reason="missing:reward"),
    ]
    metrics = aggregate("verifier_tamper", outcomes, category_name={"1.1.1": "Test Modification"})
    counts = metrics.counts
    assert (counts.true_positives, counts.false_positives) == (1, 1)
    assert (counts.true_negatives, counts.false_negatives) == (1, 1)
    assert counts.units_scored == 4
    assert counts.units_total == 5
    assert metrics.precision.value == pytest.approx(0.5)
    assert metrics.recall.value == pytest.approx(0.5)
    assert metrics.false_positive_rate.value == pytest.approx(0.5)
    assert metrics.coverage.value == pytest.approx(0.8)


def test_per_category_recall_counts_a_multi_label_row_under_every_code() -> None:
    """A row labelled 1.1.1 and 1.3.2 is a positive for both categories."""
    outcomes = [
        Outcome(unit_id="a", scored=True, positive=True, fired=True, codes=("1.1.1", "1.3.2")),
        Outcome(unit_id="b", scored=True, positive=True, fired=False, codes=("1.3.2",)),
    ]
    metrics = aggregate("verifier_tamper", outcomes)
    by_code = {c.code: c.recall for c in metrics.by_category}
    assert by_code["1.1.1"].value == pytest.approx(1.0)
    assert by_code["1.3.2"].value == pytest.approx(0.5)
    assert by_code["1.3.2"].denominator == 2
