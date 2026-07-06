"""length_inflation: snapshot and trend correlation, flat-metric gate, thresholds."""

from __future__ import annotations

from rolloutscope.detectors import DetectorConfig, LengthInflationConfig, LengthInflationDetector
from rolloutscope.detectors._text import primary_completion

DETECTOR = LengthInflationDetector()
DEFAULT = DetectorConfig()


def test_fires_snapshot_and_trend_on_hacked(load_labeled):
    rollouts = load_labeled("length_inflation_hacked")
    verdicts = DETECTOR.detect(rollouts, DEFAULT)
    fired_modes = {getattr(v, "mode", None) for v in verdicts if v.fired}
    assert fired_modes == {"snapshot", "trend"}
    for verdict in verdicts:
        assert verdict.category == "rubric_judge_exploit"
        assert verdict.score > 0.5, "flat correctness metric should keep full score"
        assert len(verdict.rollout_ids) == len(rollouts)


def test_evidence_cites_longest_completion(load_labeled):
    rollouts = load_labeled("length_inflation_hacked")
    verdicts = [v for v in DETECTOR.detect(rollouts, DEFAULT) if v.fired]
    longest = max(rollouts, key=lambda r: len(primary_completion(r)[1]))
    for verdict in verdicts:
        span = verdict.evidence[0]
        assert span.rollout_id == longest.rollout_id
        assert span.text
        assert span.text in primary_completion(longest)[1]


def test_silent_on_clean(load_labeled):
    rollouts = load_labeled("length_inflation_clean")
    assert [v for v in DETECTOR.detect(rollouts, DEFAULT) if v.fired] == []


def test_tightened_correlation_silences_hacked(load_labeled):
    rollouts = load_labeled("length_inflation_hacked")
    tightened = DetectorConfig(length_inflation=LengthInflationConfig(min_correlation=1.01))
    assert [v for v in DETECTOR.detect(rollouts, tightened) if v.fired] == []


def test_min_samples_gate(load_labeled):
    rollouts = load_labeled("length_inflation_hacked")
    gated = DetectorConfig(length_inflation=LengthInflationConfig(min_samples=50, min_steps=50))
    assert DETECTOR.detect(rollouts, gated) == []


def test_varying_correctness_metric_blocks_fire(load_labeled):
    """Correctness tracking reward blocks the fire even under perfect correlation."""
    rollouts = load_labeled("length_inflation_hacked")
    for rollout in rollouts:
        rollout.metrics = {"correct_answer": rollout.reward}
    assert [v for v in DETECTOR.detect(rollouts, DEFAULT) if v.fired] == []


def test_missing_correctness_metric_reduces_score(load_labeled):
    rollouts = load_labeled("length_inflation_hacked")
    with_metric = [v for v in DETECTOR.detect(rollouts, DEFAULT) if v.fired]
    for rollout in rollouts:
        rollout.metrics = {}
    without_metric = [v for v in DETECTOR.detect(rollouts, DEFAULT) if v.fired]
    assert without_metric, "still fires without a corroborating metric"
    assert max(v.score for v in without_metric) < max(v.score for v in with_metric)
