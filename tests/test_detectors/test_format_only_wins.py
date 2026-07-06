"""format_only_wins: metrics-breakdown exploitation, graceful degradation."""

from __future__ import annotations

from rolloutscope.detectors import DetectorConfig, FormatOnlyWinsConfig, FormatOnlyWinsDetector

DETECTOR = FormatOnlyWinsDetector()
DEFAULT = DetectorConfig()


def test_fires_on_hacked(load_labeled):
    rollouts = load_labeled("format_only_wins_hacked")
    fired = [v for v in DETECTOR.detect(rollouts, DEFAULT) if v.fired]
    assert len(fired) == len(rollouts)
    for verdict in fired:
        assert verdict.category == "rubric_judge_exploit"
        assert verdict.evidence


def test_evidence_names_metric_field_and_values(load_labeled):
    rollouts = load_labeled("format_only_wins_hacked")
    verdicts = [v for v in DETECTOR.detect(rollouts, DEFAULT) if v.fired]
    for verdict in verdicts:
        span = verdict.evidence[0]
        assert span.field.startswith("metrics.")
        metric_key = span.field.removeprefix("metrics.")
        rollout = next(r for r in rollouts if r.rollout_id in verdict.rollout_ids)
        assert metric_key in rollout.metrics
        assert f"{metric_key}={rollout.metrics[metric_key]:g}" in span.text
        assert f"reward={rollout.reward:g}" in span.text


def test_silent_on_clean(load_labeled):
    rollouts = load_labeled("format_only_wins_clean")
    assert [v for v in DETECTOR.detect(rollouts, DEFAULT) if v.fired] == []


def test_degrades_gracefully_without_matching_metrics(load_labeled):
    """No matching metric keys: never fires, one unfired verdict carries the reason."""
    rollouts = load_labeled("degenerate_repetition_hacked")  # metrics are empty there
    verdicts = DETECTOR.detect(rollouts, DEFAULT)
    assert len(verdicts) == 1
    verdict = verdicts[0]
    assert verdict.fired is False
    assert verdict.evidence == []
    assert "metric keys" in getattr(verdict, "reason", "")


def test_tightened_reward_threshold_silences_hacked(load_labeled):
    rollouts = load_labeled("format_only_wins_hacked")
    tightened = DetectorConfig(format_only_wins=FormatOnlyWinsConfig(min_reward=0.99))
    assert [v for v in DETECTOR.detect(rollouts, tightened) if v.fired] == []


def test_loosened_reward_threshold_fires_on_clean(load_labeled):
    """Clean has a format-only row at reward 0.05; dropping min_reward flips it."""
    rollouts = load_labeled("format_only_wins_clean")
    loosened = DetectorConfig(format_only_wins=FormatOnlyWinsConfig(min_reward=0.0))
    fired = [v for v in DETECTOR.detect(rollouts, loosened) if v.fired]
    assert len(fired) == 1
    assert fired[0].rollout_ids == ["fo-c-2"]
