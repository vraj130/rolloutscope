"""reward_saturation_group_collapse: dead groups, trend mode, id fallbacks."""

from __future__ import annotations

from rolloutscope.detectors import (
    DetectorConfig,
    RewardSaturationGroupCollapseConfig,
    RewardSaturationGroupCollapseDetector,
)

DETECTOR = RewardSaturationGroupCollapseDetector()
DEFAULT = DetectorConfig()


def test_fires_group_verdicts_on_hacked(load_labeled):
    rollouts = load_labeled("reward_saturation_group_collapse_hacked")
    verdicts = DETECTOR.detect(rollouts, DEFAULT)
    group_verdicts = [v for v in verdicts if v.fired and getattr(v, "mode", None) == "group"]
    # 6 dead groups: step 0 has 1, step 1 has 2, step 2 has 3
    assert len(group_verdicts) == 6
    for verdict in group_verdicts:
        assert verdict.category == "reward_saturation"
        assert len(verdict.rollout_ids) == 3, "rollout_ids must list the whole group"
        assert verdict.evidence


def test_fires_trend_verdict_in_step_mode(load_labeled):
    rollouts = load_labeled("reward_saturation_group_collapse_hacked")
    verdicts = DETECTOR.detect(rollouts, DEFAULT)
    trend = [v for v in verdicts if v.fired and getattr(v, "mode", None) == "trend"]
    assert len(trend) == 1
    span = trend[0].evidence[0]
    assert span.field == "reward"
    assert span.text == "0.33 -> 0.67 -> 1.00"


def test_group_evidence_renders_group_rewards(load_labeled):
    rollouts = load_labeled("reward_saturation_group_collapse_hacked")
    verdicts = DETECTOR.detect(rollouts, DEFAULT)
    group_verdicts = [v for v in verdicts if v.fired and getattr(v, "mode", None) == "group"]
    for verdict in group_verdicts:
        span = verdict.evidence[0]
        assert span.field == "reward"
        assert span.text == "[1.0, 1.0, 1.0]", "span text renders the dead group's rewards"


def test_stable_id_fallback_used_when_rollout_id_missing(load_labeled):
    rollouts = load_labeled("reward_saturation_group_collapse_hacked")
    assert all(r.rollout_id is None for r in rollouts), "fixture exercises the fallback"
    verdicts = DETECTOR.detect(rollouts, DEFAULT)
    ids = {rid for v in verdicts for rid in v.rollout_ids}
    assert ids and all(rid.startswith("r") for rid in ids)
    # deterministic: same input yields the same ids
    again = {rid for v in DETECTOR.detect(rollouts, DEFAULT) for rid in v.rollout_ids}
    assert ids == again


def test_silent_on_clean(load_labeled):
    rollouts = load_labeled("reward_saturation_group_collapse_clean")
    assert [v for v in DETECTOR.detect(rollouts, DEFAULT) if v.fired] == []


def test_tightened_threshold_silences_hacked(load_labeled):
    rollouts = load_labeled("reward_saturation_group_collapse_hacked")
    tightened = DetectorConfig(
        reward_saturation_group_collapse=RewardSaturationGroupCollapseConfig(
            dead_fraction_threshold=1.5
        )
    )
    assert [v for v in DETECTOR.detect(rollouts, tightened) if v.fired] == []


def test_min_group_size_gate(load_labeled):
    rollouts = load_labeled("reward_saturation_group_collapse_hacked")
    gated = DetectorConfig(
        reward_saturation_group_collapse=RewardSaturationGroupCollapseConfig(min_group_size=4)
    )
    assert [v for v in DETECTOR.detect(rollouts, gated) if v.fired] == []
