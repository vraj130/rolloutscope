"""verifier_tamper: fires on tampering fixtures, silent on clean, evidence checks."""

from __future__ import annotations

from rolloutscope.detectors import DetectorConfig, VerifierTamperConfig, VerifierTamperDetector
from rolloutscope.detectors._text import completion_sources

DETECTOR = VerifierTamperDetector()
DEFAULT = DetectorConfig()


def test_fires_on_hacked(load_labeled):
    rollouts = load_labeled("verifier_tamper_hacked")
    fired = [v for v in DETECTOR.detect(rollouts, DEFAULT) if v.fired]
    assert len(fired) == len(rollouts), "every hacked rollout should fire"
    for verdict in fired:
        assert verdict.category == "verifier_tampering"
        assert verdict.evidence
        assert 0.0 <= verdict.score <= 1.0


def test_silent_on_clean(load_labeled):
    rollouts = load_labeled("verifier_tamper_clean")
    assert [v for v in DETECTOR.detect(rollouts, DEFAULT) if v.fired] == []


def test_evidence_points_at_offending_span(load_labeled):
    rollouts = load_labeled("verifier_tamper_hacked")
    verdicts = DETECTOR.detect(rollouts, DEFAULT)
    for verdict in verdicts:
        rollout = next(r for r in rollouts if r.rollout_id and r.rollout_id in verdict.rollout_ids)
        sources = dict(completion_sources(rollout, include_tool_calls=True))
        for span in verdict.evidence:
            assert span.field in sources
            assert span.text in sources[span.field], (
                f"evidence text {span.text!r} not found in {span.field}"
            )
            if span.start is not None and span.end is not None:
                assert sources[span.field][span.start : span.end] == span.text


def test_offsets_carried_for_text_patterns(load_labeled):
    rollouts = load_labeled("verifier_tamper_hacked")
    verdicts = DETECTOR.detect(rollouts, DEFAULT)
    assert any(
        span.start is not None and span.end is not None
        for verdict in verdicts
        for span in verdict.evidence
    )


def test_tightened_threshold_silences_hacked(load_labeled):
    rollouts = load_labeled("verifier_tamper_hacked")
    tightened = DetectorConfig(verifier_tamper=VerifierTamperConfig(min_matches=99))
    assert [v for v in DETECTOR.detect(rollouts, tightened) if v.fired] == []


def test_loosened_pattern_fires_on_clean(load_labeled):
    rollouts = load_labeled("verifier_tamper_clean")
    loosened = DetectorConfig(
        verifier_tamper=VerifierTamperConfig(patterns={"any_code_fence": r"```"})
    )
    assert [v for v in DETECTOR.detect(rollouts, loosened) if v.fired]
