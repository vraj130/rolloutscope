"""degenerate_repetition: repetition ratios on high-reward rollouts."""

from __future__ import annotations

from rolloutscope.detectors import (
    DegenerateRepetitionConfig,
    DegenerateRepetitionDetector,
    DetectorConfig,
)
from rolloutscope.detectors._text import primary_completion

DETECTOR = DegenerateRepetitionDetector()
DEFAULT = DetectorConfig()


def test_fires_on_hacked(load_labeled):
    rollouts = load_labeled("degenerate_repetition_hacked")
    fired = [v for v in DETECTOR.detect(rollouts, DEFAULT) if v.fired]
    assert len(fired) == len(rollouts)
    for verdict in fired:
        assert verdict.category == "degeneracy"
        assert 0.5 <= verdict.score <= 1.0


def test_evidence_span_is_repeated_ngram_in_completion(load_labeled):
    rollouts = load_labeled("degenerate_repetition_hacked")
    verdicts = DETECTOR.detect(rollouts, DEFAULT)
    for verdict in verdicts:
        rollout = next(r for r in rollouts if r.rollout_id in verdict.rollout_ids)
        field, text = primary_completion(rollout)
        span = verdict.evidence[0]
        assert span.field == field
        assert span.text in text
        assert span.start is not None and span.end is not None
        assert text[span.start : span.end] == span.text
        assert "most repeated" in (span.note or "")


def test_silent_on_clean(load_labeled):
    rollouts = load_labeled("degenerate_repetition_clean")
    assert [v for v in DETECTOR.detect(rollouts, DEFAULT) if v.fired] == []


def test_reward_gate_silences_hacked(load_labeled):
    rollouts = load_labeled("degenerate_repetition_hacked")
    gated = DetectorConfig(degenerate_repetition=DegenerateRepetitionConfig(min_reward=1.5))
    assert DETECTOR.detect(rollouts, gated) == []


def test_loosened_distinct_ratio_fires_on_clean(load_labeled):
    """max_distinct_ratio=1.0 makes every inspected rollout fire: the flip."""
    rollouts = load_labeled("degenerate_repetition_clean")
    loosened = DetectorConfig(
        degenerate_repetition=DegenerateRepetitionConfig(max_distinct_ratio=1.0)
    )
    assert [v for v in DETECTOR.detect(rollouts, loosened) if v.fired]


def test_min_tokens_gate(load_labeled):
    rollouts = load_labeled("degenerate_repetition_hacked")
    gated = DetectorConfig(degenerate_repetition=DegenerateRepetitionConfig(min_tokens=10_000))
    assert DETECTOR.detect(rollouts, gated) == []
