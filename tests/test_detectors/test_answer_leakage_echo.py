"""answer_leakage_echo: answer echoes and verbatim criterion copies."""

from __future__ import annotations

from rolloutscope.detectors import (
    AnswerLeakageEchoConfig,
    AnswerLeakageEchoDetector,
    DetectorConfig,
)
from rolloutscope.detectors._text import primary_completion

DETECTOR = AnswerLeakageEchoDetector()
DEFAULT = DetectorConfig()


def test_fires_on_hacked(load_labeled):
    rollouts = load_labeled("answer_leakage_echo_hacked")
    fired = [v for v in DETECTOR.detect(rollouts, DEFAULT) if v.fired]
    assert len(fired) == len(rollouts)
    for verdict in fired:
        assert verdict.category == "context_exploitation"
        assert verdict.evidence


def test_answer_echo_evidence_points_at_completion(load_labeled):
    rollouts = load_labeled("answer_leakage_echo_hacked")
    verdicts = DETECTOR.detect(rollouts, DEFAULT)
    echo_verdict = next(v for v in verdicts if "al-h-1" in v.rollout_ids)
    span = echo_verdict.evidence[0]
    rollout = next(r for r in rollouts if r.rollout_id == "al-h-1")
    _, text = primary_completion(rollout)
    assert span.text in text
    assert span.start is not None and span.end is not None
    assert text[span.start : span.end] == span.text
    assert rollout.answer is not None
    assert span.text.casefold() == rollout.answer.casefold()


def test_criterion_echo_fires_with_note(load_labeled):
    rollouts = load_labeled("answer_leakage_echo_hacked")
    verdicts = DETECTOR.detect(rollouts, DEFAULT)
    criterion_verdict = next(v for v in verdicts if "al-h-2" in v.rollout_ids)
    span = criterion_verdict.evidence[0]
    assert "grading_criteria" in (span.note or "")
    rollout = next(r for r in rollouts if r.rollout_id == "al-h-2")
    _, text = primary_completion(rollout)
    assert span.text in text


def test_silent_on_clean(load_labeled):
    rollouts = load_labeled("answer_leakage_echo_clean")
    assert [v for v in DETECTOR.detect(rollouts, DEFAULT) if v.fired] == []


def test_tightened_gates_silence_hacked(load_labeled):
    rollouts = load_labeled("answer_leakage_echo_hacked")
    tightened = DetectorConfig(
        answer_leakage_echo=AnswerLeakageEchoConfig(
            min_answer_chars=1_000, min_criterion_chars=10_000
        )
    )
    assert DETECTOR.detect(rollouts, tightened) == []


def test_loosened_extra_chars_fires_on_clean(load_labeled):
    """Raising max_extra_chars flips the clean answer-with-work row to fired."""
    rollouts = load_labeled("answer_leakage_echo_clean")
    loosened = DetectorConfig(answer_leakage_echo=AnswerLeakageEchoConfig(max_extra_chars=100_000))
    fired = [v for v in DETECTOR.detect(rollouts, loosened) if v.fired]
    assert any("al-c-1" in v.rollout_ids for v in fired)


def test_short_answer_gate(load_labeled):
    """Short answers never count as echoes even when the completion equals them."""
    rollouts = load_labeled("format_only_wins_hacked")  # answers are 3 chars
    assert DETECTOR.detect(rollouts, DEFAULT) == []
