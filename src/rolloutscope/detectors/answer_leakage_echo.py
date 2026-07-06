"""answer_leakage_echo: the completion echoes the answer or the reward criterion.

Two per-rollout signals, both normalized (casefolded, whitespace-collapsed):

1. Answer echo: the ground-truth ``answer`` appears in the completion with
   negligible surrounding text, meaning the rollout produced the answer with
   no visible work. Gated on a minimum answer length so trivial short answers
   ("42", "yes") never match.
2. Criterion echo: a reward criterion, read from ``info`` values under
   criteria-like keys or from prompt lines that look like grading criteria,
   is copied verbatim into the completion.

Category: context_exploitation.

Known false-positive modes:
- Short-answer tasks where echoing the answer IS the task (final-answer math,
  classification labels); the length gate reduces but does not remove this.
- Instruction-following tasks that explicitly require restating the rubric or
  the acceptance criteria before answering.
- Extractive QA, where the expected answer is a verbatim span by design.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import ClassVar

from rolloutscope.detectors._text import (
    flexible_find,
    messages_text,
    normalize_text,
    primary_completion,
    stable_rollout_id,
)
from rolloutscope.detectors.base import AnswerLeakageEchoConfig, DetectorConfig
from rolloutscope.schema import EvidenceSpan, Rollout, Verdict


def _span_for_echo(
    rollout: Rollout,
    field: str,
    text: str,
    echoed: str,
    note: str,
) -> EvidenceSpan:
    """Evidence span for an echoed string, with raw offsets when locatable."""
    rid = stable_rollout_id(rollout)
    located = flexible_find(text, echoed)
    if located is not None:
        start, end = located
        return EvidenceSpan(
            rollout_id=rid, field=field, start=start, end=end, text=text[start:end], note=note
        )
    return EvidenceSpan(rollout_id=rid, field=field, text=echoed, note=note)


class AnswerLeakageEchoDetector:
    """Per-rollout detector for answer echoes and verbatim criterion copies."""

    name: ClassVar[str] = "answer_leakage_echo"
    category: ClassVar[str] = "context_exploitation"

    def detect(self, rollouts: Sequence[Rollout], config: DetectorConfig) -> list[Verdict]:
        """Return one fired verdict per rollout that echoes answer or criteria.

        Inputs: normalized rollouts and the full detector config (reads
        ``config.answer_leakage_echo``). A rollout's answer-echo and
        criterion-echo spans are combined into a single verdict whose score is
        the max of the configured per-signal scores.
        """
        cfg = config.answer_leakage_echo
        criteria_line = re.compile(cfg.criteria_line_regex)
        verdicts: list[Verdict] = []
        for rollout in rollouts:
            field, text = primary_completion(rollout)
            if not text:
                continue
            normalized = normalize_text(text)
            spans: list[EvidenceSpan] = []
            score = 0.0
            answer_span = self._answer_echo(rollout, field, text, normalized, cfg)
            if answer_span is not None:
                spans.append(answer_span)
                score = max(score, cfg.answer_echo_score)
            for criterion, origin in self._criteria(rollout, criteria_line, cfg):
                if normalize_text(criterion) in normalized:
                    spans.append(
                        _span_for_echo(
                            rollout,
                            field,
                            text,
                            criterion,
                            f"verbatim echo of reward criterion from {origin}",
                        )
                    )
                    score = max(score, cfg.criterion_echo_score)
            if spans:
                verdicts.append(
                    Verdict(
                        detector=self.name,
                        fired=True,
                        score=score,
                        category=self.category,
                        evidence=spans,
                        rollout_ids=[stable_rollout_id(rollout)],
                    )
                )
        return verdicts

    def _answer_echo(
        self,
        rollout: Rollout,
        field: str,
        text: str,
        normalized: str,
        cfg: AnswerLeakageEchoConfig,
    ) -> EvidenceSpan | None:
        """Span for an answer echoed with negligible surrounding work, else None."""
        if not rollout.answer:
            return None
        normalized_answer = normalize_text(rollout.answer)
        if len(normalized_answer) < cfg.min_answer_chars:
            return None
        if normalized_answer not in normalized:
            return None
        extra_chars = len(normalized) - len(normalized_answer)
        if extra_chars > cfg.max_extra_chars:
            return None
        return _span_for_echo(
            rollout,
            field,
            text,
            rollout.answer,
            f"completion echoes the answer with only {extra_chars} other characters "
            "(no visible work)",
        )

    def _criteria(
        self,
        rollout: Rollout,
        criteria_line: re.Pattern[str],
        cfg: AnswerLeakageEchoConfig,
    ) -> list[tuple[str, str]]:
        """Collect (criterion text, origin) candidates from info and prompt lines."""
        candidates: list[tuple[str, str]] = []
        lowered_patterns = [pattern.lower() for pattern in cfg.criteria_key_patterns]
        for key, value in rollout.info.items():
            if not isinstance(value, str):
                continue
            if not any(pattern in key.lower() for pattern in lowered_patterns):
                continue
            if len(normalize_text(value)) >= cfg.min_criterion_chars:
                candidates.append((value, f"info[{key!r}]"))
        prompt_text = messages_text(rollout.prompt)
        for line in prompt_text.splitlines():
            stripped = line.strip()
            if not stripped or not criteria_line.search(stripped):
                continue
            if len(normalize_text(stripped)) >= cfg.min_criterion_chars:
                candidates.append((stripped, "prompt"))
        return candidates
