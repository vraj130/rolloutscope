"""degenerate_repetition: n-gram repetition and distinct-token collapse on winners.

Inspects the completion text of high-reward rollouts only (a degenerate output
that also scored high is what indicates the reward is gamed rather than the
model merely rambling). Two signals, both configurable: the distinct-token
ratio (unique words / total words) dropping to an extreme low, and the n-gram
repetition ratio (1 - unique n-grams / total n-grams) rising to an extreme
high. Either extreme fires. Category: degeneracy.

Only a single model completion is inspected. A concatenated multi-turn
transcript (a multi_turn rollout, or a completion holding more than one chat
message) spans many turns and naturally has a low distinct-token ratio, which
would false-fire, so those inputs are skipped rather than scored.

Known false-positive modes:
- Tasks with inherently repetitive valid outputs: tables, long enumerations,
  code with heavy boilerplate, repeated units in measurements or logs.
- Outputs in agglutinative or non-whitespace-delimited languages, where the
  crude word tokenizer distorts both ratios.
- Legitimately short vocabularies (yes/no grids, numeric matrices).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from typing import ClassVar

from rolloutscope.detectors._text import clamp01, primary_completion, stable_rollout_id, word_tokens
from rolloutscope.detectors.base import DetectorConfig
from rolloutscope.schema import EvidenceSpan, MultiTurnRollout, Rollout, Verdict


def _is_concatenated_transcript(rollout: Rollout) -> bool:
    """True when the rollout output is a multi-turn transcript, not one completion.

    Repetition is only meaningful within a single model completion. A multi_turn
    rollout, or a completion that holds more than one chat message, is a
    concatenated transcript across many turns; scanning it conflates unrelated
    turns and reliably false-fires (long real transcripts have low distinct-token
    ratios), so this detector skips those inputs.
    """
    if isinstance(rollout, MultiTurnRollout):
        return True
    completion = rollout.completion
    return isinstance(completion, list) and len(completion) > 1


class DegenerateRepetitionDetector:
    """Per-rollout detector for mode-collapsed, repetitive high-reward outputs."""

    name: ClassVar[str] = "degenerate_repetition"
    category: ClassVar[str] = "degeneracy"

    def detect(self, rollouts: Sequence[Rollout], config: DetectorConfig) -> list[Verdict]:
        """Return one fired verdict per repetitive high-reward rollout.

        Inputs: normalized rollouts and the full detector config (reads
        ``config.degenerate_repetition``). Concatenated multi-turn transcripts
        (a multi_turn rollout, or a completion with more than one message) are
        skipped, since repetition is only measured within a single completion.
        Rollouts below ``min_reward`` or with fewer than ``min_tokens`` word
        tokens are skipped. The evidence span carries the most-repeated n-gram,
        located in the raw completion text when possible.
        """
        cfg = config.degenerate_repetition
        verdicts: list[Verdict] = []
        for rollout in rollouts:
            if _is_concatenated_transcript(rollout):
                continue
            if rollout.reward < cfg.min_reward:
                continue
            field, text = primary_completion(rollout)
            tokens = word_tokens(text)
            if len(tokens) < cfg.min_tokens:
                continue
            distinct_ratio = len(set(tokens)) / len(tokens)
            n = cfg.ngram_n
            ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
            if not ngrams:
                continue
            repetition = 1.0 - (len(set(ngrams)) / len(ngrams))
            fired_distinct = distinct_ratio <= cfg.max_distinct_ratio
            fired_ngram = repetition >= cfg.min_ngram_repetition
            if not (fired_distinct or fired_ngram):
                continue
            severities: list[float] = []
            if fired_distinct and cfg.max_distinct_ratio > 0:
                severities.append(
                    clamp01((cfg.max_distinct_ratio - distinct_ratio) / cfg.max_distinct_ratio)
                )
            if fired_ngram and cfg.min_ngram_repetition < 1.0:
                severities.append(
                    clamp01(
                        (repetition - cfg.min_ngram_repetition) / (1.0 - cfg.min_ngram_repetition)
                    )
                )
            score = clamp01(0.5 + 0.5 * max(severities, default=0.0))
            top_gram, top_count = Counter(ngrams).most_common(1)[0]
            span = self._ngram_span(
                rollout, field, text, top_gram, top_count, distinct_ratio, repetition, n
            )
            verdicts.append(
                Verdict(
                    detector=self.name,
                    fired=True,
                    score=score,
                    category=self.category,
                    evidence=[span],
                    rollout_ids=[stable_rollout_id(rollout)],
                )
            )
        return verdicts

    def _ngram_span(
        self,
        rollout: Rollout,
        field: str,
        text: str,
        gram: tuple[str, ...],
        count: int,
        distinct_ratio: float,
        repetition: float,
        n: int,
    ) -> EvidenceSpan:
        """Evidence span pointing at the most-repeated n-gram in the raw text."""
        note = (
            f"most repeated {n}-gram occurs {count} times; distinct-token ratio "
            f"{distinct_ratio:.2f}, {n}-gram repetition {repetition:.2f}"
        )
        pattern = r"\W+".join(re.escape(token) for token in gram)
        match = re.search(pattern, text, re.IGNORECASE)
        if match is not None:
            return EvidenceSpan(
                rollout_id=stable_rollout_id(rollout),
                field=field,
                start=match.start(),
                end=match.end(),
                text=match.group(0),
                note=note,
            )
        return EvidenceSpan(
            rollout_id=stable_rollout_id(rollout),
            field=field,
            text=" ".join(gram),
            note=note + " (offsets unavailable, tokenized form shown)",
        )
