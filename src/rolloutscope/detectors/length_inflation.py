"""length_inflation: reward correlates with output length, not task quality.

Computes the Pearson correlation between completion character length and
scalar reward across rollouts (snapshot mode) and between per-step mean length
and per-step mean reward (trend mode, when step_index is present). Fires only
when the correlation is high while independent correctness-ish metrics stay
flat; when no correctness metric exists, it fires at a reduced score with an
explanatory note. Category: rubric_judge_exploit.

Known false-positive modes:
- Tasks where longer answers are legitimately better (proof writing, essays,
  detailed explanations): length genuinely tracks quality there.
- Mixed pools where problem difficulty drives both answer length and reward.
- Datasets whose correctness metric keys do not match the configured name
  patterns, dropping the flat-metric corroboration and inviting reduced-score
  fires on legitimate runs.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from statistics import StatisticsError, fmean
from typing import ClassVar

from rolloutscope.detectors._text import (
    clamp01,
    distinct_steps,
    matching_metric_keys,
    primary_completion,
    stable_rollout_id,
)
from rolloutscope.detectors.base import DetectorConfig, LengthInflationConfig
from rolloutscope.schema import EvidenceSpan, Rollout, Verdict


def _safe_correlation(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Pearson correlation, or None on degenerate input (constant or too short)."""
    if len(xs) < 2 or len(ys) < 2:
        return None
    try:
        return statistics.correlation(list(xs), list(ys))
    except StatisticsError:
        return None


def _correctness_values(rollouts: Sequence[Rollout], patterns: Sequence[str]) -> list[float]:
    """All correctness-ish metric values across the rollouts, in rollout order."""
    values: list[float] = []
    for rollout in rollouts:
        for key in matching_metric_keys(rollout.metrics, patterns):
            values.append(rollout.metrics[key])
    return values


def _flat(values: Sequence[float], epsilon: float) -> bool:
    """True when the values' max-min range is within epsilon."""
    return (max(values) - min(values)) <= epsilon


class LengthInflationDetector:
    """Dataset-level detector for reward-length correlation with flat task metrics."""

    name: ClassVar[str] = "length_inflation"
    category: ClassVar[str] = "rubric_judge_exploit"

    def detect(self, rollouts: Sequence[Rollout], config: DetectorConfig) -> list[Verdict]:
        """Return at most one snapshot verdict and one trend verdict.

        Inputs: normalized rollouts and the full detector config (reads
        ``config.length_inflation``). Snapshot mode needs ``min_samples``
        rollouts; trend mode needs ``min_steps`` distinct step_index values.
        Verdicts carry ``mode`` ("snapshot" or "trend") as an extra field.
        """
        cfg = config.length_inflation
        verdicts: list[Verdict] = []
        samples = [(rollout, len(primary_completion(rollout)[1])) for rollout in rollouts]
        samples = [(rollout, length) for rollout, length in samples if length > 0]
        snapshot = self._snapshot_verdict(samples, cfg)
        if snapshot is not None:
            verdicts.append(snapshot)
        trend = self._trend_verdict(samples, cfg)
        if trend is not None:
            verdicts.append(trend)
        return verdicts

    def _snapshot_verdict(
        self,
        samples: list[tuple[Rollout, int]],
        cfg: LengthInflationConfig,
    ) -> Verdict | None:
        """Correlation across individual rollouts in one snapshot."""
        if len(samples) < cfg.min_samples:
            return None
        lengths = [float(length) for _, length in samples]
        rewards = [rollout.reward for rollout, _ in samples]
        correlation = _safe_correlation(lengths, rewards)
        if correlation is None or correlation < cfg.min_correlation:
            return None
        rollouts = [rollout for rollout, _ in samples]
        return self._build_verdict(rollouts, samples, correlation, "snapshot", cfg)

    def _trend_verdict(
        self,
        samples: list[tuple[Rollout, int]],
        cfg: LengthInflationConfig,
    ) -> Verdict | None:
        """Correlation between per-step mean length and per-step mean reward."""
        rollouts = [rollout for rollout, _ in samples]
        steps = distinct_steps(rollouts)
        if len(steps) < cfg.min_steps:
            return None
        mean_lengths: list[float] = []
        mean_rewards: list[float] = []
        for step in steps:
            step_samples = [(r, length) for r, length in samples if r.step_index == step]
            if not step_samples:
                return None
            mean_lengths.append(fmean(float(length) for _, length in step_samples))
            mean_rewards.append(fmean(r.reward for r, _ in step_samples))
        correlation = _safe_correlation(mean_lengths, mean_rewards)
        if correlation is None or correlation < cfg.min_correlation:
            return None
        return self._build_verdict(rollouts, samples, correlation, "trend", cfg)

    def _build_verdict(
        self,
        rollouts: Sequence[Rollout],
        samples: list[tuple[Rollout, int]],
        correlation: float,
        mode: str,
        cfg: LengthInflationConfig,
    ) -> Verdict | None:
        """Apply the flat-correctness gate and assemble the verdict."""
        correctness = _correctness_values(rollouts, cfg.correctness_metric_patterns)
        score = clamp01(correlation)
        metric_note = ""
        if correctness:
            if not _flat(correctness, cfg.correctness_flat_epsilon):
                return None
            metric_note = "; correctness metrics flat while reward tracks length"
        else:
            score *= cfg.missing_metric_score_factor
            metric_note = "; no correctness metric present to corroborate (reduced score)"
        exemplar, _ = max(samples, key=lambda pair: pair[1])
        field, text = primary_completion(exemplar)
        snippet = text[: cfg.evidence_snippet_chars]
        span = EvidenceSpan(
            rollout_id=stable_rollout_id(exemplar),
            field=field,
            start=0,
            end=len(snippet),
            text=snippet,
            note=(
                f"{mode} length-reward correlation {correlation:.2f} across "
                f"{len(samples)} rollouts{metric_note}; exemplar is the longest completion"
            ),
        )
        # model_validate keeps the extra "mode" marker (Verdict is
        # extra="allow") without passing unknown kwargs to __init__.
        return Verdict.model_validate(
            {
                "detector": self.name,
                "fired": True,
                "score": score,
                "category": self.category,
                "evidence": [span],
                "rollout_ids": [stable_rollout_id(r) for r in rollouts],
                "mode": mode,
            }
        )
