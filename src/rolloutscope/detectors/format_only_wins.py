"""format_only_wins: format components near max, correctness near zero, reward high.

Reads the per-function metrics breakdown and flags rollouts where a
format/parser component is near its max, every correctness component is near
zero, and the scalar reward still clears a threshold: the signature of a
weight-imbalanced rubric being exploited by formatting alone. Format-ish and
correctness-ish keys are identified by configurable name substrings (heuristic
defaults: format/parse vs correct/accuracy/pass/success/solve/exact).
Category: rubric_judge_exploit.

When no rollout carries both a format-ish and a correctness-ish metric key,
the detector degrades gracefully: it returns a single unfired verdict whose
extra ``reason`` field tells callers the metrics context was absent, and it
never fires.

Known false-positive modes:
- Environments that intentionally weight formatting heavily (strict schema
  compliance tasks), where format-only reward is the designed behavior.
- Metrics not normalized to [0, 1], which break the near-max and near-zero
  defaults.
- Metric names that accidentally match both pattern lists.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from rolloutscope.detectors._text import clamp01, matching_metric_keys, stable_rollout_id
from rolloutscope.detectors.base import DetectorConfig
from rolloutscope.schema import EvidenceSpan, Rollout, Verdict


class FormatOnlyWinsDetector:
    """Per-rollout detector for format-dominated reward under a weight imbalance."""

    name: ClassVar[str] = "format_only_wins"
    category: ClassVar[str] = "rubric_judge_exploit"

    def detect(self, rollouts: Sequence[Rollout], config: DetectorConfig) -> list[Verdict]:
        """Return one fired verdict per format-only winner.

        Inputs: normalized rollouts and the full detector config (reads
        ``config.format_only_wins``). If no rollout exposes both metric
        families, returns a single unfired verdict carrying a ``reason`` extra
        field so callers can tell "clean" from "no metrics to look at".
        """
        cfg = config.format_only_wins
        verdicts: list[Verdict] = []
        saw_format_key = False
        saw_correctness_key = False
        for rollout in rollouts:
            format_keys = matching_metric_keys(rollout.metrics, cfg.format_metric_patterns)
            correctness_keys = matching_metric_keys(
                rollout.metrics, cfg.correctness_metric_patterns
            )
            saw_format_key = saw_format_key or bool(format_keys)
            saw_correctness_key = saw_correctness_key or bool(correctness_keys)
            if not format_keys or not correctness_keys:
                continue
            top_format_key = max(format_keys, key=lambda key: rollout.metrics[key])
            top_correct_key = max(correctness_keys, key=lambda key: rollout.metrics[key])
            format_value = rollout.metrics[top_format_key]
            # max over correctness keys: if even the largest is near zero, all are
            max_correct_value = rollout.metrics[top_correct_key]
            if (
                format_value >= cfg.format_near_max
                and max_correct_value <= cfg.correctness_near_zero
                and rollout.reward >= cfg.min_reward
            ):
                rid = stable_rollout_id(rollout)
                span = EvidenceSpan(
                    rollout_id=rid,
                    field=f"metrics.{top_format_key}",
                    text=(
                        f"{top_format_key}={format_value:g}, "
                        f"{top_correct_key}={max_correct_value:g}, "
                        f"reward={rollout.reward:g}"
                    ),
                    note=(
                        "format component near max with correctness near zero while the "
                        "scalar reward stays high (weight-imbalance exploitation)"
                    ),
                )
                verdicts.append(
                    Verdict(
                        detector=self.name,
                        fired=True,
                        score=clamp01(rollout.reward),
                        category=self.category,
                        evidence=[span],
                        rollout_ids=[rid],
                    )
                )
        if not verdicts and not (saw_format_key and saw_correctness_key):
            missing = []
            if not saw_format_key:
                missing.append("format")
            if not saw_correctness_key:
                missing.append("correctness")
            verdicts.append(
                # model_validate keeps the extra "reason" field (Verdict is
                # extra="allow") without passing unknown kwargs to __init__;
                # an unfired verdict has no evidence span to carry the reason.
                Verdict.model_validate(
                    {
                        "detector": self.name,
                        "fired": False,
                        "score": 0.0,
                        "category": self.category,
                        "evidence": [],
                        "rollout_ids": [],
                        "reason": (
                            f"no {' or '.join(missing)} metric keys matched the configured "
                            "name patterns; detector had nothing to inspect"
                        ),
                    }
                )
            )
        return verdicts
