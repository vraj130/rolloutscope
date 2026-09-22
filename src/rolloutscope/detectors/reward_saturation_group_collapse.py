"""reward_saturation_group_collapse: on-disk proxy for GRPO zero-advantage collapse.

Groups rollouts by group_id (falling back to the example_id grouping key) and
measures within-group reward variance. A group whose variance is zero is dead:
in group-relative methods every advantage in it is zero and it contributes no
gradient (the true in-memory signal, prime-rl ``is_trainable``, is v1; this is
the black-box proxy over logged rewards). Snapshot mode reports saturated dead
groups when the dead-group fraction crosses a threshold; step mode additionally
flags a rising dead-group fraction with saturating reward means while
independent correctness metrics stay flat or fall. Category: reward_saturation.

Known false-positive modes:
- Genuinely easy tasks saturate honestly: every rollout solves the task and
  earns the same max reward with no hacking involved.
- Coarsely quantized rewards (for example strict 0/1 graders) tie within a
  group far more often than continuous rewards.
- Very small runs: with few groups, one tied group dominates the fraction
  (mitigated by ``min_groups``).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from statistics import fmean, pvariance
from typing import ClassVar

from rolloutscope.detectors._text import (
    distinct_steps,
    matching_metric_keys,
    stable_group_id,
    stable_rollout_id,
)
from rolloutscope.detectors.base import DetectorConfig, RewardSaturationGroupCollapseConfig
from rolloutscope.schema import EvidenceSpan, Rollout, Verdict

_GroupKey = tuple[int | None, str]


def _group(rollouts: Sequence[Rollout]) -> dict[_GroupKey, list[Rollout]]:
    """Group rollouts by (step_index, group id), group id falling back to example_id."""
    groups: dict[_GroupKey, list[Rollout]] = defaultdict(list)
    for rollout in rollouts:
        groups[(rollout.step_index, stable_group_id(rollout))].append(rollout)
    return groups


def _dead_stats(
    groups: dict[_GroupKey, list[Rollout]],
    cfg: RewardSaturationGroupCollapseConfig,
) -> tuple[list[_GroupKey], list[_GroupKey]]:
    """Return (eligible group keys, dead group keys) under the config."""
    eligible: list[_GroupKey] = []
    dead: list[_GroupKey] = []
    for key, members in groups.items():
        if len(members) < cfg.min_group_size:
            continue
        eligible.append(key)
        if pvariance([r.reward for r in members]) <= cfg.variance_epsilon:
            dead.append(key)
    return eligible, dead


def _correctness_mean(rollouts: Sequence[Rollout], patterns: Sequence[str]) -> float | None:
    """Mean of all correctness-ish metric values across rollouts, None if absent."""
    values: list[float] = []
    for rollout in rollouts:
        for key in matching_metric_keys(rollout.metrics, patterns):
            values.append(rollout.metrics[key])
    return fmean(values) if values else None


@dataclass
class _TrendInputs:
    """Shared step prerequisites and exact values consumed by the trend detector."""

    steps: list[int]
    reason: str | None = None
    fractions: list[float] = field(default_factory=list)
    reward_means: list[float] = field(default_factory=list)
    correctness_means: list[float | None] = field(default_factory=list)
    per_step_dead: dict[int, list[_GroupKey]] = field(default_factory=dict)
    per_step_groups: dict[int, dict[_GroupKey, list[Rollout]]] = field(default_factory=dict)


def _trend_inputs(
    rollouts: Sequence[Rollout], cfg: RewardSaturationGroupCollapseConfig
) -> _TrendInputs:
    """Compute trend inputs once per caller, preserving actual step eligibility guards."""
    inputs = _TrendInputs(steps=distinct_steps(rollouts))
    if not inputs.steps:
        inputs.reason = "missing_step_index"
        return inputs
    if len(inputs.steps) < cfg.min_steps:
        inputs.reason = "too_few_steps"
        return inputs
    for step in inputs.steps:
        step_rollouts = [r for r in rollouts if r.step_index == step]
        groups = _group(step_rollouts)
        eligible, dead = _dead_stats(groups, cfg)
        if not eligible:
            inputs.reason = "step_without_eligible_groups"
            return inputs
        inputs.fractions.append(len(dead) / len(eligible))
        inputs.per_step_dead[step] = dead
        inputs.per_step_groups[step] = groups
        inputs.reward_means.append(fmean(r.reward for r in step_rollouts))
        inputs.correctness_means.append(
            _correctness_mean(step_rollouts, cfg.correctness_metric_patterns)
        )
    return inputs


class RewardSaturationGroupCollapseDetector:
    """Group-level detector for reward variance collapse and its trend over steps."""

    name: ClassVar[str] = "reward_saturation_group_collapse"
    version: ClassVar[str] = "1"
    category: ClassVar[str] = "reward_saturation"

    def detect(self, rollouts: Sequence[Rollout], config: DetectorConfig) -> list[Verdict]:
        """Return group verdicts for saturated dead groups, plus a trend verdict.

        Inputs: normalized rollouts and the full detector config (reads
        ``config.reward_saturation_group_collapse``). Snapshot mode fires one
        verdict per saturated dead group once the overall dead-group fraction
        reaches ``dead_fraction_threshold``; when step_index is present on at
        least ``min_steps`` distinct steps, a trend verdict fires on a rising
        dead-group fraction with flat-or-falling correctness metrics. Verdicts
        carry typed group/trend modes and group/run analysis units.
        """
        cfg = config.reward_saturation_group_collapse
        verdicts: list[Verdict] = []
        groups = _group(rollouts)
        eligible, dead = _dead_stats(groups, cfg)
        if len(eligible) >= cfg.min_groups:
            dead_fraction = len(dead) / len(eligible)
            if dead_fraction >= cfg.dead_fraction_threshold:
                verdicts.extend(self._group_verdicts(groups, dead, dead_fraction, cfg))
        verdicts.extend(self._trend_verdicts(rollouts, cfg))
        return verdicts

    def _group_verdicts(
        self,
        groups: dict[_GroupKey, list[Rollout]],
        dead: list[_GroupKey],
        dead_fraction: float,
        cfg: RewardSaturationGroupCollapseConfig,
    ) -> list[Verdict]:
        """One fired verdict per dead group whose shared reward is saturated."""
        verdicts: list[Verdict] = []
        for key in dead:
            members = groups[key]
            rewards = [r.reward for r in members]
            if fmean(rewards) < cfg.saturated_reward_min:
                continue
            step, gid = key
            step_note = f" at step {step}" if step is not None else ""
            span = EvidenceSpan(
                rollout_id=stable_rollout_id(members[0]),
                field="reward",
                text=str(rewards),
                note=(
                    f"group {gid}{step_note}: zero reward variance across {len(members)} "
                    f"rollouts (dead group, zero-advantage proxy); dataset dead-group "
                    f"fraction {dead_fraction:.2f}"
                ),
            )
            verdicts.append(
                Verdict.model_validate(
                    {
                        "detector": self.name,
                        "fired": True,
                        "score": min(1.0, dead_fraction),
                        "category": self.category,
                        "evidence": [span],
                        "rollout_ids": [stable_rollout_id(r) for r in members],
                        "mode": "group",
                        "unit": "group",
                        "run_id": members[0].run_id,
                        "measurements": {
                            "group_id": gid,
                            "step_index": step,
                            "group_size": len(members),
                            "reward_mean": fmean(rewards),
                            "reward_variance": pvariance(rewards),
                            "dead_group_fraction": dead_fraction,
                        },
                    }
                )
            )
        return verdicts

    def _trend_verdicts(
        self,
        rollouts: Sequence[Rollout],
        cfg: RewardSaturationGroupCollapseConfig,
    ) -> list[Verdict]:
        """Step-mode trend verdict: rising dead fraction plus reward saturation."""
        inputs = _trend_inputs(rollouts, cfg)
        if inputs.reason is not None:
            return []
        steps = inputs.steps
        fractions = inputs.fractions
        reward_means = inputs.reward_means
        correctness_means = inputs.correctness_means
        rise = fractions[-1] - fractions[0]
        if rise < cfg.min_dead_fraction_rise:
            return []
        if fractions[-1] < cfg.dead_fraction_threshold:
            return []
        if reward_means[-1] < cfg.saturated_reward_min:
            return []
        first_correct, last_correct = correctness_means[0], correctness_means[-1]
        if (
            first_correct is not None
            and last_correct is not None
            and last_correct - first_correct > cfg.metric_flat_epsilon
        ):
            return []
        final_step = steps[-1]
        final_dead = inputs.per_step_dead[final_step]
        if not final_dead:
            return []
        dead_members = [r for key in final_dead for r in inputs.per_step_groups[final_step][key]]
        rendering = " -> ".join(f"{fraction:.2f}" for fraction in fractions)
        run_ids = {rollout.run_id for rollout in rollouts}
        span = EvidenceSpan(
            rollout_id=stable_rollout_id(dead_members[0]),
            field="reward",
            text=rendering,
            note=(
                f"dead-group fraction by step {steps[0]}..{final_step}: {rendering}; "
                f"final mean reward {reward_means[-1]:.2f} (rising zero-advantage proxy)"
            ),
        )
        return [
            Verdict.model_validate(
                {
                    "detector": self.name,
                    "fired": True,
                    "score": min(1.0, fractions[-1]),
                    "category": self.category,
                    "evidence": [span],
                    "rollout_ids": [stable_rollout_id(r) for r in dead_members],
                    "mode": "trend",
                    "unit": "run",
                    "run_id": next(iter(run_ids)) if len(run_ids) == 1 else None,
                    "measurements": {
                        "steps": steps,
                        "dead_group_fractions": fractions,
                        "dead_fraction_rise": rise,
                        "reward_means": reward_means,
                        "correctness_means": correctness_means,
                        "correctness_corroborated": first_correct is not None
                        and last_correct is not None,
                    },
                }
            )
        ]
