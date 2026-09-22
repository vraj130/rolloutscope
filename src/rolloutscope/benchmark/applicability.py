"""Which detectors may be scored on which units, and why not when they may not.

A detector that never fires because the data it reads was absent has not been
measured. Counting that silence as a true negative inflates every clean-side
number and hides the fact that the run proved nothing. So applicability is
decided before scoring and recorded per unit: a unit is scored, or it is
insufficient data with the missing signals named.

The requirement sets below are read off each detector's implementation, not
guessed. They describe what the detector needs in order to be able to reach a
verdict at all, not what it needs in order to fire.
"""

from __future__ import annotations

from enum import StrEnum
from typing import NamedTuple

from rolloutscope.schema import Rollout


class Requirement(StrEnum):
    """One signal a detector needs before it can be scored on a unit."""

    ASSISTANT_TEXT = "assistant_text"
    """Assistant-authored text or tool calls to scan."""
    REWARD = "reward"
    """A real scalar reward. A placeholder zero does not satisfy this."""
    METRICS = "metrics"
    """Named per-rollout metrics, for detectors that compare metric roles."""
    ANSWER_OR_CRITERIA = "answer_or_criteria"
    """A reference answer, or grading criteria visible in the prompt or info."""
    GROUPS = "groups"
    """Population level: enough sibling groups to compare within."""
    STEPS = "steps"
    """Population level: a step index, for trend variants."""


ROW_REQUIREMENTS: frozenset[Requirement] = frozenset(
    {
        Requirement.ASSISTANT_TEXT,
        Requirement.REWARD,
        Requirement.METRICS,
        Requirement.ANSWER_OR_CRITERIA,
    }
)
"""Requirements decidable from a single unit. The rest are population level."""


DETECTOR_REQUIREMENTS: dict[str, frozenset[Requirement]] = {
    "verifier_tamper": frozenset({Requirement.ASSISTANT_TEXT}),
    "reward_saturation_group_collapse": frozenset({Requirement.REWARD, Requirement.GROUPS}),
    "length_inflation": frozenset({Requirement.ASSISTANT_TEXT, Requirement.REWARD}),
    "format_only_wins": frozenset({Requirement.REWARD, Requirement.METRICS}),
    "degenerate_repetition": frozenset({Requirement.ASSISTANT_TEXT, Requirement.REWARD}),
    "answer_leakage_echo": frozenset({Requirement.ASSISTANT_TEXT, Requirement.ANSWER_OR_CRITERIA}),
}
"""Signals each built-in detector needs before a verdict from it means anything.

``verifier_tamper`` scans assistant text and tool-call arguments and needs
nothing else. ``length_inflation`` and ``degenerate_repetition`` both gate on
reward before they look at length or repetition, so without a real reward their
silence carries no information. ``format_only_wins`` compares a format metric
against a correctness metric and needs both to exist. ``answer_leakage_echo``
compares a completion against a reference answer or a stated grading criterion.
``reward_saturation_group_collapse`` needs rewards and sibling groups.

A detector absent from this map is treated as requiring nothing, so a
third-party detector is scored rather than silently excluded. Add an entry for
it to get honest coverage accounting.
"""


class Applicability(NamedTuple):
    """Whether a detector can be scored on a unit, and what is missing if not."""

    applicable: bool
    missing: tuple[Requirement, ...]

    @property
    def reason(self) -> str:
        """A short, stable reason code for a report, empty when applicable."""
        if self.applicable:
            return ""
        return "missing:" + ",".join(sorted(r.value for r in self.missing))


def available_signals(rollout: Rollout) -> frozenset[Requirement]:
    """Report which row-level signals a normalized rollout actually carries.

    Input: a normalized rollout. Output: the subset of :data:`ROW_REQUIREMENTS`
    the rollout satisfies.

    Availability comes from the benchmark provenance block when the mapper wrote
    one (``info["rolloutscope_benchmark"]``), because a mapper knows the
    difference between "reward is zero" and "the source had no reward". For a
    rollout from any other source, availability falls back to inspecting the row:
    a present ``metrics`` dict, a non-empty ``answer``, and, for reward, the
    assumption that a real adapter only sets a reward it read. That fallback is
    deliberately generous; it can only over-report availability, which shows up
    as a scored unit rather than as a hidden one.
    """
    provenance = rollout.info.get("rolloutscope_benchmark")
    declared = provenance if isinstance(provenance, dict) else {}

    signals: set[Requirement] = set()

    if rollout.completion:
        signals.add(Requirement.ASSISTANT_TEXT)

    if declared.get("reward_available", True):
        signals.add(Requirement.REWARD)
    if declared.get("metrics_available", bool(rollout.metrics)):
        signals.add(Requirement.METRICS)

    if declared.get("answer_available", bool(rollout.answer)):
        signals.add(Requirement.ANSWER_OR_CRITERIA)

    return frozenset(signals)


def applicability(
    detector: str,
    rollout: Rollout,
    *,
    population: frozenset[Requirement] = frozenset(),
) -> Applicability:
    """Decide whether ``detector`` may be scored on ``rollout``.

    Input: the detector name, the rollout, and the population-level signals the
    evaluated cohort satisfies (groups, steps), which a single row cannot decide
    on its own. Output: an :class:`Applicability`. A detector with no recorded
    requirements is applicable.
    """
    required = DETECTOR_REQUIREMENTS.get(detector, frozenset())
    have = available_signals(rollout) | population
    missing = tuple(sorted(required - have, key=lambda r: r.value))
    return Applicability(not missing, missing)
