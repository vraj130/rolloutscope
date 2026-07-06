"""Streaming aggregation over normalized rollouts.

Everything here is a single pass over an iterator of rollouts with bounded
memory (CLAUDE.md golden rule 6): running moments via Welford accumulators,
a fixed-edge reward histogram whose edges come from config up front (a
two-pass min/max scan is forbidden, so the default edges cover 0 to 1, the
native verifiers reward range, with explicit underflow and overflow counters
for rewards outside the configured range), bounded heaps for top-k and
bottom-k rollouts, and per-group / per-step accumulators whose size is
bounded by the number of distinct groups and steps, never by the number of
rollouts. The input iterator is consumed exactly once and its length is
never asked for.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Iterable
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, model_validator

from rolloutscope.schema import Message, Rollout
from rolloutscope.schema.ids import group_id as derive_group_id


class AggregateConfig(BaseModel):
    """Configuration for streaming aggregation.

    Inputs: ``histogram_bins`` fixed bins between ``histogram_min`` and
    ``histogram_max`` (defaults 10 bins over 0 to 1, the native verifiers
    reward range; rewards outside the range land in underflow / overflow
    counters, which keeps the histogram single-pass); ``top_k`` rollouts kept
    at each reward extreme; ``snippet_length`` caps prompt and completion
    snippet characters.
    """

    histogram_bins: int = Field(default=10, ge=1)
    histogram_min: float = 0.0
    histogram_max: float = 1.0
    top_k: int = Field(default=5, ge=0)
    snippet_length: int = Field(default=280, ge=8)

    @model_validator(mode="after")
    def _range_is_ordered(self) -> AggregateConfig:
        if not self.histogram_max > self.histogram_min:
            raise ValueError("histogram_max must be greater than histogram_min")
        return self


class RunSummary(BaseModel):
    """Whole-run scalar summary.

    Reward moments are ``None`` when the run is empty. ``reward_std`` is the
    population standard deviation (ddof 0). Rates are fractions in [0, 1]:
    ``truncation_rate`` is the fraction of rollouts with ``is_truncated`` and
    ``completion_rate`` the fraction with ``is_completed``.
    """

    row_count: int
    reward_mean: float | None = None
    reward_std: float | None = None
    reward_min: float | None = None
    reward_max: float | None = None
    truncation_rate: float | None = None
    completion_rate: float | None = None


class RewardHistogram(BaseModel):
    """Fixed-edge reward histogram computed in one pass.

    ``bin_edges`` has ``len(counts) + 1`` entries. Bin ``i`` covers
    ``[bin_edges[i], bin_edges[i + 1])``; the last bin also includes its upper
    edge so a reward exactly at the configured maximum is counted. Rewards
    below the first edge increment ``underflow``, above the last edge
    ``overflow``.
    """

    bin_edges: list[float]
    counts: list[int]
    underflow: int = 0
    overflow: int = 0


class GroupStats(BaseModel):
    """Per-group reward statistics (grouping key: ``group_id`` when present,
    else derived from ``example_id``).

    ``reward_variance`` is the population variance. ``all_identical`` is True
    when every reward in the group is exactly equal, the zero-advantage
    (dead group) signal used by saturation detectors.
    """

    count: int
    reward_mean: float
    reward_variance: float
    all_identical: bool


class StepStats(BaseModel):
    """Per-training-step statistics, present only for rollouts that carry a
    ``step_index``.

    ``dead_group_fraction`` is the fraction of groups at this step whose
    rewards are all identical, among groups with at least two rollouts
    (singleton groups carry no within-group signal and are excluded from both
    numerator and denominator; with no eligible groups the fraction is 0.0).
    ``mean_completion_length`` is the mean completion text length in
    characters.
    """

    step_index: int
    count: int
    reward_mean: float
    reward_variance: float
    dead_group_fraction: float
    mean_completion_length: float


class RolloutSnippet(BaseModel):
    """A reward-extreme rollout with truncated prompt and completion text,
    kept for the top-k / bottom-k report sections."""

    rollout_id: str | None = None
    example_id: int
    group_key: str
    reward: float
    prompt_snippet: str
    completion_snippet: str


class Aggregates(BaseModel):
    """Everything one streaming pass produces: run summary, reward histogram,
    per-group stats (sorted by group key), per-step series (sorted by step
    index, empty when no rollout carries ``step_index``), reward-extreme
    snippets, and the config that produced them."""

    run_summary: RunSummary
    reward_histogram: RewardHistogram
    group_stats: dict[str, GroupStats]
    step_series: list[StepStats]
    top_rollouts: list[RolloutSnippet]
    bottom_rollouts: list[RolloutSnippet]
    config: AggregateConfig


def content_text(value: list[Message] | str | None) -> str:
    """Flatten a prompt or completion field to plain text.

    Input: the field value, a plain string, a list of chat messages, or None.
    Message content strings are joined with newlines; structured content parts
    contribute their ``text`` values; None content (for example a pure
    tool-call message) contributes nothing.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    pieces: list[str] = []
    for message in value:
        content = message.content
        if isinstance(content, str):
            pieces.append(content)
        elif isinstance(content, list):
            for part in content:
                text = part.get("text")
                if isinstance(text, str):
                    pieces.append(text)
    return "\n".join(piece for piece in pieces if piece)


def _truncate(text: str, limit: int) -> str:
    """Cap text at ``limit`` characters, appending ``...`` when truncated."""
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


@dataclass
class _Welford:
    """Single-pass running moments plus min and max."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def add(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    @property
    def variance(self) -> float:
        """Population variance (ddof 0); 0.0 when empty."""
        return self.m2 / self.count if self.count else 0.0


@dataclass
class _StepAccumulator:
    """Per-step running state: reward moments, completion length sum, and
    per-group reward moments for the dead-group fraction."""

    reward: _Welford = field(default_factory=_Welford)
    completion_chars: int = 0
    groups: dict[str, _Welford] = field(default_factory=dict)


# Heap entries carry a sequence-number tiebreaker so equal rewards resolve to
# first-seen order and comparison never reaches the (unorderable) snippet.
_HeapEntry = tuple[float, int, RolloutSnippet]


def _push_bounded(heap: list[_HeapEntry], entry: _HeapEntry, k: int) -> None:
    """Push onto a size-k min-heap, evicting the smallest when full."""
    if len(heap) < k:
        heapq.heappush(heap, entry)
    else:
        heapq.heappushpop(heap, entry)


def aggregate_rollouts(
    rollouts: Iterable[Rollout],
    config: AggregateConfig | None = None,
) -> Aggregates:
    """Aggregate a stream of rollouts in one pass with bounded memory.

    Inputs: an iterable (typically a generator) of normalized rollouts,
    consumed exactly once and never measured with ``len``, plus an optional
    ``AggregateConfig`` (defaults documented on the model). Grouping key is
    ``group_id`` when set, else derived from ``example_id``. Step series
    entries exist only for rollouts whose ``step_index`` is not None. Returns
    an ``Aggregates`` model.
    """
    cfg = config if config is not None else AggregateConfig()
    bins = cfg.histogram_bins
    span = cfg.histogram_max - cfg.histogram_min
    bin_width = span / bins
    bin_edges = [cfg.histogram_min + i * bin_width for i in range(bins)]
    bin_edges.append(cfg.histogram_max)
    counts = [0] * bins
    underflow = 0
    overflow = 0

    overall = _Welford()
    truncated = 0
    completed = 0
    groups: dict[str, _Welford] = {}
    steps: dict[int, _StepAccumulator] = {}
    top_heap: list[_HeapEntry] = []
    bottom_heap: list[_HeapEntry] = []

    for sequence, rollout in enumerate(rollouts):
        reward = float(rollout.reward)
        overall.add(reward)
        if rollout.is_truncated:
            truncated += 1
        if rollout.is_completed:
            completed += 1

        if reward < cfg.histogram_min:
            underflow += 1
        elif reward > cfg.histogram_max:
            overflow += 1
        else:
            index = min(int((reward - cfg.histogram_min) / bin_width), bins - 1)
            counts[index] += 1

        group_key = (
            rollout.group_id
            if rollout.group_id is not None
            else derive_group_id(rollout.example_id)
        )
        groups.setdefault(group_key, _Welford()).add(reward)

        completion = content_text(rollout.completion)
        if rollout.step_index is not None:
            accumulator = steps.setdefault(rollout.step_index, _StepAccumulator())
            accumulator.reward.add(reward)
            accumulator.completion_chars += len(completion)
            accumulator.groups.setdefault(group_key, _Welford()).add(reward)

        if cfg.top_k > 0:
            snippet = RolloutSnippet(
                rollout_id=rollout.rollout_id,
                example_id=rollout.example_id,
                group_key=group_key,
                reward=reward,
                prompt_snippet=_truncate(content_text(rollout.prompt), cfg.snippet_length),
                completion_snippet=_truncate(completion, cfg.snippet_length),
            )
            _push_bounded(top_heap, (reward, -sequence, snippet), cfg.top_k)
            _push_bounded(bottom_heap, (-reward, -sequence, snippet), cfg.top_k)

    if overall.count:
        run_summary = RunSummary(
            row_count=overall.count,
            reward_mean=overall.mean,
            reward_std=math.sqrt(overall.variance),
            reward_min=overall.minimum,
            reward_max=overall.maximum,
            truncation_rate=truncated / overall.count,
            completion_rate=completed / overall.count,
        )
    else:
        run_summary = RunSummary(row_count=0)

    group_stats = {
        key: GroupStats(
            count=welford.count,
            reward_mean=welford.mean,
            reward_variance=welford.variance,
            all_identical=welford.minimum == welford.maximum,
        )
        for key, welford in sorted(groups.items())
    }

    step_series: list[StepStats] = []
    for step_index, accumulator in sorted(steps.items()):
        eligible = [w for w in accumulator.groups.values() if w.count >= 2]
        dead = sum(1 for w in eligible if w.minimum == w.maximum)
        step_series.append(
            StepStats(
                step_index=step_index,
                count=accumulator.reward.count,
                reward_mean=accumulator.reward.mean,
                reward_variance=accumulator.reward.variance,
                dead_group_fraction=dead / len(eligible) if eligible else 0.0,
                mean_completion_length=accumulator.completion_chars / accumulator.reward.count,
            )
        )

    def _extract(heap: list[_HeapEntry]) -> list[RolloutSnippet]:
        ordered = sorted(heap, key=lambda entry: (-entry[0], -entry[1]))
        return [entry[2] for entry in ordered]

    return Aggregates(
        run_summary=run_summary,
        reward_histogram=RewardHistogram(
            bin_edges=bin_edges, counts=counts, underflow=underflow, overflow=overflow
        ),
        group_stats=group_stats,
        step_series=step_series,
        top_rollouts=_extract(top_heap),
        bottom_rollouts=_extract(bottom_heap),
        config=cfg,
    )


__all__: list[str] = [
    "AggregateConfig",
    "Aggregates",
    "GroupStats",
    "RewardHistogram",
    "RolloutSnippet",
    "RunSummary",
    "StepStats",
    "aggregate_rollouts",
    "content_text",
]
