"""Aggregate math against the Phase 2 fixtures, with hand-computed expectations.

The 5-row results.jsonl has rewards 1.0, 0.2, 0.8, 0.2, 1.0 over groups grp-0
(rows 1 to 3) and grp-1 (rows 4 and 5). Hand-computed values:
- mean 0.64; population variance 0.672 / 5 = 0.1344, std sqrt(0.1344)
- min 0.2, max 1.0; truncation rate 0.0; completion rate 1.0
- grp-0: count 3, mean 2/3, population variance 0.3466667 / 3 = 0.1155556
- grp-1: count 2, mean 0.6, population variance 0.32 / 2 = 0.16
- no step_index on any row, so the step series is empty
"""

import math
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from rolloutscope.analysis import AggregateConfig, Aggregates, aggregate_rollouts
from rolloutscope.schema import MultiTurnRollout, SingleTurnRollout, read_rollouts

RolloutFactory = Callable[..., SingleTurnRollout]


def _fixture_rollouts(eval_run_dir: Path) -> Iterator[SingleTurnRollout | MultiTurnRollout]:
    return read_rollouts(eval_run_dir / "results.jsonl")


@pytest.fixture
def fixture_aggregates(eval_run_dir: Path) -> Aggregates:
    return aggregate_rollouts(_fixture_rollouts(eval_run_dir))


def test_run_summary_hand_computed(fixture_aggregates: Aggregates) -> None:
    run = fixture_aggregates.run_summary
    assert run.row_count == 5
    assert run.reward_mean == pytest.approx(0.64)
    assert run.reward_std == pytest.approx(math.sqrt(0.1344))
    assert run.reward_min == 0.2
    assert run.reward_max == 1.0
    assert run.truncation_rate == 0.0
    assert run.completion_rate == 1.0


def test_group_stats_hand_computed(fixture_aggregates: Aggregates) -> None:
    groups = fixture_aggregates.group_stats
    assert sorted(groups) == ["grp-0", "grp-1"]
    grp0 = groups["grp-0"]
    assert grp0.count == 3
    assert grp0.reward_mean == pytest.approx(2.0 / 3.0)
    assert grp0.reward_variance == pytest.approx(0.3466666666666667 / 3.0)
    assert grp0.all_identical is False
    grp1 = groups["grp-1"]
    assert grp1.count == 2
    assert grp1.reward_mean == pytest.approx(0.6)
    assert grp1.reward_variance == pytest.approx(0.16)
    assert grp1.all_identical is False


def test_no_step_index_means_no_step_series(fixture_aggregates: Aggregates) -> None:
    assert fixture_aggregates.step_series == []


def test_reward_histogram_default_bins(fixture_aggregates: Aggregates) -> None:
    hist = fixture_aggregates.reward_histogram
    assert len(hist.bin_edges) == 11
    assert hist.bin_edges[0] == 0.0
    assert hist.bin_edges[-1] == 1.0
    # 0.2 twice in bin 2, 0.8 in bin 8, 1.0 twice clamped into the last bin.
    assert hist.counts == [0, 0, 2, 0, 0, 0, 0, 0, 1, 2]
    assert hist.underflow == 0
    assert hist.overflow == 0


def test_histogram_underflow_and_overflow(make_rollout: RolloutFactory) -> None:
    rollouts = [make_rollout(-0.5), make_rollout(1.5), make_rollout(0.55)]
    hist = aggregate_rollouts(iter(rollouts)).reward_histogram
    assert hist.underflow == 1
    assert hist.overflow == 1
    assert hist.counts[5] == 1
    assert sum(hist.counts) == 1


def test_top_and_bottom_k_ordering(eval_run_dir: Path) -> None:
    config = AggregateConfig(top_k=3)
    aggregates = aggregate_rollouts(_fixture_rollouts(eval_run_dir), config)

    top = aggregates.top_rollouts
    assert [snippet.reward for snippet in top] == [1.0, 1.0, 0.8]
    # Equal rewards keep first-seen order: row 1 (example 0) before row 5 (example 1).
    assert [snippet.example_id for snippet in top] == [0, 1, 0]
    assert "Final answer: 42" in top[0].completion_snippet

    bottom = aggregates.bottom_rollouts
    assert [snippet.reward for snippet in bottom] == [0.2, 0.2, 0.8]
    assert [snippet.example_id for snippet in bottom] == [0, 1, 0]
    assert "The answer is 41." in bottom[0].completion_snippet


def test_snippets_truncated_to_config_length(make_rollout: RolloutFactory) -> None:
    config = AggregateConfig(snippet_length=10)
    rollout = make_rollout(0.5, completion="x" * 50, prompt="y" * 9)
    aggregates = aggregate_rollouts(iter([rollout]), config)
    snippet = aggregates.top_rollouts[0]
    assert snippet.completion_snippet == "x" * 10 + "..."
    assert snippet.prompt_snippet == "y" * 9


def test_streaming_accepts_pure_generator(make_rollout: RolloutFactory) -> None:
    """A pure generator has no __len__, so any len() call would raise TypeError."""

    def generate() -> Iterator[SingleTurnRollout]:
        for index in range(50):
            yield make_rollout(float(index % 2), example_id=index)

    aggregates = aggregate_rollouts(generate())
    assert aggregates.run_summary.row_count == 50
    assert aggregates.run_summary.reward_mean == pytest.approx(0.5)
    assert len(aggregates.top_rollouts) == 5


def test_step_series_hand_computed(make_rollout: RolloutFactory) -> None:
    rollouts = [
        # step 0: group A is dead (identical rewards), group B is alive.
        make_rollout(0.5, 0, group_id="A", step_index=0, completion="abcd"),
        make_rollout(0.5, 0, group_id="A", step_index=0, completion="abcd"),
        make_rollout(0.2, 1, group_id="B", step_index=0, completion="ab"),
        make_rollout(0.8, 1, group_id="B", step_index=0, completion="abcdef"),
        # step 1: the only eligible group is dead.
        make_rollout(0.3, 0, group_id="A", step_index=1, completion="ab"),
        make_rollout(0.3, 0, group_id="A", step_index=1, completion="abcd"),
        # no step_index: contributes to the run but not to the series.
        make_rollout(0.9, 2, group_id="C", completion="zz"),
    ]
    aggregates = aggregate_rollouts(iter(rollouts))
    assert aggregates.run_summary.row_count == 7
    assert [step.step_index for step in aggregates.step_series] == [0, 1]

    step0 = aggregates.step_series[0]
    assert step0.count == 4
    assert step0.reward_mean == pytest.approx(0.5)
    # Rewards 0.5, 0.5, 0.2, 0.8: squared deviations sum 0.18, population variance 0.045.
    assert step0.reward_variance == pytest.approx(0.045)
    assert step0.dead_group_fraction == pytest.approx(0.5)
    assert step0.mean_completion_length == pytest.approx(4.0)

    step1 = aggregates.step_series[1]
    assert step1.count == 2
    assert step1.reward_mean == pytest.approx(0.3)
    assert step1.reward_variance == pytest.approx(0.0)
    assert step1.dead_group_fraction == pytest.approx(1.0)
    assert step1.mean_completion_length == pytest.approx(3.0)


def test_multi_turn_fixture_row(multi_turn_path: Path) -> None:
    aggregates = aggregate_rollouts(read_rollouts(multi_turn_path))
    assert aggregates.run_summary.row_count == 1
    assert aggregates.run_summary.reward_mean == pytest.approx(1.0)
    # Message content None with tool_calls contributes nothing; the tool output
    # and final assistant text still land in the snippet.
    snippet = aggregates.top_rollouts[0]
    assert "sum of squares from 1 to 5 is 55" in snippet.completion_snippet
    assert aggregates.group_stats["grp-0"].all_identical is True


def test_empty_iterator(make_rollout: RolloutFactory) -> None:
    aggregates = aggregate_rollouts(iter([]))
    run = aggregates.run_summary
    assert run.row_count == 0
    assert run.reward_mean is None
    assert run.reward_std is None
    assert run.truncation_rate is None
    assert aggregates.group_stats == {}
    assert aggregates.step_series == []
    assert aggregates.top_rollouts == []
    assert aggregates.bottom_rollouts == []
    assert sum(aggregates.reward_histogram.counts) == 0


def test_histogram_config_must_be_ordered() -> None:
    with pytest.raises(ValidationError):
        AggregateConfig(histogram_min=1.0, histogram_max=0.0)
