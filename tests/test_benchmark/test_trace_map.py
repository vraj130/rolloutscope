"""The mapping contract: what a detector is allowed to see, and where."""

from __future__ import annotations

import json

import pytest

from rolloutscope.benchmark import TraceRowError, map_row, provenance_of
from rolloutscope.schema import Message, MultiTurnRollout

from .conftest import assistant, trace_row, user


def _texts(messages: list[Message] | str | None) -> list[str]:
    """Flatten a message list to its content strings."""
    assert isinstance(messages, list)
    return [m.content for m in messages if isinstance(m.content, str)]


def test_completion_holds_assistant_text_only(simple_row: dict[str, object]) -> None:
    """Nothing the model merely observed may appear in the completion.

    This is the whole point of the mapping. TRACE staples simulated tool output
    onto the assistant message that requested it; a mapper that flattens the two
    makes observed text indistinguishable from generated text.
    """
    rollout = map_row(simple_row)
    joined = "\n".join(_texts(rollout.completion))
    assert "Looking for failing tests." in joined
    assert "assert True" not in joined
    assert "Make the billing tests pass." not in joined


def test_prompt_holds_the_opening_instruction(simple_row: dict[str, object]) -> None:
    """The task statement lands in prompt, not in completion."""
    rollout = map_row(simple_row)
    assert _texts(rollout.prompt) == ["Make the billing tests pass."]


def test_tool_results_become_observed_context_on_the_next_step(
    simple_row: dict[str, object],
) -> None:
    """A tool result is what the model saw before its next turn, and is typed as such."""
    rollout = map_row(simple_row)
    assert isinstance(rollout, MultiTurnRollout)
    assert len(rollout.trajectory) == 2

    first, second = rollout.trajectory
    assert first.prompt is None
    observed = second.prompt
    assert isinstance(observed, list)
    assert [m.role for m in observed] == ["tool"]
    assert "assert True" in str(observed[0].content)


def test_each_step_carries_one_assistant_turn_and_its_index(
    simple_row: dict[str, object],
) -> None:
    """Turn boundaries survive, so a finding can name the responsible turn."""
    rollout = map_row(simple_row)
    assert isinstance(rollout, MultiTurnRollout)
    for index, step in enumerate(rollout.trajectory):
        assert step.extras["turn_index"] == index
        assert step.trajectory_id.endswith(f":{index}")
        assert isinstance(step.completion, list)
        assert len(step.completion) == 1
        assert step.completion[0].role == "assistant"


def test_tool_calls_are_remapped_into_the_shape_detectors_read(
    simple_row: dict[str, object],
) -> None:
    """TRACE writes {name, parameters}; detectors read {function: {name, arguments}}."""
    rollout = map_row(simple_row)
    assert isinstance(rollout.completion, list)
    calls = rollout.completion[0].tool_calls
    assert calls is not None
    function = calls[0]["function"]
    assert function["name"] == "Grep"
    assert json.loads(function["arguments"])["path"] == "tests/"


def test_a_tool_result_is_paired_with_the_call_it_answers(
    simple_row: dict[str, object],
) -> None:
    """Positional pairing links an observation back to the request that caused it."""
    rollout = map_row(simple_row)
    assert isinstance(rollout, MultiTurnRollout)
    observed = rollout.trajectory[1].prompt
    assert isinstance(observed, list)
    assert observed[0].model_extra is not None
    assert observed[0].model_extra["tool_call_id"] == "call-0-0"


def test_extra_tool_results_are_kept_rather_than_dropped() -> None:
    """51 rows in the artifact have more results than calls; none may be lost."""
    row = trace_row(
        "trajectory_0007",
        "0",
        [
            user("Check the repo."),
            assistant("Running one command.", calls=[("Bash", {"cmd": "ls"})], results=["a", "b"]),
            assistant("Done."),
        ],
    )
    rollout = map_row(row)
    assert isinstance(rollout, MultiTurnRollout)
    observed = rollout.trajectory[1].prompt
    assert isinstance(observed, list)
    assert len(observed) == 2
    assert observed[1].model_extra is not None
    assert "tool_call_id" not in observed[1].model_extra


def test_absent_signals_are_recorded_not_invented(simple_row: dict[str, object]) -> None:
    """TRACE has no reward, metrics, or answer; the mapping must say so."""
    rollout = map_row(simple_row, revision="abc123")
    provenance = provenance_of(rollout)
    assert provenance["reward_available"] is False
    assert provenance["metrics_available"] is False
    assert provenance["answer_available"] is False
    assert provenance["label"] == "1.1.1, 1.3.2"
    assert provenance["revision"] == "abc123"
    assert provenance["num_turns"] == 2
    assert provenance["num_tool_calls"] == 1
    assert provenance["num_tool_results"] == 1


def test_identity_is_derived_from_the_row_not_its_position(
    simple_row: dict[str, object],
) -> None:
    """Reordering a fetch must not renumber the rows it returned."""
    first = map_row(simple_row, position=0)
    later = map_row(simple_row, position=99)
    assert first.rollout_id == later.rollout_id == "trace-trajectory_0042"
    assert first.example_id == later.example_id == 42
    assert first.group_id == later.group_id


def test_a_row_with_no_assistant_turn_is_unusable() -> None:
    """An empty rollout must not be scored as a clean one."""
    row = trace_row("trajectory_0009", "0", [user("hello"), user("anyone there?")])
    with pytest.raises(TraceRowError, match="no assistant turn"):
        map_row(row)


@pytest.mark.parametrize(
    ("row", "match"),
    [
        ({"trajectory_id": "t", "label": "0", "conversation": "{not json"}, "not valid JSON"),
        ({"trajectory_id": "t", "label": "0", "conversation": 5}, "not a string"),
        ({"trajectory_id": "t", "conversation": "[]"}, "label is missing"),
        ({"trajectory_id": "t", "label": "0", "conversation": "[]"}, "non-empty list"),
    ],
)
def test_malformed_rows_raise_with_a_reason(row: dict[str, object], match: str) -> None:
    """Every rejection names its reason, so unusable rows can be counted by cause."""
    with pytest.raises(TraceRowError, match=match):
        map_row(row)
