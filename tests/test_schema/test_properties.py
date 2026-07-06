"""Hypothesis property tests: serialize/deserialize round-trip stability and ID
stability under reserialization, for both rollout variants."""

import orjson
from hypothesis import given
from hypothesis import strategies as st

from rolloutscope.schema import rollout_id, validate_rollout
from rolloutscope.schema.models import (
    Message,
    MultiTurnRollout,
    SingleTurnRollout,
    TrajectoryStep,
)

finite_floats = st.floats(allow_nan=False, allow_infinity=False)

messages = st.lists(
    st.builds(
        Message,
        role=st.sampled_from(["system", "user", "assistant", "tool"]),
        content=st.one_of(st.none(), st.text(max_size=120)),
    ),
    max_size=4,
)

prompt_like = st.one_of(st.none(), st.text(max_size=80), messages)

single_turn = st.builds(
    SingleTurnRollout,
    example_id=st.integers(min_value=0, max_value=10**9),
    reward=finite_floats,
    is_completed=st.booleans(),
    is_truncated=st.booleans(),
    prompt=prompt_like,
    completion=prompt_like,
    metrics=st.dictionaries(st.text(min_size=1, max_size=16), finite_floats, max_size=4),
    answer=st.one_of(st.none(), st.text(max_size=40)),
)

steps = st.lists(
    st.builds(
        TrajectoryStep,
        trajectory_id=st.text(min_size=1, max_size=12),
        prompt=prompt_like,
        completion=prompt_like,
        reward=st.one_of(st.none(), finite_floats),
        is_truncated=st.booleans(),
    ),
    max_size=3,
)

multi_turn = st.builds(
    MultiTurnRollout,
    example_id=st.integers(min_value=0, max_value=10**9),
    reward=finite_floats,
    is_completed=st.booleans(),
    is_truncated=st.booleans(),
    prompt=prompt_like,
    completion=prompt_like,
    trajectory=steps,
)

rollouts = st.one_of(single_turn, multi_turn)


@given(rollouts)
def test_roundtrip_stable(rollout) -> None:
    first_dump = rollout.model_dump(mode="json")
    wire = orjson.dumps(first_dump)
    revalidated = validate_rollout(orjson.loads(wire))
    second_dump = revalidated.model_dump(mode="json")
    assert second_dump == first_dump
    assert orjson.dumps(second_dump) == wire


@given(rollouts)
def test_rollout_id_stable_under_reserialization(rollout) -> None:
    dump = rollout.model_dump(mode="json")
    first = rollout_id(dump["example_id"], dump["prompt"], dump["completion"], dump["reward"])
    recycled = validate_rollout(orjson.loads(orjson.dumps(dump))).model_dump(mode="json")
    second = rollout_id(
        recycled["example_id"], recycled["prompt"], recycled["completion"], recycled["reward"]
    )
    assert first == second
