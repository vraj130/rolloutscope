"""Round-trip tests: real-shaped verifiers rows survive validate + dump losslessly,
including unknown extra keys at row, message, and nested-model level."""

from pathlib import Path
from typing import Any

import orjson

from rolloutscope.schema import (
    MultiTurnRollout,
    SingleTurnRollout,
    validate_rollout,
    write_rollouts,
)


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [orjson.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


def assert_subset(original: Any, dumped: Any, where: str = "$") -> None:
    """Every key and value present in the original must survive into the dump.

    The dump may add keys (kind, schema_version, defaults); it may never lose or
    change one.
    """
    if isinstance(original, dict):
        assert isinstance(dumped, dict), f"{where}: expected dict, got {type(dumped)}"
        for key, value in original.items():
            assert key in dumped, f"{where}.{key}: key lost in round-trip"
            assert_subset(value, dumped[key], f"{where}.{key}")
    elif isinstance(original, list):
        assert isinstance(dumped, list), f"{where}: expected list, got {type(dumped)}"
        assert len(original) == len(dumped), f"{where}: list length changed"
        for index, (a, b) in enumerate(zip(original, dumped, strict=True)):
            assert_subset(a, b, f"{where}[{index}]")
    else:
        assert original == dumped, f"{where}: {original!r} became {dumped!r}"


def test_single_turn_rows_roundtrip(eval_run_dir: Path) -> None:
    rows = load_rows(eval_run_dir / "results.jsonl")
    assert len(rows) == 5
    for row in rows:
        rollout = validate_rollout(row)
        assert isinstance(rollout, SingleTurnRollout)
        assert rollout.kind == "single_turn"
        assert rollout.schema_version == "1.0"
        assert_subset(row, rollout.model_dump(mode="json"))


def test_unknown_keys_survive(eval_run_dir: Path) -> None:
    rows = load_rows(eval_run_dir / "results.jsonl")
    first = validate_rollout(rows[0]).model_dump(mode="json")
    # row-level state_columns extras
    assert first["dataset_split"] == "eval"
    assert first["judge_notes"] == "clean arithmetic"
    assert first["sampler_seed"] == 101
    # message-level provider extras
    assert first["completion"][0]["refusal"] is None
    # nested-model extras (TokenUsage NotRequired keys)
    last = validate_rollout(rows[4]).model_dump(mode="json")
    assert last["token_usage"]["final_input_tokens"] == 30.0


def test_multi_turn_row_roundtrips(multi_turn_path: Path) -> None:
    (row,) = load_rows(multi_turn_path)
    rollout = validate_rollout(row)
    assert isinstance(rollout, MultiTurnRollout)
    assert rollout.kind == "multi_turn"
    assert len(rollout.trajectory) == 2
    assert rollout.trajectory[0].trajectory_id == "traj-0001"
    assert rollout.trajectory[0].completion[0].tool_calls is not None
    assert rollout.stop_condition == "no_tools_called"
    assert rollout.tool_defs is not None
    assert_subset(row, rollout.model_dump(mode="json"))
    # row-level extra survives on the multi-turn variant too
    assert rollout.model_dump(mode="json")["sandbox_id"] == "sbx-42"


def test_write_read_roundtrip_is_stable(
    eval_run_dir: Path, multi_turn_path: Path, tmp_path: Path
) -> None:
    from rolloutscope.schema import read_rollouts

    rows = load_rows(eval_run_dir / "results.jsonl") + load_rows(multi_turn_path)
    rollouts = [validate_rollout(row) for row in rows]
    out = tmp_path / "normalized.jsonl"
    assert write_rollouts(out, rollouts) == 6
    reread = list(read_rollouts(out))
    assert [r.model_dump(mode="json") for r in reread] == [
        r.model_dump(mode="json") for r in rollouts
    ]
