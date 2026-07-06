"""Prime-rl train adapter: step_index attachment and ordering, run root
resolution, golden normalized rows, and the step-name mapping."""

import logging
from pathlib import Path

import orjson
import pytest

from rolloutscope.adapters import PRIME_RL_TRAIN, step_index_from_name
from rolloutscope.schema import (
    MultiTurnRollout,
    SingleTurnRollout,
    group_id,
    rollout_id,
    run_id_from_name,
    validate_rollout,
)

# Literal ids computed once from the frozen prime_rl_run fixture.
PRIME_RUN_ID = "run-68a05070af44"
STEP0_ROW0_ROLLOUT_ID = "r6735e9b32256ae3d"

GOOD_ROW = (
    '{"example_id": 0, "prompt": [{"role": "user", "content": "hi"}], '
    '"completion": [{"role": "assistant", "content": "ok"}], "reward": 1.0, '
    '"is_completed": true, "is_truncated": false, "metrics": {}}'
)

# Full golden dump of the first step_0 row: raw fixture bytes plus adapter
# identity fields plus schema defaults, written out by hand.
STEP0_ROW0_EXPECTED_DUMP = {
    "schema_version": "1.0",
    "kind": "single_turn",
    "example_id": 0,
    "prompt": [{"role": "user", "content": "Compute 6 * 7.", "tool_calls": None}],
    "completion": [
        {"role": "assistant", "content": "6 * 7 = 43\n\nFinal answer: 43", "tool_calls": None}
    ],
    "reward": 0.0,
    "metrics": {"correct_answer": 0.0},
    "is_completed": True,
    "is_truncated": False,
    "timing": {
        "start_time": 1751710000.0,
        "setup": {"start": 1751710000.0, "end": 1751710000.05, "duration": 0.05},
        "generation": {"start": 1751710000.05, "end": 1751710001.0, "duration": 0.95},
        "scoring": {"start": 1751710001.0, "end": 1751710001.1, "duration": 0.1},
        "model": {"spans": []},
        "env": {"spans": []},
        "total": 1.05,
        "overhead": 0.0,
    },
    "token_usage": {"input_tokens": 12.0, "output_tokens": 14.0},
    "answer": "42",
    "info": {"difficulty": "easy"},
    "error": None,
    "stop_condition": None,
    "tool_defs": None,
    "rollout_id": STEP0_ROW0_ROLLOUT_ID,
    "group_id": "grp-0",
    "run_id": PRIME_RUN_ID,
    "step_index": 0,
    "env_seed": 7,
}


def read_raw(path: Path) -> list[dict]:
    """Decode every non-blank line of a JSONL file, unvalidated."""
    return [orjson.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


def expected_rollout(raw: dict, run_id: str, step_index: int | None):
    """Build the expected normalized row independently of the adapter's loader."""
    return validate_rollout(
        {
            **raw,
            "rollout_id": rollout_id(
                raw["example_id"], raw.get("prompt"), raw.get("completion"), raw["reward"]
            ),
            "group_id": group_id(raw["example_id"]),
            "run_id": run_id,
            "step_index": step_index,
        }
    )


def write_step(run_dir: Path, dirname: str, lines: list[str]) -> Path:
    """Create run_dir/dirname/train_rollouts.jsonl with the given lines."""
    step_dir = run_dir / dirname
    step_dir.mkdir(parents=True)
    (step_dir / "train_rollouts.jsonl").write_text("\n".join(lines) + "\n")
    return step_dir


def test_golden_two_step_run(prime_rl_run_dir: Path) -> None:
    rollouts = list(PRIME_RL_TRAIN.load(prime_rl_run_dir))
    assert len(rollouts) == 5
    assert [r.step_index for r in rollouts] == [0, 0, 0, 1, 1]
    assert all(r.run_id == PRIME_RUN_ID for r in rollouts)
    assert run_id_from_name("prime_rl_run") == PRIME_RUN_ID

    raw_by_step = [
        (0, read_raw(prime_rl_run_dir / "step_0" / "train_rollouts.jsonl")),
        (1, read_raw(prime_rl_run_dir / "step_1" / "train_rollouts.jsonl")),
    ]
    expected = [
        expected_rollout(raw, PRIME_RUN_ID, step) for step, rows in raw_by_step for raw in rows
    ]
    assert rollouts == expected

    assert rollouts[0].rollout_id == STEP0_ROW0_ROLLOUT_ID
    assert [r.group_id for r in rollouts] == ["grp-0", "grp-0", "grp-1", "grp-0", "grp-1"]
    assert isinstance(rollouts[3], SingleTurnRollout)
    assert isinstance(rollouts[4], MultiTurnRollout)
    assert rollouts[4].trajectory[0].trajectory_id == "traj-s1-0001"


def test_golden_full_dump_of_first_row(prime_rl_run_dir: Path) -> None:
    first = next(iter(PRIME_RL_TRAIN.load(prime_rl_run_dir)))
    assert first.model_dump(mode="json") == STEP0_ROW0_EXPECTED_DUMP


def test_load_run_manifest_orders_steps(prime_rl_run_dir: Path) -> None:
    manifest = PRIME_RL_TRAIN.load_run(prime_rl_run_dir)
    assert manifest.run_id == PRIME_RUN_ID
    assert [f.step_index for f in manifest.files] == [0, 1]
    assert [f.path.parent.name for f in manifest.files] == ["step_0", "step_1"]
    assert all(f.path.name == "train_rollouts.jsonl" for f in manifest.files)
    assert manifest.metadata == {}


def test_single_step_dir_entry_shares_run_id(prime_rl_run_dir: Path) -> None:
    rollouts = list(PRIME_RL_TRAIN.load(prime_rl_run_dir / "step_1"))
    assert len(rollouts) == 2
    assert all(r.step_index == 1 for r in rollouts)
    # the run root is the step dir's parent, so run_id matches the full run
    assert all(r.run_id == PRIME_RUN_ID for r in rollouts)


def test_direct_file_entry_shares_run_id(prime_rl_run_dir: Path) -> None:
    rollouts = list(PRIME_RL_TRAIN.load(prime_rl_run_dir / "step_0" / "train_rollouts.jsonl"))
    assert len(rollouts) == 3
    assert all(r.step_index == 0 for r in rollouts)
    assert all(r.run_id == PRIME_RUN_ID for r in rollouts)


def test_step_ordering_is_numeric_not_lexicographic(tmp_path: Path) -> None:
    run_dir = tmp_path / "long_run"
    run_dir.mkdir()
    write_step(run_dir, "step_10", [GOOD_ROW])
    write_step(run_dir, "step_2", [GOOD_ROW])
    manifest = PRIME_RL_TRAIN.load_run(run_dir)
    assert [f.step_index for f in manifest.files] == [2, 10]
    assert [r.step_index for r in PRIME_RL_TRAIN.load(run_dir)] == [2, 10]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("step_0", 0),
        ("step_12", 12),
        ("STEP_2", 2),
        ("Step_3", 3),
        ("3", 3),
        ("007", 7),
        ("step-1", None),
        ("step_", None),
        ("final", None),
        ("checkpoint_a", None),
        ("", None),
        ("step_1x", None),
    ],
)
def test_step_index_from_name(name: str, expected: int | None) -> None:
    assert step_index_from_name(name) == expected


def test_unrecognized_dir_name_means_snapshot_mode(tmp_path: Path) -> None:
    run_dir = tmp_path / "snap_run"
    run_dir.mkdir()
    write_step(run_dir, "final", [GOOD_ROW])
    manifest = PRIME_RL_TRAIN.load_run(run_dir)
    assert [f.step_index for f in manifest.files] == [None]
    rollouts = list(PRIME_RL_TRAIN.load(run_dir))
    assert all(r.step_index is None for r in rollouts)
    # a non-step dir given directly is its own run root
    direct = PRIME_RL_TRAIN.load_run(run_dir / "final")
    assert direct.files[0].step_index is None
    assert direct.run_id == run_id_from_name("final")


def test_mixed_step_and_plain_dirs_order(tmp_path: Path) -> None:
    run_dir = tmp_path / "mixed_run"
    run_dir.mkdir()
    write_step(run_dir, "extra", [GOOD_ROW])
    write_step(run_dir, "step_1", [GOOD_ROW])
    write_step(run_dir, "step_0", [GOOD_ROW])
    manifest = PRIME_RL_TRAIN.load_run(run_dir)
    assert [(f.step_index, f.path.parent.name) for f in manifest.files] == [
        (0, "step_0"),
        (1, "step_1"),
        (None, "extra"),
    ]


def test_bad_rows_skipped_and_logged(tmp_path: Path, caplog) -> None:
    run_dir = tmp_path / "messy_train"
    run_dir.mkdir()
    write_step(run_dir, "step_0", [GOOD_ROW, "nope {", '{"example_id": 1}', GOOD_ROW])
    with caplog.at_level(logging.WARNING):
        rollouts = list(PRIME_RL_TRAIN.load(run_dir))
    assert len(rollouts) == 2
    messages = [record.getMessage() for record in caplog.records]
    assert any("line 2" in m and "invalid JSON" in m for m in messages)
    assert any("line 3" in m and "reward" in m for m in messages)


def test_load_run_raises_when_nothing_found(tmp_path: Path) -> None:
    empty = tmp_path / "empty_run"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        PRIME_RL_TRAIN.load_run(empty)
    with pytest.raises(FileNotFoundError):
        PRIME_RL_TRAIN.load_run(tmp_path / "missing")


def test_detect(prime_rl_run_dir: Path, eval_run_dir: Path, tmp_path: Path) -> None:
    assert PRIME_RL_TRAIN.detect(prime_rl_run_dir)
    assert PRIME_RL_TRAIN.detect(prime_rl_run_dir / "step_0")
    assert PRIME_RL_TRAIN.detect(prime_rl_run_dir / "step_0" / "train_rollouts.jsonl")
    assert not PRIME_RL_TRAIN.detect(eval_run_dir)
    assert not PRIME_RL_TRAIN.detect(eval_run_dir / "results.jsonl")
    assert not PRIME_RL_TRAIN.detect(tmp_path / "missing")
