"""Verifiers eval adapter: golden rows, id attachment, unknown-key passthrough,
run_id derivation, and skip-and-log behavior."""

import logging
import shutil
from pathlib import Path

import orjson

from rolloutscope.adapters import VERIFIERS_EVAL, Adapter
from rolloutscope.schema import (
    MultiTurnRollout,
    SingleTurnRollout,
    group_id,
    rollout_id,
    run_id_from_manifest,
    run_id_from_name,
    validate_rollout,
)

# Literal ids computed once from the frozen Phase 2 fixtures; they must never
# drift, because rollout_id is content-derived from disk bytes.
EVAL_RUN_ID = "run-8eb1f2dd23ea"
EVAL_ROW0_ROLLOUT_ID = "r654f1b7843785977"
MULTI_TURN_RUN_ID = "run-e2887f0d1c8d"

GOOD_ROW = (
    '{"example_id": 0, "prompt": [{"role": "user", "content": "hi"}], '
    '"completion": [{"role": "assistant", "content": "ok"}], "reward": 1.0, '
    '"is_completed": true, "is_truncated": false, "metrics": {}}'
)


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


def test_satisfies_adapter_protocol() -> None:
    assert isinstance(VERIFIERS_EVAL, Adapter)


def test_golden_eval_run(eval_run_dir: Path) -> None:
    rollouts = list(VERIFIERS_EVAL.load(eval_run_dir))
    raw_rows = read_raw(eval_run_dir / "results.jsonl")
    metadata = orjson.loads((eval_run_dir / "metadata.json").read_bytes())

    assert len(rollouts) == 5
    assert all(isinstance(r, SingleTurnRollout) for r in rollouts)

    run_id = run_id_from_manifest(metadata)
    assert run_id == EVAL_RUN_ID
    for rollout, raw in zip(rollouts, raw_rows, strict=True):
        assert rollout == expected_rollout(raw, run_id, None)
        assert rollout.step_index is None

    assert rollouts[0].rollout_id == EVAL_ROW0_ROLLOUT_ID
    assert rollouts[0].run_id == EVAL_RUN_ID
    assert [r.group_id for r in rollouts] == ["grp-0", "grp-0", "grp-0", "grp-1", "grp-1"]
    assert [r.reward for r in rollouts] == [1.0, 0.2, 0.8, 0.2, 1.0]
    assert rollouts[0].metrics == {"correct_answer": 1.0, "format_reward": 1.0}
    assert rollouts[0].answer == "42"


def test_unknown_keys_preserved(eval_run_dir: Path) -> None:
    rollout = next(iter(VERIFIERS_EVAL.load(eval_run_dir)))
    dumped = rollout.model_dump()
    # row-level state_columns extras survive normalization
    assert dumped["dataset_split"] == "eval"
    assert dumped["judge_notes"] == "clean arithmetic"
    assert dumped["sampler_seed"] == 101
    # message-level provider extras survive too
    assert dumped["completion"][0]["refusal"] is None


def test_direct_results_file_matches_directory_load(eval_run_dir: Path) -> None:
    via_dir = list(VERIFIERS_EVAL.load(eval_run_dir))
    via_file = list(VERIFIERS_EVAL.load(eval_run_dir / "results.jsonl"))
    assert via_file == via_dir
    assert via_file[0].run_id == EVAL_RUN_ID


def test_load_run_manifest_carries_metadata(eval_run_dir: Path) -> None:
    manifest = VERIFIERS_EVAL.load_run(eval_run_dir)
    assert manifest.run_id == EVAL_RUN_ID
    assert [f.path.name for f in manifest.files] == ["results.jsonl"]
    assert [f.step_index for f in manifest.files] == [None]
    assert manifest.metadata["env_id"] == "synthetic-arith"
    assert manifest.metadata["rollouts_per_example"] == 3


def test_generic_jsonl_multi_turn(multi_turn_path: Path) -> None:
    rollouts = list(VERIFIERS_EVAL.load(multi_turn_path))
    assert len(rollouts) == 1
    rollout = rollouts[0]
    assert isinstance(rollout, MultiTurnRollout)
    assert rollout.run_id == run_id_from_name("multi_turn_rollout.jsonl") == MULTI_TURN_RUN_ID
    assert len(rollout.trajectory) == 2
    assert rollout.trajectory[0].trajectory_id == "traj-0001"
    assert rollout.stop_condition == "no_tools_called"
    assert rollout.tool_defs is not None
    assert rollout.model_dump()["sandbox_id"] == "sbx-42"
    raw = read_raw(multi_turn_path)[0]
    assert rollout == expected_rollout(raw, rollout.run_id, None)


def test_run_id_falls_back_to_directory_name(eval_run_dir: Path, tmp_path: Path) -> None:
    bare_run = tmp_path / "bare_run"
    bare_run.mkdir()
    shutil.copy(eval_run_dir / "results.jsonl", bare_run / "results.jsonl")
    manifest = VERIFIERS_EVAL.load_run(bare_run)
    assert manifest.run_id == run_id_from_name("bare_run")
    assert manifest.metadata == {}
    rollouts = list(VERIFIERS_EVAL.load(bare_run))
    assert all(r.run_id == run_id_from_name("bare_run") for r in rollouts)


def test_bad_rows_skipped_and_logged(tmp_path: Path, caplog) -> None:
    run_dir = tmp_path / "messy_run"
    run_dir.mkdir()
    lines = [
        GOOD_ROW,
        "this is not json {",
        "[1, 2, 3]",
        '{"example_id": 3}',
        '{"example_id": 4, "reward": "high", "is_completed": true, '
        '"is_truncated": false, "metrics": {}}',
        '{"example_id": 5, "reward": 1.0}',
        GOOD_ROW,
    ]
    (run_dir / "results.jsonl").write_text("\n".join(lines) + "\n")
    with caplog.at_level(logging.WARNING):
        rollouts = list(VERIFIERS_EVAL.load(run_dir))
    assert len(rollouts) == 2
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 5
    assert any("line 2" in m and "invalid JSON" in m for m in messages)
    assert any("line 3" in m and "not a JSON object" in m for m in messages)
    assert any("line 4" in m and "reward" in m for m in messages)
    assert any("line 5" in m for m in messages)
    assert any("line 6" in m for m in messages)


def test_detect(eval_run_dir: Path, multi_turn_path: Path, prime_rl_run_dir: Path) -> None:
    assert VERIFIERS_EVAL.detect(eval_run_dir)
    assert VERIFIERS_EVAL.detect(eval_run_dir / "results.jsonl")
    assert VERIFIERS_EVAL.detect(multi_turn_path)
    # prime-rl layouts belong to the other adapter
    assert not VERIFIERS_EVAL.detect(prime_rl_run_dir)
    assert not VERIFIERS_EVAL.detect(prime_rl_run_dir / "step_0" / "train_rollouts.jsonl")
    # non-rollout paths
    assert not VERIFIERS_EVAL.detect(eval_run_dir / "metadata.json")
    assert not VERIFIERS_EVAL.detect(eval_run_dir / "does_not_exist.jsonl")
