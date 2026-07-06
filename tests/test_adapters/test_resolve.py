"""resolve_adapter routing: layouts map to the right adapter, documented
tie-break order holds, and unrecognized paths raise."""

from pathlib import Path

import pytest

from rolloutscope.adapters import (
    ADAPTERS,
    PRIME_RL_TRAIN,
    VERIFIERS_EVAL,
    PrimeRlTrainAdapter,
    VerifiersEvalAdapter,
    resolve_adapter,
)

GOOD_ROW = (
    '{"example_id": 0, "prompt": [{"role": "user", "content": "hi"}], '
    '"completion": [{"role": "assistant", "content": "ok"}], "reward": 1.0, '
    '"is_completed": true, "is_truncated": false, "metrics": {}}'
)


def test_registry_order_is_the_documented_tie_break() -> None:
    assert isinstance(ADAPTERS[0], VerifiersEvalAdapter)
    assert isinstance(ADAPTERS[1], PrimeRlTrainAdapter)


def test_routes_verifiers_layouts(eval_run_dir: Path, multi_turn_path: Path) -> None:
    assert resolve_adapter(eval_run_dir) is VERIFIERS_EVAL
    assert resolve_adapter(eval_run_dir / "results.jsonl") is VERIFIERS_EVAL
    assert resolve_adapter(multi_turn_path) is VERIFIERS_EVAL


def test_routes_prime_rl_layouts(prime_rl_run_dir: Path) -> None:
    assert resolve_adapter(prime_rl_run_dir) is PRIME_RL_TRAIN
    assert resolve_adapter(prime_rl_run_dir / "step_0") is PRIME_RL_TRAIN
    assert resolve_adapter(prime_rl_run_dir / "step_1" / "train_rollouts.jsonl") is PRIME_RL_TRAIN


def test_tie_break_prefers_eval_artifact(tmp_path: Path) -> None:
    # a pathological directory carrying both artifact kinds resolves to
    # verifiers_eval, per the documented first-match order
    both = tmp_path / "both_run"
    both.mkdir()
    (both / "results.jsonl").write_text(GOOD_ROW + "\n")
    (both / "train_rollouts.jsonl").write_text(GOOD_ROW + "\n")
    assert resolve_adapter(both) is VERIFIERS_EVAL


def test_unrecognized_paths_raise(tmp_path: Path, eval_run_dir: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no adapter recognizes"):
        resolve_adapter(empty)
    with pytest.raises(ValueError, match="no adapter recognizes"):
        resolve_adapter(tmp_path / "does_not_exist")
    with pytest.raises(ValueError, match="no adapter recognizes"):
        resolve_adapter(eval_run_dir / "metadata.json")
