"""Synthetic performance-dataset generator: layout, counts, and determinism."""

from __future__ import annotations

import json
from pathlib import Path

from rolloutscope.adapters import resolve_adapter
from tests.script_loader import load_script


def _generate():
    return load_script("scripts/perf/generate.py")


def _tiny(generate):
    return generate.SizeSpec(
        name="tiny",
        steps=2,
        groups=3,
        rollouts_per_group=2,
        multi_turn_fraction=0.5,
        long_completion_fraction=0.25,
        turns=3,
        seed=7,
    )


def test_generate_writes_prime_rl_step_layout(tmp_path: Path) -> None:
    generate = _generate()
    stats = generate.generate_dataset(tmp_path, _tiny(generate))
    assert stats.record_count == 12
    assert stats.step_count == 2
    assert stats.group_cardinality == 3
    assert (tmp_path / "step_0" / "train_rollouts.jsonl").is_file()
    assert (tmp_path / "step_1" / "train_rollouts.jsonl").is_file()
    assert stats.input_bytes > 0
    assert stats.kind_mix["single_turn"] + stats.kind_mix["multi_turn"] == 12
    assert stats.kind_mix["multi_turn"] > 0
    assert stats.mean_trajectory_turns > 0
    meta = json.loads((tmp_path / "generator.json").read_text(encoding="utf-8"))
    assert meta == {"size": "tiny", "seed": 7}


def test_generated_dataset_loads_through_the_prime_rl_adapter(tmp_path: Path) -> None:
    generate = _generate()
    generate.generate_dataset(tmp_path, _tiny(generate))
    adapter = resolve_adapter(tmp_path)
    assert adapter.name == "prime_rl_train"
    rows = list(adapter.load(tmp_path))
    assert len(rows) == 12
    assert {row.step_index for row in rows} == {0, 1}
    assert len({row.group_id for row in rows}) == 3


def test_generate_is_byte_identical_for_the_same_seed(tmp_path: Path) -> None:
    generate = _generate()
    spec = _tiny(generate)
    first = tmp_path / "a"
    second = tmp_path / "b"
    generate.generate_dataset(first, spec)
    generate.generate_dataset(second, spec)
    for step in ("step_0", "step_1"):
        left = (first / step / "train_rollouts.jsonl").read_bytes()
        right = (second / step / "train_rollouts.jsonl").read_bytes()
        assert left == right


def test_named_sizes_are_documented() -> None:
    generate = _generate()
    assert set(generate.SIZES) == {"small", "medium", "large"}
    assert generate.SIZES["small"].record_count() == 160
    assert generate.SIZES["medium"].record_count() == 5_000
    assert generate.SIZES["large"].record_count() == 50_000
