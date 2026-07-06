"""Streaming IO: generator-based, bad rows skipped and logged with line numbers,
never fatal; migration applied on read."""

import inspect
import logging
from pathlib import Path

from rolloutscope.schema import iter_jsonl, read_rollouts
from rolloutscope.schema.models import SingleTurnRollout

GOOD_ROW = (
    '{"example_id": 0, "prompt": [{"role": "user", "content": "hi"}], '
    '"completion": [{"role": "assistant", "content": "hello"}], "reward": 1.0, '
    '"is_completed": true, "is_truncated": false, "metrics": {}}'
)
V0_ROW = (
    '{"schema_version": "0.1", "episode_id": 9, "score": 0.25, '
    '"is_completed": true, "is_truncated": false}'
)


def write_lines(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n")
    return path


def test_readers_are_generators() -> None:
    assert inspect.isgeneratorfunction(iter_jsonl)
    assert inspect.isgeneratorfunction(read_rollouts)


def test_iter_jsonl_line_numbers_and_blank_lines(tmp_path: Path) -> None:
    path = write_lines(tmp_path / "rows.jsonl", ['{"a": 1}', "", '{"b": 2}'])
    assert list(iter_jsonl(path)) == [(1, {"a": 1}), (3, {"b": 2})]


def test_bad_rows_skipped_and_logged(tmp_path: Path, caplog) -> None:
    path = write_lines(
        tmp_path / "rows.jsonl",
        [
            GOOD_ROW,
            "this is not json {",
            "[1, 2, 3]",
            '{"example_id": "missing required fields"}',
            "",
            GOOD_ROW,
        ],
    )
    with caplog.at_level(logging.WARNING, logger="rolloutscope.schema.io"):
        rollouts = list(read_rollouts(path))
    assert len(rollouts) == 2
    assert all(isinstance(r, SingleTurnRollout) for r in rollouts)
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 3
    assert any("line 2" in m and "invalid JSON" in m for m in messages)
    assert any("line 3" in m and "not a JSON object" in m for m in messages)
    assert any("line 4" in m for m in messages)


def test_read_applies_migration_chain(tmp_path: Path) -> None:
    path = write_lines(tmp_path / "rows.jsonl", [V0_ROW, GOOD_ROW])
    rollouts = list(read_rollouts(path))
    assert len(rollouts) == 2
    migrated = rollouts[0]
    assert migrated.example_id == 9
    assert migrated.reward == 0.25
    assert migrated.schema_version == "1.0"


def test_unsupported_version_skipped_not_fatal(tmp_path: Path, caplog) -> None:
    future = '{"schema_version": "9.0", "example_id": 0, "reward": 1.0}'
    path = write_lines(tmp_path / "rows.jsonl", [future, GOOD_ROW])
    with caplog.at_level(logging.WARNING, logger="rolloutscope.schema.io"):
        rollouts = list(read_rollouts(path))
    assert len(rollouts) == 1
    assert any("line 1" in record.getMessage() for record in caplog.records)
