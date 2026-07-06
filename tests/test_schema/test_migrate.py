"""Migration chain: a fake 0.x row migrates forward and re-validates; current rows
pass through the identity entry; unsupported versions raise."""

import pytest

from rolloutscope.schema import (
    SingleTurnRollout,
    UnsupportedSchemaVersionError,
    migrate_row,
    validate_rollout,
)

FAKE_V0_ROW = {
    "schema_version": "0.3",
    "episode_id": 7,
    "score": 0.5,
    "prompt": [{"role": "user", "content": "What is 2 + 2?"}],
    "completion": [{"role": "assistant", "content": "4"}],
    "is_completed": True,
    "is_truncated": False,
    "metrics": {"correct_answer": 1.0},
    "legacy_note": "kept through migration",
}


def test_v0_row_migrates_and_revalidates() -> None:
    migrated = migrate_row(dict(FAKE_V0_ROW))
    assert migrated["schema_version"] == "1.0"
    assert migrated["example_id"] == 7
    assert migrated["reward"] == 0.5
    assert "episode_id" not in migrated
    assert "score" not in migrated
    assert migrated["legacy_note"] == "kept through migration"
    rollout = validate_rollout(migrated)
    assert isinstance(rollout, SingleTurnRollout)
    assert rollout.example_id == 7
    assert rollout.reward == 0.5


def test_current_row_passes_identity_unchanged() -> None:
    row = {"schema_version": "1.0", "example_id": 1, "reward": 1.0}
    assert migrate_row(row) == row


def test_missing_version_treated_as_current() -> None:
    row = {"example_id": 1, "reward": 1.0}
    assert migrate_row(row) == row


def test_newer_major_raises() -> None:
    with pytest.raises(UnsupportedSchemaVersionError):
        migrate_row({"schema_version": "2.0", "example_id": 1, "reward": 1.0})


def test_unparseable_version_raises() -> None:
    with pytest.raises(UnsupportedSchemaVersionError):
        migrate_row({"schema_version": "not-a-version", "example_id": 1, "reward": 1.0})
