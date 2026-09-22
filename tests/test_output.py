from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from rolloutscope.output import (
    OutputTransaction,
    atomic_write_bytes,
    atomic_write_chunks,
    validate_output_paths,
)


def _temporary_files(directory: Path) -> list[Path]:
    return list(directory.glob(".*.tmp"))


def test_rejects_direct_and_relative_source_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"source")

    with pytest.raises(ValueError, match="input/output collision"):
        validate_output_paths([source], [source])

    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="input/output collision"):
        validate_output_paths([Path("source.jsonl")], [source])


def test_rejects_symlink_and_hardlink_source_aliases(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"source")
    symlink = tmp_path / "symlink.jsonl"
    symlink.symlink_to(source)
    hardlink = tmp_path / "hardlink.jsonl"
    os.link(source, hardlink)

    for alias in (symlink, hardlink):
        with pytest.raises(ValueError, match="input/output collision"):
            validate_output_paths([alias], [source])


def test_rejects_direct_relative_symlink_and_hardlink_output_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "report.json"
    first.write_bytes(b"old")
    symlink = tmp_path / "report-link.json"
    symlink.symlink_to(first)
    hardlink = tmp_path / "report-hardlink.json"
    os.link(first, hardlink)

    with pytest.raises(ValueError, match="output/output collision"):
        validate_output_paths([first, first])

    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="output/output collision"):
        validate_output_paths([first, Path("report.json")])

    for alias in (symlink, hardlink):
        with pytest.raises(ValueError, match="output/output collision"):
            validate_output_paths([first, alias])


def test_rejects_directory_and_symlink_to_directory_targets(tmp_path: Path) -> None:
    directory = tmp_path / "artifacts"
    directory.mkdir()
    symlink = tmp_path / "artifacts-link"
    symlink.symlink_to(directory, target_is_directory=True)

    for target in (directory, symlink):
        with pytest.raises(ValueError, match="output is a directory"):
            validate_output_paths([target])


def test_iteration_failure_preserves_existing_output_and_removes_temporary_file(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "report.json"
    original = b"previous report"
    destination.write_bytes(original)

    def broken_chunks() -> Iterator[bytes]:
        yield b"partial"
        raise RuntimeError("serialization failed")

    with pytest.raises(RuntimeError, match="serialization failed"):
        atomic_write_chunks(destination, broken_chunks())

    assert destination.read_bytes() == original
    assert _temporary_files(tmp_path) == []


def test_late_staging_failure_preserves_every_existing_output(tmp_path: Path) -> None:
    first = tmp_path / "report.html"
    second = tmp_path / "report.json"
    first.write_bytes(b"old html")
    second.write_bytes(b"old json")

    def broken_chunks() -> Iterator[bytes]:
        yield b"partial json"
        raise ValueError("cannot encode verdict")

    with (
        pytest.raises(ValueError, match="cannot encode verdict"),
        OutputTransaction() as transaction,
    ):
        transaction.stage(first, [b"new html"])
        transaction.stage(second, broken_chunks())

    assert first.read_bytes() == b"old html"
    assert second.read_bytes() == b"old json"
    assert _temporary_files(tmp_path) == []


def test_invalid_chunk_preserves_output_and_removes_temporary_file(tmp_path: Path) -> None:
    destination = tmp_path / "report.json"
    destination.write_bytes(b"old")

    with pytest.raises(TypeError):
        atomic_write_chunks(destination, [b"valid", "invalid"])  # type: ignore[list-item]

    assert destination.read_bytes() == b"old"
    assert _temporary_files(tmp_path) == []


def test_success_replaces_outputs_and_returns_digest_and_size(tmp_path: Path) -> None:
    first = tmp_path / "report.html"
    second = tmp_path / "report.json"
    first.write_bytes(b"old html")
    second.write_bytes(b"old json")

    with OutputTransaction() as transaction:
        first_digest, first_size = transaction.stage(first, [b"new ", b"html"])
        second_digest, second_size = transaction.stage(second, [b"new json"])

    assert first.read_bytes() == b"new html"
    assert second.read_bytes() == b"new json"
    assert first_digest == hashlib.sha256(b"new html").hexdigest()
    assert second_digest == hashlib.sha256(b"new json").hexdigest()
    assert (first_size, second_size) == (8, 8)
    assert _temporary_files(tmp_path) == []


def test_successful_atomic_write_creates_parent_directories(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "artifacts" / "report.json"

    assert atomic_write_bytes(destination, b"payload") == destination

    assert destination.read_bytes() == b"payload"
    assert _temporary_files(destination.parent) == []


def test_success_uses_same_directory_atomic_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "report.json"
    destination.write_bytes(b"old")
    original_replace = os.replace
    calls: list[tuple[Path, Path]] = []

    def tracked_replace(source: str | Path, target: str | Path) -> None:
        source_path = Path(source)
        target_path = Path(target)
        assert source_path.parent == target_path.parent == tmp_path
        assert source_path.read_bytes() == b"new"
        assert target_path.read_bytes() == b"old"
        calls.append((source_path, target_path))
        original_replace(source_path, target_path)

    monkeypatch.setattr(os, "replace", tracked_replace)
    atomic_write_bytes(destination, b"new")

    assert len(calls) == 1
    assert calls[0][1] == destination
    assert destination.read_bytes() == b"new"


def test_replacement_failure_preserves_output_and_cleans_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "report.json"
    destination.write_bytes(b"old")

    def broken_replace(source: str | Path, target: str | Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", broken_replace)
    with pytest.raises(OSError, match="replace failed"):
        atomic_write_bytes(destination, b"new")

    assert destination.read_bytes() == b"old"
    assert _temporary_files(tmp_path) == []


def test_existing_output_symlink_is_replaced_without_modifying_its_target(tmp_path: Path) -> None:
    target = tmp_path / "unrelated.txt"
    target.write_bytes(b"keep me")
    destination = tmp_path / "report.json"
    destination.symlink_to(target)

    atomic_write_bytes(destination, b"new report")

    assert not destination.is_symlink()
    assert destination.read_bytes() == b"new report"
    assert target.read_bytes() == b"keep me"
    assert _temporary_files(tmp_path) == []


def test_commit_rechecks_source_collision_and_cleans_staged_file(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"source")
    destination = tmp_path / "report.json"

    with (
        pytest.raises(ValueError, match="input/output collision"),
        OutputTransaction([source]) as transaction,
    ):
        transaction.stage(destination, [b"new report"])
        destination.symlink_to(source)

    assert source.read_bytes() == b"source"
    assert destination.is_symlink()
    assert _temporary_files(tmp_path) == []


def test_commit_rechecks_output_aliases_and_preserves_prior_outputs(tmp_path: Path) -> None:
    first = tmp_path / "report.html"
    second = tmp_path / "report.json"
    first.write_bytes(b"old html")

    with (
        pytest.raises(ValueError, match="output/output collision"),
        OutputTransaction() as transaction,
    ):
        transaction.stage(first, [b"new html"])
        transaction.stage(second, [b"new json"])
        second.symlink_to(first)

    assert first.read_bytes() == b"old html"
    assert second.is_symlink()
    assert _temporary_files(tmp_path) == []
