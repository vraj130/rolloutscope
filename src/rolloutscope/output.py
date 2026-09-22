"""Collision checks and staged, atomic writes for local output artifacts."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path


def _destination_path(path: Path) -> Path:
    """Return an absolute destination without following its final path component."""
    return path.parent.resolve() / path.name


def _aliases(first: Path, second: Path) -> bool:
    if first.resolve() == second.resolve():
        return True
    return first.exists() and second.exists() and first.samefile(second)


def validate_output_paths(outputs: Iterable[Path], inputs: Iterable[Path] = ()) -> None:
    """Reject source aliases, output aliases, and directory targets before any write."""
    destinations = list(outputs)
    sources = list(inputs)
    for index, destination in enumerate(destinations):
        if destination.is_dir():
            raise ValueError(f"output is a directory: {destination}")
        for source in sources:
            if _aliases(destination, source):
                raise ValueError(
                    f"input/output collision: output {destination} aliases input {source}"
                )
        for previous in destinations[:index]:
            if _aliases(destination, previous):
                raise ValueError(
                    f"output/output collision: output {destination} aliases output {previous}"
                )


class OutputTransaction:
    """Stage every artifact before replacement; each final replacement is atomic.

    Serialization or iteration failure replaces no destinations. A filesystem
    failure during final replacement cannot offer cross-file atomicity; the
    inventory ledger should be committed last so it never certifies a partial set.
    """

    def __init__(self, inputs: Iterable[Path] = ()) -> None:
        self.inputs = list(inputs)
        self.pending: list[tuple[Path, Path]] = []

    def __enter__(self) -> OutputTransaction:
        return self

    def stage(self, path: Path, chunks: Iterable[bytes]) -> tuple[str, int]:
        """Stage chunks next to their destination and return SHA-256 and byte length."""
        destination = _destination_path(path)
        validate_output_paths([*(item[0] for item in self.pending), destination], self.inputs)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        temporary = Path(name)
        self.pending.append((destination, temporary))
        digest = hashlib.sha256()
        size = 0
        with os.fdopen(fd, "wb") as handle:
            for chunk in chunks:
                handle.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        return digest.hexdigest(), size

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        try:
            if exc_type is None:
                validate_output_paths([destination for destination, _ in self.pending], self.inputs)
                for destination, temporary in self.pending:
                    # Narrow the time between validation and replacement. Path based
                    # APIs cannot eliminate every local filesystem race, but resolving
                    # only the parent prevents an output symlink from redirecting the
                    # replacement to an unintended target.
                    validate_output_paths(
                        [item_destination for item_destination, _ in self.pending], self.inputs
                    )
                    os.replace(temporary, destination)
                    directory_fd = os.open(destination.parent, os.O_RDONLY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
        finally:
            for _, temporary in self.pending:
                temporary.unlink(missing_ok=True)


def atomic_write_chunks(path: Path, chunks: Iterable[bytes]) -> Path:
    """Write a byte iterator atomically, preserving the destination on iteration failure."""
    with OutputTransaction() as transaction:
        transaction.stage(path, chunks)
    return path


def atomic_write_bytes(path: Path, payload: bytes) -> Path:
    """Write one payload atomically with cleanup and replacement semantics shared by outputs."""
    return atomic_write_chunks(path, [payload])
