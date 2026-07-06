"""ReportData: the single input every renderer consumes.

Renderers (terminal, JSON, HTML) are pure functions of this model (CLAUDE.md
golden rule 7). Reproducibility fields (tool version, schema version, input
file hashes) are plain data the caller fills; ``hash_file`` and
``describe_input`` are streaming helpers for doing so.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, Field

from rolloutscope.analysis.aggregates import Aggregates
from rolloutscope.schema import Finding

_SEVERITY_RANKS: dict[str, int] = {"critical": 0, "warning": 1, "info": 2}


class InputFile(BaseModel):
    """One analyzed input file, recorded for the reproducibility footer:
    file name, streaming sha256 content hash, and size in bytes."""

    name: str
    sha256: str
    size_bytes: int | None = None


class ReportData(BaseModel):
    """Everything a report renderer needs, and nothing else.

    ``tool_version`` and ``schema_version`` plus ``input_files`` form the
    reproducibility record; ``aggregates`` carries the run summary, histogram,
    group stats, step series, and reward-extreme snippets; ``findings`` is the
    assembled detector output. All fields are caller-filled plain data.
    """

    tool_version: str
    schema_version: str
    input_files: list[InputFile] = Field(default_factory=list)
    aggregates: Aggregates
    findings: list[Finding] = Field(default_factory=list)


def severity_rank(severity: str) -> int:
    """Order severities for display: critical 0, warning 1, info 2.

    Unknown severity strings sort last (rank 3) instead of raising, so a
    forward-compatible row never breaks rendering.
    """
    return _SEVERITY_RANKS.get(severity, 3)


def findings_by_severity(findings: Sequence[Finding]) -> list[Finding]:
    """Return findings stably sorted by severity (critical, warning, info).

    Input: any sequence of findings. The sort is stable, so findings of equal
    severity keep their input order.
    """
    return sorted(findings, key=lambda finding: severity_rank(finding.severity))


def hash_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute the sha256 hex digest of a file by streaming fixed-size chunks.

    Inputs: the file path and an optional chunk size in bytes (default 1 MiB).
    Never loads the whole file, so it is safe on files larger than RAM.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def describe_input(path: Path) -> InputFile:
    """Build the reproducibility record for one input file.

    Input: the file path. Returns an InputFile with the file's base name,
    streaming sha256 hash, and size in bytes.
    """
    return InputFile(name=path.name, sha256=hash_file(path), size_bytes=path.stat().st_size)


__all__ = [
    "InputFile",
    "ReportData",
    "describe_input",
    "findings_by_severity",
    "hash_file",
    "severity_rank",
]
