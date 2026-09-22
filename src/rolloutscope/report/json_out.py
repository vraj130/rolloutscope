"""Deterministic JSON serialization of ReportData for CI diffing.

Output bytes are a pure function of the ReportData contents: keys are sorted
recursively, separators and float formatting are orjson's fixed canonical
forms, indentation is a constant two spaces, and the document ends with a
single trailing newline. Two structurally equal reports serialize to
byte-identical output regardless of dict insertion order.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import orjson

from rolloutscope.report.model import ReportData
from rolloutscope.schema import Verdict

_OPTIONS = orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE


def render_json_bytes(report: ReportData) -> bytes:
    """Serialize a ReportData to deterministic JSON bytes.

    Input: the report model. Output: UTF-8 JSON bytes with recursively sorted
    keys, two-space indentation, and a trailing newline; byte-identical for
    structurally identical reports.
    """
    return orjson.dumps(report.model_dump(mode="json"), option=_OPTIONS)


def render_json(report: ReportData) -> str:
    """Serialize a ReportData to a deterministic JSON string.

    Same guarantees as ``render_json_bytes``, decoded as UTF-8.
    """
    return render_json_bytes(report).decode("utf-8")


def write_json(report: ReportData, path: Path) -> Path:
    """Write the deterministic JSON document to ``path``.

    Inputs: the report model and the output file path (parent directories are
    created if missing). Returns the path written.
    """
    from rolloutscope.output import atomic_write_bytes

    atomic_write_bytes(path, render_json_bytes(report))
    return path


def _verdict_chunks(verdicts: Iterable[Verdict]) -> Iterable[bytes]:
    """Serialize verdicts as sorted-key JSONL records without bulk buffering."""
    for verdict in verdicts:
        yield orjson.dumps(verdict.model_dump(mode="json"), option=orjson.OPT_SORT_KEYS) + b"\n"


def write_verdicts(report: ReportData, path: Path) -> Path:
    """Stream every report verdict as deterministic JSONL to ``path``.

    Inputs: the report retaining complete verdicts and an output path. The
    shared output layer commits the file atomically after all chunks serialize.
    """
    from rolloutscope.output import atomic_write_chunks

    atomic_write_chunks(path, _verdict_chunks(report.verdicts))
    return path


__all__ = ["render_json", "render_json_bytes", "write_json", "write_verdicts"]
