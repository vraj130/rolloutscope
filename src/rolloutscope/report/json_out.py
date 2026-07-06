"""Deterministic JSON serialization of ReportData for CI diffing.

Output bytes are a pure function of the ReportData contents: keys are sorted
recursively, separators and float formatting are orjson's fixed canonical
forms, indentation is a constant two spaces, and the document ends with a
single trailing newline. Two structurally equal reports serialize to
byte-identical output regardless of dict insertion order.
"""

from __future__ import annotations

from pathlib import Path

import orjson

from rolloutscope.report.model import ReportData

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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(render_json_bytes(report))
    return path


__all__ = ["render_json", "render_json_bytes", "write_json"]
