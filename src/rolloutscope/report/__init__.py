"""Report renderers: terminal, deterministic JSON, and self-contained HTML.

Every renderer is a pure function of ReportData (CLAUDE.md golden rule 7);
no detector logic lives here and no template logic lives anywhere else.
"""

from rolloutscope.report.html import render_html, write_html
from rolloutscope.report.json_out import render_json, render_json_bytes, write_json
from rolloutscope.report.model import (
    InputFile,
    ReportData,
    describe_input,
    findings_by_severity,
    hash_file,
    severity_rank,
)
from rolloutscope.report.svg import histogram, line_chart
from rolloutscope.report.terminal import render_terminal

__all__ = [
    "InputFile",
    "ReportData",
    "describe_input",
    "findings_by_severity",
    "hash_file",
    "histogram",
    "line_chart",
    "render_html",
    "render_json",
    "render_json_bytes",
    "render_terminal",
    "severity_rank",
    "write_html",
    "write_json",
]
