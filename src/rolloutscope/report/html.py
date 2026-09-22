"""Self-contained HTML report rendering.

One output file, pure function of ReportData (golden rule 7): inline CSS in a
single style block, two server-side SVG charts, details elements for
collapsibles, zero JavaScript, no CDN, no fetch, no external fonts; the file
opens from file://. The Jinja2 template ships inside the package (loaded via
PackageLoader so it works installed) with autoescape on; only the trusted SVG
strings generated here bypass escaping. The template formats ReportData and
nothing else; every derived value (stat tiles, chart inputs, config rows) is
computed here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, PackageLoader

from rolloutscope.report.model import ReportData, findings_by_severity
from rolloutscope.report.svg import histogram, line_chart
from rolloutscope.schema import Finding
from rolloutscope.schema.execution import UnitCounts

_FINDING_CAP = 50
_GROUP_CAP = 50
_STEP_CAP = 100
_SNIPPET_CAP = 20
_DIAGNOSTIC_CAP = 5
_TEXT_CAP = 400
_EVIDENCE_TEXT_CAP = 1_200
_INPUT_CAP = 100
_DETECTOR_ROW_CAP = 200
_CONFIG_ROW_CAP = 200
_ARTIFACT_CAP = 50


def _environment() -> Environment:
    """Build the Jinja2 environment with the packaged template and autoescape on."""
    return Environment(
        loader=PackageLoader("rolloutscope.report", "templates"),
        autoescape=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _fmt(value: float | None, digits: int = 4) -> str:
    """Format an optional float compactly; None renders as ``n/a``."""
    if value is None:
        return "n/a"
    return f"{value:.{digits}g}"


def _stat_tiles(report: ReportData) -> list[tuple[str, str]]:
    """Build (label, value) pairs for the run summary tiles."""
    run = report.aggregates.run_summary
    return [
        ("rollouts", str(run.row_count)),
        ("reward mean", _fmt(run.reward_mean)),
        ("reward std", _fmt(run.reward_std)),
        ("reward min", _fmt(run.reward_min)),
        ("reward max", _fmt(run.reward_max)),
        ("truncation rate", _fmt(run.truncation_rate)),
        ("completion rate", _fmt(run.completion_rate)),
        ("groups", str(len(report.aggregates.group_stats))),
    ]


def _histogram_svg(report: ReportData) -> str | None:
    """Render the reward histogram SVG, or None for an empty run."""
    if report.aggregates.run_summary.row_count == 0:
        return None
    hist = report.aggregates.reward_histogram
    bars = [float(count) for count in hist.counts]
    labels = [
        f"{low:g}-{high:g}"
        for low, high in zip(hist.bin_edges[:-1], hist.bin_edges[1:], strict=True)
    ]
    if hist.underflow:
        bars.insert(0, float(hist.underflow))
        labels.insert(0, f"under {hist.bin_edges[0]:g}")
    if hist.overflow:
        bars.append(float(hist.overflow))
        labels.append(f"over {hist.bin_edges[-1]:g}")
    return histogram(
        bars,
        labels,
        title="Reward distribution",
        x_label="reward bin",
        y_label="rollouts",
    )


def _step_svg(report: ReportData) -> str | None:
    """Render the per-step line chart SVG, or None without a step series."""
    series = report.aggregates.step_series
    if not series:
        return None
    run_ids = {getattr(item, "run_id", None) for item in series}
    if len(run_ids) > 1:
        return None
    series = series[:_STEP_CAP]
    return line_chart(
        {
            "reward mean": [(float(s.step_index), s.reward_mean) for s in series],
            "dead group fraction": [(float(s.step_index), s.dead_group_fraction) for s in series],
        },
        title="Reward mean and dead group fraction by step",
        x_label="training step",
        y_label="value",
    )


def _config_value(value: Any) -> str:
    """Render one config value for the appendix."""
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _config_rows(report: ReportData) -> list[tuple[str, str, str]]:
    """Build (owner, setting, value) rows for the config appendix.

    Rows cover the aggregation config plus every Finding's ``config_used``
    (the thresholds the detector actually ran with); duplicate rows from
    findings that share a detector and config are emitted once.
    """
    rows: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for key, value in sorted(report.aggregates.config.model_dump().items()):
        rows.extend(_flatten_config("aggregation", key, value))
    for finding in findings_by_severity(report.findings):
        for key in sorted(finding.config_used):
            for row in _flatten_config(finding.detector, key, finding.config_used[key]):
                if row not in seen:
                    seen.add(row)
                    rows.append(row)
    if report.execution is not None:
        for key, value in sorted(report.execution.effective_config.items()):
            rows.extend(_flatten_config("execution", key, value))
    return rows


def _flatten_config(owner: str, key: str, value: Any) -> list[tuple[str, str, str]]:
    """Flatten nested configuration into bounded, independently readable rows."""
    if isinstance(value, dict):
        rows: list[tuple[str, str, str]] = []
        for child_key, child_value in sorted(value.items(), key=lambda item: str(item[0])):
            rows.extend(_flatten_config(owner, f"{key}.{child_key}", child_value))
        return rows
    return [(owner, key, _bounded_text(_config_value(value)))]


def _bounded_text(value: object, cap: int = _TEXT_CAP) -> str:
    """Return display text capped to keep a portable report bounded in size."""
    text = str(value)
    return text if len(text) <= cap else text[:cap] + "..."


def _input_rows(report: ReportData) -> list[dict[str, object]]:
    """Build bounded input accounting rows from execution, falling back to hashes."""
    execution_files = report.execution.ingestion.files if report.execution is not None else []
    by_path = {item.path: item for item in execution_files}
    input_paths = {source.relative_path or source.name for source in report.input_files}
    rows: list[dict[str, object]] = []
    for source in report.input_files:
        ingestion = by_path.get(source.relative_path or source.name)
        rows.append(
            {
                "name": _bounded_text(source.relative_path or source.name),
                "format": _bounded_text(
                    source.format or (ingestion.format if ingestion else "unknown")
                ),
                "status": ingestion.status if ingestion else "not recorded",
                "accepted": ingestion.accepted if ingestion else "n/a",
                "rejected": ingestion.rejected if ingestion else "n/a",
                "blank": ingestion.blank if ingestion else "n/a",
                "duplicates": ingestion.duplicates if ingestion else "n/a",
                "examples": [
                    {
                        "line": example.line,
                        "reason": _bounded_text(example.reason),
                        "message": _bounded_text(example.message),
                    }
                    for example in (ingestion.examples[:_DIAGNOSTIC_CAP] if ingestion else [])
                ],
            }
        )
    for ingestion in execution_files:
        if ingestion.path in input_paths:
            continue
        rows.append(
            {
                "name": _bounded_text(ingestion.path),
                "format": _bounded_text(ingestion.format),
                "status": ingestion.status,
                "accepted": ingestion.accepted,
                "rejected": ingestion.rejected,
                "blank": ingestion.blank,
                "duplicates": ingestion.duplicates,
                "examples": [
                    {
                        "line": example.line,
                        "reason": _bounded_text(example.reason),
                        "message": _bounded_text(example.message),
                    }
                    for example in ingestion.examples[:_DIAGNOSTIC_CAP]
                ],
            }
        )
    return rows


def _detector_rows(report: ReportData) -> list[dict[str, object]]:
    """Summarize each detector unit and mode without joining unlike denominators."""
    rows: list[dict[str, object]] = []
    for result in sorted(report.detector_results, key=lambda item: item.detector):
        units: list[UnitCounts | None] = [*result.units] or [None]
        for unit in units:
            rows.append(
                {
                    "detector": _bounded_text(result.detector),
                    "status": result.status,
                    "unit": unit.unit if unit is not None else "n/a",
                    "mode": unit.mode if unit is not None else "n/a",
                    "run": (
                        _bounded_text(unit.measurements.get("run_id", "n/a"))
                        if unit is not None
                        else "n/a"
                    ),
                    "candidate": unit.candidate if unit is not None else "n/a",
                    "eligible": unit.eligible if unit is not None else "n/a",
                    "fired": unit.fired if unit is not None else "n/a",
                    "clean": unit.clean if unit is not None else "n/a",
                    "insufficient": unit.insufficient_data if unit is not None else "n/a",
                    "skipped": unit.skipped if unit is not None else "n/a",
                    "errors": unit.errors if unit is not None else "n/a",
                    "detail": ", ".join(
                        _bounded_text(error, 160) for error in result.errors[:_DIAGNOSTIC_CAP]
                    ),
                }
            )
    return rows


def _artifact_rows(report: ReportData) -> list[tuple[str, str, str]]:
    """Build display-only artifact references without external links or fetches."""
    if report.execution is None:
        return []
    return [
        (
            _bounded_text(item.name),
            _bounded_text(item.role),
            _bounded_text(item.sha256 or "pending"),
        )
        for item in report.execution.artifacts
    ]


def _finding_views(report: ReportData) -> list[Finding]:
    """Return bounded display copies while complete findings remain in JSON."""
    views: list[Finding] = []
    for finding in findings_by_severity(report.findings)[:_FINDING_CAP]:
        exemplars = [
            span.model_copy(
                update={
                    "text": _bounded_text(span.text, _EVIDENCE_TEXT_CAP),
                    "note": _bounded_text(span.note) if span.note is not None else None,
                }
            )
            for span in finding.exemplars[:_DIAGNOSTIC_CAP]
        ]
        views.append(
            finding.model_copy(
                update={
                    "title": _bounded_text(finding.title),
                    "description": _bounded_text(finding.description, _EVIDENCE_TEXT_CAP),
                    "exemplars": exemplars,
                }
            )
        )
    return views


def render_html(report: ReportData) -> str:
    """Render the one-file HTML report from a ReportData.

    Input: the report model. Output: a complete, self-contained HTML document
    string with findings ordered by severity, evidence spans highlighted via
    mark elements, both SVG charts when their data is present, a config
    appendix, and the reproducibility footer.
    """
    template = _environment().get_template("report.html.j2")
    input_rows = _input_rows(report)
    detector_rows = _detector_rows(report)
    config_rows = _config_rows(report)
    artifact_rows = _artifact_rows(report)
    step_run_ids = {getattr(item, "run_id", None) for item in report.aggregates.step_series}
    return template.render(
        report=report,
        stats=_stat_tiles(report),
        findings=_finding_views(report),
        finding_total=len(report.findings),
        histogram_svg=_histogram_svg(report),
        step_svg=_step_svg(report),
        step_chart_omitted=len(step_run_ids) > 1,
        config_rows=config_rows[:_CONFIG_ROW_CAP],
        config_total=len(config_rows),
        input_rows=input_rows[:_INPUT_CAP],
        input_total=len(input_rows),
        detector_rows=detector_rows[:_DETECTOR_ROW_CAP],
        detector_total=len(detector_rows),
        artifact_rows=artifact_rows[:_ARTIFACT_CAP],
        artifact_total=len(artifact_rows),
        environment_namespaces=[
            _bounded_text(item)
            for item in (report.execution.environment_namespaces[:10] if report.execution else [])
        ],
        environment_namespace_total=(
            len(report.execution.environment_namespaces) if report.execution else 0
        ),
        task_namespaces=[
            _bounded_text(item)
            for item in (report.execution.task_namespaces[:10] if report.execution else [])
        ],
        task_namespace_total=len(report.execution.task_namespaces) if report.execution else 0,
        group_rows=list(report.aggregates.group_stats.items())[:_GROUP_CAP],
        group_total=len(report.aggregates.group_stats),
        step_rows=report.aggregates.step_series[:_STEP_CAP],
        step_total=len(report.aggregates.step_series),
        top_rollouts=report.aggregates.top_rollouts[:_SNIPPET_CAP],
        bottom_rollouts=report.aggregates.bottom_rollouts[:_SNIPPET_CAP],
        footer_inputs=report.input_files[:_INPUT_CAP],
        footer_input_total=len(report.input_files),
    )


def write_html(report: ReportData, path: Path) -> Path:
    """Write the self-contained HTML report to ``path``.

    Inputs: the report model and the output file path (parent directories are
    created if missing). Returns the path written. Exactly one file is
    produced; the document references no external resources.
    """
    from rolloutscope.output import atomic_write_bytes

    atomic_write_bytes(path, render_html(report).encode("utf-8"))
    return path


__all__ = ["render_html", "write_html"]
