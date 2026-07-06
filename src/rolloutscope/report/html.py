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
    rows: list[tuple[str, str, str]] = [
        ("aggregation", key, _config_value(value))
        for key, value in report.aggregates.config.model_dump().items()
    ]
    seen: set[tuple[str, str, str]] = set()
    for finding in findings_by_severity(report.findings):
        for key in sorted(finding.config_used):
            row = (finding.detector, key, _config_value(finding.config_used[key]))
            if row not in seen:
                seen.add(row)
                rows.append(row)
    return rows


def render_html(report: ReportData) -> str:
    """Render the one-file HTML report from a ReportData.

    Input: the report model. Output: a complete, self-contained HTML document
    string with findings ordered by severity, evidence spans highlighted via
    mark elements, both SVG charts when their data is present, a config
    appendix, and the reproducibility footer.
    """
    template = _environment().get_template("report.html.j2")
    return template.render(
        report=report,
        stats=_stat_tiles(report),
        findings=findings_by_severity(report.findings),
        histogram_svg=_histogram_svg(report),
        step_svg=_step_svg(report),
        config_rows=_config_rows(report),
    )


def write_html(report: ReportData, path: Path) -> Path:
    """Write the self-contained HTML report to ``path``.

    Inputs: the report model and the output file path (parent directories are
    created if missing). Returns the path written. Exactly one file is
    produced; the document references no external resources.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(report), encoding="utf-8")
    return path


__all__ = ["render_html", "write_html"]
