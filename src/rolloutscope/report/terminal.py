"""Rich terminal rendering of ReportData.

A pure function of ReportData (golden rule 7): run header, findings table
ordered by severity with colors, and the top exemplar snippet per finding.
The Console is injectable so tests can capture output.
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from rolloutscope.report.model import ReportData, findings_by_severity
from rolloutscope.schema.execution import UnitCounts

_SEVERITY_STYLES: dict[str, str] = {
    "critical": "bold red",
    "warning": "yellow",
    "info": "cyan",
}
_SNIPPET_CHARS = 160
_FINDING_CAP = 50
_COVERAGE_ROW_CAP = 200
_INPUT_NAME_CAP = 10


def _fmt(value: float | None, digits: int = 4) -> str:
    """Format an optional float compactly; None renders as ``n/a``."""
    if value is None:
        return "n/a"
    return f"{value:.{digits}g}"


def _execution_status(report: ReportData) -> str:
    """Return the recorded analysis status, preserving legacy report readability."""
    return report.execution.status if report.execution is not None else "not recorded"


def render_terminal(report: ReportData, console: Console | None = None) -> None:
    """Print the report summary to a rich Console.

    Inputs: the ReportData to render and an optional Console (injectable for
    tests; a default stdout console is created when omitted). Output: a run
    header panel with counts and reward stats, a findings table ordered by
    severity (critical, warning, info) with severity colors, and the top
    exemplar snippet per finding. All rollout text is printed as plain text,
    never interpreted as rich markup.
    """
    active_console = console if console is not None else Console()
    run = report.aggregates.run_summary

    header = Text()
    header.append(f"analysis status: {_execution_status(report)}\n")
    header.append(f"rollouts analyzed: {run.row_count}\n")
    header.append(
        f"reward mean {_fmt(run.reward_mean)}, std {_fmt(run.reward_std)}, "
        f"min {_fmt(run.reward_min)}, max {_fmt(run.reward_max)}\n"
    )
    header.append(
        f"truncation rate {_fmt(run.truncation_rate)}, "
        f"completion rate {_fmt(run.completion_rate)}\n"
    )
    header.append(f"groups: {len(report.aggregates.group_stats)}", style="dim")
    if report.input_files:
        visible_inputs = report.input_files[:_INPUT_NAME_CAP]
        names = ", ".join(item.name[:80] for item in visible_inputs)
        if len(report.input_files) > len(visible_inputs):
            names += f", ... ({len(report.input_files)} total)"
        header.append(f"\ninputs: {names}", style="dim")
    if report.execution is not None and report.execution.environment_namespaces:
        header.append(
            "\nenvironments: " + ", ".join(report.execution.environment_namespaces[:10]),
            style="dim",
        )
    if report.execution is not None and report.execution.task_namespaces:
        header.append(
            "\ntasks: " + ", ".join(report.execution.task_namespaces[:10]),
            style="dim",
        )
    active_console.print(Panel(header, title="rolloutscope run summary", expand=False))

    if report.detector_results:
        coverage = Table(title="Detector coverage")
        coverage.add_column("detector")
        coverage.add_column("status")
        coverage.add_column("unit")
        coverage.add_column("mode")
        coverage.add_column("run")
        coverage.add_column("fired/eligible", justify="right")
        coverage.add_column("candidates", justify="right")
        coverage.add_column("insufficient", justify="right")
        coverage.add_column("skipped", justify="right")
        coverage.add_column("errors", justify="right")
        coverage_rows = 0
        for result in sorted(report.detector_results, key=lambda item: item.detector):
            units: list[UnitCounts | None] = list(result.units) or [None]
            for unit in units:
                if coverage_rows >= _COVERAGE_ROW_CAP:
                    break
                coverage.add_row(
                    result.detector,
                    result.status,
                    unit.unit if unit is not None else "n/a",
                    unit.mode if unit is not None else "n/a",
                    str(unit.measurements.get("run_id", "n/a")) if unit is not None else "n/a",
                    f"{unit.fired}/{unit.eligible}" if unit is not None else "n/a",
                    str(unit.candidate) if unit is not None else "n/a",
                    str(unit.insufficient_data) if unit is not None else "n/a",
                    str(unit.skipped) if unit is not None else "n/a",
                    str(unit.errors) if unit is not None else "n/a",
                )
                coverage_rows += 1
        active_console.print(coverage)
        total_coverage_rows = sum(max(1, len(result.units)) for result in report.detector_results)
        if total_coverage_rows > coverage_rows:
            active_console.print(
                f"Showing {coverage_rows} of {total_coverage_rows} detector coverage rows."
            )

    ordered = findings_by_severity(report.findings)
    if not ordered:
        if report.execution is not None and report.execution.status != "complete":
            active_console.print(
                f"No finding summary: analysis status is {report.execution.status}.", style="yellow"
            )
        elif report.detector_results and not any(
            unit.eligible for result in report.detector_results for unit in result.units
        ):
            active_console.print("No eligible detector units were evaluated.", style="yellow")
        else:
            active_console.print("No findings.", style="green")
        return

    visible_findings = ordered[:_FINDING_CAP]
    table = Table(title="Findings")
    table.add_column("severity")
    table.add_column("detector")
    table.add_column("unit/mode")
    table.add_column("title")
    table.add_column("fired/total", justify="right")
    table.add_column("max score", justify="right")
    for finding in visible_findings:
        style = _SEVERITY_STYLES.get(finding.severity, "")
        fired = finding.metrics.get("fired_count")
        eligible = finding.metrics.get("eligible_checks")
        if eligible is None:
            eligible = finding.metrics.get("total_verdicts")
        ratio = f"{fired:g}/{eligible:g}" if fired is not None and eligible is not None else "n/a"
        finding_unit = str(getattr(finding, "unit", "check"))
        mode = str(getattr(finding, "mode", "snapshot"))
        table.add_row(
            Text(finding.severity, style=style),
            Text(finding.detector),
            Text(f"{finding_unit}/{mode}"),
            Text(finding.title),
            Text(ratio),
            Text(_fmt(finding.metrics.get("max_score"), digits=3)),
        )
    active_console.print(table)
    if len(ordered) > len(visible_findings):
        active_console.print(
            f"Showing {len(visible_findings)} of {len(ordered)} findings; see JSON for all."
        )

    for finding in visible_findings:
        if not finding.exemplars:
            continue
        exemplar = finding.exemplars[0]
        snippet = exemplar.text.replace("\n", " ")
        if len(snippet) > _SNIPPET_CHARS:
            snippet = snippet[:_SNIPPET_CHARS] + "..."
        line = Text()
        line.append(f"{finding.severity} ", style=_SEVERITY_STYLES.get(finding.severity, ""))
        line.append(f"{finding.detector} exemplar ({exemplar.field}): ", style="bold")
        line.append(snippet, style="dim")
        active_console.print(line)


__all__ = ["render_terminal"]
