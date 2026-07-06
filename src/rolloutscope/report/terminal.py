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

_SEVERITY_STYLES: dict[str, str] = {
    "critical": "bold red",
    "warning": "yellow",
    "info": "cyan",
}
_SNIPPET_CHARS = 160


def _fmt(value: float | None, digits: int = 4) -> str:
    """Format an optional float compactly; None renders as ``n/a``."""
    if value is None:
        return "n/a"
    return f"{value:.{digits}g}"


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
        names = ", ".join(item.name for item in report.input_files)
        header.append(f"\ninputs: {names}", style="dim")
    active_console.print(Panel(header, title="rolloutscope run summary", expand=False))

    ordered = findings_by_severity(report.findings)
    if not ordered:
        active_console.print("No findings.", style="green")
        return

    table = Table(title="Findings")
    table.add_column("severity")
    table.add_column("detector")
    table.add_column("title")
    table.add_column("fired/total", justify="right")
    table.add_column("max score", justify="right")
    for finding in ordered:
        style = _SEVERITY_STYLES.get(finding.severity, "")
        fired = finding.metrics.get("fired_count")
        total = finding.metrics.get("total_verdicts")
        ratio = f"{fired:g}/{total:g}" if fired is not None and total is not None else "n/a"
        table.add_row(
            Text(finding.severity, style=style),
            Text(finding.detector),
            Text(finding.title),
            Text(ratio),
            Text(_fmt(finding.metrics.get("max_score"), digits=3)),
        )
    active_console.print(table)

    for finding in ordered:
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
