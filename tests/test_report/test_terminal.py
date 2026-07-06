"""Terminal rendering: captured through an injected recording Console."""

from pathlib import Path

from rich.console import Console

from rolloutscope.analysis import aggregate_rollouts
from rolloutscope.report import ReportData, render_terminal
from rolloutscope.schema import EvidenceSpan, Finding, read_rollouts


def _console() -> Console:
    return Console(record=True, width=140, force_terminal=False)


def _report(eval_run_dir: Path, findings: list[Finding]) -> ReportData:
    return ReportData(
        tool_version="0.1.0",
        schema_version="1.0",
        aggregates=aggregate_rollouts(read_rollouts(eval_run_dir / "results.jsonl")),
        findings=findings,
    )


def _finding(severity: str, detector: str, title: str, exemplar_text: str) -> Finding:
    return Finding(
        severity=severity,  # type: ignore[arg-type]
        title=title,
        description="hand-built finding",
        detector=detector,
        metrics={"fired_count": 1.0, "total_verdicts": 4.0, "max_score": 0.9},
        exemplars=[EvidenceSpan(rollout_id="r1", field="completion", text=exemplar_text)],
    )


def test_header_carries_run_stats(eval_run_dir: Path) -> None:
    console = _console()
    render_terminal(_report(eval_run_dir, []), console)
    text = console.export_text()
    assert "rollouts analyzed: 5" in text
    assert "reward mean 0.64" in text
    assert "truncation rate 0" in text
    assert "completion rate 1" in text


def test_no_findings_message(eval_run_dir: Path) -> None:
    console = _console()
    render_terminal(_report(eval_run_dir, []), console)
    assert "No findings." in console.export_text()


def test_findings_ordered_by_severity_with_exemplars(eval_run_dir: Path) -> None:
    findings = [
        _finding("info", "det_info", "info level finding", "info exemplar span"),
        _finding("critical", "det_crit", "critical level finding", "critical exemplar span"),
        _finding("warning", "det_warn", "warning level finding", "warning exemplar span"),
    ]
    console = _console()
    render_terminal(_report(eval_run_dir, findings), console)
    text = console.export_text()

    assert text.index("critical level finding") < text.index("warning level finding")
    assert text.index("warning level finding") < text.index("info level finding")
    for snippet in ("critical exemplar span", "warning exemplar span", "info exemplar span"):
        assert snippet in text
    assert "1/4" in text
