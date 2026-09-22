"""HTML report: self-containment, both charts, findings, evidence, and footer.

Self-containment check: the report must open from file:// with zero network
access, so nothing in the document may reference anything external. The
simplest robust check is that the reference-bearing tokens (src=, href=,
@import, url(, and the script / link / iframe / object / embed tags) never
appear at all: every style is inline, every chart is inline SVG, and no
anchor, script, or import exists. The SVG xmlns attribute carries a namespace
identifier, not a fetched resource, and contains none of these tokens.
"""

import hashlib
from collections.abc import Callable
from itertools import chain
from pathlib import Path

import pytest

from rolloutscope.analysis import aggregate_rollouts
from rolloutscope.report import (
    InputFile,
    ReportData,
    describe_input,
    hash_file,
    render_html,
    render_json,
    write_html,
)
from rolloutscope.schema import EvidenceSpan, Finding, SingleTurnRollout, read_rollouts
from rolloutscope.schema.execution import (
    DetectorExecution,
    ExecutionManifest,
    FileIngestion,
    IngestionIssue,
    IngestionSummary,
    OutputArtifact,
    UnitCounts,
)

RolloutFactory = Callable[..., SingleTurnRollout]

FORBIDDEN_TOKENS = (
    "src=",
    "href=",
    "@import",
    "url(",
    "<script",
    "<link",
    "<iframe",
    "<object",
    "<embed",
)


def _finding(severity: str, detector: str, title: str, exemplar_text: str) -> Finding:
    return Finding(
        severity=severity,  # type: ignore[arg-type]
        title=title,
        description="hand-built finding for the html test",
        detector=detector,
        metrics={"fired_count": 1.0, "total_verdicts": 2.0, "max_score": 0.9},
        config_used={"threshold": 0.75},
        exemplars=[
            EvidenceSpan(
                rollout_id="rdeadbeefdeadbeef",
                field="completion",
                text=exemplar_text,
                note="hand-built span",
            )
        ],
    )


@pytest.fixture
def report(eval_run_dir: Path, make_rollout: RolloutFactory) -> ReportData:
    fixture_rows = read_rollouts(eval_run_dir / "results.jsonl")
    step_rows = [
        make_rollout(0.5, 0, group_id="A", step_index=0),
        make_rollout(0.5, 0, group_id="A", step_index=0),
        make_rollout(0.9, 1, group_id="B", step_index=1),
        make_rollout(0.1, 1, group_id="B", step_index=1),
    ]
    findings = [
        _finding("info", "det_info", "INFO_TITLE_MARKER", "info span text"),
        _finding("critical", "det_crit", "CRITICAL_TITLE_MARKER", "SNEAKY <b>MARKER</b>"),
        _finding("warning", "det_warn", "WARNING_TITLE_MARKER", "warning span text"),
    ]
    return ReportData(
        tool_version="0.1.0-test",
        schema_version="1.0",
        input_files=[describe_input(eval_run_dir / "results.jsonl")],
        aggregates=aggregate_rollouts(chain(fixture_rows, step_rows)),
        findings=findings,
    )


def test_document_is_self_contained(report: ReportData) -> None:
    html = render_html(report)
    assert html.startswith("<!doctype html>")
    lowered = html.lower()
    for token in FORBIDDEN_TOKENS:
        assert token not in lowered, f"external-reference token {token!r} found in report"
    assert "<style>" in html


def test_both_charts_present_when_data_present(report: ReportData) -> None:
    html = render_html(report)
    assert html.count("<svg ") == 2
    assert "Reward distribution" in html
    assert "dead group fraction" in html


def test_step_chart_absent_without_step_series(eval_run_dir: Path, report: ReportData) -> None:
    no_steps = report.model_copy(
        update={"aggregates": aggregate_rollouts(read_rollouts(eval_run_dir / "results.jsonl"))}
    )
    assert render_html(no_steps).count("<svg ") == 1


def test_findings_titles_ordered_by_severity(report: ReportData) -> None:
    html = render_html(report)
    for finding in report.findings:
        assert finding.title in html
    assert (
        html.index("CRITICAL_TITLE_MARKER")
        < html.index("WARNING_TITLE_MARKER")
        < html.index("INFO_TITLE_MARKER")
    )


def test_details_blocks_and_marked_evidence(report: ReportData) -> None:
    html = render_html(report)
    assert "<details>" in html
    # Evidence text is escaped, then wrapped in a mark element.
    assert "<mark>SNEAKY &lt;b&gt;MARKER&lt;/b&gt;</mark>" in html
    assert "<b>MARKER</b>" not in html


def test_config_appendix_shows_thresholds(report: ReportData) -> None:
    html = render_html(report)
    assert "threshold" in html
    assert "0.75" in html
    # Aggregation config rows are part of the appendix too.
    assert "histogram_bins" in html


def test_reproducibility_footer(report: ReportData, eval_run_dir: Path) -> None:
    html = render_html(report)
    assert "0.1.0-test" in html
    assert "schema version 1.0" in html
    assert "results.jsonl" in html
    assert hash_file(eval_run_dir / "results.jsonl") in html


def test_write_html_produces_exactly_one_file(report: ReportData, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    written = write_html(report, out_dir / "report.html")
    assert written.is_file()
    assert [entry.name for entry in out_dir.iterdir()] == ["report.html"]
    assert written.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_hash_file_matches_hashlib(tmp_path: Path) -> None:
    target = tmp_path / "data.jsonl"
    payload = b'{"reward": 1.0}\n' * 100
    target.write_bytes(payload)
    assert hash_file(target) == hashlib.sha256(payload).hexdigest()


def test_execution_accounting_and_diagnostics_are_escaped(report: ReportData) -> None:
    ingestion = FileIngestion(
        path="nested/results.jsonl",
        format="normalized",
        observed=4,
        accepted=2,
        rejected=1,
        blank=1,
        duplicates=1,
        examples=[
            IngestionIssue(
                line=3,
                reason="invalid_row",
                message="<img src=x onerror=alert(1)>",
            )
        ],
    )
    execution = ExecutionManifest(
        tool_version="0.1.0-test",
        schema_version="1.0",
        input_format="normalized",
        status="partial",
        ingestion=IngestionSummary(files=[ingestion]),
    )
    detector = DetectorExecution(
        detector="format_only_wins",
        status="insufficient_data",
        units=[UnitCounts(candidate=2, insufficient_data=2)],
    )
    html = render_html(
        report.model_copy(
            update={
                "execution": execution,
                "detector_results": [detector],
                "findings": [],
                "input_files": [],
            }
        )
    )
    assert "Analysis status: <strong>partial</strong>" in html
    assert '2</td><td class="num">1</td><td class="num">1</td>' in html
    assert "No finding summary is shown because analysis status is partial" in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in html
    assert "<img src=x" not in html
    assert "format_only_wins" in html


def test_human_report_rows_are_bounded_with_visible_totals(report: ReportData) -> None:
    inputs = [
        InputFile(name=f"input-{index}.jsonl", sha256=f"{index:064x}"[-64:]) for index in range(101)
    ]
    detectors = [
        DetectorExecution(
            detector=f"detector-{index}",
            units=[UnitCounts(candidate=1, eligible=1, clean=1)],
        )
        for index in range(201)
    ]
    execution = ExecutionManifest(
        tool_version="0.1.0-test",
        schema_version="2.0",
        input_format="normalized",
        effective_config={f"setting-{index}": index for index in range(205)},
        artifacts=[
            OutputArtifact(name=f"artifact-{index}.json", role="test") for index in range(51)
        ],
    )
    html = render_html(
        report.model_copy(
            update={
                "input_files": inputs,
                "detector_results": detectors,
                "execution": execution,
            }
        )
    )
    assert "Showing 100 of 101 input files." in html
    assert "Showing 200 of 201 detector coverage rows." in html
    assert "Showing 200 of" in html and "configuration rows" in html
    assert "Showing 50 of 51 output artifacts." in html
    assert "showing 100 of 101 input files" in html


def test_html_bounds_evidence_but_json_retains_it(report: ReportData) -> None:
    long_text = "x" * 1_500 + "COMPLETE-EVIDENCE-END"
    finding = _finding("critical", "det", "bounded evidence", long_text)
    complete = report.model_copy(update={"findings": [finding]})
    assert "COMPLETE-EVIDENCE-END" not in render_html(complete)
    assert "COMPLETE-EVIDENCE-END" in render_json(complete)


def test_multirun_step_series_are_not_connected_in_one_chart(report: ReportData) -> None:
    original = report.aggregates.step_series
    assert original
    multi_run = [
        *[item.model_copy(update={"run_id": "run-a"}) for item in original],
        *[item.model_copy(update={"run_id": "run-b"}) for item in original],
    ]
    aggregates = report.aggregates.model_copy(update={"step_series": multi_run})
    html = render_html(report.model_copy(update={"aggregates": aggregates}))
    assert html.count("<svg ") == 1
    assert "Per-step chart omitted because it would combine separate run series" in html
    assert "run-a" in html and "run-b" in html
