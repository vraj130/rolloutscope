"""End-to-end tests for the CLI: the full adapter to detectors to report seam.

These exercise the wiring the library packages leave to the CLI, so they cover
the integration criteria in the plan: analyze produces a terminal summary, a
deterministic JSON sidecar, and a single-file HTML report that opens from
file://; convert normalizes on disk; detectors list and schema export work.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rolloutscope import __version__
from rolloutscope.cli import (
    FailOn,
    RunConfig,
    app,
    build_report_data,
    exit_code_for,
    load_run_config,
)

runner = CliRunner()

# The four detectors the demo fixture is built to fire.
EXPECTED_FIRED = {
    "answer_leakage_echo",
    "degenerate_repetition",
    "format_only_wins",
    "verifier_tamper",
}
# Any external resource reference would break the file:// use case.
_EXTERNAL_REF = re.compile(r"src=|href=|@import|url\(|<script|<link|<iframe|https?://")


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_detectors_list_shows_all_six() -> None:
    result = runner.invoke(app, ["detectors", "list"])
    assert result.exit_code == 0
    assert "6 detector(s) discovered" in result.stdout
    for name in ("verifier_tamper", "answer_leakage_echo", "reward_saturation_group_collapse"):
        assert name in result.stdout


def test_schema_export_is_valid_discriminated_union() -> None:
    result = runner.invoke(app, ["schema", "export"])
    assert result.exit_code == 0
    schema = json.loads(result.stdout)
    # The discriminated union exports a oneOf plus a discriminator mapping on kind.
    assert "oneOf" in schema
    assert schema.get("discriminator", {}).get("propertyName") == "kind"
    assert set(schema["discriminator"]["mapping"]) == {"single_turn", "multi_turn"}


def test_schema_export_to_file(tmp_path: Path) -> None:
    out = tmp_path / "sub" / "schema.json"
    result = runner.invoke(app, ["schema", "export", "--out", str(out)])
    assert result.exit_code == 0
    assert out.is_file()
    assert json.loads(out.read_text())["discriminator"]["propertyName"] == "kind"


def test_build_report_data_fires_expected_detectors(demo_dir: Path) -> None:
    report = build_report_data(demo_dir)
    assert report.tool_version == __version__
    assert report.aggregates.run_summary.row_count == 10
    fired = {finding.detector for finding in report.findings}
    assert fired >= EXPECTED_FIRED
    # Input file recorded for the reproducibility footer.
    assert [f.name for f in report.input_files] == ["results.jsonl"]
    assert report.input_files[0].sha256


def test_build_report_data_selected_detector(demo_dir: Path) -> None:
    report = build_report_data(demo_dir, selected={"verifier_tamper"})
    assert {finding.detector for finding in report.findings} == {"verifier_tamper"}


def test_build_report_data_unknown_detector_raises(demo_dir: Path) -> None:
    with pytest.raises(ValueError, match="unknown detector"):
        build_report_data(demo_dir, selected={"no_such_detector"})


def test_analyze_writes_reports(demo_dir: Path, tmp_path: Path) -> None:
    html = tmp_path / "report.html"
    js = tmp_path / "report.json"
    result = runner.invoke(app, ["analyze", str(demo_dir), "--out", str(html), "--json", str(js)])
    assert result.exit_code == 0
    assert html.is_file() and js.is_file()
    # Terminal summary printed.
    assert "rollouts analyzed: 10" in result.stdout


def test_analyze_html_is_self_contained(demo_dir: Path, tmp_path: Path) -> None:
    html = tmp_path / "report.html"
    runner.invoke(app, ["analyze", str(demo_dir), "--out", str(html), "--quiet"])
    text = html.read_text(encoding="utf-8")
    assert text.lstrip().startswith("<!doctype html>")
    # Strip the SVG xmlns attribute (a namespace URI, not a fetch) before scanning.
    scanned = text.replace('xmlns="http://www.w3.org/2000/svg"', "")
    leaks = [line for line in scanned.splitlines() if _EXTERNAL_REF.search(line)]
    assert leaks == [], f"HTML references external resources: {leaks[:3]}"


def test_analyze_json_is_deterministic(demo_dir: Path, tmp_path: Path) -> None:
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    runner.invoke(app, ["analyze", str(demo_dir), "--json", str(first), "--quiet"])
    runner.invoke(app, ["analyze", str(demo_dir), "--json", str(second), "--quiet"])
    assert first.read_bytes() == second.read_bytes()
    payload = json.loads(first.read_text())
    assert payload["tool_version"] == __version__
    assert payload["aggregates"]["run_summary"]["row_count"] == 10


def test_analyze_fail_on_critical_exits_nonzero(demo_dir: Path) -> None:
    result = runner.invoke(app, ["analyze", str(demo_dir), "--quiet", "--fail-on", "critical"])
    assert result.exit_code == 1


def test_analyze_fail_on_none_exits_zero(demo_dir: Path) -> None:
    result = runner.invoke(app, ["analyze", str(demo_dir), "--quiet"])
    assert result.exit_code == 0


def test_analyze_unknown_detector_is_clean_error(demo_dir: Path) -> None:
    result = runner.invoke(app, ["analyze", str(demo_dir), "-d", "nope", "--quiet"])
    assert result.exit_code != 0
    assert "unknown detector" in result.output


def test_convert_roundtrips(demo_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "normalized.jsonl"
    result = runner.invoke(app, ["convert", str(demo_dir), "--out", str(out)])
    assert result.exit_code == 0
    assert "wrote 10 normalized rollouts" in result.stdout

    from rolloutscope.schema import read_rollouts

    rows = list(read_rollouts(out))
    assert len(rows) == 10
    # Every converted row carries its content-derived ids.
    assert all(row.rollout_id and row.group_id and row.run_id for row in rows)
    assert {row.kind for row in rows} == {"single_turn", "multi_turn"}


def test_analyze_unrecognized_path_is_clean_error(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    result = runner.invoke(app, ["analyze", str(empty), "--quiet"])
    assert result.exit_code != 0
    assert "no adapter recognizes" in result.output


def test_load_run_config_defaults_when_none() -> None:
    config = load_run_config(None)
    assert isinstance(config, RunConfig)
    assert config.severity.critical_at == 0.8


def test_load_run_config_overrides(tmp_path: Path, demo_dir: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[severity]\ncritical_at = 0.95\nwarning_at = 0.9\n"
        "[detectors.verifier_tamper]\nmin_matches = 5\n",
        encoding="utf-8",
    )
    config = load_run_config(config_path)
    assert config.severity.critical_at == 0.95
    assert config.detectors.verifier_tamper.min_matches == 5


def test_load_run_config_bad_toml_is_clean_error(tmp_path: Path) -> None:
    from typer import BadParameter

    bad = tmp_path / "bad.toml"
    bad.write_text("this is = = not toml", encoding="utf-8")
    with pytest.raises(BadParameter):
        load_run_config(bad)


def test_config_changes_severity_via_cli(demo_dir: Path, tmp_path: Path) -> None:
    # With a very high critical threshold, formerly-critical findings drop to
    # warning, so --fail-on critical no longer trips.
    config_path = tmp_path / "config.toml"
    config_path.write_text("[severity]\ncritical_at = 1.0\nwarning_at = 0.99\n", encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "analyze",
            str(demo_dir),
            "--quiet",
            "--config",
            str(config_path),
            "--fail-on",
            "critical",
        ],
    )
    assert result.exit_code == 0


def test_exit_code_for_helper() -> None:
    report = build_report_data(Path("tests/fixtures/demo"))
    assert exit_code_for(report.findings, FailOn.none) == 0
    assert exit_code_for(report.findings, FailOn.critical) == 1
    assert exit_code_for(report.findings, FailOn.warning) == 1
