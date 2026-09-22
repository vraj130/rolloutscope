"""End-to-end tests for the CLI: the full adapter to detectors to report seam.

These exercise the wiring the library packages leave to the CLI, so they cover
the integration criteria in the plan: analyze produces a terminal summary, a
deterministic JSON sidecar, and a single-file HTML report that opens from
file://; convert normalizes on disk; detectors list and schema export work.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

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
from rolloutscope.schema.execution import DetectorExecution, UnitCounts

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
    assert [(f.name, f.role) for f in report.input_files] == [
        ("results.jsonl", "rollout input"),
        ("metadata.json", "run metadata"),
    ]
    assert all(item.sha256 for item in report.input_files)


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


def test_analyze_writes_complete_verdicts_and_ledger(demo_dir: Path, tmp_path: Path) -> None:
    html = tmp_path / "report.html"
    result = runner.invoke(app, ["analyze", str(demo_dir), "--out", str(html), "--quiet"])
    assert result.exit_code == 0
    verdicts = tmp_path / "report.html.verdicts.jsonl"
    ledger = tmp_path / "report.html.manifest.json"
    assert verdicts.is_file() and ledger.is_file()
    records = [json.loads(line) for line in verdicts.read_text().splitlines()]
    assert any(record["source_occurrences"] for record in records)
    for record in records:
        if record["rollout_ids"]:
            assert record["source_occurrences"]
        if record["fired"]:
            assert all(
                {"occurrence_id", "source_path", "line"} <= source.keys()
                for source in record["source_occurrences"]
            )
    inventory = json.loads(ledger.read_text())
    assert {item["role"] for item in inventory["artifacts"]} == {
        "HTML report",
        "complete verdict JSONL",
    }
    assert inventory["manifest_version"] == "1.0"
    for item in inventory["artifacts"]:
        artifact = tmp_path / item["name"]
        payload = artifact.read_bytes()
        assert item["sha256"] == hashlib.sha256(payload).hexdigest()
        assert item["size_bytes"] == len(payload)


def test_analyze_verdicts_only_writes_complete_artifact_and_ledger(
    demo_dir: Path, tmp_path: Path
) -> None:
    verdicts = tmp_path / "all-verdicts.jsonl"
    result = runner.invoke(app, ["analyze", str(demo_dir), "--verdicts", str(verdicts), "--quiet"])
    assert result.exit_code == 0
    assert verdicts.is_file()
    ledger = tmp_path / "all-verdicts.jsonl.manifest.json"
    assert ledger.is_file()
    assert json.loads(ledger.read_text())["artifacts"][0]["name"] == verdicts.name


def test_analyze_rejects_output_collision_before_writing(demo_dir: Path) -> None:
    source = demo_dir / "results.jsonl"
    before = source.read_bytes()
    result = runner.invoke(app, ["analyze", str(demo_dir), "--json", str(source), "--quiet"])
    assert result.exit_code != 0
    assert "input/output collision" in result.output
    assert source.read_bytes() == before


def test_convert_rejects_direct_source_collision_without_changing_bytes(
    demo_dir: Path, tmp_path: Path
) -> None:
    source = tmp_path / "raw.jsonl"
    source.write_bytes((demo_dir / "results.jsonl").read_bytes())
    before = source.read_bytes()
    result = runner.invoke(app, ["convert", str(source), "--out", str(source)])
    assert result.exit_code == 2
    assert "input/output collision" in result.output
    assert source.read_bytes() == before


def test_convert_rejects_relative_source_alias_without_changing_bytes(
    demo_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "raw.jsonl"
    source.write_bytes((demo_dir / "results.jsonl").read_bytes())
    before = source.read_bytes()
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["convert", "raw.jsonl", "--out", "./raw.jsonl"])
    assert result.exit_code == 2
    assert "input/output collision" in result.output
    assert source.read_bytes() == before


def test_convert_rejects_symlink_source_alias_without_changing_bytes(
    demo_dir: Path, tmp_path: Path
) -> None:
    source = tmp_path / "raw.jsonl"
    alias = tmp_path / "alias.jsonl"
    source.write_bytes((demo_dir / "results.jsonl").read_bytes())
    try:
        alias.symlink_to(source)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    before = source.read_bytes()
    result = runner.invoke(app, ["convert", str(source), "--out", str(alias)])
    assert result.exit_code == 2
    assert "input/output collision" in result.output
    assert source.read_bytes() == before


def test_convert_preflights_metadata_collision(demo_dir: Path, tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "results.jsonl").write_bytes((demo_dir / "results.jsonl").read_bytes())
    metadata = run / "metadata.json"
    metadata.write_text('{"model":"test"}\n', encoding="utf-8")
    before = metadata.read_bytes()
    result = runner.invoke(app, ["convert", str(run), "--out", str(metadata)])
    assert result.exit_code == 2
    assert "input/output collision" in result.output
    assert metadata.read_bytes() == before


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
    payload = json.loads(first.read_text())
    second_payload = json.loads(second.read_text())
    # Artifact names are intentionally relative to the requested primary output.
    payload["execution"].pop("artifact_manifest")
    second_payload["execution"].pop("artifact_manifest")
    payload["execution"].pop("artifacts")
    second_payload["execution"].pop("artifacts")
    assert payload == second_payload
    assert payload["tool_version"] == __version__
    assert payload["aggregates"]["run_summary"]["row_count"] == 10


def test_json_records_source_and_metadata_hashes(demo_dir: Path, tmp_path: Path) -> None:
    destination = tmp_path / "report.json"
    result = runner.invoke(app, ["analyze", str(demo_dir), "--json", str(destination), "--quiet"])
    assert result.exit_code == 0
    payload = json.loads(destination.read_text())
    inputs = {item["name"]: item for item in payload["input_files"]}
    assert set(inputs) == {"results.jsonl", "metadata.json"}
    assert inputs["metadata.json"]["role"] == "run metadata"
    for name, item in inputs.items():
        source_payload = (demo_dir / name).read_bytes()
        assert item["sha256"] == hashlib.sha256(source_payload).hexdigest()
        assert item["size_bytes"] == len(source_payload)
    ingestion = payload["execution"]["ingestion"]["files"][0]
    assert ingestion["sha256"] == inputs["results.jsonl"]["sha256"]
    assert ingestion["size_bytes"] == inputs["results.jsonl"]["size_bytes"]
    assert payload["execution"]["environment_namespaces"] == ["demo-mixed-bench"]
    assert payload["execution"]["task_namespaces"] == ["demo-mixed-bench"]
    assert inputs["results.jsonl"]["metadata"]["environment_namespace"] == "demo-mixed-bench"
    assert inputs["results.jsonl"]["metadata"]["task_namespace"] == "demo-mixed-bench"


def test_detector_failure_is_exit_two_and_preserves_outputs(
    demo_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failed_execution(
        detector: Any, rollouts: object, config: object
    ) -> tuple[list[Any], DetectorExecution]:
        name = str(detector.name)
        return [], DetectorExecution(
            detector=name,
            status="failed",
            units=[UnitCounts(unit="run", candidate=1, errors=1)],
            errors=["deliberate detector failure"],
        )

    monkeypatch.setattr("rolloutscope.cli.execute_detector", failed_execution)
    output = tmp_path / "report.json"
    output.write_bytes(b"previous output\n")
    result = runner.invoke(
        app,
        [
            "analyze",
            str(demo_dir),
            "--detector",
            "verifier_tamper",
            "--json",
            str(output),
            "--fail-on",
            "none",
            "--quiet",
        ],
    )
    assert result.exit_code == 2
    assert output.read_bytes() == b"previous output\n"
    assert not (tmp_path / "report.json.verdicts.jsonl").exists()
    assert not (tmp_path / "report.json.manifest.json").exists()


def test_output_serialization_failure_preserves_every_existing_artifact(
    demo_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    html = tmp_path / "report.html"
    verdicts = tmp_path / "report.html.verdicts.jsonl"
    ledger = tmp_path / "report.html.manifest.json"
    for path in (html, verdicts, ledger):
        path.write_bytes(b"previous " + path.name.encode() + b"\n")
    before = {path: path.read_bytes() for path in (html, verdicts, ledger)}

    def fail_render(report: object) -> str:
        raise RuntimeError("deliberate serialization failure")

    monkeypatch.setattr("rolloutscope.cli.render_html", fail_render)
    result = runner.invoke(app, ["analyze", str(demo_dir), "--out", str(html), "--quiet"])
    assert result.exit_code == 2
    assert "could not write analysis outputs" in result.output
    assert {path: path.read_bytes() for path in before} == before
    assert not list(tmp_path.glob(".*.tmp"))


def test_selected_discovery_failure_is_recorded_instead_of_unknown(
    demo_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failed_discovery(*, diagnostics: dict[str, str] | None = None) -> dict[str, object]:
        assert diagnostics is not None
        diagnostics["broken_plugin"] = "failed to load: deliberate"
        return {}

    monkeypatch.setattr("rolloutscope.cli.discover_detectors", failed_discovery)
    report = build_report_data(demo_dir, selected={"broken_plugin"})
    assert report.execution is not None
    assert report.execution.status == "failed"
    assert report.execution.selected_detectors == ["broken_plugin"]
    assert report.detector_results[0].detector == "broken_plugin"
    assert report.detector_results[0].status == "failed"


def test_unselected_discovery_failure_does_not_fail_scoped_run(
    demo_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rolloutscope.detectors import VerifierTamperDetector

    def partial_discovery(*, diagnostics: dict[str, str] | None = None) -> dict[str, object]:
        assert diagnostics is not None
        diagnostics["broken_plugin"] = "failed to load: deliberate"
        return {"verifier_tamper": VerifierTamperDetector()}

    monkeypatch.setattr("rolloutscope.cli.discover_detectors", partial_discovery)
    report = build_report_data(demo_dir, selected={"verifier_tamper"})
    assert report.execution is not None
    assert report.execution.status == "complete"
    assert report.execution.selected_detectors == ["verifier_tamper"]


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


def test_convert_partial_policy_preserves_existing_output_on_reject(
    demo_dir: Path, tmp_path: Path
) -> None:
    first = (demo_dir / "results.jsonl").read_bytes().splitlines()[0]
    source = tmp_path / "partial.jsonl"
    source.write_bytes(first + b"\n{}\n")
    output = tmp_path / "normalized.jsonl"
    output.write_bytes(b"previous output\n")

    rejected = runner.invoke(
        app,
        ["convert", str(source), "--out", str(output), "--partial", "reject"],
    )
    assert rejected.exit_code == 2
    assert "accepted=1 rejected=1: partial input rejected by policy" in rejected.output
    assert output.read_bytes() == b"previous output\n"
    assert not list(tmp_path.glob(".normalized.jsonl.*.tmp"))

    allowed = runner.invoke(
        app,
        ["convert", str(source), "--out", str(output), "--partial", "allow"],
    )
    assert allowed.exit_code == 0
    assert "wrote 1 normalized rollouts" in allowed.output
    assert output.read_bytes() != b"previous output\n"


def test_convert_zero_usable_records_preserves_existing_output(tmp_path: Path) -> None:
    source = tmp_path / "invalid.jsonl"
    source.write_bytes(b"{}\nnot-json\n\n")
    output = tmp_path / "normalized.jsonl"
    output.write_bytes(b"previous output\n")
    result = runner.invoke(app, ["convert", str(source), "--out", str(output)])
    assert result.exit_code == 2
    assert "conversion rejected: status=unsupported" in result.output
    assert all(token in result.output for token in ("accepted=0", "rejected=2", "blank=1"))
    assert output.read_bytes() == b"previous output\n"


def test_analyze_partial_policy_and_status(demo_dir: Path, tmp_path: Path) -> None:
    first = (demo_dir / "results.jsonl").read_bytes().splitlines()[0]
    source = tmp_path / "partial.jsonl"
    source.write_bytes(first + b"\n{}\n")
    allowed_path = tmp_path / "allowed.json"
    allowed = runner.invoke(
        app,
        ["analyze", str(source), "--json", str(allowed_path), "--partial", "allow", "--quiet"],
    )
    assert allowed.exit_code == 0
    payload = json.loads(allowed_path.read_text())
    assert payload["execution"]["status"] == "partial"
    assert payload["execution"]["ingestion"]["accepted"] == 1
    assert payload["execution"]["ingestion"]["rejected"] == 1

    rejected_path = tmp_path / "rejected.json"
    rejected_path.write_bytes(b"previous output\n")
    rejected = runner.invoke(
        app,
        [
            "analyze",
            str(source),
            "--json",
            str(rejected_path),
            "--partial",
            "reject",
            "--quiet",
        ],
    )
    assert rejected.exit_code == 2
    assert rejected_path.read_bytes() == b"previous output\n"


def test_analyze_zero_usable_records_is_analysis_error(tmp_path: Path) -> None:
    source = tmp_path / "invalid.jsonl"
    source.write_bytes(b"{}\nnot-json\n")
    result = runner.invoke(app, ["analyze", str(source)])
    assert result.exit_code == 2
    assert "analysis status: unsupported" in result.output
    assert "No finding summary: analysis status is unsupported" in result.output
    assert "No findings." not in result.output


def test_run_id_override_is_recorded_and_survives_conversion(
    demo_dir: Path, tmp_path: Path
) -> None:
    source = tmp_path / "legacy.jsonl"
    source.write_bytes((demo_dir / "results.jsonl").read_bytes())
    report_path = tmp_path / "report.json"
    analyzed = runner.invoke(
        app,
        [
            "analyze",
            str(source),
            "--json",
            str(report_path),
            "--run-id",
            "legacy-run",
            "--quiet",
        ],
    )
    assert analyzed.exit_code == 0
    assert json.loads(report_path.read_text())["execution"]["run_ids"] == ["legacy-run"]

    normalized = tmp_path / "normalized.jsonl"
    converted = runner.invoke(
        app,
        [
            "convert",
            str(source),
            "--out",
            str(normalized),
            "--run-id",
            "legacy-run",
        ],
    )
    assert converted.exit_code == 0
    from rolloutscope.schema import read_rollouts

    assert {row.run_id for row in read_rollouts(normalized)} == {"legacy-run"}


def test_normalized_input_rejects_run_id_override(demo_dir: Path, tmp_path: Path) -> None:
    normalized = tmp_path / "normalized.jsonl"
    converted = runner.invoke(app, ["convert", str(demo_dir), "--out", str(normalized)])
    assert converted.exit_code == 0
    result = runner.invoke(
        app,
        ["analyze", str(normalized), "--run-id", "replacement", "--quiet"],
    )
    assert result.exit_code == 2
    assert "applies only to legacy raw input" in result.output


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


@pytest.mark.parametrize(
    ("config_text", "expected_path"),
    [
        ("[severity]\ncriticall_at = 0.8\n", "severity.criticall_at"),
        (
            "[detectors.verifier_tamper]\ntest_path_regex = '[unterminated'\n",
            "detectors.verifier_tamper.test_path_regex",
        ),
    ],
)
def test_invalid_config_fails_before_input_discovery(
    tmp_path: Path, config_text: str, expected_path: str
) -> None:
    config = tmp_path / "bad.toml"
    config.write_text(config_text, encoding="utf-8")
    result = runner.invoke(
        app,
        ["analyze", str(tmp_path / "missing.jsonl"), "--config", str(config), "--quiet"],
    )
    assert result.exit_code == 2
    assert "invalid config" in result.output
    assert expected_path in result.output
    assert "no adapter recognizes" not in result.output


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
