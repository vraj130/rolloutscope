"""Command-line interface for rolloutscope.

Thin typer wrapper over the library packages (scaffold principle: all real work
lives in importable functions). The wiring functions ``load_run_config`` and
``build_report_data`` are plain, testable functions; the typer commands only
parse arguments, call them, print, and map findings to an exit code. This is the
one module allowed to import adapters, detectors, analysis, and report together.
"""

from __future__ import annotations

import enum
import hashlib
import logging
import subprocess
import tomllib
from collections.abc import Iterable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Literal

import orjson
import typer
from pydantic import BaseModel, ConfigDict, ValidationError
from rich.console import Console
from rich.table import Table

from rolloutscope import __version__
from rolloutscope.adapters import resolve_adapter
from rolloutscope.adapters.base import Adapter, RunManifest
from rolloutscope.analysis import (
    AggregateConfig,
    SeverityThresholds,
    aggregate_rollouts,
    assemble_findings,
)
from rolloutscope.detectors import DetectorConfig, discover_detectors, execute_detector
from rolloutscope.output import OutputTransaction, validate_output_paths
from rolloutscope.report import (
    InputFile,
    ReportData,
    describe_input,
    render_html,
    render_json_bytes,
    render_terminal,
)
from rolloutscope.schema import (
    SCHEMA_VERSION,
    Finding,
    MultiTurnRollout,
    SingleTurnRollout,
    SourceOccurrence,
    Verdict,
    rollout_json_schema,
    write_rollouts,
)
from rolloutscope.schema.execution import (
    DetectorExecution,
    ExecutionManifest,
    IngestionSummary,
    OutputArtifact,
)

logger = logging.getLogger(__name__)

_JSON_OPTIONS = orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE
_SEVERITY_RANK: dict[str, int] = {"info": 0, "warning": 1, "critical": 2}


class IngestionConfig(BaseModel):
    """Policy for a run with valid and invalid source records."""

    model_config = ConfigDict(extra="forbid")
    partial: Literal["allow", "reject"] = "allow"


class RunConfig(BaseModel):
    """Full run configuration, loadable from a TOML file.

    ``detectors`` carries the per-detector thresholds, ``aggregation`` the
    histogram and top-k settings, and ``severity`` the score-to-severity
    mapping. Every field defaults to its model's conservative heuristic
    defaults, so an omitted section (or no config file at all) is valid.
    """

    model_config = ConfigDict(extra="forbid")

    detectors: DetectorConfig = DetectorConfig()
    aggregation: AggregateConfig = AggregateConfig()
    severity: SeverityThresholds = SeverityThresholds()
    ingestion: IngestionConfig = IngestionConfig()


class FailOn(enum.StrEnum):
    """Minimum fired-finding severity that makes ``analyze`` exit non-zero."""

    none = "none"
    info = "info"
    warning = "warning"
    critical = "critical"


def load_run_config(path: Path | None) -> RunConfig:
    """Load a RunConfig from a TOML file, or return defaults when path is None.

    Input: an optional path to a TOML config file. The file's tables map onto
    RunConfig (``[detectors.<name>]``, ``[aggregation]``, ``[severity]``).
    Raises typer.BadParameter when the file is missing, is not valid TOML, or
    does not satisfy the config schema, so the CLI reports a clean error rather
    than a traceback.
    """
    if path is None:
        return RunConfig()
    if not path.is_file():
        raise typer.BadParameter(f"config file not found: {path}")
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise typer.BadParameter(f"invalid TOML in {path}: {exc}") from exc
    try:
        return RunConfig.model_validate(raw)
    except ValidationError as exc:
        raise typer.BadParameter(f"invalid config in {path}: {exc}") from exc


def build_report_data(
    path: Path,
    *,
    config: RunConfig | None = None,
    selected: set[str] | None = None,
    include_clean: bool = False,
    adapter: Adapter | None = None,
    manifest: RunManifest | None = None,
) -> ReportData:
    """Run the full analysis pipeline over an artifact path and return ReportData.

    Inputs: the artifact ``path`` (a run directory or a single JSONL file,
    routed by ``resolve_adapter``); an optional ``config`` (defaults applied
    when None); ``selected`` restricting which discovered detectors run (None
    means all); ``include_clean`` to emit info findings for detectors that ran
    but did not fire. Rollouts are materialized into a list because detectors do
    contrastive, whole-group analysis; aggregation then runs over that same
    list. Raises ValueError when ``selected`` names an unknown detector.
    """
    cfg = config if config is not None else RunConfig()
    active_adapter = adapter if adapter is not None else resolve_adapter(path)
    run_manifest = manifest if manifest is not None else active_adapter.load_run(path)
    metadata_inputs = [
        describe_input(
            metadata_path,
            relative_to=run_manifest.root,
            format="json",
            adapter_version=run_manifest.adapter_version,
            role="run metadata",
        )
        for metadata_path in run_manifest.metadata_sources
    ]
    diagnostics = IngestionSummary()
    rollouts = list(active_adapter.load_manifest(run_manifest, diagnostics=diagnostics))
    discovery_errors: dict[str, str] = {}
    detectors = discover_detectors(diagnostics=discovery_errors)
    known_detectors = set(detectors) | set(discovery_errors)
    if selected is not None:
        unknown = selected - known_detectors
        if unknown:
            available = ", ".join(sorted(known_detectors))
            raise ValueError(
                f"unknown detector(s): {', '.join(sorted(unknown))}. Available: {available}"
            )
        detectors = {name: det for name, det in detectors.items() if name in selected}
        discovery_errors = {
            name: message for name, message in discovery_errors.items() if name in selected
        }

    verdicts = []
    config_used: dict[str, dict[str, object]] = {}
    detector_results: list[DetectorExecution] = []
    for name, detector in sorted(detectors.items()):
        detector_verdicts, detector_execution = execute_detector(detector, rollouts, cfg.detectors)
        verdicts.extend(detector_verdicts)
        detector_results.append(detector_execution)
        sub_config = getattr(cfg.detectors, name, None)
        if sub_config is not None:
            config_used[name] = sub_config.model_dump()

    occurrences: dict[str, list[SourceOccurrence]] = {}
    for rollout in rollouts:
        if rollout.provenance is None or rollout.occurrence_id is None:
            continue
        provenance_source = SourceOccurrence(
            occurrence_id=rollout.occurrence_id,
            source_path=rollout.provenance.source_path,
            line=rollout.provenance.line,
        )
        identifiers = {
            rollout.rollout_id,
            rollout.occurrence_id,
            rollout.group_id,
            *rollout.identity_aliases.values(),
            *rollout.provenance.legacy_ids.values(),
        }
        for identifier in identifiers:
            if identifier is not None:
                occurrences.setdefault(identifier, []).append(provenance_source)
    enriched_verdicts = []
    for verdict in verdicts:
        occurrence_sources: list[SourceOccurrence] = []
        seen_sources: set[tuple[object, object, object]] = set()
        referenced_ids = [*verdict.rollout_ids, *(span.rollout_id for span in verdict.evidence)]
        for identifier in referenced_ids:
            for source in occurrences.get(identifier, []):
                key = (source.occurrence_id, source.source_path, source.line)
                if key not in seen_sources:
                    seen_sources.add(key)
                    occurrence_sources.append(source)
        enriched_verdicts.append(
            Verdict.model_validate(
                {
                    **verdict.model_dump(mode="python"),
                    "source_occurrences": [
                        source.model_dump(mode="python") for source in occurrence_sources
                    ],
                }
            )
        )
    verdicts = enriched_verdicts

    for name, message in sorted(discovery_errors.items()):
        detector_results.append(
            DetectorExecution(
                detector=name,
                status="failed",
                errors=[message],
            )
        )
    findings = assemble_findings(
        verdicts,
        thresholds=cfg.severity,
        include_clean=include_clean,
        config_used=config_used,
        executions=detector_results,
    )
    aggregates = aggregate_rollouts(rollouts, cfg.aggregation)
    actual_run_ids = sorted({rollout.run_id for rollout in rollouts if rollout.run_id is not None})
    environment_namespaces = sorted(
        {
            rollout.environment_namespace
            for rollout in rollouts
            if rollout.environment_namespace is not None
        }
    )
    if not environment_namespaces and run_manifest.environment_namespace is not None:
        environment_namespaces = [run_manifest.environment_namespace]
    task_namespaces = sorted(
        {rollout.task_namespace for rollout in rollouts if rollout.task_namespace is not None}
    )
    if not task_namespaces and run_manifest.task_namespace is not None:
        task_namespaces = [run_manifest.task_namespace]

    ingestion_by_path = {item.path: item for item in diagnostics.files}
    input_files: list[InputFile] = []
    namespace_metadata = {
        key: value
        for key, value in (
            (
                "environment_namespace",
                environment_namespaces[0] if len(environment_namespaces) == 1 else None,
            ),
            ("task_namespace", task_namespaces[0] if len(task_namespaces) == 1 else None),
        )
        if value is not None
    }
    for manifest_source in run_manifest.files:
        source_name = _relative_source_name(manifest_source.path, run_manifest.root)
        ingestion = ingestion_by_path.get(source_name)
        if ingestion is not None and ingestion.sha256 is not None:
            input_files.append(
                InputFile(
                    name=source_name,
                    relative_path=source_name,
                    sha256=ingestion.sha256,
                    size_bytes=ingestion.size_bytes,
                    format=ingestion.format,
                    adapter_version=ingestion.adapter_version,
                    metadata=namespace_metadata,
                )
            )
        else:
            input_files.append(
                describe_input(
                    manifest_source.path,
                    relative_to=run_manifest.root,
                    format=(
                        manifest_source.format
                        if manifest_source.format != "unknown"
                        else run_manifest.format
                    ),
                    adapter_version=run_manifest.adapter_version,
                    metadata=namespace_metadata,
                )
            )
    input_files.extend(metadata_inputs)

    status = diagnostics.status
    errors: list[str] = []
    unavailable = [
        result for result in detector_results if result.status in {"failed", "unsupported"}
    ]
    if unavailable:
        status = "failed"
        for result in unavailable:
            detail = "; ".join(result.errors) or result.status
            errors.append(f"detector {result.detector}: {detail}")
    selected_names = sorted({*detectors, *discovery_errors})
    execution = ExecutionManifest(
        tool_version=__version__,
        schema_version=SCHEMA_VERSION,
        input_format=run_manifest.format,
        adapter_version=run_manifest.adapter_version,
        source_revision=_source_revision(),
        implementation_sha256=_implementation_digest(),
        effective_config=cfg.model_dump(mode="json"),
        selected_detectors=selected_names,
        detector_versions={item.detector: item.version for item in detector_results},
        run_ids=actual_run_ids or [run_manifest.run_id],
        environment_namespaces=environment_namespaces,
        task_namespaces=task_namespaces,
        ingestion=diagnostics,
        status=status,
        errors=errors,
    )

    return ReportData(
        tool_version=__version__,
        schema_version=SCHEMA_VERSION,
        input_files=input_files,
        aggregates=aggregates,
        findings=findings,
        execution=execution,
        detector_results=detector_results,
        verdicts=verdicts,
    )


def _relative_source_name(path: Path, root: Path | None) -> str:
    """Return the manifest-relative source name used by ingestion accounting."""
    if root is None:
        return path.name
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _override_legacy_run_id(manifest: RunManifest, run_id: str | None) -> RunManifest:
    """Apply an explicit namespace to raw legacy input without rewriting normalized IDs."""
    if run_id is None:
        return manifest
    if not run_id.strip():
        raise ValueError("--run-id must not be empty")
    if manifest.format == "normalized":
        raise ValueError("--run-id applies only to legacy raw input, not normalized input")
    return replace(manifest, run_id=run_id)


def _require_convertible_input(diagnostics: IngestionSummary, partial: str) -> None:
    """Reject empty, unsupported, failed, or policy-rejected partial conversion input."""
    status = diagnostics.status
    if status in {"empty", "unsupported", "failed"}:
        raise ValueError(
            f"conversion rejected: status={status}; accepted={diagnostics.accepted} "
            f"rejected={diagnostics.rejected} blank={diagnostics.blank}"
        )
    if status == "partial" and partial == "reject":
        raise ValueError(
            f"accepted={diagnostics.accepted} rejected={diagnostics.rejected}: "
            "partial input rejected by policy"
        )


def _cli_error_message(exc: OSError | ValueError) -> str:
    """Lead collision diagnostics with a stable phrase before long filesystem paths."""
    message = str(exc)
    if "conflicts with input" in message:
        return f"input/output collision: {message}"
    if "conflicts with output" in message:
        return f"output/output collision: {message}"
    return message


def exit_code_for(findings: list[Finding], fail_on: FailOn) -> int:
    """Map findings to a process exit code under a ``fail_on`` threshold.

    Returns 0 when ``fail_on`` is ``none``; otherwise 1 when any finding that
    actually fired (fired_count > 0) has severity at or above the threshold,
    else 0. Clean (info, zero-fired) findings never trip the code.
    """
    if fail_on is FailOn.none:
        return 0
    threshold = _SEVERITY_RANK[fail_on.value]
    for finding in findings:
        fired = finding.metrics.get("fired_count", 0.0) > 0.0
        if fired and _SEVERITY_RANK.get(finding.severity, 0) >= threshold:
            return 1
    return 0


def _configure_logging(verbose: bool) -> None:
    """Send library skip-and-log warnings to stderr; INFO too when verbose."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


def _verdict_chunks(report: ReportData) -> Iterable[bytes]:
    """Yield deterministic JSONL records for the complete retained verdict set."""
    for verdict in report.verdicts:
        yield orjson.dumps(verdict.model_dump(mode="json"), option=orjson.OPT_SORT_KEYS) + b"\n"


def _analysis_failed(report: ReportData, config: RunConfig) -> bool:
    """Return whether execution status requires an analysis-error process exit."""
    if report.execution is None:
        return False
    if report.execution.status in {"empty", "unsupported", "failed"}:
        return True
    return report.execution.status == "partial" and config.ingestion.partial == "reject"


def _source_revision() -> str | None:
    """Return the checkout revision when this source tree is a Git worktree."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parents[2],
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _implementation_digest() -> str:
    """Hash shipped source and template bytes, including dirty worktree edits."""
    source_root = Path(__file__).parent
    digest = hashlib.sha256()
    paths = sorted(
        [*source_root.rglob("*.py"), *source_root.joinpath("report", "templates").rglob("*.j2")]
    )
    for source in paths:
        digest.update(str(source.relative_to(source_root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Offline rollout and reward-hacking debugger for the verifiers / prime-rl ecosystem.",
)
detectors_app = typer.Typer(no_args_is_help=True, help="Inspect the detector registry.")
schema_app = typer.Typer(no_args_is_help=True, help="Work with the normalized rollout schema.")
app.add_typer(detectors_app, name="detectors")
app.add_typer(schema_app, name="schema")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Print the rolloutscope version and exit.",
    ),
) -> None:
    """rolloutscope: analyze rollout artifacts for reward hacking, offline."""


@app.command()
def analyze(
    path: Path = typer.Argument(
        ...,
        help="Run directory or JSONL file of rollout artifacts to analyze.",
    ),
    out: Path | None = typer.Option(
        None, "--out", "-o", help="Write the self-contained HTML report to this path."
    ),
    json_out: Path | None = typer.Option(
        None, "--json", "-j", help="Write the deterministic JSON findings sidecar to this path."
    ),
    verdicts: Path | None = typer.Option(
        None, "--verdicts", help="Write complete deterministic verdict JSONL to this path."
    ),
    config: Path | None = typer.Option(
        None, "--config", "-c", help="TOML config file overriding detector and severity defaults."
    ),
    detector: list[str] | None = typer.Option(
        None, "--detector", "-d", help="Run only this detector (repeatable); default runs all."
    ),
    include_clean: bool = typer.Option(
        False, "--include-clean", help="Emit info findings for detectors that ran but did not fire."
    ),
    partial: Literal["allow", "reject"] | None = typer.Option(
        None, "--partial", help="Override the configured partial-ingestion policy."
    ),
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help="Override path-derived run identity for legacy raw input.",
    ),
    fail_on: FailOn = typer.Option(
        FailOn.none,
        "--fail-on",
        case_sensitive=False,
        help="Exit non-zero when a fired finding reaches this severity (default: never).",
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Suppress the terminal summary (still writes files)."
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Log skipped rows and progress to stderr."
    ),
) -> None:
    """Analyze an artifact path: run every detector and emit reports.

    Resolves the adapter for ``path``, streams and normalizes its rollouts, runs
    the discovered detectors, aggregates the run, and renders a terminal
    summary plus optional JSON and HTML reports. Exit code follows ``--fail-on``.
    """
    _configure_logging(verbose)
    run_config = load_run_config(config)
    if partial is not None:
        run_config.ingestion.partial = partial
    selected = set(detector) if detector else None
    try:
        adapter = resolve_adapter(path)
        manifest = _override_legacy_run_id(adapter.load_run(path), run_id)
        primary = out or json_out or verdicts
        verdict_path = verdicts
        ledger_path: Path | None = None
        if primary is not None:
            verdict_path = verdict_path or primary.with_name(primary.name + ".verdicts.jsonl")
            ledger_path = primary.with_name(primary.name + ".manifest.json")
        destinations = [
            item for item in (out, json_out, verdict_path, ledger_path) if item is not None
        ]
        validate_output_paths(
            destinations,
            [*(source.path for source in manifest.files), *manifest.metadata_sources],
        )
        report = build_report_data(
            path,
            config=run_config,
            selected=selected,
            include_clean=include_clean,
            adapter=adapter,
            manifest=manifest,
        )
    except (OSError, ValueError) as exc:  # source failure, unknown detector, or bad artifact
        raise typer.BadParameter(_cli_error_message(exc)) from exc

    console = Console()
    if not quiet:
        render_terminal(report, console)
    if _analysis_failed(report, run_config):
        raise typer.Exit(code=2)
    if primary is not None and report.execution is not None:
        artifacts = []
        if out is not None:
            artifacts.append(OutputArtifact(name=out.name, role="HTML report"))
        if json_out is not None:
            artifacts.append(OutputArtifact(name=json_out.name, role="JSON report"))
        assert verdict_path is not None and ledger_path is not None
        artifacts.append(OutputArtifact(name=verdict_path.name, role="complete verdict JSONL"))
        report.execution.artifacts = artifacts
        report.execution.artifact_manifest = ledger_path.name
        inputs = [*(source.path for source in manifest.files), *manifest.metadata_sources]
        try:
            with OutputTransaction(inputs) as tx:
                staged: list[tuple[Path, str, str, int]] = []
                if out is not None:
                    digest, size = tx.stage(out, [render_html(report).encode("utf-8")])
                    staged.append((out, "HTML report", digest, size))
                if json_out is not None:
                    digest, size = tx.stage(json_out, [render_json_bytes(report)])
                    staged.append((json_out, "JSON report", digest, size))
                digest, size = tx.stage(verdict_path, _verdict_chunks(report))
                staged.append((verdict_path, "complete verdict JSONL", digest, size))
                ledger = {
                    "manifest_version": "1.0",
                    "artifacts": [
                        {
                            "name": item.name,
                            "role": role,
                            "sha256": digest,
                            "size_bytes": size,
                        }
                        for item, role, digest, size in staged
                    ],
                }
                tx.stage(
                    ledger_path,
                    [
                        orjson.dumps(
                            ledger,
                            option=(
                                orjson.OPT_SORT_KEYS
                                | orjson.OPT_INDENT_2
                                | orjson.OPT_APPEND_NEWLINE
                            ),
                        )
                    ],
                )
        except Exception as exc:
            raise typer.BadParameter(f"could not write analysis outputs: {exc}") from exc
        if json_out is not None:
            typer.echo(f"wrote JSON findings to {json_out}")
        if out is not None:
            typer.echo(f"wrote HTML report to {out}")
        typer.echo(f"wrote complete verdicts to {verdict_path}")

    raise typer.Exit(code=exit_code_for(report.findings, fail_on))


@app.command()
def convert(
    path: Path = typer.Argument(..., help="Run directory or JSONL file of rollout artifacts."),
    out: Path = typer.Option(
        ..., "--out", "-o", help="Write normalized rollout JSONL to this path."
    ),
    partial: Literal["allow", "reject"] = typer.Option(
        "allow",
        "--partial",
        help="Allow or reject a mix of accepted and rejected source records.",
    ),
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help="Override path-derived run identity for legacy raw input.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Log skipped rows and progress to stderr."
    ),
) -> None:
    """Normalize on-disk artifacts into schema JSONL, streaming row by row.

    Resolves the adapter for ``path`` and writes one normalized rollout per line
    to ``out`` (content-derived ids attached, unknown keys preserved). Streams,
    so it is safe on files larger than RAM.
    """
    _configure_logging(verbose)
    try:
        adapter = resolve_adapter(path)
        manifest = _override_legacy_run_id(adapter.load_run(path), run_id)
        inputs = [*(source.path for source in manifest.files), *manifest.metadata_sources]
        validate_output_paths([out], inputs)
        diagnostics = IngestionSummary()

        def rows() -> Iterator[SingleTurnRollout | MultiTurnRollout]:
            yield from adapter.load_manifest(manifest, diagnostics=diagnostics)
            _require_convertible_input(diagnostics, partial)

        count = write_rollouts(out, rows(), source_paths=inputs)
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(_cli_error_message(exc)) from exc
    typer.echo(f"wrote {count} normalized rollouts to {out}")


@detectors_app.command("list")
def detectors_list() -> None:
    """List the discovered detectors, their categories, and their source group."""
    detectors = discover_detectors()
    table = Table(title="rolloutscope detectors")
    table.add_column("name", style="bold")
    table.add_column("category")
    for name in sorted(detectors):
        table.add_row(name, detectors[name].category)
    Console().print(table)
    typer.echo(f"{len(detectors)} detector(s) discovered")


@schema_app.command("export")
def schema_export(
    out: Path | None = typer.Option(
        None, "--out", "-o", help="Write the JSON Schema here; default prints to stdout."
    ),
) -> None:
    """Export the normalized Rollout union as JSON Schema (deterministic bytes)."""
    payload = orjson.dumps(rollout_json_schema(), option=_JSON_OPTIONS)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(payload)
        typer.echo(f"wrote JSON Schema to {out}")
    else:
        typer.echo(payload.decode("utf-8"), nl=False)
