"""Command-line interface for rolloutscope.

Thin typer wrapper over the library packages (scaffold principle: all real work
lives in importable functions). The wiring functions ``load_run_config`` and
``build_report_data`` are plain, testable functions; the typer commands only
parse arguments, call them, print, and map findings to an exit code. This is the
one module allowed to import adapters, detectors, analysis, and report together.
"""

from __future__ import annotations

import enum
import logging
import tomllib
from pathlib import Path

import orjson
import typer
from pydantic import BaseModel, ValidationError
from rich.console import Console
from rich.table import Table

from rolloutscope import __version__
from rolloutscope.adapters import resolve_adapter
from rolloutscope.analysis import (
    AggregateConfig,
    SeverityThresholds,
    aggregate_rollouts,
    assemble_findings,
)
from rolloutscope.detectors import DetectorConfig, discover_detectors
from rolloutscope.report import (
    ReportData,
    describe_input,
    render_terminal,
    write_html,
    write_json,
)
from rolloutscope.schema import SCHEMA_VERSION, Finding, rollout_json_schema, write_rollouts

logger = logging.getLogger(__name__)

_JSON_OPTIONS = orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE
_SEVERITY_RANK: dict[str, int] = {"info": 0, "warning": 1, "critical": 2}


class RunConfig(BaseModel):
    """Full run configuration, loadable from a TOML file.

    ``detectors`` carries the per-detector thresholds, ``aggregation`` the
    histogram and top-k settings, and ``severity`` the score-to-severity
    mapping. Every field defaults to its model's conservative heuristic
    defaults, so an omitted section (or no config file at all) is valid.
    """

    detectors: DetectorConfig = DetectorConfig()
    aggregation: AggregateConfig = AggregateConfig()
    severity: SeverityThresholds = SeverityThresholds()


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
    adapter = resolve_adapter(path)
    manifest = adapter.load_run(path)
    input_files = [describe_input(source.path) for source in manifest.files]

    rollouts = list(adapter.load(path))

    detectors = discover_detectors()
    if selected is not None:
        unknown = selected - set(detectors)
        if unknown:
            available = ", ".join(sorted(detectors))
            raise ValueError(
                f"unknown detector(s): {', '.join(sorted(unknown))}. Available: {available}"
            )
        detectors = {name: det for name, det in detectors.items() if name in selected}

    verdicts = []
    config_used: dict[str, dict[str, object]] = {}
    for name, detector in sorted(detectors.items()):
        verdicts.extend(detector.detect(rollouts, cfg.detectors))
        sub_config = getattr(cfg.detectors, name, None)
        if sub_config is not None:
            config_used[name] = sub_config.model_dump()

    findings = assemble_findings(
        verdicts,
        thresholds=cfg.severity,
        include_clean=include_clean,
        config_used=config_used,
    )
    aggregates = aggregate_rollouts(rollouts, cfg.aggregation)

    return ReportData(
        tool_version=__version__,
        schema_version=SCHEMA_VERSION,
        input_files=input_files,
        aggregates=aggregates,
        findings=findings,
    )


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
    config: Path | None = typer.Option(
        None, "--config", "-c", help="TOML config file overriding detector and severity defaults."
    ),
    detector: list[str] | None = typer.Option(
        None, "--detector", "-d", help="Run only this detector (repeatable); default runs all."
    ),
    include_clean: bool = typer.Option(
        False, "--include-clean", help="Emit info findings for detectors that ran but did not fire."
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
    selected = set(detector) if detector else None
    try:
        report = build_report_data(
            path, config=run_config, selected=selected, include_clean=include_clean
        )
    except ValueError as exc:  # unknown detector name, or unrecognized artifact path
        raise typer.BadParameter(str(exc)) from exc

    console = Console()
    if not quiet:
        render_terminal(report, console)
    if json_out is not None:
        write_json(report, json_out)
        typer.echo(f"wrote JSON findings to {json_out}")
    if out is not None:
        write_html(report, out)
        typer.echo(f"wrote HTML report to {out}")

    raise typer.Exit(code=exit_code_for(report.findings, fail_on))


@app.command()
def convert(
    path: Path = typer.Argument(..., help="Run directory or JSONL file of rollout artifacts."),
    out: Path = typer.Option(
        ..., "--out", "-o", help="Write normalized rollout JSONL to this path."
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
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    count = write_rollouts(out, adapter.load(path))
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
