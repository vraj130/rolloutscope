"""Adapter contract and shared plumbing for on-disk rollout artifacts.

Adapters turn artifacts written by verifiers (results.jsonl plus metadata.json)
and prime-rl (train_rollouts.jsonl under per-step directories) into normalized
Rollout rows. Core rule 3 applies: nothing here imports verifiers or prime-rl;
everything is parsed from disk per the pinned references in the
pinned upstream commits (verifiers @ 5885ab9c, prime-rl @ df2acf48).

This module owns three things and nothing else owns them:
- the Adapter protocol and the RunManifest that load_run() returns,
- the shared row-normalization path (raw trace dict to validated Rollout, with
  content-derived ids computed from the raw on-disk values before validation),
- the prime-rl step-directory name mapping (step_index_from_name), encoded here
  and nowhere else.
"""

from __future__ import annotations

import abc
import logging
import re
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import orjson
from pydantic import ValidationError

from rolloutscope.schema import (
    MultiTurnRollout,
    SingleTurnRollout,
    group_id,
    iter_jsonl,
    rollout_id,
    validate_rollout,
)
from rolloutscope.schema.execution import FileIngestion, IngestionSummary
from rolloutscope.schema.ids import attach_identity
from rolloutscope.schema.io import validation_message
from rolloutscope.schema.migrate import UnsupportedSchemaVersionError, migrate_row

logger = logging.getLogger(__name__)

RESULTS_FILENAME = "results.jsonl"
METADATA_FILENAME = "metadata.json"
TRAIN_ROLLOUTS_FILENAME = "train_rollouts.jsonl"

# TODO(open question for the orchestrator): the on-disk-format reference pins
# `step_path / "train_rollouts.jsonl"` (prime-rl @ df2acf48) but does NOT pin the
# naming convention of step_path itself. Exact question: what is the precise step
# directory name format the prime-rl orchestrator uses at pin df2acf48 (for
# example `step_{n}`, a bare integer, or something else)? Until that is pinned,
# only `step_<int>` (case-insensitive) and bare-integer directory names carry a
# step_index; every other name yields None (snapshot mode), never a guess.
_STEP_DIR_RE = re.compile(r"step_(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class SourceFile:
    """One rollout JSONL file discovered inside a run.

    ``step_index`` is the training step the file belongs to when the on-disk
    layout provides one (prime-rl step directories); None means snapshot mode.
    """

    path: Path
    step_index: int | None = None
    format: str = "unknown"


@dataclass(frozen=True)
class RunManifest:
    """Adapter-level description of a discovered run (not part of the schema).

    Carries the derived ``run_id``, the source files in load order (each with
    its optional layout-derived ``step_index``), and run-level fields parsed
    from metadata.json when that manifest is present (empty dict otherwise).
    """

    run_id: str
    files: tuple[SourceFile, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    root: Path | None = None
    format: str = "unknown"
    adapter_version: str = "1"
    metadata_sources: tuple[Path, ...] = ()
    environment_namespace: str | None = None
    task_namespace: str | None = None


@runtime_checkable
class Adapter(Protocol):
    """Protocol every artifact adapter satisfies.

    ``detect`` answers whether the adapter can handle a file or directory,
    ``load_run`` discovers and orders the run's source files, and ``load``
    streams normalized rollouts from them.
    """

    name: str

    def detect(self, path: Path) -> bool:
        """Return True when this adapter can handle the file or directory at path."""
        ...

    def load_run(self, path: Path) -> RunManifest:
        """Discover the run at path: ordered source files, step indices, run id."""
        ...

    def load(
        self, path: Path, *, diagnostics: IngestionSummary | None = None
    ) -> Iterator[SingleTurnRollout | MultiTurnRollout]:
        """Stream normalized rollouts from the run at path."""
        ...

    def load_manifest(
        self, manifest: RunManifest, *, diagnostics: IngestionSummary | None = None
    ) -> Iterator[SingleTurnRollout | MultiTurnRollout]:
        """Stream the files already discovered in manifest, with optional accounting."""
        ...


class BaseAdapter(abc.ABC):
    """Shared adapter skeleton: concrete adapters implement discovery, loading
    is the same streaming pass over the manifest for everyone."""

    name: str = "base"

    @abc.abstractmethod
    def detect(self, path: Path) -> bool:
        """Return True when this adapter can handle the file or directory at path."""

    @abc.abstractmethod
    def load_run(self, path: Path) -> RunManifest:
        """Discover the run at path: ordered source files, step indices, run id."""

    def load(
        self, path: Path, *, diagnostics: IngestionSummary | None = None
    ) -> Iterator[SingleTurnRollout | MultiTurnRollout]:
        """Stream normalized rollouts from every file of the run at path.

        Input: any path accepted by ``load_run``. Files are read in manifest
        order; each row gets the manifest's run_id and its file's step_index.
        Bad rows are skipped and logged, never raised (golden rule 6).
        """
        yield from self.load_manifest(self.load_run(path), diagnostics=diagnostics)

    def load_manifest(
        self, manifest: RunManifest, *, diagnostics: IngestionSummary | None = None
    ) -> Iterator[SingleTurnRollout | MultiTurnRollout]:
        """Load a discovered manifest once, retaining duplicates with bounded memory.

        Duplicate accounting uses a temporary on-disk SQLite index. Normalized
        rows keep their recorded identity even when mixed with raw upstream rows.
        """
        index = sqlite3.connect("") if diagnostics is not None else None
        try:
            if index is not None:
                index.execute("PRAGMA cache_size=-1024")
                index.execute("CREATE TABLE occurrences (id TEXT PRIMARY KEY)")
            discovered: list[tuple[SourceFile, FileIngestion]] = []
            for source in manifest.files:
                source_name = (
                    source.path.relative_to(manifest.root).as_posix()
                    if manifest.root is not None
                    else source.path.name
                )
                accounting = FileIngestion(
                    path=source_name,
                    format=source.format if source.format != "unknown" else manifest.format,
                    adapter_version=manifest.adapter_version,
                    error="not read: ingestion has not reached this source",
                )
                discovered.append((source, accounting))
                if diagnostics is not None:
                    diagnostics.files.append(accounting)
            for source, accounting in discovered:
                accounting.error = None
                try:
                    for row in iter_normalized_rows(
                        source.path,
                        run_id=manifest.run_id,
                        step_index=source.step_index,
                        source_name=accounting.path,
                        adapter="verifiers_eval" if self.name == "normalized" else self.name,
                        environment_namespace=manifest.environment_namespace,
                        task_namespace=manifest.task_namespace,
                        diagnostics=accounting,
                    ):
                        if index is not None:
                            result = index.execute(
                                "INSERT OR IGNORE INTO occurrences VALUES (?)", (row.occurrence_id,)
                            )
                            if result.rowcount == 0:
                                accounting.duplicates += 1
                        yield row
                except OSError as exc:
                    accounting.error = str(exc)
                    logger.warning("could not read %s: %s", source.path, exc)
                    raise
        finally:
            if index is not None:
                index.close()


def step_index_from_name(name: str) -> int | None:
    """Map a step-directory name to its step_index, or None when it carries none.

    Input: a bare directory name (not a path). Recognized forms are
    ``step_<int>`` (case-insensitive) and a bare decimal integer, per the pinned
    prime-rl layout note; any other name returns None, which means snapshot mode
    (no step ordering). See the TODO on ``_STEP_DIR_RE`` for the open question
    on the exact upstream naming.
    """
    match = _STEP_DIR_RE.fullmatch(name)
    if match:
        return int(match.group(1))
    if name.isdecimal():
        return int(name)
    return None


def read_run_metadata(directory: Path) -> dict[str, Any] | None:
    """Load metadata.json from a run directory when present and valid.

    Input: the directory to look in. Returns the parsed dict, or None when the
    file is absent, is invalid JSON, or is not a JSON object (the latter two are
    logged as warnings, never raised).
    """
    path = directory / METADATA_FILENAME
    if not path.is_file():
        return None
    try:
        parsed = orjson.loads(path.read_bytes())
    except orjson.JSONDecodeError as exc:
        logger.warning("ignoring %s: invalid JSON (%s)", path, exc)
        return None
    if not isinstance(parsed, dict):
        logger.warning("ignoring %s: not a JSON object", path)
        return None
    return parsed


def metadata_namespaces(metadata: dict[str, Any] | None) -> tuple[str | None, str | None]:
    """Extract stable environment/task namespaces from pinned run metadata.

    ``environment_namespace`` is an explicit RolloutScope override; otherwise
    the pinned verifiers ``GenerateMetadata.env_id`` field is used. A task
    namespace is accepted only from the explicit ``task_namespace`` field.
    Arbitrary row ``task``/``info`` values and mutable run summaries are never
    interpreted as namespaces.
    """

    def _string(key: str) -> str | None:
        value = metadata.get(key) if metadata is not None else None
        if not isinstance(value, str):
            return None
        normalized = value.strip()
        return normalized or None

    environment = _string("environment_namespace") or _string("env_id")
    return environment, _string("task_namespace")


def normalize_row(
    raw: dict[str, Any],
    *,
    run_id: str,
    step_index: int | None,
    source_name: str = "unknown",
    line: int = 1,
    adapter: str = "raw",
    environment_namespace: str | None = None,
    task_namespace: str | None = None,
) -> SingleTurnRollout | MultiTurnRollout:
    """Normalize one raw on-disk trace row into a validated Rollout.

    Inputs: the JSON-decoded row dict exactly as read from disk, the run id, and
    the layout-derived step index (None in snapshot mode). The identity fields
    (rollout_id, group_id) are computed from the raw values BEFORE validation so
    ids are content-derived from disk bytes; unknown keys pass through via the
    schema's extra="allow". Raises KeyError on missing required fields and
    TypeError / ValueError / pydantic.ValidationError on rows that do not fit
    the contract; streaming callers catch and skip.
    """
    if "schema_version" in raw:
        row = validate_rollout(migrate_row(raw))
        return attach_identity(
            row,
            run_id=run_id,
            source_path=source_name,
            line=line,
            adapter="normalized",
            step_index=row.step_index,
            preserve=True,
            environment_namespace=environment_namespace,
            task_namespace=task_namespace,
        )
    identity = {
        "rollout_id": rollout_id(
            raw["example_id"], raw.get("prompt"), raw.get("completion"), raw["reward"]
        ),
        "group_id": group_id(raw["example_id"]),
        "run_id": run_id,
        "step_index": step_index,
    }
    return attach_identity(
        validate_rollout({**raw, **identity}),
        run_id=run_id,
        source_path=source_name,
        line=line,
        adapter=adapter,
        step_index=step_index,
        environment_namespace=environment_namespace,
        task_namespace=task_namespace,
    )


def iter_normalized_rows(
    path: Path,
    *,
    run_id: str,
    step_index: int | None,
    source_name: str | None = None,
    adapter: str = "raw",
    environment_namespace: str | None = None,
    task_namespace: str | None = None,
    diagnostics: FileIngestion | None = None,
) -> Iterator[SingleTurnRollout | MultiTurnRollout]:
    """Stream normalized rollouts from one on-disk JSONL file.

    Inputs: the file path, the run id, and the layout-derived step index (None
    in snapshot mode). Builds on schema.io.iter_jsonl (line-by-line orjson);
    rows that are not JSON objects, miss required fields, or fail validation are
    skipped with a logged warning carrying path, line number, and reason. One
    malformed row never sinks a file (golden rule 6).
    """
    seen_formats: set[str] = set()
    for lineno, raw in iter_jsonl(path, diagnostics=diagnostics):
        if not isinstance(raw, dict):
            logger.warning("skipping %s line %d: not a JSON object", path, lineno)
            if diagnostics is not None:
                diagnostics.reject(lineno, "not_object", "not a JSON object")
            continue
        seen_formats.add("normalized" if "schema_version" in raw else adapter)
        if diagnostics is not None:
            diagnostics.format = next(iter(seen_formats)) if len(seen_formats) == 1 else "mixed"
        try:
            row = normalize_row(
                raw,
                run_id=run_id,
                step_index=step_index,
                source_name=source_name or path.name,
                line=lineno,
                adapter=adapter,
                environment_namespace=environment_namespace,
                task_namespace=task_namespace,
            )
        except UnsupportedSchemaVersionError as exc:
            logger.warning("skipping %s line %d: %s", path, lineno, exc)
            if diagnostics is not None:
                diagnostics.reject(lineno, "unsupported_schema", str(exc))
        except KeyError as exc:
            logger.warning("skipping %s line %d: missing required field %s", path, lineno, exc)
            if diagnostics is not None:
                diagnostics.reject(lineno, "missing_field", f"missing required field {exc}")
        except (TypeError, ValidationError, ValueError) as exc:
            logger.warning("skipping %s line %d: %s", path, lineno, exc)
            if diagnostics is not None:
                message = validation_message(exc) if isinstance(exc, ValueError) else str(exc)
                diagnostics.reject(lineno, "validation_error", message)
        else:
            if diagnostics is not None:
                diagnostics.accepted += 1
            yield row
