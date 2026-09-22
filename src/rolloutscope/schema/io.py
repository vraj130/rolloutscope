"""Streaming JSONL IO for normalized rollouts.

Golden rule 6: stream, never bulk-load. The reader is a generator that decodes and
validates one line at a time with orjson, applies the migration chain, and on any
bad row logs (path, line number, reason) and continues. One malformed row never
sinks a file. Assume files larger than RAM.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import orjson
from pydantic import ValidationError

from rolloutscope.output import OutputTransaction
from rolloutscope.schema.execution import FileIngestion
from rolloutscope.schema.ids import attach_identity, run_id_from_path
from rolloutscope.schema.migrate import UnsupportedSchemaVersionError, migrate_row
from rolloutscope.schema.models import (
    SCHEMA_VERSION,
    MultiTurnRollout,
    SingleTurnRollout,
    validate_rollout,
)

logger = logging.getLogger(__name__)


def iter_jsonl(
    path: Path, *, diagnostics: FileIngestion | None = None
) -> Iterator[tuple[int, Any]]:
    """Yield (line_number, decoded_value) pairs from a JSONL file, lazily.

    Line numbers are 1-based. Blank lines are tolerated and skipped silently;
    lines that fail JSON decoding are skipped with a logged warning carrying the
    line number and reason.
    """
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for lineno, line in enumerate(handle, start=1):
                digest.update(line)
                if diagnostics is not None:
                    diagnostics.observed += 1
                    diagnostics.size_bytes += len(line)
                stripped = line.strip()
                if not stripped:
                    if diagnostics is not None:
                        diagnostics.blank += 1
                    continue
                try:
                    yield lineno, orjson.loads(stripped)
                except orjson.JSONDecodeError as exc:
                    if diagnostics is not None:
                        diagnostics.reject(lineno, "invalid_json", str(exc))
                    logger.warning("skipping %s line %d: invalid JSON (%s)", path, lineno, exc)
    except OSError as exc:
        if diagnostics is not None:
            diagnostics.error = str(exc)
        logger.error("could not read %s: %s", path, exc)
        raise
    finally:
        if diagnostics is not None:
            diagnostics.sha256 = digest.hexdigest()


def read_rollouts(
    path: Path,
    *,
    diagnostics: FileIngestion | None = None,
    source_name: str | None = None,
    run_id: str | None = None,
) -> Iterator[SingleTurnRollout | MultiTurnRollout]:
    """Stream validated rollouts from a normalized JSONL file.

    Each row is migrated forward through the schema migration chain, then
    validated into the right Rollout variant. Rows that are not JSON objects,
    fail migration, or fail validation are skipped with a logged warning (line
    number plus reason), never raised.
    """
    for lineno, raw in iter_jsonl(path, diagnostics=diagnostics):
        if not isinstance(raw, dict):
            if diagnostics is not None:
                diagnostics.reject(lineno, "not_object", "not a JSON object")
            logger.warning("skipping %s line %d: not a JSON object", path, lineno)
            continue
        try:
            row = validate_rollout(migrate_row(raw))
            row = attach_identity(
                row,
                run_id=run_id or run_id_from_path(path),
                source_path=source_name or path.name,
                line=lineno,
                adapter="normalized",
                step_index=None,
                preserve=True,
            )
            if diagnostics is not None:
                diagnostics.accepted += 1
            yield row
        except (ValidationError, ValueError) as exc:
            if diagnostics is not None:
                reason = (
                    "unsupported_schema"
                    if isinstance(exc, UnsupportedSchemaVersionError)
                    else "validation_error"
                )
                diagnostics.reject(lineno, reason, validation_message(exc))
            logger.warning("skipping %s line %d: %s", path, lineno, exc)


def write_rollouts(
    path: Path,
    rollouts: Iterable[SingleTurnRollout | MultiTurnRollout],
    *,
    source_paths: Iterable[Path] = (),
) -> int:
    """Write rollouts to a JSONL file, one line per row, streaming.

    Rows are dumped with mode="json" so nested models and extras serialize to
    plain JSON types. Returns the number of rows written. Parent directories are
    created if missing.
    """
    count = 0

    def chunks() -> Iterator[bytes]:
        nonlocal count
        for rollout in rollouts:
            row = validate_rollout(rollout.model_dump(mode="python"))
            payload = row.model_dump(mode="json")
            payload["schema_version"] = SCHEMA_VERSION
            yield orjson.dumps(payload, option=orjson.OPT_APPEND_NEWLINE)
            count += 1

    with OutputTransaction(source_paths) as transaction:
        transaction.stage(path, chunks())
    return count


def validation_message(exc: ValueError) -> str:
    """Describe validation fields without embedding rejected source content in reports."""
    if isinstance(exc, ValidationError):
        return "; ".join(
            f"{'.'.join(map(str, error['loc']))}: {error['msg']}"
            for error in exc.errors(include_input=False, include_url=False)[:5]
        )
    return str(exc)[:400]
