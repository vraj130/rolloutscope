"""Streaming JSONL IO for normalized rollouts.

Golden rule 6: stream, never bulk-load. The reader is a generator that decodes and
validates one line at a time with orjson, applies the migration chain, and on any
bad row logs (path, line number, reason) and continues. One malformed row never
sinks a file. Assume files larger than RAM.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import orjson
from pydantic import ValidationError

from rolloutscope.schema.migrate import migrate_row
from rolloutscope.schema.models import MultiTurnRollout, SingleTurnRollout, validate_rollout

logger = logging.getLogger(__name__)


def iter_jsonl(path: Path) -> Iterator[tuple[int, Any]]:
    """Yield (line_number, decoded_value) pairs from a JSONL file, lazily.

    Line numbers are 1-based. Blank lines are tolerated and skipped silently;
    lines that fail JSON decoding are skipped with a logged warning carrying the
    line number and reason.
    """
    with path.open("rb") as handle:
        for lineno, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield lineno, orjson.loads(stripped)
            except orjson.JSONDecodeError as exc:
                logger.warning("skipping %s line %d: invalid JSON (%s)", path, lineno, exc)


def read_rollouts(path: Path) -> Iterator[SingleTurnRollout | MultiTurnRollout]:
    """Stream validated rollouts from a normalized JSONL file.

    Each row is migrated forward through the schema migration chain, then
    validated into the right Rollout variant. Rows that are not JSON objects,
    fail migration, or fail validation are skipped with a logged warning (line
    number plus reason), never raised.
    """
    for lineno, raw in iter_jsonl(path):
        if not isinstance(raw, dict):
            logger.warning("skipping %s line %d: not a JSON object", path, lineno)
            continue
        try:
            yield validate_rollout(migrate_row(raw))
        except (ValidationError, ValueError) as exc:
            logger.warning("skipping %s line %d: %s", path, lineno, exc)


def write_rollouts(path: Path, rollouts: Iterable[SingleTurnRollout | MultiTurnRollout]) -> int:
    """Write rollouts to a JSONL file, one line per row, streaming.

    Rows are dumped with mode="json" so nested models and extras serialize to
    plain JSON types. Returns the number of rows written. Parent directories are
    created if missing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("wb") as handle:
        for rollout in rollouts:
            handle.write(
                orjson.dumps(rollout.model_dump(mode="json"), option=orjson.OPT_APPEND_NEWLINE)
            )
            count += 1
    return count
