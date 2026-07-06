"""Adapter contract and shared plumbing for on-disk rollout artifacts.

Adapters turn artifacts written by verifiers (results.jsonl plus metadata.json)
and prime-rl (train_rollouts.jsonl under per-step directories) into normalized
Rollout rows. Core rule 3 applies: nothing here imports verifiers or prime-rl;
everything is parsed from disk per the pinned references in the
verifiers-ground-truth skill (verifiers @ 5885ab9c, prime-rl @ df2acf48).

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

    def load(self, path: Path) -> Iterator[SingleTurnRollout | MultiTurnRollout]:
        """Stream normalized rollouts from the run at path."""
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

    def load(self, path: Path) -> Iterator[SingleTurnRollout | MultiTurnRollout]:
        """Stream normalized rollouts from every file of the run at path.

        Input: any path accepted by ``load_run``. Files are read in manifest
        order; each row gets the manifest's run_id and its file's step_index.
        Bad rows are skipped and logged, never raised (golden rule 6).
        """
        manifest = self.load_run(path)
        for source in manifest.files:
            yield from iter_normalized_rows(
                source.path, run_id=manifest.run_id, step_index=source.step_index
            )


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


def normalize_row(
    raw: dict[str, Any], *, run_id: str, step_index: int | None
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
    identity = {
        "rollout_id": rollout_id(
            raw["example_id"], raw.get("prompt"), raw.get("completion"), raw["reward"]
        ),
        "group_id": group_id(raw["example_id"]),
        "run_id": run_id,
        "step_index": step_index,
    }
    return validate_rollout({**raw, **identity})


def iter_normalized_rows(
    path: Path, *, run_id: str, step_index: int | None
) -> Iterator[SingleTurnRollout | MultiTurnRollout]:
    """Stream normalized rollouts from one on-disk JSONL file.

    Inputs: the file path, the run id, and the layout-derived step index (None
    in snapshot mode). Builds on schema.io.iter_jsonl (line-by-line orjson);
    rows that are not JSON objects, miss required fields, or fail validation are
    skipped with a logged warning carrying path, line number, and reason. One
    malformed row never sinks a file (golden rule 6).
    """
    for lineno, raw in iter_jsonl(path):
        if not isinstance(raw, dict):
            logger.warning("skipping %s line %d: not a JSON object", path, lineno)
            continue
        try:
            yield normalize_row(raw, run_id=run_id, step_index=step_index)
        except KeyError as exc:
            logger.warning("skipping %s line %d: missing required field %s", path, lineno, exc)
        except (TypeError, ValidationError, ValueError) as exc:
            logger.warning("skipping %s line %d: %s", path, lineno, exc)
