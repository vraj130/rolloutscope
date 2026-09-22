"""Stable, content-derived identifiers for rollouts, groups, and runs.

ID scheme:
- ``rollout_id``: sha256 over a canonical (sorted-key) orjson serialization of
  (example_id, prompt, completion, reward), truncated to 16 hex chars. The same
  row content always yields the same id, no matter who serialized it.
- ``group_id``: derived from the grouping key ``example_id`` as ``grp-{example_id}``.
- ``run_id``: hash of the resolved source run path for raw legacy inputs;
  normalized inputs preserve their existing run identity.
- ``step_index``: an optional integer attached by adapters from on-disk step
  layout only, never guessed; it is not computed here.

Schema 2 separates occurrence identity from generation content and scoring.
New sidecars join on occurrence_id; legacy rollout IDs remain available as
aliases. No tensors or source mutations are needed to establish these IDs.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import orjson

from rolloutscope.schema.models import Rollout, SourceProvenance, validate_rollout

ROLLOUT_ID_HEX_CHARS = 16
RUN_ID_HEX_CHARS = 12


def canonical_bytes(obj: Any) -> bytes:
    """Serialize a JSON-compatible object to canonical bytes (sorted keys).

    Input: any orjson-serializable object. Output: deterministic bytes suitable
    for hashing. Two structurally equal objects always produce equal bytes.
    """
    return orjson.dumps(obj, option=orjson.OPT_SORT_KEYS)


def rollout_id(example_id: int, prompt: Any, completion: Any, reward: float) -> str:
    """Compute the content-derived rollout id.

    Inputs are the raw JSON-compatible values of the canonical fields (prompt and
    completion as message lists, strings, or None). ``reward`` is normalized to
    float so an integer-encoded reward hashes identically. Output: a 16-hex-char
    id prefixed with ``r``.
    """
    payload = {
        "example_id": example_id,
        "prompt": prompt,
        "completion": completion,
        "reward": float(reward),
    }
    digest = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    return f"r{digest[:ROLLOUT_ID_HEX_CHARS]}"


def group_id(example_id: int) -> str:
    """Derive the group id from the grouping key ``example_id``."""
    return f"grp-{example_id}"


def run_id_from_manifest(manifest: dict[str, Any]) -> str:
    """Derive the run id from a parsed run manifest (metadata.json contents)."""
    digest = hashlib.sha256(canonical_bytes(manifest)).hexdigest()
    return f"run-{digest[:RUN_ID_HEX_CHARS]}"


def run_id_from_name(name: str) -> str:
    """Derive the run id from a run directory name, for runs with no manifest."""
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
    return f"run-{digest[:RUN_ID_HEX_CHARS]}"


def run_id_from_path(path: Path) -> str:
    """Identify a legacy run by canonical location, independent of mutable summaries.

    Stable across append and rescoring at that location. Exported normalized
    records retain it when moved; use an explicit run identity for raw relocations.
    """
    return f"run-{hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:24]}"


def occurrence_id(run_id: str, source_path: str, line: int) -> str:
    """Identify one physical source occurrence, including identical repeated generations."""
    payload = {"run_id": run_id, "source_path": source_path, "line": line}
    return "occ-" + hashlib.sha256(canonical_bytes(payload)).hexdigest()[:32]


def content_fingerprint(row: Rollout) -> str:
    """Hash validated generation content and trajectories, excluding mutable scoring."""
    raw = row.model_dump(mode="json")
    trajectory = []
    for step in raw.get("trajectory") or []:
        trajectory.append(
            {
                key: value
                for key, value in step.items()
                if key not in {"reward", "advantage", "trajectory_id"}
            }
        )
    payload = {
        "prompt": raw.get("prompt"),
        "completion": raw.get("completion"),
        "trajectory": trajectory,
        "tool_defs": raw.get("tool_defs"),
    }
    return "content-" + hashlib.sha256(canonical_bytes(payload)).hexdigest()


def scoring_revision(row: Rollout) -> str:
    """Hash scoring measurements independently of source occurrence and content."""
    raw = row.model_dump(mode="json")
    payload = {
        "reward": row.reward,
        "metrics": row.metrics,
        "trajectory_scores": [
            {key: step.get(key) for key in ("reward", "advantage")}
            for step in raw.get("trajectory") or []
        ],
    }
    return "score-" + hashlib.sha256(canonical_bytes(payload)).hexdigest()


def attach_identity(
    row: Rollout,
    *,
    run_id: str,
    source_path: str,
    line: int,
    adapter: str,
    step_index: int | None,
    environment_namespace: str | None = None,
    task_namespace: str | None = None,
    preserve: bool = False,
) -> Rollout:
    """Attach original source occurrence and scoring identity to a validated row.

    Normalized readers preserve existing IDs and provenance. Legacy rollout and
    group hashes remain available for compatibility; occurrence_id is the join
    key and the identifier used by new detector evidence. No upstream keys are
    guessed and no source or metadata files are written.
    """
    values = row.model_dump(mode="python")
    if not preserve or row.run_id is None:
        values["run_id"] = run_id
    if not preserve:
        values["step_index"] = step_index
    actual_run = values["run_id"]
    values["environment_namespace"] = (
        row.environment_namespace or environment_namespace or actual_run
    )
    values["task_namespace"] = (
        row.task_namespace or task_namespace or values["environment_namespace"]
    )
    if row.rollout_id is None:
        values["rollout_id"] = rollout_id(
            row.example_id, values.get("prompt"), values.get("completion"), row.reward
        )
    if row.group_id is None:
        values["group_id"] = group_id(row.example_id)
    provenance = row.provenance if preserve else None
    if provenance is None:
        provenance = SourceProvenance(
            source_path=source_path,
            line=line,
            adapter=adapter,
            namespace=actual_run,
            legacy_ids=dict(row.identity_aliases),
        )
    values["provenance"] = provenance
    if not preserve or row.occurrence_id is None:
        values["occurrence_id"] = occurrence_id(actual_run, provenance.source_path, provenance.line)
    # Recompute content/scoring fingerprints after rescoring, preserving occurrence.
    values["content_fingerprint"] = content_fingerprint(row)
    values["scoring_revision"] = scoring_revision(row)
    return validate_rollout(values)
