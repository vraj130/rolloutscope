"""Stable, content-derived identifiers for rollouts, groups, and runs.

ID scheme (PLAN.md D-006, D-007):
- ``rollout_id``: sha256 over a canonical (sorted-key) orjson serialization of
  (example_id, prompt, completion, reward), truncated to 16 hex chars. The same
  row content always yields the same id, no matter who serialized it.
- ``group_id``: derived from the grouping key ``example_id`` as ``grp-{example_id}``.
- ``run_id``: truncated sha256 of the canonicalized run manifest (metadata.json)
  when present, else of the run directory name.
- ``step_index``: an optional integer attached by adapters from on-disk step
  layout only, never guessed; it is not computed here.

v1 join contract: future white-box stores (activations, representation drift, SAE
features) and training-signal sidecars key on (run_id, rollout_id, step_index).
Tensors are never embedded in this JSONL; they live in external stores that join
back on those three keys. Keeping rollout_id content-derived means v1 data can be
joined onto v0 artifacts without re-ingesting anything.
"""

from __future__ import annotations

import hashlib
from typing import Any

import orjson

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
