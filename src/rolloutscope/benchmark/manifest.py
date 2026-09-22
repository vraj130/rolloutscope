"""The frozen dataset manifest: what was measured, and how to get it back.

TRACE is gated and CC-BY-SA-4.0, so its rows are not redistributed here. The
manifest carries identity instead of content: the dataset revision, the exact
rows selected, a content hash per row, and the split each row belongs to. Anyone
with dataset access can refetch at the pinned revision and confirm, row by row,
that they are looking at the bytes a reported number came from.

Everything that can move a result is pinned in one place: the dataset revision,
the mapping version, the split version and weights, and the detector
configuration hash. A report records the manifest it ran against, so two reports
are comparable exactly when those pins agree.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from rolloutscope.benchmark.splits import (
    SPLIT_VERSION,
    SPLIT_WEIGHTS,
    Split,
    assign_split,
    scenario_key,
)
from rolloutscope.benchmark.taxonomy import is_hacked, parse_label
from rolloutscope.benchmark.trace_map import TRACE_MAPPING_VERSION

MANIFEST_VERSION = "1"
"""Version of the manifest document shape itself."""


class DatasetRef(BaseModel):
    """The pinned upstream artifact a manifest was built from."""

    name: str
    revision: str
    """Repository commit SHA on the Hugging Face Hub, not a branch name."""
    config: str = "default"
    split: str = "train"
    license: str
    gated: bool
    redistribution: str
    """Why row content is or is not committed alongside this manifest."""
    files: dict[str, str] = Field(default_factory=dict)
    """Path inside the dataset repo mapped to its SHA-256, when one was fetched."""


class RowRef(BaseModel):
    """One selected row: its identity, its label, and where it belongs."""

    trajectory_id: str
    label: str
    scenario_key: str
    split: Split
    content_sha256: str
    positive: bool
    codes: tuple[str, ...] = ()


class ManifestCounts(BaseModel):
    """Denominators a reader should not have to recompute to sanity-check."""

    rows: int = 0
    scenarios: int = 0
    by_split: dict[str, int] = Field(default_factory=dict)
    positives_by_split: dict[str, int] = Field(default_factory=dict)
    negatives_by_split: dict[str, int] = Field(default_factory=dict)
    by_code: dict[str, int] = Field(default_factory=dict)


class BenchmarkManifest(BaseModel):
    """A frozen, reproducible description of one benchmark population."""

    manifest_version: str = MANIFEST_VERSION
    dataset: DatasetRef
    mapping_version: str = TRACE_MAPPING_VERSION
    split_version: str = SPLIT_VERSION
    split_weights: dict[str, float] = Field(
        default_factory=lambda: {str(k): v for k, v in SPLIT_WEIGHTS.items()}
    )
    notes: list[str] = Field(default_factory=list)
    counts: ManifestCounts = Field(default_factory=ManifestCounts)
    rows: list[RowRef] = Field(default_factory=list)

    def for_split(self, split: Split) -> list[RowRef]:
        """Return the manifest rows assigned to one split, in manifest order."""
        return [row for row in self.rows if row.split == split]

    @property
    def by_trajectory_id(self) -> dict[str, RowRef]:
        """Index the manifest rows by ``trajectory_id``."""
        return {row.trajectory_id: row for row in self.rows}

    def digest(self) -> str:
        """SHA-256 over the canonical serialization of this manifest.

        Input: nothing. Output: a hex digest a report can cite so a reader can
        tell at a glance whether two reports ran against the same manifest.
        """
        return hashlib.sha256(to_json(self).encode("utf-8")).hexdigest()


def row_digest(row: dict[str, Any]) -> str:
    """Content hash of one raw TRACE row.

    Input: the raw row dict. Output: the SHA-256 of a canonical JSON encoding of
    just the three payload fields, so an unrelated column added upstream does not
    invalidate every hash while a changed conversation still does.
    """
    canonical = json.dumps(
        {
            "trajectory_id": row.get("trajectory_id"),
            "conversation": row.get("conversation"),
            "label": row.get("label"),
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def first_user_text(conversation: str) -> str:
    """Extract the opening user message from a raw conversation cell.

    Input: the JSON-encoded conversation string. Output: the first user
    message's content, or an empty string when the cell will not parse or holds
    no user turn. Used to derive the scenario key without building a rollout.
    """
    try:
        messages = json.loads(conversation)
    except (json.JSONDecodeError, TypeError):
        return ""
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            return content if isinstance(content, str) else ""
    return ""


def build_manifest(
    rows: list[dict[str, Any]],
    dataset: DatasetRef,
    *,
    notes: list[str] | None = None,
) -> BenchmarkManifest:
    """Build a manifest from raw rows.

    Input: every raw row to include, the pinned dataset reference, and optional
    free-text notes. Output: a :class:`BenchmarkManifest` with rows sorted by
    ``trajectory_id`` so the document is byte-stable whatever order the rows
    arrived in. Rows whose label is unusable are skipped, because a row that
    cannot be scored as positive or negative does not belong in a manifest that
    defines a scored population.
    """
    refs: list[RowRef] = []
    for raw in rows:
        label = raw.get("label")
        if not isinstance(label, str):
            continue
        positive = is_hacked(label)
        if positive is None:
            continue
        conversation = raw.get("conversation")
        if not isinstance(conversation, str):
            continue
        key = scenario_key(first_user_text(conversation) or str(raw.get("trajectory_id") or ""))
        refs.append(
            RowRef(
                trajectory_id=str(raw.get("trajectory_id") or ""),
                label=label,
                scenario_key=key,
                split=assign_split(key),
                content_sha256=row_digest(raw),
                positive=positive,
                codes=tuple(c for c in parse_label(label) if c != "0"),
            )
        )
    refs.sort(key=lambda ref: ref.trajectory_id)

    by_split: Counter[str] = Counter()
    positives: Counter[str] = Counter()
    negatives: Counter[str] = Counter()
    by_code: Counter[str] = Counter()
    for ref in refs:
        by_split[ref.split] += 1
        (positives if ref.positive else negatives)[ref.split] += 1
        for code in ref.codes:
            by_code[code] += 1

    counts = ManifestCounts(
        rows=len(refs),
        scenarios=len({ref.scenario_key for ref in refs}),
        by_split=dict(sorted(by_split.items())),
        positives_by_split=dict(sorted(positives.items())),
        negatives_by_split=dict(sorted(negatives.items())),
        by_code=dict(sorted(by_code.items())),
    )
    return BenchmarkManifest(dataset=dataset, notes=notes or [], counts=counts, rows=refs)


def to_json(manifest: BenchmarkManifest) -> str:
    """Serialize a manifest deterministically.

    Input: the manifest. Output: JSON with sorted keys, two-space indent, and a
    trailing newline, so committing a regenerated manifest produces an empty
    diff when nothing changed.
    """
    payload = manifest.model_dump(mode="json")
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def save_manifest(manifest: BenchmarkManifest, path: Path) -> None:
    """Write a manifest to ``path`` in the deterministic form."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json(manifest), encoding="utf-8")


def load_manifest(path: Path) -> BenchmarkManifest:
    """Read a manifest from ``path``.

    Input: the manifest path. Output: the parsed manifest. Raises
    ``pydantic.ValidationError`` on a document that does not fit the contract,
    rather than degrading to a partly populated object.
    """
    return BenchmarkManifest.model_validate_json(path.read_text(encoding="utf-8"))


class Verification(BaseModel):
    """Result of checking fetched rows against a manifest."""

    matched: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    unexpected: list[str] = Field(default_factory=list)
    changed: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when every manifest row was fetched with unchanged content."""
        return not self.missing and not self.changed


def verify(
    manifest: BenchmarkManifest,
    rows: list[dict[str, Any]],
    *,
    only: set[str] | None = None,
) -> Verification:
    """Check fetched rows against the manifest.

    Input: the manifest, the raw rows fetched, and optionally the subset of
    trajectory ids the caller actually asked for (a single split, say). Output: a
    :class:`Verification` naming rows that are missing, unexpected, or whose
    content hash no longer matches. A changed hash means the upstream revision
    pin was not honoured, which invalidates any number measured from those rows.

    Scoping with ``only`` matters: verifying a one-split fetch against the whole
    manifest would report every other split as missing, which is noise, not a
    finding.
    """
    expected = manifest.by_trajectory_id
    if only is not None:
        expected = {k: v for k, v in expected.items() if k in only}
    seen: set[str] = set()
    result = Verification()
    for raw in rows:
        trajectory_id = str(raw.get("trajectory_id") or "")
        seen.add(trajectory_id)
        ref = expected.get(trajectory_id)
        if ref is None:
            result.unexpected.append(trajectory_id)
        elif ref.content_sha256 != row_digest(raw):
            result.changed.append(trajectory_id)
        else:
            result.matched.append(trajectory_id)
    result.missing = sorted(set(expected) - seen)
    result.matched.sort()
    result.unexpected.sort()
    result.changed.sort()
    return result
