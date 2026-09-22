"""Deterministic scenario grouping and train / tuning / holdout assignment.

Two rules drive this module.

First, a split boundary is drawn between *scenarios*, never between rows, so
sibling trajectories of one task cannot straddle the boundary and leak tuning
information into the holdout. The scenario key is content derived: the
normalized opening user message, which is the task statement the trajectory was
generated from.

Second, assignment is a pure function of the scenario key. It does not depend on
row order, on dataset size, or on how many rows were fetched, so a partial fetch
assigns every row it did fetch exactly as a full fetch would, and adding rows
upstream never reshuffles the rows already assigned.

Measured caveat, recorded so nobody reads more safety into this than it provides:
at TRACE revision ``31d87f06`` the 517 rows produce 516 distinct scenario keys.
Exactly one pair of rows shares an opening message. Grouping is therefore close
to a no-op on this artifact; it is kept because it is the correct guard, it
catches that one pair, and it is what a future dataset with real sibling clusters
will need. See ``docs/benchmark-metrics.md`` for the split proportions rationale.
"""

from __future__ import annotations

import hashlib
import re
from typing import Literal, get_args

Split = Literal["train", "tuning", "holdout"]

SPLITS: tuple[Split, ...] = get_args(Split)

SPLIT_WEIGHTS: dict[Split, float] = {"train": 0.40, "tuning": 0.20, "holdout": 0.40}
"""Target share of scenarios per split.

Chosen before any holdout measurement was taken, and recorded here so the choice
is auditable rather than retrofitted. The holdout carries the largest share
because it is the only partition a reported accuracy number may come from, and a
40 percent share of 517 rows keeps the Wilson half-width near 7 points for a rate
around 0.5. Tuning is deliberately the smallest: thresholds are set from it, so
overfitting it costs the least.
"""

SPLIT_SALT = "rolloutscope/trace/split/v1"
"""Salt mixed into the assignment hash.

Fixed forever for this split version. Changing it reshuffles every assignment,
which invalidates any previously reported holdout number, so a change requires a
new split version and a new manifest, never an edit in place.
"""

SPLIT_VERSION = "1"
"""Version of the scenario-key and assignment rules in this module."""

_WHITESPACE = re.compile(r"\s+")


def normalize_scenario_text(text: str) -> str:
    """Normalize a task statement before hashing it into a scenario key.

    Input: the raw opening user message. Output: the text lowercased with all
    whitespace runs collapsed to single spaces and the ends stripped, so that
    reformatting alone (a rewrapped paragraph, trailing spaces, a changed
    indent) does not split one scenario into two.

    Known limitation: this catches exact and whitespace-equivalent duplicates
    only. Two paraphrases of the same task still receive different keys and can
    land in different splits.
    """
    return _WHITESPACE.sub(" ", text).strip().lower()


def scenario_key(text: str) -> str:
    """Return the stable scenario key for a task statement.

    Input: the raw opening user message. Output: the first 16 hex characters of
    the SHA-256 of its normalized form. Truncation to 64 bits is deliberate: the
    key appears in every manifest row and every report, and at a few thousand
    scenarios the collision probability stays far below the noise floor of the
    measurement it labels.
    """
    normalized = normalize_scenario_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _fraction(key: str) -> float:
    """Map a scenario key uniformly into [0, 1) via a salted hash."""
    digest = hashlib.sha256(f"{SPLIT_SALT}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2.0**64


def assign_split(key: str, weights: dict[Split, float] | None = None) -> Split:
    """Assign one scenario key to a split.

    Input: a scenario key and, optionally, split weights (defaults to
    :data:`SPLIT_WEIGHTS`). Output: the split name. The result depends only on
    the key and the salt, so the same scenario always lands in the same split
    whatever else was fetched alongside it.
    """
    table = weights or SPLIT_WEIGHTS
    total = sum(table[name] for name in SPLITS)
    if total <= 0:
        raise ValueError("split weights must sum to a positive number")
    position = _fraction(key) * total
    cumulative = 0.0
    for name in SPLITS:
        cumulative += table[name]
        if position < cumulative:
            return name
    return SPLITS[-1]
