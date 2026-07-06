"""Artifact adapters: verifiers and prime-rl on-disk output to normalized rows.

Adapters depend only on the normalized schema (core rule 3: no verifiers or
prime-rl imports; the on-disk contracts come from the pinned references).
``resolve_adapter`` picks the right adapter for a path via each adapter's
``detect()``; the tie-break order is documented on resolve_adapter itself.
"""

from __future__ import annotations

from pathlib import Path

from rolloutscope.adapters.base import (
    METADATA_FILENAME,
    RESULTS_FILENAME,
    TRAIN_ROLLOUTS_FILENAME,
    Adapter,
    BaseAdapter,
    RunManifest,
    SourceFile,
    iter_normalized_rows,
    normalize_row,
    read_run_metadata,
    step_index_from_name,
)
from rolloutscope.adapters.prime_rl_train import PrimeRlTrainAdapter
from rolloutscope.adapters.verifiers_eval import VerifiersEvalAdapter

VERIFIERS_EVAL = VerifiersEvalAdapter()
PRIME_RL_TRAIN = PrimeRlTrainAdapter()

# Tie-break order for resolve_adapter, first match wins.
ADAPTERS: tuple[Adapter, ...] = (VERIFIERS_EVAL, PRIME_RL_TRAIN)


def resolve_adapter(path: Path) -> Adapter:
    """Pick the adapter whose detect() accepts path (used by the CLI later).

    Input: any filesystem path. Tie-break order, first match wins:
    verifiers_eval, then prime_rl_train. The two are mutually exclusive on real
    pinned layouts: a genuine prime-rl step directory never contains
    results.jsonl (the orchestrator names its eval output
    eval_rollouts_{env_name}.jsonl), and the verifiers adapter refuses files
    named train_rollouts.jsonl. The order therefore only matters for
    pathological directories carrying both artifact kinds, where the eval
    artifact wins. Raises ValueError when no adapter recognizes path.
    """
    for adapter in ADAPTERS:
        if adapter.detect(path):
            return adapter
    raise ValueError(f"no adapter recognizes {path}")


__all__ = [
    "ADAPTERS",
    "METADATA_FILENAME",
    "PRIME_RL_TRAIN",
    "RESULTS_FILENAME",
    "TRAIN_ROLLOUTS_FILENAME",
    "VERIFIERS_EVAL",
    "Adapter",
    "BaseAdapter",
    "PrimeRlTrainAdapter",
    "RunManifest",
    "SourceFile",
    "VerifiersEvalAdapter",
    "iter_normalized_rows",
    "normalize_row",
    "read_run_metadata",
    "resolve_adapter",
    "step_index_from_name",
]
