"""Read RolloutScope normalized JSONL using its explicit schema marker.

This is RolloutScope's own format, independent of upstream version pins. Legacy
upstream rows in mixed files use the shared verifiers @ 5885ab9c raw contract.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import orjson

from rolloutscope.adapters.base import RESULTS_FILENAME, RunManifest
from rolloutscope.adapters.verifiers_eval import VerifiersEvalAdapter


class NormalizedAdapter(VerifiersEvalAdapter):
    """Detect explicit schema markers and preserve normalized row identities."""

    name: str = "normalized"

    def detect(self, path: Path) -> bool:
        """Probe until a recognizable row, skipping malformed leading records.

        A schema marker selects this adapter even if its version is unsupported
        or malformed; loading then reports the rejection without raw fallback.
        Probe reads never enter ingestion counts, which come from the load pass.
        """
        source = path / RESULTS_FILENAME if path.is_dir() else path
        if not source.is_file() or source.suffix != ".jsonl":
            return False
        with source.open("rb") as handle:
            for line in handle:
                try:
                    raw = orjson.loads(line)
                except orjson.JSONDecodeError:
                    continue
                if not isinstance(raw, dict):
                    continue
                if "schema_version" in raw:
                    return True
                required = {"example_id", "reward", "is_completed", "is_truncated", "metrics"}
                if required <= raw.keys():
                    return False
        return False

    def load_run(self, path: Path) -> RunManifest:
        """Discover normalized files with original identities resolved on load."""
        return replace(super().load_run(path), format=self.name)
