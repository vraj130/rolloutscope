"""Adapter for verifiers ``evaluate(..., save_results=True)`` output.

Parses results.jsonl (one RolloutOutput per line) plus metadata.json (a single
GenerateMetadata object) per the pinned on-disk format (verifiers @ 5885ab9c).
Upstream appends rows on resume (mode="a"), so duplicate example_ids are
possible; they are preserved as distinct rollouts because deduplication is
analysis policy, not parsing policy. Eval output carries no step ordering, so
step_index is always None (snapshot mode).
"""

from __future__ import annotations

from pathlib import Path

from rolloutscope.adapters.base import (
    RESULTS_FILENAME,
    TRAIN_ROLLOUTS_FILENAME,
    BaseAdapter,
    RunManifest,
    SourceFile,
    metadata_namespaces,
    read_run_metadata,
)
from rolloutscope.schema.ids import run_id_from_path


class VerifiersEvalAdapter(BaseAdapter):
    """Loads verifiers eval artifacts: a run directory with results.jsonl (plus
    optional metadata.json), or a direct path to a .jsonl file of trace rows."""

    name: str = "verifiers_eval"

    def detect(self, path: Path) -> bool:
        """Return True when path is a verifiers eval layout this adapter handles.

        Input: any filesystem path. True for a directory containing
        results.jsonl, or for an existing .jsonl file that is not a prime-rl
        train_rollouts.jsonl (that name belongs to the prime-rl adapter).
        Orchestrator eval files (eval_rollouts_{env_name}.jsonl) share the same
        trace row shape and are handled here as plain jsonl.
        """
        if path.is_dir():
            return (path / RESULTS_FILENAME).is_file()
        return path.suffix == ".jsonl" and path.name != TRAIN_ROLLOUTS_FILENAME and path.is_file()

    def load_run(self, path: Path) -> RunManifest:
        """Discover the eval run at path.

        Input: a run directory containing results.jsonl, or a direct path to a
        .jsonl file. Run identity derives from the resolved run directory for
        results.jsonl, and from the resolved file path for standalone files.
        Mutable metadata never affects identity. Raises
        FileNotFoundError when path holds nothing loadable.
        """
        path = path.resolve()
        if path.is_dir():
            results = path / RESULTS_FILENAME
            if not results.is_file():
                raise FileNotFoundError(f"no {RESULTS_FILENAME} in {path}")
            return self._manifest(results, metadata_dir=path)
        if not path.is_file():
            raise FileNotFoundError(f"no such file or directory: {path}")
        if path.name == RESULTS_FILENAME:
            return self._manifest(path, metadata_dir=path.parent)
        return self._manifest(path, metadata_dir=None)

    def _manifest(self, results: Path, *, metadata_dir: Path | None) -> RunManifest:
        """Build the manifest for one results file.

        Inputs: the results file path, the directory to look for metadata.json
        in (None to skip the lookup). The single
        source file always has step_index None: eval layout provides no step
        ordering and it is never guessed.
        """
        metadata = read_run_metadata(metadata_dir) if metadata_dir is not None else None
        environment_namespace, task_namespace = metadata_namespaces(metadata)
        metadata_path = metadata_dir / "metadata.json" if metadata_dir is not None else None
        return RunManifest(
            run_id=run_id_from_path(metadata_dir if metadata_dir is not None else results),
            files=(SourceFile(path=results, step_index=None, format=self.name),),
            metadata=metadata or {},
            root=results.parent,
            format=self.name,
            metadata_sources=(metadata_path,)
            if metadata_path is not None and metadata_path.is_file()
            else (),
            environment_namespace=environment_namespace,
            task_namespace=task_namespace,
        )
