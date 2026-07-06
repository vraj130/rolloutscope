"""Adapter for prime-rl orchestrator training rollouts.

Parses train_rollouts.jsonl files written per training step. Pinned layout
(prime-rl @ df2acf48): the orchestrator writes ``step_path /
"train_rollouts.jsonl"``. The row shape is the same verifiers trace shape as
eval results.jsonl (the dumped prime-rl Rollout IS a plain vf.Trace), so row
normalization is shared with the verifiers adapter through adapters.base.

Rule 9: ``advantages`` and ``is_trainable`` are in-memory training signals,
excluded from the on-disk dump along with the other orchestration metadata
(kind, env_name, group_id, policy_version). This adapter never parses for them.

step_index comes only from the step-directory name mapping in adapters.base
(step_index_from_name); unrecognized directory names mean snapshot mode
(step_index None), never a guess.
"""

from __future__ import annotations

from pathlib import Path

from rolloutscope.adapters.base import (
    TRAIN_ROLLOUTS_FILENAME,
    BaseAdapter,
    RunManifest,
    SourceFile,
    read_run_metadata,
    step_index_from_name,
)
from rolloutscope.schema import run_id_from_name


class PrimeRlTrainAdapter(BaseAdapter):
    """Loads prime-rl training artifacts: a direct train_rollouts.jsonl file, a
    single step directory containing one, or a run directory of step dirs."""

    name: str = "prime_rl_train"

    def detect(self, path: Path) -> bool:
        """Return True when path is a prime-rl train layout this adapter handles.

        Input: any filesystem path. True for a file named train_rollouts.jsonl,
        a directory containing train_rollouts.jsonl (a single step directory),
        or a directory whose immediate subdirectories contain one (a run
        directory of step directories).
        """
        if path.is_file():
            return path.name == TRAIN_ROLLOUTS_FILENAME
        if not path.is_dir():
            return False
        if (path / TRAIN_ROLLOUTS_FILENAME).is_file():
            return True
        return any(
            (child / TRAIN_ROLLOUTS_FILENAME).is_file()
            for child in path.iterdir()
            if child.is_dir()
        )

    def load_run(self, path: Path) -> RunManifest:
        """Discover the training run at path.

        Input: a direct train_rollouts.jsonl path, a single step directory, or
        a run directory containing step directories. Files are ordered
        numerically by step_index, with unindexed directories after, ordered by
        name. The run root is the parent of a step-named directory (per the
        pinned ``step_path / "train_rollouts.jsonl"`` layout), so all three
        entry forms of the same run derive the same run_id. Raises
        FileNotFoundError when path holds no train_rollouts.jsonl.
        """
        path = path.resolve()
        if path.is_file():
            if path.name != TRAIN_ROLLOUTS_FILENAME:
                raise FileNotFoundError(f"{path} is not a {TRAIN_ROLLOUTS_FILENAME} file")
            step = step_index_from_name(path.parent.name)
            root = path.parent.parent if step is not None else path.parent
            return self._manifest(root, (SourceFile(path=path, step_index=step),))
        if not path.is_dir():
            raise FileNotFoundError(f"no such file or directory: {path}")
        direct = path / TRAIN_ROLLOUTS_FILENAME
        if direct.is_file():
            step = step_index_from_name(path.name)
            root = path.parent if step is not None else path
            return self._manifest(root, (SourceFile(path=direct, step_index=step),))
        files = self._discover_step_files(path)
        if not files:
            raise FileNotFoundError(f"no {TRAIN_ROLLOUTS_FILENAME} under {path}")
        return self._manifest(path, files)

    def _discover_step_files(self, run_dir: Path) -> tuple[SourceFile, ...]:
        """Collect per-step train files from run_dir's immediate subdirectories.

        Input: the run directory. Each subdirectory containing
        train_rollouts.jsonl contributes one SourceFile carrying the
        step_index its name maps to (or None). Ordering is numeric by
        step_index (so step_10 sorts after step_2), then unindexed directories
        by name, so load order is deterministic.
        """
        found: list[tuple[int | None, Path]] = []
        for child in run_dir.iterdir():
            if not child.is_dir():
                continue
            train = child / TRAIN_ROLLOUTS_FILENAME
            if train.is_file():
                found.append((step_index_from_name(child.name), train))

        def sort_key(entry: tuple[int | None, Path]) -> tuple[int, int, str]:
            index, train = entry
            if index is None:
                return (1, 0, train.parent.name)
            return (0, index, train.parent.name)

        found.sort(key=sort_key)
        return tuple(SourceFile(path=train, step_index=index) for index, train in found)

    def _manifest(self, root: Path, files: tuple[SourceFile, ...]) -> RunManifest:
        """Build the manifest for a discovered training run.

        Inputs: the run root directory and the ordered source files. run_id is
        derived from the run root directory name; any metadata.json found at
        the root is passed through as run-level fields only.
        """
        # TODO(open question for the orchestrator): prime-rl @ df2acf48 pins no
        # run-level manifest for training runs in the on-disk-format reference.
        # Exact question: does the orchestrator write a run-level metadata or
        # config file next to the step directories, and what is it named? Until
        # that is pinned, run_id comes from the run root directory name, and a
        # metadata.json found there is passthrough only (never the run_id
        # source, to keep train run_ids stable however the manifest question
        # resolves).
        metadata = read_run_metadata(root)
        return RunManifest(
            run_id=run_id_from_name(root.name),
            files=files,
            metadata=metadata or {},
        )
