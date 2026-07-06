"""Shared fixtures for detector tests: labeled fixture loading."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from rolloutscope.schema import MultiTurnRollout, SingleTurnRollout, read_rollouts

LABELED_DIR = Path(__file__).parent.parent / "fixtures" / "labeled"
ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"

LoadLabeled = Callable[[str], list[SingleTurnRollout | MultiTurnRollout]]


@pytest.fixture
def labeled_dir() -> Path:
    return LABELED_DIR


@pytest.fixture
def artifacts_dir() -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR


@pytest.fixture
def load_labeled() -> LoadLabeled:
    """Return a loader for labeled fixture files by stem name.

    Usage: ``load_labeled("verifier_tamper_hacked")`` reads
    tests/fixtures/labeled/verifier_tamper_hacked.jsonl into rollouts.
    """

    def _load(name: str) -> list[SingleTurnRollout | MultiTurnRollout]:
        path = LABELED_DIR / f"{name}.jsonl"
        rollouts = list(read_rollouts(path))
        assert rollouts, f"fixture {path} is empty or failed to parse"
        return rollouts

    return _load
