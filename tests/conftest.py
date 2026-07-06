"""Shared test fixtures: paths into tests/fixtures/."""

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def eval_run_dir() -> Path:
    return FIXTURES_DIR / "verifiers_eval_run"


@pytest.fixture
def multi_turn_path() -> Path:
    return FIXTURES_DIR / "multi_turn_rollout.jsonl"
