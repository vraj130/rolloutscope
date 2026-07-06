"""Adapter-test fixtures: paths into tests/fixtures/ specific to Phase 3."""

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def prime_rl_run_dir() -> Path:
    return FIXTURES_DIR / "prime_rl_run"
