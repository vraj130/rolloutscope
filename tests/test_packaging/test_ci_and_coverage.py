"""Release-quality gates that must live in config, not tribal knowledge."""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
CI_YML = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def test_coverage_floor_is_at_least_85() -> None:
    """A coverage percentage below the published floor must fail the test job."""
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    fail_under = data["tool"]["coverage"]["report"]["fail_under"]
    assert fail_under >= 85


def test_ci_checks_the_lockfile_and_runs_frozen() -> None:
    """Development CI must use the committed lockfile, not a floating resolve."""
    text = CI_YML.read_text(encoding="utf-8")
    assert "uv lock --check" in text
    assert "uv run --frozen" in text


def test_ci_runs_an_isolated_wheel_smoke() -> None:
    """A wheel install outside the checkout must be part of CI, not a manual ritual."""
    text = CI_YML.read_text(encoding="utf-8")
    assert "scripts/check_packaging.py" in text
    assert "uv build" in text
