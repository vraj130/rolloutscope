"""Unit tests for the isolated wheel-install smoke helpers."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from tests.script_loader import REPO_ROOT, load_script


def test_find_wheel_selects_the_project_wheel(tmp_path: Path) -> None:
    check_packaging = load_script("scripts/check_packaging.py")

    (tmp_path / "rolloutscope-0.1.0.tar.gz").write_bytes(b"sdist")
    wheel = tmp_path / "rolloutscope-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    (tmp_path / "unrelated-1.0-py3-none-any.whl").write_bytes(b"other")
    assert check_packaging.find_wheel(tmp_path) == wheel


def test_find_wheel_rejects_an_empty_dist_dir(tmp_path: Path) -> None:
    check_packaging = load_script("scripts/check_packaging.py")
    with pytest.raises(FileNotFoundError, match="rolloutscope wheel"):
        check_packaging.find_wheel(tmp_path)


def test_expected_detectors_match_pyproject_entry_points() -> None:
    check_packaging = load_script("scripts/check_packaging.py")
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    registered = set(data["project"]["entry-points"]["rolloutscope.detectors"])
    assert set(check_packaging.EXPECTED_DETECTORS) == registered


def test_installed_module_must_not_come_from_the_source_tree() -> None:
    check_packaging = load_script("scripts/check_packaging.py")
    src = (REPO_ROOT / "src" / "rolloutscope").resolve()
    with pytest.raises(AssertionError, match="source checkout"):
        check_packaging.assert_not_source_tree(str(src / "__init__.py"), src)
    check_packaging.assert_not_source_tree(
        "/tmp/rs-smoke/lib/python3.12/site-packages/rolloutscope/__init__.py",
        src,
    )
