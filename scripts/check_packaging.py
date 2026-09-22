"""Build a wheel, install it outside the source tree, and smoke-test the install.

This script is stdlib plus the ``uv`` CLI. It does not import rolloutscope from
the checkout. The isolated venv is the proof that templates, the CLI, and
detector entry points ship in the wheel.

Usage:

    uv run python scripts/check_packaging.py
    uv run python scripts/check_packaging.py --dist dist --python 3.12
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

EXPECTED_DETECTORS: tuple[str, ...] = (
    "answer_leakage_echo",
    "degenerate_repetition",
    "format_only_wins",
    "length_inflation",
    "reward_saturation_group_collapse",
    "verifier_tamper",
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def find_wheel(dist_dir: Path) -> Path:
    """Return the rolloutscope wheel in ``dist_dir``.

    Input: a directory that may contain sdists and unrelated wheels. Output: the
    single ``rolloutscope-*.whl`` path. Raises FileNotFoundError when none exist
    and ValueError when more than one matches.
    """
    wheels = sorted(dist_dir.glob("rolloutscope-*.whl"))
    if not wheels:
        raise FileNotFoundError(f"rolloutscope wheel not found in {dist_dir}")
    if len(wheels) > 1:
        names = ", ".join(path.name for path in wheels)
        raise ValueError(f"multiple rolloutscope wheels in {dist_dir}: {names}")
    return wheels[0]


def assert_not_source_tree(module_file: str, source_root: Path) -> None:
    """Raise AssertionError when ``module_file`` lives inside ``source_root``."""
    resolved = Path(module_file).resolve()
    root = source_root.resolve()
    if resolved == root or root in resolved.parents:
        raise AssertionError(f"installed module {resolved} is inside the source checkout {root}")


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a command and raise RuntimeError with stdout/stderr on failure."""
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def build_wheel(repo: Path, dist_dir: Path) -> Path:
    """Build a wheel from ``repo`` into ``dist_dir`` and return its path."""
    dist_dir.mkdir(parents=True, exist_ok=True)
    _run(["uv", "build", "--wheel", "--out-dir", str(dist_dir)], cwd=repo)
    return find_wheel(dist_dir)


def create_venv(venv_dir: Path, python: str | None) -> Path:
    """Create an isolated venv and return its interpreter path."""
    command = ["uv", "venv", str(venv_dir)]
    if python:
        command.extend(["--python", python])
    _run(command)
    unix = venv_dir / "bin" / "python"
    if unix.exists():
        return unix
    windows = venv_dir / "Scripts" / "python.exe"
    if windows.exists():
        return windows
    raise FileNotFoundError(f"venv python not found under {venv_dir}")


def install_wheel(python: Path, wheel: Path) -> None:
    """Install ``wheel`` into the interpreter's environment using uv."""
    _run(["uv", "pip", "install", "--python", str(python), str(wheel)])


def copy_demo(repo: Path, dest: Path) -> Path:
    """Copy the bundled demo fixture into ``dest`` and return that path."""
    source = repo / "tests" / "fixtures" / "demo"
    if not source.is_dir():
        raise FileNotFoundError(f"demo fixture missing: {source}")
    return Path(shutil.copytree(source, dest))


def _cli_bin(python: Path) -> Path:
    candidate = python.parent / "rolloutscope"
    if candidate.exists():
        return candidate
    windows = python.parent / "rolloutscope.exe"
    if windows.exists():
        return windows
    raise FileNotFoundError(f"rolloutscope CLI not found next to {python}")


def run_smoke(python: Path, demo: Path, work: Path, source_root: Path) -> None:
    """Run CLI, entry-point, template, and analyze checks in the isolated env."""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    cli = _cli_bin(python)

    version = _run([str(cli), "--version"], cwd=work, env=env)
    if not version.stdout.strip():
        raise RuntimeError("rolloutscope --version printed nothing")

    listed = _run([str(cli), "detectors", "list"], cwd=work, env=env)
    missing = [name for name in EXPECTED_DETECTORS if name not in listed.stdout]
    if missing:
        raise RuntimeError(f"detectors list missing {missing}:\n{listed.stdout}")

    inspect = f"""
from importlib.metadata import entry_points
from jinja2 import Environment, PackageLoader
import rolloutscope

expected = {set(EXPECTED_DETECTORS)!r}
names = {{ep.name for ep in entry_points(group="rolloutscope.detectors")}}
if names != expected:
    raise SystemExit(f"entry points {{sorted(names)}} != {{sorted(expected)}}")
for ep in entry_points(group="rolloutscope.detectors"):
    ep.load()
Environment(loader=PackageLoader("rolloutscope.report", "templates"), autoescape=True).get_template(
    "report.html.j2"
)
print(rolloutscope.__file__)
"""
    inspection = _run([str(python), "-c", inspect], cwd=work, env=env)
    module_file = inspection.stdout.strip().splitlines()[-1]
    assert_not_source_tree(module_file, source_root)

    html = work / "report.html"
    json_out = work / "findings.json"
    _run(
        [
            str(cli),
            "analyze",
            str(demo),
            "--out",
            str(html),
            "--json",
            str(json_out),
            "--quiet",
            "--fail-on",
            "none",
        ],
        cwd=work,
        env=env,
    )
    if not html.is_file() or not json_out.is_file():
        raise RuntimeError("analyze did not write HTML and JSON artifacts")
    html_text = html.read_text(encoding="utf-8")
    if "<html" not in html_text.lower():
        raise RuntimeError("HTML report does not look like a self-contained document")


def main(argv: list[str] | None = None) -> int:
    """Build or reuse a wheel, install it in isolation, and run the smoke checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dist",
        type=Path,
        default=None,
        help="Directory containing a pre-built rolloutscope wheel (default: build one).",
    )
    parser.add_argument(
        "--python",
        default=None,
        help="Python version or path for the isolated venv (passed to uv venv --python).",
    )
    args = parser.parse_args(argv)

    tmp_root = Path(tempfile.mkdtemp(prefix="rolloutscope-packaging-"))
    try:
        dist_dir = args.dist
        if dist_dir is None:
            dist_dir = tmp_root / "dist"
            wheel = build_wheel(REPO_ROOT, dist_dir)
        else:
            wheel = find_wheel(dist_dir)
        python = create_venv(tmp_root / "venv", args.python)
        install_wheel(python, wheel)
        work = tmp_root / "work"
        work.mkdir()
        demo = copy_demo(REPO_ROOT, work / "demo")
        run_smoke(python, demo, work, REPO_ROOT / "src" / "rolloutscope")
        print(f"packaging smoke passed: {wheel.name} with {python}")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
