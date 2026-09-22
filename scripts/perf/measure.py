"""Measure analyze() wall time, peak RSS, and output size on a generated run.

This script does not import rolloutscope. It subprocesses the CLI so the
recorded RSS is the analysis process, not the generator. Results follow
``result_schema.json`` version 1.0. There is no pass/fail budget: this is a
baseline recorder, not an optimizer.

Usage:

    uv run python scripts/perf/generate.py --size small --out /tmp/rs-small
    uv run python scripts/perf/measure.py --input /tmp/rs-small --out /tmp/rs-small-out
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

RESULT_SCHEMA_VERSION = "1.0"
_PERF_PREFIX = "PERF_STATS:"
_WRAPPER = f"""
import json, resource, subprocess, sys, time
cmd = sys.argv[1:]
start = time.perf_counter()
completed = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
elapsed = time.perf_counter() - start
usage = resource.getrusage(resource.RUSAGE_CHILDREN)
print(
    {_PERF_PREFIX!r} + json.dumps({{
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        "ru_maxrss": usage.ru_maxrss,
        "stderr": (completed.stderr or "")[-2000:],
    }}),
    flush=True,
)
"""

_TOP_REQUIRED = (
    "schema_version",
    "tool_version",
    "python_version",
    "platform",
    "generator",
    "input",
    "run",
)
_INPUT_REQUIRED = (
    "bytes",
    "record_count",
    "mean_trajectory_turns",
    "group_cardinality",
    "step_count",
    "kind_mix",
)
_RUN_REQUIRED = (
    "elapsed_seconds",
    "peak_rss_bytes",
    "html_bytes",
    "json_bytes",
    "detector_mix",
    "exit_code",
)
_GENERATOR_REQUIRED = ("size", "seed")


def _generator_identity(
    dataset: Path,
    size_name: str | None,
    seed: int | None,
) -> tuple[str, int]:
    """Return size name and seed, filling gaps from ``generator.json`` when present."""
    meta_path = dataset / "generator.json"
    meta: dict[str, Any] = {}
    if meta_path.is_file():
        loaded = json.loads(meta_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            meta = loaded
    resolved_size = size_name if size_name is not None else str(meta.get("size", "custom"))
    resolved_seed = seed if seed is not None else int(meta.get("seed", 0))
    return resolved_size, resolved_seed


def rss_to_bytes(ru_maxrss: int, platform: str) -> int:
    """Convert ``resource.ru_maxrss`` to bytes.

    Linux reports kilobytes; macOS (darwin) reports bytes. Other platforms are
    treated like Linux, which is the conservative direction for published numbers.
    """
    name = platform.lower()
    if name.startswith("darwin") or name == "macos":
        return ru_maxrss
    return ru_maxrss * 1024


def validate_result(document: dict[str, Any]) -> None:
    """Reject a measurement document that is missing required 1.0 fields."""
    if document.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {RESULT_SCHEMA_VERSION!r}, "
            f"got {document.get('schema_version')!r}"
        )
    for key in _TOP_REQUIRED:
        if key not in document:
            raise ValueError(f"missing {key}")
    for key in _GENERATOR_REQUIRED:
        if key not in document["generator"]:
            raise ValueError(f"missing generator.{key}")
    for key in _INPUT_REQUIRED:
        if key not in document["input"]:
            raise ValueError(f"missing input.{key}")
    for key in _RUN_REQUIRED:
        if key not in document["run"]:
            raise ValueError(f"missing {key}")


def summarize_input(dataset: Path) -> dict[str, Any]:
    """Walk a generated run directory and return the input block of a result."""
    records = 0
    groups: set[int] = set()
    kind_mix = {"single_turn": 0, "multi_turn": 0}
    turns_total = 0
    input_bytes = 0
    step_dirs = {
        path.name for path in dataset.iterdir() if path.is_dir() and path.name.startswith("step_")
    }
    for path in sorted(dataset.rglob("*.jsonl")):
        input_bytes += path.stat().st_size
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            records += 1
            groups.add(int(row["example_id"]))
            trajectory = row.get("trajectory") or []
            if trajectory:
                kind_mix["multi_turn"] += 1
                turns_total += len(trajectory)
            else:
                kind_mix["single_turn"] += 1
    mean_turns = turns_total / records if records else 0.0
    return {
        "bytes": input_bytes,
        "record_count": records,
        "mean_trajectory_turns": mean_turns,
        "group_cardinality": len(groups),
        "step_count": len(step_dirs),
        "kind_mix": kind_mix,
    }


def _parse_perf_stats(stdout: str, stderr: str) -> dict[str, Any]:
    for line in stdout.splitlines():
        if line.startswith(_PERF_PREFIX):
            payload: dict[str, Any] = json.loads(line[len(_PERF_PREFIX) :])
            return payload
    raise RuntimeError(f"no {_PERF_PREFIX} in wrapper output\nstdout:\n{stdout}\nstderr:\n{stderr}")


def _detector_mix(findings_path: Path) -> list[str]:
    document = json.loads(findings_path.read_text(encoding="utf-8"))
    results = document.get("detector_results") or []
    names = sorted({item["detector"] for item in results if "detector" in item})
    if names:
        return names
    findings = document.get("findings") or []
    return sorted({item["detector"] for item in findings if "detector" in item})


def measure_analyze(
    dataset: Path,
    out_dir: Path,
    *,
    size_name: str,
    seed: int,
    rolloutscope_bin: str = "rolloutscope",
) -> dict[str, Any]:
    """Run ``rolloutscope analyze`` on ``dataset`` and return a versioned result."""
    out_dir.mkdir(parents=True, exist_ok=True)
    html = out_dir / "report.html"
    json_out = out_dir / "findings.json"
    binary = shutil.which(rolloutscope_bin) or rolloutscope_bin
    analyze_cmd = [
        binary,
        "analyze",
        str(dataset.resolve()),
        "--out",
        str(html),
        "--json",
        str(json_out),
        "--quiet",
        "--fail-on",
        "none",
    ]
    env = os.environ.copy()
    completed = subprocess.run(
        [sys.executable, "-c", _WRAPPER, *analyze_cmd],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=out_dir,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"measurement wrapper failed ({completed.returncode})\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    stats = _parse_perf_stats(completed.stdout, completed.stderr)
    if stats["returncode"] != 0:
        raise RuntimeError(
            f"rolloutscope analyze failed ({stats['returncode']})\n{stats.get('stderr', '')}"
        )
    findings = json.loads(json_out.read_text(encoding="utf-8"))
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "tool_version": findings.get("tool_version", "unknown"),
        "python_version": sys.version.split()[0],
        "platform": sys.platform,
        "generator": {"size": size_name, "seed": seed},
        "input": summarize_input(dataset),
        "run": {
            "elapsed_seconds": float(stats["elapsed_seconds"]),
            "peak_rss_bytes": rss_to_bytes(int(stats["ru_maxrss"]), sys.platform),
            "html_bytes": html.stat().st_size,
            "json_bytes": json_out.stat().st_size,
            "detector_mix": _detector_mix(json_out),
            "exit_code": int(stats["returncode"]),
        },
    }
    validate_result(result)
    (out_dir / "perf.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    """CLI: measure analyze on ``--input`` and write ``perf.json`` under ``--out``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Generated run directory.")
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Directory for HTML, JSON, and perf.json.",
    )
    parser.add_argument(
        "--size-name",
        default=None,
        help="Recorded generator size name (default: generator.json next to the input).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Recorded generator seed (default: generator.json next to the input).",
    )
    parser.add_argument("--bin", default="rolloutscope", dest="rolloutscope_bin")
    args = parser.parse_args(argv)
    size_name, seed = _generator_identity(args.input, args.size_name, args.seed)
    result = measure_analyze(
        args.input,
        args.out,
        size_name=size_name,
        seed=seed,
        rolloutscope_bin=args.rolloutscope_bin,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
