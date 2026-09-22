"""Performance result format and a tiny end-to-end measurement."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.script_loader import REPO_ROOT, load_script

SCHEMA_PATH = REPO_ROOT / "scripts" / "perf" / "result_schema.json"


def _generate():
    return load_script("scripts/perf/generate.py")


def _measure():
    return load_script("scripts/perf/measure.py")


def _tiny(generate):
    return generate.SizeSpec(
        name="tiny",
        steps=2,
        groups=2,
        rollouts_per_group=2,
        multi_turn_fraction=0.5,
        long_completion_fraction=0.0,
        turns=2,
        seed=3,
    )


def test_rss_to_bytes_uses_platform_units() -> None:
    measure = _measure()
    assert measure.rss_to_bytes(100, "linux") == 100 * 1024
    assert measure.rss_to_bytes(100, "darwin") == 100


def test_result_schema_file_exists_and_declares_version_1() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["properties"]["schema_version"]["const"] == "1.0"
    required = set(schema["required"])
    required_keys = {
        "schema_version",
        "tool_version",
        "python_version",
        "platform",
        "generator",
        "input",
        "run",
    }
    assert required_keys <= required


def test_validate_result_rejects_missing_peak_rss() -> None:
    document = {
        "schema_version": "1.0",
        "tool_version": "0.1.0",
        "python_version": "3.12.0",
        "platform": "darwin",
        "generator": {"size": "tiny", "seed": 1},
        "input": {
            "bytes": 10,
            "record_count": 1,
            "mean_trajectory_turns": 0.0,
            "group_cardinality": 1,
            "step_count": 1,
            "kind_mix": {"single_turn": 1, "multi_turn": 0},
        },
        "run": {
            "elapsed_seconds": 0.1,
            "html_bytes": 1,
            "json_bytes": 1,
            "detector_mix": ["verifier_tamper"],
            "exit_code": 0,
        },
    }
    measure = _measure()
    with pytest.raises(ValueError, match="peak_rss_bytes"):
        measure.validate_result(document)


def test_measure_analyze_on_tiny_generated_run(tmp_path: Path) -> None:
    generate = _generate()
    measure = _measure()
    spec = _tiny(generate)
    dataset = tmp_path / "run"
    out_dir = tmp_path / "out"
    generate.generate_dataset(dataset, spec)
    result = measure.measure_analyze(
        dataset,
        out_dir,
        size_name=spec.name,
        seed=spec.seed,
        rolloutscope_bin="rolloutscope",
    )
    measure.validate_result(result)
    assert result["generator"]["size"] == "tiny"
    assert result["input"]["record_count"] == 8
    assert result["run"]["exit_code"] == 0
    assert result["run"]["html_bytes"] > 0
    assert result["run"]["json_bytes"] > 0
    assert result["run"]["elapsed_seconds"] > 0
    assert result["run"]["peak_rss_bytes"] > 0
    assert len(result["run"]["detector_mix"]) == 6
    assert (out_dir / "report.html").is_file()
    assert (out_dir / "findings.json").is_file()
    assert (out_dir / "perf.json").is_file()
