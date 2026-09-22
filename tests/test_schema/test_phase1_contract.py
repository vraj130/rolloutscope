"""Version 2 contracts for finite values, occurrence identity, and safe serialization."""

import copy
from pathlib import Path

import orjson
import pytest
from pydantic import ValidationError

from rolloutscope.schema import (
    ROLLOUT_ADAPTER,
    EvidenceSpan,
    SingleTurnRollout,
    SourceOccurrence,
    Verdict,
    attach_identity,
    content_fingerprint,
    read_rollouts,
    rollout_json_schema,
    scoring_revision,
    validate_rollout,
    write_rollouts,
)
from rolloutscope.schema.execution import ExecutionManifest, FileIngestion, UnitCounts

RAW = {
    "example_id": 0,
    "prompt": "question",
    "completion": "answer",
    "reward": 1.0,
    "metrics": {"accuracy": 1.0},
    "is_completed": True,
    "is_truncated": False,
}


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize(
    "location", ["reward", "metrics", "timing", "token_usage", "extra", "message"]
)
def test_nonfinite_values_rejected_including_nested_extras(value: float, location: str) -> None:
    row = copy.deepcopy(RAW)
    if location == "reward":
        row["reward"] = value
    elif location == "metrics":
        row["metrics"] = {"accuracy": value}
    elif location == "timing":
        row["timing"] = {"model": {"spans": [{"duration": value}]}}
    elif location == "token_usage":
        row["token_usage"] = {"output_tokens": value}
    elif location == "extra":
        row["custom"] = {"nested": [value]}
    else:
        row["completion"] = [{"role": "assistant", "content": "answer", "provider": value}]
    with pytest.raises(ValidationError):
        validate_rollout(row)


@pytest.mark.parametrize("version", ["1.0", "3.0", "garbage"])
def test_direct_models_and_union_cannot_bypass_version_policy(version: str) -> None:
    with pytest.raises(ValidationError):
        SingleTurnRollout(**{**RAW, "schema_version": version})
    with pytest.raises(ValidationError):
        ROLLOUT_ADAPTER.validate_python({**RAW, "kind": "single_turn", "schema_version": version})


def test_writer_emits_current_version_and_preserves_future_minor_fields(tmp_path: Path) -> None:
    row = validate_rollout({**RAW, "schema_version": "2.7", "future_field": [1, 2]})
    path = tmp_path / "rows.jsonl"
    write_rollouts(path, [row])
    payload = orjson.loads(path.read_bytes())
    assert payload["schema_version"] == "2.0"
    assert payload["future_field"] == [1, 2]
    assert len(list(read_rollouts(path))) == 1


def test_content_changes_with_trajectory_but_not_scores_or_source_ids() -> None:
    data = {
        **RAW,
        "trajectory": [{"trajectory_id": "turn-a", "completion": "tool action", "reward": 1.0}],
    }
    original = validate_rollout(data)
    rescored_data = copy.deepcopy(data)
    rescored_data["reward"] = 0.1
    rescored_data["metrics"] = {"accuracy": 0.1}
    rescored_data["trajectory"][0]["reward"] = 0.1
    rescored_data["trajectory"][0]["trajectory_id"] = "turn-b"
    rescored = validate_rollout(rescored_data)
    assert content_fingerprint(rescored) == content_fingerprint(original)
    assert scoring_revision(rescored) != scoring_revision(original)
    rescored_data["trajectory"][0]["completion"] = "different action"
    assert content_fingerprint(validate_rollout(rescored_data)) != content_fingerprint(original)


def test_rescoring_normalized_row_refreshes_score_without_replacing_occurrence() -> None:
    original = attach_identity(
        validate_rollout(RAW),
        run_id="run",
        source_path="results.jsonl",
        line=2,
        adapter="raw",
        step_index=4,
    )
    updated = validate_rollout({**original.model_dump(), "reward": 0.1})
    updated = attach_identity(
        updated,
        run_id="ignored",
        source_path="export.jsonl",
        line=10,
        adapter="normalized",
        step_index=None,
        preserve=True,
    )
    assert updated.occurrence_id == original.occurrence_id
    assert updated.provenance == original.provenance
    assert updated.content_fingerprint == original.content_fingerprint
    assert updated.scoring_revision != original.scoring_revision
    assert updated.step_index == 4


def test_writer_preserves_existing_output_when_iteration_fails(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_bytes(b"previous output\n")

    def broken_rows():
        yield validate_rollout(RAW)
        raise RuntimeError("source failed")

    with pytest.raises(RuntimeError, match="source failed"):
        write_rollouts(path, broken_rows())
    assert path.read_bytes() == b"previous output\n"
    assert list(tmp_path.iterdir()) == [path]


def test_writer_rejects_input_collision(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    original = orjson.dumps(RAW) + b"\n"
    path.write_bytes(original)
    with pytest.raises(ValueError):
        write_rollouts(path, read_rollouts(path), source_paths=[path])
    assert path.read_bytes() == original


def test_reader_reports_nonfinite_numeric_source_location(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text(
        '{"example_id":0,"reward":"Infinity","is_completed":true,"is_truncated":false}\n'
    )
    counts = FileIngestion(path=path.name)
    assert list(read_rollouts(path, diagnostics=counts)) == []
    assert counts.rejected == 1
    assert counts.examples[0].line == 1
    assert "reward" in counts.examples[0].message


def test_rejected_large_integer_cannot_reach_writer() -> None:
    with pytest.raises(ValidationError, match="64-bit"):
        validate_rollout({**RAW, "extra": 2**100})


@pytest.mark.parametrize("value", [-(2**63), 2**63 - 1, 2**64 - 1])
def test_json_writer_integer_boundaries_remain_valid(value: int) -> None:
    rollout = validate_rollout({**RAW, "provider_extra": {"integer": value}})
    verdict = Verdict(
        detector="test",
        fired=False,
        score=0.0,
        category="test",
        measurements={"integer": value},
    )
    assert orjson.loads(orjson.dumps(rollout.model_dump(mode="json")))["provider_extra"] == {
        "integer": value
    }
    assert orjson.loads(orjson.dumps(verdict.model_dump(mode="json")))["measurements"] == {
        "integer": value
    }


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        {"not-json"},
        {1: "non-string-key"},
        object(),
        b"bytes",
        2**100,
    ],
)
def test_arbitrary_rollout_values_must_be_json_safe(value: object) -> None:
    with pytest.raises(ValidationError):
        validate_rollout({**RAW, "provider_extra": {"nested": [value]}})


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        {"not-json"},
        {1: "non-string-key"},
        object(),
        b"bytes",
        2**100,
    ],
)
def test_verdict_and_execution_measurements_must_be_json_safe(value: object) -> None:
    evidence = [EvidenceSpan(rollout_id="occ-1", field="reward", text="1.0")]
    with pytest.raises(ValidationError):
        Verdict(
            detector="test",
            fired=True,
            score=0.5,
            category="test",
            evidence=evidence,
            rollout_ids=["occ-1"],
            measurements={"nested": [value]},
        )
    with pytest.raises(ValidationError):
        UnitCounts(candidate=1, eligible=1, clean=1, measurements={"nested": [value]})
    with pytest.raises(ValidationError):
        ExecutionManifest(
            tool_version="test",
            schema_version="2.0",
            input_format="normalized",
            effective_config={"nested": [value]},
        )


def test_verdict_provenance_and_modes_are_typed_and_exported() -> None:
    verdict = Verdict(
        detector="test",
        fired=True,
        score=0.5,
        category="test",
        evidence=[EvidenceSpan(rollout_id="occ-1", field="reward", text="1.0")],
        rollout_ids=["occ-1"],
        mode="trend",
        unit="run",
        run_id="run-1",
        source_occurrences=[
            SourceOccurrence(occurrence_id="occ-1", source_path="step_1/rows.jsonl", line=4)
        ],
        measurements={"correlation": 0.75, "steps": [0, 1]},
    )
    payload = verdict.model_dump(mode="json")
    assert payload["mode"] == "trend"
    assert payload["unit"] == "run"
    assert payload["source_occurrences"] == [
        {"occurrence_id": "occ-1", "source_path": "step_1/rows.jsonl", "line": 4}
    ]
    assert orjson.loads(orjson.dumps(payload)) == payload
    properties = Verdict.model_json_schema()["properties"]
    assert {"mode", "unit", "run_id", "source_occurrences", "measurements"} <= properties.keys()


def test_rollout_and_execution_schemas_expose_identity_namespaces() -> None:
    rollout_schema = rollout_json_schema()
    single_turn = rollout_schema["$defs"]["SingleTurnRollout"]["properties"]
    assert {"environment_namespace", "task_namespace"} <= single_turn.keys()
    execution = ExecutionManifest(
        tool_version="test",
        schema_version="2.0",
        input_format="normalized",
        environment_namespaces=["owner/environment"],
        task_namespaces=["owner/task-set"],
    )
    assert execution.model_dump(mode="json")["environment_namespaces"] == ["owner/environment"]
    execution_properties = ExecutionManifest.model_json_schema()["properties"]
    assert {"environment_namespaces", "task_namespaces"} <= execution_properties.keys()


def test_identity_namespaces_fall_back_and_preserve_explicit_values() -> None:
    fallback = attach_identity(
        validate_rollout(RAW),
        run_id="run-1",
        source_path="results.jsonl",
        line=1,
        adapter="raw",
        step_index=None,
    )
    assert fallback.environment_namespace == "run-1"
    assert fallback.task_namespace == "run-1"

    namespaced = attach_identity(
        validate_rollout(RAW),
        run_id="run-1",
        source_path="results.jsonl",
        line=1,
        adapter="raw",
        step_index=None,
        environment_namespace="owner/environment",
        task_namespace="owner/task-set",
    )
    assert namespaced.environment_namespace == "owner/environment"
    assert namespaced.task_namespace == "owner/task-set"
    preserved = attach_identity(
        namespaced,
        run_id="ignored-run",
        source_path="export.jsonl",
        line=99,
        adapter="normalized",
        step_index=None,
        environment_namespace="ignored-environment",
        task_namespace="ignored-task",
        preserve=True,
    )
    assert preserved.environment_namespace == "owner/environment"
    assert preserved.task_namespace == "owner/task-set"
