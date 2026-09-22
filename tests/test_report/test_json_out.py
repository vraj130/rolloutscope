"""Determinism of the JSON report output (CI diffing depends on it)."""

import json
from collections.abc import Callable
from pathlib import Path

from rolloutscope.analysis import aggregate_rollouts
from rolloutscope.report import InputFile, ReportData, render_json_bytes, write_verdicts
from rolloutscope.schema import EvidenceSpan, Finding, SingleTurnRollout, Verdict

RolloutFactory = Callable[..., SingleTurnRollout]


def _build_report(
    make_rollout: RolloutFactory,
    metrics: dict[str, float],
    config_used: dict[str, object],
) -> ReportData:
    rollouts = [make_rollout(0.5, 0), make_rollout(1.0, 1)]
    finding = Finding(
        severity="warning",
        title="length_inflation: 1 of 2 checks fired (rubric_exploit)",
        description="test finding",
        detector="length_inflation",
        metrics=metrics,
        config_used=dict(config_used),
        exemplars=[],
    )
    return ReportData(
        tool_version="0.1.0",
        schema_version="1.0",
        input_files=[InputFile(name="results.jsonl", sha256="ab" * 32, size_bytes=123)],
        aggregates=aggregate_rollouts(iter(rollouts)),
        findings=[finding],
    )


def test_render_twice_is_byte_identical(make_rollout: RolloutFactory) -> None:
    report = _build_report(make_rollout, {"a": 1.0, "b": 2.0}, {"t": 0.5})
    assert render_json_bytes(report) == render_json_bytes(report)


def test_dict_insertion_order_does_not_change_bytes(make_rollout: RolloutFactory) -> None:
    forward = _build_report(make_rollout, {"a": 1.0, "b": 2.0}, {"t": 0.5, "u": 1})
    reversed_order = _build_report(make_rollout, {"b": 2.0, "a": 1.0}, {"u": 1, "t": 0.5})
    assert render_json_bytes(forward) == render_json_bytes(reversed_order)


def test_trailing_newline_and_valid_json(make_rollout: RolloutFactory) -> None:
    payload = render_json_bytes(_build_report(make_rollout, {"a": 1.0}, {}))
    assert payload.endswith(b"\n")
    assert not payload.endswith(b"\n\n")
    document = json.loads(payload)
    assert document["tool_version"] == "0.1.0"
    assert document["aggregates"]["run_summary"]["row_count"] == 2


def test_every_object_has_sorted_keys(make_rollout: RolloutFactory) -> None:
    payload = render_json_bytes(_build_report(make_rollout, {"b": 2.0, "a": 1.0}, {}))
    key_lists: list[list[str]] = []

    def record(pairs: list[tuple[str, object]]) -> dict[str, object]:
        key_lists.append([key for key, _ in pairs])
        return dict(pairs)

    json.loads(payload, object_pairs_hook=record)
    assert key_lists, "expected at least one JSON object"
    for keys in key_lists:
        assert keys == sorted(keys)


def test_write_verdicts_keeps_complete_deterministic_jsonl(
    make_rollout: RolloutFactory, tmp_path: Path
) -> None:
    report = _build_report(make_rollout, {}, {}).model_copy(
        update={
            "verdicts": [
                Verdict(
                    detector="detector",
                    fired=True,
                    score=0.8,
                    category="test",
                    rollout_ids=["r1"],
                    evidence=[EvidenceSpan(rollout_id="r1", field="completion", text="proof")],
                )
            ]
        }
    )
    path = tmp_path / "verdicts.jsonl"
    assert write_verdicts(report, path) == path
    lines = path.read_bytes().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["evidence"][0]["text"] == "proof"
