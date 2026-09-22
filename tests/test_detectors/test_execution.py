"""Coverage denominators, execution failures, and exact detector measurements."""

from __future__ import annotations

import pytest

from rolloutscope.detectors import (
    AnswerLeakageEchoDetector,
    DegenerateRepetitionDetector,
    DetectorConfig,
    FormatOnlyWinsDetector,
    LengthInflationDetector,
    RewardSaturationGroupCollapseDetector,
    VerifierTamperDetector,
    builtin_detectors,
    execute_detector,
)
from rolloutscope.detectors._text import stable_rollout_id
from rolloutscope.schema import MultiTurnRollout, SingleTurnRollout


def row(**updates):
    """Build a normalized candidate with explicit overrides."""
    return SingleTurnRollout.model_validate(
        {
            "example_id": 1,
            "reward": 1.0,
            "completion": "A useful answer.",
            "is_completed": True,
            "is_truncated": False,
            **updates,
        }
    )


def assert_partition(execution):
    for unit in execution.units:
        assert unit.candidate == unit.eligible + unit.insufficient_data + unit.skipped + unit.errors
        assert unit.eligible == unit.fired + unit.clean


def test_mixed_format_metrics_are_accounted_per_row():
    rows = [
        row(metrics={"format": 1.0, "accuracy": 0.0}),
        row(metrics={"format": 1.0, "accuracy": 1.0}),
        row(metrics={"format": 1.0}),
        row(metrics={"accuracy": 1.0}),
        row(metrics={}),
    ]
    verdicts, execution = execute_detector(FormatOnlyWinsDetector(), rows, DetectorConfig())
    unit = execution.units[0]
    assert (unit.candidate, unit.eligible, unit.fired, unit.clean, unit.insufficient_data) == (
        5,
        2,
        1,
        1,
        3,
    )
    assert verdicts[0].measurements["format_value"] == 1.0
    assert verdicts[0].measurements["correctness_value"] == 0.0
    assert_partition(execution)


def test_metric_families_on_different_rows_are_insufficient_not_clean():
    verdicts, execution = execute_detector(
        FormatOnlyWinsDetector(),
        [row(metrics={"format": 1.0}), row(metrics={"accuracy": 0.0})],
        DetectorConfig(),
    )
    assert not any(v.fired for v in verdicts)
    assert execution.status == "insufficient_data"
    assert execution.units[0].eligible == 0


def test_repetition_distinguishes_skips_from_missing_tokens():
    rows = [
        row(completion="repeat me " * 40),
        row(completion=" ".join(f"word{i}" for i in range(80))),
        row(reward=0.1, completion="repeat me " * 40),
        row(completion="short"),
        MultiTurnRollout(
            example_id=1,
            reward=1.0,
            completion="repeat me " * 40,
            trajectory=[],
            is_completed=True,
            is_truncated=False,
        ),
    ]
    verdicts, execution = execute_detector(DegenerateRepetitionDetector(), rows, DetectorConfig())
    unit = execution.units[0]
    assert (
        unit.candidate,
        unit.eligible,
        unit.fired,
        unit.clean,
        unit.insufficient_data,
        unit.skipped,
    ) == (5, 2, 1, 1, 1, 2)
    assert unit.reason_counts == {
        "reward_below_minimum": 1,
        "too_few_tokens": 1,
        "multi_turn_transcript": 1,
    }
    assert verdicts[0].measurements["distinct_token_ratio"] == pytest.approx(2 / 80)
    assert_partition(execution)


def test_min_tokens_below_ngram_size_still_reports_insufficient():
    config = DetectorConfig.model_validate(
        {"degenerate_repetition": {"min_tokens": 1, "ngram_n": 5}}
    )
    _, execution = execute_detector(
        DegenerateRepetitionDetector(), [row(completion="three words only")], config
    )
    assert execution.units[0].reason_counts == {"too_few_tokens_for_ngram": 1}
    assert execution.units[0].eligible == 0


def test_missing_answer_context_and_missing_output_are_distinct():
    _, execution = execute_detector(
        AnswerLeakageEchoDetector(),
        [row(), row(completion=None), row(answer="different long answer")],
        DetectorConfig(),
    )
    assert execution.units[0].reason_counts == {
        "missing_answer_or_criterion": 1,
        "missing_completion": 1,
    }
    assert execution.units[0].clean == 1
    _, tamper = execute_detector(VerifierTamperDetector(), [row(completion=None)], DetectorConfig())
    assert tamper.status == "insufficient_data"


@pytest.mark.parametrize("name", list(builtin_detectors()))
def test_all_builtins_account_for_empty_and_fixture_inputs(name, load_labeled):
    detector = builtin_detectors()[name]
    _, empty = execute_detector(detector, [], DetectorConfig())
    assert empty.status == "insufficient_data"
    assert all(unit.eligible == 0 for unit in empty.units)
    for kind in ("hacked", "clean"):
        rows = load_labeled(f"{name}_{kind}")
        verdicts, execution = execute_detector(detector, rows, DetectorConfig())
        assert execution.status != "failed", execution.errors
        assert execution.verdict_count == len(verdicts)
        assert_partition(execution)
        assert all(v.measurements for v in verdicts if v.fired)
        source_run_ids = {row.run_id for row in rows}
        expected_run_id = next(iter(source_run_ids)) if len(source_run_ids) == 1 else None
        assert all(verdict.run_id == expected_run_id for verdict in verdicts if verdict.fired)
        assert all(
            any(unit.mode == verdict.mode and unit.unit == verdict.unit for unit in execution.units)
            for verdict in verdicts
        )
    assert "implementation:1" in execution.version


def test_aggregate_modes_have_independent_units(load_labeled):
    length_rows = load_labeled("length_inflation_hacked")
    length_verdicts, length = execute_detector(
        LengthInflationDetector(), length_rows, DetectorConfig()
    )
    assert [(u.unit, u.mode, u.eligible, u.fired) for u in length.units] == [
        ("run", "snapshot", 1, 1),
        ("run", "trend", 1, 1),
    ]
    assert all(verdict.unit == "run" for verdict in length_verdicts)
    assert all(verdict.run_id == length_rows[0].run_id for verdict in length_verdicts)
    saturation_rows = load_labeled("reward_saturation_group_collapse_hacked")
    saturation_verdicts, saturation = execute_detector(
        RewardSaturationGroupCollapseDetector(),
        saturation_rows,
        DetectorConfig(),
    )
    assert [(u.unit, u.mode, u.eligible, u.fired) for u in saturation.units] == [
        ("group", "group", 9, 6),
        ("run", "trend", 1, 1),
    ]
    assert {verdict.unit for verdict in saturation_verdicts if verdict.mode == "group"} == {"group"}
    assert {verdict.unit for verdict in saturation_verdicts if verdict.mode == "trend"} == {"run"}
    assert all(verdict.run_id == saturation_rows[0].run_id for verdict in saturation_verdicts)


def test_correlation_degeneracy_is_not_a_clean_check():
    rows = [row(completion="same length") for _ in range(10)]
    _, execution = execute_detector(LengthInflationDetector(), rows, DetectorConfig())
    assert execution.units[0].reason_counts == {"undefined_correlation": 1}
    assert execution.units[0].eligible == 0
    assert execution.units[1].reason_counts == {"missing_step_index": 1}


def test_group_minimums_are_real_denominator_gates():
    config = DetectorConfig()
    rows = [row(group_id="a"), row(group_id="a"), row(group_id="b")]
    _, execution = execute_detector(RewardSaturationGroupCollapseDetector(), rows, config)
    assert execution.units[0].candidate == 2
    assert execution.units[0].eligible == 0
    assert execution.units[0].reason_counts == {"too_few_groups": 1, "too_few_group_members": 1}


def test_detector_failure_cannot_leave_clean_checks(monkeypatch):
    def broken(self, rollouts, config):
        raise RuntimeError("deliberate detector failure")

    monkeypatch.setattr(VerifierTamperDetector, "detect", broken)
    verdicts, execution = execute_detector(
        VerifierTamperDetector(), [row(), row(completion=None)], DetectorConfig()
    )
    assert verdicts == []
    assert execution.status == "failed"
    assert execution.errors == ["RuntimeError: deliberate detector failure"]
    assert execution.units[0].errors == 1
    assert execution.units[0].insufficient_data == 1
    assert execution.units[0].clean == 0
    assert_partition(execution)


def test_unknown_plugin_coverage_is_not_clean():
    class Plugin:
        name = "third_party"
        category = "custom"

        def detect(self, rollouts, config):
            return []

    _, execution = execute_detector(Plugin(), [row()], DetectorConfig())
    assert execution.status == "unsupported"
    assert execution.units == []
    assert execution.version.startswith("unknown:unknown:")


def test_plugin_metadata_error_is_a_visible_failed_execution():
    class Plugin:
        @property
        def name(self):
            raise RuntimeError("broken name")

        category = "custom"

        def detect(self, rollouts, config):
            return []

    verdicts, execution = execute_detector(Plugin(), [row()], DetectorConfig())
    assert verdicts == []
    assert execution.status == "failed"
    assert execution.units[0].errors == 1
    assert execution.errors == ["detector metadata error: RuntimeError: broken name"]


def test_plugin_version_error_preserves_selected_name_in_failed_execution():
    class Plugin:
        name = "third_party"
        category = "custom"

        @property
        def version(self):
            raise RuntimeError("broken version")

        def detect(self, rollouts, config):
            return []

    verdicts, execution = execute_detector(Plugin(), [row()], DetectorConfig())
    assert verdicts == []
    assert execution.detector == "third_party"
    assert execution.category == "custom"
    assert execution.status == "failed"
    assert execution.config == DetectorConfig().model_dump(mode="json")
    assert execution.errors == ["detector metadata error: RuntimeError: broken version"]


def test_occurrence_identity_wins_and_duplicate_generations_remain_distinct():
    first = row(rollout_id="same-content", occurrence_id="source-line-1", completion="assert True")
    second = first.model_copy(update={"occurrence_id": "source-line-2"})
    verdicts, execution = execute_detector(
        VerifierTamperDetector(), [first, second], DetectorConfig()
    )
    assert [v.rollout_ids for v in verdicts] == [["source-line-1"], ["source-line-2"]]
    assert execution.units[0].fired == 2
    assert stable_rollout_id(first.model_copy(update={"occurrence_id": None})) == "same-content"


def test_group_membership_and_trends_never_pool_independent_runs():
    rows = [row(run_id="run-a", group_id="same"), row(run_id="run-b", group_id="same")]
    _, execution = execute_detector(RewardSaturationGroupCollapseDetector(), rows, DetectorConfig())
    assert len(execution.units) == 4
    assert all(unit.eligible == 0 for unit in execution.units)
    assert {unit.measurements["run_id"] for unit in execution.units} == {"run-a", "run-b"}
    rows = [row(run_id=f"run-{i}", step_index=i, completion="x" * (i + 1)) for i in range(3)]
    _, length = execute_detector(LengthInflationDetector(), rows, DetectorConfig())
    assert len(length.units) == 6
    assert all(unit.eligible == 0 for unit in length.units)
