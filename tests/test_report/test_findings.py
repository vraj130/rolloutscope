"""Finding assembly from hand-built Verdicts against the frozen contract."""

from collections.abc import Callable

import pytest

from rolloutscope.analysis import SeverityThresholds, assemble_findings, severity_for_score
from rolloutscope.schema import Verdict
from rolloutscope.schema.execution import DetectorExecution, UnitCounts

VerdictFactory = Callable[..., Verdict]


def test_severity_mapping_defaults(make_verdict: VerdictFactory) -> None:
    for score, expected in ((0.9, "critical"), (0.6, "warning"), (0.3, "info")):
        findings = assemble_findings([make_verdict(score=score)])
        assert len(findings) == 1
        assert findings[0].severity == expected


def test_severity_thresholds_are_inclusive(make_verdict: VerdictFactory) -> None:
    assert assemble_findings([make_verdict(score=0.8)])[0].severity == "critical"
    assert assemble_findings([make_verdict(score=0.5)])[0].severity == "warning"
    assert assemble_findings([make_verdict(score=0.49999)])[0].severity == "info"


def test_severity_thresholds_configurable(make_verdict: VerdictFactory) -> None:
    thresholds = SeverityThresholds(critical_at=0.95, warning_at=0.9)
    findings = assemble_findings([make_verdict(score=0.92)], thresholds=thresholds)
    assert findings[0].severity == "warning"
    assert severity_for_score(0.96, thresholds) == "critical"
    assert severity_for_score(0.1, thresholds) == "info"


def test_max_fired_score_drives_severity(make_verdict: VerdictFactory) -> None:
    verdicts = [
        make_verdict(score=0.3),
        make_verdict(score=0.85),
        make_verdict(score=0.2, fired=False),
    ]
    findings = assemble_findings(verdicts)
    assert len(findings) == 1
    finding = findings[0]
    assert finding.severity == "critical"
    assert finding.metrics["max_score"] == pytest.approx(0.85)
    assert finding.metrics["fired_count"] == 2.0
    assert finding.metrics["total_verdicts"] == 3.0
    assert finding.metrics["fired_rate"] == pytest.approx(2.0 / 3.0)


def test_grouping_by_detector_and_category(make_verdict: VerdictFactory) -> None:
    verdicts = [
        make_verdict(detector="det_a", category="cat_x", score=0.9),
        make_verdict(detector="det_a", category="cat_y", score=0.3),
        make_verdict(detector="det_b", category="cat_x", score=0.6),
    ]
    findings = assemble_findings(verdicts)
    assert len(findings) == 3
    # Sorted by severity rank, then detector.
    assert [(f.detector, f.severity) for f in findings] == [
        ("det_a", "critical"),
        ("det_b", "warning"),
        ("det_a", "info"),
    ]


def test_exemplars_are_top_n_spans_by_verdict_score(make_verdict: VerdictFactory) -> None:
    verdicts = [
        make_verdict(score=0.7, text="span mid"),
        make_verdict(score=0.9, text="span high"),
        make_verdict(score=0.8, text="span low"),
    ]
    findings = assemble_findings(verdicts, exemplar_limit=2)
    exemplars = findings[0].exemplars
    assert [span.text for span in exemplars] == ["span high", "span low"]


def test_zero_fired_skipped_by_default(make_verdict: VerdictFactory) -> None:
    verdicts = [make_verdict(fired=False, score=0.0) for _ in range(3)]
    assert assemble_findings(verdicts) == []


def test_zero_fired_yields_info_finding_when_requested(make_verdict: VerdictFactory) -> None:
    verdicts = [make_verdict(fired=False, score=0.0, detector="clean_det") for _ in range(3)]
    findings = assemble_findings(verdicts, include_clean=True)
    assert len(findings) == 1
    finding = findings[0]
    assert finding.severity == "info"
    assert "clean_det" in finding.title
    assert finding.metrics["fired_count"] == 0.0
    assert finding.metrics["total_verdicts"] == 3.0
    assert finding.exemplars == []


def test_config_used_recorded_from_caller(make_verdict: VerdictFactory) -> None:
    findings = assemble_findings(
        [make_verdict(detector="det_a", score=0.9)],
        config_used={"det_a": {"threshold": 0.4}, "other": {"ignored": True}},
    )
    assert findings[0].config_used == {"threshold": 0.4}


def test_flagged_rollouts_deduplicated(make_verdict: VerdictFactory) -> None:
    verdicts = [
        make_verdict(score=0.6, rollout_ids=["r1", "r2"]),
        make_verdict(score=0.7, rollout_ids=["r2", "r3"]),
    ]
    findings = assemble_findings(verdicts)
    assert findings[0].metrics["flagged_rollouts"] == 3.0


def test_thresholds_must_be_ordered() -> None:
    with pytest.raises(ValueError, match="warning_at"):
        SeverityThresholds(critical_at=0.4, warning_at=0.6)


def test_execution_coverage_supplies_the_real_denominator(
    make_verdict: VerdictFactory,
) -> None:
    verdict = make_verdict(detector="det_a", category="cat_x", score=0.7)
    execution = DetectorExecution(
        detector="det_a",
        category="cat_x",
        units=[UnitCounts(candidate=7, eligible=5, fired=1, clean=4, insufficient_data=2)],
        verdict_count=1,
    )
    finding = assemble_findings([verdict], executions=[execution])[0]
    assert finding.metrics["eligible_checks"] == 5.0
    assert finding.metrics["fired_rate"] == pytest.approx(0.2)
    assert finding.metrics["candidate_checks"] == 7.0
    assert finding.metrics["insufficient_data"] == 2.0


def test_all_flagged_ids_survive_bounded_exemplars(make_verdict: VerdictFactory) -> None:
    verdicts = [
        make_verdict(score=0.9 - index / 100, rollout_ids=[f"source-line-{index}"])
        for index in range(10)
    ]
    finding = assemble_findings(verdicts, exemplar_limit=2)[0]
    assert len(finding.exemplars) == 2
    assert finding.rollout_ids == [f"source-line-{index}" for index in range(10)]


@pytest.mark.parametrize("status", ["insufficient_data", "unsupported"])
def test_unchecked_execution_is_never_reported_clean(status: str) -> None:
    units = (
        [UnitCounts(candidate=1, insufficient_data=1, reason_counts={"missing_data": 1})]
        if status == "insufficient_data"
        else []
    )
    execution = DetectorExecution(
        detector="unavailable",
        category="cat_x",
        status=status,
        units=units,
    )
    assert assemble_findings([], executions=[execution], include_clean=True) == []


def test_failed_execution_discards_finding_even_if_verdict_was_supplied(
    make_verdict: VerdictFactory,
) -> None:
    execution = DetectorExecution(
        detector="det_a",
        category="cat_x",
        status="failed",
        units=[UnitCounts(candidate=1, errors=1)],
    )
    assert (
        assemble_findings(
            [make_verdict(detector="det_a", category="cat_x")], executions=[execution]
        )
        == []
    )


def test_same_mode_with_different_units_produces_distinct_findings(
    make_verdict: VerdictFactory,
) -> None:
    rollout_verdict = make_verdict(detector="mixed", category="cat_x").model_copy(
        update={"mode": "snapshot", "unit": "rollout"}
    )
    group_verdict = make_verdict(detector="mixed", category="cat_x").model_copy(
        update={"mode": "snapshot", "unit": "group"}
    )
    execution = DetectorExecution(
        detector="mixed",
        category="cat_x",
        units=[
            UnitCounts(unit="rollout", candidate=1, eligible=1, fired=1),
            UnitCounts(unit="group", candidate=1, eligible=1, fired=1),
        ],
        verdict_count=2,
    )
    findings = assemble_findings([rollout_verdict, group_verdict], executions=[execution])
    assert {(finding.mode, finding.unit) for finding in findings} == {
        ("snapshot", "rollout"),
        ("snapshot", "group"),
    }
