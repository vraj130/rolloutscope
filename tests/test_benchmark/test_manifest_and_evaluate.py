"""The manifest pins the population; the driver refuses to widen it."""

from __future__ import annotations

from pathlib import Path

import pytest

from rolloutscope.benchmark import (
    DatasetRef,
    Requirement,
    applicability,
    available_signals,
    build_manifest,
    evaluate_rows,
    load_manifest,
    map_row,
    population_signals,
    row_digest,
    save_manifest,
    verify,
)
from rolloutscope.benchmark.manifest import first_user_text, to_json
from rolloutscope.detectors import DetectorConfig, discover_detectors

from .conftest import assistant, trace_row, user

DATASET = DatasetRef(
    name="PatronusAI/trace-dataset",
    revision="deadbeef",
    license="cc-by-sa-4.0",
    gated=True,
    redistribution="restricted: identity and hashes only",
)


def _rows(count: int = 12) -> list[dict[str, object]]:
    """Build a small labelled population of TRACE-shaped rows."""
    rows = []
    for i in range(count):
        hacked = i % 2 == 0
        label = "1.1.1" if hacked else "0"
        text = "Editing the test." if hacked else "Fixing the implementation."
        path = "tests/test_a.py" if hacked else "src/a.py"
        rows.append(
            trace_row(
                f"trajectory_{i:04d}",
                label,
                [
                    user(f"Scenario {i}: make the suite pass."),
                    assistant(text, calls=[("Edit", {"path": path})], results=["ok"]),
                ],
            )
        )
    return rows


def test_a_manifest_round_trips_byte_for_byte(tmp_path: Path) -> None:
    """Regenerating an unchanged manifest must produce an empty diff."""
    manifest = build_manifest(_rows(), DATASET)
    path = tmp_path / "manifest.json"
    save_manifest(manifest, path)
    reloaded = load_manifest(path)
    assert to_json(reloaded) == to_json(manifest)
    assert reloaded.digest() == manifest.digest()


def test_row_order_does_not_change_the_manifest() -> None:
    """A manifest must be a property of the rows, not of the fetch order."""
    rows = _rows()
    assert to_json(build_manifest(rows, DATASET)) == to_json(
        build_manifest(list(reversed(rows)), DATASET)
    )


def test_unusable_labels_are_left_out_of_the_scored_population() -> None:
    """A row that is neither positive nor negative cannot define a denominator."""
    rows = [*_rows(4), trace_row("trajectory_9999", "", [user("x"), assistant("y")])]
    manifest = build_manifest(rows, DATASET)
    assert manifest.counts.rows == 4
    assert "trajectory_9999" not in manifest.by_trajectory_id


def test_counts_agree_with_the_rows_they_summarize() -> None:
    """The stated denominators must not be recomputed to be trusted."""
    manifest = build_manifest(_rows(20), DATASET)
    assert sum(manifest.counts.by_split.values()) == manifest.counts.rows
    positives = sum(manifest.counts.positives_by_split.values())
    negatives = sum(manifest.counts.negatives_by_split.values())
    assert positives + negatives == manifest.counts.rows
    assert positives == sum(1 for row in manifest.rows if row.positive)


def test_verification_names_missing_changed_and_unexpected_rows() -> None:
    """Drift from the pinned revision must be reported, not absorbed."""
    rows = _rows(6)
    manifest = build_manifest(rows, DATASET)

    mutated = [dict(row) for row in rows]
    mutated[0]["conversation"] = mutated[0]["conversation"] + " "
    mutated.pop()
    mutated.append(trace_row("trajectory_8888", "0", [user("x"), assistant("y")]))

    result = verify(manifest, mutated)
    assert result.changed == ["trajectory_0000"]
    assert result.missing == ["trajectory_0005"]
    assert result.unexpected == ["trajectory_8888"]
    assert not result.ok


def test_verification_can_be_scoped_to_one_split() -> None:
    """Checking a single-split fetch must not report the other splits as missing."""
    rows = _rows(30)
    manifest = build_manifest(rows, DATASET)
    tuning = {ref.trajectory_id for ref in manifest.for_split("tuning")}
    if not tuning:
        pytest.skip("no tuning rows in this synthetic population")
    selected = [row for row in rows if row["trajectory_id"] in tuning]
    assert verify(manifest, selected, only=tuning).ok


def test_a_row_hash_ignores_columns_the_benchmark_does_not_read() -> None:
    """An unrelated upstream column must not invalidate every hash."""
    row = _rows(1)[0]
    assert row_digest(row) == row_digest({**row, "annotator_notes": "added upstream"})


def test_first_user_text_survives_a_broken_conversation() -> None:
    """Scenario keying must degrade rather than raise on an unparseable cell."""
    assert first_user_text("{not json") == ""
    assert first_user_text('[{"role": "assistant", "content": "hi"}]') == ""
    assert first_user_text('[{"role": "user", "content": "hello"}]') == "hello"


def test_trace_rollouts_declare_which_detectors_can_be_scored() -> None:
    """TRACE carries transcripts only, so only transcript detectors are applicable."""
    rollout = map_row(_rows(1)[0])
    signals = available_signals(rollout)
    assert Requirement.ASSISTANT_TEXT in signals
    assert Requirement.REWARD not in signals
    assert Requirement.METRICS not in signals
    assert Requirement.ANSWER_OR_CRITERIA not in signals

    assert applicability("verifier_tamper", rollout).applicable
    verdict = applicability("length_inflation", rollout)
    assert not verdict.applicable
    assert verdict.reason == "missing:reward"


def test_population_signals_need_real_sibling_groups() -> None:
    """One group, or a population of singletons, cannot support a within-group test."""
    rollouts = [map_row(row) for row in _rows(6)]
    assert Requirement.GROUPS not in population_signals(rollouts)
    assert Requirement.STEPS not in population_signals(rollouts)

    for index, rollout in enumerate(rollouts):
        rollout.group_id = f"g{index % 2}"
        rollout.step_index = index % 3
    signals = population_signals(rollouts)
    assert Requirement.GROUPS in signals
    assert Requirement.STEPS in signals


def test_the_driver_scores_only_rows_the_manifest_pins() -> None:
    """A fetch that over-returns must not widen the evaluated population."""
    rows = _rows(12)
    manifest = build_manifest(rows[:8], DATASET)
    report = evaluate_rows(
        [*rows, trace_row("trajectory_7777", "0", [user("x"), assistant("y")])],
        manifest,
        discover_detectors(),
        DetectorConfig(),
        source="test",
    )
    assert report.rows_requested == 8
    assert report.rows_mapped == 8
    assert report.manifest_verified


def test_the_driver_reports_every_selected_detector(tmp_path: Path) -> None:
    """A detector with no eligible units must still appear, with its reason."""
    manifest = build_manifest(_rows(8), DATASET)
    report = evaluate_rows(
        _rows(8), manifest, discover_detectors(), DetectorConfig(), source="test"
    )
    reported = {metric.detector for metric in report.metrics}
    assert reported == set(discover_detectors())
    for metric in report.metrics:
        if metric.counts.units_scored == 0:
            assert metric.counts.insufficient_by_reason
            assert metric.counts.true_negatives == 0


def test_the_driver_counts_unusable_rows_by_reason() -> None:
    """A row that cannot be mapped is reported, never silently dropped."""
    rows = _rows(4)
    broken = trace_row("trajectory_0100", "0", [user("hello")])
    manifest = build_manifest([*rows, broken], DATASET)
    report = evaluate_rows(
        [*rows, broken], manifest, discover_detectors(), DetectorConfig(), source="test"
    )
    assert report.rows_requested == 5
    assert report.rows_mapped == 4
    assert report.rows_unusable == 1
    assert any("assistant turn" in reason for reason in report.unusable_by_reason)


def test_the_report_pins_everything_that_could_move_a_number() -> None:
    """Two reports are comparable exactly when these agree."""
    manifest = build_manifest(_rows(8), DATASET)
    report = evaluate_rows(
        _rows(8), manifest, discover_detectors(), DetectorConfig(), source="test"
    )
    assert report.dataset_revision == "deadbeef"
    assert report.manifest_digest == manifest.digest()
    assert report.mapping_version == manifest.mapping_version
    assert report.split_version == manifest.split_version
    pins = {pin.name: pin for pin in report.detectors}
    assert set(pins) == set(discover_detectors())
    assert all(pin.source_sha256 for pin in pins.values())
    assert pins["verifier_tamper"].config["min_matches"] == 1
