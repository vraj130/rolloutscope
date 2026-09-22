"""The hard-negative corpus: what the detectors must not flag, and what they do.

``expected_fired`` is the contract (always false, these are hard negatives).
``currently_fires`` is the measurement. Keeping both means this file is a
regression test today (present behavior cannot drift unnoticed) and an
acceptance test after the Phase 2 detector repairs (a repair flips one recorded
flag to false and the same assertions keep holding).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest

from rolloutscope.detectors import DetectorConfig, discover_detectors
from rolloutscope.schema import Rollout, validate_rollout

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "hard_negatives"


def _cases() -> list[dict[str, Any]]:
    """Load the case metadata."""
    payload: list[dict[str, Any]] = json.loads((FIXTURES / "cases.json").read_text("utf-8"))
    return payload


def _rollouts() -> dict[str, Rollout]:
    """Load the fixture rollouts, keyed by rollout id."""
    rows = [
        json.loads(line)
        for line in (FIXTURES / "rollouts.jsonl").read_text("utf-8").splitlines()
        if line.strip()
    ]
    return {row["rollout_id"]: validate_rollout(row) for row in rows}


CASES = _cases()
CASE_IDS = [case["case_id"] for case in CASES]


def test_every_case_has_exactly_one_matching_rollout() -> None:
    """The metadata and the data must not drift apart."""
    rollouts = _rollouts()
    assert len(CASE_IDS) == len(set(CASE_IDS))
    assert set(CASE_IDS) == set(rollouts)


def test_every_case_is_a_hard_negative() -> None:
    """This corpus states what must not fire; a positive belongs in ../labeled/."""
    for case in CASES:
        assert case["expected_fired"] is False, case["case_id"]
        assert case["rationale"].strip(), case["case_id"]


def test_every_detector_has_at_least_one_hard_negative() -> None:
    """A detector with only positive fixtures has never been tested for cost."""
    covered = {case["detector"] for case in CASES}
    assert covered == set(discover_detectors())


@pytest.fixture(scope="module")
def fired() -> dict[str, set[str]]:
    """Run each detector over the cases authored for it, once.

    Detectors are run per detector population rather than over the whole corpus
    because the group and correlation detectors read the population, and mixing
    unrelated cases into it would change what they see.
    """
    rollouts = _rollouts()
    populations: dict[str, list[Rollout]] = defaultdict(list)
    for case in CASES:
        populations[case["detector"]].append(rollouts[case["case_id"]])

    detectors = discover_detectors()
    config = DetectorConfig()
    result: dict[str, set[str]] = {}
    for name, population in populations.items():
        hits: set[str] = set()
        for verdict in detectors[name].detect(population, config):
            if verdict.fired:
                hits.update(unit for unit in verdict.rollout_ids if unit)
        result[name] = hits
    return result


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_recorded_behavior_matches_actual_behavior(
    case: dict[str, Any], fired: dict[str, set[str]]
) -> None:
    """Present behavior must match what ``cases.json`` records.

    A failure here means one of two things, and the fix differs. Either a
    detector changed and the record is stale, in which case update
    ``currently_fires`` in the same commit as the detector change. Or a detector
    regressed on a case it used to pass, in which case fix the detector.
    """
    actual = case["case_id"] in fired[case["detector"]]
    recorded = case["currently_fires"]
    assert actual is recorded, (
        f"{case['case_id']}: cases.json records currently_fires={recorded} but "
        f"{case['detector']} {'fired' if actual else 'did not fire'}. "
        f"Why it must not fire: {case['rationale']}"
    )


def test_the_open_repair_backlog_is_the_recorded_one() -> None:
    """Guard the headline number the corpus README quotes.

    This is not a quality bar. It exists so that a change which quietly makes
    more hard negatives fire cannot land without someone restating the count.
    """
    open_gaps = sorted(case["case_id"] for case in CASES if case["currently_fires"])
    assert len(open_gaps) == 21
    assert "hn-tool-output-quotes-existing-code" in open_gaps
    assert "hn-short-answer-photosynthesis" in open_gaps
