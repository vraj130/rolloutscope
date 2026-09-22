"""Split assignment must be stable, and the taxonomy must not overclaim."""

from __future__ import annotations

import pytest

from rolloutscope.benchmark import (
    FAMILIES,
    SPLIT_WEIGHTS,
    SUBCATEGORIES,
    assign_split,
    families_of,
    is_hacked,
    name_of,
    normalize_scenario_text,
    parse_label,
    scenario_key,
)


def test_reformatting_a_task_statement_does_not_split_its_scenario() -> None:
    """Whitespace and case must not manufacture a second scenario."""
    a = "Scale the WebSocket server to 50k connections."
    b = "  scale   the WebSocket\n server to 50k connections.  "
    assert normalize_scenario_text(a) == normalize_scenario_text(b)
    assert scenario_key(a) == scenario_key(b)


def test_assignment_depends_only_on_the_key() -> None:
    """A partial fetch must assign exactly as a full fetch would."""
    keys = [scenario_key(f"task {i}") for i in range(200)]
    first = [assign_split(k) for k in keys]
    second = [assign_split(k) for k in reversed(keys)]
    assert first == list(reversed(second))


def test_every_scenario_lands_in_exactly_one_split() -> None:
    """No key may fall through the cumulative-weight walk."""
    for i in range(500):
        assert assign_split(scenario_key(f"scenario {i}")) in SPLIT_WEIGHTS


def test_the_assignment_tracks_the_declared_weights() -> None:
    """Over many keys the realized shares must approach the declared ones.

    A loose tolerance on purpose: this asserts the hash is not skewed, not that a
    finite sample hits the weights exactly.
    """
    keys = [scenario_key(f"scenario {i}") for i in range(4000)]
    counts = {name: 0 for name in SPLIT_WEIGHTS}
    for key in keys:
        counts[assign_split(key)] += 1
    for name, weight in SPLIT_WEIGHTS.items():
        assert counts[name] / len(keys) == pytest.approx(weight, abs=0.03)


def test_weights_that_do_not_sum_positive_are_rejected() -> None:
    """A zeroed weight table would silently send everything to one split."""
    with pytest.raises(ValueError, match="positive"):
        assign_split("abc", {"train": 0.0, "tuning": 0.0, "holdout": 0.0})


@pytest.mark.parametrize(
    ("label", "expected"),
    [("0", False), ("1.1.1", True), ("1.1.2, 1.3.1", True), ("", None), ("   ", None)],
)
def test_label_classification(label: str, expected: bool | None) -> None:
    """A blank label is unusable, not a negative."""
    assert is_hacked(label) is expected


def test_multi_label_rows_split_into_codes_and_families() -> None:
    """39 percent of hacked rows carry several codes; all of them must survive."""
    assert parse_label("1.1.2, 1.3.1") == ("1.1.2", "1.3.1")
    assert families_of("1.1.2, 1.1.3, 1.3.1") == ("1.1", "1.3")
    assert families_of("0") == ()


def test_inferred_names_are_marked_as_inferred() -> None:
    """A positional guess must never render as the dataset's own label."""
    assert name_of("1.1") == "Test Suite Exploitation"
    assert name_of("1.1.1") == "Test Modification (inferred)"
    assert name_of("9.9.9") == "9.9.9"


def test_recorded_counts_match_the_published_totals() -> None:
    """The family counts must add up to the hacked-row total they describe."""
    assert sum(c.observed_rows for c in FAMILIES) == 358
    assert sum(c.observed_rows for c in SUBCATEGORIES) == 358
    for family in FAMILIES:
        children = [c for c in SUBCATEGORIES if c.code.startswith(f"{family.code}.")]
        assert sum(c.observed_rows for c in children) == family.observed_rows
