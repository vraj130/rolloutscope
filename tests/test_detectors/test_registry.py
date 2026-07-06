"""Entry-point registry: discovery, categories, fallback, broken-plugin policy."""

from __future__ import annotations

import logging
from importlib.metadata import EntryPoint, entry_points

import rolloutscope.detectors.base as base
from rolloutscope.detectors import (
    DETECTOR_ENTRY_POINT_GROUP,
    builtin_detectors,
    discover_detectors,
    load_detectors,
)

EXPECTED = {
    "verifier_tamper": "verifier_tampering",
    "reward_saturation_group_collapse": "reward_saturation",
    "length_inflation": "rubric_judge_exploit",
    "format_only_wins": "rubric_judge_exploit",
    "degenerate_repetition": "degeneracy",
    "answer_leakage_echo": "context_exploitation",
}


def test_entry_points_expose_exactly_the_six_builtins():
    names = {ep.name for ep in entry_points(group=DETECTOR_ENTRY_POINT_GROUP)}
    assert names == set(EXPECTED)


def test_load_detectors_instantiates_all_with_categories():
    detectors = load_detectors()
    assert set(detectors) == set(EXPECTED)
    for name, detector in detectors.items():
        assert detector.name == name
        assert detector.category == EXPECTED[name]
        assert callable(detector.detect)


def test_builtin_fallback_matches_entry_points():
    via_entry_points = load_detectors()
    via_fallback = builtin_detectors()
    assert set(via_fallback) == set(via_entry_points)
    for name in via_fallback:
        assert type(via_fallback[name]) is type(via_entry_points[name])


def test_discover_prefers_entry_points():
    detectors = discover_detectors()
    assert set(detectors) == set(EXPECTED)


def test_broken_plugin_is_skipped_with_warning(monkeypatch, caplog):
    """One unloadable entry point must not crash discovery of the others."""
    good = list(entry_points(group=DETECTOR_ENTRY_POINT_GROUP))
    broken = EntryPoint(
        name="broken_plugin",
        value="nonexistent_module_xyz:NoSuchDetector",
        group=DETECTOR_ENTRY_POINT_GROUP,
    )
    monkeypatch.setattr(base, "entry_points", lambda group: [*good, broken])
    with caplog.at_level(logging.WARNING, logger="rolloutscope.detectors.base"):
        detectors = base.load_detectors()
    assert set(detectors) == set(EXPECTED)
    assert "broken_plugin" not in detectors
    assert any("broken_plugin" in record.message for record in caplog.records)


def test_discover_falls_back_when_no_entry_points(monkeypatch, caplog):
    monkeypatch.setattr(base, "entry_points", lambda group: [])
    with caplog.at_level(logging.WARNING, logger="rolloutscope.detectors.base"):
        detectors = base.discover_detectors()
    assert set(detectors) == set(EXPECTED)
    assert any("falling back to built-ins" in record.message for record in caplog.records)
