"""Reject misspelled configuration, nonfinite thresholds, and invalid patterns early."""

import pytest
from pydantic import ValidationError

from rolloutscope.detectors import DetectorConfig


@pytest.mark.parametrize(
    "config,location",
    [
        ({"lenght_inflation": {}}, ("lenght_inflation",)),
        ({"length_inflation": {"min_sample": 8}}, ("length_inflation", "min_sample")),
        (
            {"verifier_tamper": {"patterns": {"broken": "["}}},
            ("verifier_tamper", "patterns", "broken"),
        ),
        ({"verifier_tamper": {"test_path_regex": "["}}, ("verifier_tamper", "test_path_regex")),
        (
            {"verifier_tamper": {"edit_tool_name_regex": "["}},
            ("verifier_tamper", "edit_tool_name_regex"),
        ),
        (
            {"answer_leakage_echo": {"criteria_line_regex": "["}},
            ("answer_leakage_echo", "criteria_line_regex"),
        ),
    ],
)
def test_invalid_config_names_the_exact_key(config, location):
    with pytest.raises(ValidationError) as exc:
        DetectorConfig.model_validate(config)
    assert exc.value.errors()[0]["loc"] == location


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize(
    "name,field",
    [
        ("verifier_tamper", "base_score"),
        ("reward_saturation_group_collapse", "saturated_reward_min"),
        ("length_inflation", "min_correlation"),
        ("format_only_wins", "min_reward"),
        ("degenerate_repetition", "min_reward"),
        ("answer_leakage_echo", "answer_echo_score"),
    ],
)
def test_every_detector_rejects_nonfinite_thresholds(name, field, value):
    with pytest.raises(ValidationError):
        DetectorConfig.model_validate({name: {field: value}})


def test_defaults_and_custom_patterns_round_trip():
    config = DetectorConfig.model_validate(
        {"verifier_tamper": {"patterns": {"custom": "valid.*pattern"}}}
    )
    assert DetectorConfig.model_validate(config.model_dump()) == config
    assert config.model_config["validate_default"] is True


@pytest.mark.parametrize(
    "name,field,value",
    [
        ("length_inflation", "min_correlation", -1.01),
        ("length_inflation", "min_correlation", 1.01),
        ("reward_saturation_group_collapse", "dead_fraction_threshold", -0.01),
        ("reward_saturation_group_collapse", "dead_fraction_threshold", 1.01),
        ("reward_saturation_group_collapse", "min_dead_fraction_rise", -0.01),
        ("reward_saturation_group_collapse", "min_dead_fraction_rise", 1.01),
    ],
)
def test_fraction_and_correlation_thresholds_have_physical_ranges(name, field, value):
    with pytest.raises(ValidationError):
        DetectorConfig.model_validate({name: {field: value}})


def test_regexes_are_precompiled_for_default_and_user_config():
    defaults = DetectorConfig()
    custom = DetectorConfig.model_validate(
        {
            "verifier_tamper": {
                "patterns": {"custom": "valid.*pattern"},
                "test_path_regex": "specs/",
                "edit_tool_name_regex": "change",
            },
            "answer_leakage_echo": {"criteria_line_regex": "grading"},
        }
    )
    assert defaults.verifier_tamper.compiled_patterns
    assert custom.verifier_tamper.compiled_patterns["custom"].search("valid pattern")
    assert custom.verifier_tamper.compiled_test_path.search("specs/example.py")
    assert custom.verifier_tamper.compiled_edit_tool_name.search("CHANGE")
    assert custom.answer_leakage_echo.compiled_criteria_line.search("grading")
