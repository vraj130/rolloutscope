"""Server-side SVG helpers: structure, accessibility, and escaping."""

import pytest

from rolloutscope.report import histogram, line_chart


def test_histogram_structure() -> None:
    svg = histogram(
        [0.0, 2.0, 3.0],
        ["0-0.3", "0.3-0.6", "0.6-1"],
        title="Reward histogram test",
        x_label="reward bin",
        y_label="rollouts",
    )
    assert svg.startswith("<svg ")
    assert svg.endswith("</svg>")
    assert 'role="img"' in svg
    assert "<title>Reward histogram test</title>" in svg
    # One bar per nonzero bin, each with a native tooltip title.
    assert svg.count('class="bar"') == 2
    assert "<title>0.3-0.6: 2</title>" in svg
    assert "reward bin" in svg
    assert "rollouts" in svg


def test_histogram_requires_matching_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        histogram([1.0], ["a", "b"], title="bad")


def test_histogram_handles_empty_and_zero_data() -> None:
    assert histogram([], [], title="Empty histogram").startswith("<svg ")
    zeros = histogram([0.0, 0.0], ["a", "b"], title="All zero")
    assert zeros.count('class="bar"') == 0


def test_histogram_escapes_labels() -> None:
    svg = histogram([1.0], ["<evil>"], title="T < U")
    assert "<evil>" not in svg
    assert "&lt;evil&gt;" in svg
    assert "T &lt; U" in svg


def test_line_chart_two_series_with_legend_and_direct_labels() -> None:
    svg = line_chart(
        {
            "reward mean": [(0.0, 0.5), (1.0, 0.7), (2.0, 0.9)],
            "dead group fraction": [(0.0, 0.0), (1.0, 0.25), (2.0, 1.0)],
        },
        title="Step series test",
        x_label="training step",
        y_label="value",
    )
    assert svg.count("<polyline ") == 2
    # Legend swatch plus end-of-line direct label name each series at least twice.
    assert svg.count(">reward mean</text>") == 2
    assert svg.count(">dead group fraction</text>") == 2
    # Every point carries a native tooltip.
    assert svg.count("<circle ") == 6
    assert "<title>reward mean: x 1, y 0.7</title>" in svg
    assert "training step" in svg


def test_line_chart_single_series_has_no_legend() -> None:
    svg = line_chart(
        {"reward mean": [(0.0, 0.1), (1.0, 0.2)]},
        title="Single series",
    )
    assert svg.count("<polyline ") == 1
    # The name lives only in point tooltips, never as visible legend text.
    assert ">reward mean</text>" not in svg
