"""Server-side SVG chart helpers for the self-contained HTML report.

Exactly two chart forms, both returned as plain SVG strings with no external
references, no scripts, and no fonts beyond the system stack: ``histogram``
for bar magnitudes and ``line_chart`` for per-step series. Colors come from
the validated light-mode reference palette (dataviz skill): categorical slots
blue and aqua first, recessive gray chrome for grid and axes. Every chart
carries an accessible root ``<title>`` plus per-mark ``<title>`` elements,
which double as native browser tooltips without JavaScript. All caller text
is escaped before embedding.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from html import escape

_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#898781"
_GRID = "#e1e0d9"
_BASELINE = "#c3c2b7"
_SURFACE = "#fcfcfb"
_SERIES_COLORS = ("#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948")
_FONT = 'font-family="system-ui, sans-serif"'


def _num(value: float) -> str:
    """Format a coordinate deterministically with two decimals."""
    return f"{value:.2f}"


def _label_num(value: float) -> str:
    """Format an axis tick value compactly."""
    return f"{value:g}"


def _grid_and_ticks(
    parts: list[str],
    *,
    left: float,
    top: float,
    plot_w: float,
    plot_h: float,
    y_lo: float,
    y_hi: float,
) -> None:
    """Append horizontal hairline gridlines and y tick labels in place."""
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top + plot_h * (1.0 - fraction)
        color = _BASELINE if fraction == 0.0 else _GRID
        parts.append(
            f'<line x1="{_num(left)}" y1="{_num(y)}" x2="{_num(left + plot_w)}" '
            f'y2="{_num(y)}" stroke="{color}" stroke-width="1"/>'
        )
        if fraction in (0.0, 0.5, 1.0):
            value = y_lo + (y_hi - y_lo) * fraction
            parts.append(
                f'<text x="{_num(left - 6)}" y="{_num(y + 3.5)}" text-anchor="end" '
                f'font-size="10" fill="{_INK_MUTED}" {_FONT}>{_label_num(value)}</text>'
            )


def _frame(
    title: str,
    x_label: str,
    y_label: str,
    width: int,
    height: int,
) -> tuple[list[str], str]:
    """Open the SVG document: root element, accessible title, chart title.

    Returns the parts list and the closing tail (axis captions plus the
    closing tag are appended by the caller via the tail).
    """
    safe_title = escape(title)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        f"<title>{safe_title}</title>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{_SURFACE}"/>',
        f'<text x="16" y="22" font-size="13" font-weight="600" fill="{_INK}" {_FONT}>'
        f"{safe_title}</text>",
    ]
    tail_parts = []
    if x_label:
        tail_parts.append(
            f'<text x="{width / 2:.2f}" y="{height - 6}" text-anchor="middle" '
            f'font-size="11" fill="{_INK_SECONDARY}" {_FONT}>{escape(x_label)}</text>'
        )
    if y_label:
        tail_parts.append(
            f'<text x="12" y="{height / 2:.2f}" text-anchor="middle" font-size="11" '
            f'fill="{_INK_SECONDARY}" {_FONT} '
            f'transform="rotate(-90 12 {height / 2:.2f})">{escape(y_label)}</text>'
        )
    tail_parts.append("</svg>")
    return parts, "".join(tail_parts)


def histogram(
    bars: Sequence[float],
    labels: Sequence[str],
    *,
    title: str,
    x_label: str = "",
    y_label: str = "count",
    width: int = 640,
    height: int = 300,
) -> str:
    """Render a bar histogram as a self-contained SVG string.

    Inputs: ``bars`` (one non-negative value per bin), ``labels`` (one x-axis
    label per bin, same length), a chart ``title`` (also the accessible SVG
    title), optional axis captions, and pixel dimensions. Each bar carries a
    ``<title>`` tooltip with its label and value. Raises ValueError when bars
    and labels differ in length.
    """
    if len(bars) != len(labels):
        raise ValueError("bars and labels must have the same length")

    left, right, top, bottom = 52.0, 16.0, 36.0, 48.0
    plot_w = width - left - right
    plot_h = height - top - bottom
    y_max = max(bars, default=0.0) or 1.0

    parts, tail = _frame(title, x_label, y_label, width, height)
    _grid_and_ticks(parts, left=left, top=top, plot_w=plot_w, plot_h=plot_h, y_lo=0.0, y_hi=y_max)

    count = len(bars)
    if count:
        slot = plot_w / count
        bar_w = max(slot - 4.0, 1.0)
        label_step = max(1, math.ceil(count / 12))
        baseline = top + plot_h
        for index, (value, label) in enumerate(zip(bars, labels, strict=True)):
            x = left + index * slot + (slot - bar_w) / 2
            center = left + index * slot + slot / 2
            safe_label = escape(str(label))
            if value > 0:
                bar_h = plot_h * (value / y_max)
                y = baseline - bar_h
                radius = min(4.0, bar_w / 2, bar_h)
                path = (
                    f"M {_num(x)} {_num(baseline)} L {_num(x)} {_num(y + radius)} "
                    f"Q {_num(x)} {_num(y)} {_num(x + radius)} {_num(y)} "
                    f"L {_num(x + bar_w - radius)} {_num(y)} "
                    f"Q {_num(x + bar_w)} {_num(y)} {_num(x + bar_w)} {_num(y + radius)} "
                    f"L {_num(x + bar_w)} {_num(baseline)} Z"
                )
                parts.append(
                    f'<path class="bar" d="{path}" fill="{_SERIES_COLORS[0]}">'
                    f"<title>{safe_label}: {_label_num(value)}</title></path>"
                )
            if index % label_step == 0:
                parts.append(
                    f'<text x="{_num(center)}" y="{_num(baseline + 14)}" '
                    f'text-anchor="middle" font-size="10" fill="{_INK_MUTED}" {_FONT}>'
                    f"{safe_label}</text>"
                )

    parts.append(tail)
    return "".join(parts)


def line_chart(
    series: Mapping[str, Sequence[tuple[float, float]]],
    *,
    title: str,
    x_label: str = "",
    y_label: str = "",
    width: int = 640,
    height: int = 300,
) -> str:
    """Render one or more (x, y) series as a self-contained SVG line chart.

    Inputs: ``series`` maps series name to its points (drawn in the given
    order; hues are assigned in fixed palette order, never cycled), plus a
    chart ``title``, optional axis captions, and pixel dimensions. The y axis
    is anchored at 0 (or the data minimum when negative) on a single scale
    shared by all series. With two or more series a legend row and direct
    end-of-line labels name each series (identity is never color alone). Each
    point carries a ``<title>`` tooltip.
    """
    drawn = {name: list(points) for name, points in series.items() if points}
    all_points = [point for points in drawn.values() for point in points]

    left, top, bottom = 52.0, 36.0, 48.0
    right = 16.0
    if len(drawn) >= 2:
        longest = max(len(name) for name in drawn)
        right += longest * 6.2 + 10
        # Reserve a band between the title and the plot for the legend row.
        top = 58.0
    plot_w = width - left - right
    plot_h = height - top - bottom

    if all_points:
        x_lo = min(x for x, _ in all_points)
        x_hi = max(x for x, _ in all_points)
        y_lo = min(0.0, min(y for _, y in all_points))
        y_hi = max(y for _, y in all_points)
    else:
        x_lo, x_hi, y_lo, y_hi = 0.0, 1.0, 0.0, 1.0
    if x_hi == x_lo:
        x_hi = x_lo + 1.0
    if y_hi == y_lo:
        y_hi = y_lo + 1.0

    def scale_x(value: float) -> float:
        return left + plot_w * (value - x_lo) / (x_hi - x_lo)

    def scale_y(value: float) -> float:
        return top + plot_h * (1.0 - (value - y_lo) / (y_hi - y_lo))

    parts, tail = _frame(title, x_label, y_label, width, height)
    _grid_and_ticks(parts, left=left, top=top, plot_w=plot_w, plot_h=plot_h, y_lo=y_lo, y_hi=y_hi)

    for fraction in (0.0, 0.5, 1.0):
        value = x_lo + (x_hi - x_lo) * fraction
        parts.append(
            f'<text x="{_num(scale_x(value))}" y="{_num(top + plot_h + 14)}" '
            f'text-anchor="middle" font-size="10" fill="{_INK_MUTED}" {_FONT}>'
            f"{_label_num(value)}</text>"
        )

    for slot, (name, points) in enumerate(drawn.items()):
        color = _SERIES_COLORS[slot % len(_SERIES_COLORS)]
        safe_name = escape(name)
        coords = " ".join(f"{_num(scale_x(x))},{_num(scale_y(y))}" for x, y in points)
        parts.append(
            f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2" '
            f'stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for x, y in points:
            parts.append(
                f'<circle cx="{_num(scale_x(x))}" cy="{_num(scale_y(y))}" r="3.5" '
                f'fill="{color}" stroke="{_SURFACE}" stroke-width="1.5">'
                f"<title>{safe_name}: x {_label_num(x)}, y {_label_num(y)}</title></circle>"
            )
        if len(drawn) >= 2:
            last_x, last_y = points[-1]
            parts.append(
                f'<text x="{_num(scale_x(last_x) + 8)}" y="{_num(scale_y(last_y) + 3.5)}" '
                f'font-size="10" fill="{_INK_SECONDARY}" {_FONT}>{safe_name}</text>'
            )

    if len(drawn) >= 2:
        legend_x = left
        legend_y = 42.0
        for slot, name in enumerate(drawn):
            color = _SERIES_COLORS[slot % len(_SERIES_COLORS)]
            parts.append(
                f'<rect x="{_num(legend_x)}" y="{_num(legend_y - 8)}" width="10" height="10" '
                f'rx="2" fill="{color}"/>'
            )
            parts.append(
                f'<text x="{_num(legend_x + 14)}" y="{_num(legend_y + 1)}" font-size="10" '
                f'fill="{_INK_SECONDARY}" {_FONT}>{escape(name)}</text>'
            )
            legend_x += 14 + len(name) * 6.2 + 14

    parts.append(tail)
    return "".join(parts)


__all__ = ["histogram", "line_chart"]
