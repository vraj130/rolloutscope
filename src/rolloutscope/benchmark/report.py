"""The benchmark reporting format: one machine-readable document, one text view.

A benchmark number is worthless without the pins that make it reproducible, so
the report carries them inline: the manifest digest, the dataset revision, the
mapping version, the split evaluated, the effective detector configuration, and
a fingerprint of each detector's source. Two reports are comparable exactly when
those agree, and the text view prints them above the table so a reader cannot
miss a mismatch.

Rates print with their Wilson interval and their denominator. A detector that
could not be scored prints its coverage and the reasons, not a row of zeros.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from typing import Any

from pydantic import BaseModel, Field

from rolloutscope.benchmark.metrics import DetectorMetrics, Rate

REPORT_VERSION = "1"
"""Version of the report document shape."""


class DetectorPin(BaseModel):
    """What a detector was, exactly, when it was measured."""

    name: str
    category: str = ""
    source_sha256: str = ""
    """SHA-256 of the detector's module source.

    Detectors carry no version field of their own, so the source hash stands in
    as one: it changes whenever the implementation changes, which is the property
    a reproducibility pin needs. It is not a semantic version and must not be
    compared for ordering.
    """
    config: dict[str, Any] = Field(default_factory=dict)


class ExampleCase(BaseModel):
    """One false positive or false negative held out for human review."""

    unit_id: str
    cell: str
    codes: tuple[str, ...] = ()
    evidence_field: str = ""
    evidence_text: str = ""


class BenchmarkReport(BaseModel):
    """The full result of one benchmark run."""

    report_version: str = REPORT_VERSION
    dataset: str = ""
    dataset_revision: str = ""
    manifest_digest: str = ""
    mapping_version: str = ""
    split_version: str = ""
    split: str = ""
    source: str = ""
    """How the rows were obtained: ``parquet`` (revision pinned) or
    ``datasets-server`` (current default branch, hash verified)."""
    rows_requested: int = 0
    rows_mapped: int = 0
    rows_unusable: int = 0
    unusable_by_reason: dict[str, int] = Field(default_factory=dict)
    manifest_verified: bool = False
    verification_notes: list[str] = Field(default_factory=list)
    detectors: list[DetectorPin] = Field(default_factory=list)
    metrics: list[DetectorMetrics] = Field(default_factory=list)
    false_positives: list[ExampleCase] = Field(default_factory=list)
    false_negatives: list[ExampleCase] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def detector_fingerprint(detector: object) -> str:
    """SHA-256 of a detector's module source, or an empty string if unavailable.

    Input: a detector instance. Output: the hex digest of its defining module's
    source text. Returns an empty string for a detector whose source cannot be
    read (a compiled or dynamically generated plugin), so a report records "not
    pinned" rather than a fabricated identifier.
    """
    try:
        source = inspect.getsource(type(detector))
    except (OSError, TypeError):
        return ""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def to_json(report: BenchmarkReport) -> str:
    """Serialize a report deterministically, sorted keys and a trailing newline."""
    return (
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
    )


def _rate(rate: Rate) -> str:
    """Format a rate as ``value [low, high] n/d``, or as a reason when unmeasured."""
    if rate.value is None:
        return f"not measured (0/{rate.denominator})"
    body = f"{rate.value:.3f}"
    if rate.interval is not None:
        body += f" [{rate.interval.low:.3f}, {rate.interval.high:.3f}]"
    return f"{body} {rate.numerator}/{rate.denominator}"


def render_text(report: BenchmarkReport) -> str:
    """Render a report as plain text for a terminal.

    Input: the report. Output: a multi-line string. Every rate carries its
    denominator and interval, and a detector with zero coverage prints why it
    was not scored instead of printing zeros that look like measurements.
    """
    lines: list[str] = []
    lines.append("rolloutscope detector benchmark")
    lines.append(f"  dataset          {report.dataset} @ {report.dataset_revision}")
    lines.append(f"  source           {report.source}")
    lines.append(f"  manifest         {report.manifest_digest[:16]}")
    lines.append(
        f"  verified         {'yes' if report.manifest_verified else 'no'}"
        + ("" if report.manifest_verified else " (see notes)")
    )
    lines.append(f"  split            {report.split}")
    lines.append(
        f"  mapping/split    v{report.mapping_version} / v{report.split_version}",
    )
    lines.append(f"  rows             {report.rows_mapped} mapped, {report.rows_unusable} unusable")
    for reason, count in sorted(report.unusable_by_reason.items()):
        lines.append(f"                   {count} {reason}")
    lines.append("")

    for metric in report.metrics:
        counts = metric.counts
        lines.append(f"{metric.detector}  (unit: {metric.unit})")
        lines.append(f"  coverage         {_rate(metric.coverage)}")
        if counts.units_scored == 0:
            for reason, count in sorted(counts.insufficient_by_reason.items()):
                lines.append(f"  not scored       {count} units {reason}")
            lines.append("  no scored units, so no accuracy is reported for this detector")
            lines.append("")
            continue
        lines.append(
            f"  confusion        tp={counts.true_positives} fp={counts.false_positives} "
            f"tn={counts.true_negatives} fn={counts.false_negatives}"
        )
        lines.append(f"  precision        {_rate(metric.precision)}")
        lines.append(f"  recall           {_rate(metric.recall)}")
        lines.append(f"  fp rate          {_rate(metric.false_positive_rate)}")
        if counts.units_insufficient:
            for reason, count in sorted(counts.insufficient_by_reason.items()):
                lines.append(f"  insufficient     {count} units {reason}")
        if counts.units_error:
            for reason, count in sorted(counts.error_by_reason.items()):
                lines.append(f"  errors           {count} units {reason}")
        for category in metric.by_category:
            lines.append(f"  recall {category.code:<8} {_rate(category.recall)}  {category.name}")
        lines.append("")

    if report.notes:
        lines.append("notes")
        for note in report.notes:
            lines.append(f"  - {note}")
        lines.append("")
    if report.verification_notes:
        lines.append("verification")
        for note in report.verification_notes:
            lines.append(f"  - {note}")
        lines.append("")
    return "\n".join(lines)
