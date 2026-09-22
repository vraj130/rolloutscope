"""Detector verdicts and report findings: the frozen output contract.

These models are part of the Phase 2 freeze so the detector and report sub-agents
can build against them in parallel. A Verdict is per-rollout or per-group detector
output; a Finding is the report-level aggregation of verdicts.

The evidence span is mandatory by construction: a fired Verdict without evidence
fails validation, because a flag without the offending span is a bug (CLAUDE.md
golden rule 4).
"""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field, JsonValue, model_validator

from rolloutscope.schema.models import JsonSafeModel

Severity = Literal["info", "warning", "critical"]
AnalysisUnit = Literal["rollout", "group", "step", "run"]


class EvidenceSpan(JsonSafeModel):
    """The offending span that made a detector fire.

    ``field`` names the rollout field the span lives in (for example
    ``completion``, ``metrics.format_reward``, ``trajectory[2].completion``).
    ``start`` and ``end`` are optional character offsets into that field's text
    form; ``text`` always carries the span itself so evidence is readable without
    re-resolving offsets.
    """

    model_config = ConfigDict(extra="allow", allow_inf_nan=False)

    rollout_id: str
    field: str
    start: int | None = None
    end: int | None = None
    text: str
    note: str | None = None


class SourceOccurrence(JsonSafeModel):
    """Typed source location for one occurrence referenced by a verdict."""

    model_config = ConfigDict(extra="allow", allow_inf_nan=False)

    occurrence_id: str
    source_path: str
    line: int = Field(ge=1)


class Verdict(JsonSafeModel):
    """Structured output of one detector over one rollout or one group.

    ``score`` is a heuristic confidence or severity in [0, 1]; ``category`` is the
    taxonomy category (for example ``verifier_tampering``); ``rollout_ids`` lists
    every rollout the verdict covers (one entry for per-rollout detectors, the
    whole group for group detectors). Evidence is a list of structured spans
    (D-003); a fired verdict must carry at least one.
    """

    model_config = ConfigDict(extra="allow", allow_inf_nan=False)

    detector: str
    fired: bool
    score: float = Field(ge=0.0, le=1.0)
    category: str
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    rollout_ids: list[str] = Field(default_factory=list)
    mode: str = "snapshot"
    unit: AnalysisUnit = "rollout"
    run_id: str | None = None
    source_occurrences: list[SourceOccurrence] = Field(default_factory=list)
    measurements: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _fired_needs_evidence(self) -> Verdict:
        if self.fired and not self.evidence:
            raise ValueError("a fired verdict must carry at least one evidence span")
        return self


class Finding(JsonSafeModel):
    """Report-level aggregation of verdicts from one detector.

    ``metrics`` holds the numbers behind the finding (rates, correlations,
    counts); ``config_used`` records the exact thresholds active when the
    detector ran, for reproducibility; ``exemplars`` are selected evidence spans
    (each already carries its rollout_id).
    """

    model_config = ConfigDict(extra="allow", allow_inf_nan=False)

    severity: Severity
    title: str
    description: str
    detector: str
    metrics: dict[str, float] = Field(default_factory=dict)
    config_used: dict[str, JsonValue] = Field(default_factory=dict)
    exemplars: list[EvidenceSpan] = Field(default_factory=list)
    rollout_ids: list[str] = Field(default_factory=list)
    mode: str = "snapshot"
    unit: AnalysisUnit = "rollout"
