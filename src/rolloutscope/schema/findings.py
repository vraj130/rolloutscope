"""Detector verdicts and report findings: the frozen output contract.

These models are part of the Phase 2 freeze so the detector and report sub-agents
can build against them in parallel. A Verdict is per-rollout or per-group detector
output; a Finding is the report-level aggregation of verdicts.

The evidence span is mandatory by construction: a fired Verdict without evidence
fails validation, because a flag without the offending span is a bug (CLAUDE.md
golden rule 4).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Severity = Literal["info", "warning", "critical"]


class EvidenceSpan(BaseModel):
    """The offending span that made a detector fire.

    ``field`` names the rollout field the span lives in (for example
    ``completion``, ``metrics.format_reward``, ``trajectory[2].completion``).
    ``start`` and ``end`` are optional character offsets into that field's text
    form; ``text`` always carries the span itself so evidence is readable without
    re-resolving offsets.
    """

    model_config = ConfigDict(extra="allow")

    rollout_id: str
    field: str
    start: int | None = None
    end: int | None = None
    text: str
    note: str | None = None


class Verdict(BaseModel):
    """Structured output of one detector over one rollout or one group.

    ``score`` is a heuristic confidence or severity in [0, 1]; ``category`` is the
    taxonomy category (for example ``verifier_tampering``); ``rollout_ids`` lists
    every rollout the verdict covers (one entry for per-rollout detectors, the
    whole group for group detectors). Evidence is a list of structured spans
    (D-003); a fired verdict must carry at least one.
    """

    model_config = ConfigDict(extra="allow")

    detector: str
    fired: bool
    score: float
    category: str
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    rollout_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _fired_needs_evidence(self) -> Verdict:
        if self.fired and not self.evidence:
            raise ValueError("a fired verdict must carry at least one evidence span")
        return self


class Finding(BaseModel):
    """Report-level aggregation of verdicts from one detector.

    ``metrics`` holds the numbers behind the finding (rates, correlations,
    counts); ``config_used`` records the exact thresholds active when the
    detector ran, for reproducibility; ``exemplars`` are selected evidence spans
    (each already carries its rollout_id).
    """

    model_config = ConfigDict(extra="allow")

    severity: Severity
    title: str
    description: str
    detector: str
    metrics: dict[str, float] = Field(default_factory=dict)
    config_used: dict[str, Any] = Field(default_factory=dict)
    exemplars: list[EvidenceSpan] = Field(default_factory=list)
