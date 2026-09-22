"""Assembly of detector Verdicts into report-level Findings.

This is the only place verdicts become findings: detectors emit Verdicts,
templates format Findings, and the mapping between them lives here (CLAUDE.md
golden rule 7). Severity thresholds are conservative heuristics, clearly
labeled as such and configurable; they are not derived from any paper.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rolloutscope.schema import AnalysisUnit, EvidenceSpan, Finding, Severity, Verdict
from rolloutscope.schema.execution import DetectorExecution


class SeverityThresholds(BaseModel):
    """Heuristic mapping from max fired verdict score to Finding severity.

    A finding is critical when the max fired score is at least ``critical_at``,
    warning when at least ``warning_at``, else info. The defaults (0.8, 0.5)
    are conservative heuristics, not paper-derived values; tune them per run
    via config.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, validate_default=True)

    critical_at: float = Field(default=0.8, ge=0.0, le=1.0)
    warning_at: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _ordered(self) -> SeverityThresholds:
        if self.warning_at > self.critical_at:
            raise ValueError("warning_at must not exceed critical_at")
        return self


def severity_for_score(score: float, thresholds: SeverityThresholds) -> Severity:
    """Map a max fired verdict score to a severity via heuristic thresholds.

    Inputs: the score (expected in [0, 1]) and the threshold config. Returns
    ``critical`` at or above ``critical_at``, ``warning`` at or above
    ``warning_at``, else ``info``.
    """
    if score >= thresholds.critical_at:
        return "critical"
    if score >= thresholds.warning_at:
        return "warning"
    return "info"


def _severity_rank(severity: Severity) -> int:
    return {"critical": 0, "warning": 1, "info": 2}[severity]


def assemble_findings(
    verdicts: Sequence[Verdict],
    *,
    thresholds: SeverityThresholds | None = None,
    exemplar_limit: int = 3,
    include_clean: bool = False,
    config_used: Mapping[str, Mapping[str, Any]] | None = None,
    executions: Sequence[DetectorExecution] | None = None,
) -> list[Finding]:
    """Turn detector Verdicts into report-level Findings.

    Inputs: the verdict list; heuristic ``thresholds`` for the score-to-severity
    mapping (documented on SeverityThresholds); ``exemplar_limit`` evidence
    spans kept per finding, drawn from fired verdicts in descending score
    order (ties resolve to input order); ``include_clean`` controls whether a
    (detector, category) group with zero fired verdicts yields an info
    finding ("checked and clean" reporting for the CLI) or no finding at all;
    ``config_used`` maps detector name to the exact config that detector ran
    with, recorded on its findings for reproducibility.

    ``executions`` supplies actual eligible denominators, including detectors
    with no verdicts. Without it, the legacy explicit-verdict denominator is
    retained for library callers. Unsupported coverage has no inferred rate.
    Verdicts are grouped by (detector, category, mode, unit); each group yields at most
    one Finding whose metrics carry fired count, total verdict count, eligible
    checks, fired rate, flagged rollout count, and max fired score. Findings come back
    sorted by severity (critical, warning, info), then detector, then
    category.
    """
    active_thresholds = thresholds if thresholds is not None else SeverityThresholds()
    configs = config_used if config_used is not None else {}

    grouped: dict[tuple[str, str, str, AnalysisUnit], list[tuple[int, Verdict]]] = {}
    for index, verdict in enumerate(verdicts):
        key = (verdict.detector, verdict.category, verdict.mode, verdict.unit)
        grouped.setdefault(key, []).append((index, verdict))

    execution_map = {item.detector: item for item in executions or []}
    if executions is not None:
        for recorded_execution in executions:
            for unit in recorded_execution.units:
                grouped.setdefault(
                    (
                        recorded_execution.detector,
                        recorded_execution.category,
                        unit.mode,
                        unit.unit,
                    ),
                    [],
                )

    heuristic_note = (
        "Severity is heuristic: max fired score at or above "
        f"{active_thresholds.critical_at:g} maps to critical, at or above "
        f"{active_thresholds.warning_at:g} to warning, else info."
    )

    findings: list[Finding] = []
    for (detector, category, mode, unit_name), members in grouped.items():
        execution = execution_map.get(detector)
        if execution is not None and execution.status == "failed":
            continue
        units = (
            [unit for unit in execution.units if unit.mode == mode and unit.unit == unit_name]
            if execution
            else []
        )
        known_coverage = executions is None or bool(units)
        total = sum(unit.eligible for unit in units) if executions is not None else len(members)
        fired = [(index, verdict) for index, verdict in members if verdict.fired]
        detector_config = dict(configs.get(detector, execution.config if execution else {}))
        metrics = {
            "fired_count": float(len(fired)),
            "total_verdicts": float(len(members)),
            "flagged_rollouts": 0.0,
            "max_score": 0.0,
        }
        if known_coverage:
            metrics["eligible_checks"] = float(total)
            metrics["fired_rate"] = len(fired) / total if total else 0.0
        if units:
            metrics.update(
                {
                    "candidate_checks": float(sum(unit.candidate for unit in units)),
                    "insufficient_data": float(sum(unit.insufficient_data for unit in units)),
                    "skipped": float(sum(unit.skipped for unit in units)),
                    "errors": float(sum(unit.errors for unit in units)),
                }
            )

        if not fired:
            if include_clean and known_coverage and total > 0:
                findings.append(
                    Finding.model_validate(
                        {
                            "severity": "info",
                            "title": f"{detector} ({mode}): checked, no {category} verdicts fired",
                            "description": (
                                f"Detector '{detector}' evaluated {total} eligible "
                                f"{unit_name} units "
                                f"in category '{category}' ({mode}) and none fired."
                            ),
                            "detector": detector,
                            "metrics": metrics,
                            "config_used": detector_config,
                            "exemplars": [],
                            "mode": mode,
                            "unit": unit_name,
                        }
                    )
                )
            continue

        max_score = max(verdict.score for _, verdict in fired)
        flagged_rollouts = {
            rollout_id for _, verdict in fired for rollout_id in verdict.rollout_ids
        }
        metrics.update(flagged_rollouts=float(len(flagged_rollouts)), max_score=max_score)

        exemplars: list[EvidenceSpan] = []
        for _, verdict in sorted(fired, key=lambda item: (-item[1].score, item[0])):
            for span in verdict.evidence:
                if len(exemplars) >= exemplar_limit:
                    break
                exemplars.append(span)
            if len(exemplars) >= exemplar_limit:
                break

        result = (
            f"{len(fired)} of {total} eligible {unit_name} units fired "
            f"(rate {metrics['fired_rate']:.0%})"
            if known_coverage
            else f"{len(fired)} verdicts fired; eligibility coverage unavailable"
        )
        findings.append(
            Finding.model_validate(
                {
                    "severity": severity_for_score(max_score, active_thresholds),
                    "title": f"{detector} ({mode}): {result} ({category})",
                    "description": (
                        f"Detector '{detector}', {mode}: {result} in category '{category}' "
                        f"(max score {max_score:.2f}). {heuristic_note}"
                    ),
                    "detector": detector,
                    "metrics": metrics,
                    "config_used": detector_config,
                    "exemplars": exemplars,
                    "rollout_ids": sorted(flagged_rollouts),
                    "mode": mode,
                    "unit": unit_name,
                }
            )
        )

    findings.sort(key=lambda finding: (_severity_rank(finding.severity), finding.detector))
    return findings


__all__: list[str] = [
    "SeverityThresholds",
    "assemble_findings",
    "severity_for_score",
]
