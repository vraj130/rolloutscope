"""Assembly of detector Verdicts into report-level Findings.

This is the only place verdicts become findings: detectors emit Verdicts,
templates format Findings, and the mapping between them lives here (CLAUDE.md
golden rule 7). Severity thresholds are conservative heuristics, clearly
labeled as such and configurable; they are not derived from any paper.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, Field, model_validator

from rolloutscope.schema import EvidenceSpan, Finding, Severity, Verdict


class SeverityThresholds(BaseModel):
    """Heuristic mapping from max fired verdict score to Finding severity.

    A finding is critical when the max fired score is at least ``critical_at``,
    warning when at least ``warning_at``, else info. The defaults (0.8, 0.5)
    are conservative heuristics, not paper-derived values; tune them per run
    via config.
    """

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

    Verdicts are grouped by (detector, category); each group yields at most
    one Finding whose metrics carry fired count, total verdict count, fired
    rate, flagged rollout count, and max fired score. Findings come back
    sorted by severity (critical, warning, info), then detector, then
    category.
    """
    active_thresholds = thresholds if thresholds is not None else SeverityThresholds()
    configs = config_used if config_used is not None else {}

    grouped: dict[tuple[str, str], list[tuple[int, Verdict]]] = {}
    for index, verdict in enumerate(verdicts):
        grouped.setdefault((verdict.detector, verdict.category), []).append((index, verdict))

    heuristic_note = (
        "Severity is heuristic: max fired score at or above "
        f"{active_thresholds.critical_at:g} maps to critical, at or above "
        f"{active_thresholds.warning_at:g} to warning, else info."
    )

    findings: list[Finding] = []
    for (detector, category), members in grouped.items():
        total = len(members)
        fired = [(index, verdict) for index, verdict in members if verdict.fired]
        detector_config = dict(configs.get(detector, {}))

        if not fired:
            if include_clean:
                findings.append(
                    Finding(
                        severity="info",
                        title=f"{detector}: checked, no {category} verdicts fired",
                        description=(
                            f"Detector '{detector}' ran {total} checks in category "
                            f"'{category}' and none fired."
                        ),
                        detector=detector,
                        metrics={
                            "fired_count": 0.0,
                            "total_verdicts": float(total),
                            "fired_rate": 0.0,
                            "flagged_rollouts": 0.0,
                            "max_score": 0.0,
                        },
                        config_used=detector_config,
                        exemplars=[],
                    )
                )
            continue

        max_score = max(verdict.score for _, verdict in fired)
        fired_rate = len(fired) / total
        flagged_rollouts = {
            rollout_id for _, verdict in fired for rollout_id in verdict.rollout_ids
        }

        exemplars: list[EvidenceSpan] = []
        for _, verdict in sorted(fired, key=lambda item: (-item[1].score, item[0])):
            for span in verdict.evidence:
                if len(exemplars) >= exemplar_limit:
                    break
                exemplars.append(span)
            if len(exemplars) >= exemplar_limit:
                break

        findings.append(
            Finding(
                severity=severity_for_score(max_score, active_thresholds),
                title=f"{detector}: {len(fired)} of {total} checks fired ({category})",
                description=(
                    f"Detector '{detector}' fired on {len(fired)} of {total} checks in "
                    f"category '{category}' (rate {fired_rate:.0%}, max score "
                    f"{max_score:.2f}). {heuristic_note}"
                ),
                detector=detector,
                metrics={
                    "fired_count": float(len(fired)),
                    "total_verdicts": float(total),
                    "fired_rate": fired_rate,
                    "flagged_rollouts": float(len(flagged_rollouts)),
                    "max_score": max_score,
                },
                config_used=detector_config,
                exemplars=exemplars,
            )
        )

    findings.sort(key=lambda finding: (_severity_rank(finding.severity), finding.detector))
    return findings


__all__: list[str] = [
    "SeverityThresholds",
    "assemble_findings",
    "severity_for_score",
]
