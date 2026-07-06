"""Streaming aggregates over normalized rollouts and Verdict-to-Finding assembly.

Depends only on the normalized schema types (CLAUDE.md golden rule 3); never
imports verifiers, prime-rl, adapters, or detectors.
"""

from rolloutscope.analysis.aggregates import (
    AggregateConfig,
    Aggregates,
    GroupStats,
    RewardHistogram,
    RolloutSnippet,
    RunSummary,
    StepStats,
    aggregate_rollouts,
    content_text,
)
from rolloutscope.analysis.findings import (
    SeverityThresholds,
    assemble_findings,
    severity_for_score,
)

__all__ = [
    "AggregateConfig",
    "Aggregates",
    "GroupStats",
    "RewardHistogram",
    "RolloutSnippet",
    "RunSummary",
    "SeverityThresholds",
    "StepStats",
    "aggregate_rollouts",
    "assemble_findings",
    "content_text",
    "severity_for_score",
]
