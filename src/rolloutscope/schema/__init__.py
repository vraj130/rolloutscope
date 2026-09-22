"""The normalized data contract: the schema IS the product.

Everything downstream (adapters, detectors, analysis, report) codes against the
names re-exported here. After the Phase 2 freeze, changes require the
orchestrator, a version bump, and a migration entry.
"""

from rolloutscope.schema.findings import (
    AnalysisUnit,
    EvidenceSpan,
    Finding,
    Severity,
    SourceOccurrence,
    Verdict,
)
from rolloutscope.schema.ids import (
    attach_identity,
    content_fingerprint,
    group_id,
    occurrence_id,
    rollout_id,
    run_id_from_manifest,
    run_id_from_name,
    run_id_from_path,
    scoring_revision,
)
from rolloutscope.schema.io import iter_jsonl, read_rollouts, write_rollouts
from rolloutscope.schema.migrate import (
    CURRENT_MAJOR,
    UnsupportedSchemaVersionError,
    migrate_row,
)
from rolloutscope.schema.models import (
    ROLLOUT_ADAPTER,
    SCHEMA_VERSION,
    JsonSafeModel,
    JsonValue,
    Message,
    MultiTurnRollout,
    Rollout,
    RolloutBase,
    SingleTurnRollout,
    SourceProvenance,
    StepTokens,
    TimeSpan,
    Timing,
    TokenUsage,
    TrainingSignals,
    TrajectoryStep,
    infer_kind,
    rollout_json_schema,
    validate_rollout,
)

__all__ = [
    "CURRENT_MAJOR",
    "ROLLOUT_ADAPTER",
    "SCHEMA_VERSION",
    "AnalysisUnit",
    "EvidenceSpan",
    "Finding",
    "JsonSafeModel",
    "JsonValue",
    "Message",
    "MultiTurnRollout",
    "Rollout",
    "RolloutBase",
    "Severity",
    "SingleTurnRollout",
    "SourceOccurrence",
    "SourceProvenance",
    "StepTokens",
    "TimeSpan",
    "Timing",
    "TokenUsage",
    "TrainingSignals",
    "TrajectoryStep",
    "UnsupportedSchemaVersionError",
    "Verdict",
    "attach_identity",
    "content_fingerprint",
    "group_id",
    "infer_kind",
    "iter_jsonl",
    "migrate_row",
    "occurrence_id",
    "read_rollouts",
    "rollout_id",
    "rollout_json_schema",
    "run_id_from_manifest",
    "run_id_from_name",
    "run_id_from_path",
    "scoring_revision",
    "validate_rollout",
    "write_rollouts",
]
