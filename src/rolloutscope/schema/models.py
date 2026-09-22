"""Normalized rollout data contract for rolloutscope.

Kept
field-compatible with the verifiers RolloutOutput contract (verifiers @ 5885ab9c).
Upstream names win on any conflict.

Design rules, all load-bearing:
- Rollout is a discriminated union on ``kind`` (single_turn | multi_turn).
- Every row carries ``schema_version`` so a single leaked row is self-describing.
- ``extra="allow"`` on row, message, step, timing, and token models: verifiers
  injects arbitrary state_columns and providers add keys; dropping them is a bug.
- RL training signals (advantages, is_trainable) live in the optional
  TrainingSignals sidecar, never on the base row: they are in-memory during
  training and are not in on-disk jsonl.
"""

from __future__ import annotations

import math
import re
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)
from pydantic import JsonValue as PydanticJsonValue

JsonValue = PydanticJsonValue

SCHEMA_VERSION = "2.0"


def _check_json_safe(value: Any, path: str = "value") -> None:
    """Reject values that cannot round-trip through the supported JSON writer."""
    if value is None or isinstance(value, str | bool):
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path}: non-finite numeric values are not supported")
    if isinstance(value, float):
        return
    if isinstance(value, int):
        if not -(2**63) <= value < 2**64:
            raise ValueError(f"{path}: integer exceeds the JSON writer's 64-bit range")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path}: JSON object keys must be strings")
            _check_json_safe(item, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _check_json_safe(item, f"{path}[{index}]")
        return
    if isinstance(value, BaseModel):
        _check_json_safe(value.__dict__, path)
        _check_json_safe(value.__pydantic_extra__ or {}, f"{path}.extras")
        return
    raise ValueError(f"{path}: {type(value).__name__} is not JSON serializable")


class JsonSafeModel(BaseModel):
    """Validate nested numerics and arbitrary values against the JSON contract."""

    model_config = ConfigDict(allow_inf_nan=False, validate_default=True)

    @model_validator(mode="before")
    @classmethod
    def _raw_json_safe(cls, value: Any) -> Any:
        _check_json_safe(value)
        return value

    @model_validator(mode="after")
    def _json_safe(self) -> JsonSafeModel:
        _check_json_safe(self.__dict__)
        _check_json_safe(self.__pydantic_extra__ or {}, "extras")
        return self


class FiniteModel(JsonSafeModel):
    """Preserve provider fields while rejecting numeric values JSON cannot round-trip."""

    model_config = ConfigDict(extra="allow", allow_inf_nan=False)


class SourceProvenance(FiniteModel):
    """Original source position, retained across normalized exports and reanalysis."""

    source_path: str
    line: int = Field(ge=1)
    adapter: str
    adapter_version: str = "1"
    namespace: str | None = None
    legacy_ids: dict[str, str] = Field(default_factory=dict)


class Message(FiniteModel):
    """One chat message. ``role`` is a free string (providers add roles beyond the
    classic system/user/assistant/tool set); provider-specific keys survive via
    ``extra="allow"``."""

    model_config = ConfigDict(extra="allow")

    role: str
    content: str | list[dict[str, Any]] | None = None
    tool_calls: list[dict[str, Any]] | None = None


class StepTokens(FiniteModel):
    """Training-time token stream for one trajectory step.

    Field names match verifiers TrajectoryStepTokens exactly. Extra upstream keys
    (routed_experts, multi_modal_data, prompt_attribution) pass through.
    """

    model_config = ConfigDict(extra="allow")

    prompt_ids: list[int]
    prompt_mask: list[int]
    completion_ids: list[int]
    completion_mask: list[int]
    completion_logprobs: list[float]
    overlong_prompt: bool = False
    is_truncated: bool = False


class TimeSpan(FiniteModel):
    """A timed span in seconds (Unix timestamps), mirroring verifiers TimeSpan.
    ``duration`` is a computed field upstream and arrives already materialized in
    dumps, so it is stored as plain data here."""

    model_config = ConfigDict(extra="allow")

    start: float = 0.0
    end: float = 0.0
    duration: float | None = None


class Timing(FiniteModel):
    """Rollout-level timing, a permissive mirror of verifiers RolloutTiming (D-004).

    Every field is optional so partial or evolved upstream timing dicts round-trip
    losslessly instead of failing validation. ``model`` and ``env`` are TimeSpans
    containers upstream; their exact shape is passthrough.
    """

    model_config = ConfigDict(extra="allow")

    start_time: float | None = None
    setup: TimeSpan | None = None
    generation: TimeSpan | None = None
    scoring: TimeSpan | None = None
    model: Any = None
    env: Any = None
    total: float | None = None
    overhead: float | None = None


class TokenUsage(FiniteModel):
    """Token usage counters; ``final_input_tokens`` / ``final_output_tokens`` and any
    future upstream keys survive via ``extra="allow"``."""

    model_config = ConfigDict(extra="allow")

    input_tokens: float = 0.0
    output_tokens: float = 0.0


class TrajectoryStep(FiniteModel):
    """One environment turn, field-compatible with verifiers TrajectoryStep.

    ``response`` is the raw provider response object, kept as passthrough (D-009).
    """

    model_config = ConfigDict(extra="allow")

    prompt: list[Message] | str | None = None
    completion: list[Message] | str | None = None
    response: Any = None
    tokens: StepTokens | None = None
    reward: float | None = None
    advantage: float | None = None
    is_truncated: bool = False
    trajectory_id: str
    extras: dict[str, Any] = Field(default_factory=dict)


class RolloutBase(FiniteModel):
    """Fields shared by both rollout variants.

    Required fields mirror the verifiers RolloutOutput required set; everything
    upstream marks optional is optional here. The identity fields (rollout_id,
    group_id, run_id, step_index) default to None on raw rows and are attached by
    adapters via schema.ids (D-007); step_index comes from on-disk layout only and
    is never guessed.
    """

    model_config = ConfigDict(extra="allow")

    schema_version: str = SCHEMA_VERSION
    example_id: int
    reward: float
    metrics: dict[str, float] = Field(default_factory=dict)
    is_completed: bool
    is_truncated: bool
    timing: Timing = Field(default_factory=Timing)
    token_usage: TokenUsage | None = None
    answer: str | None = None
    info: dict[str, Any] = Field(default_factory=dict)
    error: dict[str, Any] | None = None
    stop_condition: str | None = None
    tool_defs: list[dict[str, Any]] | None = None
    # identity fields, adapter-attached, never part of upstream rows
    rollout_id: str | None = None
    group_id: str | None = None
    run_id: str | None = None
    environment_namespace: str | None = None
    task_namespace: str | None = None
    step_index: int | None = None
    occurrence_id: str | None = None
    content_fingerprint: str | None = None
    scoring_revision: str | None = None
    provenance: SourceProvenance | None = None
    identity_aliases: dict[str, str] = Field(default_factory=dict)

    @field_validator("schema_version")
    @classmethod
    def _version_supported(cls, version: str) -> str:
        if not re.fullmatch(r"\d+\.\d+(?:\.\d+)?", version):
            raise ValueError(f"unparseable schema_version {version!r}")
        if int(version.split(".")[0]) != int(SCHEMA_VERSION.split(".")[0]):
            raise ValueError(
                "schema version requires migration; use validate_rollout or read_rollouts"
            )
        return version


class SingleTurnRollout(RolloutBase):
    """A single-turn rollout: one prompt, one completion, no trajectory."""

    kind: Literal["single_turn"] = "single_turn"
    prompt: list[Message] | str | None = None
    completion: list[Message] | str | None = None


class MultiTurnRollout(RolloutBase):
    """A multi-turn rollout with a populated per-turn trajectory."""

    kind: Literal["multi_turn"] = "multi_turn"
    prompt: list[Message] | str | None = None
    completion: list[Message] | str | None = None
    trajectory: list[TrajectoryStep] = Field(default_factory=list)


Rollout: TypeAlias = Annotated[
    SingleTurnRollout | MultiTurnRollout,
    Field(discriminator="kind"),
]

ROLLOUT_ADAPTER: TypeAdapter[SingleTurnRollout | MultiTurnRollout] = TypeAdapter(Rollout)


class TrainingSignals(FiniteModel):
    """Optional sidecar for RL training signals (never on the base row).

    ``advantages`` and ``is_trainable`` exist only in memory during prime-rl
    training and are not in on-disk jsonl; v0 never parses them from disk. This
    model exists so the v1 monitor hook has a stable home, keyed on
    (run_id, rollout_id, step_index) per the join contract in schema.ids.
    """

    model_config = ConfigDict(extra="allow")

    rollout_id: str
    run_id: str | None = None
    step_index: int | None = None
    advantages: list[float] | None = None
    is_trainable: bool | None = None
    occurrence_id: str | None = None


def infer_kind(row: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a raw row dict with the ``kind`` discriminator filled in.

    Upstream verifiers rows carry no ``kind``; a row with a non-empty ``trajectory``
    is multi_turn, anything else is single_turn. Rows that already carry ``kind``
    are returned unchanged (no copy).
    """
    if "kind" in row:
        return row
    kind = "multi_turn" if row.get("trajectory") else "single_turn"
    return {**row, "kind": kind}


def validate_rollout(row: dict[str, Any]) -> SingleTurnRollout | MultiTurnRollout:
    """Validate a raw row dict into the right Rollout variant.

    Input: a JSON-decoded dict, with or without the ``kind`` discriminator.
    Output: a validated SingleTurnRollout or MultiTurnRollout. Raises
    pydantic.ValidationError on rows that do not fit the contract.
    """
    from rolloutscope.schema.migrate import migrate_row

    return ROLLOUT_ADAPTER.validate_python(infer_kind(migrate_row(row)))


def rollout_json_schema() -> dict[str, Any]:
    """Export the JSON Schema for the Rollout union.

    The output contains the discriminator mapping (propertyName ``kind``) so
    cross-language consumers can route variants without Python.
    """
    return ROLLOUT_ADAPTER.json_schema()
