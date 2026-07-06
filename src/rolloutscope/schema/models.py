"""Normalized rollout data contract for rolloutscope.

Adapted from the rollout-schema-design skill's candidate schema and kept
field-compatible with the verifiers RolloutOutput contract (verifiers @ 5885ab9c).
Upstream names win on any conflict. Deltas from the candidate are recorded in
PLAN.md (D-003 through D-009).

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

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

SCHEMA_VERSION = "1.0"


class Message(BaseModel):
    """One chat message. ``role`` is a free string (providers add roles beyond the
    classic system/user/assistant/tool set); provider-specific keys survive via
    ``extra="allow"``."""

    model_config = ConfigDict(extra="allow")

    role: str
    content: str | list[dict[str, Any]] | None = None
    tool_calls: list[dict[str, Any]] | None = None


class StepTokens(BaseModel):
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


class TimeSpan(BaseModel):
    """A timed span in seconds (Unix timestamps), mirroring verifiers TimeSpan.
    ``duration`` is a computed field upstream and arrives already materialized in
    dumps, so it is stored as plain data here."""

    model_config = ConfigDict(extra="allow")

    start: float = 0.0
    end: float = 0.0
    duration: float | None = None


class Timing(BaseModel):
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


class TokenUsage(BaseModel):
    """Token usage counters; ``final_input_tokens`` / ``final_output_tokens`` and any
    future upstream keys survive via ``extra="allow"``."""

    model_config = ConfigDict(extra="allow")

    input_tokens: float = 0.0
    output_tokens: float = 0.0


class TrajectoryStep(BaseModel):
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


class RolloutBase(BaseModel):
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
    step_index: int | None = None


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


class TrainingSignals(BaseModel):
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
    return ROLLOUT_ADAPTER.validate_python(infer_kind(row))


def rollout_json_schema() -> dict[str, Any]:
    """Export the JSON Schema for the Rollout union.

    The output contains the discriminator mapping (propertyName ``kind``) so
    cross-language consumers can route variants without Python.
    """
    return ROLLOUT_ADAPTER.json_schema()
