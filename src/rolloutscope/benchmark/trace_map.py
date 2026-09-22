"""The one TRACE row to normalized rollout mapping.

Before this module there were two mappings. The integration test flattened every
text-ish field of a row, user instructions and simulated tool output included,
into a single ``completion`` string. The validation script parsed assistant
messages, remapped tool calls, and dropped tool results entirely. They sampled
different row counts and disagreed on what a detector was even looking at, so
the two reported different numbers for the same dataset. Both callers now use
this module, and any change to the mapping bumps :data:`TRACE_MAPPING_VERSION`
and therefore invalidates every manifest and report that pinned the old one.

What a TRACE row looks like (measured at revision ``31d87f06``, 517 rows):

- ``trajectory_id``: ``"trajectory_0389"`` style, unique per row.
- ``label``: see :mod:`rolloutscope.benchmark.taxonomy`.
- ``conversation``: a JSON string holding a list of messages. Only two roles
  occur, ``user`` (3576 messages) and ``assistant`` (10125). An assistant
  message may carry ``tool_calls`` (``{"name", "parameters"}``) and, on the same
  message, ``tool_results`` (``{"output"}``) holding what those calls returned.

That last shape is the reason the mapping matters. TRACE staples the simulated
tool output onto the assistant message that requested it, so a naive flatten
makes observed output indistinguishable from something the assistant wrote. A
tool result that happens to print ``assert True`` then reads as the assistant
writing ``assert True``. This mapping keeps them apart:

- ``prompt`` holds the opening user message or messages: the task statement.
- ``completion`` holds assistant-authored messages only. Nothing the model
  merely saw appears here.
- ``trajectory`` holds one step per assistant turn. ``step.prompt`` is what the
  model observed since its previous turn (tool results from the preceding turn,
  plus any user message in between) and ``step.completion`` is the single
  assistant message it produced. Turn index and per-turn spans survive.

Known consequence, stated because it moves reported numbers: carrying tool
results at all gives the current detectors more text to match against than the
old script gave them, because none of them yet distinguish an assistant action
from observed content. That work is Phase 2's detector repair. Fidelity comes
first here; a mapping that hides the input in order to flatter a detector is not
a benchmark.

What TRACE does not provide: reward, metrics, and the reference answer. This
module records their absence in ``info`` rather than inventing values, and
:mod:`rolloutscope.benchmark.applicability` reads that record to decide which
detectors may be scored on a row at all.
"""

from __future__ import annotations

import json
import re
from typing import Any

from rolloutscope.benchmark.splits import scenario_key
from rolloutscope.schema import Message, MultiTurnRollout, TrajectoryStep, validate_rollout

TRACE_MAPPING_VERSION = "1"
"""Version of the row-to-rollout mapping.

Bump on any change that could move a detector's inputs. Manifests and benchmark
reports record it, and a report produced under a different version is not
comparable with one produced under this one.
"""

INFO_KEY = "rolloutscope_benchmark"
"""Key in ``info`` under which this module records provenance and availability."""

_ID_DIGITS = re.compile(r"(\d+)")


class TraceRowError(ValueError):
    """A TRACE row could not be mapped, with the reason attached."""


def _example_id(trajectory_id: str, fallback: int) -> int:
    """Derive the integer ``example_id`` the schema requires from a row id.

    Input: the ``trajectory_id`` cell and a positional fallback. Output: the
    trailing digit run of the id (``"trajectory_0389"`` gives ``389``), or the
    fallback when the id carries no digits. The digit form is preferred because
    it is stable under reordering, which the positional fallback is not.
    """
    digits = _ID_DIGITS.findall(trajectory_id)
    return int(digits[-1]) if digits else fallback


def _tool_call(call: dict[str, Any], call_id: str) -> dict[str, Any]:
    """Render one TRACE tool call in the OpenAI-style shape detectors read.

    Input: a ``{"name", "parameters"}`` entry and the id to stamp on it. Output:
    a ``{"id", "type", "function": {"name", "arguments"}}`` dict, arguments
    JSON-encoded. Parameters that will not serialize fall back to ``repr`` so a
    single odd row degrades to a readable argument string instead of failing.
    """
    parameters = call.get("parameters", {})
    try:
        arguments = json.dumps(parameters, sort_keys=True)
    except (TypeError, ValueError):
        arguments = repr(parameters)
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": str(call.get("name") or ""), "arguments": arguments},
    }


def _observation_messages(message: dict[str, Any], turn: int) -> list[Message]:
    """Turn one assistant message's ``tool_results`` into role-``tool`` messages.

    Input: the raw assistant message and its zero-based turn index. Output: one
    ``Message`` per result, paired positionally with the turn's tool calls so a
    result carries the ``tool_call_id`` of the call it answers. Results beyond
    the number of calls (51 messages in the artifact are uneven) are still
    emitted, with no ``tool_call_id``, rather than dropped.
    """
    results = message.get("tool_results") or []
    calls = message.get("tool_calls") or []
    messages: list[Message] = []
    for index, result in enumerate(results):
        if not isinstance(result, dict):
            continue
        payload: dict[str, Any] = {
            "role": "tool",
            "content": str(result.get("output") or ""),
            "turn_index": turn,
        }
        if index < len(calls):
            payload["tool_call_id"] = f"call-{turn}-{index}"
        messages.append(Message.model_validate(payload))
    return messages


def map_row(
    row: dict[str, Any],
    *,
    position: int = 0,
    dataset: str = "PatronusAI/trace-dataset",
    revision: str | None = None,
) -> MultiTurnRollout:
    """Map one raw TRACE row into a normalized multi-turn rollout.

    Input: the raw row dict (``trajectory_id``, ``conversation``, ``label``), its
    position in the fetch (used only as an ``example_id`` fallback), and the
    dataset id and revision to record as provenance. Output: a validated
    :class:`MultiTurnRollout` whose ``completion`` is assistant-authored text
    only and whose ``trajectory`` preserves turn boundaries and observed content.

    Raises :class:`TraceRowError` when the row has no parseable conversation or
    contains no assistant turn, so a caller can count the row as unusable with a
    reason rather than scoring an empty rollout.
    """
    trajectory_id = str(row.get("trajectory_id") or f"row-{position}")
    label = row.get("label")
    if not isinstance(label, str):
        raise TraceRowError(f"{trajectory_id}: label is missing or not a string")

    raw_conversation = row.get("conversation")
    if not isinstance(raw_conversation, str):
        raise TraceRowError(f"{trajectory_id}: conversation is missing or not a string")
    try:
        messages = json.loads(raw_conversation)
    except (json.JSONDecodeError, TypeError) as exc:
        raise TraceRowError(f"{trajectory_id}: conversation is not valid JSON ({exc})") from exc
    if not isinstance(messages, list) or not messages:
        raise TraceRowError(f"{trajectory_id}: conversation is not a non-empty list")

    prompt: list[Message] = []
    completion: list[Message] = []
    steps: list[TrajectoryStep] = []
    observed: list[Message] = []
    turn = 0
    first_user_text = ""
    tool_call_count = 0
    tool_result_count = 0

    for entry in messages:
        if not isinstance(entry, dict):
            continue
        role = entry.get("role")
        content = entry.get("content")
        text = content if isinstance(content, str) else ""

        if role == "user":
            if not first_user_text:
                first_user_text = text
            user_message = Message(role="user", content=text)
            if not completion:
                # Still in the opening instruction, before the model has spoken.
                prompt.append(user_message)
            else:
                observed.append(user_message)
            continue

        if role != "assistant":
            # No third role occurs in the artifact; keep an unexpected one as
            # observed context rather than silently discarding the row's data.
            observed.append(Message(role=str(role or "unknown"), content=text))
            continue

        raw_calls = [c for c in (entry.get("tool_calls") or []) if isinstance(c, dict)]
        tool_call_count += len(raw_calls)
        calls = [_tool_call(call, f"call-{turn}-{i}") for i, call in enumerate(raw_calls)]
        assistant = Message.model_validate(
            {
                "role": "assistant",
                "content": text,
                "tool_calls": calls or None,
                "turn_index": turn,
            }
        )
        completion.append(assistant)
        steps.append(
            TrajectoryStep(
                trajectory_id=f"{trajectory_id}:{turn}",
                prompt=list(observed) or None,
                completion=[assistant],
                extras={"turn_index": turn},
            )
        )
        observations = _observation_messages(entry, turn)
        tool_result_count += len(observations)
        observed = observations
        turn += 1

    if not completion:
        raise TraceRowError(f"{trajectory_id}: conversation contains no assistant turn")

    key = scenario_key(first_user_text) if first_user_text else scenario_key(trajectory_id)
    provenance: dict[str, Any] = {
        "dataset": dataset,
        "revision": revision,
        "trajectory_id": trajectory_id,
        "label": label,
        "mapping_version": TRACE_MAPPING_VERSION,
        "scenario_key": key,
        "num_turns": turn,
        "num_tool_calls": tool_call_count,
        "num_tool_results": tool_result_count,
        # TRACE ships transcripts only. Recording the absence keeps a detector
        # that needs one of these out of the scored population instead of
        # letting a placeholder read as a confident negative.
        "reward_available": False,
        "metrics_available": False,
        "answer_available": False,
        "step_index_available": False,
    }

    return validate_rollout(
        {
            "kind": "multi_turn",
            "example_id": _example_id(trajectory_id, position),
            "rollout_id": f"trace-{trajectory_id}",
            "group_id": key,
            "run_id": f"trace@{revision}" if revision else "trace",
            "prompt": [m.model_dump(exclude_none=True) for m in prompt] or None,
            "completion": [m.model_dump(exclude_none=True) for m in completion],
            "trajectory": [s.model_dump(exclude_none=True) for s in steps],
            "reward": 0.0,
            "metrics": {},
            "answer": None,
            "info": {INFO_KEY: provenance},
            "is_completed": True,
            "is_truncated": False,
        }
    )  # type: ignore[return-value]


def provenance_of(rollout: MultiTurnRollout) -> dict[str, Any]:
    """Return the benchmark provenance block a mapped rollout carries.

    Input: a rollout produced by :func:`map_row`. Output: the ``info`` block
    written by the mapper, or an empty dict for a rollout this module did not
    produce.
    """
    block = rollout.info.get(INFO_KEY)
    return block if isinstance(block, dict) else {}
