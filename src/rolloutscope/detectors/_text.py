"""Shared extraction helpers for detectors: one copy of the completion logic.

Completions arrive as ``list[Message] | str | None`` (and multi-turn rollouts
carry per-step completions in ``trajectory``); every detector needs the same
text extraction, stable-id fallback, and grouping helpers, so they live here
once. Offsets in evidence spans refer to the extracted text forms these
helpers produce.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence

from rolloutscope.schema import Message, MultiTurnRollout, Rollout, SingleTurnRollout
from rolloutscope.schema import group_id as derive_group_id
from rolloutscope.schema import rollout_id as derive_rollout_id

_TOKEN_RE = re.compile(r"\w+")
_WHITESPACE_RE = re.compile(r"\s+")


def message_text(message: Message, include_tool_calls: bool = False) -> str:
    """Extract readable text from one message.

    Concatenates string content and the ``text`` values of structured content
    parts. When ``include_tool_calls`` is True, each tool call is rendered as
    ``name(arguments)`` so pattern scanners see tool activity too.
    """
    parts: list[str] = []
    content = message.content
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for item in content:
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
    if include_tool_calls and message.tool_calls:
        for call in message.tool_calls:
            function = call.get("function")
            if isinstance(function, dict):
                name = function.get("name") or ""
                arguments = function.get("arguments")
                rendered_args = arguments if isinstance(arguments, str) else ""
                parts.append(f"{name}({rendered_args})")
    return "\n".join(parts)


def messages_text(
    value: list[Message] | str | None,
    include_tool_calls: bool = False,
) -> str:
    """Extract text from a completion-shaped value (message list, str, or None)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    chunks = [message_text(message, include_tool_calls) for message in value]
    return "\n".join(chunk for chunk in chunks if chunk)


def completion_sources(
    rollout: Rollout,
    include_tool_calls: bool = False,
) -> list[tuple[str, str]]:
    """Return (field, text) pairs for every completion-bearing field.

    Yields ``("completion", ...)`` when the top-level completion has text and,
    for multi-turn rollouts, one ``("trajectory[i].completion", ...)`` entry
    per non-empty step. Evidence span offsets refer to these extracted texts.
    """
    sources: list[tuple[str, str]] = []
    text = messages_text(rollout.completion, include_tool_calls)
    if text:
        sources.append(("completion", text))
    if isinstance(rollout, MultiTurnRollout):
        for index, step in enumerate(rollout.trajectory):
            step_text = messages_text(step.completion, include_tool_calls)
            if step_text:
                sources.append((f"trajectory[{index}].completion", step_text))
    return sources


def primary_completion(rollout: Rollout) -> tuple[str, str]:
    """Return a best-effort single (field, text) for the rollout's output.

    Prefers the top-level completion; falls back to the last non-empty
    trajectory step for multi-turn rollouts; returns an empty text (with field
    ``completion``) when nothing is present.
    """
    text = messages_text(rollout.completion)
    if text:
        return ("completion", text)
    if isinstance(rollout, MultiTurnRollout):
        for index in range(len(rollout.trajectory) - 1, -1, -1):
            step_text = messages_text(rollout.trajectory[index].completion)
            if step_text:
                return (f"trajectory[{index}].completion", step_text)
    return ("completion", "")


def iter_tool_calls(rollout: Rollout) -> Iterator[tuple[str, str, str]]:
    """Yield (field, tool_name, arguments_text) for every tool call on a rollout.

    Scans the top-level completion messages and, for multi-turn rollouts,
    every trajectory step's completion messages. Non-string arguments are
    yielded as empty strings rather than guessed at.
    """

    def _from_messages(
        value: list[Message] | str | None, field: str
    ) -> Iterator[tuple[str, str, str]]:
        if not isinstance(value, list):
            return
        for message in value:
            for call in message.tool_calls or []:
                function = call.get("function")
                if isinstance(function, dict):
                    name = str(function.get("name") or "")
                    arguments = function.get("arguments")
                    yield (field, name, arguments if isinstance(arguments, str) else "")

    yield from _from_messages(rollout.completion, "completion")
    if isinstance(rollout, MultiTurnRollout):
        for index, step in enumerate(rollout.trajectory):
            yield from _from_messages(step.completion, f"trajectory[{index}].completion")


def stable_rollout_id(rollout: Rollout) -> str:
    """Return the rollout's id, deriving a stable content-based one if unset.

    Hand-built fixtures may omit ``rollout_id``; the fallback reuses the
    schema's content-derived id over (example_id, prompt, completion, reward),
    so the same row always maps to the same id.
    """
    if rollout.rollout_id:
        return rollout.rollout_id
    dumped = rollout.model_dump(mode="json", include={"prompt", "completion"})
    return derive_rollout_id(
        rollout.example_id, dumped.get("prompt"), dumped.get("completion"), rollout.reward
    )


def stable_group_id(rollout: Rollout) -> str:
    """Return the rollout's group id, falling back to the example_id grouping key."""
    return rollout.group_id or derive_group_id(rollout.example_id)


def word_tokens(text: str) -> list[str]:
    """Lowercased word tokens (``\\w+`` runs) of a text."""
    return _TOKEN_RE.findall(text.lower())


def normalize_text(text: str) -> str:
    """Casefold a text and collapse every whitespace run to a single space."""
    return _WHITESPACE_RE.sub(" ", text).strip().casefold()


def flexible_find(haystack: str, needle: str) -> tuple[int, int] | None:
    """Locate ``needle`` in ``haystack`` ignoring case and whitespace runs.

    Returns (start, end) character offsets into the original haystack, or None
    when not found. Used to map normalized-text matches back to raw offsets.
    """
    parts = needle.split()
    if not parts:
        return None
    pattern = r"\s+".join(re.escape(part) for part in parts)
    match = re.search(pattern, haystack, re.IGNORECASE)
    if match is None:
        return None
    return (match.start(), match.end())


def matching_metric_keys(metrics: dict[str, float], patterns: Sequence[str]) -> list[str]:
    """Metric keys whose lowercased name contains any of the given substrings."""
    lowered = [pattern.lower() for pattern in patterns]
    return [key for key in metrics if any(pattern in key.lower() for pattern in lowered)]


def distinct_steps(rollouts: Sequence[Rollout]) -> list[int]:
    """Sorted distinct non-None step_index values across the rollouts."""
    return sorted({r.step_index for r in rollouts if r.step_index is not None})


def clamp01(value: float) -> float:
    """Clamp a float into [0.0, 1.0]."""
    return max(0.0, min(1.0, value))


__all__ = [
    "MultiTurnRollout",
    "SingleTurnRollout",
    "clamp01",
    "completion_sources",
    "distinct_steps",
    "flexible_find",
    "iter_tool_calls",
    "matching_metric_keys",
    "message_text",
    "messages_text",
    "normalize_text",
    "primary_completion",
    "stable_group_id",
    "stable_rollout_id",
    "word_tokens",
]
