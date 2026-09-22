"""The captured upstream artifacts, and the gaps between them and this reader.

These are draft adapter fixtures. No adapter consumes them yet: that work waits
on the Phase 1 identity and provenance contract. What this file does is hold the
shape claims in ``docs/compatibility-matrix.md`` to the actual bytes, so a claim
cannot quietly go stale, and pin the known gaps so a future adapter has an
executable description of what it has to fix.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from rolloutscope.detectors._text import message_text
from rolloutscope.schema import Message, validate_rollout

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "upstream"
VERIFIERS_EVAL = FIXTURES / "verifiers-0.3.1-eval"
LEGACY = FIXTURES / "prime-rl-v0.9.1.dev28-legacy"
STREAM = FIXTURES / "prime-rl-v0.9.1.dev46-stream"


def _episodes(path: Path) -> list[dict[str, Any]]:
    """Read one episode per line from a captured jsonl file."""
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


@pytest.fixture(scope="module")
def episodes() -> list[dict[str, Any]]:
    """The three captured episodes, from the verifiers eval writer."""
    return _episodes(VERIFIERS_EVAL / "traces.jsonl")


def test_the_captured_layouts_are_all_present() -> None:
    """A missing capture means the matrix is describing something absent."""
    assert (VERIFIERS_EVAL / "traces.jsonl").is_file()
    assert (LEGACY / "rollouts" / "step_0" / "train" / "all" / "traces.jsonl").is_file()
    assert (LEGACY / "rollouts" / "step_0" / "train" / "effective" / "traces.jsonl").is_file()
    assert (STREAM / "traces" / "stream" / "00000.jsonl").is_file()
    assert (STREAM / "traces" / "stream.index.jsonl").is_file()
    assert (STREAM / "traces" / "annotations" / "orchestrator" / "00000.jsonl").is_file()


def test_an_episode_record_has_no_top_level_example_id_or_reward(
    episodes: list[dict[str, Any]],
) -> None:
    """The first and largest gap: the required fields simply are not there."""
    for episode in episodes:
        assert set(episode) <= {"id", "env", "task", "group", "run", "ok", "errors", "traces"}
        assert "example_id" not in episode
        assert "reward" not in episode


def test_the_current_normalizer_rejects_an_episode_record(
    episodes: list[dict[str, Any]],
) -> None:
    """Pointing today's schema at a v1 record fails outright, it does not degrade.

    Pinned deliberately. When an adapter for this format lands, this test is the
    one that has to change, and changing it is the moment to check that the
    identity contract was honoured.
    """
    with pytest.raises(ValidationError):
        validate_rollout(episodes[0])


def test_reward_is_per_trace_named_and_weighted(episodes: list[dict[str, Any]]) -> None:
    """The scalar is sum(score * weight), and a null means scoring did not run."""
    rewards = episodes[0]["traces"][0]["rewards"]
    assert rewards["tests_pass"] == {"score": 1.0, "weight": 1.0}
    assert rewards["format_reward"] == {"score": 1.0, "weight": 0.2}
    assert rewards["unscored_judge"] is None

    scalar = sum(r["score"] * r["weight"] for r in rewards.values() if isinstance(r, dict))
    assert scalar == pytest.approx(1.2)

    index = [
        json.loads(line)
        for line in (STREAM / "traces" / "stream.index.jsonl").read_text().splitlines()
    ]
    assert index[0]["reward"] == pytest.approx(scalar)


def test_tool_calls_are_flat_not_openai_nested(episodes: list[dict[str, Any]]) -> None:
    """verifiers v1 ToolCall is {id, type, name, arguments}."""
    nodes = episodes[0]["traces"][0]["nodes"]
    assistant = next(n for n in nodes if n["message"]["role"] == "assistant")
    call = assistant["message"]["tool_calls"][0]
    assert set(call) == {"id", "type", "name", "arguments"}
    assert "function" not in call


def test_the_flat_tool_call_shape_is_invisible_to_the_current_detectors(
    episodes: list[dict[str, Any]],
) -> None:
    """The quietest gap of all, and the reason it needs a test.

    ``_text.message_text`` reads ``call["function"]["name"]``. Given a v1 tool
    call it finds no ``function`` key, renders nothing, and every tool-call
    signal disappears with no error and no warning. A run would look clean.
    """
    nodes = episodes[0]["traces"][0]["nodes"]
    assistant = next(n for n in nodes if n["message"]["role"] == "assistant")
    message = Message.model_validate(assistant["message"])
    rendered = message_text(message, include_tool_calls=True)
    assert "Editing tests/test_module.py" in rendered
    assert "Edit(" not in rendered
    assert 'tests/test_module.py"}' not in rendered


def test_identity_is_supplied_by_upstream(episodes: list[dict[str, Any]]) -> None:
    """Nothing here needs guessing or content hashing to identify a row."""
    episode = episodes[0]
    assert episode["id"] == "episode-0001"
    assert episode["group"]["id"] == "group-0001"
    assert episode["run"]["id"] == "run-0001"
    assert episode["run"]["work"]["step"] == 0
    assert episode["env"]["id"] == "rolloutscope-fixture+coding"
    trace = episode["traces"][0]
    assert trace["id"] == "trace-0001"
    assert trace["task"]["key"] == "coding/module_01"
    assert trace["task"]["hash"]


def test_a_failed_episode_carries_errors_and_no_traces(
    episodes: list[dict[str, Any]],
) -> None:
    """An adapter that keys on traces drops exactly the rows operators want."""
    failed = episodes[2]
    assert failed["ok"] is False
    assert failed["traces"] == []
    assert failed["errors"][0]["type"] == "TimeoutError"


def test_one_episode_can_hold_several_agent_traces(
    episodes: list[dict[str, Any]],
) -> None:
    """Flattening every trace into one rollout would mix separate agent seats."""
    assert len(episodes[0]["traces"]) == 2
    assert {t["id"] for t in episodes[0]["traces"]} == {"trace-0001", "trace-0002"}


def test_the_legacy_layout_writes_the_effective_cohort_twice() -> None:
    """The double-count hazard, as bytes.

    An episode that survived filtering exists under both ``all`` and
    ``effective`` with one episode id. Deduplicating on identity is correct;
    concatenating the paths is not.
    """
    root = LEGACY / "rollouts" / "step_0" / "train"
    every = _episodes(root / "all" / "traces.jsonl")
    effective = _episodes(root / "effective" / "traces.jsonl")
    assert [e["id"] for e in every] == ["episode-0001", "episode-0002", "episode-0003"]
    assert [e["id"] for e in effective] == ["episode-0001"]
    assert effective[0] == every[0]

    naive_concatenation = every + effective
    assert len(naive_concatenation) == 4
    assert len({e["id"] for e in naive_concatenation}) == 3


def test_the_stream_layout_writes_each_episode_exactly_once() -> None:
    """The current monitor returns early on the effective cohort."""
    stream = _episodes(STREAM / "traces" / "stream" / "00000.jsonl")
    assert [e["id"] for e in stream] == ["episode-0001", "episode-0002", "episode-0003"]
    assert len({e["id"] for e in stream}) == len(stream)


def test_the_index_addresses_each_episode_without_parsing_the_stream() -> None:
    """The index row carries chunk and byte offset, and the offsets are real."""
    rows = [
        json.loads(line)
        for line in (STREAM / "traces" / "stream.index.jsonl").read_text().splitlines()
    ]
    chunk = (STREAM / "traces" / "stream" / "00000.jsonl").read_bytes()
    assert [row["line"] for row in rows] == [1, 2, 3]
    for row in rows:
        assert row["chunk"] == 0
        seeked = json.loads(chunk[row["offset"] :].split(b"\n", 1)[0])
        assert seeked["id"] == row["id"]


def test_the_index_reports_a_failed_episode_as_unmeasured_not_as_zero() -> None:
    """Upstream itself distinguishes "no reward" from "reward 0.0"."""
    rows = [
        json.loads(line)
        for line in (STREAM / "traces" / "stream.index.jsonl").read_text().splitlines()
    ]
    failed = rows[2]
    assert failed["ok"] is False
    assert failed["reward"] is None
    assert failed["turns"] == 0
    assert failed["trace_ids"] == []


def test_cohort_membership_lives_in_the_annotation_stream() -> None:
    """In the current layout the cohort is a field to fold, not a directory."""
    updates = _episodes(STREAM / "traces" / "annotations" / "orchestrator" / "00000.jsonl")
    effective = {u["trace_id"] for u in updates if u["subset"] == "effective"}
    assert effective == {"trace-0001", "trace-0002"}
    assert any(u["advantage"] != 0.0 for u in updates)
