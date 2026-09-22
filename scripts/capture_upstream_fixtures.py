"""Capture sanitized upstream artifacts using the real upstream writers.

Adapter fixtures written by hand from reading a schema are guesses. These are
produced by importing the upstream models and calling the upstream serializers,
so the bytes on disk are the bytes those versions actually write. Only the
content is invented: task text, tool arguments, and model output are synthetic
placeholders, there is no real model output, no customer data, and no PII.

This script is the one place in the repository that imports verifiers. It is not
part of the package and nothing under ``src/`` imports it. Run it in a throwaway
environment so the project venv never gains a verifiers dependency:

    uv run --isolated --no-project --python 3.12 \
        --with 'verifiers==0.3.1' --with pyzstd --with orjson \
        python scripts/capture_upstream_fixtures.py

Three layouts are captured, because they are genuinely different files:

1. ``verifiers`` v1 eval: ``traces.jsonl``, one Episode per line, written by
   ``verifiers.v1.cli.output.write_episode``.
2. ``prime-rl`` legacy file monitor: ``rollouts/step_{step}/{kind}/{subset}/
   traces.jsonl``, both cohorts written separately.
3. ``prime-rl`` current file monitor: a chunked ``traces/stream`` directory with
   a sibling index and a per-producer annotation stream.

For the two prime-rl layouts the writer code is fetched from the pinned commit
and executed, rather than reimplemented here, so the index rows are the upstream
index rows.
"""

from __future__ import annotations

import inspect
import json
import shutil
import sys
import types
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_ROOT / "tests" / "fixtures" / "upstream"

VERIFIERS_VERSION = "0.3.1"
PRIME_RL_COMMIT = "dad79d1ce85390a2c818416ef100677f53f8c1ec"
PRIME_RL_TAG = "v0.9.1.dev46"
PRIME_RL_LEGACY_COMMIT = "84e7312f33e6"
PRIME_RL_LEGACY_TAG = "v0.9.1.dev28"

RAW = "https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl"


def _fetch_module(name: str, path: str, commit: str) -> types.ModuleType:
    """Fetch one upstream source file at a pinned commit and import it.

    Input: the module name to bind, the repository path, and the commit. Output:
    the imported module. Used for the two prime-rl writer helpers, which have no
    prime-rl imports of their own and so load standalone.
    """
    url = f"{RAW}/{commit}/{path}"
    with urllib.request.urlopen(url, timeout=60) as response:
        source = response.read().decode("utf-8")
    module = types.ModuleType(name)
    module.__dict__["__file__"] = url
    exec(compile(source, url, "exec"), module.__dict__)
    sys.modules[name] = module
    return module


def build_episodes() -> list[Any]:
    """Construct three sanitized episodes covering the shapes an adapter must handle.

    Output: a list of ``verifiers.v1.Episode`` objects: a grouped two-trace train
    episode carrying named weighted rewards and a test-editing tool call, a
    single-trace episode with a zero reward, and a failed episode that produced
    no trace at all. The last one exists because an adapter that drops
    trace-less episodes loses exactly the rows an operator most wants to see.
    """
    from verifiers.v1.configs.agent import WireAgentConfig
    from verifiers.v1.episode import (
        EnvInfo,
        Episode,
        GroupInfo,
        PolicySpan,
        TrainRunInfo,
        TrainWorkInfo,
    )
    from verifiers.v1.trace import AgentInfo, Error, Reward, Trace, TraceTask

    def task(index: int) -> TraceTask:
        return TraceTask(
            type="CodingTask",
            data={
                "prompt": f"Make the failing test in module_{index:02d} pass.",
                "answer": "the implementation is corrected, not the test",
            },
            key=f"coding/module_{index:02d}",
            hash=f"{index:064x}",
        )

    def trace(index: int, *, edits_test: bool, reward: float) -> Trace:
        assistant = (
            "Editing tests/test_module.py to relax the assertion."
            if edits_test
            else "Fixing the off-by-one in src/module.py."
        )
        tool_arguments = (
            {"path": "tests/test_module.py", "new": "assert True"}
            if edits_test
            else {"path": "src/module.py", "new": "return total + 1"}
        )
        return Trace(
            id=f"trace-{index:04d}",
            task=task(index),
            agent=AgentInfo(config=WireAgentConfig(), name="solver", trainable=True),
            nodes=[
                {
                    "message": {"role": "system", "content": "You are a coding agent."},
                    "sampled": False,
                },
                {
                    "parent": 0,
                    "message": {
                        "role": "user",
                        "content": f"Make the failing test in module_{index:02d} pass.",
                    },
                    "sampled": False,
                },
                {
                    "parent": 1,
                    "message": {
                        "role": "assistant",
                        "content": assistant,
                        # verifiers v1 ToolCall is flat: id / type / name /
                        # arguments. It is NOT the OpenAI-nested
                        # {"function": {"name", "arguments"}} shape the legacy
                        # rows used, and that difference is load bearing for any
                        # reader that scans tool-call text.
                        "tool_calls": [
                            {
                                "id": f"call-{index}",
                                "type": "function",
                                "name": "Edit",
                                "arguments": json.dumps(tool_arguments, sort_keys=True),
                            }
                        ],
                    },
                    "sampled": True,
                    "token_ids": [11, 12, 13, 14],
                    "mask": [True, True, True, True],
                },
                {
                    "parent": 2,
                    "message": {
                        "role": "tool",
                        "tool_call_id": f"call-{index}",
                        "content": "1 file changed, 1 insertion(+), 1 deletion(-)",
                    },
                    "sampled": False,
                },
            ],
            rewards={
                "tests_pass": Reward(score=reward, weight=1.0),
                "format_reward": Reward(score=1.0, weight=0.2),
                "unscored_judge": None,
            },
            metrics={"correctness": 0.0 if edits_test else reward, "num_edits": 1.0},
            info={"kind": "train", "advantage": 0.5 if edits_test else -0.5},
            is_completed=True,
            ok=True,
            stop_condition="stop",
        )

    run = TrainRunInfo(
        id="run-0001",
        name="sanitized-capture",
        work=TrainWorkInfo(type="train", step=0, policy=PolicySpan(start=0, end=0)),
    )
    env = EnvInfo(id="rolloutscope-fixture+coding", name="coding")

    grouped = Episode(
        id="episode-0001",
        env=env,
        task=task(1),
        group=GroupInfo(id="group-0001"),
        run=run,
        ok=True,
        traces=[
            trace(1, edits_test=True, reward=1.0),
            trace(2, edits_test=False, reward=1.0),
        ],
    )
    solo = Episode(
        id="episode-0002",
        env=env,
        task=task(3),
        group=GroupInfo(id="group-0002"),
        run=run,
        ok=True,
        traces=[trace(3, edits_test=False, reward=0.0)],
    )
    failed = Episode(
        id="episode-0003",
        env=env,
        task=task(4),
        group=GroupInfo(id="group-0002"),
        run=run,
        ok=False,
        errors=[Error(type="TimeoutError", message="the sandbox did not respond in 600s")],
        traces=[],
    )
    return [grouped, solo, failed]


def capture_verifiers_eval(episodes: list[Any], root: Path) -> Path:
    """Write ``traces.jsonl`` with the verifiers v1 eval writer."""
    from verifiers.v1.cli.output import write_episode

    out = root / f"verifiers-{VERIFIERS_VERSION}-eval"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / "traces.jsonl").write_text("")
    for episode in episodes:
        write_episode(out, episode)
    return out


def capture_prime_rl_legacy(episodes: list[Any], root: Path) -> Path:
    """Write the per-step cohort layout the legacy file monitor produced.

    Both cohorts are written, because that is what the legacy monitor did: the
    ``all`` cohort as episodes complete and the ``effective`` cohort as one batch
    on finalize. The overlap between the two directories is the double-count
    hazard a recursive reader has to be told about.
    """
    import orjson

    options = orjson.OPT_APPEND_NEWLINE | orjson.OPT_SERIALIZE_NUMPY
    out = root / f"prime-rl-{PRIME_RL_LEGACY_TAG}-legacy"
    if out.exists():
        shutil.rmtree(out)

    def write(subset: str, selected: list[Any]) -> None:
        path = out / "rollouts" / "step_0" / "train" / subset / "traces.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("ab") as handle:
            for episode in selected:
                handle.write(orjson.dumps(episode.to_record(), default=str, option=options))

    write("all", episodes)
    # The effective cohort is the subset that survived filtering: the failed
    # episode and the zero-reward one are gone, so it is a strict subset here.
    write("effective", episodes[:1])
    return out


def capture_prime_rl_stream(episodes: list[Any], root: Path) -> Path:
    """Write the chunked stream layout the current file monitor produces.

    The index rows come from the upstream ``index_row`` at the pinned commit, not
    from a local reimplementation, so the fixture's index is the real one.
    Compression is off, so the chunk stays a readable ``.jsonl`` and the fixture
    does not need a zstd reader to inspect.
    """
    import orjson

    chunks = _fetch_module(
        "prime_rl_chunks", "src/prime_rl/monitors/file/traces/chunks.py", PRIME_RL_COMMIT
    )
    index_module = _fetch_module(
        "prime_rl_index", "src/prime_rl/monitors/file/traces/index.py", PRIME_RL_COMMIT
    )

    options = orjson.OPT_APPEND_NEWLINE | orjson.OPT_SERIALIZE_NUMPY
    out = root / f"prime-rl-{PRIME_RL_TAG}-stream"
    if out.exists():
        shutil.rmtree(out)

    # The current monitor calls `to_record(float_decimals=...)`, which verifiers
    # only grew after v0.3.1 (commit cfc7c474, 2026-09-02, still unreleased at
    # capture time) even though prime-rl declares `verifiers>=0.3.1`. Adapt
    # rather than fail, and let PROVENANCE record which serialization ran.
    rounds_floats = "float_decimals" in inspect.signature(episodes[0].to_record).parameters

    stream_dir = out / "traces" / "stream"
    index_path = stream_dir.with_name(stream_dir.name + ".index.jsonl")
    stream = chunks.ChunkedJsonl(stream_dir, 1 << 20, False)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("ab") as index:
        for line, episode in enumerate(episodes, start=1):
            record = episode.to_record(float_decimals=6) if rounds_floats else episode.to_record()
            chunk, offset = stream.append(orjson.dumps(record, default=str, option=options))
            index.write(
                orjson.dumps(
                    index_module.index_row(line, record, chunk, offset),
                    default=str,
                    option=options,
                )
            )
    stream.flush()
    stream.close()

    # The effective cohort writes no second copy in this layout. What it learns
    # arrives here instead, as annotations a reader folds back onto the stream.
    annotations = out / "traces" / "annotations" / "orchestrator"
    annotations.mkdir(parents=True, exist_ok=True)
    (annotations / "00000.jsonl").write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in (
                {"trace_id": "trace-0001", "subset": "effective", "advantage": 0.5, "step": 0},
                {"trace_id": "trace-0002", "subset": "effective", "advantage": -0.5, "step": 0},
                {"trace_id": "trace-0003", "subset": "all", "advantage": 0.0, "step": 0},
            )
        ),
        encoding="utf-8",
    )
    return out


def main() -> int:
    """Capture every layout and print what was written."""
    episodes = build_episodes()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    written = [
        capture_verifiers_eval(episodes, OUT_ROOT),
        capture_prime_rl_legacy(episodes, OUT_ROOT),
        capture_prime_rl_stream(episodes, OUT_ROOT),
    ]
    for directory in written:
        print(f"wrote {directory.relative_to(REPO_ROOT)}")
        for path in sorted(p for p in directory.rglob("*") if p.is_file()):
            print(f"  {path.relative_to(directory)}  {path.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
