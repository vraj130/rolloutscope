"""J1 step b, part (ii): synthetic failing answers, graded by the proxy judge.

For each eval criterion, take an answer the gold judge marked as satisfying it (the earliest
eval step across the six M2 runs, step 0 first) and have the local Llama-3.1-8B-Instruct
rewrite it with one targeted edit so it fails the criterion:

- delete:       delete the passages that satisfy the criterion (all types except negation)
- negate:       reverse the key claim; for a "must not" criterion, add the forbidden content
                (negation and factual)
- swap_numbers: replace the numbers the criterion checks with wrong ones (numeric)
- false_fact:   replace the correct information with a plausible false statement
                (factual and numeric)

Each edited answer is graded by the proxy judge exactly as in training (full rubric, one
call, judge.py); the verdict on the target criterion is the audit result. An edit that leaves
the text unchanged, or a delete that does not shorten it, is marked invalid and excluded.
No gold calls: "fails the criterion" holds by construction, not by a judge.

Part (i) of the audit (step-0 answers gold marked "no") needs no calls; analyze_j1.py reads
it from the logged verdicts. Writes $ROLLOUTSCOPE_DATA/j1/audit_edits.jsonl.

    uv run python j1/audit.py
"""

# ruff: noqa: E501  (the prompt strings are kept verbatim, as run)

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

sys.path.insert(0, str(Path(__file__).parent.parent / "m1"))

from j1data import RUNS, criteria_table, criterion_objects, load_run, out_dir
from judge import ProxyJudge, check_health

EDITS = {
    "delete": "delete every sentence or passage that satisfies the criterion, so the answer "
    "clearly no longer satisfies it.",
    "negate": "change the content the criterion checks so that it states or does the opposite "
    "(if the criterion forbids something, add a sentence that does exactly that), so the "
    "answer clearly fails the criterion.",
    "swap_numbers": "replace the numbers the criterion checks (doses, ranges, thresholds, "
    "durations, counts) with clearly wrong numbers, so the answer clearly fails the criterion.",
    "false_fact": "replace the correct information the criterion checks with a plausible but "
    "false medical statement, so the answer clearly fails the criterion.",
}
BY_TYPE = {
    "structural": ["delete"],
    "presence": ["delete"],
    "compound": ["delete"],
    "negation": ["negate"],
    "numeric": ["delete", "swap_numbers", "false_fact"],
    "factual": ["delete", "negate", "false_fact"],
}
EDIT_PROMPT = """Here is an answer to a medical question and one grading criterion that the answer currently satisfies.

Criterion: {criterion}

Answer:
<<<
{answer}
>>>

Rewrite the answer with one change: {instruction} Keep all other text exactly as it is. Output only the rewritten answer, with no comments and no markers."""


def pick_sources(ids: list[str]) -> dict[str, dict]:
    """Earliest gold-yes answer per criterion over the six runs (step 0 first)."""
    best: dict[str, dict] = {}
    for run in RUNS:
        for step, rows in load_run(run).items():
            for src, r in rows.items():
                gv, pv = r["gold"].get("verdicts"), r["proxy"].get("verdicts")
                if not (r["gold"]["ok"] and gv):
                    continue
                for k, g in enumerate(gv):
                    cid = f"{src}:{k}"
                    if g != 1:
                        continue
                    cand = {
                        "run": run,
                        "step": step,
                        "prompt": r["prompt"],
                        "answer": r["response"],
                        "proxy_on_original": pv[k] if pv else None,
                    }
                    old = best.get(cid)
                    if old is None or (step, run) < (old["step"], old["run"]):
                        best[cid] = cand
    return best


async def run_all(jobs, base_url: str):
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=base_url, api_key="EMPTY", timeout=600, max_retries=2)
    judge = ProxyJudge(base_url, "proxy-judge", concurrency=24, max_retries=3, timeout=300.0)
    crits = criterion_objects()
    sem = asyncio.Semaphore(24)

    async def one(job):
        async with sem:
            try:
                out = await client.chat.completions.create(
                    model="proxy-judge",
                    messages=[
                        {
                            "role": "user",
                            "content": EDIT_PROMPT.format(
                                criterion=job["text"],
                                answer=job["answer"],
                                instruction=EDITS[job["kind"]],
                            ),
                        }
                    ],
                    temperature=0.0,
                    max_tokens=1500,
                )
                edited = (out.choices[0].message.content or "").strip()
            except Exception as e:
                job.update(edited=None, valid=False, why=f"edit failed: {type(e).__name__}")
                return job
        orig = job["answer"].strip()
        valid = bool(edited) and edited != orig
        if job["kind"] == "delete":
            valid = valid and len(edited) < len(orig)
        job.update(edited=edited, valid=valid, why=None if valid else "no effective edit")
        if valid:
            res = await judge.grade(job["prompt"], edited, crits[job["source_index"]])
            job["proxy_ok"] = res.ok
            job["proxy_on_edit"] = res.verdicts[job["k"]] if res.ok else None
        return job

    return await asyncio.gather(*(one(j) for j in jobs))


def main() -> None:
    base_url = os.environ.get("PROXY_URL", "http://127.0.0.1:8001/v1")
    check_health(base_url)
    out = out_dir()
    with open(out / "tags.jsonl") as fh:
        tags = {json.loads(line)["id"]: json.loads(line) for line in fh}
    table = criteria_table()
    sources = pick_sources([r["id"] for r in table])
    jobs = []
    for r in table:
        s = sources.get(r["id"])
        if s is None:
            continue
        for kind in BY_TYPE[tags[r["id"]]["type"]]:
            jobs.append(
                {
                    "id": r["id"],
                    "source_index": str(r["source_index"]),
                    "k": r["k"],
                    "type": tags[r["id"]]["type"],
                    "text": r["text"],
                    "kind": kind,
                    **s,
                }
            )
    print(f"{len(sources)} criteria with a gold-yes source answer; {len(jobs)} edits", flush=True)
    done = asyncio.run(run_all(jobs, base_url))
    with (out / "audit_edits.jsonl").open("w") as f:
        for j in done:
            j.pop("prompt", None)
            f.write(json.dumps(j, ensure_ascii=False) + "\n")
    valid = [j for j in done if j["valid"] and j.get("proxy_ok")]
    fp = sum(j["proxy_on_edit"] == 1 for j in valid)
    print(json.dumps({"edits": len(done), "valid_graded": len(valid), "proxy_yes": fp}))


if __name__ == "__main__":
    main()
