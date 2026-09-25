"""Gold-grade saved M1 evaluation generations with OpenAI gpt-6-luna through the Batch API.

Reads every eval/step_<n>.jsonl of a run, grades each response with the same grading prompt
as the proxy judge (judge.build_messages), and writes:

- eval/step_<n>.gold.jsonl  gold verdicts per row, keyed by source_index
- gold/summary.tsv          step, mean proxy, mean gold, gap, mean length
- gold/batch_*.json, gold/*.jsonl  batch ids, inputs, raw outputs (for resume and audit)
- $ROLLOUTSCOPE_DATA/m1/gold_spend.jsonl  actual cost of every job, for the project budget

Before any paid call it prints the request count and an estimated cost, and asks for
confirmation. The reasoning-token part of the estimate comes from --pilot N synchronous
requests (also confirmed first) or from --reasoning-tokens.

A request that fails in the batch is resubmitted once (--retry-rounds). A request that still
fails is counted, logged with its error, and left ungraded. It never gets a score.

Usage, from experiments/ (OPENAI_API_KEY in the environment or the repo .env):

    uv run python m1/grade_gold.py --run dryrun-qwen1.5b-s0 --pilot 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from data import criteria_from_json
from judge import build_messages, parse_verdicts, response_format, score

MODEL = "gpt-6-luna"
REASONING_EFFORT = "medium"
MAX_COMPLETION_TOKENS = 16000
BUDGET_USD = 100.0

# USD per 1M tokens. Source: https://developers.openai.com/api/docs/pricing, read 2026-09-25.
# The page does not say how reasoning tokens are billed; the API counts them inside
# completion_tokens, so they are priced at the output rate here.
PRICES = {
    "batch": {"input": 0.05, "cached_input": 0.005, "output": 0.25},
    "standard": {"input": 0.10, "cached_input": 0.01, "output": 0.50},
}
CHARS_PER_TOKEN = 4.0  # heuristic for the pre-submit estimate only


def load_rows(run: Path) -> list[dict[str, Any]]:
    """All eval rows of a run, with a stable custom_id per row."""
    rows = []
    files = sorted((run / "eval").glob("step_*.jsonl"), key=lambda p: int(p.stem.split("_")[1]))
    for path in files:
        if path.name.endswith(".gold.jsonl"):
            continue
        for line in path.read_text().splitlines():
            r = json.loads(line)
            r["custom_id"] = f"s{r['step']}-i{r['source_index']}"
            rows.append(r)
    return rows


def request_body(row: dict[str, Any]) -> dict[str, Any]:
    """Chat Completions body with the shared grading prompt and the verdict schema."""
    crit = criteria_from_json(row["criteria"])
    return {
        "model": MODEL,
        "messages": build_messages(row["prompt"], row["response"], crit),
        "response_format": response_format(len(crit)),
        "reasoning_effort": REASONING_EFFORT,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
    }


def usage_parts(usage: dict[str, Any]) -> dict[str, int]:
    """Split a usage object into uncached input, cached input, visible output, reasoning."""
    prompt = usage.get("prompt_tokens", 0)
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
    completion = usage.get("completion_tokens", 0)
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
    return {
        "input": prompt - cached,
        "cached_input": cached,
        "output": completion - reasoning,
        "reasoning": reasoning,
    }


def cost(parts: dict[str, int], tier: str) -> dict[str, float]:
    """USD cost per part. Reasoning is priced at the output rate."""
    p = PRICES[tier]
    c = {
        "input": parts["input"] * p["input"] / 1e6,
        "cached_input": parts["cached_input"] * p["cached_input"] / 1e6,
        "output": parts["output"] * p["output"] / 1e6,
        "reasoning": parts["reasoning"] * p["output"] / 1e6,
    }
    c["total"] = sum(c.values())
    return c


def spent_so_far(data_root: Path) -> float:
    """Sum of recorded gold spend across all jobs."""
    path = data_root / "m1" / "gold_spend.jsonl"
    if not path.exists():
        return 0.0
    return sum(json.loads(line)["usd_total"] for line in path.read_text().splitlines())


def record_spend(data_root: Path, entry: dict[str, Any]) -> None:
    """Append one job's actual cost to the project spend log."""
    path = data_root / "m1" / "gold_spend.jsonl"
    with path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def confirm(question: str) -> bool:
    """Ask on the terminal; anything but 'yes' declines."""
    return input(f"{question} Type 'yes' to continue: ").strip().lower() == "yes"


def run_pilot(client, rows: list[dict[str, Any]], n: int, data_root: Path, run: Path) -> float:
    """Send n synchronous standard-tier requests and return mean reasoning tokens."""
    sample = rows[:: max(1, len(rows) // n)][:n]
    est_in = sum(len(json.dumps(request_body(r)["messages"])) for r in sample) / CHARS_PER_TOKEN
    print(f"pilot: {len(sample)} standard-tier requests, ~{est_in:.0f} input tokens plus reasoning")
    if not confirm("Run the pilot?"):
        raise SystemExit("pilot declined")
    totals = {"input": 0, "cached_input": 0, "output": 0, "reasoning": 0}
    for r in sample:
        out = client.chat.completions.create(**request_body(r))
        for k, v in usage_parts(out.usage.model_dump()).items():
            totals[k] += v
    c = cost(totals, "standard")
    record_spend(
        data_root,
        {
            "run": run.name,
            "kind": "pilot",
            "tokens": totals,
            "usd_total": c["total"],
            "time": time.time(),
        },
    )
    mean_r = totals["reasoning"] / len(sample)
    print(f"pilot: mean reasoning tokens {mean_r:.0f}, cost ${c['total']:.4f}")
    return mean_r


def submit_and_wait(client, rows: list[dict[str, Any]], gold: Path, tag: str, poll: int) -> dict:
    """Submit one batch for rows (or resume it by tag) and return {custom_id: output line}."""
    meta_path = gold / f"batch_{tag}.json"
    if meta_path.exists():
        batch_id = json.loads(meta_path.read_text())["id"]
        print(f"resuming batch {batch_id}")
    else:
        inp = gold / f"batch_{tag}_input.jsonl"
        with inp.open("w") as f:
            for r in rows:
                line = {
                    "custom_id": r["custom_id"],
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": request_body(r),
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        file = client.files.create(file=inp.open("rb"), purpose="batch")
        batch = client.batches.create(
            input_file_id=file.id, endpoint="/v1/chat/completions", completion_window="24h"
        )
        batch_id = batch.id
        meta_path.write_text(json.dumps({"id": batch_id, "input_file_id": file.id}))
        print(f"submitted batch {batch_id} with {len(rows)} requests")
    while True:
        b = client.batches.retrieve(batch_id)
        rc = b.request_counts
        print(f"  {b.status}: {rc.completed}/{rc.total} done, {rc.failed} failed", flush=True)
        if b.status in ("completed", "failed", "expired", "cancelled"):
            break
        time.sleep(poll)
    results: dict[str, dict] = {}
    for fid, name in ((b.output_file_id, "output"), (b.error_file_id, "errors")):
        if not fid:
            continue
        text = client.files.content(fid).text
        (gold / f"batch_{tag}_{name}.jsonl").write_text(text)
        for line in text.splitlines():
            obj = json.loads(line)
            results[obj["custom_id"]] = obj
    if b.status != "completed":
        print(f"batch ended with status {b.status}; errors: {b.errors}")
    return results


def interpret(obj: dict | None, row: dict[str, Any]) -> dict[str, Any]:
    """Turn one batch output line into a gold result; failures carry the error, no score."""
    if obj is None:
        return {"ok": False, "error": "no result line in batch output", "usage": None}
    resp = obj.get("response") or {}
    body = resp.get("body") or {}
    usage = body.get("usage")
    if obj.get("error") or resp.get("status_code") != 200:
        err = obj.get("error") or body.get("error") or f"status {resp.get('status_code')}"
        return {"ok": False, "error": json.dumps(err)[:300], "usage": usage}
    crit = criteria_from_json(row["criteria"])
    try:
        verdicts = parse_verdicts(body["choices"][0]["message"]["content"] or "", len(crit))
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}", "usage": usage}
    return {"ok": True, "verdicts": verdicts, "score": score(verdicts, crit), "usage": usage}


def summarize(run: Path, rows: list[dict[str, Any]], gold_res: dict[str, dict]) -> str:
    """Per-step table: n, mean proxy, mean gold, gap on rows where both judged, mean length."""
    by_step: dict[int, list[dict]] = {}
    for r in rows:
        by_step.setdefault(r["step"], []).append(r)
    lines = ["step\tn\tmean_proxy\tmean_gold\tgap\tmean_len_tokens\tproxy_fail\tgold_fail"]
    for step in sorted(by_step):
        rs = by_step[step]
        both = [
            (r["proxy"]["score"], gold_res[r["custom_id"]]["score"])
            for r in rs
            if r["proxy"]["ok"] and gold_res[r["custom_id"]]["ok"]
        ]
        mp = sum(p for p, _ in both) / len(both) if both else float("nan")
        mg = sum(g for _, g in both) / len(both) if both else float("nan")
        ml = sum(r["completion_tokens"] for r in rs) / len(rs)
        pf = sum(not r["proxy"]["ok"] for r in rs)
        gf = sum(not gold_res[r["custom_id"]]["ok"] for r in rs)
        lines.append(
            f"{step}\t{len(both)}\t{mp:.4f}\t{mg:.4f}\t{mp - mg:+.4f}\t{ml:.1f}\t{pf}\t{gf}"
        )
    return "\n".join(lines)


def main() -> None:
    """Estimate, confirm, submit, wait, write verdicts, summary, and actual cost."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="run_name under $ROLLOUTSCOPE_DATA/m1/")
    parser.add_argument("--pilot", type=int, default=0, help="measure reasoning tokens first")
    parser.add_argument("--reasoning-tokens", type=float, default=None, help="per request")
    parser.add_argument("--retry-rounds", type=int, default=1)
    parser.add_argument("--poll", type=int, default=60, help="seconds between status checks")
    args = parser.parse_args()

    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    data_root = Path(os.environ["ROLLOUTSCOPE_DATA"])
    run = data_root / "m1" / args.run
    gold = run / "gold"
    gold.mkdir(exist_ok=True)
    rows = load_rows(run)
    if not rows:
        raise SystemExit(f"no eval/step_*.jsonl under {run}")
    client = OpenAI()

    resuming = (gold / "batch_r0.json").exists()
    if not resuming:
        if args.pilot:
            reasoning = run_pilot(client, rows, args.pilot, data_root, run)
        elif args.reasoning_tokens is not None:
            reasoning = args.reasoning_tokens
        else:
            raise SystemExit("give --pilot N or --reasoning-tokens to estimate reasoning cost")
        est_in = sum(len(json.dumps(request_body(r)["messages"])) for r in rows) / CHARS_PER_TOKEN
        est_out = sum(8 * len(r["criteria"]) + 10 for r in rows)
        est = cost(
            {
                "input": int(est_in),
                "cached_input": 0,
                "output": est_out,
                "reasoning": int(reasoning * len(rows)),
            },
            "batch",
        )
        spent = spent_so_far(data_root)
        print(f"requests: {len(rows)} from {len({r['step'] for r in rows})} eval steps")
        print(
            f"estimate (batch prices, input at ~{CHARS_PER_TOKEN:.0f} chars/token, no caching): "
            f"input ${est['input']:.3f}, output ${est['output']:.3f}, "
            f"reasoning ${est['reasoning']:.3f} ({reasoning:.0f} tok/request), "
            f"total ${est['total']:.3f}"
        )
        print(f"project gold spend so far ${spent:.3f} of ${BUDGET_USD:.0f}")
        if not confirm("Submit the batch?"):
            raise SystemExit("declined; nothing submitted")

    by_id = {r["custom_id"]: r for r in rows}
    raw = submit_and_wait(client, rows, gold, "r0", args.poll)
    gold_res = {cid: interpret(raw.get(cid), by_id[cid]) for cid in by_id}
    for k in range(1, args.retry_rounds + 1):
        failed = [by_id[c] for c, g in gold_res.items() if not g["ok"]]
        if not failed:
            break
        print(f"retry round {k}: {len(failed)} failed requests")
        raw_k = submit_and_wait(client, failed, gold, f"r{k}", args.poll)
        for r in failed:
            prev_usage = gold_res[r["custom_id"]]["usage"]
            res = interpret(raw_k.get(r["custom_id"]), r)
            res["prior_usage"] = prev_usage
            gold_res[r["custom_id"]] = res

    for path in sorted((run / "eval").glob("step_*.jsonl")):
        if path.name.endswith(".gold.jsonl"):
            continue
        step = int(path.stem.split("_")[1])
        with (path.parent / f"step_{step}.gold.jsonl").open("w") as f:
            for r in rows:
                if r["step"] == step:
                    g = gold_res[r["custom_id"]]
                    out = {
                        "step": step,
                        "source_index": r["source_index"],
                        "model": MODEL,
                        "reasoning_effort": REASONING_EFFORT,
                        **g,
                    }
                    f.write(json.dumps(out) + "\n")

    table = summarize(run, rows, gold_res)
    (gold / "summary.tsv").write_text(table + "\n")
    print(table)

    totals = {"input": 0, "cached_input": 0, "output": 0, "reasoning": 0}
    for g in gold_res.values():
        for u in (g.get("usage"), g.get("prior_usage")):
            if u:
                for k, v in usage_parts(u).items():
                    totals[k] += v
    c = cost(totals, "batch")
    n_fail = sum(not g["ok"] for g in gold_res.values())
    print(f"gold failures: {n_fail}/{len(gold_res)} ({n_fail / len(gold_res):.2%})")
    print(
        "actual cost (batch prices): "
        + ", ".join(f"{k} {totals[k]} tok ${c[k]:.4f}" for k in totals)
        + f"; total ${c['total']:.4f}"
    )
    record_spend(
        data_root,
        {
            "run": run.name,
            "kind": "batch",
            "tokens": totals,
            "usd_total": c["total"],
            "time": time.time(),
        },
    )
    (gold / "cost.json").write_text(
        json.dumps({"tokens": totals, "usd": c, "failures": n_fail}, indent=2)
    )


if __name__ == "__main__":
    main()
