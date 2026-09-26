"""Regrade eval rows whose proxy call failed during training, with the same proxy judge.

During training the eval grading sends all 100 calls at once, and with the 30 s per-call
timeout some calls time out while queued behind the proxy's KV cache. This script grades
only those rows again: same served model, same grading prompt and schema, temperature 0,
at lower concurrency and with a longer timeout. Rows that were graded during training are
not touched.

The original file is copied to eval_orig/step_<n>.jsonl before it is rewritten. A
regraded row keeps its training-time error under proxy.first_error and gets
proxy.regraded = true. A row that fails again stays failed.

    bash m1/serve_proxy.sh <any_log_name>
    uv run python m1/regrade_eval_proxy.py --run m1-qwen1.5b-s1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
from pathlib import Path

from data import criteria_from_json
from judge import ProxyJudge, check_health


async def regrade(run: Path, judge: ProxyJudge) -> dict[str, int]:
    """Regrade the failed proxy rows of every eval file of a run, in place."""
    counts = {"files": 0, "failed_before": 0, "failed_after": 0}
    files = [p for p in (run / "eval").glob("step_*.jsonl") if not p.name.endswith(".gold.jsonl")]
    for path in sorted(files, key=lambda p: int(p.stem.split("_")[1])):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        todo = [r for r in rows if not r["proxy"]["ok"]]
        if not todo:
            continue
        counts["files"] += 1
        counts["failed_before"] += len(todo)
        backup = run / "eval_orig" / path.name
        backup.parent.mkdir(exist_ok=True)
        if not backup.exists():
            shutil.copy2(path, backup)
        results = await judge.grade_many(
            [(r["prompt"], r["response"], criteria_from_json(r["criteria"])) for r in todo]
        )
        for r, res in zip(todo, results, strict=True):
            first_error = r["proxy"].get("first_error") or r["proxy"]["error"]
            r["proxy"] = {
                "ok": res.ok,
                "verdicts": res.verdicts,
                "score": res.score,
                "error": res.error,
                "first_error": first_error,
                "regraded": True,
            }
        failed = sum(not r["proxy"]["ok"] for r in todo)
        counts["failed_after"] += failed
        tmp = path.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        tmp.replace(path)
        print(f"{path.name}: regraded {len(todo)}, still failed {failed}", flush=True)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    run = Path(os.environ["ROLLOUTSCOPE_DATA"]) / "m1" / args.run
    base_url = os.environ.get("PROXY_URL", "http://127.0.0.1:8001/v1")
    check_health(base_url)
    judge = ProxyJudge(
        base_url,
        os.environ.get("PROXY_MODEL", "proxy-judge"),
        concurrency=args.concurrency,
        timeout=args.timeout,
    )
    counts = asyncio.run(regrade(run, judge))
    print(json.dumps({"run": args.run, **counts, "errors": judge.stats.errors}))


if __name__ == "__main__":
    main()
