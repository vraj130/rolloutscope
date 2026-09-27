"""Rubric-conditioning lift of the base policy, the diagnostic RGSD (arXiv 2606.12507) runs first.

Generates answers from the base model to the 100 held-out eval prompts twice: with the plain
prompt, and with the RGSD teacher prompt (question plus rubric, data.rubric_prompt). Both are
graded by the proxy judge against the plain prompt and the full rubric. The lift is the mean
per-prompt difference, with a 95% prompt-bootstrap interval. No gold calls.

Needs GPU 0 for vLLM and the proxy judge up on GPU 1 (serve_proxy.sh). Outputs under
$ROLLOUTSCOPE_DATA/m1/rubric_lift/: responses.jsonl, result.json.

    CUDA_VISIBLE_DEVICES=0 uv run python m1/rubric_lift.py --config m1/config_a_s1.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from data import rubric_prompt, split
from judge import ProxyJudge, check_health

SAMPLES = 4  # answers per prompt and condition
B, SEED = 10_000, 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="model, split and sampling settings")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(os.environ["ROLLOUTSCOPE_DATA"]) / "m1" / "rubric_lift"
    if out.exists() and any(out.glob("*.json*")):
        raise SystemExit(f"{out} already has results")
    out.mkdir(parents=True, exist_ok=True)
    base_url = os.environ.get("PROXY_URL", "http://127.0.0.1:8001/v1")
    check_health(base_url)

    from vllm import LLM, SamplingParams

    _, eval_set = split(cfg["n_train"], cfg["n_eval"], cfg.get("split_seed", cfg["seed"]))
    llm = LLM(
        cfg["model"], dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=4096, seed=0
    )
    params = SamplingParams(
        n=SAMPLES, temperature=cfg["temperature"], max_tokens=cfg["max_completion_length"], seed=0
    )
    conditions = {
        "plain": [e.prompt for e in eval_set],
        "rubric": [rubric_prompt(e.prompt, e.criteria) for e in eval_set],
    }
    texts: dict[str, list[list[str]]] = {}
    for name, prompts in conditions.items():
        outs = llm.chat(prompts, params)
        texts[name] = [[c.text for c in o.outputs] for o in outs]
        lengths = [len(c.token_ids) for o in outs for c in o.outputs]
        print(f"{name}: generated, mean length {np.mean(lengths):.1f} tokens", flush=True)

    judge = ProxyJudge(base_url, os.environ.get("PROXY_MODEL", "proxy-judge"), 24, 3, 300.0)
    items, keys = [], []
    for name in conditions:
        for i, e in enumerate(eval_set):
            for k, t in enumerate(texts[name][i]):
                items.append((e.prompt, t, e.criteria))  # graded on the plain prompt
                keys.append((name, i, k))
    results = asyncio.run(judge.grade_many(items))

    scores = {n: np.full((len(eval_set), SAMPLES), np.nan) for n in conditions}
    with (out / "responses.jsonl").open("w") as f:
        for (name, i, k), (_, t, _), r in zip(keys, items, results, strict=True):
            if r.ok:
                scores[name][i, k] = r.score
            f.write(
                json.dumps(
                    {
                        "condition": name,
                        "source_index": eval_set[i].source_index,
                        "sample": k,
                        "response": t,
                        "proxy": {"ok": r.ok, "score": r.score, "verdicts": r.verdicts},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    failed = int(sum(not r.ok for r in results))
    per_prompt = {n: np.nanmean(s, axis=1) for n, s in scores.items()}
    diff = per_prompt["rubric"] - per_prompt["plain"]
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(diff), size=(B, len(diff)))
    boot = np.nanmean(diff[idx], axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    result = {
        "model": cfg["model"],
        "n_prompts": len(eval_set),
        "samples_per_prompt": SAMPLES,
        "judge": "proxy (serve_proxy.sh), full rubric, plain prompt",
        "plain_mean": float(np.nanmean(per_prompt["plain"])),
        "rubric_mean": float(np.nanmean(per_prompt["rubric"])),
        "lift": float(np.nanmean(diff)),
        "lift_ci95": [float(lo), float(hi)],
        "prompts_with_positive_lift": int((diff > 0).sum()),
        "proxy_calls": len(results),
        "proxy_failures": failed,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
