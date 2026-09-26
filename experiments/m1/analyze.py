"""Offline analysis of an M1 run's eval generations: proxy versus gold, no model or API calls.

Reads eval/step_<n>.jsonl (proxy verdicts) and eval/step_<n>.gold.jsonl (gold verdicts),
joined on source_index. Label source: gold = gpt-6-luna at medium reasoning effort
(grade_gold.py); proxy = Llama-3.1-8B-Instruct bf16 (serve_proxy.sh). Both use the same keyed
grading prompt (judge.py).

Bootstrap: B resamples of the eval prompts with a fixed seed. The same resampled prompt set
is used at every step, so step-to-step comparisons are paired by prompt. Intervals are 95%
percentile intervals. "Early" is the first N_WINDOW eval steps of the run and "late" the last
N_WINDOW; late minus early is the mean over late steps minus the mean over early steps.

Writes <run>/analysis/summary.json, summary.md, and top_gaps_step<last>.md.

    uv run python m1/analyze.py --run dryrun-qwen1.5b-s0
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

B = 10_000
SEED = 0
N_WINDOW = 3  # eval steps in the early and late windows
TOP_N = 10
RESPONSE_CHARS = 800


def load(run: Path) -> dict[int, list[dict[str, Any]]]:
    """Return {step: rows sorted by source_index}, each row carrying its gold result."""
    steps: dict[int, list[dict[str, Any]]] = {}
    for path in (run / "eval").glob("step_*.jsonl"):
        if path.name.endswith(".gold.jsonl"):
            continue
        step = int(path.stem.split("_")[1])
        rows = {r["source_index"]: r for r in map(json.loads, path.read_text().splitlines())}
        gold_path = path.parent / f"step_{step}.gold.jsonl"
        for g in map(json.loads, gold_path.read_text().splitlines()):
            rows[g["source_index"]]["gold"] = g
        steps[step] = [rows[k] for k in sorted(rows)]
    return dict(sorted(steps.items()))


def ci(samples: np.ndarray) -> list[float]:
    """95% percentile interval."""
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return [float(lo), float(hi)]


def slope(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OLS slope of y (..., n_steps) on x (n_steps), vectorized over leading axes."""
    xc = x - x.mean()
    return ((y - y.mean(axis=-1, keepdims=True)) * xc).sum(axis=-1) / (xc**2).sum()


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two 1-D arrays."""
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation (Pearson on average ranks)."""

    def rank(v: np.ndarray) -> np.ndarray:
        order = v.argsort(kind="mergesort")
        r = np.empty(len(v))
        r[order] = np.arange(len(v))
        for val in np.unique(v):  # average ties
            m = v == val
            r[m] = r[m].mean()
        return r

    return pearson(rank(a), rank(b))


def main() -> None:
    """Compute the five analyses and write them under <run>/analysis/."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="run_name under $ROLLOUTSCOPE_DATA/m1/")
    args = parser.parse_args()
    run = Path(os.environ["ROLLOUTSCOPE_DATA"]) / "m1" / args.run
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    data = load(run)
    steps = np.array(list(data))
    ids = [r["source_index"] for r in data[int(steps[0])]]
    for s, rows in data.items():
        if [r["source_index"] for r in rows] != ids:
            raise SystemExit(f"step {s} does not have the same eval prompts as step {steps[0]}")
        bad = [r["source_index"] for r in rows if not (r["proxy"]["ok"] and r["gold"]["ok"])]
        if bad:
            raise SystemExit(f"step {s}: {len(bad)} rows lack a proxy or gold verdict")
    n = len(ids)

    P = np.array([[r["proxy"]["score"] for r in data[s]] for s in steps])  # (steps, prompts)
    G = np.array([[r["gold"]["score"] for r in data[s]] for s in steps])
    L = np.array([[r["completion_tokens"] for r in data[s]] for s in steps], dtype=float)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(B, n))
    Pb, Gb = P[:, idx].mean(axis=2).T, G[:, idx].mean(axis=2).T  # (B, steps)
    res: dict[str, Any] = {
        "run": args.run,
        "n_prompts": n,
        "steps": steps.tolist(),
        "bootstrap": {"B": B, "seed": SEED, "interval": "95% percentile, prompts resampled"},
        "label_source": "gold: gpt-6-luna medium effort; proxy: Llama-3.1-8B-Instruct bf16",
    }

    # 1. per-step means
    res["per_step"] = [
        {
            "step": int(s),
            "proxy": float(P[i].mean()),
            "proxy_ci": ci(Pb[:, i]),
            "gold": float(G[i].mean()),
            "gold_ci": ci(Gb[:, i]),
            "gap": float((P[i] - G[i]).mean()),
            "gap_ci": ci(Pb[:, i] - Gb[:, i]),
            "mean_len": float(L[i].mean()),
        }
        for i, s in enumerate(steps)
    ]

    # 2. slopes per 100 steps, and late minus early
    x = steps.astype(float)
    if len(steps) < 2 * N_WINDOW:
        raise SystemExit(f"need at least {2 * N_WINDOW} eval steps, found {len(steps)}")
    early = np.zeros(len(steps), dtype=bool)
    late = np.zeros(len(steps), dtype=bool)
    early[:N_WINDOW] = True
    late[-N_WINDOW:] = True
    early_steps, late_steps = steps[early].tolist(), steps[late].tolist()
    res["trend"] = {}
    for name, point, boot in (
        ("proxy", P.mean(axis=1), Pb),
        ("gold", G.mean(axis=1), Gb),
        ("gap", (P - G).mean(axis=1), Pb - Gb),
    ):
        res["trend"][name] = {
            "slope_per_100_steps": float(slope(x, point) * 100),
            "slope_ci": ci(slope(x, boot) * 100),
            "late_minus_early": float(point[late].mean() - point[early].mean()),
            "late_minus_early_ci": ci(boot[:, late].mean(axis=1) - boot[:, early].mean(axis=1)),
            "early_steps": early_steps,
            "late_steps": late_steps,
        }

    # 3. per-criterion overclaim (proxy 1, gold 0) and underclaim (proxy 0, gold 1)
    res["per_criterion"] = []
    over_b = np.empty((B, len(steps)))  # bootstrap overclaim rate per step
    over_pt = np.empty(len(steps))
    for i, s in enumerate(steps):
        pv = [np.array(r["proxy"]["verdicts"]) for r in data[int(s)]]
        gv = [np.array(r["gold"]["verdicts"]) for r in data[int(s)]]
        over = np.array([((p == 1) & (g == 0)).sum() for p, g in zip(pv, gv, strict=True)])
        under = np.array([((p == 0) & (g == 1)).sum() for p, g in zip(pv, gv, strict=True)])
        tot = np.array([len(p) for p in pv])
        ob = over[idx].sum(axis=1) / tot[idx].sum(axis=1)
        ub = under[idx].sum(axis=1) / tot[idx].sum(axis=1)
        over_b[:, i] = ob
        over_pt[i] = over.sum() / tot.sum()
        res["per_criterion"].append(
            {
                "step": int(s),
                "n_criteria": int(tot.sum()),
                "overclaim": float(over.sum() / tot.sum()),
                "overclaim_ci": ci(ob),
                "underclaim": float(under.sum() / tot.sum()),
                "underclaim_ci": ci(ub),
                "proxy_yes": float(sum(p.sum() for p in pv) / tot.sum()),
                "gold_yes": float(sum(g.sum() for g in gv) / tot.sum()),
            }
        )

    res["overclaim_trend"] = {
        "late_minus_early": float(over_pt[late].mean() - over_pt[early].mean()),
        "late_minus_early_ci": ci(over_b[:, late].mean(axis=1) - over_b[:, early].mean(axis=1)),
        "early_steps": early_steps,
        "late_steps": late_steps,
    }

    # 4. length versus score, first and last step
    res["length_correlation"] = []
    for i in (0, len(steps) - 1):
        entry: dict[str, Any] = {"step": int(steps[i])}
        for name, M in (("proxy", P), ("gold", G)):
            boot = np.array([pearson(L[i, j], M[i, j]) for j in idx[:2000]])
            entry[name] = {
                "pearson": pearson(L[i], M[i]),
                "pearson_ci": ci(boot),
                "spearman": spearman(L[i], M[i]),
            }
        res["length_correlation"].append(entry)

    (out / "summary.json").write_text(json.dumps(res, indent=2))

    # 5. largest proxy-minus-gold gaps at the last step
    last = int(steps[-1])
    rows = sorted(data[last], key=lambda r: r["proxy"]["score"] - r["gold"]["score"], reverse=True)
    md = [
        f"# Step {last}: {TOP_N} largest proxy minus gold gaps",
        "",
        f"Run `{args.run}`. Responses truncated to {RESPONSE_CHARS} characters. "
        "Verdicts: P = proxy (Llama-3.1-8B bf16), G = gold (gpt-6-luna). Only criteria where "
        "they disagree are listed.",
        "",
    ]
    for rank_, r in enumerate(rows[:TOP_N], 1):
        crit = r["criteria"]
        pv, gv = r["proxy"]["verdicts"], r["gold"]["verdicts"]
        q = "\n".join(m["content"] for m in r["prompt"])
        resp = r["response"][:RESPONSE_CHARS] + (
            "..." if len(r["response"]) > RESPONSE_CHARS else ""
        )
        md += [
            f"## {rank_}. source_index {r['source_index']}: proxy {r['proxy']['score']:.3f}, "
            f"gold {r['gold']['score']:.3f}, gap {r['proxy']['score'] - r['gold']['score']:+.3f}, "
            f"{r['completion_tokens']} tokens",
            "",
            f"**Prompt:** {q}",
            "",
            "**Response:**",
            "",
            "```text",
            resp,
            "```",
            "",
            "| # | weight | P | G | criterion |",
            "|---|---|---|---|---|",
        ]
        for k, (c, p, g) in enumerate(zip(crit, pv, gv, strict=True), 1):
            if p != g:
                text = c["criterion"].replace("|", "/")
                md.append(f"| c{k} | {c['weight']:g} | {p} | {g} | {text} |")
        md.append("")
    (out / f"top_gaps_step{last}.md").write_text("\n".join(md))

    # human-readable summary
    lines = [f"# {args.run}: proxy versus gold on {n} held-out prompts", ""]
    lines += [res["label_source"] + ". " + res["bootstrap"]["interval"] + f", B = {B}.", ""]
    lines += [
        "| step | proxy [95% CI] | gold [95% CI] | gap [95% CI] | len |",
        "|---|---|---|---|---|",
    ]
    for e in res["per_step"]:
        lines.append(
            f"| {e['step']} | {e['proxy']:.3f} [{e['proxy_ci'][0]:.3f}, {e['proxy_ci'][1]:.3f}] "
            f"| {e['gold']:.3f} [{e['gold_ci'][0]:.3f}, {e['gold_ci'][1]:.3f}] "
            f"| {e['gap']:+.3f} [{e['gap_ci'][0]:+.3f}, {e['gap_ci'][1]:+.3f}] "
            f"| {e['mean_len']:.0f} |"
        )
    lines += [
        "",
        "| series | slope per 100 steps [95% CI] | late minus early [95% CI] |",
        "|---|---|---|",
    ]
    for name, t in res["trend"].items():
        lines.append(
            f"| {name} | {t['slope_per_100_steps']:+.4f} [{t['slope_ci'][0]:+.4f}, "
            f"{t['slope_ci'][1]:+.4f}] | {t['late_minus_early']:+.4f} "
            f"[{t['late_minus_early_ci'][0]:+.4f}, {t['late_minus_early_ci'][1]:+.4f}] |"
        )
    lines += [
        "",
        f"Early = steps {early_steps}, late = steps {late_steps}.",
        "",
        "| step | criteria | overclaim [95% CI] | underclaim [95% CI] | proxy yes | gold yes |",
        "|---|---|---|---|---|---|",
    ]
    for e in res["per_criterion"]:
        lines.append(
            f"| {e['step']} | {e['n_criteria']} | {e['overclaim']:.3f} "
            f"[{e['overclaim_ci'][0]:.3f}, {e['overclaim_ci'][1]:.3f}] | {e['underclaim']:.3f} "
            f"[{e['underclaim_ci'][0]:.3f}, {e['underclaim_ci'][1]:.3f}] | "
            f"{e['proxy_yes']:.3f} | {e['gold_yes']:.3f} |"
        )
    ot = res["overclaim_trend"]
    lines += [
        "",
        f"Overclaim, late minus early: {ot['late_minus_early']:+.4f} "
        f"[{ot['late_minus_early_ci'][0]:+.4f}, {ot['late_minus_early_ci'][1]:+.4f}]",
    ]
    lines += ["", "| step | judge | Pearson [95% CI] | Spearman |", "|---|---|---|---|"]
    for e in res["length_correlation"]:
        for name in ("proxy", "gold"):
            c = e[name]
            lines.append(
                f"| {e['step']} | {name} | {c['pearson']:+.3f} [{c['pearson_ci'][0]:+.3f}, "
                f"{c['pearson_ci'][1]:+.3f}] | {c['spearman']:+.3f} |"
            )
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}/summary.json, summary.md, top_gaps_step{last}.md")


if __name__ == "__main__":
    main()
