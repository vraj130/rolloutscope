"""Shared loading for J1 (PLAN.md): eval criteria and the logged proxy and gold verdicts of M2.

Everything here reads existing M2 outputs under $ROLLOUTSCOPE_DATA/m1/; nothing calls a judge.
A criterion is identified by "<source_index>:<k>", k its position in the prompt's rubric.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "m1"))

from data import Criterion, split

RUNS = [f"m2-{c}-qwen1.5b-s{s}" for c in "abc" for s in (1, 2)]
TYPES = ["structural", "presence", "numeric", "negation", "factual", "compound"]


def data_root() -> Path:
    """$ROLLOUTSCOPE_DATA."""
    return Path(os.environ["ROLLOUTSCOPE_DATA"])


def out_dir() -> Path:
    """$ROLLOUTSCOPE_DATA/j1, created if missing."""
    d = data_root() / "j1"
    d.mkdir(exist_ok=True)
    return d


def eval_examples():
    """The 100 held-out prompts of every M1 and M2 run (split_seed 0)."""
    return split(500, 100, 0)[1]


def criteria_table() -> list[dict[str, Any]]:
    """One row per eval criterion: id, source_index, k, text, weight, prompt."""
    rows = []
    for e in eval_examples():
        for k, c in enumerate(e.criteria):
            rows.append(
                {
                    "id": f"{e.source_index}:{k}",
                    "source_index": e.source_index,
                    "k": k,
                    "text": c.text,
                    "weight": c.weight,
                }
            )
    return rows


def load_run(run: str) -> dict[int, dict[int, dict[str, Any]]]:
    """{step: {source_index: row}}, each row with the proxy and gold verdicts attached."""
    ev = data_root() / "m1" / run / "eval"
    steps: dict[int, dict[int, dict[str, Any]]] = {}
    for p in ev.glob("step_*.jsonl"):
        if p.name.endswith(".gold.jsonl"):
            continue
        step = int(p.stem.split("_")[1])
        rows = {r["source_index"]: r for r in map(json.loads, p.read_text().splitlines())}
        for g in map(json.loads, (ev / f"step_{step}.gold.jsonl").read_text().splitlines()):
            rows[g["source_index"]]["gold"] = g
        steps[step] = rows
    return dict(sorted(steps.items()))


def verdict_arrays(run_data, crit_ids: list[str]):
    """(steps, proxy[S, C], gold[S, C]) as 0/1 arrays over all criteria, NaN if missing."""
    index = {cid: i for i, cid in enumerate(crit_ids)}
    steps = list(run_data)
    P = np.full((len(steps), len(crit_ids)), np.nan)
    G = np.full_like(P, np.nan)
    for si, s in enumerate(steps):
        for src, r in run_data[s].items():
            pv, gv = r["proxy"].get("verdicts"), r["gold"].get("verdicts")
            for k in range(len(r["criteria"])):
                i = index[f"{src}:{k}"]
                if r["proxy"]["ok"] and pv is not None:
                    P[si, i] = pv[k]
                if r["gold"]["ok"] and gv is not None:
                    G[si, i] = gv[k]
    return np.array(steps), P, G


def criterion_objects() -> dict[str, list[Criterion]]:
    """source_index -> its Criterion list (for building judge calls)."""
    return {str(e.source_index): e.criteria for e in eval_examples()}
