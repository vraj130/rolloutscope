"""J1 steps b to e: audit false-positive rates, overclaim growth, and whether one predicts it.

Inputs (no judge calls): tags.jsonl and audit_edits.jsonl from tag.py and audit.py, and the
logged proxy and gold verdicts of the six M2 runs.

- Audit (i): step-0 answers the gold judge marked "no" on a criterion (six runs, 6 answers per
  prompt); false positive = proxy "yes". Audit (ii): synthetic failing edits (audit.py).
- Overclaim growth: per criterion and run, the OLS slope over eval steps of
  1[proxy yes and gold no], per 100 steps; averaged over the six runs.
- Prediction: Spearman over the six types; criterion-level logistic regression of
  1[growth > 0] on standardized features, AUC by 5-fold cross-validation grouped by prompt.
  Baselines: step-0 proxy-yes rate, criterion length (log characters).
- Gold-yes and proxy-yes rate per type over steps.

Intervals: 95% percentile bootstrap over prompts (B = 2000, seed 0). Writes
$ROLLOUTSCOPE_DATA/j1/summary.json and summary.md.

    uv run python j1/analyze_j1.py
"""

from __future__ import annotations

import json
import sys
from itertools import permutations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from j1data import RUNS, TYPES, criteria_table, load_run, out_dir, verdict_arrays

B, SEED = 2000, 0


def ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank)."""
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    for v in np.unique(x):
        m = x == v
        r[m] = r[m].mean()
    return r


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def spearman_exact_p(x, y) -> float:
    """Two-sided permutation p-value (exact for small n)."""
    obs = abs(spearman(x, y))
    perms = list(permutations(range(len(y))))
    hits = sum(abs(spearman(x, np.asarray(y)[list(p)])) >= obs - 1e-12 for p in perms)
    return hits / len(perms)


def auc(score: np.ndarray, y: np.ndarray) -> float:
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = ranks(np.concatenate([pos, neg]))
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg)))


def logit_fit(X: np.ndarray, y: np.ndarray, l2: float = 1e-3, iters: int = 50) -> np.ndarray:
    """Logistic regression by Newton's method with a small ridge; X includes an intercept."""
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w)))
        g = X.T @ (p - y) + l2 * np.r_[0, w[1:]]
        H = X.T @ (X * (p * (1 - p))[:, None]) + l2 * np.diag(np.r_[0, np.ones(len(w) - 1)])
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def cv_auc(F: np.ndarray, y: np.ndarray, groups: np.ndarray, k: int = 5) -> float:
    """Out-of-fold AUC, folds by prompt; features standardized on the training fold."""
    ug = np.unique(groups)
    fold = {g: i % k for i, g in enumerate(np.random.default_rng(SEED).permutation(ug))}
    f = np.array([fold[g] for g in groups])
    score = np.zeros(len(y))
    for i in range(k):
        tr, te = f != i, f == i
        mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-12
        Xtr = np.c_[np.ones(tr.sum()), (F[tr] - mu) / sd]
        Xte = np.c_[np.ones(te.sum()), (F[te] - mu) / sd]
        score[te] = Xte @ logit_fit(Xtr, y[tr])
    return auc(score, y)


def boot_mean_by_prompt(values: np.ndarray, groups: np.ndarray, rng) -> list[float]:
    """95% interval of the mean, resampling prompts."""
    ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}
    stats = []
    for _ in range(B):
        pick = rng.choice(ug, size=len(ug))
        v = np.concatenate([values[idx[g]] for g in pick])
        stats.append(np.nanmean(v) if len(v) else np.nan)
    lo, hi = np.nanpercentile(stats, [2.5, 97.5])
    return [float(lo), float(hi)]


def main() -> None:
    out = out_dir()
    rng = np.random.default_rng(SEED)
    table = criteria_table()
    ids = [r["id"] for r in table]
    n = len(ids)
    with open(out / "tags.jsonl") as fh:
        tags = {json.loads(line)["id"]: json.loads(line)["type"] for line in fh}
    ctype = np.array([tags[i] for i in ids])
    prompt = np.array([r["source_index"] for r in table])
    length = np.array([len(r["text"]) for r in table], float)

    # logged verdicts: [run, step, criterion]
    arrays = [verdict_arrays(load_run(r), ids) for r in RUNS]
    steps = arrays[0][0]
    P = np.stack([a[1] for a in arrays])
    G = np.stack([a[2] for a in arrays])
    OC = np.where(np.isnan(P) | np.isnan(G), np.nan, ((P == 1) & (G == 0)).astype(float))

    # (c) overclaim growth: OLS slope per run and criterion, per 100 steps, then run mean
    x = steps / 100.0

    def slopes(Ox, xs):
        xc = xs - xs.mean()
        return (
            np.nansum((Ox - np.nanmean(Ox, axis=1, keepdims=True)) * xc[None, :, None], 1)
            / (xc**2).sum()
        )

    growth_runs = slopes(OC, x)  # [run, criterion]
    growth = growth_runs.mean(0)
    growth_no0 = slopes(OC[:, 1:], x[1:]).mean(0)

    # (b)(i) step-0 gold-no answers
    P0, G0 = P[:, 0], G[:, 0]
    fp_i_num = np.nansum((P0 == 1) & (G0 == 0), 0).astype(float)
    fp_i_den = np.nansum(G0 == 0, 0).astype(float)
    fp_i = np.where(fp_i_den > 0, fp_i_num / np.maximum(fp_i_den, 1), np.nan)
    p0_yes = np.nanmean(P0, 0)

    # (b)(ii) synthetic edits
    with open(out / "audit_edits.jsonl") as fh:
        edits = [json.loads(line) for line in fh]
    valid = [e for e in edits if e["valid"] and e.get("proxy_ok")]
    col = {cid: i for i, cid in enumerate(ids)}
    fp_ii_sum, fp_ii_n = np.zeros(n), np.zeros(n)
    for e in valid:
        fp_ii_sum[col[e["id"]]] += e["proxy_on_edit"]
        fp_ii_n[col[e["id"]]] += 1
    fp_ii = np.where(fp_ii_n > 0, fp_ii_sum / np.maximum(fp_ii_n, 1), np.nan)

    summary: dict = {"n_criteria": n, "types": {}, "edits": {}}
    summary["edits"]["total"] = len(edits)
    summary["edits"]["valid_graded"] = len(valid)
    summary["edits"]["invalid"] = sum(not e["valid"] for e in edits)
    summary["edits"]["proxy_yes_on_original"] = float(
        np.mean([e["proxy_on_original"] for e in valid if e["proxy_on_original"] is not None])
    )
    kinds = sorted({e["kind"] for e in valid})
    edit_table = {}
    for t in TYPES:
        for kd in [*kinds, "all"]:
            vs = [e for e in valid if e["type"] == t and (kd == "all" or e["kind"] == kd)]
            if vs:
                v = np.array([e["proxy_on_edit"] for e in vs], float)
                g = np.array([e["source_index"] for e in vs])
                edit_table[f"{t}/{kd}"] = {
                    "n": len(vs),
                    "fp": float(v.mean()),
                    "ci": boot_mean_by_prompt(v, g, rng),
                }
    step0_src = [e for e in valid if e["step"] == 0]
    summary["edits"]["by_type_kind"] = edit_table
    summary["edits"]["step0_sources_only"] = {
        t: {
            "n": sum(e["type"] == t for e in step0_src),
            "fp": float(np.mean([e["proxy_on_edit"] for e in step0_src if e["type"] == t]))
            if any(e["type"] == t for e in step0_src)
            else None,
        }
        for t in TYPES
    }

    # per-type rows
    early, late = slice(0, 3), slice(len(steps) - 3, len(steps))
    for t in TYPES:
        m = ctype == t
        pairs_i = (G0[:, m] == 0) & ~np.isnan(P0[:, m])
        v_i = P0[:, m][pairs_i]
        g_i = np.broadcast_to(prompt[m], G0[:, m].shape)[pairs_i]
        gy = np.nanmean(G[:, :, m], axis=(0, 2))
        py = np.nanmean(P[:, :, m], axis=(0, 2))
        gy_c = np.nanmean(G[:, :, m], axis=0)  # [step, crit]
        d_gold = np.nanmean(gy_c[late], 0) - np.nanmean(gy_c[early], 0)
        py_c = np.nanmean(P[:, :, m], axis=0)
        d_proxy = np.nanmean(py_c[late], 0) - np.nanmean(py_c[early], 0)
        summary["types"][t] = {
            "n_criteria": int(m.sum()),
            "fp_step0_gold_no": float(v_i.mean()) if len(v_i) else None,
            "fp_step0_gold_no_ci": boot_mean_by_prompt(v_i.astype(float), g_i, rng),
            "fp_step0_pairs": len(v_i),
            "fp_edits": edit_table.get(f"{t}/all", {}).get("fp"),
            "fp_edits_ci": edit_table.get(f"{t}/all", {}).get("ci"),
            "fp_edits_n": edit_table.get(f"{t}/all", {}).get("n", 0),
            "overclaim_growth_per_100": float(np.mean(growth[m])),
            "overclaim_growth_ci": boot_mean_by_prompt(growth[m], prompt[m], rng),
            "overclaim_growth_excl_step0": float(np.mean(growth_no0[m])),
            "step0_proxy_yes": float(np.nanmean(p0_yes[m])),
            "mean_length_chars": float(length[m].mean()),
            "gold_yes_by_step": {int(s): float(v) for s, v in zip(steps, gy, strict=True)},
            "proxy_yes_by_step": {int(s): float(v) for s, v in zip(steps, py, strict=True)},
            "gold_yes_late_minus_early": float(np.nanmean(d_gold)),
            "gold_yes_late_minus_early_ci": boot_mean_by_prompt(d_gold, prompt[m], rng),
            "proxy_yes_late_minus_early": float(np.nanmean(d_proxy)),
            "proxy_yes_late_minus_early_ci": boot_mean_by_prompt(d_proxy, prompt[m], rng),
            "gold_yes_late_minus_early_by_condition": {
                c: float(
                    np.nanmean(
                        np.nanmean(
                            G[[i for i, r in enumerate(RUNS) if f"-{c}-" in r]][:, late][:, :, m],
                            axis=(0, 1),
                        )
                        - np.nanmean(
                            G[[i for i, r in enumerate(RUNS) if f"-{c}-" in r]][:, early][:, :, m],
                            axis=(0, 1),
                        )
                    )
                )
                for c in "abc"
            },
        }

    # (d) type level
    tv = summary["types"]
    gro = [tv[t]["overclaim_growth_per_100"] for t in TYPES]
    type_level = {}
    for name, key in [
        ("audit_step0_gold_no", "fp_step0_gold_no"),
        ("audit_edits", "fp_edits"),
        ("baseline_step0_proxy_yes", "step0_proxy_yes"),
        ("baseline_length", "mean_length_chars"),
    ]:
        xs = [tv[t][key] for t in TYPES]
        type_level[name] = {"spearman": spearman(xs, gro), "exact_p": spearman_exact_p(xs, gro)}
    summary["type_level_spearman_vs_growth"] = type_level

    # (d) criterion level
    y = (growth > 0).astype(float)
    feats = {
        "audit_edits": fp_ii,
        "audit_step0_gold_no": fp_i,
        "step0_proxy_yes": p0_yes,
        "log_length": np.log(length),
    }
    models = {
        "baselines (step0_proxy_yes + log_length)": ["step0_proxy_yes", "log_length"],
        "audit_edits": ["audit_edits"],
        "audit_step0_gold_no": ["audit_step0_gold_no"],
        "baselines + audit_edits": ["step0_proxy_yes", "log_length", "audit_edits"],
        "baselines + both audits": [
            "step0_proxy_yes",
            "log_length",
            "audit_edits",
            "audit_step0_gold_no",
        ],
    }
    common = ~np.isnan(fp_ii) & ~np.isnan(fp_i) & ~np.isnan(p0_yes)
    crit = {
        "n_criteria": int(common.sum()),
        "outcome": "1[mean overclaim slope > 0]",
        "positive_rate": float(y[common].mean()),
        "models": {},
        "spearman_vs_growth": {},
    }
    for name, cols in models.items():
        F = np.c_[[feats[c][common] for c in cols]].T
        mu, sd = F.mean(0), F.std(0) + 1e-12
        w = logit_fit(np.c_[np.ones(len(F)), (F - mu) / sd], y[common])
        crit["models"][name] = {
            "cv_auc": cv_auc(F, y[common], prompt[common]),
            "coef_standardized": {c: float(v) for c, v in zip(cols, w[1:], strict=True)},
        }
    for c, v in feats.items():
        m = ~np.isnan(v)
        crit["spearman_vs_growth"][c] = {"spearman": spearman(v[m], growth[m]), "n": int(m.sum())}
    crit["corr_audit_step0_vs_step0_proxy_yes"] = spearman(fp_i[common], p0_yes[common])
    # Sensitivity: step 0 enters both the step-0 features and the slope (regression to the
    # mean), so repeat with growth measured over eval steps 25 to 600 only.
    y2 = (growth_no0 > 0).astype(float)
    sens = {"outcome": "1[mean overclaim slope over steps 25 to 600 > 0]", "models": {}}
    for name, cols in models.items():
        F = np.c_[[feats[c][common] for c in cols]].T
        sens["models"][name] = {"cv_auc": cv_auc(F, y2[common], prompt[common])}
    sens["spearman_vs_growth"] = {
        c: spearman(v[~np.isnan(v)], growth_no0[~np.isnan(v)]) for c, v in feats.items()
    }
    crit["sensitivity_excl_step0"] = sens
    summary["criterion_level"] = crit
    summary["overall_growth_per_100"] = float(growth.mean())
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    write_md(summary, steps, out)
    print((out / "summary.md").read_text())


def f(v, d=3):
    return (
        "n/a"
        if v is None or (isinstance(v, float) and np.isnan(v))
        else f"{v:+.{d}f}"
        if d == 4
        else f"{v:.{d}f}"
    )


def ci(c, d=3):
    return "" if not c else f" [{c[0]:.{d}f}, {c[1]:.{d}f}]"


def write_md(s, steps, out: Path) -> None:
    L = ["# J1: proxy-judge audit versus overclaim growth (six M2 runs)", ""]
    e = s["edits"]
    L += [
        f"Edits: {e['total']} generated, {e['valid_graded']} valid and graded, {e['invalid']} "
        f"invalid. Proxy said yes on {e['proxy_yes_on_original']:.3f} of the original "
        "(gold-yes) answers before editing.",
        "",
        "| type | criteria | audit FP, step-0 gold-no [CI] (pairs) | audit FP, edits [CI] (n) "
        "| overclaim growth per 100 steps [CI] | step-0 proxy yes |",
        "|---|---|---|---|---|---|",
    ]
    for t, v in s["types"].items():
        L.append(
            f"| {t} | {v['n_criteria']} | {f(v['fp_step0_gold_no'])}{ci(v['fp_step0_gold_no_ci'])} "
            f"({v['fp_step0_pairs']}) | {f(v['fp_edits'])}{ci(v['fp_edits_ci'])} "
            f"({v['fp_edits_n']}) "
            f"| {v['overclaim_growth_per_100']:+.4f}{ci(v['overclaim_growth_ci'], 4)} "
            f"| {v['step0_proxy_yes']:.3f} |"
        )
    L += [
        "",
        "Edit false-positive rate by type and edit kind:",
        "",
        "| type/kind | n | FP [CI] |",
        "|---|---|---|",
    ]
    for k, v in e["by_type_kind"].items():
        if not k.endswith("/all"):
            L.append(f"| {k} | {v['n']} | {v['fp']:.3f}{ci(v['ci'])} |")
    L += ["", "Type-level Spearman with overclaim growth (n = 6 types, exact permutation p):", ""]
    for k, v in s["type_level_spearman_vs_growth"].items():
        L.append(f"- {k}: {v['spearman']:+.3f} (p = {v['exact_p']:.3f})")
    c = s["criterion_level"]
    L += [
        "",
        f"Criterion level: {c['n_criteria']} criteria with both audits, outcome "
        f"{c['outcome']}, positive rate {c['positive_rate']:.3f}.",
        "",
        "| model | CV AUC | standardized coefficients |",
        "|---|---|---|",
    ]
    for k, v in c["models"].items():
        coefs = ", ".join(f"{a} {b:+.3f}" for a, b in v["coef_standardized"].items())
        L.append(f"| {k} | {v['cv_auc']:.3f} | {coefs} |")
    L += ["", "Criterion-level Spearman with growth:", ""]
    for k, v in c["spearman_vs_growth"].items():
        L.append(f"- {k}: {v['spearman']:+.3f} (n = {v['n']})")
    L.append(
        f"- audit (step-0 gold-no) versus step-0 proxy-yes rate: "
        f"{c['corr_audit_step0_vs_step0_proxy_yes']:+.4f}"
    )
    sv = c["sensitivity_excl_step0"]
    L += ["", f"Sensitivity, outcome {sv['outcome']}:", ""]
    for k, v in sv["models"].items():
        L.append(f"- {k}: CV AUC {v['cv_auc']:.3f}")
    for k, v in sv["spearman_vs_growth"].items():
        L.append(f"- Spearman {k} vs growth (steps 25 to 600): {v:+.3f}")
    show = [s_ for s_ in steps if s_ % 100 == 0]
    L += ["", "Gold-yes rate by type over eval steps (six runs pooled):", ""]
    L += ["| type | " + " | ".join(str(x) for x in show) + " | late minus early [CI] | A / B / C |"]
    L += ["|---|" + "---|" * (len(show) + 2)]
    for t, v in s["types"].items():
        g = v["gold_yes_by_step"]
        bc = v["gold_yes_late_minus_early_by_condition"]
        L.append(
            f"| {t} | "
            + " | ".join(f"{g[x]:.3f}" for x in show)
            + f" | {v['gold_yes_late_minus_early']:+.4f}{ci(v['gold_yes_late_minus_early_ci'], 4)} "
            f"| {bc['a']:+.4f} / {bc['b']:+.4f} / {bc['c']:+.4f} |"
        )
    L += ["", "Proxy-yes rate by type over eval steps (six runs pooled):", ""]
    L += ["| type | " + " | ".join(str(x) for x in show) + " | late minus early [CI] |"]
    L += ["|---|" + "---|" * (len(show) + 1)]
    for t, v in s["types"].items():
        p = v["proxy_yes_by_step"]
        L.append(
            f"| {t} | "
            + " | ".join(f"{p[x]:.3f}" for x in show)
            + f" | {v['proxy_yes_late_minus_early']:+.4f}"
            + f"{ci(v['proxy_yes_late_minus_early_ci'], 4)} |"
        )
    (out / "summary.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
