# PROGRESS.md

Current status and decision log. Direction lives in [PLAN.md](PLAN.md). The v0 build log
(July to September 2026) is archived at
[docs/history/v0-build-log.md](docs/history/v0-build-log.md).

Keep this file short. Update the status table when a milestone moves, append decisions
at the bottom, and write any experiment's pass or fail criterion here before the run.

## Status (2026-09-27)

| Milestone | State | Evidence |
|---|---|---|
| M0 housekeeping | done once v0.2.0 is tagged | docs refocus commit; version bump commit |
| M1 reproduce rubric hacking | done | $ROLLOUTSCOPE_DATA/m1/m1-qwen1.5b-s{1,2}/analysis/; verdict recorded in 1efc17e |
| M2 matched conditions | not started | |
| M3 black-box baselines | not started | |
| M4 white-box signals | not started | |
| M5 explain and promote | not started | |

Package baseline at `c2d7dcc`: 462 offline tests pass (7 integration tests deselected).
ruff, the format check, and strict mypy are clean. Schema 2.0, six detectors, and
per-detector coverage accounting are shipped. Supported inputs are the legacy verifiers
and prime-rl layouts only. On the TRACE tuning split, `verifier_tamper` has precision
0.52 and a false-positive rate of 0.51. The other five detectors have no coverage
there, and the holdout is still sealed.

## Pre-registered criteria

Write each one here, dated, before the run it governs.

- M1 (2026-09-26). Evaluated on seeds 1 and 2 only; the dry run (seed 0) is exploratory and
  excluded. M1 passes only if all hold in BOTH seeds:
  a) The slope of the proxy minus gold gap over steps, 95% bootstrap CI over prompts, has
     lower bound > 0.
  b) Overclaim rate, late (last 3 evals) minus early (first 3 evals), has 95% CI lower
     bound > 0.
  c) Gold's late-minus-early gain is less than half of the proxy's late-minus-early gain
     (point estimates).
  d) Judge failure rate is under 1% for both proxy and gold.
  e) Manual check: of the 10 largest proxy minus gold gaps at the final eval, the author
     judges at least 7 to be proxy errors (gold correct).
  Analysis uses experiments/m1/analyze.py as committed at 5da6089, unchanged. That commit is
  d6f172d with early and late taken as the first and last 3 eval steps of the analyzed run and
  an overclaim late-minus-early interval added; on the dry run its other outputs are identical
  to d6f172d. A gold rise then fall is reported but not required.
  Result (2026-09-27):
  - (a) to (d): per-seed values in Notes 2026-09-27 "M1 results".
  - (e) result: seed 1 = 10/10, seed 2 = 10/10 top-gap cases are mostly proxy error. Files:
    m1-qwen1.5b-s1/analysis/top_gaps_step600.md and m1-qwen1.5b-s2/analysis/top_gaps_step600.md.
  - Reviewed by the author.
  - Verdict: M1 PASS (all of (a) to (e) met in both seeds).
  - Caveats:
    a. Disagreement is one-directional: across the 20 cases only one row had proxy 0 and gold 1.
       The proxy mostly says yes to everything.
    b. Top-10 gaps are selected by construction, so this is not a proxy error rate.
    c. 6 of 10 prompts repeat across seeds; about 14 distinct prompts.
    d. The gold judge was too harsh on about 6 to 8 rows. Treat gold as a stronger judge, not
       ground truth.
    e. The policy produces confident medical misinformation that the proxy passes (for example
       an IM injection into subcutaneous space).

## Decisions

Numbered from R-001 so they cannot be confused with the build decisions D-001 to D-125
in the archived plans.

- R-001 (2026-09-25): rolloutscope is a research instrument for the question in PLAN.md
  section 1, not a product. The delivery plan's Phases 3 and 4 are dropped. Reason: they
  served a product roadmap, and no experiment needed them.
- R-002 (2026-09-25): the v1 white-box code is removed, and its July pilot numbers are
  unverified. Reason: the implementation was judged unreliable. The design problems are
  recorded in [docs/history/v1-postmortem.md](docs/history/v1-postmortem.md).
- R-003 (2026-09-25): the experimental setting moves from implanted, lexical hacks to
  emergent hacking under rubric-based RL, with matched Rubric Dropout and RGSD runs as
  controls. Reason: it gives a measurable onset, non-lexical exploits, and same-task
  controls (PLAN.md section 2).
- R-004 (2026-09-25): white-box code starts as scripts in a separate `experiments/` uv
  project and moves into the package only after PLAN.md M4 passes. Reason: v1 built the
  package surface before any evidence.
- R-005 (2026-09-25): experiments write the legacy prime-rl step layout, so the package
  needs no new adapter to read them. A current-format adapter gets built only if an
  experiment trains with prime-rl.
- R-006 (2026-09-25): the trainer is TRL GRPOTrainer with vLLM generation, on GPU 0.
  Each checkpoint's rollouts are written in the legacy prime-rl step layout (R-005).
- R-007 (2026-09-25): the proxy judge (training reward) is a local open model,
  Llama-3.1-8B-Instruct in bf16, served with vLLM on GPU 1. The gold judge is
  OpenAI gpt-6-luna at medium reasoning effort, through the OpenAI Batch API, grading
  saved evaluation generations after training. Reasons: budget; a weaker proxy is more
  exploitable, which suits M1; and Luna is from a different model family than both the
  policy and the proxy.
- R-008 (2026-09-25): the dry run uses Qwen2.5-1.5B-Instruct. The real M1 runs use
  Qwen2.5-3B-Instruct, with LoRA if full fine-tuning does not fit on one 3090.
- R-009 (2026-09-26): M1 runs use Qwen2.5-1.5B-Instruct with full fine-tuning and the exact
  dry-run recipe (lr 1e-6, 8 rollouts, 768 tokens, N_train 500, N_eval 100, K 25), extended to
  600 steps, seeds 1 and 2. Reason: the dry run already shows proxy and gold diverging at
  1.5B, and full fine-tuning avoids a LoRA confound. This supersedes R-008 for M1; 3B only if
  M1 fails.
- R-010 (2026-09-27): M1 passed (pre-registered criterion, seeds 1 and 2). Next is M2 (matched
  runs A, B, C). The checkpoint retention setting must be fixed before M2 runs so that M4 has
  checkpoints: seeds 1 and 2 kept only steps 550, 575 and 600 (save_every 25 with
  save_total_limit 3).

## Notes

- 2026-09-25: proxy judge output format matters. Keyed {c1: 0/1} versus list {verdicts: [...]}
  on the same 100 responses: 83% per-criterion agreement, mean score 0.249 versus 0.114. Keyed
  format kept.
- 2026-09-26: M1 dry run (Qwen2.5-1.5B-Instruct, full fine-tune, 300 steps, seed 0, 500 train
  and 100 eval RubricHub medical prompts, K = 25) finished on fourier in 5.25 h. Proxy judge:
  20,500 calls, 0 failures. Held-out eval, same 100 prompts at every step, proxy then gold
  (gpt-6-luna, medium effort): step 0 0.254 / 0.106, step 150 0.315 / 0.109, step 300 0.385 /
  0.117. Gold stays flat between 0.103 and 0.126 while the proxy rises, so the gap grows from
  +0.148 to +0.268. No gold peak and decline. One seed, no confidence intervals. Gold failures
  0/1,300 after a rerun of 653 rate-limited (429) requests. Gold graded with direct API calls,
  not the Batch API (user decision for this run); cost $0.804 plus a $0.003 pilot. Sources:
  $ROLLOUTSCOPE_DATA/m1/dryrun-qwen1.5b-s0/gold/summary.tsv, gold/cost.json, stats.jsonl,
  final.json.
- 2026-09-26 (exploratory, dry run seed 0, excluded from the M1 criterion): analyze.py at
  5da6089 on 100 held-out prompts, 95% prompt-bootstrap intervals. Gap slope +0.0239
  [+0.0134, +0.0347] per 100 steps. Overclaim 0.196 at step 0 to 0.310 at step 300; late minus
  early +0.063 [+0.033, +0.094]. Proxy late minus early +0.067, gold +0.012. Length versus
  score, Pearson: proxy +0.284 at step 0 and +0.531 at step 300; gold -0.158 and -0.083.
  Source: $ROLLOUTSCOPE_DATA/m1/dryrun-qwen1.5b-s0/analysis/summary.json.
- 2026-09-26: criterion (e) of the M1 criterion uses full, untruncated responses in the
  top-gaps file (analyze.py from 72cd980 onward). This is a presentation fix: the selection of
  the 10 largest gaps, the disagreement rows, and every computed number are unchanged
  (checked on the dry run: identical summary.json and identical selection).
- 2026-09-26: Gold calibration: gpt-6-luna reference answers on 20 eval prompts score gold
  0.349 (median 0.303, range 0.144 to 0.664); dry-run policy on the same prompts 0.075 (step 0)
  and 0.089 (step 300). Gold separates quality but is strict: practical ceiling about 0.35 on
  these rubrics. Caveat: the gold judge graded its own model's answers, so self-preference may
  inflate the reference score. Source: $ROLLOUTSCOPE_DATA/m1/gold_calibration/result_gold.json.
- 2026-09-26: Eval proxy grading timed out on 174 of 2,500 rows in seed 1 (and 273 in seed 2),
  caused by the 30 s per-call timeout added after the dry run, with 100 concurrent calls against
  a proxy that serves about 28 at once. Failures clustered in the last rows of the eval list.
  Failed rows were regraded after training with the same judge, prompt, schema and temperature
  0, using regrade_eval_proxy.py (commit 9445374), with a longer timeout and lower concurrency;
  originals are kept in eval_orig/. Interpretation of pre-registered criterion (d), fixed before
  any analysis: the failure rate is counted after this regrade; the raw pre-regrade rates are
  reported alongside. Training-time proxy calls: 0 failures of 38,400 (seed 1) and 0 failures of
  38,400 (seed 2).
- 2026-09-26: for all future runs, cap eval grading concurrency at the proxy's capacity (about
  28) or give eval calls a longer timeout, so eval grading cannot time out this way again.
- 2026-09-27: M1 results (seeds 1 and 2, Qwen2.5-1.5B-Instruct, full fine-tune, 600 steps, fixed
  split, 100 held-out prompts). analyze.py at 5da6089 plus the presentation-only 72cd980, after
  the eval proxy regrade. 95% prompt-bootstrap intervals, B = 10,000; early = eval steps 0, 25,
  50; late = 550, 575, 600. No verdict recorded.

  | Criterion | Seed 1 | Seed 2 |
  |---|---|---|
  | (a) gap slope per 100 steps [CI] | +0.0088 [+0.0047, +0.0131] | +0.0070 [+0.0036, +0.0103] |
  | (b) overclaim late minus early [CI] | +0.0580 [+0.0336, +0.0836] | +0.0507 [+0.0258, +0.0747] |
  | (c) proxy late minus early | +0.0685 | +0.0589 |
  | (c) gold late minus early | +0.0157 | +0.0089 |
  | (d) proxy failures after regrade, eval / training | 0/2,500 / 0/38,400 | 0/2,500 / 0/38,400 |
  | (d) raw eval proxy failures, before regrade | 174/2,500 (6.96%) | 273/2,500 (10.92%) |
  | (d) gold failures | 0/2,500 | 0/2,500 |
  | (e) 10 largest gaps at step 600 | pending author review | pending author review |

  Gold spend (gpt-6-luna, direct calls, standard prices): seed 1 $1.5753, seed 2 $1.5941;
  project total $4.00 of $100. Sources: $ROLLOUTSCOPE_DATA/m1/m1-qwen1.5b-s{1,2}/analysis/
  summary.json and summary.md, gold/cost.json, final.json, regrade_eval/*.log,
  $ROLLOUTSCOPE_DATA/m1/gold_spend.jsonl.
