# PLAN.md: rolloutscope research plan

Status: current and binding. Revised 2026-09-25. This replaces the v0 build plan and the
September delivery plan, both kept in [docs/history/](docs/history/) as records.
Change this file in writing before changing direction. Do not start a second plan
document.

## 1. What rolloutscope is for

rolloutscope is a research instrument for one question:

> In RL post-training, does reward hacking show up in the policy's internal
> representations before it shows up in signals readable from logged rollouts? Can an
> unsupervised white-box signal flag it without hack labels?

The shipped package (v0) is the black-box half of that comparison. It normalizes
rollout logs, computes black-box signals over them, and reports what it could and could
not check. It is maintained, not expanded. It is not a product: no hosted service, no
external pilot program, no investigation UI.

Two rules follow from the question:

1. A white-box result counts only if it beats the black-box baselines on the same runs,
   under controls written down before the run.
2. Infrastructure gets built only when a milestone below needs it. The v1 attempt built
   a sidecar contract, report panels, and CI jobs before any experiment showed a
   signal (see [the v1 post-mortem](docs/history/v1-postmortem.md)). That order is
   reversed now.

## 2. Why the experimental setting changed

The July 2026 pilots used hacks implanted by fine-tuning that leave a visible trace in
the text: a copied canary answer, edited test files. A text rule can catch those in
principle, so a white-box win there says more about which text rules existed than about
activations. Implanted hacks also say nothing about hacking that emerges during
training, which is the case a monitor exists for.

Rubric-based RL gives the opposite regime. A recent line of work from Scale AI reports:

- Policies trained against a weak rubric judge keep improving on that judge while a
  stronger gold judge first rises, then falls (arXiv:2608.11669, arXiv:2605.12474).
- The exploits are not lexical: partial credit on compound criteria, presence-based
  coverage, verbosity, and inflated factual claims (arXiv:2605.12474,
  arXiv:2606.12507).
- Lower-hacking conditions exist on the same data: Rubric Dropout (arXiv:2608.11669)
  and verifier-free Rubric-Guided Self-Distillation, RGSD (arXiv:2606.12507).

That gives a measurable onset to predict, hacks that text rules do not trivially catch,
and same-task controls. Caveats from reading those papers: the dropout report is
single-seed work in progress, RGSD's headline comparison uses a weak-judge baseline, and
all of them treat an LLM judge as ground truth.

## 3. Milestones

Each milestone ends in a gate. Do not start the next milestone until its gate passes or
this file is revised to say why not.

### M0. Housekeeping

Archive stale plans, rewrite CLAUDE.md, record the v1 post-mortem, and cut release
0.2.0 so the tool version stops claiming 0.1.0 for schema 2.0 output.

Gate: the docs agree with the code, and the offline suite, ruff, and mypy are green.

### M1. Reproduce rubric hacking, black-box only

- Model: start with Qwen2.5-1.5B-Instruct to shake out the pipeline. The published
  effects are at 3B to 8B, so expect to scale up.
- Data: a RubricHub-Medical subset for training and a held-out HealthBench-Hard subset
  for evaluation. Confirm licenses before committing any derived data.
- Training: GRPO against a weak proxy judge, with per-criterion verdicts logged.
- Evaluation: every K steps, generate on a fixed evaluation prompt set and grade each
  response with the proxy judge and a stronger judge from a different model family (the
  gold judge). Log per-criterion verdicts from both, plus response length.
- Output format: write rollouts in the legacy prime-rl step layout
  (`step_<n>/train_rollouts.jsonl`, the row shape `scripts/perf/generate.py` writes),
  with judge scores in `metrics`. rolloutscope reads that today with no new adapter.

Gate: on at least two seeds, the gold score peaks and then declines while the proxy
keeps rising. If that does not happen, change model size, step budget, or judge
strength until it does. No white-box work starts before this gate passes.

### M2. Matched conditions

Same base model, data, seeds, and schedule:

| Run | Condition | Expected |
|---|---|---|
| A | GRPO, weak proxy judge | hacks |
| B | GRPO plus Rubric Dropout at 50 percent | hacks less |
| C | RGSD, no judge in the training loop | no judge to hack |

B and C double as the "trained but not hacking" references that the drift confound
requires: any training moves activations, so drift from the base model alone proves
nothing.

Gate: on the gold judge, run A falls below B and C after the onset. Without that
contrast, M4 has nothing to explain.

### M3. Black-box baselines

These set the bar a white-box signal has to clear:

- Proxy minus gold gap and per-criterion overclaim rate. Add a `judge_divergence`
  detector to the package that reads these from logged `metrics`. It is offline and
  CPU-only, so it fits the v0 rules.
- `length_inflation` (existing) over the step series.
- A logit-level baseline, the self-internalization gap from arXiv:2605.12474. It needs
  the model, so it is computed in `experiments/` and logged as a metric.

Gate: every baseline has an onset step on run A, computed by a rule fixed before any
white-box result is looked at.

### M4. White-box signals, in experiments/ only

- Capture mean-pooled residual-stream activations on the fixed evaluation prompts at
  each checkpoint, for the policy and the base model.
- Primary, unsupervised: per-layer drift from the base model, and drift relative to run
  C at the same step.
- Secondary, supervised: a linear probe whose labels come from gold-judge overclaim,
  never from rolloutscope detectors, split by prompt.
- Write the pass criterion into PROGRESS.md before running. Default:
  1. the white-box onset precedes the earliest black-box onset on run A by more than
     the seed-to-seed spread;
  2. the same signal stays quiet on runs B and C at that step;
  3. the null checks in section 4 pass.

Gate: pass means M5. Fail means writing up the negative result, which is also a
publishable outcome at this scale.

### M5. Explain and promote (only after M4 passes)

Use SAE features to say what moved, for example coverage and format features versus
content features. Then move the parts that proved themselves into the package behind an
optional extra, with a written data contract.

## 4. Controls every white-box number must carry

- A text baseline (TF-IDF plus logistic regression) and a length-only baseline trained
  on the same labels and splits.
- Null check: policy identical to reference gives no signal.
- Shuffled labels give chance-level probe AUROC.
- Drift is compared against a trained-but-not-hacking run, never only against the base
  model.
- When labels come from answer content, activations are pooled so they exclude the
  answer tokens, or the result is reported both ways.
- Three or more seeds, with bootstrap confidence intervals, and train and test split by
  prompt.
- The label source is never the detector being evaluated.

## 5. Out of scope

Stop and revise this file before working on any of these:

- Delivery-plan Phase 3 and 4 work: an indexed store, an investigation queue, reviewer
  adjudication, run-comparison tooling, external pilot programs, usability studies, a
  resource envelope, or a release pipeline beyond what exists.
- More tuning of text detectors against TRACE. TRACE stays as a regression benchmark
  only.
- New text detectors that no milestone needs.
- A `whitebox` subpackage in `src/`, an activation sidecar contract, or white-box report
  panels before M4 passes.
- Adapters for the current verifiers or prime-rl formats, unless an experiment trains
  with prime-rl. In that case, build only what that experiment's output needs.
- Multi-GPU capture, per-token activation storage, or live training-loop hooks.
- Reusing the July test beds without the salvage check in the post-mortem.

## 6. Package maintenance

- Keep the offline suite, ruff, and mypy green, and fix bugs as found.
- Add `judge_divergence` (M3).
- Keep the known limits stated plainly in the README: legacy input formats only,
  analysis holds the run in memory, and `verifier_tamper` has a false-positive rate
  near 0.5 on the TRACE tuning split.

## 7. Open decisions

- Trainer: TRL, verl, or prime-rl. Pick whichever reaches the M1 gate fastest, and write
  the legacy step layout either way.
- Proxy and gold judges: local models or API models, and the evaluation budget per
  checkpoint.
- Starting model size for the real runs.
- Whether CLAUDE.md, PLAN.md, and PROGRESS.md stay in the public repository. They are
  tracked as of `c2d7dcc`.
