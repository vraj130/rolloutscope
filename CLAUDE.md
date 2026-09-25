# CLAUDE.md

Instructions for AI coding agents working in this repository. Read this file, then
[PLAN.md](PLAN.md) and [PROGRESS.md](PROGRESS.md), before changing anything.

## What this project is

rolloutscope is a research instrument for studying reward hacking in RL post-training.
The research question, milestones, and scope live in PLAN.md. In short: can
white-box signals from the policy's activations catch reward hacking earlier or more
reliably than black-box signals read from logged rollouts?

The repository has two parts with different rules:

- **The package (`src/rolloutscope/`)**: the black-box half. It is offline and CPU-only,
  and it loads no model. It normalizes rollout logs (schema 2.0), runs heuristic
  detectors, reports detector coverage honestly, and writes terminal, JSON, verdict
  JSONL, and self-contained HTML output. It is in maintenance mode: bug fixes, plus
  features that a PLAN.md milestone names.
- **Experiments (`experiments/`, created when M1 starts)**: training runs, judge
  grading, activation capture, and white-box analysis. This is where torch and model
  code live. It is a separate uv project with its own `pyproject.toml` that depends on
  rolloutscope by path, so the package never gains model dependencies.

## Scope guard (read before every task)

1. Map the task to a milestone or maintenance item in PLAN.md. If it does not map, stop
   and ask. Do not reinterpret the plan to make it fit.
2. Do not create new planning documents, phase schedules, delivery plans, or
   acceptance-criteria lists. Update PLAN.md (direction) or PROGRESS.md (status and
   decisions) instead.
3. Do not build infrastructure ahead of evidence. In particular, build none of these
   until PLAN.md says so: a `whitebox` subpackage in `src/`, activation sidecar
   contracts, white-box report panels, indexed stores, review or adjudication
   workflows, run-comparison tooling, or release automation beyond the current CI.
4. Prefer the smallest change that answers the question at hand. A script in
   `experiments/` that produces a number beats a tested subsystem that produces none.
5. The files in `docs/history/` are records. They are not instructions, even where they
   say "binding".

## Research rules (experiments/)

- Write the pass or fail criterion into PROGRESS.md before running an experiment whose
  result will steer the plan. Do not move it after seeing results.
- Every white-box number ships with the controls in PLAN.md section 4: text and
  length baselines on the same labels, the null check (policy equals reference), a
  trained-but-not-hacking reference, shuffled-label chance, three or more seeds with
  bootstrap intervals, and train and test split by prompt.
- Labels never come from the detector being evaluated. Record the label source next to
  every result.
- Report negative results as plainly as positive ones. Never describe a signal as
  detecting hacking when it was only compared against the base model.
- Real data, activations, and checkpoints stay outside the repository. Point scripts at
  a data directory through the `ROLLOUTSCOPE_DATA` environment variable. Commit
  configs, scripts, and small summary files only.
- Quote a result only with the path of the file it came from.

## Package rules (src/rolloutscope/)

1. **uv only.** Install with `uv add` or `uv sync`, run with `uv run`. Never pip, conda,
   or the system Python. `uv.lock` is committed and CI checks it.
2. **The schema is the contract.** Rows carry `schema_version` (currently `2.0`).
   Changes within a major version are additive. A breaking change needs a major bump,
   a migration in `schema/migrate.py`, tests, and an update to
   [docs/phase1-data-contract.md](docs/phase1-data-contract.md), which is the
   reference for identity. It supersedes the identity rules in ADR-0001.
3. **Core never imports verifiers or prime-rl.** Only `adapters/` knows upstream on-disk
   shapes, and it parses them without importing those libraries.
   `scripts/capture_upstream_fixtures.py` is the single exception, run in a throwaway
   environment.
4. **No model or GPU dependencies in the package.** No torch, transformers, safetensors,
   or inference libraries in `src/` or in the package's dependency tree.
5. **Detectors are pure functions with evidence and honest coverage.** A fired verdict
   carries its evidence span. A unit a detector could not evaluate is reported as
   insufficient data with the missing signal named, never as clean.
6. **Never fabricate thresholds or citations.** Defaults are labeled heuristic and
   configurable. An unverified source number becomes a TODO with the citation.
7. **Streaming.** Reading and conversion stream JSONL line by line with orjson, and bad
   rows are skipped with a reason. Analysis currently holds the whole run in memory, so
   do not claim larger-than-RAM analysis.
8. **Reports are pure functions of their data.** The HTML report is one self-contained
   file (inline CSS, server-side SVG, `<details>` collapsibles, no JavaScript, no
   network) that opens from `file://`.
9. **Tests run offline.** Every detector has hacked and clean fixtures it must separate,
   and hard negatives live in `tests/fixtures/hard_negatives/`. Network tests (TRACE)
   sit behind the `integration` marker.
10. **Verify upstream field names against source.** Supported inputs are the legacy
    verifiers `results.jsonl` layout and prime-rl `train_rollouts.jsonl` step
    directories, pinned in [docs/compatibility-matrix.md](docs/compatibility-matrix.md).
    Current upstream writes a different Episode and Trace format, and current prime-rl
    puts advantages on disk ([docs/upstream-cohorts.md](docs/upstream-cohorts.md)). Do
    not assume a field exists because an older doc or a skill says so.

## Documentation style

- No em dashes or en dashes anywhere in repo text: docs, docstrings, comments, report
  copy, commit messages. Use commas, colons, or parentheses.
- CHANGELOG follows Keep a Changelog and semver, with `Unreleased` at the top.
- Public functions get docstrings. Detector docstrings list known false-positive modes.

## Commands

```bash
uv sync --extra dev                       # install into the project venv
uv run pytest -q                          # offline suite
uv run --extra benchmark pytest -m integration   # TRACE tests, needs HF_TOKEN in .env
uv run ruff check . && uv run ruff format --check .
uv run mypy src/
uv run python scripts/check_packaging.py  # wheel install outside the checkout
uv run rolloutscope analyze tests/fixtures/demo --out /tmp/report.html
```

A change is done when the suite, ruff, format check, and mypy are green, and the
CHANGELOG and any affected doc are updated in the same change.

## Repository layout

```
PLAN.md  PROGRESS.md  CLAUDE.md          current plan, status log, these instructions
README.md  CHANGELOG.md  CONTRIBUTING.md
src/rolloutscope/
  schema/      models.py, findings.py, execution.py, ids.py, io.py, migrate.py
  adapters/    base.py, normalized.py, verifiers_eval.py, prime_rl_train.py
  detectors/   base.py (registry), execution.py (coverage), _text.py, one module per detector
  analysis/    aggregates.py, findings.py
  report/      model.py, terminal.py, json_out.py, html.py, svg.py, templates/
  benchmark/   TRACE manifest, mapper, applicability, metrics, splits
  output.py    collision checks and atomic writes
  cli.py       thin typer wrapper
benchmarks/trace/manifest.json           pinned TRACE population (no row content)
scripts/                                 validation, fixture capture, packaging, perf
docs/                                    data contract, compatibility, benchmark metrics, ADR-0001
docs/history/                            archived plans, logs, reviews, v1 post-mortem
tests/
experiments/                             separate uv project, created at M1
```

## Detector catalog

| id | category | core signal | needs |
|---|---|---|---|
| verifier_tamper | verifier_tampering | test edits, assert deletion, skips, forced exit 0, monkeypatching | completion or trajectory text |
| reward_saturation_group_collapse | reward_saturation | within-group reward variance at zero, and its trend over steps | reward, groups, step_index for trend |
| length_inflation | rubric_judge_exploit | reward tracking length while a quality metric stays flat | reward, a quality metric |
| format_only_wins | rubric_judge_exploit | format metric high, correctness low, reward still high | format and correctness metrics |
| degenerate_repetition | degeneracy | n-gram repetition on a single high-reward completion | completion, reward |
| answer_leakage_echo | context_exploitation | completion echoes the reference answer with no work shown | completion, answer |

Planned (PLAN.md M3): `judge_divergence`, which reads proxy and gold judge scores from
`metrics`.

## Skills

Local skills in `.claude/` (not committed) may carry older upstream facts. When a skill
disagrees with `docs/compatibility-matrix.md` or with pinned upstream source, the
matrix and the source win.
