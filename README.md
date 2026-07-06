# rolloutscope

Offline rollout and reward-hacking debugger for the verifiers / prime-rl RL ecosystem.

rolloutscope ingests on-disk rollout artifacts (`results.jsonl`, `metadata.json`,
`train_rollouts.jsonl`), normalizes them into a versioned schema, runs reward-hacking
detectors over them, and emits a terminal summary, a JSON findings file, and a single
self-contained HTML report.

v0 is black-box and CPU-only: it analyzes logged text and numbers. It never loads a
model, never needs a GPU, and never calls the network at analysis time. Point it at a
finished run and it tells you where the reward was probably gamed, with the offending
span attached to every flag.

## Why

RL training against automatic rewards drifts toward whatever the reward actually
measures, which is not always the task. A judge that likes long answers, a rubric that
scores format, a coding harness whose tests can be deleted: each is a reward signal a
policy can climb without getting better. Those failures are visible in the logged
rollouts if you know what to look for. rolloutscope looks, over a run you already have on
disk, and shows its work.

## Install

rolloutscope uses [uv](https://docs.astral.sh/uv/) for everything.

```bash
uv sync --extra dev     # create the venv and install rolloutscope plus dev tools
```

Runtime dependencies are pydantic, orjson, typer, rich, and jinja2. There is no torch,
no transformers, no CUDA: v0 does not load models.

## Quickstart

The repository ships a synthetic demo run (`tests/fixtures/demo/`, shaped like a real
verifiers `evaluate()` output) that trips four detectors.

```bash
# Analyze a run: terminal summary, JSON sidecar, and a self-contained HTML report.
uv run rolloutscope analyze tests/fixtures/demo --out report.html --json findings.json

# Open report.html from the file system (no server needed); it references nothing external.
open report.html            # macOS; use xdg-open on Linux
```

`analyze` prints a run summary and a findings table, then writes the reports you asked
for. On the demo run it flags `verifier_tamper`, `degenerate_repetition`,
`answer_leakage_echo`, and `format_only_wins`, each with the exact span that triggered it.

Other commands:

```bash
uv run rolloutscope detectors list                 # the discovered detector registry
uv run rolloutscope convert tests/fixtures/demo --out normalized.jsonl   # raw to schema JSONL
uv run rolloutscope schema export --out schema.json # the normalized Rollout JSON Schema
uv run rolloutscope --help                          # everything
```

### What `analyze` accepts

A path to a run directory or a single JSONL file. rolloutscope picks the adapter itself:

- a verifiers eval run: a directory with `results.jsonl` (plus optional `metadata.json`),
  or a direct `.jsonl` of trace rows;
- a prime-rl training run: `train_rollouts.jsonl` files under per-step directories, or a
  single such file.

Bad rows are skipped and logged (run with `--verbose` to see them), never fatal; the
reader streams line by line, so inputs larger than RAM are fine.

### Exit codes for CI

By default `analyze` always exits 0 and leaves the judgment to you. To gate a pipeline,
raise the bar:

```bash
uv run rolloutscope analyze <run> --fail-on critical   # exit 1 if any critical finding fired
```

`--fail-on` takes `none` (default), `info`, `warning`, or `critical`.

## Detectors

All six run in snapshot mode; the saturation and length detectors gain trend variants
when the rollouts carry a `step_index`. Every threshold is a conservative, clearly
labeled heuristic and is configurable (see Configuration); none are taken from a paper
without a verified citation.

| id | category | core signal |
|---|---|---|
| `verifier_tamper` | verifier_tampering | test edits, assert deletion, skips, forced exit 0, monkeypatched checkers, always-pass bodies |
| `reward_saturation_group_collapse` | reward_saturation | within-group reward variance collapsing to zero (the on-disk GRPO dead-group proxy), and its rise over steps |
| `length_inflation` | rubric_judge_exploit | reward correlating with completion length while an independent correctness metric stays flat |
| `format_only_wins` | rubric_judge_exploit | a format or parser metric near max while correctness is near zero and the scalar reward still clears a floor |
| `degenerate_repetition` | degeneracy | high n-gram repetition and low distinct-token ratio on high-reward completions |
| `answer_leakage_echo` | context_exploitation | the completion echoing the ground-truth `answer` or a reward criterion with no work shown |

Each detector documents its known false-positive modes in its own docstring and ships at
least one labeled hacked fixture and one clean fixture that it must separate.

## Configuration

Pass a TOML file with `--config`. Tables map onto the detector, aggregation, and severity
settings; anything omitted keeps its default.

```toml
[severity]
critical_at = 0.8      # max fired score at or above this is critical
warning_at = 0.5       # at or above this is warning, else info

[detectors.length_inflation]
min_correlation = 0.85 # Pearson r of length vs reward required to fire

[aggregation]
histogram_bins = 20    # reward histogram resolution
```

```bash
uv run rolloutscope analyze <run> --config myconfig.toml
```

## Design

- **The schema is the product.** Every row is a pydantic v2 `Rollout` (a discriminated
  union on `kind`) carrying a per-row `schema_version`, with unknown upstream keys
  preserved. Changes within a major version are additive; breaking changes bump the major
  and ship a migration. See `docs/adr/0001-normalized-rollout-schema.md`.
- **Stable, content-derived IDs.** `rollout_id`, `group_id`, and `run_id` come from the
  content, so a v1 white-box store (activations, SAE features) can join onto v0 findings
  on `(run_id, rollout_id, step_index)` without re-ingesting anything.
- **Core never imports verifiers or prime-rl.** Only the adapters know the upstream
  on-disk shapes, and they parse them from pinned reference docs rather than importing
  those libraries. The schema, detectors, analysis, and report packages depend only on
  the normalized types.
- **Detectors are pure functions with evidence.** Each returns structured `Verdict`
  objects, and a fired verdict without its offending span is invalid by construction.
- **The report is a pure function of findings.** The HTML is one self-contained file:
  inline CSS, server-side SVG charts, `<details>` collapsibles, zero JavaScript, no CDN,
  no fetch. It opens from `file://`.

## Development

```bash
uv sync --extra dev
uv run pytest -q              # offline test suite
uv run pytest -m integration  # optional network tests (TRACE download, skips if offline)
uv run ruff check .           # lint
uv run ruff format .          # format
uv run mypy src/              # types
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for project conventions and a walkthrough of
writing your own detector as a plugin.

## License

Apache-2.0. See [LICENSE](LICENSE).
