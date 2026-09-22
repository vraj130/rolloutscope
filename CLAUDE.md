# CLAUDE.md

## Project

**rolloutscope** (working name, see naming note in the build prompt): an offline rollout
and reward debugger for the verifiers / prime-rl RL ecosystem. It ingests on-disk rollout
artifacts (`results.jsonl`, `metadata.json`, `train_rollouts.jsonl`), normalizes them into
a versioned schema, runs reward-hacking detectors over them, and emits a terminal summary,
a JSON findings file, and a single-file self-contained HTML report.

v0 is black-box and CPU-only: it analyzes logged text and numbers. It never loads a model,
never needs a GPU, and never calls a network at analysis time. White-box signals
(activations, representation drift, SAE features) are v1 and out of scope here, but the
schema and IDs are designed so v1 can join onto v0 data without a restructure.

## Golden rules (non-negotiable)

1. **uv only.** All installs via `uv add` / `uv sync`, all execution via `uv run`. Never
   `pip install`, never touch the system or base Python, never conda. `uv.lock` is
   committed.
2. **The schema is the product.** Every row carries `schema_version`. Changes within a
   major version are additive only. Every rollout, step, and group has a stable,
   content-derived ID. After the Phase 2 freeze, only the orchestrator session may amend
   the schema, and only with a version bump plus a migration function.
3. **Core never imports verifiers or prime-rl.** Adapters parse on-disk artifacts into the
   normalized schema. The `schema`, `detectors`, `analysis`, and `report` packages depend
   only on the normalized types.
4. **Detectors are pure functions with evidence.** Signature over normalized rollouts (or
   groups), returning structured `Verdict` objects (`fired`, `score`, `category`,
   `evidence`). The evidence span is mandatory: a flag without the offending span is a
   bug. Detectors are discovered via the entry-point registry.
5. **Never fabricate thresholds or paper metrics.** Defaults are conservative, clearly
   labeled as heuristic, and configurable. If a source number is unverified, leave a TODO
   with the citation rather than inventing a value (see the reward-hacking-detectors
   skill).
6. **Stream, never bulk-load.** JSONL is read line by line with orjson. Invalid rows are
   skipped and logged with a reason, never crash the file. Assume files larger than RAM.
7. **The report is a pure function of findings and aggregates.** No detector logic in
   templates, no template logic in detectors. The HTML report is one self-contained file:
   inline CSS, server-side SVG charts, `<details>` elements for collapsibles, zero JS if
   possible, no CDN, no fetch, opens from `file://`.
8. **Tests first, fixtures always.** Every detector ships with at least one hacked and
   one clean fixture and must separate them. Unit tests run fully offline. Anything that
   downloads (TRACE dataset, real environments) is an optional integration test behind a
   marker.
9. **Do not invent verifiers field names.** If a field is not in the
   verifiers-ground-truth skill's `core-types.md`, verify against source before using it.
   `advantages` and `is_trainable` are in-memory training signals and are NOT in on-disk
   jsonl; do not parse for them.
10. **No GPU or model dependencies in v0.** Do not add torch, transformers, cupy, or any
    inference library to the dependency tree.

## Documentation style

- **No em dashes anywhere in repo text** (README, docstrings, ADRs, CHANGELOG, comments,
  report copy). Use commas, colons, or parentheses instead. This is a hard project
  convention.
- CHANGELOG follows Keep a Changelog; versioning follows semver. `Unreleased` section at
  the top, entries under Added / Changed / Fixed / Removed.
- Every public function gets a docstring stating what it does, its inputs, and (for
  detectors) its known false-positive modes.

## Commands

```bash
uv sync --extra dev          # install everything into the project venv
uv run pytest -q             # full offline test suite
uv run pytest -m integration # optional network tests (TRACE download etc.)
uv run ruff check .          # lint
uv run ruff format .         # format
uv run mypy src/             # types
uv run rolloutscope --help   # CLI
```

## Repository layout

```
rolloutscope/
  pyproject.toml            # from python-oss-library-scaffold template
  uv.lock
  CLAUDE.md                 # this file
  README.md  CHANGELOG.md  CONTRIBUTING.md  LICENSE  PROGRESS.md  PLAN.md
  .github/workflows/ci.yml  # lint + test matrix 3.11 / 3.12 / 3.13, via uv
  docs/adr/0001-normalized-rollout-schema.md
  src/rolloutscope/
    schema/     # models.py (Rollout union, Verdict, Finding), ids.py, io.py, migrate.py
    adapters/   # base.py, verifiers_eval.py, prime_rl_train.py
    detectors/  # base.py (registry), one module per detector
    analysis/   # aggregates.py (per-snapshot and per-step stats, grouping)
    report/     # model.py, terminal.py, json_out.py, html.py, svg.py, templates/
    cli.py      # typer app, thin wrapper only
  tests/
    fixtures/   # synthetic + real verifiers-shaped rows, labeled hacked/clean pairs
    ...
```

## Data contract summary

- `Rollout`: discriminated union on `kind` (`single_turn` | `multi_turn`), pydantic v2,
  `extra="allow"`, per-row `schema_version`. Derived from the candidate in the
  rollout-schema-design skill, kept compatible with verifiers `RolloutOutput` (upstream
  wins on any conflict).
- Stable IDs: `run_id` (from manifest), `rollout_id` (content hash of canonical fields),
  `group_id` (grouping key is `example_id`), optional `step_index` attached by the
  adapter from file layout, never guessed.
- `Verdict`: per-rollout or per-group detector output (`fired`, `score`, `category`,
  `evidence`, `detector`, `rollout_ids`).
- `Finding`: report-level aggregation of verdicts (severity, title, description, metric
  values, config used, exemplar evidence).
- RL training signals (`advantages`, `is_trainable`) live in an optional sidecar model
  only, never on the base row.

## Detector catalog (v0)

| id | family | core signal | reads |
|---|---|---|---|
| verifier_tamper | test/verifier tampering | test-edit, assert-deletion, skip, exit-0, monkeypatch, always-pass patterns | completion, trajectory, info |
| reward_saturation_group_collapse | saturation / zero-advantage proxy | within-group reward variance at zero; fraction of dead groups, trend over steps | reward, group_id, step_index |
| length_inflation | rubric/judge exploit | reward vs length correlation while task metric is flat | completion, reward, metrics |
| format_only_wins | rubric/judge exploit | format/parser metric near max while correctness metric near zero, scalar reward still high | metrics, reward |
| degenerate_repetition | degeneracy | n-gram repetition and distinct-token ratios on high-reward rollouts | completion, reward |
| answer_leakage_echo | context/spec exploitation | completion echoes `answer` or context criterion with no work | completion, answer, prompt, info |

All detectors run in snapshot mode; saturation and length gain trend variants when
`step_index` is present.

## Skills map (read before coding the matching area)

- **verifiers-ground-truth**: before writing or reviewing any adapter code, and before
  using any verifiers field name. `references/core-types.md` and
  `references/on-disk-format.md` are the contract.
- **rollout-schema-design**: before touching `schema/`. Start from
  `references/candidate-schema.md`; do not invent a schema from scratch.
- **reward-hacking-detectors**: before writing any detector. Follow the Verdict contract,
  the five families, and the fixture-validation requirements in
  `references/taxonomy-and-sources.md`.
- **python-oss-library-scaffold**: for pyproject, CI, layout, plugin registry. Copy the
  templates in `assets/` and fill placeholders; do not write these files from scratch.
- **engineering:architecture**: when writing ADR-0001 for the schema decision.
- **engineering:documentation**: when writing the README, CONTRIBUTING, and docstrings.

## Sub-agent protocol

- The orchestrator session owns Phases 0 to 2 and the final integration. Sub-agents are
  spawned only after the Phase 2 schema freeze.
- Sub-agents receive: their phase spec from the build prompt, read access to everything,
  write access only to their own package plus its tests. Nobody except the orchestrator
  edits `schema/` after the freeze.
- A sub-agent is done only when its scoped tests pass (`uv run pytest tests/<area> -q`)
  and ruff is clean on its files. It reports back with a list of files touched, tests
  added, and any TODOs left.
- Every phase boundary: re-read the phase spec, run the full suite, update PROGRESS.md
  with the gate checklist, commit with a conventional message.

## Definition of done for v0

Fresh clone passes `uv sync --extra dev && uv run pytest -q` offline on macOS with no
GPU. `uv run rolloutscope analyze tests/fixtures/demo --out report.html` produces a
self-contained HTML report plus a JSON sidecar. All six detectors separate their labeled
fixture pairs. Core packages have no verifiers import. Ruff, mypy, and CI config are
green. Docs exist (README, ADR-0001, CHANGELOG 0.1.0, CONTRIBUTING) and contain no em
dashes. PROGRESS.md shows every phase gate checked.

