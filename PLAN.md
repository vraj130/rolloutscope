# PLAN.md: rolloutscope v0 build plan

Package name confirmed: **rolloutscope** (see Decisions Log D-001). This plan is the
Phase 0 deliverable of the build prompt. It is binding on all later phases; deviations
get appended to the Decisions Log, never silently applied.

## A. Architecture summary

**schema**: owns the normalized data contract and nothing else: the `Rollout`
discriminated union (single_turn | multi_turn), `Verdict`, `Finding`, the optional
`TrainingSignals` sidecar, stable content-derived IDs (`ids.py`), streaming JSONL IO
(`io.py`), and the version migration chain (`migrate.py`). Allowed imports: pydantic,
orjson, stdlib. It imports no other rolloutscope package (it is the root of the
dependency DAG) and never imports verifiers or prime-rl. After the Phase 2 freeze only
the orchestrator may change it, with a version bump plus migration.

**adapters**: parses on-disk artifacts (`results.jsonl` + `metadata.json` from verifiers
eval, `train_rollouts.jsonl` from prime-rl steps) into normalized `Rollout` rows,
attaching IDs via `schema.ids` and `step_index` from directory layout only. Allowed
imports: schema, orjson, stdlib. This is the only package that knows the upstream
on-disk shapes, and it knows them from the pinned skill references (verifiers @
`5885ab9c`, prime-rl @ `df2acf48`), never by importing those libraries.

**detectors**: pure functions `detect(rollouts, config) -> list[Verdict]` over
normalized rollouts (or groups), each carrying mandatory evidence spans, plus the
entry-point registry (`rolloutscope.detectors` group) through which built-in and
third-party detectors are discovered identically. Allowed imports: schema, stdlib. No
file IO, no network, no model calls.

**analysis**: streaming aggregates over an iterator of rollouts (run summary, reward
histogram, per-group stats, per-step series, top-k/bottom-k via heaps), single pass and
bounded memory, plus the verdict-to-Finding assembly step (grouping, severity mapping,
exemplar selection). Allowed imports: schema, stdlib. Detector logic stays out.

**report**: `ReportData` (run summary + aggregates + findings) and three pure renderers
over it: rich terminal summary, deterministic sorted-key JSON, and a single-file
self-contained HTML report (inline CSS, server-side SVG charts, `<details>`
collapsibles, zero JS, no CDN, renders from `file://`). Allowed imports: schema,
analysis (aggregate types), rich, jinja2, stdlib. No detector logic in templates.

**cli**: a thin typer app (`analyze`, `convert`, `detectors list`, `schema export`,
`--version`) that only parses arguments, wires adapters + analysis + detectors + report,
and maps findings to exit codes. All real logic lives in the library packages. This is
the only package allowed to import all of the above.

## B. Contract sketch

Base: `references/candidate-schema.md` from the rollout-schema-design skill, adapted,
not reinvented. Upstream verifiers names win on any conflict.

**Rollout union** (discriminated on `kind`):

- `RolloutBase` (pydantic v2, `extra="allow"`): `schema_version` (default `"1.0"`),
  `example_id: int`, `reward: float`, `metrics: dict[str, float]`,
  `is_completed: bool`, `is_truncated: bool`, `timing`, `token_usage`, `answer`,
  `info`, `error`, `stop_condition`, `tool_defs`; adapter-attached identity fields
  `rollout_id`, `group_id`, `run_id`, `step_index`, all optional with default `None`
  so raw verifiers rows still validate before an adapter touches them.
- `SingleTurnRollout` (`kind="single_turn"`): adds `prompt`, `completion`.
- `MultiTurnRollout` (`kind="multi_turn"`): adds `prompt`, `completion`,
  `trajectory: list[TrajectoryStep]`.
- `TrajectoryStep` (`extra="allow"`): `prompt`, `completion`, `response`, `tokens`,
  `reward`, `advantage`, `is_truncated`, `trajectory_id`, `extras`, matching upstream
  `TrajectoryStep` names exactly.
- `Message` (`extra="allow"`): `role`, `content`, `tool_calls`, tolerating provider
  extras.

Deltas from the candidate schema, with reasons (also logged as decisions):

1. `timing` stays a permissive nested model mirroring upstream `RolloutTiming`
   (`extra="allow"`), not the candidate's flattened three-float `Timing`. Upstream wins
   on shape, and lossless round-trip of real rows is a hard requirement (D-004).
2. `prompt` and `completion` are `list[Message] | str | None`. Upstream types them
   `Messages | None`, and deprecated completion-mode envs emit bare strings (D-005).
3. `TrajectoryStep` gains the upstream `response` field the candidate omitted, as
   passthrough `Any` (D-009).
4. `tool_defs` added to the base row (upstream optional field, candidate omitted it).
5. `schema_version` starts at `"1.0"` (build spec) rather than the candidate's
   `"1.0.0"`; migrations key on the major (D-008).
6. Identity fields (`rollout_id`, `group_id`, `run_id`, `step_index`) live on the row
   as optional fields so v0 files are self-joining and v1 can join without restructure
   (D-007).

**Verdict** (frozen contract for sub-agents B and C): `detector: str`, `fired: bool`,
`score: float`, `category: str`, `evidence: list[EvidenceSpan]`,
`rollout_ids: list[str]`. `EvidenceSpan`: `rollout_id`, `field` (which row field the
span lives in, e.g. `completion`), `start`/`end` (optional character offsets), `text`
(the offending span itself), `note`. Structured evidence is a superset of the skill's
`evidence: str`: the report must highlight spans, and a bare string cannot carry
location (D-003). A fired verdict with empty evidence is invalid by construction.

**Finding**: `severity: info | warning | critical`, `title`, `description`,
`detector`, `metrics: dict[str, float]`, `config_used: dict`,
`exemplars: list[EvidenceSpan]` (each span already carries its `rollout_id`).

**TrainingSignals sidecar** (optional, never on the base row): `rollout_id`, `run_id`,
`step_index`, `advantages: list[float] | None`, `is_trainable: bool | None`. v0 never
parses these from disk; the model exists so the v1 monitor hook has a home.

**ID scheme** (`schema/ids.py`):

- `rollout_id`: `sha256` over a canonical orjson serialization (sorted keys) of
  `(example_id, prompt, completion, reward)`, truncated to 16 hex chars.
- `group_id`: derived from the grouping key `example_id`: `grp-{example_id}`.
- `run_id`: truncated sha256 of canonicalized `metadata.json` bytes when present, else
  of the resolved run directory name (D-006).
- `step_index`: optional int attached by the adapter from on-disk step layout, never
  guessed; absent layout means snapshot mode (`None`).
- v1 join contract (documented in the module docstring): activation and
  training-signal stores key on `(run_id, rollout_id, step_index)`; tensors never go
  in this JSONL.

Property: reserializing the same row yields the same IDs (hypothesis-tested).

## C. Phase schedule and sub-agent assignment

| Phase | Owner | Scope | Parallel? |
|---|---|---|---|
| 0 plan | orchestrator | PLAN.md, name decision, confirmation pass | no |
| 1 scaffold | orchestrator | uv env, pyproject from template, CI, skeleton, first tests | no |
| 2 schema | orchestrator | models, ids, io, migrate, fixtures, property tests, FREEZE | no |
| 3 adapters | sub-agent A | `adapters/` + `tests/test_adapters/` | yes, after freeze |
| 4 detectors | sub-agent B | `detectors/` + `tests/test_detectors/` + `tests/fixtures/labeled/` | yes, after freeze |
| 5 analysis + report | sub-agent C | `analysis/`, `report/` + `tests/test_report/` | yes, after freeze |
| 6 CLI + integration | orchestrator | `cli.py`, demo fixtures, e2e test, seam resolution | no |
| 7 docs | orchestrator | README, ADR-0001, CONTRIBUTING, CHANGELOG 0.1.0, docstring and em-dash sweep | no |
| 8 final verification | orchestrator | fresh-clone sim, coverage gate, acceptance walk, tag v0.1.0 | no |

Sub-agent ground rules (from the build prompt): each gets its phase spec, CLAUDE.md,
and the orchestration block; write access only to its own package and tests; nobody
edits `schema/` after the freeze; contract complaints get filed in PROGRESS.md and
worked around locally. Sub-agent C builds Findings by hand in fixtures rather than
waiting on B.

## D. Decisions Log (append-only)

- D-001 (Phase 0, 2026-07-05): Package name stays `rolloutscope`. Evidence: GET
  `https://pypi.org/pypi/rolloutscope/json` returned HTTP 404 (unclaimed) on
  2026-07-05.
- D-002 (Phase 0): `git init` happens at the Phase 0 gate rather than Phase 1, so the
  plan itself is committed and every later gate has a repo to commit into. Phase 1's
  "first commit" becomes its scaffold commit.
- D-003 (Phase 0): `Verdict.evidence` is `list[EvidenceSpan]` (structured), not a bare
  string. The report must highlight offending spans with locations; a string cannot
  carry them. This is a strict superset of the detector skill's `evidence: str`
  contract; the span `text` field preserves the string form.
- D-004 (Phase 0): `timing` mirrors upstream `RolloutTiming` as a permissive nested
  model instead of the candidate's flattened floats. Upstream wins on conflicts and
  round-trip must be lossless.
- D-005 (Phase 0): `prompt` and `completion` typed `list[Message] | str | None`,
  because upstream allows `Messages | None` and completion-mode envs emit strings.
- D-006 (Phase 0): `run_id` derives from `metadata.json` content hash when present,
  else from the run directory name. `group_id` is `grp-{example_id}`. `rollout_id` is
  sha256 truncated to 16 hex over canonical `(example_id, prompt, completion, reward)`.
- D-007 (Phase 0): Identity fields sit on the row as optional adapter-attached fields
  (default `None`) so raw verifiers rows validate unchanged and adapters always
  populate them.
- D-008 (Phase 0): `schema_version` is `"1.0"` per the build spec (candidate said
  `"1.0.0"`); the migration chain keys on the major version either way.
- D-009 (Phase 0): `TrajectoryStep` includes upstream's `response` field as
  passthrough `Any`; the candidate omitted it and `extra="allow"` alone would hide the
  contract.
- D-010 (Phase 0): the pre-existing .gitignore ignores CLAUDE.md (the user's explicit
  choice, made before this build started); it stays ignored, and .claude/ (local
  skills, local settings) is ignored with it. The CLAUDE.md layout listing is treated
  as describing the working tree, not the committed tree.
- D-011 (Phase 1): dev dependencies are declared in the pyproject `dev` extra (filled
  from the scaffold template) and installed with `uv sync --extra dev`, rather than
  the literal `uv add --dev` from the build prompt. Reason: `uv add --dev` writes to
  PEP 735 dependency groups, not extras, which would break the required
  `uv sync --extra dev` workflow. Installs still go exclusively through uv.

## E. Risk self-critique

1. **Schema too rigid for real verifiers rows.** Mitigation: `extra="allow"` on the
   row, message, step, timing, and token-usage models so `state_columns` and provider
   extras survive; round-trip tests on fixtures shaped exactly per
   `on-disk-format.md` asserting unknown keys are preserved byte-for-byte through
   validate/dump; hypothesis property tests for round-trip stability.
2. **Adapter drift as verifiers evolves.** Mitigation: all format knowledge confined
   behind the single adapter boundary (`adapters/` is the only package aware of
   upstream shapes), pinned-commit references recorded in module docstrings
   (verifiers @ `5885ab9c`, prime-rl @ `df2acf48`), and a documented refresh
   procedure in the verifiers-ground-truth skill. A drift fix touches one package.
3. **Detector false positives make the tool noise.** Mitigation: every detector ships
   a labeled clean fixture that must stay silent in tests, documents its known FP
   modes in its docstring, and exposes every threshold via `DetectorConfig` with
   conservative defaults labeled heuristic. The suite emits a precision/recall
   artifact over the labeled fixtures so noise regressions are visible.
4. **Report bloat or external dependencies break the CI-artifact use case.**
   Mitigation: one self-contained HTML file with inline CSS and hand-rolled
   server-side SVG (no chart libs), `<details>` for collapsing (zero JS), no CDN, no
   fetch, no external fonts; an automated self-containment test asserts single-file
   output and no external `http(s)` references outside text content.
5. **uv misuse leaks installs into the base environment.** Mitigation: golden rule 1,
   every command in docs, CI, and this build runs through `uv run` or
   `uv sync`; CI uses `astral-sh/setup-uv`; `uv.lock` and `.python-version` are
   committed; Phase 8 runs a fresh-clone simulation in a temp dir proving no hidden
   global state.

## CONFIRMATION

Every acceptance criterion from the build prompt, mapped to the phase that satisfies
it and the phase that proves it:

| # | Acceptance criterion | Built in | Proven in |
|---|---|---|---|
| 1 | Fresh clone: `uv sync --extra dev && uv run pytest -q` green, offline, macOS, no GPU | Phase 1 (env, lock) | Phase 8 fresh-clone simulation |
| 2 | `analyze tests/fixtures/demo` produces terminal summary, deterministic JSON, self-contained HTML from `file://` | Phase 5 (renderers), Phase 6 (CLI, demo fixtures) | Phase 6 e2e test, Phase 8 walk |
| 3 | Six detectors, entry-point discoverable, each separating a labeled pair, documented FP modes, configurable thresholds, no fabricated numbers | Phase 4 | Phase 4 scoped tests, Phase 8 walk |
| 4 | schema/detectors/analysis/report import no verifiers or prime-rl; adapters never read advantages / is_trainable from disk | Phases 2 to 5 (design rule) | Phase 8 grep evidence |
| 5 | Per-row schema_version, discriminated union with JSON Schema export, extras preserved, migration chain tested, IDs stable, hypothesis tests green | Phase 2 | Phase 2 gate, Phase 8 walk |
| 6 | Streaming IO: generator-based, bad rows skipped and logged, never fatal | Phase 2 (`io.py`), Phase 3 (adapters) | Phase 2/3 tests |
| 7 | Ruff and mypy clean, CI for 3.11 to 3.13 via uv, coverage of schema + detectors at 85 percent or better | Phase 1 (CI), all phases (lint) | Phase 8 coverage run |
| 8 | README quickstart works, ADR-0001, CONTRIBUTING with third-party detector walkthrough, CHANGELOG 0.1.0, zero em dashes | Phase 7 | Phase 8 sweep |
| 9 | PROGRESS.md shows every gate checked with evidence, the freeze commit, sub-agent reports | every phase gate | Phase 8 final entry |

No criterion is unowned. The confirmation pass re-read the full build prompt on
2026-07-05 before this table was written.
