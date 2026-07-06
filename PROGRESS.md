# PROGRESS.md

Phase gate log for the rolloutscope v0 build. Each phase gets a checklist walked with
evidence (a command run or a file inspected), plus deviations and TODOs. The gate
ritual: re-read the phase spec and CLAUDE.md, verify the Definition of Done item by
item, run the suite and lint, update this file, commit.

## Phase 0: plan, think, re-read, confirm

Status: COMPLETE (2026-07-05)

- [x] CLAUDE.md read fully. Evidence: loaded at session start and re-read during the
      confirmation pass.
- [x] All four project skills read fully, including references. Evidence: SKILL.md for
      verifiers-ground-truth, rollout-schema-design, reward-hacking-detectors,
      python-oss-library-scaffold, plus core-types.md, env-classes.md,
      on-disk-format.md, candidate-schema.md, versioning-and-streaming.md,
      taxonomy-and-sources.md, pyproject.toml.template, ci.yml.template,
      plugin-pattern.md.
- [x] PLAN.md written with all five sections (architecture summary, contract sketch,
      phase schedule and sub-agent assignment, decisions log, risk self-critique).
      Evidence: PLAN.md sections A through E.
- [x] Package name decided. Evidence: GET https://pypi.org/pypi/rolloutscope/json
      returned HTTP 404 on 2026-07-05, name is free, recorded as D-001. rolloutscope
      is used everywhere.
- [x] Confirmation pass done. Evidence: PLAN.md CONFIRMATION table maps all nine
      acceptance criteria to an owning phase and a proving phase; none unowned.
- [x] Gate ritual: pytest and ruff are N/A in Phase 0 (no pyproject or code exists
      yet; the project is scaffolded in Phase 1). Recorded here rather than skipped
      silently.
- [x] Commit. Evidence: phase 0 commit in git log (git init happened at this gate per
      D-002 so the plan itself is versioned).

Deviations:
- git init pulled forward from Phase 1 to Phase 0 (D-002) so every gate can commit.
- The pre-existing .gitignore ignores CLAUDE.md (user's explicit choice, kept), even
  though the CLAUDE.md layout section lists it as checked in. The .claude/ directory
  (local skills and settings) is ignored for the same reason (D-010).

TODOs carried forward: none.

## Phase 1: environment and scaffold

Status: COMPLETE (2026-07-05)

- [x] uv verified. Evidence: `uv --version` printed uv 0.11.14.
- [x] Python pinned. Evidence: `uv python install 3.12` installed cpython 3.12.13;
      .python-version contains `3.12`; requires-python is `>=3.11,<3.14`.
- [x] pyproject from the scaffold template, every placeholder filled: hatchling build,
      metadata, `[project.scripts] rolloutscope`, entry-point group
      `rolloutscope.detectors`, ruff lint + format config, pytest + coverage + mypy
      config, dev extra. Evidence: pyproject.toml.
- [x] Runtime deps pydantic, orjson, typer, rich, jinja2; dev deps pytest, pytest-cov,
      hypothesis, ruff, mypy; nothing else, no torch or inference libraries.
      Evidence: pyproject.toml dependency lists and uv.lock.
- [x] CI from the scaffold template: one lint job (ruff check, format check, mypy),
      one test job across 3.11 / 3.12 / 3.13, all via uv with caching. Evidence:
      .github/workflows/ci.yml.
- [x] Directory skeleton per CLAUDE.md layout with empty __init__.py files and a
      placeholder import test. Evidence: src/rolloutscope/{schema,adapters,detectors,
      analysis,report}/__init__.py, report/templates/, tests/test_placeholder.py.
- [x] MIT LICENSE, .gitignore (Python + uv + macOS), CHANGELOG.md with Unreleased,
      PROGRESS.md seeded (done at Phase 0). Evidence: files present.
- [x] Smoke checks. Evidence: `uv sync --extra dev` resolved and installed;
      `uv run python -c "import rolloutscope, pydantic, orjson, typer, rich, jinja2"`
      printed imports ok; `uv run pytest -q` 1 passed; `uv run ruff check .` all
      checks passed; `uv run ruff format .` 8 files unchanged; `uv run mypy src/` no
      issues in 7 files; `uv run rolloutscope --version` printed 0.1.0.
- [x] pip never invoked at any point.
- [x] Commit at gate.

Deviations:
- Dev deps installed via the dev extra + `uv sync --extra dev` instead of the literal
  `uv add --dev` (would target dependency groups and break `--extra dev`), D-011.
- git init already happened at the Phase 0 gate (D-002); this phase's commit is the
  scaffold commit.

TODOs carried forward: none.

## Phase 2: schema contract (freeze at gate)

Status: COMPLETE (2026-07-05)

**SCHEMA FROZEN at commit 5f87593.** From this point, schema changes require the
orchestrator session, a version bump, and a migration entry. No sub-agent edits
schema/.

- [x] models.py: Rollout discriminated union on kind (single_turn | multi_turn),
      per-row schema_version starting at "1.0", extra="allow" on row, message, step,
      timing, and token models, field names compatible with verifiers RolloutOutput
      (example_id, prompt, completion, reward, timing, is_completed, is_truncated,
      metrics, answer, info, error, stop_condition, trajectory, tool_defs,
      token_usage). Evidence: src/rolloutscope/schema/models.py, round-trip tests.
- [x] TrainingSignals sidecar model, never on the base row. Evidence: models.py.
- [x] Verdict and Finding frozen in schema/findings.py: Verdict(detector, fired,
      score, category, evidence, rollout_ids) with mandatory evidence on fire
      (model validator); Finding(severity info|warning|critical, title, description,
      detector, metrics, config_used, exemplars). Evidence: findings.py.
- [x] ids.py: rollout_id sha256/16-hex over canonical (example_id, prompt,
      completion, reward); group_id grp-{example_id}; run_id from manifest hash or
      directory name; step_index adapter-attached only; v1 join contract
      (run_id, rollout_id, step_index) in the module docstring. Evidence: ids.py,
      test_ids.py.
- [x] io.py: generator-based line-by-line orjson reader, per-row validation,
      skip-and-log with row number and reason, never fatal; streaming writer.
      Evidence: io.py, test_io.py (bad JSON, non-object, invalid row, future
      version all skipped and logged).
- [x] migrate.py: registry keyed by major, migrate_row entry point, identity 1.0
      entry, unit-tested fake 0.x to 1.0 example. Evidence: migrate.py,
      test_migrate.py.
- [x] JSON Schema export with working discriminator mapping. Evidence:
      rollout_json_schema(), test_json_schema.py.
- [x] Fixtures shaped exactly per on-disk-format.md: 5-row grouped single-turn
      results.jsonl with state_columns extras, one multi-turn ToolEnv row with
      2-step trajectory, one metadata.json, all synthetic and labeled so in
      tests/fixtures/README.md.
- [x] Tests a through e: single-turn round-trip incl unknown keys, multi-turn
      round-trip incl trajectory, JSON Schema discriminator, migration chain,
      hypothesis round-trip and ID stability properties. Evidence:
      `uv run pytest -q` 25 passed.
- [x] Gate: `uv run pytest -q` 25 passed; `uv run ruff check .` all checks passed;
      `uv run ruff format .` clean; `uv run mypy src/` no issues in 12 files.
- [x] No package outside schema/ imports anything yet (all still empty).

Deviations:
- The 5-row fixture is grouped 3 + 2 rather than an even rollouts_per_example
  split; the fixture README documents the reading (one rollout dropped on error,
  as real appended files exhibit).

TODOs carried forward: none.

## Phase 3: adapters (sub-agent A)

Status: COMPLETE (2026-07-05), sub-agent A report follows; full-suite integration
gate happens with Phase 6 (per user instruction, Phase 6 waits for go-ahead).

- [x] adapters/base.py: Adapter protocol (detect / load / load_run), RunManifest +
      SourceFile, step_index_from_name (prime-rl step layout mapping encoded here
      and nowhere else: step_<int> case-insensitive or bare integer, else None),
      normalize_row computing IDs from raw disk values before validation,
      iter_normalized_rows with skip-and-log.
- [x] adapters/verifiers_eval.py: run dir or direct .jsonl; run_id from
      metadata.json hash when present, else name hash; step_index always None.
- [x] adapters/prime_rl_train.py: direct file, single step dir, or run dir of step
      dirs; numeric step ordering (step_10 after step_2); shared run_id across all
      three entry forms; never parses advantages / is_trainable.
- [x] resolve_adapter with documented tie-break (verifiers_eval first).
- [x] New synthetic fixture tests/fixtures/prime_rl_run/ (two steps, 3 + 2 rows,
      README labeling it synthetic) proving step_index attachment and ordering.
- [x] 37 tests: golden rows including literal content-derived IDs, unknown-key
      preservation, bad-row skip-and-log, resolve routing, 12 parametrized
      step-name cases. Evidence: `uv run pytest tests/test_adapters -q` 37 passed
      (re-run by orchestrator).
- [x] Orchestrator verification: scoped tests re-run green; grep shows no
      verifiers/prime-rl imports and no em dashes; schema/ untouched.

Sub-agent A TODOs (also as TODO comments in source):
1. base.py _STEP_DIR_RE: the exact prime-rl step_path directory naming at pin
   df2acf48 is not pinned by the reference; until then only step_<int> and bare
   integer names carry step_index.
2. prime_rl_train.py _manifest: whether prime-rl writes a run-level manifest next
   to step dirs is unpinned; run_id comes from the run root name for now.

Contract friction noted (no contract change): adapter skip-and-log catches a wider
exception set than io.read_rollouts; Message.tool_calls None default materializes
in dumps so golden tests assert full dumps, not raw bytes.

## Phase 4: detectors (sub-agent B)

Status: COMPLETE (2026-07-05), sub-agent B report follows. B was interrupted once by
a session limit and resumed with context intact; no work was lost.

- [x] detectors/base.py: Detector protocol detect(rollouts, config) -> list[Verdict],
      DetectorConfig with six per-detector threshold sub-models (every threshold a
      configurable Field labeled heuristic), entry-point registry over
      entry_points(group="rolloutscope.detectors") with guarded loading and a
      builtin fallback; shared text helpers in _text.py (one copy, not six).
- [x] Six detectors implemented per the CLAUDE.md catalog, each with a docstring
      listing known false-positive modes and evidence spans on every fired verdict:
      verifier_tamper, reward_saturation_group_collapse (on-disk GRPO proxy, step
      trend mode when step_index present), length_inflation, format_only_wins
      (graceful degradation when metrics absent), degenerate_repetition,
      answer_leakage_echo.
- [x] Entry points: all six registered in pyproject and discoverable. Evidence:
      orchestrator ran entry_points(group="rolloutscope.detectors") and got all six.
- [x] Labeled fixtures: tests/fixtures/labeled/ 12 synthetic hacked/clean pairs plus
      README; every detector separates its pair.
- [x] Precision/recall artifact: tests/artifacts/detector_fixture_metrics.txt, all
      six detectors precision 1.00 recall 1.00 with FP counted against ALL clean
      files, zero cross-fires.
- [x] TRACE integration test behind the integration marker, stdlib urllib only,
      skips gracefully on network/auth failure (currently the datasets-server split
      listing returns 401; license check passes).
- [x] No fabricated citations: only confirmed TRACE arXiv:2601.20103 cited;
      unverified sources (arXiv 2606.04923, 2605.02964) left as TODOs.
- [x] Orchestrator gate: `uv run pytest -q` 153 passed 1 deselected;
      `uv run mypy src/` clean on 30 files (strict); `uv run ruff check .` clean;
      no em dashes in B's files; schema/ untouched.

Contract friction (schema NOT amended): mypy rejects unknown kwargs on the frozen
Verdict constructor, so extra markers (mode, reason) attach via
Verdict.model_validate, which extra="allow" preserves at runtime.

Sub-agent B TODOs carried forward:
1. Pull and verify arXiv 2606.04923 (rubric-hacking patterns) and arXiv 2605.02964
   (RHB tool-use categories) before citing any number; thresholds stay heuristic.
2. v1: true GRPO zero-advantage signal via the TrainingSignals sidecar.
3. TRACE viewer appears gated (HTTP 401); token-authenticated fetch is a follow-up.

## Phase 5: analysis and report (sub-agent C)

Status: COMPLETE (2026-07-05), sub-agent C report follows; completed before Phase 4
(the phases are independent by design; C hand-built its Verdict fixtures).

- [x] analysis/aggregates.py: single-pass streaming aggregation (Welford moments,
      bounded heaps for top-k/bottom-k, fixed histogram bins with underflow and
      overflow counters documented as the single-pass alternative), run summary,
      per-group stats, per-step series when step_index present.
- [x] analysis/findings.py: Verdict-to-Finding assembly (grouping by detector and
      category, configurable SeverityThresholds documented heuristic: max fired
      score >= 0.8 critical, >= 0.5 warning, else info; exemplar selection;
      include_clean flag for info findings on clean detectors).
- [x] report/model.py: ReportData composing run summary, aggregates, findings, and
      reproducibility fields (tool version, schema_version, input file hashes via
      streaming hash_file).
- [x] report/terminal.py rich renderer; report/json_out.py deterministic
      (sorted keys, byte-identical re-render, verified across dict insertion
      orders); report/html.py + svg.py (exactly two chart helpers) +
      templates/report.html.j2: one self-contained file, inline CSS, zero JS,
      <details> collapsibles, <mark> evidence highlighting, config appendix,
      reproducibility footer; autoescape on; PackageLoader so it works installed.
- [x] 45 tests in tests/test_report/ including hand-computed aggregate math
      (mean 0.64, per-group variances, histogram counts [0,0,2,0,0,0,0,0,1,2],
      top/bottom-k tie order), streaming test on a pure generator, JSON byte
      determinism, HTML self-containment (no src=, href=, @import, url(, <script,
      <link, <iframe, <object, <embed anywhere).
- [x] Orchestrator verification: `uv run pytest tests/test_report -q` 45 passed
      (re-run); no em dashes, no upstream imports, no <script> in template.

Notes from sub-agent C:
- Dead-group fraction excludes singleton groups (documented heuristic).
- Finding.metrics is dict[str, float] so counts are stored as floats; renderers
  format with %g. No contract change needed.
- C observed transient mypy errors in detectors/ (Sub-agent B's package, still in
  flight at the time); B's definition of done requires mypy green, checked at the
  Phase 4 gate.

## Phase 6: CLI and integration

Status: COMPLETE (2026-07-05)

- [x] cli.py: thin typer app whose real work lives in importable, unit-tested
      functions (load_run_config, build_report_data, exit_code_for), not in the
      command bodies. This is the only module that imports adapters, detectors,
      analysis, and report together (per PLAN section A). Evidence: cli.py.
- [x] Command set from PLAN section A: analyze, convert, detectors list,
      schema export, plus the pre-existing --version. Evidence: cli.py, all
      exercised by tests/test_cli/test_e2e.py.
- [x] analyze: resolves the adapter, materializes rollouts (detectors do
      whole-group contrastive analysis, so a stream will not do), runs the
      discovered detectors, assembles findings, aggregates, and renders the
      terminal summary plus optional --json and --out HTML. --config,
      --detector (repeatable), --include-clean, --fail-on, --quiet, --verbose.
      Exit code follows --fail-on (default none, so the DoD command exits 0).
- [x] convert: streams normalized rollouts to JSONL via schema.write_rollouts
      (golden rule 6: safe on files larger than RAM); ids attached, unknown
      keys preserved.
- [x] TOML config (stdlib tomllib, no new dependency): RunConfig maps
      [detectors.<name>], [aggregation], [severity] onto the frozen config
      models; a bad file becomes a clean typer.BadParameter, not a traceback.
- [x] Demo fixtures at tests/fixtures/demo/ (verifiers-eval shaped: results.jsonl
      + metadata.json + README, all synthetic and labeled). Ten rows fire four
      detectors (verifier_tamper, degenerate_repetition, answer_leakage_echo,
      format_only_wins) and include clean rollouts for a realistic reward
      histogram. Exceeds the "3+ detectors" bar.
- [x] End-to-end tests (tests/test_cli/test_e2e.py, 20 tests): analyze writes a
      self-contained HTML report (external-reference scan finds nothing) and a
      byte-deterministic JSON sidecar; convert round-trips through read_rollouts;
      detectors list shows six; schema export is a valid discriminated union;
      --fail-on and unknown-detector / unrecognized-path error paths covered.
- [x] Seam resolution: the DoD command
      `uv run rolloutscope analyze tests/fixtures/demo --out report.html`
      produces the HTML report plus (with --json) the sidecar, exit 0. No
      contract friction surfaced at integration; schema/ untouched.
- [x] Gate: `uv run pytest -q` 173 passed, 1 deselected; `uv run mypy src/`
      clean on 30 files (strict); `uv run ruff check .` and
      `uv run ruff format --check .` clean; no verifiers / prime-rl imports in
      src; no em dashes in touched files; schema/ byte-identical to the freeze.

Deviations:
- ruff B008 flags typer's Argument()/Option() parameter-default factories.
  Resolved with the documented `[tool.ruff.lint.flake8-bugbear]
  extend-immutable-calls` config (those factories return immutable descriptor
  objects), the standard typer-plus-ruff setup, rather than restructuring every
  command signature. pyproject tooling config only; no schema or code contract
  change.
- analyze materializes rollouts into a list rather than streaming, because the
  detector contract is over a whole Sequence (contrastive group analysis).
  convert and the library aggregate remain streaming; documented on
  build_report_data.

TODOs carried forward: none new.

## Phase 7: docs and release hygiene

Status: COMPLETE (2026-07-05)

- [x] README rewritten from the placeholder: what and why, uv install, a
      quickstart whose commands were run verbatim from a fresh directory
      (analyze the demo to HTML + JSON, detectors list, convert, schema
      export), the accepted-input list, CI exit codes, the six-detector
      catalog table with categories and signals, the TOML config example, the
      design rules (schema-as-contract, content IDs, no upstream imports in
      core, evidence-carrying detectors, self-contained report), and the dev
      command suite. Evidence: README.md; all quickstart commands re-run green.
- [x] CONTRIBUTING written with the third-party detector walkthrough
      (criterion 8): dev setup, the non-negotiable conventions (uv only, no em
      dashes, schema freeze, no upstream imports, mandatory evidence, fixtures),
      and a four-step plugin guide (implement the Detector protocol, register
      the rolloutscope.detectors entry point, install and confirm discovery,
      ship a labeled fixture pair) plus the core-detector variant. The example
      detector was executed to confirm its imports and API are correct.
      Evidence: CONTRIBUTING.md; scratch run of the example detector.
- [x] CHANGELOG cut to 0.1.0 (Keep a Changelog): the accumulated Added entries
      moved under `## [0.1.0] - 2026-07-05` with the CLI, demo fixture, e2e
      suite, and docs entries appended; an empty `## [Unreleased]` kept on top.
      Evidence: CHANGELOG.md.
- [x] Docstring pass: an introspection audit over every public module found
      that every public function, class, and method carries a docstring
      (CLAUDE.md docstring rule; detectors document their false-positive modes).
      Evidence: audit script over rolloutscope.* printed all documented.
- [x] Em-dash sweep across the whole tracked repo (em, en, and horizontal bar):
      zero occurrences in any authored text. Evidence: git ls-files grep, empty.
- [x] Gate: `uv run pytest -q` 173 passed, 1 deselected; `uv run ruff check .`
      and `uv run ruff format --check .` clean; `uv run mypy src/` clean; the
      exact README quickstart commands succeed from a fresh working directory.

Deviations:
- The CLAUDE.md skills map points at an engineering:documentation skill that is
  not installed in this environment; the README, CONTRIBUTING, and docstrings
  were written directly following the python-oss-library-scaffold release
  hygiene guidance (semver, Keep a Changelog, library-first framing).

TODOs carried forward: none new.

## Phase 8: final verification

Status: COMPLETE (2026-07-05)

Fresh-clone simulation: `git clone` of the repo into a temp dir (committed state
only; CLAUDE.md and .claude/ are gitignored and correctly absent), then
`uv sync --extra dev --offline` (all deps resolved from cache, no network) and
`uv run --offline pytest -q` gave 173 passed, 1 deselected. The DoD command
`uv run rolloutscope analyze tests/fixtures/demo --out report.html --json findings.json`
ran in that clone and produced both files; the HTML's only http token is the SVG
namespace URI (`xmlns="http://www.w3.org/2000/svg"`), so it opens from file://.

Coverage gate: `pytest --cov=rolloutscope.schema --cov=rolloutscope.detectors`
reported 93% combined (well above the 85% bar); no file below 76%.

Acceptance walk, every criterion from the build prompt with its evidence:

| # | Criterion | Result | Evidence |
|---|---|---|---|
| 1 | Fresh clone syncs and tests green, offline, macOS, no GPU | PASS | clone + `uv sync --extra dev --offline` + `uv run --offline pytest -q` = 173 passed, 1 deselected |
| 2 | analyze demo yields terminal summary, deterministic JSON, self-contained HTML from file:// | PASS | DoD command in clone; HTML external-ref scan finds only the SVG xmlns; JSON byte-identical across runs (e2e test) |
| 3 | Six detectors, entry-point discoverable, each separates a labeled pair, documented FP modes, configurable thresholds, no fabricated numbers | PASS | entry_points reports all six; detector_fixture_metrics.txt shows precision 1.00 recall 1.00 each; thresholds are labeled-heuristic Fields |
| 4 | schema/detectors/analysis/report import no verifiers or prime-rl; adapters never read advantages / is_trainable from disk | PASS | grep of the four core packages is empty; the only adapter mention of those keys is a doc comment, no key access |
| 5 | Per-row schema_version, discriminated union with JSON Schema export, extras preserved, migration chain tested, IDs stable, hypothesis green | PASS | schema export shows the kind discriminator mapping; tests/test_schema incl. test_properties.py (hypothesis) pass |
| 6 | Streaming IO: generator-based, bad rows skipped and logged, never fatal | PASS | schema/io.py and adapters/base.py iterators; skip-and-log covered by test_io and adapter tests |
| 7 | Ruff and mypy clean, CI 3.11 to 3.13 via uv, schema + detectors coverage >= 85% | PASS | ruff check + format --check clean; mypy clean on 30 files; ci.yml matrix 3.11/3.12/3.13 via setup-uv; coverage 93% |
| 8 | README quickstart works, ADR-0001, CONTRIBUTING with detector walkthrough, CHANGELOG 0.1.0, zero em dashes | PASS | README commands re-run from a fresh dir; all four docs present; full-repo em/en-dash sweep empty |
| 9 | PROGRESS.md shows every gate checked with evidence, the freeze commit, sub-agent reports | PASS | this file: Phases 0 to 8 all COMPLETE with checklists; freeze recorded at 5f87593; sub-agent A/B/C reports in Phases 3/4/5 |

Schema confirmed byte-identical to the freeze (empty `git diff` against 5f87593).

Release: tagged v0.1.0 at the final commit (local annotated tag; no remote is
configured, so nothing is pushed).

Deviations: none. The v0 Definition of Done is met.
