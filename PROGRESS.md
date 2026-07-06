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

Status: pending

## Phase 4: detectors (sub-agent B)

Status: pending

## Phase 5: analysis and report (sub-agent C)

Status: pending

## Phase 6: CLI and integration

Status: pending

## Phase 7: docs and release hygiene

Status: pending

## Phase 8: final verification

Status: pending
