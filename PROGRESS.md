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

Status: pending

## Phase 2: schema contract (freeze at gate)

Status: pending

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
