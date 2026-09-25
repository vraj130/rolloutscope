# PROGRESS.md

Current status and decision log. Direction lives in [PLAN.md](PLAN.md). The v0 build log
(July to September 2026) is archived at
[docs/history/v0-build-log.md](docs/history/v0-build-log.md).

Keep this file short. Update the status table when a milestone moves, append decisions
at the bottom, and write any experiment's pass or fail criterion here before the run.

## Status (2026-09-25)

| Milestone | State | Evidence |
|---|---|---|
| M0 housekeeping | done once v0.2.0 is tagged | docs refocus commit; version bump commit |
| M1 reproduce rubric hacking | not started | |
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

None yet. Write each one here, dated, before the run it governs.

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
