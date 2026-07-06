# Contributing to rolloutscope

Thanks for helping. This guide covers the development setup, the project conventions that
are non-negotiable, and a walkthrough of the most common contribution: a new detector,
shipped as a plugin.

## Development setup

rolloutscope uses [uv](https://docs.astral.sh/uv/) for everything. Never use pip, conda,
or the system Python.

```bash
uv sync --extra dev     # create the venv and install rolloutscope plus dev tools
```

The full local check suite, all offline:

```bash
uv run pytest -q              # unit tests
uv run pytest -m integration  # optional network tests (skip gracefully offline)
uv run ruff check .           # lint
uv run ruff format .          # format
uv run mypy src/              # types (strict)
```

A change is ready when `uv run pytest -q`, `uv run ruff check .`,
`uv run ruff format --check .`, and `uv run mypy src/` are all green.

## Project conventions

- **uv only.** All installs go through `uv add` / `uv sync`, all execution through
  `uv run`. `uv.lock` is committed.
- **No em dashes in any repo text.** README, docstrings, comments, ADRs, the CHANGELOG,
  and report copy use commas, colons, or parentheses instead. This is a hard convention.
- **The schema is frozen.** The models in `src/rolloutscope/schema/` are the product's
  contract. Changes there require a version bump and a migration entry, not an ad hoc
  edit; open an issue first. Everything else (adapters, detectors, analysis, report,
  CLI) is built against those types.
- **Core never imports verifiers or prime-rl.** Only the `adapters` package knows the
  upstream on-disk shapes, and it parses them from the pinned reference docs rather than
  importing those libraries. Do not add torch, transformers, or any inference dependency:
  v0 is CPU-only and loads no model.
- **Detectors carry evidence.** Every fired verdict must include the offending span. A
  flag without its span fails validation by construction, and that is on purpose.
- **Tests ship with fixtures.** Every detector needs at least one labeled hacked fixture
  and one clean fixture, and it must separate them.

## Writing a detector (as a plugin)

Detectors are discovered through the `rolloutscope.detectors` entry-point group, exactly
the mechanism the six built-ins use. That means you can add one from your own package
without editing rolloutscope: implement the protocol, register an entry point, install
your package into the same environment.

### 1. Implement the `Detector` protocol

A detector needs a `name`, a `category`, and a pure `detect` function over normalized
rollouts that returns `Verdict` objects.

```python
# my_package/detectors.py
from collections.abc import Sequence

from rolloutscope.analysis import content_text          # flattens prompt/completion to text
from rolloutscope.detectors import DetectorConfig
from rolloutscope.schema import EvidenceSpan, Rollout, Verdict


class EmptyHighRewardDetector:
    """Flags rollouts that earned near-max reward for an empty completion.

    Known false positives: environments whose reward legitimately ignores the
    completion text (for example a pure tool-call turn scored on side effects).
    """

    name = "empty_high_reward"
    category = "degeneracy"

    def detect(self, rollouts: Sequence[Rollout], config: DetectorConfig) -> list[Verdict]:
        verdicts: list[Verdict] = []
        for rollout in rollouts:
            text = content_text(rollout.completion).strip()
            fired = rollout.reward >= 0.9 and len(text) < 3
            rollout_ids = [rollout.rollout_id] if rollout.rollout_id else []
            evidence = (
                [
                    EvidenceSpan(
                        rollout_id=rollout.rollout_id or "",
                        field="completion",
                        text=text or "<empty>",
                        note=f"reward {rollout.reward} for an empty completion",
                    )
                ]
                if fired
                else []
            )
            verdicts.append(
                Verdict(
                    detector=self.name,
                    fired=fired,
                    score=1.0 if fired else 0.0,
                    category=self.category,
                    evidence=evidence,
                    rollout_ids=rollout_ids,
                )
            )
        return verdicts
```

Notes:

- Return a verdict for every rollout you inspected, not only the ones that fired. The
  non-fired verdicts (evidence may be empty when `fired` is False) let the report say
  "checked, clean" and compute fire rates.
- A fired verdict with no evidence raises; always attach the span.
- `score` is a heuristic in [0, 1]. Do not hardcode a number from a paper without a
  verified citation; leave a TODO with the source instead.
- Group and step detectors read `rollout.group_id` and `rollout.step_index` and may emit
  one verdict per group covering many `rollout_ids`.
- Built-in detectors read their thresholds from a sub-model on `DetectorConfig`. A
  third-party detector can ignore `config` or read its own settings from its own object.

### 2. Register the entry point

In your package's `pyproject.toml`:

```toml
[project.entry-points."rolloutscope.detectors"]
empty_high_reward = "my_package.detectors:EmptyHighRewardDetector"
```

### 3. Install and confirm discovery

Install your package into the same environment as rolloutscope, then:

```bash
uv run rolloutscope detectors list
```

Your detector appears alongside the built-ins, and `analyze` runs it automatically. A
plugin that fails to import or instantiate is skipped with a logged warning rather than
crashing discovery, so a broken third-party detector never takes the tool down.

### 4. Ship fixtures

Include at least one hacked rollout your detector fires on and one clean rollout it stays
silent on, both in the normalized schema, and a test that asserts the separation. The
built-in fixtures under `tests/fixtures/labeled/` are the model to follow: synthetic,
deterministic, and globally clean (a clean fixture for one detector should not trip any
other).

## Adding a detector to the core

If a detector belongs in rolloutscope itself rather than a plugin, the shape is the same,
plus: put the module in `src/rolloutscope/detectors/`, add its threshold sub-model to
`DetectorConfig`, register it in this repo's `pyproject.toml` entry points, add the
labeled fixture pair, and add it to the catalog table in the README. Keep the
`content_text` and shared-regex helpers in `_text.py` rather than duplicating them.

## Commit and PR hygiene

- Conventional commit subjects (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`).
- Update `CHANGELOG.md` under `Unreleased` (Keep a Changelog format).
- Keep the docs in sync: a new detector updates the README catalog; a new CLI flag
  updates the README quickstart.
