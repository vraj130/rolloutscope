# Upstream compatibility matrix

What rolloutscope reads today, what upstream actually writes today, and where
those two have drifted apart. Everything here was verified against upstream
source at a pinned commit or against a captured artifact in
`tests/fixtures/upstream/`, on 2026-09-08. Nothing is recalled from memory.

This document is research output. No adapter is implemented against the current
formats yet: that work waits on the Phase 1 identity and provenance contract,
because an adapter written before identity is settled would have to be rewritten
when it lands.

## Summary

The formats moved, twice, and the second move has not been released yet.

| project | release | date | format | rolloutscope support |
|---|---|---|---|---|
| verifiers | v0.1.7 to v0.1.9.post3 | 2025-11-07 to 2026-01-12 | `results.jsonl` of `RolloutOutput`, `metadata.json` | supported (`verifiers_eval`) |
| verifiers | v0.2.0 to v0.3.0 | 2026-07-10 to 2026-08-07 | both: legacy `RolloutOutput` **and** v1 `Episode` | legacy path supported |
| verifiers | v0.3.1 | 2026-08-24 | v1 only: `traces.jsonl` of `Episode`. `verifiers/types.py` and `verifiers/utils/save_utils.py` are **gone** | **not supported** |
| verifiers | main @ `b5d0424f` | 2026-09-08 | v1, plus `to_record(float_decimals=...)` | **not supported** |
| prime-rl | up to v0.9.1.dev28 (`84e7312f`) | to 2026-09-01 | `rollouts/step_{step}/{kind}/{subset}/traces.jsonl`, both cohorts | **not supported** |
| prime-rl | v0.9.1.dev29 (`e3fdaded`) onward | 2026-09-01 onward | chunked `traces/stream/` + `traces/stream.index.jsonl` + `traces/annotations/<producer>/` | **not supported** |
| prime-rl | main @ `dad79d1c` (v0.9.1.dev46) | 2026-09-08 | as above | **not supported** |

The adapter in `src/rolloutscope/adapters/` recognizes `results.jsonl` /
`metadata.json` and `train_rollouts.jsonl`. Neither filename exists in any
current release of either project. In practice rolloutscope v0.1.0 reads
artifacts from verifiers v0.1.x and from prime-rl versions old enough to still
write `train_rollouts.jsonl`, and nothing newer.

## Verified evidence

**verifiers dropped the legacy format at v0.3.1.** `git ls-tree` on each tag
shows `verifiers/types.py` and `verifiers/utils/save_utils.py` present through
v0.3.0 and absent from v0.3.1 (`b2e4e815`, 2026-08-24) and from `main`. v0.2.0
(`6c64ce6a`, 2026-07-10) is the first tag that carries `verifiers/v1/episode.py`,
so v0.2.0 through v0.3.0 is the overlap window where both formats exist.

**prime-rl replaced the per-step cohort directories between dev28 and dev29.**
`src/prime_rl/monitors/file.py` exists at v0.9.1.dev28 and is gone at
v0.9.1.dev29 (`e3fdaded`, 2026-09-01), replaced by the `monitors/file/` package.
Both tags land on the same day; this is a fast-moving surface.

**prime-rl main needs an unreleased verifiers.** `monitors/file/monitor.py`
calls `episode.to_record(float_decimals=self.config.float_decimals)`. That
keyword arrived in verifiers commit `cfc7c474` (2026-09-02, PR #2495) and is in
no tag, while prime-rl's `pyproject.toml` declares only
`verifiers[harbor]>=0.3.1`. The declared floor is looser than the real
requirement, so pinning "the latest released verifiers" against prime-rl main
raises `TypeError: Episode.to_record() got an unexpected keyword argument
'float_decimals'`. This was reproduced during fixture capture.

## What the current record actually looks like

Captured with the real writers; see `tests/fixtures/upstream/`.

An episode record's top-level keys are `id`, `env`, `task`, `group`, `run`, `ok`,
`errors`, `traces`. Serialization is `exclude_none=True`, so an absent optional
field is missing rather than null.

Four things break the current normalizer, in order of severity.

**1. There is no top-level `example_id` or `reward`.** `RolloutBase` requires
both. An episode record fails validation outright, so pointing the existing
adapter at a v1 file does not degrade gracefully, it rejects every row.

**2. Reward is per trace, named, and weighted.** A trace carries
`rewards: {name: {"score": float, "weight": float} | null}`. The scalar is
`sum(score * weight)` over the non-null entries, which is how upstream's own
index computes it: the captured episode with `tests_pass` at weight 1.0 and
`format_reward` at weight 0.2 indexes as `reward: 1.2`. A `null` entry means
scoring did not run, which is different from a score of 0.0 and must not be
flattened into one. Unweighted named values live separately in `metrics`.

**3. Tool calls are flat.** verifiers v1 `ToolCall` is
`{"id", "type", "name", "arguments"}`. It is **not** the OpenAI-nested
`{"function": {"name", "arguments"}}` shape. `detectors/_text.py` reads
`call["function"]["name"]`, so a v1 record fed to the current detectors renders
no tool-call text at all and every tool-call signal silently disappears. This is
the quietest failure of the four and the one most likely to be mistaken for a
clean run.

**4. Identity is provided, not derived.** The record carries `episode.id`,
`group.id`, `run.id`, `run.work.step`, `env.id`, `task.key`, `task.hash`, and one
`id` per trace. None of these need to be guessed or hashed. `task.key` is stable
task identity and `task.hash` is a content hash of the task data, which maps
directly onto the identity split the Phase 1 contract has to make.

Two more differences worth recording:

- **A failed episode has an empty `traces` list**, with `ok: false` and a
  populated `errors`. It is a real row about a real dispatched task. An adapter
  that keys on traces drops exactly the rows an operator most wants to see.
- **Known task fields are materialized with their defaults.** The captured
  `task.data` gained `network_allow`, `network_block`, `artifacts`, `timeout`,
  and `resources` from the model's own defaults. Unknown fields survive in
  `model_extra`, but a reader cannot tell a defaulted field from a written one.

## The two writers are not the same writer

verifiers' eval CLI writes with
`type_adapter(type(episode)).dump_json(episode, exclude_none=True)`
(`verifiers/v1/cli/output.py`). prime-rl writes with `episode.to_record()`, which
additionally excludes `EXCLUDE_FIELDS` (`multi_modal_data`, `routed_experts`,
`sampling_mask`) from every node.

For the captured episodes the two are byte identical, because none of those
tensor fields is populated. They diverge as soon as a multi-modal or
routed-expert run is recorded: only `to_record` drops the tensors. Treat the
byte-identity as a property of these fixtures, not of the formats.

## Fixture provenance

`tests/fixtures/upstream/` holds three captured layouts. They were produced by
`scripts/capture_upstream_fixtures.py`, which imports the upstream models and
calls the upstream serializers rather than hand-writing JSON, so the bytes are
the bytes those versions write. Content is synthetic throughout: invented task
text, invented tool arguments, invented model output. No real model output, no
customer data, no PII.

| fixture | writer | pin |
|---|---|---|
| `verifiers-0.3.1-eval/` | `verifiers.v1.cli.output.write_episode` | verifiers 0.3.1 from PyPI |
| `prime-rl-v0.9.1.dev28-legacy/` | `Episode.to_record()` at the documented cohort paths | prime-rl `84e7312f` |
| `prime-rl-v0.9.1.dev46-stream/` | upstream `ChunkedJsonl` and `index_row`, fetched and executed from the pinned commit | prime-rl `dad79d1c` |

verifiers is Apache-2.0 and prime-rl is Apache-2.0; both permit this use.

## What a future adapter has to decide

Recorded here so the decisions are visible before the code exists, not
discovered during it.

1. **Format detection.** Three shapes now live under names that collide. A
   directory containing `traces.jsonl` could be a verifiers v1 eval run or a GEPA
   run; the discriminator is the presence of `configs/resolved/*.json` and the
   `run.type` field on the records, not the filename.
2. **Cohort policy.** See `upstream-cohorts.md`. There is no safe default; the
   policy has to be explicit and it has to appear in the manifest.
3. **The scalar reward.** Computing `sum(score * weight)` matches upstream's
   index, but it discards the per-component breakdown that `format_only_wins` and
   the metric-role work in Phase 2 need. Both must be carried.
4. **Multiple traces per episode.** An episode holds one trace per agent seat. A
   judge seat carries no rewards and is excluded from upstream's episode mean.
   Flattening every trace into one rollout would mix a solver with its judge.
5. **Tool-call normalization.** Either normalize the flat v1 shape into the
   nested shape the detectors read, or teach the detectors both. Normalizing at
   the adapter is preferable: one place, and the detectors keep one contract.
6. **The unreleased-verifiers gap.** Supporting prime-rl main means supporting a
   verifiers commit with no tag. Pin the commit in the matrix and say so, rather
   than implying a released version works.

## Refreshing this document

Both projects move fast enough that this matrix has a short shelf life. To
refresh: clone both repositories, re-run the tag sweep that produced the summary
table, re-run `scripts/capture_upstream_fixtures.py` against the new pins, and
update the dates and commits here. Do not update a claim without re-verifying
it against source.
