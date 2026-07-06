# ADR-0001: Normalized rollout schema as the stable contract

**Status:** Accepted
**Date:** 2026-07-05
**Deciders:** rolloutscope maintainers (orchestrator session, Phase 2)

## Context

rolloutscope debugs RL rollouts for reward hacking. Its inputs come from at least two
producers today (verifiers `evaluate()` output and prime-rl orchestrator training
dumps), both of which evolve quickly and neither of which publishes a canonical
cross-framework rollout schema. Detectors, aggregates, and reports need one typed
shape to code against, and v1 will add white-box signals (activations, representation
drift, SAE features) that must join onto v0 data without re-ingesting anything.

Forces at play: upstream field names change fast (any recalled API shape is likely
stale); rollout files are append-only JSONL that can exceed RAM; verifiers injects
arbitrary `state_columns` keys per row; RL training signals (`advantages`,
`is_trainable`) exist only in memory during training, never on disk; and the tool must
run offline, CPU-only, with no verifiers or prime-rl import in its core.

## Decision

Adopt a versioned, discriminated-union normalized schema as the product's stable
contract, frozen at commit 5f87593:

- `Rollout` is a pydantic v2 discriminated union on `kind`
  (`single_turn` | `multi_turn`), with `extra="allow"` on row, message, step, timing,
  and token models so unknown upstream keys survive round-trips.
- Field names stay compatible with verifiers `RolloutOutput` (pinned at commit
  `5885ab9c`); upstream wins on any naming conflict.
- Every row carries `schema_version` (starting `"1.0"`); changes within a major are
  additive only, breaking changes bump the major and ship a migration function.
- Stable content-derived IDs: `rollout_id` (truncated sha256 over canonical
  `example_id`, `prompt`, `completion`, `reward`), `group_id` (from `example_id`),
  `run_id` (from the manifest), optional adapter-attached `step_index`. Future
  white-box stores join on `(run_id, rollout_id, step_index)`; tensors never go in
  the JSONL.
- RL training signals live in an optional `TrainingSignals` sidecar model, never on
  the base row, because they are not on disk.
- Adapters are the only boundary that knows upstream shapes; `schema`, `detectors`,
  `analysis`, and `report` depend only on the normalized types.

## Options Considered

### Option A: Versioned normalized schema (chosen)

| Dimension | Assessment |
|-----------|------------|
| Complexity | Medium (one schema package, adapters per source, migration chain) |
| Cost | One-time modeling cost, small per-source adapter cost |
| Scalability | High: streaming JSONL, per-row self-description, additive evolution |
| Team familiarity | High: plain pydantic v2 and JSONL, no upstream runtime needed |

**Pros:** single typed contract for every consumer; offline and dependency-light
(core never imports verifiers or prime-rl); unknown keys preserved; versioned rows
survive tool upgrades; content-derived IDs make v1 joins possible without
re-ingestion; JSON Schema export serves non-Python consumers.
**Cons:** a second representation to maintain; adapters must track upstream drift;
normalization can hide upstream quirks if adapters are sloppy.

### Option B: Analyze verifiers objects directly

Import verifiers and consume `GenerateOutputs` / `State` in memory.

| Dimension | Assessment |
|-----------|------------|
| Complexity | Low at first, unbounded later (coupled to upstream internals) |
| Cost | Free start, high maintenance (verifiers moves fast) |
| Scalability | Poor: in-memory objects, no streaming contract, no versioning |
| Team familiarity | Medium: requires tracking a fast-moving private surface |

**Pros:** zero modeling work; always exactly upstream's shape.
**Cons:** couples the whole tool to verifiers releases and pulls verifiers (and its
dependency tree) into an offline debugger; prime-rl artifacts would need a second
path anyway; no stable IDs, no version field, nothing for v1 to join on; contradicts
the offline, CPU-only requirement in spirit and the core-imports rule directly.

### Option C: Per-trainer formats, no normalization

Write detectors and reports twice, once per on-disk format, dispatching on source.

| Dimension | Assessment |
|-----------|------------|
| Complexity | High and multiplicative (every detector times every format) |
| Cost | Grows with each new source format |
| Scalability | Poor: adding a source touches every consumer |
| Team familiarity | High per format, low overall coherence |

**Pros:** no normalization layer; each path is maximally faithful to its source.
**Cons:** detector logic duplicated per format; inconsistent IDs and grouping
semantics across sources; no shared fixtures; v1 join would need per-format key
schemes; the maintenance surface is the product of formats and features.

## Trade-off Analysis

The real contest is A versus B. B optimizes for day-one fidelity but makes upstream
drift a runtime problem everywhere in the tool; A makes drift a compile-time problem
confined to one adapter package with pinned references. Because verifiers eval rows
and prime-rl train rows already share the trace shape on disk, one normalized union
covers both cheaply, which removes most of C's motivation. A's main risk, silent
data loss through an over-strict model, is neutralized by `extra="allow"` plus
round-trip tests on real-shaped fixtures. A's second risk, schema churn, is
neutralized by per-row versioning with an additive-only rule and a tested migration
chain.

## Consequences

- Easier: writing detectors and reports once against one type; testing with
  synthetic fixtures; running fully offline; joining v1 white-box data on
  `(run_id, rollout_id, step_index)`; supporting new sources by adding an adapter.
- Harder: adapters must be updated when upstream formats move (mitigated by pinned
  reference docs and the single-boundary rule); contributors must learn the
  normalized names rather than reusing upstream objects.
- Revisit: when verifiers ships a breaking `RolloutOutput` change, bump the schema
  major and add a migration; when v1 lands, the `TrainingSignals` sidecar and the
  join contract in `schema/ids.py` become live surfaces and deserve their own ADR.

## Action Items

1. [x] Freeze the schema (commit 5f87593) and record the freeze in PROGRESS.md.
2. [x] Enforce the boundary: core packages import only `rolloutscope.schema`.
3. [ ] Re-verify the pinned upstream references before the first post-0.1.0 release
       and refresh the verifiers-ground-truth pins if stale.
4. [ ] Write ADR-0002 when the v1 activation sidecar join is implemented.
