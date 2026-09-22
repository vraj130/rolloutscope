# Phase 1 data contract

Status: implemented for normalized schema `2.0` and the legacy verifiers and
prime-rl adapters. This document supersedes the schema `1.0` identity rules in
ADR-0001 for new ingestion. Existing identifiers remain readable.

## Version policy

Every normalized row contains `schema_version`. A version is a string with a
major and minor component, optionally a patch component. Supported older majors
migrate before validation: the illustrative `0.x` format renames `episode_id`
and `score`, then `1.x` migrates to `2.0`. The `1.x` migration preserves all
existing IDs, step indices, and unknown fields. It additionally records existing
run, rollout, and group IDs in `identity_aliases` under `v1.*` keys.

Schema 2 is a major change because values such as infinity that schema 1 allowed
cannot reliably survive JSON serialization and analysis. Schema 2 rejects
non-finite numerical values in analysis fields and nested provider extras. It
also rejects integers outside the JSON writer's signed/unsigned 64-bit range.
Permissive upstream field preservation remains in place.

Readers accept additive versions within major 2 and preserve unknown fields.
Writers emit version `2.0`. Future major versions and malformed version markers
are rejected with `unsupported_schema` diagnostics. Direct model constructors
and the discriminated union reject versions requiring migration; callers use
`validate_rollout` or a reader to migrate.

## Identity and content

| Field | Meaning and stability |
| --- | --- |
| `run_id` | For raw legacy artifacts, a hash of the resolved absolute run location. For normalized artifacts, the existing value is preserved. |
| `environment_namespace` | The environment scope used to interpret task and group identities. It is preserved from normalized rows, or taken from explicit run metadata as described below. |
| `task_namespace` | The task-family or dataset scope within the environment. It is preserved from normalized rows, taken only from explicit run metadata, and otherwise falls back to the environment namespace. |
| `rollout_id` | The legacy hash of example, prompt, completion, and reward. Retained for compatibility; it can change when raw input is rescored and can coincide for identical generations. |
| `group_id` | The existing normalized group ID, or the legacy `grp-{example_id}` value for raw records. Group identity is scoped by run and step, never globally by this string alone. |
| `occurrence_id` | The primary new join key: a hash of run ID, relative original source path, and one-based physical line. Different lines with identical content receive different occurrence IDs. |
| `content_fingerprint` | A hash of validated prompt, completion, tool definitions, and trajectory content. It excludes top-level reward and metrics, trajectory reward and advantage, and opaque trajectory IDs. |
| `scoring_revision` | A hash of top-level reward and metrics and trajectory reward/advantage values. Recomputed after rescoring without changing occurrence identity. |
| `step_index` | Taken from recognized training directory names for raw data, or preserved from normalized data. Unrecognized layout gives `None`. |
| `identity_aliases` | Historical identifiers retained during migration. They support explicit legacy joins without treating a content hash as an occurrence ID. |

Appending source rows preserves earlier occurrence IDs. Rescoring the same row
at the same source location preserves its occurrence and content fingerprint
while changing its scoring revision. Changing generated content changes its
content fingerprint. Reordering or inserting physical source lines changes
position-based occurrence IDs; preserve a normalized export when the original
source layout must be rewritten.

The raw run-location rule separates runs with identical basenames and remains
stable when metadata cost summaries or other mutable metadata change. For an
eval `results.jsonl`, the run location is its containing directory. For a
standalone JSONL file, it is the file itself. For training files under a
recognized step directory, it is the step directory's parent. Direct file,
step-directory, and run-directory entry points therefore agree for those
training layouts.

Raw relocation to a different absolute path intentionally produces a different
fallback run ID. A normalized export retains run and occurrence identity when
copied to another machine or renamed. No persistent manifest is written into
the input directory. Callers that need raw relocation without normalized
exports pass the original run ID with the CLI `--run-id` option or carry it
explicitly into the discovered manifest API.

## Provenance and upstream identity

Every adapter-emitted row includes `provenance`:

- `source_path`: path relative to the original discovered run root.
- `line`: one-based physical source line, including blank and rejected lines in
  the count.
- `adapter` and `adapter_version`: the reader responsible for normalization.
- `namespace`: the run namespace used by the fallback identity scheme.
- `legacy_ids`: migration aliases where available.

Environment and task namespace are separate typed row fields. They do not
replace `run_id`, and `provenance.namespace` continues to record the run
namespace used by the source-occurrence fallback. For a raw verifiers eval run,
the adapter selects `environment_namespace` from a nonblank explicit
`metadata.json.environment_namespace`, then from the pinned
`GenerateMetadata.env_id` field. It selects `task_namespace` only from a
nonblank explicit `metadata.json.task_namespace`. The same optional convention
is supported for a prime-rl run root even though the pinned training layout does
not guarantee a run metadata file. Missing environment namespace falls back to
the run ID during identity attachment; missing task namespace falls back to the
resolved environment namespace.

The extraction deliberately ignores cost, reward summaries, model settings,
arbitrary `info` keys, and per-row `task` objects. Those values remain preserved
as upstream extras when present, but they are not stable run-level namespaces.
Whitespace-only or non-string namespace metadata is treated as absent. This
keeps mutable summaries from changing identity and avoids guessing task scope
from provider-specific payloads.

Normalized reads preserve existing provenance. Legacy normalized rows missing
IDs or provenance receive deterministic fallback values from the normalized
file's own location; the original pre-export location cannot be reconstructed
when it was never recorded.

The legacy on-disk contracts expose example identity and, for multi-turn data,
`trajectory_id`. These values are retained. Provider tool-call IDs, tool names,
arguments, responses, environment metadata, and task metadata survive through
the permissive row/message/step fields when actually present. Phase 1 does not
invent upstream episode, trace, or tool-action identifiers and does not map the
newer Episode/Trace format. A run namespace provides a conservative default;
cross-environment comparisons must not assume matching example IDs establish
task identity.

All permissive provider values are recursively checked against the supported
JSON value domain after Pydantic validation. Non-finite floats, integers outside
the writer's 64-bit range, non-string object keys, bytes, sets, and arbitrary
Python objects are rejected rather than being changed to null or failing during
artifact serialization.

## Format detection and mixed files

Adapter resolution probes past blank lines, invalid JSON, non-object values,
and unrecognizable leading objects. An explicit `schema_version` marker selects
the normalized adapter even when the version will be rejected. A recognizable
raw row selects the appropriate legacy adapter.

The subsequent load pass accounts for every physical input line. It dispatches
each row separately: schema-marked rows migrate and preserve identity, while
raw rows receive identity from their source location. Mixed files are accepted
under this explicit policy and record format `mixed`; normalized IDs are never
replaced with raw-layout IDs. Probe reads do not enter ingestion counts.

Discovery produces a `RunManifest` containing ordered source files, the run
root, format, adapter version, parsed metadata, metadata source paths, and the
optional environment/task namespaces extracted under the rules above.
`load_manifest` consumes exactly those discovered files, so callers can protect
all inputs before creating output artifacts without rediscovering the run.

## Ingestion accounting

For a fully read source:

`observed = accepted + rejected + blank`

Observed counts physical lines. Accepted counts validated rows, including
repeated occurrences. Rejected counts parsing, version, and validation failures.
The raw file byte length and SHA-256 are calculated in the same streaming read.

Stable rejection reason codes are `invalid_json`, `not_object`, `missing_field`,
`unsupported_schema`, and `validation_error`. Diagnostics retain no more than
five representative examples per source, each bounded in length and located by
line. Input rows remain available in the source rather than being copied into
the diagnostic ledger.

Adapter loading retains duplicate normalized occurrences and counts repeated
`occurrence_id` values across the discovered files using a temporary on-disk
SQLite index with a bounded page cache. Repeated raw generations at different
physical lines are distinct occurrences and do not increment that duplicate
counter. No deduplication policy silently removes evidence.

File and run statuses distinguish `complete`, `partial`, `empty`, `unsupported`,
and `failed`. A source with no accepted rows and at least one rejection is
`unsupported`; a blank or zero-byte source is `empty`. Read failures are recorded
and raised. Files not reached after a fatal error remain explicitly unread in
the diagnostic ledger. A digest from a failed read covers only the bytes
actually consumed and must not be treated as a complete source hash.

## Output and round-trip guarantees

`write_rollouts` streams serialized rows to an output transaction, validates
them again, and emits the current schema version. Callers pass all discovered
source paths through `source_paths` to reject source/output aliases before
writing. A failed iterator or serialization preserves an existing destination;
successful serialization is atomically installed. An existing output symlink
is replaced as a directory entry instead of being followed to overwrite its
target.

For an identity-bearing normalized row, unchanged write/read cycles preserve
run ID, legacy IDs, occurrence ID, step, provenance, content fingerprint, and
scoring revision. Reading a legacy row without identity enriches it once with
deterministic source-based identity. Subsequent normalized exports preserve
that enrichment.

Conversion and adapter loading stream records. The current analysis pipeline
materializes normalized rollouts; this contract does not claim that complete
analysis supports datasets larger than available RAM.

Verdicts carry typed `mode`, `unit`, optional `run_id`, and
`source_occurrences` fields. Each source occurrence contains an occurrence ID,
relative source path, and one-based physical line. Measurements and execution
configuration use the same recursively JSON-safe value contract, so the JSONL
artifact cannot silently turn a non-finite measurement into null. Findings
carry typed mode and unit fields while retaining the complete flagged
occurrence list independently from bounded display exemplars. Execution
manifests record the distinct environment and task namespaces observed in
accepted rows.

## Validation

The offline regressions cover training conversion round trips, repeated
normalized reads, mixed-format orderings, legacy migration, malformed/future
versions, exact line counts and source hashes, same-basename run separation,
append/rescore identity, duplicate retention across files, numerical validation,
constructor version bypasses, typed verdict provenance, JSON-safe detector
measurements, and transactional writer failure/collision.

Run the relevant checks with:

```sh
uv run --offline pytest tests/test_schema tests/test_adapters -q
uv run --offline ruff check src/rolloutscope/schema src/rolloutscope/adapters tests/test_schema tests/test_adapters
uv run --offline mypy src/rolloutscope/schema src/rolloutscope/adapters
```

Legacy upstream references remain pinned to verifiers
`5885ab9c54152e707af2a11797aa52c3eb1752da` and prime-rl
`df2acf4874af8be0300e06a4f45e65a78c305229`. Current upstream compatibility is a
separate Phase 2 workstream.
