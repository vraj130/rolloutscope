# Phase 1 implementation evidence

Phase 1 makes execution completeness, detector coverage, identity, and output provenance
visible alongside heuristic findings. It does not establish detector accuracy; signal
attribution and accuracy calibration remain Phase 2 work.

## Delivered contracts

### Safe inputs and outputs

`analyze` and `convert` resolve an adapter manifest once and preflight every destination
against all discovered rollout and metadata sources. Direct, relative, symlink, and
hard-link aliases are rejected before an output is opened. All requested report bytes are
serialized into temporary files beside their destinations. Analysis, iteration, or
serialization failure removes those temporary files and leaves existing destinations
unchanged.

HTML or JSON output creates a deterministic verdict JSONL sidecar by default. An explicit
`--verdicts PATH` works with or without a human report. The primary output also determines
the external `.manifest.json` ledger path. The ledger records the SHA-256 and byte size of
the HTML, JSON, and verdict artifacts and is committed last. It excludes its own digest to
avoid a circular checksum.

The filesystem cannot atomically replace several independent files as one operation. A
filesystem failure during the final replacement sequence can therefore leave some new
artifacts in place. The ledger-last rule ensures such a partial set is never certified by
a new ledger. Analysis and serialization failures occur before this replacement sequence
and preserve the full previous set.

### Versioned identity and normalized input

Schema 2.0 separates three concepts:

- `occurrence_id` identifies one physical source record using run identity, relative
  source path, and line.
- `content_fingerprint` identifies prompt, completion, and trajectory content while
  excluding mutable reward and scoring values.
- `scoring_revision` identifies the reward, metric, and trajectory-scoring state, so
  rescoring changes this value without creating a different occurrence.

Every occurrence carries source provenance plus environment and task namespaces. Explicit
metadata namespaces are preserved; deterministic fallbacks use environment, then run
identity. Legacy v1 `rollout_id` and `group_id` values remain available as migration
aliases. A legacy raw run receives a path-derived run ID unless `--run-id NAME` supplies a
stable external identity. Normalized data preserves its recorded run ID and rejects that
override.

The normalized adapter routes rows with a schema marker through migration before
validation, preserves existing identity and step ordering, and rejects unsupported future
major versions. Mixed files are routed per row and labeled `mixed` in ingestion
accounting. Repeated normalized reads are idempotent.

### Ingestion and execution states

Each rollout source records its selected format and adapter version, streamed content
hash, byte size, observed lines, accepted records, rejected records, blank lines,
retained duplicate occurrences, stable rejection reason counts, and up to five bounded
source-located examples. Run-level counts are computed from these file records.

The public state meanings are:

| state | meaning | analysis exit behavior |
|---|---|---|
| `complete` | every observed nonblank record was accepted and selected detectors ran | finding policy decides |
| `partial` | at least one record was accepted and at least one was rejected | allowed by default; exit 2 when partial policy is `reject` |
| `empty` | no accepted or rejected records were present | exit 2 |
| `unsupported` | records were observed but none were accepted | exit 2 |
| `failed` | source IO, selected detector discovery, coverage support, or execution failed | exit 2 |

`--fail-on` only evaluates fired finding severity after successful analysis. It cannot
turn an execution failure into success. Analysis failures do not replace requested report
artifacts. Terminal output distinguishes unavailable analysis from a completed run with no
findings.

### Detector coverage and complete evidence

Every selected detector produces an execution record even when it emits no verdict.
Coverage is partitioned into candidate, eligible, fired, eligible-clean,
insufficient-data, skipped, and error counts. The invariant is:

```text
candidate = fired + clean + insufficient_data + skipped + errors
eligible = fired + clean
```

Coverage rows remain separate by unit, mode, and run; group and run denominators are never
summed into one rate. Each record retains effective detector configuration, implementation
version, reason counts, and underlying measurements. A third-party plugin without the
coverage contract is `unsupported`, not clean.

HTML and terminal views cap large tables, finding lists, diagnostics, and evidence text
and display the shown and total counts. Multi-run step series are kept in the table and
are not connected into a misleading single chart. Deterministic JSON retains the complete
report. Verdict JSONL retains every verdict, its typed unit/mode/run fields, every evidence
span and referenced ID, and validated source occurrence mappings with source path and
line.

### Reproducibility record

The execution manifest records tool and schema versions, source revision when available,
a digest of the installed implementation and template bytes, adapter and input format,
complete effective configuration, detector selection and versions, actual run,
environment, and task namespaces, ingestion state, and artifact references. Source JSONL
hashes come from the ingestion stream. Metadata is hashed immediately after manifest
discovery and recorded as a separate input role. The external artifact ledger holds the
final output hashes because embedding an artifact's own digest in that artifact would be
self-referential.

## Acceptance evidence

The regression suite demonstrates the Phase 1 exit criteria as follows:

- `tests/test_cli/test_e2e.py` covers direct, relative, symlink, and metadata collisions;
  partial allow/reject behavior; zero usable input; stable run overrides; failed detector
  discovery and execution; output preservation after serialization failure; standalone
  verdict output; real artifact digests; source and metadata hashes; and config rejection
  before input discovery.
- `tests/test_schema/test_phase1_contract.py` covers finite JSON-safe normalized values,
  migration serialization, writer failure cleanup, source collision rejection, and
  occurrence/content/scoring identity behavior.
- `tests/test_adapters/test_normalized.py` is the converted-run equivalence diff. For a
  synthetic prime-rl run it asserts exact rollout equality after conversion and repeated
  reads, identical run and occurrence identities, group and step membership, equal
  aggregates, and equal execution output for every built-in detector.
- `tests/test_report/test_html.py` and `tests/test_report/test_terminal.py` cover bounded
  human views, visible omission totals, escaped evidence, multi-run chart separation, and
  unit/mode/run-specific denominators. `tests/test_report/test_json_out.py` covers complete
  deterministic machine output.

The completed offline exit run on 2026-09-09 was:

```bash
env PYTHON_DOTENV_DISABLED=1 uv run --offline pytest -q
# 458 passed, 7 deselected

uv run --offline ruff check .
# All checks passed!

uv run --offline ruff format --check .
# 103 files already formatted

uv run --offline mypy src
# Success: no issues found in 44 source files
```

The seven deselected tests are optional integration tests; this evidence does not claim a
network-backed upstream or TRACE run. The focused CLI/report suite contained 68 passing
tests in the same checkout.

## Remaining limits

Conversion and ingestion stream JSONL, but analysis materializes normalized rows because
whole-run and group detectors need shared access. Peak analysis memory therefore scales
with accepted input size. Current upstream layouts beyond the pinned legacy contracts are
tracked in the compatibility work and are not implied to be supported by this phase.
Detector findings remain heuristic observations; known hard negatives and frozen accuracy
evaluation belong to Phase 2.
