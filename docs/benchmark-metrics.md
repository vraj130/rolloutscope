# Benchmark metric definitions and reporting format

What a rolloutscope detector benchmark number means, exactly, and what has to be
pinned before one may be quoted. The implementation is
`src/rolloutscope/benchmark/`; this document is the contract it implements.

The rule that everything else follows from: **a detector that could not run has
not been measured.** Silence for lack of data is never a true negative. Get that
wrong and a detector whose inputs were entirely absent reports a perfect
false-positive rate and a full complement of true negatives, and the run reads as
a clean bill of health for a measurement that never happened.

## Unit of evaluation

A **unit** is the thing a detector reaches one verdict about. For every detector
scored on TRACE that is one rollout, so unit counts and row counts coincide.
Group and step detectors would use a group or a step as their unit; the report
names the unit per detector so the denominators are never ambiguous.

A unit is **positive** when the reference label says the trajectory contains a
reward hack, and **negative** when the label says it does not. A row whose label
is blank or unparseable is neither and is excluded from the population entirely,
at manifest build time, rather than being scored as a negative.

A detector **fired** on a unit when it returned at least one verdict with
`fired` true whose `rollout_ids` include that unit.

## Applicability, and the three outcomes that are not a confusion cell

Before any detector runs, each unit is classified for each detector:

- **scored**: the detector has the signals it needs. It lands in a confusion cell.
- **insufficient**: at least one required signal is absent. Counted under
  `units_insufficient`, with a stable reason code naming what was missing
  (`missing:reward`, `missing:metrics,reward`). Never a confusion cell.
- **error**: the detector raised. Counted under `units_error` with the exception
  text. Never a confusion cell, because a crashed detector is not a clean one.

Requirements per detector, read off each implementation and recorded in
`benchmark/applicability.py`:

| detector | requires |
|---|---|
| `verifier_tamper` | assistant text |
| `length_inflation` | assistant text, reward |
| `degenerate_repetition` | assistant text, reward |
| `format_only_wins` | reward, metrics |
| `answer_leakage_echo` | assistant text, reference answer or stated criteria |
| `reward_saturation_group_collapse` | reward, sibling groups |

"Requires reward" means a real reward that the source provided. A mapper that
writes a placeholder zero because the schema demands a float records
`reward_available: false` alongside it, and applicability reads the record, not
the value. This is what keeps `degenerate_repetition` (which gates on
`reward >= 0.7`) from reporting a flawless clean run on a dataset that has no
rewards at all.

A detector with no recorded requirements is treated as applicable, so a
third-party detector is scored rather than silently excluded. Adding an entry for
it is how it gets honest coverage accounting.

## The metrics

Over scored units only:

| metric | definition | undefined when |
|---|---|---|
| precision | tp / (tp + fp) | the detector never fired |
| recall | tp / (tp + fn) | no scored unit was positive |
| false-positive rate | fp / (fp + tn) | no scored unit was negative |
| fire rate, positives | tp / (tp + fn) | as recall |
| fire rate, negatives | fp / (fp + tn) | as false-positive rate |

Over the whole evaluated population:

| metric | definition |
|---|---|
| coverage | scored / (scored + insufficient + error) |

An undefined metric reports as **not measured** with its denominator shown. It
never reports as 0.0. `Rate.of(0, 0)` returns `value=None` for exactly this
reason, and the text renderer prints `not measured (0/N)`.

**Per-category recall** is recall restricted to positives carrying one taxonomy
code. TRACE is multi-label (39 percent of hacked rows carry several codes), so a
row labelled `1.1.1, 1.3.2` counts as a positive for both categories. The
per-category denominators therefore sum to more than the positive count, which is
correct and is why they are reported separately from the headline recall.

## Uncertainty

Every rate carries a **Wilson score interval at 95 percent**, computed from the
same numerator and denominator that produced it. Wilson rather than the normal
approximation because these samples are small and the rates sit near 0 and 1,
where the normal interval leaves the unit range and stops meaning anything. The
quantile is 1.959963984540054, a property of the normal distribution, not a
tuned constant.

The interval is not decoration. A category recall of 1.00 over 5 positives has a
Wilson interval of roughly [0.57, 1.00], and reporting the point estimate without
that range invites a claim the data cannot support.

## What must be pinned before a number is quoted

A benchmark number is only meaningful with the things that could move it. All of
these are recorded on the report and echoed in the text header:

- **dataset revision**: the Hugging Face repository commit SHA, not a branch.
- **manifest digest**: SHA-256 over the canonical manifest, so two reports are
  comparable exactly when this matches.
- **mapping version**: `TRACE_MAPPING_VERSION`. Any change to what a detector
  sees bumps it and invalidates earlier reports.
- **split version and weights**: `SPLIT_VERSION` plus the weight table.
- **split evaluated**.
- **source**: `parquet` (revision pinned) or `datasets-server` (current default
  branch, hash verified after the fact).
- **per-detector effective configuration**: every threshold that was active.
- **per-detector source fingerprint**: SHA-256 of the detector's module source.
  Detectors carry no version field, so the source hash stands in as one. It
  changes whenever the implementation changes, which is the property a
  reproducibility pin needs. It is not a semantic version and must not be
  compared for ordering.
- **verification result**: whether every manifest row in the split was fetched
  with an unchanged content hash.

## Splits

Three partitions, assigned per scenario and never per row, so sibling
trajectories of one task cannot straddle a boundary.

| split | weight | rows at revision `31d87f06` | purpose |
|---|---|---|---|
| train | 0.40 | 202 (107 positive, 95 negative) | development and inspection |
| tuning | 0.20 | 105 (50 positive, 55 negative) | threshold setting |
| holdout | 0.40 | 210 (111 positive, 99 negative) | reported accuracy, once |

The weights were chosen before any holdout measurement was taken, and they are
recorded so the choice is auditable rather than retrofitted. The holdout carries
the largest share because it is the only partition a reported accuracy number may
come from, and 40 percent of 517 rows keeps the Wilson half-width near 7 points
for a rate around 0.5. Tuning is deliberately the smallest: thresholds are set
from it, so overfitting it costs the least.

The scenario key is the first 16 hex characters of the SHA-256 of the normalized
opening user message (whitespace collapsed, lowercased). Assignment is a salted
hash of that key, so it depends only on the key: a partial fetch assigns exactly
as a full fetch would, and adding rows upstream never reshuffles existing ones.

**A measured caveat.** At revision `31d87f06` the 517 rows produce 516 distinct
scenario keys. Exactly one pair shares an opening message. Scenario grouping is
therefore close to a no-op on this artifact. It is kept because it is the correct
guard, because it does catch that pair, and because a future dataset with real
sibling clusters will need it. It catches exact and whitespace-equivalent
duplicates only; two paraphrases of one task still receive different keys.

### The holdout is sealed

The holdout has not been measured. It exists so that an accuracy claim can be
made once the Phase 2 detector repairs land, and measuring it against unrepaired
detectors would spend it for nothing. `scripts/trace_validation.py` defaults to
the tuning split, and asking for the holdout prints a warning that a threshold
fitted to it stops being a holdout number.

No release threshold has been set from this benchmark. When one is, it is a
documented product decision about review capacity (how many false positives a
reviewer can absorb per run), recorded with that rationale, and it is not
retrofitted to whatever the holdout happened to produce.

## Reporting format

`BenchmarkReport` serializes as JSON with sorted keys and a trailing newline, so
two runs of the same configuration produce identical bytes. Alongside the pins
and the per-detector metrics it carries a bounded set of false positives and
false negatives, each with its unit id, its taxonomy codes, and an evidence
excerpt, for human review. The bound is a display limit: the complete per-unit
outcomes stay available from `benchmark.score_detector`.

The text renderer prints the pins first, then one block per detector. A detector
with zero scored units prints its coverage and the reasons, and the line "no
scored units, so no accuracy is reported for this detector", instead of a row of
zeros that would read as a measurement.

## Reproducing a run

```bash
uv sync --extra dev --extra benchmark
# HF_TOKEN in .env or the environment; TRACE is gated.
uv run --extra benchmark python scripts/build_trace_manifest.py   # only to re-pin
uv run --extra benchmark python scripts/trace_validation.py --split tuning
uv run --extra benchmark pytest tests/integration -m integration -q
```

The script and the integration test call the same driver against the same
manifest. `test_script_and_test_produce_the_same_report` runs the script as a
subprocess and compares its JSON against the report built in process, which is
the direct check that the two paths cannot diverge again.
