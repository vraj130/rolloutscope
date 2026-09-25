> **Historical record.** This is the repository review from 2026-09-08 that produced
> the delivery plan. Its Phase 1 findings were fixed. Its product-direction section
> is superseded by [PLAN.md](../../PLAN.md).

**RolloutScope repository review, 8 September 2026**

Review baseline: commit `c663fef`, plus the existing working tree. Three review agents covered detectors, data contracts/adapters, and engineering quality; the primary review covered CLI, analysis, reporting, end-to-end behavior, and product direction. Application source was not changed. Existing local changes and generated files were preserved.

**Assessment**

RolloutScope is a well-structured early prototype with a working offline analysis pipeline. It is suitable for demonstrations, exploratory inspection, and detector development. Its present evidence does not establish dependable reward-hack detection on users' training runs. The most valuable next phase is to make ingestion trustworthy, detector coverage explicit, evidence complete, and findings useful in a real investigation.

The strongest product direction is an offline reward-integrity debugger for RL engineers: identify when reward and task quality diverge, isolate the affected tasks and steps, compare suspicious trajectories with relevant siblings, and help the user verify a repair on a subsequent run. Existing package boundaries are a useful foundation for that direction.

**Verified engineering baseline**

| Area | Observed state |
|---|---|
| Packaging | Python package version 0.1.0; Python 3.11 to 3.13 declared; uv lockfile; five runtime dependencies |
| Data layer | Pydantic discriminated single/multi-turn schema, unknown-field preservation, JSON Schema export, migration helpers, IDs, JSONL reader/writer |
| Adapters | Legacy verifiers eval and prime-rl training layouts |
| Detectors | Six configurable detectors, entry-point registry, structured verdicts/evidence |
| Analysis | Reward moments/histogram, group statistics, step summaries, reward-extreme snippets |
| Outputs | Rich terminal, deterministic JSON summary, self-contained HTML with SVG charts |
| CLI | analyze, convert, detectors list, schema export, configurable finding exit threshold |
| Offline tests | 173 passed, one network integration test deselected; Python 3.12.13 |
| Coverage | 93% reported with branch coverage enabled |
| Static checks | Ruff lint and format check passed; strict mypy passed across 30 source files |
| Smoke runs | Demo: 10 rows, four detector findings. Training fixture: five rows, two step summaries, no findings. HTML/JSON generation succeeded. |

These results verify current code behavior. They do not establish detector accuracy or fresh compatibility with every declared Python/upstream version. The network-gated TRACE evaluation was not rerun. Browser automation was unavailable, so HTML was inspected through its template, generated output, and tests, without visual browser QA. A fresh wheel installation was not exercised in this review.

**Highest-priority findings**

1. **Output collisions can destroy input data.** `convert` passes a lazy reader into `write_rollouts`, which opens the output with `wb` before consuming the reader. A disposable 5,670-byte fixture converted onto itself became an empty file, exited 0, and printed that it wrote zero rows. Reject output paths that alias any source, including symlinks; use temporary output plus atomic replacement for successful writes. Apply collision checks to report outputs as well. Evidence: [CLI conversion](../../src/rolloutscope/cli.py#L271), [writer](../../src/rolloutscope/schema/io.py#L61).

2. **Normalized output does not round-trip through the CLI analysis path.** Every ordinary `.jsonl` file is routed through the eval adapter, which overwrites stored run/group/rollout identity and `step_index`. Converting the training fixture and analyzing that output changes steps `[0,0,0,1,1]` into five `None` values; the two-step chart disappears. Add a normalized-input adapter that uses the migration-aware reader and preserves identity. This is a prerequisite for trustworthy exports, comparisons, and future sidecar joins. Evidence: [normalization](../../src/rolloutscope/adapters/base.py#L171), [eval detection](../../src/rolloutscope/adapters/verifiers_eval.py#L33), [normalized reader](../../src/rolloutscope/schema/io.py#L40).

3. **Current upstream source has moved beyond the implemented training format.** The inspected prime-rl `main` file monitor writes `rollouts/step_{step}/{kind}/{subset}/traces.jsonl` containing episode records. The current adapter recognizes `train_rollouts.jsonl` at the root or one directory level below it. Directly passing a new episode file also does not solve the shape mismatch: the normalizer requires top-level `example_id` and `reward`. Support explicit upstream format versions and add captured writer-output fixtures. This finding concerns inspected `main`; a release-by-release compatibility matrix remains to be established. Evidence: [local discovery](../../src/rolloutscope/adapters/prime_rl_train.py#L42), [upstream file monitor](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/src/prime_rl/monitors/file.py), [upstream episode model](https://raw.githubusercontent.com/PrimeIntellect-ai/verifiers/main/verifiers/v1/episode.py).

4. **An entirely unusable input can pass the advertised CI gate.** A file with one invalid JSON line and one schema-invalid object emits warnings, analyzes zero rollouts, prints “No findings,” and exits 0 even with `--fail-on info`. Retain per-file counts for rows seen, accepted, skipped, and why; expose analysis completeness in every output. No usable rows should produce an analysis-error exit, independently of finding severity. Partial ingestion should have a configurable acceptance policy. Evidence: [pipeline](../../src/rolloutscope/cli.py#L93), [exit policy](../../src/rolloutscope/cli.py#L152), [skip handling](../../src/rolloutscope/adapters/base.py#L195).

5. **The external detection baseline is weak and its two implementations disagree.** The README's reported tampering counts, 85/153 hacked and 73/147 clean, imply 55.6% hacked-row recall, 49.7% clean-row false-positive rate, and 53.8% precision on that sample. These are calculations from committed results, not newly measured performance. TRACE spans multiple hack categories, so this is also not a category-specific tampering evaluation. The integration test scans raw conversation text, including user context, while the script parses assistant messages and remaps tool calls; they sample different numbers of rows. Share one mapping and metric implementation, pin a dataset revision and sample manifest, and establish held-out evaluation. Evidence: [reported counts](../../README.md#L64), [integration representation](../../tests/integration/test_trace_dataset.py#L166), [script mapping](../../scripts/trace_validation.py#L155).

6. **Findings discard the information needed to investigate all flagged cases.** Assembly keeps at most three evidence spans per detector/category, removes the complete flagged ID list, and drops verdict extras such as snapshot/trend mode. The JSON output serializes this summary, so it cannot recover the discarded details. A ten-verdict probe retained a flagged count of ten but only three spans and no full ID list. Persist complete verdicts in a streamed JSONL artifact or indexed store; retain a bounded HTML summary that links to full local context. Evidence: [finding assembly](../../src/rolloutscope/analysis/findings.py#L56), [report contract](../../src/rolloutscope/report/model.py#L32).

7. **Reported firing rates lack meaningful denominators.** Several detectors emit only positive verdicts. On the ten-row demo, tampering produces two positive verdicts and the report says 2/2 checks fired; it does not communicate two flagged rows out of ten or the number actually eligible. A detector returning nothing also disappears even with clean reporting requested. Introduce an execution summary for every selected detector: total rows, eligible rows/groups, flagged rows/groups, insufficient-data counts, skipped reasons, and errors. Keep row and group denominators separate. Evidence: [pipeline collection](../../src/rolloutscope/cli.py#L127), [rate calculation](../../src/rolloutscope/analysis/findings.py#L124).

8. **The full analysis is not bounded by available RAM.** `build_report_data` materializes all normalized rows; detectors make additional lists, groupings, and text representations. Aggregation is incremental, but its state grows with distinct groups and steps. The HTML renders every group. The README's larger-than-RAM claim is unsupported. Correct the claim immediately, then add incremental row detectors, cohort processing, disk-backed spill, and bounded report selection. Benchmark both many long trajectories and high group cardinality. Evidence: [materialization](../../src/rolloutscope/cli.py#L115), [aggregation](../../src/rolloutscope/analysis/aggregates.py#L232), [all-group rendering](../../src/rolloutscope/report/templates/report.html.j2#L139), [claim](../../README.md#L22).

**Detector-specific changes**

| Detector | Reproduced issue or important limitation | Recommended change |
|---|---|---|
| verifier_tamper | Tool output describing preexisting skips/assertions fires at score 0.75, although the assistant never inserted them | Preserve roles and actions. Distinguish observed code, proposed edit, confirmed modification, and verifier effect. Compare file diffs and sibling behavior. |
| answer_leakage_echo | Correctly answering an answer-only question with `photosynthesis` receives score 0.8 without evidence that the reference was exposed | Make reference-answer matching an observation; require a visible source and subsequent use before calling it leakage. Respect task output requirements. |
| length_inflation | Three steps with rising reward/length and unchanged 50% binary accuracy produce no finding; replacing each binary outcome with its step mean produces findings | Compare named quality metrics aggregated by step/cohort. Do not pool different metrics or require every raw outcome to be near-identical. Account for task difficulty when interpreting correlations. |
| degenerate_repetition | The same repeated text fires as single-turn but is skipped entirely as multi-turn | Analyze each assistant generation separately; add a distinct repeated-tool-action sequence signal if evaluation supports it. Report coverage. |
| reward_saturation_group_collapse | `min_groups=100` does not prevent a trend based on two groups per step | Enforce sample-size settings consistently. Present uniform rewards as a training-health observation; corroborate with quality before attributing hacking. |
| format_only_wins | Format and correctness keys occurring in separate rows can produce no eligibility diagnostic despite no jointly inspectable row | Count eligible rows with both signals and expose missingness. Use explicit metric roles and scales. |

Source entry points: [text flattening](../../src/rolloutscope/detectors/_text.py#L59), [tampering](../../src/rolloutscope/detectors/verifier_tamper.py#L52), [answer match](../../src/rolloutscope/detectors/answer_leakage_echo.py#L120), [length gate](../../src/rolloutscope/detectors/length_inflation.py#L135), [repetition skip](../../src/rolloutscope/detectors/degenerate_repetition.py#L44), [saturation trend](../../src/rolloutscope/detectors/reward_saturation_group_collapse.py#L161), [format eligibility](../../src/rolloutscope/detectors/format_only_wins.py#L57).

The scores currently mix fixed weights, reward values, correlations, and fractions. A shared threshold of 0.8 does not make those measurements comparable confidence estimates. Expose the underlying measurements and separate suspicion, likely impact, and evidence strength. Observational signals should use language that matches what was actually measured.

**Schema, configuration, and reproducibility**

- Separate a rollout occurrence ID from a content fingerprint. The current fingerprint includes reward and omits trajectory; rescoring changes identity, while different trajectories can share identity. Repeated identical generations also need distinct occurrence identity when joining per-generation sidecars. Group identity should include the task/environment namespace, and sampling cohorts should preserve step/group membership. Run IDs derived from directory basenames are not globally unique; hashing the full eval manifest also makes a run ID change when a summary such as average reward changes. Preserve old IDs through explicit versioned migration rather than silently changing their meaning. See [identity definitions](../../src/rolloutscope/schema/ids.py#L40).
- Apply schema-version checks consistently. The normalized reader invokes migrations; the CLI adapter route directly validates a string `schema_version` without the same migration/version policy. Centralize those rules at normalized ingestion. See [reader](../../src/rolloutscope/schema/io.py#L40) and [normalizer](../../src/rolloutscope/adapters/base.py#L192).
- Reject unknown configuration keys. `criticall_at=0.99` and `min_correlaton=0.99` were silently ignored, leaving both intended settings at 0.8. Raw upstream models should remain permissive; user config should have strict key validation. Validate regexes and finite numeric values before running analysis. See [RunConfig](../../src/rolloutscope/cli.py#L47).
- Enforce finite measurements at ingestion. Pydantic accepts a reward encoded as the JSON string `"NaN"`; histogram aggregation then raises an error while converting NaN to an integer. The writer serializes the accepted reward as `null`, which the reader subsequently rejects. Every accepted normalized record should remain analyzable and serializable under its declared contract. See [reward field](../../src/rolloutscope/schema/models.py#L132) and [histogram indexing](../../src/rolloutscope/analysis/aggregates.py#L271).
- Record the complete effective config and execution manifest independently of findings. Today non-firing detector settings vanish; severity settings have no structured manifest field; adapter identity/version, metadata hash, selected detectors, and ingestion statistics are absent. Input names use only basenames, so both training files appear as `train_rollouts.jsonl`. Include relative paths and the precise dataset snapshot examined. See [ReportData](../../src/rolloutscope/report/model.py#L32) and [input description](../../src/rolloutscope/report/model.py#L79).
- Preserve the upstream sampling population. The new episode monitor distinguishes `all` from `effective` cohorts; recursive discovery must not blindly combine them and double-count episodes or obscure filtering effects. Make cohort selection explicit in the adapter and report.

**Validation that would support user trust**

The synthetic fixture metrics are file-level checks: a detector succeeds if it fires anywhere in its one hacked file and nowhere in six deliberately clean files. Their perfect precision/recall should be labeled fixture separation, with the tiny sample units made explicit. See [metric implementation](../../tests/test_detectors/test_fixture_metrics.py#L35).

The README calls TRACE trajectories real. The primary dataset card instead describes synthetically generated, human-verified trajectories with simulated tool use. It is a useful external benchmark, but it does not establish performance on operational RL logs. Correct that distinction. Sources: [TRACE paper](https://arxiv.org/abs/2601.20103), [dataset card](https://huggingface.co/datasets/PatronusAI/trace-dataset).

A credible evaluation program should include captured, versioned upstream outputs; hard negatives such as legitimate test maintenance, quoted malicious code, and answer-only tasks; actual partner-run cases; labels for the specific hack category and evidence location; and splits by task/run so sibling trajectories cannot leak between tuning and evaluation. Report precision, recall, false-positive rate, applicability, sample counts, and uncertainty separately by detector and workload. Missing answer or correctness data should appear as unavailable coverage, not a successful negative prediction.

The existing high code coverage is worth keeping. Add focused regressions for the demonstrated bugs and contracts, current-format integration fixtures, performance budgets, and an installed-wheel CLI/report smoke test. A coverage percentage alone should not serve as the release-quality criterion. CI currently has a useful Linux/Python matrix but no wheel-install smoke or enforced coverage floor. Release automation and dependency maintenance can follow the trust repairs.

Complete the public onboarding path with clone/install instructions, a linked example report, versioned input-format examples, metric prerequisites, and troubleshooting for skipped rows. Ship the compatibility contract in public docs instead of relying on ignored local skill references. Keep the completed PLAN/PROGRESS files as historical build records; new milestones should measure input support, detection usefulness, and recurring user outcomes.

**Product direction and workflow**

The first intended user should be an RL engineer running verifiable or tool-using tasks who sees reward improve but doubts that task performance improved. That is a proposed initial audience, not a validated customer profile. The review found no customer evidence sufficient to select a broader audience.

The core workflow should answer five questions:

1. Was this run ingested completely, and which checks were actually possible?
2. Where did reward, correctness, length, truncation, or group health change?
3. Which tasks and trajectories explain that change?
4. What did the assistant actually do, compared with relevant benign siblings?
5. After a rubric or environment repair, did the issue decline without harming legitimate performance?

Build a ranked investigation queue with clear reasons, full transcript context, role/turn/source locations, per-metric values, step filters, and sibling comparisons. Add reviewer outcomes such as confirmed issue, benign, and needs more context, with notes and a versioned export. The same labels can inform future evaluation. A baseline comparison command should align compatible tasks and environments, retain unmatched populations, show uncertainty, and link changes back to examples.

General charts and transcript viewing already exist elsewhere: Prime's dashboard documents reward curves, rubric scores, distributions, and individual rollouts, and Inspect's viewer supports scoring context, filtering, and sorting. My inference is that RolloutScope's strongest differentiation is independent, evidence-backed reward-integrity investigation that fits an engineer's existing workflow. Sources: [Prime monitoring](https://docs.primeintellect.ai/hosted-training/end-to-end-run), [Inspect log viewer](https://inspect.aisi.org.uk/log-viewer.html).

Keep offline analysis, small dependencies, pure detector logic, a normalized contract, safe escaped rendering, and portable report export. Extend the application with a local indexed store and an optional richer viewer when the investigation workflow requires them. Store raw evidence on disk and keep summaries bounded; SQLite is a reasonable first candidate to evaluate because it is available in the Python standard library. Choose storage from measured access patterns and performance requirements.

**Proposed delivery sequence**

These are suggested release slices and acceptance criteria, not committed delivery dates or achieved accuracy targets.

| Slice | Concrete work | Acceptance gate |
|---|---|---|
| First: trustworthy analysis | Output collision protection; normalized input route; version enforcement; strict config; ingestion diagnostics; complete verdict export; truthful coverage/rates; documentation corrections | In-place conversion cannot truncate a source; convert/analyze preserves identity and steps; all-invalid input cannot pass CI; every selected detector reports eligibility; every flagged ID remains retrievable |
| Second: reliable signals | Role/action provenance; length metric aggregation; per-turn repetition; exposure-aware leakage; consistent group gates; shared TRACE mapping; current upstream adapter with versioned fixtures | Reproduced hard negatives stay benign; demonstrated missed cases are covered; supported upstream outputs load without unexplained loss; accuracy and applicability are measured on a frozen holdout |
| Third: useful investigation | Ranked queue, full context, task/step filtering, sibling comparison, reviewer labels, run-to-run comparison | An external engineer can trace a finding to its source and verify a repair without manually searching JSONL |
| Fourth: prove repeat value | Pilot on several independent teams' runs; measure triage time, confirmed useful findings, repeated use, and false alarms; bounded-memory pipeline and installed-package checks | Partners use it for successive runs; measured analysis/report budgets hold on representative large inputs; quality targets are chosen from real alert volume and review capacity |
| Later: expand based on evidence | Additional adapters, incremental training hooks, richer semantic checks, optional white-box sidecars | Existing workflow has demonstrated demand; joins and evaluation are stable; each extension solves a measured failure case |

The first implementation batch should cover output protection, normalized round-trip fidelity, and ingestion/eligibility reporting. These changes make subsequent detector experiments and user pilots interpretable. In parallel, assemble a small set of realistic clean/hacked cases and upstream writer captures before broadening the detector catalog.

White-box analysis remains a plausible research direction once the occurrence identity and provenance contracts are dependable. The current placeholder sidecar model is preparation for that work; it does not yet implement activation capture, storage, analysis, or joins.
