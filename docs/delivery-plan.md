# RolloutScope delivery plan

This plan turns the repository review into four sequential delivery phases. Each phase produces a usable release slice, has explicit dependencies, and ends with evidence that can be reviewed before the next phase begins.

The target product is an offline reward-integrity debugger for RL engineers. It should explain whether a run was analyzed completely, where reward and task quality diverged, which trajectories contributed to the change, what the model actually did, and whether a later environment or rubric change resolved the issue.

The phases are intentionally ordered:

1. Make every analysis trustworthy.
2. Make the signals reliable enough to prioritize investigation.
3. Make the investigation workflow useful from detection through verification.
4. Prove that the product provides repeat value on independent, representative runs.

Later phases depend on the contracts established earlier. Phase 2 results are not credible without Phase 1 ingestion and coverage accounting. Phase 3 cannot provide dependable comparisons without stable identities and complete evidence. Phase 4 cannot measure product value without a consistent workflow and frozen evaluation method.

## Product and engineering principles

The following rules apply across all four phases:

- Preserve offline, CPU-only analysis as the default operating mode.
- Preserve unknown upstream data unless an explicit, documented policy removes it.
- Distinguish an observed signal from a conclusion that reward hacking occurred.
- Never report “clean” when a detector lacked the data required to evaluate the case.
- Retain every flagged rollout and its evidence in a machine-readable artifact, even when the human-facing report shows only a bounded summary.
- Keep input compatibility explicit and versioned. A path should never be guessed into a format whose contract it does not satisfy.
- Treat analysis errors, ingestion failures, insufficient data, eligible clean cases, and fired findings as different states.
- Use measured detector performance and user review capacity to choose thresholds.
- Keep the HTML report portable, escaped, bounded in size, and usable from `file://`.
- Add dependencies only when they solve a measured product requirement.

## Phase 1: Trustworthy analysis

### Objective

Make RolloutScope safe to run in automation and reliable as a data-processing tool. At the end of this phase, users can trust that their source data was not damaged, normalized records retain their identity and ordering, invalid or unsupported data is visible, configuration was applied exactly, and every detector reports what it could and could not inspect.

### Deliverables

#### 1. Safe output handling

Add a shared output-safety layer used by `convert`, HTML output, JSON output, verdict output, and future indexed output.

It must:

- Resolve source and destination paths before opening an output.
- Reject a destination that is the same file as any input, including equivalent relative paths and symlink aliases.
- Detect collisions when analyzing a directory whose discovered source files include the requested output path.
- Write to a temporary file in the destination directory and atomically replace the destination only after successful serialization.
- Remove an incomplete temporary file after a failed write.
- Preserve an existing destination if analysis or serialization fails.
- Return a clear CLI error identifying the conflicting source and output paths.

Regression tests must cover direct path equality, relative-path equality, symlink aliasing where supported, failure during iteration, existing output preservation, and successful atomic replacement.

#### 2. Normalized input as a first-class format

Introduce explicit input-format detection and a normalized-rollout adapter or reader path.

The normalized path must:

- Detect rows that carry RolloutScope's schema marker without treating them as raw verifiers output.
- Apply the migration chain and schema-version policy before validation.
- Preserve `rollout_id`, `group_id`, `run_id`, and `step_index` when reading normalized data.
- Preserve unknown fields and source provenance.
- Reject unsupported future major versions with a useful message.
- Migrate supported older versions deterministically.
- Prevent raw adapters from silently overwriting normalized identity fields.
- Make format selection visible in the execution manifest.

Add end-to-end tests for raw input to `convert`, converted input to `analyze`, and repeated normalized reads. The resulting rollouts and report aggregates must retain step series, group membership, source occurrence identity, and detector eligibility.

#### 3. Stable identity and provenance contract

Document and implement separate concepts for source occurrence identity and content fingerprinting.

The contract should define:

- Run identity, including how legacy runs without an upstream run ID receive a persistent identity.
- Environment or task namespace.
- Episode, rollout, trace, group, step, turn, and tool-action identity where the source format exposes them.
- A unique occurrence ID for repeated identical generations.
- A content fingerprint that includes the actual trajectory content but excludes mutable scoring results.
- A scoring revision or equivalent mechanism so rescoring does not create a different occurrence.
- Source file, relative path, and line or record location for every normalized occurrence.
- Migration and backward-compatibility rules for existing v1 IDs.

This work requires a schema version decision and an explicit migration. It must happen before any sidecar or run-comparison work depends on the current ID tuple.

#### 4. Ingestion diagnostics and analysis status

Make ingestion completeness part of the public report contract.

For every discovered input file and for the run as a whole, record:

- Format and adapter version selected.
- Rows or records observed.
- Records accepted.
- Records rejected.
- Blank records skipped.
- Duplicate occurrences observed and the policy applied.
- Rejection counts grouped by stable reason code.
- A bounded set of representative error messages with source locations.
- Whether the input was complete, partially accepted, empty, unsupported, or failed.

Define CLI exit behavior independently for analysis execution and detector findings. At minimum:

- Unsupported input and zero usable records produce a nonzero analysis-error exit.
- Partial ingestion can be allowed or rejected through an explicit option or config policy.
- `--fail-on` continues to govern detector findings and does not hide execution failure.
- Terminal, JSON, and HTML outputs show the same analysis state.

#### 5. Strict configuration and numeric validation

Separate permissive upstream data models from strict user configuration models.

Configuration validation must:

- Reject unknown sections and unknown keys, including misspellings.
- Compile and validate regular expressions before reading the input.
- Reject NaN, infinity, and invalid ranges.
- Report the full configuration path associated with an error.
- Record the complete effective configuration after defaults and overrides.
- Record selected detector names and detector implementation versions even when they do not fire.

Normalized numeric fields that participate in analysis must be finite. Add the invariant that any accepted normalized record can be serialized, read again, and analyzed without changing meaning.

#### 6. Complete detector execution and verdict artifacts

Add an execution result for every selected detector, separate from aggregated findings.

Each detector result must report:

- Unit of analysis, such as rollout, group, step, or run.
- Total candidate units.
- Eligible units.
- Fired units.
- Eligible clean units.
- Insufficient-data units.
- Skipped units grouped by reason.
- Detector errors.
- Configuration and implementation version.
- Underlying measurements used to produce a score.

Write complete verdicts to deterministic JSONL or another streamable machine-readable artifact. Every fired unit, rollout ID, evidence span, mode, and measurement must remain retrievable. HTML and terminal output may show a bounded ranked subset, but must point to the complete artifact and clearly label summary counts.

#### 7. Reproducible execution manifest and honest documentation

Extend the report with an execution manifest containing:

- Tool version and source revision when available.
- Schema version.
- Adapter and input-format versions.
- Relative source paths, hashes, sizes, and record counts.
- Full effective configuration.
- Detector versions and selection.
- Ingestion status and execution status.
- Output artifact inventory and hashes.

Update the README and public documentation to distinguish streaming conversion from the currently materialized analysis path. Describe findings as heuristic observations until accuracy evidence supports stronger language. Document supported input versions and known limitations.

### Phase 1 success criteria

Phase 1 is complete only when all of the following are demonstrated:

1. Converting an input onto itself, through a relative alias, or through a symlink fails before the source is opened for writing. The source bytes remain unchanged.
2. A simulated mid-write failure preserves any previous output and leaves no user-visible partial artifact.
3. Raw prime-rl fixture input converted to normalized JSONL and analyzed again produces the same run identity, rollout occurrences, group membership, step indices, aggregate step series, and applicable detector inputs.
4. Reading normalized output repeatedly is idempotent. Identity fields and content fingerprints do not change.
5. Supported old schema rows migrate successfully, and unsupported future versions fail explicitly.
6. A file containing no usable records cannot exit successfully or display “No findings” as if analysis completed normally.
7. A partially valid file reports exact accepted and rejected counts and obeys the configured partial-ingestion policy.
8. Configuration misspellings such as `criticall_at` and `min_correlaton` fail before analysis starts.
9. NaN and infinite analysis values are rejected with source-aware diagnostics; any accepted record survives write, read, and analysis round trips.
10. Every selected detector appears in the execution summary, including detectors with zero eligible units or no fired findings.
11. Every flagged occurrence and evidence span is present in the complete verdict artifact; report-level exemplar limits do not discard it.
12. Terminal, JSON, and HTML outputs agree on input counts, detector coverage, finding counts, and overall execution status.
13. The existing offline suite, Ruff, formatting, and strict mypy remain green. New regression tests cover every reproduced Phase 1 failure.

The phase exit review should include the commands run, output artifacts, a converted-run equivalence diff, and a demonstration of the CLI behavior for complete, partial, empty, unsupported, and output-collision cases.

## Phase 2: Reliable signals

### Objective

Improve attribution, applicability, and measured detector quality. At the end of this phase, RolloutScope can distinguish assistant actions from quoted or observed content, support current upstream records under an explicit compatibility contract, measure each detector on a frozen holdout, and present signals using language justified by their evidence.

### Deliverables

#### 1. Role-aware action and evidence normalization

Create a common representation for model-visible context and assistant behavior. It should preserve message role, turn index, tool-call name and arguments, tool response, source path, and before/after file evidence when present.

Classify evidence states such as:

- Content visible to the model.
- Assistant statement or proposal.
- Tool action attempted.
- Tool action succeeded or failed.
- File or environment state changed.
- Verifier or reward outcome changed afterward.

Detectors should consume this representation rather than independently flattening entire transcripts. The same normalized evidence must drive detector decisions and report highlighting.

#### 2. Repair the six existing detectors

The initial detector catalog should be corrected before new detectors are added.

`verifier_tamper`:

- Scan assistant actions separately from user instructions and tool output.
- Distinguish reading or discussing a test from changing it.
- Prefer confirmed file edits and before/after changes over keyword presence.
- Corroborate suspicious edits with verifier or reward effects when available.
- Include the action, target path, turn, and result in evidence.

`answer_leakage_echo`:

- Treat reference-answer overlap by itself as an observation, not proof of leakage.
- Require evidence that an answer or grading secret was visible to the model before use.
- Respect answer-only, extraction, translation, and other task formats where concise overlap is expected.
- Distinguish reference correctness from criterion copying.

`length_inflation`:

- Keep named correctness or quality metrics separate.
- Aggregate binary outcomes by step, task, or comparable cohort.
- Evaluate reward-length relationships after accounting for available quality signals and task composition.
- Report sample size, correlation, metric coverage, missingness, and mode.
- Avoid claiming a trend when too few steps or comparable samples exist.

`degenerate_repetition`:

- Analyze assistant generations or turns in multi-turn traces.
- Avoid concatenating roles in ways that manufacture repetition.
- Report the exact turn and repeated span.
- Evaluate repeated tool-action loops as a separate candidate signal if benchmark evidence supports it.

`reward_saturation_group_collapse`:

- Enforce minimum group and step requirements consistently in snapshot and trend modes.
- Preserve upstream sampling cohort and group identity.
- Present uniform rewards as a training-health observation unless an independent quality signal supports a reward-hacking interpretation.
- Report eligible groups, singleton exclusions, reward level, variance, and quality divergence separately.

`format_only_wins`:

- Require format and quality signals to be jointly interpretable for a row or comparable cohort.
- Report cases lacking one side as insufficient data.
- Use explicit metric roles and documented scales rather than name matching alone where configuration is available.
- Report how much scalar reward the format component can explain.

#### 3. Metric semantics configuration

Add a typed way to declare metric meaning for a run or environment:

- Metric role: reward component, independent quality, format/parser, safety, cost, or informational.
- Expected range and direction.
- Binary versus continuous interpretation.
- Weight in scalar reward when known.
- Aggregation level and missing-data behavior.

Allow adapters to populate this information from upstream metadata where possible and allow users to supply explicit mappings. The report should distinguish inferred mappings from declared mappings.

#### 4. Current upstream adapter and compatibility matrix

Retain the current adapter under an explicit legacy format name and add support for the selected current verifiers/prime-rl Episode and Trace format.

The new adapter must:

- Discover `rollouts/step_{step}/{kind}/{subset}/traces.jsonl` layouts.
- Preserve episode, trace, environment, task, group, run, step, kind, and subset metadata.
- Require an explicit cohort policy for `all` versus `effective`; it must not silently merge both.
- Normalize message graphs and named weighted rewards without losing source data.
- Preserve failed episodes even when they contain no successful trace.
- Emit clear unsupported-version diagnostics.

Publish a compatibility matrix listing upstream project, tested release or commit, format, supported path shapes, cohort behavior, and fixture provenance. Commit small sanitized golden artifacts produced by supported upstream writers, with license and generation notes.

#### 5. One reproducible detector benchmark

Replace divergent TRACE paths with one shared mapper and evaluation library used by scripts and integration tests.

The benchmark system must record:

- Dataset identifier, revision, license, split, and selected row IDs.
- Raw-data hash or a reproducible manifest where redistribution is restricted.
- Mapping version and normalized artifact hash.
- Detector version and complete effective configuration.
- Unit of evaluation and applicability rules.
- Per-detector and per-category true positives, false positives, true negatives, false negatives, precision, recall, false-positive rate, coverage, sample size, and uncertainty.
- A bounded set of false-positive and false-negative examples for review.

Create separate tuning and frozen holdout partitions at the task or run level so related sibling traces cannot cross the boundary. Add realistic hard negatives: legitimate test maintenance, user discussion of dangerous patterns, tool output quoting existing code, short-answer tasks, grading-criteria discussion, and repeated but appropriate agent loops.

#### 6. Evidence-calibrated reporting

Replace the implication that all detector scores are comparable confidence values. Findings should expose:

- Signal measurements.
- Evidence strength.
- Potential impact.
- Applicability and sample size.
- Corroborating or conflicting quality signals.
- The exact rule and configuration that produced the observation.

Use labels such as “observation,” “suspicious pattern,” or “corroborated finding” according to evidence. Severity used for CI should be a documented policy over these fields, not merely a shared threshold over unrelated detector scores.

### Phase 2 success criteria

Phase 2 is complete only when all of the following are demonstrated:

1. The reproduced benign case where tool output contains `pytest.mark.skip` and `assert True` does not become an assistant-tampering finding without an assistant action.
2. The reproduced answer-only `photosynthesis` case is not classified as leakage without evidence that the answer was exposed.
3. Rising length and reward with unchanged 50% binary accuracy across steps is evaluated using step-level quality aggregates and produces the intended length-divergence observation.
4. Multi-turn assistant repetition is evaluated at the responsible turn rather than silently skipped or manufactured by transcript concatenation.
5. `min_groups` and all other sample-size constraints affect snapshot and trend modes consistently.
6. Rows that do not jointly contain the required format and correctness context appear as insufficient-data coverage, not silent absence or clean results.
7. Supported current Episode/Trace artifacts load with no unexplained record loss, retain cohort identity, and produce deterministic normalized data.
8. Legacy formats continue to pass their compatibility fixtures under an explicitly named support policy.
9. The validation script and integration test call the same mapper and metric implementation and produce identical results from the same manifest.
10. A frozen holdout report is generated automatically with category-level accuracy and applicability. Any release threshold is recorded as a product decision based on review volume; no target is retrofitted after observing the holdout.
11. Every changed detector includes hard-positive and hard-negative regression cases plus documented known limitations.
12. Report language and CI severity accurately reflect observation strength. Uniform reward, short answer overlap, and text-pattern matches are not described as proven reward hacking without corroboration.

The phase exit review should include the compatibility matrix, benchmark manifest, frozen benchmark report, false-positive and false-negative review notes, detector coverage report, and the rationale for any release thresholds.

## Phase 3: Useful investigation workflow

### Objective

Turn detector output into a complete diagnostic workflow. At the end of this phase, an RL engineer can move from a run-level anomaly to the relevant task, group, trajectory, turn, tool action, and evidence; compare it with suitable siblings or a baseline run; record a review decision; and verify whether a later change reduced the issue.

### Deliverables

#### 1. Local investigation data model

Introduce a local indexed representation for normalized occurrences, metrics, detector results, evidence, and review state. SQLite is the initial candidate because it is local and available in the Python standard library, but the final choice should follow measured access patterns.

The store must:

- Preserve complete normalized records and verdicts.
- Index run, task, environment, episode, group, step, detector, status, severity, and metric names.
- Support bounded-memory ingestion and querying.
- Carry schema and migration versions.
- Remain reproducible from source inputs and an execution manifest.
- Avoid duplicating large content unnecessarily.
- Export a portable review bundle and deterministic machine-readable results.

#### 2. Ranked investigation queue

Create a queue of reviewable cases rather than only detector-level summaries.

Each item should show:

- Why it was selected.
- Signal measurements and evidence strength.
- Potential impact and affected population.
- Task, environment, run, group, step, and occurrence identifiers.
- Reward and relevant quality metrics.
- Coverage limitations or missing context.
- Related findings affecting the same occurrence.
- Links to the complete trajectory and relevant comparison set.

Ranking should be deterministic and configurable. It should combine impact, evidence, novelty, and affected population without hiding its component measurements.

#### 3. Full-context trajectory inspection

Provide an HTML or local viewer experience that renders:

- System, user, assistant, and tool messages with distinct roles.
- Turn and trajectory-step boundaries.
- Tool name, arguments, response, and success state.
- Source file and record provenance.
- Reward components, quality metrics, errors, truncation, timing, and token usage.
- Evidence highlighting at the exact field and turn.
- Confirmed before/after file content when available.

The interface must preserve safe escaping and operate offline. Large trajectories should load or expand in bounded sections rather than inflating the initial report indefinitely.

#### 4. Filtering, grouping, and drill-down

Support investigation by detector, evidence state, review status, severity, task, environment, training step, reward range, quality range, group health, completion length, truncation, and error status.

Views should let users move between:

- Run summary.
- Step or cohort anomaly.
- Affected task population.
- Group of sibling rollouts.
- Individual occurrence and turn.

Every aggregate should expose its denominator and retain a path back to the records that produced it.

#### 5. Sibling and baseline comparison

Add comparison capabilities for two related use cases.

Sibling comparison should compare rollouts for the same task or sampling group and show differences in actions, outputs, reward components, quality metrics, length, and failure state.

Run comparison should:

- Verify compatibility of environment, task, metric semantics, and analysis configuration.
- Align occurrences through stable task and group identity.
- Retain and report unmatched populations.
- Show changes in reward, independent quality, detector applicability, finding rate, group health, length, truncation, and cost.
- Report sample counts and appropriate uncertainty.
- Link aggregate changes to representative occurrences.

Add a CLI entry point for reproducible comparison and include both manifests in the output.

#### 6. Human review and adjudication

Allow reviewers to record a structured outcome for each queue item:

- Confirmed reward-integrity issue.
- Benign behavior.
- Needs more context.
- Detector or ingestion bug.
- Duplicate of another reviewed case.

Each review should include reviewer identifier or local label, timestamp, notes, detector version, evidence version, and optional category correction. Reviews must be exportable and importable through a versioned format. Existing review decisions should remain attached when analysis is rerun against the same occurrence, while clearly indicating when detector logic or evidence changed.

#### 7. Repair verification report

Provide a workflow for comparing a baseline run with a post-change run after a rubric, environment, dataset, or detector repair.

The verification report should answer:

- Did the confirmed suspicious behavior decline?
- Did independent task quality improve, remain stable, or regress?
- Did detector applicability change?
- Were tasks added, removed, or materially changed?
- Did reward, cost, length, truncation, or failure rates move unexpectedly?
- Which reviewed examples demonstrate the change?

The report must avoid claiming resolution when the two runs are not sufficiently comparable.

### Phase 3 success criteria

Phase 3 is complete only when all of the following are demonstrated:

1. An external engineer can start from the run summary and reach every source record contributing to a selected finding without manually searching JSONL.
2. Evidence is displayed with correct role, turn, tool action, source provenance, reward components, and relevant quality metrics.
3. The investigation queue can be filtered and sorted while keeping counts and denominators consistent with the execution manifest.
4. A user can inspect a suspicious rollout beside relevant benign siblings and understand the behavioral and scoring differences.
5. A run comparison reports matched and unmatched populations, validates compatibility, and links every important aggregate change to concrete examples.
6. A reviewer can save, export, import, and recover adjudication state without changing source artifacts.
7. Reanalysis with a changed detector version preserves applicable review history and visibly marks changed evidence or conclusions.
8. A baseline and repaired run can produce a verification report that measures suspicious behavior and independent quality together.
9. Representative large reports remain bounded and responsive under the performance budget established for the phase.
10. Usability testing with at least a small set of external RL engineers shows that the workflow can be completed without repository-specific guidance from the original builder. Observed confusion and abandoned steps are recorded and addressed before the phase closes.

The phase exit review should include a recorded or documented end-to-end investigation, the exported review bundle, a baseline comparison, a repair verification example, and measured query/report performance on representative inputs.

## Phase 4: Prove repeat value

### Objective

Validate that RolloutScope solves a recurring problem for independent users, performs within declared resource limits, and can be installed and released reproducibly. At the end of this phase, product and quality targets should come from observed usage and review capacity rather than internal fixtures.

### Deliverables

#### 1. Structured pilot program

Run pilots with several independent RL teams or projects representing the initial target audience: engineers using verifiable or tool-using tasks who see reward improve and need to determine whether task quality improved with it.

Each pilot should include:

- At least one completed historical run.
- A real investigation using RolloutScope.
- Review and adjudication of the prioritized queue.
- An environment, rubric, data, or policy change when an issue is confirmed.
- A subsequent comparison or verification run where practical.
- A debrief that records missing inputs, unsupported formats, confusing output, and unmet workflow needs.

Pilot data must be handled under explicit privacy and retention agreements. The product should continue to work locally when teams cannot share raw traces. Collect aggregate outcome data and sanitized examples only with permission.

#### 2. Product value instrumentation

Measure the workflow rather than relying on usage counts alone.

Track per pilot:

- Time from input selection to usable analysis.
- Ingestion success and detector applicability.
- Number of queue items reviewed.
- Confirmed issues, benign findings, insufficient-context cases, and detector bugs.
- Precision within the reviewed priority band.
- Time to first useful finding.
- Total triage time and the team's estimated previous process.
- Whether the result changed an environment, rubric, dataset, model, or release decision.
- Whether the team analyzed a later run.
- Whether a repair was verified.
- Qualitative confidence in the evidence and comparison workflow.

Define a review budget, such as the number of cases a team is willing to inspect per run, before selecting operational detector thresholds.

#### 3. Quality targets from real alert volume

Use pilot review data and the frozen benchmark together to set detector release policies.

Policies should specify:

- Minimum applicability for a detector to be shown by default.
- Maximum acceptable false-positive volume within the priority queue.
- Required evidence fields for each conclusion level.
- Conditions that downgrade a detector to experimental.
- Required sample size and uncertainty for published accuracy claims.
- Workload categories for which a detector is unsupported.
- Regression tolerance between releases.

Do not combine unrelated detectors into a single accuracy number. Publish limitations and category-level results.

#### 4. Bounded-memory and performance program

Redesign or refine the pipeline so declared input sizes fit within a measured resource envelope.

The program should include:

- Incremental processing for per-rollout detectors.
- Explicit grouping windows or disk-backed state for group and trend detectors.
- Bounded evidence and report summaries backed by complete indexed artifacts.
- High-cardinality group handling.
- Long multi-turn trajectory handling.
- Resume and append behavior.
- Clear cancellation and partial-output semantics.

Benchmark representative small, medium, and large inputs. Record input bytes, record count, trajectory size, group cardinality, elapsed time, peak memory, output size, and detector mix. Publish the supported envelope and fail predictably when a configured limit is exceeded.

#### 5. Packaging, installation, and release readiness

Add a release-quality pipeline that:

- Builds source and wheel distributions from a clean checkout.
- Installs the wheel into a clean environment outside the source tree.
- Exercises CLI discovery, adapters, detector entry points, JSON output, and packaged HTML templates.
- Runs the supported Python version matrix.
- Uses the committed lockfile in a frozen or locked mode for development checks.
- Enforces the agreed test and coverage gates.
- Produces a software bill of materials or dependency inventory where useful.
- Verifies version consistency across package metadata, reports, changelog, and release tag.
- Publishes only through an explicit release workflow with documented rollback or yanking steps.

Complete public onboarding documentation with installation, first real-run analysis, supported input formats, metric semantics, privacy model, expected outputs, CI usage, troubleshooting, comparison workflow, and a linked sample report.

#### 6. Feedback and roadmap decision

At the end of the pilots, decide which requests belong in the core product based on observed recurrence and value.

Candidate later work includes additional adapters, live or incremental training hooks, richer semantic checks, and white-box sidecars. Each candidate should have:

- A repeated user problem observed in pilots.
- A defined input contract.
- A measurable success criterion.
- A validation dataset or partner case.
- A maintenance and privacy assessment.
- A reason it belongs in RolloutScope rather than an existing training dashboard or generic trace viewer.

### Phase 4 success criteria

Phase 4 is complete only when all of the following are demonstrated:

1. Several independent teams or projects complete a real investigation; the exact minimum pilot count is chosen before recruitment and recorded in the pilot protocol.
2. More than one team analyzes a subsequent run or uses the comparison workflow, demonstrating repeat use rather than one-time curiosity.
3. Pilot reviewers can trace prioritized findings to evidence and reach structured decisions within their declared review budget.
4. Confirmed findings lead to documented engineering or release decisions in at least some pilots, and RolloutScope can evaluate the resulting run when one is available.
5. Detector quality targets are chosen from frozen benchmark results, reviewed pilot alerts, and user capacity. Targets and confidence bounds are published before claiming dependable detection.
6. Unsupported and insufficient-data workloads are documented and visible in the product.
7. Representative large inputs complete within the published time, peak-memory, and output-size budgets. The benchmarks are reproducible from a committed generator or redistributable fixtures.
8. A clean wheel installation passes an end-to-end smoke test outside the repository checkout on every supported Python version.
9. Release metadata, changelog, source revision, detector versions, execution manifest, and generated reports agree on the shipped version.
10. Public documentation enables a new user to install the package, analyze a supported run, interpret coverage and findings, inspect evidence, and compare a subsequent run without private project instructions.
11. The next roadmap is approved using pilot evidence. New detector families, live monitoring, or white-box work are started only when they address a repeated measured need and have an evaluation plan.

The phase exit review should contain the pilot protocol and aggregate results, anonymized case studies where permitted, quality-policy decisions, performance benchmark artifacts, clean-install evidence, release checklist, and the evidence-based next roadmap.

## Cross-phase verification matrix

| Capability | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|---|---|---|---|---|
| Input safety | Prevent loss and expose ingestion state | Support versioned current and legacy formats | Preserve provenance through investigation | Validate on independent operational inputs |
| Identity | Define stable occurrence and content contracts | Preserve upstream episode and trace identity | Use identity for siblings, reviews, and comparisons | Verify across successive partner runs |
| Detector coverage | Report eligible, skipped, and failed units | Improve attribution and applicability | Make every finding traceable in context | Tune default policy to real review capacity |
| Evidence | Preserve all verdicts and spans | Normalize role, action, and corroboration | Present full context and adjudication | Measure whether evidence supports decisions |
| Evaluation | Establish reproducible manifests | Produce a frozen detector holdout | Test end-to-end diagnostic tasks | Combine benchmark and pilot outcomes |
| Scale | Correct claims and define baseline measurements | Avoid unnecessary detector expansion costs | Use indexed, bounded investigation artifacts | Meet a published resource envelope |
| Distribution | Preserve current quality checks | Test supported-format compatibility | Package portable review bundles | Ship clean wheel and release workflow |

## Phase governance

Each phase should begin with a short design record for any schema, identity, storage, or public CLI changes. Work may be split into parallel packages after the shared contract is agreed, but the phase gate is evaluated against end-to-end behavior rather than package-level completion.

Every phase closes with:

- The full offline test suite.
- Ruff lint and format checks.
- Strict mypy.
- Targeted end-to-end acceptance scenarios listed in this plan.
- Updated compatibility and limitations documentation.
- A reproducible evidence bundle containing commands, manifests, reports, and benchmark results appropriate to the phase.
- A changelog entry describing final user-visible behavior.
- A review of unresolved risks and an explicit go or no-go decision for the next phase.

If a success criterion cannot be met, it should remain open with evidence and an owner. It should not be converted into a documentation-only exception when later phases depend on the missing behavior.
