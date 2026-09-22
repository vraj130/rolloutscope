"""The shared benchmark driver.

The validation script and the integration test both call :func:`evaluate_rows`.
That is the whole point of this module: one mapping, one applicability rule, one
metric implementation, so the two callers cannot report different numbers for the
same dataset revision again.

Order of operations, and why it is that order:

1. Restrict the fetched rows to the split under evaluation. A holdout number
   measured over rows from another split is not a holdout number.
2. Verify every remaining row against the manifest's content hash. A changed row
   means the revision pin was not honoured, which is recorded on the report
   rather than silently absorbed.
3. Map rows to rollouts, counting unusable rows with a reason.
4. Decide applicability per detector per unit, before any detector runs.
5. Run each detector once over the whole population, because group and trend
   detectors need the population, then attribute fired verdicts back to units.
6. Aggregate, and keep a bounded set of false positives and false negatives for
   human review.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

from rolloutscope.benchmark.applicability import Requirement, applicability
from rolloutscope.benchmark.manifest import BenchmarkManifest, verify
from rolloutscope.benchmark.metrics import Outcome, aggregate
from rolloutscope.benchmark.report import (
    BenchmarkReport,
    DetectorPin,
    ExampleCase,
    detector_fingerprint,
)
from rolloutscope.benchmark.splits import Split
from rolloutscope.benchmark.taxonomy import name_of
from rolloutscope.benchmark.trace_map import TraceRowError, map_row
from rolloutscope.detectors import Detector, DetectorConfig
from rolloutscope.schema import EvidenceSpan, Rollout

MAX_EXAMPLES = 10
"""How many false positives and false negatives to carry for review.

A bounded list keeps the report readable. It is a display limit only: the full
per-unit outcomes stay available from :func:`score_detector`.
"""


def population_signals(rollouts: Sequence[Rollout]) -> frozenset[Requirement]:
    """Report which population-level signals an evaluated cohort satisfies.

    Input: the rollouts under evaluation. Output: the subset of
    ``{GROUPS, STEPS}`` the cohort supports. GROUPS needs at least two groups
    that each hold at least two rollouts, since a population of singletons
    cannot support a within-group comparison. STEPS needs at least two distinct
    ``step_index`` values.
    """
    signals: set[Requirement] = set()
    groups: Counter[str] = Counter()
    steps: set[int] = set()
    for rollout in rollouts:
        if rollout.group_id:
            groups[rollout.group_id] += 1
        if rollout.step_index is not None:
            steps.add(rollout.step_index)
    if sum(1 for size in groups.values() if size >= 2) >= 2:
        signals.add(Requirement.GROUPS)
    if len(steps) >= 2:
        signals.add(Requirement.STEPS)
    return frozenset(signals)


def _fired_units(
    detector: Detector, rollouts: Sequence[Rollout], config: DetectorConfig
) -> tuple[set[str], dict[str, EvidenceSpan], str]:
    """Run one detector and report which units it fired on.

    Input: the detector, the population, and the config. Output: the set of
    rollout ids with a fired verdict, one exemplar evidence span per unit, and an
    error string (empty when the detector ran). A detector that raises is
    reported as an error for every unit rather than as a population of clean
    results.
    """
    fired: set[str] = set()
    evidence: dict[str, EvidenceSpan] = {}
    try:
        verdicts = detector.detect(rollouts, config)
    except Exception as exc:  # a broken detector must not be scored as clean
        return fired, evidence, f"{type(exc).__name__}: {exc}"
    for verdict in verdicts:
        if not verdict.fired:
            continue
        for rollout_id in verdict.rollout_ids:
            if not rollout_id:
                continue
            fired.add(rollout_id)
        for span in verdict.evidence:
            evidence.setdefault(span.rollout_id, span)
    return fired, evidence, ""


def score_detector(
    detector: Detector,
    rollouts: Sequence[Rollout],
    labels: dict[str, tuple[bool, tuple[str, ...]]],
    config: DetectorConfig,
    *,
    population: frozenset[Requirement],
) -> tuple[list[Outcome], dict[str, EvidenceSpan]]:
    """Score one detector over one population.

    Input: the detector, the rollouts, a map from rollout id to
    ``(is_positive, codes)``, the detector config, and the population-level
    signals. Output: one :class:`Outcome` per rollout plus the exemplar evidence
    spans keyed by rollout id.
    """
    fired, evidence, error = _fired_units(detector, rollouts, config)
    outcomes: list[Outcome] = []
    for rollout in rollouts:
        unit_id = rollout.rollout_id or ""
        positive, codes = labels.get(unit_id, (False, ()))
        if error:
            outcomes.append(Outcome(unit_id=unit_id, scored=False, error=error))
            continue
        verdict = applicability(detector.name, rollout, population=population)
        outcomes.append(
            Outcome(
                unit_id=unit_id,
                scored=verdict.applicable,
                reason=verdict.reason,
                fired=unit_id in fired,
                positive=positive,
                codes=codes,
            )
        )
    return outcomes, evidence


def _examples(
    outcomes: Sequence[Outcome], evidence: dict[str, EvidenceSpan], cell: str
) -> list[ExampleCase]:
    """Collect a bounded, deterministic sample of one confusion cell."""
    picked = sorted((o for o in outcomes if o.cell == cell), key=lambda o: o.unit_id)
    cases: list[ExampleCase] = []
    for outcome in picked[:MAX_EXAMPLES]:
        span = evidence.get(outcome.unit_id)
        cases.append(
            ExampleCase(
                unit_id=outcome.unit_id,
                cell=cell,
                codes=outcome.codes,
                evidence_field=span.field if span else "",
                evidence_text=(span.text[:300] if span else ""),
            )
        )
    return cases


def evaluate_rows(
    rows: Sequence[dict[str, Any]],
    manifest: BenchmarkManifest,
    detectors: dict[str, Detector],
    config: DetectorConfig,
    *,
    split: Split | None = None,
    source: str = "",
    notes: Sequence[str] = (),
) -> BenchmarkReport:
    """Evaluate detectors over a manifest-pinned population.

    Input: the raw fetched rows, the frozen manifest, the detectors to score,
    the detector config, the split to restrict to (``None`` evaluates every
    manifest row present), how the rows were obtained, and free-text notes.
    Output: a :class:`BenchmarkReport`.

    Rows absent from the manifest are ignored rather than scored, so a fetch that
    over-returns cannot widen the evaluated population behind the caller's back.
    """
    index = manifest.by_trajectory_id
    wanted = {ref.trajectory_id for ref in (manifest.for_split(split) if split else manifest.rows)}
    selected = [row for row in rows if str(row.get("trajectory_id") or "") in wanted]

    verification = verify(manifest, selected, only=wanted)
    verification_notes: list[str] = []
    if verification.missing:
        verification_notes.append(
            f"{len(verification.missing)} manifest rows were not fetched, "
            f"first: {', '.join(verification.missing[:5])}"
        )
    if verification.changed:
        verification_notes.append(
            f"{len(verification.changed)} rows no longer match their manifest hash, "
            f"first: {', '.join(verification.changed[:5])}. "
            "The revision pin was not honoured; numbers below are not comparable."
        )

    rollouts: list[Rollout] = []
    labels: dict[str, tuple[bool, tuple[str, ...]]] = {}
    unusable: Counter[str] = Counter()
    for position, row in enumerate(selected):
        try:
            rollout = map_row(
                row,
                position=position,
                dataset=manifest.dataset.name,
                revision=manifest.dataset.revision,
            )
        except TraceRowError as exc:
            unusable[str(exc).split(": ", 1)[-1]] += 1
            continue
        ref = index[str(row.get("trajectory_id") or "")]
        rollouts.append(rollout)
        labels[rollout.rollout_id or ""] = (ref.positive, ref.codes)

    population = population_signals(rollouts)
    config_payload = config.model_dump(mode="json")
    code_names = {code: name_of(code) for _, codes in labels.values() for code in codes}

    pins: list[DetectorPin] = []
    metrics = []
    false_positives: list[ExampleCase] = []
    false_negatives: list[ExampleCase] = []
    for name in sorted(detectors):
        detector = detectors[name]
        pins.append(
            DetectorPin(
                name=name,
                category=detector.category,
                source_sha256=detector_fingerprint(detector),
                config=config_payload.get(name, {}),
            )
        )
        outcomes, evidence = score_detector(
            detector, rollouts, labels, config, population=population
        )
        metrics.append(aggregate(name, outcomes, category_name=code_names))
        false_positives.extend(_examples(outcomes, evidence, "fp"))
        false_negatives.extend(_examples(outcomes, evidence, "fn"))

    return BenchmarkReport(
        dataset=manifest.dataset.name,
        dataset_revision=manifest.dataset.revision,
        manifest_digest=manifest.digest(),
        mapping_version=manifest.mapping_version,
        split_version=manifest.split_version,
        split=split or "all",
        source=source,
        rows_requested=len(selected),
        rows_mapped=len(rollouts),
        rows_unusable=sum(unusable.values()),
        unusable_by_reason=dict(sorted(unusable.items())),
        manifest_verified=verification.ok,
        verification_notes=verification_notes,
        detectors=pins,
        metrics=metrics,
        false_positives=false_positives,
        false_negatives=false_negatives,
        notes=list(notes),
    )
