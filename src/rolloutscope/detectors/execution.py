"""Account for detector coverage independently of positive-only verdict output.

Built-in guards are shared with the scoring implementations. Unknown plugins
retain their output but are explicitly unsupported for coverage accounting.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from importlib.metadata import packages_distributions, version
from typing import Any, Literal

from rolloutscope.detectors.answer_leakage_echo import AnswerLeakageEchoDetector
from rolloutscope.detectors.base import Detector, DetectorConfig
from rolloutscope.detectors.degenerate_repetition import (
    DegenerateRepetitionDetector,
    _ineligible_reason,
)
from rolloutscope.detectors.format_only_wins import FormatOnlyWinsDetector, _metric_keys
from rolloutscope.detectors.length_inflation import (
    LengthInflationDetector,
    _correctness_values,
    _correlation_inputs,
    _samples,
)
from rolloutscope.detectors.reward_saturation_group_collapse import (
    RewardSaturationGroupCollapseDetector,
    _dead_stats,
    _group,
    _trend_inputs,
)
from rolloutscope.detectors.verifier_tamper import VerifierTamperDetector, _has_inspectable_output
from rolloutscope.schema import Rollout, Verdict
from rolloutscope.schema.execution import DetectorExecution, UnitCounts

_SKIP_REASONS = {"multi_turn_transcript", "reward_below_minimum", "missing_step_index"}
_ROW_TYPES = (
    VerifierTamperDetector,
    FormatOnlyWinsDetector,
    DegenerateRepetitionDetector,
    AnswerLeakageEchoDetector,
)
_BUILTIN_TYPES = (*_ROW_TYPES, LengthInflationDetector, RewardSaturationGroupCollapseDetector)


def _implementation_version(detector: Detector, declared: str) -> str:
    """Identify an implementation using stable package and declared versions."""
    package = "rolloutscope" if type(detector) in _BUILTIN_TYPES else None
    try:
        if package is None:
            module_root = type(detector).__module__.split(".")[0]
            distributions = packages_distributions().get(module_root, [])
            package = distributions[0] if len(distributions) == 1 else None
        package_version = version(package) if package else "unknown"
    except Exception:
        package_version = "unknown"
    return f"{package or 'unknown'}:{package_version}:implementation:{declared}"


def _metadata_failure(
    detector: Detector,
    exc: Exception,
    *,
    detector_name: str | None = None,
    category: str = "unknown",
    config: dict[str, Any] | None = None,
) -> DetectorExecution:
    """Create a visible failed execution when plugin metadata cannot be inspected."""
    detector_type = type(detector)
    fallback_name = f"{detector_type.__module__}.{detector_type.__qualname__}"
    return DetectorExecution(
        detector=detector_name or fallback_name,
        category=category,
        version="unknown",
        status="failed",
        config=config or {},
        units=[
            UnitCounts(
                unit="run",
                candidate=1,
                errors=1,
                reason_counts={"detector_metadata_error": 1},
            )
        ],
        errors=[f"detector metadata error: {type(exc).__name__}: {exc}"],
    )


def _counts(
    unit: Literal["rollout", "group", "step", "run"],
    mode: str,
    reasons: Sequence[str | None],
    measurements: dict[str, Any] | None = None,
) -> UnitCounts:
    reasons_count = Counter(reason for reason in reasons if reason is not None)
    skipped = sum(count for reason, count in reasons_count.items() if reason in _SKIP_REASONS)
    eligible = sum(reason is None for reason in reasons)
    return UnitCounts(
        unit=unit,
        mode=mode,
        candidate=len(reasons),
        eligible=eligible,
        clean=eligible,
        skipped=skipped,
        insufficient_data=len(reasons) - eligible - skipped,
        reason_counts=dict(reasons_count),
        measurements=measurements or {},
    )


def _coverage(
    detector: Detector, rollouts: Sequence[Rollout], config: DetectorConfig
) -> list[UnitCounts]:
    """Use the same prerequisites as detection, keeping each scoring unit distinct."""
    reasons: list[str | None] = []
    if type(detector) in _ROW_TYPES:
        for rollout in rollouts:
            if isinstance(detector, VerifierTamperDetector):
                reason = None if _has_inspectable_output(rollout) else "missing_completion"
            elif isinstance(detector, FormatOnlyWinsDetector):
                format_keys, correctness_keys = _metric_keys(rollout, config.format_only_wins)
                reason = None if format_keys and correctness_keys else "missing_metric_pair"
            elif isinstance(detector, DegenerateRepetitionDetector):
                reason = _ineligible_reason(rollout, config.degenerate_repetition)
            elif isinstance(detector, AnswerLeakageEchoDetector):
                reason = detector._ineligible_reason(rollout, config.answer_leakage_echo)
            else:
                raise AssertionError("unhandled built-in row detector")
            reasons.append(reason)
        return [_counts("rollout", "snapshot", reasons)]
    if isinstance(detector, LengthInflationDetector):
        samples = _samples(rollouts)
        units: list[UnitCounts] = []
        for mode in ("snapshot", "trend"):
            _, reason, measured = _correlation_inputs(samples, config.length_inflation, mode)
            correctness = _correctness_values(
                [rollout for rollout, _ in samples],
                config.length_inflation.correctness_metric_patterns,
            )
            measured.update(
                input_rollout_count=len(rollouts),
                empty_completion_count=len(rollouts) - len(samples),
                correctness_value_count=len(correctness),
                correctness_range=max(correctness) - min(correctness) if correctness else None,
                correctness_corroborated=bool(correctness),
            )
            units.append(_counts("run", mode, [reason], measured))
        return units
    if isinstance(detector, RewardSaturationGroupCollapseDetector):
        cfg = config.reward_saturation_group_collapse
        groups = _group(rollouts)
        eligible, dead = _dead_stats(groups, cfg)
        eligible_keys = set(eligible)
        group_reasons = [
            "too_few_group_members"
            if key not in eligible_keys
            else "too_few_groups"
            if len(eligible) < cfg.min_groups
            else None
            for key in groups
        ]
        group_counts = _counts(
            "group",
            "group",
            group_reasons,
            {
                "groups_with_sufficient_members": len(eligible),
                "dead_groups": len(dead),
                "dead_group_fraction": len(dead) / len(eligible) if eligible else None,
            },
        )
        inputs = _trend_inputs(rollouts, cfg)
        trend_counts = _counts(
            "run",
            "trend",
            [inputs.reason],
            {
                "steps": inputs.steps,
                "dead_group_fractions": inputs.fractions,
                "reward_means": inputs.reward_means,
                "correctness_means": inputs.correctness_means,
                "missing_step_count": sum(r.step_index is None for r in rollouts),
            },
        )
        return [group_counts, trend_counts]
    return []


def execute_detector(
    detector: Detector, rollouts: Sequence[Rollout], config: DetectorConfig
) -> tuple[list[Verdict], DetectorExecution]:
    """Run one detector with explicit coverage, effective config, version, and failures.

    Positive-only detector output does not define the denominator. Each built-in
    mode records its actual applicability guards. A plugin without an accounting
    contract is marked unsupported even if its detection call returns normally.
    Exceptions discard all verdicts from that detector and mark its eligible
    units as errors, so incomplete work cannot be reported as a clean scan.
    """
    effective = config.model_dump(mode="json")
    detector_name: str | None = None
    category = "unknown"
    try:
        detector_name = detector.name
        category = detector.category
        detect = detector.detect
        declared_version = getattr(detector, "version", "unknown")
        if (
            not isinstance(detector_name, str)
            or not isinstance(category, str)
            or not callable(detect)
            or not isinstance(declared_version, str)
        ):
            raise TypeError("detector name, category, detect, or version has an invalid type")
    except Exception as exc:
        return [], _metadata_failure(
            detector,
            exc,
            detector_name=detector_name if isinstance(detector_name, str) else None,
            category=category if isinstance(category, str) else "unknown",
            config=effective,
        )

    builtin = type(detector) in _BUILTIN_TYPES
    execution = DetectorExecution(
        detector=detector_name,
        category=category,
        version=_implementation_version(detector, declared_version),
        config=effective[detector_name] if builtin else effective,
    )
    if type(detector) in (LengthInflationDetector, RewardSaturationGroupCollapseDetector):
        runs: dict[str | None, list[Rollout]] = {}
        for rollout in rollouts:
            runs.setdefault(rollout.run_id, []).append(rollout)
        if len(runs) > 1:
            all_verdicts: list[Verdict] = []
            statuses = []
            for run_id, members in runs.items():
                run_verdicts, run_execution = execute_detector(detector, members, config)
                statuses.append(run_execution.status)
                for unit in run_execution.units:
                    unit.measurements["run_id"] = run_id
                execution.units.extend(run_execution.units)
                execution.errors.extend(run_execution.errors)
                all_verdicts.extend(
                    verdict.model_copy(update={"run_id": run_id}) for verdict in run_verdicts
                )
            if "failed" in statuses:
                execution.status = "failed"
                execution.verdict_count = 0
                return [], execution
            execution.status = (
                "complete"
                if "complete" in statuses
                else "insufficient_data"
                if "insufficient_data" in statuses
                else "skipped"
            )
            execution.verdict_count = len(all_verdicts)
            return all_verdicts, execution
    try:
        execution.units = _coverage(detector, rollouts, config) if builtin else []
        verdicts = detect(rollouts, config)
        if not isinstance(verdicts, list) or any(
            not isinstance(verdict, Verdict) or verdict.detector != detector_name
            for verdict in verdicts
        ):
            raise ValueError("detector returned invalid verdicts or mismatched detector names")
        execution.verdict_count = len(verdicts)
        for index, unit in enumerate(execution.units):
            fired = sum(
                verdict.fired and getattr(verdict, "mode", "snapshot") == unit.mode
                for verdict in verdicts
            )
            payload = unit.model_dump()
            payload.update(fired=fired, clean=unit.eligible - fired)
            # Validate the output-to-unit relationship instead of silently clipping it.
            execution.units[index] = UnitCounts.model_validate(payload)
        if not builtin:
            execution.status = "unsupported"
            execution.errors = ["plugin does not provide a supported coverage accounting contract"]
        elif any(unit.eligible for unit in execution.units):
            execution.status = "complete"
        elif any(unit.insufficient_data for unit in execution.units) or not rollouts:
            execution.status = "insufficient_data"
        else:
            execution.status = "skipped"
        return verdicts, execution
    except Exception as exc:
        execution.status = "failed"
        execution.verdict_count = 0
        execution.errors = [f"{type(exc).__name__}: {exc}"]
        failed_units = []
        for unit in execution.units:
            payload = unit.model_dump()
            payload.update(eligible=0, fired=0, clean=0, errors=unit.eligible)
            payload["reason_counts"] = {**unit.reason_counts, "detector_error": unit.eligible}
            failed_units.append(UnitCounts.model_validate(payload))
        execution.units = failed_units or [
            UnitCounts(unit="run", candidate=1, errors=1, reason_counts={"detector_error": 1})
        ]
        return [], execution
