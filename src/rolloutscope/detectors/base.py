"""Detector protocol, per-detector configuration, and the entry-point registry.

Detectors are pure functions over normalized rollouts: no IO, no network, no
model calls. Each detector returns structured Verdict objects; a fired verdict
always carries at least one EvidenceSpan (the schema enforces this with a
validator).

Built-in detectors register through the ``rolloutscope.detectors`` entry-point
group in pyproject.toml, exactly the mechanism third-party packages use.
Discovery happens in :func:`load_detectors`; :func:`builtin_detectors` is a
plain-code fallback for environments where entry-point metadata is genuinely
unavailable, and :func:`discover_detectors` prefers entry points and falls back
only when discovery returns nothing.

Every numeric default in the config models below is a conservative heuristic
chosen for this project, not a value taken from any paper, and every one is
configurable. See each field description.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Sequence
from importlib.metadata import entry_points
from typing import Protocol, cast, runtime_checkable

from pydantic import BaseModel, Field

from rolloutscope.schema import Rollout, Verdict

logger = logging.getLogger(__name__)

DETECTOR_ENTRY_POINT_GROUP = "rolloutscope.detectors"

# (entry point name, module, attribute) for the plain-code fallback registry.
_BUILTIN_SPECS: tuple[tuple[str, str, str], ...] = (
    ("verifier_tamper", "rolloutscope.detectors.verifier_tamper", "VerifierTamperDetector"),
    (
        "reward_saturation_group_collapse",
        "rolloutscope.detectors.reward_saturation_group_collapse",
        "RewardSaturationGroupCollapseDetector",
    ),
    ("length_inflation", "rolloutscope.detectors.length_inflation", "LengthInflationDetector"),
    ("format_only_wins", "rolloutscope.detectors.format_only_wins", "FormatOnlyWinsDetector"),
    (
        "degenerate_repetition",
        "rolloutscope.detectors.degenerate_repetition",
        "DegenerateRepetitionDetector",
    ),
    (
        "answer_leakage_echo",
        "rolloutscope.detectors.answer_leakage_echo",
        "AnswerLeakageEchoDetector",
    ),
)

_DEFAULT_TAMPER_PATTERNS: dict[str, str] = {
    "pytest_skip": r"pytest\.(?:mark\.)?(?:skip|xfail)|@unittest\.skip|raise\s+unittest\.SkipTest",
    "forced_exit_zero": r"sys\.exit\(\s*0\s*\)|os\._exit\(\s*0\s*\)",
    "shell_force_success": r"\|\|\s*true\b",
    "constant_assertion": r"\bassert\s+(?:True\b|1\s*==\s*1)",
    "checker_monkeypatch": (
        r"(?i)(?:monkeypatch\.setattr|mock\.patch|patch\.object)\s*\(\s*['\"]?[\w.]*"
        r"(?:test|check|verif|grad|score|reward|eval)"
    ),
    "always_pass_test_body": r"def\s+test_\w+\s*\([^)]*\)\s*:\s*(?:pass\b|return\s+True\b)",
    "test_file_patch_header": (
        r"(?m)^(?:\+\+\+|---)\s+\S*(?:\btests?/|test_[\w-]+\.\w+|[\w-]+_test\.\w+)"
    ),
}

_DEFAULT_TEST_PATH_REGEX = (
    r"\btests?/[\w./-]*|\btest_[\w-]+\.py\b|\b[\w-]+_test\.py\b|\bconftest\.py\b"
)

_DEFAULT_EDIT_TOOL_REGEX = r"write|edit|patch|replace|create|apply|modify|delete|remove"

_DEFAULT_CORRECTNESS_PATTERNS: tuple[str, ...] = (
    "correct",
    "accuracy",
    "pass",
    "success",
    "solve",
    "exact",
)

_DEFAULT_FORMAT_PATTERNS: tuple[str, ...] = ("format", "parse")

_DEFAULT_CRITERIA_KEY_PATTERNS: tuple[str, ...] = ("criteri", "rubric", "grading", "checklist")

_DEFAULT_CRITERIA_LINE_REGEX = (
    r"(?i)(?:grading criteria|rubric|award(?:ed)?\s+(?:full\s+)?credit"
    r"|you will be (?:scored|graded))"
)


class VerifierTamperConfig(BaseModel):
    """Configuration for the verifier_tamper detector.

    All defaults are conservative heuristics chosen for this project, not
    values sourced from any paper.
    """

    patterns: dict[str, str] = Field(
        default_factory=lambda: dict(_DEFAULT_TAMPER_PATTERNS),
        description=(
            "Label-to-regex map of tampering patterns scanned over completion and "
            "trajectory text, tool-call renderings included (heuristic defaults)."
        ),
    )
    test_path_regex: str = Field(
        default=_DEFAULT_TEST_PATH_REGEX,
        description=(
            "Regex that identifies test-file paths inside tool-call arguments (heuristic default)."
        ),
    )
    edit_tool_name_regex: str = Field(
        default=_DEFAULT_EDIT_TOOL_REGEX,
        description=(
            "Case-insensitive regex over tool names that counts a tool call as file-editing "
            "(heuristic default)."
        ),
    )
    min_matches: int = Field(
        default=1,
        ge=1,
        description="Distinct pattern labels required before a rollout fires (heuristic).",
    )
    base_score: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Score assigned for a single matched pattern label (heuristic).",
    )
    per_extra_match_score: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Score increment per additional distinct label, capped at 1.0 (heuristic).",
    )


class RewardSaturationGroupCollapseConfig(BaseModel):
    """Configuration for the reward_saturation_group_collapse detector.

    All defaults are conservative heuristics, assuming rewards roughly in
    [0, 1]; none come from a paper.
    """

    variance_epsilon: float = Field(
        default=1e-9,
        ge=0.0,
        description="Within-group reward variance at or below this counts as zero (dead group).",
    )
    min_group_size: int = Field(
        default=2,
        ge=2,
        description=(
            "Groups smaller than this are ignored (a singleton is trivially zero-variance)."
        ),
    )
    min_groups: int = Field(
        default=2,
        ge=1,
        description=(
            "Minimum eligible groups before the dead-group fraction is assessed (heuristic)."
        ),
    )
    dead_fraction_threshold: float = Field(
        default=0.5,
        ge=0.0,
        description="Fraction of dead groups at or above which group verdicts fire (heuristic).",
    )
    saturated_reward_min: float = Field(
        default=0.5,
        description=(
            "Dead groups whose shared reward is at or above this are reported as saturated; "
            "heuristic, assumes rewards roughly in [0, 1]."
        ),
    )
    min_steps: int = Field(
        default=3,
        ge=2,
        description=(
            "Distinct step_index values required before the trend variant runs (heuristic)."
        ),
    )
    min_dead_fraction_rise: float = Field(
        default=0.25,
        description=(
            "Minimum rise in dead-group fraction between first and last step for the trend "
            "verdict (heuristic)."
        ),
    )
    metric_flat_epsilon: float = Field(
        default=0.05,
        ge=0.0,
        description=(
            "Correctness-metric step means may rise at most this much between first and last "
            "step and still count as flat (heuristic)."
        ),
    )
    correctness_metric_patterns: list[str] = Field(
        default_factory=lambda: list(_DEFAULT_CORRECTNESS_PATTERNS),
        description=(
            "Case-insensitive substrings identifying correctness-ish metric keys "
            "(heuristic defaults)."
        ),
    )


class LengthInflationConfig(BaseModel):
    """Configuration for the length_inflation detector.

    All defaults are conservative heuristics; none come from a paper.
    """

    min_samples: int = Field(
        default=8,
        ge=2,
        description="Minimum rollouts before the snapshot correlation is computed (heuristic).",
    )
    min_correlation: float = Field(
        default=0.8,
        description=(
            "Pearson correlation of length vs reward at or above which it fires (heuristic)."
        ),
    )
    correctness_metric_patterns: list[str] = Field(
        default_factory=lambda: list(_DEFAULT_CORRECTNESS_PATTERNS),
        description=(
            "Case-insensitive substrings identifying correctness-ish metric keys "
            "(heuristic defaults)."
        ),
    )
    correctness_flat_epsilon: float = Field(
        default=0.1,
        ge=0.0,
        description=(
            "Correctness values whose max-min range is at or below this count as flat; only a "
            "flat task metric lets the detector fire at full score (heuristic)."
        ),
    )
    missing_metric_score_factor: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Score multiplier applied when no correctness metric exists to corroborate (heuristic)."
        ),
    )
    min_steps: int = Field(
        default=3,
        ge=2,
        description=(
            "Distinct step_index values required before the trend variant runs (heuristic)."
        ),
    )
    evidence_snippet_chars: int = Field(
        default=160,
        ge=1,
        description="Characters of the exemplar completion carried in the evidence span.",
    )


class FormatOnlyWinsConfig(BaseModel):
    """Configuration for the format_only_wins detector.

    All defaults are conservative heuristics, assuming metrics roughly in
    [0, 1]; none come from a paper.
    """

    format_metric_patterns: list[str] = Field(
        default_factory=lambda: list(_DEFAULT_FORMAT_PATTERNS),
        description=(
            "Case-insensitive substrings identifying format/parser metric keys "
            "(heuristic defaults)."
        ),
    )
    correctness_metric_patterns: list[str] = Field(
        default_factory=lambda: list(_DEFAULT_CORRECTNESS_PATTERNS),
        description=(
            "Case-insensitive substrings identifying correctness-ish metric keys "
            "(heuristic defaults)."
        ),
    )
    format_near_max: float = Field(
        default=0.9,
        description=(
            "A format metric at or above this counts as near max; heuristic, assumes metrics "
            "roughly in [0, 1]."
        ),
    )
    correctness_near_zero: float = Field(
        default=0.1,
        description=(
            "All correctness metrics at or below this count as near zero; heuristic, assumes "
            "metrics roughly in [0, 1]."
        ),
    )
    min_reward: float = Field(
        default=0.5,
        description="Scalar reward must still reach this for the rollout to fire (heuristic).",
    )


class DegenerateRepetitionConfig(BaseModel):
    """Configuration for the degenerate_repetition detector.

    All defaults are conservative heuristics; none come from a paper.
    """

    min_reward: float = Field(
        default=0.7,
        description=(
            "Only rollouts with reward at or above this are inspected; heuristic, assumes "
            "rewards roughly in [0, 1]."
        ),
    )
    min_tokens: int = Field(
        default=30,
        ge=1,
        description="Minimum word tokens in the completion before ratios are computed (heuristic).",
    )
    ngram_n: int = Field(
        default=3, ge=2, description="n for the n-gram repetition ratio (heuristic)."
    )
    max_distinct_ratio: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Distinct-token ratio at or below which the rollout fires (heuristic).",
    )
    min_ngram_repetition: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="n-gram repetition ratio at or above which the rollout fires (heuristic).",
    )


class AnswerLeakageEchoConfig(BaseModel):
    """Configuration for the answer_leakage_echo detector.

    All defaults are conservative heuristics; none come from a paper.
    """

    min_answer_chars: int = Field(
        default=8,
        ge=1,
        description=(
            "Normalized answers shorter than this are never treated as echoes, to avoid "
            "trivial matches on short answers (heuristic)."
        ),
    )
    max_extra_chars: int = Field(
        default=64,
        ge=0,
        description=(
            "Maximum non-answer characters in the normalized completion for the echo to count "
            "as 'no work shown' (heuristic)."
        ),
    )
    criteria_key_patterns: list[str] = Field(
        default_factory=lambda: list(_DEFAULT_CRITERIA_KEY_PATTERNS),
        description=(
            "Case-insensitive substrings identifying reward-criteria keys in info "
            "(heuristic defaults)."
        ),
    )
    criteria_line_regex: str = Field(
        default=_DEFAULT_CRITERIA_LINE_REGEX,
        description=(
            "Regex marking a prompt line as a reward criterion candidate (heuristic default)."
        ),
    )
    min_criterion_chars: int = Field(
        default=24,
        ge=1,
        description=(
            "Normalized criteria shorter than this are never checked for verbatim echoes "
            "(heuristic)."
        ),
    )
    answer_echo_score: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Score for an answer echo verdict (heuristic)."
    )
    criterion_echo_score: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Score for a criterion echo verdict (heuristic)."
    )


class DetectorConfig(BaseModel):
    """Top-level detector configuration: one sub-model per built-in detector.

    Every threshold is configurable; the defaults are conservative heuristics
    (see each sub-model). Third-party detectors may read their own settings
    from their own config objects; this model only carries the built-ins.
    """

    # Plain instance defaults: pydantic v2 deep-copies mutable defaults per
    # instantiation, and mypy strict types them cleanly (a model class passed
    # as default_factory does not).
    verifier_tamper: VerifierTamperConfig = VerifierTamperConfig()
    reward_saturation_group_collapse: RewardSaturationGroupCollapseConfig = (
        RewardSaturationGroupCollapseConfig()
    )
    length_inflation: LengthInflationConfig = LengthInflationConfig()
    format_only_wins: FormatOnlyWinsConfig = FormatOnlyWinsConfig()
    degenerate_repetition: DegenerateRepetitionConfig = DegenerateRepetitionConfig()
    answer_leakage_echo: AnswerLeakageEchoConfig = AnswerLeakageEchoConfig()


@runtime_checkable
class Detector(Protocol):
    """What a detector must implement to be registrable.

    ``name`` is the stable detector id (matches the entry-point name for the
    built-ins), ``category`` is the taxonomy category every verdict carries,
    and ``detect`` is a pure function over normalized rollouts.
    """

    @property
    def name(self) -> str:
        """Stable detector id."""
        ...

    @property
    def category(self) -> str:
        """Taxonomy category for verdicts from this detector."""
        ...

    def detect(self, rollouts: Sequence[Rollout], config: DetectorConfig) -> list[Verdict]:
        """Run the detector over normalized rollouts and return verdicts."""
        ...


def load_detectors(group: str = DETECTOR_ENTRY_POINT_GROUP) -> dict[str, Detector]:
    """Discover and instantiate all detectors registered under ``group``.

    Input: the entry-point group name. Output: a dict mapping entry-point name
    to a detector instance. A broken plugin (import error, instantiation
    error) is skipped with a logged warning and never crashes discovery.
    """
    discovered: dict[str, Detector] = {}
    for ep in entry_points(group=group):
        try:
            loaded = ep.load()
        except Exception as exc:  # one broken plugin must not crash discovery
            logger.warning("skipping detector entry point %r: failed to load (%s)", ep.name, exc)
            continue
        try:
            detector = loaded() if isinstance(loaded, type) else loaded
        except Exception as exc:  # same policy for instantiation
            logger.warning(
                "skipping detector entry point %r: failed to instantiate (%s)", ep.name, exc
            )
            continue
        discovered[ep.name] = cast(Detector, detector)
    return discovered


def builtin_detectors() -> dict[str, Detector]:
    """Return instances of the six built-in detectors without entry points.

    Plain-code fallback for environments where entry-point metadata is
    genuinely unavailable (for example a vendored copy without dist-info).
    Prefer :func:`load_detectors`; built-ins normally arrive through the same
    entry-point mechanism as third-party detectors.
    """
    detectors: dict[str, Detector] = {}
    for name, module_name, attr in _BUILTIN_SPECS:
        module = importlib.import_module(module_name)
        detectors[name] = cast(Detector, getattr(module, attr)())
    return detectors


def discover_detectors(group: str = DETECTOR_ENTRY_POINT_GROUP) -> dict[str, Detector]:
    """Return all discoverable detectors, preferring entry points.

    Falls back to :func:`builtin_detectors` (with a logged warning) only when
    entry-point discovery yields nothing, which indicates missing install
    metadata rather than an intentionally empty registry.
    """
    detectors = load_detectors(group)
    if not detectors:
        logger.warning(
            "no detector entry points found for group %r; falling back to built-ins", group
        )
        return builtin_detectors()
    return detectors
