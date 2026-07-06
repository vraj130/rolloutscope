"""Reward-hacking detectors: pure functions over normalized rollouts.

Each detector is discoverable through the ``rolloutscope.detectors``
entry-point group (see :mod:`rolloutscope.detectors.base`); the classes are
also re-exported here for direct use. Detectors depend only on the normalized
schema, never on verifiers or prime-rl.
"""

from rolloutscope.detectors.answer_leakage_echo import AnswerLeakageEchoDetector
from rolloutscope.detectors.base import (
    DETECTOR_ENTRY_POINT_GROUP,
    AnswerLeakageEchoConfig,
    DegenerateRepetitionConfig,
    Detector,
    DetectorConfig,
    FormatOnlyWinsConfig,
    LengthInflationConfig,
    RewardSaturationGroupCollapseConfig,
    VerifierTamperConfig,
    builtin_detectors,
    discover_detectors,
    load_detectors,
)
from rolloutscope.detectors.degenerate_repetition import DegenerateRepetitionDetector
from rolloutscope.detectors.format_only_wins import FormatOnlyWinsDetector
from rolloutscope.detectors.length_inflation import LengthInflationDetector
from rolloutscope.detectors.reward_saturation_group_collapse import (
    RewardSaturationGroupCollapseDetector,
)
from rolloutscope.detectors.verifier_tamper import VerifierTamperDetector

__all__ = [
    "DETECTOR_ENTRY_POINT_GROUP",
    "AnswerLeakageEchoConfig",
    "AnswerLeakageEchoDetector",
    "DegenerateRepetitionConfig",
    "DegenerateRepetitionDetector",
    "Detector",
    "DetectorConfig",
    "FormatOnlyWinsConfig",
    "FormatOnlyWinsDetector",
    "LengthInflationConfig",
    "LengthInflationDetector",
    "RewardSaturationGroupCollapseConfig",
    "RewardSaturationGroupCollapseDetector",
    "VerifierTamperConfig",
    "VerifierTamperDetector",
    "builtin_detectors",
    "discover_detectors",
    "load_detectors",
]
