"""Shared builders for analysis and report tests.

Verdicts and Findings are built by hand against the frozen schema (the
detector sub-agent's output is not a dependency of these tests).
"""

from collections.abc import Callable

import pytest

from rolloutscope.schema import EvidenceSpan, SingleTurnRollout, Verdict

RolloutFactory = Callable[..., SingleTurnRollout]
VerdictFactory = Callable[..., Verdict]


@pytest.fixture
def make_rollout() -> RolloutFactory:
    """Factory for minimal single-turn rollouts with overridable fields."""

    def _make(
        reward: float,
        example_id: int = 0,
        *,
        prompt: str = "prompt text",
        completion: str = "completion text",
        group_id: str | None = None,
        step_index: int | None = None,
        is_completed: bool = True,
        is_truncated: bool = False,
        rollout_id: str | None = None,
    ) -> SingleTurnRollout:
        return SingleTurnRollout(
            example_id=example_id,
            reward=reward,
            prompt=prompt,
            completion=completion,
            group_id=group_id,
            step_index=step_index,
            is_completed=is_completed,
            is_truncated=is_truncated,
            rollout_id=rollout_id,
        )

    return _make


@pytest.fixture
def make_verdict() -> VerdictFactory:
    """Factory for hand-built verdicts; fired verdicts get one evidence span."""

    def _make(
        *,
        detector: str = "verifier_tamper",
        fired: bool = True,
        score: float = 0.9,
        category: str = "verifier_tampering",
        text: str = "offending span",
        field: str = "completion",
        note: str | None = None,
        rollout_ids: list[str] | None = None,
    ) -> Verdict:
        ids = rollout_ids if rollout_ids is not None else ["r0000000000000001"]
        evidence = (
            [EvidenceSpan(rollout_id=ids[0], field=field, text=text, note=note)] if fired else []
        )
        return Verdict(
            detector=detector,
            fired=fired,
            score=score,
            category=category,
            evidence=evidence,
            rollout_ids=ids,
        )

    return _make
