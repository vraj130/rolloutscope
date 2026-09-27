"""Rubric Dropout (M2 run B): one mask per prompt group and step, reward on kept criteria.

Runs offline, with no GPU and no proxy: the judge is a stub.

    uv run pytest m1/test_rubric_dropout.py -q
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))

from judge import GradeResult, dropout_keep, score
from train import RolloutLog, make_reward_fn


def test_keep_counts_and_minimum():
    assert len(dropout_keep(0, 0, 31, 0.5)) == 16  # floor(15.5) = 15 dropped
    assert len(dropout_keep(0, 0, 8, 0.5)) == 4
    assert len(dropout_keep(0, 0, 5, 0.5)) == 3
    assert len(dropout_keep(0, 0, 4, 0.5)) == 3  # at least three kept
    assert dropout_keep(0, 0, 2, 0.5) == [0, 1]  # never more than exist


def test_mask_is_deterministic_and_changes_with_step_and_prompt():
    a = dropout_keep(7, 3, 30, 0.5)
    assert a == dropout_keep(7, 3, 30, 0.5)
    assert a != dropout_keep(7, 4, 30, 0.5)
    assert a != dropout_keep(8, 3, 30, 0.5)


class StubJudge:
    """Grades every criterion with a fixed verdict pattern."""

    def __init__(self, verdicts):
        self.verdicts = verdicts

    async def grade_many(self, items):
        return [
            GradeResult(True, self.verdicts, score(self.verdicts, c), None, 1) for _, _, c in items
        ]


def test_group_shares_mask_and_reward_uses_kept_criteria(tmp_path):
    n = 10
    verdicts = [1, 0] * (n // 2)
    criteria = [{"criterion": f"c{i}", "weight": float(i + 1)} for i in range(n)]
    log = RolloutLog(tmp_path, eos_ids={0})
    fn = make_reward_fn(StubJudge(verdicts), log, dropout=0.5)
    g = 8
    rewards = asyncio.run(
        fn(
            prompts=[[{"role": "user", "content": "q"}]] * g,
            completions=[[{"role": "assistant", "content": "a"}]] * g,
            completion_ids=[[5, 0]] * g,
            example_id=[3] * g,
            criteria=[criteria] * g,
            trainer_state=SimpleNamespace(global_step=12),
        )
    )
    keep = dropout_keep(3, 12, n, 0.5)
    kept_w = [i + 1 for i in keep]
    expected = sum(w for i, w in zip(keep, kept_w, strict=True) if verdicts[i]) / sum(kept_w)
    assert rewards == [expected] * g
    assert all(r["info"]["dropout_kept"] == keep for r in log.rows)
    full = score(verdicts, [SimpleNamespace(weight=float(i + 1)) for i in range(n)])  # type: ignore[misc]
    assert all(r["metrics"]["proxy_score"] == full for r in log.rows)


def test_no_dropout_is_the_m1_reward(tmp_path):
    criteria = [{"criterion": "a", "weight": 1.0}, {"criterion": "b", "weight": 3.0}]
    log = RolloutLog(tmp_path, eos_ids={0})
    fn = make_reward_fn(StubJudge([1, 0]), log)
    rewards = asyncio.run(
        fn(
            prompts=[[{"role": "user", "content": "q"}]],
            completions=[[{"role": "assistant", "content": "a"}]],
            completion_ids=[[5, 0]],
            example_id=[0],
            criteria=[criteria],
            trainer_state=SimpleNamespace(global_step=1),
        )
    )
    assert rewards == [0.25]
    assert log.rows[0]["info"]["dropout_kept"] is None
