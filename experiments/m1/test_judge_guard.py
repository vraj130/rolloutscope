"""A dead proxy judge must stop training, never let it continue on unscored rollouts.

Runs offline, with no GPU: the "dead proxy" is a local port with nothing listening.

    uv run pytest m1/test_judge_guard.py -q
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from judge import FailureGuard, JudgeFailure, ProxyJudge, check_health
from train import RolloutLog, make_reward_fn

HERE = Path(__file__).parent


def dead_url() -> str:
    """A localhost URL on a port nothing listens on."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return f"http://127.0.0.1:{port}/v1"


def test_dead_proxy_mid_run_trips_guard(tmp_path):
    """The real reward path against a dead proxy yields None rewards, and the guard stops."""
    judge = ProxyJudge(dead_url(), "proxy-judge", max_retries=2, timeout=2.0)
    log = RolloutLog(tmp_path, eos_ids={0})
    reward_fn = make_reward_fn(judge, log)
    prompt = [{"role": "user", "content": "What is hypertension?"}]
    criteria = [{"criterion": "Defines hypertension.", "weight": 5.0}]
    n = 4
    rewards = asyncio.run(
        reward_fn(
            prompts=[prompt] * n,
            completions=[[{"role": "assistant", "content": "High blood pressure."}]] * n,
            completion_ids=[[5, 6, 0]] * n,
            example_id=[0] * n,
            criteria=[criteria] * n,
        )
    )
    assert rewards == [None] * n
    assert all(r["reward"] is None and r["metrics"]["judge_failed"] == 1.0 for r in log.rows)
    summary = log.flush(step=1)
    with pytest.raises(JudgeFailure, match="all 4 proxy calls failed"):
        FailureGuard(0.10, 3).check(1, summary["proxy_calls"], summary["proxy_failures"])


def test_sustained_failure_rate_trips_after_patience():
    """Above 10% for 3 consecutive steps stops; a clean step resets the streak."""
    guard = FailureGuard(max_rate=0.10, patience=3)
    guard.check(1, 64, 10)  # 15.6%
    guard.check(2, 64, 10)
    guard.check(3, 64, 6)  # 9.4%, resets
    guard.check(4, 64, 10)
    guard.check(5, 64, 10)
    with pytest.raises(JudgeFailure, match="3 consecutive steps"):
        guard.check(6, 64, 10)


def test_health_check_fails_on_dead_proxy():
    with pytest.raises(JudgeFailure, match="not reachable"):
        check_health(dead_url(), timeout=2.0)


def test_train_refuses_to_start_without_proxy(tmp_path):
    """train.py exits nonzero before loading any model, and creates no run directory."""
    env = {**os.environ, "PROXY_URL": dead_url(), "ROLLOUTSCOPE_DATA": str(tmp_path)}
    proc = subprocess.run(
        [sys.executable, str(HERE / "train.py"), "--config", str(HERE / "config.yaml")],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode != 0
    assert "not starting: proxy judge not reachable" in proc.stderr
    assert not (tmp_path / "m1").exists()
