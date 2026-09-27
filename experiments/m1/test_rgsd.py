"""RGSD pieces that run offline (no GPU, no model): teacher prompt and the top-k JSD loss.

uv run pytest m1/test_rgsd.py -q
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

import torch
from data import Criterion, rubric_prompt
from train_rgsd import rubric_context, teacher_topk_jsd


def test_teacher_prompt_is_the_rgsd_template():
    prompt = [{"role": "user", "content": "What causes gout?"}]
    crit = [Criterion("Mentions uric acid.", 5.0), Criterion("Mentions diet.", 2.0)]
    full = rubric_prompt(prompt, crit)[0]["content"]
    assert full.startswith("What causes gout?\n\nHidden evaluation criteria")
    assert "1. Mentions uric acid.\n2. Mentions diet." in full
    assert "5.0" not in full  # weights are not shown to the teacher
    assert f"What causes gout?\n\n{rubric_context(prompt, crit)}" == full


def _case(same: bool):
    torch.manual_seed(0)
    teacher = torch.randn(2, 5, 50)
    student = teacher.clone() if same else torch.randn(2, 5, 50)
    t_top, t_ids = torch.topk(torch.log_softmax(teacher, -1), k=8, dim=-1)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    return student, t_top, t_ids, mask


def test_loss_is_zero_when_student_matches_teacher():
    loss, mean = teacher_topk_jsd(*_case(same=True), beta=0.5)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    assert mean.item() == pytest.approx(0.0, abs=1e-6)


def test_loss_is_positive_and_ignores_masked_positions():
    student, t_top, t_ids, mask = _case(same=False)
    loss, _ = teacher_topk_jsd(student, t_top, t_ids, mask, beta=0.5)
    assert loss.item() > 0
    student2 = student.clone()
    student2[0, 3:] += 100.0  # masked positions of answer 0
    loss2, _ = teacher_topk_jsd(student2, t_top, t_ids, mask, beta=0.5)
    assert loss2.item() == pytest.approx(loss.item(), rel=1e-6)
