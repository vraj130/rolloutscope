"""M2 run C: Rubric-Guided Self-Distillation (RGSD, arXiv 2606.12507), no judge in the loop.

The student (Qwen2.5-1.5B-Instruct, full fine-tune, GPU 0) samples one answer per prompt with
the plain prompt. A frozen copy of the base model on GPU 1 reads the same answer with the RGSD
teacher prompt (question plus rubric, data.rubric_prompt) and gives a target distribution at
every answer token. The loss is the generalized Jensen-Shannon divergence at beta = 0.5 over the
teacher's top-128 tokens (both distributions renormalized on that support), averaged over the
answer's tokens and then over answers.

Built on TRL's SDFTTrainer (experimental) with three changes: the teacher sits on GPU 1, the
top-k support comes from the teacher (TRL takes it from the student), and there is no clipping.
The paper's JSD clip tau = 0.05 is not defined in the paper, so it is skipped (user decision
2026-09-27); TRL's importance-sampling clip does not apply (one optimizer step per generation).

Only evaluation generations are logged (eval/step_<n>.jsonl, same prompts and schedule as A
and B). The proxy judge is not running during training (GPU 1 holds the teacher), so eval rows
are written ungraded and regrade_eval_proxy.py grades them afterwards with the same judge.

    CUDA_VISIBLE_DEVICES=0,1 uv run python m1/train_rgsd.py --config m1/config_c_s1.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from data import rubric_prompt, split
from train import RolloutLog, append_jsonl, build_callbacks, run_dir

TEACHER_DEVICE = "cuda:1"


def rubric_context(prompt, criteria) -> str:
    """The text SDFTTrainer's template appends after the question: RGSD's rubric block."""
    full = rubric_prompt(prompt, criteria)[0]["content"]
    return full[len(prompt[0]["content"]) + 2 :]  # drop the question and the blank line


def teacher_topk_jsd(student_logits, teacher_top, teacher_ids, mask, beta: float):
    """Per-answer loss: generalized JSD on the teacher's top-k support, token mean, answer mean.

    ``teacher_top`` and ``teacher_ids`` are the teacher's top-k log-probs and token ids per
    position. Both distributions are renormalized on that support (no tail bucket). Returns
    (loss, mean per-token divergence).
    """
    import torch
    from trl.experimental.sdft.loss_utils import compute_divergence

    s_top = torch.gather(torch.log_softmax(student_logits.float(), dim=-1), -1, teacher_ids)
    s_top = s_top - torch.logsumexp(s_top, dim=-1, keepdim=True)
    t_top = teacher_top - torch.logsumexp(teacher_top, dim=-1, keepdim=True)
    per_token = compute_divergence(s_top, t_top, beta)
    loss = ((per_token * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
    return loss, (per_token * mask).sum() / mask.sum().clamp(min=1.0)


def make_trainer_class():
    """RGSDTrainer, built lazily so importing this module does not import torch or TRL."""
    import torch
    from trl.experimental.sdft import SDFTTrainer
    from trl.trainer.utils import create_model_from_path

    class RGSDTrainer(SDFTTrainer):
        """SDFTTrainer with a GPU 1 teacher and top-k support taken from the teacher."""

        def _setup_teacher_model(self) -> None:
            kwargs = dict(self.args.model_init_kwargs or {})
            kwargs["device_map"] = {"": TEACHER_DEVICE}
            self.teacher_model = create_model_from_path(self.model.config._name_or_path, **kwargs)
            self.teacher_model.requires_grad_(False)
            self.teacher_model.eval()

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            completion_ids = inputs["completion_ids"]
            mask = inputs["completion_mask"]
            keep = completion_ids.size(1)
            student_logits = self._forward_logits(
                model,
                torch.cat([inputs["prompt_ids"], completion_ids], dim=1),
                torch.cat([inputs["prompt_mask"], mask], dim=1),
                keep,
            )
            with torch.no_grad():
                teacher_logits = self._forward_logits(
                    self.teacher_model,
                    inputs["teacher_input_ids"].to(TEACHER_DEVICE),
                    inputs["teacher_attention_mask"].to(TEACHER_DEVICE),
                    keep,
                )
                t_logp = torch.log_softmax(teacher_logits.float(), dim=-1)
                t_top, t_ids = torch.topk(t_logp, k=self.args.distillation_topk, dim=-1)
                t_top, t_ids = t_top.to(student_logits.device), t_ids.to(student_logits.device)
                del teacher_logits, t_logp
            loss, mean = teacher_topk_jsd(
                student_logits, t_top, t_ids, mask, self.args.distillation_alpha
            )
            self._log_self_distillation_metric("train" if model.training else "eval", mean.item())
            scale = self.current_gradient_accumulation_steps if model.training else 1.0
            return loss / scale

    return RGSDTrainer


def main() -> None:
    """Build the split and the RGSD trainer, then train."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int, default=None, help="override, for smoke runs")
    parser.add_argument("--run-name", default=None, help="override run_name")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
    if args.run_name is not None:
        cfg["run_name"] = args.run_name

    out = run_dir(cfg)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; pick another run_name")
    ckpt = Path(os.path.expandvars(cfg["checkpoint_dir"])).expanduser() / cfg["run_name"]
    if ckpt.exists() and any(ckpt.iterdir()):
        raise SystemExit(f"{ckpt} is not empty; pick another run_name")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, TrainerCallback
    from trl.experimental.sdft import SDFTConfig

    train_set, eval_set = split(cfg["n_train"], cfg["n_eval"], cfg.get("split_seed", cfg["seed"]))
    (out / "split.json").write_text(
        json.dumps(
            {
                "train_source_index": [e.source_index for e in train_set],
                "eval_source_index": [e.source_index for e in eval_set],
            }
        )
    )
    ds = Dataset.from_list(
        [
            {"prompt": e.prompt, "privileged_context": rubric_context(e.prompt, e.criteria)}
            for e in train_set
        ]
    )
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    eos_ids = {i for i in (tok.eos_token_id, tok.pad_token_id) if i is not None}

    per_step = cfg["prompts_per_step"] * cfg["num_generations"]
    sdft_args = SDFTConfig(
        output_dir=str(ckpt),
        seed=cfg["seed"],
        max_steps=cfg["max_steps"],
        learning_rate=cfg["lr"],
        per_device_train_batch_size=cfg["micro_batch_size"],
        gradient_accumulation_steps=per_step // cfg["micro_batch_size"],
        num_generations=cfg["num_generations"],
        max_prompt_length=None,  # the teacher prompt carries the whole rubric; never truncate it
        max_completion_length=cfg["max_completion_length"],
        temperature=cfg["temperature"],
        distillation_mode="topk_logits",
        distillation_topk=cfg["distillation_topk"],
        distillation_alpha=cfg["distillation_beta"],
        distillation_is_clip=None,
        teacher_model_kind="base",
        teacher_prompt_template="{prompt}\n\n{privileged_context}",
        bf16=True,
        model_init_kwargs={"dtype": "bfloat16"},
        gradient_checkpointing=cfg["gradient_checkpointing"],
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=cfg["vllm_gpu_memory_utilization"],
        vllm_max_model_length=cfg["vllm_max_model_length"],
        logging_steps=1,
        save_strategy="steps",
        save_steps=cfg["save_every"],
        save_only_model=True,
        report_to="none",
    )

    _, evaluator, log_writer = build_callbacks(cfg, out, RolloutLog(out, eos_ids), None, eval_set)

    class StepWriter(TrainerCallback):
        def on_step_begin(self, args, state, control, **kw):
            self.t0 = time.monotonic()

        def on_step_end(self, args, state, control, **kw):
            row = {
                "step": state.global_step,
                "step_seconds": round(time.monotonic() - self.t0, 3),
                "gpu0_max_reserved_gib": round(torch.cuda.max_memory_reserved(0) / 2**30, 2),
                "gpu1_max_reserved_gib": round(torch.cuda.max_memory_reserved(1) / 2**30, 2),
            }
            append_jsonl(out / "stats.jsonl", row)
            print(f"[step {state.global_step}] {row['step_seconds']:.1f}s", flush=True)

    trainer = make_trainer_class()(
        model=cfg["model"],
        args=sdft_args,
        train_dataset=ds,
        processing_class=tok,
        callbacks=[StepWriter(), evaluator, log_writer],
    )
    evaluator.trainer = trainer
    t0 = time.monotonic()
    trainer.train()
    final = {"status": "completed", "wall_seconds": round(time.monotonic() - t0, 1)}
    (out / "final.json").write_text(json.dumps(final, indent=2))
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
