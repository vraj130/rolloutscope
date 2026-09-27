"""M1: GRPO on RubricHub medical against the proxy judge (PROGRESS.md R-006, R-007).

Training and vLLM generation (colocated) run on GPU 0. The proxy judge is a separate vLLM
server on GPU 1 (serve_proxy.sh). No gold calls happen here; grade_gold.py grades the saved
evaluation generations after training.

Outputs under $ROLLOUTSCOPE_DATA/m1/<run_name>/:

- step_<n>/train_rollouts.jsonl  every rollout of optimizer step n, legacy prime-rl row shape
  (scripts/perf/generate.py). A rollout whose proxy call failed after retries is written with
  reward null and metrics.judge_failed = 1. rolloutscope skips it as invalid; it is never scored.
- eval/step_<n>.jsonl            one generation per eval prompt at step n (0 = before training),
  with criteria and proxy verdicts.
- stats.jsonl                    per-step timing and proxy judge counters.
- train_log.jsonl                the trainer's own log lines.
- config.yaml, split.json        what was run.
- final.json                     totals, and the stop reason if the judge guard tripped.

The run refuses to start if the proxy's /health check fails, and stops with exit code 3 if
every proxy call in a step fails or the step failure rate stays above
judge_max_failure_rate for judge_failure_patience consecutive steps (config.yaml).

Usage, from experiments/ with the proxy server up:

    CUDA_VISIBLE_DEVICES=0 uv run python m1/train.py --config m1/config.yaml [--max-steps 5]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).parent))
# The system nvcc (CUDA 11.8 on fourier) cannot JIT-build FlashInfer's sampler; use vLLM's
# PyTorch sampler for the colocated generation engine. Must be set before vllm is imported.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from data import Example, criteria_from_json, criteria_to_json, split
from judge import (
    FailureGuard,
    GradeResult,
    JudgeFailure,
    ProxyJudge,
    check_health,
    dropout_keep,
    proxy_from_env,
    score,
)


def run_dir(cfg: dict[str, Any]) -> Path:
    """$ROLLOUTSCOPE_DATA/m1/<run_name>."""
    root = os.environ.get("ROLLOUTSCOPE_DATA")
    if not root:
        raise SystemExit("set ROLLOUTSCOPE_DATA")
    return Path(root) / "m1" / cfg["run_name"]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append one JSON line."""
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def completion_text(completion: Any) -> str:
    """TRL passes conversational completions as a message list."""
    if isinstance(completion, list):
        return "".join(m.get("content") or "" for m in completion)
    return str(completion)


class RolloutLog:
    """Rows produced by the reward function, flushed to disk once per optimizer step."""

    def __init__(self, out: Path, eos_ids: set[int]) -> None:
        self.out = out
        self.eos_ids = eos_ids
        self.rows: list[dict[str, Any]] = []
        self.reward_seconds = 0.0
        self.calls = 0
        self.failures = 0

    def flush(self, step: int) -> dict[str, Any]:
        """Write buffered rows to step_<step>/train_rollouts.jsonl and return step counters."""
        d = self.out / f"step_{step}"
        d.mkdir(parents=True, exist_ok=True)
        with (d / "train_rollouts.jsonl").open("w") as f:
            for r in self.rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        scored = [r["reward"] for r in self.rows if r["reward"] is not None]
        proxy = [r["metrics"]["proxy_score"] for r in self.rows if "proxy_score" in r["metrics"]]
        lengths = [r["metrics"]["completion_tokens"] for r in self.rows]
        summary = {
            "rollouts": len(self.rows),
            "proxy_calls": self.calls,
            "proxy_failures": self.failures,
            "reward_seconds": round(self.reward_seconds, 3),
            "mean_reward": sum(scored) / len(scored) if scored else None,
            "mean_proxy_score": sum(proxy) / len(proxy) if proxy else None,
            "mean_completion_tokens": sum(lengths) / len(lengths) if lengths else None,
            "truncated": sum(r["is_truncated"] for r in self.rows),
        }
        self.rows, self.reward_seconds, self.calls, self.failures = [], 0.0, 0, 0
        return summary


def make_reward_fn(
    judge: ProxyJudge,
    log: RolloutLog,
    dropout: float = 0.0,
    min_keep: int = 3,
    mask_seed: int | None = None,
):
    """Async TRL reward function: one proxy call per completion, rows buffered in ``log``.

    With ``dropout`` > 0 (M2 run B, Rubric Dropout), the proxy still grades the full rubric in
    one call, and the reward is the score on the criteria kept by ``dropout_keep`` for this
    prompt and step (and ``mask_seed``, see ``dropout_keep``). The full-rubric score stays in
    metrics.proxy_score.
    """

    async def proxy_reward(prompts, completions, completion_ids, example_id, criteria, **kw):
        step = kw["trainer_state"].global_step if "trainer_state" in kw else 0
        start = time.monotonic()
        crits = [criteria_from_json(c) for c in criteria]
        texts = [completion_text(c) for c in completions]
        results: list[GradeResult] = await judge.grade_many(
            list(zip(prompts, texts, crits, strict=True))
        )
        log.reward_seconds += time.monotonic() - start
        rewards: list[float | None] = []
        for p, t, ids, ex, cr, res in zip(
            prompts, texts, completion_ids, example_id, crits, results, strict=True
        ):
            truncated = len(ids) == 0 or ids[-1] not in log.eos_ids
            metrics: dict[str, float] = {
                "n_criteria": float(len(cr)),
                "completion_tokens": float(len(ids)),
            }
            reward = res.score if res.ok else None
            keep = None
            if dropout > 0:
                keep = dropout_keep(int(ex), step, len(cr), dropout, min_keep, mask_seed)
                metrics["n_kept"] = float(len(keep))
            if res.ok:
                metrics["proxy_score"] = float(res.score)  # type: ignore[arg-type]
                metrics["n_satisfied"] = float(sum(res.verdicts))  # type: ignore[arg-type]
                if keep is not None:
                    v = res.verdicts or []
                    reward = score([v[i] for i in keep], [cr[i] for i in keep])
                    metrics["dropout_reward"] = float(reward)
            else:
                metrics["judge_failed"] = 1.0
            log.rows.append(
                {
                    "example_id": int(ex),
                    "prompt": p,
                    "completion": [{"role": "assistant", "content": t}],
                    "reward": reward,
                    "is_completed": True,
                    "is_truncated": truncated,
                    "metrics": metrics,
                    "info": {
                        "proxy_verdicts": res.verdicts,
                        "judge_error": res.error,
                        "dropout_kept": keep,
                        "train_step": step,
                    },
                }
            )
            log.calls += 1
            log.failures += 0 if res.ok else 1
            rewards.append(reward)
        return rewards

    return proxy_reward


def build_callbacks(
    cfg, out: Path, log: RolloutLog, judge: ProxyJudge | None, eval_set: list[Example]
):
    """Return (rollout writer + stats + judge guard, evaluator, trainer log) callbacks.

    With ``judge=None`` the evaluator writes every eval row as ungraded, and the rollout writer
    must not be used (train_rgsd.py uses only the evaluator and the trainer log).
    """
    from transformers import TrainerCallback

    guard = FailureGuard(
        cfg.get("judge_max_failure_rate", 0.10), cfg.get("judge_failure_patience", 3)
    )

    class StepWriter(TrainerCallback):
        def on_step_begin(self, args, state, control, **kw):
            self.t0 = time.monotonic()

        def on_step_end(self, args, state, control, **kw):
            import torch

            summary = log.flush(state.global_step)
            summary.update(
                step=state.global_step,
                step_seconds=round(time.monotonic() - self.t0, 3),
                gpu0_max_allocated_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                gpu0_max_reserved_gib=round(torch.cuda.max_memory_reserved() / 2**30, 2),
                proxy_total_calls=judge.stats.calls,
                proxy_total_failures=judge.stats.failures,
            )
            append_jsonl(out / "stats.jsonl", summary)
            print(
                f"[step {state.global_step}] {summary['step_seconds']:.1f}s "
                f"reward={summary['mean_reward']} judge={summary['reward_seconds']:.1f}s "
                f"fail={summary['proxy_failures']}/{summary['proxy_calls']}",
                flush=True,
            )
            guard.check(state.global_step, summary["proxy_calls"], summary["proxy_failures"])

    class Evaluator(TrainerCallback):
        trainer = None  # set after the trainer exists

        def _run(self, step: int) -> None:
            tr = self.trainer
            tok = tr.processing_class
            prompt_ids = [
                tok(
                    tok.apply_chat_template(e.prompt, tokenize=False, add_generation_prompt=True),
                    add_special_tokens=False,
                )["input_ids"]
                for e in eval_set
            ]
            t0 = time.monotonic()
            tr.vllm_generation.sync_weights()
            _, completion_ids, _, _ = tr.vllm_generation.generate(prompt_ids, None, 1)
            texts = tok.batch_decode(completion_ids, skip_special_tokens=True)
            gen_s = time.monotonic() - t0
            items = [(e.prompt, t, e.criteria) for e, t in zip(eval_set, texts, strict=True)]
            if (
                judge is None
            ):  # no proxy during training (run C): regrade_eval_proxy.py grades later
                results = [GradeResult(False, error="deferred: graded after training")] * len(items)
            else:
                fut = asyncio.run_coroutine_threadsafe(judge.grade_many(items), tr.async_loop)
                results = fut.result()
            path = out / "eval" / f"step_{step}.jsonl"
            path.parent.mkdir(exist_ok=True)
            with path.open("w") as f:
                for e, t, ids, res in zip(eval_set, texts, completion_ids, results, strict=True):
                    row = {
                        "step": step,
                        "source_index": e.source_index,
                        "prompt": e.prompt,
                        "response": t,
                        "completion_tokens": len(ids),
                        "is_truncated": len(ids) == 0 or ids[-1] not in log.eos_ids,
                        "criteria": criteria_to_json(e.criteria),
                        "proxy": {
                            "ok": res.ok,
                            "verdicts": res.verdicts,
                            "score": res.score,
                            "error": res.error,
                        },
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            ok = [r.score for r in results if r.ok]
            print(
                f"[eval step {step}] mean_proxy={sum(ok) / max(len(ok), 1):.4f} "
                f"failed={len(results) - len(ok)}/{len(results)} "
                f"gen={gen_s:.1f}s total={time.monotonic() - t0:.1f}s",
                flush=True,
            )

        def on_train_begin(self, args, state, control, **kw):
            self._run(0)

        def on_step_end(self, args, state, control, **kw):
            if state.global_step % cfg["eval_every"] == 0:
                self._run(state.global_step)

    class LogWriter(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            append_jsonl(out / "train_log.jsonl", {"step": state.global_step, **(logs or {})})

    return StepWriter(), Evaluator(), LogWriter()


def main() -> None:
    """Build the split, the trainer, and the callbacks, then train."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int, default=None, help="override, for smoke runs")
    parser.add_argument("--run-name", default=None, help="override run_name")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
    if args.run_name is not None:
        cfg["run_name"] = args.run_name

    judge = proxy_from_env()
    try:
        check_health(str(judge.client.base_url))
    except JudgeFailure as e:
        raise SystemExit(f"not starting: {e}") from e

    out = run_dir(cfg)
    if out.exists() and any(p.name != "proxy.log" for p in out.iterdir()):
        raise SystemExit(f"{out} is not empty; pick another run_name")
    # checkpoints go to <checkpoint_dir>/<run_name> (local disk) when set, else <run>/trainer
    ckpt = out / "trainer"
    if cfg.get("checkpoint_dir"):
        ckpt = Path(os.path.expandvars(cfg["checkpoint_dir"])).expanduser() / cfg["run_name"]
        if ckpt.exists() and any(ckpt.iterdir()):
            raise SystemExit(f"{ckpt} is not empty; pick another run_name")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    # split_seed fixes the prompt split independently of the training seed (R-009 runs use 0)
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
            {"prompt": e.prompt, "example_id": i, "criteria": criteria_to_json(e.criteria)}
            for i, e in enumerate(train_set)
        ]
    )

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    eos_ids = {i for i in (tok.eos_token_id, tok.pad_token_id) if i is not None}
    log = RolloutLog(out, eos_ids)

    per_step = cfg["prompts_per_step"] * cfg["num_generations"]
    if per_step % cfg["micro_batch_size"]:
        raise SystemExit("prompts_per_step * num_generations must divide by micro_batch_size")
    grpo_args = GRPOConfig(
        output_dir=str(ckpt),
        seed=cfg["seed"],
        max_steps=cfg["max_steps"],
        learning_rate=cfg["lr"],
        per_device_train_batch_size=cfg["micro_batch_size"],
        # one generation batch (prompts_per_step prompts) per optimizer step
        gradient_accumulation_steps=per_step // cfg["micro_batch_size"],
        num_generations=cfg["num_generations"],
        max_completion_length=cfg["max_completion_length"],
        temperature=cfg["temperature"],
        beta=cfg["beta"],
        loss_type=cfg["loss_type"],
        bf16=True,
        model_init_kwargs={"dtype": "bfloat16"},
        gradient_checkpointing=cfg["gradient_checkpointing"],
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=cfg["vllm_gpu_memory_utilization"],
        vllm_max_model_length=cfg["vllm_max_model_length"],
        logging_steps=1,
        # model weights only (no optimizer state) every save_every steps, for later activation
        # capture; written to <run>/trainer/checkpoint-<n>
        save_strategy="steps" if cfg.get("save_every") else "no",
        save_steps=cfg.get("save_every") or 500,
        save_only_model=True,
        save_total_limit=cfg.get("save_total_limit"),  # None keeps every checkpoint
        report_to="none",
    )
    peft_config = None
    if cfg["lora"]:
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"
        )

    step_writer, evaluator, log_writer = build_callbacks(cfg, out, log, judge, eval_set)
    trainer = GRPOTrainer(
        model=cfg["model"],
        reward_funcs=[
            make_reward_fn(
                judge,
                log,
                cfg.get("rubric_dropout", 0.0),
                cfg.get("rubric_dropout_min_keep", 3),
                # the training seed is in the mask hash unless the config says otherwise
                # (run B seed 1 ran without it)
                cfg["seed"] if cfg.get("rubric_dropout_seeded_mask", True) else None,
            )
        ],
        args=grpo_args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=peft_config,
        callbacks=[step_writer, evaluator, log_writer],
    )
    evaluator.trainer = trainer
    t0 = time.monotonic()
    stop_reason = None
    try:
        trainer.train()
    except JudgeFailure as e:
        stop_reason = str(e)
    s = judge.stats
    final = {
        "status": "stopped" if stop_reason else "completed",
        "stop_reason": stop_reason,
        "wall_seconds": round(time.monotonic() - t0, 1),
        "proxy_calls": s.calls,
        "proxy_failures": s.failures,
        "proxy_failure_rate": s.failure_rate,
        "proxy_retries": s.retries,
        "proxy_errors": s.errors,
    }
    (out / "final.json").write_text(json.dumps(final, indent=2))
    print(json.dumps(final, indent=2))
    if stop_reason:
        print(f"STOPPED: {stop_reason}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
