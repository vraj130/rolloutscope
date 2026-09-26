# M1: GRPO against a rubric proxy judge

Scripts for PLAN.md milestone M1. Decisions R-006 to R-008 in PROGRESS.md fix the setup.

| File | Role |
|---|---|
| `config.yaml` | model, N_train, N_eval, K, lr, max steps, seed, LoRA on or off |
| `data.py` | RubricHub_v1 medical rows, seeded split by prompt |
| `judge.py` | shared grading prompt, async proxy client, score, failure guard |
| `serve_proxy.sh` | vLLM server for the proxy judge on GPU 1, detached |
| `train.py` | TRL GRPOTrainer on GPU 0, rollout and eval callbacks |
| `grade_gold.py` | gpt-6-luna gold grading of saved eval generations, direct or Batch API |
| `regrade_eval_proxy.py` | regrade eval rows whose proxy call timed out during training, same judge |
| `test_judge_guard.py` | a dead proxy stops training (`uv run pytest m1 -q`, no GPU) |

## Setup

```bash
cd experiments
uv sync                                     # pinned in pyproject.toml and uv.lock (+ dev: pytest)
export ROLLOUTSCOPE_DATA=/mnt/NAS/data/vg2097/rolloutscope-data
export HF_HOME=$ROLLOUTSCOPE_DATA/hf-cache
export HF_TOKEN_PATH=$HOME/.cache/huggingface/token   # gated Llama download needs the token
```

`OPENAI_API_KEY` is read from the environment or the repo root `.env` (gitignored).

## Run

```bash
RUN=dryrun-qwen1.5b-s0   # must match run_name in the config

# 1. proxy judge on GPU 1, detached; logs to $ROLLOUTSCOPE_DATA/m1/$RUN/proxy.log
bash m1/serve_proxy.sh $RUN
until curl -sf 127.0.0.1:8001/health; do sleep 10; done

# 2. training on GPU 0, detached the same way (5-step smoke run first, with --max-steps 5
#    --run-name smoke). Refuses to start if /health fails; exits 3 if the judge dies.
setsid nohup env CUDA_VISIBLE_DEVICES=0 uv run python m1/train.py --config m1/config.yaml \
  > $ROLLOUTSCOPE_DATA/m1/$RUN.train.log 2>&1 < /dev/null &

# 3. gold grading after training (asks before spending); --direct skips the Batch API
uv run python m1/grade_gold.py --run $RUN --pilot 5 [--direct] [--max-usd 6]

# 4. black-box read of the rollouts
cd .. && uv run rolloutscope analyze $ROLLOUTSCOPE_DATA/m1/$RUN --out /tmp/m1.html
```

For a real run, copy `config.yaml`, change `run_name`, `model`, `n_train`, `max_steps`, `seed`, and
`lora`, and pass it with `--config`. A run refuses to write into a non-empty directory (a
`proxy.log` from step 1 is allowed).

## Outputs, under `$ROLLOUTSCOPE_DATA/m1/<run_name>/` (never committed)

- `step_<n>/train_rollouts.jsonl`: every rollout of optimizer step n, legacy prime-rl row shape.
  `metrics` holds `proxy_score`, `n_criteria`, `n_satisfied`, `completion_tokens`; `info` holds
  the per-criterion proxy verdicts.
- `eval/step_<n>.jsonl`: one generation per eval prompt at step 0 and every K steps, with
  criteria and proxy verdicts. `eval/step_<n>.gold.jsonl` holds the gold verdicts.
- `gold/summary.tsv`: step, mean proxy, mean gold, gap, mean length. `gold/cost.json`: tokens
  and USD, with reasoning tokens separate. `../gold_spend.jsonl` logs spend across all runs.
- `trainer/checkpoint-<n>/`: policy weights (no optimizer state) every `save_every` steps.
- `stats.jsonl` (per-step time, judge time, failures), `train_log.jsonl`, `final.json` (status,
  and the stop reason if the judge guard tripped), `proxy.log`.

## Judge failures

A proxy call has a 30 s timeout and 3 attempts. If it still fails, the rollout gets
`reward: null` and `metrics.judge_failed = 1`: TRL gives it zero advantage, and
`rolloutscope analyze` rejects it and counts it in `execution.ingestion.rejected`. Training
stops with exit code 3 if every proxy call in a step fails, or if the step failure rate is above
`judge_max_failure_rate` (0.10) for `judge_failure_patience` (3) consecutive steps. A gold
request that fails after its retries stays ungraded. Neither is ever scored. Failure counts
are in `final.json` and `gold/cost.json`.

Note: the eval set is a held-out RubricHub medical split, not HealthBench-Hard as PLAN.md M1
describes.
