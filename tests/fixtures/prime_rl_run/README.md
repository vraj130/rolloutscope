# prime_rl_run fixture

SYNTHETIC and hand-written, deterministically, for tests. No row came from a
real model or a real training run. This is a fake prime-rl orchestrator
training-run directory following the pinned layout (prime-rl @ `df2acf48`):
each step directory carries a `train_rollouts.jsonl` whose rows are plain
verifiers-trace-shaped objects (same shape as eval `results.jsonl`, verifiers @
`5885ab9c`).

Layout:

- `step_0/train_rollouts.jsonl`: 3 rows (example 0 has 2 rollouts, example 1
  has 1). The first row is deliberately minimal-but-complete and is used as the
  full golden-dump row in tests. Rows carry a `env_seed` state_column extra to
  exercise unknown-key passthrough.
- `step_1/train_rollouts.jsonl`: 2 rows (one per example). The example 1 row is
  multi-turn with a one-step `trajectory` to prove multi_turn routing through
  the prime-rl adapter.

The two step directories prove step_index attachment (0 and 1) and numeric
ordering. Per rule 9, no row carries `advantages`, `is_trainable`, or the other
in-memory orchestration metadata (`kind`, `env_name`, `group_id`,
`policy_version`): prime-rl excludes those from the on-disk dump.
