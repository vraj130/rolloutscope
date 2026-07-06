# rolloutscope

Offline rollout and reward debugger for the verifiers / prime-rl RL ecosystem. It
ingests on-disk rollout artifacts (results.jsonl, metadata.json,
train_rollouts.jsonl), normalizes them into a versioned schema, runs reward-hacking
detectors over them, and emits a terminal summary, a JSON findings file, and a
single-file self-contained HTML report.

Full documentation lands with v0.1.0. See PLAN.md and PROGRESS.md for build status.
