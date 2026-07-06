# Test fixtures

Everything in this tree is SYNTHETIC and hand-written, deterministically, for
tests. No row came from a real model or a real evaluation run. Shapes follow the
verifiers-ground-truth skill's `references/on-disk-format.md` (verifiers @
`5885ab9c`, prime-rl @ `df2acf48`).

Contents:

- `verifiers_eval_run/`: a fake `evaluate(save_results=True)` output directory.
  `results.jsonl` holds 5 single-turn rows over 2 examples (example 0 has 3
  rollouts, example 1 has 2: the metadata says `rollouts_per_example: 3`, and the
  missing sixth row stands in for a rollout dropped on error, which real appended
  files exhibit). Rows carry arbitrary state_columns extras (`dataset_split`,
  `judge_notes`, `sampler_seed`) and provider extras inside messages (`refusal`)
  to exercise `extra="allow"` round-trips. `metadata.json` is a plausible
  GenerateMetadata-shaped manifest.
- `multi_turn_rollout.jsonl`: one ToolEnv-shaped multi-turn row with a populated
  two-step `trajectory`, `tool_defs`, and a `stop_condition`.

Later phases add `labeled/` (per-detector hacked/clean pairs, Phase 4) and
`demo/` (the README quickstart run, Phase 6).
