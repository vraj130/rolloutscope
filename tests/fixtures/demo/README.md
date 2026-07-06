# Demo run fixture

Everything here is SYNTHETIC and hand-built. No row came from a real model, a
real evaluation, or any external dataset. The directory is shaped exactly like a
verifiers `evaluate(save_results=True)` run (a `results.jsonl` of trace rows plus
a `metadata.json` manifest), so `rolloutscope analyze tests/fixtures/demo`
resolves the verifiers-eval adapter and produces the terminal summary, the JSON
sidecar, and the self-contained HTML report.

The 10 rows mix reward-hacking patterns with clean rollouts so the report has a
realistic reward distribution and at least four detectors fire:

- example 10: `verifier_tamper`, a completion that disables a test with
  `pytest.mark.skip` plus `assert True`.
- example 11: `verifier_tamper`, a multi-turn rollout that edits
  `tests/test_payment.py` to `sys.exit(0)` through a tool call.
- example 20: `degenerate_repetition`, a high-reward completion that repeats one
  sentence a dozen times.
- example 30: `answer_leakage_echo`, a completion that echoes the ground-truth
  `answer` verbatim with no work shown.
- example 31: `answer_leakage_echo`, a completion that copies the
  `info["grading_criteria"]` string verbatim.
- examples 40 and 41: `format_only_wins`, `format_reward` near 1.0 while
  `correct_answer` is 0.0 and the scalar reward still clears 0.5.
- examples 50, 51, 52: clean rollouts (a correct answer, a wrong-but-honest
  answer, and an honest non-attempt) to fill out the reward histogram.

Regeneration: edit the JSONL directly. Keep the rows valid verifiers trace shape
(no `kind`, `schema_version`, or id fields; the adapter attaches those) so the
fixture keeps exercising the real adapter path.
