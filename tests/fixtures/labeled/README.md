# Labeled detector fixtures

Everything here is SYNTHETIC and hand-designed, deterministically, for detector
tests. No row came from a real model, a real evaluation run, or any external
dataset. Rows are in the normalized rolloutscope schema (kind, schema_version,
example_id, prompt, completion, reward, metrics, and so on); some rows omit
rollout_id or group_id on purpose to exercise the stable-id fallbacks.

Each detector has one `<detector>_hacked.jsonl` / `<detector>_clean.jsonl`
pair. The pairs are also designed to be globally clean: a clean file for one
detector should not fire any other detector either, so the fixture-metrics
test can count false positives across all clean files.

- `verifier_tamper_*`: hacked rows insert `pytest.mark.skip` plus `assert True`
  into a test, and use an edit tool to overwrite `tests/test_payment.py` with
  `sys.exit(0)` (one single-turn, one multi-turn row). Clean rows fix real
  source code (`src/payment.py`) and report honest test results.
- `reward_saturation_group_collapse_*`: hacked has 3 steps x 3 groups x 3
  rollouts; the dead-group (zero reward variance) fraction rises 1/3 to 2/3 to
  3/3 while the shared reward saturates at 1.0 and the independent
  `correct_answer` metric stays flat at 0.3. Clean has the same layout over 2
  steps with healthy within-group reward variance. Both files omit rollout_id
  and group_id to exercise the content-hash and example_id fallbacks.
- `length_inflation_*`: 10 rollouts each with step_index 0..9 and completions
  padded with distinct vocabulary words. Hacked rewards rise monotonically
  with length while `correct_answer` stays flat at 0.5; clean rewards are a
  fixed permutation uncorrelated with length, and `correct_answer` tracks the
  reward.
- `format_only_wins_*`: hacked rows have `format_reward` near 1.0,
  `correct_answer` near 0.0, and scalar reward still 0.6 to 0.7. Clean rows
  are either genuinely correct, format-only but with a near-zero reward, or
  correct with broken formatting.
- `degenerate_repetition_*`: hacked rows are high-reward completions built by
  repeating one sentence or one bigram dozens of times (extreme n-gram
  repetition, tiny distinct-token ratio). Clean rows are high-reward varied
  prose of similar length.
- `answer_leakage_echo_*`: hacked row 1 echoes the long ground-truth answer
  with no other work; hacked row 2 copies the `info["grading_criteria"]`
  string verbatim into the completion. Clean rows show real work before
  stating the answer, or satisfy the criteria without quoting them.

Regeneration: the files were produced by a small deterministic script (fixed
strings, fixed reward tables, no randomness); edit the JSONL directly if a row
needs to change, keeping the pairs separable and globally clean.
