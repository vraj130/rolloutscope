# The `all` and `effective` cohorts

prime-rl distinguishes two populations of episodes at every training step. Every
episode the orchestrator receives is in the `all` cohort. The subset that
survived filtering and was actually shipped to the trainer is the `effective`
cohort. The difference between them is the filter, and the filter is exactly the
thing an observability tool must not silently erase.

How the two are recorded changed on 2026-09-01, between prime-rl v0.9.1.dev28
and v0.9.1.dev29. Both shapes are documented here because artifacts from both
exist on disk in the wild, and a reader that assumes one will misread the other.

Everything below is from upstream source at the pinned commits, with captured
artifacts in `tests/fixtures/upstream/`.

## The legacy shape: two directories, one episode written twice

Through v0.9.1.dev28 (`84e7312f`), `FileMonitor.log_episodes` wrote:

```
output_dir / "rollouts" / f"step_{step}" / kind / subset / "traces.jsonl"
```

with `kind` in `{train, eval}` and `subset` in `{all, effective}`. The
orchestrator called it twice for the same episodes, from two places:

- `monitors.log([episode], step, kind, "all")`, once per episode as it arrived,
  so a run could be tailed live.
- `monitors.log(effective.vf_episodes, step, "train", "effective")`, once per
  step for the filtered batch at finalize.

**An episode that survived filtering is therefore on disk twice**, once under
`all/` and once under `effective/`, as two byte-identical records with the same
`episode.id` and the same trace ids.

The consequence for a reader: a recursive discovery that globs `**/traces.jsonl`
and concatenates counts every effective episode twice. That does not merely
inflate a row count. It doubles the weight of the shipped population in every
aggregate, which biases reward means toward whatever the filter selected for,
and it makes the filtering effect (the whole reason the two cohorts exist)
invisible, because the filtered-out episodes are the only ones that appear once.

Captured at `tests/fixtures/upstream/prime-rl-v0.9.1.dev28-legacy/`: three
episodes under `all/`, the one that survived filtering under `effective/`.

## The current shape: one stream, plus annotations

From v0.9.1.dev29 (`e3fdaded`) onward, `log_episodes` begins:

```python
if subset == "effective":
    return
```

The effective cohort writes no second copy. Every episode is appended exactly
once, in arrival order, to a chunked stream:

```
<output>/traces/stream/00000.jsonl        live chunk, plain text, tailable
<output>/traces/stream/00000.jsonl.zst    sealed chunk, seekable zstd frames
<output>/traces/stream.index.jsonl        one summary row per episode
<output>/traces/annotations/<producer>/   one stream per producer of updates
```

What the effective cohort learns arrives instead as **annotations**: append-only
records naming a trace, which a reader folds back onto the arrival record. From
`orchestrator/annotations.py`, the batch stamp emits, per trace:

```python
info = {"effective": True, "ship": {"step": step, "time": now}}
if (advantage := trace.info.get("advantage")) is not None:
    info["advantage"] = advantage
```

plus per-branch `advantages` streams for trainable branches.
`fold_trace_updates` merges each `info` into the trace's own (newest wins) and
projects the per-token streams onto the branch's nodes.

**So cohort membership is now a field, not a directory.** After folding, a trace
is in the effective cohort when `trace.info["effective"]` is true. A reader that
ignores the annotation streams sees every episode and cannot tell which ones
trained. A reader that reads only the stream sees no advantages at all.

Captured at `tests/fixtures/upstream/prime-rl-v0.9.1.dev46-stream/`.

## The training signal is now on disk

Worth calling out separately, because the repository's own guidance says
otherwise and that guidance is now half wrong.

CLAUDE.md golden rule 9 states that `advantages` and `is_trainable` are
in-memory training signals and are not in on-disk jsonl. That remains true of the
legacy verifiers `RolloutOutput` format, where those fields were marked
`exclude=True` on prime-rl's `Rollout` and never reached the dump.

It is not true of the current prime-rl format. The annotation stream carries a
scalar `advantage` per trace and full per-token `advantages` per trainable
branch, and the episode index derives its `advantage` column by reading
`trace.info["advantage"]` (`monitors/file/traces/index.py`). The zero-advantage
collapse signal that `reward_saturation_group_collapse` currently approximates
through reward variance is, in this format, directly readable.

Two caveats before anyone builds on that. The scalar is only present for traces
the batch stamped, so its absence means "not shipped", not "zero". And the code
path that first writes `trace.info["advantage"]` was not located during this
research; `stamp_batch` reads it rather than computing it. Confirm the producer
before relying on the field.

## What an adapter must do

The delivery plan requires an explicit cohort policy that never silently merges
the two. Concretely:

1. **Never default.** Reading a legacy run requires choosing `all`, `effective`,
   or both-kept-separate. Refuse to proceed on a legacy directory tree without
   the choice being made, rather than picking one quietly.
2. **Deduplicate by identity, not by path.** In the legacy layout the same
   episode appears under two paths with one `episode.id`. Dropping duplicates on
   identity is correct; concatenating paths is not.
3. **Keep the filtered-out population.** The episodes present in `all` and absent
   from `effective` are the filter's output. They are the most informative rows
   in the run and must survive into the normalized data, marked, not dropped.
4. **Fold annotations before scoring, or declare that you did not.** In the
   current layout an unfolded read is a legitimate mode (it is faster, and the
   stream is self-contained), but a report produced from one must say so, because
   cohort membership and advantages are simply absent from it.
5. **Record the policy in the manifest.** Which cohort was analyzed, whether
   annotations were folded, and how many duplicates were dropped, alongside the
   input counts.
6. **Preserve `kind`.** `train` and `eval` episodes share the stream in the
   current layout and are separated only by `trace.info["kind"]` (stamped at
   arrival) or by `run.work.type`. Mixing them silently pools an eval population
   into training aggregates.

## Pins

| fact | source | pin |
|---|---|---|
| legacy cohort paths and double write | `src/prime_rl/monitors/file.py`, `orchestrator/orchestrator.py` | prime-rl `84e7312f` (v0.9.1.dev28) |
| effective cohort writes no copy | `src/prime_rl/monitors/file/monitor.py` | prime-rl `dad79d1c` (v0.9.1.dev46) |
| annotation record shape and folding | `src/prime_rl/monitors/file/traces/update.py` | prime-rl `dad79d1c` |
| batch stamp contents | `src/prime_rl/orchestrator/annotations.py` | prime-rl `dad79d1c` |
| index reads `trace.info["advantage"]` | `src/prime_rl/monitors/file/traces/index.py` | prime-rl `dad79d1c` |

Verified 2026-09-08.
