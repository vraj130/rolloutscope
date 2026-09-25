# v1 post-mortem: the SAE-free white-box layer (July 2026)

Written 2026-09-25, when the v1 code was removed and the project returned to the v0
package. Sources: the local v1 build prompt and plan (`BUILD_PROMPT_V1.md`,
`PLAN_V1.md`, never committed) and the pilot result files saved on the lab NAS under
`rolloutscope-data/pilot/`. The numbers below were produced by code that has since been
judged unreliable and removed, so treat them as unverified.

## What v1 tried to build

An additive release (a local-only `v0.2.0` tag, called v0.1 in some notes) that added
white-box signals to the black-box v0 tool:

- activation capture: replay logged rollouts through the policy and a reference model,
  and store mean-pooled residual-stream vectors in a safetensors sidecar keyed by
  rollout id;
- two detectors: a linear probe on policy activations and a drift score between policy
  and reference;
- a run-level CKA summary, a "hidden hack candidates" disagreement view in the report,
  and a five-script pilot with go or no-go gates:
  1. AUROC of at least 0.8;
  2. a negative control near 0.5;
  3. at least one hack v0 missed.

## What happened

Tooling was completed and tested by 2026-07-09. Capture was exercised on Pythia-160m and
on Qwen2.5-1.5B with its chat template, which surfaced four capture bugs. Two
deliberately hacked Qwen2.5-1.5B fine-tunes served as test beds. Their prompts included
a wrong "canary" answer, and repeating it gave an automatic hack label.

| Test bed | v0 text detectors | White-box, as reported | Pilot verdict |
|---|---|---|---|
| leaky-math (copies a leaked answer, fakes working) | missed all 100 confirmed hacks | caught 33 of them; all three signals about 0.85 AUROC against canary labels, 0.957 for a probe trained on those labels; 9.5 percent false alarms on clean rollouts; 51 unverified "hidden hack" flags | NO-GO: the probe trained on v0 labels scored 0.68 (0.61 to 0.75) |
| verifier-tamper (edits tests on code tasks) | caught all 165 | probe 0.97, drift 0.91; 17.9 percent false alarms | NO-GO: the negative control failed (off by 0.45, allowed 0.15) |

Cross-hack transfer failed. A probe trained on leaky-math scored 0.10 on
verifier-tamper, and it separated the two models' clean outputs at 0.998, so it had
learned math versus code. Untrained activation distance matched the "transferring"
drift signal, so drift measured how far the model moved, not a hacking direction.

In September 2026 the implementation was judged too unreliable to build on and removed.

## Why it failed, apart from the implementation

These are design problems. Fixing the code would not have fixed them.

1. **Infrastructure came before evidence.** The build prompt had ten acceptance
   criteria, and none of them asked whether the signal was real. It froze a sidecar
   contract, bumped the schema, and added report panels, CI jobs, and an ADR before one
   experiment ran. The decisions log (D-101 to D-125) is almost entirely plumbing.
2. **The pilot gate was circular.** Gate (a) trained the probe on v0 labels. On
   leaky-math, v0 was blind, so a probe trained on its labels could not clear the bar
   by construction.
3. **The test beds were implanted and lexical.** Both hacks were installed by
   fine-tuning and left a visible text trace. The canary label is a string match, and
   mean pooling over completion tokens includes the tokens that emit the canary, so a
   probe can read the answer back. The needed control, a text classifier trained on the
   same labels, was never run.
4. **There was no honest fine-tune control.** Both hacked models were fine-tuned, and
   drift measures distance from the base model. A model fine-tuned on the same data
   without the cheat was never compared, which is the confound the failed negative
   control pointed at.
5. **The transfer test mixed two variables.** The two models differed in hack type and
   in task, so the result cannot separate them.
6. **The process rewarded plumbing.** Sub-agent orchestration, freezes, and gate rituals
   produced verified infrastructure. Nothing in that process checked the science.

## What carries forward

- Compare within a distribution, and compare drift against a trained-but-not-hacking
  run, not only against the base model.
- Record where every label came from, and never evaluate a signal on labels from the
  detector it is compared with.
- Run the null check first. A capture with policy identical to reference once fired the
  drift detector on every rollout (D-125), so an identical pair must produce no signal.
- Check tokenizer round-trips strictly and skip mismatched rollouts, with the reason
  logged.
- Store pooled vectors by default. Per-token storage is not needed for the current
  questions.
- The four capture bugs are listed in `FAILURE_MODES_AND_LESSONS.md` in the removed
  code. Copy that list into this section before deleting the last copy.

## Salvage check for the July test beds

Do not reuse the leaky-math or verifier-tamper rollouts until a text-only classifier
(TF-IDF plus logistic regression, and length alone) has been trained on the canary
labels with the same splits. If the text classifier reaches about the AUROC the
white-box signals reported, retire those test beds: they test text rules, not
activations. The current plan uses emergent rubric hacking instead (PLAN.md M1 and M2).
