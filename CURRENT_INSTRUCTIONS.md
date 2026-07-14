# Current Research Instructions

Last reconciled: 2026-07-14 22:25 UTC. This file is the current operating
brief; older instructions in chat or historical closeouts do not override it.

## Objective

Produce honest evidence that intrinsic J-lens reward for emotionally charged
words, apart from the fixed KL regularizer, improves GSM8K. Do not launch new
`solved` work. Adaptive screens may suggest candidates, but only a freshly
registered attempt on untouched data may support significance.

The current and final candidate switch is V11 celebration-tail:
`yay/great/success/nice`, calibration SHA `93d05caf...8ee6`, layer 8, stride
10, weights `+1` on response fraction `.50-.75` and `+.25` on `.75-1.0`.
It directly follows the useful seed-195 development lineage using fresh L40S
seeds 220--223. Training is six fixed updates, LR `3e-6`, DAPO, LoRA r8/alpha
16, eight generations, target masking, J-lens reward plus KL beta `.02`, and
no correctness reward or accuracy-based stopping.

Evaluate treatment curves at fixed nodes `0/4/5/6`. The curve gate passes only
if the four-seed mean satisfies `M4 > M0`, `M5 >= M4`, and `M6 >= M5`. On
failure, stop without controls or final access. On pass, run exact matched
sign-flip controls on the same four seeds by negating both component weights;
then, and only after all eight runs verify, unlock one immutable nine-label
collection on the still-unopened 900-row final.

The deadline is 2026-07-15 02:00 UTC (7:00 PM Pacific). Nominal `alpha=.15` is
accepted: each registered four-seed sign test needs four strictly positive
effects, no ties, positive mean, giving exact two-sided `p=.125`. Report 95%
crossed intervals descriptively. State clearly that this is nominal evidence
inside a broader adaptive program, not familywise-error-corrected evidence.

## Execution

- Run at most five Modal GPUs before 23:00 UTC, then at most ten. The registered
  V11 launcher uses four treatment GPUs and only conditionally four controls.
- Spend at least half of experiment/GPU time on RL. Never hold ready RL behind
  word search for over an hour. Do not reopen candidate selection after V11.
- Keep every active/completed run visible in W&B. If Modal becomes unavailable,
  close the attempt immutably before registering a genuinely fresh local
  attempt; never mix L40S and RTX 4090 within one inference.
- Codex usage limits apply to current rate, never accumulated prior use. Only
  pause for 20 minutes if known recent use exceeds twice the weekly-average
  allowance rate; then resume and reassess.

## Integrity and continuity

Commit and push the exact code/config/contract before outcome-bearing work.
Preserve commands, manifests, raw histories, resolved configs, artifact and
source hashes, hardware/environment, W&B identities, attempts, failures, and
decisions so results can be reconstructed without retraining and replayed when
needed. Never inspect any sealed final, reserve, or correlation payload before
its registered gate. Never resume, pool, or silently repair closed V5--V10
attempts. V10c negative-`fuck` is terminal: its mean public curve
`.3825/.381875/.39125/.38625` at `0/2/3/4` failed the requested shape, and its
verifier also failed on an overly strict float32 standard-deviation tolerance;
no controls or protected final ran. Preserve that closeout, but draw no V11
recipe choice from its outcomes.
