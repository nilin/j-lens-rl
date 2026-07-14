# Current Research Instructions

Last reconciled: 2026-07-14 23:20 UTC. This file is the current operating
brief; older instructions in chat or historical closeouts do not override it.

## Objective

Produce honest evidence that intrinsic J-lens reward for emotionally charged
words, apart from the fixed KL regularizer, improves GSM8K. Do not launch new
`solved` work. Adaptive screens may suggest candidates, but only a freshly
registered attempt on untouched data may support significance.

The current candidate is the prospectively frozen V13 exact-long-horizon
celebration-tail follow-up:
`yay/great/success/nice`, calibration SHA `93d05caf...8ee6`, layer 8, stride
10, weights `+1` on response fraction `.50-.75` and `+.25` on `.75-1.0`.
It exactly follows the useful seed-195 development lineage using fresh L40S
seeds 228--231. Training is 20 fixed updates, LR `3e-6`, DAPO, LoRA r8/alpha
16, eight generations, target masking, J-lens reward plus KL beta `.02`, and
no correctness reward or accuracy-based stopping.

Evaluate treatment curves at fixed nodes `0/4/10/20`. The curve gate passes
only if the four-seed mean satisfies `M4 > M0`, `M10 >= M4`, and
`M20 >= M10`. Four treatments and four exact matched sign-flip controls may
train concurrently because all are prospectively fixed; only treatment curves
enter the gate. On gate failure stop without final access. On pass, and only
after all eight terminal runs verify, unlock one immutable nine-label
collection on the still-unopened 900-row final.

The deadline is 2026-07-15 02:00 UTC (7:00 PM Pacific). Nominal `alpha=.15` is
accepted: each registered four-seed sign test needs four strictly positive
effects, no ties, positive mean, giving exact two-sided `p=.125`. Report 95%
crossed intervals descriptively. State clearly that this is nominal evidence
inside a broader adaptive program, not familywise-error-corrected evidence.

## Execution

- Run at most ten Modal GPUs after 23:00 UTC. The registered V13 launcher uses
  eight joint treatment/control GPUs and at most one serial final GPU.
- Spend at least half of experiment/GPU time on RL. Never hold ready RL behind
  word search for over an hour. Do not reopen candidate selection after V13.
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
its registered gate. Never resume, pool, or silently repair closed V5--V11
attempts. V10c and V11 are terminal as recorded. V12 is also terminal: all four
treatments completed, but its mean `.382500/.393125/.386875/.388125` at
`0/4/5/6` declined at step 5, so it launched no controls and never opened the
final. Its seed-227 one-float32-ULP verifier finding is an infrastructure lesson,
not a retry basis. V13 excludes all prior rows/checkpoints from inference and
uses fresh execution identities. The complete archived seed-195 development
curve `.3825/.3925/.4000/.4125` motivates V13 but is not itself significant.
Describe any V13 `p=.125` only as nominal adaptive-program evidence, never as
an independent untouched replication or a familywise-corrected result.
