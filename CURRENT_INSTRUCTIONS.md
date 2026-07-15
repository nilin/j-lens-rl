# Current Research Instructions

Last reconciled: 2026-07-15 02:10 UTC. This is the current operating brief;
older chat instructions and historical closeouts do not override it.

## Objective and evidence standard

Produce honest, reconstructable evidence that intrinsic J-lens reward for
emotionally charged words, apart from the fixed KL regularizer, improves GSM8K.
Do not launch new `solved` experiments. Spend at least half of experiment/GPU
time on RL, and never block ready RL on word search for more than one hour.

The current experiment is V16, a development-only, adaptively selected
many-seed extension of V14's celebration result:
`yay/great/success/nice`, calibration SHA `93d05caf...8ee6`, layer 8, stride
10, weights `+1` on response fraction `.50-.75` and `+.25` on `.75-1.0`.
Training is ten fixed updates, LR `3e-6`, DAPO, LoRA r8/alpha16, eight
generations, J reward plus KL beta `.02`, and no correctness reward or
accuracy-based stopping. Sixteen treatments and sixteen exact seed-matched
sign-flip controls are fixed on fresh seeds 248--263. Eval is greedy on the
exposed 400-row curve at exactly global steps `0,2,4,6,8,10`. Every measured
node must remain in every run history, aggregate JSON/CSV, W&B aggregate, and
plot; never selectively omit a node. The early shape gate uses consecutive
measured nodes `0/2/4/6` and requires `M2>M0`, `M4>=M2`, `M6>=M4`; nodes 8 and
10 remain mandatory regardless. Primary evidence averages each treatment
seed's improvement from baseline over all five post-baseline nodes, then uses
an exact two-sided seed sign test at nominal `alpha=.15`. The matched-control
integrated sign test is separate and required for a causal reward-sign claim.
Always disclose the adaptive-program/multiplicity caveat.

V15 was closed outcome-incomplete after a Modal CPU-coordinator preemption;
four baselines and one partial step-1 value are disclosed but excluded. V15B
was registered as a five-step replacement but was never launched after the
user requested the longer complete curve. V16 inherits only its verified
preemption-safe orchestration. V16 launched from pushed commit
`e11f4fbe02fcd2b1cf279a5c651f5b6adf3f5b0f` as Modal app
`ap-jZvqIF8u5dMi8dteypxVBs`, claim
`906eefc5089c4e928a7e6f165ff07108`, root call
`fc-01KXHS11WNJ0VPYCXR86X8C5H9`, and fresh Volume
`j-lens-rl-development-v16-v14-celebration-n16-20260715a`. Its durable
manifest contains exactly 32 fixed worker call IDs; four pair-interleaved
workers run at a time and the remaining calls queue without outcome
conditioning.

## Execution and continuity

- At most four Modal GPUs may run concurrently for any new or active launch.
  Do not kill already-running work merely to apply a changed cap.
- Keep runs visible in W&B with fixed IDs and a meaningful canonical x-axis.
  Preserve raw histories, configs, source/artifact hashes, hardware, commands,
  receipts, failures, plots, and decisions so results can be reconstructed
  without retraining and the training can also be replayed.
- If Modal is unavailable, immutably close the attempt before registering a
  genuinely fresh RTX 4090 attempt; never mix hardware within one inference.
- Do not inspect protected-final, reserve, or correlation payloads unless an
  applicable frozen gate authorizes it. Never resume, pool, or silently repair
  a closed scientific attempt.
- Continue working after the former 2026-07-15 02:00 UTC target; it is a
  scheduling milestone, not a stop condition. If V14 is negative, archive it
  honestly and keep testing emotionally charged J-word ideas with fresh,
  explicitly registered designs.
- Codex usage throttling concerns the current usage rate, not accumulated prior
  use. Pause 20 minutes only when known recent rate exceeds twice the
  weekly-average allowance rate; then resume and reassess.

## Context that must remain explicit

V14 completed all eight runs cleanly. Treatment terminal-minus-baseline was
positive for all four seeds (mean `+.01125`, exact sign `p=.125`) and its dense
means contain the requested `0/3/4/5` segment
`.3825/.3875/.3875/.4025`. Its registered `0/4/5/6` gate nevertheless failed
on the step-6 dip, and treatment-minus-signflip was not significant
(`p=.625`). V16's seed count, ten-update horizon, two-step cadence, shape, and
integrated tests were selected after visible V11--V15 outcomes. This prevents
calling V16 untouched, independent, or familywise-corrected evidence.
