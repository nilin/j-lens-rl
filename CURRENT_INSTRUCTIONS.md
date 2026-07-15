# Current Research Instructions

Last reconciled: 2026-07-15 00:52 UTC. This is the current operating brief;
older chat instructions and historical closeouts do not override it.

## Objective and evidence standard

Produce honest, reconstructable evidence that intrinsic J-lens reward for
emotionally charged words, apart from the fixed KL regularizer, improves GSM8K.
Do not launch new `solved` experiments. Spend at least half of experiment/GPU
time on RL, and never block ready RL on word search for more than one hour.

The current experiment is V14, a development-only, adaptively selected dense
replication of the most defensible V11/seed-195 celebration recipe:
`yay/great/success/nice`, calibration SHA `93d05caf...8ee6`, layer 8, stride
10, weights `+1` on response fraction `.50-.75` and `+.25` on `.75-1.0`.
Training is six fixed updates, LR `3e-6`, DAPO, LoRA r8/alpha16, eight
generations, J reward plus KL beta `.02`, and no correctness reward or
accuracy-based stopping. Four treatment seeds 236--239 and four exact matched
sign-flip controls are fixed. Eval is greedy on the exposed 400-row curve at
every step `0..6`; the inherited display gate uses `0/4/5/6` and requires
`M4>M0`, `M5>=M4`, `M6>=M5`. Four strictly positive matched terminal effects
give exact two-sided sign-test `p=.125`, accepted as nominal evidence at
`alpha=.15`. Always disclose the adaptive-program/multiplicity caveat.

V14 app `ap-ez4IZH2rdlBRnw4cdHefqf`, claim
`e0657eca40da49b78830f5e7a1e47a14`, and volume
`j-lens-rl-development-v14-v11style-celebration-20260715b` are active from
pushed commit `5ee921f`. The prior two V14 infrastructure attempts are closed,
outcome-free, and must not be resumed.

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

Seed195 produced `.3825/.3925/.4000/.4125` at `0/4/10/20`, but it is one
development seed. V11 rose through step5 but was infrastructure-interrupted;
V12 dipped at step5; V13 positive treatment dipped at step10, while its
sign-flip mean was monotone but beat treatment terminally in only one of four
matched seeds. These visible results motivated V14 and prevent describing V14
as an untouched independent confirmation or familywise-corrected evidence.
