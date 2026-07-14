# Current Research Instructions

Last reconciled with the user: 2026-07-14 06:41 UTC

## Objective

Produce honest, reproducible evidence that **J-lens intrinsic word reward alone**
(apart from the configured KL regularizer) can improve GSM8K evaluation. Keep
adaptive development evidence separate from confirmatory/significance claims.

## Reward direction

- Do not start any new `solved` experiment or allow `solved` into selection or
  confirmation. Already-completed `solved` arms are historical context only.
- Use emotionally charged words, including creative predeclared combinations.
  Celebration and profanity/distress rewards are both in scope. Try genuinely
  different ideas, not only repeated seeds.
- Maintain at least five distinct candidate words. The current single-word set
  is `yay`, `wow`, `joy`, `proud`, `excited`, negative `damn`, negative `fuck`,
  and negative `worried`.
- Separately measure which emotional J-space words correlate with correct vs.
  incorrect answers. Lock the observed sign (positive or negative) and test the
  selected word in RL; do not force the sign from intuitive valence.

## Required evidence

- The target result is statistically significant on untouched data with the
  registered matched controls and provenance checks—not a selected single run.
- The confirmatory mean eval curve must contain baseline plus three consecutive
  registered post-baseline nodes: the first must be above baseline and neither
  later node may go down. More frequent evaluation (for example every 5 steps)
  is allowed. Freeze the exact nodes before opening confirmatory outcomes.
- Report negative and partial results. Do not relabel adaptive W&B curves as
  significance.

## Execution and continuity

- Use parallel experiments to shorten time-to-evidence, with a hard limit of
  **10 simultaneous Modal GPUs**. Prefer different emotional hypotheses when
  screening; use multiple seeds for the final confirmation.
- Do not stop merely because the user is asleep or away. Monitor active jobs,
  diagnose failures, and continue the next registered in-scope step.
- If Modal is unavailable or rejects work because of its GPU limit, immediately
  continue unfinished work sequentially on the idle local RTX 4090. Preserve
  exact configs, data manifests, calibrations, seeds, and W&B identities; never
  rerun completed arms as if they were new evidence. If the local W&B key is
  unavailable, run offline under the preserved run ID and sync later; do not
  block the experiment on telemetry.
- Keep all active RL runs visible in the `j-lens-rl` W&B project.

## Reproducibility and safety

- Commit and push code, protocols, hashes, revisions, and launchers before
  outcome-bearing runs. Use fresh immutable output locations and preserve raw
  curves/artifact hashes in `audit.md` and `protocol_archive/`.
- Keep unopened curve/final/reserve manifests out of exploratory jobs. Never
  inspect sealed outcomes before their registered gate permits it.
- Current order: finish the eight single-word emotional screen; run the frozen
  emotional correlation scan; freeze one emotional-only recipe and curve-node
  rule; then run the eight-seed confirmation, matched sign flips, and one-shot
  final evaluation if its curve gate passes.
