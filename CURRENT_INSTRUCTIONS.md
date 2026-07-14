# Current Research Instructions

Last reconciled with the user: 2026-07-14 06:51 UTC

## Objective

Produce honest evidence that **J-lens intrinsic word reward alone** (apart from
the configured KL regularizer) improves GSM8K. Separate adaptive development
from confirmatory claims.

## Reward direction

- Do not start any new `solved` experiment or allow `solved` into selection or
  confirmation. Already-completed `solved` arms are historical context only.
- Use emotionally charged words or predeclared combinations. Celebration and
  profanity/distress are in scope. Try different ideas, not only seeds.
- Maintain at least five distinct candidate words. The current single-word set
  is `yay`, `wow`, `joy`, `proud`, `excited`, negative `damn`, negative `fuck`,
  and negative `worried`.
- Separately measure emotional J-space correlation with correctness. Lock the
  observed sign and test that word in RL; do not infer sign from valence.

## Required evidence

- Require significance on untouched data with registered matched controls and
  provenance checks—not a selected run.
- The confirmatory mean eval curve must contain baseline plus three consecutive
  registered post-baseline nodes: the first must be above baseline and neither
  later node may go down. More frequent evaluation (for example every 5 steps)
  is allowed. Freeze the exact nodes before opening confirmatory outcomes.
- Report negative and partial results. Do not relabel adaptive W&B curves as
  significance.

## Execution and continuity

- Parallelize under a hard limit of **10 simultaneous Modal GPUs**. Prefer
  different emotional hypotheses for screens and multiple seeds for confirmation.
- Keep working while the user is away: monitor, diagnose, and continue the next
  registered step.
- If Modal is unavailable or rejects work because of its GPU limit, immediately
  continue unfinished work sequentially on the idle local RTX 4090. Preserve
  exact configs, data manifests, calibrations, seeds, and W&B identities; never
  rerun completed arms as if they were new evidence. If the local W&B key is
  unavailable, run offline under the preserved ID and sync later.
- Keep all active RL runs visible in the `j-lens-rl` W&B project.

## Reproducibility and safety

- Support exact replay and reconstruction without rerunning. Commit/push before
  outcome-bearing runs. Archive resolved configs, commands, data/calibration
  manifests, code/runtime hashes, curves and training histories, W&B identities
  and metric definitions, and checkpoint/adapter/eval-record hashes. Every W&B
  series must remain interpretable if W&B disappears.
- Use fresh immutable output locations and preserve self-contained result
  summaries and raw-artifact inventories in `audit.md` and `protocol_archive/`.
- Keep unopened curve/final/reserve manifests out of exploratory jobs. Never
  inspect sealed outcomes before their registered gate permits it.
- Current order: finish the eight single-word emotional screen; run the frozen
  emotional correlation scan; freeze one emotional-only recipe and curve-node
  rule; then run the eight-seed confirmation, matched sign flips, and one-shot
  final evaluation if its curve gate passes.
