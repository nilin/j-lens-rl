# Current Research Instructions

Last reconciled with the user: 2026-07-14 07:50 UTC

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

- Use a hard global limit of **1 simultaneous Modal GPU**; serialize all GPU
  workers and never overlap Modal GPU apps.
- Treat Codex use as a rate limit, never a cumulative stop: prior usage alone
  cannot pause work. If current/recent use exceeds twice the plan's weekly
  average rate (`weekly allowance / 84` per hour), pause 20 minutes, resume,
  and reassess the current rate.
- Keep working while the user is away: monitor, diagnose, and continue the next
  registered step.
- Never mix L40S and RTX 4090 runs within an inferential attempt. If Modal is
  unavailable, close any partial attempt immutably and register a whole fresh
  local attempt with exact configs/manifests/calibrations/seeds/W&B IDs. Use
  offline W&B and sync later if the local key is unavailable.
- Keep all active RL runs visible in the `j-lens-rl` W&B project.

## Reproducibility and safety

- Support exact replay and reconstruction without rerunning. Commit/push before
  outcome-bearing runs; archive configs, commands, manifests, code/runtime and
  artifact hashes, full histories, W&B identities, and every metric definition.
- Use fresh immutable output locations and preserve self-contained result
  summaries and raw-artifact inventories in `audit.md` and `protocol_archive/`.
- Keep unopened curve/final/reserve manifests out of exploratory jobs. Never
  inspect sealed outcomes before their registered gate permits it.
- Current state: the eight-word screen is complete and mechanically selected
  positive `joy` as its sole early-curve passer; this remains adaptive evidence.
  Its offline closeout is sealed. Correlation attempts 1--3 failed before any
  selection; run only amendment 4 on fresh Volume D, with the one-GPU exclusive
  preflight. Then freeze positive `joy`, six updates, and `0/2/4/6` before
  eight-seed confirmation, matched sign flips, and the gated one-shot final
  evaluation.
