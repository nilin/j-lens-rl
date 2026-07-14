# Current Research Instructions

Last reconciled with the user: 2026-07-14 16:22 UTC

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
- Spend at least half of experiment/GPU time on RL runs. Word-correlation
  work may run between RL attempts, but it must never hold an available RL run
  for more than one hour; after an hour, launch the ready registered RL work
  and return to word search later.
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
- Current state: V5 positive-`joy` failed its registered curve gate; V6
  `yay/great/success/nice` ended infrastructure-failed after six valid seeds;
  V7 negative-`damn/fuck` ended infrastructure-failed after two valid terminal
  seeds and partial seed 186. Preserve all as honest negative/partial evidence.
  V7's registered eight-seed gate was never evaluated, controls never ran, and
  the sealed final stayed unopened. Never resume or pool V7. V8 then failed
  closed after terminal seed 200 because its wrapper expected the wrong result
  schema; preserve that complete seed as exposed development evidence, sync its
  immutable offline W&B run, but never resume, adopt, pool, or continue V8. The
  separately registered whole V9 attempt has now launched with a corrected
  verifier, fresh seeds 208--215, and the same negative-`damn/fuck` emotional
  treatment on the one local RTX 4090. Keep it running and monitored. Run
  matched controls only if its eight-treatment curve gate passes. Do not open
  the final until its evaluation/analysis path is separately implemented and
  audited. Correlation attempt 4 remains closed with outcomes uninspected; word
  search must not delay ready RL by an hour.
