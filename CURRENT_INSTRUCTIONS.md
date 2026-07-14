# Current Research Instructions

Last reconciled with the user: 2026-07-14 20:00 UTC

## Objective and evidence

Produce honest evidence that **J-lens intrinsic word reward alone** (apart from
the configured KL regularizer) improves GSM8K. Keep adaptive development
separate from confirmatory claims.

- Start no new `solved` experiment and exclude it from future selection. Use
  emotionally charged words or predeclared combinations; try different ideas,
  not only seeds. Maintain at least five candidates. Current examples are
  `yay`, `wow`, `joy`, `proud`, `excited`, and negative `damn`, `fuck`, and
  `worried`.
- Separately measure emotional J-space association with correctness. Freeze the
  observed direction before testing that word in RL; never infer reward sign
  from semantic valence.
- Significance requires untouched data, prospectively registered matched
  controls, and provenance checks—not a selected development run.
- A qualifying confirmatory mean curve has baseline plus three registered
  post-baseline nodes: the first is above baseline and neither later node goes
  down. More frequent evaluation (for example every 5 steps) is allowed, but
  freeze the nodes before opening outcomes.
- Report negative and partial results; never relabel adaptive W&B curves as
  significance.

## Execution and continuity

- Run at most **2 Modal GPUs** at once. Never mix
  L40S and RTX 4090 runs inside one inferential attempt.
- Spend at least half of experiment/GPU time running RL. Word search must never
  hold ready registered RL for over one hour; launch RL and resume search later.
- Keep working while the user is away. If Modal is unavailable, immutably close
  any partial attempt and register an entirely fresh local attempt with new
  state, seeds, manifests, configs, W&B IDs, and one hardware type.
- Keep active/completed RL visible in the `j-lens-rl` W&B project. Offline local
  runs must be synced without rerunning training.
- Treat Codex usage as a **rate**, never a cumulative stop. Prior use cannot
  pause work. Only if known current/recent use exceeds twice the plan's weekly
  average rate (`weekly allowance / 84` per hour), pause 20 minutes, resume, and
  reassess that current rate.

## Reproducibility and safety

- Support exact replay and reconstruction without rerunning: commit/push before
  outcome-bearing runs and archive code/runtime, commands, configs, manifests,
  histories, metric definitions, artifact hashes, and W&B identities in
  `audit.md` and `protocol_archive/`.
- Never inspect sealed final/reserve/correlation outcomes before a registered
  gate permits it. Do not open any sealed final until its evaluation and
  analysis implementation is separately audited.
- Preserve V5–V8 as closed negative/partial or infrastructure-failed evidence;
  never resume or pool them. V9 completed eight negative-`damn/fuck` treatments
  on one RTX 4090; its mean `.3975/.3971875/.3971875/.3925` failed the curve
  gate. Never run V9 controls or open its final.
- The fixed Modal tournament under claim
  `1d6ea36d356c420f92e125c35a1a6aeb` completed `-fuck`, `+yay`, `-worried`;
  no arm passed the full shape and the registered development ranking selected
  `-fuck`. Preserve and close out its evidence; never resume it or reuse retired
  Volume A. A separate development-only celebration-family tail-taper probe
  (`yay/great/success/nice`, seed 193, nodes `0/2/4/6`) is now running on the
  local RTX 4090 from pushed commit `bf85a74`; it completed
  `.3975/.3975/.3950/.4075`, a terminal improvement that failed the strict
  shape. Preserve/sync it but never pool it with L40S results. The next RL lane
  should use up to two parallel Modal machines on different registered emotional
  ideas, not merely duplicate seeds. Correlation attempt 4 remains closed with
  outcomes uninspected.
