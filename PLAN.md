# J-lens reward RL: confirmatory one-page plan

## Question

Does maximizing a fixed, target-independent J-lens score for `solved` cause a
held-out increase in greedy GSM8K numeric exact match? The candidate is frozen
from prior exploratory work: Qwen2.5-0.5B-Instruct, WikiText transport and
calibration, layer-8 late-half mean, LR `3e-6`, rank-8 LoRA, and 25 updates.

## Comparisons

Run seeds 142–147 from the untouched base under two required matched
conditions: semantic J score (weight `+1`) and the exact same J score sign
flipped (weight `-1`). Prompts, source indices, order, optimizer, generation,
KL, LoRA, and horizon match within seed. Optionally run one seed-142
exact-match-reward positive control to diagnose the learning pipeline.

J-only rows contain no reference answer and register exactly one task reward.
The fixed KL regularizer remains part of GRPO. Training generations have a
64-token minimum to block the known short-output exploit. Literal target
spellings are causally masked and audited. Evaluation is unconstrained.

## Data boundaries

Reconstruct and exclude every historically used raw GSM8K-train index. Hash
order the remaining 4,063 and allocate 200 exploratory, 400 one-shot curve,
3,000 sealed-final, and 463 future-reserve examples. Exclude all 4,063 from
training. Pin raw source-index manifests, configs, artifacts, model/dataset
revisions, and the clean source commit before any v1 run.

## Curve gate

Greedy exact match is observational at steps `0,5,10,15,20,25`; it never stops
training or selects a checkpoint. The endpoint is always step 25. Across the
mean of all six semantic seeds, require:

```text
EM5 > EM0; EM10 >= EM5; EM15 >= EM10
```

No other three nodes or selected seeds may satisfy the gate. If it fails, v1 is
negative and the sealed final set remains closed.

## Final evidence

After all 12 required runs and the curve gate pass, greedily evaluate the base
once and all adapters on the same sealed 3,000 examples. Retain per-item source
index, prompt hash, completion, prediction, correctness, literal audit, and
full model/adapter/artifact/source provenance.

The primary success criterion is a positive mean semantic-minus-base paired
change whose 95% crossed seed/item bootstrap interval excludes zero, with all
six seed effects positive (two-sided exact sign-test `p=0.03125`). Report
within-seed discordant tables/McNemar diagnostics and semantic-minus-sign-flip
difference-in-differences. A higher internal score or old official-test result
is not success.

## Execution

Use `CONFIRMATORY_PROTOCOL.md` and `run_confirmatory.sh`. Preparation and every
phase fail closed on a dirty/different commit, changed artifact or manifest,
nonempty output directory, overlapping source indices, mismatched seed data,
nonfixed horizon, or incomplete curve. Record negative and interrupted runs;
never tune on the curve/final set or rerun until favorable.
