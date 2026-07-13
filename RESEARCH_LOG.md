# J-lens RL research log

Primary success metric: held-out GSM8K numeric exact match. J-lens scores are
training signals and diagnostics, not success criteria.

## 2026-07-13 — setup and implementation audit

- Pinned base model: `Qwen/Qwen2.5-0.5B-Instruct`.
- Target word: `solved` (single-token whitespace/case variants).
- Fitted Jacobian lens on WikiText at layers 8, 14, and 20.
- Base calibration: mean `-19.0801`, standard deviation `3.9104`.
- Fixed validation logging, Wilson intervals, normalized target log-probability,
  literal-token masking, nonzero KL, and post-training lens refitting support.
- Confirmed the vendored TRL delta remains limited to exposing the unwrapped
  policy and rollout token IDs to the custom reward.
- Tests: 7 passing.

Relevant commits: `dc06447`, `606a49f`, `afb8ecb`, `6543d2f`, `107124f`.

## Correctness-reward control

Configuration: all fitted J layers monitored, exact-match reward optimized,
learning rate `1e-6`, KL beta `0.02`, 8 generations.

W&B: [`v3v0h31b`](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/v3v0h31b)

| Step | Exact match (n=200) | Observation |
|---:|---:|---|
| 25 | 32.0% | Small movement at most |
| 50 | 32.0% | Flat; run stopped |

A separately loaded frozen base scored 30.5% on the same 200 examples, while
trainer-wrapped step-zero evaluations consistently scored 32.5%. Because this
wrapper/runtime discrepancy is too large to ignore, final comparisons use the
same standalone evaluator for both base and adapter on all 1,319 test examples.

## Blind `solved` reward screens

All results below use the same fixed 200-example greedy validation set. Unless
noted, literal `solved` completion rate was 0%.

| Signal | LR | Step 0 | Step 25 | Change | W&B | Decision |
|---|---:|---:|---:|---:|---|---|
| Late half, layers 8/14/20 mean | 1e-6 | 32.5% | 32.5% | 0.0 pp | [`n63gohs5`](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/n63gohs5) | Stop |
| Late half, layer 8 mean | 1e-5 | 32.5% | 29.0% | -3.5 pp | [`10ibk6v8`](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/10ibk6v8) | Harmful |
| Late half, layer 14 mean | 1e-5 | 32.5% | 30.0% | -2.5 pp | [`8n3p9946`](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/8n3p9946) | Harmful |
| Late half, layer 20 mean | 1e-5 | 32.5% | 32.5% | 0.0 pp | [`zfy82cvw`](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/zfy82cvw) | Stop |
| Final content token, layer 14 | 1e-5 | 32.5% | 25.0% | -7.5 pp | [`jboexrc3`](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/jboexrc3) | Strongly harmful |
| Final content token, layer 20 | 1e-5 | 32.5% | 30.0% | -2.5 pp | [`pf4u6rq9`](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/pf4u6rq9) | Harmful |

Conclusion: a J-score can be readily optimized while exact match falls. Layer,
temporal aggregation, and update size are therefore material; internal score
movement alone is not evidence of better reasoning.

## Pre-training reward-alignment screen

Added `analyze-jlens-alignment` in commit `e429e63`. It samples grouped base
rollouts and asks whether a higher candidate score ranks correct completions
above incorrect completions from the same prompt—the ordering GRPO actually
uses.

Replication: 100 prompts x 8 generations = 800 rollouts; 60 prompts had mixed
correct/incorrect outcomes.

| Candidate | Global correlation | Correct - incorrect | Within-prompt pair accuracy |
|---|---:|---:|---:|
| Layer 8, late-half mean | +0.083 | +0.093 SD | **60.1%** |
| Layer 14, late-half mean | +0.028 | +0.027 SD | 57.7% |
| Layer 20, final content token | **+0.202** | **+0.499 SD** | 56.6% |
| All-layer/all-position means | near zero | near zero | near chance |

Artifact: `artifacts/solved_alignment_100.json` (ignored by Git with other run
artifacts).

## Alignment-selected lower-rate runs

| Signal | LR | Step 0 | Step 25 | Step 50 | Decision |
|---|---:|---:|---:|---:|---|
| Layer 8, late-half mean | 3e-6 | 32.5% | **33.5%** | 33.0% | Best checkpoint: 25 |
| Layer 20, final content token | 3e-6 | 32.5% | 32.5% | — | Flat; stop |

Layer-8 W&B: [`xkytbmoz`](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/xkytbmoz).
Layer-20 W&B: [`r4govttw`](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/r4govttw).

The layer-8 gain is only 2 additional correct answers out of 200. Full-test
standalone evaluation on all 1,319 GSM8K examples gave:

| Model | Exact match | Correct | 95% Wilson CI |
|---|---:|---:|---:|
| Frozen base | 30.705% | 405/1,319 | 28.27–33.25% |
| Layer-8 checkpoint 25 | 30.933% | 408/1,319 | 28.50–33.48% |

The net gain is 3 answers (+0.23 percentage points). It is directionally
positive but far too small to establish a reliable effect, so a second training
seed is required.

Seed-43 replication peaked at 33.0% on the 200-example monitor at step 25,
but full-test evaluation scored 404/1,319 (30.629%), one answer below the frozen
base. The seed-42 full-test gain therefore did **not** replicate and is treated
as selection noise, not success.

## Next decision

The nine-readout ridge composite improved cross-validated within-prompt pair
accuracy from 60.1% (layer-8 late-half mean) to 62.0% on the same 100-prompt
screen. Its step-25 training result nevertheless fell from 32.5% to 32.0%
exact match (W&B
[`07ins5l5`](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/07ins5l5)).
This is a negative result: offline rank alignment did not generalize into a
validation improvement.

Next:

The expanded screen used 200 prompts x 8 generations (1,600 rollouts), with
136 mixed-outcome prompts. Layer-20 final-token was the best simple readout at
58.5% pair accuracy; layer-8 late-half mean reached 56.1%. Late/final-quarter
max readouts were near chance. The 18-way composite reached 62.9%, only 0.9
points above the prior composite that failed in RL, so it was rejected as too
weak to justify another fitted-composite run.

Artifact: `artifacts/solved_alignment_windows_200.json` (ignored).

Next: run a matched exact-match-reward control at learning rate `3e-6` to test
whether this update regime can produce a detectable validation change before
continuing the internal-reward search.

The matched control was also flat: 32.5% at step 0 and 32.5% at step 25 (W&B
[`37nto25a`](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/37nto25a)).
Thus a 25-update, `3e-6` run does not reliably move greedy exact match even
under the oracle training reward. The next variation refits the Jacobian lens
on chat-formatted GSM8K reasoning rather than generic WikiText, then screens
alignment before training.

The 100-prompt GSM8K-domain fit completed. Calibration on 50 held-out reference
reasoning transcripts: mean `-18.3153293482`, standard deviation
`4.7388755305`, target token ID `27956`. This config and its artifacts were
subsequently removed without an accepted result for the leakage-boundary reason
below.

## Rejected: reference-solution domain lens

The GSM8K-domain lens path was abandoned before any result was accepted. It
used correct GSM8K training solutions as the lens-fitting text. This was not
held-out leakage and did not update the policy, but it made the reward
indirectly solution-informed and therefore unsuitable for the intended claim.
Its screen was stopped, and its fitting option, config, and artifacts were
removed. A replacement domain variant is permitted only on the frozen base
model's own ungraded completions from GSM8K training prompts disjoint from the
RL subset. Reference answers, verifier scores, correctness labels, validation,
and test examples remain forbidden in reward construction.

## Generic WikiText `happy` reward

A separate clean concept test used a lens fitted and calibrated only on
WikiText, targeted `happy`, masked literal target-token positions, and assigned
zero reward weight to GSM8K correctness. Literal `happy` completion rate stayed
at 0% throughout. Held-out numeric exact match did not improve:

| Step | Exact match (n=200) |
|---:|---:|
| 0 | 33.5% |
| 25 | 32.5% |
| 50 | 32.5% |
| 75 | 31.5% |
| 100 | 31.5% |

W&B: [`kxor0zvs`](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/kxor0zvs).
This is a negative result. Internal happy-score movement is not counted as
success, and no full-test or second-seed follow-up is justified.

## Ungraded base-rollout lens

To domain-match without solution information, the frozen base model sampled
150 response-only completions from shuffled GSM8K training prompts 1,000–1,149,
disjoint from the 1,000 RL prompts. The fitting code reads only `question`; it
never reads reference answers, verifier scores, correctness labels, validation,
or test data. The exact sampled corpus is saved with the ignored lens artifact.

Calibration on 50 held-out ungraded rollouts: mean `-19.4511426386`, standard
deviation `6.1915322224`, target token ID `27956`.

Grouped alignment screen: 200 training prompts × 8 frozen-base generations;
145 prompts had mixed outcomes. The strongest simple readout was layer-20 final
content token: 58.00% within-prompt pair accuracy, correlation `+0.1769`, and
correct-minus-incorrect score `+0.3202` SD. All temporal means/maxima were near
or below chance. The verifier-fitted composite reached only 58.66% and is not
eligible as a reward. Artifact:
`artifacts/solved_alignment_ungraded_rollout_lens_200.json`.

Decision: screen one J-only layer-20 final-token run at LR `3e-6`, with online
W&B and the unchanged step-25 exact-match stop gate.

The run failed the gate: exact match fell from 32.5% at step 0 to 28.0% at
step 25, with 0% literal `solved` usage. It stopped immediately. W&B:
[`x0h4ul95`](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/x0h4ul95).
No full-test evaluation is justified.

Next variation: return to the generic WikiText layer-8 late-half signal—the
only clean signal with a prior positive monitor node—and reduce LR from `3e-6`
to `2e-6` to test for a slower, sustained multi-node improvement. Reward
construction and all data boundaries remain unchanged.

LR `2e-6` failed the first gate, falling from 32.5% to 31.0% at step 25; W&B
[`y2b2p5b0`](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/y2b2p5b0).
Next, reproduce the strongest LR `3e-6` setting with additional greedy monitor
nodes at steps 10 and 20 before the unchanged step-25 gate. Evaluation is
gradient-free and deterministic and does not alter rollout RNG or rewards.

## LR `3e-6` dense-evaluation reproduction

The generic WikiText `solved` lens, layer-8 late-half mean readout, and LR
`3e-6` setting were reproduced on an RTX 4090. The substantive W&B config is
identical to the earlier H100 run: seed 42, eight rollouts, 256-token maximum
completion, temperature 1, DAPO/group normalization, KL 0.02, rank-8 LoRA,
and reward weights `[0, 1]`. Only the output/run names and added deterministic
validation nodes differ.

| Step | Exact match (n=200) |
|---:|---:|
| 0 | 32.5% |
| 10 | 32.0% |
| 20 | 32.5% |
| 25 | **33.5%** |
| 35 | 32.5% |
| 50 | **33.5%** |

Literal `solved` completion rate remained 0% at every node. This passes the
requested multiple-monitor-node condition but is not yet a successful result:
the full 1,319-example verification and second-seed replication gates still
apply. W&B:
[`kwk4m0ev`](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/kwk4m0ev).

Standalone greedy evaluation of the step-50 adapter on all 1,319 held-out
examples scored 408/1,319 (30.933%, 95% Wilson CI 28.50–33.48%), versus the
frozen base's established 405/1,319 (30.705%). This is a gain of three answers
and matches the earlier seed-42 full-test result. Literal `solved` appeared in
1/1,319 completions (0.076%). The effect remains too small to accept without
the required seed-43 replication.

Seed 43 independently improved at multiple monitor nodes:

| Step | Exact match (n=200) |
|---:|---:|
| 0 | 32.5% |
| 10 | **33.5%** |
| 20 | **34.0%** |
| 25 | **33.0%** |
| 35 | **34.0%** |

Literal `solved` completion rate remained 0% throughout. Training stopped at
step 35 after two evaluations without exceeding the 34.0% best. W&B:
[`wsg6wioj`](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/wsg6wioj).
The final step-35 adapter scored 407/1,319 on the full held-out test
(30.857%, 95% Wilson CI 28.42–33.40%), two answers above the frozen base, with
0/1,319 literal `solved` completions.

The strict acceptance gate therefore passes directionally on both seeds:
408/1,319 for seed 42 and 407/1,319 for seed 43, versus 405/1,319 for the same
frozen-base evaluator. The absolute gains are only +3 and +2 answers and the
confidence intervals heavily overlap, so this is evidence of a small
replicated directional effect, not a statistically precise or large gain.
