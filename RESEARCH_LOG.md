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

1. Screen late-half and final-quarter mean/max `solved` readouts at each layer;
   a transient peak may be more informative than a sequence average.
2. Train only a readout whose cross-validated grouped ranking materially beats
   the current 60--62% range.
3. Run a matched correctness-reward control at the selected learning rate.
