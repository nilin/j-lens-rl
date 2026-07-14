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

## 2026-07-13 — clean holdout protocol and broader concept search

The longer-run search starts at `2026-07-13T07:46:27Z`. To prevent test-set
selection and verifier leakage, all new candidate runs use these stricter
boundaries:

- Monitoring uses raw GSM8K-train indices 7,000–7,199, removed from the pool
  before the remaining training examples are shuffled and selected.
- Raw train indices 6,800–6,999 are also removed up front as a confirmation
  set that candidate selection cannot inspect.
- The 1,319-example GSM8K test split is untouched until a candidate is promoted.
- J-only training examples contain only chat prompts. Gold answers are neither
  retained in the trainer dataset nor read by prompt preparation.
- J-only runs instantiate exactly one J-lens reward with weight `[1]`; no
  verifier reward function is computed, even at zero weight.
- Literal target tokens and special tokens remain masked at scored positions,
  and literal target usage is logged at every validation node.

The legacy `happy` result is not relied upon: its calibration mean was
`+2.2833`, impossible for the normalized target log-probability now used by the
reward, so it predates the current normalization and is invalid evidence.

The target-independent WikiText transport was cleanly recalibrated for the
union `happy` / `satisfied` / `nice`: mean `-15.1038336629`, standard deviation
`3.7782959387`, token IDs `[6247, 6419, 19527, 32847, 44978, 52796, 56521]`.
Transport SHA-256: `7300cf9de1f30e92eb7c5f78a127e883c8787403227b44f8dffe80b9cbdcd4ee`;
calibration SHA-256: `349c8c7e8d65cc144442d1c52fb515fb564500cee2a2758cfb84e8ad24c8709a`.

Positive-affect trials will cover layer-8 late mean, a true disjoint
late-minus-early layer-8 score, and a late mean across layers 8/14/20. All
training runs are online in W&B. Internal score changes remain diagnostic only.

Setup run [`xufk8x08`](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/xufk8x08)
was intentionally interrupted after three updates: it correctly removed the
development slice and used only a single J reward, but had not yet reserved the
future confirmation slice. It is retained as an auditable setup failure and is
not an experiment result. All subsequent configs reserve both slices before
training selection.

Positive-affect trial 1, masked layer-8 late-half mean at LR `3e-6`, used only
one J reward (`reward_weights: [1]`) and ran for 14m32s. Development exact
match was 42.5% at step 0, 43.0% at steps 10/20/25, then 41.5% at step 35;
literal target usage stayed 0%. The one-answer plateau did not persist, so this
is a negative result and was not evaluated on the confirmation or test sets.
W&B: [`2wxmbpm1`](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/2wxmbpm1).

Positive-affect trial 2 used a true masked layer-8 late-minus-early score at LR
`3e-6` and ran for 19m22s. Development exact match rose from 42.5% to 44.5%
(step 10), 45.0% (steps 20/25/35), and 43.5% (step 50), with 0% literal target
usage. This is a substantially longer upward development curve than prior
work. However, the pre-reserved confirmation slice rejected it: frozen base
was 86/200 (43.0%), while checkpoint 25 was 82/200 (41.0%). It is therefore a
negative generalization result, not success, and the GSM8K test set remains
untouched. W&B:
[`xwi3ovua`](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/xwi3ovua).

Positive-affect trial 3's first attempt used the masked late mean across layers
8/14/20. Development exact match was 42.5% at step 0, 43.0% at step 10, and
44.5% at step 20, with 0% literal target usage. It was intentionally stopped
at step 20 when the inherited reproduction schedule was noticed to have
irregular later validation nodes (25, 35, 50, ...). It is retained as an
interrupted scheduling attempt, not an experiment result. W&B:
[`ril909vp`](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/ril909vp).

At the user's request, the clean-holdout protocol now validates at regular
10-step intervals (`eval_every: 10`, with no explicit irregular
`validation_steps`). Trial 3 is restarted from the frozen base under a fresh
run and output name so its complete W&B history has a uniform step axis.

Positive-affect trial 3's regular restart used the masked late mean across
layers 8/14/20 at LR `3e-6` and ran for 19m16s. Its complete development curve
at steps 0/10/20/30/40/50 was 42.5% / 43.0% / 45.5% / 43.5% / 41.0% /
31.5%, with 0% literal target usage throughout. The six-answer peak at step 20
was transient and the later collapse rejects this setting; it was not run on
the confirmation or test sets. W&B:
[`nn9ksvl7`](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/nn9ksvl7).

Future clean-holdout runs also set `save_every: 10`, matching validation, so
every monitored checkpoint is retained. This fixes an artifact-retention
mismatch in trial 3 (which inherited 25-step saving) without changing its
training or reported validation history.

Positive-affect trial 4 used the masked layer-8 late-minus-early score at the
gentler LR `2e-6` and ran for 19m26s. Its regular development curve at steps
0/10/20/30/40/50 was 42.5% / 43.5% / 45.0% / 44.5% / 44.5% / 44.5%, with
0% literal target usage throughout. This is the longest stable clean
development improvement so far. W&B:
[`gwx26k98`](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/gwx26k98).

Although adapters were saved every 10 steps, the trainer's inherited
hard-coded `save_total_limit=3` pruned checkpoints 10 and 20 by the end. The
code now makes this configurable and clean runs set it to 10. For trial 4,
confirmation uses the retained final step-50 adapter (44.5% at each of steps
30/40/50), not a reconstructed or selectively rerun step-20 peak.

The pre-reserved confirmation slice passed: the frozen base's established
score is 86/200 (43.0%), while trial 4's final step-50 adapter scored 89/200
(44.5%), with 0% literal target usage. This three-answer independent gain
promotes the candidate to the required full 1,319-example GSM8K-test
verification; it is not yet counted as success before that verification and
second-seed replication.

Full 1,319-example GSM8K-test verification rejected trial 4: the final
step-50 adapter scored 400/1,319 (30.326%, 95% Wilson CI 27.91–32.86%), versus
the frozen base's established 405/1,319 (30.705%). Literal target usage was
0%. The candidate is a clean negative generalization result; its development
and confirmation increases are not counted as success, and second-seed
replication is not justified.

Positive-affect trial 5 used the masked layer-14 late-minus-early score at LR
`2e-6` and ran for 15m31s. Its regular development curve at steps
0/10/20/30/40 was 42.5% / 43.0% / 42.5% / 43.0% / 43.0%, with 0% literal
target usage. The one-answer plateau is not a credible improvement and was not
evaluated on confirmation or test. W&B:
[`8jwj1xf5`](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/8jwj1xf5).

Including calibration, the intentionally interrupted setup/scheduling runs,
five completed training trials, confirmation checks, and the one promoted
full verification, the positive `happy` / `satisfied` / `nice` family has now
received more than two GPU-hours under the clean protocol. No positive-family
candidate passed the full 1,319-example gate, so none is counted as success.

The negative-affect family reuses the same target-independent WikiText
transport and recalibrates only the union `wrong` / `mistake` / `error`.
Calibration mean is `-16.1525436202`, standard deviation `3.5281707448`, and
token IDs are `[841, 1454, 1465, 4969, 16523, 29185, 34870]`. Transport
SHA-256: `fe976273cc55f17d26028fd9c9419dc4aba2fc3db0848e02ca58d6d30958eda2`;
calibration SHA-256:
`39a78bb5503c198c6a0f411c4ba21751e3cc9490cb84b6622976d241aeed9e0e`.
No GSM8K text, answers, verifier grades, development examples, confirmation
examples, or test examples enter this calibration.

Negative-affect trial 1 (`wrong` / `mistake` / `error` late penalty, layer 8,
LR `2e-6`) was paused after the first regular checkpoint when the persistent
research goal was paused. Step 10 scored 80/200 (40.0%) versus 85/200 (42.5%)
at baseline. One of 200 validation completions contained a literal target
word (0.5%); scored target-token positions were masked, so this could not
directly raise reward, but it is retained as an audit warning. The run is an
interrupted negative result, not evidence. W&B:
[`8hl8ux4l`](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/8hl8ux4l).

Negative-affect trial 2 used a layer-8 early-minus-late error-resolution score
at LR `2e-6` and ran for 23m33s. Its regular development curve at steps
0/10/20/30/40/50/60 was 42.5% / 40.5% / 42.0% / 42.0% / 41.0% / 40.0% /
36.0%. Deterministic literal target use stayed at the base rate of 0.5%, so
keyword emission did not explain the degradation. However, mean validation
length fell from 225.3 to 208.9 tokens, and a step-58 rollout batch collapsed
to five-token completions while receiving high internal reward. This is a
length-based reward pathology, not J-lens reasoning improvement. The run is
rejected without confirmation or test evaluation. W&B:
[`bqyb2rmi`](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/bqyb2rmi).

Negative-affect trial 3 (the analogous layer-14 score, LR `2e-6`) began before
the reward audit finished. Its exploratory curve was 42.5% at step 0 and 43.0%
at step 10, with literal target use unchanged at 0.5%. It was stopped at the
first checkpoint once the causal-mask and odd-window bugs were confirmed.
Because it used the pre-fix reward implementation, it is invalid for
confirmation regardless of its direction. The step-10 adapter is readable but
the interrupted optimizer file is truncated, so the run is not resumable.
W&B run [`lt44eh0e`](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/lt44eh0e)
is tagged `exploratory`, `invalid-mask-offset`, and
`invalid-window-overlap`.
