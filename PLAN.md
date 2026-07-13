# J-space reward RL on GSM8K — one-page plan

## Question and primary comparison

Can RL make `Qwen/Qwen2.5-0.5B-Instruct` represent **“good”** more strongly in its J-space, and does that improve verifiable math performance? Run two paired conditions in which the **reward function is the only experimental variable**:

1. **Verifiable baseline:** RL reward is exact equality between the extracted final number and GSM8K ground truth (`0/1`).
2. **J-good:** RL reward uses only the policy’s internal J-lens score for “good.”

Both start from the same untouched checkpoint and use the same initialization seed, examples in the same order, prompts, four sampled completions, decoding settings, LoRA targets/rank, optimizer and schedule, KL coefficient, batch/accumulation, token budget, update count, evaluation set, and stopping rules. Do not initialize J-good from the correctness-trained model. Evaluate the frozen base once at step zero as a reference, not as a third training condition.

## Minimal implementation

Use Anthropic’s Apache-2.0 [`anthropics/jacobian-lens`](https://github.com/anthropics/jacobian-lens). Fit its Jacobian transport once on ~100 generic 128-token sequences at three middle/late layers. Verify which of `" good"`, `"good"`, and `"Good"` are single Qwen tokens and use all valid IDs.

For each completion, rerun a gradient-free policy forward pass and retain response-position hidden states. J-reward is the mean over chosen layers and positions of the target tokens’ J-lens log-probability, clipped to the base model’s calibration range. Freeze the fitted Jacobians, embeddings, final norm, and LM head; train rank-8 LoRA on attention/MLP projections. Re-fit the lens after training to detect exploitation of a stale lens.

Use a vendored, commit-pinned Hugging Face TRL [`GRPOTrainer`](https://github.com/huggingface/trl) for generation, the GRPO objective, KL regularization, evaluation, checkpointing, and W&B logging. A narrow trainer patch exposes the unwrapped policy and rollout token IDs to reward callables so J-good can run an additional hidden-state forward pass. Both rewards are calculated and logged in both runs; only their weights change (`[1,0]` versus `[0,1]`).

## Experiment and plots

Use a fixed 1,000-example training subset and 200 disjoint validation examples. Run a 50-update smoke test, then ~500 updates. Every 25 updates, deterministically generate validation answers and compute numeric exact-match; validation correctness is never fed to J-good.

The primary figure shares training-step x-axis and shows, for every condition:

- held-out GSM8K exact-match with binomial 95% confidence intervals (plus the frozen base’s step-zero reference);
- held-out “good” J-score; and
- the reward actually optimized by that condition.

Also report KL, output length, and literal `good` frequency. For J-good, recompute J-score after masking positions containing literal `good`, and under both the original and re-fitted lenses. The verifiable baseline is a pipeline check: it must produce a positive held-out correctness trend. J-good succeeds internally if masked held-out J-score rises by ≥1 base-model standard deviation under both lenses; it helps reasoning only if exact-match beats the frozen-base reference and moves toward the verifiable baseline.

Deliver `fit_lens.py`, `reward.py`, `train.py`, `eval.py`, a pinned config, tests for answer parsing/reward/device behavior, saved metrics, and the comparison figure.

Sources: [Anthropic overview](https://www.anthropic.com/research/global-workspace), [paper](https://transformer-circuits.pub/2026/workspace/index.html), [J-lens code](https://github.com/anthropics/jacobian-lens), [TRL baseline](https://github.com/huggingface/trl).
