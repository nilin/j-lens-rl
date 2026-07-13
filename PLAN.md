# J-space reward RL on GSM8K — one-page plan

## Question and primary comparison

Can an internal J-lens reward improve **held-out, verifiable GSM8K accuracy** for `Qwen/Qwen2.5-0.5B-Instruct`? Run two paired conditions in which the **reward function is the only experimental variable**:

1. **Verifiable baseline:** RL reward is exact equality between the extracted final number and GSM8K ground truth (`0/1`).
2. **J-solved:** RL reward uses only the policy’s internal J-lens score for “solved.”

Both start from the same untouched checkpoint and use the same initialization seed, examples in the same order, prompts, eight sampled completions, decoding settings, LoRA targets/rank, optimizer and schedule, KL coefficient, batch/accumulation, token budget, update count, evaluation set, and stopping rules. Do not initialize J-solved from the correctness-trained model. Evaluate the frozen base once at step zero as a reference, not as a third training condition.

## Minimal implementation

Use Anthropic’s Apache-2.0 [`anthropics/jacobian-lens`](https://github.com/anthropics/jacobian-lens). Fit its Jacobian transport once on ~100 generic 128-token sequences at three middle/late layers. Verify which whitespace/case variants of `solved` are single Qwen tokens and use all valid IDs.

For each completion, rerun a gradient-free policy forward pass and retain response-position hidden states. J-reward is the mean over chosen layers and positions of the target tokens’ J-lens log-probability mass, standardized against the base model and clipped to ±5 standard deviations. Mask sampled positions containing literal target tokens. Freeze the fitted Jacobians; train rank-8 LoRA on attention/MLP projections. Re-fit the lens after training to diagnose exploitation of a stale lens.

Use a vendored, commit-pinned Hugging Face TRL [`GRPOTrainer`](https://github.com/huggingface/trl) for generation, the GRPO objective, KL regularization, checkpointing, and W&B logging. A narrow trainer patch exposes the unwrapped policy and rollout token IDs to reward callables so J-solved can run an additional hidden-state forward pass. Do not otherwise modify TRL. Both rewards are calculated and logged in both runs; only their weights change (`[1,0]` versus `[0,1]`).

## Experiment and plots

Use a fixed 1,000-example training subset and 200 disjoint validation examples. Run a 50-update smoke test, then ~500 updates. Every 25 updates, deterministically generate validation answers and compute numeric exact-match; validation correctness is never fed to J-solved.

The primary figure shows held-out GSM8K exact match with binomial 95% confidence intervals for both conditions, including the frozen base at step zero. Also report KL, output length, literal `solved` frequency, and rollout rewards as diagnostics. The correctness-reward baseline is a pipeline check and should produce a positive held-out trend. J-solved succeeds only if its held-out exact match beats the frozen-base reference and moves toward the correctness-reward baseline; an increased internal score alone is not success.

Deliver `fit_lens.py`, `reward.py`, `train.py`, `eval.py`, a pinned config, tests for answer parsing/reward/device behavior, saved metrics, and the comparison figure.

Sources: [Anthropic overview](https://www.anthropic.com/research/global-workspace), [paper](https://transformer-circuits.pub/2026/workspace/index.html), [J-lens code](https://github.com/anthropics/jacobian-lens), [TRL baseline](https://github.com/huggingface/trl).
