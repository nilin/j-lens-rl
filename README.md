# J-lens RL on GSM8K

This repository tests whether an internal-state reward improves the outcome we
care about: **held-out, verifiable GSM8K exact match** on
`Qwen/Qwen2.5-0.5B-Instruct`.
Both runs use the same group-relative policy-gradient loop, LoRA setup, seed,
examples, eight rollouts per prompt, KL term, and evaluation. Only the reward callable differs:

- `configs/gsm8k.json`: verifiable numeric exact-match reward.
- `configs/jlens.json`: mean standardized J-lens log-probability mass for
`solved`, sampled every 20 response tokens. Literal target-token positions are
excluded to reduce the easiest reward-hacking path.

`score_start_fraction` selects the response window used by the J reward (`0.5`
means the later half), while `score_layers` selects any subset of the fitted
layers. These allow targeted alignment screens without refitting the lens.

The J-lens implementation is pinned to Anthropic commit
`581d398613e5602a5af361e1c34d3a92ea82ba8e`. TRL v1.0.0 is vendored under
`trl/` from upstream commit `f3e9ac1005980fded7192682599c70749785fa9b`. Its standard `GRPOTrainer` handles
generation, optimization, checkpointing, and W&B logging. Our narrow
patch exposes the policy and rollout token IDs to custom reward functions so the
J-lens reward can inspect hidden states.

## GPU setup

Use Linux, Python 3.10+, a recent NVIDIA driver, and a CUDA GPU with roughly
16 GB or more VRAM. The default 0.5B model is deliberately small.

```bash
./setup.sh
source .venv/bin/activate
wandb login
pytest -q
```

If your driver requires a different PyTorch CUDA wheel, install that wheel first,
then rerun `pip install -e '.[dev]'`. The package versions remain recorded in
`pyproject.toml`.

## 1. Fit and calibrate the lens (once)

```bash
fit-jlens \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --target-word solved \
  --layers 8,14,20 \
  --num-prompts 100 \
  --output artifacts/qwen25_05b_solved_lens.pt \
  --calibration-output artifacts/qwen25_05b_solved_calibration.json
```

Fitting is the expensive one-time stage. It checkpoints after every prompt and
resumes automatically. Increase `--dim-batch` if the GPU has spare memory;
decrease it after an OOM.

## 2. Smoke test both rewards

The overrides below perform one optimizer update. For a quick smoke test, also
temporarily set `validation_examples` to a small value in `configs/common.json`.

```bash
train-jlens-rl --config configs/gsm8k.json --updates 1 --output-dir runs/smoke-gsm8k --wandb-mode offline
train-jlens-rl --config configs/jlens.json --updates 1 --output-dir runs/smoke-jlens --wandb-mode offline
```

## 3. Run the matched experiment

Run these from the same clean base model; neither consumes the other's checkpoint.

```bash
train-jlens-rl --config configs/gsm8k.json
train-jlens-rl --config configs/jlens.json
plot-jlens-rl --output runs/comparison.png
```

Each run writes its resolved configuration, TRL trainer state/log history,
periodic LoRA checkpoints, and final adapter. It also logs to the `j-lens-rl`
W&B project by default. Both rollout rewards are computed in both runs; reward
weights are the sole experimental difference. A separate fixed, greedy
validation callback measures held-out GSM8K exact match at step zero and every
25 updates. This validation score is never used as reward in the J-lens run and
is the primary result plotted by `plot-jlens-rl`. Runs stop early after two
consecutive validation evaluations without a new best exact-match score; this
patience avoids spending the full budget on clearly flat variants while being
less sensitive to one noisy 200-example measurement.

## Standalone evaluation

```bash
eval-jlens-rl --config configs/jlens.json --adapter runs/jlens_solved_reward/final
```

To refit the lens on a trained policy—a diagnostic for stale-lens exploitation—run:

```bash
fit-jlens \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter runs/jlens_solved_reward/final \
  --target-word solved \
  --layers 8,14,20 \
  --output artifacts/qwen25_05b_solved_refit_lens.pt \
  --calibration-output artifacts/qwen25_05b_solved_refit_calibration.json
```

Then point a copy of the evaluation config at those two refitted artifacts.
This diagnostic does not change the primary success criterion: held-out exact
match relative to the frozen base and correctness-reward control.
