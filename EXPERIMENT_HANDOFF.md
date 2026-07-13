# Experiment handoff

Updated: 2026-07-13 UTC. Branch: `main`.

## Objective and acceptance gate

Train `Qwen/Qwen2.5-0.5B-Instruct` with a reward derived only from an internal
Jacobian-lens notion of `solved`, and demonstrate an improvement in held-out,
verifiable GSM8K numeric exact match. Internal-score increases are diagnostics,
not success. A candidate is not accepted from the 200-example monitor alone:
verify the selected checkpoint against the frozen base with the same standalone
evaluator on all 1,319 GSM8K test examples, and replicate across a second seed.

Stop a screen after its first 25-step evaluation if exact match does not beat
that run's step-zero value. Do not continue known-flat runs merely because their
internal reward rises.

## Reproduce the environment

```bash
git clone https://github.com/nilin/j-lens-rl.git
cd j-lens-rl
./setup.sh
.venv/bin/pytest -q
```

W&B project: `nilinabra-spare-time/j-lens-rl`. `.env` is intentionally ignored
and contains only the raw W&B API key (not `KEY=value`). Before a run:

```bash
export WANDB_API_KEY="$(tr -d '\r\n' < .env)"
```

Do not commit or print the key. Lens files, run outputs, and W&B local state are
also ignored; regenerate them with the commands below.

## Authoritative state

Read `RESEARCH_LOG.md` for the experiment table, exact metrics, decisions, and
W&B run IDs. The implementation intentionally stays on vendored TRL v1.0.0; the
only TRL delta exposes the unwrapped policy and rollout token IDs to the custom
reward. Do not change TRL unless a demonstrated blocker requires it.

Current conclusions:

- The generic WikiText-fitted `solved` lens with a layer-8 late-half mean
  readout and LR `3e-6` now passes the exact acceptance gate. Seed 42 improved
  at monitor steps 25 and 50 and scored 408/1,319 on the full test. Seed 43
  improved at steps 10, 20, 25, and 35 and scored 407/1,319. The same frozen
  base scores 405/1,319. W&B runs: `kwk4m0ev`, `wsg6wioj`.
- Treat this as a small replicated directional effect (+3 and +2 answers), not
  a large or statistically precise gain; the full-test confidence intervals
  overlap heavily.
- A nine-readout composite reached 62.0% offline pair accuracy but decreased the
  200-example monitor from 32.5% to 32.0%.
- A larger 200-prompt screen found layer-20 final-token at 58.5%, layer-8
  late-half mean at 56.1%, and the 18-way composite at 62.9%. Max/quarter-window
  readouts were near chance, so no new RL run was justified.
- The matched exact-match-reward control at LR `3e-6` was also flat, 32.5% to
  32.5% at step 25 (W&B `37nto25a`).
- A clean WikiText-fitted `happy` reward decreased the 200-example monitor from
  33.5% to 32.5% at step 25 and 31.5% at step 100 (W&B `kxor0zvs`), with 0%
  literal `happy` usage. Treat it as a negative result.

## Reproduce the accepted run

Regenerate the generic WikiText lens (expected held-out calibration mean
`-19.0812344828`, standard deviation `3.9094524086`, target token `27956`):

```bash
.venv/bin/fit-jlens \
  --corpus wikitext \
  --output artifacts/qwen25_05b_solved_lens.pt \
  --calibration-output artifacts/qwen25_05b_solved_calibration.json \
  --target-word solved --num-prompts 100 --calibration-prompts 50 \
  --layers 8,14,20 --dim-batch 16 --seed 42
```

Run both seeds online in W&B:

```bash
export WANDB_API_KEY="$(tr -d '\r\n' < .env)"
.venv/bin/train-jlens-rl \
  --config configs/jlens_late_8_lr3e6_dense_eval.json --wandb-mode online
.venv/bin/train-jlens-rl \
  --config configs/jlens_late_8_lr3e6_dense_eval_seed43.json --wandb-mode online
```

Verify the saved adapters against all 1,319 examples:

```bash
.venv/bin/eval-jlens-rl --config configs/full_eval.json \
  --adapter runs/jlens_solved_late_8_lr3e6_dense_eval/checkpoint-50 \
  --skip-jlens-metric --batch-size 16
.venv/bin/eval-jlens-rl --config configs/full_eval.json \
  --adapter runs/jlens_solved_late_8_lr3e6_dense_eval_seed43/final \
  --skip-jlens-metric --batch-size 16
```

Training uses reward weights `[0, 1]`: the GSM8K verifier is computed only as
an audit metric and has exactly zero contribution to the scalar reward or
gradient. The lens is fitted only on WikiText; no GSM8K questions, answers,
grades, validation examples, or test examples enter reward construction.

## Rejected experiment: GSM8K-reference domain lens

Do not fit or use a lens on GSM8K reference solutions. Although that path did
not update policy weights or touch held-out evaluation examples, fitting the
reward transport on correct training solutions makes the signal indirectly
solution-informed and weakens the intended internal-satisfaction-only claim.
The implementation, config, and generated artifacts for this path were removed.

Reference-solution fitting remains prohibited. A clean domain-matched variant
may use the frozen base model's own ungraded GSM8K-training completions, drawn
from prompts disjoint from the RL training subset. Save that rollout corpus for
auditability. It must never read reference answers, verifier scores, correctness
labels, validation examples, or test examples. Any candidate must pass grouped
alignment screening before RL; correctness remains evaluation-only under the
full-test and second-seed gate above.
