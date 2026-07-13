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

- WikiText-fitted `solved` readouts weakly rank correctness offline, but no
  J-reward result has replicated a full-test exact-match gain.
- Layer-8 late-half mean had one +3/1,319 result at seed 42, then -1/1,319 at
  seed 43. Treat it as noise.
- A nine-readout composite reached 62.0% offline pair accuracy but decreased the
  200-example monitor from 32.5% to 32.0%.
- A larger 200-prompt screen found layer-20 final-token at 58.5%, layer-8
  late-half mean at 56.1%, and the 18-way composite at 62.9%. Max/quarter-window
  readouts were near chance, so no new RL run was justified.
- The matched exact-match-reward control at LR `3e-6` was also flat, 32.5% to
  32.5% at step 25 (W&B `37nto25a`).

## Current experiment: domain-matched lens

The leading hypothesis is domain mismatch: the original Jacobian transport and
calibration use WikiText, while rewards are applied to chat-formatted math
reasoning. Fit a lens on GSM8K reference reasoning transcripts:

```bash
.venv/bin/fit-jlens \
  --corpus gsm8k \
  --output artifacts/qwen25_05b_solved_lens_gsm8k.pt \
  --calibration-output artifacts/qwen25_05b_solved_calibration_gsm8k.json \
  --target-word solved \
  --num-prompts 100 --calibration-prompts 50 \
  --layers 8,14,20 --dim-batch 16 --seed 42
```

This fit completed successfully on the originating machine. Its held-out
calibration is mean `-18.3153293482`, standard deviation `4.7388755305`, and
target token ID `27956`. Because `artifacts/` is ignored, another machine must
run the command above; matching these values is the reproducibility check.

The tracked `configs/jlens_gsm8k_lens.json` points at those domain artifacts.
Then run:

```bash
.venv/bin/python -m jlens_rl.analyze_alignment \
  --config configs/jlens_gsm8k_lens.json \
  --prompts 200 --generations 8 \
  --output artifacts/solved_alignment_gsm8k_lens_200.json
```

Compare five-fold prompt-group CV and simple-readout pair accuracy to the
WikiText baselines above. Only launch RL if a simple readout is materially
stronger, or if a composite improvement is large enough to distinguish it from
the previously failed 62.0% mixture. Use LR `3e-6`, patience 1, online W&B,
and the unchanged step-25 validation gate.

If the domain lens is not stronger, the next useful direction is a directly
cross-validated correctness probe constrained to features derived from the
`solved` Jacobian readouts. Keep its fitting labels disjoint by prompt and do
not call probe accuracy or internal reward movement a successful outcome.
