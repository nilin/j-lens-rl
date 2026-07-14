# J-lens RL on GSM8K

This repository tests whether a reward derived from a language model's own
J-lens word score can improve held-out, verifiable math accuracy. The current
model is `Qwen/Qwen2.5-0.5B-Instruct`; the primary outcome is greedy numeric
exact match on GSM8K-format problems.

The important distinction is:

- a J-only optimizer receives one task reward—the fixed J-lens score—and the
  configured GRPO KL regularizer; its training rows contain no gold answers;
- exact match is an observational evaluation and may not choose the training
  horizon or checkpoint in a confirmatory run; and
- a rising evaluation curve is descriptive. Statistical evidence requires a
  separate, paired, sealed evaluation across predeclared seeds.

Read [CONFIRMATORY_PROTOCOL.md](CONFIRMATORY_PROTOCOL.md) before running an
experiment. It is the authoritative protocol. `RESEARCH_LOG.md` contains the
exploratory history, including negative results; `audit.md` explains why the
old official-test results are not confirmatory evidence.

## Setup and tests

Use Linux, Python 3.10+, and an NVIDIA GPU with about 16 GB or more VRAM.

```bash
./setup.sh
source .venv/bin/activate
pytest -q
```

The repository pins the base-model and dataset revisions. Anthropic's J-lens
implementation is pinned to commit
`581d398613e5602a5af361e1c34d3a92ea82ba8e`; TRL is vendored from commit
`f3e9ac1005980fded7192682599c70749785fa9b`.

## Confirmatory run

The v1 candidate is frozen before outcomes are opened: WikiText-fitted
`solved`, layer 8, late-half mean, LR `3e-6`, six seeds, and a fixed step-25
endpoint. A matched sign-flipped J reward is the required control. One
exact-match-reward run is an optional pipeline check.

After committing all code and artifact metadata, the worktree must be clean:

```bash
git status --short
./run_confirmatory.sh prepare
./run_confirmatory.sh verify
./run_confirmatory.sh train-semantic
./run_confirmatory.sh curve
./run_confirmatory.sh train-controls
# Optional pipeline check:
./run_confirmatory.sh train-positive-control
./run_confirmatory.sh unlock
./run_confirmatory.sh final-treatment
./run_confirmatory.sh final-controls
./run_confirmatory.sh report
```

Preparation deterministically reconstructs 3,410 historically used raw
GSM8K-train indices and allocates the 4,063 unused ones into 200 exploratory,
400 one-shot curve, 3,000 sealed-final, and 463 untouched-reserve examples.
All 4,063 stay out of confirmatory training. Manifests and run outputs live in
ignored `.confirmatory/`, with hashes tied to the clean Git commit.

The curve gate is fixed before training: across the mean of all six semantic
seeds, step 5 must exceed step 0, step 10 must be at least step 5, and step 15
must be at least step 10. Runs always continue to step 25. The final set remains
locked unless all 12 semantic/sign-flip runs and this exact curve gate pass.
The gate saves a hashed figure containing every seed and the highlighted mean
curve at the predeclared nodes.

Significant positive evidence additionally requires all six sealed-set seed
effects to be positive (two-sided sign-test `p=0.03125`) and a positive
multi-seed mean change whose 95% crossed seed/item bootstrap interval excludes
zero. The sign-flip difference-in-differences and per-item paired tables are
reported, never used as alternative success definitions.

## Exploratory commands

The lower-level entry points remain useful for diagnostics:

```bash
fit-jlens --help
train-jlens-rl --help
eval-jlens-rl --help
compare-jlens-evals --help
plot-jlens-rl --help
```

Configs outside the `confirmatory_*` family are historical/exploratory. Do not
use `configs/jlens.json`, `configs/full_eval.json`, or the retired
`run_solved_layer_screen.sh` to make a confirmatory claim: those paths belong
to an adaptively reused test-monitor protocol.

Standalone evaluation writes per-example JSONL rather than only an aggregate:

```bash
eval-jlens-rl \
  --config configs/confirmatory_sealed_eval.json \
  --experiment-config configs/confirmatory_jlens_seed142.json \
  --adapter .confirmatory/runs/jlens_seed142/final \
  --indices-manifest .confirmatory/manifests/sealed_final_indices.json \
  --output-jsonl .confirmatory/evals/jlens_seed142.jsonl \
  --run-label jlens_seed142 \
  --batch-size 64 --skip-jlens-metric
```

Do not run that command before `./run_confirmatory.sh unlock`; the runner is the
guarded route. Per-example output includes source index, prompt hash,
completion, parsed prediction, correctness pair, literal-target audit, and
source/model/artifact provenance so reported changes can be reconstructed.

## Modal parallel runner

`modal_experiments.py` runs the same frozen protocol with at most five GPU
containers. It bakes the exact clean Git tree and frozen lens artifacts into
the image, uses a v2 Volume for distinct per-seed outputs, and never copies
`.env` or `modal.sh`. The durable remote orchestrator runs semantic seeds,
checks the fixed curve, runs sign-flips only on a pass, then performs the
guarded final analysis.

After `prepare`, install the pinned client, configure your local Modal profile,
and create the named W&B workload secret once:

```bash
.venv/bin/python -m pip install 'modal==1.5.2'
PATH="$PWD/.venv/bin:$PATH" bash modal.sh
read -r WANDB_API_KEY < .env
.venv/bin/modal secret create j-lens-rl-wandb WANDB_API_KEY="$WANDB_API_KEY"
unset WANDB_API_KEY
.venv/bin/modal run --detach modal_experiments.py
```

The launcher prints a function-call ID and uses Volume
`j-lens-rl-confirmatory-v1-20260714`. Monitor it with Modal's dashboard or CLI.
Download and archive the Volume promptly after completion because Volume v2 is
currently beta. See Modal's official guides for
[GPU selection](https://modal.com/docs/guide/gpu),
[parallel maps](https://modal.com/docs/guide/batch-processing), and
[Volume consistency](https://modal.com/docs/guide/volumes).
