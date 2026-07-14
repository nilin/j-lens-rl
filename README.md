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

V2 and V3 are valid negative curve-gate results and opened no sealed-final
outcome. A precommitted screen on retired development data selected the frozen
V4 `tail_taper` reward: layer-8 means over response fractions `[.5,.75)` and
`[.75,1)`, weights `1/.25`, stride 10. V4 uses eight fresh seeds 159--166,
constant LR `3e-6`, zero warmup, a fixed step-25 endpoint, and matched sign
flips with both weights negated.

After committing all code and artifact metadata, the worktree must be clean:

```bash
git status --short
./run_confirmatory.sh prepare
./run_confirmatory.sh verify
./run_confirmatory.sh train-semantic
./run_confirmatory.sh curve
./run_confirmatory.sh train-controls
./run_confirmatory.sh unlock
./run_confirmatory.sh final-evaluation
./run_confirmatory.sh report
```

Preparation verifies the hashed V3 closeout and selection archives, then
rehashes only V3's never-opened 2,100-item final parent into a 400-item V4
curve and 1,700-item sealed final. The V3 800-item curve is retired and the
64-item reserve is unchanged. Every outcome and retired-curve index stays out
of training. Ignored `.confirmatory/` manifests and outputs are tied to the
clean Git commit and source-tree fingerprint.

Across all eight semantic seeds, the frozen gate is step 2 above step 0, then
non-downward steps 4 and 6. Runs always continue to step 25. The final set
remains locked unless the gate and all 16 semantic/sign-flip runs verify.

After unlock, base plus all eight semantic and eight sign-flip adapters are
submitted as one fixed 17-label sealed collection before analysis. Significant
positive evidence requires all 8/8 semantic effects strictly positive
(`p=.0078125`), positive semantic and specificity means, crossed 95% lower
bounds above zero for both, and all provenance/record checks.

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
  --experiment-config configs/confirmatory_jlens_seed159.json \
  --adapter .confirmatory/runs/jlens_seed159/final \
  --indices-manifest .confirmatory/manifests/sealed_final_indices.json \
  --output-jsonl .confirmatory/evals/jlens_seed159.jsonl \
  --run-label jlens_seed159 \
  --batch-size 64 --skip-jlens-metric
```

Do not run that command before `./run_confirmatory.sh unlock`; the runner is the
guarded route. Per-example output includes source index, prompt hash,
completion, parsed prediction, correctness pair, literal-target audit, and
source/model/artifact provenance. The verifier reloads the pinned dataset and
recomputes prompt hashes, predictions, and correctness from each completion.

## Modal parallel runner

`modal_experiments.py` runs the same frozen protocol with at most eight pinned
L40S containers. It bakes the exact clean Git tree and frozen lens artifacts into
the image, uses a fresh v4 Volume for distinct per-seed outputs, and never copies
`.env` or `modal.sh`. The durable remote orchestrator runs semantic seeds,
checks the fixed curve, runs sign-flips only on a pass, then submits the full
fixed 17-label sealed batch and analyzes only after it completes.

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

The launcher prints a function-call ID and uses the v4 Volume named in
`modal_experiments.py`. Monitor it with Modal's dashboard or CLI.
Download and archive the Volume promptly after completion because Volume v2 is
currently beta. See Modal's official guides for
[GPU selection](https://modal.com/docs/guide/gpu),
[parallel maps](https://modal.com/docs/guide/batch-processing), and
[Volume consistency](https://modal.com/docs/guide/volumes).
