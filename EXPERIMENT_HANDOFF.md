# Experiment handoff

Updated: 2026-07-14 UTC. The next agent should execute, not redesign,
[CONFIRMATORY_PROTOCOL.md](CONFIRMATORY_PROTOCOL.md).

## Current conclusion

The code now supports a genuinely J-only task-reward path: gold answers are
removed from J training rows and only the J reward callable is registered.
The repository does **not** yet contain statistically significant evidence
that this improves held-out GSM8K accuracy. Earlier gains of two or three
official-test answers were selected through adaptive test reuse and are
exploratory. The clean affect/error searches were negative.

Do not describe an internal-score increase, a selected 200-example peak, or an
old official-test score as success. Preserve all negative outcomes.

## Your assignment

Use the fixed v2 candidate and get the predeclared evidence as quickly as
hardware safely allows:

- six semantic `solved` J-reward runs, seeds 142–147;
- six matched sign-flipped runs, the same seeds and training examples;
- optionally, one seed-142 exact-match reward run as a pipeline check;
- fixed step-25 endpoints with observational evaluations every five updates;
- then, only after the gate, one frozen-base and all paired adapter evaluations
  on the sealed 2,900-example manifest.

The exact curve criterion is the six-seed mean at steps `0/5/10/15`:
step 5 must be above baseline, followed by two non-downward steps. Do not hunt
for another triple, pick a favorable seed, stop from correctness, or select a
checkpoint. Steps 20 and 25 are logged, and step 25 is always the endpoint.

The significant-evidence criterion is separate: all six semantic sealed-set
effects must be positive (two-sided seed sign-test `p=0.03125`), the positive
mean paired change must have a 95% crossed seed/item bootstrap interval that
excludes zero, and the positive matched sign-flip difference-in-differences
must also have a crossed 95% interval above zero.

## Start here

The parent agent will commit the audited fixes. Do not prepare or train until:

```bash
cd /j-lens-rl
.venv/bin/pytest -q
git status --short
```

The tests must pass and `git status --short` must print nothing. Then:

```bash
./run_confirmatory.sh prepare
./run_confirmatory.sh verify
```

Preparation creates ignored `.confirmatory/manifests/` files and a state file
that fingerprints the clean commit, pinned model/dataset revisions, configs,
fresh split manifests, lens, and calibration. If preparation refuses, fix the
cause rather than bypassing it. Never delete or regenerate prepared manifests
after seeing any v2 correctness.

V1 is retired: its historical-index reconstruction omitted setup run
`xufk8x08`, and its partial Modal attempt exposed its curve before failing
source-provenance validation. V2 excludes that setup run's training indices,
retires all 400 v1 curve indices, and uses a new 400-item curve plus a still
sealed 2,900-item final set. Do not reuse either v1 Volume.

## Fast execution order

On one GPU:

```bash
./run_confirmatory.sh train-semantic
./run_confirmatory.sh curve
./run_confirmatory.sh train-controls
# Optional, nonblocking pipeline check:
./run_confirmatory.sh train-positive-control
./run_confirmatory.sh unlock
./run_confirmatory.sh final-treatment
./run_confirmatory.sh final-controls
./run_confirmatory.sh report
```

The required compute is the 12 semantic/sign-flipped runs. The exact-match
control is optional and does not block unlock. To save wall time on multiple
GPUs, assign distinct config files to distinct devices/agents, for example:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/train-jlens-rl \
  --config configs/confirmatory_jlens_seed142.json --wandb-mode online
CUDA_VISIBLE_DEVICES=1 .venv/bin/train-jlens-rl \
  --config configs/confirmatory_signflip_seed142.json --wandb-mode online
```

Continue through seeds 143–147 without running two processes against the same
output directory. The train command rejects a nonempty directory. All
conditions use training-generation `min_new_tokens=64` to prevent the observed
five-token collapse; final greedy evaluation intentionally has no minimum.

For the fastest guarded path, `modal_experiments.py` submits a durable remote
pipeline capped at five simultaneous pinned L40S containers. It queues the sixth seed,
applies the same curve gate before controls, and parallelizes final paired
evaluation only after unlock. Follow the credential-safe setup in `README.md`;
never upload `modal.sh` or `.env`. The Modal Volume is the experiment archive
until it is downloaded back into local `.confirmatory/`.

Batch 64 is frozen for the curve and final evaluators on the pinned L40S runtime.
If a pre-preparation memory smoke test fails, lower every condition to batch 32,
commit, and prepare a new protocol version. Never change it after preparation.

## What the guards check

`./run_confirmatory.sh unlock` checks the following before exposing final data:

- the working tree and HEAD still match the prepared state;
- all 12 required runs used the pinned config/artifacts/revisions;
- every run ended at step 25 without correctness stopping;
- each matched seed used the same 1,000 training source indices;
- no historically unused curve/final/reserve index entered training;
- each history contains exactly steps `0,5,10,15,20,25`; and
- the one predeclared mean curve passed.

Unlock also freezes hashes for every final adapter and run audit file. Final
verification reloads the pinned dataset and independently recomputes prompt
hashes, parsed predictions, and correctness from the saved completions.

Final evaluation writes auditable per-item JSONL and compares the six semantic
seeds jointly. Evaluate treatment first; if it is negative, record that result
without tuning. Evaluate sign-flip controls next. The optional exact-match
control may be omitted if compute is scarce; say so explicitly.

## Secrets and outputs

W&B project: `nilinabra-spare-time/j-lens-rl`. `.env` contains only the raw API
key and is ignored. The runner loads it without printing it. Do not commit the
key, `.confirmatory/`, `runs/`, `wandb/`, or large artifacts. Preserve the
ignored protocol state and JSONL outputs with the experiment archive because
their hashes connect the claim to the committed code.

If interrupted, do not rerun a seed until its curve looks favorable. The
guarded runner does not silently resume or overwrite training; record the run
as interrupted and use a separately declared replacement rule/protocol. It may
reuse a completed final JSONL only after checking all rows and provenance.
