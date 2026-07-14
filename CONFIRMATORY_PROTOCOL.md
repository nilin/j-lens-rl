# Confirmatory protocol v4: J-lens reward and GSM8K accuracy

Status: **conditionally selected and frozen; do not edit after preparation**.

V4 asks whether RL using only the intrinsic J-lens reward for the word
`solved` improves greedy GSM8K exact match. Correctness is observational during
training and is never a reward, stopping signal, checkpoint selector, or
hyperparameter selector.

## Why this follow-up is valid

V3 closed at its predeclared curve gate. Its mean curve was
`.435/.42125/.422375/.4185` at steps `0/5/10/15`; `curve_gate.json` recorded
`passed: false`. The V3 snapshot had ten semantic runs, no sign-flip run, no
unlock, no `evals/` directory, and only the curve gate and plot under
`evidence/`. Thus none of its 2,100 sealed-final outcomes was opened.

[`protocol_archive/v3_closeout.json`](protocol_archive/v3_closeout.json)
records that closeout and
[`protocol_archive/screen2_selection.json`](protocol_archive/screen2_selection.json)
records the precommitted candidate-selection screen. Their exact SHA-256 values
are hard-coded into the verifier, copied into prepared protocol state, and
checked again before every run.

Screen 2 evaluated four candidates on the already exposed retired V2 curve,
under a fixed priority rule. Their step `0/2/4/6` curves were:

| Priority | Candidate | Curve | Gate |
|---:|---|---|:---:|
| 1 | `ultradense5` | `.375/.3875/.370/.3675` | fail |
| 2 | `tail_taper` | `.375/.380/.380/.3825` | pass |
| 3 | `tempered_delta` | `.375/.3575/.3575/.3775` | fail |
| 4 | `layer_shrink` | `.375/.3675/.360/.380` | fail |

The first passing candidate, `tail_taper`, was therefore selected. Its later
steps `10/15/20/25` (`.3725/.3775/.3775/.385`) are archived as observational
and played no role in selection.

## Frozen treatment and control

All runs start independently from `Qwen/Qwen2.5-0.5B-Instruct` revision
`7ae557604adf67be50417f59c2c2f167def9a775`. Eight fresh seeds, 159--166,
share the same 1,000 training examples within seed, decoding, GRPO/LoRA
settings, KL coefficient, constant `3e-6` learning rate, zero warmup, and fixed
25-update endpoint.

The semantic treatment uses only one registered task reward: the calibrated
J-lens score for `solved`, sampled at stride 10 with these components:

1. layer 8, response fraction `[.5,.75)`, mean, weight `+1`;
2. layer 8, response fraction `[.75,1)`, mean, weight `+.25`.

The matched directional control negates every component (`-1` and `-.25`) and
changes nothing else. Sign-flip workers verify the stored passed semantic gate
before starting. Literal target tokens and the causal predecessor that predicts
them are masked and audited.

## Exact source-index firewall

V4 reconstructs the exact V3 allocation and requires these parent identities:

- V3 sealed-parent file SHA-256:
  `84da0c0472b4442b4f35406d1b1fbd3b956803e5f19bf51fc02f6db013224f7b`;
- V3 sealed-parent sorted-set SHA-256:
  `875334925160d6c0c49dd8cf1523e1aeb081fd90f6e4b08611eccb8394dbe4d5`.

Only that unopened 2,100-item parent is ordered by ascending
`SHA256("j-lens-rl-confirmatory-v4-screen2-2026-07-14:" + source_index)`:

| Purpose | Count | Manifest SHA-256 |
|---|---:|---|
| V4 curve | 400 | `ad348fe17d2e6bd6aac691d9bcdbb9da481f675305fa0e05c68e86dad97451c1` |
| V4 sealed final | 1,700 | `acd2d497dcf96b2f3355925bb34979b9b7b3301e4c394066fc54ea57d093b6e3` |
| untouched reserve | 64 | `cfbac5a2f4cf3cc94e1882bf412cdfc4af9c84347647fa9843dc09967f8a03a6` |

The exposed V1, V2, and V3 curves are retired. Every V4 curve/final/reserve row
and every retired curve row is excluded from training. Preparation rejects any
hash, count, overlap, parent-set, reserve, archive, config, artifact, or clean
Git mismatch.

## Fixed curve gate

Greedy observational validation runs at steps
`0,2,4,6,10,15,20,25`. Training always continues to step 25. The sole gate is
the eight-seed mean at the first four nodes:

```text
mean(step2) > mean(step0)
mean(step4) >= mean(step2)
mean(step6) >= mean(step4)
```

No later nodes, seed subset, or checkpoint may substitute. If the gate fails,
the run closes negatively: controls do not start and final data remain sealed.
The gate and plot are write-once; controls verify them against all semantic
histories.

## One sealed outcome collection and acceptance

After all eight semantic runs, a passing curve, all eight matched sign flips,
and a successful 16-run provenance check, unlock hashes every run artifact and
adapter. Modal then submits one immutable 17-label collection: base, eight
semantic adapters, and eight sign-flip adapters. Up to eight L40S workers may
queue it in waves, but no outcome analysis occurs until all 17 files finish.
There is no intermediate semantic-only decision or analysis stage.

The 17 evaluations use the same 1,700 rows, generation settings, clean commit,
source-tree fingerprint, model/lens/calibration, and pinned software/runtime.
The verifier reconstructs prompts and gold correctness from the pinned dataset
and recomputes the combined semantic and specificity report.

Call the result significant positive evidence only if every check passes:

1. the fixed `0/2/4/6` curve gate passes;
2. semantic-minus-base mean accuracy is positive;
3. its crossed seed/item 95% lower bound is above zero;
4. all eight seed effects are strictly positive, with no tie or negative,
   giving exact two-sided sign-test `p = 0.0078125`;
5. semantic-minus-sign-flip difference-in-differences is positive; and
6. its crossed seed/item 95% lower bound is above zero;
7. every split, artifact, record, runtime, source-tree, and literal-target audit
   passes.

## Execution

```bash
../j-lens-rl/.venv/bin/pytest -q
git status --short                 # empty
./run_confirmatory.sh prepare
./run_confirmatory.sh verify
./run_confirmatory.sh train-semantic
./run_confirmatory.sh curve
./run_confirmatory.sh train-controls
./run_confirmatory.sh unlock
./run_confirmatory.sh final-evaluation
./run_confirmatory.sh report
```

The Modal route is `modal_experiments.py`, app
`j-lens-rl-confirmatory-v4`, fresh Volume
`j-lens-rl-confirmatory-v4-20260714a`, L40S, and `max_containers=8`. Modal
preserves the semantic -> gate -> sign-flip -> unlock -> unconditional sealed
batch -> combined report order. It is an execution backend only and does not
alter any frozen choice.

Never overwrite an output, rerun until favorable, inspect a sealed outcome
before unlock, or call a later node significant. A failed curve or final
criterion is a valid negative result.
