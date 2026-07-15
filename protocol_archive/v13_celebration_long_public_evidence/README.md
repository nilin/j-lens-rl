# V13 celebration-long public terminal evidence

V13 is a terminal, failed-closed attempt. All eight treatment/control jobs trained
through update 20 and published reconstructable public run bundles to W&B, but
the post-training verifier accepted only two workers. Six otherwise terminal
runs were rejected because two independently logged reward-standard-deviation
reductions differed by exactly two float32 ULPs while the verifier allowed only
one. The failure occurred before the official curve artifact, unlock, or final
evaluation could be created.

The public treatment curve also fails its registered gate independently of that
infrastructure error: its update-10 mean is one correct answer out of 1,600
below its update-4 mean. This archive therefore contains negative/partial public
development evidence, not preregistered inferential evidence.

## Execution identity

| Field | Value |
|---|---|
| Modal volume | `j-lens-rl-confirmatory-v13-celebration-long-20260714a` (`vo-PmHsR7sciyRgYPUZ8JE8Dt`) |
| Modal app | `ap-AffsuHrmJkl2drAttKYaZ0` |
| Claim | `5d6c4eac2281450d97ac4870be98a2db` |
| Root call | `fc-01KXHFHKS166MWQ7SPS89HZZQJ` |
| Execution contract SHA-256 | `235789a0b92900552b67e704682817ffdf0ad291ea0574319177a6e2d8bfe40f` |
| Preflight repository commit | `218c33f657ecd597ac55f4f689ddd660691c85f4` |
| Runtime synthetic commit | `fb928bb99c0add626a7b513810fab7a67ad2a74b` |
| Runtime source-tree SHA-256 | `c9e1b3d0d2d9e7ef638a8327325fb2475a8855582d222123a3ae971327d9953d` |
| W&B group | `confirm-v13-celebration-long-u4-u10-u20` |
| Terminal status | `failed_closed`; retry/resume forbidden |

The treatment is the registered positive celebration-family J-lens reward for
`yay`, `great`, `success`, and `nice`. The matched sign-flip condition changes
only the two component weights from `+1, +0.25` to `-1, -0.25`; seed-matched
treatment/control runs use identical training rows.

## Exact public curves

Each value is exact match on the same exposed 400-row development set. The
columns are baseline, update 4, update 10, and update 20.

| Run | Curve | Terminal verifier | W&B |
|---|---|---|---|
| `jlens_seed228` | `.3825, .3875, .3750, .3925` | rejected: 2 ULP at update 11 | [run](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/confirm-v13-celebration-long-jlens-seed228) |
| `jlens_seed229` | `.3825, .3700, .3950, .3925` | rejected: 2 ULP at update 14 | [run](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/confirm-v13-celebration-long-jlens-seed229) |
| `jlens_seed230` | `.3825, .3850, .3925, .4025` | verified completion | [run](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/confirm-v13-celebration-long-jlens-seed230) |
| `jlens_seed231` | `.3825, .3975, .3750, .3775` | rejected: 2 ULP at update 9 | [run](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/confirm-v13-celebration-long-jlens-seed231) |
| `signflip_seed228` | `.3825, .3975, .4000, .3900` | rejected: 2 ULP at update 5 | [run](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/confirm-v13-celebration-long-signflip-seed228) |
| `signflip_seed229` | `.3825, .3800, .3900, .3850` | rejected: 2 ULP at updates 17, 18 | [run](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/confirm-v13-celebration-long-signflip-seed229) |
| `signflip_seed230` | `.3825, .3900, .3875, .3975` | rejected: 2 ULP at update 14 | [run](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/confirm-v13-celebration-long-signflip-seed230) |
| `signflip_seed231` | `.3825, .3750, .3875, .4000` | verified completion | [run](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/confirm-v13-celebration-long-signflip-seed231) |

The reconstructed aggregate curves are:

| Condition | Baseline | Update 4 | Update 10 | Update 20 |
|---|---:|---:|---:|---:|
| Positive celebration treatment | `.382500` (612/1600) | `.385000` (616/1600) | `.384375` (615/1600) | `.391250` (626/1600) |
| Matched sign flip | `.382500` (612/1600) | `.385625` (617/1600) | `.391250` (626/1600) | `.393125` (629/1600) |

The registered treatment gate requires `M4 > M0`, `M10 >= M4`, and
`M20 >= M10`. Those checks are respectively true, false, and true. The failed
middle inequality has margin `-1/1600 = -0.000625`, so the overall gate fails.
No official `curve_gate.json` exists because worker verification failed first.

## Descriptive public sign tests

These are post-hoc diagnostics on already exposed development curves. They are
exact two-sided sign tests over the four registered seeds, with ties excluded;
the p-values are unadjusted across three visible checkpoints and three
contrasts. Checkpoints are not pooled as independent observations.

| Update | Contrast | Seed signs | Mean effect | Exact p |
|---:|---|---:|---:|---:|
| 4 | treatment - signflip | 1 positive / 3 negative | `-0.000625` | `.625` |
| 4 | treatment - base | 3 positive / 1 negative | `+0.002500` | `.625` |
| 4 | signflip - base | 2 positive / 2 negative | `+0.003125` | `1.000` |
| 10 | treatment - signflip | 2 positive / 2 negative | `-0.006875` | `1.000` |
| 10 | treatment - base | 2 positive / 2 negative | `+0.001875` | `1.000` |
| 10 | signflip - base | 4 positive / 0 negative | `+0.008750` | `.125` |
| 20 | treatment - signflip | 3 positive / 1 negative | `-0.001875` | `.625` |
| 20 | treatment - base | 3 positive / 1 negative | `+0.008750` | `.625` |
| 20 | signflip - base | 4 positive / 0 negative | `+0.010625` | `.125` |

The two `.125` diagnostics favor sign flip over the base model, which points in
the opposite direction from the registered positive-celebration contrast.
Signflip-minus-base was not a registered acceptance test. V13's preregistered
tests remain treatment-minus-signflip and treatment-minus-base on the untouched
900-row final, conditional on a passed curve and all-eight verification. Those
conditions were not met, the protected final was not opened, and V13 has no
preregistered final p-value.

## Verifier incident

Every archived training log has the expected single J-lens reward schema and no
GSM8K correctness reward. The only failing terminal-verifier predicate is the
duplicate `reward_std` comparison. The accepted runs have an observed maximum
of one float32 ULP; every rejected run has one or more rows at exactly two ULPs.
The raw logs and exact offending updates are recorded in `inventory.json`.

This false rejection does not change the scientific curve outcome above. It
does explain why only two genuine `*.completion.json` records exist despite all
eight terminal run-result manifests and W&B terminal receipts being present.

## Public boundary and reconstruction

The archive contains 98 byte-preserved source artifacts plus this README and a
derived inventory. It deliberately excludes:

- `sealed_eval.json`;
- the protected final manifest, indices, prompts, completions, and outcomes;
- final unlock/authorization records and all eval outputs;
- checkpoint, adapter, tokenizer, optimizer, RNG, and model-weight payloads.

Public registration and run-result files retain hashes and path references to
some excluded artifacts so the attempt can be identified, but no excluded
payload bytes are present. The large training-artifact hashes are preserved in
each `run_result_manifest.json` and W&B terminal receipt.

From this directory, validate the archive and inspect the exact derived data:

```bash
sha256sum -c CHECKSUMS.sha256
jq '.public_development_analysis' inventory.json
```

`inventory.json` records every source file's byte count, SHA-256, and role,
along with execution identities, per-run curves, W&B artifacts, verifier
classification, exact sign-test inputs, and the protected-boundary assertions.
`CHECKSUMS.sha256` covers every archive file except itself.
