# J-Lens emotional-word screen forensic bundle

This directory is an offline, compact audit bundle for the two completed Modal
screens run on 2026-07-14. It preserves the raw scalar histories needed to
reconstruct every experimental W&B series, the exact resolved configurations,
data-index selections, calibration records, run manifests, screening decisions,
and checkpoint-25 trainer state. Large model/optimizer blobs are intentionally
not duplicated; their byte-level SHA-256 inventories remain embedded in each
`screen_result.json` and the source artifacts remain on the named Modal Volumes.

## Scientific status

- The active research policy is **emotionally charged reward words only**.
  `solved_*` entries under `family/runs/` are frozen historical comparators and
  are ineligible for selecting or launching future experiments.
- These are exploratory results on an explicitly exposed 400-example GSM8K
  development curve, not untouched confirmatory evidence.
- The single-word gate was fixed as `step2 > step0`, `step4 >= step2`, and
  `step6 >= step4`. Only `joy` passed: `0.3825 -> 0.3900 -> 0.3900 -> 0.4100`.
  This is a non-decreasing three-checkpoint early trajectory with the first
  post-training checkpoint above baseline; it contains a flat step and should
  not be described as three strict increases.
- The celebration taper arm has the cleanest later three-checkpoint rise:
  step 15/20/25 is `0.4000 -> 0.4100 -> 0.4200`, all above its `0.3825`
  baseline. No multiplicity-adjusted significance claim follows from either
  selected trajectory.
- Literal emission cannot explain the observed single-word curves: all eight
  emotional single-word arms have a literal-target completion rate of zero at
  every evaluation checkpoint. Celebration taper emitted a literal target in
  0.25% of evaluated completions at each checkpoint; profanity arms were zero.

## Bundle layout

For each run, the following files are retained:

- `resolved_config.json`: exact configuration consumed by the trainer.
- `run_manifest.json`: commit, dirty state, source-tree digest, pinned model and
  dataset revisions, hardware, lens/calibration/data digests.
- `data_indices.json`: exact 1,000 training and 400 validation indices.
- `validation_history.jsonl`: step-0 and scheduled greedy GSM8K evaluations.
- `log_history.json`: the full trainer history (steps 1-25 and final summary).
- `checkpoint-25/trainer_state.json`: trainer-owned step history/state.
- `screen_result.json`: normalized curve, literal rates, gate decision, and
  recursive final-adapter/checkpoint SHA-256 inventories.
- `README.md`: generated model card containing the W&B run URL and framework
  versions.

The `artifacts/` directories contain the exact small calibration payloads and
their manifests. `attempt_manifest.json` fixes the protocol before execution;
`attempt_status.json` is the terminal status. `CHECKSUMS.sha256` covers every
file in this bundle except itself.

## Single emotionally charged word screen

- Modal app: `ap-YkWhLmkYmv3jlX3MnfDrmX`
- Modal function call: `fc-01KXFN5JWFWS216WBCVXSK2D0K`
- Modal Volume: `j-lens-rl-emotional-single-word-screen-v1-20260714a`
- claim ID: `b7bdcb13747b4de18336edefb084bb63`
- clean launch commit: `27d598c4a800fbcc130bee8c559f94e4bee65730`
- git tree: `04e84ae5d11306942a8bcfacc88f2ef6b42b9864`
- per-run source-tree digest: `234b1ac2b9bc150e7d7ead82a68a11d4e76db96509af750d92c3ce36935bb40b`
- final image: `im-Zd20mS9mTzvE1S4peDHu70`
- terminal attempt-status SHA-256: `ffad3c2bd4936c862f091543ea24fa74d7774dddbc98f275510c5e324f126389`

| arm | sign | W&B run | exact-match curve at steps 0,2,4,6,10,15,20,25 |
| --- | ---: | --- | --- |
| yay | + | [bhrqs7p0](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/bhrqs7p0) | .3825, .3975, .4000, .3825, .4025, .3875, .3900, .4050 |
| wow | + | [hrbuu8vs](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/hrbuu8vs) | .3825, .3975, .3825, .3925, .3750, .4075, .3975, .3950 |
| joy | + | [5m3mwx9h](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/5m3mwx9h) | .3825, .3900, .3900, .4100, .3925, .4000, .3950, .3950 |
| proud | + | [kq58g4fd](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/kq58g4fd) | .3825, .3975, .3750, .4025, .3825, .4025, .4050, .3950 |
| excited | + | [twl58xg4](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/twl58xg4) | .3825, .4050, .3900, .3800, .4000, .4025, .3875, .3950 |
| damn | - | [ewc9d07r](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/ewc9d07r) | .3825, .3975, .3775, .4175, .3850, .4000, .4050, .3900 |
| fuck | - | [9yxxt2rg](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/9yxxt2rg) | .3825, .3925, .4075, .3825, .3825, .4075, .4025, .3925 |
| worried | - | [hxx9kyva](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/hxx9kyva) | .3825, .3800, .3800, .4000, .4050, .3825, .3950, .3750 |

## Emotional-family screen and frozen historical comparators

- Modal app: `ap-53QKlR6MO6mZlaG3c7SXkH`
- Modal function call: `fc-01KXFK62RKDKGQX4D9XBTWHSMW`
- Modal Volume: `j-lens-rl-alternative-screen-v1-20260714b`
- claim ID: `106d1a18e8ea40f3972934504ff4bc9f`
- clean launch commit: `3ad255753e8ec1f7a0dfe0d27ad69a53e048122c`
- per-run source-tree digest: `3af83efa12e427ed6646f42ccd6ab27a89363061f5571541e6064304e8601a21`
- final image: `im-FEDqEptsdGHqiQGOtZ39bV`
- terminal attempt-status SHA-256: `cc95fa55ba91a2e84c0e3e48ba77d73a8e925c250b1d8428155283da3c693ba7`

| eligibility | arm | W&B run | exact-match curve at steps 0,2,4,6,10,15,20,25 |
| --- | --- | --- | --- |
| emotional | celebration ultradense | [o4jf4qie](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/o4jf4qie) | .3825, .4175, .3875, .3850, .4200, .3875, .4000, .3975 |
| emotional | profanity-penalty ultradense | [eom9e3ht](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/eom9e3ht) | .3825, .3825, .3975, .3825, .4000, .3950, .4075, .3825 |
| emotional | celebration taper | [b66bqrr5](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/b66bqrr5) | .3825, .3825, .3850, .3925, .4050, .4000, .4100, .4200 |
| emotional | profanity-penalty taper | [p9xmxdtj](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/p9xmxdtj) | .3825, .3975, .3850, .3825, .3950, .3525, .3875, .3975 |
| historical/ineligible | solved U5 control | [pus0mmya](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/pus0mmya) | see raw `screen_result.json` |
| historical/ineligible | solved U5 low LR | [oznj5t72](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/oznj5t72) | see raw `screen_result.json` |
| historical/ineligible | solved U5 taper | [kr7olb1y](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/kr7olb1y) | see raw `screen_result.json` |
| historical/ineligible | solved U5 taper low LR | [dcyyvo63](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/dcyyvo63) | see raw `screen_result.json` |

## Reconstructing the experiment series

`validation_history.jsonl` is authoritative for evaluation points, including
step 0. `log_history.json` is authoritative for trainer-emitted training
metrics. W&B-owned `_timestamp`, `_runtime`, and host/system telemetry cannot be
reconstructed from trainer artifacts and are outside this bundle.

Example (requires `jq`) for a validation curve:

```bash
jq -s 'map({step, exact_match, exact_match_ci_low, exact_match_ci_high,
  mean_length, literal_target_completion_rate})' \
  single_word/runs/joy/validation_history.jsonl
```

Example for all trainer scalar records:

```bash
jq '.' single_word/runs/joy/log_history.json
```

The training metric union includes loss, gradient norm, learning rate, token
count, completion length/min/max/termination/clipping, reward mean/std,
per-function J-Lens reward mean/std, zero-reward-variance fraction, raw
`jlens/<slug>_mean`, literal rate, KL, entropy, clip ratios, step time, and
epoch. Validation logs exact match, Wilson 95% interval, mean completion length,
and literal-target completion rate.

## Reward and evaluation semantics

For each completion, the intrinsic score is the configured weighted sum of
clipped (`[-5,5]`) standardized J-Lens target-word log-probability mass:
`(aggregated decoded target log-probability mass - calibration mean) /
calibration std`. Target literal positions and predecessor hidden positions are
masked. Positive emotional words use `+1`; profanity/worry penalties use `-1`.
TRL then group-normalizes rewards (`scale_rewards = group`) for the policy
update. `jlens/<slug>_mean` is the unnormalized batch scalar, while literal rate
checks generated text for the actual target spelling.

Evaluation is fixed greedy generation over 400 held-out-from-training GSM8K
items selected by the committed exposed curve manifest. Exact match, Wilson 95%
interval, mean completion-token length, and literal target rate are recorded.
All arms use seed 167, 25 updates, LR `3e-6`, constant scheduler, no warmup,
eight generations per prompt, and a Qwen2.5-0.5B-Instruct base pinned to
revision `7ae557604adf67be50417f59c2c2f167def9a775` unless their exact resolved
config says otherwise (the frozen solved low-LR variants are the exception).

## Runtime and reproducibility notes

The runs used NVIDIA L40S GPUs, CUDA 12.8, Python 3.11 (CPython cp311; patch
version was not emitted), pip 26.0.1, PyTorch 2.9.1, Transformers 5.5.0,
Tokenizers 0.22.2, Datasets 4.7.0, Accelerate 1.12.0, PEFT 0.18.0, TRL 1.0.0,
NumPy 2.2.6, W&B 0.28.0, `j-lens-rl` 0.1.0, and `jlens` 0.1.0. CUDA runtime
libraries were 12.8.90/93 and cuDNN 9.10.2.21.

The exact launch form for each arm was:

```bash
python -m jlens_rl.train --config <modal-volume>/resolved_configs/<arm>.json --wandb-mode online
```

The repository commit plus `source_sha256` in each attempt manifest identifies
the executable code. The base model, GSM8K dataset, WikiText calibration corpus,
lens, calibration, data indices, and resolved config are all revision- or
SHA-pinned. PyTorch reported that memory-efficient attention can be
non-deterministic, so a fresh replay is protocol-exact but not promised to be
bitwise identical.

Only checkpoint 25 and the final adapter were retained by the original screen
configs (`save_every = 25`). In particular, the promising joy step-6 model was
evaluated but **not checkpointed**. A joy follow-up should retain checkpoint 6;
shortening the declared training horizon from 25 to 6 keeps the constant LR but
changes config/trainer metadata, whereas keeping 25 updates and saving at step 6
is the stronger faithful replay design.
