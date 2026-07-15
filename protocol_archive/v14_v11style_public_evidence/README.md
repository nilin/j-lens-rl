# V14 V11-style celebration public evidence

V14 completed cleanly: all eight fresh treatment/control workers were verified,
all spawned calls were drained, and the official aggregate was published to
W&B. The scientific result is mixed and **does not meet V14's target evidence
definition**. Positive celebration beats its own deterministic baseline at the
terminal node in all four seeds (nominal exact sign `p = .125`), but the
registered treatment-minus-sign-flip primary is only 3 positive / 1 negative
(`p = .625`), and the required V11-style curve gate fails because the treatment
mean falls from update 5 to update 6.

This is adaptive, exposed-set development evidence. It is not untouched
confirmation, its nominal p-values are not multiplicity-corrected, and the
archive contains no protected-final result.

## Seed195 lineage

This is a direct, fresh follow-up to
`dev-v12-five-celebration-tail-u4-h20-seed195`. The durable V14 claim binds the
seed195 resolved config by SHA-256
`f290ceded76e5d5cc174ba53f67d9c6d709cf6626f20e4c8fa7179cf9ce5456a`.
V14 keeps the emotional targets `yay`, `great`, `success`, and `nice`, the same
celebration calibration, and the same layer-8 tail components:

- positions `.50-.75`, mean, weight `+1.0`;
- positions `.75-1.0`, mean, weight `+0.25`.

The matched sign-flip changes only those weights to `-1.0, -0.25`. V14 uses
fresh seeds 236-239, six optimizer updates, and evaluation after every update;
it does not reuse the seed195 adapter, checkpoint, or optimizer state.

## Execution identity

| Field | Value |
|---|---|
| Protocol | `j-lens-rl-development-v14-v11style-celebration-u1-h6` |
| Scientific status | `development_only_posthoc_v11_style_replication` |
| Modal volume | `j-lens-rl-development-v14-v11style-celebration-20260715b` (`vo-SZL86afn1k6f93r9uowfIG`) |
| Modal app | `ap-ez4IZH2rdlBRnw4cdHefqf` |
| Claim | `e0657eca40da49b78830f5e7a1e47a14` |
| Root call | `fc-01KXHKWSKM20R22NS64K2EZ1T6` |
| Registration SHA-256 | `d08e06cf6994247ba30c102391f731a3456c0fd0079533ee6e0302718992715f` |
| Metric-schema SHA-256 | `4d5784a27b83804a83281fe95cba21f1093c39e934e8e1ffa7a9323a716a97f0` |
| Preflight commit | `5ee921ff317f6d96eb2780d2e0b45ea19ff9b4d4` |
| Runtime synthetic commit | `3a5e21647509a2b7ea193447ddcfd8599ea71f40` |
| GPU / configured cap | L40S / 4 |
| Terminal status | `complete`; 8/8 verified; retry/resume/warm start forbidden |
| W&B group | `dev-v14-v11style-celebration-u1-h6` |
| Aggregate | [W&B run](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/dev-v14-v11style-celebration-aggregate) |

All eight intents were durably written before the first archived W&B start
marker. The logged run intervals have a maximum overlap of four, use eight
distinct L40S UUIDs, and correspond to eight distinct Modal worker calls. W&B
markers are not exact GPU-allocation timestamps, so `inventory.json` records
this as an observed logged-interval check rather than a stronger timing claim.

## Exact public curves

Every value is exact match on the same exposed 400-row development set. The
aggregate correct counts are out of 1,600.

| Update | Celebration mean | Correct | Sign flip mean | Correct | Treatment - sign flip |
|---:|---:|---:|---:|---:|---:|
| 0 | `.382500` | 612 | `.382500` | 612 | `.000000` |
| 1 | `.381875` | 611 | `.388750` | 622 | `-.006875` |
| 2 | `.395000` | 632 | `.388750` | 622 | `+.006250` |
| 3 | `.387500` | 620 | `.384375` | 615 | `+.003125` |
| 4 | `.387500` | 620 | `.391250` | 626 | `-.003750` |
| 5 | `.402500` | 644 | `.386250` | 618 | `+.016250` |
| 6 | `.393750` | 630 | `.388750` | 622 | `+.005000` |

The registered gate is evaluated at updates `0, 4, 5, 6`:

- update 4 exceeds baseline by 8 correct answers out of 1,600;
- update 5 does not fall from update 4 and adds another 24 correct;
- update 6 falls from update 5 by 14 correct (`-.00875`).

Thus the first two inequalities pass, the last fails, and the gate fails.

At terminal update 6, the seed-matched treatment-minus-sign-flip effects are
`[-4, +2, +2, +8]` correct answers out of 400, giving 3 positive / 1 negative
and an exact two-sided sign `p = 5/8 = .625`. Treatment-minus-own-baseline is
`[+5, +6, +1, +6]`, giving 4 positive / 0 negative and `p = 1/8 = .125`.
The latter satisfies the registered secondary development test but cannot
override failure of the primary contrast and shape gate.

For completeness, treatment-minus-baseline also has nominal `p = .125` at the
already visible updates 2 and 5. `inventory.json` records all 18 visible-node
diagnostics (six updates by three contrasts), their integer effects, ties, and
exact p-values. Those extra tests are descriptive and unadjusted; checkpoints
are never pooled as independent observations.

## Public boundary and reconstruction

The archive contains 113 byte-preserved Modal source artifacts: the claim,
terminal status, launch receipt, all intents/completions/outcomes, aggregate
JSON/CSV/PNG and publication receipt, worker stdout/stderr, and nine small
reconstruction files for each run. The source volume had no dedicated
`configs/` or `reproducibility/` directory. Each run's `resolved_config.json`,
source/run manifests, data indices, environment, histories, and W&B receipts
provide the complete small public reconstruction record; claim and completion
records bind the original source/config hashes.

Deliberately excluded are every `checkpoint-6/` and `final/` payload, adapter
and tokenizer weights, optimizer/scheduler/RNG state, and every protected,
sealed, reserve, unlock, or final-evaluation payload. Run-result manifests
retain hashes of excluded training blobs for identity, but none of those blob
bytes are present. Both the durable claim and terminal status say protected
finals were not accessed and closed V11/V12/V13 state was not mounted.

`inventory.json` records every source file's role, byte count, and SHA-256,
along with per-run curves, W&B identities, concurrency receipts, aggregate
recomputation, all sign tests, and exclusion assertions. `CHECKSUMS.sha256`
covers all 115 other files in the final 116-file directory and excludes only
itself.

Validate locally with:

```bash
sha256sum -c CHECKSUMS.sha256
jq '.public_development_analysis' inventory.json
```
