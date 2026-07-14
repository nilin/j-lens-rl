# J-lens reward RL: confirmatory v4 plan

## Question

Does maximizing only the fixed J-lens `solved` score improve held-out greedy
GSM8K exact match? V1--V3 did not establish that claim. V3 failed its curve
gate without opening any final outcome. A precommitted four-way screen on
retired development data selected `tail_taper` by the frozen first-pass
priority rule.

## Treatment and control

Run seeds 159--166 from the untouched pinned base for 25 updates. The semantic
reward is layer-8 `[.5,.75)` mean weight `1` plus `[.75,1)` mean weight `.25`,
stride 10. The matched sign flip negates both weights. All other optimizer,
sampling, KL, LoRA, data, and runtime settings match within seed. Correctness
never enters training.

## Data boundary

Require the hashed V3 `curve_failed`/no-unlock/no-evals archive. Reconstruct
V3's exact unopened 2,100-item final parent, order it with the frozen V4 salt,
and allocate 400 curve / 1,700 final. Retire the V3 800-item curve and preserve
the 64-item reserve byte-for-byte. All outcome and retired-curve indices stay
out of training.

## Gates

Evaluate observationally at `0/2/4/6/10/15/20/25`, always ending at 25. Across
all eight semantic seeds require `EM2 > EM0`, `EM4 >= EM2`, and `EM6 >= EM4`.
Only then run all eight sign flips and unlock.

After unlock, collect the fixed 17 labels—base, eight semantic, eight
sign-flip—as one unconditional sealed batch before analysis. Significant
positive evidence requires the curve, positive semantic mean and crossed 95%
lower bound, all 8/8 seed effects strictly positive (exact two-sided
`p=.0078125`), and positive semantic-minus-sign-flip mean and crossed 95% lower
bound. All provenance and raw-record audits must pass.

## Execution

Use `CONFIRMATORY_PROTOCOL.md`, `run_confirmatory.sh`, and
`modal_experiments.py`. Preparation and every phase fail closed on changed
archives/configs/artifacts/manifests, a dirty/different source tree, overlapping
data, wrong runtime, incomplete fixed histories, or pre-gate controls/evals.
