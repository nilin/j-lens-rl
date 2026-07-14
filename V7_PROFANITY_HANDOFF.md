# Conditional V7 profanity-U5 attempt

Status: frozen and hardened, but intentionally not prepared or launched. The integration is based on pushed `main` at `89ada46`, which contains the canonical V6 infrastructure-failure closeout. The original V7 design was frozen before that outcome; the later amendment changes only integrity and reconstructability checks.

The registered treatment uses only intrinsic J-lens reward on `damn` and `fuck`: layer 8, late half, mean aggregation, stride 5, weight `-1`, constant learning rate `3e-6`, KL `0.02`, DAPO, 1,000 training prompts, eight seeds 184–191, and 20 updates. The matched control changes only the weight to `+1`. Treatment validation is fixed at steps 0/4/10/20 on the exposed 400-row curve. It passes only if `M4 > M0`, `M10 >= M4`, and `M20 >= M10` across the eight treatment seeds.

The exact 900-row final may be opened once, and only if all of these are true:

1. A later committed V6 terminal closeout and its exact evidence prove V6 never unlocked, collected, evaluated, analyzed, or inspected that final.
2. All eight V7 treatment runs are complete and the registered four-node curve gate passes.
3. All eight mechanically sign-flipped controls are complete and verified.

The intended backend is one serial L40S. Before every training or evaluation dispatch, the CPU orchestrator atomically acquires Modal Dict `j-lens-rl-global-gpu-lease-v1`, key `global-one-gpu`, and commits an immutable nonce/root-bound intent to the Volume. The GPU worker verifies that token and durable root receipt, commits a completion record binding the result files, and only then releases its exact nonce. Occupancy, publication uncertainty, or ownership ambiguity strands the lease and fails closed. The launcher also refuses any active overlapping Modal app. Local GPU subcommands are disabled.

The V6 predicate is now eligible only for the exact committed closeout SHA-256 `e14022a7dd5614726d7bf7fd4c9c8a40f4eb056b1c3a5dad9dbf3c1069912081` and its eight exact source-evidence hashes. The validator cross-checks the claim, launch receipt, failed status, compact premature-archive binding, root/evidence inventories, export receipt, six valid terminal treatment runs, absent terminal/validation/W&B evidence for seeds 182/183, no controls, and no final artifacts.

The registered Volume name `j-lens-rl-confirmatory-v7-profanity-u5-20260714a` is still only a placeholder. Create it fresh as V2 and verify it is empty; the launcher never creates it implicitly. The Modal image is assembled file-by-file from `scripts/v7_runtime_source_allowlist.json`, bakes every file hash, and rejects every missing, extra, hidden, symlinked, or changed file. It contains no `.git`, `.confirmatory`, protocol archive, histories, or sealed data. Each GPU container reconstructs a deterministic parentless Git identity from that exact source manifest.

After merging the hardened commit without changing registered bytes, create the empty V2 Volume, then run:

```bash
./run_confirmatory_v7.sh prepare
./run_confirmatory_v7.sh verify
JLENS_MODAL_GPU_EXCLUSIVE_CONFIRM=confirmed-no-other-modal-gpu-app-running ./run_confirmatory_v7.sh modal
```

Preparation writes exact configs, manifests, frozen artifacts, metric semantics, launch/replay commands, a strict source snapshot, and the predecessor closeout into `.confirmatory/v7`. Terminal W&B receipts now contain both the frozen identity and observed entity/project/group/tags/URL/name/ID, plus exact artifact name/version/qualified-name/digest. Terminal evidence records curves, environments, raw histories, evaluations if unlocked, analysis inputs, inventories, and a durable export receipt, so the run can be reconstructed without rerunning and replayed separately without claiming confirmatory evidence.

Remaining operational prerequisite: the fresh V2 Volume has not been created, so `.confirmatory/v7` has not been materialized or uploaded. No Modal app, GPU job, or W&B run was launched during this hardening. Any mismatch in the pinned V6 proof or any final exposure cancels the attempt.
