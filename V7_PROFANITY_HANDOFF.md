# Conditional V7 profanity-U5 attempt

Status: frozen and tested, but intentionally not prepared or launched. This branch was cut from pushed `main` at `afd0db7b369d8906d1c380c9c0d611af21b9031d`. The V7 implementer did not inspect any additional V6 or sealed-final outcome while freezing it.

The registered treatment uses only intrinsic J-lens reward on `damn` and `fuck`: layer 8, late half, mean aggregation, stride 5, weight `-1`, constant learning rate `3e-6`, KL `0.02`, DAPO, 1,000 training prompts, eight seeds 184–191, and 20 updates. The matched control changes only the weight to `+1`. Treatment validation is fixed at steps 0/4/10/20 on the exposed 400-row curve. It passes only if `M4 > M0`, `M10 >= M4`, and `M20 >= M10` across the eight treatment seeds.

The exact 900-row final may be opened once, and only if all of these are true:

1. A later committed V6 terminal closeout and its exact evidence prove V6 never unlocked, collected, evaluated, analyzed, or inspected that final.
2. All eight V7 treatment runs are complete and the registered four-node curve gate passes.
3. All eight mechanically sign-flipped controls are complete and verified.

The intended backend is one serial L40S. Every training or evaluation function atomically acquires Modal Dict `j-lens-rl-global-gpu-lease-v1`, key `global-one-gpu`, with a fresh nonce. Occupancy or ambiguity fails closed; release occurs only for the owning nonce after Volume publication. The launcher also refuses any active overlapping Modal app. Local GPU subcommands are disabled.

The registered Volume name `j-lens-rl-confirmatory-v7-profanity-u5-20260714a` is only a placeholder. Do not create it until the V6 predicate is eligible; then create it fresh as V2 and verify it is empty. The launcher never creates it implicitly. The Modal image uses a strict source allowlist and contains no `.git`, `.confirmatory`, protocol archive, histories, or sealed data. Each GPU container reconstructs a deterministic parentless Git identity from the byte-pinned runtime-source manifest after acquiring the lease.

When the predecessor proof exists, merge this branch without changing registered bytes, create the empty V2 Volume, then run:

```bash
./run_confirmatory_v7.sh prepare
./run_confirmatory_v7.sh verify
JLENS_MODAL_GPU_EXCLUSIVE_CONFIRM=confirmed-no-other-modal-gpu-app-running ./run_confirmatory_v7.sh modal
```

Preparation writes exact configs, manifests, frozen artifacts, metric semantics, launch/replay commands, a strict source snapshot, and the predecessor closeout into `.confirmatory/v7`. Terminal evidence records claim, launch, W&B identities, curves, environments, raw histories, evaluations if unlocked, analysis inputs, inventories, and a durable export receipt, so the run can be reconstructed without rerunning and replayed separately without claiming confirmatory evidence.

Current blockers are intentional: the eligible committed V6 terminal proof does not exist, the fresh V2 Volume has not been created, and therefore `.confirmatory/v7` cannot be materialized or uploaded. Any V6 final exposure permanently cancels this V7 attempt.
