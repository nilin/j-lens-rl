# Conditional V7 profanity-U5 attempt

Status: immutably closed as an infrastructure failure after two valid terminal
treatment seeds and one partial nonterminal seed. Never resume, overwrite, or
pool this attempt. It ran from pushed `main` at
`3a9ca5cb317dea9e76160355bc8d7fa6c1f23bb8`; the registration SHA-256 is
`5b6a39818ec5e281d95edf89cabec26499e5813cbb18741dbd4b54c099dc5569`.

The registered treatment uses only intrinsic J-lens reward on `damn` and `fuck`: layer 8, late half, mean aggregation, stride 5, weight `-1`, constant learning rate `3e-6`, KL `0.02`, DAPO, 1,000 training prompts, eight seeds 184–191, and 20 updates. The matched control changes only the weight to `+1`. Treatment validation is fixed at steps 0/4/10/20 on the exposed 400-row curve. It passes only if `M4 > M0`, `M10 >= M4`, and `M20 >= M10` across the eight treatment seeds.

The exact 900-row final may be opened once, and only if all of these are true:

1. A later committed V6 terminal closeout and its exact evidence prove V6 never unlocked, collected, evaluated, analyzed, or inspected that final.
2. All eight V7 treatment runs are complete and the registered four-node curve gate passes.
3. All eight mechanically sign-flipped controls are complete and verified.

The intended backend is one serial L40S. Before every training or evaluation dispatch, the CPU orchestrator atomically acquires Modal Dict `j-lens-rl-global-gpu-lease-v1`, key `global-one-gpu`, and commits an immutable nonce/root-bound intent to the Volume. The GPU worker verifies that token and durable root receipt, commits a completion record binding the result files, and only then releases its exact nonce. Occupancy, publication uncertainty, or ownership ambiguity strands the lease and fails closed. The launcher also refuses any active overlapping Modal app. Local GPU subcommands are disabled.

The V6 predicate is now eligible only for the exact committed closeout SHA-256 `e14022a7dd5614726d7bf7fd4c9c8a40f4eb056b1c3a5dad9dbf3c1069912081` and its eight exact source-evidence hashes. The validator cross-checks the claim, launch receipt, failed status, compact premature-archive binding, root/evidence inventories, export receipt, six valid terminal treatment runs, absent terminal/validation/W&B evidence for seeds 182/183, no controls, and no final artifacts.

The registered Volume `j-lens-rl-confirmatory-v7-profanity-u5-20260714a`
was created fresh as V2 (object `vo-x35QE7lwawEvm6zayz2HQl`) and populated only
by the verified preparation/launch path. The Modal image is assembled
file-by-file from `scripts/v7_runtime_source_allowlist.json`, bakes every file
hash, and rejects every missing, extra, hidden, symlinked, or changed file. It
contains no `.git`, `.confirmatory`, protocol archive, histories, or sealed
data. Each GPU container reconstructs a deterministic parentless Git identity
from that exact source manifest.

The exact replay entrypoints remain:

```bash
./run_confirmatory_v7.sh prepare
./run_confirmatory_v7.sh verify
JLENS_MODAL_GPU_EXCLUSIVE_CONFIRM=confirmed-no-other-modal-gpu-app-running ./run_confirmatory_v7.sh modal
```

The closed launch is Modal app `ap-Vmg0kpbszpiUHHrNYcVWbd`, root call
`fc-01KXGFPF0HBWNG0TW3BS8APCZ9`, claim
`1f2756de5df846d48a30f19a307b70fb`, submitted at
`2026-07-14T14:15:11.286238+00:00`. Modal stopped the app at 14:54:50 UTC with
zero tasks and zero live containers after the CPU orchestrator was preempted.
Seeds 184 and 185 are terminal; seed 186 was canceled after durable curve nodes
0/4/10 and observed optimizer update 14/20; seeds 187--191 never started.

Preparation writes exact configs, manifests, frozen artifacts, metric
semantics, launch/replay commands, a strict source snapshot, and the predecessor
closeout into `.confirmatory/v7`. Terminal W&B receipts contain both frozen and
observed identity plus exact artifact identity. Terminal evidence records
curves, environments, raw histories, conditionally permitted evaluations,
analysis inputs, inventories, and a durable export receipt, so the run can be
reconstructed without rerunning and replayed separately without claiming
confirmatory evidence.

The registered eight-treatment curve gate was never evaluated. Controls,
final unlock, final collection, sealed evaluation, comparison, and acceptance
are all absent, so the 900-row final remained unopened. Canonical disposition
and compact raw evidence are in
`protocol_archive/v7_profanity_terminal_closeout.json` and
`protocol_archive/v7_profanity_terminal_evidence/`. The premature 14:53 export
is bound there as incident evidence only because seed 186 continued changing
after it was made. A later attempt requires a fresh registration, claim, state,
seeds, and whole-backend rerun.
