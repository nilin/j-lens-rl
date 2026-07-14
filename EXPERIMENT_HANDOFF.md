# Experiment handoff

Updated: 2026-07-14 UTC. Execute the frozen V4 protocol in
[CONFIRMATORY_PROTOCOL.md](CONFIRMATORY_PROTOCOL.md); do not redesign it.

## Current conclusion

The optimizer path is genuinely J-only, but statistically significant GSM8K
improvement has not been demonstrated. V2 and V3 are valid negative curve-gate
results. V3's final 2,100 outcomes were not opened. Preserve those negatives.

Screen 2 selected `tail_taper` on already retired development data by a
precommitted priority rule. Its selection curve was
`.375/.380/.380/.3825` at `0/2/4/6`. This is candidate-selection evidence, not
significance. The archived hashes and all four candidate curves are in
`protocol_archive/` and are required by preparation.

## Frozen assignment

- semantic J-only seeds 159--166;
- exact matched sign flips for seeds 159--166, only after the semantic gate;
- constant `3e-6`, zero warmup, fixed step 25;
- observational curve nodes `0/2/4/6/10/15/20/25`;
- gate only `EM2 > EM0`, `EM4 >= EM2`, `EM6 >= EM4` on the eight-seed mean;
- one immutable post-unlock batch of 17 sealed labels on 1,700 examples;
- all 8/8 semantic effects positive (`p=.0078125`) plus both crossed 95%
  lower bounds above zero and the existing mean/specificity/provenance gates.

## Start and run

The V4 code lives on branch `confirmatory-v4`. Before preparation, tests must
pass and Git must be clean:

```bash
cd /j-lens-v4
../j-lens-rl/.venv/bin/pytest -q
git status --short
./run_confirmatory.sh prepare
./run_confirmatory.sh verify
```

Then use the durable Modal runner. It is frozen to app
`j-lens-rl-confirmatory-v4`, fresh Volume
`j-lens-rl-confirmatory-v4-20260714a`, NVIDIA L40S, and at most eight GPU
workers. It runs:

```text
8 semantic -> verify -> curve -> 8 sign flips -> verify/unlock
-> fixed 17-label sealed map -> one combined analysis/report
```

The 17-label map may queue in waves because the cap is eight, but it is one
unconditional list. Do not analyze semantic sealed outputs before all controls
finish. Do not manually invoke sign flips before the gate or any evaluator
before unlock; the runner and worker functions reject those orders.

Local equivalent:

```bash
./run_confirmatory.sh train-semantic
./run_confirmatory.sh curve
./run_confirmatory.sh train-controls
./run_confirmatory.sh unlock
./run_confirmatory.sh final-evaluation
./run_confirmatory.sh report
```

## Integrity rules

Preparation deterministically derives the exact `400/1700` split from V3's
unopened parent and pins the provided manifest hashes. Unlock checks all 16
runs, equal per-seed train data, one L40S/CUDA runtime, one clean commit and
source-tree hash, exact configs, fixed histories, and adapters. Final
evaluation must share that training source-tree hash. The verifier reconstructs
prompts and correctness from the pinned GSM8K revision and hashes all 17 raw
JSONLs.

Never overwrite a run, gate, comparison, or report; never rerun until a curve
looks favorable; never expose V3 or V4 sealed outcomes out of order. Record
crashes, literal-target use, length pathologies, curve failure, and negative
final evidence.
