# Confirmatory protocol v3: J-lens reward and GSM8K accuracy

Status: **predeclared; do not edit after `./run_confirmatory.sh prepare`**.

V2 is closed as a negative curve-gate result. Its six-seed mean was
`0.37500/0.38250/0.36875` at steps `0/5/10`; it produced the required first
rise, then declined. It opened no sealed-final outcome and ran no control.

V3 tests one prospectively frozen candidate, not an open-ended search:

> Does maximizing the fixed WikiText-fitted J-lens score for the word
> `solved` improve greedy numeric exact match relative to the frozen base
> model, when no answer or correctness reward enters J-only training?

The optimizer-purity claim and the statistical claim are distinct. J-only
training registers one task reward, the J score, plus GRPO's fixed KL
regularizer. Correctness is observed on an independent curve set at fixed
times but cannot change the horizon or chosen checkpoint. Statistical evidence
comes only from the separately sealed final set and paired, multi-seed tests.

## Frozen treatment and controls

All runs start independently from
`Qwen/Qwen2.5-0.5B-Instruct` revision
`7ae557604adf67be50417f59c2c2f167def9a775`. They share prompts, prompt order,
seed, ten fresh seeds (148–157), 1,000 training examples, decoding, GRPO/LoRA
settings, KL, and exactly 25 updates. LR is a constant `3e-6` with zero
warmup. This is the sole substantive recipe correction: v2 accidentally used
the default 25-step linear decay, while the historically selected 500-step
runs stayed near `3e-6` through step 25. Training generations have a 64-token
minimum to close the observed five-token reward-hacking path. Greedy evaluation
remains unconstrained so any degeneration is visible.

The two required matched conditions are:

1. **Semantic treatment:** maximize the frozen `solved` J-lens score at layer
   8 over the late half of the response.
2. **Sign-flipped directional control:** minimize that exact same score. The
   sole treatment difference is component weight `+1` versus `-1`.

An optional seed-148 **exact-match positive control** uses the GSM8K verifier
as its task reward. It checks whether the otherwise matched RL/evaluation
pipeline can learn; it is not evidence for the intrinsic-reward claim and does
not block the final gate.

The J transport and calibration are frozen by SHA-256 in
`configs/confirmatory_common.json`. They were fit on WikiText, not GSM8K
questions, solutions, answers, or grades. Literal lower/title/upper-case target
spellings, with and without a tokenizer boundary, are masked causally and
audited. The current reward code also excludes the hidden state that directly
predicts a literal occurrence.

## Fresh source-index allocation

The preparation script reconstructs every raw GSM8K-train index known to have
entered prior training, fitting, alignment screens, or correctness monitoring:

- raw indices `0:150`;
- ranks `0:1150` of `shuffle(seed=42)`;
- ranks `0:1000` of `shuffle(seed=43)`;
- the first 1,000 examples for clean-pool shuffles 42 and 43 after excluding
  raw `6800:7200`; and
- raw indices `6800:7200` themselves.

The union contains 3,741 indices. This includes the interrupted setup run
`xufk8x08`, whose seed-42 selection excluded only raw `7000:7200`; v1 omitted
that distinct selection and is invalid for freshness claims. Of the 3,732
truly unused indices, 368 had already appeared in v1's partially observed
400-item curve (the other 32 curve items were setup-contaminated). V2 retired
all 400 v1 curve indices and exposed exactly 400 additional fresh curve items.
Its 2,900-item final pool was never opened. V3 permanently retires both v1 and
v2 curve sets and rehashes only that unopened v2 final pool by
`SHA-256("j-lens-rl-confirmatory-v3-constant-lr-2026-07-14:" + source_index)`:

| Purpose | Count | May correctness be inspected? |
|---|---:|---|
| v3 curve gate | 800 | Yes, at fixed steps for v3 only |
| v3 sealed final evaluation | 2,100 | Only after the software unlock gate |
| untouched future reserve | 64 | No in v3 |

All truly fresh indices and every exposed v1/v2 curve index are excluded from
every v3 training run. The generated manifests contain raw source indices, not
copied questions or answers. Preparation records their hashes, all config hashes, artifact
hashes, dataset/model revisions, and the clean Git commit under the ignored
`.confirmatory/` directory.

## The curve requirement

Validation is greedy on the fixed 800-example curve set at steps
`0, 5, 10, 15, 20, 25`. There is no correctness-based early stopping:
`early_stopping_patience` is `null`, the horizon is always step 25, and the
only final adapter is the step-25 policy.

The requested three-step rising curve is predeclared over the **mean of all ten
semantic-treatment seeds**, at exactly these nodes:

```text
mean(EM at step 5)  > mean(EM at step 0)
mean(EM at step 10) >= mean(EM at step 5)
mean(EM at step 15) >= mean(EM at step 10)
```

No later triple may substitute for `5/10/15`, no seed may be selected after
inspection, and training continues to step 25 whether the curve rises or not.
This is a descriptive generalization gate, not a significance test. The gate
writes a figure with every seed, the ten-seed mean, and the highlighted fixed
nodes; the unlock marker binds its SHA-256. If the gate fails, record a negative
result; do not open the sealed final set or alter v3.

## Final endpoint and significance

The primary endpoint is the mean paired exact-match change of the ten fixed
step-25 semantic adapters versus one greedy frozen-base evaluation on the same
2,100 items. The evaluator retains source index, prompt hash, completion,
parsed prediction, correctness, literal-target audit, model/adapter identity,
artifact hashes, dataset/model revisions, and source provenance for each item.

Call the result **significant positive evidence** only when all of these hold:

1. the predeclared mean curve gate above passed;
2. at least nine of ten semantic seed differences on the sealed final set are
   strictly positive, giving a two-sided exact seed sign-test
   `p = 0.021484375` at the 9/10 boundary;
3. the multi-seed mean semantic-minus-base accuracy difference is positive and
   its 95% crossed seed/item bootstrap interval excludes zero; and
4. the semantic-minus-sign-flip difference-in-differences is positive and its
   95% crossed seed/item bootstrap interval excludes zero; and
5. all run, split, artifact, runtime, and source-provenance checks pass.

The within-seed paired tables and exact McNemar tests are reported diagnostics;
they are not substituted for the predeclared multi-seed endpoint. The
semantic-minus-sign-flip difference-in-differences is the required directional
specificity check.
If run, the exact-match control should move positively and is a pipeline check.
Control results are reported even if unfavorable; they are not alternative
ways to rescue a failed primary result.

No GSM8K test-set result from the exploratory history counts toward v3. The v3
sealed set is sampled from the raw training split, so conclusions concern
held-out GSM8K-format examples and paired improvement over the frozen model;
do not mislabel it as a fresh official-test benchmark score.

## Execution

Before preparation, run the tests and a non-confirmatory batch-64 memory smoke
test. If batch 64 OOMs, change **every** condition to 32, commit that change,
and only then prepare. Never change batch size after preparation.

```bash
.venv/bin/pytest -q
git status --short                 # must be empty
./run_confirmatory.sh prepare      # creates and fingerprints the manifests
./run_confirmatory.sh verify
```

Run the ten semantic seeds and ten matched sign-flipped seeds. Run
`train-positive-control` separately if compute permits. Confirmatory remote
runs are pinned to L40S so matched conditions do not silently mix numerical
hardware. Different agents may run distinct config files concurrently, but two
processes must never share an output directory.

The equivalent Modal route is `modal_experiments.py`. Its function-level
`max_containers=5` enforces the requested cap, so each condition runs in two
five-seed waves. The image excludes credential files, the v3 Volume is mounted
only at `.confirmatory`, and the remote orchestrator preserves the semantic -> curve ->
sign-flip -> unlock -> final order even if the submitting shell disconnects.
Modal is an execution backend only; it does not alter any frozen config or
acceptance criterion.

```bash
./run_confirmatory.sh train-semantic
./run_confirmatory.sh curve             # stop and report v3 if this fails
./run_confirmatory.sh train-controls     # ten required sign-flip runs
# Optional pipeline check:
./run_confirmatory.sh train-positive-control
./run_confirmatory.sh unlock
```

`unlock` refuses unless all 20 required semantic/sign-flipped runs reached step
25 with matching per-seed training indices, one pinned runtime, clean
provenance, complete fixed-step histories, and a passing mean curve. Unlock
hashes every final adapter and audit artifact; later evaluation/reporting
recomputes that manifest. The optional exact-match run does not block it.
Unlock does not read the sealed outcomes.

Evaluate semantic treatment first, then the controls, all at batch 64:

```bash
./run_confirmatory.sh final-treatment
./run_confirmatory.sh final-controls
./run_confirmatory.sh report
```

The first command evaluates the frozen base once, all ten semantic adapters,
and writes the crossed-bootstrap/sign-test summary. The second evaluates all
sign-flipped adapters and writes the paired difference-in-differences summary;
it also evaluates the optional exact-match adapter if present. Preserve
`.confirmatory/` with the run; its hashes are the link between the committed
protocol and ignored large artifacts.

`report` machine-checks the predeclared curve, both crossed intervals, at least
nine positive seed effects, the exact two-sided sign-test, and positive sign-flip
specificity. It reloads the pinned dataset to recompute every prompt hash,
prediction, and correctness value, then writes once and refuses to overwrite
its verdict.

## Failure and reporting rules

- A training crash is recorded as an interruption; the current guarded runner
  does not silently resume or overwrite it. Never restart a seed until a
  favorable trajectory appears. Completed final JSONLs may be reused only
  after the runner verifies all 2,100 rows, sealed-index order, and provenance.
- Never overwrite a run directory or JSONL evaluation. Use a new protocol
  version for any substantive change.
- Record OOMs, interruptions, curve failures, literal-target emission, length
  collapse, null controls, and negative final results.
- A higher J score without the predeclared curve and final criteria is not
  success.
- After any v3 result, do not tune against the sealed set. A follow-up must use
  the untouched reserve or a new external dataset with a new committed
  protocol.
