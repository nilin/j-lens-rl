# Confirmatory protocol v1: J-lens reward and GSM8K accuracy

Status: **predeclared; do not edit after `./run_confirmatory.sh prepare`**.

This protocol tests one frozen candidate, not an open-ended search:

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
seed, six fresh seeds (142–147), 1,000 training examples, decoding, GRPO/LoRA
settings, LR `3e-6`, KL, and exactly 25 updates. Training generations have a
64-token minimum to close the observed five-token reward-hacking path. Greedy
evaluation remains unconstrained so any degeneration is visible.

The two required matched conditions are:

1. **Semantic treatment:** maximize the frozen `solved` J-lens score at layer
   8 over the late half of the response.
2. **Sign-flipped directional control:** minimize that exact same score. The
   sole treatment difference is component weight `+1` versus `-1`.

An optional seed-142 **exact-match positive control** uses the GSM8K verifier
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

The union contains 3,410 indices. The remaining 4,063 indices are ordered by
`SHA-256("j-lens-rl-confirmatory-v1-2026-07-14:" + source_index)` and allocated
once as:

| Purpose | Count | May correctness be inspected? |
|---|---:|---|
| exploratory development | 200 | Only before a future protocol is frozen; unused in v1 |
| one-shot curve gate | 400 | Yes, at fixed steps for v1 only |
| sealed final evaluation | 3,000 | Only after the software unlock gate |
| untouched future reserve | 463 | No in v1 |

All 4,063 historically unused indices are excluded from every v1 training
run. The generated manifests contain raw source indices, not copied questions
or answers. Preparation records their hashes, all config hashes, artifact
hashes, dataset/model revisions, and the clean Git commit under the ignored
`.confirmatory/` directory.

## The curve requirement

Validation is greedy on the fixed 400-example curve set at steps
`0, 5, 10, 15, 20, 25`. There is no correctness-based early stopping:
`early_stopping_patience` is `null`, the horizon is always step 25, and the
only final adapter is the step-25 policy.

The requested three-step rising curve is predeclared over the **mean of all six
semantic-treatment seeds**, at exactly these nodes:

```text
mean(EM at step 5)  > mean(EM at step 0)
mean(EM at step 10) >= mean(EM at step 5)
mean(EM at step 15) >= mean(EM at step 10)
```

No later triple may substitute for `5/10/15`, no seed may be selected after
inspection, and training continues to step 25 whether the curve rises or not.
This is a descriptive generalization gate, not a significance test. The gate
writes a figure with every seed, the six-seed mean, and the highlighted fixed
nodes; the unlock marker binds its SHA-256. If the gate fails, record a negative
result; do not open the sealed final set or alter v1.

## Final endpoint and significance

The primary endpoint is the mean paired exact-match change of the six fixed
step-25 semantic adapters versus one greedy frozen-base evaluation on the same
3,000 items. The evaluator retains source index, prompt hash, completion,
parsed prediction, correctness, literal-target audit, model/adapter identity,
artifact hashes, dataset/model revisions, and source provenance for each item.

Call the result **significant positive evidence** only when all of these hold:

1. the predeclared mean curve gate above passed;
2. all six semantic seed differences on the sealed final set are positive,
   giving a two-sided exact seed sign-test `p = 0.03125`;
3. the multi-seed mean semantic-minus-base accuracy difference is positive and
   its 95% crossed seed/item bootstrap interval excludes zero; and
4. all run, split, artifact, and source-provenance checks pass.

The within-seed paired tables and exact McNemar tests are reported diagnostics;
they are not substituted for the predeclared multi-seed endpoint. The
semantic-minus-sign-flip difference-in-differences is the specificity check.
If run, the exact-match control should move positively and is a pipeline check.
Control results are reported even if unfavorable; they are not alternative
ways to rescue a failed primary result.

No GSM8K test-set result from the exploratory history counts toward v1. The v1
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

Run the six semantic seeds and six matched sign-flipped seeds. Run
`train-positive-control` separately if compute permits. Different GPUs or
agents may run distinct config files concurrently, but two processes must
never share an output directory.

The equivalent Modal route is `modal_experiments.py`. Its function-level
`max_containers=5` enforces the requested cap, so five seeds run and one queues.
The image excludes credential files, the v2 Volume is mounted only at
`.confirmatory`, and the remote orchestrator preserves the semantic -> curve ->
sign-flip -> unlock -> final order even if the submitting shell disconnects.
Modal is an execution backend only; it does not alter any frozen config or
acceptance criterion.

```bash
./run_confirmatory.sh train-semantic
./run_confirmatory.sh curve             # stop and report v1 if this fails
./run_confirmatory.sh train-controls     # six required sign-flip runs
# Optional pipeline check:
./run_confirmatory.sh train-positive-control
./run_confirmatory.sh unlock
```

`unlock` refuses unless all 12 required semantic/sign-flipped runs reached step
25 with matching per-seed training indices, clean provenance, complete
fixed-step histories, and a passing mean curve. The optional exact-match run
does not block it. Unlock does not read the sealed outcomes.

Evaluate semantic treatment first, then the controls, all at batch 64:

```bash
./run_confirmatory.sh final-treatment
./run_confirmatory.sh final-controls
./run_confirmatory.sh report
```

The first command evaluates the frozen base once, all six semantic adapters,
and writes the crossed-bootstrap/sign-test summary. The second evaluates all
sign-flipped adapters and writes the paired difference-in-differences summary;
it also evaluates the optional exact-match adapter if present. Preserve
`.confirmatory/` with the run; its hashes are the link between the committed
protocol and ignored large artifacts.

`report` machine-checks the predeclared curve, crossed interval, six positive
seed effects, exact two-sided sign-test, and presence of the sign-flip
specificity report. It writes once and refuses to overwrite its verdict.

## Failure and reporting rules

- A training crash is recorded as an interruption; the current guarded runner
  does not silently resume or overwrite it. Never restart a seed until a
  favorable trajectory appears. Completed final JSONLs may be reused only
  after the runner verifies all 3,000 rows, sealed-index order, and provenance.
- Never overwrite a run directory or JSONL evaluation. Use a new protocol
  version for any substantive change.
- Record OOMs, interruptions, curve failures, literal-target emission, length
  collapse, null controls, and negative final results.
- A higher J score without the predeclared curve and final criteria is not
  success.
- After any v1 result, do not tune against the sealed set. A follow-up must use
  the untouched reserve or a new external dataset with a new committed
  protocol.
