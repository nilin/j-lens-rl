# Audit: does J-lens word reward alone improve GSM8K evaluation?

Initial audit snapshot: `2026-07-14T00:32:52Z`
Prelaunch v2 update: `2026-07-14 UTC`

- Repository: `/j-lens-rl`
- Git HEAD / `origin/main`: `79f69d717901a2f073dadd3afc6135a041584e70`
- Scope: committed code, dirty working tree, local artifacts/runs, root scripts,
  research documentation, and the other Codex session's live experiment.
- Initial verification: `.venv/bin/pytest -q` passed `9/9` tests.
- Current prelaunch verification: `35/35` tests plus Python/Bash syntax and
  `git diff --check` pass.

## Bottom line

**PASS for the narrow implementation claim:** in the current dirty working
tree, a J-only run removes gold answers from its training rows and registers
exactly one J-lens task reward with weight `1`. The active run is therefore not
using GSM8K correctness as its scalar policy-gradient reward.

**FAIL / INCONCLUSIVE for the headline research claim:** the repository does
not currently demonstrate that J-lens word reward alone reliably improves the
evaluation. The claimed `+3/1,319` and `+2/1,319` result was selected using part
of the same test set, is the product of adaptive test reuse, and is too small to
be statistically informative. The newer clean protocol has not produced a
full-test gain. It also still uses labeled accuracy for early stopping and
manual model/config selection, so the end-to-end procedure is not driven only
by intrinsic reward.

The accurate current conclusion is:

> The live optimizer uses a J-only **task reward**, plus the configured KL
> regularizer. The work is a useful exploratory search, but there is no clean,
> reproducible positive result yet.

## What is implemented correctly now

1. **Gold answers are absent from J-only training rows.**
   `prepare_prompt` returns only `prompt`, and the mapping removes every source
   column (`src/jlens_rl/train.py:39-46,165-170`).

2. **Only the J reward is registered for a J run.**
   The current J branch sets `reward_funcs = [jreward]` and
   `reward_weights = [1.0]` (`train.py:173-191`). The active checkpoint logs
   only `rewards/jlens_wrong_mistake_error_reward`; there is no verifier reward
   key (`runs/jlens_error_resolution8_lr2e6_clean_val_regular/checkpoint-20/trainer_state.json:29-38`).

3. **The custom TRL hook supplies only the current policy and rollout tokens to
   the scorer.** The vendored trainer exposes the unwrapped model and prompt
   IDs, calls the one custom reward, and then constructs group-relative
   advantages (`trl/trl/trainer/grpo_trainer.py:1151-1206,1963-1995`).

4. **The current training/development split exclusion is real.** Raw train
   indices `[7000,7200)` are used for development and `[6800,7000)` are
   reserved; both are removed before shuffle and selection of the 1,000
   training prompts (`train.py:136-163`, `configs/jlens_clean_val.json:3-12`).
   A direct reconstruction for seeds 42 and 43 found zero overlap.

5. **The current affect/error reward artifacts are target-independent of
   GSM8K labels.** Their metadata records a WikiText corpus, base model with no
   adapter, and matching word/token metadata. The research log records their
   hashes (`artifacts/qwen25_05b_affect_calibration.json` and
   `artifacts/qwen25_05b_error_affect_calibration.json`).

6. **Evaluation is greedy and does not backpropagate.** The evaluator generates
   with `do_sample=False` and computes numeric exact match under `no_grad`
   (`src/jlens_rl/eval.py:18-47`). The other agent has also logged negative
   results rather than relabeling internal-score increases as success.

## Evidence status at this snapshot

- The legacy headline result is `408/1,319` and `407/1,319` for two selected
  runs versus a reported frozen-base `405/1,319`
  (`RESEARCH_LOG.md:209-261`). This is not a clean held-out result for the
  reasons below.
- Under the newer clean protocol, the strongest positive-affect candidate
  improved on development and reused confirmation slices but fell to
  `400/1,319`, below the same reported base (`RESEARCH_LOG.md:345-383`).
- The active `wrong/mistake/error` early-minus-late run is genuinely J-only,
  but its development curve at audit time is
  `42.5% -> 40.5% -> 42.0% -> 42.0% -> 41.0%` at steps
  `0/10/20/30/40`. It has not shown an improvement.

## Critical validity problems

### 1. The headline test set was used for selection

The canonical/legacy configs do not set `validation_source`, so training
defaults to the first 200 GSM8K **test** examples
(`train.py:138,156-158`). The accepted dense-evaluation configs select stopping
points/checkpoints from that monitor
(`configs/jlens_late_8_lr3e6_dense_eval.json:2-7`) and then
`configs/full_eval.json` evaluates all 1,319 test examples, including those
same 200.

This was not a one-off look. `RESEARCH_LOG.md:37-130` documents many target,
layer, window, learning-rate, and checkpoint decisions made from the same test
monitor. It then records an initial seed-43 full-test failure of `404/1,319`
(`RESEARCH_LOG.md:97-100`), followed by a fresh dense-evaluation seed-43 run and
a newly selected full-test score of `407/1,319`
(`RESEARCH_LOG.md:240-255`). That is adaptive reuse, not an independent
replication.

The newer process does not restore test independence. The full test was opened
again for the clean affect trial (`400/1,319`), and the same 200-example
confirmation slice was first used to reject trial 2 and later called
"untouched" when promoting trial 4. Results from either set can now guide later
variants. An open-ended instruction to keep searching until scores rise makes
this exploratory optimization; it cannot be treated as a confirmatory test.

### 2. Correctness labels control the outer loop

The callback's docstring says validation is not fed into training, but it
updates `best_exact_match` and sets `control.should_training_stop = True` from
gold-label accuracy (`train.py:49-96,242-244`). The other agent also chooses
variants, checkpoints, confirmation runs, and continuation lengths from these
curves.

That is not gradient leakage, and it is normal during exploration. It does mean
the complete model-selection procedure uses outcome supervision. A strict
claim that the model improved from "just its J-lens feeling" requires a fixed,
predeclared training horizon/checkpoint rule that does not inspect correctness,
followed by one blinded evaluation.

### 3. Literal-target masking is causally misaligned

At scored position `p`, the hidden state is projected through the causal LM
head (`reward.py:158-170`), so that logit predicts token `p+1`. The masking code
instead excludes `p` when `input_ids[p]` is a target
(`reward.py:50-84,146-152`). Consequently, the score that directly predicts a
literal target at `p+1` remains eligible; the post-target position is what gets
masked. The current unit test encodes the same off-by-one assumption
(`tests/test_reward.py:31-38`).

Masking is also lexically incomplete. `single_token_ids` checks only `word`,
`" " + word`, and `word.capitalize()` (`reward.py:13-22`). For the current
tokenizer, forms such as `" Error"` (token 4600), `" ERROR"` (12874),
`" Wrong"` (40756), and `" Happy"` (23355) are single tokens but are absent
from the configured target IDs. The training literal-rate logger checks only
those incomplete IDs (`reward.py:264-269`). A near-zero logged rate therefore
does not prove that the lexical shortcut is unavailable.

### 4. The new late-minus-early windows are not always disjoint

`sampled_response_positions` floors `start_fraction` but ceils
`end_fraction` (`reward.py:61-68`). Adjacent fractional windows overlap for odd
response lengths. With `prompt_len=10`, `response_len=39`, and `stride=20`, the
current function returns position `[29]` for both `[0,.5)` and `[.5,1)`.
The same sampled overlap occurs for response lengths 79, 119, 159, 199, and
239. The test uses an even 100-token response and therefore misses the bug.

The live and recent affect/error "transition" experiments are not always
optimizing the stated disjoint late-minus-early quantity.

### 5. Affect/error literal audits on confirmation and full test used the wrong words

`configs/clean_confirmation_eval.json` and `configs/full_eval.json` inherit
`target_words: ["solved"]` from `jlens.json`. The other agent evaluated
happy/satisfied/nice adapters with these configs and `--skip-jlens-metric`.
Skipping the lens does not skip the substring counter: `eval.py:44-46` still
uses `cfg["target_words"]`.

Thus the reported `0% literal target usage` on affect confirmation/full-test
runs measured the word `solved`, not `happy`, `satisfied`, or `nice`. Exact
match is unaffected, and the in-training development audit used the right
config, but the claimed final anti-hacking check is invalid and cannot be
reconstructed because completions were not saved.

### 6. The reported effect is statistically uninformative

The base and adapter answer the same questions, but `eval.py` retains only
aggregate accuracy and separate one-model Wilson intervals (`eval.py:63-75`).
It does not retain item IDs, predictions, correctness pairs, or discordant
counts. Marginal Wilson intervals are not an interval or test for the paired
change.

Even in the most favorable possible paired table, a net gain of three with no
losses has two-sided exact McNemar `p = 0.25`; a gain of two has `p = 0.50`.
Any offsetting wins and losses make the evidence weaker. Two positive training
seeds also do not provide a useful seed-level variance estimate. The wording
"passes the exact acceptance gate" in `EXPERIMENT_HANDOFF.md:46-53` is much
stronger than the evidence permits.

### 7. Development and final evaluators are not demonstrably identical

The log already records `32.5%` from the trainer-wrapped step-zero model versus
`30.5%` from a separately loaded base on the same first 200 examples
(`RESEARCH_LOG.md:32-35`). That 2 percentage-point discrepancy dwarfs the
claimed 0.15-0.23 point full-test gains. The paths load models differently:
training passes a model name to GRPO without explicit model dtype
(`train.py:228-236`), while standalone evaluation explicitly chooses
BF16/FP16 (`eval.py:90-92`). The exact cause has not been resolved.

Checkpoint selection on one numerical/runtime path and final verification on
another is unsafe when decisions turn on one to three answers.

### 8. The old positive result did not use the current strict code

At committed HEAD `79f69d7`, `train.py:126-147` (inspect with
`git show HEAD:src/jlens_rl/train.py`) retained answers, instantiated both the
GSM8K verifier and J reward, and used weights `[0,1]` for a J run. A zero
verifier weight means the verifier should not directly affect the scalar reward
or gradient, so this is not evidence of a nonzero extrinsic reward. It does
mean the new "answers absent; verifier not even computed" guarantee cannot be
applied retroactively to the claimed runs.

The purity changes and clean configs are currently modified/untracked. The
live W&B metadata records only Git SHA `79f69d7`
(`wandb/run-20260714_001603-bqyb2rmi/files/wandb-metadata.json:14-17`), which
does not reproduce the code actually running.

## Other important gaps

- **No matched clean null/control.** There is no same-split, same-seed sham J
  word/random transport/sign-flipped reward control, and no successful clean
  exact-match control. The existing matched correctness-reward control was
  flat (`RESEARCH_LOG.md:123-130`), even though `PLAN.md:24` says it should
  validate the pipeline. Base-versus-adapter alone cannot distinguish semantic
  J-word learning from ordinary LoRA/GRPO drift plus selection.

- **The clean surrogate sets are weak.** Development `[7000,7200)` and
  confirmation `[6800,7000)` are adjacent contiguous train slices, not a
  randomized representative partition. Their base accuracies are about
  42.5-43.0%, versus 30.7% on test, and the clean affect candidate rose on both
  but fell on test. They are useful development sets, not independent evidence
  of test-distribution improvement.

- **Readout design has used correctness labels.**
  `analyze_alignment.py:124-172` uses GSM8K answers to choose layers/windows and
  fit composites. That is reasonable supervised reward design, but it weakens
  the stronger story that the reward was chosen without outcome information.
  The earlier shuffled alignment screens also include a few raw examples later
  assigned to the clean development/confirmation ranges.

- **No stale-lens diagnostic is reported for the positive runs.** The fixed
  base-model Jacobian can become invalid as LoRA changes hidden states.
  `PLAN.md:16` and `README.md:94-106` call for post-training refitting, but no
  accepted-run result is logged. Literal-word frequency alone cannot establish
  that the optimized feature remains a meaningful internal notion rather than
  a stale-lens exploit.

- **Evaluation is not auditable.** The CLI prints one aggregate JSON object
  (`eval.py:79-111`). It does not save per-example outputs, dataset indices,
  prompt hashes, base/adapter paired correctness, adapter hash, model revision,
  lens/calibration hashes, or Git/diff identity. The full-test results live
  mainly in prose and W&B; `artifacts/`, `runs/`, and `wandb/` are ignored.

- **Artifact identity is not enforced.** `TargetJLReward` reads only
  calibration mean/std and does not validate its token IDs, words, model, or
  layers against the config/lens (`reward.py:133-136`). A mismatched calibration
  silently runs. Lens fitting also reuses `${output}.checkpoint` without a
  repository-level manifest proving model/corpus/prompt identity
  (`fit_lens.py:126-130`).

- **Canonical docs and scripts run the old protocol.** README's commands use
  `configs/jlens.json`, which monitors test by default, and README/handoff/plan
  still say both rewards are computed with weights `[0,1]`
  (`README.md:67-85`, `EXPERIMENT_HANDOFF.md:100-103`, `PLAN.md:18`). The only
  root orchestration script, `run_solved_layer_screen.sh:7-18`, waits for and
  launches legacy test-monitor configs. There is no authoritative clean
  end-to-end script for fit -> train -> blinded confirmation -> final paired
  evaluation -> multi-seed summary.

- **Reruns can mix artifacts.** Training allows a nonempty output directory and
  `append_jsonl` always appends (`train.py:128-130`, `common.py:80-83`). Reusing
  a config can combine histories/checkpoints from different runs.

- **Revisions/determinism are incomplete.** Model and dataset revisions are not
  pinned, and `seed_everything` does not enable deterministic CUDA algorithms.
  Artifact hashes in the log help, but a fresh clone cannot reproduce the live
  dirty-tree experiment from the recorded SHA.

- **Tests are too shallow for the claim.** The nine passing tests do not
  integration-test answer removal, one-reward registration, split exclusion,
  correctness-independent stopping, causal next-token masking, odd-length
  window disjointness, literal variants, config/artifact identity, output-dir
  isolation, or paired base/adapter evaluation.

## Required next experiment

The active work can continue as exploration, but no further result from the
already opened GSM8K test/confirmation sets should be called confirmatory. A
credible test should do the following before training:

1. **Fix and test the reward implementation.** Align masking with the causal
   next token (and conservatively mask the target neighborhood), cover all
   tokenizer/context variants, use one shared rounding boundary for disjoint
   windows, and validate calibration/lens/config identity.

2. **Commit/tag the exact code and artifacts first.** Record Git SHA plus dirty
   diff hash (preferably no dirty diff), model and dataset revisions, resolved
   config, lens/calibration SHA-256, training indices, environment, and adapter
   hashes. Refuse to reuse nonempty output directories.

3. **Separate exploration from confirmation.** Freeze one target/readout,
   learning rate, update count, seed list, and checkpoint rule. Do not use
   exact-match early stopping in the confirmatory runs. Predeclare a finite
   variant/seed budget and an analysis rule.

4. **Use a genuinely sealed final set.** The GSM8K test set and current
   confirmation slice have already influenced decisions. Use a new blinded
   partition or external hidden evaluation that nobody/agent inspects until all
   choices are frozen. Randomize/stratify development splits rather than using
   adjacent tail ranges.

5. **Run matched controls and enough seeds.** At minimum include frozen/no-update
   base, the semantic J reward, a sham/random-word or random/sign-flipped J
   reward, and an exact-match-reward positive control under identical data and
   hyperparameters. Use at least five predeclared training seeds; more are
   preferable for an effect this small.

6. **Make evaluation paired and durable.** Use the exact same dtype/runtime for
   development and final evaluation. Save one JSONL row per example for base
   and every adapter, including prediction/correctness and provenance. Report
   discordant counts, exact McNemar and/or paired-bootstrap confidence interval,
   the per-seed effect distribution, and a multiple-search adjustment.

7. **Check semantic/stale-lens validity.** Refit or independently validate the
   lens after training, inspect saved completions, and show that any evaluation
   gain is not explained by target spelling, length, formatting, or a generic
   policy-update control.

8. **Add integration tests and one clean orchestration script.** The script
   should fail closed on leaked ranges, wrong target words, mismatched artifact
   metadata, dirty provenance, reused output directories, and premature access
   to the sealed set.

Until those conditions are met, the defensible finding is **"J-only optimizer
path verified; improvement hypothesis not yet demonstrated."**

## Remediation implemented after the audit

The follow-up change set closes the implementation and protocol defects above;
it does **not** retroactively validate any old run:

- J-only rows contain no answers and only one task-reward callable is
  registered. Confirmatory accuracy is observational at fixed steps and cannot
  stop training or choose a checkpoint.
- Literal masking now excludes the causal predecessor and full lower/title/
  upper-case token sequences. Adjacent fractional windows use one floor
  boundary and are disjoint for odd response lengths. Lens, calibration,
  tokenizer, model revision, layers, words, token IDs, and artifact hashes are
  checked before use.
- Model/dataset revisions and runtime dtype are pinned, deterministic settings
  are enabled, nonempty output directories are rejected, and each run records
  its resolved config, source fingerprint, artifact hashes, and exact raw
  training/validation indices.
- Evaluation writes paired per-item JSONL without gold answers. The comparison
  tool reports discordant pairs, exact McNemar diagnostics, deterministic
  paired bootstraps, a crossed seed/item bootstrap, the exact seed sign test,
  and semantic-versus-sign-flip difference-in-differences.
- The frozen v2 protocol excludes 3,741 historically used GSM8K-train indices,
  permanently retires the exposed 400-item v1 curve, and allocates a new
  400-item curve, 2,900-item sealed final set, and 64-item reserve. It
  predeclares six seeds, step-25 endpoints, and the exact curve nodes
  `0/5/10/15`: step 5 must exceed step 0, then steps 10 and 15 may not fall.
  Six semantic and six sign-flipped runs are required; a one-seed exact-match
  control is optional and cannot rescue the primary result.
- Significant evidence is accepted only if that curve gate passes, all six
  sealed-set treatment effects are positive (two-sided exact sign-test
  `p=0.03125`), and the 95% crossed seed/item bootstrap intervals for both the
  mean treatment-minus-base effect and semantic-minus-sign-flip
  difference-in-differences exclude zero. A hashed curve PNG, raw-evaluation
  hashes, frozen adapter hashes, and a machine-checked final acceptance report
  are durable outputs.
- `modal_experiments.py` preserves this sequence on Modal and caps each GPU
  phase at five pinned L40S containers, queuing the sixth. It excludes credential files
  from the image and stops before controls/final evaluation when the curve gate
  fails. The first detached smoke launch was rejected before GPU dispatch by
  the clean-tree guard because Modal materialized three tracked symlinks and
  package installation left `build/` debris. The image build now restores the
  exact symlink types/targets, removes that deterministic debris, asserts a
  clean checkout, and uses a fresh state volume for each corrected launch.

The regenerated `solved` calibration is bound to the pinned model revision and
the unchanged lens transport. Historical affect/error calibrations now fail
closed until separately regenerated; their pre-fix runs remain exploratory.
At commit time, the research conclusion therefore remains **inconclusive**:
the repository is ready to collect valid evidence, but no valid v2 outcome has
yet been observed.

## Prelaunch Modal and protocol audit (2026-07-14)

The first GPU-bearing Modal attempt was manually stopped and is exploratory
only. Five semantic seeds (142–146) had started on Volume
`j-lens-rl-confirmatory-v1-20260714b`; seed 147 was queued. All five observed
step-5 accuracies exceeded their common `0.355` baseline:
`0.3725, 0.3775, 0.3650, 0.3600, 0.3650` (mean `0.3680`). Available step-10
values were `0.3800`, `0.3600`, and `0.3725`, each non-downward from its own
step 5. Literal `solved` use was `0%` to `0.25%`, and validation response length
did not inflate. These values are encouraging diagnostics, not evidence: the
runs were incomplete and invalid for the reasons below.

The stop exposed two independent blockers:

1. Modal executed the wheel-installed package under `site-packages`, so the
   old fixed-parent repository lookup emitted null Git commit, dirty-state,
   and source-tree fields in every run manifest. The pipeline would otherwise
   have spent all 12 GPU runs before failing at unlock.
2. Historical reconstruction omitted documented setup run `xufk8x08`. That
   seed-42 run selected 1,000 rows after excluding only raw `7000:7200`, a
   different pool from the already reconstructed selections. The omission
   missed 331 used indices: 32 had entered the old curve and 250 had entered
   the nominal 3,000-item final set. Therefore v1's data manifests cannot
   support a freshness claim.

V2 fixes both problems before any sealed outcome is opened:

- source-root resolution is explicit in the Modal image and fail-closed in
  training and evaluation; image construction verifies imports come from the
  baked checkout;
- the omitted setup selection is an explicit frozen reconstruction rule;
  direct reconstruction yields 3,741 historical and 3,732 truly unused rows;
- all 400 old curve indices are retired. Of those, 32 are now historical and
  368 were otherwise fresh, leaving exactly 3,364 unseen rows for the new
  `400/2,900/64` curve/final/reserve split;
- a fresh Volume has an exclusive attempt claim and rejects stale run, eval,
  evidence, or unlock paths; semantic runs are fully verified before controls;
- all confirmatory training and evaluation functions are pinned to L40S and
  record CUDA identity; mixed runtimes fail verification;
- unlock freezes hashes of every run manifest, config, data selection,
  history, terminal trainer state, and final adapter. Evaluation binds each
  label to that adapter and the prepared commit;
- final verification reloads the pinned GSM8K revision and tokenizer,
  recomputes prompt/token hashes, parsed predictions, and correctness from the
  saved completions, then recomputes both statistical summaries. Acceptance
  hashes all 13 evaluation JSONLs and refuses a nonsignificant `complete`
  status;
- the filename mismatch (`acceptance_report.json` versus `acceptance.json`),
  missing label binding, stale-output reuse, duplicate-launch race, and false
  top-level success status found in the initial Modal audit are closed.

The significant-evidence definition was frozen before v2 preparation: the
requested `0/5/10/15` six-seed mean curve, six positive sealed treatment
effects, exact two-sided sign-test `p < 0.05`, a treatment-minus-base crossed
95% interval above zero, and a semantic-minus-sign-flip crossed 95% interval
above zero. Neither an internal J-score increase nor the invalid v1 partial
curve can satisfy this definition.

Operational security note: a credential was exposed in internal agent tool
output during this audit. It is excluded from Git and the Modal image, but it
should be rotated after the active launch no longer needs it. The credential
value is intentionally not reproduced here.

## V2 Modal outcome and frozen v3 branch (2026-07-14)

Corrected v2 ran on Modal app `ap-sRwSHPcrmbyFJJlYUBOhws`, Volume
`j-lens-rl-confirmatory-v2-20260714a`, from clean commit
`2c00f7fadc3e202c1516c21c4e804ab486ff6cb7`. All six semantic manifests
recorded the same non-null source-tree hash, the pinned model/dataset and
lens/calibration hashes, `reward_type: jlens`, and `NVIDIA L40S` with CUDA
12.8. Five workers ran concurrently and seed 147 queued as intended. No
traceback, OOM, literal-target emission, stale output, or provenance mismatch
was observed before the scientific gate resolved.

The fresh, previously unseen curve gave this decisive result:

| Seed | Step 0 | Step 5 | Step 10 | Step 15 |
|---:|---:|---:|---:|---:|
| 142 | 0.3750 | 0.3625 | 0.3750 | 0.3500 |
| 143 | 0.3750 | 0.3750 | 0.3575 | 0.3725 |
| 144 | 0.3750 | 0.3850 | 0.3625 | 0.3775 |
| 145 | 0.3750 | 0.3900 | 0.3575 | 0.3925 |
| 146 | 0.3750 | 0.4025 | 0.3800 | 0.3800 |
| 147 | 0.3750 | 0.3800 | 0.3800 | 0.3900 |
| **Mean** | **0.37500** | **0.38250** | **0.36875** | **0.37708** |

Thus J-only training produced the requested first rise, `+0.75` percentage
points at step 5, but then fell `1.375` points at step 10. This violates the
predeclared no-downward-step condition and puts step 10 `0.625` points below
the frozen base. Step 15 recovered to `0.37708`, but a later recovery cannot
erase the intervening decline. V2 is a valid negative curve-gate result, not
significant evidence. The automatic verifier wrote `curve_gate.json` with
`passed: false` and a curve plot SHA-256 of
`5e7dbac3d5d6f63630008bb3fe1579428b92296a195c9a248ed747403f93261e`;
the durable attempt stage became `curve_failed`, the app stopped at
2026-07-14 02:44:53 UTC, no sign-flip control started, and none of the 2,900
sealed-final outcomes was opened. The invalid v1 result is not pooled with
this result.

The completed semantic logs also reject the obvious reward-hacking
explanations: literal `solved` emission was zero at every rollout and curve
node, within-group reward variance never collapsed, KL stayed below 0.0015,
the policy clip-region ratio stayed zero, and validation response length was
stable. Training used only the one J task reward. The failure is therefore an
accuracy-trajectory failure, not evidence that the optimizer silently saw
correctness or learned to print the target word.

One reproducibility caveat remains: PyTorch warned that Flash Attention and
memory-efficient attention backward kernels are nondeterministic even though
the code requests deterministic algorithms with `warn_only`. This does not
leak labels or bias the gate, and seed-level variation is explicitly part of
the multi-seed design, but the runs are not promised to be bitwise replayable.
V3 keeps the same pinned L40S/CUDA/software runtime rather than silently
changing the training kernel together with the scheduler; the warning and
runtime identity must remain in the evidence record.

The pre-gate contingency audit found one concrete recipe mismatch that best
explains why the early rise was not sustained. The historical dense `solved`
runs used `max_steps: 500`; their nominal linear schedule therefore stayed
near `3e-6` during the first 25 updates, and their two-seed mean at steps
`0/10/20/25` was `0.3250/0.3275/0.3325/0.3325`. V2 shortened
`max_steps` to 25 without freezing the scheduler, so the default linear
schedule logged approximately `2.52e-6`, `1.92e-6`, `1.32e-6`, `0.72e-6`,
and `0.12e-6` at steps `5/10/15/20/25`. The confirmatory config therefore did
not actually preserve the historically selected learning-rate recipe. This is
a protocol/configuration issue, not permission to select a favorable v2
checkpoint.

Before seed 147's step-10 result was available, the contingency analysis froze
the following branch conditional on v2 failure. V3 changes only the scheduler
to constant `3e-6` with zero warmup and otherwise retains the exact J-only
`solved`, layer-8 late-half, KL, sampling, LoRA, masking, fixed-step, and
observational-evaluation recipe. It uses ten new semantic seeds 148--157 and
ten matched sign-flip controls. At the five-GPU cap, each condition still takes
two waves, while acceptance requires at least 9/10 strictly positive sealed
effects (two-sided exact sign-test `p=0.021484375`) plus positive crossed
95% lower bounds for both semantic-minus-base and semantic-minus-sign-flip.

Because v2 never opens a sealed outcome, v3 may deterministically rehash only
v2's still-unopened 2,900-item sealed pool under a new frozen salt: 800 items
become the sole v3 curve and 2,100 remain sealed final; the separate 64-item
reserve stays untouched. The curve gate remains the exact ten-seed mean at
`0/5/10/15`, with step 5 strictly above baseline and no later decline, and the
only endpoint remains step 25. V2's exposed 400-item curve is permanently
retired. V3 remains the sole final-outcome look, so it does not spend alpha on
the failed descriptive v2 gate; if any v2 sealed row is opened, this branch is
invalid and must not run.

## V3 outcome and parallel reward-design screens (2026-07-14)

### Confirmatory v3 is a valid negative result

V3 ran from clean commit
`a617b59f20c84172454b7cd80b9668535f11be8f` (source-tree SHA-256
`f888e27210cf883edd6eee2dcda5dccd8813852eebc0933362ea58502b22770b`)
on Modal app `ap-HF5YGgsdlh2D6m9qELb0D1` and Volume
`j-lens-rl-confirmatory-v3-20260714a`. The app stopped normally at
2026-07-14 04:09:37 UTC. All ten run manifests record clean source, the pinned
model/dataset/lens/calibration, `reward_type: jlens`, NVIDIA L40S/CUDA 12.8,
one layer-8 late-half `solved` component, constant `3e-6`, zero warmup, fixed
25 updates, and observational evaluation every five updates. At most five GPU
workers were live at once.

The exact frozen curve was:

| Seed | Step 0 | Step 5 | Step 10 | Step 15 |
|---:|---:|---:|---:|---:|
| 148 | 0.43500 | 0.40750 | 0.43250 | 0.42250 |
| 149 | 0.43500 | 0.43000 | 0.41625 | 0.42500 |
| 150 | 0.43500 | 0.42500 | 0.41125 | 0.42125 |
| 151 | 0.43500 | 0.41750 | 0.41625 | 0.40625 |
| 152 | 0.43500 | 0.41875 | 0.42000 | 0.41375 |
| 153 | 0.43500 | 0.43000 | 0.41625 | 0.42000 |
| 154 | 0.43500 | 0.42000 | 0.42250 | 0.41125 |
| 155 | 0.43500 | 0.42500 | 0.42500 | 0.42875 |
| 156 | 0.43500 | 0.43500 | 0.43125 | 0.41500 |
| 157 | 0.43500 | 0.40375 | 0.43250 | 0.42125 |
| **Mean** | **0.435000** | **0.421250** | **0.422375** | **0.418500** |

The first change was `-1.375` percentage points, step 5 to 10 recovered only
`0.1125` points, and step 10 to 15 fell another `0.3875` points. The stored
gate has `passed: false`; the curve PNG SHA-256 is
`a1fdc390f4b4c6e9923f639a760798b6e206d5711c5942f22a172bb0043f8d95`.
The durable stage is `curve_failed`. No sign-flip control, unlock file, base
final evaluation, or sealed-final adapter evaluation exists. V3 therefore
adds valid negative evidence and no significant positive evidence.

The failure is not explained by an obvious shortcut. All 60 greedy validation
rows had zero literal-target completions and stable mean length (228.82 tokens
at baseline and 228.29 at step 25). Reward variance was positive in all 250
updates (minimum standard deviation 0.1838), KL was at most 0.001609, and no
NaN, OOM, traceback, or provenance failure occurred. Training did have mild
length pressure and eight literal occurrences among 2,000 sampled rollouts;
these occurred in isolated masked batches and are disclosed rather than
rounded down to zero. Twenty-one of 250 rollout batches were fully capped at
256 tokens, but that pressure did not transfer to greedy validation. The known
nondeterministic attention warning remains a replay caveat.

### First parallel exploratory design screen

The first independent screen ran three *different reward constructions*, not
three seeds, from clean commit `20635432d3e31c2c85418d00873a47d078ec9fa6`
on app `ap-XAiL1nGDZdmSB7NCX8oJpI`, call
`fc-01KXFBK014AX86Q33CPZMBV0MT`, and fresh Volume
`j-lens-rl-exploratory-screen-v1-20260714a`. The peak was three L40S workers.
All variants used seed 158, constant `3e-6`, the retired/exposed V2 400-item
development curve, and fixed 25-update endpoints. A combined exclusion-only
manifest kept the retired curve and every current V3 curve/final/reserve row
out of all 1,000 selected training examples. Live reconstruction found zero
train/evaluation and zero train/current-V3 overlap. Current V3 manifests and
outcomes were not mounted.

| Reward construction | 0 | 5 | 10 | 15 | 20 | 25 | Requested pattern |
|---|---:|---:|---:|---:|---:|---:|:---:|
| Layer-8 late minus early | .3750 | .3750 | .3900 | .3725 | .3750 | .3850 | No |
| Layer-8 late, stride 10 | .3750 | .3775 | .3825 | .3775 | .3750 | .3775 | No |
| Equal layers 8/14/20 late | .3750 | .3850 | .3675 | .3775 | .3825 | .3575 | No |

The dense variant was closest but its step-15 decline is still a failure. All
validation and training literal rates were zero, validation lengths remained
stable, reward variance never collapsed, all recorded learning rates were
exactly `3e-6`, and the app stopped normally with durable histories, results,
and adapters. These are adaptive development results only.

### Second parallel screen and pre-outcome v4 branch

The screen script is committed at
`56280f0a8f2da1eb3a4c6106f49b7cacca6a6489`; `40/40` tests, Python syntax,
and `git diff --check` pass. Screen 2 completed durably at
2026-07-14T04:35:41Z and the app stopped normally at 2026-07-14T04:36:44Z on
app
`ap-dt7cQSw2be0iYFolyt2nQp`, call `fc-01KXFDGBTCYDYDSNH3E21C362B`, and fresh
Volume `j-lens-rl-exploratory-screen-v2-20260714a`, capped at four L40S
workers. All four live manifests were independently checked for clean matching
source, J-only reward, L40S/CUDA 12.8, constant `3e-6`, fixed 25 updates, the
same 1,000 training examples and 400 retired validation examples, and zero
overlap with validation or any current V3 allocation. The ignored manifest
files are now pinned to their exact SHA-256 values before launch, and the claim
function is serialized to reject a simultaneous duplicate submission.

The screen evaluates only at `0/2/4/6/10/15/20/25`. Its newly fixed requested
curve is step 2 strictly above baseline, step 4 non-downward, and step 6
non-downward. The four reward ideas, in fixed priority order, are:

1. layer-8 late-half mean with stride 5 (`ultradense5`);
2. stride-10 layer-8 quarters weighted `1.0/0.25` (`tail_taper`);
3. stride-10 late `+1`, early `-0.25` (`tempered_delta`);
4. stride-10 late layers 8/14/20 weighted `0.8/0.1/0.1`
   (`layer_shrink`).

At 2026-07-14T04:23:11Z, only the first post-update node had been inspected:
`ultradense5` was `.3750 -> .3875`, `tail_taper` `.3750 -> .3800`,
`tempered_delta` `.3750 -> .3575`, and `layer_shrink` `.3750 -> .3675`.
The latter two could no longer pass. No step-4 or step-6 outcome had been used
when the following branch was frozen. If multiple variants passed, the rule
selected the first in the priority list above; neither effect magnitude nor a
later diagnostic node could override it.

The complete persisted screen was:

| Reward construction | 0 | 2 | 4 | 6 | 10 | 15 | 20 | 25 | Fixed gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| `ultradense5` | .3750 | .3875 | .3700 | .3675 | .3750 | .4050 | .3950 | .3700 | Fail |
| `tail_taper` | .3750 | .3800 | .3800 | .3825 | .3725 | .3775 | .3775 | .3850 | **Pass** |
| `tempered_delta` | .3750 | .3575 | .3575 | .3775 | .3800 | .3775 | .3775 | .3800 | Fail |
| `layer_shrink` | .3750 | .3675 | .3600 | .3800 | .3825 | .3675 | .3925 | .3775 | Fail |

Thus `tail_taper`, priority rank two, is the unique eligible recipe. Its frozen
gate is `.3750 -> .3800 -> .3800 -> .3825`: the first node rises and the next
two do not fall. Its later step-10 dip to `.3725` is retained rather than
hidden or used to reselect a different candidate; the endpoint recovers to
`.3850`. Every one of the 32 validation rows had zero literal-target
completion, validation response lengths stayed approximately 228--231 tokens,
and reward variance never collapsed. Training literal rate was also zero
except for one `tail_taper` rollout batch at step 19 (one of eight completions);
that batch occurred after the frozen gate, target/predecessor positions were
masked, and its reward standard deviation was 0.566. KL stayed below 0.0016,
the policy clip-region ratio was zero, and every persisted non-evaluation LR
row was exactly `3e-6`. Sampled training lengths did have isolated fully capped
batches (mean clipped ratios approximately 0.53--0.59), but no corresponding
greedy-evaluation length collapse or inflation occurred. No worker failed.
This is still adaptive candidate-selection evidence, not significant evidence.

Conditional v4 is permitted only after re-verifying V3's formal
`curve_failed` closeout and the absence of any V3 final outcome. It orders only
V3's unopened 2,100-item sealed parent (file SHA-256
`84da0c0472b4442b4f35406d1b1fbd3b956803e5f19bf51fc02f6db013224f7b`,
sorted-set SHA-256
`875334925160d6c0c49dd8cf1523e1aeb081fd90f6e4b08611eccb8394dbe4d5`)
by ascending `SHA256("j-lens-rl-confirmatory-v4-screen2-2026-07-14:" + index)`.
The first 400 indices are the sole new curve (manifest SHA-256
`ad348fe17d2e6bd6aac691d9bcdbb9da481f675305fa0e05c68e86dad97451c1`)
and the remaining 1,700 are the sole sealed final (manifest SHA-256
`acd2d497dcf96b2f3355925bb34979b9b7b3301e4c394066fc54ea57d093b6e3`).
The V3 800-item curve is retired and the 64-item reserve remains untouched.

V4 uses new seeds 159--166, eight semantic runs in one eight-L40S wave, the
same fixed `0/2/4/6` mean-curve gate, and fixed step-25 endpoints. On a curve
pass it runs eight matched sign flips, negating every chosen component weight
and changing nothing else. Only after all 16 runs verify may it evaluate one
base, eight semantic adapters, and eight sign-flip adapters on the new 1,700.
Acceptance requires the curve pass, positive mean semantic-minus-base effect,
a crossed seed/item 95% lower bound above zero, all eight seed effects strictly
positive (ties fail; exact two-sided sign `p=0.0078125`), positive
semantic-minus-sign-flip difference-in-differences, and its crossed 95% lower
bound above zero, plus unchanged provenance, raw-record, target-literal, and
artifact checks. No correctness-reward control is on the critical path.

The current research conclusion is therefore unchanged: the J-only optimizer
path is verified, but reliable or significant evaluation improvement has not
yet been demonstrated.

### Screen-2 resolution and confirmatory v4 implementation

The precommitted `0/2/4/6` screen gate resolved as follows:

| Priority | Reward | 0 | 2 | 4 | 6 | Gate |
|---:|---|---:|---:|---:|---:|:---:|
| 1 | `ultradense5` | .3750 | .3875 | .3700 | .3675 | fail |
| 2 | `tail_taper` | .3750 | .3800 | .3800 | .3825 | pass |
| 3 | `tempered_delta` | .3750 | .3575 | .3575 | .3775 | fail |
| 4 | `layer_shrink` | .3750 | .3675 | .3600 | .3800 | fail |

The frozen first-passing-priority rule therefore selects `tail_taper`; effect
size and later nodes do not alter that decision. Its full observational curve
at `0/2/4/6/10/15/20/25` is
`.3750/.3800/.3800/.3825/.3725/.3775/.3775/.3850`. The decline at step 10 is
outside the fixed candidate-selection gate and is disclosed, not silently
discarded. This retired-development result is still not significant evidence.

V4 was implemented in the isolated `/j-lens-v4` clone without touching a live
Volume or reading any sealed outcome. Preparation now requires byte-exact
archives of both screen selection and V3 closeout. The closeout records V3's
`curve_failed` status, ten semantic-only run labels, absence of `evals/`,
absence of an unlock, and an evidence directory containing only the failed
curve gate and plot. Both archive hashes are pinned in code and copied into
prepared state.

The allocator reconstructs V3 and verifies the unopened parent file hash
`84da0c0472b4442b4f35406d1b1fbd3b956803e5f19bf51fc02f6db013224f7b`
and sorted-set hash
`875334925160d6c0c49dd8cf1523e1aeb081fd90f6e4b08611eccb8394dbe4d5`.
It then deterministically produces the predeclared 400-item curve hash
`ad348fe17d2e6bd6aac691d9bcdbb9da481f675305fa0e05c68e86dad97451c1`
and 1,700-item final hash
`acd2d497dcf96b2f3355925bb34979b9b7b3301e4c394066fc54ea57d093b6e3`,
retires V3's 800 curve rows, and preserves the 64-item reserve unchanged.

The frozen V4 treatment is stride-10 layer-8 `tail_taper` with weights
`1/.25`; the control exactly negates both weights. Seeds are 159--166, all
runs end at update 25, and validation nodes are
`0/2/4/6/10/15/20/25`. Sign-flip workers independently verify the stored
passed `0/2/4/6` gate before training. All 16 run manifests must share one
non-null source-tree fingerprint. Final evaluators must use that same
fingerprint.

After unlock, Modal submits one immutable 17-label collection (base, all eight
semantic, all eight sign flips), capped at eight L40S workers. It performs no
semantic-only sealed analysis or decision before the complete fixed collection
finishes. Acceptance requires all eight semantic effects strictly positive
(ties fail; exact two-sided sign `p=.0078125`) and crossed 95% lower bounds
above zero for both semantic-minus-base and semantic-minus-sign-flip, in
addition to the mean, curve, record, artifact, runtime, and provenance gates.
V4 is frozen but not launched by this implementation step.
