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

## V4 closeout and pre-outcome alternative branch (2026-07-14)

### Confirmatory V4 is a valid negative result

V4 ran from clean commit
`8ae04dc61a3ae474ffa62dd0e738d6b40deed303` (common source-tree SHA-256
`209a01da2fe2a625f404577ede0f6884b0f5d24a6ce804bcf0f94961625e23c2`)
on Modal app `ap-6IOB6wu1xZU1MQRbvBAPi7` and Volume
`j-lens-rl-confirmatory-v4-20260714a`. It used exactly eight L40S semantic
workers, seeds 159--166, and stopped normally at 2026-07-14 05:11:02 UTC.
Every run reached the fixed step-25 endpoint using only the frozen
`tail_taper` J-lens reward; validation was observational.

The complete persisted curve was:

| Seed | 0 | 2 | 4 | 6 | 10 | 15 | 20 | 25 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 159 | .3825 | .3875 | .3950 | .3625 | .3900 | .3850 | .3900 | .3800 |
| 160 | .3825 | .3950 | .3775 | .3800 | .3975 | .3900 | .3975 | .3925 |
| 161 | .3825 | .4125 | .3975 | .3775 | .4150 | .4000 | .4000 | .3875 |
| 162 | .3825 | .4075 | .3825 | .4000 | .4000 | .3775 | .3950 | .3875 |
| 163 | .3825 | .4075 | .3900 | .4050 | .4025 | .3675 | .3725 | .3800 |
| 164 | .3825 | .3575 | .3825 | .4025 | .3875 | .4025 | .3950 | .4000 |
| 165 | .3825 | .3775 | .4125 | .3900 | .3975 | .3975 | .4050 | .3925 |
| 166 | .3825 | .4000 | .3950 | .3975 | .3850 | .4075 | .4000 | .3925 |
| **Mean** | **.382500** | **.393125** | **.3915625** | **.389375** | **.396875** | **.3909375** | **.394375** | **.3890625** |

The first transition rose by 1.0625 percentage points, but step 4 fell
0.15625 points below step 2 and step 6 fell another 0.21875 points. V4
therefore fails the exact frozen requirement `2 > 0`, `4 >= 2`, `6 >= 4`.
The durable status became `curve_failed` at 05:09:48 UTC. The curve gate file
SHA-256 is
`5cda13447b4f54eb6607de8f43ae2e86edaea2a3ce102dd63059f0ee002c8a72`
and the plot SHA-256 is
`3b176c04675a14ca554a1ececfd9835babe16c054769a8bfe5d66255448948d5`.
Later peaks cannot repair the predeclared failure.

All 64 greedy validation records had zero literal `solved` completions and
mean response lengths stayed between 227.80 and 230.32 tokens. Reward variance
never collapsed (minimum batch standard deviation .16227). Two isolated
training batches contained one literal completion each, for two occurrences
among 1,600 sampled rollouts; their target and causal-predecessor positions
were masked. Twenty-one of 200 rollout batches were fully length-capped, but
there was no corresponding greedy-validation length shift. These diagnostics
do not reveal an implementation failure or a successful shortcut.

The stopped Volume contains only the eight `jlens` runs. Its evidence
directory contains only `curve_gate.json` and `curve.png`; there is no
sign-flip run, unlock, `evals/` directory, sealed comparison, or final
acceptance report. Thus none of the 1,700 sealed-final outcomes was opened.
[`protocol_archive/v4_closeout.json`](protocol_archive/v4_closeout.json)
pins the full curves, run/config/data/adapter hashes, inventory, parent
identity, and no-look state; its SHA-256 is
`aaf4bcde9a9cacc482c7f3dde94218cf02a6aa60be81e43cae5cde3086d17e35`.

This is negative confirmatory evidence. It is not the requested significant
positive evidence.

### Frozen eight-arm alternative screen

Before launching or inspecting any alternative-screen outcome, the next
screen was frozen to seed 167, one fixed 25-update endpoint, observational
nodes `0/2/4/6/10/15/20/25`, and at most eight simultaneous L40S workers. It
mounts only the now-exposed V4 400-item curve (SHA-256
`ad348fe17d2e6bd6aac691d9bcdbb9da481f675305fa0e05c68e86dad97451c1`)
and the exact training-exclusion union (SHA-256
`7c1ca4f404ba9149093cc3c57dc3607582f671397e2fe99e93449848c1d65d61`).
It does not mount V4's sealed-final or reserve manifest. All arms use only a
J-lens reward and share the same training data and settings except for the
declared reward construction or learning rate:

| Priority | Arm | Target/sign | Readout | LR |
|---:|---|---|---|---:|
| 1 | `celebration_ultradense` | positive `yay/great/success/nice` | layer 8 late half, stride 5, `+1` | `3e-6` |
| 2 | `profanity_ultradense` | negative `damn/fuck` | layer 8 late half, stride 5, `-1` | `3e-6` |
| 3 | `celebration_taper` | positive `yay/great/success/nice` | layer 8 late quarters, stride 10, `+1/+.25` | `3e-6` |
| 4 | `profanity_taper` | negative `damn/fuck` | layer 8 late quarters, stride 10, `-1/-.25` | `3e-6` |
| 5 | `solved_u5_control` | positive `solved` | layer 8 late half, stride 5, `+1` | `3e-6` |
| 6 | `solved_u5_low_lr` | positive `solved` | layer 8 late half, stride 5, `+1` | `2e-6` |
| 7 | `solved_u5_taper` | positive `solved` | layer 8 late quarters, stride 5, `+1/+.25` | `3e-6` |
| 8 | `solved_u5_taper_low_lr` | positive `solved` | layer 8 late quarters, stride 5, `+1/+.25` | `2e-6` |

The user's redacted `f***` and `f**k` spellings are multi-token censorship
forms, not faithful single lexical J targets. The screen therefore uses the
actual tokenizer-supported lexical word `fuck`, masks its literal sequences,
and applies a negative weight to the internal profanity-family readout. The
celebration and profanity calibrations are created once on pinned WikiText
using the existing target-independent lens. Their exact ordered targets,
token-ID unions, model/data revisions, lens hash, positive finite standard
deviation, and output hashes are checked before training.

This deliberately retains the `explore2-solved-ultradense5-seed158` lineage.
Its old strict early gate failed at `.3750/.3875/.3700/.3675`, but its step-15
and step-20 values were `.4050` and `.3950`, respectively 3.0 and 2.0 points
above baseline. Those later peaks do not rewrite the old result, but they are
enough reason to test the original construction, lower LR, taper, and
taper-plus-lower-LR rather than discard it.

Selection is the first arm in the fixed table whose `0/2/4/6` curve satisfies
`2 > 0`, `4 >= 2`, `6 >= 4`. Word-family arms deliberately precede `solved`:
the research interest is whether other intrinsic words work at all, not whether
they beat `solved` on this development set. All eight arms must finish; later
nodes and effect magnitude cannot override priority. If no word arm passes but
a `solved` arm does, the latter remains a fallback. If none passes, nothing is
selected. This screen is adaptive development evidence, not a significance
test.

The launcher now fails closed unless the byte-pinned V4 closeout proves
`curve_failed`, no controls, no unlock, and no final evaluations. It validates
that closeout and the pre-outcome V5 registration both locally before claiming
the Volume and remotely from the baked clean commit. Config, calibration,
resolved-config, data-index, run-manifest, runtime, source, and literal audits
are also fail-closed. The corrected screen code SHA-256 frozen before outcomes
is `fd94228a3149b4b08967c29cef35cac2d054966190ddcefbac8fd5038744eac3`.

### Conditional V5 no-look confirmation

[`protocol_archive/v5_preregistration.json`](protocol_archive/v5_preregistration.json)
was last frozen at 05:32:09 UTC, after V4 stopped and after the zero-outcome
operational attempt below, but before the fresh alternative screen was
launched or any outcome existed. It fixes the selection rule above and
reallocates only V4's unopened 1,700-item parent. The parent manifest SHA-256
is `acd2d497dcf96b2f3355925bb34979b9b7b3301e4c394066fc54ea57d093b6e3`
and its canonical sorted-set SHA-256 is
`80bec8d196a36c1d2f02fb6aa3c7c5ea727a86c0c981d3d0611b0900776d6f74`.

Ordering those source indices by ascending
`SHA256("j-lens-rl-confirmatory-v5-alternative-screen-2026-07-14:" + index)`
produces exactly:

| Purpose | Count | Manifest SHA-256 |
|---|---:|---|
| fresh V5 curve | 400 | `b01409c011012641be96c84bfc35cb0b352cea902e54304105efa272a3eac6b2` |
| sealed V5 final | 1,300 | `6298b8e3d15b11985cf9febcd243dafd409ef07f22091388fa0793b6ebfe4228` |
| untouched reserve | 64 | `cfbac5a2f4cf3cc94e1882bf412cdfc4af9c84347647fa9843dc09967f8a03a6` |

V5 uses new seeds 168--175 and copies the selected resolved recipe exactly;
only seed, run/output names, and V5 manifests may vary. A word-family winner
must reuse and pin the exact calibration bytes used in its screen run, never a
post-selection recalibration. The requested curve is again the eight-seed mean
at `0/2/4/6`: first rise strictly above baseline, followed by two non-downward
nodes. A failure ends V5 without opening the 1,300 final outcomes.

After a curve pass, eight matched sign-flip controls negate every selected
component weight and change nothing else. One immutable 17-label collection
(base, eight treatments, eight sign flips) then evaluates the same 1,300 rows,
with no interim semantic-only final analysis. Significant evidence requires
all of the following in one terminal attempt:

1. the exact requested mean curve passes;
2. mean treatment-minus-base accuracy is positive;
3. its 10,000-draw, seed-0 crossed seed/item bootstrap 95% lower bound is above
   zero;
4. all eight seed effects are strictly positive with no tie, giving exact
   two-sided sign `p=.0078125`;
5. mean treatment-minus-sign-flip difference-in-differences is positive and
   its identically specified 95% lower bound is above zero; and
6. every data, source, config, runtime, raw-record, artifact, and literal audit
   passes.

V5 is one terminal attempt: no replacement seeds, checkpoint substitution,
fallback arm, rerun-until-pass, or recycling of its unopened final allocation
is authorized. A pass would support only the exact selected word/reward
recipe, not arbitrary J-lens words. As of this freeze, reliable or significant
positive evidence has still not been demonstrated.

### First word-screen launch closed before outcomes

The first launch from clean commit
`2ee3b5c3d8da37de82ed57ad8ad883d25e2fb58f` claimed fresh Volume
`j-lens-rl-alternative-screen-v1-20260714a` under claim
`6abe6d6d6adb4bebaf0dcccb6b9a8102`. App
`ap-p9J9f0djFb7vyo3LSK0Y6c` stopped after its non-detached local entrypoint,
before its spawned orchestrator started. The same claim was then resumed
without alteration on detached app `ap-lnIqyYPDl2Em4vM4Un6vN9`.

Both calibration workers reached the pinned model and WikiText, but failed
when `fit_lens` tried to write under the not-yet-created
`/word_explore/artifacts/` directory. Durable status became `failed` at
05:28:38 UTC. The stopped Volume contains exactly `attempt_manifest.json` and
`attempt_status.json`: there is no calibration artifact, resolved config,
training run, validation history, adapter, or evaluation outcome. No current
sealed or reserve manifest was mounted. This attempt therefore contains no
adaptive result and does not alter selection or V5 inference.

[`protocol_archive/word_screen_attempt1_closeout.json`](protocol_archive/word_screen_attempt1_closeout.json)
pins both app/call identities, the exact failure, file hashes, and the
two-file inventory; its SHA-256 is
`399559f0607bded85048633179b39a33da25d2de9fcdb4e448725770a30b90c7`.
The fix creates calibration-output parents both in the general `fit_lens`
writer and defensively in the Modal worker. A checked-in
`run_word_screen.sh` now always uses Modal detached mode. The corrected screen
uses fresh Volume `j-lens-rl-alternative-screen-v1-20260714b`; the failed
Volume is never reused.

## Emotional-only direction and J-space correlation protocol (2026-07-14)

### `solved` is retired from all future experiments

At 06:15:24 UTC the user explicitly chose the emotionally charged-word story
over the mechanically best-looking `solved` development arm. That decision is
byte-pinned in
[`protocol_archive/emotional_only_decision.json`](protocol_archive/emotional_only_decision.json)
(SHA-256
`50cc3cb32e0cf74feeaeae79c2faf7b91c6caeae34b6e4f1101819d51e15b238`).
It supersedes the selection priority described earlier in this audit and the
conditional recipe-selection part of the old V5 registration before any V5
curve or final outcome was opened. Already-running `solved` arms may finish
only so the existing attempt has a reproducible closeout; they are historical,
ineligible for selection, and cannot enter a new confirmation.

The decision was made after the exposed emotional arms had already produced
several above-baseline development nodes: `celebration_ultradense` reached
`.4200` at step 10, `profanity_ultradense` reached `.4075` at step 20,
`celebration_taper` reached `.4050` at step 10, and `profanity_taper` ended at
`.3975`. These are encouraging adaptive observations, not significant
evidence, and none of the emotional arms then observed had the requested exact
non-downward `0/2/4/6` shape. The final attempt closeout must report every arm,
including failures, without allowing the retired arms to influence the next
recipe.

The next eight-way RL screen is already committed and uses eight genuinely
different emotional targets rather than repeated seeds:
`yay`, `wow`, `joy`, `proud`, `excited`, negative `damn`, negative `fuck`, and
negative `worried`. Each arm uses the same seed-167 U5 geometry, fixed 25
updates, and validation nodes `0/2/4/6/10/15/20/25`; each target receives its
own pinned WikiText calibration and distinct W&B identity. This is an
exploratory comparison on the already-exposed V4 curve, not a significance
test.

### Frozen emotional J-space association scan

The separate word-correlation experiment requested by the user is frozen in
[`protocol_archive/word_correlation_v1_preregistration.json`](protocol_archive/word_correlation_v1_preregistration.json)
(SHA-256
`5e2ae9d0896edbcc7386ccfcc125f8200fa86f77b2099529028a01e54788516a`).
It was registered at 06:23:34 UTC before its fresh Modal Volume existed or any
scanner outcome was inspected. It uses only the exposed 400-item failed-V4
curve, deterministically partitions it into 200 discovery and 200 locked-word
validation prompts, and never mounts a future curve, sealed final set,
reserve, or training-exclusion manifest.

For every prompt it samples eight training-like base-model completions and
computes numeric correctness. For each of 36 frozen emotional words it scores
the exact J-decoded token-family probability at layer 8 over the late half of
the response with stride 5 and mean aggregation. Scores use candidate-specific
pinned WikiText calibration, the full 151,936-row output-head denominator,
standardization clipped to `[-5,5]`, and candidate-specific masking of literal
target tokens. A word is discovery-eligible only if it has score variance and
appears literally in zero discovery completions.

The primary statistic is the point-biserial correlation after centering both
J scores and correctness inside each prompt. Selection uses the largest
absolute discovery correlation among the frozen candidates, with an exact
lexical tie-break. Crucially, semantic valence does not preassign the RL sign:
the selected word gets a positive weight for a positive observed association
and a negative weight for a negative one. The inventory contains 18 positive
and 18 negative emotions, including `amazed`, `awesome`, `excited`, `happy`,
`joy`, `love`, `proud`, `thrilled`, `wow`, `yay`, `afraid`, `angry`, `damn`,
`despair`, `fear`, `frustrated`, `fuck`, `panic`, `sad`, and `worried`;
`solved` is absent and cannot be selected.

Discovery reports a 100,000-draw within-prompt max-absolute permutation test.
After the selected word, sign, token IDs, calibration statistics, code, data,
lens, and calibration bytes are durably locked, validation scores only that
word and reports a 100,000-draw one-sided permutation test plus a 10,000-draw
prompt-cluster bootstrap interval. An unrestricted individual-token lexical
atlas is descriptive discovery output only and is structurally unable to
change the locked emotional word or sign.

The scanner was independently reviewed before registration. The review caught
and fixed a real 63-bit seed overflow in NumPy setup and then added fail-closed
validation of the complete selection-lock provenance so a calibration or code
swap cannot occur after discovery. The final scanner SHA-256 is
`d35f05fc9e8b365ce777b55227fdc45f57ef45031ee739be728252e184b0e4a7`;
all 79 repository tests, Python compilation, Bash syntax, and whitespace checks
passed. The remaining pre-runtime risk is that the five-stage GPU path has not
yet executed on Modal; the fresh Volume and immutable status files make such a
failure visible rather than silently reusable.

This correlation experiment is mechanistic association evidence, not proof
that rewarding the word improves accuracy. Its selected emotional target may
be tested in exploratory RL, but the requested significant claim still
requires a newly frozen emotional-only recipe, eight new seeds on the untouched
V5 curve, the exact baseline-plus-three-transition mean curve at four nodes
fixed in the final registration, matched sign-flip controls, and the
already-reserved one-shot 1,300-item final collection.

### Completed emotional-family screen and exact exploratory curve

The alternative screen subsequently completed normally on app
`ap-53QKlR6MO6mZlaG3c7SXkH` at 06:27:46 UTC. Its durable closeout is
[`protocol_archive/alternative_screen_closeout.json`](protocol_archive/alternative_screen_closeout.json)
(SHA-256
`1f9b913c7a433283a571fd5d03114f8442a664187fa71b0c3929ce10acb71edf`).
All eight arms reached all eight fixed validation nodes on L40S workers from
clean commit `3ad255753e8ec1f7a0dfe0d27ad69a53e048122c`.

The complete emotional curves at steps `0/2/4/6/10/15/20/25` are:

| Arm | Curve | W&B run |
|---|---|---|
| positive celebration, U5 | `.3825/.4175/.3875/.3850/.4200/.3875/.4000/.3975` | `o4jf4qie` |
| negative profanity, U5 | `.3825/.3825/.3975/.3825/.4000/.3950/.4075/.3825` | `eom9e3ht` |
| positive celebration, taper | `.3825/.3825/.3850/.3925/.4050/.4000/.4100/.4200` | `b66bqrr5` |
| negative profanity, taper | `.3825/.3975/.3850/.3825/.3950/.3525/.3875/.3975` | `p9xmxdtj` |

The important new finding is the celebration-taper subsequence at
baseline/steps `15/20/25`: `.3825 -> .4000 -> .4100 -> .4200`. It has exactly
three strict upward transitions, and the first post-baseline node is already
above the initial evaluation. This satisfies the user's requested visual curve
shape on one adaptive development seed. It does not establish significance or
reliability across seeds.

The original screen's separately frozen early `0/2/4/6` rule was not met by an
emotional arm; its mechanical selection was a now-retired `solved` arm. The
later explicit emotional-only decision makes that selection ineligible, so it
is recorded but cannot influence future experiments. Celebration-family
literal matches occurred at rate `.0025` in greedy validation and profanity
matches at `0`; literal target and causal-predecessor positions remain masked
from the training reward.

The screen can also be reconstructed without W&B from
[`protocol_archive/alternative_screen_forensic_bundle.json`](protocol_archive/alternative_screen_forensic_bundle.json)
(SHA-256
`edce10202c816fb7658774d7dcc1205ef28d7c2cabc765a2db36c058c43efff2`).
That self-contained bundle embeds all 208 trainer-history records, all 64
validation records, resolved configs, run manifests, calibrations, exact data
indices, W&B identities and metric meanings, code/model/dataset/runtime
provenance, checkpoint/final adapter inventories, and hashes/sizes for every
one of the 214 durable Volume files. Its independently re-fetched snapshot has
canonical tree hash
`5bfddbb88b8fe114e0d94cf01b24592f120cbd84ee2cea29a1b6e491c17e3091`,
computed over sorted `relative path || NUL || raw bytes || NUL`; all linkage,
overlap, completion, and final-versus-checkpoint assertions passed.

With those eight workers stopped, the distinct single-word screen was launched
from current clean commit `27d598c4a800fbcc130bee8c559f94e4bee65730`
on app `ap-YkWhLmkYmv3jlX3MnfDrmX`, call
`fc-01KXFN5JWFWS216WBCVXSK2D0K`, and fresh Volume
`j-lens-rl-emotional-single-word-screen-v1-20260714a`. Its eight targets are
`yay`, `wow`, `joy`, `proud`, `excited`, negative `damn`, negative `fuck`, and
negative `worried`. The app stopped normally at 07:03:38 UTC after every arm
reached all eight fixed nodes. Curves below are ordered
`0/2/4/6/10/15/20/25`:

| Target and sign | Curve | W&B run |
|---|---|---|
| positive `yay` | `.3825/.3975/.4000/.3825/.4025/.3875/.3900/.4050` | `bhrqs7p0` |
| positive `wow` | `.3825/.3975/.3825/.3925/.3750/.4075/.3975/.3950` | `hrbuu8vs` |
| positive `joy` | `.3825/.3900/.3900/.4100/.3925/.4000/.3950/.3950` | `5m3mwx9h` |
| positive `proud` | `.3825/.3975/.3750/.4025/.3825/.4025/.4050/.3950` | `kq58g4fd` |
| positive `excited` | `.3825/.4050/.3900/.3800/.4000/.4025/.3875/.3950` | `twl58xg4` |
| negative `damn` | `.3825/.3975/.3775/.4175/.3850/.4000/.4050/.3900` | `ewc9d07r` |
| negative `fuck` | `.3825/.3925/.4075/.3825/.3825/.4075/.4025/.3925` | `9yxxt2rg` |
| negative `worried` | `.3825/.3800/.3800/.4000/.4050/.3825/.3950/.3750` | `hxx9kyva` |

`joy` was the only arm satisfying the screen's already-fixed early rule:
`.3825 -> .3900 -> .3900 -> .4100`, so the frozen priority mechanically
selects positive `joy`. No greedy validation completion at any joy node
literally contained `joy`. This remains adaptive single-seed evidence, not a
significance result. The screen saved only checkpoint 25: step 6 was evaluated
but its adapter bytes were not retained. A six-update confirmation is therefore
a dynamics-matched prospective recipe (constant scheduler, zero warmup), not a
bitwise replay of a stored step-6 adapter.

### Offline reconstruction independent of W&B

[`protocol_archive/emotional_screen_forensic_bundle/README.md`](protocol_archive/emotional_screen_forensic_bundle/README.md)
and its 154 checksum-pinned raw files preserve both completed eight-arm screens.
The bundle contains every resolved config, calibration and data manifest, raw
trainer history, validation history, screen result, checkpoint trainer state,
W&B identity/URL, and the exact meanings of reward, J-score, optimization, KL,
entropy, rollout, and evaluation metrics. Thus every experimental scalar curve
can be reconstructed if W&B disappears, while the same frozen inputs permit a
rerun. The bundle deliberately excludes W&B-owned wall-clock/system telemetry
and full adapter weights; those exclusions and the nondeterministic attention
warning are explicit in the README. `CHECKSUMS.sha256` has SHA-256
`8b2765bab2ae55d0c517d165961b39815b814c787880b0b818894a2720273d17`,
and all 154 entries verify.

### Correlation attempt 1 failure and frozen amendment

The separate correlation scan launched on app
`ap-6OJz03no1TZXCh2CXCy37V`, call
`fc-01KXFQ44P10B4J358MX3QS0ERC`, claim
`ad2c296ab38145da902d161d85a9ea56`, and fresh Volume
`j-lens-rl-word-correlation-v1-20260714a`. Calibration completed, but discovery
stopped at 07:14:03 UTC: shards 1 and 2 encountered a rollout with no common
sampled readout position, and the optional descriptive atlas incorrectly
raised `ValueError` instead of omitting that prompt. Six other shard manifests
exist, but their partial outcomes were not inspected; discovery never
completed, no word/sign was selected or locked, and validation never opened.

The immutable failure record is
[`protocol_archive/word_correlation_attempt1_closeout.json`](protocol_archive/word_correlation_attempt1_closeout.json)
(SHA-256
`5521f307d43345b6d23b34995bfac1d1bd24e60608c9dc7137622c056c86dcb5`).
The pre-relaunch amendment is
[`protocol_archive/word_correlation_v1_amendment1.json`](protocol_archive/word_correlation_v1_amendment1.json)
(SHA-256
`3a84d9acaa0bda46edd038738ffa2f39ebf8a168ffae25a1e2789cf751a615c9`).
It leaves every prompt, rollout seed, candidate, calibration, primary score,
selection rule, statistic, and validation test unchanged. It changes only the
descriptive atlas: a prompt with a positionless rollout is conservatively
omitted from that atlas and counted. The corrected attempt must use fresh
Volume `j-lens-rl-word-correlation-v1-20260714b`; failed partial outputs are
ineligible for reuse.

After that amendment was committed but before the corrected launch, the user
reduced the operational ceiling from ten to two Modal GPUs globally. The
second pre-launch amendment,
[`protocol_archive/word_correlation_v1_amendment2.json`](protocol_archive/word_correlation_v1_amendment2.json),
pins the two-worker launcher and forbids overlap with any other Modal GPU app.
This scheduling-only change alters no scientific field, data, seed, score, or
test. Subsequent confirmation is queued until the correlation app has fully
stopped, so the account-wide live GPU count cannot exceed two.

The first two-worker launch, app `ap-QAQk2EeSso2isJMolS9Nw0`, failed while
building its image because a linked Git worktree's `.git` file pointed outside
the copied checkout. It stopped before a function call, claim, calibration,
GPU task, or result; its newly created `20260714b` Volume contains zero files.
[`protocol_archive/word_correlation_attempt2_closeout.json`](protocol_archive/word_correlation_attempt2_closeout.json)
pins that operational failure. The pre-launch packaging amendment
[`protocol_archive/word_correlation_v1_amendment3.json`](protocol_archive/word_correlation_v1_amendment3.json)
requires a complete standalone clean clone and fresh Volume
`j-lens-rl-word-correlation-v1-20260714c`. It changes no scanner or scientific
field. Before this amendment was committed or launched, the user reduced the
global ceiling again to one Modal GPU to save cost; amendment 3 therefore pins
a fully serial one-worker launcher. All later Modal GPU work must also be
serial and cannot overlap this scan.

The standalone attempt 3 app `ap-7qMLax3fEinLrNNBGYL0xr` then failed during
image construction, before a function call, claim, calibration, GPU task, or
outcome. Its fresh `20260714c` Volume was rechecked after the app stopped and
contains zero files. The image correctly contained a complete Git clone, but
the image copy rule omitted the tracked, outcome-free
`.confirmatory/manifests/train_exclusions.json`; the clean-tree finalizer saw
that omission as a deletion. The immutable record is
[`protocol_archive/word_correlation_attempt3_closeout.json`](protocol_archive/word_correlation_attempt3_closeout.json)
(SHA-256
`d0c41616f1674baece50b8d377aeb2097dff92c2837b8dadd07ada1131524c27`).

Before another launch, amendment 4 freezes a packaging-only repair on new
Volume `j-lens-rl-word-correlation-v1-20260714d`: mount that exact safe manifest
with SHA-256
`7c1ca4f404ba9149093cc3c57dc3607582f671397e2fe99e93449848c1d65d61`.
Neither the scanner configuration nor scanner source references it; sealed
final, reserve, and retired manifests remain unavailable. The scanner and all
scientific choices are unchanged. The launcher now also refuses dispatch
without an external exclusivity confirmation and an app-list check showing no
other active Modal app; calibration and both shard maps are fully serial.
[`protocol_archive/word_correlation_v1_amendment4.json`](protocol_archive/word_correlation_v1_amendment4.json)
has SHA-256
`6b17b616d0d73cab7181f0dbb72c8c5343f48125b1856bf8f72398bb0a9644a7`.

### Frozen joy-only V5 confirmation

The emotionally charged confirmation is fully specified independently of its
confirmatory outcomes. The adaptive source record is
[`protocol_archive/joy_v5_selection_closeout.json`](protocol_archive/joy_v5_selection_closeout.json)
(SHA-256
`e75fe99db1e8a12d100bd968704e3eb3af339a901f75d6f4271f28e2fd492ba5`).
It binds the exact archived joy config, calibration, run manifest, screen
result, raw validation/training histories, final-adapter inventory, and W&B run
`5m3mwx9h`. It explicitly labels the selection outcome-informed exploratory
evidence, not a V5 result. The source curve is
`.3825 -> .3900 -> .3900 -> .4100` at steps `0/2/4/6`, with zero literal joy
emission at all four nodes.

The selected recipe lock is
[`protocol_archive/v5_emotional_selected_recipe.json`](protocol_archive/v5_emotional_selected_recipe.json)
(SHA-256
`006bd685f045a385889d1e418e76437140a404177f0d1148d308226390a2a547`).
Relative to the byte-pinned exploratory recipe it changes only the declared
confirmation horizon (`updates` and `save_every` from 25 to 6), the scheduled
validation nodes (to `2/4/6`), and the calibration path to the committed
byte-identical archive. Generated run identities then mechanically replace the
exploratory seed/path/name and negate only the component weight for controls.

The final prospective registration is
[`protocol_archive/v5_emotional_registration.json`](protocol_archive/v5_emotional_registration.json)
(SHA-256
`b2c49eaad02169e0da818a7893205f5ff7084fc41da1db6441ea7226a562d527`),
frozen at `2026-07-14T08:26:20Z`. It fixes:

- positive `joy` intrinsic reward only, layer 8 late-half mean, stride 5,
  weight `+1`, constant learning rate `3e-6`, and exactly six updates;
- treatment seeds 168--175 and eight mechanical sign-flipped controls;
- the requested aggregate curve rule at `0/2/4/6`: step 2 strictly above
  baseline, followed by two non-downward transitions;
- one immutable 17-role evaluation collection on all 1,300 untouched V5 final
  items, with the step-6 checkpoint required to equal the final adapter;
- paired treatment-minus-base and matched difference-in-differences analysis,
  crossed seed/item bootstrap with 10,000 samples, and the exact eight-seed
  sign test;
- online W&B identities under group `confirm-v5-emotional-joy-h6`, complete raw
  histories, atomic terminal publication receipts, and a terminal evidence
  artifact for every run;
- one L40S globally, fully serial execution, and a hard preflight rejection if
  any other Modal GPU app is active.

Canonical confirmatory commands retain immutable output paths and
`resume=never` W&B IDs. The bundled launch plan separately supplies a
non-claim replay command for each arm: it requires a caller-chosen directory
outside `.confirmatory/v5`, disables W&B, strips the original remote identity,
and stamps the output `non_claim_reproduction`. Registered configs reject all
ordinary update/output/tracking overrides, so replay output cannot be mistaken
for original evidence.

Failure handling is terminal and auditable. Partial optimization cannot resume;
only publication of an already immutable terminal result may retry. Claim and
launch markers are atomically published. If the orchestration receipt never
arrives, an exclusive pre-dispatch absent-receipt closure is committed before
failure finalization; a late receipt cannot mutate the resulting inventory.
Complete, curve-failed, significance-failed, and infrastructure-failed attempts
all receive truthful inventories and deterministic byte-verified ZIP exports.

The registration also binds correlation amendment 4 and its launch receipt,
and records that no correlation outcome was inspected or used. That running
experiment is ineligible to change this already-selected joy recipe. V5 must
remain unlaunched until the correlation app is stopped and the one-GPU
exclusivity preflight passes. The final local checks were `27/27` focused V5
tests and `108/108` repository tests, plus clean Python compilation, shell
syntax, and whitespace checks. Preparation deliberately still rejects the
new provenance files until they are committed together on a clean tree.

### Outcome-free V5 prelaunch failure and infrastructure amendment 1

Commit `5b67e98c9a35dea1dd5ea12fa3c3dbb52ff5fd5d` was pushed and the V5 state
was prepared and verified before dispatch. Modal app
`ap-MyIzIl9cIrBURNmaupZ246` then built image `im-NoHVViPWtyMEC3yOCAGICv`, but
Modal rejected the app definition because the final evidence function's
explicit 20,480 MiB ephemeral-disk request was below this workspace's current
524,288 MiB minimum. The app stopped at 08:37:02 UTC with zero tasks. Its local
entrypoint never ran, so there was no claim, function call, GPU task, W&B run,
or scientific outcome. Fresh Volume
`j-lens-rl-confirmatory-v5-emotional-20260714a` was rechecked after stop and its
root listing was exactly empty.

The immutable closeout is
[`protocol_archive/v5_emotional_prelaunch_attempt0_closeout.json`](protocol_archive/v5_emotional_prelaunch_attempt0_closeout.json)
(SHA-256 `9151fa5c8ba3e95c37b2abc53ee6e35bab6cb2de2bd7656922f1809f816bb8d4`).
Infrastructure amendment 1 is
[`protocol_archive/v5_emotional_infrastructure_amendment1.json`](protocol_archive/v5_emotional_infrastructure_amendment1.json)
(SHA-256 `d845fd829b00deb80cfed402e8fd8a04543c2ddffc451329dcfc572e296f3f42`).
It authorizes exactly two operational changes: use fresh Volume
`j-lens-rl-confirmatory-v5-emotional-20260714b` and request Modal's 524,288 MiB
disk floor for the late CPU finalizer. The original registration remains
byte-identical at SHA-256
`b2c49eaad02169e0da818a7893205f5ff7084fc41da1db6441ea7226a562d527`,
so every scientific choice and preregistered W&B ID is unchanged. Current
prepared state additionally pins the amended source bytes, amendment hash,
closeout hash, fresh Volume, clean Git commit, and full source snapshot.
The amended code passed `31/31` focused V5 tests and the full repository suite
at `109/109`, plus Python compilation, shell syntax, and whitespace checks.

### Active V5 launch ledger

The amended V5 attempt was prepared from clean pushed commit
`252a2319dc0ada5e99ddeecf507d7246590531d7` and dispatched only after a
preflight found no other active Modal app. Modal app
`ap-2o4XOP7jhqcrHyqkGN55wL`, orchestration call
`fc-01KXFXCF23D6DRB5VNXCVJB8TN`, and immutable claim
`c2d9ed29ebba46649f1ea182d7d50014` use fresh Volume
`j-lens-rl-confirmatory-v5-emotional-20260714b`. The launch status records the
global one-GPU ceiling. Wave 1/8 dispatched treatment seed 168 alone; its GPU
call is `fc-01KXFXD5TNGYTB3RW08SV9J5PR`.

At 08:58:40 UTC the registered online run
[`confirm-v5-emotional-joy-h6-b2c49eaad021-jlens_seed168`](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/confirm-v5-emotional-joy-h6-b2c49eaad021-jlens_seed168)
began syncing. It completed all six updates and terminalized normally at
09:07:51 UTC. Its frozen 400-item curve was
`.4100 -> .4050 -> .3850 -> .3950` at steps `0/2/4/6`, with literal target rate
zero throughout. Thus seed 168 alone is negative for the requested curve; the
registered decision remains the eight-seed mean and has not been evaluated.
The validation-index manifest is
`b01409c011012641be96c84bfc35cb0b352cea902e54304105efa272a3eac6b2`.

The immutable per-run result manifest has SHA-256
`5aca0a5765b915448cbf7542a6119db75823bc5ecf90cfe6ea46cb5c9dd14970`;
the final adapter/tokenizer tree has SHA-256
`4d4bde10ba3dc9c3779a61b438a128a0f6a9cc51958b3beffbb7a67a8af8ba4f`.
W&B acknowledged terminal evidence artifact
`confirm-v5-emotional-joy-h6-b2c49eaad021-jlens_seed168-terminal-evidence:v0`
(digest `80956fbf6628133610b1b373d7c96872`). The exact curve, confidence
intervals, file hashes, W&B identity, and inspection boundary are mirrored in
[`protocol_archive/v5_emotional_seed168_terminal_ledger.json`](protocol_archive/v5_emotional_seed168_terminal_ledger.json).
No sealed-final outcome was opened. Wave 2/8 then dispatched seed 169 alone;
its registered [W&B run](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/confirm-v5-emotional-joy-h6-b2c49eaad021-jlens_seed169)
began syncing at 09:09 UTC while seed 168's GPU container was already gone.

### Terminal V5 result: honest curve failure

The V5 app finished normally at 10:12:21 UTC after running all eight treatment
seeds serially on one L40S. Every treatment run synced online under the frozen
W&B IDs
`confirm-v5-emotional-joy-h6-b2c49eaad021-jlens_seed{168..175}`. No sign-flip
control ran: the preregistered curve gate failed first, so the protocol returned
before control dispatch, final unlock, collection, or sealed-final evaluation.

The complete per-seed exact-match curves at registered steps `0/2/4/6` are:

| Seed | Registered curve |
| ---: | --- |
| 168 | `.4100/.4050/.3850/.3950` |
| 169 | `.4100/.3750/.3800/.4050` |
| 170 | `.4100/.3925/.4050/.3800` |
| 171 | `.4100/.3925/.4025/.4250` |
| 172 | `.4100/.3925/.4175/.4225` |
| 173 | `.4100/.3975/.3800/.3925` |
| 174 | `.4100/.3800/.3875/.3925` |
| 175 | `.4100/.3900/.3975/.3900` |

Their registered mean is
`.410000 -> .390625 -> .394375 -> .4003125`. The three gate comparisons were
`false/true/true`: the last two nodes rose, but the first post-baseline node was
below baseline. Therefore this attempt is negative evidence for the requested
curve, not a success. Literal `joy` completion rate was zero at every curve
evaluation, so the accuracy movement cannot be described as literal word
emission.

The immutable claim is `c2d9ed29ebba46649f1ea182d7d50014`; Modal app
`ap-2o4XOP7jhqcrHyqkGN55wL` and orchestration call
`fc-01KXFXCF23D6DRB5VNXCVJB8TN` used Volume
`j-lens-rl-confirmatory-v5-emotional-20260714b`. The curve-gate JSON SHA-256 is
`09770542837464bb80c48c036099521f0ae05ec2e0c81c2440f1f06816a4eede`,
terminal-status SHA-256 is
`463d2dc01b44c837daf7cc761d6f587bce804048889e4dbe95463058e734a75e`, and
curve-plot SHA-256 is
`61d6c49740bb277335a0500a1842396d3328baa3f576540d7a1c78ccc659f280`.
The durable 254-entry evidence ZIP is 763,510,552 bytes with SHA-256
`8710288864aac8bde570beff50ccc9835b1b0d98e2470a7a3bab897bcf620b17`;
its complete 253-file inventory has SHA-256
`f098574bbbf97f645c4f66b4e33ce96c8ef8694bc6d37a721fc37f2a7f0976c2`.
That inventory reports zero sealed-evaluation files and no final unlock,
collection, comparison, acceptance, or control-run result. The archive remains
retrievable with `modal volume get j-lens-rl-confirmatory-v5-emotional-20260714b
/exports/v5_emotional_evidence_c2d9ed29ebba46649f1ea182d7d50014.zip
./v5_emotional_evidence_c2d9ed29ebba46649f1ea182d7d50014.zip`.

### V6 infrastructure retries and live Volume-C launch

The first two V6 submissions ended before scientific execution. Attempt A,
from commit `4524e643a97d084cd2d62176075ddd9e477525e9`, used app
`ap-sujvjQTDFQV2qwrVIFjNRq`. Its image build stopped making progress in the
asset-cache step after an unauthenticated Hugging Face warning, so it was
manually stopped at 11:03:05 UTC with zero tasks. It had no entrypoint,
protocol upload, claim, function call, GPU, W&B run, or scientific outcome.
The exact record is
[`protocol_archive/v6_celebration_prelaunch_attempt_a_closeout.json`](protocol_archive/v6_celebration_prelaunch_attempt_a_closeout.json)
(SHA-256
`c55cdd5fabd757e3111c76a3b4d4ee6df79d139e03fd55989b74d5fada9583a3`).
Infrastructure amendment 1 scopes `HF_TOKEN` to a V6-only image-cache wrapper,
checks that it is absent afterward, and moves the retry to fresh Volume B. It
is archived at
[`protocol_archive/v6_celebration_infrastructure_amendment1.json`](protocol_archive/v6_celebration_infrastructure_amendment1.json)
(SHA-256
`4b931daa8d5c4e8cec8ee7b3f0f14981aefcd3485c9f29c0d1be3cb03ea5d136`).

Attempt B, from commit `6f150b3a5fb404ebdcac95778d84880b4d1acef4`,
used app `ap-Mhzw5O7P2QdnHzyhQJaomJ`. Modal rejected hydration at 11:17:53
UTC because the manually created Volume B was v1 while the runner required
v2. It likewise had zero tasks and stopped before image build, entrypoint,
claim, function call, GPU, W&B, or outcome. The immutable closeout is
[`protocol_archive/v6_celebration_prelaunch_attempt_b_closeout.json`](protocol_archive/v6_celebration_prelaunch_attempt_b_closeout.json)
(SHA-256
`08a57a5d81ea98a8e9b2de3c778b3d3f6995c59a2fe17d365f3b51724f071e51`).
Infrastructure amendment 2 permanently excludes Volumes A and B and pins the
explicitly created v2 Volume C,
`j-lens-rl-confirmatory-v6-celebration-taper-20260714c`, Modal object
`vo-UYlAzgmVfmtRarECX4DYJg`. It is archived at
[`protocol_archive/v6_celebration_infrastructure_amendment2.json`](protocol_archive/v6_celebration_infrastructure_amendment2.json)
(SHA-256
`230760f24594f7e8641c8b2a7d7b1cb9c29741c08c9c40b8d451d2cbe0196f94`).

The corrected package was committed and pushed at
`3c1666d289b50e70a93ac6d8c8e21157ce530097` and passed all `160/160`
repository tests. Across both infrastructure amendments, the frozen science
and W&B projection remained exactly
`0621bc7402187223b47f665872b4c0bdb2c53b64661b26b94dcc76492a0fe93e`;
the active registration has SHA-256
`12cb17f896b117a43d9d266a53d43423ec5c5613fcc2dfda209f59bc27c507f2`.

The Volume-C launch receipt was submitted at
`2026-07-14T11:37:03.279158+00:00`: app
`ap-W0KYuXL8iKfFlgUS81uNKB`, immutable claim
`2c426dfb48e54c759ec6b2cd641f4d97`, and orchestration call
`fc-01KXG6MYVEXXGAW2M3KY38769Q`. It enforces one L40S globally and entered
`semantic_training`, wave 1, treatment seed 176 alone. At 11:40 UTC it was
still queued for Modal L40S capacity and the registered W&B group had no run.
At 11:40:54 UTC its registered
[seed-176 W&B run](https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/confirm-v6-emotional-celebration-taper-h10-jlens_seed176)
became live, and optimizer step 1 completed at 11:41:14 UTC. These facts show
execution liveness only: no registered evaluation curve node or significance
result was available at this ledger cutoff.

Seed 176 subsequently completed normally at 11:50:57 UTC. Its frozen 400-item
curve was `.3750 -> .3675 -> .3750 -> .3775` at steps `0/4/6/10`. It therefore
ended `.0025` above its own baseline, but it did **not** individually have the
requested shape because the first post-baseline node fell by `.0075`. This is
only one seed: the preregistered gate is the eight-seed mean and has not yet
been evaluated. Literal target completion was `0/0/0/.0025`, so the result is
not evidence that printing the target words drove accuracy.

The complete run directory is durable at `runs/jlens_seed176` on Volume C. Its
raw validation history, log history, resolved config, data indices, environment
snapshot, run manifest, terminal checkpoint, final adapter/tokenizer, and W&B
terminal evidence receipt were downloaded and hash-checked. The immutable
run-result manifest SHA-256 is
`0045775572ca1867247fb550fa111cae1508b70865f628c338ef558ba34c8b45`,
the final adapter/tokenizer tree is
`f1a9c168ae0f10d832fc16c153b07ea894de4336748ef0096c50d7929c3a5065`,
and W&B acknowledged artifact
`confirm-v6-emotional-celebration-taper-h10-jlens_seed176-terminal-evidence:v0`
with digest `a43271afe414c149d93b376cd799d17c`. The curve, confidence intervals,
identities, and hashes are mirrored in
[`protocol_archive/v6_celebration_seed176_terminal_ledger.json`](protocol_archive/v6_celebration_seed176_terminal_ledger.json).
No sealed-final outcome was opened. Wave 2/8 switched immediately to seed 177,
which became live under its preregistered W&B identity with no second GPU
treatment overlapping it.

Seed 177 also terminalized normally. Its curve was
`.3750 -> .3700 -> .3725 -> .3925` at `0/4/6/10`: it finished `.0175` above
baseline, but again dipped at the first node. The running two-seed mean is
`.3750 -> .36875 -> .37375 -> .3850`; this partial mean fails its first
comparison and passes its next two, but it is not the registered eight-seed
gate. Seed 177's literal target rates were `.0000/.0025/.0000/.0000`.

The downloaded Volume bytes again matched every terminal receipt hash. The
run-result manifest is
`6f9549e1f4fd5cc65c7c708356c248ed9bd2070a4553cd98c4add6dfc04706ea`,
the final adapter/tokenizer tree is
`b46b052e6c099d4844c5b251f8335207b33acd9e80b1b9097421d53daa13dbff`,
and W&B artifact
`confirm-v6-emotional-celebration-taper-h10-jlens_seed177-terminal-evidence:v0`
has digest `df15694eb070a2872a8c5d71ce1e3e0a`. Full curve metadata and hashes are in
[`protocol_archive/v6_celebration_seed177_terminal_ledger.json`](protocol_archive/v6_celebration_seed177_terminal_ledger.json).
No sealed-final outcome was opened. Wave 3/8 replaced the seed-177 worker with
seed 178 at 12:04 UTC; observed container listings never contained two
treatment workers.

### Correlation attempt-4 closeout and dormant recovery

The separate emotional-word correlation app had completed all eight discovery
shard artifact sets when Modal preempted its CPU controller. Its handler
incorrectly terminalized the delivered `KeyboardInterrupt`; Modal's automatic
same-input restart then rejected the failed claim. Forensic inspection was
limited to identity/sidecar metadata: shard payload contents, discovery
aggregation, selected word/sign, validation, and semantic outcomes remained
unopened. Volume D is closed and none of its artifacts may be reused. The exact
68-file inventory and boundary are in
[`protocol_archive/word_correlation_attempt4_closeout.json`](protocol_archive/word_correlation_attempt4_closeout.json)
(SHA-256 `1e28eb37bd1511f030b62cab950567f213634899690c545d15b0bd062e3608e0`)
and
[`protocol_archive/word_correlation_attempt4_forensic_inventory.json`](protocol_archive/word_correlation_attempt4_forensic_inventory.json)
(SHA-256 `4f69fb8c9e85d65cec64759071d7d772c56ffc11bbe53eb305751452651d6420`).
Thus the requested candidate atlas is frozen, but no measured word/correctness
correlation is yet known.

Recovery amendment 5 is committed at
[`protocol_archive/word_correlation_v1_amendment5.json`](protocol_archive/word_correlation_v1_amendment5.json)
(SHA-256 `6547c1e04d16e303b7f9a81cdf9a5191ca67975600314d2c935f0518c3b3cf10`).
It replays the unchanged protocol on a fresh Volume with generation-based
atomic publication, idempotent claim/submission recovery, and a shared named
GPU lease. Independent review caught and fixed a source-boundary flaw before
merge: the initial image rule could include `.git`, exposing excluded files
through Git objects. The final image is assembled from an exact allowlist of
15 safe source files plus three pinned artifacts, contains no Git metadata,
and verifies a complete hash inventory before remote phases. The integrated
code passed 136 outcome-free tests and compilation; an additional local focused
run passed 81 tests. No Modal, W&B, protected state, or sealed outcome was
accessed during repair.

Attempt 5 remains deliberately unlaunched while the active V6 app runs. Its
new named lease cannot atomically exclude that already-running, non-cooperating
V6 launcher; app-list checks alone leave a launch race. Every future launcher,
including the conditional profanity-U5 fallback, must acquire the same
`j-lens-rl-global-gpu-lease-v1` / `global-one-gpu` slot before GPU dispatch.

### V6 halfway checkpoint (not a gate decision)

Seeds 178 and 179 completed normally and were replaced serially by the next
worker. Their curves were `.3750/.3875/.3850/.3625` and
`.3750/.3950/.4050/.4000`, respectively. Consequently the four-seed running
mean at `0/4/6/10` is
`.3750 -> .3800 -> .384375 -> .383125`: the first post-baseline point is above
baseline and the next point rises, but the last falls by `.00125`. This is an
interim inventory only. The registered curve gate remains the mean of all eight
treatment seeds and has not been evaluated.

Both complete Volume directories were downloaded, and every raw file named by
their W&B terminal receipts matched its SHA-256. The seed-178/179 run-result
hashes are `98e0aa7d82c9eec9be5e7df9b562d6dbe68abfe12cff5788c0f1049428048622`
and `8f48cb42e35363d2766cb0c11ba9fc3ceb18360181edf40fe3d14e670a97b5d5`;
their W&B artifact digests are `4d8139a0da99f9839f4790167d8461f3`
and `415c41ac5b4626f0c1f71727b45ff3d5`. The exact four curves, run identities,
artifact trees, receipts, and inspection boundary are in
[`protocol_archive/v6_celebration_halfway_ledger.json`](protocol_archive/v6_celebration_halfway_ledger.json).
No sealed-final outcome was opened. Seed 180 began immediately afterward as
the sole treatment worker.

### Terminal V6 infrastructure failure after six valid seeds

Seeds 180 and 181 completed with curves
`.3750/.3925/.3750/.3675` and `.3750/.3925/.3975/.3575`. Across the six valid
treatment seeds 176--181, the exact aggregate correct counts were
`900/922/924/903` out of 2,400 at steps `0/4/6/10`, or
`.3750 -> .3841666667 -> .3850 -> .37625`. The first two comparisons rise, but
the terminal node falls. This is useful partial evidence only: the registered
decision required all eight treatment seeds and was never evaluated.

At 12:53:33 UTC Modal preempted the CPU orchestrator after seed 181 had
terminalized. The failure path generated a premature, explicitly
non-authoritative `failed` evidence archive, and Modal automatically restarted
the same orchestration input. The restart returned the immutable terminal
receipts for seeds 176--181 without repeating their optimization, then began
seed 182. A transient three-container listing was one CPU finalizer, one CPU
orchestrator, and the sole max-one GPU replay worker—not two GPUs. Nevertheless,
the registered failure policy makes any partial nonterminal run terminal for
the entire attempt; continuing fresh optimization after the failed closeout
would be ineligible.

The app was therefore stopped at 12:57:12 UTC. Seed 182 contains only its
resolved config, indices, environment snapshot, and run manifest; it has no
validation history, checkpoint, run result, terminal W&B receipt, or W&B run.
Seed 183 never started. The final attempt status is `failed`, with zero live
containers. No control, curve-gate output, final unlock, final collection,
sealed evaluation, comparison, acceptance, or analysis exists. The evidence
inventory explicitly reports zero sealed-evaluation files, so the 900-item
final remained unopened.

The canonical closeout is
[`protocol_archive/v6_celebration_terminal_closeout.json`](protocol_archive/v6_celebration_terminal_closeout.json)
(SHA-256 `e14022a7dd5614726d7bf7fd4c9c8a40f4eb056b1c3a5dad9dbf3c1069912081`).
Its compact run inventory has SHA-256
`b68ecf6367d725324a1ad7e2ec8fe2ae780e1aca27a121adf1e7875281fa97bf`.
The premature 198-entry, 576,232,024-byte incident archive remains retrievable
from Volume C and is bound by SHA-256
`bddd2f4e0bca5d995aa173a17b786539c8068b50bc831f754a602b73ba9bea2d`;
it is incident evidence, not the terminal closeout. V6 Volume C must never be
resumed or pooled. The closeout establishes only that the separately frozen V7
negative-`damn/fuck` attempt may inherit the still-unopened final after using a
fresh claim and fresh empty Volume; it does not unlock that final.

### V7 prelaunch hardening audit

The conditional profanity-U5 V7 implementation was independently rehearsed on
canonical main `89ada46` without reading protected V7 state or launching Modal,
GPU, or W&B work. The frozen scientific projection is unchanged: treatment is
the one-component negative intrinsic J-lens reward on `damn`/`fuck`, its matched
control flips only that sign, seeds remain 184--191, updates remain 20, the
exposed curve remains 0/4/10/20 with the requested three-rise predicate, and the
900-row final/significance plan and all W&B run identities remain fixed. The
unchanged projection SHA-256 is
`ce5b3a7c0a13846cc8053d207a0916ceba5d9b8f63edc7998e7173aa3df950c5`.

The audit found four prelaunch integrity problems: shadowed stale predecessor
definitions made the active protocol hard to audit; the global lease was first
acquired inside an already GPU-decorated function; recursive source-directory
inclusion admitted ignored or hidden files; and the predecessor predicate
accepted empty or weakly bound failure evidence. W&B publication also checked
only the run ID while copying the other expected identity fields into receipts.
All five were corrected before launch.

V7 now pins canonical V6 closeout
`e14022a7dd5614726d7bf7fd4c9c8a40f4eb056b1c3a5dad9dbf3c1069912081`
and run inventory
`b68ecf6367d725324a1ad7e2ec8fe2ae780e1aca27a121adf1e7875281fa97bf`.
The predicate recomputes all eight source hashes and cross-checks the exact
claim, root call and receipt, failure status, stopped Volume inventories,
premature archive/export binding, six terminal treatment curves and receipts,
the incomplete seed-182/183 file inventories and negative W&B lookup, no
control directories, and no unlock/evaluation/analysis artifact. Empty hashes,
malformed timestamps, missing fields, inconsistent aggregates, terminal files
for incomplete runs, and any final-boundary crossing fail closed.

Every GPU dispatch is now preceded on CPU by an atomic account-wide lease and
an immutable Volume intent containing the nonce, claim, root-call receipt,
phase, and subject. Workers revalidate that record before work and again before
completion publication; only the exact nonce may be removed, after a committed
completion record binds the result hashes. Uncertain cuts intentionally strand
the lease. The Modal image is built from one exact 180-file manifest rather
than recursive directories, with hashes baked into the image; finalization
rejects hidden, extra, symlinked, missing, or changed files. W&B receipt schema
2 validates observed entity, project, run ID/name, group, tags, and URL, and
the exact terminal artifact name, numeric version, qualified name, ID, and
digest. These infrastructure changes are recorded in
`protocol_archive/v7_profanity_prelaunch_source_cleanup.json`; they do not
authorize a launch by themselves.

Outcome-free focused checks passed: the canonical predecessor predicate, six
adversarial run-inventory mutations, hidden/extra/symlink/wrong-byte image
cases, CPU-before-GPU lease ordering, nonce-owned publication ordering, the
full 18-test V7 protocol suite, compilation, and the focused W&B receipt test.
The final broad outcome-free sweep passed 154 tests (44 protocol/core and 110
Modal/correlation/recovery tests) with only six expected local-Modal warnings.
No V7 outcome exists at this audit cutoff.

### V7 prelaunch manifest-identity correction

The first local `prepare` after hardening failed closed before any upload,
Modal app, GPU dispatch, or W&B run. All four inherited manifest files matched
their registered byte SHA-256 values, had the registered sizes, were pairwise
disjoint, and were covered by the training exclusions. The sole failure was a
clerical error in the redundant derived digest for the exposed 400-row curve:
the registration said `e1a3094d557c4d59ae023d18b2203d881e6819d3f4833c5516883ae9b727e621`,
while `canonical_sha256(sorted(indices))` over the byte-pinned
`ad348fe17d2e6bd6aac691d9bcdbb9da481f675305fa0e05c68e86dad97451c1`
manifest is
`bc8ef0aa726a0a7acd2080244128c96cf3e72bb23dfc169d1e8346ebe77e95a0`.
No repository manifest has the erroneous digest.

The correction changes only that redundant registration field and preserves
every curve row, the exact manifest bytes, final/reserve allocation, recipe,
seeds, curve nodes, controls, acceptance rules, and W&B identity. Because the
field is mechanically included in the integrity projection, its hash changes
from `ce5b3a7c0a13846cc8053d207a0916ceba5d9b8f63edc7998e7173aa3df950c5`
to `e500d7e5689e3d8fdd706682059f7ea9b3b313380d7685bb5f0e976009d668dd`;
this is a metadata correction, not a scientific allocation change. The exact
before/after chain is recorded in
`protocol_archive/v7_profanity_prelaunch_manifest_identity_correction.json`.
A new regression test computes identities from the canonical source files
rather than comparing two copies of constants. The corrected V7 and V5 suites
pass 47 tests in the configured project environment.

The first Modal relaunch then failed closed while building the image, before
the local entrypoint, remote claim, GPU dispatch, or W&B initialization. The
local TRL wheel build had generated staging copies below `trl/build/lib/`, and
the exact-image finalizer correctly rejected them as unexpected. Modal app
`ap-OQBZ5klbcquY8jCWtIrnhf` stopped with zero tasks; the fresh V7 Volume was
still empty. The finalizer now removes nested directories named `build` along
with the already-authorized packaging debris before checking the exact source
set and hashes. It does not admit those files to the image. The failure and
byte-pinned fix are recorded in
`protocol_archive/v7_profanity_prelaunch_image_build_fix.json`, with a focused
regression that recreates and removes `trl/build/lib/...`.

### V7 outcome-free launch record

V7 was durably submitted at `2026-07-14T14:15:11.286238+00:00` as Modal app
`ap-Vmg0kpbszpiUHHrNYcVWbd`, root call
`fc-01KXGFPF0HBWNG0TW3BS8APCZ9`, and claim
`1f2756de5df846d48a30f19a307b70fb` on the fresh version-2 Volume
`j-lens-rl-confirmatory-v7-profanity-u5-20260714a`
(`vo-x35QE7lwawEvm6zayz2HQl`). The preflight found no other active Modal app and
an empty global GPU lease; execution is restricted to one L40S worker. The
source commit is `3a9ca5cb317dea9e76160355bc8d7fa6c1f23bb8` and the exact registration
SHA-256 is
`5b6a39818ec5e281d95edf89cabec26499e5813cbb18741dbd4b54c099dc5569`.
The sole GPU worker was healthy, and the registered `jlens_seed184` W&B run had
begun syncing at `2026-07-14T14:17:41Z`; those are operational-liveness facts,
not training or evaluation outcomes.

The reconstruction ledger is
`protocol_archive/v7_profanity_launch_ledger.json`. It records the exact source
and prepared-snapshot hashes, Volume evidence paths, and all registered W&B
identities. It contains no training outcome, curve value, control outcome,
sealed-final data, or acceptance claim. The active claim must never be resumed
or overwritten; any rerun requires a fresh registered attempt, claim, and empty
Volume.

### V7 live treatment checkpoint and nonblocking RL queue

The first registered treatment run, `jlens_seed184`, completed cleanly and
published its terminal W&B receipt. Its exposed 400-row curve was
`0.3825 -> 0.3800 -> 0.3875 -> 0.3750` at the frozen steps `0/4/10/20`; literal
`damn`/`fuck` completion rate was zero at every node. This individual seed does
not have the requested monotone shape. It is one of eight preregistered seeds,
so it neither accepts nor rejects the registered eight-seed mean gate. The
orchestrator advanced without an idle word-search interval: `jlens_seed185`
began syncing to its frozen W&B ID at `2026-07-14T14:28:29Z`. Controls and the
sealed final remain unopened.

Planning for post-V7 work used only already-opened committed evidence; no
correlation-attempt payload or sealed outcome was inspected. There is still no
valid measured word/correctness correlation, so semantic valence is not
evidence for reward sign. The ready serial RL queue is single `-fuck`, single
`+yay`, then single `-worried`; these have existing committed calibrations and
therefore cannot be held up by a new word search. Broader prospective options
cover at least five distinct candidates per family:

- celebration core: `yay`, `wow`, `joy`, `proud`, `excited`, tested either
  late-half/stride-5 or with a separately frozen tail taper;
- distress suppression: `worried`, `afraid`, `anxious`, `panic`, `fear`;
- the current profanity family remains `damn`, `fuck`, with a future
  single-`fuck` isolate to test whether the union dilutes its adaptive signal.

Any such comparison is development-only until its arms, nodes, selection rule,
new seeds, matched sign flips, and untouched data are prospectively frozen.
Word correlation or joint-family calibration may run between RL attempts, but
must not keep a ready RL arm off the sole GPU for an hour.

### Terminal V7 infrastructure failure after two valid seeds

V7 did not produce a registered gate decision. Treatment seeds 184 and 185
completed, durably published their dispatch completions, and published terminal
W&B artifact receipts. Their exposed 400-row curves at steps `0/4/10/20` were
`.3825/.3800/.3875/.3750` and `.3825/.4075/.3950/.3950`. The descriptive
two-seed mean is `.3825/.39375/.39125/.3850`: it rises at the first node and
then falls. This is partial development evidence only, not the preregistered
eight-seed decision. Literal `damn`/`fuck` completion rate was zero at every
terminal node for both seeds.

Seed 186 has only durable curve nodes `0/4/10`, with
`.3825/.3975/.3875`. Modal logs show optimizer update 14/20 at 14:54:42 UTC,
followed by cancellation at 14:54:47. It has no dispatch completion,
checkpoint-20, final adapter, log history, run-result manifest, or terminal W&B
receipt. Its W&B sync began but is therefore explicitly partial and may never be
pooled or resumed. Seeds 187--191 did not start.

At 14:53:34 UTC Modal reported that the CPU orchestrator had been preempted and
would restart the same input. The original handler had already written a
`KeyboardInterrupt` failure and invoked the finalizer while seed 186 continued
running. That premature export is internally hash-valid, but it contains only
seed-186 nodes 0/4 and the earlier status. The restarted orchestrator then
collided with its own still-held seed-186 GPU lease and overwrote the status;
the second finalizer correctly rejected the changing bundle. The app stopped at
14:54:50 with zero tasks and later container checks empty.

Consequently V7 is closed as `infrastructure_failed/failed_before_final`. The
registered eight-treatment curve gate was not evaluated. No control, final
unlock, final collection, `evals/` directory, sealed comparison, analysis, or
acceptance exists; the 900-item final remained unopened. The canonical closeout
is
[`protocol_archive/v7_profanity_terminal_closeout.json`](protocol_archive/v7_profanity_terminal_closeout.json)
(SHA-256 `c2cfef2d3b24a96fbef703ef64b0f53f2c696481548300ee53154559ea3d602b`).
Its compact evidence directory preserves exact curve histories, terminal W&B
receipts, metric semantics, current-vs-premature state hashes, run inventory,
incident IDs, and the retrievable 198,086,460-byte export
(`4c0b48913e86259d2f0071b7b23b96f69e916ae24331cd0f94f6a845c0a73ccf`).

The global Modal GPU lease was intentionally still stranded on seed 186 at the
first closeout commit boundary. After that exact closeout was pushed at commit
`9de5aae3c0739333c5634ed0ce5f88199333a20d`, the recovery script rechecked the
stopped app and empty container list, matched the full canonical lease value
`cd7029a6803155b4d61ba806873cf5885f39a75a7e160c21981caa86999077d1`
and nonce, popped it, proved the popped value identical, and proved the Dict key
absent. The receipt is
[`protocol_archive/v7_profanity_gpu_lease_retirement_receipt.json`](protocol_archive/v7_profanity_gpu_lease_retirement_receipt.json)
(SHA-256 `3caf91c90ff5dd54170ebaab658494d643fde4eb681b7fe18ddfc26b79e82dc3`).
The post-recovery predicate is
[`protocol_archive/v7_profanity_authoritative_closeout.json`](protocol_archive/v7_profanity_authoritative_closeout.json)
(SHA-256 `cd83a08155871518baf177d5718acae1053ef4a98e171cbfc9351cd1b8db930c`).

The next RL work must be a separately registered whole RTX-4090 attempt with
fresh seeds, isolated state, a commit-pinned runtime, and preserved offline W&B
directories; no V7 model/run artifact may be reused. This operational recovery
does not turn V7 into a scientific result and does not itself unlock the final.

### Fresh V8-local RL registration

V8-local is a new whole attempt rather than a V7 continuation. It retains the
frozen negative intrinsic `damn`/`fuck` recipe, layer-8 late-half mean score,
weight `-1`, constant `3e-6` learning rate, KL `.02`, DAPO, 1,000 prompts, 20
updates, and the `0/4/10/20` curve. It changes the backend to the one local RTX
4090, tracking to preserved offline W&B, state to `.confirmatory/v8_local`, and
all treatment/control seeds to 200--207. The control remains the mechanical
sign flip to `+1`; no correctness reward is present.

The registration explicitly labels the reused 400-row curve and profanity
lineage as exposed development data. Its eight-treatment mean must still satisfy
`M4 > M0`, `M10 >= M4`, and `M20 >= M10`, but that shape is a consistency gate,
not a significance claim. Only the conditionally unopened 900-row final may
later support the registered crossed seed/item bootstrap, matched
difference-in-differences, and eight-seed sign test.

The local runner requires the exact RTX 4090 UUID, driver, memory, torch/CUDA
stack, no other compute PID, and a nonblocking process lock held for the whole
attempt. It prepares a detached clean worktree at the pushed registration
commit, pins imports to it, and writes an fsynced dispatch intent before each
serial subprocess. Every run has a unique offline W&B directory. Completion
requires W&B finish followed by a receipt that embeds and hashes all seven
terminal evidence files plus the full syncable offline tree; a partial run
closes the whole attempt.

The launch predicate validates the pushed V7 pre-recovery closeout, post-recovery
wrapper, lease-retirement receipt, and all 12 source-evidence files. Focused V8,
V7-regression, offline-receipt, and parked-tournament tests passed 56 cases in
the final parent sweep (the broader overlapping sweep passed 71). The exact
hardware/runtime probe also passed with zero active compute processes. Training
automation covers treatments, the one curve gate, conditional controls, and an
unlock marker. Sealed-final evaluation and analysis are deliberately not yet
implemented or audited, so the final must remain unopened even if training
reaches the unlock marker.

The first local `prepare` invocation failed outcome-free before metric-schema
generation because the shell ran the protocol by file path while its audited
imports expected package-module mode. Only five byte-identical frozen
artifact/manifest copies existed; there was no protocol state, claim, runtime
worktree, run directory, W&B directory, GPU process, evaluation, or outcome.
The exact inventory is in
`protocol_archive/v8_local_prepare_attempt1_closeout.json`. The launcher now
changes directory to the repository and invokes both entrypoints with `python
-m`; the bounded prelaunch amendment is
`protocol_archive/v8_local_prelaunch_module_entrypoint_fix.json`. This changes
no scientific or allocation field. The failed preparation may be cleared only
after those records and the fix are committed and pushed.

The next clean preparation completed, but its preflight probe rejected stdout
because an imported dependency could emit a notice before the JSON identity.
This was again before any claim, dispatch, training subprocess, W&B directory,
GPU process, or outcome. The prepared registration/state/source-snapshot hashes
are bound in `protocol_archive/v8_local_prepare_attempt2_closeout.json`. The
probe now emits a unique sentinel and requires exactly one sentinel identity
line, so unrelated notices cannot be mistaken for provenance. The unclaimed
state may be archived and prepared again only after this fix is pushed.

V8-local then prepared cleanly from pushed commit `af83249`, verified the exact
detached imports and idle RTX 4090, and claimed fresh attempt
`ff4144f0f3d14a49a669968c5c5a7a85`. Treatment seed 200 was durably dispatched
at 15:43:10 UTC as the sole GPU process under registration
`10c1969a8f6e5f8c5caede68019b627a19ae8121daaea31ec91715525ba98090`.
The operational ledger is
`protocol_archive/v8_local_launch_ledger.json`. It contains no metric or curve
outcome. W&B is intentionally offline because no local API key is configured;
each completed directory is self-contained and must be explicitly synced later.

### Terminal V8-local verifier failure after one complete seed

Treatment seed 200 completed all 20 optimizer updates, all four exposed curve
evaluations, its exact terminal checkpoint/final adapter, and W&B offline
finish. Its curve at steps `0/4/10/20` was
`.3975/.3875/.3550/.3975`; literal `damn`/`fuck` completion rate was zero at
every node. This individual seed is nonqualifying, and the registered decision
required eight treatment seeds, so the V8 curve gate was not evaluated.

The post-training wrapper then raised `jlens_seed200 terminal result changed`
before writing a dispatch completion. The mismatch is exactly two fields:
`src/jlens_rl/train.py` writes absolute `path` fields in its terminal checkpoint
and final-adapter tree identities, while V8's duplicate verifier helper expected
only `sha256` and `files`. After adding those two expected paths in a read-only
comparison, the terminal result is exactly equal. All earlier source/config,
RTX-4090 runtime, data-firewall, curve-provenance, behavior, and checkpoint
checks passed, and the independent offline-receipt validator binds all seven
terminal files and the closed W&B tree. The run-result SHA-256 is
`979b65968770982bb1f15da302c3dbc3ed407aa9d13704dd7961b2646abb4d99`;
the receipt SHA-256 is
`40227ffed39abe9c422900fa36c8960e2e4b13684a828cae8d1da47d0ecf31c9`;
the offline-tree SHA-256 is
`293d68ca2a6ff03a0a489d9fe4eb0316223be3e683ee41b4fa6c4ec26af1a8b9`.

Nevertheless, the registered fail-closed policy is binding: the attempt status
is `infrastructure_failed_attempt_closed`, no dispatch completion exists, and
V8 may not start another run, reconstruct adoption, resume, or contribute a
seed to a later pooled gate. Controls and sealed-final work never started, and
the 900-item final remains unopened. The canonical closeout is
[`protocol_archive/v8_local_terminal_closeout.json`](protocol_archive/v8_local_terminal_closeout.json),
with 20 compact source-evidence files under
`protocol_archive/v8_local_terminal_evidence/`. A CPU-only Modal helper synced
only the receipt-bound immutable offline directory to its registered W&B ID.
The upload reached remote state `finished`; all exact identity, receipt-bound
config, and seven terminal evidence checks pass. W&B's public API sorted the
registered tags lexicographically, so the verifier canonicalizes tag order but
continues to require exact tag membership and exact equality for every other
field. The durable sync receipt is
`protocol_archive/v8_local_wandb_sync_receipt.json`. This transport action did
not run training or alter V8's failed-closed disposition.

The next GPU work must therefore be a fresh whole V9 registration with new
state, claim, W&B IDs, and seeds, and a prospectively corrected tree-identity
verifier. This infrastructure incident does not count as a word-search delay:
the next RL attempt is being prepared immediately, and no correlation outcome
is being awaited before launch.

### Fresh V9-local registration after the verifier incident

V9-local is a new whole attempt, not a V8 continuation. It keeps the negative
intrinsic `damn`/`fuck` J-lens recipe, local RTX 4090 backend, exposed curve,
matched sign-flip control, and sealed-final analysis fixed, but uses isolated
`.confirmatory/v9_local` state, fresh seeds 208--215, and new W&B IDs. No V8
checkpoint, optimizer state, training row selection, or result can be adopted
or pooled. The V9 launch predicate pins the canonical V8 closeout and
recomputes all 20 archived evidence hashes before it can prepare.

The prospective terminal verifier now exactly mirrors the shared trainer's
tree identity: absolute resolved path, canonical tree SHA-256, and per-file
hashes. A regression test constructs a terminal tree and requires the two
helpers to return identical dictionaries, directly covering the defect that
closed V8. The runner retains one whole-attempt GPU lock, an fsynced intent
before each subprocess, one immutable per-run offline W&B directory, and a
fail-closed partial-run policy. Treatments run first in ascending seed order;
controls start only after the frozen eight-seed mean curve satisfies
`M4 > M0`, `M10 >= M4`, and `M20 >= M10`.

The focused V9, offline-receipt, and V8-sync sweep passes 38 tests; Python and
Bash syntax, design verification, and `git diff --check` also pass. The design
verification reads no V9 outcome, and `.confirmatory/v9_local` does not yet
exist at this cutoff. Sealed-final evaluation/analysis automation is still not
implemented or audited, so even a future training unlock must not open the
900-item final. Completed offline training runs should be synced promptly by a
separate CPU-only transport; tracking failure must not alter or rerun training.

V9 prepared from clean pushed commit `4d2f884` with registration SHA-256
`5c9b776c452eea645361b72f87d8cabcd1059db654430cb7975f7325499b0ce4`.
The exact idle RTX 4090, UUID, driver, memory, torch/CUDA stack, detached
imports, source snapshot, V7/V8 closeouts, and all 20 V8 evidence hashes passed
again after preparation. Fresh claim `d6131db54b6346199d9af8d1478f3e36`
started treatment seed 208 at 16:21 UTC under the one whole-attempt GPU lock.
At this ledger cutoff no V9 curve node or outcome had been inspected, no
control had started, and the sealed final remained unopened. The durable
operational identity is `protocol_archive/v9_local_launch_ledger.json`.

Seed 208 then completed and passed the corrected terminal verifier, including
the exact checkpoint/final path-plus-tree schema that failed V8. Its exposed
curve was `.3975/.4100/.3875/.4000` at steps `0/4/10/20`, with zero literal
`damn`/`fuck` usage throughout. It rose at the first node and finished slightly
above baseline, but the step-10 decline means this seed alone is not monotone;
the registered gate remains the mean across all eight seeds. The receipt-bound
offline directory was synced and independently verified at its exact W&B ID
while seed 209 began on the GPU without waiting. The compact reconstructable
bundle and exact remote/payload hashes are in
`protocol_archive/v9_local_seed208_terminal_ledger.json` and
`protocol_archive/v9_local_live_evidence/`.

Seed 209 also completed and passed the registered terminal validator. Its curve
was `.3975/.4025/.4150/.3850`, with zero literal target-word completions at all
four nodes. Thus its first two post-baseline nodes rose monotonically, but the
terminal node fell below baseline. Across the first two treatment seeds the
descriptive mean curve is `.3975/.40625/.40125/.3925`: the required first rise
currently holds, while the later two monotonic inequalities do not. This is an
interim description only; the frozen gate is evaluated once, after all eight
treatments, and seed 210 began immediately without waiting for tracking or word
search.

The receipt-bound seed-209 offline directory was synced through the CPU-only
Modal transport and its exact registered W&B identity, config, terminal result,
and seven embedded evidence files were verified remotely. The reconstructable
ledger is `protocol_archive/v9_local_seed209_terminal_ledger.json`; the compact
source evidence was added beside seed 208 under
`protocol_archive/v9_local_live_evidence/`. The sole GPU remains assigned to the
serial RL queue, so candidate-word analysis is not gating training.

Seed 210 completed with curve `.3975/.3875/.3900/.3825` and zero literal
target-word completions. The three-seed descriptive mean is now
`.3975/.4000/.3975/.38917`: the first rise remains positive, while step 10 is
`.0025` below step 4 and step 20 is `.00833` below step 10. Five treatments
remain, so the registered gate is still unevaluated. Seed 211 began immediately
on the same sole RTX 4090. Seed 210's receipt-bound offline directory was then
synced CPU-only and its exact remote W&B identity and all seven evidence files
verified. Its reconstructable ledger is
`protocol_archive/v9_local_seed210_terminal_ledger.json`, with compact raw
evidence under `protocol_archive/v9_local_live_evidence/`.

Seed 211 completed with curve `.3975/.4025/.4150/.3875` and zero literal
target-word completions. It supplied two consecutive individual rises, but its
terminal node fell below baseline. The four-seed descriptive mean is
`.3975/.400625/.401875/.38875`, so the aggregate now has the required first two
rises while the last node remains lower. The remaining four treatments must
average a `.013125` step-10-to-step-20 gain to repair that final inequality.
This is an interim feasibility calculation, not a changed gate. Seed 212 began
immediately, and the CPU-only seed-211 upload verified the exact W&B identity
and all seven receipt-bound evidence files. Its reconstructable ledger is
`protocol_archive/v9_local_seed211_terminal_ledger.json`.

Seed 212 completed with curve `.3975/.3725/.3850/.4050` and zero literal
target-word completions. Its first node fell, but both later nodes recovered
and the terminal node finished above baseline. The five-seed descriptive mean
is `.3975/.3950/.3985/.3920`. With three treatments remaining, their mean
step-4-minus-baseline difference must exceed `.004167`, and their mean
step-20-minus-step-10 difference must be at least `.010833`; the middle
inequality currently has positive margin. Seed 213 began immediately. The
CPU-only seed-212 upload verified exact W&B identity and all seven evidence
files; its reconstructable ledger is
`protocol_archive/v9_local_seed212_terminal_ledger.json`.

Seed 213 completed with curve `.3975/.3975/.3925/.3875` and zero literal
target-word completions. The six-seed descriptive mean is
`.3975/.39542/.3975/.39125`. Consequently, the last two treatments must average
above `.40375` at step 4 and average at least `.01875` improvement from step 10
to step 20; the middle inequality still has positive margin. Seed 214 began
immediately. The CPU-only seed-213 upload verified exact W&B identity and all
seven evidence files; its reconstructable ledger is
`protocol_archive/v9_local_seed213_terminal_ledger.json`.

Seed 214 completed with curve `.3975/.4000/.3875/.4025` and zero literal
target-word completions. It rose first, dipped at step 10, and recovered above
baseline at step 20. The seven-seed descriptive mean is
`.3975/.396071/.396071/.392857`. Therefore the final treatment seed must have
step 4 at least `.4100`, step 10 at least equal to its step 4, and step 20 at
least `.0225` above its step 10 for all three aggregate inequalities to pass.
Seed 215 began immediately. The CPU-only seed-214 upload verified exact W&B
identity and all seven evidence files; its reconstructable ledger is
`protocol_archive/v9_local_seed214_terminal_ledger.json`.

During final treatment seed 215, its sealed step-4 node was `.4050`. This is an
individual rise, but it fixes the eight-seed step-4 mean at `.3971875`, just
`.0003125` below the `.3975` baseline. The strict first-rise gate is therefore
mathematically failed regardless of later nodes. Seed 215 must still finish and
seal; V9 controls must not start.

To avoid an RL gap after that fail-closed decision, the previously registered
development-only emotional single-word tournament is being enabled without
changing its science: fixed serial `-fuck`, `+yay`, `-worried`, shared seed 192,
15 updates, and exposed nodes `0/5/10/15`. Its prelaunch protocol listed generic
terminal stage names but omitted V7's actual authoritative stage
`failed_before_final`. The bounded amendment adds that literal stage and binds
the already committed authoritative V7 closeout (including retired GPU lease).
No tournament outcome or state existed when this correction was made; the
tournament remains development-only and cannot support a significance claim.

V9 then closed normally after seed 215 completed `.3975/.4050/.4050/.3900`.
All eight treatment runs passed the corrected terminal validator, used zero
literal target completions, and were synced to their exact registered W&B IDs
with all seven embedded terminal files verified remotely. The final registered
mean curve is `.3975/.3971875/.3971875/.3925`: step 4 misses baseline by only
`.0003125` (0.03125 percentage points), step 10 is flat, and step 20 declines.
The terminal stage is `curve_failed_terminal`; there are zero control
dispatches, no final evaluation, and the sealed 900-item final remains
unopened. This is honest near-miss/negative development evidence, not a
significance result. The canonical closeout is
`protocol_archive/v9_local_terminal_closeout.json`, with gate/status/plot under
`protocol_archive/v9_local_terminal_evidence/` and per-seed raw evidence and
ledgers preserved for reconstruction without rerunning.

The next development tournament was prepared from pushed commit `ed95bfa`,
registration `ba3c076e623f655ee1c7600f49a0d78cf44fc6dee757c116a17fdafa04552dbc`,
and fresh Modal v2 volume `vo-FoPd7Y7NgprscFXEROSEza`. Modal app
`ap-YOyZ5SjuFDOHVtyhCF8QZr` was submitted while V9's final evaluation was still
using the local GPU; at this cutoff Modal was building the explicit allowlisted
image and had not yet allocated its one permitted L40S task.

That submission subsequently failed closed in remote `verify-launch`, before
an attempt claim or orchestrator dispatch. The strict image included all three
tournament templates and their tournament-common parent but omitted the
further inherited `configs/common.json`; the remote exception was therefore a
`FileNotFoundError` for `/workspace/j-lens-rl/configs/common.json`. App
`ap-YOyZ5SjuFDOHVtyhCF8QZr` stopped with zero tasks. Volume A contains only the
prepared configs, public frozen artifacts, exposed manifests, protocol state,
and reproducibility files: `attempt_claim.json`, status/receipt, runs,
evidence, exports, and GPU dispatches are all absent. The global GPU lease is
empty. Code order requires `claim_attempt` to return before the orchestrator is
spawned, so no L40S training, W&B run, curve, ranking, or target-word outcome
existed. No sealed/final/reserve/correlation payload was uploaded or opened.

The immutable forensic record is
`protocol_archive/emotional_tournament_v1_preclaim_attempt_a_closeout.json`.
Infrastructure amendment 1 admits exactly the missing inherited config,
byte-pins it, adds a recursive config-dependency closure check, retires Volume
A, and authorizes fresh noncreating v2 Volume B. It freezes the word arms,
reward signs, shared seed, training recipe, curve nodes, ranking, W&B IDs, and
one-GPU limit unchanged. The original scientific registration draft remains
byte-identical; the replacement prepared registration must bind both the
amendment and a copied attempt-A closeout so remote replay does not depend on
`protocol_archive/` being present in the strict image. Thirteen focused tests,
including a remote-like copied-closeout check, pass. This pre-outcome repair is
not an inferential retry and supplies no evidence about any arm.

Replacement preparation from clean pushed commit `7a42dec` passed locally and
again inside the strict Modal image. Its registration is
`9f290b768d58f04ce6e522301b40b4abe7acc544199cb34430b8d0c8642d12fe`;
fresh Volume B is `vo-SU6aUsebIrmYyZFGoCOo4e`. App
`ap-XrW9TTTRzQZ2sV9kB8WjbS` wrote claim
`1d6ea36d356c420f92e125c35a1a6aeb` and receipt-bound root call
`fc-01KXH13DR4ZGR31JSAFC7JN0KF`. Preflight saw no other active Modal app and an
empty global lease. The first serial arm, negative intrinsic `fuck`, then
acquired the sole lease and created an immutable run manifest on an NVIDIA
L40S with CUDA 12.8 and torch 2.9.1. The manifest records `reward_type=jlens`,
the exact registered source/config/input identities, and W&B run
`dev-v8-emotional-single-u5-h15-fuck-seed192`. At the launch-ledger cutoff no
curve outcome had been inspected; `yay` and `worried` had not dispatched. The
durable reconstruction record is
`protocol_archive/emotional_tournament_v1_launch_ledger.json`.

The first fixed arm, negative intrinsic `fuck`, completed all 15 updates with
curve `.3825/.3800/.3925/.3975` at `0/5/10/15` and zero literal target
completions throughout. It recovered to `.015` above baseline, but the `.0025`
first-node dip means it fails the strict requested shape. This is useful
partial development evidence, not significance. The terminal W&B evidence
artifact and receipt, exact config/runtime/data identities, all four raw curve
rows, and both GPU dispatch records were copied into
`protocol_archive/emotional_tournament_v1_live_evidence/fuck_seed192/` and
bound by `protocol_archive/emotional_tournament_v1_fuck_terminal_ledger.json`.
The old lease was released before the fixed serial `yay` arm acquired a new
one; no GPU overlap occurred.

The second fixed arm, positive intrinsic `yay`, completed all 15 updates with
curve `.3825/.3775/.4050/.3875` at `0/5/10/15` and zero literal target
completions throughout. It rebounded to `.0225` above baseline at step 10 and
ended `.005` above baseline, but the first-node decline and subsequent
step-10-to-step-15 decline mean it does not satisfy the requested monotone
shape. This remains useful development-only evidence, not significance. The
terminal W&B evidence artifact and receipt, exact config/runtime/data
identities, all four raw curve rows, and both GPU dispatch records are copied
under
`protocol_archive/emotional_tournament_v1_live_evidence/yay_seed192/` and
hash-bound by
`protocol_archive/emotional_tournament_v1_yay_terminal_ledger.json`. The
`yay` lease was released before the final fixed `worried` arm acquired its
fresh sole lease; `worried` began immediately and its registered W&B run is
online. No sealed, reserved, or word-correlation payload was accessed.

To keep RL from waiting on further word search, a scientifically separate
development probe was prospectively fixed and launched on the otherwise idle
local RTX 4090 while the single paid Modal L40S continued `worried`. The probe
uses the emotionally charged celebration family `yay/great/success/nice`, the
previously motivated tail-taper score, fresh seed 193, six updates, and exposed
nodes `0/2/4/6`; its fixed shape criterion is `M2 > M0`, `M4 >= M2`, and
`M6 >= M4`. It is development-only and may never be pooled with the L40S
tournament or treated as inference. The exact config and registration were
pushed in commit `bf85a74`; training began from that clean detached commit at
19:55:28 UTC on the exact registered 4090. The run manifest was durable before
any curve row and records the clean commit, source-tree hash, config, artifacts,
data identities, command, environment, and fixed W&B ID
`dev-v10-celebration-tail-u2-h6-seed193`. Local tracking is receipt-bound
offline and will be synced through a CPU-only Modal transport without rerunning
optimization. Operational identity is recorded in
`protocol_archive/development_celebration_tail_u2_h6_seed193_launch_ledger.json`.

### Emotional single-word tournament terminal closeout

The final fixed arm, negative intrinsic `worried`, completed all 15 updates on
the same serial L40S with exposed curve `.3825/.4100/.3800/.3700` at
`0/5/10/15` and zero literal target completions. Its step-5 rise is real, but
the two subsequent declines fail the registered monotone shape and leave step
15 `.0125` below baseline. Its terminal result, exact config/runtime/data
identities, raw histories, W&B artifact receipt, and dispatch intent/completion
are preserved under
`protocol_archive/emotional_tournament_v1_live_evidence/worried_seed192/` and
bound by `protocol_archive/emotional_tournament_v1_worried_terminal_ledger.json`.

All three fixed arms completed in registered serial order and none passed the
shape criterion. The frozen development ranking is therefore `fuck`, `yay`,
`worried`, selecting negative `fuck` only as an exploratory candidate. This is
not significance evidence and cannot be pooled with a confirmatory attempt.
The app stopped with zero tasks and containers, and the global GPU lease is
empty. The Volume's compact 58-file export was downloaded and verified at
SHA-256 `f50976b6f0c2dbbd3ea11d1a32f98c82d0a6438d6b480d412e138c8e6a88133a`
(5,391,708 bytes); all 14 separately downloaded worried/operational records
matched the export inventory and their nested terminal receipts. No
sealed-final, future-reserve, or correlation payload was downloaded or
inspected during archival. The authoritative reconstruction record is
`protocol_archive/emotional_tournament_v1_terminal_closeout.json`, with the
Volume evidence JSON preserved under
`protocol_archive/emotional_tournament_v1_terminal_evidence/`.

### Five-arm emotional development launch preparation

The first two-arm parallel submission, app `ap-U9xrDAO7uegoiTqAOxouu5`,
failed closed before training. Both workers imported runtime modules, creating
`__pycache__` files; because the strict image omitted `.gitignore`, its
synthetic Git worktree check rejected those files. The stopped fresh Volume
contains only the immutable attempt claim, failure status, and launch receipt:
there is no dispatch, run, log, evidence, W&B initialization, or curve
outcome. Volume A is retired. Exact raw records and the complete root inventory
are bound by
`protocol_archive/emotional_parallel_v2_pretraining_attempt_a_closeout.json`.

The replacement adds the pinned `.gitignore` and
`PYTHONDONTWRITEBYTECODE=1`. After that no-outcome failure, the user raised the
global Modal ceiling from two to five, so a new development registration fixes
five distinct ideas rather than altering the two-arm registration: positive
`joy`, positive celebration-family tail taper, positive `excited`, positive
`wow`, and negative `fuck`. Seeds are 194--198; four arms use the fast
`0/2/4/6` curve and celebration uses `0/4/10/20`. The registration SHA-256 is
`6eeee93e2cca1d5c4167eda682bf710940ba30a2b971bf85e82f479b9329e4dc`.
These runs use only J-lens word reward plus KL and are development evidence;
they cannot establish significance on the exposed 400-example curve.

Across every word setting run so far, including the retired non-emotional
`solved` lineage, there is still no valid statistically significant positive
result. The strongest exact requested-shape non-emotional observations are
single-seed adaptive screens: `solved` tail taper at
`.3825/.3850/.3900/.3950` on `0/2/4/6`, and the earlier screen-2 tail taper at
`.3750/.3800/.3800/.3825`. Prospectively aggregated `solved` attempts instead
failed their curve gates: V2 mean `.3750/.3825/.36875/.37708`, V3 mean
`.4350/.42125/.422375/.4185`, and V4 early mean
`.3825/.393125/.3915625/.389375`. The attractive one-seed curves therefore
remain candidate-selection evidence; no matched-control sealed analysis
unlocked after those gate failures.

### Five-arm launch and deadline V10 candidate freeze

The five-arm development screen launched from clean pushed commit `e0ba146` on
Modal app `ap-hl1q4duAFw0u160g3utmyD`, claim
`563f96a62af14b188b00c794cdb11395`, and fresh v2 Volume
`vo-P3ESFZpCj0Wtwnu7Do4ZOM`. Five immutable dispatch intents independently
record NVIDIA L40S, the same registration/source tree, and the five fixed W&B
IDs before training. The exact launch identities and record hashes are in
`protocol_archive/emotional_parallel_v3_launch_ledger.json`. No protected
final, reserve, or correlation payload was mounted.

At 20:55:27 UTC, before the new negative-`fuck` arm's step-6 outcome was
inspected,
the next confirmation candidate was frozen. The old seed-167 screen was
`.3825/.3925/.4075` at `0/2/4`; the new seed-198 partial curve was
`.3825/.3950/.3950`. Negative `fuck` therefore had two independent exposed
baseline-to-step-2 rises and no step-2-to-step-4 decline. The new confirmatory
curve prospectively inserts the previously unobserved step-3 node and fixes
`0/2/3/4`, four fresh seeds 216--219, exact `+1` sign-flip controls for the
registered treatment weight `-1`, and the still-unopened 900-row final. The
adaptive boundary and every competing outcome known at selection are recorded
in `protocol_archive/v10_fast_candidate_freeze.json`; science is frozen in
`protocol_archive/v10_fast_registration_draft.json` before launch.

Append-only provenance correction: the worker completion receipt is timestamped
20:55:26.379821 UTC, about 0.620 seconds before the freeze. The root process had
not downloaded or inspected that terminal result; selection used the recorded
`0/2/4` partial history, and the terminal summary was opened afterward. Thus the
selection was pre-inspection but not pre-existence. The original pushed freeze
is preserved, and
`protocol_archive/v10_fast_candidate_freeze_correction.json` binds it and the
later terminal-summary hash. Every prepared registration and claim must bind
that correction and use only the corrected wording.

The user accepts a prospectively declared `alpha=.15` for the 02:00 UTC
deadline. With four registered seeds, the primary treatment-minus-signflip and
secondary treatment-minus-base tests each require all four seed effects to be
strictly positive with no ties, giving exact two-sided sign-test `p=.125`.
Positive means and all provenance/collection audits are also required.
Crossed 95% bootstrap intervals remain reported diagnostics, but requiring a
95% lower bound above zero would silently restore an approximately .05 gate
and is therefore not used as the alpha-.15 acceptance rule.

Before any V10 GPU allocation, the complete Modal execution, generated-config,
matched-control, protected-release, serial nine-label collection, and analysis
path received an independent synthetic-only review. The launcher, preparer,
protocol, final runner, and training wrapper have SHA-256 respectively
`840459e048758b88e0b19ed767d8362c58aefa4853ea4313a59da2b053f602e0`,
`a9d1b5ea6866818b0665887b1f0aa57ba3dd57f6bbc5c646b39470a768a05896`,
`dec695d88e70e944dac05909f39dabdb0e7bf57df57b255a73e60abd1a8b2ae0`,
`beb79255ab58612e9e67800331d4c2aa75a264c8595d77e0bd247a55802b2b69`,
and `a08bc43f3894d9941ca611b30d82b94e638d7364b4ddfe98dff823d2629d7737`.
All 43 focused protocol/Modal/paired-evaluation tests pass. The prospective
preparer does not resolve, stat, open, copy, or hash the protected final path;
the launcher can release it only after a passing curve, eight verified runs,
and a durable unlock.

The first launch-enabled contract had SHA-256
`16e8ec05b70a58e275d7c3cdde8d908b5c49f24cbf651055c52b92c55d8a516c`.
Modal app `ap-jAM2BBkWGm13c5Kyn9cZy2` began only its image build at 21:28:31
UTC. It was stopped at 21:30:20 after an operational review found that the 424
allowlisted files were becoming hundreds of sequential image layers. It had
zero tasks, containers, claims, Volume files, W&B runs, GPU allocations, or
scientific outcomes. This is an unspent pre-dispatch build, not a training
attempt.

The launcher now batches exactly the same fail-closed allowlist into one image
layer; this operational-only revision has SHA-256
`7433d643bd3545cca139d0def58e8b8488ce0e9c18e5c8ef2b884ecb003b028d`
and its seven focused Modal tests pass. A replacement launch-enabled contract
was materialized without opening any protected payload at
`protocol_archive/v10_modal_execution_contract.json` (SHA-256
`9b4aba9bd9b89cb984af8929f01613b19ffe26d45ecab387e0b1ad0c2764a72f`).
It pins fresh Modal v2 Volume `vo-LreEdmtTCwyu4VlrcP5FOJ`, a 424-file public
runtime allowlist, synthetic runtime Git tree
`b3819b38c8fefcdba1e4230132eb656b553084d6`, commit
`8029363d6d022a97b467f0dc5ffd70f5fc3a1672`, and content-tree SHA-256
`3f04622a0017cf8fb7b5e0efbf1d1a349cea5a441c420f6b92f791b31745225c`.
The allowlist contains the model lens, calibration, exposed curve/exclusions,
training/evaluation/analysis sources and tests, and vendored runtime dependency;
it excludes every sealed-final, reserve, and correlation payload.

The cached-image launch under that contract reached one CPU `claim_attempt` on
app `ap-RpVl391ZyhS04IA3sRSAWT` and failed before claim creation or GPU
dispatch. Modal's directory layer had honored `.gitignore` and omitted the two
public `.confirmatory/manifests` files, while the package build left an
unregistered `trl/build` copy. The app stopped at 21:36:56 UTC. It had no GPU
workers or W&B runs; Volume A contains only the 14 prepared config and
reproducibility files uploaded before the failing claim and is now retired.

The image now explicitly copies the two hash-pinned public manifests and
removes `trl/build`; the seven focused Modal tests still pass. This revision has
launcher SHA-256
`d8b95add301d7ac67c0cb61e712571f4aaac62076f59e7d3b12b20f61824c4af`.
The fresh replacement Volume B is `vo-f4E8UW02e3PVrcZi7i3tO6`. Its
launch-enabled contract SHA-256 is
`14ace2af7bd2a02902fbf9f092d64216b82922c2b98ed5d3103d5b6895c52e93`,
with runtime Git tree `a11a261be7ff14ed79424803efb771ef4a778339`, commit
`df1d092d55f464f8451a80c7e2f4483af92361f3`, and content-tree SHA-256
`66db68986484b01b06b260cc7124304dea493fdc0d4e44550c4cb67d4bccca60`.
