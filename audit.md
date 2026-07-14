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
