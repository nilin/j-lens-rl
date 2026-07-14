#!/usr/bin/env python3
"""Prepare and guard the conditional celebration-taper V6 experiment.

V6 is intentionally inert until three committed records exist:

* ``protocol_archive/v6_celebration_selection_closeout.json`` records the
  adaptive seed-167 source evidence and the operator knowledge boundary.
* ``protocol_archive/v6_celebration_selected_recipe.json`` freezes one fully
  resolved emotional-only training recipe and byte-identifies its artifacts.
* ``protocol_archive/v6_celebration_registration.json`` byte-pins that lock,
  the runner, the untouched data split, and four explicit curve nodes.

Preparation and launch additionally require a later, committed V5 terminal
closeout proving ``curve_failed`` before any final unlock, collection, or
evaluation.  A missing, passing, incomplete, or outcome-opening V5 closeout
cancels V6.  The protocol never chooses a word, sign, recipe, checkpoint, or
curve node; it verifies the registered choices, derives eight matched
sign-flipped controls mechanically, and keeps all V6 state isolated below
``.confirmatory/v6``.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO = Path(__file__).resolve().parents[1]
STATE_DIR = REPO / ".confirmatory" / "v6"
MANIFEST_DIR = STATE_DIR / "manifests"
CONFIG_DIR = STATE_DIR / "configs"
ARTIFACT_DIR = STATE_DIR / "frozen_artifacts"
REPRODUCIBILITY_DIR = STATE_DIR / "reproducibility"
RUN_DIR = STATE_DIR / "runs"
EVAL_DIR = STATE_DIR / "evals"
EVIDENCE_DIR = STATE_DIR / "evidence"
STATE_PATH = STATE_DIR / "protocol_state.json"
CURVE_GATE_PATH = EVIDENCE_DIR / "curve_gate.json"
CURVE_PLOT_PATH = EVIDENCE_DIR / "curve.png"
COMPLETED_RUNS_PATH = EVIDENCE_DIR / "completed_runs.json"
UNLOCK_PATH = STATE_DIR / "final_unlocked.json"
FINAL_COLLECTION_PATH = STATE_DIR / "final_collection.json"
SEALED_COMPARISON_PATH = EVIDENCE_DIR / "sealed_comparison.json"
ACCEPTANCE_PATH = EVIDENCE_DIR / "acceptance.json"
BUNDLE_INVENTORY_PATH = EVIDENCE_DIR / "evidence_bundle_inventory.json"
ATTEMPT_CLAIM_PATH = STATE_DIR / "attempt_claim.json"
ATTEMPT_STATUS_PATH = STATE_DIR / "attempt_status.json"
LAUNCH_RECEIPT_PATH = STATE_DIR / "launch_receipt.json"
CLOSEOUT_CANDIDATE_PATH = EVIDENCE_DIR / "git_closeout_candidate.json"
EXPORT_PLAN_PATH = EVIDENCE_DIR / "durable_export_plan.json"
EXPORT_DIR = STATE_DIR / "exports"

REGISTRATION_PATH = (
    REPO / "protocol_archive" / "v6_celebration_registration.json"
)
DEFAULT_RECIPE_LOCK_PATH = (
    REPO / "protocol_archive" / "v6_celebration_selected_recipe.json"
)
SELECTION_CLOSEOUT_PATH = (
    REPO / "protocol_archive" / "v6_celebration_selection_closeout.json"
)
SUPERSEDED_V1_REGISTRATION_PATH = (
    REPO
    / "protocol_archive"
    / "v6_celebration_registration_v1_superseded.json"
)
PRELAUNCH_CORRECTION1_PATH = (
    REPO / "protocol_archive" / "v6_celebration_prelaunch_correction1.json"
)
SUPERSEDED_V2_REGISTRATION_PATH = (
    REPO
    / "protocol_archive"
    / "v6_celebration_registration_v2_superseded.json"
)
PRELAUNCH_CORRECTION2_PATH = (
    REPO / "protocol_archive" / "v6_celebration_prelaunch_correction2.json"
)
EMOTIONAL_DECISION_PATH = REPO / "protocol_archive" / "emotional_only_decision.json"
V5_REGISTRATION_PATH = (
    REPO / "protocol_archive" / "v5_emotional_registration.json"
)
V4_CLOSEOUT_PATH = REPO / "protocol_archive" / "v4_closeout.json"
V5_INFRASTRUCTURE_AMENDMENT1_PATH = (
    REPO / "protocol_archive" / "v5_emotional_infrastructure_amendment1.json"
)
# This file deliberately does not exist at V6 freeze.  It must be added from
# the immutable V5 export after V5 terminates, committed, and structurally
# verified before V6 can be prepared or launched.
V5_TERMINAL_CLOSEOUT_PATH = (
    REPO / "protocol_archive" / "v5_emotional_terminal_closeout.json"
)
V5_TERMINAL_EVIDENCE_DIR = (
    REPO / "protocol_archive" / "v5_emotional_terminal_evidence"
)
V5_TERMINAL_EVIDENCE_PATHS = {
    name: V5_TERMINAL_EVIDENCE_DIR / filename
    for name, filename in {
        "attempt_claim": "attempt_claim.json",
        "launch_receipt": "launch_receipt.json",
        "attempt_status": "attempt_status.json",
        "curve_gate": "curve_gate.json",
        "curve_plot": "curve.png",
        "bundle_inventory": "evidence_bundle_inventory.json",
        "root_inventory": "root_inventory.json",
        "evidence_inventory": "evidence_inventory.json",
        "durable_export_receipt": "durable_export_receipt.json",
    }.items()
}

V4_MANIFEST_DIR = REPO / ".confirmatory" / "manifests"
V4_PARENT_PATH = V4_MANIFEST_DIR / "sealed_final_indices.json"
V4_RESERVE_PATH = V4_MANIFEST_DIR / "future_reserve_indices.json"
V4_TRAIN_EXCLUSIONS_PATH = V4_MANIFEST_DIR / "train_exclusions.json"

REGISTRATION_PROTOCOL = (
    "j-lens-rl-confirmatory-v6-celebration-taper-registration-v3"
)
RECIPE_LOCK_PROTOCOL = "j-lens-rl-celebration-taper-selected-recipe-lock-v1"
SELECTION_CLOSEOUT_PROTOCOL = "j-lens-rl-v6-celebration-selection-closeout-v1"
PROTOCOL = "j-lens-rl-confirmatory-v6-celebration-taper"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
SEEDS = tuple(range(176, 184))
REQUIRED_CONDITIONS = ("jlens", "signflip")
MAX_REGISTERED_UPDATES = 25
MAX_GPU_CONTAINERS = 1
GPU_TYPE = "L40S"
GLOBAL_MODAL_GPU_LIMIT = 1
GPU_APP_OVERLAP_POLICY = "no other Modal GPU app may overlap this V6 attempt"
GPU_EXCLUSIVE_CONFIRMATION = "confirmed-no-other-modal-gpu-app-running"
BACKEND_FALLBACK_POLICY = (
    "this registration permits one whole attempt on Modal L40S with online W&B only; "
    "never mix hardware or tracking modes within an attempt; an RTX4090/offline-W&B "
    "fallback requires a separate registration made either before any Modal outcome or "
    "after immutable closeout of a partial Modal attempt, plus fresh state/claim, preserved "
    "offline W&B directories, and the same frozen IDs and scientific inputs"
)
INFRASTRUCTURE_RETRY_POLICY = (
    "a completed run may only retry publication of its immutable terminal W&B artifact; "
    "never overwrite or resume model optimization; any partial nonterminal run closes the "
    "whole attempt immutably and requires a fresh registered claim/volume"
)
MODAL_APP_NAME = "j-lens-rl-confirmatory-v6-celebration-taper"
VOLUME_NAME = "j-lens-rl-confirmatory-v6-celebration-taper-20260714a"
FINAL_LABELS = (
    "base",
    *(f"jlens_seed{seed}" for seed in SEEDS),
    *(f"signflip_seed{seed}" for seed in SEEDS),
)
WANDB_ENTITY = "nilinabra-spare-time"
WANDB_PROJECT = "j-lens-rl"
WANDB_GROUP = "confirm-v6-emotional-celebration-taper-h10"
WANDB_RUN_PREFIX = "confirm-v6-emotional-celebration-taper-h10"
WANDB_RUN_IDS = {
    label: f"{WANDB_RUN_PREFIX}-{label}"
    for label in FINAL_LABELS
    if label != "base"
}
TREATMENT_COMPONENT_WEIGHTS = (1.0, 0.25)
COMBINED_REWARD_RANGE = (-6.25, 6.25)
COMBINED_REWARD_UNIT = (
    "weighted combined two-component calibration z-score with weights [1.0, 0.25]; "
    "each component is clipped to [-5, 5], so the combined range is [-6.25, 6.25]"
)

INITIAL_OPERATOR_KNOWLEDGE_BOUNDARY = {
    "selection_agent": (
        "recipe and 0/4/6/10 nodes derived only from committed seed-167 "
        "emotional-family artifacts; no V5 outcome was inspected by that agent"
    ),
    "root_operator": (
        "root had seen partial seed-168 V5 joy results before this V6 freeze"
    ),
    "independence_scope": (
        "the partial V5 result did not select or alter the celebration recipe, "
        "curve nodes, seeds, controls, or fresh V6 repartition"
    ),
    "claim_boundary": (
        "V6 is prospective only for its newly allocated curve/final rows and "
        "does not erase the adaptive seed-167 source selection"
    ),
}
OPERATOR_KNOWLEDGE_BOUNDARY = INITIAL_OPERATOR_KNOWLEDGE_BOUNDARY
CORRECTED_OPERATOR_KNOWLEDGE_BOUNDARY = {
    "selection_agent": (
        "recipe and 0/4/6/10 nodes remain derived only from committed seed-167 "
        "emotional-family artifacts; no V5 outcome was inspected by that agent"
    ),
    "root_operator": (
        "root had seen partial results for six V5 seeds before prelaunch correction 1"
    ),
    "timing": (
        "correction 1 was frozen before root saw a V5 aggregate or terminal result, "
        "and before any V6 outcome existed"
    ),
    "independence_scope": (
        "the additional partial V5 results did not alter the celebration recipe, "
        "curve nodes, seeds, controls, fresh V6 repartition, acceptance rules, or "
        "W&B identities"
    ),
    "claim_boundary": (
        "V6 is prospective only for its newly allocated curve/final rows and "
        "does not erase the adaptive seed-167 source selection"
    ),
}
CORRECTION2_OPERATOR_KNOWLEDGE_BOUNDARY = {
    "selection_agent": (
        "recipe and 0/4/6/10 nodes remain derived only from committed seed-167 "
        "emotional-family artifacts; no V5 outcome was inspected by that agent"
    ),
    "root_operator": (
        "root knew the full terminal V5 mean and per-seed curves and terminal "
        "evidence before prelaunch correction 2"
    ),
    "timing": "correction 2 was frozen before any V6 outcome existed",
    "independence_scope": (
        "the later V5 terminal outcomes and audit corrections did not alter the "
        "byte-semantically frozen celebration recipe, curve nodes, seeds, controls, "
        "fresh V6 repartition, acceptance rules, or W&B identities"
    ),
    "claim_boundary": (
        "V6 is prospective only for its newly allocated curve/final rows and "
        "does not erase the adaptive seed-167 source selection"
    ),
}

V5_ALLOCATION_SALT = "j-lens-rl-confirmatory-v5-alternative-screen-2026-07-14"
V6_ALLOCATION_SALT = "j-lens-rl-confirmatory-v6-celebration-taper-2026-07-14"
V4_PARENT_SHA256 = "acd2d497dcf96b2f3355925bb34979b9b7b3301e4c394066fc54ea57d093b6e3"
V4_PARENT_SET_SHA256 = "80bec8d196a36c1d2f02fb6aa3c7c5ea727a86c0c981d3d0611b0900776d6f74"
V5_CURVE_SHA256 = "b01409c011012641be96c84bfc35cb0b352cea902e54304105efa272a3eac6b2"
V5_CURVE_SET_SHA256 = "cc4f78e451e35be8ec7d17d3c4337db6a1aaed71357d749dfbef828063a5413f"
V5_FINAL_SHA256 = "6298b8e3d15b11985cf9febcd243dafd409ef07f22091388fa0793b6ebfe4228"
V5_FINAL_SET_SHA256 = "60308ed7212072ab0f1b2dbbf2d818ccfdc8da65134885e4c8d731058b051801"
V6_CURVE_SHA256 = "9c04151e1dac6b226bed07873e1003ab5c399a2978ca2b0dcb0d32c3a337ade5"
V6_CURVE_SET_SHA256 = "0bf980d5426e3daffbbc1fc375756617a0b45eda6339eb97cf6003e5a31f1ec1"
V6_FINAL_SHA256 = "1c3a544053504848318594ce21eea058d902884ba10c4f39ea3fa7796109b9c8"
V6_FINAL_SET_SHA256 = "eadcd0e2fc194b0e38bc1c9f4aa1bbf6e6b3ba1043b0015eedee59aec133637c"
RESERVE_SHA256 = "cfbac5a2f4cf3cc94e1882bf412cdfc4af9c84347647fa9843dc09967f8a03a6"
TRAIN_EXCLUSIONS_SHA256 = "7c1ca4f404ba9149093cc3c57dc3607582f671397e2fe99e93449848c1d65d61"
V4_CLOSEOUT_SHA256 = "aaf4bcde9a9cacc482c7f3dde94218cf02a6aa60be81e43cae5cde3086d17e35"
EMOTIONAL_DECISION_SHA256 = "50cc3cb32e0cf74feeaeae79c2faf7b91c6caeae34b6e4f1101819d51e15b238"
V5_REGISTRATION_SHA256 = "b2c49eaad02169e0da818a7893205f5ff7084fc41da1db6441ea7226a562d527"
V5_INFRASTRUCTURE_AMENDMENT1_SHA256 = "d845fd829b00deb80cfed402e8fd8a04543c2ddffc451329dcfc572e296f3f42"
SELECTION_CLOSEOUT_SHA256 = "3efb6adaa070e8c77838d55333d7b556e2e96bf27e7a18d9740443e6c724e2b8"
SUPERSEDED_V1_REGISTRATION_SHA256 = "79f80dc0acea6c6b8410f8936ec364bcc4405eaf49ffdbc5500ec0cb103f875d"
PRELAUNCH_CORRECTION1_SHA256 = "8d1fe785a495c5a8da689d7f80fec9c5ebcd09f9770a37a3efa5f38732a8e719"
SUPERSEDED_V2_REGISTRATION_SHA256 = "b15aa79bb15d754ad6c8b5015736acd13f95651f9c9751dad0c59d710a8361dd"
PRELAUNCH_CORRECTION2_SHA256 = "f4b40790859bff1a614c279f9b693571b315a97294221bb3c5aebebb4639ff07"
SUPERSEDED_SCIENTIFIC_PROJECTION_SHA256 = (
    "0621bc7402187223b47f665872b4c0bdb2c53b64661b26b94dcc76492a0fe93e"
)

V5_PROTOCOL = "j-lens-rl-confirmatory-v5-emotional"
V5_MODAL_APP_NAME = "j-lens-rl-confirmatory-v5-emotional"
V5_VOLUME_NAME = "j-lens-rl-confirmatory-v5-emotional-20260714b"
V5_CLAIM_ID = "c2d9ed29ebba46649f1ea182d7d50014"
V5_CLAIM_GIT_COMMIT = "252a2319dc0ada5e99ddeecf507d7246590531d7"
V5_MODAL_APP_ID = "ap-2o4XOP7jhqcrHyqkGN55wL"
V5_MODAL_FUNCTION_CALL_ID = "fc-01KXFXCF23D6DRB5VNXCVJB8TN"
V5_LAUNCH_RECEIPT_SHA256 = (
    "5123864f59094b572304a9aaf1317f41a9082fdafa5725876b4801858ca216d8"
)
V5_LAUNCH_SUBMITTED_AT_UTC = "2026-07-14T08:55:08.732683+00:00"
V5_GPU_APP_OVERLAP_POLICY = "no other Modal GPU app may overlap this V5 attempt"
V5_RECIPE_LOCK_SHA256 = "006bd685f045a385889d1e418e76437140a404177f0d1148d308226390a2a547"
V5_TERMINAL_EVIDENCE_SHA256 = {
    "attempt_claim": "d8f7779ba539b4e4a23d0b5cb96f3d75b6fd97f47f3917d151f42f6e6ba098ed",
    "launch_receipt": V5_LAUNCH_RECEIPT_SHA256,
    "attempt_status": "463d2dc01b44c837daf7cc761d6f587bce804048889e4dbe95463058e734a75e",
    "curve_gate": "09770542837464bb80c48c036099521f0ae05ec2e0c81c2440f1f06816a4eede",
    "curve_plot": "61d6c49740bb277335a0500a1842396d3328baa3f576540d7a1c78ccc659f280",
    "root_inventory": "e9f08d9e5a102f4c980f0d00ada6b3ad6d810bf8f8bb790aed7607d91542af60",
    "evidence_inventory": "7f46f397b22756d6b6b2cf607ea2f4c5c4fa97e999258eb45b25a6469b88369b",
    "bundle_inventory": "f098574bbbf97f645c4f66b4e33ce96c8ef8694bc6d37a721fc37f2a7f0976c2",
    "durable_export_receipt": "155c699733510b16975099e6f9aa1566697a0dc62716510c11db38b13be4ed78",
}
V5_DURABLE_ARCHIVE_SHA256 = (
    "8710288864aac8bde570beff50ccc9835b1b0d98e2470a7a3bab897bcf620b17"
)
V3_OUTCOME_STATUS_AT_FREEZE = (
    "prelaunch correction 2: no V6 outcome existed; selecting agent used no V5 "
    "outcomes; root knew the full terminal V5 mean and per-seed curves and evidence; "
    "scientific and W&B choices remain byte-semantically identical to V1 and V2"
)

CONDITIONAL_LAUNCH_PREDICATE = {
    "closeout_path": "protocol_archive/v5_emotional_terminal_closeout.json",
    "required_protocol": "j-lens-rl-confirmatory-v5-emotional-terminal-closeout-v1",
    "required_terminal_stage": "curve_failed",
    "required_final_unlocked_present": False,
    "required_final_collection_present": False,
    "required_evals_directory_present": False,
    "required_final_evaluation_labels": [],
    "required_sealed_comparison_present": False,
    "required_final_outcomes_unopened": True,
    "required_v5_scientific_registration_sha256": V5_REGISTRATION_SHA256,
    "required_v5_infrastructure_amendment1_sha256": (
        V5_INFRASTRUCTURE_AMENDMENT1_SHA256
    ),
    "required_operational_identity": {
        "claim_id": V5_CLAIM_ID,
        "modal_app": V5_MODAL_APP_NAME,
        "modal_app_id": V5_MODAL_APP_ID,
        "modal_function_call_id": V5_MODAL_FUNCTION_CALL_ID,
        "volume": V5_VOLUME_NAME,
        "launch_receipt_sha256": V5_LAUNCH_RECEIPT_SHA256,
        "launch_submitted_at_utc": V5_LAUNCH_SUBMITTED_AT_UTC,
        "gpu_type": "L40S",
        "global_modal_gpu_limit": 1,
    },
    "required_source_evidence_paths": {
        name: path.relative_to(REPO).as_posix()
        for name, path in V5_TERMINAL_EVIDENCE_PATHS.items()
    },
    "required_source_evidence_sha256": V5_TERMINAL_EVIDENCE_SHA256,
    "required_claim_git_commit": V5_CLAIM_GIT_COMMIT,
    "required_durable_archive_sha256": V5_DURABLE_ARCHIVE_SHA256,
    "source_evidence_policy": (
        "every closeout evidence identity must name its exact registered tracked "
        "path and its SHA-256 must equal the actual committed source bytes"
    ),
    "authorization": (
        "the closeout is intentionally unavailable at V6 freeze; prepare and every "
        "launch verification require later committed bytes and record their SHA-256"
    ),
    "cancellation_rule": (
        "missing or non-curve-failed V5 closeout, any unlock/collection/evaluation, "
        "or any uncertainty about sealed-final exposure permanently cancels V6"
    ),
}

CELEBRATION_SOURCE_CONFIG_SHA256 = "a5f2105f2e45f85d9d82d347d20302c0a937eb0e2c377d539ded6e9c59939fc2"
CELEBRATION_CALIBRATION_SHA256 = "93d05caf4848e745c07d908034b36f0b1ae465d8d89e1681134869c6b87a8ee6"
CELEBRATION_SELECTION_EVIDENCE_SHA256 = {
    "resolved_config": CELEBRATION_SOURCE_CONFIG_SHA256,
    "calibration": CELEBRATION_CALIBRATION_SHA256,
    "run_manifest": "40d95256a51202c82bea0d2447e29b2b14d67d40ad27ec412468d748eb298006",
    "screen_result": "9786d6f36a2175d21e16d48f4485b6d0b8b75cccacee772f46946d14165d5718",
    "validation_history": "b45495431ca43a1270c5ffc8c8d25e6057272130594d74429d43c1137535d6ed",
    "log_history": "263d030f5305e90761aa845943092982761d5541c826334609926645efad1719",
}
CELEBRATION_DECLARED_TRANSFORMATIONS = {
    "updates": {"from": 25, "to": 10},
    "save_every": {"from": 25, "to": 10},
    "validation_steps": {
        "from": [2, 4, 6, 10, 15, 20, 25],
        "to": [4, 6, 10],
    },
    "calibration_path": {
        "from": "/word_explore/artifacts/celebration_calibration.json",
        "to": "source_calibration.path",
    },
    "generated_run_identity": {
        "seed": "replace exploratory seed 167 with registered seeds 176..183",
        "output_dir": "derive from registered V6 condition and seed",
        "run_name_and_wandb": "use the exact registered W&B identity per condition",
        "data_paths": "replace with frozen V6 curve/exclusion/final manifests",
        "artifact_paths": "replace with byte-identical V6 frozen artifacts",
        "control": "negate only the registered treatment component weights",
    },
}

SPLIT_REGISTRATION = {
    "allocation_salt": V6_ALLOCATION_SALT,
    "source_parent": {
        "role": "V5 sealed final, usable only after the conditional predicate passes",
        "size": 1300,
        "manifest_sha256": V5_FINAL_SHA256,
        "sorted_set_sha256": V5_FINAL_SET_SHA256,
    },
    "curve": {
        "size": 400,
        "manifest_sha256": V6_CURVE_SHA256,
        "sorted_set_sha256": V6_CURVE_SET_SHA256,
    },
    "sealed_final": {
        "size": 900,
        "manifest_sha256": V6_FINAL_SHA256,
        "sorted_set_sha256": V6_FINAL_SET_SHA256,
    },
    "future_reserve": {"size": 64, "manifest_sha256": RESERVE_SHA256},
    "train_exclusions_manifest_sha256": TRAIN_EXCLUSIONS_SHA256,
}

CURVE_CRITERION = (
    "the first registered post-baseline eight-seed mean is strictly above "
    "the registered baseline mean, and each of the next two registered means "
    "is greater than or equal to its predecessor"
)
MATCHED_CONTROL_RULE = (
    "negate every selected treatment score-component weight and change "
    "nothing else except condition identity"
)
ANALYSIS_REGISTRATION = {
    "primary_estimand": (
        "paired difference-in-differences: (celebration-treatment minus base) minus "
        "(signflip-control minus base), matched by seed and sealed item"
    ),
    "secondary_estimand": (
        "paired celebration-treatment minus base across seeds and items"
    ),
    "bootstrap_method": "crossed seed-and-item percentile bootstrap",
    "bootstrap_samples": 10_000,
    "bootstrap_seed": 0,
    "confidence": 0.95,
    "seed_sign_test": "exact two-sided sign test across eight registered seeds",
}
ACCEPTANCE_REGISTRATION = {
    "curve_gate": CURVE_CRITERION,
    "treatment_vs_base_mean": "> 0",
    "treatment_vs_base_crossed_95pct_ci_low": "> 0",
    "treatment_vs_base_seed_effects": "8 positive, 0 negative, 0 ties",
    "treatment_vs_base_exact_two_sided_sign_p": 0.0078125,
    "difference_in_differences_mean": "> 0",
    "difference_in_differences_crossed_95pct_ci_low": "> 0",
    "literal_provenance_environment_and_collection_audits": "all pass",
}


class ProtocolError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def serialized_json_sha256(value: Any) -> str:
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def require_clean_worktree() -> str:
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise ProtocolError(
            "V6 confirmatory work requires a clean committed tree; git status is:\n"
            + status
        )
    return git("rev-parse", "HEAD")


def _repo_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ProtocolError(f"{field} must be a non-empty repository-relative path")
    relative = Path(value)
    if relative.is_absolute():
        raise ProtocolError(f"{field} must be repository-relative")
    resolved = (REPO / relative).resolve()
    try:
        resolved.relative_to(REPO.resolve())
    except ValueError as error:
        raise ProtocolError(f"{field} escapes the repository") from error
    return resolved


def manifest_payload(indices: Iterable[int]) -> dict[str, Any]:
    return {
        "dataset": "openai/gsm8k",
        "subset": "main",
        "split": "train",
        "indices": [int(index) for index in indices],
    }


def load_indices(path: Path) -> list[int]:
    payload = json.loads(path.read_text())
    values = payload.get("indices") if isinstance(payload, dict) else payload
    if not isinstance(values, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in values
    ):
        raise ProtocolError(f"invalid index manifest: {path}")
    if len(values) != len(set(values)) or any(value < 0 for value in values):
        raise ProtocolError(f"duplicate or negative index in {path}")
    return values


def _v5_allocation_key(index: int) -> bytes:
    return hashlib.sha256(f"{V5_ALLOCATION_SALT}:{index}".encode()).digest()


def reconstruct_v5_split(parent: Sequence[int]) -> tuple[list[int], list[int]]:
    if len(parent) != 1700 or len(set(parent)) != 1700:
        raise ProtocolError("V5 source parent must contain exactly 1,700 unique indices")
    ordered = sorted((int(index) for index in parent), key=_v5_allocation_key)
    return ordered[:400], ordered[400:]


def _v6_allocation_key(index: int) -> bytes:
    return hashlib.sha256(f"{V6_ALLOCATION_SALT}:{index}".encode()).digest()


def allocate_v6(parent: Sequence[int]) -> tuple[list[int], list[int]]:
    """Split only V5's conditionally unopened 1,300-row final parent."""
    if len(parent) != 1300 or len(set(parent)) != 1300:
        raise ProtocolError("V6 source parent must contain exactly 1,300 unique indices")
    ordered = sorted((int(index) for index in parent), key=_v6_allocation_key)
    return ordered[:400], ordered[400:]


def validate_curve_steps(
    value: Any, validation_steps: Sequence[int], fixed_updates: int
) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(isinstance(step, bool) or not isinstance(step, int) for step in value)
    ):
        raise ProtocolError(
            "the final registration must explicitly contain exactly four integer curve nodes"
        )
    steps = tuple(value)
    if steps[0] != 0 or any(right <= left for left, right in zip(steps, steps[1:])):
        raise ProtocolError(
            "registered curve nodes must begin at step 0 and be strictly increasing"
        )
    if steps[-1] > fixed_updates:
        raise ProtocolError("a registered curve node exceeds the fixed training horizon")
    available = {0, *(int(step) for step in validation_steps)}
    if any(step not in available for step in steps):
        raise ProtocolError(
            f"registered curve nodes {list(steps)} are not all present in the recipe's "
            f"observational schedule {sorted(available)}"
        )
    return steps


def curve_means_pass(values: Sequence[float]) -> bool:
    """Apply the registered baseline-plus-three-transition shape criterion."""
    if len(values) != 4 or any(not math.isfinite(float(value)) for value in values):
        raise ProtocolError("curve gate requires exactly four finite mean values")
    means = [float(value) for value in values]
    return means[1] > means[0] and means[2] >= means[1] and means[3] >= means[2]


def validate_target_words(value: Any) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(word, str) or not word or word != word.strip() for word in value)
        or len(value) != len(set(value))
    ):
        raise ProtocolError("target_words must be a non-empty unique list of trimmed strings")
    # Fail closed on case, whitespace, or compound evasions.  The lens artifact
    # may retain its historical filename; only reward targets are governed by
    # the user's emotional-only decision.
    if any(re.search("solved", word, flags=re.IGNORECASE) for word in value):
        raise ProtocolError("'solved' is retired and forbidden anywhere in target_words")
    return list(value)


def observed_selected_history_scalar_series(
    target_label: str,
    score_components: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Map every scalar key observed in the byte-pinned celebration history."""
    components = validate_score_components(list(score_components))
    weights = [float(item["weight"]) for item in components]
    combined_abs_bound = 5.0 * sum(abs(weight) for weight in weights)
    combined_range = [-combined_abs_bound, combined_abs_bound]
    combined_unit = (
        "weighted combined two-component calibration z-score; each component "
        "is clipped to [-5, 5] before applying the resolved run-config weights"
    )
    mean_definition = (
        "mean weighted combined two-component J-lens output, not an individual "
        "component score; the actual signed weights come from the resolved run "
        f"config (registered treatment {weights}, matched signflip "
        f"{[-weight for weight in weights]}) and the combined score range is "
        f"{combined_range}"
    )
    std_definition = (
        "nonnegative standard deviation of the weighted combined two-component "
        "J-lens output, not an individual component; each component is clipped "
        "to [-5, 5] before applying the actual signed weights from the resolved "
        "run config"
    )
    # Transformers sends these five terminal aggregates in the final ``train``
    # log.  The W&B integration promotes them to run summary fields rather than
    # retaining them as points in the scalar history.  Keep the local record
    # mapping complete without promising a remote history series that cannot be
    # reconstructed from W&B.
    wandb_summary_keys = {
        "total_flos",
        "train_loss",
        "train_runtime",
        "train_samples_per_second",
        "train_steps_per_second",
    }
    units_and_definitions = {
        "clip_ratio/high_max": ("fraction of eligible sampled tokens", "fraction meeting the maximum upper-clipping predicate"),
        "clip_ratio/high_mean": ("fraction of eligible sampled tokens", "fraction meeting an upper-clipping predicate, averaged over the batch"),
        "clip_ratio/low_mean": ("fraction of eligible sampled tokens", "fraction meeting a lower-clipping predicate, averaged over the batch"),
        "clip_ratio/low_min": ("fraction of eligible sampled tokens", "fraction meeting the minimum lower-clipping predicate"),
        "clip_ratio/region_mean": ("fraction of eligible sampled tokens", "fraction outside the allowed clipping region, the union of low- and high-clipped tokens"),
        "completions/clipped_ratio": ("fraction", "fraction of completions clipped at the generation limit"),
        "completions/max_length": ("generated tokens", "maximum sampled completion length"),
        "completions/max_terminated_length": ("generated tokens", "maximum naturally terminated completion length"),
        "completions/mean_length": ("generated tokens", "mean sampled completion length"),
        "completions/mean_terminated_length": ("generated tokens", "mean naturally terminated completion length"),
        "completions/min_length": ("generated tokens", "minimum sampled completion length"),
        "completions/min_terminated_length": ("generated tokens", "minimum naturally terminated completion length"),
        "entropy": ("nats per sampled token", "mean policy-token entropy"),
        "epoch": ("training epochs", "fractional pass through the registered training sample budget"),
        "frac_reward_zero_std": ("fraction of prompt groups", "fraction of GRPO groups with zero reward standard deviation"),
        "grad_norm": ("L2 norm", "optimizer gradient norm before the registered clipping rule"),
        f"jlens/{target_label}_literal_rate": ("fraction of completions", "fraction containing a configured literal target whose causal positions are reward-masked"),
        f"jlens/{target_label}_mean": (combined_unit, mean_definition),
        "kl": ("nats per sampled token", "mean policy-to-reference KL estimate"),
        "learning_rate": ("optimizer coefficient", "registered optimizer learning rate"),
        "loss": ("dimensionless objective", "logged DAPO/GRPO training loss"),
        "num_tokens": ("tokens", "cumulative processed token count"),
        "reward": (combined_unit, mean_definition),
        "reward_std": (combined_unit, std_definition),
        f"rewards/jlens_{target_label}_reward/mean": (combined_unit, mean_definition),
        f"rewards/jlens_{target_label}_reward/std": (combined_unit, std_definition),
        "step": ("optimizer updates", "completed optimizer updates"),
        "step_time": ("seconds per optimizer update", "wall-clock duration of the optimizer update"),
        "total_flos": ("floating-point operations", "trainer estimate of cumulative floating-point operations"),
        "train_loss": ("dimensionless objective", "trainer-wide mean training loss"),
        "train_runtime": ("seconds", "trainer-wide training wall time"),
        "train_samples_per_second": ("prompt samples per second", "trainer-wide prompt-sample throughput"),
        "train_steps_per_second": ("optimizer updates per second", "trainer-wide optimizer-step throughput"),
        "validation/exact_match": ("fraction of 400 fixed examples", "deterministic greedy GSM8K exact-answer accuracy"),
        "validation/exact_match_ci95_high": ("accuracy fraction", "upper endpoint of the fixed-validation 95% binomial interval"),
        "validation/exact_match_ci95_low": ("accuracy fraction", "lower endpoint of the fixed-validation 95% binomial interval"),
        "validation/literal_target_completion_rate": ("fraction of completions", "fixed-validation completions containing an emotional target literal"),
        "validation/mean_length": ("generated tokens", "mean fixed-validation completion length"),
    }
    combined_mean_keys = {
        f"jlens/{target_label}_mean",
        "reward",
        f"rewards/jlens_{target_label}_reward/mean",
    }
    combined_std_keys = {
        "reward_std",
        f"rewards/jlens_{target_label}_reward/std",
    }
    result = {}
    for key, (unit, definition) in units_and_definitions.items():
        is_summary = key in wandb_summary_keys
        result[key] = {
            "local_file": "log_history.json",
            "local_field": key,
            "wandb_metric": None
            if is_summary
            else ("train/global_step" if key == "step" else f"train/{key}"),
            "wandb_summary_key": key if is_summary else None,
            "step_axis": "optimizer_update",
            "unit": unit,
            "definition": definition,
        }
        if key in combined_mean_keys:
            result[key]["range"] = combined_range
            result[key]["component_weights_by_condition"] = {
                "treatment": weights,
                "signflip_control": [-weight for weight in weights],
            }
        elif key in combined_std_keys:
            result[key]["nonnegative"] = True
            result[key]["component_weights_by_condition"] = {
                "treatment": weights,
                "signflip_control": [-weight for weight in weights],
            }
    return result


def metric_schema(
    target_words: Sequence[str],
    fixed_updates: int,
    score_components: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Semantic map for local histories and their Transformers/W&B names."""
    words = validate_target_words(list(target_words))
    label = "_".join(words)
    components = validate_score_components(list(score_components))
    component_weights = [float(item["weight"]) for item in components]
    combined_abs_bound = 5.0 * sum(abs(weight) for weight in component_weights)
    combined_range = [-combined_abs_bound, combined_abs_bound]
    combined_unit = (
        "weighted combined two-component calibration z-score; each component is "
        "clipped to [-5, 5] before applying resolved run-config weights"
    )
    combined_definition = (
        "mean weighted combined reward across the registered spatial components "
        f"with resolved run-config weights {component_weights}; this is "
        "not an individual component score"
    )
    return {
        "schema_version": 1,
        "target_words": words,
        "condition_weight_semantics": {
            "treatment": [1.0, 0.25],
            "signflip_control": [-1.0, -0.25],
            "rule": (
                "the shared schema derives aliases from each resolved run config; "
                "both conditions combine two components clipped individually to [-5, 5]"
            ),
        },
        "step_axes": {
            "optimizer_update": {
                "local_field": "step",
                "wandb_metric": "train/global_step",
                "unit": "optimizer update",
                "domain": [0, fixed_updates],
                "definition": "completed GRPO optimizer updates; validation step 0 is the unadapted initialization",
            },
            "sealed_item": {
                "local_field": "source_index",
                "wandb_metric": None,
                "unit": "GSM8K train source row",
                "definition": "one immutable sealed-final example; final evaluation is not a training time series",
            },
        },
        "wandb_rewrite": {
            "rule": "Transformers prefixes trainer log keys not beginning eval_ or test_ with 'train/'",
            "step_metric": "train/global_step",
            "step_sync": True,
        },
        "series": {
            "validation_exact_match": {
                "local_file": "validation_history.jsonl",
                "local_field": "exact_match",
                "wandb_metric": "train/validation/exact_match",
                "step_axis": "optimizer_update",
                "unit": "fraction of 400 fixed examples",
                "range": [0.0, 1.0],
                "definition": "mean deterministic greedy GSM8K exact-answer correctness on the registered curve manifest",
            },
            "validation_literal_target_completion_rate": {
                "local_file": "validation_history.jsonl",
                "local_field": "literal_target_completion_rate",
                "wandb_metric": "train/validation/literal_target_completion_rate",
                "step_axis": "optimizer_update",
                "unit": "fraction of completions",
                "range": [0.0, 1.0],
                "definition": "fraction containing at least one configured target as a case-insensitive whole-word literal",
            },
            "validation_mean_length": {
                "local_file": "validation_history.jsonl",
                "local_field": "mean_length",
                "wandb_metric": "train/validation/mean_length",
                "step_axis": "optimizer_update",
                "unit": "generated tokens per completion",
                "definition": "arithmetic mean greedy completion-token count",
            },
            "intrinsic_named_weighted_reward_mean": {
                "local_file": "log_history.json",
                "local_field": f"rewards/jlens_{label}_reward/mean",
                "wandb_metric": f"train/rewards/jlens_{label}_reward/mean",
                "step_axis": "optimizer_update",
                "unit": combined_unit,
                "range": combined_range,
                "definition": combined_definition,
            },
            "intrinsic_named_weighted_reward_std": {
                "local_file": "log_history.json",
                "local_field": f"rewards/jlens_{label}_reward/std",
                "wandb_metric": f"train/rewards/jlens_{label}_reward/std",
                "step_axis": "optimizer_update",
                "unit": combined_unit,
                "nonnegative": True,
                "definition": (
                    "nonnegative standard deviation of the named weighted combined "
                    "two-component J-lens output, not an individual component score"
                ),
            },
            "intrinsic_reward_mean": {
                "local_file": "log_history.json",
                "local_field": "reward",
                "wandb_metric": "train/reward",
                "step_axis": "optimizer_update",
                "unit": combined_unit,
                "range": combined_range,
                "definition": (
                    "mean of the sole weighted combined intrinsic reward supplied "
                    "to GRPO; no answer reward is present"
                ),
            },
            "intrinsic_reward_std": {
                "local_file": "log_history.json",
                "local_field": "reward_std",
                "wandb_metric": "train/reward_std",
                "step_axis": "optimizer_update",
                "unit": combined_unit,
                "nonnegative": True,
                "definition": (
                    "nonnegative standard deviation of the sole weighted combined "
                    "two-component intrinsic reward in the logged rollout batch; "
                    "this is not an individual component score"
                ),
            },
            "intrinsic_literal_rate": {
                "local_file": "log_history.json",
                "local_field": f"jlens/{label}_literal_rate",
                "wandb_metric": f"train/jlens/{label}_literal_rate",
                "step_axis": "optimizer_update",
                "unit": "fraction of sampled completions",
                "range": [0.0, 1.0],
                "definition": "fraction containing a tokenizer sequence for any configured literal target; those causal positions are masked from reward",
            },
            "learning_rate": {
                "local_file": "log_history.json",
                "local_field": "learning_rate",
                "wandb_metric": "train/learning_rate",
                "step_axis": "optimizer_update",
                "unit": "optimizer learning-rate coefficient",
                "definition": "learning rate emitted by the registered scheduler",
            },
            "completion_mean_length": {
                "local_file": "log_history.json",
                "local_field": "completions/mean_length",
                "wandb_metric": "train/completions/mean_length",
                "step_axis": "optimizer_update",
                "unit": "generated tokens per completion",
                "definition": "mean sampled completion-token count",
            },
            "completion_clipped_ratio": {
                "local_file": "log_history.json",
                "local_field": "completions/clipped_ratio",
                "wandb_metric": "train/completions/clipped_ratio",
                "step_axis": "optimizer_update",
                "unit": "fraction of sampled completions",
                "range": [0.0, 1.0],
                "definition": "fraction reaching the maximum generation length without termination",
            },
            "sealed_correct": {
                "local_file": "evals/<label>.jsonl",
                "local_field": "correct",
                "wandb_metric": None,
                "step_axis": "sealed_item",
                "unit": "boolean",
                "definition": "GSM8K exact-answer correctness recomputed from raw completion and pinned gold row",
            },
        },
        "observed_history_scalar_series": observed_selected_history_scalar_series(
            label, components
        ),
        "other_trainer_fields": {
            "storage": "all 38 scalar keys from the selected celebration history are explicitly mapped above and preserved verbatim in log_history.json",
            "step_axis": "optimizer_update",
            "wandb_prefix": "train/",
        },
    }


def validate_score_components(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ProtocolError("the selected recipe must contain score_components")
    components: list[dict[str, Any]] = []
    any_nonzero = False
    for index, raw in enumerate(value):
        if not isinstance(raw, dict) or "layer" not in raw or "weight" not in raw:
            raise ProtocolError(f"score component {index} requires layer and weight")
        component = dict(raw)
        layer = component["layer"]
        if isinstance(layer, bool) or not isinstance(layer, int):
            raise ProtocolError(f"score component {index} layer must be an integer")
        try:
            weight = float(component["weight"])
            start = float(component.get("start_fraction", 0.0))
            end = float(component.get("end_fraction", 1.0))
        except (TypeError, ValueError) as error:
            raise ProtocolError(f"score component {index} is not numeric") from error
        if not math.isfinite(weight) or not 0 <= start < end <= 1:
            raise ProtocolError(f"score component {index} has invalid weight or interval")
        if component.get("aggregation", "mean") not in {"mean", "max", "last"}:
            raise ProtocolError(f"score component {index} has invalid aggregation")
        any_nonzero |= weight != 0.0
        components.append(component)
    if not any_nonzero:
        raise ProtocolError("the selected recipe cannot have all-zero score weights")
    return components


def negate_score_components(value: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    components = validate_score_components(list(value))
    result: list[dict[str, Any]] = []
    for component in components:
        flipped = dict(component)
        flipped["weight"] = -float(component["weight"])
        result.append(flipped)
    return result


def _tracked_runtime_tree_sha256(prefix: str) -> str:
    names = [
        name
        for name in git("ls-files", "-z", "--", prefix).split("\0")
        if name and (REPO / name).is_file()
    ]
    if not names:
        raise ProtocolError(f"registered runtime source tree is empty: {prefix}")
    return canonical_sha256(
        {name: sha256_file(REPO / name) for name in sorted(names)}
    )


def _expected_execution_hashes() -> dict[str, str]:
    paths = {
        "protocol_sha256": REPO / "scripts" / "confirmatory_v6_protocol.py",
        "modal_runner_sha256": REPO / "modal_confirmatory_v6.py",
        "shell_launcher_sha256": REPO / "run_confirmatory_v6.sh",
        "modal_cache_assets_sha256": REPO / "scripts" / "modal_cache_assets.py",
        "modal_finalize_image_sha256": REPO / "scripts" / "modal_finalize_image.py",
        "pyproject_sha256": REPO / "pyproject.toml",
    }
    return {
        **{key: sha256_file(path) for key, path in paths.items()},
        "trl_build_metadata_sha256": canonical_sha256(
            {
                relative: sha256_file(REPO / relative)
                for relative in (
                    "trl/pyproject.toml",
                    "trl/MANIFEST.in",
                    "trl/VERSION",
                    "trl/README.md",
                    "trl/LICENSE",
                    "trl/CONTRIBUTING.md",
                )
            }
        ),
        "jlens_rl_runtime_tree_sha256": _tracked_runtime_tree_sha256("src/jlens_rl"),
        "trl_runtime_tree_sha256": _tracked_runtime_tree_sha256("trl/trl"),
    }


def _registered_execution_hashes() -> dict[str, str]:
    """Return the exact V6 execution bytes that the registration must pin."""
    return _expected_execution_hashes()


def _active_execution_identity() -> dict[str, Any]:
    """Return the exact registered runner identity used by any V6 run."""
    return {
        "modal_app": MODAL_APP_NAME,
        "volume": VOLUME_NAME,
        "gpu_type": GPU_TYPE,
        "max_parallel_gpu_workers": MAX_GPU_CONTAINERS,
        "global_modal_gpu_limit": GLOBAL_MODAL_GPU_LIMIT,
        "gpu_app_overlap_policy": GPU_APP_OVERLAP_POLICY,
        "backend_fallback_policy": BACKEND_FALLBACK_POLICY,
        "infrastructure_retry_policy": INFRASTRUCTURE_RETRY_POLICY,
        **_expected_execution_hashes(),
    }


def registration_horizon(registration: dict[str, Any]) -> int:
    updates = registration.get("fixed_updates")
    if (
        isinstance(updates, bool)
        or not isinstance(updates, int)
        or not 1 <= updates <= MAX_REGISTERED_UPDATES
    ):
        raise ProtocolError(
            "final registration must explicitly set fixed_updates in [1, 25]"
        )
    final = registration.get("final_collection")
    if not isinstance(final, dict) or final.get("terminal_adapter_step") != updates:
        raise ProtocolError(
            "final_collection.terminal_adapter_step must explicitly equal fixed_updates"
        )
    return updates


def registration_template() -> dict[str, Any]:
    """Return an intentionally incomplete template; it cannot launch V6."""
    return {
        "protocol": REGISTRATION_PROTOCOL,
        "claim_protocol": PROTOCOL,
        "frozen_at_utc": None,
        "lineage": {
            "emotional_only_decision_sha256": EMOTIONAL_DECISION_SHA256,
            "v4_closeout_sha256": V4_CLOSEOUT_SHA256,
            "v5_scientific_registration_sha256": V5_REGISTRATION_SHA256,
            "v5_infrastructure_amendment1_sha256": (
                V5_INFRASTRUCTURE_AMENDMENT1_SHA256
            ),
            "v6_prelaunch_correction1_sha256": PRELAUNCH_CORRECTION1_SHA256,
            "v6_prelaunch_correction2_sha256": PRELAUNCH_CORRECTION2_SHA256,
        },
        "supersedes_registration": {
            "path": str(SUPERSEDED_V2_REGISTRATION_PATH.relative_to(REPO)),
            "sha256": SUPERSEDED_V2_REGISTRATION_SHA256,
        },
        "prelaunch_correction": {
            "path": str(PRELAUNCH_CORRECTION2_PATH.relative_to(REPO)),
            "sha256": PRELAUNCH_CORRECTION2_SHA256,
        },
        "conditional_launch_predicate": CONDITIONAL_LAUNCH_PREDICATE,
        "operator_knowledge_boundary": CORRECTION2_OPERATOR_KNOWLEDGE_BOUNDARY,
        "selection_closeout": {
            "path": str(SELECTION_CLOSEOUT_PATH.relative_to(REPO)),
            "sha256": SELECTION_CLOSEOUT_SHA256,
        },
        "selected_recipe_lock": {
            "path": str(DEFAULT_RECIPE_LOCK_PATH.relative_to(REPO)),
            "sha256": None,
        },
        "split": SPLIT_REGISTRATION,
        "seeds": list(SEEDS),
        "fixed_updates": None,
        "curve_gate": {
            "steps": None,
            "criterion": CURVE_CRITERION,
        },
        "matched_control": MATCHED_CONTROL_RULE,
        "analysis": ANALYSIS_REGISTRATION,
        "acceptance": ACCEPTANCE_REGISTRATION,
        "final_collection": {
            "labels": list(FINAL_LABELS),
            "terminal_adapter_step": None,
            "one_immutable_collection": True,
        },
        "wandb": {
            "entity": WANDB_ENTITY,
            "project": WANDB_PROJECT,
            "group": WANDB_GROUP,
            "run_prefix": WANDB_RUN_PREFIX,
            "run_ids": WANDB_RUN_IDS,
            "tags": [
                "confirmatory-v6",
                "emotional-j-lens",
                "celebration-taper",
                "conditional-on-v5-curve-failure",
            ],
        },
        "execution": {
            "modal_app": MODAL_APP_NAME,
            "volume": VOLUME_NAME,
            "gpu_type": GPU_TYPE,
            "max_parallel_gpu_workers": MAX_GPU_CONTAINERS,
            "global_modal_gpu_limit": GLOBAL_MODAL_GPU_LIMIT,
            "gpu_app_overlap_policy": GPU_APP_OVERLAP_POLICY,
            "backend_fallback_policy": BACKEND_FALLBACK_POLICY,
            "infrastructure_retry_policy": INFRASTRUCTURE_RETRY_POLICY,
            **_registered_execution_hashes(),
        },
        "outcome_status_at_freeze": V3_OUTCOME_STATUS_AT_FREEZE,
    }


def recipe_lock_template() -> dict[str, Any]:
    return {
        "protocol": RECIPE_LOCK_PROTOCOL,
        "frozen_at_utc": None,
        "emotional_only": True,
        "selection_provenance": {
            "source_resolved_config": {
                "path": None,
                "sha256": CELEBRATION_SOURCE_CONFIG_SHA256,
            },
            "source_calibration": {
                "path": None,
                "sha256": CELEBRATION_CALIBRATION_SHA256,
            },
            "source_closeout": {
                "path": str(SELECTION_CLOSEOUT_PATH.relative_to(REPO)),
                "sha256": SELECTION_CLOSEOUT_SHA256,
            },
            "source_evidence_sha256": CELEBRATION_SELECTION_EVIDENCE_SHA256,
            "declared_transformations": CELEBRATION_DECLARED_TRANSFORMATIONS,
            "operator_knowledge_boundary": OPERATOR_KNOWLEDGE_BOUNDARY,
        },
        "resolved_training_config": None,
        "artifact_sha256": {"lens": None, "calibration": None},
    }


def _load_registration() -> tuple[dict[str, Any], str]:
    if not REGISTRATION_PATH.is_file():
        raise ProtocolError(
            f"final emotional V6 registration is absent: {REGISTRATION_PATH}; "
            "freeze the selected recipe and explicit curve nodes before prepare"
        )
    registration = json.loads(REGISTRATION_PATH.read_text())
    if not isinstance(registration, dict):
        raise ProtocolError("V6 registration must be a JSON object")
    return registration, sha256_file(REGISTRATION_PATH)


def _validate_archive_lineage(registration: dict[str, Any]) -> None:
    expected_files = {
        EMOTIONAL_DECISION_PATH: EMOTIONAL_DECISION_SHA256,
        V4_CLOSEOUT_PATH: V4_CLOSEOUT_SHA256,
        V5_REGISTRATION_PATH: V5_REGISTRATION_SHA256,
        V5_INFRASTRUCTURE_AMENDMENT1_PATH: (
            V5_INFRASTRUCTURE_AMENDMENT1_SHA256
        ),
        SELECTION_CLOSEOUT_PATH: SELECTION_CLOSEOUT_SHA256,
        SUPERSEDED_V1_REGISTRATION_PATH: SUPERSEDED_V1_REGISTRATION_SHA256,
        PRELAUNCH_CORRECTION1_PATH: PRELAUNCH_CORRECTION1_SHA256,
        SUPERSEDED_V2_REGISTRATION_PATH: SUPERSEDED_V2_REGISTRATION_SHA256,
        PRELAUNCH_CORRECTION2_PATH: PRELAUNCH_CORRECTION2_SHA256,
    }
    for path, expected in expected_files.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise ProtocolError(f"predecessor archive is absent or changed: {path}")
    expected_lineage = {
        "emotional_only_decision_sha256": EMOTIONAL_DECISION_SHA256,
        "v4_closeout_sha256": V4_CLOSEOUT_SHA256,
        "v5_scientific_registration_sha256": V5_REGISTRATION_SHA256,
        "v5_infrastructure_amendment1_sha256": (
            V5_INFRASTRUCTURE_AMENDMENT1_SHA256
        ),
        "v6_prelaunch_correction1_sha256": PRELAUNCH_CORRECTION1_SHA256,
        "v6_prelaunch_correction2_sha256": PRELAUNCH_CORRECTION2_SHA256,
    }
    if registration.get("lineage") != expected_lineage:
        raise ProtocolError("registration does not pin the emotional-only lineage")
    if registration.get("conditional_launch_predicate") != (
        CONDITIONAL_LAUNCH_PREDICATE
    ):
        raise ProtocolError("registration changed the conditional V5 launch predicate")
    if registration.get("operator_knowledge_boundary") != (
        CORRECTION2_OPERATOR_KNOWLEDGE_BOUNDARY
    ):
        raise ProtocolError("registration changed the operator knowledge boundary")
    if registration.get("supersedes_registration") != {
        "path": str(SUPERSEDED_V2_REGISTRATION_PATH.relative_to(REPO)),
        "sha256": SUPERSEDED_V2_REGISTRATION_SHA256,
    }:
        raise ProtocolError("V3 registration changed its superseded V2 identity")
    if registration.get("prelaunch_correction") != {
        "path": str(PRELAUNCH_CORRECTION2_PATH.relative_to(REPO)),
        "sha256": PRELAUNCH_CORRECTION2_SHA256,
    }:
        raise ProtocolError("V3 registration changed prelaunch correction 2")
    v1 = json.loads(SUPERSEDED_V1_REGISTRATION_PATH.read_text())
    v2 = json.loads(SUPERSEDED_V2_REGISTRATION_PATH.read_text())
    correction1 = json.loads(PRELAUNCH_CORRECTION1_PATH.read_text())
    correction2 = json.loads(PRELAUNCH_CORRECTION2_PATH.read_text())
    projection_fields = correction2.get("scientific_projection", {}).get("fields")
    if (
        not isinstance(v1, dict)
        or not isinstance(v2, dict)
        or not isinstance(projection_fields, list)
        or correction1.get("scientific_projection", {}).get("fields")
        != projection_fields
    ):
        raise ProtocolError("prelaunch corrections lack one scientific projection")
    v1_projection = {field: v1.get(field) for field in projection_fields}
    v2_projection = {field: v2.get(field) for field in projection_fields}
    v3_projection = {field: registration.get(field) for field in projection_fields}
    if (
        canonical_sha256(v1_projection)
        != SUPERSEDED_SCIENTIFIC_PROJECTION_SHA256
        or correction1["scientific_projection"].get(
            "superseded_registration_canonical_sha256"
        )
        != SUPERSEDED_SCIENTIFIC_PROJECTION_SHA256
        or correction2["scientific_projection"].get(
            "superseded_registration_canonical_sha256"
        )
        != SUPERSEDED_SCIENTIFIC_PROJECTION_SHA256
        or v2_projection != v1_projection
        or v3_projection != v1_projection
    ):
        raise ProtocolError("V3 changed a prohibited scientific/W&B V1/V2 field")
    selection_identity = registration.get("selection_closeout")
    if selection_identity != registration_template()["selection_closeout"]:
        raise ProtocolError("registration changed the V6 selection closeout identity")
    _load_tracked_pinned_json(
        selection_identity,
        "selection_closeout",
    )


def _validate_registration_shape(registration: dict[str, Any]) -> None:
    if (
        registration.get("protocol") != REGISTRATION_PROTOCOL
        or registration.get("claim_protocol") != PROTOCOL
    ):
        raise ProtocolError("wrong V6 registration protocol")
    if not isinstance(registration.get("frozen_at_utc"), str):
        raise ProtocolError("registration must record its freeze time")
    if registration.get("split") != SPLIT_REGISTRATION:
        raise ProtocolError("registration changed the untouched V6 data split")
    if registration.get("seeds") != list(SEEDS):
        raise ProtocolError("registration must use seeds 176 through 183 exactly")
    fixed_updates = registration_horizon(registration)
    if fixed_updates != 10:
        raise ProtocolError("V6 registration must freeze the step-10 adapter")
    if registration.get("matched_control") != MATCHED_CONTROL_RULE:
        raise ProtocolError("registration changed the mechanical matched-control rule")
    if registration.get("analysis") != ANALYSIS_REGISTRATION:
        raise ProtocolError("registration changed the frozen final estimands/bootstrap")
    if registration.get("acceptance") != ACCEPTANCE_REGISTRATION:
        raise ProtocolError("registration changed the frozen significance thresholds")
    expected_final = {
        "labels": list(FINAL_LABELS),
        "terminal_adapter_step": fixed_updates,
        "one_immutable_collection": True,
    }
    if registration.get("final_collection") != expected_final:
        raise ProtocolError("registration must freeze one exact 17-label final collection")
    curve = registration.get("curve_gate")
    if (
        not isinstance(curve, dict)
        or curve.get("criterion") != CURVE_CRITERION
        or curve.get("steps") != [0, 4, 6, 10]
    ):
        raise ProtocolError("registration changed the requested curve criterion")
    execution = registration.get("execution")
    expected_execution = {
        "modal_app": MODAL_APP_NAME,
        "volume": VOLUME_NAME,
        "gpu_type": GPU_TYPE,
        "max_parallel_gpu_workers": MAX_GPU_CONTAINERS,
        "global_modal_gpu_limit": GLOBAL_MODAL_GPU_LIMIT,
        "gpu_app_overlap_policy": GPU_APP_OVERLAP_POLICY,
        "backend_fallback_policy": BACKEND_FALLBACK_POLICY,
        "infrastructure_retry_policy": INFRASTRUCTURE_RETRY_POLICY,
        **_registered_execution_hashes(),
    }
    if execution != expected_execution:
        raise ProtocolError("registration does not byte-pin this exact V6 runner")
    wandb = registration.get("wandb")
    if not isinstance(wandb, dict):
        raise ProtocolError("registration must freeze W&B identities")
    if wandb != registration_template()["wandb"]:
        raise ProtocolError("registration changed the exact V6 W&B identities")
    if len(set(wandb["run_ids"].values())) != 16:
        raise ProtocolError("registration W&B run IDs are not unique")
    if any(
        not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id)
        for run_id in wandb["run_ids"].values()
    ):
        raise ProtocolError("registration W&B run ID contains unsupported characters")
    if registration.get("outcome_status_at_freeze") != V3_OUTCOME_STATUS_AT_FREEZE:
        raise ProtocolError("registration must freeze before any V6 outcome inspection")
    _validate_archive_lineage(registration)


def _validate_recipe_shape(
    recipe: dict[str, Any], expected_updates: int | None = None
) -> tuple[list[str], list[dict[str, Any]]]:
    required = {
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "dataset_revision": DATASET_REVISION,
        "reward_type": "jlens",
        "train_examples": 1000,
        "validation_examples": 400,
        "validation_batch_size": 64,
        "validation_source": "train",
        "validation_observational_only": True,
        "require_clean_repository": True,
        "early_stopping_patience": None,
        "save_total_limit": 1,
        "mask_target_tokens": True,
        "wandb_mode": "online",
    }
    for key, expected in required.items():
        if recipe.get(key) != expected:
            raise ProtocolError(
                f"selected recipe {key!r} is {recipe.get(key)!r}, expected {expected!r}"
            )
    if "base" in recipe:
        raise ProtocolError("selected recipe lock must contain a fully resolved config")
    updates = recipe.get("updates")
    if (
        isinstance(updates, bool)
        or not isinstance(updates, int)
        or not 1 <= updates <= MAX_REGISTERED_UPDATES
        or recipe.get("save_every") != updates
    ):
        raise ProtocolError(
            "selected recipe must explicitly use one fixed horizon in updates/save_every"
        )
    if expected_updates is not None and updates != expected_updates:
        raise ProtocolError(
            "selected recipe horizon does not match the final registration"
        )
    validation_steps = recipe.get("validation_steps")
    if (
        not isinstance(validation_steps, list)
        or any(isinstance(step, bool) or not isinstance(step, int) for step in validation_steps)
        or len(validation_steps) != len(set(validation_steps))
        or any(step <= 0 or step > updates for step in validation_steps)
    ):
        raise ProtocolError("selected recipe has an invalid observational schedule")
    words = validate_target_words(recipe.get("target_words"))
    components = validate_score_components(recipe.get("score_components"))
    return words, components


def _load_tracked_pinned_file(
    identity: Any,
    label: str,
    *,
    expected_sha256: str | None = None,
) -> Path:
    if not isinstance(identity, dict) or set(identity) != {"path", "sha256"}:
        raise ProtocolError(f"{label} must have exactly path and sha256")
    path = _repo_path(identity.get("path"), f"{label}.path")
    digest = identity.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ProtocolError(f"{label}.sha256 must be explicit")
    if expected_sha256 is not None and digest != expected_sha256:
        raise ProtocolError(f"{label} does not match its registered SHA-256")
    relative = path.relative_to(REPO).as_posix()
    try:
        tracked = git("ls-files", "--error-unmatch", relative)
    except subprocess.CalledProcessError as error:
        raise ProtocolError(f"{label} must be committed before registration") from error
    if tracked != relative or not path.is_file() or sha256_file(path) != digest:
        raise ProtocolError(f"{label} committed bytes changed")
    return path


def _load_tracked_pinned_json(
    identity: Any,
    label: str,
    *,
    expected_sha256: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    path = _load_tracked_pinned_file(
        identity, label, expected_sha256=expected_sha256
    )
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ProtocolError(f"{label} must be a JSON object")
    return path, payload


def verify_v5_launch_predicate() -> dict[str, Any]:
    """Prove that V5 failed its curve gate before touching its sealed final.

    The selecting agent did not inspect V5 outcomes.  This verifier validates
    the later exact pinned terminal bytes, including the failed curve record,
    but does not use any outcome value to select or change V6 science.
    """
    if not V5_TERMINAL_CLOSEOUT_PATH.is_file():
        raise ProtocolError(
            "V6 is cancelled/inert until a committed V5 curve-failed terminal "
            "closeout proves that all sealed-final outcomes remain unopened"
        )
    relative = V5_TERMINAL_CLOSEOUT_PATH.relative_to(REPO).as_posix()
    try:
        tracked = git("ls-files", "--error-unmatch", relative)
    except subprocess.CalledProcessError as error:
        raise ProtocolError("V5 terminal closeout must be committed before V6") from error
    if tracked != relative:
        raise ProtocolError("V5 terminal closeout Git identity is ambiguous")
    closeout = json.loads(V5_TERMINAL_CLOSEOUT_PATH.read_text())
    if not isinstance(closeout, dict):
        raise ProtocolError("V5 terminal closeout must be a JSON object")
    expected = {
        "protocol": CONDITIONAL_LAUNCH_PREDICATE["required_protocol"],
        "v5_registration_sha256": V5_REGISTRATION_SHA256,
        "v5_infrastructure_amendment1_sha256": (
            V5_INFRASTRUCTURE_AMENDMENT1_SHA256
        ),
        "v5_sealed_final_manifest_sha256": V5_FINAL_SHA256,
        "v5_sealed_final_sorted_set_sha256": V5_FINAL_SET_SHA256,
        "terminal_stage": CONDITIONAL_LAUNCH_PREDICATE[
            "required_terminal_stage"
        ],
        "final_unlocked_present": False,
        "final_collection_present": False,
        "evals_directory_present": False,
        "final_evaluation_labels": [],
        "sealed_comparison_present": False,
        "final_outcomes_unopened": True,
        "operational_identity": CONDITIONAL_LAUNCH_PREDICATE[
            "required_operational_identity"
        ],
    }
    if any(closeout.get(key) != value for key, value in expected.items()):
        raise ProtocolError(
            "V5 did not prove curve_failed before unlock/collection/evaluation; "
            "the conditional V6 registration is cancelled"
        )
    evidence_identities = closeout.get("source_evidence")
    expected_paths = CONDITIONAL_LAUNCH_PREDICATE[
        "required_source_evidence_paths"
    ]
    if not isinstance(evidence_identities, dict) or set(evidence_identities) != set(
        expected_paths
    ):
        raise ProtocolError("V5 closeout must identify every registered source file")
    evidence_payloads: dict[str, dict[str, Any]] = {}
    evidence_hashes: dict[str, str] = {}
    for name, expected_path in expected_paths.items():
        identity = evidence_identities[name]
        expected_sha256 = V5_TERMINAL_EVIDENCE_SHA256[name]
        if identity != {
            "path": expected_path,
            "sha256": expected_sha256,
        }:
            raise ProtocolError(f"V5 {name} evidence identity changed")
        if name == "curve_plot":
            path = _load_tracked_pinned_file(
                identity,
                "v5_terminal_evidence.curve_plot",
                expected_sha256=expected_sha256,
            )
            payload = None
        else:
            path, payload = _load_tracked_pinned_json(
                identity,
                f"v5_terminal_evidence.{name}",
                expected_sha256=expected_sha256,
            )
        if path != V5_TERMINAL_EVIDENCE_PATHS[name]:
            raise ProtocolError(f"V5 {name} resolved outside its registered path")
        if payload is not None:
            evidence_payloads[name] = payload
        evidence_hashes[name] = sha256_file(path)

    claim = evidence_payloads["attempt_claim"]
    preflight = claim.get("operational_preflight")
    if (
        claim.get("claim_id") != V5_CLAIM_ID
        or claim.get("protocol") != V5_PROTOCOL
        or claim.get("registration_sha256") != V5_REGISTRATION_SHA256
        or claim.get("recipe_lock_sha256") != V5_RECIPE_LOCK_SHA256
        or claim.get("global_modal_gpu_limit") != 1
        or claim.get("gpu_app_overlap_policy") != V5_GPU_APP_OVERLAP_POLICY
        or claim.get("git_commit") != V5_CLAIM_GIT_COMMIT
        or not isinstance(preflight, dict)
        or preflight.get("exclusive_gpu_confirmation")
        != "confirmed-no-other-modal-gpu-app-running"
        or preflight.get("active_other_modal_apps") != []
        or preflight.get("global_modal_gpu_limit") != 1
    ):
        raise ProtocolError("tracked V5 attempt claim is not the exact amended claim")

    receipt = evidence_payloads["launch_receipt"]
    expected_receipt = {
        "app_id": V5_MODAL_APP_ID,
        "claim_id": V5_CLAIM_ID,
        "function_call_id": V5_MODAL_FUNCTION_CALL_ID,
        "global_modal_gpu_limit": 1,
        "gpu_app_overlap_policy": V5_GPU_APP_OVERLAP_POLICY,
        "gpu_type": "L40S",
        "max_parallel_gpu_workers": 1,
        "modal_app": V5_MODAL_APP_NAME,
        "receipt_status": "present",
        "submitted_at_utc": V5_LAUNCH_SUBMITTED_AT_UTC,
        "volume": V5_VOLUME_NAME,
    }
    if receipt != expected_receipt:
        raise ProtocolError("tracked V5 launch receipt changed from exact Volume-B call")

    curve = evidence_payloads["curve_gate"]
    status = evidence_payloads["attempt_status"]
    status_curve = status.get("curve")
    if not isinstance(status_curve, dict):
        raise ProtocolError("tracked V5 terminal status does not bind its curve gate")
    status_curve_payload = dict(status_curve)
    curve_returncode = status_curve_payload.pop("returncode", None)
    if (
        status.get("claim_id") != V5_CLAIM_ID
        or status.get("stage") != "curve_failed"
        or curve_returncode != 2
        or status_curve_payload != curve
        or curve.get("protocol") != V5_PROTOCOL
        or curve.get("git_commit") != claim["git_commit"]
        or curve.get("registration_sha256") != V5_REGISTRATION_SHA256
        or curve.get("criterion") != CURVE_CRITERION
        or curve.get("predeclared_steps") != [0, 2, 4, 6]
        or curve.get("n_seeds") != 8
        or curve.get("examples_per_seed") != 400
        or curve.get("passed") is not False
        or curve.get("curve_plot", {}).get("sha256")
        != evidence_hashes["curve_plot"]
        or any(
            key in status
            for key in ("final_collection_id", "final_labels", "acceptance")
        )
    ):
        raise ProtocolError("tracked V5 status/curve bytes do not prove curve_failed")

    root_inventory = evidence_payloads["root_inventory"]
    evidence_inventory = evidence_payloads["evidence_inventory"]
    for payload, scope in (
        (root_inventory, "volume_root"),
        (evidence_inventory, "evidence_directory"),
    ):
        entries = payload.get("entries")
        if (
            payload.get("protocol")
            != "j-lens-rl-confirmatory-v5-emotional-terminal-inventory-v1"
            or payload.get("claim_id") != V5_CLAIM_ID
            or payload.get("volume") != V5_VOLUME_NAME
            or payload.get("scope") != scope
            or not isinstance(entries, list)
            or entries != sorted(set(entries))
            or any(
                not isinstance(item, str) or not item or "/" in item
                for item in entries
            )
        ):
            raise ProtocolError(f"tracked V5 {scope} inventory is malformed")
    root_entries = root_inventory["entries"]
    evidence_entries = evidence_inventory["entries"]
    required_root = {
        "attempt_claim.json",
        "attempt_status.json",
        "evidence",
        "launch_receipt.json",
        "manifests",
        "protocol_state.json",
        "reproducibility",
        "runs",
    }
    required_evidence_entries = {
        "curve.png",
        "curve_gate.json",
        "durable_export_plan.json",
        "evidence_bundle_inventory.json",
        "git_closeout_candidate.json",
    }
    forbidden_root = {"final_unlocked.json", "final_collection.json", "evals"}
    forbidden_evidence = {
        "acceptance.json",
        "completed_runs.json",
        "sealed_comparison.json",
    }
    if (
        not required_root <= set(root_entries)
        or not required_evidence_entries <= set(evidence_entries)
        or forbidden_root & set(root_entries)
        or forbidden_evidence & set(evidence_entries)
    ):
        raise ProtocolError("V5 closeout inventory contains opened-final artifacts")

    bundle = evidence_payloads["bundle_inventory"]
    bundle_files = bundle.get("files")
    if (
        bundle.get("schema_version") != 1
        or bundle.get("protocol") != V5_PROTOCOL
        or bundle.get("git_commit") != V5_CLAIM_GIT_COMMIT
        or bundle.get("registration_sha256") != V5_REGISTRATION_SHA256
        or bundle.get("recipe_lock_sha256") != V5_RECIPE_LOCK_SHA256
        or bundle.get("bundle_root") != "."
        or bundle.get("terminal_stage") != "curve_failed"
        or bundle.get("sealed_evaluation_file_count") != 0
        or bundle.get("registered_sealed_evaluation_file_count") != 17
        or bundle.get("analysis_inputs")
        != {"base": None, "treatments": {}, "controls": {}}
        or bundle.get("file_count") != 253
        or bundle.get("total_size_bytes") != 763400824
        or not isinstance(bundle_files, dict)
        or len(bundle_files) != 253
    ):
        raise ProtocolError("tracked V5 finalizer bundle inventory is malformed")
    forbidden_bundle_paths = {
        "final_unlocked.json",
        "final_collection.json",
        "evidence/acceptance.json",
        "evidence/completed_runs.json",
        "evidence/sealed_comparison.json",
    }
    if forbidden_bundle_paths & set(bundle_files) or any(
        name.startswith("evals/") for name in bundle_files
    ):
        raise ProtocolError("V5 finalizer bundle inventory contains opened-final artifacts")
    expected_bundle_hashes = {
        "attempt_claim.json": evidence_hashes["attempt_claim"],
        "attempt_status.json": evidence_hashes["attempt_status"],
        "launch_receipt.json": evidence_hashes["launch_receipt"],
        "evidence/curve_gate.json": evidence_hashes["curve_gate"],
        "evidence/curve.png": evidence_hashes["curve_plot"],
        "reproducibility/registration.json": V5_REGISTRATION_SHA256,
        "reproducibility/selected_recipe_lock.json": V5_RECIPE_LOCK_SHA256,
    }
    for relative_path, expected_sha256 in expected_bundle_hashes.items():
        entry = bundle_files.get(relative_path)
        if (
            not isinstance(entry, dict)
            or entry.get("sha256") != expected_sha256
            or not isinstance(entry.get("size_bytes"), int)
            or entry["size_bytes"] <= 0
        ):
            raise ProtocolError(
                f"V5 finalizer bundle inventory does not bind {relative_path}"
            )

    export_receipt = evidence_payloads["durable_export_receipt"]
    if (
        export_receipt.get("schema_version") != 1
        or export_receipt.get("archive_relative_path")
        != f"exports/v5_emotional_evidence_{V5_CLAIM_ID}.zip"
        or export_receipt.get("sha256") != V5_DURABLE_ARCHIVE_SHA256
        or export_receipt.get("size_bytes") != 763510552
        or export_receipt.get("entry_count") != 254
        or export_receipt.get("evidence_inventory_sha256")
        != evidence_hashes["bundle_inventory"]
    ):
        raise ProtocolError("tracked V5 durable-export receipt is malformed")
    return {
        "path": relative,
        "sha256": sha256_file(V5_TERMINAL_CLOSEOUT_PATH),
        "terminal_stage": closeout["terminal_stage"],
        "final_outcomes_unopened": closeout["final_outcomes_unopened"],
        "operational_identity": closeout["operational_identity"],
        "source_evidence_sha256": evidence_hashes,
    }


def expected_selected_celebration_recipe(
    source: dict[str, Any], committed_calibration_path: str
) -> dict[str, Any]:
    """Apply the only outcome-informed source-to-confirmation transformations."""
    required_source = {
        "target_words": ["yay", "great", "success", "nice"],
        "score_components": [
            {
                "aggregation": "mean",
                "end_fraction": 0.75,
                "layer": 8,
                "start_fraction": 0.5,
                "weight": 1.0,
            },
            {
                "aggregation": "mean",
                "end_fraction": 1.0,
                "layer": 8,
                "start_fraction": 0.75,
                "weight": 0.25,
            },
        ],
        "score_stride": 10,
        "learning_rate": 3e-6,
        "lr_scheduler_type": "constant",
        "updates": 25,
        "save_every": 25,
        "validation_steps": [2, 4, 6, 10, 15, 20, 25],
        "seed": 167,
        "output_dir": "/word_explore/runs/celebration_taper",
        "run_name": "word-screen-celebration-tail-taper-seed167",
        "calibration_path": "/word_explore/artifacts/celebration_calibration.json",
        "calibration_sha256": CELEBRATION_CALIBRATION_SHA256,
        "lens_sha256": "178a9671cbf41882135807bde59b828e36c6f8f98b32c809ea3346860aad10dc",
    }
    if any(source.get(key) != value for key, value in required_source.items()):
        raise ProtocolError("selected source is not the exact celebration-taper recipe")
    result = dict(source)
    result.update(
        {
            "updates": 10,
            "save_every": 10,
            "validation_steps": [4, 6, 10],
            "calibration_path": committed_calibration_path,
        }
    )
    return result


def _validate_celebration_selection_provenance(
    value: Any, recipe: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "source_resolved_config",
        "source_calibration",
        "source_closeout",
        "source_evidence_sha256",
        "declared_transformations",
        "operator_knowledge_boundary",
    }:
        raise ProtocolError("selected celebration provenance has unexpected fields")
    if value.get("source_evidence_sha256") != CELEBRATION_SELECTION_EVIDENCE_SHA256:
        raise ProtocolError("selected celebration evidence identities changed")
    if value.get("declared_transformations") != CELEBRATION_DECLARED_TRANSFORMATIONS:
        raise ProtocolError("selected celebration transformations changed")
    if value.get("operator_knowledge_boundary") != OPERATOR_KNOWLEDGE_BOUNDARY:
        raise ProtocolError("selected celebration knowledge boundary changed")
    config_path, source = _load_tracked_pinned_json(
        value["source_resolved_config"],
        "selection_provenance.source_resolved_config",
        expected_sha256=CELEBRATION_SOURCE_CONFIG_SHA256,
    )
    calibration_path, calibration = _load_tracked_pinned_json(
        value["source_calibration"],
        "selection_provenance.source_calibration",
        expected_sha256=CELEBRATION_CALIBRATION_SHA256,
    )
    closeout_path, closeout = _load_tracked_pinned_json(
        value["source_closeout"],
        "selection_provenance.source_closeout",
    )
    serialized_closeout = json.dumps(closeout, sort_keys=True)
    if any(
        digest not in serialized_closeout
        for digest in CELEBRATION_SELECTION_EVIDENCE_SHA256.values()
    ) or "celebration" not in serialized_closeout.lower():
        raise ProtocolError("source closeout does not bind the celebration evidence")
    if closeout.get("protocol") != SELECTION_CLOSEOUT_PROTOCOL:
        raise ProtocolError("wrong V6 celebration selection-closeout protocol")
    if closeout.get("operator_knowledge_boundary") != OPERATOR_KNOWLEDGE_BOUNDARY:
        raise ProtocolError("selection closeout changed the knowledge boundary")
    if calibration.get("target_words") != ["yay", "great", "success", "nice"] or (
        calibration.get("lens_sha256")
        != "178a9671cbf41882135807bde59b828e36c6f8f98b32c809ea3346860aad10dc"
    ):
        raise ProtocolError("committed calibration is not the celebration calibration")
    expected = expected_selected_celebration_recipe(
        source, calibration_path.relative_to(REPO).as_posix()
    )
    if recipe != expected:
        raise ProtocolError(
            "selected recipe is not the exact declared 25-to-10 celebration transformation"
        )
    return {
        "source_resolved_config": {
            "path": config_path.relative_to(REPO).as_posix(),
            "sha256": CELEBRATION_SOURCE_CONFIG_SHA256,
        },
        "source_calibration": {
            "path": calibration_path.relative_to(REPO).as_posix(),
            "sha256": CELEBRATION_CALIBRATION_SHA256,
        },
        "source_closeout": {
            "path": closeout_path.relative_to(REPO).as_posix(),
            "sha256": value["source_closeout"]["sha256"],
        },
    }


def _load_recipe_lock(
    registration: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Path, str]:
    identity = registration.get("selected_recipe_lock")
    if not isinstance(identity, dict):
        raise ProtocolError("registration must byte-pin a selected-recipe lock")
    lock_path = _repo_path(identity.get("path"), "selected_recipe_lock.path")
    expected_hash = identity.get("sha256")
    if not lock_path.is_file() or not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ProtocolError("registered selected-recipe lock is absent or has no SHA-256")
    actual_hash = sha256_file(lock_path)
    if actual_hash != expected_hash:
        raise ProtocolError("selected-recipe lock bytes changed after registration")
    lock = json.loads(lock_path.read_text())
    if not isinstance(lock, dict) or lock.get("protocol") != RECIPE_LOCK_PROTOCOL:
        raise ProtocolError("wrong selected-recipe lock protocol")
    if lock.get("emotional_only") is not True:
        raise ProtocolError("selected-recipe lock is not explicitly emotional-only")
    if not isinstance(lock.get("frozen_at_utc"), str):
        raise ProtocolError("selected-recipe lock must record its freeze time")
    recipe = lock.get("resolved_training_config")
    if not isinstance(recipe, dict):
        raise ProtocolError("selected-recipe lock lacks a fully resolved training config")
    _validate_celebration_selection_provenance(lock.get("selection_provenance"), recipe)
    _validate_recipe_shape(recipe, registration_horizon(registration))
    artifact_hashes = lock.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict):
        raise ProtocolError("selected-recipe lock lacks artifact SHA-256 values")
    expected_artifacts = {
        "lens": recipe.get("lens_sha256"),
        "calibration": recipe.get("calibration_sha256"),
    }
    if artifact_hashes != expected_artifacts or any(
        not isinstance(value, str) or len(value) != 64
        for value in expected_artifacts.values()
    ):
        raise ProtocolError("recipe lock and resolved config disagree on artifact bytes")
    return lock, recipe, lock_path, actual_hash


def _artifact_source_paths(recipe: dict[str, Any]) -> dict[str, Path]:
    return {
        "lens": _repo_path(recipe.get("lens_path"), "recipe.lens_path"),
        "calibration": _repo_path(
            recipe.get("calibration_path"), "recipe.calibration_path"
        ),
    }


def validate_artifacts(recipe: dict[str, Any]) -> dict[str, Any]:
    words, _ = _validate_recipe_shape(recipe)
    paths = _artifact_source_paths(recipe)
    calibration_relative = paths["calibration"].relative_to(REPO).as_posix()
    try:
        tracked_calibration = git("ls-files", "--error-unmatch", calibration_relative)
    except subprocess.CalledProcessError as error:
        raise ProtocolError(
            "selected calibration must be committed so the registered recipe is replayable"
        ) from error
    if tracked_calibration != calibration_relative:
        raise ProtocolError("selected calibration Git identity is ambiguous")
    expected = {
        "lens": recipe["lens_sha256"],
        "calibration": recipe["calibration_sha256"],
    }
    actual = {}
    for name, path in paths.items():
        if not path.is_file():
            raise ProtocolError(f"missing selected {name} artifact: {path}")
        actual[name] = sha256_file(path)
    if actual != expected:
        raise ProtocolError(f"selected artifact hash mismatch: {actual!r}")

    try:
        from jlens import JacobianLens
        from transformers import AutoTokenizer

        from jlens_rl.reward import single_token_ids, validate_calibration_metadata
    except ImportError as error:
        raise ProtocolError("run V6 preparation with the project virtualenv") from error

    lens = JacobianLens.load(str(paths["lens"]))
    layers = [int(layer) for layer in lens.source_layers]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    token_ids = single_token_ids(tokenizer, words)
    calibration = json.loads(paths["calibration"].read_text())
    try:
        validate_calibration_metadata(
            calibration,
            target_words=words,
            token_ids=token_ids,
            lens_layers=layers,
            expected_model=MODEL_NAME,
            expected_model_revision=MODEL_REVISION,
            expected_lens_sha256=actual["lens"],
        )
    except (TypeError, ValueError) as error:
        raise ProtocolError(f"selected calibration metadata is invalid: {error}") from error
    component_layers = {int(item["layer"]) for item in recipe["score_components"]}
    if not component_layers <= set(layers):
        raise ProtocolError("selected score component uses a layer absent from the lens")
    return {
        "source_path": {
            name: str(path.relative_to(REPO)) for name, path in paths.items()
        },
        "sha256": actual,
        "target_words": words,
        "token_ids": token_ids,
        "lens_layers": layers,
    }


def _run_label(condition: str, seed: int) -> str:
    if condition not in REQUIRED_CONDITIONS or seed not in SEEDS:
        raise ProtocolError("run label is outside the frozen V6 protocol")
    return f"{condition}_seed{seed}"


def config_path(condition: str, seed: int) -> Path:
    return CONFIG_DIR / f"{_run_label(condition, seed)}.json"


def run_dir(condition: str, seed: int) -> Path:
    return RUN_DIR / _run_label(condition, seed)


def _wandb_identity(
    registration: dict[str, Any], registration_sha256: str, label: str
) -> dict[str, Any]:
    wandb = registration["wandb"]
    del registration_sha256  # identities are explicit registration fields
    run_id = wandb["run_ids"].get(label)
    if not isinstance(run_id, str) or run_id != WANDB_RUN_IDS.get(label):
        raise ProtocolError(f"W&B identity is missing or changed for {label}")
    return {
        "wandb_entity": wandb["entity"],
        "wandb_project": wandb["project"],
        "wandb_group": wandb["group"],
        "wandb_tags": list(wandb["tags"]),
        "wandb_run_id": run_id,
        "wandb_url": (
            f"https://wandb.ai/{wandb['entity']}/{wandb['project']}/runs/{run_id}"
        ),
        "wandb_resume": "never",
        "run_name": run_id,
    }


def generated_configs(
    registration: dict[str, Any],
    registration_sha256: str,
    recipe: dict[str, Any],
    recipe_lock_sha256: str,
) -> dict[str, dict[str, Any]]:
    fixed_updates = registration_horizon(registration)
    words, components = _validate_recipe_shape(recipe, fixed_updates)
    validate_curve_steps(
        registration["curve_gate"]["steps"],
        recipe["validation_steps"],
        fixed_updates,
    )
    base = dict(recipe)
    schema = metric_schema(words, fixed_updates, components)
    base.update(
        {
            "validation_indices_path": ".confirmatory/v6/manifests/curve_indices.json",
            "reserved_train_indices_path": ".confirmatory/v6/manifests/train_exclusions.json",
            "validation_examples": 400,
            "lens_path": ".confirmatory/v6/frozen_artifacts/lens.pt",
            "calibration_path": ".confirmatory/v6/frozen_artifacts/calibration.json",
            "lens_sha256": recipe["lens_sha256"],
            "calibration_sha256": recipe["calibration_sha256"],
            "expected_lens_sha256": recipe["lens_sha256"],
            "expected_calibration_sha256": recipe["calibration_sha256"],
            "target_words": words,
            "score_components": components,
            "updates": fixed_updates,
            "save_every": fixed_updates,
            "registration_sha256": registration_sha256,
            "recipe_lock_sha256": recipe_lock_sha256,
            "recipe_sha256": canonical_sha256(recipe),
            "metric_schema_path": ".confirmatory/v6/reproducibility/metric_schema.json",
            "metric_schema_sha256": serialized_json_sha256(schema),
            "curve_manifest_sha256": V6_CURVE_SHA256,
            "train_exclusions_manifest_sha256": TRAIN_EXCLUSIONS_SHA256,
            "registered_code_sha256": _active_execution_identity(),
            "evidence_eligibility": "original_registered_confirmatory_attempt",
        }
    )
    result: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        treatment_label = _run_label("jlens", seed)
        treatment = dict(base)
        treatment.update(
            {
                "seed": seed,
                "output_dir": f".confirmatory/v6/runs/{treatment_label}",
                **_wandb_identity(registration, registration_sha256, treatment_label),
                "registered_command": [
                    "python",
                    "-m",
                    "jlens_rl.train",
                    "--config",
                    f".confirmatory/v6/configs/{treatment_label}.json",
                    "--wandb-mode",
                    "online",
                ],
            }
        )
        result[treatment_label] = treatment

        control_label = _run_label("signflip", seed)
        control = dict(treatment)
        control.update(
            {
                "score_components": negate_score_components(components),
                "output_dir": f".confirmatory/v6/runs/{control_label}",
                **_wandb_identity(registration, registration_sha256, control_label),
                "registered_command": [
                    "python",
                    "-m",
                    "jlens_rl.train",
                    "--config",
                    f".confirmatory/v6/configs/{control_label}.json",
                    "--wandb-mode",
                    "online",
                ],
            }
        )
        result[control_label] = control

    sealed = dict(base)
    sealed.update(
        {
            "seed": SEEDS[0],
            "evaluation_seed": 0,
            "evaluation_source": "train",
            "evaluation_indices_path": (
                ".confirmatory/v6/manifests/sealed_final_indices.json"
            ),
            "validation_examples": 900,
            "min_new_tokens": 0,
            "max_new_tokens": 256,
            "output_dir": ".confirmatory/v6/evaluation_config_unused",
            "run_name": f"{registration['wandb']['run_prefix']}-sealed-evaluation",
        }
    )
    for key in ("wandb_run_id", "wandb_group", "wandb_tags", "wandb_resume"):
        sealed.pop(key, None)
    result["sealed_eval"] = sealed
    return result


def _verify_source_manifests() -> tuple[list[int], list[int], list[int]]:
    expected = {
        V4_PARENT_PATH: V4_PARENT_SHA256,
        V4_RESERVE_PATH: RESERVE_SHA256,
        V4_TRAIN_EXCLUSIONS_PATH: TRAIN_EXCLUSIONS_SHA256,
    }
    for path, digest in expected.items():
        if not path.is_file() or sha256_file(path) != digest:
            raise ProtocolError(f"V6 source manifest is absent or changed: {path}")
    v4_parent = load_indices(V4_PARENT_PATH)
    reserve = load_indices(V4_RESERVE_PATH)
    exclusions = load_indices(V4_TRAIN_EXCLUSIONS_PATH)
    if canonical_sha256(sorted(v4_parent)) != V4_PARENT_SET_SHA256:
        raise ProtocolError("V4 sealed parent set identity changed")
    _, v5_parent = reconstruct_v5_split(v4_parent)
    if (
        serialized_json_sha256(manifest_payload(v5_parent)) != V5_FINAL_SHA256
        or canonical_sha256(sorted(v5_parent)) != V5_FINAL_SET_SHA256
    ):
        raise ProtocolError("reconstructed V5 sealed-final parent identity changed")
    if len(reserve) != 64 or set(v5_parent) & set(reserve):
        raise ProtocolError("untouched reserve is malformed or overlaps the V6 parent")
    if not (set(v5_parent) | set(reserve)) <= set(exclusions):
        raise ProtocolError("training exclusions no longer protect all V6 outcomes")
    return v5_parent, reserve, exclusions


def _verify_v6_allocations(curve: Sequence[int], final: Sequence[int]) -> None:
    if (
        len(curve) != 400
        or len(final) != 900
        or set(curve) & set(final)
        or sha256_file(MANIFEST_DIR / "curve_indices.json") != V6_CURVE_SHA256
        or sha256_file(MANIFEST_DIR / "sealed_final_indices.json") != V6_FINAL_SHA256
        or canonical_sha256(sorted(curve)) != V6_CURVE_SET_SHA256
        or canonical_sha256(sorted(final)) != V6_FINAL_SET_SHA256
    ):
        raise ProtocolError("prepared V6 curve/final allocation changed")


def _validate_frozen_artifacts(
    recipe: dict[str, Any], paths: dict[str, Path]
) -> dict[str, Any]:
    words, _ = _validate_recipe_shape(recipe)
    expected = {
        "lens": recipe["lens_sha256"],
        "calibration": recipe["calibration_sha256"],
    }
    actual: dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise ProtocolError(f"missing frozen {name} artifact: {path}")
        actual[name] = sha256_file(path)
    if actual != expected:
        raise ProtocolError(f"frozen artifact hash mismatch: {actual!r}")

    try:
        from jlens import JacobianLens
        from transformers import AutoTokenizer

        from jlens_rl.reward import single_token_ids, validate_calibration_metadata
    except ImportError as error:
        raise ProtocolError("run V6 verification with the project environment") from error
    lens = JacobianLens.load(str(paths["lens"]))
    layers = [int(layer) for layer in lens.source_layers]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    token_ids = single_token_ids(tokenizer, words)
    calibration = json.loads(paths["calibration"].read_text())
    try:
        validate_calibration_metadata(
            calibration,
            target_words=words,
            token_ids=token_ids,
            lens_layers=layers,
            expected_model=MODEL_NAME,
            expected_model_revision=MODEL_REVISION,
            expected_lens_sha256=actual["lens"],
        )
    except (TypeError, ValueError) as error:
        raise ProtocolError(f"frozen calibration metadata is invalid: {error}") from error
    if not {int(item["layer"]) for item in recipe["score_components"]} <= set(layers):
        raise ProtocolError("selected score component uses a layer absent from the lens")
    return {
        "sha256": actual,
        "target_words": words,
        "token_ids": token_ids,
        "lens_layers": layers,
    }


def _config_files() -> dict[str, Path]:
    result = {
        _run_label(condition, seed): config_path(condition, seed)
        for condition in REQUIRED_CONDITIONS
        for seed in SEEDS
    }
    result["sealed_eval"] = CONFIG_DIR / "sealed_eval.json"
    return result


def _tracked_source_inventory() -> dict[str, Any]:
    names = git("ls-files", "-z").split("\0")
    files = [name for name in names if name and (REPO / name).is_file()]
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "files": {
            name: {
                "sha256": sha256_file(REPO / name),
                "size_bytes": (REPO / name).stat().st_size,
            }
            for name in sorted(files)
        },
    }


def _write_source_snapshot(path: Path, inventory: dict[str, Any]) -> None:
    """Write a deterministic archive of every tracked source file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with zipfile.ZipFile(
        temporary, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for relative in sorted(inventory["files"]):
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, (REPO / relative).read_bytes())
    os.replace(temporary, path)


def _launch_plan(
    registration: dict[str, Any], configs: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    training = {
        label: config["registered_command"]
        for label, config in configs.items()
        if label != "sealed_eval"
    }
    reproduction_training = {
        label: [
            "python",
            "-m",
            "jlens_rl.train",
            "--config",
            f".confirmatory/v6/configs/{label}.json",
            "--reproduction-replay",
            "--output-dir",
            f"REPLACE_WITH_FRESH_OUTPUT_ROOT/{label}",
            "--wandb-mode",
            "disabled",
        ]
        for label in sorted(training)
    }
    evaluations: dict[str, list[str]] = {}
    for label in FINAL_LABELS:
        if label == "base":
            experiment = ".confirmatory/v6/configs/jlens_seed176.json"
            adapter: list[str] = []
        else:
            experiment = f".confirmatory/v6/configs/{label}.json"
            adapter = ["--adapter", f".confirmatory/v6/runs/{label}/final"]
        evaluations[label] = [
            "python",
            "-m",
            "jlens_rl.eval",
            "--config",
            ".confirmatory/v6/configs/sealed_eval.json",
            "--experiment-config",
            experiment,
            "--indices-manifest",
            ".confirmatory/v6/manifests/sealed_final_indices.json",
            "--output-jsonl",
            f".confirmatory/v6/evals/{label}.jsonl",
            "--run-label",
            label,
            "--batch-size",
            "64",
            "--skip-jlens-metric",
            *adapter,
        ]
    return {
        "schema_version": 1,
        "phase_order": [
            "eight treatment training runs",
            "registered treatment curve gate",
            "eight mechanically sign-flipped control runs",
            "one immutable 17-label sealed-final collection",
            "paired semantic and difference-in-differences analysis",
        ],
        "training_commands": training,
        "reproduction_training_commands": reproduction_training,
        "reproduction_evidence_policy": (
            "Replace REPLACE_WITH_FRESH_OUTPUT_ROOT with a new caller-owned directory. "
            "These commands disable W&B, strip the original run identity, and stamp "
            "outputs non_claim_reproduction; they can test reproducibility but can "
            "never be substituted for the original registered confirmatory attempt."
        ),
        "final_evaluation_commands": evaluations,
        "analysis_command": [
            "python",
            "-m",
            "jlens_rl.paired_eval",
            "--base-jsonl",
            ".confirmatory/v6/evals/base.jsonl",
            *[
                item
                for seed in SEEDS
                for item in (
                    "--adapter-jsonl",
                    f".confirmatory/v6/evals/jlens_seed{seed}.jsonl",
                )
            ],
            *[
                item
                for seed in SEEDS
                for item in (
                    "--control-jsonl",
                    f".confirmatory/v6/evals/signflip_seed{seed}.jsonl",
                )
            ],
            "--bootstrap-samples",
            "10000",
            "--seed",
            "0",
            "--confidence",
            "0.95",
            "--output",
            ".confirmatory/v6/evidence/sealed_comparison.json",
        ],
        "wandb": registration["wandb"],
    }


def _verify_reproducibility_files(
    registration: dict[str, Any],
    lock_path: Path,
    recipe: dict[str, Any],
    configs: dict[str, dict[str, Any]],
) -> dict[str, str]:
    expected_names = {
        "registration.json",
        "selected_recipe_lock.json",
        "v5_terminal_closeout.json",
        "metric_schema.json",
        "source_manifest.json",
        "source_snapshot.zip",
        "launch_plan.json",
    }
    actual_names = {
        path.name for path in REPRODUCIBILITY_DIR.iterdir() if path.is_file()
    }
    if actual_names != expected_names:
        raise ProtocolError(
            f"V6 reproducibility inventory changed: {sorted(actual_names)}"
        )
    if (
        (REPRODUCIBILITY_DIR / "registration.json").read_bytes()
        != REGISTRATION_PATH.read_bytes()
        or (REPRODUCIBILITY_DIR / "selected_recipe_lock.json").read_bytes()
        != lock_path.read_bytes()
        or (REPRODUCIBILITY_DIR / "v5_terminal_closeout.json").read_bytes()
        != V5_TERMINAL_CLOSEOUT_PATH.read_bytes()
    ):
        raise ProtocolError("registered decision bytes changed in reproducibility bundle")
    expected_schema = metric_schema(
        recipe["target_words"],
        registration_horizon(registration),
        recipe["score_components"],
    )
    if json.loads((REPRODUCIBILITY_DIR / "metric_schema.json").read_text()) != (
        expected_schema
    ):
        raise ProtocolError("V6 metric schema changed")
    if json.loads((REPRODUCIBILITY_DIR / "launch_plan.json").read_text()) != (
        _launch_plan(registration, configs)
    ):
        raise ProtocolError("V6 replay command plan changed")
    source_inventory = json.loads(
        (REPRODUCIBILITY_DIR / "source_manifest.json").read_text()
    )
    if source_inventory != _tracked_source_inventory():
        raise ProtocolError("V6 tracked-source inventory changed")
    with zipfile.ZipFile(REPRODUCIBILITY_DIR / "source_snapshot.zip") as archive:
        if archive.namelist() != sorted(source_inventory["files"]):
            raise ProtocolError("V6 source snapshot file list changed")
        for name, identity in source_inventory["files"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != identity["sha256"]:
                raise ProtocolError(f"V6 source snapshot content changed: {name}")
    return {
        path.name: sha256_file(path)
        for path in sorted(REPRODUCIBILITY_DIR.iterdir())
        if path.is_file()
    }


def prepare() -> None:
    commit = require_clean_worktree()
    if STATE_DIR.exists():
        raise ProtocolError(f"{STATE_DIR} already exists; V6 preparation is immutable")
    registration, registration_sha256 = _load_registration()
    _validate_registration_shape(registration)
    predecessor_predicate = verify_v5_launch_predicate()
    lock, recipe, lock_path, lock_sha256 = _load_recipe_lock(registration)
    curve_steps = validate_curve_steps(
        registration["curve_gate"]["steps"],
        recipe["validation_steps"],
        registration_horizon(registration),
    )
    source_artifacts = validate_artifacts(recipe)
    parent, reserve, exclusions = _verify_source_manifests()
    curve, final = allocate_v6(parent)

    MANIFEST_DIR.mkdir(parents=True)
    write_json(MANIFEST_DIR / "curve_indices.json", manifest_payload(curve))
    write_json(MANIFEST_DIR / "sealed_final_indices.json", manifest_payload(final))
    write_json(MANIFEST_DIR / "future_reserve_indices.json", manifest_payload(reserve))
    write_json(
        MANIFEST_DIR / "train_exclusions.json", manifest_payload(exclusions)
    )
    _verify_v6_allocations(curve, final)
    if sha256_file(MANIFEST_DIR / "future_reserve_indices.json") != RESERVE_SHA256:
        raise ProtocolError("V6 changed the untouched reserve bytes")
    if sha256_file(MANIFEST_DIR / "train_exclusions.json") != TRAIN_EXCLUSIONS_SHA256:
        raise ProtocolError("V6 changed the protected training exclusions")

    ARTIFACT_DIR.mkdir(parents=True)
    sources = _artifact_source_paths(recipe)
    frozen_paths = {
        "lens": ARTIFACT_DIR / "lens.pt",
        "calibration": ARTIFACT_DIR / "calibration.json",
    }
    for name, destination in frozen_paths.items():
        shutil.copyfile(sources[name], destination)
    frozen_artifacts = _validate_frozen_artifacts(recipe, frozen_paths)

    configs = generated_configs(
        registration, registration_sha256, recipe, lock_sha256
    )
    CONFIG_DIR.mkdir(parents=True)
    for label, config in configs.items():
        path = (
            CONFIG_DIR / "sealed_eval.json"
            if label == "sealed_eval"
            else CONFIG_DIR / f"{label}.json"
        )
        write_json(path, config)
    REPRODUCIBILITY_DIR.mkdir(parents=True)
    schema = metric_schema(
        source_artifacts["target_words"],
        registration_horizon(registration),
        recipe["score_components"],
    )
    write_json(REPRODUCIBILITY_DIR / "metric_schema.json", schema)
    if sha256_file(REPRODUCIBILITY_DIR / "metric_schema.json") != serialized_json_sha256(
        schema
    ):
        raise ProtocolError("metric-schema serialization identity changed")
    shutil.copyfile(REGISTRATION_PATH, REPRODUCIBILITY_DIR / "registration.json")
    shutil.copyfile(lock_path, REPRODUCIBILITY_DIR / "selected_recipe_lock.json")
    shutil.copyfile(
        V5_TERMINAL_CLOSEOUT_PATH,
        REPRODUCIBILITY_DIR / "v5_terminal_closeout.json",
    )
    source_inventory = _tracked_source_inventory()
    write_json(REPRODUCIBILITY_DIR / "source_manifest.json", source_inventory)
    _write_source_snapshot(
        REPRODUCIBILITY_DIR / "source_snapshot.zip", source_inventory
    )
    write_json(
        REPRODUCIBILITY_DIR / "launch_plan.json",
        _launch_plan(registration, configs),
    )
    config_hashes = {
        label: sha256_file(path) for label, path in _config_files().items()
    }
    manifest_hashes = {
        path.name: sha256_file(path) for path in sorted(MANIFEST_DIR.glob("*.json"))
    }
    wandb_identities = {
        label: {
            key: config[key]
            for key in (
                "wandb_project",
                "wandb_entity",
                "wandb_group",
                "wandb_tags",
                "wandb_run_id",
                "wandb_url",
                "wandb_resume",
                "run_name",
            )
        }
        for label, config in configs.items()
        if label != "sealed_eval"
    }
    if len({item["wandb_run_id"] for item in wandb_identities.values()}) != 16:
        raise ProtocolError("generated W&B run IDs are not unique")

    state = {
        "protocol": PROTOCOL,
        "prepared_at_utc": utc_now(),
        "git_commit": commit,
        "registration_path": str(REGISTRATION_PATH.relative_to(REPO)),
        "registration_sha256": registration_sha256,
        "conditional_v5_launch_predicate": predecessor_predicate,
        "active_modal_volume": VOLUME_NAME,
        "recipe_lock_path": str(lock_path.relative_to(REPO)),
        "recipe_lock_sha256": lock_sha256,
        "recipe_lock_protocol": lock["protocol"],
        "recipe_sha256": canonical_sha256(recipe),
        "target_words": source_artifacts["target_words"],
        "target_token_ids": source_artifacts["token_ids"],
        "lens_layers": source_artifacts["lens_layers"],
        "artifact_source_path": source_artifacts["source_path"],
        "artifact_sha256": frozen_artifacts["sha256"],
        "allocation_salt": V6_ALLOCATION_SALT,
        "split": SPLIT_REGISTRATION,
        "index_manifest_sha256": manifest_hashes,
        "seeds": list(SEEDS),
        "fixed_updates": registration_horizon(registration),
        "curve_gate_steps": list(curve_steps),
        "curve_gate_criterion": CURVE_CRITERION,
        "matched_control_rule": MATCHED_CONTROL_RULE,
        "final_labels": list(FINAL_LABELS),
        "config_sha256": config_hashes,
        "wandb_identities": wandb_identities,
        "source_sha256": _expected_execution_hashes(),
        "metric_schema_sha256": sha256_file(
            REPRODUCIBILITY_DIR / "metric_schema.json"
        ),
        "reproducibility_file_sha256": {
            path.name: sha256_file(path)
            for path in sorted(REPRODUCIBILITY_DIR.iterdir())
            if path.is_file()
        },
    }
    write_json(STATE_PATH, state)
    print(json.dumps(state, indent=2, sort_keys=True))


def _verify_prepared_manifests() -> None:
    expected = {
        "curve_indices.json": V6_CURVE_SHA256,
        "sealed_final_indices.json": V6_FINAL_SHA256,
        "future_reserve_indices.json": RESERVE_SHA256,
        "train_exclusions.json": TRAIN_EXCLUSIONS_SHA256,
    }
    for name, digest in expected.items():
        path = MANIFEST_DIR / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ProtocolError(f"prepared V6 manifest changed: {name}")
    curve = load_indices(MANIFEST_DIR / "curve_indices.json")
    final = load_indices(MANIFEST_DIR / "sealed_final_indices.json")
    reserve = load_indices(MANIFEST_DIR / "future_reserve_indices.json")
    exclusions = set(load_indices(MANIFEST_DIR / "train_exclusions.json"))
    _verify_v6_allocations(curve, final)
    if (
        len(reserve) != 64
        or set(curve) & set(final)
        or set(curve) & set(reserve)
        or set(final) & set(reserve)
        or not (set(curve) | set(final) | set(reserve)) <= exclusions
    ):
        raise ProtocolError("prepared V6 manifests overlap or escape training exclusions")


def load_and_verify_state() -> dict[str, Any]:
    commit = require_clean_worktree()
    if not STATE_PATH.is_file():
        raise ProtocolError("V6 is not prepared; run prepare only after final registration")
    state = json.loads(STATE_PATH.read_text())
    registration, registration_sha256 = _load_registration()
    _validate_registration_shape(registration)
    predecessor_predicate = verify_v5_launch_predicate()
    lock, recipe, lock_path, lock_sha256 = _load_recipe_lock(registration)
    curve_steps = validate_curve_steps(
        registration["curve_gate"]["steps"],
        recipe["validation_steps"],
        registration_horizon(registration),
    )
    expected_state = {
        "protocol": PROTOCOL,
        "git_commit": commit,
        "registration_path": str(REGISTRATION_PATH.relative_to(REPO)),
        "registration_sha256": registration_sha256,
        "conditional_v5_launch_predicate": predecessor_predicate,
        "active_modal_volume": VOLUME_NAME,
        "recipe_lock_path": str(lock_path.relative_to(REPO)),
        "recipe_lock_sha256": lock_sha256,
        "recipe_lock_protocol": lock["protocol"],
        "recipe_sha256": canonical_sha256(recipe),
        "allocation_salt": V6_ALLOCATION_SALT,
        "split": SPLIT_REGISTRATION,
        "seeds": list(SEEDS),
        "fixed_updates": registration_horizon(registration),
        "curve_gate_steps": list(curve_steps),
        "curve_gate_criterion": CURVE_CRITERION,
        "matched_control_rule": MATCHED_CONTROL_RULE,
        "final_labels": list(FINAL_LABELS),
        "source_sha256": _expected_execution_hashes(),
    }
    if any(state.get(key) != value for key, value in expected_state.items()):
        raise ProtocolError("prepared state does not match the final V6 registration")
    _verify_prepared_manifests()
    frozen_artifacts = _validate_frozen_artifacts(
        recipe,
        {
            "lens": ARTIFACT_DIR / "lens.pt",
            "calibration": ARTIFACT_DIR / "calibration.json",
        },
    )
    if (
        state.get("artifact_sha256") != frozen_artifacts["sha256"]
        or state.get("target_words") != frozen_artifacts["target_words"]
        or state.get("target_token_ids") != frozen_artifacts["token_ids"]
        or state.get("lens_layers") != frozen_artifacts["lens_layers"]
    ):
        raise ProtocolError("prepared target/calibration identity changed")
    generated = generated_configs(
        registration, registration_sha256, recipe, lock_sha256
    )
    for label, path in _config_files().items():
        if not path.is_file():
            raise ProtocolError(f"prepared config is missing: {path}")
        if json.loads(path.read_text()) != generated[label]:
            raise ProtocolError(f"prepared config content changed: {label}")
    config_hashes = {
        label: sha256_file(path) for label, path in _config_files().items()
    }
    if state.get("config_sha256") != config_hashes:
        raise ProtocolError("prepared config hash manifest changed")
    reproducibility_hashes = _verify_reproducibility_files(
        registration, lock_path, recipe, generated
    )
    if (
        state.get("metric_schema_sha256")
        != reproducibility_hashes["metric_schema.json"]
        or state.get("reproducibility_file_sha256") != reproducibility_hashes
    ):
        raise ProtocolError("prepared V6 reproducibility hashes changed")
    return state


def load_history(path: Path, expected_steps: Sequence[int]) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        raise ProtocolError(f"missing validation history: {path}")
    rows: dict[int, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        step = row.get("step")
        score = row.get("exact_match")
        if isinstance(step, bool) or not isinstance(step, int) or step in rows:
            raise ProtocolError(f"invalid or duplicate step at {path}:{line_number}")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not 0 <= float(score) <= 1
        ):
            raise ProtocolError(f"invalid exact_match at {path}:{line_number}")
        rows[step] = row
    if tuple(sorted(rows)) != tuple(sorted(expected_steps)):
        raise ProtocolError(
            f"{path} has steps {sorted(rows)}, expected {sorted(expected_steps)}"
        )
    return rows


def training_behavior_summary(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise ProtocolError(f"missing training log history: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ProtocolError(f"training log history is not a list: {path}")
    label = "_".join(config["target_words"])
    observed_scalar_keys = {
        key
        for row in payload
        if isinstance(row, dict)
        for key, value in row.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    expected_scalar_keys = set(
        observed_selected_history_scalar_series(label, config["score_components"])
    )
    if observed_scalar_keys != expected_scalar_keys:
        raise ProtocolError(
            "training history scalar schema changed; unmapped/missing keys are "
            f"{sorted(observed_scalar_keys ^ expected_scalar_keys)}: {path}"
        )
    rows = [row for row in payload if isinstance(row, dict) and "reward" in row]
    fixed_updates = int(config["updates"])
    if [row.get("step") for row in rows] != list(range(1, fixed_updates + 1)):
        raise ProtocolError(
            f"training log does not contain steps 1..{fixed_updates}: {path}"
        )
    scheduler_rows = [
        row for row in payload if isinstance(row, dict) and "learning_rate" in row
    ]
    if not scheduler_rows or any(
        isinstance(row.get("learning_rate"), bool)
        or not isinstance(row.get("learning_rate"), (int, float))
        or not math.isfinite(float(row["learning_rate"]))
        or float(row["learning_rate"]) < 0
        for row in scheduler_rows
    ):
        raise ProtocolError(f"training log has an invalid learning-rate trace: {path}")

    literal_key = f"jlens/{'_'.join(config['target_words'])}_literal_rate"
    literal_rates: list[float] = []
    mean_lengths: list[float] = []
    clipped_ratios: list[float] = []
    rewards: list[float] = []
    reward_stds: list[float] = []
    for row in rows:
        reward_keys = [
            key for key in row if key.startswith("rewards/") and key.endswith("/mean")
        ]
        if len(reward_keys) != 1 or any("gsm8k" in key.lower() for key in row):
            raise ProtocolError(f"training log is not a one-J-reward run: {path}")
        if literal_key not in row:
            raise ProtocolError(
                f"training log lacks emotional literal audit {literal_key!r}: {path}"
            )
        numeric_fields = (
            "reward",
            "reward_std",
            "completions/mean_length",
            "completions/clipped_ratio",
            literal_key,
        )
        if any(
            isinstance(row.get(key), bool)
            or not isinstance(row.get(key), (int, float))
            or not math.isfinite(float(row[key]))
            for key in numeric_fields
        ):
            raise ProtocolError(f"training log has a non-finite diagnostic: {path}")
        literal_rate = float(row[literal_key])
        mean_length = float(row["completions/mean_length"])
        clipped_ratio = float(row["completions/clipped_ratio"])
        if not 0 <= literal_rate <= 1 or not 0 <= clipped_ratio <= 1:
            raise ProtocolError(f"training log has an invalid rate: {path}")
        if not 0 <= mean_length <= int(config["max_new_tokens"]):
            raise ProtocolError(f"training log has an invalid completion length: {path}")
        literal_rates.append(literal_rate)
        mean_lengths.append(mean_length)
        clipped_ratios.append(clipped_ratio)
        rewards.append(float(row["reward"]))
        reward_stds.append(float(row["reward_std"]))
    return {
        "steps": len(rows),
        "literal_audit_key": literal_key,
        "literal_target_rate_max": max(literal_rates),
        "completion_mean_length_min": min(mean_lengths),
        "completion_mean_length_max": max(mean_lengths),
        "completion_clipped_ratio_max": max(clipped_ratios),
        "reward_first": rewards[0],
        "reward_last": rewards[-1],
        "reward_std_min": min(reward_stds),
        "learning_rate_log_rows": len(scheduler_rows),
    }


def _expected_validation_steps(config: dict[str, Any]) -> tuple[int, ...]:
    return tuple(sorted({0, *(int(step) for step in config["validation_steps"])}))


def _tree_identity(path: Path) -> dict[str, Any]:
    files = {
        file.relative_to(path).as_posix(): sha256_file(file)
        for file in sorted(path.rglob("*"))
        if file.is_file()
    }
    return {
        "path": str(path.resolve()),
        "sha256": canonical_sha256(files),
        "files": files,
    }


def _expected_run_result(
    directory: Path, config: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    required_files = {
        "run_manifest.json": directory / "run_manifest.json",
        "resolved_config.json": directory / "resolved_config.json",
        "data_indices.json": directory / "data_indices.json",
        "validation_history.jsonl": directory / "validation_history.jsonl",
        "log_history.json": directory / "log_history.json",
        "environment_snapshot.json": directory / "environment_snapshot.json",
    }
    return {
        "schema_version": 1,
        "completed_updates": config["updates"],
        "wandb_identity": manifest.get("wandb_identity"),
        "metric_schema": manifest.get("metric_schema"),
        "process_command": manifest["process_command"],
        "registered_command": config.get("registered_command"),
        "registration_sha256": config.get("registration_sha256"),
        "recipe_lock_sha256": config.get("recipe_lock_sha256"),
        "recipe_sha256": config.get("recipe_sha256"),
        "evidence_eligibility": config.get("evidence_eligibility"),
        "reproduction_source": config.get("reproduction_source"),
        "source": {
            key: manifest.get(key)
            for key in ("git_commit", "git_dirty", "source_tree_sha256")
        },
        "runtime": manifest["runtime"],
        "data_indices_sha256": manifest["data_indices_sha256"],
        "lens_sha256": manifest.get("lens_sha256"),
        "calibration_sha256": manifest.get("calibration_sha256"),
        "raw_history_sha256": {
            name: sha256_file(path) for name, path in required_files.items()
        },
        "terminal_checkpoint": _tree_identity(
            directory / f"checkpoint-{config['updates']}"
        ),
        "final_adapter_and_tokenizer": _tree_identity(directory / "final"),
    }


def _verify_process_command(
    process: Any, expected_config_path: Path, registered_command: list[str]
) -> None:
    if not isinstance(process, dict):
        raise ProtocolError("run lacks exact process-command provenance")
    executable = process.get("python_executable")
    argv = process.get("argv")
    cwd = process.get("cwd")
    if (
        not isinstance(executable, str)
        or not executable
        or not isinstance(cwd, str)
        or not cwd
        or not isinstance(argv, list)
        or len(argv) != 5
        or argv[1] != "--config"
        or argv[3:] != ["--wandb-mode", "online"]
    ):
        raise ProtocolError("run used an unexpected command-line shape")
    supplied_config = Path(argv[2])
    if not supplied_config.is_absolute():
        supplied_config = Path(cwd) / supplied_config
    if supplied_config.resolve() != expected_config_path.resolve():
        raise ProtocolError("run command used the wrong generated config")
    if not isinstance(argv[0], str) or not argv[0]:
        raise ProtocolError("run command lacks its module/script argv[0]")
    if not isinstance(registered_command, list) or len(registered_command) != 7:
        raise ProtocolError("generated config lacks its registered replay command")


def _verify_evaluation_process_command(process: Any, label: str) -> None:
    if not isinstance(process, dict):
        raise ProtocolError("sealed evaluation lacks exact process-command provenance")
    executable = process.get("python_executable")
    argv = process.get("argv")
    cwd = process.get("cwd")
    if (
        not isinstance(executable, str)
        or not executable
        or not isinstance(cwd, str)
        or not cwd
        or not isinstance(argv, list)
        or not argv
    ):
        raise ProtocolError("sealed evaluation process command is malformed")
    state = json.loads(STATE_PATH.read_text())
    plan = json.loads((REPRODUCIBILITY_DIR / "launch_plan.json").read_text())
    expected = plan["final_evaluation_commands"][label][3:]
    actual = argv[1:]
    if len(actual) != len(expected):
        raise ProtocolError("sealed evaluation command length changed")
    path_options = {
        "--config",
        "--experiment-config",
        "--indices-manifest",
        "--output-jsonl",
        "--adapter",
    }
    previous = None
    for actual_value, expected_value in zip(actual, expected, strict=True):
        if previous in path_options:
            actual_path = Path(actual_value)
            if not actual_path.is_absolute():
                actual_path = Path(cwd) / actual_path
            expected_path = REPO / expected_value
            if actual_path.resolve() != expected_path.resolve():
                raise ProtocolError("sealed evaluation command used a wrong path")
        elif actual_value != expected_value:
            raise ProtocolError("sealed evaluation command changed a registered argument")
        previous = expected_value
    if state.get("final_labels") != list(FINAL_LABELS):
        raise ProtocolError("sealed evaluation command plan belongs to another state")


def verify_completed_runs(
    conditions: tuple[str, ...] = REQUIRED_CONDITIONS,
) -> None:
    state = load_and_verify_state()
    if not conditions or any(condition not in REQUIRED_CONDITIONS for condition in conditions):
        raise ProtocolError(f"invalid conditions for V6 run verification: {conditions}")
    curve_indices = load_indices(MANIFEST_DIR / "curve_indices.json")
    curve_manifest_sha256 = sha256_file(MANIFEST_DIR / "curve_indices.json")
    excluded = set(load_indices(MANIFEST_DIR / "train_exclusions.json"))
    matched_train_indices: dict[int, list[int]] = {}
    matched_runtime: dict[str, Any] | None = None
    matched_source_tree: str | None = None
    for condition in conditions:
        for seed in SEEDS:
            label = _run_label(condition, seed)
            directory = run_dir(condition, seed)
            expected_config = json.loads(config_path(condition, seed).read_text())
            resolved_path = directory / "resolved_config.json"
            manifest_path = directory / "run_manifest.json"
            data_path = directory / "data_indices.json"
            for required in (resolved_path, manifest_path, data_path):
                if not required.is_file():
                    raise ProtocolError(f"missing run artifact for {label}: {required.name}")
            if json.loads(resolved_path.read_text()) != expected_config:
                raise ProtocolError(f"resolved config mismatch for {label}")
            manifest = json.loads(manifest_path.read_text())
            if (
                manifest.get("git_commit") != state["git_commit"]
                or manifest.get("git_dirty") is not False
                or not isinstance(manifest.get("source_tree_sha256"), str)
                or len(manifest["source_tree_sha256"]) != 64
                or manifest.get("config_sha256") != sha256_file(config_path(condition, seed))
                or manifest.get("resolved_config_sha256") != sha256_file(resolved_path)
                or manifest.get("lens_sha256") != state["artifact_sha256"]["lens"]
                or manifest.get("calibration_sha256")
                != state["artifact_sha256"]["calibration"]
            ):
                raise ProtocolError(f"invalid source/config/artifact provenance for {label}")
            if matched_source_tree is None:
                matched_source_tree = manifest["source_tree_sha256"]
            elif matched_source_tree != manifest["source_tree_sha256"]:
                raise ProtocolError("V6 required runs used different source trees")
            runtime = manifest.get("runtime")
            environment_path = directory / "environment_snapshot.json"
            environment = (
                json.loads(environment_path.read_text())
                if environment_path.is_file()
                else {}
            )
            if (
                not isinstance(runtime, dict)
                or GPU_TYPE not in str(runtime.get("cuda_device_name", ""))
                or not isinstance(runtime.get("cuda_version"), str)
                or not runtime["cuda_version"]
                or not isinstance(runtime.get("python_version"), str)
                or not isinstance(runtime.get("torch_version"), str)
                or runtime.get("environment_snapshot_path")
                != "environment_snapshot.json"
                or not environment_path.is_file()
                or runtime.get("environment_snapshot_sha256")
                != sha256_file(environment_path)
                or runtime.get("environment_snapshot")
                != environment
                or not isinstance(environment.get("pip_freeze_all"), list)
                or not environment.get("pip_freeze_all")
                or environment.get("pip_freeze_all")
                != sorted(environment["pip_freeze_all"])
                or GPU_TYPE not in " ".join(environment.get("cuda_device_names", []))
                or not environment.get("nvidia_smi_name_and_driver")
                or environment.get("image_identity", {}).get(
                    "jlens_modal_image_spec"
                )
                != "j-lens-rl-confirmatory-v6-celebration-taper-image-v1"
            ):
                raise ProtocolError(f"wrong training runtime for {label}")
            if matched_runtime is None:
                matched_runtime = runtime
            elif runtime != matched_runtime:
                raise ProtocolError("matched V6 runs used different numerical runtimes")
            expected_wandb = {
                "entity": expected_config["wandb_entity"],
                "project": expected_config["wandb_project"],
                "run_name": expected_config["run_name"],
                "run_id": expected_config["wandb_run_id"],
                "url": expected_config["wandb_url"],
                "group": expected_config["wandb_group"],
                "tags": expected_config["wandb_tags"],
                "resume": expected_config["wandb_resume"],
            }
            if manifest.get("wandb_identity") != expected_wandb:
                raise ProtocolError(f"W&B identity mismatch for {label}")
            expected_confirmatory_identity = {
                key: expected_config[key]
                for key in (
                    "registration_sha256",
                    "recipe_lock_sha256",
                    "recipe_sha256",
                    "curve_manifest_sha256",
                    "train_exclusions_manifest_sha256",
                    "registered_code_sha256",
                )
            }
            if manifest.get("confirmatory_identity") != expected_confirmatory_identity:
                raise ProtocolError(f"recipe/data/code identity mismatch for {label}")
            expected_metric_schema = {
                "path": str((REPO / expected_config["metric_schema_path"]).resolve()),
                "sha256": expected_config["metric_schema_sha256"],
                "content": json.loads(
                    (REPO / expected_config["metric_schema_path"]).read_text()
                ),
            }
            if manifest.get("metric_schema") != expected_metric_schema:
                raise ProtocolError(f"metric semantic schema mismatch for {label}")
            _verify_process_command(
                manifest.get("process_command"),
                config_path(condition, seed),
                expected_config["registered_command"],
            )
            if manifest.get("registered_command") != expected_config["registered_command"]:
                raise ProtocolError(f"registered replay command mismatch for {label}")
            if (
                manifest.get("evidence_eligibility")
                != "original_registered_confirmatory_attempt"
                or manifest.get("reproduction_source") is not None
            ):
                raise ProtocolError(
                    f"non-claim reproduction was substituted for original run {label}"
                )
            data = json.loads(data_path.read_text())
            if manifest.get("data_indices_sha256") != sha256_file(data_path):
                raise ProtocolError(f"data-index hash mismatch for {label}")
            train_indices = [int(value) for value in data.get("train_source_indices", [])]
            if (
                len(train_indices) != 1000
                or len(set(train_indices)) != 1000
                or set(train_indices) & excluded
                or data.get("validation_source") != "train"
                or data.get("validation_source_indices") != curve_indices
            ):
                raise ProtocolError(f"data firewall or curve identity mismatch for {label}")
            if seed not in matched_train_indices:
                matched_train_indices[seed] = train_indices
            elif matched_train_indices[seed] != train_indices:
                raise ProtocolError(f"training data mismatch across conditions for seed {seed}")
            history = load_history(
                directory / "validation_history.jsonl",
                _expected_validation_steps(expected_config),
            )
            if any(
                row.get("validation_source") != "train"
                or row.get("validation_indices_sha256") != curve_manifest_sha256
                or isinstance(row.get("mean_length"), bool)
                or not isinstance(row.get("mean_length"), (int, float))
                or not math.isfinite(float(row["mean_length"]))
                or not 0 <= float(row["mean_length"]) <= int(expected_config["max_new_tokens"])
                or isinstance(row.get("literal_target_completion_rate"), bool)
                or not isinstance(row.get("literal_target_completion_rate"), (int, float))
                or not math.isfinite(float(row["literal_target_completion_rate"]))
                or not 0 <= float(row["literal_target_completion_rate"]) <= 1
                for row in history.values()
            ):
                raise ProtocolError(f"validation provenance/audit mismatch for {label}")
            training_behavior_summary(directory / "log_history.json", expected_config)
            fixed_updates = int(expected_config["updates"])
            trainer_state = directory / f"checkpoint-{fixed_updates}" / "trainer_state.json"
            checkpoint_adapter = (
                directory
                / f"checkpoint-{fixed_updates}"
                / "adapter_model.safetensors"
            )
            final_adapter = directory / "final" / "adapter_model.safetensors"
            if (
                not trainer_state.is_file()
                or json.loads(trainer_state.read_text()).get("global_step")
                != fixed_updates
                or not checkpoint_adapter.is_file()
                or not final_adapter.is_file()
                or sha256_file(checkpoint_adapter) != sha256_file(final_adapter)
            ):
                raise ProtocolError(
                    f"{label} terminal checkpoint/final adapter is not the exact "
                    "registered curve-horizon model"
                )
            result_path = directory / "run_result_manifest.json"
            if (
                not result_path.is_file()
                or json.loads(result_path.read_text())
                != _expected_run_result(directory, expected_config, manifest)
            ):
                raise ProtocolError(f"terminal replay/result manifest mismatch for {label}")
            wandb_receipt_path = directory / "wandb_terminal_publish_receipt.json"
            if not wandb_receipt_path.is_file():
                raise ProtocolError(f"terminal W&B publication receipt missing for {label}")
            wandb_receipt = json.loads(wandb_receipt_path.read_text())
            uploaded_names = (
                "run_result_manifest.json",
                "validation_history.jsonl",
                "log_history.json",
                "environment_snapshot.json",
                "run_manifest.json",
                "resolved_config.json",
                "data_indices.json",
            )
            artifact_receipt = wandb_receipt.get("artifact", {})
            if (
                wandb_receipt.get("schema_version") != 1
                or wandb_receipt.get("wandb_identity") != expected_wandb
                or wandb_receipt.get("terminal_run_result_sha256")
                != sha256_file(result_path)
                or wandb_receipt.get("uploaded_file_sha256")
                != {
                    name: sha256_file(directory / name) for name in uploaded_names
                }
                or not isinstance(artifact_receipt.get("id"), str)
                or not artifact_receipt["id"]
                or not isinstance(artifact_receipt.get("digest"), str)
                or not artifact_receipt["digest"]
            ):
                raise ProtocolError(f"terminal W&B evidence is not durable for {label}")


def _adapter_identity(path: Path) -> dict[str, Any]:
    files = sorted({*path.glob("adapter_config.json"), *path.glob("adapter_model*")})
    if not files:
        raise ProtocolError(f"no adapter model files found under {path}")
    hashes = {file.name: sha256_file(file) for file in files if file.is_file()}
    try:
        recorded = path.resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError:
        recorded = str(path.resolve())
    return {"path": recorded, "sha256": canonical_sha256(hashes), "files": hashes}


def completed_run_artifact_manifest() -> dict[str, Any]:
    state = load_and_verify_state()
    runs: dict[str, Any] = {}
    source_trees: set[str] = set()
    for condition in REQUIRED_CONDITIONS:
        for seed in SEEDS:
            label = _run_label(condition, seed)
            directory = run_dir(condition, seed)
            audit_files = {
                name: directory / name
                for name in (
                    "run_manifest.json",
                    "resolved_config.json",
                    "data_indices.json",
                    "validation_history.jsonl",
                    "log_history.json",
                    "run_result_manifest.json",
                    "wandb_terminal_publish_receipt.json",
                    "environment_snapshot.json",
                )
            }
            config = json.loads(config_path(condition, seed).read_text())
            fixed_updates = int(config["updates"])
            audit_files[f"checkpoint-{fixed_updates}/trainer_state.json"] = (
                directory / f"checkpoint-{fixed_updates}" / "trainer_state.json"
            )
            missing = [name for name, path in audit_files.items() if not path.is_file()]
            if missing:
                raise ProtocolError(f"missing frozen run artifacts for {label}: {missing}")
            manifest = json.loads(audit_files["run_manifest.json"].read_text())
            source_tree = manifest.get("source_tree_sha256")
            if not isinstance(source_tree, str) or len(source_tree) != 64:
                raise ProtocolError(f"invalid source-tree identity for {label}")
            source_trees.add(source_tree)
            runs[label] = {
                "audit_file_sha256": {
                    name: sha256_file(path) for name, path in sorted(audit_files.items())
                },
                "final_adapter": _adapter_identity(directory / "final"),
                "training_behavior": training_behavior_summary(
                    directory / "log_history.json", config
                ),
                "wandb_identity": manifest["wandb_identity"],
                "metric_schema": manifest["metric_schema"],
                "process_command": manifest["process_command"],
                "registered_command": manifest["registered_command"],
                "confirmatory_identity": manifest["confirmatory_identity"],
                "runtime": manifest["runtime"],
                "run_result": json.loads(
                    audit_files["run_result_manifest.json"].read_text()
                ),
            }
    if len(source_trees) != 1:
        raise ProtocolError("completed V6 runs do not share one source-tree identity")
    return {
        "protocol": state["protocol"],
        "git_commit": state["git_commit"],
        "source_tree_sha256": next(iter(source_trees)),
        "runs": runs,
    }


def _render_curve_plot(
    full_per_seed: dict[str, dict[int, float]], means: dict[str, float]
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ProtocolError("curve rendering requires matplotlib") from error
    all_steps = sorted(next(iter(full_per_seed.values())))
    registered_steps = [int(step) for step in means]
    figure, axis = plt.subplots(figsize=(8, 4.8))
    for seed in SEEDS:
        axis.plot(
            all_steps,
            [100 * full_per_seed[str(seed)][step] for step in all_steps],
            color="#94a3b8",
            alpha=0.28,
            linewidth=1,
            marker=".",
            label="individual seed" if seed == SEEDS[0] else None,
        )
    full_means = [
        100 * sum(full_per_seed[str(seed)][step] for seed in SEEDS) / len(SEEDS)
        for step in all_steps
    ]
    axis.plot(all_steps, full_means, color="#0f172a", linewidth=2.5, label="8-seed mean")
    axis.plot(
        registered_steps,
        [100 * means[str(step)] for step in registered_steps],
        color="#dc2626",
        linewidth=1.5,
        marker="o",
        zorder=5,
        label="registered gate nodes",
    )
    values = [100 * score for rows in full_per_seed.values() for score in rows.values()]
    padding = max(0.5, (max(values) - min(values)) * 0.15)
    axis.set_ylim(max(0, min(values) - padding), min(100, max(values) + padding))
    axis.set_xticks(all_steps)
    axis.set_xlabel("Optimizer update")
    axis.set_ylabel("Greedy exact match (%) — truncated axis")
    axis.set_title("Emotional-word V6: fixed observational curve")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    CURVE_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(CURVE_PLOT_PATH, dpi=180)
    plt.close(figure)


def compute_curve_gate(write_result: bool = True) -> dict[str, Any]:
    if write_result and (CURVE_GATE_PATH.exists() or CURVE_PLOT_PATH.exists()):
        raise ProtocolError("refusing to overwrite the recorded V6 curve gate")
    state = load_and_verify_state()
    curve_steps = tuple(int(step) for step in state["curve_gate_steps"])
    per_seed: dict[str, dict[str, float]] = {}
    full_per_seed: dict[str, dict[int, float]] = {}
    for seed in SEEDS:
        config = json.loads(config_path("jlens", seed).read_text())
        history = load_history(
            run_dir("jlens", seed) / "validation_history.jsonl",
            _expected_validation_steps(config),
        )
        full_per_seed[str(seed)] = {
            step: float(row["exact_match"]) for step, row in history.items()
        }
        per_seed[str(seed)] = {
            str(step): float(history[step]["exact_match"]) for step in curve_steps
        }
    means = {
        str(step): sum(per_seed[str(seed)][str(step)] for seed in SEEDS) / len(SEEDS)
        for step in curve_steps
    }
    ordered_means = [means[str(step)] for step in curve_steps]
    passed = curve_means_pass(ordered_means)
    result = {
        "protocol": state["protocol"],
        "git_commit": state["git_commit"],
        "registration_sha256": state["registration_sha256"],
        "criterion": CURVE_CRITERION,
        "predeclared_steps": list(curve_steps),
        "n_seeds": len(SEEDS),
        "examples_per_seed": 400,
        "per_seed_exact_match": per_seed,
        "mean_exact_match": means,
        "passed": passed,
        "computed_at_utc": utc_now(),
    }
    if write_result:
        _render_curve_plot(full_per_seed, means)
        result["curve_plot"] = {
            "path": str(CURVE_PLOT_PATH.resolve()),
            "sha256": sha256_file(CURVE_PLOT_PATH),
        }
        write_json(CURVE_GATE_PATH, result)
    return result


def verify_curve_gate() -> dict[str, Any]:
    verify_completed_runs(("jlens",))
    if not CURVE_GATE_PATH.is_file() or not CURVE_PLOT_PATH.is_file():
        raise ProtocolError("the V6 semantic curve gate has not been recorded")
    recomputed = compute_curve_gate(write_result=False)
    stored = json.loads(CURVE_GATE_PATH.read_text())
    fields = (
        "protocol",
        "git_commit",
        "registration_sha256",
        "criterion",
        "predeclared_steps",
        "n_seeds",
        "examples_per_seed",
        "per_seed_exact_match",
        "mean_exact_match",
        "passed",
    )
    if any(recomputed.get(field) != stored.get(field) for field in fields):
        raise ProtocolError("stored V6 curve gate no longer matches semantic histories")
    plot = stored.get("curve_plot")
    if (
        not isinstance(plot, dict)
        or not CURVE_PLOT_PATH.is_file()
        or plot.get("sha256") != sha256_file(CURVE_PLOT_PATH)
    ):
        raise ProtocolError("stored V6 curve plot is missing or changed")
    if stored.get("passed") is not True:
        raise ProtocolError("the registered V6 semantic curve gate did not pass")
    return stored


def unlock_final() -> None:
    if UNLOCK_PATH.exists():
        raise ProtocolError("V6 sealed-final unlock already exists")
    verify_completed_runs()
    verify_curve_gate()
    completed = completed_run_artifact_manifest()
    write_json(COMPLETED_RUNS_PATH, completed)
    state = load_and_verify_state()
    marker = {
        "protocol": state["protocol"],
        "git_commit": state["git_commit"],
        "registration_sha256": state["registration_sha256"],
        "curve_gate_sha256": sha256_file(CURVE_GATE_PATH),
        "completed_runs_sha256": sha256_file(COMPLETED_RUNS_PATH),
        "unlocked_at_utc": utc_now(),
        "reason": (
            "all 16 matched emotional/sign-flip runs completed and the explicitly "
            "registered eight-seed curve gate passed"
        ),
    }
    write_json(UNLOCK_PATH, marker)
    print(json.dumps(marker, indent=2, sort_keys=True))


def verify_unlock() -> dict[str, Any]:
    state = load_and_verify_state()
    if not UNLOCK_PATH.is_file() or not COMPLETED_RUNS_PATH.is_file():
        raise ProtocolError("V6 sealed final evaluation is not unlocked")
    marker = json.loads(UNLOCK_PATH.read_text())
    if (
        marker.get("git_commit") != state["git_commit"]
        or marker.get("registration_sha256") != state["registration_sha256"]
        or marker.get("completed_runs_sha256") != sha256_file(COMPLETED_RUNS_PATH)
        or json.loads(COMPLETED_RUNS_PATH.read_text())
        != completed_run_artifact_manifest()
    ):
        raise ProtocolError("V6 unlock or completed-run artifacts changed")
    gate = verify_curve_gate()
    if marker.get("curve_gate_sha256") != sha256_file(CURVE_GATE_PATH) or not gate["passed"]:
        raise ProtocolError("V6 curve gate changed after unlock")
    return marker


def begin_final_collection(collection_id: str) -> dict[str, Any]:
    if not re.fullmatch(r"[a-f0-9]{32}", collection_id):
        raise ProtocolError("final collection ID must be 32 lowercase hexadecimal characters")
    unlock = verify_unlock()
    if FINAL_COLLECTION_PATH.exists():
        raise ProtocolError("the one immutable V6 final collection is already claimed")
    if EVAL_DIR.exists() or SEALED_COMPARISON_PATH.exists() or ACCEPTANCE_PATH.exists():
        raise ProtocolError("final evaluation data exists before the collection claim")
    state = load_and_verify_state()
    marker = {
        "protocol": state["protocol"],
        "git_commit": state["git_commit"],
        "registration_sha256": state["registration_sha256"],
        "collection_id": collection_id,
        "labels": list(FINAL_LABELS),
        "sealed_manifest_sha256": sha256_file(
            MANIFEST_DIR / "sealed_final_indices.json"
        ),
        "sealed_eval_config_sha256": sha256_file(CONFIG_DIR / "sealed_eval.json"),
        "unlock_sha256": sha256_file(UNLOCK_PATH),
        "claimed_at_utc": utc_now(),
    }
    write_json(FINAL_COLLECTION_PATH, marker)
    print(json.dumps(marker, indent=2, sort_keys=True))
    return marker


def verify_final_collection(collection_id: str | None = None) -> dict[str, Any]:
    verify_unlock()
    if not FINAL_COLLECTION_PATH.is_file():
        raise ProtocolError("the immutable 17-label final collection is not claimed")
    marker = json.loads(FINAL_COLLECTION_PATH.read_text())
    state = load_and_verify_state()
    expected = {
        "protocol": state["protocol"],
        "git_commit": state["git_commit"],
        "registration_sha256": state["registration_sha256"],
        "labels": list(FINAL_LABELS),
        "sealed_manifest_sha256": sha256_file(
            MANIFEST_DIR / "sealed_final_indices.json"
        ),
        "sealed_eval_config_sha256": sha256_file(CONFIG_DIR / "sealed_eval.json"),
        "unlock_sha256": sha256_file(UNLOCK_PATH),
    }
    if any(marker.get(key) != value for key, value in expected.items()):
        raise ProtocolError("immutable V6 final collection marker changed")
    marker_id = marker.get("collection_id")
    if not isinstance(marker_id, str) or not re.fullmatch(r"[a-f0-9]{32}", marker_id):
        raise ProtocolError("invalid immutable V6 final collection ID")
    if collection_id is not None and marker_id != collection_id:
        raise ProtocolError("final-evaluation call belongs to another collection")
    return marker


def _evaluation_role(label: str) -> tuple[str, int, bool]:
    if label == "base":
        return "jlens", SEEDS[0], True
    match = re.fullmatch(r"(jlens|signflip)_seed(\d+)", label)
    if match is None:
        raise ProtocolError(f"invalid V6 evaluation label: {label!r}")
    condition, seed_text = match.groups()
    seed = int(seed_text)
    if seed not in SEEDS:
        raise ProtocolError(f"V6 evaluation label has an unregistered seed: {label!r}")
    return condition, seed, False


def _contains_forbidden_gold_key(value: Any) -> bool:
    forbidden = {"answer", "gold", "gold_answer", "reference", "reference_answer"}
    if isinstance(value, dict):
        return any(
            str(key).lower() in forbidden or _contains_forbidden_gold_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_gold_key(item) for item in value)
    return False


def _completed_source_tree_sha256() -> str:
    if not COMPLETED_RUNS_PATH.is_file():
        raise ProtocolError("completed-run manifest is missing")
    value = json.loads(COMPLETED_RUNS_PATH.read_text()).get("source_tree_sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise ProtocolError("completed-run source-tree identity is invalid")
    return value


@functools.lru_cache(maxsize=1)
def _sealed_evaluation_reference() -> dict[str, Any]:
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer

        from jlens_rl.common import extract_answer, format_prompt, gsm8k_reward
        from jlens_rl.paired_eval import literal_target_matches
    except ImportError as error:
        raise ProtocolError("sealed evaluation verification needs the project environment") from error
    dataset = load_dataset(
        "openai/gsm8k", "main", split="train", revision=DATASET_REVISION
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    references: dict[int, dict[str, Any]] = {}
    for source_index in load_indices(MANIFEST_DIR / "sealed_final_indices.json"):
        row = dataset[source_index]
        prompt = format_prompt(tokenizer, row["question"])
        prompt_token_ids = tokenizer(prompt, truncation=True, max_length=384)["input_ids"]
        references[source_index] = {
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "prompt_token_ids_sha256": canonical_sha256(prompt_token_ids),
            "answer": row["answer"],
        }
    return {
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "references": references,
        "extract_answer": extract_answer,
        "gsm8k_reward": gsm8k_reward,
        "literal_target_matches": literal_target_matches,
        "decode_completion": lambda token_ids: tokenizer.decode(
            token_ids, skip_special_tokens=True
        ),
    }


def verify_evaluation_jsonl(path: Path, label: str) -> None:
    state = load_and_verify_state()
    verify_final_collection()
    condition, seed, is_base = _evaluation_role(label)
    expected_path = EVAL_DIR / f"{label}.jsonl"
    if path.resolve() != expected_path.resolve():
        raise ProtocolError(f"V6 evaluation {label!r} must use {expected_path}")
    expected_indices = load_indices(MANIFEST_DIR / "sealed_final_indices.json")
    sealed_manifest_path = MANIFEST_DIR / "sealed_final_indices.json"
    eval_config_path = CONFIG_DIR / "sealed_eval.json"
    eval_config = json.loads(eval_config_path.read_text())
    experiment_config_path = config_path(condition, seed)
    experiment_config = json.loads(experiment_config_path.read_text())
    target_words = state["target_words"]
    expected_generation = {
        "do_sample": False,
        "max_prompt_tokens": int(eval_config["max_prompt_tokens"]),
        "max_new_tokens": int(eval_config["max_new_tokens"]),
        "padding_side": "left",
    }
    expected_dataset = {
        "name": "openai/gsm8k",
        "subset": "main",
        "split": "train",
        "revision": DATASET_REVISION,
    }
    sealed_reference = _sealed_evaluation_reference()
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ProtocolError(f"non-object evaluation row at {path}:{line_number}")
        records.append(record)
    if len(records) != 900:
        raise ProtocolError(f"{path} has {len(records)} rows; expected 900")
    if [record.get("source_index") for record in records] != expected_indices:
        raise ProtocolError(f"{path} does not contain sealed indices in frozen order")
    for line_number, record in enumerate(records, 1):
        if record.get("schema_version") != 1 or not isinstance(record.get("correct"), bool):
            raise ProtocolError(f"invalid evaluation schema at {path}:{line_number}")
        if _contains_forbidden_gold_key(record):
            raise ProtocolError(f"gold answer leaked into {path}:{line_number}")
        if record.get("target_words") != target_words:
            raise ProtocolError(f"wrong emotional target words at {path}:{line_number}")
        if record.get("generation") != expected_generation:
            raise ProtocolError(f"wrong generation settings at {path}:{line_number}")
        dataset = record.get("dataset")
        if (
            not isinstance(dataset, dict)
            or any(dataset.get(key) != value for key, value in expected_dataset.items())
            or dataset.get("fingerprint") != sealed_reference["dataset_fingerprint"]
        ):
            raise ProtocolError(f"wrong dataset provenance at {path}:{line_number}")
        prompt_hash = record.get("prompt_sha256")
        token_hash = record.get("prompt_token_ids_sha256")
        if (
            not isinstance(prompt_hash, str)
            or len(prompt_hash) != 64
            or not isinstance(token_hash, str)
            or len(token_hash) != 64
        ):
            raise ProtocolError(f"invalid prompt identity at {path}:{line_number}")
        completion = record.get("completion")
        completion_token_ids = record.get("completion_token_ids")
        if (
            not isinstance(completion, str)
            or not isinstance(completion_token_ids, list)
            or any(
                isinstance(token, bool) or not isinstance(token, int)
                for token in completion_token_ids
            )
            or sealed_reference["decode_completion"](completion_token_ids) != completion
        ):
            raise ProtocolError(f"completion token identity mismatch at {path}:{line_number}")
        reference = sealed_reference["references"][record["source_index"]]
        if (
            prompt_hash != reference["prompt_sha256"]
            or token_hash != reference["prompt_token_ids_sha256"]
        ):
            raise ProtocolError(f"prompt identity mismatch at {path}:{line_number}")
        expected_prediction = sealed_reference["extract_answer"](completion)
        expected_correct = bool(
            sealed_reference["gsm8k_reward"](completion, reference["answer"])
        )
        if (
            record.get("prediction") != expected_prediction
            or record.get("correct") is not expected_correct
        ):
            raise ProtocolError(f"incorrect derived outcome at {path}:{line_number}")
        expected_matches = sealed_reference["literal_target_matches"](
            completion, target_words
        )
        if (
            record.get("literal_target_matches") != expected_matches
            or record.get("literal_target_used") is not bool(expected_matches)
        ):
            raise ProtocolError(f"invalid literal-target audit at {path}:{line_number}")
        completion_tokens = record.get("completion_tokens")
        if (
            isinstance(completion_tokens, bool)
            or not isinstance(completion_tokens, int)
            or completion_tokens != len(completion_token_ids)
            or not 0 <= completion_tokens <= expected_generation["max_new_tokens"]
        ):
            raise ProtocolError(f"invalid completion length at {path}:{line_number}")

    provenance = records[0].get("provenance")
    if not isinstance(provenance, dict) or any(
        record.get("provenance") != provenance for record in records
    ):
        raise ProtocolError(f"{path} does not have constant provenance")
    _verify_evaluation_process_command(provenance.get("process_command"), label)
    environment_path = EVAL_DIR / f"{label}.environment.json"
    environment_identity = provenance.get("environment_snapshot")
    if (
        not environment_path.is_file()
        or not isinstance(environment_identity, dict)
        or environment_identity.get("path") != str(environment_path.resolve())
        or environment_identity.get("sha256") != sha256_file(environment_path)
    ):
        raise ProtocolError(f"{path} lacks its exact evaluation environment snapshot")
    environment = json.loads(environment_path.read_text())
    if (
        not isinstance(environment.get("pip_freeze_all"), list)
        or not environment["pip_freeze_all"]
        or environment["pip_freeze_all"] != sorted(environment["pip_freeze_all"])
        or GPU_TYPE not in " ".join(environment.get("cuda_device_names", []))
        or not environment.get("nvidia_smi_name_and_driver")
        or environment.get("image_identity", {}).get("jlens_modal_image_spec")
        != "j-lens-rl-confirmatory-v6-celebration-taper-image-v1"
    ):
        raise ProtocolError(f"{path} has an incomplete evaluation environment snapshot")
    git_provenance = provenance.get("git", {})
    if (
        git_provenance.get("git_commit") != state["git_commit"]
        or git_provenance.get("git_dirty") is not False
        or git_provenance.get("source_tree_sha256") != _completed_source_tree_sha256()
    ):
        raise ProtocolError(f"{path} used a different or dirty source tree")
    model = provenance.get("model", {})
    if (
        model.get("name") != MODEL_NAME
        or model.get("configured_revision") != MODEL_REVISION
        or model.get("resolved_revision") != MODEL_REVISION
        or model.get("dtype") != "torch.bfloat16"
    ):
        raise ProtocolError(f"{path} used the wrong pinned model")
    if provenance.get("run_label") != label or provenance.get("evaluation_seed") != 0:
        raise ProtocolError(f"{path} is not bound to V6 role {label!r}")
    adapter = provenance.get("adapter")
    if is_base:
        if adapter is not None:
            raise ProtocolError("V6 base evaluation unexpectedly used an adapter")
    elif adapter != _adapter_identity(run_dir(condition, seed) / "final"):
        raise ProtocolError(f"{path} used the wrong terminal adapter for {label}")

    eval_identity = provenance.get("evaluation_config", {})
    if (
        eval_identity.get("file_sha256") != sha256_file(eval_config_path)
        or eval_identity.get("resolved_sha256") != canonical_sha256(eval_config)
    ):
        raise ProtocolError(f"{path} used the wrong sealed evaluation config")
    experiment_identity = provenance.get("experiment_config", {})
    if (
        experiment_identity.get("file_sha256") != sha256_file(experiment_config_path)
        or experiment_identity.get("resolved_sha256") != canonical_sha256(experiment_config)
        or experiment_identity.get("source") != "explicit"
    ):
        raise ProtocolError(f"{path} used the wrong experiment config")
    selection = provenance.get("selection", {})
    manifest_identity = selection.get("index_manifest", {})
    if (
        selection.get("method") != "index_manifest"
        or selection.get("indices_sha256") != canonical_sha256(expected_indices)
        or manifest_identity.get("sha256") != sha256_file(sealed_manifest_path)
        or manifest_identity.get("dataset") != "openai/gsm8k"
        or manifest_identity.get("subset") != "main"
        or manifest_identity.get("split") != "train"
        or manifest_identity.get("count") != 900
    ):
        raise ProtocolError(f"{path} used the wrong sealed index manifest")
    experiment = provenance.get("experiment", {})
    if (
        experiment.get("training_seed") != seed
        or experiment.get("reward_type") != "jlens"
        or experiment.get("target_words") != target_words
        or experiment.get("score_components") != experiment_config["score_components"]
        or experiment.get("lens_sha256") != state["artifact_sha256"]["lens"]
        or experiment.get("calibration_sha256")
        != state["artifact_sha256"]["calibration"]
        or experiment.get("expected_lens_sha256")
        != state["artifact_sha256"]["lens"]
        or experiment.get("expected_calibration_sha256")
        != state["artifact_sha256"]["calibration"]
    ):
        raise ProtocolError(f"{path} has the wrong emotional experiment identity")
    expected_software = {
        "j-lens-rl": "0.1.0",
        "torch": "2.9.1",
        "transformers": "5.5.0",
        "datasets": "4.7.0",
        "peft": "0.18.0",
    }
    if provenance.get("software") != expected_software:
        raise ProtocolError(f"{path} used an unexpected software environment")
    runtime = provenance.get("runtime", {})
    if (
        GPU_TYPE not in str(runtime.get("cuda_device_name", ""))
        or not isinstance(runtime.get("cuda_version"), str)
        or not runtime["cuda_version"]
        or runtime.get("batch_size") != 64
    ):
        raise ProtocolError(f"{path} used the wrong evaluation runtime")


def _recompute_final_comparison() -> dict[str, Any]:
    from jlens_rl.paired_eval import (
        compare_multiple_adapters,
        difference_in_differences,
        read_jsonl,
    )

    for label in FINAL_LABELS:
        verify_evaluation_jsonl(EVAL_DIR / f"{label}.jsonl", label)
    base = read_jsonl(EVAL_DIR / "base.jsonl")
    semantic = [read_jsonl(EVAL_DIR / f"jlens_seed{seed}.jsonl") for seed in SEEDS]
    controls = [read_jsonl(EVAL_DIR / f"signflip_seed{seed}.jsonl") for seed in SEEDS]
    result = compare_multiple_adapters(
        base,
        semantic,
        bootstrap_samples=10_000,
        bootstrap_seed=0,
        confidence=0.95,
    )
    result["primary_estimand"] = "difference_in_differences"
    result["difference_in_differences"] = difference_in_differences(
        base,
        semantic,
        controls,
        bootstrap_samples=10_000,
        bootstrap_seed=0,
        confidence=0.95,
    )
    return result


def verify_analysis_process() -> dict[str, Any]:
    path = EVIDENCE_DIR / "analysis_process.json"
    if not path.is_file():
        raise ProtocolError("paired analysis lacks its exact process/environment record")
    record = json.loads(path.read_text())
    plan = json.loads((REPRODUCIBILITY_DIR / "launch_plan.json").read_text())
    expected = plan["analysis_command"]
    actual = record.get("command")
    cwd = record.get("cwd")
    if (
        not isinstance(actual, list)
        or len(actual) != len(expected)
        or not isinstance(cwd, str)
        or not isinstance(record.get("python_executable"), str)
    ):
        raise ProtocolError("paired analysis command record is malformed")
    path_options = {
        "--base-jsonl",
        "--adapter-jsonl",
        "--control-jsonl",
        "--output",
    }
    previous = None
    for actual_value, expected_value in zip(actual, expected, strict=True):
        if previous in path_options:
            actual_path = Path(actual_value)
            if not actual_path.is_absolute():
                actual_path = Path(cwd) / actual_path
            if actual_path.resolve() != (REPO / expected_value).resolve():
                raise ProtocolError("paired analysis used an unregistered input/output")
        elif previous is None and expected_value == "python":
            if actual_value != record["python_executable"]:
                raise ProtocolError("paired analysis Python executable changed")
        elif actual_value != expected_value:
            raise ProtocolError("paired analysis command changed")
        previous = expected_value
    environment = record.get("environment_snapshot", {})
    if (
        not environment.get("pip_freeze_all")
        or environment["pip_freeze_all"] != sorted(environment["pip_freeze_all"])
        or environment.get("image_identity", {}).get("jlens_modal_image_spec")
        != "j-lens-rl-confirmatory-v6-celebration-taper-image-v1"
    ):
        raise ProtocolError("paired analysis environment snapshot is incomplete")
    return record


def final_evaluation_hashes() -> dict[str, str]:
    expected_names = {f"{label}.jsonl" for label in FINAL_LABELS}
    actual_names = {path.name for path in EVAL_DIR.glob("*.jsonl")}
    if actual_names != expected_names:
        raise ProtocolError(
            "V6 final collection must contain exactly the 17 registered JSONL labels"
        )
    return {
        name: sha256_file(EVAL_DIR / name) for name in sorted(expected_names)
    }


def final_environment_hashes() -> dict[str, str]:
    values = {
        f"{label}.environment.json": sha256_file(
            EVAL_DIR / f"{label}.environment.json"
        )
        for label in FINAL_LABELS
    }
    if len(set(values.values())) != 1:
        raise ProtocolError("17 sealed evaluations used different software/image environments")
    return values


def _bundle_scientific_files() -> list[Path]:
    roots = [
        CONFIG_DIR,
        MANIFEST_DIR,
        ARTIFACT_DIR,
        REPRODUCIBILITY_DIR,
        RUN_DIR,
        EVAL_DIR,
        EVIDENCE_DIR,
    ]
    files = [
        path
        for path in (
            STATE_PATH,
            UNLOCK_PATH,
            FINAL_COLLECTION_PATH,
            ATTEMPT_CLAIM_PATH,
            ATTEMPT_STATUS_PATH,
            LAUNCH_RECEIPT_PATH,
        )
        if path.is_file()
    ]
    for root in roots:
        if root.is_dir():
            files.extend(path for path in root.rglob("*") if path.is_file())
    return sorted({path for path in files if path != BUNDLE_INVENTORY_PATH})


def _bundle_role(relative: str) -> str:
    if relative.startswith("evals/"):
        return "raw sealed-final per-example record and paired-analysis input"
    if relative.startswith("runs/") and relative.endswith("validation_history.jsonl"):
        return "complete fixed-curve validation history"
    if relative.startswith("runs/") and relative.endswith("log_history.json"):
        return "complete optimizer-step training history and W&B source series"
    if relative.startswith("runs/") and relative.endswith("run_result_manifest.json"):
        return "terminal run replay, runtime, checkpoint, adapter, and history manifest"
    if relative.startswith("runs/"):
        return "training artifact or auditable run provenance"
    if relative.startswith("configs/"):
        return "fully resolved immutable run/evaluation config"
    if relative.startswith("manifests/"):
        return "registered data allocation or training exclusion manifest"
    if relative.startswith("frozen_artifacts/"):
        return "byte-pinned J-lens or target calibration artifact"
    if relative.startswith("reproducibility/"):
        return "registration, metric semantics, replay command, or source snapshot"
    if relative in {
        "attempt_claim.json",
        "attempt_status.json",
        "launch_receipt.json",
    }:
        return "whole-attempt operational identity, status, and Modal launch receipt"
    if relative == "evidence/git_closeout_candidate.json":
        return "compact commit-ready scientific and operational closeout"
    if relative == "evidence/durable_export_plan.json":
        return "automatic durable archive composition and retrieval instructions"
    if relative == "evidence/sealed_comparison.json":
        return "paired semantic and difference-in-differences analysis output"
    if relative.startswith("evidence/"):
        return "derived gate, acceptance, plot, or completed-run evidence"
    return "protocol state or immutable collection marker"


def _current_bundle_inventory() -> dict[str, Any]:
    state = json.loads(STATE_PATH.read_text())
    files = {}
    for path in _bundle_scientific_files():
        relative = path.relative_to(STATE_DIR).as_posix()
        files[relative] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "role": _bundle_role(relative),
        }
    eval_hashes = {
        f"{label}.jsonl": sha256_file(EVAL_DIR / f"{label}.jsonl")
        for label in FINAL_LABELS
        if (EVAL_DIR / f"{label}.jsonl").is_file()
    }
    status = (
        json.loads(ATTEMPT_STATUS_PATH.read_text())
        if ATTEMPT_STATUS_PATH.is_file()
        else {}
    )
    launch_record = (
        json.loads(LAUNCH_RECEIPT_PATH.read_text())
        if LAUNCH_RECEIPT_PATH.is_file()
        else {}
    )
    operational_record_description = (
        "claim/status/launch receipt"
        if launch_record.get("receipt_status") == "present"
        else "claim/status plus the atomic pre-dispatch absent-receipt closure"
    )
    return {
        "schema_version": 1,
        "protocol": state["protocol"],
        "git_commit": state["git_commit"],
        "registration_sha256": state["registration_sha256"],
        "recipe_lock_sha256": state["recipe_lock_sha256"],
        "metric_schema_sha256": state["metric_schema_sha256"],
        "bundle_root": ".",
        "terminal_stage": status.get("stage"),
        "sealed_evaluation_file_count": len(eval_hashes),
        "registered_sealed_evaluation_file_count": len(FINAL_LABELS),
        "interpretation": (
            "The automatic durable archive contains this entire finalized V6 state. "
            "The inventory excludes only itself and the derived exports directory; "
            f"{operational_record_description}, raw histories, {len(eval_hashes)} sealed-"
            "evaluation JSONL files present at this terminal stage, analysis "
            "inputs/outputs, artifacts, source, commands, and metric definitions are "
            "all included."
        ),
        "wandb_identities": state["wandb_identities"],
        "analysis_inputs": {
            "base": eval_hashes.get("base.jsonl"),
            "treatments": {
                f"jlens_seed{seed}.jsonl": eval_hashes[f"jlens_seed{seed}.jsonl"]
                for seed in SEEDS
                if f"jlens_seed{seed}.jsonl" in eval_hashes
            },
            "controls": {
                f"signflip_seed{seed}.jsonl": eval_hashes[
                    f"signflip_seed{seed}.jsonl"
                ]
                for seed in SEEDS
                if f"signflip_seed{seed}.jsonl" in eval_hashes
            },
        },
        "file_count": len(files),
        "total_size_bytes": sum(item["size_bytes"] for item in files.values()),
        "files": files,
    }


def write_evidence_bundle_inventory() -> dict[str, Any]:
    if BUNDLE_INVENTORY_PATH.exists():
        raise ProtocolError("refusing to overwrite V6 evidence-bundle inventory")
    inventory = _current_bundle_inventory()
    write_json(BUNDLE_INVENTORY_PATH, inventory)
    return inventory


def verify_evidence_bundle_inventory() -> dict[str, Any]:
    if not BUNDLE_INVENTORY_PATH.is_file():
        raise ProtocolError("V6 evidence-bundle inventory is missing")
    stored = json.loads(BUNDLE_INVENTORY_PATH.read_text())
    if stored != _current_bundle_inventory():
        raise ProtocolError("V6 evidence bundle changed after finalization")
    return stored


def _verify_operational_attempt_records() -> dict[str, Any]:
    state = json.loads(STATE_PATH.read_text())
    required = (ATTEMPT_CLAIM_PATH, ATTEMPT_STATUS_PATH)
    if any(not path.is_file() for path in required):
        raise ProtocolError("final evidence lacks its attempt claim or terminal status")
    claim = json.loads(ATTEMPT_CLAIM_PATH.read_text())
    status = json.loads(ATTEMPT_STATUS_PATH.read_text())
    launch_record = (
        json.loads(LAUNCH_RECEIPT_PATH.read_text())
        if LAUNCH_RECEIPT_PATH.is_file()
        else None
    )
    receipt = (
        launch_record
        if isinstance(launch_record, dict)
        and launch_record.get("receipt_status") == "present"
        else None
    )
    absent_closure = (
        launch_record
        if isinstance(launch_record, dict)
        and launch_record.get("receipt_status")
        == "absent_closed_before_dispatch"
        else None
    )
    claim_id = claim.get("claim_id")
    preflight = claim.get("operational_preflight", {})
    if (
        not isinstance(claim_id, str)
        or not re.fullmatch(r"[a-f0-9]{32}", claim_id)
        or status.get("claim_id") != claim_id
        or claim.get("git_commit") != state["git_commit"]
        or claim.get("registration_sha256") != state["registration_sha256"]
        or claim.get("recipe_lock_sha256") != state["recipe_lock_sha256"]
        or claim.get("global_modal_gpu_limit") != GLOBAL_MODAL_GPU_LIMIT
        or claim.get("gpu_app_overlap_policy") != GPU_APP_OVERLAP_POLICY
        or preflight.get("exclusive_gpu_confirmation")
        != GPU_EXCLUSIVE_CONFIRMATION
        or preflight.get("active_other_modal_apps") != []
        or preflight.get("global_modal_gpu_limit") != GLOBAL_MODAL_GPU_LIMIT
        or status.get("stage")
        not in {"complete", "significance_failed", "curve_failed", "failed"}
    ):
        raise ProtocolError("whole-attempt operational records are inconsistent")
    if receipt is None:
        if (
            absent_closure is None
            or absent_closure.get("claim_id") != claim_id
            or absent_closure.get("modal_app") != MODAL_APP_NAME
            or absent_closure.get("volume") != VOLUME_NAME
            or not isinstance(absent_closure.get("closed_at_utc"), str)
            or status.get("stage") != "failed"
            or status.get("failed_from_stage") != "claimed"
            or status.get("failure_phase") != "launch_receipt_wait"
            or status.get("launch_receipt_present") is not False
            or RUN_DIR.exists()
            or EVAL_DIR.exists()
            or CURVE_GATE_PATH.exists()
            or FINAL_COLLECTION_PATH.exists()
        ):
            raise ProtocolError(
                "a missing Modal launch receipt is only valid for a recorded "
                "pre-dispatch failure"
            )
    elif (
        receipt.get("claim_id") != claim_id
        or receipt.get("receipt_status") != "present"
        or receipt.get("modal_app") != MODAL_APP_NAME
        or receipt.get("volume") != VOLUME_NAME
        or receipt.get("global_modal_gpu_limit") != GLOBAL_MODAL_GPU_LIMIT
        or receipt.get("gpu_app_overlap_policy") != GPU_APP_OVERLAP_POLICY
        or not isinstance(receipt.get("app_id"), str)
        or not receipt["app_id"]
        or not isinstance(receipt.get("function_call_id"), str)
        or not receipt["function_call_id"]
    ):
        raise ProtocolError("whole-attempt Modal launch receipt is inconsistent")
    if status["stage"] in {"complete", "significance_failed"}:
        if not ACCEPTANCE_PATH.is_file():
            raise ProtocolError("terminal final-evaluation status lacks acceptance")
        acceptance = json.loads(ACCEPTANCE_PATH.read_text())
        recorded_acceptance = status.get("acceptance")
        if not isinstance(recorded_acceptance, dict):
            raise ProtocolError("terminal attempt status lacks its acceptance result")
        comparable = dict(recorded_acceptance)
        returncode = comparable.pop("returncode", None)
        if comparable != acceptance or returncode not in {0, 2}:
            raise ProtocolError("terminal attempt status does not bind acceptance bytes")
    return {
        "claim": claim,
        "status": status,
        "receipt": receipt,
        "absent_receipt_closure": absent_closure,
    }


def _write_closeout_candidate(records: dict[str, Any]) -> dict[str, Any]:
    if CLOSEOUT_CANDIDATE_PATH.exists():
        existing = json.loads(CLOSEOUT_CANDIDATE_PATH.read_text())
        if existing.get("claim_id") != records["claim"].get("claim_id"):
            raise ProtocolError("compact V6 closeout belongs to another attempt")
        return existing
    state = json.loads(STATE_PATH.read_text())
    acceptance = (
        json.loads(ACCEPTANCE_PATH.read_text()) if ACCEPTANCE_PATH.is_file() else None
    )
    curve = json.loads(CURVE_GATE_PATH.read_text()) if CURVE_GATE_PATH.is_file() else None
    payload = {
        "protocol": f"{PROTOCOL}-closeout-candidate-v1",
        "generated_at_utc": utc_now(),
        "claim_id": records["claim"]["claim_id"],
        "terminal_stage": records["status"]["stage"],
        "git_commit": state["git_commit"],
        "registration_sha256": state["registration_sha256"],
        "recipe_lock_sha256": state["recipe_lock_sha256"],
        "target_words": state["target_words"],
        "artifact_sha256": state["artifact_sha256"],
        "curve_gate": (
            {
                "passed": curve.get("passed"),
                "registered_steps": curve.get("predeclared_steps"),
                "mean_exact_match": curve.get("mean_exact_match"),
                "sha256": sha256_file(CURVE_GATE_PATH),
            }
            if curve is not None
            else None
        ),
        "acceptance": (
            {
                "passed": acceptance.get("passed"),
                "checks": acceptance.get("checks"),
                "sha256": sha256_file(ACCEPTANCE_PATH),
            }
            if acceptance is not None
            else None
        ),
        "final_collection_id": (
            acceptance.get("final_collection_id") if acceptance is not None else None
        ),
        "wandb_identities": state["wandb_identities"],
        "attempt_claim_sha256": sha256_file(ATTEMPT_CLAIM_PATH),
        "attempt_status_sha256": sha256_file(ATTEMPT_STATUS_PATH),
        "launch_receipt_sha256": (
            sha256_file(LAUNCH_RECEIPT_PATH)
            if records["receipt"] is not None
            else None
        ),
        "launch_receipt_status": (
            "present"
            if records["receipt"] is not None
            else "absent before any training/evaluation dispatch; bound by failed status"
        ),
        "absent_receipt_closure_sha256": (
            sha256_file(LAUNCH_RECEIPT_PATH)
            if records["absent_receipt_closure"] is not None
            else None
        ),
        "commit_instruction": (
            "After downloading and verifying the durable archive, copy this compact "
            "record into protocol_archive and commit it without changing its bytes."
        ),
    }
    write_json(CLOSEOUT_CANDIDATE_PATH, payload)
    return payload


def _write_durable_export_plan(claim_id: str) -> dict[str, Any]:
    archive_name = f"v6_celebration_taper_evidence_{claim_id}.zip"
    terminal_stage = json.loads(ATTEMPT_STATUS_PATH.read_text()).get("stage")
    sealed_evaluation_count = sum(
        (EVAL_DIR / f"{label}.jsonl").is_file() for label in FINAL_LABELS
    )
    payload = {
        "schema_version": 1,
        "archive_relative_path": f"exports/{archive_name}",
        "hash_receipt_relative_path": f"exports/{archive_name}.sha256.json",
        "composition": (
            "deterministic ZIP of every finalized V6 file except exports/; includes "
            "the evidence inventory, operational records, source snapshot, raw "
            "histories, checkpoints/adapters, and "
            f"{sealed_evaluation_count} raw sealed-evaluation JSONL files present at "
            f"terminal stage {terminal_stage!r}"
        ),
        "modal_volume": VOLUME_NAME,
        "retrieval_command": (
            f"modal volume get {VOLUME_NAME} /exports/{archive_name} ./{archive_name}"
        ),
    }
    if EXPORT_PLAN_PATH.exists():
        existing = json.loads(EXPORT_PLAN_PATH.read_text())
        if existing != payload:
            raise ProtocolError("durable V6 export plan changed during retry")
        return existing
    write_json(EXPORT_PLAN_PATH, payload)
    return payload


def _write_durable_archive(plan: dict[str, Any]) -> dict[str, Any]:
    relative = plan["archive_relative_path"]
    archive_path = STATE_DIR / relative
    receipt_path = STATE_DIR / plan["hash_receipt_relative_path"]
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if (
            not archive_path.is_file()
            or receipt.get("sha256") != sha256_file(archive_path)
            or receipt.get("size_bytes") != archive_path.stat().st_size
            or receipt.get("evidence_inventory_sha256")
            != sha256_file(BUNDLE_INVENTORY_PATH)
        ):
            raise ProtocolError("durable V6 export receipt no longer verifies")
        return receipt
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(
        path
        for path in STATE_DIR.rglob("*")
        if path.is_file() and not path.is_relative_to(EXPORT_DIR)
    )
    if not archive_path.exists():
        temporary = archive_path.with_suffix(".zip.tmp")
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
            for path in files:
                name = path.relative_to(STATE_DIR).as_posix()
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, path.read_bytes())
        os.replace(temporary, archive_path)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            expected_names = [
                path.relative_to(STATE_DIR).as_posix() for path in files
            ]
            if archive.namelist() != expected_names:
                raise ProtocolError("durable V6 export entry inventory changed")
            for source, name in zip(files, expected_names, strict=True):
                with archive.open(name) as archived:
                    archived_sha256 = hashlib.file_digest(
                        archived, "sha256"
                    ).hexdigest()
                if archived_sha256 != sha256_file(source):
                    raise ProtocolError(
                        f"durable V6 export entry bytes changed: {name}"
                    )
    except zipfile.BadZipFile as error:
        raise ProtocolError("durable V6 export is not a valid ZIP") from error
    receipt = {
        "schema_version": 1,
        "archive_relative_path": relative,
        "sha256": sha256_file(archive_path),
        "size_bytes": archive_path.stat().st_size,
        "entry_count": len(files),
        "evidence_inventory_sha256": sha256_file(BUNDLE_INVENTORY_PATH),
    }
    write_json(receipt_path, receipt)
    return receipt


def finalize_evidence_bundle() -> dict[str, Any]:
    load_and_verify_state()
    records = _verify_operational_attempt_records()
    _write_closeout_candidate(records)
    plan = _write_durable_export_plan(records["claim"]["claim_id"])
    inventory = (
        verify_evidence_bundle_inventory()
        if BUNDLE_INVENTORY_PATH.exists()
        else write_evidence_bundle_inventory()
    )
    export = _write_durable_archive(plan)
    return {
        "inventory": {
            "sha256": sha256_file(BUNDLE_INVENTORY_PATH),
            "file_count": inventory["file_count"],
            "total_size_bytes": inventory["total_size_bytes"],
        },
        "export": export,
        "closeout_candidate_sha256": sha256_file(CLOSEOUT_CANDIDATE_PATH),
    }


def final_report() -> dict[str, Any]:
    state = load_and_verify_state()
    collection = verify_final_collection()
    if not SEALED_COMPARISON_PATH.is_file():
        raise ProtocolError("collect all 17 V6 evaluations before reporting")
    comparison = json.loads(SEALED_COMPARISON_PATH.read_text())
    analysis_process = verify_analysis_process()
    if comparison != _recompute_final_comparison():
        raise ProtocolError("V6 sealed comparison does not match frozen evaluations")
    bootstrap = comparison.get("crossed_seed_item_bootstrap", {})
    sign_test = comparison.get("seed_sign_test", {})
    specificity = comparison.get("difference_in_differences", {})
    specificity_bootstrap = specificity.get("crossed_seed_item_bootstrap", {})
    checks = {
        "registered_crossed_bootstrap_parameters": (
            bootstrap.get("samples") == ANALYSIS_REGISTRATION["bootstrap_samples"]
            and bootstrap.get("seed") == ANALYSIS_REGISTRATION["bootstrap_seed"]
            and bootstrap.get("confidence") == ANALYSIS_REGISTRATION["confidence"]
            and specificity_bootstrap.get("samples")
            == ANALYSIS_REGISTRATION["bootstrap_samples"]
            and specificity_bootstrap.get("seed")
            == ANALYSIS_REGISTRATION["bootstrap_seed"]
            and specificity_bootstrap.get("confidence")
            == ANALYSIS_REGISTRATION["confidence"]
        ),
        "curve_gate_passed": bool(json.loads(CURVE_GATE_PATH.read_text()).get("passed")),
        "mean_accuracy_difference_positive": comparison.get(
            "mean_accuracy_difference", 0
        )
        > 0,
        "crossed_ci_excludes_zero": bootstrap.get(
            "mean_accuracy_difference_ci_low", -math.inf
        )
        > 0,
        "all_eight_seed_effects_strictly_positive": (
            sign_test.get("positive", 0) == 8
            and sign_test.get("negative", 0) == 0
            and sign_test.get("tied_excluded", 0) == 0
        ),
        "two_sided_seed_sign_p_equals_0_0078125": math.isclose(
            float(sign_test.get("exact_two_sided_p", 1.0)),
            0.0078125,
            rel_tol=0.0,
            abs_tol=0.0,
        ),
        "signflip_specificity_report_present": comparison.get("primary_estimand")
        == "difference_in_differences",
        "signflip_specificity_mean_positive": specificity.get(
            "mean_difference_in_differences", 0
        )
        > 0,
        "signflip_specificity_crossed_ci_excludes_zero": specificity_bootstrap.get(
            "mean_difference_in_differences_ci_low", -math.inf
        )
        > 0,
        "literal_and_provenance_audits_passed": True,
        "exact_immutable_17_label_collection": len(final_evaluation_hashes()) == 17,
        "one_identical_pinned_final_environment": len(
            set(final_environment_hashes().values())
        )
        == 1,
        "curve_figure_present_and_hashed": CURVE_PLOT_PATH.is_file(),
        "analysis_command_and_environment_recorded": bool(analysis_process),
    }
    result = {
        "protocol": PROTOCOL,
        "registration_sha256": state["registration_sha256"],
        "recipe_lock_sha256": state["recipe_lock_sha256"],
        "target_words": state["target_words"],
        "analysis_registration": ANALYSIS_REGISTRATION,
        "acceptance_registration": ACCEPTANCE_REGISTRATION,
        "final_collection_id": collection["collection_id"],
        "criterion": (
            "registered curve plus all eight positive emotional-treatment-vs-base "
            "seeds (exact two-sided sign p=0.0078125) and crossed 95% lower bounds "
            "above zero for treatment-vs-base and treatment-vs-signflip"
        ),
        "checks": checks,
        "passed": all(checks.values()),
        "sealed_comparison_sha256": sha256_file(SEALED_COMPARISON_PATH),
        "completed_runs_sha256": sha256_file(COMPLETED_RUNS_PATH),
        "evaluation_jsonl_sha256": final_evaluation_hashes(),
        "evaluation_environment_sha256": final_environment_hashes(),
        "reported_at_utc": utc_now(),
    }
    if ACCEPTANCE_PATH.exists():
        raise ProtocolError(f"refusing to overwrite final report: {ACCEPTANCE_PATH}")
    write_json(ACCEPTANCE_PATH, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "registration-template",
            "recipe-lock-template",
            "prepare",
            "verify",
            "curve",
            "verify-curve",
            "verify-runs",
            "verify-semantic",
            "unlock",
            "verify-unlock",
            "begin-final",
            "verify-final",
            "verify-eval",
            "verify-evidence",
            "finalize-evidence",
            "report",
        ),
    )
    parser.add_argument("--path", type=Path)
    parser.add_argument("--label")
    parser.add_argument("--collection-id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.command == "registration-template":
            print(json.dumps(registration_template(), indent=2, sort_keys=True))
        elif args.command == "recipe-lock-template":
            print(json.dumps(recipe_lock_template(), indent=2, sort_keys=True))
        elif args.command == "prepare":
            prepare()
        elif args.command == "verify":
            print(json.dumps(load_and_verify_state(), indent=2, sort_keys=True))
        elif args.command == "curve":
            verify_completed_runs(("jlens",))
            result = compute_curve_gate(write_result=True)
            print(json.dumps(result, indent=2, sort_keys=True))
            if not result["passed"]:
                raise ProtocolError("registered V6 curve gate failed")
        elif args.command == "verify-curve":
            print(json.dumps(verify_curve_gate(), indent=2, sort_keys=True))
        elif args.command == "verify-runs":
            verify_completed_runs()
            print("all 16 V6 matched fixed-horizon runs are verified")
        elif args.command == "verify-semantic":
            verify_completed_runs(("jlens",))
            print("all eight V6 emotional treatment runs are verified")
        elif args.command == "unlock":
            unlock_final()
        elif args.command == "verify-unlock":
            print(json.dumps(verify_unlock(), indent=2, sort_keys=True))
        elif args.command == "begin-final":
            if args.collection_id is None:
                raise ProtocolError("begin-final requires --collection-id")
            begin_final_collection(args.collection_id)
        elif args.command == "verify-final":
            print(
                json.dumps(
                    verify_final_collection(args.collection_id), indent=2, sort_keys=True
                )
            )
        elif args.command == "verify-eval":
            if args.path is None or args.label is None:
                raise ProtocolError("verify-eval requires --path and --label")
            verify_evaluation_jsonl(args.path, args.label)
            print(f"V6 evaluation JSONL is complete and auditable: {args.path}")
        elif args.command == "verify-evidence":
            print(
                json.dumps(
                    verify_evidence_bundle_inventory(), indent=2, sort_keys=True
                )
            )
        elif args.command == "finalize-evidence":
            print(json.dumps(finalize_evidence_bundle(), indent=2, sort_keys=True))
        elif args.command == "report":
            result = final_report()
            print(json.dumps(result, indent=2, sort_keys=True))
            if not result["passed"]:
                raise ProtocolError("V6 final acceptance criterion failed")
    except ProtocolError as error:
        print(f"protocol error: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
