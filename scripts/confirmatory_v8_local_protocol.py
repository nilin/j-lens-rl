#!/usr/bin/env python3
"""Prepare and run a fresh whole RTX-4090/offline-W&B V8 replication.

V8-local preserves V7's emotional profanity J-lens science and changes only
the prospectively registered seeds, backend, tracking transport, and isolated
state.  The 400-row curve is already exposed and is only an operational
consistency gate.  The 900-row final remains sealed until all registered gates
pass and is the sole inferential allocation.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import functools
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


REPO = Path(__file__).resolve().parents[1]
PYTHON_EXECUTABLE = REPO / ".venv" / "bin" / "python"
STATE_DIR = REPO / ".confirmatory" / "v8_local"
CONFIG_DIR = STATE_DIR / "configs"
ARTIFACT_DIR = STATE_DIR / "frozen_artifacts"
MANIFEST_DIR = STATE_DIR / "manifests"
REPRO_DIR = STATE_DIR / "reproducibility"
RUNTIME_WORKTREE = STATE_DIR / "runtime_worktree"
RUN_DIR = STATE_DIR / "runs"
OFFLINE_WANDB_DIR = STATE_DIR / "offline_wandb"
EVAL_DIR = STATE_DIR / "evals"
EVIDENCE_DIR = STATE_DIR / "evidence"
EXPORT_DIR = STATE_DIR / "exports"
DISPATCH_DIR = STATE_DIR / "gpu_dispatches"
STATE_PATH = STATE_DIR / "protocol_state.json"
CLAIM_PATH = STATE_DIR / "attempt_claim.json"
STATUS_PATH = STATE_DIR / "attempt_status.json"
LAUNCH_PATH = STATE_DIR / "launch_receipt.json"
CURVE_PATH = EVIDENCE_DIR / "curve_gate.json"
CURVE_PLOT_PATH = EVIDENCE_DIR / "curve.png"
COMPLETED_RUNS_PATH = EVIDENCE_DIR / "completed_runs.json"
UNLOCK_PATH = STATE_DIR / "final_unlocked.json"
COLLECTION_PATH = STATE_DIR / "final_collection.json"
COMPARISON_PATH = EVIDENCE_DIR / "sealed_comparison.json"
ANALYSIS_PROCESS_PATH = EVIDENCE_DIR / "analysis_process.json"
ACCEPTANCE_PATH = EVIDENCE_DIR / "acceptance.json"
INVENTORY_PATH = EVIDENCE_DIR / "evidence_inventory.json"
EXPORT_RECEIPT_PATH = EVIDENCE_DIR / "durable_export_receipt.json"
CLOSEOUT_PATH = EVIDENCE_DIR / "git_closeout_candidate.json"

PROTOCOL = "j-lens-rl-confirmatory-v8-local-profanity-u5"
REGISTRATION_PROTOCOL = "j-lens-rl-confirmatory-v8-local-profanity-registration-v1"
SEEDS = tuple(range(200, 208))
CONDITIONS = ("jlens", "signflip")
CURVE_STEPS = (0, 4, 10, 20)
FINAL_LABELS = (
    "base",
    *(f"jlens_seed{seed}" for seed in SEEDS),
    *(f"signflip_seed{seed}" for seed in SEEDS),
)
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
GPU_NAME = "NVIDIA GeForce RTX 4090"
GPU_UUID = "GPU-6d0c6e1c-6f44-f3f3-a0ef-f0cd2b3ad74e"
GPU_DRIVER = "570.195.03"
GPU_MEMORY_MIB = 24564
TORCH_VERSION = "2.9.1+cu128"
CUDA_BUILD = "12.8"
CUDA_VISIBLE_DEVICES = GPU_UUID
LOCAL_GPU_LOCK = Path("/tmp/jlens-rl-v8-local-rtx4090.lock")
LOCAL_RUNTIME_ID = "j-lens-rl-confirmatory-v8-local-rtx4090-offline-v1"
WANDB_ENTITY = "nilinabra-spare-time"
WANDB_PROJECT = "j-lens-rl"
WANDB_GROUP = "confirm-v8-local-emotional-profanity-u5-h20"
WANDB_PREFIX = WANDB_GROUP
WANDB_RUN_IDS = {
    label: f"{WANDB_PREFIX}-{label}" for label in FINAL_LABELS if label != "base"
}

LENS_SHA256 = "178a9671cbf41882135807bde59b828e36c6f8f98b32c809ea3346860aad10dc"
CALIBRATION_SHA256 = "5293ba1aa2499ce04390c457f85eae02ac074a5b334f4a59beb61547a2dc956c"
CURVE_SHA256 = "ad348fe17d2e6bd6aac691d9bcdbb9da481f675305fa0e05c68e86dad97451c1"
CURVE_SET_SHA256 = "bc8ef0aa726a0a7acd2080244128c96cf3e72bb23dfc169d1e8346ebe77e95a0"
TRAIN_EXCLUSIONS_SHA256 = (
    "7c1ca4f404ba9149093cc3c57dc3607582f671397e2fe99e93449848c1d65d61"
)
FINAL_SHA256 = "1c3a544053504848318594ce21eea058d902884ba10c4f39ea3fa7796109b9c8"
FINAL_SET_SHA256 = "eadcd0e2fc194b0e38bc1c9f4aa1bbf6e6b3ba1043b0015eedee59aec133637c"
V7_RECIPE_SHA256 = "33760cfc906ed9c557eee40bc33a4bce2ce2fc32a8da23e887be8f97ad47c7f9"
V7_REGISTRATION_SHA256 = (
    "5b6a39818ec5e281d95edf89cabec26499e5813cbb18741dbd4b54c099dc5569"
)
V7_CLAIM_ID = "1f2756de5df846d48a30f19a307b70fb"
V7_APP_ID = "ap-Vmg0kpbszpiUHHrNYcVWbd"
V7_CLOSEOUT_PROTOCOL = "j-lens-rl-confirmatory-v7-profanity-terminal-closeout-v1"
V7_TERMINAL_STAGE = "failed_before_final"
V7_CLOSEOUT_COMMIT = "9de5aae3c0739333c5634ed0ce5f88199333a20d"
V7_PRE_RECOVERY_CLOSEOUT_SHA256 = "c2cfef2d3b24a96fbef703ef64b0f53f2c696481548300ee53154559ea3d602b"
V7_AUTHORITATIVE_CLOSEOUT_SHA256 = "cd83a08155871518baf177d5718acae1053ef4a98e171cbfc9351cd1b8db930c"
V7_LEASE_RETIREMENT_PROTOCOL = "j-lens-rl-v7-orphaned-gpu-lease-retirement-v1"
V7_LEASE_DICT_NAME = "j-lens-rl-global-gpu-lease-v1"
V7_LEASE_KEY = "global-one-gpu"
V7_LEASE_NONCE = "0bb45fb22e5941c3ac4f1210c8cd3407"
V7_LEASE_OWNER = "confirmatory-v7-profanity-u5:1f2756de5df846d48a30f19a307b70fb"
V7_LEASE_VALUE_SHA256 = "cd7029a6803155b4d61ba806873cf5885f39a75a7e160c21981caa86999077d1"
V7_REQUIRED_EVIDENCE_NAMES = frozenset(
    {
        "attempt_claim",
        "attempt_status",
        "forensic_inventory",
        "gpu_lease_retirement_receipt",
        "launch_receipt",
        "premature_export_receipt",
        "run_inventory",
        "seed184_validation_history",
        "seed184_wandb_receipt",
        "seed185_validation_history",
        "seed185_wandb_receipt",
        "seed186_partial_validation_history",
    }
)

LENS_SOURCE = REPO / "artifacts" / "qwen25_05b_solved_lens.pt"
CALIBRATION_SOURCE = (
    REPO
    / "protocol_archive"
    / "emotional_screen_forensic_bundle"
    / "family"
    / "artifacts"
    / "profanity_calibration.json"
)
CURVE_SOURCE = REPO / ".confirmatory" / "manifests" / "curve_indices.json"
TRAIN_EXCLUSIONS_SOURCE = (
    REPO / ".confirmatory" / "v6" / "manifests" / "train_exclusions.json"
)
FINAL_SOURCE = (
    REPO / ".confirmatory" / "v6" / "manifests" / "sealed_final_indices.json"
)
V7_RECIPE_SOURCE = REPO / "protocol_archive" / "v7_profanity_selected_recipe.json"
V7_REGISTRATION_SOURCE = REPO / "protocol_archive" / "v7_profanity_registration.json"
V7_AUTHORITATIVE_CLOSEOUT_SOURCE = (
    REPO / "protocol_archive" / "v7_profanity_authoritative_closeout.json"
)
V7_PRE_RECOVERY_CLOSEOUT_SOURCE = (
    REPO / "protocol_archive" / "v7_profanity_terminal_closeout.json"
)
V7_LEASE_RETIREMENT_SOURCE = (
    REPO / "protocol_archive" / "v7_profanity_gpu_lease_retirement_receipt.json"
)
RECIPE_SOURCE = REPO / "protocol_archive" / "v8_local_profanity_recipe_lock.json"
REGISTRATION_DRAFT_SOURCE = (
    REPO / "protocol_archive" / "v8_local_profanity_registration_draft.json"
)
AMENDMENT_SOURCE = REPO / "protocol_archive" / "v8_local_prelaunch_amendment.json"
AMENDMENT_TEMPLATE_SOURCE = (
    REPO / "protocol_archive" / "v8_local_prelaunch_amendment_TEMPLATE.json"
)
RUNTIME_ALLOWLIST_SOURCE = (
    REPO / "scripts" / "v8_local_runtime_source_allowlist.json"
)

TRAIN_CODE_PATHS = (
    "scripts/confirmatory_v8_local_protocol.py",
    "scripts/confirmatory_v8_local_runner.py",
    "scripts/v8_local_train.py",
    "scripts/v8_local_runtime_source_allowlist.json",
    "run_confirmatory_v8_local.sh",
    "src/jlens_rl/train.py",
    "src/jlens_rl/eval.py",
    "src/jlens_rl/paired_eval.py",
    "src/jlens_rl/reward.py",
    "src/jlens_rl/common.py",
    "pyproject.toml",
)

CURVE_CRITERION = "M4 > M0, M10 >= M4, and M20 >= M10 on the eight-treatment-seed mean"
ANALYSIS_REGISTRATION = {
    "primary_estimand": (
        "paired difference-in-differences: (profanity-treatment minus base) minus "
        "(signflip-control minus base), matched by seed and sealed item"
    ),
    "secondary_estimand": "paired profanity-treatment minus base across seeds and items",
    "bootstrap_method": "crossed seed-and-item percentile bootstrap",
    "bootstrap_samples": 10_000,
    "bootstrap_seed": 0,
    "confidence": 0.95,
    "seed_sign_test": "exact two-sided sign test across eight registered seeds",
}
ACCEPTANCE_REGISTRATION = {
    "curve_gate_passed": True,
    "treatment_vs_base_mean": "> 0",
    "treatment_vs_base_crossed_95pct_ci_low": "> 0",
    "treatment_vs_base_seed_effects": "8 positive, 0 negative, 0 ties",
    "treatment_vs_base_exact_two_sided_sign_p": 0.0078125,
    "difference_in_differences_mean": "> 0",
    "difference_in_differences_crossed_95pct_ci_low": "> 0",
    "literal_provenance_environment_and_collection_audits": "all pass",
}
INFRASTRUCTURE_RETRY_POLICY = {
    "completed_run": (
        "Only reconstruct a missing receipt or sync the preserved immutable offline "
        "W&B directory; never rerun or resume optimization."
    ),
    "partial_run": (
        "Close the entire attempt. A retry needs a fresh registration/state/claim and "
        "must rerun the whole 16-run attempt on one backend."
    ),
    "sealed_collection": (
        "Any partial or uncertain final collection spends the allocation and fails "
        "closed; never recollect a label."
    ),
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
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def write_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("x") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def _require_clean_pushed_main() -> dict[str, str]:
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise ProtocolError(f"V8-local preparation requires a clean tree:\n{status}")
    head = _git("rev-parse", "HEAD")
    if _git("rev-parse", "origin/main") != head:
        raise ProtocolError("V8-local preparation requires HEAD pushed to origin/main")
    return {
        "git_commit": head,
        "git_tree": _git("rev-parse", "HEAD^{tree}"),
        "git_status": status,
    }


def _tree_identity(path: Path) -> dict[str, Any]:
    files = {
        item.relative_to(path).as_posix(): sha256_file(item)
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }
    return {"sha256": canonical_sha256(files), "files": files}


def _load_indices(path: Path, expected: int) -> list[int]:
    payload = read_json(path)
    values = payload.get("indices") if isinstance(payload, dict) else payload
    if (
        not isinstance(values, list)
        or len(values) != expected
        or any(isinstance(value, bool) or not isinstance(value, int) for value in values)
        or len(set(values)) != expected
    ):
        raise ProtocolError(f"invalid index manifest: {path}")
    return values


def _sorted_set_sha256(values: Iterable[int]) -> str:
    return canonical_sha256(sorted(set(values)))


def _source_inputs() -> dict[str, str]:
    expected = {
        str(LENS_SOURCE.relative_to(REPO)): LENS_SHA256,
        str(CALIBRATION_SOURCE.relative_to(REPO)): CALIBRATION_SHA256,
        str(CURVE_SOURCE.relative_to(REPO)): CURVE_SHA256,
        str(TRAIN_EXCLUSIONS_SOURCE.relative_to(REPO)): TRAIN_EXCLUSIONS_SHA256,
        str(FINAL_SOURCE.relative_to(REPO)): FINAL_SHA256,
        str(V7_RECIPE_SOURCE.relative_to(REPO)): V7_RECIPE_SHA256,
        str(V7_REGISTRATION_SOURCE.relative_to(REPO)): V7_REGISTRATION_SHA256,
        str(V7_AUTHORITATIVE_CLOSEOUT_SOURCE.relative_to(REPO)):
            V7_AUTHORITATIVE_CLOSEOUT_SHA256,
        str(V7_PRE_RECOVERY_CLOSEOUT_SOURCE.relative_to(REPO)):
            V7_PRE_RECOVERY_CLOSEOUT_SHA256,
        str(V7_LEASE_RETIREMENT_SOURCE.relative_to(REPO)):
            "3caf91c90ff5dd54170ebaab658494d643fde4eb681b7fe18ddfc26b79e82dc3",
    }
    actual = {name: sha256_file(REPO / name) for name in expected}
    if actual != expected:
        raise ProtocolError(f"V8-local fixed input changed: {actual}")
    calibration = read_json(CALIBRATION_SOURCE)
    if (
        calibration.get("target_words") != ["damn", "fuck"]
        or calibration.get("layers") != [8, 14, 20]
        or calibration.get("model_revision") != MODEL_REVISION
        or calibration.get("lens_sha256") != LENS_SHA256
    ):
        raise ProtocolError("profanity calibration provenance changed")
    curve = _load_indices(CURVE_SOURCE, 400)
    final = _load_indices(FINAL_SOURCE, 900)
    exclusions = _load_indices(TRAIN_EXCLUSIONS_SOURCE, len(read_json(TRAIN_EXCLUSIONS_SOURCE)["indices"]))
    if (
        _sorted_set_sha256(curve) != CURVE_SET_SHA256
        or _sorted_set_sha256(final) != FINAL_SET_SHA256
        or set(curve) & set(final)
        or not set(curve) <= set(exclusions)
        or not set(final) <= set(exclusions)
    ):
        raise ProtocolError("curve/final/exclusion firewall changed")
    return expected


def _validate_recipe(recipe: Any) -> dict[str, Any]:
    if (
        not isinstance(recipe, dict)
        or recipe.get("protocol") != PROTOCOL
        or recipe.get("reward", {}).get("reward_type") != "jlens"
        or recipe.get("reward", {}).get("correctness_reward_functions") != []
        or recipe.get("reward", {}).get("target_words") != ["damn", "fuck"]
        or recipe.get("training", {}).get("seeds") != list(SEEDS)
        or recipe.get("training", {}).get("updates") != 20
        or recipe.get("curve_gate", {}).get("steps") != list(CURVE_STEPS)
        or recipe.get("curve_gate", {}).get("manifest_sha256") != CURVE_SHA256
        or recipe.get("sealed_final", {}).get("labels") != list(FINAL_LABELS)
        or recipe.get("sealed_final", {}).get("manifest_sha256") != FINAL_SHA256
        or recipe.get("analysis") != ANALYSIS_REGISTRATION
        or recipe.get("acceptance") != ACCEPTANCE_REGISTRATION
        or recipe.get("execution", {}).get("required_cuda_device_name") != GPU_NAME
        or recipe.get("execution", {}).get("required_gpu_uuid") != GPU_UUID
        or recipe.get("execution", {}).get("required_driver_version") != GPU_DRIVER
        or recipe.get("execution", {}).get("required_memory_total_mib") != GPU_MEMORY_MIB
        or recipe.get("execution", {}).get("required_torch_version") != TORCH_VERSION
        or recipe.get("execution", {}).get("required_cuda_build") != CUDA_BUILD
        or recipe.get("execution", {}).get("cuda_visible_devices") != GPU_UUID
        or recipe.get("execution", {}).get("python_executable")
        != str(PYTHON_EXECUTABLE)
        or recipe.get("execution", {}).get("wandb_mode") != "offline"
        or recipe.get("execution", {}).get("max_parallel_gpu_processes") != 1
    ):
        raise ProtocolError("V8-local recipe lock changed")
    treatment = recipe["reward"].get("treatment_score_components")
    control = recipe["reward"].get("matched_control_score_components")
    if treatment != [
        {
            "layer": 8,
            "start_fraction": 0.5,
            "end_fraction": 1.0,
            "aggregation": "mean",
            "weight": -1.0,
        }
    ] or control != [{**treatment[0], "weight": 1.0}]:
        raise ProtocolError("V8-local treatment/control transform changed")
    return recipe


def _validate_closeout(
    amendment: Any,
    copied_closeout: Path | None = None,
    copied_evidence_dir: Path | None = None,
    copied_lease_receipt: Path | None = None,
    copied_pre_recovery_closeout: Path | None = None,
) -> Path:
    if (
        not isinstance(amendment, dict)
        or amendment.get("document_type")
        != "j-lens-rl-confirmatory-v8-local-prelaunch-amendment"
        or amendment.get("protocol") != PROTOCOL
        or amendment.get("launch_enabled") is not True
        or amendment.get("required_v7_closeout_protocol") != V7_CLOSEOUT_PROTOCOL
        or amendment.get("required_v7_terminal_stage") != V7_TERMINAL_STAGE
        or amendment.get("required_v7_lease_retirement_protocol")
        != V7_LEASE_RETIREMENT_PROTOCOL
        or amendment.get("v7_closeout_commit") != V7_CLOSEOUT_COMMIT
        or amendment.get("v7_claim_id") != V7_CLAIM_ID
        or amendment.get("v7_app_id") != V7_APP_ID
        or amendment.get("v7_registration_sha256") != V7_REGISTRATION_SHA256
        or amendment.get("require_final_unlocked_present") is not False
        or amendment.get("require_final_collection_present") is not False
        or amendment.get("require_evals_directory_present") is not False
        or amendment.get("require_final_evaluation_labels") != []
        or amendment.get("require_sealed_comparison_present") is not False
        or amendment.get("require_final_outcomes_unopened") is not True
        or amendment.get("require_v7_app_stopped") is not True
        or amendment.get("require_modal_gpu_lease_resolved") is not True
    ):
        raise ProtocolError("V8-local prelaunch amendment is malformed or disabled")
    relative = amendment.get("v7_authoritative_closeout_path")
    expected_sha = amendment.get("v7_authoritative_closeout_sha256")
    if (
        relative != "protocol_archive/v7_profanity_authoritative_closeout.json"
        or expected_sha != V7_AUTHORITATIVE_CLOSEOUT_SHA256
    ):
        raise ProtocolError("V8-local amendment lacks the authoritative V7 closeout")
    path = copied_closeout or (REPO / relative)
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise ProtocolError("pinned V7 closeout is missing or changed")
    wrapper = read_json(path)
    if (
        wrapper.get("document_type") != V7_CLOSEOUT_PROTOCOL
        or wrapper.get("protocol") != "j-lens-rl-confirmatory-v7-profanity-u5"
        or wrapper.get("terminal_stage") != V7_TERMINAL_STAGE
        or wrapper.get("claim_id") != V7_CLAIM_ID
        or wrapper.get("app_id") != V7_APP_ID
        or wrapper.get("registration_sha256") != V7_REGISTRATION_SHA256
        or wrapper.get("final_unlocked_present") is not False
        or wrapper.get("final_collection_present") is not False
        or wrapper.get("evals_directory_present") is not False
        or wrapper.get("final_evaluation_labels") != []
        or wrapper.get("sealed_comparison_present") is not False
        or wrapper.get("final_outcomes_unopened") is not True
        or wrapper.get("modal_app_stopped") is not True
        or wrapper.get("global_gpu_lease_resolved") is not True
        or wrapper.get("sealed_final_manifest_sha256") != FINAL_SHA256
        or wrapper.get("pre_recovery_closeout")
        != {
            "commit": V7_CLOSEOUT_COMMIT,
            "path": "protocol_archive/v7_profanity_terminal_closeout.json",
            "sha256": V7_PRE_RECOVERY_CLOSEOUT_SHA256,
        }
        or wrapper.get("lease_retirement", {}).get("receipt_sha256")
        != "3caf91c90ff5dd54170ebaab658494d643fde4eb681b7fe18ddfc26b79e82dc3"
        or wrapper.get("lease_retirement", {}).get("dict_name") != V7_LEASE_DICT_NAME
        or wrapper.get("lease_retirement", {}).get("key") != V7_LEASE_KEY
        or wrapper.get("lease_retirement", {}).get("nonce") != V7_LEASE_NONCE
        or wrapper.get("lease_retirement", {}).get("key_absent_after_pop") is not True
        or wrapper.get("lease_retirement", {}).get("popped_full_value_sha256")
        != V7_LEASE_VALUE_SHA256
    ):
        raise ProtocolError("V7 closeout does not authoritatively preserve the final")
    evidence = wrapper.get("source_evidence_sha256")
    paths = wrapper.get("source_evidence_paths")
    if (
        not isinstance(evidence, dict)
        or set(evidence) != V7_REQUIRED_EVIDENCE_NAMES
        or set(paths or {}) != V7_REQUIRED_EVIDENCE_NAMES
        or any(re.fullmatch(r"[0-9a-f]{64}", value or "") is None for value in evidence.values())
    ):
        raise ProtocolError("V7 closeout lacks an exact source-evidence chain")
    for name, digest in evidence.items():
        relative_evidence = paths[name]
        if copied_closeout is None:
            path_allowed = (
                relative_evidence
                == "protocol_archive/v7_profanity_gpu_lease_retirement_receipt.json"
                if name == "gpu_lease_retirement_receipt"
                else isinstance(relative_evidence, str)
                and relative_evidence.startswith(
                    "protocol_archive/v7_profanity_terminal_evidence/"
                )
            )
            if (
                not path_allowed
                or Path(relative_evidence).is_absolute()
                or ".." in Path(relative_evidence).parts
                or not (REPO / relative_evidence).is_file()
                or sha256_file(REPO / relative_evidence) != digest
            ):
                raise ProtocolError(f"V7 closeout evidence changed: {name}")
        else:
            if copied_evidence_dir is None:
                raise ProtocolError("copied V7 closeout lacks frozen evidence directory")
            copied = copied_evidence_dir / Path(relative_evidence).name
            if not copied.is_file() or sha256_file(copied) != digest:
                raise ProtocolError(f"copied V7 closeout evidence changed: {name}")

    receipt_relative = amendment.get("v7_gpu_lease_retirement_receipt_path")
    receipt_sha = amendment.get("v7_gpu_lease_retirement_receipt_sha256")
    if (
        receipt_relative
        != "protocol_archive/v7_profanity_gpu_lease_retirement_receipt.json"
        or receipt_sha
        != "3caf91c90ff5dd54170ebaab658494d643fde4eb681b7fe18ddfc26b79e82dc3"
    ):
        raise ProtocolError("V8-local amendment does not pin the exact lease retirement")
    receipt_path = copied_lease_receipt or (REPO / receipt_relative)
    if not receipt_path.is_file() or sha256_file(receipt_path) != receipt_sha:
        raise ProtocolError("V7 GPU lease-retirement receipt is missing or changed")
    receipt = read_json(receipt_path)
    if (
        receipt.get("protocol") != V7_LEASE_RETIREMENT_PROTOCOL
        or receipt.get("dict_name") != V7_LEASE_DICT_NAME
        or receipt.get("key") != V7_LEASE_KEY
        or receipt.get("nonce") != V7_LEASE_NONCE
        or receipt.get("owner") != V7_LEASE_OWNER
        or receipt.get("observed_full_value_sha256") != V7_LEASE_VALUE_SHA256
        or receipt.get("popped_full_value_sha256") != V7_LEASE_VALUE_SHA256
        or receipt.get("popped_equal_to_observed") is not True
        or receipt.get("key_absent_after_pop") is not True
        or receipt.get("closeout")
        != {
            "commit": V7_CLOSEOUT_COMMIT,
            "path": "protocol_archive/v7_profanity_terminal_closeout.json",
            "sha256": V7_PRE_RECOVERY_CLOSEOUT_SHA256,
            "verified_present_on_origin_main": True,
        }
        or receipt.get("app_check", {}).get("app_id") != V7_APP_ID
        or receipt.get("app_check", {}).get("state") != "stopped"
        or receipt.get("app_check", {}).get("tasks") != 0
        or receipt.get("app_check", {}).get("containers_after_stop") != []
    ):
        raise ProtocolError("V7 lease-retirement receipt is malformed")

    pre_path = copied_pre_recovery_closeout or (
        REPO / "protocol_archive/v7_profanity_terminal_closeout.json"
    )
    if (
        not pre_path.is_file()
        or sha256_file(pre_path) != V7_PRE_RECOVERY_CLOSEOUT_SHA256
    ):
        raise ProtocolError("V7 pre-recovery closeout is missing or changed")
    pre = read_json(pre_path)
    boundary = pre.get("final_boundary", {})
    app_stop = pre.get("infrastructure_incident", {}).get("app_stop", {})
    lease = pre.get("infrastructure_incident", {}).get("stranded_gpu_lease", {})
    if (
        pre.get("protocol") != V7_CLOSEOUT_PROTOCOL
        or pre.get("terminal_stage") != V7_TERMINAL_STAGE
        or pre.get("v7_attempt_disposition") != "infrastructure_failed"
        or boundary.get("final_outcomes_unopened") is not True
        or any(
            boundary.get(key) is not False
            for key in (
                "acceptance_present",
                "analysis_directory_present",
                "controls_present",
                "evals_directory_present",
                "final_collection_present",
                "final_unlocked_present",
                "sealed_comparison_present",
            )
        )
        or boundary.get("final_evaluation_labels") != []
        or app_stop.get("app_id") != V7_APP_ID
        or app_stop.get("state") != "stopped"
        or app_stop.get("tasks") != 0
        or app_stop.get("post_stop_container_count") != 0
        or lease.get("canonical_full_value_sha256") != V7_LEASE_VALUE_SHA256
        or lease.get("nonce") != V7_LEASE_NONCE
        or lease.get("owner") != V7_LEASE_OWNER
    ):
        raise ProtocolError("V7 pre-recovery closeout changed its final boundary")
    return path


def _amendment() -> tuple[dict[str, Any], Path | None]:
    if AMENDMENT_SOURCE.is_file():
        value = read_json(AMENDMENT_SOURCE)
        return value, _validate_closeout(value)
    value = read_json(AMENDMENT_TEMPLATE_SOURCE)
    if value.get("launch_enabled") is not False:
        raise ProtocolError("unbound V8-local amendment template enabled launch")
    return value, None


def _code_identity() -> dict[str, Any]:
    files = {name: sha256_file(REPO / name) for name in TRAIN_CODE_PATHS}
    raw_index = subprocess.check_output(
        ["git", "ls-files", "-s", "-z", "--", "trl"], cwd=REPO
    )
    trl_files: dict[str, dict[str, str]] = {}
    for raw_record in raw_index.split(b"\0"):
        if not raw_record:
            continue
        metadata, raw_name = raw_record.split(b"\t", 1)
        mode, blob, stage = metadata.decode().split()
        if stage != "0":
            raise ProtocolError("TRL source has a non-stage-zero Git index entry")
        trl_files[raw_name.decode()] = {"git_mode": mode, "git_blob": blob}
    value = {
        "files": files,
        "files_canonical_sha256": canonical_sha256(files),
        "trl_tree_sha256": canonical_sha256(trl_files),
        "trl_file_count": len(trl_files),
    }
    value["combined_sha256"] = canonical_sha256(value)
    return value


def _base_training_config(recipe: dict[str, Any]) -> dict[str, Any]:
    training = recipe["training"]
    return {
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "dataset_revision": DATASET_REVISION,
        "lens_path": ".confirmatory/v8_local/frozen_artifacts/lens.pt",
        "lens_sha256": LENS_SHA256,
        "expected_lens_sha256": LENS_SHA256,
        "calibration_path": ".confirmatory/v8_local/frozen_artifacts/calibration.json",
        "calibration_sha256": CALIBRATION_SHA256,
        "expected_calibration_sha256": CALIBRATION_SHA256,
        "target_words": ["damn", "fuck"],
        "train_examples": training["train_examples"],
        "validation_examples": 400,
        "validation_batch_size": 64,
        "num_generations": training["num_generations"],
        "num_generations_eval": 1,
        "max_prompt_tokens": 384,
        "max_new_tokens": training["max_new_tokens"],
        "min_new_tokens": training["min_new_tokens"],
        "temperature": 1.0,
        "updates": training["updates"],
        "learning_rate": training["learning_rate"],
        "lr_scheduler_type": training["lr_scheduler_type"],
        "warmup_steps": training["warmup_steps"],
        "warmup_ratio": training["warmup_ratio"],
        "kl_beta": training["kl_beta"],
        "loss_type": training["loss_type"],
        "scale_rewards": training["scale_rewards"],
        "gradient_accumulation_steps": training["gradient_accumulation_steps"],
        "lora_rank": training["lora_rank"],
        "lora_alpha": training["lora_alpha"],
        "score_stride": 5,
        "score_start_fraction": 0.0,
        "score_layers": [8, 14, 20],
        "score_aggregation": "mean",
        "score_include_final": False,
        "vocab_chunk_size": 16384,
        "mask_target_tokens": True,
        "eval_every": 2,
        "validation_steps": [4, 10, 20],
        "validation_source": "train",
        "validation_indices_path": ".confirmatory/v8_local/manifests/curve_indices.json",
        "reserved_train_indices_path": ".confirmatory/v8_local/manifests/train_exclusions.json",
        "validation_observational_only": True,
        "early_stopping_patience": None,
        "early_stopping_min_delta": 0.0,
        "eval_strategy": "no",
        "save_every": training["save_every"],
        "save_total_limit": training["save_total_limit"],
        "reward_type": "jlens",
        "require_clean_repository": True,
        "wandb_entity": WANDB_ENTITY,
        "wandb_project": WANDB_PROJECT,
        "wandb_group": WANDB_GROUP,
        "wandb_mode": "offline",
        "wandb_resume": "never",
        "wandb_tags": [
            "confirmatory-v8-local",
            "emotional-j-lens",
            "profanity-u5",
            "rtx4090",
            "offline-wandb",
            "curve-exposed-final-sealed",
        ],
        "curve_manifest_sha256": CURVE_SHA256,
        "train_exclusions_manifest_sha256": TRAIN_EXCLUSIONS_SHA256,
        "registered_backend": "local-rtx4090",
        "expected_cuda_device_name": GPU_NAME,
        "evidence_eligibility": "original_registered_v8_local_attempt",
    }


def _materialize_configs(
    recipe: dict[str, Any],
    registration_sha: str,
    recipe_sha: str,
    code: dict[str, Any],
    metric_schema_sha: str,
) -> dict[str, dict[str, Any]]:
    result = {}
    base = _base_training_config(recipe)
    for condition in CONDITIONS:
        for seed in SEEDS:
            label = f"{condition}_seed{seed}"
            config = dict(base)
            config.update(
                {
                    "seed": seed,
                    "score_components": recipe["reward"][
                        "treatment_score_components"
                        if condition == "jlens"
                        else "matched_control_score_components"
                    ],
                    "output_dir": f".confirmatory/v8_local/runs/{label}",
                    "run_name": f"{WANDB_PREFIX}-{label}",
                    "wandb_run_id": WANDB_RUN_IDS[label],
                    "wandb_url": (
                        f"https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}/runs/"
                        f"{WANDB_RUN_IDS[label]}"
                    ),
                    "registration_sha256": registration_sha,
                    "recipe_lock_sha256": recipe_sha,
                    "recipe_sha256": canonical_sha256(recipe),
                    "registered_code_sha256": code["combined_sha256"],
                    "metric_schema_path": (
                        ".confirmatory/v8_local/reproducibility/metric_schema.json"
                    ),
                    "metric_schema_sha256": metric_schema_sha,
                    "registered_command": [
                        str(PYTHON_EXECUTABLE),
                        "scripts/v8_local_train.py",
                        "--config",
                        f".confirmatory/v8_local/configs/{label}.json",
                        "--wandb-mode",
                        "offline",
                    ],
                }
            )
            result[label] = config
    first = result[f"jlens_seed{SEEDS[0]}"]
    sealed = dict(first)
    sealed.update(
        {
            "seed": SEEDS[0],
            "validation_examples": 900,
            "evaluation_source": "train",
            "evaluation_indices_path": (
                ".confirmatory/v8_local/manifests/sealed_final_indices.json"
            ),
            "evaluation_seed": 0,
            "min_new_tokens": 0,
            "output_dir": ".confirmatory/v8_local/evaluation_config_unused",
            "run_name": f"{WANDB_PREFIX}-sealed-evaluation",
        }
    )
    for key in (
        "wandb_run_id",
        "wandb_url",
        "wandb_resume",
        "wandb_tags",
        "registered_command",
    ):
        sealed.pop(key, None)
    result["sealed_eval"] = sealed
    return result


def _config_path(condition: str, seed: int) -> Path:
    if condition not in CONDITIONS or seed not in SEEDS:
        raise ProtocolError(f"unregistered V8-local run: {condition}/{seed}")
    return CONFIG_DIR / f"{condition}_seed{seed}.json"


def _run_dir(condition: str, seed: int) -> Path:
    return RUN_DIR / f"{condition}_seed{seed}"


def _manifest_payload(indices: Sequence[int]) -> dict[str, Any]:
    return {
        "dataset": "openai/gsm8k",
        "subset": "main",
        "split": "train",
        "indices": list(indices),
    }


def _metric_schema(recipe: dict[str, Any]) -> dict[str, Any]:
    from scripts.confirmatory_v7_protocol import metric_schema

    return metric_schema(
        recipe["reward"]["target_words"],
        int(recipe["training"]["updates"]),
        recipe["reward"]["treatment_score_components"],
    )


def _launch_plan(configs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    training = {
        label: config["registered_command"]
        for label, config in configs.items()
        if label != "sealed_eval"
    }
    evaluations: dict[str, list[str]] = {}
    for label in FINAL_LABELS:
        if label == "base":
            experiment = f".confirmatory/v8_local/configs/jlens_seed{SEEDS[0]}.json"
            adapter: list[str] = []
        else:
            experiment = f".confirmatory/v8_local/configs/{label}.json"
            adapter = ["--adapter", f".confirmatory/v8_local/runs/{label}/final"]
        evaluations[label] = [
            str(PYTHON_EXECUTABLE),
            "-m",
            "jlens_rl.eval",
            "--config",
            ".confirmatory/v8_local/configs/sealed_eval.json",
            "--experiment-config",
            experiment,
            "--indices-manifest",
            ".confirmatory/v8_local/manifests/sealed_final_indices.json",
            "--output-jsonl",
            f".confirmatory/v8_local/evals/{label}.jsonl",
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
            "eight treatments in ascending seed order",
            "record exposed-development curve gate exactly once",
            "eight sign-flip controls in ascending seed order only if gate passes",
            "one immutable 17-label sealed-final collection only after all runs verify",
            "registered paired crossed-bootstrap and sign-test analysis",
        ],
        "training_commands": training,
        "final_evaluation_commands": evaluations,
        "analysis_command": [
            str(PYTHON_EXECUTABLE),
            "-m",
            "jlens_rl.paired_eval",
            "--base-jsonl",
            ".confirmatory/v8_local/evals/base.jsonl",
            *[
                value
                for seed in SEEDS
                for value in (
                    "--adapter-jsonl",
                    f".confirmatory/v8_local/evals/jlens_seed{seed}.jsonl",
                )
            ],
            *[
                value
                for seed in SEEDS
                for value in (
                    "--control-jsonl",
                    f".confirmatory/v8_local/evals/signflip_seed{seed}.jsonl",
                )
            ],
            "--bootstrap-samples",
            "10000",
            "--seed",
            "0",
            "--confidence",
            "0.95",
            "--output",
            ".confirmatory/v8_local/evidence/sealed_comparison.json",
        ],
        "wandb": {
            "entity": WANDB_ENTITY,
            "project": WANDB_PROJECT,
            "group": WANDB_GROUP,
            "mode": "offline",
            "run_ids": WANDB_RUN_IDS,
        },
    }


def _tracked_source_inventory(commit: str) -> dict[str, Any]:
    names = _git("ls-tree", "-r", "--name-only", commit).splitlines()
    files = {
        name: {"sha256": sha256_file(REPO / name), "size": (REPO / name).stat().st_size}
        for name in names
        if (REPO / name).is_file()
    }
    return {
        "schema_version": 1,
        "git_commit": commit,
        "files": files,
        "tree_sha256": canonical_sha256(files),
    }


def _write_source_snapshot(path: Path, inventory: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(inventory["files"]):
            info = zipfile.ZipInfo(name)
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, (REPO / name).read_bytes())
    temporary.replace(path)


def _verify_source_snapshot(inventory: dict[str, Any]) -> None:
    path = REPRO_DIR / "source_snapshot.zip"
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != sorted(inventory["files"]):
            raise ProtocolError("V8-local source snapshot file list changed")
        for name, identity in inventory["files"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != identity["sha256"]:
                raise ProtocolError(f"V8-local source snapshot changed: {name}")


def _registration(
    *,
    draft: dict[str, Any],
    amendment: dict[str, Any],
    recipe_sha: str,
    code: dict[str, Any],
    source_inputs: dict[str, str],
    commit: str,
) -> dict[str, Any]:
    if (
        draft.get("document_type")
        != "j-lens-rl-confirmatory-v8-local-profanity-registration-draft"
        or draft.get("protocol") != PROTOCOL
        or draft.get("launch_gate", {}).get("enabled") is not False
        or draft.get("scientific_identity", {}).get("seeds") != list(SEEDS)
        or draft.get("backend_identity", {}).get("wandb_mode") != "offline"
        or draft.get("backend_identity", {}).get("required_gpu") != GPU_NAME
        or draft.get("backend_identity", {}).get("required_gpu_uuid") != GPU_UUID
        or draft.get("backend_identity", {}).get("required_driver_version") != GPU_DRIVER
        or draft.get("backend_identity", {}).get("required_memory_total_mib")
        != GPU_MEMORY_MIB
        or draft.get("backend_identity", {}).get("required_torch_version")
        != TORCH_VERSION
        or draft.get("backend_identity", {}).get("required_cuda_build") != CUDA_BUILD
        or draft.get("backend_identity", {}).get("python_executable")
        != str(PYTHON_EXECUTABLE)
        or draft.get("fixed_input_sha256", {}).get("v7_authoritative_closeout")
        != V7_AUTHORITATIVE_CLOSEOUT_SHA256
        or draft.get("fixed_input_sha256", {}).get("v7_pre_recovery_closeout")
        != V7_PRE_RECOVERY_CLOSEOUT_SHA256
        or draft.get("fixed_input_sha256", {}).get(
            "v7_gpu_lease_retirement_receipt"
        )
        != "3caf91c90ff5dd54170ebaab658494d643fde4eb681b7fe18ddfc26b79e82dc3"
    ):
        raise ProtocolError("V8-local committed registration draft changed")
    return {
        "schema_version": 1,
        "document_type": REGISTRATION_PROTOCOL,
        "protocol": PROTOCOL,
        "status": "prospectively_registered_launch_enabled_by_committed_v7_closeout",
        "git_commit": commit,
        "git_tree": _git("rev-parse", f"{commit}^{{tree}}"),
        "draft_sha256": sha256_file(REGISTRATION_DRAFT_SOURCE),
        "amendment_sha256": sha256_file(AMENDMENT_SOURCE),
        "v7_authoritative_closeout_sha256": amendment[
            "v7_authoritative_closeout_sha256"
        ],
        "recipe_lock_sha256": recipe_sha,
        "code_identity": code,
        "source_input_sha256": source_inputs,
        "seeds": list(SEEDS),
        "curve_steps": list(CURVE_STEPS),
        "curve_exposure_caveat": (
            "The exact 400-row curve and profanity lineage are exposed development "
            "data. The curve is an operational consistency gate, never inferential "
            "evidence; only the conditionally unopened 900-row final is inferential."
        ),
        "backend": {
            "kind": "one_local_gpu_serial",
            "gpu_name": GPU_NAME,
            "gpu_uuid": GPU_UUID,
            "driver": GPU_DRIVER,
            "memory_total_mib": GPU_MEMORY_MIB,
            "torch": TORCH_VERSION,
            "cuda_build": CUDA_BUILD,
            "wandb_mode": "offline",
            "runtime_source": "dedicated detached clean worktree at git_commit",
        },
        "phase_order": ["treatments", "curve_gate", "controls", "sealed_final"],
        "retry_policy": INFRASTRUCTURE_RETRY_POLICY,
        "analysis": ANALYSIS_REGISTRATION,
        "acceptance": ACCEPTANCE_REGISTRATION,
    }


def _create_runtime_worktree(commit: str) -> None:
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(RUNTIME_WORKTREE), commit],
        cwd=REPO,
        check=True,
    )
    link = RUNTIME_WORKTREE / ".confirmatory" / "v8_local"
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(STATE_DIR, target_is_directory=True)
    if _runtime_git("status", "--porcelain=v1", "--untracked-files=all"):
        raise ProtocolError("dedicated V8-local runtime worktree is not clean")


def _runtime_git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=RUNTIME_WORKTREE, text=True
    ).strip()


def _prepared_inventory() -> dict[str, str]:
    roots = (CONFIG_DIR, ARTIFACT_DIR, MANIFEST_DIR, REPRO_DIR)
    files: dict[str, str] = {}
    for root in roots:
        for path in sorted(root.rglob("*")):
            if path.is_file() and not path.is_symlink():
                files[path.relative_to(STATE_DIR).as_posix()] = sha256_file(path)
    return files


def prepare() -> dict[str, Any]:
    provenance = _require_clean_pushed_main()
    if STATE_DIR.exists():
        raise ProtocolError(f"{STATE_DIR} already exists; V8-local preparation is immutable")
    recipe = _validate_recipe(read_json(RECIPE_SOURCE))
    amendment, closeout_path = _amendment()
    if closeout_path is None:
        raise ProtocolError(
            "V8-local launch is disabled until a committed authoritative V7 closeout exists"
        )
    source_inputs = _source_inputs()
    code = _code_identity()
    recipe_sha = sha256_file(RECIPE_SOURCE)
    draft = read_json(REGISTRATION_DRAFT_SOURCE)
    registration = _registration(
        draft=draft,
        amendment=amendment,
        recipe_sha=recipe_sha,
        code=code,
        source_inputs=source_inputs,
        commit=provenance["git_commit"],
    )
    registration_sha = canonical_sha256(registration)

    for directory in (CONFIG_DIR, ARTIFACT_DIR, MANIFEST_DIR, REPRO_DIR):
        directory.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(LENS_SOURCE, ARTIFACT_DIR / "lens.pt")
    shutil.copyfile(CALIBRATION_SOURCE, ARTIFACT_DIR / "calibration.json")
    curve = _load_indices(CURVE_SOURCE, 400)
    exclusions_payload = read_json(TRAIN_EXCLUSIONS_SOURCE)
    exclusions = _load_indices(TRAIN_EXCLUSIONS_SOURCE, len(exclusions_payload["indices"]))
    final = _load_indices(FINAL_SOURCE, 900)
    write_json(MANIFEST_DIR / "curve_indices.json", _manifest_payload(curve))
    write_json(MANIFEST_DIR / "train_exclusions.json", _manifest_payload(exclusions))
    write_json(MANIFEST_DIR / "sealed_final_indices.json", _manifest_payload(final))
    if (
        sha256_file(MANIFEST_DIR / "curve_indices.json") != CURVE_SHA256
        or sha256_file(MANIFEST_DIR / "train_exclusions.json")
        != TRAIN_EXCLUSIONS_SHA256
        or sha256_file(MANIFEST_DIR / "sealed_final_indices.json") != FINAL_SHA256
    ):
        raise ProtocolError("V8-local materialized manifests differ byte-for-byte")

    schema = _metric_schema(recipe)
    write_json(REPRO_DIR / "metric_schema.json", schema)
    metric_schema_sha = sha256_file(REPRO_DIR / "metric_schema.json")
    configs = _materialize_configs(
        recipe, registration_sha, recipe_sha, code, metric_schema_sha
    )
    for label, config in configs.items():
        write_json(CONFIG_DIR / f"{label}.json", config)
    write_json(REPRO_DIR / "registration.json", registration)
    shutil.copyfile(RECIPE_SOURCE, REPRO_DIR / "recipe_lock.json")
    shutil.copyfile(REGISTRATION_DRAFT_SOURCE, REPRO_DIR / "registration_draft.json")
    shutil.copyfile(AMENDMENT_SOURCE, REPRO_DIR / "prelaunch_amendment.json")
    shutil.copyfile(closeout_path, REPRO_DIR / "v7_authoritative_closeout.json")
    shutil.copyfile(
        REPO / "protocol_archive/v7_profanity_terminal_closeout.json",
        REPRO_DIR / "v7_pre_recovery_closeout.json",
    )
    shutil.copyfile(
        REPO / "protocol_archive/v7_profanity_gpu_lease_retirement_receipt.json",
        REPRO_DIR / "v7_gpu_lease_retirement_receipt.json",
    )
    copied_evidence = REPRO_DIR / "v7_terminal_evidence"
    copied_evidence.mkdir()
    for relative in read_json(closeout_path)["source_evidence_paths"].values():
        source = REPO / relative
        destination = copied_evidence / source.name
        if destination.exists():
            if sha256_file(destination) != sha256_file(source):
                raise ProtocolError("V7 closeout evidence basename collision")
        else:
            shutil.copyfile(source, destination)
    write_json(REPRO_DIR / "launch_plan.json", _launch_plan(configs))
    inventory = _tracked_source_inventory(provenance["git_commit"])
    write_json(REPRO_DIR / "source_manifest.json", inventory)
    _write_source_snapshot(REPRO_DIR / "source_snapshot.zip", inventory)
    prepared = _prepared_inventory()
    state = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "registration_sha256": registration_sha,
        "recipe_lock_sha256": recipe_sha,
        "git_commit": provenance["git_commit"],
        "git_tree": provenance["git_tree"],
        "target_words": ["damn", "fuck"],
        "seeds": list(SEEDS),
        "conditions": list(CONDITIONS),
        "curve_gate_steps": list(CURVE_STEPS),
        "final_labels": list(FINAL_LABELS),
        "artifact_sha256": {"lens": LENS_SHA256, "calibration": CALIBRATION_SHA256},
        "prepared_file_sha256": prepared,
        "runtime_worktree": str(RUNTIME_WORKTREE),
        "launch_enabled": True,
        "prepared_at_utc": utc_now(),
    }
    write_json(STATE_PATH, state, exclusive=True)
    _create_runtime_worktree(provenance["git_commit"])
    return load_and_verify_state(require_launch=True)


def _verify_runtime_worktree(state: dict[str, Any]) -> None:
    if not RUNTIME_WORKTREE.is_dir():
        raise ProtocolError("dedicated V8-local runtime worktree is missing")
    if _runtime_git("rev-parse", "HEAD") != state["git_commit"]:
        raise ProtocolError("V8-local runtime worktree moved from registered commit")
    if _runtime_git("status", "--porcelain=v1", "--untracked-files=all"):
        raise ProtocolError("V8-local runtime worktree became dirty")
    link = RUNTIME_WORKTREE / ".confirmatory" / "v8_local"
    if not link.is_symlink() or link.resolve() != STATE_DIR.resolve():
        raise ProtocolError("V8-local runtime state link changed")
    code = read_json(REPRO_DIR / "registration.json")["code_identity"]
    for name, digest in code["files"].items():
        if sha256_file(RUNTIME_WORKTREE / name) != digest:
            raise ProtocolError(f"runtime source changed: {name}")


def load_and_verify_state(*, require_launch: bool = False) -> dict[str, Any]:
    if not STATE_PATH.is_file():
        raise ProtocolError("V8-local state has not been prepared")
    state = read_json(STATE_PATH)
    if (
        state.get("protocol") != PROTOCOL
        or state.get("seeds") != list(SEEDS)
        or state.get("conditions") != list(CONDITIONS)
        or state.get("curve_gate_steps") != list(CURVE_STEPS)
        or state.get("final_labels") != list(FINAL_LABELS)
        or state.get("launch_enabled") is not True
    ):
        raise ProtocolError("V8-local protocol state changed")
    registration = read_json(REPRO_DIR / "registration.json")
    if canonical_sha256(registration) != state.get("registration_sha256"):
        raise ProtocolError("V8-local registration changed")
    if (
        registration.get("git_commit") != state.get("git_commit")
        or registration.get("git_tree") != state.get("git_tree")
        or read_json(REPRO_DIR / "recipe_lock.json") != _validate_recipe(
            read_json(REPRO_DIR / "recipe_lock.json")
        )
        or sha256_file(REPRO_DIR / "recipe_lock.json")
        != state.get("recipe_lock_sha256")
    ):
        raise ProtocolError("V8-local registered recipe/source identity changed")
    expected_prepared = state.get("prepared_file_sha256")
    current_prepared = _prepared_inventory()
    # State was written after this inventory; nothing under the four frozen roots may
    # be added, removed, or changed after preparation.
    if current_prepared != expected_prepared:
        raise ProtocolError("V8-local prepared-file inventory changed")
    if (
        sha256_file(ARTIFACT_DIR / "lens.pt") != LENS_SHA256
        or sha256_file(ARTIFACT_DIR / "calibration.json") != CALIBRATION_SHA256
        or sha256_file(MANIFEST_DIR / "curve_indices.json") != CURVE_SHA256
        or sha256_file(MANIFEST_DIR / "train_exclusions.json")
        != TRAIN_EXCLUSIONS_SHA256
        or sha256_file(MANIFEST_DIR / "sealed_final_indices.json") != FINAL_SHA256
    ):
        raise ProtocolError("V8-local frozen data/artifact bytes changed")
    recipe = read_json(REPRO_DIR / "recipe_lock.json")
    schema = _metric_schema(recipe)
    if read_json(REPRO_DIR / "metric_schema.json") != schema:
        raise ProtocolError("V8-local metric schema changed")
    configs = _materialize_configs(
        recipe,
        state["registration_sha256"],
        state["recipe_lock_sha256"],
        registration["code_identity"],
        sha256_file(REPRO_DIR / "metric_schema.json"),
    )
    for label, expected in configs.items():
        if read_json(CONFIG_DIR / f"{label}.json") != expected:
            raise ProtocolError(f"generated V8-local config changed: {label}")
    if read_json(REPRO_DIR / "launch_plan.json") != _launch_plan(configs):
        raise ProtocolError("V8-local launch plan changed")
    source_inventory = read_json(REPRO_DIR / "source_manifest.json")
    _verify_source_snapshot(source_inventory)
    _verify_runtime_worktree(state)
    if require_launch:
        amendment = read_json(REPRO_DIR / "prelaunch_amendment.json")
        _validate_closeout(
            amendment,
            REPRO_DIR / "v7_authoritative_closeout.json",
            REPRO_DIR / "v7_terminal_evidence",
            REPRO_DIR / "v7_gpu_lease_retirement_receipt.json",
            REPRO_DIR / "v7_pre_recovery_closeout.json",
        )
    return state


def probe_hardware(*, require_idle: bool = True) -> dict[str, Any]:
    query = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()
    rows = [[field.strip() for field in line.split(",")] for line in query if line.strip()]
    if rows != [["0", GPU_UUID, GPU_NAME, GPU_DRIVER, str(GPU_MEMORY_MIB)]]:
        raise ProtocolError(f"exact registered one-GPU host contract failed: {rows!r}")
    compute = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()
    processes = [line.strip() for line in compute if line.strip()]
    if require_idle and processes:
        raise ProtocolError(f"registered RTX 4090 has active compute processes: {processes}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPU_UUID
    torch_probe = subprocess.check_output(
        [
            str(PYTHON_EXECUTABLE),
            "-c",
            (
                "import json,torch; print(json.dumps({"
                "'count':torch.cuda.device_count(),"
                "'name':torch.cuda.get_device_name(0) if torch.cuda.device_count() else None,"
                "'torch':torch.__version__, 'cuda':torch.version.cuda,"
                "'bf16':torch.cuda.is_bf16_supported() if torch.cuda.device_count() else False}))"
            ),
        ],
        env=env,
        text=True,
    )
    observed_torch = json.loads(torch_probe)
    expected_torch = {
        "count": 1,
        "name": GPU_NAME,
        "torch": TORCH_VERSION,
        "cuda": CUDA_BUILD,
        "bf16": True,
    }
    if observed_torch != expected_torch:
        raise ProtocolError(
            f"registered local torch/CUDA contract failed: {observed_torch!r}"
        )
    boot_id_path = Path("/proc/sys/kernel/random/boot_id")
    machine_id_path = Path("/etc/machine-id")
    return {
        "schema_version": 1,
        "gpu": {
            "index": 0,
            "uuid": GPU_UUID,
            "name": GPU_NAME,
            "driver": GPU_DRIVER,
            "memory_total_mib": GPU_MEMORY_MIB,
        },
        "torch": observed_torch,
        "foreign_compute_processes": processes,
        "boot_id_sha256": (
            hashlib.sha256(boot_id_path.read_bytes()).hexdigest()
            if boot_id_path.is_file()
            else None
        ),
        "machine_id_sha256": (
            hashlib.sha256(machine_id_path.read_bytes()).hexdigest()
            if machine_id_path.is_file()
            else None
        ),
    }


def _expected_validation_steps(config: dict[str, Any]) -> tuple[int, ...]:
    return tuple(sorted({0, *(int(step) for step in config["validation_steps"])}))


def _load_history(path: Path, expected_steps: Sequence[int]) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    if not path.is_file():
        raise ProtocolError(f"missing validation history: {path}")
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        step, score = row.get("step"), row.get("exact_match")
        if isinstance(step, bool) or not isinstance(step, int) or step in rows:
            raise ProtocolError(f"invalid validation step at {path}:{line_number}")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not 0 <= float(score) <= 1
        ):
            raise ProtocolError(f"invalid exact match at {path}:{line_number}")
        rows[step] = row
    if tuple(sorted(rows)) != tuple(sorted(expected_steps)):
        raise ProtocolError(f"wrong validation steps in {path}: {sorted(rows)}")
    return rows


def _expected_wandb_identity(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "entity": config["wandb_entity"],
        "project": config["wandb_project"],
        "run_name": config["run_name"],
        "run_id": config["wandb_run_id"],
        "url": config["wandb_url"],
        "group": config["wandb_group"],
        "tags": config["wandb_tags"],
        "resume": config["wandb_resume"],
    }


def _verify_process_command(process: Any, config_path: Path) -> None:
    if not isinstance(process, dict):
        raise ProtocolError("training run lacks process command")
    argv, cwd = process.get("argv"), process.get("cwd")
    if (
        not isinstance(argv, list)
        or len(argv) != 5
        or argv[0] != "scripts/v8_local_train.py"
        or argv[1] != "--config"
        or argv[3:] != ["--wandb-mode", "offline"]
        or process.get("python_executable") != str(PYTHON_EXECUTABLE)
        or Path(str(cwd)).resolve() != RUNTIME_WORKTREE.resolve()
    ):
        raise ProtocolError(f"training command changed: {process!r}")
    supplied = Path(argv[2])
    if not supplied.is_absolute():
        supplied = Path(cwd) / supplied
    expected = RUNTIME_WORKTREE / config_path.relative_to(REPO)
    if supplied.resolve() != expected.resolve():
        raise ProtocolError("training command used the wrong config")


def _expected_run_result(
    directory: Path, config: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    required = {
        name: directory / name
        for name in (
            "run_manifest.json",
            "resolved_config.json",
            "data_indices.json",
            "validation_history.jsonl",
            "log_history.json",
            "environment_snapshot.json",
        )
    }
    return {
        "schema_version": 1,
        "completed_updates": config["updates"],
        "wandb_identity": manifest.get("wandb_identity"),
        "metric_schema": manifest.get("metric_schema"),
        "process_command": manifest["process_command"],
        "registered_command": config["registered_command"],
        "registration_sha256": config["registration_sha256"],
        "recipe_lock_sha256": config["recipe_lock_sha256"],
        "recipe_sha256": config["recipe_sha256"],
        "evidence_eligibility": config["evidence_eligibility"],
        "reproduction_source": None,
        "source": {
            key: manifest.get(key)
            for key in ("git_commit", "git_dirty", "source_tree_sha256")
        },
        "runtime": manifest["runtime"],
        "data_indices_sha256": manifest["data_indices_sha256"],
        "lens_sha256": manifest["lens_sha256"],
        "calibration_sha256": manifest["calibration_sha256"],
        "raw_history_sha256": {name: sha256_file(path) for name, path in required.items()},
        "terminal_checkpoint": _tree_identity(directory / f"checkpoint-{config['updates']}"),
        "final_adapter_and_tokenizer": _tree_identity(directory / "final"),
    }


def _verify_dispatch_binding(label: str, result_sha: str, receipt_sha: str) -> None:
    completions = list(DISPATCH_DIR.glob(f"*-{label}.completion.json"))
    intents = list(DISPATCH_DIR.glob(f"*-{label}.intent.json"))
    if len(completions) != 1 or len(intents) != 1:
        raise ProtocolError(f"{label} lacks one exact dispatch intent/completion")
    intent, completion = read_json(intents[0]), read_json(completions[0])
    match = re.fullmatch(r"(jlens|signflip)_seed(\d+)", label)
    if match is None:
        raise ProtocolError(f"invalid dispatch label: {label}")
    condition, seed = match.group(1), int(match.group(2))
    config = _config_path(condition, seed)
    state = read_json(STATE_PATH)
    expected_command = [
        str(PYTHON_EXECUTABLE),
        "scripts/v8_local_train.py",
        "--config",
        f".confirmatory/v8_local/configs/{label}.json",
        "--wandb-mode",
        "offline",
    ]
    if (
        completion.get("intent_sha256") != sha256_file(intents[0])
        or completion.get("run_result_sha256") != result_sha
        or completion.get("offline_receipt_sha256") != receipt_sha
        or completion.get("label") != label
        or completion.get("hardware") != intent.get("hardware")
        or intent.get("hardware", {}).get("gpu", {}).get("uuid") != GPU_UUID
        or intent.get("hardware", {}).get("gpu", {}).get("name") != GPU_NAME
        or intent.get("hardware", {}).get("gpu", {}).get("driver") != GPU_DRIVER
        or intent.get("protocol") != PROTOCOL
        or intent.get("label") != label
        or intent.get("condition") != condition
        or intent.get("seed") != seed
        or intent.get("command") != expected_command
        or intent.get("cwd") != str(RUNTIME_WORKTREE.resolve())
        or intent.get("config_sha256") != sha256_file(config)
        or intent.get("registration_sha256") != state["registration_sha256"]
        or intent.get("runtime_git_commit") != state["git_commit"]
        or intent.get("offline_wandb_dir")
        != str((OFFLINE_WANDB_DIR / label).resolve())
    ):
        raise ProtocolError(f"{label} dispatch provenance changed")


def validate_training_run(
    condition: str, seed: int, *, require_dispatch_completion: bool = True
) -> dict[str, Any]:
    state = load_and_verify_state(require_launch=True)
    config_path = _config_path(condition, seed)
    config = read_json(config_path)
    label = f"{condition}_seed{seed}"
    directory = _run_dir(condition, seed)
    required = [
        directory / name
        for name in (
            "resolved_config.json",
            "run_manifest.json",
            "data_indices.json",
            "validation_history.jsonl",
            "log_history.json",
            "environment_snapshot.json",
            "run_result_manifest.json",
            "wandb_offline_terminal_receipt.json",
        )
    ]
    if any(not path.is_file() for path in required):
        raise ProtocolError(f"{label} is not a complete terminal run")
    if read_json(directory / "resolved_config.json") != config:
        raise ProtocolError(f"{label} resolved config changed")
    manifest = read_json(directory / "run_manifest.json")
    if (
        manifest.get("git_commit") != state["git_commit"]
        or manifest.get("git_dirty") is not False
        or manifest.get("config_sha256") != sha256_file(config_path)
        or manifest.get("lens_sha256") != LENS_SHA256
        or manifest.get("calibration_sha256") != CALIBRATION_SHA256
        or manifest.get("wandb_identity") != _expected_wandb_identity(config)
        or manifest.get("reward_type") != "jlens"
        or manifest.get("confirmatory_identity", {}).get("registered_code_sha256")
        != config["registered_code_sha256"]
    ):
        raise ProtocolError(f"{label} source/config/reward identity changed")
    runtime = manifest.get("runtime", {})
    environment = read_json(directory / "environment_snapshot.json")
    if (
        runtime.get("cuda_device_name") != GPU_NAME
        or runtime.get("torch_version") != TORCH_VERSION
        or runtime.get("cuda_version") != CUDA_BUILD
        or runtime.get("environment_snapshot_sha256")
        != sha256_file(directory / "environment_snapshot.json")
        or environment.get("torch", {}).get("version") != TORCH_VERSION
        or environment.get("torch", {}).get("cuda_build") != CUDA_BUILD
        or environment.get("cuda_device_names") != [GPU_NAME]
        or environment.get("nvidia_smi_name_and_driver")
        != [f"{GPU_NAME}, {GPU_DRIVER}"]
        or environment.get("image_identity", {}).get("jlens_modal_image_spec")
        != LOCAL_RUNTIME_ID
    ):
        raise ProtocolError(f"{label} used the wrong local hardware/runtime")
    _verify_process_command(manifest.get("process_command"), config_path)
    data = read_json(directory / "data_indices.json")
    exclusions = set(_load_indices(MANIFEST_DIR / "train_exclusions.json", len(read_json(MANIFEST_DIR / "train_exclusions.json")["indices"])))
    curve = _load_indices(MANIFEST_DIR / "curve_indices.json", 400)
    train_indices = data.get("train_source_indices")
    if (
        not isinstance(train_indices, list)
        or len(train_indices) != 1000
        or len(set(train_indices)) != 1000
        or set(train_indices) & exclusions
        or data.get("validation_source") != "train"
        or data.get("validation_source_indices") != curve
        or manifest.get("data_indices_sha256") != sha256_file(directory / "data_indices.json")
    ):
        raise ProtocolError(f"{label} data firewall changed")
    history = _load_history(
        directory / "validation_history.jsonl", _expected_validation_steps(config)
    )
    if any(
        row.get("validation_source") != "train"
        or row.get("validation_indices_sha256") != CURVE_SHA256
        for row in history.values()
    ):
        raise ProtocolError(f"{label} validation provenance changed")
    from scripts.confirmatory_v7_protocol import training_behavior_summary

    training_behavior_summary(directory / "log_history.json", config)
    checkpoint = directory / f"checkpoint-{config['updates']}"
    trainer_state = checkpoint / "trainer_state.json"
    if (
        not trainer_state.is_file()
        or read_json(trainer_state).get("global_step") != config["updates"]
        or not (checkpoint / "adapter_model.safetensors").is_file()
        or not (directory / "final" / "adapter_model.safetensors").is_file()
        or sha256_file(checkpoint / "adapter_model.safetensors")
        != sha256_file(directory / "final" / "adapter_model.safetensors")
    ):
        raise ProtocolError(f"{label} is not the exact terminal checkpoint")
    result_path = directory / "run_result_manifest.json"
    result = read_json(result_path)
    if result != _expected_run_result(directory, config, manifest):
        raise ProtocolError(f"{label} terminal result changed")
    from scripts.v8_local_train import validate_offline_terminal_receipt

    receipt_path = directory / "wandb_offline_terminal_receipt.json"
    validate_offline_terminal_receipt(
        output_dir=directory,
        expected_identity=_expected_wandb_identity(config),
        expected_wandb_dir=OFFLINE_WANDB_DIR / label,
    )
    if require_dispatch_completion:
        _verify_dispatch_binding(label, sha256_file(result_path), sha256_file(receipt_path))
    return {
        "label": label,
        "source_tree_sha256": manifest["source_tree_sha256"],
        "train_source_indices": train_indices,
        "run_result_sha256": sha256_file(result_path),
        "offline_receipt_sha256": sha256_file(receipt_path),
        "history": {step: float(row["exact_match"]) for step, row in history.items()},
    }


def verify_completed_runs(conditions: Sequence[str] = CONDITIONS) -> dict[str, Any]:
    if not conditions or any(condition not in CONDITIONS for condition in conditions):
        raise ProtocolError(f"invalid V8-local conditions: {conditions}")
    runs: dict[str, Any] = {}
    source_trees: set[str] = set()
    matched_indices: dict[int, list[int]] = {}
    for condition in conditions:
        for seed in SEEDS:
            item = validate_training_run(condition, seed)
            runs[item["label"]] = {key: value for key, value in item.items() if key != "history"}
            source_trees.add(item["source_tree_sha256"])
            if seed in matched_indices and matched_indices[seed] != item["train_source_indices"]:
                raise ProtocolError(f"matched treatment/control data differ for seed {seed}")
            matched_indices[seed] = item["train_source_indices"]
    if len(source_trees) != 1:
        raise ProtocolError("V8-local runs used different source trees")
    return {"source_tree_sha256": next(iter(source_trees)), "runs": runs}


def _curve_means_pass(values: Sequence[float]) -> bool:
    if len(values) != 4 or any(not math.isfinite(float(value)) for value in values):
        raise ProtocolError("curve gate requires four finite registered means")
    return values[1] > values[0] and values[2] >= values[1] and values[3] >= values[2]


def compute_curve_gate(*, write_result: bool = True) -> dict[str, Any]:
    if write_result and (CURVE_PATH.exists() or CURVE_PLOT_PATH.exists()):
        raise ProtocolError("refusing to overwrite the one V8-local curve decision")
    state = load_and_verify_state(require_launch=True)
    verify_completed_runs(("jlens",))
    per_seed: dict[str, dict[str, float]] = {}
    all_histories: dict[str, dict[int, float]] = {}
    for seed in SEEDS:
        item = validate_training_run("jlens", seed)
        all_histories[str(seed)] = item["history"]
        per_seed[str(seed)] = {
            str(step): item["history"][step] for step in CURVE_STEPS
        }
    means = {
        str(step): sum(per_seed[str(seed)][str(step)] for seed in SEEDS) / len(SEEDS)
        for step in CURVE_STEPS
    }
    result = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "git_commit": state["git_commit"],
        "registration_sha256": state["registration_sha256"],
        "data_role": "exposed_development_operational_consistency_gate_not_inference",
        "exposure_caveat": read_json(REPRO_DIR / "registration.json")[
            "curve_exposure_caveat"
        ],
        "criterion": CURVE_CRITERION,
        "steps": list(CURVE_STEPS),
        "per_seed_exact_match": per_seed,
        "mean_exact_match": means,
        "passed": _curve_means_pass([means[str(step)] for step in CURVE_STEPS]),
        "computed_at_utc": utc_now(),
    }
    if write_result:
        try:
            import matplotlib.pyplot as plt
        except ImportError as error:
            raise ProtocolError("curve plot requires matplotlib") from error
        figure, axis = plt.subplots(figsize=(8, 4.8))
        steps = sorted(next(iter(all_histories.values())))
        for seed in SEEDS:
            axis.plot(
                steps,
                [100 * all_histories[str(seed)][step] for step in steps],
                color="#94a3b8",
                alpha=0.3,
                marker=".",
            )
        axis.plot(
            list(CURVE_STEPS),
            [100 * means[str(step)] for step in CURVE_STEPS],
            color="#b91c1c",
            linewidth=2.5,
            marker="o",
            label="registered eight-seed mean",
        )
        axis.set_xlabel("Optimizer update")
        axis.set_ylabel("Greedy exact match (%)")
        axis.set_title("V8-local profanity: exposed development consistency curve")
        axis.grid(alpha=0.2)
        axis.legend()
        figure.tight_layout()
        CURVE_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(CURVE_PLOT_PATH, dpi=180)
        plt.close(figure)
        result["curve_plot_sha256"] = sha256_file(CURVE_PLOT_PATH)
        write_json(CURVE_PATH, result, exclusive=True)
    return result


def verify_curve_gate(*, require_pass: bool = True) -> dict[str, Any]:
    if not CURVE_PATH.is_file() or not CURVE_PLOT_PATH.is_file():
        raise ProtocolError("V8-local curve decision has not been recorded")
    stored = read_json(CURVE_PATH)
    recomputed = compute_curve_gate(write_result=False)
    for key in (
        "protocol",
        "git_commit",
        "registration_sha256",
        "data_role",
        "exposure_caveat",
        "criterion",
        "steps",
        "per_seed_exact_match",
        "mean_exact_match",
        "passed",
    ):
        if stored.get(key) != recomputed.get(key):
            raise ProtocolError("stored V8-local curve decision changed")
    if stored.get("curve_plot_sha256") != sha256_file(CURVE_PLOT_PATH):
        raise ProtocolError("V8-local curve plot changed")
    if require_pass and stored.get("passed") is not True:
        raise ProtocolError("V8-local exposed-development consistency gate failed")
    return stored


def unlock_final() -> dict[str, Any]:
    if UNLOCK_PATH.exists() or COMPLETED_RUNS_PATH.exists():
        raise ProtocolError("V8-local final unlock already exists")
    state = load_and_verify_state(require_launch=True)
    completed = verify_completed_runs(CONDITIONS)
    gate = verify_curve_gate(require_pass=True)
    completed.update(
        {
            "protocol": PROTOCOL,
            "git_commit": state["git_commit"],
            "registration_sha256": state["registration_sha256"],
        }
    )
    write_json(COMPLETED_RUNS_PATH, completed, exclusive=True)
    marker = {
        "protocol": PROTOCOL,
        "git_commit": state["git_commit"],
        "registration_sha256": state["registration_sha256"],
        "curve_gate_sha256": sha256_file(CURVE_PATH),
        "completed_runs_sha256": sha256_file(COMPLETED_RUNS_PATH),
        "reason": "all eight treatments, passing registered curve, and eight controls verified",
        "unlocked_at_utc": utc_now(),
    }
    write_json(UNLOCK_PATH, marker, exclusive=True)
    return marker


def verify_unlock() -> dict[str, Any]:
    if not UNLOCK_PATH.is_file() or not COMPLETED_RUNS_PATH.is_file():
        raise ProtocolError("V8-local sealed final is not unlocked")
    marker = read_json(UNLOCK_PATH)
    completed = verify_completed_runs(CONDITIONS)
    recorded = read_json(COMPLETED_RUNS_PATH)
    if (
        recorded.get("source_tree_sha256") != completed["source_tree_sha256"]
        or recorded.get("runs") != completed["runs"]
        or marker.get("completed_runs_sha256") != sha256_file(COMPLETED_RUNS_PATH)
        or marker.get("curve_gate_sha256") != sha256_file(CURVE_PATH)
    ):
        raise ProtocolError("V8-local final unlock changed")
    verify_curve_gate(require_pass=True)
    return marker


def begin_final_collection(collection_id: str) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{32}", collection_id) is None:
        raise ProtocolError("final collection ID must be 32 lowercase hexadecimal characters")
    verify_unlock()
    if COLLECTION_PATH.exists() or EVAL_DIR.exists() or COMPARISON_PATH.exists():
        raise ProtocolError("V8-local sealed final was already claimed or opened")
    state = load_and_verify_state(require_launch=True)
    marker = {
        "protocol": PROTOCOL,
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
    write_json(COLLECTION_PATH, marker, exclusive=True)
    return marker


def verify_final_collection(collection_id: str | None = None) -> dict[str, Any]:
    verify_unlock()
    if not COLLECTION_PATH.is_file():
        raise ProtocolError("V8-local final collection has not been claimed")
    marker = read_json(COLLECTION_PATH)
    if (
        marker.get("labels") != list(FINAL_LABELS)
        or marker.get("sealed_manifest_sha256") != FINAL_SHA256
        or marker.get("unlock_sha256") != sha256_file(UNLOCK_PATH)
        or (collection_id is not None and marker.get("collection_id") != collection_id)
    ):
        raise ProtocolError("V8-local final collection marker changed")
    return marker


def verify_design() -> dict[str, Any]:
    recipe = _validate_recipe(read_json(RECIPE_SOURCE))
    source_inputs = _source_inputs()
    amendment, closeout = _amendment()
    draft = read_json(REGISTRATION_DRAFT_SOURCE)
    if draft.get("launch_gate", {}).get("enabled") is not False:
        raise ProtocolError("V8-local draft unexpectedly enables launch")
    return {
        "protocol": PROTOCOL,
        "status": (
            "design_valid_launch_authorized_pending_clean_committed_pushed_source"
            if closeout is not None
            else "design_valid_launch_disabled"
        ),
        "recipe_lock_sha256": sha256_file(RECIPE_SOURCE),
        "registration_draft_sha256": sha256_file(REGISTRATION_DRAFT_SOURCE),
        "amendment_template_sha256": sha256_file(AMENDMENT_TEMPLATE_SOURCE),
        "source_input_sha256": source_inputs,
        "code_identity": _code_identity(),
        "seeds": list(SEEDS),
        "curve_steps": list(CURVE_STEPS),
        "curve_role": recipe["curve_gate"]["role"],
        "launch_enabled": amendment.get("launch_enabled"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "verify-design",
            "prepare",
            "verify",
            "verify-launch",
            "probe-hardware",
            "verify-treatments",
            "curve",
            "verify-curve",
            "verify-runs",
            "unlock",
            "verify-unlock",
            "begin-final",
            "verify-final",
        ),
    )
    parser.add_argument("--collection-id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.command == "verify-design":
            result = verify_design()
        elif args.command == "prepare":
            result = prepare()
        elif args.command == "verify":
            result = load_and_verify_state()
        elif args.command == "verify-launch":
            result = load_and_verify_state(require_launch=True)
        elif args.command == "probe-hardware":
            result = probe_hardware(require_idle=True)
        elif args.command == "verify-treatments":
            result = verify_completed_runs(("jlens",))
        elif args.command == "curve":
            result = compute_curve_gate(write_result=True)
        elif args.command == "verify-curve":
            result = verify_curve_gate(require_pass=False)
        elif args.command == "verify-runs":
            result = verify_completed_runs(CONDITIONS)
        elif args.command == "unlock":
            result = unlock_final()
        elif args.command == "verify-unlock":
            result = verify_unlock()
        elif args.command == "begin-final":
            if args.collection_id is None:
                raise ProtocolError("begin-final requires --collection-id")
            result = begin_final_collection(args.collection_id)
        else:
            result = verify_final_collection(args.collection_id)
        print(json.dumps(result, indent=2, sort_keys=True))
    except ProtocolError as error:
        print(f"protocol error: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
