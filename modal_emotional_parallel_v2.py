"""Run five fixed emotionally charged development ideas concurrently on Modal.

This is an exposed-curve development screen, never a significance attempt.
The image and fresh Volume contain no sealed, reserve, or correlation payload.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

from modal_emotional_tournament_v1 import repo_image as cached_tournament_image


LOCAL_REPO = Path(__file__).resolve().parent
REMOTE_REPO = Path("/workspace/j-lens-rl")
REMOTE_STATE = Path("/state")
APP_NAME = "j-lens-rl-development-emotional-parallel-v3"
VOLUME_NAME = "j-lens-rl-development-emotional-parallel-v3-20260714b"
PROTOCOL = "j-lens-rl-development-emotional-parallel-v3"
REGISTRATION_SHA256 = "6eeee93e2cca1d5c4167eda682bf710940ba30a2b971bf85e82f479b9329e4dc"
GPU_TYPE = "L40S"
MAX_PARALLEL_GPUS = 5
ARM_ORDER = (
    "joy_u2_h6_seed194",
    "celebration_tail_u4_h20_seed195",
    "excited_u2_h6_seed196",
    "wow_u2_h6_seed197",
    "fuck_penalty_u2_h6_seed198",
)
ARM_CONFIG = {
    "joy_u2_h6_seed194": "configs/emotional_parallel_v3_joy.json",
    "celebration_tail_u4_h20_seed195": "configs/emotional_parallel_v3_celebration.json",
    "excited_u2_h6_seed196": "configs/emotional_parallel_v3_excited.json",
    "wow_u2_h6_seed197": "configs/emotional_parallel_v3_wow.json",
    "fuck_penalty_u2_h6_seed198": "configs/emotional_parallel_v3_fuck.json",
}
ARM_STEPS = {
    "joy_u2_h6_seed194": (0, 2, 4, 6),
    "celebration_tail_u4_h20_seed195": (0, 4, 10, 20),
    "excited_u2_h6_seed196": (0, 2, 4, 6),
    "wow_u2_h6_seed197": (0, 2, 4, 6),
    "fuck_penalty_u2_h6_seed198": (0, 2, 4, 6),
}
ARM_WANDB_IDS = {
    "joy_u2_h6_seed194": "dev-v12-five-joy-u2-h6-seed194",
    "celebration_tail_u4_h20_seed195": (
        "dev-v12-five-celebration-tail-u4-h20-seed195"
    ),
    "excited_u2_h6_seed196": "dev-v12-five-excited-u2-h6-seed196",
    "wow_u2_h6_seed197": "dev-v12-five-wow-u2-h6-seed197",
    "fuck_penalty_u2_h6_seed198": "dev-v12-five-fuck-penalty-u2-h6-seed198",
}
EXPECTED_FILE_SHA256 = {
    ".gitignore": (
        "2093c1ee68d1070775e3fc36502041a32ade3c15e70c670d628e5b92060e665c"
    ),
    ".confirmatory/manifests/curve_indices.json": (
        "ad348fe17d2e6bd6aac691d9bcdbb9da481f675305fa0e05c68e86dad97451c1"
    ),
    ".confirmatory/manifests/train_exclusions.json": (
        "7c1ca4f404ba9149093cc3c57dc3607582f671397e2fe99e93449848c1d65d61"
    ),
    "artifacts/qwen25_05b_solved_lens.pt": (
        "178a9671cbf41882135807bde59b828e36c6f8f98b32c809ea3346860aad10dc"
    ),
    "configs/emotional_parallel_v3_celebration.json": (
        "3f98bfc107ee83a23d8da2c188d3caf32a1c8befd697671bbaeca6001f760669"
    ),
    "configs/emotional_parallel_v3_common.json": (
        "d4e8b8495b5df4b91a3110ef0baab08c1dcda1a5ca88b00fc4b45b099ba133ef"
    ),
    "configs/emotional_parallel_v3_excited.json": (
        "e40e7db62ef11120817c11c45a88cc501465535823c39e0cd11cfe913d3f2213"
    ),
    "configs/emotional_parallel_v3_fuck.json": (
        "e4c9de19aef2b5f44fccdd9085f5e99bdff8ac1e5cde6352128eb23030759691"
    ),
    "configs/emotional_parallel_v3_joy.json": (
        "46c8ac4e370c3ea789e9aeb3432c83eff93d643ca90acc077be2181d71ed1948"
    ),
    "configs/emotional_parallel_v3_wow.json": (
        "f31b00cfbd03185d9ea383ca98b0087d932edb182eeea8eb756824673030cff5"
    ),
    "protocol_archive/emotional_parallel_v2_pretraining_attempt_a_closeout.json": (
        "a9513baa074ff6803bf826b5a2d7dad7fff9ea63e79c5b5faca403210a5cdeec"
    ),
    "protocol_archive/emotional_parallel_v3_metric_schema.json": (
        "3ae834c6237dc1e4d5b996c6b7f16ed7f073ead70d4ab8464b06318875633b20"
    ),
    "protocol_archive/emotional_parallel_v3_registration.json": REGISTRATION_SHA256,
    "protocol_archive/emotional_screen_forensic_bundle/single_word/artifacts/excited_calibration.json": (
        "a09bcdbdf4c18c5680f2c73af35ec435b2659790f67f4fe2d415ba5d4720d2b0"
    ),
    "protocol_archive/emotional_screen_forensic_bundle/single_word/artifacts/fuck_calibration.json": (
        "f53ab990d2061f34ccf62f0bcafdc83304aab3747b3d189d279528125f67dc8d"
    ),
    "protocol_archive/emotional_screen_forensic_bundle/family/artifacts/celebration_calibration.json": (
        "93d05caf4848e745c07d908034b36f0b1ae465d8d89e1681134869c6b87a8ee6"
    ),
    "protocol_archive/emotional_screen_forensic_bundle/single_word/artifacts/joy_calibration.json": (
        "71979e2c36b10d759fc92d5b16e780ef699a34f1b7ea890a8c5f00257c8e2021"
    ),
    "protocol_archive/emotional_screen_forensic_bundle/single_word/artifacts/wow_calibration.json": (
        "7dd8f37e51eb74dff050e86a74d2bdd807bb5d066657e1124ec3e98a967cd2b6"
    ),
}
ADDED_IMAGE_FILES = (
    *EXPECTED_FILE_SHA256,
    "modal_emotional_parallel_v2.py",
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())


parallel_image = cached_tournament_image
for relative in ADDED_IMAGE_FILES:
    parallel_image = parallel_image.add_local_file(
        LOCAL_REPO / relative,
        (REMOTE_REPO / relative).as_posix(),
        copy=True,
    )
parallel_image = (
    parallel_image.env(
        {
            "GIT_AUTHOR_NAME": "J-Lens Parallel Runtime",
            "GIT_AUTHOR_EMAIL": "runtime@example.invalid",
            "GIT_COMMITTER_NAME": "J-Lens Parallel Runtime",
            "GIT_COMMITTER_EMAIL": "runtime@example.invalid",
            "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+00:00",
            "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+00:00",
            "JLENS_REPOSITORY_ROOT": REMOTE_REPO.as_posix(),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    .run_commands(
        "find . -type d -name __pycache__ -prune -exec rm -rf {} +",
        "find . -type d -name '*.egg-info' -prune -exec rm -rf {} +",
        "rm -rf .git",
        "git init -q",
        "git add -f .",
        "git commit -qm 'J-Lens five-emotional-idea runtime'",
        "test -z \"$(git status --porcelain=v1 --untracked-files=all)\"",
    )
)

app = modal.App(APP_NAME)
state_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False, version=2)
wandb_secret = modal.Secret.from_name("j-lens-rl-wandb", required_keys=["WANDB_API_KEY"])


def _verify_runtime_files() -> dict[str, str]:
    observed = {}
    for relative, expected in EXPECTED_FILE_SHA256.items():
        path = REMOTE_REPO / relative
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"runtime input changed: {relative}: {actual} != {expected}")
        observed[relative] = actual
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=REMOTE_REPO,
        text=True,
    )
    if status:
        raise RuntimeError(f"runtime Git worktree is dirty: {status}")
    return observed


def _read_claim(claim_id: str) -> dict[str, Any]:
    path = REMOTE_STATE / "attempt_claim.json"
    if not path.is_file():
        raise RuntimeError("parallel attempt has no durable claim")
    value = json.loads(path.read_text())
    if (
        value.get("claim_id") != claim_id
        or value.get("protocol") != PROTOCOL
        or value.get("registration_sha256") != REGISTRATION_SHA256
        or value.get("max_parallel_gpus") != MAX_PARALLEL_GPUS
        or value.get("arms") != list(ARM_ORDER)
    ):
        raise RuntimeError("parallel attempt claim changed")
    return value


@app.function(
    image=parallel_image,
    cpu=1,
    memory=2048,
    timeout=20 * 60,
    retries=0,
    volumes={REMOTE_STATE: state_volume},
)
def claim_attempt(claim_id: str, preflight: dict[str, Any]) -> dict[str, Any]:
    state_volume.reload()
    if any(REMOTE_STATE.iterdir()):
        raise RuntimeError("registered fresh Volume is not empty")
    value = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "scientific_status": "development_only",
        "claim_id": claim_id,
        "registration_sha256": REGISTRATION_SHA256,
        "arms": list(ARM_ORDER),
        "max_parallel_gpus": MAX_PARALLEL_GPUS,
        "source_main_commit": preflight["source_main_commit"],
        "source_tree_sha256": preflight["source_tree_sha256"],
        "volume_name": VOLUME_NAME,
        "preflight": preflight,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_exclusive(REMOTE_STATE / "attempt_claim.json", value)
    state_volume.commit()
    return value


@app.function(
    image=parallel_image,
    cpu=1,
    memory=2048,
    timeout=20 * 60,
    retries=0,
    volumes={REMOTE_STATE: state_volume},
)
def record_launch_receipt(claim_id: str, app_id: str, root_call_id: str) -> dict[str, Any]:
    state_volume.reload()
    claim = _read_claim(claim_id)
    value = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "claim_id": claim_id,
        "app_id": app_id,
        "root_call_id": root_call_id,
        "gpu_type": GPU_TYPE,
        "max_parallel_gpus": MAX_PARALLEL_GPUS,
        "wandb_run_ids": ARM_WANDB_IDS,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_exclusive(REMOTE_STATE / "launch_receipt.json", value)
    state_volume.commit()
    return value


def _wait_for_launch_receipt(claim_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        state_volume.reload()
        path = REMOTE_STATE / "launch_receipt.json"
        if path.is_file():
            value = json.loads(path.read_text())
            if value.get("claim_id") == claim_id:
                return value
        time.sleep(1)
    raise RuntimeError("durable launch receipt did not arrive")


@app.function(
    image=parallel_image,
    gpu=GPU_TYPE,
    cpu=4,
    memory=32768,
    max_containers=MAX_PARALLEL_GPUS,
    timeout=3 * 60 * 60,
    startup_timeout=60 * 60,
    retries=0,
    single_use_containers=True,
    secrets=[wandb_secret],
    volumes={REMOTE_STATE: state_volume},
)
def train_arm(label: str, claim_id: str) -> dict[str, Any]:
    if label not in ARM_ORDER:
        raise ValueError("unregistered parallel arm")
    state_volume.reload()
    claim = _read_claim(claim_id)
    source_hashes = _verify_runtime_files()
    import torch

    if torch.cuda.device_count() != 1 or GPU_TYPE not in torch.cuda.get_device_name(0):
        raise RuntimeError("parallel worker did not receive exactly one registered L40S")
    run_dir = REMOTE_STATE / "runs" / label
    if run_dir.exists():
        raise RuntimeError("parallel arm already has output; resume/replay is forbidden")
    dispatch_dir = REMOTE_STATE / "dispatches"
    intent_path = dispatch_dir / f"{label}.intent.json"
    completion_path = dispatch_dir / f"{label}.completion.json"
    failure_path = dispatch_dir / f"{label}.failure.json"
    if any(path.exists() for path in (intent_path, completion_path, failure_path)):
        raise RuntimeError("parallel arm dispatch already exists")
    intent = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "claim_id": claim_id,
        "label": label,
        "function_call_id": modal.current_function_call_id(),
        "gpu": torch.cuda.get_device_name(0),
        "config": ARM_CONFIG[label],
        "config_sha256": EXPECTED_FILE_SHA256[ARM_CONFIG[label]],
        "registration_sha256": REGISTRATION_SHA256,
        "source_main_commit": claim["source_main_commit"],
        "source_file_sha256": source_hashes,
        "status": "written_and_fsynced_before_training",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_exclusive(intent_path, intent)
    state_volume.commit()
    command = [
        sys.executable,
        "-m",
        "jlens_rl.train",
        "--config",
        ARM_CONFIG[label],
        "--wandb-mode",
        "online",
    ]
    completed = subprocess.run(
        command,
        cwd=REMOTE_REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    log_dir = REMOTE_STATE / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{label}.stdout"
    stderr_path = log_dir / f"{label}.stderr"
    stdout_path.write_text(completed.stdout)
    stderr_path.write_text(completed.stderr)
    if completed.returncode != 0:
        failure = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "claim_id": claim_id,
            "label": label,
            "intent_sha256": _sha256(intent_path),
            "returncode": completed.returncode,
            "stdout_sha256": _sha256(stdout_path),
            "stderr_sha256": _sha256(stderr_path),
            "retry_or_resume_permitted": False,
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_exclusive(failure_path, failure)
        state_volume.commit()
        raise RuntimeError(f"parallel arm {label} failed closed")
    result_path = run_dir / "run_result_manifest.json"
    receipt_path = run_dir / "wandb_terminal_publish_receipt.json"
    history_path = run_dir / "validation_history.jsonl"
    if not all(path.is_file() for path in (result_path, receipt_path, history_path)):
        raise RuntimeError("parallel arm lacks terminal/W&B evidence")
    rows = [json.loads(line) for line in history_path.read_text().splitlines() if line]
    if tuple(row.get("step") for row in rows) != ARM_STEPS[label]:
        raise RuntimeError("parallel arm curve is incomplete or reordered")
    result = json.loads(result_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    if (
        result.get("registration_sha256") != REGISTRATION_SHA256
        or result.get("wandb_identity", {}).get("run_id") != ARM_WANDB_IDS[label]
        or receipt.get("terminal_run_result_sha256") != _sha256(result_path)
    ):
        raise RuntimeError("parallel arm terminal identity changed")
    completion = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "claim_id": claim_id,
        "label": label,
        "intent_sha256": _sha256(intent_path),
        "returncode": completed.returncode,
        "stdout_sha256": _sha256(stdout_path),
        "stderr_sha256": _sha256(stderr_path),
        "run_result_manifest_sha256": _sha256(result_path),
        "wandb_terminal_publish_receipt_sha256": _sha256(receipt_path),
        "validation_history_sha256": _sha256(history_path),
        "curve": {str(row["step"]): row["exact_match"] for row in rows},
        "literal_target_completion_rate": {
            str(row["step"]): row["literal_target_completion_rate"] for row in rows
        },
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_exclusive(completion_path, completion)
    state_volume.commit()
    return completion


@app.function(
    image=parallel_image,
    cpu=1,
    memory=2048,
    max_containers=1,
    timeout=4 * 60 * 60,
    startup_timeout=60 * 60,
    retries=0,
    volumes={REMOTE_STATE: state_volume},
)
def orchestrate(claim_id: str) -> dict[str, Any]:
    try:
        receipt = _wait_for_launch_receipt(claim_id)
        root_call_id = modal.current_function_call_id()
        if receipt.get("root_call_id") != root_call_id:
            raise RuntimeError("orchestrator lacks durable root authority")
        calls = {label: train_arm.spawn(label, claim_id) for label in ARM_ORDER}
        status = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "claim_id": claim_id,
            "stage": "training_five_parallel_ideas",
            "max_parallel_gpus": MAX_PARALLEL_GPUS,
            "worker_call_ids": {label: call.object_id for label, call in calls.items()},
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_exclusive(REMOTE_STATE / "attempt_status.json", status)
        state_volume.commit()
        results = {label: calls[label].get() for label in ARM_ORDER}
        summary = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "scientific_status": "development_only_no_significance_claim",
            "claim_id": claim_id,
            "stage": "complete",
            "results": results,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_exclusive(REMOTE_STATE / "evidence" / "summary.json", summary)
        status["stage"] = "complete"
        status["completed_arms"] = list(ARM_ORDER)
        status["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        temporary = REMOTE_STATE / "attempt_status.json.tmp"
        temporary.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
        temporary.replace(REMOTE_STATE / "attempt_status.json")
        state_volume.commit()
        return summary
    except BaseException as error:
        state_volume.reload()
        status_path = REMOTE_STATE / "attempt_status.json"
        failure = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "claim_id": claim_id,
            "stage": "failed_closed",
            "error_type": type(error).__name__,
            "error": str(error),
            "retry_or_resume_permitted": False,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        temporary = status_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        temporary.replace(status_path)
        state_volume.commit()
        raise


def _local_preflight() -> dict[str, Any]:
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=LOCAL_REPO,
        text=True,
    )
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=LOCAL_REPO, text=True).strip()
    remote = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=LOCAL_REPO, text=True
    ).strip()
    if status or head != remote:
        raise RuntimeError("five-arm launch requires clean pushed main")
    for relative, expected in EXPECTED_FILE_SHA256.items():
        if _sha256(LOCAL_REPO / relative) != expected:
            raise RuntimeError(f"local registered input changed: {relative}")
    modal_cli = Path(sys.executable).with_name("modal")
    listing = json.loads(
        subprocess.check_output([str(modal_cli), "app", "list", "--json"], text=True)
    )
    active_other = [
        item
        for item in listing
        if item.get("stopped_at") is None
        and item.get("state") != "stopped"
        and item.get("app_id") != app.app_id
    ]
    if active_other:
        raise RuntimeError(f"another Modal app is active: {active_other}")
    state_volume.hydrate()
    volume_id = state_volume.object_id
    inventory_text = subprocess.check_output(
        [str(modal_cli), "volume", "ls", VOLUME_NAME, "/", "--json"], text=True
    )
    inventory = json.loads(inventory_text[inventory_text.index("[") :])
    if inventory:
        raise RuntimeError("five-arm Volume is not fresh and empty")
    tree_hash = hashlib.sha256()
    for relative in sorted(EXPECTED_FILE_SHA256):
        tree_hash.update(relative.encode())
        tree_hash.update(b"\0")
        tree_hash.update(EXPECTED_FILE_SHA256[relative].encode())
        tree_hash.update(b"\0")
    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_main_commit": head,
        "source_tree_sha256": tree_hash.hexdigest(),
        "registration_sha256": REGISTRATION_SHA256,
        "registered_file_sha256": EXPECTED_FILE_SHA256,
        "active_other_modal_apps": [],
        "volume_name": VOLUME_NAME,
        "volume_object_id": volume_id,
        "volume_version": 2,
        "max_parallel_gpus": MAX_PARALLEL_GPUS,
    }


@app.local_entrypoint()
def main() -> None:
    preflight = _local_preflight()
    claim_id = uuid.uuid4().hex
    claim_attempt.remote(claim_id, preflight)
    call = orchestrate.spawn(claim_id)
    receipt = record_launch_receipt.remote(claim_id, app.app_id or APP_NAME, call.object_id)
    print(
        json.dumps(
            {
                "status": "submitted",
                "scientific_status": "development_only",
                "app_id": app.app_id,
                "root_call_id": call.object_id,
                "claim_id": claim_id,
                "volume": VOLUME_NAME,
                "gpu_type": GPU_TYPE,
                "max_parallel_gpus": MAX_PARALLEL_GPUS,
                "arms": list(ARM_ORDER),
                "wandb_run_ids": ARM_WANDB_IDS,
                "preflight": preflight,
                "launch_receipt": receipt,
                "no_significance_or_final_claim": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
