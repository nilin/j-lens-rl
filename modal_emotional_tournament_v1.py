"""Run the post-V7 three-arm emotional-word development tournament on Modal.

The runner is intentionally nonlaunchable until the prepared state contains a
committed amendment pinning V7's terminal closeout.  It uses the workspace-wide
one-GPU lease, a fresh noncreating v2 Volume, and exactly one serial L40S worker.
No sealed/final/reserve/correlation payload is present in either image or state.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

from scripts import emotional_tournament_v1_protocol as protocol
from scripts.modal_verify_tournament_v1_volume import (
    verify_tournament_v1_volume_v2,
)


LOCAL_REPO = Path(__file__).resolve().parent
LOCAL_STATE = LOCAL_REPO / ".confirmatory" / "v8"
REMOTE_REPO = Path("/workspace/j-lens-rl")
REMOTE_STATE = REMOTE_REPO / ".confirmatory" / "v8"

APP_NAME = protocol.APP_NAME
VOLUME_NAME = protocol.VOLUME_NAME
ARM_ORDER = protocol.ARM_ORDER
SEED = protocol.SEED
CURVE_STEPS = protocol.CURVE_STEPS
GPU_TYPE = protocol.GPU_TYPE
MAX_GPU_CONTAINERS = 1
GLOBAL_MODAL_GPU_LIMIT = 1
GPU_EXCLUSIVE_CONFIRMATION = "confirmed-no-other-modal-gpu-app-running"
GPU_APP_OVERLAP_POLICY = "no other active Modal app may overlap this tournament"
GPU_LEASE_DICT_NAME = protocol.GPU_LEASE_DICT_NAME
GPU_LEASE_KEY = protocol.GPU_LEASE_KEY
GPU_LEASE_ENVIRONMENT = protocol.GPU_LEASE_ENVIRONMENT
GPU_LEASE_PROTOCOL = "j-lens-rl-global-gpu-lease-v1"
GPU_LEASE_WORKLOAD = "development-emotional-tournament-v1"
GPU_DISPATCH_DIR = REMOTE_STATE / "gpu_dispatches"


app = modal.App(APP_NAME)
state_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False, version=2)
gpu_lease = modal.Dict.from_name(
    GPU_LEASE_DICT_NAME,
    environment_name=GPU_LEASE_ENVIRONMENT,
    create_if_missing=False,
)
wandb_secret = modal.Secret.from_name("j-lens-rl-wandb")
huggingface_secret = modal.Secret.from_name(
    "huggingface-token", required_keys=["HF_TOKEN"]
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


RUNTIME_SOURCE_SHA256 = protocol.runtime_source_hashes()
RUNTIME_SOURCE_SHA256_JSON = json.dumps(
    RUNTIME_SOURCE_SHA256, sort_keys=True, separators=(",", ":")
)

repo_image = modal.Image.debian_slim(python_version="3.11")
for relative in sorted(RUNTIME_SOURCE_SHA256):
    repo_image = repo_image.add_local_file(
        LOCAL_REPO / relative,
        (REMOTE_REPO / relative).as_posix(),
        copy=True,
    )
repo_image = (
    repo_image.apt_install("git")
    .workdir(REMOTE_REPO)
    .env(
        {
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "JLENS_CONFIRMATORY_RUNTIME_STATE_ONLY": "1",
            "JLENS_REPOSITORY_ROOT": REMOTE_REPO.as_posix(),
            "JLENS_MODAL_IMAGE_SPEC": (
                "j-lens-rl-development-emotional-tournament-v1-strict-allowlist"
            ),
            "JLENS_TOURNAMENT_V1_IMAGE_FILE_SHA256": RUNTIME_SOURCE_SHA256_JSON,
            "PYTHONPATH": (
                f"{(REMOTE_REPO / 'src').as_posix()}:"
                f"{(REMOTE_REPO / 'trl').as_posix()}"
            ),
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .run_commands(
        "python -m pip install --upgrade pip==26.0.1",
        "python -m pip install './trl[peft]' '.[dev]'",
    )
    .run_commands(
        "python scripts/modal_cache_assets_tournament_v1.py",
        secrets=[huggingface_secret],
    )
    .run_commands("python scripts/modal_finalize_image_tournament_v1.py")
)


def _run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    rendered = subprocess.run(command, cwd=REMOTE_REPO, check=False, text=True)
    if check and rendered.returncode:
        raise subprocess.CalledProcessError(rendered.returncode, command)
    return rendered


def _protocol(command: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return _run(
        [
            sys.executable,
            "scripts/emotional_tournament_v1_protocol.py",
            command,
        ],
        check=check,
    )


def _write_json_exclusive_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_status(claim_id: str, stage: str, **details: Any) -> dict[str, Any]:
    claim = _attempt_claim(claim_id)
    status = {
        "schema_version": 1,
        "protocol": protocol.PROTOCOL,
        "claim_id": claim["claim_id"],
        "stage": stage,
        "scientific_status": "development_only",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        **details,
    }
    path = REMOTE_STATE / "attempt_status.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return status


def _ensure_runtime_git() -> str:
    manifest = json.loads(
        (REMOTE_STATE / "reproducibility" / "source_manifest.json").read_text()
    )
    files = manifest.get("files")
    expected_commit = manifest.get("git_commit")
    recipe = manifest.get("runtime_commit_recipe")
    if (
        not isinstance(files, dict)
        or recipe
        != {
            "author": "J-Lens Tournament Runtime <runtime@example.invalid>",
            "timestamp": "2000-01-01T00:00:00+00:00",
            "message": "J-Lens emotional tournament strict runtime source",
            "parent": None,
        }
        or not isinstance(expected_commit, str)
    ):
        raise RuntimeError("tournament runtime Git recipe is absent or changed")
    for name, identity in files.items():
        path = REMOTE_REPO / name
        if (
            not path.is_file()
            or path.is_symlink()
            or _sha256(path) != identity.get("sha256")
            or path.stat().st_size != identity.get("size_bytes")
        ):
            raise RuntimeError(f"runtime source changed before Git setup: {name}")
        path.chmod(int(identity["mode"]))
    git_dir = REMOTE_REPO / ".git"
    if not git_dir.exists():
        environment = {
            **os.environ,
            "GIT_AUTHOR_NAME": "J-Lens Tournament Runtime",
            "GIT_AUTHOR_EMAIL": "runtime@example.invalid",
            "GIT_COMMITTER_NAME": "J-Lens Tournament Runtime",
            "GIT_COMMITTER_EMAIL": "runtime@example.invalid",
            "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+00:00",
            "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+00:00",
        }
        subprocess.run(["git", "init", "-q"], cwd=REMOTE_REPO, check=True, env=environment)
        subprocess.run(
            ["git", "add", "--", *sorted(files)],
            cwd=REMOTE_REPO,
            check=True,
            env=environment,
        )
        tree = subprocess.check_output(
            ["git", "write-tree"], cwd=REMOTE_REPO, text=True, env=environment
        ).strip()
        actual_commit = subprocess.check_output(
            ["git", "commit-tree", tree],
            cwd=REMOTE_REPO,
            input="J-Lens emotional tournament strict runtime source\n",
            text=True,
            env=environment,
        ).strip()
        if actual_commit != expected_commit:
            raise RuntimeError("reconstructed runtime Git identity differs")
        subprocess.run(
            ["git", "update-ref", "HEAD", actual_commit],
            cwd=REMOTE_REPO,
            check=True,
            env=environment,
        )
        (git_dir / "info" / "exclude").write_text("*\n")
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REMOTE_REPO, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=REMOTE_REPO,
        text=True,
    )
    if actual != expected_commit or status:
        raise RuntimeError("runtime Git worktree is not exact and clean")
    return actual


def _attempt_claim(claim_id: str) -> dict[str, Any]:
    path = REMOTE_STATE / "attempt_claim.json"
    if not path.is_file():
        raise RuntimeError("tournament Volume has no immutable claim")
    claim = json.loads(path.read_text())
    if claim.get("claim_id") != claim_id:
        raise RuntimeError("tournament Volume belongs to another claim")
    return claim


def _validate_operational_preflight(value: Any) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or value.get("exclusive_gpu_confirmation") != GPU_EXCLUSIVE_CONFIRMATION
        or value.get("global_modal_gpu_limit") != 1
        or value.get("active_other_modal_apps") != []
        or value.get("v7_app_id") != protocol.V7_APP_ID
        or value.get("v7_app_observed_active") is not False
        or value.get("volume_name") != VOLUME_NAME
        or value.get("volume_version") != 2
        or not isinstance(value.get("volume_object_id"), str)
        or value.get("gpu_lease_dict") != GPU_LEASE_DICT_NAME
        or value.get("gpu_lease_key") != GPU_LEASE_KEY
        or value.get("gpu_lease_owner_at_preflight") is not None
    ):
        raise RuntimeError("tournament operational preflight is invalid")
    return value


def _validate_lease_record(value: Any) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or value.get("protocol") != GPU_LEASE_PROTOCOL
        or value.get("environment_name") != GPU_LEASE_ENVIRONMENT
        or value.get("slot") != GPU_LEASE_KEY
        or value.get("global_modal_gpu_limit") != 1
        or not isinstance(value.get("owner"), str)
        or not isinstance(value.get("workload"), str)
        or not isinstance(value.get("claim_id"), str)
        or not isinstance(value.get("submission_preflight"), dict)
        or value.get("submission_preflight_sha256")
        != _json_sha256(value.get("submission_preflight"))
    ):
        raise RuntimeError("global GPU lease record is invalid")
    if value["owner"] != f"{value['workload']}:{value['claim_id']}":
        raise RuntimeError("global GPU lease owner is invalid")
    return value


def _dispatch_stem(arm: str) -> str:
    stem = f"training-{arm}_seed{SEED}"
    if re.fullmatch(r"[a-z0-9_-]+", stem) is None:
        raise RuntimeError("unsafe tournament GPU dispatch name")
    return stem


def _intent_path(token: dict[str, Any]) -> Path:
    nonce = token.get("nonce")
    if not isinstance(nonce, str) or re.fullmatch(r"[0-9a-f]{32}", nonce) is None:
        raise RuntimeError("GPU token nonce is invalid")
    return GPU_DISPATCH_DIR / f"{_dispatch_stem(token['arm'])}-{nonce}.json"


def _completion_path(token: dict[str, Any]) -> Path:
    return _intent_path(token).with_suffix(".complete.json")


def _acquire_gpu_lease(claim_id: str, root_call_id: str, arm: str) -> dict[str, Any]:
    claim = _attempt_claim(claim_id)
    receipt = json.loads((REMOTE_STATE / "launch_receipt.json").read_text())
    if (
        receipt.get("receipt_status") != "present"
        or receipt.get("claim_id") != claim_id
        or receipt.get("function_call_id") != root_call_id
    ):
        raise RuntimeError("GPU dispatch lacks its durable root receipt")
    preflight = _validate_operational_preflight(claim["operational_preflight"])
    now = datetime.now(timezone.utc).isoformat()
    token = {
        "protocol": GPU_LEASE_PROTOCOL,
        "environment_name": GPU_LEASE_ENVIRONMENT,
        "slot": GPU_LEASE_KEY,
        "owner": f"{GPU_LEASE_WORKLOAD}:{claim_id}",
        "workload": GPU_LEASE_WORKLOAD,
        "claim_id": claim_id,
        "global_modal_gpu_limit": 1,
        "submission_preflight": preflight,
        "submission_preflight_sha256": _json_sha256(preflight),
        "acquired_at_utc": now,
        "heartbeat_at_utc": now,
        "nonce": uuid.uuid4().hex,
        "modal_app": APP_NAME,
        "volume": VOLUME_NAME,
        "root_call_id": root_call_id,
        "phase": "training",
        "arm": arm,
    }
    if gpu_lease.put(GPU_LEASE_KEY, token, skip_if_exists=True) is not True:
        existing = gpu_lease.get(GPU_LEASE_KEY, None)
        if existing is not None:
            _validate_lease_record(existing)
        raise RuntimeError(f"global GPU lease is occupied: {existing!r}")
    if gpu_lease.get(GPU_LEASE_KEY, None) != token:
        raise RuntimeError("global GPU lease ownership is ambiguous")
    intent = {
        "schema_version": 1,
        "protocol": "j-lens-rl-development-tournament-gpu-intent-v1",
        "lease": token,
        "lease_sha256": _json_sha256(token),
        "dispatch_status": "leased_before_gpu_schedule",
    }
    _write_json_exclusive_atomic(_intent_path(token), intent)
    state_volume.commit()
    state_volume.reload()
    if json.loads(_intent_path(token).read_text()) != intent:
        raise RuntimeError("durable GPU intent changed after publication")
    return token


def _verify_gpu_token(token: dict[str, Any], arm: str) -> None:
    observed = _validate_lease_record(gpu_lease.get(GPU_LEASE_KEY, None))
    claim_id = json.loads((REMOTE_STATE / "attempt_claim.json").read_text())["claim_id"]
    if (
        observed != token
        or token.get("arm") != arm
        or token.get("phase") != "training"
        or token.get("claim_id") != claim_id
    ):
        raise RuntimeError("GPU worker received a stale or invalid lease token")
    receipt = json.loads((REMOTE_STATE / "launch_receipt.json").read_text())
    if (
        receipt.get("claim_id") != claim_id
        or receipt.get("function_call_id") != token.get("root_call_id")
    ):
        raise RuntimeError("GPU token is not bound to the root receipt")
    intent = json.loads(_intent_path(token).read_text())
    if (
        intent.get("lease") != token
        or intent.get("lease_sha256") != _json_sha256(token)
        or intent.get("dispatch_status") != "leased_before_gpu_schedule"
    ):
        raise RuntimeError("GPU worker lacks its durable dispatch intent")


def _complete_gpu_dispatch(token: dict[str, Any], publication: dict[str, str]) -> None:
    _verify_gpu_token(token, token["arm"])
    if not publication or any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in publication.values()):
        raise RuntimeError("GPU result publication identity is incomplete")
    completion = {
        "schema_version": 1,
        "protocol": "j-lens-rl-development-tournament-gpu-completion-v1",
        "nonce": token["nonce"],
        "lease_sha256": _json_sha256(token),
        "publication_sha256": publication,
        "published_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = _completion_path(token)
    _write_json_exclusive_atomic(path, completion)
    state_volume.commit()
    state_volume.reload()
    if json.loads(path.read_text()) != completion:
        raise RuntimeError("durable GPU completion changed after publication")


def _release_gpu_lease(token: dict[str, Any]) -> None:
    observed = _validate_lease_record(gpu_lease.get(GPU_LEASE_KEY, None))
    if observed != token or observed.get("nonce") != token.get("nonce"):
        raise RuntimeError("refusing to release another dispatch's GPU lease")
    removed = gpu_lease.pop(GPU_LEASE_KEY)
    if removed != token:
        raise RuntimeError("GPU lease changed during release")


def _wait_for_launch_receipt(claim_id: str) -> dict[str, Any]:
    path = REMOTE_STATE / "launch_receipt.json"
    deadline = time.monotonic() + 10 * 60
    while time.monotonic() < deadline:
        state_volume.reload()
        if path.is_file():
            receipt = json.loads(path.read_text())
            if (
                receipt.get("claim_id") == claim_id
                and receipt.get("receipt_status") == "present"
            ):
                return receipt
            raise RuntimeError("launch receipt is malformed or closed")
        time.sleep(2)
    closure = {
        "schema_version": 1,
        "protocol": protocol.PROTOCOL,
        "claim_id": claim_id,
        "receipt_status": "absent_closed_before_dispatch",
        "closed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        _write_json_exclusive_atomic(path, closure)
        state_volume.commit()
    except FileExistsError:
        state_volume.reload()
    raise RuntimeError("timed out waiting for immutable launch receipt")


@app.function(
    image=repo_image,
    cpu=2,
    memory=4096,
    max_containers=1,
    timeout=10 * 60,
    startup_timeout=60 * 60,
    retries=0,
    volumes={REMOTE_STATE: state_volume},
)
def claim_attempt(claim_id: str, operational_preflight: dict[str, Any]) -> dict[str, Any]:
    state_volume.reload()
    _ensure_runtime_git()
    _protocol("verify-launch")
    operational_preflight = _validate_operational_preflight(operational_preflight)
    forbidden = [
        REMOTE_STATE / "attempt_claim.json",
        REMOTE_STATE / "attempt_status.json",
        REMOTE_STATE / "launch_receipt.json",
        REMOTE_STATE / "runs",
        REMOTE_STATE / "evidence",
        REMOTE_STATE / "exports",
        REMOTE_STATE / "gpu_dispatches",
    ]
    stale = [str(path) for path in forbidden if path.exists()]
    if stale:
        raise RuntimeError(f"fresh tournament state already has attempt data: {stale}")
    state = json.loads((REMOTE_STATE / "protocol_state.json").read_text())
    claim = {
        "schema_version": 1,
        "protocol": protocol.PROTOCOL,
        "scientific_status": "development_only",
        "claim_id": claim_id,
        "git_commit": state["git_commit"],
        "registration_sha256": state["registration_sha256"],
        "recipe_lock_sha256": state["recipe_lock_sha256"],
        "prelaunch_amendment_sha256": state["prelaunch_amendment_sha256"],
        "arm_order": list(ARM_ORDER),
        "seed": SEED,
        "curve_steps": list(CURVE_STEPS),
        "global_modal_gpu_limit": 1,
        "gpu_lease_dict": GPU_LEASE_DICT_NAME,
        "gpu_lease_key": GPU_LEASE_KEY,
        "volume_object_id": operational_preflight["volume_object_id"],
        "operational_preflight": operational_preflight,
    }
    _write_json_exclusive_atomic(REMOTE_STATE / "attempt_claim.json", claim)
    _write_status(claim_id, "claimed")
    state_volume.commit()
    return claim


@app.function(
    image=repo_image,
    cpu=1,
    memory=2048,
    max_containers=1,
    timeout=10 * 60,
    startup_timeout=60 * 60,
    retries=0,
    volumes={REMOTE_STATE: state_volume},
)
def record_launch_receipt(
    claim_id: str, app_id: str, function_call_id: str
) -> dict[str, Any]:
    state_volume.reload()
    claim = _attempt_claim(claim_id)
    status = json.loads((REMOTE_STATE / "attempt_status.json").read_text())
    if status.get("stage") != "claimed":
        raise RuntimeError("launch receipt is only valid at claimed stage")
    receipt = {
        "schema_version": 1,
        "protocol": protocol.PROTOCOL,
        "scientific_status": "development_only",
        "claim_id": claim_id,
        "receipt_status": "present",
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "modal_app": APP_NAME,
        "app_id": app_id,
        "function_call_id": function_call_id,
        "volume": VOLUME_NAME,
        "volume_object_id": claim["volume_object_id"],
        "gpu_type": GPU_TYPE,
        "max_parallel_gpu_workers": 1,
        "arm_order": list(ARM_ORDER),
        "wandb_group": "dev-v8-emotional-single-u5-h15-seed192",
    }
    _write_json_exclusive_atomic(REMOTE_STATE / "launch_receipt.json", receipt)
    state_volume.commit()
    return receipt


@app.function(
    image=repo_image,
    gpu=GPU_TYPE,
    cpu=4,
    memory=32768,
    max_containers=MAX_GPU_CONTAINERS,
    timeout=4 * 60 * 60,
    startup_timeout=60 * 60,
    retries=0,
    single_use_containers=True,
    secrets=[wandb_secret],
    volumes={REMOTE_STATE: state_volume},
)
def train_arm(arm: str, lease_token: dict[str, Any]) -> dict[str, Any]:
    if arm not in ARM_ORDER:
        raise ValueError("training arm is outside the registered tournament")
    state_volume.reload()
    _verify_gpu_token(lease_token, arm)
    try:
        _ensure_runtime_git()
        _protocol("verify")
        config = REMOTE_STATE / "configs" / f"{arm}_seed192.json"
        run_dir = REMOTE_STATE / "runs" / f"{arm}_seed192"
        result_path = run_dir / "run_result_manifest.json"
        receipt_path = run_dir / "wandb_terminal_publish_receipt.json"
        if result_path.is_file():
            _run(
                [
                    sys.executable,
                    "-m",
                    "jlens_rl.train",
                    "--config",
                    str(config),
                    "--publish-existing-result",
                ]
            )
            reused = True
        else:
            try:
                _run(
                    [
                        sys.executable,
                        "-m",
                        "jlens_rl.train",
                        "--config",
                        str(config),
                        "--wandb-mode",
                        "online",
                    ]
                )
            except subprocess.CalledProcessError:
                if not result_path.is_file():
                    raise
                if not receipt_path.is_file():
                    _run(
                        [
                            sys.executable,
                            "-m",
                            "jlens_rl.train",
                            "--config",
                            str(config),
                            "--publish-existing-result",
                        ]
                    )
            reused = False
        if not result_path.is_file() or not receipt_path.is_file():
            raise RuntimeError("completed arm lacks terminal local/W&B evidence")
        rows = [
            json.loads(line)
            for line in (run_dir / "validation_history.jsonl").read_text().splitlines()
            if line
        ]
        if [row.get("step") for row in rows] != list(CURVE_STEPS):
            raise RuntimeError("completed arm has the wrong curve nodes")
        state_volume.commit()
        _complete_gpu_dispatch(
            lease_token,
            {
                "run_result_manifest": _sha256(result_path),
                "wandb_terminal_publish_receipt": _sha256(receipt_path),
                "validation_history": _sha256(run_dir / "validation_history.jsonl"),
            },
        )
        _release_gpu_lease(lease_token)
        return {
            "arm": arm,
            "seed": SEED,
            "steps": [row["step"] for row in rows],
            "exact_match": [row["exact_match"] for row in rows],
            "literal_target_completion_rate": [
                row["literal_target_completion_rate"] for row in rows
            ],
            "reused_existing_terminal_result": reused,
        }
    except BaseException:
        # A partial or publication-ambiguous job keeps the global lease stranded.
        state_volume.commit()
        raise


@app.function(
    image=repo_image,
    cpu=2,
    memory=4096,
    max_containers=1,
    timeout=2 * 60 * 60,
    startup_timeout=60 * 60,
    retries=0,
    volumes={REMOTE_STATE: state_volume},
)
def protocol_step(command: str) -> dict[str, Any]:
    if command not in {"summarize", "finalize-evidence"}:
        raise ValueError("unsupported tournament protocol step")
    state_volume.reload()
    try:
        rendered = _protocol(command)
        if command == "summarize":
            return json.loads(
                (REMOTE_STATE / "evidence" / "tournament_summary.json").read_text()
            )
        return {
            "evidence_inventory_sha256": _sha256(
                REMOTE_STATE / "evidence" / "evidence_inventory.json"
            ),
            "durable_export_receipt_sha256": _sha256(
                REMOTE_STATE / "evidence" / "durable_export_receipt.json"
            ),
            "git_closeout_candidate_sha256": _sha256(
                REMOTE_STATE / "evidence" / "git_closeout_candidate.json"
            ),
            "returncode": rendered.returncode,
        }
    finally:
        state_volume.commit()


@app.function(
    image=repo_image,
    cpu=1,
    memory=2048,
    max_containers=1,
    timeout=16 * 60 * 60,
    startup_timeout=60 * 60,
    retries=0,
    volumes={REMOTE_STATE: state_volume},
)
def orchestrate(claim_id: str) -> dict[str, Any]:
    state_volume.reload()
    _attempt_claim(claim_id)
    failure_phase = "launch_receipt_wait"
    current_arm: str | None = None
    try:
        receipt = _wait_for_launch_receipt(claim_id)
        root_call_id = modal.current_function_call_id()
        if (
            not isinstance(root_call_id, str)
            or not root_call_id
            or receipt.get("function_call_id") != root_call_id
        ):
            raise RuntimeError("orchestrator lacks durable root authority")
        failure_phase = "initial_verify"
        _ensure_runtime_git()
        _protocol("verify-launch")
        results = []
        for index, arm in enumerate(ARM_ORDER, 1):
            current_arm = arm
            failure_phase = f"training_{arm}"
            state_volume.reload()
            _write_status(
                claim_id,
                "training",
                current_arm=arm,
                serial_position=index,
                serial_total=len(ARM_ORDER),
                completed_arms=[item["arm"] for item in results],
                max_parallel_gpu_workers=1,
            )
            state_volume.commit()
            token = _acquire_gpu_lease(claim_id, root_call_id, arm)
            results.append(train_arm.remote(arm, token))
        failure_phase = "development_summary"
        state_volume.reload()
        summary = protocol_step.remote("summarize")
        state_volume.reload()
        _write_status(
            claim_id,
            "complete",
            completed_arms=list(ARM_ORDER),
            selected_development_candidate=summary[
                "selected_development_candidate"
            ],
            selected_candidate_shape_passed=summary[
                "selected_candidate_shape_passed"
            ],
            no_significance_or_final_claim=True,
        )
        state_volume.commit()
        failure_phase = "durable_evidence_export"
        durable = protocol_step.remote("finalize-evidence")
        return {
            "stage": "complete",
            "scientific_status": "development_only",
            "runs": results,
            "summary": summary,
            "durable_evidence": durable,
        }
    except BaseException as error:
        try:
            state_volume.reload()
            _write_status(
                claim_id,
                "failed",
                failure_phase=failure_phase,
                current_arm=current_arm,
                error=repr(error),
                retry_policy=(
                    "Close this whole attempt immutably. Never resume partial "
                    "optimization or mix backends; a retry needs a fresh registered "
                    "attempt, claim, Volume, and all three arms."
                ),
            )
            state_volume.commit()
        except BaseException:
            pass
        raise


def _local_operational_preflight() -> dict[str, Any]:
    confirmation = os.environ.get("JLENS_MODAL_GPU_EXCLUSIVE_CONFIRM")
    if confirmation != GPU_EXCLUSIVE_CONFIRMATION:
        raise RuntimeError(
            "refusing tournament launch without explicit no-overlap confirmation"
        )
    local_verify = subprocess.run(
        [
            sys.executable,
            "scripts/emotional_tournament_v1_protocol.py",
            "verify-launch",
        ],
        cwd=LOCAL_REPO,
        check=False,
        text=True,
    )
    if local_verify.returncode:
        raise RuntimeError("local tournament state is not launch-enabled")
    volume_object_id = verify_tournament_v1_volume_v2()
    modal_cli = Path(sys.executable).parent / "modal"
    listing_text = subprocess.check_output(
        [str(modal_cli), "app", "list", "--json"], cwd=LOCAL_REPO, text=True
    )
    listing = json.loads(listing_text[listing_text.index("[") :])
    inventory_text = subprocess.check_output(
        [str(modal_cli), "volume", "ls", VOLUME_NAME, "/", "--json"],
        cwd=LOCAL_REPO,
        text=True,
    )
    inventory = json.loads(inventory_text[inventory_text.index("[") :])
    if inventory != []:
        raise RuntimeError(f"registered fresh Volume is not empty: {inventory}")
    current_app_id = app.app_id
    active_other_apps = [
        {
            key: item.get(key)
            for key in ("app_id", "description", "state", "tasks", "created_at")
        }
        for item in listing
        if item.get("stopped_at") is None
        and item.get("state") != "stopped"
        and item.get("app_id") != current_app_id
    ]
    if active_other_apps:
        raise RuntimeError(
            f"refusing launch while another Modal app is active: {active_other_apps}"
        )
    v7_active = any(
        item.get("app_id") == protocol.V7_APP_ID
        and item.get("stopped_at") is None
        and item.get("state") != "stopped"
        for item in listing
    )
    if v7_active:
        raise RuntimeError("V7 app remains active")
    gpu_lease.hydrate()
    lease_owner = gpu_lease.get(GPU_LEASE_KEY, None)
    if lease_owner is not None:
        raise RuntimeError(f"global GPU lease is occupied: {lease_owner!r}")
    return _validate_operational_preflight(
        {
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "exclusive_gpu_confirmation": confirmation,
            "global_modal_gpu_limit": 1,
            "active_other_modal_apps": active_other_apps,
            "v7_app_id": protocol.V7_APP_ID,
            "v7_app_observed_active": v7_active,
            "volume_name": VOLUME_NAME,
            "volume_object_id": volume_object_id,
            "volume_version": 2,
            "gpu_lease_dict": GPU_LEASE_DICT_NAME,
            "gpu_lease_key": GPU_LEASE_KEY,
            "gpu_lease_owner_at_preflight": lease_owner,
        }
    )


def _upload_protocol_state() -> None:
    protocol.verify(require_launch=True)
    allowed_roots = {
        "configs",
        "frozen_artifacts",
        "manifests",
        "reproducibility",
        "protocol_state.json",
    }
    files = sorted(path for path in LOCAL_STATE.rglob("*") if path.is_file())
    if not files or any(
        path.is_symlink()
        or path.relative_to(LOCAL_STATE).parts[0] not in allowed_roots
        for path in files
    ):
        raise RuntimeError("prepared tournament upload inventory is unsafe")
    forbidden_names = {
        "sealed_final_indices.json",
        "future_reserve_indices.json",
        "retired_v1_curve_indices.json",
        "retired_v2_curve_indices.json",
        "retired_v3_curve_indices.json",
    }
    if any(path.name in forbidden_names or "correlation" in path.name for path in files):
        raise RuntimeError("prepared state contains a forbidden outcome payload")
    with state_volume.batch_upload(force=False) as batch:
        for path in files:
            relative = path.relative_to(LOCAL_STATE)
            batch.put_file(path, f"/{relative.as_posix()}")


@app.local_entrypoint()
def main() -> None:
    operational_preflight = _local_operational_preflight()
    _upload_protocol_state()
    claim_id = uuid.uuid4().hex
    claim_attempt.remote(claim_id, operational_preflight)
    call = orchestrate.spawn(claim_id)
    receipt = record_launch_receipt.remote(
        claim_id, app.app_id or APP_NAME, call.object_id
    )
    print(
        json.dumps(
            {
                "status": "submitted",
                "scientific_status": "development_only",
                "app_id": app.app_id,
                "function_call_id": call.object_id,
                "claim_id": claim_id,
                "volume": VOLUME_NAME,
                "gpu_type": GPU_TYPE,
                "max_parallel_gpus": 1,
                "arm_order": list(ARM_ORDER),
                "seed": SEED,
                "curve_steps": list(CURVE_STEPS),
                "operational_preflight": operational_preflight,
                "launch_receipt": receipt,
                "no_significance_or_final_claim": True,
            },
            indent=2,
        )
    )
