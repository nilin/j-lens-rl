"""Run the conditional profanity-u5 V7 confirmation on Modal.

This runner is intentionally unusable until the local V7 protocol has been
prepared from a committed final registration.  It caps each GPU phase at
one L40S worker, gates controls on the registered treatment curve, and
creates exactly one immutable 17-label sealed-final collection.
"""

from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import modal

from scripts.modal_verify_v7_volume import verify_v7_volume_v2


LOCAL_REPO = Path(__file__).resolve().parent
LOCAL_STATE = LOCAL_REPO / ".confirmatory" / "v7"
REMOTE_REPO = Path("/workspace/j-lens-rl")
REMOTE_STATE = REMOTE_REPO / ".confirmatory" / "v7"
VOLUME_NAME = "j-lens-rl-confirmatory-v7-profanity-u5-20260714a"
SEEDS = tuple(range(184, 192))
MAX_GPU_CONTAINERS = 1
GPU_TYPE = "L40S"
GLOBAL_MODAL_GPU_LIMIT = 1
GPU_APP_OVERLAP_POLICY = "no other Modal GPU app may overlap this V7 attempt"
GPU_EXCLUSIVE_CONFIRMATION = "confirmed-no-other-modal-gpu-app-running"
GPU_LEASE_DICT_NAME = "j-lens-rl-global-gpu-lease-v1"
GPU_LEASE_KEY = "global-one-gpu"
GPU_LEASE_ENVIRONMENT = "main"
FINAL_LABELS = (
    "base",
    *(f"jlens_seed{seed}" for seed in SEEDS),
    *(f"signflip_seed{seed}" for seed in SEEDS),
)

app = modal.App("j-lens-rl-confirmatory-v7-profanity-u5")
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

repo_image = modal.Image.debian_slim(python_version="3.11")
repo_image = repo_image.add_local_dir(
    LOCAL_REPO / "src" / "jlens_rl",
    (REMOTE_REPO / "src" / "jlens_rl").as_posix(),
    copy=True,
)
repo_image = repo_image.add_local_dir(
    LOCAL_REPO / "trl" / "trl",
    (REMOTE_REPO / "trl" / "trl").as_posix(),
    copy=True,
)
for relative in (
    "pyproject.toml",
    "modal_confirmatory_v7.py",
    "run_confirmatory_v7.sh",
    "scripts/confirmatory_v7_protocol.py",
    "scripts/modal_cache_assets_v7.py",
    "scripts/modal_finalize_image_v7.py",
    "scripts/modal_verify_v7_volume.py",
    "artifacts/qwen25_05b_solved_lens.pt",
    "trl/pyproject.toml",
    "trl/MANIFEST.in",
    "trl/VERSION",
    "trl/README.md",
    "trl/LICENSE",
    "trl/CONTRIBUTING.md",
):
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
                "j-lens-rl-confirmatory-v7-profanity-u5-image-v1-strict-allowlist-hf-auth"
            ),
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
        "python scripts/modal_cache_assets_v7.py",
        secrets=[huggingface_secret],
    )
    .run_commands("python scripts/modal_finalize_image_v7.py")
)


def _run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    rendered = subprocess.run(command, cwd=REMOTE_REPO, check=False, text=True)
    if check and rendered.returncode:
        raise subprocess.CalledProcessError(rendered.returncode, command)
    return rendered


def _ensure_runtime_git() -> str:
    """Recreate the registered parentless Git identity outside the baked image."""
    manifest_path = REMOTE_STATE / "reproducibility" / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    files = manifest.get("files")
    expected_commit = manifest.get("git_commit")
    recipe = manifest.get("runtime_commit_recipe")
    if (
        not isinstance(files, dict)
        or not isinstance(expected_commit, str)
        or recipe
        != {
            "author": "J-Lens V7 Runtime <runtime@example.invalid>",
            "timestamp": "2000-01-01T00:00:00+00:00",
            "message": "J-Lens V7 strict runtime source",
            "parent": None,
        }
    ):
        raise RuntimeError("V7 runtime Git recipe is absent or changed")
    for name, identity in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not isinstance(identity, dict):
            raise RuntimeError("V7 runtime Git manifest contains an unsafe path")
        path = REMOTE_REPO / relative
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != identity.get("sha256")
            or path.stat().st_size != identity.get("size_bytes")
        ):
            raise RuntimeError(f"V7 runtime source changed before Git setup: {name}")
        path.chmod(int(identity["mode"]))

    git_dir = REMOTE_REPO / ".git"
    if not git_dir.exists():
        environment = {
            **os.environ,
            "GIT_AUTHOR_NAME": "J-Lens V7 Runtime",
            "GIT_AUTHOR_EMAIL": "runtime@example.invalid",
            "GIT_COMMITTER_NAME": "J-Lens V7 Runtime",
            "GIT_COMMITTER_EMAIL": "runtime@example.invalid",
            "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+00:00",
            "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+00:00",
        }
        subprocess.run(
            ["git", "init", "-q"], cwd=REMOTE_REPO, check=True, env=environment
        )
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
            input="J-Lens V7 strict runtime source\n",
            text=True,
            env=environment,
        ).strip()
        if actual_commit != expected_commit:
            raise RuntimeError(
                "reconstructed V7 runtime Git identity differs from registration"
            )
        subprocess.run(
            ["git", "update-ref", "HEAD", actual_commit],
            cwd=REMOTE_REPO,
            check=True,
            env=environment,
        )
        (git_dir / "info" / "exclude").write_text("*\n")
    actual_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REMOTE_REPO, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=REMOTE_REPO,
        text=True,
    )
    if actual_commit != expected_commit or status:
        raise RuntimeError("V7 reconstructed runtime Git worktree is not exact and clean")
    return actual_commit


def _claim_id_from_volume() -> str:
    path = REMOTE_STATE / "attempt_claim.json"
    if not path.is_file():
        raise RuntimeError("GPU work is forbidden before an immutable V7 claim")
    claim_id = json.loads(path.read_text()).get("claim_id")
    if not isinstance(claim_id, str) or not claim_id:
        raise RuntimeError("V7 attempt claim has no stable claim ID")
    return claim_id


def _acquire_gpu_lease(phase: str, subject: str) -> dict[str, Any]:
    """Atomically claim the sole global GPU key; never steal or time out."""
    payload = {
        "protocol": "j-lens-rl-global-gpu-lease-v1",
        "key": GPU_LEASE_KEY,
        "nonce": uuid.uuid4().hex,
        "claim_id": _claim_id_from_volume(),
        "modal_app": app.name,
        "volume": VOLUME_NAME,
        "phase": phase,
        "subject": subject,
        "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    inserted = gpu_lease.put(GPU_LEASE_KEY, payload, skip_if_exists=True)
    if inserted is not True:
        owner = gpu_lease.get(GPU_LEASE_KEY, None)
        raise RuntimeError(
            "global GPU lease is occupied; V7 fails closed without dispatch: "
            f"{owner!r}"
        )
    observed = gpu_lease.get(GPU_LEASE_KEY, None)
    if observed != payload:
        raise RuntimeError("global GPU lease ownership is ambiguous after acquisition")
    return payload


def _release_gpu_lease(payload: dict[str, Any]) -> None:
    """Release only the exact nonce we acquired; ambiguous state stays locked."""
    observed = gpu_lease.get(GPU_LEASE_KEY, None)
    if observed != payload or observed.get("nonce") != payload.get("nonce"):
        raise RuntimeError("refusing to release a GPU lease owned by another nonce")
    removed = gpu_lease.pop(GPU_LEASE_KEY)
    if removed != payload:
        raise RuntimeError("GPU lease changed during nonce-owned release")


def _write_json_exclusive_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Publish a complete immutable marker without an overwrite race."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # A hard-link publish is atomic and fails rather than replacing an
        # immutable marker created by another container.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _launch_receipt_is_present() -> bool:
    path = REMOTE_STATE / "launch_receipt.json"
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text()).get("receipt_status") == "present"
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False


def _protocol(
    command: str, *extra: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return _run(
        [sys.executable, "scripts/confirmatory_v7_protocol.py", command, *extra],
        check=check,
    )


def _history_summary(condition: str, seed: int) -> dict[str, Any]:
    path = REMOTE_STATE / "runs" / f"{condition}_seed{seed}" / "validation_history.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    return {
        "condition": condition,
        "seed": seed,
        "steps": [row["step"] for row in rows],
        "exact_match": [row["exact_match"] for row in rows],
        "literal_target_completion_rate": [
            row["literal_target_completion_rate"] for row in rows
        ],
    }


def _verify_attempt_claim(claim_id: str) -> dict[str, Any]:
    path = REMOTE_STATE / "attempt_claim.json"
    if not path.is_file():
        raise RuntimeError("V7 volume has no attempt claim")
    claim = json.loads(path.read_text())
    if claim.get("claim_id") != claim_id:
        raise RuntimeError("V7 volume is claimed by another launch")
    return claim


def _wait_for_launch_receipt(claim_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 10 * 60
    path = REMOTE_STATE / "launch_receipt.json"
    while time.monotonic() < deadline:
        state_volume.reload()
        if path.is_file():
            decision = json.loads(path.read_text())
            if decision.get("claim_id") != claim_id:
                raise RuntimeError("V7 launch receipt belongs to another claim")
            if decision.get("receipt_status") == "present":
                return decision
            if decision.get("receipt_status") == "absent_closed_before_dispatch":
                raise RuntimeError(
                    "Modal launch receipt was atomically closed absent before dispatch"
                )
            raise RuntimeError("V7 launch receipt marker is malformed")
        time.sleep(2)
    closure = {
        "claim_id": claim_id,
        "receipt_status": "absent_closed_before_dispatch",
        "closed_at_utc": datetime.now(timezone.utc).isoformat(),
        "reason": "timed out waiting for the immutable Modal launch receipt",
        "modal_app": app.name,
        "volume": VOLUME_NAME,
    }
    closure_created = False
    try:
        _write_json_exclusive_atomic(path, closure)
        closure_created = True
    except FileExistsError:
        pass
    if closure_created:
        # The exception handler reloads the Volume before recording failure;
        # persist the atomic closure first so that reload cannot discard it.
        state_volume.commit()
    else:
        state_volume.reload()
    decision = json.loads(path.read_text())
    if (
        decision.get("claim_id") == claim_id
        and decision.get("receipt_status") == "present"
    ):
        return decision
    raise RuntimeError("timed out waiting for the immutable Modal launch receipt")


def _record_attempt_status(claim_id: str, stage: str, **details: Any) -> dict[str, Any]:
    _verify_attempt_claim(claim_id)
    status = {
        "claim_id": claim_id,
        "stage": stage,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        **details,
    }
    path = REMOTE_STATE / "attempt_status.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return status


def _validate_operational_preflight(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError("V7 launch lacks its global-GPU operational preflight")
    if value.get("exclusive_gpu_confirmation") != GPU_EXCLUSIVE_CONFIRMATION:
        raise RuntimeError("V7 launch was not confirmed exclusive of other GPU apps")
    if value.get("global_modal_gpu_limit") != GLOBAL_MODAL_GPU_LIMIT:
        raise RuntimeError("V7 launch changed the global Modal GPU limit")
    if value.get("active_other_modal_apps") != []:
        raise RuntimeError("another Modal app remains active at V7 launch")
    if value.get("volume_name") != VOLUME_NAME:
        raise RuntimeError("V7 launch changed the registered fresh Volume name")
    if value.get("volume_version") != 2:
        raise RuntimeError("V7 launch did not verify the fresh Volume as version 2")
    if not isinstance(value.get("volume_object_id"), str):
        raise RuntimeError("V7 launch lacks the hydrated Volume object identity")
    if value.get("gpu_lease_dict") != GPU_LEASE_DICT_NAME:
        raise RuntimeError("V7 launch changed the global GPU lease Dict")
    if value.get("gpu_lease_key") != GPU_LEASE_KEY:
        raise RuntimeError("V7 launch changed the global GPU lease key")
    if value.get("gpu_lease_owner_at_preflight") is not None:
        raise RuntimeError("global GPU lease was occupied at V7 preflight")
    return value


@app.function(
    image=repo_image,
    cpu=2,
    memory=4096,
    timeout=10 * 60,
    startup_timeout=60 * 60,
    retries=0,
    volumes={REMOTE_STATE: state_volume},
)
def claim_attempt(claim_id: str, operational_preflight: dict[str, Any]) -> dict[str, Any]:
    state_volume.reload()
    _protocol("verify")
    operational_preflight = _validate_operational_preflight(operational_preflight)
    forbidden = [
        REMOTE_STATE / "runs",
        REMOTE_STATE / "evals",
        REMOTE_STATE / "evidence",
        REMOTE_STATE / "final_unlocked.json",
        REMOTE_STATE / "final_collection.json",
        REMOTE_STATE / "attempt_status.json",
        REMOTE_STATE / "launch_receipt.json",
    ]
    stale = [str(path) for path in forbidden if path.exists()]
    if stale:
        raise RuntimeError(f"V7 volume already contains attempt data: {stale}")
    claim_path = REMOTE_STATE / "attempt_claim.json"
    state = json.loads((REMOTE_STATE / "protocol_state.json").read_text())
    claim = {
        "claim_id": claim_id,
        "git_commit": state["git_commit"],
        "protocol": state["protocol"],
        "registration_sha256": state["registration_sha256"],
        "recipe_lock_sha256": state["recipe_lock_sha256"],
        "global_modal_gpu_limit": GLOBAL_MODAL_GPU_LIMIT,
        "gpu_app_overlap_policy": GPU_APP_OVERLAP_POLICY,
        "gpu_lease_dict": GPU_LEASE_DICT_NAME,
        "gpu_lease_key": GPU_LEASE_KEY,
        "volume_object_id": operational_preflight["volume_object_id"],
        "operational_preflight": operational_preflight,
    }
    try:
        _write_json_exclusive_atomic(claim_path, claim)
    except FileExistsError as error:
        raise RuntimeError("V7 volume is already claimed") from error
    _record_attempt_status(claim_id, "claimed")
    state_volume.commit()
    return claim


@app.function(
    image=repo_image,
    cpu=1,
    memory=2048,
    timeout=10 * 60,
    startup_timeout=60 * 60,
    retries=0,
    volumes={REMOTE_STATE: state_volume},
)
def record_launch_receipt(
    claim_id: str, app_id: str, function_call_id: str
) -> dict[str, Any]:
    state_volume.reload()
    _verify_attempt_claim(claim_id)
    status_path = REMOTE_STATE / "attempt_status.json"
    if (
        not status_path.is_file()
        or json.loads(status_path.read_text()).get("stage") != "claimed"
    ):
        raise RuntimeError(
            "launch receipt may only be recorded while the attempt is claimed"
        )
    late_markers = (
        REMOTE_STATE / "final_unlocked.json",
        REMOTE_STATE / "final_collection.json",
        REMOTE_STATE / "evidence" / "git_closeout_candidate.json",
        REMOTE_STATE / "evidence" / "evidence_bundle_inventory.json",
        REMOTE_STATE / "evidence" / "durable_export_plan.json",
        REMOTE_STATE / "exports",
    )
    if any(marker.exists() for marker in late_markers):
        raise RuntimeError(
            "launch receipt cannot mutate a terminal or exported attempt"
        )
    receipt = {
        "claim_id": claim_id,
        "receipt_status": "present",
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "modal_app": app.name,
        "app_id": app_id,
        "function_call_id": function_call_id,
        "volume": VOLUME_NAME,
        "gpu_type": GPU_TYPE,
        "max_parallel_gpu_workers": MAX_GPU_CONTAINERS,
        "global_modal_gpu_limit": GLOBAL_MODAL_GPU_LIMIT,
        "gpu_app_overlap_policy": GPU_APP_OVERLAP_POLICY,
        "gpu_lease_dict": GPU_LEASE_DICT_NAME,
        "gpu_lease_key": GPU_LEASE_KEY,
        "volume_object_id": json.loads(
            (REMOTE_STATE / "attempt_claim.json").read_text()
        )["volume_object_id"],
    }
    path = REMOTE_STATE / "launch_receipt.json"
    try:
        _write_json_exclusive_atomic(path, receipt)
    except FileExistsError as error:
        existing = json.loads(path.read_text())
        if existing == receipt:
            return existing
        if existing.get("receipt_status") == "absent_closed_before_dispatch":
            raise RuntimeError(
                "launch receipt was already closed absent after timeout"
            ) from error
        raise RuntimeError("V7 launch receipt already exists") from error
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
def train_config(condition: str, seed: int) -> dict[str, Any]:
    if condition not in {"jlens", "signflip"} or seed not in SEEDS:
        raise ValueError("training input is outside registered V7")
    state_volume.reload()
    lease = _acquire_gpu_lease("training", f"{condition}_seed{seed}")
    try:
        _ensure_runtime_git()
        _protocol("verify")
        if condition == "signflip":
            _protocol("verify-curve")
        config = REMOTE_STATE / "configs" / f"{condition}_seed{seed}.json"
        run_output = REMOTE_STATE / "runs" / f"{condition}_seed{seed}"
        completed = run_output / "run_result_manifest.json"
        wandb_receipt = run_output / "wandb_terminal_publish_receipt.json"
        if completed.is_file():
            # This validates a pre-existing receipt and is a cheap no-op when
            # complete. A truncated/stale receipt is republished without ever
            # resuming optimization.
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
            if not wandb_receipt.is_file():
                raise RuntimeError("terminal W&B publication produced no receipt")
            summary = _history_summary(condition, seed)
            summary["reused_existing_terminal_result"] = True
            return summary
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
            if not completed.is_file():
                raise
            if not wandb_receipt.is_file():
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
        summary = _history_summary(condition, seed)
        summary["reused_existing_terminal_result"] = False
        return summary
    finally:
        # A failed commit intentionally strands the lease for root-authorized
        # forensic recovery; no later GPU job may guess that publication worked.
        state_volume.commit()
        _release_gpu_lease(lease)


@app.function(
    image=repo_image,
    cpu=2,
    memory=4096,
    timeout=2 * 60 * 60,
    startup_timeout=60 * 60,
    retries=0,
    volumes={REMOTE_STATE: state_volume},
)
def protocol_step(command: str) -> dict[str, Any]:
    if command not in {"verify-semantic", "curve", "unlock"}:
        raise ValueError("unsupported V7 protocol step")
    state_volume.reload()
    result = _protocol(command, check=False)
    state_volume.commit()
    if command == "verify-semantic":
        return {"verified": result.returncode == 0, "returncode": result.returncode}
    path = (
        REMOTE_STATE / "evidence" / "curve_gate.json"
        if command == "curve"
        else REMOTE_STATE / "final_unlocked.json"
    )
    payload = json.loads(path.read_text()) if path.exists() else {}
    payload["returncode"] = result.returncode
    return payload


@app.function(
    image=repo_image,
    cpu=2,
    memory=4096,
    timeout=2 * 60 * 60,
    startup_timeout=60 * 60,
    retries=0,
    volumes={REMOTE_STATE: state_volume},
)
def claim_final_collection(collection_id: str) -> dict[str, Any]:
    state_volume.reload()
    result = _protocol(
        "begin-final", "--collection-id", collection_id, check=False
    )
    state_volume.commit()
    path = REMOTE_STATE / "final_collection.json"
    payload = json.loads(path.read_text()) if path.exists() else {}
    payload["returncode"] = result.returncode
    return payload


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
    volumes={REMOTE_STATE: state_volume},
)
def evaluate_label(label: str, collection_id: str) -> dict[str, Any]:
    if label not in FINAL_LABELS:
        raise ValueError("evaluation label is outside registered V7")
    state_volume.reload()
    lease = _acquire_gpu_lease("sealed_evaluation", label)
    try:
        _ensure_runtime_git()
        _protocol("verify-final", "--collection-id", collection_id)
        output = REMOTE_STATE / "evals" / f"{label}.jsonl"
        if output.exists():
            _protocol("verify-eval", "--path", str(output), "--label", label)
            return {
                "label": label,
                "collection_id": collection_id,
                "reused_verified_output": True,
            }
        if label == "base":
            experiment_config = (
                REMOTE_STATE / "configs" / f"jlens_seed{SEEDS[0]}.json"
            )
            adapter_args: list[str] = []
        else:
            condition, seed_text = label.rsplit("_seed", 1)
            experiment_config = (
                REMOTE_STATE / "configs" / f"{condition}_seed{seed_text}.json"
            )
            adapter_args = [
                "--adapter", str(REMOTE_STATE / "runs" / label / "final")
            ]
        output.parent.mkdir(parents=True, exist_ok=True)
        _run(
            [
                sys.executable,
                "-m",
                "jlens_rl.eval",
                "--config",
                str(REMOTE_STATE / "configs" / "sealed_eval.json"),
                "--experiment-config",
                str(experiment_config),
                "--indices-manifest",
                str(REMOTE_STATE / "manifests" / "sealed_final_indices.json"),
                "--output-jsonl",
                str(output),
                "--run-label",
                label,
                "--batch-size",
                "64",
                "--skip-jlens-metric",
                *adapter_args,
            ]
        )
        _protocol("verify-eval", "--path", str(output), "--label", label)
        return {
            "label": label,
            "collection_id": collection_id,
            "reused_verified_output": False,
        }
    finally:
        state_volume.commit()
        _release_gpu_lease(lease)


def _comparison_args(labels: Iterable[str], option: str) -> list[str]:
    arguments: list[str] = []
    for label in labels:
        arguments.extend([option, str(REMOTE_STATE / "evals" / f"{label}.jsonl")])
    return arguments


@app.function(
    image=repo_image,
    cpu=4,
    memory=16384,
    timeout=2 * 60 * 60,
    startup_timeout=60 * 60,
    retries=0,
    volumes={REMOTE_STATE: state_volume},
)
def analyze_final(collection_id: str) -> dict[str, Any]:
    state_volume.reload()
    _protocol("verify-final", "--collection-id", collection_id)
    evidence_dir = REMOTE_STATE / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    output = evidence_dir / "sealed_comparison.json"
    command = [
        sys.executable,
        "-m",
        "jlens_rl.paired_eval",
        "--base-jsonl",
        str(REMOTE_STATE / "evals" / "base.jsonl"),
        *_comparison_args(
            [f"jlens_seed{seed}" for seed in SEEDS], "--adapter-jsonl"
        ),
        *_comparison_args(
            [f"signflip_seed{seed}" for seed in SEEDS], "--control-jsonl"
        ),
        "--bootstrap-samples",
        "10000",
        "--seed",
        "0",
        "--confidence",
        "0.95",
        "--output",
        str(output),
    ]
    if output.exists():
        raise FileExistsError(f"refusing to overwrite V7 analysis: {output}")
    try:
        from jlens_rl.common import runtime_environment_snapshot

        analysis_process = {
            "python_executable": sys.executable,
            "command": command,
            "cwd": str(REMOTE_REPO),
            "environment_snapshot": runtime_environment_snapshot(),
        }
        (evidence_dir / "analysis_process.json").write_text(
            json.dumps(analysis_process, indent=2, sort_keys=True) + "\n"
        )
        _run(command)
        report_process = _protocol("report", check=False)
        report = json.loads((evidence_dir / "acceptance.json").read_text())
        report["returncode"] = report_process.returncode
        return report
    finally:
        state_volume.commit()


@app.function(
    image=repo_image,
    cpu=4,
    memory=16384,
    # This Modal workspace currently enforces a 512 GiB minimum explicit
    # ephemeral-disk request.  The first prelaunch attempt requested 20 GiB
    # and was rejected before the local entrypoint, claim, GPU, or W&B ran.
    ephemeral_disk=1024 * 512,
    timeout=4 * 60 * 60,
    startup_timeout=60 * 60,
    retries=0,
    volumes={REMOTE_STATE: state_volume},
)
def finalize_attempt_evidence(claim_id: str) -> dict[str, Any]:
    state_volume.reload()
    _verify_attempt_claim(claim_id)
    try:
        _protocol("finalize-evidence")
        inventory = REMOTE_STATE / "evidence" / "evidence_bundle_inventory.json"
        plan = json.loads(
            (REMOTE_STATE / "evidence" / "durable_export_plan.json").read_text()
        )
        export_receipt = json.loads(
            (REMOTE_STATE / plan["hash_receipt_relative_path"]).read_text()
        )
        import hashlib

        return {
            "inventory_sha256": hashlib.sha256(inventory.read_bytes()).hexdigest(),
            "inventory_path": str(inventory),
            "export": export_receipt,
            "retrieval_command": plan["retrieval_command"],
            "closeout_candidate": str(
                REMOTE_STATE / "evidence" / "git_closeout_candidate.json"
            ),
        }
    finally:
        state_volume.commit()


def _mapped_results(function: Any, *inputs: Iterable[Any]) -> list[Any]:
    materialized = [list(values) for values in inputs]
    results = list(
        function.map(*materialized, order_outputs=True, return_exceptions=True)
    )
    failures = [
        {
            "inputs": [values[index] for values in materialized],
            "error": repr(result),
        }
        for index, result in enumerate(results)
        if isinstance(result, BaseException)
    ]
    if failures:
        raise RuntimeError(f"{len(failures)} mapped V7 jobs failed: {failures}")
    return results


def _serial_gpu_waves(values: Iterable[Any]) -> list[list[Any]]:
    materialized = list(values)
    return [[value] for value in materialized]


@app.function(
    image=repo_image,
    cpu=1,
    memory=2048,
    timeout=23 * 60 * 60,
    startup_timeout=60 * 60,
    retries=0,
    volumes={REMOTE_STATE: state_volume},
)
def orchestrate(claim_id: str) -> dict[str, Any]:
    state_volume.reload()
    _verify_attempt_claim(claim_id)
    failure_phase = "launch_receipt_wait"
    try:
        _wait_for_launch_receipt(claim_id)
        failure_phase = "initial_protocol_verify"
        _protocol("verify")
        failure_phase = "semantic_training"
        semantic: list[dict[str, Any]] = []
        semantic_waves = _serial_gpu_waves(SEEDS)
        for wave_index, wave in enumerate(semantic_waves, 1):
            _record_attempt_status(
                claim_id,
                "semantic_training",
                gpu_wave=wave_index,
                gpu_wave_count=len(semantic_waves),
                seeds=wave,
                global_modal_gpu_limit=GLOBAL_MODAL_GPU_LIMIT,
                gpu_app_overlap_policy=GPU_APP_OVERLAP_POLICY,
            )
            state_volume.commit()
            semantic.extend(_mapped_results(train_config, ["jlens"] * len(wave), wave))
            state_volume.reload()
        semantic_check = protocol_step.remote("verify-semantic")
        if semantic_check.get("returncode") or not semantic_check.get("verified"):
            raise RuntimeError(f"V7 semantic verification failed: {semantic_check}")
        failure_phase = "curve_gate"
        gate = protocol_step.remote("curve")
        if gate.get("returncode") or not gate.get("passed"):
            state_volume.reload()
            _record_attempt_status(claim_id, "curve_failed", curve=gate)
            state_volume.commit()
            finalized = finalize_attempt_evidence.remote(claim_id)
            return {
                "stage": "curve_failed",
                "semantic": semantic,
                "curve": gate,
                "durable_evidence": finalized,
            }

        failure_phase = "control_training"
        controls: list[dict[str, Any]] = []
        control_waves = _serial_gpu_waves(SEEDS)
        for wave_index, wave in enumerate(control_waves, 1):
            state_volume.reload()
            _record_attempt_status(
                claim_id,
                "control_training",
                curve=gate,
                gpu_wave=wave_index,
                gpu_wave_count=len(control_waves),
                seeds=wave,
                global_modal_gpu_limit=GLOBAL_MODAL_GPU_LIMIT,
                gpu_app_overlap_policy=GPU_APP_OVERLAP_POLICY,
            )
            state_volume.commit()
            controls.extend(
                _mapped_results(train_config, ["signflip"] * len(wave), wave)
            )
        failure_phase = "sealed_collection_unlock"
        unlock = protocol_step.remote("unlock")
        if unlock.get("returncode"):
            raise RuntimeError(f"V7 unlock failed: {unlock}")

        failure_phase = "sealed_collection_claim"
        collection_id = uuid.uuid4().hex
        collection = claim_final_collection.remote(collection_id)
        if collection.get("returncode") or collection.get("collection_id") != collection_id:
            raise RuntimeError(f"V7 final collection claim failed: {collection}")
        state_volume.reload()
        _record_attempt_status(
            claim_id,
            "sealed_evaluation",
            final_collection_id=collection_id,
            final_labels=list(FINAL_LABELS),
        )
        state_volume.commit()
        failure_phase = "sealed_evaluation"
        sealed: list[dict[str, Any]] = []
        final_waves = _serial_gpu_waves(FINAL_LABELS)
        for wave_index, wave in enumerate(final_waves, 1):
            state_volume.reload()
            _record_attempt_status(
                claim_id,
                "sealed_evaluation",
                final_collection_id=collection_id,
                final_labels=list(FINAL_LABELS),
                gpu_wave=wave_index,
                gpu_wave_count=len(final_waves),
                labels=wave,
                global_modal_gpu_limit=GLOBAL_MODAL_GPU_LIMIT,
                gpu_app_overlap_policy=GPU_APP_OVERLAP_POLICY,
            )
            state_volume.commit()
            sealed.extend(
                _mapped_results(
                    evaluate_label,
                    wave,
                    [collection_id] * len(wave),
                )
            )
        failure_phase = "sealed_analysis"
        report = analyze_final.remote(collection_id)
        final_stage = (
            "complete"
            if report.get("passed") and report.get("returncode") == 0
            else "significance_failed"
        )
        state_volume.reload()
        _record_attempt_status(
            claim_id,
            final_stage,
            final_collection_id=collection_id,
            acceptance=report,
        )
        state_volume.commit()
        failure_phase = "durable_evidence_export"
        durable_evidence = finalize_attempt_evidence.remote(claim_id)
        return {
            "stage": final_stage,
            "curve": gate,
            "semantic_training": semantic,
            "control_training": controls,
            "final_collection": collection,
            "sealed_evaluations": sealed,
            "acceptance_report": report,
            "durable_evidence": durable_evidence,
        }
    except BaseException as error:
        try:
            state_volume.reload()
            status_path = REMOTE_STATE / "attempt_status.json"
            current = json.loads(status_path.read_text()) if status_path.is_file() else {}
            terminal_stages = {
                "complete",
                "significance_failed",
                "curve_failed",
                "failed",
            }
            if current.get("stage") not in terminal_stages:
                _record_attempt_status(
                    claim_id,
                    "failed",
                    error=repr(error),
                    failed_from_stage=current.get("stage"),
                    failure_phase=failure_phase,
                    launch_receipt_present=_launch_receipt_is_present(),
                )
                state_volume.commit()
                finalize_attempt_evidence.remote(claim_id)
            else:
                inventory_path = (
                    REMOTE_STATE / "evidence" / "evidence_bundle_inventory.json"
                )
                failure = (
                    REMOTE_STATE / "exports" / "evidence_export_error.json"
                    if inventory_path.exists()
                    else REMOTE_STATE / "evidence" / "evidence_export_error.json"
                )
                if not failure.exists():
                    failure.parent.mkdir(parents=True, exist_ok=True)
                    failure.write_text(
                        json.dumps(
                            {
                                "claim_id": claim_id,
                                "scientific_terminal_stage": current.get("stage"),
                                "error": repr(error),
                                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                            },
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                state_volume.commit()
        except BaseException:
            pass
        raise


def _upload_protocol_state() -> None:
    subprocess.run(
        [sys.executable, "scripts/confirmatory_v7_protocol.py", "verify"],
        cwd=LOCAL_REPO,
        check=True,
    )
    allowed = [LOCAL_STATE / "protocol_state.json"]
    for directory in ("configs", "manifests", "frozen_artifacts", "reproducibility"):
        allowed.extend(
            sorted(
                path
                for path in (LOCAL_STATE / directory).rglob("*")
                if path.is_file()
            )
        )
    expected_counts = {
        "configs": 17,
        "manifests": 4,
        "frozen_artifacts": 2,
        "reproducibility": 7,
    }
    actual_counts = {
        directory: sum(path.parent.name == directory for path in allowed)
        for directory in expected_counts
    }
    if actual_counts != expected_counts:
        raise RuntimeError(f"prepared V7 upload inventory changed: {actual_counts}")
    with state_volume.batch_upload(force=False) as batch:
        for path in allowed:
            relative = path.relative_to(LOCAL_STATE)
            batch.put_file(path, f"/{relative.as_posix()}")


def _local_operational_preflight() -> dict[str, Any]:
    confirmation = os.environ.get("JLENS_MODAL_GPU_EXCLUSIVE_CONFIRM")
    if confirmation != GPU_EXCLUSIVE_CONFIRMATION:
        raise RuntimeError(
            "refusing V7 launch without an external no-overlap preflight; ensure "
            "all other Modal GPU apps are idle, then set "
            f"JLENS_MODAL_GPU_EXCLUSIVE_CONFIRM={GPU_EXCLUSIVE_CONFIRMATION}"
        )
    volume_object_id = verify_v7_volume_v2()
    modal_cli = Path(sys.executable).parent / "modal"
    listing_text = subprocess.check_output(
        [str(modal_cli), "app", "list", "--json"],
        cwd=LOCAL_REPO,
        text=True,
    )
    listing = json.loads(listing_text[listing_text.index("[") :])
    inventory_text = subprocess.check_output(
        [str(modal_cli), "volume", "ls", VOLUME_NAME, "/", "--json"],
        cwd=LOCAL_REPO,
        text=True,
    )
    inventory = json.loads(inventory_text[inventory_text.index("[") :])
    if inventory != []:
        raise RuntimeError(
            f"refusing V7 launch unless the fresh Volume is empty: {inventory}"
        )
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
            "refusing V7 launch while another Modal app remains active; stop it "
            f"first: {active_other_apps}"
        )
    try:
        gpu_lease.hydrate()
        lease_owner = gpu_lease.get(GPU_LEASE_KEY, None)
    except Exception as error:
        raise RuntimeError(
            "refusing V7 launch because the registered global GPU lease Dict "
            "cannot be hydrated"
        ) from error
    if lease_owner is not None:
        raise RuntimeError(
            f"refusing V7 launch while global GPU lease is occupied: {lease_owner!r}"
        )
    return _validate_operational_preflight(
        {
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "exclusive_gpu_confirmation": confirmation,
            "global_modal_gpu_limit": GLOBAL_MODAL_GPU_LIMIT,
            "active_other_modal_apps": active_other_apps,
            "volume_name": VOLUME_NAME,
            "volume_object_id": volume_object_id,
            "volume_version": 2,
            "gpu_lease_dict": GPU_LEASE_DICT_NAME,
            "gpu_lease_key": GPU_LEASE_KEY,
            "gpu_lease_owner_at_preflight": lease_owner,
        }
    )


@app.local_entrypoint()
def main() -> None:
    operational_preflight = _local_operational_preflight()
    _upload_protocol_state()
    claim_id = uuid.uuid4().hex
    claim_attempt.remote(claim_id, operational_preflight)
    call = orchestrate.spawn(claim_id)
    receipt = record_launch_receipt.remote(
        claim_id,
        app.app_id or app.name,
        call.object_id,
    )
    print(
        json.dumps(
            {
                "status": "submitted",
                "function_call_id": call.object_id,
                "app_id": app.app_id,
                "volume": VOLUME_NAME,
                "gpu_type": GPU_TYPE,
                "max_parallel_gpus": MAX_GPU_CONTAINERS,
                "seeds": list(SEEDS),
                "final_labels": list(FINAL_LABELS),
                "operational_preflight": operational_preflight,
                "launch_receipt": receipt,
            },
            indent=2,
        )
    )
