"""Run three isolated J-only exploratory screens on Modal.

This app deliberately has no access to the confirmatory protocol Volume,
individual current evaluation manifests, or outcomes. It receives only the
already-exposed retired V2 curve plus a combined exclusion-only manifest that
keeps every current confirmatory allocation out of exploratory training. It
caps itself at three L40S workers so it can coexist with the five-worker
frozen confirmatory app.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal


LOCAL_REPO = Path(__file__).resolve().parent
LOCAL_ARTIFACTS = Path("/j-lens-rl/artifacts")
LOCAL_EXPOSED_MANIFESTS = Path("/j-lens-rl/.confirmatory/manifests")
REMOTE_REPO = Path("/workspace/j-lens-explore")
REMOTE_OUTPUT = Path("/explore")
VOLUME_NAME = "j-lens-rl-exploratory-screen-v1-20260714a"
GPU_TYPE = "L40S"
MAX_GPU_CONTAINERS = 3
VARIANTS = {
    "solved_delta3e6": "configs/explore_solved_delta3e6.json",
    "solved_dense3e6": "configs/explore_solved_dense3e6.json",
    "solved_multilayer3e6": "configs/explore_solved_multilayer3e6.json",
}
CONFIG_FILES = tuple(
    sorted(
        {
            *VARIANTS.values(),
            "configs/explore_common.json",
            "configs/confirmatory_common.json",
        }
    )
)
EXPOSED_MANIFESTS = (
    "train_exclusions.json",
    "retired_v2_curve_indices.json",
)
FORBIDDEN_MANIFESTS = (
    "curve_indices.json",
    "sealed_final_indices.json",
    "future_reserve_indices.json",
)

app = modal.App("j-lens-rl-exploratory-screen-v1")
output_volume = modal.Volume.from_name(
    VOLUME_NAME, create_if_missing=True, version=2
)
wandb_secret = modal.Secret.from_name("j-lens-rl-wandb")

repo_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .add_local_dir(
        LOCAL_REPO,
        REMOTE_REPO.as_posix(),
        copy=True,
        ignore=[
            ".venv",
            ".venv/**",
            ".env",
            "modal.sh",
            "artifacts",
            "artifacts/**",
            "runs",
            "runs/**",
            "wandb",
            "wandb/**",
            ".confirmatory",
            ".confirmatory/**",
            ".pytest_cache",
            ".pytest_cache/**",
            "**/__pycache__/**",
            "*.egg-info/**",
        ],
    )
    .add_local_file(
        LOCAL_ARTIFACTS / "qwen25_05b_solved_lens.pt",
        (REMOTE_REPO / "artifacts/qwen25_05b_solved_lens.pt").as_posix(),
        copy=True,
    )
    .add_local_file(
        LOCAL_ARTIFACTS / "qwen25_05b_solved_calibration.json",
        (REMOTE_REPO / "artifacts/qwen25_05b_solved_calibration.json").as_posix(),
        copy=True,
    )
    .add_local_file(
        LOCAL_EXPOSED_MANIFESTS / "train_exclusions.json",
        (
            REMOTE_REPO
            / ".confirmatory/manifests/train_exclusions.json"
        ).as_posix(),
        copy=True,
    )
    .add_local_file(
        LOCAL_EXPOSED_MANIFESTS / "retired_v2_curve_indices.json",
        (
            REMOTE_REPO
            / ".confirmatory/manifests/retired_v2_curve_indices.json"
        ).as_posix(),
        copy=True,
    )
    .workdir(REMOTE_REPO)
    .env(
        {
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "JLENS_REPOSITORY_ROOT": REMOTE_REPO.as_posix(),
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
        "python scripts/modal_cache_assets.py",
        "python scripts/modal_finalize_image.py",
    )
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def _git_status(root: Path) -> list[str]:
    return subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        text=True,
    ).splitlines()


def _manifest_indices(path: Path) -> set[int]:
    payload = json.loads(path.read_text())
    values = payload.get("indices") if isinstance(payload, dict) else payload
    if not isinstance(values, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in values
    ):
        raise ValueError(f"{path} does not contain integer indices")
    if len(values) != len(set(values)):
        raise ValueError(f"{path} contains duplicate indices")
    return set(values)


def _verify_local_data_firewall() -> None:
    train_exclusions = _manifest_indices(
        LOCAL_EXPOSED_MANIFESTS / "train_exclusions.json"
    )
    retired_curve = _manifest_indices(
        LOCAL_EXPOSED_MANIFESTS / "retired_v2_curve_indices.json"
    )
    if not retired_curve <= train_exclusions:
        raise RuntimeError("retired V2 validation is not excluded from training")
    for forbidden_name in FORBIDDEN_MANIFESTS:
        forbidden_indices = _manifest_indices(
            LOCAL_EXPOSED_MANIFESTS / forbidden_name
        )
        if not forbidden_indices <= train_exclusions:
            raise RuntimeError(
                f"current {forbidden_name} is not fully excluded from training"
            )
        if retired_curve & forbidden_indices:
            raise RuntimeError(
                f"retired V2 validation overlaps current {forbidden_name}"
            )


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=REMOTE_REPO, check=True, text=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _launch_manifest(claim_id: str) -> dict[str, Any]:
    status = _git_status(LOCAL_REPO)
    if status:
        raise RuntimeError(f"exploratory launch requires a clean clone: {status}")
    _verify_local_data_firewall()
    return {
        "claim_id": claim_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(LOCAL_REPO),
        "variants": VARIANTS,
        "config_sha256": {
            path: _sha256(LOCAL_REPO / path) for path in CONFIG_FILES
        },
        "artifact_sha256": {
            "qwen25_05b_solved_lens.pt": _sha256(
                LOCAL_ARTIFACTS / "qwen25_05b_solved_lens.pt"
            ),
            "qwen25_05b_solved_calibration.json": _sha256(
                LOCAL_ARTIFACTS / "qwen25_05b_solved_calibration.json"
            ),
        },
        "exposed_manifest_sha256": {
            name: _sha256(LOCAL_EXPOSED_MANIFESTS / name)
            for name in EXPOSED_MANIFESTS
        },
        "data_boundary": (
            "already-exposed retired V2 400-item curve; all current "
            "confirmatory allocations excluded from training"
        ),
        "unmounted_current_evaluation_manifests": list(FORBIDDEN_MANIFESTS),
        "local_data_firewall_verified": True,
        "gpu_type": GPU_TYPE,
        "max_parallel_gpus": MAX_GPU_CONTAINERS,
    }


def _verify_remote_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("variants") != VARIANTS:
        raise RuntimeError("variant map changed between launch and runtime")
    if manifest.get("git_commit") != _git_commit(REMOTE_REPO):
        raise RuntimeError("runtime Git commit differs from launch manifest")
    if _git_status(REMOTE_REPO):
        raise RuntimeError("runtime checkout is dirty")
    expected_configs = {
        path: _sha256(REMOTE_REPO / path) for path in CONFIG_FILES
    }
    if manifest.get("config_sha256") != expected_configs:
        raise RuntimeError("runtime config hashes differ from launch manifest")
    expected_artifacts = {
        "qwen25_05b_solved_lens.pt": _sha256(
            REMOTE_REPO / "artifacts/qwen25_05b_solved_lens.pt"
        ),
        "qwen25_05b_solved_calibration.json": _sha256(
            REMOTE_REPO / "artifacts/qwen25_05b_solved_calibration.json"
        ),
    }
    if manifest.get("artifact_sha256") != expected_artifacts:
        raise RuntimeError("runtime artifact hashes differ from launch manifest")
    expected_manifests = {
        name: _sha256(REMOTE_REPO / ".confirmatory/manifests" / name)
        for name in EXPOSED_MANIFESTS
    }
    if manifest.get("exposed_manifest_sha256") != expected_manifests:
        raise RuntimeError("runtime exposed-manifest hashes differ from launch manifest")


def _set_status(claim_id: str, stage: str, **details: Any) -> None:
    manifest = json.loads((REMOTE_OUTPUT / "attempt_manifest.json").read_text())
    if manifest.get("claim_id") != claim_id:
        raise RuntimeError("exploratory Volume claim does not match")
    _write_json(
        REMOTE_OUTPUT / "attempt_status.json",
        {
            "claim_id": claim_id,
            "stage": stage,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            **details,
        },
    )


@app.function(
    image=repo_image,
    cpu=2,
    memory=4096,
    timeout=10 * 60,
    startup_timeout=60 * 60,
    retries=0,
    volumes={REMOTE_OUTPUT: output_volume},
)
def claim_attempt(manifest: dict[str, Any]) -> dict[str, Any]:
    output_volume.reload()
    existing = [path.name for path in REMOTE_OUTPUT.iterdir()]
    if existing:
        raise RuntimeError(f"exploratory Volume is not fresh: {sorted(existing)}")
    _verify_remote_manifest(manifest)
    _write_json(REMOTE_OUTPUT / "attempt_manifest.json", manifest)
    _set_status(str(manifest["claim_id"]), "claimed")
    output_volume.commit()
    return manifest


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
    volumes={REMOTE_OUTPUT: output_volume},
)
def train_variant(label: str) -> dict[str, Any]:
    if label not in VARIANTS:
        raise ValueError(f"unknown exploratory variant: {label}")
    output_volume.reload()
    manifest = json.loads((REMOTE_OUTPUT / "attempt_manifest.json").read_text())
    _verify_remote_manifest(manifest)
    try:
        _run(
            [
                sys.executable,
                "-m",
                "jlens_rl.train",
                "--config",
                VARIANTS[label],
                "--wandb-mode",
                "online",
            ]
        )
        history_path = REMOTE_OUTPUT / "runs" / label / "validation_history.jsonl"
        rows = [json.loads(line) for line in history_path.read_text().splitlines() if line]
        by_step = {int(row["step"]): float(row["exact_match"]) for row in rows}
        expected_steps = [0, 5, 10, 15, 20, 25]
        if sorted(by_step) != expected_steps:
            raise RuntimeError(f"{label} has incomplete fixed curve: {sorted(by_step)}")
        gate = (
            by_step[5] > by_step[0]
            and by_step[10] >= by_step[5]
            and by_step[15] >= by_step[10]
        )
        result = {
            "label": label,
            "curve": {str(step): by_step[step] for step in expected_steps},
            "requested_curve_pattern": gate,
            "literal_target_completion_rate": {
                str(row["step"]): float(row["literal_target_completion_rate"])
                for row in rows
            },
        }
        _write_json(REMOTE_OUTPUT / "runs" / label / "screen_result.json", result)
        return result
    finally:
        output_volume.commit()


@app.function(
    image=repo_image,
    cpu=1,
    memory=2048,
    timeout=6 * 60 * 60,
    startup_timeout=60 * 60,
    retries=0,
    volumes={REMOTE_OUTPUT: output_volume},
)
def orchestrate(claim_id: str) -> dict[str, Any]:
    output_volume.reload()
    _set_status(claim_id, "training")
    output_volume.commit()
    try:
        labels = list(VARIANTS)
        results = list(
            train_variant.map(
                labels,
                order_outputs=True,
                return_exceptions=True,
            )
        )
        failures = [
            {"label": labels[index], "error": repr(result)}
            for index, result in enumerate(results)
            if isinstance(result, BaseException)
        ]
        if failures:
            raise RuntimeError(f"exploratory workers failed: {failures}")
        output_volume.reload()
        _set_status(claim_id, "complete", results=results)
        output_volume.commit()
        return {"stage": "complete", "results": results}
    except BaseException as error:
        try:
            output_volume.reload()
            _set_status(claim_id, "failed", error=repr(error))
            output_volume.commit()
        except BaseException:
            pass
        raise


@app.local_entrypoint()
def main() -> None:
    claim_id = uuid.uuid4().hex
    manifest = _launch_manifest(claim_id)
    claim_attempt.remote(manifest)
    call = orchestrate.spawn(claim_id)
    print(
        json.dumps(
            {
                "status": "submitted",
                "function_call_id": call.object_id,
                "volume": VOLUME_NAME,
                "gpu_type": GPU_TYPE,
                "max_parallel_gpus": MAX_GPU_CONTAINERS,
                "variants": list(VARIANTS),
            },
            indent=2,
        )
    )
