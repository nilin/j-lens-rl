"""Run the pre-registered seed-264 pair replacing V16's preempted seed 256.

Training writes first to container-local storage. Only a fully verified six-node
curve is copied to the fresh evidence Volume. Infrastructure restarts receive a
new, disclosed W&B attempt identity and can therefore restart cleanly.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

from modal_emotional_tournament_v1 import repo_image as cached_image


LOCAL_REPO = Path(__file__).resolve().parent
REMOTE_REPO = Path("/workspace/j-lens-rl")
LOCAL_RUN_ROOT = Path("/state/runs")
EVIDENCE_ROOT = Path("/evidence")
APP_NAME = "j-lens-rl-development-v16r-v14-celebration-seed264-20260715a"
VOLUME_NAME = APP_NAME
PROTOCOL = "j-lens-rl-development-v16r-v14-celebration-seed264-replacement-u2-h10"
REGISTRATION_PATH = "protocol_archive/v16r_seed264_preemption_replacement_registration.json"
REGISTRATION_SHA256 = "34cb51a1c748af00fe90ba4cc79e8a218d969ffea8d2551425cd225a7ea13fb6"
METRIC_SCHEMA_PATH = "protocol_archive/v16_v14_manyseed_curve_metric_schema.json"
METRIC_SCHEMA_SHA256 = "8f3c814334fedcf1e02f32c6622091638f8a92d8598a6b3b32c942271ca52b4d"
CURVE_MANIFEST_PATH = ".confirmatory/manifests/curve_indices.json"
CURVE_MANIFEST_SHA256 = "ad348fe17d2e6bd6aac691d9bcdbb9da481f675305fa0e05c68e86dad97451c1"
EXCLUSIONS_PATH = ".confirmatory/manifests/train_exclusions.json"
EXCLUSIONS_SHA256 = "7c1ca4f404ba9149093cc3c57dc3607582f671397e2fe99e93449848c1d65d61"
CALIBRATION_PATH = "protocol_archive/emotional_screen_forensic_bundle/family/artifacts/celebration_calibration.json"
CALIBRATION_SHA256 = "93d05caf4848e745c07d908034b36f0b1ae465d8d89e1681134869c6b87a8ee6"
LENS_PATH = "artifacts/qwen25_05b_solved_lens.pt"
LENS_SHA256 = "178a9671cbf41882135807bde59b828e36c6f8f98b32c809ea3346860aad10dc"
SEED = 264
CONDITIONS = ("jlens", "signflip")
STEPS = (0, 2, 4, 6, 8, 10)
POST_STEPS = STEPS[1:]
GPU_TYPE = "L40S"
MAX_PARALLEL_GPUS = 2
CONFIG_PATHS = {
    condition: f"configs/v16r_v14_seed264_{condition}.json"
    for condition in CONDITIONS
}
BASE_WANDB_IDS = {
    condition: f"dev-v16r-v14-celebration-{condition}-seed264"
    for condition in CONDITIONS
}
WANDB_ENTITY = "nilinabra-spare-time"
WANDB_PROJECT = "j-lens-rl"
RUNNER_PATH = "modal_v16r_seed264_replacement.py"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    base = value.pop("base", None)
    if base is None:
        return value
    result = _load_config(path.parent / base)
    result.update(value)
    return result


RUNTIME_PATHS = (
    ".gitignore",
    CURVE_MANIFEST_PATH,
    EXCLUSIONS_PATH,
    LENS_PATH,
    "configs/common.json",
    "configs/emotional_parallel_v3_common.json",
    "configs/v16r_v14_seed264_common.json",
    "configs/v16r_v14_seed264_jlens_template.json",
    "configs/v16r_v14_seed264_signflip_template.json",
    CONFIG_PATHS["jlens"],
    CONFIG_PATHS["signflip"],
    REGISTRATION_PATH,
    METRIC_SCHEMA_PATH,
    CALIBRATION_PATH,
    "pyproject.toml",
    "src/jlens_rl/common.py",
    "src/jlens_rl/eval.py",
    "src/jlens_rl/reward.py",
    "src/jlens_rl/train.py",
    RUNNER_PATH,
)
EXPECTED_SHA256 = {path: _sha256(LOCAL_REPO / path) for path in RUNTIME_PATHS}


def _validate_local_inputs() -> None:
    if EXPECTED_SHA256[REGISTRATION_PATH] != REGISTRATION_SHA256:
        raise RuntimeError("V16R registration hash changed")
    if EXPECTED_SHA256[METRIC_SCHEMA_PATH] != METRIC_SCHEMA_SHA256:
        raise RuntimeError("V16 metric schema hash changed")
    if EXPECTED_SHA256[CURVE_MANIFEST_PATH] != CURVE_MANIFEST_SHA256:
        raise RuntimeError("curve manifest hash changed")
    if EXPECTED_SHA256[EXCLUSIONS_PATH] != EXCLUSIONS_SHA256:
        raise RuntimeError("train exclusion manifest hash changed")
    if EXPECTED_SHA256[CALIBRATION_PATH] != CALIBRATION_SHA256:
        raise RuntimeError("celebration calibration hash changed")
    if EXPECTED_SHA256[LENS_PATH] != LENS_SHA256:
        raise RuntimeError("J-Lens artifact hash changed")
    registration = json.loads((LOCAL_REPO / REGISTRATION_PATH).read_text())
    if (
        registration.get("protocol") != PROTOCOL
        or registration.get("replacement_rule", {}).get("exclude_pair") != 256
        or registration.get("replacement_rule", {}).get("replacement_seed") != SEED
        or registration.get("combined_complete_pair_cohort")
        != [248, 249, 250, 251, 252, 253, 254, 255, 257, 258, 259, 260, 261, 262, 263, 264]
        or registration.get("protected_final_payloads_accessed") is not False
    ):
        raise RuntimeError("V16R registration semantics changed")
    configs = {
        condition: _load_config(LOCAL_REPO / CONFIG_PATHS[condition])
        for condition in CONDITIONS
    }
    for condition, config in configs.items():
        expected_identity = {
            "seed": SEED,
            "updates": 10,
            "eval_every": 2,
            "validation_steps": list(POST_STEPS),
            "save_every": 10,
            "registration_sha256": REGISTRATION_SHA256,
            "metric_schema_sha256": METRIC_SCHEMA_SHA256,
            "curve_manifest_sha256": CURVE_MANIFEST_SHA256,
            "train_exclusions_manifest_sha256": EXCLUSIONS_SHA256,
            "reward_type": "jlens",
            "wandb_resume": "never",
        }
        changed = {
            key: (expected, config.get(key))
            for key, expected in expected_identity.items()
            if config.get(key) != expected
        }
        if changed:
            raise RuntimeError(f"{condition} config identity changed: {changed}")
        if config.get("target_words") != ["yay", "great", "success", "nice"]:
            raise RuntimeError("V16R target words changed")
    allowed = {
        "score_components", "run_name", "wandb_run_id", "wandb_url",
        "wandb_tags", "output_dir"
    }
    unexpected = {
        key for key in configs["jlens"]
        if configs["jlens"].get(key) != configs["signflip"].get(key)
        and key not in allowed
    }
    if unexpected:
        raise RuntimeError(f"V16R pair differs outside reward sign/identity: {unexpected}")
    treatment = [float(x["weight"]) for x in configs["jlens"]["score_components"]]
    control = [float(x["weight"]) for x in configs["signflip"]["score_components"]]
    if treatment != [1.0, 0.25] or control != [-1.0, -0.25]:
        raise RuntimeError("V16R component weights changed")


_validate_local_inputs()

image = cached_image
for relative in RUNTIME_PATHS:
    image = image.add_local_file(
        LOCAL_REPO / relative, (REMOTE_REPO / relative).as_posix(), copy=True
    )
image = (
    image.env(
        {
            "GIT_AUTHOR_NAME": "J-Lens V16R Runtime",
            "GIT_AUTHOR_EMAIL": "runtime@example.invalid",
            "GIT_COMMITTER_NAME": "J-Lens V16R Runtime",
            "GIT_COMMITTER_EMAIL": "runtime@example.invalid",
            "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+00:00",
            "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+00:00",
            "JLENS_REPOSITORY_ROOT": REMOTE_REPO.as_posix(),
            "JLENS_MODAL_IMAGE_SPEC": "j-lens-rl-v16r-seed264-replacement-l40s-v1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    .run_commands(
        "find . -type d -name __pycache__ -prune -exec rm -rf {} +",
        "find . -type d -name '*.egg-info' -prune -exec rm -rf {} +",
        "rm -rf .git",
        "git init -q",
        "git add -f .",
        "git commit -qm 'J-Lens V16R seed264 replacement runtime'",
        "test -z \"$(git status --porcelain=v1 --untracked-files=all)\"",
    )
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)
wandb_secret = modal.Secret.from_name(
    "j-lens-rl-wandb", required_keys=["WANDB_API_KEY"]
)


def _write_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _runtime_hashes() -> dict[str, str]:
    observed = {path: _sha256(REMOTE_REPO / path) for path in RUNTIME_PATHS}
    if observed != EXPECTED_SHA256:
        raise RuntimeError("V16R runtime file hashes changed")
    return observed


def _verify_terminal(run_dir: Path, condition: str, wandb_id: str) -> dict[str, Any]:
    required = {
        "validation_history.jsonl", "log_history.json", "resolved_config.json",
        "run_manifest.json", "run_result_manifest.json",
        "wandb_terminal_publish_receipt.json", "data_indices.json",
    }
    if not required.issubset({p.name for p in run_dir.iterdir() if p.is_file()}):
        raise RuntimeError(f"{condition} terminal evidence is incomplete")
    rows = [json.loads(line) for line in (run_dir / "validation_history.jsonl").read_text().splitlines() if line]
    if [row.get("step") for row in rows] != list(STEPS):
        raise RuntimeError(f"{condition} curve does not have exact 0/2/4/6/8/10 nodes")
    for row in rows:
        exact = row.get("exact_match")
        if (
            isinstance(exact, bool) or not isinstance(exact, (int, float))
            or not math.isfinite(float(exact)) or not 0 <= float(exact) <= 1
            or row.get("validation_source") != "train"
            or row.get("validation_indices_sha256") != CURVE_MANIFEST_SHA256
        ):
            raise RuntimeError(f"{condition} curve row is invalid")
    config = json.loads((run_dir / "resolved_config.json").read_text())
    if (
        config.get("seed") != SEED or config.get("updates") != 10
        or config.get("eval_every") != 2
        or config.get("wandb_run_id") != wandb_id
        or config.get("reward_type") != "jlens"
        or config.get("validation_observational_only") is not True
    ):
        raise RuntimeError(f"{condition} resolved config changed")
    logs = json.loads((run_dir / "log_history.json").read_text())
    reward_rows = [row for row in logs if "reward" in row]
    validation_rows = [row for row in logs if "validation/exact_match" in row]
    if (
        [row.get("step") for row in reward_rows] != list(range(1, 11))
        or [row.get("step") for row in validation_rows] != list(POST_STEPS)
        or any("correctness" in key.lower() or "gsm8k" in key.lower() for row in reward_rows for key in row)
    ):
        raise RuntimeError(f"{condition} trainer history is not one-J-reward 10-step training")
    receipt = json.loads((run_dir / "wandb_terminal_publish_receipt.json").read_text())
    rendered_receipt = json.dumps(receipt, sort_keys=True)
    if wandb_id not in rendered_receipt:
        raise RuntimeError(f"{condition} W&B receipt does not bind the attempt ID")
    checkpoints = list(run_dir.glob("checkpoint-10/trainer_state.json"))
    if len(checkpoints) != 1 or json.loads(checkpoints[0].read_text()).get("global_step") != 10:
        raise RuntimeError(f"{condition} lacks the exact step-10 checkpoint")
    files = sorted(path for path in run_dir.rglob("*") if path.is_file())
    return {
        "condition": condition,
        "seed": SEED,
        "steps": list(STEPS),
        "curve": [float(row["exact_match"]) for row in rows],
        "wandb_run_id": wandb_id,
        "wandb_url": f"https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}/runs/{wandb_id}",
        "validation_history_sha256": _sha256(run_dir / "validation_history.jsonl"),
        "wandb_terminal_publish_receipt_sha256": _sha256(run_dir / "wandb_terminal_publish_receipt.json"),
        "file_sha256": {str(path.relative_to(run_dir)): _sha256(path) for path in files},
    }


@app.function(
    image=image,
    gpu=GPU_TYPE,
    cpu=4,
    memory=32768,
    max_containers=MAX_PARALLEL_GPUS,
    timeout=3 * 60 * 60,
    startup_timeout=60 * 60,
    retries=0,
    single_use_containers=True,
    secrets=[wandb_secret],
    volumes={EVIDENCE_ROOT: volume},
)
def train_run(condition: str, claim_id: str) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise RuntimeError("unregistered V16R condition")
    volume.reload()
    completion_path = EVIDENCE_ROOT / "dispatches" / f"{condition}_seed264.completion.json"
    if completion_path.is_file():
        return json.loads(completion_path.read_text())
    runtime_hashes = _runtime_hashes()
    if __import__("torch").cuda.device_count() != 1:
        raise RuntimeError("V16R worker did not receive exactly one GPU")
    attempts = EVIDENCE_ROOT / "attempts" / condition
    prior = sorted(attempts.glob("*.intent.json")) if attempts.exists() else []
    attempt_number = len(prior) + 1
    attempt_token = uuid.uuid4().hex[:12]
    wandb_id = f"{BASE_WANDB_IDS[condition]}-a{attempt_number}-{attempt_token}"
    base_config = _load_config(REMOTE_REPO / CONFIG_PATHS[condition])
    base_config.update(
        {
            "run_name": wandb_id,
            "wandb_run_id": wandb_id,
            "wandb_url": f"https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}/runs/{wandb_id}",
            "output_dir": f"/state/runs/{condition}_seed264",
        }
    )
    runtime_config = Path(f"/tmp/{condition}_seed264_{attempt_token}.json")
    runtime_config.write_text(json.dumps(base_config, indent=2, sort_keys=True) + "\n")
    intent = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "claim_id": claim_id,
        "condition": condition,
        "seed": SEED,
        "attempt_number": attempt_number,
        "attempt_token": attempt_token,
        "wandb_run_id": wandb_id,
        "runtime_config_sha256": _sha256(runtime_config),
        "base_config_sha256": EXPECTED_SHA256[CONFIG_PATHS[condition]],
        "status": "written_before_training",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_exclusive(attempts / f"{attempt_number:03d}-{attempt_token}.intent.json", intent)
    volume.commit()
    completed = subprocess.run(
        [sys.executable, "-m", "jlens_rl.train", "--config", str(runtime_config), "--wandb-mode", "online"],
        cwd=REMOTE_REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    attempt_log_dir = EVIDENCE_ROOT / "attempt_logs" / condition
    attempt_log_dir.mkdir(parents=True, exist_ok=True)
    (attempt_log_dir / f"{attempt_number:03d}-{attempt_token}.stdout").write_text(completed.stdout)
    (attempt_log_dir / f"{attempt_number:03d}-{attempt_token}.stderr").write_text(completed.stderr)
    if completed.returncode:
        failure = {**intent, "status": "training_failed", "returncode": completed.returncode, "failed_at_utc": datetime.now(timezone.utc).isoformat()}
        _write_exclusive(attempts / f"{attempt_number:03d}-{attempt_token}.failure.json", failure)
        volume.commit()
        raise RuntimeError(f"V16R {condition} training failed")
    local_dir = LOCAL_RUN_ROOT / f"{condition}_seed264"
    verified = _verify_terminal(local_dir, condition, wandb_id)
    terminal_dir = EVIDENCE_ROOT / "runs" / f"{condition}_seed264"
    if terminal_dir.exists():
        raise RuntimeError(f"V16R {condition} terminal directory already exists")
    shutil.copytree(local_dir, terminal_dir)
    completion = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "claim_id": claim_id,
        "attempt_intent_sha256": _sha256(attempts / f"{attempt_number:03d}-{attempt_token}.intent.json"),
        "runtime_source_sha256": runtime_hashes,
        "status": "terminal_public_run_verified",
        **verified,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_exclusive(completion_path, completion)
    volume.commit()
    return completion


@app.function(image=image, cpu=1, memory=2048, timeout=20 * 60, retries=0, volumes={EVIDENCE_ROOT: volume})
def write_json(relative: str, value: dict[str, Any]) -> dict[str, Any]:
    volume.reload()
    path = EVIDENCE_ROOT / relative
    _write_exclusive(path, value)
    volume.commit()
    return value


def _local_preflight() -> dict[str, Any]:
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=LOCAL_REPO, text=True,
    )
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=LOCAL_REPO, text=True).strip()
    pushed = subprocess.check_output(["git", "rev-parse", "origin/main"], cwd=LOCAL_REPO, text=True).strip()
    if status or head != pushed:
        raise RuntimeError("V16R launch requires an exact clean pushed main")
    _validate_local_inputs()
    listing_text = subprocess.check_output(["modal", "app", "list", "--json"], text=True)
    listing = json.loads(listing_text[listing_text.index("["):])
    overlapping = [
        {
            "app_id": item.get("app_id"),
            "description": item.get("description"),
            "tasks": item.get("tasks"),
        }
        for item in listing
        if item.get("stopped_at") is None
        and item.get("state") != "stopped"
        and str(item.get("description", "")).startswith("j-lens-rl-development")
        and item.get("description") != APP_NAME
    ]
    if overlapping:
        raise RuntimeError(
            "V16R refuses to overlap another development app; wait for V16 to drain: "
            f"{overlapping}"
        )
    return {
        "source_commit": head,
        "source_file_sha256": EXPECTED_SHA256,
        "overlapping_development_apps": overlapping,
    }


@app.local_entrypoint()
def main() -> None:
    preflight = _local_preflight()
    claim_id = uuid.uuid4().hex
    claim = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "claim_id": claim_id,
        "registration_sha256": REGISTRATION_SHA256,
        "seed": SEED,
        "conditions": list(CONDITIONS),
        "steps": list(STEPS),
        "max_parallel_gpus": MAX_PARALLEL_GPUS,
        "preflight": preflight,
        "protected_final_payloads_accessed": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json.remote("attempt_claim.json", claim)
    calls = {condition: train_run.spawn(condition, claim_id) for condition in CONDITIONS}
    dispatch = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "claim_id": claim_id,
        "worker_call_ids": {condition: call.object_id for condition, call in calls.items()},
        "recorded_before_gather": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json.remote("dispatch.json", dispatch)
    print(json.dumps(dispatch, indent=2, sort_keys=True), flush=True)
    results = {condition: calls[condition].get() for condition in CONDITIONS}
    summary = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "claim_id": claim_id,
        "status": "complete",
        "results": results,
        "protected_final_payloads_accessed": False,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json.remote("summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
