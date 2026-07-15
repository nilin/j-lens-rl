"""Run the fresh six-update V11-style celebration development replication.

Four celebration J-lens treatment seeds and four exactly seed-matched negative
sign-flip controls are evaluated on the exposed 400-row development curve at
optimizer steps 0,1,2,3,4,5,6.  This runner is deliberately separate from the
closed V11/V12/V13 attempts.  It never mounts or reads a
protected-final, reserve, V11, V12, or V13 state Volume.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import struct
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import modal

from modal_emotional_tournament_v1 import repo_image as cached_tournament_image


LOCAL_REPO = Path(__file__).resolve().parent
REMOTE_REPO = Path("/workspace/j-lens-rl")
REMOTE_STATE = Path("/state")
APP_NAME = "j-lens-rl-development-v14-v11style-celebration"
VOLUME_NAME = "j-lens-rl-development-v14-v11style-celebration-20260714a"
PROTOCOL = "j-lens-rl-development-v14-v11style-celebration-u1-h6"
REGISTRATION_PATH = "protocol_archive/v14_v11style_celebration_registration.json"
REGISTRATION_SHA256 = "b1cb72b0502f3e0fadb85043756b9d6778a83372a65ff5e12102625aee5d5719"
METRIC_SCHEMA_PATH = "protocol_archive/v14_v11style_celebration_metric_schema.json"
METRIC_SCHEMA_SHA256 = "4d5784a27b83804a83281fe95cba21f1093c39e934e8e1ffa7a9323a716a97f0"
SOURCE_CONFIG_PATH = "protocol_archive/seed195_public_evidence/terminal/resolved_config.json"
SOURCE_CONFIG_SHA256 = "f290ceded76e5d5cc174ba53f67d9c6d709cf6626f20e4c8fa7179cf9ce5456a"
CURVE_MANIFEST_PATH = ".confirmatory/manifests/curve_indices.json"
CURVE_MANIFEST_SHA256 = "ad348fe17d2e6bd6aac691d9bcdbb9da481f675305fa0e05c68e86dad97451c1"
TRAIN_EXCLUSIONS_PATH = ".confirmatory/manifests/train_exclusions.json"
TRAIN_EXCLUSIONS_SHA256 = "7c1ca4f404ba9149093cc3c57dc3607582f671397e2fe99e93449848c1d65d61"
CALIBRATION_PATH = (
    "protocol_archive/emotional_screen_forensic_bundle/family/artifacts/"
    "celebration_calibration.json"
)
CALIBRATION_SHA256 = "93d05caf4848e745c07d908034b36f0b1ae465d8d89e1681134869c6b87a8ee6"
LENS_PATH = "artifacts/qwen25_05b_solved_lens.pt"
LENS_SHA256 = "178a9671cbf41882135807bde59b828e36c6f8f98b32c809ea3346860aad10dc"
GPU_TYPE = "L40S"
MAX_PARALLEL_GPUS = 8
WORKSPACE_GPU_LIMIT = 8
SEEDS = (236, 237, 238, 239)
CONDITIONS = ("jlens", "signflip")
STEPS = tuple(range(7))
POST_BASELINE_STEPS = STEPS[1:]
V11_GATE_STEPS = (0, 4, 5, 6)
WANDB_ENTITY = "nilinabra-spare-time"
WANDB_PROJECT = "j-lens-rl"
WANDB_GROUP = "dev-v14-v11style-celebration-u1-h6"
AGGREGATE_WANDB_ID = "dev-v14-v11style-celebration-aggregate"
RUNNER_PATH = "modal_v14_v11style_celebration.py"
COMMON_CONFIG_PATH = "configs/v14_v11style_celebration_common.json"
CONFIG_PATHS = {
    (condition, seed): (
        f"configs/v14_v11style_celebration_{condition}_seed{seed}.json"
    )
    for condition in CONDITIONS
    for seed in SEEDS
}
WANDB_IDS = {
    (condition, seed): f"dev-v14-v11style-celebration-{condition}-seed{seed}"
    for condition in CONDITIONS
    for seed in SEEDS
}
LABELS = tuple(
    f"{condition}_seed{seed}" for condition in CONDITIONS for seed in SEEDS
)
TREATMENT_COMPONENTS = (
    {
        "layer": 8,
        "start_fraction": 0.5,
        "end_fraction": 0.75,
        "aggregation": "mean",
        "weight": 1.0,
    },
    {
        "layer": 8,
        "start_fraction": 0.75,
        "end_fraction": 1.0,
        "aggregation": "mean",
        "weight": 0.25,
    },
)
CONTROL_COMPONENTS = tuple(
    {**component, "weight": -float(component["weight"])}
    for component in TREATMENT_COMPONENTS
)
EXPECTED_FILE_SHA256 = {
    ".gitignore": "2093c1ee68d1070775e3fc36502041a32ade3c15e70c670d628e5b92060e665c",
    CURVE_MANIFEST_PATH: CURVE_MANIFEST_SHA256,
    TRAIN_EXCLUSIONS_PATH: TRAIN_EXCLUSIONS_SHA256,
    LENS_PATH: LENS_SHA256,
    "configs/common.json": "c397905b4d4ac0cc64d7924d304b6aede4dc831d7d6e2a8b5622b63099266960",
    "configs/emotional_parallel_v3_common.json": "d4e8b8495b5df4b91a3110ef0baab08c1dcda1a5ca88b00fc4b45b099ba133ef",
    COMMON_CONFIG_PATH: "089d46ba1a5aa3af8b23c2ed7113816b08d20cd2894badd5f4fb25314a76ff42",
    "configs/v14_v11style_celebration_jlens_seed236.json": "4c785c35aed9357442830af11ac4a7def18ab7368c37352a7930ebdddc068e26",
    "configs/v14_v11style_celebration_jlens_seed237.json": "9704e6e61cdf0788c16becafda6d39ed95d43f724d81ec3313c6d46c8019f507",
    "configs/v14_v11style_celebration_jlens_seed238.json": "305d9b57a99d41534269092c3bb16159fe8fc22632889e4f75db0ee20794e941",
    "configs/v14_v11style_celebration_jlens_seed239.json": "c9de06be35d610d74332de728fdaa3991d7975f1905be7a613a8791dc3607753",
    "configs/v14_v11style_celebration_signflip_seed236.json": "27b9ef647487ac16be07e243c0dea8dddad198e8a21033f8d606af256a1c6994",
    "configs/v14_v11style_celebration_signflip_seed237.json": "7ca0803c28e767844432376d8199578fd57cb9b6d936e5ebb55c85b382dbdb89",
    "configs/v14_v11style_celebration_signflip_seed238.json": "df81702c595730be234fc0d9267bdaa7ad07080c1f1e7da03a2d78839ab800a1",
    "configs/v14_v11style_celebration_signflip_seed239.json": "fa991c073e307c4ec1dcde5de9e006620b33b08a12eaf7f04cdf03105f940aa4",
    REGISTRATION_PATH: REGISTRATION_SHA256,
    METRIC_SCHEMA_PATH: METRIC_SCHEMA_SHA256,
    SOURCE_CONFIG_PATH: SOURCE_CONFIG_SHA256,
    CALIBRATION_PATH: CALIBRATION_SHA256,
    "modal_emotional_tournament_v1.py": "18704ab325b666c8dc66b5a5c9e025ba9beb994e63f1d69cc61b86091a6a11fd",
    "pyproject.toml": "8d61fd00ddd948627960d85f5ca2998c4ae4198104bc613afb5a324923aaa823",
    "src/jlens_rl/common.py": "6e85491315e79c308b769e02514538caef9c3a5b06cb7a3e440c63e655f6d16e",
    "src/jlens_rl/eval.py": "0403e7c35f92af20f13bb471605e044cbb66079600530e4d420b42dc8a4fd578",
    "src/jlens_rl/reward.py": "e3ac96cbdfc8b0611e0917720d8e5aef379dc3049bbb7cc006229a078ac8cd45",
    "src/jlens_rl/train.py": "048ff415ce51b50e6e0dea5ae60986d5ea7e783c3c7027057196597a594d4167",
}
IMAGE_FILES = (*EXPECTED_FILE_SHA256, RUNNER_PATH)
FORBIDDEN_RUNTIME_NAMES = (
    "sealed_final_indices.json",
    "future_reserve_indices.json",
    "retired_v3_curve_indices.json",
)
SCIENCE_KEYS_UNCHANGED_FROM_SEED195 = (
    "model_name",
    "model_revision",
    "dataset_revision",
    "lens_path",
    "lens_sha256",
    "expected_lens_sha256",
    "calibration_path",
    "calibration_sha256",
    "expected_calibration_sha256",
    "target_words",
    "train_examples",
    "validation_examples",
    "validation_batch_size",
    "num_generations",
    "num_generations_eval",
    "max_prompt_tokens",
    "max_new_tokens",
    "min_new_tokens",
    "temperature",
    "learning_rate",
    "kl_beta",
    "loss_type",
    "scale_rewards",
    "gradient_accumulation_steps",
    "lora_rank",
    "lora_alpha",
    "score_stride",
    "score_start_fraction",
    "score_layers",
    "score_aggregation",
    "score_include_final",
    "vocab_chunk_size",
    "mask_target_tokens",
    "early_stopping_patience",
    "early_stopping_min_delta",
    "eval_strategy",
    "validation_source",
    "validation_indices_path",
    "reserved_train_indices_path",
    "validation_observational_only",
    "require_clean_repository",
    "lr_scheduler_type",
    "warmup_steps",
    "warmup_ratio",
    "reward_type",
    "wandb_entity",
    "wandb_project",
    "wandb_mode",
    "wandb_resume",
)
PAIR_ALLOWED_DIFFERENCES = {
    "score_components",
    "run_name",
    "wandb_run_id",
    "wandb_url",
    "wandb_tags",
    "output_dir",
}
FLOAT32_DUPLICATE_MAX_ULPS = 4


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _float32_ulp_distance(left: float, right: float) -> int:
    """Return the representable binary32 distance for two nonnegative summaries."""
    if (
        not math.isfinite(float(left))
        or not math.isfinite(float(right))
        or float(left) < 0
        or float(right) < 0
    ):
        raise RuntimeError("float32 ULP comparison requires finite nonnegative values")
    try:
        left_bits = struct.unpack("!I", struct.pack("!f", float(left)))[0]
        right_bits = struct.unpack("!I", struct.pack("!f", float(right)))[0]
    except OverflowError as error:
        raise RuntimeError("float32 ULP comparison overflowed") from error
    return abs(left_bits - right_bits)


def _load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    base = value.pop("base", None)
    if base is None:
        return value
    result = _load_config(path.parent / base)
    result.update(value)
    return result


def _label(condition: str, seed: int) -> str:
    if condition not in CONDITIONS or seed not in SEEDS:
        raise ValueError(f"unregistered V14 run: {condition=} {seed=}")
    return f"{condition}_seed{seed}"


def expected_config(repository: Path, condition: str, seed: int) -> dict[str, Any]:
    _label(condition, seed)
    return _load_config(repository / CONFIG_PATHS[(condition, seed)])


def validate_config(repository: Path, condition: str, seed: int) -> dict[str, Any]:
    config = expected_config(repository, condition, seed)
    source = json.loads((repository / SOURCE_CONFIG_PATH).read_text())
    changed_science = {
        key: {"seed195": source.get(key), "v14": config.get(key)}
        for key in SCIENCE_KEYS_UNCHANGED_FROM_SEED195
        if config.get(key) != source.get(key)
    }
    if changed_science:
        raise RuntimeError(f"V14 changed seed195/V11 science: {changed_science}")
    expected_identity = {
        "seed": seed,
        "updates": 6,
        "eval_every": 1,
        "validation_steps": list(POST_BASELINE_STEPS),
        "save_every": 6,
        "save_total_limit": 1,
        "curve_manifest_sha256": CURVE_MANIFEST_SHA256,
        "train_exclusions_manifest_sha256": TRAIN_EXCLUSIONS_SHA256,
        "registration_sha256": REGISTRATION_SHA256,
        "metric_schema_path": METRIC_SCHEMA_PATH,
        "metric_schema_sha256": METRIC_SCHEMA_SHA256,
        "evidence_eligibility": (
            "development_only_posthoc_v11_style_no_protected_final"
        ),
        "wandb_group": WANDB_GROUP,
        "wandb_run_id": WANDB_IDS[(condition, seed)],
        "output_dir": f"/state/runs/{_label(condition, seed)}",
    }
    changed_identity = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected_identity.items()
        if config.get(key) != value
    }
    if changed_identity:
        raise RuntimeError(f"V14 config identity changed: {changed_identity}")
    expected_components = (
        TREATMENT_COMPONENTS if condition == "jlens" else CONTROL_COMPONENTS
    )
    if config.get("score_components") != [dict(item) for item in expected_components]:
        raise RuntimeError(f"V14 {_label(condition, seed)} score changed")
    peer = expected_config(
        repository, "signflip" if condition == "jlens" else "jlens", seed
    )
    if set(peer) != set(config):
        raise RuntimeError("matched treatment/control config schemas differ")
    unexpected_pair_differences = {
        key: {"this": config[key], "peer": peer[key]}
        for key in config
        if key not in PAIR_ALLOWED_DIFFERENCES and config[key] != peer[key]
    }
    if unexpected_pair_differences:
        raise RuntimeError(
            "matched treatment/control changed more than sign and identity: "
            f"{unexpected_pair_differences}"
        )
    return config


parallel_image = cached_tournament_image
for relative in IMAGE_FILES:
    parallel_image = parallel_image.add_local_file(
        LOCAL_REPO / relative,
        (REMOTE_REPO / relative).as_posix(),
        copy=True,
    )
parallel_image = (
    parallel_image.env(
        {
            "GIT_AUTHOR_NAME": "J-Lens V14 Runtime",
            "GIT_AUTHOR_EMAIL": "runtime@example.invalid",
            "GIT_COMMITTER_NAME": "J-Lens V14 Runtime",
            "GIT_COMMITTER_EMAIL": "runtime@example.invalid",
            "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+00:00",
            "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+00:00",
            "JLENS_REPOSITORY_ROOT": REMOTE_REPO.as_posix(),
            "JLENS_MODAL_IMAGE_SPEC": "j-lens-rl-development-v14-v11style-l40s-v1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    .run_commands(
        "find . -type d -name __pycache__ -prune -exec rm -rf {} +",
        "find . -type d -name '*.egg-info' -prune -exec rm -rf {} +",
        (
            "test -z \"$(find . -type f \\( "
            "-name sealed_final_indices.json -o "
            "-name future_reserve_indices.json -o "
            "-name retired_v3_curve_indices.json "
            "\\) -print -quit)\""
        ),
        "rm -rf .git",
        "git init -q",
        "git add -f .",
        "git commit -qm 'J-Lens V14 V11-style development runtime'",
        "test -z \"$(git status --porcelain=v1 --untracked-files=all)\"",
    )
)

app = modal.App(APP_NAME)
state_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False, version=2)
wandb_secret = modal.Secret.from_name(
    "j-lens-rl-wandb", required_keys=["WANDB_API_KEY"]
)


def _write_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())


def _replace_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _runtime_source_hashes(repository: Path) -> dict[str, str]:
    return {relative: _sha256(repository / relative) for relative in sorted(IMAGE_FILES)}


def _verify_runtime_files() -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in EXPECTED_FILE_SHA256.items():
        path = REMOTE_REPO / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"runtime input is absent or unsafe: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"runtime input changed: {relative}: {actual} != {expected}")
        observed[relative] = actual
    for name in FORBIDDEN_RUNTIME_NAMES:
        if any(path.is_file() for path in REMOTE_REPO.rglob(name)):
            raise RuntimeError(f"forbidden protected/reserve payload entered image: {name}")
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=REMOTE_REPO,
        text=True,
    )
    if status:
        raise RuntimeError(f"V14 runtime Git worktree is dirty: {status}")
    registration = json.loads((REMOTE_REPO / REGISTRATION_PATH).read_text())
    if (
        registration.get("protocol") != PROTOCOL
        or registration.get("scientific_status", {}).get("classification")
        != "development_only_posthoc_v11_style_replication"
        or registration.get("firewall", {}).get(
            "protected_final_payloads_mounted_or_accessed"
        )
        is not False
    ):
        raise RuntimeError("V14 registration classification or firewall changed")
    for condition in CONDITIONS:
        for seed in SEEDS:
            validate_config(REMOTE_REPO, condition, seed)
    return observed


def _read_claim(claim_id: str) -> dict[str, Any]:
    path = REMOTE_STATE / "attempt_claim.json"
    if not path.is_file():
        raise RuntimeError("V14 attempt has no durable claim")
    value = json.loads(path.read_text())
    if (
        value.get("claim_id") != claim_id
        or value.get("protocol") != PROTOCOL
        or value.get("registration_sha256") != REGISTRATION_SHA256
        or value.get("seeds") != list(SEEDS)
        or value.get("conditions") != list(CONDITIONS)
        or value.get("steps") != list(STEPS)
        or value.get("max_parallel_gpus") != MAX_PARALLEL_GPUS
        or value.get("protected_final_payloads_accessed") is not False
    ):
        raise RuntimeError("V14 attempt claim changed")
    return value


def _intent_path(condition: str, seed: int) -> Path:
    return REMOTE_STATE / "dispatches" / f"{_label(condition, seed)}.intent.json"


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
        raise RuntimeError("registered V14 Volume is not fresh and empty")
    value = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "scientific_status": "development_only_posthoc_v11_style_replication",
        "claim_id": claim_id,
        "registration_sha256": REGISTRATION_SHA256,
        "metric_schema_sha256": METRIC_SCHEMA_SHA256,
        "seeds": list(SEEDS),
        "conditions": list(CONDITIONS),
        "labels": list(LABELS),
        "steps": list(STEPS),
        "v11_gate_steps": list(V11_GATE_STEPS),
        "max_parallel_gpus": MAX_PARALLEL_GPUS,
        "workspace_gpu_limit": WORKSPACE_GPU_LIMIT,
        "wandb_run_ids": {
            _label(condition, seed): WANDB_IDS[(condition, seed)]
            for condition in CONDITIONS
            for seed in SEEDS
        },
        "aggregate_wandb_run_id": AGGREGATE_WANDB_ID,
        "preflight": preflight,
        "retry_resume_warm_start_permitted": False,
        "closed_v11_v12_v13_state_mounted": False,
        "protected_final_payloads_accessed": False,
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
def record_launch_receipt(
    claim_id: str, app_id: str, root_call_id: str
) -> dict[str, Any]:
    state_volume.reload()
    _read_claim(claim_id)
    value = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "claim_id": claim_id,
        "app_id": app_id,
        "root_call_id": root_call_id,
        "gpu_type": GPU_TYPE,
        "max_parallel_gpus": MAX_PARALLEL_GPUS,
        "wandb_group": WANDB_GROUP,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_exclusive(REMOTE_STATE / "launch_receipt.json", value)
    state_volume.commit()
    return value


def _wait_for_launch_receipt(claim_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        state_volume.reload()
        path = REMOTE_STATE / "launch_receipt.json"
        if path.is_file():
            value = json.loads(path.read_text())
            if value.get("claim_id") == claim_id:
                return value
        time.sleep(1)
    raise RuntimeError("durable V14 launch receipt did not arrive")


def validate_history_rows(rows: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    if [row.get("step") for row in rows] != list(STEPS):
        raise RuntimeError("validation history must contain exact ordered steps 0..6")
    result: dict[int, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        exact = row.get("exact_match")
        if (
            isinstance(exact, bool)
            or not isinstance(exact, (int, float))
            or not math.isfinite(float(exact))
            or not 0 <= float(exact) <= 1
            or row.get("validation_source") != "train"
            or row.get("validation_indices_sha256") != CURVE_MANIFEST_SHA256
        ):
            raise RuntimeError("validation history contains an invalid row")
        result[int(row["step"])] = row
    return result


def verify_log_history(
    logs: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    history: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    if not isinstance(logs, list) or any(not isinstance(row, dict) for row in logs):
        raise RuntimeError("Trainer log history must be a JSON row list")
    reward_rows = [row for row in logs if "reward" in row]
    if [row.get("step") for row in reward_rows] != list(POST_BASELINE_STEPS):
        raise RuntimeError("Trainer history lacks exact reward steps 1..6")
    reward_mean_key = "rewards/jlens_yay_great_success_nice_reward/mean"
    reward_std_key = "rewards/jlens_yay_great_success_nice_reward/std"
    literal_key = "jlens/yay_great_success_nice_literal_rate"
    for row in reward_rows:
        named_means = [
            key
            for key in row
            if key.startswith("rewards/") and key.endswith("/mean")
        ]
        forbidden_reward_keys = [
            key
            for key in row
            if "gsm8k" in key.lower() or "correctness" in key.lower()
        ]
        required_numeric = (
            "reward",
            "reward_std",
            reward_mean_key,
            reward_std_key,
            literal_key,
        )
        if (
            named_means != [reward_mean_key]
            or forbidden_reward_keys
            or any(
                isinstance(row.get(key), bool)
                or not isinstance(row.get(key), (int, float))
                or not math.isfinite(float(row[key]))
                for key in required_numeric
            )
            or not math.isclose(
                float(row["reward"]),
                float(row[reward_mean_key]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or float(row["reward_std"]) < 0
            or float(row[reward_std_key]) < 0
            or _float32_ulp_distance(
                float(row["reward_std"]), float(row[reward_std_key])
            )
            > FLOAT32_DUPLICATE_MAX_ULPS
            or not 0 <= float(row[literal_key]) <= 1
        ):
            raise RuntimeError("Trainer history is not an exact one-J-reward run")
        validation_merged = "validation/exact_match" in row
        if "learning_rate" in row:
            learning_rate = row["learning_rate"]
            if (
                isinstance(learning_rate, bool)
                or not isinstance(learning_rate, (int, float))
                or not math.isfinite(float(learning_rate))
                or not math.isclose(
                    float(learning_rate),
                    float(config["learning_rate"]),
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
            ):
                raise RuntimeError("Trainer history changed the constant learning rate")
        elif not validation_merged:
            raise RuntimeError(
                "learning_rate may be absent only on a validation-merged reward row"
            )
    validation_rows = [row for row in logs if "validation/exact_match" in row]
    if [row.get("step") for row in validation_rows] != list(POST_BASELINE_STEPS):
        raise RuntimeError(
            "Trainer validation rows must be exactly steps 1..6; baseline 0 is pre-train"
        )
    for row in validation_rows:
        step = int(row["step"])
        exact = row.get("validation/exact_match")
        if (
            isinstance(exact, bool)
            or not isinstance(exact, (int, float))
            or not math.isfinite(float(exact))
            or float(exact) != float(history[step]["exact_match"])
        ):
            raise RuntimeError("Trainer validation rows disagree with authoritative history")
    terminal_rows = [row for row in logs if "train_runtime" in row]
    if len(terminal_rows) != 1 or terminal_rows[0].get("step") != 6:
        raise RuntimeError("Trainer history lacks its unique step-6 terminal row")
    return {
        "optimizer_steps": len(reward_rows),
        "validation_steps": [int(row["step"]) for row in validation_rows],
        "baseline_absent_from_trainer_log": True,
        "learning_rate_rows": sum("learning_rate" in row for row in reward_rows),
        "one_j_reward_verified": True,
    }


def _verify_training_outputs(condition: str, seed: int) -> dict[str, Any]:
    label = _label(condition, seed)
    run_dir = REMOTE_STATE / "runs" / label
    required = {
        "validation_history.jsonl",
        "log_history.json",
        "resolved_config.json",
        "run_result_manifest.json",
        "wandb_terminal_publish_receipt.json",
        "run_manifest.json",
        "data_indices.json",
        "environment_snapshot.json",
    }
    if any(not (run_dir / name).is_file() for name in required):
        raise RuntimeError(f"{label} lacks terminal public evidence")
    config = expected_config(REMOTE_REPO, condition, seed)
    resolved_path = run_dir / "resolved_config.json"
    if json.loads(resolved_path.read_text()) != config:
        raise RuntimeError(f"{label} resolved config changed")
    history_path = run_dir / "validation_history.jsonl"
    history_rows = [
        json.loads(line) for line in history_path.read_text().splitlines() if line.strip()
    ]
    history = validate_history_rows(history_rows)
    logs = json.loads((run_dir / "log_history.json").read_text())
    trainer_verification = verify_log_history(logs, config, history)
    curve_manifest = json.loads((REMOTE_REPO / CURVE_MANIFEST_PATH).read_text())
    exclusions_manifest = json.loads((REMOTE_REPO / TRAIN_EXCLUSIONS_PATH).read_text())
    data = json.loads((run_dir / "data_indices.json").read_text())
    train_indices = data.get("train_source_indices")
    if (
        not isinstance(train_indices, list)
        or len(train_indices) != 1000
        or len(set(train_indices)) != 1000
        or data.get("validation_source") != "train"
        or data.get("validation_source_indices") != curve_manifest.get("indices")
        or set(train_indices) & set(exclusions_manifest.get("indices", []))
    ):
        raise RuntimeError(f"{label} changed the training/development firewall")
    run_manifest_path = run_dir / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text())
    expected_wandb = {
        "entity": WANDB_ENTITY,
        "project": WANDB_PROJECT,
        "run_name": WANDB_IDS[(condition, seed)],
        "run_id": WANDB_IDS[(condition, seed)],
        "url": config["wandb_url"],
        "group": WANDB_GROUP,
        "tags": config["wandb_tags"],
        "resume": "never",
    }
    if (
        run_manifest.get("git_dirty") is not False
        or run_manifest.get("config_sha256")
        != EXPECTED_FILE_SHA256[CONFIG_PATHS[(condition, seed)]]
        or run_manifest.get("resolved_config_sha256") != _sha256(resolved_path)
        or run_manifest.get("reward_type") != "jlens"
        or run_manifest.get("lens_sha256") != LENS_SHA256
        or run_manifest.get("calibration_sha256") != CALIBRATION_SHA256
        or run_manifest.get("wandb_identity") != expected_wandb
        or run_manifest.get("confirmatory_identity", {}).get("registration_sha256")
        != REGISTRATION_SHA256
        or run_manifest.get("data_indices_sha256")
        != _sha256(run_dir / "data_indices.json")
    ):
        raise RuntimeError(f"{label} source/config/reward identity changed")
    result_path = run_dir / "run_result_manifest.json"
    result = json.loads(result_path.read_text())
    receipt_path = run_dir / "wandb_terminal_publish_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if (
        result.get("completed_updates") != 6
        or result.get("registration_sha256") != REGISTRATION_SHA256
        or result.get("evidence_eligibility")
        != "development_only_posthoc_v11_style_no_protected_final"
        or result.get("lens_sha256") != LENS_SHA256
        or result.get("calibration_sha256") != CALIBRATION_SHA256
        or result.get("wandb_identity") != expected_wandb
        or receipt.get("terminal_run_result_sha256") != _sha256(result_path)
        or receipt.get("wandb_identity") != expected_wandb
        or receipt.get("observed_wandb_identity")
        != {key: expected_wandb[key] for key in expected_wandb if key != "resume"}
    ):
        raise RuntimeError(f"{label} terminal or W&B identity changed")
    return {
        "label": label,
        "condition": condition,
        "seed": seed,
        "config_sha256": EXPECTED_FILE_SHA256[CONFIG_PATHS[(condition, seed)]],
        "validation_history_sha256": _sha256(history_path),
        "log_history_sha256": _sha256(run_dir / "log_history.json"),
        "run_result_manifest_sha256": _sha256(result_path),
        "wandb_terminal_publish_receipt_sha256": _sha256(receipt_path),
        "wandb_url": config["wandb_url"],
        "trainer_verification": trainer_verification,
        "curve": {str(step): float(history[step]["exact_match"]) for step in STEPS},
    }


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
def train_run(condition: str, seed: int, claim_id: str) -> dict[str, Any]:
    label = _label(condition, seed)
    state_volume.reload()
    _read_claim(claim_id)
    intent_path = _intent_path(condition, seed)
    if not intent_path.is_file():
        raise RuntimeError(f"{label} has no durable pre-dispatch intent")
    intent = json.loads(intent_path.read_text())
    if (
        intent.get("claim_id") != claim_id
        or intent.get("condition") != condition
        or intent.get("seed") != seed
        or intent.get("label") != label
        or intent.get("config_sha256")
        != EXPECTED_FILE_SHA256[CONFIG_PATHS[(condition, seed)]]
        or intent.get("status") != "written_before_any_gpu_spawn"
    ):
        raise RuntimeError(f"{label} pre-dispatch intent changed")
    source_hashes = _verify_runtime_files()
    import torch

    if torch.cuda.device_count() != 1 or GPU_TYPE not in torch.cuda.get_device_name(0):
        raise RuntimeError(f"{label} did not receive exactly one registered L40S")
    run_dir = REMOTE_STATE / "runs" / label
    if run_dir.exists():
        raise RuntimeError(f"{label} already has output; resume/retry is forbidden")
    command = [
        sys.executable,
        "-m",
        "jlens_rl.train",
        "--config",
        CONFIG_PATHS[(condition, seed)],
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
    if completed.returncode:
        failure = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "claim_id": claim_id,
            "label": label,
            "returncode": completed.returncode,
            "intent_sha256": _sha256(intent_path),
            "stdout_sha256": _sha256(stdout_path),
            "stderr_sha256": _sha256(stderr_path),
            "retry_resume_warm_start_permitted": False,
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_exclusive(REMOTE_STATE / "dispatches" / f"{label}.failure.json", failure)
        state_volume.commit()
        raise RuntimeError(f"V14 training failed closed for {label}")
    verified = _verify_training_outputs(condition, seed)
    completion = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "claim_id": claim_id,
        "intent_sha256": _sha256(intent_path),
        "source_file_sha256": source_hashes,
        "stdout_sha256": _sha256(stdout_path),
        "stderr_sha256": _sha256(stderr_path),
        "status": "terminal_public_run_verified",
        **verified,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_exclusive(
        REMOTE_STATE / "dispatches" / f"{label}.completion.json", completion
    )
    state_volume.commit()
    return completion


def _sample_sem(values: Sequence[float]) -> float:
    return statistics.stdev(values) / math.sqrt(len(values))


def exact_two_sided_sign_test(effects: Sequence[float]) -> dict[str, Any]:
    values = [float(value) for value in effects]
    if any(not math.isfinite(value) for value in values):
        raise RuntimeError("sign test received a non-finite effect")
    positives = sum(value > 0 for value in values)
    negatives = sum(value < 0 for value in values)
    ties = len(values) - positives - negatives
    nonzero = positives + negatives
    if nonzero == 0:
        p_value = 1.0
    else:
        tail = min(positives, negatives)
        p_value = min(
            1.0,
            2.0
            * sum(math.comb(nonzero, k) for k in range(tail + 1))
            / (2**nonzero),
        )
    return {
        "effects": values,
        "positives": positives,
        "negatives": negatives,
        "ties": ties,
        "nonzero": nonzero,
        "mean_effect": statistics.fmean(values),
        "exact_two_sided_p": p_value,
        "nominal_alpha": 0.15,
        "success": (
            positives == len(values)
            and negatives == 0
            and ties == 0
            and statistics.fmean(values) > 0
            and p_value <= 0.15
        ),
    }


def aggregate_results(
    results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if set(results) != set(LABELS):
        raise RuntimeError("aggregate requires exactly the eight registered runs")
    rows: list[dict[str, Any]] = []
    for step in STEPS:
        treatment = [
            float(results[_label("jlens", seed)]["curve"][str(step)])
            for seed in SEEDS
        ]
        control = [
            float(results[_label("signflip", seed)]["curve"][str(step)])
            for seed in SEEDS
        ]
        paired = [left - right for left, right in zip(treatment, control, strict=True)]
        rows.append(
            {
                "global_step": step,
                "n_seed_pairs": len(SEEDS),
                "treatment_mean": statistics.fmean(treatment),
                "treatment_sem": _sample_sem(treatment),
                "signflip_mean": statistics.fmean(control),
                "signflip_sem": _sample_sem(control),
                "paired_mean": statistics.fmean(paired),
                "paired_sem": _sample_sem(paired),
                "treatment_by_seed": {
                    str(seed): value for seed, value in zip(SEEDS, treatment, strict=True)
                },
                "signflip_by_seed": {
                    str(seed): value for seed, value in zip(SEEDS, control, strict=True)
                },
                "paired_effect_by_seed": {
                    str(seed): value for seed, value in zip(SEEDS, paired, strict=True)
                },
            }
        )
    by_step = {row["global_step"]: row for row in rows}
    treatment_terminal = [
        by_step[6]["treatment_by_seed"][str(seed)] for seed in SEEDS
    ]
    signflip_terminal = [
        by_step[6]["signflip_by_seed"][str(seed)] for seed in SEEDS
    ]
    treatment_baseline = [
        by_step[0]["treatment_by_seed"][str(seed)] for seed in SEEDS
    ]
    treatment_control_effects = [
        left - right
        for left, right in zip(treatment_terminal, signflip_terminal, strict=True)
    ]
    treatment_baseline_effects = [
        terminal - baseline
        for terminal, baseline in zip(
            treatment_terminal, treatment_baseline, strict=True
        )
    ]
    shape_gate = {
        "steps": list(V11_GATE_STEPS),
        "means": [by_step[step]["treatment_mean"] for step in V11_GATE_STEPS],
    }
    shape_gate["first_above_initial"] = (
        by_step[4]["treatment_mean"] > by_step[0]["treatment_mean"]
    )
    shape_gate["no_downward_steps_4_to_5_to_6"] = (
        by_step[5]["treatment_mean"] >= by_step[4]["treatment_mean"]
        and by_step[6]["treatment_mean"] >= by_step[5]["treatment_mean"]
    )
    shape_gate["passed"] = bool(
        shape_gate["first_above_initial"]
        and shape_gate["no_downward_steps_4_to_5_to_6"]
    )
    primary = exact_two_sided_sign_test(treatment_control_effects)
    secondary = exact_two_sided_sign_test(treatment_baseline_effects)
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "scientific_status": "development_only_posthoc_v11_style_replication",
        "canonical_x_axis": "global_step",
        "seeds": list(SEEDS),
        "rows": rows,
        "v11_shape_gate": shape_gate,
        "terminal_treatment_minus_signflip": primary,
        "terminal_treatment_minus_baseline": secondary,
        "target_evidence_met": bool(
            shape_gate["passed"] and primary["success"] and secondary["success"]
        ),
        "multiplicity_caveat": (
            "Nominal development evidence only; prior visible candidate, seed, "
            "horizon, V11, V12, and seed195 outcomes are not familywise-corrected."
        ),
        "protected_final_payloads_accessed": False,
    }


def _write_aggregate_files(aggregate: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    evidence = REMOTE_STATE / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    json_path = evidence / "v14_v11style_aggregate.json"
    csv_path = evidence / "v14_v11style_curve.csv"
    png_path = evidence / "v14_v11style_curve.png"
    _write_exclusive(json_path, aggregate)
    fieldnames = [
        "global_step",
        "treatment_mean",
        "treatment_sem",
        "signflip_mean",
        "signflip_sem",
        "paired_mean",
        "paired_sem",
        *(f"jlens_seed{seed}" for seed in SEEDS),
        *(f"signflip_seed{seed}" for seed in SEEDS),
        *(f"paired_seed{seed}" for seed in SEEDS),
    ]
    temporary = csv_path.with_name(f".{csv_path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in aggregate["rows"]:
            writer.writerow(
                {
                    **{key: row[key] for key in fieldnames[:7]},
                    **{
                        f"jlens_seed{seed}": row["treatment_by_seed"][str(seed)]
                        for seed in SEEDS
                    },
                    **{
                        f"signflip_seed{seed}": row["signflip_by_seed"][str(seed)]
                        for seed in SEEDS
                    },
                    **{
                        f"paired_seed{seed}": row["paired_effect_by_seed"][str(seed)]
                        for seed in SEEDS
                    },
                }
            )
    temporary.replace(csv_path)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = aggregate["rows"]
    steps = [row["global_step"] for row in rows]
    figure, (curve_axis, effect_axis) = plt.subplots(
        2, 1, figsize=(8.5, 8.0), sharex=True, constrained_layout=True
    )
    curve_axis.errorbar(
        steps,
        [row["treatment_mean"] for row in rows],
        yerr=[row["treatment_sem"] for row in rows],
        marker="o",
        linewidth=2,
        capsize=3,
        label="celebration J-lens treatment",
    )
    curve_axis.errorbar(
        steps,
        [row["signflip_mean"] for row in rows],
        yerr=[row["signflip_sem"] for row in rows],
        marker="o",
        linewidth=2,
        capsize=3,
        label="matched negative sign-flip",
    )
    curve_axis.axhline(
        rows[0]["treatment_mean"],
        color="black",
        linestyle="--",
        linewidth=1,
        label="initial treatment mean",
    )
    curve_axis.set_ylabel("GSM8K exact match")
    curve_axis.set_title("V11-style celebration reward: exposed development curve")
    curve_axis.grid(alpha=0.25)
    curve_axis.legend()
    effect_axis.errorbar(
        steps,
        [row["paired_mean"] for row in rows],
        yerr=[row["paired_sem"] for row in rows],
        marker="o",
        linewidth=2,
        capsize=3,
        color="tab:green",
    )
    effect_axis.axhline(0.0, color="black", linestyle="--", linewidth=1)
    effect_axis.set_xlabel("optimizer global step")
    effect_axis.set_ylabel("treatment - sign-flip")
    effect_axis.grid(alpha=0.25)
    figure.savefig(png_path, dpi=180)
    plt.close(figure)
    return json_path, csv_path, png_path


def _publish_aggregate_to_wandb(
    aggregate: Mapping[str, Any], paths: Sequence[Path]
) -> dict[str, Any]:
    import wandb

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        id=AGGREGATE_WANDB_ID,
        name=AGGREGATE_WANDB_ID,
        group=WANDB_GROUP,
        tags=[
            "development-only",
            "posthoc-v11-style",
            "aggregate",
            "celebration-family",
            "matched-signflip",
            "dense-u1-h6",
        ],
        resume="never",
        config={
            "protocol": PROTOCOL,
            "registration_sha256": REGISTRATION_SHA256,
            "metric_schema_sha256": METRIC_SCHEMA_SHA256,
            "seeds": list(SEEDS),
            "steps": list(STEPS),
            "canonical_x_axis": "global_step",
            "wandb_internal_step_is_optimizer_step": False,
            "scientific_status": "development_only_posthoc_v11_style_replication",
        },
    )
    if run is None:
        raise RuntimeError("W&B aggregate run did not initialize")
    try:
        run.define_metric("global_step")
        run.define_metric("validation/*", step_metric="global_step")
        for row in aggregate["rows"]:
            run.log(
                {
                    "global_step": row["global_step"],
                    "validation/treatment_mean": row["treatment_mean"],
                    "validation/treatment_sem": row["treatment_sem"],
                    "validation/signflip_mean": row["signflip_mean"],
                    "validation/signflip_sem": row["signflip_sem"],
                    "validation/paired_mean": row["paired_mean"],
                    "validation/paired_sem": row["paired_sem"],
                    **{
                        f"validation/jlens_seed{seed}": row["treatment_by_seed"][
                            str(seed)
                        ]
                        for seed in SEEDS
                    },
                    **{
                        f"validation/signflip_seed{seed}": row["signflip_by_seed"][
                            str(seed)
                        ]
                        for seed in SEEDS
                    },
                }
            )
        run.summary["v11_shape_gate_passed"] = aggregate["v11_shape_gate"]["passed"]
        run.summary["terminal_treatment_minus_signflip_p"] = aggregate[
            "terminal_treatment_minus_signflip"
        ]["exact_two_sided_p"]
        run.summary["terminal_treatment_minus_baseline_p"] = aggregate[
            "terminal_treatment_minus_baseline"
        ]["exact_two_sided_p"]
        run.summary["target_evidence_met"] = aggregate["target_evidence_met"]
        artifact = wandb.Artifact(
            f"{AGGREGATE_WANDB_ID}-evidence",
            type="development-replication-evidence",
            metadata={
                "protocol": PROTOCOL,
                "registration_sha256": REGISTRATION_SHA256,
                "scientific_status": (
                    "development_only_posthoc_v11_style_replication"
                ),
            },
        )
        for path in paths:
            artifact.add_file(str(path), name=path.name)
        logged = run.log_artifact(artifact, aliases=["latest", "v11-style"])
        completed = logged.wait()
        if completed is not None:
            logged = completed
        return {
            "run_id": run.id,
            "url": run.url,
            "group": WANDB_GROUP,
            "artifact_id": logged.id,
            "artifact_name": logged.name,
            "artifact_version": logged.version,
            "artifact_digest": logged.digest,
        }
    finally:
        run.finish()


@app.function(
    image=parallel_image,
    cpu=2,
    memory=4096,
    max_containers=1,
    timeout=4 * 60 * 60,
    startup_timeout=60 * 60,
    retries=0,
    secrets=[wandb_secret],
    volumes={REMOTE_STATE: state_volume},
)
def orchestrate(claim_id: str) -> dict[str, Any]:
    status_path = REMOTE_STATE / "attempt_status.json"
    try:
        receipt = _wait_for_launch_receipt(claim_id)
        root_call_id = modal.current_function_call_id()
        if receipt.get("root_call_id") != root_call_id:
            raise RuntimeError("V14 orchestrator lacks durable root authority")
        _read_claim(claim_id)
        for slot, (condition, seed) in enumerate(
            (condition, seed) for condition in CONDITIONS for seed in SEEDS
        ):
            label = _label(condition, seed)
            intent = {
                "schema_version": 1,
                "protocol": PROTOCOL,
                "claim_id": claim_id,
                "root_call_id": root_call_id,
                "slot": slot,
                "condition": condition,
                "seed": seed,
                "label": label,
                "config": CONFIG_PATHS[(condition, seed)],
                "config_sha256": EXPECTED_FILE_SHA256[
                    CONFIG_PATHS[(condition, seed)]
                ],
                "wandb_run_id": WANDB_IDS[(condition, seed)],
                "status": "written_before_any_gpu_spawn",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            _write_exclusive(_intent_path(condition, seed), intent)
        state_volume.commit()
        calls = {
            (condition, seed): train_run.spawn(condition, seed, claim_id)
            for condition in CONDITIONS
            for seed in SEEDS
        }
        _replace_json(
            status_path,
            {
                "schema_version": 1,
                "protocol": PROTOCOL,
                "claim_id": claim_id,
                "stage": "all_eight_spawned_capacity_queue_allowed",
                "max_parallel_gpus": MAX_PARALLEL_GPUS,
                "workspace_gpu_limit": WORKSPACE_GPU_LIMIT,
                "worker_call_ids": {
                    _label(condition, seed): call.object_id
                    for (condition, seed), call in calls.items()
                },
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        state_volume.commit()
        results: dict[str, dict[str, Any]] = {}
        outcomes: dict[str, dict[str, Any]] = {}
        failures: list[str] = []
        for condition in CONDITIONS:
            for seed in SEEDS:
                label = _label(condition, seed)
                try:
                    completion = calls[(condition, seed)].get()
                    results[label] = completion
                    outcomes[label] = {
                        "status": "success",
                        "worker_call_id": calls[(condition, seed)].object_id,
                        "completion_sha256": _canonical_sha256(completion),
                    }
                except BaseException as error:
                    message = f"{label}: {type(error).__name__}: {error}"
                    failures.append(message)
                    outcomes[label] = {
                        "status": "failure",
                        "worker_call_id": calls[(condition, seed)].object_id,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
        state_volume.reload()
        _write_exclusive(
            REMOTE_STATE / "dispatches" / "all_worker_outcomes.json",
            {
                "schema_version": 1,
                "protocol": PROTOCOL,
                "claim_id": claim_id,
                "every_spawned_call_drained": True,
                "outcomes": outcomes,
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        state_volume.commit()
        if failures:
            raise RuntimeError(
                "V14 workers failed after all eight calls were drained: "
                + " | ".join(failures)
            )
        aggregate = aggregate_results(results)
        paths = _write_aggregate_files(aggregate)
        state_volume.commit()
        wandb_receipt = _publish_aggregate_to_wandb(aggregate, paths)
        aggregate_receipt = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "claim_id": claim_id,
            "file_sha256": {path.name: _sha256(path) for path in paths},
            "wandb": wandb_receipt,
            "canonical_x_axis": "global_step",
            "wandb_internal_step_is_optimizer_step": False,
            "published_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_exclusive(
            REMOTE_STATE / "evidence" / "aggregate_publish_receipt.json",
            aggregate_receipt,
        )
        summary = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "scientific_status": "development_only_posthoc_v11_style_replication",
            "claim_id": claim_id,
            "stage": "complete",
            "results": results,
            "aggregate": aggregate,
            "aggregate_publish_receipt": aggregate_receipt,
            "closed_v11_v12_v13_state_mounted": False,
            "protected_final_payloads_accessed": False,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_exclusive(REMOTE_STATE / "evidence" / "summary.json", summary)
        _replace_json(status_path, summary)
        state_volume.commit()
        return summary
    except BaseException as error:
        try:
            state_volume.reload()
            _replace_json(
                status_path,
                {
                    "schema_version": 1,
                    "protocol": PROTOCOL,
                    "claim_id": claim_id,
                    "stage": "failed_closed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "retry_started_training_run_permitted": False,
                    "protected_final_payloads_accessed": False,
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            state_volume.commit()
        except BaseException:
            pass
        raise


def _local_preflight() -> dict[str, Any]:
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=LOCAL_REPO,
        text=True,
    )
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=LOCAL_REPO, text=True
    ).strip()
    pushed = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=LOCAL_REPO, text=True
    ).strip()
    if status or head != pushed:
        raise RuntimeError("V14 launch requires an exact clean pushed main")
    for relative, expected in EXPECTED_FILE_SHA256.items():
        path = LOCAL_REPO / relative
        if not path.is_file() or path.is_symlink() or _sha256(path) != expected:
            raise RuntimeError(f"local registered input changed: {relative}")
    for condition in CONDITIONS:
        for seed in SEEDS:
            validate_config(LOCAL_REPO, condition, seed)
    modal_cli = Path(sys.executable).with_name("modal")
    listing_text = subprocess.check_output(
        [str(modal_cli), "app", "list", "--json"], text=True
    )
    listing = json.loads(listing_text[listing_text.index("[") :])
    active_apps = [
        {
            "app_id": item.get("app_id"),
            "description": item.get("description"),
            "state": item.get("state"),
        }
        for item in listing
        if item.get("stopped_at") is None and item.get("state") != "stopped"
    ]
    state_volume.hydrate()
    inventory_text = subprocess.check_output(
        [str(modal_cli), "volume", "ls", VOLUME_NAME, "/", "--json"], text=True
    )
    inventory = json.loads(inventory_text[inventory_text.index("[") :])
    if inventory:
        raise RuntimeError("V14 Volume must be fresh and empty")
    source_hashes = _runtime_source_hashes(LOCAL_REPO)
    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_main_commit": head,
        "source_tree_sha256": _canonical_sha256(source_hashes),
        "source_file_sha256": source_hashes,
        "registration_sha256": REGISTRATION_SHA256,
        "metric_schema_sha256": METRIC_SCHEMA_SHA256,
        "volume_name": VOLUME_NAME,
        "volume_object_id": state_volume.object_id,
        "volume_version": 2,
        "gpu_type": GPU_TYPE,
        "function_max_containers": MAX_PARALLEL_GPUS,
        "workspace_gpu_limit": WORKSPACE_GPU_LIMIT,
        "active_apps_recorded_not_mutated": active_apps,
        "closed_v11_v12_v13_state_mounted": False,
        "protected_final_payloads_inspected": False,
    }


@app.local_entrypoint()
def main() -> None:
    preflight = _local_preflight()
    claim_id = uuid.uuid4().hex
    claim_attempt.remote(claim_id, preflight)
    call = orchestrate.spawn(claim_id)
    receipt = record_launch_receipt.remote(
        claim_id, app.app_id or APP_NAME, call.object_id
    )
    print(
        json.dumps(
            {
                "status": "submitted",
                "scientific_status": (
                    "development_only_posthoc_v11_style_replication"
                ),
                "claim_id": claim_id,
                "root_call_id": call.object_id,
                "app_id": app.app_id,
                "volume": VOLUME_NAME,
                "gpu_type": GPU_TYPE,
                "function_max_containers": MAX_PARALLEL_GPUS,
                "workspace_gpu_limit": WORKSPACE_GPU_LIMIT,
                "seeds": list(SEEDS),
                "conditions": list(CONDITIONS),
                "steps": list(STEPS),
                "v11_gate_steps": list(V11_GATE_STEPS),
                "wandb_group": WANDB_GROUP,
                "wandb_run_ids": {
                    _label(condition, seed): WANDB_IDS[(condition, seed)]
                    for condition in CONDITIONS
                    for seed in SEEDS
                },
                "aggregate_wandb_run_id": AGGREGATE_WANDB_ID,
                "canonical_wandb_x_axis": "train/global_step",
                "preflight": preflight,
                "launch_receipt": receipt,
                "closed_v11_v12_v13_state_mounted": False,
                "protected_final_payloads_accessed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
