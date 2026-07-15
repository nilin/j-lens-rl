"""Run 16 fresh seed-pairs of the extended V14 celebration recipe.

Every treatment and matched sign-flip run is evaluated at the complete fixed
optimizer-step grid 0,2,4,6,8,10. No measured node is omitted from histories,
aggregates, W&B, or the plot. This development runner never mounts or reads a
protected-final, reserve, or predecessor state Volume.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
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
APP_NAME = "j-lens-rl-development-v16-v14-celebration-n16-20260715a"
VOLUME_NAME = APP_NAME
PROTOCOL = "j-lens-rl-development-v16-v14-celebration-n16-u2-h10"
REGISTRATION_PATH = "protocol_archive/v16_v14_manyseed_curve_registration.json"
REGISTRATION_SHA256 = "14950bc24a3d61902ea7c1c205b568b1399358c0fbd42f292d081bc1d2fe49f5"
METRIC_SCHEMA_PATH = "protocol_archive/v16_v14_manyseed_curve_metric_schema.json"
METRIC_SCHEMA_SHA256 = "8f3c814334fedcf1e02f32c6622091638f8a92d8598a6b3b32c942271ca52b4d"
V14_AGGREGATE_PATH = (
    "protocol_archive/v14_v11style_public_evidence/evidence/"
    "v14_v11style_aggregate.json"
)
V14_AGGREGATE_SHA256 = "7bb3a49f49c54ed438b7ffd0f6f3f277df994fc525abc838988b6c7352e012d1"
PREEMPTION_CLOSEOUT_PATH = "protocol_archive/v15_celebration_h5_preemption_closeout.json"
PREEMPTION_CLOSEOUT_SHA256 = "33bc8b4db1b19cab5417cc2623fbb6bf6be108e4e31e3cc0166f2db7c49e502e"
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
MAX_PARALLEL_GPUS = 4
USER_REQUESTED_GPU_CAP = 4
SEEDS = tuple(range(248, 264))
CONDITIONS = ("jlens", "signflip")
RUN_ORDER = tuple((condition, seed) for seed in SEEDS for condition in CONDITIONS)
OPTIMIZER_STEPS = tuple(range(1, 11))
STEPS = (0, 2, 4, 6, 8, 10)
POST_BASELINE_STEPS = STEPS[1:]
DISPLAY_GATE_STEPS = (0, 2, 4, 6)
WANDB_ENTITY = "nilinabra-spare-time"
WANDB_PROJECT = "j-lens-rl"
WANDB_GROUP = "dev-v16-v14-celebration-n16-u2-h10"
AGGREGATE_WANDB_ID = "dev-v16-v14-celebration-aggregate"
RUNNER_PATH = "modal_v16_v14_manyseed_curve.py"
COMMON_CONFIG_PATH = "configs/v16_v14_manyseed_curve_common.json"
TEMPLATE_PATHS = {
    condition: f"configs/v16_v14_manyseed_curve_{condition}_template.json"
    for condition in CONDITIONS
}
CONFIG_PATHS = {
    (condition, seed): (
        f"configs/v16_v14_manyseed_curve_{condition}_seed{seed}.json"
    )
    for condition in CONDITIONS
    for seed in SEEDS
}
WANDB_IDS = {
    (condition, seed): f"dev-v16-v14-celebration-{condition}-seed{seed}"
    for condition in CONDITIONS
    for seed in SEEDS
}
LABELS = tuple(f"{condition}_seed{seed}" for condition, seed in RUN_ORDER)
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
    ".gitignore": "0d3d2ae95afbc3a4efd65a2105913dd76679fb1dcb59836d791170b91a1fac81",
    CURVE_MANIFEST_PATH: CURVE_MANIFEST_SHA256,
    TRAIN_EXCLUSIONS_PATH: TRAIN_EXCLUSIONS_SHA256,
    LENS_PATH: LENS_SHA256,
    "configs/common.json": "c397905b4d4ac0cc64d7924d304b6aede4dc831d7d6e2a8b5622b63099266960",
    "configs/emotional_parallel_v3_common.json": "d4e8b8495b5df4b91a3110ef0baab08c1dcda1a5ca88b00fc4b45b099ba133ef",
    "configs/v16_v14_manyseed_curve_common.json": "0a29bc053035f24b0e6b151546f7f0fc8b205b9741ce308dabb6ee7332506c19",
    "configs/v16_v14_manyseed_curve_jlens_seed248.json": "0f12f9857397bab2f72185e4d9d26ffc46319755c0ab37ae99448100bceebd7b",
    "configs/v16_v14_manyseed_curve_jlens_seed249.json": "24002543152886466e02f6216cac77297847563ce460f2790fb5b33052c854e9",
    "configs/v16_v14_manyseed_curve_jlens_seed250.json": "c8ceb2b7f893f9630c68717448c13df681a6f6d8ea4685a0c9e5f299d8f4243b",
    "configs/v16_v14_manyseed_curve_jlens_seed251.json": "dd8134bd8d6a5d5b7c43414a3a5874f2030e32054b26887a87afc7f4351f4a9a",
    "configs/v16_v14_manyseed_curve_jlens_seed252.json": "9a577aa741499ee7e2add1c457e290caa3d8f181f280aef2d936e6cc3097edc0",
    "configs/v16_v14_manyseed_curve_jlens_seed253.json": "2c5101819b5b6edc0899f5ba0f9c5c951607de90e84742338a57b640f198b71d",
    "configs/v16_v14_manyseed_curve_jlens_seed254.json": "b4cef51829af7898204f369ed04e500360c31edc708cda873edc3a6d107dc3f6",
    "configs/v16_v14_manyseed_curve_jlens_seed255.json": "6421bd38f6a82be838abed12871cf13d8b1259fdc34bff915d7e25715949b1b8",
    "configs/v16_v14_manyseed_curve_jlens_seed256.json": "10e830d47812499b0322b3801fd49c4472ececb1ef75de195451d861ca92a34e",
    "configs/v16_v14_manyseed_curve_jlens_seed257.json": "a832ff308c2265f265d6593591d017800b681e1446eafe998c58be3058d9b36c",
    "configs/v16_v14_manyseed_curve_jlens_seed258.json": "ac18590a7248fc40f1eefede1aac814d509d422c18f3f96d7d411e60ee2dead3",
    "configs/v16_v14_manyseed_curve_jlens_seed259.json": "6fa9aa53b14f3d17216cef2949747e15c53b0ce59660e1cb680093bae3a1edd4",
    "configs/v16_v14_manyseed_curve_jlens_seed260.json": "6c78d7de0e34703dae9daf2c56ee0274b4b851269e6ca2ea76e175bde33cb776",
    "configs/v16_v14_manyseed_curve_jlens_seed261.json": "680fde6ca8ebf8d4fe5145c06501358d8493c19189e92537434097df877fb5ec",
    "configs/v16_v14_manyseed_curve_jlens_seed262.json": "8fc680930ecc8ff4a392a80fd2c4a77cfbcb586e5f06a914e8365f797246924d",
    "configs/v16_v14_manyseed_curve_jlens_seed263.json": "16e121a97a0cde67885d5c8dc02c26ff6a65f70fc32cb6fb199416b3bb3600e2",
    "configs/v16_v14_manyseed_curve_jlens_template.json": "a97ccd7f1bfdf09e7e6b77bded3ef526aea4cf9a21b8ae48ee9e840b9d215206",
    "configs/v16_v14_manyseed_curve_signflip_seed248.json": "c16dbba5b73cc0ac54520751b87dfbc7400a2e38f56911a5a760f779a5287484",
    "configs/v16_v14_manyseed_curve_signflip_seed249.json": "cb52f5b9ecc34370ee085e6eb49aee7aea7a74f65f683186a0b3c16a292c8ae1",
    "configs/v16_v14_manyseed_curve_signflip_seed250.json": "36f8b4461f420423503b49052201aeaa8efbf8dc182f3493a7cddb0ec7a084b5",
    "configs/v16_v14_manyseed_curve_signflip_seed251.json": "393ec867d4f6637348b9f89c1b26ffba27b1a8a9187caed1f4cfc7fa8ef5021d",
    "configs/v16_v14_manyseed_curve_signflip_seed252.json": "2d913157a61a92b4f6ef762c9fcb572157484fc9a82a2f091cd31b3f0d4d5d2d",
    "configs/v16_v14_manyseed_curve_signflip_seed253.json": "72f6ed130d3f22bc6d3c5addf6dde003c6f77a2757e380cfe969f62cd7421b3b",
    "configs/v16_v14_manyseed_curve_signflip_seed254.json": "919ebe25ea2916926f1f31c4c10c9dd124881e37c2fa1044aa5ebe89296dd3ee",
    "configs/v16_v14_manyseed_curve_signflip_seed255.json": "1bfd04416e901c357e31bee27b891e11232c898c4ac5eaf6fa11939a4868c669",
    "configs/v16_v14_manyseed_curve_signflip_seed256.json": "6fdec34659ba7021c61fcbbefa45da1ea980d56aecc0e4145ecd4bd6ca56899d",
    "configs/v16_v14_manyseed_curve_signflip_seed257.json": "b6d36a9639d8101d55476398a2e0690839b2fdbfc9bc46717a4e216c3e7f6809",
    "configs/v16_v14_manyseed_curve_signflip_seed258.json": "450fb81241ab0f1b49ed813648874e3923c32bbd0d0f49f033869c94a8d5a8bb",
    "configs/v16_v14_manyseed_curve_signflip_seed259.json": "a698c0136f29474f0479c97d3e39e7f9d3ccd5226b0feaf22ecf6487ac82e600",
    "configs/v16_v14_manyseed_curve_signflip_seed260.json": "1b88f4547aba517cd2d591ae5a8609f84a6a310ebfedab2b60aa42fc44c789f1",
    "configs/v16_v14_manyseed_curve_signflip_seed261.json": "7ecfee25b10558b6cc0ba29de39e2c3c5ec0ffdf653be67467f066ffc91cfb17",
    "configs/v16_v14_manyseed_curve_signflip_seed262.json": "70b5cf357ff764bc017a1430fc88c011d40b9e2456ccbe75fb6c974e664c60b0",
    "configs/v16_v14_manyseed_curve_signflip_seed263.json": "dfa5f6bbfa0495e24e78cc374406aa0de1670d91bc1d3ff216b047baf9aa26ce",
    "configs/v16_v14_manyseed_curve_signflip_template.json": "1d93ea5e88cd2ca2da11c7e7f56e1ec7dc3a7637b8abf48818c6894eca3b71e6",
    REGISTRATION_PATH: REGISTRATION_SHA256,
    METRIC_SCHEMA_PATH: METRIC_SCHEMA_SHA256,
    V14_AGGREGATE_PATH: V14_AGGREGATE_SHA256,
    PREEMPTION_CLOSEOUT_PATH: PREEMPTION_CLOSEOUT_SHA256,
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
VALIDATION_EXAMPLES = 400
EFFECT_QUANTUM = 1.0 / (VALIDATION_EXAMPLES * len(POST_BASELINE_STEPS))
TERMINAL_EVIDENCE_FILE_NAMES = (
    "run_result_manifest.json",
    "validation_history.jsonl",
    "log_history.json",
    "environment_snapshot.json",
    "run_manifest.json",
    "resolved_config.json",
    "data_indices.json",
)


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
        raise ValueError(f"unregistered V16 run: {condition=} {seed=}")
    return f"{condition}_seed{seed}"


def expected_config(repository: Path, condition: str, seed: int) -> dict[str, Any]:
    _label(condition, seed)
    return _load_config(repository / CONFIG_PATHS[(condition, seed)])


def validate_config(repository: Path, condition: str, seed: int) -> dict[str, Any]:
    config = expected_config(repository, condition, seed)
    source = json.loads((repository / SOURCE_CONFIG_PATH).read_text())
    changed_science = {
        key: {"seed195": source.get(key), "v16": config.get(key)}
        for key in SCIENCE_KEYS_UNCHANGED_FROM_SEED195
        if config.get(key) != source.get(key)
    }
    if changed_science:
        raise RuntimeError(f"V16 changed frozen celebration science: {changed_science}")
    expected_identity = {
        "seed": seed,
        "updates": 10,
        "eval_every": 2,
        "validation_steps": list(POST_BASELINE_STEPS),
        "save_every": 10,
        "save_total_limit": 1,
        "curve_manifest_sha256": CURVE_MANIFEST_SHA256,
        "train_exclusions_manifest_sha256": TRAIN_EXCLUSIONS_SHA256,
        "registration_sha256": REGISTRATION_SHA256,
        "metric_schema_path": METRIC_SCHEMA_PATH,
        "metric_schema_sha256": METRIC_SCHEMA_SHA256,
        "evidence_eligibility": (
            "development_only_adaptive_v14_many_seed_extension"
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
        raise RuntimeError(f"V16 config identity changed: {changed_identity}")
    expected_components = (
        TREATMENT_COMPONENTS if condition == "jlens" else CONTROL_COMPONENTS
    )
    if config.get("score_components") != [dict(item) for item in expected_components]:
        raise RuntimeError(f"V16 {_label(condition, seed)} score changed")
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
            "GIT_AUTHOR_NAME": "J-Lens V16 Runtime",
            "GIT_AUTHOR_EMAIL": "runtime@example.invalid",
            "GIT_COMMITTER_NAME": "J-Lens V16 Runtime",
            "GIT_COMMITTER_EMAIL": "runtime@example.invalid",
            "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+00:00",
            "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+00:00",
            "JLENS_REPOSITORY_ROOT": REMOTE_REPO.as_posix(),
            "JLENS_MODAL_IMAGE_SPEC": "j-lens-rl-development-v16-v14-celebration-n16-l40s-v1",
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
        "git commit -qm 'J-Lens V16 V14-recipe many-seed runtime'",
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
        raise RuntimeError(f"V16 runtime Git worktree is dirty: {status}")
    registration = json.loads((REMOTE_REPO / REGISTRATION_PATH).read_text())
    if (
        registration.get("protocol") != PROTOCOL
        or registration.get("scientific_status", {}).get("classification")
        != "development_only_adaptive_v14_many_seed_extension"
        or registration.get("firewall", {}).get(
            "protected_final_payloads_mounted_or_accessed"
        )
        is not False
    ):
        raise RuntimeError("V16 registration classification or firewall changed")
    for condition in CONDITIONS:
        for seed in SEEDS:
            validate_config(REMOTE_REPO, condition, seed)
    return observed


def _read_claim(claim_id: str) -> dict[str, Any]:
    path = REMOTE_STATE / "attempt_claim.json"
    if not path.is_file():
        raise RuntimeError("V16 attempt has no durable claim")
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
        raise RuntimeError("V16 attempt claim changed")
    return value


def _intent_path(condition: str, seed: int) -> Path:
    return REMOTE_STATE / "dispatches" / f"{_label(condition, seed)}.intent.json"


def _intent_fields(
    condition: str, seed: int, claim_id: str, root_call_id: str, slot: int
) -> dict[str, Any]:
    """Return every immutable field in a durable worker intent."""
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "claim_id": claim_id,
        "root_call_id": root_call_id,
        "slot": slot,
        "condition": condition,
        "seed": seed,
        "label": _label(condition, seed),
        "config": CONFIG_PATHS[(condition, seed)],
        "config_sha256": EXPECTED_FILE_SHA256[CONFIG_PATHS[(condition, seed)]],
        "wandb_run_id": WANDB_IDS[(condition, seed)],
        "status": "written_before_any_gpu_spawn",
    }


def _validate_recoverable_dispatch(
    status: Mapping[str, Any], claim_id: str, root_call_id: str
) -> dict[tuple[str, int], str]:
    """Validate a fully recorded dispatch and return its immutable call IDs."""
    expected_labels = set(LABELS)
    worker_call_ids = status.get("worker_call_ids")
    if (
        status.get("schema_version") != 1
        or status.get("protocol") != PROTOCOL
        or status.get("claim_id") != claim_id
        or status.get("root_call_id") != root_call_id
        or status.get("stage") != "all_32_spawned_capacity_queue_allowed"
        or status.get("max_parallel_gpus") != MAX_PARALLEL_GPUS
        or status.get("user_requested_gpu_cap") != USER_REQUESTED_GPU_CAP
        or not isinstance(worker_call_ids, dict)
        or set(worker_call_ids) != expected_labels
        or any(
            not isinstance(call_id, str) or not call_id.startswith("fc-")
            for call_id in worker_call_ids.values()
        )
    ):
        raise RuntimeError("V16 existing dispatch status is not exactly recoverable")
    return {
        (condition, seed): worker_call_ids[_label(condition, seed)]
        for condition, seed in RUN_ORDER
    }


def _validate_existing_intents(claim_id: str, root_call_id: str) -> None:
    dispatch_dir = REMOTE_STATE / "dispatches"
    expected_paths = {
        _intent_path(condition, seed)
        for condition, seed in RUN_ORDER
    }
    observed_paths = set(dispatch_dir.glob("*.intent.json"))
    if observed_paths != expected_paths:
        raise RuntimeError("V16 existing intent set is partial or unexpected")
    for slot, (condition, seed) in enumerate(RUN_ORDER):
        path = _intent_path(condition, seed)
        observed = json.loads(path.read_text())
        expected = _intent_fields(condition, seed, claim_id, root_call_id, slot)
        if any(observed.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"V16 existing intent changed: {path.name}")
        if not isinstance(observed.get("created_at_utc"), str):
            raise RuntimeError(f"V16 existing intent lacks creation time: {path.name}")


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
        raise RuntimeError("registered V16 Volume is not fresh and empty")
    value = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "scientific_status": "development_only_adaptive_v14_many_seed_extension",
        "claim_id": claim_id,
        "registration_sha256": REGISTRATION_SHA256,
        "metric_schema_sha256": METRIC_SCHEMA_SHA256,
        "seeds": list(SEEDS),
        "conditions": list(CONDITIONS),
        "labels": list(LABELS),
        "steps": list(STEPS),
        "display_gate_steps": list(DISPLAY_GATE_STEPS),
        "max_parallel_gpus": MAX_PARALLEL_GPUS,
        "user_requested_gpu_cap": USER_REQUESTED_GPU_CAP,
        "wandb_run_ids": {
            _label(condition, seed): WANDB_IDS[(condition, seed)]
            for condition, seed in RUN_ORDER
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
    raise RuntimeError("durable V16 launch receipt did not arrive")


def validate_history_rows(rows: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    if [row.get("step") for row in rows] != list(STEPS):
        raise RuntimeError(
            "validation history must contain exact ordered steps 0,2,4,6,8,10"
        )
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
    if [row.get("step") for row in reward_rows] != list(OPTIMIZER_STEPS):
        raise RuntimeError("Trainer history lacks exact reward steps 1..10")
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
            "Trainer validation rows must be exactly steps 2,4,6,8,10; "
            "baseline 0 is pre-train"
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
    if len(terminal_rows) != 1 or terminal_rows[0].get("step") != 10:
        raise RuntimeError("Trainer history lacks its unique step-10 terminal row")
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
    trainer_state_path = run_dir / "checkpoint-10" / "trainer_state.json"
    if not trainer_state_path.is_file():
        raise RuntimeError(f"{label} lacks checkpoint-10 trainer state")
    trainer_state = json.loads(trainer_state_path.read_text())
    if trainer_state.get("global_step") != 10:
        raise RuntimeError(f"{label} checkpoint global step is not 10")
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
    expected_uploads = {
        name: _sha256(run_dir / name) for name in TERMINAL_EVIDENCE_FILE_NAMES
    }
    artifact = receipt.get("artifact")
    expected_artifact_stem = f"{expected_wandb['run_id']}-terminal-evidence"
    if (
        result.get("completed_updates") != 10
        or result.get("registration_sha256") != REGISTRATION_SHA256
        or result.get("evidence_eligibility")
        != "development_only_adaptive_v14_many_seed_extension"
        or result.get("lens_sha256") != LENS_SHA256
        or result.get("calibration_sha256") != CALIBRATION_SHA256
        or result.get("wandb_identity") != expected_wandb
        or receipt.get("terminal_run_result_sha256") != _sha256(result_path)
        or receipt.get("schema_version") != 2
        or receipt.get("wandb_identity") != expected_wandb
        or receipt.get("observed_wandb_identity")
        != {key: expected_wandb[key] for key in expected_wandb if key != "resume"}
        or receipt.get("uploaded_file_sha256") != expected_uploads
        or not isinstance(artifact, dict)
        or not isinstance(artifact.get("id"), str)
        or not artifact["id"]
        or not isinstance(artifact.get("digest"), str)
        or not artifact["digest"]
        or not isinstance(artifact.get("version"), str)
        or not artifact["version"].startswith("v")
        or not artifact["version"][1:].isdigit()
        or artifact.get("name")
        != f"{expected_artifact_stem}:{artifact.get('version')}"
        or artifact.get("qualified_name")
        != (
            f"{WANDB_ENTITY}/{WANDB_PROJECT}/"
            f"{expected_artifact_stem}:{artifact.get('version')}"
        )
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
        "data_indices_sha256": _sha256(run_dir / "data_indices.json"),
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
        raise RuntimeError(f"V16 training failed closed for {label}")
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


def _sample_sd(values: Sequence[float]) -> float:
    return statistics.stdev(values)


def exact_two_sided_sign_test(effects: Sequence[float]) -> dict[str, Any]:
    raw_values = [float(value) for value in effects]
    if any(not math.isfinite(value) for value in raw_values):
        raise RuntimeError("sign test received a non-finite effect")
    scaled_values = [value / EFFECT_QUANTUM for value in raw_values]
    if any(abs(value - round(value)) > 1e-10 for value in scaled_values):
        raise RuntimeError("sign-test effect is off the registered 1/2000 lattice")
    effect_units = [int(round(value)) for value in scaled_values]
    values = [unit * EFFECT_QUANTUM for unit in effect_units]
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
        "raw_effects": raw_values,
        "effect_units": effect_units,
        "effect_quantum": EFFECT_QUANTUM,
        "positives": positives,
        "negatives": negatives,
        "ties": ties,
        "nonzero": nonzero,
        "mean_effect": statistics.fmean(values),
        "exact_two_sided_p": p_value,
        "nominal_alpha": 0.15,
        "success": (
            positives > negatives
            and statistics.fmean(values) > 0
            and p_value <= 0.15
        ),
    }


def aggregate_results(
    results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if set(results) != set(LABELS):
        raise RuntimeError("aggregate requires exactly the 32 registered runs")
    for seed in SEEDS:
        treatment_result = results[_label("jlens", seed)]
        control_result = results[_label("signflip", seed)]
        treatment_data_sha = treatment_result.get("data_indices_sha256")
        control_data_sha = control_result.get("data_indices_sha256")
        if (
            not isinstance(treatment_data_sha, str)
            or len(treatment_data_sha) != 64
            or treatment_data_sha != control_data_sha
        ):
            raise RuntimeError(f"seed {seed} treatment/control data indices differ")
        if float(treatment_result["curve"]["0"]) != float(
            control_result["curve"]["0"]
        ):
            raise RuntimeError(f"seed {seed} treatment/control baselines differ")
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
                "treatment_sd": _sample_sd(treatment),
                "treatment_sem": _sample_sem(treatment),
                "signflip_mean": statistics.fmean(control),
                "signflip_sd": _sample_sd(control),
                "signflip_sem": _sample_sem(control),
                "paired_mean": statistics.fmean(paired),
                "paired_sd": _sample_sd(paired),
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
        by_step[10]["treatment_by_seed"][str(seed)] for seed in SEEDS
    ]
    signflip_terminal = [
        by_step[10]["signflip_by_seed"][str(seed)] for seed in SEEDS
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
    integrated_baseline_effects = [
        statistics.fmean(
            by_step[step]["treatment_by_seed"][str(seed)]
            - by_step[0]["treatment_by_seed"][str(seed)]
            for step in POST_BASELINE_STEPS
        )
        for seed in SEEDS
    ]
    integrated_control_effects = [
        statistics.fmean(
            by_step[step]["paired_effect_by_seed"][str(seed)]
            for step in POST_BASELINE_STEPS
        )
        for seed in SEEDS
    ]
    shape_gate = {
        "steps": list(DISPLAY_GATE_STEPS),
        "means": [by_step[step]["treatment_mean"] for step in DISPLAY_GATE_STEPS],
    }
    shape_gate["first_above_initial"] = (
        by_step[2]["treatment_mean"] > by_step[0]["treatment_mean"]
    )
    shape_gate["no_downward_steps_2_to_4_to_6"] = (
        by_step[4]["treatment_mean"] >= by_step[2]["treatment_mean"]
        and by_step[6]["treatment_mean"] >= by_step[4]["treatment_mean"]
    )
    shape_gate["passed"] = bool(
        shape_gate["first_above_initial"]
        and shape_gate["no_downward_steps_2_to_4_to_6"]
    )
    integrated_baseline = exact_two_sided_sign_test(integrated_baseline_effects)
    integrated_control = exact_two_sided_sign_test(integrated_control_effects)
    terminal_control = exact_two_sided_sign_test(treatment_control_effects)
    terminal_baseline = exact_two_sided_sign_test(treatment_baseline_effects)
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "scientific_status": "development_only_adaptive_v14_many_seed_extension",
        "canonical_x_axis": "global_step",
        "complete_curve_steps": list(STEPS),
        "every_registered_node_retained": True,
        "seeds": list(SEEDS),
        "rows": rows,
        "early_complete_node_shape": shape_gate,
        "primary_integrated_treatment_minus_baseline": integrated_baseline,
        "paired_integrated_treatment_minus_signflip": integrated_control,
        "terminal_treatment_minus_signflip": terminal_control,
        "terminal_treatment_minus_baseline": terminal_baseline,
        "target_evidence_met": bool(
            shape_gate["passed"] and integrated_baseline["success"]
        ),
        "causal_reward_sign_evidence_met": integrated_control["success"],
        "multiplicity_caveat": (
            "Nominal adaptive development evidence only; prior visible V11-V15 "
            "outcomes and the selected reward, horizon, cadence, and analyses "
            "are not familywise-corrected."
        ),
        "protected_final_payloads_accessed": False,
    }


def _write_aggregate_files(aggregate: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    evidence = REMOTE_STATE / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    json_path = evidence / "v16_v14_manyseed_curve_aggregate.json"
    csv_path = evidence / "v16_v14_manyseed_curve.csv"
    png_path = evidence / "v16_v14_manyseed_curve.png"
    _write_exclusive(json_path, aggregate)
    fieldnames = [
        "global_step",
        "treatment_mean",
        "treatment_sd",
        "treatment_sem",
        "signflip_mean",
        "signflip_sd",
        "signflip_sem",
        "paired_mean",
        "paired_sd",
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
                    **{key: row[key] for key in fieldnames[:10]},
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
    curve_axis.set_title(
        "V14 celebration reward: complete 16-seed development curve"
    )
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
    effect_axis.set_xticks(list(STEPS))
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
            "adaptive-v14-extension",
            "aggregate",
            "celebration-family",
            "matched-signflip",
            "complete-u2-h10",
            "n16-paired",
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
            "scientific_status": (
                "development_only_adaptive_v14_many_seed_extension"
            ),
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
                    "validation/treatment_sd": row["treatment_sd"],
                    "validation/treatment_sem": row["treatment_sem"],
                    "validation/signflip_mean": row["signflip_mean"],
                    "validation/signflip_sd": row["signflip_sd"],
                    "validation/signflip_sem": row["signflip_sem"],
                    "validation/paired_mean": row["paired_mean"],
                    "validation/paired_sd": row["paired_sd"],
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
                    **{
                        f"validation/paired_seed{seed}": row[
                            "paired_effect_by_seed"
                        ][str(seed)]
                        for seed in SEEDS
                    },
                }
            )
        run.summary["early_complete_node_shape_passed"] = aggregate[
            "early_complete_node_shape"
        ]["passed"]
        run.summary["primary_integrated_baseline_p"] = aggregate[
            "primary_integrated_treatment_minus_baseline"
        ]["exact_two_sided_p"]
        run.summary["paired_integrated_control_p"] = aggregate[
            "paired_integrated_treatment_minus_signflip"
        ]["exact_two_sided_p"]
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
                    "development_only_adaptive_v14_many_seed_extension"
                ),
            },
        )
        for path in paths:
            artifact.add_file(str(path), name=path.name)
        logged = run.log_artifact(
            artifact, aliases=["latest", "v14-many-seed-complete-curve"]
        )
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
    timeout=20 * 60 * 60,
    startup_timeout=60 * 60,
    retries=0,
    secrets=[wandb_secret],
    volumes={REMOTE_STATE: state_volume},
)
def orchestrate(claim_id: str) -> dict[str, Any]:
    status_path = REMOTE_STATE / "attempt_status.json"
    root_call_id: str | None = None
    recovered_coordinator = False
    try:
        receipt = _wait_for_launch_receipt(claim_id)
        root_call_id = modal.current_function_call_id()
        if receipt.get("root_call_id") != root_call_id:
            raise RuntimeError("V16 orchestrator lacks durable root authority")
        _read_claim(claim_id)
        state_volume.reload()
        dispatch_dir = REMOTE_STATE / "dispatches"
        existing_intents = set(dispatch_dir.glob("*.intent.json"))
        if status_path.is_file():
            dispatch_status = json.loads(status_path.read_text())
            call_ids = _validate_recoverable_dispatch(
                dispatch_status, claim_id, root_call_id
            )
            _validate_existing_intents(claim_id, root_call_id)
            calls = {
                key: modal.FunctionCall.from_id(call_id)
                for key, call_id in call_ids.items()
            }
            recovered_coordinator = True
            _replace_json(
                status_path,
                {
                    **dispatch_status,
                    "coordinator_recovered": True,
                    "coordinator_recovered_at_utc": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            state_volume.commit()
        elif existing_intents:
            raise RuntimeError(
                "V16 found intents without a complete dispatch status; "
                "duplicate spawning is forbidden"
            )
        else:
            for slot, (condition, seed) in enumerate(RUN_ORDER):
                intent = {
                    **_intent_fields(
                        condition, seed, claim_id, root_call_id, slot
                    ),
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                _write_exclusive(_intent_path(condition, seed), intent)
            state_volume.commit()
            calls = {
                (condition, seed): train_run.spawn(condition, seed, claim_id)
                for condition, seed in RUN_ORDER
            }
            _replace_json(
                status_path,
                {
                    "schema_version": 1,
                    "protocol": PROTOCOL,
                    "claim_id": claim_id,
                    "root_call_id": root_call_id,
                    "stage": "all_32_spawned_capacity_queue_allowed",
                    "max_parallel_gpus": MAX_PARALLEL_GPUS,
                    "user_requested_gpu_cap": USER_REQUESTED_GPU_CAP,
                    "worker_call_ids": {
                        _label(condition, seed): call.object_id
                        for (condition, seed), call in calls.items()
                    },
                    "coordinator_recovered": False,
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            state_volume.commit()
        results: dict[str, dict[str, Any]] = {}
        outcomes: dict[str, dict[str, Any]] = {}
        failures: list[str] = []
        for condition, seed in RUN_ORDER:
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
                "V16 workers failed after all 32 calls were drained: "
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
            "scientific_status": (
                "development_only_adaptive_v14_many_seed_extension"
            ),
            "claim_id": claim_id,
            "stage": "complete",
            "results": results,
            "aggregate": aggregate,
            "aggregate_publish_receipt": aggregate_receipt,
            "closed_v11_v12_v13_state_mounted": False,
            "protected_final_payloads_accessed": False,
            "coordinator_recovered": recovered_coordinator,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_exclusive(REMOTE_STATE / "evidence" / "summary.json", summary)
        _replace_json(status_path, summary)
        state_volume.commit()
        return summary
    except BaseException as error:
        try:
            state_volume.reload()
            prior_dispatch: dict[str, Any] = {}
            if status_path.is_file():
                existing_status = json.loads(status_path.read_text())
                if isinstance(existing_status, dict):
                    prior_dispatch = {
                        key: existing_status[key]
                        for key in (
                            "stage",
                            "root_call_id",
                            "worker_call_ids",
                            "coordinator_recovered",
                        )
                        if key in existing_status
                    }
            _replace_json(
                status_path,
                {
                    "schema_version": 1,
                    "protocol": PROTOCOL,
                    "claim_id": claim_id,
                    "root_call_id": root_call_id,
                    "stage": "failed_closed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "retry_started_training_run_permitted": False,
                    "protected_final_payloads_accessed": False,
                    "prior_dispatch": prior_dispatch,
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
        raise RuntimeError("V16 launch requires an exact clean pushed main")
    for relative, expected in EXPECTED_FILE_SHA256.items():
        path = LOCAL_REPO / relative
        if not path.is_file() or path.is_symlink() or _sha256(path) != expected:
            raise RuntimeError(f"local registered input changed: {relative}")
    for condition, seed in RUN_ORDER:
        validate_config(LOCAL_REPO, condition, seed)
    modal_executable = shutil.which("modal")
    if modal_executable is None:
        raise RuntimeError("V16 preflight cannot locate the Modal CLI on PATH")
    modal_cli = Path(modal_executable)
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
        raise RuntimeError("V16 Volume must be fresh and empty")
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
        "user_requested_gpu_cap": USER_REQUESTED_GPU_CAP,
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
                    "development_only_adaptive_v14_many_seed_extension"
                ),
                "claim_id": claim_id,
                "root_call_id": call.object_id,
                "app_id": app.app_id,
                "volume": VOLUME_NAME,
                "gpu_type": GPU_TYPE,
                "function_max_containers": MAX_PARALLEL_GPUS,
                "user_requested_gpu_cap": USER_REQUESTED_GPU_CAP,
                "seeds": list(SEEDS),
                "conditions": list(CONDITIONS),
                "steps": list(STEPS),
                "display_gate_steps": list(DISPLAY_GATE_STEPS),
                "wandb_group": WANDB_GROUP,
                "wandb_run_ids": {
                    _label(condition, seed): WANDB_IDS[(condition, seed)]
                    for condition, seed in RUN_ORDER
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
        ),
        flush=True,
    )
    summary = call.get()
    print(
        json.dumps(
            {
                "status": "orchestrator_terminal",
                "claim_id": claim_id,
                "root_call_id": call.object_id,
                "stage": summary.get("stage"),
                "target_evidence_met": summary.get("aggregate", {}).get(
                    "target_evidence_met"
                ),
                "aggregate_publish_receipt": summary.get(
                    "aggregate_publish_receipt"
                ),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
