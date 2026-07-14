"""Run eight independent emotional-word J-only RL arms on the exposed V4 curve.

This is an exploratory screen, not a confirmatory evaluation.  Every arm uses
the same retained U5 readout and differs only in its single target word and the
predeclared reward sign.  The target-independent frozen lens is recalibrated
for each word on pinned WikiText before the eight fixed-length training jobs.

Only the exposed failed-V4 curve and the historical train-exclusion manifest
are mounted.  No sealed-final, future-curve, or reserve indices enter the
image.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import modal


LOCAL_REPO = Path(__file__).resolve().parent
REMOTE_REPO = Path("/workspace/j-lens-rl")
REMOTE_OUTPUT = Path("/single_word_screen")
LOCAL_ARTIFACTS = LOCAL_REPO / "artifacts"
LOCAL_MANIFESTS = LOCAL_REPO / ".confirmatory/manifests"

APP_NAME = "j-lens-rl-emotional-single-word-screen-v1"
VOLUME_NAME = "j-lens-rl-emotional-single-word-screen-v1-20260714a"
PROTOCOL = "j-lens-rl-emotional-single-word-screen-v1"
GPU_TYPE = "L40S"
MAX_GPU_CONTAINERS = 8
CALIBRATION_MAX_GPU_CONTAINERS = 8
SEED = 167

MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
LENS_SHA256 = "178a9671cbf41882135807bde59b828e36c6f8f98b32c809ea3346860aad10dc"
V4_CLOSEOUT_RELATIVE = "protocol_archive/v4_closeout.json"
V4_CLOSEOUT_SHA256 = (
    "aaf4bcde9a9cacc482c7f3dde94218cf02a6aa60be81e43cae5cde3086d17e35"
)
EXPOSED_MANIFEST_SHA256 = {
    "curve_indices.json": (
        "ad348fe17d2e6bd6aac691d9bcdbb9da481f675305fa0e05c68e86dad97451c1"
    ),
    "train_exclusions.json": (
        "7c1ca4f404ba9149093cc3c57dc3607582f671397e2fe99e93449848c1d65d61"
    ),
}

ARM_ORDER = ("yay", "wow", "joy", "proud", "excited", "damn", "fuck", "worried")
REWARD_WEIGHTS = {
    "yay": 1.0,
    "wow": 1.0,
    "joy": 1.0,
    "proud": 1.0,
    "excited": 1.0,
    "damn": -1.0,
    "fuck": -1.0,
    "worried": -1.0,
}
EXPECTED_TOKEN_IDS = {
    "yay": [97559, 138496],
    "wow": [35665, 35881, 45717, 57454, 61300],
    "joy": [4123, 15888, 27138, 79771],
    "proud": [12409, 83249],
    "excited": [12035],
    "damn": [26762, 82415, 88619, 95614],
    "fuck": [7820, 25090, 70474, 75021, 76374],
    "worried": [17811],
}
VARIANTS = {
    word: f"configs/single_word_screen_{word}.json" for word in ARM_ORDER
}
COMMON_CONFIG = "configs/single_word_screen_common.json"
EXPECTED_STEPS = (0, 2, 4, 6, 10, 15, 20, 25)
GATE_STEPS = (0, 2, 4, 6)
SOURCE_FILES = (
    "modal_single_word_screen.py",
    "run_single_word_screen.sh",
    "src/jlens_rl/fit_lens.py",
    "src/jlens_rl/reward.py",
    "src/jlens_rl/train.py",
)


app = modal.App(APP_NAME)
output_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)
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
        LOCAL_MANIFESTS / "curve_indices.json",
        (REMOTE_REPO / ".confirmatory/manifests/curve_indices.json").as_posix(),
        copy=True,
    )
    .add_local_file(
        LOCAL_MANIFESTS / "train_exclusions.json",
        (REMOTE_REPO / ".confirmatory/manifests/train_exclusions.json").as_posix(),
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


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _git(*args: str, repo: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _run(command: list[str], *, cwd: Path = REMOTE_REPO) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    base = payload.pop("base", None)
    if base:
        parent = _load_config(path.parent / base)
        parent.update(payload)
        return parent
    return payload


def _curve_pass(curve: dict[int, float]) -> bool:
    return curve[2] > curve[0] and curve[4] >= curve[2] and curve[6] >= curve[4]


def _source_hashes(repo: Path) -> dict[str, str]:
    return {relative: _sha256(repo / relative) for relative in SOURCE_FILES}


def _config_hashes(repo: Path) -> dict[str, str]:
    paths = [COMMON_CONFIG, *VARIANTS.values()]
    return {relative: _sha256(repo / relative) for relative in paths}


def _validate_v4_closeout(repo: Path) -> str:
    path = repo / V4_CLOSEOUT_RELATIVE
    if not path.is_file() or _sha256(path) != V4_CLOSEOUT_SHA256:
        raise RuntimeError("the byte-pinned V4 closeout is missing or changed")
    closeout = json.loads(path.read_text())
    expected = {
        "protocol": "j-lens-rl-confirmatory-v4",
        "git_commit": "8ae04dc61a3ae474ffa62dd0e738d6b40deed303",
        "attempt_stage": "curve_failed",
        "final_unlocked_present": False,
        "evals_directory_present": False,
        "final_evaluation_labels": [],
        "signflip_run_labels": [],
    }
    if any(closeout.get(key) != value for key, value in expected.items()):
        raise RuntimeError("V4 closeout does not prove a failed no-look attempt")
    curve = closeout.get("curve", {})
    if (
        curve.get("passed") is not False
        or curve.get("steps") != list(GATE_STEPS)
        or curve.get("full_mean_exact_match", {}).get("2")
        <= curve.get("full_mean_exact_match", {}).get("0")
        or curve.get("full_mean_exact_match", {}).get("4")
        >= curve.get("full_mean_exact_match", {}).get("2")
    ):
        raise RuntimeError("V4 failed-curve evidence changed")
    return V4_CLOSEOUT_SHA256


def _expected_component(word: str) -> list[dict[str, Any]]:
    return [
        {
            "layer": 8,
            "start_fraction": 0.5,
            "end_fraction": 1.0,
            "aggregation": "mean",
            "weight": REWARD_WEIGHTS[word],
        }
    ]


def _validate_config(word: str, config: dict[str, Any]) -> None:
    expected = {
        "model_revision": MODEL_REVISION,
        "dataset_revision": DATASET_REVISION,
        "lens_sha256": LENS_SHA256,
        "validation_source": "train",
        "validation_examples": 400,
        "validation_batch_size": 64,
        "train_examples": 1000,
        "updates": 25,
        "min_new_tokens": 64,
        "eval_every": 2,
        "validation_steps": list(EXPECTED_STEPS[1:]),
        "validation_observational_only": True,
        "require_clean_repository": True,
        "early_stopping_patience": None,
        "save_every": 25,
        "save_total_limit": 1,
        "learning_rate": 3e-6,
        "lr_scheduler_type": "constant",
        "warmup_steps": 0,
        "warmup_ratio": 0.0,
        "mask_target_tokens": True,
        "reward_type": "jlens",
        "seed": SEED,
        "wandb_mode": "online",
        "wandb_project": "j-lens-rl",
        "score_stride": 5,
        "target_words": [word],
        "score_components": _expected_component(word),
    }
    changed = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if changed:
        raise RuntimeError(f"{word} config changed: {changed}")
    if config.get("validation_indices_path") != ".confirmatory/manifests/curve_indices.json":
        raise RuntimeError(f"{word} config does not use the exposed V4 curve")
    if config.get("reserved_train_indices_path") != ".confirmatory/manifests/train_exclusions.json":
        raise RuntimeError(f"{word} config does not use the frozen train exclusions")
    if config.get("output_dir") != str(REMOTE_OUTPUT / "runs" / word):
        raise RuntimeError(f"{word} output directory changed")
    if config.get("calibration_path") != str(
        REMOTE_OUTPUT / "artifacts" / f"{word}_calibration.json"
    ):
        raise RuntimeError(f"{word} calibration path changed")
    sign = "positive" if REWARD_WEIGHTS[word] > 0 else "negative"
    if config.get("run_name") != f"single-word-screen-{word}-{sign}-seed167":
        raise RuntimeError(f"{word} W&B run name changed")


def _validate_templates(repo: Path) -> dict[str, str]:
    hashes = _config_hashes(repo)
    for word, relative in VARIANTS.items():
        _validate_config(word, _load_config(repo / relative))
    return hashes


def _launch_manifest() -> dict[str, Any]:
    status = _git("status", "--porcelain=v1", "--untracked-files=all", repo=LOCAL_REPO)
    if status:
        raise RuntimeError(f"single-word screen requires a clean committed tree:\n{status}")
    v4_closeout_sha256 = _validate_v4_closeout(LOCAL_REPO)
    config_sha256 = _validate_templates(LOCAL_REPO)
    source_sha256 = _source_hashes(LOCAL_REPO)
    actual_manifests = {
        name: _sha256(LOCAL_MANIFESTS / name) for name in EXPOSED_MANIFEST_SHA256
    }
    if actual_manifests != EXPOSED_MANIFEST_SHA256:
        raise RuntimeError(f"exposed manifest hash mismatch: {actual_manifests}")
    lens = LOCAL_ARTIFACTS / "qwen25_05b_solved_lens.pt"
    if _sha256(lens) != LENS_SHA256:
        raise RuntimeError("target-independent lens transport changed")
    return {
        "claim_id": uuid.uuid4().hex,
        "protocol": PROTOCOL,
        "git_commit": _git("rev-parse", "HEAD", repo=LOCAL_REPO),
        "git_tree": _git("rev-parse", "HEAD^{tree}", repo=LOCAL_REPO),
        "git_status": status,
        "arm_order": list(ARM_ORDER),
        "reward_weights": REWARD_WEIGHTS,
        "expected_token_ids": EXPECTED_TOKEN_IDS,
        "variants": VARIANTS,
        "seed": SEED,
        "expected_steps": list(EXPECTED_STEPS),
        "gate_steps": list(GATE_STEPS),
        "gate_criterion": "step2 > step0 and step4 >= step2 and step6 >= step4",
        "model_revision": MODEL_REVISION,
        "dataset_revision": DATASET_REVISION,
        "wikitext_revision": WIKITEXT_REVISION,
        "lens_sha256": LENS_SHA256,
        "v4_closeout_sha256": v4_closeout_sha256,
        "source_sha256": source_sha256,
        "config_sha256": config_sha256,
        "exposed_manifest_sha256": actual_manifests,
        "data_boundary": "exposed failed-V4 400-item development curve only",
        "unmounted_manifests": [
            "sealed_final_indices.json",
            "future_reserve_indices.json",
            "retired_v3_curve_indices.json",
        ],
        "gpu_type": GPU_TYPE,
        "max_parallel_gpu_workers": MAX_GPU_CONTAINERS,
        "calibration_max_parallel_gpu_workers": CALIBRATION_MAX_GPU_CONTAINERS,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _verify_remote_manifest(manifest: dict[str, Any]) -> None:
    expected_fields = {
        "protocol": PROTOCOL,
        "arm_order": list(ARM_ORDER),
        "reward_weights": REWARD_WEIGHTS,
        "expected_token_ids": EXPECTED_TOKEN_IDS,
        "variants": VARIANTS,
        "seed": SEED,
        "expected_steps": list(EXPECTED_STEPS),
        "gate_steps": list(GATE_STEPS),
        "model_revision": MODEL_REVISION,
        "dataset_revision": DATASET_REVISION,
        "wikitext_revision": WIKITEXT_REVISION,
        "lens_sha256": LENS_SHA256,
        "exposed_manifest_sha256": EXPOSED_MANIFEST_SHA256,
        "max_parallel_gpu_workers": MAX_GPU_CONTAINERS,
        "calibration_max_parallel_gpu_workers": CALIBRATION_MAX_GPU_CONTAINERS,
    }
    if any(manifest.get(key) != value for key, value in expected_fields.items()):
        raise RuntimeError("single-word screen launch manifest changed")
    if manifest.get("git_commit") != _git("rev-parse", "HEAD", repo=REMOTE_REPO):
        raise RuntimeError("remote commit differs from launch")
    if manifest.get("git_tree") != _git("rev-parse", "HEAD^{tree}", repo=REMOTE_REPO):
        raise RuntimeError("remote source tree differs from launch")
    if _git("status", "--porcelain=v1", "--untracked-files=all", repo=REMOTE_REPO):
        raise RuntimeError("remote repository is dirty")
    if manifest.get("source_sha256") != _source_hashes(REMOTE_REPO):
        raise RuntimeError("remote source hashes differ from launch")
    if manifest.get("config_sha256") != _validate_templates(REMOTE_REPO):
        raise RuntimeError("remote config hashes differ from launch")
    if manifest.get("v4_closeout_sha256") != _validate_v4_closeout(REMOTE_REPO):
        raise RuntimeError("remote V4 closeout differs from launch")
    actual_manifests = {
        name: _sha256(REMOTE_REPO / ".confirmatory/manifests" / name)
        for name in EXPOSED_MANIFEST_SHA256
    }
    if actual_manifests != EXPOSED_MANIFEST_SHA256:
        raise RuntimeError("remote exposed manifests changed")
    if _sha256(REMOTE_REPO / "artifacts/qwen25_05b_solved_lens.pt") != LENS_SHA256:
        raise RuntimeError("remote target-independent lens changed")


def _set_status(claim_id: str, stage: str, **details: Any) -> None:
    claim = json.loads((REMOTE_OUTPUT / "attempt_manifest.json").read_text())
    if claim.get("claim_id") != claim_id:
        raise RuntimeError("single-word screen Volume claim mismatch")
    _write_json(
        REMOTE_OUTPUT / "attempt_status.json",
        {
            "claim_id": claim_id,
            "stage": stage,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            **details,
        },
    )


def _require_stage(claim_id: str, expected_stage: str) -> None:
    status = json.loads((REMOTE_OUTPUT / "attempt_status.json").read_text())
    if status.get("claim_id") != claim_id or status.get("stage") != expected_stage:
        raise RuntimeError(
            f"single-word screen claim is not in {expected_stage!r} stage"
        )


@app.function(
    image=repo_image,
    cpu=2,
    memory=4096,
    max_containers=1,
    timeout=10 * 60,
    startup_timeout=60 * 60,
    retries=0,
    volumes={REMOTE_OUTPUT: output_volume},
)
def claim_attempt(manifest: dict[str, Any]) -> dict[str, Any]:
    output_volume.reload()
    existing = [path.name for path in REMOTE_OUTPUT.iterdir()]
    if existing:
        raise RuntimeError(f"single-word screen Volume is not fresh: {sorted(existing)}")
    _verify_remote_manifest(manifest)
    _write_json(REMOTE_OUTPUT / "attempt_manifest.json", manifest)
    _set_status(str(manifest["claim_id"]), "claimed")
    output_volume.commit()
    return manifest


def _calibration_command(word: str, output: Path) -> list[str]:
    if word not in ARM_ORDER:
        raise ValueError(f"unknown single-word arm: {word}")
    return [
        sys.executable,
        "-m",
        "jlens_rl.fit_lens",
        "--model-revision",
        MODEL_REVISION,
        "--wikitext-revision",
        WIKITEXT_REVISION,
        "--lens-input",
        "artifacts/qwen25_05b_solved_lens.pt",
        "--output",
        "artifacts/qwen25_05b_solved_lens.pt",
        "--calibration-output",
        str(output),
        "--target-word",
        word,
        "--num-prompts",
        "100",
        "--calibration-prompts",
        "50",
        "--layers",
        "8,14,20",
        "--seed",
        "42",
    ]


def _validate_calibration(word: str, path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    expected = {
        "target_words": [word],
        "token_ids": EXPECTED_TOKEN_IDS[word],
        "layers": [8, 14, 20],
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_revision": MODEL_REVISION,
        "adapter": None,
        "corpus": "wikitext",
        "dataset": "Salesforce/wikitext",
        "dataset_revision": WIKITEXT_REVISION,
        "lens_sha256": LENS_SHA256,
        "lens_input": "artifacts/qwen25_05b_solved_lens.pt",
        "lens_input_sha256": LENS_SHA256,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"{word} calibration metadata changed")
    mean = float(payload.get("mean", float("nan")))
    std = float(payload.get("std", float("nan")))
    if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
        raise RuntimeError(f"{word} calibration statistics are invalid")
    return payload


@app.function(
    image=repo_image,
    gpu=GPU_TYPE,
    cpu=4,
    memory=32768,
    max_containers=CALIBRATION_MAX_GPU_CONTAINERS,
    timeout=2 * 60 * 60,
    startup_timeout=60 * 60,
    retries=0,
    single_use_containers=True,
    volumes={REMOTE_OUTPUT: output_volume},
)
def calibrate_word(word: str) -> dict[str, Any]:
    if word not in ARM_ORDER:
        raise ValueError(f"unknown single-word arm: {word}")
    output_volume.reload()
    manifest = json.loads((REMOTE_OUTPUT / "attempt_manifest.json").read_text())
    _verify_remote_manifest(manifest)
    _require_stage(str(manifest["claim_id"]), "calibrating")
    output = REMOTE_OUTPUT / "artifacts" / f"{word}_calibration.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite calibration: {output}")
    try:
        _run(_calibration_command(word, output))
        payload = _validate_calibration(word, output)
        result = {
            "word": word,
            "target_words": [word],
            "token_ids": payload["token_ids"],
            "mean": payload["mean"],
            "std": payload["std"],
            "calibration_sha256": _sha256(output),
            "lens_sha256": LENS_SHA256,
            "wikitext_revision": WIKITEXT_REVISION,
        }
        _write_json(REMOTE_OUTPUT / "artifacts" / f"{word}_manifest.json", result)
        return result
    finally:
        output_volume.commit()


def _materialize_config(word: str) -> tuple[Path, dict[str, Any]]:
    config = _load_config(REMOTE_REPO / VARIANTS[word])
    calibration = REMOTE_OUTPUT / "artifacts" / f"{word}_calibration.json"
    _validate_calibration(word, calibration)
    config["calibration_path"] = str(calibration)
    config["calibration_sha256"] = _sha256(calibration)
    config["lens_sha256"] = LENS_SHA256
    _validate_config(word, config)
    path = REMOTE_OUTPUT / "resolved_configs" / f"{word}.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite resolved config: {path}")
    _write_json(path, config)
    return path, config


def _adapter_identity(path: Path) -> dict[str, Any]:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise FileNotFoundError(f"no final adapter files under {path}")
    hashes = {item.relative_to(path).as_posix(): _sha256(item) for item in files}
    if not any(name.startswith("adapter_model") for name in hashes):
        raise FileNotFoundError(f"no adapter model weights under {path}")
    return {"sha256": _canonical_sha256(hashes), "files": hashes}


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
def train_arm(word: str) -> dict[str, Any]:
    if word not in ARM_ORDER:
        raise ValueError(f"unknown single-word arm: {word}")
    output_volume.reload()
    manifest = json.loads((REMOTE_OUTPUT / "attempt_manifest.json").read_text())
    _verify_remote_manifest(manifest)
    _require_stage(str(manifest["claim_id"]), "training")
    config_path, config = _materialize_config(word)
    try:
        _run(
            [
                sys.executable,
                "-m",
                "jlens_rl.train",
                "--config",
                str(config_path),
                "--wandb-mode",
                "online",
            ]
        )
        run_dir = Path(config["output_dir"])
        saved_config = run_dir / "resolved_config.json"
        data_path = run_dir / "data_indices.json"
        validation_path = run_dir / "validation_history.jsonl"
        log_path = run_dir / "log_history.json"
        run_manifest_path = run_dir / "run_manifest.json"
        if json.loads(saved_config.read_text()) != config:
            raise RuntimeError(f"{word} saved a different resolved config")
        rows = [
            json.loads(line) for line in validation_path.read_text().splitlines() if line
        ]
        curve = {int(row["step"]): float(row["exact_match"]) for row in rows}
        if tuple(sorted(curve)) != EXPECTED_STEPS:
            raise RuntimeError(f"{word} has incomplete curve: {sorted(curve)}")
        curve_manifest = EXPOSED_MANIFEST_SHA256["curve_indices.json"]
        if any(
            row.get("validation_indices_sha256") != curve_manifest
            or row.get("validation_source") != "train"
            for row in rows
        ):
            raise RuntimeError(f"{word} used the wrong development curve")
        run_manifest = json.loads(run_manifest_path.read_text())
        if (
            run_manifest.get("git_commit") != manifest["git_commit"]
            or run_manifest.get("git_dirty") is not False
            or run_manifest.get("reward_type") != "jlens"
            or run_manifest.get("config_sha256") != _sha256(config_path)
            or run_manifest.get("resolved_config_sha256") != _sha256(saved_config)
            or run_manifest.get("data_indices_sha256") != _sha256(data_path)
            or run_manifest.get("lens_sha256") != LENS_SHA256
            or run_manifest.get("calibration_sha256") != config["calibration_sha256"]
            or "L40S" not in str(run_manifest.get("runtime", {}).get("cuda_device_name"))
        ):
            raise RuntimeError(f"{word} run provenance mismatch")
        indices = json.loads(data_path.read_text())
        train_indices = set(indices["train_source_indices"])
        validation_indices = set(indices["validation_source_indices"])
        curve_indices = set(
            json.loads(
                (REMOTE_REPO / ".confirmatory/manifests/curve_indices.json").read_text()
            )["indices"]
        )
        excluded = set(
            json.loads(
                (
                    REMOTE_REPO
                    / ".confirmatory/manifests/train_exclusions.json"
                ).read_text()
            )["indices"]
        )
        if (
            len(train_indices) != 1000
            or len(validation_indices) != 400
            or validation_indices != curve_indices
            or train_indices & validation_indices
            or train_indices & excluded
        ):
            raise RuntimeError(f"{word} violated the data firewall")
        literal_rates = {
            str(row["step"]): float(row["literal_target_completion_rate"])
            for row in rows
        }
        if any(not 0.0 <= rate <= 1.0 for rate in literal_rates.values()):
            raise RuntimeError(f"{word} has invalid literal-target audit values")
        result = {
            "word": word,
            "reward_weight": REWARD_WEIGHTS[word],
            "curve": {str(step): curve[step] for step in EXPECTED_STEPS},
            "gate_steps": list(GATE_STEPS),
            "requested_curve_pattern": _curve_pass(curve),
            "target_words": config["target_words"],
            "score_stride": config["score_stride"],
            "score_components": config["score_components"],
            "literal_target_completion_rate": literal_rates,
            "calibration_sha256": config["calibration_sha256"],
            "resolved_config_sha256": _sha256(saved_config),
            "run_manifest_sha256": _sha256(run_manifest_path),
            "data_indices_sha256": _sha256(data_path),
            "validation_history_sha256": _sha256(validation_path),
            "log_history_sha256": _sha256(log_path),
            "final_adapter": _adapter_identity(run_dir / "final"),
        }
        _write_json(run_dir / "screen_result.json", result)
        return result
    finally:
        output_volume.commit()


def _mapped(function: Any, values: Iterable[str]) -> list[Any]:
    materialized = list(values)
    results = list(function.map(materialized, order_outputs=True, return_exceptions=True))
    failures = [
        {"input": materialized[index], "error": repr(result)}
        for index, result in enumerate(results)
        if isinstance(result, BaseException)
    ]
    if failures:
        raise RuntimeError(f"{len(failures)} mapped jobs failed: {failures}")
    return results


@app.function(
    image=repo_image,
    cpu=1,
    memory=2048,
    max_containers=1,
    timeout=8 * 60 * 60,
    startup_timeout=60 * 60,
    retries=0,
    volumes={REMOTE_OUTPUT: output_volume},
)
def orchestrate(claim_id: str) -> dict[str, Any]:
    output_volume.reload()
    _require_stage(claim_id, "claimed")
    _set_status(claim_id, "calibrating")
    output_volume.commit()
    try:
        calibrations = _mapped(calibrate_word, ARM_ORDER)
        output_volume.reload()
        _set_status(claim_id, "training", calibrations=calibrations)
        output_volume.commit()
        results = _mapped(train_arm, ARM_ORDER)
        selected = next(
            (result["word"] for result in results if result["requested_curve_pattern"]),
            None,
        )
        output_volume.reload()
        _set_status(
            claim_id,
            "complete",
            calibrations=calibrations,
            results=results,
            passing_arms=[
                result["word"] for result in results if result["requested_curve_pattern"]
            ],
            selected_first_passing_arm_order=selected,
        )
        output_volume.commit()
        return {"stage": "complete", "results": results, "selected": selected}
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
    manifest = _launch_manifest()
    claim_attempt.remote(manifest)
    call = orchestrate.spawn(str(manifest["claim_id"]))
    print(
        json.dumps(
            {
                "status": "submitted",
                "function_call_id": call.object_id,
                "app": APP_NAME,
                "volume": VOLUME_NAME,
                "gpu_type": GPU_TYPE,
                "max_parallel_gpus": MAX_GPU_CONTAINERS,
                "arms": list(ARM_ORDER),
            },
            indent=2,
        )
    )
