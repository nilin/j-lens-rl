"""Screen retained ultradense and new word-family intrinsic rewards.

This is deliberately exploratory.  It reuses the target-independent frozen
WikiText transport, recalibrates two word sets on held-out WikiText, then runs
eight distinct J-only reward constructions on the exposed failed-V4 curve.
No unopened sealed-final or reserve manifest is copied into the image.
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
REMOTE_OUTPUT = Path("/word_explore")
# Isolated implementation worktrees share the canonical ignored artifacts and
# prepared manifests.  After cherry-pick, CANONICAL_REPO == LOCAL_REPO.
CANONICAL_REPO = Path("/j-lens-rl")
LOCAL_ARTIFACTS = CANONICAL_REPO / "artifacts"
LOCAL_MANIFESTS = CANONICAL_REPO / ".confirmatory/manifests"

VOLUME_NAME = "j-lens-rl-alternative-screen-v1-20260714b"
GPU_TYPE = "L40S"
MAX_GPU_CONTAINERS = 8
CALIBRATION_MAX_GPU_CONTAINERS = 2
SEED = 167

MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
LENS_SHA256 = "178a9671cbf41882135807bde59b828e36c6f8f98b32c809ea3346860aad10dc"
SOLVED_CALIBRATION_SHA256 = (
    "3607ad225cd60bed58a4dc53f78346f9e6c3f4968e8e6f6679a3565923309418"
)
V4_CLOSEOUT_RELATIVE = "protocol_archive/v4_closeout.json"
V4_CLOSEOUT_SHA256 = (
    "aaf4bcde9a9cacc482c7f3dde94218cf02a6aa60be81e43cae5cde3086d17e35"
)
V5_PREREGISTRATION_RELATIVE = "protocol_archive/v5_preregistration.json"
FAILED_SCREEN_CLOSEOUT_RELATIVE = "protocol_archive/word_screen_attempt1_closeout.json"
FAILED_SCREEN_CLOSEOUT_SHA256 = (
    "399559f0607bded85048633179b39a33da25d2de9fcdb4e448725770a30b90c7"
)
EXPOSED_MANIFEST_SHA256 = {
    "curve_indices.json": (
        "ad348fe17d2e6bd6aac691d9bcdbb9da481f675305fa0e05c68e86dad97451c1"
    ),
    "train_exclusions.json": (
        "7c1ca4f404ba9149093cc3c57dc3607582f671397e2fe99e93449848c1d65d61"
    ),
}
FAMILIES = {
    # The user's asterisk spellings are tokenizer-multitoken redactions.  The
    # actual lexical word has direct Qwen tokens and is what the J readout can
    # score faithfully.
    "profanity": ("damn", "fuck"),
    "celebration": ("yay", "great", "success", "nice"),
}
EXPECTED_TOKEN_IDS = {
    "profanity": [7820, 25090, 26762, 70474, 75021, 76374, 82415, 88619, 95614],
    "celebration": [
        2244, 2393, 5630, 6419, 7188, 8513, 13047, 21396, 28859,
        33941, 39308, 44978, 46891, 52796, 60993, 97559, 138496,
    ],
}
VARIANTS = {
    "celebration_ultradense": "configs/word_explore_celebration_ultradense.json",
    "profanity_ultradense": "configs/word_explore_profanity_ultradense.json",
    "celebration_taper": "configs/word_explore_celebration_taper.json",
    "profanity_taper": "configs/word_explore_profanity_taper.json",
    "solved_u5_control": "configs/word_explore_solved_u5_control.json",
    "solved_u5_low_lr": "configs/word_explore_solved_u5_low_lr.json",
    "solved_u5_taper": "configs/word_explore_solved_u5_taper.json",
    "solved_u5_taper_low_lr": "configs/word_explore_solved_u5_taper_low_lr.json",
}
PRIORITY = tuple(VARIANTS)
EXPECTED_STEPS = (0, 2, 4, 6, 10, 15, 20, 25)
GATE_STEPS = (0, 2, 4, 6)


app = modal.App("j-lens-rl-alternative-screen-v1")
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
        LOCAL_ARTIFACTS / "qwen25_05b_solved_calibration.json",
        (REMOTE_REPO / "artifacts/qwen25_05b_solved_calibration.json").as_posix(),
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


def _run(command: list[str], *, cwd: Path = REMOTE_REPO) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _git(*args: str, repo: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _curve_pass(curve: dict[int, float]) -> bool:
    return curve[2] > curve[0] and curve[4] >= curve[2] and curve[6] >= curve[4]


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    base = payload.pop("base", None)
    if base:
        parent = _load_config(path.parent / base)
        parent.update(payload)
        return parent
    return payload


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
        "sealed_final_manifest_sha256": (
            "acd2d497dcf96b2f3355925bb34979b9b7b3301e4c394066fc54ea57d093b6e3"
        ),
        "sealed_final_sorted_set_sha256": (
            "80bec8d196a36c1d2f02fb6aa3c7c5ea727a86c0c981d3d0611b0900776d6f74"
        ),
    }
    if any(closeout.get(key) != value for key, value in expected.items()):
        raise RuntimeError("V4 closeout does not prove a failed no-look attempt")
    curve = closeout.get("curve", {})
    if (
        curve.get("passed") is not False
        or curve.get("steps") != [0, 2, 4, 6]
        or curve.get("full_mean_exact_match", {}).get("2")
        <= curve.get("full_mean_exact_match", {}).get("0")
        or curve.get("full_mean_exact_match", {}).get("4")
        >= curve.get("full_mean_exact_match", {}).get("2")
    ):
        raise RuntimeError("V4 failed-curve evidence changed")
    inventory = closeout.get("snapshot_inventory", {})
    if (
        inventory.get("evidence_entries") != ["curve.png", "curve_gate.json"]
        or inventory.get("semantic_run_labels")
        != [f"jlens_seed{seed}" for seed in range(159, 167)]
    ):
        raise RuntimeError("V4 no-look snapshot inventory changed")
    return V4_CLOSEOUT_SHA256


def _validate_v5_preregistration(repo: Path) -> str:
    path = repo / V5_PREREGISTRATION_RELATIVE
    if not path.is_file():
        raise RuntimeError("the conditional V5 preregistration is missing")
    frozen = json.loads(path.read_text())
    screen = frozen.get("alternative_screen", {})
    if (
        frozen.get("v4_closeout_sha256") != V4_CLOSEOUT_SHA256
        or screen.get("outcome_status_at_freeze") != "not launched and not inspected"
        or screen.get("code_sha256") != _sha256(repo / "modal_word_explore.py")
        or screen.get("launcher_sha256") != _sha256(repo / "run_word_screen.sh")
        or screen.get("fit_lens_sha256")
        != _sha256(repo / "src/jlens_rl/fit_lens.py")
        or screen.get("selection_priority") != list(PRIORITY)
    ):
        raise RuntimeError("the conditional V5 preregistration changed")
    actual_configs = {
        path: _sha256(repo / path)
        for path in ["configs/word_explore_common.json", *VARIANTS.values()]
    }
    if screen.get("config_sha256") != actual_configs:
        raise RuntimeError("the preregistered screen configs changed")
    failed_closeout = repo / FAILED_SCREEN_CLOSEOUT_RELATIVE
    if (
        not failed_closeout.is_file()
        or _sha256(failed_closeout) != FAILED_SCREEN_CLOSEOUT_SHA256
        or frozen.get("failed_screen_attempt_closeout_sha256")
        != FAILED_SCREEN_CLOSEOUT_SHA256
    ):
        raise RuntimeError("the no-outcome failed screen closeout changed")
    return _sha256(path)


def _launch_manifest() -> dict[str, Any]:
    status = _git("status", "--porcelain=v1", "--untracked-files=all", repo=LOCAL_REPO)
    if status:
        raise RuntimeError(f"word screen requires a clean committed tree:\n{status}")
    v4_closeout_sha256 = _validate_v4_closeout(LOCAL_REPO)
    v5_preregistration_sha256 = _validate_v5_preregistration(LOCAL_REPO)
    actual_manifests = {
        name: _sha256(LOCAL_MANIFESTS / name) for name in EXPOSED_MANIFEST_SHA256
    }
    if actual_manifests != EXPOSED_MANIFEST_SHA256:
        raise RuntimeError(f"exposed manifest hash mismatch: {actual_manifests}")
    lens = LOCAL_ARTIFACTS / "qwen25_05b_solved_lens.pt"
    if _sha256(lens) != LENS_SHA256:
        raise RuntimeError("target-independent lens transport changed")
    solved_calibration = LOCAL_ARTIFACTS / "qwen25_05b_solved_calibration.json"
    if _sha256(solved_calibration) != SOLVED_CALIBRATION_SHA256:
        raise RuntimeError("solved calibration changed")
    return {
        "claim_id": uuid.uuid4().hex,
        "protocol": "j-lens-rl-alternative-screen-v1",
        "git_commit": _git("rev-parse", "HEAD", repo=LOCAL_REPO),
        "git_status": status,
        "variants": VARIANTS,
        "priority": list(PRIORITY),
        "families": {name: list(words) for name, words in FAMILIES.items()},
        "seed": SEED,
        "expected_steps": list(EXPECTED_STEPS),
        "gate_steps": list(GATE_STEPS),
        "gate_criterion": "step2 > step0 and step4 >= step2 and step6 >= step4",
        "model_revision": MODEL_REVISION,
        "wikitext_revision": WIKITEXT_REVISION,
        "lens_sha256": LENS_SHA256,
        "solved_calibration_sha256": SOLVED_CALIBRATION_SHA256,
        "v4_closeout_sha256": v4_closeout_sha256,
        "v5_preregistration_sha256": v5_preregistration_sha256,
        "failed_screen_attempt_closeout_sha256": FAILED_SCREEN_CLOSEOUT_SHA256,
        "config_sha256": {
            path: _sha256(LOCAL_REPO / path)
            for path in ["configs/word_explore_common.json", *VARIANTS.values()]
        },
        "exposed_manifest_sha256": actual_manifests,
        "data_boundary": "exposed failed-V4 400-item development curve only",
        "unmounted_current_confirmatory_manifests": [
            "sealed_final_indices.json",
            "future_reserve_indices.json",
            "retired_v3_curve_indices.json",
        ],
        "gpu_type": GPU_TYPE,
        "max_parallel_gpus": MAX_GPU_CONTAINERS,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _verify_remote_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("variants") != VARIANTS or manifest.get("priority") != list(PRIORITY):
        raise RuntimeError("word-screen variants or priority changed")
    if manifest.get("families") != {
        name: list(words) for name, words in FAMILIES.items()
    }:
        raise RuntimeError("word families changed")
    if manifest.get("git_commit") != _git("rev-parse", "HEAD", repo=REMOTE_REPO):
        raise RuntimeError("remote commit differs from launch")
    if _git("status", "--porcelain=v1", "--untracked-files=all", repo=REMOTE_REPO):
        raise RuntimeError("remote repository is dirty")
    configs = {
        path: _sha256(REMOTE_REPO / path)
        for path in ["configs/word_explore_common.json", *VARIANTS.values()]
    }
    if manifest.get("config_sha256") != configs:
        raise RuntimeError("remote word-screen config hashes changed")
    if manifest.get("v4_closeout_sha256") != _validate_v4_closeout(REMOTE_REPO):
        raise RuntimeError("remote V4 closeout differs from launch")
    if manifest.get("v5_preregistration_sha256") != _validate_v5_preregistration(
        REMOTE_REPO
    ):
        raise RuntimeError("remote V5 preregistration differs from launch")
    if (
        manifest.get("failed_screen_attempt_closeout_sha256")
        != FAILED_SCREEN_CLOSEOUT_SHA256
    ):
        raise RuntimeError("remote failed-screen closeout differs from launch")
    actual_manifests = {
        name: _sha256(REMOTE_REPO / ".confirmatory/manifests" / name)
        for name in EXPOSED_MANIFEST_SHA256
    }
    if actual_manifests != EXPOSED_MANIFEST_SHA256:
        raise RuntimeError("remote exposed manifests changed")
    if _sha256(REMOTE_REPO / "artifacts/qwen25_05b_solved_lens.pt") != LENS_SHA256:
        raise RuntimeError("remote target-independent lens changed")
    if (
        _sha256(REMOTE_REPO / "artifacts/qwen25_05b_solved_calibration.json")
        != SOLVED_CALIBRATION_SHA256
    ):
        raise RuntimeError("remote solved calibration changed")


def _set_status(claim_id: str, stage: str, **details: Any) -> None:
    claim = json.loads((REMOTE_OUTPUT / "attempt_manifest.json").read_text())
    if claim.get("claim_id") != claim_id:
        raise RuntimeError("word-screen Volume claim mismatch")
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
        raise RuntimeError(f"word-screen Volume is not fresh: {sorted(existing)}")
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
    max_containers=CALIBRATION_MAX_GPU_CONTAINERS,
    timeout=2 * 60 * 60,
    startup_timeout=60 * 60,
    retries=0,
    single_use_containers=True,
    volumes={REMOTE_OUTPUT: output_volume},
)
def calibrate_family(family: str) -> dict[str, Any]:
    if family not in FAMILIES:
        raise ValueError(f"unknown word family: {family}")
    output_volume.reload()
    manifest = json.loads((REMOTE_OUTPUT / "attempt_manifest.json").read_text())
    _verify_remote_manifest(manifest)
    output = REMOTE_OUTPUT / "artifacts" / f"{family}_calibration.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite calibration: {output}")
    command = [
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
        "--num-prompts",
        "100",
        "--calibration-prompts",
        "50",
        "--layers",
        "8,14,20",
        "--seed",
        "42",
    ]
    for word in FAMILIES[family]:
        command.extend(["--target-word", word])
    try:
        _run(command)
        payload = json.loads(output.read_text())
        if (
            payload.get("target_words") != list(FAMILIES[family])
            or payload.get("model_revision") != MODEL_REVISION
            or payload.get("dataset_revision") != WIKITEXT_REVISION
            or payload.get("corpus") != "wikitext"
            or payload.get("lens_sha256") != LENS_SHA256
            or payload.get("token_ids") != EXPECTED_TOKEN_IDS[family]
            or not math.isfinite(float(payload.get("std", 0)))
            or float(payload["std"]) <= 0
        ):
            raise RuntimeError(f"invalid {family} calibration metadata")
        result = {
            "family": family,
            "target_words": list(FAMILIES[family]),
            "calibration_sha256": _sha256(output),
            "lens_sha256": LENS_SHA256,
            "token_ids": payload["token_ids"],
            "mean": payload["mean"],
            "std": payload["std"],
        }
        _write_json(REMOTE_OUTPUT / "artifacts" / f"{family}_manifest.json", result)
        return result
    finally:
        output_volume.commit()


def _materialize_config(label: str) -> tuple[Path, dict[str, Any]]:
    template = REMOTE_REPO / VARIANTS[label]
    config = _load_config(template)
    if label.startswith("solved_"):
        calibration = REMOTE_REPO / "artifacts/qwen25_05b_solved_calibration.json"
        if config.get("target_words") != ["solved"]:
            raise RuntimeError("solved ultradense target changed")
        config["calibration_path"] = str(calibration)
        config["calibration_sha256"] = SOLVED_CALIBRATION_SHA256
    else:
        family = "profanity" if label.startswith("profanity_") else "celebration"
        calibration = REMOTE_OUTPUT / "artifacts" / f"{family}_calibration.json"
        metadata = json.loads(calibration.read_text())
        if metadata.get("target_words") != list(FAMILIES[family]):
            raise RuntimeError(f"{family} calibration target mismatch")
        config["calibration_path"] = str(calibration)
        config["calibration_sha256"] = _sha256(calibration)
    config["lens_sha256"] = LENS_SHA256
    path = REMOTE_OUTPUT / "resolved_configs" / f"{label}.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite resolved config: {path}")
    _write_json(path, config)
    return path, config


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
        raise ValueError(f"unknown word-screen variant: {label}")
    output_volume.reload()
    manifest = json.loads((REMOTE_OUTPUT / "attempt_manifest.json").read_text())
    _verify_remote_manifest(manifest)
    config_path, config = _materialize_config(label)
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
        saved_config_path = run_dir / "resolved_config.json"
        data_path = run_dir / "data_indices.json"
        if json.loads(saved_config_path.read_text()) != config:
            raise RuntimeError(f"{label} saved a different resolved config")
        history_rows = [
            json.loads(line)
            for line in (run_dir / "validation_history.jsonl").read_text().splitlines()
            if line
        ]
        curve = {int(row["step"]): float(row["exact_match"]) for row in history_rows}
        if tuple(sorted(curve)) != EXPECTED_STEPS:
            raise RuntimeError(f"{label} has incomplete curve: {sorted(curve)}")
        curve_manifest = EXPOSED_MANIFEST_SHA256["curve_indices.json"]
        if any(
            row.get("validation_indices_sha256") != curve_manifest
            or row.get("validation_source") != "train"
            for row in history_rows
        ):
            raise RuntimeError(f"{label} used the wrong development curve")
        run_manifest = json.loads((run_dir / "run_manifest.json").read_text())
        if (
            run_manifest.get("git_commit") != manifest["git_commit"]
            or run_manifest.get("git_dirty") is not False
            or run_manifest.get("reward_type") != "jlens"
            or run_manifest.get("config_sha256") != _sha256(config_path)
            or run_manifest.get("resolved_config_sha256") != _sha256(saved_config_path)
            or run_manifest.get("data_indices_sha256") != _sha256(data_path)
            or run_manifest.get("lens_sha256") != LENS_SHA256
            or run_manifest.get("calibration_sha256") != config["calibration_sha256"]
            or "L40S" not in str(run_manifest.get("runtime", {}).get("cuda_device_name"))
        ):
            raise RuntimeError(f"{label} run provenance mismatch")
        indices = json.loads(data_path.read_text())
        train_indices = set(indices["train_source_indices"])
        validation_indices = set(indices["validation_source_indices"])
        excluded = set(
            json.loads(
                (REMOTE_REPO / ".confirmatory/manifests/train_exclusions.json").read_text()
            )["indices"]
        )
        if (
            len(train_indices) != 1000
            or train_indices & validation_indices
            or train_indices & excluded
        ):
            raise RuntimeError(f"{label} violated the data firewall")
        result = {
            "label": label,
            "curve": {str(step): curve[step] for step in EXPECTED_STEPS},
            "gate_steps": list(GATE_STEPS),
            "requested_curve_pattern": _curve_pass(curve),
            "target_words": config["target_words"],
            "score_stride": config["score_stride"],
            "score_components": config["score_components"],
            "literal_target_completion_rate": {
                str(row["step"]): float(row["literal_target_completion_rate"])
                for row in history_rows
            },
            "calibration_sha256": config["calibration_sha256"],
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
    status = json.loads((REMOTE_OUTPUT / "attempt_status.json").read_text())
    if status.get("claim_id") != claim_id or status.get("stage") != "claimed":
        raise RuntimeError("word-screen claim is not available for orchestration")
    _set_status(claim_id, "calibrating")
    output_volume.commit()
    try:
        calibrations = _mapped(calibrate_family, FAMILIES)
        output_volume.reload()
        _set_status(claim_id, "training", calibrations=calibrations)
        output_volume.commit()
        results = _mapped(train_variant, PRIORITY)
        selected = next(
            (result["label"] for result in results if result["requested_curve_pattern"]),
            None,
        )
        output_volume.reload()
        _set_status(
            claim_id,
            "complete",
            calibrations=calibrations,
            results=results,
            selected_first_passing_priority=selected,
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
                "volume": VOLUME_NAME,
                "gpu_type": GPU_TYPE,
                "max_parallel_gpus": MAX_GPU_CONTAINERS,
                "variants": list(PRIORITY),
            },
            indent=2,
        )
    )
