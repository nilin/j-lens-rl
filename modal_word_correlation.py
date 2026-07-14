"""Run the preregistered emotional-word J-space correlation screen on Modal.

The scanner itself lives in :mod:`jlens_rl.word_correlation`; this file owns
only immutable launch provenance, the exposed-data firewall, phased execution,
and durable output bookkeeping.  Discovery and validation are deliberately
separate durable phases.  A single GPU job function enforces the registered
one-GPU ceiling across calibration, discovery, and validation.  The selected
word is written to an immutable lock between the two phases, so validation
workers cannot broaden or revise the selection.

Scanner API contract
--------------------
The source module exposes ``run_calibration``, ``run_shard``,
``merge_discovery``, ``merge_validation``, and ``build_atlas``.  The wrapper
passes only paths and frozen phase/shard identities to those callables.  Each
returns a JSON-serializable dictionary with the provenance fields checked
below.  A discovery merge additionally contains ``selection`` with
``canonical_word``, ``reward_sign``, ``association_direction`` (positive or
negative with correctness), and ``token_ids``.  Validation is given the
byte-pinned selection lock and scores only that selected word.

Only the exposed failed-V4 curve manifest, an outcome-free train-exclusion
manifest needed to reproduce the clean Git checkout, and the target-independent
transport are copied into the image.  The scanner never reads the exclusion
manifest.  No sealed-final, reserve, or retired manifest is available to these
jobs.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal


LOCAL_REPO = Path(__file__).resolve().parent
REMOTE_REPO = Path("/workspace/j-lens-rl")
REMOTE_OUTPUT = Path("/word_correlation")
LOCAL_ARTIFACTS = LOCAL_REPO / "artifacts"
LOCAL_MANIFESTS = LOCAL_REPO / ".confirmatory/manifests"

VOLUME_NAME = "j-lens-rl-word-correlation-v1-20260714e"
GPU_TYPE = "L40S"
NUM_SHARDS = 8
MAX_GPU_CONTAINERS = 1
GLOBAL_MODAL_GPU_LIMIT = 1
GPU_EXCLUSIVE_CONFIRMATION = "confirmed-no-other-modal-gpu-app-running"
CONTROLLER_RECOVERY_POLICY = (
    "same-call automatic restart with a durable single-job ledger, idempotent "
    "workers, and no terminalization of controller KeyboardInterrupt"
)

MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
LENS_SHA256 = "178a9671cbf41882135807bde59b828e36c6f8f98b32c809ea3346860aad10dc"
CURVE_MANIFEST_SHA256 = (
    "ad348fe17d2e6bd6aac691d9bcdbb9da481f675305fa0e05c68e86dad97451c1"
)
TRAIN_EXCLUSIONS_SHA256 = (
    "7c1ca4f404ba9149093cc3c57dc3607582f671397e2fe99e93449848c1d65d61"
)
ORIGINAL_PREREGISTRATION_SHA256 = (
    "5e2ae9d0896edbcc7386ccfcc125f8200fa86f77b2099529028a01e54788516a"
)

LENS_RELATIVE = "artifacts/qwen25_05b_solved_lens.pt"
CURVE_MANIFEST_RELATIVE = ".confirmatory/manifests/curve_indices.json"
TRAIN_EXCLUSIONS_RELATIVE = ".confirmatory/manifests/train_exclusions.json"
CONFIG_RELATIVE = "configs/word_correlation_v1.json"
SCANNER_RELATIVE = "src/jlens_rl/word_correlation.py"
PREREGISTRATION_RELATIVE = "protocol_archive/word_correlation_v1_preregistration.json"
CURRENT_AMENDMENT_RELATIVE = "protocol_archive/word_correlation_v1_amendment5.json"
ATTEMPT4_CLOSEOUT_RELATIVE = (
    "protocol_archive/word_correlation_attempt4_closeout.json"
)
ATTEMPT4_INVENTORY_RELATIVE = (
    "protocol_archive/word_correlation_attempt4_forensic_inventory.json"
)
AMENDMENT4_RELATIVE = "protocol_archive/word_correlation_v1_amendment4.json"

RESUMABLE_STAGES = (
    "claimed",
    "calibrating",
    "discovery_running",
    "discovery_finalizing",
    "selection_locked",
    "validation_running",
    "validation_finalizing",
    "atlas_building",
    "finalizing",
)

FORBIDDEN_MANIFEST_NAMES = (
    "sealed_final_indices.json",
    "future_reserve_indices.json",
    "retired_v3_curve_indices.json",
)


app = modal.App("j-lens-rl-word-correlation-v1")
output_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)

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
        (REMOTE_REPO / LENS_RELATIVE).as_posix(),
        copy=True,
    )
    .add_local_file(
        LOCAL_MANIFESTS / "curve_indices.json",
        (REMOTE_REPO / CURVE_MANIFEST_RELATIVE).as_posix(),
        copy=True,
    )
    .add_local_file(
        LOCAL_MANIFESTS / "train_exclusions.json",
        (REMOTE_REPO / TRAIN_EXCLUSIONS_RELATIVE).as_posix(),
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


def _git(*args: str, repo: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def _directory_hashes(path: Path) -> dict[str, str]:
    if not path.is_dir():
        raise RuntimeError(f"missing output directory: {path}")
    hashes = {
        child.relative_to(path).as_posix(): _sha256(child)
        for child in sorted(path.rglob("*"))
        if child.is_file() and not child.name.endswith(".tmp")
    }
    if not hashes:
        raise RuntimeError(f"empty output directory: {path}")
    return hashes


def _json_result(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{context} must return a JSON object")
    # Fail at the API boundary rather than after hours of GPU work if the
    # source module returns tensors, Paths, NaNs, or another non-durable value.
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{context} returned non-JSON data") from error
    return value


def _candidate_words(config: dict[str, Any]) -> set[str]:
    positive = config.get("positive_bin")
    negative = config.get("negative_bin")
    if (
        not isinstance(positive, list)
        or not positive
        or not isinstance(negative, list)
        or not negative
    ):
        raise RuntimeError("word-correlation config has no frozen emotional bins")
    raw = [*positive, *negative]
    words: set[str] = set()
    for item in raw:
        if isinstance(item, str):
            word = item
        elif isinstance(item, dict):
            word = item.get("word")
        else:
            word = None
        if not isinstance(word, str) or not word or word != word.casefold():
            raise RuntimeError(f"invalid canonical emotional candidate: {item!r}")
        if word in words:
            raise RuntimeError(f"duplicate emotional candidate: {word}")
        words.add(word)
    return words


def _validate_curve_manifest(repo: Path) -> list[int]:
    path = repo / CURVE_MANIFEST_RELATIVE
    if not path.is_file() or _sha256(path) != CURVE_MANIFEST_SHA256:
        raise RuntimeError("the exposed V4 curve manifest is missing or changed")
    payload = _load_json(path)
    indices = payload.get("indices")
    if (
        payload.get("dataset") != "openai/gsm8k"
        or payload.get("subset") != "main"
        or payload.get("split") != "train"
        or not isinstance(indices, list)
        or len(indices) != 400
        or any(not isinstance(index, int) or isinstance(index, bool) for index in indices)
        or len(set(indices)) != 400
    ):
        raise RuntimeError("the exposed V4 curve manifest has invalid contents")
    return indices


def _validate_preregistration(repo: Path) -> tuple[str, str, str, str, set[str]]:
    config_path = repo / CONFIG_RELATIVE
    scanner_path = repo / SCANNER_RELATIVE
    prereg_path = repo / PREREGISTRATION_RELATIVE
    amendment_path = repo / CURRENT_AMENDMENT_RELATIVE
    if (
        not config_path.is_file()
        or not scanner_path.is_file()
        or not prereg_path.is_file()
        or not amendment_path.is_file()
    ):
        raise RuntimeError(
            "word-correlation config, scanner, preregistration, or amendment is missing"
        )
    config_sha256 = _sha256(config_path)
    scanner_sha256 = _sha256(scanner_path)
    launcher_sha256 = _sha256(repo / "modal_word_correlation.py")
    prereg_sha256 = _sha256(prereg_path)
    if prereg_sha256 != ORIGINAL_PREREGISTRATION_SHA256:
        raise RuntimeError("the original word-correlation preregistration changed")
    prereg = _load_json(prereg_path)
    if prereg.get("config_sha256") != config_sha256:
        raise RuntimeError("word-correlation config differs from the original freeze")
    amendment = _load_json(amendment_path)
    amendment4_path = repo / AMENDMENT4_RELATIVE
    attempt4_closeout_path = repo / ATTEMPT4_CLOSEOUT_RELATIVE
    attempt4_inventory_path = repo / ATTEMPT4_INVENTORY_RELATIVE
    if (
        amendment.get("protocol")
        != "j-lens-rl-jspace-word-correlation-v1-amendment5-preemption-replay"
        or amendment.get("original_preregistration_sha256") != prereg_sha256
        or amendment.get("amendment4_sha256") != _sha256(amendment4_path)
        or amendment.get("attempt4_closeout_sha256")
        != _sha256(attempt4_closeout_path)
        or amendment.get("attempt4_forensic_inventory_sha256")
        != _sha256(attempt4_inventory_path)
        or amendment.get("scientific_protocol_changed") is not False
    ):
        raise RuntimeError("the current word-correlation amendment chain is invalid")
    current = amendment.get("new_attempt")
    expected_current = {
        "volume": VOLUME_NAME,
        "gpu_type": GPU_TYPE,
        "max_parallel_gpu_workers": MAX_GPU_CONTAINERS,
        "global_modal_gpu_limit": GLOBAL_MODAL_GPU_LIMIT,
        "no_other_modal_gpu_app_may_overlap": True,
        "scanner_sha256": scanner_sha256,
        "launcher_sha256": launcher_sha256,
        "safe_train_exclusions_sha256": TRAIN_EXCLUSIONS_SHA256,
        "controller_recovery_policy": CONTROLLER_RECOVERY_POLICY,
        "no_attempt4_artifact_may_be_reused": True,
    }
    if not isinstance(current, dict) or any(
        current.get(key) != value for key, value in expected_current.items()
    ):
        raise RuntimeError("the current word-correlation launch differs from amendment 5")
    config = _load_json(config_path)
    expected_config = {
        "protocol": "j-lens-rl-jspace-word-correlation-v1",
        "model_revision": MODEL_REVISION,
        "dataset_revision": GSM8K_REVISION,
        "indices_manifest": CURVE_MANIFEST_RELATIVE,
        "indices_manifest_sha256": CURVE_MANIFEST_SHA256,
        "lens_path": LENS_RELATIVE,
        "lens_sha256": LENS_SHA256,
        "shards": NUM_SHARDS,
    }
    if any(config.get(key) != value for key, value in expected_config.items()):
        raise RuntimeError("word-correlation config provenance changed")
    calibration = config.get("calibration")
    if (
        not isinstance(calibration, dict)
        or calibration.get("revision") != WIKITEXT_REVISION
    ):
        raise RuntimeError("word-correlation calibration revision changed")
    candidates = _candidate_words(config)
    frozen_candidates = prereg.get("emotional_candidates")
    if not isinstance(frozen_candidates, list) or set(frozen_candidates) != candidates:
        raise RuntimeError("preregistered emotional candidates differ from the config")
    return (
        prereg_sha256,
        _sha256(amendment_path),
        config_sha256,
        scanner_sha256,
        candidates,
    )


def _validate_repository_boundary(repo: Path) -> None:
    confirmatory = repo / ".confirmatory/manifests"
    present = sorted(path.name for path in confirmatory.iterdir())
    if present != ["curve_indices.json", "train_exclusions.json"]:
        raise RuntimeError(f"unexpected manifest copied into scanner image: {present}")
    if _sha256(repo / TRAIN_EXCLUSIONS_RELATIVE) != TRAIN_EXCLUSIONS_SHA256:
        raise RuntimeError("the safe train-exclusions manifest is missing or changed")
    for name in FORBIDDEN_MANIFEST_NAMES:
        if (confirmatory / name).exists():
            raise RuntimeError(f"forbidden manifest is available to scanner: {name}")
    artifacts = repo / "artifacts"
    present_artifacts = sorted(path.name for path in artifacts.iterdir())
    if present_artifacts != ["qwen25_05b_solved_lens.pt"]:
        raise RuntimeError(f"unexpected artifact copied into scanner image: {present_artifacts}")


def _launch_manifest(preflight: dict[str, Any]) -> dict[str, Any]:
    status = _git("status", "--porcelain=v1", "--untracked-files=all", repo=LOCAL_REPO)
    if status:
        raise RuntimeError(f"word-correlation launch requires a clean committed tree:\n{status}")
    curve_indices = _validate_curve_manifest(LOCAL_REPO)
    lens_path = LOCAL_REPO / LENS_RELATIVE
    if not lens_path.is_file() or _sha256(lens_path) != LENS_SHA256:
        raise RuntimeError("target-independent lens transport is missing or changed")
    prereg_sha256, amendment_sha256, config_sha256, scanner_sha256, candidates = (
        _validate_preregistration(LOCAL_REPO)
    )
    return {
        "claim_id": uuid.uuid4().hex,
        "protocol": "j-lens-rl-jspace-word-correlation-v1",
        "git_commit": _git("rev-parse", "HEAD", repo=LOCAL_REPO),
        "git_status": status,
        "model_revision": MODEL_REVISION,
        "dataset_revision": GSM8K_REVISION,
        "curve_manifest_sha256": CURVE_MANIFEST_SHA256,
        "curve_manifest_size": len(curve_indices),
        "lens_sha256": LENS_SHA256,
        "config_sha256": config_sha256,
        "scanner_sha256": scanner_sha256,
        "launcher_sha256": _sha256(LOCAL_REPO / "modal_word_correlation.py"),
        "launcher_script_sha256": _sha256(LOCAL_REPO / "run_word_correlation.sh"),
        "preregistration_sha256": prereg_sha256,
        "current_amendment_sha256": amendment_sha256,
        "emotional_candidates": sorted(candidates),
        "phase_order": ["discovery", "selection_lock", "validation"],
        "num_shards_per_phase": NUM_SHARDS,
        "gpu_type": GPU_TYPE,
        "max_parallel_gpu_workers": MAX_GPU_CONTAINERS,
        "global_modal_gpu_limit": GLOBAL_MODAL_GPU_LIMIT,
        "controller_recovery_policy": CONTROLLER_RECOVERY_POLICY,
        "gpu_exclusive_preflight": preflight,
        "data_boundary": (
            "exposed failed-V4 400-item curve plus outcome-free train exclusions"
        ),
        "mounted_inputs": [
            CURVE_MANIFEST_RELATIVE,
            TRAIN_EXCLUSIONS_RELATIVE,
            LENS_RELATIVE,
        ],
        "unmounted_manifests": list(FORBIDDEN_MANIFEST_NAMES),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _verify_remote_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("protocol") != "j-lens-rl-jspace-word-correlation-v1":
        raise RuntimeError("wrong word-correlation protocol")
    if manifest.get("git_commit") != _git("rev-parse", "HEAD", repo=REMOTE_REPO):
        raise RuntimeError("remote commit differs from launch")
    if _git("status", "--porcelain=v1", "--untracked-files=all", repo=REMOTE_REPO):
        raise RuntimeError("remote repository is dirty")
    _validate_repository_boundary(REMOTE_REPO)
    curve_indices = _validate_curve_manifest(REMOTE_REPO)
    if manifest.get("curve_manifest_size") != len(curve_indices):
        raise RuntimeError("remote curve-manifest size differs from launch")
    if _sha256(REMOTE_REPO / LENS_RELATIVE) != LENS_SHA256:
        raise RuntimeError("remote target-independent lens differs from launch")
    prereg_sha256, amendment_sha256, config_sha256, scanner_sha256, candidates = (
        _validate_preregistration(REMOTE_REPO)
    )
    preflight = manifest.get("gpu_exclusive_preflight")
    if (
        not isinstance(preflight, dict)
        or preflight.get("exclusive_gpu_confirmation")
        != GPU_EXCLUSIVE_CONFIRMATION
        or preflight.get("global_modal_gpu_limit") != GLOBAL_MODAL_GPU_LIMIT
        or preflight.get("active_other_modal_apps") != []
        or not isinstance(preflight.get("checked_at_utc"), str)
    ):
        raise RuntimeError("remote launch lacks a valid exclusive-GPU preflight")
    expected = {
        "curve_manifest_sha256": CURVE_MANIFEST_SHA256,
        "lens_sha256": LENS_SHA256,
        "config_sha256": config_sha256,
        "scanner_sha256": scanner_sha256,
        "launcher_sha256": _sha256(REMOTE_REPO / "modal_word_correlation.py"),
        "launcher_script_sha256": _sha256(REMOTE_REPO / "run_word_correlation.sh"),
        "preregistration_sha256": prereg_sha256,
        "current_amendment_sha256": amendment_sha256,
        "emotional_candidates": sorted(candidates),
        "mounted_inputs": [
            CURVE_MANIFEST_RELATIVE,
            TRAIN_EXCLUSIONS_RELATIVE,
            LENS_RELATIVE,
        ],
        "global_modal_gpu_limit": GLOBAL_MODAL_GPU_LIMIT,
        "controller_recovery_policy": CONTROLLER_RECOVERY_POLICY,
        "gpu_type": GPU_TYPE,
        "max_parallel_gpu_workers": MAX_GPU_CONTAINERS,
        "num_shards_per_phase": NUM_SHARDS,
        "phase_order": ["discovery", "selection_lock", "validation"],
        "unmounted_manifests": list(FORBIDDEN_MANIFEST_NAMES),
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise RuntimeError("remote launch provenance differs from local claim")


def _local_operational_preflight() -> dict[str, Any]:
    confirmation = os.environ.get("JLENS_MODAL_GPU_EXCLUSIVE_CONFIRM")
    if confirmation != GPU_EXCLUSIVE_CONFIRMATION:
        raise RuntimeError(
            "refusing correlation launch without an external no-overlap "
            "confirmation; verify all other Modal apps are stopped, then set "
            f"JLENS_MODAL_GPU_EXCLUSIVE_CONFIRM={GPU_EXCLUSIVE_CONFIRMATION}"
        )
    modal_cli = Path(sys.executable).parent / "modal"
    listing_text = subprocess.check_output(
        [str(modal_cli), "app", "list", "--json"],
        cwd=LOCAL_REPO,
        text=True,
    )
    listing = json.loads(listing_text[listing_text.index("[") :])
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
            "refusing correlation launch while another Modal app remains active: "
            f"{active_other_apps}"
        )
    return {
        "exclusive_gpu_confirmation": confirmation,
        "global_modal_gpu_limit": GLOBAL_MODAL_GPU_LIMIT,
        "active_other_modal_apps": active_other_apps,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _set_status(claim_id: str, stage: str, **details: Any) -> None:
    claim = _load_json(REMOTE_OUTPUT / "attempt_manifest.json")
    if claim.get("claim_id") != claim_id:
        raise RuntimeError("word-correlation Volume claim mismatch")
    _write_json(
        REMOTE_OUTPUT / "attempt_status.json",
        {
            "claim_id": claim_id,
            "stage": stage,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            **details,
        },
    )


def _validate_scanner_provenance(payload: dict[str, Any], manifest: dict[str, Any]) -> None:
    calibration = REMOTE_OUTPUT / "artifacts/calibration.json"
    if not calibration.is_file():
        raise RuntimeError("scanner output has no frozen calibration artifact")
    expected = {
        "model_revision": MODEL_REVISION,
        "dataset_revision": GSM8K_REVISION,
        "curve_manifest_sha256": CURVE_MANIFEST_SHA256,
        "lens_sha256": LENS_SHA256,
        "config_sha256": manifest["config_sha256"],
        "scanner_sha256": manifest["scanner_sha256"],
        "calibration_sha256": _sha256(calibration),
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise RuntimeError("scanner output provenance mismatch")


def _validate_shard(
    payload: dict[str, Any],
    *,
    phase: str,
    shard_index: int,
    manifest: dict[str, Any],
    selection_lock_sha256: str | None = None,
) -> dict[str, Any]:
    _validate_scanner_provenance(payload, manifest)
    indices = payload.get("source_indices")
    if (
        payload.get("phase") != phase
        or payload.get("shard_index") != shard_index
        or payload.get("num_shards") != NUM_SHARDS
        or not isinstance(indices, list)
        or not indices
        or any(not isinstance(index, int) or isinstance(index, bool) for index in indices)
        or len(indices) != len(set(indices))
    ):
        raise RuntimeError(f"invalid {phase} shard output for shard {shard_index}")
    if phase == "validation":
        if payload.get("selection_lock_sha256") != selection_lock_sha256:
            raise RuntimeError("validation shard used a different selection lock")
    elif "selection_lock_sha256" in payload:
        raise RuntimeError("discovery shard unexpectedly used a selection lock")
    return payload


def _validate_calibration_payload(
    payload: dict[str, Any], manifest: dict[str, Any]
) -> None:
    expected = {
        "model_revision": MODEL_REVISION,
        "wikitext_revision": WIKITEXT_REVISION,
        "lens_sha256": LENS_SHA256,
        "config_sha256": manifest["config_sha256"],
        "scanner_sha256": manifest["scanner_sha256"],
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise RuntimeError("calibration provenance mismatch")


def _load_committed_calibration(
    manifest: dict[str, Any],
) -> dict[str, Any] | None:
    output = REMOTE_OUTPUT / "artifacts/calibration.json"
    sidecar = REMOTE_OUTPUT / "artifacts/calibration_manifest.json"
    present = (output.is_file(), sidecar.is_file())
    if present == (False, False):
        return None
    if present != (True, True):
        raise RuntimeError("calibration output and manifest are not an atomic pair")
    payload = _load_json(output)
    _validate_calibration_payload(payload, manifest)
    result = _load_json(sidecar)
    expected = {
        "output": output.relative_to(REMOTE_OUTPUT).as_posix(),
        "output_sha256": _sha256(output),
    }
    if result != expected:
        raise RuntimeError("committed calibration manifest is invalid")
    return result


def _selection_identity(phase: str) -> tuple[Path | None, str | None]:
    if phase == "discovery":
        return None, None
    selection_path = REMOTE_OUTPUT / "selection_lock.json"
    if not selection_path.is_file():
        raise RuntimeError("validation cannot start without selection lock")
    return selection_path, _sha256(selection_path)


def _shard_paths(phase: str, shard_index: int) -> tuple[Path, Path]:
    return (
        REMOTE_OUTPUT / phase / "shards" / f"shard-{shard_index:02d}",
        REMOTE_OUTPUT
        / phase
        / "shard_manifests"
        / f"shard-{shard_index:02d}.json",
    )


def _load_committed_shard(
    phase: str,
    shard_index: int,
    manifest: dict[str, Any],
) -> dict[str, Any] | None:
    shard_dir, sidecar = _shard_paths(phase, shard_index)
    present = (shard_dir.is_dir(), sidecar.is_file())
    if present == (False, False):
        return None
    if present != (True, True):
        raise RuntimeError(
            f"{phase} shard {shard_index} output and sidecar are not an atomic pair"
        )
    selection_path, lock_sha256 = _selection_identity(phase)
    del selection_path
    payload = _load_json(shard_dir / "summary.json")
    payload = _validate_shard(
        payload,
        phase=phase,
        shard_index=shard_index,
        manifest=manifest,
        selection_lock_sha256=lock_sha256,
    )
    result = _load_json(sidecar)
    artifact_sha256 = _directory_hashes(shard_dir)
    expected = {
        "phase": phase,
        "shard_index": shard_index,
        "source_indices": payload["source_indices"],
        "output_dir": shard_dir.relative_to(REMOTE_OUTPUT).as_posix(),
        "artifact_sha256": artifact_sha256,
        "summary_sha256": artifact_sha256["summary.json"],
        "selection_lock_sha256": lock_sha256,
    }
    if result != expected:
        raise RuntimeError(f"committed {phase} shard {shard_index} sidecar is invalid")
    return result


def _run_calibration_job(manifest: dict[str, Any]) -> dict[str, Any]:
    existing = _load_committed_calibration(manifest)
    if existing is not None:
        return existing
    from jlens_rl.word_correlation import run_calibration

    with tempfile.TemporaryDirectory(prefix="word-correlation-calibration-") as raw:
        local_output = Path(raw) / "calibration.json"
        returned = run_calibration(
            config_path=REMOTE_REPO / CONFIG_RELATIVE,
            output_path=local_output,
        )
        if local_output.is_file():
            payload = _load_json(local_output)
            if returned is not None and _json_result(
                returned, "run_calibration"
            ) != payload:
                raise RuntimeError(
                    "run_calibration return value differs from its output"
                )
        else:
            payload = _json_result(returned, "run_calibration")
            _write_json(local_output, payload)
        _validate_calibration_payload(payload, manifest)
        output = REMOTE_OUTPUT / "artifacts/calibration.json"
        sidecar = REMOTE_OUTPUT / "artifacts/calibration_manifest.json"
        if output.exists() or sidecar.exists():
            raise RuntimeError("calibration pair appeared during computation")
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_output, output)
        result = {
            "output": output.relative_to(REMOTE_OUTPUT).as_posix(),
            "output_sha256": _sha256(output),
        }
        _write_json(sidecar, result)
        output_volume.commit()
        return result


def _scan_phase(phase: str, shard_index: int, manifest: dict[str, Any]) -> dict[str, Any]:
    if phase not in {"discovery", "validation"}:
        raise ValueError(f"invalid correlation phase: {phase}")
    if shard_index not in range(NUM_SHARDS):
        raise ValueError(f"invalid shard index: {shard_index}")
    existing = _load_committed_shard(phase, shard_index, manifest)
    if existing is not None:
        return existing
    selection_path, lock_sha256 = _selection_identity(phase)
    from jlens_rl.word_correlation import run_shard

    with tempfile.TemporaryDirectory(
        prefix=f"word-correlation-{phase}-{shard_index:02d}-"
    ) as raw:
        local_dir = Path(raw) / "shard"
        local_dir.mkdir()
        payload = _json_result(
            run_shard(
                config_path=REMOTE_REPO / CONFIG_RELATIVE,
                phase=phase,
                shard=shard_index,
                output_dir=local_dir,
                calibration_path=REMOTE_OUTPUT / "artifacts/calibration.json",
                selection_path=selection_path,
            ),
            f"run_shard({phase}, {shard_index})",
        )
        payload = _validate_shard(
            payload,
            phase=phase,
            shard_index=shard_index,
            manifest=manifest,
            selection_lock_sha256=lock_sha256,
        )
        summary = local_dir / "summary.json"
        if summary.exists():
            raise RuntimeError("run_shard must return, not prewrite, summary.json")
        _write_json(summary, payload)
        artifact_sha256 = _directory_hashes(local_dir)
        shard_dir, sidecar = _shard_paths(phase, shard_index)
        if shard_dir.exists() or sidecar.exists():
            raise RuntimeError(f"{phase} shard {shard_index} appeared during computation")
        shard_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(local_dir, shard_dir)
        result = {
            "phase": phase,
            "shard_index": shard_index,
            "source_indices": payload["source_indices"],
            "output_dir": shard_dir.relative_to(REMOTE_OUTPUT).as_posix(),
            "artifact_sha256": artifact_sha256,
            "summary_sha256": artifact_sha256["summary.json"],
            "selection_lock_sha256": lock_sha256,
        }
        _write_json(sidecar, result)
        output_volume.commit()
        return result


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
    if REMOTE_OUTPUT.exists():
        existing = [path.name for path in REMOTE_OUTPUT.iterdir()]
    else:
        existing = []
        REMOTE_OUTPUT.mkdir(parents=True)
    if existing:
        raise RuntimeError(f"word-correlation Volume is not fresh: {sorted(existing)}")
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
    volumes={REMOTE_OUTPUT: output_volume},
)
def gpu_job(kind: str, shard_index: int | None = None) -> dict[str, Any]:
    output_volume.reload()
    manifest = _load_json(REMOTE_OUTPUT / "attempt_manifest.json")
    _verify_remote_manifest(manifest)
    if kind == "calibration":
        if shard_index is not None:
            raise ValueError("calibration job cannot have a shard index")
        return _run_calibration_job(manifest)
    if kind not in {"discovery", "validation"} or shard_index is None:
        raise ValueError("invalid word-correlation GPU job")
    return _scan_phase(kind, shard_index, manifest)


def _phase_indices(results: list[dict[str, Any]], phase: str) -> set[int]:
    if len(results) != NUM_SHARDS:
        raise RuntimeError(f"{phase} did not return all {NUM_SHARDS} shards")
    observed_shards = {result.get("shard_index") for result in results}
    if observed_shards != set(range(NUM_SHARDS)):
        raise RuntimeError(f"{phase} shard identities are incomplete")
    flattened = [index for result in results for index in result["source_indices"]]
    if len(flattened) != len(set(flattened)):
        raise RuntimeError(f"{phase} source indices overlap across shards")
    return set(flattened)


def _validate_aggregate_payload(
    phase: str,
    payload: dict[str, Any],
    shard_results: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    _validate_scanner_provenance(payload, manifest)
    if payload.get("phase") != phase:
        raise RuntimeError(f"wrong phase in {phase} aggregate")
    expected_indices = _phase_indices(shard_results, phase)
    indices = payload.get("source_indices")
    if not isinstance(indices, list) or set(indices) != expected_indices:
        raise RuntimeError(f"{phase} aggregate source indices differ from shards")
    if phase == "validation":
        selection = REMOTE_OUTPUT / "selection_lock.json"
        if (
            not selection.is_file()
            or payload.get("selection_lock_sha256") != _sha256(selection)
        ):
            raise RuntimeError("validation aggregate used a different selection lock")


def _aggregate_paths(phase: str) -> tuple[Path, Path, Path]:
    phase_dir = REMOTE_OUTPUT / phase
    return (
        phase_dir / "aggregate.json",
        phase_dir / "merged",
        phase_dir / "aggregate_manifest.json",
    )


def _load_committed_aggregate(
    phase: str,
    shard_results: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> tuple[Path, dict[str, Any]] | None:
    output, merge_dir, sidecar = _aggregate_paths(phase)
    present = (output.is_file(), merge_dir.is_dir(), sidecar.is_file())
    if present == (False, False, False):
        return None
    if present != (True, True, True):
        raise RuntimeError(f"{phase} aggregate is not an atomic artifact set")
    payload = _load_json(output)
    _validate_aggregate_payload(phase, payload, shard_results, manifest)
    merge_hashes = _directory_hashes(merge_dir)
    provenance = merge_dir / "merge_provenance.json"
    if not provenance.is_file() or _load_json(provenance) != payload:
        raise RuntimeError(f"{phase} merge provenance differs from aggregate")
    expected = {
        "phase": phase,
        "aggregate_sha256": _sha256(output),
        "merge_artifact_sha256": merge_hashes,
    }
    if _load_json(sidecar) != expected:
        raise RuntimeError(f"committed {phase} aggregate manifest is invalid")
    return output, payload


def _merge_to_local(
    phase: str,
    shard_results: list[dict[str, Any]],
    manifest: dict[str, Any],
    merge_dir: Path,
) -> dict[str, Any]:
    shard_dirs: list[Path] = []
    for result in shard_results:
        shard_dir = REMOTE_OUTPUT / result["output_dir"]
        if _directory_hashes(shard_dir) != result["artifact_sha256"]:
            raise RuntimeError(f"{phase} shard changed before aggregation")
        shard_dirs.append(shard_dir)
    if phase == "validation":
        from jlens_rl.word_correlation import merge_validation

        merged = merge_validation(
            config_path=REMOTE_REPO / CONFIG_RELATIVE,
            shard_dirs=shard_dirs,
            calibration_path=REMOTE_OUTPUT / "artifacts/calibration.json",
            selection_path=REMOTE_OUTPUT / "selection_lock.json",
            output_dir=merge_dir,
        )
    else:
        from jlens_rl.word_correlation import merge_discovery

        merged = merge_discovery(
            config_path=REMOTE_REPO / CONFIG_RELATIVE,
            shard_dirs=shard_dirs,
            calibration_path=REMOTE_OUTPUT / "artifacts/calibration.json",
            output_dir=merge_dir,
        )
    payload = _json_result(merged, f"merge_{phase}")
    _validate_aggregate_payload(phase, payload, shard_results, manifest)
    return payload


def _publish_aggregate(
    phase: str, merge_dir: Path, payload: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    output, remote_merge, sidecar = _aggregate_paths(phase)
    if output.exists() or remote_merge.exists() or sidecar.exists():
        raise RuntimeError(f"{phase} aggregate appeared during computation")
    remote_merge.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(merge_dir, remote_merge)
    _write_json(output, payload)
    _write_json(
        sidecar,
        {
            "phase": phase,
            "aggregate_sha256": _sha256(output),
            "merge_artifact_sha256": _directory_hashes(remote_merge),
        },
    )
    return output, payload


def _validated_selection(
    discovery: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    selection = discovery.get("selection")
    if not isinstance(selection, dict):
        raise RuntimeError("discovery aggregate did not select an emotional word")
    word = selection.get("canonical_word")
    direction = selection.get("association_direction")
    reward_sign = selection.get("reward_sign")
    token_ids = selection.get("token_ids")
    config = _load_json(REMOTE_REPO / CONFIG_RELATIVE)
    expected_token_ids = config.get("expected_token_ids", {}).get(word)
    if (
        word not in set(manifest["emotional_candidates"])
        or direction not in {
            "positive_with_correctness",
            "negative_with_correctness",
        }
        or reward_sign not in {-1, 1}
        or isinstance(reward_sign, bool)
        or (direction == "positive_with_correctness") != (reward_sign == 1)
        or not isinstance(token_ids, list)
        or not token_ids
        or any(
            not isinstance(token_id, int) or isinstance(token_id, bool)
            for token_id in token_ids
        )
        or len(token_ids) != len(set(token_ids))
        or token_ids != expected_token_ids
    ):
        raise RuntimeError(f"invalid discovery selection: {selection!r}")
    return selection


def _lock_selection(
    discovery_path: Path,
    discovery: dict[str, Any],
    manifest: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    selection = _validated_selection(discovery, manifest)
    path = REMOTE_OUTPUT / "selection_lock.json"
    if path.exists():
        raise FileExistsError("refusing to overwrite selection lock")
    lock = {
        "protocol": manifest["protocol"],
        "claim_id": manifest["claim_id"],
        "git_commit": manifest["git_commit"],
        "config_sha256": manifest["config_sha256"],
        "scanner_sha256": manifest["scanner_sha256"],
        "curve_manifest_sha256": CURVE_MANIFEST_SHA256,
        "lens_sha256": LENS_SHA256,
        "calibration_sha256": _sha256(
            REMOTE_OUTPUT / "artifacts/calibration.json"
        ),
        "discovery_aggregate_sha256": _sha256(discovery_path),
        "selection": selection,
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(path, lock)
    return path, lock


def _load_selection_lock(
    discovery_path: Path,
    discovery: dict[str, Any],
    manifest: dict[str, Any],
) -> tuple[Path, dict[str, Any]] | None:
    path = REMOTE_OUTPUT / "selection_lock.json"
    if not path.exists():
        return None
    lock = _load_json(path)
    selection = _validated_selection(discovery, manifest)
    expected = {
        "protocol": manifest["protocol"],
        "claim_id": manifest["claim_id"],
        "git_commit": manifest["git_commit"],
        "config_sha256": manifest["config_sha256"],
        "scanner_sha256": manifest["scanner_sha256"],
        "curve_manifest_sha256": CURVE_MANIFEST_SHA256,
        "lens_sha256": LENS_SHA256,
        "calibration_sha256": _sha256(
            REMOTE_OUTPUT / "artifacts/calibration.json"
        ),
        "discovery_aggregate_sha256": _sha256(discovery_path),
        "selection": selection,
    }
    if (
        not isinstance(lock.get("locked_at_utc"), str)
        or any(lock.get(key) != value for key, value in expected.items())
    ):
        raise RuntimeError("committed selection lock is invalid")
    return path, lock


def _ensure_discovery_finalized(
    discovery_results: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    aggregate = _load_committed_aggregate(
        "discovery", discovery_results, manifest
    )
    lock_path = REMOTE_OUTPUT / "selection_lock.json"
    if aggregate is None:
        if lock_path.exists():
            raise RuntimeError("selection lock exists without discovery aggregate")
        with tempfile.TemporaryDirectory(
            prefix="word-correlation-discovery-merge-"
        ) as raw:
            local_merge = Path(raw) / "merged"
            local_merge.mkdir()
            payload = _merge_to_local(
                "discovery", discovery_results, manifest, local_merge
            )
            discovery_path, discovery = _publish_aggregate(
                "discovery", local_merge, payload
            )
            selection_path, selection_lock = _lock_selection(
                discovery_path, discovery, manifest
            )
            output_volume.commit()
            return (
                discovery_path,
                discovery,
                selection_path,
                selection_lock,
            )
    discovery_path, discovery = aggregate
    selection = _load_selection_lock(discovery_path, discovery, manifest)
    if selection is None:
        raise RuntimeError("discovery aggregate exists without selection lock")
    selection_path, selection_lock = selection
    return discovery_path, discovery, selection_path, selection_lock


def _ensure_validation_finalized(
    validation_results: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    aggregate = _load_committed_aggregate(
        "validation", validation_results, manifest
    )
    if aggregate is not None:
        return aggregate
    with tempfile.TemporaryDirectory(
        prefix="word-correlation-validation-merge-"
    ) as raw:
        local_merge = Path(raw) / "merged"
        local_merge.mkdir()
        payload = _merge_to_local(
            "validation", validation_results, manifest, local_merge
        )
        result = _publish_aggregate("validation", local_merge, payload)
        output_volume.commit()
        return result


def _load_committed_atlas(
    discovery_results: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> tuple[Path, dict[str, Any]] | None:
    output_dir = REMOTE_OUTPUT / "atlas"
    sidecar = REMOTE_OUTPUT / "atlas_manifest.json"
    present = (output_dir.is_dir(), sidecar.is_file())
    if present == (False, False):
        return None
    if present != (True, True):
        raise RuntimeError("lexical atlas is not an atomic artifact set")
    output = output_dir / "atlas.json"
    payload = _load_json(output)
    _validate_scanner_provenance(payload, manifest)
    if payload.get("phase") != "discovery":
        raise RuntimeError("lexical atlas is not discovery-only")
    discovery_indices = _phase_indices(discovery_results, "discovery")
    if set(payload.get("source_indices", [])) != discovery_indices:
        raise RuntimeError("lexical atlas source indices differ from discovery")
    expected = {
        "atlas_sha256": _sha256(output),
        "atlas_artifact_sha256": _directory_hashes(output_dir),
    }
    if _load_json(sidecar) != expected:
        raise RuntimeError("committed lexical atlas manifest is invalid")
    return output, payload


def _ensure_atlas(
    discovery_results: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    existing = _load_committed_atlas(discovery_results, manifest)
    if existing is not None:
        return existing
    shard_dirs: list[Path] = []
    for result in discovery_results:
        shard_dir = REMOTE_OUTPUT / result["output_dir"]
        if _directory_hashes(shard_dir) != result["artifact_sha256"]:
            raise RuntimeError("discovery shard changed before atlas construction")
        shard_dirs.append(shard_dir)
    from jlens_rl.word_correlation import build_atlas
    with tempfile.TemporaryDirectory(prefix="word-correlation-atlas-") as raw:
        local_dir = Path(raw) / "atlas"
        local_dir.mkdir()
        payload = _json_result(
            build_atlas(
                config_path=REMOTE_REPO / CONFIG_RELATIVE,
                shard_dirs=shard_dirs,
                calibration_path=REMOTE_OUTPUT / "artifacts/calibration.json",
                output_dir=local_dir,
            ),
            "build_atlas",
        )
        _validate_scanner_provenance(payload, manifest)
        if payload.get("phase") != "discovery":
            raise RuntimeError("lexical atlas is not discovery-only")
        discovery_indices = _phase_indices(discovery_results, "discovery")
        if set(payload.get("source_indices", [])) != discovery_indices:
            raise RuntimeError("lexical atlas source indices differ from discovery")
        local_output = local_dir / "atlas.json"
        if local_output.exists():
            raise RuntimeError("build_atlas must return, not prewrite, atlas.json")
        _write_json(local_output, payload)
        output_dir = REMOTE_OUTPUT / "atlas"
        sidecar = REMOTE_OUTPUT / "atlas_manifest.json"
        if output_dir.exists() or sidecar.exists():
            raise RuntimeError("lexical atlas appeared during construction")
        shutil.copytree(local_dir, output_dir)
        output = output_dir / "atlas.json"
        _write_json(
            sidecar,
            {
                "atlas_sha256": _sha256(output),
                "atlas_artifact_sha256": _directory_hashes(output_dir),
            },
        )
        output_volume.commit()
        return output, payload


def _controller_state_path() -> Path:
    return REMOTE_OUTPUT / "controller_state.json"


def _load_controller_state(claim_id: str, call_id: str) -> dict[str, Any]:
    path = _controller_state_path()
    if not path.is_file():
        status = _load_json(REMOTE_OUTPUT / "attempt_status.json")
        if status.get("claim_id") != claim_id or status.get("stage") != "claimed":
            raise RuntimeError("controller state can only be created for a fresh claim")
        state = {
            "protocol": "j-lens-rl-word-correlation-controller-v1",
            "claim_id": claim_id,
            "orchestrator_call_id": call_id,
            "active_job": None,
            "completed_jobs": [],
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(path, state)
        output_volume.commit()
        return state
    state = _load_json(path)
    if (
        state.get("protocol") != "j-lens-rl-word-correlation-controller-v1"
        or state.get("claim_id") != claim_id
        or state.get("orchestrator_call_id") != call_id
        or not isinstance(state.get("completed_jobs"), list)
        or (
            state.get("active_job") is not None
            and not isinstance(state.get("active_job"), dict)
        )
    ):
        raise RuntimeError("word-correlation controller state is invalid")
    return state


def _store_controller_state(state: dict[str, Any]) -> None:
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(_controller_state_path(), state)


def _gpu_job_spec(kind: str, shard_index: int | None) -> dict[str, Any]:
    key = "calibration" if kind == "calibration" else f"{kind}:{shard_index:02d}"
    return {
        "key": key,
        "kind": kind,
        "shard_index": shard_index,
    }


def _load_gpu_job_result(
    kind: str,
    shard_index: int | None,
    manifest: dict[str, Any],
) -> dict[str, Any] | None:
    if kind == "calibration":
        return _load_committed_calibration(manifest)
    if shard_index is None:
        raise RuntimeError("shard job has no shard index")
    return _load_committed_shard(kind, shard_index, manifest)


def _complete_controller_job(
    state: dict[str, Any], spec: dict[str, Any], call_id: str | None
) -> None:
    active = state.get("active_job")
    if active is not None and any(
        active.get(key) != value for key, value in spec.items()
    ):
        raise RuntimeError("controller active job differs from completed artifact")
    completed = state["completed_jobs"]
    if not any(item.get("key") == spec["key"] for item in completed):
        completed.append({**spec, "call_id": call_id})
    state["active_job"] = None
    _store_controller_state(state)
    output_volume.commit()


def _durable_gpu_job(
    claim_id: str,
    call_id: str,
    kind: str,
    shard_index: int | None,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    spec = _gpu_job_spec(kind, shard_index)
    output_volume.reload()
    state = _load_controller_state(claim_id, call_id)
    existing = _load_gpu_job_result(kind, shard_index, manifest)
    active = state.get("active_job")
    if existing is not None:
        active_call_id = active.get("call_id") if isinstance(active, dict) else None
        _complete_controller_job(state, spec, active_call_id)
        return existing
    if active is None:
        state["active_job"] = {**spec, "call_id": None}
        _store_controller_state(state)
        output_volume.commit()
        active = state["active_job"]
    elif any(active.get(key) != value for key, value in spec.items()):
        raise RuntimeError("controller has a different unfinished GPU job")

    active_call_id = active.get("call_id")
    if active_call_id is None:
        call = gpu_job.spawn(kind, shard_index)
        spawned_call_id = str(call.object_id)
        output_volume.reload()
        state = _load_controller_state(claim_id, call_id)
        active = state.get("active_job")
        if not isinstance(active, dict) or any(
            active.get(key) != value for key, value in spec.items()
        ):
            raise RuntimeError("controller job intent changed during dispatch")
        if active.get("call_id") not in {None, spawned_call_id}:
            raise RuntimeError("controller job acquired a conflicting call ID")
        active["call_id"] = spawned_call_id
        _store_controller_state(state)
        output_volume.commit()
        active_call_id = spawned_call_id
    else:
        call = modal.FunctionCall.from_id(str(active_call_id))

    try:
        call.get()
    except Exception:
        output_volume.reload()
        existing = _load_gpu_job_result(kind, shard_index, manifest)
        if existing is None:
            raise
    output_volume.reload()
    existing = _load_gpu_job_result(kind, shard_index, manifest)
    if existing is None:
        raise RuntimeError(f"GPU job {spec['key']} returned without a durable artifact")
    state = _load_controller_state(claim_id, call_id)
    _complete_controller_job(state, spec, str(active_call_id))
    return existing


def _ensure_phase_shards(
    claim_id: str,
    call_id: str,
    phase: str,
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    results = [
        _durable_gpu_job(claim_id, call_id, phase, shard, manifest)
        for shard in range(NUM_SHARDS)
    ]
    _phase_indices(results, phase)
    return results


_STAGE_ORDER = {stage: index for index, stage in enumerate(RESUMABLE_STAGES)}


def _advance_status(claim_id: str, target: str, **details: Any) -> None:
    status = _load_json(REMOTE_OUTPUT / "attempt_status.json")
    current = status.get("stage")
    if current not in _STAGE_ORDER or target not in _STAGE_ORDER:
        raise RuntimeError("cannot advance an invalid or terminal controller stage")
    if _STAGE_ORDER[current] < _STAGE_ORDER[target]:
        _set_status(claim_id, target, **details)
        output_volume.commit()


def _require_all_committed_shards(
    phase: str, manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for shard in range(NUM_SHARDS):
        result = _load_committed_shard(phase, shard, manifest)
        if result is None:
            raise RuntimeError(f"resume stage requires completed {phase} shard {shard}")
        results.append(result)
    _phase_indices(results, phase)
    return results


def _validate_resume_boundary(stage: str, manifest: dict[str, Any]) -> None:
    rank = _STAGE_ORDER[stage]
    if rank >= _STAGE_ORDER["discovery_running"] and _load_committed_calibration(
        manifest
    ) is None:
        raise RuntimeError("resume stage advanced past missing calibration")
    discovery: list[dict[str, Any]] | None = None
    if rank >= _STAGE_ORDER["discovery_finalizing"]:
        discovery = _require_all_committed_shards("discovery", manifest)
    if rank >= _STAGE_ORDER["selection_locked"]:
        if discovery is None:
            discovery = _require_all_committed_shards("discovery", manifest)
        aggregate = _load_committed_aggregate("discovery", discovery, manifest)
        if aggregate is None or _load_selection_lock(*aggregate, manifest) is None:
            raise RuntimeError("resume stage advanced past missing selection lock")
    validation: list[dict[str, Any]] | None = None
    if rank >= _STAGE_ORDER["validation_finalizing"]:
        validation = _require_all_committed_shards("validation", manifest)
    if rank >= _STAGE_ORDER["atlas_building"]:
        if validation is None:
            validation = _require_all_committed_shards("validation", manifest)
        if _load_committed_aggregate("validation", validation, manifest) is None:
            raise RuntimeError("resume stage advanced past missing validation aggregate")
    if rank >= _STAGE_ORDER["finalizing"]:
        if discovery is None:
            discovery = _require_all_committed_shards("discovery", manifest)
        if _load_committed_atlas(discovery, manifest) is None:
            raise RuntimeError("resume stage advanced past missing lexical atlas")


def _finalize_result(
    claim_id: str,
    manifest: dict[str, Any],
    calibration: dict[str, Any],
    discovery_shards: list[dict[str, Any]],
    discovery_path: Path,
    selection_path: Path,
    selection_lock: dict[str, Any],
    validation_shards: list[dict[str, Any]],
    validation_path: Path,
    atlas_path: Path,
) -> dict[str, Any]:
    result_path = REMOTE_OUTPUT / "result_manifest.json"
    status = _load_json(REMOTE_OUTPUT / "attempt_status.json")
    if result_path.exists():
        if status.get("stage") != "complete":
            raise RuntimeError("result manifest exists without terminal complete status")
        result = _load_json(result_path)
        result_sha256 = _sha256(result_path)
        if status.get("result_manifest_sha256") != result_sha256:
            raise RuntimeError("complete status has the wrong result manifest hash")
        return {
            "stage": "complete",
            "result_manifest_sha256": result_sha256,
            "selection": result.get("selection"),
        }

    discovery_indices = _phase_indices(discovery_shards, "discovery")
    validation_indices = _phase_indices(validation_shards, "validation")
    full_indices = set(_validate_curve_manifest(REMOTE_REPO))
    if discovery_indices & validation_indices:
        raise RuntimeError("discovery and selected-word validation overlap")
    if discovery_indices | validation_indices != full_indices:
        raise RuntimeError("discovery and validation do not partition the curve set")
    result_manifest = {
        "protocol": manifest["protocol"],
        "claim_id": claim_id,
        "git_commit": manifest["git_commit"],
        "config_sha256": manifest["config_sha256"],
        "scanner_sha256": manifest["scanner_sha256"],
        "preregistration_sha256": manifest["preregistration_sha256"],
        "current_amendment_sha256": manifest["current_amendment_sha256"],
        "curve_manifest_sha256": CURVE_MANIFEST_SHA256,
        "lens_sha256": LENS_SHA256,
        "calibration_sha256": calibration["output_sha256"],
        "controller_state_sha256": _sha256(_controller_state_path()),
        "discovery_shard_artifact_sha256": {
            str(result["shard_index"]): result["artifact_sha256"]
            for result in discovery_shards
        },
        "discovery_aggregate_sha256": _sha256(discovery_path),
        "selection_lock_sha256": _sha256(selection_path),
        "selection": selection_lock["selection"],
        "validation_shard_artifact_sha256": {
            str(result["shard_index"]): result["artifact_sha256"]
            for result in validation_shards
        },
        "validation_aggregate_sha256": _sha256(validation_path),
        "atlas_sha256": _sha256(atlas_path),
        "atlas_artifact_sha256": _directory_hashes(atlas_path.parent),
        "discovery_size": len(discovery_indices),
        "validation_size": len(validation_indices),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(result_path, result_manifest)
    result_sha256 = _sha256(result_path)
    _set_status(
        claim_id,
        "complete",
        result_manifest_sha256=result_sha256,
        selection=selection_lock["selection"],
    )
    output_volume.commit()
    return {
        "stage": "complete",
        "result_manifest_sha256": result_sha256,
        "selection": selection_lock["selection"],
    }


@app.function(
    image=repo_image,
    cpu=4,
    memory=16384,
    max_containers=1,
    timeout=10 * 60 * 60,
    startup_timeout=60 * 60,
    retries=0,
    volumes={REMOTE_OUTPUT: output_volume},
)
def orchestrate(claim_id: str) -> dict[str, Any]:
    output_volume.reload()
    manifest = _load_json(REMOTE_OUTPUT / "attempt_manifest.json")
    status = _load_json(REMOTE_OUTPUT / "attempt_status.json")
    if status.get("claim_id") != claim_id:
        raise RuntimeError("word-correlation claim is not available for orchestration")
    _verify_remote_manifest(manifest)
    if status.get("stage") == "complete":
        result = REMOTE_OUTPUT / "result_manifest.json"
        if not result.is_file() or status.get("result_manifest_sha256") != _sha256(result):
            raise RuntimeError("terminal correlation result is incomplete")
        payload = _load_json(result)
        return {
            "stage": "complete",
            "result_manifest_sha256": _sha256(result),
            "selection": payload.get("selection"),
        }
    if status.get("stage") not in RESUMABLE_STAGES:
        raise RuntimeError("word-correlation claim is terminal or not resumable")
    root_call_id = modal.current_function_call_id()
    if not isinstance(root_call_id, str) or not root_call_id:
        raise RuntimeError("orchestrator has no durable Modal function-call identity")
    try:
        _load_controller_state(claim_id, root_call_id)
        output_volume.reload()
        status = _load_json(REMOTE_OUTPUT / "attempt_status.json")
        _validate_resume_boundary(str(status["stage"]), manifest)

        _advance_status(claim_id, "calibrating")
        calibration = _durable_gpu_job(
            claim_id, root_call_id, "calibration", None, manifest
        )
        output_volume.reload()
        _advance_status(claim_id, "discovery_running")
        discovery_shards = _ensure_phase_shards(
            claim_id, root_call_id, "discovery", manifest
        )
        output_volume.reload()
        _advance_status(claim_id, "discovery_finalizing")
        (
            discovery_path,
            _discovery,
            selection_path,
            selection_lock,
        ) = _ensure_discovery_finalized(discovery_shards, manifest)
        selection_sha256 = _sha256(selection_path)
        _advance_status(
            claim_id,
            "selection_locked",
            discovery_aggregate_sha256=_sha256(discovery_path),
            selection_lock_sha256=selection_sha256,
            selection=selection_lock["selection"],
        )
        _advance_status(
            claim_id,
            "validation_running",
            selection_lock_sha256=selection_sha256,
            selection=selection_lock["selection"],
        )
        validation_shards = _ensure_phase_shards(
            claim_id, root_call_id, "validation", manifest
        )
        output_volume.reload()
        _advance_status(claim_id, "validation_finalizing")
        validation_path, _validation = _ensure_validation_finalized(
            validation_shards, manifest
        )
        discovery_indices = _phase_indices(discovery_shards, "discovery")
        validation_indices = _phase_indices(validation_shards, "validation")
        full_indices = set(_validate_curve_manifest(REMOTE_REPO))
        if discovery_indices & validation_indices:
            raise RuntimeError("discovery and selected-word validation overlap")
        if discovery_indices | validation_indices != full_indices:
            raise RuntimeError("discovery and validation do not partition the curve set")
        _advance_status(
            claim_id,
            "atlas_building",
            selection_lock_sha256=selection_sha256,
            validation_aggregate_sha256=_sha256(validation_path),
        )
        atlas_path, _atlas = _ensure_atlas(discovery_shards, manifest)
        _advance_status(
            claim_id,
            "finalizing",
            atlas_sha256=_sha256(atlas_path),
            selection_lock_sha256=selection_sha256,
        )
        return _finalize_result(
            claim_id,
            manifest,
            calibration,
            discovery_shards,
            discovery_path,
            selection_path,
            selection_lock,
            validation_shards,
            validation_path,
            atlas_path,
        )
    except KeyboardInterrupt:
        # Modal restarts a preempted Function with the same call ID.  Durable
        # stage/job checkpoints deliberately remain nonterminal for re-entry.
        raise
    except Exception as error:
        try:
            output_volume.reload()
            _set_status(claim_id, "failed", error=repr(error))
            output_volume.commit()
        except Exception:
            pass
        raise


@app.local_entrypoint()
def main() -> None:
    preflight = _local_operational_preflight()
    manifest = _launch_manifest(preflight)
    claim_attempt.remote(manifest)
    call = orchestrate.spawn(str(manifest["claim_id"]))
    print(
        json.dumps(
            {
                "status": "submitted",
                "function_call_id": call.object_id,
                "volume": VOLUME_NAME,
                "gpu_type": GPU_TYPE,
                "parallel_workers_per_phase": MAX_GPU_CONTAINERS,
                "phase_order": manifest["phase_order"],
            },
            indent=2,
        )
    )
