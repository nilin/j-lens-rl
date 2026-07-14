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
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
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
GPU_LEASE_ENVIRONMENT_NAME = "main"
GPU_LEASE_DICT_NAME = "j-lens-rl-global-gpu-lease-v1"
GPU_LEASE_PROTOCOL = "j-lens-rl-global-gpu-lease-v1"
GPU_LEASE_KEY_PREFIX = "word-correlation"
GPU_LEASE_SLOT = "global-one-gpu"
PUBLICATION_DICT_NAME = "j-lens-rl-word-correlation-v1-20260714e-publications-v1"
ROOT_AUTHORITY_DICT_NAME = "j-lens-rl-word-correlation-v1-20260714e-roots-v1"
ROOT_AUTHORITY_PROTOCOL = "j-lens-rl-word-correlation-root-authority-v1"
SUBMISSION_RECOVERY_SECONDS = 15.0
SUBMISSION_BIND_SECONDS = 60.0
CONTROLLER_RECOVERY_POLICY = (
    "same-call automatic restart with immutable generation markers, an "
    "idempotent claim/submission ledger, a durable cross-app GPU lease, a "
    "durable single-job ledger, and no terminalization of controller "
    "KeyboardInterrupt"
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

GENERATION_RELATIVE = "generations"
COMMIT_MARKER_RELATIVE = "commit_markers"

FORBIDDEN_MANIFEST_NAMES = (
    "sealed_final_indices.json",
    "future_reserve_indices.json",
    "retired_v3_curve_indices.json",
)


app = modal.App("j-lens-rl-word-correlation-v1")
output_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)
gpu_lease_registry = modal.Dict.from_name(
    GPU_LEASE_DICT_NAME,
    environment_name=GPU_LEASE_ENVIRONMENT_NAME,
    create_if_missing=True,
)
publication_registry = modal.Dict.from_name(
    PUBLICATION_DICT_NAME,
    environment_name=GPU_LEASE_ENVIRONMENT_NAME,
    create_if_missing=True,
)
root_authority_registry = modal.Dict.from_name(
    ROOT_AUTHORITY_DICT_NAME,
    environment_name=GPU_LEASE_ENVIRONMENT_NAME,
    create_if_missing=True,
)

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
    temporary.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
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
        if child.is_file()
    }
    if not hashes:
        raise RuntimeError(f"empty output directory: {path}")
    return hashes


GENERATION_MARKER_PROTOCOL = "j-lens-rl-word-correlation-generation-v1"


def _artifact_key_parts(artifact_key: str) -> tuple[str, ...]:
    key = PurePosixPath(artifact_key)
    parts = key.parts
    if (
        not artifact_key
        or key.is_absolute()
        or not parts
        or parts == (".",)
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise RuntimeError(f"unsafe artifact key: {artifact_key!r}")
    return parts


def _marker_path(artifact_key: str) -> Path:
    parts = _artifact_key_parts(artifact_key)
    return (
        REMOTE_OUTPUT
        / COMMIT_MARKER_RELATIVE
        / Path(*parts[:-1])
        / f"{parts[-1]}.json"
    )


def _new_generation_dir(artifact_key: str) -> Path:
    parts = _artifact_key_parts(artifact_key)
    generation = (
        REMOTE_OUTPUT
        / GENERATION_RELATIVE
        / Path(*parts)
        / uuid.uuid4().hex
    )
    generation.mkdir(parents=True, exist_ok=False)
    return generation


def _resolve_generation(artifact_key: str, relative: Any) -> Path:
    if not isinstance(relative, str):
        raise RuntimeError(f"{artifact_key} marker has no generation path")
    candidate = PurePosixPath(relative)
    parts = candidate.parts
    expected = (GENERATION_RELATIVE, *_artifact_key_parts(artifact_key))
    if (
        candidate.is_absolute()
        or len(parts) != len(expected) + 1
        or tuple(parts[: len(expected)]) != expected
        or any(part in {"", ".", ".."} for part in parts)
        or len(parts[-1]) != 32
        or any(character not in "0123456789abcdef" for character in parts[-1])
    ):
        raise RuntimeError(f"unsafe {artifact_key} generation path: {relative!r}")
    root = (REMOTE_OUTPUT / GENERATION_RELATIVE).resolve()
    path = (REMOTE_OUTPUT / Path(*parts)).resolve()
    if root not in path.parents:
        raise RuntimeError(f"{artifact_key} generation escapes its root")
    return path


def _validate_generation_marker_payload(
    artifact_key: str, marker: Any
) -> tuple[Path, dict[str, Any]]:
    if (
        not isinstance(marker, dict)
        or marker.get("protocol") != GENERATION_MARKER_PROTOCOL
        or marker.get("artifact_key") != artifact_key
    ):
        raise RuntimeError(f"invalid {artifact_key} commit marker identity")
    generation = _resolve_generation(artifact_key, marker.get("generation"))
    expected_hashes = marker.get("generation_artifact_sha256")
    if (
        not isinstance(expected_hashes, dict)
        or not expected_hashes
        or any(
            not isinstance(relative, str)
            or PurePosixPath(relative).is_absolute()
            or any(part in {"", ".", ".."} for part in PurePosixPath(relative).parts)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for relative, digest in expected_hashes.items()
        )
    ):
        raise RuntimeError(f"invalid {artifact_key} generation hash inventory")
    if not generation.is_dir() or _directory_hashes(generation) != expected_hashes:
        raise RuntimeError(f"committed {artifact_key} generation is incomplete or changed")
    return generation, marker


def _load_generation_marker(
    artifact_key: str,
) -> tuple[Path, dict[str, Any]] | None:
    marker_path = _marker_path(artifact_key)
    if not marker_path.is_file():
        # Unmarked or partially committed generation directories are inert.
        return None
    return _validate_generation_marker_payload(artifact_key, _load_json(marker_path))


def _materialize_selected_marker(
    artifact_key: str,
) -> tuple[Path, dict[str, Any]] | None:
    """Recover the immutable CAS winner after a CAS-to-Volume crash cut."""

    selected = publication_registry.get(artifact_key)
    existing = _load_generation_marker(artifact_key)
    if existing is not None:
        if selected is None or existing[1] != selected:
            raise RuntimeError(f"{artifact_key} marker differs from its CAS selection")
        return existing
    if selected is None:
        return None
    output_volume.reload()
    selected_generation, selected_marker = _validate_generation_marker_payload(
        artifact_key, selected
    )
    existing = _load_generation_marker(artifact_key)
    if existing is not None:
        if existing[1] != selected_marker:
            raise RuntimeError(f"{artifact_key} marker differs from its CAS selection")
        return existing
    _write_json(_marker_path(artifact_key), selected_marker)
    output_volume.commit()
    output_volume.reload()
    committed = _load_generation_marker(artifact_key)
    if (
        committed is None
        or committed[0] != selected_generation
        or committed[1] != selected_marker
    ):
        raise RuntimeError(f"{artifact_key} marker changed after CAS recovery")
    return committed


def _publish_generation(
    artifact_key: str,
    generation: Path,
    metadata: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Publish one immutable generation behind one atomic validity marker.

    The first commit makes the immutable files durable.  Only after a reload
    barrier and a full hash revalidation is the single marker written and
    committed.  A restart before that marker simply leaves an inert orphan;
    the retry creates a new generation.  A concurrently published valid marker
    wins and is never overwritten.
    """

    # The caller has already staged this generation on its Volume mount, so do
    # not reload here: a concurrent winner is recovered only after this inert
    # generation has first been committed.
    existing = _load_generation_marker(artifact_key)
    if existing is not None:
        selected = publication_registry.get(artifact_key)
        if selected is None or existing[1] != selected:
            raise RuntimeError(f"{artifact_key} marker differs from its CAS selection")
        return existing
    generation_relative = generation.relative_to(REMOTE_OUTPUT).as_posix()
    if _resolve_generation(artifact_key, generation_relative) != generation.resolve():
        raise RuntimeError(f"generation path does not match {artifact_key}")
    protected = {
        "protocol",
        "artifact_key",
        "generation",
        "generation_artifact_sha256",
        "published_at_utc",
    }
    if protected & set(metadata):
        raise RuntimeError("generation metadata overrides marker identity")
    generation_hashes = _directory_hashes(generation)
    output_volume.commit()
    output_volume.reload()
    existing = _materialize_selected_marker(artifact_key)
    if existing is not None:
        return existing
    if _directory_hashes(generation) != generation_hashes:
        raise RuntimeError(f"{artifact_key} generation changed before publication")
    marker = {
        "protocol": GENERATION_MARKER_PROTOCOL,
        "artifact_key": artifact_key,
        "generation": generation_relative,
        "generation_artifact_sha256": generation_hashes,
        "published_at_utc": datetime.now(timezone.utc).isoformat(),
        **metadata,
    }
    try:
        json.dumps(marker, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{artifact_key} marker is not durable JSON") from error
    # A named-Dict put-if-absent is the cross-container/deployment CAS.  If a
    # preempted publisher already selected another fully committed generation,
    # every retry writes those exact same marker bytes; no valid marker can be
    # replaced by a later generation.
    publication_registry.put(artifact_key, marker, skip_if_exists=True)
    committed = _materialize_selected_marker(artifact_key)
    if committed is None:
        raise RuntimeError(f"{artifact_key} marker commit did not become durable")
    return committed


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
        "gpu_lease_dict_name": GPU_LEASE_DICT_NAME,
        "gpu_lease_environment_name": GPU_LEASE_ENVIRONMENT_NAME,
        "gpu_lease_protocol": GPU_LEASE_PROTOCOL,
        "gpu_lease_slot": GPU_LEASE_SLOT,
        "publication_dict_name": PUBLICATION_DICT_NAME,
        "root_authority_dict_name": ROOT_AUTHORITY_DICT_NAME,
        "root_authority_protocol": ROOT_AUTHORITY_PROTOCOL,
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


def _validate_claim_id(claim_id: str) -> str:
    if (
        len(claim_id) != 32
        or claim_id != claim_id.casefold()
        or any(character not in "0123456789abcdef" for character in claim_id)
    ):
        raise RuntimeError("claim ID must be exactly 32 lowercase hexadecimal characters")
    return claim_id


def _default_claim_id() -> str:
    identity = {
        "protocol": "j-lens-rl-jspace-word-correlation-v1",
        "volume": VOLUME_NAME,
        "git_commit": _git("rev-parse", "HEAD", repo=LOCAL_REPO),
        "amendment_sha256": _sha256(LOCAL_REPO / CURRENT_AMENDMENT_RELATIVE),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:32]


def _launch_manifest(preflight: dict[str, Any], claim_id: str) -> dict[str, Any]:
    claim_id = _validate_claim_id(claim_id)
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
        "claim_id": claim_id,
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
        "gpu_lease_dict_name": GPU_LEASE_DICT_NAME,
        "gpu_lease_environment_name": GPU_LEASE_ENVIRONMENT_NAME,
        "gpu_lease_protocol": GPU_LEASE_PROTOCOL,
        "gpu_lease_slot": GPU_LEASE_SLOT,
        "publication_dict_name": PUBLICATION_DICT_NAME,
        "root_authority_dict_name": ROOT_AUTHORITY_DICT_NAME,
        "root_authority_protocol": ROOT_AUTHORITY_PROTOCOL,
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


def _validate_persisted_local_manifest(
    manifest: dict[str, Any], claim_id: str
) -> dict[str, Any]:
    if manifest.get("claim_id") != _validate_claim_id(claim_id):
        raise RuntimeError("persisted claim identity differs from requested claim")
    preflight = manifest.get("gpu_exclusive_preflight")
    if not isinstance(preflight, dict):
        raise RuntimeError("persisted claim has no original GPU preflight")
    expected = _launch_manifest(preflight, claim_id)
    expected["created_at_utc"] = manifest.get("created_at_utc")
    if expected != manifest:
        raise RuntimeError("persisted claim differs from this committed launcher")
    return manifest


def _validate_gpu_preflight(value: Any, context: str) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or value.get("exclusive_gpu_confirmation") != GPU_EXCLUSIVE_CONFIRMATION
        or value.get("global_modal_gpu_limit") != GLOBAL_MODAL_GPU_LIMIT
        or value.get("active_other_modal_apps") != []
        or not isinstance(value.get("checked_at_utc"), str)
    ):
        raise RuntimeError(f"{context} lacks a valid exclusive-GPU preflight")
    return value


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
    _validate_gpu_preflight(
        manifest.get("gpu_exclusive_preflight"),
        "remote launch",
    )
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
        "gpu_lease_dict_name": GPU_LEASE_DICT_NAME,
        "gpu_lease_environment_name": GPU_LEASE_ENVIRONMENT_NAME,
        "gpu_lease_protocol": GPU_LEASE_PROTOCOL,
        "gpu_lease_slot": GPU_LEASE_SLOT,
        "publication_dict_name": PUBLICATION_DICT_NAME,
        "root_authority_dict_name": ROOT_AUTHORITY_DICT_NAME,
        "root_authority_protocol": ROOT_AUTHORITY_PROTOCOL,
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


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_claim_manifest(expected_claim_id: str | None = None) -> dict[str, Any] | None:
    committed = _load_generation_marker("claim")
    if committed is None:
        return None
    generation, marker = committed
    manifest_path = generation / "attempt_manifest.json"
    manifest = _load_json(manifest_path)
    claim_id = manifest.get("claim_id")
    if not isinstance(claim_id, str):
        raise RuntimeError("committed claim has no claim ID")
    _validate_claim_id(claim_id)
    preflight = manifest.get("gpu_exclusive_preflight")
    expected = {
        "claim_id": claim_id,
        "manifest_sha256": _sha256(manifest_path),
        "original_preflight_sha256": _json_sha256(preflight),
    }
    if any(marker.get(key) != value for key, value in expected.items()):
        raise RuntimeError("committed claim marker is invalid")
    if set(marker["generation_artifact_sha256"]) != {"attempt_manifest.json"}:
        raise RuntimeError("committed claim generation has unexpected files")
    if expected_claim_id is not None and claim_id != expected_claim_id:
        raise RuntimeError("word-correlation Volume claim mismatch")
    return manifest


def _set_status(claim_id: str, stage: str, **details: Any) -> None:
    claim = _load_claim_manifest(claim_id)
    if claim is None:
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
    calibration = _committed_calibration_path(manifest)
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
    committed = _load_generation_marker("calibration")
    if committed is None:
        return None
    generation, marker = committed
    output = generation / "calibration.json"
    payload = _load_json(output)
    _validate_calibration_payload(payload, manifest)
    expected = {
        "output": output.relative_to(REMOTE_OUTPUT).as_posix(),
        "output_sha256": _sha256(output),
    }
    if (
        any(marker.get(key) != value for key, value in expected.items())
        or set(marker["generation_artifact_sha256"]) != {"calibration.json"}
    ):
        raise RuntimeError("committed calibration marker is invalid")
    return expected


def _committed_calibration_path(manifest: dict[str, Any]) -> Path:
    result = _load_committed_calibration(manifest)
    if result is None:
        raise RuntimeError("scanner output has no frozen calibration artifact")
    path = REMOTE_OUTPUT / str(result["output"])
    if not path.is_file() or _sha256(path) != result["output_sha256"]:
        raise RuntimeError("committed calibration path is incomplete or changed")
    return path


def _selection_identity(
    phase: str, manifest: dict[str, Any]
) -> tuple[Path | None, str | None]:
    if phase == "discovery":
        return None, None
    committed = _load_generation_marker("discovery/final")
    if committed is None:
        raise RuntimeError("validation cannot start without selection lock")
    generation, marker = committed
    selection_path = generation / "selection_lock.json"
    selection_sha256 = _sha256(selection_path)
    lock = _load_json(selection_path)
    expected = {
        "protocol": manifest["protocol"],
        "claim_id": manifest["claim_id"],
        "git_commit": manifest["git_commit"],
        "config_sha256": manifest["config_sha256"],
        "scanner_sha256": manifest["scanner_sha256"],
        "curve_manifest_sha256": CURVE_MANIFEST_SHA256,
        "lens_sha256": LENS_SHA256,
        "calibration_sha256": _sha256(_committed_calibration_path(manifest)),
    }
    if (
        marker.get("selection_lock_sha256") != selection_sha256
        or any(lock.get(key) != value for key, value in expected.items())
    ):
        raise RuntimeError("committed selection lock identity is invalid")
    return selection_path, selection_sha256


def _shard_key(phase: str, shard_index: int) -> str:
    return f"{phase}/shard-{shard_index:02d}"


def _load_committed_shard(
    phase: str,
    shard_index: int,
    manifest: dict[str, Any],
) -> dict[str, Any] | None:
    committed = _load_generation_marker(_shard_key(phase, shard_index))
    if committed is None:
        return None
    generation, marker = committed
    shard_dir = generation / "shard"
    selection_path, lock_sha256 = _selection_identity(phase, manifest)
    del selection_path
    payload = _load_json(shard_dir / "summary.json")
    payload = _validate_shard(
        payload,
        phase=phase,
        shard_index=shard_index,
        manifest=manifest,
        selection_lock_sha256=lock_sha256,
    )
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
    if any(marker.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"committed {phase} shard {shard_index} marker is invalid")
    if set(marker["generation_artifact_sha256"]) != {
        f"shard/{relative}" for relative in artifact_sha256
    }:
        raise RuntimeError(f"committed {phase} shard generation has unexpected files")
    return expected


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
        generation = _new_generation_dir("calibration")
        output = generation / "calibration.json"
        shutil.copyfile(local_output, output)
        result = {
            "output": output.relative_to(REMOTE_OUTPUT).as_posix(),
            "output_sha256": _sha256(output),
        }
        _publish_generation("calibration", generation, result)
        committed = _load_committed_calibration(manifest)
        if committed is None:
            raise RuntimeError("calibration marker vanished after publication")
        return committed


def _scan_phase(phase: str, shard_index: int, manifest: dict[str, Any]) -> dict[str, Any]:
    if phase not in {"discovery", "validation"}:
        raise ValueError(f"invalid correlation phase: {phase}")
    if shard_index not in range(NUM_SHARDS):
        raise ValueError(f"invalid shard index: {shard_index}")
    existing = _load_committed_shard(phase, shard_index, manifest)
    if existing is not None:
        return existing
    selection_path, lock_sha256 = _selection_identity(phase, manifest)
    calibration_path = _committed_calibration_path(manifest)
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
                calibration_path=calibration_path,
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
        artifact_key = _shard_key(phase, shard_index)
        generation = _new_generation_dir(artifact_key)
        shard_dir = generation / "shard"
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
        _publish_generation(artifact_key, generation, result)
        committed = _load_committed_shard(phase, shard_index, manifest)
        if committed is None:
            raise RuntimeError(f"{phase} shard marker vanished after publication")
        return committed


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
    if not REMOTE_OUTPUT.exists():
        REMOTE_OUTPUT.mkdir(parents=True)
    _verify_remote_manifest(manifest)
    claim_id = str(manifest["claim_id"])
    _validate_claim_id(claim_id)
    existing = _load_claim_manifest()
    if existing is None:
        allowed_orphans = {GENERATION_RELATIVE, COMMIT_MARKER_RELATIVE}
        unexpected = sorted(
            path.name
            for path in REMOTE_OUTPUT.iterdir()
            if path.name not in allowed_orphans
        )
        if unexpected:
            raise RuntimeError(
                "unclaimed word-correlation Volume has unexpected durable state: "
                f"{unexpected}"
            )
        generation = _new_generation_dir("claim")
        manifest_path = generation / "attempt_manifest.json"
        _write_json(manifest_path, manifest)
        _publish_generation(
            "claim",
            generation,
            {
                "claim_id": claim_id,
                "manifest_sha256": _sha256(manifest_path),
                "original_preflight_sha256": _json_sha256(
                    manifest["gpu_exclusive_preflight"]
                ),
            },
        )
        existing = _load_claim_manifest(claim_id)
    if existing != manifest:
        raise RuntimeError("Volume is already claimed by a different launch manifest")
    status_path = REMOTE_OUTPUT / "attempt_status.json"
    if not status_path.is_file():
        _set_status(claim_id, "claimed")
        output_volume.commit()
    else:
        status = _load_json(status_path)
        if status.get("claim_id") != claim_id or not isinstance(
            status.get("stage"), str
        ):
            raise RuntimeError("persisted claim status is invalid")
    return existing


@app.function(
    image=repo_image,
    cpu=1,
    memory=2048,
    max_containers=1,
    timeout=5 * 60,
    startup_timeout=60 * 60,
    retries=0,
    volumes={REMOTE_OUTPUT: output_volume},
)
def inspect_claim(claim_id: str) -> dict[str, Any] | None:
    output_volume.reload()
    _validate_claim_id(claim_id)
    _materialize_selected_marker("claim")
    return _load_claim_manifest(claim_id)


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
def submit_attempt(
    claim_id: str, submission_preflight: dict[str, Any]
) -> dict[str, Any]:
    """Acquire the global lease and idempotently submit one root call."""

    output_volume.reload()
    _validate_claim_id(claim_id)
    _validate_gpu_preflight(submission_preflight, "submission")
    manifest = _load_claim_manifest(claim_id)
    if manifest is None:
        raise RuntimeError("cannot submit an unclaimed word-correlation attempt")
    _verify_remote_manifest(manifest)
    status = _load_json(REMOTE_OUTPUT / "attempt_status.json")
    if status.get("claim_id") != claim_id:
        raise RuntimeError("submission status has the wrong claim ID")

    prior_state = _load_submission_state(claim_id)
    if status.get("stage") == "complete":
        authoritative = _persist_root_authority_to_volume(claim_id)
        state = _load_submission_state(claim_id)
        if authoritative is None or state is None:
            raise RuntimeError("complete attempt lacks its durable submission call ID")
        return {
            "claim_id": claim_id,
            "function_call_id": authoritative,
            "intent_id": state["intent_id"],
            "lease_owner": state["lease_owner"],
        }

    lease = _acquire_gpu_lease(claim_id, submission_preflight)
    lease_receipt = {
        **lease,
        "dict_name": GPU_LEASE_DICT_NAME,
        "environment_name": GPU_LEASE_ENVIRONMENT_NAME,
    }

    state = prior_state
    if state is None:
        state = {
            "protocol": "j-lens-rl-word-correlation-submission-v1",
            "claim_id": claim_id,
            "intent_id": _submission_intent_id(claim_id),
            "lease_slot": GPU_LEASE_SLOT,
            "lease_owner": lease["owner"],
            "gpu_lease_receipt": lease_receipt,
            "orchestrator_call_id": None,
            "controller_bound_at_utc": None,
            "spawned_call_ids": [],
            "submission_preflight_checks": [submission_preflight],
            "intent_committed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    else:
        if state.get("gpu_lease_receipt") != lease_receipt:
            raise RuntimeError("immutable GPU lease receipt changed across retry")
        authoritative = _persist_root_authority_to_volume(claim_id)
        if authoritative is not None:
            state = _load_submission_state(claim_id)
            if state is None:
                raise RuntimeError("submission state vanished during authority recovery")
            modal.FunctionCall.from_id(authoritative)
            return {
                "claim_id": claim_id,
                "function_call_id": authoritative,
                "intent_id": state["intent_id"],
                "lease_owner": state["lease_owner"],
            }
        if submission_preflight not in state["submission_preflight_checks"]:
            state["submission_preflight_checks"].append(submission_preflight)
    _store_submission_state(state)
    output_volume.commit()

    output_volume.reload()
    state = _load_submission_state(claim_id)
    if state is None:
        raise RuntimeError("submission intent vanished after commit")
    authoritative = _persist_root_authority_to_volume(claim_id)
    if authoritative is not None:
        state = _load_submission_state(claim_id)
        if state is None:
            raise RuntimeError("submission state vanished during authority recovery")
        modal.FunctionCall.from_id(authoritative)
        return {
            "claim_id": claim_id,
            "function_call_id": authoritative,
            "intent_id": state["intent_id"],
            "lease_owner": state["lease_owner"],
        }

    # A null intent seen on entry can mean that Modal accepted a prior spawn
    # immediately before preemption.  Give that root time to win the immutable
    # root-authority CAS before safely enqueueing a serialized duplicate.
    if prior_state is not None:
        deadline = time.monotonic() + SUBMISSION_RECOVERY_SECONDS
        while time.monotonic() < deadline:
            time.sleep(0.25)
            authoritative = _persist_root_authority_to_volume(claim_id)
            if authoritative is not None:
                state = _load_submission_state(claim_id)
                if state is None:
                    raise RuntimeError("submission state vanished during recovery")
                modal.FunctionCall.from_id(authoritative)
                return {
                    "claim_id": claim_id,
                    "function_call_id": authoritative,
                    "intent_id": state["intent_id"],
                    "lease_owner": state["lease_owner"],
                }

    call = orchestrate.spawn(claim_id)
    spawned_call_id = str(call.object_id)
    output_volume.reload()
    state = _load_submission_state(claim_id)
    if state is None:
        raise RuntimeError("submission state vanished after orchestrator spawn")
    if spawned_call_id not in state["spawned_call_ids"]:
        state["spawned_call_ids"].append(spawned_call_id)
    state["last_spawn_returned_at_utc"] = datetime.now(timezone.utc).isoformat()
    _store_submission_state(state)
    output_volume.commit()

    deadline = time.monotonic() + SUBMISSION_BIND_SECONDS
    while time.monotonic() < deadline:
        authoritative = _persist_root_authority_to_volume(claim_id)
        if authoritative is not None:
            state = _load_submission_state(claim_id)
            if state is None:
                raise RuntimeError("submission state vanished after root binding")
            modal.FunctionCall.from_id(authoritative)
            return {
                "claim_id": claim_id,
                "function_call_id": authoritative,
                "intent_id": state["intent_id"],
                "lease_owner": state["lease_owner"],
            }
        time.sleep(0.25)
    raise RuntimeError(
        "orchestrator spawn was accepted but no root won durable authority; "
        "retry the same claim to reattach"
    )


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
def gpu_job(
    kind: str,
    shard_index: int | None,
    root_call_id: str,
) -> dict[str, Any]:
    output_volume.reload()
    manifest = _load_claim_manifest()
    if manifest is None:
        raise RuntimeError("GPU job cannot run before a committed claim")
    _assert_gpu_lease(str(manifest["claim_id"]), root_call_id)
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
        selection, selection_sha256 = _selection_identity("validation", manifest)
        if (
            selection is None
            or payload.get("selection_lock_sha256") != selection_sha256
        ):
            raise RuntimeError("validation aggregate used a different selection lock")


def _load_committed_aggregate(
    phase: str,
    shard_results: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> tuple[Path, dict[str, Any]] | None:
    artifact_key = f"{phase}/final"
    committed = _load_generation_marker(artifact_key)
    if committed is None:
        return None
    generation, marker = committed
    output = generation / "aggregate.json"
    merge_dir = generation / "merged"
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
    if any(marker.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"committed {phase} aggregate marker is invalid")
    expected_files = {"aggregate.json", *(f"merged/{name}" for name in merge_hashes)}
    if phase == "discovery":
        selection = generation / "selection_lock.json"
        if marker.get("selection_lock_sha256") != _sha256(selection):
            raise RuntimeError("discovery marker has the wrong selection-lock hash")
        expected_files.add("selection_lock.json")
    if set(marker["generation_artifact_sha256"]) != expected_files:
        raise RuntimeError(f"committed {phase} generation has unexpected files")
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

        selection_path, _selection_sha256 = _selection_identity(phase, manifest)
        merged = merge_validation(
            config_path=REMOTE_REPO / CONFIG_RELATIVE,
            shard_dirs=shard_dirs,
            calibration_path=_committed_calibration_path(manifest),
            selection_path=selection_path,
            output_dir=merge_dir,
        )
    else:
        from jlens_rl.word_correlation import merge_discovery

        merged = merge_discovery(
            config_path=REMOTE_REPO / CONFIG_RELATIVE,
            shard_dirs=shard_dirs,
            calibration_path=_committed_calibration_path(manifest),
            output_dir=merge_dir,
        )
    payload = _json_result(merged, f"merge_{phase}")
    _validate_aggregate_payload(phase, payload, shard_results, manifest)
    return payload


def _stage_aggregate_generation(
    phase: str, merge_dir: Path, payload: dict[str, Any]
) -> tuple[Path, Path, dict[str, Any]]:
    generation = _new_generation_dir(f"{phase}/final")
    remote_merge = generation / "merged"
    shutil.copytree(merge_dir, remote_merge)
    output = generation / "aggregate.json"
    _write_json(output, payload)
    metadata = {
        "phase": phase,
        "aggregate_sha256": _sha256(output),
        "merge_artifact_sha256": _directory_hashes(remote_merge),
    }
    return generation, output, metadata


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
    path: Path,
) -> tuple[Path, dict[str, Any]]:
    selection = _validated_selection(discovery, manifest)
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
        "calibration_sha256": _sha256(_committed_calibration_path(manifest)),
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
    committed = _load_generation_marker("discovery/final")
    if committed is None:
        return None
    generation, marker = committed
    if discovery_path.parent != generation:
        raise RuntimeError("selection lock and discovery aggregate use different generations")
    path = generation / "selection_lock.json"
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
        "calibration_sha256": _sha256(_committed_calibration_path(manifest)),
        "discovery_aggregate_sha256": _sha256(discovery_path),
        "selection": selection,
    }
    if (
        marker.get("selection_lock_sha256") != _sha256(path)
        or not isinstance(lock.get("locked_at_utc"), str)
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
    if aggregate is None:
        with tempfile.TemporaryDirectory(
            prefix="word-correlation-discovery-merge-"
        ) as raw:
            local_merge = Path(raw) / "merged"
            local_merge.mkdir()
            payload = _merge_to_local(
                "discovery", discovery_results, manifest, local_merge
            )
            generation, discovery_path, metadata = _stage_aggregate_generation(
                "discovery", local_merge, payload
            )
            selection_path, selection_lock = _lock_selection(
                discovery_path,
                payload,
                manifest,
                generation / "selection_lock.json",
            )
            _publish_generation(
                "discovery/final",
                generation,
                {
                    **metadata,
                    "selection_lock_sha256": _sha256(selection_path),
                },
            )
            aggregate = _load_committed_aggregate(
                "discovery", discovery_results, manifest
            )
            if aggregate is None:
                raise RuntimeError("discovery final marker vanished after publication")
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
        generation, _output, metadata = _stage_aggregate_generation(
            "validation", local_merge, payload
        )
        _publish_generation("validation/final", generation, metadata)
        committed = _load_committed_aggregate(
            "validation", validation_results, manifest
        )
        if committed is None:
            raise RuntimeError("validation final marker vanished after publication")
        return committed


def _load_committed_atlas(
    discovery_results: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> tuple[Path, dict[str, Any]] | None:
    committed = _load_generation_marker("atlas")
    if committed is None:
        return None
    generation, marker = committed
    output_dir = generation / "atlas"
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
    if any(marker.get(key) != value for key, value in expected.items()):
        raise RuntimeError("committed lexical atlas marker is invalid")
    if set(marker["generation_artifact_sha256"]) != {
        f"atlas/{name}" for name in expected["atlas_artifact_sha256"]
    }:
        raise RuntimeError("committed lexical atlas generation has unexpected files")
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
                calibration_path=_committed_calibration_path(manifest),
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
        generation = _new_generation_dir("atlas")
        output_dir = generation / "atlas"
        shutil.copytree(local_dir, output_dir)
        output = output_dir / "atlas.json"
        _publish_generation(
            "atlas",
            generation,
            {
                "atlas_sha256": _sha256(output),
                "atlas_artifact_sha256": _directory_hashes(output_dir),
            },
        )
        committed = _load_committed_atlas(discovery_results, manifest)
        if committed is None:
            raise RuntimeError("lexical atlas marker vanished after publication")
        return committed


def _load_root_authority(claim_id: str) -> dict[str, Any] | None:
    _validate_claim_id(claim_id)
    value = root_authority_registry.get(claim_id)
    if value is None:
        return None
    if (
        not isinstance(value, dict)
        or value.get("protocol") != ROOT_AUTHORITY_PROTOCOL
        or value.get("volume") != VOLUME_NAME
        or value.get("claim_id") != claim_id
        or value.get("intent_id") != _submission_intent_id(claim_id)
        or not isinstance(value.get("orchestrator_call_id"), str)
        or not value.get("orchestrator_call_id")
        or not isinstance(value.get("bound_at_utc"), str)
    ):
        raise RuntimeError("durable root-authority record is invalid")
    return value


def _claim_root_authority(claim_id: str, call_id: str) -> dict[str, Any]:
    if not isinstance(call_id, str) or not call_id:
        raise RuntimeError("root authority requires a Modal call ID")
    candidate = {
        "protocol": ROOT_AUTHORITY_PROTOCOL,
        "volume": VOLUME_NAME,
        "claim_id": _validate_claim_id(claim_id),
        "intent_id": _submission_intent_id(claim_id),
        "orchestrator_call_id": call_id,
        "bound_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if root_authority_registry.put(claim_id, candidate, skip_if_exists=True):
        return candidate
    existing = _load_root_authority(claim_id)
    if existing is None:
        raise RuntimeError("root-authority CAS vanished")
    if existing["orchestrator_call_id"] != call_id:
        raise RuntimeError("a different orchestrator call owns root authority")
    return existing


def _gpu_lease_owner(claim_id: str) -> str:
    return f"{GPU_LEASE_KEY_PREFIX}:{_validate_claim_id(claim_id)}"


def _validate_gpu_lease_record(value: Any) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or value.get("protocol") != GPU_LEASE_PROTOCOL
        or value.get("environment_name") != GPU_LEASE_ENVIRONMENT_NAME
        or value.get("slot") != GPU_LEASE_SLOT
        or value.get("global_modal_gpu_limit") != GLOBAL_MODAL_GPU_LIMIT
        or not isinstance(value.get("owner"), str)
        or not isinstance(value.get("workload"), str)
        or not value.get("workload")
        or not isinstance(value.get("claim_id"), str)
        or not value.get("claim_id")
        or not isinstance(value.get("acquired_at_utc"), str)
        or not isinstance(value.get("heartbeat_at_utc"), str)
        or not isinstance(value.get("submission_preflight_sha256"), str)
    ):
        raise RuntimeError("global Modal GPU lease record is invalid")
    _validate_gpu_preflight(value.get("submission_preflight"), "GPU lease")
    _validate_claim_id(value["claim_id"])
    if value["submission_preflight_sha256"] != _json_sha256(
        value["submission_preflight"]
    ):
        raise RuntimeError("global Modal GPU lease preflight hash is invalid")
    if value["owner"] != f"{value['workload']}:{value['claim_id']}":
        raise RuntimeError("global Modal GPU lease owner is invalid")
    return value


def _acquire_gpu_lease(
    claim_id: str, submission_preflight: dict[str, Any]
) -> dict[str, Any]:
    """Atomically claim the one account-wide cooperative GPU slot.

    ``Dict.put(..., skip_if_exists=True)`` is the cross-app compare-and-set.
    The record has no automatic expiry: a crashed owner fails closed until the
    same claim resumes or an operator explicitly audits and releases it.
    """

    _validate_gpu_preflight(submission_preflight, "submission")
    owner = _gpu_lease_owner(claim_id)
    now = datetime.now(timezone.utc).isoformat()
    candidate = {
        "protocol": GPU_LEASE_PROTOCOL,
        "environment_name": GPU_LEASE_ENVIRONMENT_NAME,
        "slot": GPU_LEASE_SLOT,
        "owner": owner,
        "workload": GPU_LEASE_KEY_PREFIX,
        "claim_id": claim_id,
        "global_modal_gpu_limit": GLOBAL_MODAL_GPU_LIMIT,
        "submission_preflight": submission_preflight,
        "submission_preflight_sha256": _json_sha256(submission_preflight),
        "acquired_at_utc": now,
        "heartbeat_at_utc": now,
    }
    if gpu_lease_registry.put(GPU_LEASE_SLOT, candidate, skip_if_exists=True):
        return candidate
    existing = _validate_gpu_lease_record(
        gpu_lease_registry.get(GPU_LEASE_SLOT)
    )
    if existing["owner"] != owner or existing["claim_id"] != claim_id:
        raise RuntimeError(
            "global Modal GPU lease is held by another app/claim: "
            f"{existing['owner']}"
        )
    # The lease is immutable.  Same-claim launch retries may continue toward
    # root reattachment, but only the CAS-elected root can authorize GPU work.
    return existing


def _assert_gpu_lease(
    claim_id: str, root_call_id: str | None = None
) -> dict[str, Any]:
    record = _validate_gpu_lease_record(gpu_lease_registry.get(GPU_LEASE_SLOT))
    if record["owner"] != _gpu_lease_owner(claim_id):
        raise RuntimeError("this claim does not own the global Modal GPU lease")
    if root_call_id is not None:
        authority = _load_root_authority(claim_id)
        if (
            authority is None
            or authority.get("orchestrator_call_id") != root_call_id
        ):
            raise RuntimeError("this call does not own durable root authority")
    return record


def _refresh_gpu_lease(claim_id: str, root_call_id: str) -> dict[str, Any]:
    # Deliberately read-only: a stale owner must never overwrite a successor.
    return _assert_gpu_lease(claim_id, root_call_id)


def _release_gpu_lease(claim_id: str, root_call_id: str) -> None:
    value = gpu_lease_registry.get(GPU_LEASE_SLOT)
    if value is None:
        return
    record = _validate_gpu_lease_record(value)
    if record["owner"] != _gpu_lease_owner(claim_id):
        # This root already released its immutable lease and a successor owns
        # the slot.  Never let a stale completion pop that successor.
        return
    _assert_gpu_lease(claim_id, root_call_id)
    removed = gpu_lease_registry.pop(GPU_LEASE_SLOT)
    if removed != record:
        raise RuntimeError("global Modal GPU lease changed during release")


def _submission_state_path() -> Path:
    return REMOTE_OUTPUT / "submission_state.json"


def _submission_intent_id(claim_id: str) -> str:
    return _json_sha256(
        {
            "protocol": "j-lens-rl-word-correlation-submission-v1",
            "volume": VOLUME_NAME,
            "claim_id": claim_id,
        }
    )


def _load_submission_state(claim_id: str) -> dict[str, Any] | None:
    path = _submission_state_path()
    if not path.is_file():
        return None
    state = _load_json(path)
    checks = state.get("submission_preflight_checks")
    spawned = state.get("spawned_call_ids")
    receipt = state.get("gpu_lease_receipt")
    if (
        state.get("protocol") != "j-lens-rl-word-correlation-submission-v1"
        or state.get("claim_id") != claim_id
        or state.get("intent_id") != _submission_intent_id(claim_id)
        or state.get("lease_slot") != GPU_LEASE_SLOT
        or state.get("lease_owner") != _gpu_lease_owner(claim_id)
        or not isinstance(receipt, dict)
        or receipt.get("dict_name") != GPU_LEASE_DICT_NAME
        or receipt.get("environment_name") != GPU_LEASE_ENVIRONMENT_NAME
        or (
            state.get("orchestrator_call_id") is not None
            and not isinstance(state.get("orchestrator_call_id"), str)
        )
        or (
            state.get("controller_bound_at_utc") is not None
            and not isinstance(state.get("controller_bound_at_utc"), str)
        )
        or not isinstance(checks, list)
        or not checks
        or not isinstance(spawned, list)
        or any(not isinstance(call_id, str) or not call_id for call_id in spawned)
    ):
        raise RuntimeError("durable orchestrator submission state is invalid")
    _validate_gpu_lease_record(receipt)
    if receipt.get("owner") != state["lease_owner"]:
        raise RuntimeError("submission state has the wrong GPU lease receipt")
    for check in checks:
        _validate_gpu_preflight(check, "persisted submission")
    return state


def _store_submission_state(state: dict[str, Any]) -> None:
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(_submission_state_path(), state)


def _persist_root_authority_to_volume(claim_id: str) -> str | None:
    authority = _load_root_authority(claim_id)
    if authority is None:
        return None
    state = _load_submission_state(claim_id)
    if state is None:
        raise RuntimeError("root authority exists without submission intent")
    authoritative = str(authority["orchestrator_call_id"])
    if state.get("orchestrator_call_id") not in {None, authoritative}:
        raise RuntimeError("Volume submission state conflicts with root authority")
    if (
        state.get("orchestrator_call_id") == authoritative
        and state.get("controller_bound_at_utc") == authority["bound_at_utc"]
    ):
        return authoritative
    state["orchestrator_call_id"] = authoritative
    state["controller_bound_at_utc"] = authority["bound_at_utc"]
    _store_submission_state(state)
    output_volume.commit()
    return authoritative


def _bind_submission_to_controller(claim_id: str, call_id: str) -> None:
    authority = _claim_root_authority(claim_id, call_id)
    state = _load_submission_state(claim_id)
    if state is None:
        raise RuntimeError("orchestrator started without a durable submission intent")
    # The first serialized root invocation is authoritative.  This repairs the
    # only unavoidable API cut: spawn accepted but its returned call ID was not
    # yet committed by the submitter.
    if state.get("controller_bound_at_utc") is None:
        state["orchestrator_call_id"] = authority["orchestrator_call_id"]
        state["controller_bound_at_utc"] = authority["bound_at_utc"]
        _store_submission_state(state)
        output_volume.commit()
    elif state.get("orchestrator_call_id") != call_id:
        raise RuntimeError("a different orchestrator call already owns this submission")


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
    _refresh_gpu_lease(claim_id, call_id)
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
        call = gpu_job.spawn(kind, shard_index, call_id)
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
    committed = _load_committed_result(claim_id, manifest)
    status = _load_json(REMOTE_OUTPUT / "attempt_status.json")
    if committed is not None:
        result_path, result = committed
        result_sha256 = _sha256(result_path)
        if status.get("stage") != "complete":
            _set_status(
                claim_id,
                "complete",
                result_manifest_sha256=result_sha256,
                selection=result.get("selection"),
            )
            output_volume.commit()
        elif status.get("result_manifest_sha256") != result_sha256:
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
    generation = _new_generation_dir("result")
    controller_snapshot = generation / "controller_state.json"
    submission_snapshot = generation / "submission_state.json"
    shutil.copyfile(_controller_state_path(), controller_snapshot)
    shutil.copyfile(_submission_state_path(), submission_snapshot)
    controller_state = _load_json(controller_snapshot)
    submission_state = _load_json(submission_snapshot)
    if (
        controller_state.get("claim_id") != claim_id
        or controller_state.get("protocol")
        != "j-lens-rl-word-correlation-controller-v1"
        or submission_state.get("claim_id") != claim_id
        or submission_state.get("protocol")
        != "j-lens-rl-word-correlation-submission-v1"
    ):
        raise RuntimeError("final orchestration snapshots have the wrong identity")
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
        "controller_state_snapshot": controller_snapshot.relative_to(
            REMOTE_OUTPUT
        ).as_posix(),
        "controller_state_sha256": _sha256(controller_snapshot),
        "submission_state_snapshot": submission_snapshot.relative_to(
            REMOTE_OUTPUT
        ).as_posix(),
        "submission_state_sha256": _sha256(submission_snapshot),
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
    result_path = generation / "result_manifest.json"
    _write_json(result_path, result_manifest)
    result_sha256 = _sha256(result_path)
    _publish_generation(
        "result",
        generation,
        {
            "claim_id": claim_id,
            "result_manifest_sha256": result_sha256,
        },
    )
    committed = _load_committed_result(claim_id, manifest)
    if committed is None:
        raise RuntimeError("result marker vanished after publication")
    result_path, result_manifest = committed
    result_sha256 = _sha256(result_path)
    _set_status(
        claim_id,
        "complete",
        result_manifest_sha256=result_sha256,
        selection=result_manifest["selection"],
    )
    output_volume.commit()
    return {
        "stage": "complete",
        "result_manifest_sha256": result_sha256,
        "selection": result_manifest["selection"],
    }


def _load_committed_result(
    claim_id: str, manifest: dict[str, Any]
) -> tuple[Path, dict[str, Any]] | None:
    committed = _load_generation_marker("result")
    if committed is None:
        return None
    generation, marker = committed
    path = generation / "result_manifest.json"
    controller_snapshot = generation / "controller_state.json"
    submission_snapshot = generation / "submission_state.json"
    payload = _load_json(path)
    expected = {
        "protocol": manifest["protocol"],
        "claim_id": claim_id,
        "git_commit": manifest["git_commit"],
        "config_sha256": manifest["config_sha256"],
        "scanner_sha256": manifest["scanner_sha256"],
        "current_amendment_sha256": manifest["current_amendment_sha256"],
    }
    snapshot_expected = {
        "controller_state_snapshot": controller_snapshot.relative_to(
            REMOTE_OUTPUT
        ).as_posix(),
        "controller_state_sha256": _sha256(controller_snapshot),
        "submission_state_snapshot": submission_snapshot.relative_to(
            REMOTE_OUTPUT
        ).as_posix(),
        "submission_state_sha256": _sha256(submission_snapshot),
    }
    controller_state = _load_json(controller_snapshot)
    submission_state = _load_json(submission_snapshot)
    if (
        any(payload.get(key) != value for key, value in expected.items())
        or any(payload.get(key) != value for key, value in snapshot_expected.items())
        or controller_state.get("claim_id") != claim_id
        or controller_state.get("protocol")
        != "j-lens-rl-word-correlation-controller-v1"
        or submission_state.get("claim_id") != claim_id
        or submission_state.get("protocol")
        != "j-lens-rl-word-correlation-submission-v1"
        or marker.get("claim_id") != claim_id
        or marker.get("result_manifest_sha256") != _sha256(path)
        or set(marker["generation_artifact_sha256"])
        != {
            "controller_state.json",
            "result_manifest.json",
            "submission_state.json",
        }
    ):
        raise RuntimeError("committed result marker is invalid")
    return path, payload


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
    manifest = _load_claim_manifest(claim_id)
    if manifest is None:
        raise RuntimeError("word-correlation claim is not committed")
    status = _load_json(REMOTE_OUTPUT / "attempt_status.json")
    if status.get("claim_id") != claim_id:
        raise RuntimeError("word-correlation claim is not available for orchestration")
    _verify_remote_manifest(manifest)
    root_call_id = modal.current_function_call_id()
    if not isinstance(root_call_id, str) or not root_call_id:
        raise RuntimeError("orchestrator has no durable Modal function-call identity")
    _bind_submission_to_controller(claim_id, root_call_id)
    committed = _load_committed_result(claim_id, manifest)
    if committed is not None:
        result, payload = committed
        result_sha256 = _sha256(result)
        if status.get("stage") != "complete":
            _set_status(
                claim_id,
                "complete",
                result_manifest_sha256=result_sha256,
                selection=payload.get("selection"),
            )
            output_volume.commit()
        elif status.get("result_manifest_sha256") != result_sha256:
            raise RuntimeError("terminal correlation result is incomplete")
        _release_gpu_lease(claim_id, root_call_id)
        return {
            "stage": "complete",
            "result_manifest_sha256": result_sha256,
            "selection": payload.get("selection"),
        }
    if status.get("stage") == "complete":
        raise RuntimeError("terminal correlation result is incomplete")
    if status.get("stage") not in RESUMABLE_STAGES:
        raise RuntimeError("word-correlation claim is terminal or not resumable")
    try:
        _refresh_gpu_lease(claim_id, root_call_id)
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
        result = _finalize_result(
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
        _release_gpu_lease(claim_id, root_call_id)
        return result
    except KeyboardInterrupt:
        # Modal restarts a preempted Function with the same call ID.  Durable
        # stage/job checkpoints deliberately remain nonterminal for re-entry.
        raise
    except Exception as error:
        try:
            output_volume.reload()
            current_status = _load_json(REMOTE_OUTPUT / "attempt_status.json")
            if current_status.get("stage") != "complete":
                _set_status(claim_id, "failed", error=repr(error))
                output_volume.commit()
        except Exception:
            pass
        raise


@app.local_entrypoint()
def main(claim_id: str = "") -> None:
    initial_preflight = _local_operational_preflight()
    resolved_claim_id = (
        _validate_claim_id(claim_id) if claim_id else _default_claim_id()
    )
    persisted = inspect_claim.remote(resolved_claim_id)
    if persisted is None:
        manifest = _launch_manifest(initial_preflight, resolved_claim_id)
    else:
        manifest = _validate_persisted_local_manifest(
            persisted,
            resolved_claim_id,
        )
    claim_attempt.remote(manifest)
    # Recheck immediately before the remote compare-and-set lease acquisition.
    # Cooperating apps cannot cross that durable lease boundary, and any
    # already-active non-cooperating Modal app makes this fail closed.
    submission_preflight = _local_operational_preflight()
    submission = submit_attempt.remote(resolved_claim_id, submission_preflight)
    modal.FunctionCall.from_id(str(submission["function_call_id"]))
    print(
        json.dumps(
            {
                "status": "submitted",
                "claim_id": resolved_claim_id,
                "function_call_id": submission["function_call_id"],
                "submission_intent_id": submission["intent_id"],
                "gpu_lease_owner": submission["lease_owner"],
                "volume": VOLUME_NAME,
                "gpu_type": GPU_TYPE,
                "parallel_workers_per_phase": MAX_GPU_CONTAINERS,
                "phase_order": manifest["phase_order"],
            },
            indent=2,
        )
    )
