"""Run the preregistered emotional-word J-space correlation screen on Modal.

The scanner itself lives in :mod:`jlens_rl.word_correlation`; this file owns
only immutable launch provenance, the exposed-data firewall, phased execution,
and durable output bookkeeping.  Discovery and validation are deliberately
separate GPU maps.  The selected word is written to an immutable lock between
the two phases, so validation workers cannot broaden or revise the selection.

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

Only the exposed failed-V4 curve manifest and the target-independent transport
are copied into the image.  In particular, no train-exclusion, sealed-final,
reserve, or retired manifest is available to these jobs.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import modal


LOCAL_REPO = Path(__file__).resolve().parent
REMOTE_REPO = Path("/workspace/j-lens-rl")
REMOTE_OUTPUT = Path("/word_correlation")
CANONICAL_REPO = Path("/j-lens-rl")
LOCAL_ARTIFACTS = CANONICAL_REPO / "artifacts"
LOCAL_MANIFESTS = CANONICAL_REPO / ".confirmatory/manifests"

VOLUME_NAME = "j-lens-rl-word-correlation-v1-20260714a"
GPU_TYPE = "L40S"
NUM_SHARDS = 8
MAX_GPU_CONTAINERS = 8

MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
LENS_SHA256 = "178a9671cbf41882135807bde59b828e36c6f8f98b32c809ea3346860aad10dc"
CURVE_MANIFEST_SHA256 = (
    "ad348fe17d2e6bd6aac691d9bcdbb9da481f675305fa0e05c68e86dad97451c1"
)

LENS_RELATIVE = "artifacts/qwen25_05b_solved_lens.pt"
CURVE_MANIFEST_RELATIVE = ".confirmatory/manifests/curve_indices.json"
CONFIG_RELATIVE = "configs/word_correlation_v1.json"
SCANNER_RELATIVE = "src/jlens_rl/word_correlation.py"
PREREGISTRATION_RELATIVE = "protocol_archive/word_correlation_v1_preregistration.json"

FORBIDDEN_MANIFEST_NAMES = (
    "train_exclusions.json",
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


def _validate_preregistration(repo: Path) -> tuple[str, str, str, set[str]]:
    config_path = repo / CONFIG_RELATIVE
    scanner_path = repo / SCANNER_RELATIVE
    prereg_path = repo / PREREGISTRATION_RELATIVE
    if (
        not config_path.is_file()
        or not scanner_path.is_file()
        or not prereg_path.is_file()
    ):
        raise RuntimeError("word-correlation config, scanner, or preregistration is missing")
    config_sha256 = _sha256(config_path)
    scanner_sha256 = _sha256(scanner_path)
    launcher_sha256 = _sha256(repo / "modal_word_correlation.py")
    launcher_script_sha256 = _sha256(repo / "run_word_correlation.sh")
    prereg = _load_json(prereg_path)
    expected = {
        "protocol": "j-lens-rl-jspace-word-correlation-v1",
        "outcome_status_at_freeze": "not launched and not inspected",
        "model_revision": MODEL_REVISION,
        "dataset_revision": GSM8K_REVISION,
        "curve_manifest_sha256": CURVE_MANIFEST_SHA256,
        "lens_sha256": LENS_SHA256,
        "config_sha256": config_sha256,
        "scanner_sha256": scanner_sha256,
        "launcher_sha256": launcher_sha256,
        "launcher_script_sha256": launcher_script_sha256,
        "volume": VOLUME_NAME,
        "gpu_type": GPU_TYPE,
        "num_shards_per_phase": NUM_SHARDS,
        "max_parallel_gpu_workers": MAX_GPU_CONTAINERS,
        "phase_order": ["discovery", "selection_lock", "validation"],
    }
    if any(prereg.get(key) != value for key, value in expected.items()):
        changed = {
            key: {"expected": value, "actual": prereg.get(key)}
            for key, value in expected.items()
            if prereg.get(key) != value
        }
        raise RuntimeError(f"word-correlation preregistration changed: {changed}")
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
    return _sha256(prereg_path), config_sha256, scanner_sha256, candidates


def _validate_repository_boundary(repo: Path) -> None:
    confirmatory = repo / ".confirmatory/manifests"
    present = sorted(path.name for path in confirmatory.iterdir())
    if present != ["curve_indices.json"]:
        raise RuntimeError(f"unexpected manifest copied into scanner image: {present}")
    for name in FORBIDDEN_MANIFEST_NAMES:
        if (confirmatory / name).exists():
            raise RuntimeError(f"forbidden manifest is available to scanner: {name}")
    artifacts = repo / "artifacts"
    present_artifacts = sorted(path.name for path in artifacts.iterdir())
    if present_artifacts != ["qwen25_05b_solved_lens.pt"]:
        raise RuntimeError(f"unexpected artifact copied into scanner image: {present_artifacts}")


def _launch_manifest() -> dict[str, Any]:
    status = _git("status", "--porcelain=v1", "--untracked-files=all", repo=LOCAL_REPO)
    if status:
        raise RuntimeError(f"word-correlation launch requires a clean committed tree:\n{status}")
    curve_indices = _validate_curve_manifest(LOCAL_REPO)
    lens_path = LOCAL_REPO / LENS_RELATIVE
    if not lens_path.is_file() or _sha256(lens_path) != LENS_SHA256:
        raise RuntimeError("target-independent lens transport is missing or changed")
    prereg_sha256, config_sha256, scanner_sha256, candidates = (
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
        "emotional_candidates": sorted(candidates),
        "phase_order": ["discovery", "selection_lock", "validation"],
        "num_shards_per_phase": NUM_SHARDS,
        "gpu_type": GPU_TYPE,
        "max_parallel_gpu_workers": MAX_GPU_CONTAINERS,
        "data_boundary": "exposed failed-V4 400-item curve only",
        "mounted_inputs": [CURVE_MANIFEST_RELATIVE, LENS_RELATIVE],
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
    prereg_sha256, config_sha256, scanner_sha256, candidates = (
        _validate_preregistration(REMOTE_REPO)
    )
    expected = {
        "curve_manifest_sha256": CURVE_MANIFEST_SHA256,
        "lens_sha256": LENS_SHA256,
        "config_sha256": config_sha256,
        "scanner_sha256": scanner_sha256,
        "launcher_sha256": _sha256(REMOTE_REPO / "modal_word_correlation.py"),
        "launcher_script_sha256": _sha256(REMOTE_REPO / "run_word_correlation.sh"),
        "preregistration_sha256": prereg_sha256,
        "emotional_candidates": sorted(candidates),
        "mounted_inputs": [CURVE_MANIFEST_RELATIVE, LENS_RELATIVE],
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise RuntimeError("remote launch provenance differs from local claim")


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


def _scan_phase(phase: str, shard_index: int) -> dict[str, Any]:
    if phase not in {"discovery", "validation"}:
        raise ValueError(f"invalid correlation phase: {phase}")
    if shard_index not in range(NUM_SHARDS):
        raise ValueError(f"invalid shard index: {shard_index}")
    output_volume.reload()
    manifest = _load_json(REMOTE_OUTPUT / "attempt_manifest.json")
    _verify_remote_manifest(manifest)
    shard_dir = REMOTE_OUTPUT / phase / "shards" / f"shard-{shard_index:02d}"
    sidecar = (
        REMOTE_OUTPUT
        / phase
        / "shard_manifests"
        / f"shard-{shard_index:02d}.json"
    )
    if shard_dir.exists() or sidecar.exists():
        raise FileExistsError(f"refusing to overwrite {phase} shard {shard_index}")
    shard_dir.mkdir(parents=True)
    lock_sha256: str | None = None
    selection_path: Path | None = None
    if phase == "validation":
        selection_path = REMOTE_OUTPUT / "selection_lock.json"
        if not selection_path.is_file():
            raise RuntimeError("validation cannot start without selection lock")
        lock_sha256 = _sha256(selection_path)
    try:
        from jlens_rl.word_correlation import run_shard

        payload = _json_result(
            run_shard(
                config_path=REMOTE_REPO / CONFIG_RELATIVE,
                phase=phase,
                shard=shard_index,
                output_dir=shard_dir,
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
        summary = shard_dir / "summary.json"
        if summary.exists():
            raise RuntimeError("run_shard must return, not prewrite, summary.json")
        _write_json(summary, payload)
        artifact_sha256 = _directory_hashes(shard_dir)
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
        return result
    finally:
        output_volume.commit()


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
    max_containers=1,
    timeout=2 * 60 * 60,
    startup_timeout=60 * 60,
    retries=0,
    single_use_containers=True,
    volumes={REMOTE_OUTPUT: output_volume},
)
def calibrate() -> dict[str, Any]:
    output_volume.reload()
    manifest = _load_json(REMOTE_OUTPUT / "attempt_manifest.json")
    _verify_remote_manifest(manifest)
    output = REMOTE_OUTPUT / "artifacts/calibration.json"
    if output.exists():
        raise FileExistsError("refusing to overwrite word-correlation calibration")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        from jlens_rl.word_correlation import run_calibration

        returned = run_calibration(
            config_path=REMOTE_REPO / CONFIG_RELATIVE,
            output_path=output,
        )
        if output.is_file():
            payload = _load_json(output)
            if (
                returned is not None
                and _json_result(returned, "run_calibration") != payload
            ):
                raise RuntimeError("run_calibration return value differs from its output")
        else:
            payload = _json_result(returned, "run_calibration")
            _write_json(output, payload)
        expected = {
            "model_revision": MODEL_REVISION,
            "wikitext_revision": WIKITEXT_REVISION,
            "lens_sha256": LENS_SHA256,
            "config_sha256": manifest["config_sha256"],
            "scanner_sha256": manifest["scanner_sha256"],
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            raise RuntimeError("calibration provenance mismatch")
        result = {
            "output": output.relative_to(REMOTE_OUTPUT).as_posix(),
            "output_sha256": _sha256(output),
        }
        _write_json(REMOTE_OUTPUT / "artifacts/calibration_manifest.json", result)
        return result
    finally:
        output_volume.commit()


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
def discovery_shard(shard_index: int) -> dict[str, Any]:
    return _scan_phase("discovery", shard_index)


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
def validation_shard(shard_index: int) -> dict[str, Any]:
    return _scan_phase("validation", shard_index)


def _mapped(function: Any, values: Iterable[int]) -> list[dict[str, Any]]:
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


def _aggregate(
    phase: str,
    shard_results: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    output = REMOTE_OUTPUT / phase / "aggregate.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {phase} aggregate")
    merge_dir = REMOTE_OUTPUT / phase / "merged"
    if merge_dir.exists():
        raise FileExistsError(f"refusing to overwrite {phase} merged artifacts")
    merge_dir.mkdir(parents=True)
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
    _write_json(output, payload)
    _validate_scanner_provenance(payload, manifest)
    if payload.get("phase") != phase:
        raise RuntimeError(f"wrong phase in {phase} aggregate")
    expected_indices = _phase_indices(shard_results, phase)
    indices = payload.get("source_indices")
    if not isinstance(indices, list) or set(indices) != expected_indices:
        raise RuntimeError(f"{phase} aggregate source indices differ from shards")
    return output, payload


def _lock_selection(
    discovery_path: Path,
    discovery: dict[str, Any],
    manifest: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
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


def _build_atlas(
    discovery_results: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    output_dir = REMOTE_OUTPUT / "atlas"
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite lexical atlas")
    output_dir.mkdir(parents=True)
    shard_dirs: list[Path] = []
    for result in discovery_results:
        shard_dir = REMOTE_OUTPUT / result["output_dir"]
        if _directory_hashes(shard_dir) != result["artifact_sha256"]:
            raise RuntimeError("discovery shard changed before atlas construction")
        shard_dirs.append(shard_dir)
    from jlens_rl.word_correlation import build_atlas

    payload = _json_result(
        build_atlas(
            config_path=REMOTE_REPO / CONFIG_RELATIVE,
            shard_dirs=shard_dirs,
            calibration_path=REMOTE_OUTPUT / "artifacts/calibration.json",
            output_dir=output_dir,
        ),
        "build_atlas",
    )
    _validate_scanner_provenance(payload, manifest)
    if payload.get("phase") != "discovery":
        raise RuntimeError("lexical atlas is not discovery-only")
    discovery_indices = _phase_indices(discovery_results, "discovery")
    if set(payload.get("source_indices", [])) != discovery_indices:
        raise RuntimeError("lexical atlas source indices differ from discovery")
    output = output_dir / "atlas.json"
    if output.exists():
        raise RuntimeError("build_atlas must return, not prewrite, atlas.json")
    _write_json(output, payload)
    return output, payload


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
    if status.get("claim_id") != claim_id or status.get("stage") != "claimed":
        raise RuntimeError("word-correlation claim is not available for orchestration")
    _verify_remote_manifest(manifest)
    try:
        _set_status(claim_id, "calibrating")
        output_volume.commit()
        calibration = calibrate.remote()
        output_volume.reload()
        _set_status(claim_id, "discovery_running")
        output_volume.commit()
        discovery_shards = _mapped(discovery_shard, range(NUM_SHARDS))
        output_volume.reload()
        discovery_path, discovery = _aggregate(
            "discovery", discovery_shards, manifest
        )
        selection_path, selection_lock = _lock_selection(
            discovery_path, discovery, manifest
        )
        selection_sha256 = _sha256(selection_path)
        _set_status(
            claim_id,
            "selection_locked",
            discovery_aggregate_sha256=_sha256(discovery_path),
            selection_lock_sha256=selection_sha256,
            selection=selection_lock["selection"],
        )
        output_volume.commit()

        _set_status(
            claim_id,
            "validation_running",
            selection_lock_sha256=selection_sha256,
            selection=selection_lock["selection"],
        )
        output_volume.commit()
        validation_shards = _mapped(validation_shard, range(NUM_SHARDS))
        output_volume.reload()
        validation_path, validation = _aggregate(
            "validation", validation_shards, manifest
        )
        discovery_indices = _phase_indices(discovery_shards, "discovery")
        validation_indices = _phase_indices(validation_shards, "validation")
        full_indices = set(_validate_curve_manifest(REMOTE_REPO))
        if discovery_indices & validation_indices:
            raise RuntimeError("discovery and selected-word validation overlap")
        if discovery_indices | validation_indices != full_indices:
            raise RuntimeError("discovery and validation do not partition the curve set")
        if validation.get("selection_lock_sha256") != selection_sha256:
            raise RuntimeError("validation aggregate used a different selection lock")

        _set_status(
            claim_id,
            "atlas_building",
            selection_lock_sha256=selection_sha256,
            validation_aggregate_sha256=_sha256(validation_path),
        )
        output_volume.commit()
        atlas_path, _atlas = _build_atlas(discovery_shards, manifest)

        result_manifest = {
            "protocol": manifest["protocol"],
            "claim_id": claim_id,
            "git_commit": manifest["git_commit"],
            "config_sha256": manifest["config_sha256"],
            "scanner_sha256": manifest["scanner_sha256"],
            "preregistration_sha256": manifest["preregistration_sha256"],
            "curve_manifest_sha256": CURVE_MANIFEST_SHA256,
            "lens_sha256": LENS_SHA256,
            "calibration_sha256": calibration["output_sha256"],
            "discovery_shard_artifact_sha256": {
                str(result["shard_index"]): result["artifact_sha256"]
                for result in discovery_shards
            },
            "discovery_aggregate_sha256": _sha256(discovery_path),
            "selection_lock_sha256": selection_sha256,
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
        result_path = REMOTE_OUTPUT / "result_manifest.json"
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
                "parallel_workers_per_phase": MAX_GPU_CONTAINERS,
                "phase_order": manifest["phase_order"],
            },
            indent=2,
        )
    )
