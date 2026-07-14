"""Fail-closed Modal adapter for the frozen celebration-family V12 attempt.

This file is execution plumbing, not a second scientific protocol.  It is
inert unless ``JLENS_V10_MODAL_CONTRACT`` names a byte-pinned, launch-enabled
contract which is itself bound by the registered V12 protocol spec.  The
scientific protocol owns config derivation, terminal-run verification, curve
semantics, unlock semantics, the one-shot final collection, and analysis.

The GPU schedule is deliberately small:

* four treatment seeds run concurrently on L40S;
* their registered 0/2/3/4 mean curve is verified and must pass;
* only then do the four exact sign-flip controls run concurrently; and
* the existing final runner collects all nine labels serially on one L40S.

There is no retry/resume path.  A fresh, pre-existing Modal Volume and a clean
pushed source checkout are mandatory.  Merely importing this module without a
materialized contract cannot allocate a GPU or create a Volume.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import modal


LOCAL_REPO = Path(__file__).resolve().parent
REMOTE_REPO = Path("/workspace/j-lens-rl")
REMOTE_STATE = Path("/state")
CONTRACT_ENV = "JLENS_V10_MODAL_CONTRACT"
CONTRACT_PROTOCOL = "j-lens-rl-confirmatory-v12-modal-execution-contract-v1"
FROZEN_SCIENTIFIC_PROTOCOL = (
    "j-lens-rl-confirmatory-v12-celebration-u4-u5-u6"
)
SCIENCE_REGISTRATION_PATH = (
    "protocol_archive/v12_celebration_infrastructure_replacement_registration.json"
)
SCIENCE_REGISTRATION_SHA256 = (
    "f58f35419549de5905c7d873a71f67edda73289585025f9084901b61be4a9749"
)
CANDIDATE_FREEZE_PATH = "protocol_archive/v11_celebration_candidate_freeze.json"
CANDIDATE_FREEZE_SHA256 = (
    "dbdc67346906664d8768271ed93830e73de713b3e06326170a5586d8ef17d6f9"
)
INTEGRITY_AMENDMENT_PATH = (
    "protocol_archive/v11_celebration_infrastructure_closeout.json"
)
INTEGRITY_AMENDMENT_SHA256 = (
    "cbc4c78dcac153675e460e4aff344c12a44a55e34c71de300da3195f44d9c806"
)
APP_FALLBACK = "j-lens-rl-confirmatory-v12-unmaterialized"
VOLUME_FALLBACK = "j-lens-rl-confirmatory-v12-unmaterialized"
GPU_TYPE = "L40S"
IMAGE_SPEC = "j-lens-rl-v12-celebration-l40s-v1"
MAX_PARALLEL_TRAINING_GPUS = 4
MAX_PARALLEL_FINAL_GPUS = 1
SEEDS = (224, 225, 226, 227)
CONDITIONS = ("jlens", "signflip")
CURVE_STEPS = (0, 4, 5, 6)
TERMINAL_STEP = 6
CURVE_CRITERION = (
    "M4 > M0, M5 >= M4, and M6 >= M5 on the four-treatment-seed mean"
)
FINAL_MANIFEST_SHA256 = (
    "1c3a544053504848318594ce21eea058d902884ba10c4f39ea3fa7796109b9c8"
)
CALIBRATION_PATH = (
    "protocol_archive/emotional_screen_forensic_bundle/family/artifacts/"
    "celebration_calibration.json"
)
CALIBRATION_SHA256 = (
    "93d05caf4848e745c07d908034b36f0b1ae465d8d89e1681134869c6b87a8ee6"
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
CONTROL_COMPONENTS = (
    {
        "layer": 8,
        "start_fraction": 0.5,
        "end_fraction": 0.75,
        "aggregation": "mean",
        "weight": -1.0,
    },
    {
        "layer": 8,
        "start_fraction": 0.75,
        "end_fraction": 1.0,
        "aggregation": "mean",
        "weight": -0.25,
    },
)
TREATMENT_LABELS = tuple(f"jlens_seed{seed}" for seed in SEEDS)
CONTROL_LABELS = tuple(f"signflip_seed{seed}" for seed in SEEDS)
FINAL_LABELS = ("base", *TREATMENT_LABELS, *CONTROL_LABELS)
# The execution contract cannot be part of ``runtime_source.files`` because it
# commits to that inventory (including its tree identity).  Bake it at its
# registered repository path as a separately hash-bound control input instead.
REMOTE_CONTRACT_PATH = (
    REMOTE_REPO / "protocol_archive/v10_modal_execution_contract.json"
)
REQUIRED_RUNTIME_FILES = {
    "modal_confirmatory_v10_fast.py",
    "scripts/confirmatory_v10_final_protocol.py",
    "scripts/confirmatory_v10_final_runner.py",
    "scripts/confirmatory_v10_train.py",
    "scripts/modal_cache_assets_v7.py",
    "artifacts/qwen25_05b_solved_lens.pt",
    CALIBRATION_PATH,
    ".confirmatory/manifests/curve_indices.json",
    ".confirmatory/manifests/train_exclusions.json",
    "src/jlens_rl/common.py",
    "src/jlens_rl/eval.py",
    "src/jlens_rl/paired_eval.py",
    "src/jlens_rl/train.py",
    "tests/test_v10_final_automation.py",
    "tests/test_paired_eval.py",
    "protocol_archive/v12_celebration_metric_schema.json",
    "pyproject.toml",
    SCIENCE_REGISTRATION_PATH,
    CANDIDATE_FREEZE_PATH,
    INTEGRITY_AMENDMENT_PATH,
}


class ModalV10Error(RuntimeError):
    """A prospective or runtime V10 execution invariant failed."""


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _safe_relative(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ModalV10Error(f"unsafe empty relative path: {value!r}")
    result = Path(value)
    if result.is_absolute() or ".." in result.parts or any(
        part in {"", "."} for part in result.parts
    ):
        raise ModalV10Error(f"unsafe relative path: {value!r}")
    return result


def _exact_candidate() -> dict[str, Any]:
    return {
        "target_words": ["yay", "great", "success", "nice"],
        "calibration_path": CALIBRATION_PATH,
        "calibration_sha256": CALIBRATION_SHA256,
        "score_stride": 10,
        "treatment_score_components": [dict(item) for item in TREATMENT_COMPONENTS],
        "matched_control_score_components": [dict(item) for item in CONTROL_COMPONENTS],
        "learning_rate": 3e-6,
    }


def _exact_design() -> dict[str, Any]:
    return {
        "seeds": list(SEEDS),
        "conditions": list(CONDITIONS),
        "terminal_step": TERMINAL_STEP,
        "curve_steps": list(CURVE_STEPS),
        "curve_criterion": CURVE_CRITERION,
        "treatment_parallelism": MAX_PARALLEL_TRAINING_GPUS,
        "control_parallelism": MAX_PARALLEL_TRAINING_GPUS,
        "controls_require_curve_pass": True,
        "final_examples": 900,
        "final_manifest_sha256": FINAL_MANIFEST_SHA256,
        "final_labels": list(FINAL_LABELS),
        "final_parallelism": MAX_PARALLEL_FINAL_GPUS,
        "final_collection_order": list(FINAL_LABELS),
        "alpha": 0.15,
        "sign_test": {
            "alternative": "two-sided",
            "n_seed_pairs": 4,
            "all_positive_exact_p": 0.125,
            "contrasts": ["treatment_minus_base", "treatment_minus_signflip"],
        },
        "mean_effects_must_be_positive": True,
    }


def execution_plan() -> list[dict[str, Any]]:
    """Return the fixed prospective schedule without touching any state."""
    return [
        {
            "phase": "treatment_training",
            "gpu_type": GPU_TYPE,
            "parallelism": 4,
            "jobs": [
                {"label": label, "condition": "jlens", "seed": seed, "slot": slot}
                for slot, (label, seed) in enumerate(zip(TREATMENT_LABELS, SEEDS))
            ],
        },
        {
            "phase": "registered_curve_gate",
            "gpu_type": None,
            "parallelism": 0,
            "steps": list(CURVE_STEPS),
            "criterion": CURVE_CRITERION,
            "failure_action": "stop_without_controls_or_final",
        },
        {
            "phase": "matched_signflip_training",
            "gpu_type": GPU_TYPE,
            "parallelism": 4,
            "requires": "registered_curve_gate_passed",
            "jobs": [
                {
                    "label": label,
                    "condition": "signflip",
                    "seed": seed,
                    "slot": slot,
                }
                for slot, (label, seed) in enumerate(zip(CONTROL_LABELS, SEEDS))
            ],
        },
        {
            "phase": "sealed_final_collection",
            "gpu_type": GPU_TYPE,
            "parallelism": 1,
            "requires": "all_eight_runs_verified_and_unlocked",
            "jobs": [{"sequence": i, "label": label} for i, label in enumerate(FINAL_LABELS, 1)],
        },
    ]


def validate_contract_shape(
    value: Any, *, allow_disabled: bool = False
) -> dict[str, Any]:
    """Validate the execution contract without reading scientific payloads."""
    required = {
        "schema_version", "protocol", "launch_enabled", "repository_path",
        "scientific_protocol", "science_registration", "candidate", "design",
        "hardware", "modal", "runtime_source", "prepared_state",
        "protected_final", "tracking", "image_identity",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ModalV10Error("V10 Modal contract has an unexpected top-level schema")
    if (
        value.get("schema_version") != 1
        or value.get("protocol") != CONTRACT_PROTOCOL
        or not isinstance(value.get("launch_enabled"), bool)
        or (not allow_disabled and value.get("launch_enabled") is not True)
        or value.get("scientific_protocol") != FROZEN_SCIENTIFIC_PROTOCOL
        or value.get("candidate") != _exact_candidate()
        or value.get("design") != _exact_design()
    ):
        raise ModalV10Error("V10 Modal contract changed the frozen candidate/design")
    _safe_relative(value.get("repository_path"))
    if value.get("science_registration") != {
        "draft_path": SCIENCE_REGISTRATION_PATH,
        "draft_sha256": SCIENCE_REGISTRATION_SHA256,
        "candidate_freeze_path": CANDIDATE_FREEZE_PATH,
        "candidate_freeze_sha256": CANDIDATE_FREEZE_SHA256,
        "integrity_amendment_path": INTEGRITY_AMENDMENT_PATH,
        "integrity_amendment_sha256": INTEGRITY_AMENDMENT_SHA256,
    }:
        raise ModalV10Error("V10 science-registration binding changed")
    amendment = value["science_registration"]

    if value.get("hardware") != {
        "backend": "modal",
        "gpu_type": GPU_TYPE,
        "max_parallel_training_workers": 4,
        "max_parallel_final_workers": 1,
        "one_gpu_per_worker": True,
    }:
        raise ModalV10Error("V10 Modal hardware or GPU ceiling changed")
    if value.get("image_identity") != {
        "jlens_modal_image_spec": IMAGE_SPEC,
        "modal_image_id_policy": "record_observed_per_container_not_prebound",
    }:
        raise ModalV10Error("V10 Modal image identity policy changed")

    modal_config = value.get("modal")
    modal_keys = {
        "app_name", "state_volume_name", "state_volume_object_id",
        "state_volume_version", "wandb_secret_name", "huggingface_secret_name",
    }
    if (
        not isinstance(modal_config, dict)
        or set(modal_config) != modal_keys
        or not all(
            isinstance(modal_config.get(key), str) and modal_config[key]
            for key in modal_keys - {"state_volume_version"}
        )
        or modal_config.get("state_volume_version") != 2
        or not str(modal_config["state_volume_object_id"]).startswith("vo-")
    ):
        raise ModalV10Error("V10 Modal resource identity is incomplete")

    source = value.get("runtime_source")
    source_keys = {
        "files", "git_tree", "git_commit", "source_tree_sha256", "commit_recipe"
    }
    if (
        not isinstance(source, dict)
        or set(source) != source_keys
        or re.fullmatch(r"[0-9a-f]{40}", str(source.get("git_tree"))) is None
        or re.fullmatch(r"[0-9a-f]{40}", str(source.get("git_commit"))) is None
        or not _is_sha256(source.get("source_tree_sha256"))
        or source.get("commit_recipe")
        != {
            "author": "J-Lens V10 Modal Runtime <runtime@example.invalid>",
            "timestamp": "2000-01-01T00:00:00+00:00",
            "message": "J-Lens V10 byte-pinned Modal runtime",
            "parent": None,
        }
        or not isinstance(source.get("files"), dict)
        or not source["files"]
    ):
        raise ModalV10Error("V10 Modal runtime-source identity is incomplete")
    for name, identity in source["files"].items():
        _safe_relative(name)
        if (
            not isinstance(identity, dict)
            or set(identity) != {"sha256", "size_bytes", "mode"}
            or not _is_sha256(identity.get("sha256"))
            or not isinstance(identity.get("size_bytes"), int)
            or identity["size_bytes"] < 0
            or identity.get("mode") not in {0o644, 0o755}
        ):
            raise ModalV10Error(f"unsafe runtime-source identity for {name!r}")
    if not REQUIRED_RUNTIME_FILES <= set(source["files"]):
        missing = sorted(REQUIRED_RUNTIME_FILES - set(source["files"]))
        raise ModalV10Error(f"V10 Modal runtime allowlist lacks {missing}")
    if value["science_registration"]["integrity_amendment_path"] not in source["files"]:
        raise ModalV10Error("V10 runtime omits the registered integrity amendment")

    prepared = value.get("prepared_state")
    if (
        not isinstance(prepared, dict)
        or set(prepared)
        != {"local_path", "remote_path", "expected_files"}
        or not isinstance(prepared.get("local_path"), str)
        or prepared.get("remote_path") != REMOTE_STATE.as_posix()
        or not isinstance(prepared.get("expected_files"), list)
        or prepared["expected_files"] != sorted(set(prepared["expected_files"]))
        or not prepared["expected_files"]
        or any(not isinstance(name, str) for name in prepared["expected_files"])
    ):
        raise ModalV10Error("V10 prepared-state inventory is incomplete")
    for name in prepared["expected_files"]:
        _safe_relative(name)
    if "manifests/sealed_final_indices.json" in prepared["expected_files"]:
        raise ModalV10Error("protected final manifest was staged before unlock")
    protected = value.get("protected_final")
    if (
        not isinstance(protected, dict)
        or set(protected)
        != {"local_path", "remote_relative_path", "sha256", "release_policy"}
        or not isinstance(protected.get("local_path"), str)
        or protected.get("remote_relative_path")
        != "manifests/sealed_final_indices.json"
        or protected.get("sha256") != FINAL_MANIFEST_SHA256
        or protected.get("release_policy")
        != "opaque_upload_only_after_passing_curve_all_eight_runs_and_unlock"
    ):
        raise ModalV10Error("V10 protected-final release contract changed")
    if value.get("tracking") != {
        "training_mode": "online",
        "wandb_visible_during_training": True,
        "resume": "never",
    }:
        raise ModalV10Error("V10 tracking/evidence preservation policy changed")
    return value


def expected_spec_modal_binding(contract: Mapping[str, Any], digest: str) -> dict[str, Any]:
    if not _is_sha256(digest):
        raise ModalV10Error("execution contract SHA-256 is malformed")
    return {
        "contract_path": contract["repository_path"],
        "contract_sha256": digest,
    }


def validate_scientific_binding(
    contract: Mapping[str, Any], spec: Mapping[str, Any], contract_digest: str
) -> None:
    """Require the registered spec to bind this exact Modal execution."""
    candidate = contract["candidate"]
    design = contract["design"]
    calibration = spec.get("artifacts", {})
    training = spec.get("training", {})
    hardware = spec.get("hardware", {})
    final = spec.get("final_collection", {})
    curve = spec.get("curve_gate", {})
    registration = contract["science_registration"]
    if (
        spec.get("protocol") != contract["scientific_protocol"]
        or spec.get("target_words") != candidate["target_words"]
        or spec.get("seeds") != design["seeds"]
        or spec.get("conditions") != design["conditions"]
        or spec.get("terminal_step") != TERMINAL_STEP
        or curve.get("steps") != list(CURVE_STEPS)
        or curve.get("criterion") != CURVE_CRITERION
        or spec.get("treatment_score_components")
        != candidate["treatment_score_components"]
        or spec.get("matched_control_score_components")
        != candidate["matched_control_score_components"]
        or calibration.get("calibration_path") != CALIBRATION_PATH
        or calibration.get("calibration_sha256") != CALIBRATION_SHA256
        or training.get("updates") != 6
        or training.get("learning_rate") != 3e-6
        or training.get("score_stride") != 10
        or training.get("validation_steps") != [4, 5, 6]
        or str(hardware.get("backend")).lower() != "modal"
        or "L40S" not in str(hardware.get("device_name"))
        or hardware.get("max_gpu_processes") != 1
        or hardware.get("gpu_per_worker") != 1
        or hardware.get("max_modal_gpus_before_2026_07_14_23_00_utc") != 5
        or hardware.get("max_modal_gpus_at_or_after_2026_07_14_23_00_utc") != 10
        or final.get("count") != 900
        or final.get("manifest_sha256") != FINAL_MANIFEST_SHA256
        or final.get("manifest_path")
        != "/state/manifests/sealed_final_indices.json"
        or final.get("labels") != list(FINAL_LABELS)
        or spec.get("modal_execution")
        != expected_spec_modal_binding(contract, contract_digest)
        or spec.get("wandb", {}).get("mode") != "online"
        or spec.get("paths", {}).get("state_config_prefix") != REMOTE_STATE.as_posix()
        or spec.get("repository") != REMOTE_REPO.as_posix()
        or spec.get("git_commit") != contract["runtime_source"]["git_commit"]
        or spec.get("source_tree_sha256")
        != contract["runtime_source"]["source_tree_sha256"]
        or spec.get("science_registration")
        != {"path": registration["draft_path"], "sha256": registration["draft_sha256"]}
        or spec.get("candidate_freeze")
        != {
            "path": registration["candidate_freeze_path"],
            "sha256": registration["candidate_freeze_sha256"],
        }
        or spec.get("candidate_freeze_correction")
        != {
            "path": registration["integrity_amendment_path"],
            "sha256": registration["integrity_amendment_sha256"],
        }
    ):
        raise ModalV10Error(
            "registered V12 spec does not bind the frozen celebration Modal attempt"
        )


def curve_gate_from_histories(
    histories: Mapping[str, Mapping[int | str, Mapping[str, Any]]]
) -> dict[str, Any]:
    """Build the exact four-treatment-seed gate from already verified histories."""
    if set(histories) != set(TREATMENT_LABELS):
        raise ModalV10Error("curve gate requires exactly four treatment histories")
    per_seed: dict[str, dict[str, float]] = {}
    for seed in SEEDS:
        label = f"jlens_seed{seed}"
        history = histories[label]
        normalized = {int(step): row for step, row in history.items()}
        if set(normalized) != set(CURVE_STEPS):
            raise ModalV10Error(
                f"{label} history is not exact "
                + "/".join(str(step) for step in CURVE_STEPS)
            )
        values: dict[str, float] = {}
        for step in CURVE_STEPS:
            exact = normalized[step].get("exact_match")
            if (
                isinstance(exact, bool)
                or not isinstance(exact, (int, float))
                or not 0 <= float(exact) <= 1
            ):
                raise ModalV10Error(f"{label} step {step} exact_match is invalid")
            values[str(step)] = float(exact)
        per_seed[str(seed)] = values
    means = {
        str(step): sum(per_seed[str(seed)][str(step)] for seed in SEEDS) / len(SEEDS)
        for step in CURVE_STEPS
    }
    ordered = [means[str(step)] for step in CURVE_STEPS]
    passed = ordered[1] > ordered[0] and ordered[2] >= ordered[1] and ordered[3] >= ordered[2]
    return {
        "steps": list(CURVE_STEPS),
        "criterion": CURVE_CRITERION,
        "n_seeds": len(SEEDS),
        "per_seed_exact_match": per_seed,
        "mean_exact_match": means,
        "passed": passed,
    }


def _load_local_contract() -> tuple[Path | None, dict[str, Any] | None, str | None]:
    raw_path = os.environ.get(CONTRACT_ENV)
    if not raw_path:
        return None, None, None
    path = Path(raw_path).resolve()
    if not path.is_file() or path.is_symlink():
        raise ModalV10Error(f"{CONTRACT_ENV} is absent or unsafe: {path}")
    value = validate_contract_shape(json.loads(path.read_text()))
    expected_path = (LOCAL_REPO / value["repository_path"]).resolve()
    if path != expected_path:
        raise ModalV10Error("materialized contract path differs from its repository binding")
    return path, value, _sha256(path)


LOCAL_CONTRACT_PATH, CONTRACT, CONTRACT_SHA256 = _load_local_contract()
APP_NAME = CONTRACT["modal"]["app_name"] if CONTRACT else APP_FALLBACK
VOLUME_NAME = (
    CONTRACT["modal"]["state_volume_name"] if CONTRACT else VOLUME_FALLBACK
)

app = modal.App(APP_NAME)
state_volume = modal.Volume.from_name(
    VOLUME_NAME, create_if_missing=False, version=2
)
wandb_secret = modal.Secret.from_name(
    CONTRACT["modal"]["wandb_secret_name"] if CONTRACT else "j-lens-rl-wandb"
)
huggingface_secret = modal.Secret.from_name(
    CONTRACT["modal"]["huggingface_secret_name"]
    if CONTRACT
    else "huggingface-token"
)


def _build_image() -> modal.Image:
    image = modal.Image.debian_slim(python_version="3.11")
    if CONTRACT is None or LOCAL_CONTRACT_PATH is None or CONTRACT_SHA256 is None:
        return image
    expected_files = set(CONTRACT["runtime_source"]["files"])
    expected_directories = {
        parent.as_posix()
        for relative in expected_files
        for parent in Path(relative).parents
        if parent != Path(".")
    }
    for relative, identity in sorted(CONTRACT["runtime_source"]["files"].items()):
        local_path = LOCAL_REPO / relative
        if (
            not local_path.is_file()
            or local_path.is_symlink()
            or _sha256(local_path) != identity["sha256"]
            or local_path.stat().st_size != identity["size_bytes"]
        ):
            raise ModalV10Error(f"registered runtime source changed: {relative}")
    def ignore_unregistered(path: Path) -> bool:
        try:
            relative = path.relative_to(LOCAL_REPO).as_posix()
        except ValueError:
            relative = path.as_posix().lstrip("./")
        return relative not in expected_files and relative not in expected_directories

    # One allowlisted directory layer avoids hundreds of sequential COPY image
    # builds while retaining the exact same fail-closed runtime inventory.
    image = image.add_local_dir(
        LOCAL_REPO,
        REMOTE_REPO.as_posix(),
        copy=True,
        ignore=ignore_unregistered,
    )
    # Modal's local-directory layer honors repository ignore rules; these two
    # public, gitignored manifests therefore need explicit allowlisted copies.
    for relative in (
        ".confirmatory/manifests/curve_indices.json",
        ".confirmatory/manifests/train_exclusions.json",
    ):
        image = image.add_local_file(
            LOCAL_REPO / relative,
            (REMOTE_REPO / relative).as_posix(),
            copy=True,
        )
    image = image.add_local_file(
        LOCAL_CONTRACT_PATH, REMOTE_CONTRACT_PATH.as_posix(), copy=True
    )
    return (
        image.apt_install("git")
        .workdir(REMOTE_REPO)
        .env(
            {
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "JLENS_MODAL_IMAGE_SPEC": IMAGE_SPEC,
                "JLENS_V10_MODAL_CONTRACT_SHA256": CONTRACT_SHA256,
                "JLENS_REPOSITORY_ROOT": REMOTE_REPO.as_posix(),
                "PYTHONPATH": (
                    f"{(REMOTE_REPO / 'src').as_posix()}:"
                    f"{(REMOTE_REPO / 'trl').as_posix()}"
                ),
                "PYTHONDONTWRITEBYTECODE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONUNBUFFERED": "1",
            }
        )
        .run_commands(
            "python -m pip install --upgrade pip==26.0.1",
            "python -m pip install './trl[peft]' '.[dev]'",
        )
        .run_commands(
            "python scripts/modal_cache_assets_v7.py", secrets=[huggingface_secret]
        )
        .run_commands(
            "find . -type d -name __pycache__ -prune -exec rm -rf {} +",
            "find . -type d -name '*.egg-info' -prune -exec rm -rf {} +",
            "rm -rf build trl/build .git",
        )
    )


repo_image = _build_image()


def _write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())


def _replace_status(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _runtime_contract() -> tuple[dict[str, Any], str]:
    if not REMOTE_CONTRACT_PATH.is_file():
        raise ModalV10Error("V10 GPU work is inert without the baked contract")
    digest = _sha256(REMOTE_CONTRACT_PATH)
    if digest != os.environ.get("JLENS_V10_MODAL_CONTRACT_SHA256"):
        raise ModalV10Error("baked V10 contract identity changed")
    contract = validate_contract_shape(json.loads(REMOTE_CONTRACT_PATH.read_text()))
    if os.environ.get("JLENS_MODAL_IMAGE_SPEC") != contract["image_identity"][
        "jlens_modal_image_spec"
    ]:
        raise ModalV10Error("runtime Modal image spec differs from its contract")
    return contract, digest


def _verify_runtime_source(contract: Mapping[str, Any]) -> None:
    expected = contract["runtime_source"]["files"]
    registered_contract = Path(contract["repository_path"])
    if (
        REMOTE_CONTRACT_PATH != REMOTE_REPO / registered_contract
        or not REMOTE_CONTRACT_PATH.is_file()
        or REMOTE_CONTRACT_PATH.is_symlink()
        or _sha256(REMOTE_CONTRACT_PATH)
        != os.environ.get("JLENS_V10_MODAL_CONTRACT_SHA256")
    ):
        raise ModalV10Error("separately bound V10 execution contract changed")
    observed: set[str] = set()
    for path in REMOTE_REPO.rglob("*"):
        if ".git" in path.relative_to(REMOTE_REPO).parts:
            continue
        if path.is_symlink():
            raise ModalV10Error(f"runtime source contains a symlink: {path}")
        if path.is_file():
            relative = path.relative_to(REMOTE_REPO).as_posix()
            if relative == registered_contract.as_posix():
                continue
            observed.add(relative)
    if observed != set(expected):
        raise ModalV10Error(
            f"runtime source inventory changed: missing={sorted(set(expected)-observed)}, "
            f"unexpected={sorted(observed-set(expected))}"
        )
    for relative, identity in expected.items():
        path = REMOTE_REPO / relative
        if (
            _sha256(path) != identity["sha256"]
            or path.stat().st_size != identity["size_bytes"]
        ):
            raise ModalV10Error(f"runtime source bytes changed: {relative}")
        path.chmod(identity["mode"])


def _ensure_runtime_git(contract: Mapping[str, Any]) -> None:
    _verify_runtime_source(contract)
    source = contract["runtime_source"]
    recipe = source["commit_recipe"]
    environment = {
        **os.environ,
        "GIT_AUTHOR_NAME": "J-Lens V10 Modal Runtime",
        "GIT_AUTHOR_EMAIL": "runtime@example.invalid",
        "GIT_COMMITTER_NAME": "J-Lens V10 Modal Runtime",
        "GIT_COMMITTER_EMAIL": "runtime@example.invalid",
        "GIT_AUTHOR_DATE": recipe["timestamp"],
        "GIT_COMMITTER_DATE": recipe["timestamp"],
    }
    if not (REMOTE_REPO / ".git").exists():
        subprocess.run(["git", "init", "-q"], cwd=REMOTE_REPO, check=True, env=environment)
        subprocess.run(
            ["git", "add", "--", *sorted(source["files"])],
            cwd=REMOTE_REPO,
            check=True,
            env=environment,
        )
        tree = subprocess.check_output(
            ["git", "write-tree"], cwd=REMOTE_REPO, text=True, env=environment
        ).strip()
        if tree != source["git_tree"]:
            raise ModalV10Error("reconstructed V10 Git tree differs from registration")
        commit = subprocess.check_output(
            ["git", "commit-tree", tree],
            cwd=REMOTE_REPO,
            input=f"{recipe['message']}\n",
            text=True,
            env=environment,
        ).strip()
        if commit != source["git_commit"]:
            raise ModalV10Error("reconstructed V10 Git commit differs from registration")
        subprocess.run(
            ["git", "update-ref", "HEAD", commit], cwd=REMOTE_REPO, check=True
        )
        (REMOTE_REPO / ".git" / "info" / "exclude").write_text("*\n")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REMOTE_REPO, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=REMOTE_REPO,
        text=True,
    )
    if commit != source["git_commit"] or status:
        raise ModalV10Error("V10 runtime Git identity is not exact and clean")


def _load_protocol_context() -> tuple[Any, Any, dict[str, Any], str]:
    contract, digest = _runtime_contract()
    _ensure_runtime_git(contract)
    from scripts import confirmatory_v10_final_protocol as protocol

    spec_path = REMOTE_STATE / "reproducibility" / "final_protocol_spec.json"
    if not spec_path.is_file():
        raise ModalV10Error("prepared V10 protocol spec is absent")
    spec = json.loads(spec_path.read_text())
    # This call is intentionally first-party and strict.  In particular, the
    # old local-4090 V10 protocol cannot be repurposed as Modal evidence.
    protocol.validate_spec(spec)
    if "modal_execution" not in getattr(protocol, "REGISTERED_SPEC_FIELDS", ()):
        raise ModalV10Error("V10 registration projection does not bind Modal execution")
    validate_scientific_binding(contract, spec, digest)
    context = protocol.load_context(REMOTE_STATE)
    return protocol, context, contract, digest


def _attempt_claim(claim_id: str) -> dict[str, Any]:
    path = REMOTE_STATE / "attempt_claim.json"
    if not path.is_file():
        raise ModalV10Error("V10 attempt claim is absent")
    value = json.loads(path.read_text())
    if (
        value.get("claim_id") != claim_id
        or value.get("contract_sha256") != _sha256(REMOTE_CONTRACT_PATH)
        or value.get("execution_plan") != execution_plan()
        or value.get("retry_or_resume_permitted") is not False
    ):
        raise ModalV10Error("V10 attempt claim changed")
    return value


def _wait_for_root_receipt(claim_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 600
    path = REMOTE_STATE / "launch_receipt.json"
    while time.monotonic() < deadline:
        state_volume.reload()
        if path.is_file():
            value = json.loads(path.read_text())
            if value.get("claim_id") != claim_id or value.get("status") != "present":
                raise ModalV10Error("V10 launch receipt changed")
            return value
        time.sleep(2)
    raise ModalV10Error("durable V10 root-call receipt did not arrive")


def _wait_for_protected_upload(claim_id: str, root_call_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 3 * 60 * 60
    path = REMOTE_STATE / "protected_final_upload_receipt.json"
    while time.monotonic() < deadline:
        state_volume.reload()
        if path.is_file():
            value = json.loads(path.read_text())
            if (
                value.get("claim_id") != claim_id
                or value.get("root_call_id") != root_call_id
                or value.get("sha256") != FINAL_MANIFEST_SHA256
                or value.get("uploaded_only_after_unlock") is not True
            ):
                raise ModalV10Error("protected-final upload receipt changed")
            return value
        time.sleep(2)
    raise ModalV10Error("protected final was not released after durable unlock")


def _dispatch_path(label: str, kind: str) -> Path:
    if label not in {*TREATMENT_LABELS, *CONTROL_LABELS, "sealed_final"}:
        raise ModalV10Error(f"unregistered V10 dispatch label: {label}")
    return REMOTE_STATE / "evidence" / "modal_dispatches" / f"{label}.{kind}.json"


def _write_training_intents(
    claim_id: str, root_call_id: str, condition: str
) -> list[dict[str, Any]]:
    labels = TREATMENT_LABELS if condition == "jlens" else CONTROL_LABELS
    intents = []
    for slot, (label, seed) in enumerate(zip(labels, SEEDS)):
        config_path = REMOTE_STATE / "configs" / f"{label}.json"
        value = {
            "schema_version": 1,
            "protocol": CONTRACT_PROTOCOL,
            "claim_id": claim_id,
            "root_call_id": root_call_id,
            "phase": "treatment_training" if condition == "jlens" else "matched_signflip_training",
            "condition": condition,
            "seed": seed,
            "label": label,
            "slot": slot,
            "gpu_type": GPU_TYPE,
            "config_sha256": _sha256(config_path),
            "contract_sha256": _sha256(REMOTE_CONTRACT_PATH),
            "status": "written_before_gpu_schedule",
        }
        _write_json_exclusive(_dispatch_path(label, "intent"), value)
        intents.append(value)
    state_volume.commit()
    return intents


def _verify_training_intent(intent: Mapping[str, Any]) -> None:
    label = str(intent.get("label"))
    observed = json.loads(_dispatch_path(label, "intent").read_text())
    if observed != intent or intent.get("status") != "written_before_gpu_schedule":
        raise ModalV10Error("V10 GPU worker lacks an exact pre-dispatch intent")
    _attempt_claim(str(intent.get("claim_id")))


def _verified_run(protocol: Any, context: Any, condition: str, seed: int) -> tuple[Any, Any, Any]:
    firewall = protocol.verify_nonprotected_bindings(context)
    return protocol._verify_one_completed_run(
        context,
        condition,
        seed,
        curve_indices=firewall["curve_indices"],
        excluded_indices=set(firewall["excluded_indices"]),
    )


@app.function(
    image=repo_image,
    cpu=4,
    memory=32768,
    timeout=3 * 60 * 60,
    startup_timeout=60 * 60,
    retries=0,
    volumes={REMOTE_STATE: state_volume},
)
def claim_attempt(claim_id: str, preflight: dict[str, Any]) -> dict[str, Any]:
    state_volume.reload()
    protocol, context, contract, digest = _load_protocol_context()
    readiness = protocol.verify_preunlock_readiness(context)
    if readiness.get("protected_final_manifest_read") is not False:
        raise ModalV10Error("preunlock readiness touched the protected final manifest")
    value = {
        "schema_version": 1,
        "protocol": CONTRACT_PROTOCOL,
        "scientific_protocol": contract["scientific_protocol"],
        "claim_id": claim_id,
        "contract_sha256": digest,
        "preflight": preflight,
        "execution_plan": execution_plan(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "retry_or_resume_permitted": False,
        "protected_final_outcomes_read": False,
    }
    _write_json_exclusive(REMOTE_STATE / "attempt_claim.json", value)
    state_volume.commit()
    return value


@app.function(
    image=repo_image,
    cpu=1,
    memory=2048,
    timeout=20 * 60,
    retries=0,
    volumes={REMOTE_STATE: state_volume},
)
def record_launch_receipt(claim_id: str, root_call_id: str) -> dict[str, Any]:
    state_volume.reload()
    _attempt_claim(claim_id)
    value = {
        "schema_version": 1,
        "protocol": CONTRACT_PROTOCOL,
        "claim_id": claim_id,
        "root_call_id": root_call_id,
        "app_id": app.app_id or APP_NAME,
        "status": "present",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_exclusive(REMOTE_STATE / "launch_receipt.json", value)
    state_volume.commit()
    return value


@app.function(
    image=repo_image,
    cpu=1,
    memory=2048,
    timeout=4 * 60 * 60,
    retries=0,
    volumes={REMOTE_STATE: state_volume},
)
def await_protected_final_release(claim_id: str, root_call_id: str) -> dict[str, Any]:
    """Wait without touching the protected local file until remote unlock."""
    deadline = time.monotonic() + 3 * 60 * 60
    authorization_path = REMOTE_STATE / "protected_final_upload_authorized.json"
    status_path = REMOTE_STATE / "attempt_status.json"
    while time.monotonic() < deadline:
        state_volume.reload()
        if authorization_path.is_file():
            value = json.loads(authorization_path.read_text())
            if (
                value.get("claim_id") != claim_id
                or value.get("root_call_id") != root_call_id
                or value.get("final_manifest_sha256") != FINAL_MANIFEST_SHA256
                or value.get("curve_passed") is not True
                or value.get("all_eight_runs_verified") is not True
                or value.get("unlock_verified") is not True
            ):
                raise ModalV10Error("protected-final release authorization changed")
            return {"authorized": True, "authorization": value}
        if status_path.is_file():
            status = json.loads(status_path.read_text())
            if status.get("stage") in {
                "curve_failed_no_controls_or_final",
                "failed_closed",
            }:
                return {"authorized": False, "terminal_status": status}
        time.sleep(2)
    raise ModalV10Error("timed out waiting for protected-final release authorization")


@app.function(
    image=repo_image,
    cpu=1,
    memory=2048,
    timeout=20 * 60,
    retries=0,
    volumes={REMOTE_STATE: state_volume},
)
def record_protected_final_upload(claim_id: str, root_call_id: str) -> dict[str, Any]:
    """Hash, but never parse, the post-unlock protected upload."""
    state_volume.reload()
    authorization_path = REMOTE_STATE / "protected_final_upload_authorized.json"
    authorization = json.loads(authorization_path.read_text())
    if (
        authorization.get("claim_id") != claim_id
        or authorization.get("root_call_id") != root_call_id
        or authorization.get("unlock_verified") is not True
    ):
        raise ModalV10Error("protected upload lacks exact post-unlock authorization")
    protected_path = REMOTE_STATE / "manifests" / "sealed_final_indices.json"
    if (
        not protected_path.is_file()
        or protected_path.is_symlink()
        or _sha256(protected_path) != FINAL_MANIFEST_SHA256
    ):
        raise ModalV10Error("post-unlock protected upload identity differs")
    value = {
        "schema_version": 1,
        "protocol": CONTRACT_PROTOCOL,
        "claim_id": claim_id,
        "root_call_id": root_call_id,
        "authorization_sha256": _sha256(authorization_path),
        "sha256": FINAL_MANIFEST_SHA256,
        "uploaded_only_after_unlock": True,
        "semantic_payload_inspected": False,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_exclusive(REMOTE_STATE / "protected_final_upload_receipt.json", value)
    state_volume.commit()
    return value


@app.function(
    image=repo_image,
    gpu=GPU_TYPE,
    cpu=4,
    memory=32768,
    max_containers=MAX_PARALLEL_TRAINING_GPUS,
    timeout=3 * 60 * 60,
    startup_timeout=60 * 60,
    retries=0,
    single_use_containers=True,
    secrets=[huggingface_secret, wandb_secret],
    volumes={REMOTE_STATE: state_volume},
)
def train_one(intent: dict[str, Any]) -> dict[str, Any]:
    state_volume.reload()
    _verify_training_intent(intent)
    condition = intent["condition"]
    seed = int(intent["seed"])
    label = intent["label"]
    if (
        condition not in CONDITIONS
        or seed not in SEEDS
        or label != f"{condition}_seed{seed}"
    ):
        raise ModalV10Error("training worker input is outside frozen V10")
    protocol, context, _contract, _digest = _load_protocol_context()
    if condition == "signflip":
        gate = _build_and_verify_curve(protocol, context, write=False)
        if gate["passed"] is not True:
            raise ModalV10Error("signflip worker cannot run before a passing curve")
    config = protocol.expected_training_config(context, condition, seed)
    config_path = context.config_dir / f"{label}.json"
    if json.loads(config_path.read_text()) != config:
        raise ModalV10Error(f"registered config changed for {label}")
    command = config["registered_command"]
    completed = subprocess.run(command, cwd=context.repository, check=False)
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)
    record, _history, _indices = _verified_run(protocol, context, condition, seed)
    completion = {
        "schema_version": 1,
        "protocol": CONTRACT_PROTOCOL,
        "claim_id": intent["claim_id"],
        "label": label,
        "intent_sha256": _sha256(_dispatch_path(label, "intent")),
        "run_result_sha256": record["run_result_sha256"],
        "validation_history_sha256": record["validation_history_sha256"],
        "receipt_sha256": record["receipt_sha256"],
        "status": "terminal_run_verified_before_completion",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_exclusive(_dispatch_path(label, "completion"), completion)
    state_volume.commit()
    return completion


def _build_and_verify_curve(protocol: Any, context: Any, *, write: bool) -> dict[str, Any]:
    histories = {}
    for seed in SEEDS:
        _record, history, _indices = _verified_run(protocol, context, "jlens", seed)
        histories[f"jlens_seed{seed}"] = history
    value = curve_gate_from_histories(histories)
    if value["criterion"] != protocol.CURVE_CRITERION or tuple(value["steps"]) != tuple(
        protocol.CURVE_STEPS
    ):
        raise ModalV10Error("launcher/protocol curve semantics differ")
    path = context.curve_path
    if write:
        _write_json_exclusive(path, value)
        state_volume.commit()
    else:
        if not path.is_file() or json.loads(path.read_text()) != value:
            raise ModalV10Error("stored treatment curve changed")
    return value


def _write_and_verify_unlock(protocol: Any, context: Any) -> dict[str, Any]:
    firewall = protocol.verify_nonprotected_bindings(context)
    runs: dict[str, Any] = {}
    histories: dict[str, Any] = {}
    matched_train: dict[int, list[int]] = {}
    for condition in CONDITIONS:
        for seed in SEEDS:
            record, history, indices = protocol._verify_one_completed_run(
                context,
                condition,
                seed,
                curve_indices=firewall["curve_indices"],
                excluded_indices=set(firewall["excluded_indices"]),
            )
            runs[record["label"]] = record
            histories[record["label"]] = history
            if seed in matched_train and matched_train[seed] != indices:
                raise ModalV10Error(f"treatment/control training rows differ for seed {seed}")
            matched_train[seed] = indices
    inventory = {
        "schema_version": 1,
        "protocol": context.spec["protocol"],
        "git_commit": context.spec["git_commit"],
        "registration_sha256": context.spec["registration_sha256"],
        "recipe_lock_sha256": context.spec["recipe_lock_sha256"],
        "recipe_sha256": context.spec["recipe_sha256"],
        "registered_code_sha256": context.spec["registered_code_sha256"],
        "registered_spec_projection_sha256": protocol.registered_spec_projection_sha256(
            context.spec
        ),
        "seeds": list(SEEDS),
        "conditions": list(CONDITIONS),
        "terminal_step": TERMINAL_STEP,
        "hardware": context.spec["hardware"],
        "source_tree_sha256": context.spec["source_tree_sha256"],
        "runs": runs,
    }
    _write_json_exclusive(context.completed_runs_path, inventory)
    state_volume.commit()
    completed = protocol.verify_completed_inventory(context, firewall)
    curve = protocol._verify_curve(context, completed)
    if curve.get("passed") is not True:
        raise ModalV10Error("V10 controls completed without a passing registered curve")
    unlock = {
        "protocol": context.spec["protocol"],
        "git_commit": context.spec["git_commit"],
        "registration_sha256": context.spec["registration_sha256"],
        "curve_gate_sha256": protocol.sha256_file(context.curve_path),
        "completed_runs_sha256": protocol.sha256_file(context.completed_runs_path),
        "automation_audit_sha256": context.spec["automation_audit"]["sha256"],
        "recipe_lock_sha256": context.spec["recipe_lock_sha256"],
        "final_manifest_sha256": context.spec["final_collection"]["manifest_sha256"],
        "disjointness_receipt_sha256": context.spec["firewall"]["disjointness_receipt"]["sha256"],
        "recipe_sha256": context.spec["recipe_sha256"],
        "registered_code_sha256": context.spec["registered_code_sha256"],
        "registered_spec_projection_sha256": protocol.registered_spec_projection_sha256(
            context.spec
        ),
    }
    _write_json_exclusive(context.unlock_path, unlock)
    state_volume.commit()
    protocol._verify_unlock(context)
    return unlock


@app.function(
    image=repo_image,
    gpu=GPU_TYPE,
    cpu=4,
    memory=32768,
    max_containers=MAX_PARALLEL_FINAL_GPUS,
    timeout=4 * 60 * 60,
    startup_timeout=60 * 60,
    retries=0,
    single_use_containers=True,
    secrets=[huggingface_secret],
    volumes={REMOTE_STATE: state_volume},
)
def collect_final(claim_id: str, root_call_id: str) -> dict[str, Any]:
    state_volume.reload()
    _attempt_claim(claim_id)
    receipt = json.loads((REMOTE_STATE / "launch_receipt.json").read_text())
    if receipt.get("root_call_id") != root_call_id:
        raise ModalV10Error("final worker lacks durable root authority")
    protocol, context, _contract, _digest = _load_protocol_context()
    protocol._verify_unlock(context)
    intent = {
        "schema_version": 1,
        "protocol": CONTRACT_PROTOCOL,
        "claim_id": claim_id,
        "root_call_id": root_call_id,
        "label": "sealed_final",
        "gpu_type": GPU_TYPE,
        "parallelism": 1,
        "collection_order": list(FINAL_LABELS),
        "status": "written_before_serial_final_gpu_collection",
        "protected_final_outcomes_read": False,
    }
    _write_json_exclusive(_dispatch_path("sealed_final", "intent"), intent)
    state_volume.commit()
    from scripts import confirmatory_v10_final_runner as final_runner

    # The registered runner writes its one-shot claim before opening the sealed
    # manifest and evaluates all nine labels serially before semantic parsing.
    result = final_runner.run_final_collection(REMOTE_STATE)
    state_volume.commit()
    completion = {
        "schema_version": 1,
        "protocol": CONTRACT_PROTOCOL,
        "claim_id": claim_id,
        "intent_sha256": _sha256(_dispatch_path("sealed_final", "intent")),
        "acceptance_sha256": protocol.sha256_file(context.acceptance_path),
        "stage": result.get("stage"),
        "status": "registered_serial_final_runner_completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_exclusive(_dispatch_path("sealed_final", "completion"), completion)
    state_volume.commit()
    return {"completion": completion, "result": result}


def _run_parallel_training(
    claim_id: str, root_call_id: str, condition: str
) -> list[dict[str, Any]]:
    intents = _write_training_intents(claim_id, root_call_id, condition)
    calls = [train_one.spawn(intent) for intent in intents]
    results: list[dict[str, Any]] = []
    failures: list[str] = []
    for intent, call in zip(intents, calls):
        label = str(intent["label"])
        try:
            results.append(call.get())
        except BaseException as error:
            failures.append(f"{label}: {type(error).__name__}: {error}")
    if failures:
        raise ModalV10Error(
            "parallel training workers failed after every dispatch was drained: "
            + " | ".join(failures)
        )
    return results


@app.function(
    image=repo_image,
    cpu=2,
    memory=4096,
    timeout=8 * 60 * 60,
    startup_timeout=60 * 60,
    retries=0,
    volumes={REMOTE_STATE: state_volume},
)
def orchestrate(claim_id: str) -> dict[str, Any]:
    state_volume.reload()
    _attempt_claim(claim_id)
    receipt = _wait_for_root_receipt(claim_id)
    root_call_id = modal.current_function_call_id()
    if receipt.get("root_call_id") != root_call_id:
        raise ModalV10Error("V10 orchestrator lacks durable root-call authority")
    status_path = REMOTE_STATE / "attempt_status.json"
    try:
        _replace_status(status_path, {"stage": "four_treatments", "claim_id": claim_id})
        state_volume.commit()
        treatments = _run_parallel_training(claim_id, root_call_id, "jlens")
        state_volume.reload()
        protocol, context, _contract, _digest = _load_protocol_context()
        gate = _build_and_verify_curve(protocol, context, write=True)
        if gate["passed"] is not True:
            terminal = {
                "stage": "curve_failed_no_controls_or_final",
                "claim_id": claim_id,
                "treatments": treatments,
                "curve": gate,
                "retry_or_resume_permitted": False,
            }
            _replace_status(status_path, terminal)
            state_volume.commit()
            return terminal
        _replace_status(status_path, {"stage": "four_signflips", "claim_id": claim_id})
        state_volume.commit()
        controls = _run_parallel_training(claim_id, root_call_id, "signflip")
        state_volume.reload()
        protocol, context, _contract, _digest = _load_protocol_context()
        unlock = _write_and_verify_unlock(protocol, context)
        authorization = {
            "schema_version": 1,
            "protocol": CONTRACT_PROTOCOL,
            "claim_id": claim_id,
            "root_call_id": root_call_id,
            "curve_gate_sha256": protocol.sha256_file(context.curve_path),
            "completed_runs_sha256": protocol.sha256_file(context.completed_runs_path),
            "unlock_sha256": protocol.sha256_file(context.unlock_path),
            "final_manifest_sha256": FINAL_MANIFEST_SHA256,
            "curve_passed": True,
            "all_eight_runs_verified": True,
            "unlock_verified": True,
            "protected_final_accessed": False,
            "authorized_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json_exclusive(
            REMOTE_STATE / "protected_final_upload_authorized.json", authorization
        )
        _replace_status(
            status_path,
            {"stage": "awaiting_post_unlock_protected_upload", "claim_id": claim_id},
        )
        state_volume.commit()
        protected_upload = _wait_for_protected_upload(claim_id, root_call_id)
        _replace_status(status_path, {"stage": "serial_nine_label_final", "claim_id": claim_id})
        state_volume.commit()
        final = collect_final.remote(claim_id, root_call_id)
        stage = final["result"].get("stage")
        result = {
            "stage": stage,
            "claim_id": claim_id,
            "treatments": treatments,
            "curve": gate,
            "controls": controls,
            "unlock": unlock,
            "protected_final_upload": protected_upload,
            "final": final,
        }
        _replace_status(status_path, result)
        state_volume.commit()
        return result
    except BaseException as error:
        try:
            state_volume.reload()
            _replace_status(
                status_path,
                {
                    "stage": "failed_closed",
                    "claim_id": claim_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "retry_or_resume_permitted": False,
                },
            )
            state_volume.commit()
        except BaseException:
            pass
        raise


def _prepared_state_inventory(contract: Mapping[str, Any]) -> tuple[Path, dict[str, str]]:
    root = (LOCAL_REPO / contract["prepared_state"]["local_path"]).resolve()
    try:
        root.relative_to(LOCAL_REPO)
    except ValueError as error:
        raise ModalV10Error("prepared state escapes the local repository") from error
    if not root.is_dir() or root.is_symlink():
        raise ModalV10Error("prepared V10 state directory is absent or unsafe")
    observed: dict[str, str] = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ModalV10Error(f"prepared V10 state contains a symlink: {path}")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            # Opaque hashing is permitted; this function never JSON-parses the
            # protected final manifest or any outcome-bearing file.
            observed[relative] = _sha256(path)
    expected = contract["prepared_state"]["expected_files"]
    if set(observed) != set(expected):
        raise ModalV10Error(
            f"prepared V10 state inventory changed: missing={sorted(set(expected)-set(observed))}, "
            f"unexpected={sorted(set(observed)-set(expected))}"
        )
    forbidden_top_level = {"runs", "evals", "evidence", "exports"}
    if any(Path(name).parts[0] in forbidden_top_level for name in observed):
        raise ModalV10Error("prepared state already contains outcome-bearing outputs")
    return root, observed


def _local_preflight() -> tuple[dict[str, Any], Path]:
    if CONTRACT is None or LOCAL_CONTRACT_PATH is None or CONTRACT_SHA256 is None:
        raise ModalV10Error(
            f"launcher is inert; set {CONTRACT_ENV} to the registered contract"
        )
    validate_contract_shape(CONTRACT)
    spec_path = (
        LOCAL_REPO
        / CONTRACT["prepared_state"]["local_path"]
        / "reproducibility"
        / "final_protocol_spec.json"
    )
    spec = json.loads(spec_path.read_text())
    from scripts import confirmatory_v10_final_protocol as protocol

    protocol.validate_spec(spec)
    if "modal_execution" not in getattr(protocol, "REGISTERED_SPEC_FIELDS", ()):
        raise ModalV10Error("registered V10 projection does not bind Modal execution")
    validate_scientific_binding(CONTRACT, spec, CONTRACT_SHA256)
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
        raise ModalV10Error("V10 Modal launch requires an exact clean pushed main")
    root, inventory = _prepared_state_inventory(CONTRACT)
    modal_cli = Path(sys.executable).with_name("modal")
    listing_text = subprocess.check_output(
        [str(modal_cli), "app", "list", "--json"], text=True
    )
    listing = json.loads(listing_text[listing_text.index("[") :])
    active_other = [
        item
        for item in listing
        if item.get("stopped_at") is None
        and item.get("state") != "stopped"
        and item.get("app_id") != app.app_id
    ]
    if active_other:
        raise ModalV10Error(
            "V10 reserves four of the five allowed GPUs and requires all other "
            f"Modal apps stopped: {active_other}"
        )
    state_volume.hydrate()
    if state_volume.object_id != CONTRACT["modal"]["state_volume_object_id"]:
        raise ModalV10Error("Modal Volume object ID differs from registration")
    inventory_text = subprocess.check_output(
        [str(modal_cli), "volume", "ls", VOLUME_NAME, "/", "--json"], text=True
    )
    remote_inventory = json.loads(inventory_text[inventory_text.index("[") :])
    if remote_inventory:
        raise ModalV10Error("V10 requires its registered Volume to be fresh and empty")
    return (
        {
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_head": head,
            "git_origin_main": pushed,
            "contract_sha256": CONTRACT_SHA256,
            "prepared_state_tree_sha256": _canonical_sha256(inventory),
            "prepared_state_file_sha256": inventory,
            "protected_final_manifest_inspected": False,
            "active_other_modal_apps": [],
            "state_volume_name": VOLUME_NAME,
            "state_volume_object_id": state_volume.object_id,
            "state_volume_version": 2,
            "gpu_type": GPU_TYPE,
            "max_parallel_gpus": 4,
        },
        root,
    )


def _upload_prepared_state(root: Path) -> None:
    assert CONTRACT is not None
    expected = CONTRACT["prepared_state"]["expected_files"]
    with state_volume.batch_upload(force=False) as batch:
        for relative in expected:
            batch.put_file(root / relative, f"/{relative}")


def _upload_protected_final_after_unlock() -> None:
    """Opaque local upload; caller must first obtain durable remote authorization."""
    assert CONTRACT is not None
    path = (LOCAL_REPO / CONTRACT["protected_final"]["local_path"]).resolve()
    try:
        path.relative_to(LOCAL_REPO)
    except ValueError as error:
        raise ModalV10Error("protected final path escapes the local repository") from error
    if not path.is_file() or path.is_symlink() or _sha256(path) != FINAL_MANIFEST_SHA256:
        raise ModalV10Error("authorized protected final has the wrong opaque identity")
    with state_volume.batch_upload(force=False) as batch:
        batch.put_file(path, "/manifests/sealed_final_indices.json")


@app.local_entrypoint()
def main() -> None:
    preflight, state_root = _local_preflight()
    _upload_prepared_state(state_root)
    claim_id = uuid.uuid4().hex
    claim_attempt.remote(claim_id, preflight)
    call = orchestrate.spawn(claim_id)
    receipt = record_launch_receipt.remote(claim_id, call.object_id)
    release = await_protected_final_release.remote(claim_id, call.object_id)
    protected_receipt = None
    if release.get("authorized") is True:
        _upload_protected_final_after_unlock()
        protected_receipt = record_protected_final_upload.remote(
            claim_id, call.object_id
        )
    result = call.get()
    print(
        json.dumps(
            {
                "status": "submitted",
                "claim_id": claim_id,
                "root_call_id": call.object_id,
                "app_id": app.app_id,
                "contract_sha256": CONTRACT_SHA256,
                "volume": VOLUME_NAME,
                "gpu_type": GPU_TYPE,
                "max_parallel_gpus": 4,
                "execution_plan": execution_plan(),
                "runtime_estimate_minutes_after_image_ready": {
                    "four_treatments": [8, 18],
                    "four_signflips_if_gate_passes": [8, 18],
                    "serial_nine_label_final_and_analysis": [35, 65],
                    "total": [51, 101],
                },
                "preflight": preflight,
                "launch_receipt": receipt,
                "protected_final_release": release,
                "protected_final_upload_receipt": protected_receipt,
                "result": result,
            },
            indent=2,
            sort_keys=True,
        )
    )
