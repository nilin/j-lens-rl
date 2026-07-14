"""Prepare, verify, and archive the post-V7 emotional-word tournament.

This protocol is deliberately development-only.  It sees only the already
exposed 400-row curve, never mounts a sealed/final/reserve/correlation payload,
and cannot launch until a later committed amendment pins V7's terminal
closeout.  The three outcome-bearing jobs are single-word J-lens-reward runs;
GSM8K correctness is used only for observational validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / ".confirmatory" / "v8"
CONFIG_DIR = STATE_DIR / "configs"
ARTIFACT_DIR = STATE_DIR / "frozen_artifacts"
MANIFEST_DIR = STATE_DIR / "manifests"
REPRO_DIR = STATE_DIR / "reproducibility"
EVIDENCE_DIR = STATE_DIR / "evidence"

PROTOCOL = "j-lens-rl-development-emotional-tournament-v1"
ARM_ORDER = ("fuck", "yay", "worried")
SEED = 192
CURVE_STEPS = (0, 5, 10, 15)
GPU_TYPE = "L40S"
MAX_GPU_CONTAINERS = 1
VOLUME_NAME = "j-lens-rl-development-emotional-tournament-u5-h15-20260714b"
APP_NAME = "j-lens-rl-development-emotional-tournament-u5-h15-v1"
GPU_LEASE_DICT_NAME = "j-lens-rl-global-gpu-lease-v1"
GPU_LEASE_KEY = "global-one-gpu"
GPU_LEASE_ENVIRONMENT = "main"
V7_CLAIM_ID = "1f2756de5df846d48a30f19a307b70fb"
V7_APP_ID = "ap-Vmg0kpbszpiUHHrNYcVWbd"
TERMINAL_V7_STAGES = {
    "complete",
    "curve_failed",
    "significance_failed",
    "infrastructure_failed",
    "failed_before_final",
    "failed",
}

MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
LENS_SHA256 = "178a9671cbf41882135807bde59b828e36c6f8f98b32c809ea3346860aad10dc"
CURVE_SHA256 = "ad348fe17d2e6bd6aac691d9bcdbb9da481f675305fa0e05c68e86dad97451c1"
TRAIN_EXCLUSIONS_SHA256 = (
    "7c1ca4f404ba9149093cc3c57dc3607582f671397e2fe99e93449848c1d65d61"
)
CALIBRATION_SHA256 = {
    "fuck": "f53ab990d2061f34ccf62f0bcafdc83304aab3747b3d189d279528125f67dc8d",
    "yay": "1fa6a65a94c7b682adca7029847d38257958e442034269a1cbe5524cffc66da2",
    "worried": "69cb69a85e455668a4948c62d2bc80271cde247d70d9479ad3e8e61a6105b65c",
}
TOKEN_IDS = {
    "fuck": [7820, 25090, 70474, 75021, 76374],
    "yay": [97559, 138496],
    "worried": [17811],
}
WEIGHTS = {"fuck": -1.0, "yay": 1.0, "worried": -1.0}

TEMPLATE_COMMON = ROOT / "configs" / "emotional_tournament_v1_common.json"
TEMPLATE_PATHS = {
    arm: ROOT / "configs" / f"emotional_tournament_v1_{arm}.json"
    for arm in ARM_ORDER
}
LEGACY_TEMPLATE_SHA256 = {
    "common": "6fa88bdef0607a84c71a64df446a7f56302c38a04d0c811743769053fec6efc2",
    "fuck": "81f6e988c3e449600fc61619b106e1f28e24906aecbef0ef9c9723f7e30ded88",
    "yay": "c6bbc69b7f2b4fbcb4ed4225e1fa4c838eeb7a403f415eda8a9b03f9e65f6fdf",
    "worried": "f4d7650c29d4e933bcc250b40350a051a512161e465ba024f22cfd873e70baa2",
}
LEGACY_PATHS = {
    "common": ROOT / "configs" / "single_word_screen_common.json",
    **{
        arm: ROOT / "configs" / f"single_word_screen_{arm}.json"
        for arm in ARM_ORDER
    },
}
CALIBRATION_SOURCE_DIR = (
    ROOT
    / "protocol_archive"
    / "emotional_screen_forensic_bundle"
    / "single_word"
    / "artifacts"
)
CALIBRATION_ATTEMPT_MANIFEST = (
    ROOT
    / "protocol_archive"
    / "emotional_screen_forensic_bundle"
    / "single_word"
    / "attempt_manifest.json"
)
CALIBRATION_ATTEMPT_MANIFEST_SHA256 = (
    "8b4b4969eaf4c1ed83d2756a30bc73093b0739412f2cf4705d90c9b4adde4143"
)
LENS_SOURCE = ROOT / "artifacts" / "qwen25_05b_solved_lens.pt"
MANIFEST_SOURCES = {
    "curve_indices.json": ROOT / ".confirmatory" / "manifests" / "curve_indices.json",
    "train_exclusions.json": ROOT
    / ".confirmatory"
    / "manifests"
    / "train_exclusions.json",
}
MANIFEST_SHA256 = {
    "curve_indices.json": CURVE_SHA256,
    "train_exclusions.json": TRAIN_EXCLUSIONS_SHA256,
}

RECIPE_LOCK_SOURCE = (
    ROOT / "protocol_archive" / "emotional_tournament_v1_recipe_lock.json"
)
REGISTRATION_DRAFT_SOURCE = (
    ROOT / "protocol_archive" / "emotional_tournament_v1_registration_draft.json"
)
AMENDMENT_SOURCE = (
    ROOT / "protocol_archive" / "emotional_tournament_v1_prelaunch_amendment.json"
)
AMENDMENT_TEMPLATE_SOURCE = (
    ROOT
    / "protocol_archive"
    / "emotional_tournament_v1_prelaunch_amendment_TEMPLATE.json"
)
INFRASTRUCTURE_AMENDMENT_SOURCE = (
    ROOT
    / "protocol_archive"
    / "emotional_tournament_v1_infrastructure_amendment1.json"
)
RUNTIME_ALLOWLIST_SOURCE = (
    ROOT / "scripts" / "emotional_tournament_v1_runtime_source_allowlist.json"
)

METRIC_SCHEMA = {
    "schema_version": 1,
    "protocol": PROTOCOL,
    "validation/exact_match": (
        "Greedy GSM8K numeric exact-match mean on the fixed exposed 400-row curve; "
        "validation is observational and never a training reward."
    ),
    "validation/exact_match_ci_low_high": "Wilson 95% binomial interval.",
    "validation/literal_target_completion_rate": (
        "Fraction of decoded validation completions containing the arm target as a "
        "case-insensitive whole word."
    ),
    "reward": (
        "Mean sole intrinsic J-lens reward; no correctness reward is configured."
    ),
    "jlens_label_mean": (
        "Weighted, clipped standardized J-lens target-word log-probability mass."
    ),
    "kl": "Configured reference-model KL diagnostic/regularizer; beta is 0.02.",
    "selection": (
        "Development ranking only: shape-pass, step15-minus-step0, step15, then "
        "fixed arm order. It is not a significance test."
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_json_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def _load_config(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    base = payload.pop("base", None)
    if base is None:
        return payload
    parent = _load_config(path.parent / base)
    parent.update(payload)
    return parent


def _all_values(value: Any, key: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, dict):
        for child_key, child in value.items():
            if child_key == key:
                found.append(child)
            found.extend(_all_values(child, key))
    elif isinstance(value, list):
        for child in value:
            found.extend(_all_values(child, key))
    return found


def _require_clean_pushed_main() -> dict[str, str]:
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise RuntimeError(f"tournament preparation requires a clean tree:\n{status}")
    head = _git("rev-parse", "HEAD")
    origin = _git("rev-parse", "origin/main")
    if head != origin:
        raise RuntimeError("tournament preparation requires HEAD pushed to origin/main")
    return {
        "git_commit": head,
        "git_tree": _git("rev-parse", "HEAD^{tree}"),
        "git_status": status,
    }


def _validate_recipe_lock(recipe: Any) -> dict[str, Any]:
    if not isinstance(recipe, dict) or recipe.get("protocol") != PROTOCOL:
        raise RuntimeError("development recipe lock is malformed")
    if recipe.get("scientific_status") != "development_only_on_exposed_curve":
        raise RuntimeError("recipe lock does not prohibit confirmatory interpretation")
    arms = recipe.get("arms_in_fixed_serial_order")
    if (
        not isinstance(arms, list)
        or [item.get("arm") for item in arms if isinstance(item, dict)]
        != list(ARM_ORDER)
    ):
        raise RuntimeError("recipe arm order changed")
    for item in arms:
        arm = item["arm"]
        if (
            item.get("target_words") != [arm]
            or item.get("weight") != WEIGHTS[arm]
            or item.get("calibration_sha256") != CALIBRATION_SHA256[arm]
            or item.get("token_ids") != TOKEN_IDS[arm]
        ):
            raise RuntimeError(f"recipe for {arm} changed")
    if recipe.get("shared_seed") != SEED:
        raise RuntimeError("development seed changed")
    curve = recipe.get("development_curve", {})
    if (
        curve.get("nodes") != list(CURVE_STEPS)
        or curve.get("manifest_sha256") != CURVE_SHA256
        or curve.get("train_exclusions_sha256") != TRAIN_EXCLUSIONS_SHA256
    ):
        raise RuntimeError("development curve identity changed")
    training = recipe.get("training", {})
    expected = {
        "updates": 15,
        "learning_rate": 3e-6,
        "lr_scheduler_type": "constant",
        "warmup_steps": 0,
        "kl_beta": 0.02,
        "loss_type": "dapo",
        "scale_rewards": "group",
        "reward_type": "jlens",
        "reward_functions": ["intrinsic_j_lens_target_word_reward"],
        "train_examples": 1000,
        "num_generations": 8,
        "min_new_tokens": 64,
        "max_new_tokens": 256,
        "mask_target_tokens": True,
        "score_stride": 5,
    }
    if any(training.get(key) != value for key, value in expected.items()):
        raise RuntimeError("training recipe changed")
    ranking = recipe.get("ranking", {})
    if ranking.get("keys") != [
        "shape_pass_boolean",
        "accuracy_step15_minus_step0",
        "accuracy_step15",
        "negative_fixed_arm_order_index",
    ] or ranking.get("always_complete_all_three_arms") is not True:
        raise RuntimeError("development ranking changed")
    forbidden = set(recipe.get("forbidden_inputs_and_claims", []))
    required_forbidden = {
        "gsm8k_correctness_training_reward",
        "sealed_final_indices_or_outputs",
        "future_or_reserve_indices_or_outputs",
        "unopened_word_correlation_payloads",
        "confirmatory_significance_claim",
        "final_evidence_claim",
    }
    if forbidden != required_forbidden:
        raise RuntimeError("forbidden-input/claim boundary changed")
    return recipe


def _validate_calibration(arm: str, path: Path) -> dict[str, Any]:
    if sha256_file(path) != CALIBRATION_SHA256[arm]:
        raise RuntimeError(f"{arm} calibration bytes changed")
    payload = _read_json(path)
    expected = {
        "target_words": [arm],
        "token_ids": TOKEN_IDS[arm],
        "layers": [8, 14, 20],
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_revision": MODEL_REVISION,
        "adapter": None,
        "corpus": "wikitext",
        "dataset": "Salesforce/wikitext",
        "dataset_revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
        "lens_sha256": LENS_SHA256,
        "lens_input_sha256": LENS_SHA256,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"{arm} calibration provenance changed")
    if not isinstance(payload.get("mean"), (int, float)) or not isinstance(
        payload.get("std"), (int, float)
    ) or payload["std"] <= 0:
        raise RuntimeError(f"{arm} calibration statistics are invalid")
    return payload


def _validate_source_inputs() -> dict[str, Any]:
    if sha256_file(LENS_SOURCE) != LENS_SHA256:
        raise RuntimeError("target-independent lens changed")
    actual_manifests = {
        name: sha256_file(path) for name, path in MANIFEST_SOURCES.items()
    }
    if actual_manifests != MANIFEST_SHA256:
        raise RuntimeError(f"exposed manifest identity changed: {actual_manifests}")
    if sha256_file(CALIBRATION_ATTEMPT_MANIFEST) != CALIBRATION_ATTEMPT_MANIFEST_SHA256:
        raise RuntimeError("calibration source attempt manifest changed")
    actual_legacy = {name: sha256_file(path) for name, path in LEGACY_PATHS.items()}
    if actual_legacy != LEGACY_TEMPLATE_SHA256:
        raise RuntimeError("legacy source templates changed")
    calibration = {}
    for arm in ARM_ORDER:
        calibration[arm] = _validate_calibration(
            arm, CALIBRATION_SOURCE_DIR / f"{arm}_calibration.json"
        )
    return {
        "lens_sha256": LENS_SHA256,
        "manifest_sha256": actual_manifests,
        "calibration_sha256": dict(CALIBRATION_SHA256),
        "calibration_source_attempt_manifest_sha256": (
            CALIBRATION_ATTEMPT_MANIFEST_SHA256
        ),
        "legacy_template_sha256": actual_legacy,
        "calibration_metadata": calibration,
    }


def _validate_template(arm: str, config: dict[str, Any]) -> None:
    expected = {
        "model_revision": MODEL_REVISION,
        "dataset_revision": DATASET_REVISION,
        "lens_sha256": LENS_SHA256,
        "expected_lens_sha256": LENS_SHA256,
        "target_words": [arm],
        "calibration_sha256": CALIBRATION_SHA256[arm],
        "expected_calibration_sha256": CALIBRATION_SHA256[arm],
        "seed": SEED,
        "train_examples": 1000,
        "validation_examples": 400,
        "validation_batch_size": 64,
        "updates": 15,
        "num_generations": 8,
        "num_generations_eval": 1,
        "min_new_tokens": 64,
        "max_new_tokens": 256,
        "learning_rate": 3e-6,
        "kl_beta": 0.02,
        "loss_type": "dapo",
        "scale_rewards": "group",
        "score_stride": 5,
        "mask_target_tokens": True,
        "eval_every": 5,
        "validation_steps": [5, 10, 15],
        "validation_observational_only": True,
        "early_stopping_patience": None,
        "save_every": 5,
        "save_total_limit": 3,
        "reward_type": "jlens",
        "wandb_entity": "nilinabra-spare-time",
        "wandb_project": "j-lens-rl",
        "wandb_group": "dev-v8-emotional-single-u5-h15-seed192",
        "wandb_mode": "online",
        "wandb_resume": "never",
        "evidence_eligibility": (
            "development_only_exposed_curve_no_significance_claim"
        ),
    }
    changed = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if changed:
        raise RuntimeError(f"{arm} template changed: {changed}")
    if config.get("score_components") != [
        {
            "layer": 8,
            "start_fraction": 0.5,
            "end_fraction": 1.0,
            "aggregation": "mean",
            "weight": WEIGHTS[arm],
        }
    ]:
        raise RuntimeError(f"{arm} score component changed")
    expected_stem = f"dev-v8-emotional-single-u5-h15-{arm}-seed192"
    if (
        config.get("run_name") != expected_stem
        or config.get("wandb_run_id") != expected_stem
        or config.get("wandb_url")
        != f"https://wandb.ai/nilinabra-spare-time/j-lens-rl/runs/{expected_stem}"
        or config.get("output_dir") != f".confirmatory/v8/runs/{arm}_seed192"
        or config.get("calibration_path")
        != f".confirmatory/v8/frozen_artifacts/{arm}_calibration.json"
        or config.get("validation_indices_path")
        != ".confirmatory/v8/manifests/curve_indices.json"
        or config.get("reserved_train_indices_path")
        != ".confirmatory/v8/manifests/train_exclusions.json"
    ):
        raise RuntimeError(f"{arm} paths or tracking identity changed")


def _validate_templates() -> dict[str, str]:
    result = {"common": sha256_file(TEMPLATE_COMMON)}
    for arm, path in TEMPLATE_PATHS.items():
        result[arm] = sha256_file(path)
        _validate_template(arm, _load_config(path))
    return result


def _runtime_allowlist() -> list[str]:
    payload = _read_json(RUNTIME_ALLOWLIST_SOURCE)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("protocol")
        != "j-lens-rl-development-emotional-tournament-runtime-allowlist-v1"
    ):
        raise RuntimeError("tournament runtime allowlist is malformed")
    base_path = ROOT / payload.get("base_allowlist", "")
    base = _read_json(base_path)
    names = set(base.get("files", []))
    remove = payload.get("remove", [])
    add = payload.get("add", [])
    if (
        not isinstance(remove, list)
        or not isinstance(add, list)
        or len(remove) != len(set(remove))
        or len(add) != len(set(add))
        or not set(remove) <= names
    ):
        raise RuntimeError("tournament runtime allowlist delta is invalid")
    names.difference_update(remove)
    names.update(add)
    materialized = sorted(names)
    required_config_dependencies: set[str] = set()
    pending = [TEMPLATE_COMMON, *TEMPLATE_PATHS.values()]
    while pending:
        config_path = pending.pop()
        relative_config = config_path.relative_to(ROOT).as_posix()
        if relative_config in required_config_dependencies:
            continue
        required_config_dependencies.add(relative_config)
        config_payload = _read_json(config_path)
        base = config_payload.get("base")
        if base is not None:
            if (
                not isinstance(base, str)
                or not base
                or Path(base).is_absolute()
                or ".." in Path(base).parts
            ):
                raise RuntimeError(f"unsafe inherited config path: {base!r}")
            pending.append(config_path.parent / base)
    missing_dependencies = sorted(required_config_dependencies - names)
    if missing_dependencies:
        raise RuntimeError(
            "tournament runtime allowlist omits inherited config dependencies: "
            f"{missing_dependencies}"
        )
    for name in materialized:
        path = ROOT / name
        relative = Path(name)
        if (
            not isinstance(name, str)
            or not name
            or relative.is_absolute()
            or ".." in relative.parts
            or any(part.startswith(".") for part in relative.parts)
            or not path.is_file()
            or path.is_symlink()
            or "protocol_archive" in relative.parts
            or ".confirmatory" in relative.parts
        ):
            raise RuntimeError(f"unsafe tournament runtime source: {name!r}")
    return materialized


def runtime_source_hashes() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in _runtime_allowlist()}


def _synthetic_git_identity(names: list[str]) -> tuple[str, str]:
    environment = {
        **os.environ,
        "GIT_AUTHOR_NAME": "J-Lens Tournament Runtime",
        "GIT_AUTHOR_EMAIL": "runtime@example.invalid",
        "GIT_COMMITTER_NAME": "J-Lens Tournament Runtime",
        "GIT_COMMITTER_EMAIL": "runtime@example.invalid",
        "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+00:00",
        "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+00:00",
    }
    with tempfile.TemporaryDirectory(prefix="jlens-tournament-source-") as raw:
        work = Path(raw)
        subprocess.run(["git", "init", "-q"], cwd=work, check=True, env=environment)
        for name in names:
            destination = work / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, destination)
        subprocess.run(
            ["git", "add", "--", *names], cwd=work, check=True, env=environment
        )
        tree = subprocess.check_output(
            ["git", "write-tree"], cwd=work, text=True, env=environment
        ).strip()
        commit = subprocess.check_output(
            ["git", "commit-tree", tree],
            cwd=work,
            input="J-Lens emotional tournament strict runtime source\n",
            text=True,
            env=environment,
        ).strip()
    return tree, commit


def _write_source_snapshot(path: Path, names: list[str]) -> None:
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            info = zipfile.ZipInfo(name, date_time=(2000, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            mode = stat.S_IMODE((ROOT / name).stat().st_mode)
            info.external_attr = mode << 16
            archive.writestr(info, (ROOT / name).read_bytes())


def _make_source_manifest(path: Path) -> dict[str, Any]:
    names = _runtime_allowlist()
    tree, commit = _synthetic_git_identity(names)
    files = {
        name: {
            "sha256": sha256_file(ROOT / name),
            "size_bytes": (ROOT / name).stat().st_size,
            "mode": stat.S_IMODE((ROOT / name).stat().st_mode),
        }
        for name in names
    }
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "files": files,
        "file_count": len(files),
        "source_tree_sha256": canonical_sha256(
            {name: value["sha256"] for name, value in files.items()}
        ),
        "synthetic_git_tree": tree,
        "git_commit": commit,
        "runtime_commit_recipe": {
            "author": "J-Lens Tournament Runtime <runtime@example.invalid>",
            "timestamp": "2000-01-01T00:00:00+00:00",
            "message": "J-Lens emotional tournament strict runtime source",
            "parent": None,
        },
    }
    _write_json(path, payload)
    return payload


def _amendment_and_closeout() -> tuple[dict[str, Any], Path | None]:
    if AMENDMENT_SOURCE.is_file():
        amendment = _read_json(AMENDMENT_SOURCE)
        closeout_path = _validate_amendment(amendment)
        return amendment, closeout_path
    amendment = _read_json(AMENDMENT_TEMPLATE_SOURCE)
    if amendment.get("launch_enabled") is not False:
        raise RuntimeError("unbound amendment template must disable launch")
    _validate_amendment_shape(amendment)
    return amendment, None


def _validate_infrastructure_amendment(
    value: Any, *, copied_closeout: Path | None = None
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError("infrastructure amendment is malformed")
    closeout_relative = value.get("preclaim_closeout_path")
    closeout_sha256 = value.get("preclaim_closeout_sha256")
    if (
        value.get("schema_version") != 1
        or value.get("document_type")
        != "j-lens-rl-development-emotional-tournament-infrastructure-amendment"
        or value.get("protocol") != PROTOCOL
        or value.get("amendment_number") != 1
        or value.get("scientific_recipe_changed") is not False
        or value.get("outcome_data_observed_before_amendment") is not False
        or value.get("superseded_volume")
        != "j-lens-rl-development-emotional-tournament-u5-h15-20260714a"
        or value.get("replacement_volume") != VOLUME_NAME
        or value.get("replacement_volume_version") != 2
        or value.get("added_runtime_source") != "configs/common.json"
        or value.get("added_runtime_source_sha256")
        != sha256_file(ROOT / "configs/common.json")
        or closeout_relative
        != "protocol_archive/emotional_tournament_v1_preclaim_attempt_a_closeout.json"
        or not isinstance(closeout_sha256, str)
        or len(closeout_sha256) != 64
    ):
        raise RuntimeError("infrastructure amendment changed or is incomplete")
    closeout_path = copied_closeout or (ROOT / closeout_relative)
    if not closeout_path.is_file() or sha256_file(closeout_path) != closeout_sha256:
        raise RuntimeError("infrastructure amendment preclaim closeout is missing or changed")
    closeout = _read_json(closeout_path)
    if (
        closeout.get("scientific_status")
        != "infrastructure_failure_before_claim_no_outcome_data"
        or closeout.get("outcome_boundary", {}).get("attempt_claim_written") is not False
        or closeout.get("outcome_boundary", {}).get("gpu_training_dispatched") is not False
        or closeout.get("outcome_boundary", {}).get("curve_observed") is not False
    ):
        raise RuntimeError("infrastructure amendment is not supported by a pre-outcome closeout")
    return value


def _validate_amendment_shape(amendment: Any) -> dict[str, Any]:
    if (
        not isinstance(amendment, dict)
        or amendment.get("schema_version") != 1
        or amendment.get("document_type")
        != "j-lens-rl-development-emotional-tournament-prelaunch-amendment"
        or amendment.get("protocol") != PROTOCOL
        or amendment.get("v7_claim_id") != V7_CLAIM_ID
        or amendment.get("v7_app_id") != V7_APP_ID
        or amendment.get("require_v7_app_stopped_at_launch") is not True
        or amendment.get("require_shared_gpu_lease_free_at_launch") is not True
    ):
        raise RuntimeError("prelaunch amendment is malformed")
    return amendment


def _validate_amendment(
    amendment: Any, *, copied_closeout: Path | None = None
) -> Path:
    amendment = _validate_amendment_shape(amendment)
    if amendment.get("launch_enabled") is not True:
        raise RuntimeError("prelaunch amendment has not enabled launch")
    relative = amendment.get("v7_terminal_closeout_path")
    digest = amendment.get("v7_terminal_closeout_sha256")
    stage = amendment.get("v7_terminal_stage")
    if (
        not isinstance(relative, str)
        or not relative.startswith("protocol_archive/")
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or not isinstance(digest, str)
        or len(digest) != 64
        or stage not in TERMINAL_V7_STAGES
    ):
        raise RuntimeError("prelaunch amendment lacks a safe terminal V7 identity")
    path = copied_closeout or (ROOT / relative)
    if not path.is_file() or sha256_file(path) != digest:
        raise RuntimeError("pinned V7 terminal closeout is missing or changed")
    closeout = _read_json(path)
    if V7_CLAIM_ID not in _all_values(closeout, "claim_id"):
        raise RuntimeError("V7 closeout is not bound to the active claim")
    app_values = _all_values(closeout, "app_id") + _all_values(closeout, "modal_app_id")
    if V7_APP_ID not in app_values:
        raise RuntimeError("V7 closeout is not bound to the active app")
    observed_stages = set(
        _all_values(closeout, "stage")
        + _all_values(closeout, "attempt_stage")
        + _all_values(closeout, "terminal_stage")
    )
    if stage not in observed_stages:
        raise RuntimeError("V7 closeout does not contain the amended terminal stage")
    return path


def _materialize_config(
    arm: str,
    registration_sha256: str,
    recipe_file_sha256: str,
    recipe_canonical_sha256: str,
    metric_schema_sha256: str,
    registered_code_sha256: dict[str, Any],
    amendment_sha256: str,
) -> dict[str, Any]:
    config = _load_config(TEMPLATE_PATHS[arm])
    _validate_template(arm, config)
    config.update(
        {
            "registration_sha256": registration_sha256,
            "recipe_lock_sha256": recipe_file_sha256,
            "recipe_sha256": recipe_canonical_sha256,
            "curve_manifest_sha256": CURVE_SHA256,
            "train_exclusions_manifest_sha256": TRAIN_EXCLUSIONS_SHA256,
            "prelaunch_amendment_sha256": amendment_sha256,
            "metric_schema_path": ".confirmatory/v8/reproducibility/metric_schema.json",
            "metric_schema_sha256": metric_schema_sha256,
            "registered_code_sha256": registered_code_sha256,
            "registered_command": [
                "python",
                "-m",
                "jlens_rl.train",
                "--config",
                f".confirmatory/v8/configs/{arm}_seed192.json",
                "--wandb-mode",
                "online",
            ],
        }
    )
    return config


def prepare() -> dict[str, Any]:
    provenance = _require_clean_pushed_main()
    if STATE_DIR.exists():
        raise FileExistsError(
            "refusing to overwrite .confirmatory/v8; archive/remove only an "
            "unclaimed disabled preparation before retrying"
        )
    source_inputs = _validate_source_inputs()
    recipe = _validate_recipe_lock(_read_json(RECIPE_LOCK_SOURCE))
    templates = _validate_templates()
    draft = _read_json(REGISTRATION_DRAFT_SOURCE)
    if (
        draft.get("protocol") != PROTOCOL
        or draft.get("status")
        != "scientific_design_frozen_launch_disabled_pending_v7_terminal_amendment"
        or draft.get("launch_gate", {}).get("enabled") is not False
        or draft.get("execution", {}).get("fresh_volume")
        != "j-lens-rl-development-emotional-tournament-u5-h15-20260714a"
    ):
        raise RuntimeError("registration draft changed")
    if (
        draft.get("recipe_lock_sha256") != sha256_file(RECIPE_LOCK_SOURCE)
        or draft.get("fixed_public_input_sha256")
        != {
            "lens": LENS_SHA256,
            "curve_indices": CURVE_SHA256,
            "train_exclusions": TRAIN_EXCLUSIONS_SHA256,
            **{
                f"{arm}_calibration": CALIBRATION_SHA256[arm]
                for arm in ARM_ORDER
            },
        }
        or draft.get("source_provenance", {}).get(
            "calibration_source_attempt_manifest_sha256"
        )
        != CALIBRATION_ATTEMPT_MANIFEST_SHA256
        or draft.get("source_provenance", {}).get("legacy_template_sha256")
        != LEGACY_TEMPLATE_SHA256
        or draft.get("source_provenance", {}).get("new_template_sha256")
        != templates
    ):
        raise RuntimeError("registration draft source identities changed")
    amendment, closeout_source = _amendment_and_closeout()
    infrastructure_amendment = _validate_infrastructure_amendment(
        _read_json(INFRASTRUCTURE_AMENDMENT_SOURCE)
    )
    infrastructure_closeout_source = (
        ROOT / infrastructure_amendment["preclaim_closeout_path"]
    )
    enabled = amendment.get("launch_enabled") is True

    temporary = STATE_DIR.with_name(f".v8-preparing-{uuid.uuid4().hex}")
    temporary.mkdir(parents=True)
    try:
        config_dir = temporary / "configs"
        artifact_dir = temporary / "frozen_artifacts"
        manifest_dir = temporary / "manifests"
        repro_dir = temporary / "reproducibility"
        for directory in (config_dir, artifact_dir, manifest_dir, repro_dir):
            directory.mkdir(parents=True)
        shutil.copy2(LENS_SOURCE, artifact_dir / "lens.pt")
        for arm in ARM_ORDER:
            shutil.copy2(
                CALIBRATION_SOURCE_DIR / f"{arm}_calibration.json",
                artifact_dir / f"{arm}_calibration.json",
            )
        for name, source in MANIFEST_SOURCES.items():
            shutil.copy2(source, manifest_dir / name)
        shutil.copy2(RECIPE_LOCK_SOURCE, repro_dir / "recipe_lock.json")
        shutil.copy2(REGISTRATION_DRAFT_SOURCE, repro_dir / "registration_draft.json")
        selected_amendment_source = (
            AMENDMENT_SOURCE if AMENDMENT_SOURCE.is_file() else AMENDMENT_TEMPLATE_SOURCE
        )
        shutil.copy2(selected_amendment_source, repro_dir / "prelaunch_amendment.json")
        shutil.copy2(
            INFRASTRUCTURE_AMENDMENT_SOURCE,
            repro_dir / "infrastructure_amendment1.json",
        )
        shutil.copy2(
            infrastructure_closeout_source,
            repro_dir / "preclaim_attempt_a_closeout.json",
        )
        if closeout_source is not None:
            shutil.copy2(closeout_source, repro_dir / "v7_terminal_closeout.json")

        source_manifest = _make_source_manifest(repro_dir / "source_manifest.json")
        _write_source_snapshot(repro_dir / "source_snapshot.zip", list(source_manifest["files"]))
        _write_json(repro_dir / "metric_schema.json", METRIC_SCHEMA)

        recipe_file_sha = sha256_file(repro_dir / "recipe_lock.json")
        amendment_sha = sha256_file(repro_dir / "prelaunch_amendment.json")
        infrastructure_amendment_sha = sha256_file(
            repro_dir / "infrastructure_amendment1.json"
        )
        infrastructure_closeout_sha = sha256_file(
            repro_dir / "preclaim_attempt_a_closeout.json"
        )
        source_manifest_sha = sha256_file(repro_dir / "source_manifest.json")
        source_snapshot_sha = sha256_file(repro_dir / "source_snapshot.zip")
        metric_schema_sha = sha256_file(repro_dir / "metric_schema.json")
        registered_code = {
            "modal_app": APP_NAME,
            "volume": VOLUME_NAME,
            "volume_version": 2,
            "gpu_type": GPU_TYPE,
            "max_parallel_gpu_workers": MAX_GPU_CONTAINERS,
            "global_gpu_lease_dict": GPU_LEASE_DICT_NAME,
            "global_gpu_lease_key": GPU_LEASE_KEY,
            "runtime_source_sha256": {
                name: value["sha256"] for name, value in source_manifest["files"].items()
            },
            "runtime_source_tree_sha256": source_manifest["source_tree_sha256"],
            "synthetic_runtime_git_commit": source_manifest["git_commit"],
        }
        registration = {
            "schema_version": 1,
            "document_type": (
                "j-lens-rl-development-emotional-tournament-prepared-registration"
            ),
            "protocol": PROTOCOL,
            "status": "launch_enabled" if enabled else "launch_disabled_pending_v7_terminal",
            "scientific_status": "development_only_on_exposed_curve",
            "no_significance_or_final_claim": True,
            "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
            **provenance,
            "recipe_lock_sha256": recipe_file_sha,
            "recipe_canonical_sha256": canonical_sha256(recipe),
            "registration_draft_sha256": sha256_file(
                repro_dir / "registration_draft.json"
            ),
            "prelaunch_amendment_sha256": amendment_sha,
            "prelaunch_amendment": amendment,
            "infrastructure_amendment1_sha256": infrastructure_amendment_sha,
            "infrastructure_amendment1": infrastructure_amendment,
            "preclaim_attempt_a_closeout_sha256": infrastructure_closeout_sha,
            "v7_terminal_closeout_sha256": (
                sha256_file(repro_dir / "v7_terminal_closeout.json")
                if closeout_source is not None
                else None
            ),
            "template_sha256": templates,
            "source_inputs": source_inputs,
            "source_manifest_sha256": source_manifest_sha,
            "source_snapshot_sha256": source_snapshot_sha,
            "metric_schema_sha256": metric_schema_sha,
            "registered_code_sha256": registered_code,
            "arms_in_fixed_serial_order": list(ARM_ORDER),
            "shared_seed": SEED,
            "curve_steps": list(CURVE_STEPS),
            "ranking_keys": recipe["ranking"]["keys"],
            "always_complete_all_three_arms": True,
            "forbidden_payload_mounts": [
                "sealed_final",
                "future_reserve",
                "retired_curve",
                "word_correlation",
            ],
        }
        _write_json(repro_dir / "registration.json", registration)
        registration_sha = sha256_file(repro_dir / "registration.json")
        config_sha = {}
        for arm in ARM_ORDER:
            config = _materialize_config(
                arm,
                registration_sha,
                recipe_file_sha,
                canonical_sha256(recipe),
                metric_schema_sha,
                registered_code,
                amendment_sha,
            )
            path = config_dir / f"{arm}_seed192.json"
            _write_json(path, config)
            config_sha[arm] = sha256_file(path)
        launch_plan = {
            "protocol": PROTOCOL,
            "launch_enabled": enabled,
            "arm_order": list(ARM_ORDER),
            "commands": {
                "prepare": "./run_emotional_tournament_v1.sh prepare",
                "verify": "./run_emotional_tournament_v1.sh verify-launch",
                "modal": (
                    "JLENS_MODAL_GPU_EXCLUSIVE_CONFIRM="
                    "confirmed-no-other-modal-gpu-app-running "
                    "./run_emotional_tournament_v1.sh modal"
                ),
            },
            "volume": VOLUME_NAME,
            "volume_version": 2,
            "app": APP_NAME,
            "gpu_type": GPU_TYPE,
            "max_parallel_gpu_workers": 1,
            "wandb_group": "dev-v8-emotional-single-u5-h15-seed192",
            "wandb_run_ids": {
                arm: f"dev-v8-emotional-single-u5-h15-{arm}-seed192"
                for arm in ARM_ORDER
            },
        }
        _write_json(repro_dir / "launch_plan.json", launch_plan)
        protocol_state = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "stage": "prepared_launch_enabled" if enabled else "prepared_launch_disabled",
            "scientific_status": "development_only",
            "git_commit": provenance["git_commit"],
            "git_tree": provenance["git_tree"],
            "registration_sha256": registration_sha,
            "recipe_lock_sha256": recipe_file_sha,
            "prelaunch_amendment_sha256": amendment_sha,
            "infrastructure_amendment1_sha256": infrastructure_amendment_sha,
            "preclaim_attempt_a_closeout_sha256": infrastructure_closeout_sha,
            "source_manifest_sha256": source_manifest_sha,
            "source_snapshot_sha256": source_snapshot_sha,
            "metric_schema_sha256": metric_schema_sha,
            "config_sha256": config_sha,
            "arm_order": list(ARM_ORDER),
            "seed": SEED,
            "curve_steps": list(CURVE_STEPS),
            "volume": VOLUME_NAME,
            "volume_version": 2,
            "app": APP_NAME,
        }
        _write_json(temporary / "protocol_state.json", protocol_state)
        temporary.replace(STATE_DIR)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    verify(require_launch=enabled)
    return _read_json(STATE_DIR / "protocol_state.json")


def _prepared_closeout_path() -> Path | None:
    path = REPRO_DIR / "v7_terminal_closeout.json"
    return path if path.is_file() else None


def verify(*, require_launch: bool = False) -> dict[str, Any]:
    state = _read_json(STATE_DIR / "protocol_state.json")
    registration = _read_json(REPRO_DIR / "registration.json")
    recipe = _validate_recipe_lock(_read_json(REPRO_DIR / "recipe_lock.json"))
    amendment = _read_json(REPRO_DIR / "prelaunch_amendment.json")
    infrastructure_amendment = _validate_infrastructure_amendment(
        _read_json(REPRO_DIR / "infrastructure_amendment1.json"),
        copied_closeout=REPRO_DIR / "preclaim_attempt_a_closeout.json",
    )
    if (
        state.get("protocol") != PROTOCOL
        or registration.get("protocol") != PROTOCOL
        or state.get("registration_sha256")
        != sha256_file(REPRO_DIR / "registration.json")
        or state.get("recipe_lock_sha256") != sha256_file(REPRO_DIR / "recipe_lock.json")
        or state.get("prelaunch_amendment_sha256")
        != sha256_file(REPRO_DIR / "prelaunch_amendment.json")
        or state.get("infrastructure_amendment1_sha256")
        != sha256_file(REPRO_DIR / "infrastructure_amendment1.json")
        or state.get("preclaim_attempt_a_closeout_sha256")
        != sha256_file(REPRO_DIR / "preclaim_attempt_a_closeout.json")
        or state.get("source_manifest_sha256")
        != sha256_file(REPRO_DIR / "source_manifest.json")
        or state.get("source_snapshot_sha256")
        != sha256_file(REPRO_DIR / "source_snapshot.zip")
        or state.get("metric_schema_sha256")
        != sha256_file(REPRO_DIR / "metric_schema.json")
        or state.get("arm_order") != list(ARM_ORDER)
        or state.get("seed") != SEED
        or state.get("curve_steps") != list(CURVE_STEPS)
    ):
        raise RuntimeError("prepared tournament state identity changed")
    if registration.get("scientific_status") != "development_only_on_exposed_curve":
        raise RuntimeError("prepared registration changed scientific status")
    if (
        registration.get("infrastructure_amendment1_sha256")
        != state.get("infrastructure_amendment1_sha256")
        or registration.get("infrastructure_amendment1")
        != infrastructure_amendment
        or registration.get("preclaim_attempt_a_closeout_sha256")
        != state.get("preclaim_attempt_a_closeout_sha256")
    ):
        raise RuntimeError("prepared registration changed infrastructure amendment")
    source_manifest = _read_json(REPRO_DIR / "source_manifest.json")
    current_hashes = runtime_source_hashes()
    registered_hashes = {
        name: value.get("sha256")
        for name, value in source_manifest.get("files", {}).items()
    }
    if current_hashes != registered_hashes:
        raise RuntimeError("runtime source differs from prepared registration")
    if registration.get("registered_code_sha256", {}).get(
        "runtime_source_sha256"
    ) != current_hashes:
        raise RuntimeError("registration runtime source map changed")
    if _read_json(REPRO_DIR / "metric_schema.json") != METRIC_SCHEMA:
        raise RuntimeError("metric semantics changed")
    if sha256_file(ARTIFACT_DIR / "lens.pt") != LENS_SHA256:
        raise RuntimeError("prepared lens changed")
    for arm in ARM_ORDER:
        _validate_calibration(arm, ARTIFACT_DIR / f"{arm}_calibration.json")
    actual_manifest = {
        name: sha256_file(MANIFEST_DIR / name) for name in MANIFEST_SHA256
    }
    if actual_manifest != MANIFEST_SHA256:
        raise RuntimeError("prepared exposed manifests changed")
    amendment_enabled = amendment.get("launch_enabled") is True
    if amendment_enabled:
        closeout = _prepared_closeout_path()
        if closeout is None:
            raise RuntimeError("enabled preparation lacks copied V7 closeout")
        _validate_amendment(amendment, copied_closeout=closeout)
        if registration.get("v7_terminal_closeout_sha256") != sha256_file(closeout):
            raise RuntimeError("registration changed the V7 closeout identity")
    else:
        _validate_amendment_shape(amendment)
        if registration.get("v7_terminal_closeout_sha256") is not None:
            raise RuntimeError("disabled preparation unexpectedly binds a closeout")
    if require_launch and not amendment_enabled:
        raise RuntimeError(
            "tournament launch is disabled until a committed amendment pins V7 terminal"
        )
    if require_launch and state.get("stage") != "prepared_launch_enabled":
        raise RuntimeError("protocol state is not launch-enabled")
    expected_config_sha = {}
    for arm in ARM_ORDER:
        expected = _materialize_config(
            arm,
            state["registration_sha256"],
            state["recipe_lock_sha256"],
            canonical_sha256(recipe),
            state["metric_schema_sha256"],
            registration["registered_code_sha256"],
            state["prelaunch_amendment_sha256"],
        )
        path = CONFIG_DIR / f"{arm}_seed192.json"
        if _read_json(path) != expected:
            raise RuntimeError(f"prepared {arm} config changed")
        expected_config_sha[arm] = sha256_file(path)
    if state.get("config_sha256") != expected_config_sha:
        raise RuntimeError("prepared config inventory changed")
    return {
        "verified": True,
        "launch_enabled": amendment_enabled,
        "scientific_status": "development_only",
        "registration_sha256": state["registration_sha256"],
        "config_sha256": expected_config_sha,
    }


def _curve_rows(arm: str) -> list[dict[str, Any]]:
    path = STATE_DIR / "runs" / f"{arm}_seed192" / "validation_history.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _validate_run(arm: str) -> dict[str, Any]:
    run_dir = STATE_DIR / "runs" / f"{arm}_seed192"
    config_path = CONFIG_DIR / f"{arm}_seed192.json"
    config = _read_json(config_path)
    if _read_json(run_dir / "resolved_config.json") != config:
        raise RuntimeError(f"{arm} saved config differs from registration")
    rows = _curve_rows(arm)
    if [row.get("step") for row in rows] != list(CURVE_STEPS):
        raise RuntimeError(f"{arm} curve is incomplete or reordered")
    if any(
        row.get("validation_source") != "train"
        or row.get("validation_indices_sha256") != CURVE_SHA256
        for row in rows
    ):
        raise RuntimeError(f"{arm} used a nonregistered curve")
    curve = {int(row["step"]): float(row["exact_match"]) for row in rows}
    if any(not 0 <= value <= 1 for value in curve.values()):
        raise RuntimeError(f"{arm} curve contains invalid accuracy")
    run_manifest = _read_json(run_dir / "run_manifest.json")
    source_manifest = _read_json(REPRO_DIR / "source_manifest.json")
    if (
        run_manifest.get("git_commit") != source_manifest.get("git_commit")
        or run_manifest.get("git_dirty") is not False
        or run_manifest.get("reward_type") != "jlens"
        or run_manifest.get("config_sha256") != sha256_file(config_path)
        or run_manifest.get("resolved_config_sha256")
        != sha256_file(run_dir / "resolved_config.json")
        or run_manifest.get("lens_sha256") != LENS_SHA256
        or run_manifest.get("calibration_sha256") != CALIBRATION_SHA256[arm]
        or GPU_TYPE not in str(run_manifest.get("runtime", {}).get("cuda_device_name"))
    ):
        raise RuntimeError(f"{arm} run provenance mismatch")
    data = _read_json(run_dir / "data_indices.json")
    train = set(data.get("train_source_indices", []))
    validation = set(data.get("validation_source_indices", []))
    curve_indices = set(_read_json(MANIFEST_DIR / "curve_indices.json")["indices"])
    exclusions = set(_read_json(MANIFEST_DIR / "train_exclusions.json")["indices"])
    if (
        len(train) != 1000
        or len(validation) != 400
        or validation != curve_indices
        or train & validation
        or train & exclusions
    ):
        raise RuntimeError(f"{arm} data firewall failed")
    result = _read_json(run_dir / "run_result_manifest.json")
    receipt = _read_json(run_dir / "wandb_terminal_publish_receipt.json")
    if (
        result.get("completed_updates") != 15
        or result.get("registration_sha256")
        != _read_json(STATE_DIR / "protocol_state.json")["registration_sha256"]
        or result.get("evidence_eligibility")
        != "development_only_exposed_curve_no_significance_claim"
        or receipt.get("terminal_run_result_sha256")
        != sha256_file(run_dir / "run_result_manifest.json")
    ):
        raise RuntimeError(f"{arm} terminal evidence identity mismatch")
    for step in (5, 10, 15):
        if not (run_dir / f"checkpoint-{step}").is_dir():
            raise RuntimeError(f"{arm} is missing checkpoint {step}")
    passed = shape_pass(curve)
    return {
        "arm": arm,
        "weight": WEIGHTS[arm],
        "curve": {str(step): curve[step] for step in CURVE_STEPS},
        "shape_pass": passed,
        "step15_minus_step0": curve[15] - curve[0],
        "step15": curve[15],
        "literal_target_completion_rate": {
            str(row["step"]): row["literal_target_completion_rate"] for row in rows
        },
        "config_sha256": sha256_file(config_path),
        "run_result_manifest_sha256": sha256_file(
            run_dir / "run_result_manifest.json"
        ),
        "wandb_terminal_publish_receipt_sha256": sha256_file(
            run_dir / "wandb_terminal_publish_receipt.json"
        ),
    }


def shape_pass(curve: dict[int, float]) -> bool:
    return (
        curve[5] > curve[0]
        and curve[10] >= curve[5]
        and curve[15] >= curve[10]
    )


def rank_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if [item.get("arm") for item in results] != list(ARM_ORDER):
        raise RuntimeError("ranking requires all three arms in registered order")
    order_index = {arm: index for index, arm in enumerate(ARM_ORDER)}
    return sorted(
        results,
        key=lambda item: (
            bool(item["shape_pass"]),
            float(item["step15_minus_step0"]),
            float(item["step15"]),
            -order_index[item["arm"]],
        ),
        reverse=True,
    )


def summarize() -> dict[str, Any]:
    verify(require_launch=True)
    results = [_validate_run(arm) for arm in ARM_ORDER]
    ranked = rank_results(results)
    winner = ranked[0]
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "scientific_status": "development_only_no_significance_or_final_claim",
        "all_three_arms_completed": True,
        "fixed_serial_order": list(ARM_ORDER),
        "shared_seed": SEED,
        "curve_steps": list(CURVE_STEPS),
        "ranking_keys": [
            "shape_pass_boolean",
            "accuracy_step15_minus_step0",
            "accuracy_step15",
            "negative_fixed_arm_order_index",
        ],
        "arms": results,
        "ranking": [item["arm"] for item in ranked],
        "selected_development_candidate": winner["arm"],
        "selected_candidate_shape_passed": winner["shape_pass"],
        "interpretation": (
            "Exploratory selection on an exposed curve; confirmation requires a "
            "fresh prospective registration, new seeds, matched sign flips, and "
            "untouched-data tests."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = EVIDENCE_DIR / "tournament_summary.json"
    if path.exists():
        existing = _read_json(path)
        comparable = dict(existing)
        comparable.pop("generated_at_utc", None)
        expected = dict(payload)
        expected.pop("generated_at_utc", None)
        if comparable != expected:
            raise RuntimeError("immutable tournament summary already differs")
        return existing
    _write_json_exclusive(path, payload)
    return payload


def _full_inventory(excluded: set[Path]) -> dict[str, dict[str, Any]]:
    result = {}
    for path in sorted(STATE_DIR.rglob("*")):
        if not path.is_file() or path in excluded:
            continue
        result[path.relative_to(STATE_DIR).as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return result


def finalize_evidence() -> dict[str, Any]:
    summary = summarize()
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    inventory_path = EVIDENCE_DIR / "evidence_inventory.json"
    export_path = STATE_DIR / "exports" / "emotional_tournament_v1_evidence.zip"
    receipt_path = EVIDENCE_DIR / "durable_export_receipt.json"
    closeout_path = EVIDENCE_DIR / "git_closeout_candidate.json"
    if any(path.exists() for path in (inventory_path, export_path, receipt_path, closeout_path)):
        raise FileExistsError("refusing to overwrite terminal tournament evidence")
    excluded = {inventory_path, export_path, receipt_path, closeout_path}
    inventory = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "scope": (
            "All attempt files before inventory/export/receipt/closeout creation, "
            "including checkpoint and adapter blobs."
        ),
        "files": _full_inventory(excluded),
    }
    _write_json_exclusive(inventory_path, inventory)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    compact_names = []
    for name in sorted(inventory["files"]):
        path = STATE_DIR / name
        if path.suffix.lower() in {".safetensors", ".pt", ".pth", ".bin"}:
            continue
        parts = Path(name).parts
        if "runs" in parts and any(
            part == "final" or part.startswith("checkpoint-") for part in parts
        ):
            continue
        compact_names.append(name)
    compact_names.append(inventory_path.relative_to(STATE_DIR).as_posix())
    with zipfile.ZipFile(export_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in compact_names:
            archive.write(STATE_DIR / name, arcname=name)
    receipt = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "scientific_status": "development_only",
        "export_relative_path": export_path.relative_to(STATE_DIR).as_posix(),
        "export_sha256": sha256_file(export_path),
        "export_bytes": export_path.stat().st_size,
        "evidence_inventory_sha256": sha256_file(inventory_path),
        "compact_file_count": len(compact_names),
        "retrieval_command": (
            f"modal volume get {VOLUME_NAME} /exports/{export_path.name} "
            f"./{export_path.name}"
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_exclusive(receipt_path, receipt)
    state = _read_json(STATE_DIR / "protocol_state.json")
    claim = _read_json(STATE_DIR / "attempt_claim.json")
    launch = _read_json(STATE_DIR / "launch_receipt.json")
    closeout = {
        "schema_version": 1,
        "document_type": "j-lens-rl-development-emotional-tournament-closeout-candidate",
        "protocol": PROTOCOL,
        "terminal_stage": "complete",
        "scientific_status": "development_only_no_significance_or_final_claim",
        "claim_id": claim["claim_id"],
        "app_id": launch["app_id"],
        "function_call_id": launch["function_call_id"],
        "registration_sha256": state["registration_sha256"],
        "summary_sha256": sha256_file(EVIDENCE_DIR / "tournament_summary.json"),
        "inventory_sha256": sha256_file(inventory_path),
        "export_receipt_sha256": sha256_file(receipt_path),
        "selected_development_candidate": summary["selected_development_candidate"],
        "selected_candidate_shape_passed": summary["selected_candidate_shape_passed"],
    }
    _write_json_exclusive(closeout_path, closeout)
    return {"summary": summary, "export": receipt, "closeout": closeout}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("prepare", "verify", "verify-launch", "summarize", "finalize-evidence"),
    )
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare()
    elif args.command == "verify":
        result = verify(require_launch=False)
    elif args.command == "verify-launch":
        result = verify(require_launch=True)
    elif args.command == "summarize":
        result = summarize()
    else:
        result = finalize_evidence()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
