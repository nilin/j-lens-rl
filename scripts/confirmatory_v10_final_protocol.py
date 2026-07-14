"""Prospectively frozen final-collection machinery for confirmatory V12.

This module does not create a V12 registration, prepare state, unlock a final
set, or read the protected final manifest.  A separately created prospective
registration must materialize ``final_protocol_spec.json`` and an unlock marker.
The final path then enforces one immutable 9-label collection (base,
four treatments, four matched sign flips), verifies every raw record, performs
the predeclared paired analysis, and writes one acceptance decision.

The production final set must not be accessed merely to test this module.
Tests inject synthetic references and temporary manifests instead.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


PROTOCOL_FAMILY = "j-lens-rl-confirmatory-v12-celebration-infra-replacement"
PROTOCOL_ID = "j-lens-rl-confirmatory-v12-celebration-u4-u5-u6"
SCHEMA_VERSION = 1
SEEDS = tuple(range(224, 228))
CONDITIONS = ("jlens", "signflip")
CURVE_STEPS = (0, 4, 5, 6)
TERMINAL_STEP = 6
FINAL_EXAMPLES = 900
TARGET_WORDS = ("yay", "great", "success", "nice")
CALIBRATION_SHA256 = "93d05caf4848e745c07d908034b36f0b1ae465d8d89e1681134869c6b87a8ee6"
FINAL_MANIFEST_SHA256 = "1c3a544053504848318594ce21eea058d902884ba10c4f39ea3fa7796109b9c8"
SCIENCE_REGISTRATION_PATH = (
    "protocol_archive/v12_celebration_infrastructure_replacement_registration.json"
)
SCIENCE_REGISTRATION_SHA256 = (
    "f58f35419549de5905c7d873a71f67edda73289585025f9084901b61be4a9749"
)
CANDIDATE_FREEZE_PATH = "protocol_archive/v11_celebration_candidate_freeze.json"
CANDIDATE_FREEZE_SHA256 = "dbdc67346906664d8768271ed93830e73de713b3e06326170a5586d8ef17d6f9"
CANDIDATE_FREEZE_CORRECTION_PATH = (
    "protocol_archive/v11_celebration_infrastructure_closeout.json"
)
CANDIDATE_FREEZE_CORRECTION_SHA256 = (
    "cbc4c78dcac153675e460e4aff344c12a44a55e34c71de300da3195f44d9c806"
)
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
LENS_PATH = "artifacts/qwen25_05b_solved_lens.pt"
LENS_SHA256 = "178a9671cbf41882135807bde59b828e36c6f8f98b32c809ea3346860aad10dc"
CALIBRATION_PATH = (
    "protocol_archive/emotional_screen_forensic_bundle/family/artifacts/"
    "celebration_calibration.json"
)
CURVE_MANIFEST_PATH = ".confirmatory/manifests/curve_indices.json"
CURVE_MANIFEST_SHA256 = "ad348fe17d2e6bd6aac691d9bcdbb9da481f675305fa0e05c68e86dad97451c1"
TRAIN_EXCLUSIONS_PATH = ".confirmatory/manifests/train_exclusions.json"
TRAIN_EXCLUSIONS_SHA256 = "7c1ca4f404ba9149093cc3c57dc3607582f671397e2fe99e93449848c1d65d61"
FINAL_MANIFEST_PATH = "/state/manifests/sealed_final_indices.json"
TREATMENT_SCORE_COMPONENTS = (
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
FINAL_LABELS = (
    "base",
    *(f"jlens_seed{seed}" for seed in SEEDS),
    *(f"signflip_seed{seed}" for seed in SEEDS),
)
CURVE_CRITERION = "M4 > M0, M5 >= M4, and M6 >= M5 on the four-treatment-seed mean"
COLLECTION_POLICY = (
    "collect all 9 registered labels serially before semantically parsing, scoring, "
    "comparing, reporting, or exposing any per-label outcome; opaque SHA-256 receipt "
    "hashing is allowed"
)
MATCHED_CONTROL_RULE = (
    "negate every selected treatment score-component weight; preserve the "
    "seed, training rows, optimizer, horizon, decoding, and all other science"
)
ANALYSIS_REGISTRATION = {
    "primary_estimand": (
        "paired difference-in-differences: (selected-emotional-treatment minus "
        "base) minus (matched-signflip minus base), matched by seed and final item"
    ),
    "secondary_estimand": (
        "paired selected-emotional-treatment minus base across seeds and final items"
    ),
    "bootstrap_method": "crossed seed-and-item percentile bootstrap",
    "bootstrap_samples": 10_000,
    "bootstrap_seed": 0,
    "confidence": 0.95,
    "crossed_95pct_intervals": "reported descriptively, not used as acceptance gates",
    "seed_sign_test": "exact two-sided sign test across four registered seeds",
    "acceptance_alpha": 0.15,
}
ACCEPTANCE_REGISTRATION = {
    "curve_gate_passed": True,
    "treatment_vs_base_mean": "> 0",
    "treatment_vs_base_seed_effects": "4 positive, 0 negative, 0 ties",
    "treatment_vs_base_exact_two_sided_sign_p": 0.125,
    "difference_in_differences_mean": "> 0",
    "difference_in_differences_seed_effects": "4 positive, 0 negative, 0 ties",
    "difference_in_differences_exact_two_sided_sign_p": 0.125,
    "crossed_95pct_intervals": "reported descriptively, not used as acceptance gates",
    "literal_provenance_environment_collection_and_audit_checks": "all pass",
}
ANALYSIS_SOURCE_PROBE_PROGRAM = """\
import hashlib
import json
from pathlib import Path
import jlens_rl.common as common
import jlens_rl.paired_eval as paired_eval

def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

print(json.dumps({
    "loaded_source_identity": {
        "jlens_rl.common": {
            "path": str(Path(common.__file__).resolve()),
            "sha256": digest(common.__file__),
        },
        "jlens_rl.paired_eval": {
            "path": str(Path(paired_eval.__file__).resolve()),
            "sha256": digest(paired_eval.__file__),
        },
    },
    "environment_snapshot": common.runtime_environment_snapshot(),
}, sort_keys=True))
"""
EXPECTED_SOFTWARE = {
    "j-lens-rl": "0.1.0",
    "torch": "2.9.1",
    "transformers": "5.5.0",
    "datasets": "4.7.0",
    "peft": "0.18.0",
}
AUDITED_SOURCE_PATHS = {
    "protocol": "scripts/confirmatory_v10_final_protocol.py",
    "runner": "scripts/confirmatory_v10_final_runner.py",
    "training_entrypoint": "scripts/confirmatory_v10_train.py",
    "shared_train": "src/jlens_rl/train.py",
    "eval": "src/jlens_rl/eval.py",
    "common": "src/jlens_rl/common.py",
    "paired_eval": "src/jlens_rl/paired_eval.py",
}
AUDITED_TEST_PATHS = (
    "tests/test_v10_final_automation.py",
    "tests/test_paired_eval.py",
)
TERMINAL_EVIDENCE_NAMES = (
    "run_result_manifest.json",
    "validation_history.jsonl",
    "log_history.json",
    "environment_snapshot.json",
    "run_manifest.json",
    "resolved_config.json",
    "data_indices.json",
)
OFFLINE_SYNC_POLICY = (
    "Preserve this completed offline directory with its seven embedded terminal "
    "files; sync this directory only. Never rerun or resume optimization to "
    "repair tracking infrastructure."
)
TRAINING_CONFIG_KEYS = {
    "model_name", "model_revision", "dataset_revision", "lens_path",
    "lens_sha256", "expected_lens_sha256", "calibration_path",
    "calibration_sha256", "expected_calibration_sha256", "target_words",
    "train_examples", "validation_examples", "validation_batch_size",
    "num_generations", "num_generations_eval", "max_prompt_tokens",
    "max_new_tokens", "min_new_tokens", "temperature", "updates",
    "learning_rate", "lr_scheduler_type", "warmup_steps", "warmup_ratio",
    "kl_beta", "loss_type", "scale_rewards", "gradient_accumulation_steps",
    "lora_rank", "lora_alpha", "score_stride", "score_start_fraction",
    "score_layers", "score_aggregation", "score_include_final",
    "vocab_chunk_size", "mask_target_tokens", "eval_every",
    "validation_steps", "validation_source", "validation_indices_path",
    "reserved_train_indices_path", "validation_observational_only",
    "early_stopping_patience", "early_stopping_min_delta", "eval_strategy",
    "save_every", "save_total_limit", "reward_type",
    "require_clean_repository", "wandb_entity", "wandb_project",
    "wandb_group", "wandb_mode", "wandb_resume", "wandb_tags",
    "curve_manifest_sha256", "train_exclusions_manifest_sha256",
    "registered_backend", "expected_cuda_device_name",
    "evidence_eligibility", "seed", "score_components", "output_dir",
    "run_name", "wandb_run_id", "wandb_url", "registration_sha256",
    "recipe_lock_sha256", "recipe_sha256", "registered_code_sha256",
    "registered_spec_projection_sha256", "metric_schema_path",
    "metric_schema_sha256", "registered_command",
}
REGISTERED_SPEC_FIELDS = (
    "schema_version", "protocol_family", "protocol", "repository",
    "python_executable", "gpu_lock_path", "recipe_sha256",
    "registered_code_sha256", "target_words", "seeds", "conditions",
    "terminal_step", "curve_gate", "matched_control_rule", "analysis",
    "acceptance", "final_collection", "artifacts", "model", "dataset",
    "hardware", "software", "treatment_score_components",
    "matched_control_score_components", "training", "paths", "firewall",
    "metric_schema", "wandb", "science_registration", "candidate_freeze",
    "candidate_freeze_correction", "modal_execution",
)


class FinalProtocolError(RuntimeError):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _is_commit(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None


def curve_means_pass(values: Sequence[float]) -> bool:
    if len(values) != 4 or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0 <= float(value) <= 1
        for value in values
    ):
        raise FinalProtocolError("curve gate requires four finite accuracy fractions")
    return values[1] > values[0] and values[2] >= values[1] and values[3] >= values[2]


def evaluation_role(label: str) -> tuple[str, int, bool]:
    if label == "base":
        return "jlens", SEEDS[0], True
    match = re.fullmatch(r"(jlens|signflip)_seed(\d+)", label)
    if match is None:
        raise FinalProtocolError(f"unregistered V10 final label: {label!r}")
    condition, seed_text = match.groups()
    seed = int(seed_text)
    if seed not in SEEDS:
        raise FinalProtocolError(f"unregistered V10 seed in final label: {label!r}")
    return condition, seed, False


@dataclass(frozen=True)
class FinalContext:
    state_dir: Path
    repository: Path
    spec_path: Path
    spec: dict[str, Any]

    @property
    def config_dir(self) -> Path:
        return self.state_dir / "configs"

    @property
    def run_dir(self) -> Path:
        return self.state_dir / "runs"

    @property
    def manifest_dir(self) -> Path:
        return self.state_dir / "manifests"

    @property
    def eval_dir(self) -> Path:
        return self.state_dir / "evals"

    @property
    def evidence_dir(self) -> Path:
        return self.state_dir / "evidence"

    @property
    def unlock_path(self) -> Path:
        return self.state_dir / "final_unlocked.json"

    @property
    def completed_runs_path(self) -> Path:
        return self.evidence_dir / "completed_runs.json"

    @property
    def curve_path(self) -> Path:
        return self.evidence_dir / "curve_gate.json"

    @property
    def collection_path(self) -> Path:
        return self.state_dir / "final_collection.json"

    @property
    def comparison_path(self) -> Path:
        return self.evidence_dir / "sealed_comparison.json"

    @property
    def analysis_process_path(self) -> Path:
        return self.evidence_dir / "analysis_process.json"

    @property
    def acceptance_path(self) -> Path:
        return self.evidence_dir / "acceptance.json"

    @property
    def failure_path(self) -> Path:
        return self.evidence_dir / "final_collection_failure.json"


def _resolve_bound_path(repository: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise FinalProtocolError(f"{field} must be a non-empty path")
    path = Path(value)
    return path.absolute() if path.is_absolute() else (repository / path).absolute()


def _tree_identity(path: Path) -> dict[str, Any]:
    original = Path(path)
    if original.is_symlink():
        raise FinalProtocolError(f"terminal tree is absent or unsafe: {original}")
    path = original.resolve()
    if not path.is_dir():
        raise FinalProtocolError(f"terminal tree is absent or unsafe: {path}")
    files: dict[str, str] = {}
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            raise FinalProtocolError(f"terminal tree contains a symlink: {item}")
        if item.is_file():
            files[item.relative_to(path).as_posix()] = sha256_file(item)
    if not files:
        raise FinalProtocolError(f"terminal tree is empty: {path}")
    return {"path": str(path), "sha256": canonical_sha256(files), "files": files}


def _offline_inventory(root: Path) -> list[dict[str, Any]]:
    if root.is_symlink() or not root.is_dir():
        raise FinalProtocolError("offline W&B root is absent or symlinked")
    records: list[dict[str, Any]] = []
    for current, dirnames, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        retained: list[str] = []
        for name in dirnames:
            path = current_path / name
            if path.is_symlink():
                records.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "type": "symlink",
                        "target": os.readlink(path),
                        "mode": oct(stat.S_IMODE(path.lstat().st_mode)),
                    }
                )
            elif path.is_dir():
                retained.append(name)
            else:
                raise FinalProtocolError(f"special offline W&B directory entry: {path}")
        dirnames[:] = retained
        for name in filenames:
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                records.append(
                    {
                        "path": relative,
                        "type": "symlink",
                        "target": os.readlink(path),
                        "mode": oct(stat.S_IMODE(path.lstat().st_mode)),
                    }
                )
                continue
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags)
            digest = hashlib.sha256()
            try:
                before = os.fstat(descriptor)
                if not stat.S_ISREG(before.st_mode):
                    raise FinalProtocolError(f"special offline W&B file entry: {path}")
                while True:
                    chunk = os.read(descriptor, 1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                after = os.fstat(descriptor)
                if (
                    before.st_dev,
                    before.st_ino,
                    before.st_size,
                    before.st_mtime_ns,
                ) != (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                ):
                    raise FinalProtocolError(f"offline W&B file changed during audit: {path}")
            finally:
                os.close(descriptor)
            records.append(
                {
                    "path": relative,
                    "type": "file",
                    "sha256": digest.hexdigest(),
                    "size_bytes": after.st_size,
                    "mode": oct(stat.S_IMODE(after.st_mode)),
                }
            )
    return sorted(records, key=lambda item: item["path"])


def _score_components(value: Any, field: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise FinalProtocolError(f"{field} must contain score components")
    expected_keys = {
        "layer", "start_fraction", "end_fraction", "aggregation", "weight"
    }
    result: list[dict[str, Any]] = []
    for component in value:
        if (
            not isinstance(component, dict)
            or set(component) != expected_keys
            or isinstance(component.get("layer"), bool)
            or not isinstance(component.get("layer"), int)
            or component["layer"] < 0
            or isinstance(component.get("start_fraction"), bool)
            or not isinstance(component.get("start_fraction"), (int, float))
            or isinstance(component.get("end_fraction"), bool)
            or not isinstance(component.get("end_fraction"), (int, float))
            or not 0 <= float(component["start_fraction"])
            < float(component["end_fraction"]) <= 1
            or component.get("aggregation") != "mean"
            or isinstance(component.get("weight"), bool)
            or not isinstance(component.get("weight"), (int, float))
            or not math.isfinite(float(component["weight"]))
            or float(component["weight"]) == 0
        ):
            raise FinalProtocolError(f"{field} contains an invalid score component")
        normalized = dict(component)
        normalized["weight"] = float(component["weight"])
        result.append(normalized)
    return result


def _negated_components(value: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for component in value:
        item = dict(component)
        item["weight"] = -float(item["weight"])
        result.append(item)
    return result


def _manifest_payload(path: Path, *, expected_count: int | None = None) -> tuple[dict[str, Any], list[int]]:
    value = read_json(path)
    if not isinstance(value, dict) or set(value) != {"dataset", "subset", "split", "indices"}:
        raise FinalProtocolError(f"manifest has an unexpected schema: {path}")
    indices = value.get("indices")
    if (
        value.get("dataset") != "openai/gsm8k"
        or value.get("subset") != "main"
        or value.get("split") != "train"
        or not isinstance(indices, list)
        or (expected_count is not None and len(indices) != expected_count)
        or len(indices) != len(set(indices))
        or any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in indices)
    ):
        raise FinalProtocolError(f"manifest metadata/indices are invalid: {path}")
    return value, list(indices)


def registered_spec_projection(spec: dict[str, Any]) -> dict[str, Any]:
    missing = [field for field in REGISTERED_SPEC_FIELDS if field not in spec]
    if missing:
        raise FinalProtocolError(f"registered V10 spec projection lacks {missing}")
    return {field: spec[field] for field in REGISTERED_SPEC_FIELDS}


def registered_spec_projection_sha256(spec: dict[str, Any]) -> str:
    return canonical_sha256(registered_spec_projection(spec))


def registered_recipe(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "training_entrypoint": spec["paths"]["training_entrypoint"],
        "training_config_keys": sorted(TRAINING_CONFIG_KEYS),
        "training_config_derivation": (
            "exact expected_training_config(protocol, condition, seed) for all "
            "four treatment and four matched-signflip runs"
        ),
        "sealed_evaluation_derivation": "exact expected_sealed_eval_config(protocol)",
        "registered_code_sha256": spec["registered_code_sha256"],
    }


def expected_registration_document(spec: dict[str, Any]) -> dict[str, Any]:
    projection = registered_spec_projection(spec)
    return {
        "schema_version": 1,
        "protocol": spec["protocol"],
        "status": "registered_before_v10_training_and_final_unlock",
        "protected_payloads_accessed": False,
        "registered_spec_projection": projection,
        "registered_spec_projection_sha256": canonical_sha256(projection),
    }


def expected_recipe_lock_document(spec: dict[str, Any]) -> dict[str, Any]:
    recipe = registered_recipe(spec)
    return {
        "schema_version": 1,
        "protocol": spec["protocol"],
        "registered_spec_projection_sha256": registered_spec_projection_sha256(spec),
        "recipe": recipe,
        "recipe_sha256": canonical_sha256(recipe),
    }


def validate_spec(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FinalProtocolError("final protocol spec must be a JSON object")
    spec = value
    if (
        spec.get("schema_version") != SCHEMA_VERSION
        or spec.get("protocol_family") != PROTOCOL_FAMILY
        or spec.get("protocol") != PROTOCOL_ID
        or spec.get("seeds") != list(SEEDS)
        or spec.get("conditions") != list(CONDITIONS)
        or spec.get("terminal_step") != TERMINAL_STEP
        or spec.get("curve_gate")
        != {"steps": list(CURVE_STEPS), "criterion": CURVE_CRITERION}
        or spec.get("matched_control_rule") != MATCHED_CONTROL_RULE
        or spec.get("analysis") != ANALYSIS_REGISTRATION
        or spec.get("acceptance") != ACCEPTANCE_REGISTRATION
    ):
        raise FinalProtocolError("final spec changed the frozen four-seed V10 design")
    if (
        spec.get("final_collection", {}).get("count") != FINAL_EXAMPLES
        or spec.get("final_collection", {}).get("labels") != list(FINAL_LABELS)
        or spec.get("final_collection", {}).get("single_immutable_collection") is not True
        or not _is_sha256(spec.get("final_collection", {}).get("manifest_sha256"))
        or spec.get("final_collection", {}).get("manifest_sha256")
        != FINAL_MANIFEST_SHA256
        or spec.get("final_collection", {}).get("manifest_path")
        != FINAL_MANIFEST_PATH
        or spec.get("final_collection", {}).get("manifest_metadata")
        != {"dataset": "openai/gsm8k", "subset": "main", "split": "train"}
    ):
        raise FinalProtocolError("final spec changed the immutable 9-label collection")
    if spec.get("target_words") != list(TARGET_WORDS):
        raise FinalProtocolError(
            "V12 targets must remain the frozen celebration-family words"
        )
    if (
        not _is_commit(spec.get("git_commit"))
        or not _is_sha256(spec.get("source_tree_sha256"))
        or any(
            not isinstance(spec.get(field), str) or not spec[field]
            for field in ("repository", "python_executable", "gpu_lock_path")
        )
        or not _is_sha256(spec.get("registration_sha256"))
        or not _is_sha256(spec.get("recipe_lock_sha256"))
        or not isinstance(spec.get("registration_path"), str)
        or not isinstance(spec.get("recipe_lock_path"), str)
        or not _is_sha256(spec.get("artifacts", {}).get("lens_sha256"))
        or not _is_sha256(spec.get("artifacts", {}).get("calibration_sha256"))
        or spec.get("artifacts", {}).get("calibration_sha256")
        != CALIBRATION_SHA256
        or spec.get("artifacts", {}).get("lens_sha256") != LENS_SHA256
        or spec.get("artifacts", {}).get("lens_path") != LENS_PATH
        or spec.get("artifacts", {}).get("calibration_path") != CALIBRATION_PATH
    ):
        raise FinalProtocolError("future V10 source/registration/artifact binding is incomplete")
    model = spec.get("model", {})
    dataset = spec.get("dataset", {})
    if (
        model.get("name") != MODEL_NAME
        or model.get("revision") != MODEL_REVISION
        or model.get("dtype") != "torch.bfloat16"
        or dataset.get("name") != "openai/gsm8k"
        or dataset.get("subset") != "main"
        or dataset.get("split") != "train"
        or dataset.get("revision") != DATASET_REVISION
        or not isinstance(dataset.get("size"), int)
        or dataset["size"] <= FINAL_EXAMPLES
    ):
        raise FinalProtocolError("future V10 model/dataset identity is incomplete")
    hardware = spec.get("hardware", {})
    if (
        set(hardware)
        != {
            "backend", "max_gpu_processes", "gpu_per_worker",
            "max_modal_gpus_before_2026_07_14_23_00_utc",
            "max_modal_gpus_at_or_after_2026_07_14_23_00_utc",
            "device_name", "driver_version", "cuda_version", "torch_version",
            "memory_total_mib",
        }
        or hardware.get("backend") != "modal"
        or hardware.get("max_gpu_processes") != 1
        or hardware.get("gpu_per_worker") != 1
        or hardware.get("max_modal_gpus_before_2026_07_14_23_00_utc") != 5
        or hardware.get("max_modal_gpus_at_or_after_2026_07_14_23_00_utc") != 10
        or "L40S" not in str(hardware.get("device_name"))
        or not isinstance(hardware.get("driver_version"), str)
        or not hardware["driver_version"]
        or not isinstance(hardware.get("cuda_version"), str)
        or not hardware["cuda_version"]
        or not isinstance(hardware.get("torch_version"), str)
        or not hardware["torch_version"]
        or not isinstance(hardware.get("memory_total_mib"), int)
        or hardware["memory_total_mib"] <= 0
    ):
        raise FinalProtocolError("future V10 must bind one exact Modal NVIDIA L40S")
    if spec.get("software") != EXPECTED_SOFTWARE:
        raise FinalProtocolError("future V10 software versions changed")
    treatment = _score_components(
        spec.get("treatment_score_components"), "treatment_score_components"
    )
    if (
        treatment != [dict(component) for component in TREATMENT_SCORE_COMPONENTS]
        or spec.get("matched_control_score_components")
        != _negated_components(treatment)
    ):
        raise FinalProtocolError(
            "future V12 treatment/signflip differs from the frozen celebration-tail recipe"
        )
    training = spec.get("training", {})
    if training != {
        "train_examples": 1000,
        "validation_examples": 400,
        "validation_batch_size": 64,
        "num_generations": 8,
        "num_generations_eval": 1,
        "max_prompt_tokens": 384,
        "max_new_tokens": 256,
        "min_new_tokens": 64,
        "temperature": 1.0,
        "updates": 6,
        "learning_rate": 3e-6,
        "lr_scheduler_type": "constant",
        "warmup_steps": 0,
        "warmup_ratio": 0.0,
        "kl_beta": 0.02,
        "loss_type": "dapo",
        "scale_rewards": "group",
        "gradient_accumulation_steps": 1,
        "lora_rank": 8,
        "lora_alpha": 16,
        "score_stride": 10,
        "score_start_fraction": 0.5,
        "score_layers": [8],
        "score_aggregation": "mean",
        "score_include_final": False,
        "vocab_chunk_size": 16384,
        "mask_target_tokens": True,
        "eval_every": 1,
        "validation_steps": [4, 5, 6],
        "validation_observational_only": True,
        "early_stopping_patience": None,
        "early_stopping_min_delta": 0.0,
        "save_every": 6,
        "save_total_limit": 1,
    }:
        raise FinalProtocolError("future V10 training schema/hyperparameters changed")
    paths = spec.get("paths", {})
    expected_path_keys = {
        "lens_config_path", "calibration_config_path", "curve_config_path",
        "train_exclusions_config_path", "metric_schema_config_path",
        "state_config_prefix", "training_entrypoint",
    }
    if (
        not isinstance(paths, dict)
        or set(paths) != expected_path_keys
        or any(not isinstance(item, str) or not item for item in paths.values())
        or paths["training_entrypoint"] != "scripts/confirmatory_v10_train.py"
        or paths["lens_config_path"] != LENS_PATH
        or paths["calibration_config_path"] != CALIBRATION_PATH
        or paths["curve_config_path"] != CURVE_MANIFEST_PATH
        or paths["train_exclusions_config_path"] != TRAIN_EXCLUSIONS_PATH
    ):
        raise FinalProtocolError("future V10 config/runtime paths are incomplete")
    firewall = spec.get("firewall", {})
    if set(firewall) != {"curve_manifest", "train_exclusions", "disjointness_receipt"}:
        raise FinalProtocolError("future V10 firewall binding is incomplete")
    for name in ("curve_manifest", "train_exclusions"):
        item = firewall.get(name, {})
        if (
            not isinstance(item.get("path"), str)
            or not _is_sha256(item.get("sha256"))
            or not isinstance(item.get("count"), int)
            or item["count"] <= 0
        ):
            raise FinalProtocolError(f"future V10 {name} binding is incomplete")
    if firewall["curve_manifest"]["count"] != 400:
        raise FinalProtocolError("future V10 development curve must have 400 rows")
    if (
        firewall["curve_manifest"]
        != {
            "path": CURVE_MANIFEST_PATH,
            "sha256": CURVE_MANIFEST_SHA256,
            "count": 400,
        }
        or firewall["train_exclusions"].get("path") != TRAIN_EXCLUSIONS_PATH
        or firewall["train_exclusions"].get("sha256") != TRAIN_EXCLUSIONS_SHA256
    ):
        raise FinalProtocolError("future V10 exposed-data firewall differs from registration")
    receipt_binding = firewall.get("disjointness_receipt", {})
    if (
        not isinstance(receipt_binding.get("path"), str)
        or not _is_sha256(receipt_binding.get("sha256"))
    ):
        raise FinalProtocolError("future V10 disjointness receipt is incomplete")
    metric = spec.get("metric_schema", {})
    if not isinstance(metric.get("path"), str) or not _is_sha256(metric.get("sha256")):
        raise FinalProtocolError("future V10 metric schema is not bound")
    wandb = spec.get("wandb", {})
    if (
        wandb.get("entity") != "nilinabra-spare-time"
        or wandb.get("project") != "j-lens-rl"
        or wandb.get("group") != "confirm-v12-celebration-u4-u5-u6"
        or wandb.get("mode") != "online"
        or not isinstance(wandb.get("tags"), list)
        or not wandb["tags"]
        or set(wandb.get("run_ids", {})) != set(FINAL_LABELS[1:])
        or len(set(wandb["run_ids"].values())) != 8
        or wandb["run_ids"]
        != {
            f"{condition}_seed{seed}": (
                f"confirm-v12-celebration-{condition}-seed{seed}"
            )
            for condition in CONDITIONS
            for seed in SEEDS
        }
    ):
        raise FinalProtocolError("future V10 W&B identity inventory is incomplete")
    config_sha = spec.get("config_sha256")
    expected_config_names = {"sealed_eval", *FINAL_LABELS[1:]}
    if (
        not isinstance(config_sha, dict)
        or set(config_sha) != expected_config_names
        or any(not _is_sha256(digest) for digest in config_sha.values())
    ):
        raise FinalProtocolError("future V10 config hash inventory is incomplete")
    audit = spec.get("automation_audit", {})
    if (
        not isinstance(audit.get("path"), str)
        or not audit["path"]
        or not _is_sha256(audit.get("sha256"))
    ):
        raise FinalProtocolError("future V10 must bind a separate final-automation audit")
    if not _is_sha256(spec.get("registered_code_sha256")) or not _is_sha256(
        spec.get("recipe_sha256")
    ):
        raise FinalProtocolError("future V10 registered code/recipe identity is incomplete")
    if spec["recipe_sha256"] != canonical_sha256(registered_recipe(spec)):
        raise FinalProtocolError("future V10 recipe identity does not match its exact schema")
    if (
        spec.get("science_registration")
        != {
            "path": SCIENCE_REGISTRATION_PATH,
            "sha256": SCIENCE_REGISTRATION_SHA256,
        }
        or spec.get("candidate_freeze")
        != {"path": CANDIDATE_FREEZE_PATH, "sha256": CANDIDATE_FREEZE_SHA256}
        or spec.get("candidate_freeze_correction")
        != {
            "path": CANDIDATE_FREEZE_CORRECTION_PATH,
            "sha256": CANDIDATE_FREEZE_CORRECTION_SHA256,
        }
        or not isinstance(spec.get("modal_execution"), dict)
        or set(spec["modal_execution"]) != {"contract_path", "contract_sha256"}
        or not isinstance(spec["modal_execution"].get("contract_path"), str)
        or not spec["modal_execution"]["contract_path"]
        or not _is_sha256(spec["modal_execution"].get("contract_sha256"))
    ):
        raise FinalProtocolError("future V10 does not bind the frozen prospective science")
    return spec


def load_context(state_dir: str | Path) -> FinalContext:
    state = Path(state_dir).resolve()
    spec_path = state / "reproducibility" / "final_protocol_spec.json"
    if not spec_path.is_file():
        raise FinalProtocolError("future V10 final protocol spec is absent")
    spec = validate_spec(read_json(spec_path))
    if (
        not Path(spec.get("repository", "")).is_absolute()
        or not Path(spec.get("python_executable", "")).is_absolute()
        or not Path(spec.get("python_executable", "")).is_file()
        or not Path(spec.get("gpu_lock_path", "")).is_absolute()
    ):
        raise FinalProtocolError(
            "future V10 must bind absolute repository, Python, and GPU-lock paths"
        )
    repository = _resolve_bound_path(Path.cwd(), spec.get("repository"), "repository")
    if not repository.is_dir() or repository.is_symlink():
        raise FinalProtocolError("bound V10 runtime repository is absent")
    return FinalContext(state, repository.resolve(), spec_path, spec)


def expected_training_config(context: FinalContext, condition: str, seed: int) -> dict[str, Any]:
    if condition not in CONDITIONS or seed not in SEEDS:
        raise FinalProtocolError("requested an unregistered V10 training config")
    spec = context.spec
    training = spec["training"]
    paths = spec["paths"]
    label = f"{condition}_seed{seed}"
    run_id = spec["wandb"]["run_ids"][label]
    config: dict[str, Any] = {
        "model_name": spec["model"]["name"],
        "model_revision": spec["model"]["revision"],
        "dataset_revision": spec["dataset"]["revision"],
        "lens_path": paths["lens_config_path"],
        "lens_sha256": spec["artifacts"]["lens_sha256"],
        "expected_lens_sha256": spec["artifacts"]["lens_sha256"],
        "calibration_path": paths["calibration_config_path"],
        "calibration_sha256": spec["artifacts"]["calibration_sha256"],
        "expected_calibration_sha256": spec["artifacts"]["calibration_sha256"],
        "target_words": spec["target_words"],
        **training,
        "eval_strategy": "no",
        "validation_source": "train",
        "validation_indices_path": paths["curve_config_path"],
        "reserved_train_indices_path": paths["train_exclusions_config_path"],
        "reward_type": "jlens",
        "require_clean_repository": True,
        "wandb_entity": spec["wandb"]["entity"],
        "wandb_project": spec["wandb"]["project"],
        "wandb_group": spec["wandb"]["group"],
        "wandb_mode": "online",
        "wandb_resume": "never",
        "wandb_tags": spec["wandb"]["tags"],
        "curve_manifest_sha256": spec["firewall"]["curve_manifest"]["sha256"],
        "train_exclusions_manifest_sha256": spec["firewall"]["train_exclusions"]["sha256"],
        "registered_backend": "modal-l40s",
        "expected_cuda_device_name": spec["hardware"]["device_name"],
        "evidence_eligibility": "original_registered_v10_modal_attempt",
        "seed": seed,
        "score_components": (
            spec["treatment_score_components"]
            if condition == "jlens"
            else spec["matched_control_score_components"]
        ),
        "output_dir": f"{paths['state_config_prefix']}/runs/{label}",
        "run_name": run_id,
        "wandb_run_id": run_id,
        "wandb_url": (
            f"https://wandb.ai/{spec['wandb']['entity']}/{spec['wandb']['project']}/runs/{run_id}"
        ),
        "registration_sha256": spec["registration_sha256"],
        "recipe_lock_sha256": spec["recipe_lock_sha256"],
        "recipe_sha256": spec["recipe_sha256"],
        "registered_code_sha256": spec["registered_code_sha256"],
        "registered_spec_projection_sha256": registered_spec_projection_sha256(spec),
        "metric_schema_path": paths["metric_schema_config_path"],
        "metric_schema_sha256": spec["metric_schema"]["sha256"],
        "registered_command": [
            spec["python_executable"],
            paths["training_entrypoint"],
            "--config",
            f"{paths['state_config_prefix']}/configs/{label}.json",
            "--wandb-mode",
            "online",
        ],
    }
    if set(config) != TRAINING_CONFIG_KEYS:
        missing = sorted(TRAINING_CONFIG_KEYS - set(config))
        extra = sorted(set(config) - TRAINING_CONFIG_KEYS)
        raise FinalProtocolError(f"internal V10 config schema drift: missing={missing}, extra={extra}")
    return config


def expected_sealed_eval_config(context: FinalContext) -> dict[str, Any]:
    result = dict(expected_training_config(context, "jlens", SEEDS[0]))
    result.update(
        {
            "validation_examples": FINAL_EXAMPLES,
            "evaluation_source": "train",
            "evaluation_indices_path": context.spec["final_collection"]["manifest_path"],
            "evaluation_seed": 0,
            "min_new_tokens": 0,
            "output_dir": f"{context.spec['paths']['state_config_prefix']}/evaluation_config_unused",
            "run_name": f"{context.spec['wandb']['group']}-sealed-evaluation",
        }
    )
    for key in ("wandb_run_id", "wandb_url", "wandb_resume", "wandb_tags", "registered_command"):
        result.pop(key)
    return result


def verify_bound_configs(context: FinalContext) -> dict[str, str]:
    expected_names = {"sealed_eval.json", *(f"{label}.json" for label in FINAL_LABELS[1:])}
    entries = list(context.config_dir.iterdir()) if context.config_dir.is_dir() else []
    observed_names = {path.name for path in entries if path.is_file()}
    if (
        context.config_dir.is_symlink()
        or observed_names != expected_names
        or any(path.is_symlink() or not path.is_file() for path in entries)
    ):
        raise FinalProtocolError("V10 config directory is not the exact 8-run plus sealed-eval inventory")
    hashes: dict[str, str] = {}
    for condition in CONDITIONS:
        for seed in SEEDS:
            label = f"{condition}_seed{seed}"
            path = context.config_dir / f"{label}.json"
            expected = expected_training_config(context, condition, seed)
            if read_json(path) != expected or sha256_file(path) != context.spec["config_sha256"][label]:
                raise FinalProtocolError(f"V10 config schema/hash changed for {label}")
            hashes[label] = sha256_file(path)
    sealed_path = context.config_dir / "sealed_eval.json"
    if (
        read_json(sealed_path) != expected_sealed_eval_config(context)
        or sha256_file(sealed_path) != context.spec["config_sha256"]["sealed_eval"]
    ):
        raise FinalProtocolError("V10 sealed evaluation config schema/hash changed")
    hashes["sealed_eval"] = sha256_file(sealed_path)
    return hashes


def _audit_design() -> dict[str, Any]:
    return {
        "seeds": list(SEEDS),
        "curve_steps": list(CURVE_STEPS),
        "terminal_step": TERMINAL_STEP,
        "final_labels": list(FINAL_LABELS),
        "final_examples": FINAL_EXAMPLES,
        "analysis": ANALYSIS_REGISTRATION,
        "acceptance": ACCEPTANCE_REGISTRATION,
    }


def verify_automation_audit(context: FinalContext) -> dict[str, Any]:
    binding = context.spec["automation_audit"]
    path = _state_repro_file(context, binding["path"], "automation_audit.path")
    if not path.is_file() or path.is_symlink() or sha256_file(path) != binding["sha256"]:
        raise FinalProtocolError("separately reviewed final-automation audit is absent or changed")
    value = read_json(path)
    bound_sources = {
        name: context.repository / relative
        for name, relative in AUDITED_SOURCE_PATHS.items()
    }
    active_protocol = Path(__file__).resolve()
    active_runner = active_protocol.with_name("confirmatory_v10_final_runner.py")
    if (
        any(not path.is_file() or path.is_symlink() for path in bound_sources.values())
        or sha256_file(active_protocol) != sha256_file(bound_sources["protocol"])
        or sha256_file(active_runner) != sha256_file(bound_sources["runner"])
    ):
        raise FinalProtocolError("loaded final automation differs from the bound repository")
    bound_tests = {
        relative: context.repository / relative for relative in AUDITED_TEST_PATHS
    }
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "schema_version", "decision", "protected_payloads_accessed", "auditor",
            "audited_commit", "source_sha256", "test_source_sha256", "design",
            "test_command", "tests_passed",
        }
        or value.get("schema_version") != 1
        or value.get("decision") != "approved_before_final_unlock"
        or value.get("protected_payloads_accessed") is not False
        or not isinstance(value.get("auditor"), str)
        or not value["auditor"]
        or value.get("audited_commit") != context.spec["git_commit"]
        or value.get("source_sha256")
        != {name: sha256_file(source) for name, source in bound_sources.items()}
        or context.spec["registered_code_sha256"]
        != canonical_sha256(
            {name: sha256_file(source) for name, source in bound_sources.items()}
        )
        or any(not path.is_file() or path.is_symlink() for path in bound_tests.values())
        or value.get("test_source_sha256")
        != {relative: sha256_file(source) for relative, source in bound_tests.items()}
        or value.get("design") != _audit_design()
        or value.get("test_command")
        != [
            context.spec["python_executable"], "-m", "pytest", "-q",
            *AUDITED_TEST_PATHS,
        ]
        or not isinstance(value.get("tests_passed"), int)
        or value["tests_passed"] <= 0
    ):
        raise FinalProtocolError("final-automation audit does not approve these exact bytes/design")
    return value


def _bound_file(context: FinalContext, value: str, field: str) -> Path:
    path = _resolve_bound_path(context.repository, value, field)
    try:
        path.resolve().relative_to(context.repository.resolve())
    except ValueError as error:
        raise FinalProtocolError(f"{field} escapes the registered repository") from error
    return path


def _state_repro_file(context: FinalContext, value: str, field: str) -> Path:
    path = _resolve_bound_path(context.repository, value, field)
    root = (context.state_dir / "reproducibility").resolve()
    try:
        path.resolve().relative_to(root)
    except ValueError as error:
        raise FinalProtocolError(f"{field} escapes the prepared reproducibility state") from error
    return path


def verify_nonprotected_bindings(context: FinalContext) -> dict[str, Any]:
    """Verify every public/dev input without touching the protected final manifest."""
    spec = context.spec
    bindings = {
        "registration": (
            _state_repro_file(context, spec["registration_path"], "registration_path"),
            spec["registration_sha256"],
        ),
        "recipe_lock": (
            _state_repro_file(context, spec["recipe_lock_path"], "recipe_lock_path"),
            spec["recipe_lock_sha256"],
        ),
        "science_registration": (
            _bound_file(
                context,
                spec["science_registration"]["path"],
                "science_registration.path",
            ),
            spec["science_registration"]["sha256"],
        ),
        "candidate_freeze": (
            _bound_file(
                context,
                spec["candidate_freeze"]["path"],
                "candidate_freeze.path",
            ),
            spec["candidate_freeze"]["sha256"],
        ),
        "candidate_freeze_correction": (
            _bound_file(
                context,
                spec["candidate_freeze_correction"]["path"],
                "candidate_freeze_correction.path",
            ),
            spec["candidate_freeze_correction"]["sha256"],
        ),
        "modal_execution": (
            _bound_file(
                context,
                spec["modal_execution"]["contract_path"],
                "modal_execution.contract_path",
            ),
            spec["modal_execution"]["contract_sha256"],
        ),
        "lens": (
            _bound_file(context, spec["artifacts"]["lens_path"], "artifacts.lens_path"),
            spec["artifacts"]["lens_sha256"],
        ),
        "calibration": (
            _bound_file(
                context,
                spec["artifacts"]["calibration_path"],
                "artifacts.calibration_path",
            ),
            spec["artifacts"]["calibration_sha256"],
        ),
        "metric_schema": (
            _bound_file(context, spec["metric_schema"]["path"], "metric_schema.path"),
            spec["metric_schema"]["sha256"],
        ),
    }
    for name, (path, digest) in bindings.items():
        if not path.is_file() or path.is_symlink() or sha256_file(path) != digest:
            raise FinalProtocolError(f"bound V10 {name} is absent, unsafe, or changed")
    registration = read_json(bindings["registration"][0])
    recipe_lock = read_json(bindings["recipe_lock"][0])
    if registration != expected_registration_document(spec):
        raise FinalProtocolError(
            "bound V10 registration does not commit to the exact science spec"
        )
    if recipe_lock != expected_recipe_lock_document(spec):
        raise FinalProtocolError("bound V10 recipe lock does not match the exact derivation")
    firewall = spec["firewall"]
    curve_path = _bound_file(
        context, firewall["curve_manifest"]["path"], "firewall.curve_manifest.path"
    )
    exclusions_path = _bound_file(
        context,
        firewall["train_exclusions"]["path"],
        "firewall.train_exclusions.path",
    )
    if (
        not curve_path.is_file()
        or curve_path.is_symlink()
        or not exclusions_path.is_file()
        or exclusions_path.is_symlink()
        or sha256_file(curve_path) != firewall["curve_manifest"]["sha256"]
        or sha256_file(exclusions_path) != firewall["train_exclusions"]["sha256"]
    ):
        raise FinalProtocolError("bound V10 development/firewall manifest changed")
    _, curve_indices = _manifest_payload(
        curve_path, expected_count=firewall["curve_manifest"]["count"]
    )
    _, excluded_indices = _manifest_payload(
        exclusions_path, expected_count=firewall["train_exclusions"]["count"]
    )
    if (
        any(index >= spec["dataset"]["size"] for index in curve_indices)
        or any(index >= spec["dataset"]["size"] for index in excluded_indices)
        or not set(curve_indices) <= set(excluded_indices)
    ):
        raise FinalProtocolError("V10 development/firewall manifest range or containment failed")
    receipt_binding = firewall["disjointness_receipt"]
    receipt_path = _state_repro_file(
        context, receipt_binding["path"], "firewall.disjointness_receipt.path"
    )
    if (
        not receipt_path.is_file()
        or receipt_path.is_symlink()
        or sha256_file(receipt_path) != receipt_binding["sha256"]
    ):
        raise FinalProtocolError("prospective final disjointness receipt is absent or changed")
    receipt = read_json(receipt_path)
    if (
        not isinstance(receipt, dict)
        or set(receipt)
        != {
            "schema_version", "protocol", "status",
            "protected_final_manifest_sha256", "curve_manifest_sha256",
            "train_exclusions_manifest_sha256", "protected_final_outcomes_read",
            "checks",
        }
        or receipt.get("schema_version") != 1
        or receipt.get("protocol") != spec["protocol"]
        or receipt.get("status") != "prospectively_verified_before_v10_final_unlock"
        or receipt.get("protected_final_manifest_sha256")
        != spec["final_collection"]["manifest_sha256"]
        or receipt.get("curve_manifest_sha256") != firewall["curve_manifest"]["sha256"]
        or receipt.get("train_exclusions_manifest_sha256")
        != firewall["train_exclusions"]["sha256"]
        or receipt.get("protected_final_outcomes_read") is not False
        or receipt.get("checks")
        != {
            "final_indices_disjoint_from_development_curve": True,
            "final_indices_in_training_exclusions": True,
            "development_curve_in_training_exclusions": True,
        }
    ):
        raise FinalProtocolError("prospective final disjointness receipt is ineligible")
    verify_bound_configs(context)
    return {
        "binding_sha256": {name: digest for name, (_path, digest) in bindings.items()},
        "curve_indices": curve_indices,
        "excluded_indices": excluded_indices,
        "disjointness_receipt_sha256": sha256_file(receipt_path),
        "registered_spec_projection_sha256": registered_spec_projection_sha256(spec),
        "protected_final_manifest_read": False,
    }


def _history_rows(path: Path) -> dict[int, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if [row.get("step") for row in rows] != list(CURVE_STEPS):
        raise FinalProtocolError(f"training history is not exact 0/2/3/4: {path}")
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        exact = row.get("exact_match")
        if (
            isinstance(exact, bool)
            or not isinstance(exact, (int, float))
            or not math.isfinite(float(exact))
            or not 0 <= float(exact) <= 1
        ):
            raise FinalProtocolError(f"training history has invalid exact match: {path}")
        result[int(row["step"])] = row
    return result


def _expected_log_scalar_keys(config: dict[str, Any]) -> set[str]:
    label = "_".join(config["target_words"])
    return {
        "clip_ratio/high_max", "clip_ratio/high_mean", "clip_ratio/low_mean",
        "clip_ratio/low_min", "clip_ratio/region_mean",
        "completions/clipped_ratio", "completions/max_length",
        "completions/max_terminated_length", "completions/mean_length",
        "completions/mean_terminated_length", "completions/min_length",
        "completions/min_terminated_length", "entropy", "epoch",
        "frac_reward_zero_std", "grad_norm", f"jlens/{label}_literal_rate",
        f"jlens/{label}_mean", "kl", "learning_rate", "loss", "num_tokens",
        "reward", "reward_std", f"rewards/jlens_{label}_reward/mean",
        f"rewards/jlens_{label}_reward/std", "step", "step_time", "total_flos",
        "train_loss", "train_runtime", "train_samples_per_second",
        "train_steps_per_second", "validation/exact_match",
        "validation/exact_match_ci95_high", "validation/exact_match_ci95_low",
        "validation/literal_target_completion_rate", "validation/mean_length",
    }


def _training_behavior_summary(
    path: Path,
    config: dict[str, Any],
    validation_history: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
        raise FinalProtocolError(f"training log history is not an exact row list: {path}")
    observed_scalar_keys = {
        key
        for row in payload
        for key, value in row.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    if observed_scalar_keys != _expected_log_scalar_keys(config):
        raise FinalProtocolError(f"training log scalar schema changed: {path}")
    label = "_".join(config["target_words"])
    literal_key = f"jlens/{label}_literal_rate"
    reward_mean_key = f"rewards/jlens_{label}_reward/mean"
    reward_std_key = f"rewards/jlens_{label}_reward/std"
    reward_rows = [row for row in payload if "reward" in row]
    if [row.get("step") for row in reward_rows] != list(
        range(1, int(config["updates"]) + 1)
    ):
        raise FinalProtocolError(f"training log does not contain exact optimizer steps: {path}")
    reward_bound = 5.0 * sum(
        abs(float(component["weight"])) for component in config["score_components"]
    )
    for row in reward_rows:
        reward_keys = [
            key for key in row if key.startswith("rewards/") and key.endswith("/mean")
        ]
        validation_merged = "validation/exact_match" in row
        numeric_fields = (
            "reward", "reward_std", "completions/mean_length",
            "completions/clipped_ratio", literal_key, reward_mean_key,
            reward_std_key,
        )
        if (
            reward_keys != [reward_mean_key]
            or any("gsm8k" in key.lower() for key in row)
            or any(
                isinstance(row.get(key), bool)
                or not isinstance(row.get(key), (int, float))
                or not math.isfinite(float(row[key]))
                for key in numeric_fields
            )
            or not -reward_bound <= float(row["reward"]) <= reward_bound
            or not -reward_bound <= float(row[reward_mean_key]) <= reward_bound
            or float(row["reward_std"]) < 0
            or float(row[reward_std_key]) < 0
            or not math.isclose(
                float(row["reward"]), float(row[reward_mean_key]),
                rel_tol=0.0, abs_tol=1e-12,
            )
            or not math.isclose(
                float(row["reward_std"]), float(row[reward_std_key]),
                rel_tol=0.0, abs_tol=1e-7,
            )
            or not 0 <= float(row[literal_key]) <= 1
            or not 0 <= float(row["completions/clipped_ratio"]) <= 1
            or not 0 <= float(row["completions/mean_length"])
            <= int(config["max_new_tokens"])
            or (
                "learning_rate" in row
                and (
                    isinstance(row.get("learning_rate"), bool)
                    or not isinstance(row.get("learning_rate"), (int, float))
                    or not math.isfinite(float(row["learning_rate"]))
                    or not math.isclose(
                        float(row["learning_rate"]),
                        float(config["learning_rate"]),
                        rel_tol=0.0,
                        abs_tol=1e-15,
                    )
                )
            )
            or (not validation_merged and "learning_rate" not in row)
        ):
            raise FinalProtocolError(f"training log has invalid one-J-reward behavior: {path}")
    validation_rows = [row for row in payload if "validation/exact_match" in row]
    if [row.get("step") for row in validation_rows] != list(CURVE_STEPS[1:]):
        raise FinalProtocolError(f"training log has a wrong validation sequence: {path}")
    for row in validation_rows:
        step = int(row["step"])
        fields = (
            "validation/exact_match", "validation/exact_match_ci95_high",
            "validation/exact_match_ci95_low",
            "validation/literal_target_completion_rate", "validation/mean_length",
        )
        if (
            any(
                isinstance(row.get(key), bool)
                or not isinstance(row.get(key), (int, float))
                or not math.isfinite(float(row[key]))
                for key in fields
            )
            or float(row["validation/exact_match"])
            != float(validation_history[step]["exact_match"])
            or not 0 <= float(row["validation/exact_match_ci95_low"])
            <= float(row["validation/exact_match"])
            <= float(row["validation/exact_match_ci95_high"]) <= 1
            or not 0 <= float(row["validation/literal_target_completion_rate"]) <= 1
            or not 0 <= float(row["validation/mean_length"])
            <= int(config["max_new_tokens"])
        ):
            raise FinalProtocolError(f"training log validation behavior changed: {path}")
    terminal_rows = [row for row in payload if "train_runtime" in row]
    terminal_fields = (
        "total_flos", "train_loss", "train_runtime",
        "train_samples_per_second", "train_steps_per_second",
    )
    if (
        len(terminal_rows) != 1
        or terminal_rows[0].get("step") != int(config["updates"])
        or any(
            isinstance(terminal_rows[0].get(key), bool)
            or not isinstance(terminal_rows[0].get(key), (int, float))
            or not math.isfinite(float(terminal_rows[0][key]))
            for key in terminal_fields
        )
        or any(
            float(terminal_rows[0][key]) < 0
            for key in terminal_fields
            if key != "train_loss"
        )
    ):
        raise FinalProtocolError(f"training log terminal summary changed: {path}")
    return {
        "optimizer_steps": len(reward_rows),
        "validation_steps": [int(row["step"]) for row in validation_rows],
        "literal_audit_key": literal_key,
        "literal_target_rate_max": max(float(row[literal_key]) for row in reward_rows),
        "reward_first": float(reward_rows[0]["reward"]),
        "reward_last": float(reward_rows[-1]["reward"]),
        "learning_rate_rows": sum("learning_rate" in row for row in payload),
    }


def _expected_wandb_identity(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "entity": config["wandb_entity"],
        "project": config["wandb_project"],
        "run_name": config["run_name"],
        "run_id": config["wandb_run_id"],
        "url": config["wandb_url"],
        "group": config["wandb_group"],
        "tags": config["wandb_tags"],
        "resume": "never",
    }


def _verify_offline_receipt_tree(
    directory: Path,
    receipt_path: Path,
    receipt: dict[str, Any],
    expected_identity: dict[str, Any],
    evidence_hashes: dict[str, str],
) -> None:
    raw_wandb_dir = receipt.get("wandb_dir")
    raw_root = receipt.get("offline_run_root")
    if (
        not isinstance(raw_wandb_dir, str)
        or not Path(raw_wandb_dir).is_absolute()
        or not isinstance(raw_root, str)
        or not Path(raw_root).is_absolute()
    ):
        raise FinalProtocolError("offline W&B receipt paths are not absolute")
    wandb_dir = Path(raw_wandb_dir)
    root = Path(raw_root)
    if (
        wandb_dir.is_symlink()
        or not wandb_dir.is_dir()
        or root.is_symlink()
        or not root.is_dir()
    ):
        raise FinalProtocolError("offline W&B receipt root is absent or symlinked")
    wandb_dir = wandb_dir.resolve(strict=True)
    root = root.resolve(strict=True)
    try:
        relative = root.relative_to(wandb_dir).as_posix()
    except ValueError as error:
        raise FinalProtocolError("offline W&B receipt root escaped WANDB_DIR") from error
    run_id = expected_identity["run_id"]
    if (
        raw_wandb_dir != str(wandb_dir)
        or raw_root != str(root)
        or receipt.get("offline_run_root_relative_to_wandb_dir") != relative
        or not root.name.startswith("offline-run-")
        or not root.name.endswith(f"-{run_id}")
    ):
        raise FinalProtocolError("offline W&B receipt root identity changed")
    try:
        receipt_path.resolve(strict=True).relative_to(root)
    except ValueError:
        pass
    else:
        raise FinalProtocolError("offline W&B receipt contaminated its own inventory")
    files_dir = root / "files"
    if files_dir.is_symlink() or not files_dir.is_dir():
        raise FinalProtocolError("offline W&B files directory is absent or symlinked")
    observed = receipt.get("observed_offline_identity")
    expected_observed = {
        "run_id": expected_identity["run_id"],
        "entity": expected_identity["entity"],
        "project": expected_identity["project"],
        "run_name": expected_identity["run_name"],
        "url": None,
        "group": expected_identity["group"],
        "tags": expected_identity["tags"],
        "resume": "never",
        "mode": "offline",
        "run_files_dir": str(files_dir.resolve()),
    }
    if observed != expected_observed:
        raise FinalProtocolError("observed offline W&B identity changed")
    for name, digest in evidence_hashes.items():
        embedded = files_dir / name
        if embedded.is_symlink() or not embedded.is_file() or sha256_file(embedded) != digest:
            raise FinalProtocolError(f"offline W&B embedded evidence changed: {name}")
    archive = root / f"run-{run_id}.wandb"
    if archive.is_symlink() or not archive.is_file():
        raise FinalProtocolError("completed offline W&B archive is absent")
    inventory = _offline_inventory(root)
    if (
        receipt.get("offline_run_file_symlink_inventory") != inventory
        or receipt.get("offline_run_file_symlink_count") != len(inventory)
        or receipt.get("offline_run_tree_sha256") != canonical_sha256(inventory)
        or receipt.get("sync_policy") != OFFLINE_SYNC_POLICY
    ):
        raise FinalProtocolError("offline W&B terminal tree/policy changed")
    try:
        directory.resolve(strict=True).relative_to(root)
    except ValueError:
        pass
    else:
        raise FinalProtocolError("run evidence directory is nested inside offline W&B root")


def _verify_training_process_command(
    context: FinalContext, value: Any, config: dict[str, Any]
) -> None:
    expected_entrypoint = (
        context.repository / context.spec["paths"]["training_entrypoint"]
    ).resolve()
    argv = value.get("argv") if isinstance(value, dict) else None
    if (
        not isinstance(value, dict)
        or set(value) != {"python_executable", "argv", "cwd"}
        or value.get("python_executable") != context.spec["python_executable"]
        or value.get("cwd") != str(context.repository.resolve())
        or not isinstance(argv, list)
        or not argv
        or (context.repository / str(argv[0])).resolve() != expected_entrypoint
        or argv[1:] != config["registered_command"][2:]
    ):
        raise FinalProtocolError("registered V10 training process command changed")


def _verify_one_completed_run(
    context: FinalContext,
    condition: str,
    seed: int,
    *,
    curve_indices: Sequence[int],
    excluded_indices: set[int],
) -> tuple[dict[str, Any], dict[int, dict[str, Any]], list[int]]:
    label = f"{condition}_seed{seed}"
    directory = context.run_dir / label
    config_path = context.config_dir / f"{label}.json"
    config = expected_training_config(context, condition, seed)
    files = {
        name: directory / name
        for name in (*TERMINAL_EVIDENCE_NAMES, "wandb_terminal_publish_receipt.json")
    }
    if any(not path.is_file() or path.is_symlink() for path in files.values()):
        raise FinalProtocolError(f"{label} lacks an exact terminal evidence inventory")
    if read_json(files["resolved_config.json"]) != config:
        raise FinalProtocolError(f"{label} resolved config differs from the bound config")
    manifest = read_json(files["run_manifest.json"])
    hardware = context.spec["hardware"]
    runtime = manifest.get("runtime", {})
    environment = read_json(files["environment_snapshot.json"])
    metric_path = _bound_file(
        context, context.spec["metric_schema"]["path"], "metric_schema.path"
    )
    expected_metric = {
        "path": str(metric_path.resolve()),
        "sha256": context.spec["metric_schema"]["sha256"],
        "content": read_json(metric_path),
    }
    expected_confirmatory_identity = {
        "registration_sha256": context.spec["registration_sha256"],
        "recipe_lock_sha256": context.spec["recipe_lock_sha256"],
        "recipe_sha256": context.spec["recipe_sha256"],
        "curve_manifest_sha256": context.spec["firewall"]["curve_manifest"]["sha256"],
        "train_exclusions_manifest_sha256": context.spec["firewall"]["train_exclusions"][
            "sha256"
        ],
        "registered_code_sha256": context.spec["registered_code_sha256"],
    }
    _verify_training_process_command(context, manifest.get("process_command"), config)
    if (
        manifest.get("git_commit") != context.spec["git_commit"]
        or manifest.get("git_dirty") is not False
        or manifest.get("source_tree_sha256") != context.spec["source_tree_sha256"]
        or manifest.get("config_sha256") != sha256_file(config_path)
        or manifest.get("resolved_config_sha256") != sha256_file(files["resolved_config.json"])
        or manifest.get("config_path") != str(config_path.resolve())
        or manifest.get("model_name") != context.spec["model"]["name"]
        or manifest.get("model_revision") != context.spec["model"]["revision"]
        or manifest.get("dataset") != "openai/gsm8k:main"
        or manifest.get("dataset_revision") != context.spec["dataset"]["revision"]
        or manifest.get("lens_sha256") != context.spec["artifacts"]["lens_sha256"]
        or manifest.get("calibration_sha256") != context.spec["artifacts"]["calibration_sha256"]
        or manifest.get("reward_type") != "jlens"
        or manifest.get("registered_command") != config["registered_command"]
        or manifest.get("evidence_eligibility")
        != "original_registered_v10_modal_attempt"
        or manifest.get("reproduction_source") is not None
        or manifest.get("confirmatory_identity") != expected_confirmatory_identity
        or manifest.get("metric_schema") != expected_metric
        or manifest.get("wandb_identity") != _expected_wandb_identity(config)
        or runtime.get("cuda_device_name") != hardware["device_name"]
        or runtime.get("torch_version") != hardware["torch_version"]
        or runtime.get("cuda_version") != hardware["cuda_version"]
        or runtime.get("environment_snapshot_path") != "environment_snapshot.json"
        or runtime.get("environment_snapshot_sha256")
        != sha256_file(files["environment_snapshot.json"])
        or runtime.get("environment_snapshot") != environment
        or environment.get("python", {}).get("executable")
        != context.spec["python_executable"]
        or environment.get("torch", {}).get("version") != hardware["torch_version"]
        or environment.get("torch", {}).get("cuda_build") != hardware["cuda_version"]
        or environment.get("pip_freeze_all")
        != sorted(environment.get("pip_freeze_all", []))
        or not environment.get("pip_freeze_all")
        or hardware["device_name"] not in environment.get("cuda_device_names", [])
        or not any(
            hardware["device_name"] in line and hardware["driver_version"] in line
            for line in environment.get("nvidia_smi_name_and_driver", [])
        )
        or len(environment.get("nvidia_smi_uuid_name_and_driver", [])) != 1
        or not environment["nvidia_smi_uuid_name_and_driver"][0].startswith("GPU-")
        or hardware["device_name"]
        not in environment["nvidia_smi_uuid_name_and_driver"][0]
        or hardware["driver_version"]
        not in environment["nvidia_smi_uuid_name_and_driver"][0]
    ):
        raise FinalProtocolError(f"{label} source/config/reward/hardware identity changed")
    data = read_json(files["data_indices.json"])
    train_indices = data.get("train_source_indices")
    if (
        not isinstance(train_indices, list)
        or len(train_indices) != context.spec["training"]["train_examples"]
        or len(set(train_indices)) != len(train_indices)
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < context.spec["dataset"]["size"]
            for index in train_indices
        )
        or set(train_indices) & excluded_indices
        or data.get("validation_source") != "train"
        or data.get("validation_source_indices") != list(curve_indices)
        or manifest.get("data_indices_sha256") != sha256_file(files["data_indices.json"])
    ):
        raise FinalProtocolError(f"{label} training/development firewall changed")
    history = _history_rows(files["validation_history.jsonl"])
    if any(
        row.get("validation_source") != "train"
        or row.get("validation_indices_sha256")
        != context.spec["firewall"]["curve_manifest"]["sha256"]
        for row in history.values()
    ):
        raise FinalProtocolError(f"{label} history used a wrong development curve")
    training_behavior = _training_behavior_summary(
        files["log_history.json"], config, history
    )
    result = read_json(files["run_result_manifest.json"])
    final_tree = _tree_identity(directory / "final")
    checkpoint_tree = _tree_identity(directory / f"checkpoint-{TERMINAL_STEP}")
    raw_hashes = {
        name: sha256_file(files[name])
        for name in TERMINAL_EVIDENCE_NAMES
        if name != "run_result_manifest.json"
    }
    if (
        result.get("schema_version") != 1
        or result.get("completed_updates") != TERMINAL_STEP
        or result.get("registration_sha256") != context.spec["registration_sha256"]
        or result.get("recipe_lock_sha256") != context.spec["recipe_lock_sha256"]
        or result.get("recipe_sha256") != context.spec["recipe_sha256"]
        or result.get("registered_command") != config["registered_command"]
        or result.get("process_command") != manifest.get("process_command")
        or result.get("metric_schema") != expected_metric
        or result.get("source")
        != {
            "git_commit": context.spec["git_commit"],
            "git_dirty": False,
            "source_tree_sha256": context.spec["source_tree_sha256"],
        }
        or result.get("runtime") != runtime
        or result.get("data_indices_sha256") != sha256_file(files["data_indices.json"])
        or result.get("lens_sha256") != context.spec["artifacts"]["lens_sha256"]
        or result.get("calibration_sha256") != context.spec["artifacts"]["calibration_sha256"]
        or result.get("raw_history_sha256") != raw_hashes
        or result.get("terminal_checkpoint") != checkpoint_tree
        or result.get("final_adapter_and_tokenizer") != final_tree
        or result.get("wandb_identity") != _expected_wandb_identity(config)
        or result.get("evidence_eligibility") != "original_registered_v10_modal_attempt"
        or result.get("reproduction_source") is not None
    ):
        raise FinalProtocolError(f"{label} run-result identity changed")
    receipt = read_json(files["wandb_terminal_publish_receipt.json"])
    evidence_hashes = {name: sha256_file(files[name]) for name in TERMINAL_EVIDENCE_NAMES}
    expected_identity = _expected_wandb_identity(config)
    expected_observed = {
        key: expected_identity[key]
        for key in ("run_id", "entity", "project", "run_name", "url", "group", "tags")
    }
    artifact = receipt.get("artifact", {}) if isinstance(receipt, dict) else {}
    version = artifact.get("version") if isinstance(artifact, dict) else None
    artifact_base = f"{expected_identity['run_id']}-terminal-evidence"
    if (
        not isinstance(receipt, dict)
        or set(receipt)
        != {
            "schema_version", "wandb_identity", "observed_wandb_identity",
            "artifact", "terminal_run_result_sha256", "uploaded_file_sha256",
        }
        or receipt.get("schema_version") != 2
        or receipt.get("wandb_identity") != expected_identity
        or receipt.get("observed_wandb_identity") != expected_observed
        or receipt.get("terminal_run_result_sha256")
        != evidence_hashes["run_result_manifest.json"]
        or receipt.get("uploaded_file_sha256") != evidence_hashes
        or not isinstance(artifact, dict)
        or set(artifact) != {"id", "name", "version", "digest", "qualified_name"}
        or not isinstance(artifact.get("id"), str)
        or not artifact["id"]
        or not isinstance(artifact.get("digest"), str)
        or not artifact["digest"]
        or not isinstance(version, str)
        or re.fullmatch(r"v[0-9]+", version) is None
        or artifact.get("name") != f"{artifact_base}:{version}"
        or artifact.get("qualified_name")
        != (
            f"{expected_identity['entity']}/{expected_identity['project']}/"
            f"{artifact_base}:{version}"
        )
    ):
        raise FinalProtocolError(f"{label} terminal W&B receipt identity changed")
    record = {
        "label": label,
        "condition": condition,
        "seed": seed,
        "config_sha256": sha256_file(config_path),
        "run_result_sha256": sha256_file(files["run_result_manifest.json"]),
        "receipt_sha256": sha256_file(files["wandb_terminal_publish_receipt.json"]),
        "validation_history_sha256": sha256_file(files["validation_history.jsonl"]),
        "log_history_sha256": sha256_file(files["log_history.json"]),
        "training_behavior": training_behavior,
        "data_indices_sha256": sha256_file(files["data_indices.json"]),
        "terminal_adapter": final_tree,
        "hardware": hardware,
        "source_tree_sha256": context.spec["source_tree_sha256"],
    }
    return record, history, list(train_indices)


def verify_completed_inventory(context: FinalContext, firewall: dict[str, Any]) -> dict[str, Any]:
    if not context.completed_runs_path.is_file():
        raise FinalProtocolError("V10 completed-run inventory is absent")
    observed_run_labels = {
        path.name
        for path in context.run_dir.iterdir()
        if path.is_dir() and not path.is_symlink()
    } if context.run_dir.is_dir() else set()
    if (
        context.run_dir.is_symlink()
        or observed_run_labels != set(FINAL_LABELS[1:])
        or any(path.is_file() or path.is_symlink() for path in context.run_dir.iterdir())
    ):
        raise FinalProtocolError("V10 run directory is not the exact 8-label inventory")
    stored = read_json(context.completed_runs_path)
    expected_runs: dict[str, Any] = {}
    histories: dict[str, dict[int, dict[str, Any]]] = {}
    matched_train: dict[int, list[int]] = {}
    for condition in CONDITIONS:
        for seed in SEEDS:
            record, history, train_indices = _verify_one_completed_run(
                context,
                condition,
                seed,
                curve_indices=firewall["curve_indices"],
                excluded_indices=set(firewall["excluded_indices"]),
            )
            label = record["label"]
            expected_runs[label] = record
            histories[label] = history
            if seed in matched_train and matched_train[seed] != train_indices:
                raise FinalProtocolError(f"treatment/signflip train rows differ for seed {seed}")
            matched_train[seed] = train_indices
    expected = {
        "schema_version": 1,
        "protocol": context.spec["protocol"],
        "git_commit": context.spec["git_commit"],
        "registration_sha256": context.spec["registration_sha256"],
        "recipe_lock_sha256": context.spec["recipe_lock_sha256"],
        "recipe_sha256": context.spec["recipe_sha256"],
        "registered_code_sha256": context.spec["registered_code_sha256"],
        "registered_spec_projection_sha256": registered_spec_projection_sha256(
            context.spec
        ),
        "seeds": list(SEEDS),
        "conditions": list(CONDITIONS),
        "terminal_step": TERMINAL_STEP,
        "hardware": context.spec["hardware"],
        "source_tree_sha256": context.spec["source_tree_sha256"],
        "runs": expected_runs,
    }
    if stored != expected or set(stored.get("runs", {})) != set(FINAL_LABELS[1:]):
        raise FinalProtocolError("V10 completed inventory is not the exact 8 registered runs")
    return {"stored": stored, "histories": histories, "runs": expected_runs}


def verify_preunlock_readiness(context: FinalContext) -> dict[str, Any]:
    """Audit-only check that intentionally does not read a final manifest."""
    _guard_not_failed(context)
    validate_spec(context.spec)
    audit = verify_automation_audit(context)
    bindings = verify_nonprotected_bindings(context)
    return {
        "ready_for_separate_training_protocol_to_unlock": True,
        "audit_sha256": sha256_file(
            _resolve_bound_path(
                context.repository,
                context.spec["automation_audit"]["path"],
                "automation_audit.path",
            )
        ),
        "design": _audit_design(),
        "protected_payloads_accessed": audit["protected_payloads_accessed"],
        "protected_final_manifest_read": bindings["protected_final_manifest_read"],
    }


def _verify_curve(context: FinalContext, completed: dict[str, Any]) -> dict[str, Any]:
    if not context.curve_path.is_file():
        raise FinalProtocolError("registered four-seed curve gate is absent")
    value = read_json(context.curve_path)
    per_seed = {
        str(seed): {
            str(step): float(completed["histories"][f"jlens_seed{seed}"][step]["exact_match"])
            for step in CURVE_STEPS
        }
        for seed in SEEDS
    }
    means = {
        str(step): sum(per_seed[str(seed)][str(step)] for seed in SEEDS) / len(SEEDS)
        for step in CURVE_STEPS
    }
    ordered = [means[str(step)] for step in CURVE_STEPS]
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "steps", "criterion", "n_seeds", "per_seed_exact_match",
            "mean_exact_match", "passed",
        }
        or value.get("steps") != list(CURVE_STEPS)
        or value.get("criterion") != CURVE_CRITERION
        or value.get("n_seeds") != len(SEEDS)
        or value.get("per_seed_exact_match") != per_seed
        or value.get("mean_exact_match") != means
        or value.get("passed") is not True
        or not curve_means_pass(ordered)
    ):
        raise FinalProtocolError("registered four-seed curve gate did not pass exactly")
    return value


def _verify_unlock(context: FinalContext) -> dict[str, Any]:
    _guard_not_failed(context)
    if not context.unlock_path.is_file() or not context.completed_runs_path.is_file():
        raise FinalProtocolError("future V10 final is not unlocked by the training protocol")
    audit = verify_automation_audit(context)
    firewall = verify_nonprotected_bindings(context)
    completed = verify_completed_inventory(context, firewall)
    curve = _verify_curve(context, completed)
    unlock = read_json(context.unlock_path)
    if (
        not isinstance(unlock, dict)
        or set(unlock)
        != {
            "protocol", "git_commit", "registration_sha256", "curve_gate_sha256",
            "completed_runs_sha256", "automation_audit_sha256",
            "recipe_lock_sha256", "final_manifest_sha256",
            "disjointness_receipt_sha256", "recipe_sha256",
            "registered_code_sha256", "registered_spec_projection_sha256",
        }
        or unlock.get("protocol") != context.spec["protocol"]
        or unlock.get("git_commit") != context.spec["git_commit"]
        or unlock.get("registration_sha256") != context.spec["registration_sha256"]
        or unlock.get("curve_gate_sha256") != sha256_file(context.curve_path)
        or unlock.get("completed_runs_sha256") != sha256_file(context.completed_runs_path)
        or unlock.get("automation_audit_sha256")
        != context.spec["automation_audit"]["sha256"]
        or unlock.get("recipe_lock_sha256") != context.spec["recipe_lock_sha256"]
        or unlock.get("recipe_sha256") != context.spec["recipe_sha256"]
        or unlock.get("registered_code_sha256")
        != context.spec["registered_code_sha256"]
        or unlock.get("registered_spec_projection_sha256")
        != registered_spec_projection_sha256(context.spec)
        or unlock.get("final_manifest_sha256")
        != context.spec["final_collection"]["manifest_sha256"]
        or unlock.get("disjointness_receipt_sha256")
        != context.spec["firewall"]["disjointness_receipt"]["sha256"]
    ):
        raise FinalProtocolError("future V10 unlock is not bound to all 8 exact runs/audit")
    return {
        "unlock": unlock,
        "completed": completed,
        "curve": curve,
        "audit": audit,
        "firewall": firewall,
    }


def _guard_not_failed(context: FinalContext) -> None:
    if context.failure_path.exists():
        raise FinalProtocolError("V10 final collection previously failed; continuation is forbidden")


def record_final_failure(context: FinalContext, value: dict[str, Any]) -> None:
    if context.failure_path.exists():
        return
    forbidden = {"schema_version", "protocol", "stage", "retry_or_resume_permitted"}
    if set(value) & forbidden:
        raise FinalProtocolError("failure detail attempted to replace fixed terminal fields")
    write_json_exclusive(
        context.failure_path,
        {
            **value,
            "schema_version": 1,
            "protocol": context.spec["protocol"],
            "stage": "final_collection_failed_closed_allocation_spent",
            "retry_or_resume_permitted": False,
        },
    )


def _active_repository_provenance(context: FinalContext) -> dict[str, Any]:
    from jlens_rl.common import repository_provenance

    return repository_provenance(context.repository)


def verify_preclaim(context: FinalContext) -> dict[str, Any]:
    _guard_not_failed(context)
    bound = _verify_unlock(context)
    provenance = _active_repository_provenance(context)
    if (
        provenance.get("git_commit") != context.spec["git_commit"]
        or provenance.get("git_dirty") is not False
        or provenance.get("source_tree_sha256") != context.spec["source_tree_sha256"]
        or bound["audit"].get("audited_commit") != context.spec["git_commit"]
    ):
        raise FinalProtocolError("one-shot final claim requires the exact clean audited Git tree")
    return {**bound, "active_repository_provenance": provenance}


def _final_manifest_path(context: FinalContext) -> Path:
    return _resolve_bound_path(
        context.repository,
        context.spec["final_collection"]["manifest_path"],
        "final_collection.manifest_path",
    )


def _open_and_validate_final_manifest(
    context: FinalContext, firewall: dict[str, Any]
) -> list[int]:
    manifest = _final_manifest_path(context)
    if (
        not manifest.is_file()
        or manifest.is_symlink()
        or sha256_file(manifest) != context.spec["final_collection"]["manifest_sha256"]
    ):
        raise FinalProtocolError("protected final manifest is absent, unsafe, or changed")
    payload, indices = _manifest_payload(manifest, expected_count=FINAL_EXAMPLES)
    metadata = {key: payload[key] for key in ("dataset", "subset", "split")}
    if (
        metadata != context.spec["final_collection"]["manifest_metadata"]
        or any(index >= context.spec["dataset"]["size"] for index in indices)
        or set(indices) & set(firewall["curve_indices"])
        or not set(indices) <= set(firewall["excluded_indices"])
    ):
        raise FinalProtocolError("protected final metadata/range/disjointness validation failed")
    return indices


def begin_final_collection(
    context: FinalContext,
    collection_id: str,
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{32}", collection_id) is None:
        raise FinalProtocolError("final collection ID must be 32 lowercase hex characters")
    bound = verify_preclaim(context)
    forbidden = (
        context.collection_path,
        context.eval_dir,
        context.comparison_path,
        context.analysis_process_path,
        context.acceptance_path,
        context.failure_path,
    )
    if any(path.exists() for path in forbidden):
        raise FinalProtocolError("final collection has already started or left an outcome-bearing trace")
    marker = {
        "schema_version": 1,
        "protocol": context.spec["protocol"],
        "git_commit": context.spec["git_commit"],
        "registration_sha256": context.spec["registration_sha256"],
        "registered_spec_projection_sha256": registered_spec_projection_sha256(
            context.spec
        ),
        "collection_id": collection_id,
        "labels": list(FINAL_LABELS),
        "sealed_manifest_sha256": context.spec["final_collection"]["manifest_sha256"],
        "sealed_eval_config_sha256": context.spec["config_sha256"]["sealed_eval"],
        "unlock_sha256": sha256_file(context.unlock_path),
        "automation_audit_sha256": context.spec["automation_audit"]["sha256"],
        "active_repository_provenance": {
            key: bound["active_repository_provenance"][key]
            for key in ("git_commit", "git_dirty", "source_tree_sha256")
        },
        "collection_policy": COLLECTION_POLICY,
    }
    write_json_exclusive(context.collection_path, marker)
    try:
        indices = _open_and_validate_final_manifest(context, bound["firewall"])
    except BaseException as error:
        record_final_failure(
            context,
            {
                "collection_id": collection_id,
                "error_type": type(error).__name__,
                "error": str(error),
                "failure_phase": "protected_manifest_open_after_one_shot_claim",
            },
        )
        raise
    return {
        **marker,
        "preconditions": {"curve_passed": bound["curve"]["passed"]},
        "opened_final_manifest_count": len(indices),
    }


def verify_final_collection(context: FinalContext, collection_id: str | None = None) -> dict[str, Any]:
    _guard_not_failed(context)
    if not context.collection_path.is_file():
        raise FinalProtocolError("future V10 final collection has not been claimed")
    marker = read_json(context.collection_path)
    bound = _verify_unlock(context)
    expected = {
        "schema_version": 1,
        "protocol": context.spec["protocol"],
        "git_commit": context.spec["git_commit"],
        "registration_sha256": context.spec["registration_sha256"],
        "registered_spec_projection_sha256": registered_spec_projection_sha256(
            context.spec
        ),
        "labels": list(FINAL_LABELS),
        "sealed_manifest_sha256": context.spec["final_collection"]["manifest_sha256"],
        "sealed_eval_config_sha256": context.spec["config_sha256"]["sealed_eval"],
        "unlock_sha256": sha256_file(context.unlock_path),
        "automation_audit_sha256": context.spec["automation_audit"]["sha256"],
        "active_repository_provenance": {
            "git_commit": context.spec["git_commit"],
            "git_dirty": False,
            "source_tree_sha256": context.spec["source_tree_sha256"],
        },
        "collection_policy": COLLECTION_POLICY,
    }
    if (
        set(marker) != {*expected, "collection_id"}
        or any(marker.get(key) != value for key, value in expected.items())
        or re.fullmatch(r"[0-9a-f]{32}", str(marker.get("collection_id"))) is None
    ):
        raise FinalProtocolError("future V10 final collection marker changed")
    if collection_id is not None and marker.get("collection_id") != collection_id:
        raise FinalProtocolError("future V10 final collection ID changed")
    try:
        _open_and_validate_final_manifest(context, bound["firewall"])
    except BaseException as error:
        record_final_failure(
            context,
            {
                "collection_id": marker["collection_id"],
                "error_type": type(error).__name__,
                "error": str(error),
                "failure_phase": "protected_manifest_revalidation_after_claim",
            },
        )
        raise
    return marker


@dataclass(frozen=True)
class ReferenceBundle:
    indices: list[int]
    dataset_fingerprint: str | None
    prompt_sha256: dict[int, str]
    prompt_token_ids_sha256: dict[int, str]
    answers: dict[int, str]
    decode_completion: Callable[[list[int]], str]
    extract_answer: Callable[[str], str | None]
    is_correct: Callable[[str, str], bool]
    literal_matches: Callable[[str, Sequence[str]], list[str]]


def load_production_references(context: FinalContext) -> ReferenceBundle:
    """Load protected references only after a valid final collection claim."""
    verify_final_collection(context)
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer

        from jlens_rl.common import extract_answer, format_prompt, gsm8k_reward
        from jlens_rl.paired_eval import literal_target_matches
    except ImportError as error:
        raise FinalProtocolError("final verification needs the pinned project environment") from error
    _, indices = _manifest_payload(
        _final_manifest_path(context),
        expected_count=FINAL_EXAMPLES,
    )
    dataset_spec = context.spec["dataset"]
    model_spec = context.spec["model"]
    dataset = load_dataset(
        dataset_spec["name"],
        dataset_spec["subset"],
        split=dataset_spec["split"],
        revision=dataset_spec["revision"],
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_spec["name"], revision=model_spec["revision"]
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    eval_config = read_json(context.config_dir / "sealed_eval.json")
    prompt_hashes: dict[int, str] = {}
    token_hashes: dict[int, str] = {}
    answers: dict[int, str] = {}
    for source_index in indices:
        row = dataset[source_index]
        prompt = format_prompt(tokenizer, row["question"])
        tokens = tokenizer(
            prompt,
            truncation=True,
            max_length=int(eval_config["max_prompt_tokens"]),
        )["input_ids"]
        prompt_hashes[source_index] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        token_hashes[source_index] = canonical_sha256(tokens)
        answers[source_index] = row["answer"]
    return ReferenceBundle(
        indices=indices,
        dataset_fingerprint=getattr(dataset, "_fingerprint", None),
        prompt_sha256=prompt_hashes,
        prompt_token_ids_sha256=token_hashes,
        answers=answers,
        decode_completion=lambda tokens: tokenizer.decode(tokens, skip_special_tokens=True),
        extract_answer=extract_answer,
        is_correct=lambda completion, answer: bool(gsm8k_reward(completion, answer)),
        literal_matches=literal_target_matches,
    )


def _contains_forbidden_gold_key(value: Any) -> bool:
    forbidden = {"answer", "gold", "gold_answer", "reference", "reference_answer"}
    if isinstance(value, dict):
        return any(
            str(key).lower() in forbidden or _contains_forbidden_gold_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_gold_key(item) for item in value)
    return False


def _adapter_identity(context: FinalContext, label: str) -> dict[str, Any]:
    path = context.run_dir / label / "final"
    files = sorted({*path.glob("adapter_config.json"), *path.glob("adapter_model*")})
    if not files:
        raise FinalProtocolError(f"terminal adapter is absent for {label}")
    hashes = {file.name: sha256_file(file) for file in files if file.is_file()}
    try:
        recorded = path.resolve().relative_to(context.repository.resolve()).as_posix()
    except ValueError:
        recorded = str(path.resolve())
    return {"path": recorded, "sha256": canonical_sha256(hashes), "files": hashes}


def _expected_eval_arguments(context: FinalContext, label: str) -> list[str]:
    condition, seed, is_base = evaluation_role(label)
    arguments = [
        "--config",
        str(context.config_dir / "sealed_eval.json"),
        "--experiment-config",
        str(context.config_dir / f"{condition}_seed{seed}.json"),
        "--indices-manifest",
        str(_final_manifest_path(context)),
        "--output-jsonl",
        str(context.eval_dir / f"{label}.jsonl"),
        "--run-label",
        label,
        "--batch-size",
        "64",
        "--skip-jlens-metric",
    ]
    if not is_base:
        arguments.extend(["--adapter", str(context.run_dir / label / "final")])
    return arguments


def expected_eval_command(context: FinalContext, label: str) -> list[str]:
    return [
        context.spec["python_executable"],
        "-m",
        "jlens_rl.eval",
        *_expected_eval_arguments(context, label),
    ]


def expected_runtime_overrides(context: FinalContext) -> dict[str, str]:
    return {
        "PYTHONPATH": str((context.repository / "src").resolve()),
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
        "JLENS_REPOSITORY_ROOT": str(context.repository.resolve()),
    }


def _verify_process_command(context: FinalContext, process: Any, label: str) -> None:
    if not isinstance(process, dict):
        raise FinalProtocolError("final evaluation lacks process provenance")
    argv = process.get("argv")
    expected_module = (context.repository / "src" / "jlens_rl" / "eval.py").resolve()
    if (
        process.get("python_executable") != context.spec["python_executable"]
        or process.get("cwd") != str(context.repository.resolve())
        or not isinstance(argv, list)
        or not argv
        or Path(argv[0]).resolve() != expected_module
        or sha256_file(expected_module)
        != verify_automation_audit(context)["source_sha256"]["eval"]
        or argv[1:] != _expected_eval_arguments(context, label)
    ):
        raise FinalProtocolError("final evaluation command changed")


def _verify_dispatch(context: FinalContext, label: str) -> None:
    dispatch_dir = context.evidence_dir / "final_dispatches"
    intent_path = dispatch_dir / f"{label}.intent.json"
    completion_path = dispatch_dir / f"{label}.completion.json"
    if not intent_path.is_file() or not completion_path.is_file():
        raise FinalProtocolError(f"final evaluation dispatch receipt is absent for {label}")
    intent = read_json(intent_path)
    completion = read_json(completion_path)
    observed_hardware = intent.get("hardware", {}) if isinstance(intent, dict) else {}
    expected_hardware = {
        **context.spec["hardware"],
        "observed_gpu_uuid": observed_hardware.get("observed_gpu_uuid"),
    }
    sequence = FINAL_LABELS.index(label) + 1
    collection_id = read_json(context.collection_path)["collection_id"]
    stdout_path = context.evidence_dir / "sealed_collection_logs" / f"{label}.stdout"
    stderr_path = context.evidence_dir / "sealed_collection_logs" / f"{label}.stderr"
    if (
        set(intent)
        != {
            "schema_version", "protocol", "collection_id", "sequence", "label",
            "hardware", "command", "cwd", "environment_overrides", "status",
            "outcome_inspected_before_full_collection",
        }
        or intent.get("schema_version") != 1
        or intent.get("protocol") != context.spec["protocol"]
        or intent.get("label") != label
        or intent.get("sequence") != sequence
        or observed_hardware != expected_hardware
        or not isinstance(observed_hardware.get("observed_gpu_uuid"), str)
        or not observed_hardware["observed_gpu_uuid"].startswith("GPU-")
        or intent.get("collection_id") != collection_id
        or intent.get("command") != expected_eval_command(context, label)
        or intent.get("cwd") != str(context.repository.resolve())
        or intent.get("environment_overrides") != expected_runtime_overrides(context)
        or intent.get("status") != "written_and_fsynced_before_gpu_process"
        or intent.get("outcome_inspected_before_full_collection") is not False
        or set(completion)
        != {
            "schema_version", "protocol", "collection_id", "sequence", "label",
            "intent_sha256", "jsonl_sha256", "environment_sha256", "stdout_sha256",
            "stderr_sha256", "returncode", "outcome_inspected_before_full_collection",
            "command", "cwd", "environment_overrides",
        }
        or completion.get("schema_version") != 1
        or completion.get("protocol") != context.spec["protocol"]
        or completion.get("label") != label
        or completion.get("sequence") != sequence
        or completion.get("collection_id") != collection_id
        or completion.get("intent_sha256") != sha256_file(intent_path)
        or completion.get("jsonl_sha256")
        != sha256_file(context.eval_dir / f"{label}.jsonl")
        or completion.get("environment_sha256")
        != sha256_file(context.eval_dir / f"{label}.environment.json")
        or not stdout_path.is_file()
        or not stderr_path.is_file()
        or completion.get("stdout_sha256") != sha256_file(stdout_path)
        or completion.get("stderr_sha256") != sha256_file(stderr_path)
        or completion.get("returncode") != 0
        or completion.get("outcome_inspected_before_full_collection") is not False
        or completion.get("command") != expected_eval_command(context, label)
        or completion.get("cwd") != str(context.repository.resolve())
        or completion.get("environment_overrides") != expected_runtime_overrides(context)
    ):
        raise FinalProtocolError(f"final evaluation dispatch changed for {label}")


def verify_dispatch_inventory(context: FinalContext) -> dict[str, str]:
    dispatch_dir = context.evidence_dir / "final_dispatches"
    log_dir = context.evidence_dir / "sealed_collection_logs"
    expected_dispatch = {
        *(f"{label}.intent.json" for label in FINAL_LABELS),
        *(f"{label}.completion.json" for label in FINAL_LABELS),
    }
    expected_logs = {
        *(f"{label}.stdout" for label in FINAL_LABELS),
        *(f"{label}.stderr" for label in FINAL_LABELS),
    }
    dispatch_entries = list(dispatch_dir.iterdir()) if dispatch_dir.is_dir() else []
    log_entries = list(log_dir.iterdir()) if log_dir.is_dir() else []
    observed_dispatch = {path.name for path in dispatch_entries if path.is_file()}
    observed_logs = {path.name for path in log_entries if path.is_file()}
    if (
        dispatch_dir.is_symlink()
        or log_dir.is_symlink()
        or observed_dispatch != expected_dispatch
        or observed_logs != expected_logs
        or any(path.is_symlink() or not path.is_file() for path in dispatch_entries)
        or any(path.is_symlink() or not path.is_file() for path in log_entries)
    ):
        raise FinalProtocolError("final dispatch/log inventory is not exact")
    for label in FINAL_LABELS:
        _verify_dispatch(context, label)
    return {
        path.relative_to(context.state_dir).as_posix(): sha256_file(path)
        for directory in (dispatch_dir, log_dir)
        for path in sorted(directory.iterdir())
    }


def verify_evaluation_jsonl(
    context: FinalContext,
    label: str,
    *,
    references: ReferenceBundle | None = None,
) -> None:
    verify_final_collection(context)
    condition, seed, is_base = evaluation_role(label)
    path = context.eval_dir / f"{label}.jsonl"
    if not path.is_file():
        raise FinalProtocolError(f"final evaluation JSONL is absent for {label}")
    references = references or load_production_references(context)
    _, expected_indices = _manifest_payload(
        _final_manifest_path(context),
        expected_count=FINAL_EXAMPLES,
    )
    if references.indices != expected_indices:
        raise FinalProtocolError("reference bundle does not match the final manifest")
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(records) != FINAL_EXAMPLES:
        raise FinalProtocolError(f"{label} has {len(records)} final rows, expected 900")
    if [record.get("source_index") for record in records] != expected_indices:
        raise FinalProtocolError(f"{label} final rows are missing, duplicated, or reordered")
    eval_config_path = context.config_dir / "sealed_eval.json"
    eval_config = read_json(eval_config_path)
    experiment_config_path = context.config_dir / f"{condition}_seed{seed}.json"
    experiment_config = read_json(experiment_config_path)
    expected_generation = {
        "do_sample": False,
        "max_prompt_tokens": int(eval_config["max_prompt_tokens"]),
        "max_new_tokens": int(eval_config["max_new_tokens"]),
        "padding_side": "left",
    }
    expected_dataset = {
        "name": context.spec["dataset"]["name"],
        "subset": context.spec["dataset"]["subset"],
        "split": context.spec["dataset"]["split"],
        "revision": context.spec["dataset"]["revision"],
        "fingerprint": references.dataset_fingerprint,
    }
    target_words = context.spec["target_words"]
    for line_number, record in enumerate(records, 1):
        if (
            not isinstance(record, dict)
            or record.get("schema_version") != 1
            or not isinstance(record.get("correct"), bool)
            or _contains_forbidden_gold_key(record)
            or record.get("dataset") != expected_dataset
            or record.get("target_words") != target_words
            or record.get("generation") != expected_generation
        ):
            raise FinalProtocolError(f"invalid final record schema/provenance at {label}:{line_number}")
        source_index = record["source_index"]
        completion = record.get("completion")
        token_ids = record.get("completion_token_ids")
        if (
            record.get("prompt_sha256") != references.prompt_sha256[source_index]
            or record.get("prompt_token_ids_sha256")
            != references.prompt_token_ids_sha256[source_index]
            or not isinstance(completion, str)
            or not isinstance(token_ids, list)
            or any(isinstance(token, bool) or not isinstance(token, int) for token in token_ids)
            or references.decode_completion(token_ids) != completion
            or record.get("prediction") != references.extract_answer(completion)
            or record.get("correct")
            is not references.is_correct(completion, references.answers[source_index])
            or isinstance(record.get("completion_tokens"), bool)
            or not isinstance(record.get("completion_tokens"), int)
            or record.get("completion_tokens") != len(token_ids)
            or not 0 <= len(token_ids) <= expected_generation["max_new_tokens"]
        ):
            raise FinalProtocolError(f"derived completion outcome changed at {label}:{line_number}")
        matches = references.literal_matches(completion, target_words)
        if (
            record.get("literal_target_matches") != matches
            or record.get("literal_target_used") is not bool(matches)
        ):
            raise FinalProtocolError(f"literal-target audit changed at {label}:{line_number}")
    provenance = records[0].get("provenance")
    if not isinstance(provenance, dict) or any(
        record.get("provenance") != provenance for record in records
    ):
        raise FinalProtocolError(f"{label} final provenance is not constant")
    _verify_process_command(context, provenance.get("process_command"), label)
    environment_path = context.eval_dir / f"{label}.environment.json"
    environment_identity = provenance.get("environment_snapshot", {})
    if (
        not environment_path.is_file()
        or environment_identity.get("path") != str(environment_path.resolve())
        or environment_identity.get("sha256") != sha256_file(environment_path)
    ):
        raise FinalProtocolError(f"{label} environment snapshot is absent or changed")
    environment = read_json(environment_path)
    hardware = context.spec["hardware"]
    if (
        environment.get("pip_freeze_all") != sorted(environment.get("pip_freeze_all", []))
        or not environment.get("pip_freeze_all")
        or hardware["device_name"] not in environment.get("cuda_device_names", [])
        or not any(
            hardware["device_name"] in line and hardware["driver_version"] in line
            for line in environment.get("nvidia_smi_name_and_driver", [])
        )
        or len(environment.get("nvidia_smi_uuid_name_and_driver", [])) != 1
        or not environment["nvidia_smi_uuid_name_and_driver"][0].startswith("GPU-")
        or hardware["device_name"]
        not in environment["nvidia_smi_uuid_name_and_driver"][0]
        or hardware["driver_version"]
        not in environment["nvidia_smi_uuid_name_and_driver"][0]
    ):
        raise FinalProtocolError(f"{label} did not use the registered final environment")
    git = provenance.get("git", {})
    model = provenance.get("model", {})
    if (
        git.get("git_commit") != context.spec["git_commit"]
        or git.get("git_dirty") is not False
        or git.get("source_tree_sha256") != context.spec["source_tree_sha256"]
        or model.get("name") != context.spec["model"]["name"]
        or model.get("configured_revision") != context.spec["model"]["revision"]
        or model.get("resolved_revision") != context.spec["model"]["revision"]
        or model.get("dtype") != context.spec["model"]["dtype"]
        or provenance.get("run_label") != label
        or provenance.get("evaluation_seed") != 0
        or provenance.get("software") != context.spec["software"]
        or provenance.get("runtime", {}).get("cuda_device_name")
        != hardware["device_name"]
        or provenance.get("runtime", {}).get("cuda_version") != hardware["cuda_version"]
        or provenance.get("runtime", {}).get("batch_size") != 64
    ):
        raise FinalProtocolError(f"{label} source/model/runtime provenance changed")
    adapter = provenance.get("adapter")
    if (is_base and adapter is not None) or (
        not is_base and adapter != _adapter_identity(context, label)
    ):
        raise FinalProtocolError(f"{label} used the wrong terminal adapter")
    if (
        provenance.get("evaluation_config", {}).get("file_sha256")
        != sha256_file(eval_config_path)
        or provenance.get("evaluation_config", {}).get("resolved_sha256")
        != canonical_sha256(eval_config)
        or provenance.get("experiment_config", {}).get("file_sha256")
        != sha256_file(experiment_config_path)
        or provenance.get("experiment_config", {}).get("resolved_sha256")
        != canonical_sha256(experiment_config)
        or provenance.get("experiment_config", {}).get("source") != "explicit"
    ):
        raise FinalProtocolError(f"{label} used a wrong final/experiment config")
    selection = provenance.get("selection", {})
    manifest_identity = selection.get("index_manifest", {})
    if (
        selection.get("method") != "index_manifest"
        or selection.get("indices_sha256") != canonical_sha256(expected_indices)
        or manifest_identity.get("sha256")
        != context.spec["final_collection"]["manifest_sha256"]
        or manifest_identity.get("dataset") != context.spec["dataset"]["name"]
        or manifest_identity.get("subset") != context.spec["dataset"]["subset"]
        or manifest_identity.get("split") != context.spec["dataset"]["split"]
        or manifest_identity.get("count") != FINAL_EXAMPLES
    ):
        raise FinalProtocolError(f"{label} used a wrong final index manifest")
    experiment = provenance.get("experiment", {})
    if (
        experiment.get("training_seed") != seed
        or experiment.get("reward_type") != "jlens"
        or experiment.get("target_words") != target_words
        or experiment.get("score_components") != experiment_config.get("score_components")
        or experiment.get("lens_sha256") != context.spec["artifacts"]["lens_sha256"]
        or experiment.get("calibration_sha256")
        != context.spec["artifacts"]["calibration_sha256"]
        or experiment.get("expected_lens_sha256")
        != context.spec["artifacts"]["lens_sha256"]
        or experiment.get("expected_calibration_sha256")
        != context.spec["artifacts"]["calibration_sha256"]
    ):
        raise FinalProtocolError(f"{label} emotional experiment identity changed")
    _verify_dispatch(context, label)


def verify_all_evaluations(
    context: FinalContext, *, references: ReferenceBundle | None = None
) -> dict[str, str]:
    verify_final_collection(context)
    expected_jsonl = {f"{label}.jsonl" for label in FINAL_LABELS}
    expected_environment = {f"{label}.environment.json" for label in FINAL_LABELS}
    entries = list(context.eval_dir.iterdir()) if context.eval_dir.is_dir() else []
    observed = {path.name for path in entries if path.is_file()}
    if (
        context.eval_dir.is_symlink()
        or observed != expected_jsonl | expected_environment
        or any(path.is_symlink() or not path.is_file() for path in entries)
    ):
        raise FinalProtocolError(
            "final directory must contain exactly 9 JSONLs and 9 environments; "
            "pending, temporary, duplicate, and unregistered files fail closed"
        )
    verify_dispatch_inventory(context)
    references = references or load_production_references(context)
    for label in FINAL_LABELS:
        verify_evaluation_jsonl(context, label, references=references)
    return {
        f"{label}.jsonl": sha256_file(context.eval_dir / f"{label}.jsonl")
        for label in FINAL_LABELS
    }


def recompute_comparison(
    context: FinalContext, *, references: ReferenceBundle | None = None
) -> dict[str, Any]:
    from jlens_rl.paired_eval import (
        compare_multiple_adapters,
        difference_in_differences,
        read_jsonl,
    )

    verify_all_evaluations(context, references=references)
    base = read_jsonl(context.eval_dir / "base.jsonl")
    treatments = [
        read_jsonl(context.eval_dir / f"jlens_seed{seed}.jsonl") for seed in SEEDS
    ]
    controls = [
        read_jsonl(context.eval_dir / f"signflip_seed{seed}.jsonl") for seed in SEEDS
    ]
    result = compare_multiple_adapters(
        base,
        treatments,
        bootstrap_samples=ANALYSIS_REGISTRATION["bootstrap_samples"],
        bootstrap_seed=ANALYSIS_REGISTRATION["bootstrap_seed"],
        confidence=ANALYSIS_REGISTRATION["confidence"],
    )
    result["primary_estimand"] = "difference_in_differences"
    result["difference_in_differences"] = difference_in_differences(
        base,
        treatments,
        controls,
        bootstrap_samples=ANALYSIS_REGISTRATION["bootstrap_samples"],
        bootstrap_seed=ANALYSIS_REGISTRATION["bootstrap_seed"],
        confidence=ANALYSIS_REGISTRATION["confidence"],
    )
    return result


def expected_analysis_arguments(context: FinalContext) -> list[str]:
    arguments = [
        "-m",
        "jlens_rl.paired_eval",
        "--base-jsonl",
        str(context.eval_dir / "base.jsonl"),
    ]
    for seed in SEEDS:
        arguments.extend(
            ["--adapter-jsonl", str(context.eval_dir / f"jlens_seed{seed}.jsonl")]
        )
    for seed in SEEDS:
        arguments.extend(
            ["--control-jsonl", str(context.eval_dir / f"signflip_seed{seed}.jsonl")]
        )
    arguments.extend(
        [
            "--bootstrap-samples",
            str(ANALYSIS_REGISTRATION["bootstrap_samples"]),
            "--seed",
            str(ANALYSIS_REGISTRATION["bootstrap_seed"]),
            "--confidence",
            str(ANALYSIS_REGISTRATION["confidence"]),
            "--output",
            str(context.comparison_path),
        ]
    )
    return arguments


def expected_analysis_command(context: FinalContext) -> list[str]:
    return [context.spec["python_executable"], *expected_analysis_arguments(context)]


def expected_analysis_probe_command(context: FinalContext) -> list[str]:
    return [context.spec["python_executable"], "-c", ANALYSIS_SOURCE_PROBE_PROGRAM]


def expected_analysis_loaded_source_identity(
    context: FinalContext,
) -> dict[str, dict[str, str]]:
    audit = verify_automation_audit(context)
    return {
        "jlens_rl.common": {
            "path": str((context.repository / AUDITED_SOURCE_PATHS["common"]).resolve()),
            "sha256": audit["source_sha256"]["common"],
        },
        "jlens_rl.paired_eval": {
            "path": str(
                (context.repository / AUDITED_SOURCE_PATHS["paired_eval"]).resolve()
            ),
            "sha256": audit["source_sha256"]["paired_eval"],
        },
    }


def verify_analysis_probe_payload(
    context: FinalContext, value: Any
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "loaded_source_identity",
        "environment_snapshot",
    }:
        raise FinalProtocolError("final analysis source probe schema changed")
    environment = value.get("environment_snapshot")
    if (
        value.get("loaded_source_identity")
        != expected_analysis_loaded_source_identity(context)
        or not isinstance(environment, dict)
        or environment.get("python", {}).get("executable")
        != context.spec["python_executable"]
        or environment.get("pip_freeze_all")
        != sorted(environment.get("pip_freeze_all", []))
        or not environment.get("pip_freeze_all")
        or environment.get("torch", {}).get("version")
        != context.spec["hardware"]["torch_version"]
        or environment.get("torch", {}).get("cuda_build")
        != context.spec["hardware"]["cuda_version"]
    ):
        raise FinalProtocolError("final analysis loaded source/environment changed")
    return value


def verify_analysis_process(context: FinalContext) -> dict[str, Any]:
    if not context.analysis_process_path.is_file():
        raise FinalProtocolError("final paired analysis process record is absent")
    value = read_json(context.analysis_process_path)
    environment = value.get("environment_snapshot", {})
    probe_payload = {
        "loaded_source_identity": value.get("loaded_source_identity"),
        "environment_snapshot": environment,
    }
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "schema_version", "python_executable", "command", "cwd",
            "environment_overrides", "input_sha256", "source_probe_command",
            "source_probe_returncode", "loaded_source_identity",
            "environment_snapshot",
        }
        or value.get("schema_version") != 1
        or value.get("python_executable") != context.spec["python_executable"]
        or value.get("cwd") != str(context.repository.resolve())
        or value.get("command") != expected_analysis_command(context)
        or value.get("environment_overrides") != expected_runtime_overrides(context)
        or value.get("source_probe_command") != expected_analysis_probe_command(context)
        or value.get("source_probe_returncode") != 0
        or value.get("loaded_source_identity")
        != expected_analysis_loaded_source_identity(context)
        or value.get("input_sha256")
        != {
            f"{label}.jsonl": sha256_file(context.eval_dir / f"{label}.jsonl")
            for label in FINAL_LABELS
        }
        or environment.get("python", {}).get("executable")
        != context.spec["python_executable"]
        or environment.get("pip_freeze_all")
        != sorted(environment.get("pip_freeze_all", []))
        or not environment.get("pip_freeze_all")
        or environment.get("torch", {}).get("version")
        != context.spec["hardware"]["torch_version"]
        or environment.get("torch", {}).get("cuda_build")
        != context.spec["hardware"]["cuda_version"]
    ):
        raise FinalProtocolError("final paired analysis command/input binding changed")
    verify_analysis_probe_payload(context, probe_payload)
    return value


def final_environment_hashes(context: FinalContext) -> dict[str, str]:
    values = {
        f"{label}.environment.json": sha256_file(
            context.eval_dir / f"{label}.environment.json"
        )
        for label in FINAL_LABELS
    }
    if len(set(values.values())) != 1:
        raise FinalProtocolError("9 final evaluations used different environments")
    return values


def final_report(
    context: FinalContext, *, references: ReferenceBundle | None = None
) -> dict[str, Any]:
    collection = verify_final_collection(context)
    if not context.comparison_path.is_file():
        raise FinalProtocolError("collect and analyze all 9 labels before reporting")
    stored = read_json(context.comparison_path)
    recomputed = recompute_comparison(context, references=references)
    if stored != recomputed:
        raise FinalProtocolError("stored final comparison changed from raw evaluations")
    process = verify_analysis_process(context)
    bound = _verify_unlock(context)
    curve = bound["curve"]
    bootstrap = stored.get("crossed_seed_item_bootstrap", {})
    sign = stored.get("seed_sign_test", {})
    specificity = stored.get("difference_in_differences", {})
    specificity_bootstrap = specificity.get("crossed_seed_item_bootstrap", {})
    specificity_sign = specificity.get("seed_sign_test", {})
    eval_hashes = {
        f"{label}.jsonl": sha256_file(context.eval_dir / f"{label}.jsonl")
        for label in FINAL_LABELS
    }
    dispatch_hashes = verify_dispatch_inventory(context)
    environment_hashes = final_environment_hashes(context)
    checks = {
        "registered_four_seed_curve_passed": curve.get("passed") is True,
        "registered_bootstrap_parameters": (
            bootstrap.get("samples") == 10_000
            and bootstrap.get("seed") == 0
            and bootstrap.get("confidence") == 0.95
            and specificity_bootstrap.get("samples") == 10_000
            and specificity_bootstrap.get("seed") == 0
            and specificity_bootstrap.get("confidence") == 0.95
        ),
        "treatment_mean_positive": stored.get("mean_accuracy_difference", 0) > 0,
        "all_four_treatment_seed_effects_positive_no_ties": (
            sign.get("positive") == 4
            and sign.get("negative") == 0
            and sign.get("tied_excluded") == 0
        ),
        "exact_two_sided_seed_sign_p_is_0_125": math.isclose(
            float(sign.get("exact_two_sided_p", 1.0)),
            0.125,
            rel_tol=0.0,
            abs_tol=0.0,
        ),
        "matched_signflip_difference_in_differences_mean_positive": specificity.get(
            "mean_difference_in_differences", 0
        )
        > 0,
        "all_four_matched_treatment_vs_signflip_seed_effects_positive_no_ties": (
            specificity_sign.get("positive") == 4
            and specificity_sign.get("negative") == 0
            and specificity_sign.get("tied_excluded") == 0
        ),
        "matched_treatment_vs_signflip_exact_two_sided_seed_sign_p_is_0_125": (
            math.isclose(
                float(specificity_sign.get("exact_two_sided_p", 1.0)),
                0.125,
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ),
        "exact_immutable_9_label_collection": len(eval_hashes) == len(FINAL_LABELS),
        "exact_dispatch_receipt_and_log_inventory": len(dispatch_hashes)
        == 4 * len(FINAL_LABELS),
        "identical_final_environments": len(set(environment_hashes.values())) == 1,
        "raw_literal_outcome_and_provenance_verification_passed": True,
        "separate_preunlock_automation_audit_passed": bool(
            verify_automation_audit(context)
        ),
        "analysis_command_and_inputs_bound": bool(process),
    }
    result = {
        "schema_version": 1,
        "protocol": context.spec["protocol"],
        "registration_sha256": context.spec["registration_sha256"],
        "recipe_lock_sha256": context.spec["recipe_lock_sha256"],
        "recipe_sha256": context.spec["recipe_sha256"],
        "registered_code_sha256": context.spec["registered_code_sha256"],
        "registered_spec_projection_sha256": registered_spec_projection_sha256(
            context.spec
        ),
        "target_words": context.spec["target_words"],
        "curve_steps": list(CURVE_STEPS),
        "analysis_registration": ANALYSIS_REGISTRATION,
        "acceptance_registration": ACCEPTANCE_REGISTRATION,
        "final_collection_id": collection["collection_id"],
        "criterion": (
            "the registered 0/2/3/4 curve plus positive means and four strictly "
            "positive seed effects (exact two-sided p=.125 < alpha=.15) for both "
            "treatment-minus-base and matched treatment-minus-signflip; crossed "
            "95% intervals are descriptive and are not acceptance gates"
        ),
        "descriptive_crossed_95pct_intervals": {
            "treatment_minus_base": {
                "low": bootstrap.get("mean_accuracy_difference_ci_low"),
                "high": bootstrap.get("mean_accuracy_difference_ci_high"),
            },
            "matched_treatment_minus_signflip": {
                "low": specificity_bootstrap.get(
                    "mean_difference_in_differences_ci_low"
                ),
                "high": specificity_bootstrap.get(
                    "mean_difference_in_differences_ci_high"
                ),
            },
            "used_as_acceptance_gate": False,
        },
        "checks": checks,
        "passed": all(checks.values()),
        "sealed_comparison_sha256": sha256_file(context.comparison_path),
        "analysis_process_sha256": sha256_file(context.analysis_process_path),
        "evaluation_jsonl_sha256": eval_hashes,
        "evaluation_environment_sha256": environment_hashes,
        "dispatch_and_log_sha256": dispatch_hashes,
        "config_sha256": verify_bound_configs(context),
        "completed_runs_sha256": sha256_file(context.completed_runs_path),
        "curve_gate_sha256": sha256_file(context.curve_path),
        "unlock_sha256": sha256_file(context.unlock_path),
        "collection_sha256": sha256_file(context.collection_path),
        "protected_final_manifest_sha256": context.spec["final_collection"][
            "manifest_sha256"
        ],
        "automation_audit_sha256": context.spec["automation_audit"]["sha256"],
    }
    if context.acceptance_path.exists():
        raise FinalProtocolError("refusing to overwrite future V10 final acceptance")
    write_json_exclusive(context.acceptance_path, result)
    return result


def design_summary() -> dict[str, Any]:
    return {
        "status": "frozen_inert_until_separate_code_audit_registration_and_unlock",
        "protocol_family": PROTOCOL_FAMILY,
        "protocol": PROTOCOL_ID,
        "design": _audit_design(),
        "matched_control_rule": MATCHED_CONTROL_RULE,
        "target_words": list(TARGET_WORDS),
        "protected_manifest_payload_accessed_here": False,
    }
