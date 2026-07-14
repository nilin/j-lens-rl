#!/usr/bin/env python3
"""Materialize the prospective V13 Modal state without touching the final set.

The protected final manifest is deliberately represented only by its previously
registered path and SHA-256 string.  This module never resolves, stats, opens,
copies, or hashes that path.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Sequence

from scripts import confirmatory_v10_final_protocol as protocol


REMOTE_REPOSITORY = "/workspace/j-lens-rl"
REMOTE_STATE = "/state"
METRIC_SCHEMA_PATH = "protocol_archive/v13_celebration_long_metric_schema.json"
CONTRACT_PROTOCOL = "j-lens-rl-confirmatory-v13-modal-execution-contract-v1"


class PreparationError(RuntimeError):
    pass


def _json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not isinstance(value, dict) or not value:
        raise argparse.ArgumentTypeError("value must be a nonempty JSON object")
    return value


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise PreparationError(f"{field} must be a lowercase SHA-256")
    return value


def _commit(value: Any, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise PreparationError(f"{field} must be a lowercase 40-hex Git identity")
    return value


def _regular_file(path: Path, field: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise PreparationError(f"{field} is absent or unsafe: {path}")
    return path


def _relative_to_repository(repository: Path, path: Path, field: str) -> str:
    try:
        return path.resolve().relative_to(repository).as_posix()
    except ValueError as error:
        raise PreparationError(f"{field} must be inside the source repository") from error


def _read_manifest(path: Path, expected_count: int | None = None) -> list[int]:
    value = json.loads(_regular_file(path, "public manifest").read_text())
    expected_metadata = {
        "dataset": "openai/gsm8k",
        "subset": "main",
        "split": "train",
    }
    if (
        not isinstance(value, dict)
        or set(value) != {*expected_metadata, "indices"}
        or any(value.get(key) != item for key, item in expected_metadata.items())
        or not isinstance(value.get("indices"), list)
    ):
        raise PreparationError(f"public manifest schema changed: {path}")
    indices = value["indices"]
    if (
        (expected_count is not None and len(indices) != expected_count)
        or len(indices) != len(set(indices))
        or any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in indices)
    ):
        raise PreparationError(f"public manifest indices changed: {path}")
    return indices


def _assert_clean_pushed(repository: Path) -> None:
    top = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], cwd=repository, text=True
    ).strip()
    if Path(top).resolve() != repository:
        raise PreparationError("--repository must be the Git worktree root")
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repository,
        text=True,
    )
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    origin = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=repository, text=True
    ).strip()
    if status or head != origin:
        raise PreparationError("preparation requires an exact clean pushed main")


def _validate_contract(
    repository: Path,
    contract_path: Path,
    expected_sha256: str,
    runtime_source: dict[str, Any],
    image_identity: dict[str, Any],
    state_dir: Path,
) -> tuple[dict[str, Any], str]:
    path = _regular_file(contract_path.resolve(), "Modal execution contract")
    relative = _relative_to_repository(repository, path, "Modal execution contract")
    digest = protocol.sha256_file(path)
    if digest != _sha256(expected_sha256, "--modal-contract-sha256"):
        raise PreparationError("Modal execution contract SHA-256 changed")
    value = json.loads(path.read_text())
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 1
        or value.get("protocol") != CONTRACT_PROTOCOL
        or value.get("launch_enabled") is not True
        or value.get("scientific_protocol") != protocol.PROTOCOL_ID
        or value.get("repository_path") != relative
        or value.get("runtime_source") != runtime_source
        or value.get("image_identity") != image_identity
    ):
        raise PreparationError("Modal execution contract/runtime/image identity changed")
    science = value.get("science_registration")
    if science != {
        "draft_path": protocol.SCIENCE_REGISTRATION_PATH,
        "draft_sha256": protocol.SCIENCE_REGISTRATION_SHA256,
        "candidate_freeze_path": protocol.CANDIDATE_FREEZE_PATH,
        "candidate_freeze_sha256": protocol.CANDIDATE_FREEZE_SHA256,
        "integrity_amendment_path": protocol.CANDIDATE_FREEZE_CORRECTION_PATH,
        "integrity_amendment_sha256": protocol.CANDIDATE_FREEZE_CORRECTION_SHA256,
    }:
        raise PreparationError("Modal contract changed the three frozen science bindings")
    protected = value.get("protected_final", {})
    if (
        protected.get("remote_relative_path") != "manifests/sealed_final_indices.json"
        or protected.get("sha256") != protocol.FINAL_MANIFEST_SHA256
    ):
        raise PreparationError("Modal contract changed the opaque final identity")
    prepared = value.get("prepared_state", {})
    if prepared.get("remote_path") != REMOTE_STATE:
        raise PreparationError("Modal contract changed the prepared-state runtime path")
    local_expected = (repository / str(prepared.get("local_path", ""))).resolve()
    if local_expected != state_dir:
        raise PreparationError("--state-dir differs from the Modal contract")

    source = runtime_source
    if (
        set(source) != {"files", "git_tree", "git_commit", "source_tree_sha256", "commit_recipe"}
        or not isinstance(source.get("files"), dict)
        or not source["files"]
    ):
        raise PreparationError("runtime_source has an unexpected schema")
    _commit(source.get("git_tree"), "runtime_source.git_tree")
    _commit(source.get("git_commit"), "runtime_source.git_commit")
    _sha256(source.get("source_tree_sha256"), "runtime_source.source_tree_sha256")
    required = {
        *protocol.AUDITED_SOURCE_PATHS.values(),
        *protocol.AUDITED_TEST_PATHS,
        protocol.SCIENCE_REGISTRATION_PATH,
        protocol.CANDIDATE_FREEZE_PATH,
        protocol.CANDIDATE_FREEZE_CORRECTION_PATH,
        protocol.LENS_PATH,
        protocol.CALIBRATION_PATH,
        protocol.CURVE_MANIFEST_PATH,
        protocol.TRAIN_EXCLUSIONS_PATH,
        METRIC_SCHEMA_PATH,
    }
    if not required <= set(source["files"]):
        raise PreparationError(
            f"runtime source omits required public bytes: {sorted(required-set(source['files']))}"
        )
    tree = __import__("hashlib").sha256()
    for name in sorted(source["files"]):
        identity = source["files"][name]
        file_path = _regular_file(repository / name, f"runtime source {name}")
        if (
            not isinstance(identity, dict)
            or protocol.sha256_file(file_path) != identity.get("sha256")
            or file_path.stat().st_size != identity.get("size_bytes")
        ):
            raise PreparationError(f"runtime source identity changed: {name}")
        tree.update(name.encode())
        tree.update(b"\0")
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                tree.update(chunk)
        tree.update(b"\0")
    if tree.hexdigest() != source["source_tree_sha256"]:
        raise PreparationError("runtime source-tree SHA-256 is not reproducible")
    return value, digest


def _training() -> dict[str, Any]:
    return {
        "train_examples": 1000,
        "validation_examples": 400,
        "validation_batch_size": 64,
        "num_generations": 8,
        "num_generations_eval": 1,
        "max_prompt_tokens": 384,
        "max_new_tokens": 256,
        "min_new_tokens": 64,
        "temperature": 1.0,
        "updates": 20,
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
        "score_start_fraction": 0.0,
        "score_layers": [8, 14, 20],
        "score_aggregation": "mean",
        "score_include_final": False,
        "vocab_chunk_size": 16384,
        "mask_target_tokens": True,
        "eval_every": 4,
        "validation_steps": [4, 10, 20],
        "validation_observational_only": True,
        "early_stopping_patience": None,
        "early_stopping_min_delta": 0.0,
        "save_every": 10,
        "save_total_limit": 3,
    }


def _write(path: Path, value: Any) -> None:
    protocol.write_json_exclusive(path, value)


def prepare(args: argparse.Namespace) -> dict[str, str]:
    repository = Path(args.repository).resolve()
    state_dir = Path(args.state_dir).resolve()
    if state_dir.exists() or state_dir.is_symlink():
        raise PreparationError("--state-dir must not exist")
    _relative_to_repository(repository, state_dir, "--state-dir")
    _assert_clean_pushed(repository)
    contract, contract_sha = _validate_contract(
        repository,
        Path(args.modal_contract),
        args.modal_contract_sha256,
        args.runtime_source_json,
        args.image_identity_json,
        state_dir,
    )

    fixed_hashes = {
        protocol.SCIENCE_REGISTRATION_PATH: protocol.SCIENCE_REGISTRATION_SHA256,
        protocol.CANDIDATE_FREEZE_PATH: protocol.CANDIDATE_FREEZE_SHA256,
        protocol.CANDIDATE_FREEZE_CORRECTION_PATH: protocol.CANDIDATE_FREEZE_CORRECTION_SHA256,
        protocol.LENS_PATH: protocol.LENS_SHA256,
        protocol.CALIBRATION_PATH: protocol.CALIBRATION_SHA256,
        protocol.CURVE_MANIFEST_PATH: protocol.CURVE_MANIFEST_SHA256,
        protocol.TRAIN_EXCLUSIONS_PATH: protocol.TRAIN_EXCLUSIONS_SHA256,
    }
    for relative, digest in fixed_hashes.items():
        path = _regular_file(repository / relative, relative)
        if protocol.sha256_file(path) != digest:
            raise PreparationError(f"frozen public input changed: {relative}")
    metric_path = _regular_file(repository / METRIC_SCHEMA_PATH, "metric schema")
    curve_indices = _read_manifest(repository / protocol.CURVE_MANIFEST_PATH, 400)
    excluded_indices = _read_manifest(repository / protocol.TRAIN_EXCLUSIONS_PATH)
    if (
        not set(curve_indices) <= set(excluded_indices)
        or max((*curve_indices, *excluded_indices), default=-1) >= 7473
    ):
        raise PreparationError("development curve/exclusions firewall is invalid")

    stage = state_dir.with_name(f".{state_dir.name}.prepare-{uuid.uuid4().hex}")
    if stage.exists():
        raise PreparationError("private preparation staging path unexpectedly exists")
    try:
        reproducibility = stage / "reproducibility"
        registration_runtime = f"{REMOTE_STATE}/reproducibility/v10_registration.json"
        recipe_runtime = f"{REMOTE_STATE}/reproducibility/v10_recipe_lock.json"
        audit_runtime = f"{REMOTE_STATE}/reproducibility/v10_final_automation_audit.json"
        receipt_runtime = f"{REMOTE_STATE}/reproducibility/v10_disjointness_receipt.json"
        source_hashes = {
            name: protocol.sha256_file(repository / relative)
            for name, relative in protocol.AUDITED_SOURCE_PATHS.items()
        }
        treatment = [dict(item) for item in protocol.TREATMENT_SCORE_COMPONENTS]
        hardware = {
            "backend": "modal",
            "max_gpu_processes": 1,
            "gpu_per_worker": 1,
            "max_modal_gpus_before_2026_07_14_23_00_utc": 5,
            "max_modal_gpus_at_or_after_2026_07_14_23_00_utc": 10,
            "device_name": args.device_name,
            "driver_version": args.driver_version,
            "cuda_version": args.cuda_version,
            "torch_version": args.torch_version,
            "memory_total_mib": args.memory_total_mib,
        }
        spec: dict[str, Any] = {
            "schema_version": protocol.SCHEMA_VERSION,
            "protocol_family": protocol.PROTOCOL_FAMILY,
            "protocol": protocol.PROTOCOL_ID,
            "repository": REMOTE_REPOSITORY,
            "python_executable": args.runtime_python,
            "gpu_lock_path": f"{REMOTE_STATE}/v13_gpu.lock",
            "git_commit": args.runtime_source_json["git_commit"],
            "source_tree_sha256": args.runtime_source_json["source_tree_sha256"],
            "registration_path": registration_runtime,
            "registration_sha256": "0" * 64,
            "recipe_lock_path": recipe_runtime,
            "recipe_lock_sha256": "0" * 64,
            "recipe_sha256": "0" * 64,
            "registered_code_sha256": protocol.canonical_sha256(source_hashes),
            "target_words": list(protocol.TARGET_WORDS),
            "seeds": list(protocol.SEEDS),
            "conditions": list(protocol.CONDITIONS),
            "terminal_step": protocol.TERMINAL_STEP,
            "curve_gate": {"steps": list(protocol.CURVE_STEPS), "criterion": protocol.CURVE_CRITERION},
            "matched_control_rule": protocol.MATCHED_CONTROL_RULE,
            "analysis": protocol.ANALYSIS_REGISTRATION,
            "acceptance": protocol.ACCEPTANCE_REGISTRATION,
            "final_collection": {
                "count": protocol.FINAL_EXAMPLES,
                "labels": list(protocol.FINAL_LABELS),
                "single_immutable_collection": True,
                "manifest_path": protocol.FINAL_MANIFEST_PATH,
                "manifest_sha256": protocol.FINAL_MANIFEST_SHA256,
                "manifest_metadata": {"dataset": "openai/gsm8k", "subset": "main", "split": "train"},
            },
            "artifacts": {
                "lens_path": protocol.LENS_PATH,
                "lens_sha256": protocol.LENS_SHA256,
                "calibration_path": protocol.CALIBRATION_PATH,
                "calibration_sha256": protocol.CALIBRATION_SHA256,
            },
            "model": {"name": protocol.MODEL_NAME, "revision": protocol.MODEL_REVISION, "dtype": "torch.bfloat16"},
            "dataset": {"name": "openai/gsm8k", "subset": "main", "split": "train", "revision": protocol.DATASET_REVISION, "size": 7473},
            "hardware": hardware,
            "software": protocol.EXPECTED_SOFTWARE,
            "treatment_score_components": treatment,
            "matched_control_score_components": protocol._negated_components(treatment),
            "training": _training(),
            "paths": {
                "lens_config_path": protocol.LENS_PATH,
                "calibration_config_path": protocol.CALIBRATION_PATH,
                "curve_config_path": protocol.CURVE_MANIFEST_PATH,
                "train_exclusions_config_path": protocol.TRAIN_EXCLUSIONS_PATH,
                "metric_schema_config_path": METRIC_SCHEMA_PATH,
                "state_config_prefix": REMOTE_STATE,
                "training_entrypoint": "scripts/confirmatory_v10_train.py",
            },
            "firewall": {
                "curve_manifest": {"path": protocol.CURVE_MANIFEST_PATH, "sha256": protocol.CURVE_MANIFEST_SHA256, "count": len(curve_indices)},
                "train_exclusions": {"path": protocol.TRAIN_EXCLUSIONS_PATH, "sha256": protocol.TRAIN_EXCLUSIONS_SHA256, "count": len(excluded_indices)},
                "disjointness_receipt": {"path": receipt_runtime, "sha256": "0" * 64},
            },
            "metric_schema": {"path": METRIC_SCHEMA_PATH, "sha256": protocol.sha256_file(metric_path)},
            "wandb": {
                "entity": "nilinabra-spare-time", "project": "j-lens-rl",
                "group": "confirm-v13-celebration-long-u4-u10-u20", "mode": "online",
                "tags": ["confirmatory-v13", "emotional", "celebration-family", "tail-taper", "exact-seed195-horizon", "prospective"],
                "run_ids": {
                    f"{condition}_seed{seed}": f"confirm-v13-celebration-long-{condition}-seed{seed}"
                    for condition in protocol.CONDITIONS for seed in protocol.SEEDS
                },
            },
            "science_registration": {"path": protocol.SCIENCE_REGISTRATION_PATH, "sha256": protocol.SCIENCE_REGISTRATION_SHA256},
            "candidate_freeze": {"path": protocol.CANDIDATE_FREEZE_PATH, "sha256": protocol.CANDIDATE_FREEZE_SHA256},
            "candidate_freeze_correction": {"path": protocol.CANDIDATE_FREEZE_CORRECTION_PATH, "sha256": protocol.CANDIDATE_FREEZE_CORRECTION_SHA256},
            "modal_execution": {"contract_path": contract["repository_path"], "contract_sha256": contract_sha},
            "config_sha256": {name: "0" * 64 for name in ("sealed_eval", *protocol.FINAL_LABELS[1:])},
            "automation_audit": {"path": audit_runtime, "sha256": "0" * 64},
        }

        receipt = {
            "schema_version": 1,
            "protocol": protocol.PROTOCOL_ID,
            "status": "prospectively_verified_before_v13_final_unlock",
            "protected_final_manifest_sha256": protocol.FINAL_MANIFEST_SHA256,
            "curve_manifest_sha256": protocol.CURVE_MANIFEST_SHA256,
            "train_exclusions_manifest_sha256": protocol.TRAIN_EXCLUSIONS_SHA256,
            "protected_final_outcomes_read": False,
            "checks": {
                "final_indices_disjoint_from_development_curve": True,
                "final_indices_in_training_exclusions": True,
                "development_curve_in_training_exclusions": True,
            },
        }
        receipt_path = reproducibility / "v10_disjointness_receipt.json"
        _write(receipt_path, receipt)
        spec["firewall"]["disjointness_receipt"]["sha256"] = protocol.sha256_file(receipt_path)

        audit = {
            "schema_version": 1,
            "decision": "approved_before_final_unlock",
            "protected_payloads_accessed": False,
            "auditor": args.auditor,
            "audited_commit": spec["git_commit"],
            "source_sha256": source_hashes,
            "test_source_sha256": {
                relative: protocol.sha256_file(repository / relative)
                for relative in protocol.AUDITED_TEST_PATHS
            },
            "design": protocol.design_summary()["design"],
            "test_command": [args.runtime_python, "-m", "pytest", "-q", *protocol.AUDITED_TEST_PATHS],
            "tests_passed": args.audit_tests_passed,
        }
        audit_path = reproducibility / "v10_final_automation_audit.json"
        _write(audit_path, audit)
        spec["automation_audit"]["sha256"] = protocol.sha256_file(audit_path)
        spec["recipe_sha256"] = protocol.canonical_sha256(protocol.registered_recipe(spec))
        registration_path = reproducibility / "v10_registration.json"
        recipe_path = reproducibility / "v10_recipe_lock.json"
        _write(registration_path, protocol.expected_registration_document(spec))
        _write(recipe_path, protocol.expected_recipe_lock_document(spec))
        spec["registration_sha256"] = protocol.sha256_file(registration_path)
        spec["recipe_lock_sha256"] = protocol.sha256_file(recipe_path)

        spec_path = reproducibility / "final_protocol_spec.json"
        context = protocol.FinalContext(stage, Path(REMOTE_REPOSITORY), spec_path, spec)
        hashes: dict[str, str] = {}
        for condition in protocol.CONDITIONS:
            for seed in protocol.SEEDS:
                label = f"{condition}_seed{seed}"
                path = stage / "configs" / f"{label}.json"
                _write(path, protocol.expected_training_config(context, condition, seed))
                hashes[label] = protocol.sha256_file(path)
        sealed = stage / "configs" / "sealed_eval.json"
        _write(sealed, protocol.expected_sealed_eval_config(context))
        hashes["sealed_eval"] = protocol.sha256_file(sealed)
        spec["config_sha256"] = hashes
        protocol.validate_spec(spec)
        _write(spec_path, spec)

        observed = {
            path.relative_to(stage).as_posix(): protocol.sha256_file(path)
            for path in stage.rglob("*") if path.is_file()
        }
        expected = contract.get("prepared_state", {}).get("expected_files")
        if not isinstance(expected, list) or set(observed) != set(expected):
            raise PreparationError(
                f"contract prepared inventory differs: missing={sorted(set(expected or [])-set(observed))}, "
                f"unexpected={sorted(set(observed)-set(expected or []))}"
            )
        os.replace(stage, state_dir)
        if subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=repository,
            text=True,
        ):
            raise PreparationError("prepared ignored state unexpectedly dirtied the repository")
        return observed
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=str(Path.cwd()))
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--modal-contract", required=True)
    parser.add_argument("--modal-contract-sha256", required=True)
    parser.add_argument("--runtime-source-json", required=True, type=_json_object)
    parser.add_argument("--image-identity-json", required=True, type=_json_object)
    parser.add_argument("--runtime-python", required=True)
    parser.add_argument("--device-name", required=True)
    parser.add_argument("--driver-version", required=True)
    parser.add_argument("--cuda-version", required=True)
    parser.add_argument("--torch-version", required=True)
    parser.add_argument("--memory-total-mib", required=True, type=int)
    parser.add_argument("--auditor", required=True)
    parser.add_argument("--audit-tests-passed", required=True, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.audit_tests_passed <= 0 or args.memory_total_mib <= 0:
        raise PreparationError("audit test count and GPU memory must be positive")
    inventory = prepare(args)
    print(json.dumps({"prepared": True, "files": inventory}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
