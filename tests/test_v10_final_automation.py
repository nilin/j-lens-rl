from __future__ import annotations

import ast
import copy
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pytest

from jlens_rl.common import extract_answer, repository_provenance
from jlens_rl.paired_eval import literal_target_matches
from scripts import confirmatory_v10_final_protocol as final
from scripts import confirmatory_v10_final_runner as runner


@pytest.fixture(autouse=True)
def _restore_frozen_artifact_hashes() -> object:
    names = (
        "LENS_SHA256", "CALIBRATION_SHA256", "CURVE_MANIFEST_SHA256",
        "TRAIN_EXCLUSIONS_SHA256", "FINAL_MANIFEST_SHA256", "FINAL_MANIFEST_PATH",
    )
    original = {name: getattr(final, name) for name in names}
    yield
    for name, value in original.items():
        setattr(final, name, value)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _copy_audited_sources(repository: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    for relative in (*final.AUDITED_SOURCE_PATHS.values(), *final.AUDITED_TEST_PATHS):
        source = root / relative
        if (
            not source.is_file()
            and relative == final.AUDITED_SOURCE_PATHS["training_entrypoint"]
        ):
            # The production wrapper is deliberately future/inert in this module;
            # the synthetic repository materializes it immediately below.
            continue
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())


def _bound_artifact(repository: Path, relative: str, content: str) -> tuple[str, str]:
    path = repository / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return relative, final.sha256_file(path)


def _hardware() -> dict[str, object]:
    return {
        "backend": "modal",
        "max_gpu_processes": 1,
        "gpu_per_worker": 1,
        "max_modal_gpus_before_2026_07_14_23_00_utc": 5,
        "max_modal_gpus_at_or_after_2026_07_14_23_00_utc": 10,
        "device_name": "NVIDIA L40S",
        "driver_version": "570.195.03",
        "cuda_version": "12.8",
        "torch_version": "2.9.1+cu128",
        "memory_total_mib": 24564,
    }


def _training() -> dict[str, object]:
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
    }


def _runtime(context: final.FinalContext, environment_path: Path) -> dict[str, object]:
    environment = json.loads(environment_path.read_text())
    return {
        "cuda_device_name": context.spec["hardware"]["device_name"],
        "cuda_version": context.spec["hardware"]["cuda_version"],
        "python_version": sys.version,
        "torch_version": context.spec["hardware"]["torch_version"],
        "environment_snapshot_path": "environment_snapshot.json",
        "environment_snapshot_sha256": final.sha256_file(environment_path),
        "environment_snapshot": environment,
    }


def _synthetic_log_history(
    config: dict[str, object], curve: dict[int, float]
) -> list[dict[str, float | int]]:
    label = "_".join(config["target_words"])  # type: ignore[arg-type]
    literal_key = f"jlens/{label}_literal_rate"
    reward_mean_key = f"rewards/jlens_{label}_reward/mean"
    reward_std_key = f"rewards/jlens_{label}_reward/std"
    rows: list[dict[str, float | int]] = []
    updates = int(config["updates"])
    for step in range(1, updates + 1):
        rows.append(
            {
                "clip_ratio/high_max": 0.0,
                "clip_ratio/high_mean": 0.0,
                "clip_ratio/low_mean": 0.0,
                "clip_ratio/low_min": 0.0,
                "clip_ratio/region_mean": 0.0,
                "completions/clipped_ratio": 0.0,
                "completions/max_length": 128.0,
                "completions/max_terminated_length": 128.0,
                "completions/mean_length": 100.0,
                "completions/mean_terminated_length": 100.0,
                "completions/min_length": 64.0,
                "completions/min_terminated_length": 64.0,
                "entropy": 1.0,
                "epoch": step / updates,
                "frac_reward_zero_std": 0.0,
                "grad_norm": 1.0,
                literal_key: 0.0,
                f"jlens/{label}_mean": 0.1,
                "kl": 0.01,
                "learning_rate": float(config["learning_rate"]),
                "loss": 0.1,
                "num_tokens": float(step * 100),
                "reward": 0.1,
                "reward_std": 0.2,
                reward_mean_key: 0.1,
                reward_std_key: 0.2,
                "step": step,
                "step_time": 1.0,
            }
        )
    for step, exact_match in curve.items():
        if step == final.CURVE_STEPS[0]:
            continue
        rows.append(
            {
                "validation/exact_match": exact_match,
                "validation/exact_match_ci95_high": min(1.0, exact_match + 0.02),
                "validation/exact_match_ci95_low": max(0.0, exact_match - 0.02),
                "validation/literal_target_completion_rate": 0.0,
                "validation/mean_length": 100.0,
                "step": step,
            }
        )
    rows.append(
        {
            "total_flos": 1.0,
            "train_loss": 0.1,
            "train_runtime": 15.0,
            "train_samples_per_second": 1.0,
            "train_steps_per_second": 1.0,
            "step": updates,
        }
    )
    return rows


def _create_completed_runs(
    context: final.FinalContext,
    curve_indices: list[int],
    excluded_indices: list[int],
) -> None:
    run_records: dict[str, object] = {}
    per_seed: dict[str, dict[str, float]] = {}
    treatment_curve = {0: 0.30, 4: 0.35, 5: 0.36, 6: 0.37}
    control_curve = {0: 0.30, 4: 0.29, 5: 0.28, 6: 0.27}
    train_indices = list(range(max(excluded_indices) + 1, max(excluded_indices) + 1001))
    for condition in final.CONDITIONS:
        for seed in final.SEEDS:
            label = f"{condition}_seed{seed}"
            directory = context.run_dir / label
            directory.mkdir(parents=True)
            config_path = context.config_dir / f"{label}.json"
            config = json.loads(config_path.read_text())
            _write(directory / "resolved_config.json", config)
            data = {
                "train_source_indices": train_indices,
                "validation_source": "train",
                "validation_source_indices": curve_indices,
            }
            _write(directory / "data_indices.json", data)
            curve = treatment_curve if condition == "jlens" else control_curve
            history_rows = [
                {
                    "step": step,
                    "exact_match": value,
                    "validation_source": "train",
                    "validation_indices_sha256": context.spec["firewall"][
                        "curve_manifest"
                    ]["sha256"],
                }
                for step, value in curve.items()
            ]
            (directory / "validation_history.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in history_rows)
            )
            _write(directory / "log_history.json", _synthetic_log_history(config, curve))
            environment = {
                "python": {"executable": context.spec["python_executable"]},
                "pip_freeze_all": ["torch==2.9.1"],
                "torch": {
                    "version": context.spec["hardware"]["torch_version"],
                    "cuda_build": context.spec["hardware"]["cuda_version"],
                },
                "cuda_device_names": [context.spec["hardware"]["device_name"]],
                "nvidia_smi_name_and_driver": [
                    f"{context.spec['hardware']['device_name']}, "
                    f"{context.spec['hardware']['driver_version']}"
                ],
                "nvidia_smi_uuid_name_and_driver": [
                    "GPU-00000000-0000-0000-0000-000000000000, "
                    f"{context.spec['hardware']['device_name']}, "
                    f"{context.spec['hardware']['driver_version']}"
                ],
            }
            environment_path = directory / "environment_snapshot.json"
            _write(environment_path, environment)
            runtime = _runtime(context, environment_path)
            metric_path = context.repository / context.spec["metric_schema"]["path"]
            process_command = {
                "python_executable": context.spec["python_executable"],
                "argv": config["registered_command"][1:],
                "cwd": str(context.repository.resolve()),
            }
            metric_schema = {
                "path": str(metric_path.resolve()),
                "sha256": context.spec["metric_schema"]["sha256"],
                "content": json.loads(metric_path.read_text()),
            }
            manifest = {
                "git_commit": context.spec["git_commit"],
                "git_dirty": False,
                "source_tree_sha256": context.spec["source_tree_sha256"],
                "config_path": str(config_path.resolve()),
                "config_sha256": final.sha256_file(config_path),
                "resolved_config_sha256": final.sha256_file(
                    directory / "resolved_config.json"
                ),
                "model_name": context.spec["model"]["name"],
                "model_revision": context.spec["model"]["revision"],
                "dataset": "openai/gsm8k:main",
                "dataset_revision": context.spec["dataset"]["revision"],
                "lens_sha256": context.spec["artifacts"]["lens_sha256"],
                "calibration_sha256": context.spec["artifacts"]["calibration_sha256"],
                "reward_type": "jlens",
                "process_command": process_command,
                "registered_command": config["registered_command"],
                "evidence_eligibility": "original_registered_v10_modal_attempt",
                "reproduction_source": None,
                "confirmatory_identity": {
                    "registration_sha256": context.spec["registration_sha256"],
                    "recipe_lock_sha256": context.spec["recipe_lock_sha256"],
                    "recipe_sha256": context.spec["recipe_sha256"],
                    "curve_manifest_sha256": context.spec["firewall"]["curve_manifest"][
                        "sha256"
                    ],
                    "train_exclusions_manifest_sha256": context.spec["firewall"][
                        "train_exclusions"
                    ]["sha256"],
                    "registered_code_sha256": context.spec["registered_code_sha256"],
                },
                "metric_schema": metric_schema,
                "wandb_identity": final._expected_wandb_identity(config),
                "runtime": runtime,
                "data_indices_sha256": final.sha256_file(directory / "data_indices.json"),
            }
            _write(directory / "run_manifest.json", manifest)
            for tree_name in ("final", f"checkpoint-{final.TERMINAL_STEP}"):
                _write(directory / tree_name / "adapter_config.json", {"label": label})
                (directory / tree_name / "adapter_model.safetensors").write_bytes(
                    f"synthetic-{tree_name}-{label}".encode()
                )
            raw_hashes = {
                name: final.sha256_file(directory / name)
                for name in final.TERMINAL_EVIDENCE_NAMES
                if name != "run_result_manifest.json"
            }
            result = {
                "schema_version": 1,
                "completed_updates": final.TERMINAL_STEP,
                "registration_sha256": context.spec["registration_sha256"],
                "recipe_lock_sha256": context.spec["recipe_lock_sha256"],
                "recipe_sha256": context.spec["recipe_sha256"],
                "registered_command": config["registered_command"],
                "process_command": process_command,
                "metric_schema": metric_schema,
                "source": {
                    "git_commit": context.spec["git_commit"],
                    "git_dirty": False,
                    "source_tree_sha256": context.spec["source_tree_sha256"],
                },
                "runtime": runtime,
                "data_indices_sha256": final.sha256_file(directory / "data_indices.json"),
                "lens_sha256": context.spec["artifacts"]["lens_sha256"],
                "calibration_sha256": context.spec["artifacts"]["calibration_sha256"],
                "raw_history_sha256": raw_hashes,
                "terminal_checkpoint": final._tree_identity(
                    directory / f"checkpoint-{final.TERMINAL_STEP}"
                ),
                "final_adapter_and_tokenizer": final._tree_identity(directory / "final"),
                "wandb_identity": final._expected_wandb_identity(config),
                "evidence_eligibility": "original_registered_v10_modal_attempt",
                "reproduction_source": None,
            }
            _write(directory / "run_result_manifest.json", result)
            evidence_hashes = {
                name: final.sha256_file(directory / name)
                for name in final.TERMINAL_EVIDENCE_NAMES
            }
            identity = final._expected_wandb_identity(config)
            version = "v0"
            artifact_base = f"{identity['run_id']}-terminal-evidence"
            receipt = {
                "schema_version": 2,
                "wandb_identity": identity,
                "observed_wandb_identity": {
                    key: identity[key]
                    for key in (
                        "run_id", "entity", "project", "run_name", "url",
                        "group", "tags",
                    )
                },
                "artifact": {
                    "id": f"synthetic-{identity['run_id']}",
                    "name": f"{artifact_base}:{version}",
                    "version": version,
                    "digest": "synthetic-digest",
                    "qualified_name": (
                        f"{identity['entity']}/{identity['project']}/"
                        f"{artifact_base}:{version}"
                    ),
                },
                "terminal_run_result_sha256": evidence_hashes["run_result_manifest.json"],
                "uploaded_file_sha256": evidence_hashes,
            }
            _write(directory / "wandb_terminal_publish_receipt.json", receipt)
            record, _history, _train = final._verify_one_completed_run(
                context,
                condition,
                seed,
                curve_indices=curve_indices,
                excluded_indices=set(excluded_indices),
            )
            run_records[label] = record
            if condition == "jlens":
                per_seed[str(seed)] = {
                    str(step): float(value) for step, value in treatment_curve.items()
                }
    completed = {
        "schema_version": 1,
        "protocol": context.spec["protocol"],
        "git_commit": context.spec["git_commit"],
        "registration_sha256": context.spec["registration_sha256"],
        "recipe_lock_sha256": context.spec["recipe_lock_sha256"],
        "recipe_sha256": context.spec["recipe_sha256"],
        "registered_code_sha256": context.spec["registered_code_sha256"],
        "registered_spec_projection_sha256": final.registered_spec_projection_sha256(
            context.spec
        ),
        "seeds": list(final.SEEDS),
        "conditions": list(final.CONDITIONS),
        "terminal_step": final.TERMINAL_STEP,
        "hardware": context.spec["hardware"],
        "source_tree_sha256": context.spec["source_tree_sha256"],
        "runs": run_records,
    }
    _write(context.completed_runs_path, completed)
    means = {
        str(step): sum(per_seed[str(seed)][str(step)] for seed in final.SEEDS)
        / len(final.SEEDS)
        for step in final.CURVE_STEPS
    }
    _write(
        context.curve_path,
        {
            "steps": list(final.CURVE_STEPS),
            "criterion": final.CURVE_CRITERION,
            "n_seeds": len(final.SEEDS),
            "per_seed_exact_match": per_seed,
            "mean_exact_match": means,
            "passed": True,
        },
    )


def _future_context(
    tmp_path: Path, *, final_indices_override: list[int] | None = None
) -> final.FinalContext:
    repository = tmp_path / "repo"
    state = tmp_path / "state"
    repository.mkdir(parents=True)
    _copy_audited_sources(repository)
    source_root = Path(__file__).resolve().parents[1]
    for relative in (
        final.SCIENCE_REGISTRATION_PATH,
        final.CANDIDATE_FREEZE_PATH,
        final.CANDIDATE_FREEZE_CORRECTION_PATH,
    ):
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((source_root / relative).read_bytes())
    registration_path = str(state / "reproducibility" / "v10_registration.json")
    recipe_path = str(state / "reproducibility" / "v10_recipe_lock.json")
    lens_path, lens_sha = _bound_artifact(repository, final.LENS_PATH, "lens\n")
    calibration_path, calibration_sha = _bound_artifact(
        repository, final.CALIBRATION_PATH, "calibration\n"
    )
    metric_path, metric_sha = _bound_artifact(
        repository,
        "protocol_archive/metric_schema.json",
        '{"schema_version": 1, "metric": "gsm8k_exact_match"}\n',
    )
    _bound_artifact(repository, "scripts/confirmatory_v10_train.py", "training entrypoint\n")
    curve_manifest = repository / final.CURVE_MANIFEST_PATH
    exclusions_manifest = repository / final.TRAIN_EXCLUSIONS_PATH
    final.FINAL_MANIFEST_PATH = str(state / "manifests" / "sealed_final_indices.json")
    final_manifest = Path(final.FINAL_MANIFEST_PATH)
    modal_contract_path, modal_contract_sha = _bound_artifact(
        repository,
        "protocol_archive/v10_synthetic_modal_execution.json",
        '{"backend":"Modal","gpu":"L40S"}\n',
    )
    curve_indices = list(range(400))
    final_indices = final_indices_override or list(range(400, 1300))
    excluded_indices = list(range(1300))
    manifest_metadata = {
        "dataset": "openai/gsm8k",
        "subset": "main",
        "split": "train",
    }
    _write(curve_manifest, {**manifest_metadata, "indices": curve_indices})
    _write(exclusions_manifest, {**manifest_metadata, "indices": excluded_indices})
    _write(final_manifest, {**manifest_metadata, "indices": final_indices})
    final.LENS_SHA256 = lens_sha
    final.CALIBRATION_SHA256 = calibration_sha
    final.CURVE_MANIFEST_SHA256 = final.sha256_file(curve_manifest)
    final.TRAIN_EXCLUSIONS_SHA256 = final.sha256_file(exclusions_manifest)
    final.FINAL_MANIFEST_SHA256 = final.sha256_file(final_manifest)
    source_hashes = {
        name: final.sha256_file(repository / relative)
        for name, relative in final.AUDITED_SOURCE_PATHS.items()
    }
    protocol_name = final.PROTOCOL_ID
    disjointness_path = state / "reproducibility" / "v10_disjointness.json"
    disjointness = {
        "schema_version": 1,
        "protocol": protocol_name,
        "status": "prospectively_verified_before_v10_final_unlock",
        "protected_final_manifest_sha256": final.sha256_file(final_manifest),
        "curve_manifest_sha256": final.sha256_file(curve_manifest),
        "train_exclusions_manifest_sha256": final.sha256_file(exclusions_manifest),
        "protected_final_outcomes_read": False,
        "checks": {
            "final_indices_disjoint_from_development_curve": True,
            "final_indices_in_training_exclusions": True,
            "development_curve_in_training_exclusions": True,
        },
    }
    _write(disjointness_path, disjointness)
    treatment = [dict(component) for component in final.TREATMENT_SCORE_COMPONENTS]
    commit = "0" * 40
    source_tree_sha = "0" * 64
    audit_path = state / "reproducibility" / "v10_final_automation_audit.json"
    spec: dict[str, object] = {
        "schema_version": 1,
        "protocol_family": final.PROTOCOL_FAMILY,
        "protocol": protocol_name,
        "repository": str(repository),
        "python_executable": sys.executable,
        "gpu_lock_path": str(tmp_path / "gpu.lock"),
        "git_commit": commit,
        "source_tree_sha256": source_tree_sha,
        "registration_path": registration_path,
        "registration_sha256": "0" * 64,
        "recipe_lock_path": recipe_path,
        "recipe_lock_sha256": "0" * 64,
        "recipe_sha256": "0" * 64,
        "registered_code_sha256": final.canonical_sha256(source_hashes),
        "target_words": list(final.TARGET_WORDS),
        "seeds": list(final.SEEDS),
        "conditions": list(final.CONDITIONS),
        "terminal_step": final.TERMINAL_STEP,
        "curve_gate": {
            "steps": list(final.CURVE_STEPS),
            "criterion": final.CURVE_CRITERION,
        },
        "matched_control_rule": final.MATCHED_CONTROL_RULE,
        "analysis": final.ANALYSIS_REGISTRATION,
        "acceptance": final.ACCEPTANCE_REGISTRATION,
        "final_collection": {
            "count": final.FINAL_EXAMPLES,
            "labels": list(final.FINAL_LABELS),
            "single_immutable_collection": True,
            "manifest_path": final.FINAL_MANIFEST_PATH,
            "manifest_sha256": final.sha256_file(final_manifest),
            "manifest_metadata": manifest_metadata,
        },
        "artifacts": {
            "lens_path": lens_path,
            "lens_sha256": lens_sha,
            "calibration_path": calibration_path,
            "calibration_sha256": calibration_sha,
        },
        "model": {
            "name": "Qwen/Qwen2.5-0.5B-Instruct",
            "revision": final.MODEL_REVISION,
            "dtype": "torch.bfloat16",
        },
        "dataset": {
            "name": "openai/gsm8k",
            "subset": "main",
            "split": "train",
            "revision": final.DATASET_REVISION,
            "size": 4000,
        },
        "hardware": _hardware(),
        "software": final.EXPECTED_SOFTWARE,
        "treatment_score_components": treatment,
        "matched_control_score_components": final._negated_components(treatment),
        "training": _training(),
        "paths": {
            "lens_config_path": lens_path,
            "calibration_config_path": calibration_path,
            "curve_config_path": final.CURVE_MANIFEST_PATH,
            "train_exclusions_config_path": final.TRAIN_EXCLUSIONS_PATH,
            "metric_schema_config_path": metric_path,
            "state_config_prefix": str(state),
            "training_entrypoint": "scripts/confirmatory_v10_train.py",
        },
        "firewall": {
            "curve_manifest": {
                "path": final.CURVE_MANIFEST_PATH,
                "sha256": final.sha256_file(curve_manifest),
                "count": len(curve_indices),
            },
            "train_exclusions": {
                "path": final.TRAIN_EXCLUSIONS_PATH,
                "sha256": final.sha256_file(exclusions_manifest),
                "count": len(excluded_indices),
            },
            "disjointness_receipt": {
                "path": str(disjointness_path),
                "sha256": final.sha256_file(disjointness_path),
            },
        },
        "metric_schema": {"path": metric_path, "sha256": metric_sha},
        "wandb": {
            "entity": "nilinabra-spare-time",
            "project": "j-lens-rl",
            "group": "confirm-v12-celebration-u4-u5-u6",
            "mode": "online",
            "tags": [
                "confirmatory-v12",
                "emotional",
                "celebration-family",
                "tail-taper",
                "prospective",
            ],
            "run_ids": {
                f"{condition}_seed{seed}": (
                    f"confirm-v12-celebration-{condition}-seed{seed}"
                )
                for condition in final.CONDITIONS
                for seed in final.SEEDS
            },
        },
        "science_registration": {
            "path": final.SCIENCE_REGISTRATION_PATH,
            "sha256": final.SCIENCE_REGISTRATION_SHA256,
        },
        "candidate_freeze": {
            "path": final.CANDIDATE_FREEZE_PATH,
            "sha256": final.CANDIDATE_FREEZE_SHA256,
        },
        "candidate_freeze_correction": {
            "path": final.CANDIDATE_FREEZE_CORRECTION_PATH,
            "sha256": final.CANDIDATE_FREEZE_CORRECTION_SHA256,
        },
        "modal_execution": {
            "contract_path": modal_contract_path,
            "contract_sha256": modal_contract_sha,
        },
        "config_sha256": {name: "0" * 64 for name in ("sealed_eval", *final.FINAL_LABELS[1:])},
        "automation_audit": {
            "path": str(audit_path),
            "sha256": "0" * 64,
        },
    }
    spec["recipe_sha256"] = final.canonical_sha256(final.registered_recipe(spec))
    registration_file = Path(registration_path)
    recipe_file = Path(recipe_path)
    _write(registration_file, final.expected_registration_document(spec))
    _write(recipe_file, final.expected_recipe_lock_document(spec))
    spec["registration_sha256"] = final.sha256_file(registration_file)
    spec["recipe_lock_sha256"] = final.sha256_file(recipe_file)
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "v10-test@example.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "V10 Test"], cwd=repository, check=True
    )
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "synthetic V10 source"],
        cwd=repository,
        check=True,
    )
    provenance = repository_provenance(repository)
    assert provenance["git_dirty"] is False
    spec["git_commit"] = provenance["git_commit"]
    spec["source_tree_sha256"] = provenance["source_tree_sha256"]
    audit = {
        "schema_version": 1,
        "decision": "approved_before_final_unlock",
        "protected_payloads_accessed": False,
        "auditor": "independent-test-auditor",
        "audited_commit": spec["git_commit"],
        "source_sha256": source_hashes,
        "test_source_sha256": {
            relative: final.sha256_file(repository / relative)
            for relative in final.AUDITED_TEST_PATHS
        },
        "design": final.design_summary()["design"],
        "test_command": [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_v10_final_automation.py",
            "tests/test_paired_eval.py",
        ],
        "tests_passed": 1,
    }
    _write(audit_path, audit)
    spec["automation_audit"] = {
        "path": str(audit_path),
        "sha256": final.sha256_file(audit_path),
    }
    spec_path = state / "reproducibility" / "final_protocol_spec.json"
    provisional = final.FinalContext(state, repository, spec_path, spec)  # type: ignore[arg-type]
    config_hashes: dict[str, str] = {}
    for condition in final.CONDITIONS:
        for seed in final.SEEDS:
            label = f"{condition}_seed{seed}"
            path = state / "configs" / f"{label}.json"
            _write(path, final.expected_training_config(provisional, condition, seed))
            config_hashes[label] = final.sha256_file(path)
    sealed_path = state / "configs" / "sealed_eval.json"
    _write(sealed_path, final.expected_sealed_eval_config(provisional))
    config_hashes["sealed_eval"] = final.sha256_file(sealed_path)
    spec["config_sha256"] = config_hashes
    _write(spec_path, spec)
    context = final.load_context(state)
    _create_completed_runs(context, curve_indices, excluded_indices)
    unlock = {
        "protocol": context.spec["protocol"],
        "git_commit": context.spec["git_commit"],
        "registration_sha256": context.spec["registration_sha256"],
        "curve_gate_sha256": final.sha256_file(context.curve_path),
        "completed_runs_sha256": final.sha256_file(context.completed_runs_path),
        "automation_audit_sha256": context.spec["automation_audit"]["sha256"],
        "recipe_lock_sha256": context.spec["recipe_lock_sha256"],
        "recipe_sha256": context.spec["recipe_sha256"],
        "registered_code_sha256": context.spec["registered_code_sha256"],
        "registered_spec_projection_sha256": final.registered_spec_projection_sha256(
            context.spec
        ),
        "final_manifest_sha256": context.spec["final_collection"]["manifest_sha256"],
        "disjointness_receipt_sha256": context.spec["firewall"]["disjointness_receipt"][
            "sha256"
        ],
    }
    _write(context.unlock_path, unlock)
    return context


def _begin(context: final.FinalContext, collection_id: str = "3" * 32) -> str:
    final.begin_final_collection(context, collection_id)
    return collection_id


def _references(context: final.FinalContext) -> final.ReferenceBundle:
    indices = list(range(400, 1300))

    def decode(tokens: list[int]) -> str:
        source_index, correct = tokens
        return f"#### {source_index}" if correct else "#### -1"

    return final.ReferenceBundle(
        indices=indices,
        dataset_fingerprint="synthetic-fingerprint",
        prompt_sha256={index: f"prompt-{index}" for index in indices},
        prompt_token_ids_sha256={index: f"tokens-{index}" for index in indices},
        answers={index: f"#### {index}" for index in indices},
        decode_completion=decode,
        extract_answer=extract_answer,
        is_correct=lambda completion, answer: extract_answer(completion)
        == extract_answer(answer),
        literal_matches=literal_target_matches,
    )


def _correctness_predicate(value: bool | int | Callable[[int], bool]) -> Callable[[int], bool]:
    if callable(value):
        return value
    if isinstance(value, bool):
        return lambda _index: value
    return lambda index: index < 400 + value


def _write_synthetic_label(
    context: final.FinalContext,
    label: str,
    references: final.ReferenceBundle,
    *,
    correct: bool | int | Callable[[int], bool],
) -> None:
    condition, seed, is_base = final.evaluation_role(label)
    environment_path = context.eval_dir / f"{label}.environment.json"
    environment = {
        "pip_freeze_all": ["a==1"],
        "cuda_device_names": [context.spec["hardware"]["device_name"]],
        "nvidia_smi_name_and_driver": [
            f"{context.spec['hardware']['device_name']}, "
            f"{context.spec['hardware']['driver_version']}"
        ],
        "nvidia_smi_uuid_name_and_driver": [
            "GPU-00000000-0000-0000-0000-000000000000, "
            f"{context.spec['hardware']['device_name']}, "
            f"{context.spec['hardware']['driver_version']}"
        ],
    }
    _write(environment_path, environment)
    eval_config_path = context.config_dir / "sealed_eval.json"
    experiment_path = context.config_dir / f"{condition}_seed{seed}.json"
    eval_config = json.loads(eval_config_path.read_text())
    experiment = json.loads(experiment_path.read_text())
    provenance = {
        "run_label": label,
        "evaluation_seed": 0,
        "process_command": {
            "python_executable": context.spec["python_executable"],
            "argv": [
                str((context.repository / final.AUDITED_SOURCE_PATHS["eval"]).resolve()),
                *final._expected_eval_arguments(context, label),
            ],
            "cwd": str(context.repository.resolve()),
        },
        "environment_snapshot": {
            "path": str(environment_path.resolve()),
            "sha256": final.sha256_file(environment_path),
        },
        "model": {
            "name": context.spec["model"]["name"],
            "configured_revision": context.spec["model"]["revision"],
            "resolved_revision": context.spec["model"]["revision"],
            "dtype": context.spec["model"]["dtype"],
        },
        "adapter": None if is_base else final._adapter_identity(context, label),
        "evaluation_config": {
            "file_sha256": final.sha256_file(eval_config_path),
            "resolved_sha256": final.canonical_sha256(eval_config),
        },
        "experiment_config": {
            "file_sha256": final.sha256_file(experiment_path),
            "resolved_sha256": final.canonical_sha256(experiment),
            "source": "explicit",
        },
        "experiment": {
            "training_seed": seed,
            "reward_type": "jlens",
            "target_words": context.spec["target_words"],
            "score_components": experiment["score_components"],
            "lens_sha256": context.spec["artifacts"]["lens_sha256"],
            "calibration_sha256": context.spec["artifacts"]["calibration_sha256"],
            "expected_lens_sha256": context.spec["artifacts"]["lens_sha256"],
            "expected_calibration_sha256": context.spec["artifacts"][
                "calibration_sha256"
            ],
        },
        "selection": {
            "method": "index_manifest",
            "indices_sha256": final.canonical_sha256(references.indices),
            "index_manifest": {
                "sha256": context.spec["final_collection"]["manifest_sha256"],
                "dataset": context.spec["dataset"]["name"],
                "subset": context.spec["dataset"]["subset"],
                "split": context.spec["dataset"]["split"],
                "count": final.FINAL_EXAMPLES,
            },
        },
        "git": {
            "git_commit": context.spec["git_commit"],
            "git_dirty": False,
            "source_tree_sha256": context.spec["source_tree_sha256"],
        },
        "software": context.spec["software"],
        "runtime": {
            "cuda_device_name": context.spec["hardware"]["device_name"],
            "cuda_version": context.spec["hardware"]["cuda_version"],
            "batch_size": 64,
        },
    }
    generation = {
        "do_sample": False,
        "max_prompt_tokens": 384,
        "max_new_tokens": 256,
        "padding_side": "left",
    }
    is_correct = _correctness_predicate(correct)
    rows = []
    for source_index in references.indices:
        token_ids = [source_index, int(is_correct(source_index))]
        completion = references.decode_completion(token_ids)
        rows.append(
            {
                "schema_version": 1,
                "dataset": {
                    "name": context.spec["dataset"]["name"],
                    "subset": context.spec["dataset"]["subset"],
                    "split": context.spec["dataset"]["split"],
                    "revision": context.spec["dataset"]["revision"],
                    "fingerprint": references.dataset_fingerprint,
                },
                "source_index": source_index,
                "prompt_sha256": references.prompt_sha256[source_index],
                "prompt_token_ids_sha256": references.prompt_token_ids_sha256[source_index],
                "completion": completion,
                "completion_token_ids": token_ids,
                "prediction": references.extract_answer(completion),
                "correct": references.is_correct(
                    completion, references.answers[source_index]
                ),
                "completion_tokens": len(token_ids),
                "target_words": context.spec["target_words"],
                "literal_target_matches": [],
                "literal_target_used": False,
                "generation": generation,
                "provenance": provenance,
            }
        )
    output = context.eval_dir / f"{label}.jsonl"
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    dispatch = context.evidence_dir / "final_dispatches"
    logs = context.evidence_dir / "sealed_collection_logs"
    stdout_path = logs / f"{label}.stdout"
    stderr_path = logs / f"{label}.stderr"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text("")
    stderr_path.write_text("")
    collection_id = json.loads(context.collection_path.read_text())["collection_id"]
    sequence = final.FINAL_LABELS.index(label) + 1
    intent = {
        "schema_version": 1,
        "protocol": context.spec["protocol"],
        "collection_id": collection_id,
        "sequence": sequence,
        "label": label,
        "hardware": {
            **context.spec["hardware"],
            "observed_gpu_uuid": "GPU-00000000-0000-0000-0000-000000000000",
        },
        "command": final.expected_eval_command(context, label),
        "cwd": str(context.repository.resolve()),
        "environment_overrides": final.expected_runtime_overrides(context),
        "status": "written_and_fsynced_before_gpu_process",
        "outcome_inspected_before_full_collection": False,
    }
    intent_path = dispatch / f"{label}.intent.json"
    _write(intent_path, intent)
    _write(
        dispatch / f"{label}.completion.json",
        {
            "schema_version": 1,
            "protocol": context.spec["protocol"],
            "collection_id": collection_id,
            "sequence": sequence,
            "label": label,
            "intent_sha256": final.sha256_file(intent_path),
            "jsonl_sha256": final.sha256_file(output),
            "environment_sha256": final.sha256_file(environment_path),
            "stdout_sha256": final.sha256_file(stdout_path),
            "stderr_sha256": final.sha256_file(stderr_path),
            "returncode": 0,
            "outcome_inspected_before_full_collection": False,
            "command": final.expected_eval_command(context, label),
            "cwd": str(context.repository.resolve()),
            "environment_overrides": final.expected_runtime_overrides(context),
        },
    )


def _analysis_process(context: final.FinalContext) -> dict[str, object]:
    audit = final.verify_automation_audit(context)
    return {
        "schema_version": 1,
        "python_executable": context.spec["python_executable"],
        "command": final.expected_analysis_command(context),
        "cwd": str(context.repository.resolve()),
        "environment_overrides": final.expected_runtime_overrides(context),
        "input_sha256": {
            f"{label}.jsonl": final.sha256_file(context.eval_dir / f"{label}.jsonl")
            for label in final.FINAL_LABELS
        },
        "source_probe_command": final.expected_analysis_probe_command(context),
        "source_probe_returncode": 0,
        "loaded_source_identity": {
            "jlens_rl.common": {
                "path": str(
                    (context.repository / final.AUDITED_SOURCE_PATHS["common"]).resolve()
                ),
                "sha256": audit["source_sha256"]["common"],
            },
            "jlens_rl.paired_eval": {
                "path": str(
                    (
                        context.repository / final.AUDITED_SOURCE_PATHS["paired_eval"]
                    ).resolve()
                ),
                "sha256": audit["source_sha256"]["paired_eval"],
            },
        },
        "environment_snapshot": {
            "python": {"executable": context.spec["python_executable"]},
            "pip_freeze_all": ["a==1"],
            "torch": {
                "version": context.spec["hardware"]["torch_version"],
                "cuda_build": context.spec["hardware"]["cuda_version"],
            },
        },
    }


def test_design_is_four_fresh_seeds_and_exact_requested_curve() -> None:
    assert final.SEEDS == tuple(range(224, 228))
    assert final.CURVE_STEPS == (0, 4, 5, 6)
    assert final.TERMINAL_STEP == 6
    assert final.TARGET_WORDS == ("yay", "great", "success", "nice")
    assert final.TREATMENT_SCORE_COMPONENTS == (
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
    assert len(final.FINAL_LABELS) == 9
    assert final.ACCEPTANCE_REGISTRATION[
        "treatment_vs_base_exact_two_sided_sign_p"
    ] == 0.125
    assert final.curve_means_pass([0.39, 0.40, 0.40, 0.41])
    assert not final.curve_means_pass([0.39, 0.39, 0.40, 0.41])
    assert not final.curve_means_pass([0.39, 0.40, 0.399, 0.41])


def test_module_is_frozen_and_inert_until_registration() -> None:
    summary = final.design_summary()
    assert summary["protected_manifest_payload_accessed_here"] is False
    assert "frozen_inert" in summary["status"]
    source = Path(final.__file__).read_text().lower()
    assert final.FINAL_MANIFEST_SHA256 in source
    assert 'target_words = ("yay", "great", "success", "nice")' in source


def test_spec_rejects_design_schema_and_signflip_mutations(tmp_path: Path) -> None:
    context = _future_context(tmp_path)
    mutations = []
    value = copy.deepcopy(context.spec)
    value["seeds"] = list(range(224, 232))
    mutations.append(value)
    value = copy.deepcopy(context.spec)
    value["training"]["updates"] = 5
    mutations.append(value)
    value = copy.deepcopy(context.spec)
    value["target_words"] = ["solved"]
    mutations.append(value)
    value = copy.deepcopy(context.spec)
    value["matched_control_score_components"][0]["weight"] = -0.5
    mutations.append(value)
    value = copy.deepcopy(context.spec)
    value["final_collection"]["manifest_metadata"]["split"] = "test"
    mutations.append(value)
    for malformed in mutations:
        with pytest.raises(final.FinalProtocolError):
            final.validate_spec(malformed)


def test_preunlock_reads_no_final_and_binds_configs_and_audit(tmp_path: Path) -> None:
    context = _future_context(tmp_path)
    final._final_manifest_path(context).unlink()
    ready = final.verify_preunlock_readiness(context)
    assert ready["protected_final_manifest_read"] is False
    assert ready["protected_payloads_accessed"] is False


def test_post_registration_science_drift_is_rejected(tmp_path: Path) -> None:
    context = _future_context(tmp_path)
    context.spec["wandb"]["tags"].append("post-registration-drift")
    with pytest.raises(final.FinalProtocolError, match="registration does not commit"):
        final.verify_preunlock_readiness(context)


def test_config_schema_hash_and_exact_signflip_are_enforced(tmp_path: Path) -> None:
    context = _future_context(tmp_path)
    config_path = context.config_dir / f"signflip_seed{final.SEEDS[0]}.json"
    config = json.loads(config_path.read_text())
    config["score_components"][0]["weight"] = -0.5
    _write(config_path, config)
    with pytest.raises(final.FinalProtocolError, match="config schema/hash"):
        final.verify_preunlock_readiness(context)


def test_exact_completed_inventory_and_curve_recomputation(tmp_path: Path) -> None:
    context = _future_context(tmp_path)
    inventory = json.loads(context.completed_runs_path.read_text())
    inventory["runs"].pop(f"jlens_seed{final.SEEDS[0]}")
    _write(context.completed_runs_path, inventory)
    with pytest.raises(final.FinalProtocolError, match="exact 8"):
        final.verify_preclaim(context)

    context = _future_context(tmp_path / "curve")
    curve = json.loads(context.curve_path.read_text())
    curve["mean_exact_match"]["5"] = 0.99
    _write(context.curve_path, curve)
    unlock = json.loads(context.unlock_path.read_text())
    unlock["curve_gate_sha256"] = final.sha256_file(context.curve_path)
    _write(context.unlock_path, unlock)
    with pytest.raises(final.FinalProtocolError, match="curve gate"):
        final.verify_preclaim(context)


def test_run_result_receipt_history_adapter_and_hardware_are_enforced(
    tmp_path: Path,
) -> None:
    context = _future_context(tmp_path)
    label = f"jlens_seed{final.SEEDS[0]}"
    manifest_path = context.run_dir / label / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["runtime"]["cuda_device_name"] = "other GPU"
    _write(manifest_path, manifest)
    with pytest.raises(final.FinalProtocolError, match="hardware identity"):
        final.verify_preclaim(context)


@pytest.mark.parametrize(
    "mutation",
    (
        "run_result", "wandb_receipt", "wandb_tree", "history",
        "log_history_semantic", "terminal_adapter", "audited_test",
    ),
)
def test_each_completed_evidence_binding_rejects_mutation(
    tmp_path: Path, mutation: str
) -> None:
    context = _future_context(tmp_path)
    label = f"jlens_seed{final.SEEDS[0]}"
    directory = context.run_dir / label
    if mutation == "run_result":
        path = directory / "run_result_manifest.json"
        value = json.loads(path.read_text())
        value["completed_updates"] = final.TERMINAL_STEP - 1
        _write(path, value)
    elif mutation == "wandb_receipt":
        path = directory / "wandb_terminal_publish_receipt.json"
        value = json.loads(path.read_text())
        value["observed_wandb_identity"] = {}
        _write(path, value)
    elif mutation == "wandb_tree":
        receipt = json.loads(
            (directory / "wandb_terminal_publish_receipt.json").read_text()
        )
        receipt["artifact"]["digest"] = ""
        _write(directory / "wandb_terminal_publish_receipt.json", receipt)
    elif mutation == "history":
        path = directory / "validation_history.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[1]["exact_match"] = 0.99
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    elif mutation == "log_history_semantic":
        path = directory / "log_history.json"
        rows = json.loads(path.read_text())
        rows[1]["step"] = 1
        _write(path, rows)
        result_path = directory / "run_result_manifest.json"
        result = json.loads(result_path.read_text())
        result["raw_history_sha256"]["log_history.json"] = final.sha256_file(path)
        _write(result_path, result)
    elif mutation == "terminal_adapter":
        (directory / "final" / "adapter_model.safetensors").write_bytes(b"changed")
    else:
        path = context.repository / final.AUDITED_TEST_PATHS[1]
        path.write_text(path.read_text() + "\n# changed audited test\n")
    with pytest.raises(final.FinalProtocolError):
        final.verify_preclaim(context)


def test_reward_std_verifier_allows_float32_reduction_noise_only(
    tmp_path: Path,
) -> None:
    context = _future_context(tmp_path)
    label = f"jlens_seed{final.SEEDS[0]}"
    directory = context.run_dir / label
    config = json.loads((context.config_dir / f"{label}.json").read_text())
    history = final._history_rows(directory / "validation_history.jsonl")
    log_path = directory / "log_history.json"
    rows = json.loads(log_path.read_text())
    reward_std_key = (
        f"rewards/jlens_{'_'.join(config['target_words'])}_reward/std"
    )

    rows[0][reward_std_key] = float(rows[0]["reward_std"]) + 3e-8
    _write(log_path, rows)
    final._training_behavior_summary(log_path, config, history)

    rows[0][reward_std_key] = float(rows[0]["reward_std"]) + 1.1e-7
    _write(log_path, rows)
    with pytest.raises(final.FinalProtocolError, match="one-J-reward behavior"):
        final._training_behavior_summary(log_path, config, history)


def test_validation_merged_reward_rows_may_omit_only_their_learning_rate(
    tmp_path: Path,
) -> None:
    context = _future_context(tmp_path)
    label = f"jlens_seed{final.SEEDS[0]}"
    directory = context.run_dir / label
    config = json.loads((context.config_dir / f"{label}.json").read_text())
    history = final._history_rows(directory / "validation_history.jsonl")
    log_path = directory / "log_history.json"
    rows = json.loads(log_path.read_text())
    validation = {
        int(row["step"]): row
        for row in rows
        if "validation/exact_match" in row
    }

    # The trainer can merge evaluation scalars into the optimizer log at an
    # evaluation step.  In that representation W&B omits learning_rate from
    # precisely those merged rows.  Baseline step 0 is recorded in
    # validation_history, not in the trainer log_history.
    merged: list[dict[str, object]] = []
    for row in rows:
        if "validation/exact_match" in row:
            continue
        if "reward" in row and int(row["step"]) in final.CURVE_STEPS[1:]:
            row = {**row, **validation[int(row["step"])]}
            row.pop("learning_rate")
        merged.append(row)
    _write(log_path, merged)
    summary = final._training_behavior_summary(log_path, config, history)
    assert summary["optimizer_steps"] == config["updates"]
    assert summary["validation_steps"] == list(final.CURVE_STEPS[1:])
    assert summary["learning_rate_rows"] == 3

    wrong_at_eval = copy.deepcopy(merged)
    next(
        row
        for row in wrong_at_eval
        if row.get("step") == 4 and "reward" in row
    )["learning_rate"] = float(config["learning_rate"]) * 2
    _write(log_path, wrong_at_eval)
    with pytest.raises(final.FinalProtocolError, match="one-J-reward behavior"):
        final._training_behavior_summary(log_path, config, history)

    missing_off_eval = copy.deepcopy(merged)
    next(
        row
        for row in missing_off_eval
        if row.get("step") == 2 and "reward" in row
    ).pop("learning_rate")
    _write(log_path, missing_off_eval)
    with pytest.raises(final.FinalProtocolError, match="one-J-reward behavior"):
        final._training_behavior_summary(log_path, config, history)


def test_firewall_receipt_and_training_disjointness_are_enforced(tmp_path: Path) -> None:
    context = _future_context(tmp_path)
    receipt_path = Path(context.spec["firewall"]["disjointness_receipt"]["path"])
    receipt = json.loads(receipt_path.read_text())
    receipt["protected_final_outcomes_read"] = True
    _write(receipt_path, receipt)
    context.spec["firewall"]["disjointness_receipt"]["sha256"] = final.sha256_file(
        receipt_path
    )
    registration_path = context.repository / context.spec["registration_path"]
    recipe_path = context.repository / context.spec["recipe_lock_path"]
    _write(registration_path, final.expected_registration_document(context.spec))
    _write(recipe_path, final.expected_recipe_lock_document(context.spec))
    context.spec["registration_sha256"] = final.sha256_file(registration_path)
    context.spec["recipe_lock_sha256"] = final.sha256_file(recipe_path)
    with pytest.raises(final.FinalProtocolError, match="ineligible"):
        final.verify_preunlock_readiness(context)


def test_claim_requires_exact_clean_commit_tree_and_audit_bytes(tmp_path: Path) -> None:
    context = _future_context(tmp_path)
    dirty_path = context.repository / "untracked-dirty-source.txt"
    dirty_path.write_text("dirty\n")
    with pytest.raises(final.FinalProtocolError, match="exact clean audited"):
        final.begin_final_collection(context, "4" * 32)
    dirty_path.unlink()
    paired_source = context.repository / final.AUDITED_SOURCE_PATHS["paired_eval"]
    paired_source.write_text(paired_source.read_text() + "\n# mutation\n")
    with pytest.raises(final.FinalProtocolError, match="exact bytes"):
        final.verify_preunlock_readiness(context)


def test_future_training_entrypoint_is_covered_by_exact_code_audit(
    tmp_path: Path,
) -> None:
    context = _future_context(tmp_path)
    audit = json.loads(
        Path(context.spec["automation_audit"]["path"]).read_text()
    )
    assert "training_entrypoint" in audit["source_sha256"]
    entrypoint = context.repository / final.AUDITED_SOURCE_PATHS["training_entrypoint"]
    entrypoint.write_text(entrypoint.read_text() + "changed\n")
    with pytest.raises(final.FinalProtocolError, match="exact bytes"):
        final.verify_preunlock_readiness(context)


def test_final_manifest_metadata_range_is_opened_only_after_irrevocable_claim(
    tmp_path: Path,
) -> None:
    invalid_indices = list(range(400, 1300))
    invalid_indices[-1] = 4000
    context = _future_context(tmp_path, final_indices_override=invalid_indices)
    with pytest.raises(final.FinalProtocolError, match="range/disjointness"):
        _begin(context)
    assert context.collection_path.is_file()
    assert context.failure_path.is_file()
    with pytest.raises(final.FinalProtocolError, match="continuation is forbidden"):
        final.verify_preunlock_readiness(context)


def test_raw_final_verifier_recomputes_every_outcome_and_rejects_mutation(
    tmp_path: Path,
) -> None:
    context = _future_context(tmp_path)
    _begin(context)
    references = _references(context)
    _write_synthetic_label(context, "base", references, correct=False)
    final.verify_evaluation_jsonl(context, "base", references=references)
    output = context.eval_dir / "base.jsonl"
    rows = output.read_text().splitlines()
    first = json.loads(rows[0])
    first["correct"] = True
    rows[0] = json.dumps(first, sort_keys=True)
    output.write_text("\n".join(rows) + "\n")
    completion_path = context.evidence_dir / "final_dispatches" / "base.completion.json"
    completion = json.loads(completion_path.read_text())
    completion["jsonl_sha256"] = final.sha256_file(output)
    _write(completion_path, completion)
    with pytest.raises(final.FinalProtocolError, match="derived completion outcome"):
        final.verify_evaluation_jsonl(context, "base", references=references)


def test_dispatch_inventory_logs_sequence_command_and_argv0_are_bound(tmp_path: Path) -> None:
    context = _future_context(tmp_path)
    _begin(context)
    references = _references(context)
    _write_synthetic_label(context, "base", references, correct=False)
    log_path = context.evidence_dir / "sealed_collection_logs" / "base.stdout"
    log_path.write_text("changed")
    with pytest.raises(final.FinalProtocolError, match="dispatch changed"):
        final._verify_dispatch(context, "base")
    process = {
        "python_executable": context.spec["python_executable"],
        "argv": ["/wrong/eval.py", *final._expected_eval_arguments(context, "base")],
        "cwd": str(context.repository.resolve()),
    }
    with pytest.raises(final.FinalProtocolError, match="command changed"):
        final._verify_process_command(context, process, "base")


def test_runner_refuses_partial_label_resume_before_gpu_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _future_context(tmp_path)
    collection_id = _begin(context)
    context.eval_dir.mkdir(parents=True)
    (context.eval_dir / "base.jsonl").write_text("partial\n")
    monkeypatch.setattr(
        runner,
        "probe_hardware",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not probe")),
    )
    with pytest.raises(final.FinalProtocolError, match="never resume"):
        runner._run_one_label(context, collection_id, "base", 1)


def test_runner_rejects_out_of_order_label_before_gpu_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _future_context(tmp_path)
    collection_id = _begin(context)
    monkeypatch.setattr(
        runner,
        "probe_hardware",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not probe")),
    )
    with pytest.raises(final.FinalProtocolError, match="out of registered order"):
        runner._run_one_label(
            context, collection_id, f"signflip_seed{final.SEEDS[-1]}", 1
        )


def test_runner_collects_all_labels_before_any_verification_or_analysis() -> None:
    source = Path(runner.__file__).read_text()
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_final_collection"
    )
    body = ast.get_source_segment(source, function) or ""
    assert body.index("_run_one_label") < body.index("verify_all_evaluations")
    assert body.index("verify_all_evaluations") < body.index("_run_analysis")
    assert body.index("_run_analysis") < body.index("final_report")
    assert "for sequence, label in enumerate(protocol.FINAL_LABELS, 1)" in body


def test_analysis_forces_bound_pythonpath_and_loaded_source_identity(tmp_path: Path) -> None:
    context = _future_context(tmp_path)
    context.eval_dir.mkdir(parents=True)
    for label in final.FINAL_LABELS:
        (context.eval_dir / f"{label}.jsonl").write_text("synthetic\n")
    process = _analysis_process(context)
    _write(context.analysis_process_path, process)
    assert final.verify_analysis_process(context) == process
    original = copy.deepcopy(process)
    process["environment_overrides"]["PYTHONPATH"] = "/wrong"  # type: ignore[index]
    _write(context.analysis_process_path, process)
    with pytest.raises(final.FinalProtocolError, match="analysis command/input"):
        final.verify_analysis_process(context)
    process = original
    process["loaded_source_identity"]["jlens_rl.paired_eval"]["sha256"] = "0" * 64  # type: ignore[index]
    _write(context.analysis_process_path, process)
    with pytest.raises(final.FinalProtocolError, match="analysis command/input"):
        final.verify_analysis_process(context)


def test_runner_rejects_wrong_analysis_probe_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _future_context(tmp_path)
    _begin(context)
    context.eval_dir.mkdir(parents=True)
    for label in final.FINAL_LABELS:
        (context.eval_dir / f"{label}.jsonl").write_text("synthetic\n")
    environment = _analysis_process(context)["environment_snapshot"]
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "loaded_source_identity": {"wrong": True},
                    "environment_snapshot": environment,
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    with pytest.raises(final.FinalProtocolError, match="loaded source/environment"):
        runner._run_analysis(context)
    assert calls == [final.expected_analysis_probe_command(context)]
    assert not context.analysis_process_path.exists()
    assert not context.comparison_path.exists()


def test_real_paired_and_did_recomputation_drives_primary_acceptance(tmp_path: Path) -> None:
    context = _future_context(tmp_path)
    _begin(context)
    references = _references(context)
    _write_synthetic_label(context, "base", references, correct=200)
    for seed in final.SEEDS:
        _write_synthetic_label(
            context, f"jlens_seed{seed}", references, correct=500
        )
        _write_synthetic_label(
            context, f"signflip_seed{seed}", references, correct=250
        )
    comparison = final.recompute_comparison(context, references=references)
    _write(context.comparison_path, comparison)
    _write(context.analysis_process_path, _analysis_process(context))
    report = final.final_report(context, references=references)
    assert report["passed"] is True
    assert report["recipe_sha256"] == context.spec["recipe_sha256"]
    assert report["registered_code_sha256"] == context.spec["registered_code_sha256"]
    assert report["registered_spec_projection_sha256"] == (
        final.registered_spec_projection_sha256(context.spec)
    )
    assert comparison["mean_accuracy_difference"] > 0
    assert comparison["difference_in_differences"][
        "mean_difference_in_differences"
    ] > 0
    assert math.isclose(comparison["mean_accuracy_difference"], 300 / 900)
    assert math.isclose(
        comparison["difference_in_differences"]["mean_difference_in_differences"],
        250 / 900,
    )
    assert comparison["seed_sign_test"] == {
        "positive": 4,
        "negative": 0,
        "tied_excluded": 0,
        "exact_two_sided_p": 0.125,
    }
    assert comparison["difference_in_differences"]["seed_sign_test"] == {
        "positive": 4,
        "negative": 0,
        "tied_excluded": 0,
        "exact_two_sided_p": 0.125,
    }
    (context.evidence_dir / "sealed_collection_logs" / "extra-directory").mkdir()
    (context.evidence_dir / "final_dispatches" / "extra-directory").mkdir()
    with pytest.raises(final.FinalProtocolError, match="inventory is not exact"):
        final.verify_dispatch_inventory(context)


def test_alpha_15_sign_gates_pass_when_descriptive_95pct_intervals_touch_zero(
    tmp_path: Path,
) -> None:
    context = _future_context(tmp_path)
    _begin(context)
    references = _references(context)
    _write_synthetic_label(context, "base", references, correct=200)
    for seed in final.SEEDS:
        _write_synthetic_label(
            context, f"jlens_seed{seed}", references, correct=201
        )
        _write_synthetic_label(
            context, f"signflip_seed{seed}", references, correct=200
        )
    comparison = final.recompute_comparison(context, references=references)
    assert comparison["crossed_seed_item_bootstrap"][
        "mean_accuracy_difference_ci_low"
    ] <= 0
    assert comparison["difference_in_differences"]["crossed_seed_item_bootstrap"][
        "mean_difference_in_differences_ci_low"
    ] <= 0
    _write(context.comparison_path, comparison)
    _write(context.analysis_process_path, _analysis_process(context))
    report = final.final_report(context, references=references)
    assert report["passed"] is True
    assert report["descriptive_crossed_95pct_intervals"][
        "used_as_acceptance_gate"
    ] is False
    assert "treatment_crossed_ci_low_positive" not in report["checks"]
    assert "matched_signflip_crossed_ci_low_positive" not in report["checks"]


def test_real_negative_did_fails_acceptance_even_when_treatment_beats_base(
    tmp_path: Path,
) -> None:
    context = _future_context(tmp_path)
    _begin(context)
    references = _references(context)
    _write_synthetic_label(context, "base", references, correct=200)
    for seed in final.SEEDS:
        _write_synthetic_label(
            context, f"jlens_seed{seed}", references, correct=500
        )
        _write_synthetic_label(
            context, f"signflip_seed{seed}", references, correct=600
        )
    comparison = final.recompute_comparison(context, references=references)
    _write(context.comparison_path, comparison)
    _write(context.analysis_process_path, _analysis_process(context))
    report = final.final_report(context, references=references)
    assert comparison["mean_accuracy_difference"] > 0
    assert comparison["difference_in_differences"][
        "mean_difference_in_differences"
    ] < 0
    assert report["passed"] is False
    assert not report["checks"][
        "matched_signflip_difference_in_differences_mean_positive"
    ]
