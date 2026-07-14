from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import torch

# The vendored repository directory and its Python package are both named `trl`.
# Put the repository root on sys.path so it wins over the outer namespace directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "trl"))

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from .common import (
    GSM8K_REVISION,
    QWEN_MODEL_REVISION,
    SYSTEM_PROMPT,
    append_jsonl,
    extract_answer,
    load_config,
    load_index_manifest,
    model_dtype,
    require_clean_repository_provenance,
    repository_provenance,
    resolve_repository_root,
    runtime_environment_snapshot,
    seed_everything,
    sha256_file,
)
from .eval import evaluate
from .reward import TRLTargetJLReward, TargetJLReward


TERMINAL_EVIDENCE_FILE_NAMES = (
    "run_result_manifest.json",
    "validation_history.jsonl",
    "log_history.json",
    "environment_snapshot.json",
    "run_manifest.json",
    "resolved_config.json",
    "data_indices.json",
)


def _write_json_atomic(path: Path, payload: Any) -> None:
    """Publish a JSON record only after its complete bytes are on disk."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def gsm8k_reward_trl(completions: list[list[dict[str, str]]], answer: list[str], **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [float(extract_answer(text) == extract_answer(gold)) for text, gold in zip(contents, answer, strict=True)]


def prepare_example(example: dict[str, str]) -> dict[str, Any]:
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ],
        "answer": example["answer"],
    }


def prepare_prompt(example: dict[str, str]) -> dict[str, Any]:
    """Policy input for J-only training; intentionally excludes the gold answer."""
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ]
    }


def create_run_directory(path: str | Path) -> Path:
    output_dir = Path(path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; use a new run directory"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def configure_wandb_environment(cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Apply optional byte-pinned W&B identity fields from a run config."""
    os.environ["WANDB_PROJECT"] = cfg["wandb_project"]
    os.environ["WANDB_MODE"] = cfg["wandb_mode"]
    optional_environment = {
        "wandb_entity": "WANDB_ENTITY",
        "wandb_run_id": "WANDB_RUN_ID",
        "wandb_group": "WANDB_RUN_GROUP",
        "wandb_resume": "WANDB_RESUME",
    }
    for config_key, environment_key in optional_environment.items():
        value = cfg.get(config_key)
        if value is not None:
            if not isinstance(value, str) or not value:
                raise ValueError(f"{config_key} must be a non-empty string")
            os.environ[environment_key] = value
        else:
            os.environ.pop(environment_key, None)
    wandb_tags = cfg.get("wandb_tags")
    if wandb_tags is not None:
        if (
            not isinstance(wandb_tags, list)
            or not wandb_tags
            or any(not isinstance(tag, str) or not tag for tag in wandb_tags)
        ):
            raise ValueError("wandb_tags must be a non-empty list of strings")
        os.environ["WANDB_TAGS"] = ",".join(wandb_tags)
    else:
        os.environ.pop("WANDB_TAGS", None)
    if "wandb_run_id" not in cfg:
        return None
    return {
        "entity": cfg.get("wandb_entity"),
        "project": cfg["wandb_project"],
        "run_name": cfg.get("run_name"),
        "run_id": cfg["wandb_run_id"],
        "url": cfg.get("wandb_url"),
        "group": cfg.get("wandb_group"),
        "tags": cfg.get("wandb_tags"),
        "resume": cfg.get("wandb_resume"),
    }


class DeterministicValidationCallback(TrainerCallback):
    """Log fixed greedy GSM8K accuracy, with optional exploratory stopping."""

    def __init__(self, tokenizer: Any, rows: Any, cfg: dict[str, Any]) -> None:
        self.tokenizer = tokenizer
        self.rows = rows
        self.cfg = cfg
        self.trainer: Any = None
        self.best_exact_match: float | None = None
        self.evaluations_without_improvement = 0
        self.validation_identity = {
            "validation_source": cfg.get("validation_source", "test"),
            "validation_indices_sha256": (
                sha256_file(cfg["validation_indices_path"])
                if cfg.get("validation_indices_path")
                else None
            ),
        }

    def evaluate_and_log(self, model: Any, step: int) -> dict[str, float]:
        metrics = evaluate(
            model, self.tokenizer, self.rows, self.cfg, None,
            self.cfg.get("validation_batch_size", 16),
        )
        append_jsonl(
            Path(self.cfg["output_dir"]) / "validation_history.jsonl",
            {"step": step, **self.validation_identity, **metrics},
        )
        self.trainer.log({f"validation/{key}": value for key, value in metrics.items()})
        return metrics

    def on_step_end(self, args, state, control, model=None, **kwargs):
        evaluation_steps = self.cfg.get("validation_steps")
        should_evaluate = (
            state.global_step in evaluation_steps
            if evaluation_steps is not None
            else state.global_step % self.cfg["eval_every"] == 0
        )
        if should_evaluate:
            metrics = self.evaluate_and_log(model, state.global_step)
            score = metrics["exact_match"]
            min_delta = self.cfg.get("early_stopping_min_delta", 0.0)
            if self.best_exact_match is None or score > self.best_exact_match + min_delta:
                self.best_exact_match = score
                self.evaluations_without_improvement = 0
            else:
                self.evaluations_without_improvement += 1
            patience = (
                None
                if self.cfg.get("validation_observational_only", False)
                else self.cfg.get("early_stopping_patience")
            )
            stopping_start = self.cfg.get("early_stopping_start_step", 0)
            if (
                patience
                and state.global_step >= stopping_start
                and self.evaluations_without_improvement >= patience
            ):
                control.should_training_stop = True
        return control


class RunManifestCallback(TrainerCallback):
    """Attach dirty-tree and artifact identity to the remote run record."""

    def __init__(self, manifest: dict[str, Any], output_dir: Path, enabled: bool) -> None:
        self.manifest = manifest
        self.output_dir = output_dir
        self.enabled = enabled

    def on_train_begin(self, args, state, control, **kwargs):
        if not self.enabled:
            return control
        import wandb

        if wandb.run is not None:
            wandb.config.update(
                {
                    "experiment_manifest": self.manifest,
                    "metric_schema": self.manifest.get("metric_schema"),
                    "wandb_identity": self.manifest.get("wandb_identity"),
                },
                allow_val_change=True,
            )
            for name in (
                "run_manifest.json",
                "resolved_config.json",
                "data_indices.json",
                "environment_snapshot.json",
            ):
                wandb.save(
                    str(self.output_dir / name),
                    base_path=str(self.output_dir),
                    policy="now",
                )
        return control


def _tree_identity(path: Path) -> dict[str, Any]:
    files = {
        file.relative_to(path).as_posix(): sha256_file(file)
        for file in sorted(path.rglob("*"))
        if file.is_file()
    }
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "files": files,
    }


def write_run_result_manifest(
    *,
    output_dir: Path,
    cfg: dict[str, Any],
    run_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Write the terminal replay/forensics record for a pinned run."""
    checkpoint = output_dir / f"checkpoint-{cfg['updates']}"
    final = output_dir / "final"
    required_files = {
        "run_manifest.json": output_dir / "run_manifest.json",
        "resolved_config.json": output_dir / "resolved_config.json",
        "data_indices.json": output_dir / "data_indices.json",
        "validation_history.jsonl": output_dir / "validation_history.jsonl",
        "log_history.json": output_dir / "log_history.json",
        "environment_snapshot.json": output_dir / "environment_snapshot.json",
    }
    missing = [name for name, path in required_files.items() if not path.is_file()]
    if missing or not checkpoint.is_dir() or not final.is_dir():
        raise RuntimeError(f"cannot finalize run-result manifest; missing {missing}")
    result = {
        "schema_version": 1,
        "completed_updates": cfg["updates"],
        "wandb_identity": run_manifest.get("wandb_identity"),
        "metric_schema": run_manifest.get("metric_schema"),
        "process_command": run_manifest["process_command"],
        "registered_command": cfg.get("registered_command"),
        "registration_sha256": cfg.get("registration_sha256"),
        "recipe_lock_sha256": cfg.get("recipe_lock_sha256"),
        "recipe_sha256": cfg.get("recipe_sha256"),
        "evidence_eligibility": cfg.get("evidence_eligibility"),
        "reproduction_source": cfg.get("reproduction_source"),
        "source": {
            key: run_manifest.get(key)
            for key in ("git_commit", "git_dirty", "source_tree_sha256")
        },
        "runtime": run_manifest["runtime"],
        "data_indices_sha256": run_manifest["data_indices_sha256"],
        "lens_sha256": run_manifest.get("lens_sha256"),
        "calibration_sha256": run_manifest.get("calibration_sha256"),
        "raw_history_sha256": {
            name: sha256_file(path) for name, path in required_files.items()
        },
        "terminal_checkpoint": _tree_identity(checkpoint),
        "final_adapter_and_tokenizer": _tree_identity(final),
    }
    path = output_dir / "run_result_manifest.json"
    _write_json_atomic(path, result)
    return result


def _expected_observed_wandb_identity(identity: Any) -> dict[str, Any]:
    if not isinstance(identity, dict):
        raise RuntimeError("terminal result lacks its frozen W&B identity")
    expected = {
        "run_id": identity.get("run_id"),
        "entity": identity.get("entity"),
        "project": identity.get("project"),
        "run_name": identity.get("run_name"),
        "url": identity.get("url"),
        "group": identity.get("group"),
        "tags": identity.get("tags"),
    }
    if (
        any(
            not isinstance(expected[key], str) or not expected[key]
            for key in ("run_id", "entity", "project", "run_name", "url", "group")
        )
        or not isinstance(expected["tags"], list)
        or not expected["tags"]
        or any(not isinstance(tag, str) or not tag for tag in expected["tags"])
    ):
        raise RuntimeError("frozen W&B identity is incomplete")
    return expected


def _observe_active_wandb_identity(run: Any, identity: Any) -> dict[str, Any]:
    """Read back every W&B run field that the confirmatory config freezes."""
    expected = _expected_observed_wandb_identity(identity)
    raw_tags = getattr(run, "tags", None)
    observed = {
        "run_id": getattr(run, "id", None),
        "entity": getattr(run, "entity", None),
        "project": getattr(run, "project", None),
        "run_name": getattr(run, "name", None),
        "url": getattr(run, "url", None),
        "group": getattr(run, "group", None),
        "tags": list(raw_tags) if isinstance(raw_tags, (list, tuple)) else raw_tags,
    }
    if observed != expected:
        raise RuntimeError(
            "active W&B run does not match the frozen observable identity: "
            f"{observed!r} != {expected!r}"
        )
    return observed


def _validate_terminal_artifact_identity(
    artifact_identity: Any, wandb_identity: Any
) -> dict[str, Any]:
    expected_run = _expected_observed_wandb_identity(wandb_identity)
    if not isinstance(artifact_identity, dict):
        raise RuntimeError("W&B terminal artifact identity is absent")
    version = artifact_identity.get("version")
    base_name = f"{expected_run['run_id']}-terminal-evidence"
    expected_name = f"{base_name}:{version}"
    expected_qualified_name = (
        f"{expected_run['entity']}/{expected_run['project']}/{expected_name}"
    )
    if (
        not isinstance(artifact_identity.get("id"), str)
        or not artifact_identity["id"]
        or not isinstance(artifact_identity.get("digest"), str)
        or not artifact_identity["digest"]
        or not isinstance(version, str)
        or re.fullmatch(r"v[0-9]+", version) is None
        or artifact_identity.get("name") != expected_name
        or artifact_identity.get("qualified_name") != expected_qualified_name
    ):
        raise RuntimeError("W&B did not confirm the exact terminal evidence artifact")
    return artifact_identity


def publish_run_result_to_wandb(
    *,
    output_dir: Path,
    result: dict[str, Any],
    enabled: bool,
) -> dict[str, Any] | None:
    """Attach terminal hashes and complete raw histories to the active W&B run."""
    if not enabled:
        return None
    import wandb

    if wandb.run is None:
        raise RuntimeError("cannot publish terminal evidence without an active W&B run")
    wandb_identity = result.get("wandb_identity")
    observed_wandb_identity = _observe_active_wandb_identity(
        wandb.run, wandb_identity
    )
    expected_run_id = observed_wandb_identity["run_id"]
    wandb.config.update(
        {"terminal_run_result": result},
        allow_val_change=True,
    )
    names = TERMINAL_EVIDENCE_FILE_NAMES
    for name in names:
        wandb.save(
            str(output_dir / name),
            base_path=str(output_dir),
            policy="now",
        )
    artifact_name = f"{expected_run_id}-terminal-evidence"
    artifact = wandb.Artifact(
        artifact_name,
        type="confirmatory-run-evidence",
        metadata={
            "wandb_identity": result.get("wandb_identity"),
            "registration_sha256": result.get("registration_sha256"),
            "recipe_lock_sha256": result.get("recipe_lock_sha256"),
            "terminal_checkpoint": result.get("terminal_checkpoint"),
            "final_adapter_and_tokenizer": result.get(
                "final_adapter_and_tokenizer"
            ),
            "raw_history_sha256": result.get("raw_history_sha256"),
            "evidence_eligibility": result.get("evidence_eligibility"),
            "reproduction_source": result.get("reproduction_source"),
        },
    )
    for name in names:
        artifact.add_file(str(output_dir / name), name=name)
    logged = wandb.run.log_artifact(artifact, aliases=["terminal-evidence"])
    completed = logged.wait()
    if completed is not None:
        logged = completed
    artifact_identity = {
        key: getattr(logged, key, None)
        for key in ("id", "name", "version", "digest", "qualified_name")
    }
    _validate_terminal_artifact_identity(artifact_identity, wandb_identity)
    receipt = {
        "schema_version": 2,
        "wandb_identity": wandb_identity,
        "observed_wandb_identity": observed_wandb_identity,
        "artifact": artifact_identity,
        "terminal_run_result_sha256": sha256_file(
            output_dir / "run_result_manifest.json"
        ),
        "uploaded_file_sha256": {
            name: sha256_file(output_dir / name) for name in names
        },
    }
    receipt_path = output_dir / "wandb_terminal_publish_receipt.json"
    _write_json_atomic(receipt_path, receipt)
    wandb.config.update(
        {"terminal_evidence_receipt": receipt},
        allow_val_change=True,
    )
    wandb.save(
        str(receipt_path),
        base_path=str(output_dir),
        policy="now",
    )
    return receipt


def _load_valid_terminal_publish_receipt(
    *, output_dir: Path, result: dict[str, Any]
) -> dict[str, Any] | None:
    """Return a complete local receipt, or force an infrastructure-only retry."""
    path = output_dir / "wandb_terminal_publish_receipt.json"
    if not path.is_file():
        return None
    try:
        receipt = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    artifact = receipt.get("artifact", {}) if isinstance(receipt, dict) else {}
    wandb_identity = result.get("wandb_identity")
    expected_uploads = {
        name: sha256_file(output_dir / name) for name in TERMINAL_EVIDENCE_FILE_NAMES
    }
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema_version") != 2
        or receipt.get("wandb_identity") != wandb_identity
        or receipt.get("observed_wandb_identity")
        != _expected_observed_wandb_identity(wandb_identity)
        or receipt.get("terminal_run_result_sha256")
        != sha256_file(output_dir / "run_result_manifest.json")
        or receipt.get("uploaded_file_sha256") != expected_uploads
    ):
        return None
    try:
        _validate_terminal_artifact_identity(artifact, wandb_identity)
    except RuntimeError:
        return None
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"])
    parser.add_argument(
        "--publish-existing-result",
        action="store_true",
        help="Infrastructure-only retry of terminal W&B evidence for a completed registered run.",
    )
    parser.add_argument(
        "--reproduction-replay",
        action="store_true",
        help=(
            "Run a registered config as an explicitly non-claim reproduction; "
            "requires a fresh --output-dir and --wandb-mode disabled."
        ),
    )
    parser.add_argument(
        "--replay-config-smoke-test",
        action="store_true",
        help=(
            "Validate and derive the non-claim replay identity, then exit before "
            "loading artifacts, data, models, or outcomes. Requires "
            "--reproduction-replay."
        ),
    )
    parser.add_argument(
        "--skip-jlens-metric",
        action="store_true",
        help="Do not load or compute J-lens scores (useful for short GSM8K-only smoke tests).",
    )
    return parser.parse_args()


def _confirmatory_state_root(path: Path) -> Path | None:
    return next(
        (
            candidate
            for candidate in (path, *path.parents)
            if candidate.parent.name == ".confirmatory"
            and re.fullmatch(r"v[1-9][0-9]*", candidate.name)
        ),
        None,
    )


def configure_reproduction_replay(
    cfg: dict[str, Any], *, output_dir: str | None, wandb_mode: str | None
) -> dict[str, Any]:
    """Derive a non-claim replay without reusing original output/W&B identities."""
    if cfg.get("registration_sha256") is None:
        raise ValueError("reproduction replay requires a registered config")
    if output_dir is None or wandb_mode != "disabled":
        raise ValueError(
            "reproduction replay requires --output-dir and --wandb-mode disabled"
        )
    if Path(output_dir).resolve() == Path(cfg["output_dir"]).resolve():
        raise ValueError("reproduction replay output must differ from the original run")

    registered_output = Path(cfg["output_dir"]).resolve()
    registered_state = _confirmatory_state_root(registered_output)
    if registered_state is None:
        raise ValueError(
            "registered replay config has no recognizable versioned "
            "confirmatory state root"
        )
    replay_output = Path(output_dir).resolve()
    replay_state = _confirmatory_state_root(replay_output)
    if replay_state is not None:
        raise ValueError(
            "reproduction replay output must be outside every immutable "
            f"confirmatory state, including the immutable {registered_state.name.upper()} state"
        )
    result = dict(cfg)
    original_wandb_identity = {
        key: result.get(key)
        for key in (
            "wandb_entity",
            "wandb_project",
            "wandb_group",
            "wandb_tags",
            "wandb_run_id",
            "wandb_url",
            "wandb_resume",
            "run_name",
        )
    }
    result["output_dir"] = output_dir
    result["wandb_mode"] = "disabled"
    result["run_name"] = f"nonclaim-replay-{cfg.get('run_name', 'registered-run')}"
    for key in (
        "wandb_entity",
        "wandb_group",
        "wandb_tags",
        "wandb_run_id",
        "wandb_url",
        "wandb_resume",
    ):
        result.pop(key, None)
    result["evidence_eligibility"] = "non_claim_reproduction"
    result["reproduction_source"] = {
        "registration_sha256": cfg.get("registration_sha256"),
        "recipe_lock_sha256": cfg.get("recipe_lock_sha256"),
        "recipe_sha256": cfg.get("recipe_sha256"),
        "registered_command": cfg.get("registered_command"),
        "original_output_dir": cfg.get("output_dir"),
        "original_wandb_identity": original_wandb_identity,
        "interpretation": (
            "re-execution for reproducibility only; never eligible as the original "
            "registered confirmatory attempt"
        ),
    }
    return result


def apply_training_cli_overrides(
    cfg: dict[str, Any],
    *,
    updates: int | None,
    output_dir: str | None,
    wandb_mode: str | None,
) -> dict[str, Any]:
    """Apply ordinary overrides while failing closed for registered configs."""
    if cfg.get("registration_sha256") is not None:
        if updates is not None or output_dir is not None:
            raise ValueError(
                "registered configs forbid update/output overrides; use "
                "--reproduction-replay for a non-claim fresh output"
            )
        if wandb_mode is not None and wandb_mode != cfg.get("wandb_mode"):
            raise ValueError(
                "registered configs forbid tracking-mode changes; use "
                "--reproduction-replay with disabled W&B"
            )
    result = dict(cfg)
    if updates is not None:
        result["updates"] = updates
    if output_dir is not None:
        result["output_dir"] = output_dir
    if wandb_mode is not None:
        result["wandb_mode"] = wandb_mode
    return result


def republish_existing_run_result(
    cfg: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    """Resume only the frozen W&B identity and retry its terminal evidence upload."""
    if cfg.get("registration_sha256") is None or cfg.get("wandb_mode") != "online":
        raise ValueError("terminal W&B retry is only valid for a registered online run")
    result_path = output_dir / "run_result_manifest.json"
    resolved_path = output_dir / "resolved_config.json"
    if not result_path.is_file() or not resolved_path.is_file():
        raise FileNotFoundError("completed registered run lacks its terminal result/config")
    if json.loads(resolved_path.read_text()) != cfg:
        raise ValueError("existing run config differs from the frozen retry config")
    result = json.loads(result_path.read_text())
    if result.get("wandb_identity", {}).get("run_id") != cfg.get("wandb_run_id"):
        raise ValueError("existing terminal result has a different W&B identity")
    receipt = _load_valid_terminal_publish_receipt(
        output_dir=output_dir, result=result
    )
    if receipt is not None:
        return receipt

    import wandb

    os.environ["WANDB_RESUME"] = "must"
    run = wandb.init(
        entity=cfg["wandb_entity"],
        project=cfg["wandb_project"],
        id=cfg["wandb_run_id"],
        name=cfg["run_name"],
        group=cfg["wandb_group"],
        tags=cfg["wandb_tags"],
        resume="must",
    )
    if run is None:
        raise RuntimeError("W&B did not resume the frozen run for evidence retry")
    try:
        receipt = publish_run_result_to_wandb(
            output_dir=output_dir,
            result=result,
            enabled=True,
        )
        if receipt is None:
            raise RuntimeError("terminal W&B evidence retry produced no receipt")
        return receipt
    finally:
        wandb.finish()


def main() -> None:
    args = parse_args()
    if args.replay_config_smoke_test and not args.reproduction_replay:
        raise ValueError("--replay-config-smoke-test requires --reproduction-replay")
    cfg = load_config(args.config)
    if args.reproduction_replay:
        if args.updates is not None or args.publish_existing_result:
            raise ValueError(
                "reproduction replay forbids update changes and terminal publication"
            )
        cfg = configure_reproduction_replay(
            cfg, output_dir=args.output_dir, wandb_mode=args.wandb_mode
        )
    else:
        cfg = apply_training_cli_overrides(
            cfg,
            updates=args.updates,
            output_dir=args.output_dir,
            wandb_mode=args.wandb_mode,
        )
    if args.replay_config_smoke_test:
        print(
            json.dumps(
                {
                    "status": "valid_non_claim_replay_config",
                    "output_dir": cfg["output_dir"],
                    "wandb_mode": cfg["wandb_mode"],
                    "evidence_eligibility": cfg["evidence_eligibility"],
                    "reproduction_source": cfg["reproduction_source"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if cfg["reward_type"] not in {"gsm8k", "jlens"}:
        raise ValueError("reward_type must be gsm8k or jlens")
    cfg.setdefault("model_revision", QWEN_MODEL_REVISION)
    cfg.setdefault("dataset_revision", GSM8K_REVISION)
    expected_lens_sha256 = cfg.get(
        "expected_lens_sha256", cfg.get("lens_sha256")
    )
    expected_calibration_sha256 = cfg.get(
        "expected_calibration_sha256", cfg.get("calibration_sha256")
    )
    artifact_hashes: dict[str, str] = {}
    if cfg["reward_type"] == "jlens":
        for name, key, expected in (
            ("lens", "lens_path", expected_lens_sha256),
            ("calibration", "calibration_path", expected_calibration_sha256),
        ):
            actual = sha256_file(cfg[key])
            artifact_hashes[name] = actual
            if expected is not None and actual != expected:
                raise ValueError(
                    f"{name} artifact SHA-256 does not match the frozen config: "
                    f"{actual} != {expected}"
                )
    metric_schema: dict[str, Any] | None = None
    metric_schema_path = cfg.get("metric_schema_path")
    if metric_schema_path is not None:
        expected_metric_schema_sha256 = cfg.get("metric_schema_sha256")
        actual_metric_schema_sha256 = sha256_file(metric_schema_path)
        if (
            not isinstance(expected_metric_schema_sha256, str)
            or actual_metric_schema_sha256 != expected_metric_schema_sha256
        ):
            raise ValueError(
                "metric schema does not match the frozen config: "
                f"{actual_metric_schema_sha256} != {expected_metric_schema_sha256}"
            )
        metric_schema = json.loads(Path(metric_schema_path).read_text())
    seed_everything(cfg["seed"])
    # Confirmatory configs pin the external tracking identity. Ordinary
    # configs omit those optional fields and retain the prior W&B behavior.
    wandb_identity = configure_wandb_environment(cfg)

    repo_root = resolve_repository_root(__file__)
    source_provenance = repository_provenance(repo_root)
    if cfg.get("require_clean_repository", False):
        require_clean_repository_provenance(source_provenance)

    if args.publish_existing_result:
        if args.updates is not None or args.output_dir is not None:
            raise ValueError("terminal W&B retry forbids training/output overrides")
        receipt = republish_existing_run_result(cfg, Path(cfg["output_dir"]))
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return

    output_dir = create_run_directory(cfg["output_dir"])
    (output_dir / "resolved_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    environment_snapshot = runtime_environment_snapshot()
    environment_path = output_dir / "environment_snapshot.json"
    environment_path.write_text(
        json.dumps(environment_snapshot, indent=2, sort_keys=True) + "\n"
    )

    run_manifest: dict[str, Any] = {
        **source_provenance,
        "config_path": str(Path(args.config).resolve()),
        "config_sha256": sha256_file(args.config),
        "resolved_config_sha256": sha256_file(output_dir / "resolved_config.json"),
        "model_name": cfg["model_name"],
        "model_revision": cfg["model_revision"],
        "dataset": "openai/gsm8k:main",
        "dataset_revision": cfg["dataset_revision"],
        "reward_type": cfg["reward_type"],
        "process_command": {
            "python_executable": sys.executable,
            "argv": list(sys.argv),
            "cwd": str(Path.cwd().resolve()),
        },
        "registered_command": cfg.get("registered_command"),
        "evidence_eligibility": cfg.get("evidence_eligibility"),
        "reproduction_source": cfg.get("reproduction_source"),
        "confirmatory_identity": {
            key: cfg.get(key)
            for key in (
                "registration_sha256",
                "recipe_lock_sha256",
                "recipe_sha256",
                "curve_manifest_sha256",
                "train_exclusions_manifest_sha256",
                "registered_code_sha256",
            )
            if key in cfg
        },
        "metric_schema": (
            {
                "path": str(Path(metric_schema_path).resolve()),
                "sha256": cfg["metric_schema_sha256"],
                "content": metric_schema,
            }
            if metric_schema is not None
            else None
        ),
        "runtime": {
            "cuda_device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "cuda_version": torch.version.cuda,
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "environment_snapshot_path": "environment_snapshot.json",
            "environment_snapshot_sha256": sha256_file(environment_path),
            "environment_snapshot": environment_snapshot,
        },
    }
    if wandb_identity is not None:
        run_manifest["wandb_identity"] = wandb_identity
    if cfg["reward_type"] == "jlens":
        run_manifest.update({
            "lens_path": str(Path(cfg["lens_path"]).resolve()),
            "lens_sha256": artifact_hashes["lens"],
            "calibration_path": str(Path(cfg["calibration_path"]).resolve()),
            "calibration_sha256": artifact_hashes["calibration"],
        })

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name"], revision=cfg["model_revision"]
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"

    raw = load_dataset(
        "openai/gsm8k", "main", revision=cfg["dataset_revision"]
    )
    raw_train = raw["train"].add_column("_source_index", range(len(raw["train"])))
    validation_source = cfg.get("validation_source", "test")
    if validation_source == "train":
        validation_manifest = cfg.get("validation_indices_path")
        if validation_manifest:
            validation_indices = load_index_manifest(validation_manifest)
            if not validation_indices or max(validation_indices) >= len(raw_train):
                raise ValueError("training-split validation manifest is empty or out of range")
            if len(validation_indices) != int(cfg["validation_examples"]):
                raise ValueError("validation manifest size does not match validation_examples")
            raw_eval_dataset = raw_train.select(validation_indices)
        else:
            validation_offset = int(cfg["validation_offset"])
            validation_end = validation_offset + int(cfg["validation_examples"])
            if validation_offset < 0 or validation_end > len(raw_train):
                raise ValueError("training-split validation slice is out of range")
            validation_indices = list(range(validation_offset, validation_end))
            raw_eval_dataset = raw_train.select(validation_indices)
        excluded_indices = set(validation_indices)
        reserved_manifest = cfg.get("reserved_train_indices_path")
        if reserved_manifest:
            reserved_indices = load_index_manifest(reserved_manifest)
            if reserved_indices and max(reserved_indices) >= len(raw_train):
                raise ValueError("reserved training manifest is out of range")
            excluded_indices.update(reserved_indices)
        excluded_ranges: list[tuple[int, int]] = []
        excluded_ranges.extend(
            (int(start), int(end))
            for start, end in cfg.get("reserved_train_ranges", [])
        )
        train_pool = raw_train.filter(
            lambda row: (
                row["_source_index"] not in excluded_indices
                and not any(
                    start <= row["_source_index"] < end
                    for start, end in excluded_ranges
                )
            )
        )
    elif validation_source == "test":
        validation_manifest = cfg.get("validation_indices_path")
        validation_indices = (
            load_index_manifest(validation_manifest)
            if validation_manifest
            else list(range(cfg["validation_examples"]))
        )
        if not validation_indices or max(validation_indices) >= len(raw["test"]):
            raise ValueError("test-split validation manifest is empty or out of range")
        if len(validation_indices) != int(cfg["validation_examples"]):
            raise ValueError("validation manifest size does not match validation_examples")
        raw_eval_dataset = raw["test"].select(validation_indices)
        train_pool = raw_train
    else:
        raise ValueError("validation_source must be train or test")
    train_dataset = train_pool.shuffle(seed=cfg["seed"]).select(
        range(cfg["train_examples"])
    )
    selected_train_indices = [int(value) for value in train_dataset["_source_index"]]
    selected_validation_indices = (
        [int(value) for value in raw_eval_dataset["_source_index"]]
        if "_source_index" in raw_eval_dataset.column_names
        else [int(value) for value in validation_indices]
    )
    if validation_source == "train" and set(selected_train_indices) & set(selected_validation_indices):
        raise AssertionError("training and validation indices overlap")
    (output_dir / "data_indices.json").write_text(json.dumps({
        "train_source_indices": selected_train_indices,
        "validation_source": validation_source,
        "validation_source_indices": selected_validation_indices,
    }, indent=2) + "\n")
    run_manifest["data_indices_sha256"] = sha256_file(output_dir / "data_indices.json")
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n"
    )
    eval_dataset = raw_eval_dataset
    train_preparer = prepare_prompt if cfg["reward_type"] == "jlens" else prepare_example
    train_dataset = train_dataset.map(
        train_preparer, remove_columns=train_dataset.column_names
    )
    eval_dataset = eval_dataset.map(
        prepare_example, remove_columns=eval_dataset.column_names
    )

    if cfg["reward_type"] == "gsm8k":
        reward_funcs = [gsm8k_reward_trl]
        reward_weights = [1.0]
    elif args.skip_jlens_metric:
        raise ValueError("--skip-jlens-metric is only valid with the GSM8K reward")
    else:
        jreward = TRLTargetJLReward(
            TargetJLReward(
                cfg["lens_path"], cfg["calibration_path"], tokenizer,
                cfg["target_words"], cfg["score_stride"], cfg["mask_target_tokens"],
                cfg.get("vocab_chunk_size", 16384),
                cfg.get("score_start_fraction", 0.0), cfg.get("score_layers"),
                cfg.get("score_aggregation", "mean"), cfg.get("score_include_final", False),
                cfg.get("score_components"),
                cfg.get("score_end_fraction", 1.0),
                expected_model=cfg["model_name"],
                expected_model_revision=cfg.get("model_revision"),
                expected_lens_sha256=expected_lens_sha256,
            )
        )
        reward_funcs = [jreward]
        reward_weights = [1.0]

    training_dtype = model_dtype()
    training_args = GRPOConfig(
        output_dir=str(output_dir),
        run_name=cfg.get("run_name", f"gsm8k-{cfg['reward_type']}-reward"),
        max_steps=cfg["updates"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg.get("lr_scheduler_type", "linear"),
        warmup_steps=int(cfg.get("warmup_steps", 0)),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.0)),
        beta=cfg["kl_beta"],
        per_device_train_batch_size=cfg["num_generations"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_generations=cfg["num_generations"],
        generation_batch_size=cfg["num_generations"],
        max_completion_length=cfg["max_new_tokens"],
        temperature=cfg["temperature"],
        reward_weights=reward_weights,
        loss_type=cfg["loss_type"],
        scale_rewards=cfg["scale_rewards"],
        eval_strategy=cfg["eval_strategy"],
        eval_steps=cfg["eval_every"],
        per_device_eval_batch_size=cfg["num_generations"],
        num_generations_eval=cfg["num_generations_eval"],
        logging_steps=1,
        save_steps=cfg["save_every"],
        save_total_limit=cfg.get("save_total_limit", 3),
        report_to=["wandb"] if cfg["wandb_mode"] != "disabled" else ["none"],
        bf16=training_dtype == torch.bfloat16,
        fp16=training_dtype == torch.float16,
        gradient_checkpointing=True,
        use_vllm=False,
        seed=cfg["seed"],
        generation_kwargs=(
            {"min_new_tokens": int(cfg["min_new_tokens"])}
            if cfg.get("min_new_tokens") is not None
            else None
        ),
        model_init_kwargs={
            "revision": cfg["model_revision"],
            "dtype": training_dtype,
        },
    )
    peft_config = LoraConfig(
        r=cfg["lora_rank"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    trainer = GRPOTrainer(
        model=cfg["model_name"],
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    validation_callback = DeterministicValidationCallback(
        tokenizer, raw_eval_dataset, cfg
    )
    validation_callback.trainer = trainer
    trainer.add_callback(validation_callback)
    trainer.add_callback(RunManifestCallback(
        run_manifest,
        output_dir,
        cfg["wandb_mode"] != "disabled",
    ))
    initial_metrics = validation_callback.evaluate_and_log(trainer.model, 0)
    validation_callback.best_exact_match = initial_metrics["exact_match"]
    trainer.train()
    trainer.save_model(output_dir / "final")
    tokenizer.save_pretrained(output_dir / "final")
    (output_dir / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=2) + "\n")
    if cfg.get("registration_sha256") is not None:
        terminal_result = write_run_result_manifest(
            output_dir=output_dir,
            cfg=cfg,
            run_manifest=run_manifest,
        )
        publish_run_result_to_wandb(
            output_dir=output_dir,
            result=terminal_result,
            enabled=cfg["wandb_mode"] != "disabled",
        )


if __name__ == "__main__":
    main()
