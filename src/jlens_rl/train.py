from __future__ import annotations

import argparse
import json
import os
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
    repository_provenance,
    seed_everything,
    sha256_file,
)
from .eval import evaluate
from .reward import TRLTargetJLReward, TargetJLReward


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
            wandb.config.update({"experiment_manifest": self.manifest}, allow_val_change=True)
            wandb.save(
                str(self.output_dir / "run_manifest.json"),
                base_path=str(self.output_dir),
                policy="now",
            )
            wandb.save(
                str(self.output_dir / "data_indices.json"),
                base_path=str(self.output_dir),
                policy="now",
            )
        return control


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"])
    parser.add_argument(
        "--skip-jlens-metric",
        action="store_true",
        help="Do not load or compute J-lens scores (useful for short GSM8K-only smoke tests).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.updates is not None:
        cfg["updates"] = args.updates
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    if args.wandb_mode is not None:
        cfg["wandb_mode"] = args.wandb_mode
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
    seed_everything(cfg["seed"])
    os.environ["WANDB_PROJECT"] = cfg["wandb_project"]
    os.environ["WANDB_MODE"] = cfg["wandb_mode"]

    output_dir = create_run_directory(cfg["output_dir"])
    (output_dir / "resolved_config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    repo_root = Path(__file__).resolve().parents[2]
    run_manifest: dict[str, Any] = {
        **repository_provenance(repo_root),
        "config_path": str(Path(args.config).resolve()),
        "config_sha256": sha256_file(args.config),
        "resolved_config_sha256": sha256_file(output_dir / "resolved_config.json"),
        "model_name": cfg["model_name"],
        "model_revision": cfg["model_revision"],
        "dataset": "openai/gsm8k:main",
        "dataset_revision": cfg["dataset_revision"],
        "reward_type": cfg["reward_type"],
    }
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


if __name__ == "__main__":
    main()
