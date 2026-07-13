from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# The vendored repository directory and its Python package are both named `trl`.
# Put the repository root on sys.path so it wins over the outer namespace directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "trl"))

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from .common import SYSTEM_PROMPT, extract_answer, load_config, seed_everything
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


class DeterministicValidationCallback(TrainerCallback):
    """Log fixed, greedy GSM8K accuracy without feeding it into training."""

    def __init__(self, tokenizer: Any, rows: Any, cfg: dict[str, Any]) -> None:
        self.tokenizer = tokenizer
        self.rows = rows
        self.cfg = cfg
        self.trainer: Any = None

    def evaluate_and_log(self, model: Any) -> None:
        metrics = evaluate(
            model, self.tokenizer, self.rows, self.cfg, None,
            self.cfg.get("validation_batch_size", 16),
        )
        self.trainer.log({f"validation/{key}": value for key, value in metrics.items()})

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.cfg["eval_every"] == 0:
            self.evaluate_and_log(model)
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
    seed_everything(cfg["seed"])
    os.environ["WANDB_PROJECT"] = cfg["wandb_project"]
    os.environ["WANDB_MODE"] = cfg["wandb_mode"]

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"

    raw = load_dataset("openai/gsm8k", "main")
    train_dataset = raw["train"].shuffle(seed=cfg["seed"]).select(range(cfg["train_examples"]))
    raw_eval_dataset = raw["test"].select(range(cfg["validation_examples"]))
    eval_dataset = raw_eval_dataset
    train_dataset = train_dataset.map(prepare_example, remove_columns=["question"])
    eval_dataset = eval_dataset.map(prepare_example, remove_columns=["question"])

    reward_funcs = [gsm8k_reward_trl]
    reward_weights = [1.0]
    if not args.skip_jlens_metric:
        jreward = TRLTargetJLReward(
            TargetJLReward(
                cfg["lens_path"], cfg["calibration_path"], tokenizer,
                cfg["target_words"], cfg["score_stride"], cfg["mask_target_tokens"],
            )
        )
        reward_funcs.append(jreward)
        reward_weights = [1.0, 0.0] if cfg["reward_type"] == "gsm8k" else [0.0, 1.0]
    elif cfg["reward_type"] != "gsm8k":
        raise ValueError("--skip-jlens-metric is only valid with the GSM8K reward")

    training_args = GRPOConfig(
        output_dir=str(output_dir),
        run_name=f"gsm8k-{cfg['reward_type']}-reward",
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
        save_total_limit=3,
        report_to=["wandb"] if cfg["wandb_mode"] != "disabled" else ["none"],
        bf16=True,
        gradient_checkpointing=True,
        use_vllm=False,
        seed=cfg["seed"],
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
    validation_callback.evaluate_and_log(trainer.model)
    trainer.train()
    trainer.save_model(output_dir / "final")
    tokenizer.save_pretrained(output_dir / "final")
    (output_dir / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=2) + "\n")


if __name__ == "__main__":
    main()
