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
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from .common import SYSTEM_PROMPT, extract_answer, load_config, seed_everything
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"])
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
    eval_dataset = raw["test"].select(range(cfg["validation_examples"]))
    train_dataset = train_dataset.map(prepare_example, remove_columns=["question"])
    eval_dataset = eval_dataset.map(prepare_example, remove_columns=["question"])

    jreward = TRLTargetJLReward(
        TargetJLReward(
            cfg["lens_path"], cfg["calibration_path"], tokenizer,
            cfg["target_words"], cfg["score_stride"],
        )
    )
    reward_funcs = [gsm8k_reward_trl, jreward]
    reward_weights = [1.0, 0.0] if cfg["reward_type"] == "gsm8k" else [0.0, 1.0]

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
        eval_strategy="steps",
        eval_steps=cfg["eval_every"],
        per_device_eval_batch_size=cfg["num_generations"],
        num_generations_eval=cfg["num_generations"],
        logging_steps=1,
        save_steps=cfg["eval_every"],
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
    trainer.train()
    trainer.save_model(output_dir / "final")
    tokenizer.save_pretrained(output_dir / "final")
    (output_dir / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=2) + "\n")


if __name__ == "__main__":
    main()
