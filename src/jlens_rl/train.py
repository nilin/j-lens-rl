from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import append_jsonl, format_prompt, gsm8k_reward, load_config, model_dtype, seed_everything
from .eval import evaluate
from .reward import TargetJLReward


def completion_mask(sequences: torch.Tensor, prompt_len: int, eos_id: int | None) -> torch.Tensor:
    mask = torch.ones_like(sequences[:, prompt_len:], dtype=torch.float32)
    if eos_id is not None:
        eos = sequences[:, prompt_len:].eq(eos_id)
        after = eos.cumsum(dim=1) - eos.long()
        mask *= after.eq(0)
    return mask


def completion_logps(logits: torch.Tensor, sequences: torch.Tensor,
                     prompt_len: int) -> torch.Tensor:
    # Token at absolute position p is predicted by logits at p-1.
    prediction = logits[:, prompt_len - 1 : -1].float()
    targets = sequences[:, prompt_len:]
    return F.log_softmax(prediction, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--updates", type=int)
    p.add_argument("--output-dir")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.updates is not None:
        cfg["updates"] = args.updates
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if cfg["reward_type"] not in {"gsm8k", "jlens"}:
        raise ValueError("reward_type must be gsm8k or jlens")
    if not torch.cuda.is_available():
        raise RuntimeError("training requires a CUDA GPU")
    seed_everything(cfg["seed"])
    outdir = Path(cfg["output_dir"])
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "resolved_config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], torch_dtype=model_dtype(), device_map={"": "cuda:0"}
    )
    base.config.use_cache = False
    lora = LoraConfig(
        r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"], lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora)
    model.train()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=cfg["learning_rate"]
    )
    jreward = TargetJLReward(
        cfg["lens_path"], cfg["calibration_path"], tokenizer,
        cfg["target_words"], cfg["score_stride"],
    )

    train_ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=cfg["seed"])
    train_ds = train_ds.select(range(min(cfg["train_examples"], len(train_ds))))
    val_ds = load_dataset("openai/gsm8k", "main", split="test")
    val_ds = val_ds.select(range(min(cfg["validation_examples"], len(val_ds))))
    metrics_path = outdir / "metrics.jsonl"
    optimizer.zero_grad(set_to_none=True)

    micro = 0
    for update in range(1, cfg["updates"] + 1):
        update_rewards, update_kls, update_losses = [], [], []
        for _ in range(cfg["gradient_accumulation_steps"]):
            row = train_ds[micro % len(train_ds)]
            micro += 1
            prompt = format_prompt(tokenizer, row["question"])
            encoded = tokenizer(
                [prompt] * cfg["num_generations"], return_tensors="pt", padding=True,
                truncation=True, max_length=cfg["max_prompt_tokens"],
            ).to(model.device)
            prompt_len = encoded.input_ids.shape[1]
            model.eval()
            with torch.no_grad():
                sequences = model.generate(
                    **encoded, max_new_tokens=cfg["max_new_tokens"], do_sample=True,
                    temperature=cfg["temperature"],
                    pad_token_id=tokenizer.pad_token_id,
                )
            model.train()
            cmask = completion_mask(sequences, prompt_len, tokenizer.eos_token_id).to(model.device)
            attention = torch.cat(
                [torch.ones_like(sequences[:, :prompt_len]), cmask.long()], dim=1
            )
            policy = model(sequences, attention_mask=attention, output_hidden_states=True, use_cache=False)
            logps = completion_logps(policy.logits, sequences, prompt_len)
            with torch.no_grad(), model.disable_adapter():
                reference = model(sequences, attention_mask=attention, use_cache=False)
                ref_logps = completion_logps(reference.logits, sequences, prompt_len)

            texts = tokenizer.batch_decode(sequences[:, prompt_len:], skip_special_tokens=True)
            if cfg["reward_type"] == "gsm8k":
                rewards = [gsm8k_reward(text, row["answer"]) for text in texts]
            else:
                rewards = []
                for i in range(sequences.shape[0]):
                    hs = tuple(h[i : i + 1] for h in policy.hidden_states)
                    rewards.append(jreward(model, hs, prompt_len, attention[i]))
            r = torch.tensor(rewards, device=model.device, dtype=torch.float32)
            advantage = (r - r.mean()) / (r.std(unbiased=False) + 1e-4)
            denom = cmask.sum(dim=1).clamp_min(1)
            pg = -(advantage * (logps * cmask).sum(dim=1) / denom).mean()
            log_ratio = ref_logps - logps
            kl_tokens = torch.exp(log_ratio) - log_ratio - 1
            kl = (kl_tokens * cmask).sum() / cmask.sum().clamp_min(1)
            loss = (pg + cfg["kl_beta"] * kl) / cfg["gradient_accumulation_steps"]
            loss.backward()
            update_rewards.extend(rewards)
            update_kls.append(float(kl.detach()))
            update_losses.append(float(loss.detach() * cfg["gradient_accumulation_steps"]))
            del policy, reference

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        row_metrics = {
            "step": update, "split": "train", "reward_type": cfg["reward_type"],
            "reward": float(np.mean(update_rewards)), "kl": float(np.mean(update_kls)),
            "loss": float(np.mean(update_losses)),
        }
        append_jsonl(metrics_path, row_metrics)
        print(json.dumps(row_metrics), flush=True)

        if update == 1 or update % cfg["eval_every"] == 0 or update == cfg["updates"]:
            result = evaluate(model, tokenizer, val_ds, cfg, jreward)
            result.update({"step": update, "split": "validation", "reward_type": cfg["reward_type"]})
            append_jsonl(metrics_path, result)
            print(json.dumps(result), flush=True)
            model.save_pretrained(outdir / f"checkpoint-{update}")

    model.save_pretrained(outdir / "final")
    tokenizer.save_pretrained(outdir / "final")


if __name__ == "__main__":
    main()
