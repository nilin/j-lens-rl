from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import binomial_ci95, format_prompt, gsm8k_reward, load_config, model_dtype
from .reward import TargetJLReward


@torch.no_grad()
def evaluate(model: Any, tokenizer: Any, rows: Any, cfg: dict[str, Any],
             jreward: TargetJLReward | None, batch_size: int = 16) -> dict[str, float]:
    model.eval()
    correct: list[float] = []
    jscores: list[float] = []
    lengths: list[int] = []
    literal_targets = 0
    for start in range(0, len(rows), batch_size):
        batch = [rows[index] for index in range(start, min(start + batch_size, len(rows)))]
        prompts = [format_prompt(tokenizer, row["question"]) for row in batch]
        encoded = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=cfg["max_prompt_tokens"],
        ).to(model.device)
        prompt_width = encoded.input_ids.shape[1]
        seq = model.generate(
            **encoded, max_new_tokens=cfg["max_new_tokens"], do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        for index, row in enumerate(batch):
            completion_ids = seq[index, prompt_width:]
            eos = (completion_ids == tokenizer.eos_token_id).nonzero()
            if eos.numel():
                completion_ids = completion_ids[: int(eos[0].item()) + 1]
            text = tokenizer.decode(completion_ids, skip_special_tokens=True)
            literal_targets += int(
                any(word.lower() in text.lower() for word in cfg["target_words"])
            )
            correct.append(gsm8k_reward(text, row["answer"]))
            lengths.append(int(completion_ids.numel()))
            if jreward is not None:
                prompt_ids = encoded.input_ids[index][encoded.attention_mask[index].bool()]
                unpadded = torch.cat([prompt_ids, completion_ids]).unsqueeze(0)
                mask = torch.ones_like(unpadded)
                out = model(
                    unpadded, attention_mask=mask, output_hidden_states=True,
                    use_cache=False,
                )
                jscores.append(
                    jreward(
                        model, out.hidden_states, len(prompt_ids), mask[0],
                        input_ids=unpadded[0],
                    )
                )
    n = len(correct)
    p = float(np.mean(correct))
    ci_low, ci_high = binomial_ci95(round(sum(correct)), n)
    result = {
        "exact_match": p,
        "exact_match_ci95_low": ci_low,
        "exact_match_ci95_high": ci_high,
        "mean_length": float(np.mean(lengths)),
        "literal_target_completion_rate": literal_targets / n,
    }
    if jscores:
        result["jlens_reward"] = float(np.mean(jscores))
    model.train()
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--adapter")
    p.add_argument("--skip-jlens-metric", action="store_true")
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()
    cfg = load_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], torch_dtype=model_dtype(), device_map={"": "cuda:0"}
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    reward = None if args.skip_jlens_metric else TargetJLReward(
        cfg["lens_path"], cfg["calibration_path"], tokenizer,
        cfg["target_words"], cfg["score_stride"], cfg["mask_target_tokens"],
        cfg.get("vocab_chunk_size", 16384),
        cfg.get("score_start_fraction", 0.0), cfg.get("score_layers"),
    )
    ds = load_dataset("openai/gsm8k", "main", split="test")
    rows = ds.select(range(min(cfg["validation_examples"], len(ds))))
    print(json.dumps(evaluate(model, tokenizer, rows, cfg, reward, args.batch_size), indent=2))


if __name__ == "__main__":
    main()
