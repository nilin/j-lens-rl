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

from .common import format_prompt, gsm8k_reward, load_config, model_dtype
from .reward import TargetJLReward


@torch.no_grad()
def evaluate(model: Any, tokenizer: Any, rows: Any, cfg: dict[str, Any],
             jreward: TargetJLReward | None) -> dict[str, float]:
    model.eval()
    correct: list[float] = []
    jscores: list[float] = []
    lengths: list[int] = []
    for row in rows:
        prompt = format_prompt(tokenizer, row["question"])
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True,
                            max_length=cfg["max_prompt_tokens"]).to(model.device)
        prompt_len = encoded.input_ids.shape[1]
        seq = model.generate(
            **encoded, max_new_tokens=cfg["max_new_tokens"], do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        completion_ids = seq[0, prompt_len:]
        text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        correct.append(gsm8k_reward(text, row["answer"]))
        lengths.append(int(completion_ids.numel()))
        if jreward is not None:
            mask = torch.ones_like(seq)
            out = model(seq, attention_mask=mask, output_hidden_states=True, use_cache=False)
            jscores.append(jreward(model, out.hidden_states, prompt_len, mask[0]))
    n = len(correct)
    p = float(np.mean(correct))
    result = {
        "exact_match": p,
        "exact_match_ci95": 1.96 * float(np.sqrt(max(p * (1 - p), 1e-12) / n)),
        "mean_length": float(np.mean(lengths)),
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
    args = p.parse_args()
    cfg = load_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], torch_dtype=model_dtype(), device_map={"": "cuda:0"}
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    reward = None if args.skip_jlens_metric else TargetJLReward(
        cfg["lens_path"], cfg["calibration_path"], tokenizer,
        cfg["target_words"], cfg["score_stride"],
    )
    ds = load_dataset("openai/gsm8k", "main", split="test")
    rows = ds.select(range(min(cfg["validation_examples"], len(ds))))
    print(json.dumps(evaluate(model, tokenizer, rows, cfg, reward), indent=2))


if __name__ == "__main__":
    main()
