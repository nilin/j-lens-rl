from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import jlens
from .common import SYSTEM_PROMPT, model_dtype, seed_everything
from .reward import single_token_ids, target_log_probs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--adapter", help="Optional trained LoRA adapter to merge before refitting")
    p.add_argument("--output", default="artifacts/qwen25_05b_solved_lens.pt")
    p.add_argument("--calibration-output", default="artifacts/qwen25_05b_solved_calibration.json")
    p.add_argument("--target-word", action="append")
    p.add_argument("--num-prompts", type=int, default=100)
    p.add_argument("--calibration-prompts", type=int, default=50)
    p.add_argument("--layers", default="8,14,20")
    p.add_argument("--dim-batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--corpus", choices=("wikitext", "gsm8k"), default="wikitext")
    return p.parse_args()


def lens_corpus(name: str, tokenizer: AutoTokenizer, total: int) -> list[str]:
    if name == "wikitext":
        corpus = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
        return [x["text"] for x in corpus if len(x["text"].split()) >= 80][:total]
    corpus = load_dataset("openai/gsm8k", "main", split="train")
    texts = []
    for row in corpus.select(range(total)):
        texts.append(tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["answer"]},
            ],
            tokenize=False,
        ))
    return texts


def main() -> None:
    args = parse_args()
    target_words = args.target_word or ["solved"]
    seed_everything(args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=model_dtype(), device_map="cuda"
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    wrapped = jlens.from_hf(model, tokenizer, force_bos=False)
    layers = [int(x) for x in args.layers.split(",")]
    prompts = lens_corpus(
        args.corpus, tokenizer, args.num_prompts + args.calibration_prompts
    )
    lens = jlens.fit(
        wrapped, prompts[: args.num_prompts], source_layers=layers,
        dim_batch=args.dim_batch, max_seq_len=128,
        checkpoint_path=args.output + ".checkpoint",
    )
    lens.save(args.output)

    # Calibrate raw target scores on held-out generic text.
    ids = single_token_ids(tokenizer, target_words)
    raw: list[float] = []
    norm, head = wrapped._final_norm, wrapped._lm_head
    for text in prompts[args.num_prompts :]:
        input_ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).input_ids.cuda()
        out = model(input_ids, output_hidden_states=True, use_cache=False)
        pos = list(range(19, input_ids.shape[1], 20)) or [input_ids.shape[1] - 1]
        for layer in lens.source_layers:
            h = lens.transport(out.hidden_states[layer + 1][0, pos].float(), layer)
            normalized = norm(h.to(norm.weight.dtype))
            raw.extend(target_log_probs(normalized, head, ids).cpu().tolist())
    stats = {"mean": float(np.mean(raw)), "std": float(np.std(raw)), "token_ids": ids,
             "target_words": target_words, "layers": layers, "model": args.model,
             "adapter": args.adapter, "corpus": args.corpus}
    Path(args.calibration_output).write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
