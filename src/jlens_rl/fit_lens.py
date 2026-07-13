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
    p.add_argument(
        "--corpus", choices=("wikitext", "gsm8k_rollouts"), default="wikitext",
        help="Fit on generic text or ungraded base-model GSM8K completions.",
    )
    p.add_argument(
        "--rollout-offset", type=int, default=1000,
        help="Start after the shuffled RL training subset for disjoint rollout prompts.",
    )
    p.add_argument("--rollout-batch-size", type=int, default=8)
    return p.parse_args()


@torch.no_grad()
def ungraded_gsm8k_rollouts(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    total: int,
    seed: int,
    offset: int,
    batch_size: int,
) -> list[str]:
    """Sample response-only fitting text without ever reading answers or grades."""
    dataset = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=seed)
    rows = dataset.select(range(offset, offset + total))
    tokenizer.padding_side = "left"
    texts: list[str] = []
    for start in range(0, total, batch_size):
        questions = rows[start : min(start + batch_size, total)]["question"]
        prompts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for question in questions
        ]
        encoded = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=384,
        ).to(model.device)
        prompt_width = encoded.input_ids.shape[1]
        generated = model.generate(
            **encoded, max_new_tokens=128, do_sample=True, temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
        for sequence in generated[:, prompt_width:]:
            texts.append(tokenizer.decode(sequence, skip_special_tokens=True))
    return texts


def main() -> None:
    args = parse_args()
    target_words = args.target_word or ["solved"]
    seed_everything(args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=model_dtype(), device_map="cuda"
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    wrapped = jlens.from_hf(model, tokenizer, force_bos=False)
    layers = [int(x) for x in args.layers.split(",")]
    total = args.num_prompts + args.calibration_prompts
    if args.corpus == "wikitext":
        corpus = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
        prompts = [x["text"] for x in corpus if len(x["text"].split()) >= 80][:total]
    else:
        prompts = ungraded_gsm8k_rollouts(
            model, tokenizer, total, args.seed, args.rollout_offset,
            args.rollout_batch_size,
        )
        corpus_path = Path(args.output + ".ungraded_rollouts.json")
        corpus_path.write_text(json.dumps({
            "source": "base_model_ungraded_gsm8k_train_rollouts",
            "seed": args.seed,
            "offset": args.rollout_offset,
            "count": len(prompts),
            "texts": prompts,
        }, indent=2) + "\n")
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
             "adapter": args.adapter, "corpus": args.corpus,
             "rollout_offset": args.rollout_offset if args.corpus == "gsm8k_rollouts" else None}
    Path(args.calibration_output).write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
