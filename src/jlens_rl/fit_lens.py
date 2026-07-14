from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import jlens
from jlens import JacobianLens
from .common import (
    GSM8K_REVISION,
    QWEN_MODEL_REVISION,
    SYSTEM_PROMPT,
    WIKITEXT_REVISION,
    model_dtype,
    seed_everything,
    sha256_file,
)
from .reward import single_token_ids, target_log_probs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--model-revision", default=QWEN_MODEL_REVISION)
    p.add_argument("--wikitext-revision", default=WIKITEXT_REVISION)
    p.add_argument("--gsm8k-revision", default=GSM8K_REVISION)
    p.add_argument("--adapter", help="Optional trained LoRA adapter to merge before refitting")
    p.add_argument(
        "--lens-input",
        help="Reuse an existing target-independent transport and only recalibrate words.",
    )
    p.add_argument("--output", default="artifacts/qwen25_05b_solved_lens.pt")
    p.add_argument("--calibration-output", default="artifacts/qwen25_05b_solved_calibration.json")
    p.add_argument("--target-word", action="append")
    p.add_argument("--num-prompts", type=int, default=100)
    p.add_argument("--calibration-prompts", type=int, default=50)
    p.add_argument("--layers", default="8,14,20")
    p.add_argument("--dim-batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--checkpoint-path",
        help=(
            "Optional explicit jlens.fit checkpoint. No implicit output-based "
            "checkpoint is reused."
        ),
    )
    p.add_argument(
        "--resume-checkpoint",
        action="store_true",
        help="Resume only when the checkpoint's generated identity manifest matches.",
    )
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
    dataset_revision: str,
) -> list[str]:
    """Sample response-only fitting text without ever reading answers or grades."""
    dataset = load_dataset(
        "openai/gsm8k", "main", split="train", revision=dataset_revision
    ).shuffle(seed=seed)
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


def write_calibration(path: str | Path, stats: dict[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(stats, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    target_words = args.target_word or ["solved"]
    seed_everything(args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.model_revision,
        dtype=model_dtype(), device_map="cuda"
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    wrapped = jlens.from_hf(model, tokenizer, force_bos=False)
    layers = [int(x) for x in args.layers.split(",")]
    total = args.num_prompts + args.calibration_prompts
    if args.corpus == "wikitext":
        corpus = load_dataset(
            "Salesforce/wikitext", "wikitext-103-raw-v1", split="train",
            revision=args.wikitext_revision,
        )
        prompts = [x["text"] for x in corpus if len(x["text"].split()) >= 80][:total]
    else:
        prompts = ungraded_gsm8k_rollouts(
            model, tokenizer, total, args.seed, args.rollout_offset,
            args.rollout_batch_size, args.gsm8k_revision,
        )
        corpus_path = Path(args.output + ".ungraded_rollouts.json")
        corpus_path.write_text(json.dumps({
            "source": "base_model_ungraded_gsm8k_train_rollouts",
            "seed": args.seed,
            "offset": args.rollout_offset,
            "count": len(prompts),
            "model": args.model,
            "model_revision": args.model_revision,
            "dataset": "openai/gsm8k",
            "dataset_revision": args.gsm8k_revision,
            "texts": prompts,
        }, indent=2) + "\n")
    if len(prompts) != total:
        raise ValueError(f"requested {total} corpus prompts but found {len(prompts)}")
    lens_input_sha256 = None
    if args.lens_input:
        lens_input_sha256 = sha256_file(args.lens_input)
        lens = JacobianLens.load(args.lens_input)
        if layers != list(lens.source_layers):
            raise ValueError(
                f"requested layers {layers} do not match reused lens {lens.source_layers}"
            )
    else:
        fit_identity = {
            "model": args.model,
            "model_revision": args.model_revision,
            "adapter": args.adapter,
            "adapter_sha256": (
                {
                    path.name: sha256_file(path)
                    for path in sorted(Path(args.adapter).glob("adapter_*"))
                    if path.is_file()
                }
                if args.adapter
                else None
            ),
            "corpus": args.corpus,
            "wikitext_revision": args.wikitext_revision,
            "gsm8k_revision": args.gsm8k_revision,
            "rollout_offset": args.rollout_offset,
            "seed": args.seed,
            "num_prompts": args.num_prompts,
            "layers": layers,
            "dim_batch": args.dim_batch,
            "max_seq_len": 128,
            "fit_prompts_sha256": hashlib.sha256(
                json.dumps(
                    prompts[: args.num_prompts],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        }
        checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else None
        if checkpoint_path is not None:
            checkpoint_manifest = Path(str(checkpoint_path) + ".manifest.json")
            checkpoint_exists = checkpoint_path.exists() or checkpoint_manifest.exists()
            if checkpoint_exists and not args.resume_checkpoint:
                raise FileExistsError(
                    "fit checkpoint already exists; pass --resume-checkpoint only "
                    "for an identity-matched continuation"
                )
            if checkpoint_exists:
                if not checkpoint_path.exists() or not checkpoint_manifest.is_file():
                    raise ValueError("fit checkpoint and identity manifest must both exist")
                if json.loads(checkpoint_manifest.read_text()) != fit_identity:
                    raise ValueError("fit checkpoint identity does not match this fit")
            else:
                checkpoint_manifest.parent.mkdir(parents=True, exist_ok=True)
                checkpoint_manifest.write_text(
                    json.dumps(fit_identity, indent=2, sort_keys=True) + "\n"
                )
        lens = jlens.fit(
            wrapped, prompts[: args.num_prompts], source_layers=layers,
            dim_batch=args.dim_batch, max_seq_len=128,
            checkpoint_path=(str(checkpoint_path) if checkpoint_path else None),
            resume=args.resume_checkpoint,
        )
    if (
        args.lens_input is None
        or Path(args.lens_input).resolve() != Path(args.output).resolve()
    ):
        lens.save(args.output)
    lens_sha256 = sha256_file(args.output)

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
    if not raw:
        raise ValueError("calibration produced no raw scores")
    mean = float(np.mean(raw))
    std = float(np.std(raw))
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError(f"invalid calibration statistics: mean={mean}, std={std}")
    stats = {
        "mean": mean,
        "std": std,
        "token_ids": ids,
        "target_words": target_words,
        "layers": layers,
        "model": args.model,
        "model_revision": args.model_revision,
        "adapter": args.adapter,
        "corpus": args.corpus,
        "dataset": (
            "Salesforce/wikitext" if args.corpus == "wikitext" else "openai/gsm8k"
        ),
        "dataset_revision": (
            args.wikitext_revision
            if args.corpus == "wikitext"
            else args.gsm8k_revision
        ),
        "lens_path": args.output,
        "lens_sha256": lens_sha256,
        "lens_input": args.lens_input,
        "lens_input_sha256": lens_input_sha256,
        "rollout_offset": (
            args.rollout_offset if args.corpus == "gsm8k_rollouts" else None
        ),
    }
    write_calibration(args.calibration_output, stats)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
