from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import format_prompt, gsm8k_reward, load_config, model_dtype, seed_everything
from .reward import TargetJLReward


def summarize(scores: list[float], correct: list[float], groups: list[int]) -> dict[str, float]:
    score = np.asarray(scores)
    label = np.asarray(correct)
    corr = float(np.corrcoef(score, label)[0, 1]) if label.std() and score.std() else 0.0
    correct_mean = float(score[label == 1].mean()) if (label == 1).any() else float("nan")
    incorrect_mean = float(score[label == 0].mean()) if (label == 0).any() else float("nan")
    pair_wins: list[float] = []
    group_differences: list[float] = []
    for group in sorted(set(groups)):
        idx = np.asarray(groups) == group
        positive = score[idx & (label == 1)]
        negative = score[idx & (label == 0)]
        if len(positive) and len(negative):
            group_differences.append(float(positive.mean() - negative.mean()))
            for pos in positive:
                for neg in negative:
                    pair_wins.append(float(pos > neg) + 0.5 * float(pos == neg))
    return {
        "correlation": corr,
        "correct_mean": correct_mean,
        "incorrect_mean": incorrect_mean,
        "mean_correct_minus_incorrect": correct_mean - incorrect_mean,
        "within_prompt_mean_difference": float(np.mean(group_differences)) if group_differences else float("nan"),
        "within_prompt_pair_accuracy": float(np.mean(pair_wins)) if pair_wins else float("nan"),
        "mixed_outcome_prompts": len(group_differences),
    }


def fit_group_composite(
    candidate_scores: dict[str, list[float]], correct: list[float], groups: list[int], ridge: float = 1.0
) -> dict[str, Any]:
    names = list(candidate_scores)
    features = np.column_stack([candidate_scores[name] for name in names])
    labels = np.asarray(correct, dtype=float)
    group_ids = np.asarray(groups)

    centered_features = features.copy()
    centered_labels = labels.copy()
    for group in np.unique(group_ids):
        idx = group_ids == group
        centered_features[idx] -= centered_features[idx].mean(axis=0)
        centered_labels[idx] -= centered_labels[idx].mean()

    predictions = np.zeros(len(labels))
    for fold in range(5):
        test = group_ids % 5 == fold
        train = ~test
        scale = centered_features[train].std(axis=0)
        scale[scale < 1e-6] = 1.0
        x_train = centered_features[train] / scale
        weights = np.linalg.solve(
            x_train.T @ x_train + ridge * np.eye(x_train.shape[1]),
            x_train.T @ centered_labels[train],
        ) / scale
        predictions[test] = centered_features[test] @ weights

    scale = centered_features.std(axis=0)
    scale[scale < 1e-6] = 1.0
    x = centered_features / scale
    weights = np.linalg.solve(x.T @ x + ridge * np.eye(x.shape[1]), x.T @ centered_labels) / scale
    return {
        "ridge": ridge,
        "weights": {name: float(weight) for name, weight in zip(names, weights, strict=True)},
        "cross_validated": summarize(predictions.tolist(), correct, groups),
    }


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/jlens.json")
    p.add_argument("--prompts", type=int, default=30)
    p.add_argument("--generations", type=int, default=8)
    p.add_argument("--output", default="artifacts/solved_alignment.json")
    args = p.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], torch_dtype=model_dtype(), device_map={"": "cuda:0"}
    )
    model.eval()
    scorer = TargetJLReward(
        cfg["lens_path"], cfg["calibration_path"], tokenizer, cfg["target_words"],
        cfg["score_stride"], cfg["mask_target_tokens"], cfg.get("vocab_chunk_size", 16384),
    )
    candidates = [
        (f"layer{layer}_{window}", layer, window)
        for layer in scorer.lens.source_layers
        for window in ("all_mean", "late_mean", "final")
    ]
    candidate_scores: dict[str, list[float]] = {name: [] for name, _, _ in candidates}
    correct: list[float] = []
    groups: list[int] = []

    dataset = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=cfg["seed"])
    for group, row in enumerate(dataset.select(range(args.prompts))):
        prompt = format_prompt(tokenizer, row["question"])
        encoded = tokenizer(
            [prompt] * args.generations, return_tensors="pt", padding=True,
            truncation=True, max_length=cfg["max_prompt_tokens"],
        ).to(model.device)
        prompt_width = encoded.input_ids.shape[1]
        generated = model.generate(
            **encoded, max_new_tokens=cfg["max_new_tokens"], do_sample=True,
            temperature=cfg["temperature"], pad_token_id=tokenizer.pad_token_id,
        )
        sequences: list[torch.Tensor] = []
        prompt_lengths: list[int] = []
        for index in range(args.generations):
            completion = generated[index, prompt_width:]
            eos = (completion == tokenizer.eos_token_id).nonzero()
            if eos.numel():
                completion = completion[: int(eos[0].item()) + 1]
            prompt_ids = encoded.input_ids[index][encoded.attention_mask[index].bool()]
            sequences.append(torch.cat([prompt_ids, completion]))
            prompt_lengths.append(len(prompt_ids))
            text = tokenizer.decode(completion, skip_special_tokens=True)
            correct.append(gsm8k_reward(text, row["answer"]))
            groups.append(group)
        ids = pad_sequence(sequences, batch_first=True, padding_value=tokenizer.pad_token_id)
        mask = pad_sequence(
            [torch.ones_like(sequence) for sequence in sequences], batch_first=True, padding_value=0
        )
        output = model(ids, attention_mask=mask, output_hidden_states=True, use_cache=False)
        for name, layer, window in candidates:
            scorer.score_layers = [layer]
            scorer.score_start_fraction = 0.5 if window != "all_mean" else 0.0
            scorer.score_aggregation = "last" if window == "final" else "mean"
            scorer.score_include_final = window == "final"
            candidate_scores[name].extend(
                scorer(model, output.hidden_states, prompt_lengths[i], mask[i], i, ids[i])
                for i in range(args.generations)
            )
        print(f"alignment prompt {group + 1}/{args.prompts}", flush=True)

    result: dict[str, Any] = {
        "prompts": args.prompts,
        "generations": args.generations,
        "overall_exact_match": float(np.mean(correct)),
        "candidates": {
            name: summarize(scores, correct, groups) for name, scores in candidate_scores.items()
        },
        "composite": fit_group_composite(candidate_scores, correct, groups),
        "samples": {
            "correct": correct,
            "groups": groups,
            "scores": candidate_scores,
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
