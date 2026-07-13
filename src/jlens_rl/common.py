from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

SYSTEM_PROMPT = (
    "Solve the math problem. Show concise reasoning, then put only the final "
    "numeric answer after '#### '."
)


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    cfg = json.loads(path.read_text())
    if "base" in cfg:
        base = load_config(path.parent / cfg.pop("base"))
        base.update(cfg)
        cfg = base
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_answer(text: str) -> str | None:
    marked = re.findall(r"####\s*([-+]?[$\d][\d,]*(?:\.\d+)?)", text)
    candidates = marked or re.findall(r"[-+]?[$\d][\d,]*(?:\.\d+)?", text)
    if not candidates:
        return None
    value = candidates[-1].replace("$", "").replace(",", "")
    try:
        number = float(value)
        return str(int(number)) if number.is_integer() else str(number)
    except ValueError:
        return None


def gsm8k_reward(completion: str, reference: str) -> float:
    return float(extract_answer(completion) == extract_answer(reference))


def binomial_ci95(successes: int, total: int) -> tuple[float, float]:
    """Wilson 95% interval, which remains meaningful at 0% and 100%."""
    if total <= 0:
        raise ValueError("total must be positive")
    z = 1.96
    p = successes / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    radius = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return float(max(0, center - radius)), float(min(1, center + radius))


def format_prompt(tokenizer: Any, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def model_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")
