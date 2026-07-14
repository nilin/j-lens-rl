"""Bake V6's pinned public assets with scoped Hugging Face authentication."""

import os

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from jlens_rl.common import GSM8K_REVISION, QWEN_MODEL_REVISION


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is required by the V6 authenticated asset-cache build layer"
        )
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    AutoTokenizer.from_pretrained(
        model_name,
        revision=QWEN_MODEL_REVISION,
        token=token,
    )
    AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=QWEN_MODEL_REVISION,
        dtype="auto",
        token=token,
    )
    load_dataset(
        "openai/gsm8k",
        "main",
        split="train",
        revision=GSM8K_REVISION,
        token=token,
    )


if __name__ == "__main__":
    main()
