"""Bake the pinned public model and GSM8K dataset into the Modal image."""

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from jlens_rl.common import GSM8K_REVISION, QWEN_MODEL_REVISION


def main() -> None:
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    AutoTokenizer.from_pretrained(model_name, revision=QWEN_MODEL_REVISION)
    AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=QWEN_MODEL_REVISION,
        dtype="auto",
    )
    load_dataset(
        "openai/gsm8k",
        "main",
        split="train",
        revision=GSM8K_REVISION,
    )


if __name__ == "__main__":
    main()
