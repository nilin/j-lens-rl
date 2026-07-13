from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from .common import binomial_ci95


def read_validation(path: str) -> list[dict]:
    """Read the fixed greedy validation measurements logged by our callback."""
    validation_path = Path(path).with_name("validation_history.jsonl")
    if validation_path.exists():
        return [json.loads(line) for line in validation_path.read_text().splitlines()]
    return [
        row
        for row in json.loads(Path(path).read_text())
        if "validation/exact_match" in row
    ]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gsm8k", default="runs/gsm8k_reward/log_history.json")
    p.add_argument("--jlens", default="runs/jlens_solved_reward/log_history.json")
    p.add_argument("--validation-examples", type=int, default=200)
    p.add_argument("--output", default="runs/comparison.png")
    args = p.parse_args()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, path in [("GSM8K reward", args.gsm8k), ("J-lens solved reward", args.jlens)]:
        rows = read_validation(path)
        steps = [r["step"] for r in rows]
        exact = [r.get("exact_match", r.get("validation/exact_match")) for r in rows]
        intervals = [
            binomial_ci95(round(x * args.validation_examples), args.validation_examples)
            for x in exact
        ]
        ax.plot(steps, exact, marker="o", label=label)
        ax.fill_between(steps, [x[0] for x in intervals],
                        [x[1] for x in intervals], alpha=.15)
    ax.set_ylabel("Held-out GSM8K exact match")
    ax.set_xlabel("Optimizer update")
    ax.set_ylim(0, 1)
    ax.grid(alpha=.25)
    ax.legend()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
