from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def read_validation(path: str) -> list[dict]:
    return [row for row in json.loads(Path(path).read_text())
            if "eval_rewards/gsm8k_reward_trl/mean" in row]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gsm8k", default="runs/gsm8k_reward/log_history.json")
    p.add_argument("--jlens", default="runs/jlens_solved_reward/log_history.json")
    p.add_argument("--validation-examples", type=int, default=200)
    p.add_argument("--output", default="runs/comparison.png")
    args = p.parse_args()
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 7))
    for label, path in [("GSM8K reward", args.gsm8k), ("J-lens solved reward", args.jlens)]:
        rows = read_validation(path)
        steps = [r["step"] for r in rows]
        exact = [r["eval_rewards/gsm8k_reward_trl/mean"] for r in rows]
        ci = [1.96 * (max(x * (1 - x), 1e-12) / args.validation_examples) ** .5 for x in exact]
        axes[0].plot(steps, exact, marker="o", label=label)
        axes[0].fill_between(steps, [x-y for x, y in zip(exact, ci)],
                            [x+y for x, y in zip(exact, ci)], alpha=.15)
        axes[1].plot(steps, [r["eval_rewards/jlens_solved_reward/mean"] for r in rows], marker="o", label=label)
    axes[0].set_ylabel("Held-out GSM8K exact match")
    axes[1].set_ylabel("Held-out solved J-score (z)")
    axes[1].set_xlabel("Optimizer update")
    for ax in axes:
        ax.grid(alpha=.25); ax.legend()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
