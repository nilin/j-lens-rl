#!/usr/bin/env python3
"""Prepare and guard the predeclared J-lens confirmatory experiment.

This script deliberately does not choose hyperparameters or checkpoints from
accuracy.  It creates source-index manifests from historically unused GSM8K
training examples, fingerprints the committed protocol and artifacts, verifies
completed matched runs, and evaluates the one predeclared curve gate.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO = Path(__file__).resolve().parents[1]
STATE_DIR = REPO / ".confirmatory"
MANIFEST_DIR = STATE_DIR / "manifests"
STATE_PATH = STATE_DIR / "protocol_state.json"
CURVE_GATE_PATH = STATE_DIR / "evidence" / "curve_gate.json"
CURVE_PLOT_PATH = STATE_DIR / "evidence" / "curve.png"
COMPLETED_RUNS_PATH = STATE_DIR / "evidence" / "completed_runs.json"
UNLOCK_PATH = STATE_DIR / "final_unlocked.json"
ACCEPTANCE_PATH = STATE_DIR / "evidence" / "acceptance.json"

MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
PROTOCOL = "j-lens-rl-confirmatory-v2"
V1_ALLOCATION_SALT = "j-lens-rl-confirmatory-v1-2026-07-14"
ALLOCATION_SALT = "j-lens-rl-confirmatory-v2-2026-07-14"
SEEDS = tuple(range(142, 148))
REQUIRED_CONDITIONS = ("jlens", "signflip")
CONFIG_SEEDS = {
    "jlens": SEEDS,
    "signflip": SEEDS,
    "gsm8k": (142,),
}
FIXED_UPDATES = 25
CURVE_STEPS = (0, 5, 10, 15)
ALL_VALIDATION_STEPS = tuple(range(0, FIXED_UPDATES + 1, 5))

SPLIT_SIZES = {
    "curve_indices.json": 400,
    "sealed_final_indices.json": 2900,
    "future_reserve_indices.json": 64,
}

# Every historically documented shuffled training selection. The last rule is
# the interrupted xufk8x08 setup run, which excluded only its dev slice.
HISTORICAL_SHUFFLE_RULES = (
    (42, 1150, ()),
    (43, 1000, ()),
    (42, 1000, ((6800, 7200),)),
    (43, 1000, ((6800, 7200),)),
    (42, 1000, ((7000, 7200),)),
)

class ProtocolError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def require_clean_worktree() -> str:
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise ProtocolError(
            "confirmatory work requires a clean committed tree; git status is:\n"
            + status
        )
    return git("rev-parse", "HEAD")


def load_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(path.read_text())
    base_name = cfg.pop("base", None)
    if base_name:
        base = load_config(path.parent / base_name)
        base.update(cfg)
        cfg = base
    return cfg


def config_path(condition: str, seed: int) -> Path:
    return REPO / "configs" / f"confirmatory_{condition}_seed{seed}.json"


def all_config_paths() -> list[Path]:
    paths = [REPO / "configs" / "confirmatory_common.json"]
    paths.extend(
        config_path(condition, seed)
        for condition, seeds in CONFIG_SEEDS.items()
        for seed in seeds
    )
    paths.append(REPO / "configs" / "confirmatory_sealed_eval.json")
    return paths


def run_dir(condition: str, seed: int) -> Path:
    return STATE_DIR / "runs" / f"{condition}_seed{seed}"


def manifest_payload(indices: Iterable[int]) -> dict[str, Any]:
    return {
        "dataset": "openai/gsm8k",
        "subset": "main",
        "split": "train",
        "indices": [int(index) for index in indices],
    }


def load_indices(path: Path) -> list[int]:
    payload = json.loads(path.read_text())
    values = payload.get("indices") if isinstance(payload, dict) else payload
    if not isinstance(values, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in values
    ):
        raise ProtocolError(f"invalid index manifest: {path}")
    if len(values) != len(set(values)) or any(value < 0 for value in values):
        raise ProtocolError(f"duplicate or negative index in {path}")
    return values


def validate_configs(require_manifests: bool) -> dict[str, str]:
    common = load_config(REPO / "configs" / "confirmatory_common.json")
    required_common = {
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_revision": MODEL_REVISION,
        "dataset_revision": DATASET_REVISION,
        "validation_source": "train",
        "validation_indices_path": ".confirmatory/manifests/curve_indices.json",
        "reserved_train_indices_path": ".confirmatory/manifests/train_exclusions.json",
        "validation_examples": 400,
        "validation_batch_size": 64,
        "train_examples": 1000,
        "updates": FIXED_UPDATES,
        "min_new_tokens": 64,
        "eval_every": 5,
        "validation_observational_only": True,
        "require_clean_repository": True,
        "validation_steps": None,
        "early_stopping_patience": None,
        "save_every": FIXED_UPDATES,
        "save_total_limit": 1,
        "learning_rate": 3e-6,
    }
    for key, expected in required_common.items():
        if common.get(key) != expected:
            raise ProtocolError(
                f"confirmatory_common {key!r} is {common.get(key)!r}, expected {expected!r}"
            )

    semantic_reference: dict[int, dict[str, Any]] = {}
    for hash_key in ("lens_sha256", "calibration_sha256"):
        value = common.get(hash_key)
        if not isinstance(value, str) or len(value) != 64:
            raise ProtocolError(f"confirmatory_common {hash_key!r} must be a SHA-256")

    for condition, condition_seeds in CONFIG_SEEDS.items():
        for seed in condition_seeds:
            path = config_path(condition, seed)
            cfg = load_config(path)
            if cfg.get("seed") != seed:
                raise ProtocolError(f"wrong seed in {path}")
            expected_output = f".confirmatory/runs/{condition}_seed{seed}"
            if cfg.get("output_dir") != expected_output:
                raise ProtocolError(f"wrong output_dir in {path}")
            expected_reward = "gsm8k" if condition == "gsm8k" else "jlens"
            if cfg.get("reward_type") != expected_reward:
                raise ProtocolError(f"wrong reward_type in {path}")
            if condition in {"jlens", "signflip"}:
                components = cfg.get("score_components")
                expected_weight = 1.0 if condition == "jlens" else -1.0
                if not isinstance(components, list) or len(components) != 1:
                    raise ProtocolError(f"{path} must have exactly one score component")
                component = components[0]
                expected_component = {
                    "layer": 8,
                    "start_fraction": 0.5,
                    "end_fraction": 1.0,
                    "aggregation": "mean",
                    "weight": expected_weight,
                }
                if component != expected_component:
                    raise ProtocolError(f"unexpected score component in {path}")

            # For a seed, every optimizer/generation/data setting must match.
            comparable = dict(cfg)
            for key in ("reward_type", "score_components", "run_name", "output_dir"):
                comparable.pop(key, None)
            if condition == "jlens":
                semantic_reference[seed] = comparable
            elif comparable != semantic_reference[seed]:
                raise ProtocolError(
                    f"condition {condition} seed {seed} is not matched to semantic treatment"
                )

    eval_cfg = load_config(REPO / "configs" / "confirmatory_sealed_eval.json")
    expected_eval = {
        "evaluation_source": "train",
        "evaluation_indices_path": ".confirmatory/manifests/sealed_final_indices.json",
        "validation_examples": 2900,
        "min_new_tokens": 0,
    }
    for key, expected in expected_eval.items():
        if eval_cfg.get(key) != expected:
            raise ProtocolError(f"sealed eval {key!r} is not frozen to {expected!r}")

    if require_manifests:
        if len(load_indices(MANIFEST_DIR / "curve_indices.json")) != 400:
            raise ProtocolError("curve manifest must contain exactly 400 indices")
        if len(load_indices(MANIFEST_DIR / "sealed_final_indices.json")) != 2900:
            raise ProtocolError("sealed-final manifest must contain exactly 2,900 indices")

    return {str(path.relative_to(REPO)): sha256_file(path) for path in all_config_paths()}


def validate_artifacts() -> dict[str, str]:
    common = load_config(REPO / "configs" / "confirmatory_common.json")
    lens = REPO / "artifacts" / "qwen25_05b_solved_lens.pt"
    calibration = REPO / "artifacts" / "qwen25_05b_solved_calibration.json"
    for path in (lens, calibration):
        if not path.is_file():
            raise ProtocolError(f"missing frozen artifact: {path}")
    actual = {
        str(lens.relative_to(REPO)): sha256_file(lens),
        str(calibration.relative_to(REPO)): sha256_file(calibration),
    }
    expected = {
        str(lens.relative_to(REPO)): common["lens_sha256"],
        str(calibration.relative_to(REPO)): common["calibration_sha256"],
    }
    if actual != expected:
        raise ProtocolError(f"artifact hash mismatch: {actual!r}")
    metadata = json.loads(calibration.read_text())
    if metadata.get("target_words") != ["solved"]:
        raise ProtocolError("calibration is not for the frozen 'solved' target")
    if metadata.get("model") != "Qwen/Qwen2.5-0.5B-Instruct":
        raise ProtocolError("calibration model does not match the protocol")
    if metadata.get("model_revision") != MODEL_REVISION:
        raise ProtocolError("calibration model revision does not match the protocol")
    token_ids = metadata.get("token_ids")
    if metadata.get("layers") != [8, 14, 20]:
        raise ProtocolError("calibration layers do not match the frozen lens")
    if not isinstance(token_ids, list) or not token_ids or any(
        isinstance(token, bool) or not isinstance(token, int) for token in token_ids
    ):
        raise ProtocolError("calibration token IDs are missing or malformed")
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise ProtocolError("run this command with .venv/bin/python") from error
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct", revision=MODEL_REVISION
    )
    variants = {
        prefix + spelling
        for spelling in ("solved", "Solved", "SOLVED")
        for prefix in ("", " ")
    }
    live_ids = sorted({
        encoded[0]
        for variant in variants
        if len(encoded := tokenizer.encode(variant, add_special_tokens=False)) == 1
    })
    if sorted(token_ids) != live_ids:
        raise ProtocolError(
            f"calibration token IDs {sorted(token_ids)} do not match pinned tokenizer {live_ids}"
        )
    return actual


def reconstruct_historical_indices() -> tuple[list[int], list[int], int]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise ProtocolError("run this command with .venv/bin/python") from error

    raw = load_dataset(
        "openai/gsm8k", "main", split="train", revision=DATASET_REVISION
    )
    raw = raw.add_column("_source_index", range(len(raw)))
    if len(raw) != 7473:
        raise ProtocolError(f"expected 7,473 GSM8K train rows, found {len(raw)}")

    historical = set(range(150))
    historical.update(range(6800, 7200))
    v1_historical: set[int] | None = None
    for rule_number, (seed, count, excluded_ranges) in enumerate(
        HISTORICAL_SHUFFLE_RULES, 1
    ):
        pool = raw
        if excluded_ranges:
            pool = raw.filter(
                lambda row: not any(
                    start <= row["_source_index"] < end
                    for start, end in excluded_ranges
                )
            )
        historical.update(
            pool.shuffle(seed=seed).select(range(count))["_source_index"]
        )
        if rule_number == len(HISTORICAL_SHUFFLE_RULES) - 1:
            v1_historical = set(historical)
    if v1_historical is None:
        raise ProtocolError("could not reconstruct the retired v1 allocation")
    return (
        sorted(int(index) for index in historical),
        sorted(int(index) for index in v1_historical),
        len(raw),
    )


def allocation_key(index: int, salt: str = ALLOCATION_SALT) -> bytes:
    return hashlib.sha256(f"{salt}:{index}".encode()).digest()


def prepare() -> None:
    commit = require_clean_worktree()
    if STATE_DIR.exists():
        raise ProtocolError(
            f"{STATE_DIR} already exists; do not overwrite a prepared protocol"
        )
    config_hashes = validate_configs(require_manifests=False)
    artifact_hashes = validate_artifacts()
    historical, v1_historical, dataset_size = reconstruct_historical_indices()
    fresh_set = set(range(dataset_size)) - set(historical)
    v1_fresh = sorted(
        set(range(dataset_size)) - set(v1_historical),
        key=lambda index: allocation_key(index, V1_ALLOCATION_SALT),
    )
    retired_v1_curve = v1_fresh[200:600]
    allocatable = sorted(fresh_set - set(retired_v1_curve), key=allocation_key)
    if (
        len(historical) != 3741
        or len(fresh_set) != 3732
        or len(retired_v1_curve) != 400
        or len(set(retired_v1_curve) & set(historical)) != 32
        or len(allocatable) != 3364
    ):
        raise ProtocolError(
            "historical/fresh/retired allocation counts changed: "
            f"{len(historical)}/{len(fresh_set)}/{len(retired_v1_curve)}/"
            f"{len(allocatable)}"
        )

    cursor = 0
    allocations: dict[str, list[int]] = {}
    for name, size in SPLIT_SIZES.items():
        allocations[name] = allocatable[cursor : cursor + size]
        cursor += size
    if cursor != len(allocatable):
        raise ProtocolError("predeclared split sizes do not exhaust the fresh pool")

    MANIFEST_DIR.mkdir(parents=True)
    write_json(MANIFEST_DIR / "historical_exclusions.json", manifest_payload(historical))
    write_json(
        MANIFEST_DIR / "retired_v1_curve_indices.json",
        manifest_payload(retired_v1_curve),
    )
    for name, indices in allocations.items():
        write_json(MANIFEST_DIR / name, manifest_payload(indices))
    # All fresh indices and every exposed v1 curve item stay out of v2 training.
    train_exclusions = sorted(fresh_set | set(retired_v1_curve))
    write_json(
        MANIFEST_DIR / "train_exclusions.json",
        manifest_payload(train_exclusions),
    )

    manifest_hashes = {
        str(path.relative_to(REPO)): sha256_file(path)
        for path in sorted(MANIFEST_DIR.glob("*.json"))
    }
    state = {
        "protocol": PROTOCOL,
        "prepared_at_utc": utc_now(),
        "git_commit": commit,
        "allocation_algorithm": "ascending SHA-256(salt + ':' + raw_source_index)",
        "allocation_salt": ALLOCATION_SALT,
        "dataset": "openai/gsm8k:main",
        "dataset_revision": DATASET_REVISION,
        "dataset_train_rows": dataset_size,
        "historically_used_count": len(historical),
        "historically_unused_count": len(fresh_set),
        "retired_v1_curve_count": len(retired_v1_curve),
        "retired_v1_curve_historical_overlap": len(
            set(retired_v1_curve) & set(historical)
        ),
        "historical_shuffle_rules": HISTORICAL_SHUFFLE_RULES,
        "split_sizes": SPLIT_SIZES,
        "curve_gate_steps": list(CURVE_STEPS),
        "fixed_training_updates": FIXED_UPDATES,
        "seeds": list(SEEDS),
        "config_sha256": config_hashes,
        "artifact_sha256": artifact_hashes,
        "index_manifest_sha256": manifest_hashes,
    }
    write_json(STATE_PATH, state)
    print(json.dumps(state, indent=2, sort_keys=True))


def load_and_verify_state() -> dict[str, Any]:
    commit = require_clean_worktree()
    if not STATE_PATH.is_file():
        raise ProtocolError("protocol is not prepared; run the prepare command first")
    state = json.loads(STATE_PATH.read_text())
    expected_state = {
        "protocol": PROTOCOL,
        "allocation_salt": ALLOCATION_SALT,
        "historically_used_count": 3741,
        "historically_unused_count": 3732,
        "retired_v1_curve_count": 400,
        "retired_v1_curve_historical_overlap": 32,
        "split_sizes": SPLIT_SIZES,
        "curve_gate_steps": list(CURVE_STEPS),
        "fixed_training_updates": FIXED_UPDATES,
        "seeds": list(SEEDS),
    }
    if any(state.get(key) != value for key, value in expected_state.items()):
        raise ProtocolError("prepared state does not match the frozen v2 protocol")
    if canonical_sha256(state.get("historical_shuffle_rules")) != canonical_sha256(
        HISTORICAL_SHUFFLE_RULES
    ):
        raise ProtocolError("prepared historical reconstruction rules changed")
    if state.get("git_commit") != commit:
        raise ProtocolError(
            f"prepared commit {state.get('git_commit')} does not match HEAD {commit}"
        )
    if state.get("config_sha256") != validate_configs(require_manifests=True):
        raise ProtocolError("a frozen config changed after protocol preparation")
    if state.get("artifact_sha256") != validate_artifacts():
        raise ProtocolError("a frozen artifact changed after protocol preparation")
    actual_manifest_hashes = {
        str(path.relative_to(REPO)): sha256_file(path)
        for path in sorted(MANIFEST_DIR.glob("*.json"))
    }
    if state.get("index_manifest_sha256") != actual_manifest_hashes:
        raise ProtocolError("an index manifest changed after protocol preparation")

    split_sets = {
        name: set(load_indices(MANIFEST_DIR / name)) for name in SPLIT_SIZES
    }
    names = list(split_sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if split_sets[left] & split_sets[right]:
                raise ProtocolError(f"manifest overlap: {left} and {right}")
    retired = set(load_indices(MANIFEST_DIR / "retired_v1_curve_indices.json"))
    historical = set(load_indices(MANIFEST_DIR / "historical_exclusions.json"))
    train_exclusions = set(load_indices(MANIFEST_DIR / "train_exclusions.json"))
    if any(values & retired for values in split_sets.values()):
        raise ProtocolError("a v2 outcome split reuses the exposed v1 curve")
    if any(values & historical for values in split_sets.values()):
        raise ProtocolError("a v2 outcome split contains a historically used index")
    if sum(map(len, split_sets.values())) != 3364:
        raise ProtocolError("v2 outcome manifests no longer contain 3,364 indices")
    if len(historical) != 3741 or len(retired) != 400 or len(train_exclusions) != 3764:
        raise ProtocolError("historical or retired exclusion counts changed")
    if train_exclusions != set().union(*split_sets.values(), retired):
        raise ProtocolError("training exclusions do not cover every v2 outcome index")
    return state


def load_history(path: Path) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        raise ProtocolError(f"missing validation history: {path}")
    rows: dict[int, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        step = row.get("step")
        score = row.get("exact_match")
        if isinstance(step, bool) or not isinstance(step, int):
            raise ProtocolError(f"invalid step at {path}:{line_number}")
        if step in rows:
            raise ProtocolError(f"duplicate validation step {step} in {path}")
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise ProtocolError(f"invalid exact_match at {path}:{line_number}")
        if not 0 <= float(score) <= 1:
            raise ProtocolError(f"exact_match is outside [0,1] at {path}:{line_number}")
        rows[step] = row
    if tuple(sorted(rows)) != ALL_VALIDATION_STEPS:
        raise ProtocolError(
            f"{path} has steps {sorted(rows)}, expected {list(ALL_VALIDATION_STEPS)}"
        )
    return rows


def training_behavior_summary(path: Path) -> dict[str, Any]:
    """Validate and summarize the J-only rollout diagnostics for one run."""
    if not path.is_file():
        raise ProtocolError(f"missing training log history: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ProtocolError(f"training log history is not a list: {path}")
    rows = [row for row in payload if isinstance(row, dict) and "reward" in row]
    if [row.get("step") for row in rows] != list(range(1, FIXED_UPDATES + 1)):
        raise ProtocolError(f"training log does not contain steps 1..{FIXED_UPDATES}: {path}")

    literal_rates: list[float] = []
    mean_lengths: list[float] = []
    clipped_ratios: list[float] = []
    rewards: list[float] = []
    reward_stds: list[float] = []
    for row in rows:
        reward_keys = [
            key for key in row if key.startswith("rewards/") and key.endswith("/mean")
        ]
        if len(reward_keys) != 1 or any("gsm8k" in key.lower() for key in row):
            raise ProtocolError(f"training log is not a one-J-reward run: {path}")
        if "jlens/solved_literal_rate" not in row:
            raise ProtocolError(f"training log lacks the solved literal audit: {path}")
        numeric_fields = (
            "reward",
            "reward_std",
            "completions/mean_length",
            "completions/clipped_ratio",
            "jlens/solved_literal_rate",
        )
        if any(
            not isinstance(row.get(key), (int, float))
            or isinstance(row.get(key), bool)
            or not math.isfinite(float(row[key]))
            for key in numeric_fields
        ):
            raise ProtocolError(f"training log has a non-finite diagnostic: {path}")
        literal_rate = float(row["jlens/solved_literal_rate"])
        mean_length = float(row["completions/mean_length"])
        clipped_ratio = float(row["completions/clipped_ratio"])
        if not 0 <= literal_rate <= 1 or not 0 <= clipped_ratio <= 1:
            raise ProtocolError(f"training log has an invalid rate: {path}")
        if not 0 <= mean_length <= 256:
            raise ProtocolError(f"training log has an invalid completion length: {path}")
        literal_rates.append(literal_rate)
        mean_lengths.append(mean_length)
        clipped_ratios.append(clipped_ratio)
        rewards.append(float(row["reward"]))
        reward_stds.append(float(row["reward_std"]))
    return {
        "steps": len(rows),
        "literal_target_rate_max": max(literal_rates),
        "completion_mean_length_min": min(mean_lengths),
        "completion_mean_length_max": max(mean_lengths),
        "completion_clipped_ratio_max": max(clipped_ratios),
        "reward_first": rewards[0],
        "reward_last": rewards[-1],
        "reward_std_min": min(reward_stds),
    }


def render_curve_plot(
    full_per_seed: dict[str, dict[int, float]], means: dict[str, float]
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ProtocolError("matplotlib is required to render the curve evidence") from error
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    all_values: list[float] = []
    for seed in SEEDS:
        values = [100 * full_per_seed[str(seed)][step] for step in ALL_VALIDATION_STEPS]
        all_values.extend(values)
        axis.plot(
            ALL_VALIDATION_STEPS, values, color="#9ca3af", alpha=0.5,
            linewidth=1, marker=".", label="individual seed" if seed == SEEDS[0] else None,
        )
    mean_values = [
        100 * sum(full_per_seed[str(seed)][step] for seed in SEEDS) / len(SEEDS)
        for step in ALL_VALIDATION_STEPS
    ]
    axis.plot(
        ALL_VALIDATION_STEPS, mean_values, color="#1d4ed8", linewidth=2.5,
        marker="o", label="six-seed mean",
    )
    gate_values = [100 * means[str(step)] for step in CURVE_STEPS]
    axis.scatter(
        CURVE_STEPS, gate_values, color="#dc2626", zorder=5,
        label="predeclared gate nodes",
    )
    padding = max(1.0, (max(all_values) - min(all_values)) * 0.15)
    axis.set_ylim(max(0, min(all_values) - padding), min(100, max(all_values) + padding))
    axis.set_xticks(ALL_VALIDATION_STEPS)
    axis.set_xlabel("Optimizer update")
    axis.set_ylabel("Greedy exact match (%) — truncated axis")
    axis.set_title("Confirmatory J-lens treatment: fixed observational curve")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    CURVE_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(CURVE_PLOT_PATH, dpi=180)
    plt.close(figure)


def compute_curve_gate(write_result: bool = True) -> dict[str, Any]:
    state = load_and_verify_state()
    per_seed: dict[str, dict[str, float]] = {}
    full_per_seed: dict[str, dict[int, float]] = {}
    for seed in SEEDS:
        history = load_history(run_dir("jlens", seed) / "validation_history.jsonl")
        full_per_seed[str(seed)] = {
            step: float(history[step]["exact_match"]) for step in ALL_VALIDATION_STEPS
        }
        per_seed[str(seed)] = {
            str(step): float(history[step]["exact_match"]) for step in CURVE_STEPS
        }
    means = {
        str(step): sum(per_seed[str(seed)][str(step)] for seed in SEEDS) / len(SEEDS)
        for step in CURVE_STEPS
    }
    passed = (
        means["5"] > means["0"]
        and means["10"] >= means["5"]
        and means["15"] >= means["10"]
    )
    result = {
        "protocol": state["protocol"],
        "git_commit": state["git_commit"],
        "criterion": "mean(step5) > mean(step0), mean(step10) >= mean(step5), mean(step15) >= mean(step10)",
        "predeclared_steps": list(CURVE_STEPS),
        "n_seeds": len(SEEDS),
        "examples_per_seed": 400,
        "per_seed_exact_match": per_seed,
        "mean_exact_match": means,
        "passed": passed,
        "computed_at_utc": utc_now(),
    }
    if write_result:
        render_curve_plot(full_per_seed, means)
        result["curve_plot"] = {
            "path": str(CURVE_PLOT_PATH.resolve()),
            "sha256": sha256_file(CURVE_PLOT_PATH),
        }
        write_json(CURVE_GATE_PATH, result)
    return result


def verify_completed_runs(
    conditions: tuple[str, ...] = REQUIRED_CONDITIONS,
) -> None:
    state = load_and_verify_state()
    if not conditions or any(condition not in REQUIRED_CONDITIONS for condition in conditions):
        raise ProtocolError(f"invalid conditions for run verification: {conditions}")
    curve_indices = load_indices(MANIFEST_DIR / "curve_indices.json")
    curve_manifest_sha256 = sha256_file(MANIFEST_DIR / "curve_indices.json")
    excluded = set(load_indices(MANIFEST_DIR / "train_exclusions.json"))
    matched_train_indices: dict[int, list[int]] = {}
    matched_runtime: dict[str, Any] | None = None

    for condition in conditions:
        for seed in SEEDS:
            directory = run_dir(condition, seed)
            expected_cfg = load_config(config_path(condition, seed))
            resolved_path = directory / "resolved_config.json"
            if not resolved_path.is_file() or json.loads(resolved_path.read_text()) != expected_cfg:
                raise ProtocolError(f"resolved config mismatch for {condition} seed {seed}")
            manifest = json.loads((directory / "run_manifest.json").read_text())
            if (
                manifest.get("git_commit") != state["git_commit"]
                or manifest.get("git_dirty") is not False
                or not isinstance(manifest.get("source_tree_sha256"), str)
                or len(manifest["source_tree_sha256"]) != 64
            ):
                raise ProtocolError(f"invalid source provenance for {condition} seed {seed}")
            runtime = manifest.get("runtime")
            if (
                not isinstance(runtime, dict)
                or "L40S" not in str(runtime.get("cuda_device_name", ""))
                or not isinstance(runtime.get("cuda_version"), str)
                or not runtime["cuda_version"]
            ):
                raise ProtocolError(f"wrong training runtime for {condition} seed {seed}")
            if matched_runtime is None:
                matched_runtime = runtime
            elif runtime != matched_runtime:
                raise ProtocolError("required runs used different numerical runtimes")
            expected_config_path = config_path(condition, seed)
            if manifest.get("config_sha256") != sha256_file(expected_config_path):
                raise ProtocolError(f"config hash mismatch for {condition} seed {seed}")
            if manifest.get("resolved_config_sha256") != sha256_file(resolved_path):
                raise ProtocolError(f"resolved-config hash mismatch for {condition} seed {seed}")
            data = json.loads((directory / "data_indices.json").read_text())
            if manifest.get("data_indices_sha256") != sha256_file(
                directory / "data_indices.json"
            ):
                raise ProtocolError(f"data-index hash mismatch for {condition} seed {seed}")
            train_indices = [int(value) for value in data["train_source_indices"]]
            if len(train_indices) != 1000 or len(set(train_indices)) != 1000:
                raise ProtocolError(f"invalid training indices for {condition} seed {seed}")
            if set(train_indices) & excluded:
                raise ProtocolError(f"reserved index entered training for {condition} seed {seed}")
            if data.get("validation_source") != "train":
                raise ProtocolError(f"wrong validation source for {condition} seed {seed}")
            if data.get("validation_source_indices") != curve_indices:
                raise ProtocolError(f"wrong curve indices for {condition} seed {seed}")
            if seed not in matched_train_indices:
                matched_train_indices[seed] = train_indices
            elif matched_train_indices[seed] != train_indices:
                raise ProtocolError(f"training data mismatch across conditions for seed {seed}")

            history = load_history(directory / "validation_history.jsonl")
            if any(
                row.get("validation_source") != "train"
                or row.get("validation_indices_sha256") != curve_manifest_sha256
                or not isinstance(row.get("mean_length"), (int, float))
                or not math.isfinite(float(row["mean_length"]))
                or not 0 <= float(row["mean_length"]) <= 256
                or not isinstance(
                    row.get("literal_target_completion_rate"), (int, float)
                )
                or not math.isfinite(float(row["literal_target_completion_rate"]))
                or not 0 <= float(row["literal_target_completion_rate"]) <= 1
                for row in history.values()
            ):
                raise ProtocolError(
                    f"validation identity mismatch for {condition} seed {seed}"
                )
            training_behavior_summary(directory / "log_history.json")
            trainer_state_path = directory / f"checkpoint-{FIXED_UPDATES}" / "trainer_state.json"
            trainer_state = json.loads(trainer_state_path.read_text())
            if trainer_state.get("global_step") != FIXED_UPDATES:
                raise ProtocolError(f"wrong terminal step for {condition} seed {seed}")
            if not (directory / "final" / "adapter_model.safetensors").is_file():
                raise ProtocolError(f"missing final adapter for {condition} seed {seed}")


def completed_run_artifact_manifest() -> dict[str, Any]:
    """Fingerprint every run artifact whose identity is frozen at unlock."""
    state = load_and_verify_state()
    runs: dict[str, Any] = {}
    for condition in REQUIRED_CONDITIONS:
        for seed in SEEDS:
            directory = run_dir(condition, seed)
            label = f"{condition}_seed{seed}"
            audit_files = {
                name: directory / name
                for name in (
                    "run_manifest.json",
                    "resolved_config.json",
                    "data_indices.json",
                    "validation_history.jsonl",
                    "log_history.json",
                )
            }
            audit_files["checkpoint-25/trainer_state.json"] = (
                directory / "checkpoint-25" / "trainer_state.json"
            )
            missing = [name for name, path in audit_files.items() if not path.is_file()]
            if missing:
                raise ProtocolError(f"missing frozen run artifacts for {label}: {missing}")
            runs[label] = {
                "audit_file_sha256": {
                    name: sha256_file(path) for name, path in sorted(audit_files.items())
                },
                "final_adapter": _adapter_identity(directory / "final"),
                "training_behavior": training_behavior_summary(
                    directory / "log_history.json"
                ),
            }
    return {
        "protocol": state["protocol"],
        "git_commit": state["git_commit"],
        "runs": runs,
    }


def unlock_final() -> None:
    verify_completed_runs()
    gate = compute_curve_gate(write_result=True)
    if not gate["passed"]:
        raise ProtocolError(
            "the one predeclared curve gate failed; final evaluation remains sealed"
        )
    state = load_and_verify_state()
    if COMPLETED_RUNS_PATH.exists() or UNLOCK_PATH.exists():
        raise ProtocolError("refusing to overwrite an existing unlock chain")
    completed_runs = completed_run_artifact_manifest()
    write_json(COMPLETED_RUNS_PATH, completed_runs)
    marker = {
        "protocol": state["protocol"],
        "git_commit": state["git_commit"],
        "curve_gate_sha256": sha256_file(CURVE_GATE_PATH),
        "completed_runs_sha256": sha256_file(COMPLETED_RUNS_PATH),
        "unlocked_at_utc": utc_now(),
        "reason": "all 12 matched semantic/sign-flip runs completed and the predeclared six-seed mean curve gate passed",
    }
    write_json(UNLOCK_PATH, marker)
    print(json.dumps(marker, indent=2, sort_keys=True))


def verify_unlock() -> None:
    state = load_and_verify_state()
    if not UNLOCK_PATH.is_file():
        raise ProtocolError("sealed final evaluation is not unlocked")
    marker = json.loads(UNLOCK_PATH.read_text())
    if marker.get("git_commit") != state["git_commit"]:
        raise ProtocolError("unlock marker belongs to another commit")
    if (
        not COMPLETED_RUNS_PATH.is_file()
        or marker.get("completed_runs_sha256") != sha256_file(COMPLETED_RUNS_PATH)
        or json.loads(COMPLETED_RUNS_PATH.read_text())
        != completed_run_artifact_manifest()
    ):
        raise ProtocolError("a run artifact changed after sealed-evaluation unlock")
    gate = compute_curve_gate(write_result=False)
    stored_gate = json.loads(CURVE_GATE_PATH.read_text())
    gate_fields = (
        "criterion", "predeclared_steps", "n_seeds", "examples_per_seed",
        "per_seed_exact_match", "mean_exact_match", "passed",
    )
    if any(gate.get(field) != stored_gate.get(field) for field in gate_fields):
        raise ProtocolError("stored curve gate no longer matches the run histories")
    plot = stored_gate.get("curve_plot", {})
    if not CURVE_PLOT_PATH.is_file() or plot.get("sha256") != sha256_file(CURVE_PLOT_PATH):
        raise ProtocolError("curve figure is missing or changed")
    if not gate["passed"] or marker.get("curve_gate_sha256") != sha256_file(CURVE_GATE_PATH):
        raise ProtocolError("curve gate or unlock marker changed")


def _evaluation_role(label: str) -> tuple[str, int, bool]:
    if label == "base":
        return "jlens", SEEDS[0], True
    match = re.fullmatch(r"(jlens|signflip|gsm8k)_seed(\d+)", label)
    if match is None:
        raise ProtocolError(f"invalid evaluation label: {label!r}")
    condition, seed_text = match.groups()
    seed = int(seed_text)
    if seed not in CONFIG_SEEDS[condition]:
        raise ProtocolError(f"evaluation label has an unregistered seed: {label!r}")
    return condition, seed, False


def _adapter_identity(path: Path) -> dict[str, Any]:
    files = sorted(
        {
            *path.glob("adapter_config.json"),
            *path.glob("adapter_model*"),
        }
    )
    if not files:
        raise ProtocolError(f"no adapter model files found under {path}")
    hashes = {file.name: sha256_file(file) for file in files if file.is_file()}
    try:
        recorded_path = path.resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError:
        recorded_path = str(path.resolve())
    return {
        "path": recorded_path,
        "sha256": canonical_sha256(hashes),
        "files": hashes,
    }


def _contains_forbidden_gold_key(value: Any) -> bool:
    forbidden = {"answer", "gold", "gold_answer", "reference", "reference_answer"}
    if isinstance(value, dict):
        return any(
            str(key).lower() in forbidden or _contains_forbidden_gold_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_gold_key(item) for item in value)
    return False


@functools.lru_cache(maxsize=1)
def _sealed_evaluation_reference() -> dict[str, Any]:
    """Reconstruct the sealed prompts and labels from the pinned dataset."""
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer

        from jlens_rl.common import extract_answer, format_prompt, gsm8k_reward
    except ImportError as error:
        raise ProtocolError(
            "sealed evaluation verification requires the project environment"
        ) from error

    dataset = load_dataset(
        "openai/gsm8k",
        "main",
        split="train",
        revision=DATASET_REVISION,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        revision=MODEL_REVISION,
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    references: dict[int, dict[str, Any]] = {}
    for source_index in load_indices(MANIFEST_DIR / "sealed_final_indices.json"):
        row = dataset[source_index]
        prompt = format_prompt(tokenizer, row["question"])
        prompt_token_ids = tokenizer(
            prompt,
            truncation=True,
            max_length=384,
        )["input_ids"]
        references[source_index] = {
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "prompt_token_ids_sha256": canonical_sha256(prompt_token_ids),
            "answer": row["answer"],
        }
    return {
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "references": references,
        "extract_answer": extract_answer,
        "gsm8k_reward": gsm8k_reward,
        "decode_completion": lambda token_ids: tokenizer.decode(
            token_ids, skip_special_tokens=True
        ),
    }


def verify_evaluation_jsonl(path: Path, label: str) -> None:
    state = load_and_verify_state()
    condition, seed, is_base = _evaluation_role(label)
    expected_path = STATE_DIR / "evals" / f"{label}.jsonl"
    if path.resolve() != expected_path.resolve():
        raise ProtocolError(
            f"evaluation label {label!r} must use {expected_path}, not {path}"
        )
    expected_indices = load_indices(MANIFEST_DIR / "sealed_final_indices.json")
    sealed_manifest_path = MANIFEST_DIR / "sealed_final_indices.json"
    expected_generation = {
        "do_sample": False,
        "max_prompt_tokens": 384,
        "max_new_tokens": 256,
        "padding_side": "left",
    }
    expected_dataset = {
        "name": "openai/gsm8k",
        "subset": "main",
        "split": "train",
        "revision": DATASET_REVISION,
    }
    sealed_reference = _sealed_evaluation_reference()
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ProtocolError(f"non-object evaluation row at {path}:{line_number}")
        records.append(record)
    if len(records) != len(expected_indices):
        raise ProtocolError(
            f"{path} has {len(records)} rows; expected {len(expected_indices)}"
        )
    if [record.get("source_index") for record in records] != expected_indices:
        raise ProtocolError(f"{path} does not contain the sealed indices in frozen order")
    for line_number, record in enumerate(records, 1):
        if record.get("schema_version") != 1:
            raise ProtocolError(f"invalid schema version at {path}:{line_number}")
        if not isinstance(record.get("correct"), bool):
            raise ProtocolError(f"invalid correctness at {path}:{line_number}")
        prompt_hash = record.get("prompt_sha256")
        if not isinstance(prompt_hash, str) or len(prompt_hash) != 64:
            raise ProtocolError(f"invalid prompt hash at {path}:{line_number}")
        token_hash = record.get("prompt_token_ids_sha256")
        if not isinstance(token_hash, str) or len(token_hash) != 64:
            raise ProtocolError(f"invalid prompt-token hash at {path}:{line_number}")
        if _contains_forbidden_gold_key(record):
            raise ProtocolError(f"gold answer leaked into auditable output at {path}:{line_number}")
        if record.get("target_words") != ["solved"]:
            raise ProtocolError(f"wrong target words at {path}:{line_number}")
        if record.get("generation") != expected_generation:
            raise ProtocolError(f"wrong generation settings at {path}:{line_number}")
        dataset = record.get("dataset")
        if not isinstance(dataset, dict) or any(
            dataset.get(key) != value for key, value in expected_dataset.items()
        ):
            raise ProtocolError(f"wrong dataset provenance at {path}:{line_number}")
        if dataset.get("fingerprint") != sealed_reference["dataset_fingerprint"]:
            raise ProtocolError(f"wrong dataset fingerprint at {path}:{line_number}")
        completion = record.get("completion")
        if not isinstance(completion, str):
            raise ProtocolError(f"missing completion at {path}:{line_number}")
        completion_token_ids = record.get("completion_token_ids")
        if (
            not isinstance(completion_token_ids, list)
            or any(
                isinstance(token, bool) or not isinstance(token, int)
                for token in completion_token_ids
            )
            or sealed_reference["decode_completion"](completion_token_ids)
            != completion
        ):
            raise ProtocolError(f"completion token identity mismatch at {path}:{line_number}")
        reference = sealed_reference["references"][record["source_index"]]
        if (
            prompt_hash != reference["prompt_sha256"]
            or token_hash != reference["prompt_token_ids_sha256"]
        ):
            raise ProtocolError(f"prompt identity mismatch at {path}:{line_number}")
        expected_prediction = sealed_reference["extract_answer"](completion)
        expected_correct = bool(
            sealed_reference["gsm8k_reward"](completion, reference["answer"])
        )
        if (
            record.get("prediction") != expected_prediction
            or record.get("correct") is not expected_correct
        ):
            raise ProtocolError(f"incorrect derived outcome at {path}:{line_number}")
        expected_matches = (
            ["solved"]
            if re.search(r"(?<!\w)solved(?!\w)", completion, flags=re.IGNORECASE)
            else []
        )
        if (
            record.get("literal_target_matches") != expected_matches
            or record.get("literal_target_used") is not bool(expected_matches)
        ):
            raise ProtocolError(f"invalid literal-target audit at {path}:{line_number}")
        completion_tokens = record.get("completion_tokens")
        if (
            isinstance(completion_tokens, bool)
            or not isinstance(completion_tokens, int)
            or completion_tokens < 0
            or completion_tokens > expected_generation["max_new_tokens"]
            or completion_tokens != len(completion_token_ids)
        ):
            raise ProtocolError(f"invalid completion length at {path}:{line_number}")
    provenance = records[0].get("provenance")
    if not isinstance(provenance, dict) or any(
        record.get("provenance") != provenance for record in records
    ):
        raise ProtocolError(f"{path} does not have constant provenance")
    git_provenance = provenance.get("git", {})
    if (
        git_provenance.get("git_commit") != state["git_commit"]
        or git_provenance.get("git_dirty") is not False
        or not isinstance(git_provenance.get("source_tree_sha256"), str)
        or len(git_provenance["source_tree_sha256"]) != 64
    ):
        raise ProtocolError(f"{path} was evaluated from different or dirty source")
    model = provenance.get("model", {})
    if (
        model.get("name") != "Qwen/Qwen2.5-0.5B-Instruct"
        or model.get("configured_revision") != MODEL_REVISION
        or model.get("resolved_revision") != MODEL_REVISION
        or model.get("dtype") != "torch.bfloat16"
    ):
        raise ProtocolError(f"{path} used the wrong model revision")
    if provenance.get("run_label") != label or provenance.get("evaluation_seed") != 0:
        raise ProtocolError(f"{path} is not bound to evaluation role {label!r}")

    adapter = provenance.get("adapter")
    if is_base:
        if adapter is not None:
            raise ProtocolError("base evaluation unexpectedly used an adapter")
    else:
        expected_adapter = _adapter_identity(run_dir(condition, seed) / "final")
        if adapter != expected_adapter:
            raise ProtocolError(f"{path} used the wrong adapter for {label}")

    eval_config_path = REPO / "configs" / "confirmatory_sealed_eval.json"
    eval_config = load_config(eval_config_path)
    eval_identity = provenance.get("evaluation_config", {})
    if (
        eval_identity.get("file_sha256") != sha256_file(eval_config_path)
        or eval_identity.get("resolved_sha256") != canonical_sha256(eval_config)
    ):
        raise ProtocolError(f"{path} used the wrong evaluation config")

    experiment_config_path = config_path(condition, seed)
    experiment_config = load_config(experiment_config_path)
    experiment_identity = provenance.get("experiment_config", {})
    if (
        experiment_identity.get("file_sha256") != sha256_file(experiment_config_path)
        or experiment_identity.get("resolved_sha256")
        != canonical_sha256(experiment_config)
        or experiment_identity.get("source") != "explicit"
    ):
        raise ProtocolError(f"{path} used the wrong experiment config")

    selection = provenance.get("selection", {})
    manifest_identity = selection.get("index_manifest", {})
    if (
        selection.get("method") != "index_manifest"
        or selection.get("indices_sha256") != canonical_sha256(expected_indices)
        or manifest_identity.get("sha256") != sha256_file(sealed_manifest_path)
        or manifest_identity.get("dataset") != "openai/gsm8k"
        or manifest_identity.get("subset") != "main"
        or manifest_identity.get("split") != "train"
        or manifest_identity.get("count") != len(expected_indices)
    ):
        raise ProtocolError(f"{path} used the wrong sealed index manifest")

    experiment = provenance.get("experiment", {})
    common = load_config(REPO / "configs" / "confirmatory_common.json")
    if (
        experiment.get("training_seed") != seed
        or experiment.get("reward_type")
        != ("gsm8k" if condition == "gsm8k" else "jlens")
        or experiment.get("target_words") != ["solved"]
        or experiment.get("score_components") != experiment_config["score_components"]
    ):
        raise ProtocolError(f"{path} has the wrong experiment identity")
    if experiment.get("lens_sha256") != common["lens_sha256"]:
        raise ProtocolError(f"{path} used the wrong lens")
    if experiment.get("calibration_sha256") != common["calibration_sha256"]:
        raise ProtocolError(f"{path} used the wrong calibration")

    expected_software = {
        "j-lens-rl": "0.1.0",
        "torch": "2.9.1",
        "transformers": "5.5.0",
        "datasets": "4.7.0",
        "peft": "0.18.0",
    }
    if provenance.get("software") != expected_software:
        raise ProtocolError(f"{path} used an unexpected software environment")
    runtime = provenance.get("runtime", {})
    if (
        "L40S" not in str(runtime.get("cuda_device_name", ""))
        or not isinstance(runtime.get("cuda_version"), str)
        or not runtime["cuda_version"]
        or runtime.get("batch_size") != 64
    ):
        raise ProtocolError(f"{path} used the wrong evaluation runtime")


def _recompute_final_comparisons() -> tuple[dict[str, Any], dict[str, Any]]:
    from jlens_rl.paired_eval import (
        compare_multiple_adapters,
        difference_in_differences,
        read_jsonl,
    )

    labels = [
        "base",
        *(f"jlens_seed{seed}" for seed in SEEDS),
        *(f"signflip_seed{seed}" for seed in SEEDS),
    ]
    for label in labels:
        verify_evaluation_jsonl(STATE_DIR / "evals" / f"{label}.jsonl", label)

    base = read_jsonl(STATE_DIR / "evals" / "base.jsonl")
    semantic_sets = [
        read_jsonl(STATE_DIR / "evals" / f"jlens_seed{seed}.jsonl")
        for seed in SEEDS
    ]
    control_sets = [
        read_jsonl(STATE_DIR / "evals" / f"signflip_seed{seed}.jsonl")
        for seed in SEEDS
    ]
    semantic = compare_multiple_adapters(
        base,
        semantic_sets,
        bootstrap_samples=10_000,
        bootstrap_seed=0,
        confidence=0.95,
    )
    semantic["primary_estimand"] = semantic["comparison"]
    specificity = dict(semantic)
    specificity["primary_estimand"] = "difference_in_differences"
    specificity["difference_in_differences"] = difference_in_differences(
        base,
        semantic_sets,
        control_sets,
        bootstrap_samples=10_000,
        bootstrap_seed=0,
        confidence=0.95,
    )
    return semantic, specificity


def final_evaluation_hashes() -> dict[str, str]:
    labels = [
        "base",
        *(f"jlens_seed{seed}" for seed in SEEDS),
        *(f"signflip_seed{seed}" for seed in SEEDS),
    ]
    return {
        f"{label}.jsonl": sha256_file(STATE_DIR / "evals" / f"{label}.jsonl")
        for label in labels
    }


def final_report() -> dict[str, Any]:
    verify_unlock()
    semantic_path = STATE_DIR / "evidence" / "semantic_vs_base.json"
    specificity_path = STATE_DIR / "evidence" / "semantic_vs_signflip.json"
    if not semantic_path.is_file() or not specificity_path.is_file():
        raise ProtocolError("run final-treatment and final-controls before reporting")
    semantic = json.loads(semantic_path.read_text())
    specificity = json.loads(specificity_path.read_text())
    recomputed_semantic, recomputed_specificity = _recompute_final_comparisons()
    if semantic != recomputed_semantic:
        raise ProtocolError("semantic paired result does not match frozen evaluations")
    if specificity != recomputed_specificity:
        raise ProtocolError("specificity result does not match frozen evaluations")
    bootstrap = semantic.get("crossed_seed_item_bootstrap", {})
    sign_test = semantic.get("seed_sign_test", {})
    specificity_result = specificity.get("difference_in_differences", {})
    specificity_bootstrap = specificity_result.get("crossed_seed_item_bootstrap", {})
    checks = {
        "curve_gate_passed": bool(json.loads(CURVE_GATE_PATH.read_text()).get("passed")),
        "mean_accuracy_difference_positive": semantic.get("mean_accuracy_difference", 0) > 0,
        "crossed_ci_excludes_zero": bootstrap.get(
            "mean_accuracy_difference_ci_low", -math.inf
        ) > 0,
        "all_six_seed_effects_positive": (
            sign_test.get("positive") == 6
            and sign_test.get("negative") == 0
            and sign_test.get("tied_excluded") == 0
        ),
        "two_sided_seed_sign_p_below_0_05": sign_test.get(
            "exact_two_sided_p", 1.0
        ) < 0.05,
        "signflip_specificity_report_present": (
            specificity.get("primary_estimand") == "difference_in_differences"
        ),
        "signflip_specificity_mean_positive": (
            specificity_result.get("mean_difference_in_differences", 0) > 0
        ),
        "signflip_specificity_crossed_ci_excludes_zero": (
            specificity_bootstrap.get(
                "mean_difference_in_differences_ci_low", -math.inf
            )
            > 0
        ),
        "curve_figure_present_and_hashed": CURVE_PLOT_PATH.is_file(),
    }
    result = {
        "protocol": PROTOCOL,
        "criterion": "predeclared curve plus six positive semantic-vs-base seeds, crossed 95% CIs above zero for semantic-vs-base and semantic-vs-signflip, and exact two-sided seed sign p < 0.05",
        "checks": checks,
        "passed": all(checks.values()),
        "semantic_vs_base_sha256": sha256_file(semantic_path),
        "semantic_vs_signflip_sha256": sha256_file(specificity_path),
        "completed_runs_sha256": sha256_file(COMPLETED_RUNS_PATH),
        "evaluation_jsonl_sha256": final_evaluation_hashes(),
        "reported_at_utc": utc_now(),
    }
    if ACCEPTANCE_PATH.exists():
        raise ProtocolError(f"refusing to overwrite final report: {ACCEPTANCE_PATH}")
    write_json(ACCEPTANCE_PATH, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "prepare", "verify", "curve", "verify-runs", "unlock",
            "verify-semantic", "verify-unlock", "verify-eval", "report",
        ),
    )
    parser.add_argument("--path", type=Path)
    parser.add_argument("--label")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.command == "prepare":
            prepare()
        elif args.command == "verify":
            print(json.dumps(load_and_verify_state(), indent=2, sort_keys=True))
        elif args.command == "curve":
            verify_completed_runs(("jlens",))
            result = compute_curve_gate(write_result=True)
            print(json.dumps(result, indent=2, sort_keys=True))
            if not result["passed"]:
                raise ProtocolError("predeclared curve gate failed")
        elif args.command == "verify-runs":
            verify_completed_runs()
            print("all 12 required fixed-horizon runs match the frozen protocol")
        elif args.command == "verify-semantic":
            verify_completed_runs(("jlens",))
            print("all six semantic fixed-horizon runs match the frozen protocol")
        elif args.command == "unlock":
            unlock_final()
        elif args.command == "verify-unlock":
            verify_unlock()
            print("sealed final evaluation is unlocked and provenance is intact")
        elif args.command == "verify-eval":
            if args.path is None or args.label is None:
                raise ProtocolError("verify-eval requires --path and --label")
            verify_evaluation_jsonl(args.path, args.label)
            print(f"evaluation JSONL is complete and auditable: {args.path}")
        elif args.command == "report":
            result = final_report()
            print(json.dumps(result, indent=2, sort_keys=True))
            if not result["passed"]:
                raise ProtocolError("predeclared significant-evidence criterion failed")
    except (OSError, KeyError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError, ProtocolError) as error:
        print(f"protocol error: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
