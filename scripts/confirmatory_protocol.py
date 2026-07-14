#!/usr/bin/env python3
"""Prepare and guard the predeclared J-lens confirmatory experiment.

This script deliberately does not choose hyperparameters or checkpoints from
accuracy.  It creates source-index manifests from historically unused GSM8K
training examples, fingerprints the committed protocol and artifacts, verifies
completed matched runs, and evaluates the one predeclared curve gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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
UNLOCK_PATH = STATE_DIR / "final_unlocked.json"
ACCEPTANCE_PATH = STATE_DIR / "evidence" / "acceptance.json"

MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
ALLOCATION_SALT = "j-lens-rl-confirmatory-v1-2026-07-14"
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
    "exploratory_dev_indices.json": 200,
    "curve_indices.json": 400,
    "sealed_final_indices.json": 3000,
    "future_reserve_indices.json": 463,
}

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
        "validation_examples": 3000,
        "min_new_tokens": 0,
    }
    for key, expected in expected_eval.items():
        if eval_cfg.get(key) != expected:
            raise ProtocolError(f"sealed eval {key!r} is not frozen to {expected!r}")

    if require_manifests:
        if len(load_indices(MANIFEST_DIR / "curve_indices.json")) != 400:
            raise ProtocolError("curve manifest must contain exactly 400 indices")
        if len(load_indices(MANIFEST_DIR / "sealed_final_indices.json")) != 3000:
            raise ProtocolError("sealed-final manifest must contain exactly 3000 indices")

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


def reconstruct_historical_indices() -> tuple[list[int], int]:
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
    for seed, count in ((42, 1150), (43, 1000)):
        historical.update(raw.shuffle(seed=seed).select(range(count))["_source_index"])
    clean_pool = raw.filter(lambda row: not 6800 <= row["_source_index"] < 7200)
    for seed in (42, 43):
        historical.update(
            clean_pool.shuffle(seed=seed).select(range(1000))["_source_index"]
        )
    return sorted(int(index) for index in historical), len(raw)


def allocation_key(index: int) -> bytes:
    return hashlib.sha256(f"{ALLOCATION_SALT}:{index}".encode()).digest()


def prepare() -> None:
    commit = require_clean_worktree()
    if STATE_DIR.exists():
        raise ProtocolError(
            f"{STATE_DIR} already exists; do not overwrite a prepared protocol"
        )
    config_hashes = validate_configs(require_manifests=False)
    artifact_hashes = validate_artifacts()
    historical, dataset_size = reconstruct_historical_indices()
    fresh = sorted(set(range(dataset_size)) - set(historical), key=allocation_key)
    if len(historical) != 3410 or len(fresh) != 4063:
        raise ProtocolError(
            f"historical/fresh counts changed: {len(historical)}/{len(fresh)}"
        )

    cursor = 0
    allocations: dict[str, list[int]] = {}
    for name, size in SPLIT_SIZES.items():
        allocations[name] = fresh[cursor : cursor + size]
        cursor += size
    if cursor != len(fresh):
        raise ProtocolError("predeclared split sizes do not exhaust the fresh pool")

    MANIFEST_DIR.mkdir(parents=True)
    write_json(MANIFEST_DIR / "historical_exclusions.json", manifest_payload(historical))
    for name, indices in allocations.items():
        write_json(MANIFEST_DIR / name, manifest_payload(indices))
    # All historically fresh indices stay out of every confirmatory training run.
    write_json(MANIFEST_DIR / "train_exclusions.json", manifest_payload(fresh))

    manifest_hashes = {
        str(path.relative_to(REPO)): sha256_file(path)
        for path in sorted(MANIFEST_DIR.glob("*.json"))
    }
    state = {
        "protocol": "j-lens-rl-confirmatory-v1",
        "prepared_at_utc": utc_now(),
        "git_commit": commit,
        "allocation_algorithm": "ascending SHA-256(salt + ':' + raw_source_index)",
        "allocation_salt": ALLOCATION_SALT,
        "dataset": "openai/gsm8k:main",
        "dataset_revision": DATASET_REVISION,
        "dataset_train_rows": dataset_size,
        "historically_used_count": len(historical),
        "historically_unused_count": len(fresh),
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
    if sum(map(len, split_sets.values())) != 4063:
        raise ProtocolError("fresh manifests no longer contain exactly 4,063 indices")
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
        rows[step] = row
    if tuple(sorted(rows)) != ALL_VALIDATION_STEPS:
        raise ProtocolError(
            f"{path} has steps {sorted(rows)}, expected {list(ALL_VALIDATION_STEPS)}"
        )
    return rows


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


def verify_completed_runs() -> None:
    state = load_and_verify_state()
    curve_indices = load_indices(MANIFEST_DIR / "curve_indices.json")
    excluded = set(load_indices(MANIFEST_DIR / "train_exclusions.json"))
    matched_train_indices: dict[int, list[int]] = {}

    for condition in REQUIRED_CONDITIONS:
        for seed in SEEDS:
            directory = run_dir(condition, seed)
            expected_cfg = load_config(config_path(condition, seed))
            resolved_path = directory / "resolved_config.json"
            if not resolved_path.is_file() or json.loads(resolved_path.read_text()) != expected_cfg:
                raise ProtocolError(f"resolved config mismatch for {condition} seed {seed}")
            manifest = json.loads((directory / "run_manifest.json").read_text())
            if manifest.get("git_commit") != state["git_commit"] or manifest.get("git_dirty"):
                raise ProtocolError(f"invalid source provenance for {condition} seed {seed}")
            data = json.loads((directory / "data_indices.json").read_text())
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

            load_history(directory / "validation_history.jsonl")
            trainer_state_path = directory / f"checkpoint-{FIXED_UPDATES}" / "trainer_state.json"
            trainer_state = json.loads(trainer_state_path.read_text())
            if trainer_state.get("global_step") != FIXED_UPDATES:
                raise ProtocolError(f"wrong terminal step for {condition} seed {seed}")
            if not (directory / "final" / "adapter_model.safetensors").is_file():
                raise ProtocolError(f"missing final adapter for {condition} seed {seed}")


def unlock_final() -> None:
    verify_completed_runs()
    gate = compute_curve_gate(write_result=True)
    if not gate["passed"]:
        raise ProtocolError(
            "the one predeclared curve gate failed; final evaluation remains sealed"
        )
    state = load_and_verify_state()
    marker = {
        "protocol": state["protocol"],
        "git_commit": state["git_commit"],
        "curve_gate_sha256": sha256_file(CURVE_GATE_PATH),
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


def verify_evaluation_jsonl(path: Path) -> None:
    state = load_and_verify_state()
    expected_indices = load_indices(MANIFEST_DIR / "sealed_final_indices.json")
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
        if not isinstance(record.get("correct"), bool):
            raise ProtocolError(f"invalid correctness at {path}:{line_number}")
        prompt_hash = record.get("prompt_sha256")
        if not isinstance(prompt_hash, str) or len(prompt_hash) != 64:
            raise ProtocolError(f"invalid prompt hash at {path}:{line_number}")
        if any(key in record for key in ("answer", "gold", "reference")):
            raise ProtocolError(f"gold answer leaked into auditable output at {path}:{line_number}")
    provenance = records[0].get("provenance")
    if not isinstance(provenance, dict) or any(
        record.get("provenance") != provenance for record in records
    ):
        raise ProtocolError(f"{path} does not have constant provenance")
    git_provenance = provenance.get("git", {})
    if (
        git_provenance.get("git_commit") != state["git_commit"]
        or git_provenance.get("git_dirty")
    ):
        raise ProtocolError(f"{path} was evaluated from different or dirty source")
    model = provenance.get("model", {})
    if model.get("configured_revision") != MODEL_REVISION:
        raise ProtocolError(f"{path} used the wrong model revision")
    experiment = provenance.get("experiment", {})
    common = load_config(REPO / "configs" / "confirmatory_common.json")
    if experiment.get("lens_sha256") != common["lens_sha256"]:
        raise ProtocolError(f"{path} used the wrong lens")
    if experiment.get("calibration_sha256") != common["calibration_sha256"]:
        raise ProtocolError(f"{path} used the wrong calibration")


def final_report() -> dict[str, Any]:
    verify_unlock()
    semantic_path = STATE_DIR / "evidence" / "semantic_vs_base.json"
    specificity_path = STATE_DIR / "evidence" / "semantic_vs_signflip.json"
    if not semantic_path.is_file() or not specificity_path.is_file():
        raise ProtocolError("run final-treatment and final-controls before reporting")
    semantic = json.loads(semantic_path.read_text())
    specificity = json.loads(specificity_path.read_text())
    bootstrap = semantic.get("crossed_seed_item_bootstrap", {})
    sign_test = semantic.get("seed_sign_test", {})
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
        "curve_figure_present_and_hashed": CURVE_PLOT_PATH.is_file(),
    }
    result = {
        "protocol": "j-lens-rl-confirmatory-v1",
        "criterion": "predeclared curve plus six positive seeds, crossed 95% CI above zero, and exact two-sided seed sign p < 0.05",
        "checks": checks,
        "passed": all(checks.values()),
        "semantic_vs_base_sha256": sha256_file(semantic_path),
        "semantic_vs_signflip_sha256": sha256_file(specificity_path),
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
            "verify-unlock", "verify-eval", "report",
        ),
    )
    parser.add_argument("--path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.command == "prepare":
            prepare()
        elif args.command == "verify":
            print(json.dumps(load_and_verify_state(), indent=2, sort_keys=True))
        elif args.command == "curve":
            result = compute_curve_gate(write_result=True)
            print(json.dumps(result, indent=2, sort_keys=True))
            if not result["passed"]:
                raise ProtocolError("predeclared curve gate failed")
        elif args.command == "verify-runs":
            verify_completed_runs()
            print("all 12 required fixed-horizon runs match the frozen protocol")
        elif args.command == "unlock":
            unlock_final()
        elif args.command == "verify-unlock":
            verify_unlock()
            print("sealed final evaluation is unlocked and provenance is intact")
        elif args.command == "verify-eval":
            if args.path is None:
                raise ProtocolError("verify-eval requires --path")
            verify_evaluation_jsonl(args.path)
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
