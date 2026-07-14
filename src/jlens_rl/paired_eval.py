from __future__ import annotations

import argparse
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


SCHEMA_VERSION = 1
DATASET_NAME = "openai/gsm8k"
DATASET_SUBSET = "main"


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def literal_target_matches(text: str, target_words: Sequence[str]) -> list[str]:
    """Return configured target words present as case-insensitive whole words."""
    import re

    matches: list[str] = []
    for word in target_words:
        if not word:
            raise ValueError("target words must not be empty")
        pattern = rf"(?<!\w){re.escape(word)}(?!\w)"
        if re.search(pattern, text, flags=re.IGNORECASE):
            matches.append(word)
    return matches


def load_index_manifest(
    path: str | Path,
    *,
    expected_split: str,
    dataset_size: int,
    expected_count: int | None = None,
) -> tuple[list[int], dict[str, Any]]:
    """Load and validate a sealed set of dataset source indices.

    The preferred format is an object with ``dataset``, ``subset``, ``split``,
    and ``indices``. A bare list is accepted for small local experiments, but
    the expected split still comes from the evaluation config.
    """
    manifest_path = Path(path)
    raw = json.loads(manifest_path.read_text())
    if isinstance(raw, list):
        indices = raw
        dataset = DATASET_NAME
        subset = DATASET_SUBSET
        split = expected_split
    elif isinstance(raw, dict):
        try:
            indices = raw["indices"]
        except KeyError as error:
            raise ValueError("index manifest must contain an 'indices' list") from error
        dataset = raw.get("dataset", raw.get("dataset_name", DATASET_NAME))
        subset = raw.get("subset", raw.get("dataset_subset", DATASET_SUBSET))
        split = raw.get("split", expected_split)
    else:
        raise ValueError("index manifest must be a JSON object or list")

    if dataset != DATASET_NAME or subset != DATASET_SUBSET:
        raise ValueError(
            f"index manifest identifies {dataset!r}/{subset!r}, expected "
            f"{DATASET_NAME!r}/{DATASET_SUBSET!r}"
        )
    if split != expected_split:
        raise ValueError(
            f"index manifest split {split!r} does not match evaluation split "
            f"{expected_split!r}"
        )
    if not isinstance(indices, list):
        raise ValueError("index manifest 'indices' must be a list")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
        raise ValueError("every source index must be an integer")
    if len(indices) != len(set(indices)):
        raise ValueError("index manifest contains duplicate source indices")
    if any(index < 0 or index >= dataset_size for index in indices):
        raise ValueError("index manifest contains an out-of-range source index")
    if expected_count is not None and len(indices) != expected_count:
        raise ValueError(
            f"index manifest has {len(indices)} entries; expected {expected_count}"
        )
    if not indices:
        raise ValueError("index manifest must contain at least one source index")

    identity = {
        "path": str(manifest_path.resolve()),
        "sha256": file_sha256(manifest_path),
        "dataset": dataset,
        "subset": subset,
        "split": split,
        "count": len(indices),
    }
    return indices, identity


def exact_mcnemar_p(base_only_correct: int, adapter_only_correct: int) -> float:
    """Two-sided exact McNemar p-value using the conditional binomial test."""
    if base_only_correct < 0 or adapter_only_correct < 0:
        raise ValueError("discordant counts must be non-negative")
    discordant = base_only_correct + adapter_only_correct
    if discordant == 0:
        return 1.0
    tail = min(base_only_correct, adapter_only_correct)
    numerator = 2 * sum(math.comb(discordant, k) for k in range(tail + 1))
    denominator = 1 << discordant
    return float(min(Fraction(1, 1), Fraction(numerator, denominator)))


def paired_bootstrap_ci(
    base_correct: Sequence[bool | int | float],
    adapter_correct: Sequence[bool | int | float],
    *,
    samples: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile CI for adapter-minus-base accuracy with paired resampling."""
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    base = np.asarray(base_correct, dtype=np.float64)
    adapter = np.asarray(adapter_correct, dtype=np.float64)
    if base.ndim != 1 or adapter.ndim != 1 or len(base) != len(adapter):
        raise ValueError("paired correctness arrays must be one-dimensional and equal length")
    if len(base) == 0:
        raise ValueError("paired correctness arrays must not be empty")
    if not (np.isin(base, [0.0, 1.0]).all() and np.isin(adapter, [0.0, 1.0]).all()):
        raise ValueError("correctness values must be boolean or zero/one")

    differences = adapter - base
    generator = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    # Chunking bounds peak memory even when evaluating a much larger dataset.
    chunk_size = max(1, min(samples, 1_000_000 // len(differences)))
    for start in range(0, samples, chunk_size):
        stop = min(start + chunk_size, samples)
        sampled_indices = generator.integers(
            0, len(differences), size=(stop - start, len(differences))
        )
        estimates[start:stop] = differences[sampled_indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, [alpha, 1.0 - alpha], method="linear")
    return float(low), float(high)


def paired_statistics(
    base_correct: Sequence[bool | int | float],
    adapter_correct: Sequence[bool | int | float],
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, Any]:
    base_raw = np.asarray(base_correct)
    adapter_raw = np.asarray(adapter_correct)
    base = base_raw.astype(np.float64)
    adapter = adapter_raw.astype(np.float64)
    if base.ndim != 1 or adapter.ndim != 1 or len(base) != len(adapter):
        raise ValueError("paired correctness arrays must be one-dimensional and equal length")
    if len(base) == 0:
        raise ValueError("paired correctness arrays must not be empty")
    if not (np.isin(base, [0, 1]).all() and np.isin(adapter, [0, 1]).all()):
        raise ValueError("correctness values must be boolean or zero/one")
    base = base.astype(np.int8)
    adapter = adapter.astype(np.int8)

    both_correct = int(np.sum((base == 1) & (adapter == 1)))
    both_wrong = int(np.sum((base == 0) & (adapter == 0)))
    base_only = int(np.sum((base == 1) & (adapter == 0)))
    adapter_only = int(np.sum((base == 0) & (adapter == 1)))
    ci_low, ci_high = paired_bootstrap_ci(
        base,
        adapter,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
        confidence=confidence,
    )
    base_successes = int(base.sum())
    adapter_successes = int(adapter.sum())
    difference = float(adapter.mean() - base.mean())
    p_value = exact_mcnemar_p(base_only, adapter_only)
    alpha = 1.0 - confidence
    return {
        "n": len(base),
        "base": {"correct": base_successes, "accuracy": float(base.mean())},
        "adapter": {
            "correct": adapter_successes,
            "accuracy": float(adapter.mean()),
        },
        "accuracy_difference": difference,
        "paired_table": {
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "base_only_correct": base_only,
            "adapter_only_correct": adapter_only,
            "discordant": base_only + adapter_only,
        },
        "mcnemar_exact_two_sided_p": p_value,
        "paired_bootstrap": {
            "method": "paired_percentile",
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "confidence": confidence,
            "accuracy_difference_ci_low": ci_low,
            "accuracy_difference_ci_high": ci_high,
        },
        "significant_improvement": bool(
            difference > 0 and p_value < alpha and ci_low > 0
        ),
    }


def crossed_seed_item_bootstrap_ci(
    differences: np.ndarray,
    *,
    samples: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap a seed-by-item effect matrix along both crossed axes."""
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 2 or 0 in values.shape:
        raise ValueError("differences must be a non-empty seed-by-item matrix")
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")

    seed_count, item_count = values.shape
    generator = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    chunk_size = max(1, min(samples, 2_000_000 // values.size))
    for start in range(0, samples, chunk_size):
        stop = min(start + chunk_size, samples)
        batch = stop - start
        sampled_seeds = generator.integers(
            0, seed_count, size=(batch, seed_count)
        )
        sampled_items = generator.integers(
            0, item_count, size=(batch, item_count)
        )
        sampled = values[
            sampled_seeds[:, :, np.newaxis], sampled_items[:, np.newaxis, :]
        ]
        estimates[start:stop] = sampled.mean(axis=(1, 2))
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, [alpha, 1.0 - alpha], method="linear")
    return float(low), float(high)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on {path}:{line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"JSONL record on {path}:{line_number} is not an object")
            records.append(record)
    if not records:
        raise ValueError(f"evaluation file {path} contains no records")
    return records


def _pair_key(record: dict[str, Any]) -> tuple[str, str, str, int, str]:
    dataset = record.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("each evaluation record must contain dataset provenance")
    try:
        return (
            str(dataset["name"]),
            str(dataset["subset"]),
            str(dataset["split"]),
            int(record["source_index"]),
            str(record["prompt_sha256"]),
        )
    except KeyError as error:
        raise ValueError(f"evaluation record is missing {error.args[0]!r}") from error


def _index_records(records: Iterable[dict[str, Any]], label: str) -> dict[Any, dict[str, Any]]:
    indexed: dict[Any, dict[str, Any]] = {}
    for record in records:
        key = _pair_key(record)
        if key in indexed:
            raise ValueError(f"{label} evaluation contains duplicate item {key}")
        if record.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"{label} evaluation has unsupported schema version "
                f"{record.get('schema_version')!r}"
            )
        if not isinstance(record.get("correct"), bool):
            raise ValueError(f"{label} evaluation correctness must be boolean")
        indexed[key] = record
    return indexed


def _constant_field(records: Sequence[dict[str, Any]], field: str, label: str) -> Any:
    values = {canonical_json_sha256(record.get(field)) for record in records}
    if len(values) != 1:
        raise ValueError(f"{label} evaluation has inconsistent {field}")
    return records[0].get(field)


def _manifest_identity(provenance: dict[str, Any]) -> dict[str, Any] | None:
    manifest = provenance.get("selection", {}).get("index_manifest")
    if manifest is None:
        return None
    if not isinstance(manifest, dict):
        raise ValueError("index manifest provenance must be an object")
    return {key: value for key, value in manifest.items() if key != "path"}


def compare_evaluation_records(
    base_records: Sequence[dict[str, Any]],
    adapter_records: Sequence[dict[str, Any]],
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, Any]:
    base_by_key = _index_records(base_records, "base")
    adapter_by_key = _index_records(adapter_records, "adapter")
    if base_by_key.keys() != adapter_by_key.keys():
        missing_adapter = len(base_by_key.keys() - adapter_by_key.keys())
        missing_base = len(adapter_by_key.keys() - base_by_key.keys())
        raise ValueError(
            "base and adapter evaluations do not contain the same items "
            f"({missing_adapter} missing from adapter, {missing_base} missing from base)"
        )

    base_target_words = _constant_field(base_records, "target_words", "base")
    adapter_target_words = _constant_field(adapter_records, "target_words", "adapter")
    if base_target_words != adapter_target_words:
        raise ValueError("base and adapter evaluations used different target_words")
    base_generation = _constant_field(base_records, "generation", "base")
    adapter_generation = _constant_field(adapter_records, "generation", "adapter")
    if base_generation != adapter_generation:
        raise ValueError("base and adapter evaluations used different generation settings")
    base_provenance = _constant_field(base_records, "provenance", "base")
    adapter_provenance = _constant_field(adapter_records, "provenance", "adapter")
    if not isinstance(base_provenance, dict) or not isinstance(adapter_provenance, dict):
        raise ValueError("base and adapter records must contain provenance objects")
    if base_provenance.get("model") != adapter_provenance.get("model"):
        raise ValueError("base and adapter evaluations used different base model revisions")
    if base_provenance.get("evaluation_seed") != adapter_provenance.get(
        "evaluation_seed"
    ):
        raise ValueError("base and adapter evaluations used different evaluation seeds")
    base_manifest = _manifest_identity(base_provenance)
    adapter_manifest = _manifest_identity(adapter_provenance)
    if base_manifest != adapter_manifest:
        raise ValueError("base and adapter evaluations used different index manifests")

    ordered_keys = sorted(base_by_key)
    for key in ordered_keys:
        base_record = base_by_key[key]
        adapter_record = adapter_by_key[key]
        base_dataset = base_record["dataset"]
        adapter_dataset = adapter_record["dataset"]
        for field in ("revision", "fingerprint"):
            if base_dataset.get(field) != adapter_dataset.get(field):
                raise ValueError(f"dataset {field} differs for paired item {key}")

    statistics = paired_statistics(
        [base_by_key[key]["correct"] for key in ordered_keys],
        [adapter_by_key[key]["correct"] for key in ordered_keys],
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        confidence=confidence,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "comparison": "adapter_minus_base",
        "target_words": base_target_words,
        **statistics,
        "base_provenance": base_provenance,
        "adapter_provenance": adapter_provenance,
    }


def compare_multiple_adapters(
    base_records: Sequence[dict[str, Any]],
    adapter_record_sets: Sequence[Sequence[dict[str, Any]]],
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Compare multiple independently trained adapters to one frozen base."""
    if len(adapter_record_sets) < 2:
        raise ValueError("multi-adapter comparison requires at least two adapters")
    single_results = [
        compare_evaluation_records(
            base_records,
            records,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            confidence=confidence,
        )
        for records in adapter_record_sets
    ]
    provenance_hashes = [
        canonical_json_sha256(result["adapter_provenance"])
        for result in single_results
    ]
    if len(provenance_hashes) != len(set(provenance_hashes)):
        raise ValueError("multi-adapter comparison contains duplicate adapter provenance")
    training_seeds = [
        result["adapter_provenance"].get("experiment", {}).get("training_seed")
        for result in single_results
    ]
    if any(seed is None for seed in training_seeds):
        raise ValueError("multi-adapter provenance must record every training seed")
    if len(training_seeds) != len(set(training_seeds)):
        raise ValueError("multi-adapter comparison contains duplicate training seeds")

    base_by_key = _index_records(base_records, "base")
    ordered_keys = sorted(base_by_key)
    base = np.asarray(
        [base_by_key[key]["correct"] for key in ordered_keys], dtype=np.int8
    )
    adapter_rows: list[np.ndarray] = []
    for index, records in enumerate(adapter_record_sets):
        adapter_by_key = _index_records(records, f"adapter {index + 1}")
        adapter_rows.append(
            np.asarray(
                [adapter_by_key[key]["correct"] for key in ordered_keys],
                dtype=np.int8,
            )
        )
    adapters = np.stack(adapter_rows)
    differences = adapters - base[np.newaxis, :]
    mean_difference = float(differences.mean())
    ci_low, ci_high = crossed_seed_item_bootstrap_ci(
        differences,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
        confidence=confidence,
    )

    seed_effects = differences.mean(axis=1)
    positive = int(np.sum(seed_effects > 0))
    negative = int(np.sum(seed_effects < 0))
    tied = int(np.sum(seed_effects == 0))
    sign_p = exact_mcnemar_p(negative, positive)
    alpha = 1.0 - confidence
    per_seed: list[dict[str, Any]] = []
    for index, result in enumerate(single_results):
        per_seed.append(
            {
                "index": index,
                "adapter": result["adapter"],
                "accuracy_difference": result["accuracy_difference"],
                "paired_table": result["paired_table"],
                "mcnemar_exact_two_sided_p": result[
                    "mcnemar_exact_two_sided_p"
                ],
                "paired_bootstrap": result["paired_bootstrap"],
                "provenance": result["adapter_provenance"],
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "comparison": "mean_adapter_minus_base_across_seeds",
        "target_words": single_results[0]["target_words"],
        "n": len(base),
        "seed_count": len(adapter_record_sets),
        "base": {
            "correct": int(base.sum()),
            "accuracy": float(base.mean()),
        },
        "mean_adapter_accuracy": float(adapters.mean()),
        "mean_accuracy_difference": mean_difference,
        "crossed_seed_item_bootstrap": {
            "method": "crossed_seed_and_item_percentile",
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "confidence": confidence,
            "mean_accuracy_difference_ci_low": ci_low,
            "mean_accuracy_difference_ci_high": ci_high,
        },
        "seed_sign_test": {
            "positive": positive,
            "negative": negative,
            "tied_excluded": tied,
            "exact_two_sided_p": sign_p,
        },
        "significant_mean_improvement": bool(
            mean_difference > 0 and ci_low > 0
        ),
        "seed_significant_improvement": bool(
            mean_difference > 0 and sign_p < alpha
        ),
        "per_seed": per_seed,
        "base_provenance": single_results[0]["base_provenance"],
    }


def difference_in_differences(
    base_records: Sequence[dict[str, Any]],
    adapter_record_sets: Sequence[Sequence[dict[str, Any]]],
    control_record_sets: Sequence[Sequence[dict[str, Any]]],
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Estimate mean (semantic - base) - (control - base) across seeds/items."""
    if not adapter_record_sets or len(adapter_record_sets) != len(control_record_sets):
        raise ValueError("controls must be supplied one-for-one with adapters")

    # These comparisons enforce common prompts, manifests, model revisions,
    # generation settings, references, and target-word audits for every run.
    for records in [*adapter_record_sets, *control_record_sets]:
        compare_evaluation_records(
            base_records,
            records,
            bootstrap_samples=1,
            bootstrap_seed=bootstrap_seed,
            confidence=confidence,
        )

    adapter_provenance = [
        _constant_field(records, "provenance", "adapter")
        for records in adapter_record_sets
    ]
    control_provenance = [
        _constant_field(records, "provenance", "control")
        for records in control_record_sets
    ]
    identities = [
        canonical_json_sha256(provenance)
        for provenance in [*adapter_provenance, *control_provenance]
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("adapter/control comparison contains duplicate provenance")
    adapter_seeds = [
        provenance.get("experiment", {}).get("training_seed")
        for provenance in adapter_provenance
    ]
    control_seeds = [
        provenance.get("experiment", {}).get("training_seed")
        for provenance in control_provenance
    ]
    if adapter_seeds != control_seeds or any(seed is None for seed in adapter_seeds):
        raise ValueError(
            "adapters and controls must record matching training seeds in CLI order"
        )

    base_by_key = _index_records(base_records, "base")
    ordered_keys = sorted(base_by_key)

    def correctness_matrix(
        record_sets: Sequence[Sequence[dict[str, Any]]], label: str
    ) -> np.ndarray:
        rows: list[np.ndarray] = []
        for index, records in enumerate(record_sets):
            indexed = _index_records(records, f"{label} {index + 1}")
            rows.append(
                np.asarray(
                    [indexed[key]["correct"] for key in ordered_keys],
                    dtype=np.int8,
                )
            )
        return np.stack(rows)

    adapters = correctness_matrix(adapter_record_sets, "adapter")
    controls = correctness_matrix(control_record_sets, "control")
    effects = adapters - controls
    mean_effect = float(effects.mean())
    ci_low, ci_high = crossed_seed_item_bootstrap_ci(
        effects,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
        confidence=confidence,
    )
    seed_effects = effects.mean(axis=1)
    positive = int(np.sum(seed_effects > 0))
    negative = int(np.sum(seed_effects < 0))
    tied = int(np.sum(seed_effects == 0))
    sign_p = exact_mcnemar_p(negative, positive)
    alpha = 1.0 - confidence

    per_seed: list[dict[str, Any]] = []
    for index in range(len(adapter_record_sets)):
        stats = paired_statistics(
            controls[index],
            adapters[index],
            bootstrap_samples=1,
            bootstrap_seed=bootstrap_seed,
            confidence=confidence,
        )
        per_seed.append(
            {
                "index": index,
                "semantic": stats["adapter"],
                "control": stats["base"],
                "difference_in_differences": stats["accuracy_difference"],
                "paired_table_control_vs_semantic": stats["paired_table"],
                "mcnemar_exact_two_sided_p": stats[
                    "mcnemar_exact_two_sided_p"
                ],
                "semantic_provenance": adapter_provenance[index],
                "control_provenance": control_provenance[index],
            }
        )
    return {
        "estimand": "(semantic_adapter-base)-(matched_control-base)",
        "algebraically_equivalent_to": "semantic_adapter-matched_control",
        "seed_count": len(adapter_record_sets),
        "n": len(ordered_keys),
        "mean_semantic_accuracy": float(adapters.mean()),
        "mean_control_accuracy": float(controls.mean()),
        "mean_difference_in_differences": mean_effect,
        "crossed_seed_item_bootstrap": {
            "method": "crossed_seed_and_item_percentile",
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "confidence": confidence,
            "mean_difference_in_differences_ci_low": ci_low,
            "mean_difference_in_differences_ci_high": ci_high,
        },
        "seed_sign_test": {
            "positive": positive,
            "negative": negative,
            "tied_excluded": tied,
            "exact_two_sided_p": sign_p,
        },
        "significant_improvement": bool(mean_effect > 0 and ci_low > 0),
        "seed_significant_improvement": bool(
            mean_effect > 0 and sign_p < alpha
        ),
        "per_seed": per_seed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare paired, per-example base and adapter evaluations."
    )
    parser.add_argument("--base-jsonl", required=True)
    parser.add_argument("--adapter-jsonl", action="append", required=True)
    parser.add_argument(
        "--control-jsonl",
        action="append",
        help="Matched control JSONL; repeat once per adapter in the same seed order.",
    )
    parser.add_argument("--output")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter_paths = [Path(path).resolve() for path in args.adapter_jsonl]
    control_paths = [Path(path).resolve() for path in (args.control_jsonl or [])]
    all_paths = [Path(args.base_jsonl).resolve(), *adapter_paths, *control_paths]
    if len(all_paths) != len(set(all_paths)):
        raise ValueError("base, adapter, and control JSONL paths must be unique")
    if control_paths and len(control_paths) != len(adapter_paths):
        raise ValueError("pass exactly one --control-jsonl per --adapter-jsonl")
    base_records = read_jsonl(args.base_jsonl)
    adapter_record_sets = [read_jsonl(path) for path in adapter_paths]
    if len(adapter_record_sets) == 1:
        result = compare_evaluation_records(
            base_records,
            adapter_record_sets[0],
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.seed,
            confidence=args.confidence,
        )
    else:
        result = compare_multiple_adapters(
            base_records,
            adapter_record_sets,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.seed,
            confidence=args.confidence,
        )
    if control_paths:
        controls = [read_jsonl(path) for path in control_paths]
        result["primary_estimand"] = "difference_in_differences"
        result["difference_in_differences"] = difference_in_differences(
            base_records,
            adapter_record_sets,
            controls,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.seed,
            confidence=args.confidence,
        )
    else:
        result["primary_estimand"] = result["comparison"]
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite {output}; pass --overwrite")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(rendered)
        temporary.replace(output)
    print(rendered, end="")


if __name__ == "__main__":
    main()
