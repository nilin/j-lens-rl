from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import (
    binomial_ci95,
    extract_answer,
    format_prompt,
    gsm8k_reward,
    load_config,
    model_dtype,
    require_clean_repository_provenance,
    repository_provenance,
    resolve_repository_root,
    runtime_environment_snapshot,
    seed_everything,
)
from .paired_eval import (
    DATASET_NAME,
    DATASET_SUBSET,
    SCHEMA_VERSION,
    canonical_json_sha256,
    file_sha256,
    literal_target_matches,
    load_index_manifest,
    text_sha256,
)
from .reward import TargetJLReward


@torch.no_grad()
def evaluate(model: Any, tokenizer: Any, rows: Any, cfg: dict[str, Any],
             jreward: TargetJLReward | None, batch_size: int = 16, *,
             source_indices: list[int] | None = None,
             dataset_provenance: dict[str, Any] | None = None,
             provenance: dict[str, Any] | None = None,
             output_jsonl: str | Path | None = None,
             overwrite: bool = False) -> dict[str, Any]:
    if output_jsonl is not None:
        if source_indices is None or len(source_indices) != len(rows):
            raise ValueError("auditable output requires one source index per row")
        if dataset_provenance is None or provenance is None:
            raise ValueError("auditable output requires dataset and run provenance")
        output_path = Path(output_jsonl)
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"refusing to overwrite {output_path}; pass --overwrite"
            )
    else:
        output_path = None

    was_training = model.training
    model.eval()
    correct: list[float] = []
    jscores: list[float] = []
    lengths: list[int] = []
    literal_targets = 0
    records: list[dict[str, Any]] = []
    generation = {
        "do_sample": False,
        "max_prompt_tokens": int(cfg["max_prompt_tokens"]),
        "max_new_tokens": int(cfg["max_new_tokens"]),
        "padding_side": tokenizer.padding_side,
    }
    for start in range(0, len(rows), batch_size):
        batch = [rows[index] for index in range(start, min(start + batch_size, len(rows)))]
        prompts = [format_prompt(tokenizer, row["question"]) for row in batch]
        encoded = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=cfg["max_prompt_tokens"],
        ).to(model.device)
        prompt_width = encoded.input_ids.shape[1]
        seq = model.generate(
            **encoded, max_new_tokens=cfg["max_new_tokens"], do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        for index, row in enumerate(batch):
            item_index = start + index
            completion_ids = seq[index, prompt_width:]
            eos = (completion_ids == tokenizer.eos_token_id).nonzero()
            if eos.numel():
                completion_ids = completion_ids[: int(eos[0].item()) + 1]
            text = tokenizer.decode(completion_ids, skip_special_tokens=True)
            target_matches = literal_target_matches(text, cfg["target_words"])
            literal_targets += int(bool(target_matches))
            item_correct = gsm8k_reward(text, row["answer"])
            correct.append(item_correct)
            lengths.append(int(completion_ids.numel()))
            item_jscore: float | None = None
            if jreward is not None:
                prompt_ids = encoded.input_ids[index][encoded.attention_mask[index].bool()]
                unpadded = torch.cat([prompt_ids, completion_ids]).unsqueeze(0)
                mask = torch.ones_like(unpadded)
                out = model(
                    unpadded, attention_mask=mask, output_hidden_states=True,
                    use_cache=False,
                )
                item_jscore = jreward(
                    model, out.hidden_states, len(prompt_ids), mask[0],
                    input_ids=unpadded[0],
                )
                jscores.append(item_jscore)
            else:
                prompt_ids = encoded.input_ids[index][encoded.attention_mask[index].bool()]

            if output_path is not None:
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "dataset": dataset_provenance,
                    "source_index": source_indices[item_index],
                    "prompt_sha256": text_sha256(prompts[index]),
                    "prompt_token_ids_sha256": canonical_json_sha256(
                        prompt_ids.tolist()
                    ),
                    "completion": text,
                    "completion_token_ids": completion_ids.tolist(),
                    "prediction": extract_answer(text),
                    "correct": bool(item_correct),
                    "completion_tokens": int(completion_ids.numel()),
                    "target_words": list(cfg["target_words"]),
                    "literal_target_matches": target_matches,
                    "literal_target_used": bool(target_matches),
                    "generation": generation,
                    "provenance": provenance,
                }
                if item_jscore is not None:
                    record["jlens_reward"] = float(item_jscore)
                records.append(record)
    n = len(correct)
    p = float(np.mean(correct))
    ci_low, ci_high = binomial_ci95(round(sum(correct)), n)
    result = {
        "exact_match": p,
        "exact_match_ci95_low": ci_low,
        "exact_match_ci95_high": ci_high,
        "mean_length": float(np.mean(lengths)),
        "literal_target_completion_rate": literal_targets / n,
    }
    if jscores:
        result["jlens_reward"] = float(np.mean(jscores))
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        with temporary.open("w") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        temporary.replace(output_path)
        result["per_example_jsonl"] = str(output_path.resolve())
    model.train(was_training)
    return result


def _config_identity(path: str | Path, resolved: dict[str, Any]) -> dict[str, Any]:
    config_path = Path(path).resolve()
    return {
        "path": str(config_path),
        "file_sha256": file_sha256(config_path),
        "resolved_sha256": canonical_json_sha256(resolved),
    }


def _adapter_identity(path: str | Path) -> dict[str, Any]:
    adapter_path = Path(path).resolve()
    if adapter_path.is_file():
        files = [adapter_path]
    else:
        files = sorted(
            {
                *adapter_path.glob("adapter_config.json"),
                *adapter_path.glob("adapter_model*"),
            }
        )
    if not files:
        raise FileNotFoundError(f"no adapter model files found under {adapter_path}")
    hashes = {file.name: file_sha256(file) for file in files if file.is_file()}
    repository = resolve_repository_root(__file__)
    try:
        recorded_path = adapter_path.relative_to(repository).as_posix()
    except ValueError:
        recorded_path = str(adapter_path)
    return {
        "path": recorded_path,
        "sha256": canonical_json_sha256(hashes),
        "files": hashes,
    }


def _git_provenance() -> dict[str, Any]:
    repository = resolve_repository_root(__file__)
    return repository_provenance(repository)


def _software_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in ("j-lens-rl", "torch", "transformers", "datasets", "peft"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def _resolve_config_path(value: str | Path, relative_to: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return Path(relative_to).resolve().parent / path


def _experiment_config(
    *,
    evaluation_config: dict[str, Any],
    evaluation_config_path: str | Path,
    requested_path: str | None,
    adapter_path: str | None,
    auditable: bool,
) -> tuple[dict[str, Any], Path, str]:
    configured_path = requested_path or evaluation_config.get("experiment_config_path")
    if configured_path:
        path = _resolve_config_path(configured_path, evaluation_config_path)
        return load_config(path), path, "explicit"
    if adapter_path:
        adapter = Path(adapter_path).resolve()
        candidates = [
            adapter / "resolved_config.json",
            adapter.parent / "resolved_config.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                return load_config(candidate), candidate, "adapter_resolved_config"
        if auditable:
            raise ValueError(
                "auditable adapter evaluation requires --experiment-config when "
                "resolved_config.json cannot be found beside the adapter"
            )
    if auditable:
        raise ValueError(
            "auditable base evaluation requires --experiment-config so literal "
            "target words come from the training experiment"
        )
    path = Path(evaluation_config_path)
    return evaluation_config, path, "evaluation_config"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--adapter")
    p.add_argument(
        "--experiment-config",
        help=(
            "Training/resolved config that owns target_words and J-lens settings. "
            "Required for auditable base output; auto-discovered beside adapters."
        ),
    )
    p.add_argument("--indices-manifest")
    p.add_argument("--output-jsonl")
    p.add_argument("--run-label")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip-jlens-metric", action="store_true")
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()
    cfg = load_config(args.config)
    evaluation_seed = int(cfg.get("evaluation_seed", 0))
    seed_everything(evaluation_seed)
    experiment_cfg, experiment_config_path, experiment_config_source = _experiment_config(
        evaluation_config=cfg,
        evaluation_config_path=args.config,
        requested_path=args.experiment_config,
        adapter_path=args.adapter,
        auditable=args.output_jsonl is not None,
    )
    git_provenance = _git_provenance()
    if cfg.get("require_clean_repository", False) or experiment_cfg.get(
        "require_clean_repository", False
    ):
        require_clean_repository_provenance(git_provenance)
    environment_identity = None
    environment_path: Path | None = None
    environment_temporary: Path | None = None
    requested_output_path: Path | None = None
    working_output_path: Path | None = None
    if args.output_jsonl is not None:
        requested_output_path = Path(args.output_jsonl)
        working_output_path = requested_output_path.with_suffix(
            requested_output_path.suffix + ".pending"
        )
        environment_path = requested_output_path.with_suffix(".environment.json")
        if requested_output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"refusing to overwrite evaluation output: {requested_output_path}"
            )
        if (
            environment_path.exists()
            and requested_output_path.exists()
            and not args.overwrite
        ):
            raise FileExistsError(
                f"refusing to overwrite evaluation environment: {environment_path}"
            )
        environment = runtime_environment_snapshot()
        environment_path.parent.mkdir(parents=True, exist_ok=True)
        environment_temporary = environment_path.with_suffix(".environment.json.tmp")
        environment_temporary.write_text(
            json.dumps(environment, indent=2, sort_keys=True) + "\n"
        )
        environment_identity = {
            "path": str(environment_path.resolve()),
            "sha256": file_sha256(environment_temporary),
        }
    if experiment_cfg["model_name"] != cfg["model_name"]:
        raise ValueError("evaluation and experiment configs name different base models")
    if not experiment_cfg.get("target_words"):
        raise ValueError("experiment config must define non-empty target_words")
    runtime_cfg = dict(cfg)
    runtime_cfg["target_words"] = list(experiment_cfg["target_words"])

    model_revision = experiment_cfg.get("model_revision")
    dataset_revision = cfg.get("dataset_revision")
    if args.output_jsonl and not model_revision:
        raise ValueError("auditable output requires a pinned model_revision")
    if args.output_jsonl and not dataset_revision:
        raise ValueError("auditable output requires a pinned dataset_revision")
    if cfg.get("model_revision") not in (None, model_revision):
        raise ValueError("evaluation and experiment configs pin different model revisions")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name"], revision=model_revision
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], revision=model_revision, dtype=model_dtype(),
        device_map={"": "cuda:0"},
    )
    resolved_model_revision = getattr(model.config, "_commit_hash", None)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    expected_calibration_sha256 = experiment_cfg.get(
        "expected_calibration_sha256", experiment_cfg.get("calibration_sha256")
    )
    actual_calibration_sha256 = file_sha256(experiment_cfg["calibration_path"])
    if (
        expected_calibration_sha256 is not None
        and expected_calibration_sha256 != actual_calibration_sha256
    ):
        raise ValueError(
            "calibration artifact does not match expected_calibration_sha256: "
            f"{actual_calibration_sha256!r} != {expected_calibration_sha256!r}"
        )
    expected_lens_sha256 = experiment_cfg.get(
        "expected_lens_sha256", experiment_cfg.get("lens_sha256")
    )
    reward = None if args.skip_jlens_metric else TargetJLReward(
        experiment_cfg["lens_path"], experiment_cfg["calibration_path"], tokenizer,
        experiment_cfg["target_words"], experiment_cfg["score_stride"],
        experiment_cfg["mask_target_tokens"],
        experiment_cfg.get("vocab_chunk_size", 16384),
        experiment_cfg.get("score_start_fraction", 0.0),
        experiment_cfg.get("score_layers"),
        experiment_cfg.get("score_aggregation", "mean"),
        experiment_cfg.get("score_include_final", False),
        experiment_cfg.get("score_components"),
        experiment_cfg.get("score_end_fraction", 1.0),
        expected_model=experiment_cfg["model_name"],
        expected_model_revision=model_revision,
        expected_lens_sha256=expected_lens_sha256,
    )
    evaluation_source = cfg.get("evaluation_source", "test")
    if evaluation_source not in {"train", "test"}:
        raise ValueError("evaluation_source must be train or test")
    dataset_kwargs = {"revision": dataset_revision} if dataset_revision else {}
    ds = load_dataset(
        DATASET_NAME, DATASET_SUBSET, split=evaluation_source, **dataset_kwargs
    )
    manifest_path = args.indices_manifest or cfg.get("evaluation_indices_path")
    expected_count = int(cfg["validation_examples"])
    if manifest_path:
        if int(cfg.get("evaluation_offset", 0)) != 0:
            raise ValueError("evaluation_offset cannot be combined with an index manifest")
        resolved_manifest_path = _resolve_config_path(manifest_path, args.config)
        source_indices, manifest_identity = load_index_manifest(
            resolved_manifest_path,
            expected_split=evaluation_source,
            dataset_size=len(ds),
            expected_count=expected_count,
        )
        selection = {
            "method": "index_manifest",
            "index_manifest": manifest_identity,
            "indices_sha256": canonical_json_sha256(source_indices),
        }
    else:
        offset = int(cfg.get("evaluation_offset", 0))
        end = offset + expected_count
        if offset < 0 or end > len(ds):
            raise ValueError("evaluation slice is out of range")
        source_indices = list(range(offset, end))
        selection = {
            "method": "contiguous_slice",
            "offset": offset,
            "count": expected_count,
            "index_manifest": None,
            "indices_sha256": canonical_json_sha256(source_indices),
        }
    rows = ds.select(source_indices)
    dataset_provenance = {
        "name": DATASET_NAME,
        "subset": DATASET_SUBSET,
        "split": evaluation_source,
        "revision": dataset_revision,
        "fingerprint": getattr(ds, "_fingerprint", None),
    }
    provenance = {
        "run_label": args.run_label or ("adapter" if args.adapter else "base"),
        "evaluation_seed": evaluation_seed,
        "process_command": {
            "python_executable": sys.executable,
            "argv": list(sys.argv),
            "cwd": str(Path.cwd().resolve()),
        },
        "environment_snapshot": environment_identity,
        "model": {
            "name": cfg["model_name"],
            "configured_revision": model_revision,
            "resolved_revision": resolved_model_revision or model_revision,
            "dtype": str(next(model.parameters()).dtype),
        },
        "adapter": _adapter_identity(args.adapter) if args.adapter else None,
        "evaluation_config": _config_identity(args.config, cfg),
        "experiment_config": {
            **_config_identity(experiment_config_path, experiment_cfg),
            "source": experiment_config_source,
        },
        "experiment": {
            "training_seed": experiment_cfg.get("seed"),
            "reward_type": experiment_cfg.get("reward_type"),
            "target_words": list(experiment_cfg["target_words"]),
            "score_components": experiment_cfg.get("score_components"),
            "lens_path": experiment_cfg.get("lens_path"),
            "lens_sha256": file_sha256(experiment_cfg["lens_path"]),
            "expected_lens_sha256": expected_lens_sha256,
            "calibration_path": experiment_cfg.get("calibration_path"),
            "calibration_sha256": actual_calibration_sha256,
            "expected_calibration_sha256": expected_calibration_sha256,
        },
        "selection": selection,
        "git": git_provenance,
        "software": _software_versions(),
        "runtime": {
            "cuda_device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "cuda_version": torch.version.cuda,
            "batch_size": args.batch_size,
        },
    }
    metrics = evaluate(
        model, tokenizer, rows, runtime_cfg, reward, args.batch_size,
        source_indices=source_indices,
        dataset_provenance=dataset_provenance,
        provenance=provenance,
        output_jsonl=working_output_path,
        overwrite=True if working_output_path is not None else args.overwrite,
    )
    if (
        environment_temporary is not None
        and environment_path is not None
        and working_output_path is not None
        and requested_output_path is not None
    ):
        environment_temporary.replace(environment_path)
        working_output_path.replace(requested_output_path)
        metrics["per_example_jsonl"] = str(requested_output_path.resolve())
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
