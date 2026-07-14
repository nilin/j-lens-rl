import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from jlens_rl.common import (
    load_index_manifest,
    repository_provenance,
    require_clean_repository_provenance,
    resolve_repository_root,
)
from jlens_rl.train import DeterministicValidationCallback, create_run_directory


def _load_protocol_module():
    path = Path(__file__).resolve().parents[1] / "scripts/confirmatory_protocol.py"
    spec = importlib.util.spec_from_file_location("confirmatory_protocol_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_index_manifest_accepts_object_and_rejects_duplicates(tmp_path):
    valid = tmp_path / "valid.json"
    valid.write_text(json.dumps({"indices": [5, 1, 9]}))
    assert load_index_manifest(valid) == [5, 1, 9]

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(json.dumps({"indices": [1, 1]}))
    with pytest.raises(ValueError, match="duplicate"):
        load_index_manifest(duplicate)


def test_repository_provenance_fingerprints_dirty_tree():
    provenance = repository_provenance(".")
    assert len(provenance["git_commit"]) == 40
    assert len(provenance["source_tree_sha256"]) == 64
    assert isinstance(provenance["git_dirty"], bool)


def test_repository_root_survives_wheel_style_module_path(monkeypatch, tmp_path):
    checkout = Path(__file__).resolve().parents[1]
    monkeypatch.delenv("JLENS_REPOSITORY_ROOT", raising=False)
    monkeypatch.chdir(checkout)
    fake_module = tmp_path / "site-packages/jlens_rl/train.py"
    assert resolve_repository_root(fake_module) == checkout

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("JLENS_REPOSITORY_ROOT", str(checkout))
    assert resolve_repository_root(fake_module) == checkout


def test_required_repository_provenance_fails_closed():
    valid = {**repository_provenance("."), "git_dirty": False}
    require_clean_repository_provenance(valid)

    with pytest.raises(RuntimeError, match="commit provenance"):
        require_clean_repository_provenance(
            {"git_commit": None, "git_dirty": None, "source_tree_sha256": None}
        )
    with pytest.raises(RuntimeError, match="clean Git"):
        require_clean_repository_provenance({**valid, "git_dirty": True})


def test_observational_validation_cannot_stop_training():
    callback = DeterministicValidationCallback(
        tokenizer=None,
        rows=None,
        cfg={
            "eval_every": 5,
            "early_stopping_patience": 1,
            "validation_observational_only": True,
        },
    )
    callback.best_exact_match = 1.0
    callback.evaluate_and_log = lambda model, step: {"exact_match": 0.0}
    control = SimpleNamespace(should_training_stop=False)
    result = callback.on_step_end(
        args=None,
        state=SimpleNamespace(global_step=5),
        control=control,
        model=None,
    )
    assert result.should_training_stop is False


def test_run_directory_must_be_empty(tmp_path):
    new_dir = create_run_directory(tmp_path / "new-run")
    assert new_dir.is_dir()
    (new_dir / "old-result.json").write_text("{}")
    with pytest.raises(FileExistsError, match="not empty"):
        create_run_directory(new_dir)


def test_protocol_verifies_labeled_evaluation_contract(monkeypatch, tmp_path):
    protocol = _load_protocol_module()
    state_dir = tmp_path / ".confirmatory"
    manifest_dir = state_dir / "manifests"
    eval_dir = state_dir / "evals"
    config_dir = tmp_path / "configs"
    manifest_dir.mkdir(parents=True)
    eval_dir.mkdir()
    config_dir.mkdir()
    sealed_manifest = manifest_dir / "sealed_final_indices.json"
    sealed_manifest.write_text(json.dumps({"indices": [7]}))
    eval_config_path = config_dir / "confirmatory_sealed_eval.json"
    experiment_config_path = config_dir / "confirmatory_jlens_seed142.json"
    eval_config_path.write_text("{}")
    experiment_config_path.write_text("{}")

    score_components = [{
        "layer": 8,
        "start_fraction": 0.5,
        "end_fraction": 1.0,
        "aggregation": "mean",
        "weight": 1.0,
    }]
    eval_config = {"kind": "sealed"}
    experiment_config = {"score_components": score_components}
    common = {"lens_sha256": "1" * 64, "calibration_sha256": "2" * 64}

    monkeypatch.setattr(protocol, "REPO", tmp_path)
    monkeypatch.setattr(protocol, "STATE_DIR", state_dir)
    monkeypatch.setattr(protocol, "MANIFEST_DIR", manifest_dir)
    monkeypatch.setattr(
        protocol,
        "load_and_verify_state",
        lambda: {"git_commit": "a" * 40},
    )
    monkeypatch.setattr(
        protocol,
        "config_path",
        lambda condition, seed: experiment_config_path,
    )
    monkeypatch.setattr(
        protocol,
        "_sealed_evaluation_reference",
        lambda: {
            "dataset_fingerprint": "fingerprint",
            "references": {
                7: {
                    "prompt_sha256": "3" * 64,
                    "prompt_token_ids_sha256": "4" * 64,
                    "answer": "#### 2",
                }
            },
            "extract_answer": lambda completion: "2",
            "gsm8k_reward": lambda completion, answer: 1.0,
            "decode_completion": lambda token_ids: "reasoning #### 2",
        },
    )

    def fake_load_config(path):
        name = Path(path).name
        if name == "confirmatory_common.json":
            return common
        if name == "confirmatory_sealed_eval.json":
            return eval_config
        return experiment_config

    monkeypatch.setattr(protocol, "load_config", fake_load_config)
    record = {
        "schema_version": 1,
        "dataset": {
            "name": "openai/gsm8k",
            "subset": "main",
            "split": "train",
            "revision": protocol.DATASET_REVISION,
            "fingerprint": "fingerprint",
        },
        "source_index": 7,
        "prompt_sha256": "3" * 64,
        "prompt_token_ids_sha256": "4" * 64,
        "completion": "reasoning #### 2",
        "completion_token_ids": list(range(12)),
        "prediction": "2",
        "correct": True,
        "completion_tokens": 12,
        "target_words": ["solved"],
        "literal_target_matches": [],
        "literal_target_used": False,
        "generation": {
            "do_sample": False,
            "max_prompt_tokens": 384,
            "max_new_tokens": 256,
            "padding_side": "left",
        },
        "provenance": {
            "run_label": "base",
            "evaluation_seed": 0,
            "model": {
                "name": "Qwen/Qwen2.5-0.5B-Instruct",
                "configured_revision": protocol.MODEL_REVISION,
                "resolved_revision": protocol.MODEL_REVISION,
                "dtype": "torch.bfloat16",
            },
            "adapter": None,
            "evaluation_config": {
                "file_sha256": protocol.sha256_file(eval_config_path),
                "resolved_sha256": protocol.canonical_sha256(eval_config),
            },
            "experiment_config": {
                "file_sha256": protocol.sha256_file(experiment_config_path),
                "resolved_sha256": protocol.canonical_sha256(experiment_config),
                "source": "explicit",
            },
            "experiment": {
                "training_seed": 142,
                "reward_type": "jlens",
                "target_words": ["solved"],
                "score_components": score_components,
                "lens_sha256": common["lens_sha256"],
                "calibration_sha256": common["calibration_sha256"],
            },
            "selection": {
                "method": "index_manifest",
                "indices_sha256": protocol.canonical_sha256([7]),
                "index_manifest": {
                    "sha256": protocol.sha256_file(sealed_manifest),
                    "dataset": "openai/gsm8k",
                    "subset": "main",
                    "split": "train",
                    "count": 1,
                },
            },
            "git": {
                "git_commit": "a" * 40,
                "git_dirty": False,
                "source_tree_sha256": "5" * 64,
            },
            "software": {
                "j-lens-rl": "0.1.0",
                "torch": "2.9.1",
                "transformers": "5.5.0",
                "datasets": "4.7.0",
                "peft": "0.18.0",
            },
            "runtime": {
                "cuda_device_name": "NVIDIA L40S",
                "cuda_version": "12.8",
                "batch_size": 64,
            },
        },
    }
    output = eval_dir / "base.jsonl"
    output.write_text(json.dumps(record) + "\n")
    protocol.verify_evaluation_jsonl(output, "base")

    record["correct"] = False
    output.write_text(json.dumps(record) + "\n")
    with pytest.raises(protocol.ProtocolError, match="derived outcome"):
        protocol.verify_evaluation_jsonl(output, "base")

    record["correct"] = True
    record["provenance"]["run_label"] = "wrong"
    output.write_text(json.dumps(record) + "\n")
    with pytest.raises(protocol.ProtocolError, match="evaluation role"):
        protocol.verify_evaluation_jsonl(output, "base")


def test_acceptance_report_requires_signflip_specificity(monkeypatch, tmp_path):
    protocol = _load_protocol_module()
    state_dir = tmp_path / ".confirmatory"
    evidence = state_dir / "evidence"
    evidence.mkdir(parents=True)
    semantic_path = evidence / "semantic_vs_base.json"
    specificity_path = evidence / "semantic_vs_signflip.json"
    curve_path = evidence / "curve_gate.json"
    curve_plot = evidence / "curve.png"
    completed_runs = evidence / "completed_runs.json"
    acceptance = evidence / "acceptance.json"

    semantic = {
        "mean_accuracy_difference": 0.02,
        "crossed_seed_item_bootstrap": {
            "mean_accuracy_difference_ci_low": 0.001,
        },
        "seed_sign_test": {
            "positive": 6,
            "negative": 0,
            "tied_excluded": 0,
            "exact_two_sided_p": 0.03125,
        },
    }
    specificity = {
        "primary_estimand": "difference_in_differences",
        "difference_in_differences": {
            "mean_difference_in_differences": 0.01,
            "crossed_seed_item_bootstrap": {
                "mean_difference_in_differences_ci_low": 0.0001,
            },
        },
    }
    semantic_path.write_text(json.dumps(semantic))
    specificity_path.write_text(json.dumps(specificity))
    curve_path.write_text(json.dumps({"passed": True}))
    curve_plot.write_bytes(b"png")
    completed_runs.write_text("{}")
    monkeypatch.setattr(protocol, "STATE_DIR", state_dir)
    monkeypatch.setattr(protocol, "CURVE_GATE_PATH", curve_path)
    monkeypatch.setattr(protocol, "CURVE_PLOT_PATH", curve_plot)
    monkeypatch.setattr(protocol, "COMPLETED_RUNS_PATH", completed_runs)
    monkeypatch.setattr(protocol, "ACCEPTANCE_PATH", acceptance)
    monkeypatch.setattr(protocol, "verify_unlock", lambda: None)
    monkeypatch.setattr(
        protocol,
        "_recompute_final_comparisons",
        lambda: (semantic, specificity),
    )
    monkeypatch.setattr(
        protocol,
        "final_evaluation_hashes",
        lambda: {"base.jsonl": "f" * 64},
    )
    assert protocol.final_report()["passed"] is True

    acceptance.unlink()
    specificity["difference_in_differences"]["crossed_seed_item_bootstrap"][
        "mean_difference_in_differences_ci_low"
    ] = -0.001
    specificity_path.write_text(json.dumps(specificity))
    assert protocol.final_report()["passed"] is False

    modal_source = (Path(__file__).resolve().parents[1] / "modal_experiments.py").read_text()
    assert 'evidence_dir / "acceptance.json"' in modal_source
    assert "acceptance_report.json" not in modal_source
    assert "batch_upload(force=False)" in modal_source
    assert "batch_upload(force=True)" not in modal_source


def test_v2_allocation_retires_setup_run_and_exposed_curve():
    protocol = _load_protocol_module()
    assert (42, 1000, ((7000, 7200),)) in protocol.HISTORICAL_SHUFFLE_RULES
    assert protocol.SPLIT_SIZES == {
        "curve_indices.json": 400,
        "sealed_final_indices.json": 2900,
        "future_reserve_indices.json": 64,
    }
    assert protocol.PROTOCOL == "j-lens-rl-confirmatory-v2"


def test_training_behavior_log_rejects_extrinsic_reward(monkeypatch, tmp_path):
    protocol = _load_protocol_module()
    monkeypatch.setattr(protocol, "FIXED_UPDATES", 1)
    path = tmp_path / "log_history.json"
    row = {
        "step": 1,
        "reward": 0.0,
        "reward_std": 1.0,
        "rewards/jlens_solved_reward/mean": 0.0,
        "completions/mean_length": 128.0,
        "completions/clipped_ratio": 0.0,
        "jlens/solved_literal_rate": 0.0,
    }
    path.write_text(json.dumps([row]))
    assert protocol.training_behavior_summary(path)["steps"] == 1

    row["rewards/gsm8k_reward/mean"] = 0.0
    path.write_text(json.dumps([row]))
    with pytest.raises(protocol.ProtocolError, match="one-J-reward"):
        protocol.training_behavior_summary(path)


def test_unlock_chain_detects_adapter_mutation(monkeypatch, tmp_path):
    protocol = _load_protocol_module()
    state_dir = tmp_path / ".confirmatory"
    evidence_dir = state_dir / "evidence"
    evidence_dir.mkdir(parents=True)
    monkeypatch.setattr(protocol, "STATE_DIR", state_dir)
    monkeypatch.setattr(protocol, "SEEDS", (142,))
    monkeypatch.setattr(protocol, "REQUIRED_CONDITIONS", ("jlens", "signflip"))
    monkeypatch.setattr(
        protocol,
        "load_and_verify_state",
        lambda: {"protocol": protocol.PROTOCOL, "git_commit": "a" * 40},
    )

    for condition in protocol.REQUIRED_CONDITIONS:
        directory = state_dir / "runs" / f"{condition}_seed142"
        for relative in (
            "run_manifest.json",
            "resolved_config.json",
            "data_indices.json",
            "validation_history.jsonl",
            "log_history.json",
            "checkpoint-25/trainer_state.json",
            "final/adapter_config.json",
            "final/adapter_model.safetensors",
        ):
            path = directory / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if relative == "log_history.json":
                rows = [
                    {
                        "step": step,
                        "reward": float(step),
                        "reward_std": 1.0,
                        "rewards/jlens_solved_reward/mean": float(step),
                        "completions/mean_length": 128.0,
                        "completions/clipped_ratio": 0.0,
                        "jlens/solved_literal_rate": 0.0,
                    }
                    for step in range(1, 26)
                ]
                path.write_text(json.dumps(rows))
            else:
                path.write_text(relative)

    completed_path = evidence_dir / "completed_runs.json"
    curve_path = evidence_dir / "curve_gate.json"
    curve_plot = evidence_dir / "curve.png"
    unlock_path = state_dir / "final_unlocked.json"
    monkeypatch.setattr(protocol, "COMPLETED_RUNS_PATH", completed_path)
    monkeypatch.setattr(protocol, "CURVE_GATE_PATH", curve_path)
    monkeypatch.setattr(protocol, "CURVE_PLOT_PATH", curve_plot)
    monkeypatch.setattr(protocol, "UNLOCK_PATH", unlock_path)

    completed = protocol.completed_run_artifact_manifest()
    protocol.write_json(completed_path, completed)
    curve_plot.write_bytes(b"curve")
    gate = {
        "criterion": "gate",
        "predeclared_steps": [0, 5, 10, 15],
        "n_seeds": 1,
        "examples_per_seed": 400,
        "per_seed_exact_match": {},
        "mean_exact_match": {},
        "passed": True,
    }
    stored_gate = {
        **gate,
        "curve_plot": {"sha256": protocol.sha256_file(curve_plot)},
    }
    protocol.write_json(curve_path, stored_gate)
    protocol.write_json(
        unlock_path,
        {
            "git_commit": "a" * 40,
            "curve_gate_sha256": protocol.sha256_file(curve_path),
            "completed_runs_sha256": protocol.sha256_file(completed_path),
        },
    )
    monkeypatch.setattr(protocol, "compute_curve_gate", lambda write_result=False: gate)
    protocol.verify_unlock()

    adapter = state_dir / "runs/jlens_seed142/final/adapter_model.safetensors"
    adapter.write_text("mutated")
    with pytest.raises(protocol.ProtocolError, match="run artifact changed"):
        protocol.verify_unlock()
