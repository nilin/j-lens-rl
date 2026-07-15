from __future__ import annotations

import hashlib
import inspect
import json
import math
from pathlib import Path

import pytest

import modal_v16_v14_manyseed_curve as v16


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_registration_and_every_runtime_input_are_byte_pinned() -> None:
    assert sha256(ROOT / v16.REGISTRATION_PATH) == v16.REGISTRATION_SHA256
    assert sha256(ROOT / v16.METRIC_SCHEMA_PATH) == v16.METRIC_SCHEMA_SHA256
    for relative, expected in v16.EXPECTED_FILE_SHA256.items():
        assert sha256(ROOT / relative) == expected

    registration = json.loads((ROOT / v16.REGISTRATION_PATH).read_text())
    assert registration["protocol"] == v16.PROTOCOL
    assert "before_any_v16_gpu_wandb_run_or_outcome" in registration["status"]
    assert registration["scientific_status"]["classification"] == (
        "development_only_adaptive_v14_many_seed_extension"
    )
    evaluation = registration["observational_evaluation"]
    assert evaluation["steps"] == [0, 2, 4, 6, 8, 10]
    assert evaluation["complete_curve_required"] is True
    assert evaluation["omit_or_select_intermediate_nodes_permitted"] is False
    assert registration["fresh_runs"]["seed_pairs"] == 16
    assert registration["fresh_runs"]["total_runs"] == 32
    assert registration["execution"]["function_max_containers"] == 4
    assert registration["execution"]["user_requested_concurrent_gpu_cap"] == 4
    assert registration["firewall"][
        "protected_final_payloads_mounted_or_accessed"
    ] is False


def test_fresh_identity_has_16_pairs_and_the_complete_curve() -> None:
    assert v16.SEEDS == tuple(range(248, 264))
    assert v16.CONDITIONS == ("jlens", "signflip")
    assert v16.OPTIMIZER_STEPS == tuple(range(1, 11))
    assert v16.STEPS == (0, 2, 4, 6, 8, 10)
    assert v16.POST_BASELINE_STEPS == (2, 4, 6, 8, 10)
    assert v16.DISPLAY_GATE_STEPS == (0, 2, 4, 6)
    assert len(v16.LABELS) == len(set(v16.LABELS)) == 32
    assert len(v16.WANDB_IDS) == len(set(v16.WANDB_IDS.values())) == 32
    assert v16.MAX_PARALLEL_GPUS == v16.USER_REQUESTED_GPU_CAP == 4
    assert v16.RUN_ORDER[:4] == (
        ("jlens", 248),
        ("signflip", 248),
        ("jlens", 249),
        ("signflip", 249),
    )


def test_all_32_configs_preserve_v14_science_and_matched_signflip() -> None:
    for seed in v16.SEEDS:
        treatment = v16.validate_config(ROOT, "jlens", seed)
        signflip = v16.validate_config(ROOT, "signflip", seed)
        assert treatment["seed"] == signflip["seed"] == seed
        assert treatment["updates"] == signflip["updates"] == 10
        assert treatment["eval_every"] == signflip["eval_every"] == 2
        assert treatment["validation_steps"] == [2, 4, 6, 8, 10]
        assert treatment["early_stopping_patience"] is None
        assert treatment["target_words"] == ["yay", "great", "success", "nice"]
        assert treatment["score_components"] == [
            dict(item) for item in v16.TREATMENT_COMPONENTS
        ]
        assert signflip["score_components"] == [
            dict(item) for item in v16.CONTROL_COMPONENTS
        ]
        assert {
            key for key in treatment if treatment[key] != signflip[key]
        } == v16.PAIR_ALLOWED_DIFFERENCES


def _history() -> dict[int, dict[str, object]]:
    return {
        step: {
            "step": step,
            "validation_source": "train",
            "validation_indices_sha256": v16.CURVE_MANIFEST_SHA256,
            "exact_match": 0.38 + 0.001 * step,
        }
        for step in v16.STEPS
    }


def _trainer_logs() -> list[dict[str, float | int]]:
    history = _history()
    rows: list[dict[str, float | int]] = []
    for step in v16.OPTIMIZER_STEPS:
        row: dict[str, float | int] = {
            "step": step,
            "reward": 0.01 * step,
            "reward_std": 0.02,
            "rewards/jlens_yay_great_success_nice_reward/mean": 0.01 * step,
            "rewards/jlens_yay_great_success_nice_reward/std": 0.02,
            "jlens/yay_great_success_nice_literal_rate": 0.0,
        }
        if step in v16.POST_BASELINE_STEPS:
            row["validation/exact_match"] = float(history[step]["exact_match"])
        else:
            row["learning_rate"] = 3e-6
        rows.append(row)
    rows.append({"step": 10, "train_runtime": 1.0})
    return rows


def test_sparse_eval_verifier_requires_all_rewards_and_exact_eval_nodes() -> None:
    history = _history()
    config = v16.expected_config(ROOT, "jlens", v16.SEEDS[0])
    summary = v16.verify_log_history(_trainer_logs(), config, history)
    assert summary["optimizer_steps"] == 10
    assert summary["validation_steps"] == [2, 4, 6, 8, 10]
    assert summary["baseline_absent_from_trainer_log"] is True

    missing_reward = _trainer_logs()
    del missing_reward[2]
    with pytest.raises(RuntimeError, match="reward steps 1..10"):
        v16.verify_log_history(missing_reward, config, history)

    missing_eval = _trainer_logs()
    missing_eval[5].pop("validation/exact_match")
    missing_eval[5]["learning_rate"] = 3e-6
    with pytest.raises(RuntimeError, match="2,4,6,8,10"):
        v16.verify_log_history(missing_eval, config, history)


def test_archived_wandb_export_proves_global_step_axis_and_baseline() -> None:
    exported = json.loads(
        (ROOT / "protocol_archive/seed195_public_evidence/wandb_export.json").read_text()
    )
    history = exported["payload"]["history"]
    validation = [
        row for row in history if row.get("train/validation/exact_match") is not None
    ]
    assert [row["train/global_step"] for row in validation] == [0, 4, 10, 20]
    assert validation[0]["train/validation/exact_match"] == 0.3825
    assert validation[0]["_step"] == 0
    assert validation[1]["_step"] != validation[1]["train/global_step"]


def _synthetic_results() -> dict[str, dict[str, object]]:
    treatment_curve = {
        "0": 0.38,
        "2": 0.39,
        "4": 0.395,
        "6": 0.40,
        "8": 0.3975,
        "10": 0.405,
    }
    control_curve = {
        "0": 0.38,
        "2": 0.3825,
        "4": 0.385,
        "6": 0.3875,
        "8": 0.3875,
        "10": 0.39,
    }
    return {
        v16._label(condition, seed): {
            "curve": treatment_curve if condition == "jlens" else control_curve,
            "data_indices_sha256": f"{seed:064x}",
        }
        for condition in v16.CONDITIONS
        for seed in v16.SEEDS
    }


def test_aggregate_retains_every_node_and_reports_sd_sem_and_registered_tests() -> None:
    aggregate = v16.aggregate_results(_synthetic_results())
    assert aggregate["complete_curve_steps"] == [0, 2, 4, 6, 8, 10]
    assert aggregate["every_registered_node_retained"] is True
    assert [row["global_step"] for row in aggregate["rows"]] == [0, 2, 4, 6, 8, 10]
    assert all(
        all(key in row for key in ("treatment_sd", "treatment_sem", "signflip_sd", "signflip_sem", "paired_sd", "paired_sem"))
        for row in aggregate["rows"]
    )
    assert aggregate["early_complete_node_shape"]["passed"] is True
    assert aggregate["primary_integrated_treatment_minus_baseline"]["success"] is True
    assert aggregate["paired_integrated_treatment_minus_signflip"]["success"] is True
    assert aggregate["target_evidence_met"] is True
    assert aggregate["causal_reward_sign_evidence_met"] is True


def test_sign_test_uses_predeclared_alpha_not_an_all_positive_requirement() -> None:
    result = v16.exact_two_sided_sign_test([0.01] * 12 + [-0.001] * 4)
    assert result["positives"] == 12
    assert result["negatives"] == 4
    assert math.isclose(result["exact_two_sided_p"], 0.076812744140625)
    assert result["mean_effect"] > 0
    assert result["success"] is True

    opposite_direction = v16.exact_two_sided_sign_test([-0.01] * 12 + [0.1] * 4)
    assert opposite_direction["exact_two_sided_p"] <= 0.15
    assert opposite_direction["mean_effect"] > 0
    assert opposite_direction["positives"] < opposite_direction["negatives"]
    assert opposite_direction["success"] is False

    cancellation = v16.exact_two_sided_sign_test(
        [0.0025 + 0.0475 - 0.05]
    )
    assert cancellation["effect_units"] == [0]
    assert cancellation["ties"] == 1

    with pytest.raises(RuntimeError, match="off the registered 1/2000 lattice"):
        v16.exact_two_sided_sign_test([0.00025])


def test_runner_caps_four_gpus_and_recovers_only_exact_32_call_dispatch() -> None:
    source = (ROOT / v16.RUNNER_PATH).read_text()
    orchestrator = inspect.getsource(v16.orchestrate.get_raw_f())
    assert "max_containers=MAX_PARALLEL_GPUS" in source
    assert '"stage": "all_32_spawned_capacity_queue_allowed"' in orchestrator
    assert "modal.FunctionCall.from_id" in orchestrator
    assert orchestrator.index("modal.FunctionCall.from_id") < orchestrator.index(
        "train_run.spawn"
    )
    assert "after all 32 calls were drained" in orchestrator
    assert '"every_spawned_call_drained": True' in orchestrator
    assert "create_if_missing=False" in source
    assert '"complete_curve_steps": list(STEPS)' in source
    assert "fieldnames[:10]" in source
    assert "receipt.get(\"uploaded_file_sha256\") != expected_uploads" in source
    assert "receipt.get(\"schema_version\") != 2" in source
    assert 'run_dir / "checkpoint-10" / "trainer_state.json"' in source
    assert 'artifact["version"][1:].isdigit()' in source
    assert 'f"validation/paired_seed{seed}"' in source
    assert "effect_axis.set_xticks(list(STEPS))" in source
