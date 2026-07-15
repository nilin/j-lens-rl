from __future__ import annotations

import hashlib
import inspect
import json
import math
import struct
from pathlib import Path

import pytest

import modal_v14_v11style_celebration as v14


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_registration_schema_and_every_runtime_input_are_byte_pinned() -> None:
    assert sha256(ROOT / v14.REGISTRATION_PATH) == v14.REGISTRATION_SHA256
    assert sha256(ROOT / v14.METRIC_SCHEMA_PATH) == v14.METRIC_SCHEMA_SHA256
    assert sha256(ROOT / v14.SOURCE_CONFIG_PATH) == v14.SOURCE_CONFIG_SHA256
    for relative, expected in v14.EXPECTED_FILE_SHA256.items():
        assert sha256(ROOT / relative) == expected

    registration = json.loads((ROOT / v14.REGISTRATION_PATH).read_text())
    assert registration["protocol"] == v14.PROTOCOL
    assert "before_any_v14_gpu_training_wandb_run_or_outcome" in registration["status"]
    assert registration["scientific_status"] == {
        "classification": "development_only_posthoc_v11_style_replication",
        "untouched_independent_confirmation": False,
        "reason": (
            "The motivating seed195 curve and V11/V12/V13 outcomes were public "
            "before this registration, the V13 outcome informed the disclosed "
            "choice to retain positive celebration as treatment, and dense "
            "intermediate validation is an explicit new descriptive addition."
        ),
        "closed_protocols_mutated_or_reopened": False,
        "protected_final_access_authorized": False,
        "nominal_p_values_permitted_with_caveat": True,
    }
    assert registration["metric_schema"]["sha256"] == v14.METRIC_SCHEMA_SHA256
    assert registration["firewall"]["protected_final_payloads_mounted_or_accessed"] is False
    assert registration["execution"]["function_max_containers"] == 4
    assert registration["execution"]["local_entrypoint_waits_for_terminal_orchestrator_result"] is True
    assert registration["execution"]["app_name"] == v14.APP_NAME
    assert registration["execution"]["fresh_volume"] == v14.VOLUME_NAME


def test_fresh_run_identity_and_dense_v11_gate_are_exact() -> None:
    assert v14.SEEDS == (236, 237, 238, 239)
    assert not set(v14.SEEDS) & set(range(220, 232))
    assert v14.CONDITIONS == ("jlens", "signflip")
    assert v14.STEPS == (0, 1, 2, 3, 4, 5, 6)
    assert v14.V11_GATE_STEPS == (0, 4, 5, 6)
    assert v14.MAX_PARALLEL_GPUS == v14.WORKSPACE_GPU_LIMIT == 4
    assert len(v14.LABELS) == len(set(v14.LABELS)) == 8
    assert len(v14.WANDB_IDS) == len(set(v14.WANDB_IDS.values())) == 8
    assert all(run_id.startswith("dev-v14-") for run_id in v14.WANDB_IDS.values())
    assert "confirm-v11" not in v14.WANDB_GROUP
    assert "confirm-v12" not in v14.WANDB_GROUP
    assert "confirm-v13" not in v14.WANDB_GROUP


def test_configs_freeze_v11_science_and_exact_matched_signflip() -> None:
    treatment_keys: set[str] | None = None
    for seed in v14.SEEDS:
        treatment = v14.validate_config(ROOT, "jlens", seed)
        signflip = v14.validate_config(ROOT, "signflip", seed)
        treatment_keys = treatment_keys or set(treatment)
        assert set(treatment) == set(signflip) == treatment_keys
        assert treatment["seed"] == signflip["seed"] == seed
        assert treatment["updates"] == signflip["updates"] == 6
        assert treatment["validation_steps"] == signflip["validation_steps"] == [
            1,
            2,
            3,
            4,
            5,
            6,
        ]
        assert treatment["validation_observational_only"] is True
        assert treatment["early_stopping_patience"] is None
        assert treatment["target_words"] == ["yay", "great", "success", "nice"]
        assert treatment["learning_rate"] == signflip["learning_rate"] == 3e-6
        assert treatment["kl_beta"] == signflip["kl_beta"] == 0.02
        assert treatment["loss_type"] == signflip["loss_type"] == "dapo"
        assert treatment["scale_rewards"] == signflip["scale_rewards"] == "group"
        assert treatment["num_generations"] == signflip["num_generations"] == 8
        assert treatment["score_components"] == [
            dict(item) for item in v14.TREATMENT_COMPONENTS
        ]
        assert signflip["score_components"] == [
            dict(item) for item in v14.CONTROL_COMPONENTS
        ]
        differences = {
            key for key in treatment if treatment[key] != signflip[key]
        }
        assert differences == v14.PAIR_ALLOWED_DIFFERENCES


def _history() -> dict[int, dict[str, object]]:
    return {
        step: {
            "step": step,
            "validation_source": "train",
            "validation_indices_sha256": v14.CURVE_MANIFEST_SHA256,
            "exact_match": 0.38 + 0.0025 * step,
        }
        for step in v14.STEPS
    }


def _merged_reward_row(step: int, exact: float) -> dict[str, float | int]:
    return {
        "step": step,
        "reward": 0.1 * step,
        "reward_std": 0.02,
        "rewards/jlens_yay_great_success_nice_reward/mean": 0.1 * step,
        "rewards/jlens_yay_great_success_nice_reward/std": 0.02,
        "jlens/yay_great_success_nice_literal_rate": 0.0,
        "validation/exact_match": exact,
    }


def _captured_v11_shape_logs() -> list[dict[str, float | int]]:
    history = _history()
    return [
        *[
            _merged_reward_row(step, float(history[step]["exact_match"]))
            for step in v14.POST_BASELINE_STEPS
        ],
        {"step": 6, "train_runtime": 1.0},
    ]


def test_corrected_verifier_accepts_baseline_absent_and_lr_absent_on_merged_rows() -> None:
    history = _history()
    config = v14.expected_config(ROOT, "jlens", v14.SEEDS[0])
    summary = v14.verify_log_history(_captured_v11_shape_logs(), config, history)
    assert summary == {
        "optimizer_steps": 6,
        "validation_steps": [1, 2, 3, 4, 5, 6],
        "baseline_absent_from_trainer_log": True,
        "learning_rate_rows": 0,
        "one_j_reward_verified": True,
    }


def test_corrected_verifier_rejects_baseline_row_missing_node_and_bad_lr_rules() -> None:
    history = _history()
    config = v14.expected_config(ROOT, "jlens", v14.SEEDS[0])

    baseline_in_log = _captured_v11_shape_logs()
    baseline_in_log.insert(0, {"step": 0, "validation/exact_match": 0.38})
    with pytest.raises(RuntimeError, match="exactly steps 1..6"):
        v14.verify_log_history(baseline_in_log, config, history)

    missing_node = _captured_v11_shape_logs()
    missing_node[2].pop("validation/exact_match")
    missing_node[2]["learning_rate"] = 3e-6
    with pytest.raises(RuntimeError, match="exactly steps 1..6"):
        v14.verify_log_history(missing_node, config, history)

    missing_lr_on_nonvalidation = _captured_v11_shape_logs()
    missing_lr_on_nonvalidation[1].pop("validation/exact_match")
    with pytest.raises(RuntimeError, match="learning_rate may be absent"):
        v14.verify_log_history(missing_lr_on_nonvalidation, config, history)

    bad_lr = _captured_v11_shape_logs()
    bad_lr[0]["learning_rate"] = 2e-6
    with pytest.raises(RuntimeError, match="constant learning rate"):
        v14.verify_log_history(bad_lr, config, history)


def test_corrected_verifier_rejects_correctness_or_second_reward() -> None:
    history = _history()
    config = v14.expected_config(ROOT, "jlens", v14.SEEDS[0])
    logs = _captured_v11_shape_logs()
    logs[0]["rewards/gsm8k_reward/mean"] = 1.0
    with pytest.raises(RuntimeError, match="one-J-reward"):
        v14.verify_log_history(logs, config, history)


def _float32_from_bits(bits: int) -> float:
    return struct.unpack("!f", struct.pack("!I", bits))[0]


def test_corrected_verifier_accepts_two_ulp_duplicate_std_but_rejects_five() -> None:
    history = _history()
    config = v14.expected_config(ROOT, "jlens", v14.SEEDS[0])
    one_bits = struct.unpack("!I", struct.pack("!f", 1.0))[0]

    two_ulp = _captured_v11_shape_logs()
    two_ulp[0]["reward_std"] = 1.0
    two_ulp[0]["rewards/jlens_yay_great_success_nice_reward/std"] = (
        _float32_from_bits(one_bits + 2)
    )
    assert v14.verify_log_history(two_ulp, config, history)[
        "one_j_reward_verified"
    ] is True

    five_ulp = _captured_v11_shape_logs()
    five_ulp[0]["reward_std"] = 1.0
    five_ulp[0]["rewards/jlens_yay_great_success_nice_reward/std"] = (
        _float32_from_bits(one_bits + 5)
    )
    with pytest.raises(RuntimeError, match="one-J-reward"):
        v14.verify_log_history(five_ulp, config, history)


def _synthetic_results() -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for seed_index, seed in enumerate(v14.SEEDS):
        baseline = 0.38
        treatment = {
            "0": baseline,
            "1": 0.3825,
            "2": 0.385,
            "3": 0.3875,
            "4": 0.39,
            "5": 0.395,
            "6": 0.405 + 0.0025 * seed_index,
        }
        signflip = {
            "0": baseline,
            "1": 0.38,
            "2": 0.38,
            "3": 0.3825,
            "4": 0.3825,
            "5": 0.385,
            "6": 0.39 + 0.0025 * seed_index,
        }
        results[v14._label("jlens", seed)] = {"curve": treatment}
        results[v14._label("signflip", seed)] = {"curve": signflip}
    return results


def test_aggregate_predeclared_shape_and_nominal_significance() -> None:
    aggregate = v14.aggregate_results(_synthetic_results())
    assert aggregate["v11_shape_gate"]["steps"] == [0, 4, 5, 6]
    assert aggregate["v11_shape_gate"]["first_above_initial"] is True
    assert aggregate["v11_shape_gate"]["no_downward_steps_4_to_5_to_6"] is True
    assert aggregate["v11_shape_gate"]["passed"] is True
    for key in (
        "terminal_treatment_minus_signflip",
        "terminal_treatment_minus_baseline",
    ):
        test = aggregate[key]
        assert test["positives"] == 4
        assert test["negatives"] == test["ties"] == 0
        assert math.isclose(test["exact_two_sided_p"], 0.125)
        assert test["success"] is True
    assert aggregate["target_evidence_met"] is True
    assert aggregate["scientific_status"].startswith("development_only")


def test_modal_runner_is_fresh_noncreating_parallel_and_drains_every_call() -> None:
    source = (ROOT / v14.RUNNER_PATH).read_text()
    orchestrator = inspect.getsource(v14.orchestrate.get_raw_f())
    assert "create_if_missing=False" in source
    assert "max_containers=MAX_PARALLEL_GPUS" in source
    assert "for condition in CONDITIONS\n            for seed in SEEDS" in orchestrator
    assert orchestrator.index("state_volume.commit()") < orchestrator.index(
        "train_run.spawn"
    )
    assert "for condition in CONDITIONS:" in orchestrator
    assert "completion = calls[(condition, seed)].get()" in orchestrator
    assert "every_spawned_call_drained\": True" in orchestrator
    assert "after all eight calls were drained" in orchestrator
    local_entrypoint = source[source.index("@app.local_entrypoint()") :]
    assert local_entrypoint.index("orchestrate.spawn") < local_entrypoint.index(
        "record_launch_receipt.remote"
    ) < local_entrypoint.index("call.get()")
    assert "run.define_metric(\"global_step\")" in source
    assert 'step_metric="global_step"' in source
    assert v14.VOLUME_NAME not in {
        "j-lens-rl-confirmatory-v11-celebration-20260714b",
        "j-lens-rl-confirmatory-v12-celebration-20260714a",
        "j-lens-rl-confirmatory-v13-celebration-long-20260714a",
    }
    assert all(name in source for name in v14.FORBIDDEN_RUNTIME_NAMES)
