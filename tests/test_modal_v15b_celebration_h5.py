from __future__ import annotations

import hashlib
import inspect
import json
import math
from pathlib import Path

import pytest

import modal_v15b_celebration_h5 as v15b


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_registration_and_runtime_inputs_are_byte_pinned() -> None:
    assert sha256(ROOT / v15b.REGISTRATION_PATH) == v15b.REGISTRATION_SHA256
    assert sha256(ROOT / v15b.METRIC_SCHEMA_PATH) == v15b.METRIC_SCHEMA_SHA256
    assert sha256(ROOT / v15b.PREEMPTION_CLOSEOUT_PATH) == v15b.PREEMPTION_CLOSEOUT_SHA256
    for relative, expected in v15b.EXPECTED_FILE_SHA256.items():
        assert sha256(ROOT / relative) == expected

    registration = json.loads((ROOT / v15b.REGISTRATION_PATH).read_text())
    assert registration["protocol"] == v15b.PROTOCOL
    assert "before_any_v15b_gpu_wandb_run_or_outcome" in registration["status"]
    assert registration["scientific_status"]["classification"] == (
        "development_only_adaptive_v14_horizon_replication"
    )
    assert registration["execution"]["function_max_containers"] == 4
    assert registration["execution"]["workspace_gpu_limit"] == 4
    assert "FunctionCall.from_id" in registration["execution"][
        "coordinator_preemption_recovery"
    ]
    assert registration["firewall"][
        "protected_final_payloads_mounted_or_accessed"
    ] is False


def test_identity_and_configs_preserve_the_fixed_v15_science() -> None:
    assert v15b.SEEDS == (244, 245, 246, 247)
    assert v15b.CONDITIONS == ("jlens", "signflip")
    assert v15b.STEPS == (0, 1, 2, 3, 4, 5)
    assert v15b.DISPLAY_GATE_STEPS == (0, 3, 4, 5)
    assert v15b.MAX_PARALLEL_GPUS == v15b.WORKSPACE_GPU_LIMIT == 4
    assert len(v15b.LABELS) == len(set(v15b.LABELS)) == 8
    assert len(v15b.WANDB_IDS) == len(set(v15b.WANDB_IDS.values())) == 8
    assert all(value.startswith("dev-v15b-") for value in v15b.WANDB_IDS.values())

    for seed in v15b.SEEDS:
        treatment = v15b.validate_config(ROOT, "jlens", seed)
        signflip = v15b.validate_config(ROOT, "signflip", seed)
        assert treatment["seed"] == signflip["seed"] == seed
        assert treatment["updates"] == signflip["updates"] == 5
        assert treatment["validation_steps"] == [1, 2, 3, 4, 5]
        assert treatment["target_words"] == ["yay", "great", "success", "nice"]
        assert treatment["score_components"] == [
            dict(item) for item in v15b.TREATMENT_COMPONENTS
        ]
        assert signflip["score_components"] == [
            dict(item) for item in v15b.CONTROL_COMPONENTS
        ]
        assert {
            key for key in treatment if treatment[key] != signflip[key]
        } == v15b.PAIR_ALLOWED_DIFFERENCES


def _synthetic_results() -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for index, seed in enumerate(v15b.SEEDS):
        results[v15b._label("jlens", seed)] = {
            "curve": {
                "0": 0.38,
                "1": 0.3825,
                "2": 0.385,
                "3": 0.3875,
                "4": 0.39,
                "5": 0.405 + 0.0025 * index,
            }
        }
        results[v15b._label("signflip", seed)] = {
            "curve": {
                "0": 0.38,
                "1": 0.38,
                "2": 0.38,
                "3": 0.3825,
                "4": 0.3825,
                "5": 0.39 + 0.0025 * index,
            }
        }
    return results


def test_predeclared_shape_and_both_nominal_tests() -> None:
    aggregate = v15b.aggregate_results(_synthetic_results())
    shape = aggregate["display_shape_gate"]
    assert shape["steps"] == [0, 3, 4, 5]
    assert shape["means"] == pytest.approx([0.38, 0.3875, 0.39, 0.40875])
    assert shape["first_above_initial"] is True
    assert shape["no_downward_steps_3_to_4_to_5"] is True
    assert shape["passed"] is True
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


def _dispatch_status(root_call_id: str = "fc-root") -> dict[str, object]:
    return {
        "schema_version": 1,
        "protocol": v15b.PROTOCOL,
        "claim_id": "claim",
        "root_call_id": root_call_id,
        "stage": "all_eight_spawned_capacity_queue_allowed",
        "max_parallel_gpus": 4,
        "workspace_gpu_limit": 4,
        "worker_call_ids": {
            label: f"fc-{index}" for index, label in enumerate(v15b.LABELS)
        },
    }


def test_dispatch_recovery_requires_exact_root_and_all_eight_call_ids() -> None:
    recovered = v15b._validate_recoverable_dispatch(
        _dispatch_status(), "claim", "fc-root"
    )
    assert set(recovered) == {
        (condition, seed)
        for condition in v15b.CONDITIONS
        for seed in v15b.SEEDS
    }
    assert len(set(recovered.values())) == 8

    wrong_root = _dispatch_status("fc-other")
    with pytest.raises(RuntimeError, match="not exactly recoverable"):
        v15b._validate_recoverable_dispatch(wrong_root, "claim", "fc-root")

    partial = _dispatch_status()
    partial["worker_call_ids"].pop(v15b.LABELS[-1])
    with pytest.raises(RuntimeError, match="not exactly recoverable"):
        v15b._validate_recoverable_dispatch(partial, "claim", "fc-root")


def test_recovery_reattaches_before_fresh_spawn_and_preserves_call_ids() -> None:
    source = (ROOT / v15b.RUNNER_PATH).read_text()
    orchestrator = inspect.getsource(v15b.orchestrate.get_raw_f())
    recovery = orchestrator.index("modal.FunctionCall.from_id")
    fresh_intent = orchestrator.index("_write_exclusive(_intent_path")
    fresh_spawn = orchestrator.index("train_run.spawn")
    assert recovery < fresh_intent < fresh_spawn
    assert '"root_call_id": root_call_id' in orchestrator
    assert '"prior_dispatch": prior_dispatch' in orchestrator
    assert "duplicate spawning is forbidden" in orchestrator
    assert "max_containers=MAX_PARALLEL_GPUS" in source
    assert "create_if_missing=False" in source
    assert "completion = calls[(condition, seed)].get()" in orchestrator
    assert '"every_spawned_call_drained": True' in orchestrator
    assert "call.get()" in source[source.index("@app.local_entrypoint()") :]
