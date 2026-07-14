from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "protocol_archive" / "v10_modal_execution_contract.template.json"
SPEC = importlib.util.spec_from_file_location(
    "modal_confirmatory_v10_fast", ROOT / "modal_confirmatory_v10_fast.py"
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def function_source(name: str) -> str:
    source = (ROOT / "modal_confirmatory_v10_fast.py").read_text()
    tree = ast.parse(source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name
    )
    return ast.get_source_segment(source, node) or ""


def contract() -> dict:
    return json.loads(TEMPLATE.read_text())


def test_template_is_exact_but_deliberately_inert() -> None:
    value = runner.validate_contract_shape(contract(), allow_disabled=True)
    assert value["launch_enabled"] is False
    with pytest.raises(runner.ModalV10Error, match="candidate/design"):
        runner.validate_contract_shape(value)
    assert value["science_registration"] == {
        "draft_path": runner.SCIENCE_REGISTRATION_PATH,
        "draft_sha256": runner.SCIENCE_REGISTRATION_SHA256,
        "candidate_freeze_path": runner.CANDIDATE_FREEZE_PATH,
        "candidate_freeze_sha256": runner.CANDIDATE_FREEZE_SHA256,
        "integrity_amendment_path": runner.INTEGRITY_AMENDMENT_PATH,
        "integrity_amendment_sha256": runner.INTEGRITY_AMENDMENT_SHA256,
    }
    assert sha256(ROOT / runner.SCIENCE_REGISTRATION_PATH) == runner.SCIENCE_REGISTRATION_SHA256
    assert sha256(ROOT / runner.CANDIDATE_FREEZE_PATH) == runner.CANDIDATE_FREEZE_SHA256
    assert sha256(ROOT / runner.INTEGRITY_AMENDMENT_PATH) == runner.INTEGRITY_AMENDMENT_SHA256


def test_plan_is_four_treatments_gate_four_controls_then_serial_nine() -> None:
    plan = runner.execution_plan()
    assert [phase["phase"] for phase in plan] == [
        "treatment_training",
        "registered_curve_gate",
        "matched_signflip_training",
        "sealed_final_collection",
    ]
    assert [job["seed"] for job in plan[0]["jobs"]] == [216, 217, 218, 219]
    assert [job["slot"] for job in plan[0]["jobs"]] == [0, 1, 2, 3]
    assert plan[1]["steps"] == [0, 2, 3, 4]
    assert plan[1]["failure_action"] == "stop_without_controls_or_final"
    assert plan[2]["requires"] == "registered_curve_gate_passed"
    assert [job["slot"] for job in plan[2]["jobs"]] == [0, 1, 2, 3]
    assert plan[3]["parallelism"] == 1
    assert [job["label"] for job in plan[3]["jobs"]] == list(runner.FINAL_LABELS)
    assert max(phase["parallelism"] for phase in plan) == 4


def test_curve_gate_requires_first_rise_and_no_later_drop() -> None:
    histories = {
        f"jlens_seed{seed}": {
            0: {"exact_match": 0.38},
            2: {"exact_match": 0.39 + offset},
            3: {"exact_match": 0.40 + offset},
            4: {"exact_match": 0.40 + offset},
        }
        for offset, seed in zip((0.0, 0.01, -0.005, 0.005), runner.SEEDS)
    }
    gate = runner.curve_gate_from_histories(histories)
    assert gate["passed"] is True
    assert gate["mean_exact_match"]["2"] > gate["mean_exact_match"]["0"]
    histories["jlens_seed216"][4]["exact_match"] = 0.30
    assert runner.curve_gate_from_histories(histories)["passed"] is False
    with pytest.raises(runner.ModalV10Error, match="exactly four"):
        runner.curve_gate_from_histories({"jlens_seed216": histories["jlens_seed216"]})


def test_contract_freezes_negative_fuck_and_exact_positive_signflip() -> None:
    value = contract()
    treatment = value["candidate"]["treatment_score_components"]
    control = value["candidate"]["matched_control_score_components"]
    assert treatment == [{
        "layer": 8,
        "start_fraction": 0.5,
        "end_fraction": 1.0,
        "aggregation": "mean",
        "weight": -1.0,
    }]
    assert control == [{**treatment[0], "weight": 1.0}]
    changed = copy.deepcopy(value)
    changed["candidate"]["matched_control_score_components"][0]["weight"] = 0.0
    with pytest.raises(runner.ModalV10Error, match="candidate/design"):
        runner.validate_contract_shape(changed, allow_disabled=True)


def test_registered_spec_must_bind_exact_contract_and_modal_runtime() -> None:
    value = contract()
    digest = "a" * 64
    source = value["runtime_source"]
    registration = value["science_registration"]
    spec = {
        "protocol": runner.FROZEN_SCIENTIFIC_PROTOCOL,
        "target_words": ["fuck"],
        "seeds": list(runner.SEEDS),
        "conditions": list(runner.CONDITIONS),
        "terminal_step": 4,
        "curve_gate": {"steps": [0, 2, 3, 4], "criterion": runner.CURVE_CRITERION},
        "treatment_score_components": value["candidate"]["treatment_score_components"],
        "matched_control_score_components": value["candidate"]["matched_control_score_components"],
        "artifacts": {
            "calibration_path": runner.CALIBRATION_PATH,
            "calibration_sha256": runner.CALIBRATION_SHA256,
        },
        "training": {
            "updates": 4,
            "learning_rate": 3e-6,
            "score_stride": 5,
            "validation_steps": [2, 3, 4],
        },
        "hardware": {
            "backend": "modal",
            "device_name": "NVIDIA L40S",
            "max_gpu_processes": 1,
            "gpu_per_worker": 1,
            "max_modal_gpus_before_2026_07_14_23_00_utc": 5,
            "max_modal_gpus_at_or_after_2026_07_14_23_00_utc": 10,
        },
        "final_collection": {
            "count": 900,
            "manifest_path": "/state/manifests/sealed_final_indices.json",
            "manifest_sha256": runner.FINAL_MANIFEST_SHA256,
            "labels": list(runner.FINAL_LABELS),
        },
        "wandb": {"mode": "online"},
        "paths": {"state_config_prefix": "/state"},
        "repository": "/workspace/j-lens-rl",
        "git_commit": source["git_commit"],
        "source_tree_sha256": source["source_tree_sha256"],
        "science_registration": {"path": registration["draft_path"], "sha256": registration["draft_sha256"]},
        "candidate_freeze": {"path": registration["candidate_freeze_path"], "sha256": registration["candidate_freeze_sha256"]},
        "candidate_freeze_correction": {"path": registration["integrity_amendment_path"], "sha256": registration["integrity_amendment_sha256"]},
        "modal_execution": runner.expected_spec_modal_binding(value, digest),
    }
    runner.validate_scientific_binding(value, spec, digest)
    spec["modal_execution"]["contract_sha256"] = "b" * 64
    with pytest.raises(runner.ModalV10Error, match="does not bind"):
        runner.validate_scientific_binding(value, spec, digest)


def test_protected_manifest_is_absent_until_unlock_watcher_releases_it() -> None:
    value = contract()
    assert "manifests/sealed_final_indices.json" not in value["prepared_state"]["expected_files"]
    assert value["protected_final"]["release_policy"].startswith("opaque_upload_only_after")
    orchestration = function_source("orchestrate")
    assert orchestration.index("_write_and_verify_unlock") < orchestration.index(
        "protected_final_upload_authorized.json"
    )
    assert orchestration.index("protected_final_upload_authorized.json") < orchestration.index(
        "_wait_for_protected_upload"
    )
    assert orchestration.index("_wait_for_protected_upload") < orchestration.index(
        "collect_final.remote"
    )
    main = function_source("main")
    assert main.index("await_protected_final_release.remote") < main.index(
        "_upload_protected_final_after_unlock"
    )
    assert main.index("_upload_protected_final_after_unlock") < main.index(
        "record_protected_final_upload.remote"
    )
    preflight = function_source("_local_preflight")
    assert "_upload_protected_final_after_unlock" not in preflight
    assert "sealed_final_indices.json" not in preflight


def test_launcher_uses_existing_protocol_for_serial_final_and_never_creates_volume() -> None:
    source = (ROOT / "modal_confirmatory_v10_fast.py").read_text()
    assert "create_if_missing=False, version=2" in source
    assert "max_containers=MAX_PARALLEL_TRAINING_GPUS" in source
    assert "max_containers=MAX_PARALLEL_FINAL_GPUS" in source
    final = function_source("collect_final")
    assert "final_runner.run_final_collection(REMOTE_STATE)" in final
    assert "json.loads" not in function_source("_upload_protected_final_after_unlock")
    assert "batch.put_file(path, \"/manifests/sealed_final_indices.json\")" in function_source(
        "_upload_protected_final_after_unlock"
    )


def test_runtime_inventory_treats_exact_contract_as_separate_bound_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    contract_path = repository / "protocol_archive" / "contract.json"
    source_path = repository / "worker.py"
    contract_path.parent.mkdir(parents=True)
    contract_path.write_text('{"control": "contract"}\n')
    source_path.write_text("print('worker')\n")
    digest = sha256(contract_path)
    value = {
        "repository_path": "protocol_archive/contract.json",
        "runtime_source": {
            "files": {
                "worker.py": {
                    "sha256": sha256(source_path),
                    "size_bytes": source_path.stat().st_size,
                    "mode": 0o644,
                }
            }
        },
    }
    monkeypatch.setattr(runner, "REMOTE_REPO", repository)
    monkeypatch.setattr(runner, "REMOTE_CONTRACT_PATH", contract_path)
    monkeypatch.setenv("JLENS_V10_MODAL_CONTRACT_SHA256", digest)

    runner._verify_runtime_source(value)
    (repository / "unexpected.txt").write_text("unexpected\n")
    with pytest.raises(runner.ModalV10Error, match="runtime source inventory"):
        runner._verify_runtime_source(value)
