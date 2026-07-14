from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "scripts" / "confirmatory_v6_protocol.py"


def _load_protocol():
    spec = importlib.util.spec_from_file_location("confirmatory_v6_test", PROTOCOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v6 = _load_protocol()


def _valid_registration() -> dict:
    registration = v6.registration_template()
    registration["frozen_at_utc"] = "2026-07-14T09:30:00Z"
    registration["selected_recipe_lock"]["sha256"] = v6.sha256_file(
        ROOT / "protocol_archive" / "v6_celebration_selected_recipe.json"
    )
    registration["fixed_updates"] = 10
    registration["curve_gate"]["steps"] = [0, 4, 6, 10]
    registration["final_collection"]["terminal_adapter_step"] = 10
    return registration


def _valid_v5_closeout() -> dict:
    return {
        "protocol": v6.CONDITIONAL_LAUNCH_PREDICATE["required_protocol"],
        "v5_registration_sha256": v6.V5_REGISTRATION_SHA256,
        "v5_sealed_final_manifest_sha256": v6.V5_FINAL_SHA256,
        "v5_sealed_final_sorted_set_sha256": v6.V5_FINAL_SET_SHA256,
        "terminal_stage": "curve_failed",
        "final_unlocked_present": False,
        "final_collection_present": False,
        "evals_directory_present": False,
        "final_evaluation_labels": [],
        "sealed_comparison_present": False,
        "final_outcomes_unopened": True,
        "volume_inventory": {
            "root_entries": [
                "attempt_claim.json",
                "attempt_status.json",
                "evidence",
                "manifests",
                "protocol_state.json",
                "reproducibility",
                "runs",
            ],
            "evidence_entries": ["curve.png", "curve_gate.json"],
        },
        "source_evidence_sha256": {
            "attempt_status": "1" * 64,
            "curve_gate": "2" * 64,
            "volume_inventory": "3" * 64,
            "durable_export": "4" * 64,
        },
    }


def _install_closeout(tmp_path: Path, monkeypatch, payload: dict) -> Path:
    repo = tmp_path / "repo"
    path = repo / "protocol_archive" / "v5_emotional_terminal_closeout.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    monkeypatch.setattr(v6, "REPO", repo)
    monkeypatch.setattr(v6, "V5_TERMINAL_CLOSEOUT_PATH", path)
    monkeypatch.setattr(
        v6,
        "git",
        lambda *args: "protocol_archive/v5_emotional_terminal_closeout.json",
    )
    return path


def _reconstruct_v4_parent() -> tuple[list[int], list[int]]:
    path = ROOT / "scripts" / "confirmatory_protocol.py"
    spec = importlib.util.spec_from_file_location("confirmatory_v4_for_v6_test", path)
    assert spec is not None and spec.loader is not None
    v4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v4)

    historical, v1_historical, dataset_size = v4.reconstruct_historical_indices()
    fresh = set(range(dataset_size)) - set(historical)
    v1_fresh = sorted(
        set(range(dataset_size)) - set(v1_historical),
        key=lambda index: v4.allocation_key(index, v4.V1_ALLOCATION_SALT),
    )
    retired_v1_curve = v1_fresh[200:600]
    v2_allocatable = sorted(
        fresh - set(retired_v1_curve),
        key=lambda index: v4.allocation_key(index, v4.V2_ALLOCATION_SALT),
    )
    curve_end = v4.V2_SPLIT_SIZES["curve_indices.json"]
    sealed_end = curve_end + v4.V2_SPLIT_SIZES["sealed_final_indices.json"]
    v2 = {
        "curve_indices.json": v2_allocatable[:curve_end],
        "sealed_final_indices.json": v2_allocatable[curve_end:sealed_end],
        "future_reserve_indices.json": v2_allocatable[sealed_end:],
    }
    v3, _ = v4.reallocate_v2_parent(v2)
    v4_split, _ = v4.reallocate_v3_parent(v3)
    return v4_split["sealed_final_indices.json"], v4_split[
        "future_reserve_indices.json"
    ]


def test_v6_isolated_exact_recipe_seeds_labels_and_one_gpu() -> None:
    assert v6.STATE_DIR == ROOT / ".confirmatory" / "v6"
    assert v6.SEEDS == tuple(range(176, 184))
    assert v6.MAX_GPU_CONTAINERS == v6.GLOBAL_MODAL_GPU_LIMIT == 1
    assert len(v6.FINAL_LABELS) == 17
    assert v6.FINAL_LABELS[1] == "jlens_seed176"
    assert v6.FINAL_LABELS[-1] == "signflip_seed183"
    assert v6.VOLUME_NAME == (
        "j-lens-rl-confirmatory-v6-celebration-taper-20260714a"
    )
    lock = json.loads(
        (ROOT / "protocol_archive" / "v6_celebration_selected_recipe.json").read_text()
    )
    recipe = lock["resolved_training_config"]
    assert recipe["target_words"] == ["yay", "great", "success", "nice"]
    assert all("solved" not in word.lower() for word in recipe["target_words"])
    assert recipe["updates"] == recipe["save_every"] == 10
    assert recipe["validation_steps"] == [4, 6, 10]
    assert recipe["kl_beta"] == 0.02
    assert recipe["reward_type"] == "jlens"


def test_v6_split_exactly_repartitions_v5_unopened_parent_and_retains_reserve() -> None:
    v4_parent, reserve = _reconstruct_v4_parent()
    v5_curve, v5_parent = v6.reconstruct_v5_split(v4_parent)
    assert len(v5_curve) == 400
    assert len(v5_parent) == 1300
    assert v6.serialized_json_sha256(v6.manifest_payload(v5_parent)) == (
        v6.V5_FINAL_SHA256
    )
    assert v6.canonical_sha256(sorted(v5_parent)) == v6.V5_FINAL_SET_SHA256
    curve, final = v6.allocate_v6(v5_parent)
    assert len(curve) == 400 and len(final) == 900
    assert set(curve).isdisjoint(final)
    assert set(curve) | set(final) == set(v5_parent)
    assert v6.serialized_json_sha256(v6.manifest_payload(curve)) == (
        v6.V6_CURVE_SHA256
    )
    assert v6.serialized_json_sha256(v6.manifest_payload(final)) == (
        v6.V6_FINAL_SHA256
    )
    assert v6.canonical_sha256(sorted(curve)) == v6.V6_CURVE_SET_SHA256
    assert v6.canonical_sha256(sorted(final)) == v6.V6_FINAL_SET_SHA256
    assert len(reserve) == 64
    assert v6.serialized_json_sha256(v6.manifest_payload(reserve)) == (
        v6.RESERVE_SHA256
    )
    assert set(reserve).isdisjoint(v5_parent)


def test_selection_closeout_and_lock_bind_only_committed_seed167_evidence() -> None:
    closeout_path = (
        ROOT / "protocol_archive" / "v6_celebration_selection_closeout.json"
    )
    lock_path = ROOT / "protocol_archive" / "v6_celebration_selected_recipe.json"
    closeout = json.loads(closeout_path.read_text())
    lock = json.loads(lock_path.read_text())
    assert v6.sha256_file(closeout_path) == v6.SELECTION_CLOSEOUT_SHA256
    assert closeout["operator_knowledge_boundary"] == v6.OPERATOR_KNOWLEDGE_BOUNDARY
    assert closeout["scientific_status"]["v5_outcomes_inspected_by_selecting_agent"] is False
    assert closeout["observed_exploratory_source"]["registered_v6_nodes"] == {
        "steps": [0, 4, 6, 10],
        "exact_match": [0.3825, 0.385, 0.3925, 0.405],
    }
    for identity in closeout["source_evidence"].values():
        assert v6.sha256_file(ROOT / identity["path"]) == identity["sha256"]
    source_path = ROOT / lock["selection_provenance"]["source_resolved_config"][
        "path"
    ]
    calibration_path = lock["selection_provenance"]["source_calibration"]["path"]
    expected = v6.expected_selected_celebration_recipe(
        json.loads(source_path.read_text()), calibration_path
    )
    assert lock["resolved_training_config"] == expected
    assert lock["selection_provenance"]["operator_knowledge_boundary"] == (
        v6.OPERATOR_KNOWLEDGE_BOUNDARY
    )


def test_conditional_v5_predicate_accepts_only_curve_failed_unopened_closeout(
    tmp_path, monkeypatch
) -> None:
    payload = _valid_v5_closeout()
    path = _install_closeout(tmp_path, monkeypatch, payload)
    result = v6.verify_v5_launch_predicate()
    assert result["terminal_stage"] == "curve_failed"
    assert result["final_outcomes_unopened"] is True
    assert result["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("terminal_stage", "complete"),
        ("final_unlocked_present", True),
        ("final_collection_present", True),
        ("evals_directory_present", True),
        ("final_evaluation_labels", ["base"]),
        ("sealed_comparison_present", True),
        ("final_outcomes_unopened", False),
    ],
)
def test_conditional_v5_predicate_cancels_on_any_open_or_nonfailure_state(
    tmp_path, monkeypatch, field: str, value
) -> None:
    payload = _valid_v5_closeout()
    payload[field] = value
    _install_closeout(tmp_path, monkeypatch, payload)
    with pytest.raises(v6.ProtocolError, match="cancelled"):
        v6.verify_v5_launch_predicate()


@pytest.mark.parametrize(
    ("root_entry", "evidence_entry"),
    [
        ("final_unlocked.json", None),
        ("final_collection.json", None),
        ("evals", None),
        (None, "sealed_comparison.json"),
    ],
)
def test_conditional_v5_predicate_cancels_on_forbidden_inventory(
    tmp_path, monkeypatch, root_entry: str | None, evidence_entry: str | None
) -> None:
    payload = _valid_v5_closeout()
    if root_entry:
        payload["volume_inventory"]["root_entries"].append(root_entry)
    if evidence_entry:
        payload["volume_inventory"]["evidence_entries"].append(evidence_entry)
    _install_closeout(tmp_path, monkeypatch, payload)
    with pytest.raises(v6.ProtocolError, match="opened-final artifacts"):
        v6.verify_v5_launch_predicate()


def test_conditional_v5_predicate_fails_closed_when_closeout_missing(
    tmp_path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    monkeypatch.setattr(v6, "REPO", repo)
    monkeypatch.setattr(
        v6,
        "V5_TERMINAL_CLOSEOUT_PATH",
        repo / "protocol_archive" / "v5_emotional_terminal_closeout.json",
    )
    with pytest.raises(v6.ProtocolError, match="cancelled/inert"):
        v6.verify_v5_launch_predicate()


def test_registration_freezes_condition_nodes_controls_wandb_and_cancellation() -> None:
    registration = _valid_registration()
    assert registration["curve_gate"]["steps"] == [0, 4, 6, 10]
    assert registration["fixed_updates"] == 10
    assert registration["seeds"] == list(range(176, 184))
    assert registration["conditional_launch_predicate"] == (
        v6.CONDITIONAL_LAUNCH_PREDICATE
    )
    assert registration["operator_knowledge_boundary"] == (
        v6.OPERATOR_KNOWLEDGE_BOUNDARY
    )
    assert registration["wandb"]["run_ids"] == v6.WANDB_RUN_IDS
    assert len(set(registration["wandb"]["run_ids"].values())) == 16
    assert registration["execution"]["max_parallel_gpu_workers"] == 1
    assert registration["execution"]["global_modal_gpu_limit"] == 1
    v6._validate_registration_shape(registration)
    for mutation, match in (
        (("curve_gate", "steps", [0, 2, 4, 6]), "curve criterion"),
        (("fixed_updates", None, 25), "step-10"),
        (("seeds", None, list(range(168, 176))), "176 through 183"),
        (("execution", "max_parallel_gpu_workers", 2), "exact V6 runner"),
    ):
        changed = json.loads(json.dumps(registration))
        outer, inner, value = mutation
        if inner is None:
            changed[outer] = value
        else:
            changed[outer][inner] = value
        if outer == "fixed_updates":
            changed["final_collection"]["terminal_adapter_step"] = value
        with pytest.raises(v6.ProtocolError, match=match):
            v6._validate_registration_shape(changed)


def test_committed_registration_is_the_exact_completed_template() -> None:
    registration_path = (
        ROOT / "protocol_archive" / "v6_celebration_registration.json"
    )
    registration = json.loads(registration_path.read_text())
    expected = _valid_registration()
    expected["frozen_at_utc"] = registration["frozen_at_utc"]
    assert registration == expected
    v6._validate_registration_shape(registration)


def test_generated_configs_use_exact_wandb_ids_and_matched_signflips() -> None:
    registration = _valid_registration()
    lock_path = ROOT / "protocol_archive" / "v6_celebration_selected_recipe.json"
    lock = json.loads(lock_path.read_text())
    configs = v6.generated_configs(
        registration,
        "a" * 64,
        lock["resolved_training_config"],
        v6.sha256_file(lock_path),
    )
    assert len(configs) == 17
    for seed in v6.SEEDS:
        semantic = configs[f"jlens_seed{seed}"]
        control = configs[f"signflip_seed{seed}"]
        assert semantic["seed"] == control["seed"] == seed
        assert semantic["target_words"] == ["yay", "great", "success", "nice"]
        assert semantic["updates"] == semantic["save_every"] == 10
        assert semantic["validation_steps"] == [4, 6, 10]
        assert semantic["curve_manifest_sha256"] == v6.V6_CURVE_SHA256
        assert semantic["wandb_run_id"] == v6.WANDB_RUN_IDS[f"jlens_seed{seed}"]
        assert control["wandb_run_id"] == v6.WANDB_RUN_IDS[
            f"signflip_seed{seed}"
        ]
        assert control["score_components"] == v6.negate_score_components(
            semantic["score_components"]
        )
        assert [item["weight"] for item in semantic["score_components"]] == [
            1.0,
            0.25,
        ]
        assert [item["weight"] for item in control["score_components"]] == [
            -1.0,
            -0.25,
        ]
    sealed = configs["sealed_eval"]
    assert sealed["validation_examples"] == 900
    assert sealed["evaluation_indices_path"] == (
        ".confirmatory/v6/manifests/sealed_final_indices.json"
    )


def test_modal_and_shell_runners_are_serial_fresh_and_protocol_gated() -> None:
    modal_source = (ROOT / "modal_confirmatory_v6.py").read_text()
    shell_source = (ROOT / "run_confirmatory_v6.sh").read_text()
    assert 'VOLUME_NAME = "j-lens-rl-confirmatory-v6-celebration-taper-20260714a"' in modal_source
    assert "SEEDS = tuple(range(176, 184))" in modal_source
    assert "MAX_GPU_CONTAINERS = 1" in modal_source
    assert "GLOBAL_MODAL_GPU_LIMIT = 1" in modal_source
    assert "_serial_gpu_waves" in modal_source
    assert '_protocol("verify")' in modal_source
    assert 'if condition == "signflip":' in modal_source
    assert '_protocol("verify-curve")' in modal_source
    assert '"reproducibility": 7' in modal_source
    assert "j-lens-rl-confirmatory-v6-celebration-taper-image-v1" in modal_source
    assert "SEEDS=(176 177 178 179 180 181 182 183)" in shell_source
    assert '"$STATE/configs/jlens_seed176.json"' in shell_source
    assert "solved_seed" not in modal_source + shell_source
