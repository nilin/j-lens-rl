from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
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


def _valid_v5_source_payloads() -> dict[str, object]:
    git_commit = "a" * 40
    curve_plot_bytes = b"tracked V5 curve plot fixture\n"
    receipt = {
        "app_id": v6.V5_MODAL_APP_ID,
        "claim_id": v6.V5_CLAIM_ID,
        "function_call_id": v6.V5_MODAL_FUNCTION_CALL_ID,
        "global_modal_gpu_limit": 1,
        "gpu_app_overlap_policy": v6.V5_GPU_APP_OVERLAP_POLICY,
        "gpu_type": "L40S",
        "max_parallel_gpu_workers": 1,
        "modal_app": v6.V5_MODAL_APP_NAME,
        "receipt_status": "present",
        "submitted_at_utc": v6.V5_LAUNCH_SUBMITTED_AT_UTC,
        "volume": v6.V5_VOLUME_NAME,
    }
    curve = {
        "protocol": v6.V5_PROTOCOL,
        "git_commit": git_commit,
        "registration_sha256": v6.V5_REGISTRATION_SHA256,
        "criterion": v6.CURVE_CRITERION,
        "predeclared_steps": [0, 2, 4, 6],
        "n_seeds": 8,
        "examples_per_seed": 400,
        "passed": False,
        "curve_plot": {
            "path": "/volume/v5/evidence/curve.png",
            "sha256": hashlib.sha256(curve_plot_bytes).hexdigest(),
        },
    }
    return {
        "attempt_claim": {
            "claim_id": v6.V5_CLAIM_ID,
            "git_commit": git_commit,
            "protocol": v6.V5_PROTOCOL,
            "registration_sha256": v6.V5_REGISTRATION_SHA256,
            "recipe_lock_sha256": v6.V5_RECIPE_LOCK_SHA256,
            "global_modal_gpu_limit": 1,
            "gpu_app_overlap_policy": v6.V5_GPU_APP_OVERLAP_POLICY,
            "operational_preflight": {
                "checked_at_utc": "2026-07-14T08:54:00+00:00",
                "exclusive_gpu_confirmation": (
                    "confirmed-no-other-modal-gpu-app-running"
                ),
                "global_modal_gpu_limit": 1,
                "active_other_modal_apps": [],
            },
        },
        "launch_receipt": receipt,
        "attempt_status": {
            "claim_id": v6.V5_CLAIM_ID,
            "stage": "curve_failed",
            "updated_at_utc": "2026-07-14T10:00:00+00:00",
            "curve": {**curve, "returncode": 2},
        },
        "curve_gate": curve,
        "curve_plot": curve_plot_bytes,
        "root_inventory": {
            "protocol": (
                "j-lens-rl-confirmatory-v5-emotional-terminal-inventory-v1"
            ),
            "claim_id": v6.V5_CLAIM_ID,
            "volume": v6.V5_VOLUME_NAME,
            "scope": "volume_root",
            "entries": sorted(
                [
                    "attempt_claim.json",
                    "attempt_status.json",
                    "evidence",
                    "exports",
                    "launch_receipt.json",
                    "manifests",
                    "protocol_state.json",
                    "reproducibility",
                    "runs",
                ]
            ),
        },
        "evidence_inventory": {
            "protocol": (
                "j-lens-rl-confirmatory-v5-emotional-terminal-inventory-v1"
            ),
            "claim_id": v6.V5_CLAIM_ID,
            "volume": v6.V5_VOLUME_NAME,
            "scope": "evidence_directory",
            "entries": sorted(
                [
                    "curve.png",
                    "curve_gate.json",
                    "durable_export_plan.json",
                    "evidence_bundle_inventory.json",
                    "git_closeout_candidate.json",
                ]
            ),
        },
        "durable_export_receipt": {
            "schema_version": 1,
            "archive_relative_path": (
                f"exports/v5_emotional_evidence_{v6.V5_CLAIM_ID}.zip"
            ),
            "sha256": "b" * 64,
            "size_bytes": 123456,
            "entry_count": 99,
            "evidence_inventory_sha256": "c" * 64,
        },
    }


def _valid_v5_closeout(source_evidence: dict[str, dict]) -> dict:
    return {
        "protocol": v6.CONDITIONAL_LAUNCH_PREDICATE["required_protocol"],
        "v5_registration_sha256": v6.V5_REGISTRATION_SHA256,
        "v5_infrastructure_amendment1_sha256": (
            v6.V5_INFRASTRUCTURE_AMENDMENT1_SHA256
        ),
        "v5_sealed_final_manifest_sha256": v6.V5_FINAL_SHA256,
        "v5_sealed_final_sorted_set_sha256": v6.V5_FINAL_SET_SHA256,
        "terminal_stage": "curve_failed",
        "final_unlocked_present": False,
        "final_collection_present": False,
        "evals_directory_present": False,
        "final_evaluation_labels": [],
        "sealed_comparison_present": False,
        "final_outcomes_unopened": True,
        "operational_identity": v6.CONDITIONAL_LAUNCH_PREDICATE[
            "required_operational_identity"
        ],
        "source_evidence": source_evidence,
    }


def _install_closeout(
    tmp_path: Path,
    monkeypatch,
    *,
    closeout_mutation=None,
    source_mutation=None,
) -> tuple[Path, dict, dict[str, Path]]:
    repo = tmp_path / "repo"
    evidence_dir = repo / "protocol_archive" / "v5_emotional_terminal_evidence"
    evidence_dir.mkdir(parents=True)
    source_payloads = _valid_v5_source_payloads()
    if source_mutation is not None:
        source_mutation(source_payloads)
    source_paths: dict[str, Path] = {}
    source_identities: dict[str, dict] = {}
    filenames = {
        name: Path(path).name
        for name, path in v6.CONDITIONAL_LAUNCH_PREDICATE[
            "required_source_evidence_paths"
        ].items()
    }
    for name, source_payload in source_payloads.items():
        source_path = evidence_dir / filenames[name]
        if isinstance(source_payload, bytes):
            source_path.write_bytes(source_payload)
        else:
            source_path.write_text(
                json.dumps(source_payload, indent=2, sort_keys=True) + "\n"
            )
        source_paths[name] = source_path
        source_identities[name] = {
            "path": source_path.relative_to(repo).as_posix(),
            "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        }
    payload = _valid_v5_closeout(source_identities)
    if closeout_mutation is not None:
        closeout_mutation(payload)
    path = repo / "protocol_archive" / "v5_emotional_terminal_closeout.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    monkeypatch.setattr(v6, "REPO", repo)
    monkeypatch.setattr(v6, "V5_TERMINAL_CLOSEOUT_PATH", path)
    monkeypatch.setattr(v6, "V5_TERMINAL_EVIDENCE_DIR", evidence_dir)
    monkeypatch.setattr(v6, "V5_TERMINAL_EVIDENCE_PATHS", source_paths)
    monkeypatch.setattr(
        v6,
        "git",
        lambda *args: args[-1],
    )
    return path, payload, source_paths


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
    path, _, source_paths = _install_closeout(tmp_path, monkeypatch)
    assert hashlib.sha256(source_paths["launch_receipt"].read_bytes()).hexdigest() == (
        v6.V5_LAUNCH_RECEIPT_SHA256
    )
    result = v6.verify_v5_launch_predicate()
    assert result["terminal_stage"] == "curve_failed"
    assert result["final_outcomes_unopened"] is True
    assert result["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["operational_identity"]["claim_id"] == v6.V5_CLAIM_ID
    assert result["source_evidence_sha256"] == {
        name: hashlib.sha256(source_path.read_bytes()).hexdigest()
        for name, source_path in source_paths.items()
    }


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
    _install_closeout(
        tmp_path,
        monkeypatch,
        closeout_mutation=lambda payload: payload.__setitem__(field, value),
    )
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
    def mutate(payloads):
        if root_entry:
            payloads["root_inventory"]["entries"].append(root_entry)
            payloads["root_inventory"]["entries"].sort()
        if evidence_entry:
            payloads["evidence_inventory"]["entries"].append(evidence_entry)
            payloads["evidence_inventory"]["entries"].sort()

    _install_closeout(tmp_path, monkeypatch, source_mutation=mutate)
    with pytest.raises(v6.ProtocolError, match="opened-final artifacts"):
        v6.verify_v5_launch_predicate()


@pytest.mark.parametrize(
    ("source_name", "mutation", "match"),
    [
        (
            "attempt_claim",
            lambda value: value.__setitem__("claim_id", "0" * 32),
            "exact amended claim",
        ),
        (
            "launch_receipt",
            lambda value: value.__setitem__("app_id", "ap-wrong"),
            "registered SHA-256|committed bytes changed",
        ),
        (
            "attempt_status",
            lambda value: value.__setitem__("stage", "complete"),
            "status/curve bytes",
        ),
        (
            "curve_gate",
            lambda value: value.__setitem__("passed", True),
            "status/curve bytes",
        ),
    ],
)
def test_conditional_v5_predicate_rejects_semantically_false_tracked_sources(
    tmp_path, monkeypatch, source_name, mutation, match
) -> None:
    def mutate(payloads):
        mutation(payloads[source_name])

    _install_closeout(tmp_path, monkeypatch, source_mutation=mutate)
    with pytest.raises(v6.ProtocolError, match=match):
        v6.verify_v5_launch_predicate()


def test_conditional_v5_predicate_rejects_fake_unbacked_hex_identity(
    tmp_path, monkeypatch
) -> None:
    _, payload, _ = _install_closeout(tmp_path, monkeypatch)
    payload["source_evidence"]["attempt_status"]["sha256"] = "1" * 64
    v6.V5_TERMINAL_CLOSEOUT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    with pytest.raises(v6.ProtocolError, match="committed bytes changed"):
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
        v6.CORRECTED_OPERATOR_KNOWLEDGE_BOUNDARY
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


def test_v2_correction_preserves_every_registered_scientific_and_wandb_field() -> None:
    v1_path = (
        ROOT
        / "protocol_archive"
        / "v6_celebration_registration_v1_superseded.json"
    )
    correction_path = (
        ROOT / "protocol_archive" / "v6_celebration_prelaunch_correction1.json"
    )
    v2_path = ROOT / "protocol_archive" / "v6_celebration_registration.json"
    assert v6.sha256_file(v1_path) == v6.SUPERSEDED_REGISTRATION_SHA256
    assert v6.sha256_file(correction_path) == v6.PRELAUNCH_CORRECTION1_SHA256
    v1 = json.loads(v1_path.read_text())
    correction = json.loads(correction_path.read_text())
    v2 = json.loads(v2_path.read_text())
    fields = correction["scientific_projection"]["fields"]
    v1_projection = {field: v1[field] for field in fields}
    v2_projection = {field: v2[field] for field in fields}
    assert v1_projection == v2_projection
    assert v6.canonical_sha256(v1_projection) == (
        v6.SUPERSEDED_SCIENTIFIC_PROJECTION_SHA256
    )
    assert correction["knowledge_boundary_at_correction"] == {
        "root_operator": (
            "root had seen partial results for six V5 seeds before this correction"
        ),
        "timing": (
            "the correction was frozen before root saw a V5 aggregate or terminal result"
        ),
        "v6_outcomes": (
            "no V6 training, curve, control, or sealed-final outcome existed"
        ),
        "selection_effect": (
            "the additional V5 partials did not alter the already selected "
            "celebration recipe, curve nodes, seeds, controls, data repartition, "
            "acceptance rules, or W&B identities"
        ),
    }


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


def test_metric_schema_describes_the_weighted_combined_reward_range() -> None:
    lock = json.loads(
        (
            ROOT / "protocol_archive" / "v6_celebration_selected_recipe.json"
        ).read_text()
    )
    recipe = lock["resolved_training_config"]
    schema = v6.metric_schema(
        recipe["target_words"], recipe["updates"], recipe["score_components"]
    )
    named = schema["series"]["intrinsic_named_weighted_reward_mean"]
    total = schema["series"]["intrinsic_reward_mean"]
    assert named["range"] == total["range"] == [-6.25, 6.25]
    assert "weighted" in named["unit"]
    assert "not an individual component" in named["definition"]
    assert "weighted combined" in total["definition"]


def test_generated_v6_config_can_be_derived_as_a_safe_nonclaim_replay(
    tmp_path,
) -> None:
    from jlens_rl.train import configure_reproduction_replay

    registration = _valid_registration()
    lock_path = ROOT / "protocol_archive" / "v6_celebration_selected_recipe.json"
    lock = json.loads(lock_path.read_text())
    original = v6.generated_configs(
        registration,
        "a" * 64,
        lock["resolved_training_config"],
        v6.sha256_file(lock_path),
    )["jlens_seed176"]
    fresh = tmp_path / "outside-confirmatory" / "jlens_seed176"
    replay = configure_reproduction_replay(
        original, output_dir=str(fresh), wandb_mode="disabled"
    )
    assert replay["output_dir"] == str(fresh)
    assert replay["evidence_eligibility"] == "non_claim_reproduction"
    assert replay["reproduction_source"]["original_output_dir"] == (
        ".confirmatory/v6/runs/jlens_seed176"
    )
    assert "wandb_run_id" not in replay
    with pytest.raises(ValueError, match="immutable V6 state"):
        configure_reproduction_replay(
            original,
            output_dir=".confirmatory/v6/replay-forbidden",
            wandb_mode="disabled",
        )


def test_actual_generated_v6_replay_cli_smoke_exits_before_any_outcome(
    tmp_path,
) -> None:
    registration = _valid_registration()
    lock_path = ROOT / "protocol_archive" / "v6_celebration_selected_recipe.json"
    lock = json.loads(lock_path.read_text())
    original = v6.generated_configs(
        registration,
        "a" * 64,
        lock["resolved_training_config"],
        v6.sha256_file(lock_path),
    )["jlens_seed176"]
    config_path = tmp_path / "generated_v6_jlens_seed176.json"
    config_path.write_text(json.dumps(original, indent=2, sort_keys=True) + "\n")
    replay_output = tmp_path / "fresh-replay-output"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "jlens_rl.train",
            "--config",
            str(config_path),
            "--reproduction-replay",
            "--output-dir",
            str(replay_output),
            "--wandb-mode",
            "disabled",
            "--replay-config-smoke-test",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    result = json.loads(completed.stdout)
    assert result["status"] == "valid_non_claim_replay_config"
    assert result["output_dir"] == str(replay_output)
    assert result["evidence_eligibility"] == "non_claim_reproduction"
    assert not replay_output.exists()


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
