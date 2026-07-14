from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "scripts" / "confirmatory_v5_protocol.py"


def _load_protocol():
    spec = importlib.util.spec_from_file_location("confirmatory_v5_test", PROTOCOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v5 = _load_protocol()


def _load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    base = config.pop("base", None)
    if base:
        parent = _load_config(path.parent / base)
        parent.update(config)
        return parent
    return config


def _registration(
    curve_steps: list[int] | None = None, fixed_updates: int = 25
) -> dict:
    registration = v5.registration_template()
    registration["frozen_at_utc"] = "2026-07-14T07:00:00Z"
    registration["curve_gate"]["steps"] = curve_steps or [0, 15, 20, 25]
    registration["fixed_updates"] = fixed_updates
    registration["final_collection"]["terminal_adapter_step"] = fixed_updates
    registration["wandb"]["entity"] = "nilinabra-spare-time"
    registration["wandb"]["group"] = "confirm-v5-emotional"
    registration["wandb"]["run_prefix"] = "confirm-v5-emotional-yay"
    return registration


def _emotional_recipe() -> dict:
    recipe = _load_config(ROOT / "configs" / "single_word_screen_yay.json")
    # A selected lock must use a repository-relative, byte-pinned calibration
    # path. Artifact existence is tested during preparation, not config derivation.
    recipe["calibration_path"] = "protocol_archive/calibrations/yay.json"
    recipe["calibration_sha256"] = "b" * 64
    return recipe


def test_v5_isolated_state_new_seeds_and_exact_final_labels() -> None:
    assert v5.STATE_DIR == ROOT / ".confirmatory" / "v5"
    assert v5.SEEDS == tuple(range(168, 176))
    assert v5.MAX_GPU_CONTAINERS == 1
    assert v5.GLOBAL_MODAL_GPU_LIMIT == 1
    assert v5.FINAL_LABELS == (
        "base",
        *(f"jlens_seed{seed}" for seed in range(168, 176)),
        *(f"signflip_seed{seed}" for seed in range(168, 176)),
    )
    assert len(v5.FINAL_LABELS) == 17
    source = PROTOCOL_PATH.read_text()
    assert 'REPO / ".confirmatory" / "v5"' in source
    assert 'REPO / ".confirmatory"\n' not in source


def test_v5_allocation_reproduces_untouched_400_1300_split() -> None:
    parent_path = ROOT / ".confirmatory" / "manifests" / "sealed_final_indices.json"
    parent = json.loads(parent_path.read_text())["indices"]
    curve, final = v5.allocate_v5(parent)
    assert len(curve) == 400
    assert len(final) == 1300
    assert set(curve).isdisjoint(final)
    assert set(curve) | set(final) == set(parent)
    assert v5.canonical_sha256(sorted(curve)) == v5.V5_CURVE_SET_SHA256
    assert v5.canonical_sha256(sorted(final)) == v5.V5_FINAL_SET_SHA256
    assert v5.sha256_file(parent_path) == v5.V4_PARENT_SHA256
    assert v5.sha256_file(
        ROOT / ".confirmatory" / "manifests" / "future_reserve_indices.json"
    ) == v5.RESERVE_SHA256


def test_curve_nodes_are_registration_required_not_chosen_by_code() -> None:
    template = v5.registration_template()
    assert template["curve_gate"]["steps"] is None
    assert template["fixed_updates"] is None
    assert template["final_collection"]["terminal_adapter_step"] is None
    assert v5.validate_curve_steps(
        [0, 15, 20, 25], [2, 4, 6, 10, 15, 20, 25], 25
    ) == (0, 15, 20, 25)
    assert v5.validate_curve_steps(
        [0, 2, 4, 6], [2, 4, 6, 10, 15, 20, 25], 6
    ) == (0, 2, 4, 6)
    assert v5.curve_means_pass([0.38, 0.39, 0.39, 0.40])
    assert not v5.curve_means_pass([0.38, 0.38, 0.39, 0.40])
    assert not v5.curve_means_pass([0.38, 0.39, 0.385, 0.40])
    with pytest.raises(v5.ProtocolError, match="exactly four"):
        v5.validate_curve_steps(None, [15, 20, 25], 25)
    with pytest.raises(v5.ProtocolError, match="strictly increasing"):
        v5.validate_curve_steps([0, 20, 15, 25], [15, 20, 25], 25)
    with pytest.raises(v5.ProtocolError, match="not all present"):
        v5.validate_curve_steps([0, 5, 10, 15], [15, 20, 25], 25)


def test_recipe_lock_template_binds_exact_joy_source_and_only_declared_transform() -> None:
    template = v5.recipe_lock_template()
    provenance = template["selection_provenance"]
    assert provenance["source_resolved_config"]["sha256"] == (
        v5.JOY_SOURCE_CONFIG_SHA256
    )
    assert provenance["source_calibration"]["sha256"] == (
        v5.JOY_CALIBRATION_SHA256
    )
    assert provenance["source_evidence_sha256"] == (
        v5.JOY_SELECTION_EVIDENCE_SHA256
    )
    assert provenance["declared_transformations"] == (
        v5.JOY_DECLARED_TRANSFORMATIONS
    )
    source = {
        "target_words": ["joy"],
        "score_components": [
            {
                "aggregation": "mean",
                "end_fraction": 1.0,
                "layer": 8,
                "start_fraction": 0.5,
                "weight": 1.0,
            }
        ],
        "score_stride": 5,
        "learning_rate": 3e-6,
        "lr_scheduler_type": "constant",
        "updates": 25,
        "save_every": 25,
        "validation_steps": [2, 4, 6, 10, 15, 20, 25],
        "seed": 167,
        "output_dir": "/single_word_screen/runs/joy",
        "run_name": "single-word-screen-joy-positive-seed167",
        "calibration_path": "/single_word_screen/artifacts/joy_calibration.json",
        "calibration_sha256": v5.JOY_CALIBRATION_SHA256,
        "lens_sha256": "178a9671cbf41882135807bde59b828e36c6f8f98b32c809ea3346860aad10dc",
        "unchanged_field": "must survive byte-for-byte",
    }
    transformed = v5.expected_selected_joy_recipe(
        source, "protocol_archive/calibrations/joy.json"
    )
    changed = {
        key for key in source | transformed if source.get(key) != transformed.get(key)
    }
    assert changed == {"updates", "save_every", "validation_steps", "calibration_path"}
    assert transformed["updates"] == transformed["save_every"] == 6
    assert transformed["validation_steps"] == [2, 4, 6]
    bad = dict(source, learning_rate=1e-5)
    with pytest.raises(v5.ProtocolError, match="exact joy U5"):
        v5.expected_selected_joy_recipe(bad, "protocol_archive/calibrations/joy.json")


def test_frozen_joy_source_closeout_and_recipe_bind_exact_archived_bytes() -> None:
    closeout_path = ROOT / "protocol_archive" / "joy_v5_selection_closeout.json"
    lock_path = ROOT / "protocol_archive" / "v5_emotional_selected_recipe.json"
    closeout = json.loads(closeout_path.read_text())
    lock = json.loads(lock_path.read_text())
    assert closeout["scientific_status"] == {
        "confirmatory": False,
        "evidence_role": "adaptive/exploratory source evidence for prospectively registered V5; not a V5 outcome",
        "selection_was_outcome_informed": True,
        "v5_outcomes_inspected_at_closeout": False,
    }
    assert closeout["wandb_identity"]["run_id"] == "5m3mwx9h"
    assert closeout["observed_exploratory_screen"]["exact_match"] == [
        0.3825,
        0.39,
        0.39,
        0.41,
    ]
    evidence = closeout["source_evidence"]
    for name in (
        "resolved_config",
        "calibration",
        "run_manifest",
        "screen_result",
        "validation_history",
        "log_history",
    ):
        assert v5.sha256_file(ROOT / evidence[name]["path"]) == evidence[name][
            "sha256"
        ]
    assert {
        name: identity["sha256"] for name, identity in evidence.items()
    } == v5.JOY_SELECTION_EVIDENCE_SHA256
    source_path = ROOT / lock["selection_provenance"]["source_resolved_config"][
        "path"
    ]
    calibration_path = lock["selection_provenance"]["source_calibration"]["path"]
    assert lock["selection_provenance"]["source_closeout"] == {
        "path": closeout_path.relative_to(ROOT).as_posix(),
        "sha256": v5.sha256_file(closeout_path),
    }
    assert lock["resolved_training_config"] == v5.expected_selected_joy_recipe(
        json.loads(source_path.read_text()), calibration_path
    )
    assert lock["artifact_sha256"] == {
        "lens": lock["resolved_training_config"]["lens_sha256"],
        "calibration": lock["resolved_training_config"]["calibration_sha256"],
    }


@pytest.mark.parametrize(
    "words",
    [["solved"], ["Solved"], ["UNSOLVED"], ["yay", "pre-solved-post"]],
)
def test_solved_is_rejected_anywhere_in_target_words(words: list[str]) -> None:
    with pytest.raises(v5.ProtocolError, match="retired"):
        v5.validate_target_words(words)
    assert v5.validate_target_words(["yay", "joy", "damn"]) == [
        "yay",
        "joy",
        "damn",
    ]


def test_matched_control_negates_every_and_only_component_weight() -> None:
    treatment = [
        {
            "layer": 8,
            "start_fraction": 0.5,
            "end_fraction": 0.75,
            "aggregation": "mean",
            "weight": 1.25,
        },
        {
            "layer": 14,
            "start_fraction": 0.75,
            "end_fraction": 1.0,
            "aggregation": "last",
            "weight": -0.5,
        },
    ]
    control = v5.negate_score_components(treatment)
    assert [item["weight"] for item in control] == [-1.25, 0.5]
    for original, flipped in zip(treatment, control):
        assert {key: value for key, value in original.items() if key != "weight"} == {
            key: value for key, value in flipped.items() if key != "weight"
        }
    assert treatment[0]["weight"] == 1.25


def test_generated_configs_freeze_emotional_recipe_controls_and_wandb() -> None:
    registration = _registration()
    recipe = _emotional_recipe()
    configs = v5.generated_configs(registration, "a" * 64, recipe, "e" * 64)
    assert len(configs) == 17
    run_ids = set()
    for seed in v5.SEEDS:
        semantic = configs[f"jlens_seed{seed}"]
        control = configs[f"signflip_seed{seed}"]
        assert semantic["seed"] == control["seed"] == seed
        assert semantic["target_words"] == control["target_words"] == ["yay"]
        assert semantic["validation_examples"] == 400
        assert semantic["validation_indices_path"] == (
            ".confirmatory/v5/manifests/curve_indices.json"
        )
        assert semantic["reserved_train_indices_path"] == (
            ".confirmatory/v5/manifests/train_exclusions.json"
        )
        assert semantic["calibration_path"] == (
            ".confirmatory/v5/frozen_artifacts/calibration.json"
        )
        assert control["score_components"] == v5.negate_score_components(
            semantic["score_components"]
        )
        assert semantic["wandb_resume"] == control["wandb_resume"] == "never"
        assert semantic["wandb_group"] == control["wandb_group"]
        assert semantic["wandb_run_id"] != control["wandb_run_id"]
        run_ids.update([semantic["wandb_run_id"], control["wandb_run_id"]])

        allowed_differences = {
            "score_components",
            "output_dir",
            "run_name",
            "wandb_run_id",
            "wandb_url",
            "registered_command",
        }
        assert {
            key for key in semantic if semantic.get(key) != control.get(key)
        } == allowed_differences
    assert len(run_ids) == 16
    sealed = configs["sealed_eval"]
    assert sealed["validation_examples"] == 1300
    assert sealed["evaluation_indices_path"] == (
        ".confirmatory/v5/manifests/sealed_final_indices.json"
    )
    assert sealed["min_new_tokens"] == 0


@pytest.mark.parametrize(
    ("config_name", "registered_steps", "fixed_updates"),
    [
        ("single_word_screen_joy.json", [0, 2, 4, 6], 6),
        ("word_explore_celebration_taper.json", [0, 15, 20, 25], 25),
    ],
)
def test_generator_is_recipe_and_curve_node_agnostic(
    config_name: str, registered_steps: list[int], fixed_updates: int
) -> None:
    recipe = _load_config(ROOT / "configs" / config_name)
    recipe["calibration_path"] = "protocol_archive/calibrations/selected.json"
    recipe["calibration_sha256"] = "c" * 64
    recipe["updates"] = fixed_updates
    recipe["save_every"] = fixed_updates
    recipe["validation_steps"] = [
        step for step in recipe["validation_steps"] if step <= fixed_updates
    ]
    registration = _registration(registered_steps, fixed_updates)
    configs = v5.generated_configs(registration, "d" * 64, recipe, "e" * 64)
    assert configs["jlens_seed168"]["target_words"] == recipe["target_words"]
    assert configs["jlens_seed168"]["score_components"] == recipe[
        "score_components"
    ]
    assert configs["signflip_seed168"]["score_components"] == (
        v5.negate_score_components(recipe["score_components"])
    )
    assert registration["curve_gate"]["steps"] == registered_steps
    assert configs["jlens_seed168"]["updates"] == fixed_updates
    assert configs["jlens_seed168"]["save_every"] == fixed_updates
    assert configs["sealed_eval"]["updates"] == fixed_updates


def test_calibration_and_token_audit_generalizes_to_emotional_words(tmp_path) -> None:
    from jlens import JacobianLens
    from transformers import AutoTokenizer

    from jlens_rl.reward import single_token_ids

    lens_path = ROOT / "artifacts" / "qwen25_05b_solved_lens.pt"
    lens = JacobianLens.load(str(lens_path))
    tokenizer = AutoTokenizer.from_pretrained(v5.MODEL_NAME, revision=v5.MODEL_REVISION)
    words = ["yay", "joy"]
    calibration = {
        "mean": -12.0,
        "std": 3.0,
        "token_ids": single_token_ids(tokenizer, words),
        "target_words": words,
        "layers": list(lens.source_layers),
        "model": v5.MODEL_NAME,
        "model_revision": v5.MODEL_REVISION,
        "lens_sha256": v5.sha256_file(lens_path),
    }
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps(calibration, sort_keys=True) + "\n")
    recipe = _emotional_recipe()
    recipe["target_words"] = words
    recipe["lens_sha256"] = v5.sha256_file(lens_path)
    recipe["calibration_sha256"] = v5.sha256_file(calibration_path)
    identity = v5._validate_frozen_artifacts(
        recipe,
        {"lens": lens_path, "calibration": calibration_path},
    )
    assert identity["target_words"] == words
    assert identity["token_ids"] == calibration["token_ids"]
    assert identity["lens_layers"] == list(lens.source_layers)

    calibration["token_ids"] = calibration["token_ids"][:-1]
    calibration_path.write_text(json.dumps(calibration, sort_keys=True) + "\n")
    recipe["calibration_sha256"] = v5.sha256_file(calibration_path)
    with pytest.raises(v5.ProtocolError, match="calibration metadata"):
        v5._validate_frozen_artifacts(
            recipe,
            {"lens": lens_path, "calibration": calibration_path},
        )


def test_metric_schema_is_explicit_and_byte_pinned_into_every_run() -> None:
    registration = _registration([0, 2, 4, 6])
    recipe = _emotional_recipe()
    schema = v5.metric_schema(recipe["target_words"], 25)
    assert schema["step_axes"]["optimizer_update"]["wandb_metric"] == (
        "train/global_step"
    )
    assert schema["series"]["validation_exact_match"] == {
        "local_file": "validation_history.jsonl",
        "local_field": "exact_match",
        "wandb_metric": "train/validation/exact_match",
        "step_axis": "optimizer_update",
        "unit": "fraction of 400 fixed examples",
        "range": [0.0, 1.0],
        "definition": "mean deterministic greedy GSM8K exact-answer correctness on the registered curve manifest",
    }
    observed = schema["observed_history_scalar_series"]
    assert len(observed) == 38
    assert set(observed) == set(v5.observed_joy_history_scalar_series("yay"))
    summary_keys = {
        "total_flos",
        "train_loss",
        "train_runtime",
        "train_samples_per_second",
        "train_steps_per_second",
    }
    assert all(
        item["local_file"] == "log_history.json"
        and item["step_axis"] == "optimizer_update"
        and item["unit"]
        and item["definition"]
        for item in observed.values()
    )
    assert {
        key for key, item in observed.items() if item["wandb_metric"] is None
    } == summary_keys
    assert {
        key for key, item in observed.items() if item["wandb_summary_key"] is not None
    } == summary_keys
    for key, item in observed.items():
        if key in summary_keys:
            assert item["wandb_summary_key"] == key
        else:
            assert item["wandb_metric"]
            assert item["wandb_summary_key"] is None
    configs = v5.generated_configs(registration, "1" * 64, recipe, "2" * 64)
    for label, config in configs.items():
        assert config["metric_schema_path"] == (
            ".confirmatory/v5/reproducibility/metric_schema.json"
        )
        assert config["metric_schema_sha256"] == v5.serialized_json_sha256(schema)
        assert config["registration_sha256"] == "1" * 64
        assert config["recipe_lock_sha256"] == "2" * 64
        assert config["recipe_sha256"] == v5.canonical_sha256(recipe)
        if label != "sealed_eval":
            assert config["registered_command"][0:3] == [
                "python",
                "-m",
                "jlens_rl.train",
            ]


def test_terminal_run_result_manifest_hashes_raw_histories_and_models(tmp_path) -> None:
    from jlens_rl.train import write_run_result_manifest

    output = tmp_path / "run"
    checkpoint = output / "checkpoint-25"
    final = output / "final"
    checkpoint.mkdir(parents=True)
    final.mkdir()
    for name, payload in {
        "run_manifest.json": "run\n",
        "resolved_config.json": "config\n",
        "data_indices.json": "data\n",
        "validation_history.jsonl": "validation\n",
        "log_history.json": "logs\n",
        "environment_snapshot.json": "{}\n",
    }.items():
        (output / name).write_text(payload)
    (checkpoint / "trainer_state.json").write_text('{"global_step": 25}\n')
    (checkpoint / "adapter_model.safetensors").write_bytes(b"checkpoint")
    (final / "adapter_model.safetensors").write_bytes(b"final")
    config = {
        "updates": 25,
        "registered_command": ["python", "-m", "jlens_rl.train"],
        "registration_sha256": "1" * 64,
        "recipe_lock_sha256": "2" * 64,
        "recipe_sha256": "3" * 64,
    }
    manifest = {
        "wandb_identity": {"run_id": "id"},
        "metric_schema": {"sha256": "4" * 64},
        "process_command": {"argv": ["train"]},
        "git_commit": "5" * 40,
        "git_dirty": False,
        "source_tree_sha256": "6" * 64,
        "runtime": {"cuda_device_name": "L40S"},
        "data_indices_sha256": "7" * 64,
        "lens_sha256": "8" * 64,
        "calibration_sha256": "9" * 64,
    }
    result = write_run_result_manifest(
        output_dir=output, cfg=config, run_manifest=manifest
    )
    assert result["raw_history_sha256"]["validation_history.jsonl"] == (
        v5.sha256_file(output / "validation_history.jsonl")
    )
    assert result["terminal_checkpoint"]["files"] == {
        "adapter_model.safetensors": v5.sha256_file(
            checkpoint / "adapter_model.safetensors"
        ),
        "trainer_state.json": v5.sha256_file(checkpoint / "trainer_state.json"),
    }
    assert result["final_adapter_and_tokenizer"]["files"] == {
        "adapter_model.safetensors": v5.sha256_file(
            final / "adapter_model.safetensors"
        )
    }
    assert json.loads((output / "run_result_manifest.json").read_text()) == result


def test_runtime_snapshot_pins_packages_python_driver_and_image_recipe() -> None:
    from jlens_rl.common import runtime_environment_snapshot

    snapshot = runtime_environment_snapshot()
    assert snapshot["pip_freeze_all"]
    assert snapshot["pip_freeze_all"] == sorted(snapshot["pip_freeze_all"])
    assert snapshot["python"]["version"]
    assert snapshot["python"]["executable"]
    assert "cuda_build" in snapshot["torch"]
    assert "nvidia_smi_name_and_driver" in snapshot
    assert set(snapshot["image_identity"]) == {
        "jlens_modal_image_spec",
        "modal_image_id",
    }


def test_final_bundle_covers_raw_evals_analysis_and_forensic_sources() -> None:
    protocol_source = PROTOCOL_PATH.read_text()
    modal_source = (ROOT / "modal_confirmatory_v5.py").read_text()
    eval_source = (ROOT / "src" / "jlens_rl" / "eval.py").read_text()
    assert "source_snapshot.zip" in protocol_source
    assert "pip_freeze_all" in protocol_source
    assert "evidence_bundle_inventory.json" in protocol_source
    assert 'return "raw sealed-final per-example record and paired-analysis input"' in (
        protocol_source
    )
    assert "run_result_manifest.json" in protocol_source
    assert "analysis_process.json" in modal_source
    assert "JLENS_MODAL_IMAGE_SPEC" in modal_source
    assert 'with_suffix(".environment.json")' in eval_source
    assert '"process_command"' in eval_source
    assert "hashlib.file_digest" in protocol_source
    assert "sealed_evaluation_file_count" in protocol_source


def test_registration_guard_pins_runner_split_lineage_and_identities() -> None:
    registration = _registration()
    v5._validate_registration_shape(registration)
    changed = json.loads(json.dumps(registration))
    changed["split"]["sealed_final"]["size"] = 1299
    with pytest.raises(v5.ProtocolError, match="data split"):
        v5._validate_registration_shape(changed)
    changed = json.loads(json.dumps(registration))
    changed["execution"]["max_parallel_gpu_workers"] = 8
    with pytest.raises(v5.ProtocolError, match="exact V5 runner"):
        v5._validate_registration_shape(changed)
    changed = json.loads(json.dumps(registration))
    changed["analysis"]["bootstrap_samples"] = 9999
    with pytest.raises(v5.ProtocolError, match="estimands/bootstrap"):
        v5._validate_registration_shape(changed)
    changed = json.loads(json.dumps(registration))
    changed["acceptance"]["difference_in_differences_mean"] = ">= 0"
    with pytest.raises(v5.ProtocolError, match="significance thresholds"):
        v5._validate_registration_shape(changed)
    changed = json.loads(json.dumps(registration))
    changed["wandb"]["project"] = "wrong-project"
    with pytest.raises(v5.ProtocolError, match="exactly 'j-lens-rl'"):
        v5._validate_registration_shape(changed)
    changed = json.loads(json.dumps(registration))
    changed["final_collection"]["labels"].pop()
    with pytest.raises(v5.ProtocolError, match="17-label"):
        v5._validate_registration_shape(changed)
    changed = json.loads(json.dumps(registration))
    changed["fixed_updates"] = 6
    with pytest.raises(v5.ProtocolError, match="terminal_adapter_step"):
        v5._validate_registration_shape(changed)
    assert "trl_build_metadata_sha256" in registration["execution"]


def test_final_registration_is_exact_current_joy_h6_prospective_template() -> None:
    path = ROOT / "protocol_archive" / "v5_emotional_registration.json"
    registration = json.loads(path.read_text())
    expected = v5.registration_template()
    expected["frozen_at_utc"] = "2026-07-14T08:26:20Z"
    expected["selected_recipe_lock"]["sha256"] = v5.sha256_file(
        ROOT / "protocol_archive" / "v5_emotional_selected_recipe.json"
    )
    expected["fixed_updates"] = 6
    expected["curve_gate"]["steps"] = [0, 2, 4, 6]
    expected["final_collection"]["terminal_adapter_step"] = 6
    expected["wandb"]["entity"] = "nilinabra-spare-time"
    expected["wandb"]["group"] = "confirm-v5-emotional-joy-h6"
    expected["wandb"]["run_prefix"] = "confirm-v5-emotional-joy-h6"
    assert registration == expected
    v5._validate_registration_shape(registration)


def test_outcome_free_infrastructure_amendment_preserves_registration() -> None:
    registration_path = ROOT / "protocol_archive" / "v5_emotional_registration.json"
    closeout_path = (
        ROOT
        / "protocol_archive"
        / "v5_emotional_prelaunch_attempt0_closeout.json"
    )
    amendment_path = (
        ROOT / "protocol_archive" / "v5_emotional_infrastructure_amendment1.json"
    )
    closeout = json.loads(closeout_path.read_text())
    amendment = json.loads(amendment_path.read_text())
    assert v5.sha256_file(registration_path) == v5.ORIGINAL_REGISTRATION_SHA256
    assert v5.sha256_file(closeout_path) == v5.V5_PRELAUNCH_CLOSEOUT_SHA256
    assert v5.sha256_file(amendment_path) == v5.V5_INFRASTRUCTURE_AMENDMENT1_SHA256
    assert closeout["outcome_boundary"] == {
        "claim_created": False,
        "function_call_dispatched": False,
        "gpu_task_started": False,
        "local_entrypoint_ran": False,
        "scientific_outcome_exists": False,
        "scientific_outcome_inspected": False,
        "wandb_run_created": False,
    }
    assert amendment["scientific_protocol_changed"] is False
    assert amendment["authorized_changes"]["volume"] == {
        "from": v5.ORIGINAL_VOLUME_NAME,
        "to": v5.VOLUME_NAME,
    }
    assert amendment["authorized_changes"]["finalizer_ephemeral_disk_mib"][
        "to"
    ] == 524288


def test_modal_runner_has_gated_controls_and_one_mapped_final_collection() -> None:
    source = (ROOT / "modal_confirmatory_v5.py").read_text()
    assert 'REMOTE_STATE = REMOTE_REPO / ".confirmatory" / "v5"' in source
    assert 'VOLUME_NAME = "j-lens-rl-confirmatory-v5-emotional-20260714b"' in source
    assert "ephemeral_disk=1024 * 512" in source
    assert "MAX_GPU_CONTAINERS = 1" in source
    assert "GLOBAL_MODAL_GPU_LIMIT = 1" in source
    assert "_serial_gpu_waves" in source
    assert "GPU_APP_OVERLAP_POLICY" in source
    assert source.count(".add_local_file(") >= 3
    assert 'LOCAL_REPO / "artifacts" / "qwen25_05b_solved_lens.pt"' in source
    assert 'LOCAL_REPO / ".confirmatory" / "manifests" / "curve_indices.json"' in source
    assert (
        'LOCAL_REPO / ".confirmatory" / "manifests" / "train_exclusions.json"'
        in source
    )
    assert 'if condition == "signflip":' in source
    assert '_protocol("verify-curve")' in source
    assert "claim_final_collection.remote(collection_id)" in source
    assert "final_waves = _serial_gpu_waves(FINAL_LABELS)" in source
    assert "evaluate_label,\n                    wave," in source
    assert "[collection_id] * len(wave)" in source
    assert "semantic_evals" not in source
    assert "control_evals" not in source
    assert "batch_upload(force=False)" in source
    assert "batch_upload(force=True)" not in source
    assert '"reproducibility": 6' in source
    orchestrate = source[source.index("def orchestrate(") : source.index("def _upload_protocol_state")]
    assert orchestrate.index("try:") < orchestrate.index(
        "_wait_for_launch_receipt(claim_id)"
    ) < orchestrate.index('_protocol("verify")')
    assert 'failure_phase = "launch_receipt_wait"' in orchestrate
    receipt_writer = source[
        source.index("def record_launch_receipt(") : source.index("def train_config(")
    ]
    assert receipt_writer.index('get("stage") != "claimed"') < (
        receipt_writer.index("_write_json_exclusive_atomic(path, receipt)")
    )
    assert "launch receipt cannot mutate a terminal or exported attempt" in receipt_writer


def test_train_supports_byte_pinned_wandb_identity_without_changing_old_configs() -> None:
    source = (ROOT / "src" / "jlens_rl" / "train.py").read_text()
    for environment_key in (
        "WANDB_RUN_ID",
        "WANDB_RUN_GROUP",
        "WANDB_RESUME",
        "WANDB_TAGS",
    ):
        assert environment_key in source
    assert 'if "wandb_run_id" not in cfg:' in source
    assert 'run_manifest["wandb_identity"]' in source
    assert '"terminal_run_result": result' in source
    assert '"validation_history.jsonl"' in source
    assert '"run_result_manifest.json"' in source


def test_nonclaim_replay_uses_fresh_external_path_and_no_original_wandb(
    tmp_path,
) -> None:
    from jlens_rl.train import (
        apply_training_cli_overrides,
        configure_reproduction_replay,
    )

    registration = json.loads(
        (ROOT / "protocol_archive" / "v5_emotional_registration.json").read_text()
    )
    lock = json.loads(
        (ROOT / "protocol_archive" / "v5_emotional_selected_recipe.json").read_text()
    )
    registration_sha256 = v5.sha256_file(
        ROOT / "protocol_archive" / "v5_emotional_registration.json"
    )
    lock_sha256 = v5.sha256_file(
        ROOT / "protocol_archive" / "v5_emotional_selected_recipe.json"
    )
    configs = v5.generated_configs(
        registration,
        registration_sha256,
        lock["resolved_training_config"],
        lock_sha256,
    )
    original = configs["jlens_seed168"]
    fresh = tmp_path / "outside-v5" / "jlens_seed168"
    replay = configure_reproduction_replay(
        original, output_dir=str(fresh), wandb_mode="disabled"
    )
    assert replay["output_dir"] == str(fresh)
    assert replay["wandb_mode"] == "disabled"
    assert replay["evidence_eligibility"] == "non_claim_reproduction"
    assert replay["reproduction_source"]["original_output_dir"] == original[
        "output_dir"
    ]
    assert replay["reproduction_source"]["original_wandb_identity"][
        "wandb_run_id"
    ] == original["wandb_run_id"]
    for key in (
        "wandb_entity",
        "wandb_group",
        "wandb_tags",
        "wandb_run_id",
        "wandb_url",
        "wandb_resume",
    ):
        assert key not in replay
    with pytest.raises(ValueError, match="immutable V5 state"):
        configure_reproduction_replay(
            original,
            output_dir=configs["signflip_seed168"]["output_dir"],
            wandb_mode="disabled",
        )
    canonical = apply_training_cli_overrides(
        original, updates=None, output_dir=None, wandb_mode="online"
    )
    assert canonical == original
    with pytest.raises(ValueError, match="update/output overrides"):
        apply_training_cli_overrides(
            original,
            updates=None,
            output_dir=str(fresh),
            wandb_mode="online",
        )
    with pytest.raises(ValueError, match="tracking-mode changes"):
        apply_training_cli_overrides(
            original, updates=None, output_dir=None, wandb_mode="disabled"
        )
    plan = v5._launch_plan(registration, configs)
    assert set(plan["reproduction_training_commands"]) == {
        label for label in configs if label != "sealed_eval"
    }
    for label, command in plan["reproduction_training_commands"].items():
        assert "--reproduction-replay" in command
        assert command[-2:] == ["--wandb-mode", "disabled"]
        output = command[command.index("--output-dir") + 1]
        assert output == f"REPLACE_WITH_FRESH_OUTPUT_ROOT/{label}"
        assert ".confirmatory/v5" not in output
    assert "never be substituted" in plan["reproduction_evidence_policy"]


def test_absent_launch_receipt_closes_atomically_and_has_truthful_closeout(
    tmp_path, monkeypatch
) -> None:
    import modal_confirmatory_v5 as runner

    state_dir = tmp_path / "v5"
    path_values = {
        "STATE_DIR": state_dir,
        "CONFIG_DIR": state_dir / "configs",
        "MANIFEST_DIR": state_dir / "manifests",
        "ARTIFACT_DIR": state_dir / "frozen_artifacts",
        "REPRODUCIBILITY_DIR": state_dir / "reproducibility",
        "RUN_DIR": state_dir / "runs",
        "EVAL_DIR": state_dir / "evals",
        "EVIDENCE_DIR": state_dir / "evidence",
        "STATE_PATH": state_dir / "protocol_state.json",
        "UNLOCK_PATH": state_dir / "final_unlocked.json",
        "FINAL_COLLECTION_PATH": state_dir / "final_collection.json",
        "ATTEMPT_CLAIM_PATH": state_dir / "attempt_claim.json",
        "ATTEMPT_STATUS_PATH": state_dir / "attempt_status.json",
        "LAUNCH_RECEIPT_PATH": state_dir / "launch_receipt.json",
        "CURVE_GATE_PATH": state_dir / "evidence" / "curve_gate.json",
        "ACCEPTANCE_PATH": state_dir / "evidence" / "acceptance.json",
        "CLOSEOUT_CANDIDATE_PATH": state_dir
        / "evidence"
        / "git_closeout_candidate.json",
        "BUNDLE_INVENTORY_PATH": state_dir
        / "evidence"
        / "evidence_bundle_inventory.json",
    }
    for name, value in path_values.items():
        monkeypatch.setattr(v5, name, value)
    state_dir.mkdir()
    claim_id = "a" * 32
    state = {
        "protocol": v5.PROTOCOL,
        "git_commit": "b" * 40,
        "registration_sha256": "c" * 64,
        "recipe_lock_sha256": "d" * 64,
        "metric_schema_sha256": "e" * 64,
        "wandb_identities": {},
        "target_words": ["joy"],
        "artifact_sha256": {"lens": "f" * 64, "calibration": "1" * 64},
    }
    claim = {
        "claim_id": claim_id,
        "git_commit": state["git_commit"],
        "registration_sha256": state["registration_sha256"],
        "recipe_lock_sha256": state["recipe_lock_sha256"],
        "global_modal_gpu_limit": v5.GLOBAL_MODAL_GPU_LIMIT,
        "gpu_app_overlap_policy": v5.GPU_APP_OVERLAP_POLICY,
        "operational_preflight": {
            "exclusive_gpu_confirmation": v5.GPU_EXCLUSIVE_CONFIRMATION,
            "active_other_modal_apps": [],
            "global_modal_gpu_limit": v5.GLOBAL_MODAL_GPU_LIMIT,
        },
    }
    status = {
        "claim_id": claim_id,
        "stage": "failed",
        "failed_from_stage": "claimed",
        "failure_phase": "launch_receipt_wait",
        "launch_receipt_present": False,
    }
    class FakeVolume:
        def __init__(self) -> None:
            self.commits = 0
            self.reloads = 0

        def commit(self) -> None:
            self.commits += 1

        def reload(self) -> None:
            self.reloads += 1

    fake_volume = FakeVolume()
    monkeypatch.setattr(runner, "REMOTE_STATE", state_dir)
    monkeypatch.setattr(runner, "state_volume", fake_volume)
    monotonic_values = iter((0.0, 601.0))
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(monotonic_values))
    with pytest.raises(RuntimeError, match="timed out waiting"):
        runner._wait_for_launch_receipt(claim_id)
    assert fake_volume.commits == 1
    closure = json.loads(v5.LAUNCH_RECEIPT_PATH.read_text())
    assert closure["receipt_status"] == "absent_closed_before_dispatch"
    for path, payload in (
        (v5.STATE_PATH, state),
        (v5.ATTEMPT_CLAIM_PATH, claim),
        (v5.ATTEMPT_STATUS_PATH, status),
    ):
        path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    records = v5._verify_operational_attempt_records()
    assert records["receipt"] is None
    assert records["absent_receipt_closure"] == closure
    before = v5._current_bundle_inventory()
    with pytest.raises(FileExistsError):
        runner._write_json_exclusive_atomic(
            v5.LAUNCH_RECEIPT_PATH,
            {"claim_id": claim_id, "receipt_status": "present"},
        )
    assert json.loads(v5.LAUNCH_RECEIPT_PATH.read_text()) == closure
    assert v5._current_bundle_inventory() == before
    closeout = v5._write_closeout_candidate(records)
    assert closeout["launch_receipt_sha256"] is None
    assert closeout["absent_receipt_closure_sha256"] == v5.sha256_file(
        v5.LAUNCH_RECEIPT_PATH
    )
    assert closeout["launch_receipt_status"].startswith("absent before")


def test_pinned_wandb_identity_is_applied_and_returned(monkeypatch) -> None:
    from jlens_rl.train import configure_wandb_environment

    for key in (
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_MODE",
        "WANDB_RUN_ID",
        "WANDB_RUN_GROUP",
        "WANDB_RESUME",
        "WANDB_TAGS",
    ):
        monkeypatch.delenv(key, raising=False)
    config = {
        "wandb_project": "j-lens-rl",
        "wandb_entity": "nilinabra-spare-time",
        "wandb_mode": "online",
        "run_name": "confirm-v5-yay-jlens-seed168",
        "wandb_run_id": "confirm-v5-deadbeef-jlens_seed168",
        "wandb_url": "https://wandb.ai/entity/project/runs/id",
        "wandb_group": "confirm-v5-yay",
        "wandb_resume": "never",
        "wandb_tags": ["confirmatory-v5", "emotional-j-lens"],
    }
    identity = configure_wandb_environment(config)
    assert identity == {
        "entity": "nilinabra-spare-time",
        "project": "j-lens-rl",
        "run_name": config["run_name"],
        "run_id": config["wandb_run_id"],
        "url": config["wandb_url"],
        "group": config["wandb_group"],
        "tags": config["wandb_tags"],
        "resume": "never",
    }
    assert __import__("os").environ["WANDB_RUN_ID"] == config["wandb_run_id"]
    assert __import__("os").environ["WANDB_ENTITY"] == config["wandb_entity"]
    assert __import__("os").environ["WANDB_RUN_GROUP"] == config["wandb_group"]
    assert __import__("os").environ["WANDB_RESUME"] == "never"
    assert __import__("os").environ["WANDB_TAGS"] == (
        "confirmatory-v5,emotional-j-lens"
    )
    assert configure_wandb_environment(
        {"wandb_project": "j-lens-rl", "wandb_mode": "offline"}
    ) is None
    with pytest.raises(ValueError, match="wandb_tags"):
        configure_wandb_environment(
            {
                "wandb_project": "j-lens-rl",
                "wandb_mode": "online",
                "wandb_tags": [],
            }
        )


def test_terminal_wandb_evidence_includes_hash_manifest_and_raw_histories(
    tmp_path, monkeypatch
) -> None:
    from jlens_rl.train import (
        _load_valid_terminal_publish_receipt,
        _observe_active_wandb_identity,
        _validate_terminal_artifact_identity,
        publish_run_result_to_wandb,
    )

    names = (
        "run_result_manifest.json",
        "validation_history.jsonl",
        "log_history.json",
        "environment_snapshot.json",
        "run_manifest.json",
        "resolved_config.json",
        "data_indices.json",
    )
    for name in names:
        (tmp_path / name).write_text("{}\n")

    updates = []
    uploads = []

    class FakeConfig:
        def update(self, value, *, allow_val_change):
            updates.append((value, allow_val_change))

    class FakeArtifact:
        def __init__(self, name, *, type, metadata):
            self.id = "artifact-id"
            self.name = name
            self.version = "v0"
            self.digest = "artifact-digest"
            self.qualified_name = None
            self.type = type
            self.metadata = metadata
            self.files = []

        def add_file(self, path, *, name):
            self.files.append((path, name))

        def wait(self):
            self.name = f"{self.name}:v0"
            self.qualified_name = f"entity/project/{self.name}"
            return self

    frozen_identity = {
        "run_id": "frozen-id",
        "entity": "entity",
        "project": "project",
        "run_name": "frozen-name",
        "url": "https://wandb.ai/entity/project/runs/frozen-id",
        "group": "frozen-group",
        "tags": ["confirmatory-v5", "emotional-j-lens"],
        "resume": "never",
    }
    fake_run = SimpleNamespace(
        id=frozen_identity["run_id"],
        entity=frozen_identity["entity"],
        project=frozen_identity["project"],
        name=frozen_identity["run_name"],
        url=frozen_identity["url"],
        group=frozen_identity["group"],
        tags=tuple(frozen_identity["tags"]),
    )
    fake_run.log_artifact = lambda artifact, **kwargs: artifact

    fake_wandb = SimpleNamespace(
        run=fake_run,
        config=FakeConfig(),
        save=lambda path, **kwargs: uploads.append((path, kwargs)),
        Artifact=FakeArtifact,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    result = {
        "wandb_identity": frozen_identity,
        "terminal_checkpoint": {"sha256": "a" * 64},
        "final_adapter_and_tokenizer": {"sha256": "b" * 64},
        "raw_history_sha256": {"validation_history.jsonl": "c" * 64},
    }
    receipt = publish_run_result_to_wandb(
        output_dir=tmp_path,
        result=result,
        enabled=True,
    )
    assert updates[0] == ({"terminal_run_result": result}, True)
    assert updates[1] == ({"terminal_evidence_receipt": receipt}, True)
    assert [Path(path).name for path, _ in uploads] == [
        *names,
        "wandb_terminal_publish_receipt.json",
    ]
    assert all(item[1]["policy"] == "now" for item in uploads)
    assert receipt["artifact"]["id"] == "artifact-id"
    assert receipt["artifact"]["name"] == "frozen-id-terminal-evidence:v0"
    assert receipt["artifact"]["version"] == "v0"
    assert receipt["artifact"]["qualified_name"] == (
        "entity/project/frozen-id-terminal-evidence:v0"
    )
    assert receipt["observed_wandb_identity"] == {
        key: frozen_identity[key]
        for key in ("run_id", "entity", "project", "run_name", "url", "group", "tags")
    }
    assert (tmp_path / "wandb_terminal_publish_receipt.json").is_file()
    assert not (tmp_path / "wandb_terminal_publish_receipt.json.tmp").exists()
    assert _load_valid_terminal_publish_receipt(
        output_dir=tmp_path, result=result
    ) == receipt
    (tmp_path / "wandb_terminal_publish_receipt.json").write_text("{truncated")
    assert _load_valid_terminal_publish_receipt(
        output_dir=tmp_path, result=result
    ) is None

    fake_run.id = "wrong-id"
    with pytest.raises(RuntimeError, match="frozen observable identity"):
        publish_run_result_to_wandb(
            output_dir=tmp_path,
            result=result,
            enabled=True,
        )
    fake_run.id = frozen_identity["run_id"]
    for attribute, bad_value in (
        ("entity", "wrong-entity"),
        ("project", "wrong-project"),
        ("group", "wrong-group"),
        ("tags", ("wrong-tag",)),
        ("url", "https://wandb.ai/wrong"),
    ):
        original = getattr(fake_run, attribute)
        setattr(fake_run, attribute, bad_value)
        with pytest.raises(RuntimeError, match="frozen observable identity"):
            _observe_active_wandb_identity(fake_run, frozen_identity)
        setattr(fake_run, attribute, original)

    exact_artifact = receipt["artifact"]
    _validate_terminal_artifact_identity(exact_artifact, frozen_identity)
    for field, bad_value in (
        ("name", "wrong:v0"),
        ("version", "latest"),
        ("qualified_name", "wrong/project/name:v0"),
    ):
        malformed = dict(exact_artifact)
        malformed[field] = bad_value
        with pytest.raises(RuntimeError, match="exact terminal evidence artifact"):
            _validate_terminal_artifact_identity(malformed, frozen_identity)


def test_v4_entrypoints_and_state_paths_are_untouched() -> None:
    assert (ROOT / "scripts" / "confirmatory_protocol.py").is_file()
    assert (ROOT / "modal_experiments.py").is_file()
    assert (ROOT / "run_confirmatory.sh").is_file()
    assert (ROOT / ".confirmatory" / "protocol_state.json").is_file()
    diff = __import__("subprocess").check_output(
        [
            "git",
            "diff",
            "--name-only",
            "--",
            "scripts/confirmatory_protocol.py",
            "modal_experiments.py",
            "run_confirmatory.sh",
            "configs/confirmatory_common.json",
        ],
        cwd=ROOT,
        text=True,
    )
    assert diff == ""
