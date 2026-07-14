from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "protocol_archive"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v7 = load_module("confirmatory_v7_test", ROOT / "scripts" / "confirmatory_v7_protocol.py")
volume_probe = load_module(
    "modal_verify_v7_volume_test", ROOT / "scripts" / "modal_verify_v7_volume.py"
)
VOLUME_NAME = volume_probe.VOLUME_NAME
verify_v7_volume_v2 = volume_probe.verify_v7_volume_v2


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finalized_registration() -> dict:
    path = ARCHIVE / "v7_profanity_registration.json"
    assert path.is_file(), "the V7 design must include its final registration"
    return json.loads(path.read_text())


def selected_lock() -> dict:
    return json.loads((ARCHIVE / "v7_profanity_selected_recipe.json").read_text())


def test_selection_closeout_freezes_requested_design_and_boundary() -> None:
    path = ARCHIVE / "v7_profanity_selection_closeout.json"
    closeout = json.loads(path.read_text())
    design = closeout["design"]
    assert sha256(path) == v7.SELECTION_CLOSEOUT_SHA256
    assert design["target_words"] == ["damn", "fuck"]
    assert design["calibration_sha256"] == v7.PROFANITY_CALIBRATION_SHA256
    assert design["score_components"] == [
        {
            "aggregation": "mean",
            "end_fraction": 1.0,
            "layer": 8,
            "start_fraction": 0.5,
            "weight": -1.0,
        }
    ]
    assert design["score_stride"] == 5
    assert design["seeds"] == list(range(184, 192))
    assert design["fixed_updates"] == 20
    assert design["curve_gate"]["steps"] == [0, 4, 10, 20]
    assert closeout["v6_dependency"][
        "terminal_closeout_intentionally_absent_at_v7_freeze"
    ] is True


def test_selected_recipe_is_exact_one_component_profanity_u5() -> None:
    lock = selected_lock()
    recipe = lock["resolved_training_config"]
    assert lock["emotional_only"] is True
    assert recipe["target_words"] == ["damn", "fuck"]
    assert recipe["score_components"][0] == {
        "aggregation": "mean",
        "end_fraction": 1.0,
        "layer": 8,
        "start_fraction": 0.5,
        "weight": -1.0,
    }
    assert recipe["score_stride"] == 5
    assert recipe["learning_rate"] == 3e-6
    assert recipe["lr_scheduler_type"] == "constant"
    assert recipe["kl_beta"] == 0.02
    assert recipe["loss_type"] == "dapo"
    assert recipe["updates"] == recipe["save_every"] == 20
    assert recipe["validation_steps"] == [4, 10, 20]
    assert recipe["train_examples"] == 1000
    assert recipe["validation_examples"] == 400


@pytest.mark.parametrize("target", [["solved"], ["ERROR"], ["error-prone"]])
def test_retired_non_emotional_targets_fail_closed(target: list[str]) -> None:
    with pytest.raises(v7.ProtocolError, match="retired non-emotional"):
        v7.validate_target_words(target)


def test_curve_gate_is_exact_requested_shape() -> None:
    assert v7.curve_means_pass([0.38, 0.39, 0.39, 0.40]) is True
    assert v7.curve_means_pass([0.38, 0.38, 0.39, 0.40]) is False
    assert v7.curve_means_pass([0.38, 0.39, 0.385, 0.40]) is False
    assert v7.curve_means_pass([0.38, 0.39, 0.40, 0.395]) is False


def test_registration_freezes_science_wandb_and_conditional_final() -> None:
    registration = finalized_registration()
    lock_path = ARCHIVE / "v7_profanity_selected_recipe.json"
    assert registration["seeds"] == list(range(184, 192))
    assert registration["fixed_updates"] == 20
    assert registration["curve_gate"]["steps"] == [0, 4, 10, 20]
    assert registration["selected_recipe_lock"]["sha256"] == sha256(lock_path)
    assert registration["final_collection"] == {
        "labels": [
            "base",
            *[f"jlens_seed{seed}" for seed in range(184, 192)],
            *[f"signflip_seed{seed}" for seed in range(184, 192)],
        ],
        "one_immutable_collection": True,
        "terminal_adapter_step": 20,
    }
    assert registration["split"]["sealed_final"]["size"] == 900
    predicate = registration["conditional_launch_predicate"]
    assert predicate["required_final_outcomes_unopened"] is True
    assert predicate["required_final_evaluation_labels"] == []
    wandb = registration["wandb"]
    assert len(wandb["run_ids"]) == len(set(wandb["run_ids"].values())) == 16
    assert set(wandb["run_ids"]) == {
        *[f"jlens_seed{seed}" for seed in range(184, 192)],
        *[f"signflip_seed{seed}" for seed in range(184, 192)],
    }


def test_generated_configs_have_exact_signs_ids_horizon_and_curve() -> None:
    registration = finalized_registration()
    lock_path = ARCHIVE / "v7_profanity_selected_recipe.json"
    lock = selected_lock()
    generated = v7.generated_configs(
        registration,
        sha256(ARCHIVE / "v7_profanity_registration.json"),
        lock["resolved_training_config"],
        sha256(lock_path),
    )
    assert set(generated) == {
        "sealed_eval",
        *[f"jlens_seed{seed}" for seed in range(184, 192)],
        *[f"signflip_seed{seed}" for seed in range(184, 192)],
    }
    for seed in range(184, 192):
        treatment = generated[f"jlens_seed{seed}"]
        control = generated[f"signflip_seed{seed}"]
        assert treatment["score_components"][0]["weight"] == -1.0
        assert control["score_components"][0]["weight"] == 1.0
        for config in (treatment, control):
            assert config["updates"] == config["save_every"] == 20
            assert config["validation_steps"] == [4, 10, 20]
            assert config["wandb_resume"] == "never"
            assert config["wandb_run_id"].endswith(
                f"-{config['output_dir'].rsplit('/', 1)[-1]}"
            )


def test_metric_schema_describes_the_actual_one_component_reward() -> None:
    recipe = selected_lock()["resolved_training_config"]
    schema = v7.metric_schema(
        recipe["target_words"], recipe["updates"], recipe["score_components"]
    )
    semantics = schema["condition_weight_semantics"]
    assert semantics["treatment"] == [-1.0]
    assert semantics["signflip_control"] == [1.0]
    named = schema["series"]["intrinsic_named_weighted_reward_mean"]
    assert named["range"] == [-5.0, 5.0]
    assert "one-component" in named["unit"]


def test_conditional_v6_closeout_is_missing_and_protocol_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing = tmp_path / "not-yet-created.json"
    monkeypatch.setattr(v7, "V6_TERMINAL_CLOSEOUT_PATH", missing)
    with pytest.raises(v7.ProtocolError, match="V7 is inert"):
        v7.verify_v6_launch_predicate()


def test_modal_image_is_strict_and_gpu_functions_hold_global_lease() -> None:
    source = (ROOT / "modal_confirmatory_v7.py").read_text()
    assert "add_local_dir(\n    LOCAL_REPO," not in source
    assert 'LOCAL_REPO / "src" / "jlens_rl"' in source
    assert 'LOCAL_REPO / "trl" / "trl"' in source
    assert "protocol_archive" not in source
    assert 'create_if_missing=False, version=2' in source
    assert 'GPU_LEASE_DICT_NAME = "j-lens-rl-global-gpu-lease-v1"' in source
    assert 'GPU_LEASE_KEY = "global-one-gpu"' in source
    assert "skip_if_exists=True" in source
    for function_name in ("train_config", "evaluate_label"):
        start = source.index(f"def {function_name}(")
        end = source.find("\n@app.function", start)
        body = source[start : end if end != -1 else None]
        assert "_acquire_gpu_lease(" in body
        assert "state_volume.commit()" in body
        assert "_release_gpu_lease(lease)" in body
        assert body.index("state_volume.commit()") < body.index(
            "_release_gpu_lease(lease)"
        )
    assert "_serial_gpu_waves(SEEDS)" in source
    assert "_serial_gpu_waves(FINAL_LABELS)" in source


def test_modal_volume_probe_is_noncreating_and_version_two() -> None:
    calls: list[tuple[str, bool, int]] = []

    class FakeVolume:
        object_id = "vo-fresh"

        def hydrate(self) -> None:
            return None

    def factory(name: str, *, create_if_missing: bool, version: int) -> FakeVolume:
        calls.append((name, create_if_missing, version))
        return FakeVolume()

    assert verify_v7_volume_v2(factory) == "vo-fresh"
    assert calls == [(VOLUME_NAME, False, 2)]


def test_runtime_source_snapshot_allowlist_excludes_outcomes_and_git() -> None:
    names = v7._runtime_source_names()
    assert "scripts/confirmatory_v7_protocol.py" in names
    assert "modal_confirmatory_v7.py" in names
    assert any(name.startswith("src/jlens_rl/") for name in names)
    assert any(name.startswith("trl/trl/") for name in names)
    assert not any(
        part in {".git", ".confirmatory", "protocol_archive", "history", "evals"}
        for name in names
        for part in Path(name).parts
    )


def test_runtime_source_has_a_deterministic_parentless_git_identity() -> None:
    first = v7._tracked_source_inventory()
    second = v7._tracked_source_inventory()
    assert first == second
    assert len(first["git_commit"]) == 40
    assert first["source_git_commit"] == subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    assert first["runtime_commit_recipe"]["parent"] is None


def test_shell_forbids_unleased_local_gpu_and_names_exact_seeds() -> None:
    source = (ROOT / "run_confirmatory_v7.sh").read_text()
    assert "SEEDS=(184 185 186 187 188 189 190 191)" in source
    assert "refusing unleased local GPU execution" in source
    assert "train-jlens-rl" not in source
    assert "eval-jlens-rl" not in source


def test_registration_execution_hashes_match_current_allowlisted_code() -> None:
    registration = finalized_registration()
    execution = registration["execution"]
    assert execution["volume"] == VOLUME_NAME
    assert execution["volume_version"] == 2
    assert execution["volume_status"].startswith("placeholder-")
    assert execution["gpu_type"] == "L40S"
    assert execution["max_parallel_gpu_workers"] == 1
    assert execution["gpu_lease"] == {
        "dict_name": "j-lens-rl-global-gpu-lease-v1",
        "environment": "main",
        "key": "global-one-gpu",
        "policy": v7.GPU_LEASE_POLICY,
    }
    for key, digest in v7._expected_execution_hashes().items():
        assert execution[key] == digest
