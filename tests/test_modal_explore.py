import importlib.util
from pathlib import Path

from jlens_rl.common import load_config


def _load_modal_explore():
    path = Path(__file__).resolve().parents[1] / "modal_explore.py"
    spec = importlib.util.spec_from_file_location("modal_explore_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


modal_explore = _load_modal_explore()


def test_exploratory_modal_runner_has_a_four_gpu_data_firewall() -> None:
    assert modal_explore.MAX_GPU_CONTAINERS == 4
    assert set(modal_explore.EXPOSED_MANIFESTS) == {
        "train_exclusions.json",
        "retired_v2_curve_indices.json",
    }
    assert all(
        forbidden not in modal_explore.EXPOSED_MANIFESTS
        for forbidden in modal_explore.FORBIDDEN_MANIFESTS
    )
    assert set(modal_explore.EXPECTED_MANIFEST_SHA256) == {
        *modal_explore.EXPOSED_MANIFESTS,
        *modal_explore.FORBIDDEN_MANIFESTS,
    }
    modal_explore._verify_local_data_firewall()


def test_exploratory_gate_requires_first_rise_and_no_later_drop() -> None:
    assert modal_explore.GATE_STEPS == (0, 2, 4, 6)
    assert modal_explore._requested_curve_pattern(
        {0: 0.375, 2: 0.3775, 4: 0.3775, 6: 0.38}
    )
    assert not modal_explore._requested_curve_pattern(
        {0: 0.375, 2: 0.375, 4: 0.38, 6: 0.3825}
    )
    assert not modal_explore._requested_curve_pattern(
        {0: 0.375, 2: 0.38, 4: 0.3775, 6: 0.3825}
    )


def test_exploratory_configs_are_fixed_j_only_matched_screens() -> None:
    configs = {
        label: load_config(Path(path))
        for label, path in modal_explore.VARIANTS.items()
    }
    assert set(configs) == {
        "solved_ultradense5",
        "solved_tail_taper",
        "solved_tempered_delta",
        "solved_layer_shrink",
    }
    for label, config in configs.items():
        assert config["reward_type"] == "jlens", label
        assert config["target_words"] == ["solved"], label
        assert config["seed"] == 158, label
        assert config["updates"] == 25, label
        assert config["eval_every"] == 2, label
        assert config["validation_steps"] == [2, 4, 6, 10, 15, 20, 25], label
        assert config["validation_observational_only"] is True, label
        assert config["early_stopping_patience"] is None, label
        assert config["validation_examples"] == 400, label
        assert config["validation_indices_path"].endswith(
            "retired_v2_curve_indices.json"
        ), label
        assert config["reserved_train_indices_path"].endswith(
            "train_exclusions.json"
        ), label
        assert config["learning_rate"] == 3e-6, label
        assert config["lr_scheduler_type"] == "constant", label
        assert config["warmup_steps"] == 0, label
        assert config["warmup_ratio"] == 0.0, label
        assert config["require_clean_repository"] is True, label
        assert config["output_dir"] == f"/explore2/runs/{label}", label

    assert configs["solved_ultradense5"]["score_stride"] == 5
    assert configs["solved_tail_taper"]["score_components"][1]["weight"] == 0.25
    assert configs["solved_tempered_delta"]["score_components"][1]["weight"] == -0.25
    assert {
        component["layer"]
        for component in configs["solved_layer_shrink"]["score_components"]
    } == {8, 14, 20}
