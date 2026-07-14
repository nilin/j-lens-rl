import json
from pathlib import Path

import modal_word_explore as screen


ROOT = Path(__file__).resolve().parents[1]


def test_word_screen_is_bounded_and_uses_only_exposed_manifests():
    assert screen.MAX_GPU_CONTAINERS == 8
    assert screen.CALIBRATION_MAX_GPU_CONTAINERS == 2
    assert screen.SEED == 167
    assert screen.EXPECTED_STEPS == (0, 2, 4, 6, 10, 15, 20, 25)
    assert screen.GATE_STEPS == (0, 2, 4, 6)
    assert set(screen.EXPOSED_MANIFEST_SHA256) == {
        "curve_indices.json",
        "train_exclusions.json",
    }
    source = (ROOT / "modal_word_explore.py").read_text()
    for forbidden in (
        'add_local_file(\n        LOCAL_MANIFESTS / "sealed_final_indices.json"',
        'add_local_file(\n        LOCAL_MANIFESTS / "future_reserve_indices.json"',
    ):
        assert forbidden not in source
    assert screen._validate_v4_closeout(ROOT) == screen.V4_CLOSEOUT_SHA256
    closeout = json.loads((ROOT / screen.V4_CLOSEOUT_RELATIVE).read_text())
    assert closeout["attempt_stage"] == "curve_failed"
    assert closeout["final_unlocked_present"] is False
    assert closeout["evals_directory_present"] is False
    assert closeout["final_evaluation_labels"] == []
    assert closeout["signflip_run_labels"] == []


def test_word_families_and_variants_are_exact_j_only_constructions():
    assert screen.FAMILIES == {
        "profanity": ("damn", "fuck"),
        "celebration": ("yay", "great", "success", "nice"),
    }
    assert screen.PRIORITY == (
        "celebration_ultradense",
        "profanity_ultradense",
        "celebration_taper",
        "profanity_taper",
        "solved_u5_control",
        "solved_u5_low_lr",
        "solved_u5_taper",
        "solved_u5_taper_low_lr",
    )
    configs = {
        label: screen._load_config(ROOT / relative)
        for label, relative in screen.VARIANTS.items()
    }
    for label, config in configs.items():
        assert config["reward_type"] == "jlens"
        assert config["seed"] == 167
        assert config["updates"] == 25
        assert config["learning_rate"] in {2e-6, 3e-6}
        assert config["lr_scheduler_type"] == "constant"
        assert config["warmup_steps"] == 0
        assert config["early_stopping_patience"] is None
        assert config["validation_observational_only"] is True
        assert config["mask_target_tokens"] is True
        assert config["validation_examples"] == 400
        assert config["validation_steps"] == [2, 4, 6, 10, 15, 20, 25]
        assert config["validation_indices_path"].endswith(
            "curve_indices.json"
        )
        assert config["reserved_train_indices_path"].endswith(
            "train_exclusions.json"
        )
        if label.startswith("solved_"):
            assert config["target_words"] == ["solved"]
            assert all(
                component["weight"] > 0
                for component in config["score_components"]
            )
            assert config["score_stride"] == 5
            assert config["learning_rate"] == (
                2e-6 if label.endswith("low_lr") else 3e-6
            )
            expected_weights = (
                [1.0, 0.25] if "taper" in label else [1.0]
            )
            assert [
                component["weight"] for component in config["score_components"]
            ] == expected_weights
            continue

        family = "profanity" if label.startswith("profanity_") else "celebration"
        assert config["target_words"] == list(screen.FAMILIES[family])
        weights = [component["weight"] for component in config["score_components"]]
        assert all(weight < 0 for weight in weights) == (family == "profanity")
        assert config["score_stride"] == (5 if label.endswith("ultradense") else 10)
        assert weights == (
            [-1.0] if label == "profanity_ultradense"
            else [1.0] if label == "celebration_ultradense"
            else [-1.0, -0.25] if label == "profanity_taper"
            else [1.0, 0.25]
        )


def test_requested_curve_gate_is_exact():
    assert screen._curve_pass({0: 0.3, 2: 0.31, 4: 0.31, 6: 0.32})
    assert not screen._curve_pass({0: 0.3, 2: 0.3, 4: 0.31, 6: 0.32})
    assert not screen._curve_pass({0: 0.3, 2: 0.31, 4: 0.30, 6: 0.32})
    assert not screen._curve_pass({0: 0.3, 2: 0.31, 4: 0.32, 6: 0.31})


def test_committed_templates_are_valid_json():
    for path in [
        ROOT / "configs/word_explore_common.json",
        *(ROOT / relative for relative in screen.VARIANTS.values()),
    ]:
        assert isinstance(json.loads(path.read_text()), dict)
