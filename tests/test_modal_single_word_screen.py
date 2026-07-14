import json
from pathlib import Path

import modal_single_word_screen as screen


ROOT = Path(__file__).resolve().parents[1]


def test_single_word_screen_is_bounded_and_only_mounts_exposed_manifests():
    assert screen.APP_NAME == "j-lens-rl-emotional-single-word-screen-v1"
    assert (
        screen.VOLUME_NAME
        == "j-lens-rl-emotional-single-word-screen-v1-20260714a"
    )
    assert screen.MAX_GPU_CONTAINERS == 8
    assert screen.CALIBRATION_MAX_GPU_CONTAINERS == 8
    assert screen.SEED == 167
    assert screen.EXPECTED_STEPS == (0, 2, 4, 6, 10, 15, 20, 25)
    assert screen.GATE_STEPS == (0, 2, 4, 6)
    assert set(screen.EXPOSED_MANIFEST_SHA256) == {
        "curve_indices.json",
        "train_exclusions.json",
    }
    source = (ROOT / "modal_single_word_screen.py").read_text()
    for forbidden in (
        'LOCAL_MANIFESTS / "sealed_final_indices.json"',
        'LOCAL_MANIFESTS / "future_reserve_indices.json"',
        'LOCAL_MANIFESTS / "retired_v3_curve_indices.json"',
    ):
        assert forbidden not in source
    assert source.count(".add_local_file(") == 3
    assert screen._validate_v4_closeout(ROOT) == screen.V4_CLOSEOUT_SHA256
    launcher = (ROOT / "run_single_word_screen.sh").read_text()
    assert "modal run --detach" in launcher
    assert "modal_single_word_screen.py" in launcher


def test_single_word_arms_signs_and_token_ids_are_frozen():
    assert screen.ARM_ORDER == (
        "yay",
        "wow",
        "joy",
        "proud",
        "excited",
        "damn",
        "fuck",
        "worried",
    )
    assert screen.REWARD_WEIGHTS == {
        "yay": 1.0,
        "wow": 1.0,
        "joy": 1.0,
        "proud": 1.0,
        "excited": 1.0,
        "damn": -1.0,
        "fuck": -1.0,
        "worried": -1.0,
    }
    assert screen.EXPECTED_TOKEN_IDS == {
        "yay": [97559, 138496],
        "wow": [35665, 35881, 45717, 57454, 61300],
        "joy": [4123, 15888, 27138, 79771],
        "proud": [12409, 83249],
        "excited": [12035],
        "damn": [26762, 82415, 88619, 95614],
        "fuck": [7820, 25090, 70474, 75021, 76374],
        "worried": [17811],
    }


def test_every_arm_is_the_same_fixed_u5_j_only_recipe_except_word_and_sign():
    hashes = screen._validate_templates(ROOT)
    assert set(hashes) == {screen.COMMON_CONFIG, *screen.VARIANTS.values()}
    for word, relative in screen.VARIANTS.items():
        config = screen._load_config(ROOT / relative)
        assert config["reward_type"] == "jlens"
        assert config["target_words"] == [word]
        assert config["seed"] == 167
        assert config["updates"] == 25
        assert config["learning_rate"] == 3e-6
        assert config["lr_scheduler_type"] == "constant"
        assert config["warmup_steps"] == 0
        assert config["warmup_ratio"] == 0.0
        assert config["early_stopping_patience"] is None
        assert config["validation_observational_only"] is True
        assert config["mask_target_tokens"] is True
        assert config["score_stride"] == 5
        assert config["score_components"] == [
            {
                "layer": 8,
                "start_fraction": 0.5,
                "end_fraction": 1.0,
                "aggregation": "mean",
                "weight": screen.REWARD_WEIGHTS[word],
            }
        ]
        assert config["validation_examples"] == 400
        assert config["validation_steps"] == [2, 4, 6, 10, 15, 20, 25]
        assert config["validation_indices_path"] == (
            ".confirmatory/manifests/curve_indices.json"
        )
        assert config["reserved_train_indices_path"] == (
            ".confirmatory/manifests/train_exclusions.json"
        )


def test_calibration_is_one_word_at_a_time_on_pinned_wikitext():
    for word in screen.ARM_ORDER:
        output = screen.REMOTE_OUTPUT / "artifacts" / f"{word}_calibration.json"
        command = screen._calibration_command(word, output)
        assert command.count("--target-word") == 1
        assert command[command.index("--target-word") + 1] == word
        assert command[command.index("--wikitext-revision") + 1] == (
            screen.WIKITEXT_REVISION
        )
        assert command[command.index("--lens-input") + 1] == (
            "artifacts/qwen25_05b_solved_lens.pt"
        )
        assert command[command.index("--num-prompts") + 1] == "100"
        assert command[command.index("--calibration-prompts") + 1] == "50"
        assert command[command.index("--layers") + 1] == "8,14,20"
        assert command[command.index("--seed") + 1] == "42"


def test_requested_curve_gate_is_exact():
    assert screen._curve_pass({0: 0.30, 2: 0.31, 4: 0.31, 6: 0.32})
    assert not screen._curve_pass({0: 0.30, 2: 0.30, 4: 0.31, 6: 0.32})
    assert not screen._curve_pass({0: 0.30, 2: 0.31, 4: 0.30, 6: 0.32})
    assert not screen._curve_pass({0: 0.30, 2: 0.31, 4: 0.32, 6: 0.31})


def test_all_new_templates_are_valid_json():
    for path in [
        ROOT / screen.COMMON_CONFIG,
        *(ROOT / relative for relative in screen.VARIANTS.values()),
    ]:
        assert isinstance(json.loads(path.read_text()), dict)
