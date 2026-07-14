from __future__ import annotations

import hashlib
import json
from pathlib import Path

from jlens_rl.common import load_config


ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT / "protocol_archive" / "emotional_parallel_v2_registration.json"
SCRIPT = ROOT / "modal_emotional_parallel_v2.py"
REGISTRATION_SHA256 = "aee43b3e97988d3dad7fd8d8794373993071e73c95afdc30165d8da63016185c"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_registration_freezes_two_different_emotional_ideas() -> None:
    value = json.loads(REGISTRATION.read_text())
    assert sha256(REGISTRATION) == REGISTRATION_SHA256
    assert value["global_modal_gpu_limit"] == 2
    assert value["hardware"]["max_parallel_gpu_workers"] == 2
    assert [arm["label"] for arm in value["arms"]] == [
        "joy_u2_h6_seed194",
        "celebration_tail_u4_h20_seed195",
    ]
    assert [arm["seed"] for arm in value["arms"]] == [194, 195]
    assert [arm["curve_steps"] for arm in value["arms"]] == [
        [0, 2, 4, 6],
        [0, 4, 10, 20],
    ]
    targets = [word for arm in value["arms"] for word in arm["target_words"]]
    assert targets == ["joy", "yay", "great", "success", "nice"]
    assert "solved" not in [word.lower() for word in targets]
    assert value["shared"]["correctness_reward_used_for_training"] is False
    assert value["interpretation"]["cannot_support_significance"] is True


def test_materialized_configs_are_exact_online_jlens_runs() -> None:
    cases = {
        "emotional_parallel_v2_joy.json": {
            "seed": 194,
            "updates": 6,
            "validation_steps": [2, 4, 6],
            "target_words": ["joy"],
            "wandb_run_id": "dev-v11-parallel-joy-u2-h6-seed194",
        },
        "emotional_parallel_v2_celebration.json": {
            "seed": 195,
            "updates": 20,
            "validation_steps": [4, 10, 20],
            "target_words": ["yay", "great", "success", "nice"],
            "wandb_run_id": "dev-v11-parallel-celebration-tail-u4-h20-seed195",
        },
    }
    for name, expected in cases.items():
        config = load_config(ROOT / "configs" / name)
        for key, value in expected.items():
            assert config[key] == value
        assert config["reward_type"] == "jlens"
        assert config["registration_sha256"] == REGISTRATION_SHA256
        assert config["wandb_mode"] == "online"
        assert config["wandb_resume"] == "never"
        assert config["validation_observational_only"] is True
        assert config["require_clean_repository"] is True
        assert config["evidence_eligibility"].startswith("development_only")


def test_launcher_has_two_gpu_ceiling_fresh_volume_and_terminal_receipts() -> None:
    source = SCRIPT.read_text()
    assert 'MAX_PARALLEL_GPUS = 2' in source
    assert 'max_containers=MAX_PARALLEL_GPUS' in source
    assert 'calls = {label: train_arm.spawn(label, claim_id) for label in ARM_ORDER}' in source
    assert 'create_if_missing=False, version=2' in source
    assert 'wandb_terminal_publish_receipt.json' in source
    assert 'retry_or_resume_permitted": False' in source
    assert 'sealed_final_indices.json' not in source
    assert 'future_reserve_indices.json' not in source
    assert 'word_correlation' not in source


def test_launcher_binds_every_registered_runtime_file() -> None:
    from modal_emotional_parallel_v2 import EXPECTED_FILE_SHA256 as identities

    for relative, expected in identities.items():
        assert sha256(ROOT / relative) == expected
