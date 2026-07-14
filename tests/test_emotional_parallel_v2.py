from __future__ import annotations

import hashlib
import json
from pathlib import Path

from jlens_rl.common import load_config


ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT / "protocol_archive" / "emotional_parallel_v3_registration.json"
SCRIPT = ROOT / "modal_emotional_parallel_v2.py"
REGISTRATION_SHA256 = "6eeee93e2cca1d5c4167eda682bf710940ba30a2b971bf85e82f479b9329e4dc"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_registration_freezes_five_different_emotional_ideas() -> None:
    value = json.loads(REGISTRATION.read_text())
    assert sha256(REGISTRATION) == REGISTRATION_SHA256
    assert value["global_modal_gpu_limit"] == 5
    assert value["hardware"]["max_parallel_gpu_workers"] == 5
    assert [arm["label"] for arm in value["arms"]] == [
        "joy_u2_h6_seed194",
        "celebration_tail_u4_h20_seed195",
        "excited_u2_h6_seed196",
        "wow_u2_h6_seed197",
        "fuck_penalty_u2_h6_seed198",
    ]
    assert [arm["seed"] for arm in value["arms"]] == [194, 195, 196, 197, 198]
    assert [arm["curve_steps"] for arm in value["arms"]] == [
        [0, 2, 4, 6],
        [0, 4, 10, 20],
        [0, 2, 4, 6],
        [0, 2, 4, 6],
        [0, 2, 4, 6],
    ]
    targets = [word for arm in value["arms"] for word in arm["target_words"]]
    assert targets == [
        "joy",
        "yay",
        "great",
        "success",
        "nice",
        "excited",
        "wow",
        "fuck",
    ]
    assert "solved" not in [word.lower() for word in targets]
    assert value["shared"]["correctness_reward_used_for_training"] is False
    assert value["interpretation"]["cannot_support_significance"] is True


def test_materialized_configs_are_exact_online_jlens_runs() -> None:
    cases = {
        "emotional_parallel_v3_joy.json": {
            "seed": 194,
            "updates": 6,
            "validation_steps": [2, 4, 6],
            "target_words": ["joy"],
            "wandb_run_id": "dev-v12-five-joy-u2-h6-seed194",
        },
        "emotional_parallel_v3_celebration.json": {
            "seed": 195,
            "updates": 20,
            "validation_steps": [4, 10, 20],
            "target_words": ["yay", "great", "success", "nice"],
            "wandb_run_id": "dev-v12-five-celebration-tail-u4-h20-seed195",
        },
        "emotional_parallel_v3_excited.json": {
            "seed": 196,
            "updates": 6,
            "validation_steps": [2, 4, 6],
            "target_words": ["excited"],
            "wandb_run_id": "dev-v12-five-excited-u2-h6-seed196",
        },
        "emotional_parallel_v3_wow.json": {
            "seed": 197,
            "updates": 6,
            "validation_steps": [2, 4, 6],
            "target_words": ["wow"],
            "wandb_run_id": "dev-v12-five-wow-u2-h6-seed197",
        },
        "emotional_parallel_v3_fuck.json": {
            "seed": 198,
            "updates": 6,
            "validation_steps": [2, 4, 6],
            "target_words": ["fuck"],
            "wandb_run_id": "dev-v12-five-fuck-penalty-u2-h6-seed198",
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


def test_launcher_has_five_gpu_ceiling_fresh_volume_and_terminal_receipts() -> None:
    source = SCRIPT.read_text()
    assert 'MAX_PARALLEL_GPUS = 5' in source
    assert 'max_containers=MAX_PARALLEL_GPUS' in source
    assert 'calls = {label: train_arm.spawn(label, claim_id) for label in ARM_ORDER}' in source
    assert 'create_if_missing=False, version=2' in source
    assert 'PYTHONDONTWRITEBYTECODE": "1"' in source
    assert '".gitignore": (' in source
    assert "j-lens-rl-development-emotional-parallel-v3-20260714b" in source
    assert 'wandb_terminal_publish_receipt.json' in source
    assert 'retry_or_resume_permitted": False' in source
    assert 'sealed_final_indices.json' not in source
    assert 'future_reserve_indices.json' not in source
    assert 'word_correlation' not in source


def test_launcher_binds_every_registered_runtime_file() -> None:
    from modal_emotional_parallel_v2 import EXPECTED_FILE_SHA256 as identities

    for relative, expected in identities.items():
        assert sha256(ROOT / relative) == expected
