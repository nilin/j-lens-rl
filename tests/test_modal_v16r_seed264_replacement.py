import json
from pathlib import Path

import modal_v16r_seed264_replacement as runner


ROOT = Path(__file__).resolve().parents[1]


def test_replacement_rule_is_fixed_and_yields_16_complete_pairs():
    registration = json.loads((ROOT / runner.REGISTRATION_PATH).read_text())
    assert registration["replacement_rule"] == {
        "trigger": "first V16 treatment run lacking a verified terminal completion solely because of infrastructure preemption",
        "exclude_pair": 256,
        "replacement_seed": 264,
        "replacement_conditions": ["jlens", "signflip"],
        "rule_fixed_before_seed264_launch": True,
        "no_result_based_seed_choice": True,
    }
    cohort = registration["combined_complete_pair_cohort"]
    assert len(cohort) == len(set(cohort)) == 16
    assert 256 not in cohort
    assert cohort[-1] == 264


def test_seed264_changes_no_v14_science_relative_to_v16_seed263():
    allowed = {
        "evidence_eligibility",
        "output_dir",
        "registration_sha256",
        "run_name",
        "seed",
        "wandb_group",
        "wandb_run_id",
        "wandb_tags",
        "wandb_url",
    }
    for condition in runner.CONDITIONS:
        old = runner._load_config(
            ROOT / f"configs/v16_v14_manyseed_curve_{condition}_seed263.json"
        )
        new = runner._load_config(ROOT / runner.CONFIG_PATHS[condition])
        changed = {key for key in set(old) | set(new) if old.get(key) != new.get(key)}
        assert changed == allowed


def test_pair_differs_only_by_reward_sign_and_public_identity():
    runner._validate_local_inputs()
    treatment = runner._load_config(ROOT / runner.CONFIG_PATHS["jlens"])
    control = runner._load_config(ROOT / runner.CONFIG_PATHS["signflip"])
    assert [component["weight"] for component in treatment["score_components"]] == [1, 0.25]
    assert [component["weight"] for component in control["score_components"]] == [-1, -0.25]
    assert treatment["validation_steps"] == control["validation_steps"] == [2, 4, 6, 8, 10]
    assert treatment["updates"] == control["updates"] == 10


def test_workers_use_ephemeral_training_and_terminal_only_volume_copy():
    source = (ROOT / runner.RUNNER_PATH).read_text()
    assert "volumes={EVIDENCE_ROOT: volume}" in source
    assert "LOCAL_RUN_ROOT / f\"{condition}_seed264\"" in source
    assert "shutil.copytree(local_dir, terminal_dir)" in source
    assert "attempt_number = len(prior) + 1" in source
    assert "terminal_public_run_verified" in source
