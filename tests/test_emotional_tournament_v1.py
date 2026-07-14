import json
from pathlib import Path

import pytest

import modal_emotional_tournament_v1 as runner
from scripts import emotional_tournament_v1_protocol as tournament
from scripts.modal_finalize_image_tournament_v1 import (
    validate_exact_image_inventory,
)
from scripts.modal_verify_tournament_v1_volume import (
    VOLUME_NAME,
    verify_tournament_v1_volume_v2,
)


ROOT = Path(__file__).resolve().parents[1]


def test_scientific_design_is_exact_and_strictly_development_only():
    recipe = tournament._validate_recipe_lock(
        tournament._read_json(tournament.RECIPE_LOCK_SOURCE)
    )
    assert tournament.ARM_ORDER == ("fuck", "yay", "worried")
    assert tournament.WEIGHTS == {"fuck": -1.0, "yay": 1.0, "worried": -1.0}
    assert tournament.SEED == 192
    assert tournament.CURVE_STEPS == (0, 5, 10, 15)
    assert recipe["scientific_status"] == "development_only_on_exposed_curve"
    assert recipe["training"]["reward_type"] == "jlens"
    assert recipe["training"]["reward_functions"] == [
        "intrinsic_j_lens_target_word_reward"
    ]
    assert "gsm8k_correctness_training_reward" in recipe[
        "forbidden_inputs_and_claims"
    ]
    assert recipe["ranking"]["always_complete_all_three_arms"] is True


def test_templates_are_same_u5_recipe_except_word_sign_and_identity():
    hashes = tournament._validate_templates()
    assert set(hashes) == {"common", *tournament.ARM_ORDER}
    common_fields = None
    ignored = {
        "target_words",
        "calibration_path",
        "expected_calibration_sha256",
        "calibration_sha256",
        "score_components",
        "run_name",
        "wandb_run_id",
        "wandb_url",
        "output_dir",
    }
    for arm in tournament.ARM_ORDER:
        config = tournament._load_config(tournament.TEMPLATE_PATHS[arm])
        tournament._validate_template(arm, config)
        assert config["reward_type"] == "jlens"
        assert config["target_words"] == [arm]
        assert config["score_components"][0]["weight"] == tournament.WEIGHTS[arm]
        assert config["updates"] == 15
        assert config["validation_steps"] == [5, 10, 15]
        assert config["eval_every"] == 5
        assert config["seed"] == 192
        assert config["kl_beta"] == 0.02
        assert config["evidence_eligibility"].startswith("development_only")
        shared = {key: value for key, value in config.items() if key not in ignored}
        if common_fields is None:
            common_fields = shared
        else:
            assert shared == common_fields


def test_exact_existing_calibrations_and_exposed_inputs_are_reused():
    inputs = tournament._validate_source_inputs()
    assert inputs["lens_sha256"] == tournament.LENS_SHA256
    assert inputs["manifest_sha256"] == tournament.MANIFEST_SHA256
    assert inputs["calibration_sha256"] == tournament.CALIBRATION_SHA256
    for arm in tournament.ARM_ORDER:
        payload = inputs["calibration_metadata"][arm]
        assert payload["target_words"] == [arm]
        assert payload["token_ids"] == tournament.TOKEN_IDS[arm]
        assert payload["lens_sha256"] == tournament.LENS_SHA256


def test_enabled_amendment_binds_authoritative_v7_closeout():
    amendment, closeout = tournament._amendment_and_closeout()
    assert amendment["launch_enabled"] is True
    assert closeout == ROOT / "protocol_archive/v7_profanity_authoritative_closeout.json"
    assert amendment["v7_terminal_stage"] == "failed_before_final"
    assert tournament.sha256_file(closeout) == amendment["v7_terminal_closeout_sha256"]
    assert tournament._validate_amendment(amendment) == closeout


def test_shape_and_ranking_are_predeclared_and_deterministic():
    assert tournament.shape_pass({0: 0.38, 5: 0.39, 10: 0.39, 15: 0.40})
    assert not tournament.shape_pass({0: 0.38, 5: 0.38, 10: 0.40, 15: 0.41})
    assert not tournament.shape_pass({0: 0.38, 5: 0.40, 10: 0.39, 15: 0.41})
    results = [
        {
            "arm": "fuck",
            "shape_pass": False,
            "step15_minus_step0": 0.08,
            "step15": 0.46,
        },
        {
            "arm": "yay",
            "shape_pass": True,
            "step15_minus_step0": 0.01,
            "step15": 0.39,
        },
        {
            "arm": "worried",
            "shape_pass": True,
            "step15_minus_step0": 0.02,
            "step15": 0.40,
        },
    ]
    assert [item["arm"] for item in tournament.rank_results(results)] == [
        "worried",
        "yay",
        "fuck",
    ]
    tied = [
        {"arm": arm, "shape_pass": True, "step15_minus_step0": 0.01, "step15": 0.4}
        for arm in tournament.ARM_ORDER
    ]
    assert [item["arm"] for item in tournament.rank_results(tied)] == list(
        tournament.ARM_ORDER
    )
    with pytest.raises(RuntimeError, match="all three arms"):
        tournament.rank_results(tied[:2])


def test_runtime_allowlist_excludes_old_v7_runtime_and_outcome_payloads():
    names = tournament._runtime_allowlist()
    assert names == sorted(set(names))
    assert "configs/common.json" in names
    assert "modal_emotional_tournament_v1.py" in names
    assert "scripts/emotional_tournament_v1_protocol.py" in names
    for forbidden in (
        "modal_confirmatory_v7.py",
        "scripts/confirmatory_v7_protocol.py",
        "scripts/modal_verify_v7_volume.py",
        "scripts/modal_cache_assets_v7.py",
        "scripts/modal_finalize_image_v7.py",
        "run_confirmatory_v7.sh",
    ):
        assert forbidden not in names
    assert not any("protocol_archive" in Path(name).parts for name in names)
    assert not any(".confirmatory" in Path(name).parts for name in names)
    assert not any("sealed" in Path(name).name for name in names)
    assert not any("correlation" in Path(name).name for name in names)


def test_infrastructure_amendment_is_pre_outcome_and_binds_fresh_volume(
    tmp_path, monkeypatch
):
    amendment = tournament._validate_infrastructure_amendment(
        tournament._read_json(tournament.INFRASTRUCTURE_AMENDMENT_SOURCE)
    )
    assert amendment["scientific_recipe_changed"] is False
    assert amendment["outcome_data_observed_before_amendment"] is False
    assert amendment["replacement_volume"] == tournament.VOLUME_NAME
    assert amendment["added_runtime_source"] == "configs/common.json"
    assert amendment["added_runtime_source_sha256"] == tournament.sha256_file(
        ROOT / "configs/common.json"
    )
    copied_closeout = tmp_path / "preclaim_attempt_a_closeout.json"
    copied_closeout.write_bytes(
        (ROOT / amendment["preclaim_closeout_path"]).read_bytes()
    )
    remote_root = tmp_path / "remote_runtime"
    (remote_root / "configs").mkdir(parents=True)
    (remote_root / "configs/common.json").write_bytes(
        (ROOT / "configs/common.json").read_bytes()
    )
    monkeypatch.setattr(tournament, "ROOT", remote_root)
    assert tournament._validate_infrastructure_amendment(
        amendment, copied_closeout=copied_closeout
    ) == amendment


def test_modal_runner_is_one_noncreating_l40s_in_fixed_serial_order():
    assert runner.APP_NAME == tournament.APP_NAME
    assert runner.VOLUME_NAME == tournament.VOLUME_NAME
    assert runner.ARM_ORDER == ("fuck", "yay", "worried")
    assert runner.MAX_GPU_CONTAINERS == 1
    assert runner.GLOBAL_MODAL_GPU_LIMIT == 1
    assert runner.GPU_TYPE == "L40S"
    source = (ROOT / "modal_emotional_tournament_v1.py").read_text()
    assert "create_if_missing=False, version=2" in source
    assert "for index, arm in enumerate(ARM_ORDER, 1):" in source
    assert "results.append(train_arm.remote(arm, token))" in source
    assert ".map(" not in source
    assert "skip_if_exists=True" in source
    assert "gpu_lease.pop(GPU_LEASE_KEY)" in source
    assert "_protocol(\"verify-launch\")" in source
    assert "active_other_apps" in source
    assert "V7 app remains active" in source


def test_upload_boundary_names_only_exposed_manifests():
    source = (ROOT / "modal_emotional_tournament_v1.py").read_text()
    assert '"curve_indices.json"' not in source  # state uploader is generic and guarded
    assert '"sealed_final_indices.json"' in source  # explicit rejection list only
    assert "forbidden outcome payload" in source
    assert "correlation" in source  # explicit filename rejection only
    recipe = json.loads(tournament.RECIPE_LOCK_SOURCE.read_text())
    assert recipe["development_curve"]["manifest_sha256"] == tournament.CURVE_SHA256


def test_strict_image_inventory_rejects_extra_or_changed_files(tmp_path):
    (tmp_path / "ok.py").write_text("ok\n")
    digest = tournament.sha256_file(tmp_path / "ok.py")
    validate_exact_image_inventory(tmp_path, {"ok.py": digest})
    (tmp_path / "extra.py").write_text("extra\n")
    with pytest.raises(RuntimeError, match="unexpected"):
        validate_exact_image_inventory(tmp_path, {"ok.py": digest})
    (tmp_path / "extra.py").unlink()
    (tmp_path / "ok.py").write_text("changed\n")
    with pytest.raises(RuntimeError, match="wrong_hash"):
        validate_exact_image_inventory(tmp_path, {"ok.py": digest})


def test_volume_verifier_requires_existing_v2_identity():
    calls = []

    class FakeVolume:
        object_id = "vo-test"

        def hydrate(self):
            calls.append("hydrate")

    def factory(name, **kwargs):
        calls.append((name, kwargs))
        return FakeVolume()

    assert verify_tournament_v1_volume_v2(factory) == "vo-test"
    assert calls == [
        (VOLUME_NAME, {"create_if_missing": False, "version": 2}),
        "hydrate",
    ]


def test_launch_and_replay_identities_are_unique_and_frozen():
    ids = []
    for arm in tournament.ARM_ORDER:
        config = tournament._load_config(tournament.TEMPLATE_PATHS[arm])
        ids.append(config["wandb_run_id"])
        assert config["wandb_resume"] == "never"
        assert config["wandb_group"] == "dev-v8-emotional-single-u5-h15-seed192"
        assert config["wandb_url"].endswith(config["wandb_run_id"])
    assert len(ids) == len(set(ids)) == 3
    launcher = (ROOT / "run_emotional_tournament_v1.sh").read_text()
    assert "modal run --detach" in launcher
    assert "verify-launch" in launcher


def test_registration_draft_pins_current_source_identities():
    draft = json.loads(tournament.REGISTRATION_DRAFT_SOURCE.read_text())
    assert draft["status"].endswith("pending_v7_terminal_amendment")
    assert draft["launch_gate"]["enabled"] is False
    assert draft["recipe_lock_sha256"] == tournament.sha256_file(
        tournament.RECIPE_LOCK_SOURCE
    )
    assert draft["source_provenance"]["new_template_sha256"] == (
        tournament._validate_templates()
    )
    assert draft["source_provenance"][
        "calibration_source_attempt_manifest_sha256"
    ] == tournament.CALIBRATION_ATTEMPT_MANIFEST_SHA256
