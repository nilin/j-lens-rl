from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts import confirmatory_v9_local_protocol as protocol
from scripts import confirmatory_v9_local_runner as runner


def test_science_lock_is_exact_and_emotional_only() -> None:
    recipe = protocol._validate_recipe(protocol.read_json(protocol.RECIPE_SOURCE))
    assert recipe["reward"]["target_words"] == ["damn", "fuck"]
    assert recipe["reward"]["correctness_reward_functions"] == []
    assert recipe["reward"]["treatment_score_components"] == [
        {
            "layer": 8,
            "start_fraction": 0.5,
            "end_fraction": 1.0,
            "aggregation": "mean",
            "weight": -1.0,
        }
    ]
    assert recipe["reward"]["matched_control_score_components"][0]["weight"] == 1.0


def test_fresh_seeds_curve_and_final_are_fixed() -> None:
    recipe = protocol.read_json(protocol.RECIPE_SOURCE)
    assert recipe["training"]["seeds"] == list(range(208, 216))
    assert recipe["curve_gate"]["steps"] == [0, 4, 10, 20]
    assert recipe["sealed_final"]["labels"] == list(protocol.FINAL_LABELS)


def test_real_authoritative_v7_closeout_and_retirement_validate() -> None:
    amendment = protocol.read_json(protocol.AMENDMENT_SOURCE)
    assert protocol._validate_closeout(amendment).name == (
        "v7_profanity_authoritative_closeout.json"
    )


def test_real_v8_closeout_and_all_enumerated_evidence_validate() -> None:
    amendment = protocol.read_json(protocol.AMENDMENT_SOURCE)
    closeout = protocol._validate_v8_closeout(amendment)
    assert closeout.name == "v8_local_terminal_closeout.json"
    assert len(protocol.read_json(closeout)["durable_source_evidence"]) == 20


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("launch_enabled", False),
        ("v7_claim_id", "0" * 32),
        ("require_final_outcomes_unopened", False),
        ("require_modal_gpu_lease_resolved", False),
        ("v7_gpu_lease_retirement_receipt_sha256", "0" * 64),
    ],
)
def test_amendment_mutations_fail_closed(field: str, value: object) -> None:
    amendment = copy.deepcopy(protocol.read_json(protocol.AMENDMENT_SOURCE))
    amendment[field] = value
    with pytest.raises(protocol.ProtocolError):
        protocol._validate_closeout(amendment)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("v8_terminal_closeout_sha256", "0" * 64),
        ("v8_claim_id", "0" * 32),
        ("v8_failed_label", "jlens_seed201"),
        ("require_v8_final_outcomes_unopened", False),
        ("require_v8_attempt_permanently_closed", False),
    ],
)
def test_v8_closeout_amendment_mutations_fail_closed(
    field: str, value: object
) -> None:
    amendment = copy.deepcopy(protocol.read_json(protocol.AMENDMENT_SOURCE))
    amendment[field] = value
    with pytest.raises(protocol.ProtocolError):
        protocol._validate_v8_closeout(amendment)


def test_terminal_tree_identity_exactly_matches_shared_trainer(tmp_path: Path) -> None:
    from jlens_rl import train

    tree = tmp_path / "checkpoint-20"
    tree.mkdir()
    (tree / "adapter_model.safetensors").write_bytes(b"adapter")
    (tree / "trainer_state.json").write_text('{"global_step": 20}\n')
    assert protocol._tree_identity(tree) == train._tree_identity(tree)
    assert protocol._tree_identity(tree)["path"] == str(tree.resolve())


def test_curve_predicate_is_exact() -> None:
    assert protocol._curve_means_pass([0.38, 0.39, 0.39, 0.40])
    assert not protocol._curve_means_pass([0.38, 0.38, 0.39, 0.40])
    assert not protocol._curve_means_pass([0.38, 0.40, 0.39, 0.41])
    assert not protocol._curve_means_pass([0.38, 0.40, 0.41, 0.40])


def test_runtime_environment_pins_frozen_source_gpu_and_offline_wandb() -> None:
    env = runner.runtime_environment("jlens_seed208")
    assert env["CUDA_VISIBLE_DEVICES"] == protocol.GPU_UUID
    assert env["JLENS_REPOSITORY_ROOT"] == str(protocol.RUNTIME_WORKTREE.resolve())
    assert env["PYTHONPATH"].split(":")[0] == str(
        protocol.RUNTIME_WORKTREE.resolve() / "src"
    )
    assert env["WANDB_MODE"] == "offline"
    assert env["WANDB_DIR"] == str(
        protocol.OFFLINE_WANDB_DIR / "jlens_seed208"
    )


def test_code_identity_uses_only_tracked_trl_entries() -> None:
    identity = protocol._code_identity()
    tracked = protocol._git("ls-files", "--", "trl").splitlines()
    assert identity["trl_file_count"] == len(tracked) == 405
    assert "egg-info" not in json.dumps(identity)


def test_launch_plan_is_serially_gated_and_offline() -> None:
    recipe = protocol._validate_recipe(protocol.read_json(protocol.RECIPE_SOURCE))
    code = protocol._code_identity()
    configs = protocol._materialize_configs(
        recipe, "a" * 64, protocol.sha256_file(protocol.RECIPE_SOURCE), code, "b" * 64
    )
    plan = protocol._launch_plan(configs)
    assert plan["phase_order"][0].startswith("eight treatments")
    assert "curve gate" in plan["phase_order"][1]
    assert "controls" in plan["phase_order"][2]
    assert plan["wandb"]["mode"] == "offline"
    assert list(configs)[:8] == [f"jlens_seed{seed}" for seed in protocol.SEEDS]


def test_design_verification_reports_launch_authorization_not_preparation() -> None:
    result = protocol.verify_design()
    assert result["launch_enabled"] is True
    assert result["status"].endswith("pending_clean_committed_pushed_source")
    assert not protocol.STATE_PATH.exists()


def test_shell_launcher_uses_package_module_entrypoints() -> None:
    launcher = Path("run_confirmatory_v9_local.sh").read_text()
    assert "-m scripts.confirmatory_v9_local_protocol" in launcher
    assert "-m scripts.confirmatory_v9_local_runner" in launcher
