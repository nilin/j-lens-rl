import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = ROOT / "protocol_archive/v5_preregistration.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_alternative_screen_is_byte_pinned_before_outcomes():
    frozen = json.loads(PREREGISTRATION.read_text())
    screen = frozen["alternative_screen"]
    assert screen["outcome_status_at_freeze"] == "not launched and not inspected"
    assert screen["all_variants_must_complete"] is True
    assert screen["code_sha256"] == sha256(ROOT / "modal_word_explore.py")
    for relative, expected in screen["config_sha256"].items():
        assert sha256(ROOT / relative) == expected
    assert screen["selection_priority"] == [
        "celebration_ultradense",
        "profanity_ultradense",
        "celebration_taper",
        "profanity_taper",
        "solved_u5_control",
        "solved_u5_low_lr",
        "solved_u5_taper",
        "solved_u5_taper_low_lr",
    ]


def test_v4_no_look_closeout_and_v5_split_are_frozen():
    frozen = json.loads(PREREGISTRATION.read_text())
    assert frozen["v4_closeout_sha256"] == sha256(
        ROOT / "protocol_archive/v4_closeout.json"
    )
    closeout = json.loads((ROOT / "protocol_archive/v4_closeout.json").read_text())
    assert closeout["attempt_stage"] == "curve_failed"
    assert closeout["final_unlocked_present"] is False
    assert closeout["evals_directory_present"] is False
    assert closeout["signflip_run_labels"] == []
    split = frozen["no_look_split"]
    assert split["source_parent_size"] == 1700
    assert split["curve_size"] == 400
    assert split["sealed_final_size"] == 1300
    assert split["curve_manifest_sha256"] == (
        "b01409c011012641be96c84bfc35cb0b352cea902e54304105efa272a3eac6b2"
    )
    assert split["sealed_final_manifest_sha256"] == (
        "6298b8e3d15b11985cf9febcd243dafd409ef07f22091388fa0793b6ebfe4228"
    )


def test_v5_acceptance_is_one_shot_and_contains_requested_curve():
    confirm = json.loads(PREREGISTRATION.read_text())["confirmation"]
    assert confirm["seeds"] == list(range(168, 176))
    assert confirm["curve_steps"] == [0, 2, 4, 6]
    assert confirm["curve_gate"] == (
        "eight-seed mean step2 > step0, step4 >= step2, step6 >= step4"
    )
    assert confirm["terminal_one_shot"] is True
    assert confirm["seed_replacements_allowed"] is False
    assert confirm["use_step_25_adapter_only"] is True
    acceptance = confirm["acceptance"]
    assert acceptance["strictly_positive_seed_effects_required"] == 8
    assert acceptance["ties_or_negative_seed_effects_allowed"] == 0
    assert acceptance["two_sided_exact_sign_p"] == 0.0078125
