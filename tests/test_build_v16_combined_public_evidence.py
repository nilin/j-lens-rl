import json
from pathlib import Path

from scripts import build_v16_combined_public_evidence as build
import modal_publish_v16_combined as publisher


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "protocol_archive/v16_combined_public_evidence"


def test_combined_archive_rebuilds_exact_registered_results():
    build.main()
    aggregate = json.loads((ARCHIVE / "evidence/aggregate.json").read_text())
    assert aggregate["included_complete_pair_seeds"] == [
        248, 249, 250, 251, 252, 253, 254, 255,
        257, 258, 259, 260, 261, 262, 263, 264,
    ]
    assert aggregate["excluded_pair"]["seed"] == 256
    assert aggregate["excluded_pair"]["partial_treatment_curve"] == {
        "0": 0.3825, "2": 0.3925, "4": 0.395, "6": 0.4125
    }
    primary = aggregate["tests"]["primary_treatment_integrated_vs_baseline"]
    assert primary["positive"] == 16
    assert primary["negative"] == primary["ties"] == 0
    assert primary["mean"] == 0.010218750000000006
    assert primary["p_two_sided_exact"] == 3.0517578125e-05
    matched = aggregate["tests"]["matched_treatment_minus_signflip_integrated"]
    assert (matched["positive"], matched["negative"], matched["ties"]) == (5, 9, 2)
    assert matched["mean"] == -0.0013124999999999977
    assert matched["p_two_sided_exact"] == 0.4239501953125


def test_complete_plot_grid_and_shape_are_not_selected():
    aggregate = json.loads((ARCHIVE / "evidence/aggregate.json").read_text())
    assert aggregate["steps"] == [0, 2, 4, 6, 8, 10]
    assert aggregate["all_nodes_retained"] is True
    assert aggregate["early_shape"] == {
        "steps": [0, 2, 4, 6],
        "treatment_means": [0.3825, 0.39453125, 0.39359375, 0.3896875],
        "requires": "M2>M0 and M4>=M2 and M6>=M4",
        "pass": False,
    }
    assert aggregate["individual_three_strict_rise_seeds"] == {
        "jlens": [251], "signflip": [251, 258]
    }
    rows = (ARCHIVE / "evidence/curve_rows.csv").read_text().splitlines()
    assert len(rows) == 1 + 16 * 2 * 6


def test_archive_excludes_weights_secrets_and_protected_payloads():
    forbidden_suffixes = {".safetensors", ".pt", ".pth", ".bin"}
    forbidden_names = {
        "sealed_final_indices.json",
        "future_reserve_indices.json",
        "retired_v3_curve_indices.json",
    }
    files = [path for path in ARCHIVE.rglob("*") if path.is_file()]
    assert not [path for path in files if path.suffix in forbidden_suffixes]
    assert not [path for path in files if path.name in forbidden_names]
    aggregate = json.loads((ARCHIVE / "evidence/aggregate.json").read_text())
    assert aggregate["protected_final_payloads_accessed"] is False


def test_wandb_publisher_uses_canonical_global_step_and_complete_files():
    assert publisher.WANDB_ID == "dev-v16-v14-celebration-combined-n16-aggregate"
    assert set(publisher.FILES) == {
        "aggregate.json",
        "aggregate_curve.csv",
        "curve_rows.csv",
        "per_seed_effects.csv",
        "aggregate_curve.svg",
    }
    source = (ROOT / "modal_publish_v16_combined.py").read_text()
    assert 'run.define_metric("train/global_step")' in source
    assert 'step_metric="train/global_step"' in source
    assert '"evidence/all_nodes_retained": True' in source
    assert '"evidence/early_shape_pass"' in source

    receipt = json.loads(
        (ARCHIVE / "evidence/wandb_aggregate_publish_receipt.json").read_text()
    )
    assert receipt["source_commit"] == "0d80c3dfb30ca74f834f82e42ecb09696a5e77c9"
    assert receipt["global_steps"] == [0, 2, 4, 6, 8, 10]
    assert receipt["all_nodes_retained"] is True
    assert receipt["artifact_digest"] == "1e6b4c0966c21dc465986459d83b76f9"
