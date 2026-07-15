"""Publish the verified V16 + V16R aggregate evidence to W&B on CPU."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import modal


ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "protocol_archive/v16_combined_public_evidence/evidence"
APP_NAME = "j-lens-rl-publish-v16-combined-evidence-20260715a"
WANDB_ENTITY = "nilinabra-spare-time"
WANDB_PROJECT = "j-lens-rl"
WANDB_ID = "dev-v16-v14-celebration-combined-n16-aggregate"
WANDB_GROUP = "dev-v16-v14-celebration-n16-u2-h10-combined"
FILES = (
    "aggregate.json",
    "aggregate_curve.csv",
    "curve_rows.csv",
    "per_seed_effects.csv",
    "aggregate_curve.svg",
)
REMOTE = Path("/evidence")

image = modal.Image.debian_slim(python_version="3.11").pip_install("wandb==0.28.0")
for name in FILES:
    image = image.add_local_file(EVIDENCE / name, (REMOTE / name).as_posix(), copy=True)

app = modal.App(APP_NAME)
wandb_secret = modal.Secret.from_name(
    "j-lens-rl-wandb", required_keys=["WANDB_API_KEY"]
)


@app.function(image=image, cpu=1, memory=2048, timeout=20 * 60, retries=0, secrets=[wandb_secret])
def publish(source_commit: str) -> dict[str, Any]:
    import wandb

    aggregate = json.loads((REMOTE / "aggregate.json").read_text())
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        id=WANDB_ID,
        name=WANDB_ID,
        group=WANDB_GROUP,
        job_type="aggregate-evidence",
        resume="never",
        config={
            "source_commit": source_commit,
            "classification": aggregate["classification"],
            "included_complete_pair_seeds": aggregate["included_complete_pair_seeds"],
            "excluded_pair": aggregate["excluded_pair"],
            "global_steps": aggregate["steps"],
            "all_nodes_retained": True,
            "protected_final_payloads_accessed": False,
        },
        tags=[
            "development-only",
            "adaptive-v14-extension",
            "celebration-family",
            "n16-complete-pairs",
            "all-eval-nodes",
            "combined-v16-v16r",
        ],
    )
    run.define_metric("train/global_step")
    run.define_metric("aggregate/*", step_metric="train/global_step")
    treatment_means: list[float] = []
    control_means: list[float] = []
    for step in aggregate["steps"]:
        treatment = aggregate["curve_summary"]["jlens"][str(step)]
        control = aggregate["curve_summary"]["signflip"][str(step)]
        treatment_means.append(treatment["mean"])
        control_means.append(control["mean"])
        run.log(
            {
                "train/global_step": step,
                "aggregate/treatment_exact_match_mean": treatment["mean"],
                "aggregate/treatment_exact_match_sd": treatment["sample_sd"],
                "aggregate/treatment_exact_match_sem": treatment["sem"],
                "aggregate/signflip_exact_match_mean": control["mean"],
                "aggregate/signflip_exact_match_sd": control["sample_sd"],
                "aggregate/signflip_exact_match_sem": control["sem"],
            }
        )
    run.log(
        {
            "aggregate/complete_eval_curve": wandb.plot.line_series(
                xs=aggregate["steps"],
                ys=[treatment_means, control_means],
                keys=["positive celebration J reward", "sign-flip control"],
                title="Complete 16-seed GSM8K eval curves",
                xname="global optimizer step",
            )
        }
    )
    primary = aggregate["tests"]["primary_treatment_integrated_vs_baseline"]
    matched = aggregate["tests"]["matched_treatment_minus_signflip_integrated"]
    run.summary.update(
        {
            "evidence/n_pairs": aggregate["n_pairs"],
            "evidence/treatment_integrated_mean": primary["mean"],
            "evidence/treatment_integrated_positive_seeds": primary["positive"],
            "evidence/treatment_integrated_sign_p_two_sided": primary["p_two_sided_exact"],
            "evidence/matched_integrated_mean": matched["mean"],
            "evidence/matched_integrated_positive_seeds": matched["positive"],
            "evidence/matched_integrated_negative_seeds": matched["negative"],
            "evidence/matched_integrated_ties": matched["ties"],
            "evidence/matched_integrated_sign_p_two_sided": matched["p_two_sided_exact"],
            "evidence/early_shape_pass": aggregate["early_shape"]["pass"],
            "evidence/all_nodes_retained": True,
            "evidence/protected_final_payloads_accessed": False,
        }
    )
    artifact = wandb.Artifact(
        name="v16-v16r-combined-public-evidence",
        type="evaluation-evidence",
        metadata={
            "source_commit": source_commit,
            "n_pairs": aggregate["n_pairs"],
            "steps": aggregate["steps"],
            "classification": aggregate["classification"],
        },
    )
    for name in FILES:
        artifact.add_file(str(REMOTE / name), name=name)
    logged = run.log_artifact(artifact)
    logged.wait()
    receipt = {
        "schema_version": 1,
        "source_commit": source_commit,
        "wandb_entity": WANDB_ENTITY,
        "wandb_project": WANDB_PROJECT,
        "wandb_run_id": WANDB_ID,
        "wandb_url": run.url,
        "wandb_group": WANDB_GROUP,
        "artifact_name": logged.name,
        "artifact_id": logged.id,
        "artifact_digest": logged.digest,
        "artifact_version": logged.version,
        "global_steps": aggregate["steps"],
        "all_nodes_retained": True,
        "protected_final_payloads_accessed": False,
    }
    run.finish()
    return receipt


@app.local_entrypoint()
def main() -> None:
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT,
        text=True,
    )
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    pushed = subprocess.check_output(["git", "rev-parse", "origin/main"], cwd=ROOT, text=True).strip()
    if status or head != pushed:
        raise RuntimeError("aggregate publication requires an exact clean pushed main")
    receipt = publish.remote(head)
    path = EVIDENCE / "wandb_aggregate_publish_receipt.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
