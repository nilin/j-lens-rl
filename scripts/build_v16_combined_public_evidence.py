#!/usr/bin/env python3
"""Verify and summarize the 16-pair V16 + V16R public evidence archive."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "protocol_archive/v16_combined_public_evidence"
ORIGINAL = ARCHIVE / "original"
RECOVERY = ARCHIVE / "recovery"
EVIDENCE = ARCHIVE / "evidence"
STEPS = (0, 2, 4, 6, 8, 10)
POST_STEPS = STEPS[1:]
SEEDS = (248, 249, 250, 251, 252, 253, 254, 255, 257, 258, 259, 260, 261, 262, 263, 264)
EXCLUDED_SEED = 256
CONDITIONS = ("jlens", "signflip")
CURVE_MANIFEST_SHA256 = "ad348fe17d2e6bd6aac691d9bcdbb9da481f675305fa0e05c68e86dad97451c1"
V16_CLAIM_ID = "906eefc5089c4e928a7e6f165ff07108"
V16R_CLAIM_ID = "c72b1500666c4da4be9cee3840c86e7b"
V16R_REGISTRATION_SHA256 = "34cb51a1c748af00fe90ba4cc79e8a218d969ffea8d2551425cd225a7ea13fb6"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def run_dir(condition: str, seed: int) -> Path:
    source = RECOVERY if seed == 264 else ORIGINAL
    return source / "runs" / f"{condition}_seed{seed}"


def completion(condition: str, seed: int) -> dict[str, Any]:
    source = RECOVERY if seed == 264 else ORIGINAL
    value = load_json(source / "dispatches" / f"{condition}_seed{seed}.completion.json")
    expected_claim = V16R_CLAIM_ID if seed == 264 else V16_CLAIM_ID
    if (
        value.get("claim_id") != expected_claim
        or value.get("condition") != condition
        or value.get("seed") != seed
        or value.get("status") != "terminal_public_run_verified"
    ):
        raise RuntimeError(f"invalid completion identity: {condition=} {seed=}")
    return value


def load_curve(condition: str, seed: int) -> dict[int, float]:
    directory = run_dir(condition, seed)
    rows = [
        json.loads(line)
        for line in (directory / "validation_history.jsonl").read_text().splitlines()
        if line
    ]
    if [row.get("step") for row in rows] != list(STEPS):
        raise RuntimeError(f"incomplete curve: {condition=} {seed=}")
    result: dict[int, float] = {}
    for row in rows:
        exact = row.get("exact_match")
        if (
            isinstance(exact, bool)
            or not isinstance(exact, (int, float))
            or not math.isfinite(float(exact))
            or not 0 <= float(exact) <= 1
            or row.get("validation_source") != "train"
            or row.get("validation_indices_sha256") != CURVE_MANIFEST_SHA256
            or not math.isclose(float(exact) * 400, round(float(exact) * 400), abs_tol=1e-7)
        ):
            raise RuntimeError(f"invalid curve row: {condition=} {seed=} {row=}")
        result[int(row["step"])] = float(exact)
    complete = completion(condition, seed)
    if complete.get("validation_history_sha256") != sha256(directory / "validation_history.jsonl"):
        raise RuntimeError(f"completion/history mismatch: {condition=} {seed=}")
    receipt = directory / "wandb_terminal_publish_receipt.json"
    if complete.get("wandb_terminal_publish_receipt_sha256") != sha256(receipt):
        raise RuntimeError(f"completion/W&B receipt mismatch: {condition=} {seed=}")
    config = load_json(directory / "resolved_config.json")
    if (
        config.get("seed") != seed
        or config.get("updates") != 10
        or config.get("eval_every") != 2
        or config.get("validation_steps") != list(POST_STEPS)
        or config.get("reward_type") != "jlens"
        or config.get("validation_observational_only") is not True
        or config.get("target_words") != ["yay", "great", "success", "nice"]
    ):
        raise RuntimeError(f"science/config mismatch: {condition=} {seed=}")
    return result


def exact_two_sided_sign(values: list[float]) -> dict[str, Any]:
    epsilon = 1e-12
    positives = sum(value > epsilon for value in values)
    negatives = sum(value < -epsilon for value in values)
    ties = len(values) - positives - negatives
    n = positives + negatives
    tail = min(positives, negatives)
    probability = min(
        1.0,
        2.0 * sum(math.comb(n, k) for k in range(tail + 1)) / (2**n),
    ) if n else 1.0
    return {
        "n_total": len(values),
        "n_nonzero": n,
        "positive": positives,
        "negative": negatives,
        "ties": ties,
        "mean": statistics.fmean(values),
        "p_two_sided_exact": probability,
    }


def sample_summary(values: list[float]) -> dict[str, float]:
    sd = statistics.stdev(values)
    return {
        "mean": statistics.fmean(values),
        "sample_sd": sd,
        "sem": sd / math.sqrt(len(values)),
    }


def fmt(value: float) -> str:
    return f"{value:.8f}".rstrip("0").rstrip(".")


def write_svg(aggregate: dict[str, Any]) -> None:
    width, height = 900, 540
    left, right, top, bottom = 85, 35, 55, 75
    plot_w, plot_h = width - left - right, height - top - bottom
    y_min, y_max = 0.36, 0.415

    def x(step: int) -> float:
        return left + step / 10 * plot_w

    def y(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    colors = {"jlens": "#e45756", "signflip": "#4c78a8"}
    names = {"jlens": "Positive celebration J reward", "signflip": "Sign-flip control"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.axis{stroke:#333;stroke-width:1.3}.grid{stroke:#ddd;stroke-width:1}.curve{fill:none;stroke-width:3}.err{stroke-width:1.5}</style>',
        '<text x="450" y="28" text-anchor="middle" font-size="19" font-weight="bold">V14 recipe: complete 16-seed eval curves (mean ± SEM)</text>',
    ]
    for tick in (0.36, 0.37, 0.38, 0.39, 0.40, 0.41):
        yy = y(tick)
        parts.extend(
            [
                f'<line class="grid" x1="{left}" y1="{yy:.2f}" x2="{width-right}" y2="{yy:.2f}"/>',
                f'<text x="{left-12}" y="{yy+5:.2f}" text-anchor="end" font-size="13">{tick:.2f}</text>',
            ]
        )
    for step in STEPS:
        xx = x(step)
        parts.extend(
            [
                f'<line class="grid" x1="{xx:.2f}" y1="{top}" x2="{xx:.2f}" y2="{height-bottom}"/>',
                f'<text x="{xx:.2f}" y="{height-bottom+25}" text-anchor="middle" font-size="13">{step}</text>',
            ]
        )
    parts.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}"/>',
            f'<line class="axis" x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}"/>',
            f'<text x="{left+plot_w/2:.2f}" y="{height-18}" text-anchor="middle" font-size="15">global optimizer step</text>',
            f'<text x="20" y="{top+plot_h/2:.2f}" text-anchor="middle" font-size="15" transform="rotate(-90 20 {top+plot_h/2:.2f})">GSM8K exact match</text>',
        ]
    )
    for index, condition in enumerate(CONDITIONS):
        summaries = aggregate["curve_summary"][condition]
        points = " ".join(f"{x(step):.2f},{y(summaries[str(step)]['mean']):.2f}" for step in STEPS)
        color = colors[condition]
        parts.append(f'<polyline class="curve" stroke="{color}" points="{points}"/>')
        for step in STEPS:
            mean = summaries[str(step)]["mean"]
            sem = summaries[str(step)]["sem"]
            xx, y1, y2 = x(step), y(mean - sem), y(mean + sem)
            parts.extend(
                [
                    f'<line class="err" stroke="{color}" x1="{xx:.2f}" y1="{y1:.2f}" x2="{xx:.2f}" y2="{y2:.2f}"/>',
                    f'<line class="err" stroke="{color}" x1="{xx-5:.2f}" y1="{y1:.2f}" x2="{xx+5:.2f}" y2="{y1:.2f}"/>',
                    f'<line class="err" stroke="{color}" x1="{xx-5:.2f}" y1="{y2:.2f}" x2="{xx+5:.2f}" y2="{y2:.2f}"/>',
                    f'<circle cx="{xx:.2f}" cy="{y(mean):.2f}" r="4" fill="white" stroke="{color}" stroke-width="2.5"/>',
                ]
            )
        legend_x, legend_y = 510, 55 + index * 25
        parts.extend(
            [
                f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x+32}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>',
                f'<text x="{legend_x+40}" y="{legend_y+5}" font-size="13">{names[condition]}</text>',
            ]
        )
    parts.append("</svg>\n")
    (EVIDENCE / "aggregate_curve.svg").write_text("\n".join(parts))


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    original_status = load_json(ORIGINAL / "root/attempt_status.json")
    if (
        original_status.get("stage") != "failed_closed"
        or "jlens_seed256 already has output" not in original_status.get("error", "")
        or original_status.get("protected_final_payloads_accessed") is not False
    ):
        raise RuntimeError("V16 closeout is not the expected sole preemption failure")
    partial_rows = [
        json.loads(line)
        for line in (ORIGINAL / "runs/jlens_seed256/validation_history.jsonl").read_text().splitlines()
        if line
    ]
    if [row.get("step") for row in partial_rows] != [0, 2, 4, 6]:
        raise RuntimeError("excluded seed256 partial history changed")
    if (ORIGINAL / "dispatches/jlens_seed256.completion.json").exists():
        raise RuntimeError("excluded seed256 unexpectedly has a completion")
    recovery_summary = load_json(RECOVERY / "root/summary.json")
    if (
        recovery_summary.get("status") != "complete"
        or recovery_summary.get("claim_id") != V16R_CLAIM_ID
        or recovery_summary.get("protected_final_payloads_accessed") is not False
    ):
        raise RuntimeError("V16R recovery summary changed")

    curves = {
        condition: {seed: load_curve(condition, seed) for seed in SEEDS}
        for condition in CONDITIONS
    }
    for seed in SEEDS:
        if curves["jlens"][seed][0] != curves["signflip"][seed][0]:
            raise RuntimeError(f"pair baseline mismatch for seed {seed}")
        left = load_json(run_dir("jlens", seed) / "data_indices.json")
        right = load_json(run_dir("signflip", seed) / "data_indices.json")
        if left != right:
            raise RuntimeError(f"pair data indices mismatch for seed {seed}")

    curve_summary = {
        condition: {
            str(step): sample_summary([curves[condition][seed][step] for seed in SEEDS])
            for step in STEPS
        }
        for condition in CONDITIONS
    }
    treatment_baseline: list[float] = []
    control_baseline: list[float] = []
    paired: list[float] = []
    treatment_terminal: list[float] = []
    paired_terminal: list[float] = []
    per_seed: list[dict[str, Any]] = []
    for seed in SEEDS:
        treatment = curves["jlens"][seed]
        control = curves["signflip"][seed]
        treatment_effect = statistics.fmean(treatment[step] - treatment[0] for step in POST_STEPS)
        control_effect = statistics.fmean(control[step] - control[0] for step in POST_STEPS)
        paired_effect = statistics.fmean(treatment[step] - control[step] for step in POST_STEPS)
        treatment_baseline.append(treatment_effect)
        control_baseline.append(control_effect)
        paired.append(paired_effect)
        treatment_terminal.append(treatment[10] - treatment[0])
        paired_terminal.append(treatment[10] - control[10])
        per_seed.append(
            {
                "seed": seed,
                "treatment_integrated_vs_baseline": treatment_effect,
                "control_integrated_vs_baseline": control_effect,
                "treatment_minus_control_integrated": paired_effect,
                "treatment_terminal_vs_baseline": treatment[10] - treatment[0],
                "treatment_minus_control_terminal": treatment[10] - control[10],
            }
        )
    early = [curve_summary["jlens"][str(step)]["mean"] for step in (0, 2, 4, 6)]
    aggregate = {
        "schema_version": 1,
        "protocol": "j-lens-rl-development-v16-plus-v16r-combined-complete-pairs",
        "classification": "adaptive_development_evidence_not_independent_or_familywise_corrected",
        "source_commits": {
            "v16_launch": "e11f4fbe02fcd2b1cf279a5c651f5b6adf3f5b0f",
            "v16r_launch": "e7ff8b32147428b8f04d53e1448f3e3a8a3a27e0",
        },
        "claim_ids": {"v16": V16_CLAIM_ID, "v16r": V16R_CLAIM_ID},
        "registration_sha256": V16R_REGISTRATION_SHA256,
        "included_complete_pair_seeds": list(SEEDS),
        "excluded_pair": {
            "seed": EXCLUDED_SEED,
            "reason": "treatment infrastructure-preempted after step6; pair excluded wholesale",
            "partial_treatment_curve": {str(row["step"]): float(row["exact_match"]) for row in partial_rows},
        },
        "steps": list(STEPS),
        "n_pairs": len(SEEDS),
        "curve_summary": curve_summary,
        "early_shape": {
            "steps": [0, 2, 4, 6],
            "treatment_means": early,
            "requires": "M2>M0 and M4>=M2 and M6>=M4",
            "pass": early[1] > early[0] and early[2] >= early[1] and early[3] >= early[2],
        },
        "tests": {
            "primary_treatment_integrated_vs_baseline": exact_two_sided_sign(treatment_baseline),
            "matched_treatment_minus_signflip_integrated": exact_two_sided_sign(paired),
            "control_integrated_vs_baseline": exact_two_sided_sign(control_baseline),
            "treatment_terminal_vs_baseline": exact_two_sided_sign(treatment_terminal),
            "matched_terminal_treatment_minus_signflip": exact_two_sided_sign(paired_terminal),
        },
        "individual_three_strict_rise_seeds": {
            condition: [
                seed for seed in SEEDS
                if curves[condition][seed][2] > curves[condition][seed][0]
                and curves[condition][seed][4] > curves[condition][seed][2]
                and curves[condition][seed][6] > curves[condition][seed][4]
            ]
            for condition in CONDITIONS
        },
        "per_seed": per_seed,
        "all_nodes_retained": True,
        "protected_final_payloads_accessed": False,
    }
    (EVIDENCE / "aggregate.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")

    with (EVIDENCE / "curve_rows.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "condition", "global_step", "exact_match"])
        for seed in SEEDS:
            for condition in CONDITIONS:
                for step in STEPS:
                    writer.writerow([seed, condition, step, fmt(curves[condition][seed][step])])
    with (EVIDENCE / "aggregate_curve.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["global_step", "treatment_mean", "treatment_sd", "treatment_sem", "control_mean", "control_sd", "control_sem"])
        for step in STEPS:
            treatment = curve_summary["jlens"][str(step)]
            control = curve_summary["signflip"][str(step)]
            writer.writerow([step, *(fmt(treatment[key]) for key in ("mean", "sample_sd", "sem")), *(fmt(control[key]) for key in ("mean", "sample_sd", "sem"))])
    with (EVIDENCE / "per_seed_effects.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_seed[0]))
        writer.writeheader()
        writer.writerows(per_seed)
    write_svg(aggregate)

    primary = aggregate["tests"]["primary_treatment_integrated_vs_baseline"]
    matched = aggregate["tests"]["matched_treatment_minus_signflip_integrated"]
    readme = f"""# V16 + V16R compact public evidence\n\nThis archive reconstructs 16 complete V14-recipe celebration/sign-flip seed pairs at every fixed eval node `0,2,4,6,8,10`. Included seeds are `{','.join(map(str, SEEDS))}`. Seed 256 is excluded pairwise because its treatment was Modal-preempted after step 6; its partial history remains under `original/runs/jlens_seed256`. Seed 264 was preregistered as the next-unused infrastructure replacement before launch.\n\nTreatment integrated improvement over its own baseline is positive for {primary['positive']}/{primary['n_total']} seeds, mean `{primary['mean']:.8f}`, exact two-sided sign `p={primary['p_two_sided_exact']:.8g}`. The matched treatment-minus-signflip integrated contrast has {matched['positive']} positive, {matched['negative']} negative, and {matched['ties']} tied seeds, mean `{matched['mean']:.8f}`, exact two-sided sign `p={matched['p_two_sided_exact']:.8g}`. Thus there is strong nominal adaptive evidence of learning relative to initial eval, but no evidence that positive celebration reward outperforms the matched sign-flip control. The registered aggregate early-shape gate is `{aggregate['early_shape']['pass']}`.\n\n`evidence/curve_rows.csv` contains every individual node; `aggregate_curve.csv` contains mean, sample SD, and SEM at every node; `per_seed_effects.csv` contains all registered effects; `aggregate_curve.svg` is the complete plot. `original` and `recovery` contain byte-preserved public Modal artifacts. Checkpoints, adapters, optimizer state, model weights, credentials, and protected-final data are intentionally excluded.\n"""
    (ARCHIVE / "README.md").write_text(readme)

    inventory_files = sorted(
        path for path in ARCHIVE.rglob("*")
        if path.is_file() and path.name not in {"CHECKSUMS.sha256", "inventory.json"}
    )
    inventory = {
        "schema_version": 1,
        "file_count_excluding_inventory_and_checksums": len(inventory_files),
        "files": [
            {"path": path.relative_to(ARCHIVE).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in inventory_files
        ],
    }
    (ARCHIVE / "inventory.json").write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    checksum_files = sorted(
        path for path in ARCHIVE.rglob("*")
        if path.is_file() and path.name != "CHECKSUMS.sha256"
    )
    (ARCHIVE / "CHECKSUMS.sha256").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(ARCHIVE).as_posix()}\n" for path in checksum_files)
    )
    print(json.dumps(aggregate["tests"], indent=2, sort_keys=True))
    print(json.dumps(aggregate["early_shape"], indent=2, sort_keys=True))
    print(f"archive_files={len(checksum_files) + 1}")


if __name__ == "__main__":
    main()
