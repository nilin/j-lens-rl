"""Run the frozen confirmatory protocol on Modal with at most five GPUs.

Prepare and commit the local protocol before launching. The default action
uploads only the hashed protocol state/manifests, then spawns a durable remote
orchestrator. It runs semantic seeds first, applies the fixed curve gate,
runs sign-flipped controls only on a pass, and finally performs the sealed
paired evaluation and machine significance report.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import modal


LOCAL_REPO = Path(__file__).resolve().parent
REMOTE_REPO = Path("/workspace/j-lens-rl")
REMOTE_STATE = REMOTE_REPO / ".confirmatory"
VOLUME_NAME = "j-lens-rl-confirmatory-v1-20260714b"
SEEDS = tuple(range(142, 148))
MAX_GPU_CONTAINERS = 5

app = modal.App("j-lens-rl-confirmatory-v1")
state_volume = modal.Volume.from_name(
    VOLUME_NAME,
    create_if_missing=True,
    version=2,
)
wandb_secret = modal.Secret.from_name("j-lens-rl-wandb")

repo_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .add_local_dir(
        LOCAL_REPO,
        REMOTE_REPO.as_posix(),
        copy=True,
        ignore=[
            ".venv",
            ".venv/**",
            ".env",
            "modal.sh",
            "artifacts",
            "artifacts/**",
            "runs",
            "runs/**",
            "wandb",
            "wandb/**",
            ".confirmatory",
            ".confirmatory/**",
            ".pytest_cache",
            ".pytest_cache/**",
            "**/__pycache__/**",
            "*.egg-info/**",
        ],
    )
    .add_local_file(
        LOCAL_REPO / "artifacts/qwen25_05b_solved_lens.pt",
        (REMOTE_REPO / "artifacts/qwen25_05b_solved_lens.pt").as_posix(),
        copy=True,
    )
    .add_local_file(
        LOCAL_REPO / "artifacts/qwen25_05b_solved_calibration.json",
        (REMOTE_REPO / "artifacts/qwen25_05b_solved_calibration.json").as_posix(),
        copy=True,
    )
    .workdir(REMOTE_REPO)
    .run_commands(
        "python -m pip install --upgrade pip==26.0.1",
        "python -m pip install './trl[peft]' '.[dev]'",
        "python scripts/modal_cache_assets.py",
        "python scripts/modal_finalize_image.py",
    )
    .env(
        {
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
        }
    )
)


def _run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    rendered = subprocess.run(
        command,
        cwd=REMOTE_REPO,
        check=False,
        text=True,
    )
    if check and rendered.returncode:
        raise subprocess.CalledProcessError(rendered.returncode, command)
    return rendered


def _protocol(command: str, *extra: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return _run(
        [sys.executable, "scripts/confirmatory_protocol.py", command, *extra],
        check=check,
    )


def _history_summary(condition: str, seed: int) -> dict[str, Any]:
    history_path = REMOTE_STATE / "runs" / f"{condition}_seed{seed}" / "validation_history.jsonl"
    rows = [json.loads(line) for line in history_path.read_text().splitlines() if line]
    return {
        "condition": condition,
        "seed": seed,
        "steps": [row["step"] for row in rows],
        "exact_match": [row["exact_match"] for row in rows],
        "literal_target_completion_rate": [
            row["literal_target_completion_rate"] for row in rows
        ],
    }


@app.function(
    image=repo_image,
    gpu=["L40S", "A100-40GB"],
    max_containers=MAX_GPU_CONTAINERS,
    timeout=4 * 60 * 60,
    startup_timeout=60 * 60,
    retries=0,
    single_use_containers=True,
    secrets=[wandb_secret],
    volumes={REMOTE_STATE: state_volume},
)
def train_config(condition: str, seed: int) -> dict[str, Any]:
    if condition not in {"jlens", "signflip"} or seed not in SEEDS:
        raise ValueError("training input is outside the frozen protocol")
    state_volume.reload()
    _protocol("verify")
    config = f"configs/confirmatory_{condition}_seed{seed}.json"
    try:
        _run(
            [
                sys.executable,
                "-m",
                "jlens_rl.train",
                "--config",
                config,
                "--wandb-mode",
                "online",
            ]
        )
        return _history_summary(condition, seed)
    finally:
        state_volume.commit()


@app.function(
    image=repo_image,
    gpu=["L40S", "A100-40GB"],
    max_containers=MAX_GPU_CONTAINERS,
    timeout=4 * 60 * 60,
    startup_timeout=60 * 60,
    retries=0,
    single_use_containers=True,
    volumes={REMOTE_STATE: state_volume},
)
def evaluate_label(label: str) -> dict[str, Any]:
    allowed = {"base"} | {
        f"{condition}_seed{seed}"
        for condition in ("jlens", "signflip")
        for seed in SEEDS
    }
    if label not in allowed:
        raise ValueError("evaluation label is outside the frozen protocol")
    state_volume.reload()
    _protocol("verify-unlock")
    output = REMOTE_STATE / "evals" / f"{label}.jsonl"
    if output.exists():
        _protocol("verify-eval", "--path", str(output))
        return {"label": label, "reused_verified_output": True}

    if label == "base":
        experiment_config = "configs/confirmatory_jlens_seed142.json"
        adapter_args: list[str] = []
    else:
        condition, seed_text = label.rsplit("_seed", 1)
        experiment_config = f"configs/confirmatory_{condition}_seed{seed_text}.json"
        adapter_args = [
            "--adapter",
            str(REMOTE_STATE / "runs" / label / "final"),
        ]
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        _run(
            [
                sys.executable,
                "-m",
                "jlens_rl.eval",
                "--config",
                "configs/confirmatory_sealed_eval.json",
                "--experiment-config",
                experiment_config,
                "--indices-manifest",
                str(REMOTE_STATE / "manifests/sealed_final_indices.json"),
                "--output-jsonl",
                str(output),
                "--run-label",
                label,
                "--batch-size",
                "64",
                "--skip-jlens-metric",
                *adapter_args,
            ]
        )
        _protocol("verify-eval", "--path", str(output))
        return {"label": label, "reused_verified_output": False}
    finally:
        state_volume.commit()


@app.function(
    image=repo_image,
    timeout=2 * 60 * 60,
    startup_timeout=60 * 60,
    retries=0,
    volumes={REMOTE_STATE: state_volume},
)
def protocol_step(command: str) -> dict[str, Any]:
    if command not in {"curve", "unlock"}:
        raise ValueError("unsupported protocol step")
    state_volume.reload()
    result = _protocol(command, check=False)
    state_volume.commit()
    if command == "curve":
        path = REMOTE_STATE / "evidence/curve_gate.json"
    else:
        path = REMOTE_STATE / "final_unlocked.json"
    payload = json.loads(path.read_text()) if path.exists() else {}
    payload["returncode"] = result.returncode
    return payload


def _comparison_args(labels: Iterable[str], option: str) -> list[str]:
    arguments: list[str] = []
    for label in labels:
        arguments.extend([option, str(REMOTE_STATE / "evals" / f"{label}.jsonl")])
    return arguments


@app.function(
    image=repo_image,
    timeout=2 * 60 * 60,
    startup_timeout=60 * 60,
    retries=0,
    volumes={REMOTE_STATE: state_volume},
)
def analyze_final(stage: str) -> dict[str, Any]:
    if stage not in {"semantic", "specificity"}:
        raise ValueError("unsupported analysis stage")
    state_volume.reload()
    base = str(REMOTE_STATE / "evals/base.jsonl")
    semantic_labels = [f"jlens_seed{seed}" for seed in SEEDS]
    evidence_dir = REMOTE_STATE / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    if stage == "semantic":
        output = evidence_dir / "semantic_vs_base.json"
        command = [
            sys.executable,
            "-m",
            "jlens_rl.paired_eval",
            "--base-jsonl",
            base,
            *_comparison_args(semantic_labels, "--adapter-jsonl"),
            "--output",
            str(output),
        ]
    else:
        output = evidence_dir / "semantic_vs_signflip.json"
        command = [
            sys.executable,
            "-m",
            "jlens_rl.paired_eval",
            "--base-jsonl",
            base,
            *_comparison_args(semantic_labels, "--adapter-jsonl"),
            *_comparison_args(
                [f"signflip_seed{seed}" for seed in SEEDS],
                "--control-jsonl",
            ),
            "--output",
            str(output),
        ]
    if output.exists():
        raise FileExistsError(f"refusing to overwrite analysis: {output}")
    _run(command)
    if stage == "specificity":
        report_process = _protocol("report", check=False)
        report_path = evidence_dir / "acceptance_report.json"
        report = json.loads(report_path.read_text())
        report["returncode"] = report_process.returncode
        state_volume.commit()
        return report
    state_volume.commit()
    return json.loads(output.read_text())


def _mapped_results(function: Any, *inputs: Iterable[Any]) -> list[Any]:
    results = list(
        function.map(
            *inputs,
            order_outputs=False,
            return_exceptions=True,
        )
    )
    failures = [result for result in results if isinstance(result, BaseException)]
    if failures:
        raise RuntimeError(f"{len(failures)} mapped Modal jobs failed: {failures}")
    return results


@app.function(
    image=repo_image,
    timeout=23 * 60 * 60,
    startup_timeout=60 * 60,
    retries=0,
    volumes={REMOTE_STATE: state_volume},
)
def orchestrate() -> dict[str, Any]:
    state_volume.reload()
    _protocol("verify")

    semantic = _mapped_results(train_config, ["jlens"] * len(SEEDS), SEEDS)
    gate = protocol_step.remote("curve")
    if gate.get("returncode") or not gate.get("passed"):
        return {"stage": "curve_failed", "semantic": semantic, "curve": gate}

    controls = _mapped_results(train_config, ["signflip"] * len(SEEDS), SEEDS)
    unlock = protocol_step.remote("unlock")
    if unlock.get("returncode"):
        raise RuntimeError(f"protocol unlock failed: {unlock}")

    semantic_labels = ["base", *(f"jlens_seed{seed}" for seed in SEEDS)]
    semantic_evals = _mapped_results(evaluate_label, semantic_labels)
    semantic_result = analyze_final.remote("semantic")

    control_labels = [f"signflip_seed{seed}" for seed in SEEDS]
    control_evals = _mapped_results(evaluate_label, control_labels)
    report = analyze_final.remote("specificity")
    return {
        "stage": "complete",
        "curve": gate,
        "semantic_training": semantic,
        "control_training": controls,
        "semantic_evaluations": semantic_evals,
        "control_evaluations": control_evals,
        "semantic_result": semantic_result,
        "acceptance_report": report,
    }


def _upload_protocol_state() -> None:
    subprocess.run(
        [sys.executable, "scripts/confirmatory_protocol.py", "verify"],
        cwd=LOCAL_REPO,
        check=True,
    )
    paths = [LOCAL_REPO / ".confirmatory/protocol_state.json"]
    paths.extend(sorted((LOCAL_REPO / ".confirmatory/manifests").glob("*.json")))
    with state_volume.batch_upload(force=True) as batch:
        for path in paths:
            relative = path.relative_to(LOCAL_REPO / ".confirmatory")
            batch.put_file(path, f"/{relative.as_posix()}")


@app.local_entrypoint()
def main() -> None:
    _upload_protocol_state()
    call = orchestrate.spawn()
    print(
        json.dumps(
            {
                "status": "submitted",
                "function_call_id": call.object_id,
                "volume": VOLUME_NAME,
                "max_parallel_gpus": MAX_GPU_CONTAINERS,
            },
            indent=2,
        )
    )
