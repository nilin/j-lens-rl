"""Run the frozen confirmatory protocol on Modal with at most eight GPUs.

Prepare and commit the local protocol before launching. The default action
uploads only the hashed protocol state/manifests, then spawns a durable remote
orchestrator. It runs semantic seeds first, applies the fixed curve gate,
runs sign-flipped controls only on a pass, and finally submits one immutable
17-label sealed-evaluation batch before any outcome analysis.
"""

from __future__ import annotations

import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import modal


LOCAL_REPO = Path(__file__).resolve().parent
REMOTE_REPO = Path("/workspace/j-lens-rl")
REMOTE_STATE = REMOTE_REPO / ".confirmatory"
VOLUME_NAME = "j-lens-rl-confirmatory-v4-20260714a"
SEEDS = tuple(range(159, 167))
MAX_GPU_CONTAINERS = 8
GPU_TYPE = "L40S"
SEALED_LABELS = (
    "base",
    *(f"jlens_seed{seed}" for seed in SEEDS),
    *(f"signflip_seed{seed}" for seed in SEEDS),
)

app = modal.App("j-lens-rl-confirmatory-v4")
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
    .env(
        {
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "JLENS_REPOSITORY_ROOT": REMOTE_REPO.as_posix(),
            "PYTHONPATH": (
                f"{(REMOTE_REPO / 'src').as_posix()}:"
                f"{(REMOTE_REPO / 'trl').as_posix()}"
            ),
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .run_commands(
        "python -m pip install --upgrade pip==26.0.1",
        "python -m pip install './trl[peft]' '.[dev]'",
        "python scripts/modal_cache_assets.py",
        "python scripts/modal_finalize_image.py",
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


def _verify_attempt_claim(claim_id: str) -> dict[str, Any]:
    claim_path = REMOTE_STATE / "attempt_claim.json"
    if not claim_path.is_file():
        raise RuntimeError("confirmatory volume has no attempt claim")
    claim = json.loads(claim_path.read_text())
    if claim.get("claim_id") != claim_id:
        raise RuntimeError("confirmatory volume is claimed by another launch")
    return claim


def _record_attempt_status(
    claim_id: str,
    stage: str,
    **details: Any,
) -> dict[str, Any]:
    _verify_attempt_claim(claim_id)
    status = {
        "claim_id": claim_id,
        "stage": stage,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        **details,
    }
    path = REMOTE_STATE / "attempt_status.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return status


@app.function(
    image=repo_image,
    cpu=2,
    memory=4096,
    timeout=10 * 60,
    startup_timeout=60 * 60,
    retries=0,
    volumes={REMOTE_STATE: state_volume},
)
def claim_attempt(claim_id: str) -> dict[str, Any]:
    state_volume.reload()
    _protocol("verify")
    forbidden = [
        REMOTE_STATE / "runs",
        REMOTE_STATE / "evals",
        REMOTE_STATE / "evidence",
        REMOTE_STATE / "final_unlocked.json",
    ]
    stale = [str(path) for path in forbidden if path.exists()]
    if stale:
        raise RuntimeError(f"confirmatory volume already contains attempt data: {stale}")
    claim_path = REMOTE_STATE / "attempt_claim.json"
    state = json.loads((REMOTE_STATE / "protocol_state.json").read_text())
    claim = {
        "claim_id": claim_id,
        "git_commit": state["git_commit"],
        "protocol": state["protocol"],
    }
    try:
        with claim_path.open("x") as handle:
            json.dump(claim, handle, sort_keys=True)
            handle.write("\n")
    except FileExistsError as error:
        raise RuntimeError("confirmatory volume is already claimed") from error
    _record_attempt_status(claim_id, "claimed")
    state_volume.commit()
    return claim


@app.function(
    image=repo_image,
    gpu=GPU_TYPE,
    cpu=4,
    memory=32768,
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
    if condition == "signflip":
        _protocol("verify-curve")
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
    gpu=GPU_TYPE,
    cpu=4,
    memory=32768,
    max_containers=MAX_GPU_CONTAINERS,
    timeout=4 * 60 * 60,
    startup_timeout=60 * 60,
    retries=0,
    single_use_containers=True,
    volumes={REMOTE_STATE: state_volume},
)
def evaluate_label(label: str) -> dict[str, Any]:
    if label not in SEALED_LABELS:
        raise ValueError("evaluation label is outside the frozen protocol")
    state_volume.reload()
    _protocol("verify-unlock")
    output = REMOTE_STATE / "evals" / f"{label}.jsonl"
    if output.exists():
        _protocol("verify-eval", "--path", str(output), "--label", label)
        return {"label": label, "reused_verified_output": True}

    if label == "base":
        experiment_config = "configs/confirmatory_jlens_seed159.json"
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
        _protocol("verify-eval", "--path", str(output), "--label", label)
        return {"label": label, "reused_verified_output": False}
    finally:
        state_volume.commit()


@app.function(
    image=repo_image,
    cpu=2,
    memory=4096,
    timeout=2 * 60 * 60,
    startup_timeout=60 * 60,
    retries=0,
    volumes={REMOTE_STATE: state_volume},
)
def protocol_step(command: str) -> dict[str, Any]:
    if command not in {"verify-semantic", "curve", "unlock"}:
        raise ValueError("unsupported protocol step")
    state_volume.reload()
    result = _protocol(command, check=False)
    state_volume.commit()
    if command == "verify-semantic":
        return {"verified": result.returncode == 0, "returncode": result.returncode}
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
    cpu=4,
    memory=16384,
    timeout=2 * 60 * 60,
    startup_timeout=60 * 60,
    retries=0,
    volumes={REMOTE_STATE: state_volume},
)
def analyze_final() -> dict[str, Any]:
    state_volume.reload()
    base = str(REMOTE_STATE / "evals/base.jsonl")
    semantic_labels = [f"jlens_seed{seed}" for seed in SEEDS]
    evidence_dir = REMOTE_STATE / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    output = evidence_dir / "sealed_comparison.json"
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
    try:
        _run(command)
        report_process = _protocol("report", check=False)
        report_path = evidence_dir / "acceptance.json"
        report = json.loads(report_path.read_text())
        report["returncode"] = report_process.returncode
        return report
    finally:
        state_volume.commit()


def _mapped_results(function: Any, *inputs: Iterable[Any]) -> list[Any]:
    materialized_inputs = [list(values) for values in inputs]
    results = list(
        function.map(
            *materialized_inputs,
            order_outputs=True,
            return_exceptions=True,
        )
    )
    failures = [
        {
            "inputs": [values[index] for values in materialized_inputs],
            "error": repr(result),
        }
        for index, result in enumerate(results)
        if isinstance(result, BaseException)
    ]
    if failures:
        raise RuntimeError(f"{len(failures)} mapped Modal jobs failed: {failures}")
    return results


@app.function(
    image=repo_image,
    cpu=1,
    memory=2048,
    timeout=23 * 60 * 60,
    startup_timeout=60 * 60,
    retries=0,
    volumes={REMOTE_STATE: state_volume},
)
def orchestrate(claim_id: str) -> dict[str, Any]:
    state_volume.reload()
    _verify_attempt_claim(claim_id)
    _protocol("verify")
    try:
        _record_attempt_status(claim_id, "semantic_training")
        state_volume.commit()
        semantic = _mapped_results(train_config, ["jlens"] * len(SEEDS), SEEDS)
        semantic_check = protocol_step.remote("verify-semantic")
        if semantic_check.get("returncode") or not semantic_check.get("verified"):
            raise RuntimeError(f"semantic run verification failed: {semantic_check}")
        gate = protocol_step.remote("curve")
        if gate.get("returncode") or not gate.get("passed"):
            state_volume.reload()
            _record_attempt_status(claim_id, "curve_failed", curve=gate)
            state_volume.commit()
            return {"stage": "curve_failed", "semantic": semantic, "curve": gate}

        state_volume.reload()
        _record_attempt_status(claim_id, "control_training", curve=gate)
        state_volume.commit()
        controls = _mapped_results(train_config, ["signflip"] * len(SEEDS), SEEDS)
        unlock = protocol_step.remote("unlock")
        if unlock.get("returncode"):
            raise RuntimeError(f"protocol unlock failed: {unlock}")

        state_volume.reload()
        _record_attempt_status(claim_id, "sealed_evaluation")
        state_volume.commit()
        sealed_evaluations = _mapped_results(evaluate_label, SEALED_LABELS)
        report = analyze_final.remote()
        final_stage = (
            "complete"
            if report.get("passed") and report.get("returncode") == 0
            else "significance_failed"
        )
        state_volume.reload()
        _record_attempt_status(claim_id, final_stage, acceptance=report)
        state_volume.commit()
        return {
            "stage": final_stage,
            "curve": gate,
            "semantic_training": semantic,
            "control_training": controls,
            "sealed_evaluations": sealed_evaluations,
            "acceptance_report": report,
        }
    except BaseException as error:
        try:
            state_volume.reload()
            _record_attempt_status(claim_id, "failed", error=repr(error))
            state_volume.commit()
        except BaseException:
            pass
        raise


def _upload_protocol_state() -> None:
    subprocess.run(
        [sys.executable, "scripts/confirmatory_protocol.py", "verify"],
        cwd=LOCAL_REPO,
        check=True,
    )
    paths = [LOCAL_REPO / ".confirmatory/protocol_state.json"]
    paths.extend(sorted((LOCAL_REPO / ".confirmatory/manifests").glob("*.json")))
    # Never overwrite an active attempt's prepared state. On a fresh Volume the
    # upload succeeds once; duplicate submissions fail before reaching GPUs.
    with state_volume.batch_upload(force=False) as batch:
        for path in paths:
            relative = path.relative_to(LOCAL_REPO / ".confirmatory")
            batch.put_file(path, f"/{relative.as_posix()}")


@app.local_entrypoint()
def main() -> None:
    _upload_protocol_state()
    claim_id = uuid.uuid4().hex
    claim_attempt.remote(claim_id)
    call = orchestrate.spawn(claim_id)
    print(
        json.dumps(
            {
                "status": "submitted",
                "function_call_id": call.object_id,
                "volume": VOLUME_NAME,
                "gpu_type": GPU_TYPE,
                "max_parallel_gpus": MAX_GPU_CONTAINERS,
            },
            indent=2,
        )
    )
