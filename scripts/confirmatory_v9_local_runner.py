#!/usr/bin/env python3
"""Fail-closed serial local runner for the registered V9 profanity replication."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from scripts import confirmatory_v9_local_protocol as protocol


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_exclusive_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


@contextmanager
def whole_attempt_gpu_lock() -> Iterator[None]:
    protocol.LOCAL_GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with protocol.LOCAL_GPU_LOCK.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise protocol.ProtocolError(
                "another process holds the registered V9-local GPU lock"
            ) from error
        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "protocol": protocol.PROTOCOL,
                    "acquired_at_utc": protocol.utc_now(),
                },
                sort_keys=True,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def runtime_environment(label: str) -> dict[str, str]:
    runtime = protocol.RUNTIME_WORKTREE.resolve()
    state = protocol.STATE_DIR.resolve()
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": protocol.GPU_UUID,
            "JLENS_REPOSITORY_ROOT": str(runtime),
            "JLENS_MODAL_IMAGE_SPEC": protocol.LOCAL_RUNTIME_ID,
            "PYTHONPATH": os.pathsep.join((str(runtime / "src"), str(runtime))),
            "PYTHONNOUSERSITE": "1",
            "WANDB_MODE": "offline",
            "WANDB_DIR": str(protocol.OFFLINE_WANDB_DIR / label),
            "WANDB_CACHE_DIR": str(state / "wandb_cache" / label),
            "WANDB_DATA_DIR": str(state / "wandb_data" / label),
            "WANDB_CONFIG_DIR": str(state / "wandb_config" / label),
            "WANDB_SILENT": "false",
        }
    )
    return env


def verify_frozen_imports() -> dict[str, str]:
    env = runtime_environment("preflight")
    sentinel = "__V9_FROZEN_IMPORT_IDENTITY__="
    output = subprocess.check_output(
        [
            str(protocol.PYTHON_EXECUTABLE),
            "-c",
            (
                "import json,jlens_rl,jlens_rl.train;"
                f"print('{sentinel}'+json.dumps({{'package':jlens_rl.__file__,"
                "'train':jlens_rl.train.__file__}))"
            ),
        ],
        cwd=protocol.RUNTIME_WORKTREE,
        env=env,
        text=True,
    )
    identity_lines = [
        line[len(sentinel) :]
        for line in output.splitlines()
        if line.startswith(sentinel)
    ]
    if len(identity_lines) != 1:
        raise protocol.ProtocolError("frozen import probe emitted no unique identity")
    observed = json.loads(identity_lines[0])
    runtime = protocol.RUNTIME_WORKTREE.resolve()
    for key, value in observed.items():
        try:
            Path(value).resolve().relative_to(runtime)
        except (TypeError, ValueError) as error:
            raise protocol.ProtocolError(
                f"{key} imported mutable/nonregistered source: {value!r}"
            ) from error
    return observed


def _status(value: dict[str, Any]) -> None:
    temporary = protocol.STATUS_PATH.with_name(
        f".{protocol.STATUS_PATH.name}.{uuid.uuid4().hex}.tmp"
    )
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(protocol.STATUS_PATH)
    _fsync_directory(protocol.STATUS_PATH.parent)


def _claim() -> dict[str, Any]:
    state = protocol.load_and_verify_state(require_launch=True)
    if protocol.CLAIM_PATH.is_file():
        claim = protocol.read_json(protocol.CLAIM_PATH)
        status = protocol.read_json(protocol.STATUS_PATH) if protocol.STATUS_PATH.is_file() else {}
        if (
            claim.get("protocol") != protocol.PROTOCOL
            or claim.get("git_commit") != state["git_commit"]
            or status.get("claim_id") != claim.get("claim_id")
            or status.get("stage") not in {
                "running_treatments",
                "treatments_complete_curve_passed_controls_not_started",
                "running_controls",
            }
        ):
            raise protocol.ProtocolError("existing V9-local claim is not safely resumable")
        return claim
    if any(
        path.exists()
        for path in (
            protocol.RUN_DIR,
            protocol.OFFLINE_WANDB_DIR,
            protocol.DISPATCH_DIR,
            protocol.CURVE_PATH,
            protocol.UNLOCK_PATH,
            protocol.COLLECTION_PATH,
            protocol.EVAL_DIR,
            protocol.COMPARISON_PATH,
            protocol.ANALYSIS_PROCESS_PATH,
            protocol.ACCEPTANCE_PATH,
        )
    ):
        raise protocol.ProtocolError("V9-local output exists before the one attempt claim")
    claim = {
        "schema_version": 1,
        "protocol": protocol.PROTOCOL,
        "claim_id": uuid.uuid4().hex,
        "git_commit": state["git_commit"],
        "registration_sha256": state["registration_sha256"],
        "backend": "single-local-rtx4090",
        "tracking": "offline-wandb",
        "gpu_uuid": protocol.GPU_UUID,
        "claimed_at_utc": protocol.utc_now(),
    }
    write_exclusive_json(protocol.CLAIM_PATH, claim)
    _status(
        {
            "protocol": protocol.PROTOCOL,
            "claim_id": claim["claim_id"],
            "stage": "running_treatments",
            "updated_at_utc": protocol.utc_now(),
        }
    )
    return claim


def _dispatch_paths(sequence: int, label: str) -> tuple[Path, Path, Path]:
    prefix = f"{sequence:03d}-{label}"
    return (
        protocol.DISPATCH_DIR / f"{prefix}.intent.json",
        protocol.DISPATCH_DIR / f"{prefix}.completion.json",
        protocol.DISPATCH_DIR / f"{prefix}.console.log",
    )


def _adopt_or_fail_open_intent(
    *,
    claim: dict[str, Any],
    condition: str,
    seed: int,
    sequence: int,
    intent_path: Path,
    completion_path: Path,
) -> dict[str, Any]:
    label = f"{condition}_seed{seed}"
    try:
        verified = protocol.validate_training_run(
            condition, seed, require_dispatch_completion=False
        )
    except Exception as error:
        raise protocol.ProtocolError(
            f"open dispatch for {label} is partial/nonterminal; whole attempt is closed"
        ) from error
    intent = protocol.read_json(intent_path)
    expected_label = f"{condition}_seed{seed}"
    expected_config = protocol._config_path(condition, seed)
    if (
        intent.get("protocol") != protocol.PROTOCOL
        or intent.get("claim_id") != claim.get("claim_id")
        or intent.get("sequence") != sequence
        or intent.get("label") != expected_label
        or intent.get("condition") != condition
        or intent.get("seed") != seed
        or intent.get("config_sha256") != protocol.sha256_file(expected_config)
        or intent.get("runtime_git_commit")
        != protocol.load_and_verify_state(require_launch=True)["git_commit"]
        or intent.get("offline_wandb_dir")
        != str((protocol.OFFLINE_WANDB_DIR / expected_label).resolve())
    ):
        raise protocol.ProtocolError(f"open dispatch intent changed for {expected_label}")
    completion = {
        "schema_version": 1,
        "protocol": protocol.PROTOCOL,
        "claim_id": intent["claim_id"],
        "label": label,
        "intent_sha256": protocol.sha256_file(intent_path),
        "run_result_sha256": verified["run_result_sha256"],
        "offline_receipt_sha256": verified["offline_receipt_sha256"],
        "hardware": intent["hardware"],
        "returncode": 0,
        "adopted_complete_terminal_output_after_orchestrator_interruption": True,
        "completed_at_utc": protocol.utc_now(),
    }
    write_exclusive_json(completion_path, completion)
    return protocol.validate_training_run(condition, seed)


def run_one(claim: dict[str, Any], condition: str, seed: int, sequence: int) -> dict[str, Any]:
    label = f"{condition}_seed{seed}"
    protocol.load_and_verify_state(require_launch=True)
    verify_frozen_imports()
    if any(
        path.exists()
        for path in (
            protocol.COLLECTION_PATH,
            protocol.EVAL_DIR,
            protocol.COMPARISON_PATH,
            protocol.ANALYSIS_PROCESS_PATH,
            protocol.ACCEPTANCE_PATH,
        )
    ):
        raise protocol.ProtocolError("sealed-final artifacts exist during training")
    intent_path, completion_path, console_path = _dispatch_paths(sequence, label)
    if completion_path.is_file():
        return protocol.validate_training_run(condition, seed)
    if intent_path.is_file():
        return _adopt_or_fail_open_intent(
            claim=claim,
            condition=condition,
            seed=seed,
            sequence=sequence,
            intent_path=intent_path,
            completion_path=completion_path,
        )
    output = protocol._run_dir(condition, seed)
    wandb_dir = protocol.OFFLINE_WANDB_DIR / label
    if output.exists() or wandb_dir.exists() or console_path.exists():
        raise protocol.ProtocolError(f"unclaimed partial output exists for {label}")
    hardware = protocol.probe_hardware(require_idle=True)
    config_path = protocol._config_path(condition, seed)
    command = [
        str(protocol.PYTHON_EXECUTABLE),
        "scripts/v9_local_train.py",
        "--config",
        f".confirmatory/v9_local/configs/{label}.json",
        "--wandb-mode",
        "offline",
    ]
    intent = {
        "schema_version": 1,
        "protocol": protocol.PROTOCOL,
        "claim_id": claim["claim_id"],
        "sequence": sequence,
        "condition": condition,
        "seed": seed,
        "label": label,
        "command": command,
        "cwd": str(protocol.RUNTIME_WORKTREE.resolve()),
        "config_sha256": protocol.sha256_file(config_path),
        "registration_sha256": claim["registration_sha256"],
        "runtime_git_commit": protocol._runtime_git("rev-parse", "HEAD"),
        "hardware": hardware,
        "offline_wandb_dir": str(wandb_dir.resolve()),
        "created_at_utc": protocol.utc_now(),
    }
    write_exclusive_json(intent_path, intent)
    env = runtime_environment(label)
    for directory_key in ("WANDB_DIR", "WANDB_CACHE_DIR", "WANDB_DATA_DIR", "WANDB_CONFIG_DIR"):
        Path(env[directory_key]).parent.mkdir(parents=True, exist_ok=True)
    with console_path.open("x") as console:
        completed = subprocess.run(
            command,
            cwd=protocol.RUNTIME_WORKTREE,
            env=env,
            stdout=console,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        console.flush()
        os.fsync(console.fileno())
    if completed.returncode != 0:
        raise protocol.ProtocolError(
            f"{label} exited {completed.returncode}; whole attempt is closed"
        )
    verified = protocol.validate_training_run(
        condition, seed, require_dispatch_completion=False
    )
    completion = {
        "schema_version": 1,
        "protocol": protocol.PROTOCOL,
        "claim_id": claim["claim_id"],
        "label": label,
        "intent_sha256": protocol.sha256_file(intent_path),
        "run_result_sha256": verified["run_result_sha256"],
        "offline_receipt_sha256": verified["offline_receipt_sha256"],
        "hardware": hardware,
        "returncode": completed.returncode,
        "console_sha256": protocol.sha256_file(console_path),
        "completed_at_utc": protocol.utc_now(),
    }
    write_exclusive_json(completion_path, completion)
    return protocol.validate_training_run(condition, seed)


def run_training_attempt(*, treatments_only: bool = False) -> dict[str, Any]:
    with whole_attempt_gpu_lock():
        protocol.load_and_verify_state(require_launch=True)
        protocol.probe_hardware(require_idle=True)
        verify_frozen_imports()
        claim = _claim()
        try:
            for offset, seed in enumerate(protocol.SEEDS, 1):
                run_one(claim, "jlens", seed, offset)
            if not protocol.CURVE_PATH.exists():
                gate = protocol.compute_curve_gate(write_result=True)
            else:
                gate = protocol.verify_curve_gate(require_pass=False)
            if not gate["passed"]:
                result = {
                    "protocol": protocol.PROTOCOL,
                    "claim_id": claim["claim_id"],
                    "stage": "curve_failed_terminal",
                    "curve_gate_sha256": protocol.sha256_file(protocol.CURVE_PATH),
                    "controls_started": False,
                    "final_started": False,
                    "updated_at_utc": protocol.utc_now(),
                }
                _status(result)
                return result
            if treatments_only:
                result = {
                    "protocol": protocol.PROTOCOL,
                    "claim_id": claim["claim_id"],
                    "stage": "treatments_complete_curve_passed_controls_not_started",
                    "curve_gate_sha256": protocol.sha256_file(protocol.CURVE_PATH),
                    "updated_at_utc": protocol.utc_now(),
                }
                _status(result)
                return result
            _status(
                {
                    "protocol": protocol.PROTOCOL,
                    "claim_id": claim["claim_id"],
                    "stage": "running_controls",
                    "updated_at_utc": protocol.utc_now(),
                }
            )
            for offset, seed in enumerate(protocol.SEEDS, len(protocol.SEEDS) + 1):
                run_one(claim, "signflip", seed, offset)
            unlock = protocol.unlock_final() if not protocol.UNLOCK_PATH.exists() else protocol.verify_unlock()
            result = {
                "protocol": protocol.PROTOCOL,
                "claim_id": claim["claim_id"],
                "stage": "training_complete_final_unlocked_not_collected",
                "unlock_sha256": protocol.sha256_file(protocol.UNLOCK_PATH),
                "sealed_final_started": False,
                "updated_at_utc": protocol.utc_now(),
            }
            _status(result)
            return {**result, "unlock": unlock}
        except BaseException as error:
            _status(
                {
                    "protocol": protocol.PROTOCOL,
                    "claim_id": claim["claim_id"],
                    "stage": "infrastructure_failed_attempt_closed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "retry_policy": protocol.INFRASTRUCTURE_RETRY_POLICY,
                    "updated_at_utc": protocol.utc_now(),
                }
            )
            raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("probe", "run-treatments", "run-training"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.command == "probe":
            protocol.load_and_verify_state(require_launch=True)
            print(json.dumps(protocol.probe_hardware(require_idle=True), indent=2, sort_keys=True))
            print(json.dumps(verify_frozen_imports(), indent=2, sort_keys=True))
        else:
            result = run_training_attempt(treatments_only=args.command == "run-treatments")
            print(json.dumps(result, indent=2, sort_keys=True))
    except protocol.ProtocolError as error:
        print(f"protocol error: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
