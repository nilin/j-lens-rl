"""Serial Modal-L40S runner for the frozen V10 final collection.

The runner is intentionally unusable without a future registered state, a
separately approved automation audit, eight completed matched training runs,
and a valid final unlock.  It holds one container-local GPU lock, collects all 9 labels
without semantically inspecting any label outcome (opaque receipt hashing is
allowed), then verifies/analyzes the whole fixed collection exactly once.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from scripts import confirmatory_v10_final_protocol as protocol


@contextmanager
def one_gpu_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise protocol.FinalProtocolError("the registered Modal GPU is already leased") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def probe_hardware(context: protocol.FinalContext, *, require_idle: bool) -> dict[str, Any]:
    import torch

    expected = context.spec["hardware"]
    if (
        torch.__version__ != expected["torch_version"]
        or torch.version.cuda != expected["cuda_version"]
    ):
        raise protocol.FinalProtocolError("registered torch/CUDA runtime changed")
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if torch.cuda.device_count() != 1 or len(rows) != 1:
        raise protocol.FinalProtocolError("Modal worker must expose exactly one GPU")
    fields = [field.strip() for field in rows[0].split(",")]
    selected = (
        {
            "observed_gpu_uuid": fields[0],
            "device_name": fields[1],
            "driver_version": fields[2],
            "memory_total_mib": int(fields[3]),
        }
        if len(fields) == 4
        else None
    )
    if (
        selected is None
        or not selected["observed_gpu_uuid"].startswith("GPU-")
        or any(
        selected[key] != expected[key]
        for key in ("device_name", "driver_version", "memory_total_mib")
        )
    ):
        raise protocol.FinalProtocolError("registered Modal NVIDIA L40S identity changed")
    active: list[dict[str, Any]] = []
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    for line in processes.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[1] == selected["observed_gpu_uuid"]:
            active.append({"pid": int(fields[0]), "gpu_uuid": fields[1]})
    if require_idle and active:
        raise protocol.FinalProtocolError(f"registered Modal GPU is not idle: {active}")
    return {**expected, **selected, "active_compute_processes": active}


def _write_failure(context: protocol.FinalContext, payload: dict[str, Any]) -> None:
    protocol.record_final_failure(context, payload)


def _write_text_exclusive(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _assert_serial_prefix(
    context: protocol.FinalContext, label: str, sequence: int
) -> None:
    if (
        not 1 <= sequence <= len(protocol.FINAL_LABELS)
        or protocol.FINAL_LABELS[sequence - 1] != label
    ):
        raise protocol.FinalProtocolError("final label/sequence is out of registered order")
    prior = protocol.FINAL_LABELS[: sequence - 1]
    expected_by_directory = {
        context.eval_dir: {
            *(f"{item}.jsonl" for item in prior),
            *(f"{item}.environment.json" for item in prior),
        },
        context.evidence_dir / "final_dispatches": {
            *(f"{item}.intent.json" for item in prior),
            *(f"{item}.completion.json" for item in prior),
        },
        context.evidence_dir / "sealed_collection_logs": {
            *(f"{item}.stdout" for item in prior),
            *(f"{item}.stderr" for item in prior),
        },
    }
    for directory, expected in expected_by_directory.items():
        entries = list(directory.iterdir()) if directory.is_dir() else []
        observed = {path.name for path in entries if path.is_file()}
        if (
            directory.is_symlink()
            or observed != expected
            or any(path.is_symlink() or not path.is_file() for path in entries)
        ):
            raise protocol.FinalProtocolError(
                "partial/replayed final labels are terminal; a final collection may never resume"
            )
    for item in prior:
        output = context.eval_dir / f"{item}.jsonl"
        environment = context.eval_dir / f"{item}.environment.json"
        if any(path.is_symlink() or not path.is_file() for path in (output, environment)):
            raise protocol.FinalProtocolError("a prior final label is incomplete or unsafe")
        protocol._verify_dispatch(context, item)


def _run_one_label(
    context: protocol.FinalContext,
    collection_id: str,
    label: str,
    sequence: int,
) -> dict[str, Any]:
    protocol.verify_final_collection(context, collection_id)
    _assert_serial_prefix(context, label, sequence)
    output = context.eval_dir / f"{label}.jsonl"
    environment = context.eval_dir / f"{label}.environment.json"
    dispatch_dir = context.evidence_dir / "final_dispatches"
    intent_path = dispatch_dir / f"{label}.intent.json"
    completion_path = dispatch_dir / f"{label}.completion.json"
    log_dir = context.evidence_dir / "sealed_collection_logs"
    stdout_path = log_dir / f"{label}.stdout"
    stderr_path = log_dir / f"{label}.stderr"
    hardware = probe_hardware(context, require_idle=True)
    hardware.pop("active_compute_processes", None)
    intent = {
        "schema_version": 1,
        "protocol": context.spec["protocol"],
        "collection_id": collection_id,
        "sequence": sequence,
        "label": label,
        "hardware": hardware,
        "command": protocol.expected_eval_command(context, label),
        "cwd": str(context.repository.resolve()),
        "environment_overrides": protocol.expected_runtime_overrides(context),
        "status": "written_and_fsynced_before_gpu_process",
        "outcome_inspected_before_full_collection": False,
    }
    protocol.write_json_exclusive(intent_path, intent)
    env = dict(os.environ)
    overrides = protocol.expected_runtime_overrides(context)
    env.update(overrides)
    command = protocol.expected_eval_command(context, label)
    completed = subprocess.run(
        command,
        cwd=context.repository,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    _write_text_exclusive(stdout_path, completed.stdout)
    _write_text_exclusive(stderr_path, completed.stderr)
    if completed.returncode != 0 or not output.is_file() or not environment.is_file():
        raise protocol.FinalProtocolError(
            f"final evaluation {label} failed; the whole final allocation is spent"
        )
    completion = {
        "schema_version": 1,
        "protocol": context.spec["protocol"],
        "collection_id": collection_id,
        "sequence": sequence,
        "label": label,
        "intent_sha256": protocol.sha256_file(intent_path),
        "jsonl_sha256": protocol.sha256_file(output),
        "environment_sha256": protocol.sha256_file(environment),
        "stdout_sha256": protocol.sha256_file(stdout_path),
        "stderr_sha256": protocol.sha256_file(stderr_path),
        "returncode": completed.returncode,
        "outcome_inspected_before_full_collection": False,
        "command": command,
        "cwd": str(context.repository.resolve()),
        "environment_overrides": overrides,
    }
    protocol.write_json_exclusive(completion_path, completion)
    return completion


def _run_analysis(context: protocol.FinalContext) -> dict[str, Any]:
    protocol.verify_final_collection(context)
    if context.comparison_path.exists() or context.analysis_process_path.exists():
        raise protocol.FinalProtocolError("refusing to overwrite/repeat final paired analysis")
    input_hashes = {
        f"{label}.jsonl": protocol.sha256_file(context.eval_dir / f"{label}.jsonl")
        for label in protocol.FINAL_LABELS
    }
    command = protocol.expected_analysis_command(context)
    overrides = protocol.expected_runtime_overrides(context)
    env = dict(os.environ)
    env.update(overrides)
    probe_command = protocol.expected_analysis_probe_command(context)
    probe = subprocess.run(
        probe_command,
        cwd=context.repository,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if probe.returncode != 0:
        raise protocol.FinalProtocolError("registered final analysis source probe failed")
    try:
        probe_value = json.loads(probe.stdout)
    except json.JSONDecodeError as error:
        raise protocol.FinalProtocolError(
            "registered final analysis source probe returned invalid JSON"
        ) from error
    protocol.verify_analysis_probe_payload(context, probe_value)
    process = {
        "schema_version": 1,
        "python_executable": context.spec["python_executable"],
        "command": command,
        "cwd": str(context.repository.resolve()),
        "environment_overrides": overrides,
        "input_sha256": input_hashes,
        "source_probe_command": probe_command,
        "source_probe_returncode": probe.returncode,
        "loaded_source_identity": probe_value.get("loaded_source_identity"),
        "environment_snapshot": probe_value.get("environment_snapshot"),
    }
    protocol.write_json_exclusive(context.analysis_process_path, process)
    completed = subprocess.run(
        command,
        cwd=context.repository,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0 or not context.comparison_path.is_file():
        raise protocol.FinalProtocolError("registered final paired analysis failed")
    return {
        "comparison_sha256": protocol.sha256_file(context.comparison_path),
        "process_sha256": protocol.sha256_file(context.analysis_process_path),
        "returncode": completed.returncode,
    }


def run_final_collection(state_dir: str | Path) -> dict[str, Any]:
    context = protocol.load_context(state_dir)
    protocol.verify_preunlock_readiness(context)
    lock_path = Path(context.spec["gpu_lock_path"])
    with one_gpu_lock(lock_path):
        probe_hardware(context, require_idle=True)
        collection_id = uuid.uuid4().hex
        protocol.begin_final_collection(context, collection_id)
        try:
            completions = [
                _run_one_label(context, collection_id, label, sequence)
                for sequence, label in enumerate(protocol.FINAL_LABELS, 1)
            ]
            # No JSONL is parsed, scored, compared, or exposed above this line.
            # Opaque SHA-256 receipt hashing is allowed.  Only once all 9 fixed
            # labels exist do we verify the complete collection semantically.
            protocol.verify_all_evaluations(context)
            analysis = _run_analysis(context)
            report = protocol.final_report(context)
            return {
                "stage": "complete" if report["passed"] else "significance_failed",
                "collection_id": collection_id,
                "dispatches": completions,
                "analysis": analysis,
                "acceptance": report,
            }
        except BaseException as error:
            _write_failure(
                context,
                {
                    "collection_id": collection_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "failure_phase": "serial_collection_or_analysis",
                },
            )
            raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("design", "verify-preunlock", "run"))
    parser.add_argument("--state-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "design":
        result = protocol.design_summary()
    else:
        if args.state_dir is None:
            raise SystemExit("--state-dir is required")
        context = protocol.load_context(args.state_dir)
        if args.command == "verify-preunlock":
            result = protocol.verify_preunlock_readiness(context)
        else:
            result = run_final_collection(args.state_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
