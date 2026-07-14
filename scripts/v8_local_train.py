"""Run one V8-local config and seal its completed offline W&B directory.

The shared trainer normally publishes a terminal W&B artifact online.  V8
changes only that transport.  This wrapper queues the same seven terminal
files into an identity-pinned offline run, closes W&B, and only then writes a
local receipt outside the W&B directory.  The receipt never claims that a
remote artifact exists.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jlens_rl import train


OFFLINE_RECEIPT_NAME = "wandb_offline_terminal_receipt.json"
EVIDENCE_FILE_NAMES = (
    "run_result_manifest.json",
    "validation_history.jsonl",
    "log_history.json",
    "environment_snapshot.json",
    "run_manifest.json",
    "resolved_config.json",
    "data_indices.json",
)
_SAFE_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")
SYNC_POLICY = (
    "Preserve this completed offline directory with its seven embedded terminal "
    "files; sync this directory only. Never rerun or resume optimization to "
    "repair tracking infrastructure."
)


@dataclass(frozen=True)
class PendingOfflinePublication:
    output_dir: Path
    wandb_dir: Path
    offline_run_root: Path
    wandb_identity: dict[str, Any]
    observed_offline_identity: dict[str, Any]
    terminal_run_result_sha256: str
    queued_evidence_sha256: dict[str, str]


_pending_publication: PendingOfflinePublication | None = None


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _absolute_directory(path: str | Path, *, label: str) -> Path:
    raw = Path(path)
    if not raw.is_absolute():
        raise RuntimeError(f"{label} must be an absolute path")
    if raw.is_symlink() or not raw.is_dir():
        raise RuntimeError(f"{label} must be an existing non-symlink directory")
    return raw.resolve(strict=True)


def _is_relative_to(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _expected_offline_identity(identity: Any) -> dict[str, Any]:
    expected = train._expected_observed_wandb_identity(identity)
    assert isinstance(identity, dict)  # established by the shared validator
    if identity.get("resume") != "never":
        raise RuntimeError("frozen offline W&B identity must forbid resume")
    canonical_url = (
        f"https://wandb.ai/{expected['entity']}/{expected['project']}/runs/"
        f"{expected['run_id']}"
    )
    if expected["url"] != canonical_url:
        raise RuntimeError("frozen offline W&B target URL is not canonical")
    if _SAFE_RUN_ID.fullmatch(expected["run_id"]) is None:
        raise RuntimeError("frozen offline W&B run ID is path-unsafe")
    return {**expected, "resume": "never"}


def _observe_offline_identity(run: Any, identity: Any) -> dict[str, Any]:
    expected = _expected_offline_identity(identity)
    settings = getattr(run, "settings", None)
    raw_tags = getattr(run, "tags", None)
    observed = {
        "run_id": getattr(run, "id", None),
        "entity": getattr(run, "entity", None),
        "project": getattr(run, "project", None),
        "run_name": getattr(run, "name", None),
        # Offline W&B deliberately has no observable remote URL.  The exact
        # future target URL remains frozen in ``wandb_identity`` above.
        "url": getattr(run, "url", None),
        "group": getattr(run, "group", None),
        "tags": list(raw_tags) if isinstance(raw_tags, (list, tuple)) else raw_tags,
        "resume": os.environ.get("WANDB_RESUME"),
        "mode": getattr(settings, "mode", None),
        "run_files_dir": getattr(run, "dir", None),
    }
    comparable = {
        key: observed[key]
        for key in ("run_id", "entity", "project", "run_name", "group", "tags")
    }
    expected_comparable = {
        key: expected[key]
        for key in ("run_id", "entity", "project", "run_name", "group", "tags")
    }
    if comparable != expected_comparable:
        raise RuntimeError(
            "offline W&B run differs from its frozen identity: "
            f"{comparable!r} != {expected_comparable!r}"
        )
    if observed["url"] is not None:
        raise RuntimeError("offline W&B unexpectedly exposed an online run URL")
    if observed["resume"] != "never" or observed["mode"] != "offline":
        raise RuntimeError("V8-local W&B mode/resume policy changed")

    expected_environment = {
        "WANDB_MODE": "offline",
        "WANDB_RUN_ID": expected["run_id"],
        "WANDB_ENTITY": expected["entity"],
        "WANDB_PROJECT": expected["project"],
        "WANDB_RUN_GROUP": expected["group"],
        "WANDB_TAGS": ",".join(expected["tags"]),
        "WANDB_RESUME": "never",
    }
    actual_environment = {key: os.environ.get(key) for key in expected_environment}
    if actual_environment != expected_environment:
        raise RuntimeError(
            "offline W&B environment differs from registration: "
            f"{actual_environment!r} != {expected_environment!r}"
        )
    return observed


def _offline_run_name_matches(name: str, run_id: str) -> bool:
    prefix = "offline-run-"
    suffix = f"-{run_id}"
    return name.startswith(prefix) and name.endswith(suffix) and len(name) > len(
        prefix
    ) + len(suffix)


def _captured_offline_root(
    observed: dict[str, Any], wandb_dir: Path, run_id: str
) -> Path:
    raw_files_dir = observed.get("run_files_dir")
    if not isinstance(raw_files_dir, str) or not raw_files_dir:
        raise RuntimeError("offline W&B run did not expose its files directory")
    files_dir_path = Path(raw_files_dir)
    if not files_dir_path.is_absolute():
        raise RuntimeError("offline W&B files directory is not absolute")
    if files_dir_path.is_symlink() or not files_dir_path.is_dir():
        raise RuntimeError("offline W&B files directory is missing or symlinked")
    files_dir = files_dir_path.resolve(strict=True)
    root = files_dir.parent
    if (
        files_dir.name != "files"
        or root.is_symlink()
        or not root.is_dir()
        or not _offline_run_name_matches(root.name, run_id)
        or not _is_relative_to(root, wandb_dir)
    ):
        raise RuntimeError("offline W&B files directory escaped its registered root")
    return root


def _evidence_hashes(output_dir: Path) -> dict[str, str]:
    if tuple(train.TERMINAL_EVIDENCE_FILE_NAMES) != EVIDENCE_FILE_NAMES:
        raise RuntimeError("shared terminal evidence inventory changed")
    hashes: dict[str, str] = {}
    for name in EVIDENCE_FILE_NAMES:
        path = output_dir / name
        if path.is_symlink() or not path.is_file() or path.parent.resolve() != output_dir:
            raise RuntimeError(f"terminal evidence path escaped the run directory: {name}")
        hashes[name] = train.sha256_file(path)
    return hashes


def publish_run_result_offline(
    *, output_dir: Path, result: dict[str, Any], enabled: bool
) -> dict[str, Any] | None:
    """Queue terminal evidence; receipt publication waits for ``wandb.finish``."""
    global _pending_publication
    if not enabled:
        return None
    if _pending_publication is not None:
        raise RuntimeError("an offline terminal publication is already pending")

    import wandb

    run = wandb.run
    if run is None:
        raise RuntimeError("offline terminal evidence requires an active W&B run")
    resolved_output = _absolute_directory(
        Path(output_dir).absolute(), label="output_dir"
    )
    receipt_path = resolved_output / OFFLINE_RECEIPT_NAME
    if receipt_path.exists() or receipt_path.with_suffix(receipt_path.suffix + ".tmp").exists():
        raise RuntimeError("offline terminal receipt already exists or is incomplete")
    raw_wandb_dir = os.environ.get("WANDB_DIR")
    if raw_wandb_dir is None:
        raise RuntimeError("V8-local requires a per-run WANDB_DIR")
    wandb_dir = _absolute_directory(raw_wandb_dir, label="WANDB_DIR")

    expected = _expected_offline_identity(result.get("wandb_identity"))
    observed = _observe_offline_identity(run, result.get("wandb_identity"))
    offline_root = _captured_offline_root(observed, wandb_dir, expected["run_id"])
    if _is_relative_to(resolved_output, offline_root):
        raise RuntimeError("offline receipt would be written inside the W&B run root")
    hashes = _evidence_hashes(resolved_output)
    terminal_hash = hashes["run_result_manifest.json"]

    queued_record = {
        "schema_version": 1,
        "transport": "offline_wandb_pending_explicit_sync",
        "remote_artifact_claimed": False,
        "wandb_identity": result["wandb_identity"],
        "terminal_run_result_sha256": terminal_hash,
        "queued_evidence_sha256": hashes,
    }
    wandb.config.update({"terminal_run_result": result}, allow_val_change=True)
    wandb.config.update(
        {"offline_terminal_evidence_queue": queued_record}, allow_val_change=True
    )
    for name in EVIDENCE_FILE_NAMES:
        wandb.save(
            str(resolved_output / name),
            base_path=str(resolved_output),
            policy="now",
        )

    _pending_publication = PendingOfflinePublication(
        output_dir=resolved_output,
        wandb_dir=wandb_dir,
        offline_run_root=offline_root,
        wandb_identity=dict(result["wandb_identity"]),
        observed_offline_identity=observed,
        terminal_run_result_sha256=terminal_hash,
        queued_evidence_sha256=hashes,
    )
    return queued_record


def _find_offline_run_root(wandb_dir: Path, run_id: str) -> Path:
    candidates: list[Path] = []
    for current, dirnames, _filenames in os.walk(wandb_dir, followlinks=False):
        current_path = Path(current)
        retained: list[str] = []
        for name in dirnames:
            path = current_path / name
            if path.is_symlink():
                continue
            if _offline_run_name_matches(name, run_id):
                candidates.append(path.resolve(strict=True))
                continue
            retained.append(name)
        dirnames[:] = retained
    unique = sorted(set(candidates))
    if len(unique) != 1:
        raise RuntimeError(
            f"expected exactly one completed offline W&B root for {run_id!r}; "
            f"found {len(unique)}"
        )
    root = unique[0]
    if root.is_symlink() or not _is_relative_to(root, wandb_dir):
        raise RuntimeError("located offline W&B root escaped WANDB_DIR")
    return root


def _hash_regular_file(path: Path) -> tuple[str, int, str]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"offline inventory entry is not a regular file: {path}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise RuntimeError(f"offline W&B file changed during inventory: {path}")
        return digest.hexdigest(), after.st_size, oct(stat.S_IMODE(after.st_mode))
    finally:
        os.close(descriptor)


def _offline_inventory(root: Path) -> list[dict[str, Any]]:
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("offline W&B root is missing or symlinked")
    records: list[dict[str, Any]] = []
    for current, dirnames, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        retained: list[str] = []
        for name in dirnames:
            path = current_path / name
            if path.is_symlink():
                relative = path.relative_to(root).as_posix()
                records.append(
                    {
                        "path": relative,
                        "type": "symlink",
                        "target": os.readlink(path),
                        "mode": oct(stat.S_IMODE(path.lstat().st_mode)),
                    }
                )
            elif path.is_dir():
                retained.append(name)
            else:
                raise RuntimeError(f"special offline W&B directory entry: {path}")
        dirnames[:] = retained
        for name in filenames:
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                records.append(
                    {
                        "path": relative,
                        "type": "symlink",
                        "target": os.readlink(path),
                        "mode": oct(stat.S_IMODE(path.lstat().st_mode)),
                    }
                )
            elif path.is_file():
                digest, size, mode = _hash_regular_file(path)
                records.append(
                    {
                        "path": relative,
                        "type": "file",
                        "sha256": digest,
                        "size_bytes": size,
                        "mode": mode,
                    }
                )
            else:
                raise RuntimeError(f"special offline W&B file entry: {path}")
    return sorted(records, key=lambda item: item["path"])


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise RuntimeError("short write while embedding offline W&B evidence")
        view = view[written:]


def _atomic_embed_evidence_file(
    *, source: Path, queued: Path, expected_sha256: str
) -> None:
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"terminal evidence source is not regular: {source.name}")
    if not queued.is_symlink():
        raise RuntimeError(
            f"offline W&B queued evidence is not the expected symlink: {source.name}"
        )
    raw_target = os.readlink(queued)
    if not Path(raw_target).is_absolute() or queued.resolve(strict=True) != source:
        raise RuntimeError(
            f"queued W&B evidence symlink escaped its source: {source.name}"
        )
    if train.sha256_file(source) != expected_sha256:
        raise RuntimeError(f"terminal evidence changed before embedding: {source.name}")

    temporary = queued.with_name(f".{queued.name}.v8-embed.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise RuntimeError(f"stale offline evidence embedding temp: {source.name}")
    source_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    destination_flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
    )
    source_descriptor = os.open(source, source_flags)
    destination_descriptor: int | None = None
    try:
        before = os.fstat(source_descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"terminal evidence source is not regular: {source.name}")
        destination_descriptor = os.open(temporary, destination_flags, 0o600)
        digest = hashlib.sha256()
        while True:
            chunk = os.read(source_descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            _write_all(destination_descriptor, chunk)
        after = os.fstat(source_descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) or digest.hexdigest() != expected_sha256:
            raise RuntimeError(f"terminal evidence changed while embedding: {source.name}")
        os.fchmod(destination_descriptor, stat.S_IMODE(before.st_mode))
        os.fsync(destination_descriptor)
        os.close(destination_descriptor)
        destination_descriptor = None
        os.replace(temporary, queued)
        directory_descriptor = os.open(
            queued.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if destination_descriptor is not None:
            os.close(destination_descriptor)
        os.close(source_descriptor)
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
    if queued.is_symlink() or not queued.is_file():
        raise RuntimeError(f"embedded W&B evidence is not regular: {source.name}")
    if train.sha256_file(queued) != expected_sha256:
        raise RuntimeError(f"embedded W&B evidence bytes changed: {source.name}")


def _embed_queued_evidence(pending: PendingOfflinePublication, root: Path) -> None:
    files_dir = root / "files"
    if files_dir.is_symlink() or not files_dir.is_dir():
        raise RuntimeError("offline W&B files directory is missing or symlinked")
    expected_names = set(pending.queued_evidence_sha256)
    observed_symlinks = {
        path.name for path in files_dir.iterdir() if path.is_symlink()
    }
    if observed_symlinks != expected_names:
        raise RuntimeError(
            "offline W&B queued symlink set changed: "
            f"{sorted(observed_symlinks)!r} != {sorted(expected_names)!r}"
        )
    for name, expected_hash in pending.queued_evidence_sha256.items():
        _atomic_embed_evidence_file(
            source=pending.output_dir / name,
            queued=files_dir / name,
            expected_sha256=expected_hash,
        )


def _validate_embedded_evidence(
    pending: PendingOfflinePublication, root: Path
) -> None:
    for name, expected_hash in pending.queued_evidence_sha256.items():
        queued = root / "files" / name
        if queued.is_symlink() or not queued.is_file():
            raise RuntimeError(f"offline W&B did not embed regular evidence: {name}")
        if train.sha256_file(queued) != expected_hash:
            raise RuntimeError(f"embedded W&B evidence bytes changed: {name}")


def finalize_offline_terminal_receipt(
    pending: PendingOfflinePublication,
) -> dict[str, Any]:
    """Seal the exact offline tree after W&B has fully stopped writing it."""
    import wandb

    if wandb.run is not None:
        raise RuntimeError("cannot inventory offline W&B before wandb.finish()")
    root = _find_offline_run_root(pending.wandb_dir, pending.wandb_identity["run_id"])
    if root != pending.offline_run_root:
        raise RuntimeError("post-finish offline W&B root differs from the active run")
    receipt_path = pending.output_dir / OFFLINE_RECEIPT_NAME
    if _is_relative_to(receipt_path.resolve(strict=False), root):
        raise RuntimeError("offline terminal receipt would contaminate its own inventory")
    if receipt_path.exists() or receipt_path.with_suffix(receipt_path.suffix + ".tmp").exists():
        raise RuntimeError("offline terminal receipt already exists or is incomplete")

    _embed_queued_evidence(pending, root)
    _validate_embedded_evidence(pending, root)
    archive = root / f"run-{pending.wandb_identity['run_id']}.wandb"
    if archive.is_symlink() or not archive.is_file():
        raise RuntimeError("completed offline W&B archive is missing")
    inventory = _offline_inventory(root)
    receipt = {
        "schema_version": 1,
        "transport": "offline_wandb_pending_explicit_sync",
        "sync_completed": False,
        "remote_artifact_claimed": False,
        "wandb_identity": pending.wandb_identity,
        "observed_offline_identity": pending.observed_offline_identity,
        "wandb_dir": str(pending.wandb_dir),
        "offline_run_root": str(root),
        "offline_run_root_relative_to_wandb_dir": root.relative_to(
            pending.wandb_dir
        ).as_posix(),
        "terminal_run_result_sha256": pending.terminal_run_result_sha256,
        "queued_evidence_sha256": pending.queued_evidence_sha256,
        "embedded_evidence_relative_paths": {
            name: f"files/{name}" for name in EVIDENCE_FILE_NAMES
        },
        "embedded_evidence_storage": "regular_files_inside_offline_run_root",
        "offline_run_file_symlink_inventory": inventory,
        "offline_run_file_symlink_count": len(inventory),
        "offline_run_tree_sha256": _canonical_sha256(inventory),
        "sync_policy": SYNC_POLICY,
    }
    train._write_json_atomic(receipt_path, receipt)
    return receipt


def validate_offline_terminal_receipt(
    output_dir: str | Path,
    expected_identity: dict[str, Any],
    expected_wandb_dir: str | Path,
) -> dict[str, Any]:
    """Validate a receipt, its seven sources, and its still-exact sync tree."""
    output = _absolute_directory(
        Path(output_dir).absolute(), label="receipt output_dir"
    )
    wandb_dir = _absolute_directory(expected_wandb_dir, label="expected WANDB_DIR")
    expected = _expected_offline_identity(expected_identity)
    receipt_path = output / OFFLINE_RECEIPT_NAME
    temporary_path = receipt_path.with_suffix(receipt_path.suffix + ".tmp")
    if receipt_path.is_symlink() or not receipt_path.is_file() or temporary_path.exists():
        raise RuntimeError("offline terminal receipt is absent, symlinked, or incomplete")
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("offline terminal receipt is not valid JSON") from error
    exact_keys = {
        "schema_version",
        "transport",
        "sync_completed",
        "remote_artifact_claimed",
        "wandb_identity",
        "observed_offline_identity",
        "wandb_dir",
        "offline_run_root",
        "offline_run_root_relative_to_wandb_dir",
        "terminal_run_result_sha256",
        "queued_evidence_sha256",
        "embedded_evidence_relative_paths",
        "embedded_evidence_storage",
        "offline_run_file_symlink_inventory",
        "offline_run_file_symlink_count",
        "offline_run_tree_sha256",
        "sync_policy",
    }
    if not isinstance(receipt, dict) or set(receipt) != exact_keys:
        raise RuntimeError("offline terminal receipt schema changed")
    if (
        receipt["schema_version"] != 1
        or receipt["transport"] != "offline_wandb_pending_explicit_sync"
        or receipt["sync_completed"] is not False
        or receipt["remote_artifact_claimed"] is not False
        or receipt["wandb_identity"] != expected_identity
        or receipt["wandb_dir"] != str(wandb_dir)
        or receipt["embedded_evidence_relative_paths"]
        != {name: f"files/{name}" for name in EVIDENCE_FILE_NAMES}
        or receipt["embedded_evidence_storage"]
        != "regular_files_inside_offline_run_root"
        or receipt["sync_policy"] != SYNC_POLICY
    ):
        raise RuntimeError("offline terminal receipt identity/transport changed")

    root = _find_offline_run_root(wandb_dir, expected["run_id"])
    recorded_root = receipt["offline_run_root"]
    recorded_relative = receipt["offline_run_root_relative_to_wandb_dir"]
    if (
        not isinstance(recorded_root, str)
        or not Path(recorded_root).is_absolute()
        or Path(recorded_root).is_symlink()
        or Path(recorded_root).resolve(strict=True) != root
        or recorded_relative != root.relative_to(wandb_dir).as_posix()
        or _is_relative_to(receipt_path.resolve(strict=True), root)
    ):
        raise RuntimeError("offline terminal receipt root escaped its registration")

    observed = receipt["observed_offline_identity"]
    if not isinstance(observed, dict):
        raise RuntimeError("offline terminal receipt lacks observed identity")
    expected_observable = {
        key: expected[key]
        for key in ("run_id", "entity", "project", "run_name", "group", "tags")
    }
    actual_observable = {key: observed.get(key) for key in expected_observable}
    raw_files_dir = observed.get("run_files_dir")
    if (
        actual_observable != expected_observable
        or observed.get("url") is not None
        or observed.get("resume") != "never"
        or observed.get("mode") != "offline"
        or not isinstance(raw_files_dir, str)
        or not Path(raw_files_dir).is_absolute()
        or Path(raw_files_dir).resolve(strict=True) != root / "files"
    ):
        raise RuntimeError("observed offline W&B identity is not exact")

    hashes = _evidence_hashes(output)
    result_path = output / "run_result_manifest.json"
    try:
        result = json.loads(result_path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("terminal result manifest is not valid JSON") from error
    if not isinstance(result, dict) or result.get("wandb_identity") != expected_identity:
        raise RuntimeError("terminal result has a different frozen W&B identity")
    if (
        receipt["queued_evidence_sha256"] != hashes
        or receipt["terminal_run_result_sha256"]
        != hashes["run_result_manifest.json"]
    ):
        raise RuntimeError("terminal evidence hashes differ from the offline receipt")

    pending = PendingOfflinePublication(
        output_dir=output,
        wandb_dir=wandb_dir,
        offline_run_root=root,
        wandb_identity=dict(expected_identity),
        observed_offline_identity=dict(observed),
        terminal_run_result_sha256=hashes["run_result_manifest.json"],
        queued_evidence_sha256=hashes,
    )
    _validate_embedded_evidence(pending, root)
    archive = root / f"run-{expected['run_id']}.wandb"
    if archive.is_symlink() or not archive.is_file():
        raise RuntimeError("completed offline W&B archive is missing")
    inventory = _offline_inventory(root)
    if (
        receipt["offline_run_file_symlink_inventory"] != inventory
        or receipt["offline_run_file_symlink_count"] != len(inventory)
        or receipt["offline_run_tree_sha256"] != _canonical_sha256(inventory)
    ):
        raise RuntimeError("offline W&B tree changed after its terminal receipt")
    return receipt


def _finish_wandb(*, exit_code: int, require_active: bool) -> None:
    import wandb

    if wandb.run is None:
        if require_active:
            raise RuntimeError("offline W&B run vanished before explicit finish")
        return
    wandb.finish(exit_code=exit_code)
    if wandb.run is not None:
        raise RuntimeError("wandb.finish() did not close the offline run")


def main() -> None:
    global _pending_publication
    if _pending_publication is not None:
        raise RuntimeError("stale offline terminal publication state")
    original_publisher = train.publish_run_result_to_wandb
    train.publish_run_result_to_wandb = publish_run_result_offline
    try:
        try:
            train.main()
        except BaseException as error:
            try:
                _finish_wandb(exit_code=1, require_active=False)
            except BaseException as finish_error:
                error.add_note(f"offline W&B cleanup also failed: {finish_error!r}")
            raise
        else:
            if _pending_publication is None:
                _finish_wandb(exit_code=1, require_active=False)
                raise RuntimeError(
                    "registered V8-local training produced no terminal queue"
                )
            _finish_wandb(exit_code=0, require_active=True)
            finalize_offline_terminal_receipt(_pending_publication)
            _pending_publication = None
    finally:
        train.publish_run_result_to_wandb = original_publisher


if __name__ == "__main__":
    main()
