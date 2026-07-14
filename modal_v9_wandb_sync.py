"""CPU-only, one-run sync for a sealed V9-local offline W&B run.

This transport is deliberately outside the registered optimization path.  It
reads one completed run, validates its exact registered config and canonical
V9 terminal receipt twice, packages only the receipt-bound offline W&B tree,
and asks a CPU-only Modal function to publish it. Existing remote run IDs are
never overwritten; ``modal run ...::verify_main`` is the separate read-only path.

Example, only after the run has a terminal receipt::

    .venv/bin/modal run modal_v9_wandb_sync.py::main --label jlens_seed208
"""

from __future__ import annotations

import hashlib
import json
import os
import posixpath
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable

import modal


APP_NAME = "j-lens-rl-v9-local-wandb-sync"
STAGING_VOLUME_NAME = "j-lens-rl-v9-local-wandb-sync-staging-v1"
STAGING_MOUNT = Path("/staging")
REGISTERED_REPO = Path("/j-lens-rl")
REGISTERED_STATE = REGISTERED_REPO / ".confirmatory" / "v9_local"
LOCAL_REPO = Path(__file__).resolve().parent
LOCAL_STATE = LOCAL_REPO / ".confirmatory" / "v9_local"

PROTOCOL = "j-lens-rl-confirmatory-v9-local-profanity-u5"
REGISTERED_GIT_COMMIT = "4d2f884f246b85b9215a8332632e7092a76deb80"
REGISTRATION_SHA256 = "5c9b776c452eea645361b72f87d8cabcd1059db654430cb7975f7325499b0ce4"
RECIPE_LOCK_SHA256 = "b58aab172b63ba7e6f5d512d6f650ccc4c203dce54295c80512a6d98e85dad14"
WANDB_ENTITY = "nilinabra-spare-time"
WANDB_PROJECT = "j-lens-rl"
WANDB_GROUP = "confirm-v9-local-emotional-profanity-u5-h20"
WANDB_PREFIX = WANDB_GROUP
WANDB_TAGS = [
    "confirmatory-v9-local",
    "emotional-j-lens",
    "profanity-u5",
    "rtx4090",
    "offline-wandb",
    "curve-exposed-final-sealed",
]
SEEDS = tuple(range(208, 216))
CONDITIONS = ("jlens", "signflip")
CONFIG_SHA256 = {
    "jlens_seed208": "3a89310664a16857c9f083e463b2930f3197508536e7ffea1ffa368b23c1e647",
    "jlens_seed209": "7711e0ae2ce43d7ba6a37032a68086aba4c77b0629020772e871cc688421d532",
    "jlens_seed210": "05e7dafe3facbb0f53ffee8b5222619fc383cffb5e377e1e9f7761f312aba27d",
    "jlens_seed211": "723faa3ec6ad6e32050cd7f38d80000eea2ce212f5a9cf9b41593bad6e8d6897",
    "jlens_seed212": "3e8b39222b8a45dadcd3192c2a42e8d88cd0d56172e85ef57feb71b689231493",
    "jlens_seed213": "6628291663bff7c99069c11732ade5f7864fdce0e7d37999c2803aa07be1cf4b",
    "jlens_seed214": "654e34a9edc97c4651a64314f174a4d99f7ec06358cd9ed489ddb886398039f1",
    "jlens_seed215": "74e5661d3f2b505f1b7fc7e9ea21399605ed4b55da9587c559392ed51b036b00",
    "signflip_seed208": "5d195a44b0ea91f73e18b0e6690e5db3f7cf2bb75cbcbf14a622cd0b848e4260",
    "signflip_seed209": "22c813470a3f32b361ab90569c5f29c900d0996415c38ae8be20a204e12ac2e2",
    "signflip_seed210": "abb67626f7003c201938a3eae159551a79524cdb66ec4720269db06fc4fb9ce0",
    "signflip_seed211": "d16dfd2140df6a0790db8cc46ec4bf63966c0e5769e8a74926d65aca52d311b7",
    "signflip_seed212": "7acf61c8edaf0de19a9f066a925699122f926844a15bca0e3a44210a427ace15",
    "signflip_seed213": "fc1fb698827543590a17166861ba5b41d25163ee8df3ad6e19f95c00c1991031",
    "signflip_seed214": "dec23e46e9294393873e7913e7088c98db12e00fdbda41ca4337ab4b26c86f2e",
    "signflip_seed215": "2a874f2a11cf7d627b902e83f952c61c7bf4b2d6ddcf6c18b539ad5eb0b7a9a8",
}
EVIDENCE_FILE_NAMES = (
    "run_result_manifest.json",
    "validation_history.jsonl",
    "log_history.json",
    "environment_snapshot.json",
    "run_manifest.json",
    "resolved_config.json",
    "data_indices.json",
)
OFFLINE_RECEIPT_NAME = "wandb_offline_terminal_receipt.json"
RECEIPT_KEYS = {
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
SYNC_POLICY = (
    "Preserve this completed offline directory with its seven embedded terminal "
    "files; sync this directory only. Never rerun or resume optimization to "
    "repair tracking infrastructure."
)
MAX_PAYLOAD_BYTES = 2_000_000_000
_SHA256 = re.compile(r"[0-9a-f]{64}")
_LABEL = re.compile(r"(jlens|signflip)_seed(\d+)")
_WANDB_CORE_DEBUG_TARGET = re.compile(
    r"/j-lens-rl/\.confirmatory/v9_local/wandb_cache/"
    r"(?P<label>(?:jlens|signflip)_seed\d+)/wandb/logs/"
    r"core-debug-\d{8}_\d{6}\.log"
)


app = modal.App(APP_NAME)
staging_volume = modal.Volume.from_name(
    STAGING_VOLUME_NAME, create_if_missing=True, version=2
)
wandb_secret = modal.Secret.from_name(
    "j-lens-rl-wandb", required_keys=["WANDB_API_KEY"]
)
sync_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "wandb==0.28.0"
)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"not a regular file: {path}")
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
            raise RuntimeError(f"file changed while hashing: {path}")
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _read_json_bytes(payload: bytes, *, label: str) -> Any:
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is not valid JSON") from error


def _registered_identity(label: str) -> dict[str, Any]:
    match = _LABEL.fullmatch(label)
    if match is None or int(match.group(2)) not in SEEDS:
        raise RuntimeError(f"not a registered V9-local training label: {label!r}")
    run_id = f"{WANDB_PREFIX}-{label}"
    return {
        "entity": WANDB_ENTITY,
        "project": WANDB_PROJECT,
        "run_name": run_id,
        "run_id": run_id,
        "url": f"https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}/runs/{run_id}",
        "group": WANDB_GROUP,
        "tags": WANDB_TAGS,
        "resume": "never",
    }


def _safe_relative(value: Any, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise RuntimeError(f"unsafe {label}")
    path = PurePosixPath(value)
    if (
        path == PurePosixPath(".")
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != value
    ):
        raise RuntimeError(f"unsafe {label}: {value!r}")
    return path


def _safe_symlink_target(
    entry: PurePosixPath, target: Any, *, expected_label: str | None = None
) -> str:
    if not isinstance(target, str) or not target or "\x00" in target:
        raise RuntimeError(f"unsafe symlink target for {entry}")
    raw = PurePosixPath(target)
    if raw.is_absolute():
        match = _WANDB_CORE_DEBUG_TARGET.fullmatch(target)
        if (
            entry != PurePosixPath("logs/debug-core.log")
            or match is None
            or (expected_label is not None and match.group("label") != expected_label)
        ):
            raise RuntimeError(f"absolute symlink target in offline W&B tree: {entry}")
        return target
    normalized = posixpath.normpath((entry.parent / raw).as_posix())
    if normalized == ".." or normalized.startswith("../"):
        raise RuntimeError(f"escaping symlink target in offline W&B tree: {entry}")
    return target


def _validated_inventory(
    receipt: dict[str, Any], *, expected_label: str | None = None
) -> list[dict[str, Any]]:
    inventory = receipt.get("offline_run_file_symlink_inventory")
    count = receipt.get("offline_run_file_symlink_count")
    if (
        not isinstance(inventory, list)
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count != len(inventory)
        or receipt.get("offline_run_tree_sha256") != _canonical_sha256(inventory)
    ):
        raise RuntimeError("offline receipt inventory binding changed")
    paths: list[str] = []
    leaves: set[PurePosixPath] = set()
    for item in inventory:
        if not isinstance(item, dict):
            raise RuntimeError("malformed offline receipt inventory entry")
        relative = _safe_relative(item.get("path"), label="inventory path")
        kind = item.get("type")
        mode = item.get("mode")
        try:
            parsed_mode = int(mode, 8) if isinstance(mode, str) else -1
        except ValueError as error:
            raise RuntimeError("malformed offline receipt inventory mode") from error
        if not 0 <= parsed_mode <= 0o7777 or mode != oct(parsed_mode):
            raise RuntimeError("malformed offline receipt inventory mode")
        if kind == "file":
            if set(item) != {"path", "type", "sha256", "size_bytes", "mode"}:
                raise RuntimeError("malformed regular-file inventory entry")
            size = item.get("size_bytes")
            if (
                _SHA256.fullmatch(str(item.get("sha256"))) is None
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size < 0
            ):
                raise RuntimeError("malformed regular-file inventory entry")
        elif kind == "symlink":
            if set(item) != {"path", "type", "target", "mode"}:
                raise RuntimeError("malformed symlink inventory entry")
            _safe_symlink_target(
                relative, item.get("target"), expected_label=expected_label
            )
        else:
            raise RuntimeError("special entry in offline receipt inventory")
        if relative in leaves:
            raise RuntimeError("duplicate offline receipt inventory path")
        paths.append(relative.as_posix())
        leaves.add(relative)
    if paths != sorted(paths):
        raise RuntimeError("offline receipt inventory is not canonically sorted")
    for leaf in leaves:
        if any(parent in leaves for parent in leaf.parents if parent != PurePosixPath(".")):
            raise RuntimeError("offline receipt inventory has a leaf/parent collision")
    return inventory


def _validate_receipt(
    receipt: Any, *, label: str, registered_wandb_dir: Path
) -> dict[str, Any]:
    expected = _registered_identity(label)
    if not isinstance(receipt, dict) or set(receipt) != RECEIPT_KEYS:
        raise RuntimeError("offline terminal receipt schema changed")
    expected_embedded = {name: f"files/{name}" for name in EVIDENCE_FILE_NAMES}
    hashes = receipt.get("queued_evidence_sha256")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("transport") != "offline_wandb_pending_explicit_sync"
        or receipt.get("sync_completed") is not False
        or receipt.get("remote_artifact_claimed") is not False
        or receipt.get("wandb_identity") != expected
        or receipt.get("wandb_dir") != str(registered_wandb_dir)
        or receipt.get("embedded_evidence_relative_paths") != expected_embedded
        or receipt.get("embedded_evidence_storage")
        != "regular_files_inside_offline_run_root"
        or receipt.get("sync_policy") != SYNC_POLICY
        or not isinstance(hashes, dict)
        or set(hashes) != set(EVIDENCE_FILE_NAMES)
        or any(_SHA256.fullmatch(str(value)) is None for value in hashes.values())
        or receipt.get("terminal_run_result_sha256")
        != hashes["run_result_manifest.json"]
    ):
        raise RuntimeError("offline terminal receipt identity/transport changed")

    relative_root = _safe_relative(
        receipt.get("offline_run_root_relative_to_wandb_dir"),
        label="offline root relative path",
    )
    recorded_root = registered_wandb_dir / Path(*relative_root.parts)
    run_id = expected["run_id"]
    if (
        receipt.get("offline_run_root") != str(recorded_root)
        or not relative_root.name.startswith("offline-run-")
        or not relative_root.name.endswith(f"-{run_id}")
    ):
        raise RuntimeError("offline terminal receipt root changed")
    observed = receipt.get("observed_offline_identity")
    observable_keys = ("run_id", "entity", "project", "run_name", "group", "tags")
    if (
        not isinstance(observed, dict)
        or {key: observed.get(key) for key in observable_keys}
        != {key: expected[key] for key in observable_keys}
        or observed.get("url") is not None
        or observed.get("resume") != "never"
        or observed.get("mode") != "offline"
        or observed.get("run_files_dir") != str(recorded_root / "files")
    ):
        raise RuntimeError("observed offline W&B identity changed")

    inventory = _validated_inventory(receipt, expected_label=label)
    by_path = {item["path"]: item for item in inventory}
    for name, expected_hash in hashes.items():
        item = by_path.get(f"files/{name}")
        if item is None or item.get("type") != "file" or item.get("sha256") != expected_hash:
            raise RuntimeError(f"receipt does not bind embedded terminal evidence: {name}")
    archive = by_path.get(f"run-{run_id}.wandb")
    if archive is None or archive.get("type") != "file":
        raise RuntimeError("completed offline W&B archive is missing from receipt")
    return receipt


def _filesystem_inventory(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for current, dirnames, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        retained: list[str] = []
        for name in dirnames:
            path = current_path / name
            if path.is_symlink():
                records.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "type": "symlink",
                        "target": os.readlink(path),
                        "mode": oct(stat.S_IMODE(path.lstat().st_mode)),
                    }
                )
            elif path.is_dir():
                retained.append(name)
            else:
                raise RuntimeError("special directory entry in offline W&B tree")
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
                records.append(
                    {
                        "path": relative,
                        "type": "file",
                        "sha256": _sha256_file(path),
                        "size_bytes": path.stat().st_size,
                        "mode": oct(stat.S_IMODE(path.stat().st_mode)),
                    }
                )
            else:
                raise RuntimeError("special file entry in offline W&B tree")
    return sorted(records, key=lambda item: item["path"])


def _add_regular(tar: tarfile.TarFile, source: Path, name: str, mode: int) -> None:
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"payload source is not regular: {source}")
    info = tarfile.TarInfo(name)
    info.type = tarfile.REGTYPE
    info.size = source.stat().st_size
    info.mode = mode
    info.mtime = 0
    info.uid = info.gid = 0
    with source.open("rb") as handle:
        tar.addfile(info, handle)


def _build_payload_archive(
    *, receipt_path: Path, offline_root: Path, archive_path: Path, label: str
) -> None:
    receipt = _read_json_bytes(receipt_path.read_bytes(), label="offline receipt")
    inventory = _validated_inventory(receipt, expected_label=label)
    with tarfile.open(archive_path, "w", format=tarfile.PAX_FORMAT) as tar:
        _add_regular(tar, receipt_path, OFFLINE_RECEIPT_NAME, 0o600)
        for item in inventory:
            relative = _safe_relative(item["path"], label="inventory path")
            source = offline_root.joinpath(*relative.parts)
            archive_name = f"offline_run/{relative.as_posix()}"
            if item["type"] == "file":
                _add_regular(tar, source, archive_name, int(item["mode"], 8))
            else:
                if not source.is_symlink() or os.readlink(source) != item["target"]:
                    raise RuntimeError(f"offline symlink changed: {relative}")
                info = tarfile.TarInfo(archive_name)
                info.type = tarfile.SYMTYPE
                info.linkname = item["target"]
                info.mode = int(item["mode"], 8)
                info.mtime = 0
                info.uid = info.gid = 0
                tar.addfile(info)
    os.chmod(archive_path, 0o600)


def _validate_payload_archive(
    archive_path: Path, *, label: str, expected_receipt_sha256: str
) -> dict[str, Any]:
    if _SHA256.fullmatch(expected_receipt_sha256) is None:
        raise RuntimeError("invalid expected receipt digest")
    if archive_path.is_symlink() or not archive_path.is_file():
        raise RuntimeError("sync payload is not a regular file")
    if archive_path.stat().st_size > MAX_PAYLOAD_BYTES:
        raise RuntimeError("sync payload exceeds its size cap")
    with tarfile.open(archive_path, "r:") as tar:
        members = tar.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or OFFLINE_RECEIPT_NAME not in names:
            raise RuntimeError("sync payload has duplicate entries or no receipt")
        receipt_member = tar.getmember(OFFLINE_RECEIPT_NAME)
        if not receipt_member.isreg() or receipt_member.size > 1_000_000:
            raise RuntimeError("sync payload receipt entry is invalid")
        receipt_stream = tar.extractfile(receipt_member)
        if receipt_stream is None:
            raise RuntimeError("sync payload receipt is unreadable")
        receipt_bytes = receipt_stream.read()
        if hashlib.sha256(receipt_bytes).hexdigest() != expected_receipt_sha256:
            raise RuntimeError("sync payload receipt digest changed")
        receipt = _read_json_bytes(receipt_bytes, label="sync payload receipt")
        _validate_receipt(
            receipt,
            label=label,
            registered_wandb_dir=REGISTERED_STATE / "offline_wandb" / label,
        )
        inventory = _validated_inventory(receipt, expected_label=label)
        expected_names = {
            OFFLINE_RECEIPT_NAME,
            *(f"offline_run/{item['path']}" for item in inventory),
        }
        if set(names) != expected_names:
            raise RuntimeError("sync payload contains data outside the sealed receipt")
        total = len(receipt_bytes)
        for item in inventory:
            member = tar.getmember(f"offline_run/{item['path']}")
            if item["type"] == "file":
                if (
                    not member.isreg()
                    or member.size != item["size_bytes"]
                    or member.mode != int(item["mode"], 8)
                ):
                    raise RuntimeError(f"payload metadata changed: {item['path']}")
                stream = tar.extractfile(member)
                if stream is None:
                    raise RuntimeError(f"payload file is unreadable: {item['path']}")
                digest = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
                if digest.hexdigest() != item["sha256"]:
                    raise RuntimeError(f"payload bytes changed: {item['path']}")
                total += member.size
            elif (
                not member.issym()
                or member.linkname != item["target"]
                or member.mode != int(item["mode"], 8)
            ):
                raise RuntimeError(f"payload symlink changed: {item['path']}")
        if total > MAX_PAYLOAD_BYTES:
            raise RuntimeError("receipt-bound payload contents exceed their size cap")
    return receipt


def _validate_protocol_state(state_dir: Path, label: str) -> None:
    path = state_dir / "protocol_state.json"
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("registered V9-local protocol state is missing")
    state = _read_json_bytes(path.read_bytes(), label="registered protocol state")
    if not isinstance(state, dict):
        raise RuntimeError("V9-local protocol/config registration changed")
    prepared = state.get("prepared_file_sha256")
    config_name = f"configs/{label}.json"
    if (
        state.get("schema_version") != 1
        or state.get("protocol") != PROTOCOL
        or state.get("git_commit") != REGISTERED_GIT_COMMIT
        or state.get("registration_sha256") != REGISTRATION_SHA256
        or state.get("recipe_lock_sha256") != RECIPE_LOCK_SHA256
        or state.get("launch_enabled") is not True
        or state.get("seeds") != list(SEEDS)
        or state.get("conditions") != list(CONDITIONS)
        or not isinstance(prepared, dict)
        or prepared.get(config_name) != CONFIG_SHA256[label]
    ):
        raise RuntimeError("V9-local protocol/config registration changed")


def _local_config_identity(label: str, state_dir: Path) -> dict[str, Any]:
    expected = _registered_identity(label)
    _validate_protocol_state(state_dir, label)
    config_path = state_dir / "configs" / f"{label}.json"
    if config_path.is_symlink() or not config_path.is_file():
        raise RuntimeError(f"missing registered V9-local config for {label}")
    if _sha256_file(config_path) != CONFIG_SHA256[label]:
        raise RuntimeError("prepared V9-local config bytes changed")
    config = _read_json_bytes(config_path.read_bytes(), label="registered config")
    match = _LABEL.fullmatch(label)
    assert match is not None
    condition, raw_seed = match.groups()
    observed = {
        "entity": config.get("wandb_entity"),
        "project": config.get("wandb_project"),
        "run_name": config.get("run_name"),
        "run_id": config.get("wandb_run_id"),
        "url": config.get("wandb_url"),
        "group": config.get("wandb_group"),
        "tags": config.get("wandb_tags"),
        "resume": config.get("wandb_resume"),
    }
    score_components = config.get("score_components")
    expected_weight = -1.0 if condition == "jlens" else 1.0
    if (
        observed != expected
        or config.get("wandb_mode") != "offline"
        or config.get("updates") != 20
        or config.get("seed") != int(raw_seed)
        or config.get("target_words") != ["damn", "fuck"]
        or not isinstance(score_components, list)
        or len(score_components) != 1
        or score_components[0].get("weight") != expected_weight
        or config.get("registration_sha256") != REGISTRATION_SHA256
        or config.get("recipe_lock_sha256") != RECIPE_LOCK_SHA256
        or config.get("evidence_eligibility")
        != "original_registered_v9_local_attempt"
        or config.get("output_dir") != f".confirmatory/v9_local/runs/{label}"
    ):
        raise RuntimeError("prepared config differs from the registered V9 identity")
    return expected


def _default_canonical_validator(
    output_dir: Path, identity: dict[str, Any], wandb_dir: Path
) -> dict[str, Any]:
    from scripts.v9_local_train import validate_offline_terminal_receipt

    return validate_offline_terminal_receipt(output_dir, identity, wandb_dir)


def _prepare_local_payload(
    label: str,
    archive_path: Path,
    *,
    state_dir: Path = LOCAL_STATE,
    canonical_validator: Callable[[Path, dict[str, Any], Path], dict[str, Any]]
    | None = None,
) -> dict[str, Any]:
    if state_dir.resolve() != REGISTERED_STATE:
        raise RuntimeError("V9-local sync must use the registered /j-lens-rl state root")
    identity = _local_config_identity(label, state_dir)
    output_dir = state_dir / "runs" / label
    wandb_dir = state_dir / "offline_wandb" / label
    validator = canonical_validator or _default_canonical_validator
    receipt = validator(output_dir, identity, wandb_dir)
    receipt_path = output_dir / OFFLINE_RECEIPT_NAME
    receipt_sha256 = _sha256_file(receipt_path)
    if receipt != _read_json_bytes(receipt_path.read_bytes(), label="offline receipt"):
        raise RuntimeError("canonical validator returned different receipt bytes")
    _validate_receipt(receipt, label=label, registered_wandb_dir=wandb_dir)
    offline_root = Path(receipt["offline_run_root"])
    _build_payload_archive(
        receipt_path=receipt_path,
        offline_root=offline_root,
        archive_path=archive_path,
        label=label,
    )
    # Close both races: state evidence changing and the offline tree changing.
    if validator(output_dir, identity, wandb_dir) != receipt:
        raise RuntimeError("offline receipt/tree changed while packaging")
    if _sha256_file(receipt_path) != receipt_sha256:
        raise RuntimeError("offline receipt changed while packaging")
    _validate_payload_archive(
        archive_path, label=label, expected_receipt_sha256=receipt_sha256
    )
    return {
        "label": label,
        "receipt_sha256": receipt_sha256,
        "offline_run_tree_sha256": receipt["offline_run_tree_sha256"],
        "payload_sha256": _sha256_file(archive_path),
        "payload_size_bytes": archive_path.stat().st_size,
    }


def _validate_embedded_terminal_evidence(
    offline_root: Path, receipt: dict[str, Any], label: str
) -> dict[str, Any]:
    for name, expected_hash in receipt["queued_evidence_sha256"].items():
        path = offline_root / "files" / name
        if path.is_symlink() or not path.is_file() or _sha256_file(path) != expected_hash:
            raise RuntimeError(f"embedded terminal evidence differs: {name}")
    result = _read_json_bytes(
        (offline_root / "files" / "run_result_manifest.json").read_bytes(),
        label="embedded terminal result",
    )
    if (
        not isinstance(result, dict)
        or result.get("schema_version") != 1
        or result.get("completed_updates") != 20
        or result.get("wandb_identity") != _registered_identity(label)
        or result.get("registration_sha256") != REGISTRATION_SHA256
        or result.get("recipe_lock_sha256") != RECIPE_LOCK_SHA256
        or not isinstance(result.get("source"), dict)
        or result["source"].get("git_commit") != REGISTERED_GIT_COMMIT
        or result["source"].get("git_dirty") is not False
        or result.get("evidence_eligibility")
        != "original_registered_v9_local_attempt"
    ):
        raise RuntimeError("embedded terminal result is not the registered completed run")
    return result


def _extract_payload(
    archive_path: Path,
    destination: Path,
    *,
    label: str,
    expected_receipt_sha256: str,
) -> tuple[dict[str, Any], Path]:
    receipt = _validate_payload_archive(
        archive_path, label=label, expected_receipt_sha256=expected_receipt_sha256
    )
    if destination.exists():
        raise RuntimeError("ephemeral sync extraction path already exists")
    destination.mkdir(parents=True, mode=0o700)
    offline_root = destination / Path(receipt["offline_run_root"]).name
    offline_root.mkdir(mode=0o700)
    with tarfile.open(archive_path, "r:") as tar:
        receipt_stream = tar.extractfile(tar.getmember(OFFLINE_RECEIPT_NAME))
        if receipt_stream is None:
            raise RuntimeError("sync payload receipt is unreadable")
        (destination / OFFLINE_RECEIPT_NAME).write_bytes(receipt_stream.read())
        for item in _validated_inventory(receipt, expected_label=label):
            relative = _safe_relative(item["path"], label="inventory path")
            target = offline_root.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            member = tar.getmember(f"offline_run/{item['path']}")
            if item["type"] == "file":
                source = tar.extractfile(member)
                if source is None:
                    raise RuntimeError(f"payload file is unreadable: {relative}")
                with target.open("xb") as handle:
                    shutil.copyfileobj(source, handle, length=1024 * 1024)
                target.chmod(int(item["mode"], 8))
            else:
                os.symlink(item["target"], target)
    observed = _filesystem_inventory(offline_root)
    if (
        observed != receipt["offline_run_file_symlink_inventory"]
        or _canonical_sha256(observed) != receipt["offline_run_tree_sha256"]
    ):
        raise RuntimeError("extracted offline W&B tree differs from its receipt")
    _validate_embedded_terminal_evidence(offline_root, receipt, label)
    return receipt, offline_root


def _require_external_debug_link_is_dangling(
    offline_root: Path, receipt: dict[str, Any], label: str
) -> None:
    """Prevent W&B's one allowed absolute debug link from exposing host data."""
    for item in _validated_inventory(receipt, expected_label=label):
        if item["type"] != "symlink" or not PurePosixPath(item["target"]).is_absolute():
            continue
        relative = _safe_relative(item["path"], label="inventory path")
        _safe_symlink_target(relative, item["target"], expected_label=label)
        if Path(item["target"]).exists():
            raise RuntimeError("registered W&B debug symlink is not dangling remotely")


def _copy_verified(source: Path, destination: Path, expected_sha256: str) -> None:
    if _SHA256.fullmatch(expected_sha256) is None:
        raise RuntimeError("invalid payload digest")
    if source.is_symlink() or not source.is_file() or source.stat().st_size > MAX_PAYLOAD_BYTES:
        raise RuntimeError("staged payload is missing, symlinked, or oversized")
    digest = hashlib.sha256()
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        for chunk in iter(lambda: incoming.read(1024 * 1024), b""):
            digest.update(chunk)
            outgoing.write(chunk)
    if digest.hexdigest() != expected_sha256:
        destination.unlink(missing_ok=True)
        raise RuntimeError("staged payload digest changed")


def _wandb_sync_command(offline_root: Path, identity: dict[str, Any]) -> list[str]:
    executable = shutil.which("wandb")
    if executable is None:
        raise RuntimeError("pinned W&B CLI is unavailable")
    return [
        executable,
        "sync",
        "--id",
        identity["run_id"],
        "--project",
        identity["project"],
        "--entity",
        identity["entity"],
        "--no-sync-tensorboard",
        "--include-offline",
        "--no-include-online",
        str(offline_root),
    ]


def _lookup_remote_run(api: Any, identity: dict[str, Any]) -> Any | None:
    from wandb.errors import CommError

    try:
        return api.run(
            f"{identity['entity']}/{identity['project']}/{identity['run_id']}"
        )
    except CommError as error:
        if str(error).startswith("Could not find run "):
            return None
        raise RuntimeError("W&B remote identity lookup failed") from None


def _verify_remote_run(
    remote: Any, identity: dict[str, Any], receipt: dict[str, Any], result: dict[str, Any]
) -> dict[str, Any]:
    observed = {
        "entity": remote.entity,
        "project": remote.project,
        "run_name": remote.name,
        "run_id": remote.id,
        "url": remote.url,
        "group": remote.group,
        "tags": sorted(remote.tags),
    }
    expected = {key: identity[key] for key in observed}
    expected["tags"] = sorted(expected["tags"])
    if observed != expected:
        raise RuntimeError("synced W&B run differs from its registered identity")
    queue = {
        "schema_version": 1,
        "transport": "offline_wandb_pending_explicit_sync",
        "remote_artifact_claimed": False,
        "wandb_identity": identity,
        "terminal_run_result_sha256": receipt["terminal_run_result_sha256"],
        "queued_evidence_sha256": receipt["queued_evidence_sha256"],
    }
    if (
        remote.config.get("offline_terminal_evidence_queue") != queue
        or remote.config.get("terminal_run_result") != result
    ):
        raise RuntimeError("synced W&B config lacks the receipt-bound terminal evidence")
    remote_files = sorted(file.name for file in remote.files())
    if not set(EVIDENCE_FILE_NAMES).issubset(remote_files):
        raise RuntimeError("synced W&B run lacks terminal evidence files")
    return {
        **observed,
        "state": remote.state,
        "terminal_evidence_files": sorted(EVIDENCE_FILE_NAMES),
    }


def _validated_staged_request(
    label: str, staged_path: str, payload_sha256: str, receipt_sha256: str
) -> tuple[dict[str, Any], Path]:
    identity = _registered_identity(label)
    expected_staged_path = f"/payloads/{receipt_sha256}-{label}.tar"
    if (
        _SHA256.fullmatch(receipt_sha256) is None
        or _SHA256.fullmatch(payload_sha256) is None
        or staged_path != expected_staged_path
    ):
        raise RuntimeError("staged payload identity is not canonical")
    relative = _safe_relative(staged_path.removeprefix("/"), label="staged payload path")
    return identity, STAGING_MOUNT.joinpath(*relative.parts)


@app.function(
    image=sync_image,
    cpu=1.0,
    memory=1024,
    timeout=900,
    secrets=[wandb_secret],
    volumes={STAGING_MOUNT.as_posix(): staging_volume},
)
def sync_one_completed_v9_run(
    label: str,
    staged_path: str,
    payload_sha256: str,
    receipt_sha256: str,
) -> dict[str, Any]:
    """Validate and sync exactly one staged terminal run; never overwrite."""
    identity, source = _validated_staged_request(
        label, staged_path, payload_sha256, receipt_sha256
    )
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("Modal W&B secret did not provide WANDB_API_KEY")
    os.environ.update(
        {
            "WANDB_BASE_URL": "https://api.wandb.ai",
            "WANDB_ENTITY": identity["entity"],
            "WANDB_PROJECT": identity["project"],
            "WANDB_SILENT": "true",
            "WANDB_CONSOLE": "off",
        }
    )
    staging_volume.reload()
    with tempfile.TemporaryDirectory(prefix="v9-wandb-sync-") as temporary:
        work = Path(temporary)
        archive = work / "payload.tar"
        _copy_verified(source, archive, payload_sha256)
        receipt, offline_root = _extract_payload(
            archive,
            work / "extracted",
            label=label,
            expected_receipt_sha256=receipt_sha256,
        )
        result = _validate_embedded_terminal_evidence(offline_root, receipt, label)
        _require_external_debug_link_is_dangling(offline_root, receipt, label)

        import wandb

        api = wandb.Api(timeout=30)
        if _lookup_remote_run(api, identity) is not None:
            raise RuntimeError("registered W&B run ID already exists; refusing overwrite")
        completed = subprocess.run(
            _wandb_sync_command(offline_root, identity),
            cwd=work,
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=600,
            check=False,
        )
        if completed.returncode != 0:
            # Do not surface output from a process carrying the API secret.
            raise RuntimeError(f"W&B sync failed with exit code {completed.returncode}")
        remote = None
        for _attempt in range(10):
            remote = _lookup_remote_run(api, identity)
            if remote is not None:
                break
            time.sleep(2)
        if remote is None:
            raise RuntimeError("synced W&B run was not visible after upload")
        remote_record = _verify_remote_run(remote, identity, receipt, result)
        return {
            "schema_version": 1,
            "transport": "modal_cpu_offline_wandb_sync",
            "label": label,
            "payload_sha256": payload_sha256,
            "receipt_sha256": receipt_sha256,
            "offline_run_tree_sha256": receipt["offline_run_tree_sha256"],
            "remote": remote_record,
            "synced_at_utc": datetime.now(timezone.utc).isoformat(),
        }


@app.function(
    image=sync_image,
    cpu=1.0,
    memory=1024,
    timeout=300,
    secrets=[wandb_secret],
    volumes={STAGING_MOUNT.as_posix(): staging_volume},
)
def verify_existing_synced_v9_run(
    label: str,
    staged_path: str,
    payload_sha256: str,
    receipt_sha256: str,
) -> dict[str, Any]:
    """Verify an existing remote against the sealed payload; never upload."""
    identity, source = _validated_staged_request(
        label, staged_path, payload_sha256, receipt_sha256
    )
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("Modal W&B secret did not provide WANDB_API_KEY")
    staging_volume.reload()
    with tempfile.TemporaryDirectory(prefix="v9-wandb-verify-") as temporary:
        work = Path(temporary)
        archive = work / "payload.tar"
        _copy_verified(source, archive, payload_sha256)
        receipt, offline_root = _extract_payload(
            archive,
            work / "extracted",
            label=label,
            expected_receipt_sha256=receipt_sha256,
        )
        result = _validate_embedded_terminal_evidence(offline_root, receipt, label)
        _require_external_debug_link_is_dangling(offline_root, receipt, label)

        import wandb

        remote = _lookup_remote_run(wandb.Api(timeout=30), identity)
        if remote is None:
            raise RuntimeError("registered W&B run is absent after sync")
        remote_record = _verify_remote_run(remote, identity, receipt, result)
        return {
            "schema_version": 1,
            "transport": "modal_cpu_existing_wandb_verification",
            "label": label,
            "payload_sha256": payload_sha256,
            "receipt_sha256": receipt_sha256,
            "offline_run_tree_sha256": receipt["offline_run_tree_sha256"],
            "remote": remote_record,
            "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        }


@app.local_entrypoint()
def main(label: str) -> None:
    """Stage and sync one receipt-valid V9-local run; never starts training."""
    with tempfile.TemporaryDirectory(prefix="v9-wandb-sync-package-") as temporary:
        archive = Path(temporary) / "payload.tar"
        prepared = _prepare_local_payload(label, archive)
        staged_path = f"/payloads/{prepared['receipt_sha256']}-{label}.tar"
        with staging_volume.batch_upload(force=False) as batch:
            batch.put_file(archive, staged_path, mode=0o600)
        result = sync_one_completed_v9_run.remote(
            label,
            staged_path,
            prepared["payload_sha256"],
            prepared["receipt_sha256"],
        )
    print(json.dumps(result, indent=2, sort_keys=True))


@app.local_entrypoint(name="verify")
def verify_main(label: str) -> None:
    """Verify a prior sync against the still-sealed local evidence."""
    with tempfile.TemporaryDirectory(prefix="v9-wandb-verify-package-") as temporary:
        archive = Path(temporary) / "payload.tar"
        prepared = _prepare_local_payload(label, archive)
        staged_path = f"/payloads/{prepared['receipt_sha256']}-{label}.tar"
        result = verify_existing_synced_v9_run.remote(
            label,
            staged_path,
            prepared["payload_sha256"],
            prepared["receipt_sha256"],
        )
    print(json.dumps(result, indent=2, sort_keys=True))
