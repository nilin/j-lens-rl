"""Verify the allowlisted, outcome-firewalled correlation image snapshot.

This image deliberately is not a Git checkout.  Its source provenance is
computed in the clean local launcher, baked as immutable image environment
metadata, and checked here after package installation and asset caching.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path, PurePosixPath
from typing import Any


REPO = Path(__file__).resolve().parents[1]
PROVENANCE_ENV = "JLENS_CORRELATION_SOURCE_PROVENANCE_JSON"
PROTOCOL = "j-lens-rl-word-correlation-source-snapshot-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_provenance() -> dict[str, Any]:
    raw = os.environ.get(PROVENANCE_ENV)
    if not raw:
        raise RuntimeError("correlation image has no baked source provenance")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError("correlation source provenance is invalid JSON") from error
    if not isinstance(payload, dict):
        raise RuntimeError("correlation source provenance is not an object")
    hashes = payload.get("image_file_sha256")
    if (
        payload.get("protocol") != PROTOCOL
        or payload.get("git_dirty") is not False
        or payload.get("repository_metadata_included") is not False
        or not isinstance(payload.get("git_commit"), str)
        or len(payload["git_commit"]) != 40
        or any(character not in "0123456789abcdef" for character in payload["git_commit"])
        or payload.get("git_status_sha256") != hashlib.sha256(b"").hexdigest()
        or not isinstance(hashes, dict)
        or not hashes
    ):
        raise RuntimeError("correlation source provenance has an invalid identity")
    for relative, digest in hashes.items():
        path = PurePosixPath(relative) if isinstance(relative, str) else None
        if (
            path is None
            or path.is_absolute()
            or not path.parts
            or any(part in {"", ".", "..", ".git"} for part in path.parts)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise RuntimeError("correlation source provenance has an unsafe file")
    claimed = payload.get("source_snapshot_sha256")
    without_claim = dict(payload)
    without_claim.pop("source_snapshot_sha256", None)
    if claimed != _canonical_sha256(without_claim):
        raise RuntimeError("correlation source snapshot hash is invalid")
    return payload


def _remove_build_byproducts() -> None:
    shutil.rmtree(REPO / "build", ignore_errors=True)
    for path in sorted(REPO.rglob("*"), reverse=True):
        if path.is_dir() and (
            path.name == "__pycache__" or path.name.endswith(".egg-info")
        ):
            shutil.rmtree(path)
        elif path.is_file() and path.suffix == ".pyc":
            path.unlink()


def _actual_files() -> set[str]:
    files: set[str] = set()
    for path in REPO.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"correlation image contains a symlink: {path}")
        if path.is_file():
            files.add(path.relative_to(REPO).as_posix())
    return files


def main() -> None:
    provenance = _load_provenance()
    expected = provenance["image_file_sha256"]
    if (REPO / ".git").exists() or any(path.name == ".git" for path in REPO.rglob(".git")):
        raise RuntimeError("correlation image must not contain Git metadata")

    # Import from the allowlisted source tree once, then remove deterministic
    # packaging/import byproducts before enforcing the exact file inventory.
    import jlens_rl.word_correlation as scanner

    scanner_path = Path(scanner.__file__).resolve()
    try:
        scanner_path.relative_to(REPO / "src")
    except ValueError as error:
        raise RuntimeError("correlation scanner imports outside the snapshot") from error
    _remove_build_byproducts()

    actual = _actual_files()
    if actual != set(expected):
        raise RuntimeError(
            "correlation image file inventory differs from the allowlist: "
            f"missing={sorted(set(expected) - actual)}, "
            f"extra={sorted(actual - set(expected))}"
        )
    actual_hashes = {relative: _sha256(REPO / relative) for relative in sorted(actual)}
    if actual_hashes != expected:
        raise RuntimeError("correlation image bytes differ from baked provenance")


if __name__ == "__main__":
    main()
