"""Reject every file outside the tournament's exact runtime inventory."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
FORBIDDEN_PARTS = {
    ".git",
    ".confirmatory",
    "protocol_archive",
    "wandb",
    "runs",
    "evals",
    "history",
    "histories",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_build_debris() -> None:
    shutil.rmtree(REPO / "build", ignore_errors=True)
    for path in sorted(REPO.rglob("*"), reverse=True):
        if path.is_dir() and (
            path.name == "build"
            or path.name == "__pycache__"
            or path.name.endswith(".egg-info")
        ):
            shutil.rmtree(path, ignore_errors=True)
        elif path.is_file() and path.suffix in {".pyc", ".pyo"}:
            path.unlink()


def validate_exact_image_inventory(repo: Path, expected: object) -> None:
    if (
        not isinstance(expected, dict)
        or not expected
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or Path(name).is_absolute()
            or ".." in Path(name).parts
            or any(part.startswith(".") for part in Path(name).parts)
            for name, digest in expected.items()
        )
    ):
        raise RuntimeError("tournament image source inventory is malformed")
    unsafe = []
    observed: set[str] = set()
    for path in repo.rglob("*"):
        relative = path.relative_to(repo)
        name = relative.as_posix()
        if (
            path.is_symlink()
            or FORBIDDEN_PARTS & set(relative.parts)
            or any(part.startswith(".") for part in relative.parts)
        ):
            unsafe.append(name)
        if path.is_file():
            observed.add(name)
    missing = sorted(set(expected) - observed)
    unexpected = sorted(observed - set(expected))
    wrong_hash = sorted(
        name
        for name in observed & set(expected)
        if _sha256(repo / name) != expected[name]
    )
    if unsafe or missing or unexpected or wrong_hash:
        raise RuntimeError(
            "tournament strict Modal image inventory failed: "
            f"unsafe={sorted(unsafe)}, missing={missing}, "
            f"unexpected={unexpected}, wrong_hash={wrong_hash}"
        )


def main() -> None:
    if os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN leaked beyond the isolated cache-build layer")
    _remove_build_debris()
    try:
        expected = json.loads(os.environ["JLENS_TOURNAMENT_V1_IMAGE_FILE_SHA256"])
    except (KeyError, json.JSONDecodeError) as error:
        raise RuntimeError("tournament image lacks its exact source inventory") from error
    validate_exact_image_inventory(REPO, expected)

    import jlens_rl
    import jlens_rl.train

    for module in (jlens_rl, jlens_rl.train):
        try:
            Path(module.__file__).resolve().relative_to(REPO / "src")
        except ValueError as error:
            raise RuntimeError(
                f"runtime imports {module.__name__} outside allowlisted source"
            ) from error


if __name__ == "__main__":
    main()
