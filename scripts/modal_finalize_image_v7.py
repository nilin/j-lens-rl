"""Prove the V7 Modal image contains only its outcome-free runtime allowlist."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
EXACT_FILES = {
    "pyproject.toml",
    "modal_confirmatory_v7.py",
    "run_confirmatory_v7.sh",
    "scripts/confirmatory_v7_protocol.py",
    "scripts/modal_cache_assets_v7.py",
    "scripts/modal_finalize_image_v7.py",
    "scripts/modal_verify_v7_volume.py",
    "artifacts/qwen25_05b_solved_lens.pt",
    "trl/pyproject.toml",
    "trl/MANIFEST.in",
    "trl/VERSION",
    "trl/README.md",
    "trl/LICENSE",
    "trl/CONTRIBUTING.md",
}
ALLOWED_TREES = ("src/jlens_rl/", "trl/trl/")
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


def _remove_build_debris() -> None:
    shutil.rmtree(REPO / "build", ignore_errors=True)
    for path in sorted(REPO.rglob("*"), reverse=True):
        if path.is_dir() and (
            path.name == "__pycache__" or path.name.endswith(".egg-info")
        ):
            shutil.rmtree(path, ignore_errors=True)
        elif path.is_file() and path.suffix in {".pyc", ".pyo"}:
            path.unlink()


def main() -> None:
    if os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN leaked beyond the isolated V7 cache-build layer")
    _remove_build_debris()
    observed = {
        path.relative_to(REPO).as_posix()
        for path in REPO.rglob("*")
        if path.is_file()
    }
    forbidden = [
        name for name in observed if FORBIDDEN_PARTS & set(Path(name).parts)
    ]
    unexpected = [
        name
        for name in observed
        if name not in EXACT_FILES
        and not any(name.startswith(prefix) for prefix in ALLOWED_TREES)
    ]
    missing = sorted(EXACT_FILES - observed)
    if forbidden or unexpected or missing:
        raise RuntimeError(
            "V7 strict Modal image inventory failed: "
            f"forbidden={sorted(forbidden)}, unexpected={sorted(unexpected)}, "
            f"missing={missing}"
        )

    import jlens_rl
    import jlens_rl.train

    for module in (jlens_rl, jlens_rl.train):
        try:
            Path(module.__file__).resolve().relative_to(REPO / "src")
        except ValueError as error:
            raise RuntimeError(
                f"V7 runtime imports {module.__name__} outside allowlisted source"
            ) from error


if __name__ == "__main__":
    main()
