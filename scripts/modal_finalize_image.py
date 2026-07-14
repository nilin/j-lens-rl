"""Restore exact checkout semantics after Modal image construction.

Modal's local-directory copy materializes symlinks as regular files, and an
editable/source package build can leave an untracked ``build/`` directory.
The confirmatory protocol intentionally refuses to run from a dirty checkout,
so repair those deterministic image-build effects and fail the image build if
anything else differs from the committed source.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
TRACKED_SYMLINKS = {
    Path("trl/.cursor/BUGBOT.md"): "../.ai/AGENTS.md",
    Path("trl/AGENTS.md"): ".ai/AGENTS.md",
    Path("trl/CLAUDE.md"): ".ai/AGENTS.md",
}


def _restore_symlink(relative_path: Path, target: str) -> None:
    path = REPO / relative_path
    if path.is_symlink() and os.readlink(path) == target:
        return
    if path.is_dir() and not path.is_symlink():
        raise RuntimeError(f"refusing to replace unexpected directory: {path}")
    if path.exists() or path.is_symlink():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(target)


def main() -> None:
    for relative_path, target in TRACKED_SYMLINKS.items():
        _restore_symlink(relative_path, target)

    shutil.rmtree(REPO / "build", ignore_errors=True)

    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=REPO,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    if result.stdout:
        raise RuntimeError(
            "Modal image checkout is not clean after deterministic repairs:\n"
            f"{result.stdout}"
        )


if __name__ == "__main__":
    main()
