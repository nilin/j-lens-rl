from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

SYSTEM_PROMPT = (
    "Solve the math problem. Show concise reasoning, then put only the final "
    "numeric answer after '#### '."
)

QWEN_MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"


def runtime_environment_snapshot() -> dict[str, Any]:
    """Capture replay-critical software, CUDA/driver, OS, and image identity."""
    freeze = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze", "--all"], text=True
    ).splitlines()
    try:
        driver = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
    except (OSError, subprocess.CalledProcessError):
        driver = []
    os_release_path = Path("/etc/os-release")
    return {
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": platform.platform(),
        "os_release": (
            os_release_path.read_text().splitlines()
            if os_release_path.is_file()
            else []
        ),
        "pip_freeze_all": sorted(line for line in freeze if line),
        "torch": {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        },
        "nvidia_smi_name_and_driver": driver,
        "cuda_device_names": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
        "image_identity": {
            "jlens_modal_image_spec": os.environ.get("JLENS_MODAL_IMAGE_SPEC"),
            "modal_image_id": os.environ.get("MODAL_IMAGE_ID"),
        },
    }


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    cfg = json.loads(path.read_text())
    if "base" in cfg:
        base = load_config(path.parent / cfg.pop("base"))
        base.update(cfg)
        cfg = base
    return cfg


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_index_manifest(path: str | Path) -> list[int]:
    """Load and validate a JSON index manifest.

    Manifests may be a bare list or an object with an ``indices`` list. Keeping
    selection by raw source index makes data boundaries independent of dataset
    shuffling and easy to audit.
    """
    source = Path(path)
    payload = json.loads(source.read_text())
    values = payload.get("indices") if isinstance(payload, dict) else payload
    if not isinstance(values, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in values
    ):
        raise ValueError(f"{source} must contain a list of integer indices")
    if any(value < 0 for value in values):
        raise ValueError(f"{source} contains a negative index")
    if len(values) != len(set(values)):
        raise ValueError(f"{source} contains duplicate indices")
    return values


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_repository_root(module_file: str | Path | None = None) -> Path:
    """Locate the checkout whose source is executing.

    Wheel installs place ``__file__`` under site-packages, so deriving the
    repository from a fixed number of parents can silently lose Git
    provenance.  Confirmatory remote jobs set an explicit root; local source
    runs fall back to the current Git worktree and then the module location.
    """
    override = os.environ.get("JLENS_REPOSITORY_ROOT")
    candidates = [Path(override)] if override else [Path.cwd()]
    if module_file is not None:
        candidates.append(Path(module_file).resolve().parent)

    for candidate in candidates:
        try:
            top_level = subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=candidate,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            if override:
                raise RuntimeError(
                    "JLENS_REPOSITORY_ROOT is not inside a Git worktree: "
                    f"{candidate.resolve()}"
                ) from None
            continue
        return Path(top_level).resolve()

    if module_file is not None:
        path = Path(module_file).resolve()
        if len(path.parents) >= 3:
            return path.parents[2]
    return Path.cwd().resolve()


def repository_provenance(root: str | Path) -> dict[str, Any]:
    """Return a content fingerprint that still identifies a dirty worktree."""
    root = Path(root).resolve()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=root,
            text=True,
        ).splitlines()
        names = subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=root,
            text=True,
        ).splitlines()
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None, "source_tree_sha256": None}

    digest = hashlib.sha256()
    for name in sorted(names):
        path = root / name
        if not path.is_file():
            continue
        digest.update(name.encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return {
        "git_commit": commit,
        "git_dirty": bool(status),
        "git_status": status,
        "source_tree_sha256": digest.hexdigest(),
    }


def require_clean_repository_provenance(provenance: dict[str, Any]) -> None:
    """Fail closed when a run cannot prove clean committed source identity."""
    commit = provenance.get("git_commit")
    source_tree = provenance.get("source_tree_sha256")
    if not isinstance(commit, str) or len(commit) != 40:
        raise RuntimeError("required Git commit provenance is unavailable")
    if provenance.get("git_dirty") is not False:
        raise RuntimeError("confirmatory execution requires a clean Git worktree")
    if not isinstance(source_tree, str) or len(source_tree) != 64:
        raise RuntimeError("required source-tree fingerprint is unavailable")


def extract_answer(text: str) -> str | None:
    marked = re.findall(r"####\s*([-+]?[$\d][\d,]*(?:\.\d+)?)", text)
    candidates = marked or re.findall(r"[-+]?[$\d][\d,]*(?:\.\d+)?", text)
    if not candidates:
        return None
    value = candidates[-1].replace("$", "").replace(",", "")
    try:
        number = float(value)
        return str(int(number)) if number.is_integer() else str(number)
    except ValueError:
        return None


def gsm8k_reward(completion: str, reference: str) -> float:
    return float(extract_answer(completion) == extract_answer(reference))


def binomial_ci95(successes: int, total: int) -> tuple[float, float]:
    """Wilson 95% interval, which remains meaningful at 0% and 100%."""
    if total <= 0:
        raise ValueError("total must be positive")
    z = 1.96
    p = successes / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    radius = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return float(max(0, center - radius)), float(min(1, center + radius))


def format_prompt(tokenizer: Any, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def model_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")
