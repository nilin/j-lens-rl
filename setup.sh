#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip==26.0.1
python -m pip install -e './trl[peft]'
python -m pip install -e '.[dev]'

python - <<'PY'
from importlib.metadata import version
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd() / "trl"))
import torch, transformers, jlens, trl
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("jlens", jlens.__file__)
print("trl", version("trl"), trl.__file__)
if not torch.cuda.is_available():
    print("WARNING: CUDA is not visible; fitting/training is intended for a CUDA GPU.")
PY

echo "Setup complete. Activate with: source .venv/bin/activate"
