#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export WANDB_API_KEY="$(tr -d '\r\n' < .env)"

# Let the late-half/all-layer run finish (normally via validation early stopping),
# then screen each fitted layer independently from the untouched base model.
until [[ -f runs/jlens_solved_late_all/final/adapter_model.safetensors ]]; do
  sleep 30
done

for config in \
  configs/jlens_late_8.json \
  configs/jlens_late_14.json \
  configs/jlens_late_20.json
do
  .venv/bin/train-jlens-rl --config "$config" --wandb-mode online
done
