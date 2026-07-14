#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-.venv/bin/python}"
TRAIN="${TRAIN:-.venv/bin/train-jlens-rl}"
EVAL="${EVAL:-.venv/bin/eval-jlens-rl}"
PROTOCOL=("$PYTHON" scripts/confirmatory_v6_protocol.py)
STATE=.confirmatory/v6
SEEDS=(176 177 178 179 180 181 182 183)
export JLENS_MODAL_IMAGE_SPEC="${JLENS_MODAL_IMAGE_SPEC:-j-lens-rl-confirmatory-v6-celebration-taper-image-v1}"

usage() {
  echo "usage: $0 {registration-template|recipe-lock-template|prepare|verify|train-semantic|curve|train-controls|unlock|final-evaluation|report|modal}"
}

load_wandb_key() {
  if [[ -z "${WANDB_API_KEY:-}" && -f .env ]]; then
    export WANDB_API_KEY="$(tr -d '\r\n' < .env)"
  fi
}

train_condition() {
  local condition="$1"
  local seed
  "${PROTOCOL[@]}" verify >/dev/null
  if [[ "$condition" == "signflip" ]]; then
    "${PROTOCOL[@]}" verify-curve >/dev/null
  fi
  load_wandb_key
  for seed in "${SEEDS[@]}"; do
    "$TRAIN" \
      --config "$STATE/configs/${condition}_seed${seed}.json" \
      --wandb-mode online
  done
}

eval_if_missing() {
  local output="$1"
  local label="$2"
  local experiment_config="$3"
  shift 3
  if [[ -e "$output" ]]; then
    "${PROTOCOL[@]}" verify-eval --path "$output" --label "$label" >/dev/null
    echo "reusing complete, verified V6 evaluation: $output"
    return 0
  fi
  "$EVAL" \
    --config "$STATE/configs/sealed_eval.json" \
    --experiment-config "$experiment_config" \
    --indices-manifest "$STATE/manifests/sealed_final_indices.json" \
    --output-jsonl "$output" \
    --run-label "$label" \
    --batch-size 64 \
    --skip-jlens-metric \
    "$@"
  "${PROTOCOL[@]}" verify-eval --path "$output" --label "$label" >/dev/null
}

final_evaluation() {
  local seed output collection_id
  "${PROTOCOL[@]}" verify-unlock >/dev/null
  if [[ ! -e "$STATE/final_collection.json" ]]; then
    collection_id="$($PYTHON -c 'import uuid; print(uuid.uuid4().hex)')"
    "${PROTOCOL[@]}" begin-final --collection-id "$collection_id" >/dev/null
  fi
  collection_id="$($PYTHON -c 'import json; print(json.load(open(".confirmatory/v6/final_collection.json"))["collection_id"])')"
  "${PROTOCOL[@]}" verify-final --collection-id "$collection_id" >/dev/null
  mkdir -p "$STATE/evals" "$STATE/evidence"

  eval_if_missing \
    "$STATE/evals/base.jsonl" \
    base \
    "$STATE/configs/jlens_seed176.json"

  local compare_args=(--base-jsonl "$STATE/evals/base.jsonl")
  for seed in "${SEEDS[@]}"; do
    output="$STATE/evals/jlens_seed${seed}.jsonl"
    eval_if_missing \
      "$output" \
      "jlens_seed${seed}" \
      "$STATE/configs/jlens_seed${seed}.json" \
      --adapter "$STATE/runs/jlens_seed${seed}/final"
    compare_args+=(--adapter-jsonl "$output")
  done
  for seed in "${SEEDS[@]}"; do
    output="$STATE/evals/signflip_seed${seed}.jsonl"
    eval_if_missing \
      "$output" \
      "signflip_seed${seed}" \
      "$STATE/configs/signflip_seed${seed}.json" \
      --adapter "$STATE/runs/signflip_seed${seed}/final"
    compare_args+=(--control-jsonl "$output")
  done
  if [[ -e "$STATE/evidence/sealed_comparison.json" ]]; then
    echo "refusing to overwrite V6 sealed comparison"
    return 2
  fi
  local analysis_args=(
    "${compare_args[@]}"
    --bootstrap-samples 10000
    --seed 0
    --confidence 0.95
    --output "$STATE/evidence/sealed_comparison.json"
  )
  "$PYTHON" - "$STATE/evidence/analysis_process.json" "${analysis_args[@]}" <<'PY'
import json
import os
import sys
from pathlib import Path

from jlens_rl.common import runtime_environment_snapshot

output = Path(sys.argv[1])
payload = {
    "python_executable": sys.executable,
    "command": [sys.executable, "-m", "jlens_rl.paired_eval", *sys.argv[2:]],
    "cwd": str(Path.cwd().resolve()),
    "environment_snapshot": runtime_environment_snapshot(),
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
  "$PYTHON" -m jlens_rl.paired_eval "${analysis_args[@]}"
  "${PROTOCOL[@]}" report
}

case "${1:-}" in
  registration-template)
    "${PROTOCOL[@]}" registration-template
    ;;
  recipe-lock-template)
    "${PROTOCOL[@]}" recipe-lock-template
    ;;
  prepare)
    "${PROTOCOL[@]}" prepare
    ;;
  verify)
    "${PROTOCOL[@]}" verify
    ;;
  train-semantic)
    train_condition jlens
    ;;
  curve)
    "${PROTOCOL[@]}" curve
    ;;
  train-controls)
    train_condition signflip
    ;;
  unlock)
    "${PROTOCOL[@]}" unlock
    ;;
  final-evaluation)
    final_evaluation
    ;;
  report)
    "${PROTOCOL[@]}" report
    ;;
  modal)
    source "${MODAL_CREDENTIALS:-modal.sh}"
    .venv/bin/modal run --detach modal_confirmatory_v6.py
    ;;
  *)
    usage
    exit 2
    ;;
esac
