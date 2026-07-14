#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-.venv/bin/python}"
TRAIN="${TRAIN:-.venv/bin/train-jlens-rl}"
EVAL="${EVAL:-.venv/bin/eval-jlens-rl}"
COMPARE="${COMPARE:-.venv/bin/compare-jlens-evals}"
PROTOCOL=("$PYTHON" scripts/confirmatory_protocol.py)
SEEDS=(159 160 161 162 163 164 165 166)

usage() {
  echo "usage: $0 {prepare|verify|train-semantic|train-controls|train-all|curve|unlock|final-evaluation|report}"
}

load_wandb_key() {
  if [[ -z "${WANDB_API_KEY:-}" && -f .env ]]; then
    export WANDB_API_KEY="$(tr -d '\r\n' < .env)"
  fi
}

train_condition() {
  local condition="$1"
  local seed config
  "${PROTOCOL[@]}" verify >/dev/null
  if [[ "$condition" == "signflip" ]]; then
    "${PROTOCOL[@]}" verify-curve >/dev/null
  fi
  load_wandb_key
  for seed in "${SEEDS[@]}"; do
    config="configs/confirmatory_${condition}_seed${seed}.json"
    "$TRAIN" --config "$config" --wandb-mode online
  done
}

eval_if_missing() {
  local output="$1"
  local label
  label="$(basename "$output" .jsonl)"
  shift
  if [[ -e "$output" ]]; then
    "${PROTOCOL[@]}" verify-eval --path "$output" --label "$label" >/dev/null
    echo "reusing complete, verified evaluation: $output"
    return 0
  fi
  "$EVAL" "$@" --batch-size 64 --output-jsonl "$output" --skip-jlens-metric
  "${PROTOCOL[@]}" verify-eval --path "$output" --label "$label" >/dev/null
}

final_evaluation() {
  local seed output
  "${PROTOCOL[@]}" verify-unlock >/dev/null
  mkdir -p .confirmatory/evals .confirmatory/evidence
  eval_if_missing .confirmatory/evals/base.jsonl \
    --config configs/confirmatory_sealed_eval.json \
    --experiment-config configs/confirmatory_jlens_seed159.json \
    --indices-manifest .confirmatory/manifests/sealed_final_indices.json \
    --run-label base
  local compare_args=(--base-jsonl .confirmatory/evals/base.jsonl)
  for seed in "${SEEDS[@]}"; do
    output=".confirmatory/evals/jlens_seed${seed}.jsonl"
    eval_if_missing "$output" \
      --config configs/confirmatory_sealed_eval.json \
      --experiment-config "configs/confirmatory_jlens_seed${seed}.json" \
      --indices-manifest .confirmatory/manifests/sealed_final_indices.json \
      --adapter ".confirmatory/runs/jlens_seed${seed}/final" \
      --run-label "jlens_seed${seed}"
    compare_args+=(--adapter-jsonl "$output")
  done
  for seed in "${SEEDS[@]}"; do
    output=".confirmatory/evals/signflip_seed${seed}.jsonl"
    eval_if_missing "$output" \
      --config configs/confirmatory_sealed_eval.json \
      --experiment-config "configs/confirmatory_signflip_seed${seed}.json" \
      --indices-manifest .confirmatory/manifests/sealed_final_indices.json \
      --adapter ".confirmatory/runs/signflip_seed${seed}/final" \
      --run-label "signflip_seed${seed}"
    compare_args+=(--control-jsonl "$output")
  done
  [[ ! -e .confirmatory/evidence/sealed_comparison.json ]] || {
    echo "refusing to overwrite sealed comparison"
    return 2
  }
  "$COMPARE" "${compare_args[@]}" \
    --output .confirmatory/evidence/sealed_comparison.json
}

case "${1:-}" in
  prepare)
    "${PROTOCOL[@]}" prepare
    ;;
  verify)
    "${PROTOCOL[@]}" verify
    ;;
  train-semantic)
    train_condition jlens
    ;;
  train-controls)
    train_condition signflip
    ;;
  train-all)
    train_condition jlens
    "${PROTOCOL[@]}" curve
    train_condition signflip
    ;;
  curve)
    "${PROTOCOL[@]}" curve
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
  *)
    usage
    exit 2
    ;;
esac
