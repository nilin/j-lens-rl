#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-.venv/bin/python}"
TRAIN="${TRAIN:-.venv/bin/train-jlens-rl}"
EVAL="${EVAL:-.venv/bin/eval-jlens-rl}"
COMPARE="${COMPARE:-.venv/bin/compare-jlens-evals}"
PROTOCOL=("$PYTHON" scripts/confirmatory_protocol.py)
SEEDS=(142 143 144 145 146 147)

usage() {
  echo "usage: $0 {prepare|verify|train-semantic|train-controls|train-positive-control|train-all|curve|unlock|final-treatment|final-controls|report}"
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
  load_wandb_key
  for seed in "${SEEDS[@]}"; do
    config="configs/confirmatory_${condition}_seed${seed}.json"
    "$TRAIN" --config "$config" --wandb-mode online
  done
}

eval_if_missing() {
  local output="$1"
  shift
  if [[ -e "$output" ]]; then
    "${PROTOCOL[@]}" verify-eval --path "$output" >/dev/null
    echo "reusing complete, verified evaluation: $output"
    return 0
  fi
  "$EVAL" "$@" --batch-size 64 --output-jsonl "$output" --skip-jlens-metric
}

final_treatment() {
  local seed output
  "${PROTOCOL[@]}" verify-unlock >/dev/null
  mkdir -p .confirmatory/evals .confirmatory/evidence
  eval_if_missing .confirmatory/evals/base.jsonl \
    --config configs/confirmatory_sealed_eval.json \
    --experiment-config configs/confirmatory_jlens_seed142.json \
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
  if [[ ! -e .confirmatory/evidence/semantic_vs_base.json ]]; then
    "$COMPARE" "${compare_args[@]}" \
      --output .confirmatory/evidence/semantic_vs_base.json
  fi
}

final_controls() {
  local condition seed output
  "${PROTOCOL[@]}" verify-unlock >/dev/null
  [[ -f .confirmatory/evals/base.jsonl ]] || {
    echo "run final-treatment first"
    return 2
  }
  for condition in signflip; do
    local compare_args=(--base-jsonl .confirmatory/evals/base.jsonl)
    for seed in "${SEEDS[@]}"; do
      [[ -f ".confirmatory/evals/jlens_seed${seed}.jsonl" ]] || {
        echo "missing semantic evaluation for seed $seed; run final-treatment first"
        return 2
      }
      compare_args+=(--adapter-jsonl ".confirmatory/evals/jlens_seed${seed}.jsonl")
    done
    for seed in "${SEEDS[@]}"; do
      output=".confirmatory/evals/${condition}_seed${seed}.jsonl"
      eval_if_missing "$output" \
        --config configs/confirmatory_sealed_eval.json \
        --experiment-config "configs/confirmatory_${condition}_seed${seed}.json" \
        --indices-manifest .confirmatory/manifests/sealed_final_indices.json \
        --adapter ".confirmatory/runs/${condition}_seed${seed}/final" \
        --run-label "${condition}_seed${seed}"
      compare_args+=(--control-jsonl "$output")
    done
    if [[ ! -e ".confirmatory/evidence/semantic_vs_${condition}.json" ]]; then
      "$COMPARE" "${compare_args[@]}" \
        --output ".confirmatory/evidence/semantic_vs_${condition}.json"
    fi
  done
  if [[ -f .confirmatory/runs/gsm8k_seed142/final/adapter_model.safetensors ]]; then
    output=".confirmatory/evals/gsm8k_seed142.jsonl"
    eval_if_missing "$output" \
      --config configs/confirmatory_sealed_eval.json \
      --experiment-config configs/confirmatory_gsm8k_seed142.json \
      --indices-manifest .confirmatory/manifests/sealed_final_indices.json \
      --adapter .confirmatory/runs/gsm8k_seed142/final \
      --run-label gsm8k_seed142
    if [[ ! -e .confirmatory/evidence/gsm8k_control_vs_base.json ]]; then
      "$COMPARE" \
        --base-jsonl .confirmatory/evals/base.jsonl \
        --adapter-jsonl "$output" \
        --output .confirmatory/evidence/gsm8k_control_vs_base.json
    fi
  else
    echo "optional exact-match control is absent; semantic/sign-flip evidence is still complete"
  fi
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
  train-positive-control)
    "${PROTOCOL[@]}" verify >/dev/null
    load_wandb_key
    "$TRAIN" --config configs/confirmatory_gsm8k_seed142.json --wandb-mode online
    ;;
  train-all)
    train_condition jlens
    train_condition signflip
    ;;
  curve)
    "${PROTOCOL[@]}" curve
    ;;
  unlock)
    "${PROTOCOL[@]}" unlock
    ;;
  final-treatment)
    final_treatment
    ;;
  final-controls)
    final_controls
    ;;
  report)
    "${PROTOCOL[@]}" report
    ;;
  *)
    usage
    exit 2
    ;;
esac
