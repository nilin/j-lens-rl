#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-.venv/bin/python}"
PROTOCOL=("$PYTHON" scripts/confirmatory_v7_protocol.py)
SEEDS=(184 185 186 187 188 189 190 191)
export JLENS_MODAL_IMAGE_SPEC="${JLENS_MODAL_IMAGE_SPEC:-j-lens-rl-confirmatory-v7-profanity-u5-image-v1-strict-allowlist-hf-auth}"

usage() {
  echo "usage: $0 {registration-template|recipe-lock-template|prepare|verify|curve|unlock|report|modal}"
  echo "GPU training/evaluation is intentionally Modal-only so every GPU entrypoint uses the global Dict lease."
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
  curve)
    "${PROTOCOL[@]}" curve
    ;;
  unlock)
    "${PROTOCOL[@]}" unlock
    ;;
  report)
    "${PROTOCOL[@]}" report
    ;;
  modal)
    source "${MODAL_CREDENTIALS:-modal.sh}"
    .venv/bin/modal run --detach modal_confirmatory_v7.py
    ;;
  train-semantic|train-controls|final-evaluation)
    echo "refusing unleased local GPU execution; use '$0 modal' after all registered gates pass" >&2
    exit 2
    ;;
  *)
    usage
    exit 2
    ;;
esac
