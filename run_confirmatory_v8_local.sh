#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python="$repo/.venv/bin/python"
cd "$repo"

case "${1:-}" in
  verify-design|prepare|verify|verify-launch|probe-hardware|verify-treatments|curve|verify-curve|verify-runs|unlock|verify-unlock|begin-final|verify-final)
    exec "$python" -m scripts.confirmatory_v8_local_protocol "$@"
    ;;
  probe|run-treatments|run-training)
    exec "$python" -m scripts.confirmatory_v8_local_runner "$@"
    ;;
  *)
    echo "usage: $0 {verify-design|prepare|verify|verify-launch|probe|run-treatments|run-training|verify-treatments|curve|verify-curve|verify-runs|unlock|verify-unlock|begin-final|verify-final}" >&2
    exit 2
    ;;
esac
