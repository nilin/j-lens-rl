#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="${repo_dir}/.venv/bin:${PATH}"

command="${1:-}"
case "${command}" in
  prepare|verify|verify-launch|summarize|finalize-evidence)
    exec python "${repo_dir}/scripts/emotional_tournament_v1_protocol.py" "${command}"
    ;;
  modal)
    set -a
    source "${repo_dir}/modal.sh" >/dev/null
    set +a
    exec modal run --detach "${repo_dir}/modal_emotional_tournament_v1.py"
    ;;
  *)
    echo "usage: $0 {prepare|verify|verify-launch|summarize|finalize-evidence|modal}" >&2
    exit 2
    ;;
esac
