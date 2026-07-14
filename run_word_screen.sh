#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="${repo_dir}/.venv/bin:${PATH}"
set -a
source "${repo_dir}/modal.sh" >/dev/null
set +a
exec modal run --detach "${repo_dir}/modal_word_explore.py"
