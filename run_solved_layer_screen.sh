#!/usr/bin/env bash
set -euo pipefail

cat >&2 <<'EOF'
This legacy adaptive layer screen is retired: it monitored GSM8K test examples
and selected runs with correctness-based early stopping. It is not valid
confirmatory evidence.

Use ./run_confirmatory.sh after reading CONFIRMATORY_PROTOCOL.md. The new runner
requires a clean committed tree, fresh source-index manifests, fixed horizons,
matched controls, and a separately sealed final evaluation.
EOF
exit 2
