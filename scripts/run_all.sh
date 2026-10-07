#!/usr/bin/env bash
set -euo pipefail
# Optional N repeats the whole workflow: run_all.sh 5 (or --campaign-repeats 5).
exec "${PYTHON:-python3}" "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib/experiments.py" all "$@"
