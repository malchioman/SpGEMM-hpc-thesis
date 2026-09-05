#!/usr/bin/env bash
set -euo pipefail

BINARY="${BINARY:-./build/spgemm_one_sided_get}" \
RESULTS="${RESULTS:-results/one_sided_get/benchmarks.tsv}" \
  exec bash "$(dirname "$0")/run_strong_scaling.sh" "$@"
