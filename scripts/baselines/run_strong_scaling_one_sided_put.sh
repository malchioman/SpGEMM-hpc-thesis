#!/usr/bin/env bash
set -euo pipefail

BINARY="${BINARY:-./build/spgemm_one_sided_put}" \
RESULTS="${RESULTS:-results/one_sided_put/benchmarks_v2.tsv}" \
  exec bash "$(dirname "$0")/run_strong_scaling.sh" "$@"
