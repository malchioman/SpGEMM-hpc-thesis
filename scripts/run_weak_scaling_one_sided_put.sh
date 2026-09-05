#!/usr/bin/env bash
set -euo pipefail

BINARY="${BINARY:-./build/spgemm_one_sided_put}" \
RESULTS="${RESULTS:-results/one_sided_put/benchmarks.tsv}" \
  exec bash "$(dirname "$0")/run_weak_scaling.sh" "$@"
