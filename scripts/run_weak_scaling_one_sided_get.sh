#!/usr/bin/env bash
set -euo pipefail

BINARY="${BINARY:-./build/spgemm_one_sided_get}" \
RESULTS="${RESULTS:-results/one_sided_get/benchmarks_v2.tsv}" \
  exec bash "$(dirname "$0")/run_weak_scaling.sh" "$@"
