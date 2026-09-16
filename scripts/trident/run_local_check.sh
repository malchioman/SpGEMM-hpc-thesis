#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == --help ]]; then
  printf 'Usage: bash scripts/trident/run_local_check.sh\nLocal correctness only; not a network scaling measurement. See scripts/README.md.\n'
  exit 0
fi
if (( $# != 0 )); then
  printf 'Usage: bash scripts/trident/run_local_check.sh\n' >&2
  exit 2
fi
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
WARMUP="${WARMUP:-0}" REPEATS="${REPEATS:-1}" TRIALS="${TRIALS:-1}"
trident_init local_check logical_test
for nodes in "${trident_nodes[@]}"; do
  trident_launch "$nodes" trident_local_check \
    --rows 19 --cols 23 --b-cols 13 --nnz-per-row 3 --b-nnz-per-row 2
done
