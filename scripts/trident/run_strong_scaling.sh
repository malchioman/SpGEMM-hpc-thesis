#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == --help ]]; then
  printf 'Usage: bash scripts/trident/run_strong_scaling.sh <matrix-a.mtx> [matrix-b.mtx]\nSee scripts/README.md for configuration.\n'
  exit 0
fi
if (( $# < 1 || $# > 2 )); then
  printf 'Usage: bash scripts/trident/run_strong_scaling.sh <matrix-a.mtx> [matrix-b.mtx]\n' >&2
  exit 2
fi
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
trident_init strong_scaling
for matrix in "$@"; do
  [[ -f "$matrix" && -r "$matrix" ]] || trident_error "cannot read matrix: $matrix"
done
input_args=(--matrix-a "$1")
if (( $# == 2 )); then input_args+=(--matrix-b "$2"); fi
for nodes in "${trident_nodes[@]}"; do
  trident_launch "$nodes" trident_strong_scaling "${input_args[@]}"
done
