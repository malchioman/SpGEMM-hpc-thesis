#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == --help ]]; then
  printf 'Usage: bash scripts/trident/run_weak_scaling.sh\nSee scripts/README.md for configuration.\n'
  exit 0
fi
if (( $# != 0 )); then
  printf 'Usage: bash scripts/trident/run_weak_scaling.sh\n' >&2
  exit 2
fi
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
trident_init weak_scaling
ROWS_PER_RANK="${ROWS_PER_RANK:-4096}"
COLS_PER_RANK="${COLS_PER_RANK:-4096}"
B_COLS_PER_RANK="${B_COLS_PER_RANK:-4096}"
NNZ_PER_ROW="${NNZ_PER_ROW:-16}"
B_NNZ_PER_ROW="${B_NNZ_PER_ROW:-16}"
for name in ROWS_PER_RANK COLS_PER_RANK B_COLS_PER_RANK NNZ_PER_ROW B_NNZ_PER_ROW; do
  trident_integer "$name" "${!name}"
done
rows_list=() cols_list=() b_cols_list=()
for nodes in "${trident_nodes[@]}"; do
  ranks="$(trident_product ranks "$nodes" "$RANKS_PER_NODE")"
  rows="$(trident_product rows "$ROWS_PER_RANK" "$ranks")"
  cols="$(trident_product cols "$COLS_PER_RANK" "$ranks")"
  b_cols="$(trident_product b_cols "$B_COLS_PER_RANK" "$ranks")"
  (( NNZ_PER_ROW <= cols )) || trident_error "NNZ_PER_ROW exceeds A's columns"
  (( B_NNZ_PER_ROW <= b_cols )) || trident_error "B_NNZ_PER_ROW exceeds B's columns"
  trident_product a_nnz "$rows" "$NNZ_PER_ROW" >/dev/null
  trident_product b_nnz "$cols" "$B_NNZ_PER_ROW" >/dev/null
  # The shared synthetic generator forms row*17+entry using signed int arithmetic.
  (( rows - 1 <= (2147483647 - NNZ_PER_ROW + 1) / 17 )) || trident_error "A exceeds the synthetic index limit"
  (( cols - 1 <= (2147483647 - B_NNZ_PER_ROW + 1) / 17 )) || trident_error "B exceeds the synthetic index limit"
  rows_list+=("$rows") cols_list+=("$cols") b_cols_list+=("$b_cols")
done
for index in "${!trident_nodes[@]}"; do
  trident_launch "${trident_nodes[index]}" trident_weak_scaling \
    --rows "${rows_list[index]}" --cols "${cols_list[index]}" --b-cols "${b_cols_list[index]}" \
    --nnz-per-row "$NNZ_PER_ROW" --b-nnz-per-row "$B_NNZ_PER_ROW"
done
