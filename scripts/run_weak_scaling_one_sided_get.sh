#!/usr/bin/env bash
set -euo pipefail

binary="${BINARY:-./build/spmm_one_sided_get}"
launcher="${MPI_LAUNCHER:-mpirun}"
ranks_list="${RANKS:-1 2 4 8}"
threads="${THREADS:-1}"
rows_per_rank="${ROWS_PER_RANK:-4096}"
cols_per_rank="${COLS_PER_RANK:-4096}"
nnz_per_row="${NNZ_PER_ROW:-16}"
dense_cols="${DENSE_COLS:-64}"
schedule="${SCHEDULE:-guided}"
chunk="${CHUNK:-64}"
warmup="${WARMUP:-2}"
repeats="${REPEATS:-10}"
trials="${TRIALS:-5}"
results="${RESULTS:-results/one_sided_get/benchmarks.tsv}"

for ranks in $ranks_list; do
  rows=$((rows_per_rank * ranks))
  cols=$((cols_per_rank * ranks))
  "$launcher" -np "$ranks" "$binary" \
    --rows "$rows" \
    --cols "$cols" \
    --nnz-per-row "$nnz_per_row" \
    --dense-cols "$dense_cols" \
    --threads "$threads" \
    --schedule "$schedule" \
    --chunk "$chunk" \
    --warmup "$warmup" \
    --repeats "$repeats" \
    --trials "$trials" \
    --experiment weak_scaling \
    --results "$results"
done
