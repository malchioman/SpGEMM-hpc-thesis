#!/usr/bin/env bash
set -euo pipefail

binary="${BINARY:-./build/spgemm_two_sided}"
launcher="${MPI_LAUNCHER:-mpirun}"
ranks_list="${RANKS:-1 2 4 8}"
threads="${THREADS:-1}"
rows_per_rank="${ROWS_PER_RANK:-4096}"
cols_per_rank="${COLS_PER_RANK:-4096}"
b_cols_per_rank="${B_COLS_PER_RANK:-4096}"
nnz_per_row="${NNZ_PER_ROW:-16}"
b_nnz_per_row="${B_NNZ_PER_ROW:-16}"
schedule="${SCHEDULE:-guided}"
chunk="${CHUNK:-64}"
warmup="${WARMUP:-2}"
repeats="${REPEATS:-10}"
trials="${TRIALS:-5}"
results="${RESULTS:-results/two_sided/benchmarks_v2.tsv}"

for ranks in $ranks_list; do
  rows=$((rows_per_rank * ranks))
  cols=$((cols_per_rank * ranks))
  b_cols=$((b_cols_per_rank * ranks))
  "$launcher" -np "$ranks" "$binary" \
    --rows "$rows" \
    --cols "$cols" \
    --b-cols "$b_cols" \
    --nnz-per-row "$nnz_per_row" \
    --b-nnz-per-row "$b_nnz_per_row" \
    --threads "$threads" \
    --schedule "$schedule" \
    --chunk "$chunk" \
    --warmup "$warmup" \
    --repeats "$repeats" \
    --trials "$trials" \
    --experiment weak_scaling \
    --results "$results"
done
