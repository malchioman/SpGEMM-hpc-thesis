#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/run_strong_scaling_one_sided_put.sh <matrix.mtx>" >&2
  exit 2
fi

matrix="$1"
binary="${BINARY:-./build/spmm_one_sided_put}"
launcher="${MPI_LAUNCHER:-mpirun}"
ranks_list="${RANKS:-1 2 4 8}"
threads="${THREADS:-1}"
dense_cols="${DENSE_COLS:-64}"
schedule="${SCHEDULE:-guided}"
chunk="${CHUNK:-64}"
warmup="${WARMUP:-2}"
repeats="${REPEATS:-10}"
trials="${TRIALS:-5}"
results="${RESULTS:-results/one_sided_put/benchmarks.tsv}"

for ranks in $ranks_list; do
  "$launcher" -np "$ranks" "$binary" \
    --matrix "$matrix" \
    --dense-cols "$dense_cols" \
    --threads "$threads" \
    --schedule "$schedule" \
    --chunk "$chunk" \
    --warmup "$warmup" \
    --repeats "$repeats" \
    --trials "$trials" \
    --experiment strong_scaling \
    --results "$results"
done
