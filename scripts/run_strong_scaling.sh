#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash scripts/run_strong_scaling.sh <matrix-a.mtx> [matrix-b.mtx]" >&2
  exit 2
fi

matrix_a="$1"
matrix_b="${2:-}"
binary="${BINARY:-./build/spgemm_two_sided}"
launcher="${MPI_LAUNCHER:-mpirun}"
ranks_list="${RANKS:-1 2 4 8}"
threads="${THREADS:-1}"
schedule="${SCHEDULE:-guided}"
chunk="${CHUNK:-64}"
warmup="${WARMUP:-2}"
repeats="${REPEATS:-10}"
trials="${TRIALS:-5}"
results="${RESULTS:-results/two_sided/benchmarks_v2.tsv}"

for ranks in $ranks_list; do
  command=(
    "$launcher" -np "$ranks" "$binary"
    --matrix-a "$matrix_a"
    --threads "$threads"
    --schedule "$schedule"
    --chunk "$chunk"
    --warmup "$warmup"
    --repeats "$repeats"
    --trials "$trials"
    --experiment strong_scaling
    --results "$results"
  )
  if [[ -n "$matrix_b" ]]; then
    command+=(--matrix-b "$matrix_b")
  fi
  "${command[@]}"
done
