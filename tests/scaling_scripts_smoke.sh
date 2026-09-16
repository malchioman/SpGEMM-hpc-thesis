#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
scratch="$(mktemp -d)"
trap 'rm -rf -- "$scratch"' EXIT
mkdir "$scratch/paths with spaces"
launcher="$scratch/paths with spaces/mock mpirun"
binary="$scratch/paths with spaces/trident binary"
cp "$repo/tests/fixtures/mock_mpirun.sh" "$launcher"
cp "$launcher" "$binary"
chmod +x "$launcher" "$binary"
matrix_a="$scratch/paths with spaces/A.mtx"
matrix_b="$scratch/paths with spaces/B.mtx"
hostfile="$scratch/paths with spaces/hosts.txt"
cp "$repo/tests/data/rectangular_a.mtx" "$matrix_a"
cp "$repo/tests/data/rectangular_b.mtx" "$matrix_b"
touch "$hostfile"
inputs=()
checks=0

fail() {
  printf 'FAIL: %s\n' "$*" >&2
  exit 1
}

run_case() {
  local expected="$1" script="$2" status=0
  shift 2
  calls="$(mktemp -d "$scratch/calls.XXXXXX")"
  env -i PATH="$PATH" HOME="${HOME:-$scratch}" \
    MOCK_LOG_DIR="$calls" MPI_LAUNCHER="$launcher" BINARY="$binary" "$@" \
    bash "$repo/scripts/$script" "${inputs[@]}" > "$scratch/stdout" 2> "$scratch/stderr" || status=$?
  if (( status != expected )); then
    cat "$scratch/stdout" "$scratch/stderr" >&2
    fail "$script: expected exit $expected, got $status"
  fi
  ((checks += 1))
}

expect_runs() {
  local files=("$calls"/call.*)
  (( ${#files[@]} == $1 )) || fail "expected $1 launches, got ${#files[@]}"
}

read_call() {
  # NUL-separated records preserve argument boundaries, including spaces.
  mapfile -d '' -t args < "$calls/call.$1"
}

expect_arg() {
  local arg
  for arg in "${args[@]}"; do
    if [[ "$arg" == "$1" ]]; then return 0; fi
  done
  fail "missing argument: $1"
}

expect_option() {
  local index
  for ((index=0; index+1<${#args[@]}; index+=1)); do
    if [[ "${args[index]}" == "$1" && "${args[index+1]}" == "$2" ]]; then return 0; fi
  done
  fail "missing option/value: $1 $2"
}

reject_arg() {
  local arg
  for arg in "${args[@]}"; do
    [[ "$arg" != "$1" ]] || fail "unexpected argument: $1"
  done
}

# Check that scripts locate their helpers and default paths outside the repo cwd.
cd "$scratch"
for script in "$repo"/scripts/{baselines,trident}/*.sh "$repo"/tests/fixtures/*.sh; do
  bash -n "$script"
  if LC_ALL=C grep -q $'\r' "$script"; then fail "CRLF shell script: $script"; fi
done

inputs=("$repo/tests/data/tiny_symmetric.mtx")
run_case 0 trident/run_strong_scaling.sh
expect_runs 2
for call in 0 1; do
  read_call "$call"
  expect_arg "$binary"
  expect_option --matrix-a "${inputs[0]}"
  expect_option --map-by ppr:2:node:PE=1
  expect_option --bind-to core
  expect_arg --nooversubscribe
  expect_option --results "$repo/results/trident/trident binary/strong_scaling_v1.tsv"
  expect_option --experiment trident_strong_scaling
  expect_option --warmup 2
  expect_option --repeats 10
  expect_option --trials 5
  reject_arg --matrix-b
  reject_arg --logical-node-size
  reject_arg --no-validate
  if (( call == 0 )); then expect_option -np 2; else expect_option -np 8; fi
done

inputs=("$matrix_a" "$matrix_b")
run_case 0 trident/run_strong_scaling.sh NODES="1 4" RANKS_PER_NODE=3 THREADS=4 \
  HOSTFILE="$hostfile" RESULTS="$scratch/paths with spaces/results.tsv" \
  SCHEDULE=dynamic CHUNK=2 WARMUP=0 REPEATS=3 TRIALS=2 VALIDATE=0
expect_runs 2
read_call 1
expect_option -np 12
expect_option --map-by ppr:3:node:PE=4
expect_option --threads 4
expect_option --hostfile "$hostfile"
expect_option --matrix-a "$matrix_a"
expect_option --matrix-b "$matrix_b"
expect_option --results "$scratch/paths with spaces/results.tsv"
expect_option --schedule dynamic
expect_option --chunk 2
expect_option --warmup 0
expect_option --repeats 3
expect_option --trials 2
expect_arg --no-validate

inputs=()
run_case 0 trident/run_weak_scaling.sh NODES="1 4" ROWS_PER_RANK=3 \
  COLS_PER_RANK=5 B_COLS_PER_RANK=7 NNZ_PER_ROW=2 B_NNZ_PER_ROW=3
expect_runs 2
for call in 0 1; do
  read_call "$call"
  if (( call == 0 )); then ranks=2; else ranks=8; fi
  expect_option -np "$ranks"
  expect_option --rows "$((3 * ranks))"
  expect_option --cols "$((5 * ranks))"
  expect_option --b-cols "$((7 * ranks))"
  expect_option --nnz-per-row 2
  expect_option --b-nnz-per-row 3
  expect_option --results "$repo/results/trident/trident binary/weak_scaling_v1.tsv"
  expect_option --experiment trident_weak_scaling
  reject_arg --logical-node-size
done

run_case 0 trident/run_local_check.sh
expect_runs 2
read_call 1
expect_option -np 8
expect_option --logical-node-size 2
expect_option --host localhost
expect_option --bind-to none
expect_arg --oversubscribe
expect_option --rows 19
expect_option --cols 23
expect_option --b-cols 13
expect_option --warmup 0
expect_option --repeats 1
expect_option --trials 1
expect_option --results "$repo/results/tmp/trident binary_local_check_v1.tsv"
expect_option --experiment trident_local_check
reject_arg --map-by
reject_arg --no-validate

for setting in 'NODES=1 2' 'NODES= ' 'NODES=0' 'NODES=01' 'NODES=2147483648' \
  'RANKS_PER_NODE=0' 'THREADS=-1' 'THREADS=1+1' 'CHUNK=0' 'WARMUP=-1' \
  'REPEATS=0' 'TRIALS=0' 'SCHEDULE=invalid' 'VALIDATE=2' 'DRY_RUN=2' \
  'ROWS_PER_RANK=2147483647' 'ROWS_PER_RANK=100000000' \
  'ROWS_PER_RANK=16000000' 'COLS_PER_RANK=16000000' \
  'NNZ_PER_ROW=999999' 'B_NNZ_PER_ROW=999999' 'NNZ_PER_ROW=0' 'HOSTFILE=/missing'; do
  run_case 2 trident/run_weak_scaling.sh "$setting"
  expect_runs 0
done
run_case 2 trident/run_weak_scaling.sh NODES=4 RANKS_PER_NODE=2147483647
expect_runs 0
run_case 2 trident/run_local_check.sh VALIDATE=0
expect_runs 0
run_case 2 trident/run_local_check.sh HOSTFILE="$hostfile"
expect_runs 0
run_case 2 trident/run_weak_scaling.sh BINARY="$scratch/missing-binary"
expect_runs 0
run_case 2 trident/run_weak_scaling.sh MPI_LAUNCHER="$scratch/missing-launcher"
expect_runs 0
run_case 2 trident/run_strong_scaling.sh
expect_runs 0
inputs=("$scratch/missing-matrix")
run_case 2 trident/run_strong_scaling.sh
expect_runs 0
inputs=("$matrix_a" "$matrix_b" extra)
run_case 2 trident/run_strong_scaling.sh
expect_runs 0
inputs=(unexpected)
run_case 2 trident/run_weak_scaling.sh
expect_runs 0
run_case 2 trident/run_local_check.sh
expect_runs 0

inputs=()
run_case 17 trident/run_weak_scaling.sh MOCK_MPI_EXIT=17
expect_runs 1
run_case 0 trident/run_weak_scaling.sh DRY_RUN=1 BINARY="$scratch/missing-binary" \
  MPI_LAUNCHER="$scratch/missing-launcher" RESULTS="$scratch/not-created/results.tsv"
expect_runs 0
[[ ! -e "$scratch/not-created" ]] || fail 'dry-run created results'
grep -q 'topology=physical nodes=4' "$scratch/stdout" || fail 'dry-run omitted node count'
inputs=("$matrix_a" "$matrix_b")
run_case 0 trident/run_strong_scaling.sh DRY_RUN=1 BINARY=
expect_runs 0
grep -q '/build/trident_two_sided' "$scratch/stdout" || fail 'incorrect default binary'
inputs=(--help)
for script in run_strong_scaling run_weak_scaling run_local_check; do
  run_case 0 "trident/$script.sh"
  expect_runs 0
done

for variant in two_sided one_sided_get one_sided_put; do
  suffix="_$variant"
  if [[ "$variant" == two_sided ]]; then suffix=; fi
  for mode in strong weak; do
    inputs=()
    if [[ "$mode" == strong ]]; then inputs=("$matrix_a" "$matrix_b"); fi
    run_case 0 "baselines/run_${mode}_scaling${suffix}.sh" BINARY= RANKS=1 THREADS=3
    expect_runs 1
    read_call 0
    expect_arg "./build/spgemm_$variant"
    expect_option --results "results/$variant/benchmarks_v2.tsv"
    expect_option -np 1
    expect_option --threads 3
    expect_option --experiment "${mode}_scaling"
    if [[ "$mode" == strong ]]; then
      expect_option --matrix-a "$matrix_a"
      expect_option --matrix-b "$matrix_b"
    else
      expect_option --rows 4096
      expect_option --cols 4096
      expect_option --b-cols 4096
    fi
  done
done

printf 'PASS: %s script cases\n' "$checks"
