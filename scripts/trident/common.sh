#!/usr/bin/env bash
# Shared launch configuration for the Trident experiment scripts; source this file.

trident_error() {
  printf 'Trident scripts: %s\n' "$*" >&2
  exit 2
}

trident_integer() {
  local name="$1" value="$2" minimum="${3:-1}"
  [[ "$value" =~ ^(0|[1-9][0-9]*)$ ]] || trident_error "$name must be a decimal integer"
  (( ${#value} <= 10 )) || trident_error "$name exceeds the signed-int limit"
  (( value >= minimum && value <= 2147483647 )) || trident_error "$name must be in [$minimum, 2147483647]"
}

trident_product() {
  local name="$1" left="$2" right="$3"
  (( left <= 2147483647 / right )) || trident_error "$name exceeds the signed-int limit"
  printf '%s\n' "$((left * right))"
}

trident_init() {
  local mode="$1" name nodes side ranks
  trident_topology="${2:-physical}"
  trident_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
  BINARY="${BINARY:-}"
  if [[ -z "${VARIANT:-}" ]]; then
    case "${BINARY##*/}" in
      trident_hybrid) VARIANT=hybrid ;;
      trident_get) VARIANT=get ;;
      trident_get_pipeline) VARIANT=get_pipeline ;;
      trident_put) VARIANT=put ;;
    esac
  fi
  VARIANT="${VARIANT:-two_sided}"
  case "$VARIANT" in
    two_sided|get|get_pipeline|put) trident_service_threads=0 ;;
    hybrid) trident_service_threads=1 ;;
    *) trident_error "VARIANT must be two_sided, hybrid, get, get_pipeline or put" ;;
  esac
  BINARY="${BINARY:-$trident_repo/build/trident_$VARIANT}"
  case "${BINARY##*/}" in
    trident_two_sided|trident_hybrid|trident_get|trident_get_pipeline|trident_put)
      [[ "${BINARY##*/}" == "trident_$VARIANT" ]] || trident_error "BINARY and VARIANT disagree" ;;
  esac
  MPI_LAUNCHER="${MPI_LAUNCHER:-mpirun}"
  NODES="${NODES:-1 4}"
  RANKS_PER_NODE="${RANKS_PER_NODE:-2}"
  THREADS="${THREADS:-1}"
  SCHEDULE="${SCHEDULE:-guided}"
  CHUNK="${CHUNK:-64}"
  WARMUP="${WARMUP:-2}"
  REPEATS="${REPEATS:-10}"
  TRIALS="${TRIALS:-5}"
  VALIDATE="${VALIDATE:-1}"
  DRY_RUN="${DRY_RUN:-0}"
  HOSTFILE="${HOSTFILE:-}"
  if [[ "$trident_topology" == logical_test ]]; then
    RESULTS="${RESULTS:-$trident_repo/results/tmp/${BINARY##*/}_${mode}_v1.tsv}"
    [[ "$VALIDATE" == 1 ]] || trident_error "the local correctness check requires VALIDATE=1"
    [[ -z "$HOSTFILE" ]] || trident_error "HOSTFILE is not used by the localhost-only check"
  else
    RESULTS="${RESULTS:-$trident_repo/results/trident/${BINARY##*/}/${mode}_v1.tsv}"
  fi
  for name in RANKS_PER_NODE THREADS CHUNK REPEATS TRIALS; do
    trident_integer "$name" "${!name}"
  done
  trident_integer WARMUP "$WARMUP" 0
  trident_cpus_per_rank=$((THREADS + trident_service_threads))
  trident_integer CPUS_PER_RANK "$trident_cpus_per_rank"
  [[ "$VALIDATE" == 0 || "$VALIDATE" == 1 ]] || trident_error "VALIDATE must be 0 or 1"
  [[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || trident_error "DRY_RUN must be 0 or 1"
  case "$SCHEDULE" in
    static|dynamic|guided|auto) ;;
    *) trident_error "SCHEDULE must be static, dynamic, guided or auto" ;;
  esac
  read -r -a trident_nodes <<< "${NODES//$'\n'/ }"
  (( ${#trident_nodes[@]} > 0 )) || trident_error "NODES must contain at least one node count"
  # Validate the entire sweep before starting any MPI job.
  for nodes in "${trident_nodes[@]}"; do
    trident_integer NODES "$nodes"
    side=1
    while (( side * side < nodes )); do ((side += 1)); done
    (( side * side == nodes )) || trident_error "NODES entries must be perfect squares (1, 4, 9, 16, ...)"
    ranks="$(trident_product ranks "$nodes" "$RANKS_PER_NODE")"
  done
  if [[ -n "$HOSTFILE" ]]; then
    [[ -r "$HOSTFILE" && -f "$HOSTFILE" ]] || trident_error "cannot read HOSTFILE: $HOSTFILE"
  fi
  if [[ "$DRY_RUN" == 0 ]]; then
    [[ -x "$BINARY" && -f "$BINARY" ]] || trident_error "BINARY is not executable: $BINARY"
    command -v "$MPI_LAUNCHER" >/dev/null || trident_error "MPI_LAUNCHER is not available: $MPI_LAUNCHER"
  fi
  trident_benchmark_args=(--threads "$THREADS" --schedule "$SCHEDULE" --chunk "$CHUNK"
    --warmup "$WARMUP" --repeats "$REPEATS" --trials "$TRIALS" --results "$RESULTS")
  if [[ "$VALIDATE" == 0 ]]; then trident_benchmark_args+=(--no-validate); fi
}

trident_launch() {
  local nodes="$1" experiment="$2" ranks
  shift 2
  ranks="$(trident_product ranks "$nodes" "$RANKS_PER_NODE")"
  local command=("$MPI_LAUNCHER" -np "$ranks")
  if [[ "$trident_topology" == logical_test ]]; then
    command+=(--host localhost --oversubscribe --bind-to none)
  else
    command+=(--map-by "ppr:$RANKS_PER_NODE:node:PE=$trident_cpus_per_rank" --bind-to core --nooversubscribe)
    if [[ -n "$HOSTFILE" ]]; then command+=(--hostfile "$HOSTFILE"); fi
  fi
  command+=("$BINARY" "$@" "${trident_benchmark_args[@]}" --experiment "$experiment")
  if [[ "$trident_topology" == logical_test ]]; then
    command+=(--logical-node-size "$RANKS_PER_NODE")
  fi
  printf '# topology=%s nodes=%s ranks_per_node=%s ranks=%s threads=%s variant=%s service_threads=%s\n' \
    "$trident_topology" "$nodes" "$RANKS_PER_NODE" "$ranks" "$THREADS" "$VARIANT" "$trident_service_threads"
  printf '%q ' "${command[@]}"
  printf '\n'
  if [[ "$DRY_RUN" == 0 ]]; then "${command[@]}"; fi
}
