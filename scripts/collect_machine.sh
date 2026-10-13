#!/usr/bin/env bash
# Run once per physical compute node, outside the benchmark timers.
set -euo pipefail
if [[ ${1:-} == --help || $# == 0 || $# -gt 2 ]]; then
    echo "Usage: bash scripts/collect_machine.sh OUTPUT_DIR [BUILD_DIR]"
    echo "Writes a separate report per invocation; missing optional tools are recorded."
    [[ ${1:-} == --help ]] && exit 0
    exit 2
fi
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${2:-$repo/build}"
if [[ "$build_dir" != /* ]]; then build_dir="$repo/$build_dir"; fi
mkdir -p -- "$1"
host="$(hostname -s)"
host="${host//[^a-zA-Z0-9_.-]/_}"
report="$(mktemp "$1/${host}.XXXXXX.txt")"

probe() {
    printf '\n$'
    printf ' %q' "$@"
    printf '\n'
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "UNAVAILABLE: $1"
    elif "$@"; then
        :
    else
        echo "COMMAND_FAILED: exit=$? (report continues)"
    fi
}

{
    echo "SpGEMM machine report (read-only probes; no performance measurements)"
    probe date -u +%Y-%m-%dT%H:%M:%SZ
    probe hostname -f
    probe uname -a
    probe cat /etc/os-release
    probe lscpu
    probe lscpu -e=CPU,NODE,SOCKET,CORE,ONLINE
    probe numactl --hardware
    probe free -b
    probe cat /proc/meminfo
    probe taskset -pc "$$"
    probe df -hT "$repo" "$build_dir"
    for path in /sys/devices/system/cpu/cpu0/cpufreq/scaling_driver \
                /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor \
                /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq \
                /sys/devices/system/cpu/intel_pstate/no_turbo \
                /sys/devices/system/cpu/cpufreq/boost; do
        if [[ -r "$path" ]]; then probe cat "$path"; fi
    done
    probe ibstat
    probe ibv_devinfo
    probe ucx_info -v
    probe module -t list
    probe cmake --version
    probe python3 --version
    probe mpicxx --showme
    probe mpicxx --version
    probe mpirun --version
    probe ompi_info --all
    echo
    echo "Selected allocation/runtime variables (not a full environment dump):"
    for key in SLURM_JOB_ID SLURM_JOB_PARTITION SLURM_JOB_NODELIST \
               SLURM_JOB_NUM_NODES SLURM_NTASKS SLURM_TASKS_PER_NODE \
               SLURM_CPUS_PER_TASK SLURM_CPUS_ON_NODE SLURM_MEM_PER_NODE \
               SLURM_CPU_BIND SLURM_HINT PBS_JOBID PBS_NODEFILE LOADEDMODULES \
               OMP_NUM_THREADS OMP_DYNAMIC OMP_PLACES OMP_PROC_BIND \
               OMPI_MCA_pml OMPI_MCA_btl OMPI_MCA_osc \
               UCX_TLS UCX_NET_DEVICES FI_PROVIDER; do
        printf '%s=%s\n' "$key" "${!key-UNSET}"
    done
    echo "The runner sets its own OMP variables; see the frozen experiment configuration."
    if [[ -n ${SLURM_JOB_ID:-} ]]; then
        probe scontrol show job "$SLURM_JOB_ID"
    fi
    probe git -C "$repo" rev-parse HEAD
    probe git -C "$repo" status --short
    probe git -C "$repo" diff --no-ext-diff HEAD -- src scripts CMakeLists.txt
    probe cat "$build_dir/CMakeCache.txt"
    for variant in spgemm_two_sided spgemm_one_sided_get spgemm_one_sided_put \
                   trident_two_sided trident_hybrid trident_get trident_get_pipeline trident_put; do
        probe sha256sum "$build_dir/$variant"
        if [[ -f "$build_dir/$variant" ]]; then probe ldd "$build_dir/$variant"; fi
    done
} >"$report" 2>&1
printf 'Machine report: %s\n' "$report"
