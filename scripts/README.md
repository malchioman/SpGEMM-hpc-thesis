# Experiment scripts

These Bash scripts target Linux with Open MPI (`mpirun`). Build the executables
with the root CMake project before running experiments. Examples below start in
the repository root.

```text
scripts/
  baselines/
    run_strong_scaling.sh
    run_weak_scaling.sh
    run_strong_scaling_one_sided_get.sh
    run_weak_scaling_one_sided_get.sh
    run_strong_scaling_one_sided_put.sh
    run_weak_scaling_one_sided_put.sh
  trident/
    common.sh
    run_strong_scaling.sh
    run_weak_scaling.sh
    run_local_check.sh
```

## Baselines

The six existing scripts have moved into `baselines/`; their benchmark behavior,
environment variables, executable names, and result paths are unchanged. The
old `scripts/run_*.sh` paths no longer exist. Run these scripts from the repository
root, or override `BINARY` and `RESULTS` explicitly.

```bash
RANKS="1 2 4 8" THREADS=4 \
  bash scripts/baselines/run_strong_scaling.sh matrices/A.mtx matrices/B.mtx
RANKS="1 2 4 8" THREADS=4 \
  bash scripts/baselines/run_weak_scaling_one_sided_get.sh
```

The `RANKS` list controls total MPI ranks; these baseline scripts do not enforce
Trident's hierarchical placement. They retain validation and do not implement
the Trident-specific `VALIDATE` or `DRY_RUN` switches described below.

## Trident: physical-node experiments

`NODES` contains physical node counts, each a perfect square: `1 4 9 16 ...`.
`RANKS_PER_NODE` is fixed across the sweep, as is `THREADS` per rank. Each launch
uses `NODES * RANKS_PER_NODE` MPI ranks. The scripts explicitly map ranks with
`--map-by ppr:<ranks-per-node>:node:PE=<cpus-per-rank> --bind-to core --nooversubscribe`.
Two-sided and GET use `cpus-per-rank=THREADS`; hybrid uses `THREADS+1` for its service
worker. The process is bound to the resulting CPU set, without separate pinning
of the worker. `THREADS` always means compute threads.
This follows the mapping and binding options in the
[Open MPI mpirun manual](https://docs.open-mpi.org/en/v5.0.x/man-openmpi/man1/mpirun.1.html).

Start inside a cluster allocation large enough for the largest requested node
count, or supply `HOSTFILE` with available hosts and their slots. Each node needs
at least `RANKS_PER_NODE * cpus-per-rank` allocated CPU cores, with enough MPI slots
for its ranks. The scripts do not allocate cluster resources or configure SSH.
The MPI installation, executable, and input paths must be accessible on all
participating nodes. Physical runs never use `--logical-node-size`.

Choose `VARIANT=two_sided` (the default), `VARIANT=hybrid` or `VARIANT=get`.
All variants use the same three scripts; the former hybrid-specific wrappers
have been removed. Hybrid requires an MPI installation supporting
`MPI_THREAD_MULTIPLE`; GET and two-sided require only `MPI_THREAD_FUNNELED`.
One invocation runs one selected variant, not all variants automatically.

```bash
VARIANT=hybrid NODES="1 4" RANKS_PER_NODE=2 THREADS=4 VALIDATE=0 \
  bash scripts/trident/run_strong_scaling.sh matrices/A.mtx matrices/B.mtx
VARIANT=hybrid NODES="1 4" RANKS_PER_NODE=2 THREADS=4 VALIDATE=0 \
  bash scripts/trident/run_weak_scaling.sh
```

At equal `THREADS`, hybrid requests more cores than two-sided. For equal total
cores per rank, use one fewer compute thread for hybrid and report that choice.
See [hybrid measurements](../docs/trident_hybrid.md#measurements-and-limits).
GET and two-sided use the same core count at equal `THREADS`. GET includes its
product-boundary synchronization in timed regions; see
[GET measurements](../docs/trident_get.md#measurements-and-limits).

```bash
VARIANT=get NODES="1 4" RANKS_PER_NODE=2 THREADS=4 VALIDATE=0 \
  bash scripts/trident/run_strong_scaling.sh matrices/A.mtx matrices/B.mtx
VARIANT=get NODES="1 4" RANKS_PER_NODE=2 THREADS=4 VALIDATE=0 \
  bash scripts/trident/run_weak_scaling.sh
```

### Strong scaling

The matrices remain fixed while the number of nodes increases. One matrix
computes `A * A` and must be square; two matrices compute `A * B` and must satisfy
`A.cols == B.rows`.

```bash
NODES="1 4 9" RANKS_PER_NODE=2 THREADS=4 \
  bash scripts/trident/run_strong_scaling.sh matrices/A.mtx matrices/B.mtx

NODES="1 4" RANKS_PER_NODE=2 THREADS=4 HOSTFILE=hosts.txt VALIDATE=0 \
  bash scripts/trident/run_strong_scaling.sh matrices/A.mtx
```

### Weak scaling

The synthetic input family matches the baseline scripts. For total rank count
`P`, `A` has shape `(ROWS_PER_RANK * P, COLS_PER_RANK * P)` and `B` has shape
`(COLS_PER_RANK * P, B_COLS_PER_RANK * P)`. The nonzeros per input row stay fixed.
This keeps the average number of scalar products per rank constant, not
necessarily each rank's workload, communication volume, or replicated memory.
The 2D partitioning and matrix sparsity can still cause imbalance.

```bash
NODES="1 4 9" RANKS_PER_NODE=2 THREADS=4 VALIDATE=0 \
  ROWS_PER_RANK=4096 COLS_PER_RANK=4096 B_COLS_PER_RANK=4096 \
  NNZ_PER_ROW=16 B_NNZ_PER_ROW=16 \
  bash scripts/trident/run_weak_scaling.sh
```

The whole sweep is checked before the first launch for invalid dimensions,
nonzero counts, and known signed-int limits of the synthetic input generator.
These checks do not guarantee that the inputs or result fit in memory.

## Local correctness check

```bash
bash scripts/trident/run_local_check.sh
VARIANT=hybrid bash scripts/trident/run_local_check.sh
VARIANT=get bash scripts/trident/run_local_check.sh
```

This runs small rectangular synthetic products on localhost, by default with
one and four **logical** nodes and two ranks per logical node. It passes
`--logical-node-size`, permits oversubscription, disables CPU binding, and
requires validation (`VALIDATE=0` is rejected). It does not use a hostfile.
The executable records `topology=logical_test`. This is a functional check, not
a measurement of inter-node network performance or scaling.

## Trident configuration

Set environment variables before the script invocation:

| Variable | Default | Meaning |
| --- | --- | --- |
| `VARIANT` | `two_sided` | Backend: `two_sided`, `hybrid` or `get` |
| `BINARY` | `<repo>/build/trident_<VARIANT>` | Override executable path; must match the selected backend |
| `MPI_LAUNCHER` | `mpirun` | Open MPI launcher executable, not a command containing extra flags |
| `NODES` | `1 4` | Space-separated square node counts; logical nodes only in the local check |
| `RANKS_PER_NODE` | `2` | MPI ranks per physical or logical node |
| `THREADS` | `1` | OpenMP threads per rank |
| `HOSTFILE` | unset | Optional Open MPI hostfile for physical runs |
| `SCHEDULE`, `CHUNK` | `guided`, `64` | OpenMP schedule and chunk |
| `WARMUP`, `REPEATS`, `TRIALS` | `2`, `10`, `5` | Benchmark sampling; local check defaults to `0`, `1`, `1` |
| `VALIDATE` | `1` | Set to `0` to pass `--no-validate` in physical experiments |
| `DRY_RUN` | `0` | Set to `1` to print commands without launching MPI or writing results |
| `RESULTS` | see below | Destination TSV, appended by the executable |
| `ROWS_PER_RANK`, `COLS_PER_RANK`, `B_COLS_PER_RANK` | `4096` each | Weak-scaling dimension factors |
| `NNZ_PER_ROW`, `B_NNZ_PER_ROW` | `16` each | Weak-scaling nonzeros per input row |

Default binary and result paths are anchored to the repository, regardless of
the current directory. User-supplied relative paths are relative to the calling
directory. All arguments are passed as Bash arrays, including paths with spaces.
The printed command is shell-escaped. A failed launch stops the sweep and
propagates its exit status; earlier successful TSV rows remain available.
If `VARIANT` is omitted and the basename of `BINARY` is `trident_hybrid` or
`trident_get`, the corresponding variant is inferred.
An explicit variant conflicting with a known executable name is rejected. For
renamed/custom executables, specify `VARIANT` so CPU binding remains correct.

Preview the physical mapping without a built executable or an allocation:

```bash
DRY_RUN=1 NODES="1 4 9" RANKS_PER_NODE=2 THREADS=4 \
  bash scripts/trident/run_strong_scaling.sh tests/data/tiny_symmetric.mtx
```

A dry run still checks numeric options and supplied input/hostfile paths; it
does not check MPI availability, allocation capacity, or Matrix Market contents.

By default, results are kept separate by executable and experiment:

- `results/trident/<binary-name>/strong_scaling_v1.tsv`
- `results/trident/<binary-name>/weak_scaling_v1.tsv`
- `results/tmp/<binary-name>_local_check_v1.tsv` (local checks, ignored by Git)

The experiment labels are `trident_strong_scaling`, `trident_weak_scaling`, and
`trident_local_check`. The executable also records detected topology and its
benchmark protocol. Do not combine Trident TSV rows with the baselines' different
schema/protocol. Keep future variants with different protocols in separate files.
Hybrid defaults therefore write under `results/trident/trident_hybrid/`, with
local checks under `results/tmp/trident_hybrid_local_check_v1.tsv`. Two-sided
result paths and protocol are unchanged.
GET defaults write under `results/trident/trident_get/`, with local checks under
`results/tmp/trident_get_local_check_v1.tsv` and protocol `trident_get_csr_v1`.

Validation is enabled unless explicitly disabled for physical experiments.
`VALIDATE=0` skips the serial reference and comparison, but the current executable
still keeps global inputs on rank 0 and gathers the final result. It does not
remove all memory limits for large matrices. See
[Trident CPU](../docs/trident.md) for timing definitions and algorithmic limits.

## Script tests

On Linux, CTest registers `scaling_scripts_smoke` when Bash is available. It uses
a mock launcher to check argument boundaries, placement, sweep validation,
failure propagation, local-test isolation, and the relocated baseline wrappers.
Run it directly with Bash 4.4 or later:

```bash
bash tests/scaling_scripts_smoke.sh
```

This does not replace the real MPI local check or physical-node tests on the
university cluster.
