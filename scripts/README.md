# Build, inputs and experiments

Run the shell scripts on Linux/WSL with Python 3.8+, CMake 3.21+, a C++17 compiler,
OpenMP and **Open MPI**. The matrix and analysis tools need only Python's standard
library. Benchmark timing code is unchanged.

## Eight entrypoints

| Script | Purpose | Inputs / outputs |
| --- | --- | --- |
| `run_pilot.sh` | Correctness check of all eight implementations on square, permuted and rectangular products | `cage8`; `test-results/pilot/` |
| `run_strong_scaling.sh` | Fixed `A*A` over selected node counts | All ten main matrices; `results/strong_scaling/` |
| `analyze_phases.sh` | Export recorded phase timings, without new MPI runs | Strong scaling, permutation and rectangular observations; `results/phase_analysis/` |
| `analyze_structure.sh` | Export original-square measurements with input/output densities and average input nnz per row | Strong scaling observations; `results/matrix_structure/` |
| `run_permutation.sh` | Run `(P*A*P^T)^2`, seed 42 | `HV15R`, `dielFilterV3real`, `cage15`, `archaea`; `results/permutation/` |
| `run_rectangular.sh` | Run `A*R` with the prepared deterministic sparse R | All ten main matrices; `results/rectangular/` |
| `run_all.sh` | Execute the six stages: pilot, strong scaling, permutation, rectangular, then both exports | Same files as individual scripts |
| `build.sh` | Configure and compile the CMake project in Release mode | Executables in `build/` by default |

The wrappers share `lib/experiments.py` and [experiments.json](experiments.json), so
launch settings, checks and TSV handling are identical in individual and complete
campaigns. Scientific choices and limitations are in the
[experiment plan](../docs/experiments.md).

## Prepare and run

From the repository root:

```bash
bash scripts/build.sh --jobs 8
python3 scripts/matrices.py fetch --ids cage8
python3 scripts/matrices.py prepare --ids cage8 --seed 42
python3 scripts/matrices.py fetch --tier all
python3 scripts/matrices.py prepare --tier all --seed 42

# Preview without launching MPI or requiring downloaded inputs/binaries:
bash scripts/run_all.sh --dry-run

# Inside an appropriate cluster allocation, after setting experiments.json:
bash scripts/run_all.sh
```

`run_all.sh` does not download inputs, build executables or request a scheduler
allocation. It checks inputs for all selected stages before starting. Load the
target's compiler/MPI modules and obtain the allocation beforehand. CRESCO-8's
modules, queue and available resources still need to be confirmed; the checked-in
resource values are editable examples, not a verified cluster configuration.

Build output stays in the chosen CMake directory, separate from matrix files:

```bash
bash scripts/build.sh --build-dir build-wsl --jobs 4
bash scripts/run_pilot.sh --build-dir build-wsl --nodes 1

# Optional compiler override, passed to CMake after --:
bash scripts/build.sh -- -DCMAKE_CXX_COMPILER=mpicxx
```

Use a single-configuration generator (Unix Makefiles or Ninja). `--build-dir` must
match between build and run commands. Relative build, matrix, result and hostfile
paths are resolved against the repository, even when invoking a script elsewhere.
The `--config` filename is relative to the current working directory. Set `PYTHON`
to choose the interpreter used by the shell wrappers.

## Shared settings

Edit `experiments.json`, or override individual values on the command line:

```bash
bash scripts/run_strong_scaling.sh --nodes 1 4 9 --ranks-per-node 2 --threads 4
bash scripts/run_rectangular.sh --matrices cage12 --runs 3
bash scripts/run_permutation.sh --permutation-matrices HV15R cage15 --seed 42
bash scripts/run_all.sh --config scripts/experiments.json --hostfile hosts.txt
bash scripts/run_all.sh --help
```

| Setting | Checked-in value | Meaning |
| --- | --- | --- |
| `nodes` | `[1, 4]` | Physical-node counts; must be perfect squares for Trident |
| `ranks_per_node` | `2` | Fixed MPI processes on each node |
| `threads` | `4` | OpenMP compute threads per rank |
| `cpus_per_rank` | `null` | Defaults to `threads + 1` for **all** variants, including room for Hybrid's worker |
| `runs` | `3` | Independent MPI launches per input pair, resource configuration and implementation |
| `warmup`, `repeats`, `trials` | `2`, `10`, `5` | Within each launch: warmup and product sampling; not independent process launches |
| `schedule`, `chunk` | `guided`, `64` | Same OpenMP settings across variants |
| `validate` | `false` | Main campaigns skip the serial reference; use `--validate` to enable it |
| `seed` | `42` | Must match prepared permutation manifests |
| `matrices` | `["all"]` | Ten main inputs, or an explicit list of catalogue names |
| `pilot_matrices` | `["cage8"]` | Small correctness inputs; override with `--pilot-matrices` |
| `permutation_matrices` | Four selected names | Only permuted products; original references come from strong scaling |
| `variants` | All eight | Subset accepted via `--variants trident_get trident_get_pipeline`, for example |
| `timeout` | `0` | Per-launch limit in seconds; zero uses no additional timeout |

The pilot always validates, uses one independent launch per case/configuration,
zero warmups, one repeat and one trial. It is excluded from both analysis exports.
Use `--runs` to collect more independent MPI launches without repeating the whole
workflow. For example, `--runs 5 --repeats 10 --trials 5` appends five observations
per input pair, resource configuration and implementation; each observation
summarizes 50 timed products as P90 values. The 50 products share the same MPI
processes and are not 50 independent launches. A later rerun appends more rows;
there is no resume or deduplication.

`--timeout` on experiment scripts limits each complete MPI launch, including
input loading, setup, all products, gathering and any serial validation. It is
not a timeout for one multiplication or for the entire campaign. For example,
`--timeout 7200` allows two hours per launch; `--timeout 0` imposes no script
deadline. Scheduler walltime still limits the job. The 120-second timeout used
for small functional checks should not be copied into the thesis campaign.

`--matrices cage8` also selects cage8 for permutation unless an explicit
`--permutation-matrices` is supplied. The pilot selection remains separate.

Open MPI receives `--map-by ppr:R:node:PE=C --bind-to core --nooversubscribe`, with
R ranks per node and C reserved cores per rank. All variants get the same CPU
budget; the worker is not separately pinned to a dedicated core. The environment
sets `OMP_NUM_THREADS`, `OMP_DYNAMIC=FALSE`, `OMP_PLACES=cores` and
`OMP_PROC_BIND=close`. Request enough physical cores per node for `R*C`.
`--mpi-launcher` selects the Open MPI `mpirun` executable, not a generic `srun`/MPICH
adapter. `mpi_args` in JSON (or repeated `--mpi-arg=TOKEN` on the CLI) can supply
MPI transport options; CLI tokens replace the configured list. Mapping and binding
overrides are rejected. Trident's reported physical topology is checked against
the requested configuration before accepting a result.

### Local functional check

Logical nodes allow testing the hierarchical paths on a workstation:

```bash
bash scripts/run_pilot.sh --build-dir build-wsl --local-check --nodes 1 4 --threads 2

# Small check of the complete collection/export workflow:
bash scripts/run_all.sh --build-dir build-wsl --local-check --nodes 1 \
  --matrices cage8 --threads 2 --runs 1 --warmup 0 --repeats 1 --trials 1
```

This mode enables validation, runs on localhost with oversubscription and no rank
binding, and passes `--logical-node-size` to Trident. Results go under
`test-results/local-check/` by default. A result root inside the repository's
`results/` tree is mirrored under `test-results/`; a custom root outside that
tree is retained. Local checks append `local-check/` to the resulting root.
They record `topology=logical_test` and `cpus_per_rank=NA`; these are functional
checks, not physical strong-scaling measurements. Pass `--local-check` to the
analysis scripts too when exporting these temporary observations.

## Result files and repetition

`results/` is reserved for thesis measurements and analysis exports. Pilot checks
are routed to `test-results/pilot/` by default, including during `run_all.sh`.
For a physical-cluster rehearsal of the entire workflow, pass
`--results-dir test-results/unitn-check` to keep all its stages together outside
the thesis results. Use a fresh check directory for each rehearsal.

Pilot and diagnostic paths inside the repository's `results/` tree are mirrored
under `test-results/`, preserving any subdirectories. Explicit result roots
outside that tree are used directly. Existing files are not moved automatically.

Every experiment directory contains one TSV per selected executable, for example
`results/strong_scaling/trident_get.tsv`. Actual runs append one row per completed
MPI launch, without timestamps, execution IDs or averages. Inputs (`matrix_a`,
`matrix_b`) use paths relative to the matrix directory; permuted names retain the
seed. Rows also record dimensions, actual CSR nnz, resources, sampling settings,
protocol, validation and all existing timing measures.

The collector normalizes the two existing benchmark schemas without changing C++
instrumentation. Baseline `halo_setup_seconds` and Trident `plan_setup_seconds`
remain separate. Unsupported fields are `NA`, including inter/intra-node timings
for the row-distributed baselines. Trident `compute_gflops` is exported as
`compute_gflops_p90`, matching its use of compute P90. First-product time includes
setup; independently computed phase P90s must not be summed or stacked.

Re-running an experiment appends new observations to the same file. Re-running an
analysis **regenerates** its selected implementation files from all accumulated
source observations, preserving identical repetitions without duplicating them.
Analysis scripts export tables; they do not calculate means, speedups or plots.
`analyze_structure.sh` uses actual benchmark nnz (after symmetry expansion and
duplicate handling), not the catalogue's stored-entry counts. Densities and mean
row degree describe only part of matrix structure; cross-matrix interpretation
also uses the application families in the experiment plan.

Permutation files contain the permuted runs only: compare them with the matching
original rows in `strong_scaling/`. Phase exports combine strong, permutation and
rectangular inputs, identifiable by their two filenames. Exports use every source
row for the selected variants, regardless of `--nodes` or `--matrices`; those flags
select **new runs**, not filters that discard previously collected observations.

Input hashes are verified once per input per command, outside the measured region.
An incompatible existing TSV header, incomplete row, bad exit status, unexpected
protocol/topology or failed validation stops the campaign. A failed launch is not
appended; diagnostics remain under `test-results/tmp/` by default, or `tmp/` under
the corresponding check root for a custom result directory. Successful raw intermediate
files are removed after their observation is appended. Completed observations
survive a later failure. A result-directory lock prevents concurrent campaign/export
writers. After a forcibly killed process, remove `.experiments.lock` only after
confirming that no process still uses it. There is no automatic resume/deduplication:
a retry adds new repetitions for the selections it runs.

## Matrix tools

```bash
python3 scripts/matrices.py list --tier all
python3 scripts/matrices.py fetch --tier all
python3 scripts/matrices.py prepare --tier all --seed 42
```

Downloads have no total-duration limit. The downloader's `--timeout` controls
blocking network operations and defaults to 300 seconds. To disable that socket
timeout explicitly, use `python3 scripts/matrices.py fetch --tier all --timeout 0`.
Keeping a finite socket timeout allows fallback to another configured URL when
a server stops responding. Disabling it does not fix HTTP errors or broken URLs.

`all` selects the ten main matrices; `--ids cage8` selects the separate pilot.
Original matrices and synthetic R inputs live in `bin/matrices/`, permutations in
`bin/matrices/permuted/`, and metadata in `bin/matrices/metadata/`. Verified files
are reused. `--root /shared/path/matrices` changes the download/preparation root;
use the corresponding `--matrix-dir /shared/path/matrices` in experiment scripts.
No network access is needed during experiments.

## Checks

```bash
python3 tests/matrix_tools_test.py
python3 tests/experiment_scripts_test.py
ctest --test-dir build -R '^(matrix_tools|experiment_scripts)$' --output-on-failure
```

The offline campaign tests exercise all eight output contracts, repeated appends,
idempotent exports, local-result isolation, configuration, input/schema checks and
failed launch handling. Numerical MPI checks are separate from these fixture tests.
