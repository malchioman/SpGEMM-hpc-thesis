# Distributed Sparse Matrix-Matrix Multiplication (SpGEMM) — MPI + OpenMP

## Introduction

This repository contains the CPU implementations and experiment tools for a thesis
on distributed **Sparse General Matrix-Matrix Multiplication (SpGEMM)**.

**Author:** Massimo Malchiodi, University of Trento

The operation is `C = A * B`, with sparse matrices stored in **Compressed Sparse
Row (CSR)** format. MPI distributes the matrices and communication across
processes; OpenMP parallelizes the local multiplication within each process.
Both square and rectangular products are supported, provided `A.cols == B.rows`.

The project compares two families of implementations:

- **Row-distributed baselines:** three implementations with the same row
  partitioning and local kernel, using two-sided messages, one-sided GET, or
  one-sided PUT to exchange the required rows of B.
- **Trident CPU variants:** five implementations with hierarchical 2D partitioning
  across nodes and 1D row partitioning within each node, exploring different
  inter-node communication protocols and GET prefetching.

The repository also provides Matrix Market input, synthetic inputs, numerical
validation, automated experiments, TSV results, and optional resource orchestration
through SbatchMan for Slurm and OpenPBS/PBS Pro.

> The build and experiment commands below target Linux, including WSL2 and HPC
> clusters. Cluster modules, queues and resource requests must match the target
> system; the checked-in scheduler profiles are examples.

## Contents

- [Repository layout](#repository-layout)
- [Requirements and environment setup](#requirements-and-environment-setup)
- [Build and first run](#build-and-first-run)
- [Matrix inputs](#matrix-inputs)
- [Row-distributed SpGEMM baselines](#row-distributed-spgemm-baselines)
- [Hierarchical Trident CPU variants](#hierarchical-trident-cpu-variants)
- [Command-line options](#command-line-options)
- [Experiments and cluster execution](#experiments-and-cluster-execution)
- [Output and benchmark timings](#output-and-benchmark-timings)
- [Validation and tests](#validation-and-tests)
- [Technical documentation and references](#technical-documentation-and-references)

## Repository layout

```text
SpGEMM-hpc-thesis/
  src/
    common/             CSR storage, Matrix Market I/O and shared benchmark helpers
    baselines/          Row-distributed kernels and two-sided / GET / PUT exchange
    trident/
      common/           Process grid, partitioning, CPU kernel and benchmark driver
      two_sided/        Two-sided inter-node exchange
      hybrid/           RMA requests and two-sided responses
      get/              Direct MPI_Get exchange
      get_pipeline/     MPI_Get with double buffering and next-stage prefetch
      put/              Direct MPI_Put exchange
  scripts/              Build, matrix preparation, experiments and analysis tools
    sbatchman/          Slurm and PBS profiles and scheduler integration guide
  tests/                C++/MPI and Python tests, with small Matrix Market fixtures
  docs/                 Algorithm details, experiment plan and scientific sources
  matrices/             Downloaded/prepared inputs (generated, ignored by Git)
  build/                CMake build and executables (generated, ignored by Git)
  results/              Benchmark measurements and analysis exports
  test-results/         Pilot runs and local checks (generated, ignored by Git)
  CMakeLists.txt
  README.md
```

The eight benchmark executables are produced directly in `build/` when using
these build commands. Test executables and support libraries are built there
as well. Input matrices are stored separately in `matrices/`.

## Requirements and environment setup

| Component | Requirement |
| --- | --- |
| C++ compiler | C++17 with OpenMP support |
| CMake | 3.21 or newer |
| MPI | MPI-3 RMA support and development headers/libraries |
| MPI thread support | `MPI_THREAD_FUNNELED`; `trident_hybrid` requires `MPI_THREAD_MULTIPLE` |
| Shell | Bash for the helper scripts |
| Python | 3.8+ for matrix, experiment and analysis tools; 3.10+ for optional SbatchMan |

The experiment scripts use **Open MPI** mapping and binding options. Matrix
preparation and analysis use Python's standard library. Matrix Market I/O is
implemented in this repository; no external matrix library is needed. The Trident
variants run on CPU and require no CUDA or GPU software stack.

### Linux / Ubuntu / WSL2

On Ubuntu, install the compiler, build tools, Open MPI and Python:

```bash
sudo apt update
sudo apt install -y build-essential cmake git openmpi-bin libopenmpi-dev python3
```

Check the installed tools, including the minimum CMake version:

```bash
cmake --version
mpicxx --version
mpirun --version
python3 --version
```

On Windows, use a Linux environment through WSL2 for this workflow. If WSL is
not installed, run `wsl --install` in an administrator PowerShell, complete the
distribution setup, then run the commands above inside its Linux terminal.

### HPC cluster

Load the site's C++ compiler, Open MPI, CMake and Python modules before building
or running. Use the same compiler/MPI environment for both steps. The repository,
binaries, inputs and result paths must be accessible on all participating nodes.

For scheduled jobs, configure the site's module names, queue/partition, walltime
and memory in a local scheduler profile. The
[SbatchMan guide](scripts/sbatchman/README.md) covers both supported schedulers.

## Build and first run

### Clone and compile

```bash
git clone https://github.com/malchioman/SpGEMM-hpc-thesis.git
cd SpGEMM-hpc-thesis
bash scripts/build.sh --jobs 4
```

The build script configures a Release build and compiles all targets. Use a
single-configuration generator such as Unix Makefiles or Ninja. The equivalent
manual commands are:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 4
```

To use a different build directory or select the MPI compiler wrapper explicitly:

```bash
bash scripts/build.sh --build-dir build-wsl --jobs 4 -- -DCMAKE_CXX_COMPILER=mpicxx
```

Use the same directory in subsequent executable paths, `ctest --test-dir`, and
experiment-script `--build-dir` options. All examples below run from the
repository root and use `build/`.

### First correctness check

This small rectangular product uses files already included in the repository;
no matrix download is needed:

```bash
mpirun -np 2 ./build/spgemm_two_sided \
  --matrix-a tests/data/rectangular_a.mtx \
  --matrix-b tests/data/rectangular_b.mtx \
  --threads 1 --warmup 0 --repeats 1 --trials 1 \
  --results test-results/first_run.tsv
```

Rank 0 prints matrix dimensions, timings and `validation=PASS` on success, and
appends a row to `test-results/first_run.tsv`. The output directory is created
automatically. This run checks the setup; its timings are not scaling results.

## Matrix inputs

### Synthetic matrices

Without matrix-file arguments, the executable generates deterministic sparse
inputs in memory with a fixed number of nonzeros per row. `--rows` and `--cols`
define A; `--cols` and `--b-cols` define B. The nonzero counts per row can be set
separately:

```bash
mpirun -np 2 ./build/spgemm_one_sided_get \
  --rows 128 --cols 96 --b-cols 64 \
  --nnz-per-row 4 --b-nnz-per-row 4 \
  --threads 2 --warmup 1 --repeats 3 --trials 2 \
  --results test-results/synthetic_get.tsv
```

These inputs are useful for functional checks and require no `.mtx` files.

### Matrix Market files

All eight executables accept the same input options:

| Input arguments | Product |
| --- | --- |
| `--matrix A.mtx` | `A * A`; A must be square |
| `--matrix-a A.mtx` | Reuse A as B and compute `A * A`; A must be square |
| `--matrix-a A.mtx --matrix-b B.mtx` | `A * B`; requires `A.cols == B.rows` |

The reader accepts Matrix Market **coordinate** files with `real`, `integer`, or
`pattern` entries. It supports `general`, `symmetric`, `skew-symmetric`, and
`hermitian` storage declarations for these non-complex fields. Symmetry declarations
require square matrices; `general` inputs may be rectangular. Complex-valued input
is not supported. File paths in direct executable commands are relative to the
current working directory unless absolute paths are supplied.

The reader preserves expanded input entries, including unsorted columns,
duplicates and explicit zeros. Input `nnz` metrics count these stored entries;
only product outputs combine equal coordinates and omit zero sums. Malformed
records, trailing undeclared entries and nonzero skew-symmetric diagonals are rejected.

### Download and prepare catalogue inputs

The matrix tool reads [scripts/matrices_catalog.json](scripts/matrices_catalog.json).
Inspect the existing catalogue and prepare the small pilot input with:

```bash
python3 scripts/matrices.py list
python3 scripts/matrices.py fetch --ids cage8
python3 scripts/matrices.py prepare --ids cage8 --seed 42
```

Preparation retains the original matrix and creates a simultaneous row/column
permutation `P*A*P^T`, a deterministic sparse rectangular operand R for `A*R`, and
manifests with file hashes. The resulting layout is:

```text
matrices/
  cage8.mtx
  cage8_restriction.mtx
  permuted/
    cage8_permuted_s42.mtx
  metadata/
    cage8.source.json
    cage8_s42.cases.json
```

Use `--ids` to prepare a chosen subset. `--tier all` selects the ten main entries
in the current catalogue; it excludes the separate pilot/core tiers. Verified
files are reused. For shared storage, pass `--root /shared/path/matrices` to the
matrix tool and `--matrix-dir /shared/path/matrices` to the experiment scripts.

The [experiment plan](docs/experiments.md) records the current matrix proposals
and comparison design. The catalogue and campaign selections can be refined
before the final measurements.

## Row-distributed SpGEMM baselines

Each rank owns contiguous row blocks of A and B and computes the corresponding
rows of C. Before multiplication, it obtains the remote B rows referenced by its
local A entries. OpenMP parallelizes the output rows using a Gustavson-style
accumulator.

| Executable | Exchange of required B rows |
| --- | --- |
| `spgemm_two_sided` | Owners pack rows and exchange payloads with `MPI_Isend` / `MPI_Irecv` |
| `spgemm_one_sided_get` | Consumers fetch payloads from the owners' CSR windows with `MPI_Get` |
| `spgemm_one_sided_put` | Owners push payloads into the consumers' halo windows with `MPI_Put` |

The three baselines use the `prepared_halo_v2` protocol: requests, row lengths,
offsets and buffers are prepared once; column indices and values are transferred
again for every product. Communication timings include the transport's packing
and synchronization work where applicable.

After preparing `cage8`, run all three with the same settings:

```bash
for variant in spgemm_two_sided spgemm_one_sided_get spgemm_one_sided_put; do
  mpirun -np 2 "./build/$variant" \
    --matrix matrices/cage8.mtx --threads 2 --schedule guided \
    --warmup 2 --repeats 10 --trials 5 \
    --results "test-results/baselines/$variant.tsv"
done
```

## Hierarchical Trident CPU variants

Trident uses a square `q × q` grid of nodes, with row slices distributed among
the MPI ranks within each node. The output C stays on its owning ranks while
partial products are accumulated over the stages. All five variants share the
partitioning, local CPU kernel and two-sided intra-node aggregation of B.

| Executable | Inter-node protocol | Execution |
| --- | --- | --- |
| `trident_two_sided` | `MPI_Isend` / `MPI_Irecv` | Complete the exchange for each stage before computing |
| `trident_hybrid` | RMA request slots, two-sided payload responses | A service thread responds to remote requests |
| `trident_get` | `MPI_Get` from input CSR windows | Complete reads for each stage before computing |
| `trident_get_pipeline` | Same GET transport, two A/B buffer pairs | Prefetch the next stage and defer completion until needed |
| `trident_put` | `MPI_Put` into receive buffers | Push each stage's payload with buffer-reuse and completion barriers |

These are CPU adaptations of the hierarchical Trident algorithm. Their scope and
relationship to the original GPU implementation are documented in
[Trident CPU](docs/trident.md). Effective GET overlap depends on MPI progress and
must be established through measurements.

### Node layout and CPU resources

- The physical node count must be a perfect square: **1, 4, 9, 16, ...**.
- Every node must have the same number of MPI ranks. The total rank count need
  not itself be a perfect square.
- `--threads T` sets the OpenMP compute threads per rank. Hybrid additionally
  uses a service worker for inter-node requests and requires `MPI_THREAD_MULTIPLE`.
- For comparisons involving Hybrid, reserve `T + 1` cores per rank for every
  variant, keeping the compute-thread count and total CPU budget equal.

The square constraint applies to the node grid; compatible rectangular input
matrices remain supported. With one node, there is no remote inter-node exchange
or next-stage GET prefetch to evaluate.

### Local check of the hierarchical paths

On a single workstation, eight ranks with `--logical-node-size 2` emulate four
nodes with two ranks each. Open MPI's oversubscription and binding options below
are for this local functional check:

```bash
mpirun -np 8 --oversubscribe --bind-to none ./build/trident_get_pipeline \
  --logical-node-size 2 \
  --matrix-a tests/data/rectangular_a.mtx \
  --matrix-b tests/data/rectangular_b.mtx \
  --threads 2 --warmup 0 --repeats 2 --trials 1 \
  --results test-results/trident_get_pipeline_logical.tsv
```

The same command accepts any Trident executable; choose a corresponding result
filename. Logical topology is allowed only on one physical host and is recorded
as `topology=logical_test`. Use physical nodes for network and scaling measurements.

For all eight implementations, the prepared pilot can also be checked with:

```bash
bash scripts/run_pilot.sh --local-check --nodes 1 4 --threads 2
```

### Direct launch on physical nodes

Inside an allocation of **four nodes with at least ten cores per node**, this
Open MPI example places two ranks per node, each with four compute threads and
five reserved cores. It uses the previously prepared pilot input:

```bash
mpirun -np 8 --map-by ppr:2:node:PE=5 --bind-to core --nooversubscribe \
  ./build/trident_get --matrix matrices/cage8.mtx --threads 4 \
  --warmup 2 --repeats 10 --trials 5 \
  --results test-results/trident_get_cluster_pilot.tsv
```

For a protocol comparison, keep the input and placement fixed and select the
other Trident executables with separate result files. Use the experiment scripts
below for campaigns over multiple inputs and resource configurations.

## Command-line options

All eight executables share the following options. Defaults here refer to direct
executable launches; experiment scripts apply their own configuration.

| Option | Default | Meaning |
| --- | --- | --- |
| `--matrix PATH` | None | Use a square Matrix Market input for both A and B |
| `--matrix-a PATH`, `--matrix-b PATH` | None | Select input files separately; A alone is reused as B |
| `--rows N`, `--cols N`, `--b-cols N` | `1024` each | Synthetic A rows, shared inner dimension, and B columns |
| `--nnz-per-row N`, `--b-nnz-per-row N` | `16` each | Synthetic nonzeros per row; cannot exceed the corresponding column count |
| `--threads N` | `1` | OpenMP compute threads per rank |
| `--schedule NAME` | `guided` | `static`, `dynamic`, `guided`, or `auto` |
| `--chunk N` | `64` | OpenMP scheduling chunk; ignored by `auto` |
| `--warmup N` | `2` | Extra untimed products after the first product |
| `--repeats N` | `10` | Timed products per trial |
| `--trials N` | `5` | Groups of repetitions within the same MPI launch |
| `--no-validate` | Validation enabled | Skip the serial reference and comparison |
| `--results PATH` | Variant-specific | TSV file to append to |
| `--experiment TAG` | `manual` | Label stored in the result row |
| `--help` | — | Print usage and exit |

Trident additionally accepts `--logical-node-size N` for local topology tests.
MPI process counts are set by the launcher, for example `mpirun -np 4`.
Shared numeric options require decimal integers within the signed-int range;
`--warmup` permits zero, and the others must be positive. Values such as `1e3`,
`2.5` or `2junk` are rejected.

## Experiments and cluster execution

The scripts share [scripts/experiments.json](scripts/experiments.json). Configure
the selected matrices, node counts, MPI ranks per node, threads and repetitions
before a campaign. The current resource settings are examples to adapt to the
cluster allocation.

| Script | Purpose |
| --- | --- |
| `run_pilot.sh` | Validate all selected variants on original, permuted and rectangular pilot inputs |
| `run_strong_scaling.sh` | Measure fixed `A*A` products across node counts |
| `run_permutation.sh` | Measure `(P*A*P^T)^2` for comparison with original inputs |
| `run_rectangular.sh` | Measure `A*R` using prepared rectangular operands |
| `analyze_phases.sh` | Export phase timings from recorded observations |
| `analyze_structure.sh` | Export structure descriptors alongside recorded square-product measurements |
| `run_all.sh` | Run the pilot, three measurement stages, then both analysis exports |

### Preview and run within an allocation

Preview the configured commands without downloaded inputs, binaries or MPI runs:

```bash
bash scripts/run_all.sh --dry-run
```

After building, preparing the selected inputs, and obtaining a suitable allocation:

```bash
bash scripts/run_all.sh

# Repeat the whole workflow five times, with one MPI launch per measurement case:
bash scripts/run_all.sh 5 --runs 1
```

`run_all.sh` uses an existing allocation; it does not build, download matrices or
request nodes. The Open MPI launcher reserves `threads + 1` cores per rank by
default for every variant. Each node therefore needs at least
`ranks_per_node * (threads + 1)` allocated cores with that default.

The pilot always validates. Main campaigns currently default to validation
disabled in `experiments.json`; pass `--validate` to enable it. `--runs` controls
independent MPI launches, while `--repeats` and `--trials` control samples within
each launch. For example, `--runs 3 --repeats 10 --trials 5` produces three result
rows, each summarizing 50 timed products per input/configuration/variant.

### Request resources through SbatchMan

Preview a one-node pilot job for either scheduler:

```bash
python3 scripts/prepare_sbatchman.py --scheduler slurm \
  --campaign pilot-001 --action pilot --dry-run -- --nodes 1

python3 scripts/prepare_sbatchman.py --scheduler pbs \
  --campaign pilot-001 --action pilot --dry-run -- --nodes 1
```

The preview requires only Python's standard library and submits no jobs. Follow
the [SbatchMan guide](scripts/sbatchman/README.md) to install the optional tool,
adapt a local profile, generate frozen job configurations, submit and inspect
status. One job is generated per node count; sequential jobs append to the same
result files as `run_all.sh`. The generator's optional `--isolate-results` flag
separates results by campaign, cluster and resources for test runs.

Full script options, local campaign checks, result handling and analysis commands
are documented in the [script guide](scripts/README.md).

## Output and benchmark timings

### Terminal report and TSV files

Every completed benchmark reports the input/output dimensions and nonzeros,
MPI/OpenMP configuration, timings, throughput and validation status on rank 0.
It also appends a TSV row, creating parent directories as needed.

| Direct executable | Default result file |
| --- | --- |
| `spgemm_two_sided` | `results/two_sided/benchmarks_v2.tsv` |
| `spgemm_one_sided_get` | `results/one_sided_get/benchmarks_v2.tsv` |
| `spgemm_one_sided_put` | `results/one_sided_put/benchmarks_v2.tsv` |
| Each `trident_*` executable | `results/trident/<executable>_v1.tsv` |

Use `--results` to select another file. Its header must match the executable's
schema. Old baseline `benchmarks.tsv` files use an earlier timing protocol and
must not be pooled with `benchmarks_v2.tsv` samples.

The experiment scripts collect one file per variant in each experiment directory,
for example `results/strong_scaling/trident_get.tsv`. Pilot and local checks go
under `test-results/`. Repeated campaigns append observations; they do not resume
or deduplicate previous runs. Analysis scripts regenerate derived tables from
the accumulated observations without launching new products.

### What the timers measure

| Field | Measured work |
| --- | --- |
| `distribution_seconds` | Initial matrix distribution from rank 0 |
| `halo_setup_seconds` / `plan_setup_seconds` | Baseline halo preparation / Trident plan, buffers and backend preparation |
| `first_product_seconds` | Setup plus the first complete product, measured together once |
| `full_product_seconds` | Distribution, setup, first product and first gather, measured directly once |
| `first_gather_seconds` | Collection of that first product, before warmup |
| `communication_p90_seconds` | Communication work during repeated products, including protocol synchronization |
| `compute_p90_seconds` | Local SpGEMM computation during repeated products |
| `end_to_end_p90_seconds` | Complete repeated product, excluding input loading, distribution, setup, gather and validation |
| `gather_seconds` | Final reconstruction of C on rank 0 |

Trident additionally reports `inter_node_p90_seconds` and `intra_node_p90_seconds`.
Each duration is reduced to its maximum across ranks. Repeated-product P90 values
use `repeats * trials` samples after the first product and extra warmups; trials
share the same MPI processes. First-product time already includes setup.
The `*_samples_seconds` arrays preserve the individual rank-maximum durations.
The full-product timer excludes file loading, process-grid construction, validation
and output. Measurement definitions and the instrumentation review are recorded in
[the experiment plan](docs/experiments.md#reading-the-existing-measurements) and
[the measurement audit](docs/measurement-audit.md).

Phase maxima and P90s are computed independently, so they need not sum to the
total. For Hybrid and GET Pipeline, phase times describe main-thread work and
waiting and do not directly measure all concurrent network activity. Use
end-to-end product time as the primary comparison with matched inputs/resources.

The reported compute throughput counts two floating-point operations per scalar
multiply-add, using `sum_(i,k in A) nnz(B[k,:])` scalar multiplications. TSV rows
record `benchmark_protocol`: baselines use `prepared_halo_v2`; Trident variants
have their own protocols described in the linked technical documents. Baselines
and Trident also differ in partitioning, payloads and local accumulation, which
must be considered when interpreting their comparison.

## Validation and tests

Direct launches validate by default: after gathering C, rank 0 compares it with
a serial SpGEMM result. Validation requires a maximum absolute error **strictly
below `1e-10`**. NaN and infinity fail. A failed comparison records `FAIL` and
returns a nonzero exit status on all ranks.

`--no-validate` skips the serial reference and comparison and records
`validation=SKIPPED`, `max_abs_error=NA`. The final C is still gathered, and rank 0
still retains global A and B. This removes serial validation time but does not
remove the root-memory limit. CSR indices and MPI counts also remain int-sized.
The separate `result_finite` flag checks the final C for NaN/infinity without a
serial product. Experiment scripts reject non-finite products even when validation
is skipped. A `SKIPPED` row is not equivalent to `PASS`. See the
[measurement definitions](docs/experiments.md#reading-the-existing-measurements)
for raw samples, full-product time and structural-work metrics.

Run the automated suite after building:

```bash
ctest --test-dir build --output-on-failure
```

The suite covers input parsing, numeric options, rectangular/empty matrices,
uneven partitions, validation failures and all eight implementations. Trident
tests also check topology, independent dense reference products, cancellation,
changed payloads across repeated products, and RMA buffer lifetimes.

Some MPI tests use more ranks than a small workstation has available slots. For
local Open MPI correctness testing, allow oversubscription with:

```bash
OMPI_MCA_rmaps_base_oversubscribe=1 ctest --test-dir build --output-on-failure
```

The Python tooling tests can also be run separately without compiling C++:

```bash
python3 tests/matrix_tools_test.py
python3 tests/experiment_scripts_test.py
python3 tests/sbatchman_scripts_test.py
```

Local logical-node tests establish functional behavior; physical multi-node
measurements are needed to assess communication costs and scaling.

## Technical documentation and references

| Document | Contents |
| --- | --- |
| [Trident CPU](docs/trident.md) | Algorithm, source provenance, partitioning and two-sided execution |
| [Trident Hybrid](docs/trident_hybrid.md) | RMA requests, service thread and resource requirements |
| [Trident GET](docs/trident_get.md) | Input windows, synchronization and buffer lifetime |
| [Trident GET Pipeline](docs/trident_get_pipeline.md) | Prefetch scheduling, double buffering and timing interpretation |
| [Trident PUT](docs/trident_put.md) | Push protocol and stage synchronization |
| [Experiment plan](docs/experiments.md) | Proposed comparisons, matrix inputs and measurement methodology |
| [Script guide](scripts/README.md) | Build, preparation, experiments and analysis options |
| [SbatchMan guide](scripts/sbatchman/README.md) | Slurm and OpenPBS/PBS Pro setup and job submission |
| [Scientific and technical sources](docs/sources.md) | MPI/OpenMP standards, Matrix Market and SpGEMM literature |
| [BibTeX references](docs/references.bib) | Bibliography entries for the thesis |
