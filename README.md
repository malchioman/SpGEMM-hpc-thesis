# spgemm-hpc-thesis

Repository for a thesis on CPU-distributed Sparse General Matrix-Matrix
Multiplication (SpGEMM), computing `C = A * B` with `A`, `B`, and `C` stored as
sparse CSR matrices.

The repository contains three row-distributed `MPI + OpenMP` SpGEMM baselines
and four hierarchical CPU Trident variants: `trident_two_sided`, `trident_hybrid`,
`trident_get` and `trident_get_pipeline`.
The baselines use the following communication variants:

- `A` is distributed by contiguous row blocks; each rank computes the matching
  row block of `C`;
- `B` is distributed by rows across MPI ranks;
- `spgemm_two_sided` exchanges sparse row lengths, column indices, and values
  with `MPI_Isend`/`MPI_Irecv`;
- `spgemm_one_sided_get` exposes each local CSR block of `B` through MPI windows
  and fetches remote row bounds, column indices, and values with `MPI_Get`;
- `spgemm_one_sided_put` exposes each rank's sparse halo buffers through MPI
  windows, then lets the owning ranks push requested rows of `B` with `MPI_Put`;
- the local multiplication uses a Gustavson-style row accumulator and is
  parallelized across output rows with OpenMP;
- rank 0 gathers the distributed CSR result and validates it against a serial
  SpGEMM implementation.

## Source layout

```text
src/
  common/
    csr_matrix.hpp
    matrix_market.cpp
    matrix_market.hpp
    spgemm_common.cpp
    spgemm_common.hpp
  baselines/
    two_sided_spgemm.cpp
    one_sided_get_spgemm.cpp
    one_sided_put_spgemm.cpp
    spgemm_exchange.cpp
    spgemm_exchange.hpp
  trident/
    common/
    two_sided/
    hybrid/
    get/
    get_pipeline/
```

`common` contains CSR storage, Matrix Market input, shared MPI/benchmark helpers,
and serial validation. It also retains the existing row-distribution utilities.
`baselines` contains the three row-distributed SpGEMM programs, their remote-row
exchange code, and the distributed local kernel. `trident/common` contains the
hierarchical process grid, partitioning, static-Cannon stage plan, CPU kernel,
and benchmark driver shared by Trident variants. The shared two-sided intra-node
backend lives in `trident/common/intra_node_two_sided.*`. `trident/two_sided`,
`trident/hybrid`, `trident/get` and `trident/get_pipeline` each contain their own inter-node backend and
executable entry point. All mains explicitly select an intra-node and an
inter-node backend; no variant depends on another variant's library.
Both GET variants reuse the input windows and split-phase GET/flush operations
in `trident/common/rma_get.*`.

The CMake targets `matrix_market` and `spgemm_support` provide common utilities.
`spgemm_baseline_support` depends on those utilities and adds the baseline-specific
exchange and computation code. Executable names and run commands are unchanged.

`trident_support` shares the existing common utilities but is independent of the
baseline halo implementation. `trident_two_sided` uses a 2D grid of physical nodes
and 1D row slices inside each node, with `MPI_Isend`/`MPI_Irecv` both between and
within nodes. This initial version is staged, without the GPU original's request
queues or communication/computation overlap. See [Trident CPU](docs/trident.md)
for source provenance, exact scope, cluster/local commands, timing definitions,
and the shared structure intended for subsequent variants.

`trident_hybrid` reuses that partitioning, kernel and intra-node backend, adding
RMA request slots and a service thread for two-sided inter-node responses.
It requires `MPI_THREAD_MULTIPLE`; the two-sided version remains FUNNELED.
See [Trident hybrid](docs/trident_hybrid.md) for protocol details and the extra
CPU-core budget. Neither backend implements next-stage prefetch/pipelining.

`trident_get` exposes the existing local CSR inputs and retrieves remote A/B
slices with `MPI_Get`, still using two-sided intra-node aggregation. It has no
service worker or pipeline and requires only `MPI_THREAD_FUNNELED`. Publication
and reader-completion barriers are included in product timings. See
[Trident GET](docs/trident_get.md) for synchronization and buffer-lifetime rules.

`trident_get_pipeline` uses the same GET transport but alternates two A/B buffer
pairs: it starts the next stage before intra-node aggregation and computation of
the current one. It remains FUNNELED, without an additional service core. Actual
overlap depends on MPI progress; see [pipelined GET](docs/trident_get_pipeline.md)
for scheduling, memory costs and interpretation of phase timings.

## Build

A C++17 compiler, an MPI distribution with development files (for example Open
MPI), and an OpenMP-capable compiler are required.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build --output-on-failure
```

On a Linux cluster with Open MPI:

```bash
mpirun -np 4 ./build/spgemm_two_sided --rows 4096 --cols 4096 --b-cols 4096 --nnz-per-row 16 --b-nnz-per-row 16 --threads 8 --schedule guided --repeats 10 --trials 5
mpirun -np 4 ./build/spgemm_one_sided_get --rows 4096 --cols 4096 --b-cols 4096 --nnz-per-row 16 --b-nnz-per-row 16 --threads 8 --schedule guided --repeats 10 --trials 5
mpirun -np 4 ./build/spgemm_one_sided_put --rows 4096 --cols 4096 --b-cols 4096 --nnz-per-row 16 --b-nnz-per-row 16 --threads 8 --schedule guided --repeats 10 --trials 5
```

To multiply Matrix Market coordinate matrices, pass `--matrix-a` and `--matrix-b`.
Their dimensions must satisfy `A.cols == B.rows`:

```bash
mpirun -np 4 ./build/spgemm_two_sided --matrix-a matrices/A.mtx --matrix-b matrices/B.mtx --threads 8 --schedule guided --repeats 10 --trials 5
mpirun -np 4 ./build/spgemm_one_sided_get --matrix-a matrices/A.mtx --matrix-b matrices/B.mtx --threads 8 --schedule guided --repeats 10 --trials 5
mpirun -np 4 ./build/spgemm_one_sided_put --matrix-a matrices/A.mtx --matrix-b matrices/B.mtx --threads 8 --schedule guided --repeats 10 --trials 5
```

Passing only `--matrix-a path/to/A.mtx` reuses the loaded A as B and computes
`A * A`; A must be square. The shorthand `--matrix path/to/A.mtx` also uses the
same Matrix Market file as both inputs. In both cases, the summary and TSV
record that file as the source of both A and B. Specifying `--matrix-b` instead
selects `A * B`, subject to the dimension compatibility rule above.

The reader supports `real`, `integer`, and `pattern` fields, and recognizes the
`general`, `symmetric`, `skew-symmetric`, and `hermitian` storage-mode tokens for
those non-complex inputs. The three symmetry modes require square matrices;
`general` inputs may be square or rectangular. Synthetic inputs use `--rows` for rows of `A`, `--cols`
for columns of `A` and rows of `B`, `--b-cols` for columns of `B`, and separate
nonzero controls for the two inputs.
Numeric options must be fully parsed decimal integers within the signed-int
range. Values such as `1e3`, `2junk`, and `2.5` are rejected, not truncated.
`--warmup` permits zero; the other shared numeric options must be positive.

All three baseline executables use the `prepared_halo_v2` benchmark protocol. With fixed
input sparsity, they prepare requests, row lengths, offsets, and communication
buffers once. Every subsequent product transfers column indices and values
again. Two-sided communication includes packing the outgoing rows; GET and PUT
access the existing CSR buffers directly. Protocol-specific synchronization
remains part of the measured communication cost.

The timings distinguish:

- `distribution_seconds`: distribution of A and B from rank 0;
- `halo_setup_seconds`: metadata, buffer allocation, window creation where
  applicable, and synchronization making the prepared plan ready on all ranks;
- `first_product_seconds`: setup plus the first payload exchange and local
  multiplication, measured together before warmup (one sample per execution);
- `communication_p90_seconds`, `compute_p90_seconds`, `end_to_end_p90_seconds`:
  repeated products reusing the prepared plan, after `--warmup` extra products;
- `gather_seconds`: collection of the final result, outside product timings.

Each duration is reduced to its maximum across ranks. First-product and setup
endpoints are captured before the timing reductions. Repeated-product P90 values
are computed over `--repeats * --trials` samples within the same MPI run; trials
are not independent process launches. `end_to_end` includes communication and
local computation, excluding distribution, setup, gathering, and validation.
Independently reduced phase maxima/P90 values need not add up to the total.
The reported compute throughput uses `sum_(i,k in A) nnz(B[k,:])` scalar
multiplications, counted as two floating-point operations per multiply-add.

Each execution appends a TSV row under `results/<variant>/benchmarks_v2.tsv`.
Existing `benchmarks.tsv` files are preserved; their timing protocol differs
and their samples must not be pooled with v2 samples. A supplied `--results`
file must be empty or have the matching v2 header. The TSV records input shapes,
nonzeros, timings, throughput, validation status, and `benchmark_protocol`.

Validation checks the final product against serial SpGEMM with maximum absolute
error below `1e-10`. NaN and infinity always fail. A validation failure is recorded
as `FAIL` and returns a nonzero exit status on every rank. Rank 0 still holds the
complete inputs and final result for validation; this limits large-scale runs.
The tests cover numerical validation, the three baseline transports, rectangular
and empty matrices, uneven partitions, ranks without rows, and multiple OpenMP
threads. Trident adds hierarchical topology tests and independent dense-product
checks, including cancellation and repeated products with changed input payloads.
Parser tests cover every shared numeric option, malformed suffixes, integer
bounds and zero warmup. MPI CLI regressions verify that all seven executables
reject malformed numeric arguments without writing benchmark results.

Validation is enabled by default, runs outside the timed samples, and requires
a maximum absolute error strictly below `1e-10`. Non-finite values in either result, including matching
infinities, fail validation. On validation failure, the program records `FAIL`
and all ranks return a nonzero exit code after normal MPI cleanup.

Add `--no-validate` to any baseline or Trident executable to skip both the serial
reference product and the result comparison:

```bash
mpirun -np 4 ./build/spgemm_one_sided_get --matrix-a matrices/A.mtx --matrix-b matrices/B.mtx --threads 8 --no-validate
```

In this mode, both the summary and TSV report `validation=SKIPPED` and
`max_abs_error=NA`, not `PASS` or a zero error. A completed run returns success
without certifying numerical correctness. Input checks and runtime errors remain
active. The final result is still gathered on rank 0, and global A and B remain
there, so this flag removes the serial validation cost but not all root-memory
limits for large inputs. The distributed algorithm and timed regions are unchanged.
The baseline scaling scripts keep validation enabled unless their executable
invocation is amended to include `--no-validate`. Trident scripts expose the
equivalent setting through `VALIDATE=0`.

## Scaling experiments

Strong scaling keeps the input matrices and threads per rank fixed while varying
the number of ranks. With one square Matrix Market matrix:

```bash
RANKS="1 2 4 8" THREADS=8 bash scripts/baselines/run_strong_scaling.sh matrices/bcsstk18.mtx
RANKS="1 2 4 8" THREADS=8 bash scripts/baselines/run_strong_scaling_one_sided_get.sh matrices/bcsstk18.mtx
RANKS="1 2 4 8" THREADS=8 bash scripts/baselines/run_strong_scaling_one_sided_put.sh matrices/bcsstk18.mtx
```

With two compatible Matrix Market matrices:

```bash
RANKS="1 2 4 8" THREADS=8 bash scripts/baselines/run_strong_scaling.sh matrices/A.mtx matrices/B.mtx
RANKS="1 2 4 8" THREADS=8 bash scripts/baselines/run_strong_scaling_one_sided_get.sh matrices/A.mtx matrices/B.mtx
RANKS="1 2 4 8" THREADS=8 bash scripts/baselines/run_strong_scaling_one_sided_put.sh matrices/A.mtx matrices/B.mtx
```

Weak scaling uses synthetic matrices and increases global dimensions in
proportion to the number of ranks:

```bash
RANKS="1 2 4 8" THREADS=8 ROWS_PER_RANK=4096 COLS_PER_RANK=4096 \
  B_COLS_PER_RANK=4096 NNZ_PER_ROW=16 B_NNZ_PER_ROW=16 \
  bash scripts/baselines/run_weak_scaling.sh
RANKS="1 2 4 8" THREADS=8 ROWS_PER_RANK=4096 COLS_PER_RANK=4096 \
  B_COLS_PER_RANK=4096 NNZ_PER_ROW=16 B_NNZ_PER_ROW=16 \
  bash scripts/baselines/run_weak_scaling_one_sided_get.sh
RANKS="1 2 4 8" THREADS=8 ROWS_PER_RANK=4096 COLS_PER_RANK=4096 \
  B_COLS_PER_RANK=4096 NNZ_PER_ROW=16 B_NNZ_PER_ROW=16 \
  bash scripts/baselines/run_weak_scaling_one_sided_put.sh
```

The scaling scripts set the `experiment` field automatically. Unless `RESULTS`
is overridden, the two-sided scripts write to `results/two_sided/benchmarks_v2.tsv`,
the get wrappers write to `results/one_sided_get/benchmarks_v2.tsv`, and the put
wrappers write to `results/one_sided_put/benchmarks_v2.tsv`.

### Trident scripts

The existing scripts now live in `scripts/baselines/`. Dedicated Trident scripts
live in `scripts/trident/`, with shared launch settings in `common.sh`. They use
Open MPI and vary square physical-node counts while keeping ranks per node and
OpenMP threads explicit:

```bash
NODES="1 4 9" RANKS_PER_NODE=2 THREADS=4 VALIDATE=0 \
  bash scripts/trident/run_strong_scaling.sh matrices/A.mtx matrices/B.mtx
NODES="1 4 9" RANKS_PER_NODE=2 THREADS=4 VALIDATE=0 \
  bash scripts/trident/run_weak_scaling.sh
bash scripts/trident/run_local_check.sh
VARIANT=hybrid bash scripts/trident/run_local_check.sh
VARIANT=get bash scripts/trident/run_local_check.sh
VARIANT=get_pipeline bash scripts/trident/run_local_check.sh
VARIANT=hybrid NODES="1 4 9" RANKS_PER_NODE=2 THREADS=4 VALIDATE=0 \
  bash scripts/trident/run_strong_scaling.sh matrices/A.mtx matrices/B.mtx
VARIANT=hybrid NODES="1 4 9" RANKS_PER_NODE=2 THREADS=4 VALIDATE=0 \
  bash scripts/trident/run_weak_scaling.sh
```

Physical experiments require a suitable allocation or `HOSTFILE`; the local
check uses logical nodes on localhost and is not a scaling measurement.
Set `DRY_RUN=1` to preview commands. Validation is enabled by default.
Generic Trident scripts accept `VARIANT=two_sided`, `VARIANT=hybrid`, `VARIANT=get`
or `VARIANT=get_pipeline`. All four variants use the same scripts directly,
without variant-specific wrappers.
Physical hybrid launches reserve one additional CPU core per rank for service;
`THREADS` still specifies compute threads. Both GET variants and two-sided use `THREADS` cores
per rank. Scripts do not allocate resources.
See [experiment scripts](scripts/README.md) for all settings, separate result
paths, and the weak-scaling input model.

The implementations' technical and scientific sources are listed in
[`docs/sources.md`](docs/sources.md), with BibTeX citations in
[`docs/references.bib`](docs/references.bib).
