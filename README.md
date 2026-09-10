# spgemm-hpc-thesis

Repository for a thesis on CPU-distributed Sparse General Matrix-Matrix
Multiplication (SpGEMM), computing `C = A * B` with `A`, `B`, and `C` stored as
sparse CSR matrices.

The corrected implementation family is `MPI + OpenMP` with three communication
variants:

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

For a square matrix product `A * A`, `--matrix path/to/A.mtx` is a shorthand for
using the same Matrix Market file as both inputs.

The reader supports `real`, `integer`, and `pattern` fields, and recognizes the
`general`, `symmetric`, `skew-symmetric`, and `hermitian` storage-mode tokens for
those non-complex inputs. Synthetic inputs use `--rows` for rows of `A`, `--cols`
for columns of `A` and rows of `B`, `--b-cols` for columns of `B`, and separate
nonzero controls for the two inputs.

All three executables use the `prepared_halo_v2` benchmark protocol. With fixed
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
The tests cover numerical validation, all three transports, rectangular and
empty matrices, uneven partitions, ranks without rows, and multiple OpenMP threads.

## Scaling experiments

Strong scaling keeps the input matrices and threads per rank fixed while varying
the number of ranks. With one square Matrix Market matrix:

```bash
RANKS="1 2 4 8" THREADS=8 bash scripts/run_strong_scaling.sh matrices/bcsstk18.mtx
RANKS="1 2 4 8" THREADS=8 bash scripts/run_strong_scaling_one_sided_get.sh matrices/bcsstk18.mtx
RANKS="1 2 4 8" THREADS=8 bash scripts/run_strong_scaling_one_sided_put.sh matrices/bcsstk18.mtx
```

With two compatible Matrix Market matrices:

```bash
RANKS="1 2 4 8" THREADS=8 bash scripts/run_strong_scaling.sh matrices/A.mtx matrices/B.mtx
RANKS="1 2 4 8" THREADS=8 bash scripts/run_strong_scaling_one_sided_get.sh matrices/A.mtx matrices/B.mtx
RANKS="1 2 4 8" THREADS=8 bash scripts/run_strong_scaling_one_sided_put.sh matrices/A.mtx matrices/B.mtx
```

Weak scaling uses synthetic matrices and increases global dimensions in
proportion to the number of ranks:

```bash
RANKS="1 2 4 8" THREADS=8 ROWS_PER_RANK=4096 COLS_PER_RANK=4096 \
  B_COLS_PER_RANK=4096 NNZ_PER_ROW=16 B_NNZ_PER_ROW=16 \
  bash scripts/run_weak_scaling.sh
RANKS="1 2 4 8" THREADS=8 ROWS_PER_RANK=4096 COLS_PER_RANK=4096 \
  B_COLS_PER_RANK=4096 NNZ_PER_ROW=16 B_NNZ_PER_ROW=16 \
  bash scripts/run_weak_scaling_one_sided_get.sh
RANKS="1 2 4 8" THREADS=8 ROWS_PER_RANK=4096 COLS_PER_RANK=4096 \
  B_COLS_PER_RANK=4096 NNZ_PER_ROW=16 B_NNZ_PER_ROW=16 \
  bash scripts/run_weak_scaling_one_sided_put.sh
```

The scaling scripts set the `experiment` field automatically. Unless `RESULTS`
is overridden, the two-sided scripts write to `results/two_sided/benchmarks_v2.tsv`,
the get wrappers write to `results/one_sided_get/benchmarks_v2.tsv`, and the put
wrappers write to `results/one_sided_put/benchmarks_v2.tsv`.

The implementations' technical and scientific sources are listed in
[`docs/sources.md`](docs/sources.md), with BibTeX citations in
[`docs/references.bib`](docs/references.bib).
