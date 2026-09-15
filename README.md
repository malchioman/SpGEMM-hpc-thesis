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
```

`common` contains CSR storage, Matrix Market input, shared MPI/benchmark helpers,
and serial validation. It also retains the existing row-distribution utilities.
`baselines` contains the three row-distributed SpGEMM programs, their remote-row
exchange code, and the distributed local kernel. `trident` is reserved for the
future CPU implementation and is currently tracked with a `.gitkeep` file.

The CMake targets `matrix_market` and `spgemm_support` provide common utilities.
`spgemm_baseline_support` depends on those utilities and adds the baseline-specific
exchange and computation code. Executable names and run commands are unchanged.

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

Each executable performs untimed warmup iterations, then records the maximum
communication, computation, and end-to-end time across ranks for every timed
sample. The reported throughput uses the scalar multiplication count
`sum_(i,k in A) nnz(B[k,:])`, counted as two floating-point operations per
multiply-add. The two-sided executable rebuilds the remote-row request plan and
re-exchanges the referenced sparse rows of `B` for each timed sample. The
one-sided executables build the remote-row request plan once: the get variant
refetches remote row bounds and sparse payloads with RMA, while the put variant
reuses agreed target offsets and refreshes the sparse halo with RMA puts.

Each execution appends a TSV row under `results/<variant>/benchmarks.tsv`. The
file contains input shapes, input/output nonzeros, timings, throughput, and
validation status; change the destination with `--results path/to/output.tsv`.

Validation is enabled by default, runs outside the timed samples, and requires
a maximum absolute error strictly below `1e-10`. Non-finite values in either result, including matching
infinities, fail validation. On validation failure, the program records `FAIL`
and all ranks return a nonzero exit code after normal MPI cleanup.

Add `--no-validate` to any of the three executables to skip both the serial
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
The scaling scripts keep validation enabled unless their executable invocation
is amended to include `--no-validate`.

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
is overridden, the two-sided scripts write to `results/two_sided/benchmarks.tsv`,
the get wrappers write to `results/one_sided_get/benchmarks.tsv`, and the put
wrappers write to `results/one_sided_put/benchmarks.tsv`.

The implementations' technical and scientific sources are listed in
[`docs/sources.md`](docs/sources.md), with BibTeX citations in
[`docs/references.bib`](docs/references.bib).
