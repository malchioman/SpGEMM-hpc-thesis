# spmm-hpc-thesis

Repository for a thesis on CPU-distributed Sparse Matrix-Dense Matrix Multiplication (SpMM).

The first available baseline is `MPI + OpenMP` with two-sided point-to-point communication:

- `A` is a CSR sparse matrix distributed by contiguous row blocks;
- `B` is distributed by rows across MPI ranks;
- each rank performs an explicit halo exchange with `MPI_Isend`/`MPI_Irecv` to obtain the remote rows of `B` needed by its local portion of `A`;
- the local kernel is parallelized with OpenMP and the number of threads is configurable;
- rank 0 validates the result against a serial implementation.

## Build

A C++17 compiler, an MPI distribution with development files (for example Open
MPI), and an OpenMP-capable compiler are required.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

On a Linux cluster with Open MPI:

```bash
mpirun -np 4 ./build/spmm_two_sided --rows 4096 --cols 4096 --dense-cols 64 --nnz-per-row 16 --threads 1 --schedule guided --repeats 10 --trials 5
mpirun -np 4 ./build/spmm_two_sided --rows 4096 --cols 4096 --dense-cols 64 --nnz-per-row 16 --threads 8 --schedule guided --repeats 10 --trials 5
```

To use a real Matrix Market coordinate matrix, pass `--matrix`. Loading and CSR
conversion happen before measurements begin:

```bash
mpirun -np 4 ./build/spmm_two_sided --matrix matrices/bcsstk18.mtx --dense-cols 64 --threads 8 --schedule guided --repeats 10 --trials 5
```

The reader supports `real`, `integer`, and `pattern` fields, as well as
`general`, `symmetric`, `skew-symmetric`, and `hermitian` matrices. The shape of
`A` is read from the file; `--rows`, `--cols`, and `--nnz-per-row` only affect
the default synthetic matrix.

Comparing `--threads 1` with a larger value isolates the contribution of OpenMP
for a fixed number of MPI ranks. The executable performs an untimed warmup and
then records the maximum computation time across ranks for every timed sample.
It reports the 90th percentile (P90) of communication, computation, and
end-to-end samples, as well as one-off distribution, halo-plan setup, result
gathering, and numerical validation. Dense matrix `B` changes for every sample;
the required remote values are exchanged again before each SpMM. In the current
baseline, the request plan is also rebuilt for each sample and is therefore
included in the communication P90. The
default OpenMP schedule is `guided`; `static`, `dynamic`, and `auto` remain
selectable with `--schedule`, and `--chunk` applies to the first three schedules.

Each execution appends a TSV row to `results/two_sided/benchmarks.tsv`. The file
contains the configuration, timings, throughput, and validation result; change
the destination with `--results path/to/output.tsv`.

## Scaling experiments

Strong scaling keeps the input matrix, dense-column count, and threads per rank
fixed while varying the number of ranks. Run it with:

```bash
RANKS="1 2 4 8" THREADS=8 bash scripts/run_strong_scaling.sh matrices/bcsstk18.mtx
```

Weak scaling uses synthetic matrices and increases global rows and columns in
proportion to the number of ranks, keeping `ROWS_PER_RANK`, `COLS_PER_RANK`,
`NNZ_PER_ROW`, `DENSE_COLS`, and `THREADS` fixed:

```bash
RANKS="1 2 4 8" THREADS=8 ROWS_PER_RANK=4096 COLS_PER_RANK=4096 \
  bash scripts/run_weak_scaling.sh
```

Both scripts append to the same TSV and set the `experiment` field automatically.

The baseline's technical and scientific sources are listed in [`docs/sources.md`](docs/sources.md), with BibTeX citations in [`docs/references.bib`](docs/references.bib).
