# Sources for the distributed SpGEMM implementation

This document records the sources used for the corrected `MPI + OpenMP` Sparse
General Matrix-Matrix Multiplication implementations.

## Standards, APIs, and input formats

- [MPI: A Message-Passing Interface Standard, Version 4.1](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report.pdf), MPI Forum, 2023. Normative source for `MPI_Init_thread`, `MPI_THREAD_FUNNELED`, `MPI_COMM_WORLD`, rank/size queries, sends/receives, request completion, broadcasts, barriers, reductions, `MPI_Wtime`, finalization, error handling, and one-sided RMA.
- MPI 4.1 HTML sections for the communication model used here:
  [Starting MPI Processes](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node267.htm),
  [Finalizing MPI](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node268.htm),
  [Communicator Accessors](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node187.htm),
  [Point-to-Point Communication](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node53.htm),
  [Blocking Send and Receive Operations](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node55.htm),
  [Nonblocking Communication](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node71.htm),
  [Communication Completion](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node74.htm),
  [Multiple Completions](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node76.htm),
  [Collective Communication](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node113.htm),
  [Barrier Synchronization](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node119.htm),
  [Broadcast](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node120.htm),
  [Global Reduction Operations](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node129.htm),
  [Timers and Synchronization](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node262.htm),
  [Error Handling](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node252.htm),
  [MPI Functionality that is Always Available](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node277.htm),
  [Aborting MPI Processes](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node278.htm),
  [Window Creation](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node309.htm),
  [Window Destruction](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node313.htm),
  [Put](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node317.htm),
  [Get](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node318.htm),
  [Memory Model](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node326.htm),
  [Lock](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node330.htm),
  and [Flush and Sync](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node331.htm).
- [OpenMP API Specification 5.2](https://www.openmp.org/spec-html/5.2/openmp.html), OpenMP Architecture Review Board, 2021. Reference for OpenMP parallel regions, worksharing-loop directives, runtime schedules, `omp_set_num_threads`, `omp_set_schedule`, and `omp_set_dynamic`.
- [Matrix Market File Formats](https://math.nist.gov/MatrixMarket/formats.html), NIST. Reference for the coordinate input parsed by `readMatrixMarket`: one-based coordinate entries, the supported `real`, `integer`, and `pattern` fields, and the storage-mode tokens recognized by the reader (`general`, `symmetric`, `skew-symmetric`, `hermitian`). The implementation does not parse the Matrix Market `complex` field; Hermitian input is therefore usable here only when the values are in a supported non-complex field.
- R. F. Boisvert, R. Pozo, and K. Remington, [The Matrix Market Exchange Formats: Initial Design](https://www.nist.gov/publications/matrix-market-exchange-formats-initial-design), NISTIR 5935, 1996. Formal report behind the Matrix Market exchange format; used as format background, not as an algorithmic source.

## SpGEMM literature

- F. G. Gustavson, [Two Fast Algorithms for Sparse Matrices: Multiplication and Permuted Transposition](https://doi.org/10.1145/355791.355796), ACM Transactions on Mathematical Software, 1978. Classical row-wise sparse-matrix multiplication reference; the local kernel follows this accumulator-based style.
- A. Buluç and J. R. Gilbert, [Parallel Sparse Matrix-Matrix Multiplication and Indexing: Implementation and Experiments](https://doi.org/10.1137/110848244), SIAM Journal on Scientific Computing, 2012. Context for distributed-memory SpGEMM as a sparse linear algebra primitive and for the cost of sparse data movement; this repository uses a simpler one-dimensional row distribution rather than their two-dimensional distribution.

## Hybrid MPI/OpenMP programming

- R. Rabenseifner, G. Hager, and G. Jost, [Hybrid MPI/OpenMP Parallel Programming on Clusters of Multi-Core SMP Nodes](https://doi.org/10.1109/PDP.2009.43), PDP 2009. Context for the hybrid programming model used in this repository: MPI handles distributed-memory communication across ranks, while OpenMP handles shared-memory parallelism inside each rank. This source supports the execution model, not a SpGEMM-specific algorithm.

## One-sided MPI/RMA literature

- T. Hoefler et al., [Remote Memory Access Programming in MPI-3](https://doi.org/10.1145/2780584), ACM Transactions on Parallel Computing, 2015. Background for passive-target RMA epochs, synchronization, and memory model considerations used by the one-sided variants.
- J. Dinan et al., [An Implementation and Evaluation of the MPI 3.0 One-Sided Communication Interface](https://doi.org/10.1002/cpe.3758), Concurrency and Computation: Practice and Experience, 2016. Implementation and performance context for interpreting the RMA variants; not a source for repository-specific code.

## Implementation assumptions

- `A`, `B`, and `C` are stored as CSR matrices.
- `A` is distributed by contiguous row blocks. Each rank owns and computes the corresponding row block of `C`.
- `B` is also distributed by contiguous row blocks. For every local nonzero `A(i,k)`, the owning rank of row `B(k,:)` is found from the `B` row partition.
- The two-sided baseline rebuilds the remote-row request plan for each timed sample, exchanges sparse row lengths, column indices, and values with `MPI_Isend`/`MPI_Irecv`, then computes local rows of `C`.
- The `spgemm_one_sided_get` variant exposes each local CSR block of `B` through separate MPI windows for `rowPtr`, `columnIndices`, and `values`. A requesting rank first fetches row bounds, then fetches the corresponding sparse payload.
- The `spgemm_one_sided_put` variant builds request metadata once, exchanges row lengths and target offsets with point-to-point setup messages, exposes sparse halo arrays through MPI windows, and lets the owner ranks push sparse row payloads into the requesting ranks' halo buffers.
- RMA windows are held in passive-target `MPI_Win_lock_all` epochs and completed with `MPI_Win_flush_all`. The put variant uses a barrier and `MPI_Win_sync` before local reads of remotely written halo buffers.
- The local SpGEMM kernel uses one accumulator per output row, sorts touched output columns, drops exact zero accumulated values, and writes the result as CSR.
- Each timed sample is reduced to the maximum rank time. The results file stores P90 communication, computation, and end-to-end timings across samples after warmup.
- Rank 0 gathers the distributed CSR result and validates it against a serial SpGEMM implementation.
