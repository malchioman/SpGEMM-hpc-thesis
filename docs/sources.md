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
- A. Buluç and J. R. Gilbert, [Parallel Sparse Matrix-Matrix Multiplication and Indexing: Implementation and Experiments](https://doi.org/10.1137/110848244), SIAM Journal on Scientific Computing, 2012. Context for distributed-memory SpGEMM and sparse data movement. The three baselines use one-dimensional row distribution; the new hierarchical variant follows Trident, documented separately below.

## Trident CPU sources

- J. Bellavita, L. Pichetti, T. Pasquali, F. Vella, and G. Guidi, *Communication-Avoiding SpGEMM via Trident Partitioning on Hierarchical GPU Interconnects*, ICS 2026. The supplied paper's Sections 3.2-3.3 and Algorithms 1-2 support the coarse 2D partition, intra-node 1D row slices, staggered static-owner schedule, local B aggregation and C-stationary updates.
- [Official Trident implementation at c37debac](https://github.com/HicrestLaboratory/Trident/tree/c37debaccc58b72859f1837f260900a40094848c), inspected 2026-09-16. `LocalSpGEMMTask`, `TaskQueue`, `TileHolder::node_allgather` and `hns_spgemm_async` provide the corresponding implementation reference. [The pinned distributed_mmio submodule](https://github.com/HicrestLaboratory/distributed_mmio/tree/2e7c9b3c3205f4f019269b155f9381c8c4974b43) provides the hierarchical ownership/indexing reference.
- MPI 4.1 also supports `MPI_Comm_dup`, `MPI_Comm_split`, `MPI_Comm_split_type` with `MPI_COMM_TYPE_SHARED`, and `MPI_Comm_free` used by the CPU topology layer. The Trident two-sided backend implements node-local allgather semantics with explicit point-to-point messages.
- The hybrid variant additionally follows the request/response organization of [`MessageQueue::notify/wait`](https://github.com/HicrestLaboratory/Trident/blob/c37debaccc58b72859f1837f260900a40094848c/include/message_queue.cuh) and the service threads in `hns_spgemm_async`. Its CPU queue uses generation-indexed slots and MPI atomic accesses, not the original GPU queue verbatim. MPI 4.1 [accumulate functions](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node320.htm) support `MPI_Accumulate(MPI_REPLACE)` for publication and `MPI_Fetch_and_op(MPI_NO_OP)` for polling; `MPI_Win_flush` completes these accesses. Concurrent service/main-thread MPI calls require `MPI_THREAD_MULTIPLE`.

The GET variant is a CPU transport experiment based on MPI 4.1
[Get](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node318.htm),
[Flush and Sync](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node331.htm)
and [RMA semantics](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node337.htm).
It exposes local A/B CSR arrays directly, completes each stage's reads with
flushes, and uses publication/completion barriers to separate products. This
protocol is not attributed to the original GPU implementation; its partitioning
and local aggregation retain the shared Trident provenance.

The pipelined GET variant uses the same MPI GET/flush semantics and shared RMA
transport, with two alternating A/B buffer pairs and one-stage lookahead. It
delays completion of the next stage until its data are needed; it does not change
partitioning or intra-node aggregation. This CPU schedule is an experimental
variant, not an attribution to the GPU implementation, and does not by itself
prove autonomous MPI progress or effective overlap. See
[pipelined GET](trident_get_pipeline.md) for scheduling, lifetime and timing rules.

The new code is a CPU reimplementation of these algorithmic components, not a
direct copy of the GPU source. [The provenance mapping and differences](trident.md)
document the changed range splitting, actual-node discovery, OpenMP kernel,
staged two-sided inter-node schedule, and lack of request queues, work stealing,
and overlap in the first two-sided version. [Trident hybrid](trident_hybrid.md)
documents the added request service, its differences from the original, and the
lack of next-stage pipelining. [Trident GET](trident_get.md) documents direct
input exposure, buffer lifetime and product-boundary synchronization. The Trident
paper is not evidence for the performance of these CPU versions.

## Hybrid MPI/OpenMP programming

- R. Rabenseifner, G. Hager, and G. Jost, [Hybrid MPI/OpenMP Parallel Programming on Clusters of Multi-Core SMP Nodes](https://doi.org/10.1109/PDP.2009.43), PDP 2009. Context for the hybrid programming model used in this repository: MPI handles distributed-memory communication across ranks, while OpenMP handles shared-memory parallelism inside each rank. This source supports the execution model, not a SpGEMM-specific algorithm.

## One-sided MPI/RMA literature

- T. Hoefler et al., [Remote Memory Access Programming in MPI-3](https://doi.org/10.1145/2780584), ACM Transactions on Parallel Computing, 2015. Background for passive-target RMA epochs, synchronization, and memory model considerations used by the one-sided variants.
- J. Dinan et al., [An Implementation and Evaluation of the MPI 3.0 One-Sided Communication Interface](https://doi.org/10.1002/cpe.3758), Concurrency and Computation: Practice and Experience, 2016. Implementation and performance context for interpreting the RMA variants; not a source for repository-specific code.

## Baseline implementation assumptions

- `A`, `B`, and `C` are stored as CSR matrices. The Matrix Market reader rejects nonsquare inputs declared `symmetric`, `skew-symmetric`, or `hermitian` before expanding their entries; `general` inputs may be rectangular.
- `A` is distributed by contiguous row blocks. Each rank owns and computes the corresponding row block of `C`.
- `B` is also distributed by contiguous row blocks. For every local nonzero `A(i,k)`, the owning rank of row `B(k,:)` is found from the `B` row partition.
- The two-sided baseline builds the request plan and exchanges sparse row lengths once. Each product packs the requested rows into reusable buffers, exchanges column indices and values with `MPI_Isend`/`MPI_Irecv`, and computes local rows of `C`.
- The `spgemm_one_sided_get` variant exposes each local CSR block of `B` through separate MPI windows for `rowPtr`, `columnIndices`, and `values`. A requesting rank fetches row bounds once during setup, then reuses those bounds to fetch column indices and values for every product.
- The `spgemm_one_sided_put` variant builds request metadata once, exchanges row lengths and target offsets with point-to-point setup messages, exposes sparse halo arrays through MPI windows, and lets the owner ranks push sparse row payloads into the requesting ranks' halo buffers.
- RMA windows are held in passive-target `MPI_Win_lock_all` epochs and completed with `MPI_Win_flush_all`. The put variant uses a barrier and `MPI_Win_sync` before local reads of remotely written halo buffers. Exposed storage outlives `MPI_Win_free`; unexpected rank-local exceptions abort directly rather than attempting potentially unmatched collective cleanup.
- The local SpGEMM kernel uses one accumulator per output row, sorts touched output columns, drops exact zero accumulated values, and writes the result as CSR.
- All variants use the `prepared_halo_v2` protocol for fixed sparsity: setup and the first product including setup are timed separately from subsequent products reusing the plan. Each duration is reduced to the maximum rank time. The results file stores P90 communication, computation, and end-to-end timings across repeated products after warmup. The first-product measurement is a single sample, not a P90.
- Rank 0 gathers the distributed CSR result and, by default, validates it against a serial SpGEMM implementation outside the timed samples. The benchmark-specific validation policy requires a maximum absolute error strictly below `1e-10` and rejects non-finite values in either result. A failed validation is recorded as `FAIL`; its status is broadcast to all ranks, which return a nonzero exit code after normal MPI cleanup. The project-specific `--no-validate` flag skips the serial product and comparison, reports `SKIPPED` with `max_abs_error=NA`, and permits a successful exit without certifying numerical correctness. It does not disable input checks or final result gathering, nor remove global A and B from rank 0.

Trident retains the input and validation policies above but uses different
partitioning, payloads and timing protocols (`trident_staged_csr_v1` and
`trident_hybrid_rma_requests_v1`, `trident_get_csr_v1` and `trident_get_pipeline_csr_v1`); see
[Trident CPU](trident.md). The baseline halo assumptions do not describe Trident.
