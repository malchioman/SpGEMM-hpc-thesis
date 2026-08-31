# Sources for the distributed SpMM implementations

This document records the sources used for the `MPI + OpenMP` two-sided baseline
and for the `MPI` one-sided/RMA variants. Standards, implementation
documentation, and scientific literature are deliberately kept distinct.

## Standards and official documentation

- [MPI: A Message-Passing Interface Standard, Version 4.1](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report.pdf), MPI Forum, 2023. Normative source for point-to-point communication, one-sided communication, passive-target synchronization, the RMA memory model, and `MPI_THREAD_FUNNELED`.
- MPI 4.1 HTML sections for the specific RMA mechanisms used here:
  [Window Creation](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node309.htm),
  [Put](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node317.htm),
  [Get](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node318.htm),
  [Memory Model](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node326.htm),
  [Lock](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node330.htm), and
  [Flush and Sync](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node331.htm).
- Open MPI API manual pages for the exact RMA calls exercised by the one-sided implementations:
  [MPI_Win_create](https://docs.open-mpi.org/en/main/man-openmpi/man3/MPI_Win_create.3.html),
  [MPI_Get](https://docs.open-mpi.org/en/main/man-openmpi/man3/MPI_Get.3.html),
  [MPI_Put](https://docs.open-mpi.org/en/main/man-openmpi/man3/MPI_Put.3.html),
  [MPI_Win_lock_all](https://docs.open-mpi.org/en/main/man-openmpi/man3/MPI_Win_lock_all.3.html),
  [MPI_Win_flush_all](https://docs.open-mpi.org/en/main/man-openmpi/man3/MPI_Win_flush_all.3.html),
  [MPI_Win_sync](https://docs.open-mpi.org/en/main/man-openmpi/man3/MPI_Win_sync.3.html), and
  [MPI_Win_unlock_all](https://docs.open-mpi.org/en/main/man-openmpi/man3/MPI_Win_unlock_all.3.html).
- [OpenMP API Specification 5.2](https://www.openmp.org/specifications/), OpenMP Architecture Review Board, 2021. Reference for `parallel for`, runtime scheduling, and the OpenMP runtime routines.
- [Matrix Market Exchange Formats](https://math.nist.gov/MatrixMarket/formats.html), National Institute of Standards and Technology. Reference for the coordinate input format, its one-based indices, fields, and symmetry conventions.

## Scientific literature

- P. Koanantakool et al., [Communication-Avoiding Parallel Sparse-Dense Matrix-Matrix Multiplication](https://doi.org/10.1109/IPDPS.2016.117), IPDPS 2016. Direct reference for distributed SpMM and for the central role of communication cost.
- G. Schubert et al., [Hybrid-parallel sparse matrix-vector multiplication with explicit communication overlap on current multicore-based systems](https://arxiv.org/abs/1106.5908), Parallel Processing Letters, 2011. Although it studies SpMV, it motivates comparing pure MPI and hybrid MPI+OpenMP and cautions against assuming that nonblocking communication automatically provides useful overlap.
- C. Lively et al., [Energy and performance characteristics of different parallel implementations of scientific applications on multicore systems](https://doi.org/10.1177/1094342011414749), The International Journal of High Performance Computing Applications, 2011. Relevant precedent for comparing MPI-only and hybrid MPI+OpenMP applications through strong- and weak-scaling experiments.

## Decisions applied here

- Sparse matrix `A` uses CSR and is distributed by contiguous row blocks.
- A Matrix Market coordinate file can be read by rank 0 and converted to CSR before timing starts. Only the required CSR blocks are then distributed to the other ranks.
- Each rank owns the corresponding row block of dense matrix `B`, and `B` is regenerated deterministically for every warmup and timed sample.
- The two-sided implementation preserves the existing halo semantics: each sample rebuilds the remote-row request plan locally and then exchanges row counts, row identifiers, and dense values with `MPI_Isend`/`MPI_Irecv`.
- The `one_sided_get` implementation exposes each rank's local block of `B` through a stable MPI window and uses a passive-target `MPI_Win_lock_all` epoch with `MPI_Get`, `MPI_Win_flush_all`, and `MPI_Win_unlock_all`. After local stores to the exposed window memory, the rank calls `MPI_Win_sync` before the barrier that releases remote gets.
- The `one_sided_put` implementation discovers remote row requests and target slots once, exposes a stable halo receive buffer through an MPI window, and refreshes that halo with `MPI_Put` in a passive-target epoch. After the put phase completes, a barrier plus `MPI_Win_sync` makes the received halo visible to local load accesses before the SpMM kernel starts.
- Setup, communication, computation, and gathering remain measured separately. For the RMA variants, `halo_setup_seconds` includes one-time plan discovery plus the initial halo population; the per-sample P90 metrics cover only the repeated halo refresh and computation phases.
- Each timed kernel sample is reduced to the maximum rank time, the results file stores the P90 across samples after untimed warmup iterations, and validation on rank 0 compares the gathered distributed result against a serial SpMM using the same dense-matrix epoch.
- All implementations keep the same OpenMP schedule interface (`guided` by default, with `static`, `dynamic`, `auto`, and `--chunk`), the same comparable TSV columns, and separate result directories under `results/`.

The corresponding BibTeX citations are in [`references.bib`](references.bib).
