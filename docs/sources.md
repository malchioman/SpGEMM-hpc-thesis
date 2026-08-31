# Sources for the two-sided baseline

This document records the sources used for the initial `MPI + OpenMP` two-sided
baseline. Standards, implementation documentation, and scientific literature
are deliberately kept distinct.

## Standards and official documentation

- [MPI: A Message-Passing Interface Standard, Version 4.1](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report.pdf), MPI Forum, 2023. Normative source for point-to-point communication, nonblocking operations, and MPI thread-support levels.
- [Open MPI MPI_Init_thread documentation](https://docs.open-mpi.org/en/v5.0.x/man-openmpi/man3/MPI_Init_thread.3.html). Operational reference for requesting and checking `MPI_THREAD_FUNNELED`; the [Open MPI MPI API manual pages](https://docs.open-mpi.org/en/main/man-openmpi/man3/index.html) cover `MPI_Isend`, `MPI_Irecv`, and `MPI_Waitall`.
- [OpenMP API Specification 5.2](https://www.openmp.org/specifications/), OpenMP Architecture Review Board, 2021. Reference for `parallel for`, runtime scheduling, and the OpenMP runtime routines.
- [Matrix Market Exchange Formats](https://math.nist.gov/MatrixMarket/formats.html), National Institute of Standards and Technology. Reference for the coordinate input format, its one-based indices, fields, and symmetry conventions.

## Scientific literature

- P. Koanantakool et al., [Communication-Avoiding Parallel Sparse-Dense Matrix-Matrix Multiplication](https://doi.org/10.1109/IPDPS.2016.117), IPDPS 2016. Direct reference for distributed SpMM and for the central role of communication cost.
- G. Schubert et al., [Hybrid-parallel sparse matrix-vector multiplication with explicit communication overlap on current multicore-based systems](https://arxiv.org/abs/1106.5908), Parallel Processing Letters, 2011. Although it studies SpMV, it motivates comparing pure MPI and hybrid MPI+OpenMP and cautions against assuming that nonblocking MPI automatically provides useful overlap.
- C. Lively et al., [Energy and performance characteristics of different parallel implementations of scientific applications on multicore systems](https://doi.org/10.1177/1094342011414749), The International Journal of High Performance Computing Applications, 2011. Relevant precedent for comparing MPI-only and hybrid MPI+OpenMP applications through strong- and weak-scaling experiments.

## Decisions applied here

- Sparse matrix `A` uses CSR and is distributed by contiguous row blocks.
- A Matrix Market coordinate file can be read by rank 0 and converted to CSR before timing starts. Only the required CSR blocks are then distributed to the other ranks.
- Each rank owns the corresponding row block of dense matrix `B`.
- Before the kernel, an explicit two-sided `MPI_Isend`/`MPI_Irecv` protocol exchanges the required remote rows of `B`. This is a halo exchange, not a global replication of `B`.
- The local SpMM kernel is parallelized across local rows with OpenMP. MPI is called only by the initial thread, therefore the program requests `MPI_THREAD_FUNNELED`.
- The default OpenMP schedule is `guided` with a configurable chunk size. `static`, `dynamic`, and `auto` remain available because the best schedule depends on matrix irregularity and hardware.
- Distribution, halo-plan setup, communication, computation, and result gathering are measured separately. Fresh remote values of `B` are exchanged for every timed sample. The current baseline rebuilds the request plan for each sample; a later optimization can reuse that plan while retaining the same communication semantics.
- Each timed kernel sample is reduced to the maximum rank time, and the results file stores the P90 across samples after untimed warmup iterations.

The corresponding BibTeX citations are in [`references.bib`](references.bib).
