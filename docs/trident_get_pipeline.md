# Trident CPU: pipelined inter-node GET

`trident_get_pipeline` retains Trident's 2D+1D partition, static-Cannon order,
C-stationary CPU/OpenMP kernel and two-sided intra-node B aggregation. It issues
`MPI_Get` reads for stage r+1 before consuming stage r, using two alternating
pairs of A/B buffers, and defers `MPI_Win_flush` until the incoming data are
needed. The objective is to measure the benefit and memory cost of prefetch
relative to simple GET while keeping the transport primitives common. There is
no application service thread, work stealing, intra-node RMA or pipeline across
different products.

## Sources and shared code

The algorithmic provenance is the same as [Trident CPU](trident.md#source-provenance-and-reuse).
This is an experimental CPU lookahead schedule, not a claim that the original
GPU repository uses the same implementation. The RMA references are MPI 4.1
[Get](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node318.htm),
[Flush and Sync](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node331.htm),
and [Semantics and Correctness](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node337.htm).

Both GET variants use `common/rma_get.*` (`GetTileTransport`): the same six input
windows, passive-target lock epochs, full-CSR transfers and product-boundary
synchronization. Its `startFetch` issues a pair of reads; `completeFetch` flushes
them. Only one pair may be pending. Simple GET calls both consecutively in
`fetch`, preserving its original staged behavior and protocol identifier.
The pipeline defers completion until that stage is needed. Neither variant
depends on the other variant's library.

The existing product driver still calls `begin/fetch/finish`. Lookahead stays
inside `get_pipeline/inter_node_exchange.*`; distribution, plan generation,
intra-node communication, accumulation and benchmark output remain shared.
`get_pipeline/main_get_pipeline.cpp` selects this backend and its metadata.

## Prefetch and completion

Simple GET issues and completes the current stage's reads before intra-node
aggregation and computation. Pipelined GET completes the current reads, issues
the next stage's reads into the other buffer pair, and proceeds with the current
stage's aggregation and computation before completing the prefetched reads.
Partitioning, stage order, local kernel, input windows and product-boundary
synchronization remain common. The comparison therefore studies the effect of
prefetch scheduling and double buffering with the same RMA transport.

Return from `MPI_Get` does not certify that the destination buffer is ready to
read. The current implementation establishes completion with `MPI_Win_flush`,
which completes all outstanding RMA operations issued by the calling process
to the specified target on that window, at both origin and target. The buffer
is consumed only after this completion, as detailed below.

An alternative is `MPI_Rget`, which returns an `MPI_Request` for each operation.
Waiting on that request, or successfully testing it for completion, establishes
that the fetched data are available in the origin buffer. Both approaches can
support prefetch with double buffering; they differ in completion granularity
and may have different costs. This implementation uses `MPI_Get` with deferred
flushes. See MPI 4.1's
[request-based RMA operations](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node325.htm)
and [flush semantics](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node331.htm).

## Schedule and lifetime

1. Setup reserves one additional A/B pair, sized for the largest stage, and
   exposes the original local inputs through the shared GET transport.
2. `begin` publishes inputs with window syncs and the same barrier as simple GET,
   then issues the first stage's GETs into the incoming pair without flushing.
3. `fetch(r)` completes those reads before any consumer accesses their contents.
   It swaps the completed incoming pair with the workspace's A/B pair. This
   swaps vector ownership, without copying the received payload.
4. If stage r+1 exists, `fetch(r)` immediately starts its GETs into the now-unused
   pair. It returns without flushing those new reads.
5. The unchanged driver assembles B inside the node and computes stage r while
   r+1 has been issued. The next `fetch` completes r+1 and repeats the swap.
6. The last stage has no speculative successor. `finish` requires every stage
   to have been consumed, then runs the same reader-completion barrier and
   window syncs as simple GET. All reads are complete before inputs can change.

Only the A/B input buffers are doubled: there is still one assembled B panel and
one C accumulator. All buffers are reused across products. Local input sources
are copied synchronously; they do not issue GETs. The split-phase helper rejects
overlapping fetches, completion without a pending fetch, and finish with pending
reads. The pipeline rejects skipped, repeated and out-of-order stages.

The backend binds an immutable execution plan; `fetch` must receive the actual
stage objects from that plan in order. The plan must outlive the backend. Use
one persistent product workspace. Input arrays must keep their addresses, sizes
and tile shapes until collective `close`, as in [simple GET](trident_get.md).
Valid in-place CSR updates between products remain supported. Pending incoming
arrays are never exposed to the kernel or resized before completion. The RMA
transport aborts on unclosed destruction before its incoming arrays are destroyed.

With one node there is only one stage: there is no next stage to prefetch, and
the shared GET transport skips windows and product-boundary barriers. At least
four logical or physical nodes are needed to exercise remote lookahead.

## Progress and measurements

This version requests `MPI_THREAD_FUNNELED`, with MPI calls only on the main
thread, and uses the same CPU-core budget as simple GET. No progress worker or
MPI-specific asynchronous-progress setting is enabled automatically.

Posting a GET before computation permits overlap but does not establish that
transfers actually progress during OpenMP computation on a particular MPI/network
stack. The intervening intra-node MPI calls may also contribute to progress.
Replacing these calls with `MPI_Rget` would not by itself guarantee background
progress during computation either. Actual overlap and any performance benefit
require measurement on the cluster; see MPI 4.1's
[RMA progress discussion](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node340.htm).

The TSV schema is unchanged, with `implementation=trident_get_pipeline`,
`benchmark_protocol=trident_get_pipeline_csr_v1`,
`inter_node_transport=one_sided_get_pipeline` and
`intra_node_transport=trident_two_sided`. Setup includes window creation and
extra buffer reservation. Product timing includes priming, swaps, GET issue,
remaining completion waits and both boundary barriers.

`inter_node_p90_seconds` measures main-thread time inside the backend, not total
network-transfer duration: prefetched reads can progress during the intra-node
or compute intervals. Neither the phase times nor their differences directly
measure hidden communication. Use end-to-end product time as the primary
comparison against simple GET, holding inputs, topology, threads and MPI settings
fixed, and report the extra memory for one A/B pair. Existing centralized input,
gather and integer-count limits remain unchanged.

## Running and tests

The CMake build produces `build/trident_get_pipeline`. Use the direct-launch
examples in [Trident CPU](trident.md#running), selecting that executable.
The [experiment plan](experiments.md) describes the comparisons; the
[script guide](../scripts/README.md) covers the shared campaign commands and TSVs.

The shared dense oracle covers rectangular, signed, uneven, empty and zero-sized
inputs, cancellation, repeated changed payloads, delayed ranks and large messages.
Pipeline lifecycle tests use PMPI to observe real GET/flush calls: the next
stage must already be issued when `fetch` returns, its completion must remain
deferred, and its destinations must be separate from the current workspace.
They check two-buffer reuse, final drain, invalid lifecycle/order calls and all
three CSR arrays across 24 changed products on 1x1, 2x2 and 3x3 logical grids.
These are functional checks, not proof of network overlap or physical scaling.
