# Trident CPU: pipelined inter-node GET

`trident_get_pipeline` retains Trident's 2D+1D partition, static-Cannon order,
C-stationary CPU/OpenMP kernel and two-sided intra-node B aggregation. It changes
the inter-node GET schedule: retrieve stage r+1 while consuming stage r, using
two alternating pairs of A/B buffers. There is no application service thread,
work stealing, intra-node RMA or pipeline across different products.

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

This version uses the same `MPI_Get` and `MPI_Win_flush` primitives as simple
GET, not `MPI_Rget`; the controlled change is their scheduling and double
buffering. It requests `MPI_THREAD_FUNNELED`, with MPI calls only on the main
thread, and uses the same CPU-core budget as simple GET. No progress worker or
MPI-specific asynchronous-progress setting is enabled automatically.

Posting a GET before computation permits overlap but does not establish that
transfers actually progress during OpenMP computation on a particular MPI/network
stack. The intervening intra-node MPI calls may also contribute to progress.
Actual overlap and any performance benefit require measurement on the cluster.

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

The CMake build produces `build/trident_get_pipeline`. Use the existing scripts:

```bash
VARIANT=get_pipeline bash scripts/trident/run_local_check.sh

VARIANT=get_pipeline NODES="1 4" RANKS_PER_NODE=2 THREADS=4 VALIDATE=0 \
  bash scripts/trident/run_strong_scaling.sh matrices/A.mtx matrices/B.mtx

VARIANT=get_pipeline NODES="1 4" RANKS_PER_NODE=2 THREADS=4 VALIDATE=0 \
  bash scripts/trident/run_weak_scaling.sh
```

No new wrapper scripts are added. Default physical results are under
`results/trident/trident_get_pipeline/`; local checks use
`results/tmp/trident_get_pipeline_local_check_v1.tsv`.

The shared dense oracle covers rectangular, signed, uneven, empty and zero-sized
inputs, cancellation, repeated changed payloads, delayed ranks and large messages.
Pipeline lifecycle tests use PMPI to observe real GET/flush calls: the next
stage must already be issued when `fetch` returns, its completion must remain
deferred, and its destinations must be separate from the current workspace.
They check two-buffer reuse, final drain, invalid lifecycle/order calls and all
three CSR arrays across 24 changed products on 1x1, 2x2 and 3x3 logical grids.
These are functional checks, not proof of network overlap or physical scaling.
