# Trident CPU: staged inter-node PUT

`trident_put` is the fifth CPU variant, with one-sided inter-node CSR payloads
and the unchanged two-sided intra-node B aggregation. It retains Trident's
2D+1D partition, static-Cannon stage order and C-stationary OpenMP kernel.
There is no next-stage prefetch, double buffering, service worker or pipeline.

## Sources and shared code

Algorithmic provenance follows [Trident CPU](trident.md#source-provenance-and-reuse).
The staged push protocol is a CPU experiment, not a reproduction of the GPU
repository's request queues. MPI 4.1 provides the normative references for
[Put](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node317.htm),
[Flush and Sync](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node331.htm),
and [RMA semantics](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node337.htm).

`put/inter_node_exchange.*` implements the transport and its synchronization;
`put/main_put.cpp` selects it. Grid discovery, distribution, execution plan,
product loop, intra-node exchange, accumulation, validation and output remain
in `common`. No variant depends on another variant's library.
The inter-node factory now also receives `ProductWorkspace&`, allowing PUT to
expose the existing receive arrays during setup. Other factories ignore that
additional argument; their protocols are unchanged.

## Stage protocol

Setup sizes the workspace's A/B arrays for the largest planned stage and creates
six windows over their row pointers, column indices and values. The windows
remain in passive-target `MPI_Win_lock_all` epochs until collective `close`.
Unlike GET, PUT exposes the receivers, not the original input matrices.

In round r, an owner `(i,j,k)` pushes its original A slice to
`(i,(j-i-r) mod q,k)` and its original B slice to `((i-j-r) mod q,j,k)`.
These are the existing plan's `aTarget` and `bTarget`, inverse to its pull
sources. Each receive array has exactly one writer in each round.

1. `begin` validates the input shapes and workspace binding, then resets the
   stage counter. It issues no transfers.
2. `fetch(r)` requires the next stage of the bound plan and the bound workspace.
   It resizes the receive arrays within their fixed allocations and copies any
   locally owned input into the corresponding receiver.
3. Window syncs followed by a world barrier establish that every rank has
   finished consuming the previous stage and prepared its next receive arrays.
   This also separates successive products; fast ranks cannot overwrite data
   that slow ranks still use.
4. Each owner issues `MPI_Put` for the full CSR payload of each remotely needed
   input, then flushes all three corresponding windows at that destination.
   Local destinations do not use RMA. Empty tiles still transfer row pointers;
   zero-length indices and values require no PUT.
5. A second world barrier follows every rank's outgoing flushes. Window syncs
   then make incoming writes visible before the receiver reads them. A barrier
   alone is neither RMA completion nor public/private memory synchronization.
6. Only now does `fetch` return. The shared driver assembles B inside the node
   and computes the current partial product. No future-stage payload is issued.
7. `finish` checks that all stages were consumed. There are no remaining writes
   to drain and no additional product-boundary barrier is needed.

Thus the inter-node **payload is one-sided**, but this initial version uses
collective control synchronization: two barriers per stage, not an asynchronous
notification protocol. With one physical or logical node, inter-node sources
are local, so the backend skips windows, flushes, syncs and both barriers.
Intra-node exchange still runs when that node contains multiple MPI ranks.

## Storage and lifetime

The immutable plan and persistent workspace must outlive the backend and its
collective `close`. Workspace A/B vectors must not be moved, swapped, shrunk in
capacity or reallocated while exposed. Their logical lengths can vary between
stages within the maximum allocation; resizing occurs only before the readiness
barrier, when no remote write can race with it. The backend checks storage
addresses/capacities and rejects an unrelated workspace or out-of-order stages.
Unexpected unclosed destruction aborts instead of freeing live window storage.

There is one receive pair and one assembled B panel, with no extra payload copy
after reception. The existing intra-node assembly still copies B into its panel.
Original inputs must retain the planned shapes and CSR sizes and remain unchanged
throughout each product. Valid same-shape value/index/row-pointer updates between
products are supported. Input allocations may be replaced between products:
unlike GET, they are not exposed in windows. Metadata is reused, not payloads.

## Measurements and running

PUT requests `MPI_THREAD_FUNNELED`, makes MPI calls only on the main thread and
uses `THREADS` compute cores per rank, without an additional service core.
Window creation belongs to setup; stage resizes, local copies, PUTs, flushes,
syncs and both barriers belong to the inter-node product timing.

Compare this first version primarily with simple GET, holding the inputs,
partition, rank/thread layout and MPI settings fixed. This compares complete
push and pull protocols, not just the cost of `MPI_Put` versus `MPI_Get`:
PUT's two barriers per stage differ from GET's two per product. Compare with
GET pipeline separately to avoid conflating direction and overlap. End-to-end
product time is the primary metric; no speedup is claimed without cluster data.
Centralized input/gather, root A/B storage even without validation, and integer
CSR/MPI count limits remain unchanged.

The TSV schema is unchanged, with `implementation=trident_put`,
`benchmark_protocol=trident_put_staged_csr_v1`,
`inter_node_transport=one_sided_put` and `intra_node_transport=trident_two_sided`.
Use the existing scripts, without new wrappers:

```bash
VARIANT=put bash scripts/trident/run_local_check.sh

VARIANT=put NODES="1 4" RANKS_PER_NODE=2 THREADS=4 VALIDATE=0 \
  bash scripts/trident/run_strong_scaling.sh matrices/A.mtx matrices/B.mtx

VARIANT=put NODES="1 4" RANKS_PER_NODE=2 THREADS=4 VALIDATE=0 \
  bash scripts/trident/run_weak_scaling.sh
```

Default physical results are under `results/trident/trident_put/`; local checks
use `results/tmp/trident_put_local_check_v1.tsv`. One script invocation runs one
selected variant. See the [script guide](../scripts/README.md) for configuration.

## Tests

The shared independent dense oracle checks rectangular/signed products, uneven
partitions, empty and zero-sized inputs, cancellation, changed payloads across
repeated products, reversed communicator ranks, large messages and skewed ranks.
PUT lifecycle tests use PMPI to inspect actual PUT destinations and the ordering
of sync, readiness barrier, PUT, flush, completion barrier and sync. They check
that no next stage is posted, receiver addresses remain stable, delayed readers
retain their payload, and all three CSR arrays are fresh across 24 generations.
Invalid lifecycle, input shape, workspace and stage-order calls are rejected.
Local logical grids exercise the protocol, not physical network scaling; real
multi-node correctness and performance must also be checked on the cluster.
