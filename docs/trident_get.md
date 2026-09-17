# Trident CPU: staged inter-node GET

`trident_get` retains the 2D+1D partition, static-Cannon stage order, C-stationary
OpenMP kernel and two-sided intra-node B aggregation. Only the inter-node backend
changes: consumers pull the complete CSR slices of A and B with `MPI_Get`.
There are no request queues, application service threads or next-stage prefetch.
This is **not** an entirely one-sided implementation: intra-node aggregation and
collective synchronization still use the existing MPI mechanisms.

## Sources and reuse

Partitioning and multiplication follow the same Trident components mapped in
[the shared provenance](trident.md#source-provenance-and-reuse). The GET protocol
is a CPU experiment, not a claim that the original GPU code implements this
exact transport. Its normative references are MPI 4.1
[Get](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node318.htm),
[Flush and Sync](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node331.htm),
and [Semantics and Correctness](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node337.htm).

- `common/process_grid`, `matrix_blocks`, `execution` and `benchmark` remain shared.
- `common/intra_node_two_sided` assembles B inside each node unchanged.
- `get/inter_node_exchange` implements exposure, publication, retrieval and cleanup.
- `get/main_get.cpp` selects the backend and benchmark metadata.

Inter-node factories now receive the local input matrices during setup, allowing
GET to expose their existing storage. Other backends ignore these additional
arguments; their transport behavior is unchanged. The baseline GET halo remains
separate because it retrieves selected B rows, not Trident's A/B tiles.

## Protocol and lifetime

1. Setup exposes each local matrix's row pointers, column indices and values in
   three `MPI_Win_create` windows, six windows total. Passive-target
   `MPI_Win_lock_all` epochs stay open across products. No payload snapshot or
   extra input copy is allocated by the backend.
2. `begin` checks that input shapes, sizes and buffer addresses still match the
   exposed storage. `MPI_Win_sync` publishes any in-place input updates, then a
   world-communicator barrier makes every owner's current input ready for reads.
3. Each stage issues GETs for its remote A and B slices. It retrieves row
   pointers every time as well as indices and values; zero-length index/value
   payloads are skipped. Local sources use the same local copies as the other
   backends.
4. `MPI_Win_flush` on each remote source/window completes the current stage's
   reads before the unchanged intra-node aggregation and local product start.
   There is one workspace and no prefetch of the following stage.
5. `finish` runs a world barrier after all stage reads have been flushed, then
   synchronizes the local windows before returning. Thus no rank can change an
   exposed input while another rank still reads the previous product's data.
6. `close` explicitly unlocks and frees every window collectively. The input
   arrays outlive this cleanup. Destruction before `close` aborts instead of
   unwinding through buffers that might still be remotely accessible.

With only one node, all inter-node sources are local: window creation and these
two product-boundary barriers are skipped. Intra-node aggregation still runs.
A one-node run therefore does not exercise remote GET; use at least four
logical nodes for local transport tests or four physical nodes on the cluster.

Input values, column indices and row pointers may change **in place between
products**, preserving valid CSR, tile dimensions and nonzero counts. Do not
resize, reallocate, replace or destroy the exposed arrays before `close`.
To use new storage, close the backend and create a new one. The benchmark driver
itself keeps identical inputs throughout its repetitions.

This conservative version synchronizes at product boundaries, not between
individual stages. It requires only `MPI_THREAD_FUNNELED`; OpenMP compute threads
do not call MPI. Passive-target access does not imply guaranteed background
progress or free target-side work on every MPI implementation.

## Running

The normal CMake build produces `build/trident_get`. Use the **existing** scripts
with `VARIANT=get`; there are no new GET wrapper scripts:

```bash
VARIANT=get bash scripts/trident/run_local_check.sh

VARIANT=get NODES="1 4" RANKS_PER_NODE=2 THREADS=4 VALIDATE=0 \
  bash scripts/trident/run_strong_scaling.sh matrices/A.mtx matrices/B.mtx

VARIANT=get NODES="1 4" RANKS_PER_NODE=2 THREADS=4 VALIDATE=0 \
  bash scripts/trident/run_weak_scaling.sh
```

Physical launches bind `THREADS` cores per rank, as for two-sided. GET does not
request hybrid's additional service core. Default physical result files are
under `results/trident/trident_get/`; local checks use
`results/tmp/trident_get_local_check_v1.tsv`. A/B input rules, validation controls
and script options are unchanged; see the [script guide](../scripts/README.md).

## Measurements and limits

The TSV schema is unchanged, with `implementation=trident_get`,
`benchmark_protocol=trident_get_csr_v1`, `inter_node_transport=one_sided_get`
and `intra_node_transport=trident_two_sided`. Keep protocol results separate.

Setup includes window creation and locking. The first-product time includes
setup plus the first product. Repeated inter-node phase measurements include
publication, the two product-boundary barriers, GETs, flushes and local copies.
The ending barrier may include load imbalance and waiting for slower ranks;
this is not a pure network-transfer measurement. Use end-to-end product time as
the primary comparison, accounting for these protocol differences and hybrid's
different CPU budget. No claim of higher performance is made before cluster tests.

The existing limits remain: centralized input/gather, global A/B on rank 0 even
with `--no-validate`, int-sized CSR/MPI counts and per-row hash accumulators.
The GET backend additionally requires stable exposed input storage. A future
pipelined GET must add separate in-flight buffers and completion handling;
changing the MPI primitive alone would not provide overlap.

## Tests

The shared dense oracle covers signed, rectangular, empty and uneven matrices,
cancellation, repeated changed payloads, large transfers and delayed ranks on
1x1, 2x2 and 3x3 logical grids. GET-specific lifecycle tests change all three CSR
arrays of both inputs in place across 24 generations, without external barriers,
and compare fetched payloads exactly. They also reject invalid lifecycle calls,
changed shapes and different input buffers. CLI tests cover validation, numeric
arguments, source labels, topology rejection and protocol/transport metadata.
Script tests cover selection, inference, binding, result paths and failures.

Local Linux/WSL checks establish functional coverage, not physical network
performance. Multi-node execution, MPI progress and scaling remain to be tested
on the university cluster.
