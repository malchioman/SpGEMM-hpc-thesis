# Trident CPU: hybrid request/response version

`trident_hybrid` keeps the two-sided version's 2D+1D partitioning, static input
owners, staggered stage order, C-stationary OpenMP kernel, input rules, and
node-local B aggregation. It changes the **inter-node** protocol: consumers
publish requests through MPI RMA and owners respond with two-sided CSR payloads.
It is not an intra-node GET backend or the [pipelined GET version](trident_get_pipeline.md).

## Source and adaptation

The algorithmic basis remains Sections 3.2-3.3 of the Trident paper; see the
[shared provenance](trident.md#source-provenance-and-reuse). The protocol is
inspired specifically by the original
[`MessageQueue::notify/wait`](https://github.com/HicrestLaboratory/Trident/blob/c37debaccc58b72859f1837f260900a40094848c/include/message_queue.cuh)
and the request-serving threads in
[`hns_spgemm_async`](https://github.com/HicrestLaboratory/Trident/blob/c37debaccc58b72859f1837f260900a40094848c/src/hns_spgemm.cu).
This is a CPU reimplementation, not a copied CUDA queue or a complete port of
all original optimizations.

The original queue publishes messages to indexed remote slots. Here, a bounded
queue uses one slot per consumer and input type, and a monotonically increasing
64-bit product generation replaces resettable ready flags. Consumer identity is
implicit in the slot. This is sufficient because the static-Cannon plan requests
each remote owner's A or B slice exactly once per product; it is not a general
multi-producer FIFO or a work-stealing queue.

Publication uses `MPI_Accumulate(MPI_REPLACE)` followed by `MPI_Win_flush`.
The owner polls its own slots with `MPI_Fetch_and_op(MPI_NO_OP)` and flushes
before inspecting the returned value. These are compatible atomic accesses;
there are no concurrent ordinary C++ reads/writes of exposed window storage.
The protocol consequently does not depend on MPI's unified memory model.
See the normative [MPI 4.1 accumulate operations](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node320.htm).

## Execution and lifetime

1. Setup duplicates a payload communicator and allocates a request window with
   `2*q` slots per rank, where `q*q` is the number of nodes. Slots are initialized
   through RMA, in a passive-target `MPI_Win_lock_all` epoch.
2. Each product starts one service thread per rank when `q > 1`. It serves both
   A and B requests independently of the main thread's current compute stage.
3. For each stage, the main thread posts CSR receives before publishing requests.
   Self-owned inputs use local copies and do not enqueue requests.
4. The service thread sends the requested immutable original slice using the
   shared `MPI_Isend` payload helper and completes its sends. Row pointers,
   column indices, and values all travel two-sided, including empty slices.
5. After the requested inputs arrive, the main thread runs the unchanged
   two-sided intra-node aggregation and local multiplication.
6. At product completion it joins the service thread, draining all
   `2*(q-1)` remote responses owned by this rank before inputs can change.
   Generations distinguish consecutive products without resetting remote slots.
   Window/communicator cleanup is explicit and collective, after all workers
   have stopped; exceptions abort rather than unwind through collective cleanup.

There is no global barrier inside the product. A slow rank can still delay
consumers of its data or a local B aggregation. Joining the service thread also
waits for all consumers to request this rank's inputs. Normal benchmark barriers
between timed products remain in place.

Serving other ranks can overlap the local computation, but the main thread does
**not prefetch its next stage**. There is one stage workspace, no pipeline,
no work stealing, and no claim that this backend is faster. The CPU queue uses
atomic polling and the worker is created/joined for each product; their overhead
is part of what the comparison measures.

## Shared code and resources

- `common/process_grid`, `matrix_blocks`, `execution`, `csr_transfer`, and
  `benchmark` are shared by both versions.
- `common/intra_node_two_sided` is reused directly; no second copy is created.
- `hybrid/inter_node_exchange` implements request publication and service.
- `hybrid/main_hybrid.cpp` selects this backend and its benchmark metadata.

Both executables select explicit intra-node and inter-node backends. The hybrid
library depends on `trident_support` and the thread runtime, not on
`trident_two_sided_support`. Moving the shared aggregation and extracting the
two-sided inter-node backend does not change ownership, payloads or stage order.

Hybrid requires `MPI_THREAD_MULTIPLE`, checked at startup, because the service
thread and main thread may call MPI concurrently. The existing executable still
requests only `MPI_THREAD_FUNNELED`. `--threads T` continues to mean T OpenMP
compute threads, not T total threads including the service worker.

Physical-node scripts reserve `T+1` cores per hybrid rank and T for two-sided.
This binds the whole process to that CPU set; it does not pin the worker to an
exclusive core inside it. On one node no remote-service worker is needed, but
the hybrid scripts keep the same reservation policy for the entire sweep.
Account for this difference in experiments: equal compute-thread counts use
different core budgets. An equal-core-budget comparison should reduce hybrid's
compute threads by one (and report that choice).

## Running

Build with the existing CMake commands. To check both backends locally:

```bash
bash scripts/trident/run_local_check.sh
VARIANT=hybrid bash scripts/trident/run_local_check.sh
```

On an allocation of four physical nodes, two ranks per node, four compute
threads and one extra core per rank:

```bash
mpirun -np 8 --map-by ppr:2:node:PE=5 --bind-to core --nooversubscribe \
  ./build/trident_hybrid --matrix-a matrices/A.mtx --matrix-b matrices/B.mtx \
  --threads 4 --no-validate --results results/trident/hybrid_cluster_v1.tsv
```

Use the shared strong/weak scaling and local-check scripts with `VARIANT=hybrid`;
there are no hybrid-specific wrappers. See the [script guide](../scripts/README.md).
No scheduler or resource allocation is implemented by these scripts.

## Measurements and limits

Results use `implementation=trident_hybrid`,
`benchmark_protocol=trident_hybrid_rma_requests_v1`,
`inter_node_transport=hybrid_rma_requests_two_sided`, and
`intra_node_transport=trident_two_sided`. The two-sided executable retains its
existing protocol and TSV header. Keep different protocols in separate files;
an identical header alone does not make their timing semantics identical.

Setup includes request-window creation/initialization. The first product includes
setup plus execution. For repeated products, `inter_node_p90_seconds` measures
main-thread elapsed time for starting the worker, fetching inputs (including
request publication and receive waits), and joining/draining the service.
It is **not** a measurement of all network activity or the service thread's CPU
time. Background service may also run during the measured compute/intra-node
phases. The existing maximum-across-ranks then P90 aggregation is unchanged.
Use end-to-end time as the principal comparison, and interpret phase timings
with these differences in mind.

The unchanged limits include centralized input/gather, global inputs on rank 0
even with `--no-validate`, int-sized CSR counts, and metadata collected across all
ranks. Additional hybrid costs include `MPI_THREAD_MULTIPLE`, one polling
service worker per active rank, and serialized servicing by that worker.

## Tests

The shared dense-oracle suite runs for both backends on 1x1, 2x2 and 3x3 logical
node grids, including multiple ranks per node, empty and uneven tiles, signed
values, cancellation, changed payloads, and messages beyond common eager limits.
Hybrid tests also count exactly one request/response per remote consumer/input
per product. A skewed test executes 24 changed products without intervening
gathers/barriers, with artificial rank/stage delays, then validates every result.
CLI tests check topology, transport/protocol labels, validation and its optional
disablement. Shell tests cover variant selection, extra CPU binding, result paths
and failure propagation through the shared scripts.

Local Linux/WSL tests are functional checks. Real-node scaling, MPI implementation
behavior and the additional service-thread cost must be measured on the cluster.
