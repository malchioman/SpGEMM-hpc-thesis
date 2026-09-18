# Trident CPU: first two-sided version

`trident_two_sided` implements the hierarchical 2D+1D partitioning and
C-stationary partial products described in Sections 3.2-3.3 (Algorithms 1-2) of
Bellavita et al., *Communication-Avoiding SpGEMM via Trident Partitioning on
Hierarchical GPU Interconnects*, ICS 2026. This is the CPU staged reference
variant, not a reproduction of every optimization in the GPU implementation.

The second CPU executable, `trident_hybrid`, reuses these components and changes
the inter-node request/response protocol. See [Trident hybrid](trident_hybrid.md)
for its RMA queues, service thread, resource requirements, and timing differences.
The third executable, `trident_get`, replaces inter-node payload exchange with
direct RMA reads while retaining two-sided intra-node aggregation. See
[Trident GET](trident_get.md) for its product-boundary synchronization and timing.
The fourth executable, `trident_get_pipeline`, reuses the GET transport with
double buffering and one-stage lookahead. See [pipelined GET](trident_get_pipeline.md).

## Source provenance and reuse

The upstream repository was inspected at commit
[`c37debaccc58b72859f1837f260900a40094848c`](https://github.com/HicrestLaboratory/Trident/tree/c37debaccc58b72859f1837f260900a40094848c).
The local adaptation follows these specific components:

| Upstream component | CPU component | Retained behavior and adaptation |
| --- | --- | --- |
| [`distributed_mmio` partitioning](https://github.com/HicrestLaboratory/distributed_mmio/blob/2e7c9b3c3205f4f019269b155f9381c8c4974b43/src/dmmio/partitioning.cpp), `edgeowner::groupowner`, `internodeidowner`, `edge2owner` | `common/process_grid.*`, `common/matrix_blocks.*` | Coarse 2D ownership and intra-node row slices. CPU code uses actual shared-memory groups, a coordinate-to-rank map, and balanced ranges instead of assuming consecutive world ranks or padding to equal tile sizes. |
| [`LocalSpGEMMTask` and `TaskQueue`](https://github.com/HicrestLaboratory/Trident/blob/c37debaccc58b72859f1837f260900a40094848c/include/task_queue.cuh) | `common/execution.*` | Coordinates `(i,j,k)`, stagger `h=(i+j+round)%q`, fixed input owners, and accumulation into a fixed C slice. A deterministic stage plan replaces request/task queues. |
| [`TileHolder::node_allgather`](https://github.com/HicrestLaboratory/Trident/blob/c37debaccc58b72859f1837f260900a40094848c/include/tile_holder.cuh) | `common/intra_node_two_sided.*` | Assemble all B row slices in node-rank order and reconstruct a complete CSR tile. Explicit `MPI_Irecv`/`MPI_Isend` replaces CUDA/NCCL or MPI collectives. |
| [`hns_spgemm_async`](https://github.com/HicrestLaboratory/Trident/blob/c37debaccc58b72859f1837f260900a40094848c/src/hns_spgemm.cu) | `common/execution.*`, `common/matrix_blocks.*` | Receive A/B, reconstruct B, multiply, accumulate. CPU/OpenMP replaces GPU kernels; no communication threads, request queues, work stealing, or overlap in this variant. |

This is algorithmic reuse with a CPU reimplementation, not a vendored copy or a
link against the original CUDA classes. Existing project code is reused directly
for CSR storage, Matrix Market input, options, balanced range partitioning, MPI
checks, P90, serial validation, and operation counting. No CUDA, NCCL, BCL, or
Kokkos installation is needed. Original GPU wrappers and device memory pools are
not suitable as direct dependencies for this MPI/OpenMP variant.

## Layout shared by future variants

```text
src/trident/
  common/
    process_grid.*   physical topology and logical test topology
    matrix_blocks.*  hierarchical distribution, gather, CPU accumulation
    execution.*      static-Cannon plan, backend interfaces, product driver
    csr_transfer.*   CSR resizing and two-sided payload helpers
    rma_get.*        shared GET input windows, issue/completion and synchronization
    intra_node_two_sided.*  shared node-local B aggregation
    benchmark.*      input/options, timing, validation, TSV output
  two_sided/
    inter_node_exchange.*
    main_two_sided.cpp
  hybrid/
    inter_node_exchange.*
    main_hybrid.cpp
  get/
    inter_node_exchange.*
    main_get.cpp
  get_pipeline/
    inter_node_exchange.*
    main_get_pipeline.cpp
```

`trident_support` depends only on the project's `spgemm_support` and includes the
shared two-sided intra-node backend. `trident_two_sided_support`,
`trident_hybrid_support`, `trident_get_support` and `trident_get_pipeline_support` each add their own inter-node
backend; hybrid also links the thread runtime. None depends on another variant's library or on
`spgemm_baseline_support` and its halo data structures.
`IntraNodeExchange::assemble` is the substitution point for future intra-node
GET or PUT aggregation protocols. All current versions explicitly supply an
`InterNodeExchange` (`begin/fetch/finish/close`) and reuse the same two-sided
intra-node backend. The product driver has no implicit two-sided fallback.
Grids, matrix ownership, input rules, local
accumulator, and benchmark driver remain shared. The two GET backends share
`GetTileTransport::startFetch/completeFetch`; the pipeline additionally uses a
second A/B pair and starts the next stage before intra-node aggregation and
computation of the current stage. The shared product loop remains unchanged.

## Partition and execution

For `N=q*q` nodes and `lambda` MPI ranks per node, coordinates are `(i,j,k)`.
OpenMP threads operate inside each rank and are not the third grid dimension.
The matrices may be rectangular: only `A.cols == B.rows` is required for
multiplication. The square constraint applies to the node grid, not the inputs.

1. Divide each matrix into `q*q` coarse tiles. Split each tile's rows among the
   `lambda` ranks in that node. Columns are local to the coarse column tile.
   Uneven and zero-length row/column blocks are allowed without padding.
2. In round `r`, node `(i,j)` uses inner block `h=(i+j+r)%q`.
   Rank `(i,j,k)` receives `A(i,h,k)` and `B(h,j,k)` from their static owners.
   Each owner sends its original slice to its unique consumer in that round.
3. All ranks in node `(i,j)` exchange their received B slices with nonblocking
   point-to-point messages, yielding the complete `B(h,j)` on each local rank.
   The A slices do not need an intra-node exchange.
4. Compute and accumulate `C(i,j,k) += A(i,h,k) * B(h,j)` with OpenMP over rows.
   C never moves during the product. Final CSR columns are sorted and exact
   zero sums are omitted, including cancellations between different stages.

The inter-node transport is **also two-sided**, implemented by
`TwoSidedInterNodeExchange` in `two_sided/inter_node_exchange.*`.
Its lifecycle hooks do not start workers or allocate communication resources;
all current-stage requests complete within `fetch`, before
the local product starts. There is no global barrier inside the product, but the
matching sends/receives couple stage progress across nodes. This does **not**
reproduce the original independently progressing request/response service or
communication/computation overlap. Keeping this choice explicit is necessary
when interpreting comparisons with the paper or later asynchronous variants.

## Running

Dedicated Open MPI scripts for strong scaling, weak scaling, and a localhost
correctness check are in `scripts/trident/`. See the
[script guide](../scripts/README.md) for node placement, environment variables,
dry runs, and separate result paths. Baseline scripts remain separate under
`scripts/baselines/`.

Build using the root CMake project. Example for one physical node:

```bash
mpirun -np 4 ./build/trident_two_sided \
  --rows 4096 --cols 3072 --b-cols 2048 \
  --nnz-per-row 16 --b-nnz-per-row 16 --threads 4 \
  --warmup 2 --repeats 10 --trials 5
```

Example with Open MPI on an allocation of four physical nodes, two ranks per node:

```bash
mpirun -np 8 --map-by ppr:2:node:PE=4 --bind-to core ./build/trident_two_sided \
  --matrix-a matrices/A.mtx --matrix-b matrices/B.mtx --threads 4 \
  --no-validate --results results/trident/cluster_two_sided.tsv
```

The cluster's MPI launcher and binding policy may require different launch
options. Normal execution discovers nodes with `MPI_Comm_split_type` and
`MPI_COMM_TYPE_SHARED`. All nodes must have the same rank count, and the node
count must be a perfect square. `--matrix-a A.mtx` alone computes `A*A`, requiring
square A; `--matrix-a A.mtx --matrix-b B.mtx` computes `A*B`. Matrix Market shape
checks, non-finite validation failures, and `--no-validate` retain their existing
semantics.

Local correctness test of the 2x2-node, two-rank-per-node path:

```bash
mpirun -np 8 ./build/trident_two_sided --logical-node-size 2 \
  --rows 19 --cols 23 --b-cols 13 --nnz-per-row 3 --b-nnz-per-row 2 \
  --threads 2 --warmup 1 --repeats 2 --trials 2 \
  --results results/tmp/trident_logical.tsv
```

`--logical-node-size` is accepted only when the whole job runs on a single
physical shared-memory node. It creates logical groups of consecutive ranks and
records `topology=logical_test`, even if there is just one logical node. These
runs verify the message paths; they do not measure inter-node network performance.

## Benchmark protocol and limits

Trident uses a separate TSV and `benchmark_protocol=trident_staged_csr_v1`.
The baseline protocol remains `prepared_halo_v2`. Trident's plan stores tile
shapes, nonzero counts, ownership, and per-stage intra-node offsets. Buffers are
reserved once. Every product transfers the full CSR payload, including row
pointers, indices, and values; it does not reuse remote numerical data. Only
metadata is cached, so all tile dimensions and nonzero counts must remain fixed
while a plan is reused. The driver always reuses identical inputs.

- `distribution_seconds`: initial slicing and distribution, excluding file input
  and process-grid construction.
- `plan_setup_seconds`: metadata exchange and buffer/backend preparation.
- `first_product_seconds`: setup plus the first complete multiplication;
  a single sample, excluding distribution.
- `inter_node_p90_seconds`, `intra_node_p90_seconds`: respective exchange phases,
  including waits, buffer operations and local copies (also for local sources).
  Inter-node timing includes backend `begin`/`finish` calls, which are no-ops in
  the two-sided version.
- `communication_p90_seconds`: P90 of each sample's maximum rank time for the
  sum of the two exchange phases. It is not the sum of the two separate P90s.
- `compute_p90_seconds`: CPU accumulator creation, all partial products,
  final CSR construction and accumulator destruction.
- `end_to_end_p90_seconds`: the whole product, excluding the sample's starting
  barrier and the timing reduction. No deliberate overlap is performed.
- `gather_seconds`: final reconstruction on rank 0, outside timed products.

Each sample uses the maximum rank duration before computing P90. As with the
baselines, communication and compute P90s need not sum to end-to-end P90.
Comparing this version with the 1D baselines measures the combined effects of
partitioning, payloads, local accumulation, and communication. It does not
isolate only the difference between MPI primitives. Comparing future Trident
backends should hold the shared configuration and kernel fixed.

By default, serial validation runs after gathering; non-finite results fail and
all ranks return a nonzero exit code. `--no-validate` reports `SKIPPED` and
`max_abs_error=NA`, but still gathers C and retains global A/B on rank 0.
Centralized input/gather, O(P) setup metadata, int-sized CSR/MPI counts, and
row-wise hash accumulators remain scalability limits to assess on the cluster.

## Tests

CTest includes the existing CLI/input/validation cases for this executable,
topology rejection and TSV checks, and an independent dense oracle for signed,
rectangular, empty, and cancellation cases. The product is repeated with changed
values and column indices using the same plan. Logical grids cover 1x1, 2x2 and
3x3 coarse grids; the 2x2 case includes two ranks per logical node. Test values
are small integers, so the dense comparison is exact.
A larger-message case also exercises payloads beyond common MPI eager thresholds.

Local Linux/WSL testing covers functional correctness, not physical multi-node
MPI performance. Real-node rank placement, scaling, and network costs must be
verified on the university cluster before reporting thesis performance results.
