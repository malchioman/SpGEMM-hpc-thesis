# CPU protocol comparison: experiment plan

This document records the experiments, agreed result organization and matrix
selection. The [script guide](../scripts/README.md) describes the six stages, the
complete-campaign launcher and the build command. Resource settings are shared in
`scripts/experiments.json` and remain to be finalized on CRESCO-8.

## Proposed experiments

The main dataset consists of the ten matrices listed below under `all`. The table
is a proposed execution plan; feasibility and physical-node counts will be checked
once CRESCO-8 access is available.

| Experiment or analysis | Operation and purpose | Proposed matrices | New runs needed? |
| --- | --- | --- | --- |
| Functional pilot | Validate input preparation and all eight implementations on small square, permuted and rectangular products | `cage8`; optionally `cage12` for an initial resource check | Yes, before the main campaign; excluded from scaling conclusions |
| Strong scaling and overall comparison | Fixed `A*A`, increasing physical-node counts; compare total product time, speedup and efficiency | All ten: `HV15R`, `mouse_gene`, `archaea`, `eukarya`, `isolates_subgraph4`, `isolates_subgraph5`, `cage15`, `uniparc`, `reddit`, `dielFilterV3real` | Yes |
| Phase analysis and protocol comparison | Compare inter-node, intra-node and local-compute times; examine setup/first-product costs separately | The same ten matrices; highlight `mouse_gene`, `HV15R`, `eukarya`, `isolates_subgraph4` and `cage15` to illustrate different behavior | No; use the timings recorded by the scaling runs, and later the other products |
| Effect of input structure | Compare protocol behavior across application families, sparsity and connectivity patterns at matched resources | All ten, using the original `A*A` runs | No; this is a cross-matrix analysis of the same measurements |
| Effect of permutation | Compare `A*A` with `(P*A*P^T)^2` at the same node counts and fixed seed 42 | `HV15R`, `dielFilterV3real`, `cage15`; `archaea` as a biological-network comparison | Yes for the four permuted inputs; reuse the original runs |
| Rectangular products | Compute `A*R` with the deterministic sparse rectangular R described below, and compare protocol behavior as nodes increase | All ten original A matrices, each paired with its own compatible R | Yes |

The core execution plan therefore has 24 products: ten original squares, four
permuted squares, and ten rectangular products, each evaluated at the selected
resource configurations. Phase and structure analyses do not multiply the number
of products. Preparing additional inputs is optional; a generated manifest does
not mean that every case must be run.

Compare all five Trident variants on the same inputs and resources. Use the three
row-distributed baselines (two-sided, GET and PUT) as external references, with
baseline two-sided versus Trident two-sided as the main overall comparison.
The comparison within Trident addresses protocol differences under a shared
partitioning and local kernel. Comparing different real matrices is descriptive;
it does not isolate structure alone, since dimensions, values and density also
change. The original/permuted pairs provide the controlled reordering experiment.

Weak scaling remains an optional extension outside this main plan. There is no
thread-count sweep. Compute threads and ranks per physical node stay fixed.

## Workloads and matrix selection

The checked-in [catalogue](../scripts/matrices_catalog.json) records matrix identities,
dimensions, entry-count conventions, and selection rationale. These are a starting
selection, not a promise that every squared product fits on the available machine.

| Tier | Matrix | Rows = columns | Catalogue nnz | Purpose |
| --- | --- | ---: | ---: | --- |
| pilot | [cage8](https://sparse.tamu.edu/vanHeukelum/cage8) | 1,015 | 11,003 | Small correctness and workflow check; too small for cluster scaling conclusions |
| core | [cage12](https://sparse.tamu.edu/vanHeukelum/cage12) | 130,228 | 2,032,536 | Manageable polymer graph from the same family as cage15; original/permuted comparison |
| core | [web-Google](https://sparse.tamu.edu/SNAP/web-Google) | 916,428 | 5,105,039 | Directed web graph, complementary connectivity to the polymer matrices |
| all | [mouse_gene](https://sparse.tamu.edu/Belcastro/mouse_gene) | 45,101 | 28,967,291 | Biological matrix used in the Trident paper; substantially denser input than the other candidates |
| all | [cage15](https://sparse.tamu.edu/vanHeukelum/cage15) | 5,154,859 | 99,199,551 | Exact large polymer input from the paper |
| all | [HV15R](https://sparse.tamu.edu/Fluorem/HV15R) | 2,017,169 | 283,073,458 | Exact structured CFD input used for the paper's permutation study |
| all | [dielFilterV3real](https://sparse.tamu.edu/Dziekonski/dielFilterV3real) | 1,102,824 | 89,306,020 | Symmetric finite-element electromagnetics matrix |
| all | [archaea](https://portal.nersc.gov/project/m1982/HipMCL/archaea/) | 1,644,227 | 204,792,654 | Protein similarity network |
| all | [eukarya](https://portal.nersc.gov/project/m1982/HipMCL/eukarya/) | 3,243,106 | 359,763,936 | Protein similarity network |
| all | [isolates_subgraph4](https://portal.nersc.gov/project/m1982/HipMCL/subgraphs/) | 4,372,771 | 264,799,194 (stored) | Large isolate subgraph |
| all | [isolates_subgraph5](https://portal.nersc.gov/project/m1982/HipMCL/subgraphs/) | 2,186,385 | 66,399,522 (stored) | Smaller isolate subgraph |
| all | [uniparc](https://portal.nersc.gov/project/m1982/Incremental-HipMCL/) | 2,856,197 | 29,621,573 | Protein similarity network |
| all | [reddit](https://portal.nersc.gov/project/m1982/GNN/) | 232,965 | 57,307,946 (stored) | Social graph |

Source metadata checked on 2026-09-28. SuiteSparse's **matrix dimensions** for
web-Google include 916,428 rows; the separate SNAP graph statistics mention fewer
active vertices. The catalogue uses the Matrix Market dimensions.
The two core matrices are complementary CPU workloads, not claimed to be the exact
GPU evaluation set. The `all` tier covers all ten names and counts in Table 2 of the
[Trident paper](https://arxiv.org/html/2603.21444v1); `pilot` and `core` remain
separate auxiliary sets. Sparsity alone does
not determine SpGEMM cost: row degrees, connectivity and output fill-in also matter.

**Stored versus expanded counts:** SuiteSparse counts and the general-storage
NERSC matrices use `nnz_kind=expanded` (the default). The two `isolates` sources and
`reddit` declare `symmetric` storage, and Table 2 matches their **stored** entry
counts. Those catalogue entries explicitly use `nnz_kind=stored`. The downloader
preserves the original header and values, validates the declared count, and records
both `entries` and `expanded_entries` in `metadata/<id>.source.json`. Our reader and preparation
expand each off-diagonal symmetric entry; expanded counts can therefore be larger
than the paper's table. The C++ reader preserves expanded input entries, including
duplicate coordinates and explicit zeros; it does not sort or merge them. The
product kernels accumulate their contributions. In particular, the isolates sources contain entries
in both triangles. Matching names and stored counts does not establish identical
symmetry/duplicate handling in the GPU experiments. Do not silently relabel these
files as `general` or interpret the table count as the CPU benchmark's final nnz.
The alternative NERSC files named `propermm` for isolates have different counts and
are deliberately not used as interchangeable mirrors.

Start with the pilot, then measure core inputs. Activate paper inputs individually
after checking available memory and pilot runtime. The current C++ readers build
temporary entry arrays; rank 0 retains global A/B and gathers C, even with
`--no-validate`. The size of C and temporary accumulators can dominate memory.
Catalogue nnz and successful preparation are not a memory admission test.

## Three cases per square input

`scripts/matrices.py prepare` produces a manifest describing:

1. **Square:** `C = A*A`, preserving original values and ordering.
2. **Permuted:** `A' = P*A*P^T`, then `C' = A'*A'`. The same permutation is applied
   to rows and columns, so mathematically `C' = P*(A*A)*P^T`. The seed defaults to
   42; it is independent of ranks and nodes. Floating-point summation order may
   change. The generator uses a specified SplitMix64/Fisher-Yates algorithm, records
   its version and seed, and preserves all input entries. Symmetric storage is
   expanded to general storage before writing the permuted matrix.
3. **Rectangular:** `C = A*R`, with a deterministic synthetic 1D full-weighting
   restriction matrix. For order `n`, R has `n` rows, `2*n-1` columns, and `3*n-2`
   entries. In zero-based coordinates, row `i` has weight `0.5` in column `2*i`,
   `0.25` in column `2*i-1` when `i>0`, and `0.25` in column `2*i+1` when `i<n-1`.
   Missing endpoint weights are not renormalized. R has no random seed.

The R formula follows the authors' [1D generator at commit c37debac](https://github.com/HicrestLaboratory/Trident/blob/c37debaccc58b72859f1837f260900a40094848c/amg_matrices/multigrid_solver_matrix_generator.cpp)
and [generation script](https://github.com/HicrestLaboratory/Trident/blob/c37debaccc58b72859f1837f260900a40094848c/amg_matrices/generate_restrictors.sh).
This tool implements that mathematical stencil independently. The paper's section
5.4 describes `A*R`; one [upstream launch script at that revision](https://github.com/HicrestLaboratory/Trident/blob/c37debaccc58b72859f1837f260900a40094848c/scripts/make_scripts_restrict.py)
instead places the transposed restriction operator on the left. Our campaign
explicitly uses `A*R`, with a rectangular result, and does not claim to reproduce
all original launch details. R is a synthetic transfer-operator workload; it is not
constructed by solving an AMG coarsening problem on A.

Permutation and rectangular products are separate experiments: the rectangular case
uses original A and R. Download, transformation, hashing and input parsing all occur
outside the product timers. Preparation streams entries and stores only O(n) indices
for permutation; no dense matrix or SciPy dependency is needed.

## Download and preparation

Use Python 3.8+ (standard library only) on Linux/WSL; the input tool also runs on
Windows. From the repository root:

```bash
python3 scripts/matrices.py list
python3 scripts/matrices.py fetch --ids cage8
python3 scripts/matrices.py prepare --ids cage8 --seed 42

# Once the pilot works:
python3 scripts/matrices.py fetch --tier core
python3 scripts/matrices.py prepare --tier core --seed 42

# Download all ten paper matrices:
python3 scripts/matrices.py fetch --tier all

# Preparation is separate and creates additional files:
python3 scripts/matrices.py prepare --tier all --seed 42

# Or select one paper input:
python3 scripts/matrices.py fetch --ids mouse_gene
python3 scripts/matrices.py prepare --ids mouse_gene --seed 42
```

The downloader's network timeout defaults to 300 seconds per blocking socket
operation, not per complete download. `fetch --timeout 0` disables this timeout;
a finite value lets an unresponsive source fail so the next configured mirror
can be tried. It does not change the scheduler's job walltime or make an invalid
URL available.

Original matrices and synthetic R inputs live directly in `matrices/`, following
the input directory of the [earlier PARCO project](https://github.com/malchioman/PARCO-Computing-2026-244944#repository-layout).
Permuted matrices are grouped in `matrices/permuted/`.
For example, downloading and preparing cage8 produces:

```text
matrices/
  cage8.mtx
  cage8_restriction.mtx
  permuted/
    cage8_permuted_s42.mtx
  metadata/
    cage8.source.json
    cage8_s42.cases.json
```

`fetch` creates the original matrix and its source record; `prepare` creates the
permuted matrix in `permuted/`, R, and the three-case manifest. The seed is part of the permuted
filename, while R is shared across seeds. The original matrix is not duplicated.
Archives and temporary extraction directories are cleaned up after download.
`matrices/` is ignored by Git; the catalogue stays in `scripts/`.
`--root /shared/path/matrices` changes the input directory for both commands.
Manifests use relative paths, so moving the complete directory including `metadata/`
preserves the links. A completion record is written last; an original file without
its source record is reported as incomplete and is never silently overwritten.

The downloader uses SuiteSparse archives for four paper matrices, trying the current
server then the historical official University of Florida mirror if unavailable.
For the other six, it downloads direct Matrix Market files from NERSC. It extracts
only the selected regular `.mtx` file from archives, checks format, indices, finite
values, dimensions and the catalogue's stored/expanded count convention, and records
source URL plus download/input SHA-256 (also archive SHA-256 for tarballs). These are observed hashes,
not publisher-supplied checksums. Completed inputs are reused only after checking
their recorded hash. Preparation records hashes of derived inputs too and refuses
to reuse changed files. Partial downloads/preparations never become completed inputs.

The download and preparation tools are separate from the MPI benchmarks; no network
access is required during a campaign. They replace the old project's embedded
download approach with a small standard-library command-line tool.

## Shared execution conditions

Choose physical-node counts after CRESCO-8 access and a memory/runtime pilot.
The current Trident process grid requires perfect-square node counts, such as
`1, 4, 9, 16`. These are candidate configurations, not an allocation request.
If an input does not fit on one node, use the smallest feasible common configuration
as the scaling reference and report it. A one-node run does not exercise inter-node
communication.

Use the same compute-thread count, ranks per node, matrix files, MPI implementation,
OpenMP schedule and sampling settings across variants. Reserve the same total CPU
budget per rank for every implementation, accounting for Hybrid's service worker;
for example, T compute threads and T+1 bound cores for every variant. Do not reduce
Hybrid's compute-thread count to make this comparison. Binding, MPI modules,
allocation details and repetitions remain to be finalized on the target system.

## Agreed result organization

Reserve `results/` for thesis measurements and their analysis exports.
Functional checks and diagnostics belong in the separate, Git-ignored
`test-results/` directory:

```text
results/
  strong_scaling/
  phase_analysis/
  matrix_structure/
  permutation/
  rectangular/

test-results/
  pilot/
  local-check/
  tmp/
```

The standard pilot is routed to `test-results/pilot/`, including when invoked by
`run_all.sh`. A whole physical-cluster rehearsal must use an explicit
`--results-dir test-results/<check-name>` for all of its stages. Local logical
checks use `test-results/local-check/` by default. Existing check files are
retained until deliberately moved or removed; they are not promoted to thesis
measurements just because validation passed.

Each directory contains one TSV per implementation: `spgemm_two_sided.tsv`,
`spgemm_one_sided_get.tsv`, `spgemm_one_sided_put.tsv`, `trident_two_sided.tsv`,
`trident_hybrid.tsv`, `trident_get.tsv`, `trident_get_pipeline.tsv`, and
`trident_put.tsv`. Each TSV collects all matrices and configurations for that
implementation in that experiment. The directory identifies the experiment and
the filename identifies the implementation; no additional case column is required.

Repeated executions append rows to the same TSV, including executions on later
dates. Each row represents one execution; preserve the individual observations
rather than replacing earlier rows with an average. The experiment tables
do not require timestamps or execution/session identifiers. CRESCO-8 is the fixed
target environment for the campaign.
Rows identify both input matrices by filename, plus the resource configuration
(nodes, ranks, ranks per node and compute threads) and relevant measurement
settings. Permuted filenames include the seed. Existing protocol/validation
metadata remains available to interpret the measurements.

Phase and structure tables are derived from the recorded executions without
repeating the calculations. Regenerate these derived tables from the accumulated
source rows, preserving one derived observation per relevant source row, so rerunning
an export does not count the same execution twice. Equal values in separate source
rows are legitimate repeated observations, not a reason to deduplicate them.

For later averages, group only matching input pairs and execution/measurement
configurations. Different matrices, node counts, thread counts or measurement
protocols must not be pooled merely because they share a TSV. Use the same code,
MPI environment and sampling settings for comparable repetitions. Running on the
same machine does not eliminate runtime variability, which is why repetitions
remain useful; dates are not needed to calculate their mean.

Timed product fields include P90 statistics and the individual samples for each
execution. The `*_samples_seconds` columns contain JSON arrays of length
`trials * repeats`, ordered first by trial and then by repetition. Entry
`trial * repeats + repeat` (zero-based) is the MPI maximum for that timed product;
warmups and the first product are excluded. Baselines have communication, compute
and end-to-end arrays; Trident also has inter-node and intra-node arrays. Each row
still represents one independent MPI launch, so samples remain grouped by launch.
Scripts verify sample counts, finite nonnegative values and agreement with the
reported P90, and retain the arrays in experiment and analysis exports.

The arithmetic
mean across repeated executions is a **mean of per-execution P90 values**, not the
mean of individual product times or the P90 of pooled samples. This is a descriptive
summary of repeated run-level estimates, not a way of reconstructing the overall
percentile. Use the same number of samples per execution and label the statistic
explicitly; report the number of executions and variability across their P90 values
alongside the mean. Use the saved arrays to calculate other within-launch statistics
or pooled percentiles, retaining the distinction between independent launches and
their internal repetitions. Averaging quantiles does not recover a pooled percentile
(see the [Prometheus explanation](https://prometheus.io/docs/practices/histograms/)
and [NIST percentile definition](https://www.itl.nist.gov/div898/handbook/prc/section2/prc262.htm)).
Retaining samples does not change the steady-state timers or their P90 aggregation.

## Reading the existing measurements

`full_product_seconds` is a direct MPI-maximum elapsed time from the start of
input distribution through setup, the first multiplication and collection of that
first result. Input file loading, rank-grid construction, reporting reductions,
validation, warmups, subsequent products and file output are excluded. Necessary
in-path synchronizations are included. The first gathered result is released
before warmup; `first_gather_seconds` records this collection separately, while
`gather_seconds` still measures collection of the final timed result. This is one
full-product observation per launch, not a P90 or a sum of phase maxima. Compare
it separately from the steady-state product timings. The additional first gather
changes the pre-warmup execution path; keep old and new campaigns separate.

Use end-to-end product P90 as the primary performance comparison. For each variant,
strong-scaling speedup is its own reference time divided by its time at the larger
configuration; efficiency divides that speedup by the corresponding resource ratio.
Compare Trident protocols directly at matched resources rather than equating their
individual scaling speedups with an absolute runtime ranking.

The current instrumentation is retained. Within Trident, inspect inter-node,
intra-node and local-compute phase P90 values alongside the total. Initial
distribution, setup/first product and final gather remain separate; first-product
time already includes setup. Do not add independently aggregated phase P90 values
or present them as an additive runtime breakdown.

Hybrid and GET Pipeline phase timings measure main-thread work and waiting, not
all concurrent network activity or the percentage of hidden communication. Use
GET versus GET Pipeline total product time to assess the practical pipeline benefit.
The row-distributed baselines have different phase definitions; compare them with
Trident primarily through end-to-end product time, and interpret their phases
within their respective algorithms.

### Matrix structure and assigned work

Every observation, including phase and structure exports, retains the following
metrics. They are computed after the final timed gather, even with validation
disabled, and do not add counters to the measured multiplication loops.

| Fields | Meaning |
| --- | --- |
| `a/b/c_row_nnz_min`, `a/b/c_row_nnz_max`, `a/b/c_mean_nnz_per_row` | Minimum, maximum and mean CSR nonzeros per row of each global matrix |
| `a/b/c_row_nnz_stddev`, `a/b/c_row_nnz_cv`, `a/b/c_empty_rows` | Population standard deviation, coefficient of variation (standard deviation / mean), and count of empty rows; empty rows participate in all statistics |
| `rank_a_nnz`, `rank_b_nnz`, `rank_c_nnz` | JSON integer arrays of locally owned nonzeros, in MPI rank order; their sums equal the corresponding global nnz |
| `rank_scalar_products`, `scalar_products` | JSON integer array of scalar multiplications assigned to each rank for one product, and its global sum |
| `rank_work_min`, `rank_work_max`, `rank_work_mean`, `rank_work_stddev`, `rank_work_cv` | Population statistics of the assigned scalar-product counts, including ranks with zero work |
| `rank_work_max_over_mean`, `idle_ranks` | Maximum / mean assigned work, and number of ranks assigned zero scalar products |

Here `a/b/c_...` denotes three separate columns, for example `a_row_nnz_cv`.
Input statistics refer to stored CSR entries after symmetry expansion, including
duplicates and explicit zeros, not the number of distinct nonzero coordinates.
Consequently input row counts can exceed the number of columns, and an input
`*_density` above 1 is possible; it describes stored entries / matrix size. Compare
these representation characteristics alongside the application family. C is the
final measured result with unique sorted columns and zero sums removed, including the effect
of numerical cancellation on stored nonzeros. With zero mean, CV and maximum/mean
are reported as 0; positive uniform work gives maximum/mean 1.

Assigned work is calculated exactly from the sparsity pattern and the partition:
for each stored A(i,k), count the B(k,j) entries in the rank's owned output rows
and columns. Baselines use contiguous output-row blocks; Trident uses the actual
`(i,j,k)` process-grid mapping, including the row split within each node. The total
is checked against the independent global scalar-multiplication count. The usual
GFLOP/s convention counts two arithmetic operations per scalar product. These
are structural work counts, not hardware instruction counters or measured rank
durations; hashing, allocation, cache effects and communication can still create
runtime imbalance even when counts are balanced.

The nnz arrays count owned matrix data, excluding remote copies, staging buffers
and root-only gathered copies. They do not measure communication bytes or peak
memory. Row variability and assigned-work imbalance help interpret strong scaling,
permutation and rectangular products, but do not by themselves explain every
performance difference. Matrix-structure exports also retain the existing global
densities. No extra MPI products are run for these analyses.

The regression test `benchmark_measurements` runs all eight implementations on
a hand-checkable rectangular product with unsorted columns using four and eight
ranks, empty matrices, duplicate coordinates and explicit zeros. It also verifies
that overflow to infinity is rejected by the collector even with serial validation
disabled. It checks numerical validation, complete sample arrays, P90 agreement,
timer bounds, row statistics and exact per-rank work counts. Trident uses logical
nodes here; this verifies the instrumentation, not physical-cluster performance.

`result_finite` records whether every stored value in the final C is finite. This
scan is outside all product timers and does not require a serial reference. The
experiment collector and analysis exports reject `result_finite=0`, inconsistent
P90/sample arrays, phase samples longer than their containing interval, incorrect
GFLOP/s and incompatible validation status/error values. A finite result marked
`SKIPPED` is still not a numerically validated result. `PASS` uses the existing
strict absolute tolerance of `1e-10`, which can reject legitimate rounding
differences on large-magnitude inputs; this is not a relative-error criterion.

## Matrix-tool verification

```bash
python3 tests/matrix_tools_test.py
ctest --test-dir build -R '^matrix_tools$' --output-on-failure
```

The offline checks cover permutation equivalence (including squared products),
stencil weights, symmetric inputs, malformed files, archive selection, provenance
hashes, catalogue completeness, mirror fallback, stored/expanded counts, reuse,
conflicting files and manifest portability. Numerical C++/MPI tests remain available
through the root CMake project.
