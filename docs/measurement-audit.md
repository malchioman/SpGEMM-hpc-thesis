# Measurement audit — 2026-10-02

Reviewed the repository's application code: shared CSR/input/validation helpers,
all three baseline drivers and their exchange kernel, all five Trident backends
and the shared driver, matrix preparation, experiment collection/exports, build
entrypoints and scheduler wrappers. Generated files and third-party SbatchMan
internals are outside this source review. This review includes the uncommitted
raw-sample, full-product and structural-metric additions on `refinements`.

## Defects corrected

| Finding | Effect | Correction and evidence |
| --- | --- | --- |
| Trident's new workload counter used binary search on input columns that need not be sorted | Wrong per-rank load and imbalance, even when the global operation count was correct | Count entries in each output column tile without assuming order. The failing rectangular case reported `[7,1,3,1]` instead of `[4,4,2,2]`; the regression now passes for all five backends and both tested rank layouts. |
| Input row bounds assumed unique coordinates; documentation claimed duplicate merging | Valid duplicate-bearing inputs could be rejected and stored-entry density misinterpreted | Preserve the current input representation, accept counts above the input column count, and document duplicates/explicit zeros. A hand-checkable product has 5 A entries, 7 B entries, 18 executed scalar products and 2 C nonzeros. |
| The collector did not reject non-finite C when serial validation was disabled; exports trusted source metrics too much | Invalid numerical results or inconsistent edited measurements could reach analysis tables | Add `result_finite`, scanned outside timers. Reuse metric checks at collection and export: sample count/values/P90, containing intervals, structure totals/statistics, validation status/error and GFLOP/s. Raw direct-launch `SKIPPED` output can report `result_finite=0`; campaign/analysis tables reject it. |
| Some baseline/serial CSR output counts were narrowed or added as unchecked signed ints | Large output matrices could overflow CSR offsets/counts | Check representability before narrowing/adding output counts and before relevant MPI transfers; check synthetic counts before allocation and widen synthetic index arithmetic. Boundary tests avoid allocating huge matrices. The format remains int-sized, not a 64-bit CSR implementation. |
| The C++ input reader ignored undeclared trailing entries and accepted malformed trailing fields | Direct launches could silently use only part of a malformed file | Reject extra records/tokens, invalid or non-finite values and nonzero skew-symmetric diagonals; align the Python skew-diagonal check. Preserve valid unsorted/duplicate/zero entries and Fortran exponent notation. |

## Timer and aggregation conclusions

- Each observation uses elapsed time measured on each rank with `MPI_Wtime`, then
  `MPI_MAX`. No subtraction between different ranks' clocks is used. The benchmark
  barrier precedes each timed repeated product; the reporting reduction follows
  its end timestamp.
- Warmups and the initial product are excluded from the repeated-product arrays.
  The arrays have `repeats * trials` entries in trial-major order. P90 uses nearest
  rank, `ceil(0.9 * N) - 1`, on those rank-maximum samples. `trials` do not restart
  MPI; `runs` do. An average of launch P90s is not an average of raw product times.
- `end_to_end_p90_seconds` measures repeated multiplication with reusable plans,
  including that backend's product communication/completion. It excludes loading,
  distribution, initial setup, final gather, validation and reporting.
- `full_product_seconds` is measured directly from initial distribution through
  the first gather. It includes intervening synchronization, setup and the first
  multiplication, but excludes input loading, process-grid creation, validation,
  reporting reductions and file writes. It is one observation per launch. The
  gathered first result is released before warmups.
- `first_product_seconds` already contains setup. Individual phase maxima and
  P90s cannot be summed to reconstruct a total: their maxima can belong to
  different ranks and repetitions. The collector checks containment, not an
  invalid equality between independently aggregated phases.
- Baseline communication includes packing/GET flush/PUT synchronization as
  applicable. Trident communication includes staging, copies, waits and backend
  `begin`/`finish`. Hybrid's worker activity and GET prefetch can overlap other
  phases; the phase numbers are not network-active time or hidden-communication
  percentages. Use total product time for the primary comparison.
- Baseline compute includes replacement of the previous result; Trident's local
  compute counter covers accumulator work and destruction while its outer total
  also covers result replacement. Cross-family phase comparisons therefore need
  their definitions; comparisons within each family have shared instrumentation.
- GFLOP/s is `2 * scalar_products / compute_P90 / 1e9`. It is a conventional
  arithmetic-throughput estimate, not hardware instructions, network throughput
  or throughput based on the whole product time.
- Structural metrics and finiteness scans run after the final timed gather.
  Work counts follow actual algorithm partitions and stored input entries;
  rank nnz arrays exclude halo buffers, staging and root-only global copies.

## Verification

Clean Release build: `build-wsl/review`, GCC 13.3, WSL/Open MPI, UCX over TCP/self
for local RMA checks. All eight benchmark targets and the C++ test targets built.

- 176/176 CTest entries passed (all entries except `experiment_scripts`, run
  separately). The optional test against the external SbatchMan package was
  skipped because that package is absent; the repository's scheduler/planner
  tests passed, and no jobs were submitted.
- `experiment_scripts_test.py`: 12/12 passed, including collection rejection,
  corrupted-source export rejection, repeated appends, output isolation and
  scheduler/direct-run table equivalence.
- `benchmark_measurements`: 40 MPI launches across the eight variants. Covers
  unsorted rectangular inputs on four/eight ranks, empty inputs, duplicates,
  explicit zeros and overflow with serial validation disabled. Checks exact
  rank-work expectations, raw samples, P90 and actual collection/analysis exports.
- Serial reference: 100 small integer cases compared with an independent dense
  triple-loop product, including zero dimensions, duplicates, unsorted indices,
  zeros and cancellation. Existing non-finite and tolerance tests also passed.
- Existing MPI tests covered logical 1/4/9-node layouts, multiple ranks per node,
  large messages, delayed Hybrid requests, skewed progress, repeated changing
  inputs and GET/PUT/pipeline window lifetime. Matrix tools: 15/15 passed.

## Limits relevant to the thesis

1. These are functional tests on one host. Physical-node performance, MPI/UCX
   transport, placement, actual CPU affinity, memory pressure and scaling still
   require checks on the university/target cluster. Baseline topology labels are
   inferred from the runner's launcher mapping; Trident checks its discovered
   process grid. Requested OpenMP threads/CPU budgets are not an affinity trace.
   Reserving T+1 cores does not separately pin Hybrid's worker to the spare core.
2. Validation compares the final timed C with the serial reference, not every
   timed repetition or the first gathered C. The protocol tests exercise repeated
   products separately. A `SKIPPED` finite result is not proven numerically correct.
   The existing strict absolute tolerance `< 1e-10` can reject legitimate rounding
   differences on large-magnitude inputs; it is not a relative-error test.
3. A/B remain global on rank 0 and C is gathered there. Memory capacity and int-sized
   CSR counts limit admissible matrices. This audit adds guards, not distributed
   I/O, distributed validation or a larger-index format.
4. Keep code/build, input contents, MPI stack, transport, affinity and sampling
   fixed within comparable campaigns. TSV rows have simple filenames and no build
   hash or environment fingerprint, by the current result-organization design.
   The runner checks current input hashes; it does not certify that historical
   rows were collected with the same environment. Archive old schemas/results
   before switching builds; headers alone cannot establish comparability.
5. Variant order is fixed within each repetition. Multiple independent launches
   estimate variability but do not remove temporal drift or interference. The
   analysis scripts export observations; they do not calculate confidence intervals
   or automatically select valid comparison groups.

No matrix-source URLs, algorithm protocols, result directory layout, commits,
pushes or cluster jobs were changed as part of this audit.
