# Validation and reference caches

Both campaign modes check the result from every MPI launch, after timed products
and gathering. Repeats provide timing samples; validation is performed on the
final gathered result, not independently on every internal sample. Cached
references avoid recomputing the sequential reference for every variant/node
count. They do not skip comparison of the current result.

| Mode | Comparison | Accepted status / criterion |
| --- | --- | --- |
| Full | Numerical values over the union of reference and result coordinates, including missing entries as zero, with finite-value and shape checks | `PASS` / `componentwise_dot_bound_v1` |
| Partial | Shape, finite values and exact stored-entry count in every row | `PARTIAL_PASS` / `row_nnz_exact_v1` |

Full forms a streamed sequential reference in `long double` and a coordinatewise
roundoff bound from the number and magnitude of contributing products. It checks
the candidate against that bound and records rejected coordinates, maximum error
ratio and reference precision. The numerical environment is checked; unavailable
or non-finite bounds do not authorize acceptance. This is a numerical correctness
criterion, not bitwise equality or an arbitrary absolute tolerance.

Partial computes the sequential numeric product row by row but retains only its
64-bit row counts. It does **not** compare column indices or values. Cancellation
and different accumulation orders can change whether tiny entries are stored:
HV15R square products are a known example where Partial can fail while Full
accepts the numerical result. Partial remains strict; the runner never converts
`PARTIAL_FAIL` into `PASS` or silently switches validation methods. Use Full when
the evaluation requires numerical validation, and report the criterion actually
used in the thesis.

## Cache lifecycle

`campaign_full.py` and `campaign_partial.py` prepare a separate reference job per
unique pair. The generators `prepare_full_reference` and
`prepare_partial_reference` use the same validation code as the benchmarks.

- Keys include input-content SHA-256 identities, the reference format/build and
  relevant numerical environment. A changed input or incompatible build needs
  a new reference.
- Full `.sfull` files store coordinatewise reference values and bounds; they can
  require substantial disk space. Partial `.spref` files store dimensions,
  finite status and row counts and are much smaller.
- Writers use atomic publication, integrity checks and exclusive construction
  locks. Interrupted preparation leaves no accepted partial cache file.
- Benchmarks require the matching reference to exist and verify it before use.
  Missing or corrupt data fails the launch; benchmarks do not fall back to
  expensive hidden sequential generation.
- Successful reference jobs leave readiness receipts. References are shared
  across all variants and resources. Frozen binaries and manifests prevent a
  later checkout update from changing an active campaign.

Validation diagnostics distinguish identity hashing, reference generation,
cache I/O and comparison costs. These costs are outside benchmark timing fields.
Caching preserves the comparison semantics, but disk reads, input loading,
gathering and memory pressure still contribute to total job duration. First
reference preparation must itself fit the cluster's resource and time limits.

Small Full pilots remain uncached and verify square, permuted and rectangular
products. A separate cached pilot checks the chosen cache path before scientific
measurements. Full/Partial cache unit tests, MPI cached/uncached comparisons and
HV15R cancellation regression fixtures remain part of CTest.
