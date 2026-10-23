# Distributed SpGEMM on CPUs

C++17, MPI and OpenMP implementations of sparse matrix multiplication for a
thesis on hierarchical partitioning and communication. CSR offsets and stored
entry counts are 64-bit; MPI transfers are chunked below the configured count
limit. Matrix dimensions and column indices retain their existing integer range.

## Implementations

| Executable | Partition and communication |
| --- | --- |
| `spgemm_two_sided` | Row distribution, prepared two-sided halo exchange |
| `spgemm_one_sided_get` | Row distribution, MPI GET halo exchange |
| `spgemm_one_sided_put` | Row distribution, MPI PUT halo exchange |
| `trident_two_sided` | Hierarchical 2D+1D partition, staged two-sided exchange |
| `trident_hybrid` | Hierarchical partition, RMA requests and service thread |
| `trident_get` | Hierarchical partition, direct RMA GET |
| `trident_get_pipeline` | GET with double buffering and one-stage lookahead |
| `trident_put` | Hierarchical partition, staged RMA PUT |

Trident requires a square number of physical nodes. All variants use the same
configured MPI rank and CPU budget; Hybrid's service thread is included in that
budget. The CPU adaptation and its sources are documented below.

## Build and test

Use Linux or WSL with CMake 3.21+, a C++17 compiler, MPI, OpenMP and OpenSSL
headers/libraries. Campaign tooling additionally requires Python 3.8+, Open MPI
and Slurm. Load the cluster's compiler/MPI modules before building.

```bash
bash scripts/build.sh --jobs 8
ctest --test-dir build --output-on-failure
```

`-DBUILD_TESTING=OFF` disables test targets. Numerical, MPI, cache and roundoff
regression tests remain available with the default build. Campaign-specific
historical tests have been replaced by one shared lifecycle test suite.

For a small direct executable check inside a suitable allocation:

```bash
mpirun -np 2 build/trident_get --rows 32 --cols 32 --b-cols 32 \
  --nnz-per-row 3 --b-nnz-per-row 3 --threads 1 \
  --warmup 0 --repeats 3 --trials 1 --results test-results/example.tsv
```

## Experimental campaigns

There are two public campaign entrypoints:

```bash
python3 scripts/campaign_full.py plan
python3 scripts/campaign_partial.py plan
```

Both use `scripts/campaign.json` and the same lifecycle: prepare a frozen build,
prepare cached references, submit measurements, inspect progress, resume missing
cases, and export tables. They differ only in validation mode. Full compares
numerical values with a roundoff-aware reference; Partial compares exact row
entry counts. Failed validation never becomes an accepted measurement.

The default configuration contains 23 products, eight variants, ten internal
samples, and 1, 4, 9, 16, 36, 64, 100 and 121 nodes. Only `reddit × reddit` is
excluded; the rectangular Reddit product remains. The full workflow, resource
settings and recovery command are in [Experimental campaigns](docs/experiments.md).

## Documentation

- [Campaign commands, outputs and timer definitions](docs/experiments.md)
- [Full/Partial validation and reference caches](docs/validation.md)
- [Input matrices and derived products](docs/matrix_selection.md)
- [Row-distributed baselines](docs/baselines.md)
- [Trident two-sided](docs/trident.md), [Hybrid](docs/trident_hybrid.md),
  [GET](docs/trident_get.md), [GET Pipeline](docs/trident_get_pipeline.md),
  [PUT](docs/trident_put.md)
- [Sources](docs/sources.md) and [bibliography](docs/references.bib)

`src/` contains the numerical implementations, `scripts/lib/` the shared
orchestration and TSV handling, and `tests/` the maintained correctness checks.
Generated inputs, results and caches belong outside tracked source files.
