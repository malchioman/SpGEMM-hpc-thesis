# Matrices and workloads

`scripts/matrices_catalog.json` is the input catalogue: matrix IDs, source URLs,
dimensions, entry-count conventions and selection tiers. The default campaign
uses these ten main inputs:

| Source | Matrices |
| --- | --- |
| SuiteSparse collection | `mouse_gene`, `cage15`, `HV15R`, `dielFilterV3real` |
| NERSC HipMCL datasets | `archaea`, `eukarya`, `isolates_subgraph4`, `isolates_subgraph5` |
| NERSC Incremental-HipMCL | `uniparc` |
| NERSC GNN datasets | `reddit` |

`cage8` is the small correctness pilot. `cage12` and `web-Google` are optional
catalogue inputs, outside the default campaign. Source and algorithm references
are listed in [Sources](sources.md).

## Prepared products

For each selected square input A, `matrices.py prepare --seed 42` records:

- Original product `A × A`.
- Permuted product `(P A Pᵀ) × (P A Pᵀ)`, with the recorded deterministic seeded
  permutation. The default campaign selects HV15R, dielFilterV3real, cage15 and
  archaea for this experiment.
- Rectangular product `A × R`, using the deterministic restriction-matrix
  construction implemented in `matrices.py` and its recorded upstream reference.

The default plan excludes only the original Reddit square, leaving 23 products:
nine original squares, four permuted squares and ten rectangular products.
Permutation reorders the same problem; different input matrices are a descriptive
comparison, not a controlled change of just one structural property.

Original and derived files carry metadata and SHA-256 hashes under `metadata/`.
Preparation reuses compatible files and refuses to overwrite changed inputs.
Campaign preparation and execution verify those records before measurement.
Fetching is explicit; benchmark jobs never download or regenerate matrices.

Distinguish Matrix Market **stored entries** from the CSR entries after symmetry
expansion. Input CSR retains duplicate coordinates and explicit zeros; product
accumulation combines duplicates and removes exact zeros. Catalogue counts identify their
convention; TSV `a_nnz`, `b_nnz` and `c_nnz` report the actual CSR data used by the
executables. Structural exports include density, row statistics and rank work
counts. Use those fields when describing measured workloads.
