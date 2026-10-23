# Experimental campaigns

## Configuration

Choose `scripts/campaign_full.py` for numerical validation or
`scripts/campaign_partial.py` for exact row-count validation. Both use cached
references and the same measurement code. Copy `scripts/campaign.json` to change
inputs, node counts, variants or scheduling before preparing a campaign.

| Default | Value |
| --- | --- |
| Physical nodes | 1, 4, 9, 16, 36, 64, 100, 121 |
| MPI ranks / node | 2 |
| OpenMP threads / rank | 63 |
| Reserved CPUs / rank | 64, including the Hybrid service thread |
| Internal samples | 10 repeats × 1 trial; zero warm-up iterations |
| Products | 9 original squares + 4 permuted squares + 10 rectangular products |
| Variants | All eight implementations |
| Accepted observations | 184 per node count; each row contains ten sample values |
| Slurm partition / limit | `cresco8_cpu` / 24 hours per job |
| Job size | At most 24 product–variant cases: 8 jobs per node count, 64 total |
| Reference preparation | 24 jobs: 23 products plus the cached `cage8` probe; at most 4 active chains |

`excluded_pairs` removes only `reddit.mtx × reddit.mtx`. Input files still include
Reddit for the rectangular experiment. A case is one product, one variant and
one node count. Repeats happen inside that MPI launch; they do not repeat input
loading, reference generation or the entire workflow.

`slurm` contains the partition, memory request **per node**, wall time, exclusivity
and module preamble. Change it for another cluster. Node counts must be perfect
squares. `cpus_per_rank` must cover the OpenMP team plus Hybrid's extra worker.
The runner owns rank mapping and core binding and checks the actual allocation.
Submission currently supports Slurm with Open MPI.

## Prepare and submit

Commands below run from the checkout on Linux. Existing matrices and their
metadata can be reused. To obtain inputs for the default plan:

```bash
MATRICES=/absolute/path/to/matrices
python3 scripts/matrices.py fetch --root "$MATRICES" --tier all
python3 scripts/matrices.py fetch --root "$MATRICES" --ids cage8
python3 scripts/matrices.py prepare --root "$MATRICES" --tier all --seed 42
python3 scripts/matrices.py prepare --root "$MATRICES" --ids cage8 --seed 42
```

Select one mode; use a new absolute output directory for each campaign:

```bash
MODE=full                         # or partial
ROOT="$PWD/results/campaigns/full-01"
python3 "scripts/campaign_$MODE.py" plan --config scripts/campaign.json
python3 "scripts/campaign_$MODE.py" prepare --config scripts/campaign.json \
  --root "$ROOT" --matrix-dir "$MATRICES" --build-jobs 8
```

Run `prepare` in an environment where compilation and input hashing are allowed
(on CRESCO, a compute allocation with the configured modules loaded). It checks
input provenance, copies sources and settings, compiles fresh Release binaries,
and writes hashed manifests and batch files. It submits nothing. Additional
compiler settings can be supplied as `--cmake-arg=-DNAME=VALUE`.

After preparation, **always use the frozen driver**, including after checkout
updates. Do not move the campaign root or its inputs while it is active.

```bash
DRIVER="$ROOT/metadata/source/scripts/campaign_$MODE.py"
python3 "$DRIVER" submit-references --root "$ROOT"
python3 "$DRIVER" status --root "$ROOT"
# Once every reference is ready:
python3 "$DRIVER" submit --root "$ROOT"
python3 "$DRIVER" status --root "$ROOT"
```

References run once per distinct input pair and are shared by all variants and
node counts. `--cache-dir /absolute/shared/cache` at preparation can reuse
compatible references across campaigns. Full and Partial use different cache
formats. Cache identity and checksum checks remain active on every use.

Each node count has a sequential job chain; different node counts may overlap
subject to scheduler limits. Jobs use `afterany`, so a failed block does not
permanently block later blocks. Small uncached Full pilots and cached probes
must pass before that configuration's measurements. Their outputs stay under
`checks/`, outside scientific tables. Reference generation and input reading
may still be expensive. A 24-hour request is a limit, not a completion guarantee;
choose `cases_per_job` using measured wall times and leave margin.

## Progress, resumption and outputs

`status` reports accepted cases and the last case started per block, followed by
Slurm accounting. A case is accepted only after the launch finishes, validation
passes, and all expected samples and metadata are checked.

```bash
python3 "$DRIVER" submit-references --root "$ROOT" --retry-failed
python3 "$DRIVER" submit --root "$ROOT" --retry-failed
python3 "$DRIVER" export --root "$ROOT"    # once all planned cases are accepted
```

Retry resubmits terminal incomplete jobs, skips accepted cases, and keeps the
same TSVs. Active jobs are never duplicated. An interrupted `sbatch` with an
ambiguous receipt requires checking Slurm accounting before any manual repair.
The campaign manifest and scientific configuration are immutable; changing the
validation method requires a separate campaign, not relabelling existing rows.
Retries use the original block layout. If an individual case exceeds the limit,
retrying unchanged cannot fix it; investigate its logs and resources first.

| Path under the campaign root | Contents |
| --- | --- |
| `nN/{strong_scaling,permutation,rectangular}/VARIANT.tsv` | Accepted scientific observations and sample arrays |
| `nN/{phase_analysis,matrix_structure}/VARIANT.tsv` | Derived exports; no new MPI runs |
| `metadata/`, `manifest.json` | Frozen sources, binaries, configuration, input provenance and hashes |
| `references/`, `reference-ready/` | Cached references (unless external) and readiness receipts |
| `checks/` | Pilot/probe data and diagnostics for failed launches |
| `logs/`, `state/`, `submission/` | Per-job logs, progress and submission history |

TSV publication is atomic and writers are locked per result directory. A timeout
preserves previously accepted rows; the interrupted launch is rerun. Shared
filesystem advisory locking must work on the target cluster. Historical
campaigns keep using their own frozen launchers; this interface does not migrate
or overwrite their results.

## What the timers mean

| Metric | Included work and interpretation |
| --- | --- |
| `distribution_seconds` | Initial distribution of input data |
| `halo_setup_seconds` / `plan_setup_seconds` | Baseline halo or Trident plan preparation |
| `first_product_seconds` | From the start of setup through completion of the first product; **already includes setup** |
| `first_gather_seconds`, `gather_seconds` | Gather of the first product, and final gather after the measured repeats, respectively |
| `full_product_seconds` | One interval from distribution start through the first gather, including setup and intervening synchronization |
| `end_to_end_samples_seconds` | Repeated prepared-product interval, excluding initial setup and gather |
| `communication`, `compute`, `inter_node`, `intra_node` sample arrays | Timed phases of the prepared product; availability depends on the variant |
| `*_p90_seconds` | Nearest-rank 90th percentile of the corresponding sample array |

File parsing, provenance hashing, cache access and validation are outside the
scientific timers. Initial costs and `full_product_seconds` have **one observation
per launch**, not ten independent samples. Internal samples share an allocation
and prepared state; they do not measure variability across independent jobs.
The first product runs even when warm-up is zero and precedes the sample loop.
Each reported interval/sample uses the maximum rank time. Consequently, summing
separate rank maxima does not reproduce the maximum of a complete interval.
Phase intervals may overlap, especially with GET Pipeline, and separate phase
percentiles must not be summed to reconstruct product time. Compare variants
using the prepared-product interval and use phase timings to explain differences.
Keep validation criterion and benchmark protocol labels in all analysis.
