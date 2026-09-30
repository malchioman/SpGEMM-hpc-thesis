# Resource orchestration with SbatchMan

SbatchMan requests resources, submits jobs and records their status/logs. The
existing experiment runner still handles MPI placement, OpenMP settings,
validation, independent repetitions and TSV exports. Both **Slurm** and
**OpenPBS/PBS Professional** use the same `scripts/experiments.json` settings.
The PBS profile uses `select`/`place`; it does not target Torque's `nodes:ppn`
syntax. Open MPI must have integration with the selected scheduler, and the
repository, build, matrices and output paths must be visible on all nodes.

## Prepare on the cluster

Run these commands from the repository root on Linux, using Python 3.10+ for
SbatchMan. The plan generator and allocation checks need only the standard
library and retain the experiment runner's Python 3.8+ support.

```bash
python3 -m venv .venv-sbatchman
source .venv-sbatchman/bin/activate
python -m pip install -r scripts/sbatchman/requirements.txt
sbatchman init                    # once per checkout

# Choose one profile and adapt it to the target cluster:
cp scripts/sbatchman/slurm.example.json scripts/sbatchman/slurm.local.json
# OR
cp scripts/sbatchman/pbs.example.json scripts/sbatchman/pbs.local.json
```

Edit the selected `.local.json` file:

| Key | Meaning |
| --- | --- |
| `cluster_name` | A distinct SbatchMan label for this cluster; also separates results |
| `partition` / `queue` | Slurm partition / PBS queue; `null` uses the site's default |
| `account` | Slurm account / PBS `-A`; `null` omits it |
| `walltime` | Positive `HH:MM:SS`, for the whole job, including all its MPI launches |
| `memory_per_node_mb` | Requested memory **per node**, in MB |
| `exclusive` | Slurm `--exclusive` / PBS `place=scatter:exclhost` |
| `module_init` | Optional absolute shell file to source before `module load` |
| `modules` | Compiler/MPI module names, e.g. the site's GCC and Open MPI modules |
| `python` | Python executable available on compute nodes; default `python3` |

The supplied hour/16 GiB examples are not estimates for the full ten-matrix
campaign. Confirm limits and size these values for the actual inputs. The
generator accepts partial profiles, filling omitted keys from the example.
Local profiles, generated plans, virtual environments and `SbatchMan/` state
are ignored by Git.

Build and prepare the inputs using the existing [script guide](../README.md).
Use the same compiler/MPI modules during build and execution. For the pilot:

```bash
bash scripts/build.sh --jobs 8
python3 scripts/matrices.py fetch --ids cage8
python3 scripts/matrices.py prepare --ids cage8 --seed 42
```

## Preview and generate jobs

Start with a one-node correctness pilot. Choose the command for your scheduler:

```bash
python3 scripts/prepare_sbatchman.py --scheduler slurm \
  --profile scripts/sbatchman/slurm.local.json --campaign pilot-001 \
  --action pilot --dry-run -- --nodes 1

python3 scripts/prepare_sbatchman.py --scheduler pbs \
  --profile scripts/sbatchman/pbs.local.json --campaign pilot-001 \
  --action pilot --dry-run -- --nodes 1
```

`--dry-run` prints the complete plan, without files, scheduler commands or MPI
launches. Remove it to generate files under
`scripts/sbatchman/generated/<campaign>/<cluster_name>/`. Generate on the cluster
checkout: snapshots contain absolute paths. A Windows preview is useful for
inspection, but regenerate after moving to Linux/the cluster.

Each plan contains `configs.yaml`, `jobs.yaml`, one frozen experiment JSON per
node count and one shell entrypoint per job. The `.yaml` files use JSON syntax,
which SbatchMan's YAML parser accepts; no YAML package is needed for generation.
Existing plan directories are never overwritten, so pending jobs retain their
parameters. Use a new campaign name for a changed plan or a new measurement series.

The generator prints the exact next commands. For an unchanged example Slurm
cluster label, they are:

```bash
sbatchman set-cluster-name example-slurm
sbatchman configure --file scripts/sbatchman/generated/pilot-001/example-slurm/configs.yaml --overwrite
sbatchman launch --file scripts/sbatchman/generated/pilot-001/example-slurm/jobs.yaml
sbatchman status
```

For PBS, use the generated directory and cluster label from the PBS profile.
Both schedulers use the same SbatchMan commands. Inspect the generated scheduler
templates in `SbatchMan/configs/<cluster_name>/` before submission.
The preview flag belongs to `prepare_sbatchman.py`; SbatchMan 1.0.8's `launch`
CLI has no `--dry-run` option.

After the pilot succeeds, prepare the full input set and generate the campaign:

```bash
python3 scripts/matrices.py fetch --tier all
python3 scripts/matrices.py prepare --tier all --seed 42
python3 scripts/prepare_sbatchman.py --scheduler slurm \
  --profile scripts/sbatchman/slurm.local.json --campaign scaling-001 \
  -- --nodes 1 4 --ranks-per-node 2 --threads 4
```

Switch to `--scheduler pbs --profile scripts/sbatchman/pbs.local.json` on PBS.
Options after `--` are the existing experiment options, including `--config`,
`--variants`, `--matrices`, `--runs`, `--build-dir` and `--results-dir`.
`--action` defaults to `all`: each allocation runs pilot, strong scaling,
permutation, rectangular products and the two exports in that order. A failed
pilot stops the rest of that job. Select `--action strong_scaling`, `permutation`,
`rectangular` or `pilot` to submit just one stage. Main campaigns preserve the
existing validation setting (disabled by default); the pilot always validates.

## Resource mapping and outputs

For `N` nodes, `R` MPI ranks/node, `T` OpenMP threads/rank and `C` CPUs/rank:

| Resource | Slurm | OpenPBS/PBS Pro |
| --- | --- | --- |
| Physical nodes | `--nodes=N` | `select=N`, `place=scatter` |
| MPI ranks | `--ntasks=N*R`, `--ntasks-per-node=R` | `mpiprocs=R` per chunk |
| CPU cores | `--cpus-per-task=C`, `--hint=nomultithread` | `ncpus=R*C` per chunk |
| Compute threads | Runner sets `OMP_NUM_THREADS=T` | Same, plus `ompthreads=T` |
| Memory | `--mem=<MB>M` per node | `mem=<MB>mb` per chunk |

`C` defaults to `T+1` for every implementation, preserving the common budget
including Hybrid's service worker. Node counts remain positive perfect squares
for Trident. The PBS profile assumes the site's `ncpus` represents physical
cores; confirm that convention on machines exposing hardware threads as CPUs.
The generated wrapper loads modules with fail-fast shell behavior, then checks
Slurm's node/task/CPU environment or PBS's per-host rank entries in
`PBS_NODEFILE`. PBS does not expose a portable CPU-budget environment variable;
the generated `select` request supplies that budget. Open MPI then discovers the
allocation through its scheduler integration. A login-shell invocation without
an allocation fails before launching any benchmark.

One scheduler job is generated per node count. Jobs are sequential by default
to avoid competing with each other for the network; SbatchMan uses `afterany`
dependencies, so a failed job does not suppress later node-count jobs. Within
each job, MPI launches are serial. This is not a build/download pipeline:
prepare binaries and inputs before submitting.

Measurements are isolated by campaign, cluster and resource configuration:

```text
results/campaigns/scaling-001/<cluster_name>/n4-r2-t4-c5/
  strong_scaling/<variant>.tsv
  permutation/<variant>.tsv
  rectangular/<variant>.tsv
  phase_analysis/<variant>.tsv
  matrix_structure/<variant>.tsv
```

Pilot checks and failed-launch diagnostics under the repository's `results/`
tree are mirrored under `test-results/`, as in the existing runner. Custom
result roots outside `results/` retain that root. Tables from distinct node-count
jobs stay separate; there is no automatic merge. Existing analysis scripts can
be rerun with `--results-dir` pointing at one job's result directory. Repeating
a job with the same result root appends observations, including any successful
observations from a previous partial run; it is not an automatic resume.
SbatchMan also suppresses identical submissions unless forced/archived.

This integration targets SbatchMan 1.0.8. References:
[configuration format](https://sbatchman.readthedocs.io/en/latest/learn/configurations/),
[job submission](https://sbatchman.readthedocs.io/en/latest/learn/job_submission/),
[upstream source](https://github.com/LorenzoPichetti/SbatchMan).

Run the resource/allocation regression tests with:

```bash
python3 tests/sbatchman_scripts_test.py
```

With the SbatchMan requirements installed, the same test command also exercises
the upstream YAML parser and generated submission scripts for both schedulers,
mocking only the final submission calls. Otherwise that optional check is skipped.
