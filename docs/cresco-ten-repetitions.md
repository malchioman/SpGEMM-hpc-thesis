# Ten repetitions without Reddit squared

The original binaries and frozen experiment runner from `tesi-003` are reused.
Each job runs the original cage8 pilot and all eight implementations on 9 square,
4 permuted and 10 rectangular pairs. Only `reddit.mtx * reddit.mtx` is excluded;
`reddit.mtx * reddit_restriction.mtx` remains included.

The proposed fresh campaign has 70 jobs: ten repetitions at each of
1, 4, 9, 16, 36, 64 and 100 physical nodes. Every job uses 2 MPI ranks per node,
63 compute threads and 64 allocated cores per rank, 450000 MiB per node,
exclusive allocation, and a 24-hour Slurm time limit. Sampling remains one
measured observation per executable launch, zero additional warmups, no main
validation, and the original guided schedule/chunk 64. All 24 pilot checks
validate correctness. Variant order rotates across repetitions exactly as in
the original ten-repetition planner.

There are 184 main rows and 24 pilot rows per job: 12880 main rows and 1680
pilot rows overall. Each observation stays in its own `rep-NN/nN` directory.
Existing observations are not counted toward these ten fresh repetitions.

Upload the current `scripts/extend_cresco_original.py` to the same path on the
cluster, then run the following on the CRESCO frontend when ready to submit:

```bash
bash <<'BASH'
set -euo pipefail
cd /lpor1/store_0/usr/malchiod/tirocinio-spgemm/SpGEMM-hpc-thesis
PY=.venv-sbatchman/bin/python
OUT=results/tesi-004/definitiva-001
"$PY" scripts/extend_cresco_original.py prepare \
  --root "$OUT" --node-counts 1 4 9 16 36 64 100 \
  --repetitions 10 --campaign tesi004 --after-current-extensions
"$PY" scripts/extend_cresco_original.py submit --root "$OUT"
BASH
```

The script checks one Slurm resource request per node count before submitting.
Repetitions form seven independent `afterok` chains: a failed repetition blocks
later repetitions for that node count, while other chains remain independent.
The first repetition follows any still-active `tesi003-ext-r01-nN` job of the
same size with `afterany`. Existing jobs are not cancelled or modified.
At most one job per chain runs, requiring 230 nodes if all seven are running.
Slurm QoS, priorities and other workloads may reduce the actual concurrency.

Submission receipts prevent accidental duplicate submissions. If a receipt
says the outcome of an earlier submission is uncertain, inspect the queue
before repairing that receipt. Successful rows survive a failure; do not
manually rerun an already-started job into the same output directory.

From the repository directory, inspect the campaign, including completed jobs:

```bash
.venv-sbatchman/bin/python scripts/extend_cresco_original.py status \
  --root results/tesi-004/definitiva-001
```

## Runtime estimate

The first-repetition archive supports reconstructed runtime estimates after
excluding Reddit squared of about 14 h 30 min at 1 node, 12 h 56 min at 4 nodes,
12 h 33 min at 9 nodes and 12 h 29 min at 16 nodes. These reconstruct complete
elapsed time from accounting and file timestamps; sums of benchmark product
timers exclude substantial other work and must not be used as job duration.

For 36/64/100 nodes, 12–16 hours per job is a provisional planning assumption,
not an observed result or upper bound. Ten repetitions in overlapping chains
would take roughly 6–7 days of execution before queue delays. The observed
`QOSGrpNodeLimit` prevents a reliable completion-date prediction. A practical
planning allowance is 1–2 weeks, with longer elapsed time possible if queues,
failures or the unmeasured large-node runtimes are worse than assumed.

Local checks: `python tests/cresco_extension_test.py -v`. Tests use a fixture
for the executable/TSV contract; they do not execute MPI performance measurements.
