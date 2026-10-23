# Row-distributed baselines

Each MPI rank owns contiguous row blocks of A and B and computes its corresponding
rows of C. It obtains the remote B rows referenced by its local A entries before
multiplication. The shared OpenMP kernel parallelizes output rows using a
Gustavson-style accumulator.

| Executable | Required B-row exchange |
| --- | --- |
| `spgemm_two_sided` | Owners pack rows and exchange payloads with `MPI_Isend` / `MPI_Irecv` |
| `spgemm_one_sided_get` | Consumers fetch payloads from owners' CSR windows with `MPI_Get` |
| `spgemm_one_sided_put` | Owners push payloads into consumers' halo windows with `MPI_Put` |

All three use `prepared_halo_csr64_v3`: requests, row lengths, offsets and buffers
are prepared once; column indices and values are transferred again for every
product. RMA epochs span the repeated products. Communication timings include
packing and synchronization where applicable. The local multiplication and
validation code are shared.

Unlike Trident's hierarchical 2D+1D partition, these baselines distribute rows
across all MPI ranks directly. They provide algorithmic context for the scaling
study. Comparing the Trident transports to each other isolates communication
strategy more closely than comparing either partition to the other.

For a direct check with already prepared inputs, replace the executable name
below to select another baseline:

```bash
mpirun -np 2 build/spgemm_two_sided --matrix matrices/cage8.mtx \
  --threads 1 --warmup 0 --repeats 3 --trials 1 \
  --results test-results/baseline.tsv
```

Use the common [campaign interface](experiments.md) for matched resource settings,
cached [validation](validation.md), immutable builds and accepted-result tables.
See [Sources](sources.md) for the algorithm and MPI references.
