#!/usr/bin/env python3
"""Offline executable-contract fixture; does not compute or simulate MPI timing."""
import csv
import json
import math
import os
from pathlib import Path
import sys

PROTOCOLS = {
    "spgemm_two_sided": "prepared_halo_v2",
    "spgemm_one_sided_get": "prepared_halo_v2",
    "spgemm_one_sided_put": "prepared_halo_v2",
    "trident_two_sided": "trident_staged_csr_v1",
    "trident_hybrid": "trident_hybrid_rma_requests_v1",
    "trident_get": "trident_get_csr_v1",
    "trident_get_pipeline": "trident_get_pipeline_csr_v1",
    "trident_put": "trident_put_staged_csr_v1",
}
args = sys.argv[1:]
def value(flag):
    return args[args.index(flag) + 1]

if os.environ.get("MOCK_LAUNCH_LOG"):
    with open(os.environ["MOCK_LAUNCH_LOG"], "a", encoding="utf-8") as stream:
        stream.write(json.dumps(args) + "\n")
failure = os.environ.get("MOCK_FAILURE")
if failure == "exit":
    print("Intentional launcher failure", flush=True)
    sys.exit(7)
variant = next(Path(arg).name for arg in args if Path(arg).name in PROTOCOLS)
trident = variant.startswith("trident_")
ranks = int(value("-np"))
logical = "--logical-node-size" in args
rpm = int(value("--logical-node-size")) if logical else (
    int(value("--map-by").split(":")[1]) if "--map-by" in args else ranks)
nodes = ranks // rpm
row = dict(implementation=variant if trident else variant.replace("spgemm_", "mpi_openmp_", 1),
           experiment=value("--experiment"), matrix_a_source=value("--matrix-a"),
           matrix_b_source=value("--matrix-b"))
for prefix, flag in (("a", "--matrix-a"), ("b", "--matrix-b")):
    with open(value(flag), encoding="ascii") as stream:
        header = next(line for line in stream if not line.startswith("%"))
    n, m, nnz = map(int, header.split())
    row.update({prefix + "_rows": n, prefix + "_cols": m, prefix + "_nnz": nnz})
row.update(c_nnz=4, ranks=ranks, threads_per_rank=value("--threads"),
           omp_schedule=value("--schedule"), omp_chunk=value("--chunk"),
           warmup=value("--warmup"), repeats=value("--repeats"), trials=value("--trials"),
           distribution_seconds="0.1", first_product_seconds="0.9",
           communication_p90_seconds="0.2", compute_p90_seconds="0.3",
           end_to_end_p90_seconds="0.4", gather_seconds="0.1",
           max_abs_error="NA" if "--no-validate" in args else "0",
           validation="SKIPPED" if "--no-validate" in args else "PASS",
           benchmark_protocol=PROTOCOLS[variant])
if trident:
    row.update(nodes=nodes, ranks_per_node=rpm, grid_rows=math.isqrt(nodes), grid_cols=math.isqrt(nodes),
               topology="logical_test" if logical else "physical", plan_setup_seconds="0.2",
               inter_node_p90_seconds="0.15", intra_node_p90_seconds="0.05", compute_gflops="0.01")
else:
    row.update(halo_setup_seconds="0.2", compute_gflops_p90="0.01")
if failure == "protocol":
    row["benchmark_protocol"] = "wrong_protocol"
elif failure == "validation":
    row["validation"] = "FAIL"
elif failure == "nan":
    row["end_to_end_p90_seconds"] = "nan"
elif failure == "topology" and trident:
    row["nodes"] = 999
path = Path(value("--results"))
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(row), delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerow(row)
    if failure == "duplicate":
        writer.writerow(row)
