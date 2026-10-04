#!/usr/bin/env python3
"""Shared Linux/Open MPI experiment runner and lossless TSV exports (Python 3.8+)."""
import argparse
import contextlib
import csv
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import matrices  # noqa: E402

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
RUN_ACTIONS = ("pilot", "strong_scaling", "permutation", "rectangular")
ANALYSES = ("phase_analysis", "matrix_structure")
IDENTITY = [
    "matrix_a", "matrix_b", "a_rows", "a_cols", "b_rows", "b_cols",
    "a_nnz", "b_nnz", "c_nnz", "nodes", "ranks", "ranks_per_node",
    "threads_per_rank", "cpus_per_rank", "topology", "omp_schedule", "omp_chunk",
    "warmup", "repeats", "trials", "benchmark_protocol",
]
TIMINGS = [
    "distribution_seconds", "halo_setup_seconds", "plan_setup_seconds",
    "first_product_seconds", "inter_node_p90_seconds", "intra_node_p90_seconds",
    "communication_p90_seconds", "compute_p90_seconds", "end_to_end_p90_seconds",
    "gather_seconds", "full_product_seconds", "first_gather_seconds",
]
SAMPLE_FIELDS = [phase + "_samples_seconds" for phase in
                 ("inter_node", "intra_node", "communication", "compute", "end_to_end")]
ROW_METRICS = [prefix + suffix for prefix in ("a", "b", "c") for suffix in
               ("_row_nnz_min", "_row_nnz_max", "_mean_nnz_per_row", "_row_nnz_stddev",
                "_row_nnz_cv", "_empty_rows")]
RANK_ARRAYS = ["rank_a_nnz", "rank_b_nnz", "rank_c_nnz", "rank_scalar_products"]
WORK_METRICS = ["scalar_products", "rank_work_min", "rank_work_max", "rank_work_mean",
                "rank_work_stddev", "rank_work_cv", "rank_work_max_over_mean", "idle_ranks"]
METRIC_FIELDS = ROW_METRICS + RANK_ARRAYS + WORK_METRICS + ["result_finite"]
PHASE_FIELDS = IDENTITY + TIMINGS + ["max_abs_error", "validation"] + SAMPLE_FIELDS + METRIC_FIELDS
FIELDS = IDENTITY + TIMINGS + ["max_abs_error", "validation", "compute_gflops_p90"] + SAMPLE_FIELDS + METRIC_FIELDS
STRUCTURE_FIELDS = FIELDS + ["a_density", "b_density", "c_density"]


def repo_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPO / path).resolve()


def check_root(results_dir):
    """Keep checks outside the repository's thesis-result tree."""
    try:
        relative = results_dir.relative_to(REPO / "results")
    except ValueError:
        return results_dir
    return REPO / "test-results" / relative


def output_root(cfg, action):
    return check_root(cfg.results_dir) if action == "pilot" else cfg.results_dir


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def options(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=RUN_ACTIONS + ANALYSES + ("all",))
    parser.add_argument("--config", type=Path, default=REPO / "scripts/experiments.json")
    for key in ("nodes", "matrices", "pilot-matrices", "permutation-matrices", "variants"):
        parser.add_argument("--" + key, nargs="+", type=int if key == "nodes" else str)
    for key in ("ranks-per-node", "threads", "cpus-per-rank", "runs", "warmup", "repeats",
                "trials", "chunk", "seed", "timeout"):
        parser.add_argument("--" + key, type=int)
    for key in ("build-dir", "matrix-dir", "results-dir", "mpi-launcher", "hostfile", "schedule"):
        parser.add_argument("--" + key)
    parser.add_argument("--mpi-arg", action="append", help="extra Open MPI token; use --mpi-arg=VALUE")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--validate", dest="validate", action="store_true", default=None)
    group.add_argument("--no-validate", dest="validate", action="store_false")
    parser.add_argument("--dry-run", action="store_true", help="preview without requiring inputs/binaries or writing files")
    parser.add_argument("--local-check", action="store_true", help="logical nodes on localhost; validation on, test-results/local-check by default")
    args = parser.parse_args(argv)
    defaults = read_json(REPO / "scripts/experiments.json")
    config = read_json(args.config)
    unknown = set(config) - set(defaults)
    if unknown:
        raise ValueError("Unknown configuration keys: " + ", ".join(sorted(unknown)))
    defaults.update(config)
    for key, value in vars(args).items():
        if value is not None and key in defaults:
            defaults[key] = value
    if args.mpi_arg is not None:
        defaults["mpi_args"] = args.mpi_arg
    cfg = argparse.Namespace(**defaults)
    cfg.action, cfg.dry_run, cfg.local_check = args.action, args.dry_run, args.local_check
    for key in ("ranks_per_node", "threads", "runs", "repeats", "trials", "chunk"):
        if type(getattr(cfg, key)) is not int or getattr(cfg, key) < 1:
            raise ValueError(key + " must be a positive integer")
    for key in ("warmup", "seed", "timeout"):
        if type(getattr(cfg, key)) is not int or getattr(cfg, key) < 0:
            raise ValueError(key + " must be a nonnegative integer")
    if cfg.cpus_per_rank is None:
        cfg.cpus_per_rank = cfg.threads + 1
    if type(cfg.cpus_per_rank) is not int or cfg.cpus_per_rank < cfg.threads + 1:
        raise ValueError("cpus_per_rank must be at least threads + 1 (same budget including Hybrid's worker)")
    if not isinstance(cfg.nodes, list) or not cfg.nodes or any(
            type(n) is not int or n < 1 or math.isqrt(n)**2 != n for n in cfg.nodes):
        raise ValueError("nodes must be positive perfect squares, e.g. 1 4 9 16")
    if len(set(cfg.nodes)) != len(cfg.nodes):
        raise ValueError("Repeated node counts; use runs for independent repetitions")
    if cfg.schedule not in ("static", "dynamic", "guided", "auto"):
        raise ValueError("Invalid OpenMP schedule")
    if type(cfg.validate) is not bool:
        raise ValueError("validate must be true or false")
    for key in ("variants", "matrices", "pilot_matrices", "permutation_matrices"):
        values = getattr(cfg, key)
        if not isinstance(values, list) or not values or any(not isinstance(v, str) for v in values):
            raise ValueError(key + " must be a nonempty string list")
        if len(set(values)) != len(values):
            raise ValueError("Duplicate values in " + key)
    if set(cfg.variants) - set(PROTOCOLS):
        raise ValueError("Unknown variants: " + ", ".join(sorted(set(cfg.variants) - set(PROTOCOLS))))
    if not isinstance(cfg.mpi_args, list) or any(not isinstance(v, str) for v in cfg.mpi_args):
        raise ValueError("mpi_args must be a list of tokens")
    # Mapping is owned by the runner; overriding it would mislabel baseline topology.
    forbidden = {"-n", "-np", "--np", "--n", "--host", "-H", "--hostfile", "--machinefile",
                 "--map-by", "--rank-by", "--bind-to", "--oversubscribe", "--nooversubscribe",
                 "--use-hwthread-cpus", "--app", "-app", ":"}
    if any(token.split("=", 1)[0] in forbidden for token in cfg.mpi_args):
        raise ValueError("mpi_args cannot override ranks, hosts, mapping or binding; use the dedicated options")
    for key in ("build_dir", "matrix_dir", "results_dir"):
        setattr(cfg, key, repo_path(getattr(cfg, key)))
    if cfg.local_check:
        cfg.validate = True
        if cfg.hostfile:
            raise ValueError("--local-check cannot use a hostfile")
        # Even a custom result root keeps logical tests apart from real measurements.
        cfg.results_dir = check_root(cfg.results_dir) / "local-check"
    if cfg.hostfile:
        cfg.hostfile = repo_path(cfg.hostfile)
    cfg.catalog = {entry["id"]: entry for entry in read_json(REPO / "scripts/matrices_catalog.json")["matrices"]}
    # Explicit --matrices is convenient for a complete small campaign.
    if args.matrices is not None and args.permutation_matrices is None:
        cfg.permutation_matrices = cfg.matrices
    cfg.matrices = select_matrices(cfg, cfg.matrices)
    cfg.pilot_matrices = select_matrices(cfg, cfg.pilot_matrices)
    cfg.permutation_matrices = select_matrices(cfg, cfg.permutation_matrices)
    return cfg


def select_matrices(cfg, requested):
    if requested == ["all"]:
        return [name for name, entry in cfg.catalog.items() if entry["tier"] == "all"]
    if "all" in requested or set(requested) - set(cfg.catalog):
        raise ValueError("Use all alone, or catalogue matrix names: " + ", ".join(requested))
    return requested


def cases(cfg, action):
    names = cfg.pilot_matrices if action == "pilot" else (
        cfg.permutation_matrices if action == "permutation" else cfg.matrices)
    kinds = ("square", "permuted", "rectangular") if action == "pilot" else ({
        "strong_scaling": ("square",), "permutation": ("permuted",), "rectangular": ("rectangular",)
    }[action])
    result = []
    for name in names:
        original = cfg.matrix_dir / (name + ".mtx")
        permuted = cfg.matrix_dir / "permuted" / (name + "_permuted_s%d.mtx" % cfg.seed)
        restriction = cfg.matrix_dir / (name + "_restriction.mtx")
        for kind in kinds:
            a, b = {"square": (original, original), "permuted": (permuted, permuted),
                    "rectangular": (original, restriction)}[kind]
            result.append((name, kind, a, b))
    return result


def input_preflight(cfg, work):
    """Check provenance and hashes once per distinct input, outside benchmark timers."""
    checked = {}
    for _, action_cases in work:
        for name, kind, a, b in action_cases:
            source = read_json(cfg.matrix_dir / "metadata" / (name + ".source.json"))
            records = {cfg.matrix_dir / (name + ".mtx"): source}
            if kind != "square":
                manifest_path = cfg.matrix_dir / "metadata" / (name + "_s%d.cases.json" % cfg.seed)
                manifest = read_json(manifest_path)
                if (manifest.get("schema_version") != 1 or manifest.get("generator") != matrices.GENERATOR
                        or manifest.get("matrix") != name or manifest.get("seed") != cfg.seed):
                    raise ValueError("Incompatible input manifest: " + str(manifest_path))
                for key in ("original", "permuted" if kind == "permuted" else "restriction"):
                    record = manifest["files"][key]
                    records[(manifest_path.parent / record["path"]).resolve()] = record
                if manifest["files"]["original"]["sha256"] != source["sha256"]:
                    raise ValueError("Manifest refers to a different original: " + name)
            for path in set((a, b)):
                record = records[path]
                if path not in checked:
                    print("Checking input: " + str(path), flush=True)
                    checked[path] = (matrices.sha256(path), matrices.header_of(path))
                digest, header = checked[path]
                if digest != record["sha256"] or any(header[k] != record[k] for k in ("rows", "cols", "entries")):
                    raise ValueError("Input changed since preparation: " + str(path))
            if checked[a][1]["cols"] != checked[b][1]["rows"]:
                raise ValueError("Incompatible input dimensions: " + str(a) + " / " + str(b))
    return {path: value[1] for path, value in checked.items()}


def read_table(path, fields):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != fields:
            raise ValueError("Incompatible TSV header: " + str(path))
        rows = list(reader)
        if any(None in row or any(v in (None, "") for v in row.values()) for row in rows):
            raise ValueError("Incomplete TSV row: " + str(path))
    # Appending after a partial last line would corrupt the next observation.
    with path.open("rb") as stream:
        stream.seek(-1, os.SEEK_END)
        if stream.read(1) != b"\n":
            raise ValueError("TSV has an incomplete last line: " + str(path))
    return rows


def write_table(path, fields, rows, append=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if append:
        new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
            if new:
                writer.writeheader()
            writer.writerows(rows)
        return
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextlib.contextmanager
def result_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".experiments.lock"
    try:
        lock.mkdir()
    except FileExistsError:
        raise ValueError("Another campaign/export may be writing " + str(root) +
                         "; remove .experiments.lock only after confirming it is no longer running")
    try:
        yield
    finally:
        lock.rmdir()


def sampling(cfg, action):
    # Pilot observations are correctness checks, excluded from performance exports.
    return (1, 0, 1, 1, True) if action == "pilot" else (
        cfg.runs, cfg.warmup, cfg.repeats, cfg.trials, cfg.validate)


def command(cfg, action, case, variant, nodes, raw_path):
    _, _, a, b = case
    _, warmup, repeats, trials, validate = sampling(cfg, action)
    cmd = [cfg.mpi_launcher] + cfg.mpi_args + ["-np", str(nodes * cfg.ranks_per_node)]
    if cfg.local_check:
        cmd += ["--host", "localhost", "--oversubscribe", "--bind-to", "none"]
    else:
        cmd += ["--map-by", "ppr:%d:node:PE=%d" % (cfg.ranks_per_node, cfg.cpus_per_rank),
                "--bind-to", "core", "--nooversubscribe"]
        if cfg.hostfile:
            cmd += ["--hostfile", str(cfg.hostfile)]
    cmd += [str(cfg.build_dir / variant), "--matrix-a", str(a), "--matrix-b", str(b),
            "--threads", str(cfg.threads), "--schedule", cfg.schedule, "--chunk", str(cfg.chunk),
            "--warmup", str(warmup), "--repeats", str(repeats), "--trials", str(trials),
            "--experiment", action, "--results", str(raw_path)]
    if cfg.local_check and variant.startswith("trident_"):
        cmd += ["--logical-node-size", str(cfg.ranks_per_node)]
    if not validate:
        cmd += ["--no-validate"]
    return cmd


def validate_structure_metrics(row):
    """Reject incomplete or inconsistent telemetry before appending an observation."""
    for key in ROW_METRICS + WORK_METRICS:
        value = float(row[key])
        if not math.isfinite(value) or value < 0:
            raise ValueError("Invalid structure metric: " + key)

    def agrees(key, expected):
        if not math.isclose(float(row[key]), expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Inconsistent structure metric: " + key)

    for prefix, rows_key, cols_key in (("a", "a_rows", "a_cols"), ("b", "b_rows", "b_cols"),
                                       ("c", "a_rows", "b_cols")):
        count = int(row[rows_key])
        mean = int(row[prefix + "_nnz"]) / count if count else 0
        minimum, maximum = (int(row[prefix + suffix]) for suffix in ("_row_nnz_min", "_row_nnz_max"))
        empty = int(row[prefix + "_empty_rows"])
        # Input CSR retains duplicate coordinates and explicit zeros. Stored entries per
        # row can exceed the column count; only the accumulated output is canonical.
        limit = int(row[cols_key]) if prefix == "c" else int(row[prefix + "_nnz"])
        if not (0 <= minimum <= mean <= maximum <= limit and 0 <= empty <= count):
            raise ValueError("Invalid row-degree bounds: " + prefix)
        agrees(prefix + "_mean_nnz_per_row", mean)
        agrees(prefix + "_row_nnz_cv", float(row[prefix + "_row_nnz_stddev"]) / mean if mean else 0)

    arrays = {}
    for field, total in zip(RANK_ARRAYS, ("a_nnz", "b_nnz", "c_nnz", "scalar_products")):
        values = json.loads(row[field])
        if (not isinstance(values, list) or len(values) != int(row["ranks"]) or
                any(type(value) is not int or value < 0 for value in values) or
                sum(values) != int(row[total])):
            raise ValueError("Invalid rank counts: " + field)
        arrays[field] = values
    work = arrays["rank_scalar_products"]
    mean, stddev = statistics.mean(work), statistics.pstdev(work)
    for key, expected in (("rank_work_min", min(work)), ("rank_work_max", max(work)),
                          ("rank_work_mean", mean), ("rank_work_stddev", stddev),
                          ("rank_work_cv", stddev / mean if mean else 0),
                          ("rank_work_max_over_mean", max(work) / mean if mean else 0),
                          ("idle_ranks", work.count(0))):
        agrees(key, expected)


def validate_measurements(row, trident):
    """Shared checks for a new launch and for observations reused by analysis exports."""
    validate = row["validation"] == "PASS"
    if row["validation"] not in ("PASS", "SKIPPED") or row["result_finite"] != "1":
        raise ValueError("Failed validation or non-finite product")
    if validate:
        if not 0 <= float(row["max_abs_error"]) < 1e-10:
            raise ValueError("PASS disagrees with the numerical validation error")
    elif row["max_abs_error"] != "NA":
        raise ValueError("Skipped validation must have max_abs_error=NA")
    for key in ("a_rows", "a_cols", "b_rows", "b_cols", "ranks", "repeats", "trials"):
        if int(row[key]) < 1:
            raise ValueError("Invalid positive count: " + key)
    required = [key for key in TIMINGS if key not in (
        ("halo_setup_seconds",) if trident else ("plan_setup_seconds", "inter_node_p90_seconds", "intra_node_p90_seconds"))]
    for key in required + ["compute_gflops_p90"]:
        value = float(row[key])
        if not math.isfinite(value) or value < 0:
            raise ValueError("Invalid benchmark metric: " + key)
    for key in ("a_nnz", "b_nnz", "c_nnz"):
        if int(row[key]) < 0:
            raise ValueError("Invalid nonzero count: " + key)
    if any(float(row[phase]) > float(row["full_product_seconds"]) for phase in
           ("distribution_seconds", "first_product_seconds", "first_gather_seconds")):
        raise ValueError("Full product time is shorter than a contained phase")
    setup = "plan_setup_seconds" if trident else "halo_setup_seconds"
    if float(row[setup]) > float(row["first_product_seconds"]):
        raise ValueError("First product time is shorter than setup")
    arrays = {}
    for field in SAMPLE_FIELDS if trident else SAMPLE_FIELDS[2:]:
        samples = json.loads(row[field])
        if (not isinstance(samples, list) or len(samples) != int(row["repeats"]) * int(row["trials"]) or
                any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
                    for value in samples)):
            raise ValueError("Invalid timing samples: " + field)
        p90 = sorted(samples)[math.ceil(0.90 * len(samples)) - 1]
        if not math.isclose(p90, float(row[field.replace("_samples_", "_p90_")]),
                            rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("Timing samples disagree with P90: " + field)
        arrays[field] = samples
    for field, samples in arrays.items():
        outer = arrays["communication_samples_seconds"] if field.startswith(("inter_node", "intra_node")) else arrays["end_to_end_samples_seconds"]
        if any(inner > total and not math.isclose(inner, total, rel_tol=1e-12, abs_tol=1e-15)
               for inner, total in zip(samples, outer)):
            raise ValueError("Phase samples exceed their containing interval: " + field)
    validate_structure_metrics(row)
    compute = float(row["compute_p90_seconds"])
    expected_gflops = 2 * int(row["scalar_products"]) / compute / 1e9 if compute else 0
    if not math.isclose(float(row["compute_gflops_p90"]), expected_gflops, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError("GFLOP/s disagrees with scalar products and compute P90")


def normalize(cfg, action, case, variant, nodes, raw_path, headers):
    with raw_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("Missing or duplicate benchmark header")
        rows = list(reader)
    if len(rows) != 1 or None in rows[0] or any(v is None for v in rows[0].values()):
        raise ValueError("Expected exactly one complete benchmark observation")
    row = rows[0]
    _, _, a, b = case
    _, warmup, repeats, trials, validate = sampling(cfg, action)
    trident = variant.startswith("trident_")
    identity = variant if trident else variant.replace("spgemm_", "mpi_openmp_", 1)
    expected = dict(implementation=identity, experiment=action, benchmark_protocol=PROTOCOLS[variant],
                    matrix_a_source=str(a), matrix_b_source=str(b), ranks=nodes * cfg.ranks_per_node,
                    threads_per_rank=cfg.threads, omp_schedule=cfg.schedule, omp_chunk=cfg.chunk,
                    warmup=warmup, repeats=repeats, trials=trials,
                    a_rows=headers[a]["rows"], a_cols=headers[a]["cols"],
                    b_rows=headers[b]["rows"], b_cols=headers[b]["cols"],
                    validation="PASS" if validate else "SKIPPED")
    if trident:
        expected.update(nodes=nodes, ranks_per_node=cfg.ranks_per_node,
                        topology="logical_test" if cfg.local_check else "physical",
                        grid_rows=math.isqrt(nodes), grid_cols=math.isqrt(nodes))
    for key, value in expected.items():
        if row.get(key) != str(value):
            raise ValueError("Unexpected %s: %r, expected %r" % (key, row.get(key), str(value)))
    gflops = "compute_gflops" if trident else "compute_gflops_p90"
    normalized = {key: row.get(key, "NA") for key in FIELDS}
    normalized.update(matrix_a=a.relative_to(cfg.matrix_dir).as_posix(),
                      matrix_b=b.relative_to(cfg.matrix_dir).as_posix(),
                      nodes=str(nodes), ranks_per_node=str(cfg.ranks_per_node),
                      cpus_per_rank="NA" if cfg.local_check else str(cfg.cpus_per_rank),
                      topology="logical_test" if cfg.local_check else "physical",
                      compute_gflops_p90=row[gflops])
    validate_measurements(normalized, trident)
    return normalized


def run_experiment(cfg, action, action_cases, headers):
    runs = sampling(cfg, action)[0]
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS=str(cfg.threads), OMP_DYNAMIC="FALSE", OMP_PLACES="cores",
               OMP_PROC_BIND="close")
    for case in action_cases:
        for nodes in cfg.nodes:
            for repeat in range(runs):
                for variant in cfg.variants:
                    destination = output_root(cfg, action) / action / (variant + ".tsv")
                    print("%s: %s / %s, nodes=%d, %s, execution %d/%d" % (
                        action, case[2].name, case[3].name, nodes, variant, repeat + 1, runs), flush=True)
                    if cfg.dry_run:
                        cmd = command(cfg, action, case, variant, nodes, Path("<temporary-result.tsv>"))
                        print("  " + shlex.join(cmd) + "\n  -> " + str(destination))
                        continue
                    temporary_root = check_root(cfg.results_dir) / "tmp"
                    temporary_root.mkdir(parents=True, exist_ok=True)
                    scratch = Path(tempfile.mkdtemp(prefix=variant + "-", dir=temporary_root))
                    raw_path, log_path = scratch / "raw.tsv", scratch / "launch.log"
                    try:
                        with log_path.open("w", encoding="utf-8") as log:
                            subprocess.run(command(cfg, action, case, variant, nodes, raw_path),
                                           env=env, stdout=log, stderr=subprocess.STDOUT,
                                           check=True, timeout=cfg.timeout or None)
                        row = normalize(cfg, action, case, variant, nodes, raw_path, headers)
                        write_table(destination, FIELDS, [row], append=True)
                    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
                        raise RuntimeError("%s; no observation appended for this launch. Diagnostics: %s" % (error, scratch)) from error
                    shutil.rmtree(scratch)


def analyze(cfg, action):
    sources = ("strong_scaling", "permutation", "rectangular") if action == "phase_analysis" else ("strong_scaling",)
    fields = PHASE_FIELDS if action == "phase_analysis" else STRUCTURE_FIELDS
    tables = {}
    for variant in cfg.variants:
        rows = []
        for source in sources:
            rows.extend(read_table(cfg.results_dir / source / (variant + ".tsv"), FIELDS))
        output = []
        for row in rows:
            if row["benchmark_protocol"] != PROTOCOLS[variant] or row["validation"] not in ("PASS", "SKIPPED"):
                raise ValueError("Invalid source observation for " + variant)
            validate_measurements(row, variant.startswith("trident_"))
            derived = {key: row[key] for key in fields if key in row}
            if action == "matrix_structure":
                derived.update(a_density=str(int(row["a_nnz"]) / (int(row["a_rows"]) * int(row["a_cols"]))),
                               b_density=str(int(row["b_nnz"]) / (int(row["b_rows"]) * int(row["b_cols"]))),
                               c_density=str(int(row["c_nnz"]) / (int(row["a_rows"]) * int(row["b_cols"]))))
            output.append(derived)
        tables[variant] = output
    if not any(tables.values()):
        raise ValueError("No source measurements for " + action + "; run " + ", ".join(sources) + " first")
    for variant, rows in tables.items():
        destination = cfg.results_dir / action / (variant + ".tsv")
        write_table(destination, fields, rows)
        print("%s: %d observations -> %s" % (action, len(rows), destination), flush=True)


def main(argv=None):
    cfg = options(argv)
    actions = RUN_ACTIONS + ANALYSES if cfg.action == "all" else (cfg.action,)
    work = [(action, cases(cfg, action)) for action in actions if action in RUN_ACTIONS]
    count = sum(len(items) * len(cfg.nodes) * len(cfg.variants) * sampling(cfg, action)[0]
                for action, items in work)
    display_root = output_root(cfg, cfg.action) if cfg.action != "all" else cfg.results_dir
    print("%d MPI launches; results: %s" % (count, display_root), flush=True)
    if "pilot" in actions and len(actions) > 1:
        print("Pilot checks: %s" % (output_root(cfg, "pilot") / "pilot"), flush=True)
    if cfg.local_check:
        print("LOCAL CHECK: logical nodes, no rank binding; excluded from cluster performance results.", flush=True)
    if cfg.dry_run:
        for action, action_cases in work:
            run_experiment(cfg, action, action_cases, {})
        for action in actions:
            if action in ANALYSES:
                print(action + ": regenerate per-implementation TSVs from recorded observations (no MPI launch)")
        return 0
    if work:
        if not shutil.which(cfg.mpi_launcher):
            raise ValueError("MPI launcher not found: " + cfg.mpi_launcher)
        if cfg.hostfile and not cfg.hostfile.is_file():
            raise ValueError("Hostfile not found: " + str(cfg.hostfile))
        for variant in cfg.variants:
            executable = cfg.build_dir / variant
            if not executable.is_file() or not os.access(executable, os.X_OK):
                raise ValueError("Executable missing/not executable: %s; run scripts/build.sh first" % executable)
        if not cfg.local_check and os.environ.get("SLURM_JOB_NUM_NODES"):
            if max(cfg.nodes) > int(os.environ["SLURM_JOB_NUM_NODES"]):
                raise ValueError("Requested nodes exceed the current Slurm allocation")
        headers = input_preflight(cfg, work)
    else:
        headers = {}
    with contextlib.ExitStack() as locks:
        # A complete campaign can write thesis tables and pilot checks to different roots.
        for root in sorted({output_root(cfg, action) for action in actions}):
            locks.enter_context(result_lock(root))
        # Refuse incompatible or partial tables before any expensive MPI launch.
        for action, _ in work:
            for variant in cfg.variants:
                read_table(output_root(cfg, action) / action / (variant + ".tsv"), FIELDS)
        for action in actions:
            if action in RUN_ACTIONS:
                run_experiment(cfg, action, dict(work)[action], headers)
            else:
                analyze(cfg, action)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print("Error: " + str(error), file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("Interrupted; completed observations have been retained.", file=sys.stderr)
        sys.exit(130)
