#!/usr/bin/env python3
"""Check the real MPI benchmark output contract on small deterministic products."""
import argparse
import contextlib
import csv
import importlib.util
import io
import json
import math
import statistics
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

parser = argparse.ArgumentParser()
parser.add_argument("--build-dir", type=Path, required=True)
parser.add_argument("--mpi-launcher", default="mpirun")
parser.add_argument("--numproc-flag", default="-np")
args, remaining = parser.parse_known_args()
VARIANTS = ("spgemm_two_sided", "spgemm_one_sided_get", "spgemm_one_sided_put",
            "trident_two_sided", "trident_hybrid", "trident_get",
            "trident_get_pipeline", "trident_put")
spec = importlib.util.spec_from_file_location("experiments", Path(__file__).resolve().parents[1] /
                                             "scripts/lib/experiments.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class MeasurementsTest(unittest.TestCase):
    def check_measurements(self, row, variant):
        self.assertEqual(row["validation"], "PASS")
        self.assertEqual(row["result_finite"], "1")
        total = float(row["full_product_seconds"])
        self.assertTrue(math.isfinite(total) and total > 0)
        for phase in ("distribution_seconds", "first_product_seconds", "first_gather_seconds"):
            duration = float(row[phase])
            self.assertTrue(math.isfinite(duration) and 0 <= duration <= total)
        for phase in ("communication", "compute", "end_to_end") + (
                ("inter_node", "intra_node") if variant.startswith("trident_") else ()):
            samples = json.loads(row[phase + "_samples_seconds"])
            self.assertEqual(len(samples), 6)
            self.assertTrue(all(math.isfinite(t) and t >= 0 for t in samples))
            self.assertEqual(sorted(samples)[math.ceil(.9 * len(samples)) - 1],
                             float(row[phase + "_p90_seconds"]))

    def check_structure(self, row, variant, ranks, empty):
        for prefix, degrees in (("a", [3, 0, 1, 1, 1]), ("b", [2, 3, 1]), ("c", [5, 0, 2, 3, 1])):
            if empty:
                degrees = [0, 0, 0]
            mean, stddev = statistics.mean(degrees), statistics.pstdev(degrees)
            for suffix, expected in (("_row_nnz_min", min(degrees)), ("_row_nnz_max", max(degrees)),
                                     ("_mean_nnz_per_row", mean), ("_row_nnz_stddev", stddev),
                                     ("_row_nnz_cv", stddev / mean if mean else 0),
                                     ("_empty_rows", degrees.count(0))):
                self.assertAlmostEqual(float(row[prefix + suffix]), expected)
            owned = json.loads(row["rank_" + prefix + "_nnz"])
            self.assertEqual(len(owned), ranks)
            self.assertTrue(all(type(value) is int and value >= 0 for value in owned))
            self.assertEqual(sum(owned), sum(degrees))
            self.assertEqual(int(row[prefix + "_nnz"]), sum(degrees))
            if ranks == 4 and not empty:
                expected_owned = ({"a": [3, 1, 1, 1], "b": [3, 2, 0, 1], "c": [4, 3, 2, 2]}
                                  if variant.startswith("trident_") else
                                  {"a": [3, 1, 1, 1], "b": [2, 3, 1, 0], "c": [5, 2, 3, 1]})
                self.assertEqual(owned, expected_owned[prefix])
        if empty:
            work = [0] * ranks
        elif variant.startswith("trident_"):
            work = [4, 4, 2, 2] if ranks == 4 else [3, 1, 3, 1, 2, 0, 1, 1]
        else:
            work = [6, 2, 3, 1] if ranks == 4 else [6, 0, 2, 3, 1, 0, 0, 0]
        self.assertEqual(json.loads(row["rank_scalar_products"]), work)
        self.assertEqual(int(row["scalar_products"]), 0 if empty else 12)
        mean, stddev = statistics.mean(work), statistics.pstdev(work)
        for key, expected in (("rank_work_min", min(work)), ("rank_work_max", max(work)),
                              ("rank_work_mean", mean), ("rank_work_stddev", stddev),
                              ("rank_work_cv", stddev / mean if mean else 0),
                              ("rank_work_max_over_mean", max(work) / mean if mean else 0),
                              ("idle_ranks", work.count(0))):
            self.assertAlmostEqual(float(row[key]), expected)

    def test_all_implementations(self):
        # Uneven rectangular product: 12 scalar products, 11 output nonzeros,
        # one empty A row and unsorted B columns. Expectations are computed by hand.
        with tempfile.TemporaryDirectory(prefix="spgemm-measurements-") as directory:
            a, b = Path(directory) / "a.mtx", Path(directory) / "b.mtx"
            for ranks, empty in ((4, False), (8, False), (4, True)):
                cfg = SimpleNamespace(runs=1, warmup=1, repeats=3, trials=2, validate=True,
                                      threads=1, schedule="guided", chunk=64, ranks_per_node=ranks // 4,
                                      local_check=True, matrix_dir=Path(directory), variants=VARIANTS,
                                      results_dir=Path(directory) / ("exports-%d-%s" % (ranks, empty)))
                header = "%%MatrixMarket matrix coordinate real general\n"
                a.write_text(header + ("3 3 0\n" if empty else
                             "5 3 6\n1 1 1\n1 2 1\n1 3 1\n3 1 1\n4 2 1\n5 3 1\n"))
                b.write_text(header + ("3 3 0\n" if empty else
                             "3 5 6\n1 5 1\n1 1 1\n2 4 1\n2 2 1\n2 3 1\n3 5 1\n"))
                for variant in VARIANTS:
                    with self.subTest(variant=variant, ranks=ranks, empty=empty):
                        path = Path(directory) / ("%s-%d-%s.tsv" % (variant, ranks, empty))
                        command = [args.mpi_launcher, args.numproc_flag, str(ranks),
                                   "--oversubscribe", "--bind-to", "none",
                                   str(args.build_dir.resolve() / variant),
                                   "--matrix-a", str(a), "--matrix-b", str(b),
                                   "--experiment", "strong_scaling",
                                   "--threads", "1", "--warmup", "1", "--repeats", "3",
                                   "--trials", "2", "--results", str(path)]
                        if variant.startswith("trident_"):
                            command += ["--logical-node-size", str(ranks // 4)]
                        run = subprocess.run(command, capture_output=True, text=True, timeout=60)
                        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                        with path.open(newline="", encoding="utf-8") as stream:
                            rows = list(csv.DictReader(stream, delimiter="\t"))
                        self.assertEqual(len(rows), 1)
                        self.check_measurements(rows[0], variant)
                        self.check_structure(rows[0], variant, ranks, empty)
                        headers = {a: dict(rows=3 if empty else 5, cols=3),
                                   b: dict(rows=3, cols=3 if empty else 5)}
                        normalized = runner.normalize(cfg, "strong_scaling", (None, None, a, b),
                                                      variant, 4, path, headers)
                        self.check_measurements(normalized, variant)
                        self.check_structure(normalized, variant, ranks, empty)
                        runner.write_table(cfg.results_dir / "strong_scaling" / (variant + ".tsv"),
                                           runner.FIELDS, [normalized])
                for action in runner.ANALYSES:
                    with contextlib.redirect_stdout(io.StringIO()):
                        runner.analyze(cfg, action)
                    for variant in VARIANTS:
                        with (cfg.results_dir / action / (variant + ".tsv")).open(newline="") as stream:
                            exported = list(csv.DictReader(stream, delimiter="\t"))
                        self.assertEqual(len(exported), 1)
                        self.check_measurements(exported[0], variant)
                        self.check_structure(exported[0], variant, ranks, empty)

    def test_duplicates_zeros_and_nonfinite_product(self):
        with tempfile.TemporaryDirectory(prefix="spgemm-edge-measurements-") as directory:
            a, b = Path(directory) / "a.mtx", Path(directory) / "b.mtx"
            header = "%%MatrixMarket matrix coordinate real general\n"
            for overflow in (False, True):
                a.write_text(header + ("2 2 1\n1 1 1e200\n" if overflow else
                             "2 2 5\n1 2 1\n1 1 1\n1 2 2\n1 1 -1\n1 2 0\n"))
                b.write_text(header + ("2 2 1\n1 1 1e200\n" if overflow else
                             "2 3 7\n1 3 2\n1 1 1\n1 3 -2\n2 3 2\n2 1 1\n2 3 -1\n2 2 0\n"))
                for variant in VARIANTS:
                    with self.subTest(variant=variant, overflow=overflow):
                        path = Path(directory) / (variant + str(overflow) + ".tsv")
                        command = [args.mpi_launcher, args.numproc_flag, "4", "--oversubscribe", "--bind-to", "none",
                                   str(args.build_dir.resolve() / variant), "--matrix-a", str(a), "--matrix-b", str(b),
                                   "--threads", "1", "--warmup", "0", "--repeats", "1", "--trials", "1",
                                   "--results", str(path)]
                        if variant.startswith("trident_"):
                            command += ["--logical-node-size", "1"]
                        if overflow:
                            command += ["--no-validate"]
                        run = subprocess.run(command, capture_output=True, text=True, timeout=60)
                        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                        with path.open(newline="") as stream:
                            rows = list(csv.DictReader(stream, delimiter="\t"))
                        self.assertEqual(len(rows), 1)
                        row = rows[0]
                        row["compute_gflops_p90"] = row.get("compute_gflops", row.get("compute_gflops_p90"))
                        if overflow:
                            self.assertEqual(row["validation"], "SKIPPED")
                            self.assertEqual(row["result_finite"], "0")
                            with self.assertRaisesRegex(ValueError, "non-finite product"):
                                runner.validate_measurements(row, variant.startswith("trident_"))
                        else:
                            self.assertEqual(row["validation"], "PASS")
                            self.assertEqual([int(row[p + "_nnz"]) for p in ("a", "b", "c")], [5, 7, 2])
                            self.assertEqual([int(row[p + "_row_nnz_max"]) for p in ("a", "b", "c")], [5, 4, 2])
                            self.assertEqual(int(row["scalar_products"]), 18)
                            work = [8, 10, 0, 0] if variant.startswith("trident_") else [18, 0, 0, 0]
                            self.assertEqual(json.loads(row["rank_scalar_products"]), work)
                            runner.validate_measurements(row, variant.startswith("trident_"))


if __name__ == "__main__":
    unittest.main(argv=[__file__] + remaining)
