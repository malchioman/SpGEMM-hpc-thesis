#!/usr/bin/env python3
"""Exercise campaign behavior with isolated files and a subprocess MPI fixture."""
import contextlib
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("experiments", REPO / "scripts/lib/experiments.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class CampaignTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="spgemm campaign ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.data, self.build, self.results = (self.root / s for s in ("matrices", "build", "results"))
        (self.data / "metadata").mkdir(parents=True)
        self.build.mkdir()
        source = self.data / "cage8.mtx"
        source.write_text("%%MatrixMarket matrix coordinate real general\n2 2 3\n1 1 2\n1 2 1\n2 2 3\n", encoding="ascii")
        metadata = runner.matrices.inspect_matrix(source)
        runner.matrices.write_json(self.data / "metadata/cage8.source.json", metadata)
        with contextlib.redirect_stdout(io.StringIO()):
            runner.matrices.prepare({"id": "cage8"}, self.data, 42)
        for name in runner.PROTOCOLS:
            binary = self.build / name
            binary.write_text("mock binary", encoding="ascii")
            binary.chmod(0o755)
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps(dict(
            nodes=[1], ranks_per_node=2, threads=2, runs=2, warmup=1, repeats=3, trials=2,
            matrices=["cage8"], permutation_matrices=["cage8"],
            build_dir=str(self.build), matrix_dir=str(self.data), results_dir=str(self.results),
            mpi_launcher=sys.executable, mpi_args=[str(REPO / "tests/fixtures/mock_experiment_mpi.py")]
        )), encoding="utf-8")
        self.launch_log = self.root / "launches.jsonl"
        self.environment = patch.dict(os.environ, {"MOCK_LAUNCH_LOG": str(self.launch_log)})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def run_action(self, action, *args):
        with contextlib.redirect_stdout(io.StringIO()):
            return runner.main([action, "--config", str(self.config)] + list(args))

    def rows(self, action, variant="trident_get", root=None):
        with ((root or self.results) / action / (variant + ".tsv")).open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream, delimiter="\t"))

    def launches(self):
        return [json.loads(line) for line in self.launch_log.read_text(encoding="utf-8").splitlines()]

    def test_complete_campaign_and_repeat_exports(self):
        self.run_action("all")
        self.assertEqual(len(self.launches()), 72)  # 24 pilot + 3 products * 2 launches * 8 implementations
        for variant in runner.PROTOCOLS:
            self.assertEqual(len(self.rows("pilot", variant)), 3)
            self.assertEqual(len(self.rows("strong_scaling", variant)), 2)
            phases = self.rows("phase_analysis", variant)
            self.assertEqual(len(phases), 6)
            self.assertEqual(phases[0], phases[1])  # identical repeated observations remain separate
            self.assertEqual(phases[0]["cpus_per_rank"], "3")
            self.assertEqual(phases[0]["repeats"], "3")
            self.assertEqual(phases[0]["validation"], "SKIPPED")
            self.assertEqual(phases[0]["matrix_a"], "cage8.mtx")
            self.assertEqual(phases[2]["matrix_a"], "permuted/cage8_permuted_s42.mtx")
            self.assertEqual(phases[4]["matrix_b"], "cage8_restriction.mtx")
            self.assertEqual(self.rows("pilot", variant)[0]["validation"], "PASS")
            self.assertEqual(self.rows("pilot", variant)[0]["repeats"], "1")
            for forbidden in ("timestamp", "run_id", "case", "experiment", "implementation"):
                self.assertNotIn(forbidden, phases[0])
            structure = self.rows("matrix_structure", variant)
            self.assertEqual(len(structure), 2)
            self.assertEqual(float(structure[0]["a_density"]), 0.75)
            self.assertEqual(float(structure[0]["c_density"]), 1.0)
            self.assertEqual(float(structure[0]["a_mean_nnz_per_row"]), 1.5)
            if variant.startswith("spgemm_"):
                self.assertEqual(phases[0]["inter_node_p90_seconds"], "NA")
                self.assertEqual(phases[0]["plan_setup_seconds"], "NA")
            else:
                self.assertEqual(phases[0]["halo_setup_seconds"], "NA")
        original = {path: path.read_bytes() for action in runner.ANALYSES for path in (self.results / action).glob("*.tsv")}
        self.run_action("phase_analysis")
        self.run_action("matrix_structure")
        self.assertEqual(original, {path: path.read_bytes() for path in original})
        self.assertEqual(len(self.launches()), 72)
        self.run_action("strong_scaling", "--variants", "trident_get")
        self.run_action("phase_analysis", "--variants", "trident_get")
        self.assertEqual(len(self.rows("strong_scaling")), 4)
        self.assertEqual(len(self.rows("phase_analysis")), 8)
        self.assertEqual(len(self.rows("phase_analysis", "trident_put")), 6)

    def test_failed_or_invalid_launch_is_not_appended(self):
        for failure in ("exit", "protocol", "validation", "nan", "topology", "duplicate"):
            with self.subTest(failure=failure), patch.dict(os.environ, {"MOCK_FAILURE": failure}):
                with self.assertRaisesRegex(RuntimeError, "no observation appended"):
                    self.run_action("strong_scaling", "--variants", "trident_get")
            self.assertFalse((self.results / "strong_scaling/trident_get.tsv").exists())
            self.assertFalse((self.results / ".experiments.lock").exists())
        self.assertEqual(len(list((self.results / "tmp").glob("*/launch.log"))), 6)

    def test_preflight_stops_before_launch_for_changed_inputs(self):
        path = self.data / "cage8_restriction.mtx"
        path.write_text(path.read_text() + "% changed\n", encoding="ascii")
        with self.assertRaisesRegex(ValueError, "Input changed"):
            self.run_action("all")
        self.assertFalse(self.launch_log.exists())
        self.assertFalse(self.results.exists())

    def test_preflight_rejects_existing_schema_and_partial_line(self):
        path = self.results / "strong_scaling/trident_get.tsv"
        path.parent.mkdir(parents=True)
        for content, error in (("old\theader\n", "header"), ("\t".join(runner.FIELDS), "last line")):
            with self.subTest(error=error):
                path.write_text(content, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, error):
                    self.run_action("strong_scaling", "--variants", "trident_get")
                self.assertEqual(path.read_text(), content)
                self.assertFalse(self.launch_log.exists())

    def test_invalid_resources_and_mapping_are_rejected(self):
        for args in (("--nodes", "2"), ("--nodes", "1", "1"), ("--threads", "0"),
                     ("--cpus-per-rank", "2"), ("--mpi-arg=--map-by=slot",),
                     ("--matrices", "all", "cage8"), ("--variants", "unknown")):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.run_action("strong_scaling", *args)
        self.assertFalse(self.launch_log.exists())

    def test_dry_run_has_no_dependencies_or_side_effects(self):
        shutil.rmtree(self.data)
        shutil.rmtree(self.build)
        self.run_action("all", "--dry-run", "--mpi-launcher", "nonexistent-mpirun")
        self.assertFalse(self.results.exists())
        self.assertFalse(self.launch_log.exists())

    def test_logical_topology_is_isolated_and_validated(self):
        self.run_action("strong_scaling", "--variants", "trident_get", "--nodes", "4", "--local-check")
        root = self.results / "local-check"
        row = self.rows("strong_scaling", root=root)[0]
        self.assertEqual((row["nodes"], row["ranks"], row["ranks_per_node"]), ("4", "8", "2"))
        self.assertEqual(row["cpus_per_rank"], "NA")
        self.assertEqual(row["validation"], "PASS")
        self.assertEqual(row["topology"], "logical_test")
        self.assertFalse((self.results / "strong_scaling").exists())
        self.assertIn("--oversubscribe", self.launches()[0])
        self.run_action("phase_analysis", "--variants", "trident_get", "--local-check")
        self.assertEqual(len(self.rows("phase_analysis", root=root)), 2)

    def test_missing_source_analysis_does_not_create_empty_exports(self):
        with self.assertRaisesRegex(ValueError, "No source measurements"):
            self.run_action("phase_analysis")
        self.assertFalse((self.results / "phase_analysis").exists())

    def test_result_lock_prevents_overlapping_writers(self):
        with runner.result_lock(self.results), self.assertRaisesRegex(ValueError, "Another campaign"):
            self.run_action("strong_scaling", "--variants", "trident_get")
        self.assertFalse(self.launch_log.exists())

    @unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "Linux/WSL shell entrypoints")
    def test_shell_entrypoints_from_another_directory(self):
        env = dict(os.environ, PYTHON=sys.executable)
        for script in ("run_pilot", "run_strong_scaling", "run_permutation", "run_rectangular",
                       "analyze_phases", "analyze_structure", "run_all", "build"):
            result = subprocess.run(["bash", str(REPO / "scripts" / (script + ".sh")), "--help"],
                                    cwd=self.root, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(result.returncode, 0, result.stderr.decode())


if __name__ == "__main__":
    unittest.main()
