#!/usr/bin/env python3
"""Exercise campaign behavior with isolated files and a subprocess MPI fixture."""
import contextlib
from collections import Counter
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
import prepare_sbatchman as planner  # runner adds scripts/ to the import path


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
            return runner.main([action] + list(args) + ["--config", str(self.config)])

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
            self.assertEqual(json.loads(phases[0]["compute_samples_seconds"]), [0.3] * 6)
            self.assertEqual(json.loads(self.rows("pilot", variant)[0]["compute_samples_seconds"]), [0.3])
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
        for failure in ("exit", "protocol", "validation", "nan", "topology", "duplicate",
                        "sample_count", "sample_nan", "sample_p90", "full_time",
                        "rank_count", "rank_sum", "rank_stats", "row_stats", "nonfinite_product",
                        "gflops", "phase_interval", "validation_error"):
            with self.subTest(failure=failure), patch.dict(os.environ, {"MOCK_FAILURE": failure}):
                with self.assertRaisesRegex(RuntimeError, "no observation appended"):
                    self.run_action("strong_scaling", "--variants", "trident_get")
            self.assertFalse((self.results / "strong_scaling/trident_get.tsv").exists())
            self.assertFalse((self.results / ".experiments.lock").exists())
        self.assertEqual(len(list((self.results / "tmp").glob("*/launch.log"))), 18)

    def test_analysis_rejects_corrupted_measurements(self):
        self.run_action("strong_scaling", "--variants", "trident_get")
        path = self.results / "strong_scaling/trident_get.tsv"
        original = path.read_bytes()
        for field, value in (("compute_p90_seconds", "nan"), ("compute_samples_seconds", "[]"),
                             ("rank_work_mean", "99"), ("result_finite", "0"),
                             ("validation", "PASS"), ("compute_gflops_p90", "99")):
            with self.subTest(field=field):
                rows = self.rows("strong_scaling")
                rows[0][field] = value
                runner.write_table(path, runner.FIELDS, rows)
                for action in runner.ANALYSES:
                    with self.assertRaises(ValueError):
                        self.run_action(action, "--variants", "trident_get")
                    self.assertFalse((self.results / action / "trident_get.tsv").exists())
                path.write_bytes(original)

    def test_complete_workflow_repetitions_match_separate_invocations(self):
        variant = "trident_get"
        reference = self.root / "separate"
        for _ in range(2):
            self.run_action("all", "--variants", variant, "--results-dir", str(reference))
        expected = {action: self.rows(action, root=reference)
                    for action in runner.RUN_ACTIONS + runner.ANALYSES}
        cycle = ["pilot"] * 3 + ["strong_scaling"] * 2 + ["permutation"] * 2 + ["rectangular"] * 2
        for count_args in (("2",), ("--campaign-repeats", "2")):
            with self.subTest(count_args=count_args):
                root = self.root / ("positional" if len(count_args) == 1 else "flag")
                before = len(self.launches())
                self.run_action("all", *count_args, "--variants", variant, "--results-dir", str(root))
                launches = self.launches()[before:]
                self.assertEqual([cmd[cmd.index("--experiment") + 1] for cmd in launches], cycle * 2)
                self.assertEqual({action: self.rows(action, root=root) for action in expected}, expected)
                self.assertEqual(len(list(root.rglob("*.tsv"))), 6)
                self.assertFalse((root / ".experiments.lock").exists())

    def test_failure_stops_remaining_workflow_repetitions(self):
        original_analyze = runner.analyze

        def fail_next_launch(cfg, action):
            original_analyze(cfg, action)
            if action == "matrix_structure":
                os.environ["MOCK_FAILURE"] = "exit"

        with patch.dict(os.environ), patch.object(runner, "analyze", side_effect=fail_next_launch):
            with self.assertRaisesRegex(RuntimeError, "no observation appended"):
                self.run_action("all", "3", "--runs", "1", "--variants", "trident_get")
        # The first workflow completes; the next pilot launch fails. No third workflow runs.
        self.assertEqual(len(self.launches()), 7)
        self.assertEqual(len(self.rows("pilot")), 3)
        for action in ("strong_scaling", "permutation", "rectangular", "matrix_structure"):
            self.assertEqual(len(self.rows(action)), 1)
        self.assertEqual(len(self.rows("phase_analysis")), 3)
        self.assertFalse((self.results / ".experiments.lock").exists())

    def test_invalid_campaign_repetitions_are_rejected(self):
        for args in (("all", "0"), ("all", "-2"), ("all", "--campaign-repeats", "0"),
                     ("all", "2", "--campaign-repeats", "3"),
                     ("strong_scaling", "2"), ("pilot", "--campaign-repeats", "2")):
            with self.subTest(args=args), self.assertRaises(ValueError):
                runner.options(list(args))
        for count in ("1.5", "five"):
            with self.subTest(count=count), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as failure:
                    runner.options(["all", count])
                self.assertEqual(failure.exception.code, 2)
        self.assertFalse(self.launch_log.exists())
        self.assertFalse(self.results.exists())

    def test_sbatchman_jobs_match_run_all_tables_and_append(self):
        args = ["all", "--config", str(self.config), "--nodes", "1", "4",
                "--variants", "spgemm_two_sided", "trident_get",
                "--campaign-repeats", "2", "--runs", "1"]
        with contextlib.redirect_stdout(io.StringIO()):
            runner.main(args)

        def tables(root):
            result = {}
            for path in root.rglob("*.tsv"):
                with path.open(newline="", encoding="utf-8") as stream:
                    reader = csv.reader(stream, delimiter="\t")
                    result[path.relative_to(root)] = (tuple(next(reader)), Counter(map(tuple, reader)))
            return result

        reference = tables(self.results)
        self.assertEqual(len(reference), 12)  # six stages, two implementations
        for scheduler in ("slurm", "pbs"):
            with self.subTest(scheduler=scheduler):
                root = self.root / scheduler
                cfg = planner.experiments.options(args + ["--results-dir", str(root)])
                files = planner.make_plan(scheduler, planner.load_profile(scheduler, None), cfg,
                                          "check", self.root / "plan")
                snapshots = []

                def run_job(snapshot):
                    nodes = json.loads(snapshot.read_text(encoding="utf-8"))["nodes"][0]
                    nodefile = self.root / "PBS_NODEFILE"
                    nodefile.write_text("".join(("node%d\n" % n) * 2 for n in range(nodes)), encoding="ascii")
                    env = dict(os.environ, SLURM_JOB_ID="123", SLURM_JOB_NUM_NODES=str(nodes),
                               SLURM_NTASKS=str(nodes * 2), SLURM_CPUS_PER_TASK="3",
                               SLURM_TASKS_PER_NODE="2(x%d)" % nodes,
                               PBS_JOBID="123.server", PBS_NODEFILE=str(nodefile))
                    completed = subprocess.run([
                        sys.executable, str(REPO / "scripts/lib/allocated_experiment.py"),
                        "--scheduler", scheduler, "--action", "all", "--config", str(snapshot)
                    ], env=env, capture_output=True, text=True)
                    self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

                for name, settings in files.items():
                    if name.endswith(".json"):
                        snapshot = self.root / (scheduler + "-" + name)
                        snapshot.write_text(json.dumps(settings), encoding="utf-8")
                        snapshots.append(snapshot)
                        run_job(snapshot)
                # Jobs may append in a different order, but paths, schemas and
                # observations (including duplicates) match the direct runner.
                self.assertEqual(tables(root), reference)
                self.assertFalse((root / "campaigns").exists())

                run_job(snapshots[-1])  # a rerun adds data; exports must retain both node counts
                appended = tables(root)
                for path, (header, rows) in reference.items():
                    repeated = Counter({row: count for row, count in rows.items()
                                        if row[header.index("nodes")] == "4"})
                    self.assertEqual(appended[path], (header, rows + repeated))

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
    def test_run_all_shell_repetitions_preview(self):
        shutil.rmtree(self.data)
        shutil.rmtree(self.build)
        for count_args in (["5"], ["--campaign-repeats", "5"]):
            with self.subTest(count_args=count_args):
                result = subprocess.run([
                    "bash", str(REPO / "scripts/run_all.sh"), *count_args,
                    "--config", str(self.config), "--runs", "1", "--variants", "trident_get",
                    "--dry-run", "--mpi-launcher", "nonexistent-mpirun",
                ], cwd=self.root, env=dict(os.environ, PYTHON=sys.executable),
                    capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("30 MPI launches", result.stdout)
                self.assertEqual(result.stdout.count("Workflow repetition "), 5)
                self.assertIn("Workflow repetition 5/5", result.stdout)
                for action in runner.ANALYSES:
                    self.assertEqual(result.stdout.count(action + ": regenerate"), 5)
        self.assertFalse(self.results.exists())
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
