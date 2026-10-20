"""Exercise recovery, failure isolation, provenance and submission without a cluster."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import recover_cresco_missing as recovery
e = recovery.e


class RecoveryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="recovery-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.source = self.root / "source"
        self.data = self.source / "matrices"
        self.original_build = self.source / "build"
        self.compact_build = self.root / "compact-bin"
        self.campaign = self.source / "results/tesi-003/definitiva-001"
        self.old = self.campaign / "rep-01/n1"
        self.output = self.root / "recovery"
        for directory in (self.data / "metadata", self.original_build, self.compact_build,
                          self.campaign / "metadata/plan", self.old):
            directory.mkdir(parents=True)
        for name in ("cage8", "reddit"):
            matrix = self.data / (name + ".mtx")
            matrix.write_text("%%MatrixMarket matrix coordinate real general\n2 2 3\n1 1 2\n1 2 1\n2 2 3\n")
            e.matrices.write_json(self.data / "metadata" / (name + ".source.json"), e.matrices.inspect_matrix(matrix))
            with contextlib.redirect_stdout(io.StringIO()):
                e.matrices.prepare({"id": name}, self.data, 42)
        for variant in e.PROTOCOLS:
            for build in (self.original_build, self.compact_build):
                path = build / variant
                path.write_text("fixture executable")
                path.chmod(0o755)
        settings = dict(nodes=[1], ranks_per_node=2, threads=63, cpus_per_rank=64,
                        runs=1, campaign_repeats=1, warmup=0, repeats=1, trials=1,
                        matrices=["cage8", "reddit"], pilot_matrices=["cage8"],
                        permutation_matrices=["cage8"], build_dir=str(self.original_build),
                        matrix_dir=str(self.data), results_dir=str(self.old), timeout=0,
                        mpi_launcher=sys.executable,
                        mpi_args=[str(REPO / "tests/fixtures/mock_experiment_mpi.py")])
        self.config = self.campaign / "metadata/plan/tesi003-r01-n1.json"
        recovery.save(self.config, settings)
        recovery.save(self.campaign / "metadata/profile.json",
                      e.read_json(REPO / "scripts/sbatchman/slurm.example.json"))
        self.cfg = e.options(["all", "--config", str(self.config)])
        self.log = self.root / "launches.jsonl"
        self.environment = patch.dict(os.environ, {
            "MOCK_LAUNCH_LOG": str(self.log), "MOCK_COMPACT_BUILD": str(self.compact_build),
            "SLURM_JOB_ID": "999", "SLURM_JOB_NUM_NODES": "1", "SLURM_NTASKS": "2",
            "SLURM_CPUS_PER_TASK": "64", "SLURM_TASKS_PER_NODE": "2"})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        with contextlib.redirect_stdout(io.StringIO()):
            cases = e.cases(self.cfg, "strong_scaling")
            headers = e.input_preflight(self.cfg, [("strong_scaling", cases)])
            e.run_experiment(self.cfg, "strong_scaling", cases[:1], headers)
            baseline = copy.copy(self.cfg)
            baseline.variants = [v for v in self.cfg.variants if v.startswith("spgemm_")]
            e.run_experiment(baseline, "strong_scaling", cases[1:], headers)
        self.original_files = {p: p.read_bytes() for p in self.old.rglob("*.tsv")}

    def prepare(self):
        with patch.object(recovery, "queue", return_value=[]), contextlib.redirect_stdout(io.StringIO()):
            manifest = recovery.prepare(self.source, self.campaign, self.output, [1], [1], self.compact_build)
        files = list(Path(manifest["legacy_build"]).iterdir()) + list(self.compact_build.iterdir())
        recovery.save(self.output / "metadata/binaries-sha256.json", {str(p): recovery.digest(p) for p in files})
        return manifest

    def launches(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def run_recovery(self, manifest):
        with contextlib.redirect_stdout(io.StringIO()):
            return recovery.run(manifest, 0)

    def test_only_missing_and_restart_is_idempotent(self):
        manifest = self.prepare()
        self.assertEqual((manifest["jobs"][0]["copied"], manifest["jobs"][0]["missing"]), (11, 29))
        before = len(self.launches())
        self.assertEqual(self.run_recovery(manifest), 0)
        launches = self.launches()[before:]
        self.assertEqual(len(launches), 29 + 15)  # New compact pilot, no old main cases repeated.
        main = [cmd for cmd in launches if cmd[cmd.index("--experiment") + 1] != "pilot"]
        compact = [cmd for cmd in main if any(str(self.compact_build) in arg for arg in cmd)]
        self.assertEqual(len(compact), 5)
        self.assertTrue(all(Path(cmd[cmd.index("--matrix-a") + 1]).stem == "reddit" for cmd in compact))
        self.assertEqual(self.run_recovery(manifest), 0)
        self.assertEqual(len(self.launches()), before + 44)
        self.assertEqual(self.original_files, {p: p.read_bytes() for p in self.original_files})
        target = Path(manifest["jobs"][0]["target"])
        self.assertTrue(e.read_json(target / "metadata/completion.json")["complete"])
        rows = e.read_table(target / "phase_analysis/trident_get.tsv", e.PHASE_FIELDS)
        self.assertEqual(len(rows), 5)
        self.assertEqual(sum(r["benchmark_protocol"].endswith(recovery.SUFFIX) for r in rows), 1)

    def test_failure_keeps_independent_work_and_can_resume(self):
        manifest = self.prepare()
        with patch.dict(os.environ, {"MOCK_FAIL_VARIANT": "trident_two_sided", "MOCK_FAIL_MATRIX": "reddit"}):
            self.assertEqual(self.run_recovery(manifest), 1)
        target = Path(manifest["jobs"][0]["target"])
        report = e.read_json(target / "metadata/completion.json")
        self.assertEqual(report["completed"], 35)
        self.assertFalse(report["complete"])
        self.assertEqual(len(report["failures"]), 5)
        before = len(self.launches())
        self.assertEqual(self.run_recovery(manifest), 0)
        self.assertEqual(len(self.launches()) - before, 15 + 5)

    def test_duplicate_resource_mismatch_and_changed_source_are_rejected(self):
        path = self.old / "strong_scaling/trident_get.tsv"
        rows = e.read_table(path, e.FIELDS)
        e.write_table(path, e.FIELDS, rows + rows)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.prepare()
        e.write_table(path, e.FIELDS, rows)
        rows[0]["threads_per_rank"] = "4"
        e.write_table(path, e.FIELDS, rows)
        with self.assertRaisesRegex(ValueError, "threads_per_rank"):
            self.prepare()
        path.write_bytes(self.original_files[path])
        manifest = self.prepare()
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "Original results changed"):
            self.run_recovery(manifest)

    def test_active_original_refused(self):
        with patch.object(recovery, "queue", return_value=[("1", "tesi003-r01-n1-tesi003-n1-r2-t63-c64", "RUNNING")]):
            with self.assertRaisesRegex(ValueError, "still queued"):
                recovery.prepare(self.source, self.campaign, self.output, [1], [1], self.compact_build)
        self.assertFalse(self.output.exists())

    def test_submit_scopes_hold_and_preserves_running_jobs(self):
        manifest = self.prepare()
        active = [("101", "tesi003-r02-n1-tesi003-n1-r2-t63-c64", "RUNNING"),
                  ("102", "tesi003-r03-n1-tesi003-n1-r2-t63-c64", "PENDING"),
                  ("103", "tesi003-r02-n16-tesi003-n16-r2-t63-c64", "PENDING"),
                  ("104", "other-campaign-n1", "PENDING")]
        with patch.object(recovery, "queue", return_value=active), \
                patch.object(recovery.subprocess, "run") as run, \
                patch.object(recovery.subprocess, "check_output", return_value="9999\n") as submit:
            recovery.submit(manifest)
            run.assert_called_once_with(["scontrol", "hold", "102"], check=True)
            self.assertFalse(any(arg.startswith("--dependency=") for arg in submit.call_args.args[0]))
            recovery.submit(manifest)
            self.assertEqual(submit.call_count, 1)
        script = recovery.script_for(manifest, 0)
        self.assertIn("--cpus-per-task=64", script)
        self.assertIn("--ntasks=2", script)
        self.assertNotIn("scancel", script)


if __name__ == "__main__":
    unittest.main()
