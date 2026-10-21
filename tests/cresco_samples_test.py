"""Offline safety/protocol checks. Mock MPI does not test performance or C++ correctness."""
import contextlib
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
sys.path.insert(0, str(REPO / "scripts"))
import launch_cresco_main as launch
import matrices


class SamplesTest(unittest.TestCase):
    def setUp(self):
        parent = REPO / ".test-tmp/cresco-samples"
        parent.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name).resolve()
        self.root = self.repo / launch.OUTPUT
        self.inputs = self.repo / "external-matrices"
        (self.inputs / "metadata").mkdir(parents=True)
        self.defaults = launch.read(REPO / "scripts/experiments.json")
        for name in launch.RUNTIME:
            dest = self.repo / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / name, dest)
        (self.repo / "build").mkdir()
        for variant in self.defaults["variants"] + ["CMakeCache.txt"]:
            path = self.repo / "build" / variant
            path.write_text("fixture\n")
            path.chmod(0o755)
        for name in list(launch.BLOCKS[4][1]) + ["cage8"]:
            matrix = self.inputs / (name + ".mtx")
            matrix.write_text("%%MatrixMarket matrix coordinate real general\n4 4 4\n1 1 2\n2 2 2\n3 3 2\n4 4 2\n")
            matrices.write_json(self.inputs / "metadata" / (name + ".source.json"),
                                matrices.inspect_matrix(matrix))
            with contextlib.redirect_stdout(io.StringIO()):
                matrices.prepare({"id": name}, self.inputs, 42)
        for name in ("experiments", "allocated_experiment"):
            sys.modules.pop(name, None)
        self.addCleanup(lambda: [sys.modules.pop(n, None) for n in ("experiments", "allocated_experiment")])
        self.log = self.repo / "mpi.jsonl"
        env = patch.dict(os.environ, dict(MOCK_LAUNCH_LOG=str(self.log), SLURM_JOB_ID="99001",
                         SLURM_JOB_NUM_NODES="1", SLURM_NTASKS="2", SLURM_CPUS_PER_TASK="64",
                         SLURM_TASKS_PER_NODE="2"))
        env.start()
        self.addCleanup(env.stop)

    def prepare(self):
        def fake_git(repo, *args):
            return "main" if args[0] == "branch" else (launch.REQUIRED_MAIN if args[0] == "rev-parse" else "")
        with patch.object(launch, "git", side_effect=fake_git), \
                patch.object(launch.subprocess, "run") as archive, contextlib.redirect_stdout(io.StringIO()):
            launch.prepare(self.repo, self.inputs, self.root)
        self.assertEqual(archive.call_count, 1)
        return launch.verify(self.root)

    def mock_mpi(self):
        path = self.root / "metadata/n1.json"
        config = launch.read(path)
        config.update(mpi_launcher=sys.executable,
                      mpi_args=[str(REPO / "tests/fixtures/mock_experiment_mpi.py")])
        launch.write(path, config)
        manifest = launch.read(self.root / "manifest.json")
        manifest["sha256"][path.relative_to(self.root).as_posix()] = launch.digest(path)
        launch.write(self.root / "manifest.json", manifest)

    def test_plan_resources_coverage_and_partial_configuration(self):
        manifest = self.prepare()
        self.assertEqual(len(manifest["jobs"]), 35)
        e = launch.runner(self.root)
        for n in launch.NODES:
            cfg = e.options(["all", "--config", str(self.root / f"metadata/n{n}.json")])
            self.assertEqual((cfg.repeats, cfg.trials, cfg.runs, cfg.warmup, cfg.validate), (20, 1, 1, 0, "partial"))
            pairs = [(a, c[2].name, c[3].name) for b in range(1, 6)
                     for a, cases in [launch.selected(e, cfg, b)] for c in cases]
            self.assertEqual(len(pairs), 23)
            self.assertEqual(len(set(pairs)), 23)
            self.assertNotIn(("strong_scaling", "reddit.mtx", "reddit.mtx"), pairs)
            self.assertIn(("rectangular", "reddit.mtx", "reddit_restriction.mtx"), pairs)
            self.assertEqual(sum(j["expected"] for j in manifest["jobs"] if j["nodes"] == n), 184)
            for b in range(1, 6):
                script = (self.root / f"metadata/jobs/n{n}-b{b}.sbatch").read_text()
                for flag in (f"--nodes={n}", f"--ntasks={2*n}", "--ntasks-per-node=2",
                             "--cpus-per-task=64", "--mem=450000M", "--exclusive", "--time=24:00:00"):
                    self.assertIn(flag, script)

    def test_end_to_end_samples_checks_and_idempotent_blocks(self):
        self.prepare()
        self.mock_mpi()
        with contextlib.redirect_stdout(io.StringIO()):
            for b in range(1, 6):
                launch.run(self.root, 1, b)
            for b in range(1, 6):
                launch.run(self.root, 1, b)
        calls = [json.loads(line) for line in self.log.read_text().splitlines()]
        self.assertEqual(len(calls), 216)  # 24 full pilot + 8 partial pilot + 184 main.
        self.assertTrue(all("--partial" not in c and "--no-validate" not in c for c in calls[:24]))
        self.assertTrue(all("--partial" in c and c[c.index("--repeats")+1] == "20" for c in calls[24:]))
        e = launch.runner(self.root)
        cfg = e.options(["all", "--config", str(self.root / "metadata/n1.json")])
        self.assertEqual([len(launch.existing(e, cfg, a)) for a in launch.MAIN], [72, 32, 80])
        for action in launch.MAIN:
            for v in cfg.variants:
                rows = e.read_table(self.root / f"n1/{action}/{v}.tsv", e.FIELDS)
                self.assertTrue(all(len(json.loads(r["end_to_end_samples_seconds"])) == 20 for r in rows))
                self.assertTrue(all(r["validation"] == "PARTIAL_PASS" for r in rows))
        self.assertFalse((self.root / "n1/pilot").exists())
        self.assertTrue((self.repo / "test-results/tesi-004/partial20-main/n1/cage8-full/pilot").exists())
        self.assertTrue((self.root / "state/n1-b5.complete.json").exists())

    def test_pilot_failure_stops_before_main_and_retry_keeps_saved_checks(self):
        self.prepare()
        self.mock_mpi()
        with patch.dict(os.environ, {"MOCK_FAILURE": "validation"}), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError):
                launch.run(self.root, 1, 1)
        self.assertFalse((self.root / "state/n1-b1.complete.json").exists())
        self.assertFalse((self.root / "n1/strong_scaling").exists())
        self.assertEqual(len(self.log.read_text().splitlines()), 1)
        with self.assertRaisesRegex(ValueError, "precedente"), contextlib.redirect_stdout(io.StringIO()):
            launch.run(self.root, 1, 2)

    def test_wrong_allocation_and_modified_snapshot_refused(self):
        self.prepare()
        with patch.dict(os.environ, {"SLURM_JOB_NUM_NODES": "4"}), self.assertRaises(ValueError):
            launch.run(self.root, 1, 1)
        (self.root / "metadata/bin/trident_get").write_text("changed")
        with self.assertRaisesRegex(ValueError, "congelato"):
            launch.run(self.root, 1, 1)
        self.assertFalse(self.log.exists())

    def test_submission_dependencies_and_no_duplicates(self):
        self.prepare()
        counter = iter(range(99001, 99036))
        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 0, stdout=str(next(counter))+"\n" if "--parsable" in cmd else "", stderr="")
        with patch.object(launch.subprocess, "run", side_effect=fake_run) as calls, contextlib.redirect_stdout(io.StringIO()):
            launch.submit(self.root)
            self.assertEqual(calls.call_count, 42)
            self.assertTrue(all("--test-only" in c.args[0] for c in calls.call_args_list[:7]))
            submitted = [c.args[0] for c in calls.call_args_list[7:]]
            for index, cmd in enumerate(submitted):
                deps = [s for s in cmd if s.startswith("--dependency")]
                self.assertEqual(deps, [] if index % 5 == 0 else [f"--dependency=afterok:{99000+index}"])
            launch.submit(self.root)
            self.assertEqual(calls.call_count, 42)

    def test_failed_preflight_and_uncertain_submit_do_not_submit_again(self):
        self.prepare()
        with patch.object(launch.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "sbatch")):
            with self.assertRaises(subprocess.CalledProcessError):
                launch.submit(self.root)
        self.assertEqual(list((self.root / "submission").glob("*.json")), [])
        launch.write(self.root / "submission/n1-b1.json", dict(state="submitting"))
        with patch.object(launch.subprocess, "run") as calls, self.assertRaisesRegex(ValueError, "incerto"):
            launch.submit(self.root)
        calls.assert_not_called()


if __name__ == "__main__":
    unittest.main()
