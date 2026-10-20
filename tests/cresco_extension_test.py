"""Offline launcher checks; the MPI fixture does not measure real performance."""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import extend_cresco_original as extension
import matrices


class ExtensionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="cresco-extension-")
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name).resolve()
        self.root = self.repo / extension.RELATIVE_OUTPUT
        self.original = self.repo / "results/tesi-003/definitiva-001/metadata"
        self.recovery = self.repo / "results/tesi-003/recovery-reddit-mem01/metadata"
        self.matrix_dir = self.repo / "matrices"
        (self.matrix_dir / "metadata").mkdir(parents=True)
        (self.original / "plan").mkdir(parents=True)
        (self.recovery / "legacy-bin").mkdir(parents=True)
        self.settings = extension.read(REPO / "scripts/experiments.json")
        self.settings.update(extension.read(REPO / "scripts/experiments.cresco-thesis.json"))
        self.settings["matrices"] = [entry["id"] for entry in
            extension.read(REPO / "scripts/matrices_catalog.json")["matrices"] if entry["tier"] == "all"]
        self.settings.update(nodes=[16], ranks_per_node=2, threads=63, cpus_per_rank=64,
                             runs=1, campaign_repeats=1, warmup=0, repeats=1, trials=1,
                             schedule="guided", chunk=64, validate=False, seed=42, timeout=0,
                             matrix_dir=str(self.matrix_dir), build_dir=str(self.repo / "build"),
                             results_dir=str(self.original.parent / "rep-01/n16"),
                             mpi_launcher=sys.executable,
                             mpi_args=[str(REPO / "tests/fixtures/mock_experiment_mpi.py")])
        for name in self.settings["matrices"] + ["cage8"]:
            matrix = self.matrix_dir / (name + ".mtx")
            matrix.write_text("%%MatrixMarket matrix coordinate real general\n4 4 4\n1 1 2\n2 2 2\n3 3 2\n4 4 2\n")
            matrices.write_json(self.matrix_dir / "metadata" / (name + ".source.json"),
                                matrices.inspect_matrix(matrix))
            with contextlib.redirect_stdout(io.StringIO()):
                matrices.prepare({"id": name}, self.matrix_dir, 42)
        shutil.copytree(self.matrix_dir / "metadata", self.original / "matrices")
        extension.write(self.original / "plan/tesi003-r01-n16.json", self.settings)
        extension.write(self.original / "profile.json", dict(partition="cresco8_cpu", account=None,
                        memory_per_node_mb=450000, walltime="24:00:00", exclusive=True,
                        module_init="/usr/share/Modules/init/bash",
                        modules=["intel/oneapi", "mpi_flavour/openmpi_intel-4.1.7"], python=sys.executable))
        for name in ("CMakeCache.txt", "git-commit.txt", "git-diff.patch"):
            extension.write(self.original / name, "fixture\n")
        with tarfile.open(self.original / "source.tar.gz", "w:gz") as archive:
            for name in extension.RUNTIME_FILES:
                archive.add(REPO / name, arcname=name)
        for variant in self.settings["variants"]:
            path = self.recovery / "legacy-bin" / variant
            path.write_text("original fixture executable " + variant)
            path.chmod(0o755)
        extension.write(self.recovery / "legacy-sha256.json",
                        {p.name: extension.digest(p) for p in (self.recovery / "legacy-bin").iterdir()})
        self.log = self.repo / "launches.jsonl"
        self.environment = patch.dict(os.environ, dict(MOCK_LAUNCH_LOG=str(self.log),
            SLURM_JOB_ID="99001", SLURM_JOB_NUM_NODES="36", SLURM_NTASKS="72",
            SLURM_CPUS_PER_TASK="64", SLURM_TASKS_PER_NODE="2(x36)"))
        self.environment.start()
        self.addCleanup(self.environment.stop)
        sys.modules.pop("experiments", None)
        self.addCleanup(lambda: sys.modules.pop("experiments", None))

    def prepare(self):
        with contextlib.redirect_stdout(io.StringIO()):
            extension.prepare(self.repo, self.root)
        return extension.verify(self.root)

    def test_full_first_repetition_and_large_topologies(self):
        manifest = self.prepare()
        e = sys.modules["experiments"]
        for job in manifest["jobs"]:
            config = extension.read(self.root / job["config"])
            for key in self.settings:
                if key not in ("nodes", "build_dir", "results_dir"):
                    self.assertEqual(config[key], self.settings[key], key)
            preview = io.StringIO()
            with contextlib.redirect_stdout(preview):
                e.main(["all", "--config", str(self.root / job["config"]), "--dry-run"])
            self.assertIn("208 MPI launches", preview.getvalue())
            self.assertIn("-np " + str(job["nodes"] * 2), preview.getvalue())
            self.assertNotIn("strong_scaling: reddit.mtx / reddit.mtx", preview.getvalue())
            self.assertIn("rectangular: reddit.mtx / reddit_restriction.mtx", preview.getvalue())
            script = (self.root / job["script"]).read_text()
            self.assertIn(f"#SBATCH --nodes={job['nodes']}", script)
            self.assertIn(f"#SBATCH --ntasks={2*job['nodes']}", script)
            self.assertIn("#SBATCH --mem=450000M", script)
            self.assertIn("#SBATCH --time=24:00:00", script)
            self.assertNotIn("build-compact", script)
        with contextlib.redirect_stdout(io.StringIO()):
            extension.run(self.root, 36)
        calls = [json.loads(line) for line in self.log.read_text().splitlines()]
        self.assertEqual(len(calls), 208)
        self.assertTrue(all(call[call.index("-np") + 1] == "72" for call in calls))
        result = extension.read(self.root / "rep-01/n36/completion.json")
        self.assertTrue(result["complete"])
        self.assertEqual(result["counts"], dict(pilot=24, strong_scaling=72, permutation=32, rectangular=80))
        for action in extension.MAIN:
            for variant in self.settings["variants"]:
                rows = e.read_table(self.root / f"rep-01/n36/{action}/{variant}.tsv", e.FIELDS)
                self.assertTrue(all(row["benchmark_protocol"] == e.PROTOCOLS[variant] for row in rows))
                self.assertTrue(all(row["validation"] == "SKIPPED" for row in rows))
        with self.assertRaisesRegex(ValueError, "duplicate repetition"):
            extension.run(self.root, 36)
        self.assertEqual(len(self.log.read_text().splitlines()), 208)
        with contextlib.redirect_stdout(io.StringIO()) as output, patch.object(extension.subprocess, "run"):
            extension.status(self.root)
        self.assertIn("principali 184/184 | pilot 24/24", output.getvalue())

    def test_bad_allocation_binary_or_input_refused_before_launch(self):
        self.prepare()
        for values in (dict(SLURM_JOB_NUM_NODES="64"), dict(SLURM_TASKS_PER_NODE="1(x36)")):
            with patch.dict(os.environ, values), self.assertRaises(ValueError):
                extension.run(self.root, 36)
        path = self.root / "metadata/legacy-bin/trident_get"
        contents = path.read_bytes()
        path.write_bytes(b"changed executable")
        with self.assertRaisesRegex(ValueError, "Frozen file changed"):
            extension.run(self.root, 36)
        path.write_bytes(contents)
        metadata = self.matrix_dir / "metadata/reddit.source.json"
        extension.write(metadata, {})
        with self.assertRaisesRegex(ValueError, "provenance differs"):
            extension.run(self.root, 36)
        self.assertFalse(self.log.exists())

    def test_all_requests_checked_first_and_submission_not_duplicated(self):
        self.prepare()
        counter = iter(("99001", "99002", "99003"))

        def fake_run(command, **kwargs):
            value = next(counter) + "\n" if "--parsable" in command else ""
            return subprocess.CompletedProcess(command, 0, stdout=value, stderr="")

        with patch.object(extension.subprocess, "run", side_effect=fake_run) as calls, \
                contextlib.redirect_stdout(io.StringIO()):
            extension.submit(self.root)
            self.assertEqual(calls.call_count, 6)
            self.assertTrue(all("--test-only" in c.args[0] for c in calls.call_args_list[:3]))
            self.assertTrue(all("--parsable" in c.args[0] for c in calls.call_args_list[3:]))
            extension.submit(self.root)
            self.assertEqual(calls.call_count, 6)

    def test_rejected_request_or_uncertain_submission_prevents_duplicates(self):
        self.prepare()
        with patch.object(extension.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "sbatch")) as calls:
            with self.assertRaises(subprocess.CalledProcessError):
                extension.submit(self.root)
            self.assertEqual(calls.call_count, 1)
        self.assertFalse(any((self.root / "submission").glob("*.json")))
        extension.write(self.root / "submission/36.json", dict(state="submitting"))
        with patch.object(extension.subprocess, "run") as calls:
            with self.assertRaisesRegex(ValueError, "Uncertain earlier submission"):
                extension.submit(self.root)
            calls.assert_not_called()

    def test_separate_100_node_job_preserves_existing_frozen_plan(self):
        self.prepare()
        previous = {p: p.read_bytes() for p in self.root.rglob("*")
                    if p.is_file() and "__pycache__" not in p.parts}
        with self.assertRaisesRegex(ValueError, "different node plan"):
            extension.prepare(self.repo, self.root, nodes=(100,))
        replacement = self.repo / "results/tesi-003/extension-100"
        # The real CLI uses a fresh Python process for each action.
        sys.modules.pop("experiments", None)
        with contextlib.redirect_stdout(io.StringIO()):
            extension.prepare(self.repo, replacement, nodes=(100,))
        manifest = extension.verify(replacement)
        self.assertEqual(len(manifest["jobs"]), 1)
        job = manifest["jobs"][0]
        self.assertEqual(job["nodes"], 100)
        self.assertEqual(job["name"], "tesi003-ext-r01-n100")
        script = (replacement / job["script"]).read_text()
        self.assertIn("#SBATCH --nodes=100", script)
        self.assertIn("#SBATCH --ntasks=200", script)
        self.assertIn("#SBATCH --ntasks-per-node=2", script)
        e = sys.modules["experiments"]
        cfg = e.options(["all", "--config", str(replacement / job["config"])])
        preview = io.StringIO()
        with contextlib.redirect_stdout(preview):
            e.main(["all", "--config", str(replacement / job["config"]), "--dry-run"])
        self.assertIn("208 MPI launches", preview.getvalue())
        self.assertIn("-np 200", preview.getvalue())
        self.assertNotIn("strong_scaling: reddit.mtx / reddit.mtx", preview.getvalue())
        self.assertIn("rectangular: reddit.mtx / reddit_restriction.mtx", preview.getvalue())
        # Exercise the CLI path selection and its default root without Slurm.
        command = [sys.executable, str(REPO / "scripts/extend_cresco_original.py"),
                   "prepare", "--repo", str(self.repo), "--nodes", "100"]
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        self.assertIn("Already prepared: " + str(replacement), result.stdout)
        # One fixture product checks resource/protocol normalization at 100 nodes.
        case = e.cases(cfg, "pilot")[0]
        with contextlib.redirect_stdout(io.StringIO()):
            headers = e.input_preflight(cfg, [("pilot", [case])])
            cfg.variants = ["trident_get"]
            e.run_experiment(cfg, "pilot", [case], headers)
        row = e.read_table(e.output_root(cfg, "pilot") / "pilot/trident_get.tsv", e.FIELDS)[0]
        self.assertEqual((row["nodes"], row["ranks"], row["validation"]), ("100", "200", "PASS"))
        self.assertEqual(previous, {p: p.read_bytes() for p in previous})

    def test_seventy_jobs_have_separate_results_and_seven_safe_chains(self):
        nodes = (1, 4, 9, 16, 36, 64, 100)
        self.root = self.repo / "results/tesi-004/definitiva-001"
        with contextlib.redirect_stdout(io.StringIO()):
            extension.prepare(self.repo, self.root, nodes=nodes, repetitions=10,
                              campaign="tesi004", follow_extensions=True)
        manifest = extension.verify(self.root)
        jobs = manifest["jobs"]
        self.assertEqual(len(jobs), 70)
        self.assertEqual(len({j["target"] for j in jobs}), 70)
        self.assertEqual(sum(j["expected"][a] for j in jobs for a in extension.MAIN), 12880)
        self.assertEqual(sum(j["expected"]["pilot"] for j in jobs), 1680)
        e = sys.modules["experiments"]
        for job in jobs:
            config = extension.read(self.root / job["config"])
            self.assertEqual(config["campaign_repeats"], 1)
            self.assertEqual((config["runs"], config["repeats"], config["trials"], config["warmup"]), (1, 1, 1, 0))
            self.assertEqual(config["nodes"], [job["nodes"]])
            offset = (job["repetition"] - 1) % 8
            expected = self.settings["variants"][offset:] + self.settings["variants"][:offset]
            self.assertEqual(config["variants"], expected)
            script = (self.root / job["script"]).read_text()
            self.assertIn("--repetition " + str(job["repetition"]), script)
            cfg = e.options(["all", "--config", str(self.root / job["config"])])
            self.assertEqual(len(e.cases(cfg, "strong_scaling")), 9)
            self.assertEqual(len(e.cases(cfg, "rectangular")), 10)
        counter = iter(str(n) for n in range(80000, 80070))

        def fake_run(command, **kwargs):
            value = next(counter) + "\n" if "--parsable" in command else ""
            return subprocess.CompletedProcess(command, 0, stdout=value, stderr="")

        queue = "5424540|tesi003-ext-r01-n36|RUNNING\n5424541|tesi003-ext-r01-n64|PENDING\n"
        queue += "5425000|tesi003-ext-r01-n100|PENDING\n9999|unrelated|RUNNING\n"
        with patch.dict(os.environ, {"USER": "fixture"}), \
                patch.object(extension.subprocess, "check_output", return_value=queue) as query, \
                patch.object(extension.subprocess, "run", side_effect=fake_run) as calls, \
                contextlib.redirect_stdout(io.StringIO()):
            extension.submit(self.root)
            self.assertEqual(calls.call_count, 77)
            self.assertTrue(all("--test-only" in c.args[0] for c in calls.call_args_list[:7]))
            for index, job in enumerate(jobs):
                cmd = calls.call_args_list[index + 7].args[0]
                deps = [c for c in cmd if c.startswith("--dependency=")]
                if job["repetition"] > 1:
                    self.assertEqual(deps, ["--dependency=afterok:" + str(80000 + index - 7)])
                elif job["nodes"] in (36, 64, 100):
                    prior = {36: "5424540", 64: "5424541", 100: "5425000"}[job["nodes"]]
                    self.assertEqual(deps, ["--dependency=afterany:" + prior])
                else:
                    self.assertEqual(deps, [])
                receipt = extension.read(extension.receipt_path(self.root, job))
                self.assertEqual(receipt["job_id"], str(80000 + index))
            extension.submit(self.root)
            self.assertEqual(calls.call_count, 77)
            query.assert_called_once()
        # Run selection must use the repetition, not just the first matching node count.
        cfg_seen = []
        def fake_workflow(argv):
            cfg_seen.append(extension.read(Path(argv[argv.index("--config") + 1]))["results_dir"])
            return 0
        with patch.dict(os.environ, {"SLURM_JOB_NUM_NODES": "1", "SLURM_NTASKS": "2",
                                     "SLURM_TASKS_PER_NODE": "2"}), \
                patch.object(e, "main", side_effect=fake_workflow), \
                patch.object(e, "read_table", return_value=[]):
            with self.assertRaisesRegex(ValueError, "Incomplete workflow"):
                extension.run(self.root, 1, repetition=10)
        self.assertEqual(cfg_seen, [str(self.root / "rep-10/n1")])


if __name__ == "__main__":
    unittest.main()
