#!/usr/bin/env python3
"""Resource translation and allocation checks, without a cluster or SbatchMan."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts/lib"))
import prepare_sbatchman as planner
import allocated_experiment as allocated

try:
    import sbatchman
    from sbatchman import create_configs_from_file  # distinguish the package from the profiles directory
except ImportError:
    sbatchman = None


class SbatchManTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="spgemm scheduler ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cfg = planner.experiments.options([
            "all", "--nodes", "1", "4", "--ranks-per-node", "3", "--threads", "7",
            "--matrices", "cage8", "--results-dir", str(self.root / "results")])

    def plan(self, scheduler):
        profile = planner.load_profile(scheduler, None)
        return planner.make_plan(scheduler, profile, self.cfg, "check", self.root / "jobs")

    def test_slurm_and_pbs_allocate_the_same_physical_resources(self):
        slurm = self.plan("slurm")["configs.yaml"]["example-slurm"]["configs"]
        pbs = self.plan("pbs")["configs.yaml"]["example-pbs"]["configs"]
        for nodes, s, p in zip((1, 4), slurm, pbs):
            self.assertEqual((s["nodes"], s["ntasks"], s["tasks_per_node"], s["cpus_per_task"]),
                             (nodes, nodes * 3, 3, 8))
            self.assertEqual(s["mem"], "16384M")
            self.assertIn("#SBATCH --hint=nomultithread", s["custom_headers"])
            self.assertIn("#PBS -l select=%d:ncpus=24:mpiprocs=3:ompthreads=7:mem=16384mb" % nodes,
                          p["custom_headers"])
            self.assertIn("#PBS -l place=scatter:exclhost", p["custom_headers"])
            self.assertNotIn("cpus", p)  # no conflicting job-wide resource request
            self.assertNotIn("mem", p)

    def test_snapshots_isolate_node_counts_and_preserve_sampling(self):
        files = self.plan("pbs")
        snapshots = [value for name, value in files.items() if name.endswith(".json")]
        self.assertEqual(len({cfg["results_dir"] for cfg in snapshots}), 2)
        for nodes, cfg in zip((1, 4), snapshots):
            self.assertEqual(cfg["nodes"], [nodes])
            self.assertEqual(cfg["cpus_per_rank"], 8)
            self.assertEqual(cfg["runs"], self.cfg.runs)
            self.assertEqual(cfg["permutation_matrices"], ["cage8"])
        self.assertTrue(files["jobs.yaml"]["sequential"])
        for job in files["jobs.yaml"]["jobs"]:
            command = shlex.split(job["config_jobs"][0]["command"])
            self.assertEqual(command, ["bash", str(self.root / "jobs" / (job["config"] + ".sh"))])

    def test_preview_has_no_side_effects_and_generation_cannot_overwrite(self):
        output = self.root / "plan"
        args = ["--scheduler", "pbs", "--campaign", "check", "--output-dir", str(output)]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(planner.main(args + ["--dry-run"]), 0)
            self.assertFalse(output.exists())
            self.assertEqual(planner.main(args), 0)
            before = {p.name: p.read_bytes() for p in output.iterdir()}
            with self.assertRaises(FileExistsError):
                planner.main(args)
        self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})
        self.assertNotIn(b"\r\n", (output / "n1-r2-t4-c5.sh").read_bytes())

    def test_profile_and_placement_errors_are_rejected(self):
        for changes in ({"nodes": 4}, {"memory_per_node_mb": 0}, {"walltime": "00:00:00"},
                        {"walltime": "01:80:00"}, {"exclusive": "false"},
                        {"account": "ok\n#PBS -l ncpus=1"}, {"cluster_name": "../bad"}):
            with self.subTest(changes=changes):
                profile = self.root / "profile.json"
                profile.write_text(json.dumps(changes), encoding="utf-8")
                with self.assertRaises(ValueError):
                    planner.load_profile("pbs", profile)
        for field, value in (("hostfile", Path("hosts")), ("local_check", True), ("dry_run", True)):
            cfg = argparse.Namespace(**vars(self.cfg))
            setattr(cfg, field, value)
            with self.assertRaises(ValueError):
                planner.make_plan("pbs", planner.load_profile("pbs", None), cfg, "check", self.root)

    def test_module_setup_fails_before_benchmark_and_is_shell_quoted(self):
        profile = planner.load_profile("pbs", None)
        profile.update(module_init="/opt/module init.sh", modules=["gcc/12", "openmpi/4"],
                       python="/opt/python env/bin/python3")
        files = planner.make_plan("pbs", profile, self.cfg, "check", self.root)
        script = files["n1-r3-t7-c8.sh"]
        self.assertIn("set -euo pipefail\n", script)
        self.assertIn("source '/opt/module init.sh'\nmodule load gcc/12 openmpi/4", script)
        command = shlex.split(script.splitlines()[-1])
        self.assertEqual(command[:2], ["exec", "/opt/python env/bin/python3"])

    def test_slurm_allocation_must_match_request(self):
        cfg = argparse.Namespace(**vars(self.cfg))
        cfg.nodes = [4]
        env = dict(SLURM_JOB_ID="123", SLURM_JOB_NUM_NODES="4", SLURM_NTASKS="12",
                   SLURM_CPUS_PER_TASK="8", SLURM_TASKS_PER_NODE="3(x4)")
        allocated.check_allocation(cfg, "slurm", env)
        for key, value in (("SLURM_JOB_ID", ""), ("SLURM_JOB_NUM_NODES", "1"),
                           ("SLURM_NTASKS", "4"), ("SLURM_CPUS_PER_TASK", "7"),
                           ("SLURM_TASKS_PER_NODE", "2(x2),4(x2)")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                allocated.check_allocation(cfg, "slurm", dict(env, **{key: value}))

    def test_pbs_requires_distinct_hosts_and_correct_rank_slots(self):
        cfg = argparse.Namespace(**vars(self.cfg))
        cfg.nodes = [4]
        nodefile = self.root / "pbs nodes"
        env = dict(PBS_JOBID="123.server", PBS_NODEFILE=str(nodefile))
        nodefile.write_text("".join(("host%d\n" % n) * 3 for n in range(4)), encoding="utf-8")
        allocated.check_allocation(cfg, "pbs", env)
        for content in ("host0\n" * 12, "".join(("host%d\n" % n) * 2 for n in range(4))):
            nodefile.write_text(content, encoding="utf-8")
            with self.assertRaises(ValueError):
                allocated.check_allocation(cfg, "pbs", env)

    def test_missing_allocation_stops_before_mpi(self):
        config = self.root / "config.json"
        config.write_text(json.dumps({"nodes": [1]}), encoding="utf-8")
        with patch.dict(os.environ, {}, clear=True), patch.object(allocated.experiments, "main") as run:
            for scheduler in ("slurm", "pbs"):
                with self.assertRaises(ValueError):
                    allocated.main(["--scheduler", scheduler, "--action", "all", "--config", str(config)])
            run.assert_not_called()

    @unittest.skipIf(sbatchman is None, "Optional integration check: install SbatchMan requirements")
    def test_real_sbatchman_parser_and_submission_scripts(self):
        from sbatchman.config import project_config
        from sbatchman.core import launcher

        for scheduler in ("slurm", "pbs"):
            with self.subTest(scheduler=scheduler):
                output = self.root / scheduler
                with contextlib.redirect_stdout(io.StringIO()):
                    planner.main(["--scheduler", scheduler, "--campaign", "check",
                                  "--output-dir", str(output)])
                    sbatchman.init_project(output)
                # Exercise the upstream YAML reader, template builder and job metadata.
                # Only the final sbatch/qsub call is mocked: nothing reaches a cluster.
                with patch.object(project_config, "get_project_root", return_value=output / "SbatchMan"), \
                     patch.object(launcher, "get_cluster_name", return_value="example-" + scheduler), \
                     patch.object(launcher, "get_max_queued_jobs", return_value=None), \
                     patch.object(launcher, scheduler + "_submit", side_effect=[101, 102]) as submit, \
                     contextlib.redirect_stdout(io.StringIO()):
                    configs = sbatchman.create_configs_from_file(output / "configs.yaml")
                    jobs = sbatchman.launch_jobs_from_file(output / "jobs.yaml")
                self.assertEqual(len(configs), 2)
                self.assertEqual(len(jobs), 2)
                self.assertEqual(submit.call_count, 2)
                self.assertIsNone(submit.call_args_list[0].args[2])
                self.assertEqual(submit.call_args_list[1].args[2], 101)
                scripts = [call.args[0].read_text() for call in submit.call_args_list]
                if scheduler == "slurm":
                    self.assertIn("#SBATCH --nodes=4", scripts[1])
                    self.assertIn("#SBATCH --ntasks=8", scripts[1])
                    self.assertIn("#SBATCH --cpus-per-task=5", scripts[1])
                else:
                    self.assertIn("#PBS -l select=4:ncpus=10:mpiprocs=2:ompthreads=4:mem=16384mb", scripts[1])
                for script in scripts:
                    self.assertNotIn("{CMD}", script)
                    self.assertNotIn("{EXP_DIR}", script)
                    self.assertIn(str(output), script)


if __name__ == "__main__":
    unittest.main()
