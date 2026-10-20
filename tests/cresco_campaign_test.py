#!/usr/bin/env python3
"""Offline checks for campaign coverage, isolation and diagnostic wrappers."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_cresco_campaign as campaign


class CrescoCampaignTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "results"
        self.profile = self.root / "profile.json"
        self.profile.write_text(json.dumps(dict(cluster_name="cresco8", partition="cresco8_cpu")))
        self.args = ["--output", str(self.output), "--campaign", "calibrazione-001",
                     "--profile", str(self.profile)]

    def test_dry_run_writes_nothing(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            campaign.main(self.args + ["--dry-run"])
        self.assertIn("48 sequential jobs; 384 expected observations", output.getvalue())
        self.assertFalse(self.output.exists())

    def test_coverage_frozen_settings_and_diagnostics(self):
        with patch.object(Path, "is_file", return_value=True), contextlib.redirect_stdout(io.StringIO()):
            campaign.main(self.args)
        plan = self.output / "metadata/plan"
        jobs = json.loads((plan / "jobs.yaml").read_text())
        self.assertTrue(jobs["sequential"])
        self.assertEqual(len(jobs["jobs"]), 48)
        tags = [j["config_jobs"][0]["tag"] for j in jobs["jobs"]]
        self.assertEqual(len(set(tags)), 48)
        frozen = list(plan.glob("*/*/*.json"))
        self.assertEqual(len(frozen), 48)
        counts = {}
        for path in frozen:
            cfg = json.loads(path.read_text())
            self.assertEqual(len(cfg["matrices"]), 1)
            self.assertIn(cfg["nodes"], ([1], [4]))
            self.assertEqual(cfg["runs"], 1)
            self.assertEqual(cfg["repeats"], 1)
            self.assertEqual(cfg["mpi_args"], ["--mca", "btl_vader_single_copy_mechanism", "none"])
            self.assertEqual(Path(cfg["results_dir"]), self.output)
            counts[path.parent.name] = counts.get(path.parent.name, 0) + 1
            wrapper = path.with_suffix(".sh").read_text()
            self.assertIn("collect_machine.sh", wrapper)
            self.assertIn("exit-code.txt", wrapper)
            self.assertNotIn("\nexec python", wrapper)
            self.assertIn("--action " + path.parent.name, wrapper)
        self.assertEqual(counts, dict(strong_scaling=20, rectangular=20, permutation=8))
        with self.assertRaisesRegex(ValueError, "new output"):
            campaign.main(self.args)

    def test_missing_inputs_do_not_publish_partial_plan(self):
        with patch.object(Path, "is_file", return_value=False):
            with self.assertRaisesRegex(ValueError, "Missing prepared input"):
                campaign.main(self.args)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
