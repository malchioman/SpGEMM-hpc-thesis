import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_cresco_run_all as campaign


class RunAllPlanTest(unittest.TestCase):
    def test_extension_waits_for_each_existing_lane(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            profile = root / "profile.json"
            profile.write_text(json.dumps(dict(cluster_name="cresco8", partition="cresco8_cpu")))
            output = root / "extension"
            tails = [5402337, 5402342, 5402347, 5402352]
            args = ["--profile", str(profile), "--output", str(output), "--nodes", "9", "16",
                    "--after-jobs"] + [str(n) for n in tails]
            with contextlib.redirect_stdout(io.StringIO()), patch.object(Path, "is_file", return_value=True):
                campaign.main(args)
            configs = {c["name"]: c for c in json.loads(
                (output / "metadata/plan/configs.yaml").read_text())["cresco8"]["configs"]}
            expected = json.loads((output / "metadata/expected.json").read_text())
            self.assertEqual({(e["repetition"], e["nodes"]) for e in expected},
                             {(r, n) for r in range(1, 11) for n in (9, 16)})
            for lane, tail in enumerate(tails, 1):
                jobs = json.loads((output / ("metadata/plan/lane-%d.yaml" % lane)).read_text())
                self.assertTrue(jobs["sequential"])
                self.assertEqual(len(jobs["jobs"]), 5)
                for job in jobs["jobs"]:
                    headers = configs[job["config"]]["custom_headers"]
                    self.assertIn("#SBATCH --dependency=afterany:%d" % tail, headers)
            for item in expected:
                cfg = json.loads((output / "metadata/plan" / (item["tag"] + ".json")).read_text())
                self.assertEqual(cfg["nodes"], [item["nodes"]])
                self.assertEqual(cfg["timeout"], 0)
                self.assertEqual(cfg["campaign_repeats"], 1)

    def test_native_runner_coverage_and_isolation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            profile = root / "profile.json"
            profile.write_text(json.dumps(dict(cluster_name="cresco8", partition="cresco8_cpu")))
            output = root / "campaign"
            args = ["--profile", str(profile), "--output", str(output)]
            with contextlib.redirect_stdout(io.StringIO()), patch.object(Path, "is_file", return_value=True):
                campaign.main(args)
            expected = json.loads((output / "metadata/expected.json").read_text())
            self.assertEqual(len(expected), 20)
            self.assertEqual(len({e["results_dir"] for e in expected}), 20)
            self.assertEqual({(e["repetition"], e["nodes"]) for e in expected},
                             {(r, n) for r in range(1, 11) for n in (1, 4)})
            for lane in range(1, 5):
                jobs = json.loads((output / ("metadata/plan/lane-%d.yaml" % lane)).read_text())
                self.assertTrue(jobs["sequential"])
                self.assertEqual(len(jobs["jobs"]), 5)
            for item in expected:
                path = output / "metadata/plan" / (item["tag"] + ".json")
                settings = json.loads(path.read_text())
                self.assertEqual(settings["campaign_repeats"], 1)
                self.assertEqual(settings["runs"], 1)
                self.assertEqual(settings["timeout"], 0)
                self.assertEqual(settings["repeats"] * settings["trials"], 1)
                self.assertEqual(settings["nodes"], [item["nodes"]])
                self.assertEqual(len(settings["variants"]), 8)
                self.assertEqual(len(settings["matrices"]), 10)
                self.assertEqual(len(settings["permutation_matrices"]), 4)
                script = path.with_suffix(".sh").read_text()
                self.assertIn("bash scripts/run_all.sh --config", script)
                self.assertIn("allocated_experiment.check_allocation", script)
                self.assertIn("collect_machine.sh", script)
            configs = json.loads((output / "metadata/plan/configs.yaml").read_text())
            for cfg in configs["cresco8"]["configs"]:
                self.assertTrue(cfg["exclusive"])
                self.assertEqual(cfg["time"], "24:00:00")
                self.assertEqual(cfg["mem"], "450000M")
            with self.assertRaises(ValueError):
                campaign.main(args)


if __name__ == "__main__":
    unittest.main()
