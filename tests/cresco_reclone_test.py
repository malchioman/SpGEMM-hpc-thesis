"""Exercise the reset script's job selection with a fake squeue, without mutation."""
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/reclone_cresco_main.sh"
SOURCE = SCRIPT.read_text(encoding="utf-8")
FUNCTION = re.search(r"(?ms)^scoped_jobs\(\) \{.*?^\}", SOURCE).group()
REPO = "/lpor1/store_0/usr/malchiod/tirocinio-spgemm/SpGEMM-hpc-thesis"
BASH = os.environ.get("CRESCO_TEST_BASH") or shutil.which("bash")


@unittest.skipUnless(BASH, "Bash is needed for job-selection tests")
class JobSelectionTest(unittest.TestCase):
    def select(self, rows, status=0):
        harness = "\n".join([
            "set -euo pipefail",
            "fail() { printf '%s\\n' \"$*\" >&2; exit 1; }",
            "repo=" + shlex.quote(REPO),
            "requested_parent=" + shlex.quote(str(Path(REPO).parent).replace("\\", "/")),
            "squeue() { printf '%s\\n' " + shlex.quote(rows) + "; return " + str(status) + "; }",
            FUNCTION,
            'selected=$(scoped_jobs)',
            'printf "%s" "$selected"',
        ])
        return subprocess.run([BASH, "-c", harness], text=True, capture_output=True)

    def test_known_campaign_and_checkout_only(self):
        result = self.select("\n".join([
            "100|tesi004-r01-n1|/home/malchiod",
            "101|tesi003-ext-r01-n100|/home/malchiod",
            "102|pilot|" + REPO,
            "103|another-benchmark|" + REPO + "/results/frozen",
            "104|unrelated|/home/malchiod",
            "105|unrelated|" + REPO + "-different",
            "106|another-tesi004-r01-n1|/elsewhere",
        ]))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["100", "101", "102", "103"])

    def test_empty_queue(self):
        result = self.select("")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")

    def test_scheduler_failure_is_not_an_empty_queue(self):
        result = self.select("", status=7)
        self.assertEqual(result.returncode, 7)

    def test_unexpected_job_id_stops_the_reset(self):
        result = self.select("100_[1-9]|tesi004-r01-n1|/home/malchiod")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("100_[1-9]", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
