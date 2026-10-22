"""Offline replacement of the pending preparation, never of original experiments."""
import getpass
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
import start_cresco_hv15r_early as starter
import launch_cresco_ten as ten


class EarlyStartTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.repo = Path(temporary.name) / 'repo'
        self.repo.mkdir()
        self.root = self.repo / 'results/tesi-006/hv15r-roundoff-cached-v4'
        self.cache = Path(temporary.name) / 'scratch'
        self.queue = '5487188|hvfull-cached-prepare|%s|PENDING\n' % getpass.getuser()
        self.race = False
        self.invalid_id = False
        self.calls = []
        self.body = ''
        for name, value in (
                ('validate_source', {}), ('guard_sources', None),
                ('missing_jobs', [{'action': a} for a in starter.recovery.ACTIONS])):
            mocked = patch.object(starter.recovery, name, return_value=value)
            setattr(self, name, mocked.start())
            self.addCleanup(mocked.stop)

        def command(cmd, **kwargs):
            self.calls.append(cmd)
            output = ''
            if cmd[0] == 'squeue':
                output = self.queue
            elif cmd[0] == 'scancel':
                self.queue = self.queue.replace('PENDING', 'RUNNING') if self.race else ''
            elif cmd[:2] == ['sbatch', '--parsable']:
                self.body = kwargs['input']
                output = 'unexpected' if self.invalid_id else '123456;cluster\n'
            return subprocess.CompletedProcess(cmd, 0, stdout=output, stderr='')

        mocked = patch.object(starter.subprocess, 'run', side_effect=command)
        mocked.start()
        self.addCleanup(mocked.stop)

    def start(self):
        starter.start(self.repo, '5487188', self.cache)

    def test_only_pending_preparation_is_replaced_and_monitor_id_updated(self):
        self.start()
        self.validate_source.assert_called_once_with(self.repo / 'results/tesi-006/partial10-cached', nodes=(4,))
        self.assertEqual([c for c in self.calls if c[0] == 'scancel'], [['scancel', '--state=PENDING', '5487188']])
        self.assertEqual([c for c in self.calls if c[:2] == ['sbatch', '--parsable']], [['sbatch', '--parsable']])
        self.assertNotIn('--dependency', self.body)
        self.assertIn('--early-probes', self.body)
        self.assertIn('submit-references', self.body)
        self.assertNotIn('submit-remaining', self.body)
        self.assertIn('--cpus-per-task=16', self.body)
        self.assertEqual((self.repo.parent / 'hvfull-cached-prepare.jobid').read_text(), '123456\n')
        receipt = ten.read(self.repo.parent / 'hvfull-cached-early-control/submission.json')
        self.assertEqual(receipt, dict(job_id='123456', replaced_job='5487188'))
        with self.assertRaisesRegex(ValueError, 'already recorded'):
            self.start()
        self.assertEqual(len([c for c in self.calls if c[0] == 'scancel']), 1)

    def test_different_job_owner_or_nonpending_state_is_not_cancelled(self):
        initial = self.queue
        for queue in ('', initial.replace('PENDING', 'RUNNING'),
                      initial.replace('hvfull-cached-prepare', 'cached10-n121-b03'),
                      initial.replace(getpass.getuser(), 'someone_else')):
            with self.subTest(queue=queue):
                self.queue = queue
                with self.assertRaisesRegex(ValueError, 'not your pending'):
                    self.start()
        self.assertFalse(any(c[0] in ('scancel', 'sbatch') for c in self.calls))

    def test_start_race_never_cancels_running_preparation_or_submits_replacement(self):
        self.race = True
        with self.assertRaisesRegex(ValueError, 'changed state'):
            self.start()
        self.assertIn(['scancel', '--state=PENDING', '5487188'], self.calls)
        self.assertFalse(any(c[:2] == ['sbatch', '--parsable'] for c in self.calls))
        self.assertFalse((self.repo.parent / 'hvfull-cached-prepare.jobid').exists())

    def test_active_four_node_source_and_existing_destination_block_before_cancellation(self):
        self.validate_source.side_effect = ValueError('four-node jobs still active')
        with self.assertRaisesRegex(ValueError, 'still active'):
            self.start()
        self.validate_source.side_effect = None
        self.root.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.start()
        self.assertEqual(self.calls, [])

    def test_resource_rejection_keeps_old_pending_job(self):
        real = starter.subprocess.run
        def reject(cmd, **kwargs):
            if cmd == ['sbatch', '--test-only']:
                raise subprocess.CalledProcessError(1, cmd)
            return real(cmd, **kwargs)
        with patch.object(starter.subprocess, 'run', side_effect=reject), self.assertRaises(subprocess.CalledProcessError):
            self.start()
        self.assertIn('PENDING', self.queue)
        self.assertFalse(any(c[0] == 'scancel' for c in self.calls))

    def test_ambiguous_submission_is_recorded_and_cannot_be_duplicated(self):
        self.invalid_id = True
        with self.assertRaisesRegex(ValueError, 'Ambiguous sbatch'):
            self.start()
        receipt = ten.read(self.repo.parent / 'hvfull-cached-early-control/submission.json')
        self.assertEqual(receipt['state'], 'submitting')
        with self.assertRaisesRegex(ValueError, 'already recorded'):
            self.start()
        self.assertEqual(len([c for c in self.calls if c[:2] == ['sbatch', '--parsable']]), 1)


if __name__ == '__main__':
    unittest.main()
