"""Test destructive-reset boundaries only inside a resolved local temporary workspace.

Git, Slurm, Linux identity, and mount information are mocked. No cluster access.
"""
from contextlib import ExitStack
import importlib.util
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('cresco_reset', REPO / 'scripts/reset_cresco_main.py')
RESET = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RESET)


class ResetTest(unittest.TestCase):
    def setUp(self):
        self.workspace = (REPO / '.test-tmp/cresco-reset').resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=self.workspace)
        self.root = Path(self.temporary.name).resolve()
        self.assertTrue(RESET.within(self.root, self.workspace))
        self.parent = self.root / 'store'
        self.scratch = self.root / 'scratch'
        self.parent.mkdir()
        self.scratch.mkdir()
        self.matrices = self.scratch / 'matrices'
        (self.matrices / 'metadata').mkdir(parents=True)
        (self.matrices / 'cage8.mtx').write_text('matrix cache sentinel')
        self.old = self.parent / RESET.REPO_NAME
        (self.old / 'results/old-campaign').mkdir(parents=True)
        (self.old / 'results/old-campaign/table.tsv').write_text('old result')
        self.backup = self.parent / 'SpGEMM-hpc-thesis.previous-20261020T120000Z-123'
        self.backup.mkdir()
        (self.backup / 'old-result.tsv').write_text('old backup')
        for base in (self.parent, self.scratch):
            (base / 'test-results').mkdir()
            (base / 'test-results/old.tsv').write_text('old test')
        self.unrelated = self.parent / 'unrelated'
        self.unrelated.mkdir()
        (self.unrelated / 'keep.txt').write_text('unrelated data')
        self.events = []
        self.cancelled = False
        self.stuck = False

    def tearDown(self):
        # reset() changes cwd; never leave it inside the test deletion tree.
        import os
        os.chdir(REPO)
        self.assertTrue(RESET.within(self.root.resolve(), self.workspace))
        self.temporary.cleanup()

    def fake_clone(self, argv, **kwargs):
        self.assertEqual(argv[:2], ['git', 'clone'])
        fresh = Path(argv[-1])
        self.assertEqual(fresh.parent, self.parent)
        fresh.mkdir()
        (fresh / 'fresh-main.txt').write_text('new source')
        self.events.append('clone')
        return subprocess.CompletedProcess(argv, 0)

    def fake_check(self, fresh):
        self.assertTrue((fresh / 'fresh-main.txt').is_file())
        self.events.append('verify')
        return RESET.REQUIRED_MAIN

    def fake_command(self, argv, **kwargs):
        self.events.append(argv[0])
        if argv[0] == 'squeue':
            rows = ['999|other-project|/home/malchiod']
            if not self.cancelled or self.stuck:
                rows += ['100|tesi001-r01-n1|/home/malchiod',
                         '101|tesi004-r01-n16|/home/malchiod',
                         '102|pilot|' + str(self.old),
                         '103|old-check|' + str(self.backup)]
            return '\n'.join(rows)
        if argv[0] == 'scancel':
            self.assertEqual(argv, ['scancel', '--quiet', '100', '101', '102', '103'])
            self.cancelled = True
            return ''
        raise AssertionError('Unexpected command: ' + str(argv))

    def environment(self):
        stack = ExitStack()
        stack.enter_context(patch.object(RESET, 'PARENT', self.parent))
        stack.enter_context(patch.object(RESET, 'MATRICES', self.matrices))
        stack.enter_context(patch.object(RESET.sys, 'platform', 'linux'))
        stack.enter_context(patch.object(RESET, 'current_user', return_value='malchiod'))
        stack.enter_context(patch.object(RESET.shutil, 'which', return_value='/mock/tool'))
        stack.enter_context(patch.object(RESET, 'mount_points', return_value=[]))
        stack.enter_context(patch.object(RESET, 'command', side_effect=self.fake_command))
        stack.enter_context(patch.object(RESET.subprocess, 'run', side_effect=self.fake_clone))
        stack.enter_context(patch.object(RESET, 'check_main', side_effect=self.fake_check))
        stack.enter_context(patch('sys.stdout', new=io.StringIO()))
        return stack

    def test_complete_reset_preserves_cache_and_unrelated_data(self):
        with self.environment():
            RESET.reset()
        self.assertTrue((self.old / 'fresh-main.txt').is_file())
        self.assertFalse((self.old / 'results').exists())
        self.assertFalse(self.backup.exists())
        self.assertFalse((self.parent / 'test-results').exists())
        self.assertFalse((self.scratch / 'test-results').exists())
        self.assertEqual((self.matrices / 'cage8.mtx').read_text(), 'matrix cache sentinel')
        self.assertEqual((self.unrelated / 'keep.txt').read_text(), 'unrelated data')
        self.assertLess(self.events.index('verify'), self.events.index('scancel'))
        self.assertEqual(RESET.discover_targets(self.parent, self.scratch), [self.old])

    def test_bad_clone_keeps_old_data_and_jobs(self):
        with self.environment(), patch.object(RESET, 'check_main', side_effect=RuntimeError('bad main')):
            with self.assertRaisesRegex(RuntimeError, 'bad main'):
                RESET.reset()
        self.assertFalse(self.cancelled)
        self.assertTrue((self.old / 'results/old-campaign/table.tsv').exists())
        self.assertTrue(self.backup.exists())

    def test_scheduler_error_keeps_old_data(self):
        with self.environment(), patch.object(RESET, 'command', side_effect=subprocess.CalledProcessError(1, 'squeue')):
            with self.assertRaises(subprocess.CalledProcessError):
                RESET.reset()
        self.assertTrue((self.old / 'results/old-campaign/table.tsv').exists())
        self.assertTrue(self.backup.exists())

    def test_jobs_still_running_prevent_cleanup(self):
        self.stuck = True
        with self.environment(), patch.object(RESET.time, 'monotonic', side_effect=[0, 61]):
            with self.assertRaisesRegex(RuntimeError, 'Job ancora presenti'):
                RESET.reset()
        self.assertTrue((self.old / 'results/old-campaign/table.tsv').exists())
        self.assertTrue(self.backup.exists())

    def test_forbidden_destinations(self):
        for target in (self.parent, self.matrices, self.unrelated):
            with self.subTest(target=target), self.assertRaises(RuntimeError):
                RESET.validate_target(target, {self.parent, self.scratch}, [self.matrices], [])
        with self.assertRaisesRegex(RuntimeError, 'dati protetti'):
            RESET.validate_target(self.old, {self.parent}, [self.old], [])

    def test_nested_mount_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'Mount point'):
            RESET.validate_target(self.old, {self.parent}, [self.matrices], [self.old / 'matrices'])

    def test_symlink_root_is_rejected(self):
        with patch.object(type(self.old), 'is_symlink', return_value=True):
            with self.assertRaisesRegex(RuntimeError, 'simbolica'):
                RESET.validate_target(self.old, {self.parent}, [self.matrices], [])

    def test_array_elements_and_unrelated_jobs(self):
        rows = '\n'.join(['100_2|tesi004-run|/elsewhere',
                          '999|other-project|/elsewhere',
                          '998|other-project|' + str(self.old) + '-different'])
        self.assertEqual(RESET.select_jobs(rows, [self.old]), ['100_2'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
