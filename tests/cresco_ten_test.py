"""Offline orchestration tests. Mock launches do not validate MPI or estimate runtime."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
sys.path.insert(0, str(REPO / 'tests'))
import cresco_samples_test as fixtures
import launch_cresco_main as old
import launch_cresco_ten as new


class TenTest(unittest.TestCase):
    def setUp(self):
        fixtures.SamplesTest.setUp(self)
        fixtures.SamplesTest.prepare(self)
        self.source = self.root
        for i, job in enumerate(old.read(self.source / 'manifest.json')['jobs']):
            old.write(self.source / 'submission' / (job['key'] + '.json'), dict(job_id=str(99001 + i)))
        self.root = self.repo / new.OUTPUT
        for name in ('experiments', 'allocated_experiment'):
            sys.modules.pop(name, None)
        with contextlib.redirect_stdout(io.StringIO()):
            new.prepare(self.repo, self.root)

    def mock_mpi(self, nodes=1):
        path = self.root / ('metadata/n%d.json' % nodes)
        cfg = new.read(path)
        cfg.update(mpi_launcher=sys.executable,
                   mpi_args=[str(REPO / 'tests/fixtures/mock_experiment_mpi.py')])
        new.write(path, cfg)
        manifest = new.read(self.root / 'manifest.json')
        manifest['sha256'][path.relative_to(self.root).as_posix()] = new.digest(path)
        new.write(self.root / 'manifest.json', manifest)

    def test_plan_frozen_executables_and_exact_coverage(self):
        manifest = new.verify(self.root)
        self.assertEqual(len(manifest['jobs']), 80)
        self.assertEqual(manifest['nodes'], [1, 4, 9, 16, 36, 64, 100, 121])
        base, e = new.runtime(self.root)
        for v in new.VARIANTS:
            self.assertEqual(new.digest(self.root / 'metadata/bin' / v),
                             new.digest(self.source / 'metadata/bin' / v))
        for n in new.NODES:
            cfg = e.options(['all', '--config', str(self.root / ('metadata/n%d.json' % n))])
            self.assertEqual((cfg.repeats, cfg.trials, cfg.warmup, cfg.validate), (10, 1, 0, 'partial'))
            pairs = []
            for job in (j for j in manifest['jobs'] if j['nodes'] == n):
                self.assertEqual(len(job['variants']), 4)
                for c in e.cases(cfg, job['action']):
                    if c[0] in job['matrices']:
                        pairs.extend((job['action'], c[2].name, c[3].name, v) for v in job['variants'])
                script = (self.root / job['script']).read_text()
                for flag in ('--time=24:00:00', '--mem=450000M', '--ntasks-per-node=2',
                             '--cpus-per-task=64', '--nodes=%d' % n, '--exclusive'):
                    self.assertIn(flag, script)
            self.assertEqual(len(pairs), 184)
            self.assertEqual(len(set(pairs)), 184)
            self.assertFalse(any(a == b == 'reddit.mtx' for _, a, b, _ in pairs))
            self.assertEqual(sum(a == 'reddit.mtx' for _, a, _, _ in pairs), 8)
        self.assertEqual([j['expected'] for j in manifest['jobs'][:10]], [12]*6 + [16]*2 + [40]*2)
        self.assertTrue((self.source / 'manifest.json').exists())

    def test_failed_block_does_not_block_next_and_rerun_skips_complete_rows(self):
        self.mock_mpi()
        base, e = new.runtime(self.root)
        original = e.run_experiment
        failed = []

        def fail_one(cfg, action, cases, headers):
            if (action == 'strong_scaling' and cases[0][0] == 'isolates_subgraph4'
                    and cfg.variants == ['trident_two_sided'] and not failed):
                failed.append(True)
                raise RuntimeError('Simulated failed variant')
            return original(cfg, action, cases, headers)

        with patch.object(e, 'run_experiment', side_effect=fail_one), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'prove fallite'):
                new.run(self.root, 1, 1)
            new.run(self.root, 1, 2)
        self.assertFalse(new.read(self.root / 'state/n1-b01.finished.json')['complete'])
        self.assertTrue(new.read(self.root / 'state/n1-b02.finished.json')['complete'])
        cfg = e.options(['all', '--config', str(self.root / 'metadata/n1.json')])
        self.assertEqual(len(base.existing(e, cfg, 'strong_scaling')), 23)
        before = len(self.log.read_text().splitlines())
        with contextlib.redirect_stdout(io.StringIO()):
            new.run(self.root, 1, 1)
            new.run(self.root, 1, 2)
        self.assertEqual(len(self.log.read_text().splitlines()), before + 1)
        self.assertEqual(len(base.existing(e, cfg, 'strong_scaling')), 24)
        for p in (self.root / 'n1/strong_scaling').glob('*.tsv'):
            for row in e.read_table(p, e.FIELDS):
                self.assertEqual(len(json.loads(row['end_to_end_samples_seconds'])), 10)
                self.assertEqual(row['validation'], 'PARTIAL_PASS')

    def test_pilot_failure_blocks_measurements_but_next_block_can_retry(self):
        self.mock_mpi()
        with patch.dict(os.environ, {'MOCK_FAILURE': 'validation'}), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError):
                new.run(self.root, 1, 1)
        self.assertFalse((self.root / 'n1/strong_scaling').exists())
        with contextlib.redirect_stdout(io.StringIO()):
            new.run(self.root, 1, 2)
        self.assertTrue(new.read(self.root / 'state/n1-b02.finished.json')['complete'])

    def test_atomic_append_survives_failed_replace_and_lock_survives_dead_process(self):
        _, e = new.runtime(self.root)
        table = self.repo / 'atomic/data.tsv'
        e.write_table(table, ['value'], [dict(value='first')], append=True)
        before = table.read_bytes()
        with patch.object(e.os, 'replace', side_effect=OSError('interrupted replace')):
            with self.assertRaises(OSError):
                e.write_table(table, ['value'], [dict(value='second')], append=True)
        self.assertEqual(table.read_bytes(), before)
        e.write_table(table, ['value'], [dict(value='second')], append=True)
        self.assertEqual(e.read_table(table, ['value']), [dict(value='first'), dict(value='second')])
        path = self.repo / 'lock-test'
        code = ('import sys, os; from pathlib import Path; '
                'sys.path.insert(0, sys.argv[1]); import launch_cresco_ten as m; '
                'ctx=m.file_lock(Path(sys.argv[2])); ctx.__enter__(); os._exit(9)')
        result = subprocess.run([sys.executable, '-c', code, str(REPO / 'scripts'), str(path)])
        self.assertEqual(result.returncode, 9)
        with new.file_lock(path):
            with self.assertRaises((RuntimeError, OSError)):
                with new.file_lock(path):
                    self.fail('Concurrent lock acquired')
        with new.file_lock(path):
            pass

    def test_submissions_use_afterany_and_are_idempotent(self):
        counter = iter(range(100001, 100081))
        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 0,
                stdout=(str(next(counter)) + '\n') if '--parsable' in cmd else '', stderr='')
        with patch.object(new.subprocess, 'run', side_effect=fake_run) as calls, contextlib.redirect_stdout(io.StringIO()):
            new.submit(self.root)
            self.assertEqual(calls.call_count, 88)
            submitted = [c.args[0] for c in calls.call_args_list if '--parsable' in c.args[0]]
            for i, cmd in enumerate(submitted):
                dependencies = [s for s in cmd if s.startswith('--dependency=')]
                self.assertEqual(dependencies, [] if i % 10 == 0 else ['--dependency=afterany:%d' % (100000+i)])
            new.submit(self.root)
            self.assertEqual(sum('--parsable' in c.args[0] for c in calls.call_args_list), 80)
        new.write(self.root / 'submission/n1-b01.json', dict(state='submitting'))
        with patch.object(new.subprocess, 'run'), self.assertRaisesRegex(ValueError, 'incerto'):
            new.submit(self.root)

    def test_cleanup_refuses_active_jobs_and_protects_inputs_and_new_campaign(self):
        checks = self.repo / new.OLD_CHECKS
        checks.mkdir(parents=True, exist_ok=True)
        (checks / 'old.log').write_text('old')
        sentinel = new.digest(self.inputs / 'cage8.mtx')
        with patch.object(new, 'old_queue', return_value=[('99001', 'RUNNING')]), \
                self.assertRaisesRegex(ValueError, 'ancora attivo'):
            new.retire_old(self.repo, self.root)
        self.assertTrue(self.source.exists())
        self.assertTrue(checks.exists())
        calls = []
        def command(cmd, **kwargs):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout='JobID|State\n99001|TIMEOUT\n', stderr='')
        states = [[('99002', 'PENDING')], [], [], []]
        with patch.object(new, 'old_queue', side_effect=states), \
                patch.object(new.subprocess, 'run', side_effect=command), contextlib.redirect_stdout(io.StringIO()):
            new.retire_old(self.repo, self.root)
        self.assertEqual(calls[0], ['scancel', '--state=PENDING', '99002'])
        self.assertFalse(self.source.exists())
        self.assertFalse(checks.exists())
        self.assertEqual(new.digest(self.inputs / 'cage8.mtx'), sentinel)
        new.verify(self.root)
        self.assertTrue((self.root / 'retired/accounting-tesi004.psv').exists())
        with self.assertRaises(ValueError):
            new.validate_cleanup_target(self.inputs, self.repo, self.inputs)

    def test_queue_refuses_unknown_jobs_before_mutation(self):
        for text in ('99001|wrong-name|PENDING|/tmp\n', '99999|tesi004-unknown|RUNNING|/tmp\n'):
            response = subprocess.CompletedProcess(['squeue'], 0, stdout=text, stderr='')
            with patch.object(new.subprocess, 'run', return_value=response), self.assertRaises(ValueError):
                new.old_queue(self.root, self.source)

    def test_new_node_count_allocation_and_single_launch(self):
        self.mock_mpi(121)
        base, e = new.runtime(self.root)
        cfg = e.options(['all', '--config', str(self.root / 'metadata/n121.json')])
        cfg.matrices, cfg.variants = ['mouse_gene'], ['trident_get_pipeline']
        with contextlib.redirect_stdout(io.StringIO()):
            new.perform(base, e, cfg, 'strong_scaling', e.cases(cfg, 'strong_scaling'))
        rows = e.read_table(self.root / 'n121/strong_scaling/trident_get_pipeline.tsv', e.FIELDS)
        self.assertEqual((rows[0]['nodes'], rows[0]['ranks'], rows[0]['repeats']), ('121', '242', '10'))
        with self.assertRaises(ValueError):
            new.run(self.root, 121, 1)  # Fixture allocation is one node.

    def test_full_lane_and_audit_keep_exact_samples_and_all_pairs(self):
        self.mock_mpi()
        with contextlib.redirect_stdout(io.StringIO()):
            for block in range(1, 11):
                new.run(self.root, 1, block)
            missing = new.status(self.root)
        self.assertTrue(new.read(self.root / 'state/n1.complete.json')['complete'])
        self.assertEqual(len(missing), 7 * 184)
        self.assertTrue(all(row['nodes'] != 1 for row in missing))
        calls = [json.loads(line) for line in self.log.read_text().splitlines()]
        self.assertEqual(len(calls), 24 + 8 + 184)
        self.assertTrue(all(c[c.index('--repeats') + 1] == '1' for c in calls[:24]))
        self.assertTrue(all(c[c.index('--repeats') + 1] == '10' for c in calls[24:]))
        self.assertTrue(all('--partial' in c for c in calls[24:]))
        self.assertTrue((self.root / 'n1/phase_analysis').exists())


if __name__ == '__main__':
    unittest.main()
