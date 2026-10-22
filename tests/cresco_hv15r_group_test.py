"""Offline grouped recovery: frozen per-case runner, fake scheduler and tiny MPI fixtures."""
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
import cresco_hv15r_test as fixtures
import group_cresco_hv15r as grouped
import recover_cresco_hv15r as recovery
import launch_cresco_ten as ten


class GroupedTest(unittest.TestCase):
    def setUp(self):
        fixtures.RecoveryTest.setUp(self)
        fixtures.RecoveryTest.run_probe(self)
        fixtures.RecoveryTest.run_probe(self, 'permutation')
        mock = patch.object(grouped, 'load', return_value=recovery)
        mock.start()
        self.addCleanup(mock.stop)
        self.parent_bytes = (self.root / 'manifest.json').read_bytes()
        self.metadata = ten.read(self.root / 'manifest.json')['sha256'].copy()
        self.probes = {p: p.read_bytes() for p in (self.root / 'n4').glob('*/*.tsv')}

    def prepare_submit(self):
        with contextlib.redirect_stdout(io.StringIO()):
            grouped.prepare(self.root)
            grouped.submit(self.root)

    def group_env(self):
        receipt = ten.read(self.root / 'grouped/submission/n4.json')
        return dict(SLURM_JOB_ID=receipt['job_id'], SLURM_JOB_NUM_NODES='4',
                    SLURM_NTASKS='8', SLURM_TASKS_PER_NODE='2(x4)')

    def fake_cases(self, fail=None, interrupt=None):
        previous = grouped.subprocess.run
        self.case_launches = []
        def command(cmd, **kwargs):
            if len(cmd) > 2 and cmd[1] in (str(self.root / 'metadata/source/scripts/recover_cresco_hv15r.py'),
                                         str(self.root / 'grouped/metadata/driver.py')):
                key = cmd[cmd.index('--key') + 1]
                self.case_launches.append(key)
                if key == interrupt:
                    raise OSError('simulated interruption')
                if key == fail:
                    return subprocess.CompletedProcess(cmd, 1)
                if cmd[2] == 'run-cached':
                    grouped.run_cached(self.root, key)
                else:
                    recovery.run(self.root, key)
                return subprocess.CompletedProcess(cmd, 0)
            return previous(cmd, **kwargs)
        return patch.object(grouped.subprocess, 'run', side_effect=command)

    def partial_references(self):
        manifest = ten.read(self.campaign / 'manifest.json')
        cache = Path(manifest['partial_cache'])
        cache.mkdir(exist_ok=True)
        for pair in manifest['reference_pairs']:
            key = '%064x' % pair['index']
            (cache / (key + '.spref')).write_bytes(b'fixture')
            ten.write(self.campaign / 'reference-ready' / ('%02d.json' % pair['index']),
                      dict(pair=pair, key=key,
                           generator_sha256=manifest['sha256']['metadata/bin/prepare_partial_reference']))

    def prepare_additional(self):
        self.partial_references()
        with contextlib.redirect_stdout(io.StringIO()):
            grouped.prepare(self.root, include_n100_b08=True)
            grouped.submit(self.root)

    def n100_env(self):
        receipt = ten.read(self.root / 'grouped/submission/n100.json')
        return dict(SLURM_JOB_ID=receipt['job_id'], SLURM_JOB_NUM_NODES='100',
                    SLURM_NTASKS='200', SLURM_TASKS_PER_NODE='2(x100)')

    def test_plan_skips_probes_preserves_snapshot_resources_and_has_no_dependencies(self):
        self.prepare_submit()
        _, plan, _, _ = grouped.verify(self.root)
        # Fixture has no original PASS rows: all eight node counts need recovery.
        self.assertEqual(len(plan['groups']), 8)
        self.assertEqual(sum(len(g['cases']) for g in plan['groups']), 78)
        self.assertEqual(len(next(g['cases'] for g in plan['groups'] if g['nodes'] == 4)), 8)
        for group in plan['groups']:
            script = (self.root / 'grouped' / group['script']).read_text()
            for flag in ('--time=24:00:00', '--mem=450000M', '--exclusive', '--cpus-per-task=64',
                         '--ntasks-per-node=2', '--hint=nomultithread', 'collect_machine.sh'):
                self.assertIn(flag, script)
            self.assertNotIn('--dependency', script)
            if os.name != 'nt':
                fixtures.REAL_RUN(['bash', '-n'], input=script, text=True, check=True)
        self.assertEqual((self.root / 'manifest.json').read_bytes(), self.parent_bytes)
        self.assertEqual(ten.verify(self.root)['sha256'], self.metadata)
        self.assertEqual({p: p.read_bytes() for p in self.probes}, self.probes)
        self.assertEqual((self.campaign / 'manifest.json').read_bytes(), self.original_manifest)

    def test_frozen_group_driver_imports_the_deployed_recovery_in_an_isolated_process(self):
        with contextlib.redirect_stdout(io.StringIO()):
            grouped.prepare(self.root)
        code = (
            'import importlib.util, pathlib, sys; '
            'root=pathlib.Path(sys.argv[1]); '
            'spec=importlib.util.spec_from_file_location("driver",root/"grouped/metadata/driver.py"); '
            'driver=importlib.util.module_from_spec(spec); spec.loader.exec_module(driver); '
            'r=driver.load(root); '
            'assert pathlib.Path(r.__file__).resolve()==root/"metadata/source/scripts/recover_cresco_hv15r.py"; '
            'assert pathlib.Path(r.ten.__file__).resolve().parent==root/"metadata/source/scripts"'
        )
        fixtures.REAL_RUN([sys.executable, '-c', code, str(self.root)], cwd=self.root, check=True)

    def test_submissions_are_idempotent_and_original_submitter_cannot_duplicate_them(self):
        self.prepare_submit()
        with contextlib.redirect_stdout(io.StringIO()):
            grouped.prepare(self.root)
            grouped.submit(self.root)
            recovery.submit(self.root, remaining=True)
        commands = [c for c in self.calls if c[:2] == ['sbatch', '--parsable']]
        self.assertEqual(len(commands), 8)
        self.assertFalse(any('--dependency' in arg for c in commands for arg in c))
        _, plan, _, _ = grouped.verify(self.root)
        for group in plan['groups']:
            receipt = ten.read(self.root / 'grouped/submission' / (group['key'] + '.json'))
            for key in group['cases']:
                self.assertEqual(ten.read(self.root / 'submission' / (key + '.json'))['job_id'], receipt['job_id'])

    def test_accepted_node_count_is_omitted_from_groups(self):
        original = recovery.complete
        def completed(root, source, job):
            return job['nodes'] == 1 or original(root, source, job)
        with patch.object(recovery, 'complete', side_effect=completed):
            self.prepare_submit()
            _, plan, _, _ = grouped.verify(self.root)
        self.assertEqual([g['nodes'] for g in plan['groups']], [4, 9, 16, 36, 64, 100, 121])
        self.assertEqual(sum(len(g['cases']) for g in plan['groups']), 68)

    def test_group_runs_only_remaining_cases_with_ten_samples_and_resumes_without_remeasurement(self):
        self.prepare_submit()
        with self.fake_cases(), patch.dict(os.environ, self.group_env()), contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(grouped.run(self.root, 4))
            self.assertEqual(len(self.case_launches), 8)
            before = self.log.read_bytes()
            self.assertTrue(grouped.run(self.root, 4))
            self.assertEqual(len(self.case_launches), 8)
        self.assertEqual(self.log.read_bytes(), before)
        self.assertEqual({p: p.read_bytes() for p in self.probes}, self.probes)
        self.assertEqual((self.root / 'manifest.json').read_bytes(), self.parent_bytes)
        state = ten.read(self.root / 'grouped/state/n4.json')
        self.assertTrue(state['complete'])
        for action in recovery.ACTIONS:
            for variant in recovery.VARIANTS:
                _, e, _ = recovery.runtime(self.root, self.campaign, 4, variant, action)
                rows = e.read_table(self.root / 'n4' / action / (variant + '.tsv'), e.FIELDS)
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]['validation'], 'PASS')
                self.assertEqual(rows[0]['repeats'], '10')

    def test_failure_continues_other_cases_and_keeps_group_incomplete(self):
        self.prepare_submit()
        failed = 'n4-trident_hybrid'
        with self.fake_cases(fail=failed), patch.dict(os.environ, self.group_env()), contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(grouped.run(self.root, 4))
        self.assertEqual(len(self.case_launches), 8)
        state = ten.read(self.root / 'grouped/state/n4.json')
        self.assertFalse(state['complete'])
        self.assertEqual(state['cases'][failed]['status'], 'FAILED')
        self.assertEqual(sum(r['status'] == 'PASS' for r in state['cases'].values()), 7)
        with self.fake_cases(), patch.dict(os.environ, self.group_env()), contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(grouped.run(self.root, 4))
        self.assertEqual(self.case_launches, [failed])

    def test_interruption_preserves_already_written_observations(self):
        self.prepare_submit()
        with self.fake_cases(interrupt='n4-trident_get'), patch.dict(os.environ, self.group_env()), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(OSError, 'interruption'):
            grouped.run(self.root, 4)
        state = ten.read(self.root / 'grouped/state/n4.json')
        self.assertEqual(list(state['cases']), ['n4-trident_hybrid'])
        path = self.root / 'n4/strong_scaling/trident_hybrid.tsv'
        saved = path.read_bytes()
        with self.fake_cases(), patch.dict(os.environ, self.group_env()), contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(grouped.run(self.root, 4))
        self.assertNotIn('n4-trident_hybrid', self.case_launches)
        self.assertEqual(path.read_bytes(), saved)

    def test_preliminary_source_and_existing_single_submission_block_groups(self):
        manifest = ten.read(self.root / 'manifest.json')
        manifest.update(phase='preliminary', source_results_sha256=recovery.result_snapshot(self.campaign, (4,)))
        ten.write(self.root / 'manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'preliminary'):
            grouped.prepare(self.root)
        self.assertFalse((self.root / 'grouped').exists())
        (self.root / 'manifest.json').write_bytes(self.parent_bytes)
        with contextlib.redirect_stdout(io.StringIO()):
            grouped.prepare(self.root)
        ten.write(self.root / 'submission/n1-trident_two_sided.json', dict(job_id='123'))
        with self.assertRaisesRegex(ValueError, 'already submitted'):
            grouped.submit(self.root)
        self.assertFalse(any(c[0] == 'sbatch' for c in self.calls))

    def test_manifest_script_tampering_and_wrong_allocation_are_rejected(self):
        self.prepare_submit()
        with patch.dict(os.environ, dict(SLURM_JOB_ID='999')), self.assertRaisesRegex(ValueError, 'does not match'):
            grouped.run(self.root, 4)
        path = self.root / 'grouped/metadata/jobs/n4.sbatch'
        path.write_text(path.read_text() + '# unexpected change\n')
        with self.assertRaisesRegex(ValueError, 'Snapshot modificato'):
            grouped.verify(self.root)

    def test_immediate_start_waits_for_durable_submission_receipt(self):
        self.prepare_submit()
        path = self.root / 'grouped/submission/n4.json'
        saved = ten.read(path)
        env = self.group_env()
        ten.write(path, dict(state='submitting'))
        with patch.object(grouped.time, 'sleep', side_effect=lambda _: ten.write(path, saved)) as sleep, \
                self.fake_cases(), patch.dict(os.environ, env), contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(grouped.run(self.root, 4))
        sleep.assert_called_once_with(0.5)

    def test_ambiguous_submission_is_reserved_and_cannot_be_duplicated(self):
        with contextlib.redirect_stdout(io.StringIO()):
            grouped.prepare(self.root)
        previous = grouped.subprocess.run
        def malformed(cmd, **kwargs):
            if cmd[:2] == ['sbatch', '--parsable']:
                return subprocess.CompletedProcess(cmd, 0, stdout='unknown response', stderr='')
            return previous(cmd, **kwargs)
        with patch.object(grouped.subprocess, 'run', side_effect=malformed), \
                self.assertRaisesRegex(ValueError, 'Ambiguous sbatch'):
            grouped.submit(self.root)
        with self.assertRaisesRegex(ValueError, 'Ambiguous group submission'):
            grouped.submit(self.root)
        with self.assertRaisesRegex(ValueError, 'Ambiguous prior submission'):
            recovery.submit(self.root, remaining=True)

    def test_additional_scope_is_twelve_cases_in_the_same_n100_allocation(self):
        self.prepare_additional()
        _, plan, _, jobs = grouped.verify(self.root)
        extra = plan['additional_cases']
        self.assertEqual(len(extra), 12)
        self.assertEqual({j['name'] for j in extra}, {'dielFilterV3real', 'cage15', 'archaea'})
        self.assertEqual({j['variant'] for j in extra}, set(ten.VARIANTS[4:]))
        self.assertTrue(all(j['nodes'] == 100 and j['action'] == 'permutation' for j in extra))
        group = next(g for g in plan['groups'] if g['nodes'] == 100)
        self.assertEqual(len(group['cases']), 22)
        self.assertEqual(len([c for c in self.calls if c[:2] == ['sbatch', '--parsable']]), 8)
        source_cfg = ten.read(self.campaign / 'metadata/n100.json')
        cfg = ten.read(self.root / 'grouped/metadata/cached-n100.json')
        self.assertEqual(cfg, dict(source_cfg, results_dir=str(self.root / 'additional-cached/n100')))
        self.assertNotEqual(cfg['build_dir'], str(self.root / 'metadata/bin'))
        for key in group['cases']:
            self.assertEqual(ten.read(self.root / 'submission' / (key + '.json'))['group'], 'n100')
        with contextlib.redirect_stdout(io.StringIO()):
            grouped.prepare(self.root, include_n100_b08=True)
            grouped.submit(self.root)
        self.assertEqual(len([c for c in self.calls if c[:2] == ['sbatch', '--parsable']]), 8)
        self.assertEqual((self.root / 'manifest.json').read_bytes(), self.parent_bytes)

    def test_additional_execution_keeps_original_binary_cache_and_ten_samples(self):
        self.prepare_additional()
        with self.fake_cases(), patch.dict(os.environ, self.n100_env()), contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(grouped.run(self.root, 100))
            self.assertEqual(len(self.case_launches), 22)
            before = self.log.read_bytes()
            self.assertTrue(grouped.run(self.root, 100))
            self.assertEqual(len(self.case_launches), 22)
        self.assertEqual(self.log.read_bytes(), before)
        source_cfg = ten.read(self.campaign / 'metadata/n100.json')
        commands = [json.loads(line) for line in self.log.read_text().splitlines()]
        extra = [c for c in commands if '--partial' in c and c[c.index('--experiment') + 1] == 'permutation']
        self.assertEqual(len(extra), 12)
        for cmd in extra:
            self.assertTrue(any(str(Path(source_cfg['build_dir']) / v) in cmd for v in ten.VARIANTS[4:]))
            self.assertEqual(cmd[cmd.index('--partial-cache') + 1], source_cfg['partial_cache'])
            self.assertEqual(cmd[cmd.index('--repeats') + 1], '10')
            self.assertNotIn('--full-cache', cmd)
        base, e = ten.runtime(self.campaign)
        for variant in ten.VARIANTS[4:]:
            rows = e.read_table(self.root / 'additional-cached/n100/permutation' / (variant + '.tsv'), e.FIELDS)
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(r['validation'] == 'PARTIAL_PASS' and r['repeats'] == '10' for r in rows))
        self.assertEqual(recovery.result_snapshot(self.campaign), {})
        self.assertEqual((self.campaign / 'manifest.json').read_bytes(), self.original_manifest)
        self.assertEqual((self.root / 'manifest.json').read_bytes(), self.parent_bytes)
        self.assertEqual({p: p.read_bytes() for p in self.probes}, self.probes)

    def test_already_accepted_original_case_is_excluded_from_additional_plan(self):
        base, e = ten.runtime(self.campaign)
        cfg = e.options(['permutation', '--config', str(self.campaign / 'metadata/n100.json'),
                         '--variants', 'trident_hybrid'])
        case = next(c for c in e.cases(cfg, 'permutation') if c[0] == 'cage15')
        with contextlib.redirect_stdout(io.StringIO()):
            ten.perform(base, e, cfg, 'permutation', [case])
        snapshot = recovery.result_snapshot(self.campaign)
        parent = ten.read(self.root / 'manifest.json')
        parent['source_results_sha256'] = snapshot
        ten.write(self.root / 'manifest.json', parent)
        self.prepare_additional()
        _, plan, _, _ = grouped.verify(self.root)
        self.assertEqual(len(plan['additional_cases']), 11)
        self.assertFalse(any(j['name'] == 'cage15' and j['variant'] == 'trident_hybrid'
                             for j in plan['additional_cases']))
        self.assertEqual(recovery.result_snapshot(self.campaign), snapshot)

    def test_additional_failure_never_becomes_accepted_and_preserves_previous_row(self):
        self.prepare_additional()
        _, plan, _, _ = grouped.verify(self.root)
        extra = [j for j in plan['additional_cases'] if j['variant'] == 'trident_get']
        with patch.dict(os.environ, self.n100_env()), contextlib.redirect_stdout(io.StringIO()):
            grouped.run_cached(self.root, extra[0]['key'])
            table = self.root / 'additional-cached/n100/permutation/trident_get.tsv'
            saved = table.read_bytes()
            with patch.dict(os.environ, MOCK_FAILURE='validation'), self.assertRaises(RuntimeError):
                grouped.run_cached(self.root, extra[1]['key'])
        self.assertEqual(table.read_bytes(), saved)
        self.assertFalse(grouped.complete(recovery, self.root, self.campaign, extra[1]))
        with patch.dict(os.environ, SLURM_JOB_ID='999'), self.assertRaisesRegex(ValueError, 'recorded grouped'):
            grouped.run_cached(self.root, extra[1]['key'])

    def test_frozen_additional_runner_works_in_an_isolated_process(self):
        self.prepare_additional()
        job = grouped.verify(self.root)[1]['additional_cases'][0]
        code = '''
import importlib.util
import pathlib
import subprocess
import sys
from unittest.mock import patch
root = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('driver', root / 'grouped/metadata/driver.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
real_run = subprocess.run
def scheduler(cmd, **kwargs):
    if cmd[0] == 'sacct':
        ids = cmd[cmd.index('-j') + 1].split(',')
        return subprocess.CompletedProcess(cmd, 0, stdout=''.join(i + '|COMPLETED|\\n' for i in ids))
    return real_run(cmd, **kwargs)
with patch.object(subprocess, 'run', side_effect=scheduler):
    driver.run_cached(root, sys.argv[2])
'''
        result = fixtures.REAL_RUN([sys.executable, '-c', code, str(self.root), job['key']],
                                  env=dict(os.environ, **self.n100_env()), cwd=self.root,
                                  capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(grouped.complete(recovery, self.root, self.campaign, job))

    def test_existing_group_plan_cannot_be_silently_extended(self):
        self.prepare_submit()
        with self.assertRaisesRegex(ValueError, 'excludes n100-b08'):
            grouped.prepare(self.root, include_n100_b08=True)
        self.assertEqual(len([c for c in self.calls if c[:2] == ['sbatch', '--parsable']]), 8)

    def test_additional_configuration_and_reference_guards(self):
        with self.assertRaisesRegex(ValueError, 'non preparato'):
            grouped.prepare(self.root, include_n100_b08=True)
        self.assertFalse((self.root / 'grouped').exists())
        self.prepare_additional()
        path = self.root / 'grouped/metadata/cached-n100.json'
        cfg = ten.read(path)
        cfg['repeats'] = 20
        ten.write(path, cfg)
        manifest = ten.read(self.root / 'grouped/manifest.json')
        manifest['sha256']['metadata/cached-n100.json'] = ten.digest(path)
        ten.write(self.root / 'grouped/manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'scientific configuration'):
            grouped.verify(self.root)


if __name__ == '__main__':
    unittest.main()
