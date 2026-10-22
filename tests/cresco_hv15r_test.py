"""Offline recovery orchestration: no real sbatch, no scientific timing assertions."""
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
import cresco_cached_test as fixtures
import launch_cresco_ten as ten
import recover_cresco_hv15r as recovery

REAL_RUN = subprocess.run


class RecoveryTest(unittest.TestCase):
    def setUp(self):
        fixtures.CachedTest.setUp(self)
        self.campaign = self.root
        self.root = self.repo / 'results/hvfull'
        snapshot = self.campaign / 'metadata/source'
        common = snapshot / 'src/common/spgemm_common.cpp'
        common.write_text(recovery.original_common(common.read_text()), encoding='utf-8')
        for relative in ('common/spgemm_common.hpp', 'trident/common/benchmark.cpp',
                         'baselines/two_sided_spgemm.cpp', 'baselines/one_sided_get_spgemm.cpp',
                         'baselines/one_sided_put_spgemm.cpp'):
            path = snapshot / 'src' / relative
            path.write_text(recovery.original_metadata(path.read_text()), encoding='utf-8')
        cmake = snapshot / 'CMakeLists.txt'
        text = cmake.read_text()
        for addition in recovery.CMAKE_ADDITIONS:
            text = text.replace(addition, '')
        cmake.write_text(text, encoding='utf-8')
        manifest = ten.read(self.campaign / 'manifest.json')
        for node in ten.NODES:
            path = self.campaign / ('metadata/n%d.json' % node)
            cfg = ten.read(path)
            cfg.update(mpi_launcher=sys.executable,
                       mpi_args=[str(REPO / 'tests/fixtures/mock_experiment_mpi.py')])
            ten.write(path, cfg)
        for relative in recovery.CACHE_SOURCES:
            (snapshot / 'src' / relative).unlink()
        for name in list(manifest['sha256']):
            if not (self.campaign / name).exists():
                del manifest['sha256'][name]
                continue
            manifest['sha256'][name] = ten.digest(self.campaign / name)
        for i, job in enumerate(manifest['jobs']):
            ten.write(self.campaign / 'submission' / (job['key'] + '.json'), dict(job_id=str(99001 + i)))
        ten.write(self.campaign / 'manifest.json', manifest)
        self.original_manifest = (self.campaign / 'manifest.json').read_bytes()
        self.calls = []
        self.state = 'COMPLETED'
        numeric_metadata = patch.dict(os.environ, MOCK_NUMERIC_BOUND='1')
        numeric_metadata.start()
        self.addCleanup(numeric_metadata.stop)

        def scheduler(command, **kwargs):
            self.calls.append(command)
            if command[0] == 'sacct':
                output = ''.join('%d|%s|\n' % (99001 + i, self.state) for i in range(80))
                return subprocess.CompletedProcess(command, 0, stdout=output, stderr='')
            if Path(command[0]).name == 'prepare_full_reference':
                key = ('a' if 'permuted' not in command[2] else 'b') * 64
                path = self.root / 'references' / (key + '.sfull')
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(b'SPFR0001' + key.encode('ascii') + b'0' * 104)
                return subprocess.CompletedProcess(command, 0, stdout='validation_reference_key=' + key + '\n', stderr='')
            if command[0] in ('ctest', 'sbatch'):
                return subprocess.CompletedProcess(command, 0, stdout=str(70000 + len(self.calls)) + '\n', stderr='')
            return REAL_RUN(command, **kwargs)

        mocked = patch.object(recovery.subprocess, 'run', side_effect=scheduler)
        mocked.start()
        self.addCleanup(mocked.stop)

        def build(source, directory):
            directory.mkdir()
            for variant in (*recovery.VARIANTS, 'prepare_full_reference'):
                (directory / variant).write_text('numeric recovery binary')

        with patch.object(recovery.cached, 'build', side_effect=build), \
                patch.object(recovery, 'compiler_settings', return_value={'compiler': 'same'}), \
                contextlib.redirect_stdout(io.StringIO()):
            recovery.prepare(REPO, self.campaign, self.root)
        with contextlib.redirect_stdout(io.StringIO()):
            for action in recovery.ACTIONS:
                recovery.reference(self.root, action)

    def run_probe(self, action='strong_scaling'):
        with patch.dict(os.environ, dict(SLURM_JOB_NUM_NODES='4', SLURM_NTASKS='8',
                                         SLURM_TASKS_PER_NODE='2(x4)')), \
                contextlib.redirect_stdout(io.StringIO()):
            key = 'n4-trident_two_sided' if action == 'strong_scaling' else 'n4-permutation-trident_two_sided'
            recovery.run(self.root, key)

    def test_missing_corrupt_reference_and_generator_failure_block_recovery(self):
        receipt = self.root / 'reference-ready/strong_scaling.json'
        receipt.unlink()
        with self.assertRaisesRegex(ValueError, 'reference not ready'):
            recovery.submit(self.root)
        with self.assertRaisesRegex(ValueError, 'reference not ready'):
            self.run_probe()
        failed = subprocess.CompletedProcess([], 1, stdout='', stderr='disk full')
        original_run = recovery.subprocess.run
        def fail_generator(command, **kwargs):
            if Path(command[0]).name == 'prepare_full_reference':
                return failed
            return original_run(command, **kwargs)
        with patch.object(recovery.subprocess, 'run', side_effect=fail_generator), \
                contextlib.redirect_stderr(io.StringIO()) as error_log, self.assertRaises(subprocess.CalledProcessError):
            recovery.reference(self.root, 'strong_scaling')
        self.assertIn('disk full', error_log.getvalue())
        self.assertFalse(receipt.exists())
        with contextlib.redirect_stdout(io.StringIO()):
            recovery.reference(self.root, 'strong_scaling')
        record = ten.read(receipt)
        path = self.root / 'references' / (record['key'] + '.sfull')
        path.write_bytes(b'broken')
        with self.assertRaisesRegex(ValueError, 'missing/truncated'):
            recovery.submit(self.root)
        self.assertFalse(any(c[0] == 'sbatch' for c in self.calls))

    def test_two_reference_jobs_have_separate_budgets_and_no_duplicate_submission(self):
        for action in recovery.ACTIONS:
            (self.root / 'reference-ready' / (action + '.json')).unlink()
        with contextlib.redirect_stdout(io.StringIO()):
            recovery.submit_references(self.root)
            recovery.submit_references(self.root)
        commands = [c for c in self.calls if c[:2] == ['sbatch', '--parsable']]
        self.assertEqual(len(commands), 2)
        manifest, _ = recovery.verify(self.root)
        self.assertEqual(len(manifest['reference_jobs']), 2)
        for ref in manifest['reference_jobs']:
            script = (self.root / ref['script']).read_text()
            self.assertIn('--time=24:00:00', script)
            self.assertIn('--ntasks=1', script)
            self.assertNotIn('mpirun', script)

    def test_scope_config_and_unchanged_source(self):
        manifest, source = recovery.verify(self.root)
        self.assertEqual(len(manifest['jobs']), 80)
        self.assertEqual(len({j['key'] for j in manifest['jobs']}), 80)
        self.assertEqual(manifest['probes'], ['n4-trident_two_sided', 'n4-permutation-trident_two_sided'])
        self.assertEqual((source / 'manifest.json').read_bytes(), self.original_manifest)
        self.assertFalse(any(command[0] == 'sbatch' for command in self.calls))
        for job in manifest['jobs']:
            matrix = 'HV15R.mtx' if job['action'] == 'strong_scaling' else 'permuted/HV15R_permuted_s42.mtx'
            self.assertEqual(job['pair'], [matrix, matrix])
            cfg = ten.read(self.root / ('metadata/n%d.json' % job['nodes']))
            self.assertEqual((cfg['repeats'], cfg['threads'], cfg['ranks_per_node'], cfg['cpus_per_rank']),
                             (10, 63, 2, 64))
            self.assertIs(cfg['validate'], True)
            self.assertIsNone(cfg['partial_cache'])
            self.assertEqual(cfg['full_cache'], str(self.root / 'references'))
            self.assertEqual(cfg['permutation_matrices'], ['HV15R'])
            script = (self.root / job['script']).read_text()
            for flag in ('--time=24:00:00', '--mem=450000M', '--exclusive', '--cpus-per-task=64'):
                self.assertIn(flag, script)

    def test_passing_probe_preserves_samples_and_is_idempotent(self):
        self.run_probe()
        before = self.log.read_text().splitlines()
        self.run_probe()
        self.assertEqual(before, self.log.read_text().splitlines())
        commands = [json.loads(line) for line in before]
        self.assertEqual(len(commands), 4)  # Three cage8 pilot products + one HV15R square.
        self.assertTrue(all('--partial' not in c and '--no-validate' not in c for c in commands))
        self.assertEqual(commands[-1][commands[-1].index('--repeats') + 1], '10')
        self.assertTrue(all('--full-cache' not in c for c in commands[:3]))
        self.assertIn('--full-cache', commands[-1])
        self.assertFalse((self.campaign / 'n4/strong_scaling/trident_two_sided.tsv').exists())
        self.assertEqual((self.campaign / 'manifest.json').read_bytes(), self.original_manifest)
        _, e, cfg = recovery.runtime(self.root, self.campaign, 4, 'trident_two_sided')
        rows = e.read_table(self.root / 'n4/strong_scaling/trident_two_sided.tsv', e.FIELDS)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['validation'], 'PASS')
        self.assertEqual(rows[0]['validation_criterion'], 'componentwise_dot_bound_v1')
        self.assertGreater(float(rows[0]['max_abs_error']), 1e-10)
        self.assertEqual(len(json.loads(rows[0]['end_to_end_samples_seconds'])), 10)

    def test_failed_numeric_probe_cannot_enable_other_jobs(self):
        with patch.dict(os.environ, {'MOCK_FAILURE': 'validation'}), self.assertRaises(RuntimeError):
            self.run_probe()
        with self.assertRaisesRegex(ValueError, 'probe has not passed'):
            recovery.submit(self.root, remaining=True)
        self.assertFalse(any(command[0] == 'sbatch' for command in self.calls))

    def test_legacy_numeric_row_cannot_masquerade_as_new_criterion(self):
        with patch.dict(os.environ, MOCK_NUMERIC_BOUND=''), self.assertRaises(RuntimeError):
            self.run_probe()
        with self.assertRaisesRegex(ValueError, 'probe has not passed'):
            recovery.submit(self.root, remaining=True)
        self.assertFalse(any(command[0] == 'sbatch' for command in self.calls))

    def test_permuted_probe_is_required_and_preserves_its_own_samples(self):
        self.run_probe()
        with self.assertRaisesRegex(ValueError, 'n4-permutation-trident_two_sided'):
            recovery.submit(self.root, remaining=True)
        with patch.dict(os.environ, {'MOCK_FAILURE': 'validation'}), self.assertRaises(RuntimeError):
            self.run_probe('permutation')
        with self.assertRaisesRegex(ValueError, 'n4-permutation-trident_two_sided'):
            recovery.submit(self.root, remaining=True)
        self.run_probe('permutation')
        before = self.log.read_text().splitlines()
        self.run_probe('permutation')
        self.assertEqual(before, self.log.read_text().splitlines())
        command = json.loads(before[-1])
        self.assertEqual(command[command.index('--experiment') + 1], 'permutation')
        self.assertEqual(Path(command[command.index('--matrix-a') + 1]).name, 'HV15R_permuted_s42.mtx')
        _, e, _ = recovery.runtime(self.root, self.campaign, 4, 'trident_two_sided', 'permutation')
        rows = e.read_table(self.root / 'n4/permutation/trident_two_sided.tsv', e.FIELDS)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['validation'], 'PASS')
        self.assertEqual(len(json.loads(rows[0]['end_to_end_samples_seconds'])), 10)
        self.assertFalse((self.campaign / 'n4/permutation/trident_two_sided.tsv').exists())

    def test_submission_gate_dependencies_and_duplicate_protection(self):
        with contextlib.redirect_stdout(io.StringIO()):
            recovery.submit(self.root)
            recovery.submit(self.root)
        submitted = [c for c in self.calls if c[:2] == ['sbatch', '--parsable']]
        self.assertEqual(len(submitted), 2)
        with self.assertRaisesRegex(ValueError, 'probe has not passed'):
            recovery.submit(self.root, remaining=True)
        self.run_probe()
        self.run_probe('permutation')
        with contextlib.redirect_stdout(io.StringIO()):
            recovery.submit(self.root, remaining=True)
            recovery.submit(self.root, remaining=True)
        submitted = [c for c in self.calls if c[:2] == ['sbatch', '--parsable']]
        self.assertEqual(len(submitted), 80)
        self.assertEqual(sum(any(x.startswith('--dependency=afterany:') for x in c) for c in submitted), 71)

    def test_active_source_and_uncertain_submission_are_blocked(self):
        self.state = 'RUNNING'
        with self.assertRaisesRegex(ValueError, 'still active'):
            recovery.submit(self.root)
        self.state = 'COMPLETED'
        ten.write(self.root / 'submission/n4-trident_two_sided.json', {'state': 'submitting'})
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            recovery.submit(self.root)
        self.assertFalse(any(command[0] == 'sbatch' for command in self.calls))

    def test_new_pass_in_source_is_excluded_from_plan_and_detected_after_freeze(self):
        base, e = ten.runtime(self.campaign)
        cfg = e.options(['strong_scaling', '--config', str(self.campaign / 'metadata/n1.json'),
                         '--variants', 'trident_get'])
        for i, action in enumerate(recovery.ACTIONS):
            case = next(c for c in e.cases(cfg, action) if c[0] == 'HV15R')
            with contextlib.redirect_stdout(io.StringIO()):
                headers = e.input_preflight(cfg, [(action, [case])])
                e.run_experiment(cfg, action, [case], headers)
            jobs = recovery.missing_jobs(self.campaign)
            self.assertEqual(len(jobs), 79 - i)
            self.assertNotIn('n1-trident_get', [j['key'] for j in jobs])
            if action == 'strong_scaling':
                self.assertIn('n1-permutation-trident_get', [j['key'] for j in jobs])
            else:
                self.assertNotIn('n1-permutation-trident_get', [j['key'] for j in jobs])
        with self.assertRaisesRegex(ValueError, 'result tables changed'):
            recovery.verify(self.root)


if __name__ == '__main__':
    unittest.main()
