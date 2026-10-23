#!/usr/bin/env python3
"""Portable campaign contracts using tiny inputs and fake executables/Slurm."""
import argparse
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
from lib import campaign as c


class CampaignTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='cached campaign ')
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.data = self.base / 'inputs'
        (self.data / 'metadata').mkdir(parents=True)
        source = self.data / 'cage8.mtx'
        source.write_text('%%MatrixMarket matrix coordinate real general\n2 2 3\n1 1 2\n1 2 1\n2 2 3\n')
        c.e.matrices.write_json(self.data / 'metadata/cage8.source.json', c.e.matrices.inspect_matrix(source))
        with contextlib.redirect_stdout(io.StringIO()):
            c.e.matrices.prepare({'id': 'cage8'}, self.data, 42)
        self.settings = c.read(REPO / 'scripts/campaign.json')
        self.settings['experiment'].update(nodes=[1], ranks_per_node=2, threads=1, cpus_per_rank=2,
            repeats=3, variants=['spgemm_two_sided', 'trident_get'], matrices=['cage8'],
            permutation_matrices=['cage8'], matrix_dir=str(self.data),
            mpi_launcher=sys.executable, mpi_args=[str(REPO / 'tests/fixtures/mock_experiment_mpi.py')])
        self.settings['cases_per_job'] = 2
        self.settings['slurm']['preamble'] = []
        self.path = self.base / 'settings.json'
        c.write(self.path, self.settings)
        self.launches = self.base / 'launches.jsonl'
        environment = patch.dict(os.environ, dict(MOCK_LAUNCH_LOG=str(self.launches), MOCK_NUMERIC_BOUND='1',
            SLURM_JOB_ID='123', SLURM_JOB_NUM_NODES='1', SLURM_NTASKS='2',
            SLURM_TASKS_PER_NODE='2', SLURM_CPUS_PER_TASK='2'))
        environment.start()
        self.addCleanup(environment.stop)
        # config() installs a campaign-specific diagnostic path in the shared runner.
        original_check_root = c.e.check_root
        self.addCleanup(setattr, c.e, 'check_root', original_check_root)

    def fake_build(self, source, destination, jobs, cmake_args):
        destination.mkdir(parents=True)
        for name in c.e.PROTOCOLS:
            (destination / name).write_text('fixture: no numerical computation')
        (destination / 'CMakeCache.txt').write_text('fixture build')
        for mode, suffix in [('full', '.sfull'), ('partial', '.spref')]:
            path = destination / ('prepare_' + mode + '_reference')
            path.write_text('#!' + sys.executable + '\n' +
                'import sys,hashlib\nfrom pathlib import Path\n'
                'args=sys.argv[1:]\n'
                "key=hashlib.sha256((args[1]+args[3]).encode()).hexdigest()\n" +
                'root=Path(args[args.index("--' + mode + '-cache")+1])\n' +
                'root.mkdir(parents=True,exist_ok=True)\n' +
                '(root/(key+"' + suffix + '")).write_bytes(b"fixture"*32)\n' +
                'print("validation_reference_key="+key)\n')
            path.chmod(0o755)

    def prepare(self, mode='full'):
        root = self.base / mode
        args = argparse.Namespace(config=self.path, matrix_dir=None, root=root, cache_dir=None,
                                  build_jobs=2, cmake_arg=[])
        with patch.object(c, 'build', self.fake_build), contextlib.redirect_stdout(io.StringIO()):
            c.prepare(mode, args)
        return root, c.read(root / 'manifest.json')

    def command(self, root, mode, action, *args, reference=False, success=True):
        env = dict(os.environ)
        if reference:
            env.update(SLURM_NTASKS='1', SLURM_TASKS_PER_NODE='1', SLURM_CPUS_PER_TASK='1')
        result = subprocess.run([sys.executable, str(c.driver(root, mode)), action,
            '--root', str(root), *args], env=env, capture_output=True, text=True)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def references(self, root, manifest):
        for ref in manifest['references']:
            self.command(root, manifest['mode'], 'reference', '--key', ref['key'], reference=True)

    def test_default_plan_preserves_inputs_resources_and_samples(self):
        for mode in ('full', 'partial'):
            value, cfg = c.settings(REPO / 'scripts/campaign.json', mode)
            jobs, references = c.plan(value, cfg)
            self.assertEqual((len(jobs), len(references)), (64, 24))
            self.assertEqual(sum(len(j['cases']) for j in jobs if j['nodes'] == 1), 184)
            self.assertEqual((cfg.repeats, cfg.trials, cfg.threads, cfg.ranks_per_node), (10, 1, 63, 2))
            self.assertNotIn(['reddit.mtx', 'reddit.mtx'], [r['pair'] for r in references])
            self.assertIn(['reddit.mtx', 'reddit_restriction.mtx'], [r['pair'] for r in references])

    def test_frozen_full_and_partial_lifecycle_reuses_rows_and_cache(self):
        for mode in ('full', 'partial'):
            with self.subTest(mode=mode):
                root, manifest = self.prepare(mode)
                self.command(root, mode, 'submit', success=False)
                self.references(root, manifest)
                receipt = root / 'reference-ready/ref-000.json'
                before = receipt.read_bytes()
                self.command(root, mode, 'reference', '--key', 'ref-000', reference=True)
                self.assertEqual(receipt.read_bytes(), before)
                for job in manifest['jobs']:
                    self.command(root, mode, 'run', '--key', job['key'])
                before = self.launches.read_bytes()
                for job in manifest['jobs']:
                    self.command(root, mode, 'run', '--key', job['key'])
                    self.assertFalse(c.missing(root, manifest, job))
                self.assertEqual(self.launches.read_bytes(), before)
                self.command(root, mode, 'export')
                for table in (root / 'n1').glob('*/*.tsv'):
                    for row in c.e.read_table(table, c.e.PHASE_FIELDS if table.parent.name == 'phase_analysis'
                                             else c.e.STRUCTURE_FIELDS if table.parent.name == 'matrix_structure' else c.e.FIELDS):
                        self.assertEqual(row['validation'], 'PASS' if mode == 'full' else 'PARTIAL_PASS')
                        self.assertEqual(len(json.loads(row['end_to_end_samples_seconds'])), 3)
                commands = [json.loads(line) for line in self.launches.read_text().splitlines()
                            if any(str(root) in token for token in json.loads(line))]
                self.assertEqual(len(commands), 14)  # 6 full pilots + 2 cached probes + 6 measurements
                for command in commands:
                    pilot = command[command.index('--experiment') + 1] == 'pilot'
                    self.assertEqual('--' + mode + '-cache' in command, not pilot)
                    self.assertEqual('--partial' in command, mode == 'partial' and not pilot)
                script = (root / 'metadata/jobs/n1-b001.sbatch').read_text()
                self.assertIn(str(c.driver(root, mode)), script)
                self.assertIn('--ntasks-per-node=2', script)
                self.assertLess(script.index('exec >'), script.index('collect_machine.sh'))

    def test_failed_case_does_not_discard_successes_and_resume_skips_them(self):
        root, manifest = self.prepare()
        self.references(root, manifest)
        real = c.e.run_experiment
        def fail_one(cfg, action, cases, headers):
            if cfg.results_dir == root / 'n1' and cfg.variants == ['spgemm_two_sided']:
                raise RuntimeError('deliberate failed main case')
            return real(cfg, action, cases, headers)
        with patch.object(c.e, 'run_experiment', fail_one), contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(c.run(root, manifest, 'n1-b001'))
        table = root / 'n1/strong_scaling/trident_get.tsv'
        before = table.read_bytes()
        self.assertEqual(len(c.missing(root, manifest, manifest['jobs'][0])), 1)
        self.command(root, 'full', 'run', '--key', 'n1-b001')
        self.assertEqual(table.read_bytes(), before)
        self.assertFalse(c.missing(root, manifest, manifest['jobs'][0]))

    def test_changed_snapshot_wrong_mode_and_allocation_are_rejected(self):
        root, manifest = self.prepare()
        self.command(root, 'partial', 'status', success=False)
        with patch.dict(os.environ, SLURM_NTASKS='9'):
            self.assertIn('task count', self.command(root, 'full', 'run', '--key', 'n1-b001', success=False).stderr)
        (root / 'metadata/bin/trident_get').write_text('changed')
        self.assertIn('Frozen campaign file changed', self.command(root, 'full', 'status', success=False).stderr)

    def test_submissions_are_chained_idempotent_and_retry_only_terminal_jobs(self):
        root, manifest = self.prepare()
        self.references(root, manifest)
        submitted = []
        def execute(command, **kwargs):
            if '--test-only' in command:
                return subprocess.CompletedProcess(command, 0)
            submitted.append(command)
            return subprocess.CompletedProcess(command, 0, str(1000 + len(submitted)) + '\n')
        with patch.object(c.subprocess, 'run', execute), contextlib.redirect_stdout(io.StringIO()):
            c.submit(root, manifest)
            self.assertEqual(len(submitted), 3)
            self.assertNotIn('--dependency=afterany:1001', submitted[0])
            self.assertIn('--dependency=afterany:1001', submitted[1])
            self.assertIn('--dependency=afterany:1002', submitted[2])
            with patch.object(c, 'states', return_value={'1001': 'FAILED', '1002': 'RUNNING', '1003': 'PENDING'}):
                c.submit(root, manifest)
                self.assertEqual(len(submitted), 3)
                c.submit(root, manifest, retry=True)
                self.assertEqual(len(submitted), 4)
                self.assertIn('--dependency=afterany:1003', submitted[-1])
        receipt = c.read(root / 'submission/n1-b001.json')
        self.assertEqual(receipt['history'], [dict(job_id='1001', state='FAILED')])

    def test_ambiguous_submission_and_changed_manifest_fail_closed(self):
        root, manifest = self.prepare()
        c.write(root / 'submission/ref-000.json', {'state': 'submitting'})
        with self.assertRaisesRegex(ValueError, 'Ambiguous submission'):
            c.submit(root, manifest, references=True)
        changed = copy.deepcopy(manifest)
        changed['experiment']['repeats'] += 1
        c.write(root / 'manifest.json', changed)
        self.assertIn('manifest changed', self.command(root, 'full', 'status', success=False).stderr)


if __name__ == '__main__':
    unittest.main()
