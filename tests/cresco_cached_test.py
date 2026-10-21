"""Offline frozen-campaign integration. No scheduler commands or real products."""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
sys.path.insert(0, str(REPO / 'tests'))
import cresco_ten_test as fixtures
import launch_cresco_ten as ten
import prepare_cresco_cached as cached


class CachedTest(unittest.TestCase):
    def setUp(self):
        fixtures.TenTest.setUp(self)
        self.source = self.root
        self.root = self.repo / 'results/cached10'
        for name in ('experiments', 'allocated_experiment'):
            sys.modules.pop(name, None)
        original = cached.snapshot

        def fake_build(source, directory):
            directory.mkdir()
            for name in list(ten.VARIANTS) + ['CMakeCache.txt', 'prepare_partial_reference']:
                (directory / name).write_text('new binary ' + name)

        before = ten.digest(self.source / 'manifest.json')
        with patch.object(cached, 'snapshot', side_effect=lambda _, dest: original(REPO, dest)), \
                patch.object(cached, 'build', side_effect=fake_build), \
                patch.object(cached.subprocess, 'run') as calls, contextlib.redirect_stdout(io.StringIO()):
            cached.prepare(self.repo, self.source, self.root)
        self.assertEqual(calls.call_count, 0)
        self.assertEqual(ten.digest(self.source / 'manifest.json'), before)

    def test_new_snapshot_coverage_and_reference_jobs(self):
        manifest = ten.verify(self.root)
        self.assertEqual(len(manifest['jobs']), 80)
        self.assertEqual(len(manifest['reference_pairs']), 24)
        self.assertEqual(sum(p['name'] == 'cage8' for p in manifest['reference_pairs']), 1)
        for p in manifest['reference_pairs']:
            script = (self.root / p['script']).read_text()
            for flag in ('--time=24:00:00', '--nodes=1', '--ntasks=1', 'OMP_NUM_THREADS=1'):
                self.assertIn(flag, script)
            self.assertNotIn('mpirun', script)
        for variant in ten.VARIANTS:
            self.assertNotEqual(ten.digest(self.root / 'metadata/bin' / variant),
                                ten.digest(self.source / 'metadata/bin' / variant))
        _, e = ten.runtime(self.root)
        for nodes in ten.NODES:
            cfg = e.options(['all', '--config', str(self.root / ('metadata/n%d.json' % nodes))])
            self.assertEqual((cfg.threads, cfg.ranks_per_node, cfg.cpus_per_rank, cfg.warmup, cfg.repeats),
                             (63, 2, 64, 0, 10))
            case = e.cases(cfg, 'pilot')[0]
            self.assertNotIn('--partial-cache', e.command(cfg, 'pilot', case, ten.VARIANTS[0], nodes, Path('x.tsv')))
            command = e.command(cfg, 'strong_scaling', case, ten.VARIANTS[0], nodes, Path('x.tsv'))
            self.assertIn('--partial-cache', command)
            cfg.partial_cache = None
            self.assertNotIn('--partial-cache', e.command(cfg, 'strong_scaling', case, ten.VARIANTS[0], nodes, Path('x.tsv')))

    def test_missing_references_block_submission_and_run(self):
        with patch.object(ten.subprocess, 'run') as calls:
            with self.assertRaisesRegex(ValueError, 'non preparato'):
                ten.submit(self.root)
            self.assertEqual(calls.call_count, 0)
            with self.assertRaisesRegex(ValueError, 'non preparato'):
                ten.run(self.root, 1, 1)
            self.assertEqual(calls.call_count, 0)

    def test_reference_checks_inputs_and_publishes_only_successful_receipts(self):
        manifest = ten.verify(self.root)
        key = 'a' * 64
        output = 'validation_reference_key=' + key + '\n'
        command = subprocess.CompletedProcess([], 1, stdout=output, stderr='failure')
        with patch.object(cached.subprocess, 'run', return_value=command), contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(subprocess.CalledProcessError):
            cached.reference(self.root, 0)
        self.assertFalse((self.root / 'reference-ready/00.json').exists())
        command.returncode = 0
        command.stderr = ''
        cache = Path(manifest['partial_cache'])
        cache.mkdir()
        (cache / (key + '.spref')).write_bytes(b'fixture')
        with patch.object(cached.subprocess, 'run', return_value=command) as calls, contextlib.redirect_stdout(io.StringIO()):
            for pair in manifest['reference_pairs']:
                cached.reference(self.root, pair['index'])
            self.assertEqual(calls.call_count, 24)
        ten.check_references_ready(self.root, manifest)
        # Exercise the frozen cached launcher through a complete block, including Full pilots.
        fixtures.TenTest.mock_mpi(self)
        with contextlib.redirect_stdout(io.StringIO()):
            ten.run(self.root, 1, 1)
        launches = [json.loads(line) for line in self.log.read_text().splitlines()]
        self.assertEqual(len(launches), 24 + 8 + 12)
        self.assertTrue(all('--partial-cache' not in command for command in launches[:24]))
        self.assertTrue(all('--partial-cache' in command for command in launches[24:]))
        self.assertTrue(ten.read(self.root / 'state/n1-b01.finished.json')['complete'])
        receipt = self.root / 'reference-ready/00.json'
        record = ten.read(receipt)
        record['generator_sha256'] = 'wrong'
        ten.write(receipt, record)
        with self.assertRaisesRegex(ValueError, 'incompatibile'):
            ten.check_references_ready(self.root, manifest)
        a = Path(manifest['reference_pairs'][0]['matrix_a'])
        a.write_text(a.read_text() + '% changed input\n')
        with patch.object(cached.subprocess, 'run') as calls, contextlib.redirect_stdout(io.StringIO()), \
                self.assertRaisesRegex(ValueError, 'changed'):
            cached.reference(self.root, 0)
        self.assertEqual(calls.call_count, 0)

    def test_preparation_requires_compute_allocation_and_new_root(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, 'compute'):
            cached.prepare(self.repo, self.source, self.root)
        with self.assertRaises(FileExistsError):
            cached.prepare(self.repo, self.source, self.root)


if __name__ == '__main__':
    unittest.main()
