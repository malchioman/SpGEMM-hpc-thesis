#!/usr/bin/env python3
"""Real MPI comparisons and POSIX process-lifetime cache tests, with tiny inputs."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import unittest

parser = argparse.ArgumentParser()
parser.add_argument('--build-dir', type=Path, required=True)
parser.add_argument('--mpi-launcher', default='mpirun')
args, remaining = parser.parse_known_args()
BUILD = args.build_dir.resolve()
DATA = Path(__file__).resolve().parent / 'data'
VARIANTS = ('spgemm_two_sided', 'spgemm_one_sided_get', 'spgemm_one_sided_put',
            'trident_two_sided', 'trident_hybrid', 'trident_get', 'trident_get_pipeline', 'trident_put')


class CacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='partial-reference-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cache = self.root / 'cache'

    def prepare_command(self, a, b):
        return [str(BUILD / 'prepare_partial_reference'), '--matrix-a', str(a), '--matrix-b', str(b),
                '--partial', '--partial-cache', str(self.cache)]

    def prepare(self, a, b, expected=0):
        run = subprocess.run(self.prepare_command(a, b), capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, expected, run.stdout + run.stderr)
        return run

    def test_all_variants_equivalent_and_measurements_unchanged(self):
        for case, a, b, expected in (
                ('square', 'tiny_symmetric.mtx', 'tiny_symmetric.mtx', 0),
                ('rectangle', 'rectangular_a.mtx', 'rectangular_b.mtx', 0),
                ('empty', 'empty.mtx', 'empty.mtx', 0),
                ('cancellation', 'cancellation.mtx', 'cancellation.mtx', 0),
                ('overflow', 'overflow_product.mtx', 'overflow_product.mtx', 1)):
            self.prepare(DATA / a, DATA / b, expected)
            for variant in VARIANTS:
                with self.subTest(case=case, variant=variant):
                    rows = []
                    for cached in (False, True):
                        output = self.root / ('%s-%s-%s.tsv' % (case, variant, cached))
                        command = [args.mpi_launcher, '-np', '2', '--oversubscribe', '--bind-to', 'none',
                                   str(BUILD / variant), '--matrix-a', str(DATA / a), '--matrix-b', str(DATA / b),
                                   '--partial', '--threads', '1', '--warmup', '0', '--repeats', '10',
                                   '--trials', '1', '--results', str(output)]
                        if cached:
                            command += ['--partial-cache', str(self.cache)]
                        start = time.monotonic()
                        run = subprocess.run(command, capture_output=True, text=True, timeout=25)
                        wall = time.monotonic() - start
                        self.assertEqual(run.returncode, expected, run.stdout + run.stderr)
                        self.assertIn('validation_reference_source=' + ('reused' if cached else 'regenerated'), run.stdout)
                        fields = dict(re.findall(r'(validation_\w+)=([^\s]+)', run.stdout))
                        if cached:
                            self.assertEqual(float(fields['validation_reference_build_seconds']), 0)
                        else:
                            self.assertEqual(float(fields['validation_cache_read_seconds']), 0)
                        with output.open() as stream:
                            row = next(csv.DictReader(stream, delimiter='\t'))
                        self.assertEqual(row['validation'], 'PARTIAL_FAIL' if expected else 'PARTIAL_PASS')
                        self.assertEqual(len(json.loads(row['compute_samples_seconds'])), 10)
                        rows.append(row)
                        if case == 'rectangle':
                            print(json.dumps(dict(variant=variant, cached=cached, wall_seconds=wall,
                                                  diagnostics=fields)), flush=True)
                    self.assertEqual(rows[0].keys(), rows[1].keys())
                    for key in ('a_rows', 'a_cols', 'b_rows', 'b_cols', 'c_nnz', 'validation',
                                'result_finite', 'benchmark_protocol', 'csr_offset_bits'):
                        self.assertEqual(rows[0][key], rows[1][key])

    def test_checksum_corruption_incompatibility_missing_and_changed_input(self):
        a = self.root / 'a.mtx'
        a.write_bytes((DATA / 'tiny_symmetric.mtx').read_bytes())
        self.prepare(a, a)
        path = next(self.cache.glob('*.spref'))
        original = path.read_bytes()
        self.assertEqual(hashlib.sha256(original[:-64]).hexdigest().encode(), original[-64:])
        for content in (original[:-1], original + b'x', b'SPPR0002' + original[8:],
                        original[:96] + bytes([original[96] ^ 1]) + original[97:]):
            path.write_bytes(content)
            failed = self.prepare(a, a, 1)
            self.assertNotIn('validation_reference_source=generated', failed.stdout)
        path.write_bytes(original)
        # Same path, valid different matrix, no mtime/path-based cache identity.
        a.write_text('%%MatrixMarket matrix coordinate real general\n1 1 1\n1 1 9\n')
        command = [args.mpi_launcher, '-np', '1', '--oversubscribe', str(BUILD / VARIANTS[0]),
                   '--matrix', str(a), '--partial', '--partial-cache', str(self.cache),
                   '--threads', '1', '--warmup', '0', '--repeats', '1', '--trials', '1',
                   '--results', str(self.root / 'missing.tsv')]
        run = subprocess.run(command, capture_output=True, text=True, timeout=20)
        self.assertNotEqual(run.returncode, 0)
        self.assertNotIn('PARTIAL_PASS', run.stdout)
        self.assertFalse((self.root / 'missing.tsv').exists())

    @unittest.skipUnless(os.name == 'posix', 'POSIX flock/process-death regression')
    def test_concurrent_generators_and_killed_owner(self):
        import fcntl
        a = DATA / 'cancellation.mtx'
        command = self.prepare_command(a, a)
        processes = [subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                     for _ in range(4)]
        outputs = [p.communicate(timeout=30) for p in processes]
        self.assertEqual(sum('validation_reference_source=generated' in out for out, _ in outputs), 1)
        for p, (out, err) in zip(processes, outputs):
            self.assertTrue(p.returncode == 0 or 'generator busy' in err, out + err)
        path = next(self.cache.glob('*.spref'))
        expected = path.read_bytes()
        # Readers need no lock and still see a complete old entry during a locked preparation.
        with open(str(path) + '.lock', 'a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertIn('generator busy', self.prepare(a, a, 1).stderr)
            self.assertEqual(path.read_bytes(), expected)
        # Test-owned entry only. A torn temporary file is never treated as a reference.
        path.unlink()
        Path(str(path) + '.tmp').write_bytes(b'interrupted writer')
        self.prepare(a, a)
        self.assertEqual(path.read_bytes(), expected)
        self.assertFalse(Path(str(path) + '.tmp').exists())

        # Kill the actual generator while it owns the kernel lock in numeric construction.
        dense = self.root / 'dense.mtx'
        n = 600
        with dense.open('w') as stream:
            stream.write('%%MatrixMarket matrix coordinate real general\n' + '%d %d %d\n' % (n, n, n*n))
            for row in range(n):
                stream.writelines('%d %d 1\n' % (row+1, col+1) for col in range(n))
        prior = set(self.cache.glob('*.lock'))
        process = subprocess.Popen(self.prepare_command(dense, dense), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic() + 15
            locked = False
            while time.monotonic() < deadline and process.poll() is None:
                for lockpath in set(self.cache.glob('*.lock')) - prior:
                    with lockpath.open('a+b') as lock:
                        try:
                            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            locked = True
                if locked:
                    break
                time.sleep(.002)
            self.assertTrue(locked, 'did not observe active generator')
            process.kill()
            process.communicate(timeout=10)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()
        run = self.prepare(dense, dense)
        self.assertIn('validation_reference_source=generated', run.stdout)
        self.assertIn('validation_reference_source=reused', self.prepare(dense, dense).stdout)


if __name__ == '__main__':
    unittest.main(argv=[__file__] + remaining)
