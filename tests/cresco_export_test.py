"""Export accounting, exact coverage and archive contents, without real cluster access."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
sys.path.insert(0, str(REPO / 'tests'))
import export_cresco_results as exporter


class FrozenTenFixture:
    """The known historical 23-product plan, independent of current launchers."""
    NODES = (1, 4, 9, 16, 36, 64, 100, 121)
    MAIN = ('strong_scaling', 'permutation', 'rectangular')
    VARIANTS = ('spgemm_two_sided', 'spgemm_one_sided_get', 'spgemm_one_sided_put',
                'trident_two_sided', 'trident_hybrid', 'trident_get', 'trident_get_pipeline', 'trident_put')
    GROUPS = (
        ('strong_scaling', ('mouse_gene', 'dielFilterV3real', 'isolates_subgraph4')),
        ('strong_scaling', ('cage15', 'archaea', 'isolates_subgraph5')),
        ('strong_scaling', ('HV15R', 'eukarya', 'uniparc')),
        ('permutation', ('HV15R', 'dielFilterV3real', 'cage15', 'archaea')),
        ('rectangular', ('mouse_gene', 'cage15', 'HV15R', 'dielFilterV3real', 'archaea',
                         'eukarya', 'isolates_subgraph4', 'isolates_subgraph5', 'uniparc', 'reddit')))

    @classmethod
    def plan(cls):
        return [dict(nodes=n, block=b, action=cls.GROUPS[(b - 1) // 2][0],
                     matrices=cls.GROUPS[(b - 1) // 2][1],
                     variants=cls.VARIANTS[:4] if b % 2 else cls.VARIANTS[4:])
                for n in cls.NODES for b in range(1, 11)]

    @staticmethod
    def read(path):
        return json.loads(path.read_text())

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    @staticmethod
    def runtime(root):
        raise AssertionError('Test must supply its frozen runtime')


ten = FrozenTenFixture


class ExportTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.workspace = Path(temp.name)
        self.source, self.root = self.workspace / 'original', self.workspace / 'recovery'
        ten.write(self.source / 'manifest.json', dict(jobs=ten.plan()))
        self.groups = dict(groups=[dict(key='n100', nodes=100)])
        self.original, self.jobs = {}, {}
        for nodes in ten.NODES:
            values = set()
            for b in (b for b in ten.plan() if b['nodes'] == nodes):
                for name in b['matrices']:
                    a, c = self.pair(b['action'], name)
                    for variant in b['variants']:
                        key = (b['action'], variant, a, c)
                        hv = nodes != 1 and name == 'HV15R' and b['action'] != 'rectangular' and variant.startswith('trident_')
                        extra = nodes == 100 and b['block'] == 8 and name != 'HV15R'
                        if hv or extra:
                            job_key = 'n%d-%s-%s-%s' % (nodes, b['action'], name, variant)
                            self.jobs[job_key] = dict(key=job_key, nodes=nodes, action=b['action'], variant=variant,
                                                     pair=[a, c], kind='original-partial' if extra else 'numeric')
                        else:
                            values.add(key)
            self.original[nodes] = values
        self.failed = set()
        engine = SimpleNamespace(
            options=lambda args: SimpleNamespace(nodes=[int(Path(args[-1]).stem[1:])], matrix_dir=Path('/matrices')),
            cases=lambda cfg, action: [(name, None, *(cfg.matrix_dir / p for p in self.pair(action, name)))
                                      for a, names in ten.GROUPS if a == action for name in names])
        base = SimpleNamespace(existing=lambda e, cfg, action: {k[1:] for k in self.original[cfg.nodes[0]] if k[0] == action})
        self.runner = SimpleNamespace(ten=ten, TERMINAL={'COMPLETED', 'FAILED', 'TIMEOUT'})
        self.driver = SimpleNamespace(verify=lambda root: (self.runner, self.groups, self.source, self.jobs),
                                      complete=lambda r, root, source, job: job['key'] not in self.failed)
        for mock in (patch.object(exporter, 'load_driver', return_value=self.driver),
                     patch.object(ten, 'runtime', return_value=(base, engine))):
            mock.start()
            self.addCleanup(mock.stop)

    @staticmethod
    def pair(action, name):
        if action == 'permutation':
            return ('permuted/%s_permuted_s42.mtx' % name,) * 2
        return name + '.mtx', name + ('_restriction.mtx' if action == 'rectangular' else '.mtx')

    def audit(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return exporter.audit(self.root)[0]

    def test_union_has_1472_observations_and_preserves_three_origins(self):
        report = self.audit()
        self.assertTrue(report['complete'])
        self.assertEqual(report['accepted_observations'], 1472)
        origins = [r['source'] for n in report['nodes'].values() for r in n['observations']]
        self.assertEqual(origins.count('partial10-cached'), 1390)
        self.assertEqual(origins.count('hv15r-roundoff-cached-v4'), 70)
        self.assertEqual(origins.count('additional-cached'), 12)
        self.assertTrue(all(n['accepted'] == n['expected'] == 184 for n in report['nodes'].values()))

    def test_missing_duplicate_and_unexpected_cases_cannot_claim_completion(self):
        job = next(iter(self.jobs.values()))
        self.failed.add(job['key'])
        report = self.audit()
        self.assertFalse(report['complete'])
        self.assertEqual(report['accepted_observations'], 1471)
        self.failed.clear()
        self.original[job['nodes']].add((job['action'], job['variant'], *job['pair']))
        report = self.audit()
        self.assertFalse(report['complete'])
        self.assertEqual(report['accepted_observations'], 1472)
        self.assertEqual(len(report['nodes'][str(job['nodes'])]['duplicates']), 1)
        self.original[1].add(('strong_scaling', 'trident_get', 'reddit.mtx', 'reddit.mtx'))
        self.assertEqual(len(self.audit()['nodes']['1']['unexpected']), 1)

    def test_active_unknown_and_ambiguous_recovery_receipts_block_audit(self):
        ten.write(self.root / 'grouped/submission/n100.json', dict(job_id='123'))
        for state in ('RUNNING', ''):
            reply = subprocess.CompletedProcess([], 0, stdout='123|RUNNING|\n' if state else '', stderr='')
            with patch.object(exporter.subprocess, 'run', return_value=reply), self.assertRaisesRegex(ValueError, 'still active'):
                self.audit()
        reply.stdout = '123|COMPLETED|\n'
        with patch.object(exporter.subprocess, 'run', return_value=reply):
            self.assertEqual(self.audit()['recovery_accounting'], {'123': 'COMPLETED'})
        ten.write(self.root / 'grouped/submission/n100.json', dict(state='submitting'))
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            self.audit()

    def test_archive_contains_raw_data_provenance_and_audit_without_heavy_caches(self):
        included = ['n100/permutation/trident_get.tsv', 'metadata/source/tests/data/tiny.mtx',
                    'metadata/n100.json', 'reference-ready/00.json', 'logs/n100/job/launch.log',
                    'checks/n100/tmp/failed/launch.log', 'additional-cached/n100/permutation/trident_get.tsv']
        excluded = ['metadata/build/large.bin', 'metadata/bin/trident_get', 'references/large.spref',
                    'references/large.sfull', 'metadata/source/scripts/__pycache__/test.pyc', '.campaign.lock']
        for directory in (self.root, self.source):
            for name in included + excluded:
                path = directory / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(name.encode())
        original_bytes = (self.source / included[0]).read_bytes()
        output = self.workspace / 'download.tar.gz'
        report = dict(complete=False)
        with contextlib.redirect_stdout(io.StringIO()):
            exporter.package(self.root, self.source, report, output)
        self.assertEqual((self.source / included[0]).read_bytes(), original_bytes)
        self.assertEqual(output.with_name(output.name + '.sha256').read_text().split()[0], exporter.digest(output))
        with tarfile.open(output, 'r:gz') as archive:
            names = set(archive.getnames())
            self.assertTrue(all('partial10-cached/' + name in names for name in included))
            self.assertFalse(any('partial10-cached/' + name in names for name in excluded))
            self.assertIn('hv15r-roundoff-cached-v4/additional-cached/n100/permutation/trident_get.tsv', names)
            self.assertFalse(json.load(archive.extractfile('export-audit.json'))['complete'])
            self.assertTrue(all(not m.issym() and not m.islnk() and not Path(m.name).is_absolute()
                                and '..' not in Path(m.name).parts for m in archive.getmembers()))
        with self.assertRaisesRegex(ValueError, 'already exists'):
            exporter.package(self.root, self.source, report, output)
        with self.assertRaisesRegex(ValueError, 'outside'):
            exporter.package(self.root, self.source, report, self.root / 'nested.tar.gz')


class FrozenImportTest(unittest.TestCase):
    def test_loader_uses_deployed_driver_without_checkout_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'grouped/metadata/driver.py'
            path.parent.mkdir(parents=True)
            path.write_text('identity = "frozen-campaign-driver"\n')
            module = exporter.load_driver(root)
            self.assertEqual(module.identity, 'frozen-campaign-driver')
            self.assertEqual(Path(module.__file__), path)


if __name__ == '__main__':
    unittest.main()
