#!/usr/bin/env python3
"""Real HV15R coordinate witnesses; diagnostic MPI runs, never performance data."""
import argparse
import csv
from fractions import Fraction
import json
from pathlib import Path
import re
import subprocess
import tempfile
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/lib'))
import experiments

parser = argparse.ArgumentParser()
parser.add_argument('--build-dir', type=Path, required=True)
parser.add_argument('--mpi-launcher', default='mpirun')
args = parser.parse_args()
data = Path(__file__).resolve().parent / 'data/hv15r_roundoff'
variants = ('trident_two_sided', 'trident_hybrid', 'trident_get', 'trident_get_pipeline', 'trident_put')
zero_name = 'zero-cancellation-witness'
absolute_name = 'absolute-threshold-witness'

# Independent arithmetic oracle: exact products of the binary64 operands, not
# the validator's formula. Neither sequential double result is mathematical truth.
for name in (zero_name, absolute_name):
    witness = json.loads((data / (name + '.json')).read_text())
    exact = sum((Fraction(t['a']) * Fraction(t['b']) for t in witness['terms_in_serial_order']), Fraction())
    assert float(exact) == witness['exact_dot_as_float']
    for label in ('serial', 'trident'):
        assert float(abs(Fraction(witness[label]) - exact)) == witness[label + '_error_vs_exact']
    if name == absolute_name:
        assert witness['trident_error_vs_exact'] < witness['serial_error_vs_exact']

tests = [(zero_name, 1, 'trident_two_sided', True),
         (zero_name, 3, 'trident_two_sided', True)]
tests += [(zero_name, 3, v, False) for v in variants]
tests += [(absolute_name, 4, v, False) for v in variants]
tests += [(absolute_name, 4, v, False) for v in
          ('spgemm_two_sided', 'spgemm_one_sided_get', 'spgemm_one_sided_put')]
with tempfile.TemporaryDirectory(prefix='hv15r-roundoff-') as tmp:
    for index, (name, q, variant, partial) in enumerate(tests):
        table = Path(tmp) / (str(index) + '.tsv')
        command = [args.mpi_launcher, '-np', str(2*q*q), '--oversubscribe', '--bind-to', 'none',
                   str(args.build_dir / variant), '--matrix-a', str(data / (name + '-A.mtx')),
                   '--matrix-b', str(data / (name + '-B.mtx')), '--threads', '1',
                   '--warmup', '0', '--repeats', '1', '--trials', '1', '--results', str(table)]
        if variant.startswith('trident_'):
            command += ['--logical-node-size', '2']
        if partial:
            command += ['--partial']
        run = subprocess.run(command, text=True, capture_output=True, timeout=60)
        expected_fail = partial and q == 3
        assert run.returncode == int(expected_fail), run.stdout + run.stderr
        with table.open() as stream:
            row = next(csv.DictReader(stream, delimiter='\t'))
        status = 'PARTIAL_FAIL' if expected_fail else 'PARTIAL_PASS' if partial else 'PASS'
        assert row['validation'] == status, row
        assert row['result_finite'] == '1'
        assert len(json.loads(row['end_to_end_samples_seconds'])) == 1
        if not partial:
            assert 'validation_criterion=componentwise_dot_bound_v1' in run.stdout
            assert 'validation_rejected_coordinates=0' in run.stdout
            ratio = re.search(r'validation_max_error_ratio=(\S+)', run.stdout)
            assert ratio and float(ratio[1]) <= 1, run.stdout
            assert row['validation_criterion'] == 'componentwise_dot_bound_v1'
            # Exercise the real export/analysis checks, including errors above the legacy 1e-10.
            normalized = dict(row, compute_gflops_p90=row.get('compute_gflops', row.get('compute_gflops_p90')))
            experiments.validate_measurements(normalized, variant.startswith('trident_'))
            for mutation in (dict(validation_max_error_ratio='1.001'),
                             dict(validation_rejected_coordinates='1'),
                             dict(validation_reference_digits='24'),
                             dict(validation_criterion='unknown')):
                try:
                    experiments.validate_measurements(dict(normalized, **mutation), variant.startswith('trident_'))
                except ValueError:
                    pass
                else:
                    raise AssertionError('Invalid validation metadata accepted: ' + str(mutation))
            if name == absolute_name and variant.startswith('trident_'):
                legacy = {k: v for k, v in normalized.items() if k not in experiments.VALIDATION_FIELDS}
                try:
                    experiments.validate_measurements(legacy, True)
                except ValueError:
                    pass
                else:
                    raise AssertionError('Legacy PASS above 1e-10 was silently reclassified')
        if name == zero_name:
            assert int(row['c_nnz']) == (1 if q == 1 else 0)
        elif variant.startswith('trident_'):
            assert float(row['max_abs_error']) == 1.0186340659856796e-10
        print(f'{name}: {variant}, {q*q} logical nodes: {status}', flush=True)
print(f'PASS: {len(tests)} real-coordinate MPI runs and exact rational oracles')
