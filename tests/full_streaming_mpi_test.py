#!/usr/bin/env python3
"""Reproduce strict row-nnz false failure with the actual five Trident transports."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('--build-dir', type=Path, required=True)
parser.add_argument('--mpi-launcher', default='mpirun')
args = parser.parse_args()
variants = ('trident_two_sided', 'trident_hybrid', 'trident_get', 'trident_get_pipeline', 'trident_put')
matrix = Path(__file__).resolve().parent / 'data/roundoff_stages.mtx'
with tempfile.TemporaryDirectory(prefix='full-streaming-') as tmp:
    for variant in variants:
        observations = []
        for partial in (True, False):
            output = Path(tmp) / (variant + '-' + str(partial) + '.tsv')
            command = [args.mpi_launcher, '-np', '8', '--oversubscribe', '--bind-to', 'none',
                       str(args.build_dir / variant), '--logical-node-size', '2',
                       '--matrix', str(matrix), '--threads', '2', '--warmup', '0',
                       '--repeats', '10', '--trials', '1', '--results', str(output)]
            if partial:
                command += ['--partial']
            run = subprocess.run(command, capture_output=True, text=True, timeout=45)
            assert run.returncode == (1 if partial else 0), run.stdout + run.stderr
            with output.open() as stream:
                row = next(csv.DictReader(stream, delimiter='\t'))
            assert row['validation'] == ('PARTIAL_FAIL' if partial else 'PASS'), row
            assert row['c_nnz'] == '1', row
            assert row['result_finite'] == '1'
            for field in ('compute', 'communication', 'inter_node', 'intra_node', 'end_to_end'):
                assert len(json.loads(row[field + '_samples_seconds'])) == 10
            if not partial:
                assert float(row['max_abs_error']) == 2.0**-55
                assert 'validation_row_nnz_mismatches=1' in run.stdout
                assert 'validation_support_mismatches=1' in run.stdout
                assert 'validation_full_rows=8' in run.stdout
            observations.append(row)
        assert observations[0].keys() == observations[1].keys(), 'TSV schema changed'
        assert observations[0]['benchmark_protocol'] == observations[1]['benchmark_protocol']
        print(variant + ': PARTIAL_FAIL -> numeric PASS, unchanged 10 samples and protocol', flush=True)
