#!/usr/bin/env python3
"""Small local MPI comparison; preserves logs/TSVs and makes no timing equality assertion.

Run on a compute node on HPC systems. This uses two local ranks and one thread,
not the CRESCO scientific resource configuration.
"""
import argparse
import csv
import json
from pathlib import Path
import re
import statistics
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mpi-launcher', default='mpirun')
    args = parser.parse_args()
    build, output = args.build_dir.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    matrix = output / 'dense400.mtx'
    with matrix.open('w') as stream:
        stream.write('%%MatrixMarket matrix coordinate real general\n400 400 160000\n')
        for row in range(1, 401):
            stream.writelines('%d %d 1\n' % (row, col) for col in range(1, 401))
    cache = output / 'references'
    with (output / 'prepare.log').open('w') as stream:
        subprocess.run([str(build / 'prepare_partial_reference'), '--matrix', str(matrix),
                        '--partial', '--partial-cache', str(cache)], stdout=stream,
                       stderr=subprocess.STDOUT, check=True, timeout=120)
    records = []
    for run in range(3):
        for cached in ((False, True) if run % 2 == 0 else (True, False)):
            name = '%d-%s' % (run, 'cached' if cached else 'uncached')
            tsv = output / (name + '.tsv')
            command = [args.mpi_launcher, '-np', '2', '--oversubscribe', '--bind-to', 'none',
                       str(build / 'spgemm_two_sided'), '--matrix', str(matrix), '--partial',
                       '--threads', '1', '--warmup', '0', '--repeats', '10', '--trials', '1',
                       '--results', str(tsv)]
            if cached:
                command += ['--partial-cache', str(cache)]
            start = time.monotonic()
            result = subprocess.run(command, text=True, capture_output=True, timeout=120)
            wall = time.monotonic() - start
            (output / (name + '.log')).write_text(result.stdout + result.stderr)
            result.check_returncode()
            with tsv.open() as stream:
                observation = next(csv.DictReader(stream, delimiter='\t'))
            if observation['validation'] != 'PARTIAL_PASS':
                raise RuntimeError('Comparison validation failed')
            fields = dict(re.findall(r'(validation_\w+)=([^\s]+)', result.stdout))
            record = dict(run=run, cached=cached, wall_seconds=wall, command=command,
                          diagnostics=fields, full_product_seconds=float(observation['full_product_seconds']))
            records.append(record)
            print(json.dumps(record), flush=True)
    keys = ('validation_identity_seconds', 'validation_reference_build_seconds',
            'validation_cache_read_seconds', 'validation_compare_seconds')
    summary = {}
    for cached in (False, True):
        group = [r for r in records if r['cached'] == cached]
        summary['cached' if cached else 'uncached'] = dict(
            wall_seconds=statistics.median(r['wall_seconds'] for r in group),
            **{key: statistics.median(float(r['diagnostics'][key]) for r in group) for key in keys})
    (output / 'comparison.json').write_text(json.dumps(dict(records=records, medians=summary), indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
