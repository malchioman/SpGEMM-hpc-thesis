#!/usr/bin/env python3
"""Freeze and BUILD a new cached ten-sample campaign on a compute node; never submit jobs.

One reference job per pair (23 main + cage8 square probe), each with its own 24h budget.
The existing campaign is read-only and does not need submission receipts.
"""
import argparse
import copy
import json
import os
import re
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import launch_cresco_ten as ten
import launch_cresco_main as base


def compute_node():
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_JOB_NUM_NODES'):
        raise ValueError('Run preparation/reference generation in a Slurm compute allocation')


def snapshot(repo, destination):
    # Capture the working sources, including new files, rather than silently archiving only HEAD.
    for name in ('src', 'scripts', 'tests', 'docs'):
        shutil.copytree(repo / name, destination / name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('CMakeLists.txt', 'README.md'):
        shutil.copy2(repo / name, destination / name)


def build(source, directory):
    subprocess.run(['cmake', '-S', str(source), '-B', str(directory),
                    '-DCMAKE_BUILD_TYPE=Release'], check=True)
    subprocess.run(['cmake', '--build', str(directory), '--parallel', '8'], check=True)


def check_input_metadata(root, cfg):
    for frozen in (root / 'metadata/matrices').glob('*.json'):
        if ten.read(frozen) != ten.read(cfg.matrix_dir / 'metadata' / frozen.name):
            raise ValueError('Input metadata changed since source campaign: ' + frozen.name)


def reference_batch(root, index):
    return '\n'.join([
        '#!/usr/bin/env bash', '#SBATCH --job-name=spref-%02d' % index,
        '#SBATCH --partition=cresco8_cpu', '#SBATCH --nodes=1', '#SBATCH --ntasks=1',
        '#SBATCH --cpus-per-task=1', '#SBATCH --mem=450000M', '#SBATCH --exclusive',
        '#SBATCH --time=24:00:00', '#SBATCH --hint=nomultithread',
        '#SBATCH --output=' + str(root / ('logs/reference-%02d-%%j.out' % index)),
        '#SBATCH --error=' + str(root / ('logs/reference-%02d-%%j.err' % index)),
        'set -euo pipefail', 'source /usr/share/Modules/init/bash',
        'module load intel/oneapi mpi_flavour/openmpi_intel-4.1.7',
        'export OMP_NUM_THREADS=1',
        shlex.join(['python3', str(root / 'metadata/runtime/scripts/prepare_cresco_cached.py'),
                    'reference', '--root', str(root), '--pair', str(index)]), ''])


def prepare(repo, source, root):
    compute_node()
    if ten.below(root, source) or ten.below(source, root):
        raise ValueError('New and source campaign directories must not overlap')
    old = ten.verify(source)
    if old.get('schema') != 'tesi005-partial10-v1' or old['jobs'] != ten.plan():
        raise ValueError('Source must be the prepared ten-sample campaign')
    cfg = ten.read(source / 'metadata/n1.json')
    required = dict(repeats=10, trials=1, runs=1, warmup=0, validate='partial',
                    ranks_per_node=2, threads=63, cpus_per_rank=64,
                    variants=list(ten.VARIANTS), pilot_matrices=['cage8'])
    if any(cfg.get(k) != v for k, v in required.items()):
        raise ValueError('Unexpected source campaign configuration')
    # Exclusive new root: never patch frozen executables or reuse old result/PASS rows.
    root.mkdir(parents=True, exist_ok=False)
    frozen = root / 'metadata/source'
    snapshot(repo, frozen)
    source_hashes = {p.relative_to(frozen).as_posix(): ten.digest(p)
                     for p in frozen.rglob('*') if p.is_file()}
    build(frozen, root / 'metadata/build')
    if source_hashes != {p.relative_to(frozen).as_posix(): ten.digest(p)
                         for p in frozen.rglob('*') if p.is_file() and '__pycache__' not in p.parts}:
        raise ValueError('Frozen source changed during build')
    binaries = root / 'metadata/bin'
    binaries.mkdir()
    for name in list(ten.VARIANTS) + ['prepare_partial_reference', 'CMakeCache.txt']:
        shutil.copy2(root / 'metadata/build' / name, binaries / name)
    for name in base.RUNTIME + ('scripts/prepare_cresco_cached.py', 'scripts/launch_cresco_ten.py',
                                'scripts/launch_cresco_main.py'):
        dest = root / 'metadata/runtime' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(frozen / name, dest)
    shutil.copy2(frozen / 'scripts/launch_cresco_ten.py', root / 'metadata/launcher.py')
    shutil.copy2(frozen / 'scripts/launch_cresco_main.py', root / 'metadata/base_launcher.py')
    shutil.copytree(source / 'metadata/matrices', root / 'metadata/matrices')
    ten.write(root / 'metadata/previous-manifest.json', old)
    cache = root / 'references'
    manifest = dict(schema='partial10-cached-v1', jobs=ten.plan(), nodes=list(ten.NODES),
                    checks=str(root / 'checks'), source_campaign=str(source),
                    samples_per_launch=10, independent_launches=1, validation='partial',
                    excluded_pairs=[['reddit.mtx', 'reddit.mtx']],
                    partial_cache=str(cache), reference_policy='required-read-only', sha256={})
    ten.write(root / 'manifest.json', manifest)
    # Use the newly frozen runner with the new optional configuration key.
    _, e = ten.runtime(root)
    for nodes in ten.NODES:
        config = dict(cfg, nodes=[nodes], build_dir=str(binaries), partial_cache=str(cache),
                      results_dir=str(root / ('n%d' % nodes)))
        ten.write(root / ('metadata/n%d.json' % nodes), config)
    parsed = e.options(['all', '--config', str(root / 'metadata/n1.json')])
    check_input_metadata(root, parsed)
    pairs = {}
    coverage = set()
    for job in (j for j in manifest['jobs'] if j['nodes'] == 1):
        cases = [c for c in e.cases(parsed, job['action']) if c[0] in job['matrices']]
        if len(cases) != len(job['matrices']):
            raise ValueError('Missing cases in cached campaign')
        for case in cases:
            pairs[(str(case[2]), str(case[3]))] = (job['action'], case)
            for variant in job['variants']:
                identity = (job['action'], str(case[2]), str(case[3]), variant)
                if identity in coverage:
                    raise ValueError('Duplicate campaign observation')
                coverage.add(identity)
    if len(pairs) != 23 or len(coverage) != 184 or any(
            Path(a).name == Path(b).name == 'reddit.mtx' for a, b in pairs):
        raise ValueError('Unexpected cached campaign coverage')
    probe = copy.copy(parsed)
    probe.matrices = ['cage8']
    case = e.cases(probe, 'strong_scaling')[0]
    pairs[(str(case[2]), str(case[3]))] = ('strong_scaling', case)
    work = [(action, [case]) for action, case in pairs.values()]
    e.input_preflight(parsed, work)  # Existing raw-file SHA-256 provenance checks.
    # Full pilot inputs also need to remain valid, but do not get partial references.
    e.input_preflight(parsed, [('pilot', e.cases(parsed, 'pilot'))])
    manifest['reference_pairs'] = [dict(index=i, action=action, name=case[0], kind=case[1],
        matrix_a=str(case[2]), matrix_b=str(case[3]),
        script='metadata/reference-jobs/pair-%02d.sbatch' % i)
        for i, (action, case) in enumerate(pairs.values())]
    for pair in manifest['reference_pairs']:
        ten.write(root / pair['script'], reference_batch(root, pair['index']))
    for job in manifest['jobs']:
        (root / 'logs' / job['key']).mkdir(parents=True)
        ten.write(root / job['script'], ten.batch(root, job).replace('tesi005-', 'cached10-'))
    # Exclude build intermediates; freeze runnable binaries, exact sources and configuration.
    manifest['sha256'] = {p.relative_to(root).as_posix(): ten.digest(p)
        for p in (root / 'metadata').rglob('*') if p.is_file()
        and 'build' not in p.relative_to(root / 'metadata').parts and '__pycache__' not in p.parts}
    ten.write(root / 'manifest.json', manifest)
    print('Prepared 80 experiment jobs + 24 separate reference jobs; nothing submitted: ' + str(root))


def reference(root, index):
    compute_node()
    manifest = ten.verify(root)
    pair = next(p for p in manifest['reference_pairs'] if p['index'] == index)
    _, e = ten.runtime(root)
    cfg = e.options(['all', '--config', str(root / 'metadata/n1.json')])
    check_input_metadata(root, cfg)
    case = (pair['name'], pair['kind'], Path(pair['matrix_a']), Path(pair['matrix_b']))
    e.input_preflight(cfg, [(pair['action'], [case])])
    result = subprocess.run([str(cfg.build_dir / 'prepare_partial_reference'), '--matrix-a', str(case[2]),
                    '--matrix-b', str(case[3]), '--partial', '--partial-cache', str(cfg.partial_cache)],
                    capture_output=True, text=True)
    print(result.stdout, end='', flush=True)
    print(result.stderr, end='', file=sys.stderr, flush=True)
    result.check_returncode()
    match = re.search(r'\bvalidation_reference_key=([0-9a-f]{64})\b', result.stdout)
    if not match:
        raise ValueError('Generator did not report a reference key')
    # Unique receipt per pair. Publication by rename; no benchmark PASS is recorded here.
    record = dict(pair=pair, key=match[1], generator_sha256=ten.digest(cfg.build_dir / 'prepare_partial_reference'))
    path = root / 'reference-ready' / ('%02d.json' % index)
    path.parent.mkdir(exist_ok=True)
    temporary = path.with_name(path.name + '.%d.new' % os.getpid())
    temporary.write_text(json.dumps(record) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'reference'))
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--source', type=Path)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pair', type=int)
    args = parser.parse_args()
    if args.action == 'prepare':
        if args.source is None:
            parser.error('prepare requires --source (existing tesi-005 campaign)')
        prepare(args.repo.resolve(), args.source.resolve(), args.root.resolve())
    else:
        if args.pair is None:
            parser.error('reference requires --pair')
        reference(args.root.resolve(), args.pair)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError, StopIteration, subprocess.SubprocessError) as error:
        print('ERROR: ' + str(error), file=sys.stderr)
        sys.exit(1)
