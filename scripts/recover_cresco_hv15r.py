#!/usr/bin/env python3
"""Recover missing Trident HV15R square cases, original and permuted, with numeric validation.

The cached campaign must have terminated. Never edit its snapshots, references or TSVs.
One (action, nodes, variant) per job, original ten samples/resources, Full streaming validation.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

import launch_cresco_ten as ten
import prepare_cresco_cached as cached

VARIANTS = ten.VARIANTS[3:]
ACTIONS = ('strong_scaling', 'permutation')
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}
SCHEMA = 'hv15r-roundoff-v3'
CMAKE_ADDITIONS = (
    'add_executable(spgemm_full_streaming tests/spgemm_full_streaming.cpp)\n'
    'target_link_libraries(spgemm_full_streaming PRIVATE spgemm_support)\n'
    'enable_project_warnings(spgemm_full_streaming)\n'
    'add_test(NAME spgemm_full_streaming COMMAND spgemm_full_streaming)\n',
    '    add_test(NAME full_streaming_mpi\n'
    '        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/full_streaming_mpi_test.py\n'
    '                --build-dir ${CMAKE_CURRENT_BINARY_DIR} --mpi-launcher ${MPIEXEC_EXECUTABLE})\n'
    '    set_tests_properties(full_streaming_mpi PROPERTIES TIMEOUT 180 PROCESSORS 8)\n',
    '    add_test(NAME full_roundoff_hv15r\n'
    '        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/full_roundoff_hv15r_test.py\n'
    '                --build-dir ${CMAKE_CURRENT_BINARY_DIR} --mpi-launcher ${MPIEXEC_EXECUTABLE})\n'
    '    set_tests_properties(full_roundoff_hv15r PROPERTIES TIMEOUT 300 PROCESSORS 32)\n',
)


def finished_source(source, manifest):
    ids = []
    for job in manifest['jobs']:
        path = source / 'submission' / (job['key'] + '.json')
        value = ten.read(path).get('job_id', '')
        if not str(value).isdigit():
            raise ValueError('Uncertain source submission: ' + str(path))
        ids.append(str(value))
    result = subprocess.run(['sacct', '-X', '-n', '-P', '-j', ','.join(ids),
                             '--format=JobIDRaw,State'], check=True, capture_output=True, text=True)
    states = {}
    for line in result.stdout.splitlines():
        fields = line.strip().split('|')
        if len(fields) >= 2 and fields[0] in ids:
            states[fields[0]] = fields[1].split()[0].rstrip('+')
    unresolved = {job: states.get(job, 'UNKNOWN') for job in ids if states.get(job) not in TERMINAL}
    if unresolved:
        raise ValueError('Source campaign still active or accounting incomplete: ' + str(unresolved))
    return states


def validate_source(source):
    manifest = ten.verify(source)
    if manifest.get('schema') != 'partial10-cached-v1' or manifest['jobs'] != ten.plan():
        raise ValueError('Expected the unchanged tesi-006 cached campaign')
    finished_source(source, manifest)
    return manifest


def original_metadata(text):
    text = re.sub(r'^[ \t]*// BEGIN roundoff-validation-metadata-v1\n.*?'
                  r'^[ \t]*// END roundoff-validation-metadata-v1\n', '', text, flags=re.M | re.S)
    return re.sub(r'\n{3,}', '\n\n', text)


def original_common(text):
    """Remove exactly the approved post-timing validator addition for comparison."""
    start = '// BEGIN full-streaming-validation-v2\n'
    end = '// END full-streaming-validation-v2\n\n'
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError('Expected the reviewed streaming validator patch')
    first, rest = text.split(start)
    _, last = rest.split(end)
    replacement = '        return validateFullStreaming(matrixA, matrixB, result);'
    if last.count(replacement) != 1:
        raise ValueError('Unexpected Full dispatch')
    return original_metadata(first + last.replace(replacement,
        '        validation.maxAbsoluteError = maxAbsoluteDifference(result, serialSpgemm(matrixA, matrixB));\n'
        '        validation.passed = validationPassed(*validation.maxAbsoluteError);\n'
        '        return validation;'))


def guard_sources(repo, source):
    old = source / 'metadata/source'
    old_files = {p.relative_to(old / 'src') for p in (old / 'src').rglob('*') if p.is_file()}
    new_files = {p.relative_to(repo / 'src') for p in (repo / 'src').rglob('*') if p.is_file()}
    if old_files != new_files:
        raise ValueError('Production source file set changed')
    for relative in old_files:
        before = (old / 'src' / relative).read_text(encoding='utf-8')
        after = (repo / 'src' / relative).read_text(encoding='utf-8')
        if relative.as_posix() == 'common/spgemm_common.cpp':
            after = original_common(after)
        elif relative.as_posix() in ('common/spgemm_common.hpp', 'trident/common/benchmark.cpp'):
            after = original_metadata(after)
        if before != after:
            raise ValueError('Change outside the approved validator: src/' + relative.as_posix())
    cmake = (repo / 'CMakeLists.txt').read_text(encoding='utf-8')
    for addition in CMAKE_ADDITIONS:
        if cmake.count(addition) != 1:
            raise ValueError('Unexpected recovery CMake tests')
        cmake = cmake.replace(addition, '')
    if cmake != (old / 'CMakeLists.txt').read_text(encoding='utf-8'):
        raise ValueError('Production CMake options changed')


def result_snapshot(source):
    return {p.relative_to(source).as_posix(): ten.digest(p)
            for p in source.glob('n*/*/*.tsv') if p.is_file()}


def compiler_settings(build):
    cache = {}
    for line in (build / 'CMakeCache.txt').read_text().splitlines():
        match = re.match(r'([^:#]+):[^=]+=(.*)', line)
        if match:
            cache[match[1]] = match[2]
    selected = {key: cache.get(key) for key in (
        'CMAKE_CXX_COMPILER', 'CMAKE_CXX_FLAGS', 'CMAKE_CXX_FLAGS_RELEASE', 'CMAKE_BUILD_TYPE',
        'OpenMP_CXX_FLAGS', 'SPGEMM_MPI_CHUNK_LIMIT')}
    candidates = list((build / 'CMakeFiles').glob('*/CMakeCXXCompiler.cmake'))
    if len(candidates) != 1:
        raise ValueError('Cannot establish previous/current compiler identity: ' + str(build))
    text = candidates[0].read_text()
    for key in ('CMAKE_CXX_COMPILER_ID', 'CMAKE_CXX_COMPILER_VERSION'):
        match = re.search(r'set\(' + key + r' "([^"]+)"\)', text)
        if not match:
            raise ValueError('Compiler identity missing: ' + key)
        selected[key] = match[1]
    return selected


def missing_jobs(source):
    base, e = ten.runtime(source)
    jobs = []
    for nodes in ten.NODES:
        cfg = e.options(['strong_scaling', '--config', str(source / ('metadata/n%d.json' % nodes))])
        required = dict(repeats=10, trials=1, runs=1, warmup=0, validate='partial',
                        ranks_per_node=2, threads=63, cpus_per_rank=64, schedule='guided', chunk=64)
        if any(getattr(cfg, key) != value for key, value in required.items()):
            raise ValueError('Unexpected source measurement configuration')
        for action in ACTIONS:
            done = base.existing(e, cfg, action)
            cases = [c for c in e.cases(cfg, action) if c[0] == 'HV15R']
            if len(cases) != 1:
                raise ValueError('Expected one original/permuted HV15R case: ' + action)
            pair = tuple(p.relative_to(cfg.matrix_dir).as_posix() for p in cases[0][2:])
            for variant in VARIANTS:
                if (variant,) + pair not in done:
                    suffix = variant if action == 'strong_scaling' else action + '-' + variant
                    key = 'n%d-%s' % (nodes, suffix)
                    jobs.append(dict(key=key, nodes=nodes, action=action, variant=variant, pair=list(pair),
                                     script='metadata/jobs/' + key + '.sbatch'))
    return jobs


def batch(root, job):
    q = shlex.quote
    nodes, key = job['nodes'], job['key']
    return '\n'.join([
        '#!/usr/bin/env bash', '#SBATCH --job-name=hvfull-' + key,
        '#SBATCH --partition=cresco8_cpu', '#SBATCH --nodes=%d' % nodes,
        '#SBATCH --ntasks=%d' % (2 * nodes), '#SBATCH --ntasks-per-node=2',
        '#SBATCH --cpus-per-task=64', '#SBATCH --mem=450000M', '#SBATCH --exclusive',
        '#SBATCH --hint=nomultithread', '#SBATCH --time=24:00:00',
        '#SBATCH --output=' + q(str(root / 'logs' / key / '%j.out')),
        '#SBATCH --error=' + q(str(root / 'logs' / key / '%j.err')),
        'set -euo pipefail', 'source /usr/share/Modules/init/bash',
        'module load intel/oneapi mpi_flavour/openmpi_intel-4.1.7',
        'job_dir=' + q(str(root / 'logs' / key)) + '/"${SLURM_JOB_ID}"',
        'mkdir -p "$job_dir"', 'exec > >(tee "$job_dir/launch.log") 2>&1',
        'cd -- ' + q(str(root / 'metadata/source')),
        'mpirun --mca btl_vader_single_copy_mechanism none -np "$SLURM_JOB_NUM_NODES" '
        '--map-by ppr:1:node --bind-to none bash scripts/collect_machine.sh "$job_dir" '
        + q(str(root / 'metadata/bin')),
        shlex.join(['python3', str(root / 'metadata/source/scripts/recover_cresco_hv15r.py'),
                    'run', '--root', str(root), '--key', key]), ''])


def prepare(repo, source, root):
    cached.compute_node()
    if ten.below(root, source) or ten.below(source, root):
        raise ValueError('Recovery and source roots must not overlap')
    original = validate_source(source)
    guard_sources(repo, source)
    jobs = missing_jobs(source)
    if not jobs:
        print('No missing Trident HV15R square cases; nothing prepared.')
        return
    root.mkdir(parents=True, exist_ok=False)
    frozen = root / 'metadata/source'
    cached.snapshot(repo, frozen)
    guard_sources(frozen, source)
    cached.build(frozen, root / 'metadata/build')
    previous = compiler_settings(source / 'metadata/build')
    current = compiler_settings(root / 'metadata/build')
    if previous != current:
        raise ValueError('Compiler/optimization settings changed: ' + str((previous, current)))
    env = dict(os.environ, OMP_NUM_THREADS='1', OMPI_MCA_btl_vader_single_copy_mechanism='none')
    # The MPI regression uses eight local ranks with two threads each (16 allocated CPUs).
    subprocess.run(['ctest', '--test-dir', str(root / 'metadata/build'), '--output-on-failure',
                    '-R', '^(spgemm_full_streaming|spgemm_validation_smoke|full_streaming_mpi)$'],
                   check=True, env=env)
    bins = root / 'metadata/bin'
    bins.mkdir()
    for variant in VARIANTS:
        shutil.copy2(root / 'metadata/build' / variant, bins / variant)
    for nodes in ten.NODES:
        config = ten.read(source / ('metadata/n%d.json' % nodes))
        config.update(build_dir=str(bins), results_dir=str(root / ('n%d' % nodes)),
                      validate=True, partial_cache=None, matrices=['HV15R'],
                      permutation_matrices=['HV15R'], variants=list(VARIANTS))
        ten.write(root / ('metadata/n%d.json' % nodes), config)
    for job in jobs:
        (root / 'logs' / job['key']).mkdir(parents=True)
        ten.write(root / job['script'], batch(root, job))
    probes = []
    for action in ACTIONS:
        candidates = [j for j in jobs if j['action'] == action]
        if candidates:
            probe = next((j for j in candidates if j['nodes'] == 4 and j['variant'] == 'trident_two_sided'),
                         candidates[0])
            probes.append(probe['key'])
    manifest = dict(schema=SCHEMA, source=str(source), source_manifest_sha256=ten.digest(source / 'manifest.json'),
                    source_snapshot_sha256=original['sha256'], jobs=jobs, probes=probes,
                    source_results_sha256=result_snapshot(source),
                    validation='streaming_roundoff_v2', criterion='componentwise_dot_bound_v1',
                    compiler=current, checks=str(root / 'checks'))
    manifest['sha256'] = {p.relative_to(root).as_posix(): ten.digest(p)
                          for p in (root / 'metadata').rglob('*') if p.is_file()
                          and 'build' not in p.relative_to(root / 'metadata').parts
                          and '__pycache__' not in p.parts}
    ten.write(root / 'manifest.json', manifest)
    print('Prepared %d single-case jobs. Probes: %s. Nothing submitted.' % (len(jobs), ', '.join(probes)))


def verify(root):
    manifest = ten.verify(root)
    if manifest.get('schema') != SCHEMA:
        raise ValueError('Unexpected recovery manifest')
    source = Path(manifest['source'])
    if ten.digest(source / 'manifest.json') != manifest['source_manifest_sha256']:
        raise ValueError('Source manifest changed')
    validate_source(source)
    if result_snapshot(source) != manifest['source_results_sha256']:
        raise ValueError('Source result tables changed after recovery preparation')
    return manifest, source


def runtime(root, source, nodes, variant, action='strong_scaling'):
    base, _ = ten.runtime(source)
    # Use the new reader only for recovery data. The old campaign keeps its frozen runner.
    path = root / 'metadata/source/scripts/lib/experiments.py'
    spec = importlib.util.spec_from_file_location('_hv_roundoff_experiments', path)
    e = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e)
    original_write = e.write_table
    original_validate = e.validate_measurements

    def recovery_measurements(row, trident):
        if row.get('validation_criterion') != 'componentwise_dot_bound_v1':
            raise ValueError('Recovery requires explicit componentwise numerical validation')
        original_validate(row, trident)

    def atomic_table(path, fields, rows, append=False):
        data = e.read_table(path, fields) + list(rows) if append else rows
        original_write(path, fields, data, append=False)

    e.write_table = atomic_table
    e.validate_measurements = recovery_measurements
    e.result_lock = ten.file_lock
    e.check_root = lambda path: root / 'checks' / path.relative_to(root)
    cfg = e.options([action, '--config', str(root / ('metadata/n%d.json' % nodes)),
                     '--variants', variant])
    return base, e, cfg


def complete(root, source, job):
    base, e, cfg = runtime(root, source, job['nodes'], job['variant'], job['action'])
    return (job['variant'],) + tuple(job['pair']) in base.existing(e, cfg, job['action'])


def submit(root, remaining=False):
    manifest, source = verify(root)
    probes = [next(j for j in manifest['jobs'] if j['key'] == key) for key in manifest['probes']]
    if remaining:
        for probe in probes:
            state = root / 'state' / (probe['key'] + '.json')
            if not state.is_file() or not ten.read(state).get('complete') or not complete(root, source, probe):
                raise ValueError('HV15R numeric probe has not passed: ' + probe['key'] +
                                 '. Remaining jobs are not authorized by this gate.')
        selected = [j for j in manifest['jobs'] if j['key'] not in manifest['probes']]
    else:
        selected = probes
    with ten.file_lock(root / 'submission'):
        prior = {}
        for job in selected:
            receipt = root / 'submission' / (job['key'] + '.json')
            if receipt.exists():
                job_id = str(ten.read(receipt).get('job_id', ''))
                if not job_id.isdigit():
                    raise ValueError('Ambiguous prior submission, inspect accounting: ' + str(receipt))
                prior[job['nodes']] = job_id
                continue
            if complete(root, source, job):
                continue
            command = ['sbatch', '--parsable']
            if job['nodes'] in prior:
                command += ['--dependency=afterany:' + prior[job['nodes']]]
            command += [str(root / job['script'])]
            # Resource validation happens before leaving an ambiguous-submission receipt.
            subprocess.run(['sbatch', '--test-only', str(root / job['script'])], check=True)
            with receipt.open('x', encoding='utf-8') as stream:
                json.dump(dict(state='submitting', command=command), stream)
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            job_id = result.stdout.strip().split(';')[0]
            if not job_id.isdigit():
                raise ValueError('Unexpected sbatch response: ' + result.stdout)
            ten.write(receipt, dict(job_id=job_id, command=command))
            prior[job['nodes']] = job_id
            print(job['key'] + ' JOB=' + job_id, flush=True)


def run(root, key):
    manifest, source = verify(root)
    job = next(j for j in manifest['jobs'] if j['key'] == key)
    base, e, cfg = runtime(root, source, job['nodes'], job['variant'], job['action'])
    from allocated_experiment import check_allocation
    check_allocation(cfg, 'slurm', os.environ)
    cached.check_input_metadata(source, cfg)
    # Full cage8 pilot checks remain outside the scientific result tables.
    ten.perform(base, e, cfg, 'pilot', e.cases(cfg, 'pilot'))
    cases = [c for c in e.cases(cfg, job['action']) if c[0] == 'HV15R']
    if len(cases) != 1 or tuple(p.relative_to(cfg.matrix_dir).as_posix() for p in cases[0][2:]) != tuple(job['pair']):
        raise ValueError('Unexpected recovery pair')
    ten.perform(base, e, cfg, job['action'], cases)
    if not complete(root, source, job):
        raise ValueError('No accepted full-numeric observation')
    ten.write(root / 'state' / (key + '.json'), dict(complete=True, job_id=os.environ['SLURM_JOB_ID'],
                                                   validation='PASS', strategy='streaming_roundoff_v2',
                                                   criterion='componentwise_dot_bound_v1'))
    print('NUMERIC RECOVERY COMPLETED: ' + key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'submit-probe', 'submit-remaining', 'run', 'status'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--key')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == 'prepare':
        if args.source is None:
            parser.error('prepare requires --source')
        prepare(args.repo.resolve(), args.source.resolve(), root)
    elif args.action.startswith('submit-'):
        submit(root, args.action == 'submit-remaining')
    elif args.action == 'run':
        if not args.key:
            parser.error('run requires --key')
        run(root, args.key)
    else:
        manifest, source = verify(root)
        for job in manifest['jobs']:
            print(job['key'] + ': ' + ('PASS' if complete(root, source, job) else 'MISSING/FAILED/PENDING'))
        ids = [str(ten.read(p)['job_id']) for p in (root / 'submission').glob('*.json')
               if str(ten.read(p).get('job_id', '')).isdigit()]
        if ids:
            subprocess.run(['sacct', '-X', '-j', ','.join(ids),
                            '--format=JobID,JobName%45,State,Elapsed,ExitCode'], check=True)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError, StopIteration, subprocess.SubprocessError) as error:
        print('ERROR: ' + str(error), file=sys.stderr)
        sys.exit(1)
