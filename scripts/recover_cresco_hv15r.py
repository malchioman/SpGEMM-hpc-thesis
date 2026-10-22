#!/usr/bin/env python3
"""Recover missing Trident HV15R square cases, original and permuted, with numeric validation.

Normally the cached campaign must have terminated. --early-probes only requires
the original four-node jobs to have ended; finalize waits for the entire campaign.
Never edit its snapshots, references or TSVs.
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
SCHEMA = 'hv15r-roundoff-cached-v4'
CACHE_SOURCES = {'common/full_reference.cpp', 'common/full_reference.hpp', 'common/prepare_full_reference.cpp'}
CMAKE_ADDITIONS = (
    '# BEGIN full-reference-cache-v1\nadd_executable(prepare_full_reference src/common/prepare_full_reference.cpp)\ntarget_link_libraries(prepare_full_reference PRIVATE spgemm_support)\nenable_project_warnings(prepare_full_reference)\nadd_executable(full_reference_test tests/full_reference_test.cpp)\ntarget_link_libraries(full_reference_test PRIVATE spgemm_support)\nenable_project_warnings(full_reference_test)\nadd_test(NAME full_reference COMMAND full_reference_test ${CMAKE_CURRENT_BINARY_DIR}/full-reference-test)\n# END full-reference-cache-v1\n\n',
    '        src/common/full_reference.cpp src/common/full_reference.hpp src/common/prepare_full_reference.cpp\n',
    '    src/common/full_reference.cpp\n',
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


def finished_source(source, manifest, nodes=None):
    ids = []
    for job in manifest['jobs']:
        if nodes is not None and job['nodes'] not in nodes:
            continue
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


def validate_source(source, nodes=None):
    manifest = ten.verify(source)
    if manifest.get('schema') != 'partial10-cached-v1' or manifest['jobs'] != ten.plan():
        raise ValueError('Expected the unchanged tesi-006 cached campaign')
    finished_source(source, manifest, nodes)
    return manifest


def original_cache(text):
    text = re.sub(r'^[ \t]*// BEGIN full-reference-cache-v1\n.*?'
                  r'^[ \t]*// END full-reference-cache-v1\n', '', text, flags=re.M | re.S)
    text = text.replace('(argument == "--partial-cache" || argument == "--full-cache")', 'argument == "--partial-cache"')
    text = text.replace('const std::string& partialCacheDirectory, const std::string& fullCacheDirectory)',
                        'const std::string& partialCacheDirectory)')
    text = text.replace('const std::string& partialCacheDirectory = "",\n                                 const std::string& fullCacheDirectory = "")',
                        'const std::string& partialCacheDirectory = "")')
    text = text.replace('options.partialCacheDirectory, options.fullCacheDirectory)', 'options.partialCacheDirectory)')
    return text.replace('validateFullStreaming(matrixA, matrixB, result, fullCacheDirectory)',
                        'validateFullStreaming(matrixA, matrixB, result)')


def original_metadata(text):
    text = original_cache(text)
    text = re.sub(r'^[ \t]*// BEGIN roundoff-validation-metadata-v1\n.*?'
                  r'^[ \t]*// END roundoff-validation-metadata-v1\n', '', text, flags=re.M | re.S)
    return re.sub(r'\n{3,}', '\n\n', text)


def original_common(text):
    """Remove exactly the approved post-timing validator addition for comparison."""
    text = original_cache(text)
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
    if {p.as_posix() for p in new_files - old_files} != CACHE_SOURCES or old_files - new_files:
        raise ValueError('Production source file set changed')
    for relative in old_files:
        before = (old / 'src' / relative).read_text(encoding='utf-8')
        after = original_cache((repo / 'src' / relative).read_text(encoding='utf-8'))
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


def result_snapshot(source, nodes=None):
    return {p.relative_to(source).as_posix(): ten.digest(p)
            for p in source.glob('n*/*/*.tsv') if p.is_file()
            and (nodes is None or p.relative_to(source).parts[0] in {'n%d' % n for n in nodes})}


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


def missing_jobs(source, nodes=None, read_results=True):
    base, e = ten.runtime(source)
    jobs = []
    for node_count in (ten.NODES if nodes is None else nodes):
        cfg = e.options(['strong_scaling', '--config', str(source / ('metadata/n%d.json' % node_count))])
        required = dict(repeats=10, trials=1, runs=1, warmup=0, validate='partial',
                        ranks_per_node=2, threads=63, cpus_per_rank=64, schedule='guided', chunk=64)
        if any(getattr(cfg, key) != value for key, value in required.items()):
            raise ValueError('Unexpected source measurement configuration')
        for action in ACTIONS:
            done = base.existing(e, cfg, action) if read_results else set()
            cases = [c for c in e.cases(cfg, action) if c[0] == 'HV15R']
            if len(cases) != 1:
                raise ValueError('Expected one original/permuted HV15R case: ' + action)
            pair = tuple(p.relative_to(cfg.matrix_dir).as_posix() for p in cases[0][2:])
            for variant in VARIANTS:
                if (variant,) + pair not in done:
                    suffix = variant if action == 'strong_scaling' else action + '-' + variant
                    key = 'n%d-%s' % (node_count, suffix)
                    jobs.append(dict(key=key, nodes=node_count, action=action, variant=variant, pair=list(pair),
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


def prepare(repo, source, root, reference_cache_dir=None, early_probes=False):
    cached.compute_node()
    if ten.below(root, source) or ten.below(source, root):
        raise ValueError('Recovery and source roots must not overlap')
    reference_cache_dir = reference_cache_dir or root / 'references'
    if ten.below(reference_cache_dir, source) or ten.below(source, reference_cache_dir):
        raise ValueError('Reference cache must not overlap the source campaign')
    scope = (4,) if early_probes else None
    original = validate_source(source, scope)
    guard_sources(repo, source)
    if early_probes:
        # Only inspect results whose writers are known to be terminal. Other cases
        # are candidates, not a claim that the still-running campaign missed them.
        four_node_jobs = missing_jobs(source, nodes=(4,))
        if any(not any(j['action'] == action for j in four_node_jobs) for action in ACTIONS):
            raise ValueError('Early probes require a missing four-node case for each HV15R product')
        jobs = four_node_jobs + missing_jobs(source, nodes=tuple(n for n in ten.NODES if n != 4),
                                            read_results=False)
        jobs.sort(key=lambda j: (ten.NODES.index(j['nodes']), ACTIONS.index(j['action']),
                                 VARIANTS.index(j['variant'])))
    else:
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
                    '-R', '^(full_reference|spgemm_full_streaming|spgemm_validation_smoke|full_streaming_mpi)$'],
                   check=True, env=env)
    bins = root / 'metadata/bin'
    bins.mkdir()
    for variant in (*VARIANTS, 'prepare_full_reference'):
        shutil.copy2(root / 'metadata/build' / variant, bins / variant)
    for nodes in ten.NODES:
        config = ten.read(source / ('metadata/n%d.json' % nodes))
        config.update(build_dir=str(bins), results_dir=str(root / ('n%d' % nodes)),
                      validate=True, partial_cache=None, full_cache=str(reference_cache_dir), matrices=['HV15R'],
                      permutation_matrices=['HV15R'], variants=list(VARIANTS))
        ten.write(root / ('metadata/n%d.json' % nodes), config)
    for job in jobs:
        (root / 'logs' / job['key']).mkdir(parents=True)
        ten.write(root / job['script'], batch(root, job))
    probes = []
    for action in ACTIONS:
        candidates = [j for j in jobs if j['action'] == action and (not early_probes or j['nodes'] == 4)]
        if candidates:
            probe = next((j for j in candidates if j['nodes'] == 4 and j['variant'] == 'trident_two_sided'),
                         candidates[0])
            probes.append(probe['key'])
    references = [dict(key=action, pair=next(j['pair'] for j in jobs if j['action'] == action),
                       script='metadata/reference-jobs/' + action + '.sbatch')
                  for action in ACTIONS if any(j['action'] == action for j in jobs)]
    for ref in references:
        ten.write(root / ref['script'], reference_batch(root, ref['key']))
    manifest = dict(schema=SCHEMA, source=str(source), source_manifest_sha256=ten.digest(source / 'manifest.json'),
                    source_snapshot_sha256=original['sha256'], jobs=jobs, probes=probes,
                    source_results_sha256=result_snapshot(source, scope), reference_jobs=references,
                    phase='preliminary' if early_probes else 'final',
                    full_cache=str(reference_cache_dir), reference_policy='required-read-only',
                    validation='streaming_roundoff_v2', criterion='componentwise_dot_bound_v1',
                    compiler=current, checks=str(root / 'checks'))
    manifest['sha256'] = {p.relative_to(root).as_posix(): ten.digest(p)
                          for p in (root / 'metadata').rglob('*') if p.is_file()
                          and 'build' not in p.relative_to(root / 'metadata').parts
                          and '__pycache__' not in p.parts}
    ten.write(root / 'manifest.json', manifest)
    print('Prepared %d single-case jobs + %d shared numeric references. Probes: %s. Nothing submitted.' %
          (len(jobs), len(references), ', '.join(probes)))
    if early_probes:
        print('PRELIMINARY ONLY: references and four-node probes may run now. '
              'Other cases are candidates; finalize after the source campaign ends.')


def verify(root):
    manifest = ten.verify(root)
    if manifest.get('schema') != SCHEMA:
        raise ValueError('Unexpected recovery manifest')
    source = Path(manifest['source'])
    if ten.digest(source / 'manifest.json') != manifest['source_manifest_sha256']:
        raise ValueError('Source manifest changed')
    phase = manifest.get('phase', 'final')  # Existing v4 snapshots retain their strict gate.
    if phase not in ('preliminary', 'final'):
        raise ValueError('Unknown recovery phase')
    scope = (4,) if phase == 'preliminary' else None
    if scope and any(j['nodes'] != 4 for j in manifest['jobs'] if j['key'] in manifest['probes']):
        raise ValueError('Early probes must use four nodes')
    validate_source(source, scope)
    if result_snapshot(source, scope) != manifest['source_results_sha256']:
        raise ValueError('Source result tables changed after recovery preparation')
    return manifest, source


def require_passed_probes(root, source, manifest):
    probes = [next(j for j in manifest['jobs'] if j['key'] == key) for key in manifest['probes']]
    for probe in probes:
        state = root / 'state' / (probe['key'] + '.json')
        if not state.is_file() or not ten.read(state).get('complete') or not complete(root, source, probe):
            raise ValueError('HV15R numeric probe has not passed: ' + probe['key'] +
                             '. Remaining jobs are not authorized by this gate.')


def finalize(root):
    """Freeze the final missing-case selection without rebuilding or losing probe samples."""
    with ten.file_lock(root / 'submission'):
        manifest, source = verify(root)
        if manifest.get('phase', 'final') == 'final':
            print('Recovery plan already final; nothing changed.')
            return
        validate_source(source)  # All original jobs, including pending 100/121-node jobs.
        require_passed_probes(root, source, manifest)
        require_references(root, manifest)
        jobs = missing_jobs(source)
        known = {j['key']: j for j in manifest['jobs']}
        if any(known.get(j['key']) != j for j in jobs):
            raise ValueError('Final cases differ from the prepared candidates')
        # Keep successful probe observations in the manifest and final data set.
        for key in manifest['probes']:
            if key not in {j['key'] for j in jobs}:
                jobs.append(known[key])
        manifest['preliminary_source_results_sha256'] = manifest['source_results_sha256']
        manifest.update(phase='final', jobs=jobs, source_results_sha256=result_snapshot(source))
        ten.write(root / 'manifest.json', manifest)
        print('FINAL PLAN: %d cases, including %d preserved probes. Nothing submitted or rebuilt.' %
              (len(jobs), len(manifest['probes'])))


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


def reference_batch(root, key):
    return '\n'.join([
        '#!/usr/bin/env bash', '#SBATCH --job-name=hvref-' + key,
        '#SBATCH --partition=cresco8_cpu', '#SBATCH --nodes=1', '#SBATCH --ntasks=1',
        '#SBATCH --cpus-per-task=1', '#SBATCH --mem=450000M', '#SBATCH --exclusive',
        '#SBATCH --hint=nomultithread', '#SBATCH --time=24:00:00',
        '#SBATCH --output=' + str(root / 'logs' / ('reference-' + key + '-%j.log')),
        '#SBATCH --error=' + str(root / 'logs' / ('reference-' + key + '-%j.log')),
        'set -euo pipefail', 'source /usr/share/Modules/init/bash',
        'module load intel/oneapi mpi_flavour/openmpi_intel-4.1.7',
        'export OMP_NUM_THREADS=1',
        shlex.join(['python3', str(root / 'metadata/source/scripts/recover_cresco_hv15r.py'),
                    'reference', '--root', str(root), '--key', key]), ''])


def reference(root, key):
    cached.compute_node()
    manifest, source = verify(root)
    ref = next(r for r in manifest['reference_jobs'] if r['key'] == key)
    _, e, cfg = runtime(root, source, 1, VARIANTS[0], key)
    cached.check_input_metadata(source, cfg)
    cases = [c for c in e.cases(cfg, key) if c[0] == 'HV15R']
    if len(cases) != 1 or list(p.relative_to(cfg.matrix_dir).as_posix() for p in cases[0][2:]) != ref['pair']:
        raise ValueError('Unexpected reference inputs')
    e.input_preflight(cfg, [(key, cases)])
    generator = root / 'metadata/bin/prepare_full_reference'
    result = subprocess.run([str(generator), '--matrix-a', str(cases[0][2]), '--matrix-b', str(cases[0][3]),
                             '--full-cache', manifest['full_cache']], capture_output=True, text=True)
    print(result.stdout, end='', flush=True)
    print(result.stderr, end='', file=sys.stderr, flush=True)
    result.check_returncode()
    match = re.search(r'\bvalidation_reference_key=([0-9a-f]{64})\b', result.stdout)
    if not match:
        raise ValueError('Numeric generator did not report its reference identity')
    path = Path(manifest['full_cache']) / (match[1] + '.sfull')
    receipt = root / 'reference-ready' / (key + '.json')
    ten.write(receipt, dict(pair=ref['pair'], key=match[1], generator_sha256=ten.digest(generator),
                           bytes=path.stat().st_size, criterion='componentwise_dot_bound_v1'))
    print('FULL REFERENCE READY: ' + key + ' bytes=' + str(path.stat().st_size))


def require_references(root, manifest, key=None):
    for ref in manifest['reference_jobs']:
        if key is not None and ref['key'] != key:
            continue
        receipt = root / 'reference-ready' / (ref['key'] + '.json')
        if not receipt.is_file():
            raise ValueError('Numeric reference not ready: ' + ref['key'] + '; run submit-references first')
        record = ten.read(receipt)
        if (not re.fullmatch(r'[0-9a-f]{64}', record.get('key', '')) or record.get('pair') != ref['pair'] or
                record.get('criterion') != 'componentwise_dot_bound_v1' or
                record.get('generator_sha256') != ten.digest(root / 'metadata/bin/prepare_full_reference')):
            raise ValueError('Invalid numeric reference receipt: ' + ref['key'])
        path = Path(manifest['full_cache']) / (record['key'] + '.sfull')
        if not path.is_file() or path.stat().st_size != record.get('bytes') or path.stat().st_size < 176:
            raise ValueError('Numeric reference missing/truncated: ' + str(path))
        # The executable checks full content SHA-256 while streaming every comparison.
        with path.open('rb') as stream:
            if stream.read(72) != b'SPFR0001' + record['key'].encode('ascii'):
                raise ValueError('Incompatible numeric reference header: ' + str(path))


def submit_references(root):
    manifest, _ = verify(root)
    with ten.file_lock(root / 'reference-submission'):
        for ref in manifest['reference_jobs']:
            ready = root / 'reference-ready' / (ref['key'] + '.json')
            if ready.exists():
                require_references(root, manifest, ref['key'])
                continue
            receipt = root / 'reference-submission' / (ref['key'] + '.json')
            if receipt.exists():
                if not str(ten.read(receipt).get('job_id', '')).isdigit():
                    raise ValueError('Ambiguous reference submission: ' + str(receipt))
                continue
            command = ['sbatch', '--parsable', str(root / ref['script'])]
            subprocess.run(['sbatch', '--test-only', str(root / ref['script'])], check=True)
            with receipt.open('x', encoding='utf-8') as stream:
                json.dump(dict(state='submitting', command=command), stream)
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            job_id = result.stdout.strip().split(';')[0]
            if not job_id.isdigit():
                raise ValueError('Unexpected reference sbatch response: ' + result.stdout)
            ten.write(receipt, dict(job_id=job_id, command=command))
            print('REFERENCE ' + ref['key'] + ' JOB=' + job_id, flush=True)


def submit(root, remaining=False):
    manifest, source = verify(root)
    if remaining and manifest.get('phase') == 'preliminary':
        raise ValueError('Preliminary plan: run finalize after all source jobs have ended')
    require_references(root, manifest)
    probes = [next(j for j in manifest['jobs'] if j['key'] == key) for key in manifest['probes']]
    if remaining:
        require_passed_probes(root, source, manifest)
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
    if manifest.get('phase') == 'preliminary' and key not in manifest['probes']:
        raise ValueError('Preliminary plan permits only the four-node probes; finalize first')
    require_references(root, manifest, key=next(j['action'] for j in manifest['jobs'] if j['key'] == key))
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
    parser.add_argument('action', choices=('prepare', 'finalize', 'submit-references', 'reference', 'submit-probe', 'submit-remaining', 'run', 'status'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--key')
    parser.add_argument('--reference-cache-dir', type=Path, help='shared filesystem directory with room for numeric references')
    parser.add_argument('--early-probes', action='store_true',
                        help='prepare references and four-node probes while other source node counts are still active')
    args = parser.parse_args()
    if args.early_probes and args.action != 'prepare':
        parser.error('--early-probes is only valid with prepare')
    root = args.root.resolve()
    if args.action == 'prepare':
        if args.source is None:
            parser.error('prepare requires --source')
        prepare(args.repo.resolve(), args.source.resolve(), root,
                args.reference_cache_dir.resolve() if args.reference_cache_dir else None, args.early_probes)
    elif args.action == 'finalize':
        finalize(root)
    elif args.action == 'submit-references':
        submit_references(root)
    elif args.action == 'reference':
        if not args.key:
            parser.error('reference requires --key strong_scaling or permutation')
        reference(root, args.key)
    elif args.action.startswith('submit-'):
        submit(root, args.action == 'submit-remaining')
    elif args.action == 'run':
        if not args.key:
            parser.error('run requires --key')
        run(root, args.key)
    else:
        manifest, source = verify(root)
        preliminary = manifest.get('phase') == 'preliminary'
        if preliminary:
            print('PRELIMINARY: references and four-node probes only; final recovery selection is pending.')
        for ref in manifest['reference_jobs']:
            print('REFERENCE ' + ref['key'] + ': ' + ('READY' if (root / 'reference-ready' / (ref['key'] + '.json')).is_file() else 'MISSING/PENDING'))
        for job in manifest['jobs']:
            if preliminary and job['key'] not in manifest['probes']:
                continue
            print(job['key'] + ': ' + ('PASS' if complete(root, source, job) else 'MISSING/FAILED/PENDING'))
        receipts = list((root / 'submission').glob('*.json')) + list((root / 'reference-submission').glob('*.json'))
        ids = [str(ten.read(p)['job_id']) for p in receipts
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
