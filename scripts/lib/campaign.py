"""Shared, frozen campaign lifecycle. No site-specific recovery scripts."""
import argparse
import contextlib
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

from . import experiments as e

REPO = Path(__file__).resolve().parents[2]
SCHEMA = 'spgemm-campaign-v1'
ACTIONS = ('strong_scaling', 'permutation', 'rectangular')
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(data)
    return value.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as stream:
            stream.write(value if isinstance(value, str) else json.dumps(value, indent=2) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextlib.contextmanager
def lock(path):
    """OS-owned advisory lock, automatically released on timeout/SIGKILL."""
    path.mkdir(parents=True, exist_ok=True)
    with (path / '.campaign.lock').open('a+b') as stream:
        if os.name == 'nt':
            import msvcrt
            if stream.seek(0, os.SEEK_END) == 0:
                stream.write(b'0')
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == 'nt':
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def settings(path, mode, matrix_dir=None):
    value = read(path)
    if set(value) != {'experiment', 'excluded_pairs', 'cases_per_job', 'reference_parallelism', 'slurm'}:
        raise ValueError('Unexpected campaign configuration keys')
    for field in ('cases_per_job', 'reference_parallelism'):
        if type(value[field]) is not int or value[field] < 1:
            raise ValueError(field + ' must be a positive integer')
    if any(not isinstance(pair, list) or len(pair) != 2 or
           any(not isinstance(p, str) or Path(p).is_absolute() or '..' in Path(p).parts for p in pair)
           for pair in value['excluded_pairs']):
        raise ValueError('excluded_pairs must contain relative matrix-path pairs')
    cfg_values = dict(value['experiment'], validate=True if mode == 'full' else 'partial',
                      partial_cache=None, full_cache=None)
    if matrix_dir is not None:
        cfg_values['matrix_dir'] = str(matrix_dir.resolve())
    cfg = e.options(['all'], configuration=cfg_values)
    if cfg.runs != 1 or cfg.campaign_repeats != 1 or cfg.hostfile:
        raise ValueError('Use one MPI launch per case, one campaign, and no hostfile; set repeats/trials for samples')
    slurm = value['slurm']
    if set(slurm) != {'partition', 'time', 'memory', 'exclusive', 'preamble'}:
        raise ValueError('Unexpected Slurm configuration keys')
    for key in ('partition', 'time', 'memory'):
        if not isinstance(slurm[key], str) or not re.fullmatch(r'[A-Za-z0-9_.:+-]+', slurm[key]):
            raise ValueError('Invalid Slurm ' + key)
    if type(slurm['exclusive']) is not bool or not isinstance(slurm['preamble'], list) or any(
            not isinstance(line, str) or '\n' in line or '\r' in line for line in slurm['preamble']):
        raise ValueError('Invalid Slurm exclusive/preamble settings')
    return value, cfg


def case_record(cfg, action, case, variant):
    return dict(action=action, name=case[0], kind=case[1], variant=variant,
                pair=[p.relative_to(cfg.matrix_dir).as_posix() for p in case[2:]])


def case_tuple(cfg, record):
    return (record['name'], record['kind'], *(cfg.matrix_dir / p for p in record['pair']))


def plan(value, cfg):
    excluded = {tuple(pair) for pair in value['excluded_pairs']}
    cases = [case_record(cfg, action, case, variant) for action in ACTIONS for case in e.cases(cfg, action)
             if tuple(p.relative_to(cfg.matrix_dir).as_posix() for p in case[2:]) not in excluded
             for variant in cfg.variants]
    if not cases:
        raise ValueError('The campaign contains no measurement cases')
    references = {}
    # Main products and a small cached probe; small Full pilots stay uncached.
    probe = copy.copy(cfg)
    probe.matrices = cfg.pilot_matrices
    for record in cases + [case_record(cfg, 'strong_scaling', c, cfg.variants[0])
                           for c in e.cases(probe, 'strong_scaling')]:
        key = tuple(record['pair'])
        if key not in references:
            references[key] = dict(record, key='ref-%03d' % len(references))
    pairs = {pair: ref['key'] for pair, ref in references.items()}
    for record in cases:
        record['reference'] = pairs[tuple(record['pair'])]
    jobs = []
    size = value['cases_per_job']
    for nodes in cfg.nodes:
        for offset in range(0, len(cases), size):
            jobs.append(dict(key='n%d-b%03d' % (nodes, offset // size + 1), nodes=nodes,
                             cases=cases[offset:offset + size]))
    return jobs, list(references.values())


def driver(root, mode):
    return root / ('metadata/source/scripts/campaign_%s.py' % mode)


def batch(root, manifest, job, reference=False):
    cfg, site = manifest['experiment'], manifest['slurm']
    nodes = 1 if reference else job['nodes']
    ranks, cpus = (1, 1) if reference else (cfg['ranks_per_node'], cfg['cpus_per_rank'])
    key = job['key']
    lines = ['#!/usr/bin/env bash', '#SBATCH --job-name=spgemm-' + manifest['mode'] + '-' + key,
             '#SBATCH --partition=' + site['partition'], '#SBATCH --nodes=%d' % nodes,
             '#SBATCH --ntasks=%d' % (nodes * ranks), '#SBATCH --ntasks-per-node=%d' % ranks,
             '#SBATCH --cpus-per-task=%d' % cpus, '#SBATCH --mem=' + site['memory'],
             '#SBATCH --time=' + site['time'], '#SBATCH --hint=nomultithread']
    if site['exclusive']:
        lines.append('#SBATCH --exclusive')
    for stream, suffix in (('output', 'out'), ('error', 'err')):
        lines.append('#SBATCH --%s=%s' % (stream, shlex.quote(str(root / 'logs' / key / ('%j.' + suffix)))))
    lines += ['set -euo pipefail',
              'job_dir=' + shlex.quote(str(root / 'logs' / key)) + '/"${SLURM_JOB_ID}"',
              'mkdir -p "$job_dir"', 'exec > >(tee "$job_dir/launch.log") 2>&1']
    # Capture module/setup failures too, before optional machine reports.
    lines += site['preamble']
    lines += ['export OMP_NUM_THREADS=1', 'cd -- ' + shlex.quote(str(root / 'metadata/source'))]
    if not reference:
        lines.append(shlex.join([cfg['mpi_launcher']] + cfg['mpi_args'] +
                     ['-np', str(nodes), '--map-by', 'ppr:1:node', '--bind-to', 'none',
                      'bash', 'scripts/collect_machine.sh']) + ' "$job_dir" ' +
                     shlex.quote(str(root / 'metadata/bin')))
    lines.append(shlex.join(['python3', str(driver(root, manifest['mode'])),
                            'reference' if reference else 'run', '--root', str(root), '--key', key]))
    return '\n'.join(lines) + '\n'


def build(source, destination, jobs, cmake_args):
    subprocess.run(['cmake', '-S', str(source), '-B', str(destination), '-DCMAKE_BUILD_TYPE=Release',
                    '-DBUILD_TESTING=OFF'] + cmake_args, check=True)
    subprocess.run(['cmake', '--build', str(destination), '--parallel', str(jobs)], check=True)


def prepare(mode, args):
    value, cfg = settings(args.config, mode, args.matrix_dir)
    jobs, references = plan(value, cfg)
    root = args.root.resolve()
    cache = (args.cache_dir.resolve() if args.cache_dir else root / 'references')
    if root.exists():
        raise ValueError('Campaign root already exists; use its frozen driver to resume: ' + str(root))
    if args.build_jobs < 1:
        raise ValueError('build-jobs must be positive')
    for protected in (REPO / 'src', REPO / 'scripts', REPO / 'tests', REPO / 'docs', cfg.matrix_dir):
        if root == protected or root in protected.parents or protected in root.parents:
            raise ValueError('Campaign root overlaps source files or matrices')
    if cache == cfg.matrix_dir or cfg.matrix_dir in cache.parents or cache in cfg.matrix_dir.parents:
        raise ValueError('Reference cache must not overlap the matrix directory')
    # Check every unique selected input, including all pilot products, once before freezing.
    all_cases = [r for j in jobs if j['nodes'] == cfg.nodes[0] for r in j['cases']]
    work = [(action, list(dict.fromkeys(case_tuple(cfg, r) for r in all_cases if r['action'] == action)))
            for action in ACTIONS] + [('pilot', e.cases(cfg, 'pilot'))]
    e.input_preflight(cfg, work)
    root.mkdir(parents=True)
    source = root / 'metadata/source'
    for name in ('src', 'scripts'):
        shutil.copytree(REPO / name, source / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('CMakeLists.txt', 'README.md'):
        shutil.copy2(REPO / name, source / name)
    build(source, root / 'metadata/build', args.build_jobs, args.cmake_arg)
    generator = 'prepare_%s_reference' % mode
    for name in list(cfg.variants) + [generator, 'CMakeCache.txt']:
        target = root / 'metadata/bin' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / 'metadata/build' / name, target)
    shutil.copytree(cfg.matrix_dir / 'metadata', root / 'metadata/inputs')
    keys = e.read_json(REPO / 'scripts/experiments.json')
    experiment = {k: str(getattr(cfg, k)) if isinstance(getattr(cfg, k), Path) else getattr(cfg, k) for k in keys}
    experiment.update(build_dir=str(root / 'metadata/bin'), results_dir=str(root),
                      partial_cache=str(cache) if mode == 'partial' else None,
                      full_cache=str(cache) if mode == 'full' else None)
    manifest = dict(schema=SCHEMA, mode=mode, experiment=experiment, jobs=jobs, references=references,
                    slurm=value['slurm'], reference_parallelism=value['reference_parallelism'],
                    excluded_pairs=value['excluded_pairs'], cache=str(cache),
                    created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write(root / 'metadata/campaign.json', value)
    for job, is_reference in [(j, False) for j in jobs] + [(r, True) for r in references]:
        (root / 'logs' / job['key']).mkdir(parents=True)
        write(root / 'metadata/jobs' / (job['key'] + '.sbatch'), batch(root, manifest, job, is_reference))
    manifest['sha256'] = {p.relative_to(root).as_posix(): digest(p) for p in (root / 'metadata').rglob('*')
                          if p.is_file() and 'build' not in p.relative_to(root / 'metadata').parts
                          and '__pycache__' not in p.parts}
    write(root / 'manifest.json', manifest)
    write(root / 'metadata/manifest.sha256', digest(root / 'manifest.json') + '\n')
    print('PREPARED: %d reference jobs; %d experiment jobs; no jobs submitted.' % (len(references), len(jobs)))
    print('Frozen driver: ' + str(driver(root, mode)))


def verify(root, mode):
    if digest(root / 'manifest.json') != (root / 'metadata/manifest.sha256').read_text().strip():
        raise ValueError('Frozen campaign manifest changed')
    manifest = read(root / 'manifest.json')
    if manifest.get('schema') != SCHEMA or manifest.get('mode') != mode:
        raise ValueError('Campaign schema/validation mode mismatch')
    for relative, expected in manifest['sha256'].items():
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts or digest(root / path) != expected:
            raise ValueError('Frozen campaign file changed: ' + relative)
    if Path(__file__).resolve() != root / 'metadata/source/scripts/lib/campaign.py':
        raise ValueError('Use the frozen campaign driver: ' + str(driver(root, mode)))
    return manifest


def config(root, manifest, nodes):
    values = dict(manifest['experiment'], nodes=[nodes], results_dir=str(root / ('n%d' % nodes)))
    cfg = e.options(['all'], configuration=values)
    e.check_root = lambda path: root / 'checks' / path.relative_to(root)
    return cfg


def existing(cfg, action):
    done = set()
    _, warmup, repeats, trials, validation = e.sampling(cfg, action)
    for variant in cfg.variants:
        table = e.output_root(cfg, action) / action / (variant + '.tsv')
        for row in e.read_table(table, e.FIELDS):
            expected = dict(nodes=cfg.nodes[0], ranks=cfg.nodes[0] * cfg.ranks_per_node,
                            ranks_per_node=cfg.ranks_per_node, threads_per_rank=cfg.threads,
                            cpus_per_rank=cfg.cpus_per_rank, topology='physical', omp_schedule=cfg.schedule,
                            omp_chunk=cfg.chunk, warmup=warmup, repeats=repeats, trials=trials,
                            csr_offset_bits=64, benchmark_protocol=e.PROTOCOLS[variant],
                            validation='PARTIAL_PASS' if validation == 'partial' else 'PASS',
                            validation_criterion='row_nnz_exact_v1' if validation == 'partial' else 'componentwise_dot_bound_v1')
            if any(row.get(k) != str(v) for k, v in expected.items()):
                raise ValueError('Incompatible accepted observation: ' + str(table))
            e.validate_measurements(row, variant.startswith('trident_'))
            key = (variant, row['matrix_a'], row['matrix_b'])
            if key in done:
                raise ValueError('Duplicate accepted observation: ' + str(key))
            done.add(key)
    return done


def missing(root, manifest, job):
    cfg = config(root, manifest, job['nodes'])
    present = {a: existing(cfg, a) for a in {r['action'] for r in job['cases']}}
    return [r for r in job['cases'] if (r['variant'], *r['pair']) not in present[r['action']]]


def inputs_unchanged(root, cfg):
    for path in (root / 'metadata/inputs').glob('*.json'):
        if read(path) != read(cfg.matrix_dir / 'metadata' / path.name):
            raise ValueError('Input metadata changed: ' + path.name)


def reference_ready(root, manifest, ref):
    receipt = root / 'reference-ready' / (ref['key'] + '.json')
    if not receipt.exists():
        return False
    value = read(receipt)
    key = value.get('cache_key', '')
    generator = 'metadata/bin/prepare_%s_reference' % manifest['mode']
    suffix = '.sfull' if manifest['mode'] == 'full' else '.spref'
    path = Path(manifest['cache']) / (key + suffix)
    if (not re.fullmatch('[0-9a-f]{64}', key) or value.get('pair') != ref['pair'] or
            value.get('generator_sha256') != manifest['sha256'][generator] or
            not path.is_file() or path.stat().st_size != value.get('bytes') or path.stat().st_size < 72):
        raise ValueError('Invalid/missing reference: ' + ref['key'])
    # Full checksums and input/build identity are checked by C++ on every use.
    return True


def reference(root, manifest, key):
    ref = next(r for r in manifest['references'] if r['key'] == key)
    cfg = config(root, manifest, 1)
    from .allocated_experiment import check_allocation
    allocation = copy.copy(cfg)
    allocation.ranks_per_node, allocation.cpus_per_rank = 1, 1
    check_allocation(allocation, 'slurm', os.environ)
    inputs_unchanged(root, cfg)
    e.input_preflight(cfg, [(ref['action'], [case_tuple(cfg, ref)])])
    generator = root / ('metadata/bin/prepare_%s_reference' % manifest['mode'])
    cmd = [str(generator), '--matrix-a', str(cfg.matrix_dir / ref['pair'][0]),
           '--matrix-b', str(cfg.matrix_dir / ref['pair'][1]), '--%s-cache' % manifest['mode'], manifest['cache']]
    if manifest['mode'] == 'partial':
        cmd.append('--partial')
    with lock(root / 'reference-execution' / key):
        if reference_ready(root, manifest, ref):
            print('REFERENCE ALREADY READY: ' + key, flush=True)
            return
        result = subprocess.run(cmd, env=dict(os.environ, OMP_NUM_THREADS='1'), capture_output=True, text=True)
        print(result.stdout, end='', flush=True)
        print(result.stderr, end='', file=sys.stderr, flush=True)
        result.check_returncode()
        match = re.search(r'\bvalidation_reference_key=([0-9a-f]{64})\b', result.stdout)
        if match is None:
            raise ValueError('Reference generator did not report its cache key')
        cache_key = match[1]
        suffix = '.sfull' if manifest['mode'] == 'full' else '.spref'
        path = Path(manifest['cache']) / (cache_key + suffix)
        write(root / 'reference-ready' / (key + '.json'),
              dict(cache_key=cache_key, pair=ref['pair'], bytes=path.stat().st_size,
                   generator_sha256=digest(generator)))
        reference_ready(root, manifest, ref)
        print('REFERENCE READY: ' + key, flush=True)


def perform(cfg, action, cases, headers):
    with lock(e.output_root(cfg, action)):
        present = existing(cfg, action)
        original_write = e.write_table
        def atomic_table(path, fields, rows, append=False):
            data = e.read_table(path, fields) + list(rows) if append else rows
            original_write(path, fields, data, append=False)
        e.write_table = atomic_table
        try:
            for case in cases:
                pair = [p.relative_to(cfg.matrix_dir).as_posix() for p in case[2:]]
                for variant in cfg.variants:
                    if (variant, *pair) not in present:
                        one = copy.copy(cfg)
                        one.variants = [variant]
                        e.run_experiment(one, action, [case], headers)
        finally:
            e.write_table = original_write


def run(root, manifest, key):
    job = next(j for j in manifest['jobs'] if j['key'] == key)
    cfg = config(root, manifest, job['nodes'])
    # Import from this same frozen package, never from the checkout being edited.
    from .allocated_experiment import check_allocation
    check_allocation(cfg, 'slurm', os.environ)
    if not all(reference_ready(root, manifest, r) for r in manifest['references']):
        raise ValueError('Prepare all references before running measurements')
    with lock(root / 'execution' / ('n%d' % job['nodes'])):
        inputs_unchanged(root, cfg)
        todo = missing(root, manifest, job)
        if not todo:
            print('ALREADY COMPLETE: ' + key)
            return True
        selected = copy.copy(cfg)
        selected.variants = list(dict.fromkeys(r['variant'] for r in todo))
        pilot = copy.copy(selected)
        pilot.results_dir = cfg.results_dir / 'pilot-full'
        probe = copy.copy(selected)
        probe.matrices = cfg.pilot_matrices
        probe.results_dir = root / 'checks' / ('n%d/cached-probe' % job['nodes'])
        pilots = e.cases(pilot, 'pilot')
        probes = e.cases(probe, 'strong_scaling')
        work = [(r['action'], [case_tuple(cfg, r)]) for r in todo]
        headers = e.input_preflight(cfg, work + [('pilot', pilots), ('strong_scaling', probes)])
        perform(pilot, 'pilot', pilots, headers)
        perform(probe, 'strong_scaling', probes, headers)
        state = dict(job_id=os.environ['SLURM_JOB_ID'], complete=False, failures=[])
        write(root / 'state' / (key + '.json'), state)
        for record in todo:
            one = copy.copy(cfg)
            one.variants = [record['variant']]
            state['current'] = record
            write(root / 'state' / (key + '.json'), state)
            try:
                perform(one, record['action'], [case_tuple(one, record)], headers)
            except RuntimeError as error:
                state['failures'].append(dict(case=record, error=str(error)))
                print('CASE FAILED; continuing: ' + str(error), flush=True)
            write(root / 'state' / (key + '.json'), state)
        state.update(current=None, complete=not missing(root, manifest, job))
        write(root / 'state' / (key + '.json'), state)
        print('%s: %s' % (key, 'COMPLETE' if state['complete'] else 'INCOMPLETE'), flush=True)
        return state['complete']


def states(ids):
    if not ids:
        return {}
    result = subprocess.run(['sacct', '-X', '-n', '-P', '-j', ','.join(ids), '--format=JobIDRaw,State'],
                            check=True, capture_output=True, text=True)
    parsed = {}
    for line in result.stdout.splitlines():
        fields = line.split('|')
        if len(fields) >= 2 and fields[0] in ids:
            parsed[fields[0]] = fields[1].split()[0].rstrip('+')
    if set(parsed) != set(ids):
        raise ValueError('Accounting incomplete; refusing to duplicate a possibly active job')
    return parsed


def submit(root, manifest, references=False, retry=False):
    jobs = manifest['references'] if references else manifest['jobs']
    if not references and not all(reference_ready(root, manifest, r) for r in manifest['references']):
        raise ValueError('References are not ready: submit-references, check status, then submit')
    with lock(root / 'submission'):
        records = {}
        for job in jobs:
            path = root / 'submission' / (job['key'] + '.json')
            if path.exists():
                value = read(path)
                if not str(value.get('job_id', '')).isdigit():
                    raise ValueError('Ambiguous submission; inspect accounting and ' + str(path))
                records[job['key']] = value
        status = states([str(r['job_id']) for r in records.values()])
        previous = {}
        for index, job in enumerate(jobs):
            record = records.get(job['key'])
            pool = index % manifest['reference_parallelism'] if references else job['nodes']
            if record and status[str(record['job_id'])] not in TERMINAL:
                previous[pool] = str(record['job_id'])
        for index, job in enumerate(jobs):
            key = job['key']
            done = reference_ready(root, manifest, job) if references else not missing(root, manifest, job)
            record = records.get(key)
            if record and status[str(record['job_id'])] not in TERMINAL:
                print('%s: JOB=%s %s' % (key, record['job_id'], status[str(record['job_id'])]))
                continue
            if done:
                continue
            if record and not retry:
                print(key + ': incomplete; use --retry-failed to resume missing observations')
                continue
            pool = index % manifest['reference_parallelism'] if references else job['nodes']
            script = root / 'metadata/jobs' / (key + '.sbatch')
            command = ['sbatch', '--parsable']
            if pool in previous:
                command.append('--dependency=afterany:' + previous[pool])
            command.append(str(script))
            subprocess.run(['sbatch', '--test-only', str(script)], check=True)
            receipt = root / 'submission' / (key + '.json')
            history = (record.get('history', []) + [dict(job_id=record['job_id'], state=status[str(record['job_id'])])]
                       if record else [])
            write(receipt, dict(state='submitting', command=command, history=history))
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            job_id = result.stdout.strip().split(';')[0]
            if not job_id.isdigit():
                raise ValueError('Ambiguous sbatch reply; inspect ' + str(receipt))
            write(receipt, dict(job_id=job_id, command=command, history=history))
            previous[pool] = job_id
            print('%s JOB=%s' % (key, job_id), flush=True)


def status(root, manifest):
    ready = sum(reference_ready(root, manifest, r) for r in manifest['references'])
    print('References: %d/%d ready' % (ready, len(manifest['references'])))
    for nodes in manifest['experiment']['nodes']:
        jobs = [j for j in manifest['jobs'] if j['nodes'] == nodes]
        remaining = sum(len(missing(root, manifest, j)) for j in jobs)
        total = sum(len(j['cases']) for j in jobs)
        print('NODES=%d: %d/%d accepted | %d samples/case' %
              (nodes, total - remaining, total, manifest['experiment']['repeats'] * manifest['experiment']['trials']))
        for job in jobs:
            path = root / 'state' / (job['key'] + '.json')
            if path.exists():
                current = read(path).get('current')
                if current:
                    print('  Last started in %s: %s | %s | %s' %
                          (job['key'], current['action'], current['variant'], ' x '.join(current['pair'])))
    ids = [str(read(p)['job_id']) for p in (root / 'submission').glob('*.json') if read(p).get('job_id')]
    if ids:
        subprocess.run(['sacct', '-X', '-j', ','.join(ids),
                        '--format=JobID,JobName%40,State,Elapsed,ExitCode'], check=True)


def export(root, manifest):
    for nodes in manifest['experiment']['nodes']:
        cfg = config(root, manifest, nodes)
        if any(missing(root, manifest, j) for j in manifest['jobs'] if j['nodes'] == nodes):
            raise ValueError('Campaign incomplete at %d nodes; inspect status before exporting' % nodes)
        with lock(cfg.results_dir):
            for action in e.ANALYSES:
                e.analyze(cfg, action)


def cli(mode):
    parser = argparse.ArgumentParser(description='Cached %s validation campaign' % mode)
    parser.add_argument('action', choices=('plan', 'prepare', 'submit-references', 'submit', 'reference', 'run', 'status', 'export'))
    parser.add_argument('--config', type=Path, default=REPO / 'scripts/campaign.json')
    parser.add_argument('--root', type=Path)
    parser.add_argument('--matrix-dir', type=Path)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--build-jobs', type=int, default=8)
    parser.add_argument('--cmake-arg', action='append', default=[])
    parser.add_argument('--key')
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    try:
        if args.action == 'plan':
            value, cfg = settings(args.config, mode, args.matrix_dir)
            jobs, refs = plan(value, cfg)
            print('%s: %d node configurations, %d cases/node, %d samples/case, %d jobs + %d references' %
                  (mode.upper(), len(cfg.nodes), sum(len(j['cases']) for j in jobs if j['nodes'] == cfg.nodes[0]),
                   cfg.repeats * cfg.trials, len(jobs), len(refs)))
            return
        if args.root is None:
            parser.error('this action requires --root')
        if args.retry_failed and args.action not in ('submit', 'submit-references'):
            parser.error('--retry-failed is only valid for submission')
        if args.action == 'prepare':
            prepare(mode, args)
            return
        root = args.root.resolve()
        manifest = verify(root, mode)
        if args.action.startswith('submit'):
            submit(root, manifest, args.action == 'submit-references', args.retry_failed)
        elif args.action in ('run', 'reference'):
            if not args.key:
                parser.error('run/reference requires --key')
            if args.action == 'run':
                if not run(root, manifest, args.key):
                    sys.exit(1)
            else:
                reference(root, manifest, args.key)
        else:
            globals()[args.action](root, manifest)
    except (OSError, ValueError, TypeError, RuntimeError, KeyError, StopIteration, subprocess.SubprocessError) as error:
        print('ERROR: ' + str(error), file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
