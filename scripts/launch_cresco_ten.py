#!/usr/bin/env python3
"""CRESCO tesi-005: 10 samples, 8 node counts, 80 independent-success blocks.

Freeze the existing tesi-004 executables and inputs; do not rebuild or download.
Python 3.8+. Run restart on the cluster to prepare, retire tesi-004, and submit.
"""
import argparse
import contextlib
import copy
import csv
import datetime
import getpass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

NODES = (1, 4, 9, 16, 36, 64, 100, 121)
DEFAULT_REPO = Path('/lpor1/store_0/usr/malchiod/tirocinio-spgemm/SpGEMM-hpc-thesis')
OLD_OUTPUT = 'results/tesi-004/partial20-main'
OUTPUT = 'results/tesi-005/partial10-main'
CHECKS = 'test-results/tesi-005/partial10-main'
OLD_CHECKS = 'test-results/tesi-004/partial20-main'
MAIN = ('strong_scaling', 'permutation', 'rectangular')
VARIANTS = ('spgemm_two_sided', 'spgemm_one_sided_get', 'spgemm_one_sided_put',
            'trident_two_sided', 'trident_hybrid', 'trident_get',
            'trident_get_pipeline', 'trident_put')
GROUPS = (
    ('strong_scaling', ('mouse_gene', 'dielFilterV3real', 'isolates_subgraph4')),
    ('strong_scaling', ('cage15', 'archaea', 'isolates_subgraph5')),
    ('strong_scaling', ('HV15R', 'eukarya', 'uniparc')),
    ('permutation', ('HV15R', 'dielFilterV3real', 'cage15', 'archaea')),
    ('rectangular', ('mouse_gene', 'cage15', 'HV15R', 'dielFilterV3real', 'archaea',
                     'eukarya', 'isolates_subgraph4', 'isolates_subgraph5', 'uniparc', 'reddit')),
)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.new')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(value if isinstance(value, str) else json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def verify(root):
    manifest = read(root / 'manifest.json')
    if not manifest.get('sha256'):
        raise ValueError('Snapshot incompleto: ' + str(root))
    for name, expected in manifest['sha256'].items():
        path = root / name
        if Path(name).is_absolute() or '..' in Path(name).parts or digest(path) != expected:
            raise ValueError('Snapshot modificato: ' + name)
    return manifest


@contextlib.contextmanager
def file_lock(root):
    """Kernel-owned lock: timeout/SIGKILL releases it; never delete a stale mkdir lock."""
    root.mkdir(parents=True, exist_ok=True)
    stream = (root / '.campaign.lock').open('a+b')
    acquired = False
    try:
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
        acquired = True
        yield
    except (BlockingIOError, PermissionError) as error:
        if not acquired:
            raise RuntimeError('Scrittura concorrente: ' + str(root)) from error
        raise
    finally:
        if acquired:
            if os.name == 'nt':
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def base_module(root):
    path = root / 'metadata/base_launcher.py'
    spec = importlib.util.spec_from_file_location('_tesi004_frozen_base', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def runtime(root):
    base = base_module(root)
    e = base.runner(root)
    if not getattr(e, '_tesi005_atomic', False):
        original = e.write_table

        def atomic_table(path, fields, rows, append=False):
            # Each completed observation replaces a whole valid TSV atomically.
            # A killed writer leaves the previous valid table, never half a row.
            data = e.read_table(path, fields) + list(rows) if append else rows
            original(path, fields, data, append=False)

        e.write_table = atomic_table
        e.result_lock = file_lock
        e._tesi005_atomic = True
    return base, e


def plan():
    jobs = []
    for nodes in NODES:
        for block in range(1, 11):
            action, matrices = GROUPS[(block - 1) // 2]
            variants = VARIANTS[:4] if block % 2 else VARIANTS[4:]
            key = 'n%d-b%02d' % (nodes, block)
            jobs.append(dict(key=key, nodes=nodes, block=block, action=action,
                             matrices=list(matrices), variants=list(variants),
                             expected=4 * len(matrices), script='metadata/jobs/' + key + '.sbatch'))
    return jobs


def batch(root, job):
    q = shlex.quote
    logs = root / 'logs' / job['key']
    return '\n'.join([
        '#!/usr/bin/env bash', '#SBATCH --job-name=tesi005-' + job['key'],
        '#SBATCH --partition=cresco8_cpu', '#SBATCH --nodes=%d' % job['nodes'],
        '#SBATCH --ntasks=%d' % (2 * job['nodes']), '#SBATCH --ntasks-per-node=2',
        '#SBATCH --cpus-per-task=64', '#SBATCH --mem=450000M', '#SBATCH --time=24:00:00',
        '#SBATCH --exclusive', '#SBATCH --hint=nomultithread',
        '#SBATCH --output=' + q(str(logs / '%j.out')),
        '#SBATCH --error=' + q(str(logs / '%j.err')),
        'set -euo pipefail', 'source /usr/share/Modules/init/bash',
        'module load intel/oneapi mpi_flavour/openmpi_intel-4.1.7',
        'cd -- ' + q(str(root / 'metadata/runtime')),
        'job_dir=' + q(str(logs)) + '/"${SLURM_JOB_ID}"',
        'mkdir -p "$job_dir"', 'exec > >(tee "$job_dir/launch.log") 2>&1',
        'mpirun --mca btl_vader_single_copy_mechanism none -np "$SLURM_JOB_NUM_NODES" '
        '--map-by ppr:1:node --bind-to none bash scripts/collect_machine.sh "$job_dir" '
        + q(str(root / 'metadata/bin')),
        shlex.join(['python3', str(root / 'metadata/launcher.py'), 'run', '--root', str(root),
                    '--nodes', str(job['nodes']), '--block', str(job['block'])]), ''])


def prepare(repo, root):
    if root.exists():
        manifest = verify(root)
        if manifest.get('schema') != 'tesi005-partial10-v1' or manifest['jobs'] != plan():
            raise ValueError('Campagna esistente incompatibile')
        print('Preparazione gia completa: ' + str(root))
        return
    source = repo / OLD_OUTPUT
    old = verify(source)
    if old['samples_per_launch'] != 20 or old['validation'] != 'partial':
        raise ValueError('La sorgente deve essere la campagna tesi-004 partial20')
    cfg = read(source / 'metadata/n1.json')
    required = dict(nodes=[1], variants=list(VARIANTS), ranks_per_node=2, threads=63,
                    cpus_per_rank=64, repeats=20, trials=1, runs=1, warmup=0,
                    validate='partial', schedule='guided', chunk=64, seed=42)
    if any(cfg.get(k) != v for k, v in required.items()):
        raise ValueError('Configurazione sorgente diversa da quella concordata')
    if Path(old['checks']).resolve() != (repo / OLD_CHECKS).resolve():
        raise ValueError('Directory dei controlli tesi-004 inattesa')
    for name in ('bin', 'runtime', 'matrices'):
        shutil.copytree(source / 'metadata' / name, root / 'metadata' / name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('source.tar',):
        shutil.copy2(source / 'metadata' / name, root / 'metadata' / name)
    shutil.copy2(source / 'metadata/launcher.py', root / 'metadata/base_launcher.py')
    shutil.copy2(Path(__file__), root / 'metadata/launcher.py')
    shutil.copy2(source / 'manifest.json', root / 'metadata/previous-manifest.json')
    receipts = {}
    for job in old['jobs']:
        record = read(source / 'submission' / (job['key'] + '.json'))
        if not str(record.get('job_id', '')).isdigit():
            raise ValueError('Job precedente con invio incerto: ' + job['key'])
        receipts[record['job_id']] = 'tesi004-' + job['key']
    write(root / 'metadata/previous-jobs.json', receipts)
    manifest = dict(schema='tesi005-partial10-v1', commit=old['commit'],
                    source_campaign=str(source), checks=str(repo / CHECKS), nodes=list(NODES),
                    jobs=plan(), samples_per_launch=10, independent_launches=1,
                    validation='partial', excluded_pairs=[['reddit.mtx', 'reddit.mtx']], sha256={})
    write(root / 'manifest.json', manifest)
    base, e = runtime(root)
    for nodes in NODES:
        config = dict(cfg, nodes=[nodes], repeats=10, campaign_repeats=1, timeout=0,
                      build_dir=str(root / 'metadata/bin'), results_dir=str(root / ('n%d' % nodes)))
        write(root / ('metadata/n%d.json' % nodes), config)
        parsed = e.options(['all', '--config', str(root / ('metadata/n%d.json' % nodes))])
        covered = set()
        for job in (j for j in manifest['jobs'] if j['nodes'] == nodes):
            cases = [c for c in e.cases(parsed, job['action']) if c[0] in job['matrices']]
            if len(cases) != len(job['matrices']):
                raise ValueError('Casi mancanti nel piano: ' + job['key'])
            for case in cases:
                for path in case[2:]:
                    if not path.is_file():
                        raise ValueError('Matrice gia preparata mancante: ' + str(path))
                for variant in job['variants']:
                    key = (job['action'], str(case[2]), str(case[3]), variant)
                    if key in covered:
                        raise ValueError('Prova duplicata nel piano')
                    covered.add(key)
            (root / 'logs' / job['key']).mkdir(parents=True)
            write(root / job['script'], batch(root, job))
        if len(covered) != 184 or any(Path(a).name == Path(b).name == 'reddit.mtx'
                                      for _, a, b, _ in covered):
            raise ValueError('Copertura della campagna inattesa')
    manifest['sha256'] = {p.relative_to(root).as_posix(): digest(p)
                          for p in (root / 'metadata').rglob('*') if p.is_file()
                          and '__pycache__' not in p.parts}
    write(root / 'manifest.json', manifest)
    print('PREPARATO: 80 job; 8 configurazioni; 184 prove x 10 campioni per configurazione.')


def preflight(root):
    manifest = verify(root)
    check_references_ready(root, manifest)
    if manifest['jobs'] != plan():
        raise ValueError('Piano modificato')
    for nodes in NODES:
        job = next(j for j in manifest['jobs'] if j['nodes'] == nodes)
        subprocess.run(['sbatch', '--test-only', str(root / job['script'])], check=True)


def check_references_ready(root, manifest):
    """Read readiness receipts only; C++ reads/checks the reference after its timers."""
    if manifest.get('schema') != 'partial10-cached-v1':
        return
    for pair in manifest['reference_pairs']:
        receipt = root / 'reference-ready' / ('%02d.json' % pair['index'])
        if not receipt.is_file():
            raise ValueError('Riferimento non preparato: ' + str(receipt))
        record = read(receipt)
        key = record.get('key', '')
        if (record.get('pair') != pair or re.fullmatch('[0-9a-f]{64}', key) is None or
                record.get('generator_sha256') != manifest['sha256']['metadata/bin/prepare_partial_reference'] or
                not (Path(manifest['partial_cache']) / (key + '.spref')).is_file()):
            raise ValueError('Ricevuta riferimento incompatibile: ' + str(receipt))


def submit(root, checked=False):
    manifest = verify(root)
    check_references_ready(root, manifest)
    with file_lock(root / 'submission'):
        if not checked:
            preflight(root)
        for job in manifest['jobs']:
            receipt = root / 'submission' / (job['key'] + '.json')
            if receipt.exists():
                if not str(read(receipt).get('job_id', '')).isdigit():
                    raise ValueError('Invio precedente incerto: ' + str(receipt))
                continue
            command = ['sbatch', '--parsable']
            if job['block'] > 1:
                prior = read(root / 'submission' / ('n%d-b%02d.json' % (job['nodes'], job['block']-1)))
                command += ['--dependency=afterany:' + prior['job_id']]
            command += [str(root / job['script'])]
            with receipt.open('x', encoding='utf-8') as stream:
                json.dump(dict(state='submitting', command=command), stream)
            result = subprocess.run(command, text=True, capture_output=True, check=True)
            job_id = result.stdout.strip().split(';', 1)[0]
            if not job_id.isdigit():
                raise ValueError('Risposta sbatch inattesa: ' + result.stdout)
            write(receipt, dict(job_id=job_id, command=command, response=result.stdout))
            print(job['key'] + ' JOB=' + job_id, flush=True)


def perform(base, e, cfg, action, cases, continue_errors=False):
    failures = []
    with file_lock(e.output_root(cfg, action)):
        done = base.existing(e, cfg, action)
        remaining = [(c, v) for c in cases for v in cfg.variants
                     if (v, c[2].relative_to(cfg.matrix_dir).as_posix(),
                         c[3].relative_to(cfg.matrix_dir).as_posix()) not in done]
        if remaining:
            headers = e.input_preflight(cfg, [(action, list(dict.fromkeys(c for c, _ in remaining)))])
            for case, variant in remaining:
                one = copy.copy(cfg)
                one.variants = [variant]
                try:
                    e.run_experiment(one, action, [case], headers)
                except RuntimeError as error:
                    if not continue_errors:
                        raise
                    failures.append(dict(variant=variant, matrix_a=str(case[2]),
                                         matrix_b=str(case[3]), error=str(error)))
                    print('PROVA FALLITA; proseguo con la successiva: ' + str(error), flush=True)
    return failures


def run(root, nodes, block):
    manifest = verify(root)
    check_references_ready(root, manifest)
    job = next((j for j in manifest['jobs'] if j['nodes'] == nodes and j['block'] == block), None)
    if job is None:
        raise ValueError('Configurazione/blocco non ammessi')
    base, e = runtime(root)
    from allocated_experiment import check_allocation
    cfg = e.options(['all', '--config', str(root / ('metadata/n%d.json' % nodes))])
    check_allocation(cfg, 'slurm', os.environ)
    for frozen in (root / 'metadata/matrices').glob('*.json'):
        if read(frozen) != read(cfg.matrix_dir / 'metadata' / frozen.name):
            raise ValueError('Metadati delle matrici cambiati: ' + frozen.name)
    # Every block requires actual pilot PASS rows, not a predecessor's exit status.
    # Completed checks are skipped; a failed/timed-out pilot can be retried safely.
    pilot = copy.copy(cfg)
    pilot.results_dir = Path(manifest['checks']) / ('n%d/cage8-full' % nodes)
    perform(base, e, pilot, 'pilot', e.cases(pilot, 'pilot'))
    probe = copy.copy(cfg)
    probe.matrices = ['cage8']
    probe.results_dir = Path(manifest['checks']) / ('n%d/cage8-partial10' % nodes)
    perform(base, e, probe, 'strong_scaling', e.cases(probe, 'strong_scaling'))
    selected = [c for c in e.cases(cfg, job['action']) if c[0] in job['matrices']]
    one = copy.copy(cfg)
    one.variants = job['variants']
    failures = perform(base, e, one, job['action'], selected, continue_errors=True)
    write(root / 'state' / (job['key'] + '.finished.json'),
          dict(complete=not failures, job_id=os.environ['SLURM_JOB_ID'],
               expected=job['expected'], failures=failures))
    if failures:
        raise RuntimeError('%d prove fallite nel blocco; dati completi conservati' % len(failures))
    print('BLOCCO COMPLETATO: ' + job['key'], flush=True)
    counts = {action: len(base.existing(e, cfg, action)) for action in MAIN}
    if counts == dict(strong_scaling=72, permutation=32, rectangular=80):
        with file_lock(cfg.results_dir):
            for analysis in e.ANALYSES:
                e.analyze(cfg, analysis)
        write(root / 'state' / ('n%d.complete.json' % nodes),
              dict(complete=True, counts=counts, samples_per_launch=10))
        print('NODI=%d: COMPLETO 184/184, ogni prova contiene 10 campioni' % nodes, flush=True)


def status(root):
    manifest = verify(root)
    base, e = runtime(root)
    missing = []
    for nodes in NODES:
        cfg = e.options(['all', '--config', str(root / ('metadata/n%d.json' % nodes))])
        present = {a: base.existing(e, cfg, a) for a in MAIN}
        expected = set()
        complete_blocks = 0
        for job in (j for j in manifest['jobs'] if j['nodes'] == nodes):
            complete = True
            for case in e.cases(cfg, job['action']):
                if case[0] not in job['matrices']:
                    continue
                for variant in job['variants']:
                    key = (variant, case[2].relative_to(cfg.matrix_dir).as_posix(),
                           case[3].relative_to(cfg.matrix_dir).as_posix())
                    expected.add((job['action'],) + key)
                    if key not in present[job['action']]:
                        complete = False
                        missing.append(dict(nodes=nodes, block=job['block'], action=job['action'],
                                            variant=variant, matrix_a=key[1], matrix_b=key[2]))
            complete_blocks += complete
        actual = {(a,) + key for a, values in present.items() for key in values}
        if actual - expected:
            raise ValueError('Risultati fuori dal piano: ' + str(actual - expected))
        print('NODI=%d: prove %d/184 | blocchi %d/10 | 10 campioni/prova' %
              (nodes, len(actual), complete_blocks))
        logs = list((root / 'logs').glob('n%d-b*/*/launch.log' % nodes))
        if logs:
            lines = max(logs, key=lambda p: p.stat().st_mtime).read_text(errors='replace').splitlines()
            print('  ' + next((line for line in reversed(lines)
                              if line.startswith(MAIN + ('PROVA FALLITA', 'BLOCCO'))), 'Avvio'))
    ids = [str(read(p)['job_id']) for p in (root / 'submission').glob('*.json') if 'job_id' in read(p)]
    if ids:
        subprocess.run(['sacct', '-X', '-j', ','.join(ids),
                        '--format=JobID,JobName%25,State%22,Elapsed,ExitCode'], check=True)
    return missing


def below(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def old_queue(root, source):
    known = read(root / 'metadata/previous-jobs.json')
    result = subprocess.run(['squeue', '--array', '-h', '-u', getpass.getuser(), '-o', '%i|%j|%T|%Z'],
                            text=True, capture_output=True, check=True)
    active = []
    for line in result.stdout.splitlines():
        fields = [s.strip() for s in line.split('|')]
        if len(fields) != 4:
            raise ValueError('Risposta squeue inattesa: ' + line)
        jid, name, state, directory = fields
        if jid in known:
            if name != known[jid]:
                raise ValueError('Identita job precedente non corrispondente: ' + jid)
            active.append((jid, state))
        elif name.startswith('tesi004-') or below(Path(directory).resolve(), source):
            raise ValueError('Job tesi-004 non registrato; nessuna cancellazione: ' + line)
    return active


def validate_cleanup_target(target, repo, matrices):
    if target not in (repo / OLD_OUTPUT, repo / OLD_CHECKS):
        raise ValueError('Destinazione di pulizia non autorizzata: ' + str(target))
    if target.is_symlink() or target.resolve() != target or not below(target, repo):
        raise ValueError('Destinazione simbolica o fuori dal checkout: ' + str(target))
    if below(matrices, target) or below(target, matrices):
        raise ValueError('La pulizia coinvolge le matrici')
    mountinfo = Path('/proc/self/mountinfo')
    if mountinfo.exists():
        for line in mountinfo.read_text().splitlines():
            mount = Path(re.sub(r'\\([0-7]{3})', lambda m: chr(int(m.group(1), 8)), line.split()[4]))
            if below(mount, target):
                raise ValueError('Mount point nella destinazione; pulizia rifiutata')


def retire_old(repo, root):
    manifest = verify(root)
    source = repo / OLD_OUTPUT
    if Path(manifest['source_campaign']) != source:
        raise ValueError('Campagna sorgente inattesa')
    matrices = Path(read(root / 'metadata/n1.json')['matrix_dir']).resolve()
    targets = (source, repo / OLD_CHECKS)
    for target in targets:
        validate_cleanup_target(target, repo, matrices)
    if source.exists() and digest(source / 'manifest.json') != digest(root / 'metadata/previous-manifest.json'):
        raise ValueError('Il manifest tesi-004 e cambiato dalla preparazione')
    active = old_queue(root, source)
    if any(state != 'PENDING' for _, state in active):
        raise ValueError('Un job tesi-004 e ancora attivo; pulizia rifiutata: ' + str(active))
    if active:
        subprocess.run(['scancel', '--state=PENDING'] + [jid for jid, _ in active], check=True)
        for attempt in range(10):
            active = old_queue(root, source)
            if not active:
                break
            time.sleep(1)
        if active:
            raise ValueError('Job ancora presenti; nessun risultato eliminato')
    ids = list(read(root / 'metadata/previous-jobs.json'))
    accounting = subprocess.run(['sacct', '-P', '-j', ','.join(ids),
                                 '--format=JobID,JobName,State,Elapsed,ExitCode,MaxRSS'],
                                text=True, capture_output=True, check=True)
    write(root / 'retired/accounting-tesi004.psv', accounting.stdout)
    for target in targets:
        if target.exists():
            validate_cleanup_target(target, repo, matrices)
            if old_queue(root, source):
                raise ValueError('Un job tesi-004 e ricomparso; pulizia interrotta')
            print('Elimino i risultati precedenti: ' + str(target), flush=True)
            shutil.rmtree(target)
    write(root / 'retired/completed.json', dict(paths=[str(p) for p in targets],
          timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'submit', 'run', 'status', 'audit', 'restart'))
    parser.add_argument('--repo', type=Path, default=DEFAULT_REPO)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--nodes', type=int)
    parser.add_argument('--block', type=int)
    args = parser.parse_args()
    repo = args.repo.resolve()
    root = (args.root or repo / OUTPUT).resolve()
    if args.action == 'restart':
        if sys.platform != 'linux' or getpass.getuser() != 'malchiod' or root != repo / OUTPUT:
            raise ValueError('restart va eseguito sul cluster da malchiod, nella destinazione prevista')
        with file_lock(root.parent / '.setup'):
            prepare(repo, root)
            preflight(root)
            retire_old(repo, root)
            submit(root, checked=True)
    elif args.action == 'prepare':
        with file_lock(root.parent / '.setup'):
            prepare(repo, root)
    elif args.action == 'submit':
        submit(root)
    elif args.action == 'run':
        run(root, args.nodes, args.block)
    else:
        missing = status(root)
        if args.action == 'audit':
            write(root / 'audit/missing.json', missing)
            print('PROVE MANCANTI: %d; elenco in %s' % (len(missing), root / 'audit/missing.json'))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        print('ERRORE: ' + str(error), file=sys.stderr)
        sys.exit(1)
