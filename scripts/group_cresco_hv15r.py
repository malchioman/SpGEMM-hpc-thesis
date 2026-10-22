#!/usr/bin/env python3
"""Run finalized HV15R recovery cases sequentially in one Slurm allocation per node count.

Adds a frozen orchestration layer; never edits the existing recovery snapshot,
executables, references, scientific configuration or successful observations.
"""
import argparse
import importlib.util
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

SCHEMA = 'hv15r-grouped-v1'


def load(root):
    scripts = root / 'metadata/source/scripts'
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location('_grouped_frozen_recovery', scripts / 'recover_cresco_hv15r.py')
    recovery = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recovery)
    for module in (recovery.ten, recovery.cached):
        if Path(module.__file__).resolve().parent != scripts.resolve():
            raise ValueError('Recovery dependencies must come from the frozen snapshot')
    return recovery


def ready(root):
    recovery = load(root)
    manifest, source = recovery.verify(root)
    if manifest.get('phase', 'final') != 'final':
        raise ValueError('Recovery plan is preliminary: finalize after the original campaign ends')
    recovery.require_passed_probes(root, source, manifest)
    recovery.require_references(root, manifest)
    return recovery, manifest, source


def batch(recovery, root, nodes):
    key = 'grouped-n%d' % nodes
    script = recovery.batch(root, dict(key=key, nodes=nodes))
    original = shlex.join(['python3', str(root / 'metadata/source/scripts/recover_cresco_hv15r.py'),
                           'run', '--root', str(root), '--key', key])
    replacement = shlex.join(['python3', str(root / 'grouped/metadata/driver.py'),
                              'run', '--root', str(root), '--nodes', str(nodes)])
    if script.count(original) != 1:
        raise ValueError('Unexpected frozen batch entrypoint')
    return script.replace(original, replacement)


def verify(root):
    recovery, original, source = ready(root)
    grouped = root / 'grouped'
    manifest = recovery.ten.verify(grouped)
    if (manifest.get('schema') != SCHEMA or
            manifest.get('recovery_manifest_sha256') != recovery.ten.digest(root / 'manifest.json')):
        raise ValueError('Grouped plan does not match the finalized recovery')
    jobs = {j['key']: j for j in original['jobs']}
    seen = set()
    nodes = set()
    for group in manifest['groups']:
        if group['nodes'] in nodes or group['key'] != 'n%d' % group['nodes'] or not group['cases']:
            raise ValueError('Invalid grouped allocation')
        nodes.add(group['nodes'])
        for key in group['cases']:
            if key in seen or key not in jobs or jobs[key]['nodes'] != group['nodes']:
                raise ValueError('Invalid/duplicate grouped case: ' + key)
            seen.add(key)
    return recovery, manifest, source, jobs


def prepare(root):
    recovery, original, source = ready(root)
    ten = recovery.ten
    grouped = root / 'grouped'
    with ten.file_lock(root / 'submission'):
        if (grouped / 'manifest.json').exists():
            verify(root)
            print('Grouped plan already prepared; nothing changed.')
            return
        if grouped.exists():
            raise ValueError('Incomplete grouped preparation; inspect ' + str(grouped))
        jobs = [j for j in original['jobs'] if not recovery.complete(root, source, j)]
        if not jobs:
            print('All recovery cases are already accepted; nothing prepared.')
            return
        (grouped / 'metadata').mkdir(parents=True)
        shutil.copy2(Path(__file__), grouped / 'metadata/driver.py')
        groups = []
        for nodes in sorted({j['nodes'] for j in jobs}):
            key = 'n%d' % nodes
            cases = [j['key'] for j in jobs if j['nodes'] == nodes]
            script = 'metadata/jobs/' + key + '.sbatch'
            (root / 'logs' / ('grouped-' + key)).mkdir(parents=True, exist_ok=True)
            ten.write(grouped / script, batch(recovery, root, nodes))
            groups.append(dict(key=key, nodes=nodes, cases=cases, script=script))
        manifest = dict(schema=SCHEMA, recovery_manifest_sha256=ten.digest(root / 'manifest.json'),
                        groups=groups, sha256={p.relative_to(grouped).as_posix(): ten.digest(p)
                        for p in (grouped / 'metadata').rglob('*') if p.is_file()})
        ten.write(grouped / 'manifest.json', manifest)
        for group in groups:
            print('%s: %d sequential cases, 10 samples each, 24h allocation' % (group['key'], len(group['cases'])))
        print('PREPARED %d grouped jobs; nothing submitted or rebuilt.' % len(groups))


def submit(root):
    recovery, manifest, source, jobs = verify(root)
    ten = recovery.ten
    grouped = root / 'grouped'
    with ten.file_lock(root / 'submission'):
        for group in manifest['groups']:
            receipt = grouped / 'submission' / (group['key'] + '.json')
            if receipt.exists():
                value = ten.read(receipt)
                if not str(value.get('job_id', '')).isdigit():
                    raise ValueError('Ambiguous group submission; inspect accounting and ' + str(receipt))
                print(group['key'] + ' already submitted: JOB=' + str(value['job_id']))
                continue
            cases = [key for key in group['cases'] if not recovery.complete(root, source, jobs[key])]
            if not cases:
                continue
            for key in cases:
                previous = root / 'submission' / (key + '.json')
                if previous.exists():
                    raise ValueError('Case already submitted or ambiguous; inspect ' + str(previous))
            command = ['sbatch', '--parsable', str(grouped / group['script'])]
            subprocess.run(['sbatch', '--test-only', str(grouped / group['script'])], check=True)
            ten.write(receipt, dict(state='submitting', cases=cases, command=command))
            # The original single-case submitter uses the same lock and receipts.
            # Reservations prevent it from submitting concurrent copies of these cases.
            for key in cases:
                ten.write(root / 'submission' / (key + '.json'), dict(state='submitting', group=group['key']))
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            job_id = result.stdout.strip().split(';')[0]
            if not job_id.isdigit():
                raise ValueError('Ambiguous sbatch response; inspect ' + str(receipt))
            for key in cases:
                ten.write(root / 'submission' / (key + '.json'), dict(job_id=job_id, group=group['key'], command=command))
            ten.write(receipt, dict(job_id=job_id, cases=cases, command=command))
            print('%s JOB=%s CASES=%d' % (group['key'], job_id, len(cases)), flush=True)


def run(root, nodes):
    recovery, manifest, source, jobs = verify(root)
    ten = recovery.ten
    grouped = root / 'grouped'
    group = next(g for g in manifest['groups'] if g['nodes'] == nodes)
    with ten.file_lock(grouped / 'execution' / group['key']):
        receipt_path = grouped / 'submission' / (group['key'] + '.json')
        # Slurm may start immediately, before the submitting process publishes its
        # returned job ID and all per-case reservations on the shared filesystem.
        for _ in range(60):
            receipt = ten.read(receipt_path)
            if receipt.get('state') != 'submitting':
                break
            time.sleep(0.5)
        else:
            raise ValueError('Group submission receipt is still ambiguous; inspect accounting')
        job_id = os.environ.get('SLURM_JOB_ID', '')
        if not job_id.isdigit() or str(receipt.get('job_id')) != job_id:
            raise ValueError('Allocation does not match the recorded grouped submission')
        cases = receipt.get('cases', [])
        if not cases or len(cases) != len(set(cases)) or not set(cases) <= set(group['cases']):
            raise ValueError('Invalid group case receipt')
        for key in cases:
            reservation = ten.read(root / 'submission' / (key + '.json'))
            if str(reservation.get('job_id')) != job_id or reservation.get('group') != group['key']:
                raise ValueError('Case reservation does not match the group: ' + key)
        # Check even if all cases happen to be accepted already.
        _, _, cfg = recovery.runtime(root, source, nodes, jobs[cases[0]]['variant'], jobs[cases[0]]['action'])
        from allocated_experiment import check_allocation
        check_allocation(cfg, 'slurm', os.environ)
        state = dict(job_id=job_id, complete=False, cases={})
        state_path = grouped / 'state' / (group['key'] + '.json')
        for key in cases:
            if recovery.complete(root, source, jobs[key]):
                state['cases'][key] = dict(status='PASS', reused=True)
            else:
                print('GROUP CASE START: ' + key, flush=True)
                start = time.monotonic()
                command = [sys.executable, str(root / 'metadata/source/scripts/recover_cresco_hv15r.py'),
                           'run', '--root', str(root), '--key', key]
                result = subprocess.run(command, cwd=root / 'metadata/source', check=False)
                passed = result.returncode == 0 and recovery.complete(root, source, jobs[key])
                state['cases'][key] = dict(status='PASS' if passed else 'FAILED',
                                          returncode=result.returncode, elapsed_seconds=time.monotonic() - start)
                print('GROUP CASE END: %s %s elapsed=%.1fs' %
                      (key, state['cases'][key]['status'], state['cases'][key]['elapsed_seconds']), flush=True)
            # Every observation is also saved atomically by the unchanged frozen runner.
            ten.write(state_path, state)
        state['complete'] = all(v['status'] == 'PASS' for v in state['cases'].values())
        ten.write(state_path, state)
        print('GROUP %s: %s' % (group['key'], 'COMPLETE' if state['complete'] else 'INCOMPLETE'), flush=True)
        return state['complete']


def status(root):
    recovery, manifest, source, jobs = verify(root)
    ids = []
    for group in manifest['groups']:
        completed = sum(recovery.complete(root, source, jobs[key]) for key in group['cases'])
        receipt = root / 'grouped/submission' / (group['key'] + '.json')
        job_id = str(recovery.ten.read(receipt).get('job_id', 'UNKNOWN')) if receipt.exists() else 'NOT_SUBMITTED'
        print('%s: %d/%d PASS | JOB=%s' % (group['key'], completed, len(group['cases']), job_id))
        if job_id.isdigit():
            ids.append(job_id)
    if ids:
        subprocess.run(['sacct', '-X', '-j', ','.join(ids),
                        '--format=JobID,JobName%40,State,Elapsed,ExitCode'], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'submit', 'run', 'status'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--nodes', type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == 'run':
        if args.nodes is None:
            parser.error('run requires --nodes')
        return 0 if run(root, args.nodes) else 1
    globals()[args.action](root)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, KeyError, StopIteration, subprocess.SubprocessError) as error:
        print('ERROR: ' + str(error), file=sys.stderr)
        sys.exit(1)
