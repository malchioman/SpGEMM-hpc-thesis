#!/usr/bin/env python3
"""Audit and package tesi-006 originals and recoveries for download; no TSV rewriting."""
import argparse
import datetime
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

EXCLUDED = ('metadata/build', 'metadata/bin', 'references')


def load_driver(root):
    path = root / 'grouped/metadata/driver.py'
    spec = importlib.util.spec_from_file_location('_export_frozen_group', path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    return driver


def recovery_accounting(root, runner, group_manifest, jobs):
    paths = [root / 'submission' / (key + '.json') for key in jobs]
    paths += [root / 'grouped/submission' / (g['key'] + '.json') for g in group_manifest['groups']]
    ids = set()
    for path in paths:
        if not path.exists():
            continue
        job_id = str(runner.ten.read(path).get('job_id', ''))
        if not job_id.isdigit():
            raise ValueError('Ambiguous submission receipt: ' + str(path))
        ids.add(job_id)
    if not ids:
        return {}
    result = subprocess.run(['sacct', '-X', '-n', '-P', '-j', ','.join(sorted(ids)),
                             '--format=JobIDRaw,State'], check=True, capture_output=True, text=True)
    states = {}
    for line in result.stdout.splitlines():
        fields = line.strip().split('|')
        if len(fields) >= 2 and fields[0] in ids:
            states[fields[0]] = fields[1].split()[0].rstrip('+')
    active = {job: states.get(job, 'UNKNOWN') for job in ids if states.get(job) not in runner.TERMINAL}
    if active:
        raise ValueError('Recovery jobs still active or accounting incomplete: ' + str(active))
    return states


def audit(root):
    driver = load_driver(root)
    runner, grouped, source, jobs = driver.verify(root)
    accounting = recovery_accounting(root, runner, grouped, jobs)
    ten = runner.ten
    original = ten.read(source / 'manifest.json')
    base, e = ten.runtime(source)
    report = dict(schema='tesi006-download-v1', created_utc=datetime.datetime.now(
                  datetime.timezone.utc).isoformat(), source=str(source), recovery=str(root),
                  recovery_accounting=accounting, excluded_directories=list(EXCLUDED),
                  samples_per_observation=10, nodes={}, complete=True)
    for nodes in ten.NODES:
        cfg = e.options(['all', '--config', str(source / ('metadata/n%d.json' % nodes))])
        expected = set()
        for block in (b for b in original['jobs'] if b['nodes'] == nodes):
            for case in e.cases(cfg, block['action']):
                if case[0] in block['matrices']:
                    pair = tuple(p.relative_to(cfg.matrix_dir).as_posix() for p in case[2:])
                    expected.update((block['action'], v) + pair for v in block['variants'])
        if len(expected) != 184:
            raise ValueError('Unexpected original coverage at n%d' % nodes)
        accepted = {(action,) + key for action in ten.MAIN for key in base.existing(e, cfg, action)}
        provenance = {key: 'partial10-cached' for key in accepted}
        duplicates = []
        for job in (j for j in jobs.values() if j['nodes'] == nodes):
            if driver.complete(runner, root, source, job):
                key = (job['action'], job['variant']) + tuple(job['pair'])
                if key in accepted:
                    duplicates.append(key)
                accepted.add(key)
                provenance[key] = ('additional-cached' if job.get('kind') == 'original-partial'
                                   else 'hv15r-roundoff-cached-v4')
        missing, unexpected = sorted(expected - accepted), sorted(accepted - expected)
        complete = not (missing or unexpected or duplicates)
        report['nodes'][str(nodes)] = dict(expected=len(expected), accepted=len(accepted),
            complete=complete, missing=missing, unexpected=unexpected, duplicates=duplicates,
            observations=[dict(action=k[0], variant=k[1], matrix_a=k[2], matrix_b=k[3], source=provenance[k])
                          for k in sorted(accepted)])
        report['complete'] &= complete
        print('NODI=%d: %d/184 | mancanti=%d | duplicati=%d | %s' %
              (nodes, len(accepted), len(missing), len(duplicates), 'OK' if complete else 'INCOMPLETO'))
    report['expected_observations'] = sum(n['expected'] for n in report['nodes'].values())
    report['accepted_observations'] = sum(n['accepted'] for n in report['nodes'].values())
    return report, source


def members(root):
    """Include raw tables, samples, logs, checks, sources, configs and receipts."""
    for directory, dirs, files in os.walk(root, followlinks=False):
        directory = Path(directory)
        kept = []
        for name in sorted(dirs):
            path = directory / name
            relative = path.relative_to(root).as_posix()
            if relative in EXCLUDED or name == '__pycache__':
                continue
            if path.is_symlink():
                raise ValueError('Unexpected symlink in exported data: ' + str(path))
            kept.append(name)
        dirs[:] = kept
        for name in sorted(files):
            if name.endswith('.pyc') or name == '.campaign.lock':
                continue
            path = directory / name
            if path.is_symlink() or not path.is_file():
                raise ValueError('Unexpected non-regular exported file: ' + str(path))
            yield path


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def package(root, source, report, output):
    for directory in (root, source):
        if output == directory or directory in output.parents:
            raise ValueError('Archive must be outside both exported directories')
    checksum = output.with_name(output.name + '.sha256')
    temporary = output.with_name(output.name + '.part')
    if any(p.exists() for p in (output, checksum, temporary)):
        raise ValueError('Export already exists; choose a new --output: ' + str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    files = [(path, label + '/' + path.relative_to(directory).as_posix())
             for directory, label in ((source, 'partial10-cached'), (root, 'hv15r-roundoff-cached-v4'))
             for path in members(directory)]
    report['archive_file_count'] = len(files)
    with tarfile.open(temporary, 'x:gz') as archive:
        for path, relative in files:
            archive.add(path, arcname=relative, recursive=False)
        payloads = {
            'export-audit.json': json.dumps(report, indent=2) + '\n',
            'README.txt': (
                'tesi-006: raw results and recovery observations, preserved without TSV merging.\n'
                'Read export-audit.json before treating the campaign as complete.\n'
                'partial10-cached/nN/{strong_scaling,permutation,rectangular}: original accepted tables.\n'
                'hv15r-roundoff-cached-v4/nN/{strong_scaling,permutation}: numeric Full HV15R recovery, including probes.\n'
                'hv15r-roundoff-cached-v4/additional-cached/n100/permutation: original-partial additional recovery.\n'
                'Each main observation contains 10 samples. Pilot/check/raw failure tables are diagnostics, not extra observations.\n'
                'Join accepted main observations by action, nodes, variant, matrix_a and matrix_b; preserve validation and provenance.\n'
                'Matrices, reference caches, compiled binaries and build trees are not included.\n'
                'Frozen sources, configuration, manifests, receipts and available logs are included.\n'
                'Frozen scripts contain original cluster paths and are preserved as provenance, not as a relocatable installation.\n')}
        for name, text in payloads.items():
            data = text.encode('utf-8')
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(data))
    os.replace(temporary, output)
    with checksum.open('x', encoding='ascii') as stream:
        stream.write(digest(output) + '  ' + output.name + '\n')
    print('AUDIT: ' + ('COMPLETO' if report['complete'] else 'INCOMPLETO: conservati anche i dati diagnostici'))
    print('ARCHIVIO: %s (%d bytes)' % (output, output.stat().st_size))
    print('SHA256: ' + str(checksum))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    report, source = audit(root)
    package(root, source, report, output)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError, StopIteration, subprocess.SubprocessError) as error:
        print('ERROR: ' + str(error), file=sys.stderr)
        sys.exit(1)
