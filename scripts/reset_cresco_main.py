#!/usr/bin/env python3
"""Permanently reset this project's CRESCO checkout, retaining the external matrices.

Run on a CRESCO frontend: python3 reset_cresco_main.py --execute
No build, matrix download, or new experiment submission is performed.
"""
import argparse
import datetime
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


PARENT = Path('/lpor1/store_0/usr/malchiod/tirocinio-spgemm')
MATRICES = Path('/lpor1/scratch_1/usr/malchiod/tirocinio-spgemm/matrices')
REPO_NAME = 'SpGEMM-hpc-thesis'
REMOTE = 'https://github.com/malchioman/SpGEMM-hpc-thesis.git'
REQUIRED_MAIN = '8d726706f71e3339e095b0a5ba8e4a13b7aa5937'
CHECKOUT_PATTERN = re.compile(
    r'(?:SpGEMM-hpc-thesis\.previous-|\.SpGEMM-hpc-thesis-(?:main|purge)-)'
    r'\d{8}T\d{6}Z-\d+$')
RESULT_NAMES = {'results', 'test-results', 'tesi003-prima-ripetizione.tar.gz',
                'tesi003-prima-ripetizione.tar.gz.sha256'}


def within(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def exists(path):
    return path.exists() or path.is_symlink()


def current_user():
    import pwd
    return pwd.getpwuid(os.getuid()).pw_name


def command(argv, timeout=60):
    return subprocess.run(argv, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=True, timeout=timeout).stdout.strip()


def mount_points():
    # Unlike ismount(), mountinfo also identifies bind mounts on the same device.
    content = Path('/proc/self/mountinfo').read_text()
    return [Path(re.sub(r'\\([0-7]{3})', lambda m: chr(int(m.group(1), 8)), line.split()[4]))
            for line in content.splitlines()]


def validate_target(target, parents, protected, mounts):
    if target.is_symlink():
        raise RuntimeError('Destinazione di pulizia simbolica non ammessa: ' + str(target))
    resolved = target.resolve(strict=True)
    if resolved != target or resolved.parent not in parents:
        raise RuntimeError('Destinazione fuori dalle directory autorizzate: ' + str(target))
    if target.name not in RESULT_NAMES and target.name != REPO_NAME and not CHECKOUT_PATTERN.fullmatch(target.name):
        raise RuntimeError('Nome di destinazione non autorizzato: ' + str(target))
    for keep in protected:
        if within(keep, target) or within(target, keep):
            raise RuntimeError('La cancellazione coinvolgerebbe dati protetti: ' + str(target))
    if any(within(mount, target) for mount in mounts):
        raise RuntimeError('Mount point nella destinazione; cancellazione rifiutata: ' + str(target))
    return resolved


def discover_targets(parent, scratch_parent):
    targets = []
    for child in parent.iterdir():
        if child.name == REPO_NAME or CHECKOUT_PATTERN.fullmatch(child.name) or child.name in RESULT_NAMES:
            targets.append(child)
    if scratch_parent != parent:
        targets.extend(scratch_parent / name for name in RESULT_NAMES if exists(scratch_parent / name))
    return sorted(set(targets))


def select_jobs(rows, roots):
    selected = []
    for row in rows.splitlines():
        fields = row.split('|')
        if len(fields) != 3:
            raise RuntimeError('Risposta squeue inattesa: ' + row)
        job_id, name, directory = (field.strip() for field in fields)
        path = Path(directory)
        project_directory = path.is_absolute() and any(within(path.resolve(), root) for root in roots)
        if re.match(r'^tesi00[1-4]-', name) or project_directory:
            if not re.fullmatch(r'\d+(?:_\d+)?', job_id):
                raise RuntimeError('Identificativo job da controllare prima del reset: ' + job_id)
            selected.append(job_id)
    return list(dict.fromkeys(selected))


def project_jobs(roots):
    rows = command(['squeue', '--array', '-h', '-u', 'malchiod', '-o', '%i|%j|%Z'])
    return select_jobs(rows, roots)


def check_main(fresh):
    if command(['git', '-C', str(fresh), 'branch', '--show-current']) != 'main':
        raise RuntimeError('Il clone sostitutivo non e sul branch main.')
    command(['git', '-C', str(fresh), 'merge-base', '--is-ancestor', REQUIRED_MAIN, 'HEAD'])
    checks = [('src/common/csr_matrix.hpp', 'using CsrOffset = std::int64_t;'),
              ('src/common/csr_matrix.hpp', 'std::vector<CsrOffset> rowPtr;'),
              ('src/common/spgemm_common.cpp', 'ValidationMode::Partial'),
              ('scripts/lib/experiments.py', '"--partial"')]
    for relative, marker in checks:
        if marker not in (fresh / relative).read_text(encoding='utf-8'):
            raise RuntimeError('Funzionalita richiesta non trovata in ' + relative)
    if command(['git', '-C', str(fresh), 'status', '--porcelain']):
        raise RuntimeError('Il clone sostitutivo contiene modifiche inattese.')
    return command(['git', '-C', str(fresh), 'rev-parse', 'HEAD'])


def remove_target(target, parents, protected):
    # Revalidate immediately before each filesystem mutation.
    validate_target(target, parents, protected, mount_points())
    print('Elimino definitivamente: ' + str(target), flush=True)
    if target.is_dir():
        shutil.rmtree(target)  # Does not traverse directory symlinks inside the tree.
    else:
        target.unlink()
    if exists(target):
        raise RuntimeError('La destinazione non e stata rimossa: ' + str(target))


def reset():
    if sys.platform != 'linux' or current_user() != 'malchiod':
        raise RuntimeError('Eseguire sul frontend CRESCO con utente malchiod.')
    for executable in ('git', 'squeue', 'scancel'):
        if not shutil.which(executable):
            raise RuntimeError('Comando mancante: ' + executable)
    parent = PARENT.resolve(strict=True)
    matrices = MATRICES.resolve(strict=True)
    scratch_parent = MATRICES.parent.resolve(strict=True)
    repo = parent / REPO_NAME
    if within(matrices, repo):
        raise RuntimeError('La cache delle matrici deve essere esterna al checkout.')
    if not (matrices / 'metadata').is_dir() or not (matrices / 'cage8.mtx').is_file():
        raise RuntimeError('Cache esterna delle matrici o metadati mancanti.')
    sentinel = hashlib.sha256((matrices / 'cage8.mtx').read_bytes()).hexdigest()
    parents = {parent, scratch_parent}
    targets = discover_targets(parent, scratch_parent)
    for target in targets:
        validate_target(target, parents, [matrices], mount_points())

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + str(os.getpid())
    fresh = parent / ('.SpGEMM-hpc-thesis-main-' + stamp)
    trash = parent / ('.SpGEMM-hpc-thesis-purge-' + stamp)
    if exists(fresh) or exists(trash):
        raise RuntimeError('Una directory temporanea esiste gia.')
    print('Prima verifico il nuovo main; nessun dato precedente ancora eliminato.', flush=True)
    clone_env = dict(os.environ, GIT_TERMINAL_PROMPT='0')
    subprocess.run(['git', 'clone', '--branch', 'main', '--single-branch', '--', REMOTE, str(fresh)],
                   check=True, timeout=600, env=clone_env)
    commit = check_main(fresh)
    print('main verificato: ' + commit, flush=True)
    print('Matrici conservate: ' + str(matrices), flush=True)
    for target in targets:
        print('Da eliminare: ' + str(target), flush=True)

    roots = [repo] + [target for target in targets if target.is_dir()]
    jobs = project_jobs(roots)
    if jobs:
        print('Annullo tutti i job del progetto: ' + ', '.join(jobs), flush=True)
        command(['scancel', '--quiet'] + jobs)
    deadline = time.monotonic() + 60
    while True:
        remaining = project_jobs(roots)
        if not remaining:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError('Job ancora presenti; nessun checkout eliminato: ' + ', '.join(remaining))
        time.sleep(5)

    # Both installation moves stay within the previously resolved project parent.
    # The temporary old checkout is purged below, not retained as a backup.
    check_main(fresh)
    if exists(repo):
        validate_target(repo, parents, [matrices, fresh], mount_points())
        if exists(trash) or trash.parent != parent:
            raise RuntimeError('Destinazione temporanea inattesa.')
        repo.rename(trash)
        try:
            fresh.rename(repo)
        except OSError:
            trash.rename(repo)
            raise
        targets = [trash if target == repo else target for target in targets]
    else:
        fresh.rename(repo)
    os.chdir(parent)
    for target in targets:
        if exists(target):
            remove_target(target, parents, [matrices, repo])

    if hashlib.sha256((matrices / 'cage8.mtx').read_bytes()).hexdigest() != sentinel:
        raise RuntimeError('Il controllo finale della cache non coincide.')
    leftovers = [path for path in discover_targets(parent, scratch_parent) if path != repo]
    if leftovers:
        raise RuntimeError('Cartelle di pulizia ancora presenti: ' + ', '.join(map(str, leftovers)))
    if project_jobs([repo]):
        raise RuntimeError('Sono comparsi nuovi job del progetto: controllare squeue.')
    print('\nPULIZIA DEFINITIVA COMPLETATA', flush=True)
    print('Clone nuovo: ' + str(repo))
    print('Branch: main | Commit: ' + commit)
    print('Matrici conservate: ' + str(matrices))
    print('Vecchi clone, backup riconosciuti e risultati rimossi. Nessun nuovo job inviato.')
    print('Lo storico contabile Slurm dei job terminati resta consultabile con sacct.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', required=True,
                        help='Cancel project jobs and permanently remove old project data.')
    parser.parse_args()
    try:
        reset()
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        detail = getattr(error, 'stderr', None)
        print('ERRORE: ' + str(error) + ('\n' + detail if detail else ''), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
