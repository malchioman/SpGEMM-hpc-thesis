#!/usr/bin/env python3
"""Replace only a pending HV15R preparation with an independent early preparation."""
import argparse
import getpass
from pathlib import Path
import shlex
import subprocess
import sys
import time

import launch_cresco_ten as ten
import recover_cresco_hv15r as recovery


def batch(repo, source, root, cache):
    script = repo / 'scripts/recover_cresco_hv15r.py'
    log = repo.parent / 'hvfull-cached-prepare-%j.log'
    return '\n'.join([
        '#!/usr/bin/env bash', '#SBATCH --job-name=hvfull-early-prepare',
        '#SBATCH --partition=cresco8_cpu', '#SBATCH --nodes=1', '#SBATCH --ntasks=1',
        '#SBATCH --cpus-per-task=16', '#SBATCH --mem=450000M', '#SBATCH --exclusive',
        '#SBATCH --hint=nomultithread', '#SBATCH --time=24:00:00',
        '#SBATCH --output=' + shlex.quote(str(log)), '#SBATCH --error=' + shlex.quote(str(log)),
        'set -euo pipefail', 'source /usr/share/Modules/init/bash',
        'module load intel/oneapi mpi_flavour/openmpi_intel-4.1.7',
        'cd -- ' + shlex.quote(str(repo)),
        shlex.join(['python3', str(script), 'prepare', '--early-probes', '--repo', str(repo),
                    '--source', str(source), '--root', str(root), '--reference-cache-dir', str(cache)]),
        shlex.join(['python3', str(root / 'metadata/source/scripts/recover_cresco_hv15r.py'),
                    'submit-references', '--root', str(root)]),
        "echo 'BUILD E TEST COMPLETATI; RIFERIMENTI NUMERICI INVIATI IN CODA.'", ''])


def queued(job_id):
    result = subprocess.run(['squeue', '-h', '-j', job_id, '-o', '%i|%j|%u|%T'],
                            check=True, capture_output=True, text=True)
    return [line.strip().split('|') for line in result.stdout.splitlines() if line.strip()]


def start(repo, job_id, cache):
    if not str(job_id).isdigit():
        raise ValueError('Expected a numeric preparation job ID')
    job_id = str(job_id)
    source = repo / 'results/tesi-006/partial10-cached'
    root = repo / 'results/tesi-006/hv15r-roundoff-cached-v4'
    control = repo.parent / 'hvfull-cached-early-control'
    with ten.file_lock(control):
        receipt = control / 'submission.json'
        if receipt.exists():
            raise ValueError('Early submission already recorded; inspect ' + str(receipt) + ' before retrying')
        if root.exists():
            raise ValueError('Recovery directory already exists; inspect preparation instead of replacing it: ' + str(root))
        # All original snapshots stay read-only. Only the ten four-node jobs must be terminal.
        recovery.validate_source(source, nodes=(4,))
        recovery.guard_sources(repo, source)
        plan = recovery.missing_jobs(source, nodes=(4,))
        if any(not any(j['action'] == action for j in plan) for action in recovery.ACTIONS):
            raise ValueError('No missing four-node case for each product; inspect the recovery plan first')
        if ten.below(cache, source) or ten.below(source, cache):
            raise ValueError('Reference cache must not overlap the source campaign')
        expected = [[job_id, 'hvfull-cached-prepare', getpass.getuser(), 'PENDING']]
        if queued(job_id) != expected:
            raise ValueError('The specified job is not your pending hvfull-cached-prepare; nothing cancelled')
        body = batch(repo, source, root, cache)
        subprocess.run(['sbatch', '--test-only'], input=body, text=True, check=True)
        # The state filter prevents cancelling a preparation which started during the checks.
        subprocess.run(['scancel', '--state=PENDING', job_id], check=True)
        for _ in range(10):
            remaining = queued(job_id)
            if not remaining:
                break
            if remaining != expected:
                raise ValueError('Old preparation changed state; no replacement submitted. Inspect ' + job_id)
            time.sleep(0.2)
        else:
            raise ValueError('Old preparation still appears in queue; no replacement submitted')
        if root.exists():
            raise ValueError('Preparation created its directory while checking; no replacement submitted')
        # Keep the exact submitted script and a durable intent before calling sbatch.
        ten.write(control / 'prepare.sbatch', body)
        ten.write(receipt, dict(state='submitting', replaced_job=job_id))
        result = subprocess.run(['sbatch', '--parsable'], input=body, text=True, check=True,
                                capture_output=True)
        new_id = result.stdout.strip().split(';')[0]
        if not new_id.isdigit():
            raise ValueError('Ambiguous sbatch response; inspect accounting and ' + str(receipt))
        ten.write(receipt, dict(job_id=new_id, replaced_job=job_id))
        ten.write(repo.parent / 'hvfull-cached-prepare.jobid', new_id + '\n')
        print('Replaced pending preparation %s with %s, without source-job dependencies.' % (job_id, new_id))
        print('Original campaign jobs untouched. Log: ' + str(repo.parent / ('hvfull-cached-prepare-' + new_id + '.log')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--replace-prepare-job', required=True)
    parser.add_argument('--reference-cache-dir', type=Path, required=True)
    args = parser.parse_args()
    start(args.repo.resolve(), args.replace_prepare_job, args.reference_cache_dir.resolve())


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError) as error:
        print('ERROR: ' + str(error), file=sys.stderr)
        sys.exit(1)
