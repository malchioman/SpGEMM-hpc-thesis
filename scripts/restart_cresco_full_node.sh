#!/usr/bin/env bash
# Replace the tesi-002 campaign with the same experiment at 2 MPI x 63 threads/node.
# Run on the cluster frontend with modules and .venv-sbatchman already activated.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

OLD_ROOT=results/tesi-002/definitiva-001
NEW_ROOT=results/tesi-003/definitiva-001
PROFILE=scripts/sbatchman/cresco-restart.local.json
# The job manifest can be retained separately when old result directories are removed.
OLD_MANIFEST="${OLD_MANIFEST:-$OLD_ROOT/metadata/expected.json}"
if [[ ! -f "$OLD_MANIFEST" ]]; then
  OLD_MANIFEST=scripts/sbatchman/tesi002-jobs.local.json
fi
export OLD_ROOT NEW_ROOT OLD_MANIFEST

command -v sbatchman >/dev/null
command -v squeue >/dev/null
command -v scancel >/dev/null
test -f "$PROFILE"
test ! -e "$NEW_ROOT"

# Validate the exact old campaign scope before preparing or cancelling anything.
python - <<'PY'
import json, os
from pathlib import Path
items = json.loads(Path(os.environ['OLD_MANIFEST']).read_text())
expected = {(r, n) for r in range(1, 11) for n in (1, 4, 9, 16)}
if len(items) != 40 or {(x['repetition'], x['nodes']) for x in items} != expected:
    raise SystemExit('Old campaign does not match the expected 40 jobs; nothing cancelled.')
if any(x['tag'] != 'tesi001-r%02d-n%d' % (x['repetition'], x['nodes']) for x in items):
    raise SystemExit('Unexpected old job names; nothing cancelled.')
PY

# Fully prepare and archive the new campaign before stopping the old jobs.
python scripts/prepare_cresco_run_all.py \
  --profile "$PROFILE" --campaign tesi003 \
  --nodes 1 4 9 16 --ranks-per-node 2 --threads 63 --cpus-per-rank 64 \
  --output "$NEW_ROOT"

tar --exclude='scripts/sbatchman/generated' --exclude='__pycache__' \
  -czf "$NEW_ROOT/metadata/source.tar.gz" CMakeLists.txt src scripts tests docs README.md
cp build/CMakeCache.txt "$NEW_ROOT/metadata/CMakeCache.txt"
cp -a matrices/metadata "$NEW_ROOT/metadata/matrices"
df -h . matrices > "$NEW_ROOT/metadata/storage.txt"

# Match only this campaign's expected tags (SbatchMan appends a job-name suffix).
# Cancel waiting jobs first so afterany dependencies cannot start later repetitions.
# Completed results and all old directories are retained.
python - <<'PY'
import getpass, json, os, subprocess, time
from pathlib import Path
old = Path(os.environ['OLD_ROOT'])
meta = Path(os.environ['NEW_ROOT']) / 'metadata'
tags = {x['tag'] for x in json.loads(Path(os.environ['OLD_MANIFEST']).read_text())}
def queue():
    output = subprocess.run(
        ['squeue', '-h', '-u', getpass.getuser(), '-o', '%i|%j|%T'],
        check=True, capture_output=True, text=True).stdout
    selected = []
    for line in output.splitlines():
        job_id, name, state = (x.strip() for x in line.split('|', 2))
        if any(name == tag or name.startswith(tag + '-') for tag in tags):
            if not job_id.isdigit():
                raise SystemExit('Unexpected job ID: ' + job_id)
            selected.append({'job_id': job_id, 'name': name, 'state': state})
    return selected
jobs = queue()
(meta / 'cancelled-campaign.json').write_text(json.dumps(
    {'results_dir': str(old.resolve()), 'jobs': jobs}, indent=2) + '\n')
for pending in (True, False):
    ids = [j['job_id'] for j in jobs if (j['state'] == 'PENDING') == pending]
    if ids:
        print('Stopping old campaign jobs: ' + ', '.join(ids), flush=True)
        subprocess.run(['scancel', *ids], check=True)
deadline = time.monotonic() + 300
while queue():
    if time.monotonic() >= deadline:
        raise SystemExit('Old jobs still visible after 5 minutes. New plan prepared but not submitted.')
    print('Waiting for old allocations to finish releasing nodes...', flush=True)
    time.sleep(5)
print('Old campaign stopped; existing output files retained.', flush=True)
PY

sbatchman set-cluster-name cresco8
sbatchman configure --file "$NEW_ROOT/metadata/plan/configs.yaml" --overwrite
for lane in 1 2 3 4; do
  sbatchman launch --file "$NEW_ROOT/metadata/plan/lane-$lane.yaml" \
    | tee "$NEW_ROOT/metadata/submit-lane-$lane.log"
done
squeue -u "$USER" -o '%.18i %.100j %.12T %.6D %R' \
  | tee "$NEW_ROOT/metadata/queue-after-submit.txt"
