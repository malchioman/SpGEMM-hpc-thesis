#!/usr/bin/env bash
# Run the unpacked recovery bundle, never overwrite the active repository/build.
set -euo pipefail
bundle="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source_repo="$(cd -- "${1:?Usage: bash recovery/scripts/recover_cresco_missing.sh ORIGINAL_REPO}" && pwd)"
if [[ "$bundle" == "$source_repo" ]]; then
    echo "Run this from the separate recovery bundle, not the active source tree." >&2
    exit 2
fi
output="$source_repo/results/tesi-003/recovery-reddit-mem01"
compact_build="$bundle/build-compact"
python_bin="${PYTHON:-python3}"
for tool in cmake mpirun sbatch squeue scontrol; do command -v "$tool" >/dev/null; done
if [[ -f "$output/metadata/submitted.json" ]]; then
    "$python_bin" "$bundle/scripts/recover_cresco_missing.py" --manifest "$output/manifest.json" --submit
    "$python_bin" "$bundle/scripts/recover_cresco_missing.py" --manifest "$output/manifest.json" --status
    exit 0
fi
if [[ ! -f "$output/manifest.json" ]]; then
    "$python_bin" "$bundle/scripts/recover_cresco_missing.py" \
        --source-repo "$source_repo" --output "$output" --compact-build "$compact_build"
fi
# Inherit the original compiler and optimization flags, while using a fresh build.
"$python_bin" - "$source_repo/build/CMakeCache.txt" "$bundle" "$compact_build" <<'PY'
from pathlib import Path
import subprocess
import sys
cache_path, source, build = sys.argv[1:]
cache = {}
for line in Path(cache_path).read_text().splitlines():
    if line and not line.startswith(('#', '//')) and '=' in line and ':' in line.split('=', 1)[0]:
        typed, value = line.split('=', 1)
        cache[typed.split(':', 1)[0]] = value
command = ['cmake', '-S', source, '-B', build, '-DSPGEMM_COMPACT_TRIDENT=ON']
for key in ('CMAKE_CXX_COMPILER', 'CMAKE_BUILD_TYPE', 'CMAKE_CXX_FLAGS',
            'CMAKE_CXX_FLAGS_RELEASE', 'CMAKE_CXX_FLAGS_RELWITHDEBINFO',
            'CMAKE_CXX_FLAGS_MINSIZEREL', 'CMAKE_CXX_FLAGS_DEBUG',
            'CMAKE_EXE_LINKER_FLAGS', 'SPGEMM_MPI_CHUNK_LIMIT'):
    if key in cache:
        command.append('-D' + key + '=' + cache[key])
subprocess.run(command, check=True)
subprocess.run(['cmake', '--build', build, '--parallel', '2', '--target',
                'trident_two_sided', 'trident_hybrid', 'trident_get',
                'trident_get_pipeline', 'trident_put'], check=True)
PY
"$python_bin" "$bundle/scripts/recover_cresco_missing.py" --manifest "$output/manifest.json" --seal
"$python_bin" "$bundle/scripts/recover_cresco_missing.py" --manifest "$output/manifest.json" --submit
echo "Recovery submitted. Existing results and active binaries were left intact."
