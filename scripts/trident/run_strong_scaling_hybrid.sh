#!/usr/bin/env bash
set -euo pipefail

VARIANT=hybrid exec bash "$(dirname -- "${BASH_SOURCE[0]}")/run_strong_scaling.sh" "$@"
