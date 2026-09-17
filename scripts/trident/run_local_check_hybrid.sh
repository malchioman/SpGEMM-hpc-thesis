#!/usr/bin/env bash
set -euo pipefail

VARIANT=hybrid exec bash "$(dirname -- "${BASH_SOURCE[0]}")/run_local_check.sh" "$@"
