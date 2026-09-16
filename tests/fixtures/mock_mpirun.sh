#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

calls=("$MOCK_LOG_DIR"/call.*)
printf '%s\0' "$@" > "$MOCK_LOG_DIR/call.${#calls[@]}"
exit "${MOCK_MPI_EXIT:-0}"
