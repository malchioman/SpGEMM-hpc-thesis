#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="$repo/build"
jobs=2
while (($#)); do
    case "$1" in
        --build-dir|--jobs)
            if (($# < 2)); then echo "Missing value for $1" >&2; exit 2; fi
            if [[ "$1" == --build-dir ]]; then build_dir="$2"; else jobs="$2"; fi
            shift 2 ;;
        --help|-h)
            echo "Usage: bash scripts/build.sh [--build-dir DIR] [--jobs N] [-- CMAKE_OPTIONS...]"
            echo "Release build; relative directories are resolved against the repository."
            exit 0 ;;
        --) shift; break ;;
        *) echo "Unknown option: $1 (put CMake options after --)" >&2; exit 2 ;;
    esac
done
if [[ ! "$jobs" =~ ^[1-9][0-9]*$ ]]; then echo "--jobs must be positive" >&2; exit 2; fi
if [[ "$build_dir" != /* ]]; then build_dir="$repo/$build_dir"; fi
cmake -S "$repo" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release "$@"
cmake --build "$build_dir" --config Release --parallel "$jobs"
echo "Build completed: $build_dir"
