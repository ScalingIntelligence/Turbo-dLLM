#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/launch/torchrun.sh (--config PATH | --recipe NAME) [launch options] [trainer overrides...]

Compatibility wrapper for the installed `dllm launch` command. All validation,
environment setup, preflight, and torchrun command construction live in the
Python package so checkout and pip users have identical behavior.
EOF
}

[[ "${1:-}" != -h && "${1:-}" != --help ]] || { usage; exit 0; }
exec "${PY:-python}" -m dllm_parallel.cli launch "$@"
