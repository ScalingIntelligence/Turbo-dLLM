#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/verify/portable_artifacts.sh --directory PATH [--python PATH]

Verifies the public contents of one Turbo-dLLM wheel and source distribution.
EOF
}

DIRECTORY=""
PY="${PY:-python}"
while (($#)); do
  case "$1" in
    --directory) [[ $# -ge 2 ]] || exit 2; DIRECTORY="$2"; shift 2 ;;
    --directory=*) DIRECTORY="${1#--directory=}"; shift ;;
    --python) [[ $# -ge 2 ]] || exit 2; PY="$2"; shift 2 ;;
    --python=*) PY="${1#--python=}"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -n "$DIRECTORY" ]] || { echo "--directory is required" >&2; exit 2; }
exec "$PY" -m dllm_parallel.core.profiling.portable_artifacts --directory "$DIRECTORY"
