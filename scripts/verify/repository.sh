#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/verify/repository.sh [--python PATH]

Runs portable source, import, recipe, and release-policy checks.
EOF
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-python}"

while (($#)); do
  case "$1" in
    --python) [[ $# -ge 2 ]] || exit 2; PY="$2"; shift 2 ;;
    --python=*) PY="${1#--python=}"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

cd "$ROOT"

"$PY" -m compileall -q dllm_parallel scripts tests
"$PY" -m ruff check --select F dllm_parallel
"$PY" -m dllm_parallel.core.profiling.release_gates policy --portable --root "$ROOT"
