#!/usr/bin/env bash
set -euo pipefail
usage() {
  cat <<'EOF'
Usage:
  scripts/verify/distributed.sh [--python PATH]

Runs tests marked as distributed. The caller owns process and GPU allocation.
EOF
}
PY="${PY:-python}"
while (($#)); do
  case "$1" in
    --python) [[ $# -ge 2 ]] || exit 2; PY="$2"; shift 2 ;;
    --python=*) PY="${1#--python=}"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
exec "$PY" -m pytest -m distributed tests/distributed
