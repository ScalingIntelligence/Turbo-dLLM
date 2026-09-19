#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/verify/cpu.sh [--python PATH]

Validates the local dllm_parallel checkout without starting CUDA training:
  - imports the package entrypoint;
  - prints supported backbones and parallel axes;
  - resolves every packaged smoke and example recipe;
  - byte-compiles the package and repository scripts.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${PY:-python}"

while (($#)); do
  case "$1" in
    --python)
      [[ $# -ge 2 ]] || { echo "--python requires a value" >&2; exit 2; }
      PY="$2"
      shift 2
      ;;
    --python=*)
      PY="${1#--python=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

"$PY" -m dllm_parallel.training --print-supported-configs >/dev/null

while IFS= read -r recipe; do
  "$PY" -m dllm_parallel.cli config validate --recipe "$recipe" >/dev/null
done < <("$PY" -c 'from dllm_parallel.recipes import list_recipes; print(*(item.name for item in list_recipes()), sep="\n")')

"$PY" -m compileall -q dllm_parallel scripts

echo "dllm_parallel install check passed"
