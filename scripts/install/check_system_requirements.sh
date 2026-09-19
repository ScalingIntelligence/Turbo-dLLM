#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/install/check_system_requirements.sh [--python PATH] [--gpu]

Checks the supported Python version and, with --gpu, the Linux/CUDA runtime.
It does not install or modify anything.
EOF
}

PY="${PY:-python}"
GPU=0
while (($#)); do
  case "$1" in
    --python) [[ $# -ge 2 ]] || exit 2; PY="$2"; shift 2 ;;
    --python=*) PY="${1#--python=}"; shift ;;
    --gpu) GPU=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

"$PY" -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ is required"'
if (( GPU )); then
  [[ "$(uname -s)" == Linux ]] || { echo "GPU bundles support Linux only" >&2; exit 1; }
  command -v nvidia-smi >/dev/null || { echo "nvidia-smi was not found" >&2; exit 1; }
  nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
fi
echo "system requirements passed"
