#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/verify/gpu_smoke.sh --config PATH [--nproc-per-node N]

Runs an explicit generic smoke configuration against installed GPU artifacts.
EOF
}
CONFIG=""
NPROC=1
while (($#)); do
  case "$1" in
    --config) [[ $# -ge 2 ]] || exit 2; CONFIG="$2"; shift 2 ;;
    --config=*) CONFIG="${1#--config=}"; shift ;;
    --nproc-per-node) [[ $# -ge 2 ]] || exit 2; NPROC="$2"; shift 2 ;;
    --nproc-per-node=*) NPROC="${1#--nproc-per-node=}"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -n "$CONFIG" ]] || { echo "--config is required" >&2; exit 2; }
python -m dllm_parallel.core.profiling.release_gates preflight
exec python -m dllm_parallel.cli launch --config "$CONFIG" --nproc-per-node "$NPROC"
