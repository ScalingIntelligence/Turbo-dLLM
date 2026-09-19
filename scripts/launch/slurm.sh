#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  srun --nodes=N scripts/launch/slurm.sh --config PATH [trainer overrides...]

Adapts an existing Slurm allocation to the generic torchrun launcher. It does
not encode accounts, partitions, storage paths, models, or datasets.
EOF
}

[[ "${1:-}" != -h && "${1:-}" != --help ]] || { usage; exit 0; }
[[ -n "${SLURM_NNODES:-}" && -n "${SLURM_NODEID:-}" ]] || {
  echo "an active Slurm allocation is required" >&2
  exit 2
}
MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)}"
exec "${PY:-python}" -m dllm_parallel.cli launch \
  --nnodes "$SLURM_NNODES" \
  --node-rank "$SLURM_NODEID" \
  --nproc-per-node "${SLURM_GPUS_ON_NODE:-1}" \
  --master-addr "$MASTER_ADDR" \
  "$@"
