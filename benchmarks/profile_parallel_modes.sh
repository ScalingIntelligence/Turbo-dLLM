#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/profile_block_parallel.sh --config RECIPE [options] [-- launcher/trainer overrides]

Options:
  --modes LIST               Subset of cp,bp,fused. Default: cp,bp,fused.
  --context-parallel-size N  CP degree for cp and fused. Default: 1.
  --block-parallel-size N    BP degree for bp and fused. Default: CP degree.
  --run-root DIR             Output root. Default: runs/block_parallel_<timestamp>.
  --system-trace             Capture bounded CUDA/NCCL and named phase traces.
  --system-trace-backend NAME
                             kineto or nsys. Default: kineto.
  --system-trace-start-step N
                             First optimizer step in the capture. Default: 1.
  --system-trace-steps N     Number of optimizer steps to capture. Default: 1.
  -h, --help                 Show this help.

Compares pure CP, replicated-prefix BP, and fused CP+BP with identical model,
data, optimizer, and measurement settings. BP replication is explicit.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG=""
MODES="cp,bp,fused"
CP_SIZE=1
BP_SIZE=""
RUN_ROOT=""
SYSTEM_TRACE=0
SYSTEM_TRACE_BACKEND="kineto"
SYSTEM_TRACE_START_STEP=1
SYSTEM_TRACE_STEPS=1
COMMON_ARGS=()
NODE_RANK="${DLLM_PROFILE_NODE_RANK:-0}"

while (($#)); do
  case "$1" in
    --config)
      [[ $# -ge 2 ]] || { echo "--config requires a value" >&2; exit 2; }
      CONFIG="$2"
      shift 2
      ;;
    --config=*) CONFIG="${1#--config=}"; shift ;;
    --modes)
      [[ $# -ge 2 ]] || { echo "--modes requires a value" >&2; exit 2; }
      MODES="$2"
      shift 2
      ;;
    --modes=*) MODES="${1#--modes=}"; shift ;;
    --context-parallel-size)
      [[ $# -ge 2 ]] || { echo "--context-parallel-size requires a value" >&2; exit 2; }
      CP_SIZE="$2"
      shift 2
      ;;
    --context-parallel-size=*) CP_SIZE="${1#--context-parallel-size=}"; shift ;;
    --block-parallel-size)
      [[ $# -ge 2 ]] || { echo "--block-parallel-size requires a value" >&2; exit 2; }
      BP_SIZE="$2"
      shift 2
      ;;
    --block-parallel-size=*) BP_SIZE="${1#--block-parallel-size=}"; shift ;;
    --run-root)
      [[ $# -ge 2 ]] || { echo "--run-root requires a value" >&2; exit 2; }
      RUN_ROOT="$2"
      shift 2
      ;;
    --run-root=*) RUN_ROOT="${1#--run-root=}"; shift ;;
    --system-trace) SYSTEM_TRACE=1; shift ;;
    --system-trace-backend)
      [[ $# -ge 2 ]] || { echo "--system-trace-backend requires a value" >&2; exit 2; }
      SYSTEM_TRACE_BACKEND="$2"
      shift 2
      ;;
    --system-trace-backend=*) SYSTEM_TRACE_BACKEND="${1#--system-trace-backend=}"; shift ;;
    --system-trace-start-step)
      [[ $# -ge 2 ]] || { echo "--system-trace-start-step requires a value" >&2; exit 2; }
      SYSTEM_TRACE_START_STEP="$2"
      shift 2
      ;;
    --system-trace-start-step=*) SYSTEM_TRACE_START_STEP="${1#--system-trace-start-step=}"; shift ;;
    --system-trace-steps)
      [[ $# -ge 2 ]] || { echo "--system-trace-steps requires a value" >&2; exit 2; }
      SYSTEM_TRACE_STEPS="$2"
      shift 2
      ;;
    --system-trace-steps=*) SYSTEM_TRACE_STEPS="${1#--system-trace-steps=}"; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; COMMON_ARGS=("$@"); break ;;
    *) echo "unknown block-profile option: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$CONFIG" ]] || { echo "--config is required" >&2; exit 2; }
[[ -f "$CONFIG" ]] || { echo "profile recipe not found: $CONFIG" >&2; exit 2; }
BP_SIZE="${BP_SIZE:-$CP_SIZE}"
for value in "$CP_SIZE" "$BP_SIZE"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    echo "parallel sizes must be positive integers" >&2
    exit 2
  }
done
if (( SYSTEM_TRACE )); then
  for value in "$SYSTEM_TRACE_START_STEP" "$SYSTEM_TRACE_STEPS"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
      echo "system trace step values must be positive integers" >&2
      exit 2
    }
  done
  case "$SYSTEM_TRACE_BACKEND" in
    kineto) ;;
    nsys)
      command -v nsys >/dev/null || {
        echo "the nsys trace backend requires the Nsight Systems CLI" >&2
        exit 2
      }
      ;;
    *) echo "system trace backend must be kineto or nsys" >&2; exit 2 ;;
  esac
fi
CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"

for arg in "${COMMON_ARGS[@]}"; do
  case "$arg" in
    --config|--config=*|--context-parallel-size|--context-parallel-size=*|\
    --block-parallel-size|--block-parallel-size=*|\
    --replicate-clean-prefix|--no-replicate-clean-prefix|\
    --system-trace|--no-system-trace|--system-trace-backend|\
    --system-trace-backend=*|--system-trace-dir|--system-trace-dir=*|\
    --system-trace-start-step|\
    --system-trace-start-step=*|--system-trace-steps|--system-trace-steps=*|\
    --run-dir|--run-dir=*|--training-module|--training-module=*)
      echo "block profile owns execution and output option: $arg" >&2
      exit 2
      ;;
  esac
done

IFS=',' read -r -a requested_modes <<<"$MODES"
modes_to_run=()
seen=","
for mode in "${requested_modes[@]}"; do
  case "$mode" in
    cp|bp|fused) ;;
    *) echo "unsupported block profile mode: $mode" >&2; exit 2 ;;
  esac
  [[ "$seen" == *",$mode,"* ]] && continue
  modes_to_run+=("$mode")
  seen+="$mode,"
done
(( ${#modes_to_run[@]} > 0 )) || { echo "--modes is empty" >&2; exit 2; }

RUN_ROOT="${RUN_ROOT:-$REPO_DIR/runs/block_parallel_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_ROOT"
if (( NODE_RANK == 0 )); then
  printf 'mode,status,exit_code\n' >"$RUN_ROOT/status.csv"
fi

run_case() {
  local mode="$1" cp bp case_dir log_path exit_code status trace_dir trace_output
  local -a topology_args=()
  local -a train_command=()
  case "$mode" in
    cp)
      cp="$CP_SIZE"
      bp=1
      ;;
    bp)
      cp=1
      bp="$BP_SIZE"
      topology_args+=(--replicate-clean-prefix)
      ;;
    fused)
      cp="$CP_SIZE"
      bp="$BP_SIZE"
      ;;
  esac
  case_dir="$RUN_ROOT/$mode"
  log_path="$case_dir/train.log"
  mkdir -p "$case_dir"
  if (( NODE_RANK == 0 )); then
    printf '===== %s =====\n' "$mode" | tee "$log_path"
  fi
  train_command=(
    "$REPO_DIR/scripts/launch/torchrun.sh"
    --training-module dllm_parallel.training.profile_entrypoint
    --config "$CONFIG"
    --run-dir "$case_dir"
    "${COMMON_ARGS[@]}"
    --context-parallel-size "$cp"
    --block-parallel-size "$bp"
    "${topology_args[@]}"
  )
  if (( SYSTEM_TRACE )); then
    trace_dir="$case_dir/system_trace"
    trace_output="$trace_dir/node_${NODE_RANK}"
    mkdir -p "$trace_dir"
    train_command+=(
      --system-trace
      --system-trace-backend "$SYSTEM_TRACE_BACKEND"
      --system-trace-dir "$trace_dir"
      --system-trace-start-step "$SYSTEM_TRACE_START_STEP"
      --system-trace-steps "$SYSTEM_TRACE_STEPS"
    )
    if [[ "$SYSTEM_TRACE_BACKEND" == "nsys" ]]; then
      train_command=(
        nsys profile
        --trace=cuda,nvtx,nccl
        --nccl-trace=all
        --capture-range=cudaProfilerApi
        --capture-range-end=stop
        --sample=none
        --cpuctxsw=none
        --stats=false
        --force-overwrite=true
        --output "$trace_output"
        "${train_command[@]}"
      )
    fi
  fi
  set +e
  if (( NODE_RANK == 0 )); then
    "${train_command[@]}" 2>&1 | tee -a "$log_path"
    exit_code="${PIPESTATUS[0]}"
  else
    "${train_command[@]}"
    exit_code="$?"
  fi
  set -e
  status=ok
  (( exit_code == 0 )) || status=failed
  if (( NODE_RANK == 0 )); then
    printf '%s,%s,%s\n' "$mode" "$status" "$exit_code" >>"$RUN_ROOT/status.csv"
  fi
  return "$exit_code"
}

overall_status=0
for mode in "${modes_to_run[@]}"; do
  run_case "$mode" || overall_status=1
done
if (( NODE_RANK == 0 )); then
  python "$SCRIPT_DIR/parse_profile_results.py" "$RUN_ROOT" | tee "$RUN_ROOT/results.md"
  echo "Profile artifacts: $RUN_ROOT"
fi
exit "$overall_status"
