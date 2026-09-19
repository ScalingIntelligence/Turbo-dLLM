#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/verify/vllm_dflash2.sh --target MODEL --draft DIR --prompt TEXT --expected FILE [options]

Launch native vLLM DFlash2 serving and require one fully accepted draft block.

Options:
  --target MODEL              Target/verifier model path or ID.
  --draft DIR                 Exported Turbo-dLLM DFlash2 directory.
  --prompt TEXT               Deterministic completion prompt.
  --expected FILE             Expected target continuation.
  --model NAME                Served model name (default: target model).
  --tokenizer MODEL           Tokenizer path or ID (default: target model).
  --tensor-parallel-size N    vLLM tensor-parallel size (default: 1).
  --max-model-len N           Optional vLLM maximum model length.
  --gpu-memory-utilization F  Optional vLLM memory utilization in (0, 1].
  --max-tokens N              Verification generation length (default: 128).
  --port N                    Local server port (default: 8000).
  --trust-remote-code         Allow custom model/tokenizer Python code.
EOF
}

if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then
  usage
  exit 0
fi

target=""
draft=""
prompt=""
expected=""
model=""
tokenizer=""
tensor_parallel_size=1
max_model_len=""
gpu_memory_utilization=""
max_tokens=128
port=8000
trust_remote_code=0

while (($#)); do
  case "$1" in
    --target) target="${2:?missing value for --target}"; shift 2 ;;
    --draft) draft="${2:?missing value for --draft}"; shift 2 ;;
    --prompt) prompt="${2:?missing value for --prompt}"; shift 2 ;;
    --expected) expected="${2:?missing value for --expected}"; shift 2 ;;
    --model) model="${2:?missing value for --model}"; shift 2 ;;
    --tokenizer) tokenizer="${2:?missing value for --tokenizer}"; shift 2 ;;
    --tensor-parallel-size)
      tensor_parallel_size="${2:?missing value for --tensor-parallel-size}"
      shift 2
      ;;
    --max-model-len) max_model_len="${2:?missing value for --max-model-len}"; shift 2 ;;
    --gpu-memory-utilization)
      gpu_memory_utilization="${2:?missing value for --gpu-memory-utilization}"
      shift 2
      ;;
    --max-tokens) max_tokens="${2:?missing value for --max-tokens}"; shift 2 ;;
    --port) port="${2:?missing value for --port}"; shift 2 ;;
    --trust-remote-code) trust_remote_code=1; shift ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$target" || -z "$draft" || -z "$prompt" || -z "$expected" ]]; then
  usage >&2
  exit 2
fi
model="${model:-$target}"
tokenizer="${tokenizer:-$target}"

serve=(
  dllm dflash serve-vllm
  --target "$target"
  --draft "$draft"
  --host 127.0.0.1
  --port "$port"
  --tensor-parallel-size "$tensor_parallel_size"
  --served-model-name "$model"
)
verify=(
  dllm dflash verify-vllm
  --server-url "http://127.0.0.1:${port}"
  --model "$model"
  --tokenizer "$tokenizer"
  --draft "$draft"
  --prompt "$prompt"
  --expected-text "$expected"
  --max-tokens "$max_tokens"
)
[[ -z "$max_model_len" ]] || serve+=(--max-model-len "$max_model_len")
[[ -z "$gpu_memory_utilization" ]] || serve+=(--gpu-memory-utilization "$gpu_memory_utilization")
if [[ "$trust_remote_code" == 1 ]]; then
  serve+=(--trust-remote-code)
  verify+=(--trust-remote-code)
fi

"${serve[@]}" &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true' EXIT

ready=0
for _ in {1..360}; do
  if python -c \
    'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=2).read()' \
    "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    wait "$server_pid"
  fi
  sleep 2
done
if [[ "$ready" != 1 ]]; then
  echo "vLLM did not become ready at http://127.0.0.1:${port}" >&2
  exit 1
fi

"${verify[@]}"
