#!/usr/bin/env bash
set -euo pipefail
usage() {
  cat <<'EOF'
Usage: scripts/verify/sglang_dflash2.sh --target MODEL --draft DIR --prompt TEXT --expected FILE [options]
Launch a real SGLang DFLASH server and require one deterministically accepted block.
Options:
  --target MODEL   Target/verifier model path or ID.
  --draft DIR      Exported DFlash2 draft directory.
  --prompt TEXT    Deterministic completion prompt.
  --expected FILE  File containing the expected target continuation.
  --block-size N   Draft block size (default: 16).
  --port N         Server port (default: 30000).
  --trust-remote-code  Explicitly allow custom model/tokenizer Python code.
EOF
}
if [[ ${1:-} == "--help" || $# -eq 0 ]]; then usage; exit 0; fi
target="" draft="" prompt="" expected="" block_size=16 port=30000 trust_remote_code=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) target="$2"; shift 2 ;;
    --draft) draft="$2"; shift 2 ;;
    --prompt) prompt="$2"; shift 2 ;;
    --expected) expected="$2"; shift 2 ;;
    --block-size) block_size="$2"; shift 2 ;;
    --port) port="$2"; shift 2 ;;
    --trust-remote-code) trust_remote_code=1; shift ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -n "$target" && -n "$draft" && -n "$prompt" && -n "$expected" ]] || { usage >&2; exit 2; }
python - <<'PY'
from importlib.metadata import PackageNotFoundError, version
try:
    installed = version("sglang")
except PackageNotFoundError as error:
    raise SystemExit("install the qualified runtime with: python -m pip install 'turbo-dllm[sglang]'") from error
if installed != "0.5.20":
    raise SystemExit(f"qualified SGLang version is 0.5.20, found {installed}")
PY
server_trust=()
cli_trust=()
if [[ "$trust_remote_code" == 1 ]]; then
  server_trust=(--trust-remote-code)
  cli_trust=(--trust-remote-code)
fi
python -m sglang.launch_server --model-path "$target" \
  --speculative-algorithm DFLASH --speculative-draft-model-path "$draft" \
  --speculative-dflash-block-size "$block_size" --port "$port" "${server_trust[@]}" &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT
for _ in $(seq 1 180); do
  curl -fsS "http://127.0.0.1:${port}/server_info" >/dev/null && break
  sleep 2
done
curl -fsS "http://127.0.0.1:${port}/server_info" >/dev/null
dllm dflash verify-sglang --server-url "http://127.0.0.1:${port}" \
  --model "$target" --tokenizer "$target" --prompt "$prompt" \
  --expected-text "$expected" --block-size "$block_size" --max-tokens "$block_size" \
  "${cli_trust[@]}"
