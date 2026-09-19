#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/install/install_gpu_bundle.sh (--release VERSION --auto | --manifest URL_OR_PATH)

Compatibility wrapper for the installed hash-verifying bundle installer.
Use PY=/path/to/python to select a specific environment.
EOF
}

[[ "${1:-}" != -h && "${1:-}" != --help ]] || { usage; exit 0; }
exec "${PY:-python}" -m dllm_parallel.cli bundle install "$@"
