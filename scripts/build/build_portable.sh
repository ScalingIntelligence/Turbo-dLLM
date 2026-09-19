#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/build/build_portable.sh [--python PATH] [--output-dir PATH]

Builds the portable Turbo-dLLM wheel and source distribution from a clean,
committed Git snapshot. Ignored build directories and stale package metadata
from the working copy can never enter the resulting artifacts.
EOF
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-python}"
OUTPUT_DIR="$ROOT/dist"
while (($#)); do
  case "$1" in
    --python) [[ $# -ge 2 ]] || exit 2; PY="$2"; shift 2 ;;
    --python=*) PY="${1#--python=}"; shift ;;
    --output-dir) [[ $# -ge 2 ]] || exit 2; OUTPUT_DIR="$2"; shift 2 ;;
    --output-dir=*) OUTPUT_DIR="${1#--output-dir=}"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -z "$(git -C "$ROOT" status --porcelain)" ]] || {
  echo "portable artifact builds require a clean Git tree" >&2
  exit 1
}

PY="$("$PY" -c 'import sys; print(sys.executable)')"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
STAGING_DIR="$(mktemp -d)"
trap 'rm -rf "$STAGING_DIR"' EXIT
SOURCE_DIR="$STAGING_DIR/source"
ARTIFACT_DIR="$STAGING_DIR/artifacts"
mkdir -p "$SOURCE_DIR" "$ARTIFACT_DIR"

git -C "$ROOT" archive --format=tar HEAD | tar -C "$SOURCE_DIR" -xf -
(
  cd "$STAGING_DIR"
  "$PY" -m build --sdist --wheel --outdir "$ARTIFACT_DIR" "$SOURCE_DIR"
)
cp "$ARTIFACT_DIR"/*.tar.gz "$ARTIFACT_DIR"/*.whl "$OUTPUT_DIR"/
printf 'Portable artifacts written to %s\n' "$OUTPUT_DIR"
