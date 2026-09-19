#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/build/generate_sbom.sh --artifact PATH [--output PATH]

Generates a CycloneDX JSON software bill of materials with Syft.
EOF
}

ARTIFACT=""
OUTPUT=""
while (($#)); do
  case "$1" in
    --artifact) [[ $# -ge 2 ]] || exit 2; ARTIFACT="$2"; shift 2 ;;
    --artifact=*) ARTIFACT="${1#--artifact=}"; shift ;;
    --output) [[ $# -ge 2 ]] || exit 2; OUTPUT="$2"; shift 2 ;;
    --output=*) OUTPUT="${1#--output=}"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -n "$ARTIFACT" && -f "$ARTIFACT" ]] || { echo "--artifact must name a file" >&2; exit 2; }
command -v syft >/dev/null || { echo "syft is required" >&2; exit 2; }
OUTPUT="${OUTPUT:-$ARTIFACT.cdx.json}"
exec syft "$ARTIFACT" -o "cyclonedx-json=$OUTPUT"
