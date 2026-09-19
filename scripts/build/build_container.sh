#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/build/build_container.sh \
    --base-image IMAGE@sha256:DIGEST \
    --wheel-dir PATH \
    --image IMAGE[:TAG] \
    --source-revision GIT_SHA \
    [--python PATH] [--push]

Builds the CUDA training image from a digest-pinned base and a verified set of
coordinated Turbo-dLLM, FA3, and FA4 wheels. It never resolves native
packages from an index.
EOF
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_IMAGE=""
WHEEL_DIR=""
IMAGE=""
SOURCE_REVISION=""
PY="${PY:-python}"
PUSH=0
while (($#)); do
  case "$1" in
    --base-image) [[ $# -ge 2 ]] || exit 2; BASE_IMAGE="$2"; shift 2 ;;
    --base-image=*) BASE_IMAGE="${1#--base-image=}"; shift ;;
    --wheel-dir) [[ $# -ge 2 ]] || exit 2; WHEEL_DIR="$2"; shift 2 ;;
    --wheel-dir=*) WHEEL_DIR="${1#--wheel-dir=}"; shift ;;
    --image) [[ $# -ge 2 ]] || exit 2; IMAGE="$2"; shift 2 ;;
    --image=*) IMAGE="${1#--image=}"; shift ;;
    --source-revision) [[ $# -ge 2 ]] || exit 2; SOURCE_REVISION="$2"; shift 2 ;;
    --source-revision=*) SOURCE_REVISION="${1#--source-revision=}"; shift ;;
    --python) [[ $# -ge 2 ]] || exit 2; PY="$2"; shift 2 ;;
    --python=*) PY="${1#--python=}"; shift ;;
    --push) PUSH=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$BASE_IMAGE" == *@sha256:* ]] || {
  echo "--base-image must include an immutable @sha256: digest" >&2
  exit 2
}
[[ -d "$WHEEL_DIR" ]] || { echo "--wheel-dir must be a directory" >&2; exit 2; }
[[ -n "$IMAGE" ]] || { echo "--image is required" >&2; exit 2; }
[[ "$SOURCE_REVISION" =~ ^[0-9a-f]{40}$ ]] || {
  echo "--source-revision must be a full Git commit" >&2
  exit 2
}
command -v docker >/dev/null || { echo "docker is required" >&2; exit 2; }

"$ROOT/scripts/verify/artifacts.sh" --directory "$WHEEL_DIR" --python "$PY"

CONTAINER_STAGING="$(mktemp -d)"
trap 'rm -rf "$CONTAINER_STAGING"' EXIT
mkdir -p "$CONTAINER_STAGING/wheels"
cp "$ROOT/containers/cuda/Containerfile" "$CONTAINER_STAGING/Containerfile"
cp "$ROOT/constraints/gpu-cu128.txt" "$CONTAINER_STAGING/gpu-cu128.txt"
cp "$WHEEL_DIR"/*.whl "$CONTAINER_STAGING/wheels/"

BUILD_ARGS=(
  docker buildx build
  --file "$CONTAINER_STAGING/Containerfile"
  --build-arg "BASE_IMAGE=$BASE_IMAGE"
  --build-arg "SOURCE_REVISION=$SOURCE_REVISION"
  --label "org.opencontainers.image.revision=$SOURCE_REVISION"
  --tag "$IMAGE"
)
if (( PUSH )); then
  BUILD_ARGS+=(--push)
else
  BUILD_ARGS+=(--load)
fi
BUILD_ARGS+=("$CONTAINER_STAGING")
exec "${BUILD_ARGS[@]}"
