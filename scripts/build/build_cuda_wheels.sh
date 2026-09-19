#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/build/build_cuda_wheels.sh \
    --cuda-arch-list ARCHES \
    [--python PATH] \
    [--output-dir PATH]

Builds the three production wheel artifacts from the current checkout:
  - bdlm-flash-attn-3 from the vendored Hopper sources;
  - flash-attn-4 from the vendored CuTe sources;
  - Turbo-dLLM with all manifest-verified native utility kernels.

ARCHES uses PyTorch's explicit architecture-list syntax, for example:
  8.9;9.0

Install the kernel-build extra before invoking this script. The build never
uses runtime JIT and does not infer a release architecture from the host GPU.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-python}"
OUTPUT_DIR="$ROOT/dist"
CUDA_ARCH_LIST=""

while (($#)); do
  case "$1" in
    --cuda-arch-list)
      [[ $# -ge 2 ]] || { echo "--cuda-arch-list requires a value" >&2; exit 2; }
      CUDA_ARCH_LIST="$2"
      shift 2
      ;;
    --cuda-arch-list=*)
      CUDA_ARCH_LIST="${1#--cuda-arch-list=}"
      shift
      ;;
    --python)
      [[ $# -ge 2 ]] || { echo "--python requires a value" >&2; exit 2; }
      PYTHON="$2"
      shift 2
      ;;
    --python=*)
      PYTHON="${1#--python=}"
      shift
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || { echo "--output-dir requires a value" >&2; exit 2; }
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --output-dir=*)
      OUTPUT_DIR="${1#--output-dir=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$CUDA_ARCH_LIST" ]]; then
  echo "--cuda-arch-list is required for reproducible production wheels" >&2
  exit 2
fi

if [[ -n "$(git -C "$ROOT" status --porcelain)" ]]; then
  echo "CUDA wheel builds require a clean Git tree" >&2
  exit 1
fi

PYTHON="$("$PYTHON" -c 'import sys; print(sys.executable)')"

FA3_DISABLE_SM8X="$($PYTHON - "$CUDA_ARCH_LIST" <<'PY'
import re
import sys

architectures = tuple(
    item
    for item in re.split(r"[;,\s]+", sys.argv[1].strip())
    if item
)
supported = {"8.0", "8.9", "9.0"}
invalid = sorted(set(architectures).difference(supported))
if not architectures or invalid:
    raise SystemExit(
        "production BDLM FlashAttention supports CUDA architectures "
        f"{sorted(supported)}; unsupported entries: {invalid or architectures}"
    )
print("FALSE" if any(item.startswith("8.") for item in architectures) else "TRUE")
PY
)"

PACKAGE_VERSION="$($PYTHON - "$ROOT" <<'PY'
import pathlib
import sys
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

root = pathlib.Path(sys.argv[1])
print(tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"])
PY
)"
SOURCE_REVISION="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || printf 'unknown')"

mkdir -p "$OUTPUT_DIR"
STAGING_DIR="$(mktemp -d)"
trap 'rm -rf "$STAGING_DIR"' EXIT
BUILD_ROOT="$STAGING_DIR/source"
WHEEL_DIR="$STAGING_DIR/wheels"
mkdir -p "$BUILD_ROOT" "$WHEEL_DIR"

# Build from an immutable, generated-artifact-free source snapshot. This keeps
# release wheels independent of editable installs and stale local build trees.
tar \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.py[co]' \
  --exclude='*.so' \
  --exclude='*.pyd' \
  --exclude='*.dylib' \
  --exclude='*.egg-info' \
  --exclude='third_party/flash-attention/hopper/build' \
  --exclude='third_party/flash-attention/flash_attn/cute/build' \
  --exclude='dist' \
  --exclude='native_kernels.json' \
  -C "$ROOT" -cf - \
  pyproject.toml setup.py README.md LICENSE NOTICE dllm_parallel scripts \
  third_party/flash-attention/README.md \
  third_party/flash-attention/flash_attn/cute \
  third_party/flash-attention/hopper \
  third_party/flash-attention/csrc/cutlass/include \
  | tar -C "$BUILD_ROOT" -xf -

(
  cd "$BUILD_ROOT"
  TORCH_EXTENSIONS_DIR="$STAGING_DIR/torch_extensions" \
    "$PYTHON" -m dllm_parallel.core.kernels.build \
      --cuda-arch-list "$CUDA_ARCH_LIST" \
      --package-version "$PACKAGE_VERSION"
  "$PYTHON" -m dllm_parallel.core.profiling.release_gates policy --root .
)

(
  # Running from the isolated staging parent prevents an ignored local
  # `build/` directory from shadowing the PyPA build frontend.
  cd "$STAGING_DIR"

  # The production FA3 variant is part of the artifact contract. Set every
  # feature flag explicitly so shell state cannot silently change the wheel.
  env \
    BUILD_TARGET=cuda \
    FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE \
    FLASH_ATTENTION_FORCE_CXX11_ABI=FALSE \
    FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE \
    FLASH_ATTENTION_DISABLE_BACKWARD=FALSE \
    FLASH_ATTENTION_DISABLE_SPLIT=TRUE \
    FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE \
    FLASH_ATTENTION_DISABLE_APPENDKV=TRUE \
    FLASH_ATTENTION_DISABLE_LOCAL=TRUE \
    FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE \
    FLASH_ATTENTION_DISABLE_PACKGQA=TRUE \
    FLASH_ATTENTION_DISABLE_FP16=TRUE \
    FLASH_ATTENTION_DISABLE_FP8=TRUE \
    FLASH_ATTENTION_DISABLE_VARLEN=FALSE \
    FLASH_ATTENTION_DISABLE_CLUSTER=FALSE \
    FLASH_ATTENTION_DISABLE_HDIM64=FALSE \
    FLASH_ATTENTION_DISABLE_HDIM96=TRUE \
    FLASH_ATTENTION_DISABLE_HDIM128=FALSE \
    FLASH_ATTENTION_DISABLE_HDIM192=TRUE \
    FLASH_ATTENTION_DISABLE_HDIM256=FALSE \
    FLASH_ATTENTION_DISABLE_HDIMDIFF64=TRUE \
    FLASH_ATTENTION_DISABLE_HDIMDIFF192=TRUE \
    FLASH_ATTENTION_DISABLE_SM80="$FA3_DISABLE_SM8X" \
    FLASH_ATTENTION_ENABLE_VCOLMAJOR=FALSE \
    FLASH_ATTENTION_FORCE_LEGACY_API=1 \
    FLASH_ATTENTION_USE_STABLE_API=0 \
    BDLM_FA3_SOURCE_REVISION="$SOURCE_REVISION" \
    "$PYTHON" -m build \
      --wheel \
      --no-isolation \
      --skip-dependency-check \
      --outdir "$WHEEL_DIR" \
      "$BUILD_ROOT/third_party/flash-attention/hopper"

  # FA4's CuTe kernels are compiled for the concrete shapes at runtime, but the
  # Python kernel sources themselves are a versioned production artifact. Build
  # them from the same immutable source snapshot as the trainer.
  SETUPTOOLS_SCM_PRETEND_VERSION_FOR_FLASH_ATTN_4="4.0.0b19+bdlm.${SOURCE_REVISION:0:12}" \
    "$PYTHON" -m build \
      --wheel \
      --no-isolation \
      --skip-dependency-check \
      --outdir "$WHEEL_DIR" \
      "$BUILD_ROOT/third_party/flash-attention/flash_attn/cute"

  "$PYTHON" -m build \
    --wheel \
    --no-isolation \
    --skip-dependency-check \
    --outdir "$WHEEL_DIR" \
    "$BUILD_ROOT"
)

(
  cd "$BUILD_ROOT"
  "$PYTHON" -m dllm_parallel.core.profiling.release_gates wheels \
    --directory "$WHEEL_DIR"
)

rm -f \
  "$OUTPUT_DIR"/dllm_parallel-*.whl \
  "$OUTPUT_DIR"/turbo_dllm-*.whl \
  "$OUTPUT_DIR"/bdlm_flash_attn_3-*.whl \
  "$OUTPUT_DIR"/flash_attn_4-*.whl
cp "$WHEEL_DIR"/*.whl "$OUTPUT_DIR"/
printf 'Production wheels written to %s\n' "$OUTPUT_DIR"
