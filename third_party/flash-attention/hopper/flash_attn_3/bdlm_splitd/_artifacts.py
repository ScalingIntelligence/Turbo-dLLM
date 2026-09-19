# Copyright 2026 The dllm_parallel Authors.
# SPDX-License-Identifier: Apache-2.0

"""Verified AOT artifact registry for CuTe Split-D kernels.

Production processes only load artifacts packaged by the FlashAttention wheel.
Compilation is enabled explicitly by the wheel builder or by developer mode;
there are no environment-controlled cache paths or first-use production JITs.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import hashlib
import json
import os
import pickle
from pathlib import Path
import shlex
import shutil
import subprocess
import sysconfig
import threading
from typing import Any, Hashable, Iterator, TypeAlias

import torch
import tvm_ffi

from ._capabilities import (
  ARCHITECTURE,
  EXPECTED_ARTIFACT_COUNT,
  backward_variant_keys,
  forward_variant_keys,
  manifest_capabilities,
)


CompileKeyType: TypeAlias = tuple[Hashable, ...]
CallableFunction: TypeAlias = Any

_ARTIFACT_FORMAT = "bdlm.splitd.aot.v2"
_ARTIFACT_ROOT = Path(__file__).resolve().parent / "_artifacts"
_MANIFEST_NAME = "manifest.json"
_FUNCTION_NAME = "func"
_LOAD_LOCK = threading.RLock()
_MODULE_CACHE: dict[str, object] = {}
_runtime_jit_enabled = False
_build_root: Path | None = None
_ARTIFACT_ROOT_TO_SITE_PACKAGES = 3


def configure_runtime_jit(*, allow_runtime_jit: bool) -> None:
  """Set the explicit developer-only compilation policy."""

  global _runtime_jit_enabled
  _runtime_jit_enabled = bool(allow_runtime_jit)


def is_aot_build() -> bool:
  return _build_root is not None


def runtime_jit_enabled() -> bool:
  return _runtime_jit_enabled


@contextmanager
def aot_build(output_root: Path) -> Iterator[None]:
  """Enable artifact emission for an explicit wheel/image build."""

  global _build_root, _runtime_jit_enabled
  previous_root = _build_root
  previous_jit = _runtime_jit_enabled
  _build_root = Path(output_root).resolve()
  _runtime_jit_enabled = True
  try:
    yield
  finally:
    _build_root = previous_root
    _runtime_jit_enabled = previous_jit


@lru_cache(maxsize=1)
def source_fingerprint() -> str:
  """Hash kernel sources and the ABI inputs used by exported artifacts."""

  root = Path(__file__).resolve().parent
  digest = hashlib.sha256()
  abi = (
    f"torch={torch.__version__};cuda={torch.version.cuda};"
    f"tvm_ffi={tvm_ffi.__version__};"
    f"platform={sysconfig.get_platform()}"
  )
  digest.update(abi.encode("utf-8"))
  for source in sorted(root.rglob("*.py")):
    if "__pycache__" in source.parts:
      continue
    relative = source.relative_to(root).as_posix()
    content = source.read_bytes()
    digest.update(relative.encode("utf-8"))
    digest.update(len(content).to_bytes(8, "little"))
    digest.update(content)
  return digest.hexdigest()


def _key_hash(key: CompileKeyType) -> str:
  return hashlib.sha256(pickle.dumps(key)).hexdigest()


def _artifact_base(root: Path | None = None) -> Path:
  return (root or _ARTIFACT_ROOT) / source_fingerprint()


def _manifest_path(root: Path | None = None) -> Path:
  return _artifact_base(root) / _MANIFEST_NAME


def _load_manifest(root: Path | None = None) -> dict[str, object]:
  path = _manifest_path(root)
  try:
    payload = json.loads(path.read_text(encoding="utf-8"))
  except FileNotFoundError:
    return {}
  if payload.get("format") != _ARTIFACT_FORMAT:
    raise RuntimeError(f"invalid Split-D artifact manifest format: {path}")
  if payload.get("source_fingerprint") != source_fingerprint():
    raise RuntimeError(f"stale Split-D artifact manifest: {path}")
  artifacts = payload.get("artifacts")
  if not isinstance(artifacts, dict):
    raise RuntimeError(f"invalid Split-D artifact inventory: {path}")
  return payload


@lru_cache(maxsize=1)
def verify_packaged_artifacts() -> dict[str, object]:
  """Verify the complete immutable artifact inventory before CUDA launch."""

  manifest = _load_manifest()
  if not manifest:
    raise RuntimeError(
      "packaged BDLM Split-D artifact manifest is missing; rebuild the "
      "bdlm-flash-attn-3 wheel"
    )
  if manifest.get("complete") is not True:
    raise RuntimeError("packaged BDLM Split-D artifact build is incomplete")
  if manifest.get("capabilities") != manifest_capabilities():
    raise RuntimeError("packaged BDLM Split-D capability matrix is incompatible")
  if manifest.get("expected_artifact_count") != EXPECTED_ARTIFACT_COUNT:
    raise RuntimeError("packaged BDLM Split-D artifact count contract is invalid")
  inventory = manifest["artifacts"]
  if len(inventory) != EXPECTED_ARTIFACT_COUNT:
    raise RuntimeError(
      "packaged BDLM Split-D specialization matrix is incomplete: "
      f"expected {EXPECTED_ARTIFACT_COUNT} artifacts, found {len(inventory)}"
    )
  variants = manifest.get("variants")
  if not isinstance(variants, dict):
    raise RuntimeError("packaged BDLM Split-D variant registry is missing")
  forward_variants = variants.get("forward")
  backward_variants = variants.get("backward")
  if not isinstance(forward_variants, dict) or set(forward_variants) != set(
    forward_variant_keys()
  ):
    raise RuntimeError("packaged BDLM Split-D forward variants are incomplete")
  if not isinstance(backward_variants, dict) or set(backward_variants) != set(
    backward_variant_keys()
  ):
    raise RuntimeError("packaged BDLM Split-D backward variants are incomplete")
  referenced = set(forward_variants.values())
  for entry in backward_variants.values():
    if not isinstance(entry, dict):
      raise RuntimeError("invalid packaged BDLM Split-D backward variant")
    referenced.update(entry.values())
  if referenced != set(inventory):
    raise RuntimeError(
      "packaged BDLM Split-D variant registry does not cover its artifacts"
    )
  base = _artifact_base().resolve()
  for relative_name, expected_digest in inventory.items():
    relative = Path(relative_name)
    if relative.is_absolute() or ".." in relative.parts:
      raise RuntimeError("invalid path in Split-D artifact inventory")
    path = (base / relative).resolve()
    if base not in path.parents or not path.is_file():
      raise RuntimeError(f"packaged Split-D artifact is missing: {path}")
    observed_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed_digest != expected_digest:
      raise RuntimeError(f"packaged Split-D artifact hash mismatch: {path}")
  return manifest


def _write_manifest(
  root: Path,
  artifacts: dict[str, str],
  *,
  complete: bool = False,
  variants: dict[str, dict[str, object]] | None = None,
) -> None:
  from importlib.metadata import version as distribution_version

  base = _artifact_base(root)
  base.mkdir(parents=True, exist_ok=True)
  payload = {
    "format": _ARTIFACT_FORMAT,
    "source_fingerprint": source_fingerprint(),
    "architecture": ARCHITECTURE,
    "capabilities": manifest_capabilities(),
    "complete": bool(complete),
    "expected_artifact_count": EXPECTED_ARTIFACT_COUNT,
    "torch": str(torch.__version__),
    "torch_cuda": str(torch.version.cuda),
    "cutlass": distribution_version("nvidia-cutlass-dsl"),
    "tvm_ffi": str(tvm_ffi.__version__),
    "artifacts": dict(sorted(artifacts.items())),
    "variants": variants or {},
  }
  path = base / _MANIFEST_NAME
  temporary = path.with_suffix(".tmp")
  temporary.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )
  temporary.replace(path)


def _cutlass_static_runtime() -> Path:
  import cutlass

  package_path = Path(cutlass.__file__).resolve()
  library_names = (
    "libcuda_dialect_runtime_static.a",
    "libcute_dsl_runtime_static.a",
  )
  for parent in package_path.parents:
    for library_name in library_names:
      candidate = parent / "lib" / library_name
      if candidate.is_file():
        return candidate
  raise RuntimeError(
    "CUTLASS DSL static runtime is missing; install the production "
    "nvidia-cutlass-dsl-libs-base package before building Split-D artifacts"
  )


def _runtime_library_rpath(output_path: Path, artifact_root: Path) -> str:
  """Return the relocatable path from an artifact to a sibling wheel package."""

  relative = output_path.resolve().relative_to(artifact_root.resolve())
  parent_depth = len(relative.parent.parts) + _ARTIFACT_ROOT_TO_SITE_PACKAGES
  components = [".."] * parent_depth + ["tvm_ffi", "lib"]
  return "$ORIGIN/" + "/".join(components)


def _link_artifact(object_path: Path, output_path: Path) -> None:
  """Link a CuTe export without process-global runtime-library preloads."""

  static_runtime = _cutlass_static_runtime()
  tvm_runtime = Path(tvm_ffi.libinfo.find_libtvm_ffi()).resolve()
  nvcc = shutil.which("nvcc")
  if nvcc is None:
    raise RuntimeError("nvcc is required to link Split-D AOT artifacts")
  cuda_root = Path(nvcc).resolve().parent.parent
  cuda_runtime_candidates = (
    cuda_root / "lib64/libcudart.so",
    cuda_root / "lib/libcudart.so",
  )
  cuda_runtime = next(
    (candidate for candidate in cuda_runtime_candidates if candidate.is_file()),
    None,
  )
  if cuda_runtime is None:
    raise RuntimeError(f"CUDA runtime library is missing under {cuda_root}")
  if _build_root is None:
    raise RuntimeError("Split-D artifacts may only be linked during an AOT build")
  compiler = shlex.split(
    os.environ.get("CXX") or sysconfig.get_config_var("CXX") or "c++"
  )
  if shutil.which(compiler[0]) is None:
    compiler = ["c++"]
  output_path.parent.mkdir(parents=True, exist_ok=True)
  command = [
    *compiler,
    "-shared",
    "-Wl,--no-undefined",
    str(object_path),
    str(static_runtime),
    str(cuda_runtime),
    str(tvm_runtime),
    f"-Wl,-rpath,{_runtime_library_rpath(output_path, _build_root)}",
    "-ldl",
    "-lpthread",
    "-o",
    str(output_path),
  ]
  subprocess.run(command, check=True)


class AOTArtifactCache:
  """In-memory callable cache backed by verified package artifacts."""

  def __init__(self, name: str):
    self.name = name
    self.cache: dict[CompileKeyType, CallableFunction] = {}
    self.last_key: CompileKeyType | None = None

  def __setitem__(self, key: CompileKeyType, fn: CallableFunction) -> None:
    self.last_key = key
    self.cache[key] = fn
    if _build_root is not None:
      self._export(key, fn, _build_root)

  def __getitem__(self, key: CompileKeyType) -> CallableFunction:
    if key not in self.cache:
      self._load(key)
    self.last_key = key
    return self.cache[key]

  def __contains__(self, key: CompileKeyType) -> bool:
    if key in self.cache:
      self.last_key = key
      return True
    if self._load(key):
      return True
    if not _runtime_jit_enabled:
      raise RuntimeError(
        "packaged BDLM Split-D artifact is missing for the requested kernel "
        f"variant ({self.name}/{_key_hash(key)}). Rebuild the "
        "bdlm-flash-attn-3 wheel; production training never compiles kernels."
      )
    return False

  def clear(self) -> None:
    self.cache.clear()

  def _relative_path(self, key: CompileKeyType) -> Path:
    return Path(self.name) / f"{_key_hash(key)}.so"

  def last_relative_path(self) -> str:
    if self.last_key is None:
      raise RuntimeError(f"Split-D cache {self.name} has no compiled variant")
    return self._relative_path(self.last_key).as_posix()

  def _load(self, key: CompileKeyType) -> bool:
    relative = self._relative_path(key)
    manifest = _load_manifest()
    inventory = manifest.get("artifacts") if manifest else None
    if not isinstance(inventory, dict) or relative.as_posix() not in inventory:
      return False
    path = _artifact_base() / relative
    if not path.is_file():
      raise RuntimeError(f"packaged Split-D artifact is missing: {path}")
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = inventory[relative.as_posix()]
    if observed != expected:
      raise RuntimeError(f"packaged Split-D artifact hash mismatch: {path}")
    with _LOAD_LOCK:
      module = tvm_ffi.load_module(str(path), keep_module_alive=True)
      self.cache[key] = getattr(module, _FUNCTION_NAME)
      self.last_key = key
    return True

  def _export(
    self,
    key: CompileKeyType,
    fn: CallableFunction,
    root: Path,
  ) -> None:
    relative = self._relative_path(key)
    destination = _artifact_base(root) / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    object_path = destination.with_suffix(".o")
    fn.export_to_c(
      object_file_path=str(object_path),
      function_name=_FUNCTION_NAME,
    )
    try:
      _link_artifact(object_path, destination)
    finally:
      object_path.unlink(missing_ok=True)
    manifest = _load_manifest(root)
    inventory = dict(manifest.get("artifacts", {})) if manifest else {}
    inventory[relative.as_posix()] = hashlib.sha256(
      destination.read_bytes()
    ).hexdigest()
    variants = dict(manifest.get("variants", {})) if manifest else {}
    _write_manifest(root, inventory, variants=variants)


def reset_aot_artifacts(output_root: Path) -> None:
  root = Path(output_root).resolve()
  if root.exists():
    shutil.rmtree(root)


def finalize_aot_artifacts(
  output_root: Path,
  variants: dict[str, dict[str, object]],
) -> None:
  """Seal an AOT build only after its full specialization matrix exists."""

  root = Path(output_root).resolve()
  manifest = _load_manifest(root)
  inventory = manifest.get("artifacts") if manifest else None
  if not isinstance(inventory, dict):
    raise RuntimeError("Split-D AOT build produced no artifact inventory")
  if len(inventory) != EXPECTED_ARTIFACT_COUNT:
    raise RuntimeError(
      "Split-D AOT build is incomplete: "
      f"expected {EXPECTED_ARTIFACT_COUNT} artifacts, found {len(inventory)}"
    )
  if set(variants.get("forward", {})) != set(forward_variant_keys()):
    raise RuntimeError("Split-D AOT forward variant registry is incomplete")
  if set(variants.get("backward", {})) != set(backward_variant_keys()):
    raise RuntimeError("Split-D AOT backward variant registry is incomplete")
  referenced = set(variants["forward"].values())
  for entry in variants["backward"].values():
    if not isinstance(entry, dict):
      raise RuntimeError("invalid Split-D backward variant entry")
    referenced.update(entry.values())
  if referenced != set(inventory):
    raise RuntimeError("Split-D variant registry does not cover its artifact inventory")
  _write_manifest(root, inventory, complete=True, variants=variants)


def load_packaged_variant(
  phase: str,
  variant: str,
) -> dict[str, CallableFunction]:
  """Load one verified semantic variant without importing the CuTe compiler."""

  manifest = verify_packaged_artifacts()
  variants = manifest["variants"]
  phase_variants = variants.get(phase)
  if not isinstance(phase_variants, dict) or variant not in phase_variants:
    raise RuntimeError(f"unsupported Split-D {phase} variant: {variant}")
  entry = phase_variants[variant]
  paths = {"forward": entry} if isinstance(entry, str) else entry
  if not isinstance(paths, dict):
    raise RuntimeError(f"invalid Split-D {phase} variant: {variant}")
  functions: dict[str, CallableFunction] = {}
  base = _artifact_base()
  with _LOAD_LOCK:
    for name, relative_name in paths.items():
      if not isinstance(relative_name, str):
        raise RuntimeError(f"invalid Split-D artifact path for {variant}")
      if relative_name not in manifest["artifacts"]:
        raise RuntimeError(f"unregistered Split-D artifact for {variant}")
      path = base / relative_name
      module = _MODULE_CACHE.get(relative_name)
      if module is None:
        module = tvm_ffi.load_module(str(path), keep_module_alive=True)
        _MODULE_CACHE[relative_name] = module
      functions[str(name)] = getattr(module, _FUNCTION_NAME)
  return functions


def get_jit_cache(name: str | None = None) -> AOTArtifactCache:
  if not name:
    raise ValueError("Split-D artifact caches require a stable kernel name")
  return AOTArtifactCache(name)


__all__ = [
  "AOTArtifactCache",
  "aot_build",
  "configure_runtime_jit",
  "finalize_aot_artifacts",
  "get_jit_cache",
  "is_aot_build",
  "load_packaged_variant",
  "reset_aot_artifacts",
  "runtime_jit_enabled",
  "source_fingerprint",
  "verify_packaged_artifacts",
]
