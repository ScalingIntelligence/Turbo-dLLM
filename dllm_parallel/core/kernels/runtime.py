# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Native kernel package/runtime policy."""

from __future__ import annotations

import importlib
import importlib.metadata
import hashlib
import json
import os
import shutil
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


NATIVE_KERNEL_MANIFEST_FORMAT = "dllm_parallel.native_kernels.v1"
NATIVE_KERNEL_DISTRIBUTION = "turbo-dllm"


@dataclass(frozen=True)
class KernelRuntimePolicy:
    allow_runtime_jit: bool = False


_POLICY = KernelRuntimePolicy()


def configure_kernel_runtime(*, allow_runtime_jit: bool) -> None:
    global _POLICY
    _POLICY = KernelRuntimePolicy(allow_runtime_jit=bool(allow_runtime_jit))


def kernel_runtime_policy() -> KernelRuntimePolicy:
    return _POLICY


def verify_packaged_native_kernels(
    *,
    device_capabilities: Sequence[tuple[int, int]] | None = None,
) -> tuple[str, ...]:
    """Eagerly verify every extension in the coordinated GPU package.

    Normal training keeps lazy imports so the hot path is unchanged.  Startup
    diagnostics call this function once before workers are launched, which
    catches mixed source/binary installs and unsupported GPU architectures
    before model allocation.
    """

    # Import torch before any extension so libc10 and the other PyTorch-owned
    # shared libraries are loaded into the process.
    import torch  # noqa: F401

    from dllm_parallel.core.kernels._native_specs import NATIVE_KERNEL_SPECS

    verified: list[str] = []
    for spec in NATIVE_KERNEL_SPECS:
        try:
            module = importlib.import_module(spec.package_module)
        except Exception as exc:
            raise RuntimeError(
                f"native kernel {spec.name!r} could not be imported from the "
                "coordinated GPU bundle. Reinstall one complete Turbo-dLLM "
                "bundle; do not mix extensions from another checkout or release"
            ) from exc
        _verify_native_extension(
            module=module,
            extension_name=spec.name,
            required_symbols=tuple(spec.symbols),
            source_files=tuple(spec.source_paths()),
            require_manifest=True,
        )
        for capability in sorted(set(device_capabilities or ())):
            _verify_manifest_entry(
                module=module,
                extension_name=spec.name,
                source_files=tuple(spec.source_paths()),
                device_capability=tuple(int(value) for value in capability),
            )
        verified.append(spec.name)
    return tuple(verified)


def _cuda_home() -> Path | None:
    from torch.utils.cpp_extension import CUDA_HOME

    return Path(CUDA_HOME) if CUDA_HOME else None


def verify_runtime_jit_toolchain() -> str:
    """Verify the explicit developer JIT path without compiling anything."""

    cuda_home = _cuda_home()
    if cuda_home is None:
        raise RuntimeError(
            "kernel.runtime_jit=true requires a CUDA development toolkit; "
            "PyTorch could not resolve CUDA_HOME"
        )
    nvcc = cuda_home / "bin" / "nvcc"
    if not nvcc.is_file() or not os.access(nvcc, os.X_OK):
        raise RuntimeError(
            "kernel.runtime_jit=true requires an executable nvcc at "
            f"{nvcc}; CUDA runtime-only images cannot compile kernels"
        )
    _verify_jit_host_compiler()
    _verify_jit_ninja()
    _verify_jit_cuda_headers(cuda_home)
    _verify_jit_cache()
    from dllm_parallel.core.kernels._native_specs import NATIVE_KERNEL_SPECS

    missing_sources = [
        str(path)
        for spec in NATIVE_KERNEL_SPECS
        for path in spec.source_paths()
        if not path.is_file()
    ]
    if missing_sources:
        raise RuntimeError(
            "runtime JIT source files are missing: " + ", ".join(missing_sources)
        )
    return str(nvcc)


def _verify_jit_host_compiler() -> str:
    compiler = os.environ.get("CXX", "c++").split()[0]
    resolved = shutil.which(compiler)
    if resolved is None:
        raise RuntimeError(
            f"kernel.runtime_jit=true requires a host C++ compiler; {compiler!r} "
            "was not found on PATH"
        )
    return resolved


def _verify_jit_ninja() -> None:
    from torch.utils.cpp_extension import verify_ninja_availability

    try:
        verify_ninja_availability()
    except RuntimeError as exc:
        raise RuntimeError(
            "kernel.runtime_jit=true requires Ninja; install the kernel-build extra"
        ) from exc


def _verify_jit_cuda_headers(cuda_home: Path) -> None:
    local_header = cuda_home / "include" / "cuda_runtime.h"
    if not local_header.is_file():
        raise RuntimeError(
            "kernel.runtime_jit=true requires CUDA development headers; missing "
            f"{local_header}"
        )
    from dllm_parallel.core.kernels._native_specs import (
        NATIVE_KERNEL_SPECS,
        _find_packaged_cuda_paths,
    )

    if not any(spec.needs_cublas for spec in NATIVE_KERNEL_SPECS):
        return
    local_cublas = cuda_home / "include" / "cublas_v2.h"
    local_libraries = (cuda_home / "lib64", cuda_home / "lib")
    packaged_include, packaged_lib = _find_packaged_cuda_paths()
    if not (
        (local_cublas.is_file() and any(path.is_dir() for path in local_libraries))
        or (packaged_include is not None and packaged_lib is not None)
    ):
        raise RuntimeError(
            "kernel.runtime_jit=true requires cuBLAS development headers and libraries"
        )


def _verify_jit_cache() -> None:
    from torch.utils.cpp_extension import get_default_build_root

    cache = Path(os.environ.get("TORCH_EXTENSIONS_DIR") or get_default_build_root())
    candidate = cache
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if not candidate.is_dir() or not os.access(candidate, os.W_OK):
        raise RuntimeError(
            f"kernel.runtime_jit=true requires a writable extension cache: {cache}"
        )


def load_native_extension(
    *,
    package_module: str,
    extension_name: str,
    jit_builder: Callable[[], Any],
    required_symbols: Sequence[str] = (),
    source_files: Sequence[str | Path] = (),
) -> Any:
    # PyTorch owns the shared libraries (for example libc10) linked by its C++
    # extensions. Import it before dlopen so those dependencies are globally
    # available even when callers use this generic loader directly.
    import torch  # noqa: F401

    if _POLICY.allow_runtime_jit:
        loaded_from_package = False
        module = jit_builder()
    else:
        loaded_from_package = True
        try:
            module = importlib.import_module(package_module)
        except Exception as package_exc:
            raise RuntimeError(
                f"native extension {extension_name!r} is not packaged as "
                f"{package_module!r}, and runtime CUDA JIT is disabled. Install "
                "the dllm_parallel wheel/container with packaged native kernels "
                "or set kernel.runtime_jit=true for developer builds."
            ) from package_exc
    _verify_native_extension(
        module=module,
        extension_name=extension_name,
        required_symbols=tuple(required_symbols),
        source_files=tuple(Path(item) for item in source_files),
        require_manifest=loaded_from_package and not _POLICY.allow_runtime_jit,
    )
    return module


def _verify_native_extension(
    *,
    module: Any,
    extension_name: str,
    required_symbols: tuple[str, ...],
    source_files: tuple[Path, ...],
    require_manifest: bool,
) -> None:
    for symbol in required_symbols:
        if not hasattr(module, symbol):
            raise RuntimeError(
                f"native extension {extension_name!r} is missing required "
                f"symbol {symbol!r}; rebuild the packaged dllm_parallel kernels."
            )
    capability: tuple[int, int] | None = None
    try:
        import torch
        if torch.cuda.is_available():
            capability = tuple(torch.cuda.get_device_capability())
    except Exception:
        capability = None
    if require_manifest:
        _verify_manifest_entry(
            module=module,
            extension_name=extension_name,
            source_files=source_files,
            device_capability=capability,
        )


def native_kernel_manifest_path() -> Path:
    return Path(__file__).resolve().parents[1] / "_C" / "native_kernels.json"


def source_digest(source_files: Sequence[str | Path]) -> str:
    digest = hashlib.sha256()
    for source in source_files:
        path = Path(source)
        digest.update(str(path.name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def binary_digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def installed_package_version() -> str:
    """Return the installed distribution version bound to native artifacts."""

    try:
        return importlib.metadata.version(NATIVE_KERNEL_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"{NATIVE_KERNEL_DISTRIBUTION!r} distribution metadata is unavailable; "
            "install the package before loading production native kernels"
        ) from exc


def _verify_manifest_entry(
    *,
    module: Any,
    extension_name: str,
    source_files: tuple[Path, ...],
    device_capability: tuple[int, int] | None,
) -> None:
    manifest_path = native_kernel_manifest_path()
    if not manifest_path.exists():
        raise RuntimeError(
            "packaged native kernel manifest is missing: "
            f"{manifest_path}. Rebuild the dllm_parallel native kernels."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_format = manifest.get("format")
    if manifest_format != NATIVE_KERNEL_MANIFEST_FORMAT:
        raise RuntimeError(
            f"native kernel manifest format mismatch: {manifest_format!r} != "
            f"{NATIVE_KERNEL_MANIFEST_FORMAT!r}"
        )
    manifest_package_version = manifest.get("package_version")
    package_version = installed_package_version()
    if manifest_package_version != package_version:
        raise RuntimeError(
            f"native kernel package version mismatch: "
            f"{manifest_package_version!r} != {package_version!r}"
        )
    entry = (manifest.get("kernels") or {}).get(extension_name)
    if not isinstance(entry, dict):
        raise RuntimeError(
            f"packaged native kernel manifest has no entry for {extension_name!r}"
        )
    _verify_declared_build_flags(extension_name, entry)
    if source_files:
        observed_source_hash = source_digest(source_files)
        expected_source_hash = entry.get("source_hash")
        if expected_source_hash != observed_source_hash:
            raise RuntimeError(
                f"native kernel {extension_name!r} source hash mismatch: "
                f"{expected_source_hash!r} != {observed_source_hash!r}"
            )
    module_file = getattr(module, "__file__", None)
    binary_name = entry.get("binary")
    if not isinstance(module_file, str) or not module_file:
        raise RuntimeError(f"native kernel {extension_name!r} has no binary path")
    binary_path = Path(module_file)
    if isinstance(binary_name, str) and binary_name and binary_path.name != binary_name:
        raise RuntimeError(
            f"native kernel {extension_name!r} binary mismatch: "
            f"{binary_path.name!r} != {binary_name!r}"
        )
    expected_binary_hash = entry.get("binary_hash")
    if not isinstance(expected_binary_hash, str) or not expected_binary_hash:
        raise RuntimeError(
            f"native kernel {extension_name!r} manifest is missing binary_hash"
        )
    observed_binary_hash = binary_digest(binary_path)
    if observed_binary_hash != expected_binary_hash:
        raise RuntimeError(
            f"native kernel {extension_name!r} binary hash mismatch: "
            f"{expected_binary_hash!r} != {observed_binary_hash!r}"
        )
    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    expected_suffix = entry.get("python_extension_suffix") or manifest.get(
        "python_extension_suffix"
    )
    if (
        isinstance(extension_suffix, str)
        and isinstance(expected_suffix, str)
        and extension_suffix
        and expected_suffix
        and extension_suffix != expected_suffix
    ):
        raise RuntimeError(
            f"native kernel {extension_name!r} Python ABI mismatch: "
            f"{expected_suffix!r} != {extension_suffix!r}"
        )
    supported_capabilities = {
        tuple(int(part) for part in capability)
        for capability in entry.get("supported_device_capabilities") or ()
        if len(capability) == 2
    }
    ptx_capabilities = {
        tuple(int(part) for part in capability)
        for capability in entry.get("ptx_device_capabilities") or ()
        if len(capability) == 2
    }
    if not supported_capabilities:
        raise RuntimeError(
            f"native kernel {extension_name!r} manifest has no compiled device "
            "capability set"
        )
    if device_capability is not None:
        current = tuple(int(part) for part in device_capability)
        has_compatible_ptx = any(capability <= current for capability in ptx_capabilities)
        if current not in supported_capabilities and not has_compatible_ptx:
            raise RuntimeError(
                f"native kernel {extension_name!r} was built for "
                f"{sorted(supported_capabilities)} with PTX {sorted(ptx_capabilities)}; "
                f"current device is {current}"
            )


def _verify_declared_build_flags(extension_name: str, entry: dict[str, Any]) -> None:
    """Bind a packaged binary to the flags declared by its kernel spec."""

    try:
        from dllm_parallel.core.kernels._native_specs import spec_by_name

        spec = spec_by_name(extension_name)
    except KeyError:
        return
    expected = {
        "extra_cflags": list(spec.extra_cflags),
        "extra_cuda_cflags": list(spec.extra_cuda_cflags),
        "extra_ldflags": list(spec.extra_ldflags),
    }
    observed = entry.get("build_flags")
    if observed != expected:
        raise RuntimeError(
            f"native kernel {extension_name!r} build flags mismatch: "
            f"{observed!r} != {expected!r}"
        )
