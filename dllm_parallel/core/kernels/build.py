# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Build and install packaged native kernel artifacts.

Compiles every kernel declared in
:data:`dllm_parallel.core.kernels._native_specs.NATIVE_KERNEL_SPECS` from its
``csrc`` sources, copies the resulting extension into ``dllm_parallel.core._C``, and
writes ``native_kernels.json`` describing each artifact (binary/source hashes,
ABI suffix, symbols, CUDA arch list, minimum device capability) so runtime load
and the release gate can verify it. Run this once during wheel/container build:

    python -m dllm_parallel.core.kernels.build            # or: dllm-build-kernels
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sysconfig
import time
from pathlib import Path
from typing import Any

from dllm_parallel.core.kernels._native_specs import NATIVE_KERNEL_SPECS, jit_build
from dllm_parallel.core.kernels.runtime import (
    NATIVE_KERNEL_MANIFEST_FORMAT,
    binary_digest,
    installed_package_version,
    source_digest,
)


def build_native_kernels(
    *,
    output_dir: str | Path | None = None,
    cuda_arch_list: str | None = None,
    package_version: str | None = None,
) -> dict[str, Any]:
    """Compile each native kernel from source and package it into ``_C``."""

    root = Path(__file__).resolve().parents[3]
    package_dir = Path(output_dir) if output_dir is not None else root / "dllm_parallel" / "core" / "_C"
    package_dir.mkdir(parents=True, exist_ok=True)
    resolved_arch_list = _resolve_cuda_arch_list(cuda_arch_list)
    os.environ["TORCH_CUDA_ARCH_LIST"] = resolved_arch_list
    supported_capabilities, ptx_capabilities = _parse_cuda_arch_list(
        resolved_arch_list
    )
    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    manifest: dict[str, Any] = {
        "format": NATIVE_KERNEL_MANIFEST_FORMAT,
        "package_version": _resolve_build_package_version(package_version),
        "built_unix": time.time(),
        "cuda_arch_list": resolved_arch_list,
        "supported_device_capabilities": [list(item) for item in supported_capabilities],
        "ptx_device_capabilities": [list(item) for item in ptx_capabilities],
        "python_extension_suffix": extension_suffix,
        "kernels": {},
    }
    try:
        import torch

        if torch.cuda.is_available():
            manifest["torch_cuda"] = torch.version.cuda
            manifest["device_capability"] = list(torch.cuda.get_device_capability())
            manifest["device_name"] = torch.cuda.get_device_name()
    except Exception:
        pass

    for spec in NATIVE_KERNEL_SPECS:
        extension = jit_build(spec)
        binary_path = Path(getattr(extension, "__file__"))
        suffix = "".join(binary_path.suffixes)
        if not suffix:
            suffix = extension_suffix or ".so"
        destination = package_dir / f"{spec.name}{suffix}"
        if binary_path.resolve() != destination.resolve():
            shutil.copy2(binary_path, destination)
        source_files = spec.source_paths()
        manifest["kernels"][spec.name] = {
            "package_module": spec.package_module,
            "binary": destination.name,
            "binary_hash": binary_digest(destination),
            "cuda_arch_list": resolved_arch_list,
            "supported_device_capabilities": [
                list(item) for item in supported_capabilities
            ],
            "ptx_device_capabilities": [list(item) for item in ptx_capabilities],
            "python_extension_suffix": extension_suffix,
            "source_hash": source_digest(source_files),
            "source_files": [str(path.relative_to(root)) for path in source_files],
            "required_symbols": list(spec.symbols),
            "build_flags": {
                "extra_cflags": list(spec.extra_cflags),
                "extra_cuda_cflags": list(spec.extra_cuda_cflags),
                "extra_ldflags": list(spec.extra_ldflags),
            },
            "minimum_device_capability": list(min(supported_capabilities)),
        }

    manifest_path = package_dir / "native_kernels.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _resolve_build_package_version(requested: str | None) -> str:
    """Resolve the distribution version recorded in native artifacts.

    Source-package builders pass the version from their build metadata before
    the wheel is installed. Invocations from an installed package use that
    package's distribution metadata.
    """

    if requested is None:
        return installed_package_version()
    version = str(requested).strip()
    if not version:
        raise ValueError("native kernel package version must not be empty")
    return version


def _resolve_cuda_arch_list(requested: str | None) -> str:
    value = requested or os.environ.get("TORCH_CUDA_ARCH_LIST")
    if value:
        _parse_cuda_arch_list(value)
        return value
    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            return f"{int(major)}.{int(minor)}"
    except Exception as exc:
        raise RuntimeError("failed to detect the CUDA build architecture") from exc
    raise RuntimeError(
        "native kernel builds require --cuda-arch-list or "
        "TORCH_CUDA_ARCH_LIST when no CUDA device is visible"
    )


def _parse_cuda_arch_list(
    value: str,
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    capabilities: list[tuple[int, int]] = []
    ptx: list[tuple[int, int]] = []
    for entry in re.split(r"[;,\s]+", str(value).strip()):
        if not entry:
            continue
        has_ptx = entry.upper().endswith("+PTX")
        numeric = entry[:-4] if has_ptx else entry
        match = re.fullmatch(r"(\d+)\.(\d+)", numeric)
        if match is None:
            raise ValueError(
                "CUDA architecture entries must use numeric major.minor syntax; "
                f"got {entry!r}"
            )
        capability = (int(match.group(1)), int(match.group(2)))
        if capability not in capabilities:
            capabilities.append(capability)
        if has_ptx and capability not in ptx:
            ptx.append(capability)
    if not capabilities:
        raise ValueError("CUDA architecture list must not be empty")
    return tuple(sorted(capabilities)), tuple(sorted(ptx))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build packaged dllm_parallel native kernels.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cuda-arch-list", default=None)
    parser.add_argument(
        "--package-version",
        default=None,
        help=(
            "distribution version to record when building before package "
            "installation; installed-package builds infer it from metadata"
        ),
    )
    args = parser.parse_args(argv)
    manifest = build_native_kernels(
        output_dir=args.output_dir,
        cuda_arch_list=args.cuda_arch_list,
        package_version=args.package_version,
    )
    print(json.dumps({"event": "native_kernel_build", **manifest}, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
