# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Single source of truth for the in-repo native CUDA kernels.

Every consumer (the thin Python wrappers, the ``dllm-build-kernels`` packager,
and the release gate) reads kernel metadata from :data:`NATIVE_KERNEL_SPECS`
instead of re-declaring kernel names, source files, symbols, and compile flags.
CUDA/C++ source lives in real files under ``dllm_parallel/csrc`` -- this module
only describes how to compile and locate them, and it never imports ``torch`` at
import time so it stays cheap to import.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ``_native_specs`` lives in ``dllm_parallel/core/kernels``; ``parents[1]`` is
# the ``dllm_parallel.core`` package root, which contains ``csrc`` and ``_C``.
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CSRC_DIR = _PACKAGE_ROOT / "csrc"


@dataclass(frozen=True)
class NativeKernelSpec:
    """Describes one compiled extension built from ``csrc`` sources."""

    name: str
    package_module: str
    sources: tuple[str, ...]
    symbols: tuple[str, ...]
    extra_cflags: tuple[str, ...] = ("-O3",)
    extra_cuda_cflags: tuple[str, ...] = ("-O3", "--use_fast_math")
    extra_ldflags: tuple[str, ...] = ()
    needs_cublas: bool = False

    def source_paths(self) -> tuple[Path, ...]:
        """Absolute paths to this kernel's ``csrc`` source files."""

        return tuple(CSRC_DIR / relative for relative in self.sources)


NATIVE_KERNEL_SPECS: tuple[NativeKernelSpec, ...] = (
    NativeKernelSpec(
        name="bdlm_cp_fusion",
        package_module="dllm_parallel.core._C.bdlm_cp_fusion",
        sources=(
            "cp_fusion/cp_fusion.cpp",
            "cp_fusion/cp_fusion_cuda.cu",
        ),
        symbols=(
            "merge_full_",
            "merge_compact_",
            "merge_backward_",
            "finalize_bshd_",
        ),
        extra_cflags=("-O3",),
        extra_cuda_cflags=("-O3", "--use_fast_math"),
    ),
    NativeKernelSpec(
        name="dllm_fused_linear_ce_v3",
        package_module="dllm_parallel.core._C.dllm_fused_linear_ce_v3",
        sources=(
            "fused_linear_ce/fused_linear_ce.cpp",
            "fused_linear_ce/fused_linear_ce_cuda.cu",
        ),
        symbols=(
            "chunked_linear_ce_forward",
            "chunked_linear_ce_backward",
            "chunked_linear_ce_backward_frozen_weight",
            "vocab_parallel_ce_forward_local",
            "vocab_parallel_ce_backward_local",
            "dflash_frozen_kl_forward",
            "dflash_frozen_kl_backward",
            "dflash_frozen_ce_topk_forward",
            "dflash_frozen_ce_topk_backward",
        ),
        extra_cflags=("-O3", "-std=c++17"),
        extra_cuda_cflags=("-O3", "--use_fast_math"),
        extra_ldflags=("-lcublas",),
        needs_cublas=True,
    ),
    NativeKernelSpec(
        name="dflash_cp_fusion",
        package_module="dllm_parallel.core._C.dflash_cp_fusion",
        sources=(
            "dflash_cp_fusion/dflash_cp_fusion.cpp",
            "dflash_cp_fusion/dflash_cp_fusion_cuda.cu",
        ),
        symbols=("merge_bshd_", "merge_state_"),
        extra_cflags=("-O3",),
        extra_cuda_cflags=("-O3", "--use_fast_math"),
    ),
)


def spec_by_name(name: str) -> NativeKernelSpec:
    for spec in NATIVE_KERNEL_SPECS:
        if spec.name == name:
            return spec
    raise KeyError(f"unknown native kernel spec: {name!r}")


def jit_build(spec: NativeKernelSpec) -> Any:
    """Developer-build path: compile ``spec`` from its ``csrc`` sources.

    Uses ``torch.utils.cpp_extension.load`` over the real source files (not
    inline strings), so the compiled object is byte-for-byte equivalent to a
    packaged build of the same sources and flags. Production startup loads the
    prebuilt extension instead; this is only reached for developer builds and
    by ``dllm-build-kernels`` when constructing the packaged artifacts.
    """

    from torch.utils.cpp_extension import load
    from torch.utils.cpp_extension import CUDA_HOME

    if CUDA_HOME is None:
        raise RuntimeError(
            f"native kernel {spec.name!r} requires a CUDA development toolkit"
        )

    extra_ldflags = list(spec.extra_ldflags)
    extra_include_paths: list[str] = []
    if spec.needs_cublas:
        include_dir, lib_dir = _find_packaged_cuda_paths()
        if include_dir is not None:
            extra_include_paths.append(include_dir)
        if lib_dir is not None:
            extra_ldflags.append(f"-L{lib_dir}")

    return load(
        name=spec.name,
        sources=[str(path) for path in spec.source_paths()],
        extra_cflags=list(spec.extra_cflags),
        extra_cuda_cflags=list(spec.extra_cuda_cflags),
        extra_ldflags=extra_ldflags,
        extra_include_paths=extra_include_paths,
        with_cuda=True,
        verbose=False,
    )


def _find_packaged_cuda_paths() -> tuple[str | None, str | None]:
    """Locate cuBLAS headers/libs shipped by the ``nvidia-*`` pip wheels."""

    import site
    import sys

    import torch

    candidates: list[Path] = []
    for value in [*sys.path, *site.getsitepackages(), site.getusersitepackages()]:
        if value:
            candidates.append(Path(value))
    major = str(torch.version.cuda or "").split(".", maxsplit=1)[0]
    cuda_names = [f"cu{major}", "cu13", "cu12", "cuda_runtime"]
    seen: set[Path] = set()
    for base in candidates:
        nvidia_root = base / "nvidia"
        for cuda_name in cuda_names:
            root = nvidia_root / cuda_name
            if root in seen:
                continue
            seen.add(root)
            include_dir = root / "include"
            lib_dir = root / "lib"
            if (include_dir / "cublas_v2.h").exists() and lib_dir.exists():
                return str(include_dir), str(lib_dir)
    return None, None
