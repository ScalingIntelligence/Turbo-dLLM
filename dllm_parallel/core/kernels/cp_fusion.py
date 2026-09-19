# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Python entry points for the BDLM CP/BP online-softmax merge kernels.

The CUDA/C++ source lives in ``dllm_parallel/csrc/cp_fusion`` and is compiled
into ``dllm_parallel.core._C.bdlm_cp_fusion``. This module only resolves and forwards
to the loaded extension; build flags and source files are described once in
:mod:`dllm_parallel.core.kernels._native_specs`.
"""

from __future__ import annotations

from functools import lru_cache

import torch

from dllm_parallel.core.kernels._native_specs import jit_build, spec_by_name
from dllm_parallel.core.kernels.runtime import load_native_extension

_SPEC = spec_by_name("bdlm_cp_fusion")


@lru_cache(maxsize=1)
def _extension():
    return load_native_extension(
        package_module=_SPEC.package_module,
        extension_name=_SPEC.name,
        jit_builder=lambda: jit_build(_SPEC),
        required_symbols=_SPEC.symbols,
        source_files=_SPEC.source_paths(),
    )


def merge_full_(
    old_num: torch.Tensor,
    old_m: torch.Tensor,
    old_l: torch.Tensor,
    new_num: torch.Tensor,
    new_m: torch.Tensor,
    new_l: torch.Tensor,
) -> None:
    _extension().merge_full_(
        old_num,
        old_m,
        old_l,
        new_num,
        new_m,
        new_l,
    )


def merge_compact_(
    numerator: torch.Tensor,
    m: torch.Tensor,
    l: torch.Tensor,
    query_indices: torch.Tensor,
    compact_output: torch.Tensor,
    compact_lse: torch.Tensor,
) -> None:
    _extension().merge_compact_(
        numerator,
        m,
        l,
        query_indices,
        compact_output,
        compact_lse,
    )


def merge_backward_(
    shard_output: torch.Tensor,
    shard_lse: torch.Tensor,
    final_output: torch.Tensor,
    final_lse: torch.Tensor,
    grad_output: torch.Tensor,
    shard_grad_output: torch.Tensor,
    grad_lse: torch.Tensor,
) -> None:
    _extension().merge_backward_(
        shard_output,
        shard_lse,
        final_output,
        final_lse,
        grad_output,
        shard_grad_output,
        grad_lse,
    )


def finalize_bshd_(
    numerator: torch.Tensor,
    m: torch.Tensor,
    l: torch.Tensor,
    output: torch.Tensor,
    final_lse: torch.Tensor,
) -> None:
    _extension().finalize_bshd_(numerator, m, l, output, final_lse)
