# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Native online-softmax merge operations used by DFlash attention."""

from __future__ import annotations

from functools import lru_cache

import torch

from dllm_parallel.core.kernels._native_specs import jit_build, spec_by_name
from dllm_parallel.core.kernels.runtime import load_native_extension

_SPEC = spec_by_name("dflash_cp_fusion")


@lru_cache(maxsize=1)
def _extension():
    return load_native_extension(
        package_module=_SPEC.package_module,
        extension_name=_SPEC.name,
        jit_builder=lambda: jit_build(_SPEC),
        required_symbols=_SPEC.symbols,
        source_files=_SPEC.source_paths(),
    )


def merge_bshd_(
    numerator: torch.Tensor,
    m: torch.Tensor,
    l: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
) -> None:
    _extension().merge_bshd_(numerator, m, l, output, lse)


def merge_state_(
    numerator: torch.Tensor,
    m: torch.Tensor,
    l: torch.Tensor,
    incoming_numerator: torch.Tensor,
    incoming_m: torch.Tensor,
    incoming_l: torch.Tensor,
) -> None:
    _extension().merge_state_(
        numerator,
        m,
        l,
        incoming_numerator,
        incoming_m,
        incoming_l,
    )


def verify() -> None:
    _extension()
