# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Python entry points for the native fused-scheduled LM-head cross entropy.

The CUDA/cuBLAS source lives in ``dllm_parallel/csrc/fused_linear_ce`` and is
compiled into ``dllm_parallel.core._C.dllm_fused_linear_ce_v3``. The extension keeps
the chunk scheduler below Python: it projects native-scheduled token tiles with
cuBLAS tensor-core GEMMs and runs custom CUDA kernels for the cross-entropy
stats, loss, grad-logits, and dtype casts, avoiding full sequence logits
materialization while preserving tensor-core GEMM throughput. This module only
resolves and forwards to the loaded extension; build flags and source files are
described once in :mod:`dllm_parallel.core.kernels._native_specs`.
"""

from __future__ import annotations

from functools import lru_cache

import torch

from dllm_parallel.core.kernels._native_specs import jit_build, spec_by_name
from dllm_parallel.core.kernels.runtime import load_native_extension

_SPEC = spec_by_name("dllm_fused_linear_ce_v3")
_DFLASH_SYMBOLS = tuple(
    symbol for symbol in _SPEC.symbols if symbol.startswith("dflash_")
)
_CORE_SYMBOLS = tuple(
    symbol for symbol in _SPEC.symbols if not symbol.startswith("dflash_")
)


@lru_cache(maxsize=1)
def _extension():
    # Keep DFlash additions capability-scoped.  Dense/MoE diffusion backbones
    # use this extension for ordinary or vocabulary-parallel CE and must not
    # acquire a dependency on DFlash-only entry points merely by loading it.
    return load_native_extension(
        package_module=_SPEC.package_module,
        extension_name=_SPEC.name,
        jit_builder=lambda: jit_build(_SPEC),
        required_symbols=_CORE_SYMBOLS,
        source_files=_SPEC.source_paths(),
    )


@lru_cache(maxsize=1)
def _dflash_extension():
    extension = _extension()
    for symbol in _DFLASH_SYMBOLS:
        if not hasattr(extension, symbol):
            raise RuntimeError(
                f"native extension {_SPEC.name!r} is missing DFlash capability "
                f"{symbol!r}; rebuild the packaged dllm_parallel kernels"
            )
    return extension


def forward(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    labels: torch.Tensor,
    *,
    row_block_size: int,
    ignore_index: int,
    exclude_index: int,
    logit_softcap: float,
    compute_dtype: torch.dtype,
    weight_dim_first: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del compute_dtype
    bias_tensor = (
        bias
        if bias is not None
        else torch.empty(0, dtype=weight.dtype, device=weight.device)
    )
    return tuple(
        _extension().chunked_linear_ce_forward(
            hidden,
            weight,
            bias_tensor,
            labels,
            int(row_block_size),
            int(ignore_index),
            int(exclude_index),
            float(logit_softcap),
            "",
            bool(weight_dim_first),
            bias is not None,
        )
    )


def backward(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    labels: torch.Tensor,
    lse: torch.Tensor,
    valid_count: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    row_block_size: int,
    reduction: str,
    ignore_index: int,
    exclude_index: int,
    logit_softcap: float,
    compute_dtype: torch.dtype,
    weight_dim_first: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del compute_dtype
    bias_tensor = (
        bias
        if bias is not None
        else torch.empty(0, dtype=weight.dtype, device=weight.device)
    )
    return tuple(
        _extension().chunked_linear_ce_backward(
            hidden,
            weight,
            bias_tensor,
            labels,
            lse,
            valid_count,
            grad_output,
            int(row_block_size),
            _reduction_code(reduction),
            int(ignore_index),
            int(exclude_index),
            float(logit_softcap),
            "",
            bool(weight_dim_first),
            bias is not None,
        )
    )


def backward_frozen_weight(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    labels: torch.Tensor,
    lse: torch.Tensor,
    valid_count: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    row_block_size: int,
    reduction: str,
    ignore_index: int,
    exclude_index: int,
    logit_softcap: float,
    compute_dtype: torch.dtype,
    weight_dim_first: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del compute_dtype
    bias_tensor = (
        bias
        if bias is not None
        else torch.empty(0, dtype=weight.dtype, device=weight.device)
    )
    return tuple(
        _extension().chunked_linear_ce_backward_frozen_weight(
            hidden,
            weight,
            bias_tensor,
            labels,
            lse,
            valid_count,
            grad_output,
            int(row_block_size),
            _reduction_code(reduction),
            int(ignore_index),
            int(exclude_index),
            float(logit_softcap),
            "",
            bool(weight_dim_first),
            bias is not None,
        )
    )


def vocab_parallel_forward_local(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    labels: torch.Tensor,
    *,
    vocab_start: int,
    ignore_index: int,
    exclude_index: int,
    logit_softcap: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute local vocabulary-shard softmax statistics without logits storage."""

    bias_tensor = (
        bias
        if bias is not None
        else torch.empty(0, dtype=weight.dtype, device=weight.device)
    )
    return tuple(
        _extension().vocab_parallel_ce_forward_local(
            hidden,
            weight,
            bias_tensor,
            labels,
            int(vocab_start),
            int(ignore_index),
            int(exclude_index),
            float(logit_softcap),
            bias is not None,
        )
    )


def vocab_parallel_backward_local(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    labels: torch.Tensor,
    global_lse: torch.Tensor,
    valid_count: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    reduction: str,
    vocab_start: int,
    ignore_index: int,
    exclude_index: int,
    logit_softcap: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute one vocabulary shard's exact CE gradient contributions."""

    bias_tensor = (
        bias
        if bias is not None
        else torch.empty(0, dtype=weight.dtype, device=weight.device)
    )
    return tuple(
        _extension().vocab_parallel_ce_backward_local(
            hidden,
            weight,
            bias_tensor,
            labels,
            global_lse,
            valid_count,
            grad_output,
            _reduction_code(reduction),
            int(vocab_start),
            int(ignore_index),
            int(exclude_index),
            float(logit_softcap),
            bias is not None,
        )
    )


def dflash_kl_forward(
    draft_hidden: torch.Tensor,
    draft_weight: torch.Tensor,
    teacher_hidden: torch.Tensor,
    teacher_weight: torch.Tensor,
    logit_softcap: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(
        _dflash_extension().dflash_frozen_kl_forward(
            draft_hidden,
            draft_weight,
            teacher_hidden,
            teacher_weight,
            float(logit_softcap),
        )
    )


def dflash_kl_backward(
    draft_hidden: torch.Tensor,
    draft_weight: torch.Tensor,
    teacher_hidden: torch.Tensor,
    teacher_weight: torch.Tensor,
    draft_lse: torch.Tensor,
    teacher_lse: torch.Tensor,
    grad_output: torch.Tensor,
    logit_softcap: float,
) -> torch.Tensor:
    return _dflash_extension().dflash_frozen_kl_backward(
        draft_hidden,
        draft_weight,
        teacher_hidden,
        teacher_weight,
        draft_lse,
        teacher_lse,
        grad_output,
        float(logit_softcap),
    )


def dflash_ce_topk_forward(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    top_k: int,
    row_block_size: int,
    output_multiplier: float,
    logit_softcap: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(
        _dflash_extension().dflash_frozen_ce_topk_forward(
            hidden,
            weight,
            labels,
            int(top_k),
            int(row_block_size),
            float(output_multiplier),
            float(logit_softcap),
        )
    )


def dflash_ce_topk_backward(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    lse: torch.Tensor,
    probabilities: torch.Tensor,
    top_ids: torch.Tensor,
    grad_loss: torch.Tensor,
    grad_probability: torch.Tensor,
    grad_top_values: torch.Tensor,
    *,
    row_block_size: int,
    output_multiplier: float,
    logit_softcap: float,
) -> torch.Tensor:
    return _dflash_extension().dflash_frozen_ce_topk_backward(
        hidden,
        weight,
        labels,
        lse,
        probabilities,
        top_ids,
        grad_loss,
        grad_probability,
        grad_top_values,
        int(row_block_size),
        float(output_multiplier),
        float(logit_softcap),
    )


def verify_dflash_symbols(*, loss_kind: str) -> None:
    if loss_kind not in {
        "paper_ce",
        "speculators_kl",
        "dflash",
        "dpace",
        "dpace-cumulative-confidence-only",
        "dpace-continuation-value-only",
    }:
        raise ValueError(f"unsupported DFlash loss {loss_kind!r}")
    _dflash_extension()


def _reduction_code(reduction: str) -> int:
    if reduction == "none":
        return 0
    if reduction == "sum":
        return 1
    if reduction == "mean":
        return 2
    raise ValueError(f"unsupported reduction: {reduction}")
