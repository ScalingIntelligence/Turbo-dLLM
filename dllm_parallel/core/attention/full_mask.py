# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Public full-mask attention API for production context parallelism."""

from __future__ import annotations

import math
from typing import Any

import torch

from dllm_parallel.core.attention.context_parallel_attention import (
    _CollectiveContextParallelAttentionBSHD,
    _RingContextParallelAttentionBSHD,
    _bdlm_encoded_key_start,
    _bdlm_flex_shard_attention_bshd,
    _bdlm_flex_shard_backward_exact_bshd,
    _should_use_context_parallel,
    _validate_bshd,
)
from dllm_parallel.core.attention.masks import (
    BlockDenoisingFullMask,
)
from dllm_parallel.core.attention.policy import cp_bp_policy


def full_mask_block_denoising_attention_bshd(
    query: torch.Tensor,
    key_shard: torch.Tensor,
    value_shard: torch.Tensor,
    *,
    global_seq_len: int,
    attn_mask: BlockDenoisingFullMask,
    scale: float | None = None,
    runtime: Any,
) -> torch.Tensor:
    """Evaluate the exact full graph locally or with CP-sharded K/V."""

    _validate_bshd(query, key_shard, value_shard)
    if int(global_seq_len) <= 0 or int(global_seq_len) % 2 != 0:
        raise ValueError("global_seq_len must be a positive even [noisy; clean] length")
    if not isinstance(attn_mask, BlockDenoisingFullMask):
        raise TypeError("full-mask context-parallel attention requires BlockDenoisingFullMask")
    if int(attn_mask.clean_offset) != int(global_seq_len) // 2:
        raise ValueError("full-mask clean_offset must equal half of global_seq_len")
    if attn_mask.query_blocks.ndim not in {1, 2}:
        raise ValueError("full-mask query metadata must be one- or two-dimensional")
    if int(attn_mask.query_blocks.shape[-1]) != int(query.shape[1]):
        raise ValueError("full-mask query metadata must match the local query shard")
    if (
        attn_mask.query_blocks.ndim == 2
        and int(attn_mask.query_blocks.shape[0]) != int(query.shape[0])
    ):
        raise ValueError("batched full-mask metadata must match the query batch")
    resolved_scale = scale if scale is not None else 1.0 / math.sqrt(query.shape[-1])
    if not _should_use_context_parallel(runtime):
        if int(query.shape[1]) != int(global_seq_len):
            raise ValueError("local full-mask attention requires the complete sequence")
        return _LocalMonolithicAttentionBSHD.apply(
            query,
            key_shard,
            value_shard,
            attn_mask,
            float(resolved_scale),
        )
    backend = (
        _RingContextParallelAttentionBSHD
        if cp_bp_policy(runtime).clean_kv_transport == "streaming"
        else _CollectiveContextParallelAttentionBSHD
    )
    return backend.apply(
        query,
        key_shard,
        value_shard,
        attn_mask,
        float(resolved_scale),
        runtime,
        int(global_seq_len),
    )


class _LocalMonolithicAttentionBSHD(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: BlockDenoisingFullMask,
        scale: float,
    ) -> torch.Tensor:
        output, lse = _bdlm_flex_shard_attention_bshd(
            query,
            key,
            value,
            attn_mask.query_blocks,
            attn_mask.query_is_clean,
            int(attn_mask.block_size),
            _bdlm_encoded_key_start(attn_mask, 0),
            float(scale),
            clean_offset=int(attn_mask.clean_offset),
            flex_cache=attn_mask.flex_cache,
        )
        ctx.attn_mask = attn_mask.detach()
        ctx.scale = float(scale)
        ctx.save_for_backward(
            query.detach(),
            key.detach(),
            value.detach(),
            output.detach(),
            lse.detach(),
        )
        return output

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        query, key, value, output, lse = ctx.saved_tensors
        attn_mask = ctx.attn_mask
        grad_query, grad_key, grad_value = _bdlm_flex_shard_backward_exact_bshd(
            query=query,
            key=key,
            value=value,
            final_output=output,
            final_lse=lse,
            grad_output=grad_output.contiguous(),
            query_blocks=attn_mask.query_blocks,
            query_is_clean=attn_mask.query_is_clean,
            block_size=int(attn_mask.block_size),
            key_start=_bdlm_encoded_key_start(attn_mask, 0),
            scale=float(ctx.scale),
            clean_offset=int(attn_mask.clean_offset),
            flex_cache=attn_mask.flex_cache,
            debug_nonfinite_attention=bool(attn_mask.debug_nonfinite_attention),
        )
        return (
            grad_query.to(dtype=query.dtype),
            grad_key.to(dtype=key.dtype),
            grad_value.to(dtype=value.dtype),
            None,
            None,
        )


__all__ = ["full_mask_block_denoising_attention_bshd"]
