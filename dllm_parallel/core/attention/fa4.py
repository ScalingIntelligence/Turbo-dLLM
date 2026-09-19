# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""FlashAttention-4 block-sparse kernels for compiled BDLM mask state."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable

import torch


def fa4_forward_sparse_tile(
    *,
    head_dim: int,
    head_dim_v: int,
    sparse_query_block_size: int,
) -> tuple[int, int]:
    """Return the SM90 forward tile required by block-sparse FA4."""

    from flash_attn.cute.interface import _get_device_arch, _tile_size_fwd_sm90

    arch = int(_get_device_arch())
    if arch // 10 != 9:
        raise RuntimeError("native BDLM FA4 forward requires SM90")
    config = _tile_size_fwd_sm90(
        int(head_dim),
        int(head_dim_v),
        False,
        False,
        sparse_block_size_q=int(sparse_query_block_size),
    )
    return int(config.m_block_size), int(config.n_block_size)


def fa4_backward_sparse_tile(
    *,
    head_dim: int,
    head_dim_v: int,
    sparse_query_block_size: int,
) -> tuple[int, int]:
    """Return the SM90 backward tile selected by the packaged FA4 runtime."""

    from flash_attn.cute.interface import _get_device_arch, _tile_size_bwd_sm90

    arch = int(_get_device_arch())
    if arch // 10 != 9:
        raise RuntimeError("native BDLM FA4 backward requires SM90")
    config = _tile_size_bwd_sm90(
        int(head_dim),
        int(head_dim_v),
        False,
        False,
        sparse_block_size_q=int(sparse_query_block_size),
    )
    return int(config.m_block_size), int(config.n_block_size)


@lru_cache(maxsize=1)
def _bdlm_mask_mod() -> Callable[..., Any]:
    import cutlass
    import cutlass.cute as cute
    from flash_attn.cute import utils
    from flash_attn.cute.block_sparsity import fast_sampling

    @fast_sampling
    @cute.jit
    def mask_mod(
        batch: cute.TensorSSA,
        head: cute.TensorSSA,
        query_index: cute.TensorSSA,
        key_index: cute.TensorSSA,
        seqlen_info: Any,
        aux_tensors: list[Any],
    ) -> cute.TensorSSA:
        del batch, head, seqlen_info
        query_clean_bounds = aux_tensors[0]
        local_query_blocks = aux_tensors[1]
        query_is_clean = aux_tensors[2]
        key_coordinates = aux_tensors[3]
        key_is_clean = aux_tensors[4]

        clean_start = utils.scalar_to_ssa(
            query_clean_bounds[query_index[0], 0],
            cutlass.Int32,
        )
        clean_stop = utils.scalar_to_ssa(
            query_clean_bounds[query_index[0], 1],
            cutlass.Int32,
        )
        local_query_block = utils.scalar_to_ssa(
            local_query_blocks[query_index[0]],
            cutlass.Int32,
        )
        clean_query = utils.scalar_to_ssa(
            query_is_clean[query_index[0]],
            cutlass.Boolean,
        )
        key_coordinate = utils.scalar_to_ssa(
            key_coordinates[key_index[0]],
            cutlass.Int32,
        )
        clean_key = utils.scalar_to_ssa(
            key_is_clean[key_index[0]],
            cutlass.Boolean,
        )
        false = utils.scalar_to_ssa(False, cutlass.Boolean)
        active_to_active = (
            (clean_query == false)
            & (clean_key == false)
            & (local_query_block == key_coordinate)
        )
        clean_context = (
            (clean_key != false)
            & (key_coordinate >= clean_start)
            & (key_coordinate < clean_stop)
        )
        return active_to_active | clean_context

    return mask_mod


def _backward_block_sparsity(
    block_mask: tuple[Any, ...],
) -> Any:
    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch

    (
        _query_length,
        _key_length,
        _kv_num_blocks,
        _kv_indices,
        _full_kv_num_blocks,
        _full_kv_indices,
        q_num_blocks,
        q_indices,
        full_q_num_blocks,
        full_q_indices,
        query_block_size,
        key_block_size,
        _mask_mod,
    ) = block_mask
    return BlockSparseTensorsTorch(
        mask_block_cnt=q_num_blocks,
        mask_block_idx=q_indices,
        full_block_cnt=full_q_num_blocks,
        full_block_idx=full_q_indices,
        block_size=(int(query_block_size), int(key_block_size)),
    )


def _forward_block_sparsity(block_mask: tuple[Any, ...]) -> Any:
    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch

    (
        _query_length,
        _key_length,
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        _q_num_blocks,
        _q_indices,
        _full_q_num_blocks,
        _full_q_indices,
        query_block_size,
        key_block_size,
        _mask_mod,
    ) = block_mask
    return BlockSparseTensorsTorch(
        mask_block_cnt=kv_num_blocks,
        mask_block_idx=kv_indices,
        full_block_cnt=full_kv_num_blocks,
        full_block_idx=full_kv_indices,
        block_size=(int(query_block_size), int(key_block_size)),
    )


def fa4_forward(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_mask: tuple[Any, ...],
    mask_buffers: tuple[torch.Tensor, ...],
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate exact BDLM block-sparse attention with native FA4."""

    try:
        from flash_attn.cute.interface import _flash_attn_fwd
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "BDLM native block-sparse forward requires flash-attn-4"
        ) from error
    if len(mask_buffers) != 5:
        raise RuntimeError("FA4 BDLM masks require five metadata tensors")
    if not query.is_cuda:
        raise RuntimeError("FA4 forward requires CUDA tensors")

    result = _flash_attn_fwd(
        q=query.transpose(1, 2),
        k=key.transpose(1, 2),
        v=value.transpose(1, 2),
        softmax_scale=float(scale),
        causal=False,
        pack_gqa=bool(query.shape[1] != key.shape[1]),
        mask_mod=_bdlm_mask_mod(),
        aux_tensors=[tensor.contiguous() for tensor in mask_buffers],
        block_sparse_tensors=_forward_block_sparsity(block_mask),
        return_lse=True,
    )
    output, lse = result[:2]
    return output.transpose(1, 2), lse


def fa4_backward_from_state(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    grad_output: torch.Tensor,
    grad_lse: torch.Tensor | None,
    block_mask: tuple[Any, ...],
    mask_buffers: tuple[torch.Tensor, ...],
    scale: float,
    pack_gqa: bool = False,
    grad_query: torch.Tensor | None = None,
    grad_key: torch.Tensor | None = None,
    grad_value: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiate one exact attention shard with FA4 block sparsity."""

    try:
        from flash_attn.cute.interface import _flash_attn_bwd
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "BDLM FlexAttention training requires flash-attn-4"
        ) from error
    if len(mask_buffers) != 5:
        raise RuntimeError("FA4 BDLM masks require five metadata tensors")
    if not query.is_cuda:
        raise RuntimeError("FA4 backward requires CUDA tensors")
    for name, supplied, reference in (
        ("grad_query", grad_query, query),
        ("grad_key", grad_key, key),
        ("grad_value", grad_value, value),
    ):
        if supplied is not None and supplied.stride() != reference.stride():
            raise ValueError(f"{name} must preserve the corresponding FA4 input layout")

    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)
    out = output.transpose(1, 2)
    dout = grad_output.transpose(1, 2)
    dq, dk, dv = _flash_attn_bwd(
        q=q,
        k=k,
        v=v,
        out=out,
        dout=dout,
        lse=lse,
        softmax_scale=float(scale),
        causal=False,
        deterministic=False,
        pack_gqa=bool(pack_gqa),
        mask_mod=_bdlm_mask_mod(),
        aux_tensors=[tensor.contiguous() for tensor in mask_buffers],
        block_sparse_tensors=_backward_block_sparsity(block_mask),
        dlse=grad_lse,
        dq=(None if grad_query is None else grad_query.transpose(1, 2)),
        dk=(None if grad_key is None else grad_key.transpose(1, 2)),
        dv=(None if grad_value is None else grad_value.transpose(1, 2)),
    )
    return (
        dq.transpose(1, 2),
        dk.transpose(1, 2),
        dv.transpose(1, 2),
    )


def fa4_dense_backward_from_state(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    grad_output: torch.Tensor,
    grad_lse: torch.Tensor | None,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiate one unmasked attention shard from merged state."""

    try:
        from flash_attn.cute.interface import _flash_attn_bwd
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "BDLM FlexAttention training requires flash-attn-4"
        ) from error
    if not query.is_cuda:
        raise RuntimeError("FA4 backward requires CUDA tensors")

    dq, dk, dv = _flash_attn_bwd(
        q=query.transpose(1, 2),
        k=key.transpose(1, 2),
        v=value.transpose(1, 2),
        out=output.transpose(1, 2),
        dout=grad_output.transpose(1, 2),
        lse=lse,
        softmax_scale=float(scale),
        causal=False,
        deterministic=False,
        dlse=grad_lse,
    )
    return (
        dq.transpose(1, 2),
        dk.transpose(1, 2),
        dv.transpose(1, 2),
    )


__all__ = [
    "fa4_backward_from_state",
    "fa4_backward_sparse_tile",
    "fa4_dense_backward_from_state",
    "fa4_forward",
    "fa4_forward_sparse_tile",
]
