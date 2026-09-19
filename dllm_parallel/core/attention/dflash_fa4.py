# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Native FA4 kernels and sparse plans for DFlash interval attention."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch

from dllm_parallel.core.attention.fa4 import (
    fa4_backward_sparse_tile,
    fa4_forward_sparse_tile,
)
from dllm_parallel.core.attention.masks import DFlashGlobalContextMask


# Torch FlexAttention and the packaged FA4 sparse kernels exchange block masks
# in 128-row query tiles. This is an ABI layout, not a runtime tuning policy.
_FLEX_QUERY_BLOCK_SIZE = 128


@dataclass(frozen=True)
class _IntervalPlan:
    block_mask: Any
    mask_buffers: tuple[torch.Tensor, ...]


class DFlashIntervalPlanCache:
    """Per-model-call sparse plans shared by DFlash layers and recomputation."""

    def __init__(self) -> None:
        self._plans: dict[tuple[Any, ...], _IntervalPlan] = {}

    def resolve(
        self,
        *,
        mask: DFlashGlobalContextMask,
        owner: int,
        query_length: int,
        key_start: int,
        key_length: int,
        key_intervals: tuple[tuple[int, int], ...] | None = None,
        query_tile: int,
        key_tile: int,
    ) -> _IntervalPlan:
        device = mask.anchor_valid.device
        intervals = (
            tuple((int(start), int(stop)) for start, stop in key_intervals)
            if key_intervals is not None
            else ((int(key_start), int(key_start) + int(key_length)),)
        )
        key = (
            int(owner),
            int(query_length),
            int(key_length),
            intervals,
            int(query_tile),
            int(key_tile),
            int(mask.block_size),
            int(mask.sliding_window or 0),
            tuple(int(value) for value in mask.anchor_valid.shape),
            device.type,
            device.index,
        )
        plan = self._plans.get(key)
        if plan is None:
            key_positions = _packed_key_positions(
                intervals,
                padded_length=int(key_length),
                device=device,
            )
            plan = _IntervalPlan(
                block_mask=_interval_block_mask(
                    mask=mask,
                    query_length=int(query_length),
                    key_positions=key_positions,
                    query_tile=int(query_tile),
                    key_tile=int(key_tile),
                ),
                mask_buffers=_mask_buffers(
                    mask,
                    key_positions=key_positions,
                ),
            )
            self._plans[key] = plan
        return plan


@lru_cache(maxsize=1)
def verify_dflash_fa4_runtime() -> dict[str, object]:
    """Verify the packaged FA4 capabilities required by DFlash training."""

    try:
        from flash_attn.cute import flash_bwd_postprocess
        from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "DFlash training requires the vendored flash-attn-4 runtime"
        ) from error
    if not callable(_flash_attn_fwd) or not callable(_flash_attn_bwd):
        raise RuntimeError("DFlash FA4 forward and backward kernels are unavailable")
    if not bool(
        getattr(
            flash_bwd_postprocess,
            "PACK_GQA_DQACCUM_ALIGNMENT_PROPAGATED",
            False,
        )
    ):
        raise RuntimeError(
            "DFlash packed-GQA training requires the alignment-capable "
            "vendored flash-attn-4 wheel"
        )
    return {
        "backend": "fa4_pack_gqa",
        "packed_gqa_alignment": True,
    }


def _local_block_views(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("DFlash local FA4 expects BSHD query, key, and value")
    if query.device.type != "cuda":
        raise RuntimeError("DFlash local FA4 requires CUDA tensors")
    if key.shape != value.shape:
        raise ValueError("DFlash local key and value shapes must match")
    if query.shape[:2] != key.shape[:2] or query.shape[-1] != key.shape[-1]:
        raise ValueError("DFlash local query and key token/head dimensions disagree")
    if int(query.shape[2]) % int(key.shape[2]):
        raise ValueError("DFlash local query heads must divide by key/value heads")
    block_size = int(block_size)
    if block_size <= 0 or int(query.shape[1]) % block_size:
        raise ValueError("DFlash local query length must divide by block_size")
    anchors = int(query.shape[1]) // block_size
    block_batch = int(query.shape[0]) * anchors
    return (
        query.contiguous().view(
            block_batch,
            block_size,
            int(query.shape[2]),
            int(query.shape[3]),
        ),
        key.contiguous().view(
            block_batch,
            block_size,
            int(key.shape[2]),
            int(key.shape[3]),
        ),
        value.contiguous().view(
            block_batch,
            block_size,
            int(value.shape[2]),
            int(value.shape[3]),
        ),
        anchors,
    )


@lru_cache(maxsize=1)
def _local_causal_mask_mod() -> Any:
    """Return the exact intra-block causal predicate for packed FA4 GQA."""

    import cutlass
    import cutlass.cute as cute
    from flash_attn.cute import utils

    @cute.jit
    def mask_mod(
        batch: cute.TensorSSA,
        head: cute.TensorSSA,
        query_index: cute.TensorSSA,
        key_index: cute.TensorSSA,
        seqlen_info: Any,
        aux_tensors: list[Any],
    ) -> cute.TensorSSA:
        del batch, head, seqlen_info, aux_tensors
        return utils.scalar_to_ssa(
            query_index[0] >= key_index[0],
            cutlass.Boolean,
        )

    return mask_mod


def local_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    block_size: int,
    causal: bool,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate batched dense attention within independent DFlash blocks."""

    block_query, block_key, block_value, anchors = _local_block_views(
        query,
        key,
        value,
        block_size=int(block_size),
    )
    try:
        from flash_attn.cute.interface import _flash_attn_fwd
    except (ImportError, AttributeError) as error:
        raise RuntimeError("DFlash local attention requires flash-attn-4") from error
    output, lse = _flash_attn_fwd(
        q=block_query,
        k=block_key,
        v=block_value,
        softmax_scale=float(scale),
        causal=False,
        pack_gqa=int(block_query.shape[2]) != int(block_key.shape[2]),
        mask_mod=_local_causal_mask_mod() if causal else None,
        return_lse=True,
    )[:2]
    batch = int(query.shape[0])
    output = output.reshape_as(query)
    lse = (
        lse.view(batch, anchors, int(query.shape[2]), int(block_size))
        .permute(0, 2, 1, 3)
        .reshape(batch, int(query.shape[2]), int(query.shape[1]))
        .contiguous()
    )
    return output, lse


def local_backward_from_state(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    final_output: torch.Tensor,
    final_lse: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    block_size: int,
    causal: bool,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiate local blocks from the merged local/context softmax state."""

    block_query, block_key, block_value, anchors = _local_block_views(
        query,
        key,
        value,
        block_size=int(block_size),
    )
    if final_output.shape != query.shape or grad_output.shape != query.shape:
        raise ValueError("DFlash local output tensors must match the query shape")
    expected_lse = (int(query.shape[0]), int(query.shape[2]), int(query.shape[1]))
    if tuple(final_lse.shape) != expected_lse:
        raise ValueError("DFlash local LSE shape does not match the query")
    batch = int(query.shape[0])
    block_output = final_output.contiguous().view_as(block_query)
    block_grad_output = grad_output.contiguous().view_as(block_query)
    block_lse = (
        final_lse.contiguous()
        .view(batch, int(query.shape[2]), anchors, int(block_size))
        .permute(0, 2, 1, 3)
        .reshape(batch * anchors, int(query.shape[2]), int(block_size))
        .contiguous()
    )
    try:
        from flash_attn.cute.interface import _flash_attn_bwd
    except (ImportError, AttributeError) as error:
        raise RuntimeError("DFlash local attention requires flash-attn-4") from error
    grad_query, grad_key, grad_value = _flash_attn_bwd(
        q=block_query,
        k=block_key,
        v=block_value,
        out=block_output,
        dout=block_grad_output,
        lse=block_lse,
        softmax_scale=float(scale),
        causal=False,
        deterministic=False,
        # Native packed-GQA backward can emit NaNs when the externally merged
        # LSE makes every local-block probability very small. The unpacked GQA
        # kernel is stable for this merged-state backward, whose blocks are tiny.
        pack_gqa=False,
        mask_mod=_local_causal_mask_mod() if causal else None,
        dlse=None,
    )
    return (
        grad_query.reshape_as(query),
        grad_key.reshape_as(key),
        grad_value.reshape_as(value),
    )


@lru_cache(maxsize=None)
def _interval_mask_mod(
    block_size: int,
    anchors_per_batch: int,
    sliding_window: int,
) -> Any:
    import cutlass
    import cutlass.cute as cute
    from flash_attn.cute import utils

    @cute.jit
    def mask_mod(
        batch: cute.TensorSSA,
        head: cute.TensorSSA,
        query_index: cute.TensorSSA,
        key_index: cute.TensorSSA,
        seqlen_info: Any,
        aux_tensors: list[Any],
    ) -> cute.TensorSSA:
        del head, seqlen_info
        context_starts = aux_tensors[0]
        context_stops = aux_tensors[1]
        anchor_valid = aux_tensors[2]
        key_positions = aux_tensors[3]
        anchor = query_index[0] // int(block_size)
        metadata_index = batch[0] * int(anchors_per_batch) + anchor
        start = utils.scalar_to_ssa(
            context_starts[metadata_index],
            cutlass.Int32,
        )
        stop = utils.scalar_to_ssa(
            context_stops[metadata_index],
            cutlass.Int32,
        )
        valid = utils.scalar_to_ssa(
            anchor_valid[metadata_index],
            cutlass.Boolean,
        )
        key_position = utils.scalar_to_ssa(
            key_positions[key_index[0]],
            cutlass.Int32,
        )
        in_window = utils.scalar_to_ssa(True, cutlass.Boolean)
        if int(sliding_window) > 0:
            offset_start = (
                stop + query_index[0] % int(block_size) - (int(sliding_window) - 1)
            )
            in_window = key_position >= offset_start
        return valid & in_window & (key_position >= start) & (key_position < stop)

    return mask_mod


def _ordered(dense: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    counts = dense.sum(dim=-1, dtype=torch.int32).contiguous()
    ordered = torch.argsort(
        dense.to(dtype=torch.int8),
        dim=-1,
        descending=True,
        stable=True,
    ).to(dtype=torch.int32)
    # ``contiguous()`` may preserve a non-unit stride for a size-one final
    # dimension. CuTe requires the sparse-index dimension itself to have unit
    # stride, so always copy into canonical storage.
    indices = torch.empty(
        ordered.shape,
        dtype=ordered.dtype,
        device=ordered.device,
    ).copy_(ordered)
    return counts, indices


def _forward_block_sparsity(block_mask: tuple[Any, ...]) -> Any:
    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch

    return BlockSparseTensorsTorch(
        mask_block_cnt=block_mask[2],
        mask_block_idx=block_mask[3],
        full_block_cnt=block_mask[4],
        full_block_idx=block_mask[5],
        block_size=(int(block_mask[10]), int(block_mask[11])),
    )


def _backward_block_sparsity(block_mask: tuple[Any, ...]) -> Any:
    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch

    return BlockSparseTensorsTorch(
        mask_block_cnt=block_mask[6],
        mask_block_idx=block_mask[7],
        full_block_cnt=block_mask[8],
        full_block_idx=block_mask[9],
        block_size=(int(block_mask[10]), int(block_mask[11])),
    )


def _interval_block_mask(
    *,
    mask: DFlashGlobalContextMask,
    query_length: int,
    key_positions: torch.Tensor,
    query_tile: int,
    key_tile: int,
) -> Any:
    try:
        from torch.nn.attention.flex_attention import BlockMask
    except (ImportError, AttributeError) as error:
        raise RuntimeError("DFlash FA4 requires Torch FlexAttention") from error

    batch_size, anchor_count = mask.anchor_valid.shape
    query_positions = torch.arange(
        0,
        ((int(query_length) + query_tile - 1) // query_tile) * query_tile,
        device=mask.anchor_valid.device,
        dtype=torch.int64,
    )
    query_exists = query_positions < int(query_length)
    anchor_indices = query_positions.clamp_max(int(query_length) - 1) // int(
        mask.block_size
    )
    query_exists = query_exists.view(-1, int(query_tile))
    anchor_indices = anchor_indices.view(-1, int(query_tile))

    starts = mask.context_starts[:, anchor_indices]
    if mask.sliding_window is not None:
        query_offsets = (query_positions % int(mask.block_size)).view(
            -1, int(query_tile)
        )
        starts = torch.maximum(
            starts,
            mask.context_stops[:, anchor_indices]
            + query_offsets.unsqueeze(0).to(torch.int32)
            - (int(mask.sliding_window) - 1),
        )
    stops = mask.context_stops[:, anchor_indices]
    valid = mask.anchor_valid[:, anchor_indices] & query_exists.unsqueeze(0)

    key_length = int(key_positions.numel())
    padded_key_length = ((key_length + key_tile - 1) // key_tile) * key_tile
    sentinel = torch.iinfo(torch.int32).min
    if padded_key_length != key_length:
        padded_positions = torch.full(
            (padded_key_length,),
            sentinel,
            device=key_positions.device,
            dtype=torch.int32,
        )
        padded_positions[:key_length].copy_(key_positions)
    else:
        padded_positions = key_positions
    tiled_key_positions = padded_positions.view(-1, int(key_tile))
    key_exists = tiled_key_positions.ne(sentinel)
    first_key = torch.where(
        key_exists,
        tiled_key_positions,
        torch.iinfo(torch.int32).max,
    ).amin(dim=-1)
    last_key = torch.where(
        key_exists,
        tiled_key_positions,
        sentinel,
    ).amax(dim=-1)

    occupied = (
        valid.unsqueeze(-1)
        & (starts.unsqueeze(-1) <= last_key)
        & (stops.unsqueeze(-1) > first_key)
    ).any(dim=2)
    all_queries_valid = (valid | ~query_exists.unsqueeze(0)).all(dim=2)
    full = (
        occupied
        & key_exists.all(dim=-1).unsqueeze(0)
        & all_queries_valid.unsqueeze(-1)
        & (
            (~query_exists.unsqueeze(0)).unsqueeze(-1)
            | ((starts.unsqueeze(-1) <= first_key) & (stops.unsqueeze(-1) > last_key))
        ).all(dim=2)
    )
    partial = occupied & ~full

    partial_kv_count, partial_kv_indices = _ordered(partial)
    full_kv_count, full_kv_indices = _ordered(full)
    partial_q_count, partial_q_indices = _ordered(partial.transpose(1, 2))
    full_q_count, full_q_indices = _ordered(full.transpose(1, 2))
    return BlockMask(
        seq_lengths=(int(query_length), key_length),
        kv_num_blocks=partial_kv_count[:, None],
        kv_indices=partial_kv_indices[:, None],
        full_kv_num_blocks=full_kv_count[:, None],
        full_kv_indices=full_kv_indices[:, None],
        q_num_blocks=partial_q_count[:, None],
        q_indices=partial_q_indices[:, None],
        full_q_num_blocks=full_q_count[:, None],
        full_q_indices=full_q_indices[:, None],
        BLOCK_SIZE=(int(query_tile), int(key_tile)),
        mask_mod=_torch_interval_mask_mod(
            mask,
            key_positions=key_positions,
        ),
    )


def _torch_interval_mask_mod(
    mask: DFlashGlobalContextMask,
    *,
    key_positions: torch.Tensor,
) -> Any:
    def mask_mod(
        batch: torch.Tensor,
        head: torch.Tensor,
        query_index: torch.Tensor,
        key_index: torch.Tensor,
    ) -> torch.Tensor:
        del head
        anchor = query_index // int(mask.block_size)
        key_position = key_positions[key_index]
        start = mask.context_starts[batch, anchor]
        if mask.sliding_window is not None:
            start = torch.maximum(
                start,
                mask.context_stops[batch, anchor]
                + query_index % int(mask.block_size)
                - (int(mask.sliding_window) - 1),
            )
        return (
            mask.anchor_valid[batch, anchor]
            & (key_position >= start)
            & (key_position < mask.context_stops[batch, anchor])
        )

    return mask_mod


def _mask_buffers(
    mask: DFlashGlobalContextMask,
    *,
    key_positions: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    return (
        mask.context_starts.reshape(-1).contiguous(),
        mask.context_stops.reshape(-1).contiguous(),
        mask.anchor_valid.reshape(-1).contiguous(),
        key_positions,
    )


def _packed_key_positions(
    intervals: tuple[tuple[int, int], ...],
    *,
    padded_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Map packed K/V rows back to global positions, masking padded rows."""

    chunks = [
        torch.arange(start, stop, device=device, dtype=torch.int32)
        for start, stop in intervals
        if stop > start
    ]
    valid_length = sum(int(stop) - int(start) for start, stop in intervals)
    if valid_length > int(padded_length):
        raise ValueError("DFlash packed intervals exceed the K/V tensor length")
    positions = torch.full(
        (int(padded_length),),
        torch.iinfo(torch.int32).min,
        device=device,
        dtype=torch.int32,
    )
    if chunks:
        positions[:valid_length].copy_(torch.cat(chunks))
    return positions


def interval_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: DFlashGlobalContextMask,
    *,
    plan_cache: DFlashIntervalPlanCache,
    owner: int,
    key_start: int,
    key_intervals: tuple[tuple[int, int], ...] | None = None,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_tile, key_tile = fa4_forward_sparse_tile(
        head_dim=int(query.shape[-1]),
        head_dim_v=int(value.shape[-1]),
        sparse_query_block_size=_FLEX_QUERY_BLOCK_SIZE,
    )
    plan = plan_cache.resolve(
        mask=mask,
        owner=int(owner),
        query_length=int(query.shape[1]),
        key_start=int(key_start),
        key_length=int(key.shape[1]),
        key_intervals=key_intervals,
        query_tile=query_tile,
        key_tile=key_tile,
    )
    try:
        from flash_attn.cute.interface import _flash_attn_fwd
    except (ImportError, AttributeError) as error:
        raise RuntimeError("DFlash interval attention requires flash-attn-4") from error
    output, lse = _flash_attn_fwd(
        q=query,
        k=key,
        v=value,
        softmax_scale=float(scale),
        causal=False,
        pack_gqa=int(query.shape[2]) != int(key.shape[2]),
        mask_mod=_interval_mask_mod(
            int(mask.block_size),
            int(mask.anchor_valid.shape[1]),
            int(mask.sliding_window or 0),
        ),
        aux_tensors=list(plan.mask_buffers),
        block_sparse_tensors=_forward_block_sparsity(plan.block_mask.as_tuple()),
        return_lse=True,
    )[:2]
    return output, lse


def interval_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    final_output: torch.Tensor,
    final_lse: torch.Tensor,
    grad_output: torch.Tensor,
    mask: DFlashGlobalContextMask,
    *,
    plan_cache: DFlashIntervalPlanCache,
    owner: int,
    key_start: int,
    key_intervals: tuple[tuple[int, int], ...] | None = None,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query_tile, key_tile = fa4_backward_sparse_tile(
        head_dim=int(query.shape[-1]),
        head_dim_v=int(value.shape[-1]),
        sparse_query_block_size=_FLEX_QUERY_BLOCK_SIZE,
    )
    plan = plan_cache.resolve(
        mask=mask,
        owner=int(owner),
        query_length=int(query.shape[1]),
        key_start=int(key_start),
        key_length=int(key.shape[1]),
        key_intervals=key_intervals,
        query_tile=query_tile,
        key_tile=key_tile,
    )
    try:
        from flash_attn.cute.interface import _flash_attn_bwd
    except (ImportError, AttributeError) as error:
        raise RuntimeError("DFlash interval attention requires flash-attn-4") from error
    return _flash_attn_bwd(
        q=query,
        k=key,
        v=value,
        out=final_output,
        dout=grad_output,
        lse=final_lse,
        softmax_scale=float(scale),
        causal=False,
        deterministic=False,
        pack_gqa=False,
        mask_mod=_interval_mask_mod(
            int(mask.block_size),
            int(mask.anchor_valid.shape[1]),
            int(mask.sliding_window or 0),
        ),
        aux_tensors=list(plan.mask_buffers),
        block_sparse_tensors=_backward_block_sparsity(plan.block_mask.as_tuple()),
        dlse=None,
    )


__all__ = [
    "DFlashIntervalPlanCache",
    "interval_backward",
    "interval_forward",
    "local_backward_from_state",
    "local_forward",
    "verify_dflash_fa4_runtime",
]
