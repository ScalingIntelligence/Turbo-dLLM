# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Native DFlash attention over local draft blocks and ring-sharded context."""

from __future__ import annotations

import math
from typing import Any

import torch

from dllm_parallel.core.attention.dflash_fa4 import (
    DFlashIntervalPlanCache,
    interval_backward as _fa4_interval_backward,
    interval_forward as _fa4_interval_forward,
    local_backward_from_state as _fa4_local_backward,
    local_forward as _fa4_local_forward,
)
from dllm_parallel.core.attention.layout import clean_intervals_for_runtime
from dllm_parallel.core.attention.masks import (
    DFlashGlobalContextMask,
    DFlashLocalBlockMask,
)
from dllm_parallel.core.attention.ring_transport import (
    _make_kv_ring_payload,
    _ring_exchange_flat_async,
    _ring_exchange_flat_wait,
    _ring_exchange_kv_payload_async,
    _ring_exchange_kv_payload_wait,
    _ring_exchange_tensors_async,
    _ring_exchange_tensors_wait,
    _ring_reduce_owner_payloads_p2p_flat,
)
from dllm_parallel.core.attention.cp_backend import owner_lengths
from dllm_parallel.core.kernels import cp_fusion, dflash_cp_fusion


DFLASH_ATTENTION_MAX_HEAD_DIM = 256
DFLASH_ATTENTION_HEAD_DIM_ALIGNMENT = 8


def validate_dflash_attention_head_dim(head_dim: int) -> None:
    """Validate dimensions implemented by the packaged DFlash FA4 kernels."""

    head_dim = int(head_dim)
    if (
        head_dim > DFLASH_ATTENTION_MAX_HEAD_DIM
        or head_dim % DFLASH_ATTENTION_HEAD_DIM_ALIGNMENT
    ):
        raise ValueError(
            "production DFlash attention requires head_dim <= "
            f"{DFLASH_ATTENTION_MAX_HEAD_DIM} and divisible by "
            f"{DFLASH_ATTENTION_HEAD_DIM_ALIGNMENT}"
        )


def dflash_attention(
    *,
    query: torch.Tensor,
    local_key: torch.Tensor,
    local_value: torch.Tensor,
    global_key: torch.Tensor,
    global_value: torch.Tensor,
    local_attn_mask: DFlashLocalBlockMask,
    global_attn_mask: DFlashGlobalContextMask,
    interval_plan_cache: DFlashIntervalPlanCache | None = None,
    scale: float | None,
    runtime: Any | None,
    global_seq_len: int,
    global_intervals: tuple[tuple[int, int], ...] | None = None,
    global_intervals_by_owner: (tuple[tuple[tuple[int, int], ...], ...] | None) = None,
) -> torch.Tensor:
    _validate_inputs(
        query,
        local_key,
        local_value,
        global_key,
        global_value,
        local_attn_mask,
        global_attn_mask,
    )
    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])
    rotate_query = _prefer_query_rotation(
        query=query,
        global_key=global_key,
        local_mask=local_attn_mask,
        global_mask=global_attn_mask,
        runtime=runtime,
        global_seq_len=int(global_seq_len),
    )
    group_ranks, _, _ = _ring_context(runtime)
    # At degree two the K/V-owner schedule needs three communication rounds per
    # layer (forward rotate, backward rotate, gradient return), while split
    # query needs four. The latter is latency-bound even though it moves fewer
    # bytes. Retain query rotation for larger rings, where K/V bandwidth grows
    # enough to dominate the extra collective round and stable dK recompute.
    if (
        getattr(runtime, "active_block_mode", None) == "dual_end"
        and len(group_ranks) <= 2
    ):
        rotate_query = False
    attention = _DFlashQueryRingAttention if rotate_query else _DFlashKVAttention
    if interval_plan_cache is None:
        interval_plan_cache = DFlashIntervalPlanCache()
    return attention.apply(
        query,
        local_key,
        local_value,
        global_key,
        global_value,
        local_attn_mask,
        global_attn_mask,
        interval_plan_cache,
        float(scale),
        runtime,
        int(global_seq_len),
        global_intervals,
        global_intervals_by_owner,
    )


def _prefer_query_rotation(
    *,
    query: torch.Tensor,
    global_key: torch.Tensor,
    local_mask: DFlashLocalBlockMask,
    global_mask: DFlashGlobalContextMask,
    runtime: Any | None,
    global_seq_len: int,
) -> bool:
    group_ranks, _, _ = _ring_context(runtime)
    ring_size = len(group_ranks)
    if ring_size <= 1 or int(global_mask.global_anchor_count) <= 0:
        return False
    owner_count = (
        int(getattr(runtime, "block_parallel_size", 1))
        if getattr(runtime, "active_block_mode", None) == "dual_end"
        else 1
    )
    global_anchor_count = int(global_mask.global_anchor_count)
    if owner_count <= 0 or global_anchor_count % owner_count:
        return False
    local_anchor_count = int(query.shape[1]) // int(local_mask.block_size)
    if local_anchor_count != global_anchor_count // owner_count:
        return False

    intervals = clean_intervals_for_runtime(global_seq_len, ring_size, runtime)
    max_shard_len = max(owner_lengths(intervals))
    scalar_bytes = int(query.element_size())
    query_elements = int(query.numel())
    statistic_elements = int(query.shape[0] * query.shape[1] * query.shape[2])
    key_elements = int(
        query.shape[0] * max_shard_len * global_key.shape[2] * global_key.shape[3]
    )
    metadata_bytes = sum(
        int(tensor.numel() * tensor.element_size())
        for tensor in (
            global_mask.context_starts,
            global_mask.context_stops,
            global_mask.anchor_valid,
        )
    )
    kv_transport_bytes = 6 * key_elements * scalar_bytes
    query_transport_bytes = (
        4 * query_elements * scalar_bytes
        + 8 * query_elements
        + 12 * statistic_elements
        + 2 * metadata_bytes
    )

    state_bytes = 4 * query_elements + 8 * statistic_elements
    forward_packet_bytes = query_elements * scalar_bytes + metadata_bytes
    backward_packet_bytes = (
        3 * query_elements * scalar_bytes + 4 * statistic_elements + metadata_bytes
    )
    query_forward_peak = (ring_size + 1) * state_bytes + 2 * forward_packet_bytes
    query_backward_peak = (
        (ring_size + 1) * 4 * query_elements
        + 8 * key_elements
        + 2 * backward_packet_bytes
    )
    kv_backward_peak = 12 * key_elements * scalar_bytes
    return (
        query_transport_bytes < kv_transport_bytes
        and max(query_forward_peak, query_backward_peak) < kv_backward_peak
    )


class _DFlashKVAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        query: torch.Tensor,
        local_key: torch.Tensor,
        local_value: torch.Tensor,
        global_key: torch.Tensor,
        global_value: torch.Tensor,
        local_mask: DFlashLocalBlockMask,
        global_mask: DFlashGlobalContextMask,
        interval_plan_cache: DFlashIntervalPlanCache,
        scale: float,
        runtime: Any | None,
        global_seq_len: int,
        global_intervals: tuple[tuple[int, int], ...] | None,
        global_intervals_by_owner: (tuple[tuple[tuple[int, int], ...], ...] | None),
    ) -> torch.Tensor:
        group_ranks, local_rank, group = _ring_context(runtime)
        if global_intervals_by_owner is not None:
            intervals_by_owner = tuple(global_intervals_by_owner)
        elif len(group_ranks) > 1:
            intervals_by_owner = clean_intervals_for_runtime(
                int(global_seq_len), len(group_ranks), runtime
            )
        else:
            intervals_by_owner = (
                (
                    tuple(global_intervals)
                    if global_intervals is not None
                    else ((0, int(global_seq_len)),)
                ),
            )
        shard_lengths = owner_lengths(intervals_by_owner)
        if int(global_key.shape[1]) != int(shard_lengths[local_rank]):
            raise ValueError("DFlash context K/V does not match runtime ownership")

        numerator, m, l = _empty_accumulator(query)
        local_output, local_lse = _local_forward(
            query,
            local_key,
            local_value,
            block_size=int(local_mask.block_size),
            causal=bool(local_mask.causal),
            scale=float(scale),
        )
        dflash_cp_fusion.merge_bshd_(numerator, m, l, local_output, local_lse)

        max_shard_len = _packed_context_capacity(
            shard_lengths,
            mask=global_mask,
            global_seq_len=int(global_seq_len),
        )
        current_flat, current_key, current_value = _make_kv_ring_payload(
            _pad_sequence(global_key, max_shard_len),
            _pad_sequence(global_value, max_shard_len),
        )
        initial_flat = current_flat
        backward_start_flat = (
            torch.empty_like(initial_flat) if len(group_ranks) > 1 else initial_flat
        )
        scratch_flat = torch.empty_like(initial_flat) if len(group_ranks) > 3 else None
        backward_key = current_key
        backward_value = current_value

        for step in range(len(group_ranks)):
            owner = (local_rank + step) % len(group_ranks)
            if step != len(group_ranks) - 1:
                if step == 0:
                    recv_flat = backward_start_flat
                elif step % 2 == 1:
                    recv_flat = initial_flat
                else:
                    if scratch_flat is None:
                        raise RuntimeError("DFlash ring scratch buffer is unavailable")
                    recv_flat = scratch_flat
                exchange = _ring_exchange_kv_payload_async(
                    current_flat,
                    recv=recv_flat,
                    key_shape=current_key.shape,
                    key_numel=current_key.numel(),
                    local_rank=local_rank,
                    group_ranks=group_ranks,
                    group=group,
                    phase="forward",
                )
            else:
                exchange = None

            owner_intervals = intervals_by_owner[owner]
            shard_output, shard_lse = _interval_forward(
                query,
                current_key,
                current_value,
                global_mask,
                plan_cache=interval_plan_cache,
                owner=int(owner),
                key_start=int(owner_intervals[0][0]),
                key_intervals=owner_intervals,
                scale=float(scale),
            )
            dflash_cp_fusion.merge_bshd_(
                numerator,
                m,
                l,
                shard_output,
                shard_lse,
            )

            if exchange is not None:
                current_flat, current_key, current_value = (
                    _ring_exchange_kv_payload_wait(exchange)
                )
                if step == 0:
                    backward_key = current_key
                    backward_value = current_value

        output, final_lse = _finalize(numerator, m, l, query.dtype)
        ctx.runtime = runtime
        ctx.group_ranks = group_ranks
        ctx.local_rank = local_rank
        ctx.intervals_by_owner = intervals_by_owner
        ctx.shard_lengths = tuple(int(length) for length in shard_lengths)
        ctx.max_shard_len = int(max_shard_len)
        ctx.scale = float(scale)
        ctx.local_mask = local_mask.detach()
        ctx.global_mask = global_mask.detach()
        ctx.interval_plan_cache = interval_plan_cache
        ctx.save_for_backward(
            query.detach(),
            local_key.detach(),
            local_value.detach(),
            backward_key.detach(),
            backward_value.detach(),
            output.detach(),
            final_lse.detach(),
        )
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        (
            query,
            local_key,
            local_value,
            backward_key,
            backward_value,
            output,
            final_lse,
        ) = ctx.saved_tensors
        grad_query, grad_local_key, grad_local_value = _local_backward(
            query,
            local_key,
            local_value,
            output,
            final_lse,
            grad_output.contiguous(),
            block_size=int(ctx.local_mask.block_size),
            causal=bool(ctx.local_mask.causal),
            scale=float(ctx.scale),
        )
        grad_global_key, grad_global_value, global_grad_query = _global_ring_backward(
            query=query,
            backward_key=backward_key,
            backward_value=backward_value,
            output=output,
            final_lse=final_lse,
            grad_output=grad_output.contiguous(),
            mask=ctx.global_mask,
            plan_cache=ctx.interval_plan_cache,
            scale=float(ctx.scale),
            local_rank=int(ctx.local_rank),
            group_ranks=list(ctx.group_ranks),
            intervals_by_owner=ctx.intervals_by_owner,
            shard_lengths=list(ctx.shard_lengths),
            max_shard_len=int(ctx.max_shard_len),
            group=_ring_context(ctx.runtime)[2],
        )
        grad_query = grad_query.float()
        grad_query.add_(global_grad_query)
        return (
            grad_query.to(query.dtype),
            grad_local_key.to(local_key.dtype),
            grad_local_value.to(local_value.dtype),
            grad_global_key.to(backward_key.dtype),
            grad_global_value.to(backward_value.dtype),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _DFlashQueryRingAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        query: torch.Tensor,
        local_key: torch.Tensor,
        local_value: torch.Tensor,
        global_key: torch.Tensor,
        global_value: torch.Tensor,
        local_mask: DFlashLocalBlockMask,
        global_mask: DFlashGlobalContextMask,
        interval_plan_cache: DFlashIntervalPlanCache,
        scale: float,
        runtime: Any,
        global_seq_len: int,
        global_intervals: tuple[tuple[int, int], ...] | None,
        global_intervals_by_owner: (tuple[tuple[tuple[int, int], ...], ...] | None),
    ) -> torch.Tensor:
        del global_intervals_by_owner
        group_ranks, local_rank, group = _ring_context(runtime)
        intervals_by_owner = clean_intervals_for_runtime(
            int(global_seq_len),
            len(group_ranks),
            runtime,
        )
        local_intervals = (
            tuple(global_intervals)
            if global_intervals is not None
            else intervals_by_owner[local_rank]
        )
        local_shard_length = sum(stop - start for start, stop in local_intervals)
        if int(global_key.shape[1]) != int(local_shard_length):
            raise ValueError("DFlash context K/V does not match runtime ownership")
        numerator, m, l = _query_ring_forward_states(
            query=query,
            global_key=global_key,
            global_value=global_value,
            mask=global_mask,
            plan_cache=interval_plan_cache,
            scale=float(scale),
            local_rank=local_rank,
            group_ranks=group_ranks,
            intervals=local_intervals,
            group=group,
        )
        local_output, local_lse = _local_forward(
            query,
            local_key,
            local_value,
            block_size=int(local_mask.block_size),
            causal=bool(local_mask.causal),
            scale=float(scale),
        )
        dflash_cp_fusion.merge_bshd_(numerator, m, l, local_output, local_lse)
        output, final_lse = _finalize(numerator, m, l, query.dtype)
        ctx.runtime = runtime
        ctx.group_ranks = group_ranks
        ctx.local_rank = local_rank
        ctx.intervals = local_intervals
        ctx.scale = float(scale)
        ctx.local_mask = local_mask.detach()
        ctx.global_mask = global_mask.detach()
        ctx.interval_plan_cache = interval_plan_cache
        ctx.save_for_backward(
            query.detach(),
            local_key.detach(),
            local_value.detach(),
            global_key.detach(),
            global_value.detach(),
            output.detach(),
            final_lse.detach(),
        )
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        query, local_key, local_value, global_key, global_value, output, final_lse = (
            ctx.saved_tensors
        )
        grad_output = grad_output.contiguous()
        grad_query, grad_local_key, grad_local_value = _local_backward(
            query,
            local_key,
            local_value,
            output,
            final_lse,
            grad_output,
            block_size=int(ctx.local_mask.block_size),
            causal=bool(ctx.local_mask.causal),
            scale=float(ctx.scale),
        )
        debug_enabled = bool(ctx.global_mask.debug_nonfinite_attention)
        _debug_require_finite(
            (grad_query, grad_local_key, grad_local_value),
            enabled=debug_enabled,
            phase="query_ring_local_backward",
            owner=int(ctx.local_rank),
            key_start=0,
            key_stop=int(local_key.shape[1]),
        )
        global_grad_query, grad_global_key, grad_global_value = (
            _query_ring_backward_states(
                query=query,
                global_key=global_key,
                global_value=global_value,
                output=output,
                final_lse=final_lse,
                grad_output=grad_output,
                mask=ctx.global_mask,
                plan_cache=ctx.interval_plan_cache,
                scale=float(ctx.scale),
                local_rank=int(ctx.local_rank),
                group_ranks=list(ctx.group_ranks),
                intervals=ctx.intervals,
                group=_ring_context(ctx.runtime)[2],
                stable_split_query=(
                    getattr(ctx.runtime, "active_block_mode", None) == "dual_end"
                ),
            )
        )
        _debug_require_finite(
            (global_grad_query, grad_global_key, grad_global_value),
            enabled=debug_enabled,
            phase="query_ring_global_backward",
            owner=int(ctx.local_rank),
            key_start=0,
            key_stop=int(global_key.shape[1]),
        )
        grad_query = grad_query.float()
        grad_query.add_(global_grad_query)
        _debug_require_finite(
            (grad_query,),
            enabled=debug_enabled,
            phase="query_ring_combined_backward",
            owner=int(ctx.local_rank),
            key_start=0,
            key_stop=int(global_key.shape[1]),
        )
        returned_gradients = (
            grad_query.to(query.dtype),
            grad_local_key.to(local_key.dtype),
            grad_local_value.to(local_value.dtype),
            grad_global_key.to(global_key.dtype),
            grad_global_value.to(global_value.dtype),
        )
        _debug_require_finite(
            returned_gradients,
            enabled=debug_enabled,
            phase="query_ring_returned_backward",
            owner=int(ctx.local_rank),
            key_start=0,
            key_stop=int(global_key.shape[1]),
        )
        return (
            *returned_gradients,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _query_ring_forward_states(
    *,
    query: torch.Tensor,
    global_key: torch.Tensor,
    global_value: torch.Tensor,
    mask: DFlashGlobalContextMask,
    plan_cache: DFlashIntervalPlanCache,
    scale: float,
    local_rank: int,
    group_ranks: list[int],
    intervals: tuple[tuple[int, int], ...],
    group: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    size = len(group_ranks)
    numerator_shape = (
        query.shape[0],
        query.shape[2],
        query.shape[1],
        query.shape[3],
    )
    statistic_shape = numerator_shape[:-1]
    numerators = torch.zeros(
        (size, *numerator_shape),
        device=query.device,
        dtype=torch.float32,
    )
    maxima = torch.full(
        (size, *statistic_shape),
        -torch.inf,
        device=query.device,
        dtype=torch.float32,
    )
    normalizers = torch.zeros_like(maxima)
    current = (
        query.contiguous(),
        mask.context_starts.contiguous(),
        mask.context_stops.contiguous(),
        mask.anchor_valid.contiguous(),
    )
    receive = tuple(
        tuple(torch.empty_like(tensor) for tensor in current) for _ in range(2)
    )
    send_rank = group_ranks[(local_rank - 1) % size]
    recv_rank = group_ranks[(local_rank + 1) % size]
    for step in range(size):
        origin = (local_rank + step) % size
        exchange = None
        if step != size - 1:
            exchange = _ring_exchange_tensors_async(
                current,
                receive[step % 2],
                local_rank=local_rank,
                send_rank=send_rank,
                recv_rank=recv_rank,
                group=group,
                phase="forward",
            )
        current_query, context_starts, context_stops, anchor_valid = current
        current_mask = DFlashGlobalContextMask(
            context_starts=context_starts,
            context_stops=context_stops,
            anchor_valid=anchor_valid,
            block_size=int(mask.block_size),
            global_anchor_count=int(mask.global_anchor_count),
            sliding_window=mask.sliding_window,
        )
        shard_output, shard_lse = _interval_forward(
            current_query,
            global_key,
            global_value,
            current_mask,
            plan_cache=plan_cache,
            owner=int(origin),
            key_start=int(intervals[0][0]),
            key_intervals=intervals,
            scale=float(scale),
        )
        dflash_cp_fusion.merge_bshd_(
            numerators[origin],
            maxima[origin],
            normalizers[origin],
            shard_output,
            shard_lse,
        )
        if exchange is not None:
            current = _ring_exchange_tensors_wait(exchange)
    return _ring_reduce_dflash_states(
        numerators,
        maxima,
        normalizers,
        local_rank=local_rank,
        group_ranks=group_ranks,
        group=group,
    )


def _ring_reduce_dflash_states(
    numerators: torch.Tensor,
    maxima: torch.Tensor,
    normalizers: torch.Tensor,
    *,
    local_rank: int,
    group_ranks: list[int],
    group: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    size = len(group_ranks)
    scratch = (
        torch.empty_like(numerators[0]),
        torch.empty_like(maxima[0]),
        torch.empty_like(normalizers[0]),
    )
    send_rank = group_ranks[(local_rank - 1) % size]
    recv_rank = group_ranks[(local_rank + 1) % size]
    for step in range(size - 1):
        send_owner = (local_rank + step + 1) % size
        recv_owner = (local_rank + step + 2) % size
        received = _ring_exchange_tensors_wait(
            _ring_exchange_tensors_async(
                (
                    numerators[send_owner],
                    maxima[send_owner],
                    normalizers[send_owner],
                ),
                scratch,
                local_rank=local_rank,
                send_rank=send_rank,
                recv_rank=recv_rank,
                group=group,
                phase="forward",
            )
        )
        dflash_cp_fusion.merge_state_(
            numerators[recv_owner],
            maxima[recv_owner],
            normalizers[recv_owner],
            received[0],
            received[1],
            received[2],
        )
    return numerators[local_rank], maxima[local_rank], normalizers[local_rank]


def _query_ring_backward_states(
    *,
    query: torch.Tensor,
    global_key: torch.Tensor,
    global_value: torch.Tensor,
    output: torch.Tensor,
    final_lse: torch.Tensor,
    grad_output: torch.Tensor,
    mask: DFlashGlobalContextMask,
    plan_cache: DFlashIntervalPlanCache,
    scale: float,
    local_rank: int,
    group_ranks: list[int],
    intervals: tuple[tuple[int, int], ...],
    group: Any,
    stable_split_query: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    size = len(group_ranks)
    query_grads = torch.zeros(
        (size, *query.shape),
        device=query.device,
        dtype=torch.float32,
    )
    key_grad = torch.zeros_like(global_key, dtype=torch.float32)
    value_grad = torch.zeros_like(global_value, dtype=torch.float32)
    current = (
        query.contiguous(),
        grad_output.contiguous(),
        output.contiguous(),
        final_lse.contiguous(),
        mask.context_starts.contiguous(),
        mask.context_stops.contiguous(),
        mask.anchor_valid.contiguous(),
    )
    receive = tuple(
        tuple(torch.empty_like(tensor) for tensor in current) for _ in range(2)
    )
    send_rank = group_ranks[(local_rank - 1) % size]
    recv_rank = group_ranks[(local_rank + 1) % size]
    for step in range(size):
        origin = (local_rank + step) % size
        exchange = None
        if step != size - 1:
            exchange = _ring_exchange_tensors_async(
                current,
                receive[step % 2],
                local_rank=local_rank,
                send_rank=send_rank,
                recv_rank=recv_rank,
                group=group,
                phase="backward",
            )
        (
            current_query,
            current_grad_output,
            current_output,
            current_lse,
            context_starts,
            context_stops,
            anchor_valid,
        ) = current
        current_mask = DFlashGlobalContextMask(
            context_starts=context_starts,
            context_stops=context_stops,
            anchor_valid=anchor_valid,
            block_size=int(mask.block_size),
            global_anchor_count=int(mask.global_anchor_count),
            sliding_window=mask.sliding_window,
        )
        if current_mask.sliding_window is not None and stable_split_query:
            # FA4's sparse GQA dK path is not reliable when one K/V owner sees
            # different query packets from fused BP ranks. Rebuild the exact
            # small sliding windows and use tensor-core batched matmuls. This
            # preserves query-ring communication and performs work only for
            # the attended pairs, instead of materializing dense 512K scores.
            grad_q, _unstable_grad_k, grad_v = _interval_backward(
                current_query,
                global_key,
                global_value,
                current_output,
                current_lse,
                current_grad_output,
                current_mask,
                plan_cache=plan_cache,
                owner=int(origin),
                key_start=int(intervals[0][0]),
                key_intervals=intervals,
                scale=float(scale),
            )
            del _unstable_grad_k
            _, grad_k, _ = _sliding_interval_backward(
                current_query,
                global_key,
                global_value,
                current_output,
                current_lse,
                current_grad_output,
                current_mask,
                key_intervals=intervals,
                scale=float(scale),
                key_gradient_only=True,
            )
        else:
            grad_q, grad_k, grad_v = _interval_backward(
                current_query,
                global_key,
                global_value,
                current_output,
                current_lse,
                current_grad_output,
                current_mask,
                plan_cache=plan_cache,
                owner=int(origin),
                key_start=int(intervals[0][0]),
                key_intervals=intervals,
                scale=float(scale),
            )
        query_grads[origin].add_(grad_q.float())
        key_grad.add_(grad_k.float())
        value_grad.add_(grad_v.float())
        if exchange is not None:
            current = _ring_exchange_tensors_wait(exchange)
    local_query_grad = _ring_reduce_owner_payloads_p2p_flat(
        query_grads.reshape(size, -1),
        local_rank=local_rank,
        group_ranks=group_ranks,
        group=group,
    ).view_as(query)
    return (
        local_query_grad,
        key_grad.to(global_key.dtype),
        value_grad.to(global_value.dtype),
    )


def _sliding_interval_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    final_output: torch.Tensor,
    final_lse: torch.Tensor,
    grad_output: torch.Tensor,
    mask: DFlashGlobalContextMask,
    *,
    key_intervals: tuple[tuple[int, int], ...],
    scale: float,
    key_gradient_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiate exact packed sliding windows with stable GQA matmuls.

    The final output and LSE include both local and every context owner, so the
    standard attention backward identity applies independently to this owner's
    probabilities. K/V rows are gathered into one fixed window per anchor and
    accumulated back into the owner's disjoint packed representation.
    """

    if mask.sliding_window is None:
        raise ValueError("sliding interval backward requires a sliding window")
    batch, query_length, query_heads, head_dim = query.shape
    key_heads = int(key.shape[2])
    if query_heads % key_heads:
        raise ValueError("DFlash query heads must divide by key/value heads")
    block_size = int(mask.block_size)
    anchor_count = int(mask.anchor_valid.shape[1])
    if query_length != anchor_count * block_size:
        raise ValueError("DFlash query packet does not match its anchor metadata")
    repeats = query_heads // key_heads
    history = max(1, int(mask.sliding_window) - 1)

    position_chunks = [
        torch.arange(start, stop, device=query.device, dtype=torch.int64)
        for start, stop in key_intervals
        if int(stop) > int(start)
    ]
    packed_positions = torch.cat(position_chunks)
    if int(packed_positions.numel()) != int(key.shape[1]):
        raise ValueError("DFlash packed K/V rows do not match their intervals")

    stops = mask.context_stops.to(torch.int64)
    starts = torch.maximum(
        mask.context_starts.to(torch.int64),
        stops - history,
    )
    offsets = torch.arange(history, device=query.device, dtype=torch.int64)
    wanted_positions = starts[..., None] + offsets
    window_valid = (
        mask.anchor_valid[..., None]
        & (wanted_positions < stops[..., None])
    )
    packed_indices = torch.searchsorted(packed_positions, wanted_positions)
    safe_indices = packed_indices.clamp_max(int(key.shape[1]) - 1)
    window_valid &= packed_indices < int(key.shape[1])
    window_valid &= packed_positions[safe_indices] == wanted_positions

    batch_indices = torch.arange(batch, device=query.device)[:, None, None]
    gathered_key = key[batch_indices, safe_indices]
    gathered_value = value[batch_indices, safe_indices]

    query_grouped = (
        query.view(batch, anchor_count, block_size, key_heads, repeats, head_dim)
        .permute(0, 1, 3, 4, 2, 5)
        .contiguous()
    )
    output_grouped = (
        final_output.view(
            batch, anchor_count, block_size, key_heads, repeats, head_dim
        )
        .permute(0, 1, 3, 4, 2, 5)
        .contiguous()
    )
    grad_output_grouped = (
        grad_output.view(
            batch, anchor_count, block_size, key_heads, repeats, head_dim
        )
        .permute(0, 1, 3, 4, 2, 5)
        .contiguous()
    )
    key_transposed = gathered_key.permute(0, 1, 3, 4, 2).unsqueeze(3)
    value_transposed = gathered_value.permute(0, 1, 3, 4, 2).unsqueeze(3)
    scores = torch.matmul(query_grouped, key_transposed) * float(scale)

    query_offsets = torch.arange(
        block_size, device=query.device, dtype=torch.int64
    )
    query_starts = stops[..., None] + query_offsets - history
    allowed = window_valid[..., None, :] & (
        wanted_positions[..., None, :] >= query_starts[..., :, None]
    )
    scores.masked_fill_(~allowed[:, :, None, None], -torch.inf)
    lse = (
        final_lse.view(batch, key_heads, repeats, anchor_count, block_size)
        .permute(0, 3, 1, 2, 4)
        .contiguous()
    )
    probability = torch.exp(scores.float() - lse[..., None])

    grad_probability = torch.matmul(grad_output_grouped, value_transposed)
    delta = (grad_output_grouped.float() * output_grouped.float()).sum(dim=-1)
    grad_score = (
        probability * (grad_probability.float() - delta[..., None])
    ).to(query.dtype)

    grad_key_windows = torch.matmul(
        grad_score.transpose(-2, -1), query_grouped
    ).sum(dim=3).mul_(float(scale))
    grad_key = torch.zeros_like(key, dtype=torch.float32)
    scatter_indices = safe_indices[..., None, None].expand(
        batch, anchor_count, history, key_heads, head_dim
    ).reshape(batch, anchor_count * history, key_heads, head_dim)
    grad_key.scatter_add_(
        1,
        scatter_indices,
        grad_key_windows.permute(0, 1, 3, 2, 4)
        .reshape(batch, anchor_count * history, key_heads, head_dim)
        .float(),
    )
    if key_gradient_only:
        empty = key.new_empty((0,))
        return empty, grad_key.to(key.dtype), empty

    grad_query = torch.matmul(
        grad_score,
        gathered_key.permute(0, 1, 3, 2, 4).unsqueeze(3),
    ).mul_(float(scale))
    grad_value_windows = torch.matmul(
        probability.to(query.dtype).transpose(-2, -1), grad_output_grouped
    ).sum(dim=3)
    grad_value = torch.zeros_like(value, dtype=torch.float32)
    grad_value.scatter_add_(
        1,
        scatter_indices,
        grad_value_windows.permute(0, 1, 3, 2, 4)
        .reshape(batch, anchor_count * history, key_heads, head_dim)
        .float(),
    )
    return (
        grad_query.permute(0, 1, 4, 2, 3, 5)
        .reshape_as(query)
        .to(query.dtype),
        grad_key.to(key.dtype),
        grad_value.to(value.dtype),
    )


def _global_ring_backward(
    *,
    query: torch.Tensor,
    backward_key: torch.Tensor,
    backward_value: torch.Tensor,
    output: torch.Tensor,
    final_lse: torch.Tensor,
    grad_output: torch.Tensor,
    mask: DFlashGlobalContextMask,
    plan_cache: DFlashIntervalPlanCache,
    scale: float,
    local_rank: int,
    group_ranks: list[int],
    intervals_by_owner: tuple[tuple[tuple[int, int], ...], ...],
    shard_lengths: list[int],
    max_shard_len: int,
    group: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(group_ranks) == 1:
        grad_query = torch.zeros_like(query, dtype=torch.float32)
        grad_key = torch.zeros_like(backward_key)
        grad_value = torch.zeros_like(backward_value)
        intervals = intervals_by_owner[0]
        interval_grad_q, interval_grad_k, interval_grad_v = _interval_backward(
            query,
            backward_key,
            backward_value,
            output,
            final_lse,
            grad_output,
            mask,
            plan_cache=plan_cache,
            owner=0,
            key_start=int(intervals[0][0]),
            key_intervals=intervals,
            scale=scale,
        )
        grad_query.add_(interval_grad_q.float())
        grad_key.copy_(interval_grad_k)
        grad_value.copy_(interval_grad_v)
        local_len = int(shard_lengths[0])
        return (
            grad_key[:, :local_len],
            grad_value[:, :local_len],
            grad_query,
        )

    kv_flat, key_padded, value_padded = _make_kv_ring_payload(
        _pad_sequence(backward_key, max_shard_len),
        _pad_sequence(backward_value, max_shard_len),
    )
    key_numel = key_padded.numel()
    buffers = torch.empty(
        (2, 2, kv_flat.numel()), device=kv_flat.device, dtype=kv_flat.dtype
    )
    buffers[0, 0].copy_(kv_flat)
    grad_query = torch.zeros_like(query, dtype=torch.float32)
    local_grad = torch.zeros_like(kv_flat)
    returned_grad = None

    for step in range(len(group_ranks)):
        owner = (local_rank + step + 1) % len(group_ranks)
        send_buffer = buffers[step % 2]
        recv_buffer = buffers[(step + 1) % 2]
        send_payload = (
            send_buffer[0]
            if step == 0
            else (send_buffer[1] if step == len(group_ranks) - 1 else send_buffer)
        )
        recv_payload = (
            recv_buffer[0]
            if step == 0
            else (recv_buffer[1] if step == len(group_ranks) - 1 else recv_buffer)
        )
        exchange = _ring_exchange_flat_async(
            send_payload,
            recv_payload,
            local_rank=local_rank,
            send_rank=group_ranks[(local_rank - 1) % len(group_ranks)],
            recv_rank=group_ranks[(local_rank + 1) % len(group_ranks)],
            group=group,
            communication_phase="backward",
        )
        current = send_buffer[0]
        current_key = current[:key_numel].view(key_padded.shape)
        current_value = current[key_numel:].view(value_padded.shape)
        local_grad.zero_()
        grad_key = local_grad[:key_numel].view(key_padded.shape)
        grad_value = local_grad[key_numel:].view(value_padded.shape)
        intervals = intervals_by_owner[owner]
        grad_q, grad_k, grad_v = _interval_backward(
            query,
            current_key,
            current_value,
            output,
            final_lse,
            grad_output,
            mask,
            plan_cache=plan_cache,
            owner=int(owner),
            key_start=int(intervals[0][0]),
            key_intervals=intervals,
            scale=float(scale),
        )
        grad_query.add_(grad_q.float())
        grad_key.add_(grad_k)
        grad_value.add_(grad_v)
        _ring_exchange_flat_wait(exchange)
        returned_grad = recv_buffer[1]
        if step == 0:
            returned_grad.copy_(local_grad)
        else:
            returned_grad.add_(local_grad)

    if returned_grad is None:
        raise RuntimeError("DFlash DKV ring did not return owner gradients")
    local_len = int(shard_lengths[local_rank])
    return (
        returned_grad[:key_numel].view(key_padded.shape)[:, :local_len],
        returned_grad[key_numel:].view(value_padded.shape)[:, :local_len],
        grad_query,
    )


def _interval_forward(
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
    return _fa4_interval_forward(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        mask,
        plan_cache=plan_cache,
        owner=int(owner),
        key_start=int(key_start),
        key_intervals=key_intervals,
        scale=float(scale),
    )


def _interval_backward(
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
    return _fa4_interval_backward(
        query,
        key,
        value,
        final_output,
        final_lse,
        grad_output,
        mask,
        plan_cache=plan_cache,
        owner=int(owner),
        key_start=int(key_start),
        key_intervals=key_intervals,
        scale=float(scale),
    )


def _local_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    block_size: int,
    causal: bool,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _fa4_local_forward(
        query,
        key,
        value,
        block_size=int(block_size),
        causal=bool(causal),
        scale=float(scale),
    )


def _local_backward(
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
    return _fa4_local_backward(
        query,
        key,
        value,
        final_output,
        final_lse,
        grad_output,
        block_size=int(block_size),
        causal=bool(causal),
        scale=float(scale),
    )


def _empty_accumulator(
    query: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, query_len, heads, head_dim = query.shape
    numerator = torch.zeros(
        (batch, heads, query_len, head_dim), device=query.device, dtype=torch.float32
    )
    m = torch.full(
        (batch, heads, query_len), -torch.inf, device=query.device, dtype=torch.float32
    )
    return numerator, m, torch.zeros_like(m)


def _debug_require_finite(
    tensors: tuple[torch.Tensor, ...],
    *,
    enabled: bool,
    phase: str,
    owner: int,
    key_start: int,
    key_stop: int,
) -> None:
    if not enabled:
        return
    for tensor_index, tensor in enumerate(tensors):
        if bool(torch.isfinite(tensor).all()):
            continue
        raise RuntimeError(
            "nonfinite DFlash attention tensor: "
            f"phase={phase} owner={owner} tensor={tensor_index} "
            f"key_interval=[{key_start},{key_stop}) shape={tuple(tensor.shape)}"
        )


def _finalize(
    numerator: torch.Tensor,
    m: torch.Tensor,
    l: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty(
        (
            numerator.shape[0],
            numerator.shape[2],
            numerator.shape[1],
            numerator.shape[3],
        ),
        device=numerator.device,
        dtype=dtype,
    )
    lse = torch.empty_like(m)
    cp_fusion.finalize_bshd_(numerator, m, l, output, lse)
    return output, lse


def _pad_sequence(tensor: torch.Tensor, length: int) -> torch.Tensor:
    if tensor.shape[1] == length:
        return tensor.contiguous()
    padding = torch.zeros(
        (tensor.shape[0], length - tensor.shape[1], *tensor.shape[2:]),
        device=tensor.device,
        dtype=tensor.dtype,
    )
    return torch.cat((tensor, padding), dim=1).contiguous()


def _packed_context_capacity(
    shard_lengths: list[int],
    *,
    mask: DFlashGlobalContextMask,
    global_seq_len: int,
) -> int:
    """Choose the smallest common tiled FA4/ring shape for packed K/V."""

    actual = max(int(length) for length in shard_lengths)
    del mask, global_seq_len
    # Every rank already knows all owner lengths, so the batch maximum is a
    # deterministic common ring shape. Padding for hypothetical anchor skew
    # previously moved tens of thousands of masked rows and made locality-aware
    # batches slower than dense random anchors. A 256-row tile is sufficient
    # for the native sparse kernel while keeping transport proportional to the
    # exact union of reachable context rows.
    tile = 256
    return max(tile, ((actual + tile - 1) // tile) * tile)


def _ring_context(runtime: Any | None) -> tuple[list[int], int, Any | None]:
    if runtime is None or not bool(
        getattr(runtime, "uses_context_parallel_attention", False)
    ):
        return [0], 0, None
    return (
        list(runtime.context_block_parallel_group_ranks),
        int(runtime.context_parallel_rank),
        runtime.context_block_parallel_group,
    )


def _validate_inputs(
    query: torch.Tensor,
    local_key: torch.Tensor,
    local_value: torch.Tensor,
    global_key: torch.Tensor,
    global_value: torch.Tensor,
    local_mask: DFlashLocalBlockMask,
    global_mask: DFlashGlobalContextMask,
) -> None:
    if not query.is_cuda or query.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(
            "DFlash production attention requires CUDA fp16/bf16 tensors"
        )
    if query.ndim != 4 or local_key.ndim != 4 or global_key.ndim != 4:
        raise ValueError("DFlash attention tensors must use BSHD layout")
    if local_key.shape != local_value.shape or global_key.shape != global_value.shape:
        raise ValueError("DFlash K/V pairs must have identical shapes")
    if query.shape[:2] != local_key.shape[:2] or query.shape[0] != global_key.shape[0]:
        raise ValueError("DFlash query and K/V batch/token dimensions are inconsistent")
    if query.shape[1] % int(local_mask.block_size):
        raise ValueError("DFlash query length must divide evenly by block_size")
    if local_mask.block_size != global_mask.block_size:
        raise ValueError("DFlash local and global block sizes differ")


__all__ = [
    "dflash_attention",
    "validate_dflash_attention_head_dim",
]
