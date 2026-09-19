# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Sequence-parallel reduce-scatter / scatter / all-gather region mappings."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

from dllm_parallel.core.parallel.tensor_parallel._common import (
    _ensure_group,
    _runtime_tp_group,
    _runtime_tp_rank,
    _runtime_tp_size,
)


def reduce_scatter_to_sequence_parallel_region(
    x: torch.Tensor,
    runtime: Any | None = None,
    *,
    group: Any | None = None,
    world_size: int | None = None,
) -> torch.Tensor:
    """Sum TP partial outputs and keep only this rank's sequence-row shard.

    This is the Megatron-style row-parallel output boundary used when the next
    layer consumes sequence-parallel hidden states. Forward performs
    reduce-scatter over dim 0; backward all-gathers row gradients.
    """

    if runtime is not None:
        group = _runtime_tp_group(runtime)
        world_size = _runtime_tp_size(runtime, int(world_size or 1))
    world_size = int(world_size or 1)
    if world_size <= 1:
        return x
    return _ReduceScatterToSequenceParallelRegion.apply(
        x,
        _ensure_group(group, world_size),
        world_size,
    )


def scatter_to_sequence_parallel_region(
    x: torch.Tensor,
    runtime: Any | None = None,
    *,
    group: Any | None = None,
    world_size: int | None = None,
) -> torch.Tensor:
    """Shard rows over TP ranks; backward all-gathers row gradients.

    This is the sequence-parallel boundary used after replicated/vocab-parallel
    embeddings. The input must already be padded so dim 0 divides the TP size.
    """

    if runtime is not None:
        group = _runtime_tp_group(runtime)
        world_size = _runtime_tp_size(runtime, int(world_size or 1))
    world_size = int(world_size or 1)
    if world_size <= 1:
        return x
    return _ScatterToSequenceParallelRegion.apply(
        x,
        _ensure_group(group, world_size),
        world_size,
        _runtime_tp_rank(runtime, 0),
    )


def gather_from_sequence_parallel_region(
    x: torch.Tensor,
    runtime: Any | None = None,
    *,
    group: Any | None = None,
    world_size: int | None = None,
    total_rows: int | None = None,
    reduce_scatter_grad: bool = False,
) -> torch.Tensor:
    """All-gather sequence-parallel rows.

    By default backward returns this rank's gradient slice.  Set
    ``reduce_scatter_grad`` when the gathered tensor feeds TP-sharded
    projections: their partial input gradients must be summed while restoring
    the sequence partition, which is the Megatron sequence-parallel boundary.
    """

    if runtime is not None:
        group = _runtime_tp_group(runtime)
        world_size = _runtime_tp_size(runtime, int(world_size or 1))
    world_size = int(world_size or 1)
    if world_size <= 1:
        return x[:total_rows] if total_rows is not None else x
    process_group = _ensure_group(group, world_size)
    if reduce_scatter_grad:
        gathered = _GatherFromSequenceParallelRegionWithReduceScatterGrad.apply(
            x,
            process_group,
            world_size,
        )
    else:
        gathered = _GatherFromSequenceParallelRegion.apply(
            x,
            process_group,
            world_size,
            _runtime_tp_rank(runtime, 0),
        )
    if total_rows is not None:
        gathered = gathered[: int(total_rows)]
    return gathered


def gather_active_from_sequence_parallel_region(
    x: torch.Tensor,
    *,
    packed_len: int,
    active_len: int,
    total_rows: int,
    runtime: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather only active-token rows from a sequence-parallel packed tensor.

    ``x`` is this rank's row shard of the flattened padded packed tensor. Rows
    are ordered as ``[batch, packed_token, hidden]`` flattened to
    ``[batch * packed_len, hidden]`` before padding. The returned hidden tensor
    is ordered by global flat row and the second return value contains
    ``[batch_index, packed_token_index]`` pairs for label lookup.
    """

    world_size = _runtime_tp_size(runtime, 1)
    if world_size <= 1:
        total = x[: int(total_rows)]
        ids = torch.arange(int(total_rows), device=x.device, dtype=torch.long)
        active = (ids % int(packed_len)) < int(active_len)
        ids = ids[active]
        pairs = torch.stack((ids // int(packed_len), ids % int(packed_len)), dim=-1)
        return total[active], pairs
    group = _ensure_group(_runtime_tp_group(runtime), world_size)
    rank = _runtime_tp_rank(runtime, 0)
    hidden, ids = _GatherActiveSequenceRows.apply(
        x,
        int(packed_len),
        int(active_len),
        int(total_rows),
        group,
        int(world_size),
        int(rank),
    )
    pairs = torch.stack((ids // int(packed_len), ids % int(packed_len)), dim=-1)
    return hidden, pairs


class _ReduceScatterToSequenceParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, group: Any, world_size: int) -> torch.Tensor:
        world_size = int(world_size)
        if x.shape[0] % world_size != 0:
            raise ValueError("sequence-parallel reduce-scatter requires divisible row count")
        ctx.group = group
        ctx.world_size = world_size
        ctx.input_shape = tuple(x.shape)
        output = torch.empty(
            (x.shape[0] // world_size, *x.shape[1:]),
            device=x.device,
            dtype=x.dtype,
        )
        dist.reduce_scatter_tensor(
            output,
            x.contiguous(),
            op=dist.ReduceOp.SUM,
            group=group,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_input = torch.empty(
            ctx.input_shape,
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        dist.all_gather_into_tensor(
            grad_input,
            grad_output.contiguous(),
            group=ctx.group,
        )
        return grad_input, None, None


class _ScatterToSequenceParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        group: Any,
        world_size: int,
        rank: int,
    ) -> torch.Tensor:
        if x.shape[0] % int(world_size) != 0:
            raise ValueError("sequence-parallel scatter requires divisible row count")
        ctx.group = group
        ctx.world_size = int(world_size)
        ctx.rank = int(rank)
        rows_per_rank = x.shape[0] // int(world_size)
        start = int(rank) * rows_per_rank
        stop = start + rows_per_rank
        return x[start:stop].contiguous()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        gathered = torch.empty(
            (ctx.world_size * grad_output.shape[0], *grad_output.shape[1:]),
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        dist.all_gather_into_tensor(gathered, grad_output.contiguous(), group=ctx.group)
        return gathered, None, None, None


class _GatherFromSequenceParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        group: Any,
        world_size: int,
        rank: int,
    ) -> torch.Tensor:
        ctx.world_size = int(world_size)
        ctx.rank = int(rank)
        ctx.local_rows = int(x.shape[0])
        ctx.group = group
        gathered = torch.empty(
            (ctx.world_size * ctx.local_rows, *x.shape[1:]),
            device=x.device,
            dtype=x.dtype,
        )
        dist.all_gather_into_tensor(gathered, x.contiguous(), group=group)
        return gathered

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        start = ctx.rank * ctx.local_rows
        stop = start + ctx.local_rows
        return grad_output[start:stop].contiguous(), None, None, None


class _GatherFromSequenceParallelRegionWithReduceScatterGrad(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        group: Any,
        world_size: int,
    ) -> torch.Tensor:
        ctx.world_size = int(world_size)
        ctx.local_rows = int(x.shape[0])
        ctx.group = group
        gathered = torch.empty(
            (ctx.world_size * ctx.local_rows, *x.shape[1:]),
            device=x.device,
            dtype=x.dtype,
        )
        dist.all_gather_into_tensor(gathered, x.contiguous(), group=group)
        return gathered

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_input = torch.empty(
            (ctx.local_rows, *grad_output.shape[1:]),
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        dist.reduce_scatter_tensor(
            grad_input,
            grad_output.contiguous(),
            op=dist.ReduceOp.SUM,
            group=ctx.group,
        )
        return grad_input, None, None


class _GatherActiveSequenceRows(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        packed_len: int,
        active_len: int,
        total_rows: int,
        group: Any,
        world_size: int,
        rank: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local_rows = int(x.shape[0])
        row_start = int(rank) * local_rows
        local_ids = torch.arange(
            row_start,
            row_start + local_rows,
            device=x.device,
            dtype=torch.long,
        )
        active = (local_ids < int(total_rows)) & (
            (local_ids % int(packed_len)) < int(active_len)
        )
        selected = x[active].contiguous()
        selected_ids = local_ids[active].contiguous()

        max_count = max(
            _active_rows_in_interval(
                row_start=peer_rank * local_rows,
                row_count=local_rows,
                total_rows=int(total_rows),
                packed_len=int(packed_len),
                active_len=int(active_len),
            )
            for peer_rank in range(int(world_size))
        )
        if max_count == 0:
            ctx.local_rows = local_rows
            ctx.local_selected_indices = torch.empty(0, device=x.device, dtype=torch.long)
            ctx.local_output_positions = torch.empty(0, device=x.device, dtype=torch.long)
            return x.new_empty((0, *x.shape[1:])), selected_ids

        padded = x.new_zeros((max_count, *x.shape[1:]))
        padded_ids = torch.full((max_count,), -1, device=x.device, dtype=torch.long)
        if selected.shape[0] > 0:
            padded[: selected.shape[0]].copy_(selected)
            padded_ids[: selected.shape[0]].copy_(selected_ids)

        gathered = torch.empty(
            (int(world_size) * max_count, *x.shape[1:]),
            device=x.device,
            dtype=x.dtype,
        )
        gathered_ids = torch.empty(
            (int(world_size) * max_count,),
            device=x.device,
            dtype=torch.long,
        )
        dist.all_gather_into_tensor(gathered, padded, group=group)
        dist.all_gather_into_tensor(gathered_ids, padded_ids, group=group)

        flat_hidden = gathered
        flat_ids = gathered_ids
        valid = flat_ids >= 0
        flat_hidden = flat_hidden[valid]
        flat_ids = flat_ids[valid]
        order = torch.argsort(flat_ids)
        sorted_ids = flat_ids[order].contiguous()
        sorted_hidden = flat_hidden[order].contiguous()

        local_selected_indices = torch.nonzero(active, as_tuple=False).squeeze(-1)
        if selected_ids.numel() > 0:
            local_mask = (sorted_ids >= row_start) & (sorted_ids < row_start + local_rows)
            local_output_positions = torch.nonzero(local_mask, as_tuple=False).squeeze(-1)
        else:
            local_output_positions = torch.empty(0, device=x.device, dtype=torch.long)
        ctx.local_rows = local_rows
        ctx.local_selected_indices = local_selected_indices
        ctx.local_output_positions = local_output_positions
        return sorted_hidden, sorted_ids

    @staticmethod
    def backward(ctx, grad_hidden: torch.Tensor, grad_ids: torch.Tensor | None = None):
        del grad_ids
        grad_local = grad_hidden.new_zeros((ctx.local_rows, *grad_hidden.shape[1:]))
        if ctx.local_output_positions.numel() > 0:
            grad_local.index_copy_(
                0,
                ctx.local_selected_indices,
                grad_hidden.index_select(0, ctx.local_output_positions),
            )
        return grad_local, None, None, None, None, None, None


def _active_rows_in_interval(
    *,
    row_start: int,
    row_count: int,
    total_rows: int,
    packed_len: int,
    active_len: int,
) -> int:
    """Count active packed rows in a rank interval using integer arithmetic."""

    packed_len = int(packed_len)
    if packed_len <= 0:
        raise ValueError("packed_len must be positive")
    start = max(0, int(row_start))
    stop = min(int(total_rows), start + max(0, int(row_count)))
    if stop <= start or int(active_len) <= 0:
        return 0
    active_len = min(int(active_len), packed_len)

    def prefix_count(position: int) -> int:
        complete, remainder = divmod(max(0, int(position)), packed_len)
        return complete * active_len + min(remainder, active_len)

    return prefix_count(stop) - prefix_count(start)
