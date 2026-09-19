# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Exact context/head redistribution for Qwen3.8 Gated DeltaNet layers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.distributed as dist


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        tensor: torch.Tensor,
        group: Any,
        output_split_sizes: tuple[int, ...] | None,
        input_split_sizes: tuple[int, ...] | None,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        if dist.get_world_size(group) == 1:
            return tensor
        output_rows = (
            int(tensor.shape[0])
            if output_split_sizes is None
            else sum(output_split_sizes)
        )
        output = tensor.new_empty((output_rows, *tensor.shape[1:]))
        dist.all_to_all_single(
            output,
            tensor.contiguous(),
            output_split_sizes=(
                None if output_split_sizes is None else list(output_split_sizes)
            ),
            input_split_sizes=(
                None if input_split_sizes is None else list(input_split_sizes)
            ),
            group=group,
        )
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        grad_input = _AllToAll.apply(
            grad_output,
            ctx.group,
            ctx.input_split_sizes,
            ctx.output_split_sizes,
        )
        return grad_input, None, None, None


class _AllGatherHeads(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, tensor: torch.Tensor, group: Any) -> torch.Tensor:
        ctx.group = group
        world_size = dist.get_world_size(group)
        if world_size == 1:
            return tensor
        output = tensor.new_empty((world_size * tensor.shape[0], *tensor.shape[1:]))
        dist.all_gather_into_tensor(output, tensor.contiguous(), group=group)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        world_size = dist.get_world_size(ctx.group)
        if world_size == 1:
            return grad_output, None
        if grad_output.shape[0] % world_size:
            raise RuntimeError("gathered head gradient does not divide across ranks")
        grad_input = grad_output.new_empty(
            (grad_output.shape[0] // world_size, *grad_output.shape[1:])
        )
        dist.reduce_scatter_tensor(
            grad_input,
            grad_output.contiguous(),
            group=ctx.group,
        )
        return grad_input, None


def sequence_to_head_parallel_many(
    tensors: Sequence[torch.Tensor],
    *,
    group: Any,
    logical_order: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Redistribute equal-layout projections with one all-to-all collective."""

    if not tensors:
        raise ValueError("sequence-to-head redistribution requires at least one tensor")
    reference = tensors[0]
    if reference.ndim != 3:
        raise ValueError("sequence-to-head redistribution requires [B, S, H]")
    world_size = dist.get_world_size(group)
    batch, local_tokens = map(int, reference.shape[:2])
    sequence_first = []
    local_widths = []
    for tensor in tensors:
        if tensor.ndim != 3 or tuple(tensor.shape[:2]) != (batch, local_tokens):
            raise ValueError("all projected tensors must share batch and sequence dimensions")
        width = int(tensor.shape[-1])
        if width % world_size:
            raise ValueError("every projected head width must divide across context ranks")
        sequence_first.append(tensor.transpose(0, 1).contiguous())
        local_widths.append(width // world_size)

    sends = []
    chunks = [value.chunk(world_size, dim=-1) for value in sequence_first]
    for destination in range(world_size):
        sends.append(torch.cat([value[destination] for value in chunks], dim=-1))
    received = _AllToAll.apply(torch.cat(sends, dim=0), group, None, None)
    received = received.index_select(0, logical_order).transpose(0, 1).contiguous()
    return tuple(received.split(local_widths, dim=-1))


def head_to_sequence_parallel(
    tensor: torch.Tensor,
    *,
    group: Any,
    rank_major_order: torch.Tensor,
) -> torch.Tensor:
    """Map ``[B, S, H/P]`` head shards to ``[B, S/P, H]`` sequence shards."""

    if tensor.ndim != 3:
        raise ValueError("head-to-sequence redistribution requires [B, S, H]")
    world_size = dist.get_world_size(group)
    sequence_first = tensor.transpose(0, 1).contiguous()
    if int(sequence_first.shape[0]) % world_size:
        raise ValueError("sequence length must divide evenly across context ranks")
    send = sequence_first.index_select(0, rank_major_order)
    received = _AllToAll.apply(send, group, None, None)
    local_tokens = int(sequence_first.shape[0]) // world_size
    received = received.view(world_size, local_tokens, tensor.shape[0], -1)
    received = received.permute(1, 2, 0, 3).flatten(2)
    return received.transpose(0, 1).contiguous()


def gather_boundary_heads(
    tensor: torch.Tensor,
    *,
    group: Any,
    head_axis: int,
) -> torch.Tensor:
    """Gather recurrent-head shards on every CP rank with summed backward."""

    if not 0 <= head_axis < tensor.ndim:
        raise ValueError("head_axis lies outside boundary tensor")
    permutation = (head_axis, *[axis for axis in range(tensor.ndim) if axis != head_axis])
    inverse = tuple(permutation.index(axis) for axis in range(tensor.ndim))
    head_first = tensor.permute(permutation).contiguous()
    gathered = _AllGatherHeads.apply(head_first, group)
    return gathered.permute(inverse).contiguous()


def gather_active_blocks(
    tensor: torch.Tensor,
    *,
    group: Any,
    blocks_by_rank: Sequence[Sequence[int]],
    local_blocks: Sequence[int],
    block_size: int,
) -> torch.Tensor:
    """Gather disjoint complete active blocks and restore logical token order.

    Backward reduce-scatters the summed gradient from every CP consumer to the
    rank that evaluated each block.
    """

    if tensor.ndim < 3:
        raise ValueError("active-block gather requires [B, S, ...]")
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    if len(blocks_by_rank) != world_size:
        raise ValueError("active-block ownership must cover every context rank")
    normalized = tuple(tuple(int(block) for block in blocks) for blocks in blocks_by_rank)
    local = tuple(int(block) for block in local_blocks)
    if normalized[rank] != local:
        raise ValueError("local active blocks do not match the process-group rank")
    counts = {len(blocks) for blocks in normalized}
    if len(counts) != 1:
        raise ValueError("active-block gather currently requires equal block counts")
    local_count = len(local)
    if local_count <= 0 or int(block_size) <= 0:
        raise ValueError("active-block gather requires nonempty positive-size blocks")
    if int(tensor.shape[1]) != local_count * int(block_size):
        raise ValueError("active tensor length does not match local block ownership")
    flattened = [block for blocks in normalized for block in blocks]
    if sorted(flattened) != list(range(len(flattened))):
        raise ValueError("active blocks must partition a contiguous logical sequence")

    block_first = tensor.unflatten(1, (local_count, int(block_size))).transpose(0, 1)
    gathered = _AllGatherHeads.apply(block_first.contiguous(), group)
    logical_order = torch.argsort(
        torch.tensor(flattened, device=tensor.device, dtype=torch.long)
    )
    logical = gathered.index_select(0, logical_order)
    return logical.transpose(0, 1).flatten(1, 2).contiguous()


def route_boundary_heads(
    tensor: torch.Tensor,
    *,
    group: Any,
    blocks_by_rank: Sequence[Sequence[int]],
    local_blocks: Sequence[int],
    block_axis: int,
    head_axis: int,
) -> torch.Tensor:
    """Route each block's recurrent-head shards to its unique BP owner."""

    world_size = dist.get_world_size(group)
    if len(blocks_by_rank) != world_size:
        raise ValueError("block ownership must cover every context rank")
    if tuple(int(value) for value in blocks_by_rank[dist.get_rank(group)]) != tuple(
        int(value) for value in local_blocks
    ):
        raise ValueError("local block ownership does not match the process-group rank")
    if block_axis == head_axis:
        raise ValueError("block and head axes must differ")

    permutation = (
        block_axis,
        head_axis,
        *[
            axis
            for axis in range(tensor.ndim)
            if axis not in {block_axis, head_axis}
        ],
    )
    canonical = tensor.permute(permutation).contiguous()
    block_count = int(canonical.shape[0])
    flattened = [int(block) for blocks in blocks_by_rank for block in blocks]
    if sorted(flattened) != list(range(block_count)):
        raise ValueError("block ownership must partition every boundary state exactly once")
    destination_order = torch.tensor(
        flattened,
        device=tensor.device,
        dtype=torch.long,
    )
    send = canonical.index_select(0, destination_order)
    input_splits = tuple(len(blocks) for blocks in blocks_by_rank)
    local_count = len(local_blocks)
    output_splits = (local_count,) * world_size
    received = _AllToAll.apply(send, group, output_splits, input_splits)
    received = received.view(
        world_size,
        local_count,
        canonical.shape[1],
        *canonical.shape[2:],
    )
    received = received.permute(
        1,
        0,
        2,
        *range(3, received.ndim),
    ).flatten(1, 2)

    remaining_axes = [
        axis for axis in range(tensor.ndim) if axis not in {block_axis, head_axis}
    ]
    canonical_axes = [block_axis, head_axis, *remaining_axes]
    inverse = tuple(canonical_axes.index(axis) for axis in range(tensor.ndim))
    return received.permute(inverse).contiguous()


__all__ = [
    "gather_active_blocks",
    "gather_boundary_heads",
    "head_to_sequence_parallel",
    "route_boundary_heads",
    "sequence_to_head_parallel_many",
]
