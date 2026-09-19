# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Pure-context token-row sharding for Qwen3.8 token-local operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from dllm_parallel.core.attention.layout import active_query_offsets_for_context_rank
from dllm_parallel.core.profiling.operator_trace import communication_scope


@dataclass(frozen=True)
class PureCPTokenRowPlan:
    """One TP interval's compact pure-CP row ownership."""

    compact_input_indices: torch.Tensor
    active_output_positions_by_rank: torch.Tensor
    clean_output_positions: torch.Tensor
    active_rows: int
    clean_rows: int
    compute_rows: int
    output_rows: int
    logical_input_indices: torch.Tensor
    logical_token_positions: torch.Tensor
    logical_rows: int
    logical_packed_len: int
    logical_active_len: int
    gathered_rows: int


def build_pure_cp_token_row_plan(
    *,
    batch_size: int,
    packed_len: int,
    active_len: int,
    block_size: int,
    tensor_parallel_size: int,
    tensor_parallel_rank: int,
    context_parallel_size: int,
    context_parallel_rank: int,
    device: torch.device,
) -> PureCPTokenRowPlan | None:
    """Build a CP token-offset plan inside one TP sequence-row interval."""

    dimensions = {
        "batch_size": int(batch_size),
        "packed_len": int(packed_len),
        "active_len": int(active_len),
        "block_size": int(block_size),
        "tensor_parallel_size": int(tensor_parallel_size),
        "context_parallel_size": int(context_parallel_size),
    }
    if any(value <= 0 for value in dimensions.values()):
        raise ValueError("pure-CP token-row dimensions must be positive")
    if not 0 <= int(tensor_parallel_rank) < int(tensor_parallel_size):
        raise ValueError("tensor-parallel rank lies outside its group")
    if not 0 <= int(context_parallel_rank) < int(context_parallel_size):
        raise ValueError("context-parallel rank lies outside its group")
    if int(active_len) > int(packed_len):
        raise ValueError("active length must not exceed packed length")
    if int(active_len) % int(block_size):
        return None
    if int(block_size) % int(context_parallel_size):
        return None

    total_rows = int(batch_size) * int(packed_len)
    padded_rows = (
        (total_rows + int(tensor_parallel_size) - 1)
        // int(tensor_parallel_size)
        * int(tensor_parallel_size)
    )
    output_rows = padded_rows // int(tensor_parallel_size)
    def interval_plan(tp_rank: int) -> tuple[list[list[int]], list[int]] | None:
        row_start = int(tp_rank) * output_rows
        active_by_rank = [[] for _ in range(int(context_parallel_size))]
        offset_owner = [-1] * int(block_size)
        for rank in range(int(context_parallel_size)):
            offsets = active_query_offsets_for_context_rank(
                block_size=int(block_size),
                context_parallel_size=int(context_parallel_size),
                context_parallel_rank=rank,
                device=torch.device("cpu"),
            )
            for offset in offsets.tolist():
                offset_owner[int(offset)] = rank
        if any(owner < 0 for owner in offset_owner):
            return None
        clean: list[int] = []
        for local_row in range(output_rows):
            global_row = row_start + local_row
            if global_row >= total_rows:
                continue
            token_position = global_row % int(packed_len)
            if token_position < int(active_len):
                offset = token_position % int(block_size)
                owner = offset_owner[offset]
                active_by_rank[owner].append(local_row)
            else:
                clean.append(local_row)
        active_counts = {len(value) for value in active_by_rank}
        if len(active_counts) != 1:
            return None
        return active_by_rank, clean

    peer_plans = [interval_plan(rank) for rank in range(int(tensor_parallel_size))]
    if any(plan is None for plan in peer_plans):
        return None
    concrete_peer_plans = [plan for plan in peer_plans if plan is not None]
    compute_rows = max(
        len(active_by_rank[0]) + len(clean)
        for active_by_rank, clean in concrete_peer_plans
    )
    active_by_rank, clean = concrete_peer_plans[int(tensor_parallel_rank)]
    local_active = active_by_rank[int(context_parallel_rank)]
    compact_indices = torch.tensor(
        (*local_active, *clean),
        device=device,
        dtype=torch.long,
    )
    active_positions = torch.tensor(
        active_by_rank,
        device=device,
        dtype=torch.long,
    )
    clean_positions = torch.tensor(clean, device=device, dtype=torch.long)

    # Transformer Engine gathers sequence-parallel inputs in TP-rank-major
    # order.  Every TP interval stores its owned active rows first, followed by
    # its clean rows and optional padding.  Cache the permutation that restores
    # monotonically increasing packed-row order without materializing the
    # replicated active rows.
    gathered_rows: list[tuple[int, int]] = []
    for tp_rank, (peer_active_by_rank, peer_clean) in enumerate(concrete_peer_plans):
        peer_row_start = int(tp_rank) * output_rows
        peer_valid = (
            *peer_active_by_rank[int(context_parallel_rank)],
            *peer_clean,
        )
        for compact_offset, local_row in enumerate(peer_valid):
            global_row = peer_row_start + int(local_row)
            if global_row < total_rows:
                gathered_rows.append(
                    (global_row, int(tp_rank) * int(compute_rows) + compact_offset)
                )
    gathered_rows.sort(key=lambda item: item[0])
    logical_rows = len(gathered_rows)
    if logical_rows % int(batch_size):
        return None
    logical_packed_len = logical_rows // int(batch_size)
    logical_active_len = int(active_len) // int(context_parallel_size)
    expected_logical_packed_len = logical_active_len + (
        int(packed_len) - int(active_len)
    )
    if logical_packed_len != expected_logical_packed_len:
        return None
    logical_input_indices = torch.tensor(
        [gathered_index for _, gathered_index in gathered_rows],
        device=device,
        dtype=torch.long,
    )
    logical_token_positions = torch.tensor(
        [global_row % int(packed_len) for global_row, _ in gathered_rows],
        device=device,
        dtype=torch.long,
    ).view(int(batch_size), logical_packed_len)
    return PureCPTokenRowPlan(
        compact_input_indices=compact_indices,
        active_output_positions_by_rank=active_positions,
        clean_output_positions=clean_positions,
        active_rows=len(local_active),
        clean_rows=len(clean),
        compute_rows=int(compute_rows),
        output_rows=int(output_rows),
        logical_input_indices=logical_input_indices,
        logical_token_positions=logical_token_positions,
        logical_rows=int(logical_rows),
        logical_packed_len=int(logical_packed_len),
        logical_active_len=int(logical_active_len),
        gathered_rows=int(compute_rows) * int(tensor_parallel_size),
    )


def compact_pure_cp_token_rows(
    hidden_shard: torch.Tensor,
    plan: PureCPTokenRowPlan,
) -> torch.Tensor:
    """Select owned rows and append zero rows required by TP peers."""

    if int(hidden_shard.shape[0]) != int(plan.output_rows):
        raise ValueError("hidden shard row count does not match its pure-CP plan")
    selected = hidden_shard.index_select(0, plan.compact_input_indices)
    padding = int(plan.compute_rows) - int(selected.shape[0])
    if padding < 0:
        raise RuntimeError("pure-CP compact rows exceed the planned TP maximum")
    if padding == 0:
        return selected.contiguous()
    zeros = hidden_shard.new_zeros((padding, *hidden_shard.shape[1:]))
    return torch.cat((selected, zeros), dim=0).contiguous()


def select_pure_cp_logical_rows(
    gathered_rows: torch.Tensor,
    plan: PureCPTokenRowPlan,
) -> torch.Tensor:
    """Restore logical token order from a TP-gathered compact row tensor."""

    if int(gathered_rows.shape[0]) != int(plan.gathered_rows):
        raise ValueError("TP-gathered compact rows do not match the persistent plan")
    if int(plan.logical_input_indices.numel()) != int(plan.logical_rows):
        raise RuntimeError("persistent pure-CP logical permutation is inconsistent")
    return gathered_rows.index_select(0, plan.logical_input_indices).contiguous()


def expand_pure_cp_logical_rows(
    logical_rows: torch.Tensor,
    plan: PureCPTokenRowPlan,
) -> torch.Tensor:
    """Restore TP-rank-major compact rows, including required padding."""

    if int(logical_rows.shape[0]) != int(plan.logical_rows):
        raise ValueError("logical rows do not match the persistent pure-CP plan")
    gathered_rows = logical_rows.new_zeros(
        (
            int(plan.gathered_rows),
            *logical_rows.shape[1:],
        )
    )
    gathered_rows.index_copy_(0, plan.logical_input_indices, logical_rows)
    return gathered_rows


class _ReconstructPureCPTokenRows(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        compact_output: torch.Tensor,
        active_output_positions_by_rank: torch.Tensor,
        clean_output_positions: torch.Tensor,
        active_rows: int,
        clean_rows: int,
        output_rows: int,
        group: Any,
        context_parallel_size: int,
        context_parallel_rank: int,
    ) -> torch.Tensor:
        world_size = int(context_parallel_size)
        rank = int(context_parallel_rank)
        active_rows = int(active_rows)
        clean_rows = int(clean_rows)
        if dist.get_world_size(group) != world_size or dist.get_rank(group) != rank:
            raise RuntimeError("pure-CP token-row group metadata is inconsistent")
        if tuple(active_output_positions_by_rank.shape) != (world_size, active_rows):
            raise ValueError("active output positions do not match CP ownership")
        if int(clean_output_positions.numel()) != clean_rows:
            raise ValueError("clean output positions do not match compact rows")
        if active_rows + clean_rows > int(compact_output.shape[0]):
            raise ValueError("compact output does not contain its valid planned rows")

        local_active = compact_output[:active_rows].contiguous()
        if world_size == 1:
            gathered = local_active
        else:
            gathered = compact_output.new_empty(
                (world_size * active_rows, *compact_output.shape[1:])
            )
            input_bytes = int(local_active.numel()) * int(local_active.element_size())
            with communication_scope(
                domain="mlp",
                phase="forward",
                collective="all_gather_into_tensor",
                input_bytes=input_bytes,
                logical_bytes=input_bytes * (world_size - 1),
            ):
                dist.all_gather_into_tensor(gathered, local_active, group=group)

        output = compact_output.new_zeros(
            (int(output_rows), *compact_output.shape[1:])
        )
        owner_rows = gathered.view(
            world_size, active_rows, *compact_output.shape[1:]
        )
        for owner in range(world_size):
            output.index_copy_(
                0,
                active_output_positions_by_rank[owner],
                owner_rows[owner],
            )
        if clean_rows:
            output.index_copy_(
                0,
                clean_output_positions,
                compact_output[active_rows : active_rows + clean_rows],
            )

        ctx.group = group
        ctx.world_size = world_size
        ctx.rank = rank
        ctx.active_rows = active_rows
        ctx.clean_rows = clean_rows
        ctx.compact_shape = tuple(compact_output.shape)
        ctx.save_for_backward(active_output_positions_by_rank, clean_output_positions)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        active_positions, clean_positions = ctx.saved_tensors
        owner_major = torch.cat(
            tuple(
                grad_output.index_select(0, active_positions[owner])
                for owner in range(ctx.world_size)
            ),
            dim=0,
        ).contiguous()
        if ctx.world_size == 1:
            local_active_grad = owner_major
        else:
            local_active_grad = grad_output.new_empty(
                (ctx.active_rows, *grad_output.shape[1:])
            )
            input_bytes = int(owner_major.numel()) * int(owner_major.element_size())
            with communication_scope(
                domain="mlp",
                phase="backward",
                collective="reduce_scatter_tensor",
                input_bytes=input_bytes,
                logical_bytes=input_bytes * (ctx.world_size - 1) // ctx.world_size,
            ):
                dist.reduce_scatter_tensor(
                    local_active_grad,
                    owner_major,
                    group=ctx.group,
                )
        grad_compact = grad_output.new_zeros(ctx.compact_shape)
        grad_compact[: ctx.active_rows].copy_(local_active_grad)
        if ctx.clean_rows:
            grad_compact[
                ctx.active_rows : ctx.active_rows + ctx.clean_rows
            ].copy_(grad_output.index_select(0, clean_positions))
        return grad_compact, None, None, None, None, None, None, None, None


def reconstruct_pure_cp_token_rows(
    compact_output: torch.Tensor,
    *,
    plan: PureCPTokenRowPlan,
    group: Any,
    context_parallel_size: int,
    context_parallel_rank: int,
) -> torch.Tensor:
    """Reconstruct TP-local rows with a summed-gradient CP boundary."""

    return _ReconstructPureCPTokenRows.apply(
        compact_output,
        plan.active_output_positions_by_rank,
        plan.clean_output_positions,
        int(plan.active_rows),
        int(plan.clean_rows),
        int(plan.output_rows),
        group,
        int(context_parallel_size),
        int(context_parallel_rank),
    )


__all__ = [
    "PureCPTokenRowPlan",
    "build_pure_cp_token_row_plan",
    "compact_pure_cp_token_rows",
    "expand_pure_cp_logical_rows",
    "reconstruct_pure_cp_token_rows",
    "select_pure_cp_logical_rows",
]
