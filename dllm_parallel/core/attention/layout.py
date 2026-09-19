# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Clean-token ownership and tensor layout helpers for CP/BP attention."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch

from dllm_parallel.core.attention.cp_backend import clean_shards_for_rank
from dllm_parallel.core.schedules.block import build_block_schedule


@dataclass(frozen=True)
class ContextParallelSequenceLayout:
    """Megatron-style zigzag ownership for the full BDLM computation graph.

    The logical sequence is ``[noisy tokens; clean tokens]``.  Each context
    rank owns one forward noisy interval and one reverse clean interval, exactly
    matching Megatron's two-chunk zigzag ownership. Diffusion blocks may cross
    rank boundaries because the native mask evaluates global token metadata.
    """

    intervals: tuple[tuple[int, int], ...]
    logical_positions: torch.Tensor
    model_positions: torch.Tensor
    noisy_local_indices: torch.Tensor
    noisy_positions: torch.Tensor


@dataclass(frozen=True)
class WindowedCleanExchangePlan:
    """Static variable-size exchange for one rank's visible clean windows."""

    send_counts: tuple[int, ...]
    receive_counts: tuple[int, ...]
    send_local_intervals_by_rank: tuple[tuple[tuple[int, int], ...], ...]
    receive_logical_intervals_by_rank: tuple[tuple[tuple[int, int], ...], ...]

    @property
    def send_tokens(self) -> int:
        return sum(self.send_counts)

    @property
    def receive_tokens(self) -> int:
        return sum(self.receive_counts)


def active_query_indices_for_context_rank(
    *,
    active_len: int,
    block_size: int,
    context_parallel_size: int,
    context_parallel_rank: int,
    device: torch.device,
) -> torch.Tensor:
    """Return one token-offset slice from every active block for pure CP."""

    if active_len <= 0 or block_size <= 0:
        raise ValueError("active_len and block_size must be positive")
    if active_len % block_size:
        raise ValueError("active_len must divide evenly by block_size")
    if context_parallel_size <= 0:
        raise ValueError("context_parallel_size must be positive")
    if block_size % context_parallel_size:
        raise ValueError("block_size must divide evenly by context_parallel_size")
    if not 0 <= context_parallel_rank < context_parallel_size:
        raise ValueError("context_parallel_rank out of range")

    local_offsets = active_query_offsets_for_context_rank(
        block_size=block_size,
        context_parallel_size=context_parallel_size,
        context_parallel_rank=context_parallel_rank,
        device=device,
    )
    block_starts = torch.arange(
        0,
        active_len,
        block_size,
        device=device,
        dtype=torch.long,
    )
    return (block_starts[:, None] + local_offsets[None]).reshape(-1)


def active_query_offsets_for_context_rank(
    *,
    block_size: int,
    context_parallel_size: int,
    context_parallel_rank: int,
    device: torch.device,
) -> torch.Tensor:
    """Return a two-ended, load-balanced CP slice of token offsets."""

    if block_size <= 0 or context_parallel_size <= 0:
        raise ValueError("block size and context parallel size must be positive")
    if block_size % context_parallel_size:
        raise ValueError("block_size must divide evenly by context_parallel_size")
    if not 0 <= context_parallel_rank < context_parallel_size:
        raise ValueError("context_parallel_rank out of range")

    tokens_per_rank = block_size // context_parallel_size
    if tokens_per_rank % 2:
        return torch.arange(
            context_parallel_rank * tokens_per_rank,
            (context_parallel_rank + 1) * tokens_per_rank,
            device=device,
            dtype=torch.long,
        )
    tokens_per_half = tokens_per_rank // 2
    tail_start = block_size - (context_parallel_rank + 1) * tokens_per_half
    return torch.cat(
        (
            torch.arange(
                context_parallel_rank * tokens_per_half,
                (context_parallel_rank + 1) * tokens_per_half,
                device=device,
                dtype=torch.long,
            ),
            torch.arange(
                tail_start,
                tail_start + tokens_per_half,
                device=device,
                dtype=torch.long,
            ),
        )
    )


def gather_context_parallel_sequence(
    noisy: torch.Tensor,
    clean: torch.Tensor,
    layout: ContextParallelSequenceLayout,
) -> torch.Tensor:
    """Gather local zigzag rows without materializing the global 2L sequence."""

    if noisy.shape != clean.shape:
        raise ValueError("noisy and clean tensors must have identical shapes")
    if noisy.ndim < 2:
        raise ValueError("noisy and clean tensors must include a sequence dimension")
    seq_len = int(noisy.shape[1])
    chunks: list[torch.Tensor] = []
    for start, stop in layout.intervals:
        if stop <= seq_len:
            chunks.append(noisy[:, start:stop])
        elif start >= seq_len:
            chunks.append(clean[:, start - seq_len:stop - seq_len])
        else:
            chunks.extend((noisy[:, start:seq_len], clean[:, :stop - seq_len]))
    if not chunks:
        return noisy[:, :0].contiguous()
    return torch.cat(chunks, dim=1).contiguous()


@lru_cache(maxsize=None)
def context_parallel_sequence_intervals(
    *,
    seq_len: int,
    context_parallel_size: int,
    rank: int,
) -> tuple[tuple[int, int], ...]:
    """Return this CP rank's intervals in logical ``[noisy; clean]`` order."""

    shards = clean_shards_for_rank(
        seq_len=2 * int(seq_len),
        context_parallel_size=int(context_parallel_size),
        rank=int(rank),
        layout="dual_chunk",
    )
    return tuple((int(shard.start), int(shard.stop)) for shard in shards)


@lru_cache(maxsize=None)
def all_context_parallel_sequence_intervals(
    *,
    seq_len: int,
    context_parallel_size: int,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    return tuple(
        context_parallel_sequence_intervals(
            seq_len=int(seq_len),
            context_parallel_size=int(context_parallel_size),
            rank=rank,
        )
        for rank in range(int(context_parallel_size))
    )


def clear_sequence_layout_caches() -> None:
    """Release shape-specific host metadata when the active length changes."""

    context_parallel_sequence_intervals.cache_clear()
    all_context_parallel_sequence_intervals.cache_clear()
    windowed_clean_exchange_plan.cache_clear()


def context_parallel_sequence_layout(
    *,
    seq_len: int,
    block_size: int,
    context_parallel_size: int,
    rank: int,
    device: torch.device,
) -> ContextParallelSequenceLayout:
    """Materialize the local full-sequence CP ownership metadata."""

    if int(seq_len) <= 0 or int(block_size) <= 0:
        raise ValueError("seq_len and block_size must be positive")
    if int(seq_len) % int(block_size) != 0:
        raise ValueError("seq_len must divide evenly by block_size")
    if int(context_parallel_size) <= 0:
        raise ValueError("context_parallel_size must be positive")
    if not 0 <= int(rank) < int(context_parallel_size):
        raise ValueError("rank must be in the context-parallel group")
    if int(seq_len) % int(context_parallel_size) != 0:
        raise ValueError(
            "pure context parallelism requires equal DualChunkSwap chunks; "
            "seq_len must divide evenly by context_parallel_size"
        )
    intervals = context_parallel_sequence_intervals(
        seq_len=int(seq_len),
        context_parallel_size=int(context_parallel_size),
        rank=int(rank),
    )
    logical_positions = torch.cat(
        [
            torch.arange(start, stop, device=device, dtype=torch.long)
            for start, stop in intervals
        ],
        dim=0,
    )
    query_is_clean = logical_positions >= int(seq_len)
    model_positions = torch.where(
        query_is_clean,
        logical_positions - int(seq_len),
        logical_positions,
    )
    noisy_local_indices = torch.nonzero(
        ~query_is_clean,
        as_tuple=False,
    ).flatten()
    noisy_positions = model_positions.index_select(0, noisy_local_indices)
    return ContextParallelSequenceLayout(
        intervals=intervals,
        logical_positions=logical_positions,
        model_positions=model_positions,
        noisy_local_indices=noisy_local_indices,
        noisy_positions=noisy_positions,
    )


def shard_bounds(
    num_items: int,
    num_shards: int,
    shard_index: int,
) -> tuple[int, int]:
    """Return the balanced contiguous shard interval for ``shard_index``."""

    if num_items < 0:
        raise ValueError("num_items must be non-negative")
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index out of range")
    base, remainder = divmod(num_items, num_shards)
    start = shard_index * base + min(shard_index, remainder)
    stop = start + base + (1 if shard_index < remainder else 0)
    return start, stop


def clean_intervals_for_runtime(
    seq_len: int,
    num_context_ranks: int,
    runtime: Any,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    return clean_intervals(
        seq_len,
        num_context_ranks,
        layout=runtime_clean_layout(runtime),
    )


def runtime_clean_layout(runtime: Any) -> str:
    if int(getattr(runtime, "block_parallel_size", 1) or 1) == 1:
        return "contiguous"
    policy = getattr(runtime, "cp_bp_policy", None)
    configured_layout = str(getattr(policy, "clean_kv_layout", "zigzag"))
    if configured_layout == "zigzag":
        return "dual_chunk"
    if configured_layout == "contiguous":
        return "contiguous"
    raise ValueError(
        "fused BP/CP clean_kv_layout must be 'zigzag' or 'contiguous'"
    )


def clean_intervals(
    seq_len: int,
    num_context_ranks: int,
    *,
    layout: str,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    return tuple(
        tuple(
            (shard.start, shard.stop)
            for shard in clean_shards_for_rank(
                seq_len=seq_len,
                context_parallel_size=num_context_ranks,
                rank=rank,
                layout=layout,
            )
        )
        for rank in range(num_context_ranks)
    )


@lru_cache(maxsize=None)
def windowed_clean_exchange_plan(
    *,
    seq_len: int,
    block_size: int,
    context_parallel_size: int,
    block_parallel_size: int,
    block_group_index: int,
    rank: int,
    window: int,
) -> WindowedCleanExchangePlan:
    """Return the exact clean-token exchange induced by a sliding window.

    BP target ownership and clean dual-chunk ownership are both static.  The
    plan intersects every destination rank's required clean intervals with the
    source ownership intervals, producing the variable split sizes used by one
    all-to-all in forward and its reverse exchange in backward.
    """

    if seq_len <= 0 or block_size <= 0 or window <= 0:
        raise ValueError("sequence length, block size, and window must be positive")
    if seq_len % block_size:
        raise ValueError("sequence length must divide evenly by block size")
    if context_parallel_size <= 1 or not 0 <= rank < context_parallel_size:
        raise ValueError("windowed clean exchange requires a valid multi-rank group")
    if (
        block_parallel_size < context_parallel_size
        or block_parallel_size % context_parallel_size
    ):
        raise ValueError(
            "block parallel size must be a multiple of context parallel size"
        )
    block_group_count = block_parallel_size // context_parallel_size
    if not 0 <= block_group_index < block_group_count:
        raise ValueError("block group index is outside the fused BP/CP topology")

    ownership = clean_intervals(
        seq_len,
        context_parallel_size,
        layout="dual_chunk",
    )
    schedule = build_block_schedule(
        num_blocks=seq_len // block_size,
        block_parallel_size=block_parallel_size,
        context_parallel_size=context_parallel_size,
    )
    first_block_rank = block_group_index * context_parallel_size
    required_by_rank = tuple(
        _windowed_required_clean_intervals(
            clean_intervals_for_rank=ownership[destination],
            active_blocks=tuple(
                schedule.active_blocks_by_worker[first_block_rank + destination]
            ),
            block_size=block_size,
            window=window,
        )
        for destination in range(context_parallel_size)
    )

    local_offsets: list[tuple[int, int, int]] = []
    offset = 0
    for start, stop in ownership[rank]:
        local_offsets.append((start, stop, offset))
        offset += stop - start

    send_counts: list[int] = []
    send_local_intervals_by_rank: list[tuple[tuple[int, int], ...]] = []
    for destination in range(context_parallel_size):
        destination_count = 0
        destination_intervals: list[tuple[int, int]] = []
        for owner_start, owner_stop, owner_offset in local_offsets:
            for required_start, required_stop in required_by_rank[destination]:
                start = max(owner_start, required_start)
                stop = min(owner_stop, required_stop)
                if stop <= start:
                    continue
                local_start = owner_offset + start - owner_start
                local_stop = owner_offset + stop - owner_start
                destination_intervals.append((local_start, local_stop))
                destination_count += stop - start
        send_counts.append(destination_count)
        send_local_intervals_by_rank.append(tuple(destination_intervals))

    receive_counts: list[int] = []
    receive_logical_intervals_by_rank: list[tuple[tuple[int, int], ...]] = []
    for source in range(context_parallel_size):
        source_count = 0
        source_intervals: list[tuple[int, int]] = []
        for owner_start, owner_stop in ownership[source]:
            for required_start, required_stop in required_by_rank[rank]:
                start = max(owner_start, required_start)
                stop = min(owner_stop, required_stop)
                if stop <= start:
                    continue
                source_intervals.append((start, stop))
                source_count += stop - start
        receive_counts.append(source_count)
        receive_logical_intervals_by_rank.append(tuple(source_intervals))

    return WindowedCleanExchangePlan(
        send_counts=tuple(send_counts),
        receive_counts=tuple(receive_counts),
        send_local_intervals_by_rank=tuple(send_local_intervals_by_rank),
        receive_logical_intervals_by_rank=tuple(receive_logical_intervals_by_rank),
    )


def _windowed_required_clean_intervals(
    *,
    clean_intervals_for_rank: tuple[tuple[int, int], ...],
    active_blocks: tuple[int, ...],
    block_size: int,
    window: int,
) -> tuple[tuple[int, int], ...]:
    history = window - 1
    intervals = [
        (max(0, start - history), stop)
        for start, stop in clean_intervals_for_rank
        if stop > start
    ]
    intervals.extend(
        (max(0, block * block_size - history), block * block_size)
        for block in active_blocks
        if block > 0
    )
    return _merge_intervals(intervals)


def _merge_intervals(
    intervals: list[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    merged: list[list[int]] = []
    for start, stop in sorted(intervals):
        if stop <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, stop])
        else:
            merged[-1][1] = max(merged[-1][1], stop)
    return tuple((start, stop) for start, stop in merged)


def cat_intervals(
    tensor: torch.Tensor,
    intervals: tuple[tuple[int, int], ...],
) -> torch.Tensor:
    chunks = [
        tensor[..., start:stop, :]
        for start, stop in intervals
        if start < stop
    ]
    if not chunks:
        return tensor[..., :0, :].contiguous()
    return torch.cat(chunks, dim=-2).contiguous()


def write_interval_grads(
    grad_full: torch.Tensor,
    reduced: torch.Tensor,
    intervals: tuple[tuple[int, int], ...],
) -> None:
    offset = 0
    for start, stop in intervals:
        length = stop - start
        if length <= 0:
            continue
        grad_full[..., start:stop, :] = reduced[..., offset:offset + length, :]
        offset += length
