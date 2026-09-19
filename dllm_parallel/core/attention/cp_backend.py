# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Internal scheduling helpers for the optimized BDLM CP/BP backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class CleanShard:
    """One contiguous clean-token interval owned by a context rank."""

    owner_rank: int
    start: int
    stop: int
    chunk_index: int

    @property
    def length(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class PrefixVisit:
    noisy_block: int
    owner_rank: int
    start: int
    stop: int
    skipped_self_start: int | None = None
    skipped_self_stop: int | None = None

    @property
    def visited_tokens(self) -> int:
        return self.stop - self.start

    @property
    def skipped_tokens(self) -> int:
        if self.skipped_self_start is None or self.skipped_self_stop is None:
            return 0
        return self.skipped_self_stop - self.skipped_self_start


@dataclass(frozen=True)
class CPBPCounters:
    active_blocks: int
    prefix_shards_visited: int
    prefix_shards_skipped: int
    prefix_tokens_visited: int
    prefix_tokens_skipped: int
    exposed_comm_fraction: float = 0.0
    clean_attention_ms: float = 0.0
    noisy_attention_ms: float = 0.0
    peak_memory_bytes: int = 0

    def to_metrics(self, prefix: str = "perf/cp_bp") -> dict[str, float]:
        return {
            f"{prefix}/active_blocks": float(self.active_blocks),
            f"{prefix}/prefix_shards_visited": float(self.prefix_shards_visited),
            f"{prefix}/prefix_shards_skipped": float(self.prefix_shards_skipped),
            f"{prefix}/prefix_tokens_visited": float(self.prefix_tokens_visited),
            f"{prefix}/prefix_tokens_skipped": float(self.prefix_tokens_skipped),
            f"{prefix}/exposed_comm_fraction": float(self.exposed_comm_fraction),
            f"{prefix}/clean_attention_ms": float(self.clean_attention_ms),
            f"{prefix}/noisy_attention_ms": float(self.noisy_attention_ms),
            f"{prefix}/peak_memory_bytes": float(self.peak_memory_bytes),
        }


def balanced_bounds(num_items: int, num_shards: int, shard_index: int) -> tuple[int, int]:
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


def clean_shards_for_rank(
    *,
    seq_len: int,
    context_parallel_size: int,
    rank: int,
    layout: str = "dual_chunk",
) -> tuple[CleanShard, ...]:
    """Return the clean-token ownership for ``rank``.

    ``layout="dual_chunk"`` matches TransformerEngine's head-tail ownership:
    rank ``r`` owns chunk ``r`` from the sequence head and chunk
    ``2 * context_parallel_size - r - 1`` from the tail. ``layout="contiguous"``
    gives each rank one balanced contiguous shard, which is faster when every
    rank traverses every owner and load balancing is already uniform.
    """

    if context_parallel_size <= 0:
        raise ValueError("context_parallel_size must be positive")
    if not 0 <= rank < context_parallel_size:
        raise ValueError("rank out of range")
    if layout not in {"contiguous", "dual_chunk"}:
        raise ValueError("layout must be 'contiguous' or 'dual_chunk'")
    if layout == "contiguous":
        start, stop = balanced_bounds(seq_len, context_parallel_size, rank)
        if start >= stop:
            return ()
        return (CleanShard(rank, start, stop, rank),)
    total_chunks = 2 * context_parallel_size
    chunk_ids = (rank, total_chunks - rank - 1)
    shards: list[CleanShard] = []
    for chunk_id in chunk_ids:
        start, stop = balanced_bounds(seq_len, total_chunks, chunk_id)
        if start < stop:
            shards.append(CleanShard(rank, start, stop, chunk_id))
    return tuple(shards)


def all_clean_shards(
    *,
    seq_len: int,
    context_parallel_size: int,
) -> tuple[CleanShard, ...]:
    shards = [
        shard
        for rank in range(context_parallel_size)
        for shard in clean_shards_for_rank(
            seq_len=seq_len,
            context_parallel_size=context_parallel_size,
            rank=rank,
        )
    ]
    return tuple(sorted(shards, key=lambda item: (item.start, item.stop)))



def owner_lengths(intervals: Iterable[Iterable[tuple[int, int]]]) -> list[int]:
    return [
        sum(stop - start for start, stop in owner)
        for owner in intervals
    ]


def ring_owner_traversal(local_rank: int, context_parallel_size: int) -> tuple[int, ...]:
    if context_parallel_size <= 0:
        raise ValueError("context_parallel_size must be positive")
    if not 0 <= local_rank < context_parallel_size:
        raise ValueError("local_rank out of range")
    return tuple(
        (local_rank + step) % context_parallel_size
        for step in range(context_parallel_size)
    )


def prefix_visits_for_noisy_blocks(
    *,
    active_blocks: Iterable[int],
    block_size: int,
    seq_len: int,
    context_parallel_size: int,
) -> tuple[PrefixVisit, ...]:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if seq_len % block_size != 0:
        raise ValueError("seq_len must divide evenly by block_size")
    visits: list[PrefixVisit] = []
    for noisy_block in sorted(int(block) for block in active_blocks):
        if noisy_block < 0 or (noisy_block + 1) * block_size > seq_len:
            raise ValueError("active block out of range")
        prefix_stop = noisy_block * block_size
        self_start = prefix_stop
        self_stop = self_start + block_size
        for shard in all_clean_shards(
            seq_len=seq_len,
            context_parallel_size=context_parallel_size,
        ):
            start = max(shard.start, 0)
            stop = min(shard.stop, prefix_stop)
            skip_start = max(shard.start, self_start)
            skip_stop = min(shard.stop, self_stop)
            skipped = skip_start < skip_stop
            if start < stop or skipped:
                visits.append(
                    PrefixVisit(
                        noisy_block=noisy_block,
                        owner_rank=shard.owner_rank,
                        start=start,
                        stop=stop,
                        skipped_self_start=skip_start if skipped else None,
                        skipped_self_stop=skip_stop if skipped else None,
                    )
                )
    return tuple(visits)


def cp_bp_counters_from_visits(
    *,
    active_blocks: Iterable[int],
    visits: Iterable[PrefixVisit],
    exposed_comm_fraction: float = 0.0,
    clean_attention_ms: float = 0.0,
    noisy_attention_ms: float = 0.0,
    peak_memory_bytes: int = 0,
) -> CPBPCounters:
    active = tuple(int(block) for block in active_blocks)
    visit_list = tuple(visits)
    return CPBPCounters(
        active_blocks=len(active),
        prefix_shards_visited=sum(1 for visit in visit_list if visit.visited_tokens > 0),
        prefix_shards_skipped=sum(1 for visit in visit_list if visit.skipped_tokens > 0),
        prefix_tokens_visited=sum(visit.visited_tokens for visit in visit_list),
        prefix_tokens_skipped=sum(visit.skipped_tokens for visit in visit_list),
        exposed_comm_fraction=float(exposed_comm_fraction),
        clean_attention_ms=float(clean_attention_ms),
        noisy_attention_ms=float(noisy_attention_ms),
        peak_memory_bytes=int(peak_memory_bytes),
    )


def merge_attention_stats(
    stats: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    iterator = iter(stats)
    try:
        numerator, m, l = next(iterator)
    except StopIteration as exc:
        raise ValueError("at least one shard stat is required") from exc
    merged_num = numerator.float().clone()
    merged_m = m.float().clone()
    merged_l = l.float().clone()
    for new_num, new_m, new_l in iterator:
        merged_num, merged_m, merged_l = _merge_two_stats(
            merged_num,
            merged_m,
            merged_l,
            new_num.float(),
            new_m.float(),
            new_l.float(),
        )
    return merged_num, merged_m, merged_l


def _merge_two_stats(
    old_num: torch.Tensor,
    old_m: torch.Tensor,
    old_l: torch.Tensor,
    new_num: torch.Tensor,
    new_m: torch.Tensor,
    new_l: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    next_m = torch.maximum(old_m, new_m)
    next_m_safe = torch.where(torch.isfinite(next_m), next_m, torch.zeros_like(next_m))
    old_weight = torch.where(
        torch.isfinite(old_m),
        torch.exp(old_m - next_m_safe),
        torch.zeros_like(old_m),
    )
    new_weight = torch.where(
        torch.isfinite(new_m),
        torch.exp(new_m - next_m_safe),
        torch.zeros_like(new_m),
    )
    return (
        old_weight.unsqueeze(-1) * old_num + new_weight.unsqueeze(-1) * new_num,
        next_m,
        old_weight * old_l + new_weight * new_l,
    )
