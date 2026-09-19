# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Partitioning, TP rank/size config helpers, per-rank init, and dropout RNG sync."""

from __future__ import annotations

import math
import os
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from dllm_parallel.core.parallel.tensor_parallel._common import (
    _ensure_group,
    _runtime_tp_group,
    _runtime_tp_size,
)


def partition_bounds(total_size: int, num_partitions: int, rank: int) -> tuple[int, int]:
    """Return the contiguous [start, stop) range for an uneven partition."""

    if total_size < 0:
        raise ValueError("total_size must be non-negative")
    if num_partitions <= 0:
        raise ValueError("num_partitions must be positive")
    if not 0 <= rank < num_partitions:
        raise ValueError("rank out of range")
    base = total_size // num_partitions
    rem = total_size % num_partitions
    start = rank * base + min(rank, rem)
    stop = start + base + (1 if rank < rem else 0)
    return start, stop


def partition_sizes(total_size: int, num_partitions: int) -> list[int]:
    return [
        partition_bounds(total_size, num_partitions, rank)[1]
        - partition_bounds(total_size, num_partitions, rank)[0]
        for rank in range(num_partitions)
    ]


def infer_tensor_parallel_rank(tensor_parallel_size: int) -> int:
    """Infer the local TP rank from distributed state or launcher env vars."""

    tensor_parallel_size = int(tensor_parallel_size)
    if tensor_parallel_size <= 1:
        return 0
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    return rank % tensor_parallel_size


def tensor_parallel_size_from_config(config: Any) -> int:
    parallel = getattr(config, "parallel", None)
    if parallel is None:
        return 1
    if hasattr(parallel, "get"):
        return int(parallel.get("tensor_parallel_size", 1) or 1)
    return int(getattr(parallel, "tensor_parallel_size", 1) or 1)


def tensor_parallel_rank_from_config(config: Any) -> int:
    parallel = getattr(config, "parallel", None)
    if parallel is not None:
        if hasattr(parallel, "get"):
            value = parallel.get("tensor_parallel_rank", None)
        else:
            value = getattr(parallel, "tensor_parallel_rank", None)
        if value is not None:
            return int(value)
    return infer_tensor_parallel_rank(tensor_parallel_size_from_config(config))


def set_tensor_parallel_runtime(module: nn.Module, runtime: Any | None) -> None:
    for child in module.modules():
        if child is module:
            continue
        setter = getattr(child, "set_tensor_parallel_runtime", None)
        if setter is not None:
            setter(runtime)


def sync_tensor_parallel_rng_state(
    runtime: Any | None,
    *,
    device: torch.device,
) -> None:
    """Synchronize the RNG used by dropout across TP ranks."""

    world_size = _runtime_tp_size(runtime, 1)
    if world_size <= 1:
        return
    group = _ensure_group(_runtime_tp_group(runtime), world_size)
    ranks = getattr(runtime, "tensor_parallel_group_ranks", None)
    src = int(ranks[0]) if ranks else 0

    def _sync_cuda_rng() -> bool:
        if not torch.cuda.is_available():
            return False
        cuda_device = (
            device
            if device.type == "cuda"
            else torch.device("cuda", torch.cuda.current_device())
        )
        state = torch.cuda.get_rng_state(cuda_device).to(cuda_device)
        dist.broadcast(state, src=src, group=group)
        torch.cuda.set_rng_state(state.cpu(), cuda_device)
        return True

    try:
        backend = str(dist.get_backend(group)).lower()
    except Exception:
        backend = ""
    try:
        default_backend = str(dist.get_backend()).lower()
    except Exception:
        default_backend = ""

    if device.type == "cuda" or "nccl" in backend or "nccl" in default_backend:
        _sync_cuda_rng()
        return

    state = torch.random.get_rng_state()
    try:
        dist.broadcast(state, src=src, group=group)
    except RuntimeError:
        if _sync_cuda_rng():
            return
        raise
    torch.random.set_rng_state(state.cpu())


def _fork_rng_for_tp_rank(tensor_parallel_rank: int):
    seed = torch.initial_seed() + 271_828 * int(tensor_parallel_rank)
    return torch.random.fork_rng(devices=[], enabled=True), seed


def _kaiming_uniform_partition_(tensor: torch.Tensor, tensor_parallel_rank: int) -> None:
    ctx, seed = _fork_rng_for_tp_rank(tensor_parallel_rank)
    with ctx:
        torch.manual_seed(seed)
        nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))


def _uniform_bias_(bias: torch.Tensor, fan_in: int, tensor_parallel_rank: int) -> None:
    if fan_in <= 0:
        nn.init.zeros_(bias)
        return
    bound = 1 / math.sqrt(fan_in)
    ctx, seed = _fork_rng_for_tp_rank(tensor_parallel_rank)
    with ctx:
        torch.manual_seed(seed + 17)
        nn.init.uniform_(bias, -bound, bound)
