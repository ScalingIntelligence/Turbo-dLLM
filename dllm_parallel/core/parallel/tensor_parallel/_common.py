# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Shared low-level primitives for tensor-parallel collectives.

Runtime accessors, group resolution, all-reduce helpers, and linear-gradient
helpers used across the tensor-parallel submodules. Kept in one place so there
is a single definition (no duplicated module state) and no import cycle.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.distributed as dist


def _runtime_tp_size(runtime: Any | None, default: int) -> int:
    if runtime is None:
        return default
    value = getattr(runtime, "tensor_parallel_size", None)
    if value is None:
        ranks = getattr(runtime, "tensor_parallel_group_ranks", None)
        value = len(ranks) if ranks else default
    return int(value or 1)


def _runtime_tp_rank(runtime: Any | None, default: int) -> int:
    if runtime is None:
        return default
    return int(getattr(runtime, "tensor_parallel_rank", default) or 0)


def _runtime_tp_group(runtime: Any | None) -> Any | None:
    if runtime is None:
        return None
    return getattr(runtime, "tensor_parallel_group", None)


def _ensure_group(group: Any | None, world_size: int) -> Any | None:
    if world_size <= 1:
        return group
    if group is None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "tensor parallel collectives require initialized torch.distributed"
            )
        return dist.group.WORLD
    return group


def _all_reduce_sum_(x: torch.Tensor, group: Any) -> torch.Tensor:
    work = dist.all_reduce(
        x,
        op=dist.ReduceOp.SUM,
        group=group,
        async_op=True,
    )
    block_current_stream = getattr(work, "block_current_stream", None)
    if x.is_cuda and block_current_stream is not None:
        block_current_stream()
    work.wait()
    return x


def _all_reduce_sum_async_(x: torch.Tensor, group: Any) -> Any:
    work = dist.all_reduce(
        x,
        op=dist.ReduceOp.SUM,
        group=group,
        async_op=True,
    )
    block_current_stream = getattr(work, "block_current_stream", None)
    if x.is_cuda and block_current_stream is not None:
        block_current_stream()
    return work


def _linear_weight_grad(
    grad_output: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    return grad_output.reshape(-1, grad_output.shape[-1]).t().matmul(
        x.reshape(-1, x.shape[-1])
    )


def _linear_bias_grad(grad_output: torch.Tensor) -> torch.Tensor:
    return grad_output.reshape(-1, grad_output.shape[-1]).sum(dim=0)


def _packed_linear_biases(
    biases: Sequence[torch.Tensor | None],
    *,
    output_sizes: Sequence[int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if all(bias is None for bias in biases):
        return None
    packed: list[torch.Tensor] = []
    for bias, size in zip(biases, output_sizes, strict=True):
        if bias is None:
            packed.append(torch.zeros(int(size), device=device, dtype=dtype))
        else:
            packed.append(bias)
    return torch.cat(packed, dim=0)
