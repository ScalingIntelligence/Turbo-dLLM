# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Tensor-parallel copy/reduce/gather region mappings and overlapped-dgrad linears."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

from dllm_parallel.core.parallel.tensor_parallel._common import (
    _all_reduce_sum_,
    _all_reduce_sum_async_,
    _ensure_group,
    _linear_bias_grad,
    _linear_weight_grad,
    _packed_linear_biases,
    _runtime_tp_group,
    _runtime_tp_size,
)


def _copy_with_tp_backward(x: torch.Tensor, group: Any | None, world_size: int) -> torch.Tensor:
    if world_size <= 1:
        return x
    return _CopyToTensorParallelRegion.apply(x, _ensure_group(group, world_size), world_size)


def _reduce_from_tp(x: torch.Tensor, group: Any | None, world_size: int) -> torch.Tensor:
    if world_size <= 1:
        return x
    return _ReduceFromTensorParallelRegion.apply(x, _ensure_group(group, world_size), world_size)


def copy_to_tensor_parallel_region(
    x: torch.Tensor,
    runtime: Any | None = None,
    *,
    group: Any | None = None,
    world_size: int | None = None,
) -> torch.Tensor:
    """Identity in forward, all-reduce gradients across the TP region.

    This is the public Megatron-style TP primitive used before column-parallel
    projections. A runtime may be supplied directly; explicit group/world_size
    are kept for low-level modules that cache their TP process group.
    """

    if runtime is not None:
        group = _runtime_tp_group(runtime)
        world_size = _runtime_tp_size(runtime, int(world_size or 1))
    return _copy_with_tp_backward(x, group, int(world_size or 1))


def reduce_from_tensor_parallel_region(
    x: torch.Tensor,
    runtime: Any | None = None,
    *,
    group: Any | None = None,
    world_size: int | None = None,
) -> torch.Tensor:
    """All-reduce TP partial outputs in forward, identity in backward."""

    if runtime is not None:
        group = _runtime_tp_group(runtime)
        world_size = _runtime_tp_size(runtime, int(world_size or 1))
    return _reduce_from_tp(x, group, int(world_size or 1))


def column_parallel_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    runtime: Any | None = None,
    *,
    group: Any | None = None,
    world_size: int | None = None,
    allreduce_dgrad: bool = True,
) -> torch.Tensor:
    """Column-parallel linear with Megatron-style overlapped dgrad all-reduce."""

    if runtime is not None:
        group = _runtime_tp_group(runtime)
        world_size = _runtime_tp_size(runtime, int(world_size or 1))
    world_size = int(world_size or 1)
    if world_size <= 1 or not bool(allreduce_dgrad):
        return F.linear(x, weight, bias)
    return _ColumnParallelLinearWithAsyncDgrad.apply(
        x,
        weight,
        bias,
        _ensure_group(group, world_size),
        world_size,
    )


def fused_qkv_column_parallel_linear(
    x: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    q_bias: torch.Tensor | None,
    k_bias: torch.Tensor | None,
    v_bias: torch.Tensor | None,
    runtime: Any | None = None,
    *,
    group: Any | None = None,
    world_size: int | None = None,
    allreduce_dgrad: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused q/k/v column-parallel projections with one overlapped dgrad reduce.

    Hugging Face tensor parallelism leaves Nemotron's q, k and v projections as
    separate local-sharded ``Linear`` modules. Running a wrapper all-reduce after
    the three backwards serializes communication behind all wgrad GEMMs. This
    function preserves the same math while launching one combined input-gradient
    all-reduce before the q/k/v weight-gradient GEMMs, matching Megatron's
    overlap pattern without depending on Megatron runtime globals.
    """

    if runtime is not None:
        group = _runtime_tp_group(runtime)
        world_size = _runtime_tp_size(runtime, int(world_size or 1))
    world_size = int(world_size or 1)
    if world_size <= 1 or not bool(allreduce_dgrad):
        return (
            F.linear(x, q_weight, q_bias),
            F.linear(x, k_weight, k_bias),
            F.linear(x, v_weight, v_bias),
        )
    return _FusedQKVColumnParallelLinearWithAsyncDgrad.apply(
        x,
        q_weight,
        k_weight,
        v_weight,
        q_bias,
        k_bias,
        v_bias,
        _ensure_group(group, world_size),
        world_size,
    )


def fused_gate_up_column_parallel_linear(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    gate_bias: torch.Tensor | None,
    up_bias: torch.Tensor | None,
    runtime: Any | None = None,
    *,
    group: Any | None = None,
    world_size: int | None = None,
    allreduce_dgrad: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused gated-MLP column projections with overlapped input-gradient reduce.

    This is the generic TP primitive matching the TE packed ``gate_up`` path:
    gate and up projections are launched as one packed local GEMM, and backward
    performs one input-gradient all-reduce before local weight-gradient GEMMs.
    """

    if runtime is not None:
        group = _runtime_tp_group(runtime)
        world_size = _runtime_tp_size(runtime, int(world_size or 1))
    world_size = int(world_size or 1)
    if world_size <= 1 or not bool(allreduce_dgrad):
        return (
            F.linear(x, gate_weight, gate_bias),
            F.linear(x, up_weight, up_bias),
        )
    return _FusedGateUpColumnParallelLinearWithAsyncDgrad.apply(
        x,
        gate_weight,
        up_weight,
        gate_bias,
        up_bias,
        _ensure_group(group, world_size),
        world_size,
    )


def _gather_from_tp(
    x: torch.Tensor,
    *,
    group: Any | None,
    world_size: int,
    rank: int,
    sizes: Sequence[int],
) -> torch.Tensor:
    if world_size <= 1:
        return x
    return _GatherFromTensorParallelRegion.apply(
        x,
        _ensure_group(group, world_size),
        int(world_size),
        int(rank),
        tuple(int(size) for size in sizes),
    )


class _CopyToTensorParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, group: Any, world_size: int) -> torch.Tensor:
        ctx.group = group
        ctx.world_size = int(world_size)
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.world_size > 1:
            _all_reduce_sum_(grad_output, ctx.group)
        return grad_output, None, None


class _ReduceFromTensorParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, group: Any, world_size: int) -> torch.Tensor:
        ctx.world_size = int(world_size)
        y = x.clone()
        return _all_reduce_sum_(y, group)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None, None


class _ColumnParallelLinearWithAsyncDgrad(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        group: Any,
        world_size: int,
    ) -> torch.Tensor:
        ctx.use_bias = bias is not None
        ctx.group = group
        ctx.world_size = int(world_size)
        ctx.save_for_backward(x, weight)
        return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, weight = ctx.saved_tensors
        needs_x, needs_weight, needs_bias = ctx.needs_input_grad[:3]
        grad_x = grad_weight = grad_bias = None
        work = None
        if needs_x:
            grad_x = grad_output.matmul(weight)
            if ctx.world_size > 1:
                work = _all_reduce_sum_async_(grad_x, ctx.group)
        if needs_weight:
            grad_weight = _linear_weight_grad(grad_output, x)
        if needs_bias and ctx.use_bias:
            grad_bias = _linear_bias_grad(grad_output)
        if work is not None:
            work.wait()
        return grad_x, grad_weight, grad_bias, None, None


class _FusedQKVColumnParallelLinearWithAsyncDgrad(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        v_weight: torch.Tensor,
        q_bias: torch.Tensor | None,
        k_bias: torch.Tensor | None,
        v_bias: torch.Tensor | None,
        group: Any,
        world_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx.use_q_bias = q_bias is not None
        ctx.use_k_bias = k_bias is not None
        ctx.use_v_bias = v_bias is not None
        ctx.group = group
        ctx.world_size = int(world_size)
        ctx.save_for_backward(x, q_weight, k_weight, v_weight)
        output_sizes = (q_weight.shape[0], k_weight.shape[0], v_weight.shape[0])
        ctx.output_sizes = output_sizes
        packed_weight = torch.cat((q_weight, k_weight, v_weight), dim=0)
        packed_bias = _packed_linear_biases(
            (q_bias, k_bias, v_bias),
            output_sizes=output_sizes,
            device=packed_weight.device,
            dtype=packed_weight.dtype,
        )
        packed_output = F.linear(x, packed_weight, packed_bias)
        return packed_output.split(output_sizes, dim=-1)

    @staticmethod
    def backward(
        ctx,
        grad_q: torch.Tensor,
        grad_k: torch.Tensor,
        grad_v: torch.Tensor,
    ):
        x, q_weight, k_weight, v_weight = ctx.saved_tensors
        needs = ctx.needs_input_grad
        grad_x = None
        work = None
        grad_q = grad_q.contiguous()
        grad_k = grad_k.contiguous()
        grad_v = grad_v.contiguous()
        if needs[0]:
            packed_grad = torch.cat((grad_q, grad_k, grad_v), dim=-1)
            packed_weight = torch.cat((q_weight, k_weight, v_weight), dim=0)
            grad_x = packed_grad.matmul(packed_weight)
            if ctx.world_size > 1:
                work = _all_reduce_sum_async_(grad_x, ctx.group)

        grad_q_weight = _linear_weight_grad(grad_q, x) if needs[1] else None
        grad_k_weight = _linear_weight_grad(grad_k, x) if needs[2] else None
        grad_v_weight = _linear_weight_grad(grad_v, x) if needs[3] else None
        grad_q_bias = (
            _linear_bias_grad(grad_q) if needs[4] and ctx.use_q_bias else None
        )
        grad_k_bias = (
            _linear_bias_grad(grad_k) if needs[5] and ctx.use_k_bias else None
        )
        grad_v_bias = (
            _linear_bias_grad(grad_v) if needs[6] and ctx.use_v_bias else None
        )

        if work is not None:
            work.wait()
        return (
            grad_x,
            grad_q_weight,
            grad_k_weight,
            grad_v_weight,
            grad_q_bias,
            grad_k_bias,
            grad_v_bias,
            None,
            None,
        )


class _FusedGateUpColumnParallelLinearWithAsyncDgrad(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        gate_bias: torch.Tensor | None,
        up_bias: torch.Tensor | None,
        group: Any,
        world_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx.use_gate_bias = gate_bias is not None
        ctx.use_up_bias = up_bias is not None
        ctx.group = group
        ctx.world_size = int(world_size)
        ctx.save_for_backward(x, gate_weight, up_weight)
        output_sizes = (gate_weight.shape[0], up_weight.shape[0])
        packed_weight = torch.cat((gate_weight, up_weight), dim=0)
        packed_bias = _packed_linear_biases(
            (gate_bias, up_bias),
            output_sizes=output_sizes,
            device=packed_weight.device,
            dtype=packed_weight.dtype,
        )
        packed_output = F.linear(x, packed_weight, packed_bias)
        return packed_output.split(output_sizes, dim=-1)

    @staticmethod
    def backward(
        ctx,
        grad_gate: torch.Tensor,
        grad_up: torch.Tensor,
    ):
        x, gate_weight, up_weight = ctx.saved_tensors
        needs = ctx.needs_input_grad
        grad_x = None
        work = None
        grad_gate = grad_gate.contiguous()
        grad_up = grad_up.contiguous()
        if needs[0]:
            packed_grad = torch.cat((grad_gate, grad_up), dim=-1)
            packed_weight = torch.cat((gate_weight, up_weight), dim=0)
            grad_x = packed_grad.matmul(packed_weight)
            if ctx.world_size > 1:
                work = _all_reduce_sum_async_(grad_x, ctx.group)

        grad_gate_weight = _linear_weight_grad(grad_gate, x) if needs[1] else None
        grad_up_weight = _linear_weight_grad(grad_up, x) if needs[2] else None
        grad_gate_bias = (
            _linear_bias_grad(grad_gate) if needs[3] and ctx.use_gate_bias else None
        )
        grad_up_bias = (
            _linear_bias_grad(grad_up) if needs[4] and ctx.use_up_bias else None
        )

        if work is not None:
            work.wait()
        return (
            grad_x,
            grad_gate_weight,
            grad_up_weight,
            grad_gate_bias,
            grad_up_bias,
            None,
            None,
        )


class _GatherFromTensorParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        group: Any,
        world_size: int,
        rank: int,
        sizes: tuple[int, ...],
    ) -> torch.Tensor:
        ctx.rank = int(rank)
        ctx.sizes = sizes
        max_size = max(sizes)
        if x.shape[-1] > max_size:
            raise ValueError("local tensor is larger than the largest TP partition")
        if x.shape[-1] < max_size:
            pad = max_size - x.shape[-1]
            x_for_gather = F.pad(x, (0, pad))
        else:
            x_for_gather = x.contiguous()
        gathered = [torch.empty_like(x_for_gather) for _ in range(world_size)]
        dist.all_gather(gathered, x_for_gather, group=group)
        return torch.cat(
            [part[..., :size] for part, size in zip(gathered, sizes)],
            dim=-1,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        start = sum(ctx.sizes[: ctx.rank])
        stop = start + ctx.sizes[ctx.rank]
        return grad_output[..., start:stop].contiguous(), None, None, None, None
