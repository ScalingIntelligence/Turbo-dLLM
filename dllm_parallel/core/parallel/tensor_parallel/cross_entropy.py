# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Vocab-parallel cross entropy (loss-parallel CE over sharded vocabulary)."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

from dllm_parallel.core.parallel.tensor_parallel._common import _ensure_group
from dllm_parallel.core.parallel.tensor_parallel.random import partition_bounds


def vocab_parallel_logits_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    vocab_size: int,
    tensor_parallel_size: int,
    tensor_parallel_rank: int,
    tensor_parallel_group: Any,
    ignore_index: int = -100,
    exclude_index: int | None = None,
    reduction: str = "none",
    logit_softcap: float | None = None,
) -> torch.Tensor:
    """Cross entropy for already-computed logits sharded over vocabulary.

    This is the loss-parallel counterpart to :class:`VocabParallelLinear`.
    It matches Megatron's vocab-parallel CE semantics while supporting DLLM's
    excluded mask-token column.
    """

    start, stop = partition_bounds(
        int(vocab_size),
        int(tensor_parallel_size),
        int(tensor_parallel_rank),
    )
    expected = stop - start
    if logits.shape[-1] != expected:
        raise RuntimeError(
            "sharded logits width does not match TP vocab shard: "
            f"logits={tuple(logits.shape)} expected_vocab_range=({start}, {stop})"
        )
    return _VocabParallelLogitsCrossEntropy.apply(
        logits,
        labels,
        int(start),
        int(ignore_index),
        -1 if exclude_index is None else int(exclude_index),
        _ensure_group(tensor_parallel_group, int(tensor_parallel_size)),
        str(reduction),
        0.0 if logit_softcap is None else float(logit_softcap),
    )


def runtime_vocab_parallel_logits_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    vocab_size: int,
    runtime: Any,
    ignore_index: int = -100,
    exclude_index: int | None = None,
    reduction: str = "none",
    logit_softcap: float | None = None,
) -> torch.Tensor:
    return vocab_parallel_logits_cross_entropy(
        logits,
        labels,
        vocab_size=int(vocab_size),
        tensor_parallel_size=int(getattr(runtime, "tensor_parallel_size", 1) or 1),
        tensor_parallel_rank=int(getattr(runtime, "tensor_parallel_rank", 0) or 0),
        tensor_parallel_group=getattr(runtime, "tensor_parallel_group", None),
        ignore_index=int(ignore_index),
        exclude_index=exclude_index,
        reduction=str(reduction),
        logit_softcap=logit_softcap,
    )


class _VocabParallelLogitsCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        logits: torch.Tensor,
        labels: torch.Tensor,
        vocab_start: int,
        ignore_index: int,
        exclude_index: int,
        group: Any,
        reduction: str,
        logit_softcap: float,
    ) -> torch.Tensor:
        if reduction not in {"none", "sum", "mean"}:
            raise ValueError("reduction must be 'none', 'sum', or 'mean'")
        logits_f = logits.float()
        vocab_stop = int(vocab_start) + logits.shape[-1]
        if int(vocab_start) <= int(exclude_index) < vocab_stop:
            logits_f = logits_f.clone()
            logits_f[..., int(exclude_index) - int(vocab_start)] = -float("inf")
        if float(logit_softcap) > 0:
            logits_f = torch.tanh(logits_f / float(logit_softcap)) * float(logit_softcap)

        labels_flat = labels.reshape(-1).to(torch.long)
        logits_2d = logits_f.reshape(-1, logits_f.shape[-1])
        valid = labels_flat != int(ignore_index)
        local_max = logits_2d.max(dim=-1).values
        global_max = local_max.clone()
        dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=group)

        exp_logits = torch.exp(logits_2d - global_max[:, None])
        local_sum = exp_logits.sum(dim=-1)
        global_sum = local_sum.clone()
        dist.all_reduce(global_sum, op=dist.ReduceOp.SUM, group=group)

        target_here = (
            valid
            & (labels_flat >= int(vocab_start))
            & (labels_flat < vocab_stop)
        )
        target_logits = torch.zeros_like(global_max)
        rows = torch.nonzero(target_here, as_tuple=False).squeeze(-1)
        target_logits[rows] = logits_2d[
            rows,
            labels_flat[rows] - int(vocab_start),
        ]
        dist.all_reduce(target_logits, op=dist.ReduceOp.SUM, group=group)
        losses = torch.log(global_sum) + global_max - target_logits
        losses = losses.masked_fill(~valid, 0.0)

        valid_count = valid.sum().to(torch.float32)
        ctx.save_for_backward(logits, labels_flat, global_max, global_sum, valid, valid_count)
        ctx.vocab_start = int(vocab_start)
        ctx.ignore_index = int(ignore_index)
        ctx.exclude_index = int(exclude_index)
        ctx.logits_shape = logits.shape
        ctx.logits_dtype = logits.dtype
        ctx.reduction = str(reduction)
        ctx.logit_softcap = float(logit_softcap)
        if reduction == "sum":
            return losses.sum()
        if reduction == "mean":
            return losses.sum() / valid_count.clamp_min(1.0)
        return losses.view_as(labels).to(logits.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        logits, labels_flat, global_max, global_sum, valid, valid_count = ctx.saved_tensors
        logits_f = logits.float()
        vocab_stop = int(ctx.vocab_start) + logits_f.shape[-1]
        if int(ctx.vocab_start) <= int(ctx.exclude_index) < vocab_stop:
            logits_f = logits_f.clone()
            logits_f[..., int(ctx.exclude_index) - int(ctx.vocab_start)] = -float("inf")
        raw_logits = logits_f
        if ctx.logit_softcap > 0:
            logits_f = torch.tanh(raw_logits / ctx.logit_softcap) * ctx.logit_softcap
        logits_2d = logits_f.reshape(-1, logits_f.shape[-1])
        grad_logits = torch.exp(logits_2d - global_max[:, None]) / global_sum[:, None]
        target_here = (
            valid
            & (labels_flat >= int(ctx.vocab_start))
            & (labels_flat < vocab_stop)
        )
        rows = torch.nonzero(target_here, as_tuple=False).squeeze(-1)
        grad_logits[rows, labels_flat[rows] - int(ctx.vocab_start)] -= 1.0
        if int(ctx.vocab_start) <= int(ctx.exclude_index) < vocab_stop:
            grad_logits[:, int(ctx.exclude_index) - int(ctx.vocab_start)] = 0.0
        if ctx.logit_softcap > 0:
            raw_2d = raw_logits.reshape(-1, raw_logits.shape[-1])
            tanh_value = torch.tanh(raw_2d / ctx.logit_softcap)
            grad_logits = grad_logits * (1.0 - tanh_value * tanh_value)
        grad_logits = grad_logits * valid.to(grad_logits.dtype)[:, None]
        if ctx.reduction == "none":
            grad_scale = grad_output.reshape(-1).float()
        elif ctx.reduction == "sum":
            grad_scale = torch.ones_like(global_max) * grad_output.float()
        else:
            grad_scale = torch.ones_like(global_max) * (
                grad_output.float() / valid_count.clamp_min(1.0)
            )
        grad_logits = grad_logits * grad_scale[:, None]
        return (
            grad_logits.view(ctx.logits_shape).to(ctx.logits_dtype),
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _VocabParallelCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        labels: torch.Tensor,
        vocab_start: int,
        vocab_size: int,
        ignore_index: int,
        exclude_index: int,
        vocab_block_size: int,
        group: Any,
        world_size: int,
        reduction: str,
        logit_softcap: float,
    ) -> torch.Tensor:
        if reduction not in {"none", "sum", "mean"}:
            raise ValueError("reduction must be 'none', 'sum', or 'mean'")
        original_shape = labels.shape
        hidden_2d = hidden.reshape(-1, hidden.shape[-1])
        labels_1d = labels.reshape(-1)
        num_tokens = hidden_2d.shape[0]
        vocab_stop = vocab_start + weight.shape[0]
        if hidden_2d.is_cuda:
            from dllm_parallel.core.kernels import chunked_linear_ce_native

            local_max, local_sum, target_logits, valid_count = (
                chunked_linear_ce_native.vocab_parallel_forward_local(
                    hidden_2d,
                    weight,
                    bias,
                    labels_1d,
                    vocab_start=int(vocab_start),
                    ignore_index=int(ignore_index),
                    exclude_index=int(exclude_index),
                    logit_softcap=float(logit_softcap),
                )
            )
            global_max = local_max.clone()
            dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=group)
            reduced_stats = torch.stack(
                (
                    local_sum * torch.exp(local_max - global_max),
                    target_logits,
                ),
                dim=0,
            )
            dist.all_reduce(reduced_stats, op=dist.ReduceOp.SUM, group=group)
            global_sum, target_logits = reduced_stats.unbind(dim=0)
            global_lse = global_max + torch.log(global_sum)
            valid = labels_1d != int(ignore_index)
            losses = (global_lse - target_logits).masked_fill(~valid, 0.0)
            ctx.save_for_backward(
                hidden,
                weight,
                (
                    bias
                    if bias is not None
                    else torch.empty(0, dtype=weight.dtype, device=weight.device)
                ),
                labels,
                global_lse,
                valid_count,
            )
            ctx.native = True
            ctx.group = group
            ctx.has_bias = bias is not None
            ctx.vocab_start = int(vocab_start)
            ctx.ignore_index = int(ignore_index)
            ctx.exclude_index = int(exclude_index)
            ctx.reduction = str(reduction)
            ctx.logit_softcap = float(logit_softcap)
            if reduction == "sum":
                return losses.sum()
            if reduction == "mean":
                return losses.sum() / valid_count.clamp_min(1.0)
            return losses.view(original_shape).to(hidden.dtype)

        ctx.native = False
        use_cuda_low_precision_gemm = (
            hidden_2d.is_cuda
            and hidden_2d.dtype in {torch.float16, torch.bfloat16}
            and weight.dtype == hidden_2d.dtype
        )
        compute_dtype = hidden_2d.dtype if use_cuda_low_precision_gemm else torch.float32
        row_block_size = max(1, int(num_tokens))
        vocab_block_size = int(weight.shape[0]) if int(vocab_block_size) <= 0 else max(1, int(vocab_block_size))
        weight_for_logits = weight.to(compute_dtype)
        bias_for_logits = bias.to(compute_dtype) if bias is not None else None

        losses = (
            torch.zeros(num_tokens, dtype=torch.float32, device=hidden.device)
            if reduction == "none"
            else None
        )
        loss_sum = torch.zeros((), dtype=torch.float32, device=hidden.device)
        global_max_all = torch.empty(num_tokens, dtype=torch.float32, device=hidden.device)
        global_sum_all = torch.empty(num_tokens, dtype=torch.float32, device=hidden.device)
        valid = labels_1d != ignore_index
        valid_count = valid.sum().to(torch.float32)

        for token_start in range(0, num_tokens, row_block_size):
            token_stop = min(token_start + row_block_size, num_tokens)
            h = hidden_2d[token_start:token_stop].to(compute_dtype)
            y = labels_1d[token_start:token_stop]
            logits = (h @ weight_for_logits.t()).float()
            if bias_for_logits is not None:
                logits = logits + bias_for_logits.float()
            if vocab_start <= exclude_index < vocab_stop:
                logits[:, exclude_index - vocab_start] = -float("inf")
            if float(logit_softcap) > 0:
                logits = torch.tanh(logits / float(logit_softcap)) * float(logit_softcap)
            local_max = logits.max(dim=-1).values

            dist.all_reduce(local_max, op=dist.ReduceOp.MAX, group=group)
            target_logits = torch.zeros_like(local_max)
            local_sum = torch.exp(logits - local_max[:, None]).sum(dim=-1)
            target_is_here = (
                (y != ignore_index)
                & (y >= vocab_start)
                & (y < vocab_stop)
            )
            if target_is_here.any():
                rows = torch.nonzero(target_is_here, as_tuple=False).squeeze(-1)
                target_logits[rows] = logits[rows, y[rows] - vocab_start]

            dist.all_reduce(local_sum, op=dist.ReduceOp.SUM, group=group)
            dist.all_reduce(target_logits, op=dist.ReduceOp.SUM, group=group)
            block_valid = valid[token_start:token_stop]
            block_losses = torch.log(local_sum) + local_max - target_logits
            block_losses = block_losses.masked_fill(
                ~block_valid,
                0,
            )
            if losses is not None:
                losses[token_start:token_stop] = block_losses
            else:
                loss_sum = loss_sum + block_losses.sum()
            global_max_all[token_start:token_stop] = local_max
            global_sum_all[token_start:token_stop] = local_sum

        ctx.save_for_backward(
            hidden,
            weight,
            (
                bias
                if bias is not None
                else torch.empty(0, dtype=weight.dtype, device=weight.device)
            ),
            labels,
            valid,
            global_max_all,
            global_sum_all,
            valid_count,
        )
        ctx.group = group
        ctx.world_size = int(world_size)
        ctx.has_bias = bias is not None
        ctx.vocab_start = int(vocab_start)
        ctx.ignore_index = int(ignore_index)
        ctx.exclude_index = int(exclude_index)
        ctx.row_block_size = int(row_block_size)
        ctx.vocab_block_size = int(vocab_block_size)
        ctx.compute_dtype = compute_dtype
        ctx.reduction = str(reduction)
        ctx.logit_softcap = float(logit_softcap)
        if reduction == "sum":
            return loss_sum
        if reduction == "mean":
            return loss_sum / valid_count.clamp_min(1.0)
        assert losses is not None
        return losses.view(original_shape).to(hidden.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.native:
            hidden, weight, bias, labels, global_lse, valid_count = ctx.saved_tensors
            from dllm_parallel.core.kernels import chunked_linear_ce_native

            grad_hidden, grad_weight, grad_bias = (
                chunked_linear_ce_native.vocab_parallel_backward_local(
                    hidden.reshape(-1, hidden.shape[-1]),
                    weight,
                    bias if ctx.has_bias else None,
                    labels.reshape(-1),
                    global_lse,
                    valid_count,
                    grad_output,
                    reduction=ctx.reduction,
                    vocab_start=ctx.vocab_start,
                    ignore_index=ctx.ignore_index,
                    exclude_index=ctx.exclude_index,
                    logit_softcap=ctx.logit_softcap,
                )
            )
            dist.all_reduce(grad_hidden, op=dist.ReduceOp.SUM, group=ctx.group)
            return (
                grad_hidden.view_as(hidden),
                grad_weight,
                grad_bias if ctx.has_bias else None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

        hidden, weight, bias, labels, valid, global_max_all, global_sum_all, valid_count = (
            ctx.saved_tensors
        )
        hidden_2d = hidden.reshape(-1, hidden.shape[-1])
        labels_1d = labels.reshape(-1)
        if ctx.reduction == "none":
            grad_1d = grad_output.reshape(-1).float()
        elif ctx.reduction == "sum":
            grad_1d = torch.ones(
                (hidden_2d.shape[0],),
                device=hidden.device,
                dtype=torch.float32,
            ) * grad_output.float()
        else:
            grad_1d = torch.ones(
                (hidden_2d.shape[0],),
                device=hidden.device,
                dtype=torch.float32,
            ) * (grad_output.float() / valid_count.clamp_min(1.0))
        grad_hidden = torch.zeros_like(hidden_2d, dtype=torch.float32)
        grad_weight = torch.zeros_like(weight, dtype=torch.float32)
        grad_bias = (
            torch.zeros(weight.shape[0], dtype=torch.float32, device=weight.device)
            if ctx.has_bias
            else None
        )
        weight_for_logits = weight.to(ctx.compute_dtype)
        bias_for_logits = bias.to(ctx.compute_dtype) if ctx.has_bias else None

        for token_start in range(0, hidden_2d.shape[0], ctx.row_block_size):
            token_stop = min(token_start + ctx.row_block_size, hidden_2d.shape[0])
            h = hidden_2d[token_start:token_stop].to(ctx.compute_dtype)
            y = labels_1d[token_start:token_stop]
            block_valid = valid[token_start:token_stop]
            block_grad = grad_1d[token_start:token_stop]
            block_max = global_max_all[token_start:token_stop]
            block_sum = global_sum_all[token_start:token_stop]

            logits = (h @ weight_for_logits.t()).float()
            if bias_for_logits is not None:
                logits = logits + bias_for_logits.float()
            vocab_stop = ctx.vocab_start + weight.shape[0]
            if ctx.vocab_start <= ctx.exclude_index < vocab_stop:
                logits[:, ctx.exclude_index - ctx.vocab_start] = -float("inf")
            raw_logits = logits
            if ctx.logit_softcap > 0:
                logits = torch.tanh(raw_logits / ctx.logit_softcap) * ctx.logit_softcap
            grad_logits = torch.exp(logits - block_max[:, None]) / block_sum[:, None]
            target_is_here = (
                (y != ctx.ignore_index)
                & (y >= ctx.vocab_start)
                & (y < vocab_stop)
            )
            if target_is_here.any():
                rows = torch.nonzero(target_is_here, as_tuple=False).squeeze(-1)
                grad_logits[rows, y[rows] - ctx.vocab_start] -= 1
            grad_logits = grad_logits * block_valid.to(grad_logits.dtype)[:, None]
            grad_logits = grad_logits * block_grad[:, None]
            if ctx.logit_softcap > 0:
                tanh_value = torch.tanh(raw_logits / ctx.logit_softcap)
                grad_logits = grad_logits * (1.0 - tanh_value * tanh_value)
            block_grad_hidden = grad_logits.to(ctx.compute_dtype) @ weight_for_logits
            grad_weight += grad_logits.t() @ hidden_2d[token_start:token_stop].float()
            if grad_bias is not None:
                grad_bias += grad_logits.sum(dim=0)

            dist.all_reduce(block_grad_hidden, op=dist.ReduceOp.SUM, group=ctx.group)
            grad_hidden[token_start:token_stop] = block_grad_hidden

        grad_hidden = grad_hidden.to(hidden.dtype).view_as(hidden)
        grad_weight = grad_weight.to(weight.dtype)
        if grad_bias is not None:
            grad_bias = grad_bias.to(weight.dtype)
        return (
            grad_hidden,
            grad_weight,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
