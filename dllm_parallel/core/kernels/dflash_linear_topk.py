# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Memory-bounded frozen LM-head objective with exact top-k outputs."""

from __future__ import annotations

import torch

from dllm_parallel.core.kernels import chunked_linear_ce_native


class _NativeFrozenLinearCETopK(torch.autograd.Function):
    """Native row-streamed objective; the vocabulary never crosses Python."""

    # A full-vocabulary GEMM needs a substantial M dimension to saturate an
    # H100.  The generic native scheduler also limits blocks by CUDA thread
    # residency, which is appropriate for elementwise kernels but leaves these
    # GEMMs at roughly 1K rows.  Four thousand rows keeps the largest fp32
    # objective workspace well below H100 memory while materially improving
    # tensor-core occupancy.  The native implementation still clips this to
    # the number of active rows.
    _ROW_BLOCK_SIZE = 4096

    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        top_k: int,
        vocab_block_size: int,
        output_multiplier: float,
        logit_softcap: float,
    ):
        del vocab_block_size
        hidden = hidden.contiguous()
        weight = weight.contiguous()
        labels = labels.to(device=hidden.device, dtype=torch.long).contiguous()
        loss, probability, top_values, top_ids, lse = (
            chunked_linear_ce_native.dflash_ce_topk_forward(
                hidden,
                weight,
                labels,
                top_k=int(top_k),
                row_block_size=_NativeFrozenLinearCETopK._ROW_BLOCK_SIZE,
                output_multiplier=float(output_multiplier),
                logit_softcap=float(logit_softcap),
            )
        )
        ctx.save_for_backward(hidden, weight, labels, lse, probability, top_ids)
        ctx.output_multiplier = float(output_multiplier)
        ctx.logit_softcap = float(logit_softcap)
        ctx.mark_non_differentiable(top_ids)
        return loss, probability, top_values, top_ids

    @staticmethod
    def backward(ctx, grad_loss, grad_probability, grad_top_values, _grad_top_ids):
        hidden, weight, labels, lse, probabilities, top_ids = ctx.saved_tensors
        grad_loss = (
            torch.zeros_like(probabilities) if grad_loss is None else grad_loss
        )
        grad_probability = (
            torch.zeros_like(probabilities)
            if grad_probability is None
            else grad_probability
        )
        grad_top_values = (
            torch.zeros(
                top_ids.shape,
                device=hidden.device,
                dtype=torch.float32,
            )
            if grad_top_values is None
            else grad_top_values
        )
        grad_hidden = chunked_linear_ce_native.dflash_ce_topk_backward(
            hidden,
            weight,
            labels,
            lse,
            probabilities,
            top_ids,
            grad_loss,
            grad_probability,
            grad_top_values,
            row_block_size=_NativeFrozenLinearCETopK._ROW_BLOCK_SIZE,
            output_multiplier=ctx.output_multiplier,
            logit_softcap=ctx.logit_softcap,
        )
        return grad_hidden, None, None, None, None, None, None


class _FrozenLinearCETopK(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        top_k: int,
        vocab_block_size: int,
        output_multiplier: float,
        logit_softcap: float,
    ):
        hidden = hidden.contiguous()
        weight = weight.contiguous()
        labels = labels.to(device=hidden.device, dtype=torch.long).contiguous()
        rows, width = hidden.shape
        vocabulary = int(weight.shape[0])
        block = min(max(int(vocab_block_size), int(top_k)), vocabulary)
        maxima = torch.full(
            (rows,), -torch.inf, device=hidden.device, dtype=torch.float32
        )
        sums = torch.zeros((rows,), device=hidden.device, dtype=torch.float32)
        top_values = torch.full(
            (rows, int(top_k)), -torch.inf, device=hidden.device, dtype=torch.float32
        )
        top_ids = torch.zeros(
            (rows, int(top_k)), device=hidden.device, dtype=torch.long
        )

        for start in range(0, vocabulary, block):
            stop = min(start + block, vocabulary)
            raw = hidden @ weight[start:stop].t()
            logits = raw.float() * float(output_multiplier)
            if logit_softcap > 0:
                logits = torch.tanh(logits / logit_softcap) * logit_softcap
            local_max = logits.max(dim=-1).values
            new_max = torch.maximum(maxima, local_max)
            sums = sums * torch.exp(maxima - new_max) + torch.exp(
                logits - new_max[:, None]
            ).sum(dim=-1)
            maxima = new_max
            local_k = min(int(top_k), stop - start)
            local_values, local_ids = logits.topk(local_k, dim=-1)
            local_ids = local_ids + start
            merged_values = torch.cat((top_values, local_values), dim=-1)
            merged_ids = torch.cat((top_ids, local_ids), dim=-1)
            top_values, order = merged_values.topk(int(top_k), dim=-1)
            top_ids = merged_ids.gather(-1, order)

        target_raw = (hidden * weight.index_select(0, labels)).sum(dim=-1).float()
        target_logits = target_raw * float(output_multiplier)
        if logit_softcap > 0:
            target_logits = torch.tanh(target_logits / logit_softcap) * logit_softcap
        lse = maxima + torch.log(sums)
        losses = lse - target_logits
        probabilities = torch.exp(-losses)
        ctx.save_for_backward(hidden, weight, labels, lse, probabilities, top_ids)
        ctx.vocab_block_size = block
        ctx.output_multiplier = float(output_multiplier)
        ctx.logit_softcap = float(logit_softcap)
        ctx.mark_non_differentiable(top_ids)
        return losses, probabilities, top_values, top_ids

    @staticmethod
    def backward(ctx, grad_loss, grad_probability, grad_top_values, _grad_top_ids):
        hidden, weight, labels, lse, probabilities, top_ids = ctx.saved_tensors
        rows, width = hidden.shape
        vocabulary = int(weight.shape[0])
        grad_hidden = torch.zeros(
            (rows, width), device=hidden.device, dtype=torch.float32
        )
        grad_loss_f = (
            torch.zeros_like(probabilities) if grad_loss is None else grad_loss.float()
        )
        grad_probability_f = (
            torch.zeros_like(probabilities)
            if grad_probability is None
            else grad_probability.float()
        )
        ce_scale = grad_loss_f - grad_probability_f * probabilities
        block = int(ctx.vocab_block_size)
        for start in range(0, vocabulary, block):
            stop = min(start + block, vocabulary)
            raw = hidden @ weight[start:stop].t()
            transformed = raw.float() * ctx.output_multiplier
            if ctx.logit_softcap > 0:
                tanh_value = torch.tanh(transformed / ctx.logit_softcap)
                logits = tanh_value * ctx.logit_softcap
                transform_grad = ctx.output_multiplier * (1.0 - tanh_value.square())
            else:
                logits = transformed
                transform_grad = ctx.output_multiplier
            grad_logits = torch.exp(logits - lse[:, None]) * ce_scale[:, None]
            grad_logits = grad_logits * transform_grad
            weight_block = weight[start:stop]
            if weight_block.dtype in {torch.float16, torch.bfloat16}:
                grad_hidden.add_(
                    (grad_logits.to(weight_block.dtype) @ weight_block).float()
                )
            else:
                grad_hidden.add_(grad_logits @ weight_block.float())

        selected_weight = weight.index_select(0, labels).float()
        target_raw = (hidden.float() * selected_weight).sum(dim=-1)
        if ctx.logit_softcap > 0:
            target_tanh = torch.tanh(
                target_raw * ctx.output_multiplier / ctx.logit_softcap
            )
            target_derivative = ctx.output_multiplier * (1.0 - target_tanh.square())
        else:
            target_derivative = ctx.output_multiplier
        target_scale = ce_scale[:, None] * (
            target_derivative[:, None]
            if isinstance(target_derivative, torch.Tensor)
            else target_derivative
        )
        grad_hidden.sub_(target_scale * selected_weight)

        if grad_top_values is not None:
            flat_ids = top_ids.reshape(-1)
            top_weight = (
                weight.index_select(0, flat_ids)
                .float()
                .reshape(rows, top_ids.shape[1], width)
            )
            top_raw = torch.einsum("rd,rkd->rk", hidden.float(), top_weight)
            if ctx.logit_softcap > 0:
                top_tanh = torch.tanh(
                    top_raw * ctx.output_multiplier / ctx.logit_softcap
                )
                top_derivative = ctx.output_multiplier * (1.0 - top_tanh.square())
            else:
                top_derivative = ctx.output_multiplier
            grad_hidden.add_(
                torch.einsum(
                    "rk,rkd->rd",
                    grad_top_values.float() * top_derivative,
                    top_weight,
                )
            )
        return grad_hidden.to(hidden.dtype), None, None, None, None, None, None


def frozen_linear_ce_topk(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    top_k: int,
    vocab_block_size: int = 32768,
    output_multiplier: float = 1.0,
    logit_softcap: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return CE, gold probability, and exact top-k without an ``[N,V]`` tensor."""

    if hidden.ndim != 2 or weight.ndim != 2 or labels.ndim != 1:
        raise ValueError("frozen_linear_ce_topk expects [N,D], [V,D], and [N]")
    if hidden.shape[0] != labels.shape[0] or hidden.shape[1] != weight.shape[1]:
        raise ValueError("frozen_linear_ce_topk tensor dimensions are incompatible")
    if weight.requires_grad:
        raise ValueError("frozen_linear_ce_topk requires a frozen weight")
    if not 0 < int(top_k) <= int(weight.shape[0]):
        raise ValueError("top_k must be in the vocabulary range")
    if labels.numel() and (
        int(labels.min()) < 0 or int(labels.max()) >= weight.shape[0]
    ):
        raise ValueError("labels are outside the vocabulary")
    softcap = 0.0 if logit_softcap is None else float(logit_softcap)
    if logit_softcap is not None and softcap <= 0:
        raise ValueError("logit_softcap must be positive")
    implementation = (
        _NativeFrozenLinearCETopK
        if hidden.is_cuda and hidden.dtype in {torch.float16, torch.bfloat16}
        else _FrozenLinearCETopK
    )
    return implementation.apply(
        hidden,
        weight,
        labels,
        int(top_k),
        int(vocab_block_size),
        float(output_multiplier),
        softcap,
    )


__all__ = ["frozen_linear_ce_topk"]
