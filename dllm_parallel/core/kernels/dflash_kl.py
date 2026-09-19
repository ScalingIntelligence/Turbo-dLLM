# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Streaming frozen-head KL used by DFlash distillation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class _FrozenLinearKL(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        draft_hidden: torch.Tensor,
        draft_weight: torch.Tensor,
        teacher_hidden: torch.Tensor,
        teacher_weight: torch.Tensor,
        logit_softcap: float,
    ) -> torch.Tensor:
        from dllm_parallel.core.kernels import chunked_linear_ce_native

        draft_hidden = draft_hidden.contiguous()
        draft_weight = draft_weight.contiguous()
        teacher_hidden = teacher_hidden.contiguous()
        teacher_weight = teacher_weight.contiguous()
        losses, draft_lse, teacher_lse = chunked_linear_ce_native.dflash_kl_forward(
            draft_hidden,
            draft_weight,
            teacher_hidden,
            teacher_weight,
            float(logit_softcap),
        )
        ctx.save_for_backward(
            draft_hidden,
            draft_weight,
            teacher_hidden,
            teacher_weight,
            draft_lse,
            teacher_lse,
        )
        ctx.logit_softcap = float(logit_softcap)
        return losses

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        from dllm_parallel.core.kernels import chunked_linear_ce_native

        (
            draft_hidden,
            draft_weight,
            teacher_hidden,
            teacher_weight,
            draft_lse,
            teacher_lse,
        ) = ctx.saved_tensors
        grad_hidden = chunked_linear_ce_native.dflash_kl_backward(
            draft_hidden,
            draft_weight,
            teacher_hidden,
            teacher_weight,
            draft_lse,
            teacher_lse,
            grad_output.contiguous(),
            ctx.logit_softcap,
        )
        return grad_hidden, None, None, None, None


def frozen_linear_kl(
    draft_hidden: torch.Tensor,
    draft_weight: torch.Tensor,
    teacher_hidden: torch.Tensor,
    teacher_weight: torch.Tensor,
    *,
    logit_softcap: float | None = None,
) -> torch.Tensor:
    """Return per-token forward KL without materializing full logits on CUDA."""

    if draft_hidden.ndim != 2 or teacher_hidden.ndim != 2:
        raise ValueError("DFlash KL hidden states must be [tokens, hidden]")
    if draft_hidden.shape[0] != teacher_hidden.shape[0]:
        raise ValueError("DFlash KL draft and teacher token counts differ")
    if draft_weight.ndim != 2 or teacher_weight.ndim != 2:
        raise ValueError("DFlash KL head weights must be [vocabulary, hidden]")
    if draft_weight.shape[0] != teacher_weight.shape[0]:
        raise ValueError("DFlash KL draft and teacher vocabularies differ")
    if draft_weight.shape[1] != draft_hidden.shape[1]:
        raise ValueError("DFlash KL draft head width is incompatible")
    if teacher_weight.shape[1] != teacher_hidden.shape[1]:
        raise ValueError("DFlash KL teacher head width is incompatible")
    if logit_softcap is not None and logit_softcap <= 0.0:
        raise ValueError("DFlash KL logit_softcap must be positive")
    softcap = 0.0 if logit_softcap is None else float(logit_softcap)
    if draft_hidden.is_cuda:
        tensors = (draft_weight, teacher_hidden, teacher_weight)
        if not all(tensor.is_cuda for tensor in tensors):
            raise ValueError("DFlash KL tensors must be on one CUDA device")
        if not all(tensor.dtype == draft_hidden.dtype for tensor in tensors):
            raise ValueError("DFlash KL tensors must use one dtype")
        return _FrozenLinearKL.apply(
            draft_hidden,
            draft_weight,
            teacher_hidden,
            teacher_weight,
            softcap,
        )
    if any(tensor.is_cuda for tensor in (draft_weight, teacher_hidden, teacher_weight)):
        raise ValueError("DFlash KL tensors must be on one device")
    draft_logits = F.linear(draft_hidden, draft_weight)
    teacher_logits = F.linear(teacher_hidden, teacher_weight)
    if softcap > 0.0:
        draft_logits = softcap * torch.tanh(draft_logits / softcap)
        teacher_logits = softcap * torch.tanh(teacher_logits / softcap)
    return F.kl_div(
        F.log_softmax(draft_logits, dim=-1, dtype=torch.float32),
        F.softmax(teacher_logits, dim=-1, dtype=torch.float32),
        reduction="none",
    ).sum(dim=-1)


__all__ = ["frozen_linear_kl"]
