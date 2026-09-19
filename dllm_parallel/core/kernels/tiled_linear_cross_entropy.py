# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

from typing import Optional

import torch


def _canonical_dtype(dtype: torch.dtype | str) -> torch.dtype:
  if isinstance(dtype, torch.dtype):
    return dtype
  if dtype == "float32":
    return torch.float32
  if dtype == "float64":
    return torch.float64
  if dtype == "bfloat16":
    return torch.bfloat16
  if dtype == "float16":
    return torch.float16
  raise ValueError(f"Unsupported dtype: {dtype}")


def _infer_weight_layout(
    hidden_dim: int,
    weight: torch.Tensor,
    weight_layout: str,
) -> bool:
  """Returns True when weight is [D, V], False when weight is [V, D]."""
  if weight_layout not in {"auto", "vocab_first", "dim_first"}:
    raise ValueError(
      "weight_layout must be one of 'auto', 'vocab_first', or 'dim_first'")
  if weight_layout == "vocab_first":
    if weight.ndim != 2 or weight.shape[1] != hidden_dim:
      raise ValueError(
        f"Expected vocab-first weight [V, {hidden_dim}], got {tuple(weight.shape)}")
    return False
  if weight_layout == "dim_first":
    if weight.ndim != 2 or weight.shape[0] != hidden_dim:
      raise ValueError(
        f"Expected dim-first weight [{hidden_dim}, V], got {tuple(weight.shape)}")
    return True
  if weight.ndim != 2:
    raise ValueError(f"lm_head must be a matrix, got shape {tuple(weight.shape)}")
  if weight.shape[1] == hidden_dim:
    return False
  if weight.shape[0] == hidden_dim:
    return True
  raise ValueError(
    f"Could not infer lm_head layout from hidden dim {hidden_dim} and "
    f"weight shape {tuple(weight.shape)}")


class _TiledLinearCrossEntropy(torch.autograd.Function):
  @staticmethod
  def forward(  # type: ignore[override]
      ctx,
      hidden: torch.Tensor,
      weight: torch.Tensor,
      bias: Optional[torch.Tensor],
      labels: torch.Tensor,
      row_block_size: int,
      vocab_block_size: int,
      reduction: str,
      ignore_index: int,
      exclude_index: int,
      compute_dtype: torch.dtype,
      weight_dim_first: bool,
      logit_softcap: float):
    if hidden.ndim not in {2, 3}:
      raise ValueError(f"hidden must be [N, D] or [B, S, D], got {hidden.shape}")
    if labels.shape != hidden.shape[:-1]:
      raise ValueError(
        f"labels shape {tuple(labels.shape)} must match hidden prefix "
        f"{tuple(hidden.shape[:-1])}")
    if reduction not in {"mean", "sum", "none"}:
      raise ValueError("reduction must be 'mean', 'sum', or 'none'")
    if row_block_size <= 0 or vocab_block_size <= 0:
      raise ValueError("row_block_size and vocab_block_size must be positive")

    hidden_shape = hidden.shape
    labels_shape = labels.shape
    hidden_flat = hidden.reshape(-1, hidden.shape[-1])
    labels_flat = labels.reshape(-1).to(torch.long)
    num_tokens, hidden_dim = hidden_flat.shape
    vocab_size = weight.shape[1] if weight_dim_first else weight.shape[0]
    if bias is not None and tuple(bias.shape) != (vocab_size,):
      raise ValueError(
        f"bias must have shape ({vocab_size},), got {tuple(bias.shape)}")

    valid = labels_flat != ignore_index
    if valid.any():
      valid_labels = labels_flat[valid]
      if valid_labels.min() < 0 or valid_labels.max() >= vocab_size:
        raise ValueError(
          f"labels must be in [0, {vocab_size}) or ignore_index={ignore_index}")
      if exclude_index >= 0 and (valid_labels == exclude_index).any():
        raise ValueError("labels may not equal exclude_index")

    losses = torch.zeros(num_tokens, device=hidden.device, dtype=torch.float32)
    lse = torch.empty(num_tokens, device=hidden.device, dtype=torch.float32)

    for token_start in range(0, num_tokens, row_block_size):
      token_stop = min(token_start + row_block_size, num_tokens)
      h = hidden_flat[token_start:token_stop].to(compute_dtype)
      y = labels_flat[token_start:token_stop]
      block_n = token_stop - token_start
      m = torch.full((block_n,), -float("inf"), device=hidden.device,
                     dtype=torch.float32)
      s = torch.zeros((block_n,), device=hidden.device, dtype=torch.float32)
      true_logits = torch.zeros((block_n,), device=hidden.device,
                                dtype=torch.float32)

      for vocab_start in range(0, vocab_size, vocab_block_size):
        vocab_stop = min(vocab_start + vocab_block_size, vocab_size)
        if weight_dim_first:
          w = weight[:, vocab_start:vocab_stop].to(compute_dtype)
          logits = h @ w
        else:
          w = weight[vocab_start:vocab_stop].to(compute_dtype)
          logits = h @ w.t()
        logits = logits.to(torch.float32)
        if bias is not None:
          logits = logits + bias[vocab_start:vocab_stop].to(torch.float32)
        if vocab_start <= exclude_index < vocab_stop:
          logits[:, exclude_index - vocab_start] = -float("inf")
        if logit_softcap > 0:
          logits = torch.tanh(logits / float(logit_softcap)) * float(logit_softcap)

        block_max = logits.max(dim=-1).values
        new_m = torch.maximum(m, block_max)
        s = torch.exp(m - new_m) * s + torch.exp(logits - new_m[:, None]).sum(dim=-1)
        m = new_m

        in_range = valid[token_start:token_stop] & (y >= vocab_start) & (y < vocab_stop)
        if in_range.any():
          local_y = y[in_range] - vocab_start
          true_logits[in_range] = logits[in_range, local_y]

      block_lse = m + torch.log(s)
      block_losses = block_lse - true_logits
      block_losses = torch.where(
        valid[token_start:token_stop],
        block_losses,
        torch.zeros_like(block_losses))
      lse[token_start:token_stop] = block_lse
      losses[token_start:token_stop] = block_losses

    valid_count = valid.sum().to(torch.float32)
    ctx.save_for_backward(hidden_flat, weight, bias, labels_flat, lse, valid)
    ctx.hidden_shape = hidden_shape
    ctx.labels_shape = labels_shape
    ctx.row_block_size = row_block_size
    ctx.vocab_block_size = vocab_block_size
    ctx.reduction = reduction
    ctx.ignore_index = ignore_index
    ctx.exclude_index = exclude_index
    ctx.compute_dtype = compute_dtype
    ctx.weight_dim_first = weight_dim_first
    ctx.vocab_size = vocab_size
    ctx.valid_count = valid_count
    ctx.logit_softcap = float(logit_softcap)

    if reduction == "none":
      return losses.reshape(labels_shape)
    if reduction == "sum":
      return losses.sum()
    return losses.sum() / valid_count

  @staticmethod
  def backward(ctx, grad_output):  # type: ignore[override]
    hidden_flat, weight, bias, labels_flat, lse, valid = ctx.saved_tensors
    num_tokens, hidden_dim = hidden_flat.shape
    vocab_size = ctx.vocab_size
    row_block_size = ctx.row_block_size
    vocab_block_size = ctx.vocab_block_size
    compute_dtype = ctx.compute_dtype
    weight_dim_first = ctx.weight_dim_first

    grad_hidden = torch.zeros_like(hidden_flat, dtype=torch.float32)
    grad_weight = torch.zeros_like(weight, dtype=torch.float32)
    grad_bias = (
      torch.zeros_like(bias, dtype=torch.float32)
      if bias is not None
      else None
    )

    if ctx.reduction == "none":
      token_scale = grad_output.reshape(-1).to(torch.float32)
    elif ctx.reduction == "sum":
      token_scale = torch.ones(
        (num_tokens,),
        device=hidden_flat.device,
        dtype=torch.float32) * grad_output.to(torch.float32)
    else:
      token_scale = torch.ones(
        (num_tokens,),
        device=hidden_flat.device,
        dtype=torch.float32) * (grad_output.to(torch.float32) / ctx.valid_count)
    token_scale = torch.where(valid, token_scale, torch.zeros_like(token_scale))

    for token_start in range(0, num_tokens, row_block_size):
      token_stop = min(token_start + row_block_size, num_tokens)
      h = hidden_flat[token_start:token_stop].to(compute_dtype)
      h_for_grad = hidden_flat[token_start:token_stop].to(torch.float32)
      y = labels_flat[token_start:token_stop]
      scale = token_scale[token_start:token_stop]
      block_lse = lse[token_start:token_stop]
      grad_h = torch.zeros(
        (token_stop - token_start, hidden_dim),
        device=hidden_flat.device,
        dtype=torch.float32)

      for vocab_start in range(0, vocab_size, vocab_block_size):
        vocab_stop = min(vocab_start + vocab_block_size, vocab_size)
        if weight_dim_first:
          w = weight[:, vocab_start:vocab_stop].to(compute_dtype)
          logits = h @ w
          w_for_grad = weight[:, vocab_start:vocab_stop].to(torch.float32)
        else:
          w = weight[vocab_start:vocab_stop].to(compute_dtype)
          logits = h @ w.t()
          w_for_grad = weight[vocab_start:vocab_stop].to(torch.float32)
        logits = logits.to(torch.float32)
        if bias is not None:
          logits = logits + bias[vocab_start:vocab_stop].to(torch.float32)
        if vocab_start <= ctx.exclude_index < vocab_stop:
          logits[:, ctx.exclude_index - vocab_start] = -float("inf")
        raw_logits = logits
        if ctx.logit_softcap > 0:
          logits = torch.tanh(raw_logits / ctx.logit_softcap) * ctx.logit_softcap

        grad_logits = torch.exp(logits - block_lse[:, None])
        in_range = (
          valid[token_start:token_stop]
          & (y >= vocab_start)
          & (y < vocab_stop)
        )
        if in_range.any():
          local_y = y[in_range] - vocab_start
          grad_logits[in_range, local_y] -= 1.0
        grad_logits = grad_logits * scale[:, None]
        if ctx.logit_softcap > 0:
          tanh_value = torch.tanh(raw_logits / ctx.logit_softcap)
          grad_logits = grad_logits * (1.0 - tanh_value * tanh_value)

        if weight_dim_first:
          grad_h = grad_h + grad_logits @ w_for_grad.t()
          grad_weight[:, vocab_start:vocab_stop] += h_for_grad.t() @ grad_logits
        else:
          grad_h = grad_h + grad_logits @ w_for_grad
          grad_weight[vocab_start:vocab_stop] += grad_logits.t() @ h_for_grad
        if grad_bias is not None:
          grad_bias[vocab_start:vocab_stop] += grad_logits.sum(dim=0)

      grad_hidden[token_start:token_stop] = grad_h

    grad_hidden = grad_hidden.reshape(ctx.hidden_shape).to(hidden_flat.dtype)
    grad_weight = grad_weight.to(weight.dtype)
    if grad_bias is not None:
      grad_bias = grad_bias.to(bias.dtype)
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
    )


class _ChunkedDenseLinearCrossEntropy(torch.autograd.Function):
  @staticmethod
  def forward(  # type: ignore[override]
      ctx,
      hidden: torch.Tensor,
      weight: torch.Tensor,
      bias: Optional[torch.Tensor],
      labels: torch.Tensor,
      row_block_size: int,
      reduction: str,
      ignore_index: int,
      exclude_index: int,
      compute_dtype: torch.dtype,
      weight_dim_first: bool,
      logit_softcap: float):
    hidden_shape = hidden.shape
    labels_shape = labels.shape
    hidden_flat = hidden.reshape(-1, hidden.shape[-1]).contiguous()
    labels_flat = labels.reshape(-1).to(torch.long).contiguous()
    weight = weight.contiguous()
    if bias is not None:
      bias = bias.contiguous()
    from dllm_parallel.core.kernels import chunked_linear_ce_native

    losses, lse, valid_count = chunked_linear_ce_native.forward(
      hidden_flat,
      weight,
      bias,
      labels_flat,
      row_block_size=int(row_block_size),
      ignore_index=ignore_index,
      exclude_index=exclude_index,
      logit_softcap=float(logit_softcap),
      compute_dtype=compute_dtype,
      weight_dim_first=weight_dim_first)

    ctx.save_for_backward(hidden_flat, weight, bias, labels_flat, lse, valid_count)
    ctx.hidden_shape = hidden_shape
    ctx.labels_shape = labels_shape
    ctx.row_block_size = int(row_block_size)
    ctx.reduction = reduction
    ctx.ignore_index = int(ignore_index)
    ctx.exclude_index = int(exclude_index)
    ctx.compute_dtype = compute_dtype
    ctx.weight_dim_first = weight_dim_first
    ctx.logit_softcap = float(logit_softcap)

    if reduction == "none":
      return losses.reshape(labels_shape)
    if reduction == "sum":
      return losses.sum()
    return losses.sum() / valid_count.squeeze(0)

  @staticmethod
  def backward(ctx, grad_output):  # type: ignore[override]
    hidden_flat, weight, bias, labels_flat, lse, valid_count = ctx.saved_tensors
    from dllm_parallel.core.kernels import chunked_linear_ce_native

    grad_hidden, grad_weight, grad_bias = chunked_linear_ce_native.backward(
      hidden_flat,
      weight,
      bias,
      labels_flat,
      lse,
      valid_count,
      grad_output,
      row_block_size=ctx.row_block_size,
      reduction=ctx.reduction,
      ignore_index=ctx.ignore_index,
      exclude_index=ctx.exclude_index,
      logit_softcap=ctx.logit_softcap,
      compute_dtype=ctx.compute_dtype,
      weight_dim_first=ctx.weight_dim_first)
    grad_hidden = grad_hidden.reshape(ctx.hidden_shape)
    if bias is None:
      grad_bias = None
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
    )


class _ChunkedFrozenLinearCrossEntropy(torch.autograd.Function):
  @staticmethod
  def forward(  # type: ignore[override]
      ctx,
      hidden: torch.Tensor,
      weight: torch.Tensor,
      bias: Optional[torch.Tensor],
      labels: torch.Tensor,
      row_block_size: int,
      reduction: str,
      ignore_index: int,
      exclude_index: int,
      compute_dtype: torch.dtype,
      weight_dim_first: bool,
      logit_softcap: float):
    output = _ChunkedDenseLinearCrossEntropy.forward(
      ctx,
      hidden,
      weight,
      bias,
      labels,
      row_block_size,
      reduction,
      ignore_index,
      exclude_index,
      compute_dtype,
      weight_dim_first,
      logit_softcap)
    return output

  @staticmethod
  def backward(ctx, grad_output):  # type: ignore[override]
    hidden_flat, weight, bias, labels_flat, lse, valid_count = ctx.saved_tensors
    from dllm_parallel.core.kernels import chunked_linear_ce_native

    grad_hidden, _, grad_bias = chunked_linear_ce_native.backward_frozen_weight(
      hidden_flat,
      weight,
      bias,
      labels_flat,
      lse,
      valid_count,
      grad_output,
      row_block_size=ctx.row_block_size,
      reduction=ctx.reduction,
      ignore_index=ctx.ignore_index,
      exclude_index=ctx.exclude_index,
      logit_softcap=ctx.logit_softcap,
      compute_dtype=ctx.compute_dtype,
      weight_dim_first=ctx.weight_dim_first)
    if bias is None:
      grad_bias = None
    return (
      grad_hidden.reshape(ctx.hidden_shape),
      None,
      grad_bias,
      None,
      None,
      None,
      None,
      None,
      None,
      None,
      None,
    )


def _should_use_chunked_dense_ce(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    labels: torch.Tensor,
    compute_dtype: torch.dtype,
    weight_dim_first: bool,
) -> bool:
  if not hidden.is_cuda or not weight.is_cuda or not labels.is_cuda:
    return False
  if hidden.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
    return False
  if weight.dtype != hidden.dtype:
    return False
  if bias is not None and bias.dtype != hidden.dtype:
    return False
  if compute_dtype not in {torch.float16, torch.bfloat16, torch.float32}:
    return False
  return True


def tiled_linear_cross_entropy(
    hidden: torch.Tensor,
    lm_head: torch.Tensor,
    labels: torch.Tensor,
    *,
    bias: Optional[torch.Tensor] = None,
    vocab_block_size: int = 0,
    reduction: str = "mean",
    ignore_index: int = -100,
    exclude_index: Optional[int] = None,
    logit_softcap: float | None = None,
    dtype: torch.dtype | str = torch.float32,
    weight_layout: str = "auto",
) -> torch.Tensor:
  """Linear projection plus cross entropy without materializing [N, V] logits.

  Args:
    hidden: Token representations with shape [B, S, D] or [N, D].
    lm_head: Output projection weight. By default layout is inferred as [V, D]
      when the last dimension matches D, otherwise [D, V] when the first
      dimension matches D. Use ``weight_layout`` to disambiguate square weights.
    labels: Target token ids with shape [B, S] or [N].
    bias: Optional output projection bias with shape [V].
    vocab_block_size: Vocabulary entries per CPU fallback tile. 0 selects the
      full vocabulary.
    reduction: "mean", "sum", or "none".
    ignore_index: Label value excluded from the numerator and denominator.
    exclude_index: Optional vocabulary id to exclude from the softmax
      denominator, useful for absorbing mask tokens.
    logit_softcap: Optional positive cap applied as ``cap * tanh(logit / cap)``
      before cross entropy.
    dtype: Compute dtype for tiled matmuls. Accumulation stays in fp32.
    weight_layout: "auto", "vocab_first" for [V, D], or "dim_first" for [D, V].

  Returns:
    The same loss as ``F.cross_entropy(hidden @ lm_head.T + bias, labels)`` for
    vocab-first weights, up to normal floating-point associativity differences.
  """
  if hidden.shape[:-1] != labels.shape:
    raise ValueError(
      f"labels shape {tuple(labels.shape)} must match hidden prefix "
      f"{tuple(hidden.shape[:-1])}")
  compute_dtype = _canonical_dtype(dtype)
  weight_dim_first = _infer_weight_layout(
    hidden.shape[-1],
    lm_head,
    weight_layout)
  if _should_use_chunked_dense_ce(
      hidden,
      lm_head,
      bias,
      labels,
      compute_dtype,
      weight_dim_first):
    implementation = (
      _ChunkedDenseLinearCrossEntropy
      if lm_head.requires_grad
      else _ChunkedFrozenLinearCrossEntropy
    )
    return implementation.apply(
      hidden,
      lm_head,
      bias,
      labels,
      0,
      reduction,
      int(ignore_index),
      -1 if exclude_index is None else int(exclude_index),
      compute_dtype,
      weight_dim_first,
      0.0 if logit_softcap is None else float(logit_softcap),
    )
  if hidden.is_cuda or lm_head.is_cuda or labels.is_cuda:
    raise RuntimeError(
      "CUDA tiled_linear_cross_entropy requires the native chunked dense CE "
      "path; refusing to fall back to the Python tiled implementation.")
  row_block_size = hidden.reshape(-1, hidden.shape[-1]).shape[0]
  vocab_size = lm_head.shape[1] if weight_dim_first else lm_head.shape[0]
  vocab_block_size = int(vocab_size) if vocab_block_size <= 0 else int(vocab_block_size)
  return _TiledLinearCrossEntropy.apply(
    hidden,
    lm_head,
    bias,
    labels,
    int(row_block_size),
    int(vocab_block_size),
    reduction,
    int(ignore_index),
    -1 if exclude_index is None else int(exclude_index),
    compute_dtype,
    weight_dim_first,
    0.0 if logit_softcap is None else float(logit_softcap),
  )
