# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Memory-bounded conversion of vocabulary logits to soft token embeddings."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F


def _validate(
    hidden: torch.Tensor,
    weights: Sequence[torch.Tensor],
    *,
    row_chunk_size: int,
    vocab_chunk_size: int,
    logit_softcap: float | None,
) -> int:
    if hidden.ndim < 2:
        raise ValueError("hidden must have at least two dimensions")
    if int(row_chunk_size) <= 0 or int(vocab_chunk_size) <= 0:
        raise ValueError("row and vocabulary chunk sizes must be positive")
    if not weights:
        raise ValueError("at least one vocabulary weight shard is required")
    hidden_size = int(hidden.shape[-1])
    total_vocab = 0
    for weight in weights:
        if weight.ndim != 2:
            raise ValueError("vocabulary weight must be two-dimensional")
        if int(weight.shape[1]) != hidden_size:
            raise ValueError("vocabulary weight hidden dimension does not match hidden")
        if weight.device != hidden.device:
            raise ValueError("hidden and vocabulary weights must share a device")
        total_vocab += int(weight.shape[0])
    if total_vocab <= 0:
        raise ValueError("vocabulary must contain at least one row")
    if logit_softcap is not None and float(logit_softcap) <= 0.0:
        raise ValueError("logit_softcap must be positive")
    return hidden_size


def _project_logits(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    logit_softcap: float | None,
) -> torch.Tensor:
    logits = F.linear(hidden, weight).float()
    if logit_softcap is not None:
        cap = float(logit_softcap)
        logits = torch.tanh(logits / cap) * cap
    return logits


def _online_local_embedding(
    rows: torch.Tensor,
    weights: Sequence[torch.Tensor],
    *,
    hidden_size: int,
    vocab_chunk_size: int,
    logit_softcap: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Accumulate local softmax statistics with one vocabulary projection pass."""

    row_count = int(rows.shape[0])
    row_max = torch.full(
        (row_count,),
        -torch.inf,
        dtype=torch.float32,
        device=rows.device,
    )
    denominator = torch.zeros_like(row_max)
    numerator = torch.zeros(
        (row_count, hidden_size),
        dtype=torch.float32,
        device=rows.device,
    )
    for weight in weights:
        for vocab_start in range(0, int(weight.shape[0]), int(vocab_chunk_size)):
            vocab_stop = min(
                vocab_start + int(vocab_chunk_size),
                int(weight.shape[0]),
            )
            weight_chunk = weight[vocab_start:vocab_stop]
            logits = _project_logits(rows, weight_chunk, logit_softcap)
            chunk_max = logits.amax(dim=-1)
            next_max = torch.maximum(row_max, chunk_max)
            old_scale = torch.exp(row_max - next_max)
            probabilities = torch.exp(logits - next_max[:, None])
            denominator.mul_(old_scale).add_(probabilities.sum(dim=-1))
            numerator.mul_(old_scale[:, None])
            # BF16/FP16 weights stay in their compute dtype, avoiding an
            # enormous FP32 weight-tile allocation and retaining Tensor Cores.
            matmul_probabilities = probabilities.to(weight_chunk.dtype)
            numerator.add_((matmul_probabilities @ weight_chunk).float())
            row_max = next_max
    return row_max, denominator, numerator


@torch.no_grad()
def streaming_soft_embedding_from_shards(
    hidden: torch.Tensor,
    weights: Sequence[torch.Tensor],
    *,
    row_chunk_size: int,
    vocab_chunk_size: int,
    logit_softcap: float | None = None,
    embedding_scale: float = 1.0,
) -> torch.Tensor:
    """Reference shard merge used by tests and non-distributed callers.

    Only ``row_chunk_size * vocab_chunk_size`` logits are live at once.  FP32
    softmax statistics and numerator accumulation make the result independent
    of the chosen vocabulary tiling up to normal floating-point reduction order.
    """

    weights = tuple(weights)
    hidden_size = _validate(
        hidden,
        weights,
        row_chunk_size=row_chunk_size,
        vocab_chunk_size=vocab_chunk_size,
        logit_softcap=logit_softcap,
    )
    original_shape = tuple(hidden.shape)
    flat = hidden.reshape(-1, hidden_size)
    if flat.shape[0] == 0:
        return hidden.detach().clone()
    output = torch.empty_like(flat)
    for row_start in range(0, int(flat.shape[0]), int(row_chunk_size)):
        row_stop = min(row_start + int(row_chunk_size), int(flat.shape[0]))
        rows = flat[row_start:row_stop]
        _, denominator, numerator = _online_local_embedding(
            rows,
            weights,
            hidden_size=hidden_size,
            vocab_chunk_size=int(vocab_chunk_size),
            logit_softcap=logit_softcap,
        )
        output[row_start:row_stop] = (numerator / denominator[:, None]).to(output.dtype)
    if float(embedding_scale) != 1.0:
        output.mul_(float(embedding_scale))
    return output.reshape(original_shape).detach()


@torch.no_grad()
def streaming_soft_embedding(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    *,
    row_chunk_size: int,
    vocab_chunk_size: int,
    logit_softcap: float | None = None,
    embedding_scale: float = 1.0,
    tensor_parallel_group: Any | None = None,
    tensor_parallel_size: int = 1,
) -> torch.Tensor:
    """Return ``softmax(hidden @ weight.T) @ weight`` without full logits.

    ``weight`` is the local vocabulary shard when tensor parallelism is active.
    Row maxima, denominators, and weighted embedding numerators are reduced over
    the TP group.  The routine is intentionally forward-only: native
    DiffusionGemma self-conditioning detaches its first denoising pass.
    """

    hidden_size = _validate(
        hidden,
        (weight,),
        row_chunk_size=row_chunk_size,
        vocab_chunk_size=vocab_chunk_size,
        logit_softcap=logit_softcap,
    )
    tp_size = int(tensor_parallel_size)
    if tp_size <= 1:
        return streaming_soft_embedding_from_shards(
            hidden,
            (weight,),
            row_chunk_size=int(row_chunk_size),
            vocab_chunk_size=int(vocab_chunk_size),
            logit_softcap=logit_softcap,
            embedding_scale=float(embedding_scale),
        )
    if tensor_parallel_group is None or not dist.is_initialized():
        raise RuntimeError(
            "tensor-parallel streaming soft embeddings require an initialized group"
        )

    original_shape = tuple(hidden.shape)
    flat = hidden.reshape(-1, hidden_size)
    if flat.shape[0] == 0:
        return hidden.detach().clone()
    output = torch.empty_like(flat)
    for row_start in range(0, int(flat.shape[0]), int(row_chunk_size)):
        row_stop = min(row_start + int(row_chunk_size), int(flat.shape[0]))
        rows = flat[row_start:row_stop]
        local_max, denominator, numerator = _online_local_embedding(
            rows,
            (weight,),
            hidden_size=hidden_size,
            vocab_chunk_size=int(vocab_chunk_size),
            logit_softcap=logit_softcap,
        )
        row_max = local_max.clone()
        dist.all_reduce(row_max, op=dist.ReduceOp.MAX, group=tensor_parallel_group)
        shard_scale = torch.exp(local_max - row_max)
        denominator.mul_(shard_scale)
        numerator.mul_(shard_scale[:, None])
        dist.all_reduce(denominator, op=dist.ReduceOp.SUM, group=tensor_parallel_group)
        dist.all_reduce(numerator, op=dist.ReduceOp.SUM, group=tensor_parallel_group)
        output[row_start:row_stop] = (numerator / denominator[:, None]).to(output.dtype)
    if float(embedding_scale) != 1.0:
        output.mul_(float(embedding_scale))
    return output.reshape(original_shape).detach()


__all__ = [
    "streaming_soft_embedding",
    "streaming_soft_embedding_from_shards",
]
