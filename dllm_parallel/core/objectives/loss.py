# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Loss-region helpers with explicit reduction semantics."""

from __future__ import annotations

import torch

from dllm_parallel.core.specs import LossRegion


def reduce_token_losses(
    token_losses: torch.Tensor,
    region: LossRegion,
) -> torch.Tensor:
    """Reduce token losses according to a ``LossRegion`` contract."""

    mask = region.token_mask.to(device=token_losses.device, dtype=token_losses.dtype)
    if mask.shape != token_losses.shape:
        raise ValueError(
            "LossRegion.token_mask shape must match token_losses shape, got "
            f"{tuple(mask.shape)} and {tuple(token_losses.shape)}"
        )
    masked = token_losses * mask
    if region.reduction == "none":
        return masked * float(region.scale_before_dp_average)
    numerator = masked.sum()
    if region.reduction == "sum":
        reduced = numerator
    elif region.reduction in {"mean", "token_count"}:
        denominator = _denominator(region.denominator, token_losses)
        reduced = numerator / denominator.clamp_min(torch.finfo(numerator.dtype).tiny)
    else:
        raise ValueError(f"unsupported loss reduction: {region.reduction}")
    return reduced * float(region.scale_before_dp_average)


def token_count_denominator(token_mask: torch.Tensor) -> torch.Tensor:
    return token_mask.to(torch.float32).sum()


def _denominator(value, like: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=like.device, dtype=like.dtype)
    return torch.tensor(float(value), device=like.device, dtype=like.dtype)
