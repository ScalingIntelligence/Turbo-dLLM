# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Generic teacher-forced block-diffusion objective helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dllm_parallel.core.specs import (
    BlockDiffusionObjective,
    DiffusionSchedule,
    LossRegion,
    Reduction,
    SequenceRegion,
)

if TYPE_CHECKING:
    import torch

    from dllm_parallel.core.parallel.runtime import ParallelRuntime


STANDARD_BLOCK_DIFFUSION_OBJECTIVE = "standard_block_diffusion_denoising"


def standard_block_diffusion_objective(
    *,
    mask_token_id: int | None = None,
    corruption: str | None = None,
) -> BlockDiffusionObjective:
    """Return the only fused BP/CP objective currently supported.

    Backbone modules may supply model metadata such as ``mask_token_id``;
    the denoising semantics stay centralized here.
    """

    if corruption is None:
        corruption = "absorbing_mask" if mask_token_id is not None else "model_defined"
    if corruption not in {"absorbing_mask", "model_defined"}:
        raise ValueError(f"unsupported standard block diffusion corruption: {corruption}")
    return BlockDiffusionObjective(
        name=STANDARD_BLOCK_DIFFUSION_OBJECTIVE,
        kind="block_denoising",
        clean_prefix=True,
        target_attention="bidirectional",
        corruption=corruption,
        exact_block_parallel=True,
        mask_token_id=mask_token_id,
    )


def standard_block_diffusion_schedule(
    *,
    sequence_length: int,
    block_size: int,
    mask_token_id: int | None = None,
    corruption: str | None = None,
    region_prefix: str = "block",
) -> DiffusionSchedule:
    """Build a model-neutral standard block diffusion schedule."""

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if sequence_length % block_size != 0:
        raise ValueError("sequence_length must divide evenly by block_size")
    regions = tuple(
        SequenceRegion(
            name=f"{region_prefix}_{block}",
            start=block * block_size,
            stop=(block + 1) * block_size,
            role="clean_prefix_or_noisy_target",
        )
        for block in range(sequence_length // block_size)
    )
    return DiffusionSchedule(
        regions=regions,
        attention_mode="block_causal",
        num_blocks=len(regions),
        block_size=block_size,
        objective=standard_block_diffusion_objective(
            mask_token_id=mask_token_id,
            corruption=corruption,
        ),
    )


def active_block_loss_region(
    *,
    attention_mask: torch.Tensor,
    block_size: int,
    runtime: ParallelRuntime,
    denominator: torch.Tensor | float | None = None,
    reduction: Reduction = "token_count",
) -> LossRegion:
    """Build the local active-token loss contract for exact BP training.

    The helper is model-family neutral: it only encodes the distributed
    ownership invariant from the paper. Backbone modules remain responsible for
    deciding whether their denoising objective is safe for block parallelism.
    """

    import torch
    from dllm_parallel.core.parallel.runtime import loss_scale

    if attention_mask.ndim == 2:
        seq_len = attention_mask.shape[1]
        token_mask = _active_or_all_tokens(
            seq_len=seq_len,
            block_size=block_size,
            device=attention_mask.device,
            runtime=runtime,
        )
        token_mask = attention_mask.to(torch.float32) * token_mask[None, :]
    elif attention_mask.ndim == 1:
        seq_len = attention_mask.shape[0]
        token_mask = _active_or_all_tokens(
            seq_len=seq_len,
            block_size=block_size,
            device=attention_mask.device,
            runtime=runtime,
        )
        token_mask = attention_mask.to(torch.float32) * token_mask
    else:
        raise ValueError("attention_mask must be 1D or 2D")

    if denominator is None:
        denominator = attention_mask.to(torch.float32).sum()
    return LossRegion(
        token_mask=token_mask,
        denominator=denominator,
        reduction=reduction,
        scale_before_dp_average=loss_scale(runtime),
    )


def _active_or_all_tokens(
    *,
    seq_len: int,
    block_size: int,
    device: torch.device,
    runtime: ParallelRuntime,
) -> torch.Tensor:
    import torch
    from dllm_parallel.core.parallel.runtime import active_token_mask

    token_mask = active_token_mask(
        seq_len=seq_len,
        block_size=block_size,
        device=device,
        runtime=runtime,
    )
    if token_mask is None:
        token_mask = torch.ones(seq_len, device=device)
    return token_mask
