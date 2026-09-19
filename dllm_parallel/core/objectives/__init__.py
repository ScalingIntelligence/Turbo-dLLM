# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

from dllm_parallel.core.objectives.block_diffusion import (
    STANDARD_BLOCK_DIFFUSION_OBJECTIVE,
    active_block_loss_region,
    standard_block_diffusion_objective,
    standard_block_diffusion_schedule,
)

__all__ = [
    "STANDARD_BLOCK_DIFFUSION_OBJECTIVE",
    "DFLASH_OBJECTIVE",
    "FAST_DLLM_V2_OBJECTIVE",
    "FastDLLMv2ObjectiveRuntime",
    "DFlashObjectiveBatch",
    "DFlashObjectiveRuntime",
    "reference_dflash_loss",
    "active_block_loss_region",
    "reduce_token_losses",
    "standard_block_diffusion_objective",
    "standard_block_diffusion_schedule",
    "fast_dllm_v2_schedule",
    "token_count_denominator",
]


def __getattr__(name: str):
    if name in {"reduce_token_losses", "token_count_denominator"}:
        from dllm_parallel.core.objectives import loss

        value = getattr(loss, name)
        globals()[name] = value
        return value
    if name in {
        "DFLASH_OBJECTIVE",
        "DFlashObjectiveBatch",
        "DFlashObjectiveRuntime",
        "reference_dflash_loss",
    }:
        from dllm_parallel.core.objectives import dflash

        value = getattr(dflash, name)
        globals()[name] = value
        return value
    if name in {
        "FAST_DLLM_V2_OBJECTIVE",
        "FastDLLMv2ObjectiveRuntime",
        "fast_dllm_v2_schedule",
    }:
        from dllm_parallel.core.objectives import fast_dllm_v2

        value = getattr(fast_dllm_v2, name)
        globals()[name] = value
        return value
    raise AttributeError(name)
