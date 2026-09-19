# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Nemotron Labs Diffusion backbone (HF-wrapped): metadata + packed executor."""

from dllm_parallel.core.models.backbones.nemotron.metadata import (
    build_schedule,
    summarize_config,
    validate_parallel_spec,
)

_MODEL_EXPORTS = {
    "NemotronLabsDiffusionBackboneExecutor",
    "NemotronLabsDiffusionPackedBlockDiffusionModel",
    "_TEPackedLayerProjections",
    "_hf_gated_mlp_forward",
    "_te_gated_mlp_forward",
    "build_executor",
    "build_packed_block_diffusion_model",
}

__all__ = [
    "build_schedule",
    "summarize_config",
    "validate_parallel_spec",
    "NemotronLabsDiffusionBackboneExecutor",
    "NemotronLabsDiffusionPackedBlockDiffusionModel",
    "_TEPackedLayerProjections",
    "_hf_gated_mlp_forward",
    "_te_gated_mlp_forward",
    "build_executor",
    "build_packed_block_diffusion_model",
]


def __getattr__(name: str):
    if name in _MODEL_EXPORTS:
        from dllm_parallel.core.models.backbones.nemotron import model

        value = getattr(model, name)
        globals()[name] = value
        return value
    raise AttributeError(name)
