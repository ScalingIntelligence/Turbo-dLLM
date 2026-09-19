# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Diffusion-LM backbones and no-weight model-family metadata."""

from __future__ import annotations

from dllm_parallel.core.models.registry import (
    backbone_for_config,
    backbone_for_family,
    build_model,
    build_packed_block_diffusion_model,
    build_schedule,
    executor_for_family,
    infer_family,
    summarize_config,
    validate_parallel_spec,
)
from dllm_parallel.core.models.contracts import ModelFamilySpec

_HF_LOADER_EXPORTS = {
    "HuggingFaceModelBundle",
    "load_hf_config",
    "load_hf_model_from_config",
    "load_hf_model_bundle",
    "summarize_hf_config",
}

__all__ = [
    "HuggingFaceModelBundle",
    "ModelFamilySpec",
    "backbone_for_config",
    "backbone_for_family",
    "build_model",
    "build_packed_block_diffusion_model",
    "build_schedule",
    "executor_for_family",
    "infer_family",
    "load_hf_config",
    "load_hf_model_from_config",
    "load_hf_model_bundle",
    "summarize_config",
    "summarize_hf_config",
    "validate_parallel_spec",
]


def __getattr__(name: str):
    if name in _HF_LOADER_EXPORTS:
        from dllm_parallel.core.models import hf_loader

        value = getattr(hf_loader, name)
        globals()[name] = value
        return value
    raise AttributeError(name)
