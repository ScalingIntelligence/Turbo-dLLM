# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Backbone resolver for no-weight config planning."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

from dllm_parallel.core.models.contracts import ModelFamilySpec
from dllm_parallel.core.models.compatibility import SUPPORTED_OBJECTIVES_BY_FAMILY
from dllm_parallel.core.specs import DiffusionSchedule, ParallelSpec


_BACKBONE_MODULES: dict[str, str] = {
    "causal_lm": "dllm_parallel.core.models.backbones.causal_lm",
    "dflash": "dllm_parallel.core.models.backbones.dflash",
    "diffusion_gemma": "dllm_parallel.core.models.backbones.diffusiongemma",
    "nemotron_labs_diffusion": "dllm_parallel.core.models.backbones.nemotron",
    "qwen3_8": "dllm_parallel.core.models.backbones.qwen3_8",
}
_METADATA_MODULES: dict[str, str] = {
    "causal_lm": "dllm_parallel.core.models.backbones.causal_lm",
    "diffusion_gemma": "dllm_parallel.core.models.backbones.diffusiongemma.metadata",
    "nemotron_labs_diffusion": "dllm_parallel.core.models.backbones.nemotron.metadata",
    "qwen3_8": "dllm_parallel.core.models.backbones.qwen3_8",
}
_PACKED_BACKBONE_MODULES: dict[str, str] = {
    "causal_lm": "dllm_parallel.core.models.backbones.causal_lm",
    "dflash": "dllm_parallel.core.models.backbones.dflash",
    "diffusion_gemma": "dllm_parallel.core.models.backbones.diffusiongemma",
    "nemotron_labs_diffusion": "dllm_parallel.core.models.backbones.nemotron",
    "qwen3_8": "dllm_parallel.core.models.backbones.qwen3_8",
}
_EXECUTOR_MODULES: dict[str, str] = {
    "causal_lm": "dllm_parallel.core.models.backbones.causal_lm",
    "dflash": "dllm_parallel.core.models.backbones.dflash",
    "diffusion_gemma": "dllm_parallel.core.models.backbones.diffusiongemma",
    "nemotron_labs_diffusion": "dllm_parallel.core.models.backbones.nemotron",
    "qwen3_8": "dllm_parallel.core.models.backbones.qwen3_8",
}
_BACKBONE_CACHE: dict[str, ModuleType] = {}
_METADATA_CACHE: dict[str, ModuleType] = {}
_EXECUTOR_CACHE: dict[str, Any] = {}


SUPPORTED_OPTIMIZER_BACKENDS = (
    "auto",
    "deepspeed_zero2",
    "torch_adamw",
    "torch_fused_adamw",
    "torch_distributed_adamw",
    "fsdp",
    "fsdp2",
)


def supported_families() -> tuple[str, ...]:
    return tuple(sorted(_BACKBONE_MODULES))


def supported_packed_families() -> tuple[str, ...]:
    return tuple(sorted(_PACKED_BACKBONE_MODULES))


def describe_supported_configs() -> dict[str, Any]:
    return {
        "families": list(supported_families()),
        "packed_block_diffusion_families": list(supported_packed_families()),
        "objectives_by_family": {
            family: list(objectives)
            for family, objectives in SUPPORTED_OBJECTIVES_BY_FAMILY.items()
        },
        "optimizer_backends": list(SUPPORTED_OPTIMIZER_BACKENDS),
        "parallel_axes": {
            "data_parallel": "supported",
            "context_parallel": "supported",
            "block_parallel": "supported",
            "tensor_parallel": "supported",
            "sequence_parallel": "supported",
            "pipeline_parallel": "unsupported (fail-closed)",
            "expert_parallel": "supported for DiffusionGemma MoE; dense backbones fail-closed",
        },
        "cp_bp_layout_rule": (
            "explicit replicated-prefix BP-only (context_parallel_size=1), "
            "CP-only (block_parallel_size=1), "
            "or fused CP/BP with block_parallel_size >= context_parallel_size and "
            "block_parallel_size divisible by context_parallel_size"
        ),
    }


def backbone_for_family(family: str) -> ModuleType:
    try:
        module_name = _BACKBONE_MODULES[family]
    except KeyError as exc:
        raise ValueError(f"unsupported diffusion model family: {family}") from exc
    module = _BACKBONE_CACHE.get(family)
    if module is None:
        module = importlib.import_module(module_name)
        _BACKBONE_CACHE[family] = module
    return module


def backbone_for_config(config: dict[str, Any]) -> ModuleType:
    return backbone_for_family(infer_family(config))


def metadata_for_family(family: str) -> ModuleType:
    module_name = _METADATA_MODULES.get(family)
    if module_name is None:
        return backbone_for_family(family)
    module = _METADATA_CACHE.get(family)
    if module is None:
        module = importlib.import_module(module_name)
        _METADATA_CACHE[family] = module
    return module


def executor_for_family(family: str) -> Any:
    try:
        module_name = _EXECUTOR_MODULES[family]
    except KeyError as exc:
        raise ValueError(
            f"diffusion family {family!r} does not expose a BackboneExecutor"
        ) from exc
    executor = _EXECUTOR_CACHE.get(family)
    if executor is None:
        module = importlib.import_module(module_name)
        builder = getattr(module, "build_executor", None)
        if builder is None:
            raise ValueError(
                f"diffusion family {family!r} does not expose build_executor"
            )
        executor = builder()
        _EXECUTOR_CACHE[family] = executor
    return executor


def infer_family(config: dict[str, Any]) -> str:
    explicit_family = config.get("dllm_model_family")
    if explicit_family in _BACKBONE_MODULES:
        return str(explicit_family)
    model_type = config.get("model_type")
    if model_type in _BACKBONE_MODULES:
        return str(model_type)
    architectures = tuple(config.get("architectures") or ())
    if any(str(name).endswith("ForCausalLM") for name in architectures):
        return "causal_lm"
    raise ValueError(
        "could not infer diffusion model family from config; expected "
        "a supported model_type or algo.name"
    )


def build_model(config: Any, *, vocab_size: int) -> Any:
    module = backbone_for_config(config)
    builder = getattr(module, "build_model", None)
    if builder is None:
        family = infer_family(config)
        raise ValueError(f"diffusion family {family!r} does not expose build_model")
    return builder(config, vocab_size=vocab_size)


def summarize_config(model_id: str, config: dict[str, Any]) -> ModelFamilySpec:
    return metadata_for_family(infer_family(config)).summarize_config(model_id, config)


def build_schedule(
    spec: ModelFamilySpec,
    *,
    sequence_length: int | None = None,
) -> DiffusionSchedule:
    return metadata_for_family(spec.family).build_schedule(
        spec,
        sequence_length=sequence_length,
    )


def validate_parallel_spec(
    spec: ModelFamilySpec,
    parallel: ParallelSpec,
) -> None:
    metadata_for_family(spec.family).validate_parallel_spec(spec, parallel)


def build_packed_block_diffusion_model(
    hf_model: Any,
    *,
    runtime: Any,
    seq_len: int,
    block_size: int,
    ring_attention_key_chunk_size: int = 0,
    activation_checkpointing: bool = True,
    activation_checkpointing_scope: str = "full",
    mlp_token_chunk_size: int = 0,
) -> Any:
    """Build a backbone-owned packed block-diffusion executor.

    The training path should not branch on concrete HF model families. Each
    optimized backbone owns its layer extraction and packed executor contract.
    """

    config = getattr(hf_model, "config", None)
    if config is None:
        raise ValueError("HF model must expose a config to select a DLLM backbone")
    config_dict = _config_to_dict(config)
    family = infer_family(config_dict)
    executor = executor_for_family(family)
    return executor.build_packed_block_diffusion_model(
        hf_model,
        runtime=runtime,
        seq_len=int(seq_len),
        block_size=int(block_size),
        ring_attention_key_chunk_size=int(ring_attention_key_chunk_size),
        activation_checkpointing=bool(activation_checkpointing),
        activation_checkpointing_scope=str(activation_checkpointing_scope),
        mlp_token_chunk_size=int(mlp_token_chunk_size),
    )


def _config_to_dict(config: Any) -> dict[str, Any]:
    if isinstance(config, dict):
        return dict(config)
    if hasattr(config, "to_dict"):
        value = config.to_dict()
        if isinstance(value, dict):
            return value
    raise TypeError("config must be a dict or expose to_dict()")
