# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""DiffusionGemma no-weight metadata and planning hooks."""

from __future__ import annotations

from typing import Any

from dllm_parallel.core.models.backbones.config_utils import (
    first,
    node,
    optional_int,
    required_int,
    string,
    string_tuple,
    validate_positive_parallel,
    validate_tensor_parallel_heads,
)
from dllm_parallel.core.models.contracts import ModelFamilySpec
from dllm_parallel.core.objectives.block_diffusion import (
    standard_block_diffusion_schedule,
)
from dllm_parallel.core.specs import DiffusionSchedule, ParallelSpec


FAMILY = "diffusion_gemma"


def summarize_config(
    model_id: str,
    config: dict[str, Any],
) -> ModelFamilySpec:
    text_config = node(config, "text_config")
    return ModelFamilySpec(
        model_id=model_id,
        family=FAMILY,
        model_type=config.get("model_type"),
        architecture=first(config.get("architectures")),
        remote_code_required=bool(config.get("auto_map")),
        hidden_size=optional_int(text_config.get("hidden_size")),
        num_layers=optional_int(text_config.get("num_hidden_layers")),
        num_attention_heads=optional_int(text_config.get("num_attention_heads")),
        num_key_value_heads=optional_int(text_config.get("num_key_value_heads")),
        num_global_key_value_heads=optional_int(
            text_config.get("num_global_key_value_heads")
        ),
        attention_layer_types=string_tuple(text_config.get("layer_types")),
        layer_types=string_tuple(text_config.get("layer_types")),
        head_dim=optional_int(text_config.get("head_dim")),
        global_head_dim=optional_int(text_config.get("global_head_dim")),
        vocab_size=optional_int(text_config.get("vocab_size")),
        intermediate_size=optional_int(text_config.get("intermediate_size")),
        max_position_embeddings=optional_int(
            text_config.get("max_position_embeddings")
        ),
        canvas_length=optional_int(config.get("canvas_length")),
        block_size=optional_int(config.get("block_size") or config.get("canvas_length")),
        mask_token_id=optional_int(
            config.get("mask_token_id") or text_config.get("mask_token_id")
        ),
        num_experts=optional_int(text_config.get("num_experts")),
        top_k_experts=optional_int(text_config.get("top_k_experts")),
        expert_intermediate_size=optional_int(
            text_config.get("expert_intermediate_size")
            or text_config.get("moe_intermediate_size")
        ),
        sliding_window=optional_int(text_config.get("sliding_window")),
        use_bidirectional_attention=string(
            text_config.get("use_bidirectional_attention")
        ),
        dtype=config.get("dtype") or text_config.get("dtype"),
        transformers_version=config.get("transformers_version"),
    )


def build_schedule(
    spec: ModelFamilySpec,
    *,
    sequence_length: int | None = None,
) -> DiffusionSchedule:
    block_size = required_int(
        spec.block_size or spec.canvas_length,
        "DiffusionGemma canvas_length",
    )
    seq_len = required_int(
        sequence_length or spec.max_position_embeddings,
        "DiffusionGemma sequence_length",
    )
    return standard_block_diffusion_schedule(
        sequence_length=seq_len,
        block_size=block_size,
        mask_token_id=spec.mask_token_id,
        corruption="absorbing_mask",
        region_prefix="canvas",
    )


def validate_parallel_spec(
    spec: ModelFamilySpec,
    parallel: ParallelSpec,
) -> None:
    validate_positive_parallel(parallel)
    validate_tensor_parallel_heads(spec, parallel)
    if parallel.pipeline_parallel_size != 1:
        raise ValueError("DiffusionGemma does not support pipeline_parallel_size > 1")
    if parallel.kv_backend == "ring" and parallel.context_parallel_size <= 1:
        raise ValueError("ring K/V requires context_parallel_size > 1")
    if parallel.kv_backend == "replicated" and parallel.context_parallel_size != 1:
        raise ValueError("replicated K/V requires context_parallel_size == 1")
    if parallel.kv_backend == "replicated" and parallel.sequence_parallel:
        raise ValueError(
            "DiffusionGemma sequence_parallel packed training requires "
            "context-parallel ring K/V"
        )
    if spec.num_experts and spec.num_experts % parallel.expert_parallel_size != 0:
        raise ValueError(
            "DiffusionGemma num_experts must divide evenly by "
            "expert_parallel_size"
        )
    if (
        spec.num_experts is not None
        and spec.top_k_experts is not None
        and spec.top_k_experts > spec.num_experts
    ):
        raise ValueError("DiffusionGemma top_k_experts cannot exceed num_experts")
