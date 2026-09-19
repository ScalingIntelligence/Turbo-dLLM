# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Nemotron-Labs-Diffusion no-weight metadata and planning hooks."""

from __future__ import annotations

from typing import Any

from dllm_parallel.core.models.backbones.config_utils import (
    bool_or_none,
    first,
    optional_int,
    required_int,
    string,
    validate_positive_parallel,
    validate_tensor_parallel_heads,
)
from dllm_parallel.core.models.contracts import ModelFamilySpec
from dllm_parallel.core.objectives.block_diffusion import standard_block_diffusion_schedule
from dllm_parallel.core.specs import DiffusionSchedule, ParallelSpec


FAMILY = "nemotron_labs_diffusion"


def summarize_config(
    model_id: str,
    config: dict[str, Any],
) -> ModelFamilySpec:
    return ModelFamilySpec(
        model_id=model_id,
        family=FAMILY,
        model_type=config.get("model_type"),
        architecture=first(config.get("architectures")),
        remote_code_required=bool(config.get("auto_map")),
        hidden_size=optional_int(config.get("hidden_size")),
        num_layers=optional_int(config.get("num_hidden_layers")),
        num_attention_heads=optional_int(config.get("num_attention_heads")),
        num_key_value_heads=optional_int(config.get("num_key_value_heads")),
        head_dim=optional_int(config.get("head_dim") or config.get("head_dimension")),
        vocab_size=optional_int(config.get("vocab_size")),
        intermediate_size=optional_int(config.get("intermediate_size")),
        max_position_embeddings=optional_int(config.get("max_position_embeddings")),
        block_size=optional_int(config.get("block_size")),
        diffusion_paradigm=config.get("dlm_paradigm"),
        mask_token_id=optional_int(config.get("mask_token_id")),
        attn_implementation=string(config.get("attn_implementation")),
        use_cache=bool_or_none(config.get("use_cache")),
        dtype=config.get("torch_dtype") or config.get("dtype"),
        transformers_version=config.get("transformers_version"),
    )


def build_schedule(
    spec: ModelFamilySpec,
    *,
    sequence_length: int | None = None,
) -> DiffusionSchedule:
    block_size = required_int(spec.block_size, "Nemotron block_size")
    seq_len = required_int(
        sequence_length or spec.max_position_embeddings,
        "Nemotron sequence_length",
    )
    return standard_block_diffusion_schedule(
        sequence_length=seq_len,
        block_size=block_size,
        mask_token_id=spec.mask_token_id,
        region_prefix="block",
    )


def validate_parallel_spec(
    spec: ModelFamilySpec,
    parallel: ParallelSpec,
) -> None:
    validate_positive_parallel(parallel)
    validate_tensor_parallel_heads(spec, parallel)
    if parallel.pipeline_parallel_size != 1:
        raise ValueError(
            "Nemotron-Labs-Diffusion does not support pipeline_parallel_size > 1"
        )
    if parallel.expert_parallel_size != 1:
        raise ValueError(
            "Nemotron-Labs-Diffusion does not support expert_parallel_size > 1"
        )
    if parallel.kv_backend == "ring" and parallel.context_parallel_size <= 1:
        raise ValueError("ring K/V requires context_parallel_size > 1")
    if parallel.kv_backend == "replicated" and parallel.context_parallel_size != 1:
        raise ValueError("replicated K/V requires context_parallel_size == 1")
    if parallel.kv_backend == "replicated" and parallel.sequence_parallel:
        raise ValueError(
            "Nemotron sequence_parallel packed training requires "
            "context-parallel ring K/V"
        )
