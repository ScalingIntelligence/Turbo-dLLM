# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Reusable block-diffusion training utilities.

The packaged trainer imports this module for distributed optimizer, loss, and
batch-sync semantics. These functions are the shared production path for
diffusion models wrapped by DLLM BP/CP/TP runtimes.
"""

from __future__ import annotations
from typing import Any

import torch

from dllm_parallel.core.models.registry import (
    build_packed_block_diffusion_model as build_backbone_packed_block_diffusion_model,
)
from dllm_parallel.core.optim import (
    OptimizerPolicy,
    bounded_optimizer_named_param_groups,
    build_optimizer,
    configure_deepspeed_zero2_block_loss_scale,
    deepspeed_zero2_block_backward_scale,
    deepspeed_zero2_config,
    initialize_deepspeed_zero2,
    resolve_optimizer_backend,
    resolve_zero_optimizer_impl,
    sync_deepspeed_expert_parallel_gradients,
    sync_deepspeed_runtime_model_parallel_gradients,
    sync_deepspeed_sequence_parallel_gradients,
    verify_deepspeed_runtime_available,
    warm_deepspeed_zero2_collectives,
)


__all__ = (
    "OptimizerPolicy",
    "bounded_optimizer_named_param_groups",
    "build_optimizer",
    "build_packed_block_diffusion_model",
    "configure_deepspeed_zero2_block_loss_scale",
    "deepspeed_zero2_block_backward_scale",
    "deepspeed_zero2_config",
    "initialize_deepspeed_zero2",
    "resolve_optimizer_backend",
    "resolve_zero_optimizer_impl",
    "sync_deepspeed_expert_parallel_gradients",
    "sync_deepspeed_runtime_model_parallel_gradients",
    "sync_deepspeed_sequence_parallel_gradients",
    "verify_deepspeed_runtime_available",
    "warm_deepspeed_zero2_collectives",
)


def build_packed_block_diffusion_model(
    model: torch.nn.Module,
    *,
    runtime: Any,
    seq_len: int,
    block_size: int,
    ring_attention_key_chunk_size: int = 0,
    activation_checkpointing: bool = True,
    activation_checkpointing_scope: str = "full",
    mlp_token_chunk_size: int = 0,
) -> torch.nn.Module:
    return build_backbone_packed_block_diffusion_model(
        model,
        runtime=runtime,
        seq_len=int(seq_len),
        block_size=int(block_size),
        ring_attention_key_chunk_size=int(ring_attention_key_chunk_size),
        activation_checkpointing=bool(activation_checkpointing),
        activation_checkpointing_scope=str(activation_checkpointing_scope),
        mlp_token_chunk_size=int(mlp_token_chunk_size or 0),
    )
