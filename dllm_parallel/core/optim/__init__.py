# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Optimizer contracts for DLLM distributed training."""

from __future__ import annotations

from dllm_parallel.core.optim.zero import (
    FIRST_PARTY_OPTIMIZER_BACKENDS,
    FirstPartyDistributedAdamW,
    OptimizerPolicy,
    bounded_optimizer_named_param_groups,
    build_optimizer,
    configure_deepspeed_zero2_block_loss_scale,
    deepspeed_zero2_block_backward_scale,
    deepspeed_zero2_config,
    initialize_deepspeed_zero2,
    is_first_party_optimizer_backend,
    resolve_optimizer_backend,
    resolve_zero_optimizer_impl,
    sync_deepspeed_expert_parallel_gradients,
    sync_deepspeed_runtime_model_parallel_gradients,
    sync_deepspeed_sequence_parallel_gradients,
    verify_deepspeed_runtime_available,
    warm_deepspeed_zero2_collectives,
)
from dllm_parallel.core.optim.fsdp import (
    FSDPTrainingModule,
    FSDPWrapPolicy,
    wrap_model_with_fsdp,
)

__all__ = [
    "FIRST_PARTY_OPTIMIZER_BACKENDS",
    "FSDPTrainingModule",
    "FSDPWrapPolicy",
    "FirstPartyDistributedAdamW",
    "OptimizerPolicy",
    "bounded_optimizer_named_param_groups",
    "build_optimizer",
    "configure_deepspeed_zero2_block_loss_scale",
    "deepspeed_zero2_block_backward_scale",
    "deepspeed_zero2_config",
    "initialize_deepspeed_zero2",
    "is_first_party_optimizer_backend",
    "resolve_optimizer_backend",
    "resolve_zero_optimizer_impl",
    "sync_deepspeed_expert_parallel_gradients",
    "sync_deepspeed_runtime_model_parallel_gradients",
    "sync_deepspeed_sequence_parallel_gradients",
    "verify_deepspeed_runtime_available",
    "wrap_model_with_fsdp",
    "warm_deepspeed_zero2_collectives",
]
