# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Global token-accounting policy for the canonical trainer."""

from __future__ import annotations

from typing import Any

from dllm_parallel.core.objectives.runtime import TokenAccountingPolicy
from dllm_parallel.training.run_spec import RunSpec


def build_token_accounting_policy(
    *,
    spec: RunSpec,
    runtime: Any | None,
    world_size: int,
    gradient_accumulation_steps: int,
) -> TokenAccountingPolicy:
    training_spec = spec.training
    plan = getattr(runtime, "plan", None)
    if plan is not None:
        data_parallel_size = int(getattr(runtime, "sample_parallel_size", 1) or 1)
        context_parallel_size = int(getattr(plan, "context_parallel_size", 1) or 1)
        block_parallel_size = int(getattr(plan, "block_parallel_size", 1) or 1)
        tensor_parallel_size = int(getattr(plan, "tensor_parallel_size", 1) or 1)
    else:
        data_parallel_size = int(world_size)
        context_parallel_size = 1
        block_parallel_size = 1
        tensor_parallel_size = 1
    micro_batch_size = int(training_spec.batch_size)
    return TokenAccountingPolicy(
        global_batch_size=(
            micro_batch_size
            * int(gradient_accumulation_steps)
            * int(data_parallel_size)
        ),
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=int(gradient_accumulation_steps),
        data_parallel_size=int(data_parallel_size),
        context_parallel_size=int(context_parallel_size),
        block_parallel_size=int(block_parallel_size),
        tensor_parallel_size=int(tensor_parallel_size),
        sequence_parallel=bool(getattr(runtime, "sequence_parallel", False)),
    )
