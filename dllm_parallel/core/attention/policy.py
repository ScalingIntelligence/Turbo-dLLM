# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Policy resolution for production CP/BP attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from dllm_parallel.core.specs import CPBPPolicy


@dataclass(frozen=True)
class BDLMFA3KernelPolicy:
    """Launch policy for the validated BDLM FlashAttention-3 kernels."""

    max_backward_query_rows_per_launch: int
    max_fa3_head_dim: int = 256


def default_bdlm_fa3_kernel_policy() -> BDLMFA3KernelPolicy:
    return BDLMFA3KernelPolicy(
        max_backward_query_rows_per_launch=int(torch.iinfo(torch.int16).max),
    )


def cp_bp_policy(runtime: Any | None) -> CPBPPolicy:
    policy = getattr(runtime, "cp_bp_policy", None)
    if isinstance(policy, CPBPPolicy):
        return policy
    return CPBPPolicy()
