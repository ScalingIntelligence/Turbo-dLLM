# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Profiling utilities for DLLM distributed runs."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "DFLASH_WORK_COUNT_NAMES",
    "DFlashTransformerFlops",
    "FastDLLMv2TransformerFlops",
    "FlopsModel",
    "MegatronTransformerFlops",
    "PerfProfiler",
    "block_diffusion_active_packed_factors",
    "block_diffusion_sharded_clean_factors",
    "block_diffusion_sparse_attention_pairs",
    "model_flops_utilization_pct",
    "throughput_metrics",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        module = importlib.import_module("dllm_parallel.core.profiling.perf")
        return getattr(module, name)
    raise AttributeError(name)
