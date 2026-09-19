# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Public tensor-parallel layer API.

This module is the stable import surface for backbone code. Implementations
live in ``tensor_parallel.py`` so existing optimized kernels and autograd
collectives remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dllm_parallel.core.parallel.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    VocabParallelLinear,
    column_parallel_linear,
    copy_to_tensor_parallel_region,
    fused_gate_up_column_parallel_linear,
    fused_qkv_column_parallel_linear,
    gather_active_from_sequence_parallel_region,
    gather_from_sequence_parallel_region,
    partition_bounds,
    partition_sizes,
    reduce_from_tensor_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    runtime_vocab_parallel_logits_cross_entropy,
    set_tensor_parallel_runtime,
    scatter_to_sequence_parallel_region,
    tensor_parallel_rank_from_config,
    tensor_parallel_size_from_config,
    vocab_parallel_logits_cross_entropy,
)

VocabParallelOutput = VocabParallelLinear


@dataclass(frozen=True)
class TensorParallelLayerConfig:
    tensor_parallel_size: int = 1
    tensor_parallel_rank: int = 0
    sequence_parallel: bool = False
    overlap: bool = True

    @classmethod
    def from_runtime(cls, runtime: Any | None) -> "TensorParallelLayerConfig":
        return cls(
            tensor_parallel_size=int(
                getattr(runtime, "tensor_parallel_size", 1) or 1
            ),
            tensor_parallel_rank=int(
                getattr(runtime, "tensor_parallel_rank", 0) or 0
            ),
            sequence_parallel=bool(
                getattr(runtime, "sequence_parallel", False)
            ),
            overlap=bool(getattr(runtime, "tensor_parallel_overlap", True)),
        )

    @classmethod
    def from_config(cls, config: Any) -> "TensorParallelLayerConfig":
        parallel = getattr(config, "parallel", None)
        if parallel is None:
            return cls()
        if hasattr(parallel, "get"):
            get = parallel.get
        else:
            get = lambda key, default=None: getattr(parallel, key, default)
        return cls(
            tensor_parallel_size=int(get("tensor_parallel_size", 1) or 1),
            tensor_parallel_rank=int(get("tensor_parallel_rank", 0) or 0),
            sequence_parallel=bool(get("sequence_parallel", False)),
            overlap=bool(get("tensor_parallel_overlap", True)),
        )


__all__ = [
    "ColumnParallelLinear",
    "RowParallelLinear",
    "TensorParallelLayerConfig",
    "VocabParallelEmbedding",
    "VocabParallelLinear",
    "VocabParallelOutput",
    "column_parallel_linear",
    "copy_to_tensor_parallel_region",
    "fused_gate_up_column_parallel_linear",
    "fused_qkv_column_parallel_linear",
    "gather_active_from_sequence_parallel_region",
    "gather_from_sequence_parallel_region",
    "partition_bounds",
    "partition_sizes",
    "reduce_from_tensor_parallel_region",
    "reduce_scatter_to_sequence_parallel_region",
    "runtime_vocab_parallel_logits_cross_entropy",
    "scatter_to_sequence_parallel_region",
    "set_tensor_parallel_runtime",
    "tensor_parallel_rank_from_config",
    "tensor_parallel_size_from_config",
    "vocab_parallel_logits_cross_entropy",
]
