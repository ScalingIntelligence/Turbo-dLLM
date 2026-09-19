# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Tensor-parallel layers and collectives used by BDLM DiT models.

This package preserves the public import surface of the former
``tensor_parallel.py`` module. Implementation is split across focused
submodules (``_common``, ``random``, ``mappings``, ``sequence_parallel``,
``cross_entropy``, ``layers``); the optimized kernels, autograd collectives,
and numerics are unchanged.
"""

from __future__ import annotations

from dllm_parallel.core.parallel.tensor_parallel.cross_entropy import (
    runtime_vocab_parallel_logits_cross_entropy,
    vocab_parallel_logits_cross_entropy,
)
from dllm_parallel.core.parallel.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    VocabParallelLinear,
)
from dllm_parallel.core.parallel.tensor_parallel.mappings import (
    column_parallel_linear,
    copy_to_tensor_parallel_region,
    fused_gate_up_column_parallel_linear,
    fused_qkv_column_parallel_linear,
    reduce_from_tensor_parallel_region,
)
from dllm_parallel.core.parallel.tensor_parallel.random import (
    infer_tensor_parallel_rank,
    partition_bounds,
    partition_sizes,
    set_tensor_parallel_runtime,
    sync_tensor_parallel_rng_state,
    tensor_parallel_rank_from_config,
    tensor_parallel_size_from_config,
)
from dllm_parallel.core.parallel.tensor_parallel.sequence_parallel import (
    gather_active_from_sequence_parallel_region,
    gather_from_sequence_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
)

__all__ = [
    "ColumnParallelLinear",
    "RowParallelLinear",
    "VocabParallelEmbedding",
    "VocabParallelLinear",
    "column_parallel_linear",
    "copy_to_tensor_parallel_region",
    "fused_gate_up_column_parallel_linear",
    "fused_qkv_column_parallel_linear",
    "gather_active_from_sequence_parallel_region",
    "gather_from_sequence_parallel_region",
    "infer_tensor_parallel_rank",
    "partition_bounds",
    "partition_sizes",
    "reduce_from_tensor_parallel_region",
    "reduce_scatter_to_sequence_parallel_region",
    "runtime_vocab_parallel_logits_cross_entropy",
    "scatter_to_sequence_parallel_region",
    "set_tensor_parallel_runtime",
    "sync_tensor_parallel_rng_state",
    "tensor_parallel_rank_from_config",
    "tensor_parallel_size_from_config",
    "vocab_parallel_logits_cross_entropy",
]
