# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Public contracts for diffusion-LM distributed planning.

These dataclasses are deliberately small and immutable. They describe model
semantics and distributed work without pulling in experiment configs or runtime
framework objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


AttentionMode = Literal[
    "block_causal",
    "bidirectional",
    "bidirectional_canvas",
    "hybrid_ar_dlm",
]
Reduction = Literal["none", "sum", "mean", "token_count"]
ObjectiveKind = Literal[
    "block_denoising",
    "canvas_denoising",
    "hybrid_ar_dlm",
]
CorruptionKind = Literal[
    "absorbing_mask",
    "model_defined",
]
TargetAttention = Literal[
    "bidirectional",
    "causal",
]
CPBPAttentionPolicy = Literal["production"]
CPBPTriState = Literal[
    "false",
    "true",
]
CPBPCleanKVDType = Literal["bf16"]
CPBPCleanKVLayout = Literal[
    "contiguous",
    "zigzag",
]
CPBPCleanKVTransport = Literal[
    "collective",
    "streaming",
]


@dataclass(frozen=True)
class CPBPPolicy:
    """Resolved production policy for CP and fused BP+CP attention.

    ``collective`` maximizes throughput by processing the complete clean K/V
    sequence in one packed kernel. ``streaming`` keeps clean K/V owner-sharded
    throughout attention to bound transient memory at extreme scale.
    """

    attention_policy: CPBPAttentionPolicy = "production"
    ragged_prefix: CPBPTriState = "false"
    clean_kv_dtype: CPBPCleanKVDType = "bf16"
    clean_kv_layout: CPBPCleanKVLayout = "zigzag"
    clean_kv_transport: CPBPCleanKVTransport = "collective"
    clean_reuse_overlap: bool = False
    debug_nonfinite_attention: bool = False

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "attention_policy": self.attention_policy,
            "ragged_prefix": self.ragged_prefix,
            "clean_kv_dtype": self.clean_kv_dtype,
            "clean_kv_layout": self.clean_kv_layout,
            "clean_kv_transport": self.clean_kv_transport,
            "packed_key_flex": "enabled",
            "backward_query_chunking": "disabled",
            "clean_reuse_overlap": bool(self.clean_reuse_overlap),
            "debug_nonfinite_attention": bool(self.debug_nonfinite_attention),
        }


@dataclass(frozen=True)
class ParallelSpec:
    data_parallel_size: int = 1
    context_parallel_size: int = 1
    block_parallel_size: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    expert_parallel_size: int = 1
    kv_backend: str = "replicated"
    sequence_parallel: bool = False
    tensor_parallel_overlap: bool = True

    @property
    def local_parallel_size(self) -> int:
        return max(self.context_parallel_size, self.block_parallel_size)

    @property
    def model_parallel_size(self) -> int:
        return (
            self.local_parallel_size
            * self.tensor_parallel_size
            * self.pipeline_parallel_size
            * self.expert_parallel_size
        )


@dataclass(frozen=True)
class SequenceRegion:
    name: str
    start: int
    stop: int
    role: str

    @property
    def length(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class BlockDiffusionObjective:
    """Objective semantics needed before assigning distributed work.

    ``exact_block_parallel`` means the objective decomposes into independent
    block-local denoising losses under teacher forcing. That is the invariant
    required by the fused BP/CP backend from the paper; it is intentionally
    separate from model family so Nemotron-style block denoisers and
    future compatible DLLMs can share the same optimized execution path.
    """

    name: str
    kind: ObjectiveKind
    clean_prefix: bool
    target_attention: TargetAttention
    corruption: CorruptionKind
    exact_block_parallel: bool
    mask_token_id: int | None = None
    loss_reduction: Reduction = "token_count"

    @property
    def supports_fused_cp_bp(self) -> bool:
        return (
            self.kind == "block_denoising"
            and self.clean_prefix
            and self.target_attention == "bidirectional"
            and self.exact_block_parallel
        )


@dataclass(frozen=True)
class DiffusionSchedule:
    regions: tuple[SequenceRegion, ...]
    attention_mode: AttentionMode
    num_blocks: int | None = None
    block_size: int | None = None
    objective: BlockDiffusionObjective | None = None


@dataclass(frozen=True)
class AttentionPlan:
    mode: AttentionMode
    query_region: SequenceRegion
    key_value_region: SequenceRegion
    shard_kv: bool = False
    key_chunk_size: int = 0


@dataclass(frozen=True)
class LossRegion:
    token_mask: Any
    denominator: Any
    reduction: Reduction
    scale_before_dp_average: float = 1.0
