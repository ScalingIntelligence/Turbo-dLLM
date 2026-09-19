# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Attention plans and exact CP attention backends.

Exports are lazy so metadata/model inspection imports do not eagerly load
torch, FlashAttention, or native CP/BP helpers.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "BlockDenoisingFullMask",
    "BlockDenoisingGlobalCleanMask",
    "BlockDenoisingLocalActiveMask",
    "DFlashGlobalContextMask",
    "DFlashLocalBlockMask",
    "CPBPCounters",
    "all_clean_shards",
    "clean_shards_for_rank",
    "context_parallel_attention",
    "full_mask_block_denoising_attention_bshd",
    "cp_bp_counters_from_visits",
    "merge_attention_stats",
    "prefix_visits_for_noisy_blocks",
    "pure_context_block_denoising_attention_bshd",
    "pure_context_persistent_block_denoising_attention_bshd",
    "replicated_block_denoising_attention_bshd",
    "replicated_kv_attention",
    "ring_owner_traversal",
    "ring_attention_with_local_kv",
    "ring_attention_with_local_kv_shard",
    "fused_block_context_attention_bshd",
    "ring_context_parallel_attention",
    "streaming_attention",
]


_MASK_EXPORTS = {
    "BlockDenoisingFullMask",
    "BlockDenoisingGlobalCleanMask",
    "BlockDenoisingLocalActiveMask",
    "DFlashGlobalContextMask",
    "DFlashLocalBlockMask",
}

_CP_BACKEND_EXPORTS = {
    "CPBPCounters",
    "all_clean_shards",
    "clean_shards_for_rank",
    "cp_bp_counters_from_visits",
    "merge_attention_stats",
    "prefix_visits_for_noisy_blocks",
    "ring_owner_traversal",
}

_CONTEXT_EXPORTS = {
    "context_parallel_attention",
    "replicated_block_denoising_attention_bshd",
    "replicated_kv_attention",
    "ring_attention_with_local_kv",
    "ring_attention_with_local_kv_shard",
    "fused_block_context_attention_bshd",
    "pure_context_block_denoising_attention_bshd",
    "pure_context_persistent_block_denoising_attention_bshd",
    "ring_context_parallel_attention",
    "streaming_attention",
}

_FULL_MASK_EXPORTS = {"full_mask_block_denoising_attention_bshd"}


def __getattr__(name: str) -> Any:
    if name in _MASK_EXPORTS:
        module = importlib.import_module("dllm_parallel.core.attention.masks")
        return getattr(module, name)
    if name in _CP_BACKEND_EXPORTS:
        module = importlib.import_module("dllm_parallel.core.attention.cp_backend")
        return getattr(module, name)
    if name in _CONTEXT_EXPORTS:
        module = importlib.import_module("dllm_parallel.core.attention.context_parallel_attention")
        return getattr(module, name)
    if name in _FULL_MASK_EXPORTS:
        module = importlib.import_module("dllm_parallel.core.attention.full_mask")
        return getattr(module, name)
    raise AttributeError(name)
