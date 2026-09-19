# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exact attention helpers for fused block-denoising block/context parallelism.

The public block diffusion integration computes the standard all-block objective. This
module provides both a replicated-K/V local attention primitive and the true
ring context-parallel primitive with a single logical KV cache owned by the
fused context/block-parallel group.

The core kernel streams over K/V shards and maintains the usual online softmax
statistics, so it never materializes the full ``[batch, heads, query, key]``
score tensor. The distributed ring entry point is the only context-parallel
attention path: it circulates K/V shards through the fused context/block-parallel
group and reduces shard gradients back to their owner rank in the custom
backward pass.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

from dllm_parallel.core.kernels import cp_fusion
from dllm_parallel.core.attention.cp_backend import (
    CPBPCounters as CPBPCounters,
    all_clean_shards as all_clean_shards,
    clean_shards_for_rank as clean_shards_for_rank,
    cp_bp_counters_from_visits as cp_bp_counters_from_visits,
    merge_attention_stats as merge_attention_stats,
    owner_lengths,
    prefix_visits_for_noisy_blocks as prefix_visits_for_noisy_blocks,
    ring_owner_traversal as ring_owner_traversal,
)
from dllm_parallel.core.attention.flex import (
    dense_attention_backward_from_state,
    flex_attention_backward_from_state,
)
from dllm_parallel.core.attention.fa4 import (
    fa4_backward_sparse_tile,
    fa4_forward,
    fa4_forward_sparse_tile,
)
from dllm_parallel.core.attention.layout import (
    WindowedCleanExchangePlan,
    active_query_indices_for_context_rank,
    all_context_parallel_sequence_intervals,
    cat_intervals as _cat_intervals,
    clean_intervals_for_runtime as _clean_intervals_for_runtime,
    shard_bounds as shard_bounds,
    windowed_clean_exchange_plan,
    write_interval_grads as _write_interval_grads,
)
from dllm_parallel.core.attention.masks import (
    BlockDenoisingFullMask,
    BlockDenoisingGlobalCleanMask,
    BlockDenoisingLocalActiveMask,
    BlockDenoisingPackedKeyMask,
)
from dllm_parallel.core.attention.policy import cp_bp_policy as _cp_bp_policy
from dllm_parallel.core.attention.ring_transport import (
    _make_kv_ring_payload,
    _ring_exchange_flat_async,
    _ring_exchange_flat_wait,
    _ring_exchange_kv_payload_async,
    _ring_exchange_kv_payload_wait,
    _ring_return_owner_grads_p2p_flat,
)
from dllm_parallel.core.attention.wide_head_attention import (
    is_wide_head_dim as _is_wide_head_dim,
    prepare_wide_bdlm_interval_backward_bshd as _prepare_wide_interval_backward_bshd,
    uses_native_wide_attention as _uses_native_wide_attention,
    wide_bdlm_interval_backward_from_state_bhsd as _wide_interval_backward_from_state_bhsd,
    wide_bdlm_interval_forward_bhsd as _wide_interval_forward_bhsd,
    wide_bdlm_metadata_backward_bhsd as _wide_metadata_backward_bhsd,
    wide_bdlm_metadata_forward_bhsd as _wide_metadata_forward_bhsd,
    wide_bdlm_interval_plan as _build_wide_interval_plan,
    wide_bdlm_metadata_plan as _build_wide_metadata_plan,
    wide_full_attention_bshd as _wide_full_attention_bshd,
    wide_full_backward_bshd as _wide_full_backward_bshd,
)
from dllm_parallel.core.profiling.operator_trace import communication_scope

try:
    from torch.nn.attention.flex_attention import (
        AuxRequest,
        BlockMask,
        create_block_mask,
        flex_attention,
    )
except Exception:  # pragma: no cover - depends on the installed Torch build.
    AuxRequest = None  # type: ignore[assignment,misc]
    BlockMask = None  # type: ignore[assignment,misc]
    create_block_mask = None  # type: ignore[assignment]
    flex_attention = None  # type: ignore[assignment]


_compiled_flex_attention: dict[tuple[str, bool], Any] = {}
_compiled_create_block_mask: Any | None = None
_debug_backward_call_count = 0
_FA4_D256_Q32_MAX_KEY_LENGTH = 544


def _explicit_bdlm_flex_mask(
    batch_index: torch.Tensor,
    head_index: torch.Tensor,
    query_index: torch.Tensor,
    key_index: torch.Tensor,
    query_clean_starts: torch.Tensor,
    query_clean_stops: torch.Tensor,
    local_query_blocks: torch.Tensor,
    query_is_clean: torch.Tensor,
    key_coordinates: torch.Tensor,
    key_is_clean: torch.Tensor,
) -> torch.Tensor:
    del batch_index, head_index
    clean_start = query_clean_starts[query_index]
    clean_stop = query_clean_stops[query_index]
    local_query_block = local_query_blocks[query_index]
    clean_query = query_is_clean[query_index]
    key_coordinate = key_coordinates[key_index]
    clean_key = key_is_clean[key_index]
    active_to_active = (
        (~clean_query) & (~clean_key) & (local_query_block == key_coordinate)
    )
    clean_context = (
        clean_key & (key_coordinate >= clean_start) & (key_coordinate < clean_stop)
    )
    return active_to_active | clean_context


def _sync_cuda_stream_if_needed(tensor: torch.Tensor) -> None:
    if tensor.is_cuda:
        torch.cuda.current_stream(tensor.device).synchronize()


def streaming_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
    scale: float | None = None,
    key_chunk_size: int | None = None,
) -> torch.Tensor:
    """Compute exact SDPA by streaming over key/value chunks."""

    _validate_bhsd(query, key, value)
    q_len = query.shape[-2]
    kv_len = key.shape[-2]
    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])
    if not key_chunk_size or key_chunk_size <= 0:
        key_chunk_size = kv_len

    query_f = query.float()
    value_f = value.float()
    m = torch.full(
        query.shape[:-1],
        -torch.inf,
        dtype=torch.float32,
        device=query.device,
    )
    l = torch.zeros_like(m)
    numerator = torch.zeros_like(query_f)

    q_positions = torch.arange(q_len, device=query.device)
    for start in range(0, kv_len, key_chunk_size):
        stop = min(start + key_chunk_size, kv_len)
        key_chunk = key[..., start:stop, :].float()
        value_chunk = value_f[..., start:stop, :]
        scores = torch.matmul(query_f, key_chunk.transpose(-2, -1)) * scale
        scores = _apply_masks(
            scores=scores,
            attn_mask=attn_mask,
            is_causal=is_causal,
            q_positions=q_positions,
            key_start=start,
            key_stop=stop,
        )

        local_m = torch.max(scores, dim=-1).values
        local_m_safe = torch.where(
            torch.isfinite(local_m),
            local_m,
            torch.zeros_like(local_m),
        )
        exp_scores = torch.exp(scores - local_m_safe.unsqueeze(-1))
        exp_scores = torch.where(
            torch.isfinite(scores),
            exp_scores,
            torch.zeros_like(exp_scores),
        )
        local_l = exp_scores.sum(dim=-1)
        local_numerator = torch.matmul(exp_scores, value_chunk)

        next_m = torch.maximum(m, local_m)
        next_m_safe = torch.where(
            torch.isfinite(next_m),
            next_m,
            torch.zeros_like(next_m),
        )
        old_weight = torch.where(
            torch.isfinite(m),
            torch.exp(m - next_m_safe),
            torch.zeros_like(m),
        )
        new_weight = torch.where(
            torch.isfinite(local_m),
            torch.exp(local_m - next_m_safe),
            torch.zeros_like(local_m),
        )

        numerator = (
            old_weight.unsqueeze(-1) * numerator
            + new_weight.unsqueeze(-1) * local_numerator
        )
        l = old_weight * l + new_weight * local_l
        m = next_m

    tiny = torch.finfo(torch.float32).tiny
    output = numerator.div_(l.clamp_min(tiny).unsqueeze(-1))
    output.masked_fill_(l.unsqueeze(-1) <= 0, 0)
    return output.to(dtype=query.dtype)


def replicated_kv_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
    scale: float | None = None,
    key_chunk_size: int | None = None,
    backend: str = "sdpa",
) -> torch.Tensor:
    """Exact local attention when every rank already has full K/V.

    This is deliberately not context-parallel attention: K/V is replicated on
    the rank, so there is no K/V memory sharding across the context dimension.
    """

    if backend not in {"sdpa", "streaming"}:
        raise ValueError("backend must be 'sdpa' or 'streaming'")
    if backend == "streaming":
        return streaming_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            is_causal=is_causal,
            scale=scale,
            key_chunk_size=key_chunk_size,
        )
    return F.scaled_dot_product_attention(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=is_causal,
        scale=scale,
    )


def ring_context_parallel_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
    scale: float | None = None,
    runtime: Any | None = None,
    key_chunk_size: int | None = None,
) -> torch.Tensor:
    """Exact context-parallel attention over ring-sharded K/V."""

    if not _should_use_context_parallel(runtime):
        raise RuntimeError(
            "ring_context_parallel_attention requires an initialized multi-rank "
            "CUDA context-parallel runtime"
        )
    return _RingContextParallelAttention.apply(
        query,
        key,
        value,
        attn_mask,
        bool(is_causal),
        scale,
        runtime,
        key_chunk_size,
        True,
        int(key.shape[-2]),
        False,
    )


def context_parallel_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
    scale: float | None = None,
    runtime: Any | None = None,
    key_chunk_size: int | None = None,
    shard_kv: bool = True,
    kv_backend: str = "ring",
) -> torch.Tensor:
    """Run the production ring-sharded context-parallel attention path."""

    if kv_backend != "ring" or not shard_kv:
        raise ValueError(
            "context_parallel_attention only supports ring-sharded K/V; "
            "use replicated_kv_attention for replicated K/V"
        )
    return ring_context_parallel_attention(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        is_causal=is_causal,
        scale=scale,
        runtime=runtime,
        key_chunk_size=key_chunk_size,
    )


def ring_attention_with_local_kv(
    query: torch.Tensor,
    local_key: torch.Tensor,
    local_value: torch.Tensor,
    global_key: torch.Tensor,
    global_value: torch.Tensor,
    *,
    local_attn_mask: torch.Tensor | None = None,
    global_attn_mask: torch.Tensor | None = None,
    scale: float | None = None,
    runtime: Any | None = None,
    key_chunk_size: int | None = None,
) -> torch.Tensor:
    """Attention over local-only K/V plus ring-sharded global K/V.

    This is the packed BP primitive: noisy active-block K/V is rank-local, while
    the clean-prefix K/V cache is shared as one logical ring-sharded sequence.
    """

    _validate_bhsd(query, local_key, local_value)
    _validate_bhsd(query, global_key, global_value)
    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])
    if not _should_use_context_parallel(runtime):
        raise RuntimeError(
            "ring_attention_with_local_kv requires an initialized multi-rank "
            "CUDA context-parallel runtime"
        )
    return _RingWithLocalKVAttention.apply(
        query,
        local_key,
        local_value,
        global_key,
        global_value,
        local_attn_mask,
        global_attn_mask,
        float(scale),
        runtime,
        key_chunk_size,
        int(global_key.shape[-2]),
        False,
        None,
    )


def ring_attention_with_local_kv_shard(
    query: torch.Tensor,
    local_key: torch.Tensor,
    local_value: torch.Tensor,
    global_key_shard: torch.Tensor,
    global_value_shard: torch.Tensor,
    *,
    global_seq_len: int,
    local_attn_mask: torch.Tensor | None = None,
    global_attn_mask: torch.Tensor | None = None,
    scale: float | None = None,
    runtime: Any | None = None,
    key_chunk_size: int | None = None,
) -> torch.Tensor:
    """Attention over local-only K/V plus a local shard of global K/V.

    Unlike ``ring_attention_with_local_kv``, this entry point does not accept or
    retain a replicated global K/V tensor. The caller owns only its clean-prefix
    shard, which is circulated through the ring and receives the reduced shard
    gradient directly.
    """

    _validate_bhsd(query, local_key, local_value)
    _validate_bhsd(query, global_key_shard, global_value_shard)
    if global_seq_len < 0:
        raise ValueError("global_seq_len must be non-negative")
    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])
    if not _should_use_context_parallel(runtime):
        raise RuntimeError(
            "ring_attention_with_local_kv_shard requires an initialized multi-rank "
            "CUDA context-parallel runtime"
        )
    return _RingWithLocalKVAttention.apply(
        query,
        local_key,
        local_value,
        global_key_shard,
        global_value_shard,
        local_attn_mask,
        global_attn_mask,
        float(scale),
        runtime,
        key_chunk_size,
        int(global_seq_len),
        True,
        None,
    )


def fused_block_context_attention_bshd(
    query: torch.Tensor,
    local_key: torch.Tensor,
    local_value: torch.Tensor,
    global_key_shard: torch.Tensor,
    global_value_shard: torch.Tensor,
    *,
    global_seq_len: int,
    local_attn_mask: torch.Tensor | None = None,
    global_attn_mask: torch.Tensor | None = None,
    scale: float | None = None,
    runtime: Any | None = None,
    key_chunk_size: int | None = None,
) -> torch.Tensor:
    """Production fused BP+CP attention over local blocks and sharded clean K/V."""

    del key_chunk_size
    _validate_bshd(query, local_key, local_value)
    _validate_bshd(query, global_key_shard, global_value_shard)
    if global_seq_len < 0:
        raise ValueError("global_seq_len must be non-negative")
    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])
    if not _should_use_context_parallel(runtime):
        raise RuntimeError(
            "fused_block_context_attention_bshd requires an initialized "
            "multi-rank CUDA context-parallel runtime"
        )
    if not isinstance(global_attn_mask, BlockDenoisingGlobalCleanMask):
        raise RuntimeError("fused BP/CP requires a global clean BDLM mask")
    if global_attn_mask.clean_context_window is not None:
        attention = _WindowedCollectiveWithLocalKVAttentionBSHD
    elif _cp_bp_policy(runtime).clean_kv_transport == "streaming":
        attention = _RingWithLocalKVAttentionBSHD
    else:
        attention = _CollectiveWithLocalKVAttentionBSHD
    return attention.apply(
        query,
        local_key,
        local_value,
        global_key_shard,
        global_value_shard,
        local_attn_mask,
        global_attn_mask,
        float(scale),
        runtime,
        int(global_seq_len),
    )


def pure_context_block_denoising_attention_bshd(
    query: torch.Tensor,
    local_key: torch.Tensor,
    local_value: torch.Tensor,
    global_key_shard: torch.Tensor,
    global_value_shard: torch.Tensor,
    *,
    global_seq_len: int,
    local_attn_mask: torch.Tensor | None = None,
    global_attn_mask: torch.Tensor | None = None,
    scale: float | None = None,
    runtime: Any | None = None,
) -> torch.Tensor:
    """Pure CP with a token-offset query shard inside every active block."""

    _validate_bshd(query, local_key, local_value)
    _validate_bshd(query, global_key_shard, global_value_shard)
    if not _should_use_context_parallel(runtime):
        raise RuntimeError(
            "pure_context_block_denoising_attention_bshd requires an initialized "
            "multi-rank CUDA context-parallel runtime"
        )
    if int(getattr(runtime, "block_parallel_size", 1) or 1) != 1:
        raise RuntimeError("pure context query sharding requires block_parallel_size=1")
    if not isinstance(local_attn_mask, BlockDenoisingLocalActiveMask):
        raise RuntimeError("pure context query sharding requires a local active mask")
    if not isinstance(global_attn_mask, BlockDenoisingGlobalCleanMask):
        raise RuntimeError("pure context query sharding requires a global clean mask")
    resolved_scale = scale if scale is not None else 1.0 / math.sqrt(query.shape[-1])
    if global_attn_mask.clean_context_window is not None:
        return _WindowedCollectiveWithLocalKVAttentionBSHD.apply(
            query,
            local_key,
            local_value,
            global_key_shard,
            global_value_shard,
            local_attn_mask,
            global_attn_mask,
            float(resolved_scale),
            runtime,
            int(global_seq_len),
        )
    return _PureContextShardedActiveAttentionBSHD.apply(
        query,
        local_key,
        local_value,
        global_key_shard,
        global_value_shard,
        local_attn_mask,
        global_attn_mask,
        float(resolved_scale),
        runtime,
        int(global_seq_len),
    )


def pure_context_persistent_block_denoising_attention_bshd(
    query: torch.Tensor,
    active_key_shard: torch.Tensor,
    active_value_shard: torch.Tensor,
    global_key_shard: torch.Tensor,
    global_value_shard: torch.Tensor,
    *,
    global_seq_len: int,
    local_attn_mask: torch.Tensor | None = None,
    global_attn_mask: torch.Tensor | None = None,
    scale: float | None = None,
    runtime: Any | None = None,
) -> torch.Tensor:
    """Pure CP attention whose active hidden-state rows remain CP sharded."""

    _validate_bshd(query, active_key_shard, active_value_shard)
    _validate_bshd(query, global_key_shard, global_value_shard)
    if not _should_use_context_parallel(runtime):
        raise RuntimeError("persistent pure CP requires an initialized CP runtime")
    if int(getattr(runtime, "block_parallel_size", 1) or 1) != 1:
        raise RuntimeError("persistent pure CP requires block_parallel_size=1")
    if not isinstance(local_attn_mask, BlockDenoisingLocalActiveMask):
        raise RuntimeError("persistent pure CP requires a local active mask")
    if not isinstance(global_attn_mask, BlockDenoisingGlobalCleanMask):
        raise RuntimeError("persistent pure CP requires a global clean mask")
    # ``active_blocks`` is token metadata: it contains one block id for every
    # active K/V row, not one entry per block.
    full_active_len = int(local_attn_mask.active_blocks.numel())
    if full_active_len % int(local_attn_mask.block_size):
        raise ValueError("persistent pure-CP active metadata is not block aligned")
    if full_active_len == 0:
        full_active_key, full_active_value = active_key_shard, active_value_shard
    else:
        full_active_key, full_active_value = _GatherPureCPActiveKV.apply(
            active_key_shard,
            active_value_shard,
            int(full_active_len),
            int(local_attn_mask.block_size),
            runtime.context_block_parallel_group,
            int(runtime.context_parallel_rank),
        )
    resolved_scale = scale if scale is not None else 1.0 / math.sqrt(query.shape[-1])
    return _PersistentPureContextShardedActiveAttentionBSHD.apply(
        query,
        full_active_key,
        full_active_value,
        global_key_shard,
        global_value_shard,
        local_attn_mask,
        global_attn_mask,
        float(resolved_scale),
        runtime,
        int(global_seq_len),
    )


def _windowed_exchange_tensors(
    plan: WindowedCleanExchangePlan,
    *,
    rank: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    remote_send_intervals = tuple(
        interval
        for destination, intervals in enumerate(plan.send_local_intervals_by_rank)
        if destination != int(rank)
        for interval in intervals
    )
    send_indices = (
        torch.cat(
            [
                torch.arange(start, stop, device=device, dtype=torch.long)
                for start, stop in remote_send_intervals
            ],
            dim=0,
        )
        if remote_send_intervals
        else torch.empty(0, device=device, dtype=torch.long)
    )
    local_receive_intervals = plan.receive_logical_intervals_by_rank[int(rank)]
    local_receive_positions = (
        torch.cat(
            [
                torch.arange(start, stop, device=device, dtype=torch.int32)
                for start, stop in local_receive_intervals
            ],
            dim=0,
        )
        if local_receive_intervals
        else torch.empty(0, device=device, dtype=torch.int32)
    )
    remote_receive_intervals = tuple(
        interval
        for source, intervals in enumerate(plan.receive_logical_intervals_by_rank)
        if source != int(rank)
        for interval in intervals
    )
    remote_receive_positions = (
        torch.cat(
            [
                torch.arange(start, stop, device=device, dtype=torch.int32)
                for start, stop in remote_receive_intervals
            ],
            dim=0,
        )
        if remote_receive_intervals
        else torch.empty(0, device=device, dtype=torch.int32)
    )
    remote_send_tokens = int(plan.send_tokens) - int(plan.send_counts[int(rank)])
    remote_receive_tokens = int(plan.receive_tokens) - int(
        plan.receive_counts[int(rank)]
    )
    if int(send_indices.numel()) != remote_send_tokens:
        raise RuntimeError("windowed remote send plan has inconsistent token counts")
    if int(remote_receive_positions.numel()) != remote_receive_tokens:
        raise RuntimeError("windowed remote receive plan has inconsistent token counts")
    if int(local_receive_positions.numel()) != int(plan.receive_counts[int(rank)]):
        raise RuntimeError("windowed local receive plan has inconsistent token counts")
    return send_indices, local_receive_positions, remote_receive_positions


def _windowed_fused_mask(
    global_attn_mask: BlockDenoisingGlobalCleanMask,
    local_attn_mask: BlockDenoisingLocalActiveMask,
    *,
    receive_positions: torch.Tensor,
) -> BlockDenoisingPackedKeyMask:
    cache_key = (
        "windowed_fused_mask",
        int(receive_positions.numel()),
        receive_positions.device.type,
        receive_positions.device.index,
    )
    cached = local_attn_mask.flex_cache.get(cache_key)
    if isinstance(cached, BlockDenoisingPackedKeyMask):
        return cached
    mask = BlockDenoisingPackedKeyMask(
        query_blocks=global_attn_mask.query_blocks,
        local_query_blocks=local_attn_mask.query_blocks,
        query_is_clean=global_attn_mask.query_is_clean,
        active_key_blocks=local_attn_mask.active_blocks,
        clean_key_blocks=receive_positions // int(global_attn_mask.block_size),
        block_size=int(global_attn_mask.block_size),
        query_clean_bounds=global_attn_mask.query_clean_bounds,
        clean_key_positions=receive_positions,
        backward_query_chunk_size=int(global_attn_mask.backward_query_chunk_size),
        debug_nonfinite_attention=bool(global_attn_mask.debug_nonfinite_attention),
    )
    local_attn_mask.flex_cache[cache_key] = mask
    return mask


def _all_gather_clean_kv_bshd(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    max_shard_len: int,
    group: Any,
    world_size: int,
) -> tuple[Any, torch.Tensor, torch.Tensor]:
    """Start one gather of padded clean K/V shards."""

    local = (
        torch.stack(
            (
                _pad_kv_shard_bshd(key, int(max_shard_len)),
                _pad_kv_shard_bshd(value, int(max_shard_len)),
            ),
            dim=2,
        )
        .permute(1, 0, 2, 3, 4)
        .contiguous()
    )
    gathered = torch.empty(
        (int(world_size) * int(max_shard_len), *local.shape[1:]),
        device=local.device,
        dtype=local.dtype,
    )
    input_bytes = int(local.numel()) * int(local.element_size())
    with communication_scope(
        domain="attention",
        phase="forward",
        collective="all_gather_into_tensor",
        input_bytes=input_bytes,
        logical_bytes=input_bytes * (int(world_size) - 1),
    ):
        work = dist.all_gather_into_tensor(
            gathered,
            local,
            group=group,
            async_op=True,
        )
    return work, local, gathered


def _owner_major_clean_kv_bshd(
    gathered: torch.Tensor,
    *,
    world_size: int,
    max_shard_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten gathered clean K/V without copying or changing owner order."""

    expected_tokens = int(world_size) * int(max_shard_len)
    if int(gathered.shape[0]) != expected_tokens or int(gathered.shape[2]) != 2:
        raise ValueError("gathered clean K/V layout does not match its ownership")
    owner_key = gathered[:, :, 0].permute(1, 0, 2, 3)
    owner_value = gathered[:, :, 1].permute(1, 0, 2, 3)
    return owner_key, owner_value


def _reduce_scatter_clean_kv_grads_bshd(
    grad_key: torch.Tensor,
    grad_value: torch.Tensor,
    *,
    max_shard_len: int,
    group: Any,
    world_size: int,
) -> tuple[Any, torch.Tensor, torch.Tensor]:
    """Start owner reduction of padded clean K/V gradients."""

    batch = int(grad_key.shape[0])
    key_heads = int(grad_key.shape[2])
    head_dim = int(grad_key.shape[3])
    owner_shape = (
        batch,
        int(world_size),
        int(max_shard_len),
        key_heads,
        head_dim,
    )
    packed = torch.empty(
        (
            int(world_size),
            2,
            batch,
            int(max_shard_len),
            key_heads,
            head_dim,
        ),
        device=grad_key.device,
        dtype=grad_key.dtype,
    )
    packed[:, 0].copy_(grad_key.view(owner_shape).permute(1, 0, 2, 3, 4))
    packed[:, 1].copy_(grad_value.view(owner_shape).permute(1, 0, 2, 3, 4))
    reduced = torch.empty(
        (2, batch, int(max_shard_len), key_heads, head_dim),
        device=packed.device,
        dtype=packed.dtype,
    )
    collective_input = packed.view(
        int(world_size) * 2, batch, int(max_shard_len), key_heads, head_dim
    )
    input_bytes = int(collective_input.numel()) * int(collective_input.element_size())
    with communication_scope(
        domain="attention",
        phase="backward",
        collective="reduce_scatter_tensor",
        input_bytes=input_bytes,
        logical_bytes=input_bytes * (int(world_size) - 1) // int(world_size),
    ):
        work = dist.reduce_scatter_tensor(
            reduced,
            collective_input,
            group=group,
            async_op=True,
        )
    # The input must remain alive until the asynchronous collective completes.
    return work, packed, reduced


def _owner_major_clean_grad_storage_bshd(
    clean_key: torch.Tensor,
    clean_value: torch.Tensor,
    *,
    max_shard_len: int,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate one reduce-scatter-ready buffer for clean K/V gradients."""

    batch = int(clean_key.shape[0])
    clean_tokens = int(world_size) * int(max_shard_len)
    key_heads = int(clean_key.shape[2])
    head_dim = int(clean_key.shape[3])
    if tuple(clean_key.shape) != (batch, clean_tokens, key_heads, head_dim):
        raise ValueError("clean key layout does not match its owner-major shards")
    if clean_value.shape != clean_key.shape:
        raise ValueError("clean key and value layouts must match")

    packed = torch.empty(
        (clean_tokens, batch, 2, key_heads, head_dim),
        device=clean_key.device,
        dtype=clean_key.dtype,
    )
    grad_key = packed[:, :, 0].permute(1, 0, 2, 3)
    grad_value = packed[:, :, 1].permute(1, 0, 2, 3)
    if grad_key.stride() != clean_key.stride():
        raise ValueError("clean key does not use the production owner-major layout")
    if grad_value.stride() != clean_value.stride():
        raise ValueError("clean value does not use the production owner-major layout")
    return packed, grad_key, grad_value


def _reduce_scatter_owner_major_clean_grad_storage_bshd(
    packed: torch.Tensor,
    *,
    max_shard_len: int,
    group: Any,
    world_size: int,
) -> tuple[Any, torch.Tensor]:
    """Reduce clean K/V gradients directly from their backward output storage."""

    clean_tokens, batch, kv_components, key_heads, head_dim = packed.shape
    if int(kv_components) != 2:
        raise ValueError("clean gradient storage must contain K and V")
    if int(clean_tokens) != int(world_size) * int(max_shard_len):
        raise ValueError("clean gradient storage does not match its ownership")
    reduced = torch.empty(
        (int(max_shard_len), batch, 2, key_heads, head_dim),
        device=packed.device,
        dtype=packed.dtype,
    )
    input_bytes = int(packed.numel()) * int(packed.element_size())
    with communication_scope(
        domain="attention",
        phase="backward",
        collective="reduce_scatter_tensor",
        input_bytes=input_bytes,
        logical_bytes=input_bytes * (int(world_size) - 1) // int(world_size),
    ):
        work = dist.reduce_scatter_tensor(
            reduced,
            packed,
            group=group,
            async_op=True,
        )
    return work, reduced


def _collective_clean_indices(
    global_attn_mask: BlockDenoisingGlobalCleanMask | BlockDenoisingFullMask,
    *,
    intervals_by_owner: tuple[tuple[tuple[int, int], ...], ...],
    max_shard_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map owner-major padded clean K/V to and from logical token order."""

    cache_key = (
        "collective_clean_indices",
        intervals_by_owner,
        int(max_shard_len),
        device.type,
        device.index,
    )
    cached = global_attn_mask.flex_cache.get(cache_key)
    if isinstance(cached, tuple):
        return cached

    global_seq_len = max(
        (int(stop) for intervals in intervals_by_owner for _, stop in intervals),
        default=0,
    )
    owner_to_logical = [global_seq_len] * (len(intervals_by_owner) * int(max_shard_len))
    logical_from_owner = [0] * global_seq_len
    for owner, intervals in enumerate(intervals_by_owner):
        offset = 0
        for start, stop in intervals:
            length = int(stop) - int(start)
            if length <= 0:
                continue
            for logical_position in range(int(start), int(stop)):
                owner_position = owner * int(max_shard_len) + offset
                owner_to_logical[owner_position] = logical_position
                logical_from_owner[logical_position] = owner_position
                offset += 1
    result = (
        torch.tensor(owner_to_logical, device=device, dtype=torch.long),
        torch.tensor(logical_from_owner, device=device, dtype=torch.long),
    )
    global_attn_mask.flex_cache[cache_key] = result
    return result


def _collective_owner_major_clean_mask(
    global_attn_mask: BlockDenoisingGlobalCleanMask,
    *,
    owner_to_logical: torch.Tensor,
    allow_native_wide_attention: bool = True,
) -> BlockDenoisingGlobalCleanMask:
    """Describe owner-major clean K/V directly, including padded rows."""

    cache_key = (
        "collective_owner_major_clean_mask",
        int(owner_to_logical.numel()),
        bool(allow_native_wide_attention),
        owner_to_logical.device.type,
        owner_to_logical.device.index,
    )
    cached = global_attn_mask.flex_cache.get(cache_key)
    if isinstance(cached, BlockDenoisingGlobalCleanMask):
        return cached
    owner_major_mask = BlockDenoisingGlobalCleanMask(
        query_blocks=global_attn_mask.query_blocks,
        query_is_clean=global_attn_mask.query_is_clean,
        block_size=int(global_attn_mask.block_size),
        clean_context_window=global_attn_mask.clean_context_window,
        clean_key_blocks=(owner_to_logical // int(global_attn_mask.block_size)).to(
            dtype=torch.int32
        ),
        query_clean_bounds=global_attn_mask.query_clean_bounds,
        clean_key_positions=(
            owner_to_logical.to(dtype=torch.int32)
            if global_attn_mask.query_clean_bounds is not None
            else None
        ),
        backward_query_chunk_size=int(global_attn_mask.backward_query_chunk_size),
        debug_nonfinite_attention=bool(global_attn_mask.debug_nonfinite_attention),
        allow_native_wide_attention=bool(allow_native_wide_attention),
        flex_cache=global_attn_mask.flex_cache,
    )
    global_attn_mask.flex_cache[cache_key] = owner_major_mask
    return owner_major_mask


def _collective_clean_uses_native_wide_attention(
    query: torch.Tensor,
    *,
    active_len: int,
) -> bool:
    """Allow verified native D=512 attention for clean-only collectives."""

    if int(active_len) < 0:
        raise ValueError("active_len must be nonnegative")
    return _uses_native_wide_attention(query)


def _complete_collective_clean_gather_for_native_attention(
    clean_key: torch.Tensor,
    *,
    active_len: int,
    native_clean_attention: bool,
) -> None:
    """Make a clean-only gathered buffer ready for the external AOT launch."""

    if int(active_len) == 0 and bool(native_clean_attention):
        _sync_cuda_stream_if_needed(clean_key)


def _collective_packed_key_mask(
    clean_mask: BlockDenoisingGlobalCleanMask,
    local_attn_mask: BlockDenoisingLocalActiveMask,
    *,
    local_query_blocks: torch.Tensor | None = None,
) -> BlockDenoisingPackedKeyMask:
    """Describe ``[active K/V; owner-major clean K/V]`` for one exact backward.

    The metadata only expresses the BDLM mask already available to either
    execution mode. Pure CP supplies its token-offset query shard here and does
    not receive block ownership or scheduling metadata.
    """

    clean_key_blocks = clean_mask.clean_key_blocks
    if clean_key_blocks is None:
        raise RuntimeError("collective packed attention requires clean key blocks")
    resolved_local_query_blocks = (
        local_attn_mask.query_blocks
        if local_query_blocks is None
        else local_query_blocks
    )
    if resolved_local_query_blocks.shape != clean_mask.query_blocks.shape:
        raise ValueError("packed local query metadata must match clean queries")
    return BlockDenoisingPackedKeyMask(
        query_blocks=clean_mask.query_blocks,
        local_query_blocks=resolved_local_query_blocks,
        query_is_clean=clean_mask.query_is_clean,
        active_key_blocks=local_attn_mask.active_blocks,
        clean_key_blocks=clean_key_blocks,
        block_size=int(clean_mask.block_size),
        query_clean_bounds=clean_mask.query_clean_bounds,
        clean_key_positions=clean_mask.clean_key_positions,
        backward_query_chunk_size=int(clean_mask.backward_query_chunk_size),
        debug_nonfinite_attention=bool(clean_mask.debug_nonfinite_attention),
        flex_cache=clean_mask.flex_cache,
    )


def _pack_local_clean_kv_bshd(
    local_key: torch.Tensor,
    local_value: torch.Tensor,
    clean_key: torch.Tensor,
    clean_value: torch.Tensor,
    *,
    kv_head_repeat: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack active and clean K/V into the interleaved layout used by FA4."""

    _validate_bshd(local_key, local_key, local_value)
    _validate_bshd(clean_key, clean_key, clean_value)
    if (
        int(local_key.shape[0]) != int(clean_key.shape[0])
        or tuple(local_key.shape[2:]) != tuple(clean_key.shape[2:])
        or local_key.dtype != clean_key.dtype
        or local_key.device != clean_key.device
    ):
        raise ValueError("local and clean K/V must share batch, head, and dtype layout")
    kv_head_repeat = int(kv_head_repeat)
    if kv_head_repeat <= 0:
        raise ValueError("kv_head_repeat must be positive")
    active_len = int(local_key.shape[1])
    clean_len = int(clean_key.shape[1])
    batch = int(local_key.shape[0])
    key_heads = int(local_key.shape[2]) * kv_head_repeat
    head_dim = int(local_key.shape[3])
    packed = torch.empty(
        (active_len + clean_len, batch, 2, key_heads, head_dim),
        device=local_key.device,
        dtype=local_key.dtype,
    )
    packed[:active_len, :, 0].copy_(
        local_key.repeat_interleave(kv_head_repeat, dim=2).permute(1, 0, 2, 3)
    )
    packed[:active_len, :, 1].copy_(
        local_value.repeat_interleave(kv_head_repeat, dim=2).permute(1, 0, 2, 3)
    )
    packed[active_len:, :, 0].copy_(
        clean_key.repeat_interleave(kv_head_repeat, dim=2).permute(1, 0, 2, 3)
    )
    packed[active_len:, :, 1].copy_(
        clean_value.repeat_interleave(kv_head_repeat, dim=2).permute(1, 0, 2, 3)
    )
    return (
        packed,
        packed[:, :, 0].permute(1, 0, 2, 3),
        packed[:, :, 1].permute(1, 0, 2, 3),
    )


def _collapse_repeated_kv_grad_storage_bshd(
    grad_storage: torch.Tensor,
    *,
    original_key_heads: int,
    kv_head_repeat: int,
) -> torch.Tensor:
    """Sum gradients from kernel-only duplicate KV heads."""

    original_key_heads = int(original_key_heads)
    kv_head_repeat = int(kv_head_repeat)
    if kv_head_repeat == 1:
        return grad_storage
    expected_heads = original_key_heads * kv_head_repeat
    if int(grad_storage.shape[3]) != expected_heads:
        raise ValueError("expanded KV gradient head count does not match repeat factor")
    return grad_storage.unflatten(
        3,
        (original_key_heads, kv_head_repeat),
    ).sum(dim=4)


def _collective_packed_d256_kv_head_repeat(
    query: torch.Tensor,
    key: torch.Tensor,
) -> int | None:
    """Choose an exact KV expansion that fits FA4's D=256 Pack-GQA tile."""

    query_heads = int(query.shape[2])
    key_heads = int(key.shape[2])
    if (
        int(query.shape[-1]) != 256
        or query_heads <= key_heads
        or query_heads % key_heads
    ):
        return None
    gqa_ratio = query_heads // key_heads
    kernel_gqa_ratio = math.gcd(gqa_ratio, 64)
    if kernel_gqa_ratio <= 1:
        return None
    return gqa_ratio // kernel_gqa_ratio


def _native_fa4_d256_forward_kv_head_repeat(
    query: torch.Tensor,
    key: torch.Tensor,
) -> int:
    """Choose an exact KV expansion for D=256 FA4 forward tile alignment."""

    if int(query.shape[-1]) != 256 or int(key.shape[-1]) != 256:
        return 1
    query_heads = int(query.shape[1])
    key_heads = int(key.shape[1])
    if query_heads <= key_heads or query_heads % key_heads:
        return 1
    gqa_ratio = query_heads // key_heads
    kernel_gqa_ratio = math.gcd(gqa_ratio, 128)
    if kernel_gqa_ratio <= 1:
        return 1
    return gqa_ratio // kernel_gqa_ratio


def _uses_collective_packed_d256_backward(
    query: torch.Tensor,
    key: torch.Tensor,
) -> bool:
    """Use the native packed-GQA route for D=256 collective attention."""

    return _collective_packed_d256_kv_head_repeat(query, key) is not None


def _owner_major_clean_grads_bshd(
    grad_key: torch.Tensor,
    grad_value: torch.Tensor,
    *,
    owner_to_logical: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Restore owner-major padding before clean-gradient reduce-scatter."""

    zero_key = torch.zeros_like(grad_key[:, :1])
    zero_value = torch.zeros_like(grad_value[:, :1])
    padded_key = torch.cat((grad_key, zero_key), dim=1)
    padded_value = torch.cat((grad_value, zero_value), dim=1)
    return (
        padded_key.index_select(1, owner_to_logical),
        padded_value.index_select(1, owner_to_logical),
    )


class _WindowedCollectiveWithLocalKVAttentionBSHD(torch.autograd.Function):
    """Fused BP/CP attention with exact variable-size sliding-window exchange."""

    @staticmethod
    def forward(
        ctx: Any,
        query: torch.Tensor,
        local_key: torch.Tensor,
        local_value: torch.Tensor,
        global_key_shard: torch.Tensor,
        global_value_shard: torch.Tensor,
        local_attn_mask: BlockDenoisingLocalActiveMask,
        global_attn_mask: BlockDenoisingGlobalCleanMask,
        scale: float,
        runtime: Any,
        global_seq_len: int,
    ) -> torch.Tensor:
        _validate_bshd(query, local_key, local_value)
        _validate_bshd(query, global_key_shard, global_value_shard)
        if not isinstance(local_attn_mask, BlockDenoisingLocalActiveMask):
            raise RuntimeError("windowed fused BP/CP requires a local active mask")
        if not isinstance(global_attn_mask, BlockDenoisingGlobalCleanMask):
            raise RuntimeError("windowed fused BP/CP requires a clean mask")
        window = global_attn_mask.clean_context_window
        if window is None or global_attn_mask.query_clean_bounds is None:
            raise RuntimeError("windowed fused BP/CP requires exact clean windows")

        rank = int(runtime.context_parallel_rank)
        world_size = len(runtime.context_block_parallel_group_ranks)
        block_parallel_size = int(runtime.block_parallel_size)
        block_parallel_rank = int(runtime.block_parallel_rank)
        if (
            world_size <= 1
            or block_parallel_size < world_size
            or block_parallel_size % world_size
            or block_parallel_rank % world_size != rank
        ):
            raise RuntimeError(
                "windowed fused BP/CP received an invalid CP/BP topology"
            )
        plan = windowed_clean_exchange_plan(
            seq_len=int(global_seq_len),
            block_size=int(global_attn_mask.block_size),
            context_parallel_size=world_size,
            block_parallel_size=block_parallel_size,
            block_group_index=block_parallel_rank // world_size,
            rank=rank,
            window=int(window),
        )
        send_indices, local_receive_positions, remote_receive_positions = (
            _windowed_exchange_tensors(
                plan,
                rank=rank,
                device=query.device,
            )
        )
        local_clean_len = int(global_key_shard.shape[1])
        intervals_by_owner = _clean_intervals_for_runtime(
            int(global_seq_len),
            world_size,
            runtime,
        )
        if local_clean_len != int(owner_lengths(intervals_by_owner)[rank]):
            raise ValueError("clean K/V shard length does not match windowed ownership")
        if int(local_receive_positions.numel()) != local_clean_len:
            raise RuntimeError(
                "windowed fused BP/CP must consume its complete local clean shard"
            )

        send_key = global_key_shard.index_select(1, send_indices)
        send_value = global_value_shard.index_select(1, send_indices)
        active_len = int(local_key.shape[1])
        remote_receive_tokens = int(remote_receive_positions.numel())
        remote_send_counts = list(plan.send_counts)
        remote_receive_counts = list(plan.receive_counts)
        remote_send_counts[rank] = 0
        remote_receive_counts[rank] = 0
        send_payload = (
            torch.stack((send_key, send_value), dim=2)
            .permute(1, 0, 2, 3, 4)
            .contiguous()
        )
        packed = torch.empty(
            (
                active_len + local_clean_len + remote_receive_tokens,
                int(query.shape[0]),
                2,
                int(global_key_shard.shape[2]),
                int(global_key_shard.shape[3]),
            ),
            device=query.device,
            dtype=global_key_shard.dtype,
        )
        input_bytes = int(send_payload.numel()) * int(send_payload.element_size())
        with communication_scope(
            domain="attention",
            phase="forward",
            collective="all_to_all_single",
            input_bytes=input_bytes,
            logical_bytes=input_bytes,
        ):
            work = dist.all_to_all_single(
                packed[active_len + local_clean_len :],
                send_payload,
                output_split_sizes=remote_receive_counts,
                input_split_sizes=remote_send_counts,
                group=runtime.context_block_parallel_group,
                async_op=True,
            )
        packed[:active_len, :, 0].copy_(local_key.permute(1, 0, 2, 3))
        packed[:active_len, :, 1].copy_(local_value.permute(1, 0, 2, 3))
        local_clean_slice = slice(active_len, active_len + local_clean_len)
        packed[local_clean_slice, :, 0].copy_(global_key_shard.permute(1, 0, 2, 3))
        packed[local_clean_slice, :, 1].copy_(global_value_shard.permute(1, 0, 2, 3))
        packed_key = packed[:, :, 0].permute(1, 0, 2, 3)
        packed_value = packed[:, :, 1].permute(1, 0, 2, 3)
        work.wait()
        del send_payload, send_key, send_value
        receive_positions = torch.cat(
            (local_receive_positions, remote_receive_positions),
            dim=0,
        )
        fused_mask = _windowed_fused_mask(
            global_attn_mask,
            local_attn_mask,
            receive_positions=receive_positions,
        )
        output_h, final_lse, _ = _flex_shard_stats(
            query=query.transpose(1, 2),
            key=packed_key.transpose(1, 2),
            value=packed_value.transpose(1, 2),
            attn_mask=fused_mask,
            is_causal=False,
            scale=float(scale),
            key_start=0,
            output_float=False,
            native_fa4=True,
        )
        output = output_h.transpose(1, 2)
        ctx.runtime = runtime
        ctx.scale = float(scale)
        ctx.local_clean_len = local_clean_len
        ctx.active_len = active_len
        ctx.send_counts = tuple(remote_send_counts)
        ctx.receive_counts = tuple(remote_receive_counts)
        ctx.fused_mask = _detach_attention_mask(fused_mask)
        ctx.save_for_backward(
            query.detach(),
            packed_key.detach(),
            packed_value.detach(),
            output.detach(),
            final_lse.detach(),
            send_indices,
        )
        return output

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
        None,
        None,
        None,
    ]:
        query, packed_key, packed_value, output, final_lse, send_indices = (
            ctx.saved_tensors
        )
        fused_mask = ctx.fused_mask
        if not isinstance(fused_mask, BlockDenoisingPackedKeyMask):
            raise RuntimeError("windowed fused BP/CP backward requires a packed mask")
        active_len = int(ctx.active_len)
        grad_payload = torch.empty(
            (
                int(packed_key.shape[1]),
                int(packed_key.shape[0]),
                2,
                int(packed_key.shape[2]),
                int(packed_key.shape[3]),
            ),
            dtype=packed_key.dtype,
            device=packed_key.device,
        )
        grad_key_out = grad_payload[:, :, 0].permute(1, 0, 2, 3)
        grad_value_out = grad_payload[:, :, 1].permute(1, 0, 2, 3)
        if grad_key_out.stride() != packed_key.stride():
            raise RuntimeError("windowed dK payload does not preserve the FA4 layout")
        if grad_value_out.stride() != packed_value.stride():
            raise RuntimeError("windowed dV payload does not preserve the FA4 layout")
        grad_query, _, _ = _packed_clean_flex_shard_backward_exact_bshd(
            query=query,
            key=packed_key,
            value=packed_value,
            final_output=output,
            final_lse=final_lse,
            grad_output=grad_output,
            attn_mask=fused_mask,
            key_start=0,
            scale=float(ctx.scale),
            grad_key=grad_key_out,
            grad_value=grad_value_out,
        )
        local_clean_stop = active_len + int(ctx.local_clean_len)
        collective_input = grad_payload[local_clean_stop:]
        input_bytes = int(collective_input.numel()) * int(
            collective_input.element_size()
        )
        returned = torch.empty(
            (
                int(send_indices.numel()),
                *grad_payload.shape[1:],
            ),
            device=grad_payload.device,
            dtype=grad_payload.dtype,
        )
        with communication_scope(
            domain="attention",
            phase="backward",
            collective="all_to_all_single",
            input_bytes=input_bytes,
            logical_bytes=input_bytes,
        ):
            work = dist.all_to_all_single(
                returned,
                collective_input,
                output_split_sizes=list(ctx.send_counts),
                input_split_sizes=list(ctx.receive_counts),
                group=ctx.runtime.context_block_parallel_group,
                async_op=True,
            )
        work.wait()
        local_grad_kv = grad_payload[active_len:local_clean_stop]
        if int(send_indices.numel()):
            local_grad_kv.index_add_(0, send_indices, returned)
        local_active_key_grad = grad_key_out[:, :active_len]
        local_active_value_grad = grad_value_out[:, :active_len]
        global_key_grad = local_grad_kv[:, :, 0].permute(1, 0, 2, 3)
        global_value_grad = local_grad_kv[:, :, 1].permute(1, 0, 2, 3)
        return (
            grad_query.to(dtype=query.dtype),
            local_active_key_grad,
            local_active_value_grad,
            global_key_grad,
            global_value_grad,
            None,
            None,
            None,
            None,
            None,
        )


class _RingWithLocalKVAttentionBSHD(torch.autograd.Function):
    """Stream clean-owner K/V shards through fused local-block attention."""

    @staticmethod
    def forward(
        ctx: Any,
        query: torch.Tensor,
        local_key: torch.Tensor,
        local_value: torch.Tensor,
        global_key_shard: torch.Tensor,
        global_value_shard: torch.Tensor,
        local_attn_mask: BlockDenoisingLocalActiveMask | None,
        global_attn_mask: BlockDenoisingGlobalCleanMask,
        scale: float,
        runtime: Any,
        global_seq_len: int,
    ) -> torch.Tensor:
        _validate_bshd(query, local_key, local_value)
        _validate_bshd(query, global_key_shard, global_value_shard)
        if not isinstance(global_attn_mask, BlockDenoisingGlobalCleanMask):
            raise RuntimeError("fused BP/CP requires a global clean BDLM mask")
        active_len = int(local_key.shape[1])
        if active_len > 0 and not isinstance(
            local_attn_mask,
            BlockDenoisingLocalActiveMask,
        ):
            raise RuntimeError("fused BP/CP requires a local active BDLM mask")

        local_rank = int(runtime.context_parallel_rank)
        group_ranks = list(runtime.context_block_parallel_group_ranks)
        world_size = len(group_ranks)
        if world_size <= 1:
            raise RuntimeError("fused BP/CP requires multiple context ranks")
        intervals_by_owner = _clean_intervals_for_runtime(
            int(global_seq_len),
            world_size,
            runtime,
        )
        shard_lengths = owner_lengths(intervals_by_owner)
        local_len = int(shard_lengths[local_rank])
        if int(global_key_shard.shape[1]) != local_len:
            raise ValueError("global_key_shard length does not match clean ownership")
        if int(global_value_shard.shape[1]) != local_len:
            raise ValueError("global_value_shard length does not match clean ownership")
        max_shard_len = max(shard_lengths)

        initial_flat, current_key, current_value = _make_kv_ring_payload(
            _pad_kv_shard_bshd(global_key_shard, max_shard_len),
            _pad_kv_shard_bshd(global_value_shard, max_shard_len),
        )
        current_flat = initial_flat
        backward_start_flat = torch.empty_like(initial_flat)
        scratch_flat = torch.empty_like(initial_flat) if world_size > 3 else None

        numerator, m, l = _uninitialized_shard_accumulator_bshd(query)
        initialized = False
        backward_key = None
        backward_value = None
        for step in range(world_size):
            owner = (local_rank + step) % world_size
            if step != world_size - 1:
                if step == 0:
                    recv_flat = backward_start_flat
                elif step % 2 == 1:
                    recv_flat = initial_flat
                else:
                    assert scratch_flat is not None
                    recv_flat = scratch_flat
                next_kv_work = _ring_exchange_kv_payload_async(
                    current_flat,
                    recv=recv_flat,
                    key_shape=current_key.shape,
                    key_numel=current_key.numel(),
                    local_rank=local_rank,
                    group_ranks=group_ranks,
                    group=runtime.context_block_parallel_group,
                    phase="forward",
                )
            else:
                next_kv_work = None

            shard_len = int(shard_lengths[owner])
            owner_key = current_key[:, :shard_len]
            owner_value = current_value[:, :shard_len]
            if owner == local_rank and active_len > 0:
                assert isinstance(local_attn_mask, BlockDenoisingLocalActiveMask)
                shard_key = torch.cat((local_key, owner_key), dim=1)
                shard_value = torch.cat((local_value, owner_value), dim=1)
                shard_mask: (
                    BlockDenoisingGlobalCleanMask | BlockDenoisingPackedKeyMask
                ) = _packed_local_owner_mask(
                    global_attn_mask,
                    local_attn_mask,
                    intervals=intervals_by_owner[owner],
                    device=query.device,
                )
                key_start = 0
                query_indices = None
            else:
                shard_key = owner_key
                shard_value = owner_value
                shard_mask, key_start = _packed_clean_owner_mask(
                    global_attn_mask,
                    intervals=intervals_by_owner[owner],
                    device=query.device,
                )
                query_indices = _owner_query_indices(
                    attn_mask=global_attn_mask,
                    is_causal=False,
                    query_len=int(query.shape[1]),
                    key_start=int(key_start),
                    device=query.device,
                )
            _packed_clean_flex_shard_accumulate_bshd_(
                numerator=numerator,
                m=m,
                l=l,
                query=query,
                key=shard_key,
                value=shard_value,
                attn_mask=shard_mask,
                key_start=int(key_start),
                scale=float(scale),
                initial=not initialized,
                query_indices=query_indices,
                query_cache_key=("clean_owner", int(owner)),
            )
            initialized = True

            if step != world_size - 1:
                assert next_kv_work is not None
                current_flat, current_key, current_value = (
                    _ring_exchange_kv_payload_wait(next_kv_work)
                )
                if step == 0:
                    backward_key = current_key
                    backward_value = current_value

        if not initialized:
            raise RuntimeError("fused BP/CP ring did not visit any clean K/V rows")
        if backward_key is None or backward_value is None:
            raise RuntimeError("fused BP/CP ring did not retain a backward shard")
        output, final_lse = _finalize_shard_stats_bshd(
            numerator=numerator,
            m=m,
            l=l,
            output_dtype=query.dtype,
        )
        ctx.runtime = runtime
        ctx.scale = float(scale)
        ctx.local_rank = local_rank
        ctx.group_ranks = group_ranks
        ctx.shard_lengths = tuple(int(length) for length in shard_lengths)
        ctx.intervals_by_owner = intervals_by_owner
        ctx.max_shard_len = int(max_shard_len)
        ctx.local_attn_mask = _detach_attention_mask(local_attn_mask)
        ctx.global_attn_mask = _detach_attention_mask(global_attn_mask)
        ctx.save_for_backward(
            query.detach(),
            local_key.detach(),
            local_value.detach(),
            backward_key.detach(),
            backward_value.detach(),
            output.detach(),
            final_lse.detach(),
        )
        return output

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
        None,
        None,
        None,
    ]:
        (
            query,
            local_key,
            local_value,
            backward_key,
            backward_value,
            output,
            final_lse,
        ) = ctx.saved_tensors
        global_attn_mask = ctx.global_attn_mask
        if not isinstance(global_attn_mask, BlockDenoisingGlobalCleanMask):
            raise RuntimeError("fused BP/CP backward requires a clean BDLM mask")

        (
            grad_query,
            local_grad_key,
            local_grad_value,
            clean_grad_key,
            clean_grad_value,
        ) = _ring_clean_backward_bshd(
            query=query,
            local_key=local_key,
            local_value=local_value,
            backward_key=backward_key,
            backward_value=backward_value,
            grad_output=grad_output.contiguous(),
            final_output=output,
            final_lse=final_lse,
            local_attn_mask=ctx.local_attn_mask,
            attn_mask=global_attn_mask,
            scale=float(ctx.scale),
            local_rank=int(ctx.local_rank),
            group_ranks=list(ctx.group_ranks),
            shard_lengths=list(ctx.shard_lengths),
            intervals_by_owner=tuple(ctx.intervals_by_owner),
            max_shard_len=int(ctx.max_shard_len),
            group=ctx.runtime.context_block_parallel_group,
        )
        return (
            grad_query.to(dtype=query.dtype),
            local_grad_key.to(dtype=local_key.dtype),
            local_grad_value.to(dtype=local_value.dtype),
            clean_grad_key.to(dtype=backward_key.dtype),
            clean_grad_value.to(dtype=backward_value.dtype),
            None,
            None,
            None,
            None,
            None,
        )


def _select_global_clean_mask_queries(
    mask: BlockDenoisingGlobalCleanMask,
    query_indices: torch.Tensor,
) -> BlockDenoisingGlobalCleanMask:
    bounds = (
        None
        if mask.query_clean_bounds is None
        else mask.query_clean_bounds.index_select(-2, query_indices)
    )
    return BlockDenoisingGlobalCleanMask(
        query_blocks=mask.query_blocks.index_select(-1, query_indices),
        query_is_clean=mask.query_is_clean.index_select(-1, query_indices),
        block_size=int(mask.block_size),
        clean_context_window=mask.clean_context_window,
        query_clean_bounds=bounds,
        clean_key_positions=mask.clean_key_positions,
        clean_key_blocks=mask.clean_key_blocks,
        first_key_length=mask.first_key_length,
        second_key_start=mask.second_key_start,
        backward_query_chunk_size=int(mask.backward_query_chunk_size),
        debug_nonfinite_attention=bool(mask.debug_nonfinite_attention),
        allow_native_wide_attention=bool(mask.allow_native_wide_attention),
        flex_cache=mask.flex_cache,
    )


def _all_gather_active_query_rows_bshd(
    local_rows: torch.Tensor,
    *,
    active_len: int,
    block_size: int,
    rank: int,
    world_size: int,
    group: Any,
    phase: str,
) -> torch.Tensor:
    """Gather within-block query shards and restore logical token order."""

    local = local_rows.permute(1, 0, 2, 3).contiguous()
    gathered = torch.empty(
        (int(world_size) * int(local.shape[0]), *local.shape[1:]),
        device=local.device,
        dtype=local.dtype,
    )
    input_bytes = int(local.numel()) * int(local.element_size())
    with communication_scope(
        domain="attention",
        phase=phase,
        collective="all_gather_into_tensor",
        input_bytes=input_bytes,
        logical_bytes=input_bytes * (int(world_size) - 1),
    ):
        dist.all_gather_into_tensor(gathered, local, group=group)
    del rank
    all_indices = torch.cat(
        tuple(
            active_query_indices_for_context_rank(
                active_len=int(active_len),
                block_size=int(block_size),
                context_parallel_size=int(world_size),
                context_parallel_rank=owner,
                device=local.device,
            )
            for owner in range(int(world_size))
        )
    )
    restored = torch.empty(
        (local_rows.shape[0], int(active_len), *local_rows.shape[2:]),
        device=local_rows.device,
        dtype=local_rows.dtype,
    )
    restored.index_copy_(1, all_indices, gathered.permute(1, 0, 2, 3))
    return restored


class _GatherPureCPActiveKV(torch.autograd.Function):
    """Gather token-offset active K/V shards and reduce their gradients."""

    @staticmethod
    def forward(
        ctx: Any,
        key_shard: torch.Tensor,
        value_shard: torch.Tensor,
        active_len: int,
        block_size: int,
        group: Any,
        rank: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        world_size = int(dist.get_world_size(group))
        rank = int(rank)
        if int(dist.get_rank(group)) != rank:
            raise RuntimeError("pure-CP active K/V group metadata is inconsistent")
        indices_by_rank = tuple(
            active_query_indices_for_context_rank(
                active_len=int(active_len),
                block_size=int(block_size),
                context_parallel_size=world_size,
                context_parallel_rank=owner,
                device=key_shard.device,
            )
            for owner in range(world_size)
        )
        local_rows = int(indices_by_rank[rank].numel())
        if (
            int(key_shard.shape[1]) != local_rows
            or int(value_shard.shape[1]) != local_rows
        ):
            raise ValueError("active K/V shard does not match token-offset ownership")
        packed = (
            torch.stack((key_shard, value_shard), dim=2)
            .permute(1, 0, 2, 3, 4)
            .contiguous()
        )
        gathered = packed.new_empty((world_size * local_rows, *packed.shape[1:]))
        input_bytes = int(packed.numel()) * int(packed.element_size())
        with communication_scope(
            domain="attention",
            phase="forward",
            collective="all_gather_into_tensor",
            input_bytes=input_bytes,
            logical_bytes=input_bytes * (world_size - 1),
        ):
            dist.all_gather_into_tensor(gathered, packed, group=group)
        owner_positions = torch.cat(indices_by_rank)
        logical_order = torch.argsort(owner_positions)
        logical = gathered.index_select(0, logical_order)
        ctx.group = group
        ctx.world_size = world_size
        ctx.local_rows = local_rows
        ctx.save_for_backward(torch.argsort(logical_order))
        return logical[:, :, 0].permute(1, 0, 2, 3), logical[:, :, 1].permute(
            1, 0, 2, 3
        )

    @staticmethod
    def backward(
        ctx: Any,
        grad_key: torch.Tensor,
        grad_value: torch.Tensor,
    ) -> tuple[Any, ...]:
        (rank_major_order,) = ctx.saved_tensors
        logical = (
            torch.stack((grad_key, grad_value), dim=2)
            .permute(1, 0, 2, 3, 4)
            .contiguous()
        )
        owner_major = logical.index_select(0, rank_major_order).contiguous()
        reduced = logical.new_empty((ctx.local_rows, *logical.shape[1:]))
        input_bytes = int(owner_major.numel()) * int(owner_major.element_size())
        with communication_scope(
            domain="attention",
            phase="backward",
            collective="reduce_scatter_tensor",
            input_bytes=input_bytes,
            logical_bytes=input_bytes * (ctx.world_size - 1) // ctx.world_size,
        ):
            dist.reduce_scatter_tensor(reduced, owner_major, group=ctx.group)
        return (
            reduced[:, :, 0].permute(1, 0, 2, 3),
            reduced[:, :, 1].permute(1, 0, 2, 3),
            None,
            None,
            None,
            None,
        )


def _pure_context_active_query_indices(
    *,
    active_len: int,
    block_size: int,
    world_size: int,
    rank: int,
    device: torch.device,
) -> torch.Tensor:
    if int(active_len) == 0:
        return torch.empty(0, device=device, dtype=torch.long)
    return active_query_indices_for_context_rank(
        active_len=int(active_len),
        block_size=int(block_size),
        context_parallel_size=int(world_size),
        context_parallel_rank=int(rank),
        device=device,
    )


class _PureContextShardedActiveAttentionImpl:
    """Pure CP attention with token-offset query ownership inside every block."""

    @staticmethod
    def forward(
        ctx: Any,
        query: torch.Tensor,
        local_key: torch.Tensor,
        local_value: torch.Tensor,
        global_key_shard: torch.Tensor,
        global_value_shard: torch.Tensor,
        local_attn_mask: BlockDenoisingLocalActiveMask,
        global_attn_mask: BlockDenoisingGlobalCleanMask,
        scale: float,
        runtime: Any,
        global_seq_len: int,
        persistent_active_shard: bool,
    ) -> torch.Tensor:
        persistent_active_shard = bool(persistent_active_shard)
        active_len = int(local_key.shape[1])
        rank = int(runtime.context_parallel_rank)
        group_ranks = list(runtime.context_block_parallel_group_ranks)
        world_size = len(group_ranks)
        block_size = int(local_attn_mask.block_size)
        active_indices = _pure_context_active_query_indices(
            active_len=active_len,
            block_size=block_size,
            world_size=world_size,
            rank=rank,
            device=query.device,
        )
        query_metadata_len = (
            int(global_attn_mask.query_blocks.numel())
            if persistent_active_shard
            else int(query.shape[1])
        )
        clean_indices = torch.arange(
            active_len,
            query_metadata_len,
            device=query.device,
            dtype=torch.long,
        )
        compact_indices = torch.cat((active_indices, clean_indices))
        compact_query = (
            query if persistent_active_shard else query.index_select(1, compact_indices)
        )
        local_active_len = int(active_indices.numel())
        expected_query_len = local_active_len + int(clean_indices.numel())
        if int(compact_query.shape[1]) != expected_query_len:
            raise ValueError("persistent pure-CP query rows do not match ownership")

        intervals_by_owner = _clean_intervals_for_runtime(
            int(global_seq_len),
            world_size,
            runtime,
        )
        shard_lengths = owner_lengths(intervals_by_owner)
        local_len = int(shard_lengths[rank])
        if int(global_key_shard.shape[1]) != local_len:
            raise ValueError("global_key_shard length does not match clean ownership")
        max_shard_len = max(shard_lengths)
        owner_to_logical, _ = _collective_clean_indices(
            global_attn_mask,
            intervals_by_owner=intervals_by_owner,
            max_shard_len=int(max_shard_len),
            device=query.device,
        )
        gather_work, gather_input, gathered = _all_gather_clean_kv_bshd(
            global_key_shard,
            global_value_shard,
            max_shard_len=int(max_shard_len),
            group=runtime.context_block_parallel_group,
            world_size=world_size,
        )
        active_stats = None
        if local_active_len > 0:
            active_stats = _block_diagonal_active_query_shard_stats_bshd(
                query=compact_query[:, :local_active_len],
                key=local_key,
                value=local_value,
                block_size=block_size,
                scale=float(scale),
                active_indices=active_indices,
                causal=bool(local_attn_mask.causal),
                flex_cache=local_attn_mask.flex_cache,
            )
        gather_work.wait()
        del gather_input
        clean_key, clean_value = _owner_major_clean_kv_bshd(
            gathered,
            world_size=world_size,
            max_shard_len=int(max_shard_len),
        )
        compact_global_mask = _select_global_clean_mask_queries(
            global_attn_mask,
            compact_indices,
        )
        clean_mask = _collective_owner_major_clean_mask(
            compact_global_mask,
            owner_to_logical=owner_to_logical,
        )
        packed_mask = None
        if (
            not local_attn_mask.causal
            and _uses_collective_packed_d256_backward(compact_query, local_key)
        ):
            packed_mask = _collective_packed_key_mask(
                clean_mask,
                local_attn_mask,
                local_query_blocks=local_attn_mask.query_blocks.index_select(
                    0,
                    compact_indices,
                ),
            )
        numerator, m, l = _uninitialized_shard_accumulator_bshd(compact_query)
        _packed_clean_flex_shard_accumulate_bshd_(
            numerator=numerator,
            m=m,
            l=l,
            query=compact_query,
            key=clean_key,
            value=clean_value,
            attn_mask=clean_mask,
            key_start=0,
            scale=float(scale),
            initial=True,
        )
        if active_stats is not None:
            active_num, active_m, active_l = active_stats
            _merge_active_prefix_stats_in_place(
                numerator=numerator,
                m=m,
                l=l,
                active_num=active_num,
                active_m=active_m,
                active_l=active_l,
                attn_mask=local_attn_mask,
            )
        compact_output, final_lse = _finalize_shard_stats_bshd(
            numerator=numerator,
            m=m,
            l=l,
            output_dtype=query.dtype,
        )
        if persistent_active_shard:
            output = compact_output
        else:
            active_output = _all_gather_active_query_rows_bshd(
                compact_output[:, :local_active_len],
                active_len=active_len,
                block_size=block_size,
                rank=rank,
                world_size=world_size,
                group=runtime.context_block_parallel_group,
                phase="forward",
            )
            output = torch.cat(
                (active_output, compact_output[:, local_active_len:]), dim=1
            )

        ctx.runtime = runtime
        ctx.scale = float(scale)
        ctx.rank = rank
        ctx.world_size = world_size
        ctx.active_len = active_len
        ctx.local_active_len = local_active_len
        ctx.local_len = local_len
        ctx.max_shard_len = int(max_shard_len)
        ctx.block_size = block_size
        ctx.persistent_active_shard = persistent_active_shard
        ctx.clean_key_grad_required = bool(global_key_shard.requires_grad)
        ctx.clean_value_grad_required = bool(global_value_shard.requires_grad)
        ctx.clean_mask = _detach_attention_mask(clean_mask)
        ctx.packed_mask = (
            None if packed_mask is None else _detach_attention_mask(packed_mask)
        )
        ctx.local_attn_mask = _detach_attention_mask(local_attn_mask)
        ctx.save_for_backward(
            compact_query.detach(),
            local_key.detach(),
            local_value.detach(),
            clean_key.detach(),
            clean_value.detach(),
            compact_output.detach(),
            final_lse.detach(),
            active_indices,
        )
        return output

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[Any, ...]:
        (
            compact_query,
            local_key,
            local_value,
            clean_key,
            clean_value,
            compact_output,
            final_lse,
            active_indices,
        ) = ctx.saved_tensors
        active_len = int(ctx.active_len)
        local_active_len = int(ctx.local_active_len)
        if ctx.persistent_active_shard:
            compact_grad_output = grad_output
        else:
            compact_grad_output = torch.cat(
                (
                    grad_output[:, :active_len].index_select(1, active_indices),
                    grad_output[:, active_len:],
                ),
                dim=1,
            )
        if ctx.packed_mask is not None:
            kv_head_repeat = _collective_packed_d256_kv_head_repeat(
                compact_query,
                local_key,
            )
            if kv_head_repeat is None:
                raise RuntimeError("packed D=256 backward lost its valid GQA layout")
            packed_kv, packed_key, packed_value = _pack_local_clean_kv_bshd(
                local_key,
                local_value,
                clean_key,
                clean_value,
                kv_head_repeat=kv_head_repeat,
            )
            grad_payload = torch.empty_like(packed_kv)
            packed_grad_key = grad_payload[:, :, 0].permute(1, 0, 2, 3)
            packed_grad_value = grad_payload[:, :, 1].permute(1, 0, 2, 3)
            compact_grad_query, _, _ = _packed_clean_flex_shard_backward_exact_bshd(
                query=compact_query,
                key=packed_key,
                value=packed_value,
                final_output=compact_output,
                final_lse=final_lse,
                grad_output=compact_grad_output,
                attn_mask=ctx.packed_mask,
                key_start=0,
                scale=float(ctx.scale),
                grad_key=packed_grad_key,
                grad_value=packed_grad_value,
            )
            collapsed_grad_payload = _collapse_repeated_kv_grad_storage_bshd(
                grad_payload,
                original_key_heads=int(local_key.shape[2]),
                kv_head_repeat=kv_head_repeat,
            )
            active_grad_storage = collapsed_grad_payload[:active_len]
            active_grad_key = active_grad_storage[:, :, 0].permute(1, 0, 2, 3)
            active_grad_value = active_grad_storage[:, :, 1].permute(1, 0, 2, 3)
            clean_grad_storage = collapsed_grad_payload[active_len:]
        else:
            clean_grad_storage, clean_grad_key, clean_grad_value = (
                _owner_major_clean_grad_storage_bshd(
                    clean_key,
                    clean_value,
                    max_shard_len=int(ctx.max_shard_len),
                    world_size=int(ctx.world_size),
                )
            )
            clean_query_indices = _owner_query_indices(
                attn_mask=ctx.clean_mask,
                is_causal=False,
                query_len=int(compact_query.shape[1]),
                key_start=0,
                device=compact_query.device,
            )
            clean_grad_query_compact, _, _ = (
                _packed_clean_flex_shard_backward_exact_bshd(
                    query=compact_query,
                    key=clean_key,
                    value=clean_value,
                    final_output=compact_output,
                    final_lse=final_lse,
                    grad_output=compact_grad_output,
                    attn_mask=ctx.clean_mask,
                    key_start=0,
                    scale=float(ctx.scale),
                    query_indices=clean_query_indices,
                    query_cache_key=("persistent_clean", 0),
                    grad_key=clean_grad_key,
                    grad_value=clean_grad_value,
                )
            )
            if clean_query_indices is None:
                clean_grad_query = clean_grad_query_compact
            else:
                clean_grad_query = torch.zeros_like(
                    compact_query,
                    dtype=clean_grad_query_compact.dtype,
                )
                clean_grad_query.index_copy_(
                    1,
                    clean_query_indices,
                    clean_grad_query_compact,
                )
            if local_active_len > 0:
                active_grad_query, active_grad_key, active_grad_value = (
                    _block_diagonal_active_query_shard_backward_bshd(
                        query=compact_query[:, :local_active_len],
                        key=local_key,
                        value=local_value,
                        output=compact_output[:, :local_active_len],
                        lse=final_lse[..., :local_active_len],
                        grad_output=compact_grad_output[:, :local_active_len],
                        block_size=int(ctx.block_size),
                        scale=float(ctx.scale),
                        debug_nonfinite_attention=bool(
                            ctx.local_attn_mask.debug_nonfinite_attention
                        ),
                        active_indices=active_indices,
                        causal=bool(ctx.local_attn_mask.causal),
                        flex_cache=ctx.local_attn_mask.flex_cache,
                    )
                )
            else:
                active_grad_query = torch.zeros_like(
                    compact_query[:, :local_active_len],
                    dtype=torch.float32,
                )
                active_grad_key = torch.zeros_like(local_key)
                active_grad_value = torch.zeros_like(local_value)
            compact_grad_query = clean_grad_query.float()
            compact_grad_query[:, :local_active_len].add_(active_grad_query.float())
            active_grad_storage = (
                torch.stack((active_grad_key, active_grad_value), dim=2)
                .permute(1, 0, 2, 3, 4)
                .contiguous()
            )
            active_grad_key = active_grad_storage[:, :, 0].permute(1, 0, 2, 3)
            active_grad_value = active_grad_storage[:, :, 1].permute(1, 0, 2, 3)
        clean_reduce_work = None
        clean_reduced = None
        if bool(ctx.clean_key_grad_required) or bool(ctx.clean_value_grad_required):
            clean_reduce_work, clean_reduced = (
                _reduce_scatter_owner_major_clean_grad_storage_bshd(
                    clean_grad_storage,
                    max_shard_len=int(ctx.max_shard_len),
                    group=ctx.runtime.context_block_parallel_group,
                    world_size=int(ctx.world_size),
                )
            )
        if ctx.persistent_active_shard:
            grad_query = compact_grad_query.to(dtype=compact_query.dtype)
        else:
            full_active_grad_query = _all_gather_active_query_rows_bshd(
                compact_grad_query[:, :local_active_len].to(dtype=compact_query.dtype),
                active_len=active_len,
                block_size=int(ctx.block_size),
                rank=int(ctx.rank),
                world_size=int(ctx.world_size),
                group=ctx.runtime.context_block_parallel_group,
                phase="backward",
            )
            input_bytes = int(active_grad_storage.numel()) * int(
                active_grad_storage.element_size()
            )
            with communication_scope(
                domain="attention",
                phase="backward",
                collective="all_reduce",
                input_bytes=input_bytes,
                logical_bytes=input_bytes * (int(ctx.world_size) - 1),
            ):
                active_reduce_work = dist.all_reduce(
                    active_grad_storage,
                    group=ctx.runtime.context_block_parallel_group,
                    async_op=True,
                )
            active_reduce_work.wait()
            grad_query = torch.cat(
                (
                    full_active_grad_query,
                    compact_grad_query[:, local_active_len:].to(
                        dtype=compact_query.dtype
                    ),
                ),
                dim=1,
            )
        if clean_reduce_work is not None:
            clean_reduce_work.wait()
        local_len = int(ctx.local_len)
        clean_grad_key_result = (
            None
            if not bool(ctx.clean_key_grad_required)
            else clean_reduced[:local_len, :, 0].permute(1, 0, 2, 3)
        )
        clean_grad_value_result = (
            None
            if not bool(ctx.clean_value_grad_required)
            else clean_reduced[:local_len, :, 1].permute(1, 0, 2, 3)
        )
        return (
            grad_query,
            active_grad_key.to(dtype=local_key.dtype),
            active_grad_value.to(dtype=local_value.dtype),
            clean_grad_key_result,
            clean_grad_value_result,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _PureContextShardedActiveAttentionBSHD(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, *args: Any) -> torch.Tensor:
        return _PureContextShardedActiveAttentionImpl.forward(ctx, *args, False)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        return _PureContextShardedActiveAttentionImpl.backward(ctx, grad_output)[:-1]


class _PersistentPureContextShardedActiveAttentionBSHD(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, *args: Any) -> torch.Tensor:
        return _PureContextShardedActiveAttentionImpl.forward(ctx, *args, True)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        return _PureContextShardedActiveAttentionImpl.backward(ctx, grad_output)[:-1]


def _run_phased_clean_local_backward_bshd(
    *,
    phases: Any,
    clean_grad_storage: torch.Tensor,
    clean_grad_key: torch.Tensor,
    clean_grad_value: torch.Tensor,
    launch_reduce: Any,
    run_local_backward: Any,
) -> tuple[
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    torch.Tensor,
]:
    """Overlap owner reduction with the independent dQ and local-block work."""

    phases.dkdv(grad_key=clean_grad_key, grad_value=clean_grad_value)
    reduce_work, reduced = launch_reduce(clean_grad_storage)
    clean_grad_query = phases.dq()
    local_gradients = run_local_backward()
    reduce_work.wait()
    return clean_grad_query, local_gradients, reduced


class _CollectiveWithLocalKVAttentionBSHD(torch.autograd.Function):
    """Overlapped local-block and gathered clean-prefix BP/CP attention."""

    @staticmethod
    def forward(
        ctx: Any,
        query: torch.Tensor,
        local_key: torch.Tensor,
        local_value: torch.Tensor,
        global_key_shard: torch.Tensor,
        global_value_shard: torch.Tensor,
        local_attn_mask: BlockDenoisingLocalActiveMask | None,
        global_attn_mask: BlockDenoisingGlobalCleanMask,
        scale: float,
        runtime: Any,
        global_seq_len: int,
    ) -> torch.Tensor:
        _validate_bshd(query, local_key, local_value)
        _validate_bshd(query, global_key_shard, global_value_shard)
        if not isinstance(global_attn_mask, BlockDenoisingGlobalCleanMask):
            raise RuntimeError("fused BP/CP requires a global clean BDLM mask")
        if local_key.shape[1] > 0 and not isinstance(
            local_attn_mask,
            BlockDenoisingLocalActiveMask,
        ):
            raise RuntimeError("fused BP/CP requires a local active BDLM mask")

        local_rank = int(runtime.context_parallel_rank)
        group_ranks = list(runtime.context_block_parallel_group_ranks)
        world_size = len(group_ranks)
        if world_size <= 1:
            raise RuntimeError("fused BP/CP requires multiple context ranks")
        intervals_by_owner = _clean_intervals_for_runtime(
            int(global_seq_len),
            world_size,
            runtime,
        )
        shard_lengths = owner_lengths(intervals_by_owner)
        local_len = int(shard_lengths[local_rank])
        if int(global_key_shard.shape[1]) != local_len:
            raise ValueError("global_key_shard length does not match clean ownership")
        if int(global_value_shard.shape[1]) != local_len:
            raise ValueError("global_value_shard length does not match clean ownership")
        max_shard_len = max(shard_lengths)
        owner_to_logical, _ = _collective_clean_indices(
            global_attn_mask,
            intervals_by_owner=intervals_by_owner,
            max_shard_len=int(max_shard_len),
            device=query.device,
        )
        active_len = int(local_key.shape[1])
        gather_work, gather_input, gathered = _all_gather_clean_kv_bshd(
            global_key_shard,
            global_value_shard,
            max_shard_len=int(max_shard_len),
            group=runtime.context_block_parallel_group,
            world_size=world_size,
        )
        if not isinstance(local_attn_mask, BlockDenoisingLocalActiveMask):
            raise RuntimeError("fused BP/CP requires a local active mask")

        native_clean_attention = _collective_clean_uses_native_wide_attention(
            query,
            active_len=active_len,
        )
        active_num, active_m, active_l = _block_diagonal_active_stats_bshd(
            query=query[:, :active_len],
            key=local_key,
            value=local_value,
            block_size=int(local_attn_mask.block_size),
            scale=float(scale),
            causal=bool(local_attn_mask.causal),
        )
        gather_work.wait()
        del gather_input
        clean_key, clean_value = _owner_major_clean_kv_bshd(
            gathered,
            world_size=world_size,
            max_shard_len=int(max_shard_len),
        )
        clean_mask = _collective_owner_major_clean_mask(
            global_attn_mask,
            owner_to_logical=owner_to_logical,
            allow_native_wide_attention=native_clean_attention,
        )
        packed_mask = None
        if (
            not local_attn_mask.causal
            and _uses_collective_packed_d256_backward(query, local_key)
        ):
            packed_mask = _collective_packed_key_mask(clean_mask, local_attn_mask)
        # The external D=512 launch needs NCCL's gathered write completed
        # before its first clean-only read.  Mixed attention already performs
        # intervening framework work that establishes the dependency.
        _complete_collective_clean_gather_for_native_attention(
            clean_key,
            active_len=active_len,
            native_clean_attention=native_clean_attention,
        )
        numerator, m, l = _uninitialized_shard_accumulator_bshd(query)
        _packed_clean_flex_shard_accumulate_bshd_(
            numerator=numerator,
            m=m,
            l=l,
            query=query,
            key=clean_key,
            value=clean_value,
            attn_mask=clean_mask,
            key_start=0,
            scale=float(scale),
            initial=True,
        )
        _merge_active_prefix_stats_in_place(
            numerator=numerator,
            m=m,
            l=l,
            active_num=active_num,
            active_m=active_m,
            active_l=active_l,
            attn_mask=local_attn_mask,
        )
        output, final_lse = _finalize_shard_stats_bshd(
            numerator=numerator,
            m=m,
            l=l,
            output_dtype=query.dtype,
        )
        ctx.runtime = runtime
        ctx.scale = float(scale)
        ctx.world_size = world_size
        ctx.local_len = local_len
        ctx.max_shard_len = int(max_shard_len)
        ctx.local_key_len = active_len
        ctx.local_attn_mask = _detach_attention_mask(local_attn_mask)
        ctx.clean_mask = _detach_attention_mask(clean_mask)
        ctx.packed_mask = (
            None if packed_mask is None else _detach_attention_mask(packed_mask)
        )
        ctx.save_for_backward(
            query.detach(),
            local_key.detach(),
            local_value.detach(),
            clean_key.detach(),
            clean_value.detach(),
            output.detach(),
            final_lse.detach(),
        )
        return output

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
        None,
        None,
        None,
    ]:
        (
            query,
            local_key,
            local_value,
            clean_key,
            clean_value,
            output,
            final_lse,
        ) = ctx.saved_tensors
        local_attn_mask = ctx.local_attn_mask
        clean_mask = ctx.clean_mask
        active_len = int(ctx.local_key_len)
        if not isinstance(local_attn_mask, BlockDenoisingLocalActiveMask):
            raise RuntimeError("fused BP/CP backward requires a local active mask")
        if not isinstance(clean_mask, BlockDenoisingGlobalCleanMask):
            raise RuntimeError("fused BP/CP backward requires a clean mask")

        if ctx.packed_mask is not None:
            kv_head_repeat = _collective_packed_d256_kv_head_repeat(query, local_key)
            if kv_head_repeat is None:
                raise RuntimeError("packed D=256 backward lost its valid KV layout")
            packed_kv, packed_key, packed_value = _pack_local_clean_kv_bshd(
                local_key,
                local_value,
                clean_key,
                clean_value,
                kv_head_repeat=kv_head_repeat,
            )
            grad_payload = torch.empty_like(packed_kv)
            packed_grad_key = grad_payload[:, :, 0].permute(1, 0, 2, 3)
            packed_grad_value = grad_payload[:, :, 1].permute(1, 0, 2, 3)
            grad_query, _, _ = _packed_clean_flex_shard_backward_exact_bshd(
                query=query,
                key=packed_key,
                value=packed_value,
                final_output=output,
                final_lse=final_lse,
                grad_output=grad_output,
                attn_mask=ctx.packed_mask,
                key_start=0,
                scale=float(ctx.scale),
                grad_key=packed_grad_key,
                grad_value=packed_grad_value,
            )
            collapsed_grad_payload = _collapse_repeated_kv_grad_storage_bshd(
                grad_payload,
                original_key_heads=int(local_key.shape[2]),
                kv_head_repeat=kv_head_repeat,
            )
            clean_grad_storage = collapsed_grad_payload[active_len:]
            reduce_work, reduced = _reduce_scatter_owner_major_clean_grad_storage_bshd(
                clean_grad_storage,
                max_shard_len=int(ctx.max_shard_len),
                group=ctx.runtime.context_block_parallel_group,
                world_size=int(ctx.world_size),
            )
            reduce_work.wait()
            _debug_check_backward_finite(
                "collective_packed",
                grad_query,
                packed_grad_key,
                packed_grad_value,
                query=query,
                key=packed_key,
                key_start=0,
                block_size=int(clean_mask.block_size),
                debug_nonfinite_attention=bool(clean_mask.debug_nonfinite_attention),
            )
            local_len = int(ctx.local_len)
            return (
                grad_query.to(dtype=query.dtype),
                collapsed_grad_payload[:active_len, :, 0]
                .permute(1, 0, 2, 3)
                .to(dtype=local_key.dtype),
                collapsed_grad_payload[:active_len, :, 1]
                .permute(1, 0, 2, 3)
                .to(dtype=local_value.dtype),
                reduced[:local_len, :, 0].permute(1, 0, 2, 3),
                reduced[:local_len, :, 1].permute(1, 0, 2, 3),
                None,
                None,
                None,
                None,
                None,
            )

        clean_grad_storage, clean_grad_key, clean_grad_value = (
            _owner_major_clean_grad_storage_bshd(
                clean_key,
                clean_value,
                max_shard_len=int(ctx.max_shard_len),
                world_size=int(ctx.world_size),
            )
        )

        def launch_reduce(storage: torch.Tensor) -> tuple[Any, torch.Tensor]:
            return _reduce_scatter_owner_major_clean_grad_storage_bshd(
                storage,
                max_shard_len=int(ctx.max_shard_len),
                group=ctx.runtime.context_block_parallel_group,
                world_size=int(ctx.world_size),
            )

        def run_local_backward() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return _block_diagonal_active_backward_bshd(
                query=query[:, :active_len],
                key=local_key,
                value=local_value,
                output=output[:, :active_len],
                lse=final_lse[..., :active_len],
                grad_output=grad_output[:, :active_len],
                block_size=int(local_attn_mask.block_size),
                scale=float(ctx.scale),
                backward_query_chunk_size=int(
                    local_attn_mask.backward_query_chunk_size
                ),
                debug_nonfinite_attention=bool(
                    local_attn_mask.debug_nonfinite_attention
                ),
                flex_cache=local_attn_mask.flex_cache,
                causal=bool(local_attn_mask.causal),
            )

        if _uses_native_wide_attention(query):
            phases = _prepare_packed_clean_wide_backward_bshd(
                query=query,
                key=clean_key,
                value=clean_value,
                final_output=output,
                final_lse=final_lse,
                grad_output=grad_output,
                attn_mask=clean_mask,
                scale=float(ctx.scale),
            )
            clean_grad_query, local_gradients, reduced = (
                _run_phased_clean_local_backward_bshd(
                    phases=phases,
                    clean_grad_storage=clean_grad_storage,
                    clean_grad_key=clean_grad_key,
                    clean_grad_value=clean_grad_value,
                    launch_reduce=launch_reduce,
                    run_local_backward=run_local_backward,
                )
            )
            (
                local_grad_query,
                local_active_key_grad,
                local_active_value_grad,
            ) = local_gradients
        else:
            clean_grad_query, _, _ = _packed_clean_flex_shard_backward_exact_bshd(
                query=query,
                key=clean_key,
                value=clean_value,
                final_output=output,
                final_lse=final_lse,
                grad_output=grad_output,
                attn_mask=clean_mask,
                key_start=0,
                scale=float(ctx.scale),
                grad_key=clean_grad_key,
                grad_value=clean_grad_value,
            )
            reduce_work, reduced = launch_reduce(clean_grad_storage)
            (
                local_grad_query,
                local_active_key_grad,
                local_active_value_grad,
            ) = run_local_backward()
            reduce_work.wait()
        _debug_check_backward_finite(
            "collective_clean_local",
            clean_grad_query,
            clean_grad_key,
            clean_grad_value,
            query=query,
            key=clean_key,
            key_start=0,
            block_size=int(clean_mask.block_size),
            debug_nonfinite_attention=bool(clean_mask.debug_nonfinite_attention),
        )
        _debug_check_backward_finite(
            "collective_clean_reduced",
            clean_grad_query,
            reduced[:, :, 0],
            reduced[:, :, 1],
            query=query,
            key=clean_key,
            key_start=0,
            block_size=int(clean_mask.block_size),
            debug_nonfinite_attention=bool(clean_mask.debug_nonfinite_attention),
        )
        del clean_grad_storage
        grad_query = clean_grad_query.float()
        grad_query[:, :active_len].add_(local_grad_query.float())
        local_len = int(ctx.local_len)
        global_key_grad = reduced[:local_len, :, 0].permute(1, 0, 2, 3)
        global_value_grad = reduced[:local_len, :, 1].permute(1, 0, 2, 3)
        return (
            grad_query.to(dtype=query.dtype),
            local_active_key_grad,
            local_active_value_grad,
            global_key_grad,
            global_value_grad,
            None,
            None,
            None,
            None,
            None,
        )


class _CollectiveContextParallelAttentionBSHD(torch.autograd.Function):
    """Standard CP with gathered K/V and owner-reduced K/V gradients."""

    @staticmethod
    def forward(
        ctx: Any,
        query: torch.Tensor,
        key_shard: torch.Tensor,
        value_shard: torch.Tensor,
        attn_mask: BlockDenoisingFullMask,
        scale: float,
        runtime: Any,
        global_seq_len: int,
    ) -> torch.Tensor:
        _validate_bshd(query, key_shard, value_shard)
        if not isinstance(attn_mask, BlockDenoisingFullMask):
            raise RuntimeError("pure CP requires a full block-denoising mask")
        local_rank = int(runtime.context_parallel_rank)
        group_ranks = list(runtime.context_block_parallel_group_ranks)
        world_size = len(group_ranks)
        if world_size <= 1:
            raise RuntimeError("pure CP requires multiple context ranks")
        intervals_by_owner = all_context_parallel_sequence_intervals(
            seq_len=int(global_seq_len) // 2,
            context_parallel_size=world_size,
        )
        shard_lengths = owner_lengths(intervals_by_owner)
        local_len = int(shard_lengths[local_rank])
        if int(key_shard.shape[1]) != local_len:
            raise ValueError("local key length does not match CP ownership")
        if int(value_shard.shape[1]) != local_len:
            raise ValueError("local value length does not match CP ownership")
        max_shard_len = max(shard_lengths)
        owner_to_logical, logical_from_owner = _collective_clean_indices(
            attn_mask,
            intervals_by_owner=intervals_by_owner,
            max_shard_len=int(max_shard_len),
            device=query.device,
        )
        gather_work, gather_input, gathered = _all_gather_clean_kv_bshd(
            key_shard,
            value_shard,
            max_shard_len=int(max_shard_len),
            group=runtime.context_block_parallel_group,
            world_size=world_size,
        )
        gather_work.wait()
        del gather_input
        owner_key, owner_value = _owner_major_clean_kv_bshd(
            gathered,
            world_size=world_size,
            max_shard_len=int(max_shard_len),
        )
        full_key = owner_key.index_select(1, logical_from_owner)
        full_value = owner_value.index_select(1, logical_from_owner)
        output_h, final_lse, _ = _flex_shard_stats(
            query=query.transpose(1, 2),
            key=full_key.transpose(1, 2),
            value=full_value.transpose(1, 2),
            attn_mask=attn_mask,
            is_causal=False,
            scale=float(scale),
            key_start=0,
            output_float=False,
        )
        output = output_h.transpose(1, 2)
        ctx.runtime = runtime
        ctx.scale = float(scale)
        ctx.world_size = world_size
        ctx.local_len = local_len
        ctx.max_shard_len = int(max_shard_len)
        ctx.attn_mask = _detach_attention_mask(attn_mask)
        ctx.save_for_backward(
            query.detach(),
            full_key.detach(),
            full_value.detach(),
            output.detach(),
            final_lse.detach(),
            owner_to_logical,
        )
        return output

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
        None,
        None,
    ]:
        (
            query,
            full_key,
            full_value,
            output,
            final_lse,
            owner_to_logical,
        ) = ctx.saved_tensors
        attn_mask = ctx.attn_mask
        if not isinstance(attn_mask, BlockDenoisingFullMask):
            raise RuntimeError("pure CP backward requires a full mask")
        grad_query_h, grad_key_h, grad_value_h = _flex_merged_shard_backward(
            query=query.transpose(1, 2),
            key=full_key.transpose(1, 2),
            value=full_value.transpose(1, 2),
            final_output=output.transpose(1, 2),
            final_lse=final_lse,
            grad_output=grad_output.transpose(1, 2),
            attn_mask=attn_mask,
            is_causal=False,
            scale=float(ctx.scale),
            key_start=0,
        )
        grad_key, grad_value = _owner_major_clean_grads_bshd(
            grad_key_h.transpose(1, 2),
            grad_value_h.transpose(1, 2),
            owner_to_logical=owner_to_logical,
        )
        reduce_work, reduce_input, reduced = _reduce_scatter_clean_kv_grads_bshd(
            grad_key,
            grad_value,
            max_shard_len=int(ctx.max_shard_len),
            group=ctx.runtime.context_block_parallel_group,
            world_size=int(ctx.world_size),
        )
        reduce_work.wait()
        del reduce_input
        local_len = int(ctx.local_len)
        key_grad = reduced[0, :, :local_len].to(dtype=full_key.dtype)
        value_grad = reduced[1, :, :local_len].to(dtype=full_value.dtype)
        return (
            grad_query_h.transpose(1, 2).to(dtype=query.dtype),
            key_grad,
            value_grad,
            None,
            None,
            None,
            None,
        )


def replicated_block_denoising_attention_bshd(
    query: torch.Tensor,
    local_key: torch.Tensor,
    local_value: torch.Tensor,
    global_key: torch.Tensor,
    global_value: torch.Tensor,
    *,
    local_attn_mask: BlockDenoisingLocalActiveMask | None,
    global_attn_mask: BlockDenoisingGlobalCleanMask,
    scale: float | None = None,
) -> torch.Tensor:
    """Packed block-denoising attention for a replicated clean-prefix cache.

    This is the CP=1 companion to ``fused_block_context_attention_bshd``.
    It uses the same local noisy-block and global clean-prefix BDLM attention
    primitives, online softmax merge, and exact backward, but without any
    distributed K/V transport.
    """

    _validate_bshd(query, local_key, local_value)
    _validate_bshd(query, global_key, global_value)
    if not isinstance(global_attn_mask, BlockDenoisingGlobalCleanMask):
        raise RuntimeError(
            "replicated block-denoising attention requires a global clean BDLM mask"
        )
    if local_key.shape[1] > 0 and not isinstance(
        local_attn_mask,
        BlockDenoisingLocalActiveMask,
    ):
        raise RuntimeError(
            "replicated block-denoising attention requires a local active BDLM mask"
        )
    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])
    return _ReplicatedWithLocalKVAttentionBSHD.apply(
        query,
        local_key,
        local_value,
        global_key,
        global_value,
        local_attn_mask,
        global_attn_mask,
        float(scale),
    )


class _RingContextParallelAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None,
        is_causal: bool,
        scale: float | None,
        runtime: Any,
        key_chunk_size: int | None,
        shard_kv: bool,
        global_seq_len: int,
        input_is_shard: bool,
    ) -> torch.Tensor:
        _validate_bhsd(query, key, value)
        if scale is None:
            scale = 1.0 / math.sqrt(query.shape[-1])

        local_rank = int(runtime.context_parallel_rank)
        group_ranks = list(runtime.context_block_parallel_group_ranks)
        num_context_ranks = len(group_ranks)
        if num_context_ranks <= 1 or not shard_kv:
            raise RuntimeError("ring context attention requires sharded multi-rank K/V")

        if input_is_shard:
            intervals_by_owner = all_context_parallel_sequence_intervals(
                seq_len=int(global_seq_len) // 2,
                context_parallel_size=num_context_ranks,
            )
        else:
            intervals_by_owner = _clean_intervals_for_runtime(
                int(global_seq_len),
                num_context_ranks,
                runtime,
            )
        shard_lengths = owner_lengths(intervals_by_owner)
        local_intervals = intervals_by_owner[local_rank]
        max_shard_len = max(shard_lengths)
        if input_is_shard and key.shape[-2] != shard_lengths[local_rank]:
            raise ValueError("local key length does not match zigzag CP ownership")
        if input_is_shard and value.shape[-2] != shard_lengths[local_rank]:
            raise ValueError("local value length does not match zigzag CP ownership")
        local_key = (
            key.contiguous() if input_is_shard else _cat_intervals(key, local_intervals)
        )
        local_value = (
            value.contiguous()
            if input_is_shard
            else _cat_intervals(value, local_intervals)
        )
        current_flat, current_key, current_value = _make_kv_ring_payload(
            _pad_kv_shard(local_key, max_shard_len),
            _pad_kv_shard(local_value, max_shard_len),
        )

        numerator = None
        m = None
        l = None
        for step in range(num_context_ranks):
            owner = (local_rank + step) % num_context_ranks
            shard_len = shard_lengths[owner]
            if step != num_context_ranks - 1:
                next_kv_work = _ring_exchange_kv_payload_async(
                    current_flat,
                    key_shape=current_key.shape,
                    key_numel=current_key.numel(),
                    local_rank=local_rank,
                    group_ranks=group_ranks,
                    group=runtime.context_block_parallel_group,
                    phase="forward",
                )
            else:
                next_kv_work = None

            if shard_len > 0:
                merged_stats = _owner_shard_stats_into(
                    numerator=numerator,
                    m=m,
                    l=l,
                    query=query,
                    key=current_key[..., :shard_len, :],
                    value=current_value[..., :shard_len, :],
                    owner_intervals_=intervals_by_owner[owner],
                    attn_mask=attn_mask,
                    is_causal=is_causal,
                    scale=float(scale),
                )
                if merged_stats is not None:
                    numerator, m, l = merged_stats
            if step != num_context_ranks - 1:
                assert next_kv_work is not None
                current_flat, current_key, current_value = (
                    _ring_exchange_kv_payload_wait(next_kv_work)
                )

        if numerator is None:
            output = torch.zeros_like(query)
            m = torch.full(
                query.shape[:-1],
                -torch.inf,
                dtype=torch.float32,
                device=query.device,
            )
            l = torch.zeros_like(m)
        else:
            assert l is not None and m is not None
            tiny = torch.finfo(torch.float32).tiny
            output = numerator.div_(l.clamp_min(tiny).unsqueeze(-1))
            output.masked_fill_(l.unsqueeze(-1) <= 0, 0)

        ctx.disabled = False
        ctx.runtime = runtime
        ctx.is_causal = bool(is_causal)
        ctx.scale = float(scale)
        ctx.key_chunk_size = key_chunk_size
        ctx.local_rank = local_rank
        ctx.group_ranks = group_ranks
        ctx.shard_lengths = shard_lengths
        ctx.intervals_by_owner = intervals_by_owner
        ctx.max_shard_len = max_shard_len
        ctx.key_shape = key.shape
        ctx.value_shape = value.shape
        ctx.input_is_shard = bool(input_is_shard)
        ctx.has_attn_mask = attn_mask is not None
        ctx.attn_mask = _detach_attention_mask(attn_mask)
        ctx.save_for_backward(
            query.detach(),
            local_key.detach(),
            local_value.detach(),
            output.detach().to(dtype=query.dtype),
            m.detach(),
            l.detach(),
        )
        return output.to(dtype=query.dtype)

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        if getattr(ctx, "disabled", False):
            raise RuntimeError("ring attention disabled path should not backward")

        (
            query,
            local_key,
            local_value,
            output,
            m,
            l,
        ) = ctx.saved_tensors
        attn_mask = ctx.attn_mask if ctx.has_attn_mask else None

        local_rank = int(ctx.local_rank)
        group_ranks = list(ctx.group_ranks)
        shard_lengths = list(ctx.shard_lengths)
        intervals_by_owner = tuple(ctx.intervals_by_owner)
        max_shard_len = int(ctx.max_shard_len)

        grad_query, reduced_key, reduced_value = _ring_context_backward_flex(
            query=query,
            local_key=local_key,
            local_value=local_value,
            grad_output=grad_output,
            final_output=output,
            final_m=m,
            final_l=l,
            attn_mask=attn_mask,
            is_causal=ctx.is_causal,
            scale=ctx.scale,
            local_rank=local_rank,
            group_ranks=group_ranks,
            shard_lengths=shard_lengths,
            intervals_by_owner=intervals_by_owner,
            max_shard_len=max_shard_len,
            group=ctx.runtime.context_block_parallel_group,
        )
        if ctx.input_is_shard:
            local_len = int(shard_lengths[local_rank])
            grad_key = reduced_key[..., :local_len, :]
            grad_value = reduced_value[..., :local_len, :]
        else:
            grad_key = torch.zeros(
                ctx.key_shape,
                device=query.device,
                dtype=torch.float32,
            )
            grad_value = torch.zeros(
                ctx.value_shape,
                device=query.device,
                dtype=torch.float32,
            )
            _write_interval_grads(grad_key, reduced_key, intervals_by_owner[local_rank])
            _write_interval_grads(
                grad_value, reduced_value, intervals_by_owner[local_rank]
            )
        _sync_cuda_stream_if_needed(query)
        return (
            grad_query.to(dtype=query.dtype),
            grad_key.to(dtype=local_key.dtype),
            grad_value.to(dtype=local_value.dtype),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _RingContextParallelAttentionBSHD(torch.autograd.Function):
    """Native-layout pure CP ring with overlapped K/V and owner DKV transport."""

    @staticmethod
    def forward(
        ctx: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: BlockDenoisingFullMask,
        scale: float,
        runtime: Any,
        global_seq_len: int,
    ) -> torch.Tensor:
        _validate_bshd(query, key, value)
        local_rank = int(runtime.context_parallel_rank)
        group_ranks = list(runtime.context_block_parallel_group_ranks)
        num_context_ranks = len(group_ranks)
        if num_context_ranks <= 1:
            raise RuntimeError("pure context attention requires a multi-rank CP group")

        intervals_by_owner = all_context_parallel_sequence_intervals(
            seq_len=int(global_seq_len) // 2,
            context_parallel_size=num_context_ranks,
        )
        shard_lengths = owner_lengths(intervals_by_owner)
        local_len = int(shard_lengths[local_rank])
        if int(key.shape[1]) != local_len or int(value.shape[1]) != local_len:
            raise ValueError("local K/V length does not match zigzag CP ownership")
        max_shard_len = max(shard_lengths)
        current_flat, current_key, current_value = _make_kv_ring_payload(
            _pad_kv_shard_bshd(key, max_shard_len),
            _pad_kv_shard_bshd(value, max_shard_len),
        )
        initial_flat = current_flat
        backward_start_flat = torch.empty_like(initial_flat)
        scratch_flat = torch.empty_like(initial_flat) if num_context_ranks > 3 else None

        numerator, m, l = _uninitialized_shard_accumulator_bshd(query)
        initialized = False
        backward_key = None
        backward_value = None
        for step in range(num_context_ranks):
            owner = (local_rank + step) % num_context_ranks
            if step != num_context_ranks - 1:
                if step == 0:
                    recv_flat = backward_start_flat
                elif step % 2 == 1:
                    recv_flat = initial_flat
                else:
                    assert scratch_flat is not None
                    recv_flat = scratch_flat
                next_kv_work = _ring_exchange_kv_payload_async(
                    current_flat,
                    recv=recv_flat,
                    key_shape=current_key.shape,
                    key_numel=current_key.numel(),
                    local_rank=local_rank,
                    group_ranks=group_ranks,
                    group=runtime.context_block_parallel_group,
                    phase="forward",
                )
            else:
                next_kv_work = None

            offset = 0
            for key_start, key_stop in intervals_by_owner[owner]:
                chunk_len = int(key_stop - key_start)
                if chunk_len <= 0:
                    continue
                query_indices = _full_mask_interval_query_indices(
                    attn_mask=attn_mask,
                    query_len=int(query.shape[1]),
                    key_start=int(key_start),
                    key_stop=int(key_stop),
                    device=query.device,
                )
                if query_indices is not None and query_indices.numel() == 0:
                    offset += chunk_len
                    continue
                if query_indices is None:
                    shard_query = query
                    shard_blocks = attn_mask.query_blocks
                    shard_is_clean = attn_mask.query_is_clean
                    compact_indices = None
                    initial = not initialized
                else:
                    if not initialized:
                        numerator, m, l = _empty_shard_accumulator_bshd(query)
                        initialized = True
                    shard_query = query.index_select(1, query_indices)
                    shard_blocks = attn_mask.query_blocks.index_select(
                        -1,
                        query_indices,
                    )
                    shard_is_clean = attn_mask.query_is_clean.index_select(
                        -1,
                        query_indices,
                    )
                    compact_indices = query_indices
                    initial = False
                _bdlm_flex_shard_accumulate_bshd_(
                    numerator=numerator,
                    m=m,
                    l=l,
                    query=shard_query,
                    key=current_key[:, offset : offset + chunk_len],
                    value=current_value[:, offset : offset + chunk_len],
                    query_indices=compact_indices,
                    query_blocks=shard_blocks,
                    query_is_clean=shard_is_clean,
                    block_size=int(attn_mask.block_size),
                    key_start=int(_bdlm_encoded_key_start(attn_mask, key_start)),
                    clean_offset=int(attn_mask.clean_offset),
                    flex_cache=attn_mask.flex_cache,
                    scale=float(scale),
                    initial=initial,
                )
                initialized = True
                offset += chunk_len

            if step != num_context_ranks - 1:
                assert next_kv_work is not None
                current_flat, current_key, current_value = (
                    _ring_exchange_kv_payload_wait(next_kv_work)
                )
                if step == 0:
                    backward_key = current_key
                    backward_value = current_value

        if not initialized:
            raise RuntimeError("zigzag CP ring did not visit any K/V rows")
        if backward_key is None or backward_value is None:
            raise RuntimeError("zigzag CP ring did not retain the backward start shard")
        output, final_lse = _finalize_shard_stats_bshd(
            numerator=numerator,
            m=m,
            l=l,
            output_dtype=query.dtype,
        )
        ctx.runtime = runtime
        ctx.scale = float(scale)
        ctx.local_rank = local_rank
        ctx.group_ranks = group_ranks
        ctx.shard_lengths = tuple(int(length) for length in shard_lengths)
        ctx.intervals_by_owner = intervals_by_owner
        ctx.max_shard_len = int(max_shard_len)
        ctx.attn_mask = _detach_attention_mask(attn_mask)
        ctx.save_for_backward(
            query.detach(),
            backward_key.detach(),
            backward_value.detach(),
            output.detach(),
            final_lse.detach(),
        )
        return output

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
        None,
        None,
    ]:
        query, backward_key, backward_value, output, final_lse = ctx.saved_tensors
        attn_mask = ctx.attn_mask
        if not isinstance(attn_mask, BlockDenoisingFullMask):
            raise RuntimeError("pure CP backward requires a full block-denoising mask")
        grad_query, grad_key, grad_value = _ring_context_backward_bshd(
            query=query,
            local_key=backward_key,
            local_value=backward_value,
            grad_output=grad_output.contiguous(),
            final_output=output,
            final_lse=final_lse,
            attn_mask=attn_mask,
            scale=float(ctx.scale),
            local_rank=int(ctx.local_rank),
            group_ranks=list(ctx.group_ranks),
            shard_lengths=list(ctx.shard_lengths),
            intervals_by_owner=tuple(ctx.intervals_by_owner),
            max_shard_len=int(ctx.max_shard_len),
            group=ctx.runtime.context_block_parallel_group,
        )
        return (
            grad_query.to(dtype=query.dtype),
            grad_key.to(dtype=backward_key.dtype),
            grad_value.to(dtype=backward_value.dtype),
            None,
            None,
            None,
            None,
        )


class _RingWithLocalKVAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        query: torch.Tensor,
        local_key: torch.Tensor,
        local_value: torch.Tensor,
        global_key: torch.Tensor,
        global_value: torch.Tensor,
        local_attn_mask: torch.Tensor | None,
        global_attn_mask: torch.Tensor | None,
        scale: float,
        runtime: Any,
        key_chunk_size: int | None,
        global_seq_len: int,
        global_input_is_shard: bool,
        intervals_by_owner_override: tuple[tuple[tuple[int, int], ...], ...] | None,
    ) -> torch.Tensor:
        local_rank = int(runtime.context_parallel_rank)
        group_ranks = list(runtime.context_block_parallel_group_ranks)
        num_context_ranks = len(group_ranks)
        intervals_by_owner = (
            _clean_intervals_for_runtime(global_seq_len, num_context_ranks, runtime)
            if intervals_by_owner_override is None
            else intervals_by_owner_override
        )
        if len(intervals_by_owner) != num_context_ranks:
            raise ValueError("clean interval ownership must match the CP group")
        shard_lengths = owner_lengths(intervals_by_owner)
        local_intervals = intervals_by_owner[local_rank]
        local_global_len = shard_lengths[local_rank]
        if global_input_is_shard and global_key.shape[-2] != local_global_len:
            raise ValueError(
                "global_key_shard length must match this rank's context shard"
            )
        if global_input_is_shard and global_value.shape[-2] != local_global_len:
            raise ValueError(
                "global_value_shard length must match this rank's context shard"
            )
        max_shard_len = max(shard_lengths)
        local_global_key = (
            global_key.contiguous()
            if global_input_is_shard
            else _cat_intervals(global_key, local_intervals)
        )
        local_global_value = (
            global_value.contiguous()
            if global_input_is_shard
            else _cat_intervals(global_value, local_intervals)
        )
        current_flat, current_key, current_value = _make_kv_ring_payload(
            _pad_kv_shard(local_global_key, max_shard_len),
            _pad_kv_shard(local_global_value, max_shard_len),
        )

        numerator = None
        m = None
        l = None
        local_active_stats = None

        if local_key.shape[-2] > 0:
            local_active_stats = _local_active_stats(
                query=query,
                local_key=local_key,
                local_value=local_value,
                local_attn_mask=local_attn_mask,
                scale=float(scale),
            )

        for step in range(num_context_ranks):
            owner = (local_rank + step) % num_context_ranks
            shard_len = shard_lengths[owner]
            if step != num_context_ranks - 1:
                next_kv_work = _ring_exchange_kv_payload_async(
                    current_flat,
                    key_shape=current_key.shape,
                    key_numel=current_key.numel(),
                    local_rank=local_rank,
                    group_ranks=group_ranks,
                    group=runtime.context_block_parallel_group,
                    phase="forward",
                )
            else:
                next_kv_work = None

            if shard_len > 0:
                merged_stats = _owner_shard_stats_into(
                    numerator=numerator,
                    m=m,
                    l=l,
                    query=query,
                    key=current_key[..., :shard_len, :],
                    value=current_value[..., :shard_len, :],
                    owner_intervals_=intervals_by_owner[owner],
                    attn_mask=global_attn_mask,
                    is_causal=False,
                    scale=float(scale),
                )
                if merged_stats is not None:
                    numerator, m, l = merged_stats
            if step != num_context_ranks - 1:
                assert next_kv_work is not None
                current_flat, current_key, current_value = (
                    _ring_exchange_kv_payload_wait(next_kv_work)
                )

        if local_active_stats is not None:
            active_num, active_m, active_l = local_active_stats
            if numerator is None:
                numerator, m, l = _pad_active_stats(
                    active_num=active_num,
                    active_m=active_m,
                    active_l=active_l,
                    query=query,
                )
            else:
                assert m is not None and l is not None
                _merge_active_stats_in_place(
                    numerator=numerator,
                    m=m,
                    l=l,
                    active_num=active_num,
                    active_m=active_m,
                    active_l=active_l,
                )

        if numerator is None:
            output = torch.zeros_like(query)
            m = torch.full(
                query.shape[:-1],
                -torch.inf,
                dtype=torch.float32,
                device=query.device,
            )
            l = torch.zeros_like(m)
        else:
            assert m is not None and l is not None
            tiny = torch.finfo(torch.float32).tiny
            output = numerator.div_(l.clamp_min(tiny).unsqueeze(-1))
            output.masked_fill_(l.unsqueeze(-1) <= 0, 0)

        ctx.runtime = runtime
        ctx.scale = float(scale)
        ctx.key_chunk_size = key_chunk_size
        ctx.local_rank = local_rank
        ctx.group_ranks = group_ranks
        ctx.shard_lengths = shard_lengths
        ctx.intervals_by_owner = intervals_by_owner
        ctx.max_shard_len = max_shard_len
        ctx.global_input_is_shard = bool(global_input_is_shard)
        ctx.global_key_shape = (
            torch.Size((*global_key.shape[:-2], global_seq_len, global_key.shape[-1]))
            if global_input_is_shard
            else global_key.shape
        )
        ctx.global_value_shape = (
            torch.Size(
                (*global_value.shape[:-2], global_seq_len, global_value.shape[-1])
            )
            if global_input_is_shard
            else global_value.shape
        )
        ctx.has_local_mask = local_attn_mask is not None
        ctx.has_global_mask = global_attn_mask is not None
        ctx.local_attn_mask = _detach_attention_mask(local_attn_mask)
        ctx.global_attn_mask = _detach_attention_mask(global_attn_mask)
        ctx.save_for_backward(
            query.detach(),
            local_key.detach(),
            local_value.detach(),
            local_global_key.detach(),
            local_global_value.detach(),
            output.detach().to(dtype=query.dtype),
            m.detach(),
            l.detach(),
        )
        return output.to(dtype=query.dtype)

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        (
            query,
            local_key,
            local_value,
            global_key_shard,
            global_value_shard,
            output,
            m,
            l,
        ) = ctx.saved_tensors
        local_attn_mask = ctx.local_attn_mask if ctx.has_local_mask else None
        global_attn_mask = ctx.global_attn_mask if ctx.has_global_mask else None

        local_rank = int(ctx.local_rank)
        group_ranks = list(ctx.group_ranks)
        shard_lengths = list(ctx.shard_lengths)
        intervals_by_owner = tuple(ctx.intervals_by_owner)
        max_shard_len = int(ctx.max_shard_len)

        (
            grad_query,
            local_grad_key,
            local_grad_value,
            reduced_key,
            reduced_value,
        ) = _ring_hybrid_backward_flex(
            query=query,
            local_key=local_key,
            local_value=local_value,
            global_key_shard=global_key_shard,
            global_value_shard=global_value_shard,
            grad_output=grad_output,
            final_output=output,
            final_m=m,
            final_l=l,
            local_attn_mask=local_attn_mask,
            global_attn_mask=global_attn_mask,
            scale=ctx.scale,
            global_key_shape=ctx.global_key_shape,
            global_value_shape=ctx.global_value_shape,
            local_rank=local_rank,
            group_ranks=group_ranks,
            shard_lengths=shard_lengths,
            intervals_by_owner=intervals_by_owner,
            max_shard_len=max_shard_len,
            group=ctx.runtime.context_block_parallel_group,
        )
        local_len = shard_lengths[local_rank]
        if ctx.global_input_is_shard:
            global_grad_key = reduced_key[..., :local_len, :]
            global_grad_value = reduced_value[..., :local_len, :]
        else:
            global_grad_key = torch.zeros(
                ctx.global_key_shape,
                device=query.device,
                dtype=torch.float32,
            )
            global_grad_value = torch.zeros(
                ctx.global_value_shape,
                device=query.device,
                dtype=torch.float32,
            )
            _write_interval_grads(
                global_grad_key,
                reduced_key,
                intervals_by_owner[local_rank],
            )
            _write_interval_grads(
                global_grad_value,
                reduced_value,
                intervals_by_owner[local_rank],
            )
        _sync_cuda_stream_if_needed(query)
        return (
            grad_query.to(dtype=query.dtype),
            local_grad_key.to(dtype=local_key.dtype),
            local_grad_value.to(dtype=local_value.dtype),
            global_grad_key.to(dtype=global_key_shard.dtype),
            global_grad_value.to(dtype=global_value_shard.dtype),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _ReplicatedWithLocalKVAttentionBSHD(torch.autograd.Function):
    """Native-layout packed attention for replicated clean K/V."""

    @staticmethod
    def forward(
        ctx: Any,
        query: torch.Tensor,
        local_key: torch.Tensor,
        local_value: torch.Tensor,
        global_key: torch.Tensor,
        global_value: torch.Tensor,
        local_attn_mask: BlockDenoisingLocalActiveMask | None,
        global_attn_mask: BlockDenoisingGlobalCleanMask,
        scale: float,
    ) -> torch.Tensor:
        _validate_bshd(query, local_key, local_value)
        _validate_bshd(query, global_key, global_value)
        if not isinstance(global_attn_mask, BlockDenoisingGlobalCleanMask):
            raise RuntimeError(
                "replicated block-denoising attention requires a global clean mask"
            )
        if global_attn_mask.query_blocks.ndim != 1:
            raise RuntimeError(
                "replicated block-denoising attention requires batch-shared query metadata"
            )
        if local_key.shape[1] > 0 and not isinstance(
            local_attn_mask,
            BlockDenoisingLocalActiveMask,
        ):
            raise RuntimeError(
                "replicated block-denoising attention requires a local active mask"
            )

        numerator, m, l = _uninitialized_shard_accumulator_bshd(query)
        initialized = False
        if global_key.shape[1] > 0:
            _bdlm_flex_shard_accumulate_bshd_(
                numerator=numerator,
                m=m,
                l=l,
                query=query,
                key=global_key,
                value=global_value,
                query_indices=None,
                query_blocks=global_attn_mask.query_blocks,
                query_is_clean=global_attn_mask.query_is_clean,
                block_size=int(global_attn_mask.block_size),
                key_start=0,
                clean_offset=0,
                flex_cache=global_attn_mask.flex_cache,
                scale=float(scale),
                initial=True,
            )
            initialized = True

        if local_key.shape[1] > 0:
            assert isinstance(local_attn_mask, BlockDenoisingLocalActiveMask)
            active_num, active_m, active_l = _block_diagonal_active_stats_bshd(
                query=query[:, : local_key.shape[1]],
                key=local_key,
                value=local_value,
                block_size=int(local_attn_mask.block_size),
                scale=float(scale),
                causal=bool(local_attn_mask.causal),
            )
            if not initialized:
                numerator, m, l = _empty_shard_accumulator_bshd(query)
                initialized = True
            _merge_active_prefix_stats_in_place(
                numerator=numerator,
                m=m,
                l=l,
                active_num=active_num,
                active_m=active_m,
                active_l=active_l,
                attn_mask=local_attn_mask,
            )
        if not initialized:
            numerator, m, l = _empty_shard_accumulator_bshd(query)
        output, final_lse = _finalize_shard_stats_bshd(
            numerator=numerator,
            m=m,
            l=l,
            output_dtype=query.dtype,
        )
        ctx.scale = float(scale)
        ctx.has_local_mask = local_attn_mask is not None
        ctx.local_attn_mask = _detach_attention_mask(local_attn_mask)
        ctx.global_attn_mask = _detach_attention_mask(global_attn_mask)
        ctx.save_for_backward(
            query.detach(),
            local_key.detach(),
            local_value.detach(),
            global_key.detach(),
            global_value.detach(),
            output.detach(),
            final_lse.detach(),
        )
        return output

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
        None,
    ]:
        (
            query,
            local_key,
            local_value,
            global_key,
            global_value,
            output,
            final_lse,
        ) = ctx.saved_tensors
        local_attn_mask = ctx.local_attn_mask if ctx.has_local_mask else None
        global_attn_mask = ctx.global_attn_mask
        if not isinstance(global_attn_mask, BlockDenoisingGlobalCleanMask):
            raise RuntimeError("replicated backward requires a global clean mask")
        grad_output = grad_output.contiguous()

        active_len = int(local_key.shape[1])
        if active_len > 0:
            if not isinstance(local_attn_mask, BlockDenoisingLocalActiveMask):
                raise RuntimeError("replicated backward requires a local active mask")
            local_grad_query, local_grad_key, local_grad_value = (
                _block_diagonal_active_backward_bshd(
                    query=query[:, :active_len],
                    key=local_key,
                    value=local_value,
                    output=output[:, :active_len],
                    lse=final_lse[..., :active_len],
                    grad_output=grad_output[:, :active_len],
                    block_size=int(local_attn_mask.block_size),
                    scale=float(ctx.scale),
                    backward_query_chunk_size=int(
                        local_attn_mask.backward_query_chunk_size
                    ),
                    debug_nonfinite_attention=bool(
                        local_attn_mask.debug_nonfinite_attention
                    ),
                    flex_cache=local_attn_mask.flex_cache,
                    causal=bool(local_attn_mask.causal),
                )
            )
        else:
            local_grad_query = None
            local_grad_key = torch.zeros_like(local_key)
            local_grad_value = torch.zeros_like(local_value)

        if global_key.shape[1] > 0:
            global_grad_query, global_grad_key, global_grad_value = (
                _bdlm_flex_shard_backward_exact_bshd(
                    query=query,
                    key=global_key,
                    value=global_value,
                    final_output=output,
                    final_lse=final_lse,
                    grad_output=grad_output,
                    query_blocks=global_attn_mask.query_blocks,
                    query_is_clean=global_attn_mask.query_is_clean,
                    block_size=int(global_attn_mask.block_size),
                    key_start=0,
                    scale=float(ctx.scale),
                    clean_offset=0,
                    flex_cache=global_attn_mask.flex_cache,
                    debug_nonfinite_attention=bool(
                        global_attn_mask.debug_nonfinite_attention
                    ),
                )
            )
            grad_query = global_grad_query.float()
        else:
            grad_query = torch.zeros_like(query, dtype=torch.float32)
            global_grad_key = torch.zeros_like(global_key)
            global_grad_value = torch.zeros_like(global_value)
        if local_grad_query is not None:
            grad_query[:, :active_len].add_(local_grad_query.float())
        return (
            grad_query.to(dtype=query.dtype),
            local_grad_key.to(dtype=local_key.dtype),
            local_grad_value.to(dtype=local_value.dtype),
            global_grad_key.to(dtype=global_key.dtype),
            global_grad_value.to(dtype=global_value.dtype),
            None,
            None,
            None,
        )


def _ragged_prefix_bdlm_attention_bshd(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: BlockDenoisingGlobalCleanMask,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if query.ndim != 4:
        raise ValueError("query must be [B, S, H, D]")
    block_size = int(attn_mask.block_size)
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    return _RaggedPrefixBDLMAttentionBSHD.apply(
        query,
        key,
        value,
        attn_mask.query_blocks,
        attn_mask.query_is_clean,
        block_size,
        float(scale),
    )


class _RaggedPrefixBDLMAttentionBSHD(torch.autograd.Function):
    """Per-block prefix FlexAttention with dense gathered-cache gradients."""

    @staticmethod
    def forward(
        ctx: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_blocks: torch.Tensor,
        query_is_clean: torch.Tensor,
        block_size: int,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_blocks = query_blocks.to(
            device=query.device,
            dtype=torch.int32,
        ).contiguous()
        query_is_clean = query_is_clean.to(
            device=query.device,
            dtype=torch.bool,
        ).contiguous()
        query_len = int(query.shape[1])
        if query_blocks.numel() < query_len or query_is_clean.numel() < query_len:
            raise ValueError("BDLM query metadata is shorter than query length")
        output, lse = _bdlm_flex_shard_attention_bshd(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            query_blocks,
            query_is_clean,
            int(block_size),
            0,
            float(scale),
            clean_offset=0,
        )
        ctx.save_for_backward(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            query_blocks,
            query_is_clean,
            output,
            lse,
        )
        ctx.block_size = int(block_size)
        ctx.scale = float(scale)
        return output, lse

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
        grad_lse: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None, None]:
        (
            query,
            key,
            value,
            query_blocks,
            query_is_clean,
            output,
            lse,
        ) = ctx.saved_tensors
        block_size = int(ctx.block_size)
        scale = float(ctx.scale)
        grad_query, grad_key, grad_value = _bdlm_flex_shard_backward_bshd(
            query,
            key,
            value,
            output,
            lse,
            grad_output,
            grad_lse,
            query_blocks,
            query_is_clean,
            block_size,
            0,
            scale,
            clean_offset=0,
            flex_cache=None,
            debug_nonfinite_attention=False,
        )

        return (
            grad_query.to(dtype=query.dtype),
            grad_key.to(dtype=key.dtype),
            grad_value.to(dtype=value.dtype),
            None,
            None,
            None,
            None,
        )


def _owner_shard_stats_into(
    *,
    numerator: torch.Tensor | None,
    m: torch.Tensor | None,
    l: torch.Tensor | None,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    owner_intervals_: tuple[tuple[int, int], ...],
    attn_mask: Any | None,
    is_causal: bool,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    offset = 0
    for start, stop in owner_intervals_:
        shard_len = stop - start
        if shard_len <= 0:
            continue
        shard_key = key[..., offset : offset + shard_len, :]
        shard_value = value[..., offset : offset + shard_len, :]
        query_indices = _owner_query_indices(
            attn_mask=attn_mask,
            is_causal=is_causal,
            query_len=query.shape[-2],
            key_start=start,
            device=query.device,
        )
        if query_indices is not None and query_indices.numel() == 0:
            offset += shard_len
            continue
        if (
            isinstance(
                attn_mask, (BlockDenoisingGlobalCleanMask, BlockDenoisingFullMask)
            )
            and not is_causal
        ):
            if query_indices is None:
                initial = numerator is None
                if initial:
                    numerator, m, l = _uninitialized_shard_accumulator(query=query)
                assert numerator is not None and m is not None and l is not None
                _bdlm_flex_shard_accumulate_(
                    numerator=numerator,
                    m=m,
                    l=l,
                    query=query,
                    key=shard_key,
                    value=shard_value,
                    query_indices=None,
                    query_blocks=attn_mask.query_blocks,
                    query_is_clean=attn_mask.query_is_clean,
                    block_size=int(attn_mask.block_size),
                    key_start=int(_bdlm_encoded_key_start(attn_mask, start)),
                    clean_offset=int(_bdlm_clean_offset(attn_mask)),
                    flex_cache=attn_mask.flex_cache,
                    scale=float(scale),
                    initial=initial,
                )
            elif isinstance(attn_mask, BlockDenoisingGlobalCleanMask):
                compact_query, compact_blocks, compact_is_clean = (
                    _compact_bdlm_query_metadata(
                        query=query,
                        attn_mask=attn_mask,
                        query_indices=query_indices,
                    )
                )
                if numerator is None:
                    numerator, m, l = _empty_shard_accumulator(
                        query=query,
                        m_dtype=torch.float32,
                    )
                assert numerator is not None and m is not None and l is not None
                _bdlm_flex_shard_accumulate_(
                    numerator=numerator,
                    m=m,
                    l=l,
                    query=compact_query,
                    key=shard_key,
                    value=shard_value,
                    query_indices=query_indices,
                    query_blocks=compact_blocks,
                    query_is_clean=compact_is_clean,
                    block_size=int(attn_mask.block_size),
                    key_start=int(_bdlm_encoded_key_start(attn_mask, start)),
                    clean_offset=int(_bdlm_clean_offset(attn_mask)),
                    flex_cache=attn_mask.flex_cache,
                    scale=float(scale),
                    initial=False,
                )
            else:
                raise RuntimeError(
                    "query compaction is only supported for Block-denoising global-clean masks"
                )
        elif query_indices is None:
            shard_num, shard_m, shard_l = _fused_shard_stats(
                query=query,
                key=shard_key,
                value=shard_value,
                attn_mask=attn_mask,
                is_causal=is_causal,
                scale=float(scale),
                key_start=start,
            )
            if numerator is None:
                numerator = shard_num
                m = shard_m
                l = shard_l
            else:
                assert m is not None and l is not None
                numerator, m, l = _merge_online_stats(
                    numerator,
                    m,
                    l,
                    shard_num,
                    shard_m,
                    shard_l,
                )
        else:
            raise RuntimeError(
                "query compaction is only supported for Block-denoising global-clean masks"
            )
        offset += shard_len
    if numerator is None:
        return None
    assert m is not None and l is not None
    return numerator, m, l


def _owner_query_indices(
    *,
    attn_mask: Any | None,
    is_causal: bool,
    query_len: int,
    key_start: int,
    device: torch.device,
) -> torch.Tensor | None:
    if is_causal or not isinstance(attn_mask, BlockDenoisingGlobalCleanMask):
        return None
    cache_key = ("owner_query_indices", int(query_len), int(key_start), str(device))
    cache = getattr(attn_mask, "flex_cache", None)
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    query_blocks = attn_mask.query_blocks[:query_len].to(
        device=device,
        dtype=torch.int32,
    )
    query_is_clean = attn_mask.query_is_clean[:query_len].to(
        device=device,
        dtype=torch.bool,
    )
    first_key_block = int(key_start) // int(attn_mask.block_size)
    valid = torch.where(
        query_is_clean,
        query_blocks >= first_key_block,
        query_blocks > first_key_block,
    )
    indices = torch.nonzero(valid, as_tuple=False).flatten().to(dtype=torch.long)
    result = None if indices.numel() == int(query_len) else indices
    if cache is not None:
        cache[cache_key] = result
    return result


def _full_mask_interval_query_indices(
    *,
    attn_mask: BlockDenoisingFullMask,
    query_len: int,
    key_start: int,
    key_stop: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Return queries with at least one legal key in a full-mask interval."""

    if int(key_stop) < int(key_start):
        raise ValueError("key_stop must not precede key_start")
    cache_key = (
        "full_mask_interval_query_indices",
        int(query_len),
        int(key_start),
        int(key_stop),
        str(device),
    )
    cache = attn_mask.flex_cache
    if cache_key in cache:
        return cache[cache_key]

    query_blocks = attn_mask.query_blocks[:query_len].to(
        device=device,
        dtype=torch.int32,
    )
    query_is_clean = attn_mask.query_is_clean[:query_len].to(
        device=device,
        dtype=torch.bool,
    )
    block_size = int(attn_mask.block_size)
    clean_offset = int(attn_mask.clean_offset)
    valid = torch.zeros_like(query_is_clean)

    active_start = max(int(key_start), 0)
    active_stop = min(int(key_stop), clean_offset)
    if active_stop > active_start:
        first_active_block = active_start // block_size
        last_active_block = (active_stop - 1) // block_size
        valid |= (
            ~query_is_clean
            & (query_blocks >= first_active_block)
            & (query_blocks <= last_active_block)
        )

    clean_start = max(int(key_start), clean_offset)
    if int(key_stop) > clean_start:
        first_clean_block = (clean_start - clean_offset) // block_size
        valid |= torch.where(
            query_is_clean,
            query_blocks >= first_clean_block,
            query_blocks > first_clean_block,
        )

    indices = torch.nonzero(valid, as_tuple=False).flatten().to(dtype=torch.long)
    result = None if indices.numel() == int(query_len) else indices
    cache[cache_key] = result
    return result


def _compacted_bdlm_shard_attention(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: BlockDenoisingGlobalCleanMask | BlockDenoisingFullMask,
    query_indices: torch.Tensor,
    scale: float,
    key_start: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    compact_query = query.index_select(-2, query_indices)
    compact_blocks = attn_mask.query_blocks.to(
        device=query.device,
        dtype=torch.int32,
    ).index_select(0, query_indices)
    compact_is_clean = attn_mask.query_is_clean.to(
        device=query.device,
        dtype=torch.bool,
    ).index_select(0, query_indices)
    return _bdlm_flex_shard_attention(
        compact_query,
        key,
        value,
        compact_blocks,
        compact_is_clean,
        int(attn_mask.block_size),
        int(_bdlm_encoded_key_start(attn_mask, key_start)),
        float(scale),
        clean_offset=int(_bdlm_clean_offset(attn_mask)),
        flex_cache=attn_mask.flex_cache,
    )


def _compact_bdlm_query_metadata(
    *,
    query: torch.Tensor,
    attn_mask: BlockDenoisingGlobalCleanMask,
    query_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    compact_query = query.index_select(-2, query_indices)
    compact_blocks = attn_mask.query_blocks.to(
        device=query.device,
        dtype=torch.int32,
    ).index_select(0, query_indices)
    compact_is_clean = attn_mask.query_is_clean.to(
        device=query.device,
        dtype=torch.bool,
    ).index_select(0, query_indices)
    return compact_query, compact_blocks, compact_is_clean


def _uninitialized_shard_accumulator(
    *,
    query: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    numerator = torch.empty(
        query.shape,
        dtype=torch.float32,
        device=query.device,
    )
    m = torch.empty(
        query.shape[:-1],
        dtype=torch.float32,
        device=query.device,
    )
    l = torch.empty_like(m)
    return numerator, m, l


def _uninitialized_shard_accumulator_bshd(
    query: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, query_len, heads, head_dim = query.shape
    numerator = torch.empty(
        (batch, heads, query_len, head_dim),
        dtype=torch.float32,
        device=query.device,
    )
    m = torch.empty(
        (batch, heads, query_len),
        dtype=torch.float32,
        device=query.device,
    )
    return numerator, m, torch.empty_like(m)


def _empty_shard_accumulator_bshd(
    query: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, query_len, heads, head_dim = query.shape
    numerator = torch.zeros(
        (batch, heads, query_len, head_dim),
        dtype=torch.float32,
        device=query.device,
    )
    m = torch.full(
        (batch, heads, query_len),
        -torch.inf,
        dtype=torch.float32,
        device=query.device,
    )
    return numerator, m, torch.zeros_like(m)


def _finalize_shard_stats_bshd(
    *,
    numerator: torch.Tensor,
    m: torch.Tensor,
    l: torch.Tensor,
    output_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty(
        (
            numerator.shape[0],
            numerator.shape[2],
            numerator.shape[1],
            numerator.shape[3],
        ),
        dtype=output_dtype,
        device=numerator.device,
    )
    final_lse = torch.empty_like(m)
    cp_fusion.finalize_bshd_(numerator, m, l, output, final_lse)
    return output, final_lse


def _empty_shard_accumulator(
    *,
    query: torch.Tensor,
    m_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    numerator = torch.zeros(
        query.shape,
        dtype=torch.float32,
        device=query.device,
    )
    m = torch.full(
        query.shape[:-1],
        -torch.inf,
        dtype=m_dtype,
        device=query.device,
    )
    l = torch.zeros_like(m)
    return numerator, m, l


def _bdlm_flex_shard_accumulate_(
    *,
    numerator: torch.Tensor,
    m: torch.Tensor,
    l: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_indices: torch.Tensor | None,
    query_blocks: torch.Tensor,
    query_is_clean: torch.Tensor,
    block_size: int,
    key_start: int,
    clean_offset: int,
    flex_cache: dict[tuple[Any, ...], Any],
    scale: float,
    initial: bool,
) -> None:
    if not query.is_cuda:
        raise RuntimeError("block-denoising FlexAttention requires CUDA tensors")
    attn_mask, logical_key_start = _bdlm_flex_mask(
        query_blocks=query_blocks,
        query_is_clean=query_is_clean,
        block_size=int(block_size),
        encoded_key_start=int(key_start),
        clean_offset=int(clean_offset),
        flex_cache=flex_cache,
    )
    shard_output, shard_lse, _ = _flex_shard_stats(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        is_causal=False,
        scale=float(scale),
        key_start=int(logical_key_start),
        output_float=False,
    )
    _merge_flex_shard_stats_(
        numerator=numerator,
        m=m,
        l=l,
        shard_output=shard_output,
        shard_lse=shard_lse,
        query_indices=query_indices,
        initial=bool(initial),
    )


def _bdlm_flex_shard_accumulate_bshd_(
    *,
    numerator: torch.Tensor,
    m: torch.Tensor,
    l: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_indices: torch.Tensor | None,
    query_blocks: torch.Tensor,
    query_is_clean: torch.Tensor,
    block_size: int,
    key_start: int,
    clean_offset: int,
    flex_cache: dict[tuple[Any, ...], Any],
    scale: float,
    initial: bool,
) -> None:
    if not query.is_cuda:
        raise RuntimeError("block-denoising FlexAttention requires CUDA tensors")
    attn_mask, logical_key_start = _bdlm_flex_mask(
        query_blocks=query_blocks,
        query_is_clean=query_is_clean,
        block_size=int(block_size),
        encoded_key_start=int(key_start),
        clean_offset=int(clean_offset),
        flex_cache=flex_cache,
    )
    shard_output, shard_lse, _ = _flex_shard_stats(
        query=query.transpose(1, 2),
        key=key.transpose(1, 2),
        value=value.transpose(1, 2),
        attn_mask=attn_mask,
        is_causal=False,
        scale=float(scale),
        key_start=int(logical_key_start),
        output_float=False,
        native_fa4=True,
    )
    _merge_flex_shard_stats_(
        numerator=numerator,
        m=m,
        l=l,
        shard_output=shard_output,
        shard_lse=shard_lse,
        query_indices=query_indices,
        initial=bool(initial),
    )


def _compacted_bdlm_shard_backward(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    final_output: torch.Tensor,
    final_lse: torch.Tensor,
    grad_output: torch.Tensor,
    attn_mask: BlockDenoisingGlobalCleanMask | BlockDenoisingFullMask,
    query_indices: torch.Tensor,
    key_start: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    debug_nonfinite_attention = bool(
        getattr(attn_mask, "debug_nonfinite_attention", False)
    )
    compact_query = query.index_select(-2, query_indices)
    compact_output, compact_lse = _compacted_bdlm_shard_attention(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        query_indices=query_indices,
        scale=float(scale),
        key_start=key_start,
    )
    compact_grad_output, compact_grad_lse = _merged_shard_grads(
        shard_output=compact_output,
        shard_lse=compact_lse,
        final_output=final_output.index_select(-2, query_indices),
        final_lse=final_lse.index_select(-1, query_indices),
        grad_output=grad_output.index_select(-2, query_indices),
    )
    compact_blocks = attn_mask.query_blocks.to(
        device=query.device,
        dtype=torch.int32,
    ).index_select(0, query_indices)
    compact_is_clean = attn_mask.query_is_clean.to(
        device=query.device,
        dtype=torch.bool,
    ).index_select(0, query_indices)
    return _bdlm_flex_shard_backward(
        compact_query,
        key,
        value,
        compact_output.to(dtype=query.dtype),
        compact_lse,
        compact_grad_output,
        compact_grad_lse,
        compact_blocks,
        compact_is_clean,
        int(attn_mask.block_size),
        int(_bdlm_encoded_key_start(attn_mask, key_start)),
        float(scale),
        clean_offset=int(_bdlm_clean_offset(attn_mask)),
        flex_cache=attn_mask.flex_cache,
        debug_nonfinite_attention=debug_nonfinite_attention,
    )


def _bdlm_backward_query_chunk_size(
    block_size: int,
    configured_chunk_size: int,
    *,
    query: torch.Tensor | None = None,
) -> int:
    del query
    chunk_size = int(configured_chunk_size)
    if chunk_size <= 0:
        return 0
    block_size = max(1, int(block_size))
    return max(block_size, (chunk_size // block_size) * block_size)


def _query_index_chunks(
    query_indices: torch.Tensor,
    *,
    query: torch.Tensor,
    block_size: int,
    backward_query_chunk_size: int = 0,
    head_dim: int | None = None,
) -> list[torch.Tensor]:
    del head_dim
    chunk_size = _bdlm_backward_query_chunk_size(
        block_size,
        backward_query_chunk_size,
        query=query,
    )
    if chunk_size <= 0 or int(query_indices.numel()) <= chunk_size:
        return [query_indices.contiguous()]
    return [
        query_indices[start : start + chunk_size].contiguous()
        for start in range(0, int(query_indices.numel()), chunk_size)
    ]


def _bdlm_flex_shard_backward_exact(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    final_output: torch.Tensor,
    final_lse: torch.Tensor,
    grad_output: torch.Tensor,
    query_blocks: torch.Tensor,
    query_is_clean: torch.Tensor,
    block_size: int,
    backward_query_chunk_size: int,
    key_start: int,
    scale: float,
    clean_offset: int = 0,
    flex_cache: dict[tuple[Any, ...], Any] | None = None,
    debug_nonfinite_attention: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del backward_query_chunk_size
    attn_mask, logical_key_start = _bdlm_flex_mask(
        query_blocks=query_blocks,
        query_is_clean=query_is_clean,
        block_size=int(block_size),
        encoded_key_start=int(key_start),
        clean_offset=int(clean_offset),
        flex_cache=flex_cache,
    )
    grad_query, grad_key, grad_value = _flex_merged_shard_backward(
        query=query,
        key=key,
        value=value,
        final_output=final_output,
        final_lse=final_lse,
        grad_output=grad_output,
        attn_mask=attn_mask,
        is_causal=False,
        scale=float(scale),
        key_start=int(logical_key_start),
    )
    _debug_check_backward_finite(
        "bdlm_flex_shard_exact",
        grad_query,
        grad_key,
        grad_value,
        query=query,
        key=key,
        key_start=int(key_start),
        block_size=int(block_size),
        debug_nonfinite_attention=bool(debug_nonfinite_attention),
    )
    return grad_query, grad_key, grad_value


def _bdlm_shard_backward_torch(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    final_output: torch.Tensor,
    final_lse: torch.Tensor,
    grad_output: torch.Tensor,
    query_blocks: torch.Tensor,
    query_is_clean: torch.Tensor,
    block_size: int,
    key_start: int,
    clean_offset: int = 0,
    scale: float,
    debug_nonfinite_attention: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, query_heads, query_len, head_dim = query.shape
    kv_heads = key.shape[1]
    if query_heads % kv_heads != 0:
        raise RuntimeError("query heads must be divisible by key/value heads")
    group_size = query_heads // kv_heads
    key_len = key.shape[-2]
    query_f = query.float()
    key_f = key.float()
    value_f = value.float()
    expanded_key = key_f.repeat_interleave(group_size, dim=1)
    expanded_value = value_f.repeat_interleave(group_size, dim=1)
    scores = torch.matmul(query_f, expanded_key.transpose(-2, -1)) * float(scale)
    logical_key_start, full_mask = _decode_bdlm_key_start(key_start)
    global_kv_positions = (
        torch.arange(key_len, device=query.device, dtype=torch.int32)
        + logical_key_start
    )
    query_blocks = query_blocks.to(device=query.device, dtype=torch.int32)
    query_is_clean = query_is_clean.to(device=query.device, dtype=torch.bool)
    if full_mask:
        if int(clean_offset) <= 0:
            raise ValueError(
                "full block-denoising masks require a positive clean_offset"
            )
        kv_is_clean = global_kv_positions >= int(clean_offset)
        kv_positions = torch.where(
            kv_is_clean,
            global_kv_positions - int(clean_offset),
            global_kv_positions,
        )
        kv_blocks = kv_positions // int(block_size)
        allowed = torch.where(
            query_is_clean[:, None],
            kv_is_clean[None, :] & (query_blocks[:, None] >= kv_blocks[None, :]),
            ((~kv_is_clean[None, :]) & (query_blocks[:, None] == kv_blocks[None, :]))
            | (kv_is_clean[None, :] & (query_blocks[:, None] > kv_blocks[None, :])),
        )
    else:
        kv_blocks = global_kv_positions // int(block_size)
        allowed = torch.where(
            query_is_clean[:, None],
            query_blocks[:, None] >= kv_blocks[None, :],
            query_blocks[:, None] > kv_blocks[None, :],
        )
    scores = scores.masked_fill(~allowed[None, None, :, :], -torch.inf)
    shard_lse = torch.logsumexp(scores, dim=-1)
    probs = torch.softmax(scores, dim=-1)
    probs = torch.where(
        torch.isfinite(shard_lse).unsqueeze(-1),
        probs,
        torch.zeros_like(probs),
    )
    shard_output = torch.matmul(probs, expanded_value)
    shard_grad_output, shard_grad_lse = _merged_shard_grads(
        shard_output=shard_output.to(dtype=query.dtype),
        shard_lse=shard_lse,
        final_output=final_output,
        final_lse=final_lse,
        grad_output=grad_output,
    )
    shard_grad_output = shard_grad_output.float()
    grad_value_expanded = torch.matmul(probs.transpose(-2, -1), shard_grad_output)
    grad_probs = torch.matmul(shard_grad_output, expanded_value.transpose(-2, -1))
    row_dot = (shard_grad_output * shard_output).sum(dim=-1, keepdim=True)
    grad_scores = probs * (grad_probs - row_dot + shard_grad_lse.float().unsqueeze(-1))
    grad_query = torch.matmul(grad_scores, expanded_key) * float(scale)
    grad_key_expanded = torch.matmul(grad_scores.transpose(-2, -1), query_f) * float(
        scale
    )
    grad_key = grad_key_expanded.reshape(
        batch, kv_heads, group_size, key_len, head_dim
    ).sum(dim=2)
    grad_value = grad_value_expanded.reshape(
        batch, kv_heads, group_size, key_len, head_dim
    ).sum(dim=2)
    _debug_check_backward_finite(
        "bdlm_torch",
        grad_query,
        grad_key,
        grad_value,
        query=query,
        key=key,
        key_start=int(key_start),
        block_size=int(block_size),
        debug_nonfinite_attention=bool(debug_nonfinite_attention),
    )
    return grad_query, grad_key, grad_value


def _bdlm_shard_stats_torch(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_blocks: torch.Tensor,
    query_is_clean: torch.Tensor,
    block_size: int,
    key_start: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, query_heads, query_len, _ = query.shape
    kv_heads = key.shape[1]
    if query_heads % kv_heads != 0:
        raise RuntimeError("query heads must be divisible by key/value heads")
    group_size = query_heads // kv_heads
    key_len = key.shape[-2]
    query_f = query.float()
    key_f = key.float()
    value_f = value.float()
    expanded_key = key_f.repeat_interleave(group_size, dim=1)
    expanded_value = value_f.repeat_interleave(group_size, dim=1)
    scores = torch.matmul(query_f, expanded_key.transpose(-2, -1)) * float(scale)
    kv_blocks = (
        torch.arange(key_len, device=query.device, dtype=torch.int32) + int(key_start)
    ) // int(block_size)
    query_blocks = query_blocks[:query_len].to(device=query.device, dtype=torch.int32)
    query_is_clean = query_is_clean[:query_len].to(
        device=query.device,
        dtype=torch.bool,
    )
    allowed = torch.where(
        query_is_clean[:, None],
        query_blocks[:, None] >= kv_blocks[None, :],
        query_blocks[:, None] > kv_blocks[None, :],
    )
    scores = scores.masked_fill(~allowed[None, None, :, :], -torch.inf)
    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.softmax(scores, dim=-1)
    probs = torch.where(
        torch.isfinite(lse).unsqueeze(-1),
        probs,
        torch.zeros_like(probs),
    )
    output = torch.matmul(probs, expanded_value)
    l = torch.where(torch.isfinite(lse), torch.ones_like(lse), torch.zeros_like(lse))
    return output.to(dtype=query.dtype), lse, l


def _compacted_bdlm_shard_backward_exact(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    final_output: torch.Tensor,
    final_lse: torch.Tensor,
    grad_output: torch.Tensor,
    attn_mask: BlockDenoisingGlobalCleanMask | BlockDenoisingFullMask,
    query_indices: torch.Tensor,
    key_start: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    chunks = _query_index_chunks(
        query_indices,
        query=query,
        block_size=int(attn_mask.block_size),
        backward_query_chunk_size=int(attn_mask.backward_query_chunk_size),
        head_dim=int(query.shape[-1]),
    )
    if len(chunks) == 1:
        return _compacted_bdlm_shard_backward(
            query=query,
            key=key,
            value=value,
            final_output=final_output,
            final_lse=final_lse,
            grad_output=grad_output,
            attn_mask=attn_mask,
            query_indices=query_indices,
            key_start=key_start,
            scale=float(scale),
        )

    grad_q_chunks = []
    grad_key = torch.zeros_like(key, dtype=torch.float32)
    grad_value = torch.zeros_like(value, dtype=torch.float32)
    for chunk_indices in chunks:
        chunk_grad_q, chunk_grad_k, chunk_grad_v = _compacted_bdlm_shard_backward(
            query=query,
            key=key,
            value=value,
            final_output=final_output,
            final_lse=final_lse,
            grad_output=grad_output,
            attn_mask=attn_mask,
            query_indices=chunk_indices,
            key_start=key_start,
            scale=float(scale),
        )
        grad_q_chunks.append(chunk_grad_q.float())
        grad_key.add_(chunk_grad_k.float())
        grad_value.add_(chunk_grad_v.float())
    return torch.cat(grad_q_chunks, dim=-2), grad_key, grad_value


def _nonzero_active_grad_block_indices(
    *,
    grad_output: torch.Tensor,
    grad_lse: torch.Tensor,
    block_size: int,
) -> torch.Tensor | None:
    active_len = grad_output.shape[-2]
    if active_len % int(block_size) != 0:
        raise RuntimeError("active gradient length must be block aligned")
    row_active = grad_output.detach().ne(0).any(dim=-1).any(dim=0).any(
        dim=0
    ) | grad_lse.detach().ne(0).any(dim=0).any(dim=0)
    block_active = row_active.view(active_len // int(block_size), int(block_size)).any(
        dim=1
    )
    indices = torch.nonzero(block_active, as_tuple=False).flatten().to(dtype=torch.long)
    return None if int(indices.numel()) == int(block_active.numel()) else indices


def _block_token_indices(
    block_indices: torch.Tensor,
    *,
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    offsets = torch.arange(int(block_size), device=device, dtype=torch.long)
    return (
        block_indices.to(device=device, dtype=torch.long)[:, None] * int(block_size)
        + offsets[None]
    ).reshape(-1)


def _fused_shard_stats(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Any | None,
    is_causal: bool,
    scale: float,
    key_start: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _flex_shard_stats(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        is_causal=is_causal,
        scale=float(scale),
        key_start=key_start,
    )


def _uses_native_wide_masked_attention(
    query: torch.Tensor,
    attn_mask: Any,
) -> bool:
    """Return whether the packaged masked Split-D shape is production-safe."""

    if not _uses_native_wide_attention(query):
        return False
    # Retain a per-mask escape hatch for layouts that have not passed the
    # production-shape native-kernel gate.
    return bool(getattr(attn_mask, "allow_native_wide_attention", True))


def _wide_shard_stats(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Any | None,
    is_causal: bool,
    scale: float,
    key_start: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if bool(is_causal):
        raise RuntimeError("wide-head BDLM attention does not support causal shards")
    if attn_mask is None:
        query_bshd = query.transpose(1, 2).contiguous()
        key_bshd = key.transpose(1, 2).contiguous()
        value_bshd = value.transpose(1, 2).contiguous()
        output, lse = _wide_full_attention_bshd(
            query_bshd,
            key_bshd,
            value_bshd,
            float(scale),
        )
    # Every supported BDLM mask has an exact interval representation.  On
    # Hopper this lets contiguous CP shards and packed fused shards share the
    # packaged Split-D kernel instead of sending masks without explicit input
    # intervals through the substantially slower D=512 FlexAttention path.
    elif isinstance(
        attn_mask,
        (
            BlockDenoisingLocalActiveMask,
            BlockDenoisingGlobalCleanMask,
            BlockDenoisingFullMask,
            BlockDenoisingPackedKeyMask,
        ),
    ):
        (
            query_blocks,
            local_query_blocks,
            query_is_clean,
            key_blocks,
            key_is_clean,
        ) = _flex_backward_mask_buffers(
            attn_mask=attn_mask,
            key_start=int(key_start),
            key_len=int(key.shape[-2]),
            device=query.device,
        )
        buffers = (
            query_blocks,
            local_query_blocks,
            query_is_clean,
            key_blocks,
            key_is_clean,
        )
        native_wide = _uses_native_wide_masked_attention(query, attn_mask)
        plan = _wide_metadata_plan(
            attn_mask=attn_mask,
            buffers=buffers,
            key_start=int(key_start),
            query_len=int(query.shape[-2]),
            key_len=int(key.shape[-2]),
            query_heads=int(query.shape[1]),
            key_heads=int(key.shape[1]),
            native=bool(native_wide),
            device=query.device,
        )
        if native_wide:
            output, lse = _wide_interval_forward_bhsd(
                query,
                key,
                value,
                *buffers,
                float(scale),
                plan,
            )
        else:
            output, lse = _wide_metadata_forward_bhsd(
                query,
                key,
                value,
                *_wide_execution_mask_buffers(attn_mask, buffers),
                float(scale),
                plan,
            )
    else:
        raise RuntimeError(
            f"wide-head attention does not support mask type {type(attn_mask)!r}"
        )
    if attn_mask is None:
        output = output.transpose(1, 2)
    return output, lse


def _should_use_native_fa4_forward(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    requested: bool,
) -> bool:
    if not bool(requested):
        return False
    if int(query.shape[-1]) != 256 or int(query.shape[-2]) != 32:
        return True
    # On H100 at DiffusionGemma's Q16/KV8 shape, FA4 wins through K=544;
    # compiled Flex wins at the validated K=1056 and longer shapes.
    return int(key.shape[-2]) <= _FA4_D256_Q32_MAX_KEY_LENGTH


def _flex_shard_stats(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Any | None,
    is_causal: bool,
    scale: float,
    key_start: int,
    output_float: bool = True,
    native_fa4: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _require_flex_attention(query, key, attn_mask)
    if AuxRequest is None:
        raise RuntimeError("BDLM FlexAttention requires torch>=2.10")
    if _is_wide_head_dim(int(query.shape[-1])):
        output, lse = _wide_shard_stats(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            is_causal=bool(is_causal),
            scale=float(scale),
            key_start=int(key_start),
        )
        finite = torch.isfinite(lse)
        l = torch.where(finite, torch.ones_like(lse), torch.zeros_like(lse))
        return (output.float() if bool(output_float) else output), lse, l
    native_fa4 = _should_use_native_fa4_forward(
        query=query,
        key=key,
        requested=bool(native_fa4),
    )
    kernel_family = _flex_kernel_family(attn_mask)
    block_size = None
    if native_fa4:
        block_size = fa4_forward_sparse_tile(
            head_dim=int(query.shape[-1]),
            head_dim_v=int(value.shape[-1]),
            sparse_query_block_size=128,
        )
    score_mod, block_mask = _make_flex_mods(
        attn_mask=attn_mask,
        is_causal=is_causal,
        query=query,
        key_start=key_start,
        key_len=key.shape[-2],
        build_block_mask=_flex_should_build_block_mask(query=query, key=key),
        block_size=block_size,
    )
    if native_fa4:
        if block_mask is None:
            raise RuntimeError("native FA4 forward requires a block mask")
        kv_head_repeat = _native_fa4_d256_forward_kv_head_repeat(query, key)
        if kv_head_repeat > 1:
            key = key.repeat_interleave(kv_head_repeat, dim=1)
            value = value.repeat_interleave(kv_head_repeat, dim=1)
        output, lse = fa4_forward(
            query=query,
            key=key,
            value=value,
            block_mask=block_mask.as_tuple(),
            mask_buffers=_flex_backward_mask_buffers(
                attn_mask=attn_mask,
                key_start=int(key_start),
                key_len=int(key.shape[-2]),
                device=query.device,
            ),
            scale=float(scale),
        )
    else:
        output, aux = _get_compiled_flex_attention(
            kernel_family=kernel_family,
            differentiable=torch.is_grad_enabled(),
        )(
            query,
            key,
            value,
            score_mod,
            block_mask,
            scale,
            bool(query.shape[1] != key.shape[1]),
            AuxRequest(lse=True),
            _flex_kernel_options(attn_mask=attn_mask),
        )
        if aux.lse is None:
            raise RuntimeError("BDLM FlexAttention did not return LSE statistics")
        lse = aux.lse
    finite = torch.isfinite(lse)
    l = torch.where(finite, torch.ones_like(lse), torch.zeros_like(lse))
    return (output.float() if bool(output_float) else output), lse, l


def _flex_kernel_options(
    *,
    attn_mask: Any | None,
) -> dict[str, bool] | None:
    if isinstance(attn_mask, BlockDenoisingLocalActiveMask):
        return {"BLOCKS_ARE_CONTIGUOUS": True}
    if (
        isinstance(attn_mask, BlockDenoisingGlobalCleanMask)
        and attn_mask.clean_key_blocks is None
    ):
        return {"BLOCKS_ARE_CONTIGUOUS": True}
    return None


def _packed_clean_owner_mask(
    attn_mask: BlockDenoisingGlobalCleanMask,
    *,
    intervals: tuple[tuple[int, int], ...],
    device: torch.device,
) -> tuple[BlockDenoisingGlobalCleanMask, int]:
    intervals = tuple(
        (int(start), int(stop)) for start, stop in intervals if int(stop) > int(start)
    )
    if not intervals or len(intervals) > 2:
        raise ValueError("packed clean ownership requires one or two intervals")
    cache_key = ("packed_clean_owner_mask", intervals, device.type, device.index)
    cached = attn_mask.flex_cache.get(cache_key)
    if (
        isinstance(cached, tuple)
        and len(cached) == 2
        and isinstance(cached[0], BlockDenoisingGlobalCleanMask)
    ):
        return cached
    first_start, first_stop = intervals[0]
    clean_positions = torch.cat(
        [
            torch.arange(start, stop, device=device, dtype=torch.int32)
            for start, stop in intervals
        ]
    )
    first_key_length = None
    second_key_start = None
    if len(intervals) == 2:
        first_key_length = torch.scalar_tensor(
            first_stop - first_start,
            device=device,
            dtype=torch.int32,
        )
        second_key_start = torch.scalar_tensor(
            intervals[1][0],
            device=device,
            dtype=torch.int32,
        )
    packed = BlockDenoisingGlobalCleanMask(
        query_blocks=attn_mask.query_blocks,
        query_is_clean=attn_mask.query_is_clean,
        block_size=int(attn_mask.block_size),
        clean_context_window=attn_mask.clean_context_window,
        query_clean_bounds=attn_mask.query_clean_bounds,
        clean_key_positions=(
            clean_positions if attn_mask.query_clean_bounds is not None else None
        ),
        first_key_length=first_key_length,
        second_key_start=second_key_start,
        backward_query_chunk_size=int(attn_mask.backward_query_chunk_size),
        debug_nonfinite_attention=bool(attn_mask.debug_nonfinite_attention),
        allow_native_wide_attention=bool(attn_mask.allow_native_wide_attention),
    )
    result = (packed, first_start)
    attn_mask.flex_cache[cache_key] = result
    return result


def _packed_local_owner_mask(
    global_attn_mask: BlockDenoisingGlobalCleanMask,
    local_attn_mask: BlockDenoisingLocalActiveMask,
    *,
    intervals: tuple[tuple[int, int], ...],
    device: torch.device,
) -> BlockDenoisingPackedKeyMask:
    block_size = int(global_attn_mask.block_size)
    if int(local_attn_mask.block_size) != block_size:
        raise ValueError("local and clean block sizes must match")
    clean_blocks: list[torch.Tensor] = []
    clean_positions: list[torch.Tensor] = []
    normalized_intervals: list[tuple[int, int]] = []
    for start, stop in intervals:
        start = int(start)
        stop = int(stop)
        if stop <= start:
            continue
        normalized_intervals.append((start, stop))
        positions = torch.arange(start, stop, device=device, dtype=torch.int32)
        clean_positions.append(positions)
        clean_blocks.append(positions // block_size)
    if not clean_blocks:
        raise ValueError("local packed attention requires a non-empty clean shard")
    cache_key = (
        "packed_local_owner_mask",
        tuple(normalized_intervals),
        device.type,
        device.index,
    )
    cached = local_attn_mask.flex_cache.get(cache_key)
    if isinstance(cached, BlockDenoisingPackedKeyMask):
        return cached
    packed = BlockDenoisingPackedKeyMask(
        query_blocks=global_attn_mask.query_blocks,
        local_query_blocks=local_attn_mask.query_blocks,
        query_is_clean=global_attn_mask.query_is_clean,
        active_key_blocks=local_attn_mask.active_blocks,
        clean_key_blocks=torch.cat(clean_blocks),
        block_size=block_size,
        query_clean_bounds=global_attn_mask.query_clean_bounds,
        clean_key_positions=(
            torch.cat(clean_positions)
            if global_attn_mask.query_clean_bounds is not None
            else None
        ),
        backward_query_chunk_size=int(global_attn_mask.backward_query_chunk_size),
        debug_nonfinite_attention=bool(global_attn_mask.debug_nonfinite_attention),
    )
    local_attn_mask.flex_cache[cache_key] = packed
    return packed


def _packed_clean_flex_shard_accumulate_bshd_(
    *,
    numerator: torch.Tensor,
    m: torch.Tensor,
    l: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: (
        BlockDenoisingLocalActiveMask
        | BlockDenoisingGlobalCleanMask
        | BlockDenoisingPackedKeyMask
    ),
    key_start: int,
    scale: float,
    initial: bool,
    query_indices: torch.Tensor | None = None,
    query_cache_key: tuple[str, int] | None = None,
) -> None:
    shard_query = query
    shard_mask = attn_mask
    if query_indices is not None:
        shard_query = query.index_select(1, query_indices)
        shard_mask = _compact_clean_query_mask(
            attn_mask,
            query_indices=query_indices,
            cache_key=query_cache_key,
        )
    shard_output, shard_lse, _ = _flex_shard_stats(
        query=shard_query.transpose(1, 2),
        key=key.transpose(1, 2),
        value=value.transpose(1, 2),
        attn_mask=shard_mask,
        is_causal=False,
        scale=float(scale),
        key_start=int(key_start),
        output_float=False,
        native_fa4=True,
    )
    _merge_flex_shard_stats_(
        numerator=numerator,
        m=m,
        l=l,
        shard_output=shard_output,
        shard_lse=shard_lse,
        query_indices=query_indices,
        initial=bool(initial),
    )


def _packed_clean_flex_shard_backward_exact_bshd(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    final_output: torch.Tensor,
    final_lse: torch.Tensor,
    grad_output: torch.Tensor,
    attn_mask: BlockDenoisingGlobalCleanMask,
    key_start: int,
    scale: float,
    query_indices: torch.Tensor | None = None,
    query_cache_key: tuple[str, int] | None = None,
    grad_key: torch.Tensor | None = None,
    grad_value: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shard_query = query
    shard_output = final_output
    shard_lse = final_lse
    shard_grad_output = grad_output
    shard_mask = attn_mask
    if query_indices is not None:
        shard_query = query.index_select(1, query_indices)
        shard_output = final_output.index_select(1, query_indices)
        shard_lse = final_lse.index_select(-1, query_indices)
        shard_grad_output = grad_output.index_select(1, query_indices)
        shard_mask = _compact_clean_query_mask(
            attn_mask,
            query_indices=query_indices,
            cache_key=query_cache_key,
        )
    query_h = shard_query.transpose(1, 2)
    key_h = key.transpose(1, 2)
    value_h = value.transpose(1, 2)
    grad_query, grad_key, grad_value = _flex_merged_shard_backward(
        query=query_h,
        key=key_h,
        value=value_h,
        final_output=shard_output.transpose(1, 2),
        final_lse=shard_lse,
        grad_output=shard_grad_output.transpose(1, 2),
        attn_mask=shard_mask,
        is_causal=False,
        scale=float(scale),
        key_start=int(key_start),
        grad_key=(None if grad_key is None else grad_key.transpose(1, 2)),
        grad_value=(None if grad_value is None else grad_value.transpose(1, 2)),
    )
    return (
        grad_query.transpose(1, 2),
        grad_key.transpose(1, 2),
        grad_value.transpose(1, 2),
    )


def _prepare_packed_clean_wide_backward_bshd(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    final_output: torch.Tensor,
    final_lse: torch.Tensor,
    grad_output: torch.Tensor,
    attn_mask: BlockDenoisingGlobalCleanMask | BlockDenoisingPackedKeyMask,
    scale: float,
) -> Any:
    interval_buffers = _flex_backward_mask_buffers(
        attn_mask=attn_mask,
        key_start=0,
        key_len=int(key.shape[1]),
        device=query.device,
    )
    plan = _wide_metadata_plan(
        attn_mask=attn_mask,
        buffers=interval_buffers,
        key_start=0,
        query_len=int(query.shape[1]),
        key_len=int(key.shape[1]),
        query_heads=int(query.shape[2]),
        key_heads=int(key.shape[2]),
        native=True,
        device=query.device,
    )
    return _prepare_wide_interval_backward_bshd(
        query,
        key,
        value,
        final_output,
        final_lse,
        grad_output,
        None,
        *interval_buffers,
        float(scale),
        plan,
    )


def _compact_clean_query_mask(
    attn_mask: BlockDenoisingGlobalCleanMask | BlockDenoisingPackedKeyMask,
    *,
    query_indices: torch.Tensor,
    cache_key: tuple[str, int] | None = None,
) -> BlockDenoisingGlobalCleanMask | BlockDenoisingPackedKeyMask:
    """Select visible query metadata while preserving the exact K/V layout."""

    resolved_cache_key = (
        "compact_clean_query_mask",
        cache_key,
        int(query_indices.shape[0]),
        query_indices.device.type,
        query_indices.device.index,
    )
    cached = attn_mask.flex_cache.get(resolved_cache_key)
    if isinstance(cached, attn_mask.__class__):
        return cached
    common = {
        "query_blocks": attn_mask.query_blocks.index_select(0, query_indices),
        "query_is_clean": attn_mask.query_is_clean.index_select(
            0,
            query_indices,
        ),
        "block_size": int(attn_mask.block_size),
        "query_clean_bounds": (
            None
            if attn_mask.query_clean_bounds is None
            else attn_mask.query_clean_bounds.index_select(0, query_indices)
        ),
        "backward_query_chunk_size": int(attn_mask.backward_query_chunk_size),
        "debug_nonfinite_attention": bool(attn_mask.debug_nonfinite_attention),
    }
    if isinstance(attn_mask, BlockDenoisingPackedKeyMask):
        compact: BlockDenoisingGlobalCleanMask | BlockDenoisingPackedKeyMask = (
            BlockDenoisingPackedKeyMask(
                **common,
                local_query_blocks=attn_mask.local_query_blocks.index_select(
                    0,
                    query_indices,
                ),
                active_key_blocks=attn_mask.active_key_blocks,
                clean_key_blocks=attn_mask.clean_key_blocks,
                clean_key_positions=attn_mask.clean_key_positions,
            )
        )
    else:
        compact = BlockDenoisingGlobalCleanMask(
            **common,
            clean_context_window=attn_mask.clean_context_window,
            clean_key_blocks=attn_mask.clean_key_blocks,
            clean_key_positions=attn_mask.clean_key_positions,
            first_key_length=attn_mask.first_key_length,
            second_key_start=attn_mask.second_key_start,
            allow_native_wide_attention=bool(attn_mask.allow_native_wide_attention),
        )
    attn_mask.flex_cache[resolved_cache_key] = compact
    return compact


def _flex_should_build_block_mask(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
) -> bool:
    del query, key
    return True


def _flex_attention_backward(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    grad_output: torch.Tensor,
    grad_lse: torch.Tensor,
    attn_mask: Any | None,
    is_causal: bool,
    scale: float,
    key_start: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.enable_grad():
        q = query.detach().requires_grad_(True)
        k = key.detach().requires_grad_(True)
        v = value.detach().requires_grad_(True)
        output, lse, _ = _flex_shard_stats(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            is_causal=bool(is_causal),
            scale=float(scale),
            key_start=int(key_start),
            output_float=False,
        )
        grad_q, grad_k, grad_v = torch.autograd.grad(
            (output, lse),
            (q, k, v),
            (
                grad_output.to(dtype=output.dtype),
                grad_lse.to(dtype=lse.dtype),
            ),
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )
    return grad_q, grad_k, grad_v


def _flex_backward_mask_buffers(
    *,
    attn_mask: (
        BlockDenoisingLocalActiveMask
        | BlockDenoisingGlobalCleanMask
        | BlockDenoisingFullMask
        | BlockDenoisingPackedKeyMask
    ),
    key_start: int,
    key_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    cache_key = (
        "flex_backward_mask_buffers",
        int(key_start),
        int(key_len),
        device.type,
        device.index,
    )
    cached = attn_mask.flex_cache.get(cache_key)
    if cached is not None:
        return cached

    query_blocks = attn_mask.query_blocks
    query_is_clean = attn_mask.query_is_clean
    query_clean_bounds = getattr(attn_mask, "query_clean_bounds", None)
    if query_clean_bounds is None:
        clean_stops = query_blocks.to(dtype=torch.int32) + query_is_clean.to(
            dtype=torch.int32
        )
        query_clean_bounds = torch.stack(
            (torch.zeros_like(clean_stops), clean_stops),
            dim=-1,
        )
    else:
        query_clean_bounds = query_clean_bounds.to(dtype=torch.int32)
    if isinstance(attn_mask, BlockDenoisingPackedKeyMask):
        local_query_blocks = attn_mask.local_query_blocks
        clean_coordinates = (
            attn_mask.clean_key_positions
            if attn_mask.query_clean_bounds is not None
            else attn_mask.clean_key_blocks
        )
        if clean_coordinates is None:
            raise RuntimeError("packed clean key coordinates are missing")
        key_coordinates = torch.cat((attn_mask.active_key_blocks, clean_coordinates))
        active_len = int(attn_mask.active_key_blocks.numel())
        key_is_clean = (
            torch.arange(key_len, device=device, dtype=torch.int32) >= active_len
        )
        buffers = (
            query_clean_bounds.contiguous(),
            local_query_blocks.contiguous(),
            query_is_clean.contiguous(),
            key_coordinates.contiguous(),
            key_is_clean.contiguous(),
        )
    elif isinstance(attn_mask, BlockDenoisingLocalActiveMask):
        local_query_blocks = query_blocks
        key_blocks = attn_mask.active_blocks
        key_is_clean = torch.zeros(key_len, device=device, dtype=torch.bool)
        buffers = (
            query_clean_bounds.contiguous(),
            local_query_blocks.contiguous(),
            query_is_clean.contiguous(),
            key_blocks.contiguous(),
            key_is_clean.contiguous(),
        )
    else:
        key_positions = torch.arange(key_len, device=device, dtype=torch.int32) + int(
            key_start
        )
        if isinstance(attn_mask, BlockDenoisingFullMask):
            key_is_clean = key_positions >= int(attn_mask.clean_offset)
            key_blocks = torch.where(
                key_is_clean,
                (key_positions - int(attn_mask.clean_offset))
                // int(attn_mask.block_size),
                key_positions // int(attn_mask.block_size),
            )
            key_coordinates = (
                torch.where(
                    key_is_clean,
                    key_positions - int(attn_mask.clean_offset),
                    key_blocks,
                )
                if attn_mask.query_clean_bounds is not None
                else key_blocks
            )
            buffers = (
                query_clean_bounds.contiguous(),
                query_blocks.contiguous(),
                query_is_clean.contiguous(),
                key_coordinates.contiguous(),
                key_is_clean.contiguous(),
            )
        else:
            if attn_mask.clean_key_blocks is not None:
                key_blocks = attn_mask.clean_key_blocks
            elif (
                attn_mask.first_key_length is not None
                and attn_mask.second_key_start is not None
            ):
                key_positions = torch.where(
                    torch.arange(key_len, device=device, dtype=torch.int32)
                    < attn_mask.first_key_length,
                    key_positions,
                    attn_mask.second_key_start
                    + torch.arange(key_len, device=device, dtype=torch.int32)
                    - attn_mask.first_key_length,
                )
                key_blocks = key_positions // int(attn_mask.block_size)
            else:
                key_blocks = key_positions // int(attn_mask.block_size)
            key_is_clean = torch.ones(
                key_len,
                device=device,
                dtype=torch.bool,
            )
            key_coordinates = (
                attn_mask.clean_key_positions
                if attn_mask.query_clean_bounds is not None
                else key_blocks
            )
            if key_coordinates is None:
                key_coordinates = key_positions
            buffers = (
                query_clean_bounds.contiguous(),
                query_blocks.contiguous(),
                query_is_clean.contiguous(),
                key_coordinates.contiguous(),
                key_is_clean.contiguous(),
            )
    buffers = (
        buffers[0].to(device=device, dtype=torch.int32).contiguous(),
        buffers[1].to(device=device, dtype=torch.int32).contiguous(),
        buffers[2].to(device=device, dtype=torch.bool).contiguous(),
        buffers[3].to(device=device, dtype=torch.int32).contiguous(),
        buffers[4].to(device=device, dtype=torch.bool).contiguous(),
    )
    if int(buffers[3].numel()) != int(key_len):
        raise RuntimeError("BDLM FlexAttention key metadata length mismatch")
    attn_mask.flex_cache[cache_key] = buffers
    return buffers


def _wide_metadata_plan(
    *,
    attn_mask: (
        BlockDenoisingLocalActiveMask
        | BlockDenoisingGlobalCleanMask
        | BlockDenoisingFullMask
        | BlockDenoisingPackedKeyMask
    ),
    buffers: tuple[torch.Tensor, ...],
    key_start: int,
    query_len: int,
    key_len: int,
    query_heads: int,
    key_heads: int,
    native: bool,
    device: torch.device,
) -> Any:
    cache_key = (
        "wide_metadata_plan",
        int(key_start),
        int(query_len),
        int(key_len),
        int(query_heads),
        int(key_heads),
        bool(native),
        device.type,
        device.index,
    )
    cached = attn_mask.flex_cache.get(cache_key)
    if cached is not None:
        return cached
    if bool(native):
        plan = _build_wide_interval_plan(
            *buffers,
            query_len=int(query_len),
            key_len=int(key_len),
            query_heads=int(query_heads),
            key_heads=int(key_heads),
            sparse_tile_worklist=isinstance(
                attn_mask,
                (BlockDenoisingGlobalCleanMask, BlockDenoisingPackedKeyMask),
            ),
        )
    elif getattr(attn_mask, "query_clean_bounds", None) is not None:
        plan = _build_wide_interval_plan(
            *buffers,
            query_len=int(query_len),
            key_len=int(key_len),
        )
    else:
        plan = _build_wide_metadata_plan(
            attn_mask.query_blocks,
            *buffers[1:],
            query_len=int(query_len),
            key_len=int(key_len),
        )
    attn_mask.flex_cache[cache_key] = plan
    return plan


def _wide_execution_mask_buffers(
    attn_mask: (
        BlockDenoisingLocalActiveMask
        | BlockDenoisingGlobalCleanMask
        | BlockDenoisingFullMask
        | BlockDenoisingPackedKeyMask
    ),
    buffers: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    """Keep the stable wide-attention ABI while the plan carries intervals."""

    return (attn_mask.query_blocks, *buffers[1:])


def _wide_merged_shard_backward(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    final_output: torch.Tensor,
    final_lse: torch.Tensor,
    grad_output: torch.Tensor,
    grad_final_lse: torch.Tensor | None,
    attn_mask: Any | None,
    is_causal: bool,
    scale: float,
    key_start: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(
        attn_mask,
        (
            BlockDenoisingLocalActiveMask,
            BlockDenoisingGlobalCleanMask,
            BlockDenoisingFullMask,
            BlockDenoisingPackedKeyMask,
        ),
    ):
        interval_buffers = _flex_backward_mask_buffers(
            attn_mask=attn_mask,
            key_start=int(key_start),
            key_len=int(key.shape[-2]),
            device=query.device,
        )
        if _uses_native_wide_masked_attention(query, attn_mask):
            plan = _wide_metadata_plan(
                attn_mask=attn_mask,
                buffers=interval_buffers,
                key_start=int(key_start),
                query_len=int(query.shape[-2]),
                key_len=int(key.shape[-2]),
                query_heads=int(query.shape[1]),
                key_heads=int(key.shape[1]),
                native=True,
                device=query.device,
            )
            gradients = _wide_interval_backward_from_state_bhsd(
                query,
                key,
                value,
                final_output,
                final_lse,
                grad_output,
                grad_final_lse,
                *interval_buffers,
                float(scale),
                plan,
            )
            _debug_check_backward_finite(
                "wide_native_merged_shard",
                *gradients,
                query=query,
                key=key,
                key_start=int(key_start),
                block_size=int(attn_mask.block_size),
                debug_nonfinite_attention=bool(attn_mask.debug_nonfinite_attention),
            )
            return gradients

    with torch.no_grad():
        shard_output, shard_lse = _wide_shard_stats(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            is_causal=bool(is_causal),
            scale=float(scale),
            key_start=int(key_start),
        )
        shard_grad_output, shard_grad_lse = _merged_shard_grads(
            shard_output=shard_output,
            shard_lse=shard_lse,
            final_output=final_output,
            final_lse=final_lse,
            grad_output=grad_output,
        )
        if grad_final_lse is not None:
            finite = torch.isfinite(shard_lse) & torch.isfinite(final_lse)
            shard_weight = torch.where(
                finite,
                torch.exp(shard_lse - final_lse),
                torch.zeros_like(shard_lse),
            )
            shard_grad_lse.add_(
                grad_final_lse.to(dtype=shard_grad_lse.dtype) * shard_weight
            )

    if attn_mask is None:
        query_bshd = query.transpose(1, 2).contiguous()
        key_bshd = key.transpose(1, 2).contiguous()
        value_bshd = value.transpose(1, 2).contiguous()
        grad_output_bshd = shard_grad_output.transpose(1, 2).contiguous()
        gradients = _wide_full_backward_bshd(
            query_bshd,
            key_bshd,
            value_bshd,
            shard_output.transpose(1, 2).contiguous(),
            shard_lse,
            grad_output_bshd,
            shard_grad_lse,
            float(scale),
        )
    elif isinstance(
        attn_mask,
        (
            BlockDenoisingLocalActiveMask,
            BlockDenoisingGlobalCleanMask,
            BlockDenoisingFullMask,
            BlockDenoisingPackedKeyMask,
        ),
    ):
        (
            query_blocks,
            local_query_blocks,
            query_is_clean,
            key_blocks,
            key_is_clean,
        ) = _flex_backward_mask_buffers(
            attn_mask=attn_mask,
            key_start=int(key_start),
            key_len=int(key.shape[-2]),
            device=query.device,
        )
        plan = _wide_metadata_plan(
            attn_mask=attn_mask,
            buffers=(
                query_blocks,
                local_query_blocks,
                query_is_clean,
                key_blocks,
                key_is_clean,
            ),
            key_start=int(key_start),
            query_len=int(query.shape[-2]),
            key_len=int(key.shape[-2]),
            query_heads=int(query.shape[1]),
            key_heads=int(key.shape[1]),
            native=False,
            device=query.device,
        )
        execution_buffers = _wide_execution_mask_buffers(
            attn_mask,
            (
                query_blocks,
                local_query_blocks,
                query_is_clean,
                key_blocks,
                key_is_clean,
            ),
        )
        gradients = _wide_metadata_backward_bhsd(
            query,
            key,
            value,
            shard_grad_output,
            shard_grad_lse,
            *execution_buffers,
            float(scale),
            plan,
        )
    else:
        raise RuntimeError(
            f"wide-head backward does not support mask type {type(attn_mask)!r}"
        )
    if attn_mask is None:
        return tuple(gradient.transpose(1, 2) for gradient in gradients)
    return gradients


def _uses_packed_gqa_backward(
    *,
    attn_mask: Any,
    query_heads: int,
    key_heads: int,
    head_dim: int,
) -> bool:
    if int(query_heads) <= int(key_heads):
        return False
    if isinstance(attn_mask, BlockDenoisingPackedKeyMask):
        return True
    return int(head_dim) != 256 and isinstance(
        attn_mask,
        BlockDenoisingLocalActiveMask,
    )


def _flex_merged_shard_backward(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    final_output: torch.Tensor,
    final_lse: torch.Tensor,
    grad_output: torch.Tensor,
    grad_final_lse: torch.Tensor | None = None,
    attn_mask: Any | None,
    is_causal: bool,
    scale: float,
    key_start: int,
    grad_query: torch.Tensor | None = None,
    grad_key: torch.Tensor | None = None,
    grad_value: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiate one shard from the already-merged exact attention state."""

    if _is_wide_head_dim(int(query.shape[-1])):
        result = _wide_merged_shard_backward(
            query=query,
            key=key,
            value=value,
            final_output=final_output,
            final_lse=final_lse,
            grad_output=grad_output,
            grad_final_lse=grad_final_lse,
            attn_mask=attn_mask,
            is_causal=bool(is_causal),
            scale=float(scale),
            key_start=int(key_start),
        )
        copied = []
        for actual, supplied in zip(result, (grad_query, grad_key, grad_value)):
            if supplied is None:
                copied.append(actual)
            else:
                supplied.copy_(actual)
                copied.append(supplied)
        return copied[0], copied[1], copied[2]

    if isinstance(
        attn_mask,
        (
            BlockDenoisingLocalActiveMask,
            BlockDenoisingGlobalCleanMask,
            BlockDenoisingFullMask,
            BlockDenoisingPackedKeyMask,
        ),
    ) and not bool(is_causal):
        query_heads = int(query.shape[1])
        key_heads = int(key.shape[1])
        if query_heads % key_heads:
            raise ValueError("query heads must be divisible by KV heads")
        pack_gqa = _uses_packed_gqa_backward(
            attn_mask=attn_mask,
            query_heads=int(query.shape[1]),
            key_heads=int(key.shape[1]),
            head_dim=int(query.shape[-1]),
        )
        force_torch = int(query.shape[-1]) == 256 and not bool(pack_gqa)
        if pack_gqa and isinstance(attn_mask, BlockDenoisingPackedKeyMask):
            # The exact packed builder works directly at the native FA4 tile
            # grid.  Avoid compiling the generic token-mask builder first: at
            # 256K its pointwise launch exceeds Inductor's XBLOCK limit, and
            # the resulting forward metadata is otherwise used only to recover
            # create_block_mask's default sparse query block size.
            block_mask = _make_packed_gqa_backward_block_mask(
                attn_mask=attn_mask,
                query=query,
                key=key,
                key_start=int(key_start),
            )
        else:
            _, block_mask = _make_flex_mods(
                attn_mask=attn_mask,
                is_causal=False,
                query=query,
                key_start=int(key_start),
                key_len=int(key.shape[-2]),
                build_block_mask=True,
            )
            if block_mask is None:
                raise RuntimeError("BDLM FlexAttention backward requires a block mask")
            if pack_gqa and isinstance(attn_mask, BlockDenoisingLocalActiveMask):
                block_mask = _make_local_gqa_backward_block_mask(
                    attn_mask=attn_mask,
                    query=query,
                    key=key,
                    key_start=int(key_start),
                    forward_block_mask=block_mask,
                )
            else:
                # Global-clean and full masks use native grouped-query backward.
                # Packing their query heads regressed the documented Nemotron CP
                # path without removing additional attention work.
                pack_gqa = False
        mask_buffers = _flex_backward_mask_buffers(
            attn_mask=attn_mask,
            key_start=int(key_start),
            key_len=int(key.shape[-2]),
            device=query.device,
        )
        if force_torch:
            query_clean_bounds = mask_buffers[0]
            mask_buffers = (
                query_clean_bounds[:, 0].contiguous(),
                query_clean_bounds[:, 1].contiguous(),
                *mask_buffers[1:],
            )
        return flex_attention_backward_from_state(
            query=query,
            key=key,
            value=value,
            output=final_output.to(dtype=query.dtype),
            lse=final_lse,
            grad_output=grad_output.to(dtype=query.dtype),
            grad_lse=grad_final_lse,
            block_mask=block_mask,
            mask_mod=_explicit_bdlm_flex_mask,
            mask_buffers=mask_buffers,
            scale=float(scale),
            kernel_options=_flex_kernel_options(attn_mask=attn_mask),
            packed_gqa=pack_gqa,
            force_torch=force_torch,
            grad_query=grad_query,
            grad_key=grad_key,
            grad_value=grad_value,
        )
    if attn_mask is None and not bool(is_causal):
        return dense_attention_backward_from_state(
            query=query,
            key=key,
            value=value,
            output=final_output.to(dtype=query.dtype),
            lse=final_lse,
            grad_output=grad_output.to(dtype=query.dtype),
            grad_lse=grad_final_lse,
            scale=float(scale),
        )

    with torch.enable_grad():
        q = query.detach().requires_grad_(True)
        k = key.detach().requires_grad_(True)
        v = value.detach().requires_grad_(True)
        shard_output, shard_lse, _ = _flex_shard_stats(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            is_causal=bool(is_causal),
            scale=float(scale),
            key_start=int(key_start),
            output_float=False,
        )
        with torch.no_grad():
            shard_grad_output, shard_grad_lse = _merged_shard_grads(
                shard_output=shard_output,
                shard_lse=shard_lse,
                final_output=final_output,
                final_lse=final_lse,
                grad_output=grad_output,
            )
            if grad_final_lse is not None:
                finite = torch.isfinite(shard_lse) & torch.isfinite(final_lse)
                shard_weight = torch.where(
                    finite,
                    torch.exp(shard_lse - final_lse),
                    torch.zeros_like(shard_lse),
                )
                shard_grad_lse.add_(
                    grad_final_lse.to(dtype=shard_grad_lse.dtype) * shard_weight
                )
        grad_q, grad_k, grad_v = torch.autograd.grad(
            (shard_output, shard_lse),
            (q, k, v),
            (
                shard_grad_output.to(dtype=shard_output.dtype),
                shard_grad_lse.to(dtype=shard_lse.dtype),
            ),
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )
    return grad_q, grad_k, grad_v


def _invoke_flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Any | None,
    block_mask: Any | None,
    scale: float,
    enable_gqa: bool,
    return_aux: Any,
    kernel_options: dict[str, int] | None,
) -> Any:
    assert flex_attention is not None
    return flex_attention(
        query,
        key,
        value,
        score_mod=score_mod,
        block_mask=block_mask,
        scale=scale,
        enable_gqa=enable_gqa,
        return_aux=return_aux,
        kernel_options=kernel_options,
    )


def _flex_attention_forward_local(*args: Any) -> Any:
    return _invoke_flex_attention(*args)


def _flex_attention_forward_global(*args: Any) -> Any:
    return _invoke_flex_attention(*args)


def _flex_attention_forward_full(*args: Any) -> Any:
    return _invoke_flex_attention(*args)


def _flex_attention_forward_general(*args: Any) -> Any:
    return _invoke_flex_attention(*args)


def _flex_attention_differentiable_local(*args: Any) -> Any:
    return _invoke_flex_attention(*args)


def _flex_attention_differentiable_global(*args: Any) -> Any:
    return _invoke_flex_attention(*args)


def _flex_attention_differentiable_full(*args: Any) -> Any:
    return _invoke_flex_attention(*args)


def _flex_attention_differentiable_general(*args: Any) -> Any:
    return _invoke_flex_attention(*args)


def _flex_kernel_family(attn_mask: Any | None) -> str:
    if isinstance(
        attn_mask, (BlockDenoisingLocalActiveMask, BlockDenoisingPackedKeyMask)
    ):
        return "local"
    if isinstance(attn_mask, BlockDenoisingGlobalCleanMask):
        return "global"
    if isinstance(attn_mask, BlockDenoisingFullMask):
        return "full"
    return "general"


def _get_compiled_flex_attention(
    *,
    kernel_family: str,
    differentiable: bool,
) -> Any:
    if flex_attention is None:
        raise RuntimeError("FlexAttention is required for native ring attention")
    key = (kernel_family, bool(differentiable))
    compiled = _compiled_flex_attention.get(key)
    if compiled is None:
        targets = {
            ("local", False): _flex_attention_forward_local,
            ("global", False): _flex_attention_forward_global,
            ("full", False): _flex_attention_forward_full,
            ("general", False): _flex_attention_forward_general,
            ("local", True): _flex_attention_differentiable_local,
            ("global", True): _flex_attention_differentiable_global,
            ("full", True): _flex_attention_differentiable_full,
            ("general", True): _flex_attention_differentiable_general,
        }
        compiled = torch.compile(
            targets[key],
            dynamic=False,
            fullgraph=True,
            mode="max-autotune-no-cudagraphs",
        )
        _compiled_flex_attention[key] = compiled
    return compiled


def _flex_full_shard_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    output, lse, _ = _flex_shard_stats(
        query=query,
        key=key,
        value=value,
        attn_mask=None,
        is_causal=False,
        scale=float(scale),
        key_start=0,
        output_float=False,
    )
    return output, lse


def _flex_full_shard_attention_bshd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    output, lse, _ = _flex_shard_stats(
        query=query.transpose(1, 2),
        key=key.transpose(1, 2),
        value=value.transpose(1, 2),
        attn_mask=None,
        is_causal=False,
        scale=float(scale),
        key_start=0,
        output_float=False,
    )
    return output.transpose(1, 2), lse


def _flex_full_shard_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    grad_output: torch.Tensor,
    grad_lse: torch.Tensor,
    scale: float,
    debug_nonfinite_attention: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del output, lse
    grad_q, grad_k, grad_v = _flex_attention_backward(
        query=query,
        key=key,
        value=value,
        grad_output=grad_output,
        grad_lse=grad_lse,
        attn_mask=None,
        is_causal=False,
        scale=float(scale),
        key_start=0,
    )
    _debug_check_backward_finite(
        "flex_full",
        grad_q,
        grad_k,
        grad_v,
        query=query,
        key=key,
        key_start=0,
        block_size=0,
        debug_nonfinite_attention=bool(debug_nonfinite_attention),
    )
    return grad_q, grad_k, grad_v


def _flex_full_shard_backward_bshd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    grad_output: torch.Tensor,
    grad_lse: torch.Tensor | None,
    scale: float,
    debug_nonfinite_attention: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del output, lse
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)
    dq, dk, dv = _flex_attention_backward(
        query=q,
        key=k,
        value=v,
        grad_output=grad_output.transpose(1, 2),
        grad_lse=(
            torch.zeros(
                (query.shape[0], query.shape[2], query.shape[1]),
                device=query.device,
                dtype=torch.float32,
            )
            if grad_lse is None
            else grad_lse
        ),
        attn_mask=None,
        is_causal=False,
        scale=float(scale),
        key_start=0,
    )
    if debug_nonfinite_attention:
        _debug_check_backward_finite(
            "flex_full_bshd",
            dq,
            dk,
            dv,
            query=q,
            key=k,
            key_start=0,
            block_size=0,
            debug_nonfinite_attention=True,
        )
    return dq.transpose(1, 2), dk.transpose(1, 2), dv.transpose(1, 2)


def _full_shard_backward_torch(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    grad_output: torch.Tensor,
    grad_lse: torch.Tensor,
    scale: float,
    debug_nonfinite_attention: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, query_heads, _, head_dim = query.shape
    kv_heads = key.shape[1]
    if query_heads % kv_heads != 0:
        raise RuntimeError("query heads must be divisible by key/value heads")
    group_size = query_heads // kv_heads
    key_len = key.shape[-2]
    query_f = query.float()
    key_f = key.float()
    value_f = value.float()
    expanded_key = key_f.repeat_interleave(group_size, dim=1)
    expanded_value = value_f.repeat_interleave(group_size, dim=1)
    scores = torch.matmul(query_f, expanded_key.transpose(-2, -1)) * float(scale)
    probs = torch.softmax(scores, dim=-1)
    shard_output = torch.matmul(probs, expanded_value)
    grad_output_f = grad_output.float()
    grad_value_expanded = torch.matmul(probs.transpose(-2, -1), grad_output_f)
    grad_probs = torch.matmul(grad_output_f, expanded_value.transpose(-2, -1))
    row_dot = (grad_output_f * shard_output).sum(dim=-1, keepdim=True)
    grad_scores = probs * (grad_probs - row_dot + grad_lse.float().unsqueeze(-1))
    grad_query = torch.matmul(grad_scores, expanded_key) * float(scale)
    grad_key_expanded = torch.matmul(grad_scores.transpose(-2, -1), query_f) * float(
        scale
    )
    grad_key = grad_key_expanded.reshape(
        batch, kv_heads, group_size, key_len, head_dim
    ).sum(dim=2)
    grad_value = grad_value_expanded.reshape(
        batch, kv_heads, group_size, key_len, head_dim
    ).sum(dim=2)
    _debug_check_backward_finite(
        "fa3_full_torch",
        grad_query,
        grad_key,
        grad_value,
        query=query,
        key=key,
        key_start=0,
        block_size=0,
        debug_nonfinite_attention=bool(debug_nonfinite_attention),
    )
    return grad_query, grad_key, grad_value


def _bdlm_flex_shard_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_blocks: torch.Tensor,
    query_is_clean: torch.Tensor,
    block_size: int,
    key_start: int,
    scale: float,
    clean_offset: int = 0,
    flex_cache: dict[tuple[Any, ...], Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not query.is_cuda:
        raise RuntimeError("block-denoising FlexAttention requires CUDA tensors")
    attn_mask, logical_key_start = _bdlm_flex_mask(
        query_blocks=query_blocks,
        query_is_clean=query_is_clean,
        block_size=int(block_size),
        encoded_key_start=int(key_start),
        clean_offset=int(clean_offset),
        flex_cache=flex_cache,
    )
    output, lse, _ = _flex_shard_stats(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        is_causal=False,
        scale=float(scale),
        key_start=int(logical_key_start),
        output_float=False,
    )
    return output, lse


def _bdlm_flex_shard_attention_bshd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_blocks: torch.Tensor,
    query_is_clean: torch.Tensor,
    block_size: int,
    key_start: int,
    scale: float,
    clean_offset: int = 0,
    flex_cache: dict[tuple[Any, ...], Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not query.is_cuda:
        raise RuntimeError("block-denoising FlexAttention requires CUDA tensors")
    attn_mask, logical_key_start = _bdlm_flex_mask(
        query_blocks=query_blocks,
        query_is_clean=query_is_clean,
        block_size=int(block_size),
        encoded_key_start=int(key_start),
        clean_offset=int(clean_offset),
        flex_cache=flex_cache,
    )
    output, lse, _ = _flex_shard_stats(
        query=query.transpose(1, 2),
        key=key.transpose(1, 2),
        value=value.transpose(1, 2),
        attn_mask=attn_mask,
        is_causal=False,
        scale=float(scale),
        key_start=int(logical_key_start),
        output_float=False,
    )
    return output.transpose(1, 2), lse


def _bdlm_flex_shard_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    grad_output: torch.Tensor,
    grad_lse: torch.Tensor | None,
    query_blocks: torch.Tensor,
    query_is_clean: torch.Tensor,
    block_size: int,
    key_start: int,
    scale: float,
    clean_offset: int = 0,
    flex_cache: dict[tuple[Any, ...], Any] | None = None,
    debug_nonfinite_attention: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del output, lse
    attn_mask, logical_key_start = _bdlm_flex_mask(
        query_blocks=query_blocks,
        query_is_clean=query_is_clean,
        block_size=int(block_size),
        encoded_key_start=int(key_start),
        clean_offset=int(clean_offset),
        flex_cache=flex_cache,
    )
    grad_q, grad_k, grad_v = _flex_attention_backward(
        query=query,
        key=key,
        value=value,
        grad_output=grad_output,
        grad_lse=(
            torch.zeros(
                query.shape[:-1],
                device=query.device,
                dtype=torch.float32,
            )
            if grad_lse is None
            else grad_lse
        ),
        attn_mask=attn_mask,
        is_causal=False,
        scale=float(scale),
        key_start=int(logical_key_start),
    )
    _debug_check_backward_finite(
        "bdlm_flex_shard",
        grad_q,
        grad_k,
        grad_v,
        query=query,
        key=key,
        key_start=int(key_start),
        block_size=int(block_size),
        debug_nonfinite_attention=bool(debug_nonfinite_attention),
    )
    return grad_q, grad_k, grad_v


def _bdlm_flex_shard_backward_bshd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    grad_output: torch.Tensor,
    grad_lse: torch.Tensor | None,
    query_blocks: torch.Tensor,
    query_is_clean: torch.Tensor,
    block_size: int,
    key_start: int,
    scale: float,
    *,
    clean_offset: int,
    flex_cache: dict[tuple[Any, ...], Any] | None,
    debug_nonfinite_attention: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del output, lse
    attn_mask, logical_key_start = _bdlm_flex_mask(
        query_blocks=query_blocks,
        query_is_clean=query_is_clean,
        block_size=int(block_size),
        encoded_key_start=int(key_start),
        clean_offset=int(clean_offset),
        flex_cache=flex_cache,
    )
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)
    dq, dk, dv = _flex_attention_backward(
        query=q,
        key=k,
        value=v,
        grad_output=grad_output.transpose(1, 2),
        grad_lse=(
            torch.zeros(
                (query.shape[0], query.shape[2], query.shape[1]),
                device=query.device,
                dtype=torch.float32,
            )
            if grad_lse is None
            else grad_lse
        ),
        attn_mask=attn_mask,
        is_causal=False,
        scale=float(scale),
        key_start=int(logical_key_start),
    )
    if debug_nonfinite_attention:
        _debug_check_backward_finite(
            "bdlm_flex_shard_bshd",
            dq,
            dk,
            dv,
            query=q,
            key=k,
            key_start=int(key_start),
            block_size=int(block_size),
            debug_nonfinite_attention=True,
        )
    return dq.transpose(1, 2), dk.transpose(1, 2), dv.transpose(1, 2)


def _bdlm_flex_shard_backward_exact_bshd(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    final_output: torch.Tensor,
    final_lse: torch.Tensor,
    grad_output: torch.Tensor,
    query_blocks: torch.Tensor,
    query_is_clean: torch.Tensor,
    block_size: int,
    key_start: int,
    scale: float,
    clean_offset: int,
    flex_cache: dict[tuple[Any, ...], Any] | None,
    debug_nonfinite_attention: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    attn_mask, logical_key_start = _bdlm_flex_mask(
        query_blocks=query_blocks,
        query_is_clean=query_is_clean,
        block_size=int(block_size),
        encoded_key_start=int(key_start),
        clean_offset=int(clean_offset),
        flex_cache=flex_cache,
    )
    query_h = query.transpose(1, 2)
    key_h = key.transpose(1, 2)
    value_h = value.transpose(1, 2)
    grad_query, grad_key, grad_value = _flex_merged_shard_backward(
        query=query_h,
        key=key_h,
        value=value_h,
        final_output=final_output.transpose(1, 2),
        final_lse=final_lse,
        grad_output=grad_output.transpose(1, 2),
        attn_mask=attn_mask,
        is_causal=False,
        scale=float(scale),
        key_start=int(logical_key_start),
    )
    if debug_nonfinite_attention:
        _debug_check_backward_finite(
            "bdlm_flex_shard_exact_bshd",
            grad_query,
            grad_key,
            grad_value,
            query=query_h,
            key=key_h,
            key_start=int(key_start),
            block_size=int(block_size),
            debug_nonfinite_attention=True,
        )
    return (
        grad_query.transpose(1, 2),
        grad_key.transpose(1, 2),
        grad_value.transpose(1, 2),
    )


def _debug_check_backward_finite(
    tag: str,
    grad_q: torch.Tensor,
    grad_k: torch.Tensor,
    grad_v: torch.Tensor,
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    key_start: int,
    block_size: int,
    debug_nonfinite_attention: bool = False,
) -> None:
    if not bool(debug_nonfinite_attention):
        return
    global _debug_backward_call_count
    _debug_backward_call_count += 1
    for name, tensor in (("dq", grad_q), ("dk", grad_k), ("dv", grad_v)):
        finite = torch.isfinite(tensor)
        if finite.all():
            print(
                "attention_backward_finite "
                f"tag={tag} call={_debug_backward_call_count} tensor={name} "
                f"max_abs={float(tensor.detach().abs().max().cpu())} "
                f"shape={tuple(tensor.shape)}",
                flush=True,
            )
            continue
        finite_values = tensor[finite]
        max_abs = (
            float(finite_values.detach().abs().max().cpu())
            if finite_values.numel() > 0
            else float("nan")
        )
        raise RuntimeError(
            "nonfinite attention backward "
            f"tag={tag} call={_debug_backward_call_count} tensor={name} "
            f"nan={int(torch.isnan(tensor).sum().item())} "
            f"posinf={int(torch.isposinf(tensor).sum().item())} "
            f"neginf={int(torch.isneginf(tensor).sum().item())} "
            f"max_abs_finite={max_abs} "
            f"query_shape={tuple(query.shape)} key_shape={tuple(key.shape)} "
            f"key_start={int(key_start)} block_size={int(block_size)}"
        )


def _local_active_stats(
    *,
    query: torch.Tensor,
    local_key: torch.Tensor,
    local_value: torch.Tensor,
    local_attn_mask: Any | None,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    active_len = local_key.shape[-2]
    if active_len <= 0:
        raise ValueError("active_len must be positive")
    if isinstance(local_attn_mask, BlockDenoisingLocalActiveMask):
        block_size = int(local_attn_mask.block_size)
        if block_size <= 0 or active_len % block_size != 0:
            raise RuntimeError(
                "Block-denoising local active attention requires block-aligned active K/V"
            )
        active_num, active_m, active_l = _block_diagonal_active_stats(
            query=query[..., :active_len, :],
            key=local_key,
            value=local_value,
            block_size=block_size,
            scale=scale,
        )
        return active_num, active_m, active_l
    return _fused_shard_stats(
        query=query[..., :active_len, :],
        key=local_key,
        value=local_value,
        attn_mask=local_attn_mask,
        is_causal=False,
        scale=scale,
        key_start=0,
    )


def _local_active_backward_contribution(
    *,
    query: torch.Tensor,
    local_key: torch.Tensor,
    local_value: torch.Tensor,
    grad_output: torch.Tensor,
    final_output: torch.Tensor,
    final_lse: torch.Tensor,
    local_attn_mask: Any | None,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(local_attn_mask, BlockDenoisingLocalActiveMask):
        raise RuntimeError(
            "local active backward requires native block-denoising active masks"
        )
    active_len = local_key.shape[-2]
    active_output, active_m, active_l = _block_diagonal_active_stats(
        query=query[..., :active_len, :],
        key=local_key,
        value=local_value,
        block_size=int(local_attn_mask.block_size),
        scale=float(scale),
    )
    local_grad_output, local_grad_lse = _merged_shard_grads(
        shard_output=active_output,
        shard_lse=_final_lse_from_stats(active_m.float(), active_l.float()),
        final_output=final_output[..., :active_len, :],
        final_lse=final_lse[..., :active_len],
        grad_output=grad_output[..., :active_len, :],
    )
    active_block_size = int(local_attn_mask.block_size)
    active_block_indices = _nonzero_active_grad_block_indices(
        grad_output=local_grad_output,
        grad_lse=local_grad_lse,
        block_size=active_block_size,
    )
    if active_block_indices is None:
        local_grad_query_active, local_grad_key, local_grad_value = (
            _block_diagonal_active_backward(
                query=query[..., :active_len, :],
                key=local_key,
                value=local_value,
                output=active_output,
                lse=active_m,
                grad_output=local_grad_output,
                grad_lse=local_grad_lse,
                block_size=active_block_size,
                scale=float(scale),
            )
        )
    elif active_block_indices.numel() == 0:
        local_grad_query_active = torch.zeros_like(
            query[..., :active_len, :],
            dtype=torch.float32,
        )
        local_grad_key = torch.zeros_like(local_key, dtype=torch.float32)
        local_grad_value = torch.zeros_like(local_value, dtype=torch.float32)
    else:
        compact_query_indices = _block_token_indices(
            active_block_indices,
            block_size=active_block_size,
            device=query.device,
        )
        compact_grad_q, compact_grad_k, compact_grad_v = (
            _block_diagonal_active_backward(
                query=query[..., :active_len, :].index_select(
                    -2, compact_query_indices
                ),
                key=local_key.index_select(-2, compact_query_indices),
                value=local_value.index_select(-2, compact_query_indices),
                output=active_output.index_select(-2, compact_query_indices),
                lse=active_m.index_select(-1, compact_query_indices),
                grad_output=local_grad_output.index_select(-2, compact_query_indices),
                grad_lse=local_grad_lse.index_select(-1, compact_query_indices),
                block_size=active_block_size,
                scale=float(scale),
            )
        )
        local_grad_query_active = torch.zeros_like(
            query[..., :active_len, :],
            dtype=torch.float32,
        )
        local_grad_key = torch.zeros_like(local_key, dtype=torch.float32)
        local_grad_value = torch.zeros_like(local_value, dtype=torch.float32)
        local_grad_query_active.index_copy_(
            -2,
            compact_query_indices,
            compact_grad_q.float(),
        )
        local_grad_key.index_copy_(-2, compact_query_indices, compact_grad_k.float())
        local_grad_value.index_copy_(-2, compact_query_indices, compact_grad_v.float())
    local_grad_query = torch.zeros_like(query, dtype=torch.float32)
    local_grad_query[..., :active_len, :].copy_(local_grad_query_active.float())
    return local_grad_query, local_grad_key.float(), local_grad_value.float()


def _pad_active_stats(
    *,
    active_num: torch.Tensor,
    active_m: torch.Tensor,
    active_l: torch.Tensor,
    query: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    active_len = active_num.shape[-2]
    query_len = query.shape[-2]
    if active_len == query_len:
        return active_num, active_m, active_l
    clean_len = query_len - active_len
    if clean_len < 0:
        raise ValueError("local active K/V length cannot exceed query length")
    clean_num = torch.zeros(
        (*active_num.shape[:-2], clean_len, active_num.shape[-1]),
        device=query.device,
        dtype=active_num.dtype,
    )
    clean_m = torch.full(
        (*active_m.shape[:-1], clean_len),
        -torch.inf,
        device=query.device,
        dtype=active_m.dtype,
    )
    clean_l = torch.zeros(
        (*active_l.shape[:-1], clean_len),
        device=query.device,
        dtype=active_l.dtype,
    )
    return (
        torch.cat((active_num, clean_num), dim=-2),
        torch.cat((active_m, clean_m), dim=-1),
        torch.cat((active_l, clean_l), dim=-1),
    )


def _merge_active_stats_in_place(
    *,
    numerator: torch.Tensor,
    m: torch.Tensor,
    l: torch.Tensor,
    active_num: torch.Tensor,
    active_m: torch.Tensor,
    active_l: torch.Tensor,
) -> None:
    active_len = active_num.shape[-2]
    if active_len <= 0:
        return
    merged_num, merged_m, merged_l = _merge_online_stats(
        numerator[..., :active_len, :],
        m[..., :active_len],
        l[..., :active_len],
        active_num,
        active_m,
        active_l,
    )
    numerator[..., :active_len, :].copy_(merged_num)
    m[..., :active_len].copy_(merged_m)
    l[..., :active_len].copy_(merged_l)


def _merge_active_prefix_stats_in_place(
    *,
    numerator: torch.Tensor,
    m: torch.Tensor,
    l: torch.Tensor,
    active_num: torch.Tensor,
    active_m: torch.Tensor,
    active_l: torch.Tensor,
    attn_mask: BlockDenoisingLocalActiveMask,
) -> None:
    """Merge a contiguous active-query prefix without FP32 temporaries."""

    active_len = int(active_num.shape[-2])
    if active_len <= 0:
        return
    if not numerator.is_cuda:
        _merge_active_stats_in_place(
            numerator=numerator,
            m=m,
            l=l,
            active_num=active_num,
            active_m=active_m,
            active_l=active_l,
        )
        return
    cache_key = (
        "active_prefix_indices",
        active_len,
        numerator.device.type,
        numerator.device.index,
    )
    active_indices = attn_mask.flex_cache.get(cache_key)
    if not isinstance(active_indices, torch.Tensor):
        active_indices = torch.arange(
            active_len,
            device=numerator.device,
            dtype=torch.long,
        )
        attn_mask.flex_cache[cache_key] = active_indices
    cp_fusion.merge_compact_(
        numerator,
        m,
        l,
        active_indices,
        active_num.contiguous(),
        active_m.contiguous(),
    )


def _block_diagonal_active_stats(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_size: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact local active attention for block-denoising block-diagonal noisy K/V."""

    active_len = query.shape[-2]
    if active_len != key.shape[-2] or active_len != value.shape[-2]:
        raise ValueError("active query/key/value lengths must match")
    if active_len % block_size != 0:
        raise ValueError("active length must divide evenly by block_size")
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    batch, query_heads, _, head_dim = query.shape
    kv_heads = key.shape[1]
    if query_heads % kv_heads != 0:
        raise ValueError("query heads must be divisible by key/value heads")
    num_blocks = active_len // int(block_size)

    block_query = (
        query.view(batch, query_heads, num_blocks, int(block_size), head_dim)
        .permute(0, 2, 1, 3, 4)
        .reshape(batch * num_blocks, query_heads, int(block_size), head_dim)
    )
    block_key = (
        key.view(batch, kv_heads, num_blocks, int(block_size), head_dim)
        .permute(0, 2, 1, 3, 4)
        .reshape(batch * num_blocks, kv_heads, int(block_size), head_dim)
    )
    block_value = (
        value.view(batch, kv_heads, num_blocks, int(block_size), head_dim)
        .permute(0, 2, 1, 3, 4)
        .reshape(batch * num_blocks, kv_heads, int(block_size), head_dim)
    )
    block_output, block_lse = _flex_full_shard_attention(
        block_query,
        block_key,
        block_value,
        float(scale),
    )
    output = (
        block_output.reshape(batch, num_blocks, query_heads, int(block_size), head_dim)
        .permute(0, 2, 1, 3, 4)
        .reshape(batch, query_heads, active_len, head_dim)
    )
    m = (
        block_lse.reshape(batch, num_blocks, query_heads, int(block_size))
        .permute(0, 2, 1, 3)
        .reshape(batch, query_heads, active_len)
    )
    l = torch.ones_like(m)
    return output, m, l


def _block_diagonal_active_stats_bshd(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_size: int,
    scale: float,
    causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Block-diagonal active attention without a full BSHD-to-BHSD copy."""

    active_len = int(query.shape[1])
    if active_len != int(key.shape[1]) or active_len != int(value.shape[1]):
        raise ValueError("active query/key/value lengths must match")
    if active_len % int(block_size) != 0:
        raise ValueError("active length must divide evenly by block_size")
    batch, _, query_heads, head_dim = query.shape
    kv_heads = int(key.shape[2])
    if query_heads % kv_heads != 0:
        raise ValueError("query heads must be divisible by key/value heads")
    if active_len == 0:
        numerator = query.new_empty(
            (batch, query_heads, 0, head_dim),
            dtype=torch.float32,
        )
        stats = query.new_empty((batch, query_heads, 0), dtype=torch.float32)
        return numerator, stats, stats.clone()
    num_blocks = active_len // int(block_size)
    block_query = query.contiguous().view(
        batch * num_blocks,
        int(block_size),
        query_heads,
        head_dim,
    )
    block_key = key.contiguous().view(
        batch * num_blocks,
        int(block_size),
        kv_heads,
        head_dim,
    )
    block_value = value.contiguous().view_as(block_key)
    if causal:
        block_output, block_lse, _ = _flex_shard_stats(
            query=block_query.transpose(1, 2),
            key=block_key.transpose(1, 2),
            value=block_value.transpose(1, 2),
            attn_mask=None,
            is_causal=True,
            scale=float(scale),
            key_start=0,
            output_float=False,
        )
        block_output = block_output.transpose(1, 2)
    else:
        block_output, block_lse = _flex_full_shard_attention_bshd(
            block_query,
            block_key,
            block_value,
            float(scale),
        )
    numerator = (
        block_output.view(
            batch,
            num_blocks,
            int(block_size),
            query_heads,
            head_dim,
        )
        .permute(0, 3, 1, 2, 4)
        .reshape(batch, query_heads, active_len, head_dim)
    )
    m = (
        block_lse.view(batch, num_blocks, query_heads, int(block_size))
        .permute(0, 2, 1, 3)
        .reshape(batch, query_heads, active_len)
    )
    return numerator, m, torch.ones_like(m)


def _active_query_shard_causal_mask(
    *,
    active_indices: torch.Tensor | None,
    num_blocks: int,
    local_block_tokens: int,
    block_size: int,
    device: torch.device,
    flex_cache: dict[tuple[Any, ...], Any] | None,
) -> BlockDenoisingLocalActiveMask:
    if active_indices is None:
        raise ValueError("causal active query sharding requires active_indices")
    indices = active_indices.reshape(int(num_blocks), int(local_block_tokens))
    offsets = indices.remainder(int(block_size))
    first_offsets = offsets[0]
    if not torch.equal(offsets, first_offsets.unsqueeze(0).expand_as(offsets)):
        raise ValueError("active query shards must use identical offsets in every block")
    cache = {} if flex_cache is None else flex_cache
    cache_key = (
        "active_query_shard_causal_mask",
        tuple(int(value) for value in first_offsets.detach().cpu().tolist()),
        int(block_size),
        device.type,
        device.index,
    )
    cached = cache.get(cache_key)
    if isinstance(cached, BlockDenoisingLocalActiveMask):
        return cached
    mask = BlockDenoisingLocalActiveMask(
        query_blocks=torch.zeros(
            int(local_block_tokens), device=device, dtype=torch.int32
        ),
        query_is_clean=torch.zeros(
            int(local_block_tokens), device=device, dtype=torch.bool
        ),
        active_blocks=torch.zeros(int(block_size), device=device, dtype=torch.int32),
        block_size=int(block_size),
        causal=True,
        query_token_offsets=first_offsets.to(device=device, dtype=torch.long),
        flex_cache=cache,
    )
    cache[cache_key] = mask
    return mask


def _block_diagonal_active_query_shard_stats_bshd(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_size: int,
    scale: float,
    active_indices: torch.Tensor | None = None,
    causal: bool = False,
    flex_cache: dict[tuple[Any, ...], Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Block-diagonal attention for a within-block shard of active queries."""

    active_kv_len = int(key.shape[1])
    if active_kv_len != int(value.shape[1]):
        raise ValueError("active key/value lengths must match")
    if active_kv_len % int(block_size):
        raise ValueError("active key/value length must divide evenly by block_size")
    num_blocks = active_kv_len // int(block_size)
    local_query_len = int(query.shape[1])
    if local_query_len % num_blocks:
        raise ValueError("local active query length must divide evenly by block count")
    local_block_tokens = local_query_len // num_blocks
    if local_block_tokens <= 0 or local_block_tokens > int(block_size):
        raise ValueError("invalid local active query width")

    batch, _, query_heads, head_dim = query.shape
    kv_heads = int(key.shape[2])
    if query_heads % kv_heads:
        raise ValueError("query heads must be divisible by key/value heads")
    block_query = query.contiguous().view(
        batch * num_blocks,
        local_block_tokens,
        query_heads,
        head_dim,
    )
    block_key = key.contiguous().view(
        batch * num_blocks,
        int(block_size),
        kv_heads,
        head_dim,
    )
    block_value = value.contiguous().view_as(block_key)
    shard_mask = _active_query_shard_causal_mask(
        active_indices=active_indices,
        num_blocks=num_blocks,
        local_block_tokens=local_block_tokens,
        block_size=int(block_size),
        device=query.device,
        flex_cache=flex_cache,
    ) if causal else None
    if causal:
        block_output, block_lse, _ = _flex_shard_stats(
            query=block_query.transpose(1, 2),
            key=block_key.transpose(1, 2),
            value=block_value.transpose(1, 2),
            attn_mask=shard_mask,
            is_causal=False,
            scale=float(scale),
            key_start=0,
            output_float=False,
        )
        block_output = block_output.transpose(1, 2)
    else:
        block_output, block_lse = _flex_full_shard_attention_bshd(
            block_query,
            block_key,
            block_value,
            float(scale),
        )
    numerator = (
        block_output.view(
            batch,
            num_blocks,
            local_block_tokens,
            query_heads,
            head_dim,
        )
        .permute(0, 3, 1, 2, 4)
        .reshape(batch, query_heads, local_query_len, head_dim)
    )
    m = (
        block_lse.view(batch, num_blocks, query_heads, local_block_tokens)
        .permute(0, 2, 1, 3)
        .reshape(batch, query_heads, local_query_len)
    )
    return numerator, m, torch.ones_like(m)


def _block_diagonal_active_query_shard_backward_bshd(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    grad_output: torch.Tensor,
    block_size: int,
    scale: float,
    debug_nonfinite_attention: bool,
    active_indices: torch.Tensor | None = None,
    causal: bool = False,
    flex_cache: dict[tuple[Any, ...], Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward for a within-block shard of active queries and full active K/V."""

    active_kv_len = int(key.shape[1])
    if active_kv_len != int(value.shape[1]) or active_kv_len % int(block_size):
        raise ValueError("active key/value length must be block aligned")
    num_blocks = active_kv_len // int(block_size)
    local_query_len = int(query.shape[1])
    if local_query_len % num_blocks:
        raise ValueError("local active query length must divide evenly by block count")
    local_block_tokens = local_query_len // num_blocks
    batch, _, query_heads, head_dim = query.shape
    kv_heads = int(key.shape[2])

    def blocked_query(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.contiguous().view(
            batch * num_blocks,
            local_block_tokens,
            query_heads,
            head_dim,
        )

    block_query = blocked_query(query).transpose(1, 2)
    block_key = (
        key.contiguous()
        .view(batch * num_blocks, int(block_size), kv_heads, head_dim)
        .transpose(1, 2)
    )
    block_value = (
        value.contiguous()
        .view(batch * num_blocks, int(block_size), kv_heads, head_dim)
        .transpose(1, 2)
    )
    block_lse = (
        lse.contiguous()
        .view(batch, query_heads, num_blocks, local_block_tokens)
        .permute(0, 2, 1, 3)
        .reshape(batch * num_blocks, query_heads, local_block_tokens)
    )
    shard_mask = _active_query_shard_causal_mask(
        active_indices=active_indices,
        num_blocks=num_blocks,
        local_block_tokens=local_block_tokens,
        block_size=int(block_size),
        device=query.device,
        flex_cache=flex_cache,
    ) if causal else None
    if causal and int(head_dim) == 256:
        if shard_mask is None or shard_mask.query_token_offsets is None:
            raise RuntimeError("causal active backward requires query token offsets")
        grad_q, grad_k, grad_v = _causal_active_merged_shard_backward_torch(
            query=block_query,
            key=block_key,
            value=block_value,
            final_output=blocked_query(output).transpose(1, 2),
            final_lse=block_lse,
            grad_output=blocked_query(grad_output).transpose(1, 2),
            query_token_offsets=shard_mask.query_token_offsets,
            scale=float(scale),
        )
    else:
        grad_q, grad_k, grad_v = _flex_merged_shard_backward(
            query=block_query,
            key=block_key,
            value=block_value,
            final_output=blocked_query(output).transpose(1, 2),
            final_lse=block_lse,
            grad_output=blocked_query(grad_output).transpose(1, 2),
            attn_mask=shard_mask,
            is_causal=False,
            scale=float(scale),
            key_start=0,
        )
    if debug_nonfinite_attention:
        _debug_check_backward_finite(
            "block_diagonal_active_query_shard_bshd",
            grad_q,
            grad_k,
            grad_v,
            query=block_query,
            key=block_key,
            key_start=0,
            block_size=int(block_size),
            debug_nonfinite_attention=True,
        )
    return (
        grad_q.transpose(1, 2).reshape(batch, local_query_len, query_heads, head_dim),
        grad_k.transpose(1, 2).reshape(batch, active_kv_len, kv_heads, head_dim),
        grad_v.transpose(1, 2).reshape(batch, active_kv_len, kv_heads, head_dim),
    )


def _causal_active_merged_shard_backward_torch(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    final_output: torch.Tensor,
    final_lse: torch.Tensor,
    grad_output: torch.Tensor,
    query_token_offsets: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact D=256 causal active-shard derivative from merged attention state."""

    batch_blocks, query_heads, query_len, head_dim = query.shape
    key_heads = int(key.shape[1])
    key_len = int(key.shape[-2])
    if query_heads % key_heads:
        raise ValueError("query heads must be divisible by key/value heads")
    if int(query_token_offsets.numel()) != int(query_len):
        raise ValueError("causal query offsets must match the query shard length")
    group_size = query_heads // key_heads
    scores_per_block = max(1, query_heads * query_len * key_len)
    blocks_per_chunk = max(1, 32_000_000 // scores_per_block)
    query_offsets = query_token_offsets.to(device=query.device, dtype=torch.long)
    key_offsets = torch.arange(key_len, device=query.device, dtype=torch.long)
    allowed = key_offsets[None, :] <= query_offsets[:, None]
    grad_query_chunks: list[torch.Tensor] = []
    grad_key_chunks: list[torch.Tensor] = []
    grad_value_chunks: list[torch.Tensor] = []

    for start in range(0, int(batch_blocks), int(blocks_per_chunk)):
        stop = min(start + int(blocks_per_chunk), int(batch_blocks))
        query_f = query[start:stop].float()
        key_f = key[start:stop].float()
        value_f = value[start:stop].float()
        expanded_key = key_f.repeat_interleave(group_size, dim=1)
        expanded_value = value_f.repeat_interleave(group_size, dim=1)
        scores = torch.matmul(query_f, expanded_key.transpose(-2, -1)) * float(
            scale
        )
        scores.masked_fill_(~allowed[None, None], -torch.inf)
        shard_lse = torch.logsumexp(scores, dim=-1)
        probabilities = torch.softmax(scores, dim=-1)
        shard_output = torch.matmul(probabilities, expanded_value)
        merged_lse = final_lse[start:stop]
        finite = torch.isfinite(shard_lse) & torch.isfinite(merged_lse)
        weight = torch.where(
            finite,
            torch.exp(shard_lse - merged_lse),
            torch.zeros_like(shard_lse),
        )
        output_gradient = grad_output[start:stop].float()
        shard_grad_output = output_gradient * weight.unsqueeze(-1)
        shard_grad_lse = weight * (
            output_gradient
            * (shard_output - final_output[start:stop].float())
        ).sum(dim=-1)
        shard_grad_lse.masked_fill_(~finite, 0)

        grad_value_expanded = torch.matmul(
            probabilities.transpose(-2, -1),
            shard_grad_output,
        )
        grad_probabilities = torch.matmul(
            shard_grad_output,
            expanded_value.transpose(-2, -1),
        )
        row_dot = (shard_grad_output * shard_output).sum(dim=-1, keepdim=True)
        grad_scores = probabilities * (
            grad_probabilities - row_dot + shard_grad_lse.unsqueeze(-1)
        )
        grad_query_chunks.append(
            torch.matmul(grad_scores, expanded_key) * float(scale)
        )
        grad_key_expanded = (
            torch.matmul(grad_scores.transpose(-2, -1), query_f) * float(scale)
        )
        grad_key_chunks.append(
            grad_key_expanded.reshape(
                stop - start,
                key_heads,
                group_size,
                key_len,
                head_dim,
            ).sum(dim=2)
        )
        grad_value_chunks.append(
            grad_value_expanded.reshape(
                stop - start,
                key_heads,
                group_size,
                key_len,
                head_dim,
            ).sum(dim=2)
        )

    return (
        torch.cat(grad_query_chunks, dim=0),
        torch.cat(grad_key_chunks, dim=0),
        torch.cat(grad_value_chunks, dim=0),
    )


def _block_diagonal_active_backward_bshd(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    grad_output: torch.Tensor,
    block_size: int,
    scale: float,
    backward_query_chunk_size: int,
    debug_nonfinite_attention: bool,
    flex_cache: dict[tuple[Any, ...], Any],
    causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    active_len = int(query.shape[1])
    if active_len != int(key.shape[1]) or active_len != int(value.shape[1]):
        raise ValueError("active query/key/value lengths must match")
    if active_len % int(block_size) != 0:
        raise ValueError("active length must divide evenly by block_size")
    batch, _, query_heads, head_dim = query.shape
    kv_heads = int(key.shape[2])
    if active_len == 0:
        return torch.zeros_like(query), torch.zeros_like(key), torch.zeros_like(value)
    num_blocks = active_len // int(block_size)
    configured_chunk_size = _bdlm_backward_query_chunk_size(
        int(block_size),
        int(backward_query_chunk_size),
        query=query.transpose(1, 2),
    )
    if configured_chunk_size > 0:
        chunk_tokens = min(active_len, int(configured_chunk_size))
    else:
        chunk_tokens = active_len
    chunk_tokens = max(
        int(block_size),
        (int(chunk_tokens) // int(block_size)) * int(block_size),
    )
    if chunk_tokens < active_len:
        grad_query_chunks: list[torch.Tensor] = []
        grad_key_chunks: list[torch.Tensor] = []
        grad_value_chunks: list[torch.Tensor] = []
        for start in range(0, active_len, chunk_tokens):
            stop = min(start + chunk_tokens, active_len)
            grad_q, grad_k, grad_v = _block_diagonal_active_backward_bshd(
                query=query[:, start:stop],
                key=key[:, start:stop],
                value=value[:, start:stop],
                output=output[:, start:stop],
                lse=lse[..., start:stop],
                grad_output=grad_output[:, start:stop],
                block_size=int(block_size),
                scale=float(scale),
                backward_query_chunk_size=0,
                debug_nonfinite_attention=bool(debug_nonfinite_attention),
                flex_cache=flex_cache,
                causal=bool(causal),
            )
            grad_query_chunks.append(grad_q)
            grad_key_chunks.append(grad_k)
            grad_value_chunks.append(grad_v)
        return (
            torch.cat(grad_query_chunks, dim=1),
            torch.cat(grad_key_chunks, dim=1),
            torch.cat(grad_value_chunks, dim=1),
        )

    def blocked(tensor: torch.Tensor, heads: int) -> torch.Tensor:
        return tensor.contiguous().view(
            batch * num_blocks,
            int(block_size),
            heads,
            head_dim,
        )

    block_lse = (
        lse.contiguous()
        .view(batch, query_heads, num_blocks, int(block_size))
        .permute(0, 2, 1, 3)
        .reshape(batch * num_blocks, query_heads, int(block_size))
    )
    block_query = blocked(query, query_heads).transpose(1, 2)
    block_key = blocked(key, kv_heads).transpose(1, 2)
    block_value = blocked(value, kv_heads).transpose(1, 2)
    block_output = blocked(output, query_heads)
    block_grad_output = blocked(grad_output, query_heads)
    grad_q, grad_k, grad_v = _flex_merged_shard_backward(
        query=block_query,
        key=block_key,
        value=block_value,
        final_output=block_output.transpose(1, 2),
        final_lse=block_lse,
        grad_output=block_grad_output.transpose(1, 2),
        attn_mask=None,
        is_causal=bool(causal),
        scale=float(scale),
        key_start=0,
    )
    if debug_nonfinite_attention:
        _debug_check_backward_finite(
            "block_diagonal_active_bshd",
            grad_q,
            grad_k,
            grad_v,
            query=block_query,
            key=block_key,
            key_start=0,
            block_size=int(block_size),
            debug_nonfinite_attention=True,
        )
    return (
        grad_q.transpose(1, 2).reshape(batch, active_len, query_heads, head_dim),
        grad_k.transpose(1, 2).reshape(batch, active_len, kv_heads, head_dim),
        grad_v.transpose(1, 2).reshape(batch, active_len, kv_heads, head_dim),
    )


def _block_diagonal_active_stats_torch(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_size: int,
    scale: float,
    causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    active_len = query.shape[-2]
    batch, query_heads, _, head_dim = query.shape
    kv_heads = key.shape[1]
    group_size = query_heads // kv_heads
    num_blocks = active_len // int(block_size)
    block_query = (
        query.float()
        .view(batch, query_heads, num_blocks, int(block_size), head_dim)
        .permute(0, 2, 1, 3, 4)
    )
    block_key = (
        key.float()
        .view(batch, kv_heads, num_blocks, int(block_size), head_dim)
        .permute(0, 2, 1, 3, 4)
        .repeat_interleave(group_size, dim=2)
    )
    block_value = (
        value.float()
        .view(batch, kv_heads, num_blocks, int(block_size), head_dim)
        .permute(0, 2, 1, 3, 4)
        .repeat_interleave(group_size, dim=2)
    )
    scores = torch.matmul(block_query, block_key.transpose(-2, -1)) * float(scale)
    if causal:
        causal_mask = torch.ones(
            (int(block_size), int(block_size)),
            device=scores.device,
            dtype=torch.bool,
        ).tril_()
        scores.masked_fill_(~causal_mask, -torch.inf)
    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.softmax(scores, dim=-1)
    output = torch.matmul(probs, block_value)
    output = (
        output.reshape(batch, num_blocks, query_heads, int(block_size), head_dim)
        .permute(0, 2, 1, 3, 4)
        .reshape(batch, query_heads, active_len, head_dim)
    )
    m = (
        lse.reshape(batch, num_blocks, query_heads, int(block_size))
        .permute(0, 2, 1, 3)
        .reshape(batch, query_heads, active_len)
    )
    return output.to(dtype=query.dtype), m, torch.ones_like(m)


def _require_flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    attn_mask: Any | None,
) -> None:
    if flex_attention is None or create_block_mask is None:
        raise RuntimeError("ring attention requires a Torch build with FlexAttention")
    if query.device.type != "cuda" or key.device.type != "cuda":
        raise RuntimeError("ring attention requires CUDA FlexAttention tensors")
    if query.shape[-1] < 16 or key.shape[-1] < 16:
        raise RuntimeError(
            "ring attention requires FlexAttention head dimensions >= 16"
        )
    if query.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise RuntimeError(
            "ring attention requires float16, bfloat16, or float32 query tensors"
        )
    if key.dtype != query.dtype:
        raise RuntimeError("ring attention requires query/key dtype match")
    if (
        torch.is_tensor(attn_mask)
        and attn_mask is not None
        and attn_mask.dtype != torch.bool
    ):
        raise RuntimeError("ring attention requires boolean block-denoising masks")


def _make_flex_mods(
    *,
    attn_mask: Any | None,
    is_causal: bool,
    query: torch.Tensor,
    key_start: int,
    key_len: int,
    build_block_mask: bool = True,
    block_size: tuple[int, int] | None = None,
) -> tuple[Any | None, Any | None]:
    q_len = query.shape[-2]
    mask_chunk = None
    mask_kind = "none"
    mask_spec = None
    key_start_tensor = None
    if isinstance(
        attn_mask,
        (
            BlockDenoisingLocalActiveMask,
            BlockDenoisingGlobalCleanMask,
            BlockDenoisingFullMask,
            BlockDenoisingPackedKeyMask,
        ),
    ):
        mask_kind = "block_denoising_spec"
        mask_spec = attn_mask
        cache_key = (
            mask_spec.__class__.__name__,
            bool(is_causal),
            int(q_len),
            int(key_start),
            int(key_len),
            block_size,
            query.device.type,
            query.device.index,
        )
        cached = mask_spec.flex_cache.get(cache_key)
        if cached is not None:
            return cached
        key_start_tensor = torch.scalar_tensor(
            key_start,
            device=query.device,
            dtype=torch.int32,
        )
    elif attn_mask is not None:
        mask_chunk = attn_mask[..., key_start : key_start + key_len]
        if mask_chunk.ndim == 2:
            mask_kind = "qk"
        elif (
            mask_chunk.ndim == 4
            and mask_chunk.shape[0] == 1
            and mask_chunk.shape[1] == 1
        ):
            mask_chunk = mask_chunk[0, 0]
            mask_kind = "qk"
        elif mask_chunk.ndim == 4 and mask_chunk.shape[1] == 1:
            mask_kind = "bqk"
        elif mask_chunk.ndim == 4:
            mask_kind = "bhqk"
        else:
            raise ValueError("FlexAttention masks must have shape [Q,K] or [B,H,Q,K]")

    def mask_mod(b, h, q_idx, kv_idx):
        if mask_kind == "none":
            allowed = torch.ones_like(q_idx, dtype=torch.bool)
        elif mask_kind == "block_denoising_spec":
            allowed = _block_denoising_mask_spec_allowed(
                mask_spec,
                q_idx,
                kv_idx,
                key_start_tensor,
            )
        elif mask_kind == "qk":
            allowed = mask_chunk[q_idx, kv_idx]
        elif mask_kind == "bqk":
            allowed = mask_chunk[b, 0, q_idx, kv_idx]
        else:
            allowed = mask_chunk[b, h, q_idx, kv_idx]
        if is_causal:
            allowed = allowed & ((key_start + kv_idx) <= q_idx)
        return allowed

    if mask_kind == "none" and not is_causal:
        return None, None

    def score_mod(score, b, h, q_idx, kv_idx):
        return torch.where(mask_mod(b, h, q_idx, kv_idx), score, -float("inf"))

    if not bool(build_block_mask):
        return score_mod, None

    block_mask_builder = _get_compiled_create_block_mask()
    block_mask_kwargs = {
        "B": None,
        "H": None,
        "Q_LEN": q_len,
        "KV_LEN": key_len,
        "device": query.device,
    }
    if block_size is not None:
        block_mask_kwargs["BLOCK_SIZE"] = (
            int(block_size[0]),
            int(block_size[1]),
        )
    block_mask = block_mask_builder(mask_mod, **block_mask_kwargs)
    block_mask = _normalize_flex_block_mask_storage(block_mask)
    result = (None, block_mask)
    if mask_spec is not None:
        mask_spec.flex_cache[cache_key] = result
    return result


def _make_packed_gqa_backward_block_mask(
    *,
    attn_mask: BlockDenoisingPackedKeyMask,
    query: torch.Tensor,
    key: torch.Tensor,
    key_start: int,
    forward_block_mask: Any | None = None,
) -> Any:
    """Build exact FA4 backward metadata in Pack-GQA query coordinates."""

    query_heads = int(query.shape[1])
    key_heads = int(key.shape[1])
    if query_heads <= key_heads or query_heads % key_heads:
        raise ValueError("packed GQA metadata requires grouped-query attention")
    query_heads_per_key_head = query_heads // key_heads
    query_tile, key_tile = fa4_backward_sparse_tile(
        head_dim=int(query.shape[-1]),
        head_dim_v=int(key.shape[-1]),
        sparse_query_block_size=(
            128 if forward_block_mask is None else int(forward_block_mask.BLOCK_SIZE[0])
        ),
    )
    cache_key = (
        "packed_gqa_backward_block_mask",
        int(query.shape[-2]),
        int(key.shape[-2]),
        query_heads_per_key_head,
        query_tile,
        key_tile,
        int(key_start),
        query.device.type,
        query.device.index,
    )
    cached = attn_mask.flex_cache.get(cache_key)
    if cached is not None:
        return cached

    key_start_tensor = torch.scalar_tensor(
        int(key_start),
        device=query.device,
        dtype=torch.int32,
    )

    def packed_mask_mod(b, h, packed_q_idx, kv_idx):
        del b, h
        query_index = packed_q_idx // query_heads_per_key_head
        return _block_denoising_mask_spec_allowed(
            attn_mask,
            query_index,
            kv_idx,
            key_start_tensor,
        )

    block_mask = _make_packed_gqa_tile_block_mask(
        attn_mask=attn_mask,
        query_heads_per_key_head=query_heads_per_key_head,
        query_tile=query_tile,
        key_tile=key_tile,
        mask_mod=packed_mask_mod,
    )
    attn_mask.flex_cache[cache_key] = block_mask
    return block_mask


def _make_local_gqa_backward_block_mask(
    *,
    attn_mask: (
        BlockDenoisingLocalActiveMask
        | BlockDenoisingGlobalCleanMask
        | BlockDenoisingFullMask
    ),
    query: torch.Tensor,
    key: torch.Tensor,
    key_start: int,
    forward_block_mask: Any,
) -> Any:
    """Build exact Pack-GQA metadata for a non-packed BDLM mask."""

    query_heads = int(query.shape[1])
    key_heads = int(key.shape[1])
    if query_heads <= key_heads or query_heads % key_heads:
        raise ValueError("local GQA metadata requires grouped-query attention")
    query_heads_per_key_head = query_heads // key_heads
    query_tile, key_tile = fa4_backward_sparse_tile(
        head_dim=int(query.shape[-1]),
        head_dim_v=int(key.shape[-1]),
        sparse_query_block_size=int(forward_block_mask.BLOCK_SIZE[0]),
    )
    cache_key = (
        "local_gqa_backward_block_mask",
        int(query.shape[-2]),
        int(key.shape[-2]),
        query_heads_per_key_head,
        query_tile,
        key_tile,
        int(key_start),
        query.device.type,
        query.device.index,
    )
    cached = attn_mask.flex_cache.get(cache_key)
    if cached is not None:
        return cached

    key_start_tensor = torch.scalar_tensor(
        int(key_start),
        device=query.device,
        dtype=torch.int32,
    )

    def packed_mask_mod(b, h, packed_q_idx, kv_idx):
        del b, h
        query_index = packed_q_idx // query_heads_per_key_head
        return _block_denoising_mask_spec_allowed(
            attn_mask,
            query_index,
            kv_idx,
            key_start_tensor,
        )

    (
        query_clean_bounds,
        local_query_blocks,
        query_is_clean,
        key_coordinates,
        key_is_clean,
    ) = _flex_backward_mask_buffers(
        attn_mask=attn_mask,
        key_start=int(key_start),
        key_len=int(key.shape[-2]),
        device=query.device,
    )
    query_blocks = attn_mask.query_blocks
    query_len = int(query.shape[-2])
    key_len = int(key.shape[-2])
    if (
        int(query_blocks.numel()) != query_len
        or int(local_query_blocks.numel()) != query_len
        or int(query_is_clean.numel()) != query_len
        or int(key_coordinates.numel()) != key_len
    ):
        raise RuntimeError("BDLM GQA metadata lengths must match attention tensors")

    if getattr(attn_mask, "query_clean_bounds", None) is not None:
        block_mask = _get_compiled_create_block_mask()(
            packed_mask_mod,
            B=None,
            H=None,
            Q_LEN=query_len * query_heads_per_key_head,
            KV_LEN=key_len,
            device=query.device,
            BLOCK_SIZE=(query_tile, key_tile),
        )
        block_mask = _normalize_flex_block_mask_storage(block_mask)
        attn_mask.flex_cache[cache_key] = block_mask
        return block_mask

    if isinstance(attn_mask, BlockDenoisingLocalActiveMask):
        active_key_len = key_len
    elif isinstance(attn_mask, BlockDenoisingGlobalCleanMask):
        active_key_len = 0
    else:
        active_key_len = max(
            0,
            min(key_len, int(attn_mask.clean_offset) - int(key_start)),
        )
    packed_mask = BlockDenoisingPackedKeyMask(
        query_blocks=query_blocks,
        local_query_blocks=local_query_blocks,
        query_is_clean=query_is_clean,
        active_key_blocks=key_coordinates[:active_key_len],
        clean_key_blocks=key_coordinates[active_key_len:],
        block_size=int(attn_mask.block_size),
        backward_query_chunk_size=int(attn_mask.backward_query_chunk_size),
        debug_nonfinite_attention=bool(attn_mask.debug_nonfinite_attention),
        flex_cache=attn_mask.flex_cache,
    )
    block_mask = _make_packed_gqa_tile_block_mask(
        attn_mask=packed_mask,
        query_heads_per_key_head=query_heads_per_key_head,
        query_tile=query_tile,
        key_tile=key_tile,
        mask_mod=packed_mask_mod,
    )
    # Preserve the fixed metadata strides used by the prior CP backward path.
    # Native FA4 consumes the sparse counts but specializes on tensor layout.
    block_mask = _normalize_flex_block_mask_storage(block_mask)
    attn_mask.flex_cache[cache_key] = block_mask
    return block_mask


def _make_packed_gqa_tile_block_mask(
    *,
    attn_mask: BlockDenoisingPackedKeyMask,
    query_heads_per_key_head: int,
    query_tile: int,
    key_tile: int,
    mask_mod: Any,
) -> Any:
    """Construct exact Pack-GQA sparsity directly at FA4 tile granularity."""

    if BlockMask is None:
        raise RuntimeError("Torch FlexAttention BlockMask is unavailable")
    group_size = int(query_heads_per_key_head)
    if group_size <= 1 or int(query_tile) % group_size:
        raise ValueError("packed query tiles must contain complete head groups")

    query_blocks = attn_mask.query_blocks.repeat_interleave(group_size)
    local_query_blocks = attn_mask.local_query_blocks.repeat_interleave(group_size)
    query_is_clean = attn_mask.query_is_clean.repeat_interleave(group_size)
    key_blocks = torch.cat(
        (attn_mask.active_key_blocks, attn_mask.clean_key_blocks),
        dim=0,
    )
    key_is_clean = torch.cat(
        (
            torch.zeros_like(attn_mask.active_key_blocks, dtype=torch.bool),
            torch.ones_like(attn_mask.clean_key_blocks, dtype=torch.bool),
        ),
        dim=0,
    )
    query_clean_bounds = (
        None
        if attn_mask.query_clean_bounds is None
        else attn_mask.query_clean_bounds.repeat_interleave(group_size, dim=0)
    )
    key_clean_positions = (
        None
        if attn_mask.clean_key_positions is None
        else torch.cat(
            (
                torch.zeros_like(attn_mask.active_key_blocks),
                attn_mask.clean_key_positions,
            ),
            dim=0,
        )
    )

    def tiled(values: torch.Tensor, tile: int, pad_value: int | bool) -> torch.Tensor:
        padding = (-int(values.numel())) % int(tile)
        if padding:
            values = F.pad(values, (0, padding), value=pad_value)
        return values.reshape(-1, int(tile))

    query_valid = tiled(
        torch.ones_like(query_blocks, dtype=torch.bool), query_tile, False
    )
    key_valid = tiled(torch.ones_like(key_blocks, dtype=torch.bool), key_tile, False)
    query_blocks = tiled(query_blocks, query_tile, 0)
    local_query_blocks = tiled(local_query_blocks, query_tile, 0)
    query_is_clean = tiled(query_is_clean, query_tile, False)
    key_blocks = tiled(key_blocks, key_tile, 0)
    key_is_clean = tiled(key_is_clean, key_tile, False)
    if query_clean_bounds is not None:
        query_clean_starts = tiled(query_clean_bounds[:, 0], query_tile, 0)
        query_clean_stops = tiled(query_clean_bounds[:, 1], query_tile, 0)
        if key_clean_positions is None:
            raise RuntimeError("exact packed mask is missing clean key positions")
        key_clean_positions = tiled(key_clean_positions, key_tile, 0)

    def role_summary(
        blocks: torch.Tensor,
        role: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        selected = role & valid
        upper = torch.iinfo(blocks.dtype).max
        minimum = torch.where(selected, blocks, upper).amin(dim=-1)
        maximum = torch.where(selected, blocks, -1).amax(dim=-1)
        return selected.any(dim=-1), minimum, maximum

    query_active, query_active_min, query_active_max = role_summary(
        query_blocks, ~query_is_clean, query_valid
    )
    query_clean, query_clean_min, query_clean_max = role_summary(
        query_blocks, query_is_clean, query_valid
    )
    _, local_active_min, local_active_max = role_summary(
        local_query_blocks, ~query_is_clean, query_valid
    )
    key_active, key_active_min, key_active_max = role_summary(
        key_blocks, ~key_is_clean, key_valid
    )
    key_clean, key_clean_min, key_clean_max = role_summary(
        key_blocks, key_is_clean, key_valid
    )

    qa = query_active[:, None]
    qc = query_clean[:, None]
    ka = key_active[None, :]
    kc = key_clean[None, :]
    active_pair = (
        qa
        & ka
        & (local_active_max[:, None] >= key_active_min[None, :])
        & (key_active_max[None, :] >= local_active_min[:, None])
    )
    if query_clean_bounds is None:
        active_clean_pair = (
            qa & kc & (query_active_max[:, None] > key_clean_min[None, :])
        )
        clean_pair = qc & kc & (query_clean_max[:, None] >= key_clean_min[None, :])
        clean_context_pair = active_clean_pair | clean_pair
        clean_context_full = (
            ~(qa & kc) | (query_active_min[:, None] > key_clean_max[None, :])
        ) & (~(qc & kc) | (query_clean_min[:, None] >= key_clean_max[None, :]))
    else:
        upper = torch.iinfo(query_clean_starts.dtype).max
        query_start_min = torch.where(query_valid, query_clean_starts, upper).amin(
            dim=-1
        )
        query_start_max = torch.where(query_valid, query_clean_starts, -1).amax(dim=-1)
        query_stop_min = torch.where(query_valid, query_clean_stops, upper).amin(dim=-1)
        query_stop_max = torch.where(query_valid, query_clean_stops, -1).amax(dim=-1)
        clean_key_min = torch.where(
            key_is_clean & key_valid, key_clean_positions, upper
        ).amin(dim=-1)
        clean_key_max = torch.where(
            key_is_clean & key_valid, key_clean_positions, -1
        ).amax(dim=-1)
        query_present = query_valid.any(dim=-1)[:, None]
        clean_context_pair = (
            query_present
            & kc
            & (clean_key_max[None, :] >= query_start_min[:, None])
            & (clean_key_min[None, :] < query_stop_max[:, None])
        )
        clean_context_full = (~kc) | (
            (clean_key_min[None, :] >= query_start_max[:, None])
            & (clean_key_max[None, :] < query_stop_min[:, None])
        )
    occupied = active_pair | clean_context_pair

    active_pair_full = (
        (local_active_min[:, None] == local_active_max[:, None])
        & (key_active_min[None, :] == key_active_max[None, :])
        & (local_active_min[:, None] == key_active_min[None, :])
    )
    full = occupied
    full = full & (~(qa & ka) | active_pair_full)
    full = full & ~(qc & ka)
    full = full & clean_context_full
    full = full & query_valid.all(dim=-1)[:, None]
    full = full & key_valid.all(dim=-1)[None, :]
    partial = occupied & ~full

    def ordered(dense: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        counts = dense.sum(dim=-1, dtype=torch.int32)
        width = int(counts.max().item()) if counts.numel() else 0
        if width == 0:
            indices = torch.empty(
                (*dense.shape[:-1], 0),
                device=dense.device,
                dtype=torch.int32,
            )
        else:
            indices = torch.argsort(
                dense.to(dtype=torch.int8),
                dim=-1,
                descending=True,
                stable=True,
            )[..., :width].to(dtype=torch.int32)
        return counts.contiguous(), indices.contiguous()

    partial_kv_count, partial_kv_indices = ordered(partial)
    full_kv_count, full_kv_indices = ordered(full)
    partial_q_count, partial_q_indices = ordered(partial.transpose(0, 1))
    full_q_count, full_q_indices = ordered(full.transpose(0, 1))
    return BlockMask(
        seq_lengths=(
            int(attn_mask.query_blocks.numel()) * group_size,
            int(
                attn_mask.active_key_blocks.numel() + attn_mask.clean_key_blocks.numel()
            ),
        ),
        kv_num_blocks=partial_kv_count[None, None],
        kv_indices=partial_kv_indices[None, None],
        full_kv_num_blocks=full_kv_count[None, None],
        full_kv_indices=full_kv_indices[None, None],
        q_num_blocks=partial_q_count[None, None],
        q_indices=partial_q_indices[None, None],
        full_q_num_blocks=full_q_count[None, None],
        full_q_indices=full_q_indices[None, None],
        BLOCK_SIZE=(int(query_tile), int(key_tile)),
        mask_mod=mask_mod,
    )


def _get_compiled_create_block_mask() -> Any:
    """Return the fused FlexAttention block-mask builder.

    Eager ``create_block_mask`` materializes the full token-level mask before
    reducing it to block metadata. Compiling the official PyTorch builder fuses
    that reduction and keeps mask construction proportional to the block grid.
    """

    global _compiled_create_block_mask
    if _compiled_create_block_mask is None:
        if create_block_mask is None:
            raise RuntimeError("Torch FlexAttention block-mask creation is unavailable")
        _compiled_create_block_mask = torch.compile(
            create_block_mask,
            fullgraph=True,
        )
    return _compiled_create_block_mask


def _normalize_flex_block_mask_storage(block_mask: Any) -> Any:
    """Use fixed block-grid metadata shapes without changing sparse entries."""

    if BlockMask is None:
        raise RuntimeError("Torch FlexAttention BlockMask is unavailable")
    q_block_size, kv_block_size = block_mask.BLOCK_SIZE
    q_blocks = math.ceil(int(block_mask.seq_lengths[0]) / int(q_block_size))
    kv_blocks = math.ceil(int(block_mask.seq_lengths[1]) / int(kv_block_size))

    def pad_indices(indices: torch.Tensor | None, width: int) -> torch.Tensor | None:
        if indices is None or int(indices.shape[-1]) == int(width):
            return indices
        if int(indices.shape[-1]) > int(width):
            raise RuntimeError("FlexAttention BlockMask exceeds its logical block grid")
        return F.pad(indices, (0, int(width) - int(indices.shape[-1])))

    return BlockMask(
        seq_lengths=block_mask.seq_lengths,
        kv_num_blocks=block_mask.kv_num_blocks,
        kv_indices=pad_indices(block_mask.kv_indices, kv_blocks),
        full_kv_num_blocks=block_mask.full_kv_num_blocks,
        full_kv_indices=pad_indices(block_mask.full_kv_indices, kv_blocks),
        q_num_blocks=block_mask.q_num_blocks,
        q_indices=pad_indices(block_mask.q_indices, q_blocks),
        full_q_num_blocks=block_mask.full_q_num_blocks,
        full_q_indices=pad_indices(block_mask.full_q_indices, q_blocks),
        BLOCK_SIZE=block_mask.BLOCK_SIZE,
        mask_mod=block_mask.mask_mod,
    )


def _block_denoising_mask_spec_allowed(
    mask_spec: (
        BlockDenoisingLocalActiveMask
        | BlockDenoisingGlobalCleanMask
        | BlockDenoisingFullMask
        | BlockDenoisingPackedKeyMask
    ),
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    key_start: int | torch.Tensor,
) -> torch.Tensor:
    q_block = mask_spec.query_blocks[q_idx]
    q_is_clean = mask_spec.query_is_clean[q_idx]
    if isinstance(mask_spec, BlockDenoisingLocalActiveMask):
        kv_block = mask_spec.active_blocks[kv_idx]
        allowed = (~q_is_clean) & (q_block == kv_block)
        if mask_spec.causal:
            query_offset = (
                q_idx % int(mask_spec.block_size)
                if mask_spec.query_token_offsets is None
                else mask_spec.query_token_offsets[q_idx]
            )
            allowed = allowed & (
                (kv_idx % int(mask_spec.block_size))
                <= query_offset
            )
        return allowed

    if isinstance(mask_spec, BlockDenoisingPackedKeyMask):
        local_q_block = mask_spec.local_query_blocks[q_idx]
        active_len = mask_spec.active_key_blocks.shape[0]
        if active_len == 0:
            kv_is_clean = torch.ones_like(kv_idx, dtype=torch.bool)
            active_kv_block = torch.zeros_like(kv_idx, dtype=torch.int32)
        else:
            kv_is_clean = kv_idx >= active_len
            active_token = torch.clamp(kv_idx, max=active_len - 1)
            active_kv_block = mask_spec.active_key_blocks[active_token]
        clean_token = torch.clamp(kv_idx - active_len, min=0)
        clean_kv_block = mask_spec.clean_key_blocks[clean_token]
        kv_block = torch.where(kv_is_clean, clean_kv_block, active_kv_block)
        active_to_active = (~q_is_clean) & (~kv_is_clean) & (local_q_block == kv_block)
        if mask_spec.query_clean_bounds is not None:
            if mask_spec.clean_key_positions is None:
                raise RuntimeError("exact packed mask is missing clean key positions")
            clean_kv_position = mask_spec.clean_key_positions[clean_token]
            clean_start = mask_spec.query_clean_bounds[:, 0][q_idx]
            clean_stop = mask_spec.query_clean_bounds[:, 1][q_idx]
            clean_context = (
                kv_is_clean
                & (clean_kv_position >= clean_start)
                & (clean_kv_position < clean_stop)
            )
        else:
            active_to_clean = (~q_is_clean) & kv_is_clean & (q_block > kv_block)
            clean_to_clean = q_is_clean & kv_is_clean & (q_block >= kv_block)
            clean_context = active_to_clean | clean_to_clean
        return active_to_active | clean_context

    if isinstance(mask_spec, BlockDenoisingFullMask):
        kv_pos = key_start + kv_idx
        clean_offset = int(mask_spec.clean_offset)
        kv_is_clean = kv_pos >= clean_offset
        key_position = kv_pos - clean_offset
        kv_block = torch.where(
            kv_is_clean,
            key_position // int(mask_spec.block_size),
            kv_pos // int(mask_spec.block_size),
        )
        active_to_active = (~q_is_clean) & (~kv_is_clean) & (q_block == kv_block)
        if mask_spec.query_clean_bounds is not None:
            clean_start = mask_spec.query_clean_bounds[:, 0][q_idx]
            clean_stop = mask_spec.query_clean_bounds[:, 1][q_idx]
            clean_context = (
                kv_is_clean
                & (key_position >= clean_start)
                & (key_position < clean_stop)
            )
        else:
            active_to_clean = (~q_is_clean) & kv_is_clean & (q_block > kv_block)
            clean_to_clean = q_is_clean & kv_is_clean & (q_block >= kv_block)
            clean_context = active_to_clean | clean_to_clean
        return active_to_active | clean_context

    if mask_spec.clean_key_blocks is not None:
        kv_block = mask_spec.clean_key_blocks[kv_idx]
    else:
        kv_pos = key_start + kv_idx
        if (
            mask_spec.first_key_length is not None
            and mask_spec.second_key_start is not None
        ):
            kv_pos = torch.where(
                kv_idx < mask_spec.first_key_length,
                kv_pos,
                mask_spec.second_key_start + kv_idx - mask_spec.first_key_length,
            )
        kv_block = kv_pos // int(mask_spec.block_size)
    if mask_spec.query_clean_bounds is not None:
        if mask_spec.clean_key_positions is not None:
            key_position = mask_spec.clean_key_positions[kv_idx]
        else:
            key_position = key_start + kv_idx
        clean_start = mask_spec.query_clean_bounds[:, 0][q_idx]
        clean_stop = mask_spec.query_clean_bounds[:, 1][q_idx]
        return (key_position >= clean_start) & (key_position < clean_stop)
    active_to_clean = (~q_is_clean) & (q_block > kv_block)
    clean_to_clean = q_is_clean & (q_block >= kv_block)
    return active_to_clean | clean_to_clean




def _detach_attention_mask(mask: Any | None) -> Any | None:
    if mask is None:
        return None
    if torch.is_tensor(mask):
        return mask.detach()
    if isinstance(
        mask,
        (
            BlockDenoisingLocalActiveMask,
            BlockDenoisingGlobalCleanMask,
            BlockDenoisingFullMask,
            BlockDenoisingPackedKeyMask,
        ),
    ):
        return mask.detach()
    raise TypeError(f"unsupported attention mask type: {type(mask)!r}")


def _final_lse_from_stats(m: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
    return torch.where(
        l > 0,
        m + torch.log(l.clamp_min(torch.finfo(torch.float32).tiny)),
        torch.full_like(m, -torch.inf),
    )


def _merged_shard_grads(
    *,
    shard_output: torch.Tensor,
    shard_lse: torch.Tensor,
    final_output: torch.Tensor,
    final_lse: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if grad_output.is_cuda:
        shard_grad_output = torch.empty(
            grad_output.shape,
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        grad_lse = torch.empty(
            shard_lse.shape,
            device=shard_lse.device,
            dtype=torch.float32,
        )
        cp_fusion.merge_backward_(
            shard_output,
            shard_lse.contiguous(),
            final_output,
            final_lse.contiguous(),
            grad_output,
            shard_grad_output,
            grad_lse,
        )
        return shard_grad_output, grad_lse

    finite = torch.isfinite(shard_lse) & torch.isfinite(final_lse)
    weight = torch.where(
        finite,
        torch.exp(shard_lse - final_lse),
        torch.zeros_like(shard_lse),
    )
    grad_output_float = grad_output.float()
    shard_grad_output = (grad_output_float * weight.unsqueeze(-1)).to(
        dtype=grad_output.dtype
    )
    grad_lse = weight * (
        grad_output_float * (shard_output.float() - final_output.float())
    ).sum(dim=-1)
    grad_lse.masked_fill_(~finite, 0)
    return shard_grad_output, grad_lse


def _bdlm_encoded_key_start(
    mask: BlockDenoisingGlobalCleanMask | BlockDenoisingFullMask, key_start: int
) -> int:
    if isinstance(mask, BlockDenoisingFullMask):
        return -int(key_start) - 1
    return int(key_start)


def _decode_bdlm_key_start(encoded_key_start: int) -> tuple[int, bool]:
    """Decode the full-mask flag carried with a logical K/V shard offset."""

    encoded_key_start = int(encoded_key_start)
    if encoded_key_start < 0:
        return -encoded_key_start - 1, True
    return encoded_key_start, False


def _bdlm_flex_mask(
    *,
    query_blocks: torch.Tensor,
    query_is_clean: torch.Tensor,
    block_size: int,
    encoded_key_start: int,
    clean_offset: int,
    flex_cache: dict[tuple[Any, ...], Any] | None,
) -> tuple[BlockDenoisingGlobalCleanMask | BlockDenoisingFullMask, int]:
    key_start, full_mask = _decode_bdlm_key_start(encoded_key_start)
    common = {
        "query_blocks": query_blocks.to(dtype=torch.int32),
        "query_is_clean": query_is_clean.to(dtype=torch.bool),
        "block_size": int(block_size),
        "flex_cache": {} if flex_cache is None else flex_cache,
    }
    if full_mask:
        return (
            BlockDenoisingFullMask(
                **common,
                clean_offset=int(clean_offset),
            ),
            int(key_start),
        )
    return BlockDenoisingGlobalCleanMask(**common), int(key_start)


def _merge_flex_shard_stats_(
    *,
    numerator: torch.Tensor,
    m: torch.Tensor,
    l: torch.Tensor,
    shard_output: torch.Tensor,
    shard_lse: torch.Tensor,
    query_indices: torch.Tensor | None,
    initial: bool,
) -> None:
    if query_indices is None:
        if initial:
            numerator.copy_(shard_output)
            m.copy_(shard_lse)
            l.copy_(torch.isfinite(shard_lse).to(dtype=l.dtype))
            return
        cp_fusion.merge_full_(
            numerator,
            m,
            l,
            shard_output,
            shard_lse,
            torch.isfinite(shard_lse).to(dtype=l.dtype),
        )
        return
    if initial:
        numerator.zero_()
        m.fill_(-torch.inf)
        l.zero_()
    cp_fusion.merge_compact_(
        numerator,
        m,
        l,
        query_indices.contiguous(),
        shard_output,
        shard_lse,
    )


def _bdlm_clean_offset(
    mask: BlockDenoisingGlobalCleanMask | BlockDenoisingFullMask,
) -> int:
    return int(mask.clean_offset) if isinstance(mask, BlockDenoisingFullMask) else 0


def _block_diagonal_active_backward(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    grad_output: torch.Tensor,
    grad_lse: torch.Tensor,
    block_size: int,
    scale: float,
    backward_query_chunk_size: int = 0,
    debug_nonfinite_attention: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    active_len = query.shape[-2]
    if active_len == 0:
        return (
            torch.zeros_like(query),
            torch.zeros_like(key),
            torch.zeros_like(value),
        )
    if active_len % block_size != 0:
        raise RuntimeError(
            "Block-denoising local active backward requires block-aligned inputs"
        )
    batch, query_heads, _, head_dim = query.shape
    kv_heads = key.shape[1]
    if query_heads % kv_heads != 0:
        raise RuntimeError("query heads must be divisible by key/value heads")
    num_blocks = active_len // int(block_size)
    configured_chunk_size = _bdlm_backward_query_chunk_size(
        block_size,
        backward_query_chunk_size,
        query=query,
    )
    if configured_chunk_size > 0:
        chunk_tokens = min(int(active_len), int(configured_chunk_size))
    else:
        chunk_tokens = int(active_len)
    chunk_tokens = max(
        int(block_size), (int(chunk_tokens) // int(block_size)) * int(block_size)
    )
    if chunk_tokens < active_len:
        grad_q_chunks: list[torch.Tensor] = []
        grad_k_chunks: list[torch.Tensor] = []
        grad_v_chunks: list[torch.Tensor] = []
        for start in range(0, active_len, chunk_tokens):
            stop = min(start + chunk_tokens, active_len)
            grad_q, grad_k, grad_v = _block_diagonal_active_backward(
                query=query[..., start:stop, :],
                key=key[..., start:stop, :],
                value=value[..., start:stop, :],
                output=output[..., start:stop, :],
                lse=lse[..., start:stop],
                grad_output=grad_output[..., start:stop, :],
                grad_lse=grad_lse[..., start:stop],
                block_size=int(block_size),
                scale=float(scale),
                backward_query_chunk_size=0,
                debug_nonfinite_attention=bool(debug_nonfinite_attention),
            )
            grad_q_chunks.append(grad_q)
            grad_k_chunks.append(grad_k)
            grad_v_chunks.append(grad_v)
        return (
            torch.cat(grad_q_chunks, dim=-2),
            torch.cat(grad_k_chunks, dim=-2),
            torch.cat(grad_v_chunks, dim=-2),
        )
    block_query = (
        query.contiguous()
        .view(batch, query_heads, num_blocks, int(block_size), head_dim)
        .permute(0, 2, 1, 3, 4)
        .reshape(batch * num_blocks, query_heads, int(block_size), head_dim)
    )
    block_key = (
        key.contiguous()
        .view(batch, kv_heads, num_blocks, int(block_size), head_dim)
        .permute(0, 2, 1, 3, 4)
        .reshape(batch * num_blocks, kv_heads, int(block_size), head_dim)
    )
    block_value = (
        value.contiguous()
        .view(batch, kv_heads, num_blocks, int(block_size), head_dim)
        .permute(0, 2, 1, 3, 4)
        .reshape(batch * num_blocks, kv_heads, int(block_size), head_dim)
    )
    block_output = (
        output.contiguous()
        .view(batch, query_heads, num_blocks, int(block_size), head_dim)
        .permute(0, 2, 1, 3, 4)
        .reshape(batch * num_blocks, query_heads, int(block_size), head_dim)
    )
    block_grad_output = (
        grad_output.contiguous()
        .view(batch, query_heads, num_blocks, int(block_size), head_dim)
        .permute(0, 2, 1, 3, 4)
        .reshape(batch * num_blocks, query_heads, int(block_size), head_dim)
    )
    block_lse = (
        lse.contiguous()
        .view(batch, query_heads, num_blocks, int(block_size))
        .permute(0, 2, 1, 3)
        .reshape(batch * num_blocks, query_heads, int(block_size))
    )
    block_grad_lse = (
        grad_lse.contiguous()
        .view(batch, query_heads, num_blocks, int(block_size))
        .permute(0, 2, 1, 3)
        .reshape(batch * num_blocks, query_heads, int(block_size))
    )
    grad_q, grad_k, grad_v = _flex_full_shard_backward(
        block_query,
        block_key,
        block_value,
        block_output,
        block_lse,
        block_grad_output,
        block_grad_lse,
        float(scale),
        bool(debug_nonfinite_attention),
    )
    return (
        grad_q.reshape(batch, num_blocks, query_heads, int(block_size), head_dim)
        .permute(0, 2, 1, 3, 4)
        .reshape(batch, query_heads, active_len, head_dim),
        grad_k.reshape(batch, num_blocks, kv_heads, int(block_size), head_dim)
        .permute(0, 2, 1, 3, 4)
        .reshape(batch, kv_heads, active_len, head_dim),
        grad_v.reshape(batch, num_blocks, kv_heads, int(block_size), head_dim)
        .permute(0, 2, 1, 3, 4)
        .reshape(batch, kv_heads, active_len, head_dim),
    )


def _block_diagonal_active_backward_torch(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    grad_output: torch.Tensor,
    grad_lse: torch.Tensor,
    block_size: int,
    scale: float,
    debug_nonfinite_attention: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    active_len = query.shape[-2]
    batch, query_heads, _, head_dim = query.shape
    kv_heads = key.shape[1]
    group_size = query_heads // kv_heads
    num_blocks = active_len // int(block_size)
    block_query = (
        query.float()
        .view(batch, query_heads, num_blocks, int(block_size), head_dim)
        .permute(0, 2, 1, 3, 4)
    )
    block_key = (
        key.float()
        .view(batch, kv_heads, num_blocks, int(block_size), head_dim)
        .permute(0, 2, 1, 3, 4)
    )
    block_value = (
        value.float()
        .view(batch, kv_heads, num_blocks, int(block_size), head_dim)
        .permute(0, 2, 1, 3, 4)
    )
    expanded_key = block_key.repeat_interleave(group_size, dim=2)
    expanded_value = block_value.repeat_interleave(group_size, dim=2)
    scores = torch.matmul(block_query, expanded_key.transpose(-2, -1)) * float(scale)
    probs = torch.softmax(scores, dim=-1)
    block_output = (
        output.float()
        .view(batch, query_heads, num_blocks, int(block_size), head_dim)
        .permute(0, 2, 1, 3, 4)
    )
    block_grad_output = (
        grad_output.float()
        .view(batch, query_heads, num_blocks, int(block_size), head_dim)
        .permute(0, 2, 1, 3, 4)
    )
    block_grad_lse = (
        grad_lse.float()
        .view(batch, query_heads, num_blocks, int(block_size))
        .permute(0, 2, 1, 3)
    )
    grad_value_expanded = torch.matmul(probs.transpose(-2, -1), block_grad_output)
    grad_probs = torch.matmul(block_grad_output, expanded_value.transpose(-2, -1))
    row_dot = (block_grad_output * block_output).sum(dim=-1, keepdim=True)
    grad_scores = probs * (grad_probs - row_dot + block_grad_lse.unsqueeze(-1))
    grad_query = torch.matmul(grad_scores, expanded_key) * float(scale)
    grad_key_expanded = torch.matmul(
        grad_scores.transpose(-2, -1), block_query
    ) * float(scale)
    grad_key = grad_key_expanded.reshape(
        batch,
        num_blocks,
        kv_heads,
        group_size,
        int(block_size),
        head_dim,
    ).sum(dim=3)
    grad_value = grad_value_expanded.reshape(
        batch,
        num_blocks,
        kv_heads,
        group_size,
        int(block_size),
        head_dim,
    ).sum(dim=3)
    grad_query = (
        grad_query.reshape(batch, num_blocks, query_heads, int(block_size), head_dim)
        .permute(0, 2, 1, 3, 4)
        .reshape(batch, query_heads, active_len, head_dim)
    )
    grad_key = grad_key.permute(0, 2, 1, 3, 4).reshape(
        batch, kv_heads, active_len, head_dim
    )
    grad_value = grad_value.permute(0, 2, 1, 3, 4).reshape(
        batch, kv_heads, active_len, head_dim
    )
    _debug_check_backward_finite(
        "block_diagonal_torch",
        grad_query,
        grad_key,
        grad_value,
        query=query,
        key=key,
        key_start=0,
        block_size=int(block_size),
        debug_nonfinite_attention=bool(debug_nonfinite_attention),
    )
    return grad_query, grad_key, grad_value


def _ring_context_backward_flex(
    *,
    query: torch.Tensor,
    local_key: torch.Tensor,
    local_value: torch.Tensor,
    grad_output: torch.Tensor,
    final_output: torch.Tensor,
    final_m: torch.Tensor,
    final_l: torch.Tensor,
    attn_mask: Any | None,
    is_causal: bool,
    scale: float,
    local_rank: int,
    group_ranks: list[int],
    shard_lengths: list[int],
    intervals_by_owner: tuple[tuple[tuple[int, int], ...], ...],
    max_shard_len: int,
    group: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if is_causal or not isinstance(
        attn_mask, (BlockDenoisingGlobalCleanMask, BlockDenoisingFullMask)
    ):
        raise RuntimeError(
            "ring context backward requires a structured block-denoising mask"
        )
    num_context_ranks = len(group_ranks)
    current_flat, current_key, current_value = _make_kv_ring_payload(
        _pad_kv_shard(local_key, max_shard_len),
        _pad_kv_shard(local_value, max_shard_len),
    )
    current_grad_flat = torch.zeros_like(current_flat)
    current_grad_key = current_grad_flat[: current_key.numel()].view(current_key.shape)
    current_grad_value = current_grad_flat[current_key.numel() :].view(
        current_value.shape
    )
    final_lse = _final_lse_from_stats(final_m.float(), final_l.float())
    grad_query = torch.zeros_like(query, dtype=torch.float32)

    for step in range(num_context_ranks):
        owner = (local_rank + step) % num_context_ranks
        shard_len = shard_lengths[owner]
        if step != num_context_ranks - 1:
            next_kv_work = _ring_exchange_kv_payload_async(
                current_flat,
                key_shape=current_key.shape,
                key_numel=current_key.numel(),
                local_rank=local_rank,
                group_ranks=group_ranks,
                group=group,
                phase="backward",
            )
        else:
            next_kv_work = None

        if shard_len > 0:
            offset = 0
            for key_start, key_stop in intervals_by_owner[owner]:
                chunk_len = key_stop - key_start
                if chunk_len <= 0:
                    continue
                shard_key = current_key[
                    ..., offset : offset + chunk_len, :
                ].contiguous()
                shard_value = current_value[
                    ..., offset : offset + chunk_len, :
                ].contiguous()
                bdlm_key_start = _bdlm_encoded_key_start(attn_mask, key_start)
                query_indices = _owner_query_indices(
                    attn_mask=attn_mask,
                    is_causal=is_causal,
                    query_len=query.shape[-2],
                    key_start=key_start,
                    device=query.device,
                )
                if query_indices is not None and query_indices.numel() == 0:
                    offset += chunk_len
                    continue
                if query_indices is None:
                    grad_q, grad_k, grad_v = _bdlm_flex_shard_backward_exact(
                        query=query,
                        key=shard_key,
                        value=shard_value,
                        final_output=final_output,
                        final_lse=final_lse,
                        grad_output=grad_output,
                        query_blocks=attn_mask.query_blocks,
                        query_is_clean=attn_mask.query_is_clean,
                        block_size=int(attn_mask.block_size),
                        backward_query_chunk_size=int(
                            attn_mask.backward_query_chunk_size
                        ),
                        key_start=int(bdlm_key_start),
                        scale=float(scale),
                        clean_offset=int(_bdlm_clean_offset(attn_mask)),
                        flex_cache=attn_mask.flex_cache,
                        debug_nonfinite_attention=bool(
                            getattr(attn_mask, "debug_nonfinite_attention", False)
                        ),
                    )
                    grad_query = grad_query + grad_q.float()
                else:
                    grad_q, grad_k, grad_v = _compacted_bdlm_shard_backward_exact(
                        query=query,
                        key=shard_key,
                        value=shard_value,
                        final_output=final_output,
                        final_lse=final_lse,
                        grad_output=grad_output,
                        attn_mask=attn_mask,
                        query_indices=query_indices,
                        key_start=key_start,
                        scale=float(scale),
                    )
                    grad_query.index_add_(-2, query_indices, grad_q.float())
                current_grad_key[..., offset : offset + chunk_len, :].add_(
                    grad_k.to(dtype=current_grad_key.dtype)
                )
                current_grad_value[..., offset : offset + chunk_len, :].add_(
                    grad_v.to(dtype=current_grad_value.dtype)
                )
                offset += chunk_len

        next_grad_work = _ring_exchange_kv_payload_async(
            current_grad_flat,
            key_shape=current_grad_key.shape,
            key_numel=current_grad_key.numel(),
            local_rank=local_rank,
            group_ranks=group_ranks,
            group=group,
            phase="backward",
        )
        if step != num_context_ranks - 1:
            assert next_kv_work is not None
            current_flat, current_key, current_value = _ring_exchange_kv_payload_wait(
                next_kv_work
            )
        current_grad_flat, current_grad_key, current_grad_value = (
            _ring_exchange_kv_payload_wait(next_grad_work)
        )

    local_len = int(shard_lengths[local_rank])
    return (
        grad_query,
        current_grad_key[..., :local_len, :],
        current_grad_value[..., :local_len, :],
    )


def _ring_clean_backward_bshd(
    *,
    query: torch.Tensor,
    local_key: torch.Tensor,
    local_value: torch.Tensor,
    backward_key: torch.Tensor,
    backward_value: torch.Tensor,
    grad_output: torch.Tensor,
    final_output: torch.Tensor,
    final_lse: torch.Tensor,
    local_attn_mask: BlockDenoisingLocalActiveMask | None,
    attn_mask: BlockDenoisingGlobalCleanMask | BlockDenoisingPackedKeyMask,
    scale: float,
    local_rank: int,
    group_ranks: list[int],
    shard_lengths: list[int],
    intervals_by_owner: tuple[tuple[tuple[int, int], ...], ...],
    max_shard_len: int,
    group: Any,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Stream clean K/V and its gradient accumulator back to each owner."""

    world_size = len(group_ranks)
    current_flat, current_key, current_value = _make_kv_ring_payload(
        backward_key,
        backward_value,
    )
    key_numel = current_key.numel()
    comm_buffers = torch.empty(
        (2, 2, current_flat.numel()),
        device=current_flat.device,
        dtype=current_flat.dtype,
    )
    comm_buffers[0, 0].copy_(current_flat)
    local_grad_flat = torch.zeros_like(current_flat)
    local_grad_key = local_grad_flat[:key_numel].view(current_key.shape)
    local_grad_value = local_grad_flat[key_numel:].view(current_value.shape)
    grad_query = torch.zeros_like(query, dtype=torch.float32)
    local_active_grad_key = torch.zeros_like(local_key)
    local_active_grad_value = torch.zeros_like(local_value)
    returned_grad_flat = None

    for step in range(world_size):
        owner = (local_rank + step + 1) % world_size
        send_buffer = comm_buffers[step % 2]
        recv_buffer = comm_buffers[(step + 1) % 2]
        if step == 0:
            send_payload = send_buffer[0]
            recv_payload = recv_buffer[0]
        elif step == world_size - 1:
            send_payload = send_buffer[1]
            recv_payload = recv_buffer[1]
        else:
            send_payload = send_buffer
            recv_payload = recv_buffer
        exchange_work = _ring_exchange_flat_async(
            send_payload,
            recv_payload,
            local_rank=local_rank,
            send_rank=group_ranks[(local_rank - 1) % world_size],
            recv_rank=group_ranks[(local_rank + 1) % world_size],
            group=group,
            communication_phase="backward",
        )
        current_flat = send_buffer[0]
        current_key = current_flat[:key_numel].view(backward_key.shape)
        current_value = current_flat[key_numel:].view(backward_value.shape)

        local_grad_flat.zero_()
        shard_len = int(shard_lengths[owner])
        active_len = int(local_key.shape[1]) if owner == local_rank else 0
        if active_len:
            if not isinstance(local_attn_mask, BlockDenoisingLocalActiveMask):
                raise RuntimeError("fused BP/CP backward requires an active mask")
            shard_key = torch.cat((local_key, current_key[:, :shard_len]), dim=1)
            shard_value = torch.cat(
                (local_value, current_value[:, :shard_len]),
                dim=1,
            )
            shard_mask: BlockDenoisingGlobalCleanMask | BlockDenoisingPackedKeyMask = (
                _packed_local_owner_mask(
                    attn_mask,
                    local_attn_mask,
                    intervals=intervals_by_owner[owner],
                    device=query.device,
                )
            )
            key_start = 0
            query_indices = None
        else:
            shard_key = current_key[:, :shard_len]
            shard_value = current_value[:, :shard_len]
            shard_mask, key_start = _packed_clean_owner_mask(
                attn_mask,
                intervals=intervals_by_owner[owner],
                device=query.device,
            )
            query_indices = _owner_query_indices(
                attn_mask=attn_mask,
                is_causal=False,
                query_len=int(query.shape[1]),
                key_start=int(key_start),
                device=query.device,
            )
        grad_q, grad_k, grad_v = _packed_clean_flex_shard_backward_exact_bshd(
            query=query,
            key=shard_key,
            value=shard_value,
            final_output=final_output,
            final_lse=final_lse,
            grad_output=grad_output,
            attn_mask=shard_mask,
            key_start=int(key_start),
            scale=float(scale),
            query_indices=query_indices,
            query_cache_key=("clean_owner", int(owner)),
        )
        if query_indices is None:
            grad_query.add_(grad_q.float())
        else:
            grad_query.index_add_(1, query_indices, grad_q.float())
        if active_len:
            local_active_grad_key.copy_(grad_k[:, :active_len])
            local_active_grad_value.copy_(grad_v[:, :active_len])
            grad_k = grad_k[:, active_len:]
            grad_v = grad_v[:, active_len:]
        local_grad_key[:, :shard_len].copy_(grad_k.to(dtype=local_grad_key.dtype))
        local_grad_value[:, :shard_len].copy_(grad_v.to(dtype=local_grad_value.dtype))

        _ring_exchange_flat_wait(exchange_work)
        returned_grad_flat = recv_buffer[1]
        if step == 0:
            returned_grad_flat.copy_(local_grad_flat)
        else:
            returned_grad_flat.add_(local_grad_flat)

    assert returned_grad_flat is not None
    returned_grad_key = returned_grad_flat[:key_numel].view(backward_key.shape)
    returned_grad_value = returned_grad_flat[key_numel:].view(backward_value.shape)
    local_len = int(shard_lengths[local_rank])
    return (
        grad_query,
        local_active_grad_key,
        local_active_grad_value,
        returned_grad_key[:, :local_len],
        returned_grad_value[:, :local_len],
    )


def _ring_context_backward_bshd(
    *,
    query: torch.Tensor,
    local_key: torch.Tensor,
    local_value: torch.Tensor,
    grad_output: torch.Tensor,
    final_output: torch.Tensor,
    final_lse: torch.Tensor,
    attn_mask: BlockDenoisingFullMask,
    scale: float,
    local_rank: int,
    group_ranks: list[int],
    shard_lengths: list[int],
    intervals_by_owner: tuple[tuple[tuple[int, int], ...], ...],
    max_shard_len: int,
    group: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_context_ranks = len(group_ranks)
    local_kv_flat, local_key_padded, local_value_padded = _make_kv_ring_payload(
        _pad_kv_shard_bshd(local_key, max_shard_len),
        _pad_kv_shard_bshd(local_value, max_shard_len),
    )
    key_numel = local_key_padded.numel()
    comm_buffers = torch.empty(
        (2, 2, local_kv_flat.numel()),
        device=local_kv_flat.device,
        dtype=local_kv_flat.dtype,
    )
    comm_buffers[0, 0].copy_(local_kv_flat)
    local_grad_flat = torch.zeros_like(local_kv_flat)
    local_grad_key = local_grad_flat[:key_numel].view(local_key_padded.shape)
    local_grad_value = local_grad_flat[key_numel:].view(local_value_padded.shape)
    grad_query = torch.zeros_like(query, dtype=torch.float32)
    returned_grad_flat = None

    for step in range(num_context_ranks):
        owner = (local_rank + step + 1) % num_context_ranks
        send_buffer = comm_buffers[step % 2]
        recv_buffer = comm_buffers[(step + 1) % 2]
        if step == 0:
            send_payload = send_buffer[0]
            recv_payload = recv_buffer[0]
        elif step == num_context_ranks - 1:
            send_payload = send_buffer[1]
            recv_payload = recv_buffer[1]
        else:
            send_payload = send_buffer
            recv_payload = recv_buffer
        exchange_work = _ring_exchange_flat_async(
            send_payload,
            recv_payload,
            local_rank=local_rank,
            send_rank=group_ranks[(local_rank - 1) % num_context_ranks],
            recv_rank=group_ranks[(local_rank + 1) % num_context_ranks],
            group=group,
            communication_phase="backward",
        )
        current_flat = send_buffer[0]
        current_key = current_flat[:key_numel].view(local_key_padded.shape)
        current_value = current_flat[key_numel:].view(local_value_padded.shape)

        local_grad_flat.zero_()
        offset = 0
        for key_start, key_stop in intervals_by_owner[owner]:
            chunk_len = int(key_stop - key_start)
            if chunk_len <= 0:
                continue
            query_indices = _full_mask_interval_query_indices(
                attn_mask=attn_mask,
                query_len=int(query.shape[1]),
                key_start=int(key_start),
                key_stop=int(key_stop),
                device=query.device,
            )
            if query_indices is not None and query_indices.numel() == 0:
                offset += chunk_len
                continue
            if query_indices is None:
                shard_query = query
                shard_output = final_output
                shard_lse = final_lse
                shard_grad_output = grad_output
                shard_blocks = attn_mask.query_blocks
                shard_is_clean = attn_mask.query_is_clean
            else:
                shard_query = query.index_select(1, query_indices)
                shard_output = final_output.index_select(1, query_indices)
                shard_lse = final_lse.index_select(-1, query_indices)
                shard_grad_output = grad_output.index_select(1, query_indices)
                shard_blocks = attn_mask.query_blocks.index_select(-1, query_indices)
                shard_is_clean = attn_mask.query_is_clean.index_select(
                    -1,
                    query_indices,
                )
            grad_q, grad_k, grad_v = _bdlm_flex_shard_backward_exact_bshd(
                query=shard_query,
                key=current_key[:, offset : offset + chunk_len],
                value=current_value[:, offset : offset + chunk_len],
                final_output=shard_output,
                final_lse=shard_lse,
                grad_output=shard_grad_output,
                query_blocks=shard_blocks,
                query_is_clean=shard_is_clean,
                block_size=int(attn_mask.block_size),
                key_start=int(_bdlm_encoded_key_start(attn_mask, key_start)),
                scale=float(scale),
                clean_offset=int(attn_mask.clean_offset),
                flex_cache=attn_mask.flex_cache,
                debug_nonfinite_attention=bool(attn_mask.debug_nonfinite_attention),
            )
            if query_indices is None:
                grad_query.add_(grad_q.float())
            else:
                grad_query.index_add_(1, query_indices, grad_q.float())
            local_grad_key[:, offset : offset + chunk_len].add_(
                grad_k.to(dtype=local_grad_key.dtype)
            )
            local_grad_value[:, offset : offset + chunk_len].add_(
                grad_v.to(dtype=local_grad_value.dtype)
            )
            offset += chunk_len

        _ring_exchange_flat_wait(exchange_work)
        returned_grad_flat = recv_buffer[1]
        if step == 0:
            returned_grad_flat.copy_(local_grad_flat)
        else:
            returned_grad_flat.add_(local_grad_flat)

    assert returned_grad_flat is not None
    returned_grad_key = returned_grad_flat[:key_numel].view(local_key_padded.shape)
    returned_grad_value = returned_grad_flat[key_numel:].view(local_value_padded.shape)
    local_len = int(shard_lengths[local_rank])
    return (
        grad_query,
        returned_grad_key[:, :local_len],
        returned_grad_value[:, :local_len],
    )


def _ring_hybrid_backward_flex(
    *,
    query: torch.Tensor,
    local_key: torch.Tensor,
    local_value: torch.Tensor,
    global_key_shard: torch.Tensor,
    global_value_shard: torch.Tensor,
    grad_output: torch.Tensor,
    final_output: torch.Tensor,
    final_m: torch.Tensor,
    final_l: torch.Tensor,
    local_attn_mask: torch.Tensor | None,
    global_attn_mask: torch.Tensor | None,
    scale: float,
    global_key_shape: torch.Size,
    global_value_shape: torch.Size,
    local_rank: int,
    group_ranks: list[int],
    shard_lengths: list[int],
    intervals_by_owner: tuple[tuple[tuple[int, int], ...], ...],
    max_shard_len: int,
    group: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(global_attn_mask, BlockDenoisingGlobalCleanMask):
        raise RuntimeError(
            "hybrid ring backward requires a structured global clean mask"
        )
    num_context_ranks = len(group_ranks)
    current_flat, current_key, current_value = _make_kv_ring_payload(
        _pad_kv_shard(global_key_shard, max_shard_len),
        _pad_kv_shard(global_value_shard, max_shard_len),
    )
    final_lse = _final_lse_from_stats(final_m.float(), final_l.float())
    grad_query = torch.zeros_like(query, dtype=torch.float32)
    if local_key.shape[-2] > 0:
        if not isinstance(local_attn_mask, BlockDenoisingLocalActiveMask):
            raise RuntimeError(
                "hybrid ring backward requires a structured local active mask"
            )
        active_len = local_key.shape[-2]
        active_output, active_m, active_l = _block_diagonal_active_stats(
            query=query[..., :active_len, :],
            key=local_key,
            value=local_value,
            block_size=int(local_attn_mask.block_size),
            scale=float(scale),
        )
        local_grad_output, local_grad_lse = _merged_shard_grads(
            shard_output=active_output,
            shard_lse=_final_lse_from_stats(active_m.float(), active_l.float()),
            final_output=final_output[..., :active_len, :],
            final_lse=final_lse[..., :active_len],
            grad_output=grad_output[..., :active_len, :],
        )
        active_block_size = int(local_attn_mask.block_size)
        active_block_indices = _nonzero_active_grad_block_indices(
            grad_output=local_grad_output,
            grad_lse=local_grad_lse,
            block_size=active_block_size,
        )
        if active_block_indices is None:
            local_grad_query_active, local_grad_key, local_grad_value = (
                _block_diagonal_active_backward(
                    query=query[..., :active_len, :],
                    key=local_key,
                    value=local_value,
                    output=active_output,
                    lse=active_m,
                    grad_output=local_grad_output,
                    grad_lse=local_grad_lse,
                    block_size=active_block_size,
                    scale=float(scale),
                    backward_query_chunk_size=int(
                        local_attn_mask.backward_query_chunk_size
                    ),
                    debug_nonfinite_attention=bool(
                        local_attn_mask.debug_nonfinite_attention
                    ),
                    flex_cache=local_attn_mask.flex_cache,
                )
            )
        elif active_block_indices.numel() == 0:
            local_grad_query_active = torch.zeros_like(
                query[..., :active_len, :],
                dtype=torch.float32,
            )
            local_grad_key = torch.zeros_like(local_key, dtype=torch.float32)
            local_grad_value = torch.zeros_like(local_value, dtype=torch.float32)
        else:
            compact_query_indices = _block_token_indices(
                active_block_indices,
                block_size=active_block_size,
                device=query.device,
            )
            compact_grad_q, compact_grad_k, compact_grad_v = (
                _block_diagonal_active_backward(
                    query=query[..., :active_len, :].index_select(
                        -2, compact_query_indices
                    ),
                    key=local_key.index_select(-2, compact_query_indices),
                    value=local_value.index_select(-2, compact_query_indices),
                    output=active_output.index_select(-2, compact_query_indices),
                    lse=active_m.index_select(-1, compact_query_indices),
                    grad_output=local_grad_output.index_select(
                        -2, compact_query_indices
                    ),
                    grad_lse=local_grad_lse.index_select(-1, compact_query_indices),
                    block_size=active_block_size,
                    scale=float(scale),
                    backward_query_chunk_size=int(
                        local_attn_mask.backward_query_chunk_size
                    ),
                    debug_nonfinite_attention=bool(
                        local_attn_mask.debug_nonfinite_attention
                    ),
                )
            )
            local_grad_query_active = torch.zeros_like(
                query[..., :active_len, :],
                dtype=torch.float32,
            )
            local_grad_key = torch.zeros_like(local_key, dtype=torch.float32)
            local_grad_value = torch.zeros_like(local_value, dtype=torch.float32)
            local_grad_query_active.index_copy_(
                -2, compact_query_indices, compact_grad_q.float()
            )
            local_grad_key.index_copy_(
                -2, compact_query_indices, compact_grad_k.float()
            )
            local_grad_value.index_copy_(
                -2, compact_query_indices, compact_grad_v.float()
            )
        grad_query[..., :active_len, :] += local_grad_query_active.float()
        local_grad_key = local_grad_key.float()
        local_grad_value = local_grad_value.float()
    else:
        local_grad_key = torch.zeros_like(local_key, dtype=torch.float32)
        local_grad_value = torch.zeros_like(local_value, dtype=torch.float32)

    owner_grad_flat, owner_grad_keys, owner_grad_values = (
        _allocate_owner_grad_flat_buffer(
            global_key_shape,
            global_value_shape,
            max_shard_len=max_shard_len,
            num_context_ranks=num_context_ranks,
            device=query.device,
            dtype=global_key_shard.dtype,
        )
    )

    for step in range(num_context_ranks):
        owner = (local_rank + step) % num_context_ranks
        shard_len = shard_lengths[owner]
        if step != num_context_ranks - 1:
            next_kv_work = _ring_exchange_kv_payload_async(
                current_flat,
                key_shape=current_key.shape,
                key_numel=current_key.numel(),
                local_rank=local_rank,
                group_ranks=group_ranks,
                group=group,
                phase="backward",
            )
        else:
            next_kv_work = None

        if shard_len > 0:
            offset = 0
            for key_start, key_stop in intervals_by_owner[owner]:
                chunk_len = key_stop - key_start
                if chunk_len <= 0:
                    continue
                shard_key = current_key[
                    ..., offset : offset + chunk_len, :
                ].contiguous()
                shard_value = current_value[
                    ..., offset : offset + chunk_len, :
                ].contiguous()
                query_indices = _owner_query_indices(
                    attn_mask=global_attn_mask,
                    is_causal=False,
                    query_len=query.shape[-2],
                    key_start=key_start,
                    device=query.device,
                )
                if query_indices is not None and query_indices.numel() == 0:
                    offset += chunk_len
                    continue
                if query_indices is None:
                    grad_q, grad_k, grad_v = _bdlm_flex_shard_backward_exact(
                        query=query,
                        key=shard_key,
                        value=shard_value,
                        final_output=final_output,
                        final_lse=final_lse,
                        grad_output=grad_output,
                        query_blocks=global_attn_mask.query_blocks,
                        query_is_clean=global_attn_mask.query_is_clean,
                        block_size=int(global_attn_mask.block_size),
                        backward_query_chunk_size=int(
                            global_attn_mask.backward_query_chunk_size
                        ),
                        key_start=int(key_start),
                        scale=float(scale),
                        clean_offset=int(_bdlm_clean_offset(global_attn_mask)),
                        flex_cache=global_attn_mask.flex_cache,
                        debug_nonfinite_attention=bool(
                            global_attn_mask.debug_nonfinite_attention
                        ),
                    )
                    grad_query = grad_query + grad_q.float()
                else:
                    grad_q, grad_k, grad_v = _compacted_bdlm_shard_backward_exact(
                        query=query,
                        key=shard_key,
                        value=shard_value,
                        final_output=final_output,
                        final_lse=final_lse,
                        grad_output=grad_output,
                        attn_mask=global_attn_mask,
                        query_indices=query_indices,
                        key_start=key_start,
                        scale=float(scale),
                    )
                    grad_query.index_add_(-2, query_indices, grad_q.float())
                owner_grad_keys[owner, ..., offset : offset + chunk_len, :] = grad_k
                owner_grad_values[owner, ..., offset : offset + chunk_len, :] = grad_v
                offset += chunk_len

        if step != num_context_ranks - 1:
            assert next_kv_work is not None
            current_flat, current_key, current_value = _ring_exchange_kv_payload_wait(
                next_kv_work
            )

    reduced_key, reduced_value = _ring_return_owner_grads_p2p_flat(
        owner_grad_flat,
        key_shape=owner_grad_keys[local_rank].shape,
        key_numel=owner_grad_keys[local_rank].numel(),
        local_rank=local_rank,
        group_ranks=group_ranks,
        group=group,
    )
    return grad_query, local_grad_key, local_grad_value, reduced_key, reduced_value


def _merge_online_stats(
    old_num: torch.Tensor,
    old_m: torch.Tensor,
    old_l: torch.Tensor,
    new_num: torch.Tensor,
    new_m: torch.Tensor,
    new_l: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if (
        old_num.is_cuda
        and old_num.dtype == torch.float32
        and new_num.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and old_m.dtype == torch.float32
        and old_l.dtype == torch.float32
        and new_m.dtype == torch.float32
        and new_l.dtype == torch.float32
        and old_num.is_contiguous()
        and old_m.is_contiguous()
        and old_l.is_contiguous()
        and new_num.is_contiguous()
        and new_m.is_contiguous()
        and new_l.is_contiguous()
    ):
        cp_fusion.merge_full_(old_num, old_m, old_l, new_num, new_m, new_l)
        return old_num, old_m, old_l
    next_m = torch.maximum(old_m, new_m)
    next_m_safe = torch.where(
        torch.isfinite(next_m),
        next_m,
        torch.zeros_like(next_m),
    )
    old_weight = torch.where(
        torch.isfinite(old_m),
        torch.exp(old_m - next_m_safe),
        torch.zeros_like(old_m),
    )
    new_weight = torch.where(
        torch.isfinite(new_m),
        torch.exp(new_m - next_m_safe),
        torch.zeros_like(new_m),
    )
    old_num.mul_(old_weight.unsqueeze(-1))
    new_num.mul_(new_weight.unsqueeze(-1))
    old_num.add_(new_num)
    old_l.mul_(old_weight)
    new_l.mul_(new_weight)
    old_l.add_(new_l)
    return old_num, next_m, old_l


def _apply_masks(
    *,
    scores: torch.Tensor,
    attn_mask: torch.Tensor | None,
    is_causal: bool,
    q_positions: torch.Tensor,
    key_start: int,
    key_stop: int,
) -> torch.Tensor:
    if is_causal:
        key_positions = torch.arange(key_start, key_stop, device=scores.device)
        causal_mask = key_positions.unsqueeze(0) <= q_positions.unsqueeze(1)
        scores = scores.masked_fill(~causal_mask, -torch.inf)

    if attn_mask is None:
        return scores

    mask_chunk = attn_mask[..., key_start:key_stop]
    if mask_chunk.dtype == torch.bool:
        scores = scores.masked_fill(~mask_chunk, -torch.inf)
    else:
        scores = scores + mask_chunk
    return scores


def _allocate_owner_grad_buffer(
    tensor_shape: torch.Size,
    *,
    max_shard_len: int,
    num_context_ranks: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if max_shard_len < 0:
        raise ValueError("max_shard_len must be non-negative")
    if num_context_ranks <= 0:
        raise ValueError("num_context_ranks must be positive")
    return torch.zeros(
        (num_context_ranks, *tensor_shape[:-2], max_shard_len, tensor_shape[-1]),
        device=device,
        dtype=dtype,
    )


def _allocate_owner_grad_flat_buffer(
    key_shape: torch.Size,
    value_shape: torch.Size,
    *,
    max_shard_len: int,
    num_context_ranks: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if key_shape[:-2] != value_shape[:-2] or key_shape[-1] != value_shape[-1]:
        raise ValueError("key/value gradient shapes must match except sequence length")
    key_buffer_shape = (
        num_context_ranks,
        *key_shape[:-2],
        max_shard_len,
        key_shape[-1],
    )
    value_buffer_shape = (
        num_context_ranks,
        *value_shape[:-2],
        max_shard_len,
        value_shape[-1],
    )
    key_numel = math.prod(key_buffer_shape[1:])
    value_numel = math.prod(value_buffer_shape[1:])
    flat = torch.zeros(
        (num_context_ranks, key_numel + value_numel),
        device=device,
        dtype=dtype,
    )
    key_view = flat[:, :key_numel].view(key_buffer_shape)
    value_view = flat[:, key_numel:].view(value_buffer_shape)
    return flat, key_view, value_view


def _pad_kv_shard(tensor: torch.Tensor, padded_len: int) -> torch.Tensor:
    pad_len = padded_len - tensor.shape[-2]
    if pad_len < 0:
        raise ValueError("padded_len must be at least the shard length")
    if pad_len == 0:
        return tensor
    pad_shape = (*tensor.shape[:-2], pad_len, tensor.shape[-1])
    padding = torch.zeros(pad_shape, device=tensor.device, dtype=tensor.dtype)
    return torch.cat((tensor, padding), dim=-2)


def _pad_kv_shard_bshd(tensor: torch.Tensor, padded_len: int) -> torch.Tensor:
    pad_len = int(padded_len) - int(tensor.shape[1])
    if pad_len < 0:
        raise ValueError("padded_len must be at least the BSHD shard length")
    if pad_len == 0:
        return tensor
    padding = torch.zeros(
        (tensor.shape[0], pad_len, tensor.shape[2], tensor.shape[3]),
        device=tensor.device,
        dtype=tensor.dtype,
    )
    return torch.cat((tensor, padding), dim=1)


def _should_use_context_parallel(runtime: Any | None) -> bool:
    return (
        runtime is not None
        and bool(getattr(runtime, "enabled", False))
        and int(getattr(runtime, "local_parallel_size", 1)) > 1
        and dist.is_available()
        and dist.is_initialized()
    )


def _validate_bhsd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> None:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must have shape [B, H, S, D]")
    if query.shape[0] != key.shape[0] or key.shape[:2] != value.shape[:2]:
        raise ValueError(
            "query, key, and value must share batch dimensions and key/value heads"
        )
    if query.shape[1] % key.shape[1] != 0:
        raise ValueError("query heads must be divisible by key/value heads")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key head dimensions must match")
    if key.shape[-2] != value.shape[-2] or key.shape[-1] != value.shape[-1]:
        raise ValueError("key and value must share sequence/head dimensions")


def _validate_bshd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> None:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must have shape [B, S, H, D]")
    if query.shape[0] != key.shape[0] or key.shape[0] != value.shape[0]:
        raise ValueError("query, key, and value must share batch dimensions")
    if query.shape[2] % key.shape[2] != 0:
        raise ValueError("query heads must be divisible by key/value heads")
    if key.shape[2] != value.shape[2]:
        raise ValueError("key and value must share key/value heads")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key head dimensions must match")
    if key.shape[1] != value.shape[1] or key.shape[-1] != value.shape[-1]:
        raise ValueError("key and value must share sequence/head dimensions")
