# Copyright 2026 The bdlm_parallel Authors.
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

import math
import os
import tempfile
import traceback
from contextlib import nullcontext
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.multiprocessing as mp
import pytest
from torch.utils.checkpoint import checkpoint

import dllm_parallel.core.attention.context_parallel_attention as cp_attention
from dllm_parallel.core.attention.flex import _require_native_pack_gqa_backward
from dllm_parallel.core.attention.full_mask import (
    full_mask_block_denoising_attention_bshd,
)
from dllm_parallel.core.attention.context_parallel_attention import (
    BlockDenoisingFullMask,
    BlockDenoisingGlobalCleanMask,
    BlockDenoisingLocalActiveMask,
    BlockDenoisingPackedKeyMask,
    _allocate_owner_grad_buffer,
    _block_denoising_mask_spec_allowed,
    _bdlm_flex_shard_attention_bshd,
    _decode_bdlm_key_start,
    _full_mask_interval_query_indices,
    _owner_query_indices,
    _ragged_prefix_bdlm_attention_bshd,
    all_clean_shards,
    clean_shards_for_rank,
    context_parallel_attention,
    cp_bp_counters_from_visits,
    merge_attention_stats,
    prefix_visits_for_noisy_blocks,
    replicated_block_denoising_attention_bshd,
    replicated_kv_attention,
    ring_owner_traversal,
    ring_context_parallel_attention,
    ring_attention_with_local_kv,
    fused_block_context_attention_bshd,
    shard_bounds,
    streaming_attention,
)
from dllm_parallel.core.attention.layout import (
    active_query_indices_for_context_rank,
    all_context_parallel_sequence_intervals,
    context_parallel_sequence_layout,
    gather_context_parallel_sequence,
    runtime_clean_layout,
    windowed_clean_exchange_plan,
)
from dllm_parallel.core.attention.ring_transport import (
    _block_current_stream_or_wait,
    _ring_exchange_flat_async,
    _ring_exchange_flat_wait,
)
from dllm_parallel.core.schedules.block import build_block_schedule
from dllm_parallel.core.specs import CPBPPolicy


def test_pure_cp_active_query_indices_shard_offsets_inside_every_block() -> None:
    expected = (
        [0, 7, 8, 15],
        [1, 6, 9, 14],
        [2, 5, 10, 13],
        [3, 4, 11, 12],
    )

    actual = tuple(
        active_query_indices_for_context_rank(
            active_len=16,
            block_size=8,
            context_parallel_size=4,
            context_parallel_rank=rank,
            device=torch.device("cpu"),
        ).tolist()
        for rank in range(4)
    )

    assert actual == expected
    assert sorted(index for shard in actual for index in shard) == list(range(16))
    assert all({index // 8 for index in shard} == {0, 1} for shard in actual)


def test_compact_block_active_attention_preserves_block_major_query_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = torch.arange(1 * 4 * 2 * 4, dtype=torch.float32).view(1, 4, 2, 4)
    key = torch.zeros(1, 8, 1, 4)
    value = torch.zeros_like(key)

    def fake_attention(
        block_query: torch.Tensor,
        block_key: torch.Tensor,
        block_value: torch.Tensor,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del block_key, block_value, scale
        lse = torch.arange(
            block_query.shape[0] * block_query.shape[1] * block_query.shape[2],
            dtype=torch.float32,
        ).view(block_query.shape[0], block_query.shape[2], block_query.shape[1])
        return block_query + 100.0, lse

    monkeypatch.setattr(
        cp_attention,
        "_flex_full_shard_attention_bshd",
        fake_attention,
    )

    numerator, m, l = cp_attention._block_diagonal_active_query_shard_stats_bshd(
        query=query,
        key=key,
        value=value,
        block_size=4,
        scale=0.5,
    )

    torch.testing.assert_close(numerator, (query + 100.0).transpose(1, 2))
    assert m.shape == (1, 2, 4)
    torch.testing.assert_close(l, torch.ones_like(m))


def test_compact_block_active_attention_backward_restores_full_kv_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = torch.arange(1 * 4 * 2 * 4, dtype=torch.float32).view(1, 4, 2, 4)
    key = torch.arange(1 * 8 * 1 * 4, dtype=torch.float32).view(1, 8, 1, 4)
    value = key + 20.0
    output = query + 40.0
    lse = torch.zeros(1, 2, 4)
    grad_output = torch.ones_like(output)

    def fake_backward(**kwargs: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        block_query = kwargs["query"]
        block_key = kwargs["key"]
        block_value = kwargs["value"]
        assert isinstance(block_query, torch.Tensor)
        assert isinstance(block_key, torch.Tensor)
        assert isinstance(block_value, torch.Tensor)
        return block_query + 1.0, block_key + 2.0, block_value + 3.0

    monkeypatch.setattr(cp_attention, "_flex_merged_shard_backward", fake_backward)

    grad_query, grad_key, grad_value = (
        cp_attention._block_diagonal_active_query_shard_backward_bshd(
            query=query,
            key=key,
            value=value,
            output=output,
            lse=lse,
            grad_output=grad_output,
            block_size=4,
            scale=0.5,
            debug_nonfinite_attention=False,
        )
    )

    torch.testing.assert_close(grad_query, query + 1.0)
    torch.testing.assert_close(grad_key, key + 2.0)
    torch.testing.assert_close(grad_value, value + 3.0)


def test_ring_work_uses_stream_dependency_without_host_wait() -> None:
    class Work:
        blocked = False
        waited = False

        def block_current_stream(self) -> None:
            self.blocked = True

        def wait(self) -> None:
            self.waited = True

    work = Work()
    _block_current_stream_or_wait(work)

    assert work.blocked
    assert not work.waited


def test_flex_attention_compiles_for_static_production_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compile_calls: list[dict[str, object]] = []

    def fake_compile(target, **kwargs):
        compile_calls.append(kwargs)
        return target

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(cp_attention, "flex_attention", lambda *args, **kwargs: None)
    cp_attention._compiled_flex_attention.clear()
    try:
        cp_attention._get_compiled_flex_attention(
            kernel_family="full",
            differentiable=True,
        )
    finally:
        cp_attention._compiled_flex_attention.clear()

    assert compile_calls == [
        {
            "dynamic": False,
            "fullgraph": True,
            "mode": "max-autotune-no-cudagraphs",
        }
    ]


def test_fa4_mask_metadata_uses_declared_integer_and_boolean_types() -> None:
    mask = cp_attention.BlockDenoisingPackedKeyMask(
        query_blocks=torch.tensor([1, 1], dtype=torch.int64),
        local_query_blocks=torch.tensor([1, 1], dtype=torch.int64),
        query_is_clean=torch.tensor([False, True]),
        active_key_blocks=torch.tensor([1, 1], dtype=torch.int64),
        clean_key_blocks=torch.tensor([0, 1], dtype=torch.int64),
        block_size=2,
        query_clean_bounds=torch.tensor([[0, 2], [0, 3]], dtype=torch.int64),
        clean_key_positions=torch.tensor([0, 2], dtype=torch.int64),
    )

    buffers = cp_attention._flex_backward_mask_buffers(
        attn_mask=mask,
        key_start=0,
        key_len=4,
        device=torch.device("cpu"),
    )

    assert [buffer.dtype for buffer in buffers] == [
        torch.int32,
        torch.int32,
        torch.bool,
        torch.int32,
        torch.bool,
    ]


def test_wide_sparse_tile_schedule_is_fused_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query_blocks = torch.tensor([0, 1], dtype=torch.int32)
    query_is_clean = torch.tensor([False, True])
    query_bounds = torch.tensor(((0, 0), (0, 2)), dtype=torch.int32)
    buffers = (
        query_bounds,
        query_blocks,
        query_is_clean,
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([True, True]),
    )
    full_mask = BlockDenoisingFullMask(
        query_blocks=query_blocks,
        query_is_clean=query_is_clean,
        block_size=1,
        clean_offset=2,
    )
    fused_mask = cp_attention.BlockDenoisingGlobalCleanMask(
        query_blocks=query_blocks,
        query_is_clean=query_is_clean,
        block_size=1,
        query_clean_bounds=query_bounds,
        clean_key_positions=torch.tensor([0, 1], dtype=torch.int32),
    )
    sparse_requests: list[bool] = []

    def build_plan(*args: object, **kwargs: object) -> object:
        del args
        sparse_requests.append(bool(kwargs.get("sparse_tile_worklist", False)))
        return object()

    monkeypatch.setattr(cp_attention, "_build_wide_interval_plan", build_plan)
    for mask in (full_mask, fused_mask):
        cp_attention._wide_metadata_plan(
            attn_mask=mask,
            buffers=buffers,
            key_start=0,
            query_len=2,
            key_len=2,
            query_heads=4,
            key_heads=2,
            native=True,
            device=torch.device("cpu"),
        )

    assert sparse_requests == [False, True]


def test_d256_native_packed_gqa_backward_is_fused_only() -> None:
    query_blocks = torch.zeros(1, dtype=torch.int32)
    query_is_clean = torch.zeros(1, dtype=torch.bool)
    packed = BlockDenoisingPackedKeyMask(
        query_blocks=query_blocks,
        local_query_blocks=query_blocks,
        query_is_clean=query_is_clean,
        active_key_blocks=query_blocks,
        clean_key_blocks=query_blocks,
        block_size=1,
    )
    full = BlockDenoisingFullMask(
        query_blocks=query_blocks,
        query_is_clean=query_is_clean,
        block_size=1,
        clean_offset=1,
    )

    assert cp_attention._uses_packed_gqa_backward(
        attn_mask=packed,
        query_heads=16,
        key_heads=8,
        head_dim=256,
    )
    assert not cp_attention._uses_packed_gqa_backward(
        attn_mask=full,
        query_heads=16,
        key_heads=8,
        head_dim=256,
    )


def test_packed_gqa_backward_skips_generic_token_mask_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query_blocks = torch.tensor([0, 1], dtype=torch.int32)
    packed = BlockDenoisingPackedKeyMask(
        query_blocks=query_blocks,
        local_query_blocks=query_blocks,
        query_is_clean=torch.zeros(2, dtype=torch.bool),
        active_key_blocks=query_blocks,
        clean_key_blocks=query_blocks,
        block_size=1,
    )
    query = torch.empty(1, 4, 2, 256)
    key = torch.empty(1, 2, 4, 256)
    value = torch.empty_like(key)
    output = torch.empty_like(query)
    lse = torch.empty(1, 4, 2, dtype=torch.float32)
    packed_block_mask = object()
    calls: list[dict[str, object]] = []

    def reject_generic_builder(**kwargs: object) -> None:
        del kwargs
        raise AssertionError("packed backward must not build a generic token mask")

    def fake_packed_builder(**kwargs: object) -> object:
        calls.append(kwargs)
        return packed_block_mask

    def fake_backward(**kwargs: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert kwargs["block_mask"] is packed_block_mask
        assert kwargs["packed_gqa"] is True
        assert kwargs["force_torch"] is False
        return query, key, value

    monkeypatch.setattr(cp_attention, "_make_flex_mods", reject_generic_builder)
    monkeypatch.setattr(
        cp_attention,
        "_make_packed_gqa_backward_block_mask",
        fake_packed_builder,
    )
    monkeypatch.setattr(
        cp_attention,
        "flex_attention_backward_from_state",
        fake_backward,
    )

    actual = cp_attention._flex_merged_shard_backward(
        query=query,
        key=key,
        value=value,
        final_output=output,
        final_lse=lse,
        grad_output=torch.empty_like(output),
        attn_mask=packed,
        is_causal=False,
        scale=0.125,
        key_start=0,
    )

    assert actual[0] is query
    assert actual[1] is key
    assert actual[2] is value
    assert len(calls) == 1
    assert "forward_block_mask" not in calls[0]


def test_collective_d256_backward_packs_existing_cp_mask_metadata() -> None:
    full_query_blocks = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int32)
    full_query_is_clean = torch.tensor([False, False, False, True, True, True])
    selected = torch.tensor([0, 2, 3, 4, 5])
    local = BlockDenoisingLocalActiveMask(
        query_blocks=full_query_blocks,
        query_is_clean=full_query_is_clean,
        active_blocks=torch.tensor([0, 0, 1, 1], dtype=torch.int32),
        block_size=2,
    )
    clean = BlockDenoisingGlobalCleanMask(
        query_blocks=full_query_blocks.index_select(0, selected),
        query_is_clean=full_query_is_clean.index_select(0, selected),
        block_size=2,
        clean_key_blocks=torch.tensor([0, 1, 2], dtype=torch.int32),
    )

    packed = cp_attention._collective_packed_key_mask(
        clean,
        local,
        local_query_blocks=full_query_blocks.index_select(0, selected),
    )

    assert torch.equal(packed.query_blocks, clean.query_blocks)
    assert torch.equal(
        packed.local_query_blocks,
        full_query_blocks.index_select(0, selected),
    )
    assert torch.equal(packed.active_key_blocks, local.active_blocks)
    assert torch.equal(packed.clean_key_blocks, clean.clean_key_blocks)
    assert packed.flex_cache is clean.flex_cache


def test_collective_packed_d256_backward_dispatch_is_shape_limited() -> None:
    query = torch.empty(1, 4, 12, 256)
    key = torch.empty(1, 4, 2, 256)
    assert cp_attention._uses_collective_packed_d256_backward(query, key)
    assert cp_attention._collective_packed_d256_kv_head_repeat(query, key) == 3
    assert not cp_attention._uses_collective_packed_d256_backward(
        torch.empty(1, 4, 12, 128),
        torch.empty(1, 4, 2, 128),
    )
    assert not cp_attention._uses_collective_packed_d256_backward(
        torch.empty(1, 4, 2, 256),
        torch.empty(1, 4, 2, 256),
    )


def test_native_fa4_d256_forward_expands_qwen_gqa_heads_for_tile_alignment() -> None:
    query = torch.empty(1, 12, 4, 256)
    key = torch.empty(1, 2, 4, 256)

    assert cp_attention._native_fa4_d256_forward_kv_head_repeat(query, key) == 3
    assert (
        cp_attention._native_fa4_d256_forward_kv_head_repeat(
            torch.empty(1, 12, 4, 128),
            torch.empty(1, 2, 4, 128),
        )
        == 1
    )


def test_repeated_kv_gradients_collapse_into_original_gqa_heads() -> None:
    expanded = torch.arange(2 * 1 * 2 * 6 * 4, dtype=torch.float32).view(
        2,
        1,
        2,
        6,
        4,
    )

    actual = cp_attention._collapse_repeated_kv_grad_storage_bshd(
        expanded,
        original_key_heads=2,
        kv_head_repeat=3,
    )

    expected = expanded.view(2, 1, 2, 2, 3, 4).sum(dim=4)
    torch.testing.assert_close(actual, expected)


def test_wide_cp_mask_uses_native_interval_dispatch_without_explicit_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mask = BlockDenoisingFullMask(
        query_blocks=torch.tensor([1, 1], dtype=torch.int32),
        query_is_clean=torch.tensor([False, True]),
        block_size=2,
        clean_offset=4,
    )
    query = torch.empty(1, 2, 2, 512)
    key = torch.empty(1, 2, 8, 512)
    value = torch.empty_like(key)
    expected_output = torch.empty_like(query)
    expected_lse = torch.empty(1, 2, 2)
    expected_plan = object()
    calls: list[tuple[torch.Tensor, ...]] = []

    monkeypatch.setattr(cp_attention, "_uses_native_wide_attention", lambda _: True)

    def interval_forward(*args):
        calls.append(args)
        return expected_output, expected_lse

    monkeypatch.setattr(cp_attention, "_wide_interval_forward_bhsd", interval_forward)
    monkeypatch.setattr(
        cp_attention,
        "_wide_metadata_plan",
        lambda **kwargs: expected_plan if kwargs["native"] else pytest.fail(
            "native wide attention must request a native interval plan"
        ),
    )

    output, lse = cp_attention._wide_shard_stats(
        query=query,
        key=key,
        value=value,
        attn_mask=mask,
        is_causal=False,
        scale=1.0,
        key_start=0,
    )

    assert output is expected_output
    assert lse is expected_lse
    clean_bounds = calls[0][3]
    torch.testing.assert_close(
        clean_bounds,
        torch.tensor([[0, 1], [0, 2]], dtype=torch.int32),
    )
    assert calls[0][-1] is expected_plan


def test_wide_cp_backward_uses_native_interval_dispatch_without_explicit_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mask = BlockDenoisingFullMask(
        query_blocks=torch.tensor([1, 1], dtype=torch.int32),
        query_is_clean=torch.tensor([False, True]),
        block_size=2,
        clean_offset=4,
    )
    query = torch.empty(1, 2, 2, 512)
    key = torch.empty(1, 2, 8, 512)
    value = torch.empty_like(key)
    output = torch.empty_like(query)
    lse = torch.empty(1, 2, 2)
    grad_output = torch.empty_like(output)
    expected = tuple(torch.empty_like(tensor) for tensor in (query, key, value))
    expected_plan = object()
    calls: list[tuple[torch.Tensor, ...]] = []

    monkeypatch.setattr(cp_attention, "_uses_native_wide_attention", lambda _: True)

    def interval_backward(*args):
        calls.append(args)
        return expected

    monkeypatch.setattr(
        cp_attention,
        "_wide_interval_backward_from_state_bhsd",
        interval_backward,
    )
    monkeypatch.setattr(
        cp_attention,
        "_wide_shard_stats",
        lambda **_: pytest.fail("native backward must not replay Flex attention"),
    )
    monkeypatch.setattr(
        cp_attention,
        "_wide_metadata_plan",
        lambda **kwargs: expected_plan if kwargs["native"] else pytest.fail(
            "native wide attention must request a native interval plan"
        ),
    )

    gradients = cp_attention._wide_merged_shard_backward(
        query=query,
        key=key,
        value=value,
        final_output=output,
        final_lse=lse,
        grad_output=grad_output,
        grad_final_lse=None,
        attn_mask=mask,
        is_causal=False,
        scale=1.0,
        key_start=0,
    )

    assert gradients == expected
    clean_bounds = calls[0][7]
    torch.testing.assert_close(
        clean_bounds,
        torch.tensor([[0, 1], [0, 2]], dtype=torch.int32),
    )
    assert calls[0][-1] is expected_plan


def test_exact_clean_context_intervals_match_full_and_packed_layouts() -> None:
    query_blocks = torch.tensor([2, 2], dtype=torch.int32)
    local_query_blocks = torch.tensor([2, 2], dtype=torch.int32)
    query_is_clean = torch.tensor([False, True])
    query_clean_bounds = torch.tensor([[5, 8], [5, 9]], dtype=torch.int32)

    full = BlockDenoisingFullMask(
        query_blocks=query_blocks,
        query_is_clean=query_is_clean,
        block_size=4,
        clean_offset=12,
        query_clean_bounds=query_clean_bounds,
    )
    full_allowed = cp_attention._block_denoising_mask_spec_allowed(
        full,
        torch.arange(2)[:, None],
        torch.arange(24)[None, :],
        0,
    )

    packed = cp_attention.BlockDenoisingPackedKeyMask(
        query_blocks=query_blocks,
        local_query_blocks=local_query_blocks,
        query_is_clean=query_is_clean,
        active_key_blocks=torch.tensor([2, 2, 2, 2], dtype=torch.int32),
        clean_key_blocks=torch.tensor([2, 1, 0, 1, 2], dtype=torch.int32),
        block_size=4,
        query_clean_bounds=query_clean_bounds,
        clean_key_positions=torch.tensor([8, 5, 1, 7, 9], dtype=torch.int32),
    )
    packed_allowed = cp_attention._block_denoising_mask_spec_allowed(
        packed,
        torch.arange(2)[:, None],
        torch.arange(9)[None, :],
        0,
    )

    expected_full = torch.zeros((2, 24), dtype=torch.bool)
    expected_full[0, 8:12] = True
    expected_full[0, 17:20] = True
    expected_full[1, 17:21] = True
    expected_packed = torch.tensor(
        [
            [True, True, True, True, False, True, False, True, False],
            [False, False, False, False, True, True, False, True, False],
        ],
        dtype=torch.bool,
    )
    torch.testing.assert_close(full_allowed, expected_full)
    torch.testing.assert_close(packed_allowed, expected_packed)


def test_windowed_clean_exchange_is_pairwise_consistent_and_sparse() -> None:
    plans = [
        windowed_clean_exchange_plan(
            seq_len=32,
            block_size=4,
            context_parallel_size=4,
            block_parallel_size=4,
            block_group_index=0,
            rank=rank,
            window=8,
        )
        for rank in range(4)
    ]

    for source, source_plan in enumerate(plans):
        send_indices, local_positions, remote_positions = (
            cp_attention._windowed_exchange_tensors(
                source_plan,
                rank=source,
                device=torch.device("cpu"),
            )
        )
        assert int(send_indices.numel()) == (
            source_plan.send_tokens - source_plan.send_counts[source]
        )
        assert int(local_positions.numel()) == source_plan.receive_counts[source]
        assert int(remote_positions.numel()) == (
            source_plan.receive_tokens - source_plan.receive_counts[source]
        )
        for destination, destination_plan in enumerate(plans):
            assert (
                source_plan.send_counts[destination]
                == (destination_plan.receive_counts[source])
            )
        assert source_plan.receive_tokens < 32
        assert source_plan.send_tokens < 32


def test_windowed_clean_exchange_supports_replicated_context_parallel_rings() -> None:
    plans_by_group = [
        [
            windowed_clean_exchange_plan(
                seq_len=32,
                block_size=4,
                context_parallel_size=2,
                block_parallel_size=4,
                block_group_index=group,
                rank=rank,
                window=8,
            )
            for rank in range(2)
        ]
        for group in range(2)
    ]

    for plans in plans_by_group:
        for source, source_plan in enumerate(plans):
            for destination, destination_plan in enumerate(plans):
                assert (
                    source_plan.send_counts[destination]
                    == (destination_plan.receive_counts[source])
                )
            assert source_plan.receive_tokens < 32
            assert source_plan.send_tokens < 32

    assert plans_by_group[0] != plans_by_group[1]


def test_exact_clean_context_intervals_compile_to_flex_block_mask() -> None:
    mask = BlockDenoisingFullMask(
        query_blocks=torch.tensor([2, 2], dtype=torch.int32),
        query_is_clean=torch.tensor([False, True]),
        block_size=4,
        clean_offset=12,
        query_clean_bounds=torch.tensor([[5, 8], [5, 9]], dtype=torch.int32),
    )
    query = torch.empty(1, 1, 2, 16)

    _, block_mask = cp_attention._make_flex_mods(
        attn_mask=mask,
        is_causal=False,
        query=query,
        key_start=0,
        key_len=24,
    )

    assert block_mask is not None


def _dense_attention_from_mask_bshd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attn_mask: (
        BlockDenoisingLocalActiveMask
        | BlockDenoisingGlobalCleanMask
        | BlockDenoisingFullMask
        | cp_attention.BlockDenoisingPackedKeyMask
    ),
    key_start: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_idx = torch.arange(query.shape[1], device=query.device)[:, None]
    kv_idx = torch.arange(key.shape[1], device=query.device)[None, :]
    allowed = cp_attention._block_denoising_mask_spec_allowed(
        attn_mask,
        q_idx,
        kv_idx,
        int(key_start),
    )
    repeats = query.shape[2] // key.shape[2]
    expanded_key = key.float().repeat_interleave(repeats, dim=2)
    expanded_value = value.float().repeat_interleave(repeats, dim=2)
    scores = torch.einsum("bqhd,bkhd->bhqk", query.float(), expanded_key) * float(scale)
    scores = scores.masked_fill(~allowed[None, None], -torch.inf)
    lse = torch.logsumexp(scores, dim=-1)
    probabilities = torch.softmax(scores, dim=-1)
    probabilities = torch.where(
        torch.isfinite(lse).unsqueeze(-1),
        probabilities,
        torch.zeros_like(probabilities),
    )
    output = torch.einsum("bhqk,bkhd->bqhd", probabilities, expanded_value)
    return output.to(dtype=query.dtype), lse


def _dense_exact_shard_stats_bhsd(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: (
        BlockDenoisingLocalActiveMask
        | BlockDenoisingGlobalCleanMask
        | BlockDenoisingFullMask
        | cp_attention.BlockDenoisingPackedKeyMask
    ),
    is_causal: bool,
    scale: float,
    key_start: int,
    output_float: bool,
    native_fa4: bool,
):
    del native_fa4
    if is_causal:
        raise AssertionError(
            "windowed fused collective test expects non-causal attention"
        )
    output, lse = _dense_attention_from_mask_bshd(
        query.transpose(1, 2).contiguous(),
        key.transpose(1, 2).contiguous(),
        value.transpose(1, 2).contiguous(),
        attn_mask=attn_mask,
        key_start=int(key_start),
        scale=float(scale),
    )
    if output_float:
        output = output.float()
    return output.transpose(1, 2).contiguous(), lse, None


def _dense_exact_packed_backward_bshd(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    final_output: torch.Tensor,
    final_lse: torch.Tensor,
    grad_output: torch.Tensor,
    attn_mask: BlockDenoisingGlobalCleanMask | cp_attention.BlockDenoisingPackedKeyMask,
    key_start: int,
    scale: float,
    query_indices: torch.Tensor | None = None,
    grad_key: torch.Tensor | None = None,
    grad_value: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del final_output, final_lse
    if query_indices is not None:
        raise AssertionError(
            "windowed fused collective test does not compact query rows"
        )
    with torch.enable_grad():
        q = query.detach().clone().requires_grad_(True)
        k = key.detach().clone().requires_grad_(True)
        v = value.detach().clone().requires_grad_(True)
        output, lse = _dense_attention_from_mask_bshd(
            q,
            k,
            v,
            attn_mask=attn_mask,
            key_start=int(key_start),
            scale=float(scale),
        )
        grad_query, actual_grad_key, actual_grad_value = torch.autograd.grad(
            (output.float(), lse),
            (q, k, v),
            (grad_output.float(), torch.zeros_like(lse)),
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )
    if grad_key is not None:
        grad_key.copy_(actual_grad_key)
        actual_grad_key = grad_key
    if grad_value is not None:
        grad_value.copy_(actual_grad_value)
        actual_grad_value = grad_value
    return grad_query, actual_grad_key, actual_grad_value


def test_packed_gqa_tile_block_mask_matches_exact_token_mask() -> None:
    mask = cp_attention.BlockDenoisingPackedKeyMask(
        query_blocks=torch.tensor([0, 2, 2, 1, 3, 3, 4], dtype=torch.int32),
        local_query_blocks=torch.tensor([0, 2, 2, 1, 3, 3, 4], dtype=torch.int32),
        query_is_clean=torch.tensor([False, False, False, True, True, False, True]),
        active_key_blocks=torch.tensor([0, 0, 2, 2, 3], dtype=torch.int32),
        clean_key_blocks=torch.tensor([0, 1, 1, 3, 4, 4], dtype=torch.int32),
        block_size=3,
    )
    group_size = 2
    query_tile = 4
    key_tile = 4
    block_mask = cp_attention._make_packed_gqa_tile_block_mask(
        attn_mask=mask,
        query_heads_per_key_head=group_size,
        query_tile=query_tile,
        key_tile=key_tile,
        mask_mod=lambda b, h, q, k: q == k,
    )

    virtual_query = torch.arange(mask.query_blocks.numel()).repeat_interleave(
        group_size
    )
    key_index = torch.arange(
        mask.active_key_blocks.numel() + mask.clean_key_blocks.numel()
    )
    exact = cp_attention._block_denoising_mask_spec_allowed(
        mask,
        virtual_query[:, None],
        key_index[None, :],
        0,
    )
    padded_query = math.ceil(exact.shape[0] / query_tile) * query_tile
    padded_key = math.ceil(exact.shape[1] / key_tile) * key_tile
    exact = F.pad(
        exact,
        (0, padded_key - exact.shape[1], 0, padded_query - exact.shape[0]),
    )
    tiles = exact.reshape(
        padded_query // query_tile,
        query_tile,
        padded_key // key_tile,
        key_tile,
    ).permute(0, 2, 1, 3)
    expected_full = tiles.all(dim=(-2, -1))
    expected_partial = tiles.any(dim=(-2, -1)) & ~expected_full

    def dense(counts: torch.Tensor, indices: torch.Tensor, width: int) -> torch.Tensor:
        result = torch.zeros((*counts.shape, width), dtype=torch.bool)
        columns = torch.arange(indices.shape[-1])
        valid = columns < counts[..., None]
        result.scatter_(-1, indices.to(torch.int64), valid)
        return result

    actual_partial = dense(
        block_mask.kv_num_blocks[0, 0],
        block_mask.kv_indices[0, 0],
        expected_partial.shape[-1],
    )
    actual_full = dense(
        block_mask.full_kv_num_blocks[0, 0],
        block_mask.full_kv_indices[0, 0],
        expected_full.shape[-1],
    )
    actual_occupied = actual_partial | actual_full
    expected_occupied = expected_partial | expected_full
    assert torch.all(actual_occupied | ~expected_occupied)
    assert not torch.any(actual_partial & actual_full)
    assert not torch.any(actual_full & ~expected_occupied)


def test_packed_gqa_tile_block_mask_exact_bounds_matches_token_mask() -> None:
    mask = cp_attention.BlockDenoisingPackedKeyMask(
        query_blocks=torch.tensor([0, 0, 2, 2, 3, 3, 4], dtype=torch.int32),
        local_query_blocks=torch.tensor([0, 0, 2, 2, 3, 3, 4], dtype=torch.int32),
        query_is_clean=torch.tensor([False, False, False, True, True, False, True]),
        active_key_blocks=torch.tensor([0, 0, 2, 2, 3], dtype=torch.int32),
        clean_key_blocks=torch.tensor([0, 1, 1, 3, 4, 4], dtype=torch.int32),
        query_clean_bounds=torch.tensor(
            [[0, 1], [0, 2], [1, 4], [0, 5], [3, 8], [4, 9], [7, 10]],
            dtype=torch.int32,
        ),
        clean_key_positions=torch.tensor([0, 2, 3, 5, 8, 9], dtype=torch.int32),
        block_size=3,
    )
    group_size = 2
    query_tile = 4
    key_tile = 4
    block_mask = cp_attention._make_packed_gqa_tile_block_mask(
        attn_mask=mask,
        query_heads_per_key_head=group_size,
        query_tile=query_tile,
        key_tile=key_tile,
        mask_mod=lambda b, h, q, k: q == k,
    )

    virtual_query = torch.arange(mask.query_blocks.numel()).repeat_interleave(
        group_size
    )
    key_index = torch.arange(
        mask.active_key_blocks.numel() + mask.clean_key_blocks.numel()
    )
    exact = cp_attention._block_denoising_mask_spec_allowed(
        mask,
        virtual_query[:, None],
        key_index[None, :],
        0,
    )
    padded_query = math.ceil(exact.shape[0] / query_tile) * query_tile
    padded_key = math.ceil(exact.shape[1] / key_tile) * key_tile
    exact = F.pad(
        exact,
        (0, padded_key - exact.shape[1], 0, padded_query - exact.shape[0]),
    )
    tiles = exact.reshape(
        padded_query // query_tile,
        query_tile,
        padded_key // key_tile,
        key_tile,
    ).permute(0, 2, 1, 3)
    expected_full = tiles.all(dim=(-2, -1))
    expected_occupied = tiles.any(dim=(-2, -1))

    def dense(counts: torch.Tensor, indices: torch.Tensor, width: int) -> torch.Tensor:
        result = torch.zeros((*counts.shape, width), dtype=torch.bool)
        columns = torch.arange(indices.shape[-1])
        valid = columns < counts[..., None]
        result.scatter_(-1, indices.to(torch.int64), valid)
        return result

    actual_partial = dense(
        block_mask.kv_num_blocks[0, 0],
        block_mask.kv_indices[0, 0],
        expected_occupied.shape[-1],
    )
    actual_full = dense(
        block_mask.full_kv_num_blocks[0, 0],
        block_mask.full_kv_indices[0, 0],
        expected_full.shape[-1],
    )
    actual_occupied = actual_partial | actual_full
    assert torch.all(actual_occupied | ~expected_occupied)
    assert not torch.any(actual_partial & actual_full)
    assert not torch.any(actual_full & ~expected_full)


@pytest.mark.parametrize("mask_kind", ("local", "global", "full"))
def test_nonpacked_gqa_tile_block_mask_matches_exact_token_mask(
    mask_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block_size = 2
    query_blocks = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1], dtype=torch.int32)
    query_is_clean = torch.tensor([False, False, False, False, True, True, True, True])
    if mask_kind == "local":
        mask = BlockDenoisingLocalActiveMask(
            query_blocks=query_blocks,
            query_is_clean=query_is_clean,
            active_blocks=torch.tensor([0, 0, 1, 1], dtype=torch.int32),
            block_size=block_size,
        )
        key_start = 0
        key_len = 4
    elif mask_kind == "global":
        mask = BlockDenoisingGlobalCleanMask(
            query_blocks=query_blocks,
            query_is_clean=query_is_clean,
            block_size=block_size,
        )
        key_start = 0
        key_len = 4
    else:
        mask = BlockDenoisingFullMask(
            query_blocks=query_blocks,
            query_is_clean=query_is_clean,
            block_size=block_size,
            clean_offset=4,
        )
        key_start = 0
        key_len = 8

    monkeypatch.setattr(
        cp_attention,
        "fa4_backward_sparse_tile",
        lambda **kwargs: (4, 4),
    )

    def reject_compiled_builder() -> None:
        raise AssertionError("non-packed GQA metadata must not compile a token mask")

    monkeypatch.setattr(
        cp_attention,
        "_get_compiled_create_block_mask",
        reject_compiled_builder,
    )
    group_size = 2
    block_mask = cp_attention._make_local_gqa_backward_block_mask(
        attn_mask=mask,
        query=torch.empty(1, 4, query_blocks.numel(), 64),
        key=torch.empty(1, 2, key_len, 64),
        key_start=key_start,
        forward_block_mask=SimpleNamespace(BLOCK_SIZE=(4, 4)),
    )

    virtual_query = torch.arange(query_blocks.numel()).repeat_interleave(group_size)
    exact = cp_attention._block_denoising_mask_spec_allowed(
        mask,
        virtual_query[:, None],
        torch.arange(key_len)[None, :],
        key_start,
    )
    padded_query = math.ceil(exact.shape[0] / 4) * 4
    padded_key = math.ceil(exact.shape[1] / 4) * 4
    exact = F.pad(
        exact,
        (0, padded_key - exact.shape[1], 0, padded_query - exact.shape[0]),
    )
    tiles = exact.reshape(
        padded_query // 4,
        4,
        padded_key // 4,
        4,
    ).permute(0, 2, 1, 3)
    expected_full = tiles.all(dim=(-2, -1))
    expected_occupied = tiles.any(dim=(-2, -1))

    def dense(counts: torch.Tensor, indices: torch.Tensor, width: int) -> torch.Tensor:
        result = torch.zeros((*counts.shape, width), dtype=torch.bool)
        columns = torch.arange(indices.shape[-1])
        valid = columns < counts[..., None]
        result.scatter_(-1, indices.to(torch.int64), valid)
        return result

    actual_partial = dense(
        block_mask.kv_num_blocks[0, 0],
        block_mask.kv_indices[0, 0],
        expected_occupied.shape[-1],
    )
    actual_full = dense(
        block_mask.full_kv_num_blocks[0, 0],
        block_mask.full_kv_indices[0, 0],
        expected_occupied.shape[-1],
    )
    assert torch.equal(actual_partial | actual_full, expected_occupied)
    assert torch.equal(actual_full, expected_full)
    assert block_mask.kv_indices.shape[-1] == expected_occupied.shape[-1]
    assert block_mask.q_indices.shape[-1] == expected_occupied.shape[-2]


def test_packed_gqa_backward_is_limited_to_local_or_packed_masks() -> None:
    query_blocks = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
    query_is_clean = torch.tensor([False, False, True, True])
    local = BlockDenoisingLocalActiveMask(
        query_blocks=query_blocks,
        query_is_clean=query_is_clean,
        active_blocks=query_blocks,
        block_size=2,
    )
    global_clean = BlockDenoisingGlobalCleanMask(
        query_blocks=query_blocks,
        query_is_clean=query_is_clean,
        block_size=2,
    )
    full = BlockDenoisingFullMask(
        query_blocks=query_blocks,
        query_is_clean=query_is_clean,
        block_size=2,
        clean_offset=2,
    )
    packed = cp_attention.BlockDenoisingPackedKeyMask(
        query_blocks=query_blocks,
        local_query_blocks=query_blocks,
        query_is_clean=query_is_clean,
        active_key_blocks=query_blocks[:2],
        clean_key_blocks=query_blocks[2:],
        block_size=2,
    )

    for mask in (local, packed):
        assert cp_attention._uses_packed_gqa_backward(
            attn_mask=mask,
            query_heads=32,
            key_heads=8,
            head_dim=128,
        )
    for mask in (global_clean, full):
        assert not cp_attention._uses_packed_gqa_backward(
            attn_mask=mask,
            query_heads=32,
            key_heads=8,
            head_dim=128,
        )
    assert not cp_attention._uses_packed_gqa_backward(
        attn_mask=local,
        query_heads=32,
        key_heads=8,
        head_dim=256,
    )


def test_fa4_native_pack_gqa_backward_matches_generic_gqa() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    try:
        from flash_attn.cute.interface import flash_attn_func
    except ImportError:
        pytest.skip("requires the packaged FA4 CuTe backend")

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(123)
    dtype = torch.bfloat16
    q_shape = (1, 64, 8, 64)
    kv_shape = (1, 64, 2, 64)
    q = torch.randn(q_shape, generator=generator, device=device, dtype=dtype)
    k = torch.randn(kv_shape, generator=generator, device=device, dtype=dtype)
    v = torch.randn(kv_shape, generator=generator, device=device, dtype=dtype)
    grad = torch.randn(q_shape, generator=generator, device=device, dtype=dtype)

    def _run(pack_gqa: bool) -> tuple[torch.Tensor, ...]:
        q_run = q.clone().requires_grad_(True)
        k_run = k.clone().requires_grad_(True)
        v_run = v.clone().requires_grad_(True)
        out, _ = flash_attn_func(
            q_run,
            k_run,
            v_run,
            pack_gqa=pack_gqa,
        )
        (out * grad).sum().backward()
        return out, q_run.grad, k_run.grad, v_run.grad

    generic = _run(False)
    packed = _run(True)
    errors: list[str] = []
    for name, actual, expected in zip(
        ("output", "dQ", "dK", "dV"),
        packed,
        generic,
        strict=True,
    ):
        try:
            torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)
        except AssertionError as exc:
            errors.append(f"{name}:\n{exc}")
    assert not errors, "\n\n".join(errors)


def test_fa4_runtime_rejects_missing_native_pack_gqa_backward() -> None:
    with pytest.raises(RuntimeError, match="native Pack-GQA backward"):
        _require_native_pack_gqa_backward(SimpleNamespace())

    _require_native_pack_gqa_backward(
        SimpleNamespace(_validate_pack_gqa_backward_capability=lambda **_: None)
    )


def test_full_mask_coordinate_decode_is_centralized() -> None:
    assert _decode_bdlm_key_start(17) == (17, False)
    assert _decode_bdlm_key_start(-18) == (17, True)


def test_streaming_attention_matches_sdpa_with_boolean_mask() -> None:
    torch.manual_seed(0)
    q = torch.randn(2, 3, 5, 4)
    k = torch.randn(2, 3, 7, 4)
    v = torch.randn(2, 3, 7, 4)
    mask = torch.ones(1, 1, 5, 7, dtype=torch.bool)
    mask[..., 2, 1:4] = False
    mask[..., 4, 6] = False

    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
    actual = streaming_attention(q, k, v, attn_mask=mask, key_chunk_size=2)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_streaming_attention_matches_sdpa_with_causal_mask() -> None:
    torch.manual_seed(1)
    q = torch.randn(1, 2, 6, 8)
    k = torch.randn(1, 2, 6, 8)
    v = torch.randn(1, 2, 6, 8)

    expected = F.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=0.0,
        is_causal=True,
    )
    actual = streaming_attention(q, k, v, is_causal=True, key_chunk_size=3)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_streaming_attention_matches_sdpa_with_additive_mask() -> None:
    torch.manual_seed(2)
    q = torch.randn(2, 1, 4, 8)
    k = torch.randn(2, 1, 6, 8)
    v = torch.randn(2, 1, 6, 8)
    mask = torch.zeros(1, 1, 4, 6)
    mask[..., 0, 2] = -10000.0
    mask[..., 3, :2] = -10000.0

    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
    actual = streaming_attention(q, k, v, attn_mask=mask, key_chunk_size=1)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_streaming_attention_backward_matches_sdpa() -> None:
    torch.manual_seed(3)
    q = torch.randn(1, 2, 5, 4, requires_grad=True)
    k = torch.randn(1, 2, 5, 4, requires_grad=True)
    v = torch.randn(1, 2, 5, 4, requires_grad=True)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    actual = streaming_attention(q, k, v, is_causal=True, key_chunk_size=2).sum()
    expected = F.scaled_dot_product_attention(
        q_ref,
        k_ref,
        v_ref,
        dropout_p=0.0,
        is_causal=True,
    ).sum()
    actual.backward()
    expected.backward()

    torch.testing.assert_close(q.grad, q_ref.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(k.grad, k_ref.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(v.grad, v_ref.grad, atol=1e-5, rtol=1e-5)


def test_context_parallel_attention_rejects_replicated_kv_name() -> None:
    q = torch.randn(1, 1, 2, 16)
    k = torch.randn(1, 1, 2, 16)
    v = torch.randn(1, 1, 2, 16)

    with pytest.raises(ValueError, match="replicated_kv_attention"):
        context_parallel_attention(q, k, v, kv_backend="replicated")


def test_replicated_kv_attention_matches_sdpa() -> None:
    torch.manual_seed(4)
    q = torch.randn(1, 2, 4, 16)
    k = torch.randn(1, 2, 5, 16)
    v = torch.randn(1, 2, 5, 16)
    mask = torch.ones(1, 1, 4, 5, dtype=torch.bool)
    mask[..., 1, 3] = False

    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
    actual = replicated_kv_attention(q, k, v, attn_mask=mask)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_replicated_block_attention_routes_directly_to_native_bshd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = torch.randn(1, 4, 4, 8)
    local_key = torch.randn(1, 2, 2, 8)
    local_value = torch.randn_like(local_key)
    clean_key = torch.randn(1, 2, 2, 8)
    clean_value = torch.randn_like(clean_key)
    local_mask = BlockDenoisingLocalActiveMask(
        query_blocks=torch.tensor([0, 0, 0, 1], dtype=torch.int32),
        query_is_clean=torch.tensor([False, False, True, True]),
        active_blocks=torch.tensor([0, 0], dtype=torch.int32),
        block_size=2,
    )
    global_mask = BlockDenoisingGlobalCleanMask(
        query_blocks=torch.tensor([0, 0, 0, 1], dtype=torch.int32),
        query_is_clean=torch.tensor([False, False, True, True]),
        block_size=2,
    )
    expected = torch.randn_like(query)
    captured: tuple[object, ...] | None = None

    def native_apply(*args: object) -> torch.Tensor:
        nonlocal captured
        captured = args
        return expected

    monkeypatch.setattr(
        cp_attention._ReplicatedWithLocalKVAttentionBSHD,
        "apply",
        native_apply,
    )

    actual = replicated_block_denoising_attention_bshd(
        query,
        local_key,
        local_value,
        clean_key,
        clean_value,
        local_attn_mask=local_mask,
        global_attn_mask=global_mask,
    )

    assert actual is expected
    assert captured is not None
    assert captured[0] is query
    assert captured[1] is local_key
    assert captured[2] is local_value
    assert captured[3] is clean_key
    assert captured[4] is clean_value
    assert captured[5] is local_mask
    assert captured[6] is global_mask


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="replicated BDLM FlexAttention requires CUDA",
)
def test_replicated_block_attention_matches_dense_forward_and_backward() -> None:
    torch.manual_seed(43)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch = 2
    query_heads = 4
    kv_heads = 2
    head_dim = 64
    block_size = 16
    num_blocks = 3
    active_len = num_blocks * block_size
    clean_len = (num_blocks - 1) * block_size
    query_len = active_len + clean_len
    active_blocks = torch.repeat_interleave(
        torch.arange(num_blocks, device=device, dtype=torch.int32),
        block_size,
    )
    clean_blocks = torch.repeat_interleave(
        torch.arange(num_blocks - 1, device=device, dtype=torch.int32),
        block_size,
    )
    query_blocks = torch.cat((active_blocks, clean_blocks))
    query_is_clean = torch.arange(query_len, device=device) >= active_len
    local_mask = BlockDenoisingLocalActiveMask(
        query_blocks=query_blocks,
        query_is_clean=query_is_clean,
        active_blocks=active_blocks,
        block_size=block_size,
    )
    global_mask = BlockDenoisingGlobalCleanMask(
        query_blocks=query_blocks,
        query_is_clean=query_is_clean,
        block_size=block_size,
    )
    query = torch.randn(
        batch,
        query_len,
        query_heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    local_key = torch.randn(
        batch,
        active_len,
        kv_heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    local_value = torch.randn_like(local_key, requires_grad=True)
    clean_key = torch.randn(
        batch,
        clean_len,
        kv_heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    clean_value = torch.randn_like(clean_key, requires_grad=True)
    grad = torch.randn_like(query)
    query_ref = query.detach().transpose(1, 2).contiguous().requires_grad_(True)
    local_key_ref = local_key.detach().transpose(1, 2).contiguous().requires_grad_(True)
    local_value_ref = (
        local_value.detach().transpose(1, 2).contiguous().requires_grad_(True)
    )
    clean_key_ref = clean_key.detach().transpose(1, 2).contiguous().requires_grad_(True)
    clean_value_ref = (
        clean_value.detach().transpose(1, 2).contiguous().requires_grad_(True)
    )
    dense_mask = torch.cat(
        (
            _hybrid_local_dense_mask(local_mask),
            _global_clean_dense_mask(global_mask, key_len=clean_len),
        ),
        dim=-1,
    ).view(1, 1, query_len, active_len + clean_len)
    reference = F.scaled_dot_product_attention(
        query_ref,
        torch.cat((local_key_ref, clean_key_ref), dim=-2),
        torch.cat((local_value_ref, clean_value_ref), dim=-2),
        attn_mask=dense_mask,
        dropout_p=0.0,
        enable_gqa=True,
    )
    actual = replicated_block_denoising_attention_bshd(
        query,
        local_key,
        local_value,
        clean_key,
        clean_value,
        local_attn_mask=local_mask,
        global_attn_mask=global_mask,
    )
    torch.testing.assert_close(
        actual.transpose(1, 2),
        reference,
        atol=6e-2,
        rtol=6e-2,
    )
    (actual * grad).sum().backward()
    (reference * grad.transpose(1, 2)).sum().backward()
    for actual_grad, reference_grad in (
        (query.grad.transpose(1, 2), query_ref.grad),
        (local_key.grad.transpose(1, 2), local_key_ref.grad),
        (local_value.grad.transpose(1, 2), local_value_ref.grad),
        (clean_key.grad.transpose(1, 2), clean_key_ref.grad),
        (clean_value.grad.transpose(1, 2), clean_value_ref.grad),
    ):
        torch.testing.assert_close(
            actual_grad,
            reference_grad,
            atol=8e-2,
            rtol=8e-2,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="BDLM FlexAttention ragged prefix path requires CUDA",
)
def test_ragged_prefix_bdlm_attention_matches_dense_shard_backward() -> None:
    torch.manual_seed(41)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    block_size = 16
    query_blocks = torch.cat(
        (
            torch.ones(block_size, device=device, dtype=torch.int32),
            torch.full((block_size,), 2, device=device, dtype=torch.int32),
            torch.ones(block_size, device=device, dtype=torch.int32),
            torch.full((block_size,), 3, device=device, dtype=torch.int32),
        )
    )
    query_is_clean = torch.cat(
        (
            torch.zeros(2 * block_size, device=device, dtype=torch.bool),
            torch.ones(2 * block_size, device=device, dtype=torch.bool),
        )
    )
    mask = BlockDenoisingGlobalCleanMask(
        query_blocks=query_blocks,
        query_is_clean=query_is_clean,
        block_size=block_size,
    )
    query = torch.randn(
        2, 4 * block_size, 2, 64, device=device, dtype=dtype, requires_grad=True
    )
    key = torch.randn(
        2, 4 * block_size, 2, 64, device=device, dtype=dtype, requires_grad=True
    )
    value = torch.randn_like(key, requires_grad=True)
    query_ref = query.detach().clone().requires_grad_(True)
    key_ref = key.detach().clone().requires_grad_(True)
    value_ref = value.detach().clone().requires_grad_(True)
    grad = torch.randn_like(query)

    ragged, ragged_lse = _ragged_prefix_bdlm_attention_bshd(
        query=query,
        key=key,
        value=value,
        attn_mask=mask,
        scale=1.0 / math.sqrt(64),
    )
    dense, dense_lse = _bdlm_flex_shard_attention_bshd(
        query_ref,
        key_ref,
        value_ref,
        mask.query_blocks,
        mask.query_is_clean,
        block_size,
        0,
        1.0 / math.sqrt(64),
    )
    (ragged * grad).sum().backward()
    (dense * grad).sum().backward()

    torch.testing.assert_close(ragged, dense, atol=6e-2, rtol=6e-2)
    torch.testing.assert_close(ragged_lse, dense_lse, atol=6e-2, rtol=6e-2)
    torch.testing.assert_close(query.grad, query_ref.grad, atol=8e-2, rtol=8e-2)
    torch.testing.assert_close(key.grad, key_ref.grad, atol=8e-2, rtol=8e-2)
    torch.testing.assert_close(value.grad, value_ref.grad, atol=8e-2, rtol=8e-2)


def test_shard_bounds_balances_contiguous_ranges() -> None:
    ranges = [shard_bounds(10, 4, rank) for rank in range(4)]

    assert ranges == [(0, 3), (3, 6), (6, 8), (8, 10)]


def test_clean_ownership_covers_each_token_once() -> None:
    shards = all_clean_shards(
        seq_len=17,
        context_parallel_size=4,
    )
    covered = [token for shard in shards for token in range(shard.start, shard.stop)]

    assert sorted(covered) == list(range(17))
    assert len(covered) == len(set(covered))
    assert {shard.owner_rank for shard in shards} == {0, 1, 2, 3}


def test_clean_ownership_uses_dual_chunk_head_tail_layout() -> None:
    by_rank = [
        [
            (shard.start, shard.stop, shard.chunk_index)
            for shard in clean_shards_for_rank(
                seq_len=16,
                context_parallel_size=4,
                rank=rank,
            )
        ]
        for rank in range(4)
    ]

    assert by_rank == [
        [(0, 2, 0), (14, 16, 7)],
        [(2, 4, 1), (12, 14, 6)],
        [(4, 6, 2), (10, 12, 5)],
        [(6, 8, 3), (8, 10, 4)],
    ]


def test_fused_bp_cp_runtime_maps_zigzag_to_dual_chunk_clean_ownership() -> None:
    runtime = SimpleNamespace(
        block_parallel_size=4,
        cp_bp_policy=SimpleNamespace(clean_kv_layout="zigzag"),
    )

    assert runtime_clean_layout(runtime) == "dual_chunk"


def test_monolithic_cp_zigzag_owns_every_noisy_and_clean_token_once() -> None:
    seq_len = 32
    cp_size = 4
    intervals = all_context_parallel_sequence_intervals(
        seq_len=seq_len,
        context_parallel_size=cp_size,
    )
    covered = [
        position
        for rank_intervals in intervals
        for start, stop in rank_intervals
        for position in range(start, stop)
    ]

    assert sorted(covered) == list(range(2 * seq_len))
    assert len(covered) == len(set(covered))
    for rank, rank_intervals in enumerate(intervals):
        assert rank_intervals == (
            (rank * 8, (rank + 1) * 8),
            (2 * seq_len - (rank + 1) * 8, 2 * seq_len - rank * 8),
        )


def test_monolithic_cp_layout_keeps_local_loss_rows_disjoint() -> None:
    layouts = [
        context_parallel_sequence_layout(
            seq_len=32,
            block_size=4,
            context_parallel_size=4,
            rank=rank,
            device=torch.device("cpu"),
        )
        for rank in range(4)
    ]
    noisy_positions = torch.cat([layout.noisy_positions for layout in layouts])

    assert torch.equal(torch.sort(noisy_positions).values, torch.arange(32))
    for layout in layouts:
        assert layout.logical_positions.numel() == 16
        assert layout.noisy_positions.numel() == 8
        assert torch.equal(layout.noisy_local_indices, torch.arange(8))
        assert torch.all(layout.logical_positions[:8] < 32)
        assert torch.all(layout.logical_positions[8:] >= 32)


def test_monolithic_cp_gathers_local_rows_without_global_materialization() -> None:
    noisy = torch.arange(32).view(1, 32)
    clean = torch.arange(100, 132).view(1, 32)
    layout = context_parallel_sequence_layout(
        seq_len=32,
        block_size=4,
        context_parallel_size=4,
        rank=2,
        device=torch.device("cpu"),
    )

    local = gather_context_parallel_sequence(noisy, clean, layout)

    assert torch.equal(
        local,
        torch.tensor(
            [[16, 17, 18, 19, 20, 21, 22, 23, 108, 109, 110, 111, 112, 113, 114, 115]]
        ),
    )


def test_monolithic_cp_supports_diffusion_blocks_split_across_rank_boundaries() -> None:
    layout = context_parallel_sequence_layout(
        seq_len=24,
        block_size=4,
        context_parallel_size=4,
        rank=0,
        device=torch.device("cpu"),
    )

    assert layout.intervals == ((0, 6), (42, 48))


def test_monolithic_cp_requires_equal_dual_chunks() -> None:
    with pytest.raises(ValueError, match="equal DualChunkSwap chunks"):
        context_parallel_sequence_layout(
            seq_len=30,
            block_size=2,
            context_parallel_size=4,
            rank=0,
            device=torch.device("cpu"),
        )


def test_monolithic_cp_full_mask_matches_block_diffusion_edges() -> None:
    seq_len = 8
    block_size = 2
    layout = context_parallel_sequence_layout(
        seq_len=seq_len,
        block_size=block_size,
        context_parallel_size=2,
        rank=1,
        device=torch.device("cpu"),
    )
    mask = BlockDenoisingFullMask(
        query_blocks=(layout.model_positions // block_size).to(torch.int32),
        query_is_clean=layout.logical_positions >= seq_len,
        block_size=block_size,
        clean_offset=seq_len,
    )
    q_idx = torch.arange(layout.logical_positions.numel())[:, None]
    kv_idx = torch.arange(2 * seq_len)[None, :]
    actual = _block_denoising_mask_spec_allowed(mask, q_idx, kv_idx, 0)

    q_logical = layout.logical_positions[:, None]
    q_clean = q_logical >= seq_len
    q_pos = torch.where(q_clean, q_logical - seq_len, q_logical)
    q_block = q_pos // block_size
    kv_clean = kv_idx >= seq_len
    kv_pos = torch.where(kv_clean, kv_idx - seq_len, kv_idx)
    kv_block = kv_pos // block_size
    expected = torch.where(
        q_clean,
        kv_clean & (q_block >= kv_block),
        ((~kv_clean) & (q_block == kv_block)) | (kv_clean & (q_block > kv_block)),
    )
    torch.testing.assert_close(actual, expected)


def test_monolithic_cp_full_mask_uses_global_key_owner_coordinates() -> None:
    seq_len = 16
    block_size = 4
    layout = context_parallel_sequence_layout(
        seq_len=seq_len,
        block_size=block_size,
        context_parallel_size=2,
        rank=0,
        device=torch.device("cpu"),
    )
    mask = BlockDenoisingFullMask(
        query_blocks=(layout.model_positions // block_size).to(torch.int32),
        query_is_clean=layout.logical_positions >= seq_len,
        block_size=block_size,
        clean_offset=seq_len,
    )
    q_idx = torch.arange(layout.logical_positions.numel())[:, None]

    for key_start in (0, 8, 16, 24):
        local_kv_idx = torch.arange(8)[None, :]
        actual = _block_denoising_mask_spec_allowed(
            mask,
            q_idx,
            local_kv_idx,
            key_start,
        )
        global_kv = key_start + local_kv_idx
        kv_clean = global_kv >= seq_len
        kv_pos = torch.where(kv_clean, global_kv - seq_len, global_kv)
        kv_block = kv_pos // block_size
        q_clean = mask.query_is_clean[:, None]
        q_block = mask.query_blocks[:, None]
        expected = torch.where(
            q_clean,
            kv_clean & (q_block >= kv_block),
            ((~kv_clean) & (q_block == kv_block)) | (kv_clean & (q_block > kv_block)),
        )
        torch.testing.assert_close(actual, expected)


def test_full_mask_rejects_invalid_global_boundary_and_metadata() -> None:
    blocks = torch.arange(4, dtype=torch.int32)
    clean = torch.zeros(4, dtype=torch.bool)

    with pytest.raises(ValueError, match="positive clean_offset"):
        BlockDenoisingFullMask(
            query_blocks=blocks,
            query_is_clean=clean,
            block_size=2,
            clean_offset=0,
        )
    with pytest.raises(ValueError, match="identical shapes"):
        BlockDenoisingFullMask(
            query_blocks=blocks,
            query_is_clean=clean[:2],
            block_size=2,
            clean_offset=4,
        )


def test_pure_cp_api_routes_to_standard_collective_without_bp_decomposition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = torch.randn(1, 4, 2, 8)
    key = torch.randn(1, 4, 1, 8)
    value = torch.randn_like(key)
    mask = BlockDenoisingFullMask(
        query_blocks=torch.tensor([0, 1, 1, 0], dtype=torch.int32),
        query_is_clean=torch.tensor([False, False, True, True]),
        block_size=2,
        clean_offset=4,
    )
    runtime = SimpleNamespace(enabled=True, local_parallel_size=2)
    expected = torch.randn_like(query)
    captured: tuple[object, ...] | None = None

    def monolithic_apply(*args: object) -> torch.Tensor:
        nonlocal captured
        captured = args
        return expected

    def bp_apply(*args: object) -> torch.Tensor:
        del args
        raise AssertionError("pure CP must not use the BP local/clean decomposition")

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        cp_attention._CollectiveContextParallelAttentionBSHD,
        "apply",
        monolithic_apply,
    )
    monkeypatch.setattr(
        cp_attention._CollectiveWithLocalKVAttentionBSHD,
        "apply",
        bp_apply,
    )

    actual = full_mask_block_denoising_attention_bshd(
        query,
        key,
        value,
        global_seq_len=8,
        attn_mask=mask,
        runtime=runtime,
    )

    assert actual is expected
    assert captured is not None
    assert captured[0] is query
    assert captured[1] is key
    assert captured[2] is value
    assert captured[3] is mask
    assert captured[5:] == (runtime, 8)


def test_pure_cp_packed_api_routes_to_within_block_query_sharding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = torch.randn(1, 6, 2, 8)
    local_key = torch.randn(1, 4, 1, 8)
    local_value = torch.randn_like(local_key)
    clean_key = torch.randn(1, 2, 1, 8)
    clean_value = torch.randn_like(clean_key)
    local_mask = BlockDenoisingLocalActiveMask(
        query_blocks=torch.tensor([0, 0, 1, 1, 0, 0], dtype=torch.int32),
        query_is_clean=torch.tensor([False] * 4 + [True] * 2),
        active_blocks=torch.tensor([0, 0, 1, 1], dtype=torch.int32),
        block_size=2,
    )
    clean_mask = BlockDenoisingGlobalCleanMask(
        query_blocks=local_mask.query_blocks,
        query_is_clean=local_mask.query_is_clean,
        block_size=2,
    )
    runtime = SimpleNamespace(
        enabled=True,
        local_parallel_size=2,
        context_parallel_rank=0,
        context_block_parallel_group_ranks=[0, 1],
        context_block_parallel_group=object(),
        block_parallel_size=1,
    )
    expected = torch.randn_like(query)
    captured: tuple[object, ...] | None = None

    def sharded_apply(*args: object) -> torch.Tensor:
        nonlocal captured
        captured = args
        return expected

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        cp_attention._PureContextShardedActiveAttentionBSHD,
        "apply",
        sharded_apply,
    )

    actual = cp_attention.pure_context_block_denoising_attention_bshd(
        query=query,
        local_key=local_key,
        local_value=local_value,
        global_key_shard=clean_key,
        global_value_shard=clean_value,
        global_seq_len=4,
        local_attn_mask=local_mask,
        global_attn_mask=clean_mask,
        runtime=runtime,
    )

    assert actual is expected
    assert captured is not None
    assert captured[:5] == (query, local_key, local_value, clean_key, clean_value)
    assert captured[5] is local_mask
    assert captured[6] is clean_mask
    assert captured[8:] == (runtime, 4)


def test_persistent_pure_cp_uses_token_count_for_active_kv_gather(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = torch.randn(1, 4, 2, 8)
    active_key = torch.randn(1, 2, 1, 8)
    active_value = torch.randn_like(active_key)
    clean_key = torch.randn(1, 2, 1, 8)
    clean_value = torch.randn_like(clean_key)
    local_mask = BlockDenoisingLocalActiveMask(
        query_blocks=torch.tensor([0, 0, 1, 1, 0, 0], dtype=torch.int32),
        query_is_clean=torch.tensor([False] * 4 + [True] * 2),
        active_blocks=torch.tensor([0, 0, 1, 1], dtype=torch.int32),
        block_size=2,
    )
    clean_mask = BlockDenoisingGlobalCleanMask(
        query_blocks=local_mask.query_blocks,
        query_is_clean=local_mask.query_is_clean,
        block_size=2,
    )
    runtime = SimpleNamespace(
        enabled=True,
        local_parallel_size=2,
        context_parallel_rank=0,
        context_block_parallel_group_ranks=[0, 1],
        context_block_parallel_group=object(),
        block_parallel_size=1,
    )
    gathered_args: tuple[object, ...] | None = None
    expected_key = torch.randn(1, 4, 1, 8)
    expected_value = torch.randn_like(expected_key)
    expected = torch.randn_like(query)

    def gather_apply(*args: object):
        nonlocal gathered_args
        gathered_args = args
        return expected_key, expected_value

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(cp_attention._GatherPureCPActiveKV, "apply", gather_apply)
    monkeypatch.setattr(
        cp_attention._PersistentPureContextShardedActiveAttentionBSHD,
        "apply",
        lambda *args: expected,
    )

    actual = cp_attention.pure_context_persistent_block_denoising_attention_bshd(
        query=query,
        active_key_shard=active_key,
        active_value_shard=active_value,
        global_key_shard=clean_key,
        global_value_shard=clean_value,
        global_seq_len=4,
        local_attn_mask=local_mask,
        global_attn_mask=clean_mask,
        runtime=runtime,
    )

    assert actual is expected
    assert gathered_args is not None
    assert gathered_args[2:4] == (4, 2)


def test_owner_major_clean_gradient_storage_is_reduce_scatter_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world_size = 4
    max_shard_len = 3
    batch = 2
    heads = 2
    head_dim = 8
    gathered = torch.empty(
        (world_size * max_shard_len, batch, 2, heads, head_dim),
        dtype=torch.bfloat16,
    )
    clean_key, clean_value = cp_attention._owner_major_clean_kv_bshd(
        gathered,
        world_size=world_size,
        max_shard_len=max_shard_len,
    )
    packed, grad_key, grad_value = cp_attention._owner_major_clean_grad_storage_bshd(
        clean_key,
        clean_value,
        max_shard_len=max_shard_len,
        world_size=world_size,
    )
    grad_key.fill_(1)
    grad_value.fill_(2)

    captured: torch.Tensor | None = None
    communication: dict[str, object] | None = None

    class Work:
        def wait(self) -> None:
            return None

    def reduce_scatter_tensor(
        output: torch.Tensor,
        input: torch.Tensor,
        *,
        group: object,
        async_op: bool,
    ) -> Work:
        nonlocal captured
        assert group == "group"
        assert async_op
        captured = input
        output.copy_(input[:max_shard_len])
        return Work()

    monkeypatch.setattr(dist, "reduce_scatter_tensor", reduce_scatter_tensor)

    def capture_communication(**metadata: object):
        nonlocal communication
        communication = metadata
        return nullcontext()

    monkeypatch.setattr(cp_attention, "communication_scope", capture_communication)
    work, reduced = cp_attention._reduce_scatter_owner_major_clean_grad_storage_bshd(
        packed,
        max_shard_len=max_shard_len,
        group="group",
        world_size=world_size,
    )
    work.wait()

    assert captured is packed
    input_bytes = int(packed.numel()) * int(packed.element_size())
    assert communication == {
        "domain": "attention",
        "phase": "backward",
        "collective": "reduce_scatter_tensor",
        "input_bytes": input_bytes,
        "logical_bytes": input_bytes * (world_size - 1) // world_size,
    }
    assert grad_key.stride() == clean_key.stride()
    assert grad_value.stride() == clean_value.stride()
    torch.testing.assert_close(reduced[:, :, 0], torch.ones_like(reduced[:, :, 0]))
    torch.testing.assert_close(reduced[:, :, 1], 2 * torch.ones_like(reduced[:, :, 1]))


def test_clean_kv_all_gather_records_runtime_logical_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world_size = 4
    key = torch.zeros((2, 3, 4, 5), dtype=torch.bfloat16)
    value = torch.ones_like(key)
    communication: dict[str, object] | None = None

    class Work:
        def wait(self) -> None:
            return None

    def all_gather_into_tensor(
        output: torch.Tensor,
        input: torch.Tensor,
        *,
        group: object,
        async_op: bool,
    ) -> Work:
        assert group == "group"
        assert async_op
        output[: input.shape[0]].copy_(input)
        return Work()

    def capture_communication(**metadata: object):
        nonlocal communication
        communication = metadata
        return nullcontext()

    monkeypatch.setattr(dist, "all_gather_into_tensor", all_gather_into_tensor)
    monkeypatch.setattr(cp_attention, "communication_scope", capture_communication)
    _, local, _ = cp_attention._all_gather_clean_kv_bshd(
        key,
        value,
        max_shard_len=4,
        group="group",
        world_size=world_size,
    )

    input_bytes = int(local.numel()) * int(local.element_size())
    assert communication == {
        "domain": "attention",
        "phase": "forward",
        "collective": "all_gather_into_tensor",
        "input_bytes": input_bytes,
        "logical_bytes": input_bytes * (world_size - 1),
    }


@pytest.mark.parametrize(
    ("transport", "expected_backend", "rejected_backend"),
    [
        (
            "collective",
            "_CollectiveWithLocalKVAttentionBSHD",
            "_RingWithLocalKVAttentionBSHD",
        ),
        (
            "streaming",
            "_RingWithLocalKVAttentionBSHD",
            "_CollectiveWithLocalKVAttentionBSHD",
        ),
    ],
)
def test_fused_cp_bp_api_routes_to_selected_production_transport(
    monkeypatch: pytest.MonkeyPatch,
    transport: str,
    expected_backend: str,
    rejected_backend: str,
) -> None:
    query = torch.randn(1, 4, 4, 8)
    local_key = torch.randn(1, 2, 2, 8)
    local_value = torch.randn_like(local_key)
    clean_key = torch.randn(1, 2, 2, 8)
    clean_value = torch.randn_like(clean_key)
    local_mask = BlockDenoisingLocalActiveMask(
        query_blocks=torch.tensor([0, 0, 0, 1], dtype=torch.int32),
        query_is_clean=torch.tensor([False, False, True, True]),
        active_blocks=torch.tensor([0], dtype=torch.int32),
        block_size=2,
    )
    global_mask = BlockDenoisingGlobalCleanMask(
        query_blocks=torch.tensor([0, 0, 0, 1], dtype=torch.int32),
        query_is_clean=torch.tensor([False, False, True, True]),
        block_size=2,
    )
    runtime = SimpleNamespace(
        enabled=True,
        local_parallel_size=2,
        block_parallel_size=2,
        cp_bp_policy=CPBPPolicy(clean_kv_transport=transport),
    )
    expected = torch.randn_like(query)
    captured: tuple[object, ...] | None = None

    def native_apply(*args: object) -> torch.Tensor:
        nonlocal captured
        captured = args
        return expected

    def old_apply(*args: object) -> torch.Tensor:
        del args
        raise AssertionError("fused BP/CP routed to the unselected transport")

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(getattr(cp_attention, expected_backend), "apply", native_apply)
    monkeypatch.setattr(getattr(cp_attention, rejected_backend), "apply", old_apply)

    actual = fused_block_context_attention_bshd(
        query,
        local_key,
        local_value,
        clean_key,
        clean_value,
        global_seq_len=4,
        local_attn_mask=local_mask,
        global_attn_mask=global_mask,
        runtime=runtime,
    )

    assert actual is expected
    assert captured is not None
    assert captured[0] is query
    assert captured[1] is local_key
    assert captured[2] is local_value
    assert captured[3] is clean_key
    assert captured[4] is clean_value
    assert captured[5] is local_mask
    assert captured[6] is global_mask
    assert captured[8:] == (runtime, 4)


def test_ring_owner_traversal_starts_local_and_visits_all_ranks() -> None:
    traversal = ring_owner_traversal(local_rank=2, context_parallel_size=4)

    assert traversal == (2, 3, 0, 1)


def test_prefix_visits_skip_noisy_blocks_own_clean_targets() -> None:
    visits = prefix_visits_for_noisy_blocks(
        active_blocks=[0, 3],
        block_size=2,
        seq_len=8,
        context_parallel_size=2,
    )
    counters = cp_bp_counters_from_visits(active_blocks=[0, 3], visits=visits)

    assert counters.active_blocks == 2
    assert counters.prefix_tokens_visited == 6
    assert counters.prefix_tokens_skipped == 4
    for visit in visits:
        self_start = visit.noisy_block * 2
        self_stop = self_start + 2
        assert visit.stop <= self_start
        if visit.skipped_tokens:
            assert visit.skipped_self_start is not None
            assert visit.skipped_self_stop is not None
            assert (
                self_start
                <= visit.skipped_self_start
                < visit.skipped_self_stop
                <= self_stop
            )


def test_global_clean_query_compaction_preserves_active_queries() -> None:
    block_size = 2
    mask = BlockDenoisingGlobalCleanMask(
        query_blocks=torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.int32),
        query_is_clean=torch.tensor(
            [False, False, False, False, True, True, True, True],
            dtype=torch.bool,
        ),
        block_size=block_size,
    )

    sparse = _owner_query_indices(
        attn_mask=mask,
        is_causal=False,
        query_len=8,
        key_start=4,
        device=torch.device("cpu"),
    )
    nearly_full_mask = BlockDenoisingGlobalCleanMask(
        query_blocks=torch.arange(16, dtype=torch.int32),
        query_is_clean=torch.zeros(16, dtype=torch.bool),
        block_size=block_size,
    )
    nearly_full = _owner_query_indices(
        attn_mask=nearly_full_mask,
        is_causal=False,
        query_len=16,
        key_start=0,
        device=torch.device("cpu"),
    )

    assert sparse is not None
    assert sparse.tolist() == [3, 6, 7]
    assert nearly_full is not None
    assert nearly_full.tolist() == list(range(1, 16))


def test_packed_dual_chunk_clean_mask_preserves_logical_blocks() -> None:
    block_size = 2
    base_mask = BlockDenoisingGlobalCleanMask(
        query_blocks=torch.arange(8, dtype=torch.int32),
        query_is_clean=torch.tensor(
            [False, False, False, False, True, True, True, True],
            dtype=torch.bool,
        ),
        block_size=block_size,
    )
    packed_mask, first_key_start = cp_attention._packed_clean_owner_mask(
        base_mask,
        intervals=((0, 4), (12, 16)),
        device=torch.device("cpu"),
    )
    q_idx = torch.arange(8, dtype=torch.long)[:, None]
    kv_idx = torch.arange(8, dtype=torch.long)[None, :]
    actual = _block_denoising_mask_spec_allowed(
        packed_mask,
        q_idx,
        kv_idx,
        first_key_start,
    )
    expected_key_blocks = torch.tensor([0, 0, 1, 1, 6, 6, 7, 7])
    expected = torch.where(
        base_mask.query_is_clean[:, None],
        base_mask.query_blocks[:, None] >= expected_key_blocks[None, :],
        base_mask.query_blocks[:, None] > expected_key_blocks[None, :],
    )

    torch.testing.assert_close(actual, expected)


def test_flex_kernel_options_require_provably_contiguous_masks() -> None:
    block_size = 2
    query_blocks = torch.arange(8, dtype=torch.int32) // block_size
    query_is_clean = torch.arange(8) >= 4

    global_clean = BlockDenoisingGlobalCleanMask(
        query_blocks=query_blocks,
        query_is_clean=query_is_clean,
        block_size=block_size,
    )
    local_active = BlockDenoisingLocalActiveMask(
        query_blocks=query_blocks,
        query_is_clean=query_is_clean,
        active_blocks=query_blocks,
        block_size=block_size,
    )
    full = BlockDenoisingFullMask(
        query_blocks=query_blocks,
        query_is_clean=query_is_clean,
        block_size=block_size,
        clean_offset=4,
    )

    assert cp_attention._flex_kernel_options(
        attn_mask=global_clean,
    ) == {"BLOCKS_ARE_CONTIGUOUS": True}
    assert cp_attention._flex_kernel_options(
        attn_mask=local_active,
    ) == {"BLOCKS_ARE_CONTIGUOUS": True}
    assert cp_attention._flex_kernel_options(attn_mask=full) is None
    assert cp_attention._flex_kernel_options(attn_mask=None) is None


def test_full_mask_query_compaction_disabled_for_correctness() -> None:
    block_size = 2
    mask = BlockDenoisingFullMask(
        query_blocks=torch.arange(8, dtype=torch.int32) // block_size,
        query_is_clean=torch.arange(8) >= 4,
        block_size=block_size,
        clean_offset=4,
    )

    assert (
        _owner_query_indices(
            attn_mask=mask,
            is_causal=False,
            query_len=8,
            key_start=4,
            device=torch.device("cpu"),
        )
        is None
    )


@pytest.mark.parametrize("key_start,key_stop", ((0, 4), (4, 8), (8, 12), (3, 10)))
def test_full_mask_interval_query_compaction_matches_exact_mask(
    key_start: int,
    key_stop: int,
) -> None:
    block_size = 2
    clean_offset = 8
    query_blocks = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.int32)
    query_is_clean = torch.tensor(
        [False, False, False, False, True, True, True, True],
        dtype=torch.bool,
    )
    mask = BlockDenoisingFullMask(
        query_blocks=query_blocks,
        query_is_clean=query_is_clean,
        block_size=block_size,
        clean_offset=clean_offset,
    )
    indices = _full_mask_interval_query_indices(
        attn_mask=mask,
        query_len=int(query_blocks.numel()),
        key_start=key_start,
        key_stop=key_stop,
        device=torch.device("cpu"),
    )
    selected = torch.arange(query_blocks.numel()) if indices is None else indices
    expected = torch.nonzero(
        _block_denoising_mask_spec_allowed(
            mask,
            torch.arange(query_blocks.numel())[:, None],
            torch.arange(key_stop - key_start)[None, :],
            key_start,
        ).any(dim=-1),
        as_tuple=False,
    ).flatten()
    torch.testing.assert_close(selected, expected)


def test_fp32_lse_merge_is_order_invariant() -> None:
    torch.manual_seed(5)
    stats = []
    for _ in range(4):
        logits = torch.randn(2, 3)
        m = logits.max(dim=-1).values
        exp = torch.exp(logits - m.unsqueeze(-1))
        value = torch.randn(2, 3, 5)
        stats.append(
            (torch.matmul(exp.unsqueeze(-2), value).squeeze(-2), m, exp.sum(dim=-1))
        )

    merged = merge_attention_stats(stats)
    reversed_merged = merge_attention_stats(reversed(stats))

    for actual, expected in zip(merged, reversed_merged, strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_owner_grad_buffer_matches_p2p_return_ring_layout() -> None:
    buffer = _allocate_owner_grad_buffer(
        torch.Size((2, 3, 7, 64)),
        max_shard_len=4,
        num_context_ranks=3,
        device=torch.device("cpu"),
    )

    assert buffer.shape == (3, 2, 3, 4, 64)
    assert buffer.dtype == torch.float32
    assert buffer.sum().item() == 0.0


def _ring_query(rank: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(100 + rank)
    return torch.randn(1, 2, 4, 64, generator=generator)


def _ring_grad(rank: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(200 + rank)
    return torch.randn(1, 2, 4, 64, generator=generator)


def _ring_global_mask_spec(device: torch.device) -> BlockDenoisingGlobalCleanMask:
    return BlockDenoisingGlobalCleanMask(
        query_blocks=torch.tensor([1, 2, 2, 3], device=device, dtype=torch.int32),
        query_is_clean=torch.tensor([False, False, True, True], device=device),
        block_size=2,
    )


def _global_clean_dense_mask(
    mask: BlockDenoisingGlobalCleanMask,
    *,
    key_len: int,
    key_start: int = 0,
) -> torch.Tensor:
    kv_pos = key_start + torch.arange(key_len, device=mask.query_blocks.device)
    kv_block = kv_pos // int(mask.block_size)
    q_block = mask.query_blocks[:, None]
    q_clean = mask.query_is_clean[:, None]
    return ((~q_clean) & (q_block > kv_block)) | (q_clean & (q_block >= kv_block))


def _owned_clean_intervals(
    seq_len: int,
    world_size: int,
    rank: int,
    *,
    layout: str = "contiguous",
) -> list[tuple[int, int]]:
    return [
        (shard.start, shard.stop)
        for shard in clean_shards_for_rank(
            seq_len=seq_len,
            context_parallel_size=world_size,
            rank=rank,
            layout=layout,
        )
    ]


def _assert_owned_intervals_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    intervals: list[tuple[int, int]],
    *,
    atol: float,
    rtol: float,
) -> None:
    owned = torch.zeros(actual.shape[-2], dtype=torch.bool, device=actual.device)
    for start, stop in intervals:
        torch.testing.assert_close(
            actual[..., start:stop, :],
            expected[..., start:stop, :],
            atol=atol,
            rtol=rtol,
        )
        owned[start:stop] = True
    assert actual[..., ~owned, :].abs().sum().item() == 0


def _ring_runtime(
    rank: int,
    world_size: int,
    *,
    block_parallel_size: int = 1,
    block_parallel_rank: int | None = None,
) -> SimpleNamespace:
    if block_parallel_rank is None:
        block_parallel_rank = rank if block_parallel_size > 1 else 0
    return SimpleNamespace(
        enabled=True,
        local_parallel_size=world_size,
        context_parallel_rank=rank,
        block_parallel_size=block_parallel_size,
        block_parallel_rank=block_parallel_rank,
        context_block_parallel_group=dist.group.WORLD,
        context_block_parallel_group_ranks=list(range(world_size)),
    )


def _gather_bshd_intervals(
    tensor: torch.Tensor,
    intervals: tuple[tuple[int, int], ...],
) -> torch.Tensor:
    return torch.cat(
        [tensor[:, start:stop] for start, stop in intervals],
        dim=1,
    ).contiguous()


def _dense_full_bdlm_attention_bshd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    query_logical_positions: torch.Tensor,
    seq_len: int,
    block_size: int,
) -> torch.Tensor:
    query_heads = int(query.shape[2])
    kv_heads = int(key.shape[2])
    if query_heads % kv_heads != 0:
        raise ValueError("query heads must be divisible by K/V heads")
    expanded_key = key.repeat_interleave(query_heads // kv_heads, dim=2)
    expanded_value = value.repeat_interleave(query_heads // kv_heads, dim=2)
    scale = 1.0 / math.sqrt(query.shape[-1])
    scores = (
        torch.einsum(
            "bqhd,bkhd->bhqk",
            query.float(),
            expanded_key.float(),
        )
        * scale
    )

    query_is_clean = query_logical_positions >= int(seq_len)
    query_positions = torch.where(
        query_is_clean,
        query_logical_positions - int(seq_len),
        query_logical_positions,
    )
    query_blocks = query_positions // int(block_size)
    key_logical_positions = torch.arange(
        2 * int(seq_len),
        device=query.device,
    )
    key_is_clean = key_logical_positions >= int(seq_len)
    key_positions = torch.where(
        key_is_clean,
        key_logical_positions - int(seq_len),
        key_logical_positions,
    )
    key_blocks = key_positions // int(block_size)
    allowed = torch.where(
        query_is_clean[:, None],
        key_is_clean[None, :] & (query_blocks[:, None] >= key_blocks[None, :]),
        ((~key_is_clean[None, :]) & (query_blocks[:, None] == key_blocks[None, :]))
        | (key_is_clean[None, :] & (query_blocks[:, None] > key_blocks[None, :])),
    )
    probabilities = torch.softmax(
        scores.masked_fill(~allowed[None, None], -torch.inf),
        dim=-1,
    )
    return torch.einsum("bhqk,bkhd->bqhd", probabilities, expanded_value.float())


def test_local_monolithic_attention_matches_dense_forward_and_backward() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDLM FlexAttention")
    device = torch.device("cuda", 0)
    seq_len = 32
    block_size = 8
    logical_positions = torch.arange(2 * seq_len, device=device)
    generator = torch.Generator(device=device).manual_seed(2026)
    full_query = torch.randn(
        1, 2 * seq_len, 4, 64, device=device, dtype=torch.bfloat16, generator=generator
    )
    full_key = torch.randn(
        1, 2 * seq_len, 2, 64, device=device, dtype=torch.bfloat16, generator=generator
    )
    full_value = torch.randn_like(full_key)
    query = full_query.clone().requires_grad_(True)
    key = full_key.clone().requires_grad_(True)
    value = full_value.clone().requires_grad_(True)
    mask = BlockDenoisingFullMask(
        query_blocks=(logical_positions.remainder(seq_len) // block_size).to(
            torch.int32
        ),
        query_is_clean=logical_positions >= seq_len,
        block_size=block_size,
        clean_offset=seq_len,
    )
    actual = full_mask_block_denoising_attention_bshd(
        query,
        key,
        value,
        global_seq_len=2 * seq_len,
        attn_mask=mask,
        runtime=SimpleNamespace(enabled=True, local_parallel_size=1),
    )
    grad_output = torch.randn_like(actual)
    actual.backward(grad_output)

    reference_query = full_query.float().requires_grad_(True)
    reference_key = full_key.float().requires_grad_(True)
    reference_value = full_value.float().requires_grad_(True)
    reference = _dense_full_bdlm_attention_bshd(
        reference_query,
        reference_key,
        reference_value,
        query_logical_positions=logical_positions,
        seq_len=seq_len,
        block_size=block_size,
    )
    reference.backward(grad_output.float())

    torch.testing.assert_close(actual.float(), reference, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(
        query.grad.float(), reference_query.grad, atol=3e-2, rtol=3e-2
    )
    torch.testing.assert_close(
        key.grad.float(), reference_key.grad, atol=3e-2, rtol=3e-2
    )
    torch.testing.assert_close(
        value.grad.float(), reference_value.grad, atol=3e-2, rtol=3e-2
    )


def _monolithic_cp_ring_worker(
    rank: int,
    world_size: int,
    init_file: str,
    queue,
    head_dim: int = 128,
    clean_kv_transport: str = "collective",
) -> None:
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
        )
        device = torch.device("cuda", rank)
        dtype = torch.bfloat16
        seq_len = 64
        block_size = 16
        batch = 2
        query_heads = 4
        kv_heads = 2
        head_dim = int(head_dim)
        generator = torch.Generator(device=device).manual_seed(710)
        full_query = torch.randn(
            batch,
            2 * seq_len,
            query_heads,
            head_dim,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        full_key = torch.randn(
            batch,
            2 * seq_len,
            kv_heads,
            head_dim,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        full_value = torch.randn(
            full_key.shape,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        intervals_by_owner = all_context_parallel_sequence_intervals(
            seq_len=seq_len,
            context_parallel_size=world_size,
        )
        local_intervals = intervals_by_owner[rank]
        local_positions = torch.cat(
            [
                torch.arange(start, stop, device=device, dtype=torch.long)
                for start, stop in local_intervals
            ]
        )
        local_model_positions = torch.where(
            local_positions >= seq_len,
            local_positions - seq_len,
            local_positions,
        )
        query = (
            _gather_bshd_intervals(
                full_query,
                local_intervals,
            )
            .clone()
            .requires_grad_(True)
        )
        key = (
            _gather_bshd_intervals(
                full_key,
                local_intervals,
            )
            .clone()
            .requires_grad_(True)
        )
        value = (
            _gather_bshd_intervals(
                full_value,
                local_intervals,
            )
            .clone()
            .requires_grad_(True)
        )
        mask = BlockDenoisingFullMask(
            query_blocks=(local_model_positions // block_size).to(torch.int32),
            query_is_clean=local_positions >= seq_len,
            block_size=block_size,
            clean_offset=seq_len,
        )
        runtime = _ring_runtime(rank, world_size)
        runtime.cp_bp_policy = CPBPPolicy(clean_kv_transport=clean_kv_transport)
        output = full_mask_block_denoising_attention_bshd(
            query,
            key,
            value,
            global_seq_len=2 * seq_len,
            attn_mask=mask,
            runtime=runtime,
        )

        reference_key = full_key.clone().requires_grad_(True)
        reference_value = full_value.clone().requires_grad_(True)
        reference_queries: list[torch.Tensor] = []
        reference_outputs: list[torch.Tensor] = []
        reference_loss = reference_key.new_tensor(0.0)
        local_grad = None
        for query_rank, query_intervals in enumerate(intervals_by_owner):
            reference_query = (
                _gather_bshd_intervals(
                    full_query,
                    query_intervals,
                )
                .clone()
                .requires_grad_(True)
            )
            reference_positions = torch.cat(
                [
                    torch.arange(start, stop, device=device, dtype=torch.long)
                    for start, stop in query_intervals
                ]
            )
            reference_output = _dense_full_bdlm_attention_bshd(
                reference_query,
                reference_key,
                reference_value,
                query_logical_positions=reference_positions,
                seq_len=seq_len,
                block_size=block_size,
            )
            grad_generator = torch.Generator(device=device).manual_seed(
                810 + query_rank
            )
            reference_grad = torch.randn(
                reference_output.shape,
                generator=grad_generator,
                device=device,
                dtype=dtype,
            )
            reference_loss = (
                reference_loss + (reference_output * reference_grad.float()).sum()
            )
            if query_rank == rank:
                local_grad = reference_grad
            reference_queries.append(reference_query)
            reference_outputs.append(reference_output)

        assert local_grad is not None
        torch.testing.assert_close(
            output.float(),
            reference_outputs[rank],
            atol=6e-2,
            rtol=6e-2,
        )
        (output * local_grad).sum().backward()
        reference_loss.backward()
        torch.testing.assert_close(
            query.grad,
            reference_queries[rank].grad,
            atol=8e-2,
            rtol=8e-2,
        )
        torch.testing.assert_close(
            key.grad,
            _gather_bshd_intervals(reference_key.grad, local_intervals),
            atol=8e-2,
            rtol=8e-2,
        )
        torch.testing.assert_close(
            value.grad,
            _gather_bshd_intervals(reference_value.grad, local_intervals),
            atol=8e-2,
            rtol=8e-2,
        )
        queue.put((rank, "ok", ""))
    except Exception:  # pragma: no cover - child reports failure to parent.
        queue.put((rank, "error", traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("clean_kv_transport", ["collective", "streaming"])
def test_monolithic_cp_ring_matches_dense_full_graph_forward_and_backward(
    world_size: int,
    clean_kv_transport: str,
) -> None:
    if (
        not dist.is_available()
        or not torch.cuda.is_available()
        or torch.cuda.device_count() < world_size
    ):
        pytest.skip(f"requires {world_size} CUDA devices")
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    with tempfile.NamedTemporaryFile(delete=False) as init:
        init_file = init.name
    try:
        processes = [
            ctx.Process(
                target=_monolithic_cp_ring_worker,
                args=(rank, world_size, init_file, queue, 128, clean_kv_transport),
            )
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=120)
        for process in processes:
            assert process.exitcode == 0
        results = [queue.get(timeout=5) for _ in range(world_size)]
        errors = [message for _, status, message in results if status != "ok"]
        assert not errors, "\n".join(errors)
    finally:
        if os.path.exists(init_file):
            os.unlink(init_file)


@pytest.mark.parametrize("clean_kv_transport", ["collective", "streaming"])
def test_wide_head_cp_ring_matches_dense_full_graph_forward_and_backward(clean_kv_transport: str) -> None:
    world_size = 2
    if (
        not dist.is_available()
        or not torch.cuda.is_available()
        or torch.cuda.device_count() < world_size
    ):
        pytest.skip(f"requires {world_size} CUDA devices")
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    with tempfile.NamedTemporaryFile(delete=False) as init:
        init_file = init.name
    try:
        processes = [
            ctx.Process(
                target=_monolithic_cp_ring_worker,
                args=(rank, world_size, init_file, queue, 512, clean_kv_transport),
            )
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=300)
        for process in processes:
            assert process.exitcode == 0
        results = [queue.get(timeout=5) for _ in range(world_size)]
        errors = [message for _, status, message in results if status != "ok"]
        assert not errors, "\n".join(errors)
    finally:
        if os.path.exists(init_file):
            os.unlink(init_file)


def _checkpointed_multilayer_cp_ring_worker(
    rank: int,
    world_size: int,
    init_file: str,
    queue,
) -> None:
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
        )
        device = torch.device("cuda", rank)
        dtype = torch.bfloat16
        seq_len = 32_768
        block_size = 32
        hidden_size = 512
        query_heads = 4
        kv_heads = 2
        head_dim = 128
        intervals_by_owner = all_context_parallel_sequence_intervals(
            seq_len=seq_len,
            context_parallel_size=world_size,
        )
        local_intervals = intervals_by_owner[rank]
        local_positions = torch.cat(
            [
                torch.arange(start, stop, device=device, dtype=torch.long)
                for start, stop in local_intervals
            ]
        )
        local_model_positions = torch.where(
            local_positions >= seq_len,
            local_positions - seq_len,
            local_positions,
        )
        mask = BlockDenoisingFullMask(
            query_blocks=(local_model_positions // block_size).to(torch.int32),
            query_is_clean=local_positions >= seq_len,
            block_size=block_size,
            clean_offset=seq_len,
        )
        generator = torch.Generator(device=device).manual_seed(910)
        hidden = torch.randn(
            1,
            local_positions.numel(),
            hidden_size,
            generator=generator,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        weights = []
        for _ in range(3):
            weights.append(
                tuple(
                    torch.randn(
                        shape,
                        generator=generator,
                        device=device,
                        dtype=dtype,
                    )
                    .div_(math.sqrt(hidden_size))
                    .requires_grad_(True)
                    for shape in (
                        (hidden_size, query_heads * head_dim),
                        (hidden_size, kv_heads * head_dim),
                        (hidden_size, kv_heads * head_dim),
                        (query_heads * head_dim, hidden_size),
                    )
                )
            )

        for query_weight, key_weight, value_weight, output_weight in weights:

            def layer_forward(
                states: torch.Tensor,
                qw: torch.Tensor = query_weight,
                kw: torch.Tensor = key_weight,
                vw: torch.Tensor = value_weight,
                ow: torch.Tensor = output_weight,
            ) -> torch.Tensor:
                query = (states @ qw).view(
                    states.shape[0], states.shape[1], query_heads, head_dim
                )
                key = (states @ kw).view(
                    states.shape[0], states.shape[1], kv_heads, head_dim
                )
                value = (states @ vw).view(
                    states.shape[0], states.shape[1], kv_heads, head_dim
                )
                attended = full_mask_block_denoising_attention_bshd(
                    query,
                    key,
                    value,
                    global_seq_len=2 * seq_len,
                    attn_mask=mask,
                    runtime=_ring_runtime(rank, world_size),
                )
                return states + attended.flatten(2) @ ow

            hidden = checkpoint(layer_forward, hidden, use_reentrant=True)

        hidden.float().square().mean().backward()
        gradients = [hidden.grad] if hidden.is_leaf else []
        gradients.extend(
            parameter.grad for layer_weights in weights for parameter in layer_weights
        )
        if any(gradient is None for gradient in gradients):
            raise RuntimeError("checkpointed CP graph did not produce all gradients")
        if any(not torch.isfinite(gradient).all() for gradient in gradients):
            raise RuntimeError("checkpointed CP graph produced nonfinite gradients")
        completed = torch.ones((), device=device)
        dist.all_reduce(completed)
        queue.put((rank, "ok", ""))
    except Exception:  # pragma: no cover - child reports failure to parent.
        queue.put((rank, "error", traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _large_ring_transport_worker(
    rank: int,
    world_size: int,
    init_file: str,
    queue,
) -> None:
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
        )
        device = torch.device("cuda", rank)
        group_ranks = list(range(world_size))
        # Combined K/V plus returning dK/V payload for 32K CP4 GQA.
        payload_elements = 2 * 2 * 16_384 * 2 * 128
        send = torch.full(
            (payload_elements,),
            float(rank),
            dtype=torch.bfloat16,
            device=device,
        )
        recv = torch.empty_like(send)
        for step in range(world_size):
            work = _ring_exchange_flat_async(
                send,
                recv,
                local_rank=rank,
                send_rank=group_ranks[(rank - 1) % world_size],
                recv_rank=group_ranks[(rank + 1) % world_size],
                group=dist.group.WORLD,
            )
            _ring_exchange_flat_wait(work)
            expected = float((rank + step + 1) % world_size)
            if not torch.all(recv == expected):
                raise RuntimeError(
                    "large ring payload was received from the wrong owner"
                )
            send, recv = recv, send
        queue.put((rank, "ok", ""))
    except Exception:  # pragma: no cover - child reports failure to parent.
        queue.put((rank, "error", traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_large_ring_transport_completes_on_four_ranks() -> None:
    world_size = 4
    if (
        not dist.is_available()
        or not torch.cuda.is_available()
        or torch.cuda.device_count() < world_size
    ):
        pytest.skip(f"requires {world_size} CUDA devices")
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    with tempfile.NamedTemporaryFile(delete=False) as init:
        init_file = init.name
    try:
        processes = [
            ctx.Process(
                target=_large_ring_transport_worker,
                args=(rank, world_size, init_file, queue),
            )
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=120)
        for process in processes:
            assert process.exitcode == 0
        results = [queue.get(timeout=5) for _ in range(world_size)]
        errors = [message for _, status, message in results if status != "ok"]
        assert not errors, "\n".join(errors)
    finally:
        if os.path.exists(init_file):
            os.unlink(init_file)


def test_checkpointed_multilayer_cp_ring_completes_on_four_ranks() -> None:
    world_size = 4
    if (
        not dist.is_available()
        or not torch.cuda.is_available()
        or torch.cuda.device_count() < world_size
    ):
        pytest.skip(f"requires {world_size} CUDA devices")
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    with tempfile.NamedTemporaryFile(delete=False) as init:
        init_file = init.name
    try:
        processes = [
            ctx.Process(
                target=_checkpointed_multilayer_cp_ring_worker,
                args=(rank, world_size, init_file, queue),
            )
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=120)
        for process in processes:
            assert process.exitcode == 0
        results = [queue.get(timeout=5) for _ in range(world_size)]
        errors = [message for _, status, message in results if status != "ok"]
        assert not errors, "\n".join(errors)
    finally:
        if os.path.exists(init_file):
            os.unlink(init_file)


def _ring_reference_loss(
    key: torch.Tensor,
    value: torch.Tensor,
    world_size: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    queries = [
        _ring_query(rank).to(device=key.device, dtype=key.dtype).requires_grad_(True)
        for rank in range(world_size)
    ]
    loss = key.new_tensor(0.0)
    mask = _global_clean_dense_mask(
        _ring_global_mask_spec(key.device),
        key_len=key.shape[-2],
    ).view(1, 1, 4, key.shape[-2])
    for rank, query in enumerate(queries):
        out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=0.0,
        )
        loss = (
            loss + (out * _ring_grad(rank).to(device=key.device, dtype=key.dtype)).sum()
        )
    return loss, queries


def _hybrid_query(
    rank: int,
    head_dim: int = 64,
    query_heads: int = 2,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(300 + rank)
    return torch.randn(
        1,
        int(query_heads),
        64,
        int(head_dim),
        generator=generator,
    )


def _hybrid_grad(
    rank: int,
    head_dim: int = 64,
    query_heads: int = 2,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(400 + rank)
    return torch.randn(
        1,
        int(query_heads),
        64,
        int(head_dim),
        generator=generator,
    )


def _hybrid_local_mask_spec(device: torch.device) -> BlockDenoisingLocalActiveMask:
    block_size = 16
    return BlockDenoisingLocalActiveMask(
        query_blocks=torch.cat(
            (
                torch.zeros(block_size, device=device, dtype=torch.int32),
                torch.ones(block_size, device=device, dtype=torch.int32),
                torch.zeros(block_size, device=device, dtype=torch.int32),
                torch.ones(block_size, device=device, dtype=torch.int32),
            )
        ),
        query_is_clean=torch.arange(4 * block_size, device=device) >= 2 * block_size,
        active_blocks=torch.repeat_interleave(
            torch.arange(2, device=device, dtype=torch.int32),
            block_size,
        ),
        block_size=block_size,
    )


def _hybrid_local_dense_mask(mask: BlockDenoisingLocalActiveMask) -> torch.Tensor:
    q_block = mask.query_blocks[:, None]
    q_clean = mask.query_is_clean[:, None]
    active_block = mask.active_blocks[None, :]
    return (~q_clean) & (q_block == active_block)


def _hybrid_global_mask_spec(device: torch.device) -> BlockDenoisingGlobalCleanMask:
    block_size = 16
    return BlockDenoisingGlobalCleanMask(
        query_blocks=torch.cat(
            (
                torch.ones(block_size, device=device, dtype=torch.int32),
                torch.full((block_size,), 2, device=device, dtype=torch.int32),
                torch.ones(block_size, device=device, dtype=torch.int32),
                torch.full((block_size,), 2, device=device, dtype=torch.int32),
            )
        ),
        query_is_clean=torch.arange(4 * block_size, device=device) >= 2 * block_size,
        block_size=block_size,
    )


def _hybrid_reference_loss(
    local_keys: list[torch.Tensor],
    local_values: list[torch.Tensor],
    global_key: torch.Tensor,
    global_value: torch.Tensor,
    world_size: int,
    head_dim: int = 64,
    query_heads: int = 2,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    device = global_key.device
    queries = [
        _hybrid_query(rank, head_dim, query_heads)
        .to(
            device=device,
            dtype=global_key.dtype,
        )
        .requires_grad_(True)
        for rank in range(world_size)
    ]
    loss = global_key.new_tensor(0.0)
    for rank, query in enumerate(queries):
        key = torch.cat((local_keys[rank], global_key), dim=-2)
        value = torch.cat((local_values[rank], global_value), dim=-2)
        local_mask = _hybrid_local_mask_spec(device)
        global_mask = _hybrid_global_mask_spec(device)
        mask = torch.cat(
            (
                _hybrid_local_dense_mask(local_mask),
                _global_clean_dense_mask(global_mask, key_len=global_key.shape[-2]),
            ),
            dim=-1,
        ).view(1, 1, query.shape[-2], key.shape[-2])
        out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=0.0,
            enable_gqa=int(query_heads) != int(key.shape[1]),
        )
        loss = (
            loss
            + (
                out
                * _hybrid_grad(rank, head_dim, query_heads).to(
                    device=device,
                    dtype=global_key.dtype,
                )
            ).sum()
        )
    return loss, queries


def _nccl_ring_attention_worker(
    rank: int,
    world_size: int,
    init_file: str,
    queue,
) -> None:
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
        )
        device = torch.device("cuda", rank)
        generator = torch.Generator(device=device).manual_seed(31)
        dtype = torch.bfloat16
        full_key = torch.randn(
            1, 2, 7, 64, generator=generator, device=device, dtype=dtype
        )
        full_value = torch.randn(
            1, 2, 7, 64, generator=generator, device=device, dtype=dtype
        )

        query = _ring_query(rank).to(device=device, dtype=dtype).requires_grad_(True)
        key = full_key.clone().requires_grad_(True)
        value = full_value.clone().requires_grad_(True)
        out = ring_context_parallel_attention(
            query,
            key,
            value,
            attn_mask=_ring_global_mask_spec(device),
            runtime=_ring_runtime(rank, world_size),
            key_chunk_size=2,
        )

        ref_key = full_key.clone().requires_grad_(True)
        ref_value = full_value.clone().requires_grad_(True)
        ref_queries = [
            _ring_query(worker_rank).to(device=device, dtype=dtype).requires_grad_(True)
            for worker_rank in range(world_size)
        ]
        ref_loss = ref_key.new_tensor(0.0)
        for worker_rank, ref_query in enumerate(ref_queries):
            ref_out = F.scaled_dot_product_attention(
                ref_query,
                ref_key,
                ref_value,
                attn_mask=_global_clean_dense_mask(
                    _ring_global_mask_spec(device),
                    key_len=ref_key.shape[-2],
                ).view(1, 1, 4, ref_key.shape[-2]),
                dropout_p=0.0,
            )
            ref_loss = (
                ref_loss
                + (
                    ref_out * _ring_grad(worker_rank).to(device=device, dtype=dtype)
                ).sum()
            )

        ref_out_for_rank = F.scaled_dot_product_attention(
            ref_queries[rank],
            full_key,
            full_value,
            attn_mask=_global_clean_dense_mask(
                _ring_global_mask_spec(device),
                key_len=full_key.shape[-2],
            ).view(1, 1, 4, full_key.shape[-2]),
            dropout_p=0.0,
        )
        torch.testing.assert_close(out, ref_out_for_rank, atol=5e-2, rtol=5e-2)

        (out * _ring_grad(rank).to(device=device, dtype=dtype)).sum().backward()
        ref_loss.backward()
        intervals = _owned_clean_intervals(full_key.shape[-2], world_size, rank)
        torch.testing.assert_close(
            query.grad,
            ref_queries[rank].grad,
            atol=5e-2,
            rtol=5e-2,
        )
        _assert_owned_intervals_close(
            key.grad,
            ref_key.grad,
            atol=5e-2,
            rtol=5e-2,
            intervals=intervals,
        )
        _assert_owned_intervals_close(
            value.grad,
            ref_value.grad,
            atol=5e-2,
            rtol=5e-2,
            intervals=intervals,
        )
        queue.put((rank, "ok", ""))
    except Exception:  # pragma: no cover - child reports failure to parent.
        queue.put((rank, "error", traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _hybrid_ring_attention_worker(
    rank: int,
    world_size: int,
    init_file: str,
    queue,
) -> None:
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
        )
        device = torch.device("cuda", rank)
        generator = torch.Generator(device=device).manual_seed(22)
        dtype = torch.bfloat16
        full_global_key = torch.randn(
            1, 2, 31, 64, generator=generator, device=device, dtype=dtype
        )
        full_global_value = torch.randn(
            1, 2, 31, 64, generator=generator, device=device, dtype=dtype
        )
        local_keys = [
            torch.randn(1, 2, 32, 64, generator=generator, device=device, dtype=dtype)
            for _ in range(world_size)
        ]
        local_values = [
            torch.randn(1, 2, 32, 64, generator=generator, device=device, dtype=dtype)
            for _ in range(world_size)
        ]

        query = _hybrid_query(rank).to(device=device, dtype=dtype).requires_grad_(True)
        local_key = local_keys[rank].clone().requires_grad_(True)
        local_value = local_values[rank].clone().requires_grad_(True)
        global_key = full_global_key.clone().requires_grad_(True)
        global_value = full_global_value.clone().requires_grad_(True)
        out = ring_attention_with_local_kv(
            query,
            local_key,
            local_value,
            global_key,
            global_value,
            local_attn_mask=_hybrid_local_mask_spec(device),
            global_attn_mask=_hybrid_global_mask_spec(device),
            runtime=_ring_runtime(
                rank,
                world_size,
                block_parallel_size=world_size,
            ),
            key_chunk_size=2,
        )

        ref_local_keys = [
            tensor.clone().requires_grad_(i == rank)
            for i, tensor in enumerate(local_keys)
        ]
        ref_local_values = [
            tensor.clone().requires_grad_(i == rank)
            for i, tensor in enumerate(local_values)
        ]
        ref_global_key = full_global_key.clone().requires_grad_(True)
        ref_global_value = full_global_value.clone().requires_grad_(True)
        ref_loss, ref_queries = _hybrid_reference_loss(
            ref_local_keys,
            ref_local_values,
            ref_global_key,
            ref_global_value,
            world_size,
        )
        ref_key = torch.cat((local_keys[rank], full_global_key), dim=-2)
        ref_value = torch.cat((local_values[rank], full_global_value), dim=-2)
        local_mask = _hybrid_local_mask_spec(device)
        global_mask = _hybrid_global_mask_spec(device)
        ref_mask = torch.cat(
            (
                _hybrid_local_dense_mask(local_mask),
                _global_clean_dense_mask(
                    global_mask, key_len=full_global_key.shape[-2]
                ),
            ),
            dim=-1,
        ).view(1, 1, ref_queries[rank].shape[-2], ref_key.shape[-2])
        ref_out = F.scaled_dot_product_attention(
            ref_queries[rank].to(device),
            ref_key,
            ref_value,
            attn_mask=ref_mask,
            dropout_p=0.0,
        )
        torch.testing.assert_close(out, ref_out, atol=5e-2, rtol=5e-2)

        (out * _hybrid_grad(rank).to(device=device, dtype=dtype)).sum().backward()
        ref_loss.backward()
        intervals = _owned_clean_intervals(
            full_global_key.shape[-2],
            world_size,
            rank,
            layout="dual_chunk",
        )
        torch.testing.assert_close(
            query.grad,
            ref_queries[rank].grad,
            atol=5e-2,
            rtol=5e-2,
        )
        torch.testing.assert_close(
            local_key.grad,
            ref_local_keys[rank].grad,
            atol=5e-2,
            rtol=5e-2,
        )
        torch.testing.assert_close(
            local_value.grad,
            ref_local_values[rank].grad,
            atol=5e-2,
            rtol=5e-2,
        )
        _assert_owned_intervals_close(
            global_key.grad,
            ref_global_key.grad,
            atol=5e-2,
            rtol=5e-2,
            intervals=intervals,
        )
        _assert_owned_intervals_close(
            global_value.grad,
            ref_global_value.grad,
            atol=5e-2,
            rtol=5e-2,
            intervals=intervals,
        )
        queue.put((rank, "ok", ""))
    except Exception:  # pragma: no cover - child reports failure to parent.
        queue.put((rank, "error", traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _hybrid_sharded_collective_attention_worker(
    rank: int,
    world_size: int,
    init_file: str,
    queue,
    head_dim: int = 64,
    query_heads: int = 8,
    key_value_heads: int = 2,
    clean_kv_transport: str = "collective",
) -> None:
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
        )
        device = torch.device("cuda", rank)
        generator = torch.Generator(device=device).manual_seed(22)
        dtype = torch.bfloat16
        head_dim = int(head_dim)
        key_value_heads = int(key_value_heads)
        full_global_key = torch.randn(
            1,
            key_value_heads,
            31,
            head_dim,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        full_global_value = torch.randn(
            1,
            key_value_heads,
            31,
            head_dim,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        local_keys = [
            torch.randn(
                1,
                key_value_heads,
                32,
                head_dim,
                generator=generator,
                device=device,
                dtype=dtype,
            )
            for _ in range(world_size)
        ]
        local_values = [
            torch.randn(
                1,
                key_value_heads,
                32,
                head_dim,
                generator=generator,
                device=device,
                dtype=dtype,
            )
            for _ in range(world_size)
        ]

        intervals = _owned_clean_intervals(
            full_global_key.shape[-2],
            world_size,
            rank,
            layout="dual_chunk",
        )
        query = (
            _hybrid_query(rank, head_dim, query_heads)
            .to(device=device, dtype=dtype)
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        local_key = (
            local_keys[rank].transpose(1, 2).contiguous().clone().requires_grad_(True)
        )
        local_value = (
            local_values[rank].transpose(1, 2).contiguous().clone().requires_grad_(True)
        )
        global_key_shard = (
            torch.cat(
                [full_global_key[..., start:stop, :] for start, stop in intervals],
                dim=-2,
            )
            .transpose(1, 2)
            .contiguous()
            .clone()
            .requires_grad_(True)
        )
        global_value_shard = (
            torch.cat(
                [full_global_value[..., start:stop, :] for start, stop in intervals],
                dim=-2,
            )
            .transpose(1, 2)
            .contiguous()
            .clone()
            .requires_grad_(True)
        )
        runtime = _ring_runtime(
            rank,
            world_size,
            block_parallel_size=world_size,
        )
        runtime.cp_bp_policy = CPBPPolicy(clean_kv_transport=clean_kv_transport)
        out_bshd = fused_block_context_attention_bshd(
            query,
            local_key,
            local_value,
            global_key_shard,
            global_value_shard,
            global_seq_len=full_global_key.shape[-2],
            local_attn_mask=_hybrid_local_mask_spec(device),
            global_attn_mask=_hybrid_global_mask_spec(device),
            runtime=runtime,
            key_chunk_size=2,
        )
        out = out_bshd.transpose(1, 2)

        ref_local_keys = [
            tensor.clone().requires_grad_(i == rank)
            for i, tensor in enumerate(local_keys)
        ]
        ref_local_values = [
            tensor.clone().requires_grad_(i == rank)
            for i, tensor in enumerate(local_values)
        ]
        ref_global_key = full_global_key.clone().requires_grad_(True)
        ref_global_value = full_global_value.clone().requires_grad_(True)
        ref_loss, ref_queries = _hybrid_reference_loss(
            ref_local_keys,
            ref_local_values,
            ref_global_key,
            ref_global_value,
            world_size,
            head_dim,
            query_heads,
        )
        ref_key = torch.cat((local_keys[rank], full_global_key), dim=-2)
        ref_value = torch.cat((local_values[rank], full_global_value), dim=-2)
        local_mask = _hybrid_local_mask_spec(device)
        global_mask = _hybrid_global_mask_spec(device)
        ref_mask = torch.cat(
            (
                _hybrid_local_dense_mask(local_mask),
                _global_clean_dense_mask(
                    global_mask, key_len=full_global_key.shape[-2]
                ),
            ),
            dim=-1,
        ).view(1, 1, ref_queries[rank].shape[-2], ref_key.shape[-2])
        ref_out = F.scaled_dot_product_attention(
            ref_queries[rank].to(device),
            ref_key,
            ref_value,
            attn_mask=ref_mask,
            dropout_p=0.0,
            enable_gqa=int(query_heads) != int(ref_key.shape[1]),
        )
        tolerance = 1.2e-1 if head_dim > 256 else 5e-2
        torch.testing.assert_close(
            out,
            ref_out,
            atol=tolerance,
            rtol=tolerance,
        )

        (
            out_bshd
            * _hybrid_grad(rank, head_dim, query_heads)
            .to(
                device=device,
                dtype=dtype,
            )
            .transpose(1, 2)
        ).sum().backward()
        ref_loss.backward()

        gradient_errors: list[str] = []

        def _check_gradient(
            name: str, actual: torch.Tensor, expected: torch.Tensor
        ) -> None:
            try:
                torch.testing.assert_close(
                    actual,
                    expected,
                    atol=tolerance,
                    rtol=tolerance,
                )
            except AssertionError as exc:
                gradient_errors.append(f"{name}:\n{exc}")

        _check_gradient("dQ", query.grad.transpose(1, 2), ref_queries[rank].grad)
        _check_gradient(
            "local dK",
            local_key.grad.transpose(1, 2),
            ref_local_keys[rank].grad,
        )
        _check_gradient(
            "local dV",
            local_value.grad.transpose(1, 2),
            ref_local_values[rank].grad,
        )
        _check_gradient(
            "clean-shard dK",
            global_key_shard.grad.transpose(1, 2),
            torch.cat(
                [ref_global_key.grad[..., start:stop, :] for start, stop in intervals],
                dim=-2,
            ),
        )
        _check_gradient(
            "clean-shard dV",
            global_value_shard.grad.transpose(1, 2),
            torch.cat(
                [
                    ref_global_value.grad[..., start:stop, :]
                    for start, stop in intervals
                ],
                dim=-2,
            ),
        )
        assert not gradient_errors, "\n\n".join(gradient_errors)
        queue.put((rank, "ok", ""))
    except Exception:  # pragma: no cover - child reports failure to parent.
        queue.put((rank, "error", traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _windowed_fused_collective_worker(
    rank: int,
    world_size: int,
    init_file: str,
    queue,
    block_parallel_size: int,
    block_group_index: int,
    production_cuda: bool = False,
) -> None:
    try:
        original_shard_stats = cp_attention._flex_shard_stats
        original_backward = cp_attention._packed_clean_flex_shard_backward_exact_bshd
        if not production_cuda:
            cp_attention._flex_shard_stats = _dense_exact_shard_stats_bhsd
            cp_attention._packed_clean_flex_shard_backward_exact_bshd = (
                _dense_exact_packed_backward_bshd
            )
        if production_cuda:
            torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl" if production_cuda else "gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
        )
        device = torch.device("cuda", rank) if production_cuda else torch.device("cpu")
        dtype = torch.bfloat16 if production_cuda else torch.float32
        block_size = 4
        seq_len = max(32, 2 * block_size * block_parallel_size)
        window = 8
        query_heads = 32 if production_cuda else 16
        kv_heads = 2
        head_dim = 128 if production_cuda else 512
        schedule = build_block_schedule(
            num_blocks=seq_len // block_size,
            block_parallel_size=block_parallel_size,
            context_parallel_size=world_size,
        )
        first_block_rank = block_group_index * world_size
        active_blocks_by_rank = [
            tuple(sorted(schedule.active_blocks_by_worker[first_block_rank + worker]))
            for worker in range(world_size)
        ]
        clean_intervals_by_rank = [
            tuple(
                (shard.start, shard.stop)
                for shard in clean_shards_for_rank(
                    seq_len=seq_len,
                    context_parallel_size=world_size,
                    rank=worker,
                    layout="dual_chunk",
                )
            )
            for worker in range(world_size)
        ]

        generator = torch.Generator(device=device).manual_seed(812)
        full_clean_key = torch.randn(
            1,
            kv_heads,
            seq_len,
            head_dim,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        full_clean_value = torch.randn(
            full_clean_key.shape,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        local_keys = []
        local_values = []
        queries = []
        query_positions = []
        for worker in range(world_size):
            active_positions = torch.cat(
                [
                    torch.arange(
                        block * block_size,
                        (block + 1) * block_size,
                        device=device,
                        dtype=torch.long,
                    )
                    for block in active_blocks_by_rank[worker]
                ]
            )
            clean_positions = torch.cat(
                [
                    torch.arange(start, stop, device=device, dtype=torch.long)
                    for start, stop in clean_intervals_by_rank[worker]
                ]
            )
            positions = torch.cat((active_positions, clean_positions))
            query_positions.append((active_positions, clean_positions, positions))
            local_keys.append(
                torch.randn(
                    1,
                    kv_heads,
                    int(active_positions.numel()),
                    head_dim,
                    generator=generator,
                    device=device,
                    dtype=dtype,
                )
            )
            local_values.append(
                torch.randn(
                    local_keys[-1].shape,
                    generator=generator,
                    device=device,
                    dtype=dtype,
                )
            )
            queries.append(
                torch.randn(
                    1,
                    query_heads,
                    int(positions.numel()),
                    head_dim,
                    generator=generator,
                    device=device,
                    dtype=dtype,
                )
            )

        active_positions, clean_positions, positions = query_positions[rank]
        query = queries[rank].transpose(1, 2).contiguous().requires_grad_(True)
        local_key = local_keys[rank].transpose(1, 2).contiguous().requires_grad_(True)
        local_value = (
            local_values[rank].transpose(1, 2).contiguous().requires_grad_(True)
        )
        intervals = clean_intervals_by_rank[rank]
        clean_key = _gather_bshd_intervals(
            full_clean_key.transpose(1, 2), intervals
        ).requires_grad_(True)
        clean_value = _gather_bshd_intervals(
            full_clean_value.transpose(1, 2), intervals
        ).requires_grad_(True)
        query_is_clean = torch.arange(int(positions.numel()), device=device) >= int(
            active_positions.numel()
        )
        clean_stops = torch.where(
            query_is_clean,
            positions + 1,
            (positions // block_size) * block_size,
        ).to(torch.int32)
        histories = torch.where(
            query_is_clean,
            torch.full_like(clean_stops, window),
            torch.full_like(clean_stops, window - 1),
        )
        clean_bounds = torch.stack(
            (torch.clamp(clean_stops - histories, min=0), clean_stops), dim=-1
        )
        query_blocks = (positions // block_size).to(torch.int32)
        local_mask = BlockDenoisingLocalActiveMask(
            query_blocks=query_blocks,
            query_is_clean=query_is_clean,
            active_blocks=(active_positions // block_size).to(torch.int32),
            block_size=block_size,
        )
        global_mask = BlockDenoisingGlobalCleanMask(
            query_blocks=query_blocks,
            query_is_clean=query_is_clean,
            block_size=block_size,
            clean_context_window=window,
            query_clean_bounds=clean_bounds,
            clean_key_positions=clean_positions.to(torch.int32),
        )
        output = fused_block_context_attention_bshd(
            query,
            local_key,
            local_value,
            clean_key,
            clean_value,
            global_seq_len=seq_len,
            local_attn_mask=local_mask,
            global_attn_mask=global_mask,
            runtime=_ring_runtime(
                rank,
                world_size,
                block_parallel_size=block_parallel_size,
                block_parallel_rank=first_block_rank + rank,
            ),
        )

        reference_clean_key = full_clean_key.clone().requires_grad_(True)
        reference_clean_value = full_clean_value.clone().requires_grad_(True)
        reference_queries = []
        reference_local_keys = []
        reference_local_values = []
        reference_grads = []
        reference_loss = reference_clean_key.new_tensor(0.0)
        reference_for_rank = None
        scale = 1.0 / math.sqrt(head_dim)
        for worker in range(world_size):
            worker_active, _, worker_positions = query_positions[worker]
            worker_query = queries[worker].clone().requires_grad_(True)
            worker_key = local_keys[worker].clone().requires_grad_(True)
            worker_value = local_values[worker].clone().requires_grad_(True)
            reference_queries.append(worker_query)
            reference_local_keys.append(worker_key)
            reference_local_values.append(worker_value)
            worker_clean = torch.arange(
                int(worker_positions.numel()), device=device
            ) >= int(worker_active.numel())
            worker_stops = torch.where(
                worker_clean,
                worker_positions + 1,
                (worker_positions // block_size) * block_size,
            )
            worker_history = torch.where(
                worker_clean,
                torch.full_like(worker_stops, window),
                torch.full_like(worker_stops, window - 1),
            )
            worker_starts = torch.clamp(worker_stops - worker_history, min=0)
            clean_position = torch.arange(seq_len, device=device)
            clean_allowed = (clean_position[None] >= worker_starts[:, None]) & (
                clean_position[None] < worker_stops[:, None]
            )
            local_allowed = (~worker_clean[:, None]) & (
                (worker_positions // block_size)[:, None]
                == (worker_active // block_size)[None]
            )
            key_h = torch.cat((worker_key, reference_clean_key), dim=-2)
            value_h = torch.cat((worker_value, reference_clean_value), dim=-2)
            allowed = torch.cat((local_allowed, clean_allowed), dim=-1)
            scores = (
                torch.einsum(
                    "bhqd,bhkd->bhqk",
                    worker_query.float(),
                    key_h.repeat_interleave(query_heads // kv_heads, dim=1).float(),
                )
                * scale
            )
            probabilities = torch.softmax(
                scores.masked_fill(~allowed[None, None], -torch.inf), dim=-1
            )
            reference_output = torch.einsum(
                "bhqk,bhkd->bhqd",
                probabilities,
                value_h.repeat_interleave(query_heads // kv_heads, dim=1).float(),
            )
            if worker == rank:
                reference_for_rank = reference_output
            grad = torch.randn(
                reference_output.shape,
                generator=torch.Generator(device=device).manual_seed(900 + worker),
                device=device,
                dtype=dtype,
            )
            reference_grads.append(grad)
            reference_loss = reference_loss + (reference_output * grad.float()).sum()

        if reference_for_rank is None:
            raise RuntimeError("windowed reference output was not constructed")
        tolerance = 5e-2 if production_cuda else 1e-5
        torch.testing.assert_close(
            output.transpose(1, 2).float(),
            reference_for_rank,
            atol=tolerance,
            rtol=tolerance,
        )
        grad_output = reference_grads[rank]
        (output.transpose(1, 2) * grad_output).sum().backward()
        reference_loss.backward()
        torch.testing.assert_close(
            query.grad.transpose(1, 2),
            reference_queries[rank].grad,
            atol=tolerance,
            rtol=tolerance,
        )
        torch.testing.assert_close(
            local_key.grad.transpose(1, 2),
            reference_local_keys[rank].grad,
            atol=tolerance,
            rtol=tolerance,
        )
        torch.testing.assert_close(
            local_value.grad.transpose(1, 2),
            reference_local_values[rank].grad,
            atol=tolerance,
            rtol=tolerance,
        )
        expected_clean_key = _gather_bshd_intervals(
            reference_clean_key.grad.transpose(1, 2), intervals
        )
        torch.testing.assert_close(
            clean_key.grad,
            expected_clean_key,
            atol=tolerance,
            rtol=tolerance,
        )
        expected_clean_value = _gather_bshd_intervals(
            reference_clean_value.grad.transpose(1, 2), intervals
        )
        torch.testing.assert_close(
            clean_value.grad,
            expected_clean_value,
            atol=tolerance,
            rtol=tolerance,
        )
        queue.put((rank, "ok", ""))
    except Exception:  # pragma: no cover - child reports failure to parent.
        queue.put((rank, "error", traceback.format_exc()))
    finally:
        cp_attention._flex_shard_stats = original_shard_stats
        cp_attention._packed_clean_flex_shard_backward_exact_bshd = original_backward
        if dist.is_initialized():
            dist.destroy_process_group()


def test_ring_attention_matches_replicated_sdpa_forward_and_backward() -> None:
    if (
        not dist.is_available()
        or not torch.cuda.is_available()
        or torch.cuda.device_count() < 2
    ):
        pytest.skip("requires at least two CUDA devices")
    for world_size in (2, 3):
        if torch.cuda.device_count() < world_size:
            continue
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        with tempfile.NamedTemporaryFile(delete=False) as init:
            init_file = init.name
        try:
            processes = [
                ctx.Process(
                    target=_nccl_ring_attention_worker,
                    args=(rank, world_size, init_file, queue),
                )
                for rank in range(world_size)
            ]
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=30)
            for process in processes:
                assert process.exitcode == 0

            results = [queue.get(timeout=5) for _ in range(world_size)]
            errors = [message for _, status, message in results if status != "ok"]
            assert not errors, "\n".join(errors)
        finally:
            if os.path.exists(init_file):
                os.unlink(init_file)


def test_ring_attention_matches_replicated_sdpa_with_nccl_reduce_scatter() -> None:
    if (
        not dist.is_available()
        or not torch.cuda.is_available()
        or torch.cuda.device_count() < 2
    ):
        pytest.skip("requires at least two CUDA devices")
    world_size = 2
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    with tempfile.NamedTemporaryFile(delete=False) as init:
        init_file = init.name
    try:
        processes = [
            ctx.Process(
                target=_nccl_ring_attention_worker,
                args=(rank, world_size, init_file, queue),
            )
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=60)
        for process in processes:
            assert process.exitcode == 0

        results = [queue.get(timeout=5) for _ in range(world_size)]
        errors = [message for _, status, message in results if status != "ok"]
        assert not errors, "\n".join(errors)
    finally:
        if os.path.exists(init_file):
            os.unlink(init_file)


def test_hybrid_ring_attention_matches_replicated_sdpa_forward_and_backward() -> None:
    if (
        not dist.is_available()
        or not torch.cuda.is_available()
        or torch.cuda.device_count() < 2
    ):
        pytest.skip("requires at least two CUDA devices")
    world_size = 2
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    with tempfile.NamedTemporaryFile(delete=False) as init:
        init_file = init.name
    try:
        processes = [
            ctx.Process(
                target=_hybrid_ring_attention_worker,
                args=(rank, world_size, init_file, queue),
            )
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=30)
        for process in processes:
            assert process.exitcode == 0

        results = [queue.get(timeout=5) for _ in range(world_size)]
        errors = [message for _, status, message in results if status != "ok"]
        assert not errors, "\n".join(errors)
    finally:
        if os.path.exists(init_file):
            os.unlink(init_file)


@pytest.mark.parametrize(
    ("world_size", "clean_kv_transport"),
    [(2, "collective"), (2, "streaming"), (4, "collective"), (4, "streaming")],
)
def test_hybrid_sharded_attention_matches_replicated_sdpa_forward_and_backward(
    world_size: int,
    clean_kv_transport: str,
) -> None:
    if (
        not dist.is_available()
        or not torch.cuda.is_available()
        or torch.cuda.device_count() < world_size
    ):
        pytest.skip(f"requires {world_size} CUDA devices")
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    with tempfile.NamedTemporaryFile(delete=False) as init:
        init_file = init.name
    try:
        processes = [
            ctx.Process(
                target=_hybrid_sharded_collective_attention_worker,
                args=(rank, world_size, init_file, queue, 64, 8, 2, clean_kv_transport),
            )
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=300)
        for process in processes:
            assert process.exitcode == 0

        results = [queue.get(timeout=5) for _ in range(world_size)]
        errors = [message for _, status, message in results if status != "ok"]
        assert not errors, "\n".join(errors)
    finally:
        if os.path.exists(init_file):
            os.unlink(init_file)


def test_hybrid_sharded_qwen_d256_collective_matches_dense_backward() -> None:
    if (
        not dist.is_available()
        or not torch.cuda.is_available()
        or torch.cuda.device_count() < 2
    ):
        pytest.skip("requires two CUDA devices")
    world_size = 2
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    with tempfile.NamedTemporaryFile(delete=False) as init:
        init_file = init.name
    try:
        processes = [
            ctx.Process(
                target=_hybrid_sharded_collective_attention_worker,
                args=(rank, world_size, init_file, queue, 256, 12, 2, "collective"),
            )
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=300)
        for process in processes:
            assert process.exitcode == 0

        results = [queue.get(timeout=5) for _ in range(world_size)]
        errors = [message for _, status, message in results if status != "ok"]
        assert not errors, "\n".join(errors)
    finally:
        if os.path.exists(init_file):
            os.unlink(init_file)


@pytest.mark.parametrize(
    ("world_size", "block_parallel_size", "block_group_index"),
    ((2, 2, 0), (2, 4, 0), (2, 4, 1), (4, 4, 0), (8, 8, 0)),
)
def test_windowed_fused_collective_matches_dense_forward_and_backward(
    world_size: int,
    block_parallel_size: int,
    block_group_index: int,
) -> None:
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("requires torch.distributed with the Gloo backend")
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    with tempfile.NamedTemporaryFile(delete=False) as init:
        init_file = init.name
    try:
        processes = [
            ctx.Process(
                target=_windowed_fused_collective_worker,
                args=(
                    rank,
                    world_size,
                    init_file,
                    queue,
                    block_parallel_size,
                    block_group_index,
                ),
            )
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=300)
        for process in processes:
            assert process.exitcode == 0
        results = [queue.get(timeout=5) for _ in range(world_size)]
        errors = [message for _, status, message in results if status != "ok"]
        assert not errors, "\n".join(errors)
    finally:
        if os.path.exists(init_file):
            os.unlink(init_file)


def test_windowed_fused_collective_production_cuda_matches_dense_backward() -> None:
    if (
        not dist.is_available()
        or not dist.is_nccl_available()
        or not torch.cuda.is_available()
        or torch.cuda.device_count() < 2
    ):
        pytest.skip("requires at least two CUDA devices with NCCL")
    world_size = 2
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    with tempfile.NamedTemporaryFile(delete=False) as init:
        init_file = init.name
    try:
        processes = [
            ctx.Process(
                target=_windowed_fused_collective_worker,
                args=(rank, world_size, init_file, queue, world_size, 0, True),
            )
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=600)
        for process in processes:
            assert process.exitcode == 0
        results = [queue.get(timeout=5) for _ in range(world_size)]
        errors = [message for _, status, message in results if status != "ok"]
        assert not errors, "\n".join(errors)
    finally:
        if os.path.exists(init_file):
            os.unlink(init_file)


def test_hybrid_sharded_hd128_attention_matches_replicated_sdpa_forward_and_backward() -> (
    None
):
    if (
        not dist.is_available()
        or not torch.cuda.is_available()
        or torch.cuda.device_count() < 2
    ):
        pytest.skip("requires at least two CUDA devices")
    world_size = 2
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    with tempfile.NamedTemporaryFile(delete=False) as init:
        init_file = init.name
    try:
        processes = [
            ctx.Process(
                target=_hybrid_sharded_collective_attention_worker,
                args=(rank, world_size, init_file, queue, 128, 32, 8),
            )
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join()
        for process in processes:
            assert process.exitcode == 0

        results = [queue.get(timeout=5) for _ in range(world_size)]
        errors = [message for _, status, message in results if status != "ok"]
        assert not errors, "\n".join(errors)
    finally:
        if os.path.exists(init_file):
            os.unlink(init_file)


def test_hybrid_sharded_hd256_attention_matches_replicated_sdpa_forward_and_backward() -> (
    None
):
    if (
        not dist.is_available()
        or not torch.cuda.is_available()
        or torch.cuda.device_count() < 2
    ):
        pytest.skip("requires at least two CUDA devices")
    world_size = 2
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    with tempfile.NamedTemporaryFile(delete=False) as init:
        init_file = init.name
    try:
        processes = [
            ctx.Process(
                target=_hybrid_sharded_collective_attention_worker,
                args=(rank, world_size, init_file, queue, 256, 16, 8),
            )
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=300)
        for process in processes:
            assert process.exitcode == 0

        results = [queue.get(timeout=5) for _ in range(world_size)]
        errors = [message for _, status, message in results if status != "ok"]
        assert not errors, "\n".join(errors)
    finally:
        if os.path.exists(init_file):
            os.unlink(init_file)


@pytest.mark.parametrize("clean_kv_transport", ("collective", "streaming"))
def test_wide_head_hybrid_sharded_attention_matches_dense_forward_and_backward(
    clean_kv_transport: str,
) -> None:
    if (
        not dist.is_available()
        or not torch.cuda.is_available()
        or torch.cuda.device_count() < 2
    ):
        pytest.skip("requires at least two CUDA devices")
    world_size = 2
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    with tempfile.NamedTemporaryFile(delete=False) as init:
        init_file = init.name
    try:
        processes = [
            ctx.Process(
                target=_hybrid_sharded_collective_attention_worker,
                args=(
                    rank,
                    world_size,
                    init_file,
                    queue,
                    512,
                    16,
                    2,
                    clean_kv_transport,
                ),
            )
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=300)
        for process in processes:
            assert process.exitcode == 0
        results = [queue.get(timeout=5) for _ in range(world_size)]
        errors = [message for _, status, message in results if status != "ok"]
        assert not errors, "\n".join(errors)
    finally:
        if os.path.exists(init_file):
            os.unlink(init_file)


def test_phased_clean_backward_launches_reduction_before_remaining_compute() -> None:
    assert hasattr(cp_attention, "_run_phased_clean_local_backward_bshd")

    events: list[str] = []
    clean_grad_storage = torch.empty(1)
    clean_grad_key = torch.empty(1)
    clean_grad_value = torch.empty(1)
    clean_grad_query = torch.empty(1)
    reduced = torch.empty(1)
    local_gradients = (torch.empty(1), torch.empty(1), torch.empty(1))

    class Phases:
        @staticmethod
        def dkdv(*, grad_key, grad_value):
            assert grad_key is clean_grad_key
            assert grad_value is clean_grad_value
            events.append("dkdv")
            return grad_key, grad_value

        @staticmethod
        def dq():
            events.append("dq")
            return clean_grad_query

    class Work:
        @staticmethod
        def wait() -> None:
            events.append("wait")

    def launch_reduce(storage):
        assert storage is clean_grad_storage
        events.append("reduce")
        return Work(), reduced

    def run_local_backward():
        events.append("local")
        return local_gradients

    result = cp_attention._run_phased_clean_local_backward_bshd(
        phases=Phases(),
        clean_grad_storage=clean_grad_storage,
        clean_grad_key=clean_grad_key,
        clean_grad_value=clean_grad_value,
        launch_reduce=launch_reduce,
        run_local_backward=run_local_backward,
    )

    assert events == ["dkdv", "reduce", "dq", "local", "wait"]
    assert result == (clean_grad_query, local_gradients, reduced)
