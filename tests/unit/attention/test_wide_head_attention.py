# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import pytest
import torch

import dllm_parallel.core.attention.wide_head_attention as wide_head
from dllm_parallel.core.kernels import cp_fusion
from dllm_parallel.core.attention.wide_head_attention import (
    _bdlm_allowed,
    _compiled_create_block_mask,
    _interval_live_tiles,
    _interval_tile_envelopes,
    _interval_tile_worklists,
    is_wide_head_dim,
    wide_bdlm_attention_bshd,
    wide_bdlm_interval_backward_from_state_bhsd,
    wide_bdlm_interval_forward_bhsd,
    wide_bdlm_interval_plan,
    wide_full_attention_bshd,
)


def test_native_splitd_dispatch_is_limited_to_d512(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Tensor:
        is_cuda = True
        device = torch.device("cuda", 0)

        def __init__(self, head_dim: int) -> None:
            self.shape = (1, 1, 1, head_dim)

    monkeypatch.setattr(
        wide_head,
        "_device_supports_bdlm_splitd",
        lambda *_: True,
    )

    assert not wide_head._uses_native_bdlm_splitd(Tensor(128))
    assert wide_head._uses_native_bdlm_splitd(Tensor(512))


def test_wide_interval_backward_phases_use_one_prepared_native_state(
    monkeypatch,
) -> None:
    assert hasattr(wide_head, "prepare_wide_bdlm_interval_backward_bshd")

    calls: list[tuple[str, object]] = []
    native_state = object()
    grad_query = torch.empty(1)
    grad_key = torch.empty(1)
    grad_value = torch.empty(1)

    class FakeSplitD:
        @staticmethod
        def splitd_interval_backward_prepare(*args, **kwargs):
            calls.append(("prepare", kwargs))
            return native_state

        @staticmethod
        def splitd_backward_dkdv(state, *, grad_key=None, grad_value=None):
            assert state is native_state
            assert grad_key is not None
            assert grad_value is not None
            calls.append(("dkdv", state))
            grad_key.fill_(1.0)
            grad_value.fill_(2.0)
            return grad_key, grad_value

        @staticmethod
        def splitd_backward_dq(state, *, grad_query=None):
            assert state is native_state
            assert grad_query is not None
            calls.append(("dq", state))
            grad_query.fill_(3.0)
            return grad_query

    monkeypatch.setattr(wide_head, "_uses_native_bdlm_splitd", lambda _: True)
    monkeypatch.setattr(wide_head, "_require_bdlm_splitd", FakeSplitD)

    metadata = tuple(torch.empty(0) for _ in range(5))
    plan = wide_head.WideIntervalPlan(
        flex_plan=None,
        forward_query_tile_bounds=None,
        backward_query_tile_bounds=torch.empty(0),
        key_tile_bounds=torch.empty(0),
        forward_query_tile_offsets=None,
        forward_key_tile_work_items=None,
        backward_query_tile_offsets=None,
        backward_key_tile_work_items=None,
        key_tile_offsets=None,
        query_tile_work_items=None,
        uses_sparse_tile_worklist=False,
    )
    tensors = tuple(torch.empty(0) for _ in range(6))
    phases = wide_head.prepare_wide_bdlm_interval_backward_bshd(
        *tensors,
        None,
        *metadata,
        1.0,
        plan,
    )

    assert phases.dkdv(grad_key=grad_key, grad_value=grad_value) == (
        grad_key,
        grad_value,
    )
    assert phases.dq(grad_query=grad_query) is grad_query
    torch.testing.assert_close(grad_key, torch.ones_like(grad_key))
    torch.testing.assert_close(grad_value, torch.full_like(grad_value, 2.0))
    torch.testing.assert_close(grad_query, torch.full_like(grad_query, 3.0))
    assert [name for name, _ in calls] == ["prepare", "dkdv", "dq"]


def test_wide_block_mask_builder_is_compiled_and_cached(monkeypatch) -> None:
    sentinel = object()
    calls: list[tuple[object, bool, bool]] = []

    def compile_builder(builder: object, *, fullgraph: bool, dynamic: bool) -> object:
        calls.append((builder, fullgraph, dynamic))
        return sentinel

    monkeypatch.setattr(torch, "compile", compile_builder)
    _compiled_create_block_mask.cache_clear()
    try:
        assert _compiled_create_block_mask() is sentinel
        assert _compiled_create_block_mask() is sentinel
    finally:
        _compiled_create_block_mask.cache_clear()

    assert len(calls) == 1
    assert calls[0][1] is True
    assert calls[0][2] is False


@pytest.mark.parametrize("device", (
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="requires CUDA",
    )),
))
@pytest.mark.parametrize("plan_kind", ("block", "metadata", "interval"))
def test_wide_mask_counts_match_eager_across_block_grid_shapes(plan_kind, device) -> None:
    # Each case exercises its own shape sequence, not prior tests' compile cache.
    torch.compiler.reset()
    wide_head._compiled_create_block_mask.cache_clear()
    try:
        for query_len, key_len in ((256, 256), (80, 32), (32, 160), (80, 32)):
            blocks = torch.arange(query_len, device=device, dtype=torch.int32) // 16
            clean_queries = torch.zeros(query_len, device=device, dtype=torch.bool)
            key_positions = torch.arange(key_len, device=device, dtype=torch.int32)
            clean_keys = torch.ones(key_len, device=device, dtype=torch.bool)
            if plan_kind == "block":
                actual = wide_head._flex_plan(
                    blocks, clean_queries, 16, 0, query_len, key_len, 0, False,
                )
            elif plan_kind == "metadata":
                actual = wide_head._metadata_flex_plan(
                    blocks, blocks, clean_queries, key_positions // 16, clean_keys,
                    query_len=query_len, key_len=key_len,
                )
            else:
                bounds = torch.stack((torch.zeros_like(blocks), blocks * 16), dim=1)
                actual = wide_head._interval_flex_plan(
                    bounds, blocks, clean_queries, key_positions, clean_keys,
                    query_len=query_len, key_len=key_len,
                )
            expected = wide_head.create_block_mask(
                actual.mask_mod, B=None, H=None,
                Q_LEN=query_len, KV_LEN=key_len, device=device,
            )
            for name in (
                "kv_num_blocks", "full_kv_num_blocks", "q_num_blocks", "full_q_num_blocks",
            ):
                torch.testing.assert_close(getattr(actual, name), getattr(expected, name))
            torch.testing.assert_close(actual.to_dense(), expected.to_dense())
    finally:
        wide_head._compiled_create_block_mask.cache_clear()
        torch.compiler.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires native CP fusion")
def test_merge_backward_empty_shard_rows_are_exact_zero() -> None:
    device = torch.device("cuda")
    shard_output = torch.full(
        (1, 4, 8, 64),
        torch.nan,
        device=device,
        dtype=torch.bfloat16,
    )
    shard_lse = torch.full((1, 4, 8), -torch.inf, device=device)
    final_output = torch.randn_like(shard_output)
    final_lse = torch.randn_like(shard_lse)
    grad_output = torch.randn_like(shard_output)
    shard_grad_output = torch.empty_like(grad_output)
    grad_lse = torch.empty_like(shard_lse)

    cp_fusion.merge_backward_(
        shard_output,
        shard_lse,
        final_output,
        final_lse,
        grad_output,
        shard_grad_output,
        grad_lse,
    )

    torch.testing.assert_close(shard_grad_output, torch.zeros_like(shard_grad_output))
    torch.testing.assert_close(grad_lse, torch.zeros_like(grad_lse))


def _allowed_rows(
    query_blocks: torch.Tensor,
    query_is_clean: torch.Tensor,
    *,
    key_len: int,
    block_size: int,
    key_start: int,
    clean_offset: int,
    full_mask: bool = False,
) -> torch.Tensor:
    q_idx = torch.arange(query_blocks.numel(), device=query_blocks.device)[:, None]
    kv_idx = torch.arange(key_len, device=query_blocks.device)[None, :]
    return _bdlm_allowed(
        query_blocks,
        query_is_clean,
        q_idx,
        kv_idx,
        block_size=block_size,
        key_start=key_start,
        clean_offset=clean_offset,
        full_mask=full_mask,
    )


def test_wide_head_capability_is_explicit() -> None:
    assert not is_wide_head_dim(256)
    assert not is_wide_head_dim(320)
    assert is_wide_head_dim(512)
    assert not is_wide_head_dim(640)


def test_interval_tile_envelopes_never_drop_live_mask_entries() -> None:
    query_bounds = torch.tensor(
        ((0, 3), (2, 7), (8, 8), (1, 10), (5, 9)), dtype=torch.int32
    )
    local_query_blocks = torch.tensor((2, 3, 1, 0, 4), dtype=torch.int32)
    query_is_clean = torch.tensor((False, True, False, False, True))
    key_coordinates = torch.tensor((9, 2, 5, 3, 0, 4, 8, 1), dtype=torch.int32)
    key_is_clean = torch.tensor((True, False, True, False, True, False, True, True))
    query_tile_bounds, key_tile_bounds = _interval_tile_envelopes(
        query_bounds,
        local_query_blocks,
        query_is_clean,
        key_coordinates,
        key_is_clean,
        query_rows_per_tile=2,
        key_rows_per_tile=2,
    )
    allowed = (
        (~query_is_clean[:, None])
        & (~key_is_clean[None, :])
        & (local_query_blocks[:, None] == key_coordinates[None, :])
    ) | (
        key_is_clean[None, :]
        & (key_coordinates[None, :] >= query_bounds[:, :1])
        & (key_coordinates[None, :] < query_bounds[:, 1:])
    )
    for query_row, key_row in allowed.nonzero().tolist():
        query_tile = query_row // 2
        key_tile = key_row // 2
        assert query_tile_bounds[query_tile, 0] <= key_tile
        assert key_tile < query_tile_bounds[query_tile, 1]
        assert key_tile_bounds[key_tile, 0] <= query_tile
        assert query_tile < key_tile_bounds[key_tile, 1]


def test_interval_tile_worklists_preserve_candidate_tile_support() -> None:
    query_bounds = torch.tensor(
        ((0, 3), (2, 7), (8, 8), (1, 10), (5, 9)), dtype=torch.int32
    )
    local_query_blocks = torch.tensor((2, 3, 1, 0, 4), dtype=torch.int32)
    query_is_clean = torch.tensor((False, True, False, False, True))
    key_coordinates = torch.tensor((9, 2, 5, 3, 0, 4, 8, 1), dtype=torch.int32)
    key_is_clean = torch.tensor((True, False, True, False, True, False, True, True))
    live_tiles = _interval_live_tiles(
        query_bounds,
        local_query_blocks,
        query_is_clean,
        key_coordinates,
        key_is_clean,
        query_rows_per_tile=2,
        key_rows_per_tile=2,
    )
    query_offsets, key_indices, key_offsets, query_indices = (
        _interval_tile_worklists(
            query_bounds,
            local_query_blocks,
            query_is_clean,
            key_coordinates,
            key_is_clean,
            query_rows_per_tile=2,
            key_rows_per_tile=2,
        )
    )

    rebuilt = torch.zeros_like(live_tiles)
    for query_tile in range(int(live_tiles.shape[0])):
        start = int(query_offsets[query_tile])
        stop = int(query_offsets[query_tile + 1])
        rebuilt[query_tile, (key_indices[start:stop] // 2).long()] = True
    torch.testing.assert_close(rebuilt, live_tiles)

    rebuilt_transpose = torch.zeros_like(live_tiles.transpose(0, 1))
    for key_tile in range(int(live_tiles.shape[1])):
        start = int(key_offsets[key_tile])
        stop = int(key_offsets[key_tile + 1])
        rebuilt_transpose[key_tile, (query_indices[start:stop] // 2).long()] = True
    torch.testing.assert_close(rebuilt_transpose, live_tiles.transpose(0, 1))

    allowed = (
        (~query_is_clean[:, None])
        & (~key_is_clean[None, :])
        & (local_query_blocks[:, None] == key_coordinates[None, :])
    ) | (
        key_is_clean[None, :]
        & (key_coordinates[None, :] >= query_bounds[:, :1])
        & (key_coordinates[None, :] < query_bounds[:, 1:])
    )
    classified_full = torch.zeros_like(live_tiles)
    for query_tile in range(int(live_tiles.shape[0])):
        start = int(query_offsets[query_tile])
        stop = int(query_offsets[query_tile + 1])
        for work_item in key_indices[start:stop].tolist():
            if work_item % 2:
                key_tile = work_item // 2
                classified_full[query_tile, key_tile] = True
                tile = allowed[
                    query_tile * 2 : (query_tile + 1) * 2,
                    key_tile * 2 : (key_tile + 1) * 2,
                ]
                assert tile.shape == (2, 2)
                assert tile.all()

    expected_full = torch.zeros_like(live_tiles)
    for query_tile in range(int(live_tiles.shape[0])):
        for key_tile in range(int(live_tiles.shape[1])):
            tile = allowed[
                query_tile * 2 : (query_tile + 1) * 2,
                key_tile * 2 : (key_tile + 1) * 2,
            ]
            expected_full[query_tile, key_tile] = (
                tile.shape == (2, 2) and bool(tile.all())
            )
    torch.testing.assert_close(classified_full, expected_full)


@pytest.mark.parametrize("query_len,key_len", ((0, 8), (8, 0), (0, 0)))
def test_interval_tile_worklists_handle_empty_dimensions(
    query_len: int,
    key_len: int,
) -> None:
    query_bounds = torch.zeros(query_len, 2, dtype=torch.int32)
    local_query_blocks = torch.zeros(query_len, dtype=torch.int32)
    query_is_clean = torch.zeros(query_len, dtype=torch.bool)
    key_coordinates = torch.zeros(key_len, dtype=torch.int32)
    key_is_clean = torch.ones(key_len, dtype=torch.bool)
    query_offsets, key_indices, key_offsets, query_indices = (
        _interval_tile_worklists(
            query_bounds,
            local_query_blocks,
            query_is_clean,
            key_coordinates,
            key_is_clean,
            query_rows_per_tile=2,
            key_rows_per_tile=2,
        )
    )
    assert query_offsets.shape == ((query_len + 1) // 2 + 1,)
    assert key_offsets.shape == ((key_len + 1) // 2 + 1,)
    assert key_indices.numel() == 0
    assert query_indices.numel() == 0


def test_wide_head_mask_rejects_encoded_coordinates() -> None:
    with pytest.raises(ValueError, match="nonnegative logical offset"):
        _allowed_rows(
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([False]),
            key_len=1,
            block_size=1,
            key_start=-1,
            clean_offset=0,
        )


@pytest.mark.parametrize("key_start", (0, 3, 8))
def test_global_prefix_mask_matches_bdlm_semantics(key_start: int) -> None:
    block_size = 4
    key_len = 9
    query_blocks = (0, 1, 2, 3, 0, 1, 2, 3)
    query_is_clean = (False, False, False, False, True, True, True, True)
    actual = _allowed_rows(
        torch.tensor(query_blocks),
        torch.tensor(query_is_clean),
        key_len=key_len,
        block_size=block_size,
        key_start=key_start,
        clean_offset=0,
    )
    logical_keys = key_start + torch.arange(key_len)
    expected = torch.stack(
        [
            logical_keys < (block + int(clean)) * block_size
            for block, clean in zip(query_blocks, query_is_clean)
        ]
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("logical_key_start", (0, 5, 16, 21))
def test_full_mask_matches_bdlm_semantics(logical_key_start: int) -> None:
    block_size = 4
    clean_offset = 16
    key_len = 11
    query_blocks = (0, 1, 2, 3, 0, 1, 2, 3)
    query_is_clean = (False, False, False, False, True, True, True, True)
    actual = _allowed_rows(
        torch.tensor(query_blocks),
        torch.tensor(query_is_clean),
        key_len=key_len,
        block_size=block_size,
        key_start=logical_key_start,
        clean_offset=clean_offset,
        full_mask=True,
    )
    logical_keys = logical_key_start + torch.arange(key_len)
    keys_are_clean = logical_keys >= clean_offset
    key_positions = torch.where(keys_are_clean, logical_keys - clean_offset, logical_keys)
    key_blocks = key_positions // block_size
    expected = torch.stack(
        [
            (
                keys_are_clean & (key_blocks <= block)
                if clean
                else ((~keys_are_clean & (key_blocks == block)) | (keys_are_clean & (key_blocks < block)))
            )
            for block, clean in zip(query_blocks, query_is_clean)
        ]
    )
    torch.testing.assert_close(actual, expected)


def _dense_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    allowed: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    repeats = query.shape[2] // key.shape[2]
    expanded_key = key.float().repeat_interleave(repeats, dim=2)
    expanded_value = value.float().repeat_interleave(repeats, dim=2)
    scores = torch.einsum("bqhd,bkhd->bhqk", query.float(), expanded_key) * scale
    scores = scores.masked_fill(~allowed[None, None], -torch.inf)
    lse = torch.logsumexp(scores, dim=-1)
    probabilities = torch.softmax(scores, dim=-1)
    probabilities = torch.where(
        torch.isfinite(lse).unsqueeze(-1), probabilities, torch.zeros_like(probabilities)
    )
    output = torch.einsum("bhqk,bkhd->bqhd", probabilities, expanded_value)
    return output, lse


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires native wide-head kernels")
@pytest.mark.parametrize("shared_kv", (False, True))
def test_wide_full_attention_output_lse_and_backward_match_dense(
    shared_kv: bool,
) -> None:
    torch.manual_seed(71)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    scale = 512.0**-0.5
    query = torch.randn(1, 48, 16, 512, device=device, dtype=dtype, requires_grad=True)
    key = torch.randn(1, 64, 2, 512, device=device, dtype=dtype, requires_grad=True)
    value = key if shared_kv else torch.randn_like(key, requires_grad=True)
    query_ref = query.detach().clone().requires_grad_(True)
    key_ref = key.detach().clone().requires_grad_(True)
    value_ref = key_ref if shared_kv else value.detach().clone().requires_grad_(True)
    grad_output = torch.randn_like(query)
    grad_lse = torch.randn(1, 16, 48, device=device, dtype=torch.float32)

    output, lse = wide_full_attention_bshd(query, key, value, scale)
    expected, expected_lse = _dense_attention(
        query_ref,
        key_ref,
        value_ref,
        torch.ones(48, 64, dtype=torch.bool, device=device),
        scale,
    )
    torch.autograd.backward((output, lse), (grad_output, grad_lse))
    torch.autograd.backward((expected, expected_lse), (grad_output.float(), grad_lse))

    torch.testing.assert_close(output.float(), expected, atol=7e-2, rtol=7e-2)
    torch.testing.assert_close(lse, expected_lse, atol=7e-2, rtol=7e-2)
    torch.testing.assert_close(query.grad.float(), query_ref.grad.float(), atol=1e-1, rtol=1e-1)
    torch.testing.assert_close(key.grad.float(), key_ref.grad.float(), atol=1e-1, rtol=1e-1)
    if not shared_kv:
        torch.testing.assert_close(
            value.grad.float(), value_ref.grad.float(), atol=1e-1, rtol=1e-1
        )


@pytest.mark.parametrize("full_mask", (False, True))
@pytest.mark.parametrize("shared_kv", (False, True))
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires native wide-head kernels")
def test_wide_bdlm_attention_and_backward_match_dense(
    full_mask: bool,
    shared_kv: bool,
) -> None:
    torch.manual_seed(79 + int(full_mask))
    device = torch.device("cuda")
    dtype = torch.bfloat16
    block_size = 32
    clean_offset = 128 if full_mask else 0
    logical_key_start = 96 if full_mask else 32
    key_start = logical_key_start
    query_blocks = torch.tensor(
        [2] * 32 + [4] * 32 + [2] * 32 + [5] * 32,
        device=device,
        dtype=torch.int32,
    )
    query_is_clean = torch.tensor([False] * 64 + [True] * 64, device=device)
    query = torch.randn(1, 128, 16, 512, device=device, dtype=dtype, requires_grad=True)
    key = torch.randn(1, 128, 2, 512, device=device, dtype=dtype, requires_grad=True)
    value = key if shared_kv else torch.randn_like(key, requires_grad=True)
    query_ref = query.detach().clone().requires_grad_(True)
    key_ref = key.detach().clone().requires_grad_(True)
    value_ref = key_ref if shared_kv else value.detach().clone().requires_grad_(True)
    scale = 512.0**-0.5

    allowed = _allowed_rows(
        query_blocks,
        query_is_clean,
        key_len=key.shape[1],
        block_size=block_size,
        key_start=key_start,
        clean_offset=clean_offset,
        full_mask=full_mask,
    )
    output, lse = wide_bdlm_attention_bshd(
        query,
        key,
        value,
        query_blocks,
        query_is_clean,
        block_size,
        key_start,
        scale,
        clean_offset,
        full_mask=full_mask,
    )
    expected, expected_lse = _dense_attention(
        query_ref, key_ref, value_ref, allowed, scale
    )
    grad_output = torch.randn_like(query)
    grad_lse = torch.randn_like(lse)
    torch.autograd.backward((output, lse), (grad_output, grad_lse))
    torch.autograd.backward((expected, expected_lse), (grad_output.float(), grad_lse))

    torch.testing.assert_close(output.float(), expected, atol=8e-2, rtol=8e-2)
    finite = torch.isfinite(expected_lse)
    torch.testing.assert_close(lse[finite], expected_lse[finite], atol=8e-2, rtol=8e-2)
    torch.testing.assert_close(query.grad.float(), query_ref.grad.float(), atol=1.2e-1, rtol=1.2e-1)
    torch.testing.assert_close(key.grad.float(), key_ref.grad.float(), atol=1.2e-1, rtol=1.2e-1)
    if not shared_kv:
        torch.testing.assert_close(
            value.grad.float(), value_ref.grad.float(), atol=1.2e-1, rtol=1.2e-1
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires native Split-D")
@pytest.mark.parametrize("scale", (512.0**-0.5, 1.0))
@pytest.mark.parametrize("sparse_tile_worklist", (False, True))
def test_wide_interval_shard_backward_reuses_merged_state(
    scale: float,
    sparse_tile_worklist: bool,
) -> None:
    torch.manual_seed(97)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    query_len, key_len = 64, 128
    query_bshd = torch.randn(1, query_len, 4, 512, device=device, dtype=dtype)
    key_bshd = torch.randn(1, key_len, 2, 512, device=device, dtype=dtype)
    value_bshd = torch.randn_like(key_bshd)
    query = query_bshd.transpose(1, 2)
    key = key_bshd.transpose(1, 2)
    value = value_bshd.transpose(1, 2)
    active_query_len = query_len // 2
    active_key_len = key_len // 4
    clean_starts = torch.arange(query_len, device=device, dtype=torch.int32) // 4
    clean_stops = clean_starts + 65
    query_bounds = torch.stack((clean_starts, clean_stops), dim=-1).contiguous()
    local_query_blocks = torch.arange(
        query_len, device=device, dtype=torch.int32
    ) % 4
    query_is_clean = torch.arange(query_len, device=device) >= active_query_len
    key_is_clean = torch.arange(key_len, device=device) >= active_key_len
    key_coordinates = torch.cat(
        (
            torch.arange(active_key_len, device=device, dtype=torch.int32) % 4,
            torch.arange(
                key_len - active_key_len, device=device, dtype=torch.int32
            ),
        )
    )

    plan = wide_bdlm_interval_plan(
        query_bounds,
        local_query_blocks,
        query_is_clean,
        key_coordinates,
        key_is_clean,
        query_len=query_len,
        key_len=key_len,
        query_heads=int(query.shape[1]),
        key_heads=int(key.shape[1]),
        sparse_tile_worklist=sparse_tile_worklist,
    )
    output, lse = wide_bdlm_interval_forward_bhsd(
        query,
        key,
        value,
        query_bounds,
        local_query_blocks,
        query_is_clean,
        key_coordinates,
        key_is_clean,
        scale,
        plan,
    )
    grad_output = torch.randn_like(output)
    grad_lse = torch.randn_like(lse)
    shard_gradients = []
    for start, stop in ((0, key_len // 2), (key_len // 2, key_len)):
        shard_plan = wide_bdlm_interval_plan(
            query_bounds,
            local_query_blocks,
            query_is_clean,
            key_coordinates[start:stop].contiguous(),
            key_is_clean[start:stop].contiguous(),
            query_len=query_len,
            key_len=stop - start,
            query_heads=int(query.shape[1]),
            key_heads=int(key.shape[1]),
            sparse_tile_worklist=sparse_tile_worklist,
        )
        shard_gradients.append(
            wide_bdlm_interval_backward_from_state_bhsd(
                query,
                key[:, :, start:stop],
                value[:, :, start:stop],
                output,
                lse,
                grad_output,
                grad_lse,
                query_bounds,
                local_query_blocks,
                query_is_clean,
                key_coordinates[start:stop].contiguous(),
                key_is_clean[start:stop].contiguous(),
                scale,
                shard_plan,
            )
        )
    actual_dq = shard_gradients[0][0] + shard_gradients[1][0]
    actual_dk = torch.cat((shard_gradients[0][1], shard_gradients[1][1]), dim=2)
    actual_dv = torch.cat((shard_gradients[0][2], shard_gradients[1][2]), dim=2)

    query_ref = query_bshd.detach().float().requires_grad_(True)
    key_ref = key_bshd.detach().float().requires_grad_(True)
    value_ref = value_bshd.detach().float().requires_grad_(True)
    allowed = (
        (~query_is_clean[:, None])
        & (~key_is_clean[None, :])
        & (local_query_blocks[:, None] == key_coordinates[None, :])
    ) | (
        key_is_clean[None, :]
        & (key_coordinates[None, :] >= clean_starts[:, None])
        & (key_coordinates[None, :] < clean_stops[:, None])
    )
    expected, expected_lse = _dense_attention(
        query_ref,
        key_ref,
        value_ref,
        allowed,
        scale,
    )
    torch.autograd.backward(
        (expected, expected_lse),
        (grad_output.transpose(1, 2).float(), grad_lse),
    )

    torch.testing.assert_close(
        output.transpose(1, 2).float(), expected, atol=8e-2, rtol=8e-2
    )
    torch.testing.assert_close(lse, expected_lse, atol=8e-2, rtol=8e-2)
    gradient_atol = 3e-1 if scale == 1.0 else 1.2e-1
    gradient_rtol = 1.5e-1 if scale == 1.0 else 1.2e-1
    torch.testing.assert_close(
        actual_dq.transpose(1, 2).float(),
        query_ref.grad,
        atol=gradient_atol,
        rtol=gradient_rtol,
    )
    torch.testing.assert_close(
        actual_dk.transpose(1, 2).float(),
        key_ref.grad,
        atol=gradient_atol,
        rtol=gradient_rtol,
    )
    torch.testing.assert_close(
        actual_dv.transpose(1, 2).float(),
        value_ref.grad,
        atol=gradient_atol,
        rtol=gradient_rtol,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires native Split-D")
def test_sparse_interval_worklist_matches_bounded_schedule() -> None:
    """Skipping an interior fully masked tile preserves forward and backward."""

    torch.manual_seed(98)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    query_len, key_len = 64, 192
    query = torch.randn(1, 4, query_len, 512, device=device, dtype=dtype)
    key = torch.randn(1, 2, key_len, 512, device=device, dtype=dtype)
    value = torch.randn_like(key)
    query_bounds = torch.tensor(
        (0, 64), device=device, dtype=torch.int32
    ).expand(query_len, 2).contiguous()
    local_query_blocks = torch.zeros(query_len, device=device, dtype=torch.int32)
    query_is_clean = torch.zeros(query_len, device=device, dtype=torch.bool)
    key_coordinates = torch.cat(
        (
            torch.zeros(64, device=device, dtype=torch.int32),
            torch.ones(64, device=device, dtype=torch.int32),
            torch.arange(64, device=device, dtype=torch.int32),
        )
    )
    key_is_clean = torch.cat(
        (
            torch.zeros(128, device=device, dtype=torch.bool),
            torch.ones(64, device=device, dtype=torch.bool),
        )
    )
    metadata = (
        query_bounds,
        local_query_blocks,
        query_is_clean,
        key_coordinates,
        key_is_clean,
    )
    bounded_plan = wide_bdlm_interval_plan(
        *metadata,
        query_len=query_len,
        key_len=key_len,
        query_heads=4,
        key_heads=2,
    )
    sparse_plan = wide_bdlm_interval_plan(
        *metadata,
        query_len=query_len,
        key_len=key_len,
        query_heads=4,
        key_heads=2,
        sparse_tile_worklist=True,
    )
    bounded_output, bounded_lse = wide_bdlm_interval_forward_bhsd(
        query, key, value, *metadata, 1.0, bounded_plan
    )
    sparse_output, sparse_lse = wide_bdlm_interval_forward_bhsd(
        query, key, value, *metadata, 1.0, sparse_plan
    )
    grad_output = torch.randn_like(bounded_output)
    grad_lse = torch.randn_like(bounded_lse)
    bounded_gradients = wide_bdlm_interval_backward_from_state_bhsd(
        query,
        key,
        value,
        bounded_output,
        bounded_lse,
        grad_output,
        grad_lse,
        *metadata,
        1.0,
        bounded_plan,
    )
    sparse_gradients = wide_bdlm_interval_backward_from_state_bhsd(
        query,
        key,
        value,
        sparse_output,
        sparse_lse,
        grad_output,
        grad_lse,
        *metadata,
        1.0,
        sparse_plan,
    )

    torch.testing.assert_close(sparse_output, bounded_output, atol=0, rtol=0)
    torch.testing.assert_close(sparse_lse, bounded_lse, atol=0, rtol=0)
    for sparse_gradient, bounded_gradient in zip(
        sparse_gradients, bounded_gradients, strict=True
    ):
        torch.testing.assert_close(
            sparse_gradient, bounded_gradient, atol=0, rtol=0
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires native Split-D")
def test_wide_interval_backward_zeros_unreachable_key_tiles() -> None:
    """Bounded dK/dV scheduling must leave fully masked key tiles at zero."""

    torch.manual_seed(99)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    query_len, key_len = 64, 128
    query = torch.randn(1, 16, query_len, 512, device=device, dtype=dtype)
    key = torch.randn(1, 2, key_len, 512, device=device, dtype=dtype)
    value = torch.randn_like(key)
    query_bounds = torch.zeros(query_len, 2, device=device, dtype=torch.int32)
    local_query_blocks = torch.zeros(query_len, device=device, dtype=torch.int32)
    query_is_clean = torch.zeros(query_len, device=device, dtype=torch.bool)
    key_coordinates = torch.cat(
        (
            torch.zeros(key_len // 2, device=device, dtype=torch.int32),
            torch.ones(key_len // 2, device=device, dtype=torch.int32),
        )
    )
    key_is_clean = torch.zeros(key_len, device=device, dtype=torch.bool)
    metadata = (
        query_bounds,
        local_query_blocks,
        query_is_clean,
        key_coordinates,
        key_is_clean,
    )
    plan = wide_bdlm_interval_plan(
        *metadata,
        query_len=query_len,
        key_len=key_len,
        query_heads=16,
        key_heads=2,
    )
    torch.testing.assert_close(
        plan.key_tile_bounds[-1],
        torch.zeros(2, device=device, dtype=torch.int32),
        atol=0,
        rtol=0,
    )
    output, lse = wide_bdlm_interval_forward_bhsd(
        query,
        key,
        value,
        *metadata,
        1.0,
        plan,
    )
    gradients = wide_bdlm_interval_backward_from_state_bhsd(
        query,
        key,
        value,
        output,
        lse,
        torch.randn_like(output),
        None,
        *metadata,
        1.0,
        plan,
    )

    for gradient in gradients:
        assert torch.isfinite(gradient).all()
    torch.testing.assert_close(
        gradients[1][:, :, key_len // 2 :],
        torch.zeros_like(gradients[1][:, :, key_len // 2 :]),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        gradients[2][:, :, key_len // 2 :],
        torch.zeros_like(gradients[2][:, :, key_len // 2 :]),
        atol=0,
        rtol=0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires native Split-D")
def test_wide_interval_backward_zeros_unreachable_query_tiles() -> None:
    """Bounded dQ scheduling must leave fully masked query tiles at zero."""

    torch.manual_seed(100)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    query_len, key_len = 128, 64
    query = torch.randn(1, 16, query_len, 512, device=device, dtype=dtype)
    key = torch.randn(1, 2, key_len, 512, device=device, dtype=dtype)
    value = torch.randn_like(key)
    query_bounds = torch.zeros(query_len, 2, device=device, dtype=torch.int32)
    local_query_blocks = torch.cat(
        (
            torch.ones(query_len // 2, device=device, dtype=torch.int32),
            torch.zeros(query_len // 2, device=device, dtype=torch.int32),
        )
    )
    query_is_clean = torch.zeros(query_len, device=device, dtype=torch.bool)
    key_coordinates = torch.zeros(key_len, device=device, dtype=torch.int32)
    key_is_clean = torch.zeros(key_len, device=device, dtype=torch.bool)
    metadata = (
        query_bounds,
        local_query_blocks,
        query_is_clean,
        key_coordinates,
        key_is_clean,
    )
    plan = wide_bdlm_interval_plan(
        *metadata,
        query_len=query_len,
        key_len=key_len,
        query_heads=16,
        key_heads=2,
    )
    torch.testing.assert_close(
        plan.backward_query_tile_bounds[0],
        torch.zeros(2, device=device, dtype=torch.int32),
        atol=0,
        rtol=0,
    )
    output, lse = wide_bdlm_interval_forward_bhsd(
        query,
        key,
        value,
        *metadata,
        1.0,
        plan,
    )
    gradients = wide_bdlm_interval_backward_from_state_bhsd(
        query,
        key,
        value,
        output,
        lse,
        torch.randn_like(output),
        None,
        *metadata,
        1.0,
        plan,
    )

    for gradient in gradients:
        assert torch.isfinite(gradient).all()
    torch.testing.assert_close(
        gradients[0][:, :, : query_len // 2],
        torch.zeros_like(gradients[0][:, :, : query_len // 2]),
        atol=0,
        rtol=0,
    )
    assert torch.count_nonzero(gradients[0][:, :, query_len // 2 :]) > 0


def _full_clean_interval_metadata(
    *,
    query_len: int,
    key_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    query_bounds = torch.empty(query_len, 2, device=device, dtype=torch.int32)
    query_bounds[:, 0] = 0
    query_bounds[:, 1] = key_len
    return (
        query_bounds,
        torch.zeros(query_len, device=device, dtype=torch.int32),
        torch.zeros(query_len, device=device, dtype=torch.bool),
        torch.arange(key_len, device=device, dtype=torch.int32),
        torch.ones(key_len, device=device, dtype=torch.bool),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires native Split-D")
def test_diffusiongemma_global_attention_32k_backward_matches_dense() -> None:
    """Validate scale-1 D=512 numerics at DiffusionGemma's production key length."""

    torch.manual_seed(101)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    query_len, key_len = 64, 32 * 1024
    query = torch.randn(1, 16, query_len, 512, device=device, dtype=dtype)
    key = torch.randn(1, 2, key_len, 512, device=device, dtype=dtype)
    value = torch.randn_like(key)
    metadata = _full_clean_interval_metadata(
        query_len=query_len,
        key_len=key_len,
        device=device,
    )
    plan = wide_bdlm_interval_plan(
        *metadata,
        query_len=query_len,
        key_len=key_len,
        query_heads=16,
        key_heads=2,
    )
    output, lse = wide_bdlm_interval_forward_bhsd(
        query,
        key,
        value,
        *metadata,
        1.0,
        plan,
    )
    grad_output = torch.randn_like(output)
    grad_lse = torch.randn_like(lse)
    actual_dq, actual_dk, actual_dv = wide_bdlm_interval_backward_from_state_bhsd(
        query,
        key,
        value,
        output,
        lse,
        grad_output,
        grad_lse,
        *metadata,
        1.0,
        plan,
    )

    query_ref = query.transpose(1, 2).detach().float().requires_grad_(True)
    key_ref = key.transpose(1, 2).detach().float().requires_grad_(True)
    value_ref = value.transpose(1, 2).detach().float().requires_grad_(True)
    expected, expected_lse = _dense_attention(
        query_ref,
        key_ref,
        value_ref,
        torch.ones(query_len, key_len, device=device, dtype=torch.bool),
        1.0,
    )
    torch.autograd.backward(
        (expected, expected_lse),
        (grad_output.transpose(1, 2).float(), grad_lse),
    )

    torch.testing.assert_close(
        output.transpose(1, 2).float(), expected, atol=8e-2, rtol=8e-2
    )
    torch.testing.assert_close(lse, expected_lse, atol=8e-2, rtol=8e-2)
    torch.testing.assert_close(
        actual_dq.transpose(1, 2).float(), query_ref.grad, atol=1.2e-1, rtol=1.2e-1
    )
    torch.testing.assert_close(
        actual_dk.transpose(1, 2).float(), key_ref.grad, atol=1.2e-1, rtol=1.2e-1
    )
    torch.testing.assert_close(
        actual_dv.transpose(1, 2).float(), value_ref.grad, atol=1.2e-1, rtol=1.2e-1
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires native Split-D")
def test_diffusiongemma_global_attention_production_geometry_is_finite() -> None:
    """Stress the native backward at one CP rank's 32K global-layer geometry."""

    torch.manual_seed(103)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    query_len, key_len = 8 * 1024, 32 * 1024
    query = torch.randn(1, 16, query_len, 512, device=device, dtype=dtype)
    key = torch.randn(1, 2, key_len, 512, device=device, dtype=dtype)
    value = torch.randn_like(key)
    metadata = _full_clean_interval_metadata(
        query_len=query_len,
        key_len=key_len,
        device=device,
    )
    plan = wide_bdlm_interval_plan(
        *metadata,
        query_len=query_len,
        key_len=key_len,
        query_heads=16,
        key_heads=2,
        sparse_tile_worklist=True,
    )
    output, lse = wide_bdlm_interval_forward_bhsd(
        query,
        key,
        value,
        *metadata,
        1.0,
        plan,
    )
    gradients = wide_bdlm_interval_backward_from_state_bhsd(
        query,
        key,
        value,
        output,
        lse,
        torch.randn_like(output),
        torch.randn_like(lse),
        *metadata,
        1.0,
        plan,
    )

    for name, tensor in zip(("output", "lse", "dq", "dk", "dv"), (output, lse, *gradients)):
        assert torch.isfinite(tensor).all(), f"nonfinite {name}"
