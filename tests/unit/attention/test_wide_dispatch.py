# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import torch

import dllm_parallel.core.attention.context_parallel_attention as cp_attention
from dllm_parallel.core.attention.context_parallel_attention import (
    BlockDenoisingFullMask,
)


def test_d256_q32_forward_dispatch_uses_measured_h100_crossover() -> None:
    query = torch.zeros((1, 16, 32, 256), dtype=torch.bfloat16)
    short_key = torch.zeros((1, 8, 544, 256), dtype=torch.bfloat16)
    long_key = torch.zeros((1, 8, 1056, 256), dtype=torch.bfloat16)
    d128_query = torch.zeros((1, 32, 32, 128), dtype=torch.bfloat16)
    d128_key = torch.zeros((1, 8, 4096, 128), dtype=torch.bfloat16)

    assert cp_attention._should_use_native_fa4_forward(
        query=query,
        key=short_key,
        requested=True,
    )
    assert not cp_attention._should_use_native_fa4_forward(
        query=query,
        key=long_key,
        requested=True,
    )
    assert cp_attention._should_use_native_fa4_forward(
        query=d128_query,
        key=d128_key,
        requested=True,
    )
    assert not cp_attention._should_use_native_fa4_forward(
        query=query,
        key=short_key,
        requested=False,
    )


def test_wide_masked_cp_shard_uses_metadata_dispatch(monkeypatch) -> None:
    query = torch.zeros((1, 2, 4, 512), dtype=torch.bfloat16)
    key = torch.zeros((1, 1, 8, 512), dtype=torch.bfloat16)
    value = torch.zeros_like(key)
    mask = BlockDenoisingFullMask(
        query_blocks=torch.tensor([0, 1, 0, 1], dtype=torch.int32),
        query_is_clean=torch.tensor([False, False, True, True]),
        block_size=2,
        clean_offset=4,
    )
    observed: dict[str, torch.Tensor] = {}

    def metadata_attention(
        query_bhsd: torch.Tensor,
        key_bhsd: torch.Tensor,
        value_bhsd: torch.Tensor,
        query_blocks: torch.Tensor,
        local_query_blocks: torch.Tensor,
        query_is_clean: torch.Tensor,
        key_blocks: torch.Tensor,
        key_is_clean: torch.Tensor,
        scale: float,
        plan: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del key_bhsd, value_bhsd, local_query_blocks, scale
        observed["query"] = query_bhsd
        observed["query_blocks"] = query_blocks
        observed["query_is_clean"] = query_is_clean
        observed["key_blocks"] = key_blocks
        observed["key_is_clean"] = key_is_clean
        observed["plan"] = plan
        return query_bhsd.clone(), torch.zeros((1, 2, 4), dtype=torch.float32)

    monkeypatch.setattr(
        cp_attention,
        "_wide_metadata_forward_bhsd",
        metadata_attention,
    )
    output, lse = cp_attention._wide_shard_stats(
        query=query,
        key=key,
        value=value,
        attn_mask=mask,
        is_causal=False,
        scale=0.125,
        key_start=0,
    )

    assert output.shape == query.shape
    assert observed["query"] is query
    assert lse.shape == (1, 2, 4)
    torch.testing.assert_close(observed["query_blocks"], mask.query_blocks)
    torch.testing.assert_close(observed["query_is_clean"], mask.query_is_clean)
    assert observed["key_blocks"].shape == (8,)
    assert observed["key_is_clean"].shape == (8,)
    assert observed["plan"] is not None


def test_wide_metadata_plan_is_cached_on_mask(monkeypatch) -> None:
    query = torch.zeros((1, 2, 4, 512), dtype=torch.bfloat16)
    key = torch.zeros((1, 1, 8, 512), dtype=torch.bfloat16)
    value = torch.zeros_like(key)
    mask = BlockDenoisingFullMask(
        query_blocks=torch.tensor([0, 1, 0, 1], dtype=torch.int32),
        query_is_clean=torch.tensor([False, False, True, True]),
        block_size=2,
        clean_offset=4,
    )
    calls = 0

    def build_plan(*args, **kwargs) -> object:
        nonlocal calls
        del args, kwargs
        calls += 1
        return object()

    def metadata_attention(
        query_bhsd: torch.Tensor,
        key_bhsd: torch.Tensor,
        value_bhsd: torch.Tensor,
        query_blocks: torch.Tensor,
        local_query_blocks: torch.Tensor,
        query_is_clean: torch.Tensor,
        key_blocks: torch.Tensor,
        key_is_clean: torch.Tensor,
        scale: float,
        plan: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del (
            key_bhsd,
            value_bhsd,
            query_blocks,
            local_query_blocks,
            query_is_clean,
            key_blocks,
            key_is_clean,
            scale,
            plan,
        )
        return query_bhsd.clone(), torch.zeros((1, 2, 4), dtype=torch.float32)

    monkeypatch.setattr(cp_attention, "_build_wide_metadata_plan", build_plan)
    monkeypatch.setattr(
        cp_attention,
        "_wide_metadata_forward_bhsd",
        metadata_attention,
    )
    for key_start in (0, 0, 4):
        cp_attention._wide_shard_stats(
            query=query,
            key=key,
            value=value,
            attn_mask=mask,
            is_causal=False,
            scale=0.125,
            key_start=key_start,
        )

    assert calls == 2
