from __future__ import annotations

import math
import os
import tempfile
import traceback
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from dllm_parallel.core.attention.dflash_attention import (
    _prefer_query_rotation,
    dflash_attention,
)
from dllm_parallel.core.attention.masks import (
    DFlashGlobalContextMask,
    DFlashLocalBlockMask,
)
from dllm_parallel.core.kernels import dflash_cp_fusion


def _dense_dflash_attention(
    query: torch.Tensor,
    local_key: torch.Tensor,
    local_value: torch.Tensor,
    global_key: torch.Tensor,
    global_value: torch.Tensor,
    *,
    context_starts: torch.Tensor,
    context_stops: torch.Tensor,
    block_size: int,
    causal: bool,
) -> torch.Tensor:
    query_heads = query.shape[2]
    repeats = query_heads // local_key.shape[2]
    local_key = local_key.repeat_interleave(repeats, dim=2)
    local_value = local_value.repeat_interleave(repeats, dim=2)
    global_key = global_key.repeat_interleave(repeats, dim=2)
    global_value = global_value.repeat_interleave(repeats, dim=2)

    scale = 1.0 / math.sqrt(query.shape[-1])
    local_scores = torch.einsum(
        "bqhd,bkhd->bhqk", query.float(), local_key.float()
    ) * scale
    global_scores = torch.einsum(
        "bqhd,bkhd->bhqk", query.float(), global_key.float()
    ) * scale

    query_positions = torch.arange(query.shape[1], device=query.device)
    query_anchors = query_positions // block_size
    local_positions = torch.arange(local_key.shape[1], device=query.device)
    local_anchors = local_positions // block_size
    local_allowed = query_anchors[:, None] == local_anchors[None, :]
    if causal:
        local_allowed &= (
            query_positions.remainder(block_size)[:, None]
            >= local_positions.remainder(block_size)[None, :]
        )
    local_scores.masked_fill_(~local_allowed[None, None], -torch.inf)

    global_positions = torch.arange(global_key.shape[1], device=query.device)
    starts = context_starts[:, query_anchors]
    stops = context_stops[:, query_anchors]
    global_allowed = (global_positions[None, None, :] >= starts[:, :, None]) & (
        global_positions[None, None, :] < stops[:, :, None]
    )
    global_scores.masked_fill_(~global_allowed[:, None], -torch.inf)

    scores = torch.cat((local_scores, global_scores), dim=-1)
    probabilities = torch.softmax(scores, dim=-1)
    values = torch.cat((local_value, global_value), dim=1)
    return torch.einsum("bhqk,bkhd->bqhd", probabilities, values.float())


def _flex_dflash_attention(
    query: torch.Tensor,
    local_key: torch.Tensor,
    local_value: torch.Tensor,
    global_key: torch.Tensor,
    global_value: torch.Tensor,
    *,
    context_starts: torch.Tensor,
    context_stops: torch.Tensor,
    anchor_valid: torch.Tensor,
    block_size: int,
    causal: bool,
) -> torch.Tensor:
    """Reference FlexAttention oracle for the DFlash mask."""

    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    query_length = int(query.shape[1])
    key_length = query_length + int(global_key.shape[1])

    def mask_mod(
        batch: torch.Tensor,
        head: torch.Tensor,
        query_index: torch.Tensor,
        key_index: torch.Tensor,
    ) -> torch.Tensor:
        del head
        query_anchor = query_index // int(block_size)
        local_key_index = key_index.clamp_max(query_length - 1)
        local_key_anchor = local_key_index // int(block_size)
        local_allowed = (key_index < query_length) & (
            query_anchor == local_key_anchor
        )
        if causal:
            local_allowed &= (
                query_index.remainder(int(block_size))
                >= local_key_index.remainder(int(block_size))
            )
        global_position = key_index - query_length
        global_allowed = (key_index >= query_length) & (
            global_position >= context_starts[batch, query_anchor]
        ) & (global_position < context_stops[batch, query_anchor])
        return anchor_valid[batch, query_anchor] & (
            local_allowed | global_allowed
        )

    block_mask = create_block_mask(
        mask_mod,
        B=int(query.shape[0]),
        H=None,
        Q_LEN=query_length,
        KV_LEN=key_length,
        device=query.device,
    )

    @torch.compile(fullgraph=True, mode="max-autotune-no-cudagraphs")
    def compiled_attention(
        query_arg: torch.Tensor,
        key_arg: torch.Tensor,
        value_arg: torch.Tensor,
    ) -> torch.Tensor:
        return flex_attention(
            query_arg,
            key_arg,
            value_arg,
            block_mask=block_mask,
            enable_gqa=int(query_arg.shape[1]) != int(key_arg.shape[1]),
        )

    key = torch.cat((local_key, global_key), dim=1)
    value = torch.cat((local_value, global_value), dim=1)
    return compiled_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
    ).transpose(1, 2)


def test_dflash_sparse_indices_have_canonical_inner_stride() -> None:
    from dllm_parallel.core.attention.dflash_fa4 import _ordered

    dense = torch.ones((1, 1, 4), dtype=torch.bool).transpose(1, 2)

    _, indices = _ordered(dense)

    assert indices.shape == (1, 4, 1)
    assert indices.stride(-1) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_dflash_native_state_merge_matches_fp32_reference() -> None:
    torch.manual_seed(91)
    device = torch.device("cuda")
    numerator = torch.randn((2, 3, 5, 64), device=device, dtype=torch.float32)
    maximum = torch.randn((2, 3, 5), device=device, dtype=torch.float32)
    normalizer = torch.rand_like(maximum)
    incoming_numerator = torch.randn_like(numerator)
    incoming_maximum = torch.randn_like(maximum)
    incoming_normalizer = torch.rand_like(normalizer)

    next_maximum = torch.maximum(maximum, incoming_maximum)
    old_weight = torch.exp(maximum - next_maximum)
    incoming_weight = torch.exp(incoming_maximum - next_maximum)
    expected_numerator = (
        old_weight[..., None] * numerator
        + incoming_weight[..., None] * incoming_numerator
    )
    expected_normalizer = (
        old_weight * normalizer + incoming_weight * incoming_normalizer
    )

    dflash_cp_fusion.merge_state_(
        numerator,
        maximum,
        normalizer,
        incoming_numerator,
        incoming_maximum,
        incoming_normalizer,
    )
    torch.testing.assert_close(numerator, expected_numerator)
    torch.testing.assert_close(maximum, next_maximum)
    torch.testing.assert_close(normalizer, expected_normalizer)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("context_length", [12, 512])
def test_dflash_attention_matches_dense_forward_and_backward(
    causal: bool,
    batch: int,
    context_length: int,
) -> None:
    torch.manual_seed(173)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    anchors, block_size = 2, 4
    query_heads, kv_heads, head_dim = 32, 8, 128
    query_length = anchors * block_size

    query = torch.randn(
        batch,
        query_length,
        query_heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    local_key = torch.randn(
        batch,
        query_length,
        kv_heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    local_value = torch.randn_like(local_key, requires_grad=True)
    global_key = torch.randn(
        batch,
        context_length,
        kv_heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    global_value = torch.randn_like(global_key, requires_grad=True)
    context_starts = torch.tensor(
        [[0, 2], [1, 3]][:batch],
        device=device,
        dtype=torch.int32,
    )
    context_stops = torch.tensor(
        [
            [context_length // 3, context_length - 2],
            [context_length // 2, context_length - 1],
        ][:batch],
        device=device,
        dtype=torch.int32,
    )
    anchor_valid = torch.ones((batch, anchors), device=device, dtype=torch.bool)

    observed = dflash_attention(
        query=query,
        local_key=local_key,
        local_value=local_value,
        global_key=global_key,
        global_value=global_value,
        local_attn_mask=DFlashLocalBlockMask(
            anchor_valid=anchor_valid,
            block_size=block_size,
            causal=causal,
        ),
        global_attn_mask=DFlashGlobalContextMask(
            context_starts=context_starts,
            context_stops=context_stops,
            anchor_valid=anchor_valid,
            block_size=block_size,
        ),
        scale=None,
        runtime=None,
        global_seq_len=context_length,
    )

    reference_inputs = [
        tensor.detach().clone().requires_grad_(True)
        for tensor in (query, local_key, local_value, global_key, global_value)
    ]
    reference = _dense_dflash_attention(
        *reference_inputs,
        context_starts=context_starts,
        context_stops=context_stops,
        block_size=block_size,
        causal=causal,
    )
    torch.testing.assert_close(observed.float(), reference, atol=3e-2, rtol=3e-2)

    grad_output = torch.randn_like(observed)
    observed_gradients = torch.autograd.grad(
        observed,
        (query, local_key, local_value, global_key, global_value),
        grad_output,
    )
    reference_gradients = torch.autograd.grad(
        reference,
        tuple(reference_inputs),
        grad_output.float(),
    )
    for observed_gradient, reference_gradient in zip(
        observed_gradients,
        reference_gradients,
        strict=True,
    ):
        torch.testing.assert_close(
            observed_gradient.float(),
            reference_gradient.float(),
            atol=7e-2,
            rtol=7e-2,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("block_size", [8, 16])
def test_dflash_local_fa4_matches_merged_dense_state(
    causal: bool,
    block_size: int,
) -> None:
    from dllm_parallel.core.attention.dflash_fa4 import (
        local_backward_from_state,
        local_forward,
    )

    torch.manual_seed(1907 + block_size + int(causal))
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch, anchors = 1, 32
    query_heads, kv_heads, head_dim = 32, 8, 128
    query_length = anchors * block_size
    query = torch.randn(
        batch,
        query_length,
        query_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )
    key = torch.randn(
        batch,
        query_length,
        kv_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )
    value = torch.randn_like(key)
    scale = head_dim**-0.5
    local_output, local_lse = local_forward(
        query,
        key,
        value,
        block_size=block_size,
        causal=causal,
        scale=scale,
    )

    context_length = 64
    context_key = torch.randn(
        batch * anchors,
        context_length,
        kv_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )
    context_value = torch.randn_like(context_key)
    block_query = query.view(
        batch * anchors,
        block_size,
        query_heads,
        head_dim,
    )
    repeated_context_key = context_key.repeat_interleave(
        query_heads // kv_heads,
        dim=2,
    )
    repeated_context_value = context_value.repeat_interleave(
        query_heads // kv_heads,
        dim=2,
    )
    context_scores = torch.einsum(
        "bqhd,bkhd->bhqk",
        block_query.float(),
        repeated_context_key.float(),
    ) * scale
    context_lse = torch.logsumexp(context_scores, dim=-1)
    context_output = torch.einsum(
        "bhqk,bkhd->bqhd",
        torch.softmax(context_scores, dim=-1),
        repeated_context_value.float(),
    )
    local_lse_blocks = (
        local_lse.view(batch, query_heads, anchors, block_size)
        .permute(0, 2, 1, 3)
        .reshape(batch * anchors, query_heads, block_size)
    )
    merged_lse = torch.logaddexp(local_lse_blocks, context_lse)
    local_weight = torch.exp(local_lse_blocks - merged_lse).transpose(1, 2)[..., None]
    context_weight = torch.exp(context_lse - merged_lse).transpose(1, 2)[..., None]
    merged_output = (
        local_weight * local_output.view_as(block_query).float()
        + context_weight * context_output
    ).to(dtype)
    merged_output = merged_output.view_as(query)
    merged_lse = (
        merged_lse.view(batch, anchors, query_heads, block_size)
        .permute(0, 2, 1, 3)
        .reshape(batch, query_heads, query_length)
        .contiguous()
    )
    grad_output = torch.randn_like(query)
    observed = local_backward_from_state(
        query,
        key,
        value,
        merged_output,
        merged_lse,
        grad_output,
        block_size=block_size,
        causal=causal,
        scale=scale,
    )

    reference_query = query.detach().clone().float().requires_grad_(True)
    reference_key = key.detach().clone().float().requires_grad_(True)
    reference_value = value.detach().clone().float().requires_grad_(True)
    reference_context_key = context_key.detach().clone().float()
    reference_context_value = context_value.detach().clone().float()
    reference_query_blocks = reference_query.view_as(block_query)
    reference_key_blocks = reference_key.view(
        batch * anchors,
        block_size,
        kv_heads,
        head_dim,
    ).repeat_interleave(query_heads // kv_heads, dim=2)
    reference_value_blocks = reference_value.view(
        batch * anchors,
        block_size,
        kv_heads,
        head_dim,
    ).repeat_interleave(query_heads // kv_heads, dim=2)
    local_scores = torch.einsum(
        "bqhd,bkhd->bhqk",
        reference_query_blocks,
        reference_key_blocks,
    ) * scale
    if causal:
        positions = torch.arange(block_size, device=device)
        local_scores = local_scores.masked_fill(
            positions[None, :] > positions[:, None],
            -torch.inf,
        )
    reference_context_key = reference_context_key.repeat_interleave(
        query_heads // kv_heads,
        dim=2,
    )
    reference_context_value = reference_context_value.repeat_interleave(
        query_heads // kv_heads,
        dim=2,
    )
    global_scores = torch.einsum(
        "bqhd,bkhd->bhqk",
        reference_query_blocks.detach(),
        reference_context_key,
    ) * scale
    scores = torch.cat((local_scores, global_scores), dim=-1)
    values = torch.cat((reference_value_blocks, reference_context_value), dim=1)
    reference_output = torch.einsum(
        "bhqk,bkhd->bqhd",
        torch.softmax(scores, dim=-1),
        values,
    ).reshape_as(reference_query)
    reference = torch.autograd.grad(
        reference_output,
        (reference_query, reference_key, reference_value),
        grad_output.float(),
    )
    for actual, expected in zip(observed, reference, strict=True):
        torch.testing.assert_close(
            actual.float(),
            expected.float(),
            atol=8e-2,
            rtol=8e-2,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_dflash_attention_matches_dense_at_production_head_shape() -> None:
    torch.manual_seed(811)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch, anchors, block_size = 1, 16, 8
    query_heads, kv_heads, head_dim = 32, 8, 128
    query_length = anchors * block_size
    context_length = 512
    tensors = (
        torch.randn(
            batch,
            query_length,
            query_heads,
            head_dim,
            device=device,
            dtype=dtype,
            requires_grad=True,
        ),
        torch.randn(
            batch,
            query_length,
            kv_heads,
            head_dim,
            device=device,
            dtype=dtype,
            requires_grad=True,
        ),
        torch.randn(
            batch,
            query_length,
            kv_heads,
            head_dim,
            device=device,
            dtype=dtype,
            requires_grad=True,
        ),
        torch.randn(
            batch,
            context_length,
            kv_heads,
            head_dim,
            device=device,
            dtype=dtype,
            requires_grad=True,
        ),
        torch.randn(
            batch,
            context_length,
            kv_heads,
            head_dim,
            device=device,
            dtype=dtype,
            requires_grad=True,
        ),
    )
    context_starts = torch.zeros(
        (batch, anchors), device=device, dtype=torch.int32
    )
    context_stops = torch.linspace(
        context_length // anchors,
        context_length,
        anchors,
        device=device,
        dtype=torch.float32,
    ).to(torch.int32)[None]
    anchor_valid = torch.ones(
        (batch, anchors), device=device, dtype=torch.bool
    )
    observed = dflash_attention(
        query=tensors[0],
        local_key=tensors[1],
        local_value=tensors[2],
        global_key=tensors[3],
        global_value=tensors[4],
        local_attn_mask=DFlashLocalBlockMask(
            anchor_valid=anchor_valid,
            block_size=block_size,
        ),
        global_attn_mask=DFlashGlobalContextMask(
            context_starts=context_starts,
            context_stops=context_stops,
            anchor_valid=anchor_valid,
            block_size=block_size,
        ),
        scale=None,
        runtime=None,
        global_seq_len=context_length,
    )
    reference_inputs = tuple(
        tensor.detach().clone().requires_grad_(True) for tensor in tensors
    )
    reference = _flex_dflash_attention(
        *reference_inputs,
        context_starts=context_starts,
        context_stops=context_stops,
        anchor_valid=anchor_valid,
        block_size=block_size,
        causal=False,
    )
    torch.testing.assert_close(
        observed.float(), reference.float(), atol=3e-2, rtol=3e-2
    )
    grad_output = torch.randn_like(observed)
    observed_gradients = torch.autograd.grad(observed, tensors, grad_output)
    reference_gradients = torch.autograd.grad(
        reference,
        reference_inputs,
        grad_output,
    )
    for observed_gradient, reference_gradient in zip(
        observed_gradients,
        reference_gradients,
        strict=True,
    ):
        torch.testing.assert_close(
            observed_gradient.float(),
            reference_gradient.float(),
            atol=7e-2,
            rtol=7e-2,
        )


def _distributed_dflash_worker(
    rank: int,
    world_size: int,
    init_file: str,
    queue: object,
    context_length: int,
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
        batch, anchors, block_size = 1, 2, 4
        query_heads, kv_heads, head_dim = 4, 2, 64
        query_length = anchors * block_size
        shard_length = context_length // world_size

        def random_tensor(seed: int, shape: tuple[int, ...]) -> torch.Tensor:
            generator = torch.Generator(device=device).manual_seed(seed)
            return torch.randn(
                shape,
                generator=generator,
                device=device,
                dtype=dtype,
            )

        full_global_key = random_tensor(
            701,
            (batch, context_length, kv_heads, head_dim),
        )
        full_global_value = random_tensor(
            702,
            (batch, context_length, kv_heads, head_dim),
        )
        queries = [
            random_tensor(
                710 + worker_rank,
                (batch, query_length, query_heads, head_dim),
            )
            for worker_rank in range(world_size)
        ]
        local_keys = [
            random_tensor(
                720 + worker_rank,
                (batch, query_length, kv_heads, head_dim),
            )
            for worker_rank in range(world_size)
        ]
        local_values = [
            random_tensor(
                730 + worker_rank,
                (batch, query_length, kv_heads, head_dim),
            )
            for worker_rank in range(world_size)
        ]
        output_grads = [
            random_tensor(
                740 + worker_rank,
                (batch, query_length, query_heads, head_dim),
            )
            for worker_rank in range(world_size)
        ]
        context_starts = torch.tensor(
            [[0, 2]],
            device=device,
            dtype=torch.int32,
        )
        context_stops = torch.tensor(
            [[context_length // 3, context_length - 2]],
            device=device,
            dtype=torch.int32,
        )
        anchor_valid = torch.ones(
            (batch, anchors),
            device=device,
            dtype=torch.bool,
        )
        local_mask = DFlashLocalBlockMask(
            anchor_valid=anchor_valid,
            block_size=block_size,
            causal=False,
        )
        global_mask = DFlashGlobalContextMask(
            context_starts=context_starts,
            context_stops=context_stops,
            anchor_valid=anchor_valid,
            block_size=block_size,
            global_anchor_count=anchors * world_size,
        )
        shard_start = rank * shard_length
        shard_stop = shard_start + shard_length
        query = queries[rank].clone().requires_grad_(True)
        local_key = local_keys[rank].clone().requires_grad_(True)
        local_value = local_values[rank].clone().requires_grad_(True)
        global_key = full_global_key[:, shard_start:shard_stop].clone().requires_grad_(True)
        global_value = full_global_value[:, shard_start:shard_stop].clone().requires_grad_(True)
        runtime = SimpleNamespace(
            uses_context_parallel_attention=True,
            context_parallel_rank=rank,
            block_parallel_size=1,
            context_attention_size=world_size,
            active_block_mode="all_blocks",
            context_block_parallel_group=dist.group.WORLD,
            context_block_parallel_group_ranks=list(range(world_size)),
        )
        observed = dflash_attention(
            query=query,
            local_key=local_key,
            local_value=local_value,
            global_key=global_key,
            global_value=global_value,
            local_attn_mask=local_mask,
            global_attn_mask=global_mask,
            scale=None,
            runtime=runtime,
            global_seq_len=context_length,
        )

        reference_queries = [value.clone().requires_grad_(True) for value in queries]
        reference_local_keys = [
            value.clone().requires_grad_(True) for value in local_keys
        ]
        reference_local_values = [
            value.clone().requires_grad_(True) for value in local_values
        ]
        reference_global_key = full_global_key.clone().requires_grad_(True)
        reference_global_value = full_global_value.clone().requires_grad_(True)
        reference_loss = reference_global_key.new_tensor(0.0)
        reference_for_rank = None
        for worker_rank in range(world_size):
            reference_output = _dense_dflash_attention(
                reference_queries[worker_rank],
                reference_local_keys[worker_rank],
                reference_local_values[worker_rank],
                reference_global_key,
                reference_global_value,
                context_starts=context_starts,
                context_stops=context_stops,
                block_size=block_size,
                causal=False,
            )
            if worker_rank == rank:
                reference_for_rank = reference_output
            reference_loss = reference_loss + (
                reference_output * output_grads[worker_rank].float()
            ).sum()
        assert reference_for_rank is not None
        torch.testing.assert_close(
            observed.float(),
            reference_for_rank,
            atol=3e-2,
            rtol=3e-2,
        )

        (observed * output_grads[rank]).sum().backward()
        reference_loss.backward()
        comparisons = (
            (query.grad, reference_queries[rank].grad),
            (local_key.grad, reference_local_keys[rank].grad),
            (local_value.grad, reference_local_values[rank].grad),
            (
                global_key.grad,
                reference_global_key.grad[:, shard_start:shard_stop],
            ),
            (
                global_value.grad,
                reference_global_value.grad[:, shard_start:shard_stop],
            ),
        )
        for actual, expected in comparisons:
            torch.testing.assert_close(
                actual.float(),
                expected.float(),
                atol=8e-2,
                rtol=8e-2,
            )
        queue.put((rank, "ok", ""))
    except Exception:  # pragma: no cover - child reports to parent.
        queue.put((rank, "error", traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="two CUDA devices are required",
)
@pytest.mark.parametrize("context_length", [12, 512])
def test_dflash_ring_matches_dense_forward_and_backward(
    context_length: int,
) -> None:
    world_size = 2
    context = mp.get_context("spawn")
    queue = context.Queue()
    with tempfile.NamedTemporaryFile(delete=False) as rendezvous:
        init_file = rendezvous.name
    os.unlink(init_file)
    processes = [
        context.Process(
            target=_distributed_dflash_worker,
            args=(rank, world_size, init_file, queue, context_length),
        )
        for rank in range(world_size)
    ]
    try:
        for process in processes:
            process.start()
        results = [queue.get(timeout=180) for _ in range(world_size)]
        for process in processes:
            process.join(timeout=180)
        failures = [message for _, status, message in results if status != "ok"]
        assert not failures, "\n".join(failures)
        assert all(process.exitcode == 0 for process in processes)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
        if os.path.exists(init_file):
            os.unlink(init_file)


def _selection_inputs(
    *,
    context_length: int,
    global_anchor_count: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    DFlashLocalBlockMask,
    DFlashGlobalContextMask,
    SimpleNamespace,
]:
    ring_size = 4
    block_size = 8
    local_anchors = global_anchor_count // ring_size
    query = torch.empty(
        (1, local_anchors * block_size, 32, 128),
        dtype=torch.bfloat16,
        device="meta",
    )
    global_key = torch.empty(
        (1, context_length // ring_size, 8, 128),
        dtype=torch.bfloat16,
        device="meta",
    )
    anchor_valid = torch.ones(
        (1, local_anchors),
        dtype=torch.bool,
        device="meta",
    )
    local_mask = DFlashLocalBlockMask(
        anchor_valid=anchor_valid,
        block_size=block_size,
    )
    global_mask = DFlashGlobalContextMask(
        context_starts=torch.zeros(
            (1, local_anchors),
            dtype=torch.int32,
            device="meta",
        ),
        context_stops=torch.full(
            (1, local_anchors),
            context_length,
            dtype=torch.int32,
            device="meta",
        ),
        anchor_valid=anchor_valid,
        block_size=block_size,
        global_anchor_count=global_anchor_count,
    )
    runtime = SimpleNamespace(
        uses_context_parallel_attention=True,
        context_parallel_rank=0,
        context_attention_size=ring_size,
        block_parallel_size=ring_size,
        active_block_mode="dual_end",
        context_block_parallel_group=None,
        context_block_parallel_group_ranks=list(range(ring_size)),
        cp_bp_policy=SimpleNamespace(clean_kv_layout="zigzag"),
    )
    return query, global_key, local_mask, global_mask, runtime


def test_dflash_transport_policy_keeps_kv_ring_for_large_query_set() -> None:
    query, global_key, local_mask, global_mask, runtime = _selection_inputs(
        context_length=32 * 1024,
        global_anchor_count=3072,
    )

    assert not _prefer_query_rotation(
        query=query,
        global_key=global_key,
        local_mask=local_mask,
        global_mask=global_mask,
        runtime=runtime,
        global_seq_len=32 * 1024,
    )


def test_dflash_transport_policy_rotates_small_query_set() -> None:
    query, global_key, local_mask, global_mask, runtime = _selection_inputs(
        context_length=128 * 1024,
        global_anchor_count=64,
    )

    assert _prefer_query_rotation(
        query=query,
        global_key=global_key,
        local_mask=local_mask,
        global_mask=global_mask,
        runtime=runtime,
        global_seq_len=128 * 1024,
    )
