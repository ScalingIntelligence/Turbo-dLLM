# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import os
import tempfile
import traceback
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn

import dllm_parallel.core.attention.context_parallel_attention as cp_attention
from dllm_parallel.core.attention.layout import active_query_indices_for_context_rank

from dllm_parallel.core.models.backbones.qwen3_8.model import (
    Qwen38PackedBlockDiffusionModel,
    _SequenceParallelMeta,
)
from dllm_parallel.core.models.backbones.qwen3_8.token_local import (
    build_pure_cp_token_row_plan,
    compact_pure_cp_token_rows,
    expand_pure_cp_logical_rows,
    reconstruct_pure_cp_token_rows,
    select_pure_cp_logical_rows,
)


class _CountingLinear(nn.Linear):
    def __init__(self, features: int) -> None:
        super().__init__(features, features, bias=False)
        self.seen_rows = -1

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.seen_rows = int(inputs.shape[0])
        return super().forward(inputs)


class _BiasFreeSwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.norm_weight = nn.Parameter(torch.ones(hidden_size))
        self.gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        variance = inputs.float().square().mean(dim=-1, keepdim=True)
        normalized = (
            inputs.float() * torch.rsqrt(variance + 1e-6)
        ).to(dtype=inputs.dtype) * self.norm_weight
        return self.down(F.silu(self.gate(normalized)) * self.up(normalized))


def _plan(*, tensor_rank: int, context_rank: int, batch_size: int = 1):
    return build_pure_cp_token_row_plan(
        batch_size=batch_size,
        packed_len=80,
        active_len=64,
        block_size=8,
        tensor_parallel_size=2,
        tensor_parallel_rank=tensor_rank,
        context_parallel_size=4,
        context_parallel_rank=context_rank,
        device=torch.device("cpu"),
    )


def test_pure_cp_token_row_plan_partitions_offsets_inside_every_block() -> None:
    tp0_expected = (
        [0, 7, 8, 15, 16, 23, 24, 31, 32, 39],
        [1, 6, 9, 14, 17, 22, 25, 30, 33, 38],
        [2, 5, 10, 13, 18, 21, 26, 29, 34, 37],
        [3, 4, 11, 12, 19, 20, 27, 28, 35, 36],
    )
    tp1_expected = (
        [0, 7, 8, 15, 16, 23],
        [1, 6, 9, 14, 17, 22],
        [2, 5, 10, 13, 18, 21],
        [3, 4, 11, 12, 19, 20],
    )

    plans = {
        (tp_rank, cp_rank): _plan(
            tensor_rank=tp_rank,
            context_rank=cp_rank,
        )
        for tp_rank in range(2)
        for cp_rank in range(4)
    }

    assert all(plan is not None for plan in plans.values())
    assert {plan.compute_rows for plan in plans.values() if plan is not None} == {22}
    for cp_rank in range(4):
        tp0 = plans[(0, cp_rank)]
        tp1 = plans[(1, cp_rank)]
        assert tp0 is not None
        assert tp1 is not None
        assert tp0.active_rows == 10
        assert tp0.clean_rows == 0
        assert tp0.output_rows == 40
        assert tp0.compact_input_indices.tolist() == tp0_expected[cp_rank]
        assert tp0.active_output_positions_by_rank.tolist() == list(tp0_expected)

        assert tp1.active_rows == 6
        assert tp1.clean_rows == 16
        assert tp1.output_rows == 40
        assert tp1.compact_input_indices.tolist() == [
            *tp1_expected[cp_rank],
            *range(24, 40),
        ]
        assert tp1.active_output_positions_by_rank.tolist() == list(tp1_expected)
        assert tp1.clean_output_positions.tolist() == list(range(24, 40))


def test_pure_cp_token_row_plan_balances_two_batches_without_dummy_rows() -> None:
    plans = [
        _plan(tensor_rank=tp_rank, context_rank=cp_rank, batch_size=2)
        for tp_rank in range(2)
        for cp_rank in range(4)
    ]

    assert all(plan is not None for plan in plans)
    assert {plan.compute_rows for plan in plans if plan is not None} == {32}
    assert {plan.active_rows for plan in plans if plan is not None} == {16}
    assert {plan.clean_rows for plan in plans if plan is not None} == {16}


def test_32k_tp2_cp4_layout_reduces_each_mlp_to_11264_rows() -> None:
    plans = {
        (tp_rank, cp_rank): build_pure_cp_token_row_plan(
            batch_size=1,
            packed_len=40_960,
            active_len=32_768,
            block_size=256,
            tensor_parallel_size=2,
            tensor_parallel_rank=tp_rank,
            context_parallel_size=4,
            context_parallel_rank=cp_rank,
            device=torch.device("cpu"),
        )
        for tp_rank in range(2)
        for cp_rank in range(4)
    }

    assert all(plan is not None for plan in plans.values())
    assert {plan.compute_rows for plan in plans.values() if plan is not None} == {
        11_264
    }
    for cp_rank in range(4):
        first = plans[(0, cp_rank)]
        second = plans[(1, cp_rank)]
        assert first is not None
        assert second is not None
        assert (first.active_rows, first.clean_rows) == (5_120, 0)
        assert (second.active_rows, second.clean_rows) == (3_072, 8_192)


def test_compact_pure_cp_token_rows_zero_pads_to_the_tp_peer_maximum() -> None:
    plan = _plan(tensor_rank=0, context_rank=0)
    assert plan is not None
    hidden = torch.arange(80, dtype=torch.float32).view(40, 2)

    compact = compact_pure_cp_token_rows(hidden, plan)

    assert compact.shape == (22, 2)
    torch.testing.assert_close(
        compact[:10],
        hidden.index_select(0, torch.tensor([0, 7, 8, 15, 16, 23, 24, 31, 32, 39])),
    )
    assert torch.count_nonzero(compact[10:]) == 0


@pytest.mark.parametrize("context_rank", range(4))
def test_persistent_pure_cp_rows_round_trip_tp_gather_layout(
    context_rank: int,
) -> None:
    plans = [
        _plan(tensor_rank=tensor_rank, context_rank=context_rank)
        for tensor_rank in range(2)
    ]
    assert all(plan is not None for plan in plans)
    concrete = [plan for plan in plans if plan is not None]
    hidden = torch.arange(80, dtype=torch.float32).view(80, 1)
    tp_shards = hidden.chunk(2, dim=0)
    gathered_compact = torch.cat(
        [
            compact_pure_cp_token_rows(shard, plan)
            for shard, plan in zip(tp_shards, concrete, strict=True)
        ],
        dim=0,
    )

    logical = select_pure_cp_logical_rows(gathered_compact, concrete[0])
    expected_positions = concrete[0].logical_token_positions.flatten()
    torch.testing.assert_close(logical[:, 0], expected_positions.float())
    expanded = expand_pure_cp_logical_rows(logical, concrete[0])
    torch.testing.assert_close(
        expanded.index_select(0, concrete[0].logical_input_indices),
        logical,
    )
    padding = torch.ones(expanded.shape[0], dtype=torch.bool)
    padding[concrete[0].logical_input_indices] = False
    assert torch.count_nonzero(expanded[padding]) == 0


def test_pure_cp_token_row_plan_balances_split_tp_interval() -> None:
    plan = build_pure_cp_token_row_plan(
        batch_size=1,
        packed_len=40,
        active_len=32,
        block_size=8,
        tensor_parallel_size=2,
        tensor_parallel_rank=0,
        context_parallel_size=4,
        context_parallel_rank=0,
        device=torch.device("cpu"),
    )

    assert plan is not None
    assert plan.active_rows == 5
    assert plan.compute_rows == 11
    assert plan.logical_active_len == 8


def _reconstruct_worker(rank: int, world_size: int, init_file: str, queue) -> None:
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
        )
        plan = build_pure_cp_token_row_plan(
            batch_size=1,
            packed_len=24,
            active_len=16,
            block_size=8,
            tensor_parallel_size=1,
            tensor_parallel_rank=0,
            context_parallel_size=world_size,
            context_parallel_rank=rank,
            device=torch.device("cpu"),
        )
        assert plan is not None
        base_hidden = torch.arange(72, dtype=torch.float32).view(24, 3) / 37.0
        base_weight = torch.arange(15, dtype=torch.float32).view(5, 3) / 19.0
        hidden = base_hidden.clone().requires_grad_(True)
        weight = base_weight.clone().requires_grad_(True)
        compact = compact_pure_cp_token_rows(hidden, plan)
        compact_output = F.linear(compact, weight)
        output = reconstruct_pure_cp_token_rows(
            compact_output,
            plan=plan,
            group=dist.group.WORLD,
            context_parallel_size=world_size,
            context_parallel_rank=rank,
        )
        torch.testing.assert_close(output, F.linear(base_hidden, base_weight))

        downstream = (
            torch.arange(120, dtype=torch.float32).view(24, 5) / 113.0
            + float(rank)
        )
        (output * downstream).sum().backward()
        dist.all_reduce(weight.grad)
        dist.all_reduce(hidden.grad)

        reference_hidden = base_hidden.clone().requires_grad_(True)
        reference_weight = base_weight.clone().requires_grad_(True)
        reference_output = F.linear(reference_hidden, reference_weight)
        reference_loss = sum(
            (
                reference_output
                * (
                    torch.arange(120, dtype=torch.float32).view(24, 5) / 113.0
                    + float(peer)
                )
            ).sum()
            for peer in range(world_size)
        )
        reference_loss.backward()
        torch.testing.assert_close(weight.grad, reference_weight.grad)
        torch.testing.assert_close(hidden.grad, reference_hidden.grad)
        queue.put((rank, "ok", ""))
    except Exception:  # pragma: no cover - child reports failure to parent.
        queue.put((rank, "error", traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_reconstruct_pure_cp_token_rows_matches_replicated_forward_and_backward() -> (
    None
):
    if not dist.is_available() or not dist.is_gloo_available():
        return
    world_size = 2
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    with tempfile.NamedTemporaryFile(delete=False) as init:
        init_file = init.name
    try:
        processes = [
            ctx.Process(
                target=_reconstruct_worker,
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


def _active_kv_gather_worker(
    rank: int,
    world_size: int,
    init_file: str,
    queue,
) -> None:
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
        )
        active_len = 8
        block_size = 4
        indices = active_query_indices_for_context_rank(
            active_len=active_len,
            block_size=block_size,
            context_parallel_size=world_size,
            context_parallel_rank=rank,
            device=torch.device("cpu"),
        )
        full_key = torch.arange(16, dtype=torch.float32).view(1, 8, 1, 2) / 17
        full_value = full_key + 3
        key = full_key.index_select(1, indices).clone().requires_grad_(True)
        value = full_value.index_select(1, indices).clone().requires_grad_(True)
        gathered_key, gathered_value = cp_attention._GatherPureCPActiveKV.apply(
            key,
            value,
            active_len,
            block_size,
            dist.group.WORLD,
            rank,
        )
        torch.testing.assert_close(gathered_key, full_key)
        torch.testing.assert_close(gathered_value, full_value)
        key_weight = torch.arange(16, dtype=torch.float32).view_as(full_key) + rank
        value_weight = key_weight * 2
        (gathered_key * key_weight).sum().add(
            (gathered_value * value_weight).sum()
        ).backward()
        expected_key_grad = sum(
            torch.arange(16, dtype=torch.float32).view_as(full_key) + peer
            for peer in range(world_size)
        ).index_select(1, indices)
        torch.testing.assert_close(key.grad, expected_key_grad)
        torch.testing.assert_close(value.grad, expected_key_grad * 2)
        queue.put((rank, "ok", ""))
    except Exception:  # pragma: no cover - child reports failure to parent.
        queue.put((rank, "error", traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_active_kv_gather_restores_block_offsets_and_reduces_gradients() -> None:
    if not dist.is_available() or not dist.is_gloo_available():
        return
    world_size = 2
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    with tempfile.NamedTemporaryFile(delete=False) as init:
        init_file = init.name
    try:
        processes = [
            ctx.Process(
                target=_active_kv_gather_worker,
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
        results = [queue.get(timeout=5) for _ in processes]
        errors = [message for _, status, message in results if status != "ok"]
        assert not errors, "\n".join(errors)
    finally:
        if os.path.exists(init_file):
            os.unlink(init_file)


def _model_mlp_worker(
    rank: int,
    world_size: int,
    init_file: str,
    checkpointing: bool,
    queue,
) -> None:
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
        )
        runtime = SimpleNamespace(
            block_parallel_size=1,
            configured_context_parallel_size=world_size,
            context_attention_size=world_size,
            context_parallel_rank=rank,
            context_block_parallel_group=dist.group.WORLD,
            sequence_parallel=True,
            tensor_parallel_size=2,
            tensor_parallel_rank=0,
        )
        owner = object.__new__(Qwen38PackedBlockDiffusionModel)
        nn.Module.__init__(owner)
        owner.runtime = runtime
        owner.block_size = 8
        owner.activation_checkpointing = bool(checkpointing)
        owner.activation_checkpointing_scope = "mlp" if checkpointing else "none"
        owner._adapter_checkpointing = False
        owner._pure_cp_token_row_plan_cache = {}
        layer = nn.Identity()
        mlp = _CountingLinear(3)
        base_weight = torch.arange(9, dtype=torch.float32).view(3, 3) / 19.0
        with torch.no_grad():
            mlp.weight.copy_(base_weight)
        owner._te_packed_by_layer_id = {id(layer): SimpleNamespace(mlp=mlp)}
        meta = _SequenceParallelMeta(
            batch_size=1,
            packed_len=80,
            hidden_size=3,
            total_rows=80,
            padded_rows=80,
            active_len=64,
        )
        base_hidden = torch.arange(120, dtype=torch.float32).view(40, 3) / 37.0
        hidden = base_hidden.clone().requires_grad_(True)

        output = owner._mlp_forward_sequence_parallel(layer, hidden, meta)

        assert mlp.seen_rows == 28
        torch.testing.assert_close(output, F.linear(base_hidden, base_weight))
        downstream = (
            torch.arange(120, dtype=torch.float32).view(40, 3) / 113.0
            + float(rank)
        )
        (output * downstream).sum().backward()
        dist.all_reduce(mlp.weight.grad)
        dist.all_reduce(hidden.grad)

        reference_hidden = base_hidden.clone().requires_grad_(True)
        reference_weight = base_weight.clone().requires_grad_(True)
        reference_output = F.linear(reference_hidden, reference_weight)
        reference_loss = sum(
            (
                reference_output
                * (
                    torch.arange(120, dtype=torch.float32).view(40, 3) / 113.0
                    + float(peer)
                )
            ).sum()
            for peer in range(world_size)
        )
        reference_loss.backward()
        torch.testing.assert_close(mlp.weight.grad, reference_weight.grad)
        torch.testing.assert_close(hidden.grad, reference_hidden.grad)
        queue.put((rank, "ok", ""))
    except Exception:  # pragma: no cover - child reports failure to parent.
        queue.put((rank, "error", traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("checkpointing", [False, True])
def test_sequence_parallel_model_shards_pure_cp_mlp_rows(
    checkpointing: bool,
) -> None:
    if not dist.is_available() or not dist.is_gloo_available():
        return
    world_size = 2
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    with tempfile.NamedTemporaryFile(delete=False) as init:
        init_file = init.name
    try:
        processes = [
            ctx.Process(
                target=_model_mlp_worker,
                args=(rank, world_size, init_file, checkpointing, queue),
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


def _cuda_nccl_worker(
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
        torch.manual_seed(31)
        plan = build_pure_cp_token_row_plan(
            batch_size=1,
            packed_len=80,
            active_len=64,
            block_size=8,
            tensor_parallel_size=2,
            tensor_parallel_rank=0,
            context_parallel_size=world_size,
            context_parallel_rank=rank,
            device=device,
        )
        assert plan is not None
        module = _BiasFreeSwiGLU(32, 64).to(
            device=device,
            dtype=torch.bfloat16,
        )
        reference = _BiasFreeSwiGLU(32, 64).to(
            device=device,
            dtype=torch.bfloat16,
        )
        reference.load_state_dict(module.state_dict())
        base_hidden = (
            torch.arange(40 * 32, device=device, dtype=torch.float32)
            .view(40, 32)
            .remainder(97)
            .div(53.0)
            .to(torch.bfloat16)
        )
        hidden = base_hidden.clone().requires_grad_(True)
        compact = compact_pure_cp_token_rows(hidden, plan)
        output = reconstruct_pure_cp_token_rows(
            module(compact),
            plan=plan,
            group=dist.group.WORLD,
            context_parallel_size=world_size,
            context_parallel_rank=rank,
        )
        reference_hidden = base_hidden.clone().requires_grad_(True)
        reference_output = reference(reference_hidden)
        torch.testing.assert_close(output, reference_output, rtol=3e-2, atol=3e-2)

        downstream_base = (
            torch.arange(40 * 32, device=device, dtype=torch.float32)
            .view(40, 32)
            .remainder(89)
            .div(47.0)
            .to(torch.bfloat16)
        )
        (output * (downstream_base + rank)).sum().backward()
        for parameter in module.parameters():
            assert parameter.grad is not None
            dist.all_reduce(parameter.grad)
        assert hidden.grad is not None
        dist.all_reduce(hidden.grad)

        reference_loss = sum(
            (reference_output * (downstream_base + peer)).sum()
            for peer in range(world_size)
        )
        reference_loss.backward()
        for actual, expected in zip(
            module.parameters(),
            reference.parameters(),
            strict=True,
        ):
            torch.testing.assert_close(
                actual.grad,
                expected.grad,
                rtol=8e-2,
                atol=8e-2,
            )
        torch.testing.assert_close(
            hidden.grad,
            reference_hidden.grad,
            rtol=8e-2,
            atol=8e-2,
        )
        torch.cuda.synchronize(device)
        queue.put((rank, "ok", ""))
    except Exception:  # pragma: no cover - child reports failure to parent.
        queue.put((rank, "error", traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
def test_cuda_nccl_pure_cp_token_local_mlp_parity(world_size: int) -> None:
    if not dist.is_nccl_available() or torch.cuda.device_count() < world_size:
        pytest.skip(f"CUDA/NCCL with {world_size} devices is required")
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    with tempfile.NamedTemporaryFile(delete=False) as init:
        init_file = init.name
    try:
        processes = [
            ctx.Process(
                target=_cuda_nccl_worker,
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
