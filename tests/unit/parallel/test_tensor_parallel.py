# Copyright 2026 The bdlm_parallel Authors.
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
from omegaconf import OmegaConf

from dllm_parallel.core.parallel.tensor_parallel import (
    fused_gate_up_column_parallel_linear,
    gather_active_from_sequence_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
)
from dllm_parallel.core.parallel.tensor_parallel_layers import (
    ColumnParallelLinear as PublicColumnParallelLinear,
    RowParallelLinear as PublicRowParallelLinear,
    TensorParallelLayerConfig,
    VocabParallelEmbedding as PublicVocabParallelEmbedding,
    VocabParallelOutput,
    fused_gate_up_column_parallel_linear as public_fused_gate_up_column_parallel_linear,
    reduce_scatter_to_sequence_parallel_region as public_reduce_scatter_to_sequence_parallel_region,
)
from dllm_parallel.core.parallel.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    VocabParallelLinear,
)
from dllm_parallel.core.parallel.tensor_parallel.sequence_parallel import (
    _active_rows_in_interval,
)
from dllm_parallel.core.parallel.tensor_parallel.layers import (
    _VocabParallelEmbeddingFunction,
)
from dllm_parallel.core.parallel.grad_sync import (
    all_reduce_sequence_parallel_replicated_gradients,
    _gradient_reduce_group_for_parameter,
)


def test_sequence_parallel_replicated_gradients_are_summed(monkeypatch) -> None:
    module = torch.nn.Module()
    module.norm_weight = torch.nn.Parameter(torch.ones(2))
    module.norm_weight._dllm_sequence_parallel_replicated = True
    module.norm_weight.grad = torch.tensor([2.0, 3.0])
    runtime = SimpleNamespace(
        tensor_parallel_size=2,
        tensor_parallel_group="tp",
        sequence_parallel=True,
    )

    class Work:
        def block_current_stream(self) -> None:
            return None

        def wait(self) -> None:
            return None

    def fake_all_reduce_coalesced(tensors, *, op, group, async_op):
        assert op == dist.ReduceOp.SUM
        assert group == "tp"
        assert async_op is True
        for tensor in tensors:
            tensor.mul_(2.0)
        return Work()

    monkeypatch.setattr(dist, "all_reduce_coalesced", fake_all_reduce_coalesced)

    all_reduce_sequence_parallel_replicated_gradients(module, runtime)

    torch.testing.assert_close(module.norm_weight.grad, torch.tensor([4.0, 6.0]))


def test_fsdp_data_shards_do_not_hide_replicated_tp_gradients() -> None:
    class Shard:
        pass

    parameter = torch.nn.Parameter(torch.ones(2))
    parameter.placements = (Shard(),)
    parameter.device_mesh = SimpleNamespace(mesh_dim_names=("data_parallel",))
    runtime = SimpleNamespace(tensor_parallel_size=2)

    group, size, is_sharded = _gradient_reduce_group_for_parameter(
        parameter,
        runtime=runtime,
        shard_group="cp",
        shard_group_size=1,
        replicated_group="model",
        replicated_group_size=2,
    )

    assert (group, size, is_sharded) == ("model", 2, False)


def test_tp_dtensor_shards_reduce_only_across_context_workers() -> None:
    class Shard:
        pass

    parameter = torch.nn.Parameter(torch.ones(2))
    parameter.placements = (Shard(),)
    parameter.device_mesh = SimpleNamespace(mesh_dim_names=("tp",))
    runtime = SimpleNamespace(tensor_parallel_size=2)

    group, size, is_sharded = _gradient_reduce_group_for_parameter(
        parameter,
        runtime=runtime,
        shard_group="cp",
        shard_group_size=4,
        replicated_group="model",
        replicated_group_size=8,
    )

    assert (group, size, is_sharded) == ("cp", 4, True)


def test_public_tensor_parallel_layer_api_reexports_optimized_layers() -> None:
    runtime = SimpleNamespace(
        tensor_parallel_size=2,
        tensor_parallel_rank=1,
        sequence_parallel=True,
        tensor_parallel_overlap=False,
    )

    assert PublicColumnParallelLinear is ColumnParallelLinear
    assert PublicRowParallelLinear is RowParallelLinear
    assert PublicVocabParallelEmbedding is VocabParallelEmbedding
    assert VocabParallelOutput is VocabParallelLinear
    assert (
        public_reduce_scatter_to_sequence_parallel_region
        is reduce_scatter_to_sequence_parallel_region
    )
    assert public_fused_gate_up_column_parallel_linear is fused_gate_up_column_parallel_linear
    assert TensorParallelLayerConfig.from_runtime(runtime) == TensorParallelLayerConfig(
        tensor_parallel_size=2,
        tensor_parallel_rank=1,
        sequence_parallel=True,
        overlap=False,
    )
    config = OmegaConf.create(
        {
            "parallel": {
                "tensor_parallel_size": 4,
                "tensor_parallel_rank": 3,
                "sequence_parallel": True,
                "tensor_parallel_overlap": False,
            }
        }
    )
    assert TensorParallelLayerConfig.from_config(config) == TensorParallelLayerConfig(
        tensor_parallel_size=4,
        tensor_parallel_rank=3,
        sequence_parallel=True,
        overlap=False,
    )


def test_vocab_parallel_embedding_backward_accepts_no_local_tokens() -> None:
    weight = torch.randn(2, 4, requires_grad=True)
    token_ids = torch.tensor([[2, 3]], dtype=torch.long)

    output = _VocabParallelEmbeddingFunction.apply(
        token_ids,
        weight,
        0,
        2,
        None,
        1,
    )
    output.sum().backward()

    torch.testing.assert_close(output, torch.zeros_like(output))
    torch.testing.assert_close(weight.grad, torch.zeros_like(weight))


@pytest.mark.parametrize(
    ("row_start", "row_count", "total_rows", "packed_len", "active_len", "expected"),
    (
        (0, 6, 10, 5, 3, 4),
        (6, 6, 10, 5, 3, 2),
        (3, 4, 10, 5, 3, 2),
        (10, 4, 10, 5, 3, 0),
        (0, 10, 10, 5, 5, 10),
        (0, 10, 10, 5, 0, 0),
    ),
)
def test_active_sequence_row_count_is_exact(
    row_start: int,
    row_count: int,
    total_rows: int,
    packed_len: int,
    active_len: int,
    expected: int,
) -> None:
    assert _active_rows_in_interval(
        row_start=row_start,
        row_count=row_count,
        total_rows=total_rows,
        packed_len=packed_len,
        active_len=active_len,
    ) == expected


def _runtime(rank: int, world_size: int) -> SimpleNamespace:
    return SimpleNamespace(
        enabled=True,
        local_parallel_size=1,
        tensor_parallel_size=world_size,
        tensor_parallel_rank=rank,
        tensor_parallel_group=dist.group.WORLD,
        tensor_parallel_group_ranks=list(range(world_size)),
        context_parallel_rank=0,
        block_parallel_rank=0,
        context_block_parallel_group=dist.group.WORLD,
        context_block_parallel_group_ranks=[rank],
        model_parallel_group=dist.group.WORLD,
        model_parallel_group_ranks=list(range(world_size)),
        kv_backend="replicated",
    )


def _sequence_parallel_boundary_worker(rank: int, world_size: int, init_file: str, queue) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
        )
        runtime = SimpleNamespace(
            tensor_parallel_size=world_size,
            tensor_parallel_rank=rank,
            tensor_parallel_group=dist.group.WORLD,
        )
        full = torch.arange(24, dtype=torch.float32).view(12, 2).requires_grad_(True)
        local = scatter_to_sequence_parallel_region(full, runtime)
        active_hidden, active_pairs = gather_active_from_sequence_parallel_region(
            local,
            packed_len=5,
            active_len=3,
            total_rows=10,
            runtime=runtime,
        )
        expected_ids = torch.tensor([0, 1, 2, 5, 6, 7], dtype=torch.long)
        expected_pairs = torch.stack((expected_ids // 5, expected_ids % 5), dim=-1)
        torch.testing.assert_close(active_pairs.cpu(), expected_pairs)
        torch.testing.assert_close(active_hidden, full.detach().index_select(0, expected_ids))
        active_hidden.sum().backward()
        expected_grad = torch.zeros_like(full)
        expected_grad.index_fill_(0, expected_ids, 1.0)
        torch.testing.assert_close(full.grad, expected_grad)
        queue.put((rank, "ok", ""))
    except Exception:  # pragma: no cover - child reports failure to parent.
        queue.put((rank, "error", traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        os.environ.pop("RANK", None)
        os.environ.pop("WORLD_SIZE", None)


def _sequence_parallel_reduce_scatter_worker(
    rank: int,
    world_size: int,
    init_file: str,
    queue,
) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
        )
        runtime = SimpleNamespace(
            tensor_parallel_size=world_size,
            tensor_parallel_rank=rank,
            tensor_parallel_group=dist.group.WORLD,
        )
        partial = (
            torch.arange(24, dtype=torch.float32).view(12, 2)
            + 100 * rank
        ).requires_grad_(True)
        out = reduce_scatter_to_sequence_parallel_region(partial, runtime)
        full_sum = sum(
            torch.arange(24, dtype=torch.float32).view(12, 2) + 100 * candidate
            for candidate in range(world_size)
        )
        rows_per_rank = full_sum.shape[0] // world_size
        expected = full_sum[rank * rows_per_rank : (rank + 1) * rows_per_rank]
        torch.testing.assert_close(out, expected)
        out.sum().backward()
        torch.testing.assert_close(partial.grad, torch.ones_like(partial))
        queue.put((rank, "ok", ""))
    except Exception:  # pragma: no cover - child reports failure to parent.
        queue.put((rank, "error", traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        os.environ.pop("RANK", None)
        os.environ.pop("WORLD_SIZE", None)


def _fused_gate_up_worker(
    rank: int,
    world_size: int,
    init_file: str,
    queue,
) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
        )
        torch.manual_seed(2345 + rank)
        runtime = SimpleNamespace(
            tensor_parallel_size=world_size,
            tensor_parallel_rank=rank,
            tensor_parallel_group=dist.group.WORLD,
        )
        x = torch.randn(3, 5, 7, requires_grad=True)
        gate_weight = torch.randn(4 + rank, 7, requires_grad=True)
        up_weight = torch.randn(6 + rank, 7, requires_grad=True)
        gate_bias = torch.randn(4 + rank, requires_grad=True)
        up_bias = torch.randn(6 + rank, requires_grad=True)

        expected_x = x.detach().clone().requires_grad_(True)
        expected_gate_weight = gate_weight.detach().clone().requires_grad_(True)
        expected_up_weight = up_weight.detach().clone().requires_grad_(True)
        expected_gate_bias = gate_bias.detach().clone().requires_grad_(True)
        expected_up_bias = up_bias.detach().clone().requires_grad_(True)

        gate, up = fused_gate_up_column_parallel_linear(
            x,
            gate_weight,
            up_weight,
            gate_bias,
            up_bias,
            runtime,
        )
        expected_gate = F.linear(expected_x, expected_gate_weight, expected_gate_bias)
        expected_up = F.linear(expected_x, expected_up_weight, expected_up_bias)
        torch.testing.assert_close(gate, expected_gate)
        torch.testing.assert_close(up, expected_up)

        (gate.square().mean() + 0.7 * up.square().mean()).backward()
        (expected_gate.square().mean() + 0.7 * expected_up.square().mean()).backward()
        expected_x_grad = expected_x.grad.contiguous()
        dist.all_reduce(expected_x_grad, op=dist.ReduceOp.SUM, group=dist.group.WORLD)

        torch.testing.assert_close(x.grad, expected_x_grad)
        torch.testing.assert_close(gate_weight.grad, expected_gate_weight.grad)
        torch.testing.assert_close(up_weight.grad, expected_up_weight.grad)
        torch.testing.assert_close(gate_bias.grad, expected_gate_bias.grad)
        torch.testing.assert_close(up_bias.grad, expected_up_bias.grad)
        queue.put((rank, "ok", ""))
    except Exception:  # pragma: no cover - child reports failure to parent.
        queue.put((rank, "error", traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        os.environ.pop("RANK", None)
        os.environ.pop("WORLD_SIZE", None)


def test_fused_gate_up_column_parallel_linear_matches_separate_linears() -> None:
    world_size = 2
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    with tempfile.NamedTemporaryFile(delete=False) as handle:
        init_file = handle.name
    try:
        processes = [
            ctx.Process(
                target=_fused_gate_up_worker,
                args=(rank, world_size, init_file, queue),
            )
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=90)
        for process in processes:
            assert process.exitcode == 0
        results = [queue.get(timeout=10) for _ in range(world_size)]
        errors = [payload for _, status, payload in results if status != "ok"]
        assert not errors, errors[0] if errors else ""
    finally:
        try:
            os.unlink(init_file)
        except FileNotFoundError:
            pass

def test_sequence_parallel_reduce_scatter_sums_and_shards_rows() -> None:
    world_size = 2
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    with tempfile.NamedTemporaryFile(delete=False) as handle:
        init_file = handle.name
    try:
        processes = [
            ctx.Process(
                target=_sequence_parallel_reduce_scatter_worker,
                args=(rank, world_size, init_file, queue),
            )
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=90)
        for process in processes:
            assert process.exitcode == 0
        results = [queue.get(timeout=10) for _ in range(world_size)]
        errors = [payload for _, status, payload in results if status != "ok"]
        assert not errors, errors[0] if errors else ""
    finally:
        try:
            os.unlink(init_file)
        except FileNotFoundError:
            pass


def test_sequence_parallel_boundary_collectives_route_active_rows() -> None:
    world_size = 2
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    with tempfile.NamedTemporaryFile(delete=False) as handle:
        init_file = handle.name
    try:
        processes = [
            ctx.Process(
                target=_sequence_parallel_boundary_worker,
                args=(rank, world_size, init_file, queue),
            )
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=90)
        for process in processes:
            assert process.exitcode == 0
        results = [queue.get(timeout=10) for _ in range(world_size)]
        errors = [payload for _, status, payload in results if status != "ok"]
        assert not errors, errors[0] if errors else ""
    finally:
        try:
            os.unlink(init_file)
        except FileNotFoundError:
            pass
