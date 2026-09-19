# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import argparse
import os
import socket
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import dllm_parallel.core.parallel.deepspeed as deepspeed_strategy
from dllm_parallel.core.parallel.deepspeed import (
    _DLLMDeepSpeedMPU,
    build_dllm_deepspeed_mpu,
    _iter_zero_gradient_tensors,
    sync_model_parallel_zero_gradients,
    sync_expert_parallel_zero_gradients,
    sync_sequence_parallel_zero_gradients,
)
from dllm_parallel.core.optim.zero import (
    configure_deepspeed_zero2_block_loss_scale,
    deepspeed_zero2_block_backward_scale,
)
from dllm_parallel.core.parallel.groups import DLLMProcessGroupCollection
from dllm_parallel.core.parallel.topology import build_parallel_plan


@dataclass
class _MPU:
    data_group: object
    model_group: object
    data_rank: int
    model_rank: int
    data_world: int = 2
    model_world: int = 2

    def get_model_parallel_rank(self) -> int:
        return self.model_rank

    def get_model_parallel_group(self):
        return self.model_group

    def get_model_parallel_world_size(self) -> int:
        return self.model_world

    def get_data_parallel_rank(self) -> int:
        return self.data_rank

    def get_data_parallel_group(self):
        return self.data_group

    def get_data_parallel_world_size(self) -> int:
        return self.data_world


class _ScalarModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))

    def forward(self, coeff: torch.Tensor) -> torch.Tensor:
        return (self.weight * coeff).sum()


def test_dllm_mpu_scopes_zero_to_sample_parallel_groups(monkeypatch) -> None:
    plan = build_parallel_plan(
        num_blocks=8,
        world_size=8,
        data_parallel_size=2,
        context_parallel_size=4,
        block_parallel_size=4,
    )
    created_groups: list[tuple[int, ...]] = []

    def fake_new_group(ranks):
        group = tuple(ranks)
        created_groups.append(group)
        return group

    monkeypatch.setattr(deepspeed_strategy.dist, "get_rank", lambda: 3)
    monkeypatch.setattr(deepspeed_strategy.dist, "get_world_size", lambda: 8)
    monkeypatch.setattr(deepspeed_strategy.dist, "new_group", fake_new_group)

    mpu = _DLLMDeepSpeedMPU.from_plan(plan)

    assert mpu.get_data_parallel_rank() == 3
    assert mpu.get_data_parallel_world_size() == 8
    assert mpu.get_data_parallel_group() == tuple(range(8))
    assert mpu.get_model_parallel_rank() == 3
    assert mpu.get_model_parallel_world_size() == 4
    assert mpu.get_model_parallel_group() == (0, 1, 2, 3)
    assert tuple(range(8)) in created_groups
    assert (0, 1, 2, 3) in created_groups
    assert mpu.zero_partitions_sample_parallel


def test_dllm_mpu_with_tp_uses_full_model_group_but_cp_bp_sync_group(
    monkeypatch,
) -> None:
    plan = build_parallel_plan(
        num_blocks=8,
        world_size=8,
        data_parallel_size=2,
        context_parallel_size=2,
        block_parallel_size=2,
        tensor_parallel_size=2,
    )
    created_groups: list[tuple[int, ...]] = []

    def fake_new_group(ranks):
        group = tuple(ranks)
        created_groups.append(group)
        return group

    monkeypatch.setattr(deepspeed_strategy.dist, "get_rank", lambda: 3)
    monkeypatch.setattr(deepspeed_strategy.dist, "get_world_size", lambda: 8)
    monkeypatch.setattr(deepspeed_strategy.dist, "new_group", fake_new_group)

    mpu = _DLLMDeepSpeedMPU.from_plan(plan)

    assert mpu.get_data_parallel_group() == (1, 3, 5, 7)
    assert mpu.get_data_parallel_world_size() == 4
    assert mpu.get_model_parallel_rank() == 3
    assert mpu.get_model_parallel_world_size() == 4
    assert mpu.get_model_parallel_group() == (0, 1, 2, 3)
    assert mpu.context_block_parallel_group == (1, 3)
    assert mpu.context_block_parallel_world_size == 2
    assert (1, 3) in created_groups
    assert (1, 3, 5, 7) in created_groups
    assert mpu.zero_partitions_sample_parallel


def test_dllm_mpu_with_more_bp_than_cp_uses_cp_ring_group(monkeypatch) -> None:
    plan = build_parallel_plan(
        num_blocks=8,
        world_size=4,
        data_parallel_size=1,
        context_parallel_size=2,
        block_parallel_size=4,
    )
    created_groups: list[tuple[int, ...]] = []

    def fake_new_group(ranks):
        group = tuple(ranks)
        created_groups.append(group)
        return group

    monkeypatch.setattr(deepspeed_strategy.dist, "get_rank", lambda: 2)
    monkeypatch.setattr(deepspeed_strategy.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(deepspeed_strategy.dist, "new_group", fake_new_group)

    mpu = _DLLMDeepSpeedMPU.from_plan(plan)

    assert mpu.get_data_parallel_group() == (0, 1, 2, 3)
    assert mpu.get_model_parallel_rank() == 2
    assert mpu.get_model_parallel_world_size() == 4
    assert mpu.get_model_parallel_group() == (0, 1, 2, 3)
    assert mpu.context_block_parallel_group == (2, 3)
    assert mpu.context_block_parallel_world_size == 2
    assert (2, 3) in created_groups
    assert (0, 1, 2, 3) in created_groups
    assert mpu.zero_partitions_sample_parallel


def test_dllm_mpu_reuses_process_group_collection() -> None:
    plan = build_parallel_plan(
        num_blocks=8,
        world_size=4,
        data_parallel_size=1,
        context_parallel_size=2,
        block_parallel_size=4,
    )
    groups = DLLMProcessGroupCollection(
        plan=plan,
        rank=2,
        assignment=plan.rank_assignments[2],
        data_parallel_group=("regular-dp",),
        optimizer_data_parallel_group=("zero-dp",),
        context_block_parallel_group=("cp-bp",),
        clean_replica_group=("clean",),
        tensor_parallel_group=("tp",),
        pipeline_parallel_group=("pp",),
        expert_parallel_group=("ep",),
        model_input_group=("model-input",),
        model_parallel_group=("model",),
        optimizer_data_parallel_group_ranks=[0, 1, 2, 3],
        zero_partitions_sample_parallel=True,
    )
    runtime = SimpleNamespace(process_groups=groups)

    mpu = build_dllm_deepspeed_mpu(runtime)

    assert mpu.get_data_parallel_group() == ("zero-dp",)
    assert mpu.get_data_parallel_world_size() == 4
    assert mpu.get_model_parallel_group() == ("model",)
    assert mpu.context_block_parallel_group == ("cp-bp",)
    assert mpu.zero_partitions_sample_parallel


def test_block_objective_backward_scale_uses_objective_scale_only() -> None:
    runtime = SimpleNamespace(
        block_parallel_size=8,
        active_block_mode="dual_end",
        process_groups=SimpleNamespace(
            optimizer_data_parallel_group_ranks=list(range(16)),
        ),
    )

    scale = deepspeed_zero2_block_backward_scale(
        runtime,
        objective_scale=None,
    )

    assert scale == 1.0 / 8.0


def test_non_block_objective_keeps_unscaled_backward() -> None:
    scale = deepspeed_zero2_block_backward_scale(
        SimpleNamespace(block_parallel_size=8, active_block_mode="all_blocks"),
        objective_scale=None,
    )

    assert scale == 1.0


def test_block_objective_scale_uses_deepspeed_optimizer_boundary() -> None:
    scales: list[float] = []
    zero_optimizer = SimpleNamespace(override_loss_scale=scales.append)

    configured = configure_deepspeed_zero2_block_loss_scale(
        zero_optimizer,
        backward_scale=0.125,
    )

    assert configured
    assert scales == [0.125]


def test_ep1_reuses_model_parallel_group_for_model_input_broadcasts() -> None:
    plan = build_parallel_plan(
        num_blocks=8,
        world_size=4,
        data_parallel_size=1,
        context_parallel_size=4,
        block_parallel_size=4,
    )
    built: list[str] = []

    def build_group(_plan, _rank, group_name):
        built.append(group_name)
        return object()

    groups = DLLMProcessGroupCollection.from_plan(
        plan,
        rank=0,
        group_builder=build_group,
        create_groups=True,
    )

    assert groups.model_input_group is groups.model_parallel_group
    assert "model_input_group" not in built


def test_iter_zero_gradient_tensors_flattens_nested_averaged_gradients() -> None:
    first = torch.tensor([1.0])
    second = torch.tensor([2.0])
    third = torch.tensor([3.0])
    optimizer = SimpleNamespace(
        averaged_gradients={
            0: [first, None, (second,)],
            1: {"partition": third},
        }
    )

    assert list(_iter_zero_gradient_tensors(optimizer)) == [first, second, third]


def test_expert_parallel_zero_sync_averages_dense_and_normalizes_experts(
    monkeypatch,
) -> None:
    dense = torch.tensor([2.0])
    expert = torch.tensor([6.0])
    zero_optimizer = SimpleNamespace(
        averaged_gradients=[[dense], [expert]],
        optimizer=SimpleNamespace(
            param_groups=[{}, {"_dllm_expert_parallel_sharded": True}]
        ),
    )
    calls: list[tuple[object, bool]] = []

    class Work:
        def block_current_stream(self) -> None:
            return None

        def wait(self) -> None:
            return None

    def fake_all_reduce_coalesced(tensors, *, op, group, async_op):
        del op
        calls.append((group, async_op))
        for tensor in tensors:
            tensor.mul_(3.0)
        return Work()

    monkeypatch.setattr(
        deepspeed_strategy.dist,
        "all_reduce_coalesced",
        fake_all_reduce_coalesced,
    )

    synchronized = sync_expert_parallel_zero_gradients(
        zero_optimizer=zero_optimizer,
        expert_parallel_group="ep",
        expert_parallel_world_size=2,
    )

    assert synchronized == 2
    assert calls == [("ep", True)]
    assert dense.item() == pytest.approx(3.0)
    assert expert.item() == pytest.approx(3.0)


def test_sequence_parallel_zero_sync_sums_only_replicated_gradient_groups(
    monkeypatch,
) -> None:
    dense = torch.tensor([2.0])
    sequence_parallel = torch.tensor([3.0])
    zero_optimizer = SimpleNamespace(
        averaged_gradients=[[dense], [sequence_parallel]],
        optimizer=SimpleNamespace(
            param_groups=[{}, {"_dllm_sequence_parallel_replicated": True}]
        ),
    )
    calls: list[tuple[object, bool]] = []

    class Work:
        def block_current_stream(self) -> None:
            return None

        def wait(self) -> None:
            return None

    def fake_all_reduce_coalesced(tensors, *, op, group, async_op):
        assert op == dist.ReduceOp.SUM
        calls.append((group, async_op))
        for tensor in tensors:
            tensor.mul_(2.0)
        return Work()

    monkeypatch.setattr(
        deepspeed_strategy.dist,
        "all_reduce_coalesced",
        fake_all_reduce_coalesced,
    )

    synchronized = sync_sequence_parallel_zero_gradients(
        zero_optimizer=zero_optimizer,
        tensor_parallel_group="tp",
        tensor_parallel_world_size=2,
    )

    assert synchronized == 1
    assert calls == [("tp", True)]
    assert dense.item() == pytest.approx(2.0)
    assert sequence_parallel.item() == pytest.approx(6.0)


def _build_groups(rank: int):
    data_group = None
    model_group = None
    for ranks in ([0, 2], [1, 3]):
        group = dist.new_group(ranks=list(ranks))
        if rank in ranks:
            data_group = group
    for ranks in ([0, 1], [2, 3]):
        group = dist.new_group(ranks=list(ranks))
        if rank in ranks:
            model_group = group
    return data_group, model_group


def _deepspeed_scalar_worker(rank: int, port: int, queue) -> None:
    try:
        import deepspeed
    except ModuleNotFoundError:
        queue.put(("skip", "deepspeed is not installed"))
        return

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    os.environ["LOCAL_RANK"] = str(rank)
    dist.init_process_group("nccl", rank=rank, world_size=4)
    torch.cuda.set_device(rank)
    try:
        data_group, model_group = _build_groups(rank)
        local_rank = rank % 2
        data_rank = rank // 2
        model = _ScalarModel().cuda(rank)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        config = {
            "train_batch_size": 2,
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "zero_allow_untested_optimizer": True,
            "zero_optimization": {"stage": 2},
            "fp16": {"enabled": False},
            "bf16": {"enabled": False},
        }
        engine, _, _, _ = deepspeed.initialize(
            args=argparse.Namespace(device_rank=rank, local_rank=rank),
            model=model,
            model_parameters=model.parameters(),
            optimizer=optimizer,
            config=config,
            mpu=_MPU(
                data_group=data_group,
                model_group=model_group,
                data_rank=data_rank,
                model_rank=local_rank,
            ),
            dist_init_required=False,
        )
        coeff = torch.tensor(
            [2.0 if local_rank == 0 else 6.0],
            device=engine.device,
        )
        engine.backward(engine(coeff))
        sync_model_parallel_zero_gradients(
            zero_optimizer=engine.optimizer,
            model_parallel_group=model_group,
            model_parallel_world_size=2,
        )
        engine.step()
        value = engine.module.weight.detach().float().clone()
        gathered = [torch.zeros_like(value) for _ in range(4)]
        dist.all_gather(gathered, value)
        if rank == 0:
            queue.put(("ok", [float(t.cpu().item()) for t in gathered]))
    except Exception as exc:  # pragma: no cover - child reports to parent.
        queue.put(("error", repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 4,
    reason="DeepSpeed ZeRO-2 regression requires four CUDA devices",
)
def test_model_parallel_zero_grad_sync_keeps_replicated_params_equal() -> None:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    processes = [
        ctx.Process(
            target=_deepspeed_scalar_worker,
            args=(rank, port, queue),
        )
        for rank in range(4)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=90)
    for process in processes:
        assert process.exitcode == 0

    status, payload = queue.get(timeout=10)
    if status == "skip":
        pytest.skip(payload)
    assert status == "ok", payload
    torch.testing.assert_close(
        torch.tensor(payload),
        torch.full((4,), 0.6),
        atol=1e-6,
        rtol=1e-6,
    )


def _deepspeed_sequence_parallel_scalar_worker(rank: int, port: int, queue) -> None:
    try:
        import deepspeed
    except ModuleNotFoundError:
        queue.put(("skip", "deepspeed is not installed"))
        return

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    os.environ["LOCAL_RANK"] = str(rank)
    dist.init_process_group("nccl", rank=rank, world_size=4)
    torch.cuda.set_device(rank)
    try:
        data_group, tensor_group = _build_groups(rank)
        tensor_rank = rank % 2
        data_rank = rank // 2
        model = _ScalarModel().cuda(rank)
        optimizer = torch.optim.SGD(
            [
                {
                    "params": list(model.parameters()),
                    "_dllm_sequence_parallel_replicated": True,
                }
            ],
            lr=0.1,
        )
        config = {
            "train_batch_size": 2,
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "zero_allow_untested_optimizer": True,
            "zero_optimization": {"stage": 2},
            "fp16": {"enabled": False},
            "bf16": {"enabled": False},
        }
        engine, _, _, _ = deepspeed.initialize(
            args=argparse.Namespace(device_rank=rank, local_rank=rank),
            model=model,
            model_parameters=optimizer.param_groups,
            optimizer=optimizer,
            config=config,
            mpu=_MPU(
                data_group=data_group,
                model_group=tensor_group,
                data_rank=data_rank,
                model_rank=tensor_rank,
            ),
            dist_init_required=False,
        )
        # The two TP ranks own disjoint token rows. ZeRO averages each row's
        # gradient over data replicas; sequence parallelism must then sum the
        # two row contributions before the optimizer update.
        coeff = torch.tensor(
            [2.0 if tensor_rank == 0 else 6.0],
            device=engine.device,
        )
        engine.backward(engine(coeff))
        sync_sequence_parallel_zero_gradients(
            zero_optimizer=engine.optimizer,
            tensor_parallel_group=tensor_group,
            tensor_parallel_world_size=2,
        )
        engine.step()
        value = engine.module.weight.detach().float().clone()
        gathered = [torch.zeros_like(value) for _ in range(4)]
        dist.all_gather(gathered, value)
        if rank == 0:
            queue.put(("ok", [float(t.cpu().item()) for t in gathered]))
    except Exception as exc:  # pragma: no cover - child reports to parent.
        queue.put(("error", repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 4,
    reason="DeepSpeed ZeRO-2 sequence-parallel regression requires four CUDA devices",
)
def test_sequence_parallel_zero_grad_sync_matches_full_token_gradient() -> None:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    processes = [
        ctx.Process(
            target=_deepspeed_sequence_parallel_scalar_worker,
            args=(rank, port, queue),
        )
        for rank in range(4)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=90)
    for process in processes:
        assert process.exitcode == 0

    status, payload = queue.get(timeout=10)
    if status == "skip":
        pytest.skip(payload)
    assert status == "ok", payload
    torch.testing.assert_close(
        torch.tensor(payload),
        torch.full((4,), 0.2),
        atol=1e-6,
        rtol=1e-6,
    )
