# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import os
import tempfile
import traceback
from multiprocessing import get_context
from types import SimpleNamespace

import torch
import torch.distributed as dist

from dllm_parallel.training.optimizer_setup import _clip_grad_norm_global


def _global_clip_worker(rank: int, world_size: int, init_file: str, queue) -> None:
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
        )
        tensor_parallel_rank = rank % 2
        local_parallel_rank = rank // 2
        runtime = SimpleNamespace(
            model_parallel_group=dist.group.WORLD,
            model_parallel_size=world_size,
            local_parallel_rank=local_parallel_rank,
            tensor_parallel_rank=tensor_parallel_rank,
            expert_parallel_rank=0,
        )

        replicated = torch.nn.Parameter(torch.zeros(1))
        replicated.grad = torch.tensor([3.0])
        tensor_shard = torch.nn.Parameter(torch.zeros(1))
        tensor_shard._dllm_tensor_parallel_sharded = True
        shard_gradient = 4.0 if tensor_parallel_rank == 0 else 12.0
        tensor_shard.grad = torch.tensor([shard_gradient])

        total_norm = _clip_grad_norm_global(
            [replicated, tensor_shard],
            max_norm=6.5,
            runtime=runtime,
        )
        expected_scale = 6.5 / (13.0 + 1.0e-6)
        torch.testing.assert_close(torch.tensor(total_norm), torch.tensor(13.0))
        torch.testing.assert_close(
            replicated.grad,
            torch.tensor([3.0 * expected_scale]),
        )
        torch.testing.assert_close(
            tensor_shard.grad,
            torch.tensor([shard_gradient * expected_scale]),
        )
        queue.put((rank, "ok", ""))
    except Exception:  # pragma: no cover - child reports failure to parent.
        queue.put((rank, "error", traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_global_gradient_clipping_counts_cp_replicas_once_and_all_tp_shards() -> None:
    world_size = 4
    context = get_context("spawn")
    queue = context.Queue()
    with tempfile.NamedTemporaryFile(delete=False) as handle:
        init_file = handle.name
    try:
        processes = [
            context.Process(
                target=_global_clip_worker,
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
