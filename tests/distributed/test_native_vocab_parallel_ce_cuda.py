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

from dllm_parallel.core.parallel.tensor_parallel import (
    VocabParallelLinear,
    partition_bounds,
)


pytestmark = pytest.mark.distributed


def _runtime(rank: int, world_size: int) -> SimpleNamespace:
    return SimpleNamespace(
        tensor_parallel_size=world_size,
        tensor_parallel_rank=rank,
        tensor_parallel_group=dist.group.WORLD,
    )


def _worker(rank: int, world_size: int, init_file: str, queue) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
        )
        device = torch.device("cuda", rank)
        generator = torch.Generator(device="cpu").manual_seed(7419)
        num_tokens, hidden_size, vocab_size = 19, 64, 37
        hidden_values = torch.randn(
            num_tokens,
            hidden_size,
            generator=generator,
            dtype=torch.float32,
        ).to(torch.bfloat16)
        weight_values = torch.randn(
            vocab_size,
            hidden_size,
            generator=generator,
            dtype=torch.float32,
        ).to(torch.bfloat16)
        bias_values = torch.randn(
            vocab_size,
            generator=generator,
            dtype=torch.float32,
        ).to(torch.bfloat16)
        labels = (torch.arange(num_tokens, dtype=torch.long) * 7 + 3) % vocab_size
        labels[4] = -100
        labels[13] = -100
        exclude_index = vocab_size - 1

        reference_hidden = hidden_values.float().to(device).requires_grad_(True)
        reference_weight = weight_values.float().to(device).requires_grad_(True)
        reference_bias = bias_values.float().to(device).requires_grad_(True)
        reference_logits = F.linear(
            reference_hidden,
            reference_weight,
            reference_bias,
        )
        excluded = torch.arange(vocab_size, device=device) == exclude_index
        reference_logits = reference_logits.masked_fill(excluded, -float("inf"))
        reference_loss = F.cross_entropy(
            reference_logits,
            labels.to(device),
            ignore_index=-100,
            reduction="sum",
        )
        reference_loss.backward()

        layer = VocabParallelLinear(
            hidden_size,
            vocab_size,
            bias=True,
            tensor_parallel_size=world_size,
            tensor_parallel_rank=rank,
            gather_output=False,
        ).to(device=device, dtype=torch.bfloat16)
        layer.set_tensor_parallel_runtime(_runtime(rank, world_size))
        start, stop = partition_bounds(vocab_size, world_size, rank)
        with torch.no_grad():
            layer.weight.copy_(weight_values[start:stop].to(device))
            assert layer.bias is not None
            layer.bias.copy_(bias_values[start:stop].to(device))
        hidden = hidden_values.to(device).requires_grad_(True)
        loss = layer.parallel_cross_entropy(
            hidden,
            labels.to(device),
            ignore_index=-100,
            exclude_index=exclude_index,
            reduction="sum",
        )
        loss.backward()

        torch.testing.assert_close(loss.float(), reference_loss, atol=2e-2, rtol=2e-3)
        torch.testing.assert_close(
            hidden.grad.float(),
            reference_hidden.grad,
            atol=5e-2,
            rtol=5e-2,
        )
        torch.testing.assert_close(
            layer.weight.grad.float(),
            reference_weight.grad[start:stop],
            atol=5e-2,
            rtol=5e-2,
        )
        torch.testing.assert_close(
            layer.bias.grad.float(),
            reference_bias.grad[start:stop],
            atol=5e-2,
            rtol=5e-2,
        )
        queue.put((rank, "ok", ""))
    except Exception:  # pragma: no cover - child reports failure to parent.
        queue.put((rank, "error", traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        os.environ.pop("RANK", None)
        os.environ.pop("WORLD_SIZE", None)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires two CUDA devices",
)
def test_native_vocab_parallel_ce_matches_dense_cuda() -> None:
    world_size = 2
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    with tempfile.NamedTemporaryFile(delete=False) as handle:
        init_file = handle.name
    try:
        processes = [
            ctx.Process(
                target=_worker,
                args=(rank, world_size, init_file, queue),
            )
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=180)
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
