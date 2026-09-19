# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from dllm_parallel.training.block_diffusion_trainer import (
    _is_data_sample_source,
    _local_masked_token_accuracy,
    _local_masked_token_metrics,
    _step_throughput_metrics,
    _training_throughput_metrics,
)
from dllm_parallel.core.parallel.topology import build_parallel_plan


def test_masked_token_accuracy_matches_owned_target_rows() -> None:
    model = SimpleNamespace(
        output_head=torch.nn.Linear(4, 4, bias=False),
        module=None,
    )
    with torch.no_grad():
        model.output_head.weight.copy_(torch.eye(4))
    labels = torch.tensor([[-100, 0, 2, 1]])
    positions = torch.tensor([1, 2, 3])
    hidden = torch.eye(4)[torch.tensor([0, 2, 1])].unsqueeze(0)

    correct, count = _local_masked_token_accuracy(
        model=model,
        output=(hidden, positions),
        labels=labels,
        mask_token_id=3,
        token_tile_size=2,
    )

    assert (correct, count) == (3, 3)


def test_masked_token_accuracy_supports_sequence_parallel_positions() -> None:
    model = SimpleNamespace(
        output_head=torch.nn.Linear(4, 4, bias=False),
        module=None,
    )
    with torch.no_grad():
        model.output_head.weight.copy_(torch.eye(4))
    labels = torch.tensor([[-100, 1, -100], [-100, -100, 2]])
    positions = torch.tensor([[0, 1], [1, 2]])
    hidden = torch.eye(4)[torch.tensor([1, 2])]

    correct, count = _local_masked_token_accuracy(
        model=model,
        output=(hidden, positions),
        labels=labels,
        mask_token_id=3,
        token_tile_size=1,
    )

    assert (correct, count) == (2, 2)


def test_masked_token_accuracy_sums_disjoint_cp_target_rows() -> None:
    model = SimpleNamespace(
        output_head=torch.nn.Linear(4, 4, bias=False),
        module=None,
    )
    with torch.no_grad():
        model.output_head.weight.copy_(torch.eye(4))
    labels = torch.tensor([[-100, -100, 2, 1]])

    first = _local_masked_token_accuracy(
        model=model,
        output=(torch.eye(4)[torch.tensor([0, 1])].unsqueeze(0), torch.tensor([0, 1])),
        labels=labels,
        mask_token_id=3,
        token_tile_size=2,
    )
    second = _local_masked_token_accuracy(
        model=model,
        output=(torch.eye(4)[torch.tensor([2, 1])].unsqueeze(0), torch.tensor([2, 3])),
        labels=labels,
        mask_token_id=3,
        token_tile_size=2,
    )

    assert first == (0, 0)
    assert second == (2, 2)
    assert tuple(sum(values) for values in zip(first, second, strict=True)) == (2, 2)


def test_masked_token_metrics_reports_tiled_cross_entropy() -> None:
    model = SimpleNamespace(
        output_head=torch.nn.Linear(4, 4, bias=False),
        final_logit_softcap=None,
        module=None,
    )
    with torch.no_grad():
        model.output_head.weight.copy_(torch.eye(4))
    labels = torch.tensor([[-100, 0, 2, 1]])
    positions = torch.tensor([1, 2, 3])
    hidden = torch.eye(4)[torch.tensor([0, 2, 1])].unsqueeze(0)

    correct, count, negative_log_likelihood = _local_masked_token_metrics(
        model=model,
        output=(hidden, positions),
        labels=labels,
        mask_token_id=3,
        token_tile_size=2,
    )

    expected = 3.0 * float(torch.log(torch.tensor(torch.e + 2.0)) - 1.0)
    assert (correct, count) == (3, 3)
    assert negative_log_likelihood == pytest.approx(expected)


def test_data_sample_source_counts_independent_expert_parallel_samples() -> None:
    plan = build_parallel_plan(
        num_blocks=8,
        world_size=8,
        data_parallel_size=2,
        context_parallel_size=2,
        block_parallel_size=2,
        tensor_parallel_size=1,
        expert_parallel_size=2,
    )
    runtime = SimpleNamespace(plan=plan)

    sources = [
        rank
        for rank in range(plan.world_size)
        if _is_data_sample_source(runtime=runtime, rank=rank)
    ]

    assert len(sources) == plan.data_parallel_size * plan.expert_parallel_size
    assert {
        (
            plan.rank_assignments[rank].data_parallel_rank,
            plan.rank_assignments[rank].expert_parallel_rank,
        )
        for rank in sources
    } == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_step_throughput_distinguishes_device_and_unique_token_rates() -> None:
    metrics = _step_throughput_metrics(
        input_tokens=8000,
        valid_tokens=2000,
        active_tokens=500,
        elapsed_ms=100.0,
        world_size=8,
        data_parallel_size=2,
    )

    assert metrics["steps_per_s"] == pytest.approx(10.0)
    assert metrics["input_tokens_per_s_global"] == pytest.approx(80_000.0)
    assert metrics["input_tokens_per_s_per_gpu"] == pytest.approx(10_000.0)
    assert metrics["supervised_tokens_per_s_global"] == pytest.approx(20_000.0)
    assert metrics["supervised_tokens_per_s_per_gpu"] == pytest.approx(2_500.0)
    assert metrics["tokens_per_s_per_gpu"] == pytest.approx(10_000.0)
    assert metrics["tokens_per_s_global"] == pytest.approx(80_000.0)
    assert metrics["unique_tokens_per_s_global"] == pytest.approx(20_000.0)
    assert metrics["active_tokens_per_s_global"] == pytest.approx(5_000.0)
    assert metrics["active_token_fraction"] == pytest.approx(0.25)
    assert metrics["supervised_token_fraction"] == pytest.approx(0.25)


def test_training_throughput_uses_only_source_rank_token_counts() -> None:
    rank_metrics = [
        {
            "measured_steps": 2,
            "measured_input_tokens": 8000 if rank == 0 else 0,
            "measured_valid_tokens": 2000 if rank == 0 else 0,
            "measured_active_tokens": 500 if rank == 0 else 0,
        }
        for rank in range(4)
    ]

    metrics = _training_throughput_metrics(
        rank_metrics=rank_metrics,
        elapsed_ms=50.0,
        world_size=4,
        data_parallel_size=1,
    )

    assert metrics["step_time_ms"] == pytest.approx(50.0)
    assert metrics["input_tokens_per_s_global"] == pytest.approx(80_000.0)
    assert metrics["unique_tokens_per_s_global"] == pytest.approx(20_000.0)
    assert metrics["measured_input_tokens"] == 8000.0
    assert metrics["measured_valid_tokens"] == 2000.0
