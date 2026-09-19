# Copyright 2026 The bdlm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

from dataclasses import dataclass

import pytest
from omegaconf import OmegaConf

import dllm_parallel.core.parallel.runtime as parallel_runtime
from dllm_parallel.core.parallel.runtime import build_parallel_runtime, loss_scale
from dllm_parallel.core.parallel.topology import build_parallel_plan


@dataclass(frozen=True)
class ProfileCase:
    name: str
    world_size: int
    data_parallel_size: int
    context_parallel_size: int
    block_parallel_size: int
    tensor_parallel_size: int
    active_block_mode: str
    kv_backend: str
    expected_layout: str
    expected_loss_scale: float
    uses_context_parallel_attention: bool


PROFILE_DRIVER_CASES = (
    ProfileCase(
        name="dp_tp",
        world_size=4,
        data_parallel_size=2,
        context_parallel_size=1,
        block_parallel_size=1,
        tensor_parallel_size=2,
        active_block_mode="all_blocks",
        kv_backend="replicated",
        expected_layout="dp_x_tp",
        expected_loss_scale=1.0,
        uses_context_parallel_attention=False,
    ),
    ProfileCase(
        name="dp_cp",
        world_size=4,
        data_parallel_size=2,
        context_parallel_size=2,
        block_parallel_size=1,
        tensor_parallel_size=1,
        active_block_mode="all_blocks",
        kv_backend="ring",
        expected_layout="dp_x_cp",
        expected_loss_scale=1.0,
        uses_context_parallel_attention=True,
    ),
    ProfileCase(
        name="dp_bp",
        world_size=4,
        data_parallel_size=2,
        context_parallel_size=1,
        block_parallel_size=2,
        tensor_parallel_size=1,
        active_block_mode="dual_end",
        kv_backend="replicated",
        expected_layout="dp_x_bp",
        expected_loss_scale=2.0,
        uses_context_parallel_attention=False,
    ),
    ProfileCase(
        name="dp_cp_bp",
        world_size=4,
        data_parallel_size=2,
        context_parallel_size=2,
        block_parallel_size=2,
        tensor_parallel_size=1,
        active_block_mode="dual_end",
        kv_backend="ring",
        expected_layout="dp_x_fused_cp_bp",
        expected_loss_scale=2.0,
        uses_context_parallel_attention=True,
    ),
    ProfileCase(
        name="dp_cp_bp_more_bp",
        world_size=4,
        data_parallel_size=1,
        context_parallel_size=2,
        block_parallel_size=4,
        tensor_parallel_size=1,
        active_block_mode="dual_end",
        kv_backend="ring",
        expected_layout="dp_x_fused_cp_bp",
        expected_loss_scale=4.0,
        uses_context_parallel_attention=True,
    ),
)


def test_profile_driver_dp_case_builds_explicit_data_parallel_runtime() -> None:

    config = OmegaConf.create(
        {
            "mode": "train",
            "algo": {"name": "standard_block_diffusion"},
            "model": {"length": 16384},
            "block_size": 32,
            "parallel": {
                "context_parallel_size": 1,
                "block_parallel_size": 1,
                "tensor_parallel_size": 1,
                "active_block_mode": "all_blocks",
                "kv_backend": "replicated",
            },
        }
    )

    runtime = build_parallel_runtime(config)

    assert runtime.enabled
    assert runtime.plan is not None
    assert runtime.local_parallel_size == 1
    assert runtime.model_parallel_size == 1
    assert loss_scale(runtime) == 1.0


@pytest.mark.parametrize("case", PROFILE_DRIVER_CASES, ids=lambda case: case.name)
def test_profile_driver_model_parallel_cases_build_expected_plan(case: ProfileCase) -> None:
    plan = build_parallel_plan(
        num_blocks=512,
        world_size=case.world_size,
        data_parallel_size=case.data_parallel_size,
        context_parallel_size=case.context_parallel_size,
        block_parallel_size=case.block_parallel_size,
        tensor_parallel_size=case.tensor_parallel_size,
    )

    assert plan.layout == case.expected_layout
    assert plan.data_parallel_size == case.data_parallel_size
    assert plan.local_parallel_size == max(
        case.context_parallel_size, case.block_parallel_size
    )
    assert plan.model_parallel_size == (
        plan.local_parallel_size * case.tensor_parallel_size
    )
    assert len(plan.rank_assignments) == case.world_size


@pytest.mark.parametrize("case", PROFILE_DRIVER_CASES, ids=lambda case: case.name)
def test_profile_driver_model_parallel_cases_build_expected_runtime(
    case: ProfileCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(parallel_runtime.dist, "is_available", lambda: True)
    monkeypatch.setattr(parallel_runtime.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(parallel_runtime.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(
        parallel_runtime.dist,
        "get_world_size",
        lambda: case.world_size,
    )
    monkeypatch.setattr(
        parallel_runtime,
        "_new_current_group",
        lambda *args, **kwargs: None,
    )
    config = OmegaConf.create(
        {
            "mode": "train",
            "algo": {"name": "standard_block_diffusion"},
            "model": {"length": 16384},
            "block_size": 32,
            "parallel": {
                "data_parallel_size": case.data_parallel_size,
                "context_parallel_size": case.context_parallel_size,
                "block_parallel_size": case.block_parallel_size,
                "tensor_parallel_size": case.tensor_parallel_size,
                "active_block_mode": case.active_block_mode,
                "kv_backend": case.kv_backend,
            },
        }
    )

    runtime = build_parallel_runtime(config)

    assert runtime.enabled
    assert runtime.plan is not None
    assert runtime.plan.layout == case.expected_layout
    assert runtime.model_parallel_size == (
        max(case.context_parallel_size, case.block_parallel_size)
        * case.tensor_parallel_size
    )
    assert runtime.uses_context_parallel_attention is (
        case.uses_context_parallel_attention
    )
    assert loss_scale(runtime) == case.expected_loss_scale
