# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Compose model-family schedules with rank topology plans."""

from __future__ import annotations

from dataclasses import dataclass

from dllm_parallel.core.parallel.topology import ParallelPlan, build_parallel_plan
from dllm_parallel.core.models.registry import validate_parallel_spec
from dllm_parallel.core.models.contracts import ModelFamilySpec
from dllm_parallel.core.specs import DiffusionSchedule, ParallelSpec


@dataclass(frozen=True)
class DistributedTrainingPlan:
    model: ModelFamilySpec
    parallel: ParallelSpec
    schedule: DiffusionSchedule
    topology: ParallelPlan | None


def build_distributed_training_plan(
    model: ModelFamilySpec,
    parallel: ParallelSpec,
    schedule: DiffusionSchedule,
    *,
    world_size: int,
) -> DistributedTrainingPlan:
    """Build a rank plan when the schedule has block ownership semantics."""

    validate_parallel_spec(model, parallel)
    expected_world_size = parallel.data_parallel_size * parallel.model_parallel_size
    if int(world_size) != int(expected_world_size):
        raise ValueError(
            "world_size must equal data_parallel_size * model_parallel_size "
            "for the requested ParallelSpec"
        )

    topology = None
    if schedule.num_blocks is not None:
        _validate_block_schedule_parallelism(model, parallel, schedule)
        topology = build_parallel_plan(
            num_blocks=schedule.num_blocks,
            world_size=world_size,
            data_parallel_size=parallel.data_parallel_size,
            context_parallel_size=parallel.context_parallel_size,
            block_parallel_size=parallel.block_parallel_size,
            tensor_parallel_size=parallel.tensor_parallel_size,
            pipeline_parallel_size=parallel.pipeline_parallel_size,
            expert_parallel_size=parallel.expert_parallel_size,
        )
    return DistributedTrainingPlan(
        model=model,
        parallel=parallel,
        schedule=schedule,
        topology=topology,
    )


def _validate_block_schedule_parallelism(
    model: ModelFamilySpec,
    parallel: ParallelSpec,
    schedule: DiffusionSchedule,
) -> None:
    objective = schedule.objective
    if int(parallel.block_parallel_size) <= 1:
        return
    if objective is None or not objective.supports_fused_cp_bp:
        raise ValueError(
            "block_parallel_size > 1 requires an exact teacher-forced "
            "block-denoising objective"
        )
