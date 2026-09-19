# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Process-group collection for DLLM DP/CP/BP/TP training."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable

import torch.distributed as dist

from dllm_parallel.core.parallel.topology import ParallelPlan, RankAssignment

GroupBuilder = Callable[[ParallelPlan, int, str], Any]


@dataclass(frozen=True)
class DLLMProcessGroupOptions:
    """Typed process-group construction policy."""

    default_timeout_seconds: float | None = None
    group_timeout_seconds: Mapping[str, float] | None = None

    def timeout_for(self, group_name: str) -> timedelta | None:
        value: float | None = None
        if self.group_timeout_seconds is not None:
            value = self.group_timeout_seconds.get(group_name)
        if value is None:
            value = self.default_timeout_seconds
        if value is None:
            return None
        value = float(value)
        if value <= 0:
            raise ValueError("process-group timeout seconds must be positive")
        return timedelta(seconds=value)

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "default_timeout_seconds": self.default_timeout_seconds,
            "group_timeout_seconds": dict(self.group_timeout_seconds or {}),
        }


@dataclass(frozen=True)
class DLLMProcessGroupCollection:
    """Single source of truth for rank-local distributed process groups.

    ``data_parallel_group`` is the expert-data-parallel domain: it varies DP
    while holding EP fixed. ``model_input_group`` varies CP/BP and TP while
    holding DP and EP fixed, so one input is shared only by ranks cooperating
    on the same model invocation. Dense EP gradients are synchronized as a
    staged DP then EP reduction, so the logical dense-DP rank set does not need
    its own NCCL communicator.
    """

    plan: ParallelPlan
    rank: int
    assignment: RankAssignment
    data_parallel_group: Any | None
    optimizer_data_parallel_group: Any | None
    context_block_parallel_group: Any | None
    clean_replica_group: Any | None
    tensor_parallel_group: Any | None
    pipeline_parallel_group: Any | None
    expert_parallel_group: Any | None
    model_input_group: Any | None
    model_parallel_group: Any | None
    optimizer_data_parallel_group_ranks: list[int]
    zero_partitions_sample_parallel: bool
    options: DLLMProcessGroupOptions = DLLMProcessGroupOptions()

    @classmethod
    def from_plan(
        cls,
        plan: ParallelPlan,
        *,
        rank: int | None = None,
        group_builder: GroupBuilder | None = None,
        create_groups: bool | None = None,
        options: DLLMProcessGroupOptions | None = None,
    ) -> "DLLMProcessGroupCollection":
        options = options or DLLMProcessGroupOptions()
        if rank is None:
            rank = _current_rank_or_zero()
        assignment = plan.rank_assignments[int(rank)]
        optimizer_group_ranks = optimizer_data_parallel_group_ranks(plan, assignment)
        if create_groups is None:
            create_groups = dist.is_available() and dist.is_initialized()

        def build(group_attr: str) -> Any | None:
            if not create_groups:
                return None
            if group_builder is not None:
                return group_builder(plan, int(rank), group_attr)
            return current_group_from_plan(
                plan,
                int(rank),
                group_attr,
                timeout=options.timeout_for(group_attr),
            )

        optimizer_group = None
        if create_groups:
            if group_builder is not None:
                optimizer_group = group_builder(
                    plan,
                    int(rank),
                    "optimizer_data_parallel_group",
                )
            else:
                optimizer_group = current_group_from_rank_lists(
                    int(rank),
                    [
                        optimizer_data_parallel_group_ranks(plan, candidate)
                        for candidate in plan.rank_assignments
                    ],
                    timeout=options.timeout_for("optimizer_data_parallel_group"),
                    create_singleton_groups=True,
                )

        data_parallel_group = build("data_parallel_group")
        model_parallel_group = build("model_parallel_group")
        if tuple(assignment.model_input_group) == tuple(
            assignment.model_parallel_group
        ):
            model_input_group = model_parallel_group
        else:
            model_input_group = build("model_input_group")

        return cls(
            plan=plan,
            rank=int(rank),
            assignment=assignment,
            data_parallel_group=data_parallel_group,
            optimizer_data_parallel_group=optimizer_group,
            context_block_parallel_group=build("context_block_parallel_group"),
            clean_replica_group=build("clean_replica_group"),
            tensor_parallel_group=build("tensor_parallel_group"),
            pipeline_parallel_group=build("pipeline_parallel_group"),
            expert_parallel_group=build("expert_parallel_group"),
            model_input_group=model_input_group,
            model_parallel_group=model_parallel_group,
            optimizer_data_parallel_group_ranks=optimizer_group_ranks,
            zero_partitions_sample_parallel=(
                tuple(optimizer_group_ranks) != tuple(assignment.data_parallel_group)
            ),
            options=options,
        )

    def destroy(self) -> None:
        """Destroy non-world groups owned by this collection."""

        if not dist.is_available() or not dist.is_initialized():
            return
        seen: set[int] = set()
        for group in (
            self.data_parallel_group,
            self.optimizer_data_parallel_group,
            self.context_block_parallel_group,
            self.clean_replica_group,
            self.tensor_parallel_group,
            self.pipeline_parallel_group,
            self.expert_parallel_group,
            self.model_input_group,
            self.model_parallel_group,
        ):
            if group is None or group is dist.group.WORLD:
                continue
            group_id = id(group)
            if group_id in seen:
                continue
            seen.add(group_id)
            dist.destroy_process_group(group)

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "data_parallel_group_ranks": self.assignment.data_parallel_group,
            "dense_data_parallel_group_ranks": (
                self.assignment.dense_data_parallel_group
            ),
            "optimizer_data_parallel_group_ranks": (
                self.optimizer_data_parallel_group_ranks
            ),
            "context_block_parallel_group_ranks": (
                self.assignment.context_block_parallel_group
            ),
            "clean_replica_group_ranks": self.assignment.clean_replica_group,
            "tensor_parallel_group_ranks": self.assignment.tensor_parallel_group,
            "pipeline_parallel_group_ranks": self.assignment.pipeline_parallel_group,
            "expert_parallel_group_ranks": self.assignment.expert_parallel_group,
            "model_input_group_ranks": self.assignment.model_input_group,
            "model_parallel_group_ranks": self.assignment.model_parallel_group,
            "zero_partitions_sample_parallel": self.zero_partitions_sample_parallel,
            "options": self.options.to_log_dict(),
        }


def current_group_from_plan(
    plan: ParallelPlan,
    rank: int,
    group_attr: str,
    *,
    timeout: timedelta | None = None,
) -> Any | None:
    return current_group_from_rank_lists(
        rank,
        [getattr(assignment, group_attr) for assignment in plan.rank_assignments],
        timeout=timeout,
    )


def current_group_from_rank_lists(
    rank: int,
    groups: list[list[int]],
    *,
    timeout: timedelta | None = None,
    create_singleton_groups: bool = False,
) -> Any | None:
    current_group = None
    seen: set[tuple[int, ...]] = set()
    for group_ranks in groups:
        group_ranks_tuple = tuple(int(item) for item in group_ranks)
        if group_ranks_tuple in seen:
            continue
        seen.add(group_ranks_tuple)
        if len(group_ranks_tuple) <= 1:
            if rank in group_ranks_tuple and bool(create_singleton_groups):
                kwargs = {
                    "ranks": list(group_ranks_tuple),
                    "use_local_synchronization": True,
                }
                if timeout is not None:
                    kwargs["timeout"] = timeout
                current_group = dist.new_group(**kwargs)
            elif rank in group_ranks_tuple:
                current_group = None
            continue
        # Parallel axes require independent NCCL ordering domains even when an
        # axis happens to contain every rank. Aliasing such an axis to WORLD
        # lets optimizer collectives interleave with CP P2P on one communicator.
        kwargs = {"ranks": list(group_ranks_tuple)}
        if timeout is not None:
            kwargs["timeout"] = timeout
        group = dist.new_group(**kwargs)
        if rank in group_ranks_tuple:
            current_group = group
    return current_group


def optimizer_data_parallel_group_ranks(
    plan: ParallelPlan,
    assignment: RankAssignment,
) -> list[int]:
    return [
        candidate.global_rank
        for candidate in plan.rank_assignments
        if candidate.pipeline_parallel_rank == assignment.pipeline_parallel_rank
        and candidate.tensor_parallel_rank == assignment.tensor_parallel_rank
        and candidate.expert_parallel_rank == assignment.expert_parallel_rank
    ]


def _current_rank_or_zero() -> int:
    if not dist.is_available():
        return 0
    try:
        return int(dist.get_rank())
    except Exception:
        return 0
