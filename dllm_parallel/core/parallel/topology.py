# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Rank topology for the v1 BDLM parallel training target."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

from dllm_parallel.core.schedules.block import BlockSchedule, build_block_schedule


@dataclass(frozen=True)
class RankAssignment:
    """Logical assignment for one global rank."""

    global_rank: int
    data_parallel_rank: int
    local_parallel_rank: int
    context_parallel_rank: int
    block_parallel_rank: int
    tensor_parallel_rank: int
    pipeline_parallel_rank: int
    expert_parallel_rank: int
    data_parallel_group: list[int]
    dense_data_parallel_group: list[int]
    context_block_parallel_group: list[int]
    clean_replica_group: list[int]
    tensor_parallel_group: list[int]
    pipeline_parallel_group: list[int]
    expert_parallel_group: list[int]
    model_input_group: list[int]
    model_parallel_group: list[int]


@dataclass(frozen=True)
class ParallelPlan:
    """Serializable v1 data/context/block/tensor-parallel plan."""

    world_size: int
    data_parallel_size: int
    context_parallel_size: int
    block_parallel_size: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    expert_parallel_size: int
    local_parallel_size: int
    model_parallel_size: int
    layout: str
    placement: str
    rank_assignments: list[RankAssignment]
    block_schedule: BlockSchedule

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_parallel_plan(
    *,
    num_blocks: int,
    world_size: int,
    data_parallel_size: int,
    context_parallel_size: int,
    block_parallel_size: int,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    expert_parallel_size: int = 1,
    node_size: int | None = None,
    placement_policy: str = "auto",
) -> ParallelPlan:
    """Build a DP x PP x local(CP/BP) x TP x EP plan.

    The CP/BP dimension is a local axis. A fused CP/BP run may use one CP
    ring across the whole BP group (``context_parallel_size == block_parallel_size``)
    or multiple replicated CP rings when ``block_parallel_size`` is a multiple
    of ``context_parallel_size``. The latter trades bounded clean-KV replication
    for more active-block parallelism.

    * BP-only: ``context_parallel_size == 1``;
    * CP-only: ``block_parallel_size == 1``;
    * fused CP/BP: ``block_parallel_size >= context_parallel_size`` and
      ``block_parallel_size % context_parallel_size == 0``.
    The default placement is topology-aware for the common torchrun rank order
    where ranks from one node are contiguous. Tensor-parallel groups are the
    fastest-changing axis, so TP collectives stay inside a node whenever
    ``tensor_parallel_size <= node_size``. The local CP/BP axis is next, which
    keeps fused CP rings node-local when the node has enough GPUs for
    ``context_parallel_size * tensor_parallel_size``. If a model-parallel group
    is larger than one node, it is packed over the minimum number of contiguous
    nodes while preserving that axis order.

    ``inter_node_cp`` is an explicit profiling placement for complete-node
    launches. It keeps TP groups node-local while striping each matched CP/BP
    group across nodes. This isolates inter-node K/V transport without changing
    the logical DP/TP/CP/BP topology.
    """

    _require_positive("world_size", world_size)
    _require_positive("data_parallel_size", data_parallel_size)
    _require_positive("context_parallel_size", context_parallel_size)
    _require_positive("block_parallel_size", block_parallel_size)
    _require_positive("tensor_parallel_size", tensor_parallel_size)
    _require_positive("pipeline_parallel_size", pipeline_parallel_size)
    _require_positive("expert_parallel_size", expert_parallel_size)
    if placement_policy not in {"auto", "inter_node_cp"}:
        raise ValueError("placement_policy must be 'auto' or 'inter_node_cp'")
    if placement_policy == "inter_node_cp" and (
        context_parallel_size <= 1
        or block_parallel_size not in {1, context_parallel_size}
    ):
        raise ValueError(
            "inter_node_cp placement requires CP-only or matched CP/BP with "
            "context_parallel_size > 1"
        )

    if not _supported_local_layout(context_parallel_size, block_parallel_size):
        raise ValueError(
            "v1 supports BP-only (context_parallel_size=1), CP-only "
            "(block_parallel_size=1), or fused CP/BP with "
            "block_parallel_size >= context_parallel_size and "
            "block_parallel_size divisible by context_parallel_size"
        )

    local_parallel_size = _local_parallel_size(
        context_parallel_size,
        block_parallel_size,
    )
    model_parallel_size = (
        pipeline_parallel_size
        * local_parallel_size
        * tensor_parallel_size
        * expert_parallel_size
    )
    expected_world_size = data_parallel_size * model_parallel_size
    if world_size != expected_world_size:
        raise ValueError(
            "world_size must equal data_parallel_size * "
            "pipeline_parallel_size * local_parallel_size * "
            "tensor_parallel_size * expert_parallel_size"
        )
    if node_size is not None:
        _require_positive("node_size", int(node_size))

    physical_rank_by_logical_rank = _physical_rank_by_logical_rank(
        world_size=world_size,
        data_parallel_size=data_parallel_size,
        local_parallel_size=local_parallel_size,
        model_parallel_size=model_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        expert_parallel_size=expert_parallel_size,
        node_size=node_size,
        placement_policy=placement_policy,
    )
    logical_rank_by_physical_rank = _invert_rank_permutation(
        physical_rank_by_logical_rank
    )
    rank_assignments = [
        _rank_assignment(
            global_rank=physical_rank,
            logical_rank=logical_rank_by_physical_rank[physical_rank],
            physical_rank_by_logical_rank=physical_rank_by_logical_rank,
            data_parallel_size=data_parallel_size,
            local_parallel_size=local_parallel_size,
            model_parallel_size=model_parallel_size,
            context_parallel_size=context_parallel_size,
            block_parallel_size=block_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            expert_parallel_size=expert_parallel_size,
        )
        for physical_rank in range(world_size)
    ]
    block_schedule = build_block_schedule(
        num_blocks=num_blocks,
        block_parallel_size=block_parallel_size,
        context_parallel_size=context_parallel_size,
    )

    return ParallelPlan(
        world_size=world_size,
        data_parallel_size=data_parallel_size,
        context_parallel_size=context_parallel_size,
        block_parallel_size=block_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        expert_parallel_size=expert_parallel_size,
        local_parallel_size=local_parallel_size,
        model_parallel_size=model_parallel_size,
        layout=_layout_name(
            context_parallel_size,
            block_parallel_size,
            tensor_parallel_size,
            pipeline_parallel_size,
            expert_parallel_size,
        ),
        placement=_placement_name(
            world_size=world_size,
            data_parallel_size=data_parallel_size,
            model_parallel_size=model_parallel_size,
            node_size=node_size,
            context_parallel_size=context_parallel_size,
            local_parallel_size=local_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            expert_parallel_size=expert_parallel_size,
            placement_policy=placement_policy,
        ),
        rank_assignments=rank_assignments,
        block_schedule=block_schedule,
    )


def _rank_assignment(
    *,
    global_rank: int,
    logical_rank: int,
    physical_rank_by_logical_rank: tuple[int, ...],
    data_parallel_size: int,
    local_parallel_size: int,
    model_parallel_size: int,
    context_parallel_size: int,
    block_parallel_size: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    expert_parallel_size: int,
) -> RankAssignment:
    data_parallel_rank = logical_rank // model_parallel_size
    model_parallel_rank = logical_rank % model_parallel_size
    per_pipeline = local_parallel_size * tensor_parallel_size * expert_parallel_size
    pipeline_parallel_rank = model_parallel_rank // per_pipeline
    pipeline_remainder = model_parallel_rank % per_pipeline
    per_local = tensor_parallel_size * expert_parallel_size
    local_parallel_rank = pipeline_remainder // per_local
    local_remainder = pipeline_remainder % per_local
    tensor_parallel_rank = local_remainder // expert_parallel_size
    expert_parallel_rank = local_remainder % expert_parallel_size
    context_parallel_rank = _context_parallel_rank(
        local_parallel_rank=local_parallel_rank,
        context_parallel_size=context_parallel_size,
    )
    block_parallel_rank = _block_parallel_rank(
        local_parallel_rank=local_parallel_rank,
        block_parallel_size=block_parallel_size,
    )
    data_parallel_group = [
        _physical_rank_from_coords(
            data_parallel_rank=dp_rank,
            pipeline_parallel_rank=pipeline_parallel_rank,
            local_parallel_rank=local_parallel_rank,
            tensor_parallel_rank=tensor_parallel_rank,
            expert_parallel_rank=expert_parallel_rank,
            physical_rank_by_logical_rank=physical_rank_by_logical_rank,
            model_parallel_size=model_parallel_size,
            per_pipeline=per_pipeline,
            per_local=per_local,
            expert_parallel_size=expert_parallel_size,
        )
        for dp_rank in range(data_parallel_size)
    ]
    dense_data_parallel_group = [
        _physical_rank_from_coords(
            data_parallel_rank=dp_rank,
            pipeline_parallel_rank=pipeline_parallel_rank,
            local_parallel_rank=local_parallel_rank,
            tensor_parallel_rank=tensor_parallel_rank,
            expert_parallel_rank=ep_rank,
            physical_rank_by_logical_rank=physical_rank_by_logical_rank,
            model_parallel_size=model_parallel_size,
            per_pipeline=per_pipeline,
            per_local=per_local,
            expert_parallel_size=expert_parallel_size,
        )
        for dp_rank in range(data_parallel_size)
        for ep_rank in range(expert_parallel_size)
    ]
    context_block_parallel_group = list(
        _physical_rank_from_coords(
            data_parallel_rank=data_parallel_rank,
            pipeline_parallel_rank=pipeline_parallel_rank,
            local_parallel_rank=local_rank,
            tensor_parallel_rank=tensor_parallel_rank,
            expert_parallel_rank=expert_parallel_rank,
            physical_rank_by_logical_rank=physical_rank_by_logical_rank,
            model_parallel_size=model_parallel_size,
            per_pipeline=per_pipeline,
            per_local=per_local,
            expert_parallel_size=expert_parallel_size,
        )
        for local_rank in _context_block_local_ranks(
            local_parallel_rank=local_parallel_rank,
            local_parallel_size=local_parallel_size,
            context_parallel_size=context_parallel_size,
            block_parallel_size=block_parallel_size,
        )
    )
    clean_replica_group = list(
        _physical_rank_from_coords(
            data_parallel_rank=data_parallel_rank,
            pipeline_parallel_rank=pipeline_parallel_rank,
            local_parallel_rank=local_rank,
            tensor_parallel_rank=tensor_parallel_rank,
            expert_parallel_rank=expert_parallel_rank,
            physical_rank_by_logical_rank=physical_rank_by_logical_rank,
            model_parallel_size=model_parallel_size,
            per_pipeline=per_pipeline,
            per_local=per_local,
            expert_parallel_size=expert_parallel_size,
        )
        for local_rank in _clean_replica_local_ranks(
            local_parallel_rank=local_parallel_rank,
            local_parallel_size=local_parallel_size,
            context_parallel_size=context_parallel_size,
            block_parallel_size=block_parallel_size,
        )
    )
    tensor_parallel_group = list(
        _physical_rank_from_coords(
            data_parallel_rank=data_parallel_rank,
            pipeline_parallel_rank=pipeline_parallel_rank,
            local_parallel_rank=local_parallel_rank,
            tensor_parallel_rank=tp_rank,
            expert_parallel_rank=expert_parallel_rank,
            physical_rank_by_logical_rank=physical_rank_by_logical_rank,
            model_parallel_size=model_parallel_size,
            per_pipeline=per_pipeline,
            per_local=per_local,
            expert_parallel_size=expert_parallel_size,
        )
        for tp_rank in range(tensor_parallel_size)
    )
    pipeline_parallel_group = list(
        _physical_rank_from_coords(
            data_parallel_rank=data_parallel_rank,
            pipeline_parallel_rank=pp_rank,
            local_parallel_rank=local_parallel_rank,
            tensor_parallel_rank=tensor_parallel_rank,
            expert_parallel_rank=expert_parallel_rank,
            physical_rank_by_logical_rank=physical_rank_by_logical_rank,
            model_parallel_size=model_parallel_size,
            per_pipeline=per_pipeline,
            per_local=per_local,
            expert_parallel_size=expert_parallel_size,
        )
        for pp_rank in range(pipeline_parallel_size)
    )
    expert_parallel_group = list(
        _physical_rank_from_coords(
            data_parallel_rank=data_parallel_rank,
            pipeline_parallel_rank=pipeline_parallel_rank,
            local_parallel_rank=local_parallel_rank,
            tensor_parallel_rank=tensor_parallel_rank,
            expert_parallel_rank=ep_rank,
            physical_rank_by_logical_rank=physical_rank_by_logical_rank,
            model_parallel_size=model_parallel_size,
            per_pipeline=per_pipeline,
            per_local=per_local,
            expert_parallel_size=expert_parallel_size,
        )
        for ep_rank in range(expert_parallel_size)
    )
    model_input_group = [
        _physical_rank_from_coords(
            data_parallel_rank=data_parallel_rank,
            pipeline_parallel_rank=pipeline_parallel_rank,
            local_parallel_rank=local_rank,
            tensor_parallel_rank=tp_rank,
            expert_parallel_rank=expert_parallel_rank,
            physical_rank_by_logical_rank=physical_rank_by_logical_rank,
            model_parallel_size=model_parallel_size,
            per_pipeline=per_pipeline,
            per_local=per_local,
            expert_parallel_size=expert_parallel_size,
        )
        for local_rank in range(local_parallel_size)
        for tp_rank in range(tensor_parallel_size)
    ]
    model_parallel_group = [
        physical_rank_by_logical_rank[logical_model_rank]
        for logical_model_rank in range(
            data_parallel_rank * model_parallel_size,
            (data_parallel_rank + 1) * model_parallel_size,
        )
    ]

    return RankAssignment(
        global_rank=global_rank,
        data_parallel_rank=data_parallel_rank,
        local_parallel_rank=local_parallel_rank,
        context_parallel_rank=context_parallel_rank,
        block_parallel_rank=block_parallel_rank,
        tensor_parallel_rank=tensor_parallel_rank,
        pipeline_parallel_rank=pipeline_parallel_rank,
        expert_parallel_rank=expert_parallel_rank,
        data_parallel_group=data_parallel_group,
        dense_data_parallel_group=dense_data_parallel_group,
        context_block_parallel_group=context_block_parallel_group,
        clean_replica_group=clean_replica_group,
        tensor_parallel_group=tensor_parallel_group,
        pipeline_parallel_group=pipeline_parallel_group,
        expert_parallel_group=expert_parallel_group,
        model_input_group=model_input_group,
        model_parallel_group=model_parallel_group,
    )


def _physical_rank_from_coords(
    *,
    physical_rank_by_logical_rank: tuple[int, ...],
    **kwargs: int,
) -> int:
    return physical_rank_by_logical_rank[_rank_from_coords(**kwargs)]


def _rank_from_coords(
    *,
    data_parallel_rank: int,
    pipeline_parallel_rank: int,
    local_parallel_rank: int,
    tensor_parallel_rank: int,
    expert_parallel_rank: int,
    model_parallel_size: int,
    per_pipeline: int,
    per_local: int,
    expert_parallel_size: int,
) -> int:
    return (
        data_parallel_rank * model_parallel_size
        + pipeline_parallel_rank * per_pipeline
        + local_parallel_rank * per_local
        + tensor_parallel_rank * expert_parallel_size
        + expert_parallel_rank
    )


def _physical_rank_by_logical_rank(
    *,
    world_size: int,
    data_parallel_size: int,
    local_parallel_size: int,
    model_parallel_size: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    expert_parallel_size: int,
    node_size: int | None,
    placement_policy: str,
) -> tuple[int, ...]:
    """Map logical ranks to physical ranks using topology-aware placement.

    Without node information, logical and physical ranks are identical. When a
    complete node topology is known and at least one full model replica fits in
    a node, distribute data-parallel replicas round-robin across nodes while
    keeping each model-parallel replica contiguous inside one node. This makes
    ``node_size`` affect the real process groups instead of only the placement
    label, and keeps TP/CP/BP groups node-local for the common one-node model
    replica case.

    When a dense MoE model-parallel replica spans multiple complete nodes, use
    the production TP/EP placement: TP and local CP/BP stay inside each node,
    while EP spans the same GPU slot across nodes. For torchrun-style rank
    ordering this maps logical ``(tp, ep)`` coordinates to physical
    ``node=ep, gpu=tp`` inside each data replica.
    """

    if node_size is None:
        if placement_policy != "auto":
            raise ValueError(
                "inter_node_cp placement requires LOCAL_WORLD_SIZE or node_size"
            )
        return tuple(range(world_size))
    node_size = int(node_size)
    if node_size <= 0 or world_size % node_size != 0:
        raise ValueError(
            "node_size must be positive and divide world_size for "
            "topology-aware placement"
        )

    if placement_policy == "inter_node_cp":
        return _physical_rank_by_inter_node_cp(
            world_size=world_size,
            data_parallel_size=data_parallel_size,
            local_parallel_size=local_parallel_size,
            model_parallel_size=model_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            expert_parallel_size=expert_parallel_size,
            node_size=node_size,
        )

    expert_inter_node = _physical_rank_by_tp_intra_node_ep_inter_node(
        world_size=world_size,
        data_parallel_size=data_parallel_size,
        local_parallel_size=local_parallel_size,
        model_parallel_size=model_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        expert_parallel_size=expert_parallel_size,
        node_size=node_size,
    )
    if expert_inter_node is not None:
        return expert_inter_node

    if model_parallel_size > node_size or node_size % model_parallel_size != 0:
        return tuple(range(world_size))

    node_count = world_size // node_size
    replicas_per_node = node_size // model_parallel_size
    if node_count <= 1 or replicas_per_node <= 0:
        return tuple(range(world_size))

    physical_by_logical = [0 for _ in range(world_size)]
    for data_parallel_rank in range(data_parallel_size):
        node_rank = data_parallel_rank % node_count
        replica_slot = data_parallel_rank // node_count
        if replica_slot >= replicas_per_node:
            raise RuntimeError(
                "topology placement cannot assign every data-parallel replica "
                "to a node-local model-parallel slot"
            )
        physical_base = node_rank * node_size + replica_slot * model_parallel_size
        logical_base = data_parallel_rank * model_parallel_size
        for model_parallel_rank in range(model_parallel_size):
            physical_by_logical[logical_base + model_parallel_rank] = (
                physical_base + model_parallel_rank
            )
    return tuple(physical_by_logical)


def _physical_rank_by_inter_node_cp(
    *,
    world_size: int,
    data_parallel_size: int,
    local_parallel_size: int,
    model_parallel_size: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    expert_parallel_size: int,
    node_size: int,
) -> tuple[int, ...]:
    """Stripe matched CP/BP groups across nodes while TP remains node-local."""

    node_count = world_size // node_size
    if node_count <= 1:
        raise ValueError("inter_node_cp placement requires at least two nodes")
    if pipeline_parallel_size != 1:
        raise ValueError(
            "inter_node_cp placement currently requires pipeline_parallel_size=1"
        )
    if local_parallel_size % node_count:
        raise ValueError(
            "inter_node_cp placement requires the CP/BP degree to divide evenly "
            "across nodes"
        )
    ranks_per_replica_per_node = model_parallel_size // node_count
    if (
        model_parallel_size % node_count
        or data_parallel_size * ranks_per_replica_per_node != node_size
    ):
        raise ValueError(
            "inter_node_cp placement requires complete, equally striped model "
            "replicas across every allocated node"
        )

    local_ranks_per_node = local_parallel_size // node_count
    inner_parallel_size = tensor_parallel_size * expert_parallel_size
    physical_by_logical = [0 for _ in range(world_size)]
    for data_parallel_rank in range(data_parallel_size):
        logical_base = data_parallel_rank * model_parallel_size
        for local_parallel_rank in range(local_parallel_size):
            physical_node = local_parallel_rank // local_ranks_per_node
            local_rank_in_node = local_parallel_rank % local_ranks_per_node
            for tensor_parallel_rank in range(tensor_parallel_size):
                for expert_parallel_rank in range(expert_parallel_size):
                    inner_rank = (
                        tensor_parallel_rank * expert_parallel_size
                        + expert_parallel_rank
                    )
                    logical_model_rank = (
                        local_parallel_rank * inner_parallel_size + inner_rank
                    )
                    physical_gpu = (
                        data_parallel_rank * ranks_per_replica_per_node
                        + local_rank_in_node * inner_parallel_size
                        + inner_rank
                    )
                    physical_by_logical[logical_base + logical_model_rank] = (
                        physical_node * node_size + physical_gpu
                    )
    if len(set(physical_by_logical)) != world_size:
        raise RuntimeError("inter_node_cp placement did not produce a rank permutation")
    return tuple(physical_by_logical)


def _physical_rank_by_tp_intra_node_ep_inter_node(
    *,
    world_size: int,
    data_parallel_size: int,
    local_parallel_size: int,
    model_parallel_size: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    expert_parallel_size: int,
    node_size: int,
) -> tuple[int, ...] | None:
    local_tp_size = local_parallel_size * tensor_parallel_size
    if (
        expert_parallel_size <= 1
        or pipeline_parallel_size != 1
        or local_tp_size != node_size
        or model_parallel_size != local_tp_size * expert_parallel_size
    ):
        return None

    node_count = world_size // node_size
    nodes_per_replica = expert_parallel_size
    if data_parallel_size * nodes_per_replica > node_count:
        return None

    per_local = tensor_parallel_size * expert_parallel_size
    physical_by_logical = [0 for _ in range(world_size)]
    for data_parallel_rank in range(data_parallel_size):
        replica_node_base = data_parallel_rank * nodes_per_replica
        logical_base = data_parallel_rank * model_parallel_size
        for model_parallel_rank in range(model_parallel_size):
            local_parallel_rank = model_parallel_rank // per_local
            local_remainder = model_parallel_rank % per_local
            tensor_parallel_rank = local_remainder // expert_parallel_size
            expert_parallel_rank = local_remainder % expert_parallel_size
            physical_node = replica_node_base + expert_parallel_rank
            physical_gpu = local_parallel_rank * tensor_parallel_size + tensor_parallel_rank
            physical_by_logical[logical_base + model_parallel_rank] = (
                physical_node * node_size + physical_gpu
            )
    if len(set(physical_by_logical)) != world_size:
        return None
    return tuple(physical_by_logical)


def _invert_rank_permutation(rank_by_rank: tuple[int, ...]) -> tuple[int, ...]:
    inverse = [0 for _ in rank_by_rank]
    for source_rank, target_rank in enumerate(rank_by_rank):
        inverse[int(target_rank)] = int(source_rank)
    return tuple(inverse)


def _supported_local_layout(
    context_parallel_size: int,
    block_parallel_size: int,
) -> bool:
    if context_parallel_size == 1 or block_parallel_size == 1:
        return True
    return (
        block_parallel_size >= context_parallel_size
        and block_parallel_size % context_parallel_size == 0
    )


def _local_parallel_size(context_parallel_size: int, block_parallel_size: int) -> int:
    return max(context_parallel_size, block_parallel_size)


def _context_parallel_rank(
    *,
    local_parallel_rank: int,
    context_parallel_size: int,
) -> int:
    if context_parallel_size == 1:
        return 0
    return local_parallel_rank % context_parallel_size


def _block_parallel_rank(
    *,
    local_parallel_rank: int,
    block_parallel_size: int,
) -> int:
    if block_parallel_size == 1:
        return 0
    return local_parallel_rank


def _context_block_local_ranks(
    *,
    local_parallel_rank: int,
    local_parallel_size: int,
    context_parallel_size: int,
    block_parallel_size: int,
) -> range:
    if context_parallel_size == 1 or block_parallel_size == 1:
        return range(local_parallel_size)
    group_start = (local_parallel_rank // context_parallel_size) * context_parallel_size
    return range(group_start, group_start + context_parallel_size)


def _clean_replica_local_ranks(
    *,
    local_parallel_rank: int,
    local_parallel_size: int,
    context_parallel_size: int,
    block_parallel_size: int,
) -> list[int]:
    if context_parallel_size == 1 or block_parallel_size == 1:
        return [local_parallel_rank]
    context_rank = _context_parallel_rank(
        local_parallel_rank=local_parallel_rank,
        context_parallel_size=context_parallel_size,
    )
    return [
        rank
        for rank in range(local_parallel_size)
        if _context_parallel_rank(
            local_parallel_rank=rank,
            context_parallel_size=context_parallel_size,
        ) == context_rank
    ]


def _layout_name(
    context_parallel_size: int,
    block_parallel_size: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    expert_parallel_size: int,
) -> str:
    if context_parallel_size > 1 and block_parallel_size > 1:
        base = "dp_x_fused_cp_bp"
    elif context_parallel_size > 1:
        base = "dp_x_cp"
    elif block_parallel_size > 1:
        base = "dp_x_bp"
    else:
        base = "dp"
    if tensor_parallel_size > 1:
        base = f"{base}_x_tp" if base != "dp" else "dp_x_tp"
    if pipeline_parallel_size > 1:
        base = f"{base}_x_pp"
    if expert_parallel_size > 1:
        base = f"{base}_x_ep"
    return base


def _placement_name(
    *,
    world_size: int,
    data_parallel_size: int,
    model_parallel_size: int,
    node_size: int | None,
    context_parallel_size: int,
    local_parallel_size: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    expert_parallel_size: int,
    placement_policy: str,
) -> str:
    if placement_policy == "inter_node_cp":
        return "inter_node_cp_tp_inner"
    if node_size is None:
        return "contiguous_tp_inner"
    node_count = world_size // node_size if node_size > 0 and world_size % node_size == 0 else 0
    if (
        expert_parallel_size > 1
        and pipeline_parallel_size == 1
        and local_parallel_size * tensor_parallel_size == node_size
        and model_parallel_size == node_size * expert_parallel_size
        and data_parallel_size * expert_parallel_size <= node_count
    ):
        return "tp_intra_node_ep_inter_node"
    if model_parallel_size <= node_size:
        return "node_local_model_parallel_tp_inner"
    if context_parallel_size * tensor_parallel_size <= node_size:
        return "multi_node_model_parallel_node_local_cp_tp_inner"
    return "multi_node_model_parallel_tp_inner"


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a v1 BDLM DP x local CP/BP parallel plan."
    )
    parser.add_argument("--num-blocks", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--data-parallel-size", type=int, required=True)
    parser.add_argument("--context-parallel-size", type=int, required=True)
    parser.add_argument("--block-parallel-size", type=int, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--expert-parallel-size", type=int, default=1)
    parser.add_argument("--node-size", type=int, default=None)
    parser.add_argument(
        "--placement-policy",
        choices=("auto", "inter_node_cp"),
        default="auto",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    plan = build_parallel_plan(
        num_blocks=args.num_blocks,
        world_size=args.world_size,
        data_parallel_size=args.data_parallel_size,
        context_parallel_size=args.context_parallel_size,
        block_parallel_size=args.block_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        expert_parallel_size=args.expert_parallel_size,
        node_size=args.node_size,
        placement_policy=args.placement_policy,
    )
    print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
