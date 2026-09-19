# Copyright 2026 The bdlm_parallel Authors.
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

import pytest

from dllm_parallel.core.parallel.topology import build_parallel_plan


def test_four_h100_single_replica_plan() -> None:
    plan = build_parallel_plan(
        num_blocks=32,
        world_size=4,
        data_parallel_size=1,
        context_parallel_size=4,
        block_parallel_size=4,
    )

    assert plan.layout == "dp_x_fused_cp_bp"
    assert plan.local_parallel_size == 4
    assert [
        rank.context_block_parallel_group
        for rank in plan.rank_assignments
    ] == [[0, 1, 2, 3]] * 4
    assert [rank.data_parallel_group for rank in plan.rank_assignments] == [
        [0],
        [1],
        [2],
        [3],
    ]
    assert plan.block_schedule.active_prefix_cost_by_worker == [124] * 4


def test_eight_h100_single_replica_plan() -> None:
    plan = build_parallel_plan(
        num_blocks=32,
        world_size=8,
        data_parallel_size=1,
        context_parallel_size=8,
        block_parallel_size=8,
    )

    assert plan.local_parallel_size == 8
    assert plan.rank_assignments[0].context_block_parallel_group == list(
        range(8)
    )
    assert plan.rank_assignments[7].block_parallel_rank == 7
    assert plan.block_schedule.active_prefix_cost_by_worker == [62] * 8


def test_eight_h100_two_replica_plan() -> None:
    plan = build_parallel_plan(
        num_blocks=32,
        world_size=8,
        data_parallel_size=2,
        context_parallel_size=4,
        block_parallel_size=4,
    )

    assert plan.rank_assignments[0].data_parallel_group == [0, 4]
    assert plan.rank_assignments[3].data_parallel_group == [3, 7]
    assert plan.rank_assignments[4].context_block_parallel_group == [
        4,
        5,
        6,
        7,
    ]
    assert plan.block_schedule.active_prefix_cost_by_worker == [124] * 4


def test_dp_cp_bp_tp_plan_keeps_tp_and_context_groups_orthogonal() -> None:
    plan = build_parallel_plan(
        num_blocks=32,
        world_size=8,
        data_parallel_size=2,
        context_parallel_size=2,
        block_parallel_size=2,
        tensor_parallel_size=2,
    )

    assert plan.layout == "dp_x_fused_cp_bp_x_tp"
    assert plan.local_parallel_size == 2
    assert plan.model_parallel_size == 4
    assert plan.tensor_parallel_size == 2
    assert plan.rank_assignments[0].context_block_parallel_group == [0, 2]
    assert plan.rank_assignments[1].context_block_parallel_group == [1, 3]
    assert plan.rank_assignments[0].tensor_parallel_group == [0, 1]
    assert plan.rank_assignments[2].tensor_parallel_group == [2, 3]
    assert plan.rank_assignments[0].model_parallel_group == [0, 1, 2, 3]
    assert plan.rank_assignments[4].model_parallel_group == [4, 5, 6, 7]
    assert plan.rank_assignments[0].data_parallel_group == [0, 4]
    assert plan.rank_assignments[3].data_parallel_group == [3, 7]
    assert [rank.tensor_parallel_rank for rank in plan.rank_assignments[:4]] == [
        0,
        1,
        0,
        1,
    ]


def test_node_local_tp_cp_bp_plan_records_topology_aware_placement() -> None:
    plan = build_parallel_plan(
        num_blocks=32,
        world_size=16,
        data_parallel_size=4,
        context_parallel_size=2,
        block_parallel_size=2,
        tensor_parallel_size=2,
        node_size=4,
    )

    assert plan.placement == "node_local_model_parallel_tp_inner"
    assert plan.rank_assignments[0].model_parallel_group == [0, 1, 2, 3]
    assert plan.rank_assignments[0].tensor_parallel_group == [0, 1]
    assert plan.rank_assignments[2].tensor_parallel_group == [2, 3]
    assert plan.rank_assignments[0].context_block_parallel_group == [0, 2]
    assert plan.rank_assignments[1].context_block_parallel_group == [1, 3]


def test_node_size_round_robins_model_replicas_across_nodes() -> None:
    plan = build_parallel_plan(
        num_blocks=32,
        world_size=16,
        data_parallel_size=4,
        context_parallel_size=2,
        block_parallel_size=2,
        tensor_parallel_size=2,
        node_size=8,
    )

    assert plan.placement == "node_local_model_parallel_tp_inner"
    assert plan.rank_assignments[0].model_parallel_group == [0, 1, 2, 3]
    assert plan.rank_assignments[8].model_parallel_group == [8, 9, 10, 11]
    assert plan.rank_assignments[4].model_parallel_group == [4, 5, 6, 7]
    assert plan.rank_assignments[0].data_parallel_group == [0, 8, 4, 12]
    assert plan.rank_assignments[1].data_parallel_group == [1, 9, 5, 13]
    assert plan.rank_assignments[8].data_parallel_rank == 1
    assert plan.rank_assignments[4].data_parallel_rank == 2


def test_multi_node_tp_cp_bp_plan_keeps_cp_and_tp_groups_node_local_when_possible() -> None:
    plan = build_parallel_plan(
        num_blocks=32,
        world_size=16,
        data_parallel_size=2,
        context_parallel_size=2,
        block_parallel_size=4,
        tensor_parallel_size=2,
        node_size=4,
    )

    assert plan.placement == "multi_node_model_parallel_node_local_cp_tp_inner"
    assert plan.rank_assignments[0].model_parallel_group == list(range(8))
    assert plan.rank_assignments[0].tensor_parallel_group == [0, 1]
    assert plan.rank_assignments[2].tensor_parallel_group == [2, 3]
    assert plan.rank_assignments[0].context_block_parallel_group == [0, 2]
    assert plan.rank_assignments[1].context_block_parallel_group == [1, 3]
    assert plan.rank_assignments[4].context_block_parallel_group == [4, 6]
    assert plan.rank_assignments[0].clean_replica_group == [0, 4]


def test_tensor_parallel_only_plan() -> None:
    plan = build_parallel_plan(
        num_blocks=32,
        world_size=4,
        data_parallel_size=2,
        context_parallel_size=1,
        block_parallel_size=1,
        tensor_parallel_size=2,
    )

    assert plan.layout == "dp_x_tp"
    assert plan.local_parallel_size == 1
    assert plan.model_parallel_size == 2
    assert [rank.tensor_parallel_group for rank in plan.rank_assignments] == [
        [0, 1],
        [0, 1],
        [2, 3],
        [2, 3],
    ]
    assert [rank.data_parallel_group for rank in plan.rank_assignments] == [
        [0, 2],
        [1, 3],
        [0, 2],
        [1, 3],
    ]


def test_pipeline_and_expert_axes_are_orthogonal_to_fused_local_and_tp() -> None:
    plan = build_parallel_plan(
        num_blocks=32,
        world_size=32,
        data_parallel_size=2,
        context_parallel_size=2,
        block_parallel_size=2,
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        expert_parallel_size=2,
    )

    rank0 = plan.rank_assignments[0]
    rank1 = plan.rank_assignments[1]
    rank2 = plan.rank_assignments[2]
    rank4 = plan.rank_assignments[4]
    rank8 = plan.rank_assignments[8]
    assert plan.layout == "dp_x_fused_cp_bp_x_tp_x_pp_x_ep"
    assert plan.model_parallel_size == 16
    assert rank0.data_parallel_group == [0, 16]
    assert rank0.dense_data_parallel_group == [0, 1, 16, 17]
    assert rank0.expert_parallel_group == [0, 1]
    assert rank0.tensor_parallel_group == [0, 2]
    assert rank0.context_block_parallel_group == [0, 4]
    assert rank0.pipeline_parallel_group == [0, 8]
    assert rank1.expert_parallel_rank == 1
    assert rank2.tensor_parallel_rank == 1
    assert rank4.local_parallel_rank == 1
    assert rank8.pipeline_parallel_rank == 1
    assert rank0.model_input_group == [0, 2, 4, 6]
    assert rank1.model_input_group == [1, 3, 5, 7]


def test_ep_ranks_are_independent_samples_but_share_dense_gradients() -> None:
    plan = build_parallel_plan(
        num_blocks=8,
        world_size=8,
        data_parallel_size=2,
        context_parallel_size=1,
        block_parallel_size=1,
        tensor_parallel_size=2,
        expert_parallel_size=2,
    )

    rank0 = plan.rank_assignments[0]
    rank1 = plan.rank_assignments[1]
    assert rank0.model_input_group == [0, 2]
    assert rank1.model_input_group == [1, 3]
    assert rank0.data_parallel_group == [0, 4]
    assert rank1.data_parallel_group == [1, 5]
    assert rank0.dense_data_parallel_group == [0, 1, 4, 5]
    assert rank1.dense_data_parallel_group == [0, 1, 4, 5]


def test_dense_moe_plan_keeps_tp_intra_node_and_ep_inter_node() -> None:
    plan = build_parallel_plan(
        num_blocks=32,
        world_size=16,
        data_parallel_size=1,
        context_parallel_size=1,
        block_parallel_size=1,
        tensor_parallel_size=4,
        expert_parallel_size=4,
        node_size=4,
    )

    assert plan.placement == "tp_intra_node_ep_inter_node"
    assert plan.rank_assignments[0].tensor_parallel_group == [0, 1, 2, 3]
    assert plan.rank_assignments[4].tensor_parallel_group == [4, 5, 6, 7]
    assert plan.rank_assignments[0].expert_parallel_group == [0, 4, 8, 12]
    assert plan.rank_assignments[1].expert_parallel_group == [1, 5, 9, 13]
    assert plan.rank_assignments[0].model_parallel_group == [
        0,
        4,
        8,
        12,
        1,
        5,
        9,
        13,
        2,
        6,
        10,
        14,
        3,
        7,
        11,
        15,
    ]


def test_dense_moe_plan_preserves_disjoint_node_blocks_for_dp_replicas() -> None:
    plan = build_parallel_plan(
        num_blocks=32,
        world_size=16,
        data_parallel_size=2,
        context_parallel_size=1,
        block_parallel_size=1,
        tensor_parallel_size=4,
        expert_parallel_size=2,
        node_size=4,
    )

    assert plan.placement == "tp_intra_node_ep_inter_node"
    assert plan.rank_assignments[0].data_parallel_group == [0, 8]
    assert plan.rank_assignments[4].expert_parallel_group == [0, 4]
    assert plan.rank_assignments[8].data_parallel_rank == 1
    assert plan.rank_assignments[8].tensor_parallel_group == [8, 9, 10, 11]
    assert plan.rank_assignments[12].expert_parallel_group == [8, 12]


def test_replicated_bp_plan_uses_single_context_rank() -> None:
    plan = build_parallel_plan(
        num_blocks=32,
        world_size=4,
        data_parallel_size=2,
        context_parallel_size=1,
        block_parallel_size=2,
    )

    assert plan.local_parallel_size == 2
    assert plan.context_parallel_size == 1
    assert plan.block_parallel_size == 2
    assert [rank.context_parallel_rank for rank in plan.rank_assignments] == [
        0,
        0,
        0,
        0,
    ]
    assert [rank.block_parallel_rank for rank in plan.rank_assignments] == [
        0,
        1,
        0,
        1,
    ]
    assert [rank.context_block_parallel_group for rank in plan.rank_assignments] == [
        [0, 1],
        [0, 1],
        [2, 3],
        [2, 3],
    ]
    assert [rank.clean_replica_group for rank in plan.rank_assignments] == [
        [0],
        [1],
        [2],
        [3],
    ]
    assert [rank.data_parallel_group for rank in plan.rank_assignments] == [
        [0, 2],
        [1, 3],
        [0, 2],
        [1, 3],
    ]


def test_context_parallel_only_plan_uses_single_block_rank() -> None:
    plan = build_parallel_plan(
        num_blocks=32,
        world_size=4,
        data_parallel_size=2,
        context_parallel_size=2,
        block_parallel_size=1,
    )

    assert plan.layout == "dp_x_cp"
    assert plan.local_parallel_size == 2
    assert plan.context_parallel_size == 2
    assert plan.block_parallel_size == 1
    assert [rank.context_parallel_rank for rank in plan.rank_assignments] == [
        0,
        1,
        0,
        1,
    ]
    assert [rank.block_parallel_rank for rank in plan.rank_assignments] == [
        0,
        0,
        0,
        0,
    ]
    assert [rank.context_block_parallel_group for rank in plan.rank_assignments] == [
        [0, 1],
        [0, 1],
        [2, 3],
        [2, 3],
    ]
    assert [rank.data_parallel_group for rank in plan.rank_assignments] == [
        [0, 2],
        [1, 3],
        [0, 2],
        [1, 3],
    ]


def test_independent_bp_over_replicated_cp_rings_plan() -> None:
    plan = build_parallel_plan(
        num_blocks=32,
        world_size=4,
        data_parallel_size=1,
        context_parallel_size=2,
        block_parallel_size=4,
    )

    assert plan.layout == "dp_x_fused_cp_bp"
    assert plan.local_parallel_size == 4
    assert [rank.context_parallel_rank for rank in plan.rank_assignments] == [
        0,
        1,
        0,
        1,
    ]
    assert [rank.block_parallel_rank for rank in plan.rank_assignments] == [
        0,
        1,
        2,
        3,
    ]
    assert [rank.context_block_parallel_group for rank in plan.rank_assignments] == [
        [0, 1],
        [0, 1],
        [2, 3],
        [2, 3],
    ]
    assert [rank.clean_replica_group for rank in plan.rank_assignments] == [
        [0, 2],
        [1, 3],
        [0, 2],
        [1, 3],
    ]
    assert [rank.model_parallel_group for rank in plan.rank_assignments] == [
        [0, 1, 2, 3],
    ] * 4
    assert plan.block_schedule.active_prefix_cost_by_worker == [124] * 4


def test_v1_rejects_independent_cp_bp_mesh_that_cannot_tile_cp_rings() -> None:
    with pytest.raises(ValueError, match="divisible"):
        build_parallel_plan(
            num_blocks=32,
            world_size=4,
            data_parallel_size=1,
            context_parallel_size=3,
            block_parallel_size=4,
        )


def test_v1_rejects_world_size_mismatch() -> None:
    with pytest.raises(ValueError, match="world_size must equal"):
        build_parallel_plan(
            num_blocks=32,
            world_size=8,
            data_parallel_size=2,
            context_parallel_size=8,
            block_parallel_size=8,
        )


def test_topology_aware_placement_rejects_invalid_node_size() -> None:
    with pytest.raises(ValueError, match="node_size"):
        build_parallel_plan(
            num_blocks=32,
            world_size=8,
            data_parallel_size=2,
            context_parallel_size=4,
            block_parallel_size=4,
            node_size=3,
        )


def test_inter_node_cp_stripes_cp_and_keeps_tp_and_dp_node_local() -> None:
    plan = build_parallel_plan(
        num_blocks=256,
        world_size=16,
        data_parallel_size=2,
        context_parallel_size=4,
        block_parallel_size=4,
        tensor_parallel_size=2,
        node_size=8,
        placement_policy="inter_node_cp",
    )

    assert plan.placement == "inter_node_cp_tp_inner"
    assert plan.rank_assignments[0].context_block_parallel_group == [0, 2, 8, 10]
    assert plan.rank_assignments[1].context_block_parallel_group == [1, 3, 9, 11]
    assert plan.rank_assignments[0].tensor_parallel_group == [0, 1]
    assert plan.rank_assignments[8].tensor_parallel_group == [8, 9]
    assert plan.rank_assignments[0].data_parallel_group == [0, 4]
    assert plan.rank_assignments[8].data_parallel_group == [8, 12]

    cp_plan = build_parallel_plan(
        num_blocks=256,
        world_size=16,
        data_parallel_size=2,
        context_parallel_size=4,
        block_parallel_size=1,
        tensor_parallel_size=2,
        node_size=8,
        placement_policy="inter_node_cp",
    )
    assert cp_plan.rank_assignments[0].context_block_parallel_group == [0, 2, 8, 10]


def test_inter_node_cp_rejects_node_local_or_ep_layouts() -> None:
    with pytest.raises(ValueError, match="at least two nodes"):
        build_parallel_plan(
            num_blocks=256,
            world_size=8,
            data_parallel_size=2,
            context_parallel_size=4,
            block_parallel_size=4,
            node_size=8,
            placement_policy="inter_node_cp",
        )

    ep_plan = build_parallel_plan(
        num_blocks=256,
        world_size=16,
        data_parallel_size=2,
        context_parallel_size=4,
        block_parallel_size=4,
        expert_parallel_size=2,
        node_size=8,
        placement_policy="inter_node_cp",
    )
    assert ep_plan.rank_assignments[0].context_block_parallel_group == [0, 2, 8, 10]
    assert ep_plan.rank_assignments[1].context_block_parallel_group == [1, 3, 9, 11]
    assert ep_plan.rank_assignments[0].expert_parallel_group == [0, 1]
    assert ep_plan.rank_assignments[8].expert_parallel_group == [8, 9]

    with pytest.raises(ValueError, match="CP-only or matched CP/BP"):
        build_parallel_plan(
            num_blocks=256,
            world_size=16,
            data_parallel_size=1,
            context_parallel_size=4,
            block_parallel_size=8,
            tensor_parallel_size=2,
            node_size=8,
            placement_policy="inter_node_cp",
        )
