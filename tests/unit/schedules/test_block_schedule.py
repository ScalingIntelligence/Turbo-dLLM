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

from dllm_parallel.core.schedules.block import build_block_schedule, dual_end_pairs


def test_dual_end_block_order() -> None:
    assert dual_end_pairs(8) == [(0, 7), (1, 6), (2, 5), (3, 4)]


def test_dual_end_block_order_with_odd_middle_task() -> None:
    assert dual_end_pairs(5) == [(0, 4), (1, 3), (2,)]


def test_contiguous_schedule_preserves_exact_ownership_without_balancing() -> None:
    schedule = build_block_schedule(
        num_blocks=16,
        block_parallel_size=4,
        context_parallel_size=4,
        active_block_schedule="contiguous",
    )

    assert schedule.active_block_schedule == "contiguous"
    assert schedule.active_blocks_by_worker == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
    ]
    assert schedule.active_prefix_cost_by_worker == [6, 22, 38, 54]


def test_schedule_policy_can_be_selected_by_ablation_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DLLM_BLOCK_SCHEDULE_POLICY", "contiguous")
    schedule = build_block_schedule(
        num_blocks=8,
        block_parallel_size=2,
        context_parallel_size=2,
    )

    assert schedule.active_block_schedule == "contiguous"
    assert schedule.active_blocks_by_worker == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    ]


def test_exact_pair_balancing() -> None:
    schedule = build_block_schedule(
        num_blocks=32,
        block_parallel_size=8,
        context_parallel_size=8,
    )

    assert schedule.active_block_pairs_by_worker == [
        [[0, 31], [1, 30]],
        [[2, 29], [3, 28]],
        [[4, 27], [5, 26]],
        [[6, 25], [7, 24]],
        [[8, 23], [9, 22]],
        [[10, 21], [11, 20]],
        [[12, 19], [13, 18]],
        [[14, 17], [15, 16]],
    ]
    assert schedule.active_prefix_cost_by_worker == [62] * 8


def test_all_active_blocks_assigned_exactly_once() -> None:
    schedule = build_block_schedule(
        num_blocks=32,
        block_parallel_size=8,
        context_parallel_size=8,
    )

    active_blocks = [
        block
        for worker_blocks in schedule.active_blocks_by_worker
        for block in worker_blocks
    ]
    assert sorted(active_blocks) == list(range(32))


def test_odd_active_blocks_assigned_exactly_once() -> None:
    schedule = build_block_schedule(
        num_blocks=5,
        block_parallel_size=2,
        context_parallel_size=1,
    )

    assert schedule.active_block_pairs_by_worker == [
        [[0, 4], [2]],
        [[1, 3]],
    ]
    active_blocks = [
        block
        for worker_blocks in schedule.active_blocks_by_worker
        for block in worker_blocks
    ]
    assert sorted(active_blocks) == list(range(5))


def test_non_divisible_pair_count_uses_balanced_contiguous_ranges() -> None:
    schedule = build_block_schedule(
        num_blocks=10,
        block_parallel_size=4,
        context_parallel_size=1,
    )

    assert schedule.active_block_pairs_by_worker == [
        [[0, 9], [1, 8]],
        [[2, 7]],
        [[3, 6]],
        [[4, 5]],
    ]


def test_short_dual_end_schedule_splits_pairs_into_complete_blocks() -> None:
    schedule = build_block_schedule(
        num_blocks=5,
        block_parallel_size=4,
        context_parallel_size=1,
    )

    assert all(schedule.active_blocks_by_worker)
    assert sorted(
        block
        for worker_blocks in schedule.active_blocks_by_worker
        for block in worker_blocks
    ) == list(range(5))


def test_all_kv_blocks_owned_exactly_once() -> None:
    schedule = build_block_schedule(
        num_blocks=32,
        block_parallel_size=8,
        context_parallel_size=8,
    )

    kv_blocks = [
        block
        for rank_blocks in schedule.kv_blocks_by_context_rank
        for block in rank_blocks
    ]
    assert sorted(kv_blocks) == list(range(32))


def test_kv_replication_factor_remains_one() -> None:
    schedule = build_block_schedule(
        num_blocks=32,
        block_parallel_size=8,
        context_parallel_size=8,
    )

    assert schedule.kv_replication_factor == 1.0


def test_four_h100_layout_is_balanced() -> None:
    schedule = build_block_schedule(
        num_blocks=32,
        block_parallel_size=4,
        context_parallel_size=4,
    )

    assert schedule.active_prefix_cost_by_worker == [124] * 4
    assert schedule.kv_blocks_by_context_rank == [
        list(range(0, 8)),
        list(range(8, 16)),
        list(range(16, 24)),
        list(range(24, 32)),
    ]
    assert schedule.kv_replication_factor == 1.0


def test_eight_h100_layout_is_balanced() -> None:
    schedule = build_block_schedule(
        num_blocks=32,
        block_parallel_size=8,
        context_parallel_size=8,
    )

    assert schedule.active_prefix_cost_by_worker == [62] * 8
    assert schedule.kv_blocks_by_context_rank == [
        list(range(0, 4)),
        list(range(4, 8)),
        list(range(8, 12)),
        list(range(12, 16)),
        list(range(16, 20)),
        list(range(20, 24)),
        list(range(24, 28)),
        list(range(28, 32)),
    ]
    assert schedule.kv_replication_factor == 1.0
