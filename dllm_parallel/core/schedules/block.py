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

"""Block and context-parallel schedule construction for BDLM training.

The MVP scheduler models two separate ownership problems:

* active diffusion blocks are assigned to block-parallel workers in a dual-end
  order, pairing early and late blocks to balance prefix-attention cost;
* K/V cache blocks are owned by context-parallel ranks exactly once, avoiding
  replication while preserving a single logical cache.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Sequence


@dataclass(frozen=True)
class BlockSchedule:
    """Serializable BDLM block/context-parallel schedule."""

    num_blocks: int
    block_parallel_size: int
    context_parallel_size: int
    active_block_schedule: str
    active_block_pairs_by_worker: list[list[list[int]]]
    active_blocks_by_worker: list[list[int]]
    kv_blocks_by_context_rank: list[list[int]]
    active_prefix_cost_by_worker: list[int]
    kv_replication_factor: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def dual_end_pairs(num_blocks: int) -> list[tuple[int, ...]]:
    """Return dual-end tasks ordered from sequence ends inward.

    Even-length schedules produce ``(left, right)`` pairs. Odd-length schedules
    add the leftover middle block as a one-element task, matching the paper's
    dual-end policy.
    """

    _require_positive("num_blocks", num_blocks)

    pairs = [(left, num_blocks - 1 - left) for left in range(num_blocks // 2)]
    if num_blocks % 2 != 0:
        pairs.append((num_blocks // 2,))
    return pairs


def build_block_schedule(
    num_blocks: int,
    block_parallel_size: int,
    context_parallel_size: int,
    active_block_schedule: str | None = None,
) -> BlockSchedule:
    """Build and validate the MVP BDLM block/context-parallel schedule."""

    _require_positive("num_blocks", num_blocks)
    _require_positive("block_parallel_size", block_parallel_size)
    _require_positive("context_parallel_size", context_parallel_size)

    schedule_policy = resolve_active_block_schedule(active_block_schedule)
    if schedule_policy == "dual_end":
        pair_assignments = _dual_end_assignments(
            num_blocks=num_blocks,
            block_parallel_size=block_parallel_size,
        )
    else:
        pair_assignments = [
            [(block,) for block in shard]
            for shard in _balanced_contiguous_shards(
                num_blocks,
                block_parallel_size,
            )
        ]

    active_blocks_by_worker = [
        [block for pair in worker_pairs for block in pair]
        for worker_pairs in pair_assignments
    ]
    kv_blocks_by_context_rank = _balanced_contiguous_shards(
        num_blocks, context_parallel_size
    )
    prefix_cost_by_worker = [sum(blocks) for blocks in active_blocks_by_worker]

    schedule = BlockSchedule(
        num_blocks=num_blocks,
        block_parallel_size=block_parallel_size,
        context_parallel_size=context_parallel_size,
        active_block_schedule=schedule_policy,
        active_block_pairs_by_worker=[
            [list(task) for task in worker_pairs] for worker_pairs in pair_assignments
        ],
        active_blocks_by_worker=active_blocks_by_worker,
        kv_blocks_by_context_rank=kv_blocks_by_context_rank,
        active_prefix_cost_by_worker=prefix_cost_by_worker,
        kv_replication_factor=_kv_replication_factor(
            kv_blocks_by_context_rank, num_blocks
        ),
    )
    validate_block_schedule(schedule)
    return schedule


def validate_block_schedule(schedule: BlockSchedule) -> None:
    """Raise ``ValueError`` when the schedule violates MVP invariants."""

    expected_blocks = list(range(schedule.num_blocks))

    active_blocks = _flatten(schedule.active_blocks_by_worker)
    if sorted(active_blocks) != expected_blocks:
        raise ValueError("active blocks must cover every block exactly once")

    kv_blocks = _flatten(schedule.kv_blocks_by_context_rank)
    if sorted(kv_blocks) != expected_blocks:
        raise ValueError("KV ownership must cover every block exactly once")

    if schedule.kv_replication_factor != 1.0:
        raise ValueError("KV replication factor must be 1.0")

    if (
        schedule.active_block_schedule == "dual_end"
        and len(set(schedule.active_prefix_cost_by_worker)) != 1
    ):
        max_cost = max(schedule.active_prefix_cost_by_worker)
        min_cost = min(schedule.active_prefix_cost_by_worker)
        if max_cost - min_cost > schedule.num_blocks:
            raise ValueError("active prefix cost is unexpectedly imbalanced")


def resolve_active_block_schedule(value: str | None = None) -> str:
    """Resolve the target-block assignment policy used by fused CP+BP.

    Production defaults to the load-balanced dual-end schedule. The contiguous
    alternative is intentionally exposed for controlled ablations while
    preserving complete, disjoint target-block ownership and the same loss.
    """

    resolved = value
    if resolved is None:
        resolved = os.environ.get("DLLM_BLOCK_SCHEDULE_POLICY", "dual_end")
    resolved = str(resolved).strip().lower().replace("-", "_")
    if resolved not in {"dual_end", "contiguous"}:
        raise ValueError("active block schedule must be 'dual_end' or 'contiguous'")
    return resolved


def _dual_end_assignments(
    *,
    num_blocks: int,
    block_parallel_size: int,
) -> list[list[tuple[int, ...]]]:
    active_pairs = dual_end_pairs(num_blocks)
    if num_blocks < block_parallel_size:
        raise ValueError(
            "num_blocks must be at least block_parallel_size so every worker "
            "owns at least one complete block"
        )
    if len(active_pairs) < block_parallel_size:
        return _short_dual_end_assignments(
            num_blocks=num_blocks,
            block_parallel_size=block_parallel_size,
        )
    middle_task = active_pairs[-1] if len(active_pairs[-1]) == 1 else None
    paired_tasks = active_pairs[:-1] if middle_task is not None else active_pairs
    pair_assignments: list[list[tuple[int, ...]]] = []
    pair_start = 0
    pairs_per_worker, remainder = divmod(
        len(paired_tasks),
        block_parallel_size,
    )
    for worker in range(block_parallel_size):
        pair_count = pairs_per_worker + int(worker < remainder)
        pair_stop = pair_start + pair_count
        pair_assignments.append(paired_tasks[pair_start:pair_stop])
        pair_start = pair_stop
    if middle_task is not None:
        middle_owner = min(
            range(block_parallel_size),
            key=lambda worker: sum(
                block for task in pair_assignments[worker] for block in task
            ),
        )
        pair_assignments[middle_owner].append(middle_task)
    return pair_assignments


def _short_dual_end_assignments(
    *,
    num_blocks: int,
    block_parallel_size: int,
) -> list[list[tuple[int, ...]]]:
    """Balance blocks when intact dual-end pairs cannot fill every worker.

    Dual-end pairs are a load-balancing device, not indivisible objective work:
    the paper assigns complete *blocks* to workers. In the short regime
    ``P <= B < 2P - 1``, schedule individual blocks from greatest to least
    prefix-attention cost and repeatedly place the next block on the currently
    lightest worker. This follows the same opposite-end balancing principle
    while ensuring every worker owns at least one complete block.
    """

    assignments: list[list[tuple[int, ...]]] = [[] for _ in range(block_parallel_size)]
    costs = [0] * block_parallel_size
    for block in range(num_blocks - 1, -1, -1):
        worker = min(
            range(block_parallel_size),
            key=lambda candidate: (costs[candidate], candidate),
        )
        assignments[worker].append((block,))
        costs[worker] += block
    return assignments


def _balanced_contiguous_shards(num_items: int, num_shards: int) -> list[list[int]]:
    base, remainder = divmod(num_items, num_shards)
    shards: list[list[int]] = []
    start = 0
    for shard_index in range(num_shards):
        shard_size = base + (1 if shard_index < remainder else 0)
        stop = start + shard_size
        shards.append(list(range(start, stop)))
        start = stop
    return shards


def _kv_replication_factor(
    kv_blocks_by_context_rank: Sequence[Sequence[int]], num_blocks: int
) -> float:
    if num_blocks == 0:
        raise ValueError("num_blocks must be positive")
    return sum(len(blocks) for blocks in kv_blocks_by_context_rank) / num_blocks


def _flatten(blocks_by_owner: Sequence[Sequence[int]]) -> list[int]:
    return [block for blocks in blocks_by_owner for block in blocks]


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a BDLM block/context-parallel schedule."
    )
    parser.add_argument("--num-blocks", type=int, required=True)
    parser.add_argument("--block-parallel-size", type=int, required=True)
    parser.add_argument("--context-parallel-size", type=int, required=True)
    parser.add_argument(
        "--active-block-schedule",
        choices=("dual_end", "contiguous"),
        default="dual_end",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    schedule = build_block_schedule(
        num_blocks=args.num_blocks,
        block_parallel_size=args.block_parallel_size,
        context_parallel_size=args.context_parallel_size,
        active_block_schedule=args.active_block_schedule,
    )
    print(json.dumps(schedule.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
