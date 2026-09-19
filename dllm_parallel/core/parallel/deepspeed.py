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

"""DeepSpeed ZeRO group helpers for DLLM fused CP/BP/TP training.

DeepSpeed's default ZeRO partitioning over the full world is not the right
semantics for ``DP x fused(CP, BP)`` runs. Inside a fused CP/BP group,
ranks cooperate on one sample-parallel computation and exchange activations/KV.
For optimizer sharding, those CP/BP ranks are sample-parallel workers for the
same tensor-parallel parameter shard, so ZeRO should partition over
``DP x fused(CP, BP)`` for each TP rank.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from dllm_parallel.core.parallel.groups import DLLMProcessGroupCollection
from dllm_parallel.core.parallel.topology import ParallelPlan


@dataclass
class _DLLMDeepSpeedMPU:
    """Minimal DeepSpeed model-parallel unit for fused CP/BP/TP groups."""

    data_parallel_rank: int
    data_parallel_world_size: int
    model_parallel_rank: int
    model_parallel_world_size: int
    pipeline_parallel_rank: int
    pipeline_parallel_world_size: int
    expert_parallel_rank: int
    expert_parallel_world_size: int
    data_parallel_group: Any
    model_parallel_group: Any
    context_block_parallel_group: Any
    pipeline_parallel_group: Any
    expert_parallel_group: Any
    context_block_parallel_world_size: int
    zero_partitions_sample_parallel: bool

    @classmethod
    def from_plan(cls, plan: ParallelPlan) -> "_DLLMDeepSpeedMPU":
        return cls.from_process_groups(
            DLLMProcessGroupCollection.from_plan(plan, create_groups=True)
        )

    @classmethod
    def from_process_groups(
        cls,
        groups: DLLMProcessGroupCollection,
    ) -> "_DLLMDeepSpeedMPU":
        plan = groups.plan
        rank = groups.rank
        assignment = groups.assignment
        zero_data_parallel_ranks = groups.optimizer_data_parallel_group_ranks
        return cls(
            data_parallel_rank=zero_data_parallel_ranks.index(rank),
            data_parallel_world_size=len(zero_data_parallel_ranks),
            model_parallel_rank=(
                assignment.pipeline_parallel_rank
                * plan.local_parallel_size
                * plan.tensor_parallel_size
                * plan.expert_parallel_size
                + assignment.local_parallel_rank
                * plan.tensor_parallel_size
                * plan.expert_parallel_size
                + assignment.tensor_parallel_rank * plan.expert_parallel_size
                + assignment.expert_parallel_rank
            ),
            model_parallel_world_size=plan.model_parallel_size,
            pipeline_parallel_rank=assignment.pipeline_parallel_rank,
            pipeline_parallel_world_size=plan.pipeline_parallel_size,
            expert_parallel_rank=assignment.expert_parallel_rank,
            expert_parallel_world_size=plan.expert_parallel_size,
            data_parallel_group=groups.optimizer_data_parallel_group,
            model_parallel_group=groups.model_parallel_group,
            context_block_parallel_group=groups.context_block_parallel_group,
            pipeline_parallel_group=groups.pipeline_parallel_group,
            expert_parallel_group=groups.expert_parallel_group,
            context_block_parallel_world_size=len(
                assignment.context_block_parallel_group
            ),
            zero_partitions_sample_parallel=groups.zero_partitions_sample_parallel,
        )

    def get_model_parallel_rank(self) -> int:
        return self.model_parallel_rank

    def get_model_parallel_group(self) -> Any:
        return self.model_parallel_group

    def get_model_parallel_world_size(self) -> int:
        return self.model_parallel_world_size

    def get_data_parallel_rank(self) -> int:
        return self.data_parallel_rank

    def get_data_parallel_group(self) -> Any:
        return self.data_parallel_group

    def get_data_parallel_world_size(self) -> int:
        return self.data_parallel_world_size

    def get_pipeline_model_parallel_group(self) -> Any:
        return self.pipeline_parallel_group

    def get_pipeline_model_parallel_rank(self) -> int:
        return self.pipeline_parallel_rank

    def get_pipeline_model_parallel_world_size(self) -> int:
        return self.pipeline_parallel_world_size

    def get_expert_parallel_group(self) -> Any:
        return self.expert_parallel_group

    def get_expert_model_parallel_group(self) -> Any:
        return self.expert_parallel_group

    def get_expert_model_parallel_rank(self) -> int:
        return self.expert_parallel_rank

    def get_expert_model_parallel_world_size(self) -> int:
        return self.expert_parallel_world_size

def build_dllm_deepspeed_mpu(source: Any) -> "_DLLMDeepSpeedMPU":
    """Build the DeepSpeed MPU that preserves DLLM DP/MP partition semantics."""

    process_groups = getattr(source, "process_groups", None)
    if isinstance(process_groups, DLLMProcessGroupCollection):
        return _DLLMDeepSpeedMPU.from_process_groups(process_groups)
    if isinstance(source, DLLMProcessGroupCollection):
        return _DLLMDeepSpeedMPU.from_process_groups(source)
    if isinstance(source, ParallelPlan):
        return _DLLMDeepSpeedMPU.from_plan(source)
    plan = getattr(source, "plan", None)
    if isinstance(plan, ParallelPlan):
        return _DLLMDeepSpeedMPU.from_plan(plan)
    raise TypeError("build_dllm_deepspeed_mpu requires a runtime, process groups, or plan")


def _local_parallel_size(context_parallel_size: int, block_parallel_size: int) -> int:
    return max(context_parallel_size, block_parallel_size)


def sync_model_parallel_zero_gradients(
    *,
    zero_optimizer: Any,
    model_parallel_group: Any,
    model_parallel_world_size: int,
) -> int:
    """Average ZeRO-2 gradient partitions across local CP/BP ranks.

    DeepSpeed has already reduced these tensors over the data-parallel group.
    DLLM local CP/BP ranks jointly compute one logical sample-parallel gradient,
    so the ZeRO partition gradient must also be averaged over the local
    model-parallel group before the optimizer updates its fp32 partition.
    """

    if model_parallel_world_size <= 1:
        return 0
    if not dist.is_available() or not dist.is_initialized():
        return 0

    grads = list(_iter_zero_gradient_tensors(zero_optimizer))
    for grad in grads:
        if grad.device.type == "cpu":
            raise RuntimeError(
                "DLLM local CP/BP gradient synchronization does not support "
                "CPU-offloaded ZeRO gradients."
            )
        if grad.is_sparse:
            raise RuntimeError(
                "DLLM local CP/BP gradient synchronization does not support "
                "sparse ZeRO gradients."
            )
    if not grads:
        return 0

    synced = 0
    with torch.no_grad():
        buckets = list(_bucket_dense_tensors(grads, max_bytes=0))
        pending: list[tuple[Any, list[torch.Tensor], list[torch.Tensor]]] = []
        for bucket in buckets:
            targets, reduce_tensors = _coalesced_reduce_tensors(bucket)
            work = dist.all_reduce_coalesced(
                reduce_tensors,
                op=dist.ReduceOp.SUM,
                group=model_parallel_group,
                async_op=True,
            )
            block_current_stream = getattr(work, "block_current_stream", None)
            if block_current_stream is not None:
                block_current_stream()
            pending.append((work, targets, reduce_tensors))
        for work, targets, reduce_tensors in pending:
            work.wait()
            _foreach_div_(reduce_tensors, float(model_parallel_world_size))
            for target, reduced in zip(targets, reduce_tensors, strict=True):
                if target is not reduced:
                    target.copy_(reduced)
            synced += len(targets)
    return synced


def sync_expert_parallel_zero_gradients(
    *,
    zero_optimizer: Any,
    expert_parallel_group: Any,
    expert_parallel_world_size: int,
) -> int:
    """Apply exact dense/expert EP gradient semantics to ZeRO partitions."""

    ep_size = int(expert_parallel_world_size)
    if ep_size <= 1:
        return 0
    if expert_parallel_group is None:
        raise RuntimeError("expert-parallel ZeRO synchronization requires an EP group")
    gradient_groups = _zero_averaged_gradient_groups(zero_optimizer)
    optimizer = getattr(zero_optimizer, "optimizer", None)
    param_groups = getattr(optimizer, "param_groups", None)
    if param_groups is None:
        param_groups = getattr(zero_optimizer, "param_groups", None)
    if param_groups is None or len(param_groups) != len(gradient_groups):
        raise RuntimeError(
            "DeepSpeed ZeRO optimizer groups do not match averaged-gradient groups"
        )

    pending: list[tuple[Any, list[tuple[torch.Tensor, torch.Tensor]]]] = []
    synchronized = 0
    with torch.no_grad():
        for param_group, gradients in zip(param_groups, gradient_groups, strict=True):
            tensors = list(_iter_tensors(gradients))
            if not tensors:
                continue
            synchronized += len(tensors)
            if bool(param_group.get("_dllm_expert_parallel_sharded", False)):
                _foreach_div_(tensors, float(ep_size))
                continue
            for bucket in _bucket_dense_tensors(tensors, max_bytes=0):
                targets, reduce_tensors = _coalesced_reduce_tensors(bucket)
                work = dist.all_reduce_coalesced(
                    reduce_tensors,
                    op=dist.ReduceOp.SUM,
                    group=expert_parallel_group,
                    async_op=True,
                )
                block_current_stream = getattr(work, "block_current_stream", None)
                if callable(block_current_stream):
                    block_current_stream()
                pending.append(
                    (work, list(zip(targets, reduce_tensors, strict=True)))
                )
        for work, entries in pending:
            work.wait()
            reduced = [value for _, value in entries]
            _foreach_div_(reduced, float(ep_size))
            for target, value in entries:
                if target is not value:
                    target.copy_(value)
    return synchronized


def sync_sequence_parallel_zero_gradients(
    *,
    zero_optimizer: Any,
    tensor_parallel_group: Any,
    tensor_parallel_world_size: int,
) -> int:
    """Sum sequence-parallel replicated ZeRO partitions across TP ranks.

    Sequence-parallel ranks own disjoint token rows. Their LayerNorm and other
    replicated parameter gradients are therefore partial sums, matching
    Megatron's sequence-parallel gradient semantics. The sum must be applied to
    ZeRO's averaged-gradient partitions after backward, not to ``param.grad``
    tensors that ZeRO-2 may already have released.
    """

    tp_size = int(tensor_parallel_world_size)
    if tp_size <= 1:
        return 0
    if tensor_parallel_group is None:
        raise RuntimeError("sequence-parallel ZeRO synchronization requires a TP group")
    gradient_groups = _zero_averaged_gradient_groups(zero_optimizer)
    optimizer = getattr(zero_optimizer, "optimizer", None)
    param_groups = getattr(optimizer, "param_groups", None)
    if param_groups is None:
        param_groups = getattr(zero_optimizer, "param_groups", None)
    if param_groups is None or len(param_groups) != len(gradient_groups):
        raise RuntimeError(
            "DeepSpeed ZeRO optimizer groups do not match averaged-gradient groups"
        )

    pending: list[tuple[Any, list[tuple[torch.Tensor, torch.Tensor]]]] = []
    synchronized = 0
    with torch.no_grad():
        for param_group, gradients in zip(param_groups, gradient_groups, strict=True):
            if not bool(param_group.get("_dllm_sequence_parallel_replicated", False)):
                continue
            tensors = list(_iter_tensors(gradients))
            synchronized += len(tensors)
            for bucket in _bucket_dense_tensors(tensors, max_bytes=0):
                targets, reduce_tensors = _coalesced_reduce_tensors(bucket)
                work = dist.all_reduce_coalesced(
                    reduce_tensors,
                    op=dist.ReduceOp.SUM,
                    group=tensor_parallel_group,
                    async_op=True,
                )
                block_current_stream = getattr(work, "block_current_stream", None)
                if callable(block_current_stream):
                    block_current_stream()
                pending.append(
                    (work, list(zip(targets, reduce_tensors, strict=True)))
                )
        for work, entries in pending:
            work.wait()
            for target, value in entries:
                if target is not value:
                    target.copy_(value)
    return synchronized


def _zero_averaged_gradient_groups(zero_optimizer: Any) -> list[Any]:
    averaged = getattr(zero_optimizer, "averaged_gradients", None)
    if isinstance(averaged, dict):
        return [averaged[index] for index in sorted(averaged)]
    if isinstance(averaged, (list, tuple)):
        return list(averaged)
    raise RuntimeError("DeepSpeed ZeRO averaged gradients are unavailable before step")


def _iter_zero_gradient_tensors(zero_optimizer: Any):
    averaged_gradients = getattr(zero_optimizer, "averaged_gradients", None)
    if averaged_gradients is not None:
        yielded = False
        for grad in _iter_tensors(averaged_gradients):
            yielded = True
            yield grad
        if yielded:
            return

    for partition in getattr(zero_optimizer, "single_partition_of_fp32_groups", ()):
        grad = getattr(partition, "grad", None)
        if grad is not None:
            yield grad


def _bucket_dense_tensors(
    tensors: list[torch.Tensor],
    *,
    max_bytes: int,
):
    bucket: list[torch.Tensor] = []
    bucket_bytes = 0
    bucket_key: tuple[torch.device, torch.dtype] | None = None
    for tensor in tensors:
        key = (tensor.device, tensor.dtype)
        tensor_bytes = tensor.numel() * tensor.element_size()
        if (
            bucket
            and (
                bucket_key != key
                or (max_bytes > 0 and bucket_bytes + tensor_bytes > max_bytes)
            )
        ):
            yield bucket
            bucket = []
            bucket_bytes = 0
            bucket_key = None
        bucket.append(tensor)
        bucket_bytes += tensor_bytes
        bucket_key = key
    if bucket:
        yield bucket


def _coalesced_reduce_tensors(
    tensors: list[torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    targets: list[torch.Tensor] = []
    reduce_tensors: list[torch.Tensor] = []
    for tensor in tensors:
        targets.append(tensor)
        reduce_tensors.append(tensor if tensor.is_contiguous() else tensor.contiguous())
    return targets, reduce_tensors


def _foreach_div_(tensors: list[torch.Tensor], value: float) -> None:
    if not tensors:
        return
    try:
        torch._foreach_div_(tensors, value)
    except RuntimeError:
        for tensor in tensors:
            tensor.div_(value)


def _iter_tensors(value: Any):
    if value is None:
        return
    if torch.is_tensor(value):
        yield value
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)
