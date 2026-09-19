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

"""Runtime helpers for v1 BDLM CP/BP/TP training integration.

The user-facing topology is inferred from ``parallel.block_parallel_size`` and
``parallel.context_parallel_size`` and ``parallel.tensor_parallel_size``.
``ParallelRuntime.enabled`` is still kept as rank-local derived state, but
there is intentionally no ``parallel.enabled`` config knob.
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch
import torch.distributed as dist

from dllm_parallel.core.parallel.groups import (
    DLLMProcessGroupCollection,
    DLLMProcessGroupOptions,
    current_group_from_plan,
)
from dllm_parallel.core.schedules.block import build_block_schedule
from dllm_parallel.core.parallel.preflight import validate_supported_training_axes
from dllm_parallel.core.parallel.topology import ParallelPlan, build_parallel_plan
from dllm_parallel.core.specs import CPBPPolicy

_ACTIVE_TOKEN_INDICES_CACHE: dict[tuple[object, ...], torch.Tensor] = {}


@dataclass(slots=True)
class PendingCleanReplicaTensor:
    """Asynchronous clean-replica transfer with owner-gradient return."""

    output: torch.Tensor
    owner_input: torch.Tensor
    work: Any | None
    src_rank: int
    replica_rank: int
    replica_size: int
    group: Any | None

    def wait(self) -> torch.Tensor:
        if self.work is not None:
            self.work.wait()
        if self.group is None or self.replica_size <= 1:
            return self.output
        return _CleanReplicaBackward.apply(
            self.output,
            self.owner_input,
            int(self.src_rank),
            int(self.replica_rank),
            int(self.replica_size),
            self.group,
        )


@dataclass(frozen=True)
class ParallelRuntime:
    """Rank-local runtime state for fused context/block parallel training."""

    enabled: bool
    active_block_mode: str
    rank: int
    world_size: int
    data_parallel_rank: int
    local_parallel_rank: int
    context_parallel_rank: int
    block_parallel_rank: int
    context_block_parallel_group_ranks: list[int]
    data_parallel_group_ranks: list[int]
    dense_data_parallel_group_ranks: list[int] | None = None
    clean_replica_group_ranks: list[int] | None = None
    tensor_parallel_rank: int = 0
    pipeline_parallel_rank: int = 0
    expert_parallel_rank: int = 0
    tensor_parallel_group_ranks: list[int] | None = None
    pipeline_parallel_group_ranks: list[int] | None = None
    expert_parallel_group_ranks: list[int] | None = None
    node_size: int | None = None
    model_input_group_ranks: list[int] | None = None
    model_parallel_group_ranks: list[int] | None = None
    sequence_parallel: bool = False
    tensor_parallel_overlap: bool = True
    cp_bp_policy: CPBPPolicy = CPBPPolicy()
    kv_backend: str = "replicated"
    context_block_parallel_group: Any | None = None
    clean_replica_group: Any | None = None
    data_parallel_group: Any | None = None
    tensor_parallel_group: Any | None = None
    pipeline_parallel_group: Any | None = None
    expert_parallel_group: Any | None = None
    model_input_group: Any | None = None
    model_parallel_group: Any | None = None
    process_groups: DLLMProcessGroupCollection | None = None
    plan: ParallelPlan | None = None

    @property
    def local_parallel_size(self) -> int:
        if self.plan is not None:
            return int(self.plan.local_parallel_size)
        return len(self.context_block_parallel_group_ranks)

    @property
    def context_block_src_rank(self) -> int:
        return self.context_block_parallel_group_ranks[0]

    @property
    def clean_replica_rank(self) -> int:
        ranks = self.clean_replica_group_ranks
        if not ranks:
            return 0
        return ranks.index(self.rank)

    @property
    def clean_replica_src_rank(self) -> int:
        ranks = self.clean_replica_group_ranks
        if ranks:
            return ranks[0]
        return self.rank

    @property
    def clean_replica_size(self) -> int:
        ranks = self.clean_replica_group_ranks
        return len(ranks) if ranks else 1

    @property
    def tensor_parallel_size(self) -> int:
        if self.plan is not None:
            return int(self.plan.tensor_parallel_size)
        ranks = self.tensor_parallel_group_ranks
        return len(ranks) if ranks else 1

    @property
    def model_parallel_size(self) -> int:
        if self.plan is not None:
            return int(self.plan.model_parallel_size)
        ranks = self.model_parallel_group_ranks
        if ranks:
            return len(ranks)
        return self.local_parallel_size * self.tensor_parallel_size

    @property
    def data_parallel_size(self) -> int:
        """Number of independent data replicas in the resolved topology."""

        if self.plan is not None:
            return int(self.plan.data_parallel_size)
        ranks = self.data_parallel_group_ranks
        return len(ranks) if ranks else 1

    @property
    def pipeline_parallel_size(self) -> int:
        if self.plan is not None:
            return int(self.plan.pipeline_parallel_size)
        ranks = self.pipeline_parallel_group_ranks
        return len(ranks) if ranks else 1

    @property
    def expert_parallel_size(self) -> int:
        if self.plan is not None:
            return int(self.plan.expert_parallel_size)
        ranks = self.expert_parallel_group_ranks
        return len(ranks) if ranks else 1

    @property
    def sample_parallel_rank(self) -> int:
        """Rank among independent input microbatches (DP x EP)."""

        return (
            self.data_parallel_rank * self.expert_parallel_size
            + self.expert_parallel_rank
        )

    @property
    def sample_parallel_size(self) -> int:
        data_parallel_size = (
            int(self.plan.data_parallel_size)
            if self.plan is not None
            else len(self.data_parallel_group_ranks)
        )
        return data_parallel_size * self.expert_parallel_size

    @property
    def model_input_src_rank(self) -> int:
        ranks = self.model_input_group_ranks
        if ranks:
            return ranks[0]
        return self.model_parallel_src_rank

    @property
    def model_parallel_src_rank(self) -> int:
        ranks = self.model_parallel_group_ranks
        if ranks:
            return ranks[0]
        return self.context_block_src_rank

    @property
    def uses_context_parallel_attention(self) -> bool:
        """Whether attention actually shards K/V across context ranks."""

        return (
            self.enabled
            and self.kv_backend == "ring"
            and (
                int(self.plan.context_parallel_size)
                if self.plan is not None
                else len(self.context_block_parallel_group_ranks)
            )
            > 1
        )

    @property
    def context_attention_size(self) -> int:
        """Number of ranks participating in true context-parallel attention."""

        if self.uses_context_parallel_attention:
            if self.plan is not None:
                return int(self.plan.context_parallel_size)
            return self.local_parallel_size
        return 1

    @property
    def block_parallel_size(self) -> int:
        if self.plan is not None:
            return int(self.plan.block_parallel_size)
        return self.local_parallel_size

    @property
    def configured_context_parallel_size(self) -> int:
        if self.plan is not None:
            return int(self.plan.context_parallel_size)
        return self.context_attention_size


def local_parallel_enabled_from_config(config: Any) -> bool:
    """Return whether config requests any model-parallel topology."""

    parallel_config = _get_parallel_config(config)
    if parallel_config is None:
        return False
    _reject_removed_enabled_key(parallel_config)
    block_parallel_size = _get_int(parallel_config, "block_parallel_size", 1)
    context_parallel_size = _get_int(parallel_config, "context_parallel_size", 1)
    tensor_parallel_size = _get_int(parallel_config, "tensor_parallel_size", 1)
    expert_parallel_size = _get_int(parallel_config, "expert_parallel_size", 1)
    return (
        block_parallel_size > 1
        or context_parallel_size > 1
        or tensor_parallel_size > 1
        or expert_parallel_size > 1
    )


def data_parallel_coordinates(
    runtime: ParallelRuntime | None,
    *,
    rank: int,
    world_size: int,
    distributed_data_parallel: bool,
) -> tuple[int, int]:
    """Return this process's data-shard coordinate from the resolved topology."""

    plan = getattr(runtime, "plan", None)
    if runtime is not None and plan is not None:
        return (
            int(getattr(runtime, "sample_parallel_rank", 0) or 0),
            int(getattr(runtime, "sample_parallel_size", 1) or 1),
        )
    if distributed_data_parallel:
        return int(rank), max(1, int(world_size))
    return 0, 1


def build_parallel_runtime(config: Any) -> ParallelRuntime:
    """Build rank-local runtime state from ``config.parallel``."""

    parallel_config = _get_parallel_config(config)
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    world_size = (
        dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    )

    if parallel_config is None:
        return ParallelRuntime(
            enabled=False,
            active_block_mode="disabled",
            rank=rank,
            world_size=world_size,
            data_parallel_rank=rank,
            local_parallel_rank=0,
            context_parallel_rank=0,
            block_parallel_rank=0,
            context_block_parallel_group_ranks=[rank],
            data_parallel_group_ranks=[rank],
            dense_data_parallel_group_ranks=[rank],
            tensor_parallel_rank=0,
            tensor_parallel_group_ranks=[rank],
            model_input_group_ranks=[rank],
            model_parallel_group_ranks=[rank],
            sequence_parallel=False,
            tensor_parallel_overlap=True,
            cp_bp_policy=CPBPPolicy(),
            kv_backend="replicated",
        )

    _reject_removed_enabled_key(parallel_config)

    context_parallel_size = _get_int(parallel_config, "context_parallel_size", 1)
    block_parallel_size = _get_int(parallel_config, "block_parallel_size", 1)
    if block_parallel_size <= 0:
        raise ValueError("parallel.block_parallel_size must be positive")
    if context_parallel_size <= 0:
        raise ValueError("parallel.context_parallel_size must be positive")
    tensor_parallel_size = _get_int(parallel_config, "tensor_parallel_size", 1)
    if tensor_parallel_size <= 0:
        raise ValueError("parallel.tensor_parallel_size must be positive")
    sequence_parallel = _get_bool(parallel_config, "sequence_parallel", False)
    tensor_parallel_overlap = _get_bool(
        parallel_config,
        "tensor_parallel_overlap",
        True,
    )
    cp_bp_policy = _cp_bp_policy_from_config(parallel_config)
    if sequence_parallel and tensor_parallel_size <= 1:
        raise ValueError(
            "parallel.sequence_parallel requires parallel.tensor_parallel_size > 1"
        )
    pipeline_parallel_size = _get_int(parallel_config, "pipeline_parallel_size", 1)
    if pipeline_parallel_size <= 0:
        raise ValueError("parallel.pipeline_parallel_size must be positive")
    expert_parallel_size = _get_int(parallel_config, "expert_parallel_size", 1)
    if expert_parallel_size <= 0:
        raise ValueError("parallel.expert_parallel_size must be positive")
    validate_supported_training_axes(
        pipeline_parallel_size=pipeline_parallel_size,
    )
    if not _supported_local_layout(context_parallel_size, block_parallel_size):
        raise ValueError(
            "v1 supports BP-only (parallel.context_parallel_size=1), "
            "CP-only (parallel.block_parallel_size=1), or fused CP/BP "
            "with parallel.block_parallel_size >= parallel.context_parallel_size "
            "and parallel.block_parallel_size divisible by "
            "parallel.context_parallel_size"
        )
    kv_backend = str(parallel_config.get("kv_backend", "replicated"))
    if kv_backend not in {"replicated", "ring"}:
        raise ValueError("parallel.kv_backend must be 'replicated' or 'ring'")
    replicated_attention_backend = str(
        parallel_config.get("replicated_attention_backend", "sdpa")
    )
    if replicated_attention_backend not in {"sdpa", "streaming"}:
        raise ValueError(
            "parallel.replicated_attention_backend must be 'sdpa', or 'streaming'"
        )
    if kv_backend == "replicated" and context_parallel_size != 1:
        raise ValueError(
            "parallel.kv_backend='replicated' requires parallel.context_parallel_size=1"
        )
    if kv_backend == "ring" and context_parallel_size <= 1:
        raise ValueError(
            "parallel.kv_backend='ring' requires parallel.context_parallel_size > 1"
        )
    if _get_str(config, "mode", "train") == "sample_eval":
        raise ValueError("local BP/CP topology is only supported for training/eval")
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
    if world_size % model_parallel_size != 0:
        raise ValueError(
            "world_size must divide evenly by local_parallel_size * "
            "tensor_parallel_size"
        )

    data_parallel_size = parallel_config.get("data_parallel_size", None)
    if data_parallel_size is None:
        data_parallel_size = world_size // model_parallel_size
    data_parallel_size = int(data_parallel_size)
    active_block_mode = str(parallel_config.get("active_block_mode", "all_blocks"))
    if active_block_mode not in {"disabled", "all_blocks", "dual_end"}:
        raise ValueError(
            "parallel.active_block_mode must be 'disabled', 'all_blocks', or 'dual_end'"
        )
    if active_block_mode == "dual_end" and block_parallel_size <= 1:
        raise ValueError(
            "parallel.active_block_mode='dual_end' requires block_parallel_size > 1"
        )
    if active_block_mode != "dual_end" and block_parallel_size > 1:
        raise ValueError(
            "complete target-block ownership requires active_block_mode='dual_end'"
        )
    num_blocks = _infer_num_blocks_from_config(config, parallel_config)
    node_size = _infer_node_size()
    placement_policy = str(parallel_config.get("placement_policy", "auto"))
    plan = build_parallel_plan(
        num_blocks=num_blocks,
        world_size=world_size,
        data_parallel_size=data_parallel_size,
        context_parallel_size=context_parallel_size,
        block_parallel_size=block_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        expert_parallel_size=expert_parallel_size,
        node_size=node_size,
        placement_policy=placement_policy,
    )
    process_groups = _build_process_group_collection(
        plan,
        rank,
        _process_group_options_from_config(parallel_config),
    )
    assignment = process_groups.assignment

    return ParallelRuntime(
        enabled=True,
        active_block_mode=active_block_mode,
        rank=rank,
        world_size=world_size,
        data_parallel_rank=assignment.data_parallel_rank,
        local_parallel_rank=assignment.local_parallel_rank,
        context_parallel_rank=assignment.context_parallel_rank,
        block_parallel_rank=assignment.block_parallel_rank,
        context_block_parallel_group_ranks=(assignment.context_block_parallel_group),
        data_parallel_group_ranks=assignment.data_parallel_group,
        dense_data_parallel_group_ranks=assignment.dense_data_parallel_group,
        clean_replica_group_ranks=assignment.clean_replica_group,
        tensor_parallel_rank=assignment.tensor_parallel_rank,
        pipeline_parallel_rank=assignment.pipeline_parallel_rank,
        expert_parallel_rank=assignment.expert_parallel_rank,
        tensor_parallel_group_ranks=assignment.tensor_parallel_group,
        pipeline_parallel_group_ranks=assignment.pipeline_parallel_group,
        expert_parallel_group_ranks=assignment.expert_parallel_group,
        node_size=node_size,
        model_input_group_ranks=assignment.model_input_group,
        model_parallel_group_ranks=assignment.model_parallel_group,
        sequence_parallel=sequence_parallel,
        tensor_parallel_overlap=tensor_parallel_overlap,
        cp_bp_policy=cp_bp_policy,
        kv_backend=kv_backend,
        context_block_parallel_group=process_groups.context_block_parallel_group,
        clean_replica_group=process_groups.clean_replica_group,
        data_parallel_group=process_groups.data_parallel_group,
        tensor_parallel_group=process_groups.tensor_parallel_group,
        pipeline_parallel_group=process_groups.pipeline_parallel_group,
        expert_parallel_group=process_groups.expert_parallel_group,
        model_input_group=process_groups.model_input_group,
        model_parallel_group=process_groups.model_parallel_group,
        process_groups=process_groups,
        plan=plan,
    )


def warm_parallel_runtime_collectives(runtime: ParallelRuntime | None) -> None:
    """Eagerly initialize NCCL communicators for all runtime process groups.

    Fused CP/BP/TP runs touch different process groups in different phases:
    batch broadcast in forward setup, CP/BP P2P/A2A during attention, TP/SP
    collectives in layer backward, and ZeRO/DP collectives in the optimizer
    tail. Warming every configured group before model activations are allocated
    prevents lazy communicator scratch allocation from appearing as a late-step
    HBM spike.
    """

    if (
        runtime is None
        or not runtime.enabled
        or not dist.is_available()
        or not dist.is_initialized()
        or not torch.cuda.is_available()
    ):
        return
    device = torch.device("cuda", torch.cuda.current_device())
    token = torch.zeros((), device=device)
    seen: set[int] = set()
    for group in (
        runtime.context_block_parallel_group,
        runtime.clean_replica_group,
        runtime.data_parallel_group,
        runtime.tensor_parallel_group,
        runtime.pipeline_parallel_group,
        runtime.expert_parallel_group,
        runtime.model_input_group,
        runtime.model_parallel_group,
    ):
        if group is None:
            continue
        group_id = id(group)
        if group_id in seen:
            continue
        seen.add(group_id)
        try:
            world_size = dist.get_world_size(group)
        except Exception:
            world_size = 1
        if world_size <= 1:
            continue
        dist.all_reduce(token, op=dist.ReduceOp.SUM, group=group)
    torch.cuda.synchronize(device)


def destroy_parallel_runtime_process_groups(runtime: ParallelRuntime | None) -> None:
    """Destroy non-world process groups created for a runtime."""

    if runtime is None or runtime.process_groups is None:
        return
    runtime.process_groups.destroy()


def broadcast_batch_to_context_block_group(
    batch: dict[str, Any],
    runtime: ParallelRuntime,
) -> dict[str, Any]:
    """Broadcast tensor batch values from the model-parallel source rank."""

    if not _should_use_collectives(runtime):
        return batch

    return {key: _broadcast_value(value, runtime) for key, value in batch.items()}


def broadcast_tensor_to_model_parallel_group(
    value: torch.Tensor,
    runtime: ParallelRuntime,
) -> torch.Tensor:
    """Broadcast one tensor from the model-parallel source rank."""

    if not _should_use_collectives(runtime):
        return value
    dist.broadcast(
        value,
        src=runtime.model_input_src_rank,
        group=runtime.model_input_group,
    )
    return value


def clean_replica_tensor(value: torch.Tensor, runtime: ParallelRuntime) -> torch.Tensor:
    """Broadcast owner clean states across BP replicas with gradient return.

    Clean-token states are identical for BP workers that share a context rank.
    Forward broadcasts the owner state to the replicas; backward sums all
    replica gradients back to the owner and returns zero to non-owner duplicate
    clean paths.
    """

    return begin_clean_replica_tensor(value, runtime).wait()


def begin_clean_replica_tensor(
    value: torch.Tensor,
    runtime: ParallelRuntime,
) -> PendingCleanReplicaTensor:
    """Launch clean-state sharing and return a waitable tensor handle."""

    if (
        not _should_use_collectives(runtime)
        or runtime.clean_replica_size <= 1
        or runtime.clean_replica_group is None
    ):
        return PendingCleanReplicaTensor(
            output=value,
            owner_input=value,
            work=None,
            src_rank=int(getattr(runtime, "clean_replica_src_rank", 0) or 0),
            replica_rank=int(getattr(runtime, "clean_replica_rank", 0) or 0),
            replica_size=int(getattr(runtime, "clean_replica_size", 1) or 1),
            group=None,
        )
    replica_rank = int(runtime.clean_replica_rank)
    with torch.no_grad():
        output = (
            value.detach().contiguous()
            if replica_rank == 0
            else torch.empty_like(value)
        )
        work = dist.broadcast(
            output,
            src=int(runtime.clean_replica_src_rank),
            group=runtime.clean_replica_group,
            async_op=True,
        )
    return PendingCleanReplicaTensor(
        output=output,
        owner_input=value,
        work=work,
        src_rank=int(runtime.clean_replica_src_rank),
        replica_rank=replica_rank,
        replica_size=int(runtime.clean_replica_size),
        group=runtime.clean_replica_group,
    )


def active_token_mask(
    *,
    seq_len: int,
    block_size: int,
    device: torch.device,
    runtime: ParallelRuntime,
) -> torch.Tensor | None:
    """Return the blocks owned by this block-parallel worker.

    Context parallelism shards clean context state and attention transport, not
    the independent target-block objectives.  Consequently only the explicit
    block-parallel ``dual_end`` mode may construct a target-token ownership
    mask. ``all_blocks`` leaves target partitioning to the model's standard
    token-row context-parallel layout.
    """

    if not _uses_block_objective_ownership(runtime):
        return None
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if seq_len % block_size != 0:
        raise ValueError("seq_len must divide evenly by block_size")

    active_blocks = active_blocks_for_runtime(
        seq_len=seq_len,
        block_size=block_size,
        runtime=runtime,
    )
    if active_blocks is None:
        raise RuntimeError("block-objective ownership changed during mask construction")
    mask = torch.zeros(seq_len, dtype=torch.float32, device=device)
    if active_blocks:
        block_starts = torch.tensor(active_blocks, dtype=torch.long, device=device)
        offsets = torch.arange(block_size, dtype=torch.long, device=device)
        indices = (block_starts[:, None] * block_size + offsets[None]).reshape(-1)
        mask[indices] = 1.0
    return mask


class _CleanReplicaBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        output: torch.Tensor,
        value: torch.Tensor,
        src_rank: int,
        replica_rank: int,
        replica_size: int,
        group: Any,
    ) -> torch.Tensor:
        ctx.replica_rank = int(replica_rank)
        ctx.replica_size = int(replica_size)
        ctx.group = group
        ctx.input_shape = tuple(value.shape)
        return output

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[None, torch.Tensor | None, None, None, None, None]:
        if int(ctx.replica_size) <= 1:
            return None, grad_output, None, None, None, None
        grad = grad_output.contiguous()
        work = dist.all_reduce(
            grad,
            op=dist.ReduceOp.SUM,
            group=ctx.group,
            async_op=True,
        )
        work.wait()
        if int(ctx.replica_rank) == 0:
            return None, grad, None, None, None, None
        return None, torch.zeros_like(grad_output), None, None, None, None


def active_token_indices(
    *,
    seq_len: int,
    block_size: int,
    device: torch.device,
    runtime: ParallelRuntime,
) -> torch.Tensor | None:
    """Return target-token indices owned by this block-parallel worker."""

    if not _uses_block_objective_ownership(runtime):
        return None
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if seq_len % block_size != 0:
        raise ValueError("seq_len must divide evenly by block_size")

    active_blocks = active_blocks_for_runtime(
        seq_len=seq_len,
        block_size=block_size,
        runtime=runtime,
    )
    if active_blocks is None:
        raise RuntimeError("block-objective ownership changed during index construction")
    device_index = device.index
    if device.type == "cuda" and device_index is None:
        device_index = torch.cuda.current_device()
    cache_key = (
        int(seq_len),
        int(block_size),
        int(runtime.block_parallel_size),
        int(runtime.configured_context_parallel_size),
        int(runtime.context_parallel_rank),
        int(runtime.block_parallel_rank),
        tuple(int(block) for block in active_blocks),
        device.type,
        device_index,
    )
    cached = _ACTIVE_TOKEN_INDICES_CACHE.get(cache_key)
    if cached is not None and cached.device == device:
        return cached
    if not active_blocks:
        indices = torch.empty(0, dtype=torch.long, device=device)
        _ACTIVE_TOKEN_INDICES_CACHE[cache_key] = indices
        return indices
    block_starts = torch.tensor(active_blocks, dtype=torch.long, device=device)
    offsets = torch.arange(block_size, dtype=torch.long, device=device)
    indices = (block_starts[:, None] * block_size + offsets[None]).reshape(-1)
    _ACTIVE_TOKEN_INDICES_CACHE[cache_key] = indices
    return indices


def active_clean_prefix_length(
    *,
    seq_len: int,
    block_size: int,
    runtime: ParallelRuntime,
) -> int | None:
    """Return the clean-prefix extent required by this BP worker.

    The ownership schedule is constructed on the host. Carrying this metadata
    with the corresponding CUDA token indices avoids synchronizing the device
    merely to recover the largest owned block during model execution.
    """

    if not _uses_block_objective_ownership(runtime):
        return None
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if seq_len % block_size != 0:
        raise ValueError("seq_len must divide evenly by block_size")
    active_blocks = active_blocks_for_runtime(
        seq_len=seq_len,
        block_size=block_size,
        runtime=runtime,
    )
    if active_blocks is None:
        raise RuntimeError("block-objective ownership changed during prefix planning")
    if not active_blocks:
        return 0
    return int(active_blocks[-1]) * int(block_size)


def loss_scale(runtime: ParallelRuntime) -> float:
    """Scale local shard loss so distributed averaging preserves full gradients."""

    if _uses_block_objective_ownership(runtime):
        return float(runtime.block_parallel_size)
    return 1.0


def _uses_block_objective_ownership(runtime: ParallelRuntime) -> bool:
    """Whether this rank owns complete target-block objective terms."""

    return (
        runtime.enabled
        and runtime.active_block_mode == "dual_end"
        and runtime.block_parallel_size > 1
    )


def active_blocks_for_runtime(
    *,
    seq_len: int,
    block_size: int,
    runtime: ParallelRuntime,
) -> tuple[int, ...] | None:
    """Return host-resident target-block ownership for this runtime.

    The schedule is static for a sequence length and topology. Model code that
    needs block metadata should use this helper instead of recovering it from
    CUDA token-index tensors, which would introduce a device synchronization.
    """

    if not _uses_block_objective_ownership(runtime):
        return None
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if seq_len % block_size != 0:
        raise ValueError("seq_len must divide evenly by block_size")
    return _cached_active_blocks(
        seq_len,
        block_size,
        int(runtime.block_parallel_size),
        int(runtime.configured_context_parallel_size),
        int(runtime.block_parallel_rank),
    )


@lru_cache(maxsize=128)
def _cached_active_blocks(
    seq_len: int,
    block_size: int,
    block_parallel_size: int,
    context_parallel_size: int,
    block_parallel_rank: int,
) -> tuple[int, ...]:
    num_blocks = seq_len // block_size
    schedule = build_block_schedule(
        num_blocks=num_blocks,
        block_parallel_size=block_parallel_size,
        context_parallel_size=context_parallel_size,
    )
    return tuple(sorted(schedule.active_blocks_by_worker[block_parallel_rank]))


def _new_current_group(
    plan: ParallelPlan,
    rank: int,
    group_attr: str,
    options: DLLMProcessGroupOptions | None = None,
) -> Any:
    options = options or DLLMProcessGroupOptions()
    if group_attr == "optimizer_data_parallel_group":
        from dllm_parallel.core.parallel.groups import (
            current_group_from_rank_lists,
            optimizer_data_parallel_group_ranks,
        )

        return current_group_from_rank_lists(
            rank,
            [
                optimizer_data_parallel_group_ranks(plan, assignment)
                for assignment in plan.rank_assignments
            ],
            timeout=options.timeout_for(group_attr),
            create_singleton_groups=True,
        )
    return current_group_from_plan(
        plan,
        rank,
        group_attr,
        timeout=options.timeout_for(group_attr),
    )


def _build_process_group_collection(
    plan: ParallelPlan,
    rank: int,
    options: DLLMProcessGroupOptions | None = None,
) -> DLLMProcessGroupCollection:
    options = options or DLLMProcessGroupOptions()

    def build_group(plan: ParallelPlan, rank: int, group_attr: str) -> Any:
        return _new_current_group(plan, rank, group_attr, options)

    return DLLMProcessGroupCollection.from_plan(
        plan,
        rank=rank,
        group_builder=build_group,
        options=options,
    )


def _process_group_options_from_config(config: Any) -> DLLMProcessGroupOptions:
    group_timeouts = _get_mapping(config, "process_group_timeouts")
    return DLLMProcessGroupOptions(
        default_timeout_seconds=_get_optional_float(
            config,
            "process_group_timeout_seconds",
        ),
        group_timeout_seconds=group_timeouts,
    )


def _cp_bp_policy_from_config(config: Any) -> CPBPPolicy:
    cp_bp = _get_node(config, "cp_bp") or _get_node(config, "cp_bp_policy")
    attention_policy = _get_str(
        cp_bp,
        "attention_policy",
        _get_str(config, "cp_bp_attention_policy", "production"),
    )
    ragged_prefix = _get_str(
        cp_bp,
        "ragged_prefix",
        _get_str(config, "cp_bp_ragged_prefix", "false"),
    )
    clean_kv_dtype = _get_str(
        cp_bp,
        "clean_kv_dtype",
        _get_str(config, "cp_bp_clean_kv_dtype", "bf16"),
    )
    clean_kv_layout = _get_str(
        cp_bp,
        "clean_kv_layout",
        _get_str(config, "cp_bp_clean_kv_layout", "zigzag"),
    )
    clean_kv_transport = _get_str(
        cp_bp,
        "clean_kv_transport",
        _get_str(config, "cp_bp_clean_kv_transport", "collective"),
    )
    if attention_policy != "production":
        raise ValueError("parallel.cp_bp.attention_policy must be production")
    if ragged_prefix not in {"false", "true"}:
        raise ValueError("parallel.cp_bp.ragged_prefix must be false or true")
    if clean_kv_dtype != "bf16":
        raise ValueError(
            f"parallel.cp_bp.clean_kv_dtype={clean_kv_dtype} is not supported; "
            "expected bf16"
        )
    if clean_kv_layout not in {"contiguous", "zigzag"}:
        raise ValueError("parallel.cp_bp.clean_kv_layout must be contiguous or zigzag")
    if clean_kv_transport not in {"collective", "streaming"}:
        raise ValueError(
            "parallel.cp_bp.clean_kv_transport must be collective or streaming"
        )
    return CPBPPolicy(
        attention_policy=attention_policy,  # type: ignore[arg-type]
        ragged_prefix=ragged_prefix,  # type: ignore[arg-type]
        clean_kv_dtype=clean_kv_dtype,  # type: ignore[arg-type]
        clean_kv_layout=clean_kv_layout,  # type: ignore[arg-type]
        clean_kv_transport=clean_kv_transport,  # type: ignore[arg-type]
        clean_reuse_overlap=_get_bool(
            cp_bp,
            "clean_reuse_overlap",
            _get_bool(config, "cp_bp_clean_reuse_overlap", False),
        ),
        debug_nonfinite_attention=_get_bool(
            cp_bp,
            "debug_nonfinite_attention",
            _get_bool(config, "cp_bp_debug_nonfinite_attention", False),
        ),
    )


def _should_use_collectives(runtime: ParallelRuntime) -> bool:
    return (
        runtime.enabled
        and runtime.model_parallel_size > 1
        and dist.is_available()
        and dist.is_initialized()
    )


def _infer_node_size() -> int | None:
    for name in ("LOCAL_WORLD_SIZE", "SLURM_GPUS_ON_NODE"):
        value = os.environ.get(name)
        if value is None:
            continue
        try:
            parsed = int(value)
        except ValueError:
            continue
        if parsed > 0:
            return parsed
    return None


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


def _broadcast_value(value: Any, runtime: ParallelRuntime) -> Any:
    if torch.is_tensor(value):
        dist.broadcast(
            value,
            src=runtime.model_input_src_rank,
            group=runtime.model_input_group,
        )
    return value


def _get_parallel_config(config: Any) -> Any | None:
    if hasattr(config, "get"):
        return config.get("parallel", None)
    return getattr(config, "parallel", None)


def _infer_num_blocks_from_config(config: Any, parallel_config: Any) -> int:
    explicit_num_blocks = _get_optional_int(parallel_config, "num_blocks")
    if explicit_num_blocks is not None:
        if explicit_num_blocks <= 0:
            raise ValueError("parallel.num_blocks must be positive")
        return explicit_num_blocks

    model_config = _get_node(config, "model")
    block_size = _get_optional_int(config, "block_size") or _get_optional_int(
        model_config, "block_size"
    )
    seq_len = (
        _get_optional_int(model_config, "length")
        or _get_optional_int(model_config, "max_position_embeddings")
        or _get_optional_int(config, "max_position_embeddings")
    )
    if block_size is None or seq_len is None:
        raise ValueError(
            "parallel runtime needs parallel.num_blocks or model sequence "
            "metadata plus block_size"
        )
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if seq_len % block_size != 0:
        raise ValueError("sequence length must divide evenly by block_size")
    return int(seq_len) // int(block_size)


def _get_node(node: Any, key: str) -> Any:
    if node is None:
        return None
    if hasattr(node, "get"):
        return node.get(key, None)
    return getattr(node, key, None)


def _get_int(node: Any, key: str, default: int) -> int:
    if node is None:
        return int(default)
    if hasattr(node, "get"):
        value = node.get(key, default)
    else:
        value = getattr(node, key, default)
    return int(default if value is None else value)


def _get_bool(node: Any, key: str, default: bool) -> bool:
    if node is None:
        return bool(default)
    if hasattr(node, "get"):
        value = node.get(key, default)
    else:
        value = getattr(node, key, default)
    if value is None:
        return bool(default)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on", "enabled"}:
            return True
        if lowered in {"0", "false", "no", "off", "disabled"}:
            return False
    return bool(value)


def _get_str(node: Any, key: str, default: str) -> str:
    if node is None:
        return str(default)
    if hasattr(node, "get"):
        value = node.get(key, default)
    else:
        value = getattr(node, key, default)
    return str(default if value is None else value)


def _get_optional_int(node: Any, key: str) -> int | None:
    if node is None:
        return None
    if hasattr(node, "get"):
        value = node.get(key, None)
    else:
        value = getattr(node, key, None)
    if value is None:
        return None
    return int(value)


def _get_optional_float(node: Any, key: str) -> float | None:
    if node is None:
        return None
    if hasattr(node, "get"):
        value = node.get(key, None)
    else:
        value = getattr(node, key, None)
    if value is None:
        return None
    return float(value)


def _get_mapping(node: Any, key: str) -> dict[str, float] | None:
    if node is None:
        return None
    if hasattr(node, "get"):
        value = node.get(key, None)
    else:
        value = getattr(node, key, None)
    if value is None:
        return None
    return {str(name): float(seconds) for name, seconds in dict(value).items()}


def _reject_removed_enabled_key(parallel_config: Any) -> None:
    try:
        has_enabled = "enabled" in parallel_config
    except TypeError:
        has_enabled = hasattr(parallel_config, "enabled")
    if has_enabled:
        raise ValueError(
            "parallel.enabled has been removed; set "
            "parallel.block_parallel_size, parallel.context_parallel_size, "
            "and parallel.tensor_parallel_size to request a parallel topology"
        )
