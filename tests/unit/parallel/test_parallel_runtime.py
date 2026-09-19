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
import torch

import dllm_parallel.core.parallel.runtime as parallel_runtime
from dllm_parallel.core.parallel.runtime import (
    ParallelRuntime,
    active_blocks_for_runtime,
    active_clean_prefix_length,
    active_token_indices,
    active_token_mask,
    build_parallel_runtime,
    loss_scale,
)


def _runtime(
    block_rank: int,
    mode: str = "dual_end",
    kv_backend: str = "replicated",
) -> ParallelRuntime:
    return ParallelRuntime(
        enabled=True,
        active_block_mode=mode,
        rank=block_rank,
        world_size=4,
        data_parallel_rank=0,
        local_parallel_rank=block_rank,
        context_parallel_rank=block_rank,
        block_parallel_rank=block_rank,
        context_block_parallel_group_ranks=[0, 1, 2, 3],
        data_parallel_group_ranks=[block_rank],
        kv_backend=kv_backend,
    )


def test_dual_end_active_token_mask_for_four_way_runtime() -> None:
    mask = active_token_mask(
        seq_len=32,
        block_size=1,
        device=torch.device("cpu"),
        runtime=_runtime(block_rank=0),
    )

    assert mask is not None
    assert torch.nonzero(mask).squeeze(-1).tolist() == [
        0,
        1,
        2,
        3,
        28,
        29,
        30,
        31,
    ]


def test_dual_end_active_token_indices_are_cached_per_runtime_device() -> None:
    runtime = _runtime(block_rank=0)

    first = active_token_indices(
        seq_len=32,
        block_size=1,
        device=torch.device("cpu"),
        runtime=runtime,
    )
    second = active_token_indices(
        seq_len=32,
        block_size=1,
        device=torch.device("cpu"),
        runtime=runtime,
    )

    assert first is second
    assert first.tolist() == [0, 1, 2, 3, 28, 29, 30, 31]


def test_dual_end_clean_prefix_length_comes_from_host_schedule() -> None:
    assert active_blocks_for_runtime(
        seq_len=32,
        block_size=1,
        runtime=_runtime(block_rank=0),
    ) == (0, 1, 2, 3, 28, 29, 30, 31)
    assert active_clean_prefix_length(
        seq_len=32,
        block_size=1,
        runtime=_runtime(block_rank=0),
    ) == 31
    assert active_clean_prefix_length(
        seq_len=32,
        block_size=1,
        runtime=_runtime(block_rank=1),
    ) == 27
    assert active_clean_prefix_length(
        seq_len=32,
        block_size=1,
        runtime=_runtime(block_rank=0, mode="all_blocks"),
    ) is None
    assert active_blocks_for_runtime(
        seq_len=32,
        block_size=1,
        runtime=_runtime(block_rank=0, mode="all_blocks"),
    ) is None


def test_all_blocks_mode_does_not_construct_block_ownership() -> None:
    mask = active_token_mask(
        seq_len=32,
        block_size=1,
        device=torch.device("cpu"),
        runtime=_runtime(block_rank=0, mode="all_blocks"),
    )

    assert mask is None
    assert loss_scale(_runtime(block_rank=0, mode="all_blocks")) == 1.0


def test_disabled_mode_never_constructs_packed_active_ownership() -> None:
    runtime = _runtime(block_rank=0, mode="disabled", kv_backend="ring")

    assert active_token_indices(
        seq_len=32,
        block_size=1,
        device=torch.device("cpu"),
        runtime=runtime,
    ) is None
    assert active_token_mask(
        seq_len=32,
        block_size=1,
        device=torch.device("cpu"),
        runtime=runtime,
    ) is None


def test_dual_end_loss_scale_offsets_ddp_world_averaging() -> None:
    assert loss_scale(_runtime(block_rank=0)) == 4.0


def test_active_token_mask_rejects_partial_blocks() -> None:
    with pytest.raises(ValueError, match="seq_len"):
        active_token_mask(
            seq_len=31,
            block_size=4,
            device=torch.device("cpu"),
            runtime=_runtime(block_rank=0),
        )


def test_replicated_kv_is_not_context_parallel_attention() -> None:
    runtime = _runtime(block_rank=0, kv_backend="replicated")

    assert not runtime.uses_context_parallel_attention
    assert runtime.context_attention_size == 1
    assert runtime.local_parallel_size == 4


def test_ring_kv_is_context_parallel_attention() -> None:
    runtime = _runtime(block_rank=0, kv_backend="ring")

    assert runtime.uses_context_parallel_attention
    assert runtime.context_attention_size == 4


def test_runtime_rejects_removed_parallel_enabled_flag() -> None:
    config = ({
        "mode": "train",
        "algo": {"name": "standard_block_diffusion"},
        "model": {"length": 32},
        "block_size": 4,
        "parallel": {
            "enabled": True,
            "data_parallel_size": 1,
            "context_parallel_size": 1,
            "block_parallel_size": 1,
            "active_block_mode": "all_blocks",
            "kv_backend": "ring",
        },
    })

    with pytest.raises(ValueError, match="parallel.enabled has been removed"):
        build_parallel_runtime(config)


def test_runtime_rejects_ring_kv_with_single_context_rank() -> None:
    config = ({
        "mode": "train",
        "algo": {"name": "standard_block_diffusion"},
        "model": {"length": 32},
        "block_size": 4,
        "parallel": {
            "data_parallel_size": 1,
            "context_parallel_size": 1,
            "block_parallel_size": 1,
            "active_block_mode": "dual_end",
            "kv_backend": "ring",
        },
    })

    with pytest.raises(ValueError, match="context_parallel_size > 1"):
        build_parallel_runtime(config)


def test_runtime_rejects_replicated_kv_with_context_parallel_rank() -> None:
    config = ({
        "mode": "train",
        "algo": {"name": "standard_block_diffusion"},
        "model": {"length": 32},
        "block_size": 4,
        "parallel": {
            "data_parallel_size": 2,
            "context_parallel_size": 2,
            "block_parallel_size": 2,
            "active_block_mode": "dual_end",
            "kv_backend": "replicated",
        },
    })

    with pytest.raises(ValueError, match="context_parallel_size=1"):
        build_parallel_runtime(config)


def test_runtime_builds_plain_dp_plan_when_model_parallel_sizes_are_one() -> None:
    config = ({
        "mode": "train",
        "algo": {"name": "standard_block_diffusion"},
        "model": {"length": 32},
        "block_size": 4,
        "parallel": {
            "data_parallel_size": None,
            "context_parallel_size": 1,
            "block_parallel_size": 1,
            "active_block_mode": "all_blocks",
            "kv_backend": "replicated",
        },
    })

    runtime = build_parallel_runtime(config)

    assert runtime.enabled
    assert runtime.plan is not None
    assert runtime.plan.data_parallel_size == 1
    assert runtime.data_parallel_size == 1
    assert runtime.active_block_mode == "all_blocks"
    mask = active_token_mask(
        seq_len=32,
        block_size=4,
        device=torch.device("cpu"),
        runtime=runtime,
    )
    assert mask is None
    assert loss_scale(runtime) == 1.0


def test_runtime_accepts_replicated_bp_with_single_context_rank(monkeypatch) -> None:
    monkeypatch.setattr(parallel_runtime.dist, "is_available", lambda: True)
    monkeypatch.setattr(parallel_runtime.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(parallel_runtime.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(parallel_runtime.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(parallel_runtime, "_new_current_group", lambda *args, **kwargs: None)
    config = ({
        "mode": "train",
        "algo": {"name": "standard_block_diffusion"},
        "model": {"length": 32},
        "block_size": 4,
        "parallel": {
            "data_parallel_size": 2,
            "context_parallel_size": 1,
            "block_parallel_size": 2,
            "active_block_mode": "dual_end",
            "kv_backend": "replicated",
        },
    })

    runtime = build_parallel_runtime(config)

    assert runtime.local_parallel_size == 2
    assert runtime.data_parallel_size == 2
    assert runtime.block_parallel_size == 2
    assert runtime.configured_context_parallel_size == 1
    assert runtime.context_parallel_rank == 0
    assert runtime.block_parallel_rank == 0
    assert runtime.context_block_parallel_group_ranks == [0, 1]
    assert runtime.data_parallel_group_ranks == [0, 2]
    assert not runtime.uses_context_parallel_attention
    assert runtime.context_attention_size == 1
    assert loss_scale(runtime) == 2.0


def test_runtime_attaches_process_group_collection(monkeypatch) -> None:
    monkeypatch.setattr(parallel_runtime.dist, "is_available", lambda: True)
    monkeypatch.setattr(parallel_runtime.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(parallel_runtime.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(parallel_runtime.dist, "get_world_size", lambda: 4)

    def fake_group(plan, rank, group_attr, options=None):
        assert rank == 1
        assert options is not None
        return (group_attr, rank)

    monkeypatch.setattr(parallel_runtime, "_new_current_group", fake_group)
    config = ({
        "mode": "train",
        "algo": {"name": "standard_block_diffusion"},
        "model": {"length": 32},
        "block_size": 4,
        "parallel": {
            "data_parallel_size": 2,
            "context_parallel_size": 1,
            "block_parallel_size": 2,
            "active_block_mode": "dual_end",
            "kv_backend": "replicated",
            "process_group_timeout_seconds": 900,
            "process_group_timeouts": {"model_parallel_group": 1200},
        },
    })

    runtime = build_parallel_runtime(config)

    assert runtime.process_groups is not None
    assert runtime.process_groups.plan is runtime.plan
    assert runtime.context_block_parallel_group == (
        "context_block_parallel_group",
        1,
    )
    assert runtime.model_parallel_group == ("model_parallel_group", 1)
    assert runtime.process_groups.optimizer_data_parallel_group == (
        "optimizer_data_parallel_group",
        1,
    )
    assert runtime.process_groups.optimizer_data_parallel_group_ranks == [0, 1, 2, 3]
    assert runtime.process_groups.zero_partitions_sample_parallel
    assert (
        runtime.process_groups.options.default_timeout_seconds
        == 900
    )
    assert (
        runtime.process_groups.options.group_timeout_seconds["model_parallel_group"]
        == 1200
    )


def test_runtime_resolves_typed_cp_bp_policy(monkeypatch) -> None:
    monkeypatch.setattr(parallel_runtime.dist, "is_available", lambda: True)
    monkeypatch.setattr(parallel_runtime.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(parallel_runtime.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(parallel_runtime.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(parallel_runtime, "_new_current_group", lambda *args, **kwargs: None)
    config = ({
        "mode": "train",
        "algo": {"name": "standard_block_diffusion"},
        "model": {"length": 32},
        "block_size": 4,
        "parallel": {
            "data_parallel_size": 1,
            "context_parallel_size": 4,
            "block_parallel_size": 4,
            "active_block_mode": "dual_end",
            "kv_backend": "ring",
            "cp_bp": {
                "attention_policy": "production",
                "ragged_prefix": "true",
                "clean_kv_layout": "contiguous",
                "clean_kv_dtype": "bf16",
                "clean_reuse_overlap": True,
                "debug_nonfinite_attention": True,
            },
        },
    })

    runtime = build_parallel_runtime(config)

    assert runtime.cp_bp_policy.attention_policy == "production"
    assert runtime.cp_bp_policy.ragged_prefix == "true"
    assert runtime.cp_bp_policy.clean_kv_layout == "contiguous"
    assert runtime.cp_bp_policy.clean_reuse_overlap
    assert runtime.cp_bp_policy.debug_nonfinite_attention
    assert runtime.cp_bp_policy.to_log_dict()["attention_policy"] == "production"
    assert runtime.cp_bp_policy.to_log_dict()["backward_query_chunking"] == "disabled"


def test_runtime_rejects_unvalidated_cp_bp_attention_policy(monkeypatch) -> None:
    monkeypatch.setattr(parallel_runtime.dist, "is_available", lambda: True)
    monkeypatch.setattr(parallel_runtime.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(parallel_runtime.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(parallel_runtime.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(parallel_runtime, "_new_current_group", lambda *args, **kwargs: None)
    config = ({
        "mode": "train",
        "algo": {"name": "standard_block_diffusion"},
        "model": {"length": 32},
        "block_size": 4,
        "parallel": {
            "data_parallel_size": 1,
            "context_parallel_size": 4,
            "block_parallel_size": 4,
            "active_block_mode": "dual_end",
            "kv_backend": "ring",
            "cp_bp": {"attention_policy": "experimental"},
        },
    })

    with pytest.raises(ValueError, match="attention_policy must be production"):
        build_parallel_runtime(config)


def test_runtime_rejects_unimplemented_fp8_clean_kv_policy(monkeypatch) -> None:
    monkeypatch.setattr(parallel_runtime.dist, "is_available", lambda: True)
    monkeypatch.setattr(parallel_runtime.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(parallel_runtime.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(parallel_runtime.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(parallel_runtime, "_new_current_group", lambda *args, **kwargs: None)
    config = ({
        "mode": "train",
        "algo": {"name": "standard_block_diffusion"},
        "model": {"length": 32},
        "block_size": 4,
        "parallel": {
            "data_parallel_size": 1,
            "context_parallel_size": 4,
            "block_parallel_size": 4,
            "active_block_mode": "dual_end",
            "kv_backend": "ring",
            "cp_bp": {"clean_kv_dtype": "fp8"},
        },
    })

    with pytest.raises(ValueError, match="clean_kv_dtype=fp8"):
        build_parallel_runtime(config)


def test_runtime_accepts_ring_cp_with_single_block_rank(monkeypatch) -> None:
    monkeypatch.setattr(parallel_runtime.dist, "is_available", lambda: True)
    monkeypatch.setattr(parallel_runtime.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(parallel_runtime.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(parallel_runtime.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(parallel_runtime, "_new_current_group", lambda *args, **kwargs: None)
    config = ({
        "mode": "train",
        "algo": {"name": "standard_block_diffusion"},
        "model": {"length": 32},
        "block_size": 4,
        "parallel": {
            "data_parallel_size": 2,
            "context_parallel_size": 2,
            "block_parallel_size": 1,
            "active_block_mode": "all_blocks",
            "kv_backend": "ring",
        },
    })

    runtime = build_parallel_runtime(config)

    assert runtime.local_parallel_size == 2
    assert runtime.block_parallel_size == 1
    assert runtime.configured_context_parallel_size == 2
    assert runtime.context_parallel_rank == 0
    assert runtime.block_parallel_rank == 0
    assert runtime.context_block_parallel_group_ranks == [0, 1]
    assert runtime.data_parallel_group_ranks == [0, 2]
    assert runtime.uses_context_parallel_attention
    assert runtime.context_attention_size == 2
    assert loss_scale(runtime) == 1.0


def test_runtime_accepts_more_bp_workers_than_cp_ranks(monkeypatch) -> None:
    monkeypatch.setattr(parallel_runtime.dist, "is_available", lambda: True)
    monkeypatch.setattr(parallel_runtime.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(parallel_runtime.dist, "get_rank", lambda: 2)
    monkeypatch.setattr(parallel_runtime.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(parallel_runtime, "_new_current_group", lambda *args, **kwargs: None)
    config = ({
        "mode": "train",
        "algo": {"name": "standard_block_diffusion"},
        "model": {"length": 32},
        "block_size": 4,
        "parallel": {
            "data_parallel_size": 1,
            "context_parallel_size": 2,
            "block_parallel_size": 4,
            "active_block_mode": "dual_end",
            "kv_backend": "ring",
        },
    })

    runtime = build_parallel_runtime(config)
    idx = active_token_indices(
        seq_len=32,
        block_size=4,
        device=torch.device("cpu"),
        runtime=runtime,
    )

    assert runtime.local_parallel_size == 4
    assert runtime.model_parallel_size == 4
    assert runtime.block_parallel_size == 4
    assert runtime.configured_context_parallel_size == 2
    assert runtime.context_parallel_rank == 0
    assert runtime.block_parallel_rank == 2
    assert runtime.context_block_parallel_group_ranks == [2, 3]
    assert runtime.model_parallel_group_ranks == [0, 1, 2, 3]
    assert runtime.uses_context_parallel_attention
    assert runtime.context_attention_size == 2
    assert idx is not None and idx.tolist() == list(range(8, 12)) + list(range(20, 24))
    assert loss_scale(runtime) == 4.0


def test_runtime_keeps_equal_cp_bp_as_fused_dual_end(monkeypatch) -> None:
    monkeypatch.setattr(parallel_runtime.dist, "is_available", lambda: True)
    monkeypatch.setattr(parallel_runtime.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(parallel_runtime.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(parallel_runtime.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(parallel_runtime, "_new_current_group", lambda *args, **kwargs: None)
    config = ({
        "mode": "train",
        "algo": {"name": "standard_block_diffusion"},
        "model": {"length": 32},
        "block_size": 4,
        "parallel": {
            "data_parallel_size": 1,
            "context_parallel_size": 4,
            "block_parallel_size": 4,
            "active_block_mode": "dual_end",
            "kv_backend": "ring",
        },
    })

    runtime = build_parallel_runtime(config)
    idx = active_token_indices(
        seq_len=32,
        block_size=4,
        device=torch.device("cpu"),
        runtime=runtime,
    )

    assert runtime.active_block_mode == "dual_end"
    assert runtime.local_parallel_size == 4
    assert runtime.model_parallel_size == 4
    assert runtime.block_parallel_size == 4
    assert runtime.configured_context_parallel_size == 4
    assert runtime.context_attention_size == 4
    assert runtime.uses_context_parallel_attention
    assert idx is not None and idx.tolist() == list(range(0, 4)) + list(range(28, 32))
    assert loss_scale(runtime) == 4.0


def test_cp_only_all_blocks_has_no_complete_block_owner_indices(monkeypatch) -> None:
    monkeypatch.setattr(parallel_runtime.dist, "is_available", lambda: True)
    monkeypatch.setattr(parallel_runtime.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(parallel_runtime.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(parallel_runtime, "_new_current_group", lambda *args, **kwargs: None)
    config = ({
        "mode": "train",
        "algo": {"name": "standard_block_diffusion"},
        "model": {"length": 32},
        "block_size": 4,
        "parallel": {
            "data_parallel_size": 2,
            "context_parallel_size": 2,
            "block_parallel_size": 1,
            "active_block_mode": "all_blocks",
            "kv_backend": "ring",
        },
    })

    monkeypatch.setattr(parallel_runtime.dist, "get_rank", lambda: 0)
    rank0 = build_parallel_runtime(config)
    monkeypatch.setattr(parallel_runtime.dist, "get_rank", lambda: 1)
    rank1 = build_parallel_runtime(config)

    idx0 = active_token_indices(
        seq_len=32,
        block_size=4,
        device=torch.device("cpu"),
        runtime=rank0,
    )
    idx1 = active_token_indices(
        seq_len=32,
        block_size=4,
        device=torch.device("cpu"),
        runtime=rank1,
    )

    assert idx0 is None
    assert idx1 is None
    assert loss_scale(rank0) == 1.0
    assert loss_scale(rank1) == 1.0


def test_cp_only_dual_end_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(parallel_runtime.dist, "is_available", lambda: True)
    monkeypatch.setattr(parallel_runtime.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(parallel_runtime.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(parallel_runtime.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(parallel_runtime, "_new_current_group", lambda *args, **kwargs: None)
    config = ({
        "mode": "train",
        "algo": {"name": "standard_block_diffusion"},
        "model": {"length": 32},
        "block_size": 4,
        "parallel": {
            "data_parallel_size": 2,
            "context_parallel_size": 2,
            "block_parallel_size": 1,
            "active_block_mode": "dual_end",
            "kv_backend": "ring",
        },
    })

    with pytest.raises(ValueError, match="dual_end.*block_parallel_size"):
        build_parallel_runtime(config)


def test_runtime_rejects_unknown_kv_backend() -> None:
    config = ({
        "mode": "train",
        "algo": {"name": "standard_block_diffusion"},
        "model": {"length": 32},
        "block_size": 4,
        "parallel": {
            "data_parallel_size": 1,
            "context_parallel_size": 2,
            "block_parallel_size": 1,
            "active_block_mode": "all_blocks",
            "kv_backend": "experimental",
        },
    })

    with pytest.raises(ValueError, match="replicated.*ring"):
        build_parallel_runtime(config)


def test_runtime_accepts_tensor_parallel_only(monkeypatch) -> None:
    monkeypatch.setattr(parallel_runtime.dist, "is_available", lambda: True)
    monkeypatch.setattr(parallel_runtime.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(parallel_runtime.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(parallel_runtime.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(parallel_runtime, "_new_current_group", lambda *args, **kwargs: None)
    config = ({
        "mode": "train",
        "algo": {"name": "standard_block_diffusion"},
        "model": {"length": 32},
        "block_size": 4,
        "parallel": {
            "data_parallel_size": 2,
            "context_parallel_size": 1,
            "block_parallel_size": 1,
            "tensor_parallel_size": 2,
            "active_block_mode": "all_blocks",
            "kv_backend": "replicated",
        },
    })

    runtime = build_parallel_runtime(config)

    assert runtime.enabled
    assert runtime.local_parallel_size == 1
    assert runtime.tensor_parallel_size == 2
    assert runtime.tensor_parallel_rank == 1
    assert runtime.model_parallel_size == 2
    assert runtime.tensor_parallel_group_ranks == [0, 1]
    assert runtime.context_block_parallel_group_ranks == [1]
    assert runtime.model_parallel_group_ranks == [0, 1]
    assert runtime.data_parallel_group_ranks == [1, 3]
    assert not runtime.uses_context_parallel_attention
    assert loss_scale(runtime) == 1.0


def test_runtime_records_tensor_parallel_sequence_and_overlap_knobs(monkeypatch) -> None:
    monkeypatch.setattr(parallel_runtime.dist, "is_available", lambda: True)
    monkeypatch.setattr(parallel_runtime.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(parallel_runtime.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(parallel_runtime.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(parallel_runtime, "_new_current_group", lambda *args, **kwargs: None)
    config = ({
        "mode": "train",
        "algo": {"name": "standard_block_diffusion"},
        "model": {"length": 16},
        "block_size": 8,
        "parallel": {
            "data_parallel_size": 1,
            "context_parallel_size": 1,
            "block_parallel_size": 1,
            "tensor_parallel_size": 2,
            "sequence_parallel": True,
            "tensor_parallel_overlap": False,
            "active_block_mode": "all_blocks",
            "kv_backend": "replicated",
        },
    })

    runtime = build_parallel_runtime(config)

    assert runtime.sequence_parallel is True
    assert runtime.tensor_parallel_overlap is False


def test_cp_only_dual_end_does_not_activate_block_loss_scaling() -> None:
    plan = parallel_runtime.build_parallel_plan(
        num_blocks=8,
        world_size=2,
        data_parallel_size=1,
        context_parallel_size=2,
        block_parallel_size=1,
    )
    runtime = ParallelRuntime(
        enabled=True,
        active_block_mode="dual_end",
        rank=0,
        world_size=2,
        data_parallel_rank=0,
        local_parallel_rank=0,
        context_parallel_rank=0,
        block_parallel_rank=0,
        context_block_parallel_group_ranks=[0, 1],
        data_parallel_group_ranks=[0],
        kv_backend="ring",
        plan=plan,
    )

    assert loss_scale(runtime) == 1.0


def test_cp_only_all_blocks_runtime_scale_excludes_model_token_shard_scale() -> None:
    plan = parallel_runtime.build_parallel_plan(
        num_blocks=8,
        world_size=2,
        data_parallel_size=1,
        context_parallel_size=2,
        block_parallel_size=1,
    )
    runtime = ParallelRuntime(
        enabled=True,
        active_block_mode="all_blocks",
        rank=0,
        world_size=2,
        data_parallel_rank=0,
        local_parallel_rank=0,
        context_parallel_rank=0,
        block_parallel_rank=0,
        context_block_parallel_group_ranks=[0, 1],
        data_parallel_group_ranks=[0],
        kv_backend="ring",
        plan=plan,
    )

    assert loss_scale(runtime) == 1.0
