from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from dllm_parallel.core.optim.zero import (
    FirstPartyDistributedAdamW,
    _configure_deepspeed_zero2_unclipped_step,
    bounded_optimizer_named_param_groups,
    build_optimizer,
    deepspeed_zero2_config,
    resolve_optimizer_backend,
    resolve_zero_optimizer_impl,
)


def test_optimizer_backend_resolution_fails_fast_for_distributed_only_backends() -> (
    None
):
    assert resolve_optimizer_backend("auto", distributed=False) == "torch_adamw"
    assert resolve_optimizer_backend("auto", distributed=True) == "deepspeed_zero2"
    assert (
        resolve_optimizer_backend("deepspeed_zero2", distributed=True)
        == "deepspeed_zero2"
    )
    for backend in ("deepspeed_zero2", "fsdp", "fsdp2"):
        with pytest.raises(ValueError, match="multi-rank distributed launch"):
            resolve_optimizer_backend(backend, distributed=False)


def test_optimizer_groups_isolate_expert_parameters_without_size_bounds() -> None:
    dense = torch.nn.Parameter(torch.ones(2))
    expert = torch.nn.Parameter(torch.ones(3))
    expert._dllm_expert_parallel_sharded = True

    groups = bounded_optimizer_named_param_groups(
        [("dense", dense), ("expert", expert)],
        max_elements=0,
        isolate_vocab_embedding=False,
        isolate_expert_parameters=True,
    )

    assert [group["params"] for group in groups] == [[dense], [expert]]
    assert "_dllm_expert_parallel_sharded" not in groups[0]
    assert groups[1]["_dllm_expert_parallel_sharded"] is True


def test_optimizer_groups_isolate_sequence_parallel_replicated_parameters() -> None:
    dense = torch.nn.Parameter(torch.ones(2))
    sequence_parallel = torch.nn.Parameter(torch.ones(3))
    sequence_parallel._dllm_sequence_parallel_replicated = True
    tensor_shard = torch.nn.Parameter(torch.ones(4))
    tensor_shard._dllm_sequence_parallel_replicated = True
    tensor_shard._dllm_tensor_parallel_sharded = True

    groups = bounded_optimizer_named_param_groups(
        [
            ("dense", dense),
            ("norm", sequence_parallel),
            ("column", tensor_shard),
        ],
        max_elements=0,
        isolate_vocab_embedding=False,
        isolate_sequence_parallel_parameters=True,
    )

    assert [group["params"] for group in groups] == [
        [dense, tensor_shard],
        [sequence_parallel],
    ]
    assert "_dllm_sequence_parallel_replicated" not in groups[0]
    assert groups[1]["_dllm_sequence_parallel_replicated"] is True


def test_optimizer_groups_do_not_isolate_replicated_vocab_embedding() -> None:
    dense = torch.nn.Parameter(torch.ones(2))
    embedding = torch.nn.Parameter(torch.ones(3))
    embedding._dllm_vocab_embedding_shard = True

    groups = bounded_optimizer_named_param_groups(
        [("dense", dense), ("embedding", embedding)],
        max_elements=0,
        isolate_vocab_embedding=True,
    )

    assert [group["params"] for group in groups] == [[dense, embedding]]
    assert "_dllm_use_torch_fused_adamw" not in groups[0]


def test_optimizer_groups_isolate_tensor_parallel_vocab_embedding() -> None:
    dense = torch.nn.Parameter(torch.ones(2))
    embedding = torch.nn.Parameter(torch.ones(3))
    embedding._dllm_vocab_embedding_shard = True
    embedding._dllm_tensor_parallel_sharded = True

    groups = bounded_optimizer_named_param_groups(
        [("dense", dense), ("embedding", embedding)],
        max_elements=0,
        isolate_vocab_embedding=True,
    )

    assert [group["params"] for group in groups] == [[dense], [embedding]]
    assert groups[1]["_dllm_use_torch_fused_adamw"] is True


def test_zero_optimizer_auto_uses_hybrid_fused_adam_for_unsafe_tp_dual_end() -> None:
    runtime = SimpleNamespace(tensor_parallel_size=2, active_block_mode="dual_end")

    assert (
        resolve_zero_optimizer_impl(requested="auto", runtime=runtime)
        == "deepspeed_fused_adam_hybrid"
    )
    with pytest.raises(RuntimeError, match="first-party safe optimizer"):
        resolve_zero_optimizer_impl(requested="deepspeed_fused_adam", runtime=runtime)


def test_build_optimizer_supports_first_party_distributed_adamw() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    optimizer = build_optimizer(
        [{"params": [parameter]}],
        impl="torch_distributed_adamw",
        lr=1e-3,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
    )

    assert isinstance(optimizer, FirstPartyDistributedAdamW)
    assert optimizer.process_group is None


def test_first_party_distributed_adamw_can_step_after_explicit_sync() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = FirstPartyDistributedAdamW([parameter], lr=0.1)
    synchronize_calls = 0

    def synchronize() -> None:
        nonlocal synchronize_calls
        synchronize_calls += 1

    optimizer.synchronize_gradients = synchronize  # type: ignore[method-assign]
    parameter.grad = torch.ones_like(parameter)
    optimizer.step(synchronize_gradients=False)

    assert synchronize_calls == 0
    assert parameter.item() < 1.0


def test_first_party_distributed_adamw_averages_distributed_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    parameter.grad = torch.tensor([2.0, 4.0])
    optimizer = FirstPartyDistributedAdamW([parameter], lr=0.1)

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda _group: 2)
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce_coalesced",
        lambda tensors, **_kwargs: [tensor.mul_(2.0) for tensor in tensors],
    )

    optimizer.synchronize_gradients()

    torch.testing.assert_close(parameter.grad, torch.tensor([2.0, 4.0]))


def test_deepspeed_zero2_config_includes_gradient_clipping_when_enabled() -> None:
    config = deepspeed_zero2_config(
        train_micro_batch_size_per_gpu=2,
        lr=1e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        dtype="bf16",
        reduce_bucket_size=1024,
        contiguous_gradients=True,
        gradient_clip_norm=0.75,
    )

    assert config["gradient_clipping"] == 0.75


def test_unclipped_deepspeed_zero_policy_skips_unused_global_norm() -> None:
    optimizer = SimpleNamespace(
        clip_grad=0.0,
        device=torch.device("cpu"),
        scaled_global_norm=lambda: torch.ones((), dtype=torch.float32),
    )

    _configure_deepspeed_zero2_unclipped_step(
        optimizer,
        gradient_clip_norm=0.0,
    )

    assert optimizer._dllm_global_norm_skipped_when_unclipped is True
    assert optimizer.scaled_global_norm().item() == 0.0


def test_clipped_deepspeed_zero_policy_keeps_global_norm() -> None:
    optimizer = SimpleNamespace(
        clip_grad=1.0,
        device=torch.device("cpu"),
        scaled_global_norm=lambda: torch.ones((), dtype=torch.float32),
    )

    _configure_deepspeed_zero2_unclipped_step(
        optimizer,
        gradient_clip_norm=1.0,
    )

    assert not hasattr(optimizer, "_dllm_global_norm_skipped_when_unclipped")
    assert optimizer.scaled_global_norm().item() == 1.0
