# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from dllm_parallel.core.adapters import GroupedExpertLoRA, LoRALinear, apply_lora
from dllm_parallel.core.models.backbones.diffusiongemma.expert_parallel import (
    _reference_grouped_gated_experts,
)
from dllm_parallel.core.models.backbones.nemotron import _TEPackedLayerProjections
from dllm_parallel.training.run_spec import RunSpec


def _adapter_spec(**overrides: object) -> SimpleNamespace:
    values = {
        "type": "lora",
        "rank": 3,
        "alpha": 6.0,
        "dropout": 0.0,
        "targets": ("attention", "mlp"),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_lora_linear_is_exact_at_initialization_and_trains_only_adapters() -> None:
    torch.manual_seed(3)
    base = torch.nn.Linear(7, 11, bias=True)
    hidden = torch.randn(2, 5, 7)
    expected = base(hidden)
    module = LoRALinear(base, rank=3, alpha=6.0, dropout=0.0)

    torch.testing.assert_close(module(hidden), expected, atol=0.0, rtol=0.0)
    module(hidden).square().mean().backward()
    assert module.lora_b.grad is not None


def test_apply_lora_freezes_packed_base_and_none_is_a_noop() -> None:
    layer = _TEPackedLayerProjections(
        qkv=torch.nn.Linear(8, 12, bias=False),
        qkv_local_sizes=(4, 4, 4),
        o_proj=torch.nn.Linear(4, 8, bias=False),
        gate_up=torch.nn.Linear(8, 16, bias=False),
        gate_up_local_sizes=(8, 8),
        gated_activation=None,
        down_proj=torch.nn.Linear(8, 8, bias=False),
        activation=torch.nn.functional.silu,
    )
    model = torch.nn.Module()
    model._te_packed_layers = torch.nn.ModuleList([layer])
    original_ids = tuple(id(parameter) for parameter in model.parameters())
    assert apply_lora(
        model,
        SimpleNamespace(type="none"),
        SimpleNamespace(tensor_parallel_size=1),
    ) is None
    assert tuple(id(parameter) for parameter in model.parameters()) == original_ids
    assert all(parameter.requires_grad for parameter in model.parameters())

    installation = apply_lora(
        model,
        _adapter_spec(),
        SimpleNamespace(tensor_parallel_size=1),
    )
    assert installation is not None
    assert installation.linear_modules == 4
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    frozen = [parameter for parameter in model.parameters() if not parameter.requires_grad]
    assert trainable and frozen
    assert all(getattr(parameter, "_dllm_lora_parameter", False) for parameter in trainable)


def test_lora_enables_adapter_checkpointing_contract() -> None:
    layer = _TEPackedLayerProjections(
        qkv=torch.nn.Linear(8, 12, bias=False),
        qkv_local_sizes=(4, 4, 4),
        o_proj=torch.nn.Linear(4, 8, bias=False),
        gate_up=torch.nn.Linear(8, 16, bias=False),
        gate_up_local_sizes=(8, 8),
        gated_activation=None,
        down_proj=torch.nn.Linear(8, 8, bias=False),
        activation=torch.nn.functional.silu,
    )

    class AdapterCheckpointModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self._te_packed_layers = torch.nn.ModuleList([layer])
            self.adapter_checkpointing = False

        def enable_adapter_checkpointing(self) -> None:
            self.adapter_checkpointing = True

    model = AdapterCheckpointModel()
    apply_lora(model, _adapter_spec(), SimpleNamespace(tensor_parallel_size=1))
    assert model.adapter_checkpointing


def test_nonreentrant_checkpoint_retains_lora_graph_with_frozen_input() -> None:
    module = LoRALinear(
        torch.nn.Linear(8, 8, bias=False),
        rank=2,
        alpha=4.0,
        dropout=0.0,
    )
    for parameter in module.base.parameters():
        parameter.requires_grad_(False)
    hidden = torch.randn(4, 8)
    loss = checkpoint(module, hidden, use_reentrant=False).square().mean()
    loss.backward()
    assert module.lora_b.grad is not None


def test_grouped_expert_lora_matches_explicit_weight_update_and_gradients() -> None:
    torch.manual_seed(9)
    counts = torch.tensor([2, 1], dtype=torch.int64)
    hidden = torch.randn(3, 5, requires_grad=True)
    gate_up = torch.randn(2, 12, 5)
    down = torch.randn(2, 5, 6)
    adapter = GroupedExpertLoRA(
        gate_up_weight=gate_up,
        down_weight=down,
        rank=2,
        alpha=4.0,
        dropout=0.0,
        expert_parallel_sharded=True,
    )
    with torch.no_grad():
        adapter.gate_up_b.normal_()
        adapter.down_b.normal_()
    expected_hidden = hidden.detach().clone().requires_grad_(True)
    effective_gate_up = gate_up + adapter.scale * torch.bmm(
        adapter.gate_up_b,
        adapter.gate_up_a,
    )
    effective_down = down + adapter.scale * torch.bmm(
        adapter.down_b,
        adapter.down_a,
    )
    actual = _reference_grouped_gated_experts(
        hidden,
        gate_up,
        down,
        counts,
        "silu",
        lora_adapter=adapter,
    )
    expected = _reference_grouped_gated_experts(
        expected_hidden,
        effective_gate_up,
        effective_down,
        counts,
        "silu",
    )
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(hidden.grad, expected_hidden.grad, atol=1e-4, rtol=1e-4)
    assert all(
        getattr(parameter, "_dllm_expert_parallel_sharded", False)
        for parameter in adapter.parameters()
    )


def test_lora_spec_validation_is_typed_and_fails_before_model_loading() -> None:
    spec = RunSpec.from_mapping(
        {
            "adapter": {
                "type": "lora",
                "rank": 8,
                "alpha": 16.0,
                "targets": ["attention", "mlp"],
            },
            "topology": {"tensor_parallel_size": 1},
        }
    )
    assert spec.adapter.targets == ("attention", "mlp")
    with pytest.raises(ValueError, match="sequence_parallel=true"):
        RunSpec.from_mapping(
            {
                "adapter": {
                    "type": "lora",
                    "rank": 8,
                    "alpha": 16.0,
                    "targets": ["attention", "mlp"],
                },
                "topology": {"tensor_parallel_size": 2},
            }
        )
