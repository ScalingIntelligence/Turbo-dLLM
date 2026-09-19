# Copyright 2026 The bdlm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from dllm_parallel.core.attention.layout import context_parallel_sequence_intervals
from dllm_parallel.core.models.backbones.nemotron import (
    NemotronLabsDiffusionPackedBlockDiffusionModel,
    _TEPackedLayerProjections,
    _hf_gated_mlp_forward,
    _te_gated_mlp_forward,
)
from dllm_parallel.core.models.backbones.nemotron.model import (
    _build_te_rmsnorm_column_linear,
    build_packed_block_diffusion_model,
)
from dllm_parallel.core.models.backbones.nemotron.context_parallel import (
    NemotronContextParallelModel,
)


def test_bounded_sequence_activation_clears_shape_dependent_caches() -> None:
    model = NemotronLabsDiffusionPackedBlockDiffusionModel.__new__(
        NemotronLabsDiffusionPackedBlockDiffusionModel
    )
    torch.nn.Module.__init__(model)
    model.max_seq_len = 16
    model.seq_len = 16
    model.block_size = 4
    model._packed_layout_cache = {("old",): object()}
    model._layer_attention_mask_cache = {("old",): object()}
    model._clean_expert_token_plan_cache = {("old",): object()}
    context_parallel_sequence_intervals.cache_clear()
    context_parallel_sequence_intervals(
        seq_len=16,
        context_parallel_size=2,
        rank=0,
    )

    model._activate_sequence_length(16)

    assert model.seq_len == 16
    assert model._packed_layout_cache
    assert model._layer_attention_mask_cache
    assert model._clean_expert_token_plan_cache
    assert context_parallel_sequence_intervals.cache_info().currsize == 1

    model._activate_sequence_length(8)

    assert model.seq_len == 8
    assert not model._packed_layout_cache
    assert not model._layer_attention_mask_cache
    assert not model._clean_expert_token_plan_cache
    assert context_parallel_sequence_intervals.cache_info().currsize == 0
    with pytest.raises(ValueError, match="block_size"):
        model._activate_sequence_length(6)
    with pytest.raises(ValueError, match="configured maximum"):
        model._activate_sequence_length(20)


def test_te_gated_mlp_token_chunking_preserves_outputs_and_gradients() -> None:
    torch.manual_seed(2026)
    hidden_size = 16
    intermediate_size = 40
    layer = _TEPackedLayerProjections(
        qkv=torch.nn.Identity(),
        qkv_local_sizes=(0, 0, 0),
        o_proj=torch.nn.Identity(),
        gate_up=torch.nn.Linear(hidden_size, 2 * intermediate_size, bias=True),
        gate_up_local_sizes=(intermediate_size, intermediate_size),
        gated_activation=None,
        down_proj=torch.nn.Linear(intermediate_size, hidden_size, bias=True),
        activation=torch.nn.functional.silu,
    )
    reference = torch.randn(3, 17, hidden_size, dtype=torch.float32, requires_grad=True)
    chunked = reference.detach().clone().requires_grad_(True)

    reference_out = _te_gated_mlp_forward(layer, reference, token_chunk_size=0)
    chunked_out = _te_gated_mlp_forward(layer, chunked, token_chunk_size=11)
    torch.testing.assert_close(chunked_out, reference_out, atol=1e-5, rtol=1e-5)

    reference_loss = reference_out.square().mean()
    chunked_loss = chunked_out.square().mean()
    reference_loss.backward()
    chunked_loss.backward()
    torch.testing.assert_close(chunked.grad, reference.grad, atol=1e-5, rtol=1e-5)


def test_gemma4_te_rmsnorm_column_linear_packs_norm_and_projection_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeLayerNormLinear(torch.nn.Module):
        def __init__(self, in_features: int, out_features: int, **kwargs: object) -> None:
            super().__init__()
            captured.update(kwargs)
            self.layer_norm_weight = torch.nn.Parameter(torch.empty(in_features))
            self.weight = torch.nn.Parameter(torch.empty(out_features, in_features))

    pytorch = ModuleType("transformer_engine.pytorch")
    pytorch.LayerNormLinear = _FakeLayerNormLinear
    package = ModuleType("transformer_engine")
    package.pytorch = pytorch
    monkeypatch.setitem(sys.modules, "transformer_engine", package)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch", pytorch)

    norm = torch.nn.RMSNorm(8, eps=1e-5)
    query = torch.nn.Linear(8, 5, bias=False)
    key = torch.nn.Linear(8, 3, bias=False)
    packed = _build_te_rmsnorm_column_linear(
        norm,
        ("query", query),
        ("key", key),
        runtime=SimpleNamespace(tensor_parallel_size=1, tensor_parallel_group=None),
        dtype=torch.float32,
        device=torch.device("cpu"),
        name="test_gemma4_qkv",
    )

    torch.testing.assert_close(packed.layer_norm_weight, norm.weight)
    torch.testing.assert_close(
        packed.weight,
        torch.cat((query.weight, key.weight), dim=0),
    )
    assert packed._dllm_local_split_sizes == (5, 3)
    assert captured["normalization"] == "RMSNorm"
    assert captured["sequence_parallel"] is True
    assert captured["parallel_mode"] == "column"


class _HFGatedMLP(torch.nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=True)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=True)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=True)
        self.act_fn = torch.nn.functional.silu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def test_hf_gated_mlp_token_chunking_preserves_outputs_and_gradients() -> None:
    torch.manual_seed(2027)
    mlp = _HFGatedMLP(hidden_size=16, intermediate_size=40)
    reference = torch.randn(3, 17, 16, dtype=torch.float32, requires_grad=True)
    chunked = reference.detach().clone().requires_grad_(True)

    reference_out = _hf_gated_mlp_forward(mlp, reference, token_chunk_size=0)
    chunked_out = _hf_gated_mlp_forward(mlp, chunked, token_chunk_size=11)
    torch.testing.assert_close(chunked_out, reference_out, atol=1e-5, rtol=1e-5)

    reference_loss = reference_out.square().mean()
    chunked_loss = chunked_out.square().mean()
    reference_loss.backward()
    chunked_loss.backward()
    torch.testing.assert_close(chunked.grad, reference.grad, atol=1e-5, rtol=1e-5)


def test_complete_target_block_packing_is_bp_only(monkeypatch) -> None:
    model = NemotronLabsDiffusionPackedBlockDiffusionModel.__new__(
        NemotronLabsDiffusionPackedBlockDiffusionModel
    )
    torch.nn.Module.__init__(model)
    model.seq_len = 8
    model.block_size = 2
    model.runtime = SimpleNamespace(
        enabled=True,
        active_block_mode="dual_end",
        kv_backend="ring",
        context_attention_size=2,
        configured_context_parallel_size=2,
        context_parallel_rank=0,
        block_parallel_size=2,
        block_parallel_rank=0,
    )
    model._packed_layout_cache = {}
    monkeypatch.setattr(
        NemotronLabsDiffusionPackedBlockDiffusionModel,
        "_debug_nonfinite_attention_enabled",
        lambda self: False,
    )
    monkeypatch.setattr(
        NemotronLabsDiffusionPackedBlockDiffusionModel,
        "_local_clean_positions",
        lambda self, device: torch.tensor([0, 1, 6, 7], device=device),
    )
    monkeypatch.setattr(
        NemotronLabsDiffusionPackedBlockDiffusionModel,
        "_packed_masks",
        lambda self, **kwargs: (object(), object()),
    )

    layout = model._packed_layout(torch.device("cpu"))

    assert torch.equal(layout.active_positions, torch.tensor([0, 1, 6, 7]))
    assert torch.equal(layout.clean_positions, torch.tensor([0, 1, 6, 7]))
    assert torch.equal(
        layout.packed_positions,
        torch.tensor([0, 1, 6, 7, 0, 1, 6, 7]),
    )


@pytest.mark.parametrize(
    ("block_parallel_rank", "expected_active", "expected_clean"),
    (
        (0, [0, 1, 6, 7], [0, 1, 2, 3, 4, 5]),
        (1, [2, 3, 4, 5], [0, 1, 2, 3]),
    ),
)
def test_pure_bp_packs_only_owned_targets_with_replicated_prefix(
    monkeypatch,
    block_parallel_rank: int,
    expected_active: list[int],
    expected_clean: list[int],
) -> None:
    model = NemotronLabsDiffusionPackedBlockDiffusionModel.__new__(
        NemotronLabsDiffusionPackedBlockDiffusionModel
    )
    torch.nn.Module.__init__(model)
    model.seq_len = 8
    model.block_size = 2
    model.runtime = SimpleNamespace(
        enabled=True,
        active_block_mode="dual_end",
        kv_backend="replicated",
        context_attention_size=1,
        configured_context_parallel_size=1,
        context_parallel_rank=0,
        block_parallel_size=2,
        block_parallel_rank=block_parallel_rank,
    )
    model._packed_layout_cache = {}
    monkeypatch.setattr(
        NemotronLabsDiffusionPackedBlockDiffusionModel,
        "_debug_nonfinite_attention_enabled",
        lambda self: False,
    )
    monkeypatch.setattr(
        NemotronLabsDiffusionPackedBlockDiffusionModel,
        "_packed_masks",
        lambda self, **kwargs: (object(), object()),
    )

    layout = model._packed_layout(torch.device("cpu"))

    assert layout.active_positions.tolist() == expected_active
    assert layout.clean_positions.tolist() == expected_clean
    assert layout.packed_positions.tolist() == expected_active + expected_clean


def test_replicated_non_bp_execution_keeps_all_targets() -> None:
    model = NemotronLabsDiffusionPackedBlockDiffusionModel.__new__(
        NemotronLabsDiffusionPackedBlockDiffusionModel
    )
    torch.nn.Module.__init__(model)
    model.seq_len = 8
    model.block_size = 2
    model.runtime = SimpleNamespace(
        enabled=True,
        active_block_mode="all_blocks",
        kv_backend="replicated",
        context_attention_size=1,
        configured_context_parallel_size=1,
        context_parallel_rank=0,
        block_parallel_size=1,
        block_parallel_rank=0,
    )
    model._packed_layout_cache = {}

    layout = model._packed_layout(torch.device("cpu"))

    assert layout.active_positions.tolist() == list(range(8))
    assert layout.clean_positions.tolist() == list(range(6))


def test_pure_cp_builds_token_row_context_executor(monkeypatch) -> None:
    sentinel = object()
    runtime = SimpleNamespace(
        kv_backend="ring",
        context_attention_size=4,
        block_parallel_size=1,
    )
    monkeypatch.setattr(
        "dllm_parallel.core.models.backbones.nemotron.context_parallel."
        "build_context_parallel_model",
        lambda *args, **kwargs: sentinel,
    )

    result = build_packed_block_diffusion_model(
        object(),
        runtime=runtime,
        seq_len=32,
        block_size=8,
    )

    assert result is sentinel


def test_pure_cp_loss_normalization_scales_token_rows_not_blocks(monkeypatch) -> None:
    model = NemotronContextParallelModel.__new__(NemotronContextParallelModel)
    torch.nn.Module.__init__(model)
    model.runtime = SimpleNamespace(context_attention_size=4)
    monkeypatch.setattr(
        NemotronLabsDiffusionPackedBlockDiffusionModel,
        "distributed_block_diffusion_loss",
        lambda self, *args, **kwargs: kwargs["bp_loss_scale"],
    )

    scale = model.distributed_block_diffusion_loss(
        torch.empty(0),
        torch.empty(0),
        torch.empty(0),
        torch.empty(0),
    )

    assert scale == 4.0
