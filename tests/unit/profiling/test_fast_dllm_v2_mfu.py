# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dllm_parallel.core.profiling.perf import FastDLLMv2TransformerFlops
from dllm_parallel.training import block_diffusion_trainer


def _qwen38_spec() -> SimpleNamespace:
    return SimpleNamespace(
        family="qwen3_8",
        num_layers=4,
        attention_layer_types=(
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ),
        hidden_size=8,
        intermediate_size=12,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        num_experts=None,
        attention_output_gate=True,
        linear_key_head_dim=2,
        linear_value_head_dim=3,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_conv_kernel_dim=3,
        sliding_window=None,
    )


def _rank_metrics(
    values: list[tuple[int, int, int]],
) -> list[dict[str, int]]:
    return [
        {
            "measured_steps": 2,
            "measured_input_tokens": input_tokens,
            "measured_valid_tokens": valid_tokens,
            "measured_active_tokens": active_tokens,
        }
        for input_tokens, valid_tokens, active_tokens in values
    ]


def test_fast_dllm_v2_flops_match_megatron_gdn_convention() -> None:
    model = FastDLLMv2TransformerFlops(
        layer_types=(
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ),
        hidden_size=8,
        intermediate_size=12,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=20,
        attention_output_gate=True,
        linear_key_head_dim=2,
        linear_value_head_dim=3,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_conv_kernel_dim=3,
    )
    breakdown = model.flops_breakdown_per_step(
        transformer_token_rows=32,
        full_attention_pairs=160,
        sliding_attention_pairs=0,
        vocabulary_token_rows=7,
    )

    query_width = 2 * 4
    kv_width = 1 * 4
    qk_dim = 2 * 2
    value_dim = 3 * 2
    gdn_input_width = 2 * qk_dim + 2 * value_dim + 2 * 2
    expected = {
        "attention_projection": 6
        * 32
        * (8 * (2 * query_width + 2 * kv_width) + query_width * 8),
        "attention_core": 12 * 160 * query_width,
        "gated_delta_net": 3
        * 6
        * 32
        * (
            8 * gdn_input_width
            + 3 * (2 * qk_dim + value_dim)
            + 4 * 2 * 3**2
            + 8 * value_dim
        ),
        "mlp": 4 * 6 * 32 * (3 * 8 * 12),
        "vocabulary": 6 * 7 * 8 * 20,
    }
    assert breakdown == pytest.approx(expected)
    assert model.flops_per_step(
        transformer_token_rows=32,
        full_attention_pairs=160,
        sliding_attention_pairs=0,
        vocabulary_token_rows=7,
    ) == pytest.approx(sum(expected.values()))


def test_fast_dllm_v2_mfu_counts_paired_views_and_is_topology_invariant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        block_diffusion_trainer,
        "detect_hardware_preset",
        lambda **_kwargs: "test_accelerator",
    )
    monkeypatch.setattr(
        block_diffusion_trainer,
        "peak_flops_for_hardware",
        lambda _name: 1.0e15,
    )

    # Totals cover two measured steps. Each step executes two complementary
    # length-eight views, hence 16 input rows, 32 packed transformer rows,
    # and seven shifted vocabulary targets.
    concentrated = _rank_metrics([(32, 14, 14), (0, 0, 0), (0, 0, 0), (0, 0, 0)])
    distributed = _rank_metrics([(16, 7, 7), (16, 7, 7), (0, 0, 0), (0, 0, 0)])

    first = block_diffusion_trainer._megatron_mfu_metrics(
        rank_metrics=concentrated,
        model_spec=_qwen38_spec(),
        objective_name="fast_dllm_v2",
        seq_len=8,
        block_size=2,
        vocab_size=20,
        world_size=4,
        elapsed_ms=1000.0,
    )
    second = block_diffusion_trainer._megatron_mfu_metrics(
        rank_metrics=distributed,
        model_spec=_qwen38_spec(),
        objective_name="fast_dllm_v2",
        seq_len=8,
        block_size=2,
        vocab_size=20,
        world_size=4,
        elapsed_ms=1000.0,
    )

    expected = FastDLLMv2TransformerFlops(
        layer_types=_qwen38_spec().attention_layer_types,
        hidden_size=8,
        intermediate_size=12,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=20,
        attention_output_gate=True,
        linear_key_head_dim=2,
        linear_value_head_dim=3,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_conv_kernel_dim=3,
    ).flops_per_step(
        transformer_token_rows=32,
        full_attention_pairs=160,
        sliding_attention_pairs=0,
        vocabulary_token_rows=7,
    )
    assert first["mfu_unavailable_reason"] is None
    assert first["mfu_method"] == "megatron_fast_dllm_v2_qwen3_8_v1"
    assert first["mfu_paired_views_per_step"] == 2
    assert first["mfu_transformer_token_rows_per_step"] == 32
    assert first["mfu_vocabulary_token_rows_per_step"] == 7
    assert first["mfu_full_attention_pairs_per_layer_per_step"] == 160
    assert first["model_flops_per_step"] == pytest.approx(expected)
    assert first["mfu_pct"] == pytest.approx(100 * expected / 4.0e15)
    assert second["model_flops_per_step"] == pytest.approx(expected)
    assert second["mfu_pct"] == pytest.approx(first["mfu_pct"])


def test_fast_dllm_v2_mfu_fails_closed_for_incomplete_hybrid_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        block_diffusion_trainer,
        "detect_hardware_preset",
        lambda **_kwargs: "test_accelerator",
    )
    monkeypatch.setattr(
        block_diffusion_trainer,
        "peak_flops_for_hardware",
        lambda _name: 1.0e15,
    )
    spec = _qwen38_spec()
    spec.linear_key_head_dim = None
    metrics = block_diffusion_trainer._megatron_mfu_metrics(
        rank_metrics=_rank_metrics([(32, 14, 14)]),
        model_spec=spec,
        objective_name="fast_dllm_v2",
        seq_len=8,
        block_size=2,
        vocab_size=20,
        world_size=1,
        elapsed_ms=1000.0,
    )
    assert metrics["mfu_pct"] is None
    assert "Gated-DeltaNet metadata" in metrics["mfu_unavailable_reason"]


def test_fast_dllm_v2_mfu_uses_loaded_prefix_for_truncated_smoke_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        block_diffusion_trainer,
        "detect_hardware_preset",
        lambda **_kwargs: "test_accelerator",
    )
    monkeypatch.setattr(
        block_diffusion_trainer,
        "peak_flops_for_hardware",
        lambda _name: 1.0e15,
    )
    spec = _qwen38_spec()
    spec.num_layers = 2

    metrics = block_diffusion_trainer._megatron_mfu_metrics(
        rank_metrics=_rank_metrics([(32, 14, 14)]),
        model_spec=spec,
        objective_name="fast_dllm_v2",
        seq_len=8,
        block_size=2,
        vocab_size=20,
        world_size=1,
        elapsed_ms=1000.0,
    )

    assert metrics["mfu_unavailable_reason"] is None
    assert metrics["mfu_linear_attention_layers"] == 2
    assert metrics["mfu_full_attention_layers"] == 0


def test_fast_dllm_v2_mfu_supports_registered_dense_causal_lms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        block_diffusion_trainer,
        "detect_hardware_preset",
        lambda **_kwargs: "test_accelerator",
    )
    monkeypatch.setattr(
        block_diffusion_trainer,
        "peak_flops_for_hardware",
        lambda _name: 1.0e15,
    )
    spec = SimpleNamespace(
        family="causal_lm",
        num_layers=2,
        attention_layer_types=None,
        hidden_size=8,
        intermediate_size=12,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        num_experts=None,
        attention_output_gate=False,
        sliding_window=None,
    )
    metrics = block_diffusion_trainer._megatron_mfu_metrics(
        rank_metrics=_rank_metrics([(32, 14, 14)]),
        model_spec=spec,
        objective_name="fast_dllm_v2",
        seq_len=8,
        block_size=2,
        vocab_size=20,
        world_size=1,
        elapsed_ms=1000.0,
    )

    expected = FastDLLMv2TransformerFlops(
        layer_types=("full_attention", "full_attention"),
        hidden_size=8,
        intermediate_size=12,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=20,
    ).flops_per_step(
        transformer_token_rows=32,
        full_attention_pairs=160,
        sliding_attention_pairs=0,
        vocabulary_token_rows=7,
    )
    assert metrics["mfu_method"] == "megatron_fast_dllm_v2_causal_lm_v1"
    assert metrics["model_flops_per_step"] == pytest.approx(expected)
