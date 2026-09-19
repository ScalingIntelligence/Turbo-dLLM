# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dllm_parallel.core.profiling.perf import (
    DFLASH_WORK_COUNT_NAMES,
    DFlashTransformerFlops,
)
from dllm_parallel.training import block_diffusion_trainer


def _rank_metrics(*, counts: list[int], steps: int = 2) -> list[dict[str, object]]:
    base = {
        "measured_steps": steps,
        "measured_objective_count_names": list(DFLASH_WORK_COUNT_NAMES),
        "measured_active_tokens": 0,
        "measured_valid_tokens": 0,
    }
    return [
        {**base, "measured_objective_counts": counts},
        {**base, "measured_objective_counts": []},
    ]


def test_dflash_mfu_uses_exact_useful_work_and_is_topology_independent(
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
    model_spec = SimpleNamespace(
        family="dflash",
        num_layers=2,
        hidden_size=8,
        intermediate_size=24,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        target_hidden_size=12,
        target_feature_width=36,
        draft_vocab_size=32,
        attention_layer_types=("full_attention", "sliding_attention"),
        sliding_window_non_causal=False,
    )
    # Totals over two measured steps. Per step: 32 context rows, three
    # anchors, 200 full-context pairs, 120 sliding-context pairs, ten
    # supervised rows.
    counts = [64, 6, 400, 240, 20]
    metrics = block_diffusion_trainer._megatron_mfu_metrics(
        rank_metrics=_rank_metrics(counts=counts),
        model_spec=model_spec,
        objective_name="dflash_distillation",
        seq_len=32,
        block_size=4,
        vocab_size=32,
        world_size=2,
        elapsed_ms=1000.0,
        dflash_loss_kind="speculators_kl",
    )

    attention_pairs = (200 + 3 * 16) + (120 + 3 * 10)
    expected = DFlashTransformerFlops(
        num_layers=2,
        hidden_size=8,
        intermediate_size=24,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        target_hidden_size=12,
        target_feature_width=36,
        draft_vocab_size=32,
    ).flops_per_step(
        context_token_rows=32,
        draft_token_rows=12,
        attention_pairs=attention_pairs,
        supervised_token_rows=10,
        loss_kind="speculators_kl",
    )
    assert metrics["model_flops_per_step"] == pytest.approx(expected)
    assert metrics["mfu_attention_pairs_per_step"] == attention_pairs
    assert metrics["mfu_pct"] == pytest.approx(100 * expected / (2.0e15))

    faster = block_diffusion_trainer._megatron_mfu_metrics(
        rank_metrics=_rank_metrics(counts=counts),
        model_spec=model_spec,
        objective_name="dflash_distillation",
        seq_len=32,
        block_size=4,
        vocab_size=32,
        world_size=2,
        elapsed_ms=500.0,
        dflash_loss_kind="speculators_kl",
    )
    assert faster["model_flops_per_step"] == metrics["model_flops_per_step"]
    assert faster["mfu_pct"] == pytest.approx(2 * metrics["mfu_pct"])
