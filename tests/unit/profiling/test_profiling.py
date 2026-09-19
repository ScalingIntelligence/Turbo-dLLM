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

"""Unit tests for the pure profiling logic (no GPU / torch required)."""

from types import SimpleNamespace

import pytest

from dllm_parallel.core.profiling.perf import (
    DFlashTransformerFlops,
    DiffusionGemmaTransformerFlops,
    FlopsModel,
    MegatronTransformerFlops,
    PerfConfig,
    PerfProfiler,
    block_diffusion_active_packed_factors,
    block_diffusion_context_parallel_factors,
    block_diffusion_sharded_clean_factors,
    block_diffusion_sparse_attention_pairs,
    compute_breakdown,
    diffusiongemma_block_attention_pairs,
    is_comm_kernel,
    model_flops_utilization_pct,
    throughput_metrics,
)


# -- FLOPs model -------------------------------------------------------------


def test_flops_dense_six_n_rule_without_attention_or_duplication():
    # Zero layers/hidden -> attention term vanishes; only the 6N dense term.
    m = FlopsModel(num_params=1_000, num_layers=0, hidden_size=0, seq_len=128)
    assert m.flops_per_data_token() == pytest.approx(6_000.0)
    assert m.flops_for_tokens(10) == pytest.approx(60_000.0)


def test_flops_seq_factor_doubles_dense_and_quadruples_attention():
    base = FlopsModel(2_000, num_layers=4, hidden_size=64, seq_len=256,
                      seq_factor=1.0)
    dup = FlopsModel(2_000, num_layers=4, hidden_size=64, seq_len=256,
                     seq_factor=2.0)
    dense = 6.0 * 2_000
    attn = 12.0 * 4 * 64 * 256
    assert base.flops_per_data_token() == pytest.approx(dense + attn)
    # seq_factor scales dense by f and attention by f**2.
    assert dup.flops_per_data_token() == pytest.approx(2 * dense + 4 * attn)


def test_flops_model_accepts_separate_dense_and_attention_factors():
    m = FlopsModel(
        num_params=2_000,
        num_layers=4,
        hidden_size=64,
        seq_len=256,
        dense_token_factor=1.5,
        attention_pair_factor=0.75,
    )
    dense = 6.0 * 2_000 * 1.5
    attn = 12.0 * 4 * 64 * 256 * 0.75
    assert m.flops_per_data_token() == pytest.approx(dense + attn)


def test_megatron_transformer_flops_counts_gqa_swiglu_attention_and_vocab():
    model = MegatronTransformerFlops(
        num_layers=2,
        hidden_size=8,
        intermediate_size=24,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        vocab_size=32,
    )
    token_rows = 10
    pairs = 25
    vocab_rows = 3
    q_width = 8
    kv_width = 4
    projections = 6 * token_rows * (8 * (q_width + 2 * kv_width) + q_width * 8)
    attention = 12 * pairs * q_width
    swiglu = 6 * token_rows * 3 * 8 * 24
    vocab = 6 * vocab_rows * 8 * 32
    assert model.flops_per_step(
        transformer_token_rows=token_rows,
        attention_pairs=pairs,
        vocabulary_token_rows=vocab_rows,
    ) == pytest.approx(2 * (projections + attention + swiglu) + vocab)


def test_diffusiongemma_flops_count_mixed_attention_dense_and_topk_moe():
    model = DiffusionGemmaTransformerFlops(
        layer_types=("sliding_attention", "full_attention"),
        hidden_size=8,
        intermediate_size=12,
        expert_intermediate_size=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_global_key_value_heads=1,
        head_dim=4,
        global_head_dim=8,
        num_experts=4,
        top_k_experts=2,
        vocab_size=16,
    )
    breakdown = model.flops_breakdown_per_step(
        transformer_token_rows=10,
        sliding_attention_pairs=11,
        full_attention_pairs=13,
        self_conditioning_token_rows=5,
        vocabulary_token_rows=3,
    )
    sliding_projection = 6 * 10 * (8 * (8 + 2 * 4) + 8 * 8)
    # Full attention reuses K as V, so it has Q/K/O but no V projection.
    full_projection = 6 * 10 * (8 * (16 + 8) + 16 * 8)
    assert breakdown == pytest.approx(
        {
            "attention_projection": sliding_projection + full_projection,
            "attention_core": 12 * (11 * 8 + 13 * 16),
            "dense_mlp": 2 * 6 * 10 * 3 * 8 * 12,
            "router": 2 * 6 * 10 * 8 * 4,
            "routed_experts": 2 * 6 * 10 * 2 * 3 * 8 * 4,
            "self_conditioning": 6 * 5 * 3 * 8 * 12,
            "vocabulary": 6 * 3 * 8 * 16,
        }
    )
    assert model.flops_per_step(
        transformer_token_rows=10,
        sliding_attention_pairs=11,
        full_attention_pairs=13,
        self_conditioning_token_rows=5,
        vocabulary_token_rows=3,
    ) == pytest.approx(sum(breakdown.values()))


def test_megatron_mfu_uses_global_model_flops_and_aggregate_peak():
    assert model_flops_utilization_pct(
        model_flops_per_step=4.0e12,
        elapsed_ms=1000.0,
        world_size=4,
        peak_flops_per_gpu=2.0e12,
    ) == pytest.approx(50.0)


def test_dflash_transformer_flops_counts_trainable_and_frozen_products():
    model = DFlashTransformerFlops(
        num_layers=2,
        hidden_size=8,
        intermediate_size=24,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        target_hidden_size=12,
        target_feature_width=36,
        draft_vocab_size=32,
    )
    context_rows = 10
    draft_rows = 6
    pairs = 25
    supervised_rows = 4
    q_width = 8
    kv_width = 4
    expected = {
        "target_projection": 4 * context_rows * 36 * 8,
        "attention_projection": 6
        * 2
        * (
            draft_rows * 8 * q_width
            + (draft_rows + context_rows) * 8 * (2 * kv_width)
            + draft_rows * q_width * 8
        ),
        "attention_core": 12 * pairs * q_width,
        "mlp": 6 * 2 * draft_rows * 3 * 8 * 24,
        "output": 4 * supervised_rows * 8 * 32
        + 2 * supervised_rows * 12 * 32,
    }
    observed = model.flops_breakdown_per_step(
        context_token_rows=context_rows,
        draft_token_rows=draft_rows,
        attention_pairs=pairs,
        supervised_token_rows=supervised_rows,
        loss_kind="speculators_kl",
    )
    assert observed == pytest.approx(expected)
    assert model.flops_per_step(
        context_token_rows=context_rows,
        draft_token_rows=draft_rows,
        attention_pairs=pairs,
        supervised_token_rows=supervised_rows,
        loss_kind="speculators_kl",
    ) == pytest.approx(sum(expected.values()))


def test_block_diffusion_full_sparse_attention_pair_count():
    # T=8, block=2 -> 4 blocks. Full BD3-LM mask has
    # block_size**2 * num_blocks * (num_blocks + 1) allowed pairs.
    assert block_diffusion_sparse_attention_pairs(seq_len=8, block_size=2) == 80


def test_diffusiongemma_attention_pairs_match_exact_clean_causal_bounds():
    # Four two-token blocks: 16 noisy-local pairs, 24 noisy-to-clean pairs,
    # and 36 token-causal clean-to-clean pairs.
    assert diffusiongemma_block_attention_pairs(
        seq_len=8,
        block_size=2,
        sliding_window=None,
    ) == 76
    # A four-token window reduces noisy-to-clean to 16 and clean-to-clean to 26.
    assert diffusiongemma_block_attention_pairs(
        seq_len=8,
        block_size=2,
        sliding_window=4,
    ) == 58
    assert diffusiongemma_block_attention_pairs(
        seq_len=8,
        block_size=2,
        sliding_window=16,
    ) == 76


def test_block_diffusion_active_packed_factors_for_sparse_flex_attention():
    dense_factor, pair_factor = block_diffusion_active_packed_factors(
        seq_len=8,
        block_size=2,
        active_blocks=[0, 3],
        sparse_attention=True,
    )
    # Active xt tokens: 2 blocks -> 4 tokens. Prefix x0: 3 blocks -> 6 tokens.
    assert dense_factor == pytest.approx(10 / 8)
    # xt pairs: (1 + 4) * 4 = 20. x0 prefix pairs: (1 + 2 + 3) * 4 = 24.
    assert pair_factor == pytest.approx(44 / 64)


def test_block_diffusion_active_packed_factors_for_dense_sdpa_attention():
    dense_factor, pair_factor = block_diffusion_active_packed_factors(
        seq_len=8,
        block_size=2,
        active_blocks=[0, 3],
        sparse_attention=False,
    )
    assert dense_factor == pytest.approx(10 / 8)
    assert pair_factor == pytest.approx((10 / 8) ** 2)


def _fake_module(*, attn_backend, local_parallel_size=1,
                 active_block_mode="disabled", block_parallel_rank=0,
                 context_parallel_rank=0, kv_backend="replicated",
                 block_parallel_size=None):
    cfg = SimpleNamespace(
        algo={"name": "standard_block_diffusion"},
        model=SimpleNamespace(length=8, attn_backend=attn_backend),
        block_size=2,
    )
    runtime = SimpleNamespace(
        enabled=local_parallel_size > 1,
        active_block_mode=active_block_mode,
        local_parallel_size=local_parallel_size,
        block_parallel_size=(
            local_parallel_size if block_parallel_size is None
            else block_parallel_size
        ),
        block_parallel_rank=block_parallel_rank,
        context_parallel_rank=context_parallel_rank,
        context_attention_size=2,
        configured_context_parallel_size=2,
        kv_backend=kv_backend,
    )
    return SimpleNamespace(config=cfg, parallel_runtime=runtime)


def test_perf_profiler_uses_sparse_pairs_for_block_diffusion():
    profiler = PerfProfiler(PerfConfig())
    dense_factor, pair_factor = profiler._flop_factors(
        _fake_module(attn_backend="flex")
    )
    assert dense_factor == pytest.approx(2.0)
    assert pair_factor == pytest.approx(80 / 64)


def test_perf_profiler_uses_dense_pairs_for_block_diffusion():
    profiler = PerfProfiler(PerfConfig())
    dense_factor, pair_factor = profiler._flop_factors(
        _fake_module(attn_backend="sdpa", local_parallel_size=2)
    )
    assert dense_factor == pytest.approx(2.0)
    assert pair_factor == pytest.approx(4.0)


def test_perf_profiler_uses_packed_sparse_pairs_for_block_diffusion():
    profiler = PerfProfiler(PerfConfig())
    dense_factor, pair_factor = profiler._flop_factors(
        _fake_module(
            attn_backend="flex",
            local_parallel_size=2,
            active_block_mode="dual_end",
            block_parallel_rank=0,
        )
    )
    assert dense_factor == pytest.approx(10 / 8)
    assert pair_factor == pytest.approx(44 / 64)


def test_block_diffusion_sharded_clean_factors_for_ring_cp_bp():
    dense_factor, pair_factor = block_diffusion_sharded_clean_factors(
        seq_len=8,
        block_size=2,
        active_blocks=[0, 3],
        context_parallel_size=2,
        context_parallel_rank=0,
    )
    # Active xt: 4 tokens. Clean shard rank 0: tokens [0,4), so 4 tokens.
    assert dense_factor == pytest.approx(1.0)
    # Local xt pairs: 2 active blocks * 4. Active->clean: (0 + 3) * 4.
    # Local clean shard x0 pairs: blocks 0 and 1 -> (1 + 2) * 4.
    assert pair_factor == pytest.approx(32 / 64)


def test_perf_profiler_uses_sharded_clean_factors_for_ring_cp_bp():
    profiler = PerfProfiler(PerfConfig())
    dense_factor, pair_factor = profiler._flop_factors(
        _fake_module(
            attn_backend="sdpa",
            local_parallel_size=2,
            active_block_mode="dual_end",
            block_parallel_rank=0,
            context_parallel_rank=0,
            kv_backend="ring",
        )
    )
    assert dense_factor == pytest.approx(1.0)
    assert pair_factor == pytest.approx(32 / 64)


def test_perf_profiler_uses_sharded_clean_factors_for_pure_ring_cp():
    profiler = PerfProfiler(PerfConfig())
    dense_factor, pair_factor = profiler._flop_factors(
        _fake_module(
            attn_backend="sdpa",
            local_parallel_size=2,
            active_block_mode="all_blocks",
            context_parallel_rank=0,
            kv_backend="ring",
            block_parallel_size=1,
        )
    )
    # Pure CP owns four noisy and four clean token rows, not target blocks.
    assert dense_factor == pytest.approx(1.0)
    assert pair_factor == pytest.approx(40 / 64)


def test_context_parallel_factors_cover_full_graph_across_ranks():
    factors = [
        block_diffusion_context_parallel_factors(
            seq_len=8,
            block_size=2,
            context_parallel_size=2,
            context_parallel_rank=rank,
        )
        for rank in range(2)
    ]
    assert sum(dense for dense, _ in factors) == pytest.approx(2.0)
    assert sum(pairs for _, pairs in factors) == pytest.approx(80 / 64)


# -- comm classification -----------------------------------------------------


@pytest.mark.parametrize("name", [
    "ncclDevKernel_AllReduce_Sum",
    "c10d::allreduce_",
    "AllGather",
    "reduce_scatter_tensor",
    "ncclSend",
])
def test_is_comm_kernel_matches_collectives(name):
    assert is_comm_kernel(name)


@pytest.mark.parametrize("name", [
    "triton_flex_attention_5",
    "ampere_bf16_gemm",
    "elementwise_kernel",
    "",
])
def test_is_comm_kernel_rejects_compute(name):
    assert not is_comm_kernel(name)


# -- breakdown arithmetic ----------------------------------------------------


def test_breakdown_none_on_empty():
    assert compute_breakdown([]) is None


def test_breakdown_pure_compute_no_comm_no_idle():
    # Two back-to-back compute kernels filling [0, 20]; no gaps, no comm.
    intervals = [
        (0.0, 10.0, "gemm", False),
        (10.0, 20.0, "gemm", False),
    ]
    b = compute_breakdown(intervals)
    assert b.window_ms == pytest.approx(20.0)
    assert b.compute_ms == pytest.approx(20.0)
    assert b.comm_ms == 0.0
    assert b.exposed_comm_ms == 0.0
    assert b.idle_ms == pytest.approx(0.0)
    assert b.compute_frac == pytest.approx(1.0)


def test_breakdown_idle_gap_is_counted():
    # Compute [0,5] then [15,20]: a 10ms bubble in the middle.
    intervals = [
        (0.0, 5.0, "gemm", False),
        (15.0, 20.0, "gemm", False),
    ]
    b = compute_breakdown(intervals)
    assert b.window_ms == pytest.approx(20.0)
    assert b.busy_ms == pytest.approx(10.0)
    assert b.idle_ms == pytest.approx(10.0)
    assert b.idle_frac == pytest.approx(0.5)


def test_breakdown_fully_overlapped_comm_is_not_exposed():
    # Comm [0,10] runs entirely under compute [0,10] (separate streams).
    intervals = [
        (0.0, 10.0, "gemm", False),
        (0.0, 10.0, "ncclAllReduce", True),
    ]
    b = compute_breakdown(intervals)
    assert b.comm_ms == pytest.approx(10.0)
    assert b.compute_ms == pytest.approx(10.0)
    assert b.exposed_comm_ms == pytest.approx(0.0)  # hidden behind compute
    assert b.idle_ms == pytest.approx(0.0)
    assert b.window_ms == pytest.approx(10.0)


def test_breakdown_partially_exposed_comm():
    # Compute [0,10]; comm [8,18]. Overlap 2ms -> 8ms exposed on critical path.
    intervals = [
        (0.0, 10.0, "gemm", False),
        (8.0, 18.0, "ncclAllReduce", True),
    ]
    b = compute_breakdown(intervals)
    assert b.window_ms == pytest.approx(18.0)
    assert b.comm_ms == pytest.approx(10.0)
    assert b.exposed_comm_ms == pytest.approx(8.0)
    assert b.busy_ms == pytest.approx(18.0)  # union [0,18]
    assert b.idle_ms == pytest.approx(0.0)


def test_breakdown_overlapping_compute_kernels_union_not_double_counted():
    # Two overlapping compute kernels [0,10] and [5,15] -> union 15ms.
    intervals = [
        (0.0, 10.0, "gemmA", False),
        (5.0, 15.0, "gemmB", False),
    ]
    b = compute_breakdown(intervals)
    assert b.compute_ms == pytest.approx(15.0)
    assert b.busy_ms == pytest.approx(15.0)
    assert b.idle_ms == pytest.approx(0.0)


def test_breakdown_measured_tflops_passthrough():
    b = compute_breakdown([(0.0, 1.0, "gemm", False)], measured_tflops=123.4)
    assert b.measured_tflops == pytest.approx(123.4)
    assert "measured_tflops" in b.as_dict()


# -- throughput / MFU window -------------------------------------------------


def test_throughput_metrics_basic_rates_and_global_scale():
    # 1000 tokens over 2 steps in 100 ms on 4 GPUs.
    m = throughput_metrics(tokens=1000, steps=2, elapsed_ms=100.0, world_size=4)
    assert m["perf/step_time_ms"] == pytest.approx(50.0)
    assert m["perf/tokens_per_s_per_gpu"] == pytest.approx(10_000.0)
    assert m["perf/tokens_per_s_global"] == pytest.approx(40_000.0)
    assert m["perf/unique_tokens_per_s_global"] == pytest.approx(40_000.0)
    assert "perf/mfu" not in m  # no FlopsModel -> no MFU


def test_throughput_metrics_reports_unique_global_data_rate_for_bp():
    # In DP=2, BP=2 on 4 GPUs, device-global throughput uses all 4 GPUs but
    # unique training-token progress uses only the 2 data-parallel replicas.
    m = throughput_metrics(
        tokens=1000,
        steps=2,
        elapsed_ms=100.0,
        world_size=4,
        data_parallel_size=2,
    )
    assert m["perf/tokens_per_s_per_gpu"] == pytest.approx(10_000.0)
    assert m["perf/tokens_per_s_global"] == pytest.approx(40_000.0)
    assert m["perf/unique_tokens_per_s_global"] == pytest.approx(20_000.0)


def test_perf_profiler_infers_data_parallel_size_for_enabled_bp_runtime():
    runtime = SimpleNamespace(
        enabled=True,
        data_parallel_group_ranks=[0, 2],
        local_parallel_size=2,
    )
    module = SimpleNamespace(parallel_runtime=runtime)
    assert PerfProfiler._infer_data_parallel_size(module, world_size=4) == 2


def test_perf_profiler_uses_world_size_as_data_parallel_size_without_bp():
    module = SimpleNamespace(parallel_runtime=None)
    assert PerfProfiler._infer_data_parallel_size(module, world_size=4) == 4


def test_throughput_metrics_mfu_against_peak():
    flops = FlopsModel(num_params=1_000, num_layers=0, hidden_size=0, seq_len=1)
    # 6000 FLOPs/token * 1000 tokens / 1s = 6e6 FLOP/s; peak 1.2e7 -> 50% MFU.
    m = throughput_metrics(tokens=1000, steps=1, elapsed_ms=1000.0,
                           flops=flops, peak_flops=1.2e7)
    assert m["perf/mfu"] == pytest.approx(0.5)
    assert m["perf/mfu_pct"] == pytest.approx(50.0)


@pytest.mark.parametrize("tokens,steps,elapsed_ms", [
    (1000, 0, 100.0),    # no optimizer steps in window
    (1000, 5, 0.0),      # zero-elapsed window
    (1000, 5, 0.5),      # below min_window_ms -> degenerate
])
def test_throughput_metrics_degenerate_window_returns_none(tokens, steps, elapsed_ms):
    assert throughput_metrics(
        tokens=tokens, steps=steps, elapsed_ms=elapsed_ms
    ) is None
