#!/usr/bin/env python3
"""Benchmark the production CP/BP attention dispatch at real model head shapes."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Callable

import torch
import torch._dynamo

import dllm_parallel.core.attention.context_parallel_attention as cp_attention
from dllm_parallel.core.attention.flex import flex_attention_backward_from_state
from dllm_parallel.core.attention.masks import BlockDenoisingPackedKeyMask
from dllm_parallel.core.attention.wide_head_attention import (
    _metadata_backward,
    _metadata_flex_plan,
    _metadata_forward,
)


BLOCK_SIZE = 32
QUERY_LENGTH = 32


@dataclass(frozen=True)
class ModelShape:
    family: str
    attention_kind: str
    query_heads: int
    key_value_heads: int
    head_dim: int
    dispatch: str


SHAPES = (
    ModelShape(
        family="nemotron",
        attention_kind="gqa",
        query_heads=32,
        key_value_heads=8,
        head_dim=128,
        dispatch="fa4_forward_fa4_backward",
    ),
    ModelShape(
        family="diffusiongemma",
        attention_kind="local_gqa",
        query_heads=16,
        key_value_heads=8,
        head_dim=256,
        dispatch="adaptive_fa4_or_compiled_flex_forward_native_packed_gqa_backward",
    ),
    ModelShape(
        family="diffusiongemma",
        attention_kind="global_gqa",
        query_heads=16,
        key_value_heads=2,
        head_dim=512,
        dispatch="compiled_flex_metadata_forward_backward",
    ),
)


def _percentile(samples: list[float], quantile: float) -> float:
    ordered = sorted(samples)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return float(ordered[index])


def _measure_cuda(
    function: Callable[[], Any],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()

    allocated_before = int(torch.cuda.memory_allocated())
    torch.cuda.reset_peak_memory_stats()
    samples: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        stop.record()
        stop.synchronize()
        samples.append(float(start.elapsed_time(stop)))
    peak_bytes = max(0, int(torch.cuda.max_memory_allocated()) - allocated_before)
    return {
        "min_ms": min(samples),
        "p50_ms": statistics.median(samples),
        "p95_ms": _percentile(samples, 0.95),
        "mean_ms": statistics.fmean(samples),
        "incremental_peak_mib": peak_bytes / (1024.0 * 1024.0),
    }


def _make_inputs(
    shape: ModelShape,
    *,
    clean_context_length: int,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    BlockDenoisingPackedKeyMask,
]:
    generator = torch.Generator(device=device).manual_seed(
        1701 + shape.head_dim + clean_context_length
    )
    dtype = torch.bfloat16
    key_length = BLOCK_SIZE + int(clean_context_length)
    query = torch.randn(
        1,
        shape.query_heads,
        QUERY_LENGTH,
        shape.head_dim,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    key = torch.randn(
        1,
        shape.key_value_heads,
        key_length,
        shape.head_dim,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    value = torch.randn(
        key.shape,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    grad_output = torch.randn(
        query.shape,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    target_block = int(clean_context_length) // BLOCK_SIZE
    mask = BlockDenoisingPackedKeyMask(
        query_blocks=torch.full(
            (QUERY_LENGTH,),
            target_block,
            device=device,
            dtype=torch.int32,
        ),
        local_query_blocks=torch.zeros(
            QUERY_LENGTH,
            device=device,
            dtype=torch.int32,
        ),
        query_is_clean=torch.zeros(
            QUERY_LENGTH,
            device=device,
            dtype=torch.bool,
        ),
        active_key_blocks=torch.zeros(
            BLOCK_SIZE,
            device=device,
            dtype=torch.int32,
        ),
        clean_key_blocks=(
            torch.arange(clean_context_length, device=device, dtype=torch.int32)
            // BLOCK_SIZE
        ),
        block_size=BLOCK_SIZE,
    )
    return query, key, value, grad_output, mask


def _production_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: BlockDenoisingPackedKeyMask,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        output, lse, _ = cp_attention._flex_shard_stats(
            query=query,
            key=key,
            value=value,
            attn_mask=mask,
            is_causal=False,
            scale=float(scale),
            key_start=0,
            output_float=False,
            native_fa4=True,
        )
    return output, lse


def _production_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    grad_output: torch.Tensor,
    mask: BlockDenoisingPackedKeyMask,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return cp_attention._flex_merged_shard_backward(
        query=query,
        key=key,
        value=value,
        final_output=output,
        final_lse=lse,
        grad_output=grad_output,
        attn_mask=mask,
        is_causal=False,
        scale=float(scale),
        key_start=0,
    )


def _flex_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: BlockDenoisingPackedKeyMask,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        output, lse, _ = cp_attention._flex_shard_stats(
            query=query,
            key=key,
            value=value,
            attn_mask=mask,
            is_causal=False,
            scale=float(scale),
            key_start=0,
            output_float=False,
            native_fa4=False,
        )
    return output, lse


def _flex_backward_from_merged_state(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    grad_output: torch.Tensor,
    mask: BlockDenoisingPackedKeyMask,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, block_mask = cp_attention._make_flex_mods(
        attn_mask=mask,
        is_causal=False,
        query=query,
        key_start=0,
        key_len=int(key.shape[-2]),
        build_block_mask=True,
    )
    if block_mask is None:
        raise RuntimeError("compiled Flex reference requires a block mask")
    return flex_attention_backward_from_state(
        query=query,
        key=key,
        value=value,
        output=output,
        lse=lse,
        grad_output=grad_output,
        grad_lse=None,
        block_mask=block_mask,
        mask_mod=cp_attention._explicit_bdlm_flex_mask,
        mask_buffers=cp_attention._flex_backward_mask_buffers(
            attn_mask=mask,
            key_start=0,
            key_len=int(key.shape[-2]),
            device=query.device,
        ),
        scale=float(scale),
        kernel_options=cp_attention._flex_kernel_options(attn_mask=mask),
        packed_gqa=False,
        force_torch=True,
    )


def _assert_close(
    actual: tuple[torch.Tensor, ...],
    expected: tuple[torch.Tensor, ...],
    *,
    tolerance: float,
) -> None:
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(
            actual_tensor.float(),
            expected_tensor.float(),
            atol=float(tolerance),
            rtol=float(tolerance),
        )


def _wide_cached_operations(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    grad_output: torch.Tensor,
    mask: BlockDenoisingPackedKeyMask,
    scale: float,
) -> tuple[Callable[[], Any], Callable[[], Any], Callable[[], Any]]:
    buffers = cp_attention._flex_backward_mask_buffers(
        attn_mask=mask,
        key_start=0,
        key_len=int(key.shape[-2]),
        device=query.device,
    )
    query_blocks, local_query_blocks, query_is_clean, key_blocks, key_is_clean = (
        buffers
    )
    query_bshd = query.transpose(1, 2).contiguous()
    key_bshd = key.transpose(1, 2).contiguous()
    value_bshd = value.transpose(1, 2).contiguous()
    plan = _metadata_flex_plan(
        query_blocks,
        local_query_blocks,
        query_is_clean,
        key_blocks,
        key_is_clean,
        query_len=int(query_bshd.shape[1]),
        key_len=int(key_bshd.shape[1]),
    )

    def build_plan() -> Any:
        return _metadata_flex_plan(
            query_blocks,
            local_query_blocks,
            query_is_clean,
            key_blocks,
            key_is_clean,
            query_len=int(query_bshd.shape[1]),
            key_len=int(key_bshd.shape[1]),
        )

    def cached_forward() -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return _metadata_forward(
                query_bshd,
                key_bshd,
                value_bshd,
                query_blocks,
                local_query_blocks,
                query_is_clean,
                key_blocks,
                key_is_clean,
                float(scale),
                plan,
            )

    def cached_backward() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            shard_output, shard_lse = _metadata_forward(
                query_bshd,
                key_bshd,
                value_bshd,
                query_blocks,
                local_query_blocks,
                query_is_clean,
                key_blocks,
                key_is_clean,
                float(scale),
                plan,
            )
            shard_grad_output, shard_grad_lse = cp_attention._merged_shard_grads(
                shard_output=shard_output.transpose(1, 2),
                shard_lse=shard_lse,
                final_output=output,
                final_lse=lse,
                grad_output=grad_output,
            )
        return _metadata_backward(
            query_bshd,
            key_bshd,
            value_bshd,
            shard_grad_output.transpose(1, 2).contiguous(),
            shard_grad_lse,
            query_blocks,
            local_query_blocks,
            query_is_clean,
            key_blocks,
            key_is_clean,
            float(scale),
            plan,
        )

    return build_plan, cached_forward, cached_backward


def _benchmark_case(
    shape: ModelShape,
    *,
    clean_context_length: int,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> dict[str, Any]:
    query, key, value, grad_output, mask = _make_inputs(
        shape,
        clean_context_length=clean_context_length,
        device=device,
    )
    scale = 1.0 / math.sqrt(shape.head_dim)
    torch.cuda.synchronize()
    cold_start = time.perf_counter()
    output, lse = _production_forward(query, key, value, mask, scale)
    _production_backward(
        query,
        key,
        value,
        output,
        lse,
        grad_output,
        mask,
        scale,
    )
    torch.cuda.synchronize()
    cold_compile_seconds = time.perf_counter() - cold_start

    production_forward = lambda: _production_forward(
        query, key, value, mask, scale
    )
    production_backward = lambda: _production_backward(
        query,
        key,
        value,
        output,
        lse,
        grad_output,
        mask,
        scale,
    )
    production_gradients = production_backward()

    result: dict[str, Any] = {
        "shape": asdict(shape),
        "query_length": QUERY_LENGTH,
        "active_key_length": BLOCK_SIZE,
        "clean_context_length": int(clean_context_length),
        "total_key_length": int(key.shape[-2]),
        "dtype": str(query.dtype),
        "cold_compile_seconds": cold_compile_seconds,
        "production_forward": _measure_cuda(
            production_forward,
            warmup=warmup,
            iterations=iterations,
        ),
        "production_exact_backward": _measure_cuda(
            production_backward,
            warmup=warmup,
            iterations=iterations,
        ),
    }

    if shape.head_dim < 512:
        reference_mask = _make_inputs(
            shape,
            clean_context_length=clean_context_length,
            device=device,
        )[-1]
        reference_output, reference_lse = _flex_forward(
            query,
            key,
            value,
            reference_mask,
            scale,
        )
        reference_gradients = _flex_backward_from_merged_state(
            query,
            key,
            value,
            reference_output,
            reference_lse,
            grad_output,
            reference_mask,
            scale,
        )
        _assert_close(
            (output, lse),
            (reference_output, reference_lse),
            tolerance=5e-2,
        )
        _assert_close(
            production_gradients,
            reference_gradients,
            tolerance=5e-2,
        )
        reference_forward = lambda: _flex_forward(
            query,
            key,
            value,
            reference_mask,
            scale,
        )
        reference_backward = lambda: _flex_backward_from_merged_state(
            query,
            key,
            value,
            reference_output,
            reference_lse,
            grad_output,
            reference_mask,
            scale,
        )
        result["compiled_flex_forward"] = _measure_cuda(
            reference_forward,
            warmup=warmup,
            iterations=iterations,
        )
        result["compiled_flex_exact_backward"] = _measure_cuda(
            reference_backward,
            warmup=warmup,
            iterations=iterations,
        )
        result["production_over_flex_forward"] = (
            result["production_forward"]["p50_ms"]
            / result["compiled_flex_forward"]["p50_ms"]
        )
        result["production_over_flex_backward"] = (
            result["production_exact_backward"]["p50_ms"]
            / result["compiled_flex_exact_backward"]["p50_ms"]
        )
    else:
        build_plan, cached_forward, cached_backward = _wide_cached_operations(
            query=query,
            key=key,
            value=value,
            output=output,
            lse=lse,
            grad_output=grad_output,
            mask=mask,
            scale=scale,
        )
        cached_output, cached_lse = cached_forward()
        cached_gradients = cached_backward()
        _assert_close(
            (output.transpose(1, 2), lse),
            (cached_output, cached_lse),
            tolerance=1.2e-1,
        )
        _assert_close(
            tuple(gradient.transpose(1, 2) for gradient in production_gradients),
            cached_gradients,
            tolerance=1.2e-1,
        )
        result["metadata_plan_build"] = _measure_cuda(
            build_plan,
            warmup=warmup,
            iterations=iterations,
        )
        result["cached_plan_forward"] = _measure_cuda(
            cached_forward,
            warmup=warmup,
            iterations=iterations,
        )
        result["cached_plan_exact_backward"] = _measure_cuda(
            cached_backward,
            warmup=warmup,
            iterations=iterations,
        )
        result["production_over_cached_forward"] = (
            result["production_forward"]["p50_ms"]
            / result["cached_plan_forward"]["p50_ms"]
        )
        result["production_over_cached_backward"] = (
            result["production_exact_backward"]["p50_ms"]
            / result["cached_plan_exact_backward"]["p50_ms"]
        )

    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def _parse_lengths(value: str) -> list[int]:
    lengths = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not lengths or any(length <= 0 for length in lengths):
        raise argparse.ArgumentTypeError("context lengths must be positive")
    if any(length % BLOCK_SIZE for length in lengths):
        raise argparse.ArgumentTypeError(
            f"context lengths must be divisible by block size {BLOCK_SIZE}"
        )
    if any(length > 4096 for length in lengths):
        raise argparse.ArgumentTypeError("context lengths are bounded at 4096")
    return lengths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--context-lengths",
        type=_parse_lengths,
        default=_parse_lengths("256,1024,4096"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if hasattr(torch._dynamo.config, "recompile_limit"):
        torch._dynamo.config.recompile_limit = max(
            int(torch._dynamo.config.recompile_limit),
            len(SHAPES) * len(args.context_lengths) + 8,
        )
    if args.warmup < 0 or not 1 <= args.iterations <= 100:
        raise ValueError("warmup must be nonnegative and iterations must be 1..100")
    if not torch.cuda.is_available():
        raise RuntimeError("cross-family attention benchmark requires CUDA")
    device = torch.device("cuda", 0)
    capability = torch.cuda.get_device_capability(device)
    if capability != (9, 0):
        raise RuntimeError(f"benchmark requires H100 SM90, got SM{capability[0]}{capability[1]}")

    report = {
        "schema_version": 1,
        "device": torch.cuda.get_device_name(device),
        "compute_capability": list(capability),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "warmup": int(args.warmup),
        "iterations": int(args.iterations),
        "cases": [
            _benchmark_case(
                shape,
                clean_context_length=context_length,
                warmup=int(args.warmup),
                iterations=int(args.iterations),
                device=device,
            )
            for shape in SHAPES
            for context_length in args.context_lengths
        ],
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "benchmark_complete", **report}, sort_keys=True))


if __name__ == "__main__":
    main()
