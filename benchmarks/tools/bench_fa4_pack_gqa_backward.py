#!/usr/bin/env python3
# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Bounded FA4 Pack-GQA backward verifier for fused BDLM attention.

The default shape represents one fused Nemotron CP4/BP4 rank at 32K context:

    Q:   [batch=2, heads=32, q_len=16384, head_dim=128]
    K/V: [batch=2, heads=8,  k_len=40960, head_dim=128]

K/V rows are laid out exactly as the production fused path:
``[rank-local noisy rows; global clean rows]``. Query rows are
``[rank-local noisy rows; rank-local dual-chunk clean rows]``. Block ownership
comes from the production dual-end scheduler, and the benchmark builds the
production ``BlockDenoisingPackedKeyMask`` through the current mask APIs.

The branch matrix compares native Pack-GQA gradients against generic FA4 for
every production dual-end rank at two reduced sequence lengths. Timings use
CUDA events and include every
GPU kernel launched by ``fa4_backward_from_state`` (preprocess, attention
backward, and postprocess), while excluding mask construction and JIT
compilation.

Run from the repository root:

    python benchmarks/tools/bench_fa4_pack_gqa_backward.py
    python benchmarks/tools/bench_fa4_pack_gqa_backward.py \
        --q-length 8192 --k-length 20480 --active-length 4096
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import statistics
from typing import Any

import torch


BATCH_SIZE = 2
QUERY_HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128
DTYPE = torch.bfloat16

MAX_SEQUENCE_LENGTH = 131_072
MAX_WARMUP_ITERATIONS = 20
MAX_MEASURED_ITERATIONS = 100


@dataclass(frozen=True)
class Shape:
    q_length: int
    k_length: int
    active_length: int
    parallel_size: int
    rank: int
    block_size: int

    @property
    def clean_length(self) -> int:
        return self.k_length - self.active_length

    @property
    def clean_query_length(self) -> int:
        return self.q_length - self.active_length

    def validate(self) -> None:
        lengths = {
            "q_length": self.q_length,
            "k_length": self.k_length,
            "active_length": self.active_length,
            "clean_length": self.clean_length,
            "clean_query_length": self.clean_query_length,
        }
        for name, value in lengths.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
            if value > MAX_SEQUENCE_LENGTH:
                raise ValueError(
                    f"{name}={value} exceeds the bounded harness limit "
                    f"{MAX_SEQUENCE_LENGTH}"
                )
            if value % self.block_size:
                raise ValueError(
                    f"{name}={value} must be divisible by block_size="
                    f"{self.block_size}"
                )
        if self.parallel_size <= 1:
            raise ValueError("parallel_size must be greater than one")
        if not 0 <= self.rank < self.parallel_size:
            raise ValueError("rank must be in [0, parallel_size)")
        if self.clean_length % self.parallel_size:
            raise ValueError("clean_length must divide evenly across parallel ranks")
        if self.clean_query_length != self.clean_length // self.parallel_size:
            raise ValueError(
                "clean query rows must equal one sharded clean-sequence share: "
                "q_length - active_length must equal "
                "(k_length - active_length) / parallel_size"
            )

        num_blocks = self.clean_length // self.block_size
        if num_blocks % self.parallel_size:
            raise ValueError("clean blocks must divide evenly across parallel ranks")
        expected_active = self.clean_length // self.parallel_size
        if self.active_length != expected_active:
            raise ValueError(
                "the representative fused layout requires one dual-end target "
                "share per rank: active_length must equal "
                "clean_length / parallel_size"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch": BATCH_SIZE,
            "query_heads": QUERY_HEADS,
            "kv_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "dtype": str(DTYPE).removeprefix("torch."),
            "q_bhsd": [BATCH_SIZE, QUERY_HEADS, self.q_length, HEAD_DIM],
            "k_bhsd": [BATCH_SIZE, KV_HEADS, self.k_length, HEAD_DIM],
            "v_bhsd": [BATCH_SIZE, KV_HEADS, self.k_length, HEAD_DIM],
            "q_kernel_bshd": [BATCH_SIZE, self.q_length, QUERY_HEADS, HEAD_DIM],
            "kv_kernel_bshd": [BATCH_SIZE, self.k_length, KV_HEADS, HEAD_DIM],
            "active_length": self.active_length,
            "clean_length": self.clean_length,
            "clean_query_length": self.clean_query_length,
            "parallel_size": self.parallel_size,
            "rank": self.rank,
            "block_size": self.block_size,
        }


@dataclass
class AttentionState:
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    output: torch.Tensor
    lse: torch.Tensor
    grad_output: torch.Tensor
    block_mask: tuple[Any, ...]
    packed_gqa_block_mask: tuple[Any, ...]
    mask_buffers: tuple[torch.Tensor, ...]
    scale: float

    def backward_args(self, *, pack_gqa: bool) -> dict[str, Any]:
        return {
            "query": self.query,
            "key": self.key,
            "value": self.value,
            "output": self.output,
            "lse": self.lse,
            "grad_output": self.grad_output,
            "grad_lse": None,
            "block_mask": (
                self.packed_gqa_block_mask if pack_gqa else self.block_mask
            ),
            "mask_buffers": self.mask_buffers,
            "scale": self.scale,
            "pack_gqa": pack_gqa,
        }


def _token_rows(blocks: list[int], block_size: int, device: torch.device) -> torch.Tensor:
    block_tensor = torch.tensor(blocks, device=device, dtype=torch.long)
    offsets = torch.arange(block_size, device=device, dtype=torch.long)
    return (block_tensor[:, None] * block_size + offsets[None]).reshape(-1)


def _build_packed_mask(shape: Shape, device: torch.device) -> Any:
    from dllm_parallel.core.attention.cp_backend import clean_shards_for_rank
    from dllm_parallel.core.attention.masks import BlockDenoisingPackedKeyMask
    from dllm_parallel.core.schedules.block import build_block_schedule

    shape.validate()
    num_blocks = shape.clean_length // shape.block_size
    schedule = build_block_schedule(
        num_blocks=num_blocks,
        block_parallel_size=shape.parallel_size,
        context_parallel_size=shape.parallel_size,
    )
    active_blocks = schedule.active_blocks_by_worker[shape.rank]
    active_positions = _token_rows(active_blocks, shape.block_size, device)

    clean_shards = clean_shards_for_rank(
        seq_len=shape.clean_length,
        context_parallel_size=shape.parallel_size,
        rank=shape.rank,
        layout="dual_chunk",
    )
    clean_positions = torch.cat(
        tuple(
            torch.arange(shard.start, shard.stop, device=device, dtype=torch.long)
            for shard in clean_shards
        )
    )
    if active_positions.numel() != shape.active_length:
        raise RuntimeError("dual-end scheduler produced an unexpected active length")
    if clean_positions.numel() != shape.clean_query_length:
        raise RuntimeError("dual-chunk scheduler produced an unexpected clean length")

    owner_major_clean_positions = torch.cat(
        tuple(
            torch.cat(
                tuple(
                    torch.arange(
                        shard.start,
                        shard.stop,
                        device=device,
                        dtype=torch.long,
                    )
                    for shard in clean_shards_for_rank(
                        seq_len=shape.clean_length,
                        context_parallel_size=shape.parallel_size,
                        rank=owner,
                        layout="dual_chunk",
                    )
                )
            )
            for owner in range(shape.parallel_size)
        )
    )
    if owner_major_clean_positions.numel() != shape.clean_length:
        raise RuntimeError("clean ownership must cover the logical sequence exactly once")

    query_positions = torch.cat((active_positions, clean_positions))
    query_blocks = (query_positions // shape.block_size).to(torch.int32)
    query_is_clean = torch.cat(
        (
            torch.zeros(shape.active_length, device=device, dtype=torch.bool),
            torch.ones(shape.clean_query_length, device=device, dtype=torch.bool),
        )
    )
    return BlockDenoisingPackedKeyMask(
        query_blocks=query_blocks,
        local_query_blocks=query_blocks,
        query_is_clean=query_is_clean,
        active_key_blocks=(
            active_positions // shape.block_size
        ).to(torch.int32),
        clean_key_blocks=(owner_major_clean_positions // shape.block_size).to(
            torch.int32
        ),
        block_size=shape.block_size,
    )


def _make_state(shape: Shape, device: torch.device, seed: int) -> AttentionState:
    from dllm_parallel.core.attention.context_parallel_attention import (
        _flex_backward_mask_buffers,
        _make_flex_mods,
        _make_packed_gqa_backward_block_mask,
    )
    from dllm_parallel.core.attention.fa4 import fa4_forward

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    query = torch.randn(
        BATCH_SIZE,
        QUERY_HEADS,
        shape.q_length,
        HEAD_DIM,
        device=device,
        dtype=DTYPE,
        generator=generator,
    )
    key = torch.randn(
        BATCH_SIZE,
        KV_HEADS,
        shape.k_length,
        HEAD_DIM,
        device=device,
        dtype=DTYPE,
        generator=generator,
    )
    value = torch.randn(
        BATCH_SIZE,
        KV_HEADS,
        shape.k_length,
        HEAD_DIM,
        device=device,
        dtype=DTYPE,
        generator=generator,
    )
    grad_output = torch.randn(
        query.shape,
        device=device,
        dtype=DTYPE,
        generator=generator,
    )

    packed_mask = _build_packed_mask(shape, device)
    _, forward_block_mask = _make_flex_mods(
        attn_mask=packed_mask,
        is_causal=False,
        query=query,
        key_start=0,
        key_len=shape.k_length,
        build_block_mask=True,
    )
    if forward_block_mask is None:
        raise RuntimeError("production mask API did not construct a block mask")
    packed_gqa_block_mask = _make_packed_gqa_backward_block_mask(
        attn_mask=packed_mask,
        query=query,
        key=key,
        key_start=0,
        forward_block_mask=forward_block_mask,
    )
    mask_buffers = _flex_backward_mask_buffers(
        attn_mask=packed_mask,
        key_start=0,
        key_len=shape.k_length,
        device=device,
    )
    scale = HEAD_DIM**-0.5
    output, lse = fa4_forward(
        query=query,
        key=key,
        value=value,
        block_mask=forward_block_mask.as_tuple(),
        mask_buffers=mask_buffers,
        scale=scale,
    )
    return AttentionState(
        query=query,
        key=key,
        value=value,
        output=output,
        lse=lse,
        grad_output=grad_output,
        block_mask=forward_block_mask.as_tuple(),
        packed_gqa_block_mask=packed_gqa_block_mask.as_tuple(),
        mask_buffers=mask_buffers,
        scale=scale,
    )
def _error_metrics(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, float | bool]:
    difference = actual.float() - expected.float()
    expected_float = expected.float()
    return {
        "bitwise_equal": bool(torch.equal(actual, expected)),
        "max_abs": difference.abs().max().item(),
        "mean_abs": difference.abs().mean().item(),
        "relative_l2": (
            difference.norm() / expected_float.norm().clamp_min(1.0e-12)
        ).item(),
    }


def _assert_gradients_close(
    *,
    actual: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    actual_name: str,
    expected_name: str,
    atol: float,
    rtol: float,
) -> dict[str, dict[str, float | bool]]:
    names = ("dq", "dk", "dv")
    metrics = {
        name: _error_metrics(observed, reference)
        for name, observed, reference in zip(names, actual, expected, strict=True)
    }
    for name, observed, reference in zip(names, actual, expected, strict=True):
        torch.testing.assert_close(
            observed,
            reference,
            atol=atol,
            rtol=rtol,
            msg=lambda message, gradient=name: (
                f"{actual_name} {gradient} does not match {expected_name}: "
                f"{message}"
            ),
        )
    return metrics


def _mask_branch_summary(
    state: AttentionState,
) -> dict[str, dict[str, int]]:
    def summarize(block_mask: tuple[Any, ...]) -> dict[str, int]:
        partial_q_counts = block_mask[6]
        full_q_counts = block_mask[8]
        return {
            "partial_q_blocks": int(partial_q_counts.sum().item()),
            "full_q_blocks": int(full_q_counts.sum().item()),
        }

    return {
        "forward_coordinates": summarize(state.block_mask),
        "packed_gqa_coordinates": summarize(state.packed_gqa_block_mask),
    }


def _check_gradients(
    shape: Shape,
    device: torch.device,
    seed: int,
    atol: float,
    rtol: float,
    generic_reference: bool,
) -> dict[str, Any]:
    from dllm_parallel.core.attention.fa4 import fa4_backward_from_state

    state = _make_state(shape, device, seed)
    packed = fa4_backward_from_state(**state.backward_args(pack_gqa=True))
    generic = (
        fa4_backward_from_state(**state.backward_args(pack_gqa=False))
        if generic_reference
        else None
    )
    torch.cuda.synchronize(device)

    comparisons: dict[str, Any] = {}
    if generic is not None:
        comparisons["pack_gqa_vs_generic"] = _assert_gradients_close(
            actual=packed,
            expected=generic,
            actual_name="Pack-GQA",
            expected_name="generic FA4",
            atol=atol,
            rtol=rtol,
        )
    return {
        "status": "passed",
        "atol": atol,
        "rtol": rtol,
        "shape": shape.to_dict(),
        "mask_branches": _mask_branch_summary(state),
        "comparisons": comparisons,
    }


def _time_variants(
    device: torch.device,
    warmup: int,
    iterations: int,
    *,
    variants: list[tuple[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    from dllm_parallel.core.attention.fa4 import fa4_backward_from_state

    # First invocations may compile CuTe DSL kernels. They are intentionally
    # outside both warmup and measured samples.
    for _, arguments in variants:
        gradients = fa4_backward_from_state(**arguments)
        del gradients
    torch.cuda.synchronize(device)

    for iteration in range(warmup):
        ordered = variants if iteration % 2 == 0 else list(reversed(variants))
        for _, arguments in ordered:
            gradients = fa4_backward_from_state(**arguments)
            del gradients
    torch.cuda.synchronize(device)

    events = {
        path: [
            (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for _ in range(iterations)
        ]
        for path, _ in variants
    }
    for iteration in range(iterations):
        ordered = variants if iteration % 2 == 0 else list(reversed(variants))
        for path, arguments in ordered:
            start, stop = events[path][iteration]
            start.record()
            gradients = fa4_backward_from_state(**arguments)
            stop.record()
            del gradients
    torch.cuda.synchronize(device)

    results: list[dict[str, Any]] = []
    for path, _ in variants:
        samples = [start.elapsed_time(stop) for start, stop in events[path]]
        ordered_samples = sorted(samples)
        p90_index = math.ceil(0.9 * len(ordered_samples)) - 1
        results.append(
            {
                "path": path,
                "warmup_iterations": warmup,
                "measured_iterations": iterations,
                "milliseconds": {
                    "mean": statistics.fmean(samples),
                    "median": statistics.median(samples),
                    "minimum": min(samples),
                    "maximum": max(samples),
                    "p90": ordered_samples[p90_index],
                },
                "samples": samples,
            }
        )
    return results


def _benchmark_variants(
    shape: Shape,
    device: torch.device,
    seed: int,
    warmup: int,
    iterations: int,
    time_generic: bool,
) -> dict[str, Any]:
    state = _make_state(shape, device, seed)
    variants = [
        ("pack_gqa", state.backward_args(pack_gqa=True)),
    ]
    if time_generic:
        variants.append(("generic", state.backward_args(pack_gqa=False)))
    timings = _time_variants(
        device,
        warmup,
        iterations,
        variants=variants,
    )
    result = {
        "shape": shape.to_dict(),
        "mask_branches": _mask_branch_summary(state),
        "variants": timings,
    }
    if time_generic:
        medians = {
            timing["path"]: timing["milliseconds"]["median"]
            for timing in timings
        }
        result["pack_gqa_speedup"] = medians["generic"] / medians["pack_gqa"]
    return result


def _branch_matrix_shapes(
    *,
    parallel_size: int,
    block_size: int,
) -> list[Shape]:
    shapes: list[Shape] = []
    for local_block_count in (4, 6):
        active_length = local_block_count * block_size
        clean_length = active_length * parallel_size
        for rank in range(parallel_size):
            shapes.append(
                Shape(
                    q_length=2 * active_length,
                    k_length=clean_length + active_length,
                    active_length=active_length,
                    parallel_size=parallel_size,
                    rank=rank,
                    block_size=block_size,
                )
            )
    return shapes


def _correctness_cases(args: argparse.Namespace) -> list[Shape]:
    if args.branch_matrix:
        return _branch_matrix_shapes(
            parallel_size=args.parallel_size,
            block_size=args.block_size,
        )
    return [
        Shape(
            q_length=args.correctness_q_length,
            k_length=args.correctness_k_length,
            active_length=args.correctness_active_length,
            parallel_size=args.parallel_size,
            rank=args.rank,
            block_size=args.block_size,
        )
    ]


def _bounded_count(name: str, value: int, maximum: int) -> int:
    if not 0 <= value <= maximum:
        raise argparse.ArgumentTypeError(
            f"{name} must be in [0, {maximum}], got {value}"
        )
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify and benchmark FA4 Pack-GQA backward for the fused "
            "Nemotron BDLM shape."
        ),
    )
    parser.add_argument("--q-length", type=int, default=16_384)
    parser.add_argument("--k-length", type=int, default=40_960)
    parser.add_argument("--active-length", type=int, default=8_192)
    parser.add_argument("--parallel-size", type=int, default=4)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument(
        "--warmup",
        type=lambda value: _bounded_count(
            "warmup", int(value), MAX_WARMUP_ITERATIONS
        ),
        default=3,
    )
    parser.add_argument(
        "--iterations",
        type=lambda value: _bounded_count(
            "iterations", int(value), MAX_MEASURED_ITERATIONS
        ),
        default=10,
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--time-generic",
        action="store_true",
        help="also time the generic backward at the large benchmark shape",
    )
    parser.add_argument(
        "--generic-reference",
        action="store_true",
        help="compare reduced-case gradients against generic FA4 backward",
    )
    parser.add_argument(
        "--branch-matrix",
        action="store_true",
        help=(
            "run reduced correctness cases at two lengths for every dual-end "
            "rank"
        ),
    )
    parser.add_argument(
        "--correctness-only",
        action="store_true",
        help="run the reduced gradient gate without the large timing case",
    )
    parser.add_argument(
        "--timing-only",
        action="store_true",
        help="time the production shape without constructing reduced test masks",
    )
    parser.add_argument("--correctness-q-length", type=int, default=1024)
    parser.add_argument("--correctness-k-length", type=int, default=2560)
    parser.add_argument("--correctness-active-length", type=int, default=512)
    parser.add_argument("--atol", type=float, default=3.0e-2)
    parser.add_argument("--rtol", type=float, default=3.0e-2)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.correctness_only and args.timing_only:
        raise ValueError("correctness-only and timing-only are mutually exclusive")
    if args.iterations == 0 and not args.correctness_only:
        raise ValueError("iterations must be positive when timing is enabled")
    if not torch.cuda.is_available():
        raise RuntimeError("this FA4 benchmark requires a CUDA GPU")
    device = torch.device("cuda", torch.cuda.current_device())
    capability = torch.cuda.get_device_capability(device)
    if capability[0] != 9:
        raise RuntimeError(
            "FA4 Pack-GQA backward requires an SM90 GPU; "
            f"found compute capability {capability[0]}.{capability[1]}"
        )

    correctness_shapes = [] if args.timing_only else _correctness_cases(args)
    for shape in correctness_shapes:
        shape.validate()
    report: dict[str, Any] = {
        "device": {
            "name": torch.cuda.get_device_name(device),
            "compute_capability": list(capability),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "correctness": [
            _check_gradients(
                shape,
                device,
                args.seed + case_index,
                args.atol,
                args.rtol,
                args.generic_reference,
            )
            for case_index, shape in enumerate(correctness_shapes)
        ],
    }

    if not args.correctness_only:
        benchmark_shape = Shape(
            q_length=args.q_length,
            k_length=args.k_length,
            active_length=args.active_length,
            parallel_size=args.parallel_size,
            rank=args.rank,
            block_size=args.block_size,
        )
        benchmark_shape.validate()
        report["benchmark"] = _benchmark_variants(
            benchmark_shape,
            device,
            args.seed + len(correctness_shapes),
            args.warmup,
            args.iterations,
            args.time_generic,
        )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
