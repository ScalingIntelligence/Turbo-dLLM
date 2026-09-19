# Copyright 2026 The dllm_parallel Authors.
# SPDX-License-Identifier: Apache-2.0

"""Immutable capability contract for packaged Split-D training kernels."""

from __future__ import annotations


ARCHITECTURE = "sm_90a"
HEAD_DIM = 512
TILE_M = 64
TILE_N = 64
TRAINING_DTYPES = ("bfloat16", "float16")
SUPPORTED_MASKS = (
  "bdlm",
  "bounded_interval",
  "dense",
  "interval",
  "sparse_interval",
)
SUPPORTED_Q_HEADS_PER_KV_HEAD = (1, 2, 4, 8)
SUPPORTED_GRAD_LSE = (False, True)

# Forward, dQ, and dK/dV each specialize by dtype, mask, and local GQA
# ratio. Backward preprocessing specializes only by dtype and dLSE presence.
EXPECTED_ARTIFACT_COUNT = len(TRAINING_DTYPES) * (
  3 * len(SUPPORTED_MASKS) * len(SUPPORTED_Q_HEADS_PER_KV_HEAD)
  + len(SUPPORTED_GRAD_LSE)
)


def manifest_capabilities() -> dict[str, object]:
  return {
    "architecture": ARCHITECTURE,
    "head_dim": HEAD_DIM,
    "training_dtypes": list(TRAINING_DTYPES),
    "masks": list(SUPPORTED_MASKS),
    "q_heads_per_kv_head": list(SUPPORTED_Q_HEADS_PER_KV_HEAD),
    "grad_lse": list(SUPPORTED_GRAD_LSE),
  }


def forward_variant_keys() -> tuple[str, ...]:
  return tuple(
    f"{dtype}:{mask}:gqa{ratio}"
    for dtype in TRAINING_DTYPES
    for mask in SUPPORTED_MASKS
    for ratio in SUPPORTED_Q_HEADS_PER_KV_HEAD
  )


def backward_variant_keys() -> tuple[str, ...]:
  return tuple(
    f"{dtype}:{mask}:gqa{ratio}:dlse{int(grad_lse)}"
    for dtype in TRAINING_DTYPES
    for mask in SUPPORTED_MASKS
    for ratio in SUPPORTED_Q_HEADS_PER_KV_HEAD
    for grad_lse in SUPPORTED_GRAD_LSE
  )


__all__ = [
  "ARCHITECTURE",
  "EXPECTED_ARTIFACT_COUNT",
  "HEAD_DIM",
  "SUPPORTED_GRAD_LSE",
  "SUPPORTED_MASKS",
  "SUPPORTED_Q_HEADS_PER_KV_HEAD",
  "TILE_M",
  "TILE_N",
  "TRAINING_DTYPES",
  "backward_variant_keys",
  "forward_variant_keys",
  "manifest_capabilities",
]
