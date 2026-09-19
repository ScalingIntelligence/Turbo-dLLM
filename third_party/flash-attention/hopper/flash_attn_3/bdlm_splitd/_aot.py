# Copyright 2026 The dllm_parallel Authors.
# SPDX-License-Identifier: Apache-2.0

"""Build the complete packaged SM90a D=512 training artifact set."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from . import (
  bdlm_splitd_backward,
  bdlm_splitd_forward,
  splitd_interval_backward,
  splitd_interval_forward,
  splitd_full_backward,
  splitd_full_forward,
)
from ._capabilities import (
  HEAD_DIM,
  SUPPORTED_Q_HEADS_PER_KV_HEAD,
  TRAINING_DTYPES,
)
from ._backward import _bwd_preprocess, _splitd_backward_sm90
from ._artifacts import (
  aot_build,
  finalize_aot_artifacts,
  reset_aot_artifacts,
)
from ._forward import _splitd_forward_sm90
from ._runtime import SM90_BWD_TILE_M, SM90_BWD_TILE_N, SM90_FWD_TILE_M


_KV_HEADS = 1
_SEQUENCE_LENGTH = max(SM90_FWD_TILE_M, SM90_BWD_TILE_M)
_TORCH_DTYPES = {
  "bfloat16": torch.bfloat16,
  "float16": torch.float16,
}


def _forward_variant_key(
  dtype_name: str,
  mask: str,
  q_heads_per_kv_head: int,
) -> str:
  return f"{dtype_name}:{mask}:gqa{q_heads_per_kv_head}"


def _backward_variant_key(
  dtype_name: str,
  mask: str,
  q_heads_per_kv_head: int,
  has_grad_lse: bool,
) -> str:
  return (
    f"{dtype_name}:{mask}:gqa{q_heads_per_kv_head}:"
    f"dlse{int(has_grad_lse)}"
  )


def _record_forward(
  variants: dict[str, dict[str, object]],
  dtype_name: str,
  mask: str,
  q_heads_per_kv_head: int,
) -> None:
  variants["forward"][
    _forward_variant_key(dtype_name, mask, q_heads_per_kv_head)
  ] = _splitd_forward_sm90.compile_cache.last_relative_path()


def _record_backward(
  variants: dict[str, dict[str, object]],
  dtype_name: str,
  mask: str,
  q_heads_per_kv_head: int,
  has_grad_lse: bool,
) -> None:
  variants["backward"][
    _backward_variant_key(
      dtype_name,
      mask,
      q_heads_per_kv_head,
      has_grad_lse,
    )
  ] = {
    "preprocess": _bwd_preprocess.compile_cache.last_relative_path(),
    "dkdv": _splitd_backward_sm90.compile_cache_dkdv.last_relative_path(),
    "dq": _splitd_backward_sm90.compile_cache_dq.last_relative_path(),
  }


def _compile_variant(
  dtype_name: str,
  dtype: torch.dtype,
  q_heads_per_kv_head: int,
  variants: dict[str, dict[str, object]],
) -> None:
  query = torch.empty(
    (1, _SEQUENCE_LENGTH, q_heads_per_kv_head, HEAD_DIM),
    device="cuda",
    dtype=dtype,
  )
  key = torch.empty(
    (1, _SEQUENCE_LENGTH, _KV_HEADS, HEAD_DIM),
    device="cuda",
    dtype=dtype,
  )
  value = torch.empty_like(key)
  grad_output = torch.empty_like(query)
  scale = HEAD_DIM**-0.5

  output, lse = splitd_full_forward(query, key, value, scale)
  _record_forward(variants, dtype_name, "dense", q_heads_per_kv_head)
  for grad_lse in (None, torch.empty_like(lse)):
    splitd_full_backward(
      query,
      key,
      value,
      output,
      lse,
      grad_output,
      grad_lse,
      scale,
    )
    _record_backward(
      variants,
      dtype_name,
      "dense",
      q_heads_per_kv_head,
      grad_lse is not None,
    )

  query_blocks = torch.empty(
    (_SEQUENCE_LENGTH,), device="cuda", dtype=torch.int32
  )
  query_is_clean = torch.empty(
    (_SEQUENCE_LENGTH,), device="cuda", dtype=torch.bool
  )
  query_clean_bounds = torch.empty(
    (_SEQUENCE_LENGTH, 2), device="cuda", dtype=torch.int32
  )
  local_query_blocks = torch.empty_like(query_blocks)
  key_coordinates = torch.empty_like(query_blocks)
  key_is_clean = torch.empty_like(query_is_clean)

  forward_query_tiles = (
    _SEQUENCE_LENGTH * q_heads_per_kv_head + SM90_FWD_TILE_M - 1
  ) // SM90_FWD_TILE_M
  backward_query_tiles = (
    _SEQUENCE_LENGTH + SM90_BWD_TILE_M - 1
  ) // SM90_BWD_TILE_M
  key_tiles = (
    _SEQUENCE_LENGTH + SM90_BWD_TILE_N - 1
  ) // SM90_BWD_TILE_N
  forward_query_tile_offsets = torch.empty(
    (forward_query_tiles + 1,), device="cuda", dtype=torch.int32
  )
  forward_key_tile_work_items = torch.empty(
    (forward_query_tiles * key_tiles,), device="cuda", dtype=torch.int32
  )
  backward_query_tile_offsets = torch.empty(
    (backward_query_tiles + 1,), device="cuda", dtype=torch.int32
  )
  backward_key_tile_work_items = torch.empty(
    (backward_query_tiles * key_tiles,), device="cuda", dtype=torch.int32
  )
  key_tile_offsets = torch.empty(
    (key_tiles + 1,), device="cuda", dtype=torch.int32
  )
  query_tile_work_items = torch.empty(
    (backward_query_tiles * key_tiles,), device="cuda", dtype=torch.int32
  )
  output, lse = splitd_interval_forward(
    query,
    key,
    value,
    query_clean_bounds,
    local_query_blocks,
    query_is_clean,
    key_coordinates,
    key_is_clean,
    scale,
    query_tile_offsets=forward_query_tile_offsets,
    key_tile_work_items=forward_key_tile_work_items,
  )
  _record_forward(variants, dtype_name, "sparse_interval", q_heads_per_kv_head)
  for grad_lse in (None, torch.empty_like(lse)):
    splitd_interval_backward(
      query,
      key,
      value,
      output,
      lse,
      grad_output,
      grad_lse,
      query_clean_bounds,
      local_query_blocks,
      query_is_clean,
      key_coordinates,
      key_is_clean,
      scale,
      query_tile_offsets=backward_query_tile_offsets,
      key_tile_work_items=backward_key_tile_work_items,
      key_tile_offsets=key_tile_offsets,
      query_tile_work_items=query_tile_work_items,
    )
    _record_backward(
      variants,
      dtype_name,
      "sparse_interval",
      q_heads_per_kv_head,
      grad_lse is not None,
    )

  output, lse = bdlm_splitd_forward(
    query,
    key,
    value,
    query_blocks,
    query_is_clean,
    _SEQUENCE_LENGTH,
    0,
    scale,
  )
  _record_forward(variants, dtype_name, "bdlm", q_heads_per_kv_head)
  for grad_lse in (None, torch.empty_like(lse)):
    bdlm_splitd_backward(
      query,
      key,
      value,
      output,
      lse,
      grad_output,
      grad_lse,
      query_blocks,
      query_is_clean,
      _SEQUENCE_LENGTH,
      0,
      scale,
    )
    _record_backward(
      variants,
      dtype_name,
      "bdlm",
      q_heads_per_kv_head,
      grad_lse is not None,
    )

  output, lse = splitd_interval_forward(
    query,
    key,
    value,
    query_clean_bounds,
    local_query_blocks,
    query_is_clean,
    key_coordinates,
    key_is_clean,
    scale,
  )
  _record_forward(variants, dtype_name, "interval", q_heads_per_kv_head)
  for grad_lse in (None, torch.empty_like(lse)):
    splitd_interval_backward(
      query,
      key,
      value,
      output,
      lse,
      grad_output,
      grad_lse,
      query_clean_bounds,
      local_query_blocks,
      query_is_clean,
      key_coordinates,
      key_is_clean,
      scale,
    )
    _record_backward(
      variants,
      dtype_name,
      "interval",
      q_heads_per_kv_head,
      grad_lse is not None,
    )

  forward_query_tile_bounds = torch.empty(
    (
      (
        _SEQUENCE_LENGTH * q_heads_per_kv_head
        + SM90_BWD_TILE_M
        - 1
      )
      // SM90_BWD_TILE_M,
      2,
    ),
    device="cuda",
    dtype=torch.int32,
  )
  backward_query_tile_bounds = torch.empty(
    ((_SEQUENCE_LENGTH + SM90_BWD_TILE_M - 1) // SM90_BWD_TILE_M, 2),
    device="cuda",
    dtype=torch.int32,
  )
  key_tile_bounds = torch.empty_like(backward_query_tile_bounds)
  output, lse = splitd_interval_forward(
    query,
    key,
    value,
    query_clean_bounds,
    local_query_blocks,
    query_is_clean,
    key_coordinates,
    key_is_clean,
    scale,
    forward_query_tile_bounds,
    key_tile_bounds,
  )
  _record_forward(variants, dtype_name, "bounded_interval", q_heads_per_kv_head)
  for grad_lse in (None, torch.empty_like(lse)):
    splitd_interval_backward(
      query,
      key,
      value,
      output,
      lse,
      grad_output,
      grad_lse,
      query_clean_bounds,
      local_query_blocks,
      query_is_clean,
      key_coordinates,
      key_is_clean,
      scale,
      backward_query_tile_bounds,
      key_tile_bounds,
    )
    _record_backward(
      variants,
      dtype_name,
      "bounded_interval",
      q_heads_per_kv_head,
      grad_lse is not None,
    )


def build(output_root: Path) -> None:
  output_root = Path(output_root).resolve()
  reset_aot_artifacts(output_root)
  variants: dict[str, dict[str, object]] = {
    "forward": {},
    "backward": {},
  }
  with aot_build(output_root), FakeTensorMode():
    for dtype_name in TRAINING_DTYPES:
      dtype = _TORCH_DTYPES[dtype_name]
      for q_heads_per_kv_head in SUPPORTED_Q_HEADS_PER_KV_HEAD:
        _compile_variant(
          dtype_name,
          dtype,
          q_heads_per_kv_head,
          variants,
        )
  finalize_aot_artifacts(output_root, variants)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--output-root", type=Path, required=True)
  args = parser.parse_args()
  build(args.output_root)


if __name__ == "__main__":
  main()
