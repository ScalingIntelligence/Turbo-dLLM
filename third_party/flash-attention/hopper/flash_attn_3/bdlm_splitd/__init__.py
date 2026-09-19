# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Native D=512 block-diffusion FlashAttention for Hopper."""

from __future__ import annotations

from functools import lru_cache
import importlib
import threading
import weakref

import torch

from . import _aot_runtime
from ._capabilities import (
  HEAD_DIM,
  SUPPORTED_Q_HEADS_PER_KV_HEAD,
  TILE_M,
  TILE_N,
)
from ._artifacts import (
  configure_runtime_jit,
  is_aot_build,
  runtime_jit_enabled,
  verify_packaged_artifacts,
)


_MASK_CACHE_LOCK = threading.Lock()
_MASK_PARAMETER_CACHE: dict[
  int,
  tuple[
    weakref.ReferenceType[torch.Tensor],
    dict[tuple[str, int | None, int, int, int, bool], torch.Tensor],
  ],
] = {}

def _validate_qkv(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
) -> None:
  if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
    raise ValueError("BDLM Split-D expects BSHD query/key/value tensors")
  if key.shape != value.shape:
    raise ValueError("BDLM Split-D key and value shapes must match")
  if query.shape[0] != key.shape[0] or query.shape[-1] != key.shape[-1]:
    raise ValueError("BDLM Split-D query/key batch and head dimensions must match")
  if key.device != query.device or value.device != query.device:
    raise RuntimeError("BDLM Split-D Q/K/V devices must match")
  if query.shape[2] % key.shape[2] != 0:
    raise ValueError("BDLM Split-D query heads must be divisible by KV heads")
  q_heads_per_kv_head = int(query.shape[2] // key.shape[2])
  if q_heads_per_kv_head not in SUPPORTED_Q_HEADS_PER_KV_HEAD:
    raise ValueError(
      "packaged BDLM Split-D artifacts support Q/KV head ratios "
      f"{SUPPORTED_Q_HEADS_PER_KV_HEAD}, got {q_heads_per_kv_head}"
    )
  if query.shape[-1] != HEAD_DIM:
    raise ValueError(f"BDLM Split-D requires head_dim={HEAD_DIM}")
  if query.dtype not in (torch.float16, torch.bfloat16):
    raise TypeError("BDLM Split-D requires fp16 or bf16 Q/K/V")
  if key.dtype != query.dtype or value.dtype != query.dtype:
    raise TypeError("BDLM Split-D Q/K/V dtypes must match")
  if not query.is_contiguous() or not key.is_contiguous() or not value.is_contiguous():
    raise ValueError("BDLM Split-D Q/K/V tensors must be contiguous")
  if not query.is_cuda or not key.is_cuda or not value.is_cuda:
    raise RuntimeError("BDLM Split-D requires CUDA tensors")
  if is_aot_build():
    return
  capability = torch.cuda.get_device_capability(query.device)
  if capability[0] != 9:
    raise RuntimeError(
      "BDLM D=512 Split-D requires Hopper compute capability 9.x; "
      f"got {capability[0]}.{capability[1]}"
    )


def _mask_params(
  owner: torch.Tensor,
  block_size: int,
  key_start: int,
  clean_offset: int,
  full_mask: bool,
) -> torch.Tensor:
  """Cache immutable launch metadata for the lifetime of its query layout."""

  owner_id = id(owner)
  key = (
    owner.device.type,
    owner.device.index,
    int(block_size),
    int(key_start),
    int(clean_offset),
    bool(full_mask),
  )
  with _MASK_CACHE_LOCK:
    entry = _MASK_PARAMETER_CACHE.get(owner_id)
    if entry is None or entry[0]() is not owner:
      def release(reference: weakref.ReferenceType[torch.Tensor]) -> None:
        with _MASK_CACHE_LOCK:
          current = _MASK_PARAMETER_CACHE.get(owner_id)
          if current is not None and current[0] is reference:
            _MASK_PARAMETER_CACHE.pop(owner_id, None)

      reference = weakref.ref(owner, release)
      parameters: dict[
        tuple[str, int | None, int, int, int, bool], torch.Tensor
      ] = {}
      _MASK_PARAMETER_CACHE[owner_id] = (reference, parameters)
    else:
      parameters = entry[1]
    params = parameters.get(key)
    if params is None:
      params = torch.tensor(
        [int(block_size), int(key_start), int(clean_offset), int(full_mask)],
        dtype=torch.int32,
        device=owner.device,
      )
      parameters[key] = params
    return params


def clear_splitd_metadata_cache() -> None:
  """Release cached launch metadata after a model/topology is destroyed."""

  with _MASK_CACHE_LOCK:
    _MASK_PARAMETER_CACHE.clear()


def _metadata(
  query: torch.Tensor,
  query_blocks: torch.Tensor,
  query_is_clean: torch.Tensor,
  block_size: int,
  key_start: int,
  clean_offset: int,
  full_mask: bool,
) -> list[torch.Tensor]:
  query_len = int(query.shape[1])
  if query_blocks.ndim != 1 or query_is_clean.ndim != 1:
    raise ValueError("BDLM Split-D query metadata must be one-dimensional")
  if query_blocks.numel() < query_len or query_is_clean.numel() < query_len:
    raise ValueError("BDLM Split-D query metadata is shorter than query length")
  if int(block_size) <= 0:
    raise ValueError("BDLM Split-D block_size must be positive")
  if int(key_start) < 0:
    raise ValueError("BDLM Split-D key_start must be a nonnegative logical offset")
  if bool(full_mask) and int(clean_offset) <= 0:
    raise ValueError("full BDLM Split-D masks require a positive clean_offset")
  if not bool(full_mask) and int(clean_offset) != 0:
    raise ValueError("prefix BDLM Split-D masks require clean_offset=0")
  if query_blocks.device != query.device or query_is_clean.device != query.device:
    raise RuntimeError("BDLM Split-D metadata must be on the Q/K/V device")
  if query_blocks.dtype != torch.int32:
    raise TypeError("BDLM Split-D query_blocks must use torch.int32")
  if query_is_clean.dtype != torch.bool:
    raise TypeError("BDLM Split-D query_is_clean must use torch.bool")
  if not query_blocks.is_contiguous() or not query_is_clean.is_contiguous():
    raise ValueError("BDLM Split-D query metadata must be contiguous")
  blocks = query_blocks[:query_len]
  is_clean = query_is_clean[:query_len]
  params = _mask_params(
    query_blocks,
    int(block_size),
    int(key_start),
    int(clean_offset),
    bool(full_mask),
  )
  return [blocks, is_clean, params]


def _interval_metadata(
  query: torch.Tensor,
  key: torch.Tensor,
  query_clean_bounds: torch.Tensor,
  local_query_blocks: torch.Tensor,
  query_is_clean: torch.Tensor,
  key_coordinates: torch.Tensor,
  key_is_clean: torch.Tensor,
  query_tile_bounds: torch.Tensor | None = None,
  key_tile_bounds: torch.Tensor | None = None,
  expected_query_tiles: int | None = None,
  *,
  query_tile_offsets: torch.Tensor | None = None,
  key_tile_work_items: torch.Tensor | None = None,
  key_tile_offsets: torch.Tensor | None = None,
  query_tile_work_items: torch.Tensor | None = None,
) -> list[torch.Tensor]:
  query_len = int(query.shape[1])
  key_len = int(key.shape[1])
  if query_clean_bounds.shape != (query_len, 2):
    raise ValueError("Split-D query_clean_bounds must have shape [query_len, 2]")
  for name, tensor, length in (
    ("local_query_blocks", local_query_blocks, query_len),
    ("query_is_clean", query_is_clean, query_len),
    ("key_coordinates", key_coordinates, key_len),
    ("key_is_clean", key_is_clean, key_len),
  ):
    if tensor.ndim != 1 or int(tensor.numel()) != length:
      raise ValueError(f"Split-D {name} must have shape [{length}]")
  for name, tensor in (
    ("query_clean_bounds", query_clean_bounds),
    ("local_query_blocks", local_query_blocks),
    ("key_coordinates", key_coordinates),
  ):
    if tensor.dtype != torch.int32:
      raise TypeError(f"Split-D {name} must use torch.int32")
  if query_is_clean.dtype != torch.bool or key_is_clean.dtype != torch.bool:
    raise TypeError("Split-D query/key role metadata must use torch.bool")
  metadata = [
    query_clean_bounds,
    local_query_blocks,
    query_is_clean,
    key_coordinates,
    key_is_clean,
  ]
  if (query_tile_bounds is None) != (key_tile_bounds is None):
    raise ValueError("Split-D tile bounds must be supplied together")
  query_worklist = (query_tile_offsets, key_tile_work_items)
  key_worklist = (key_tile_offsets, query_tile_work_items)
  if any(tensor is not None for tensor in query_worklist) and not all(
    tensor is not None for tensor in query_worklist
  ):
    raise ValueError("Split-D query tile worklist tensors must be supplied together")
  if any(tensor is not None for tensor in key_worklist) and not all(
    tensor is not None for tensor in key_worklist
  ):
    raise ValueError("Split-D key tile worklist tensors must be supplied together")
  if key_tile_offsets is not None and query_tile_offsets is None:
    raise ValueError("Split-D key worklists require a query worklist")
  if query_tile_bounds is not None and query_tile_offsets is not None:
    raise ValueError("Split-D tile bounds and tile worklists are mutually exclusive")
  if query_tile_bounds is not None and key_tile_bounds is not None:
    if expected_query_tiles is None:
      expected_query_tiles = (query_len + TILE_M - 1) // TILE_M
    expected_key_tiles = (key_len + TILE_N - 1) // TILE_N
    if query_tile_bounds.shape != (expected_query_tiles, 2):
      raise ValueError("Split-D query tile bounds have an invalid shape")
    if key_tile_bounds.shape != (expected_key_tiles, 2):
      raise ValueError("Split-D key tile bounds have an invalid shape")
    if query_tile_bounds.dtype != torch.int32 or key_tile_bounds.dtype != torch.int32:
      raise TypeError("Split-D tile bounds must use torch.int32")
    metadata.extend((query_tile_bounds, key_tile_bounds))
  elif query_tile_offsets is not None:
    if expected_query_tiles is None:
      expected_query_tiles = (query_len + TILE_M - 1) // TILE_M
    expected_key_tiles = (key_len + TILE_N - 1) // TILE_N
    if query_tile_offsets.shape != (expected_query_tiles + 1,):
      raise ValueError("Split-D query tile offsets have an invalid shape")
    if key_tile_work_items.ndim != 1:
      raise ValueError("Split-D key tile work items must be one-dimensional")
    if any(tensor.dtype != torch.int32 for tensor in query_worklist):
      raise TypeError("Split-D query tile worklists must use torch.int32")
    metadata.extend(query_worklist)
    if key_tile_offsets is not None:
      if key_tile_offsets.shape != (expected_key_tiles + 1,):
        raise ValueError("Split-D key tile offsets have an invalid shape")
      if query_tile_work_items.ndim != 1:
        raise ValueError("Split-D query tile work items must be one-dimensional")
      if any(tensor.dtype != torch.int32 for tensor in key_worklist):
        raise TypeError("Split-D key tile worklists must use torch.int32")
      metadata.extend(key_worklist)
  if any(tensor.device != query.device for tensor in metadata):
    raise RuntimeError("Split-D interval metadata must be on the Q/K/V device")
  if any(not tensor.is_contiguous() for tensor in metadata):
    raise ValueError("Split-D interval metadata must be contiguous")
  return metadata


def configure_splitd_runtime(*, allow_runtime_jit: bool) -> None:
  """Configure the explicit developer-only compilation policy."""

  configure_runtime_jit(allow_runtime_jit=allow_runtime_jit)


def splitd_tile_shape() -> tuple[int, int]:
  """Return the packaged query/key scheduler tile shape."""

  return int(TILE_M), int(TILE_N)


@lru_cache(maxsize=1)
def _dsl_backend() -> tuple[object, object, object, object]:
  """Import compiler-backed kernels only for wheel builds or developer JIT."""

  try:
    forward = importlib.import_module(f"{__name__}._forward")
    backward = importlib.import_module(f"{__name__}._backward")
    mask = importlib.import_module(f"{__name__}._mask")
  except ImportError as error:
    raise RuntimeError(
      "Split-D developer JIT requires the bdlm-flash-attn-3"
      "[developer-jit] dependencies"
    ) from error
  return (
    forward._splitd_forward_sm90,
    backward._splitd_backward_sm90,
    mask.bdlm_mask_mod,
    mask.interval_mask_mod,
  )


def _uses_dsl_backend() -> bool:
  return is_aot_build() or runtime_jit_enabled()


def verify_splitd_artifacts() -> dict[str, object]:
  if runtime_jit_enabled():
    return {"mode": "developer_jit"}
  return verify_packaged_artifacts()


def bdlm_splitd_forward(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  query_blocks: torch.Tensor,
  query_is_clean: torch.Tensor,
  block_size: int,
  key_start: int,
  scale: float,
  clean_offset: int = 0,
  *,
  full_mask: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Compute exact masked D=512 attention and per-row LSE."""

  _validate_qkv(query, key, value)
  metadata = _metadata(
    query,
    query_blocks,
    query_is_clean,
    int(block_size),
    int(key_start),
    int(clean_offset),
    bool(full_mask),
  )
  if not _uses_dsl_backend():
    return _aot_runtime.forward(
      query,
      key,
      value,
      float(scale),
      metadata,
      mask="bdlm",
    )
  splitd_forward, _, mask_mod, _ = _dsl_backend()
  # The masked kernel initializes skipped rows before launch; allocating here
  # avoids paying for the same memset twice.
  output = torch.empty_like(query)
  lse = torch.empty(
    (int(query.shape[0]), int(query.shape[2]), int(query.shape[1])),
    dtype=torch.float32,
    device=query.device,
  )
  return splitd_forward(
    query.detach(),
    key.detach(),
    value.detach(),
    softmax_scale=float(scale),
    causal=False,
    pack_gqa=bool(query.shape[2] != key.shape[2]),
    mask_mod=mask_mod,
    return_lse=True,
    out=output,
    lse=lse,
    aux_tensors=metadata,
    interval_metadata=False,
  )


def bdlm_splitd_backward(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  output: torch.Tensor,
  lse: torch.Tensor,
  grad_output: torch.Tensor,
  grad_lse: torch.Tensor | None,
  query_blocks: torch.Tensor,
  query_is_clean: torch.Tensor,
  block_size: int,
  key_start: int,
  scale: float,
  clean_offset: int = 0,
  *,
  full_mask: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Compute exact D=512 Q/K/V gradients, including an external LSE loss."""

  _validate_qkv(query, key, value)
  metadata = _metadata(
    query,
    query_blocks,
    query_is_clean,
    int(block_size),
    int(key_start),
    int(clean_offset),
    bool(full_mask),
  )
  if not _uses_dsl_backend():
    return _aot_runtime.backward(
      query,
      key,
      value,
      output,
      lse,
      grad_output,
      grad_lse,
      float(scale),
      metadata,
      mask="bdlm",
    )
  _, splitd_backward, mask_mod, _ = _dsl_backend()
  return splitd_backward(
    query.detach(),
    key.detach(),
    value.detach(),
    output.detach(),
    grad_output.to(dtype=query.dtype).contiguous(),
    lse.detach().float().contiguous(),
    softmax_scale=float(scale),
    causal=False,
    dlse=None if grad_lse is None else grad_lse.detach().float().contiguous(),
    aux_tensors=metadata,
    mask_mod=mask_mod,
    interval_metadata=False,
  )


def splitd_interval_forward(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  query_clean_bounds: torch.Tensor,
  local_query_blocks: torch.Tensor,
  query_is_clean: torch.Tensor,
  key_coordinates: torch.Tensor,
  key_is_clean: torch.Tensor,
  scale: float,
  query_tile_bounds: torch.Tensor | None = None,
  key_tile_bounds: torch.Tensor | None = None,
  *,
  query_tile_offsets: torch.Tensor | None = None,
  key_tile_work_items: torch.Tensor | None = None,
  key_tile_offsets: torch.Tensor | None = None,
  query_tile_work_items: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Compute exact D=512 attention over packed clean/active K/V metadata."""

  _validate_qkv(query, key, value)
  q_heads_per_kv_head = int(query.shape[2]) // int(key.shape[2])
  query_rows_per_tile = TILE_M // q_heads_per_kv_head
  metadata = _interval_metadata(
    query,
    key,
    query_clean_bounds,
    local_query_blocks,
    query_is_clean,
    key_coordinates,
    key_is_clean,
    query_tile_bounds,
    key_tile_bounds,
    (int(query.shape[1]) + query_rows_per_tile - 1) // query_rows_per_tile,
    query_tile_offsets=query_tile_offsets,
    key_tile_work_items=key_tile_work_items,
    key_tile_offsets=key_tile_offsets,
    query_tile_work_items=query_tile_work_items,
  )
  bounded = query_tile_bounds is not None
  sparse = query_tile_offsets is not None
  if not _uses_dsl_backend():
    return _aot_runtime.forward(
      query,
      key,
      value,
      float(scale),
      metadata,
      mask="sparse_interval" if sparse else ("bounded_interval" if bounded else "interval"),
    )
  splitd_forward, _, _, mask_mod = _dsl_backend()
  output = torch.empty_like(query)
  lse = torch.empty(
    (int(query.shape[0]), int(query.shape[2]), int(query.shape[1])),
    dtype=torch.float32,
    device=query.device,
  )
  return splitd_forward(
    query.detach(),
    key.detach(),
    value.detach(),
    softmax_scale=float(scale),
    causal=False,
    pack_gqa=bool(query.shape[2] != key.shape[2]),
    mask_mod=mask_mod,
    return_lse=True,
    out=output,
    lse=lse,
    aux_tensors=metadata,
    interval_metadata=True,
    precomputed_tile_bounds=bounded,
    precomputed_tile_worklist=sparse,
  )


def splitd_interval_backward(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  output: torch.Tensor,
  lse: torch.Tensor,
  grad_output: torch.Tensor,
  grad_lse: torch.Tensor | None,
  query_clean_bounds: torch.Tensor,
  local_query_blocks: torch.Tensor,
  query_is_clean: torch.Tensor,
  key_coordinates: torch.Tensor,
  key_is_clean: torch.Tensor,
  scale: float,
  query_tile_bounds: torch.Tensor | None = None,
  key_tile_bounds: torch.Tensor | None = None,
  *,
  query_tile_offsets: torch.Tensor | None = None,
  key_tile_work_items: torch.Tensor | None = None,
  key_tile_offsets: torch.Tensor | None = None,
  query_tile_work_items: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Differentiate exact D=512 packed attention from supplied merged state."""

  _validate_qkv(query, key, value)
  metadata = _interval_metadata(
    query,
    key,
    query_clean_bounds,
    local_query_blocks,
    query_is_clean,
    key_coordinates,
    key_is_clean,
    query_tile_bounds,
    key_tile_bounds,
    query_tile_offsets=query_tile_offsets,
    key_tile_work_items=key_tile_work_items,
    key_tile_offsets=key_tile_offsets,
    query_tile_work_items=query_tile_work_items,
  )
  bounded = query_tile_bounds is not None
  sparse = query_tile_offsets is not None
  if not _uses_dsl_backend():
    return _aot_runtime.backward(
      query,
      key,
      value,
      output,
      lse,
      grad_output,
      grad_lse,
      float(scale),
      metadata,
      mask="sparse_interval" if sparse else ("bounded_interval" if bounded else "interval"),
    )
  _, splitd_backward, _, mask_mod = _dsl_backend()
  return splitd_backward(
    query.detach(),
    key.detach(),
    value.detach(),
    output.detach(),
    grad_output.to(dtype=query.dtype).contiguous(),
    lse.detach().float().contiguous(),
    softmax_scale=float(scale),
    causal=False,
    dlse=None if grad_lse is None else grad_lse.detach().float().contiguous(),
    aux_tensors=metadata,
    mask_mod=mask_mod,
    interval_metadata=True,
    precomputed_tile_bounds=bounded,
    precomputed_tile_worklist=sparse,
  )


def splitd_interval_backward_prepare(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  output: torch.Tensor,
  lse: torch.Tensor,
  grad_output: torch.Tensor,
  grad_lse: torch.Tensor | None,
  query_clean_bounds: torch.Tensor,
  local_query_blocks: torch.Tensor,
  query_is_clean: torch.Tensor,
  key_coordinates: torch.Tensor,
  key_is_clean: torch.Tensor,
  scale: float,
  query_tile_bounds: torch.Tensor | None = None,
  key_tile_bounds: torch.Tensor | None = None,
  *,
  query_tile_offsets: torch.Tensor | None = None,
  key_tile_work_items: torch.Tensor | None = None,
  key_tile_offsets: torch.Tensor | None = None,
  query_tile_work_items: torch.Tensor | None = None,
) -> object:
  """Prepare an AOT D=512 interval backward for split gradient launches."""

  _validate_qkv(query, key, value)
  if _uses_dsl_backend():
    raise RuntimeError(
      "phased Split-D backward requires packaged production artifacts"
    )
  metadata = _interval_metadata(
    query,
    key,
    query_clean_bounds,
    local_query_blocks,
    query_is_clean,
    key_coordinates,
    key_is_clean,
    query_tile_bounds,
    key_tile_bounds,
    query_tile_offsets=query_tile_offsets,
    key_tile_work_items=key_tile_work_items,
    key_tile_offsets=key_tile_offsets,
    query_tile_work_items=query_tile_work_items,
  )
  bounded = query_tile_bounds is not None
  sparse = query_tile_offsets is not None
  return _aot_runtime.prepare_backward(
    query,
    key,
    value,
    output,
    lse,
    grad_output,
    grad_lse,
    float(scale),
    metadata,
    mask="sparse_interval" if sparse else ("bounded_interval" if bounded else "interval"),
  )


def splitd_backward_dkdv(
  state: object,
  *,
  grad_key: torch.Tensor | None = None,
  grad_value: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Launch dK/dV from one prepared production backward state."""

  if not isinstance(state, _aot_runtime.PreparedBackward):
    raise TypeError("invalid prepared Split-D backward state")
  return _aot_runtime.backward_dkdv(
    state,
    grad_key=grad_key,
    grad_value=grad_value,
  )


def splitd_backward_dq(
  state: object,
  *,
  grad_query: torch.Tensor | None = None,
) -> torch.Tensor:
  """Launch dQ from one prepared production backward state."""

  if not isinstance(state, _aot_runtime.PreparedBackward):
    raise TypeError("invalid prepared Split-D backward state")
  return _aot_runtime.backward_dq(state, grad_query=grad_query)


def splitd_full_forward(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Compute dense D=512 attention with the native Split-D kernel."""

  _validate_qkv(query, key, value)
  if not _uses_dsl_backend():
    return _aot_runtime.forward(query, key, value, float(scale), None)
  splitd_forward, _, _, _ = _dsl_backend()
  return splitd_forward(
    query.detach(),
    key.detach(),
    value.detach(),
    softmax_scale=float(scale),
    causal=False,
    pack_gqa=bool(query.shape[2] != key.shape[2]),
    return_lse=True,
  )


def splitd_full_backward(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  output: torch.Tensor,
  lse: torch.Tensor,
  grad_output: torch.Tensor,
  grad_lse: torch.Tensor | None,
  scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Compute dense D=512 Q/K/V gradients with native Split-D kernels."""

  _validate_qkv(query, key, value)
  if not _uses_dsl_backend():
    return _aot_runtime.backward(
      query,
      key,
      value,
      output,
      lse,
      grad_output,
      grad_lse,
      float(scale),
      None,
    )
  _, splitd_backward, _, _ = _dsl_backend()
  return splitd_backward(
    query.detach(),
    key.detach(),
    value.detach(),
    output.detach(),
    grad_output.to(dtype=query.dtype).contiguous(),
    lse.detach().float().contiguous(),
    softmax_scale=float(scale),
    causal=False,
    dlse=None if grad_lse is None else grad_lse.detach().float().contiguous(),
  )


class _BDLMSplitDAttention(torch.autograd.Function):
  @staticmethod
  def forward(
    ctx: object,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_blocks: torch.Tensor,
    query_is_clean: torch.Tensor,
    block_size: int,
    key_start: int,
    scale: float,
    clean_offset: int,
    full_mask: bool,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    output, lse = bdlm_splitd_forward(
      query,
      key,
      value,
      query_blocks,
      query_is_clean,
      int(block_size),
      int(key_start),
      float(scale),
      int(clean_offset),
      full_mask=bool(full_mask),
    )
    ctx.save_for_backward(
      query,
      key,
      value,
      output,
      lse,
      query_blocks,
      query_is_clean,
    )
    ctx.block_size = int(block_size)
    ctx.key_start = int(key_start)
    ctx.scale = float(scale)
    ctx.clean_offset = int(clean_offset)
    ctx.full_mask = bool(full_mask)
    return output, lse

  @staticmethod
  def backward(
    ctx: object,
    grad_output: torch.Tensor,
    grad_lse: torch.Tensor | None,
  ) -> tuple[torch.Tensor | None, ...]:
    query, key, value, output, lse, query_blocks, query_is_clean = ctx.saved_tensors
    grad_query, grad_key, grad_value = bdlm_splitd_backward(
      query,
      key,
      value,
      output,
      lse,
      grad_output,
      grad_lse,
      query_blocks,
      query_is_clean,
      ctx.block_size,
      ctx.key_start,
      ctx.scale,
      ctx.clean_offset,
      full_mask=ctx.full_mask,
    )
    return grad_query, grad_key, grad_value, None, None, None, None, None, None, None


class _SplitDFullAttention(torch.autograd.Function):
  @staticmethod
  def forward(
    ctx: object,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    output, lse = splitd_full_forward(query, key, value, float(scale))
    ctx.save_for_backward(query, key, value, output, lse)
    ctx.scale = float(scale)
    return output, lse

  @staticmethod
  def backward(
    ctx: object,
    grad_output: torch.Tensor,
    grad_lse: torch.Tensor | None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
    query, key, value, output, lse = ctx.saved_tensors
    gradients = splitd_full_backward(
      query,
      key,
      value,
      output,
      lse,
      grad_output,
      grad_lse,
      ctx.scale,
    )
    return *gradients, None


def bdlm_splitd_attention(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  query_blocks: torch.Tensor,
  query_is_clean: torch.Tensor,
  block_size: int,
  key_start: int,
  scale: float,
  clean_offset: int = 0,
  *,
  full_mask: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
  return _BDLMSplitDAttention.apply(
    query,
    key,
    value,
    query_blocks,
    query_is_clean,
    int(block_size),
    int(key_start),
    float(scale),
    int(clean_offset),
    bool(full_mask),
  )


def splitd_full_attention(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  return _SplitDFullAttention.apply(query, key, value, float(scale))


__all__ = [
  "HEAD_DIM",
  "SUPPORTED_Q_HEADS_PER_KV_HEAD",
  "bdlm_splitd_attention",
  "bdlm_splitd_backward",
  "bdlm_splitd_forward",
  "clear_splitd_metadata_cache",
  "configure_splitd_runtime",
  "splitd_full_attention",
  "splitd_full_backward",
  "splitd_full_forward",
  "splitd_interval_backward",
  "splitd_interval_backward_prepare",
  "splitd_interval_forward",
  "splitd_backward_dkdv",
  "splitd_backward_dq",
  "splitd_tile_shape",
  "verify_splitd_artifacts",
]
