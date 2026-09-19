# Copyright 2026 The dllm_parallel Authors.
# SPDX-License-Identifier: Apache-2.0

"""Torch-facing runtime for packaged D=512 Split-D artifacts.

This module intentionally has no CuTe compiler dependency. Production training
loads immutable TVM-FFI functions selected by the semantic artifact manifest.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import tvm_ffi

from ._capabilities import HEAD_DIM, TILE_M
from ._artifacts import load_packaged_variant


@dataclass(frozen=True)
class PreparedBackward:
  """Validated Split-D state shared by independently launched gradient phases."""

  query: torch.Tensor
  key: torch.Tensor
  value: torch.Tensor
  grad_output: torch.Tensor
  dpsum: torch.Tensor
  lse_log2: torch.Tensor
  metadata: list[torch.Tensor] | None
  scale: float
  functions: dict[str, object] | None


def _dtype_name(dtype: torch.dtype) -> str:
  if dtype == torch.bfloat16:
    return "bfloat16"
  if dtype == torch.float16:
    return "float16"
  raise TypeError("native Split-D requires bf16 or fp16 tensors")


def _variant_base(
  query: torch.Tensor,
  key: torch.Tensor,
  *,
  mask: str,
) -> str:
  ratio = int(query.shape[2] // key.shape[2])
  return f"{_dtype_name(query.dtype)}:{mask}:gqa{ratio}"


def _call_on_current_stream(fn: object, *args: object, device: torch.device) -> None:
  stream = torch.cuda.current_stream(device=device)
  with tvm_ffi.use_torch_stream(torch.cuda.stream(stream)):
    fn(*args)


def _validate_tensor(
  tensor: torch.Tensor,
  *,
  name: str,
  shape: tuple[int, ...],
  dtype: torch.dtype,
  device: torch.device,
) -> None:
  if tensor.shape != shape:
    raise ValueError(f"{name} has shape {tensor.shape}, expected {shape}")
  if tensor.dtype != dtype:
    raise TypeError(f"{name} has dtype {tensor.dtype}, expected {dtype}")
  if tensor.device != device:
    raise RuntimeError(f"{name} is on {tensor.device}, expected {device}")


def forward(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  scale: float,
  metadata: list[torch.Tensor] | None,
  mask: str = "dense",
) -> tuple[torch.Tensor, torch.Tensor]:
  batch, query_len, query_heads, _ = query.shape
  key_len = int(key.shape[1])
  output = torch.empty_like(query)
  lse = torch.empty(
    (batch, query_heads, query_len),
    dtype=torch.float32,
    device=query.device,
  )
  masked = mask != "dense"
  if masked or key_len == 0:
    output.zero_()
    lse.fill_(-torch.inf)
  if query_len == 0 or key_len == 0:
    return output, lse

  variant = _variant_base(query, key, mask=mask)
  function = load_packaged_variant("forward", variant)["forward"]
  _call_on_current_stream(
    function,
    query.detach(),
    key.detach(),
    value.detach(),
    output.detach(),
    lse,
    float(scale),
    None,
    None,
    None,
    None,
    metadata,
    device=query.device,
  )
  return output, lse


def prepare_backward(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  output: torch.Tensor,
  lse: torch.Tensor,
  grad_output: torch.Tensor,
  grad_lse: torch.Tensor | None,
  scale: float,
  metadata: list[torch.Tensor] | None,
  mask: str = "dense",
) -> PreparedBackward:
  output = output.contiguous()
  grad_output = grad_output.to(dtype=query.dtype).contiguous()
  lse = lse.float().contiguous()
  grad_lse = None if grad_lse is None else grad_lse.float().contiguous()

  batch, query_len, query_heads, head_dim = query.shape
  if head_dim != HEAD_DIM:
    raise ValueError(f"native Split-D requires head_dim={HEAD_DIM}")
  output_shape = tuple(query.shape)
  lse_shape = (batch, query_heads, query_len)
  _validate_tensor(
    output,
    name="output",
    shape=output_shape,
    dtype=query.dtype,
    device=query.device,
  )
  _validate_tensor(
    grad_output,
    name="grad_output",
    shape=output_shape,
    dtype=query.dtype,
    device=query.device,
  )
  _validate_tensor(
    lse,
    name="lse",
    shape=lse_shape,
    dtype=torch.float32,
    device=query.device,
  )
  if grad_lse is not None:
    _validate_tensor(
      grad_lse,
      name="grad_lse",
      shape=lse_shape,
      dtype=torch.float32,
      device=query.device,
    )
  rounded_query_len = ((query_len + TILE_M - 1) // TILE_M) * TILE_M
  dpsum = torch.empty(
    (batch, query_heads, rounded_query_len),
    dtype=torch.float32,
    device=query.device,
  )
  lse_log2 = torch.empty_like(dpsum)
  functions = None
  if query_len > 0 and key.shape[1] > 0:
    variant = (
      f"{_variant_base(query, key, mask=mask)}:"
      f"dlse{int(grad_lse is not None)}"
    )
    functions = load_packaged_variant("backward", variant)
    _call_on_current_stream(
      functions["preprocess"],
      output.detach(),
      grad_output,
      dpsum,
      lse,
      lse_log2,
      None,
      None,
      None,
      grad_lse,
      device=query.device,
    )
  return PreparedBackward(
    query=query,
    key=key,
    value=value,
    grad_output=grad_output,
    dpsum=dpsum,
    lse_log2=lse_log2,
    metadata=metadata,
    scale=float(scale),
    functions=functions,
  )


def backward_dkdv(
  state: PreparedBackward,
  *,
  grad_key: torch.Tensor | None = None,
  grad_value: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  dk = torch.zeros_like(state.key) if grad_key is None else grad_key
  dv = torch.zeros_like(state.value) if grad_value is None else grad_value
  _validate_tensor(
    dk,
    name="grad_key",
    shape=tuple(state.key.shape),
    dtype=state.key.dtype,
    device=state.key.device,
  )
  _validate_tensor(
    dv,
    name="grad_value",
    shape=tuple(state.value.shape),
    dtype=state.value.dtype,
    device=state.value.device,
  )
  if grad_key is not None:
    dk.zero_()
  if grad_value is not None:
    dv.zero_()
  if state.functions is None:
    return dk, dv
  _call_on_current_stream(
    state.functions["dkdv"],
    state.query.detach(),
    state.key.detach(),
    state.value.detach(),
    state.grad_output,
    state.lse_log2,
    state.dpsum,
    dk,
    dv,
    state.scale,
    None,
    None,
    state.metadata,
    device=state.query.device,
  )
  return dk, dv


def backward_dq(
  state: PreparedBackward,
  *,
  grad_query: torch.Tensor | None = None,
) -> torch.Tensor:
  dq = torch.zeros_like(state.query) if grad_query is None else grad_query
  _validate_tensor(
    dq,
    name="grad_query",
    shape=tuple(state.query.shape),
    dtype=state.query.dtype,
    device=state.query.device,
  )
  if grad_query is not None:
    dq.zero_()
  if state.functions is None:
    return dq
  _call_on_current_stream(
    state.functions["dq"],
    state.query.detach(),
    state.key.detach(),
    state.value.detach(),
    state.grad_output,
    state.lse_log2,
    state.dpsum,
    dq,
    state.scale,
    None,
    None,
    state.metadata,
    device=state.query.device,
  )
  return dq


def backward(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  output: torch.Tensor,
  lse: torch.Tensor,
  grad_output: torch.Tensor,
  grad_lse: torch.Tensor | None,
  scale: float,
  metadata: list[torch.Tensor] | None,
  mask: str = "dense",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  # Keep the ordinary backward on its proven direct allocation path.  The
  # prepared state below is reserved for callers that explicitly overlap the
  # independent dK/dV and dQ phases; routing every backward through it changed
  # allocator lifetime at very long sequence lengths.
  output = output.contiguous()
  grad_output = grad_output.to(dtype=query.dtype).contiguous()
  lse = lse.float().contiguous()
  grad_lse = None if grad_lse is None else grad_lse.float().contiguous()

  batch, query_len, query_heads, head_dim = query.shape
  if head_dim != HEAD_DIM:
    raise ValueError(f"native Split-D requires head_dim={HEAD_DIM}")
  output_shape = tuple(query.shape)
  lse_shape = (batch, query_heads, query_len)
  _validate_tensor(
    output,
    name="output",
    shape=output_shape,
    dtype=query.dtype,
    device=query.device,
  )
  _validate_tensor(
    grad_output,
    name="grad_output",
    shape=output_shape,
    dtype=query.dtype,
    device=query.device,
  )
  _validate_tensor(
    lse,
    name="lse",
    shape=lse_shape,
    dtype=torch.float32,
    device=query.device,
  )
  if grad_lse is not None:
    _validate_tensor(
      grad_lse,
      name="grad_lse",
      shape=lse_shape,
      dtype=torch.float32,
      device=query.device,
    )
  dq = torch.zeros_like(query)
  dk = torch.zeros_like(key)
  dv = torch.zeros_like(value)
  if query_len == 0 or key.shape[1] == 0:
    return dq, dk, dv

  rounded_query_len = ((query_len + TILE_M - 1) // TILE_M) * TILE_M
  dpsum = torch.empty(
    (batch, query_heads, rounded_query_len),
    dtype=torch.float32,
    device=query.device,
  )
  lse_log2 = torch.empty_like(dpsum)
  variant = (
    f"{_variant_base(query, key, mask=mask)}:"
    f"dlse{int(grad_lse is not None)}"
  )
  functions = load_packaged_variant("backward", variant)
  _call_on_current_stream(
    functions["preprocess"],
    output.detach(),
    grad_output,
    dpsum,
    lse,
    lse_log2,
    None,
    None,
    None,
    grad_lse,
    device=query.device,
  )
  _call_on_current_stream(
    functions["dkdv"],
    query.detach(),
    key.detach(),
    value.detach(),
    grad_output,
    lse_log2,
    dpsum,
    dk,
    dv,
    float(scale),
    None,
    None,
    metadata,
    device=query.device,
  )
  _call_on_current_stream(
    functions["dq"],
    query.detach(),
    key.detach(),
    value.detach(),
    grad_output,
    lse_log2,
    dpsum,
    dq,
    float(scale),
    None,
    None,
    metadata,
    device=query.device,
  )
  return dq, dk, dv


__all__ = [
  "PreparedBackward",
  "backward",
  "backward_dkdv",
  "backward_dq",
  "forward",
  "prepare_backward",
]
