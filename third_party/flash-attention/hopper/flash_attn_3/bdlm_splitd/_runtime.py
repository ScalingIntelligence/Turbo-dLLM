# Copyright 2026 The dllm_parallel Authors.
# Portions adapted from FlashAttention (BSD-3-Clause).
# SPDX-License-Identifier: Apache-2.0

"""Deterministic compiler contract for Hopper D=512 Split-D attention."""

from __future__ import annotations

from typing import Callable, Optional

import cutlass
import torch
import tvm_ffi

from ._capabilities import ARCHITECTURE, HEAD_DIM, TILE_M, TILE_N
from ._artifacts import is_aot_build


SM90_FWD_TILE_M = TILE_M
SM90_FWD_TILE_N = TILE_N
SM90_BWD_TILE_M = TILE_M
SM90_BWD_TILE_N = TILE_N
SM90_COMPILE_OPTIONS = (
  f"--enable-tvm-ffi --gpu-arch {ARCHITECTURE} "
  "--ptxas-options '--verbose --warn-on-spills --warn-on-local-memory-usage'"
)

torch2cute_dtype_map = {
  torch.float16: cutlass.Float16,
  torch.bfloat16: cutlass.BFloat16,
}


def is_fake_mode() -> bool:
  """Return whether the explicit wheel-build compiler is active."""

  return is_aot_build()


def maybe_contiguous(tensor: torch.Tensor | None) -> torch.Tensor | None:
  if tensor is None or tensor.is_contiguous():
    return tensor
  return tensor.contiguous()


def _call_with_tvm_ffi_current_stream(fn, *args, device: torch.device):
  if is_fake_mode() or device.type != "cuda":
    return fn(*args)
  stream = torch.cuda.current_stream(device=device)
  with tvm_ffi.use_torch_stream(torch.cuda.stream(stream)):
    return fn(*args)


def _validate_tensor(
  tensor: torch.Tensor | None,
  name: str,
  expected_shape: tuple[int, ...],
  expected_dtype: torch.dtype,
  expected_device: torch.device,
) -> None:
  if tensor is None:
    raise ValueError(f"{name} must not be None")
  if tensor.shape != expected_shape:
    raise ValueError(f"{name} has shape {tensor.shape}, expected {expected_shape}")
  if tensor.dtype != expected_dtype:
    raise TypeError(f"{name} has dtype {tensor.dtype}, expected {expected_dtype}")
  if tensor.device != expected_device:
    raise RuntimeError(f"{name} is on {tensor.device}, expected {expected_device}")


def _validate_sm90_arch(device: torch.device) -> tuple[int, str]:
  if is_aot_build():
    return 90, ARCHITECTURE
  major, minor = torch.cuda.get_device_capability(device)
  if major != 9:
    raise RuntimeError(
      "native D=512 Split-D requires Hopper compute capability 9.x; "
      f"got {major}.{minor}"
    )
  return major * 10 + minor, ARCHITECTURE


def _validate_training_dtype(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  requires_grad: bool,
) -> None:
  del key, value, requires_grad
  if query.dtype not in torch2cute_dtype_map:
    raise TypeError("native D=512 Split-D requires fp16 or bf16 inputs")


def _validate_cu_seqlens(
  tensor: torch.Tensor | None,
  name: str,
  batch_size: int | None = None,
  total_tokens: int | None = None,
) -> None:
  if tensor is None:
    return
  if tensor.ndim != 1 or tensor.numel() == 0:
    raise ValueError(f"{name} must be a nonempty one-dimensional tensor")
  if batch_size is not None and tensor.shape != (batch_size + 1,):
    raise ValueError(f"{name} must have shape ({batch_size + 1},)")
  if tensor.dtype != torch.int32 or not tensor.is_contiguous():
    raise TypeError(f"{name} must be contiguous torch.int32")
  if is_fake_mode():
    return
  if not tensor.is_cuda:
    raise RuntimeError(f"{name} must be a CUDA tensor")
  if int(tensor[0].item()) != 0:
    raise ValueError(f"{name}[0] must be zero")
  if total_tokens is not None and int(tensor[-1].item()) != total_tokens:
    raise ValueError(f"{name}[-1] must equal {total_tokens}")
  if tensor.numel() > 1 and bool(torch.any(tensor[1:] < tensor[:-1]).item()):
    raise ValueError(f"{name} must be monotonically non-decreasing")


def _validate_max_seqlen_for_cu_seqlens(
  tensor: torch.Tensor | None,
  name: str,
  max_seqlen: int | None,
  max_name: str,
) -> None:
  if tensor is None:
    return
  if isinstance(max_seqlen, bool) or not isinstance(max_seqlen, int):
    raise TypeError(f"{max_name} must be an integer when {name} is provided")
  if max_seqlen < 0:
    raise ValueError(f"{max_name} must be non-negative")
  if not is_fake_mode():
    lengths = tensor[1:] - tensor[:-1]
    actual_max = int(lengths.max().item()) if lengths.numel() else 0
    if max_seqlen < actual_max:
      raise ValueError(f"{max_name}={max_seqlen} is smaller than {actual_max}")


def _validate_qkv_common(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  cu_seqlens_q: torch.Tensor | None = None,
  cu_seqlens_k: torch.Tensor | None = None,
) -> tuple[int, int | None, int, int, int, int, int, int]:
  query_rank = 3 if cu_seqlens_q is not None else 4
  key_rank = 3 if cu_seqlens_k is not None else 4
  if query.ndim != query_rank or key.ndim != key_rank or value.ndim != key_rank:
    raise ValueError("native Split-D received incompatible Q/K/V ranks")

  query_heads, head_dim = query.shape[-2:]
  key_len, kv_heads, key_dim = key.shape[-3:]
  value_len, value_heads, value_dim = value.shape[-3:]
  if head_dim != HEAD_DIM or key_dim != HEAD_DIM or value_dim != HEAD_DIM:
    raise ValueError(f"native Split-D requires Q/K/V head_dim={HEAD_DIM}")
  if query_heads % kv_heads != 0:
    raise ValueError("query heads must be divisible by K/V heads")
  if key_len != value_len or kv_heads != value_heads:
    raise ValueError("key and value sequence/head shapes must match")
  if query.dtype not in torch2cute_dtype_map or key.dtype != query.dtype or value.dtype != query.dtype:
    raise TypeError("native Split-D requires matching fp16 or bf16 Q/K/V")
  if not is_fake_mode() and (not query.is_cuda or not key.is_cuda or not value.is_cuda):
    raise RuntimeError("native Split-D requires CUDA Q/K/V tensors")
  if key.device != query.device or value.device != query.device:
    raise RuntimeError("native Split-D requires Q/K/V on the same device")

  if cu_seqlens_q is None:
    batch_size, query_len = query.shape[:2]
    total_q = batch_size * query_len
  else:
    batch_size = cu_seqlens_q.numel() - 1
    query_len = None
    total_q = query.shape[0]
    _validate_cu_seqlens(cu_seqlens_q, "cu_seqlens_q", batch_size, total_q)

  if cu_seqlens_k is None:
    expected_key_shape = (batch_size, key_len, kv_heads, head_dim)
  else:
    _validate_cu_seqlens(cu_seqlens_k, "cu_seqlens_k", batch_size, key_len)
    expected_key_shape = (key_len, kv_heads, head_dim)
  if key.shape != expected_key_shape or value.shape != expected_key_shape:
    raise ValueError("native Split-D received incompatible Q/K/V batch shapes")
  return (
    batch_size,
    query_len,
    total_q,
    key_len,
    query_heads,
    kv_heads,
    head_dim,
    value_dim,
  )


def _unsupported_training_features(
  requires_grad: bool,
  softcap: float | None,
  local: bool,
  score_mod: Optional[Callable],
  mask_mod: Optional[Callable],
  aux_tensors: list[torch.Tensor] | None,
) -> None:
  del mask_mod, aux_tensors
  unsupported: list[str] = []
  if requires_grad and softcap is not None:
    unsupported.append("softcap")
  if requires_grad and local:
    unsupported.append("local/window attention")
  if requires_grad and score_mod is not None:
    unsupported.append("score_mod")
  if unsupported:
    raise NotImplementedError(
      "native D=512 Split-D training does not support " + ", ".join(unsupported)
    )


def _resolve_causal_local_window(
  causal: bool,
  window_size_left: int | None,
  window_size_right: int | None,
  mask_mod: Optional[Callable] = None,
) -> tuple[bool, bool, int | None, int | None]:
  if window_size_left is None and window_size_right is None:
    return causal, False, None, None
  if causal:
    raise ValueError("causal and window attention are mutually exclusive")
  if mask_mod is not None:
    raise ValueError("BDLM masking and window attention are mutually exclusive")
  if window_size_left is not None and window_size_right is not None:
    if window_size_left < 0 and window_size_right < 0:
      return False, False, None, None
    if window_size_left < 0 and window_size_right == 0:
      return True, False, None, None
  return False, True, window_size_left, window_size_right
