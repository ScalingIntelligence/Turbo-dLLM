# Copyright 2026 The dllm_parallel Authors.
# Portions adapted from FlashAttention (BSD-3-Clause).
# SPDX-License-Identifier: Apache-2.0

"""Native FlashAttention Split-D backward compiler for Hopper head_dim=512."""

import math
from typing import Callable, Optional, Tuple

import torch
import cutlass.cute as cute
from cutlass import Int32, Float32
from quack.compile_utils import make_fake_tensor as fake_tensor

from ._runtime import (
  SM90_BWD_TILE_M,
  SM90_BWD_TILE_N,
  SM90_COMPILE_OPTIONS,
  is_fake_mode,
  maybe_contiguous,
  _call_with_tvm_ffi_current_stream,
  _validate_tensor,
  _validate_sm90_arch,
  _validate_max_seqlen_for_cu_seqlens,
  _validate_qkv_common,
  _resolve_causal_local_window,
  torch2cute_dtype_map,
)
from ._bwd_preprocess import BDLMSplitDBwdPreprocess
from ._dkdv_d512_sm90 import BDLMSplitDBwdDKDVSm90
from ._dq_d512_sm90 import BDLMSplitDBwdDQSm90
from ._artifacts import get_jit_cache
from ._cute.cute_dsl_utils import (
  to_cute_tensor,
  to_cute_aux_tensor,
  get_aux_tensor_metadata,
)

def _make_fake_bwd_preprocess_tensors(dtype, varlen_q):
  sym = cute.sym_int
  div = 128 // dtype.width  # 8 for fp16/bf16
  b, seqlen_q, h_q, d_v = sym(), sym(), sym(), sym()
  seqlen_q_rounded = sym()
  total_q, total_q_rounded = sym(), sym()
  b_seqlenq = (b, seqlen_q) if not varlen_q else (total_q, )
  mO = fake_tensor(dtype, (*b_seqlenq, h_q, d_v), divisibility=div)
  mdO = fake_tensor(dtype, (*b_seqlenq, h_q, d_v), divisibility=div)
  if not varlen_q:
    mLSE = fake_tensor(Float32, (b, h_q, seqlen_q), divisibility=1)
    mLSElog2 = fake_tensor(Float32, (b, h_q, seqlen_q_rounded), divisibility=4)
    mPdPsum = fake_tensor(Float32, (b, h_q, seqlen_q_rounded), divisibility=4)
  else:
    mLSE = fake_tensor(Float32, (h_q, total_q), divisibility=1)
    mLSElog2 = fake_tensor(Float32, (h_q, total_q_rounded), divisibility=4)
    mPdPsum = fake_tensor(Float32, (h_q, total_q_rounded), divisibility=4)
  return mO, mdO, mLSE, mLSElog2, mPdPsum


def _compile_bwd_preprocess(
  dtype,
  head_dim,
  head_dim_v,
  m_block_size,
  has_cuseqlens_q,
  has_dlse,
  device_arch,
  cute_arch_key,
):
  """Compile bwd preprocess kernel using cute fake tensors."""
  batchp1 = cute.sym_int()
  mO, mdO, mLSE, mLSElog2, mPdPsum = _make_fake_bwd_preprocess_tensors(
    dtype, varlen_q=has_cuseqlens_q
  )
  mCuSeqlensQ = fake_tensor(
    Int32, (batchp1, ), divisibility=1
  ) if has_cuseqlens_q else None
  mdLSE = fake_tensor(Float32, mLSE.shape, divisibility=1) if has_dlse else None
  splitd_bwd_pre = BDLMSplitDBwdPreprocess(
    dtype, head_dim, head_dim_v, m_block_size
  )
  return cute.compile(
    splitd_bwd_pre,
    mO,
    mdO,
    mPdPsum,
    mLSE,
    mLSElog2,
    None,
    mCuSeqlensQ,
    None,
    mdLSE,
    cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    options=SM90_COMPILE_OPTIONS,
  )


def _bwd_preprocess(
  out,
  dout,
  dpsum,
  lse,
  lse_log2,
  cu_seqlens_q,
  dlse,
  dtype,
  head_dim,
  head_dim_v,
  m_block_size,
  device_arch,
  cute_arch_key,
):
  """Backward preprocess: compute (o * dout).sum(dim=-1) - dLSE, and lse * log2_e."""
  is_varlen = cu_seqlens_q is not None
  compile_key = (
    dtype,
    head_dim,
    head_dim_v,
    m_block_size,
    is_varlen,
    dlse is not None,
    device_arch,
    cute_arch_key,
  )
  if compile_key not in _bwd_preprocess.compile_cache:
    _bwd_preprocess.compile_cache[compile_key] = _compile_bwd_preprocess(
      *compile_key
    )
  if not is_fake_mode():
    _call_with_tvm_ffi_current_stream(
      _bwd_preprocess.compile_cache[compile_key],
      out,
      dout,
      dpsum,
      lse,
      lse_log2,
      None,
      cu_seqlens_q,
      None,
      dlse,
      device=out.device,
    )


_bwd_preprocess.compile_cache = get_jit_cache("bwd_pre_sm90")


def _splitd_backward_sm90(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  out: torch.Tensor,
  dout: torch.Tensor,
  lse: torch.Tensor,
  softmax_scale: Optional[float] = None,
  causal: bool = False,
  softcap: float = 0.0,
  window_size_left: Optional[int] = None,
  window_size_right: Optional[int] = None,
  cu_seqlens_q: Optional[torch.Tensor] = None,
  cu_seqlens_k: Optional[torch.Tensor] = None,
  max_seqlen_q: Optional[int] = None,
  max_seqlen_k: Optional[int] = None,
  dq: Optional[torch.Tensor] = None,
  dk: Optional[torch.Tensor] = None,
  dv: Optional[torch.Tensor] = None,
  dlse: Optional[torch.Tensor] = None,
  aux_tensors: Optional[list[torch.Tensor]] = None,
  mask_mod: Optional[Callable] = None,
  interval_metadata: bool = False,
  precomputed_tile_bounds: bool = False,
  precomputed_tile_worklist: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Run the repository's Hopper Split-D FlashAttention backward kernels."""
  if cu_seqlens_q is not None or cu_seqlens_k is not None:
    raise NotImplementedError("BDLM Split-D supports fixed-length BSHD tensors")
  if causal or window_size_left is not None or window_size_right is not None:
    raise NotImplementedError("BDLM Split-D supports noncausal attention only")
  device_arch, cute_arch_key = _validate_sm90_arch(q.device)

  if softcap != 0.0:
    raise NotImplementedError("SplitD backward does not support softcap yet")

  causal, local, window_size_left, window_size_right = _resolve_causal_local_window(
    causal, window_size_left, window_size_right
  )
  if local:
    raise NotImplementedError(
      "SplitD backward does not support local/window attention yet"
    )

  # Fixed tile specialization for the Hopper D=512 kernel.
  m_block_size = SM90_BWD_TILE_M
  n_block_size = SM90_BWD_TILE_N

  q, k, v, out, dout, lse, cu_seqlens_q, cu_seqlens_k = [
    maybe_contiguous(t)
    for t in (q, k, v, out, dout, lse, cu_seqlens_q, cu_seqlens_k)
  ]
  (
    batch_size,
    seqlen_q,
    total_q,
    seqlen_k,
    num_head,
    num_head_kv,
    head_dim,
    head_dim_v,
  ) = _validate_qkv_common(
    q, k, v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k
  )
  _validate_max_seqlen_for_cu_seqlens(
    cu_seqlens_q, "cu_seqlens_q", max_seqlen_q, "max_seqlen_q"
  )
  _validate_max_seqlen_for_cu_seqlens(
    cu_seqlens_k, "cu_seqlens_k", max_seqlen_k, "max_seqlen_k"
  )
  if cu_seqlens_q is None:
    seqlen_q_for_rounding = seqlen_q
  else:
    seqlen_q_for_rounding = max_seqlen_q if max_seqlen_q is not None else total_q

  seqlen_q_rounded = (
    seqlen_q_for_rounding + m_block_size - 1
  ) // m_block_size * m_block_size
  device = q.device
  out_torch_dtype = q.dtype
  if cu_seqlens_q is not None:
    out_shape = (total_q, num_head, head_dim_v)
    lse_shape = (num_head, total_q)
  else:
    out_shape = (batch_size, seqlen_q, num_head, head_dim_v)
    lse_shape = (batch_size, num_head, seqlen_q)
  _validate_tensor(out, "out", out_shape, out_torch_dtype, device)
  _validate_tensor(dout, "dout", out_shape, out_torch_dtype, device)
  _validate_tensor(lse, "lse", lse_shape, torch.float32, device)
  if dlse is not None:
    dlse = maybe_contiguous(dlse)
    _validate_tensor(dlse, "dlse", lse_shape, torch.float32, device)
  if softmax_scale is None:
    softmax_scale = 1.0 / math.sqrt(head_dim)
  qhead_per_kvhead = num_head // num_head_kv
  if head_dim != 512 or head_dim_v != 512:
    raise ValueError("BDLM Split-D backward requires head_dim == head_dim_v == 512")
  expected_metadata = (
    9 if precomputed_tile_worklist
    else (7 if precomputed_tile_bounds else (5 if interval_metadata else 3))
  )
  if aux_tensors is not None and (
    mask_mod is None or len(aux_tensors) != expected_metadata
  ):
    raise ValueError("masked BDLM Split-D backward received an invalid metadata contract")
  aux_tensors = (
    [maybe_contiguous(t) for t in aux_tensors]
    if aux_tensors is not None else None
  )
  aux_tensor_metadata = (
    get_aux_tensor_metadata(aux_tensors) if aux_tensors is not None else None
  )

  if dq is None:
    dq = torch.zeros_like(q)
  else:
    _validate_tensor(dq, "dq", q.shape, out_torch_dtype, device)
    if not is_fake_mode():
      dq.zero_()

  if dk is None:
    dk = torch.zeros_like(k)
  else:
    _validate_tensor(dk, "dk", k.shape, out_torch_dtype, device)
    if not is_fake_mode():
      dk.zero_()

  if dv is None:
    dv = torch.zeros_like(v)
  else:
    _validate_tensor(dv, "dv", v.shape, out_torch_dtype, device)
    if not is_fake_mode():
      dv.zero_()

  if total_q == 0 or seqlen_k == 0:
    return dq, dk, dv

  if cu_seqlens_q is None:
    dpsum = torch.empty(
      batch_size,
      num_head,
      seqlen_q_rounded,
      dtype=torch.float32,
      device=device
    )
    lse_log2 = torch.empty(
      batch_size,
      num_head,
      seqlen_q_rounded,
      dtype=torch.float32,
      device=device
    )
  else:
    total_q_rounded_padded = (
      total_q + cu_seqlens_q.shape[0] * m_block_size - 1
    ) // m_block_size * m_block_size
    dpsum = torch.empty(
      num_head, total_q_rounded_padded, dtype=torch.float32, device=device
    )
    lse_log2 = torch.empty(
      num_head, total_q_rounded_padded, dtype=torch.float32, device=device
    )

  dtype = torch2cute_dtype_map[q.dtype]
  current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

  # (1) Preprocess dpsum and lse_log2 for the SplitD backward kernels.
  _bwd_preprocess(
    out,
    dout,
    dpsum,
    lse,
    lse_log2,
    cu_seqlens_q,
    dlse,
    dtype,
    head_dim,
    head_dim_v,
    m_block_size,
    device_arch,
    cute_arch_key,
  )

  bwd_key = (
    "d512",
    dtype,
    head_dim,
    head_dim_v,
    causal,
    m_block_size,
    n_block_size,
    cu_seqlens_q is not None,
    cu_seqlens_k is not None,
    qhead_per_kvhead,
    interval_metadata,
    precomputed_tile_bounds,
    precomputed_tile_worklist,
    aux_tensor_metadata,
    device_arch,
    cute_arch_key,
  )
  if bwd_key not in _splitd_backward_sm90.compile_cache_dkdv:
    q_t, k_t, v_t, do_t = [to_cute_tensor(t) for t in (q, k, v, dout)]
    dk_t, dv_t = [to_cute_tensor(t) for t in (dk, dv)]
    lse_log2_t = to_cute_tensor(lse_log2, assumed_align=4)
    dpsum_t = to_cute_tensor(dpsum, assumed_align=4)
    cu_seqlens_q_t = (
      to_cute_tensor(cu_seqlens_q, assumed_align=4, leading_dim=0)
      if cu_seqlens_q is not None else None
    )
    cu_seqlens_k_t = (
      to_cute_tensor(cu_seqlens_k, assumed_align=4, leading_dim=0)
      if cu_seqlens_k is not None else None
    )
    cute_aux_tensors = (
      [to_cute_aux_tensor(t) for t in aux_tensors]
      if aux_tensors is not None else None
    )

    splitd_dkdv = BDLMSplitDBwdDKDVSm90(
      dtype,
      head_dim,
      head_dim_v=head_dim_v,
      is_causal=causal,
      qhead_per_kvhead=qhead_per_kvhead,
      tile_m=m_block_size,
      tile_n=n_block_size,
      mask_mod=mask_mod,
      interval_metadata=interval_metadata,
      precomputed_tile_bounds=precomputed_tile_bounds,
      precomputed_tile_worklist=precomputed_tile_worklist,
    )
    _splitd_backward_sm90.compile_cache_dkdv[bwd_key] = cute.compile(
      splitd_dkdv,
      q_t,
      k_t,
      v_t,
      do_t,
      lse_log2_t,
      dpsum_t,
      dk_t,
      dv_t,
      softmax_scale,
      cu_seqlens_q_t,
      cu_seqlens_k_t,
      cute_aux_tensors,
      current_stream,
      options=SM90_COMPILE_OPTIONS,
    )

  if bwd_key not in _splitd_backward_sm90.compile_cache_dq:
    q_t2, k_t2, v_t2, do_t2 = [to_cute_tensor(t) for t in (q, k, v, dout)]
    dq_t = to_cute_tensor(dq)
    lse_log2_t2 = to_cute_tensor(lse_log2, assumed_align=4)
    dpsum_t2 = to_cute_tensor(dpsum, assumed_align=4)
    cu_seqlens_q_t2 = (
      to_cute_tensor(cu_seqlens_q, assumed_align=4, leading_dim=0)
      if cu_seqlens_q is not None else None
    )
    cu_seqlens_k_t2 = (
      to_cute_tensor(cu_seqlens_k, assumed_align=4, leading_dim=0)
      if cu_seqlens_k is not None else None
    )
    cute_aux_tensors2 = (
      [to_cute_aux_tensor(t) for t in aux_tensors]
      if aux_tensors is not None else None
    )

    splitd_dq = BDLMSplitDBwdDQSm90(
      dtype,
      head_dim,
      head_dim_v=head_dim_v,
      is_causal=causal,
      qhead_per_kvhead=qhead_per_kvhead,
      tile_m=m_block_size,
      tile_n=n_block_size,
      mask_mod=mask_mod,
      interval_metadata=interval_metadata,
      precomputed_tile_bounds=precomputed_tile_bounds,
      precomputed_tile_worklist=precomputed_tile_worklist,
    )
    _splitd_backward_sm90.compile_cache_dq[bwd_key] = cute.compile(
      splitd_dq,
      q_t2,
      k_t2,
      v_t2,
      do_t2,
      lse_log2_t2,
      dpsum_t2,
      dq_t,
      softmax_scale,
      cu_seqlens_q_t2,
      cu_seqlens_k_t2,
      cute_aux_tensors2,
      current_stream,
      options=SM90_COMPILE_OPTIONS,
    )

  # Execute dKdV and dQ kernels
  if not is_fake_mode():
    _call_with_tvm_ffi_current_stream(
      _splitd_backward_sm90.compile_cache_dkdv[bwd_key],
      q.detach(),
      k.detach(),
      v.detach(),
      dout,
      lse_log2,
      dpsum,
      dk,
      dv,
      softmax_scale,
      cu_seqlens_q,
      cu_seqlens_k,
      aux_tensors,
      device=device,
    )
    _call_with_tvm_ffi_current_stream(
      _splitd_backward_sm90.compile_cache_dq[bwd_key],
      q.detach(),
      k.detach(),
      v.detach(),
      dout,
      lse_log2,
      dpsum,
      dq,
      softmax_scale,
      cu_seqlens_q,
      cu_seqlens_k,
      aux_tensors,
      device=device,
    )

  return dq, dk, dv


_splitd_backward_sm90.compile_cache_dkdv = get_jit_cache(
  "bdlm_bwd_splitd_dkdv_sm90"
)
_splitd_backward_sm90.compile_cache_dq = get_jit_cache(
  "bdlm_bwd_splitd_dq_sm90"
)
