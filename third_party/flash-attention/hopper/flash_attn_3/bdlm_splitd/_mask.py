# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Exact block-diffusion masking for the D=512 Split-D kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr

from . import _cute as utils
from ._cute.seqlen_info import SeqlenInfoQK


QUERY_BLOCKS = 0
QUERY_IS_CLEAN = 1
MASK_PARAMS = 2
BLOCK_SIZE = 0
KEY_START = 1
CLEAN_OFFSET = 2
FULL_MASK = 3

INTERVAL_QUERY_BOUNDS = 0
INTERVAL_LOCAL_QUERY_BLOCKS = 1
INTERVAL_QUERY_IS_CLEAN = 2
INTERVAL_KEY_COORDINATES = 3
INTERVAL_KEY_IS_CLEAN = 4
INTERVAL_QUERY_TILE_BOUNDS = 5
INTERVAL_KEY_TILE_BOUNDS = 6
INTERVAL_QUERY_TILE_OFFSETS = 5
INTERVAL_KEY_TILE_WORK_ITEMS = 6
INTERVAL_KEY_TILE_OFFSETS = 7
INTERVAL_QUERY_TILE_WORK_ITEMS = 8


@cute.jit
def bdlm_mask_mod(
  batch: cute.TensorSSA,
  head: cute.TensorSSA,
  q_idx: cute.TensorSSA,
  kv_idx: cute.TensorSSA,
  seqlen_info,
  aux_tensors: list,
) -> cute.TensorSSA:
  """Return the exact BDLM clean-prefix/own-noisy-block mask predicate."""

  query_blocks = aux_tensors[QUERY_BLOCKS]
  query_is_clean = aux_tensors[QUERY_IS_CLEAN]
  params = aux_tensors[MASK_PARAMS]

  query_row = utils.ssa_to_scalar(q_idx)
  key_column = utils.ssa_to_scalar(kv_idx)
  query_block = query_blocks[query_row]
  query_clean = query_is_clean[query_row]
  block_size = params[BLOCK_SIZE]
  logical_key_start = params[KEY_START]
  clean_offset = params[CLEAN_OFFSET]
  full_mask = params[FULL_MASK] != Int32(0)

  logical_key = logical_key_start + key_column
  key_clean = logical_key >= clean_offset
  key_position = logical_key
  if key_clean:
    key_position = logical_key - clean_offset
  key_block = key_position // block_size

  clean_query = query_clean
  prefix_allowed = (
    (clean_query & (query_block >= key_block))
    | ((~clean_query) & (query_block > key_block))
  )
  full_allowed = (
    ((~clean_query) & (~key_clean) & (query_block == key_block))
    | ((~clean_query) & key_clean & (query_block > key_block))
    | (clean_query & key_clean & (query_block >= key_block))
  )
  allowed = (full_mask & full_allowed) | ((~full_mask) & prefix_allowed)
  return utils.scalar_to_ssa(allowed, cutlass.Boolean)


@cute.jit
def interval_mask_mod(
  batch: cute.TensorSSA,
  head: cute.TensorSSA,
  q_idx: cute.TensorSSA,
  kv_idx: cute.TensorSSA,
  seqlen_info,
  aux_tensors: list,
) -> cute.TensorSSA:
  """Return the exact clean-interval and local-active mask predicate."""

  del batch, head, seqlen_info
  query_bounds = aux_tensors[INTERVAL_QUERY_BOUNDS]
  local_query_blocks = aux_tensors[INTERVAL_LOCAL_QUERY_BLOCKS]
  query_is_clean = aux_tensors[INTERVAL_QUERY_IS_CLEAN]
  key_coordinates = aux_tensors[INTERVAL_KEY_COORDINATES]
  key_is_clean = aux_tensors[INTERVAL_KEY_IS_CLEAN]

  query_row = utils.ssa_to_scalar(q_idx)
  key_column = utils.ssa_to_scalar(kv_idx)
  clean_start = query_bounds[query_row, 0]
  clean_stop = query_bounds[query_row, 1]
  local_query_block = local_query_blocks[query_row]
  clean_query = query_is_clean[query_row]
  key_coordinate = key_coordinates[key_column]
  clean_key = key_is_clean[key_column]

  active_to_active = (
    (~clean_query)
    & (~clean_key)
    & (local_query_block == key_coordinate)
  )
  clean_context = (
    clean_key
    & (key_coordinate >= clean_start)
    & (key_coordinate < clean_stop)
  )
  return utils.scalar_to_ssa(active_to_active | clean_context, cutlass.Boolean)


@dataclass(frozen=True)
class BDLMBlockInfo:
  """Tile bounds for the exact BDLM mask.

  Forward and dQ own one query tile, so scanning that fixed-size tile yields
  exact K-tile bounds before any QK/PV work. dK/dV uses its full Q range: its
  ordered accumulation protocol cannot skip an interior predecessor tile.
  """

  tile_m: cutlass.Constexpr[int]
  tile_n: cutlass.Constexpr[int]
  is_causal: cutlass.Constexpr[bool]
  is_local: cutlass.Constexpr[bool] = False
  window_size_left: Optional[Int32] = None
  window_size_right: Optional[Int32] = None
  qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1
  aux_tensors: Optional[list] = None
  interval_metadata: cutlass.Constexpr[bool] = False
  precomputed_tile_bounds: cutlass.Constexpr[bool] = False
  precomputed_tile_worklist: cutlass.Constexpr[bool] = False

  @cute.jit
  def get_n_block_count(
    self,
    seqlen_info: SeqlenInfoQK,
    m_block: Int32,
  ) -> Int32:
    if const_expr(
      self.precomputed_tile_worklist and self.aux_tensors is not None
    ):
      offsets = self.aux_tensors[INTERVAL_QUERY_TILE_OFFSETS]
      return offsets[m_block + 1] - offsets[m_block]
    n_block_min, n_block_max = self.get_n_block_min_max(seqlen_info, m_block)
    return max(n_block_max - n_block_min, Int32(0))

  @cute.jit
  def get_n_block_and_full(
    self,
    seqlen_info: SeqlenInfoQK,
    m_block: Int32,
    index: Int32,
  ) -> Tuple[Int32, cutlass.Boolean]:
    if const_expr(
      self.precomputed_tile_worklist and self.aux_tensors is not None
    ):
      offsets = self.aux_tensors[INTERVAL_QUERY_TILE_OFFSETS]
      work_items = self.aux_tensors[INTERVAL_KEY_TILE_WORK_ITEMS]
      stop = offsets[m_block + 1]
      work_item = work_items[stop - Int32(1) - index]
      # The low bit carries the full-tile flag; the remaining bits are index.
      return work_item // Int32(2), (work_item % Int32(2)) != Int32(0)
    _, n_block_max = self.get_n_block_min_max(seqlen_info, m_block)
    return n_block_max - Int32(1) - index, cutlass.Boolean(False)

  @cute.jit
  def get_n_block(
    self,
    seqlen_info: SeqlenInfoQK,
    m_block: Int32,
    index: Int32,
  ) -> Int32:
    n_block, _ = self.get_n_block_and_full(seqlen_info, m_block, index)
    return n_block

  @cute.jit
  def get_m_block_count(
    self,
    seqlen_info: SeqlenInfoQK,
    n_block: Int32,
  ) -> Int32:
    if const_expr(
      self.precomputed_tile_worklist and self.aux_tensors is not None
    ):
      offsets = self.aux_tensors[INTERVAL_KEY_TILE_OFFSETS]
      return offsets[n_block + 1] - offsets[n_block]
    m_block_min, m_block_max = self.get_m_block_min_max(seqlen_info, n_block)
    return max(m_block_max - m_block_min, Int32(0))

  @cute.jit
  def get_m_block_and_full(
    self,
    seqlen_info: SeqlenInfoQK,
    n_block: Int32,
    index: Int32,
  ) -> Tuple[Int32, cutlass.Boolean]:
    if const_expr(
      self.precomputed_tile_worklist and self.aux_tensors is not None
    ):
      offsets = self.aux_tensors[INTERVAL_KEY_TILE_OFFSETS]
      work_items = self.aux_tensors[INTERVAL_QUERY_TILE_WORK_ITEMS]
      work_item = work_items[offsets[n_block] + index]
      # The low bit carries the full-tile flag; the remaining bits are index.
      return work_item // Int32(2), (work_item % Int32(2)) != Int32(0)
    m_block_min, _ = self.get_m_block_min_max(seqlen_info, n_block)
    return m_block_min + index, cutlass.Boolean(False)

  @cute.jit
  def get_m_block(
    self,
    seqlen_info: SeqlenInfoQK,
    n_block: Int32,
    index: Int32,
  ) -> Int32:
    m_block, _ = self.get_m_block_and_full(seqlen_info, n_block, index)
    return m_block

  @cute.jit
  def get_n_block_min_max(
    self,
    seqlen_info: SeqlenInfoQK,
    m_block: Int32,
  ) -> Tuple[Int32, Int32]:
    if const_expr(self.aux_tensors is None):
      return Int32(0), cute.ceil_div(seqlen_info.seqlen_k, self.tile_n)
    if const_expr(self.interval_metadata):
      if const_expr(self.precomputed_tile_bounds):
        bounds = self.aux_tensors[INTERVAL_QUERY_TILE_BOUNDS]
        return bounds[m_block, 0], bounds[m_block, 1]
      # Packed clean K/V coordinates need not be monotonic in storage order.
      # The element mask remains exact; block sparsity is supplied by the
      # distributed exchange, which already sends only required clean rows.
      return Int32(0), cute.ceil_div(seqlen_info.seqlen_k, self.tile_n)

    query_blocks = self.aux_tensors[QUERY_BLOCKS]
    query_is_clean = self.aux_tensors[QUERY_IS_CLEAN]
    params = self.aux_tensors[MASK_PARAMS]
    block_size = params[BLOCK_SIZE]
    logical_key_start = params[KEY_START]
    clean_offset = params[CLEAN_OFFSET]
    full_mask = params[FULL_MASK] != Int32(0)

    packed_row_begin = m_block * self.tile_m
    packed_row_end = packed_row_begin + self.tile_m
    if const_expr(self.qhead_per_kvhead_packgqa > 1):
      row_begin = packed_row_begin // self.qhead_per_kvhead_packgqa
      row_end = cute.ceil_div(packed_row_end, self.qhead_per_kvhead_packgqa)
    else:
      row_begin = packed_row_begin
      row_end = packed_row_end
    row_end = min(row_end, seqlen_info.seqlen_q)

    min_key_begin = logical_key_start + seqlen_info.seqlen_k
    max_key_stop = logical_key_start
    full_clean_start = clean_offset
    noisy_only_shard = full_mask & (
      logical_key_start + seqlen_info.seqlen_k <= full_clean_start
    )
    clean_only_shard = full_mask & (logical_key_start >= full_clean_start)

    # Query ownership may be dual-ended or otherwise non-monotonic. Inspect
    # every live row so bounds remain exact for all legal packed layouts. The
    # compiler fully unrolls this fixed tile reduction into scalar loads/minmax.
    for row_offset in cutlass.range(self.tile_m, unroll_full=True):
      row = row_begin + row_offset
      if row < row_end:
        q_block = query_blocks[row]
        q_clean = query_is_clean[row]
        clean_increment = Int32(0)
        if q_clean:
          clean_increment = Int32(1)
        key_begin = logical_key_start
        key_stop = logical_key_start
        if noisy_only_shard:
          if not q_clean:
            key_begin = q_block * block_size
            key_stop = key_begin + block_size
        elif clean_only_shard:
          key_begin = full_clean_start
          key_stop = full_clean_start + (q_block + clean_increment) * block_size
        elif full_mask:
          key_begin = Int32(0)
          key_stop = full_clean_start + (q_block + clean_increment) * block_size
        else:
          key_begin = Int32(0)
          key_stop = (q_block + clean_increment) * block_size
        min_key_begin = min(min_key_begin, key_begin)
        max_key_stop = max(max_key_stop, key_stop)

    relative_begin = max(min_key_begin - logical_key_start, Int32(0))
    relative_stop = min(
      max_key_stop - logical_key_start,
      seqlen_info.seqlen_k,
    )
    n_block_min = relative_begin // self.tile_n
    n_block_max = cute.ceil_div(relative_stop, self.tile_n)
    if relative_stop <= relative_begin:
      n_block_min = Int32(0)
      n_block_max = Int32(0)
    return n_block_min, n_block_max

  @cute.jit
  def get_m_block_min_max(
    self,
    seqlen_info: SeqlenInfoQK,
    n_block: Int32,
  ) -> Tuple[Int32, Int32]:
    if const_expr(
      self.interval_metadata
      and self.precomputed_tile_bounds
      and self.aux_tensors is not None
    ):
      bounds = self.aux_tensors[INTERVAL_KEY_TILE_BOUNDS]
      return bounds[n_block, 0], bounds[n_block, 1]
    return Int32(0), cute.ceil_div(seqlen_info.seqlen_q, self.tile_m)
