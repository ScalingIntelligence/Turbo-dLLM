# Portions adapted from FlashAttention (BSD-3-Clause).
# Copyright (c) 2025, Tri Dao.

"""Fixed-grid SM90 scheduler used by Split-D training kernels."""

from dataclasses import dataclass
from typing import Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Int32
from cutlass._mlir import ir
from quack.cute_dsl_utils import ParamsBase

try:
  from typing import override
except ImportError:
  from typing_extensions import override


class WorkTileInfo(cutlass.utils.WorkTileInfo):
  """Four-axis Split-D work coordinate: block, head, batch, split."""

  @override
  def __new_from_mlir_values__(self, values: list[ir.Value]) -> "WorkTileInfo":
    if len(values) != 5:
      raise ValueError("Split-D work tiles require five MLIR values")
    tile = cutlass.new_from_mlir_values(self._tile_idx, values[:-1])
    valid = cutlass.new_from_mlir_values(self._is_valid_tile, [values[-1]])
    return WorkTileInfo(tile, valid)


@dataclass
class TileSchedulerArguments(ParamsBase):
  num_block: Int32
  num_head: Int32
  num_batch: Int32


class SingleTileScheduler:
  """Map one CTA directly to one fixed-length attention tile."""

  @dataclass
  class Params(ParamsBase):
    num_block: Int32
    num_head: Int32
    num_batch: Int32

    @staticmethod
    def create(
      args: TileSchedulerArguments,
      *,
      loc=None,
      ip=None,
    ) -> "SingleTileScheduler.Params":
      del loc, ip
      return SingleTileScheduler.Params(
        args.num_block,
        args.num_head,
        args.num_batch,
      )

  def __init__(
    self,
    params: Params,
    block_coordinate: cute.Coord,
    *,
    loc=None,
    ip=None,
  ):
    self.params = params
    self._block_coordinate = block_coordinate
    self._is_first_block = True
    self._loc = loc
    self._ip = ip

  @staticmethod
  def to_underlying_arguments(
    args: TileSchedulerArguments,
    *,
    loc=None,
    ip=None,
  ) -> Params:
    return SingleTileScheduler.Params.create(args, loc=loc, ip=ip)

  @staticmethod
  def create(params: Params, *, loc=None, ip=None) -> "SingleTileScheduler":
    return SingleTileScheduler(
      params,
      cute.arch.block_idx(),
      loc=loc,
      ip=ip,
    )

  @staticmethod
  def get_grid_shape(
    params: Params,
    *,
    loc=None,
    ip=None,
  ) -> Tuple[Int32, Int32, Int32]:
    del loc, ip
    return (params.num_block, params.num_head, params.num_batch)

  def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
    del loc, ip
    block, head, batch = self._block_coordinate
    return WorkTileInfo(
      (block, head, batch, Int32(0)),
      self._is_first_block,
    )

  def initial_work_tile_info(self, *, loc=None, ip=None) -> WorkTileInfo:
    return self.get_current_work(loc=loc, ip=ip)

  def prefetch_next_work(self, *, loc=None, ip=None) -> None:
    del loc, ip

  def advance_to_next_work(self, *, loc=None, ip=None) -> WorkTileInfo:
    self._is_first_block = False
    return self.get_current_work(loc=loc, ip=ip)

  def producer_tail(self, *, loc=None, ip=None) -> None:
    del loc, ip

  def __extract_mlir_values__(self):
    values, self._values_pos = [], []
    for item in (self.params, self._block_coordinate):
      item_values = cutlass.extract_mlir_values(item)
      values.extend(item_values)
      self._values_pos.append(len(item_values))
    return values

  def __new_from_mlir_values__(self, values):
    items = []
    for item, count in zip(
      (self.params, self._block_coordinate),
      self._values_pos,
    ):
      items.append(cutlass.new_from_mlir_values(item, values[:count]))
      values = values[count:]
    return SingleTileScheduler(*items, loc=self._loc, ip=self._ip)


__all__ = ["SingleTileScheduler", "TileSchedulerArguments", "WorkTileInfo"]
