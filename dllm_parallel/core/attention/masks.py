# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Block-diffusion attention mask specs for CP/BP kernels."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class BlockDenoisingLocalActiveMask:
    """Kernel-side mask for active-query to rank-local noisy active K/V."""

    query_blocks: torch.Tensor
    query_is_clean: torch.Tensor
    active_blocks: torch.Tensor
    block_size: int
    causal: bool = False
    query_token_offsets: torch.Tensor | None = None
    backward_query_chunk_size: int = 0
    debug_nonfinite_attention: bool = False
    flex_cache: dict[tuple[Any, ...], Any] = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )

    def detach(self) -> "BlockDenoisingLocalActiveMask":
        return BlockDenoisingLocalActiveMask(
            query_blocks=self.query_blocks.detach(),
            query_is_clean=self.query_is_clean.detach(),
            active_blocks=self.active_blocks.detach(),
            block_size=int(self.block_size),
            causal=bool(self.causal),
            query_token_offsets=(
                None
                if self.query_token_offsets is None
                else self.query_token_offsets.detach()
            ),
            backward_query_chunk_size=int(self.backward_query_chunk_size),
            debug_nonfinite_attention=bool(self.debug_nonfinite_attention),
            flex_cache=self.flex_cache,
        )


@dataclass(frozen=True)
class BlockDenoisingGlobalCleanMask:
    """Kernel-side mask for packed queries attending to logical clean K/V."""

    query_blocks: torch.Tensor
    query_is_clean: torch.Tensor
    block_size: int
    clean_context_window: int | None = None
    query_clean_bounds: torch.Tensor | None = None
    clean_key_positions: torch.Tensor | None = None
    clean_key_blocks: torch.Tensor | None = None
    first_key_length: torch.Tensor | None = None
    second_key_start: torch.Tensor | None = None
    backward_query_chunk_size: int = 0
    debug_nonfinite_attention: bool = False
    allow_native_wide_attention: bool = True
    flex_cache: dict[tuple[Any, ...], Any] = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if int(self.block_size) <= 0:
            raise ValueError("block_size must be positive")
        if self.clean_context_window is not None and int(self.clean_context_window) <= 0:
            raise ValueError("clean_context_window must be positive")
        if self.query_blocks.shape != self.query_is_clean.shape:
            raise ValueError("query_blocks and query_is_clean must have identical shapes")
        if self.query_clean_bounds is not None:
            if self.query_clean_bounds.shape != (*self.query_blocks.shape, 2):
                raise ValueError("query_clean_bounds must have shape [query_rows, 2]")
            if self.clean_key_positions is None:
                raise ValueError(
                    "exact clean-context bounds require clean_key_positions"
                )
        if (
            self.clean_key_positions is not None
            and self.clean_key_blocks is not None
            and self.clean_key_positions.shape != self.clean_key_blocks.shape
        ):
            raise ValueError("clean key position and block metadata must match")

    def detach(self) -> "BlockDenoisingGlobalCleanMask":
        return BlockDenoisingGlobalCleanMask(
            query_blocks=self.query_blocks.detach(),
            query_is_clean=self.query_is_clean.detach(),
            block_size=int(self.block_size),
            clean_context_window=self.clean_context_window,
            query_clean_bounds=(
                None
                if self.query_clean_bounds is None
                else self.query_clean_bounds.detach()
            ),
            clean_key_positions=(
                None
                if self.clean_key_positions is None
                else self.clean_key_positions.detach()
            ),
            clean_key_blocks=(
                None
                if self.clean_key_blocks is None
                else self.clean_key_blocks.detach()
            ),
            first_key_length=(
                None
                if self.first_key_length is None
                else self.first_key_length.detach()
            ),
            second_key_start=(
                None
                if self.second_key_start is None
                else self.second_key_start.detach()
            ),
            backward_query_chunk_size=int(self.backward_query_chunk_size),
            debug_nonfinite_attention=bool(self.debug_nonfinite_attention),
            allow_native_wide_attention=bool(self.allow_native_wide_attention),
            flex_cache=self.flex_cache,
        )


@dataclass(frozen=True)
class BlockDenoisingFullMask:
    """Kernel-side mask for full all-block block-denoising attention."""

    query_blocks: torch.Tensor
    query_is_clean: torch.Tensor
    block_size: int
    clean_offset: int
    query_clean_bounds: torch.Tensor | None = None
    backward_query_chunk_size: int = 0
    debug_nonfinite_attention: bool = False
    flex_cache: dict[tuple[Any, ...], Any] = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if int(self.block_size) <= 0:
            raise ValueError("block_size must be positive")
        if int(self.clean_offset) <= 0:
            raise ValueError("full block-denoising masks require a positive clean_offset")
        if self.query_blocks.shape != self.query_is_clean.shape:
            raise ValueError("query_blocks and query_is_clean must have identical shapes")
        if (
            self.query_clean_bounds is not None
            and self.query_clean_bounds.shape != (*self.query_blocks.shape, 2)
        ):
            raise ValueError("query_clean_bounds must have shape [query_rows, 2]")

    def detach(self) -> "BlockDenoisingFullMask":
        return BlockDenoisingFullMask(
            query_blocks=self.query_blocks.detach(),
            query_is_clean=self.query_is_clean.detach(),
            block_size=int(self.block_size),
            clean_offset=int(self.clean_offset),
            query_clean_bounds=(
                None
                if self.query_clean_bounds is None
                else self.query_clean_bounds.detach()
            ),
            backward_query_chunk_size=int(self.backward_query_chunk_size),
            debug_nonfinite_attention=bool(self.debug_nonfinite_attention),
            flex_cache=self.flex_cache,
        )


@dataclass(frozen=True)
class BlockDenoisingPackedKeyMask:
    """Flex mask for K/V laid out as [local noisy active ; global clean]."""

    query_blocks: torch.Tensor
    local_query_blocks: torch.Tensor
    query_is_clean: torch.Tensor
    active_key_blocks: torch.Tensor
    clean_key_blocks: torch.Tensor
    block_size: int
    query_clean_bounds: torch.Tensor | None = None
    clean_key_positions: torch.Tensor | None = None
    backward_query_chunk_size: int = 0
    debug_nonfinite_attention: bool = False
    flex_cache: dict[tuple[Any, ...], Any] = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if int(self.block_size) <= 0:
            raise ValueError("block_size must be positive")
        if (
            self.query_blocks.shape != self.query_is_clean.shape
            or self.local_query_blocks.shape != self.query_is_clean.shape
        ):
            raise ValueError("packed query metadata must have identical shapes")
        if self.active_key_blocks.ndim != 1 or self.clean_key_blocks.ndim != 1:
            raise ValueError("packed key block metadata must be one-dimensional")
        if self.query_clean_bounds is not None:
            if self.query_clean_bounds.shape != (*self.query_blocks.shape, 2):
                raise ValueError("query_clean_bounds must have shape [query_rows, 2]")
            if self.clean_key_positions is None:
                raise ValueError(
                    "exact clean-context bounds require clean_key_positions"
                )
        if (
            self.clean_key_positions is not None
            and self.clean_key_positions.shape != self.clean_key_blocks.shape
        ):
            raise ValueError("clean key position and block metadata must match")

    def detach(self) -> "BlockDenoisingPackedKeyMask":
        return BlockDenoisingPackedKeyMask(
            query_blocks=self.query_blocks.detach(),
            local_query_blocks=self.local_query_blocks.detach(),
            query_is_clean=self.query_is_clean.detach(),
            active_key_blocks=self.active_key_blocks.detach(),
            clean_key_blocks=self.clean_key_blocks.detach(),
            block_size=int(self.block_size),
            query_clean_bounds=(
                None
                if self.query_clean_bounds is None
                else self.query_clean_bounds.detach()
            ),
            clean_key_positions=(
                None
                if self.clean_key_positions is None
                else self.clean_key_positions.detach()
            ),
            backward_query_chunk_size=int(self.backward_query_chunk_size),
            debug_nonfinite_attention=bool(self.debug_nonfinite_attention),
            flex_cache=self.flex_cache,
        )


@dataclass(frozen=True)
class DFlashGlobalContextMask:
    """Per-anchor target-context intervals for DFlash global K/V shards."""

    context_starts: torch.Tensor
    context_stops: torch.Tensor
    anchor_valid: torch.Tensor
    block_size: int
    global_anchor_count: int = 0
    sliding_window: int | None = None
    backward_query_chunk_size: int = 0
    debug_nonfinite_attention: bool = False

    def __post_init__(self) -> None:
        if int(self.block_size) < 2:
            raise ValueError("DFlash block_size must be at least two")
        if self.context_starts.shape != self.context_stops.shape:
            raise ValueError("DFlash context interval tensors must have identical shapes")
        if self.context_starts.shape != self.anchor_valid.shape:
            raise ValueError("DFlash anchor validity must match context intervals")
        if self.context_starts.ndim != 2:
            raise ValueError("DFlash context intervals must have shape [batch, anchors]")
        if self.context_starts.dtype != torch.int32 or self.context_stops.dtype != torch.int32:
            raise TypeError("DFlash context intervals must use int32")
        if self.anchor_valid.dtype != torch.bool:
            raise TypeError("DFlash anchor validity must use bool")
        if self.sliding_window is not None and int(self.sliding_window) <= 0:
            raise ValueError("DFlash sliding_window must be positive")

    def detach(self) -> "DFlashGlobalContextMask":
        return DFlashGlobalContextMask(
            context_starts=self.context_starts.detach(),
            context_stops=self.context_stops.detach(),
            anchor_valid=self.anchor_valid.detach(),
            block_size=self.block_size,
            global_anchor_count=self.global_anchor_count,
            sliding_window=self.sliding_window,
            backward_query_chunk_size=self.backward_query_chunk_size,
            debug_nonfinite_attention=self.debug_nonfinite_attention,
        )


@dataclass(frozen=True)
class DFlashLocalBlockMask:
    """Own-block visibility for packed DFlash synthetic K/V."""

    anchor_valid: torch.Tensor
    block_size: int
    causal: bool = False
    backward_query_chunk_size: int = 0
    debug_nonfinite_attention: bool = False

    def __post_init__(self) -> None:
        if int(self.block_size) < 2:
            raise ValueError("DFlash block_size must be at least two")
        if self.anchor_valid.ndim != 2:
            raise ValueError("DFlash anchor validity must have shape [batch, anchors]")
        if self.anchor_valid.dtype != torch.bool:
            raise TypeError("DFlash anchor validity must use bool")

    def detach(self) -> "DFlashLocalBlockMask":
        return DFlashLocalBlockMask(
            anchor_valid=self.anchor_valid.detach(),
            block_size=self.block_size,
            causal=self.causal,
            backward_query_chunk_size=self.backward_query_chunk_size,
            debug_nonfinite_attention=self.debug_nonfinite_attention,
        )
