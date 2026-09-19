# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Native wide-head FlashAttention for block-diffusion models.

The standard FA3 kernels cover head dimensions through 256. Gemma 4 global
attention uses dimension 512, so Hopper uses a packaged BDLM-masked Split-D
FlashAttention specialization. Other device generations retain the compiled
block-sparse compatibility path. Both preserve the output/LSE contract used by
distributed online-softmax merging.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import importlib
from typing import Any

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

try:
    from torch.nn.attention.flex_attention import AuxRequest
except ImportError:  # Torch < 2.10 remains usable for non-wide FA3 models.
    AuxRequest = None  # type: ignore[assignment,misc]


WIDE_HEAD_DIM = 512


@dataclass(frozen=True)
class WideHeadKernelMetadata:
    package: str
    version: str
    backend: str
    torch_version: str
    torch_cuda_version: str | None
    device_capability: tuple[int, int] | None
    supported_head_dims: tuple[int, int]
    required_ops: tuple[str, ...]

    def to_log_dict(self) -> dict[str, object]:
        return {
            "package": self.package,
            "version": self.version,
            "backend": self.backend,
            "torch_version": self.torch_version,
            "torch_cuda_version": self.torch_cuda_version,
            "device_capability": (
                list(self.device_capability)
                if self.device_capability is not None
                else None
            ),
            "supported_head_dims": list(self.supported_head_dims),
            "required_ops": list(self.required_ops),
        }


def is_wide_head_dim(head_dim: int) -> bool:
    return int(head_dim) == WIDE_HEAD_DIM


@lru_cache(maxsize=1)
def _require_bdlm_splitd() -> object:
    module = importlib.import_module("flash_attn_3.bdlm_splitd")
    from dllm_parallel.core.kernels.runtime import kernel_runtime_policy

    configure = getattr(module, "configure_splitd_runtime", None)
    if not callable(configure):
        raise RuntimeError(
            "packaged flash_attn_3 is missing its Split-D runtime policy API"
        )
    configure(
        allow_runtime_jit=bool(kernel_runtime_policy().allow_runtime_jit)
    )
    for name in (
        "bdlm_splitd_attention",
        "bdlm_splitd_backward",
        "bdlm_splitd_forward",
        "splitd_full_attention",
        "splitd_full_backward",
        "splitd_full_forward",
        "splitd_backward_dkdv",
        "splitd_backward_dq",
        "splitd_interval_backward",
        "splitd_interval_backward_prepare",
        "splitd_interval_forward",
        "splitd_tile_shape",
        "verify_splitd_artifacts",
    ):
        if not callable(getattr(module, name, None)):
            raise RuntimeError(f"packaged flash_attn_3 is missing {name}")
    module.verify_splitd_artifacts()
    return module


@lru_cache(maxsize=None)
def _device_supports_bdlm_splitd(
    device_type: str,
    device_index: int | None,
) -> bool:
    if device_type != "cuda":
        return False
    return torch.cuda.get_device_capability(
        torch.device(device_type, device_index)
    )[0] == 9


def _uses_native_bdlm_splitd(tensor: torch.Tensor) -> bool:
    return (
        int(tensor.shape[-1]) == 512
        and tensor.is_cuda
        and _device_supports_bdlm_splitd(tensor.device.type, tensor.device.index)
    )


def uses_native_wide_attention(tensor: torch.Tensor) -> bool:
    """Return whether packaged D=512 Hopper kernels own this tensor."""

    return _uses_native_bdlm_splitd(tensor)


def verify_wide_head_attention_kernels() -> WideHeadKernelMetadata:
    """Verify the installed Split-D package and its training operator ABI."""

    capability = (
        tuple(int(value) for value in torch.cuda.get_device_capability())
        if torch.cuda.is_available()
        else None
    )
    if capability is not None and capability < (8, 0):
        raise RuntimeError(
            "native wide-head attention requires compute capability >= 8.0; "
            f"got {capability[0]}.{capability[1]}"
        )
    if capability is not None and capability[0] == 9:
        _require_bdlm_splitd()
    elif AuxRequest is None:
        raise RuntimeError("wide-head FlexAttention requires torch>=2.10")
    package = importlib.import_module("flash_attn_3")
    return WideHeadKernelMetadata(
        package="flash_attn_3",
        version=str(package.__version__),
        backend=(
            "bdlm_flash_attn_split_d"
            if capability is not None and capability[0] == 9
            else "block_sparse_flex"
        ),
        torch_version=str(torch.__version__),
        torch_cuda_version=(
            str(torch.version.cuda) if torch.version.cuda is not None else None
        ),
        device_capability=capability,
        supported_head_dims=(WIDE_HEAD_DIM, WIDE_HEAD_DIM),
        required_ops=(
            "splitd_full_forward",
            "splitd_full_backward",
            "splitd_interval_forward",
            "splitd_interval_backward",
        ),
    )


def _validate_inputs(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> None:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("wide-head attention expects BSHD query/key/value tensors")
    if key.shape != value.shape:
        raise ValueError("wide-head key and value shapes must match")
    if query.shape[0] != key.shape[0] or query.shape[-1] != key.shape[-1]:
        raise ValueError("wide-head query/key batch and head dimensions must match")
    if key.device != query.device or value.device != query.device:
        raise RuntimeError("wide-head query/key/value devices must match")
    if query.shape[2] % key.shape[2] != 0:
        raise ValueError("wide-head query heads must be divisible by key/value heads")
    if not query.is_cuda or not key.is_cuda or not value.is_cuda:
        raise RuntimeError("wide-head Split-D attention requires CUDA tensors")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("wide-head Split-D attention requires fp16 or bf16")
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise RuntimeError("wide-head query/key/value dtypes must match")
    if not is_wide_head_dim(query.shape[-1]):
        raise RuntimeError(
            f"Split-D head_dim must be {WIDE_HEAD_DIM}, got {query.shape[-1]}"
        )


def _raw_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_inputs(query, key, value)
    if _uses_native_bdlm_splitd(query):
        return _require_bdlm_splitd().splitd_full_forward(
            query, key, value, float(scale)
        )
    if AuxRequest is None:
        raise RuntimeError("wide-head FlexAttention requires torch>=2.10")
    output, aux = _compiled_flex_attention()(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        scale=float(scale),
        enable_gqa=bool(query.shape[2] != key.shape[2]),
        return_aux=AuxRequest(lse=True),
        kernel_options=_WIDE_FLEX_KERNEL_OPTIONS,
    )
    if aux.lse is None:
        raise RuntimeError("wide-head FlexAttention did not return LSE statistics")
    return output.transpose(1, 2).contiguous(), aux.lse


def _raw_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    grad_output: torch.Tensor,
    grad_lse: torch.Tensor | None,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate_inputs(query, key, value)
    if _uses_native_bdlm_splitd(query):
        return _require_bdlm_splitd().splitd_full_backward(
            query,
            key,
            value,
            output,
            lse,
            grad_output,
            grad_lse,
            float(scale),
        )
    with torch.enable_grad():
        q = query.detach().requires_grad_(True)
        k = key.detach().requires_grad_(True)
        v = value.detach().requires_grad_(True)
        recomputed_output, recomputed_lse = _raw_forward(q, k, v, float(scale))
        if grad_lse is None:
            grad_lse = torch.zeros_like(recomputed_lse)
        return torch.autograd.grad(
            (recomputed_output, recomputed_lse),
            (q, k, v),
            (grad_output.to(recomputed_output.dtype), grad_lse.float()),
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )


class _WideFullAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: object,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output, lse = _raw_forward(query, key, value, float(scale))
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
        gradients = _raw_backward(
            query, key, value, output, lse, grad_output, grad_lse, ctx.scale
        )
        return *gradients, None


def wide_full_attention_bshd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _uses_native_bdlm_splitd(query):
        return _require_bdlm_splitd().splitd_full_attention(
            query, key, value, float(scale)
        )
    return _WideFullAttention.apply(query, key, value, float(scale))


def wide_full_backward_bshd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    grad_output: torch.Tensor,
    grad_lse: torch.Tensor | None,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _raw_backward(
        query, key, value, output, lse, grad_output, grad_lse, float(scale)
    )


def _bdlm_allowed(
    query_blocks: torch.Tensor,
    query_is_clean: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    *,
    block_size: int,
    key_start: int,
    clean_offset: int,
    full_mask: bool,
) -> torch.Tensor:
    query_block = query_blocks[q_idx]
    query_clean = query_is_clean[q_idx]
    if int(key_start) < 0:
        raise ValueError("wide-head key_start must be a nonnegative logical offset")
    logical_key = int(key_start) + kv_idx
    if not full_mask:
        key_block = logical_key // int(block_size)
        return torch.where(
            query_clean,
            query_block >= key_block,
            query_block > key_block,
        )

    key_clean = logical_key >= int(clean_offset)
    key_position = torch.where(
        key_clean,
        logical_key - int(clean_offset),
        logical_key,
    )
    key_block = key_position // int(block_size)
    active_to_active = (~query_clean) & (~key_clean) & (query_block == key_block)
    active_to_clean = (~query_clean) & key_clean & (query_block > key_block)
    clean_to_clean = query_clean & key_clean & (query_block >= key_block)
    return active_to_active | active_to_clean | clean_to_clean


FlexPlan = Any
_TILE_WORK_ITEM_STRIDE = 2


@dataclass(frozen=True)
class WideIntervalPlan:
    """Cached mask-preserving tile schedule for native Split-D attention."""

    flex_plan: FlexPlan
    forward_query_tile_bounds: torch.Tensor | None
    backward_query_tile_bounds: torch.Tensor | None
    key_tile_bounds: torch.Tensor | None
    forward_query_tile_offsets: torch.Tensor | None
    forward_key_tile_work_items: torch.Tensor | None
    backward_query_tile_offsets: torch.Tensor | None
    backward_key_tile_work_items: torch.Tensor | None
    key_tile_offsets: torch.Tensor | None
    query_tile_work_items: torch.Tensor | None
    uses_sparse_tile_worklist: bool


@dataclass(frozen=True)
class WideIntervalBackwardPhases:
    """Prepared native D=512 backward with independently launchable gradients."""

    backend: Any
    state: Any

    def dkdv(
        self,
        *,
        grad_key: torch.Tensor | None = None,
        grad_value: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.backend.splitd_backward_dkdv(
            self.state,
            grad_key=grad_key,
            grad_value=grad_value,
        )

    def dq(
        self,
        *,
        grad_query: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.backend.splitd_backward_dq(
            self.state,
            grad_query=grad_query,
        )


def _group_range(
    values: torch.Tensor,
    valid: torch.Tensor,
    rows_per_group: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    groups = (int(values.numel()) + int(rows_per_group) - 1) // int(rows_per_group)
    if groups == 0:
        empty_values = values.new_empty((0,))
        empty_valid = valid.new_empty((0,))
        return empty_values, empty_values, empty_valid
    padded_rows = groups * int(rows_per_group)
    if padded_rows != int(values.numel()):
        pad = padded_rows - int(values.numel())
        values = torch.nn.functional.pad(values, (0, pad))
        valid = torch.nn.functional.pad(valid, (0, pad))
    values = values.view(groups, int(rows_per_group))
    valid = valid.view(groups, int(rows_per_group))
    upper = torch.iinfo(values.dtype).max
    lower = torch.iinfo(values.dtype).min
    minimum = torch.where(valid, values, upper).amin(dim=1)
    maximum = torch.where(valid, values, lower).amax(dim=1)
    return minimum, maximum, valid.any(dim=1)


def _interval_live_tiles(
    query_clean_bounds: torch.Tensor,
    local_query_blocks: torch.Tensor,
    query_is_clean: torch.Tensor,
    key_coordinates: torch.Tensor,
    key_is_clean: torch.Tensor,
    *,
    query_rows_per_tile: int,
    key_rows_per_tile: int,
) -> torch.Tensor:
    """Return a support-preserving tile mask for the token-level interval mask."""

    query_tiles = (
        int(query_clean_bounds.shape[0]) + int(query_rows_per_tile) - 1
    ) // int(query_rows_per_tile)
    key_tiles = (
        int(key_coordinates.numel()) + int(key_rows_per_tile) - 1
    ) // int(key_rows_per_tile)
    if query_tiles == 0 or key_tiles == 0:
        return torch.zeros(
            (query_tiles, key_tiles),
            device=query_clean_bounds.device,
            dtype=torch.bool,
        )

    clean_start, _, has_clean_interval = _group_range(
        query_clean_bounds[:, 0],
        query_clean_bounds[:, 1] > query_clean_bounds[:, 0],
        int(query_rows_per_tile),
    )
    _, clean_stop, _ = _group_range(
        query_clean_bounds[:, 1],
        query_clean_bounds[:, 1] > query_clean_bounds[:, 0],
        int(query_rows_per_tile),
    )
    active_query_min, active_query_max, has_active_query = _group_range(
        local_query_blocks,
        ~query_is_clean,
        int(query_rows_per_tile),
    )
    clean_key_min, clean_key_max, has_clean_key = _group_range(
        key_coordinates,
        key_is_clean,
        int(key_rows_per_tile),
    )
    active_key_min, active_key_max, has_active_key = _group_range(
        key_coordinates,
        ~key_is_clean,
        int(key_rows_per_tile),
    )
    clean_tiles = (
        has_clean_interval[:, None]
        & has_clean_key[None, :]
        & (clean_key_max[None, :] >= clean_start[:, None])
        & (clean_key_min[None, :] < clean_stop[:, None])
    )
    active_tiles = (
        has_active_query[:, None]
        & has_active_key[None, :]
        & (active_key_max[None, :] >= active_query_min[:, None])
        & (active_key_min[None, :] <= active_query_max[:, None])
    )
    return clean_tiles | active_tiles


def _tile_csr(
    support: torch.Tensor,
    fully_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode exact tile work as CSR.

    Each work item stores the tile index and a fully-valid flag, allowing the
    kernel to bypass elementwise masking only for tiles proven valid for every
    query-key pair.
    """

    if support.ndim != 2 or support.dtype != torch.bool:
        raise TypeError("tile support must be a two-dimensional boolean tensor")
    if fully_valid.shape != support.shape or fully_valid.dtype != torch.bool:
        raise TypeError("fully-valid tile metadata must match tile support")
    counts = support.sum(dim=1, dtype=torch.int32)
    offsets = torch.empty(
        int(support.shape[0]) + 1,
        device=support.device,
        dtype=torch.int32,
    )
    offsets[0] = 0
    if counts.numel():
        torch.cumsum(counts, dim=0, out=offsets[1:])
    coordinates = support.nonzero(as_tuple=False)
    indices = coordinates[:, 1].to(dtype=torch.int32)
    flags = fully_valid[coordinates[:, 0], coordinates[:, 1]].to(torch.int32)
    work_items = indices.mul(_TILE_WORK_ITEM_STRIDE).add_(flags)
    return offsets.contiguous(), work_items.contiguous()


def _interval_full_tiles(
    query_clean_bounds: torch.Tensor,
    local_query_blocks: torch.Tensor,
    query_is_clean: torch.Tensor,
    key_coordinates: torch.Tensor,
    key_is_clean: torch.Tensor,
    *,
    query_rows_per_tile: int,
    key_rows_per_tile: int,
) -> torch.Tensor:
    """Return tiles for which the interval predicate is identically true."""

    query_rows = int(query_clean_bounds.shape[0])
    key_rows = int(key_coordinates.numel())
    query_tiles = (query_rows + int(query_rows_per_tile) - 1) // int(
        query_rows_per_tile
    )
    key_tiles = (key_rows + int(key_rows_per_tile) - 1) // int(key_rows_per_tile)
    if query_tiles == 0 or key_tiles == 0:
        return torch.zeros(
            (query_tiles, key_tiles),
            device=query_clean_bounds.device,
            dtype=torch.bool,
        )

    def _padded_groups(
        values: torch.Tensor,
        rows: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        padded = ((int(values.numel()) + rows - 1) // rows) * rows
        valid = torch.arange(padded, device=values.device) < int(values.numel())
        if padded != int(values.numel()):
            values = torch.nn.functional.pad(
                values,
                (0, padded - int(values.numel())),
            )
        return values.view(-1, rows), valid.view(-1, rows)

    clean_starts, query_valid = _padded_groups(
        query_clean_bounds[:, 0], int(query_rows_per_tile)
    )
    clean_stops, _ = _padded_groups(
        query_clean_bounds[:, 1], int(query_rows_per_tile)
    )
    query_blocks, _ = _padded_groups(
        local_query_blocks, int(query_rows_per_tile)
    )
    clean_queries, _ = _padded_groups(
        query_is_clean, int(query_rows_per_tile)
    )
    key_coords, key_valid = _padded_groups(
        key_coordinates, int(key_rows_per_tile)
    )
    clean_keys, _ = _padded_groups(key_is_clean, int(key_rows_per_tile))

    integer_max = torch.iinfo(query_clean_bounds.dtype).max
    integer_min = torch.iinfo(query_clean_bounds.dtype).min
    query_start = torch.where(query_valid, clean_starts, integer_min).amax(dim=1)
    query_stop = torch.where(query_valid, clean_stops, integer_max).amin(dim=1)
    query_block_min = torch.where(query_valid, query_blocks, integer_max).amin(
        dim=1
    )
    query_block_max = torch.where(query_valid, query_blocks, integer_min).amax(
        dim=1
    )
    all_queries_active = (~clean_queries | ~query_valid).all(dim=1)

    clean_key_valid = key_valid & clean_keys
    active_key_valid = key_valid & ~clean_keys
    has_clean_key = clean_key_valid.any(dim=1)
    has_active_key = active_key_valid.any(dim=1)
    clean_key_min = torch.where(clean_key_valid, key_coords, integer_max).amin(dim=1)
    clean_key_max = torch.where(clean_key_valid, key_coords, integer_min).amax(dim=1)
    active_key_min = torch.where(active_key_valid, key_coords, integer_max).amin(
        dim=1
    )
    active_key_max = torch.where(active_key_valid, key_coords, integer_min).amax(
        dim=1
    )

    clean_valid = (
        (~has_clean_key[None, :])
        | (
            (clean_key_min[None, :] >= query_start[:, None])
            & (clean_key_max[None, :] < query_stop[:, None])
        )
    )
    active_valid = (
        (~has_active_key[None, :])
        | (
            all_queries_active[:, None]
            & (query_block_min == query_block_max)[:, None]
            & (active_key_min == active_key_max)[None, :]
            & (active_key_min[None, :] == query_block_min[:, None])
        )
    )
    complete_query_tile = query_valid.all(dim=1)
    complete_key_tile = key_valid.all(dim=1)
    return (
        clean_valid
        & active_valid
        & complete_query_tile[:, None]
        & complete_key_tile[None, :]
    )


def _interval_tile_worklists(
    query_clean_bounds: torch.Tensor,
    local_query_blocks: torch.Tensor,
    query_is_clean: torch.Tensor,
    key_coordinates: torch.Tensor,
    key_is_clean: torch.Tensor,
    *,
    query_rows_per_tile: int,
    key_rows_per_tile: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Derive query-major and key-major native candidate-tile worklists."""

    live_tiles = _interval_live_tiles(
        query_clean_bounds,
        local_query_blocks,
        query_is_clean,
        key_coordinates,
        key_is_clean,
        query_rows_per_tile=int(query_rows_per_tile),
        key_rows_per_tile=int(key_rows_per_tile),
    )
    fully_valid = _interval_full_tiles(
        query_clean_bounds,
        local_query_blocks,
        query_is_clean,
        key_coordinates,
        key_is_clean,
        query_rows_per_tile=int(query_rows_per_tile),
        key_rows_per_tile=int(key_rows_per_tile),
    )
    query_offsets, key_indices = _tile_csr(live_tiles, fully_valid)
    key_offsets, query_indices = _tile_csr(
        live_tiles.transpose(0, 1).contiguous(),
        fully_valid.transpose(0, 1).contiguous(),
    )
    return query_offsets, key_indices, key_offsets, query_indices


def _interval_tile_envelopes(
    query_clean_bounds: torch.Tensor,
    local_query_blocks: torch.Tensor,
    query_is_clean: torch.Tensor,
    key_coordinates: torch.Tensor,
    key_is_clean: torch.Tensor,
    *,
    query_rows_per_tile: int,
    key_rows_per_tile: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive diagnostic tile envelopes from the exact tile support."""

    live_tiles = _interval_live_tiles(
        query_clean_bounds,
        local_query_blocks,
        query_is_clean,
        key_coordinates,
        key_is_clean,
        query_rows_per_tile=int(query_rows_per_tile),
        key_rows_per_tile=int(key_rows_per_tile),
    )
    if int(live_tiles.shape[0]) == 0 or int(live_tiles.shape[1]) == 0:
        return (
            torch.zeros(
                (int(live_tiles.shape[0]), 2),
                device=live_tiles.device,
                dtype=torch.int32,
            ),
            torch.zeros(
                (int(live_tiles.shape[1]), 2),
                device=live_tiles.device,
                dtype=torch.int32,
            ),
        )
    key_tiles = int(live_tiles.shape[1])
    key_indices = torch.arange(key_tiles, device=live_tiles.device)
    query_first = torch.where(live_tiles, key_indices[None, :], key_tiles).amin(dim=1)
    query_last = torch.where(live_tiles, key_indices[None, :], -1).amax(dim=1) + 1
    query_has_work = live_tiles.any(dim=1)
    query_bounds = torch.stack(
        (
            torch.where(query_has_work, query_first, 0),
            torch.where(query_has_work, query_last, 0),
        ),
        dim=1,
    )

    query_tiles = int(live_tiles.shape[0])
    query_indices = torch.arange(query_tiles, device=live_tiles.device)
    key_first = torch.where(live_tiles, query_indices[:, None], query_tiles).amin(dim=0)
    key_last = torch.where(live_tiles, query_indices[:, None], -1).amax(dim=0) + 1
    key_has_work = live_tiles.any(dim=0)
    key_bounds = torch.stack(
        (
            torch.where(key_has_work, key_first, 0),
            torch.where(key_has_work, key_last, 0),
        ),
        dim=1,
    )
    return (
        query_bounds.to(dtype=torch.int32).contiguous(),
        key_bounds.to(dtype=torch.int32).contiguous(),
    )


@lru_cache(maxsize=1)
def _compiled_create_block_mask() -> object:
    """Build FlexAttention metadata without a quadratic token-mask temporary."""

    # Dynamic singleton block grids can miscompile reverse-mask reductions.
    return torch.compile(create_block_mask, dynamic=False, fullgraph=True)


# D=512 retains twice the query/output state of PyTorch's D>256 default tile.
# A 32x32 forward tile and PyTorch's 16x16 backward tile keep the complete
# native attention working set below Hopper's per-CTA shared-memory limit.
_WIDE_FLEX_KERNEL_OPTIONS = {
    "BACKEND": "TRITON",
    "fwd_BLOCK_M": 32,
    "fwd_BLOCK_N": 32,
    "fwd_num_stages": 3,
    "fwd_num_warps": 4,
    "bwd_BLOCK_M1": 16,
    "bwd_BLOCK_N1": 16,
    "bwd_BLOCK_M2": 16,
    "bwd_BLOCK_N2": 16,
    "bwd_num_stages": 1,
    "bwd_num_warps": 4,
}


def _flex_plan(
    query_blocks: torch.Tensor,
    query_is_clean: torch.Tensor,
    block_size: int,
    key_start: int,
    query_len: int,
    key_len: int,
    clean_offset: int,
    full_mask: bool,
) -> FlexPlan:
    def mask_mod(
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        del b, h
        return _bdlm_allowed(
            query_blocks,
            query_is_clean,
            q_idx,
            kv_idx,
            block_size=int(block_size),
            key_start=int(key_start),
            clean_offset=int(clean_offset),
            full_mask=bool(full_mask),
        )

    block_mask = _compiled_create_block_mask()(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=int(query_len),
        KV_LEN=int(key_len),
        device=query_blocks.device,
    )
    return block_mask


@lru_cache(maxsize=1)
def _compiled_flex_attention() -> object:
    return torch.compile(flex_attention, dynamic=False)


def _masked_forward(
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
    plan: FlexPlan | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_inputs(query, key, value)
    if AuxRequest is None:
        raise RuntimeError("native masked wide-head attention requires torch>=2.10")
    if plan is None:
        plan = _flex_plan(
            query_blocks,
            query_is_clean,
            block_size,
            key_start,
            query.shape[1],
            key.shape[1],
            clean_offset,
            full_mask,
        )
    block_mask = plan
    output, aux = _compiled_flex_attention()(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        block_mask=block_mask,
        scale=float(scale),
        enable_gqa=bool(query.shape[2] != key.shape[2]),
        return_aux=AuxRequest(lse=True),
        kernel_options=_WIDE_FLEX_KERNEL_OPTIONS,
    )
    if aux.lse is None:
        raise RuntimeError("wide-head FlexAttention did not return LSE statistics")
    lse = aux.lse
    return output.transpose(1, 2).contiguous(), lse


def _masked_backward(
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
    clean_offset: int,
    full_mask: bool,
    plan: FlexPlan | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if grad_lse is None:
        grad_lse = torch.zeros_like(lse)
    if plan is None:
        plan = _flex_plan(
            query_blocks,
            query_is_clean,
            block_size,
            key_start,
            query.shape[1],
            key.shape[1],
            clean_offset,
            full_mask,
        )
    with torch.enable_grad():
        q = query.detach().requires_grad_(True)
        k = key.detach().requires_grad_(True)
        v = value.detach().requires_grad_(True)
        recomputed_output, recomputed_lse = _masked_forward(
            q,
            k,
            v,
            query_blocks,
            query_is_clean,
            int(block_size),
            int(key_start),
            float(scale),
            int(clean_offset),
            full_mask=bool(full_mask),
            plan=plan,
        )
        return torch.autograd.grad(
            (recomputed_output, recomputed_lse),
            (q, k, v),
            (grad_output.to(recomputed_output.dtype), grad_lse.float()),
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )


def _metadata_flex_plan(
    query_blocks: torch.Tensor,
    local_query_blocks: torch.Tensor,
    query_is_clean: torch.Tensor,
    key_coordinates: torch.Tensor,
    key_is_clean: torch.Tensor,
    *,
    query_len: int,
    key_len: int,
) -> FlexPlan:
    """Build a wide-head mask for arbitrary packed active/clean K/V order."""

    def mask_mod(b, h, q_idx, kv_idx):
        del b, h
        query_block = query_blocks[q_idx]
        local_query_block = local_query_blocks[q_idx]
        clean_query = query_is_clean[q_idx]
        key_block = key_coordinates[kv_idx]
        clean_key = key_is_clean[kv_idx]
        active_to_active = (
            (~clean_query)
            & (~clean_key)
            & (local_query_block == key_block)
        )
        active_to_clean = (~clean_query) & clean_key & (query_block > key_block)
        clean_to_clean = clean_query & clean_key & (query_block >= key_block)
        return active_to_active | active_to_clean | clean_to_clean

    return _compiled_create_block_mask()(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=int(query_len),
        KV_LEN=int(key_len),
        device=query_blocks.device,
    )


def _interval_flex_plan(
    query_clean_bounds: torch.Tensor,
    local_query_blocks: torch.Tensor,
    query_is_clean: torch.Tensor,
    key_coordinates: torch.Tensor,
    key_is_clean: torch.Tensor,
    *,
    query_len: int,
    key_len: int,
) -> FlexPlan:
    """Build exact token-interval metadata for packed clean/active attention."""

    query_clean_starts = query_clean_bounds[:, 0]
    query_clean_stops = query_clean_bounds[:, 1]

    def mask_mod(b, h, q_idx, kv_idx):
        del b, h
        clean_start = query_clean_starts[q_idx]
        clean_stop = query_clean_stops[q_idx]
        local_query_block = local_query_blocks[q_idx]
        clean_query = query_is_clean[q_idx]
        key_coordinate = key_coordinates[kv_idx]
        clean_key = key_is_clean[kv_idx]
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
        return active_to_active | clean_context

    return _compiled_create_block_mask()(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=int(query_len),
        KV_LEN=int(key_len),
        device=query_clean_bounds.device,
    )


def wide_bdlm_metadata_plan(
    query_blocks: torch.Tensor,
    local_query_blocks: torch.Tensor,
    query_is_clean: torch.Tensor,
    key_blocks: torch.Tensor,
    key_is_clean: torch.Tensor,
    *,
    query_len: int,
    key_len: int,
) -> FlexPlan:
    """Build reusable BlockMask metadata for a packed wide-head shard."""

    return _metadata_flex_plan(
        query_blocks,
        local_query_blocks,
        query_is_clean,
        key_blocks,
        key_is_clean,
        query_len=int(query_len),
        key_len=int(key_len),
    )


def wide_bdlm_interval_plan(
    query_clean_bounds: torch.Tensor,
    local_query_blocks: torch.Tensor,
    query_is_clean: torch.Tensor,
    key_coordinates: torch.Tensor,
    key_is_clean: torch.Tensor,
    *,
    query_len: int,
    key_len: int,
    query_heads: int | None = None,
    key_heads: int | None = None,
    sparse_tile_worklist: bool = False,
) -> FlexPlan | WideIntervalPlan:
    """Build reusable exact interval and native tile metadata."""

    if query_heads is None and key_heads is None:
        return _interval_flex_plan(
            query_clean_bounds,
            local_query_blocks,
            query_is_clean,
            key_coordinates,
            key_is_clean,
            query_len=int(query_len),
            key_len=int(key_len),
        )
    if query_heads is None or key_heads is None:
        raise ValueError("wide-head query/key head counts must be supplied together")
    if int(query_heads) % int(key_heads):
        raise ValueError("wide-head query heads must be divisible by key heads")
    splitd = _require_bdlm_splitd()
    tile_m, tile_n = splitd.splitd_tile_shape()
    heads_per_kv_head = int(query_heads) // int(key_heads)
    if int(tile_m) % heads_per_kv_head:
        raise ValueError("Split-D query tile must be divisible by the GQA ratio")
    if sparse_tile_worklist:
        forward_query_bounds = backward_query_bounds = key_bounds = None
        (
            forward_query_offsets,
            forward_key_work_items,
            _,
            _,
        ) = _interval_tile_worklists(
            query_clean_bounds,
            local_query_blocks,
            query_is_clean,
            key_coordinates,
            key_is_clean,
            query_rows_per_tile=int(tile_m) // heads_per_kv_head,
            key_rows_per_tile=int(tile_n),
        )
        (
            backward_query_offsets,
            backward_key_work_items,
            key_offsets,
            query_work_items,
        ) = _interval_tile_worklists(
            query_clean_bounds,
            local_query_blocks,
            query_is_clean,
            key_coordinates,
            key_is_clean,
            query_rows_per_tile=int(tile_m),
            key_rows_per_tile=int(tile_n),
        )
    else:
        forward_query_offsets = forward_key_work_items = None
        backward_query_offsets = backward_key_work_items = None
        key_offsets = query_work_items = None
        forward_query_bounds, _ = _interval_tile_envelopes(
            query_clean_bounds,
            local_query_blocks,
            query_is_clean,
            key_coordinates,
            key_is_clean,
            query_rows_per_tile=int(tile_m) // heads_per_kv_head,
            key_rows_per_tile=int(tile_n),
        )
        backward_query_bounds, key_bounds = _interval_tile_envelopes(
            query_clean_bounds,
            local_query_blocks,
            query_is_clean,
            key_coordinates,
            key_is_clean,
            query_rows_per_tile=int(tile_m),
            key_rows_per_tile=int(tile_n),
        )
    return WideIntervalPlan(
        flex_plan=None,
        forward_query_tile_bounds=forward_query_bounds,
        backward_query_tile_bounds=backward_query_bounds,
        key_tile_bounds=key_bounds,
        forward_query_tile_offsets=forward_query_offsets,
        forward_key_tile_work_items=forward_key_work_items,
        backward_query_tile_offsets=backward_query_offsets,
        backward_key_tile_work_items=backward_key_work_items,
        key_tile_offsets=key_offsets,
        query_tile_work_items=query_work_items,
        uses_sparse_tile_worklist=bool(sparse_tile_worklist),
    )


def wide_bdlm_interval_forward_bhsd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_clean_bounds: torch.Tensor,
    local_query_blocks: torch.Tensor,
    query_is_clean: torch.Tensor,
    key_coordinates: torch.Tensor,
    key_is_clean: torch.Tensor,
    scale: float,
    plan: FlexPlan | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate exact packed D=512 attention from BHSD tensors."""

    if _uses_native_bdlm_splitd(query):
        if not isinstance(plan, WideIntervalPlan):
            raise RuntimeError("native Split-D interval attention requires a tile plan")
        if plan.uses_sparse_tile_worklist:
            if (
                plan.forward_query_tile_offsets is None
                or plan.forward_key_tile_work_items is None
            ):
                raise RuntimeError("native Split-D forward worklist is incomplete")
            schedule = {
                "query_tile_offsets": plan.forward_query_tile_offsets,
                "key_tile_work_items": plan.forward_key_tile_work_items,
            }
        else:
            if plan.forward_query_tile_bounds is None or plan.key_tile_bounds is None:
                raise RuntimeError("native Split-D forward tile bounds are incomplete")
            schedule = {
                "query_tile_bounds": plan.forward_query_tile_bounds,
                "key_tile_bounds": plan.key_tile_bounds,
            }
        output, lse = _require_bdlm_splitd().splitd_interval_forward(
            query.transpose(1, 2).contiguous(),
            key.transpose(1, 2).contiguous(),
            value.transpose(1, 2).contiguous(),
            query_clean_bounds,
            local_query_blocks,
            query_is_clean,
            key_coordinates,
            key_is_clean,
            float(scale),
            **schedule,
        )
        return output.transpose(1, 2), lse
    if plan is None:
        plan = _interval_flex_plan(
            query_clean_bounds,
            local_query_blocks,
            query_is_clean,
            key_coordinates,
            key_is_clean,
            query_len=int(query.shape[2]),
            key_len=int(key.shape[2]),
        )
    elif isinstance(plan, WideIntervalPlan):
        plan = plan.flex_plan
    return _metadata_forward_bhsd(
        query,
        key,
        value,
        query_clean_bounds,
        local_query_blocks,
        query_is_clean,
        key_coordinates,
        key_is_clean,
        float(scale),
        plan,
    )


def wide_bdlm_interval_backward_from_state_bhsd(
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
    plan: WideIntervalPlan,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiate a packed D=512 shard from merged output and LSE state."""

    if not _uses_native_bdlm_splitd(query):
        raise RuntimeError(
            "merged-state D=512 interval backward requires packaged Split-D"
        )
    if plan.uses_sparse_tile_worklist:
        if any(
            value is None
            for value in (
                plan.backward_query_tile_offsets,
                plan.backward_key_tile_work_items,
                plan.key_tile_offsets,
                plan.query_tile_work_items,
            )
        ):
            raise RuntimeError("native Split-D backward worklists are incomplete")
        schedule = {
            "query_tile_offsets": plan.backward_query_tile_offsets,
            "key_tile_work_items": plan.backward_key_tile_work_items,
            "key_tile_offsets": plan.key_tile_offsets,
            "query_tile_work_items": plan.query_tile_work_items,
        }
    else:
        if plan.backward_query_tile_bounds is None or plan.key_tile_bounds is None:
            raise RuntimeError("native Split-D backward tile bounds are incomplete")
        schedule = {
            "query_tile_bounds": plan.backward_query_tile_bounds,
            "key_tile_bounds": plan.key_tile_bounds,
        }
    gradients = _require_bdlm_splitd().splitd_interval_backward(
        query.transpose(1, 2).contiguous(),
        key.transpose(1, 2).contiguous(),
        value.transpose(1, 2).contiguous(),
        output.transpose(1, 2).contiguous(),
        lse,
        grad_output.transpose(1, 2).contiguous(),
        grad_lse,
        query_clean_bounds,
        local_query_blocks,
        query_is_clean,
        key_coordinates,
        key_is_clean,
        float(scale),
        **schedule,
    )
    return tuple(gradient.transpose(1, 2) for gradient in gradients)


def prepare_wide_bdlm_interval_backward_bshd(
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
    plan: WideIntervalPlan,
) -> WideIntervalBackwardPhases:
    """Prepare one native D=512 backward for split dK/dV and dQ launches."""

    if not _uses_native_bdlm_splitd(query):
        raise RuntimeError("phased D=512 backward requires packaged Split-D")
    if plan.uses_sparse_tile_worklist:
        if any(
            tensor is None
            for tensor in (
                plan.backward_query_tile_offsets,
                plan.backward_key_tile_work_items,
                plan.key_tile_offsets,
                plan.query_tile_work_items,
            )
        ):
            raise RuntimeError("native Split-D backward worklists are incomplete")
        schedule = {
            "query_tile_offsets": plan.backward_query_tile_offsets,
            "key_tile_work_items": plan.backward_key_tile_work_items,
            "key_tile_offsets": plan.key_tile_offsets,
            "query_tile_work_items": plan.query_tile_work_items,
        }
    else:
        if plan.backward_query_tile_bounds is None or plan.key_tile_bounds is None:
            raise RuntimeError("native Split-D backward tile bounds are incomplete")
        schedule = {
            "query_tile_bounds": plan.backward_query_tile_bounds,
            "key_tile_bounds": plan.key_tile_bounds,
        }
    backend = _require_bdlm_splitd()
    state = backend.splitd_interval_backward_prepare(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        output.contiguous(),
        lse,
        grad_output.contiguous(),
        grad_lse,
        query_clean_bounds,
        local_query_blocks,
        query_is_clean,
        key_coordinates,
        key_is_clean,
        float(scale),
        **schedule,
    )
    return WideIntervalBackwardPhases(backend=backend, state=state)


def _metadata_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_blocks: torch.Tensor,
    local_query_blocks: torch.Tensor,
    query_is_clean: torch.Tensor,
    key_blocks: torch.Tensor,
    key_is_clean: torch.Tensor,
    scale: float,
    plan: FlexPlan | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    output, lse = _metadata_forward_bhsd(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        query_blocks,
        local_query_blocks,
        query_is_clean,
        key_blocks,
        key_is_clean,
        float(scale),
        plan,
    )
    return output.transpose(1, 2).contiguous(), lse


def _metadata_forward_bhsd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_blocks: torch.Tensor,
    local_query_blocks: torch.Tensor,
    query_is_clean: torch.Tensor,
    key_blocks: torch.Tensor,
    key_is_clean: torch.Tensor,
    scale: float,
    plan: FlexPlan | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate metadata attention without copying an already-BHSD shard."""

    _validate_inputs(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
    )
    if AuxRequest is None:
        raise RuntimeError("wide-head metadata attention requires torch>=2.10")
    if plan is None:
        plan = _metadata_flex_plan(
            query_blocks,
            local_query_blocks,
            query_is_clean,
            key_blocks,
            key_is_clean,
            query_len=int(query.shape[2]),
            key_len=int(key.shape[2]),
        )
    output, aux = _compiled_flex_attention()(
        query,
        key,
        value,
        block_mask=plan,
        scale=float(scale),
        enable_gqa=bool(query.shape[1] != key.shape[1]),
        return_aux=AuxRequest(lse=True),
        kernel_options=_WIDE_FLEX_KERNEL_OPTIONS,
    )
    if aux.lse is None:
        raise RuntimeError("wide-head metadata attention did not return LSE")
    return output, aux.lse


def _metadata_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    grad_output: torch.Tensor,
    grad_lse: torch.Tensor | None,
    query_blocks: torch.Tensor,
    local_query_blocks: torch.Tensor,
    query_is_clean: torch.Tensor,
    key_blocks: torch.Tensor,
    key_is_clean: torch.Tensor,
    scale: float,
    plan: FlexPlan | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gradients = _metadata_backward_bhsd(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        grad_output.transpose(1, 2),
        grad_lse,
        query_blocks,
        local_query_blocks,
        query_is_clean,
        key_blocks,
        key_is_clean,
        float(scale),
        plan,
    )
    return tuple(gradient.transpose(1, 2).contiguous() for gradient in gradients)


def _metadata_backward_bhsd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    grad_output: torch.Tensor,
    grad_lse: torch.Tensor | None,
    query_blocks: torch.Tensor,
    local_query_blocks: torch.Tensor,
    query_is_clean: torch.Tensor,
    key_blocks: torch.Tensor,
    key_is_clean: torch.Tensor,
    scale: float,
    plan: FlexPlan | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiate metadata attention without materializing BSHD copies."""

    if plan is None:
        plan = _metadata_flex_plan(
            query_blocks,
            local_query_blocks,
            query_is_clean,
            key_blocks,
            key_is_clean,
            query_len=int(query.shape[2]),
            key_len=int(key.shape[2]),
        )
    with torch.enable_grad():
        q = query.detach().requires_grad_(True)
        k = key.detach().requires_grad_(True)
        v = value.detach().requires_grad_(True)
        output, lse = _metadata_forward_bhsd(
            q,
            k,
            v,
            query_blocks,
            local_query_blocks,
            query_is_clean,
            key_blocks,
            key_is_clean,
            float(scale),
            plan,
        )
        lse_grad = torch.zeros_like(lse) if grad_lse is None else grad_lse
        return torch.autograd.grad(
            (output, lse),
            (q, k, v),
            (grad_output.to(output.dtype), lse_grad.float()),
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )


class _WideMetadataAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: object,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_blocks: torch.Tensor,
        local_query_blocks: torch.Tensor,
        query_is_clean: torch.Tensor,
        key_blocks: torch.Tensor,
        key_is_clean: torch.Tensor,
        scale: float,
        plan: FlexPlan | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if plan is None:
            plan = _metadata_flex_plan(
                query_blocks,
                local_query_blocks,
                query_is_clean,
                key_blocks,
                key_is_clean,
                query_len=int(query.shape[1]),
                key_len=int(key.shape[1]),
            )
        output, lse = _metadata_forward(
            query,
            key,
            value,
            query_blocks,
            local_query_blocks,
            query_is_clean,
            key_blocks,
            key_is_clean,
            float(scale),
            plan,
        )
        ctx.save_for_backward(
            query,
            key,
            value,
            query_blocks,
            local_query_blocks,
            query_is_clean,
            key_blocks,
            key_is_clean,
        )
        ctx.scale = float(scale)
        ctx.plan = plan
        return output, lse

    @staticmethod
    def backward(
        ctx: object,
        grad_output: torch.Tensor,
        grad_lse: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        (
            query,
            key,
            value,
            query_blocks,
            local_query_blocks,
            query_is_clean,
            key_blocks,
            key_is_clean,
        ) = ctx.saved_tensors
        gradients = _metadata_backward(
            query,
            key,
            value,
            grad_output,
            grad_lse,
            query_blocks,
            local_query_blocks,
            query_is_clean,
            key_blocks,
            key_is_clean,
            ctx.scale,
            ctx.plan,
        )
        return *gradients, None, None, None, None, None, None, None


def wide_bdlm_metadata_attention_bshd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_blocks: torch.Tensor,
    local_query_blocks: torch.Tensor,
    query_is_clean: torch.Tensor,
    key_blocks: torch.Tensor,
    key_is_clean: torch.Tensor,
    scale: float,
    plan: FlexPlan | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate D=512 BDLM attention with explicit packed K/V metadata."""

    return _WideMetadataAttention.apply(
        query,
        key,
        value,
        query_blocks,
        local_query_blocks,
        query_is_clean,
        key_blocks,
        key_is_clean,
        float(scale),
        plan,
    )


def wide_bdlm_metadata_backward_bshd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    grad_output: torch.Tensor,
    grad_lse: torch.Tensor | None,
    query_blocks: torch.Tensor,
    local_query_blocks: torch.Tensor,
    query_is_clean: torch.Tensor,
    key_blocks: torch.Tensor,
    key_is_clean: torch.Tensor,
    scale: float,
    plan: FlexPlan | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiate D=512 BDLM attention with explicit packed K/V metadata."""

    return _metadata_backward(
        query,
        key,
        value,
        grad_output,
        grad_lse,
        query_blocks,
        local_query_blocks,
        query_is_clean,
        key_blocks,
        key_is_clean,
        float(scale),
        plan,
    )


def wide_bdlm_metadata_forward_bhsd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_blocks: torch.Tensor,
    local_query_blocks: torch.Tensor,
    query_is_clean: torch.Tensor,
    key_blocks: torch.Tensor,
    key_is_clean: torch.Tensor,
    scale: float,
    plan: FlexPlan | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate a BHSD metadata shard managed by distributed attention."""

    return _metadata_forward_bhsd(
        query,
        key,
        value,
        query_blocks,
        local_query_blocks,
        query_is_clean,
        key_blocks,
        key_is_clean,
        float(scale),
        plan,
    )


def wide_bdlm_metadata_backward_bhsd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    grad_output: torch.Tensor,
    grad_lse: torch.Tensor | None,
    query_blocks: torch.Tensor,
    local_query_blocks: torch.Tensor,
    query_is_clean: torch.Tensor,
    key_blocks: torch.Tensor,
    key_is_clean: torch.Tensor,
    scale: float,
    plan: FlexPlan | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiate a BHSD metadata shard managed by distributed attention."""

    return _metadata_backward_bhsd(
        query,
        key,
        value,
        grad_output,
        grad_lse,
        query_blocks,
        local_query_blocks,
        query_is_clean,
        key_blocks,
        key_is_clean,
        float(scale),
        plan,
    )


class _WideBDLMAttention(torch.autograd.Function):
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
        plan = _flex_plan(
            query_blocks,
            query_is_clean,
            int(block_size),
            int(key_start),
            int(query.shape[1]),
            int(key.shape[1]),
            int(clean_offset),
            bool(full_mask),
        )
        output, lse = _masked_forward(
            query, key, value, query_blocks, query_is_clean, block_size,
            key_start, scale, clean_offset, full_mask, plan
        )
        ctx.save_for_backward(
            query, key, value, output, lse, query_blocks, query_is_clean
        )
        ctx.block_size = int(block_size)
        ctx.key_start = int(key_start)
        ctx.scale = float(scale)
        ctx.clean_offset = int(clean_offset)
        ctx.full_mask = bool(full_mask)
        ctx.plan = plan
        return output, lse

    @staticmethod
    def backward(
        ctx: object,
        grad_output: torch.Tensor,
        grad_lse: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None, None, None, None, None]:
        query, key, value, output, lse, query_blocks, query_is_clean = ctx.saved_tensors
        gradients = _masked_backward(
            query, key, value, output, lse, grad_output, grad_lse,
            query_blocks, query_is_clean, ctx.block_size, ctx.key_start,
            ctx.scale, ctx.clean_offset, ctx.full_mask, ctx.plan
        )
        return *gradients, None, None, None, None, None, None, None


def wide_bdlm_attention_bshd(
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
    if int(key_start) < 0:
        raise ValueError("wide-head key_start must be a nonnegative logical offset")
    if _uses_native_bdlm_splitd(query):
        return _require_bdlm_splitd().bdlm_splitd_attention(
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
    return _WideBDLMAttention.apply(
        query, key, value, query_blocks, query_is_clean, int(block_size),
        int(key_start), float(scale), int(clean_offset), bool(full_mask)
    )


def wide_bdlm_backward_bshd(
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
    if int(key_start) < 0:
        raise ValueError("wide-head key_start must be a nonnegative logical offset")
    if _uses_native_bdlm_splitd(query):
        return _require_bdlm_splitd().bdlm_splitd_backward(
            query,
            key,
            value,
            output,
            lse,
            grad_output,
            grad_lse,
            query_blocks,
            query_is_clean,
            int(block_size),
            int(key_start),
            float(scale),
            int(clean_offset),
            full_mask=bool(full_mask),
        )
    return _masked_backward(
        query, key, value, output, lse, grad_output, grad_lse,
        query_blocks, query_is_clean, int(block_size), int(key_start),
        float(scale), int(clean_offset), bool(full_mask)
    )


__all__ = [
    "WIDE_HEAD_DIM",
    "WideHeadKernelMetadata",
    "WideIntervalBackwardPhases",
    "is_wide_head_dim",
    "prepare_wide_bdlm_interval_backward_bshd",
    "uses_native_wide_attention",
    "wide_bdlm_interval_backward_from_state_bhsd",
    "wide_bdlm_interval_forward_bhsd",
    "wide_bdlm_metadata_attention_bshd",
    "wide_bdlm_metadata_backward_bhsd",
    "wide_bdlm_metadata_backward_bshd",
    "wide_bdlm_metadata_forward_bhsd",
    "wide_bdlm_interval_plan",
    "wide_bdlm_metadata_plan",
    "verify_wide_head_attention_kernels",
    "wide_bdlm_attention_bshd",
    "wide_bdlm_backward_bshd",
    "wide_full_attention_bshd",
    "wide_full_backward_bshd",
]
