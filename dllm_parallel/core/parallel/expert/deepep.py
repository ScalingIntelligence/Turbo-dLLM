# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""DeepEP v2 token dispatch/combine for expert-parallel training."""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch.distributed import ProcessGroup
from torch.utils._python_dispatch import _disable_current_modes

try:
    import deep_ep
    from deep_ep import Buffer, ElasticBuffer
except ImportError as error:  # pragma: no cover - exercised on production hosts.
    raise ImportError(
        "expert_parallel_size > 1 requires DeepEP v2. Install the qualified "
        "binary bundle or the pinned revision in constraints/deepep-source.txt."
    ) from error


DeepEPTransport = Literal["nvlink", "elastic"]


@dataclass(slots=True)
class _NVLinkDispatchHandle:
    native_handle: Any
    recv_token_index: torch.Tensor | None = None
    route_restore_order: torch.Tensor | None = None
    route_lengths: torch.Tensor | None = None
    num_recv_tokens: int = 0


_buffer: Buffer | ElasticBuffer | None = None
_handle_cache: dict[int, Any] = {}
_handle_counter = 0
_pending_dispatch_events: dict[int, Any] = {}
_pending_combine_event: Any | None = None

_DEEPEP_MIN_VERSION = (2, 0, 0)


@dataclass(frozen=True, slots=True)
class DeepEPPreflight:
    deep_ep_version: str | None
    nccl_distribution: str | None
    nccl_version: str | None


def validate_deepep_install() -> DeepEPPreflight:
    """Validate the production DeepEP package before CUDA-heavy setup."""

    version = _distribution_version("deep_ep")
    if version is not None and _version_tuple(version) < _DEEPEP_MIN_VERSION:
        raise RuntimeError(
            "expert_parallel_size > 1 requires DeepEP v2; "
            f"found deep_ep {version}"
        )
    if not hasattr(deep_ep, "ElasticBuffer") or not hasattr(deep_ep, "Buffer"):
        raise RuntimeError(
            "DeepEP v2 must provide both ElasticBuffer and the NVLink Buffer"
        )

    nccl_name, nccl_version = _nccl_distribution_version()
    return DeepEPPreflight(
        deep_ep_version=version,
        nccl_distribution=nccl_name,
        nccl_version=nccl_version,
    )

def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _nccl_distribution_version() -> tuple[str | None, str | None]:
    for name in ("nvidia-nccl-cu13", "nvidia-nccl-cu12", "nvidia-nccl-cu11"):
        version = _distribution_version(name)
        if version is not None:
            return name, version
    return None, None


def _version_tuple(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for piece in version.replace("+", ".").replace("-", ".").split("."):
        if not piece.isdigit():
            break
        parts.append(int(piece))
    return tuple(parts)


def _next_handle_id() -> torch.Tensor:
    global _handle_counter
    _handle_counter += 1
    return torch.tensor([_handle_counter], dtype=torch.int64, device="cpu")


def _resolve_dispatch_num_sms(buffer: ElasticBuffer, num_experts: int, num_topk: int) -> int:
    """Resolve DeepEP communication SM count through DeepEP's v2 model.

    DeepEP stores the chosen count on the returned handle; combine/backward reuse
    that exact value. If the runtime cannot derive a count, surface the failure
    instead of silently switching to a guessed transport policy.
    """

    num_sms = int(buffer.get_theoretical_num_sms(int(num_experts), int(num_topk)))
    if num_sms <= 0:
        raise RuntimeError("DeepEP returned a non-positive communication SM count")
    device_props = torch.cuda.get_device_properties(torch.cuda.current_device())
    device_sms = int(device_props.multi_processor_count)
    return min(num_sms, device_sms)


_lib = torch.library.Library("dllm_deepep", "DEF")
_lib.define(
    "dispatch(Tensor x, Tensor topk_idx, Tensor topk_weights, "
    "int num_experts, int num_max_tokens_per_rank) "
    "-> (Tensor, Tensor, Tensor, Tensor, Tensor)"
)
_lib.define("combine(Tensor x, Tensor handle_id, bool will_backward) -> Tensor")


@torch.library.impl(_lib, "dispatch", "CUDA")
def _dispatch_impl(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    num_max_tokens_per_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    global _pending_dispatch_events
    buffer = _require_buffer()
    if isinstance(buffer, Buffer):
        layout_start = buffer.capture()
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            layout_event,
        ) = buffer.get_dispatch_layout(
            topk_idx,
            num_experts,
            previous_event=layout_start,
            async_finish=True,
            allocate_on_comm_stream=True,
        )
        (
            recv_x,
            recv_topk_idx,
            recv_scores,
            _recv_counts,
            native_handle,
            event,
        ) = buffer.dispatch(
            x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=layout_event,
            async_finish=True,
            allocate_on_comm_stream=True,
        )
        if recv_topk_idx is None or recv_scores is None:
            raise RuntimeError("DeepEP NVLink dispatch did not return route metadata")
        handle: Any = _NVLinkDispatchHandle(
            native_handle=native_handle,
            num_recv_tokens=int(recv_x.shape[0]),
        )
        counts = torch.empty(0, dtype=torch.int32, device=x.device)
    else:
        num_sms = _resolve_dispatch_num_sms(
            buffer,
            num_experts,
            topk_idx.shape[1],
        )
        recv_x, recv_topk_idx, recv_scores, handle, event = buffer.dispatch(
            x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_experts=num_experts,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            num_sms=num_sms,
            async_with_compute_stream=True,
            do_cpu_sync=False,
            do_expand=True,
        )
        if recv_scores is None:
            raise RuntimeError("DeepEP expanded dispatch did not return route scores")
        counts = handle.num_unaligned_recv_tokens_per_expert
        if counts is None or not torch.is_tensor(counts) or not counts.is_cuda:
            raise RuntimeError("DeepEP expanded dispatch requires GPU expert counts")
        recv_topk_idx = torch.empty(0, dtype=topk_idx.dtype, device=x.device)
    handle_id = _next_handle_id()
    key = int(handle_id.item())
    _handle_cache[key] = handle
    _pending_dispatch_events[key] = event
    return (
        recv_x,
        recv_topk_idx,
        recv_scores,
        counts,
        handle_id,
    )


def _dispatch_setup_context(ctx: Any, inputs: tuple[Any, ...], output: tuple[Any, ...]) -> None:
    x, *_ = inputs
    *_, handle_id = output
    ctx.input_dtype = x.dtype
    ctx.score_dtype = inputs[2].dtype
    ctx.saved_handle = _handle_cache.get(int(handle_id.item()))


def _dispatch_backward(
    ctx: Any,
    grad_recv_x: torch.Tensor | None,
    _grad_recv_topk_idx: torch.Tensor | None,
    grad_recv_scores: torch.Tensor | None,
    _grad_counts: torch.Tensor | None,
    _grad_handle_id: torch.Tensor | None,
) -> tuple[torch.Tensor | None, None, torch.Tensor | None, None, None]:
    if grad_recv_x is None:
        return None, None, None, None, None
    buffer = _require_buffer()
    handle = ctx.saved_handle
    if handle is None:
        raise RuntimeError("DeepEP dispatch handle was released before backward")
    if isinstance(handle, _NVLinkDispatchHandle):
        grad_x, grad_scores, _event = buffer.combine(
            grad_recv_x,
            handle.native_handle,
            topk_weights=(
                grad_recv_scores.float() if grad_recv_scores is not None else None
            ),
            async_finish=False,
        )
    else:
        grad_x, grad_scores, _event = buffer.combine(
            grad_recv_x,
            handle=handle,
            topk_weights=(
                grad_recv_scores.float() if grad_recv_scores is not None else None
            ),
            num_sms=handle.num_sms,
        )
    grad_scores = None if grad_scores is None else grad_scores.to(ctx.score_dtype)
    return grad_x.to(ctx.input_dtype), None, grad_scores, None, None


@torch.library.impl(_lib, "combine", "CUDA")
def _combine_impl(
    x: torch.Tensor,
    handle_id: torch.Tensor,
    will_backward: bool,
) -> torch.Tensor:
    global _pending_combine_event
    buffer = _require_buffer()
    key = int(handle_id.item())
    handle = _handle_cache.get(key) if will_backward else _handle_cache.pop(key, None)
    if handle is None:
        raise RuntimeError("DeepEP combine handle was not found")
    if isinstance(handle, _NVLinkDispatchHandle):
        if (
            handle.recv_token_index is None
            or handle.route_restore_order is None
            or handle.route_lengths is None
        ):
            raise RuntimeError("DeepEP NVLink route expansion was not initialized")
        recv_x = _collapse_nvlink_routes(
            x,
            handle.route_restore_order,
            handle.route_lengths,
        )
        combine_start = buffer.capture()
        combined, _combined_scores, event = buffer.combine(
            recv_x,
            handle.native_handle,
            topk_weights=None,
            previous_event=combine_start,
            async_finish=True,
            allocate_on_comm_stream=True,
        )
    else:
        combined, _combined_scores, event = buffer.combine(
            x,
            handle=handle,
            topk_weights=None,
            num_sms=handle.num_sms,
            async_with_compute_stream=True,
        )
    _pending_combine_event = event
    return combined


def _combine_setup_context(ctx: Any, inputs: tuple[Any, ...], output: torch.Tensor) -> None:
    del output
    _x, handle_id, _will_backward = inputs
    ctx.saved_handle = _handle_cache.pop(int(handle_id.item()), None)


def _combine_backward(
    ctx: Any,
    grad_combined: torch.Tensor,
) -> tuple[torch.Tensor, None, None]:
    buffer = _require_buffer()
    handle = ctx.saved_handle
    if handle is None:
        raise RuntimeError("DeepEP combine handle was released before backward")
    if isinstance(handle, _NVLinkDispatchHandle):
        grad_recv, _idx, _scores, _counts, _native_handle, _event = buffer.dispatch(
            grad_combined,
            handle=handle.native_handle,
            async_finish=False,
        )
        if handle.recv_token_index is None:
            raise RuntimeError("DeepEP NVLink route expansion was not initialized")
        grad_x = grad_recv.index_select(0, handle.recv_token_index)
    else:
        grad_x, _idx, _scores, _handle, _event = buffer.dispatch(
            grad_combined,
            handle=handle,
            num_sms=handle.num_sms,
            do_cpu_sync=False,
            do_expand=True,
        )
    return grad_x, None, None


torch.library.register_autograd(
    "dllm_deepep::dispatch",
    _dispatch_backward,
    setup_context=_dispatch_setup_context,
)
torch.library.register_autograd(
    "dllm_deepep::combine",
    _combine_backward,
    setup_context=_combine_setup_context,
)


@dataclass(slots=True)
class DeepEPDispatchState:
    handle_id: torch.Tensor
    routed_scores: torch.Tensor


@dataclass(slots=True)
class DeepEPPendingDispatch:
    recv_x: torch.Tensor
    recv_topk_idx: torch.Tensor
    recv_scores: torch.Tensor
    counts: torch.Tensor
    handle_id: torch.Tensor
    num_experts: int


def dispatch_tokens(
    hidden_states: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    num_experts: int,
    group: ProcessGroup,
    transport: DeepEPTransport,
) -> tuple[torch.Tensor, torch.Tensor, DeepEPDispatchState]:
    """Dispatch tokens to local experts with DeepEP training semantics."""

    pending = begin_dispatch_tokens(
        hidden_states,
        topk_idx,
        topk_weights,
        num_experts=num_experts,
        group=group,
        transport=transport,
    )
    return finish_dispatch_tokens(pending)


def begin_dispatch_tokens(
    hidden_states: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    num_experts: int,
    group: ProcessGroup,
    transport: DeepEPTransport,
) -> DeepEPPendingDispatch:
    """Launch DeepEP dispatch and return a pending routed-token descriptor.

    Callers that have independent compute may run it before
    :func:`finish_dispatch_tokens`, which waits for DeepEP and constructs the
    expert-major layout consumed by grouped GEMM.
    """

    if hidden_states.ndim != 2:
        raise ValueError("DeepEP dispatch expects [tokens, hidden] input")
    if not hidden_states.is_cuda:
        raise ValueError("DeepEP dispatch requires CUDA tensors")
    if topk_idx.shape != topk_weights.shape:
        raise ValueError("DeepEP top-k indices and weights must have the same shape")

    topk_idx = topk_idx.to(device=hidden_states.device, dtype=deep_ep.topk_idx_t)
    topk_weights = topk_weights.to(device=hidden_states.device, dtype=torch.float32)
    topk_idx = topk_idx.masked_fill(topk_weights == 0, -1).contiguous()
    topk_weights = topk_weights.contiguous()
    capacity = int(hidden_states.shape[0])

    with _disable_current_modes():
        _get_buffer(
            group,
            hidden=int(hidden_states.shape[1]),
            num_max_tokens_per_rank=capacity,
            num_topk=int(topk_idx.shape[1]),
            transport=transport,
        )

    (
        recv_x,
        recv_topk_idx,
        recv_scores,
        counts,
        handle_id,
    ) = torch.ops.dllm_deepep.dispatch(
        hidden_states.contiguous(),
        topk_idx,
        topk_weights,
        int(num_experts),
        capacity,
    )
    return DeepEPPendingDispatch(
        recv_x=recv_x,
        recv_topk_idx=recv_topk_idx,
        recv_scores=recv_scores,
        counts=counts,
        handle_id=handle_id,
        num_experts=int(num_experts),
    )


def finish_dispatch_tokens(
    pending: DeepEPPendingDispatch,
) -> tuple[torch.Tensor, torch.Tensor, DeepEPDispatchState]:
    """Wait for a pending DeepEP dispatch and build expert-major token order."""

    wait_dispatch(pending.handle_id)
    handle = _handle_cache.get(int(pending.handle_id.item()))
    if isinstance(handle, _NVLinkDispatchHandle):
        buffer = _require_buffer()
        if not isinstance(buffer, Buffer):
            raise RuntimeError("DeepEP NVLink handle has a non-NVLink buffer")
        local_experts = pending.num_experts // int(buffer.group_size)
        valid = (pending.recv_topk_idx >= 0) & (
            pending.recv_topk_idx < local_experts
        )
        route = torch.nonzero(valid, as_tuple=False)
        route_token_index = route[:, 0].to(dtype=torch.long)
        local_expert = pending.recv_topk_idx[route[:, 0], route[:, 1]]
        order = torch.argsort(local_expert, stable=True)
        route = route.index_select(0, order)
        local_expert = local_expert.index_select(0, order)
        recv_token_index = route[:, 0].to(dtype=torch.long)
        recv_route_index = route[:, 1].to(dtype=torch.long)
        handle.recv_token_index = recv_token_index
        route_restore_order = torch.empty_like(order)
        route_restore_order.scatter_(
            0,
            order,
            torch.arange(order.numel(), device=order.device, dtype=order.dtype),
        )
        handle.route_restore_order = route_restore_order
        handle.route_lengths = torch.bincount(
            route_token_index,
            minlength=handle.num_recv_tokens,
        )
        routed_x = pending.recv_x.index_select(0, recv_token_index)
        routed_scores = pending.recv_scores[
            recv_token_index,
            recv_route_index,
        ]
        counts = torch.bincount(
            local_expert.to(dtype=torch.long),
            minlength=local_experts,
        ).to(dtype=torch.int32)
    else:
        routed_x = pending.recv_x
        routed_scores = pending.recv_scores
        counts = pending.counts
    state = DeepEPDispatchState(
        handle_id=pending.handle_id,
        routed_scores=routed_scores,
    )
    return routed_x, counts, state


def wait_dispatch(handle_id: torch.Tensor) -> None:
    if torch.compiler.is_compiling():
        return
    event = _pending_dispatch_events.pop(int(handle_id.item()), None)
    if event is not None:
        event.current_stream_wait()


def _collapse_nvlink_routes(
    expert_output: torch.Tensor,
    route_restore_order: torch.Tensor,
    route_lengths: torch.Tensor,
) -> torch.Tensor:
    """Sum expert-major routes into DeepEP's received-token order."""

    token_major_output = expert_output.index_select(0, route_restore_order)
    return torch.segment_reduce(
        token_major_output,
        "sum",
        lengths=route_lengths,
        axis=0,
    )


def combine_tokens(
    expert_output: torch.Tensor,
    state: DeepEPDispatchState,
) -> torch.Tensor:
    """Combine local expert outputs back to the source token ranks."""

    weighted_output = expert_output * state.routed_scores.to(
        dtype=expert_output.dtype
    ).unsqueeze(-1)
    combined = torch.ops.dllm_deepep.combine(
        weighted_output.contiguous(),
        state.handle_id,
        torch.is_grad_enabled(),
    )
    sync_combine()
    return combined


def sync_combine() -> None:
    global _pending_combine_event
    if torch.compiler.is_compiling():
        return
    if _pending_combine_event is not None:
        _pending_combine_event.current_stream_wait()
    _pending_combine_event = None


def _get_buffer(
    group: ProcessGroup,
    *,
    hidden: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    transport: DeepEPTransport,
) -> Buffer | ElasticBuffer:
    global _buffer
    if transport == "nvlink":
        dispatch_config = Buffer.get_dispatch_config(group.size())
        combine_config = Buffer.get_combine_config(group.size())
        hidden_bytes = hidden * torch.empty((), dtype=torch.bfloat16).element_size()
        needed_bytes = max(
            dispatch_config.get_nvl_buffer_size_hint(hidden_bytes, group.size()),
            combine_config.get_nvl_buffer_size_hint(hidden_bytes, group.size()),
        )
        if (
            isinstance(_buffer, Buffer)
            and _buffer.group == group
            and int(_buffer.num_nvl_bytes) >= int(needed_bytes)
        ):
            return _buffer
        _destroy_buffer()
        _buffer = Buffer(
            group,
            num_nvl_bytes=needed_bytes,
            num_rdma_bytes=0,
            explicitly_destroy=True,
        )
    elif transport == "elastic":
        needed_bytes = ElasticBuffer.get_buffer_size_hint(
            group,
            num_max_tokens_per_rank,
            hidden,
            num_topk=num_topk,
            use_fp8_dispatch=False,
        )
        if (
            isinstance(_buffer, ElasticBuffer)
            and _buffer.group == group
            and int(_buffer.num_bytes) >= int(needed_bytes)
        ):
            return _buffer
        _destroy_buffer()
        _buffer = ElasticBuffer(
            group,
            num_bytes=needed_bytes,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            hidden=hidden,
            num_topk=num_topk,
            use_fp8_dispatch=False,
            explicitly_destroy=True,
        )
    else:
        raise ValueError(f"unsupported DeepEP transport: {transport}")
    return _buffer


def _destroy_buffer() -> None:
    global _buffer
    if _buffer is not None:
        _buffer.destroy()
    _buffer = None


def _require_buffer() -> Buffer | ElasticBuffer:
    if _buffer is None:
        raise RuntimeError("DeepEP buffer has not been initialized")
    return _buffer
