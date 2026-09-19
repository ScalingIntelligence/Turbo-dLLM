# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Ring transport helpers for CP/BP attention.

These functions implement the validated owner-sharded clean-KV transport used by
the bounded-memory fused CP/BP policy.
"""

from __future__ import annotations

import contextlib
from typing import Any

import torch
import torch.distributed as dist

from dllm_parallel.core.profiling.operator_trace import communication_scope


_cp_comm_streams: dict[int, Any] = {}


def _make_kv_ring_payload(
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if key.shape != value.shape:
        raise ValueError("key and value shards must have identical padded shapes")
    key_contiguous = key.contiguous()
    value_contiguous = value.contiguous()
    key_numel = key_contiguous.numel()
    payload = torch.empty(
        key_numel + value_contiguous.numel(),
        device=key.device,
        dtype=key.dtype,
    )
    payload[:key_numel].copy_(key_contiguous.view(-1))
    payload[key_numel:].copy_(value_contiguous.view(-1))
    return (
        payload,
        payload[:key_numel].view(key_contiguous.shape),
        payload[key_numel:].view(key_contiguous.shape),
    )


def _ring_exchange_kv_payload_async(
    payload: torch.Tensor,
    *,
    recv: torch.Tensor | None = None,
    key_shape: torch.Size,
    key_numel: int,
    local_rank: int,
    group_ranks: list[int],
    group: Any,
    phase: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Size, int, list[Any], Any | None]:
    if payload.ndim != 1:
        raise ValueError("K/V ring payload must be flat")
    if recv is None:
        recv = torch.empty_like(payload)
    elif (
        recv.shape != payload.shape
        or recv.dtype != payload.dtype
        or recv.device != payload.device
    ):
        raise ValueError("preallocated K/V receive buffer must match the payload")
    send_rank = group_ranks[(local_rank - 1) % len(group_ranks)]
    recv_rank = group_ranks[(local_rank + 1) % len(group_ranks)]
    work = _ring_exchange_flat_async(
        payload,
        recv,
        local_rank=local_rank,
        send_rank=send_rank,
        recv_rank=recv_rank,
        group=group,
        communication_phase=phase,
    )
    send, recv, reqs, comm_stream = work
    return send, recv, key_shape, int(key_numel), reqs, comm_stream


def _ring_exchange_kv_payload_wait(
    work: tuple[torch.Tensor, torch.Tensor, torch.Size, int, list[Any], Any | None],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    send, recv, key_shape, key_numel, reqs, comm_stream = work
    for req in reqs:
        _block_current_stream_or_wait(req)
    if comm_stream is not None:
        torch.cuda.current_stream(recv.device).wait_stream(comm_stream)
    del send
    key = recv[:key_numel].view(key_shape)
    value = recv[key_numel:].view(key_shape)
    return recv, key, value


def _ring_exchange_flat_async(
    send: torch.Tensor,
    recv: torch.Tensor,
    *,
    local_rank: int,
    send_rank: int,
    recv_rank: int,
    group: Any,
    communication_phase: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[Any], Any | None]:
    if send.shape != recv.shape:
        raise ValueError("send and recv flat buffers must have the same shape")
    if send.device != recv.device:
        raise ValueError("send and recv flat buffers must be on the same device")
    send_group_peer = dist.get_group_rank(group, int(send_rank))
    recv_group_peer = dist.get_group_rank(group, int(recv_rank))

    def p2p_ops() -> list[Any]:
        send_op = dist.P2POp(
            dist.isend,
            send,
            group=group,
            group_peer=send_group_peer,
        )
        recv_op = dist.P2POp(
            dist.irecv,
            recv,
            group=group,
            group_peer=recv_group_peer,
        )
        if local_rank % 2 == 0:
            return [send_op, recv_op]
        return [recv_op, send_op]

    input_bytes = int(send.numel()) * int(send.element_size())
    scope = (
        communication_scope(
            domain="attention",
            phase=communication_phase,
            collective="batch_isend_irecv",
            input_bytes=input_bytes,
            logical_bytes=input_bytes,
        )
        if communication_phase is not None
        else contextlib.nullcontext()
    )
    with scope:
        comm_stream = _get_cp_comm_stream(send.device)
        if comm_stream is None:
            reqs = dist.batch_isend_irecv(p2p_ops())
        else:
            comm_stream.wait_stream(torch.cuda.current_stream(send.device))
            with torch.cuda.stream(comm_stream):
                reqs = dist.batch_isend_irecv(p2p_ops())
            send.record_stream(comm_stream)
            recv.record_stream(comm_stream)
    return send, recv, reqs, comm_stream


def _ring_exchange_flat_wait(
    work: tuple[torch.Tensor, torch.Tensor, list[Any], Any | None],
) -> None:
    send, recv, reqs, comm_stream = work
    for req in reqs:
        _block_current_stream_or_wait(req)
    if comm_stream is not None:
        torch.cuda.current_stream(recv.device).wait_stream(comm_stream)
    del send


def _ring_exchange_tensors_async(
    send: tuple[torch.Tensor, ...],
    recv: tuple[torch.Tensor, ...],
    *,
    local_rank: int,
    send_rank: int,
    recv_rank: int,
    group: Any,
    phase: str,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], list[Any], Any | None]:
    if not send or len(send) != len(recv):
        raise ValueError("ring tensor packets must contain matching tensors")
    device = send[0].device
    for source, destination in zip(send, recv, strict=True):
        if source.shape != destination.shape or source.dtype != destination.dtype:
            raise ValueError("ring tensor packet buffers must match")
        if source.device != device or destination.device != device:
            raise ValueError("ring tensor packets must use one device")
    send_peer = dist.get_group_rank(group, int(send_rank))
    recv_peer = dist.get_group_rank(group, int(recv_rank))
    sends = [
        dist.P2POp(dist.isend, tensor, group=group, group_peer=send_peer)
        for tensor in send
    ]
    receives = [
        dist.P2POp(dist.irecv, tensor, group=group, group_peer=recv_peer)
        for tensor in recv
    ]
    operations = sends + receives if local_rank % 2 == 0 else receives + sends
    input_bytes = sum(
        int(tensor.numel()) * int(tensor.element_size()) for tensor in send
    )
    with communication_scope(
        domain="attention",
        phase=phase,
        collective="batch_isend_irecv",
        input_bytes=input_bytes,
        logical_bytes=input_bytes,
    ):
        comm_stream = _get_cp_comm_stream(device)
        if comm_stream is None:
            requests = dist.batch_isend_irecv(operations)
        else:
            comm_stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(comm_stream):
                requests = dist.batch_isend_irecv(operations)
            for tensor in (*send, *recv):
                tensor.record_stream(comm_stream)
    return send, recv, requests, comm_stream


def _ring_exchange_tensors_wait(
    work: tuple[
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        list[Any],
        Any | None,
    ],
) -> tuple[torch.Tensor, ...]:
    send, recv, requests, comm_stream = work
    for request in requests:
        _block_current_stream_or_wait(request)
    if comm_stream is not None:
        torch.cuda.current_stream(recv[0].device).wait_stream(comm_stream)
    del send
    return recv


def _block_current_stream_or_wait(work: Any) -> None:
    block_current_stream = getattr(work, "block_current_stream", None)
    if block_current_stream is not None:
        block_current_stream()
        return
    work.wait()


def _get_cp_comm_stream(device: torch.device) -> Any | None:
    if device.type != "cuda":
        return None
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    stream = _cp_comm_streams.get(int(device_index))
    if stream is None:
        stream = torch.cuda.Stream(device=device_index)
        _cp_comm_streams[int(device_index)] = stream
    return stream


def _ring_return_owner_grads_p2p_flat(
    owner_grad_flat: torch.Tensor,
    *,
    key_shape: torch.Size,
    key_numel: int,
    local_rank: int,
    group_ranks: list[int],
    group: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    if owner_grad_flat.ndim != 2:
        raise ValueError("owner_grad_flat must be [owners, payload]")
    num_context_ranks = owner_grad_flat.shape[0]
    if len(group_ranks) != num_context_ranks:
        raise ValueError("group_ranks must match owner_grad_flat")
    if not 0 <= local_rank < num_context_ranks:
        raise ValueError("local_rank is outside owner_grad_flat")
    if key_numel < 0 or key_numel > owner_grad_flat.shape[1]:
        raise ValueError("key_numel is outside owner payload")
    if num_context_ranks == 1:
        local = owner_grad_flat[local_rank]
        return (
            local[:key_numel].view(key_shape),
            local[key_numel:].view(key_shape),
        )

    payload_numel = owner_grad_flat.shape[1]
    if payload_numel == 0:
        local = owner_grad_flat[local_rank]
        return (
            local[:key_numel].view(key_shape),
            local[key_numel:].view(key_shape),
        )

    local = _ring_reduce_owner_payloads_p2p_flat(
        owner_grad_flat,
        local_rank=local_rank,
        group_ranks=group_ranks,
        group=group,
    )
    return (
        local[:key_numel].view(key_shape),
        local[key_numel:].view(key_shape),
    )


def _ring_reduce_owner_payloads_p2p_flat(
    owner_payloads: torch.Tensor,
    *,
    local_rank: int,
    group_ranks: list[int],
    group: Any,
) -> torch.Tensor:
    if owner_payloads.ndim != 2:
        raise ValueError("owner_payloads must be [owners, payload]")
    num_context_ranks = owner_payloads.shape[0]
    if len(group_ranks) != num_context_ranks:
        raise ValueError("group_ranks must match owner_payloads")
    if not 0 <= local_rank < num_context_ranks:
        raise ValueError("local_rank is outside owner_payloads")
    if num_context_ranks == 1 or owner_payloads.shape[1] == 0:
        return owner_payloads[local_rank].contiguous()

    chunks = owner_payloads.contiguous()
    recv = torch.empty_like(chunks[local_rank])
    send_rank = group_ranks[(local_rank - 1) % num_context_ranks]
    recv_rank = group_ranks[(local_rank + 1) % num_context_ranks]

    for step in range(num_context_ranks - 1):
        send_owner = (local_rank + step + 1) % num_context_ranks
        recv_owner = (local_rank + step + 2) % num_context_ranks
        work = _ring_exchange_flat_async(
            chunks[send_owner],
            recv,
            local_rank=local_rank,
            send_rank=send_rank,
            recv_rank=recv_rank,
            group=group,
            communication_phase="backward",
        )
        _ring_exchange_flat_wait(work)
        chunks[recv_owner].add_(recv)

    return chunks[local_rank]
