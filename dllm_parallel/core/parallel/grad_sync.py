# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Gradient synchronization helpers for fused CP/BP/TP runtimes."""

from __future__ import annotations

from typing import Any

import torch


def all_reduce_model_parallel_gradients(
    module: Any,
    runtime: Any,
    *,
    bucket_cap_mb: int = 0,
) -> None:
    """Average replicated parameter gradients across model-parallel workers."""

    if not runtime.enabled:
        return
    import torch.distributed as dist

    shard_group = runtime.context_block_parallel_group
    shard_group_size = len(runtime.context_block_parallel_group_ranks)
    replicated_group = getattr(runtime, "model_parallel_group", None)
    replicated_group_size = len(getattr(runtime, "model_parallel_group_ranks", []) or [])
    if (
        (shard_group is None or shard_group_size <= 1)
        and (replicated_group is None or replicated_group_size <= 1)
    ):
        return
    bucket_cap_bytes = max(0, int(bucket_cap_mb)) * 1024 * 1024
    pending: list[tuple[Any, list[torch.Tensor], list[torch.Tensor], int]] = []

    def flush(group: Any, group_size: int, entries: list[tuple[Any, Any]]) -> None:
        if not entries:
            return
        targets = [entry[0] for entry in entries]
        reduce_tensors = [entry[1] for entry in entries]
        work = dist.all_reduce_coalesced(
            reduce_tensors,
            op=dist.ReduceOp.SUM,
            group=group,
            async_op=True,
        )
        block_current_stream = getattr(work, "block_current_stream", None)
        if block_current_stream is not None:
            block_current_stream()
        pending.append((work, targets, reduce_tensors, int(group_size)))

    def append_bucket(
        buckets: dict[tuple[Any, int, Any, Any], list[list[tuple[Any, Any]]]],
        key: tuple[Any, int, Any, Any],
        entry: tuple[Any, Any],
        size: int,
    ) -> None:
        key_buckets = buckets.setdefault(key, [])
        if not key_buckets or (
            bucket_cap_bytes > 0
            and _bucket_size_bytes(key_buckets[-1]) + size > bucket_cap_bytes
        ):
            key_buckets.append([])
        key_buckets[-1].append(entry)

    sharded_buckets: dict[tuple[Any, int, Any, Any], list[list[tuple[Any, Any]]]] = {}
    replicated_buckets: dict[tuple[Any, int, Any, Any], list[list[tuple[Any, Any]]]] = {}

    for parameter in module.parameters():
        grad = parameter.grad
        if grad is None:
            continue
        group, group_size, is_sharded = _gradient_reduce_group_for_parameter(
            parameter,
            runtime=runtime,
            shard_group=shard_group,
            shard_group_size=shard_group_size,
            replicated_group=replicated_group,
            replicated_group_size=replicated_group_size,
        )
        if group is None or group_size <= 1:
            continue
        target_grad = _dtensor_local_tensor(grad)
        reduce_grad = target_grad if target_grad.is_contiguous() else target_grad.contiguous()
        key = (group, int(group_size), reduce_grad.device, reduce_grad.dtype)
        size = reduce_grad.numel() * reduce_grad.element_size()
        append_bucket(
            sharded_buckets if is_sharded else replicated_buckets,
            key,
            (target_grad, reduce_grad),
            size,
        )

    for buckets in (sharded_buckets, replicated_buckets):
        for key, bucket_list in buckets.items():
            group, group_size, _, _ = key
            for entries in bucket_list:
                flush(group, group_size, entries)

    for work, targets, reduce_tensors, group_size in pending:
        work.wait()
        _foreach_div_(reduce_tensors, float(group_size))
        for target, reduced in zip(targets, reduce_tensors, strict=True):
            if target is not reduced:
                target.copy_(reduced)


def all_reduce_sequence_parallel_replicated_gradients(
    module: Any,
    runtime: Any,
    *,
    bucket_cap_mb: int = 0,
) -> None:
    """Sum sequence-parallel replicated parameter grads over TP ranks."""

    tp_size = int(getattr(runtime, "tensor_parallel_size", 1) or 1)
    tp_group = getattr(runtime, "tensor_parallel_group", None)
    if (
        tp_size <= 1
        or tp_group is None
        or not bool(getattr(runtime, "sequence_parallel", False))
    ):
        return
    import torch.distributed as dist

    bucket_cap_bytes = max(0, int(bucket_cap_mb)) * 1024 * 1024
    buckets: list[list[tuple[torch.nn.Parameter, torch.Tensor | None, torch.Tensor]]] = []
    for parameter in module.parameters():
        if not bool(getattr(parameter, "requires_grad", False)):
            continue
        if not bool(getattr(parameter, "_dllm_sequence_parallel_replicated", False)):
            continue
        if bool(getattr(parameter, "_dllm_tensor_parallel_sharded", False)):
            continue
        if bool(getattr(parameter, "_dllm_expert_parallel_sharded", False)):
            continue
        if _dtensor_has_model_sharded_placement(parameter):
            continue
        grad = getattr(parameter, "grad", None)
        target_grad = _dtensor_local_tensor(grad) if grad is not None else None
        if target_grad is None:
            parameter_tensor = _dtensor_local_tensor(parameter)
            tensor = torch.zeros_like(parameter_tensor, memory_format=torch.contiguous_format)
        else:
            tensor = target_grad if target_grad.is_contiguous() else target_grad.contiguous()
        size = tensor.numel() * tensor.element_size()
        if not buckets or (
            bucket_cap_bytes > 0
            and _bucket_size_bytes(buckets[-1]) + size > bucket_cap_bytes
        ):
            buckets.append([])
        buckets[-1].append((parameter, target_grad, tensor))

    pending: list[
        tuple[
            Any,
            list[torch.nn.Parameter],
            list[torch.Tensor | None],
            list[torch.Tensor],
        ]
    ] = []
    for bucket in buckets:
        if not bucket:
            continue
        parameters = [item[0] for item in bucket]
        targets = [item[1] for item in bucket]
        tensors = [item[2] for item in bucket]
        work = dist.all_reduce_coalesced(
            tensors,
            op=dist.ReduceOp.SUM,
            group=tp_group,
            async_op=True,
        )
        block_current_stream = getattr(work, "block_current_stream", None)
        if block_current_stream is not None:
            block_current_stream()
        pending.append((work, parameters, targets, tensors))
    for work, parameters, targets, tensors in pending:
        work.wait()
        for parameter, target, tensor in zip(parameters, targets, tensors, strict=True):
            if target is not tensor:
                if target is None:
                    parameter.grad = tensor
                else:
                    target.copy_(tensor)


def all_reduce_data_parallel_gradients(
    module: Any,
    runtime: Any,
    *,
    bucket_cap_mb: int = 0,
) -> None:
    """Average gradients across first-party data-parallel replicas."""

    group = getattr(runtime, "data_parallel_group", None)
    ranks = getattr(runtime, "data_parallel_group_ranks", None) or []
    group_size = len(ranks)
    if group is None or group_size <= 1:
        return
    import torch.distributed as dist

    bucket_cap_bytes = max(0, int(bucket_cap_mb)) * 1024 * 1024
    buckets: dict[tuple[Any, Any], list[list[tuple[torch.Tensor, torch.Tensor]]]] = {}

    def append(entry: tuple[torch.Tensor, torch.Tensor]) -> None:
        _, reduce_grad = entry
        key = (reduce_grad.device, reduce_grad.dtype)
        key_buckets = buckets.setdefault(key, [])
        size = reduce_grad.numel() * reduce_grad.element_size()
        if not key_buckets or (
            bucket_cap_bytes > 0
            and _bucket_size_bytes(key_buckets[-1]) + size > bucket_cap_bytes
        ):
            key_buckets.append([])
        key_buckets[-1].append(entry)

    for parameter in module.parameters():
        grad = getattr(parameter, "grad", None)
        if grad is None:
            continue
        target_grad = _dtensor_local_tensor(grad)
        reduce_grad = target_grad if target_grad.is_contiguous() else target_grad.contiguous()
        append((target_grad, reduce_grad))

    pending: list[tuple[Any, list[torch.Tensor], list[torch.Tensor]]] = []
    for bucket_list in buckets.values():
        for entries in bucket_list:
            if not entries:
                continue
            targets = [entry[0] for entry in entries]
            reduce_tensors = [entry[1] for entry in entries]
            work = dist.all_reduce_coalesced(
                reduce_tensors,
                op=dist.ReduceOp.SUM,
                group=group,
                async_op=True,
            )
            block_current_stream = getattr(work, "block_current_stream", None)
            if block_current_stream is not None:
                block_current_stream()
            pending.append((work, targets, reduce_tensors))

    for work, targets, reduce_tensors in pending:
        work.wait()
        _foreach_div_(reduce_tensors, float(group_size))
        for target, reduced in zip(targets, reduce_tensors, strict=True):
            if target is not reduced:
                target.copy_(reduced)


def _dtensor_local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    to_local = getattr(tensor, "to_local", None)
    if to_local is None:
        return tensor
    return to_local()


def _gradient_reduce_group_for_parameter(
    parameter: torch.Tensor,
    *,
    runtime: Any,
    shard_group: Any,
    shard_group_size: int,
    replicated_group: Any,
    replicated_group_size: int,
) -> tuple[Any, int, bool]:
    if bool(getattr(parameter, "_dllm_tensor_parallel_sharded", False)):
        return shard_group, int(shard_group_size), True
    if bool(getattr(parameter, "_dllm_expert_parallel_sharded", False)):
        return None, 1, True
    if _dtensor_has_model_sharded_placement(parameter):
        return shard_group, int(shard_group_size), True
    if int(getattr(runtime, "tensor_parallel_size", 1) or 1) > 1:
        return replicated_group, int(replicated_group_size), False
    return shard_group, int(shard_group_size), True


def _bucket_size_bytes(entries: list[tuple[Any, ...]]) -> int:
    return sum(
        tensor.numel() * tensor.element_size()
        for entry in entries
        for tensor in entry[-1:]
    )


def _foreach_div_(tensors: list[torch.Tensor], value: float) -> None:
    if not tensors:
        return
    try:
        torch._foreach_div_(tensors, value)
    except RuntimeError:
        for tensor in tensors:
            tensor.div_(value)


def _dtensor_has_model_sharded_placement(tensor: torch.Tensor) -> bool:
    placements = getattr(tensor, "placements", None)
    if placements is None:
        spec = getattr(tensor, "_spec", None)
        placements = getattr(spec, "placements", None)
    if placements is None:
        return False
    mesh = getattr(tensor, "device_mesh", None)
    if mesh is None:
        mesh = getattr(getattr(tensor, "_spec", None), "mesh", None)
    mesh_dim_names = getattr(mesh, "mesh_dim_names", None)
    if not mesh_dim_names or len(mesh_dim_names) != len(placements):
        return any(type(placement).__name__ == "Shard" for placement in placements)
    data_parallel_names = {"data", "data_parallel", "dp"}
    return any(
        type(placement).__name__ == "Shard"
        and str(mesh_dim_names[index]) not in data_parallel_names
        for index, placement in enumerate(placements)
    )
