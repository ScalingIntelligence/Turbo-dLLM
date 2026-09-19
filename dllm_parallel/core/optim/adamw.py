# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""First-party distributed AdamW with explicit gradient synchronization.

Relocated verbatim from ``zero.py`` to separate the first-party optimizer from
the DeepSpeed ZeRO-2 integration. Behavior is unchanged; ``zero.py`` re-exports
``FirstPartyDistributedAdamW`` so existing import paths keep working.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist


class FirstPartyDistributedAdamW(torch.optim.AdamW):
    """AdamW with explicit first-party gradient synchronization.

    The optimizer synchronizes gradients over ``process_group`` immediately
    before the AdamW update. If no process group is supplied, or torch
    distributed is not initialized, it behaves like regular AdamW. CP/BP/TP
    trainers can use this with their optimizer data-parallel group when they
    want to avoid delegating the optimizer step to DeepSpeed.
    """

    def __init__(
        self,
        params: Any,
        *,
        process_group: Any | None = None,
        average_gradients: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(params, **kwargs)
        self.process_group = process_group
        self.average_gradients = bool(average_gradients)

    def step(
        self,
        closure: Any | None = None,
        *,
        synchronize_gradients: bool = True,
    ) -> Any:
        if synchronize_gradients:
            self.synchronize_gradients()
        return super().step(closure=closure)

    def synchronize_gradients(self) -> None:
        if not dist.is_available() or not dist.is_initialized():
            return
        group = self.process_group
        if group is None:
            group = dist.group.WORLD
        world_size = int(dist.get_world_size(group))
        if world_size <= 1:
            return
        buckets: dict[
            tuple[torch.device, torch.dtype],
            list[tuple[torch.Tensor, torch.Tensor]],
        ] = {}
        for group_spec in self.param_groups:
            for parameter in group_spec["params"]:
                grad = getattr(parameter, "grad", None)
                if grad is None:
                    continue
                reduce_tensor = grad if grad.is_contiguous() else grad.contiguous()
                buckets.setdefault(
                    (reduce_tensor.device, reduce_tensor.dtype),
                    [],
                ).append(
                    (
                        grad,
                        reduce_tensor,
                    )
                )
        reduced_gradients: list[torch.Tensor] = []
        for entries in buckets.values():
            if not entries:
                continue
            tensors = [entry[1] for entry in entries]
            dist.all_reduce_coalesced(
                tensors,
                op=dist.ReduceOp.SUM,
                group=group,
            )
            reduced_gradients.extend(tensors)
            if self.average_gradients:
                _foreach_div_(tensors, float(world_size))
            for target, reduced in entries:
                if target is not reduced:
                    target.copy_(reduced)
        if reduced_gradients and not _gradients_are_finite(reduced_gradients):
            raise RuntimeError(
                "first-party distributed AdamW refuses to synchronize "
                "nonfinite gradients"
            )


def _gradients_are_finite(gradients: list[torch.Tensor]) -> bool:
    buckets: dict[tuple[torch.device, torch.dtype], list[torch.Tensor]] = {}
    for gradient in gradients:
        buckets.setdefault((gradient.device, gradient.dtype), []).append(gradient)
    finite_norms = [
        torch.isfinite(norm)
        for bucket in buckets.values()
        for norm in torch._foreach_norm(bucket, 2.0)
    ]
    finite = torch.stack(finite_norms).all()
    return bool(finite)


def _foreach_div_(tensors: list[torch.Tensor], value: float) -> None:
    """Scale a homogeneous gradient bucket without allocating replacements."""

    if not tensors:
        return
    try:
        torch._foreach_div_(tensors, value)
    except RuntimeError:
        for tensor in tensors:
            tensor.div_(value)
