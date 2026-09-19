# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Optimizer-step wiring for the block-diffusion trainer.

This module owns the trainer-facing optimization policy: learning-rate schedule
state, scheduler stepping, and global gradient clipping across DeepSpeed,
first-party torch optimizers, and FSDP. It deliberately has no dependency on
model backbones or attention kernels.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.distributed as dist


class RunSpecLRScheduler:
    """Small serializable LR scheduler driven by ``RunSpec.scheduler``."""

    def __init__(
        self,
        optimizer: Any,
        *,
        scheduler_type: str,
        warmup_steps: int,
        decay_steps: int | None,
        min_lr: float,
        weight_decay_style: str = "constant",
        weight_decay_start: float | None = None,
        weight_decay_end: float | None = None,
        weight_decay_steps: int | None = None,
    ) -> None:
        if optimizer is None or not hasattr(optimizer, "param_groups"):
            raise RuntimeError("LR scheduler requires an optimizer with param_groups")
        self.optimizer = optimizer
        self.scheduler_type = str(scheduler_type)
        self.warmup_steps = max(0, int(warmup_steps))
        self.decay_steps = None if decay_steps is None else int(decay_steps)
        self.min_lr = float(min_lr)
        self.weight_decay_style = str(weight_decay_style)
        self.weight_decay_steps = (
            None if weight_decay_steps is None else int(weight_decay_steps)
        )
        self.base_lrs = [
            float(group.get("initial_lr", group.get("lr", 0.0)))
            for group in optimizer.param_groups
        ]
        self.base_weight_decays = [
            float(group.get("initial_weight_decay", group.get("weight_decay", 0.0)))
            for group in optimizer.param_groups
        ]
        self.weight_decay_start = (
            None if weight_decay_start is None else float(weight_decay_start)
        )
        self.weight_decay_end = (
            None if weight_decay_end is None else float(weight_decay_end)
        )
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group.setdefault("initial_lr", float(base_lr))
        for group, base_weight_decay in zip(
            self.optimizer.param_groups,
            self.base_weight_decays,
        ):
            group.setdefault("initial_weight_decay", float(base_weight_decay))
        self.last_step = 0

    def step(self, step: int) -> float:
        self.last_step = int(step)
        lrs = self._lrs_for_step(self.last_step)
        weight_decays = self._weight_decays_for_step(self.last_step)
        for group, lr, weight_decay in zip(
            self.optimizer.param_groups,
            lrs,
            weight_decays,
        ):
            group["lr"] = float(lr)
            group["weight_decay"] = float(weight_decay)
        return float(lrs[0]) if lrs else 0.0

    def current_lr(self) -> float:
        if not self.optimizer.param_groups:
            return 0.0
        return float(self.optimizer.param_groups[0].get("lr", 0.0))

    def get_last_lr(self) -> list[float]:
        return [float(group.get("lr", 0.0)) for group in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": self.scheduler_type,
            "warmup_steps": int(self.warmup_steps),
            "decay_steps": self.decay_steps,
            "min_lr": float(self.min_lr),
            "base_lrs": [float(value) for value in self.base_lrs],
            "weight_decay_style": self.weight_decay_style,
            "weight_decay_steps": self.weight_decay_steps,
            "weight_decay_start": self.weight_decay_start,
            "weight_decay_end": self.weight_decay_end,
            "base_weight_decays": [
                float(value) for value in self.base_weight_decays
            ],
            "last_step": int(self.last_step),
            "lr": self.current_lr(),
        }

    def load_state_dict(self, state: dict[str, Any] | None) -> None:
        if not state:
            return
        saved_type = str(state.get("type", self.scheduler_type))
        if saved_type != self.scheduler_type:
            raise RuntimeError(
                "checkpoint scheduler type does not match current run: "
                f"{saved_type!r} != {self.scheduler_type!r}"
            )
        saved_base_lrs = state.get("base_lrs")
        if saved_base_lrs is not None:
            if len(saved_base_lrs) != len(self.optimizer.param_groups):
                raise RuntimeError(
                    "checkpoint scheduler param-group count does not match "
                    "current optimizer"
                )
            self.base_lrs = [float(value) for value in saved_base_lrs]
            for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                group["initial_lr"] = float(base_lr)
        saved_base_weight_decays = state.get("base_weight_decays")
        if saved_base_weight_decays is not None:
            if len(saved_base_weight_decays) != len(self.optimizer.param_groups):
                raise RuntimeError(
                    "checkpoint scheduler weight-decay param-group count does "
                    "not match current optimizer"
                )
            self.base_weight_decays = [
                float(value) for value in saved_base_weight_decays
            ]
            for group, base_weight_decay in zip(
                self.optimizer.param_groups,
                self.base_weight_decays,
            ):
                group["initial_weight_decay"] = float(base_weight_decay)
        self.last_step = int(state.get("last_step", 0))
        if self.last_step > 0:
            self.step(self.last_step)

    def _lrs_for_step(self, step: int) -> list[float]:
        return [self._lr_for_base_lr(base_lr, step) for base_lr in self.base_lrs]

    def _lr_for_base_lr(self, base_lr: float, step: int) -> float:
        base_lr = float(base_lr)
        min_lr = min(float(self.min_lr), base_lr)
        step = max(1, int(step))
        if self.warmup_steps > 0 and step <= self.warmup_steps:
            return min_lr + (base_lr - min_lr) * (step / float(self.warmup_steps))
        if self.scheduler_type == "constant":
            return base_lr
        decay_steps = self.decay_steps
        if decay_steps is None:
            decay_steps = max(1, int(self.warmup_steps) + 1)
        decay_steps = max(1, int(decay_steps))
        if self.scheduler_type == "inverse_square_root":
            warmup_steps = max(1, int(self.warmup_steps))
            lr = base_lr * (warmup_steps**0.5) / (step**0.5)
            return max(min_lr, lr)
        decay_step = min(
            max(0, step - int(self.warmup_steps)),
            decay_steps,
        )
        progress = decay_step / float(decay_steps)
        if self.scheduler_type == "linear":
            factor = 1.0 - progress
        elif self.scheduler_type == "cosine":
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            raise ValueError(f"unsupported scheduler type: {self.scheduler_type}")
        return min_lr + (base_lr - min_lr) * factor

    def _weight_decays_for_step(self, step: int) -> list[float]:
        return [
            self._weight_decay_for_base_weight_decay(base_weight_decay, step)
            for base_weight_decay in self.base_weight_decays
        ]

    def _weight_decay_for_base_weight_decay(
        self,
        base_weight_decay: float,
        step: int,
    ) -> float:
        start = (
            float(base_weight_decay)
            if self.weight_decay_start is None
            else float(self.weight_decay_start)
        )
        end = (
            float(base_weight_decay)
            if self.weight_decay_end is None
            else float(self.weight_decay_end)
        )
        if self.weight_decay_style == "constant":
            return end
        steps = self.weight_decay_steps
        if steps is None:
            steps = self.decay_steps
        if steps is None:
            steps = max(1, int(self.warmup_steps) + 1)
        steps = max(1, int(steps))
        progress = min(max(0, int(step)), steps) / float(steps)
        delta = end - start
        if self.weight_decay_style == "linear":
            factor = progress
        elif self.weight_decay_style == "cosine":
            factor = 0.5 * (math.cos(math.pi * (1.0 - progress)) + 1.0)
        else:
            raise ValueError(
                f"unsupported weight decay style: {self.weight_decay_style}"
            )
        return start + delta * factor


_RunSpecLRScheduler = RunSpecLRScheduler


def _scheduler_optimizer(*, optimizer: Any | None, ds_engine: Any | None) -> Any:
    if ds_engine is None:
        return optimizer
    candidate = getattr(ds_engine, "optimizer", None)
    if candidate is not None and hasattr(candidate, "param_groups"):
        return candidate
    nested = getattr(candidate, "optimizer", None)
    if nested is not None and hasattr(nested, "param_groups"):
        return nested
    return candidate


def _clip_gradients_for_optimizer(
    model: Any,
    *,
    ds_engine: Any | None,
    max_norm: float,
    runtime: Any | None = None,
) -> float | None:
    if float(max_norm) <= 0.0:
        return None
    if ds_engine is not None:
        _validate_deepspeed_gradient_clipping(
            ds_engine,
            max_norm=float(max_norm),
        )
        return None
    return _clip_torch_gradients(
        model,
        max_norm=float(max_norm),
        runtime=runtime,
    )


def _deepspeed_global_grad_norm(ds_engine: Any | None) -> float | None:
    """Read DeepSpeed's already-computed pre-clip norm after ``step``.

    ZeRO computes this norm internally while applying clipping. Reading the
    cached scalar avoids a duplicate model-parallel norm collective. DeepSpeed
    does not refresh the cache on an overflow step, so never expose that stale
    value when its overflow flag is set.
    """

    optimizer = getattr(ds_engine, "optimizer", None)
    if optimizer is None or bool(getattr(optimizer, "overflow", False)):
        return None
    if bool(getattr(optimizer, "_dllm_global_norm_skipped_when_unclipped", False)):
        return None
    value = getattr(optimizer, "_global_grad_norm", None)
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        value = value.detach().item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clip_torch_gradients(
    model: Any,
    *,
    max_norm: float,
    runtime: Any | None = None,
) -> float:
    fsdp_clip = getattr(model, "clip_grad_norm_", None)
    if callable(fsdp_clip):
        value = fsdp_clip(max_norm=float(max_norm))
        return _optional_float(value)
    parameters = [
        parameter
        for parameter in getattr(model, "parameters")()
        if getattr(parameter, "grad", None) is not None
    ]
    if not parameters:
        return 0.0
    return _clip_grad_norm_global(
        parameters,
        max_norm=float(max_norm),
        runtime=runtime,
    )


def _clip_grad_norm_global(
    parameters: list[torch.nn.Parameter],
    *,
    max_norm: float,
    runtime: Any | None,
    norm_type: float = 2.0,
) -> float:
    if float(norm_type) != 2.0:
        raise ValueError("dllm_parallel global gradient clipping supports L2 norm")
    owned_parameters = [
        parameter
        for parameter in parameters
        if parameter.grad is not None and _owns_logical_gradient_norm(parameter, runtime)
    ]
    gradients = [
        _local_gradient_tensor(parameter.grad).detach()
        for parameter in owned_parameters
    ]
    if gradients:
        devices = {gradient.device for gradient in gradients}
        if len(devices) != 1:
            raise RuntimeError("global gradient clipping expects one device per rank")
        buckets: dict[torch.dtype, list[torch.Tensor]] = {}
        for gradient in gradients:
            buckets.setdefault(gradient.dtype, []).append(gradient)
        norms = [
            norm
            for bucket in buckets.values()
            for norm in torch._foreach_norm(bucket, 2.0)
        ]
        local_sq = torch.stack([norm.float().square() for norm in norms]).sum()
    else:
        device = _gradient_norm_device(parameters)
        local_sq = torch.zeros((), dtype=torch.float32, device=device)
    data_sharded = any(
        _has_dtensor_shard(parameter, "data") for parameter in parameters
    )
    process_group = (
        dist.group.WORLD
        if data_sharded and dist.is_available() and dist.is_initialized()
        else getattr(runtime, "model_parallel_group", None)
    )
    reduction_size = (
        int(dist.get_world_size(process_group))
        if data_sharded and process_group is not None
        else int(getattr(runtime, "model_parallel_size", 1) or 1)
    )
    if (
        process_group is not None
        and reduction_size > 1
        and dist.is_available()
        and dist.is_initialized()
    ):
        dist.all_reduce(local_sq, op=dist.ReduceOp.SUM, group=process_group)
    total_norm = torch.sqrt(local_sq)
    total_norm_value = float(total_norm)
    if not math.isfinite(total_norm_value):
        raise RuntimeError("refusing to clip nonfinite gradients")
    clip_coef = float(max_norm) / (total_norm_value + 1.0e-6)
    if clip_coef < 1.0:
        _scale_gradients_in_place(parameters, clip_coef)
    return total_norm_value


def _owns_logical_gradient_norm(parameter: torch.nn.Parameter, runtime: Any | None) -> bool:
    """Count every logical parameter shard exactly once across model parallelism."""

    if runtime is None or int(getattr(runtime, "model_parallel_size", 1) or 1) <= 1:
        return True
    if int(getattr(runtime, "local_parallel_rank", 0) or 0) != 0:
        return False
    tensor_sharded = bool(
        getattr(parameter, "_dllm_tensor_parallel_sharded", False)
    ) or _has_dtensor_shard(parameter, "tensor")
    expert_sharded = bool(
        getattr(parameter, "_dllm_expert_parallel_sharded", False)
    ) or _has_dtensor_shard(parameter, "expert")
    if not tensor_sharded and int(getattr(runtime, "tensor_parallel_rank", 0) or 0) != 0:
        return False
    if not expert_sharded and int(getattr(runtime, "expert_parallel_rank", 0) or 0) != 0:
        return False
    return True


def _has_dtensor_shard(parameter: torch.nn.Parameter, axis: str) -> bool:
    placements = getattr(parameter, "placements", None)
    if placements is None:
        placements = getattr(getattr(parameter, "_spec", None), "placements", None)
    if placements is None:
        return False
    mesh = getattr(parameter, "device_mesh", None)
    if mesh is None:
        mesh = getattr(getattr(parameter, "_spec", None), "mesh", None)
    mesh_dim_names = getattr(mesh, "mesh_dim_names", None)
    if not mesh_dim_names or len(mesh_dim_names) != len(placements):
        return False
    aliases = {
        "data": {"data", "data_parallel", "dp"},
        "tensor": {"tensor", "tensor_parallel", "tp"},
        "expert": {"expert", "expert_parallel", "ep"},
    }[axis]
    return any(
        type(placement).__name__ == "Shard" and str(mesh_dim_names[index]) in aliases
        for index, placement in enumerate(placements)
    )


def _local_gradient_tensor(gradient: torch.Tensor) -> torch.Tensor:
    to_local = getattr(gradient, "to_local", None)
    return gradient if not callable(to_local) else to_local()


def _gradient_norm_device(parameters: list[torch.nn.Parameter]) -> torch.device:
    for parameter in parameters:
        grad = parameter.grad
        if grad is not None:
            return grad.device
    return torch.device("cpu")


def _scale_gradients_in_place(
    parameters: list[torch.nn.Parameter],
    scale: float,
) -> None:
    grouped: dict[tuple[torch.device, torch.dtype], list[torch.Tensor]] = {}
    for parameter in parameters:
        grad = parameter.grad
        if grad is None:
            continue
        local_grad = _local_gradient_tensor(grad).detach()
        key = (local_grad.device, local_grad.dtype)
        grouped.setdefault(key, []).append(local_grad)
    foreach_mul = getattr(torch, "_foreach_mul_", None)
    for gradients in grouped.values():
        if callable(foreach_mul):
            foreach_mul(gradients, float(scale))
        else:
            for gradient in gradients:
                gradient.mul_(float(scale))


def _validate_deepspeed_gradient_clipping(ds_engine: Any, *, max_norm: float) -> None:
    """Ensure ZeRO owns clipping exactly once during ``engine.step()``."""

    optimizer = getattr(ds_engine, "optimizer", None)
    configured = float(getattr(optimizer, "clip_grad", 0.0) or 0.0)
    if not math.isclose(configured, float(max_norm), rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError(
            "DeepSpeed ZeRO gradient clipping does not match the resolved "
            f"RunSpec: optimizer={configured}, spec={float(max_norm)}"
        )


def _optional_float(value: Any) -> float:
    if value is None:
        return 0.0
    if torch.is_tensor(value):
        return float(value.detach().float().cpu())
    return float(value)
