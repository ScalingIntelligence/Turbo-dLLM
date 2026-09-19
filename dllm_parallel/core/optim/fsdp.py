# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""First-party FSDP/FSDP2 wrapping helpers.

The public functions keep FSDP imports lazy so environments that only use the
DeepSpeed or plain AdamW paths do not need to import distributed sharding APIs
at package import time.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class FSDPWrapPolicy:
    """Resolved model wrapping policy for first-party sharded training."""

    backend: str = "fsdp"
    mixed_precision: str | None = None
    sharding_strategy: str = "full_shard"
    use_orig_params: bool = True

    def normalized_backend(self) -> str:
        backend = str(self.backend)
        if backend not in {"fsdp", "fsdp2"}:
            raise ValueError("FSDP backend must be 'fsdp' or 'fsdp2'")
        return backend


class FSDPTrainingModule(torch.nn.Module):
    """Evaluate the objective while FSDP parameters remain materialized.

    Packed DLLM models return hidden states and consume vocabulary parameters
    in the fused loss. Keeping both operations in one module forward prevents
    FSDP from resharding those parameters between transformer execution and
    cross entropy.
    """

    def __init__(self, model: torch.nn.Module, training_task: Any) -> None:
        super().__init__()
        self.model = model
        self.training_task = training_task

    def forward(self, prepared_batch: Any) -> torch.Tensor:
        output = self.training_task.forward(self.model, prepared_batch)
        return self.training_task.loss(self.model, prepared_batch, output)


def wrap_model_with_fsdp(
    model: torch.nn.Module,
    *,
    policy: FSDPWrapPolicy | None = None,
    process_group: Any | None = None,
    device_id: int | torch.device | None = None,
    device_mesh: Any | None = None,
    layer_modules: tuple[torch.nn.Module, ...] = (),
) -> torch.nn.Module:
    """Wrap ``model`` with FSDP1 or FSDP2 according to ``policy``.

    FSDP2 is composable and expects a device mesh in distributed launches.
    Nested units may declare ``__fsdp_forward_methods__`` when optimized model
    execution calls methods other than ``forward``.
    """

    resolved = policy or FSDPWrapPolicy()
    backend = resolved.normalized_backend()
    if backend == "fsdp2":
        mesh = device_mesh or _device_mesh_from_process_group(process_group)
        return _wrap_model_with_fsdp2(
            model,
            policy=resolved,
            device_mesh=mesh,
            layer_modules=layer_modules,
        )
    return _wrap_model_with_fsdp1(
        model,
        policy=resolved,
        process_group=process_group,
        device_id=device_id,
    )


def _wrap_model_with_fsdp1(
    model: torch.nn.Module,
    *,
    policy: FSDPWrapPolicy,
    process_group: Any | None,
    device_id: int | torch.device | None,
) -> torch.nn.Module:
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel
    except Exception as exc:
        raise RuntimeError("PyTorch FSDP is unavailable in this environment") from exc

    kwargs: dict[str, Any] = {
        "process_group": process_group,
        "use_orig_params": bool(policy.use_orig_params),
    }
    if device_id is not None:
        kwargs["device_id"] = device_id
    mixed_precision = _fsdp1_mixed_precision(policy.mixed_precision)
    if mixed_precision is not None:
        kwargs["mixed_precision"] = mixed_precision
    sharding_strategy = _fsdp1_sharding_strategy(policy.sharding_strategy)
    if sharding_strategy is not None:
        kwargs["sharding_strategy"] = sharding_strategy
    return FullyShardedDataParallel(model, **kwargs)


def is_fsdp_wrapped_model(model: Any) -> bool:
    """Return whether ``model`` is managed by PyTorch FSDP/FSDP2."""

    try:
        from torch.distributed.fsdp import FullyShardedDataParallel

        if isinstance(model, FullyShardedDataParallel):
            return True
    except Exception:
        pass
    return bool(
        hasattr(model, "_fsdp_state")
        or hasattr(model, "_fully_sharded_module")
        or type(model).__name__ == "FSDPModule"
    )


def is_fsdp2_wrapped_model(model: Any) -> bool:
    """Return whether ``model`` uses PyTorch's composable FSDP2 contract."""

    try:
        from torch.distributed.fsdp import FSDPModule

        return isinstance(model, FSDPModule)
    except Exception:
        return type(model).__name__ == "FSDPModule"


def fsdp2_state_dicts(
    model: Any,
    optimizer: Any | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Produce canonical sharded model and optimizer state for FSDP2."""

    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_state_dict,
    )

    optimizers: Any = () if optimizer is None else optimizer
    model_state, optimizer_state = get_state_dict(
        model,
        optimizers,
        options=StateDictOptions(full_state_dict=False, cpu_offload=True),
    )
    return model_state, optimizer_state if optimizer is not None else None


def load_fsdp2_state_dicts(
    model: Any,
    optimizer: Any | None,
    model_state: dict[str, Any],
    optimizer_state: dict[str, Any] | None,
    *,
    strict: bool,
) -> None:
    """Restore canonical sharded FSDP2 state on the current DP mesh."""

    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        set_state_dict,
    )

    optimizers: Any = () if optimizer is None else optimizer
    set_state_dict(
        model,
        optimizers,
        model_state_dict=model_state,
        optim_state_dict=optimizer_state or {},
        options=StateDictOptions(
            full_state_dict=False,
            cpu_offload=True,
            strict=bool(strict),
        ),
    )


@contextlib.contextmanager
def fsdp_sharded_state_dict_context(model: Any, optimizer: Any | None = None):
    """Use sharded FSDP state dicts for exact-topology distributed checkpoints."""

    try:
        from torch.distributed.fsdp import (
            FullOptimStateDictConfig,
            FullyShardedDataParallel,
            ShardedOptimStateDictConfig,
            ShardedStateDictConfig,
            StateDictType,
        )
    except Exception:
        yield
        return
    if not isinstance(model, FullyShardedDataParallel):
        yield
        return
    state_config = ShardedStateDictConfig(offload_to_cpu=True)
    optim_config = ShardedOptimStateDictConfig(offload_to_cpu=True)
    try:
        with FullyShardedDataParallel.state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
            state_dict_config=state_config,
            optim_state_dict_config=optim_config,
        ):
            yield
    except TypeError:
        # Older PyTorch builds use FullOptimStateDictConfig in this position.
        # Keep exact sharded model state and let optimizer conversion below
        # validate support for optimizer state.
        with FullyShardedDataParallel.state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
            state_dict_config=state_config,
            optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True),
        ):
            yield


def fsdp_optimizer_state_dict(model: Any, optimizer: Any | None) -> dict[str, Any] | None:
    if optimizer is None:
        return None
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel
    except Exception:
        return optimizer.state_dict()
    if not isinstance(model, FullyShardedDataParallel):
        return optimizer.state_dict()
    return FullyShardedDataParallel.optim_state_dict(model, optimizer)


def load_fsdp_optimizer_state_dict(
    model: Any,
    optimizer: Any | None,
    state: dict[str, Any] | None,
) -> None:
    if optimizer is None or state is None:
        return
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel
    except Exception:
        optimizer.load_state_dict(state)
        return
    if not isinstance(model, FullyShardedDataParallel):
        optimizer.load_state_dict(state)
        return
    optimizer.load_state_dict(
        FullyShardedDataParallel.optim_state_dict_to_load(model, optimizer, state)
    )


def _wrap_model_with_fsdp2(
    model: torch.nn.Module,
    *,
    policy: FSDPWrapPolicy,
    device_mesh: Any,
    layer_modules: tuple[torch.nn.Module, ...],
) -> torch.nn.Module:
    try:
        from torch.distributed.fsdp import (
            fully_shard,
            register_fsdp_forward_method,
        )
    except Exception as exc:
        raise RuntimeError("PyTorch FSDP2 is unavailable in this environment") from exc

    if policy.sharding_strategy == "full_shard":
        reshard_after_forward = True
    elif policy.sharding_strategy == "shard_grad_op":
        reshard_after_forward = False
    else:
        raise ValueError(
            "FSDP2 supports sharding_strategy=full_shard or shard_grad_op; "
            "hybrid sharding requires an explicit two-dimensional DP mesh"
        )
    kwargs: dict[str, Any] = {
        "mesh": device_mesh,
        "reshard_after_forward": reshard_after_forward,
    }
    if policy.mixed_precision is not None:
        kwargs["mp_policy"] = _fsdp2_mixed_precision_policy(policy.mixed_precision)
    for module in layer_modules:
        fully_shard(module, **kwargs)
        for method_name in _fsdp2_forward_methods(module):
            register_fsdp_forward_method(module, method_name)
    result = fully_shard(model, **kwargs)
    return model if result is None else result


def _fsdp2_forward_methods(module: torch.nn.Module) -> tuple[str, ...]:
    """Return custom module methods that form FSDP2 execution boundaries."""

    method_names = tuple(getattr(module, "__fsdp_forward_methods__", ()))
    for method_name in method_names:
        if not isinstance(method_name, str) or not method_name:
            raise TypeError("FSDP2 forward method names must be non-empty strings")
        if not callable(getattr(module, method_name, None)):
            raise ValueError(
                f"FSDP2 forward method {method_name!r} is not callable on "
                f"{type(module).__name__}"
            )
    return method_names


def _device_mesh_from_process_group(process_group: Any | None) -> Any:
    if process_group is None:
        raise ValueError("FSDP2 requires an explicit data-parallel process group")
    try:
        from torch.distributed.device_mesh import DeviceMesh
    except Exception as exc:
        raise RuntimeError("PyTorch DeviceMesh is unavailable") from exc
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    return DeviceMesh.from_group(
        process_group,
        device_type,
        mesh_dim_names=("data_parallel",),
    )


def _fsdp1_mixed_precision(dtype_name: str | None) -> Any | None:
    if dtype_name is None:
        return None
    try:
        from torch.distributed.fsdp import MixedPrecision
    except Exception as exc:
        raise RuntimeError("PyTorch FSDP MixedPrecision is unavailable") from exc
    dtype = _torch_dtype(dtype_name)
    return MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)


def _fsdp1_sharding_strategy(strategy_name: str) -> Any | None:
    strategy_name = str(strategy_name)
    if strategy_name in {"", "default"}:
        return None
    try:
        from torch.distributed.fsdp import ShardingStrategy
    except Exception as exc:
        raise RuntimeError("PyTorch FSDP ShardingStrategy is unavailable") from exc
    names = {
        "full_shard": "FULL_SHARD",
        "shard_grad_op": "SHARD_GRAD_OP",
        "hybrid_shard": "HYBRID_SHARD",
        "no_shard": "NO_SHARD",
    }
    try:
        return getattr(ShardingStrategy, names[strategy_name])
    except KeyError as exc:
        raise ValueError(f"unsupported FSDP sharding strategy: {strategy_name}") from exc


def _fsdp2_mixed_precision_policy(dtype_name: str) -> Any:
    try:
        from torch.distributed.fsdp import MixedPrecisionPolicy
    except Exception as exc:
        raise RuntimeError("PyTorch FSDP2 MixedPrecisionPolicy is unavailable") from exc
    dtype = _torch_dtype(dtype_name)
    return MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=dtype)


def _torch_dtype(dtype_name: str) -> torch.dtype:
    normalized = str(dtype_name).lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported mixed-precision dtype: {dtype_name}")
