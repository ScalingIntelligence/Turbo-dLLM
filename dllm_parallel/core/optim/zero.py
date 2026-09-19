# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Production optimizer policy for DLLM DP/CP/BP/TP training.

The runtime contract is library-owned even when the implementation delegates
the distributed optimizer step to DeepSpeed ZeRO-2. This keeps parameter
grouping, vocab-shard safety, process-group scoping, and CUDA-op preflight in
one place instead of spreading optimizer policy through trainer scripts.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from dllm_parallel.core.optim.adamw import FirstPartyDistributedAdamW

FIRST_PARTY_OPTIMIZER_BACKENDS = frozenset(
    {
        "torch_adamw",
        "torch_fused_adamw",
        "torch_distributed_adamw",
        "fsdp",
        "fsdp2",
    }
)


@dataclass(frozen=True)
class OptimizerPolicy:
    """Resolved optimizer policy for one DLLM training job."""

    backend: str
    zero_optimizer_impl: str | None
    param_group_max_elements: int
    reduce_bucket_size: int
    contiguous_gradients: bool
    overlap_comm: bool

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "zero_optimizer_impl": self.zero_optimizer_impl,
            "param_group_max_elements": int(self.param_group_max_elements),
            "reduce_bucket_size": int(self.reduce_bucket_size),
            "contiguous_gradients": bool(self.contiguous_gradients),
            "overlap_comm": bool(self.overlap_comm),
        }


def resolve_optimizer_backend(
    requested: str,
    *,
    distributed: bool,
) -> str:
    requested = str(requested)
    if requested != "auto":
        if not distributed and requested in {"deepspeed_zero2", "fsdp", "fsdp2"}:
            raise ValueError(
                f"optimizer backend {requested!r} requires a multi-rank distributed "
                "launch; use 'torch_adamw' for a single-process run"
            )
        return requested
    if distributed:
        return "deepspeed_zero2"
    return "torch_adamw"


def is_first_party_optimizer_backend(backend: str) -> bool:
    return str(backend) in FIRST_PARTY_OPTIMIZER_BACKENDS


def verify_deepspeed_runtime_available() -> None:
    """Import the packaged DeepSpeed runtime before model allocation."""

    try:
        import deepspeed  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "DeepSpeed runtime preflight failed before model loading; verify the "
            "production installation and CUDA_HOME"
        ) from exc


def initialize_deepspeed_zero2(
    *,
    model: torch.nn.Module,
    runtime: Any,
    train_micro_batch_size_per_gpu: int,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
    dtype: str,
    reduce_bucket_size: int,
    contiguous_gradients: bool,
    overlap_comm: bool = True,
    optimizer_impl: str,
    param_group_max_elements: int,
    gradient_clip_norm: float = 0.0,
    allow_op_build: bool = False,
) -> Any:
    if runtime.plan is None:
        raise RuntimeError(
            "DeepSpeed ZeRO initialization requires a concrete parallel plan"
        )
    try:
        import deepspeed
        from dllm_parallel.core.parallel.deepspeed import build_dllm_deepspeed_mpu
    except ImportError as exc:
        raise RuntimeError(
            "deepspeed is required for production CP/BP/TP optimizer training"
        ) from exc

    mpu = build_dllm_deepspeed_mpu(runtime)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    named_model_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    resolved_optimizer_impl = resolve_zero_optimizer_impl(
        requested=optimizer_impl,
        runtime=runtime,
    )
    resolved_contiguous_gradients = bool(contiguous_gradients)
    if int(getattr(runtime, "expert_parallel_size", 1) or 1) > 1:
        resolved_contiguous_gradients = False
    resolved_reduce_bucket_size = _resolve_zero_reduce_bucket_size(
        int(reduce_bucket_size),
        named_model_parameters,
    )
    config = deepspeed_zero2_config(
        train_micro_batch_size_per_gpu=int(train_micro_batch_size_per_gpu),
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        dtype=dtype,
        reduce_bucket_size=resolved_reduce_bucket_size,
        contiguous_gradients=resolved_contiguous_gradients,
        overlap_comm=bool(overlap_comm),
        gradient_clip_norm=float(gradient_clip_norm),
    )
    model_parameters = [parameter for _, parameter in named_model_parameters]
    resolved_param_group_max_elements = _resolve_zero_param_group_max_elements(
        requested=int(param_group_max_elements),
        named_parameters=named_model_parameters,
        zero_partition_world_size=int(mpu.get_data_parallel_world_size()),
        optimizer_impl=resolved_optimizer_impl,
    )
    optimizer_param_groups = bounded_optimizer_named_param_groups(
        named_model_parameters,
        max_elements=resolved_param_group_max_elements,
        isolate_vocab_embedding=resolved_optimizer_impl
        == "deepspeed_fused_adam_hybrid",
        isolate_expert_parameters=(
            int(getattr(runtime, "expert_parallel_size", 1) or 1) > 1
        ),
        isolate_sequence_parallel_parameters=(
            bool(getattr(runtime, "sequence_parallel", False))
            and int(getattr(runtime, "tensor_parallel_size", 1) or 1) > 1
        ),
    )
    optimizer = build_optimizer(
        optimizer_param_groups,
        impl=resolved_optimizer_impl,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        allow_deepspeed_op_build=bool(allow_op_build),
    )
    engine, _, _, _ = deepspeed.initialize(
        args=argparse.Namespace(local_rank=local_rank, device_rank=local_rank),
        config=config,
        model=model,
        model_parameters=model_parameters,
        optimizer=optimizer,
        mpu=mpu,
        dist_init_required=False,
    )
    warm_deepspeed_zero2_collectives(mpu)
    engine._dllm_zero_optimizer_impl = resolved_optimizer_impl
    engine._dllm_optimizer_policy = OptimizerPolicy(
        backend="deepspeed_zero2",
        zero_optimizer_impl=resolved_optimizer_impl,
        param_group_max_elements=int(resolved_param_group_max_elements),
        reduce_bucket_size=int(resolved_reduce_bucket_size),
        contiguous_gradients=bool(resolved_contiguous_gradients),
        overlap_comm=bool(overlap_comm),
    )
    engine.optimizer._dllm_skip_model_parallel_grad_sync = bool(
        getattr(mpu, "zero_partitions_sample_parallel", False)
    )
    engine.optimizer._dllm_source_parameter_ids = {
        id(parameter) for parameter in model.parameters()
    }
    _configure_deepspeed_zero2_unclipped_step(
        engine.optimizer,
        gradient_clip_norm=float(gradient_clip_norm),
    )
    return engine


def deepspeed_zero2_block_backward_scale(
    runtime: Any,
    *,
    objective_scale: float | None,
) -> float:
    """Return a collective-safe loss scale for ZeRO-sharded BP gradients.

    BP scales each rank-local objective before DeepSpeed averages gradients
    over the DP x BP optimizer group. Applying the reciprocal objective scale
    keeps activation gradients at their natural magnitude. The factor is
    restored on ZeRO's FP32 averaged-gradient partitions before clipping and
    the optimizer step. DeepSpeed already divides ZeRO-2 reduction buckets by
    the optimizer-group width, so that width must not be applied again here.
    """

    block_parallel_size = int(getattr(runtime, "block_parallel_size", 1) or 1)
    if (
        block_parallel_size <= 1
        or str(getattr(runtime, "active_block_mode", "all_blocks")) != "dual_end"
    ):
        return 1.0

    resolved_objective_scale = (
        float(block_parallel_size)
        if objective_scale is None
        else float(objective_scale)
    )
    if resolved_objective_scale <= 0.0 or not math.isfinite(resolved_objective_scale):
        raise ValueError("block objective loss scale must be finite and positive")

    return 1.0 / resolved_objective_scale


def configure_deepspeed_zero2_block_loss_scale(
    zero_optimizer: Any,
    *,
    backward_scale: float,
) -> bool:
    """Apply BP scaling through DeepSpeed's FP32 optimizer unscale boundary."""

    backward_scale = float(backward_scale)
    if backward_scale == 1.0:
        return False
    if backward_scale <= 0.0 or not math.isfinite(backward_scale):
        raise ValueError("DeepSpeed backward scale must be finite and positive")
    override_loss_scale = getattr(zero_optimizer, "override_loss_scale", None)
    if not callable(override_loss_scale):
        raise RuntimeError(
            "DeepSpeed ZeRO optimizer does not support external loss scaling"
        )
    override_loss_scale(backward_scale)
    return True


def _configure_deepspeed_zero2_unclipped_step(
    zero_optimizer: Any,
    *,
    gradient_clip_norm: float,
) -> None:
    """Avoid DeepSpeed's unused FP64 global-norm scratch when clipping is off.

    DeepSpeed ZeRO-2 computes ``scaled_global_norm()`` on every step even when
    ``clip_grad == 0``. In BF16 long-context runs that norm allocates FP64
    temporary tensors proportional to the ZeRO averaged-gradient partitions,
    which is both unnecessary for the parameter update and enough to OOM at
    HBM-saturating batch sizes. When clipping is disabled, ``unscale_and_clip``
    uses only ``loss_scale``; the global norm is retained only for logging.
    Returning a device scalar therefore preserves the optimizer update while
    removing a tail allocation that is not part of the training semantics.
    """

    if float(gradient_clip_norm) > 0.0:
        return
    if float(getattr(zero_optimizer, "clip_grad", 0.0) or 0.0) > 0.0:
        return

    def _zero_scaled_global_norm(self: Any, norm_type: float = 2) -> torch.Tensor:
        if float(norm_type) != 2.0:
            raise AssertionError("only L2 norm supported")
        device = getattr(self, "device", None)
        if device is None:
            device = (
                torch.device("cuda", torch.cuda.current_device())
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        return torch.zeros((), device=device, dtype=torch.float32)

    zero_optimizer.scaled_global_norm = _zero_scaled_global_norm.__get__(
        zero_optimizer,
        type(zero_optimizer),
    )
    zero_optimizer._dllm_global_norm_skipped_when_unclipped = True


def warm_deepspeed_zero2_collectives(mpu: Any) -> None:
    """Eagerly initialize NCCL communicators used by ZeRO's optimizer tail."""

    if not dist.is_available() or not dist.is_initialized():
        return
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda", torch.cuda.current_device())
    token = torch.zeros((), device=device)
    seen: set[int] = set()
    for group in (
        getattr(mpu, "model_parallel_group", None),
        getattr(mpu, "data_parallel_group", None),
        getattr(mpu, "context_block_parallel_group", None),
    ):
        if group is None:
            continue
        group_id = id(group)
        if group_id in seen:
            continue
        seen.add(group_id)
        try:
            world_size = dist.get_world_size(group)
        except Exception:
            world_size = 1
        if world_size <= 1:
            continue
        dist.all_reduce(token, op=dist.ReduceOp.SUM, group=group)
    torch.cuda.synchronize(device)


def build_optimizer(
    param_groups: list[dict[str, Any]],
    *,
    impl: str,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
    process_group: Any | None = None,
    allow_deepspeed_op_build: bool = False,
) -> torch.optim.Optimizer:
    if impl == "deepspeed_fused_adam_hybrid":
        verify_deepspeed_fused_adam_available(
            allow_op_build=bool(allow_deepspeed_op_build)
        )
        return HybridDeepSpeedFusedAdamW(
            param_groups,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
    if impl == "deepspeed_fused_adam":
        verify_deepspeed_fused_adam_available(
            allow_op_build=bool(allow_deepspeed_op_build)
        )
        return _build_deepspeed_fused_adam(
            param_groups,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
    if impl == "torch_adamw":
        return torch.optim.AdamW(
            param_groups,
            lr=float(lr),
            betas=(float(betas[0]), float(betas[1])),
            eps=float(eps),
            weight_decay=float(weight_decay),
        )
    if impl == "torch_distributed_adamw":
        return FirstPartyDistributedAdamW(
            param_groups,
            process_group=process_group,
            lr=float(lr),
            betas=(float(betas[0]), float(betas[1])),
            eps=float(eps),
            weight_decay=float(weight_decay),
        )
    if impl == "torch_fused_adamw":
        return torch.optim.AdamW(
            param_groups,
            lr=float(lr),
            betas=(float(betas[0]), float(betas[1])),
            eps=float(eps),
            weight_decay=float(weight_decay),
            fused=True,
        )
    raise ValueError(f"unsupported optimizer implementation: {impl}")


class HybridDeepSpeedFusedAdamW(torch.optim.Optimizer):
    """Use DeepSpeed FusedAdam except for known-unsafe vocab embedding shards."""

    def __init__(
        self,
        param_groups: list[dict[str, Any]],
        *,
        lr: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float,
    ) -> None:
        defaults = {
            "lr": float(lr),
            "betas": (float(betas[0]), float(betas[1])),
            "eps": float(eps),
            "weight_decay": float(weight_decay),
        }
        super().__init__(param_groups, defaults)
        self._fused = _build_deepspeed_fused_adam(
            self.param_groups,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        self._torch_fused = torch.optim.AdamW(
            self.param_groups,
            lr=float(lr),
            betas=(float(betas[0]), float(betas[1])),
            eps=float(eps),
            weight_decay=float(weight_decay),
            fused=True,
        )
        self._fused.state = self.state
        self._torch_fused.state = self.state

    def step(self, closure: Any | None = None) -> Any:
        active_groups = list(self.param_groups)
        torch_groups = [
            group
            for group in active_groups
            if bool(group.get("_dllm_use_torch_fused_adamw", False))
        ]
        fused_groups = [
            group
            for group in active_groups
            if not bool(group.get("_dllm_use_torch_fused_adamw", False))
        ]
        loss = None
        if fused_groups:
            self._fused.param_groups = fused_groups
            loss = self._fused.step(closure=closure)
        if torch_groups:
            self._torch_fused.param_groups = torch_groups
            torch_loss = self._torch_fused.step(closure=closure)
            if loss is None:
                loss = torch_loss
        self._fused.param_groups = self.param_groups
        self._torch_fused.param_groups = self.param_groups
        return loss


def _build_deepspeed_fused_adam(
    param_groups: list[dict[str, Any]],
    *,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    from deepspeed.ops.adam import FusedAdam

    return FusedAdam(
        param_groups,
        lr=float(lr),
        betas=(float(betas[0]), float(betas[1])),
        eps=float(eps),
        weight_decay=float(weight_decay),
    )


def verify_deepspeed_fused_adam_available(*, allow_op_build: bool = False) -> None:
    """Fail early if the production image does not contain FusedAdam CUDA ops."""

    if allow_op_build:
        return
    try:
        from deepspeed.accelerator import get_accelerator
        from deepspeed.git_version_info import accelerator_name, installed_ops
        from deepspeed.ops.op_builder import FusedAdamBuilder
    except Exception as exc:
        raise RuntimeError(
            "DeepSpeed FusedAdam op builder is unavailable; install DeepSpeed "
            "with prebuilt CUDA ops in the production image"
        ) from exc
    runtime_accelerator = get_accelerator()._name
    if not bool(installed_ops.get("fused_adam", False)):
        raise RuntimeError(
            "DeepSpeed package metadata does not declare a prebuilt FusedAdam op"
        )
    if accelerator_name != runtime_accelerator:
        raise RuntimeError(
            "DeepSpeed FusedAdam was packaged for accelerator "
            f"{accelerator_name!r}, but runtime accelerator is {runtime_accelerator!r}"
        )
    builder = FusedAdamBuilder()
    is_installed = getattr(builder, "is_installed", None)
    if callable(is_installed):
        try:
            if bool(is_installed()):
                return
        except Exception as exc:
            raise RuntimeError("DeepSpeed FusedAdam packaged-op check failed") from exc
    for module_name in ("deepspeed.ops.adam.fused_adam_op",):
        try:
            __import__(module_name)
            return
        except Exception:
            continue
    raise RuntimeError(
        "DeepSpeed FusedAdam CUDA op is not prebuilt. Build it into the "
        "production image or set optimizer.allow_deepspeed_op_build=true only "
        "in developer recipes."
    )


def resolve_zero_optimizer_impl(
    *,
    requested: str,
    runtime: Any,
) -> str:
    unsafe_deepspeed_fused_adam = (
        int(getattr(runtime, "tensor_parallel_size", 1) or 1) > 1
        and str(getattr(runtime, "active_block_mode", "")) == "dual_end"
    )
    if requested != "auto":
        if requested == "deepspeed_fused_adam" and unsafe_deepspeed_fused_adam:
            raise RuntimeError(
                "deepspeed_fused_adam is not a valid explicit optimizer for "
                "HF tensor-parallel dual-end BP because it produced nonfinite "
                "sharded embedding updates in validation. Use "
                "zero_optimizer_impl=auto for the first-party safe optimizer "
                "policy, or request torch_fused_adamw/torch_adamw explicitly."
            )
        return str(requested)
    if unsafe_deepspeed_fused_adam:
        return "deepspeed_fused_adam_hybrid"
    return "deepspeed_fused_adam"


def bounded_optimizer_named_param_groups(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    *,
    max_elements: int,
    isolate_vocab_embedding: bool,
    isolate_expert_parameters: bool = False,
    isolate_sequence_parallel_parameters: bool = False,
) -> list[dict[str, Any]]:
    if (
        max_elements <= 0
        and not isolate_vocab_embedding
        and not isolate_expert_parameters
        and not isolate_sequence_parallel_parameters
    ):
        return [{"params": [parameter for _, parameter in named_parameters]}]
    parameters_by_kind: dict[tuple[bool, bool, bool], list[torch.nn.Parameter]] = {}
    for _name, parameter in named_parameters:
        is_expert = bool(
            isolate_expert_parameters
            and getattr(parameter, "_dllm_expert_parallel_sharded", False)
        )
        is_sequence_parallel = bool(
            isolate_sequence_parallel_parameters
            and not is_expert
            and _is_sequence_parallel_replicated_parameter(parameter)
        )
        use_torch_fused_adamw = bool(
            isolate_vocab_embedding
            and getattr(parameter, "_dllm_vocab_embedding_shard", False)
            and _is_model_parallel_shard(parameter)
        )
        kind = (use_torch_fused_adamw, is_expert, is_sequence_parallel)
        parameters_by_kind.setdefault(kind, []).append(parameter)

    groups: list[dict[str, Any]] = []
    for kind, parameters in parameters_by_kind.items():
        use_torch_fused_adamw, is_expert, is_sequence_parallel = kind
        current: list[torch.nn.Parameter] = []
        current_elements = 0

        def flush_current() -> None:
            nonlocal current, current_elements
            if not current:
                return
            group: dict[str, Any] = {"params": current}
            if use_torch_fused_adamw:
                group["_dllm_use_torch_fused_adamw"] = True
            if is_expert:
                group["_dllm_expert_parallel_sharded"] = True
            if is_sequence_parallel:
                group["_dllm_sequence_parallel_replicated"] = True
            groups.append(group)
            current = []
            current_elements = 0

        for parameter in parameters:
            numel = int(parameter.numel())
            if max_elements > 0 and current and current_elements + numel > max_elements:
                flush_current()
            current.append(parameter)
            current_elements += numel
            if max_elements > 0 and current_elements >= max_elements:
                flush_current()
        flush_current()
    return groups


def _is_model_parallel_shard(parameter: torch.Tensor) -> bool:
    """Return whether a parameter is physically sharded outside data parallelism."""

    if bool(getattr(parameter, "_dllm_tensor_parallel_sharded", False)):
        return True
    if bool(getattr(parameter, "_dllm_expert_parallel_sharded", False)):
        return True
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
        return any(type(placement).__name__ == "Shard" for placement in placements)
    data_parallel_names = {"data", "data_parallel", "dp"}
    return any(
        type(placement).__name__ == "Shard"
        and str(mesh_dim_names[index]) not in data_parallel_names
        for index, placement in enumerate(placements)
    )


def _is_sequence_parallel_replicated_parameter(parameter: torch.Tensor) -> bool:
    if not bool(getattr(parameter, "_dllm_sequence_parallel_replicated", False)):
        return False
    if bool(getattr(parameter, "_dllm_tensor_parallel_sharded", False)):
        return False
    if bool(getattr(parameter, "_dllm_expert_parallel_sharded", False)):
        return False
    placements = getattr(parameter, "placements", None)
    if placements is None:
        spec = getattr(parameter, "_spec", None)
        placements = getattr(spec, "placements", None)
    if placements is None:
        return True
    mesh = getattr(parameter, "device_mesh", None)
    if mesh is None:
        mesh = getattr(getattr(parameter, "_spec", None), "mesh", None)
    mesh_dim_names = getattr(mesh, "mesh_dim_names", None)
    if not mesh_dim_names or len(mesh_dim_names) != len(placements):
        return not any(type(placement).__name__ == "Shard" for placement in placements)
    data_parallel_names = {"data", "data_parallel", "dp"}
    return not any(
        type(placement).__name__ == "Shard"
        and str(mesh_dim_names[index]) not in data_parallel_names
        for index, placement in enumerate(placements)
    )


# ``FirstPartyDistributedAdamW`` now lives in ``adamw.py`` and is imported at the
# top of this module; ``build_optimizer`` and re-exports continue to use it.


def sync_deepspeed_runtime_model_parallel_gradients(
    *,
    zero_optimizer: Any,
    runtime: Any,
) -> None:
    """Synchronize ZeRO gradients across the local logical-model workers."""

    if bool(getattr(zero_optimizer, "_dllm_skip_model_parallel_grad_sync", False)):
        return
    tensor_parallel_size = int(getattr(runtime, "tensor_parallel_size", 1) or 1)
    clean_replica_size = int(getattr(runtime, "clean_replica_size", 1) or 1)
    if tensor_parallel_size == 1 and clean_replica_size > 1:
        group = getattr(runtime, "model_parallel_group", None)
        ranks = getattr(runtime, "model_parallel_group_ranks", None) or []
    else:
        group = getattr(runtime, "context_block_parallel_group", None)
        ranks = getattr(runtime, "context_block_parallel_group_ranks", None) or []
    from dllm_parallel.core.parallel.deepspeed import sync_model_parallel_zero_gradients

    sync_model_parallel_zero_gradients(
        zero_optimizer=zero_optimizer,
        model_parallel_group=group,
        model_parallel_world_size=len(ranks),
    )


def sync_deepspeed_sequence_parallel_gradients(
    *,
    zero_optimizer: Any,
    runtime: Any,
) -> None:
    """Sum ZeRO-2 partitions for sequence-parallel replicated parameters."""

    if not bool(getattr(runtime, "sequence_parallel", False)):
        return
    tp_size = int(getattr(runtime, "tensor_parallel_size", 1) or 1)
    if tp_size <= 1:
        return
    from dllm_parallel.core.parallel.deepspeed import (
        sync_sequence_parallel_zero_gradients,
    )

    sync_sequence_parallel_zero_gradients(
        zero_optimizer=zero_optimizer,
        tensor_parallel_group=getattr(runtime, "tensor_parallel_group", None),
        tensor_parallel_world_size=tp_size,
    )


def sync_deepspeed_expert_parallel_gradients(
    *,
    zero_optimizer: Any,
    runtime: Any,
) -> None:
    """Normalize expert gradients and average dense gradients across EP.

    EP ranks consume distinct input microbatches. DeepEP routes every source
    token to the owning expert, so an expert partition already contains the sum
    of all EP-source contributions and needs only the global-batch ``1/EP``
    normalization. Replicated dense parameters instead hold one source rank's
    gradient and must be averaged across the EP group. Optimizer groups are
    separated by parameter kind during initialization, allowing this operation
    to run on ZeRO's contiguous averaged-gradient partitions.
    """

    from dllm_parallel.core.parallel.deepspeed import (
        sync_expert_parallel_zero_gradients,
    )

    sync_expert_parallel_zero_gradients(
        zero_optimizer=zero_optimizer,
        expert_parallel_group=getattr(runtime, "expert_parallel_group", None),
        expert_parallel_world_size=int(
            getattr(runtime, "expert_parallel_size", 1) or 1
        ),
    )


def deepspeed_zero2_config(
    *,
    train_micro_batch_size_per_gpu: int,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
    dtype: str,
    reduce_bucket_size: int,
    contiguous_gradients: bool,
    overlap_comm: bool = True,
    gradient_clip_norm: float = 0.0,
) -> dict[str, Any]:
    reduce_bucket_size = max(1, int(reduce_bucket_size))
    train_micro_batch_size_per_gpu = max(1, int(train_micro_batch_size_per_gpu))
    config = {
        "train_micro_batch_size_per_gpu": train_micro_batch_size_per_gpu,
        "gradient_accumulation_steps": 1,
        "zero_allow_untested_optimizer": True,
        "bf16": {"enabled": dtype == "bf16"},
        "fp16": {"enabled": dtype == "fp16"},
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": bool(contiguous_gradients),
            "overlap_comm": bool(overlap_comm),
            "reduce_scatter": True,
            "allgather_partitions": True,
            "reduce_bucket_size": reduce_bucket_size,
            "allgather_bucket_size": reduce_bucket_size,
        },
        "wall_clock_breakdown": False,
    }
    if float(gradient_clip_norm) > 0.0:
        config["gradient_clipping"] = float(gradient_clip_norm)
    return config


def _resolve_zero_reduce_bucket_size(
    requested: int,
    named_parameters: list[tuple[str, torch.nn.Parameter]],
) -> int:
    if int(requested) > 0:
        return int(requested)
    largest_trainable_parameter = max(
        (int(parameter.numel()) for _, parameter in named_parameters),
        default=1,
    )
    return max(1, largest_trainable_parameter)


def _resolve_zero_param_group_max_elements(
    *,
    requested: int,
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    zero_partition_world_size: int,
    optimizer_impl: str,
) -> int:
    """Bound ZeRO flat optimizer partitions to fused-kernel addressable buffers.

    The default policy creates balanced groups at the ZeRO partitioning
    granularity instead of flattening the whole model into one optimizer group.
    This keeps DeepSpeed's optimizer-tail scratch and padding proportional to a
    shard's logical work while avoiding recipe-level chunk constants.
    """

    zero_partition_world_size = max(1, int(zero_partition_world_size))
    fp32_elements_per_int32_addressable_buffer = (
        int(torch.iinfo(torch.int32).max)
        // torch.empty((), dtype=torch.float32).element_size()
    )
    safe_group_elements = (
        fp32_elements_per_int32_addressable_buffer * zero_partition_world_size
    )
    if int(requested) <= 0:
        total_trainable_elements = sum(
            int(parameter.numel()) for _, parameter in named_parameters
        )
        largest_trainable_parameter = max(
            (int(parameter.numel()) for _, parameter in named_parameters),
            default=1,
        )
        if str(optimizer_impl) in {"torch_fused_adamw", "torch_adamw"}:
            return int(min(safe_group_elements, largest_trainable_parameter))
        balanced_group_elements = math.ceil(
            max(1, total_trainable_elements) / zero_partition_world_size
        )
        return int(
            min(
                safe_group_elements,
                max(largest_trainable_parameter, balanced_group_elements),
            )
        )
    if int(requested) > safe_group_elements:
        raise ValueError(
            "zero_param_group_max_elements would create ZeRO optimizer "
            "partitions larger than fused optimizer kernels can safely address: "
            f"requested={int(requested)} max_supported={int(safe_group_elements)}"
        )
    return int(requested)
