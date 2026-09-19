"""Production block-diffusion training entrypoint.

This module is the single packaged trainer for supported DLLM backbones.
Profiling scripts are thin wrappers around it so optimizer, TP/CP/BP,
corruption, loss, and synchronization semantics live in the library rather
than in harness code.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import statistics
import time
import traceback
from dataclasses import asdict, dataclass, replace
from typing import Any

import torch
import torch.distributed as dist

from dllm_parallel.core.checkpoint import (  # noqa: E402
    load_training_checkpoint,
    save_training_checkpoint,
)
from dllm_parallel.training.checkpointing import (  # noqa: E402
    checkpoint_log_payload,
    due_duration_checkpoint_fractions,
    resolve_load_checkpoint,
    should_save_checkpoint,
)
from dllm_parallel.training.fault_tolerance import (  # noqa: E402
    install_fault_handlers,
    write_failure_marker,
)
from dllm_parallel.core.parallel.runtime import (  # noqa: E402
    build_parallel_runtime,
    warm_parallel_runtime_collectives,
)
from dllm_parallel.core.parallel.preflight import (  # noqa: E402
    infer_parallel_mesh_sizes,
    launch_environment_metadata,
    launch_environment_policy,
    prepare_parallel_environment,
    validate_supported_training_axes,
)
from dllm_parallel.core.parallel.grad_sync import (  # noqa: E402
    all_reduce_data_parallel_gradients,
    all_reduce_model_parallel_gradients,
    all_reduce_sequence_parallel_replicated_gradients,
)
from dllm_parallel.core.models import load_hf_config  # noqa: E402
from dllm_parallel.core.models.compatibility import (  # noqa: E402
    validate_objective_for_family,
)
from dllm_parallel.training.run_metadata import RunContext, run_metadata  # noqa: E402
from dllm_parallel.training.metrics import WandBRunLogger  # noqa: E402
from dllm_parallel.core.models.registry import (  # noqa: E402
    describe_supported_configs,
    executor_for_family,
    supported_families,
)
from dllm_parallel.training.block_diffusion import (  # noqa: E402
    initialize_deepspeed_zero2,
    build_optimizer,
    configure_deepspeed_zero2_block_loss_scale,
    deepspeed_zero2_block_backward_scale,
    resolve_optimizer_backend,
    sync_deepspeed_expert_parallel_gradients,
    sync_deepspeed_runtime_model_parallel_gradients,
    sync_deepspeed_sequence_parallel_gradients,
    verify_deepspeed_runtime_available,
)
from dllm_parallel.core.optim import (  # noqa: E402
    FSDPTrainingModule,
    FSDPWrapPolicy,
    wrap_model_with_fsdp,
)
from dllm_parallel.training.run_spec import (  # noqa: E402
    RunSpec,
    build_run_spec_arg_parser,
    run_spec_from_args,
    run_spec_from_argv,
)
from dllm_parallel.core.kernels.runtime import configure_kernel_runtime  # noqa: E402
from dllm_parallel.training.optimizer_setup import (  # noqa: E402
    _RunSpecLRScheduler,
    _scheduler_optimizer,
    _clip_gradients_for_optimizer,
    _deepspeed_global_grad_norm,
)
from dllm_parallel.training.token_accounting import (  # noqa: E402
    build_token_accounting_policy,
)
from dllm_parallel.core.profiling.perf import (  # noqa: E402
    DFLASH_WORK_COUNT_NAMES,
    DFlashTransformerFlops,
    DiffusionGemmaTransformerFlops,
    FastDLLMv2TransformerFlops,
    MegatronTransformerFlops,
    block_diffusion_sparse_attention_pairs,
    detect_hardware_preset,
    diffusiongemma_block_attention_pairs,
    diffusiongemma_clean_attention_pairs,
    model_flops_utilization_pct,
    peak_flops_for_hardware,
)
from dllm_parallel.core.profiling.system_trace import SystemTrace  # noqa: E402
from dllm_parallel.training.execution import (  # noqa: E402
    DistributedExecution,
    PRODUCTION_EXECUTION,
)

_CPU_OBJECT_GROUP: Any | None = None


def _profile_step_is_measured(step: int, profiler_spec: Any) -> bool:
    """Exclude warm-up and CUPTI-instrumented steps from throughput timing."""
    step = int(step)
    if step <= int(profiler_spec.warmup_steps):
        return False
    if not bool(profiler_spec.system_trace):
        return True
    trace_start = int(profiler_spec.system_trace_start_step)
    trace_stop = trace_start + int(profiler_spec.system_trace_steps)
    return not trace_start <= step < trace_stop


@dataclass(frozen=True)
class _TrainingModelConfig:
    config: Any
    spec: Any
    family: str
    hf: bool


@dataclass(frozen=True)
class BlockDiffusionDebugConfig:
    grad_finite: bool = False
    optimizer_finite: bool = False
    trace: bool = False
    phase_timing: bool = False
    phase_timing_sync: bool = False


@dataclass(frozen=True)
class BlockDiffusionTrainingConfig:
    """Resolved CLI/config values for one block-diffusion training run."""

    debug: BlockDiffusionDebugConfig
    spec: RunSpec


class _AsyncLossReadback:
    """Reduce loss and token diagnostics while backward executes.

    The loss is averaged over every distributed rank because CP/BP ranks own
    disjoint, appropriately scaled objective rows while replicated TP/EP ranks
    report the same logical scalar. Input, supervised, and active token counts
    are contributed by one source rank per data-parallel sample and are summed
    without another collective.
    """

    def __init__(
        self,
        device: torch.device,
        *,
        distributed: bool,
        world_size: int,
    ) -> None:
        self._device_values = torch.zeros(5, dtype=torch.float64, device=device)
        self._host_values = torch.empty(5, dtype=torch.float64, pin_memory=True)
        self._ready = torch.cuda.Event()
        self._distributed = bool(distributed)
        self._world_size = int(world_size)
        self._reduce_work: Any | None = None
        self._enqueued = False

    def begin(self) -> None:
        self._device_values.zero_()
        self._reduce_work = None
        self._enqueued = False

    def accumulate(
        self,
        loss: torch.Tensor,
        *,
        active_tokens: int | torch.Tensor = 0,
        valid_tokens: int | torch.Tensor = 0,
        input_tokens: int | torch.Tensor = 0,
    ) -> None:
        if loss.numel() != 1:
            raise RuntimeError("training loss must be scalar")
        detached = loss.detach().to(dtype=torch.float64).reshape(())
        self._device_values[0].add_(detached)
        self._device_values[1].add_(torch.logical_not(torch.isfinite(detached)))
        self._device_values[2].add_(active_tokens)
        self._device_values[3].add_(valid_tokens)
        self._device_values[4].add_(input_tokens)

    def enqueue(self) -> None:
        if self._distributed:
            self._reduce_work = dist.all_reduce(
                self._device_values,
                op=dist.ReduceOp.SUM,
                async_op=True,
            )
        else:
            self._copy_to_host()
        self._enqueued = True

    def resolve(self) -> tuple[float, bool, int, int, int]:
        if not self._enqueued:
            raise RuntimeError("loss readback must be enqueued before it is resolved")
        if self._reduce_work is not None:
            self._reduce_work.wait()
            self._device_values[0].div_(float(self._world_size))
            self._copy_to_host()
        self._ready.synchronize()
        return (
            float(self._host_values[0]),
            bool(self._host_values[1] == 0),
            int(round(float(self._host_values[2]))),
            int(round(float(self._host_values[3]))),
            int(round(float(self._host_values[4]))),
        )

    def _copy_to_host(self) -> None:
        self._host_values.copy_(self._device_values, non_blocking=True)
        self._ready.record()


class _GradientProductionProbe:
    """Observe leaf gradients before ZeRO partitions and releases them."""

    def __init__(self, model: torch.nn.Module) -> None:
        self._names = tuple(
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
        self._seen: dict[str, torch.Tensor] = {}
        self._nonfinite: dict[str, torch.Tensor] = {}
        self._max_abs: dict[str, torch.Tensor] = {}
        self._handles = [
            parameter.register_hook(self._hook(name))
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]

    def _hook(self, name: str):
        def observe(gradient: torch.Tensor) -> torch.Tensor:
            local = _dtensor_local_tensor(gradient)
            self._seen[name] = torch.ones((), dtype=torch.bool, device=local.device)
            self._nonfinite[name] = torch.logical_not(torch.isfinite(local).all())
            self._max_abs[name] = local.detach().abs().amax().float()
            return gradient

        return observe

    def begin_step(self) -> None:
        self._seen.clear()
        self._nonfinite.clear()
        self._max_abs.clear()

    def report(self, rank: int) -> bool:
        seen_names = tuple(name for name in self._names if name in self._seen)
        missing = tuple(name for name in self._names if name not in self._seen)
        first_nonfinite: str | None = None
        if seen_names:
            flags = torch.stack([self._nonfinite[name] for name in seen_names])
            bad_indices = torch.nonzero(flags, as_tuple=False).flatten()
            if bad_indices.numel():
                first_nonfinite = seen_names[int(bad_indices[0].item())]
            maxima = torch.stack([self._max_abs[name] for name in seen_names])
            max_index = int(maxima.argmax().item())
            max_abs = float(maxima[max_index].item())
            max_abs_name = seen_names[max_index]
            nonzero_count = int(torch.count_nonzero(maxima).item())
        else:
            max_abs = 0.0
            max_abs_name = None
            nonzero_count = 0
        print(
            json.dumps(
                {
                    "event": "gradient_production_probe",
                    "rank": int(rank),
                    "trainable_count": len(self._names),
                    "produced_count": len(seen_names),
                    "nonzero_count": nonzero_count,
                    "max_abs": max_abs,
                    "max_abs_name": max_abs_name,
                    "missing_count": len(missing),
                    "first_missing": list(missing[:8]),
                    "first_nonfinite": first_nonfinite,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return not missing and first_nonfinite is None


class _ParameterUpdateProbe:
    """Track one adapter output factor across an optimizer step.

    LoRA output factors are zero-initialized, so the first ``lora_b`` tensor is
    both small to snapshot and expected to move on the first useful step.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        trainable = tuple(
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
        preferred = tuple(
            item
            for item in trainable
            if item[0].endswith(("lora_b", "gate_up_b", "down_b"))
        )
        candidates = preferred or trainable
        self._name, self._parameter = candidates[0] if candidates else (None, None)
        self._before: torch.Tensor | None = None

    def begin_step(self) -> None:
        if self._parameter is None:
            self._before = None
            return
        self._before = _dtensor_local_tensor(self._parameter).detach().clone()

    def report(self, rank: int) -> bool:
        if self._parameter is None or self._before is None:
            changed = False
            max_abs_delta = 0.0
        else:
            after = _dtensor_local_tensor(self._parameter).detach()
            delta = (after.float() - self._before.float()).abs()
            max_abs_delta = float(delta.amax().item()) if delta.numel() else 0.0
            changed = bool(torch.count_nonzero(delta).item())
        print(
            json.dumps(
                {
                    "event": "parameter_update_probe",
                    "rank": int(rank),
                    "name": self._name,
                    "changed": changed,
                    "max_abs_delta": max_abs_delta,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return changed


def _collective_gradient_validation(
    local_ok: bool,
    *,
    runtime: Any,
    device: torch.device,
) -> bool:
    """Make the pre-update gradient verdict identical on every worker."""

    if (
        not bool(getattr(runtime, "enabled", False))
        or not dist.is_available()
        or not dist.is_initialized()
        or int(dist.get_world_size()) <= 1
    ):
        return bool(local_ok)
    verdict = torch.tensor(
        1 if local_ok else 0,
        device=device,
        dtype=torch.int32,
    )
    dist.all_reduce(verdict, op=dist.ReduceOp.MIN)
    return bool(verdict.item())


def _optimizer_step_loss_plan(
    data_batches: list[Any],
    *,
    accumulation_loss_denominator: Any,
    trajectory_loss_reduction: str,
    trajectory_terminal_turn_weight: float,
) -> tuple[tuple[float | torch.Tensor | None, ...], tuple[float, ...]]:
    """Build per-turn normalization and weights for one optimizer step.

    ``token_mean`` preserves the historical accumulated-token objective.
    ``turn_mean`` normalizes every assistant turn independently. Trajectory
    groups retain source order, so their final indexed turn is terminal.
    """
    if not data_batches:
        raise RuntimeError("data runtime returned an empty optimizer step")
    if trajectory_loss_reduction == "token_mean":
        shared = accumulation_loss_denominator(data_batches)
        return (shared,) * len(data_batches), (1.0,) * len(data_batches)
    if trajectory_loss_reduction != "turn_mean":
        raise ValueError(
            "trajectory loss reduction must be 'token_mean' or 'turn_mean'"
        )
    terminal_weight = float(trajectory_terminal_turn_weight)
    if not math.isfinite(terminal_weight) or terminal_weight <= 0.0:
        raise ValueError("trajectory terminal-turn weight must be finite and positive")
    denominators = tuple(
        accumulation_loss_denominator([batch]) for batch in data_batches
    )
    weights = [1.0] * len(data_batches)
    weights[-1] = terminal_weight
    return denominators, tuple(weights)


def build_arg_parser() -> argparse.ArgumentParser:
    return build_run_spec_arg_parser(
        description="Run production block-diffusion optimizer steps."
    )


def config_from_args(args: Any) -> BlockDiffusionTrainingConfig:
    spec = run_spec_from_args(args)
    return BlockDiffusionTrainingConfig(
        debug=BlockDiffusionDebugConfig(
            grad_finite=bool(spec.debug.grad_finite),
            optimizer_finite=bool(spec.debug.optimizer_finite),
            trace=bool(spec.debug.trace),
            phase_timing=bool(spec.profiler.phase_timing),
            phase_timing_sync=bool(spec.profiler.phase_timing_sync),
        ),
        spec=spec,
    )


def training_config_from_spec(spec: RunSpec) -> BlockDiffusionTrainingConfig:
    return BlockDiffusionTrainingConfig(
        debug=BlockDiffusionDebugConfig(
            grad_finite=bool(spec.debug.grad_finite),
            optimizer_finite=bool(spec.debug.optimizer_finite),
            trace=bool(spec.debug.trace),
            phase_timing=bool(spec.profiler.phase_timing),
            phase_timing_sync=bool(spec.profiler.phase_timing_sync),
        ),
        spec=spec,
    )


def _parse_group_timeouts(
    values: list[str] | tuple[str, ...] | dict[str, Any] | None,
) -> dict[str, float]:
    if isinstance(values, dict):
        parsed_values = {
            str(group): float(seconds) for group, seconds in values.items()
        }
        if any(seconds <= 0 for seconds in parsed_values.values()):
            raise ValueError("--process-group-timeout seconds must be positive")
        return parsed_values
    timeouts: dict[str, float] = {}
    for value in values or ():
        if "=" not in value:
            raise ValueError("--process-group-timeout values must be GROUP=SECONDS")
        group_name, seconds = value.split("=", 1)
        group_name = group_name.strip()
        if not group_name:
            raise ValueError("--process-group-timeout group name cannot be empty")
        parsed = float(seconds)
        if parsed <= 0:
            raise ValueError("--process-group-timeout seconds must be positive")
        timeouts[group_name] = parsed
    return timeouts


def format_supported_configs() -> str:
    return json.dumps(describe_supported_configs(), indent=2, sort_keys=True)


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    spec, parsed = run_spec_from_argv(parser, argv)
    if bool(getattr(parsed, "print_supported_configs", False)):
        print(format_supported_configs())
        return
    if os.environ.get("RANK", "0") == "0":
        print(
            json.dumps(
                {
                    "event": "resolved_run_spec",
                    "spec": spec.to_dict(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    config = training_config_from_spec(spec)
    run_block_diffusion_training(config)


def run_block_diffusion_training(
    config: BlockDiffusionTrainingConfig,
    *,
    execution: DistributedExecution = PRODUCTION_EXECUTION,
) -> None:
    run_spec = config.spec
    model_spec = run_spec.model
    objective_spec = run_spec.objective
    data_spec = run_spec.data
    training_spec = run_spec.training
    evaluation_spec = run_spec.evaluation
    topology_spec = run_spec.topology
    optimizer_spec = run_spec.optimizer
    scheduler_spec = run_spec.scheduler
    checkpoint_spec = run_spec.checkpointing
    profiler_spec = run_spec.profiler
    logging_spec = run_spec.logging
    debug_spec = run_spec.debug
    kernel_spec = run_spec.kernel
    launch_spec = run_spec.launch
    trace_enabled = bool(config.debug.trace)
    configure_kernel_runtime(allow_runtime_jit=bool(run_spec.kernel.runtime_jit))
    validate_supported_training_axes()
    model_family = _resolve_model_family(run_spec)
    config_bundle = _load_training_model_config(
        run_spec,
        family=model_family,
    )
    _validate_objective_family(
        objective_spec,
        family=str(config_bundle.spec.family),
    )
    run_context = RunContext.create(resolved_spec=run_spec.to_dict())
    executor = executor_for_family(config_bundle.spec.family)
    executor.validate_run_spec(run_spec, config=config_bundle.config)
    resolved_block_size = _resolve_block_size(
        objective_spec.block_size,
        config_bundle.config,
    )
    if (
        objective_spec.name
        in {"standard_block_diffusion", "fast_dllm_v2"}
        and int(model_spec.seq_len) % int(resolved_block_size) != 0
    ):
        raise ValueError("model.seq_len must divide evenly by objective block size")
    if (
        int(topology_spec.context_parallel_size) > 1
        and int(model_spec.seq_len) % int(topology_spec.context_parallel_size) != 0
    ):
        raise ValueError(
            "context parallelism requires equal DualChunkSwap chunks; model.seq_len must "
            "divide evenly by topology.context_parallel_size"
        )
    prepare_parallel_environment(
        sequence_parallel=bool(topology_spec.sequence_parallel),
        context_parallel_size=int(topology_spec.context_parallel_size),
        block_parallel_size=int(topology_spec.block_parallel_size),
        tensor_parallel_size=int(topology_spec.tensor_parallel_size),
        expert_parallel_size=int(topology_spec.expert_parallel_size),
        tensor_parallel_overlap=bool(topology_spec.tensor_parallel_overlap),
    )
    install_fault_handlers()

    distributed = _init_distributed()
    rank = dist.get_rank() if distributed else 0
    run_failed = False
    try:
        world_size = dist.get_world_size() if distributed else 1
        optimizer_backend = resolve_optimizer_backend(
            optimizer_spec.backend,
            distributed=distributed,
        )
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

        if rank == 0:
            print(
                json.dumps(
                    {
                        "event": "run_metadata",
                        **run_metadata(),
                        "launch_environment_policy": launch_environment_policy(
                            sequence_parallel=bool(topology_spec.sequence_parallel),
                            context_parallel_size=int(
                                topology_spec.context_parallel_size
                            ),
                            block_parallel_size=int(topology_spec.block_parallel_size),
                            tensor_parallel_size=int(
                                topology_spec.tensor_parallel_size
                            ),
                            expert_parallel_size=int(
                                topology_spec.expert_parallel_size
                            ),
                            tensor_parallel_overlap=bool(
                                topology_spec.tensor_parallel_overlap
                            ),
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        dtype = torch.bfloat16 if model_spec.dtype == "bf16" else torch.float16
        backbone_capabilities = executor.capabilities()
        transformers = (
            _import_transformers() if backbone_capabilities.uses_transformers else None
        )
        _trace(trace_enabled, rank, -1, "before_load_model_config")
        _trace(trace_enabled, rank, -1, "after_load_model_config")
        config = config_bundle.config
        if hasattr(config, "dlm_paradigm"):
            config.dlm_paradigm = "bidirectional"
        _apply_model_config_overrides(config, model_spec)
        config_bundle = replace(
            config_bundle,
            spec=executor.metadata(config, model_id=str(model_spec.id)),
        )
        if (
            world_size <= 1
            and max(
                int(topology_spec.context_parallel_size),
                int(topology_spec.block_parallel_size),
                int(topology_spec.tensor_parallel_size),
                int(topology_spec.expert_parallel_size),
            )
            > 1
        ):
            raise RuntimeError("model-parallel topology requires a multi-rank launch")
        block_size = int(resolved_block_size)
        attention_kernel_metadata = executor.verify_native_kernels(run_spec)
        active_block_mode = execution.active_block_mode(
            block_parallel_size=int(topology_spec.block_parallel_size)
        )
        mesh_sizes = infer_parallel_mesh_sizes(
            world_size=world_size,
            context_parallel_size=topology_spec.context_parallel_size,
            block_parallel_size=topology_spec.block_parallel_size,
            tensor_parallel_size=topology_spec.tensor_parallel_size,
            expert_parallel_size=topology_spec.expert_parallel_size,
        )
        _trace(trace_enabled, rank, -1, "before_build_parallel_runtime")
        runtime = build_parallel_runtime(
            {
                "mode": "train",
                "model": {"length": model_spec.seq_len},
                "block_size": block_size,
                "parallel": {
                    "num_blocks": executor.parallel_work_units(run_spec),
                    "data_parallel_size": mesh_sizes.data_parallel_size,
                    "context_parallel_size": topology_spec.context_parallel_size,
                    "block_parallel_size": topology_spec.block_parallel_size,
                    "tensor_parallel_size": topology_spec.tensor_parallel_size,
                    "expert_parallel_size": topology_spec.expert_parallel_size,
                    "placement_policy": topology_spec.placement_policy,
                    "sequence_parallel": bool(topology_spec.sequence_parallel),
                    "tensor_parallel_overlap": bool(
                        topology_spec.tensor_parallel_overlap
                    ),
                    "cp_bp": {
                        "attention_policy": kernel_spec.cp_bp_attention_policy,
                        "clean_kv_layout": kernel_spec.cp_bp_clean_kv_layout,
                        "clean_kv_transport": kernel_spec.cp_bp_clean_kv_transport,
                        "debug_nonfinite_attention": bool(
                            kernel_spec.cp_bp_debug_nonfinite_attention
                        ),
                    },
                    "process_group_timeout_seconds": (
                        topology_spec.process_group_timeout_seconds
                    ),
                    "process_group_timeouts": _parse_group_timeouts(
                        topology_spec.process_group_timeout
                    ),
                    "active_block_mode": active_block_mode,
                    "kv_backend": _runtime_kv_backend(
                        run_spec,
                        family=model_family,
                    ),
                },
            }
        )
        _trace(trace_enabled, rank, -1, "after_build_parallel_runtime")
        warm_parallel_runtime_collectives(runtime)
        _trace(trace_enabled, rank, -1, "after_warm_parallel_runtime_collectives")
        if optimizer_backend == "deepspeed_zero2":
            verify_deepspeed_runtime_available()

        if hasattr(config, "use_cache"):
            config.use_cache = False

        run_context.with_runtime(
            runtime=runtime,
            kernel_metadata={
                "attention_kernel": attention_kernel_metadata,
                "runtime_jit": bool(run_spec.kernel.runtime_jit),
            },
            optimizer_policy=run_spec.optimizer.__dict__,
            checkpoint_policy=run_spec.checkpointing.__dict__,
            profiler_policy=run_spec.profiler.__dict__,
            logging_policy=run_spec.logging.__dict__,
        )
        run_context.metadata["launch_environment_policy"] = launch_environment_policy(
            sequence_parallel=bool(topology_spec.sequence_parallel),
            context_parallel_size=int(topology_spec.context_parallel_size),
            block_parallel_size=int(topology_spec.block_parallel_size),
            tensor_parallel_size=int(topology_spec.tensor_parallel_size),
            expert_parallel_size=int(topology_spec.expert_parallel_size),
            tensor_parallel_overlap=bool(topology_spec.tensor_parallel_overlap),
        )
        if rank == 0:
            plan = getattr(runtime, "plan", None)
            print(
                json.dumps(
                    {
                        "event": "run_context",
                        **run_context.to_log_dict(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        wandb_logger = WandBRunLogger.from_run_spec(
            spec=run_spec.to_dict(),
            run_context=run_context.to_log_dict(),
            rank=rank,
        )
        wandb_logging_enabled = wandb_logger.enabled()
        if rank == 0 and bool(logging_spec.wandb):
            print(
                json.dumps(
                    {
                        "event": "wandb_logger",
                        "enabled": wandb_logging_enabled,
                        "mode": logging_spec.wandb_mode,
                        "project": logging_spec.wandb_project,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if rank == 0:
            print(
                json.dumps(
                    {
                        "event": "loading_model",
                        "model_id": model_spec.id,
                        "world_size": world_size,
                        "seq_len": model_spec.seq_len,
                        "batch_size": training_spec.batch_size,
                        "dtype": model_spec.dtype,
                        "family": config_bundle.spec.family,
                        "hidden_size": config_bundle.spec.hidden_size,
                        "layers": config_bundle.spec.num_layers,
                        "heads": config_bundle.spec.num_attention_heads,
                        "kv_heads": config_bundle.spec.num_key_value_heads,
                        "head_dim": config_bundle.spec.head_dim,
                        "parallel_layout": getattr(plan, "layout", None),
                        "parallel_placement": getattr(plan, "placement", None),
                        "parallel_data_parallel_size": getattr(
                            plan,
                            "data_parallel_size",
                            None,
                        ),
                        "parallel_model_parallel_size": getattr(
                            plan,
                            "model_parallel_size",
                            None,
                        ),
                        "parallel_tensor_parallel_size": getattr(
                            plan,
                            "tensor_parallel_size",
                            None,
                        ),
                        "parallel_expert_parallel_size": getattr(
                            plan,
                            "expert_parallel_size",
                            None,
                        ),
                        "parallel_sample_parallel_size": getattr(
                            runtime,
                            "sample_parallel_size",
                            None,
                        ),
                        "parallel_sequence_parallel": bool(
                            getattr(runtime, "sequence_parallel", False)
                        ),
                        "parallel_tensor_parallel_overlap": bool(
                            getattr(runtime, "tensor_parallel_overlap", True)
                        ),
                        "parallel_context_parallel_size": getattr(
                            plan,
                            "context_parallel_size",
                            None,
                        ),
                        "parallel_block_parallel_size": getattr(
                            plan,
                            "block_parallel_size",
                            None,
                        ),
                        "parallel_execution": execution.name,
                        "parallel_process_groups": (
                            runtime.process_groups.to_log_dict()
                            if runtime is not None
                            and runtime.process_groups is not None
                            else None
                        ),
                        "launch_environment": launch_environment_metadata(),
                        "cp_bp_policy": (
                            runtime.cp_bp_policy.to_log_dict()
                            if runtime is not None
                            else None
                        ),
                        "attention_kernel": attention_kernel_metadata,
                        "kernel_runtime_jit": bool(run_spec.kernel.runtime_jit),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        model_start = time.perf_counter()
        tokenizer = None
        tokenizer_required = (
            (
                data_spec.input_mode == "text"
                and backbone_capabilities.tokenizer_required_for_text
            )
            or (
                data_spec.input_mode == "dataset"
                and backbone_capabilities.tokenizer_required_for_dataset
            )
            or (
                backbone_capabilities.uses_transformers
                and _config_value(config, "mask_token_id", nested="text_config") is None
            )
        )
        if tokenizer_required:
            if transformers is None:
                transformers = _import_transformers()
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                executor.tokenizer_model_id(run_spec),
                trust_remote_code=model_spec.trust_remote_code,
                revision=model_spec.revision,
            )
        prepare_tokenizer = getattr(executor, "prepare_tokenizer_and_config", None)
        if callable(prepare_tokenizer):
            prepare_tokenizer(run_spec, config=config, tokenizer=tokenizer)
        executor.validate_tokenizer_data_compatibility(run_spec, tokenizer)

        _trace(trace_enabled, rank, -1, "before_load_model_weights")
        base_model = executor.build_model(
            model_id=str(model_spec.id),
            revision=model_spec.revision,
            config=config,
            runtime=runtime,
            dtype=dtype,
            device=device,
            trust_remote_code=bool(model_spec.trust_remote_code),
        )
        _trace(trace_enabled, rank, -1, "after_load_model_weights")
        _apply_loaded_model_config_overrides(base_model, model_spec)
        model = execution.build_distributed_model(
            executor,
            base_model,
            runtime=runtime,
            spec=run_spec,
        )
        from dllm_parallel.core.adapters import apply_lora

        # Keep replicated adapter initialization topology-invariant without
        # perturbing the process RNG consumed later by training kernels.
        adapter_rng_devices = [device.index] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=adapter_rng_devices):
            torch.manual_seed(int(launch_spec.seed))
            adapter_installation = apply_lora(model, run_spec.adapter, runtime)
        torch.manual_seed(
            int(launch_spec.seed)
            + int(getattr(runtime, "sample_parallel_rank", 0) or 0)
        )
        model.train()
        mask_token_id = _mask_token_id(config, tokenizer=tokenizer)
        vocab_size = _vocab_size(config, tokenizer=tokenizer)
        block_size = int(resolved_block_size)
        if (
            objective_spec.name
            in {"standard_block_diffusion", "fast_dllm_v2"}
            and model_spec.seq_len % block_size != 0
        ):
            raise ValueError("seq_len must divide evenly by block_size")
        training_task = executor.build_training_task(
            spec=run_spec,
            runtime=runtime,
            device=device,
            seed=int(launch_spec.seed) + 104729 + rank,
            data_parallel_seed=(
                int(launch_spec.seed)
                + 104729
                + int(getattr(runtime, "sample_parallel_rank", rank))
            ),
            mask_token_id=mask_token_id,
            block_size=block_size,
            vocab_size=vocab_size,
            token_accounting=build_token_accounting_policy(
                spec=run_spec,
                runtime=runtime,
                world_size=world_size,
                gradient_accumulation_steps=int(
                    training_spec.gradient_accumulation_steps
                ),
            ),
            model_metadata=config_bundle.spec,
        )
        _startup_event(
            rank,
            "model_ready",
            elapsed_ms=(time.perf_counter() - model_start) * 1000.0,
            adapter=(
                asdict(adapter_installation)
                if adapter_installation is not None
                else None
            ),
        )
        optimizer_start = time.perf_counter()
        _startup_event(rank, "optimizer_initializing", backend=optimizer_backend)
        if optimizer_backend in {"fsdp", "fsdp2"}:
            if runtime is None or int(runtime.data_parallel_size) <= 1:
                raise ValueError(f"{optimizer_backend} requires data_parallel_size > 1")
            fsdp_layer_modules = (
                tuple(executor.fsdp_modules(model))
                if optimizer_backend == "fsdp2"
                else ()
            )
            model = FSDPTrainingModule(model, training_task)
            model = wrap_model_with_fsdp(
                model,
                policy=FSDPWrapPolicy(
                    backend=optimizer_backend,
                    mixed_precision=(
                        optimizer_spec.fsdp_mixed_precision
                        if optimizer_spec.fsdp_mixed_precision is not None
                        else model_spec.dtype
                    ),
                    sharding_strategy=str(optimizer_spec.fsdp_sharding_strategy),
                    use_orig_params=bool(optimizer_spec.fsdp_use_orig_params),
                ),
                process_group=(
                    getattr(runtime, "data_parallel_group", None)
                    if runtime is not None
                    else None
                ),
                device_id=local_rank if torch.cuda.is_available() else None,
                layer_modules=fsdp_layer_modules,
            )
        ds_engine = None
        optimizer = None
        if optimizer_backend == "deepspeed_zero2":
            if runtime is None:
                raise RuntimeError("DeepSpeed ZeRO requires a distributed runtime")
            ds_engine = initialize_deepspeed_zero2(
                model=model,
                runtime=runtime,
                train_micro_batch_size_per_gpu=int(training_spec.batch_size),
                lr=float(optimizer_spec.lr),
                betas=(
                    float(optimizer_spec.adam_beta1),
                    float(optimizer_spec.adam_beta2),
                ),
                eps=float(optimizer_spec.adam_eps),
                weight_decay=float(optimizer_spec.weight_decay),
                dtype=model_spec.dtype,
                reduce_bucket_size=int(optimizer_spec.zero_reduce_bucket_size),
                contiguous_gradients=bool(optimizer_spec.zero_contiguous_gradients),
                overlap_comm=bool(optimizer_spec.zero_overlap_comm),
                optimizer_impl=str(optimizer_spec.zero_optimizer_impl),
                param_group_max_elements=int(
                    optimizer_spec.zero_param_group_max_elements
                ),
                gradient_clip_norm=float(optimizer_spec.gradient_clip_norm),
                allow_op_build=bool(run_spec.optimizer.allow_deepspeed_op_build),
            )
            model = ds_engine
        elif optimizer_backend in {
            "torch_adamw",
            "torch_fused_adamw",
            "torch_distributed_adamw",
            "fsdp",
            "fsdp2",
        }:
            optimizer = build_optimizer(
                [
                    {
                        "params": [
                            parameter
                            for parameter in model.parameters()
                            if parameter.requires_grad
                        ]
                    }
                ],
                impl=(
                    "torch_distributed_adamw"
                    if optimizer_backend == "torch_distributed_adamw"
                    else (
                        "torch_fused_adamw"
                        if optimizer_backend == "torch_fused_adamw"
                        else "torch_adamw"
                    )
                ),
                lr=float(optimizer_spec.lr),
                betas=(
                    float(optimizer_spec.adam_beta1),
                    float(optimizer_spec.adam_beta2),
                ),
                eps=float(optimizer_spec.adam_eps),
                weight_decay=float(optimizer_spec.weight_decay),
                process_group=(
                    getattr(runtime, "data_parallel_group", None)
                    if runtime is not None
                    else None
                ),
            )
        else:
            raise ValueError(f"unsupported optimizer backend: {optimizer_backend}")
        scheduler_optimizer = _scheduler_optimizer(
            optimizer=optimizer,
            ds_engine=ds_engine,
        )
        scheduler = _RunSpecLRScheduler(
            scheduler_optimizer,
            scheduler_type=str(scheduler_spec.type),
            warmup_steps=int(scheduler_spec.warmup_steps),
            decay_steps=(
                scheduler_spec.decay_steps
                if scheduler_spec.decay_steps is not None
                else int(training_spec.steps)
            ),
            min_lr=float(scheduler_spec.min_lr),
            weight_decay_style=str(scheduler_spec.weight_decay_style),
            weight_decay_start=scheduler_spec.weight_decay_start,
            weight_decay_end=scheduler_spec.weight_decay_end,
            weight_decay_steps=scheduler_spec.weight_decay_steps,
        )
        zero_block_backward_scale = 1.0
        if ds_engine is not None:
            zero_block_backward_scale = deepspeed_zero2_block_backward_scale(
                runtime,
                objective_scale=objective_spec.bp_loss_scale,
            )
        if ds_engine is not None:
            configure_deepspeed_zero2_block_loss_scale(
                ds_engine.optimizer,
                backward_scale=zero_block_backward_scale,
            )
        _startup_event(
            rank,
            "optimizer_ready",
            backend=optimizer_backend,
            elapsed_ms=(time.perf_counter() - optimizer_start) * 1000.0,
        )
        gradient_probe = (
            _GradientProductionProbe(model) if debug_spec.grad_finite else None
        )
        parameter_update_probe = (
            _ParameterUpdateProbe(model) if debug_spec.optimizer_finite else None
        )
        if debug_spec.grad_finite:
            probe_root = getattr(model, "module", model)
            enable_boundary_probes = getattr(
                probe_root,
                "enable_gradient_boundary_probes",
                None,
            )
            if callable(enable_boundary_probes):
                enable_boundary_probes(rank=rank)
        loaded_checkpoint_state: dict[str, Any] = {}
        load_checkpoint_dir, load_checkpoint_tag, load_checkpoint_reason = (
            resolve_load_checkpoint(run_spec)
        )
        if load_checkpoint_dir is not None:
            _startup_event(rank, "checkpoint_loading", path=load_checkpoint_dir)
            loaded_checkpoint_state = load_training_checkpoint(
                load_checkpoint_dir,
                tag=load_checkpoint_tag,
                model=model if ds_engine is None else None,
                optimizer=optimizer,
                scheduler=scheduler,
                deepspeed_engine=ds_engine,
                runtime=runtime,
                device=device,
                barrier_group=_object_gather_group() if distributed else None,
            )
            if ds_engine is not None:
                scheduler.load_state_dict(
                    loaded_checkpoint_state.get("scheduler_state")
                )
            if rank == 0:
                print(
                    json.dumps(
                        checkpoint_log_payload(
                            event="checkpoint_loaded",
                            path=load_checkpoint_dir,
                            tag=load_checkpoint_tag,
                            step=int(loaded_checkpoint_state.get("step", 0)),
                            reason=load_checkpoint_reason,
                        ),
                        sort_keys=True,
                    ),
                    flush=True,
                )
            executor.load_checkpoint_hooks(loaded_checkpoint_state, model)
            from dllm_parallel.core.adapters import validate_adapter_checkpoint

            validate_adapter_checkpoint(loaded_checkpoint_state, model)
        random_token_pool = None
        if data_spec.input_mode == "random":
            random_token_pool = _synthetic_random_token_pool(
                base_model,
                runtime=runtime,
                vocab_size=vocab_size,
                sample_vocab_size=data_spec.vocab_sample_size,
                mask_token_id=mask_token_id,
                device=device,
            )
            if rank == 0 and random_token_pool is not None:
                print(
                    json.dumps(
                        {
                            "event": "synthetic_random_token_pool",
                            "tokens": int(random_token_pool.numel()),
                            "vocab_size": vocab_size,
                            "sample_vocab_size": data_spec.vocab_sample_size,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        data_start = time.perf_counter()
        _startup_event(rank, "data_initializing", input_mode=data_spec.input_mode)
        data_runtime = executor.build_data_runtime(
            spec=run_spec,
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            mask_token_id=mask_token_id,
            device=device,
            dtype=dtype,
            seed=int(launch_spec.seed) + rank,
            token_pool=random_token_pool,
            runtime=runtime,
            rank=rank,
            world_size=world_size,
        )
        data_runtime.load_state_dict(
            loaded_checkpoint_state.get("dataloader_state")
            if loaded_checkpoint_state
            else None
        )
        training_task.objective_runtime.load_state_dict(
            loaded_checkpoint_state.get("objective_state")
            if loaded_checkpoint_state
            else None
        )
        if rank == 0:
            print(
                json.dumps(
                    {
                        "event": "training_state_runtime",
                        "data": data_runtime.to_log_dict(),
                        "objective": training_task.objective_runtime.to_log_dict(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        _startup_event(
            rank,
            "data_ready",
            elapsed_ms=(time.perf_counter() - data_start) * 1000.0,
        )

        completed_steps = int(loaded_checkpoint_state.get("step", 0))
        if not loaded_checkpoint_state:
            training_task.skip(data_runtime, int(objective_spec.skip_rng_steps))
        _startup_event(rank, "training_start", completed_steps=completed_steps)
        torch.cuda.reset_peak_memory_stats(device)
        timings: list[float] = []
        losses: list[float] = []
        measured_active_tokens = torch.zeros((), dtype=torch.int64, device=device)
        measured_valid_tokens = 0
        measured_input_tokens = 0
        performance_count_names = tuple(
            getattr(training_task, "performance_count_names", ())
        )
        measured_objective_counts = (
            torch.zeros(
                len(performance_count_names),
                dtype=torch.int64,
                device=device,
            )
            if performance_count_names
            else None
        )
        phase_timings: dict[str, list[float]] = {}
        checkpoint_results: list[Any] = []
        saved_duration_checkpoint_fractions: set[float] = set()
        gradient_accumulation_steps = int(training_spec.gradient_accumulation_steps)
        if gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        trajectory_loss_reduction = str(data_spec.trajectory_loss_reduction)
        trajectory_terminal_turn_weight = float(
            data_spec.trajectory_terminal_turn_weight
        )
        target_total_steps = int(training_spec.steps)
        remaining_steps = max(0, target_total_steps - int(completed_steps))
        training_started_at = time.monotonic()
        max_duration_seconds = training_spec.max_duration_seconds
        training_deadline = (
            training_started_at + float(max_duration_seconds)
            if max_duration_seconds is not None
            else None
        )
        duration_control = (
            torch.zeros(
                1 + len(checkpoint_spec.save_duration_fractions),
                dtype=torch.uint8,
                device=device,
            )
            if training_deadline is not None and distributed
            else None
        )
        duration_limit_reached = False
        actual_completed_steps = int(completed_steps)
        prepare_training_batch = training_task.prepare
        accumulation_loss_denominator = training_task.accumulation_loss_denominator
        run_training_forward = training_task.forward
        training_output_shape = training_task.output_shape
        compute_training_loss = training_task.loss
        # Selector-loss scheduling is a DFlash training capability, not a
        # structural convention for every backbone task.  Family-gating the
        # optional hook prevents an unrelated task with the same method name
        # from acquiring new step-time behavior.
        set_training_optimizer_step = (
            getattr(training_task, "set_optimizer_step", None)
            if model_family == "dflash"
            else None
        )
        loss_readback = _AsyncLossReadback(
            device,
            distributed=distributed,
            world_size=world_size,
        )
        is_data_sample_source = _is_data_sample_source(runtime=runtime, rank=rank)
        system_trace = SystemTrace(
            enabled=bool(profiler_spec.system_trace),
            backend=str(profiler_spec.system_trace_backend),
            output_dir=profiler_spec.system_trace_dir,
            start_step=int(profiler_spec.system_trace_start_step),
            steps=int(profiler_spec.system_trace_steps),
            device=device,
            rank=rank,
        )
        forward_model = ds_engine if ds_engine is not None else model
        torch.cuda.synchronize(device)
        for local_step in range(remaining_steps):
            global_step = int(completed_steps) + int(local_step) + 1
            if callable(set_training_optimizer_step):
                set_training_optimizer_step(global_step - 1, target_total_steps)
            timed_step = _profile_step_is_measured(global_step, profiler_spec)
            if (
                int(profiler_spec.warmup_steps) > 0
                and global_step == int(profiler_spec.warmup_steps) + 1
            ):
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            system_trace.begin_step(global_step)
            step_phase_timings: dict[str, float] = {}
            phase_start = start

            def finish_phase(name: str) -> None:
                nonlocal phase_start
                if not profiler_spec.phase_timing:
                    return
                if profiler_spec.phase_timing_sync:
                    torch.cuda.synchronize(device)
                now = time.perf_counter()
                step_phase_timings[name] = (
                    step_phase_timings.get(name, 0.0) + (now - phase_start) * 1000.0
                )
                phase_start = now

            with system_trace.phase("zero_grad"):
                if ds_engine is not None:
                    ds_engine.zero_grad()
                else:
                    assert optimizer is not None
                    optimizer.zero_grad(set_to_none=True)
            if gradient_probe is not None:
                gradient_probe.begin_step()
            if parameter_update_probe is not None:
                parameter_update_probe.begin_step()
            loss_readback.begin()
            with system_trace.phase("data"):
                data_batches = data_runtime.next_optimizer_step_batches(
                    gradient_accumulation_steps
                )
                step_accumulation_steps = len(data_batches)
                if step_accumulation_steps <= 0:
                    raise RuntimeError("data runtime returned an empty optimizer step")
                step_loss_denominators, step_loss_weights = _optimizer_step_loss_plan(
                    data_batches,
                    accumulation_loss_denominator=(accumulation_loss_denominator),
                    trajectory_loss_reduction=trajectory_loss_reduction,
                    trajectory_terminal_turn_weight=(trajectory_terminal_turn_weight),
                )
                step_loss_weight_sum = float(sum(step_loss_weights))
            with _gradient_accumulation_context(
                forward_model,
                optimizer_backend=optimizer_backend,
                accumulation_steps=step_accumulation_steps,
            ):
                for micro_step in range(step_accumulation_steps):
                    micro_global_step = (global_step - 1) * 1_000_000 + micro_step
                    with system_trace.phase("data"):
                        data_batch = data_batches[micro_step]
                        prepared_batch = prepare_training_batch(
                            data_batch,
                            loss_denominator=step_loss_denominators[micro_step],
                        )
                    if timed_step:
                        corrupted_batch = getattr(prepared_batch, "corrupted", None)
                        if corrupted_batch is not None:
                            measured_active_tokens.add_(corrupted_batch.active_tokens)
                            measured_valid_tokens += int(corrupted_batch.valid_tokens)
                            measured_input_tokens += int(
                                corrupted_batch.clean_input_ids.numel()
                            )
                        if measured_objective_counts is not None:
                            objective_counts = getattr(
                                prepared_batch,
                                "performance_counts",
                                None,
                            )
                            if not isinstance(objective_counts, torch.Tensor):
                                raise RuntimeError(
                                    "objective profiling counts must be a tensor"
                                )
                            if (
                                objective_counts.shape
                                != measured_objective_counts.shape
                            ):
                                raise RuntimeError(
                                    "objective profiling count shape does not match its schema"
                                )
                            measured_objective_counts.add_(objective_counts)
                    outputs = None
                    no_sync_context = _microbatch_no_sync_context(
                        forward_model,
                        optimizer_backend=optimizer_backend,
                        enabled=micro_step + 1 < step_accumulation_steps,
                    )
                    with no_sync_context:
                        _trace(trace_enabled, rank, micro_global_step, "before_forward")
                        with system_trace.phase("forward"):
                            if optimizer_backend in {"fsdp", "fsdp2"}:
                                loss = forward_model(prepared_batch)
                                outputs = None
                                output_shape: tuple[int, ...] = ()
                            else:
                                outputs = run_training_forward(
                                    forward_model, prepared_batch
                                )
                                output_shape = training_output_shape(outputs)
                                if debug_spec.grad_finite:
                                    _attach_output_gradient_probe(outputs, rank)
                        _trace(
                            trace_enabled,
                            rank,
                            micro_global_step,
                            "after_forward",
                            logits_shape=output_shape,
                        )
                        finish_phase("forward")
                        _trace(
                            trace_enabled,
                            rank,
                            micro_global_step,
                            "before_distributed_loss",
                        )
                        with system_trace.phase("loss"):
                            if optimizer_backend not in {"fsdp", "fsdp2"}:
                                loss = compute_training_loss(
                                    forward_model,
                                    prepared_batch,
                                    outputs,
                                )
                        _trace(
                            trace_enabled,
                            rank,
                            micro_global_step,
                            "after_distributed_loss",
                        )
                        finish_phase("loss")
                        corrupted_batch = getattr(prepared_batch, "corrupted", None)
                        loss_readback.accumulate(
                            loss * step_loss_weights[micro_step],
                            active_tokens=(
                                corrupted_batch.active_tokens
                                if is_data_sample_source and corrupted_batch is not None
                                else 0
                            ),
                            valid_tokens=(
                                int(corrupted_batch.valid_tokens)
                                if is_data_sample_source and corrupted_batch is not None
                                else 0
                            ),
                            input_tokens=(
                                int(corrupted_batch.clean_input_ids.numel())
                                if is_data_sample_source and corrupted_batch is not None
                                else 0
                            ),
                        )
                        if micro_step + 1 == step_accumulation_steps:
                            loss_readback.enqueue()
                        loss_for_backward = (
                            loss * step_loss_weights[micro_step] / step_loss_weight_sum
                        )
                        _trace(
                            trace_enabled, rank, micro_global_step, "before_backward"
                        )
                        with system_trace.phase("backward"):
                            if ds_engine is not None:
                                ds_engine.set_gradient_accumulation_boundary(
                                    micro_step + 1 == step_accumulation_steps
                                )
                                ds_engine.backward(loss_for_backward)
                            else:
                                loss_for_backward.backward()
                        _trace(trace_enabled, rank, micro_global_step, "after_backward")
                        finish_phase("backward")
                    loss = None
                    loss_for_backward = None
                    outputs = None
                    prepared_batch = None
                    data_batch = None
                data_batches = None
                step_loss_denominators = None
                step_loss_weights = None
            if gradient_probe is not None:
                local_gradients_ok = gradient_probe.report(rank)
                gradients_ok = _collective_gradient_validation(
                    local_gradients_ok,
                    runtime=runtime,
                    device=device,
                )
                if not gradients_ok:
                    raise RuntimeError(
                        "gradient production validation failed before optimizer update: "
                        f"rank={rank} step={global_step} local_ok={local_gradients_ok}"
                    )
            (
                loss_value_for_log,
                loss_is_finite,
                step_active_tokens,
                step_valid_tokens,
                step_input_tokens,
            ) = loss_readback.resolve()
            loss_value_for_log /= step_loss_weight_sum
            if not loss_is_finite:
                raise RuntimeError(
                    "nonfinite block-diffusion loss before optimizer update: "
                    f"rank={rank} step={global_step} loss={loss_value_for_log}"
                )
            if debug_spec.grad_finite:
                _print_first_nonfinite_gradient(model, rank)
            _trace(
                trace_enabled, rank, global_step, "before_model_parallel_grad_reduce"
            )
            with system_trace.phase("gradient_sync"):
                if ds_engine is None and bool(
                    getattr(runtime, "sequence_parallel", False)
                ):
                    all_reduce_sequence_parallel_replicated_gradients(
                        getattr(forward_model, "module", forward_model),
                        runtime,
                    )
                if ds_engine is not None:
                    sync_deepspeed_runtime_model_parallel_gradients(
                        zero_optimizer=ds_engine.optimizer,
                        runtime=runtime,
                    )
                    sync_deepspeed_sequence_parallel_gradients(
                        zero_optimizer=ds_engine.optimizer,
                        runtime=runtime,
                    )
                    sync_deepspeed_expert_parallel_gradients(
                        zero_optimizer=ds_engine.optimizer,
                        runtime=runtime,
                    )
                else:
                    all_reduce_model_parallel_gradients(model, runtime)
                    if optimizer_backend == "torch_distributed_adamw":
                        assert optimizer is not None
                        optimizer.synchronize_gradients()
                    elif optimizer_backend not in {"fsdp", "fsdp2"}:
                        all_reduce_data_parallel_gradients(model, runtime)
            _trace(trace_enabled, rank, global_step, "after_model_parallel_grad_reduce")
            finish_phase("model_parallel_grad_reduce")
            if debug_spec.grad_finite:
                _print_first_nonfinite_gradient(model, rank, prefix="after_reduce")
            # The optimizer step only needs gradients/ZeRO state. Drop large
            # forward/loss tensors before DeepSpeed's grad-norm and partition
            # update path so HBM-saturating runs do not carry dead activations
            # into the optimizer tail.
            if not training_spec.skip_optimizer_step:
                if debug_spec.optimizer_finite and ds_engine is not None:
                    _print_first_nonfinite_zero_gradient(ds_engine.optimizer, rank)
                lr_for_step = scheduler.step(global_step)
                with system_trace.phase("gradient_clip"):
                    grad_norm = _clip_gradients_for_optimizer(
                        model,
                        ds_engine=ds_engine,
                        max_norm=float(optimizer_spec.gradient_clip_norm),
                        runtime=runtime,
                    )
                if grad_norm is not None:
                    finish_phase("gradient_clip")
                _trace(
                    trace_enabled,
                    rank,
                    global_step,
                    "before_optimizer_step",
                    lr=lr_for_step,
                    grad_norm=grad_norm,
                )
                with system_trace.phase("optimizer"):
                    if ds_engine is not None:
                        ds_engine.step()
                        grad_norm = _deepspeed_global_grad_norm(ds_engine)
                    else:
                        assert optimizer is not None
                        if optimizer_backend == "torch_distributed_adamw":
                            optimizer.step(synchronize_gradients=False)
                        else:
                            optimizer.step()
                _trace(trace_enabled, rank, global_step, "after_optimizer_step")
                finish_phase("optimizer_step")
                if parameter_update_probe is not None:
                    parameter_update_probe.report(rank)
            else:
                lr_for_step = scheduler.current_lr()
                grad_norm = None
            if debug_spec.grad_finite and not training_spec.skip_optimizer_step:
                _print_first_nonfinite_parameter(model, rank)
            if debug_spec.optimizer_finite and ds_engine is not None:
                _print_first_nonfinite_zero_optimizer_state(ds_engine.optimizer, rank)
            if training_deadline is not None:
                torch.cuda.synchronize(device)
                (
                    duration_limit_reached,
                    duration_checkpoint_fractions,
                ) = _distributed_duration_checkpoint_control(
                    spec=run_spec,
                    elapsed_seconds=time.monotonic() - training_started_at,
                    saved_fractions=saved_duration_checkpoint_fractions,
                    rank=rank,
                    distributed=distributed,
                    control=duration_control,
                )
            else:
                duration_checkpoint_fractions = ()
            step_checkpoint_due = should_save_checkpoint(
                spec=run_spec,
                global_step=global_step,
                target_total_steps=target_total_steps,
                training_complete=duration_limit_reached,
            )
            checkpoint_requests: list[tuple[str, float | None]] = []
            for fraction in duration_checkpoint_fractions:
                percentage = int(round(float(fraction) * 100.0))
                checkpoint_requests.append(
                    (
                        f"duration_{percentage:03d}pct_step_{global_step:08d}",
                        float(fraction),
                    )
                )
            if step_checkpoint_due and not checkpoint_requests:
                checkpoint_requests.append((f"step_{global_step:08d}", None))
            if checkpoint_requests:
                if checkpoint_results and checkpoint_results[-1].async_save:
                    checkpoint_results[-1].wait()
                backbone_state = executor.save_checkpoint_hooks(
                    getattr(model, "module", model)
                )
                from dllm_parallel.core.adapters import adapter_metadata

                saved_adapter_metadata = adapter_metadata(
                    getattr(model, "module", model)
                )
                if saved_adapter_metadata is not None:
                    backbone_state = {
                        **backbone_state,
                        "adapter": saved_adapter_metadata,
                    }
                for checkpoint_tag, duration_fraction in checkpoint_requests:
                    checkpoint_result = save_training_checkpoint(
                        checkpoint_spec.save_checkpoint_dir,
                        tag=checkpoint_tag,
                        step=global_step,
                        model=model if ds_engine is None else None,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        deepspeed_engine=ds_engine,
                        runtime=runtime,
                        objective_state=training_task.objective_runtime.state_dict(),
                        dataloader_state=data_runtime.state_dict(),
                        config=run_spec.to_dict(),
                        scheduler_state=scheduler.state_dict(),
                        kernel_metadata={
                            "attention_kernel": attention_kernel_metadata,
                            "runtime_jit": bool(run_spec.kernel.runtime_jit),
                        },
                        run_metadata=run_context.to_log_dict(),
                        profiler_metadata={
                            "phase_timing": bool(profiler_spec.phase_timing),
                            "phase_timing_sync": bool(profiler_spec.phase_timing_sync),
                        },
                        backbone_state=backbone_state,
                        device=device,
                        async_save=bool(checkpoint_spec.async_checkpoint_save),
                        model_only=bool(checkpoint_spec.model_only),
                        keep_last_n=int(checkpoint_spec.keep_last_n),
                        barrier_group=_object_gather_group() if distributed else None,
                    )
                    checkpoint_results.append(checkpoint_result)
                    if duration_fraction is not None:
                        saved_duration_checkpoint_fractions.add(duration_fraction)
                    if rank == 0:
                        print(
                            json.dumps(
                                checkpoint_log_payload(
                                    event="checkpoint_saved",
                                    path=str(checkpoint_result.path),
                                    tag=checkpoint_result.tag,
                                    step=int(global_step),
                                    async_save=bool(checkpoint_result.async_save),
                                    model_only=bool(checkpoint_spec.model_only),
                                    duration_fraction=duration_fraction,
                                ),
                                sort_keys=True,
                            ),
                            flush=True,
                        )
            torch.cuda.synchronize(device)
            system_trace.end_step(global_step)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            step_performance = _step_throughput_metrics(
                input_tokens=step_input_tokens,
                valid_tokens=step_valid_tokens,
                active_tokens=step_active_tokens,
                elapsed_ms=elapsed_ms,
                world_size=world_size,
                data_parallel_size=int(
                    getattr(getattr(runtime, "plan", None), "data_parallel_size", 1)
                    or 1
                ),
            )
            timed = timed_step
            if timed:
                timings.append(elapsed_ms)
                losses.append(float(loss_value_for_log))
                if profiler_spec.phase_timing:
                    for name, value in step_phase_timings.items():
                        phase_timings.setdefault(name, []).append(float(value))
            if rank == 0:
                step_payload = {
                    "event": "step",
                    "step": global_step,
                    "completed_steps": global_step,
                    "target_total_steps": target_total_steps,
                    "timed": timed,
                    "loss": float(loss_value_for_log),
                    "lr": float(lr_for_step),
                    "grad_norm": grad_norm,
                    "gradient_clip_norm": float(optimizer_spec.gradient_clip_norm),
                    "ms": elapsed_ms,
                    "active_tokens": step_active_tokens,
                    "valid_tokens": step_valid_tokens,
                    "input_tokens": step_input_tokens,
                    "perf": step_performance,
                    "optimizer_step": not training_spec.skip_optimizer_step,
                    "gradient_accumulation_steps": step_accumulation_steps,
                    "trajectory_loss_reduction": trajectory_loss_reduction,
                    "trajectory_terminal_turn_weight": (
                        trajectory_terminal_turn_weight
                    ),
                    "configured_gradient_accumulation_steps": (
                        gradient_accumulation_steps
                    ),
                    "optimizer_backend": optimizer_backend,
                    "zero_optimizer_impl": getattr(
                        ds_engine,
                        "_dllm_zero_optimizer_impl",
                        None,
                    )
                    if ds_engine is not None
                    else None,
                    "zero_sample_parallel": bool(
                        getattr(
                            getattr(ds_engine, "optimizer", None),
                            "_dllm_skip_model_parallel_grad_sync",
                            False,
                        )
                    )
                    if ds_engine is not None
                    else None,
                    "optimizer_overflow": bool(
                        getattr(
                            getattr(ds_engine, "optimizer", None), "overflow", False
                        )
                    )
                    if ds_engine is not None
                    else None,
                    "peak_mib": torch.cuda.max_memory_allocated(device) / 1024 / 1024,
                    "phase_ms": step_phase_timings
                    if profiler_spec.phase_timing
                    else None,
                }
                print(json.dumps(step_payload, sort_keys=True), flush=True)
                if wandb_logging_enabled:
                    wandb_logger.log_step(step_payload)
            actual_completed_steps = int(global_step)
            if duration_limit_reached:
                break

        if not timings:
            timings.append(0.0)
        if not losses:
            losses.append(0.0)

        if distributed:
            torch.cuda.synchronize(device)
            dist.barrier(group=_object_gather_group())

        evaluation_metrics = None
        if evaluation_spec.batches > 0:
            evaluation_metrics = _evaluate_masked_token_accuracy(
                executor=executor,
                run_spec=run_spec,
                model=forward_model,
                tokenizer=tokenizer,
                vocab_size=vocab_size,
                mask_token_id=mask_token_id,
                block_size=block_size,
                runtime=runtime,
                device=device,
                dtype=dtype,
                rank=rank,
                world_size=world_size,
                distributed=distributed,
            )
            if rank == 0:
                print(
                    json.dumps(
                        {"event": "evaluation", **evaluation_metrics},
                        sort_keys=True,
                    ),
                    flush=True,
                )

        rank_metrics = {
            "rank": rank,
            "avg_ms": statistics.mean(timings),
            "min_ms": min(timings),
            "max_ms": max(timings),
            "avg_loss": statistics.mean(losses),
            "peak_mib": torch.cuda.max_memory_allocated(device) / 1024 / 1024,
            "measured_active_tokens": (
                int(measured_active_tokens) if is_data_sample_source else 0
            ),
            "measured_valid_tokens": (
                int(measured_valid_tokens) if is_data_sample_source else 0
            ),
            "measured_input_tokens": (
                int(measured_input_tokens) if is_data_sample_source else 0
            ),
            "measured_steps": len(timings),
            "measured_objective_count_names": list(performance_count_names),
            "measured_objective_counts": (
                [int(value) for value in measured_objective_counts.tolist()]
                if is_data_sample_source and measured_objective_counts is not None
                else []
            ),
            "phase_avg_ms": {
                name: statistics.mean(values) for name, values in phase_timings.items()
            },
        }
        if distributed:
            gathered: list[Any] = [None for _ in range(world_size)]
            dist.all_gather_object(
                gathered,
                rank_metrics,
                group=_object_gather_group(),
            )
        else:
            gathered = [rank_metrics]
        if rank == 0:
            max_avg_ms = max(item["avg_ms"] for item in gathered)
            throughput = _training_throughput_metrics(
                rank_metrics=gathered,
                elapsed_ms=max_avg_ms,
                world_size=world_size,
                data_parallel_size=int(
                    getattr(getattr(runtime, "plan", None), "data_parallel_size", 1)
                    or 1
                ),
            )
            mfu_metrics = _megatron_mfu_metrics(
                rank_metrics=gathered,
                elapsed_ms=max_avg_ms,
                model_spec=config_bundle.spec,
                objective_name=str(objective_spec.name),
                seq_len=int(model_spec.seq_len),
                block_size=int(block_size),
                vocab_size=int(vocab_size),
                world_size=int(world_size),
                dflash_loss_kind=str(objective_spec.dflash_loss),
                adapter_type=str(run_spec.adapter.type),
                context_parallel_size=int(run_spec.topology.context_parallel_size),
                block_parallel_size=int(run_spec.topology.block_parallel_size),
            )
            summary_payload = {
                "event": "summary",
                "model_id": model_spec.id,
                "seq_len": model_spec.seq_len,
                "batch_size": training_spec.batch_size,
                "steps": training_spec.steps,
                "profile_warmup_steps": int(profiler_spec.warmup_steps),
                "completed_steps": actual_completed_steps,
                "target_total_steps": target_total_steps,
                "max_duration_seconds": max_duration_seconds,
                "duration_limit_reached": duration_limit_reached,
                "training_elapsed_seconds": time.monotonic() - training_started_at,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "scheduler": scheduler.state_dict(),
                "gradient_clip_norm": float(optimizer_spec.gradient_clip_norm),
                "optimizer_step": not training_spec.skip_optimizer_step,
                "optimizer_backend": optimizer_backend,
                "parallel_layout": getattr(
                    getattr(runtime, "plan", None),
                    "layout",
                    None,
                ),
                "parallel_placement": getattr(
                    getattr(runtime, "plan", None),
                    "placement",
                    None,
                ),
                "parallel_data_parallel_size": getattr(
                    getattr(runtime, "plan", None),
                    "data_parallel_size",
                    None,
                ),
                "parallel_model_parallel_size": getattr(
                    getattr(runtime, "plan", None),
                    "model_parallel_size",
                    None,
                ),
                "parallel_tensor_parallel_size": getattr(
                    getattr(runtime, "plan", None),
                    "tensor_parallel_size",
                    None,
                ),
                "parallel_expert_parallel_size": getattr(
                    getattr(runtime, "plan", None),
                    "expert_parallel_size",
                    None,
                ),
                "parallel_sample_parallel_size": getattr(
                    runtime,
                    "sample_parallel_size",
                    None,
                ),
                "parallel_sequence_parallel": bool(
                    getattr(runtime, "sequence_parallel", False)
                )
                if runtime is not None
                else None,
                "parallel_tensor_parallel_overlap": bool(
                    getattr(runtime, "tensor_parallel_overlap", True)
                )
                if runtime is not None
                else None,
                "parallel_context_parallel_size": getattr(
                    getattr(runtime, "plan", None),
                    "context_parallel_size",
                    None,
                ),
                "parallel_block_parallel_size": getattr(
                    getattr(runtime, "plan", None),
                    "block_parallel_size",
                    None,
                ),
                "parallel_execution": execution.name,
                "zero_optimizer_impl": getattr(
                    ds_engine,
                    "_dllm_zero_optimizer_impl",
                    None,
                )
                if ds_engine is not None
                else None,
                "zero_sample_parallel": bool(
                    getattr(
                        getattr(ds_engine, "optimizer", None),
                        "_dllm_skip_model_parallel_grad_sync",
                        False,
                    )
                )
                if ds_engine is not None
                else None,
                "cp_bp_policy": (
                    runtime.cp_bp_policy.to_log_dict() if runtime is not None else None
                ),
                "max_avg_ms": max_avg_ms,
                "max_peak_mib": max(item["peak_mib"] for item in gathered),
                "throughput": throughput,
                **mfu_metrics,
                "rank_metrics": gathered,
                "data": data_runtime.to_log_dict(),
                "objective": training_task.objective_runtime.to_log_dict(),
                "evaluation": evaluation_metrics,
            }
            print(json.dumps(summary_payload, sort_keys=True), flush=True)
            if wandb_logging_enabled:
                wandb_logger.log_summary(summary_payload)
        if distributed:
            dist.barrier(group=_object_gather_group())
    except BaseException as exc:
        run_failed = True
        fatal_payload = {
            "event": "fatal_exception",
            "rank": rank,
            "type": exc.__class__.__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_failure_marker(
            run_dir=os.environ.get("DLLM_RUN_DIR"),
            payload=fatal_payload,
        )
        print(
            json.dumps(fatal_payload, sort_keys=True),
            flush=True,
        )
        raise
    finally:
        if "wandb_logger" in locals():
            wandb_logger.finish(exit_code=1 if run_failed else 0)
        for checkpoint_result in locals().get("checkpoint_results", []):
            checkpoint_result.wait()
        if distributed:
            from dllm_parallel.core.parallel.transformer_engine import (
                destroy_transformer_engine_userbuffers,
            )

            destroy_transformer_engine_userbuffers()
            runtime_for_shutdown = locals().get("runtime")
            process_groups = getattr(runtime_for_shutdown, "process_groups", None)
            if hasattr(process_groups, "destroy"):
                process_groups.destroy()
            global _CPU_OBJECT_GROUP
            if (
                _CPU_OBJECT_GROUP is not None
                and _CPU_OBJECT_GROUP is not dist.group.WORLD
            ):
                dist.destroy_process_group(_CPU_OBJECT_GROUP)
                _CPU_OBJECT_GROUP = None
            dist.destroy_process_group()


def _evaluate_masked_token_accuracy(
    *,
    executor: Any,
    run_spec: RunSpec,
    model: Any,
    tokenizer: Any | None,
    vocab_size: int,
    mask_token_id: int,
    block_size: int,
    runtime: Any,
    device: torch.device,
    dtype: torch.dtype,
    rank: int,
    world_size: int,
    distributed: bool,
) -> dict[str, Any]:
    evaluation = run_spec.evaluation
    if evaluation.dataset_path is None or int(evaluation.batches) <= 0:
        raise ValueError("held-out evaluation requires a dataset and positive batches")
    if int(getattr(runtime, "tensor_parallel_size", 1) or 1) != 1:
        raise ValueError(
            "masked-token accuracy currently requires tensor_parallel_size=1"
        )

    evaluation_run_spec = replace(
        run_spec,
        data=replace(
            run_spec.data,
            input_mode="dataset",
            dataset_path=str(evaluation.dataset_path),
            blend_seed=int(evaluation.seed),
            shuffle=True,
        ),
        training=replace(
            run_spec.training,
            gradient_accumulation_steps=1,
        ),
    )
    evaluation_data = executor.build_data_runtime(
        spec=evaluation_run_spec,
        tokenizer=tokenizer,
        vocab_size=int(vocab_size),
        mask_token_id=int(mask_token_id),
        device=device,
        dtype=dtype,
        seed=int(evaluation.seed) + int(rank),
        token_pool=None,
        runtime=runtime,
        rank=int(rank),
        world_size=int(world_size),
    )
    evaluation_task = executor.build_training_task(
        spec=evaluation_run_spec,
        runtime=runtime,
        device=device,
        seed=int(evaluation.seed) + int(rank),
        data_parallel_seed=(
            int(evaluation.seed) + int(getattr(runtime, "sample_parallel_rank", rank))
        ),
        mask_token_id=int(mask_token_id),
        block_size=int(block_size),
        vocab_size=int(vocab_size),
        token_accounting=build_token_accounting_policy(
            spec=evaluation_run_spec,
            runtime=runtime,
            world_size=int(world_size),
            gradient_accumulation_steps=1,
        ),
    )

    was_training = bool(model.training)
    model.eval()
    local_correct = 0
    local_targets = 0
    local_negative_log_likelihood = 0.0
    try:
        with torch.no_grad():
            for _ in range(int(evaluation.batches)):
                batch = evaluation_data.next_batch()
                denominator = evaluation_task.accumulation_loss_denominator([batch])
                prepared = evaluation_task.prepare(
                    batch,
                    loss_denominator=denominator,
                )
                output = evaluation_task.forward(model, prepared)
                correct, targets, negative_log_likelihood = _local_masked_token_metrics(
                    model=model,
                    output=output,
                    labels=prepared.corrupted.labels,
                    mask_token_id=int(mask_token_id),
                    token_tile_size=int(block_size),
                )
                local_correct += int(correct)
                local_targets += int(targets)
                local_negative_log_likelihood += float(negative_log_likelihood)
    finally:
        model.train(was_training)

    counts = torch.tensor(
        [local_correct, local_targets],
        dtype=torch.int64,
        device=device,
    )
    negative_log_likelihood = torch.tensor(
        local_negative_log_likelihood,
        dtype=torch.float64,
        device=device,
    )
    if distributed:
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        dist.all_reduce(negative_log_likelihood, op=dist.ReduceOp.SUM)
    correct = int(counts[0].item())
    targets = int(counts[1].item())
    if targets <= 0:
        raise RuntimeError("held-out evaluation produced no masked target tokens")
    cross_entropy = float(negative_log_likelihood.item()) / float(targets)
    return {
        "dataset_path": str(evaluation.dataset_path),
        "batches": int(evaluation.batches),
        "seed": int(evaluation.seed),
        "masked_token_correct": correct,
        "masked_token_count": targets,
        "masked_token_accuracy": correct / float(targets),
        "masked_token_cross_entropy": cross_entropy,
        "masked_token_perplexity": math.exp(min(cross_entropy, 700.0)),
        "data": evaluation_data.to_log_dict(),
    }


def _local_masked_token_accuracy(
    *,
    model: Any,
    output: tuple[torch.Tensor, torch.Tensor],
    labels: torch.Tensor,
    mask_token_id: int,
    token_tile_size: int,
) -> tuple[int, int]:
    correct, targets, _ = _local_masked_token_metrics(
        model=model,
        output=output,
        labels=labels,
        mask_token_id=mask_token_id,
        token_tile_size=token_tile_size,
    )
    return correct, targets


def _local_masked_token_metrics(
    *,
    model: Any,
    output: Any,
    labels: torch.Tensor,
    mask_token_id: int,
    token_tile_size: int,
) -> tuple[int, int, float]:
    if isinstance(output, tuple):
        active_hidden, active_positions = output
    else:
        active_hidden = getattr(output, "decoder_hidden", None)
        active_positions = getattr(output, "active_positions", None)
        if not isinstance(active_hidden, torch.Tensor) or not isinstance(
            active_positions, torch.Tensor
        ):
            raise TypeError(
                "evaluation output must be a (hidden, positions) tuple or expose "
                "tensor decoder_hidden and active_positions"
            )
    if active_positions.ndim == 2:
        if active_positions.shape[-1] != 2:
            raise ValueError(
                "sequence-parallel active positions must have shape [N, 2]"
            )
        active_labels = labels[
            active_positions[:, 0].to(torch.long),
            active_positions[:, 1].to(torch.long),
        ]
    elif active_positions.ndim == 1:
        active_labels = labels.index_select(1, active_positions.to(torch.long))
    else:
        raise ValueError("active positions must be one- or two-dimensional")

    hidden = active_hidden.reshape(-1, active_hidden.shape[-1])
    target = active_labels.reshape(-1)
    if int(hidden.shape[0]) != int(target.numel()):
        raise RuntimeError("active hidden rows and labels do not align")
    valid = target.ne(-100)
    hidden = hidden[valid]
    target = target[valid]
    if target.numel() == 0:
        return 0, 0, 0.0

    module = model
    while getattr(module, "module", None) is not None:
        module = module.module
    output_head = getattr(module, "output_head", None)
    weight = getattr(output_head, "weight", None)
    bias = getattr(output_head, "bias", None)
    if not isinstance(weight, torch.Tensor):
        raise TypeError("masked-token accuracy requires an output head weight")
    if bias is not None and not isinstance(bias, torch.Tensor):
        raise TypeError("output head bias must be a tensor or None")
    if not 0 <= int(mask_token_id) < int(weight.shape[0]):
        raise ValueError("mask token ID is outside the output vocabulary")

    tile = int(token_tile_size)
    if tile <= 0:
        raise ValueError("token tile size must be positive")
    correct = torch.zeros((), dtype=torch.int64, device=hidden.device)
    negative_log_likelihood = torch.zeros((), dtype=torch.float64, device=hidden.device)
    logit_softcap = getattr(module, "final_logit_softcap", None)
    for start in range(0, int(hidden.shape[0]), tile):
        tile_hidden = hidden.narrow(0, start, min(tile, int(hidden.shape[0]) - start))
        logits = torch.nn.functional.linear(tile_hidden, weight, bias)
        if logit_softcap is not None:
            cap = float(logit_softcap)
            logits = torch.tanh(logits.float() / cap) * cap
        logits[:, int(mask_token_id)] = -torch.inf
        tile_target = target.narrow(0, start, int(logits.shape[0]))
        prediction = logits.argmax(dim=-1)
        correct.add_(prediction.eq(tile_target).sum())
        negative_log_likelihood.add_(
            torch.nn.functional.cross_entropy(
                logits.float(),
                tile_target,
                reduction="sum",
            ).to(torch.float64)
        )
    return (
        int(correct.item()),
        int(target.numel()),
        float(negative_log_likelihood.item()),
    )


def _is_data_sample_source(*, runtime: Any, rank: int) -> bool:
    """Return true for exactly one model rank per independent input sample."""

    plan = getattr(runtime, "plan", None)
    assignments = getattr(plan, "rank_assignments", None)
    if assignments is not None and 0 <= int(rank) < len(assignments):
        assignment = assignments[int(rank)]
        return (
            int(getattr(assignment, "pipeline_parallel_rank", 0)) == 0
            and int(getattr(assignment, "local_parallel_rank", 0)) == 0
            and int(getattr(assignment, "tensor_parallel_rank", 0)) == 0
        )
    return int(rank) == int(getattr(runtime, "model_input_src_rank", 0))


def _step_throughput_metrics(
    *,
    input_tokens: int,
    valid_tokens: int,
    active_tokens: int,
    elapsed_ms: float,
    world_size: int,
    data_parallel_size: int,
) -> dict[str, float]:
    """Build exact input and objective-token rates without another collective."""

    input_tokens = int(input_tokens)
    valid_tokens = int(valid_tokens)
    active_tokens = int(active_tokens)
    elapsed_ms = float(elapsed_ms)
    world_size = int(world_size)
    data_parallel_size = int(data_parallel_size)
    if input_tokens <= 0 or valid_tokens <= 0 or elapsed_ms <= 0:
        return {}
    if world_size <= 0 or data_parallel_size <= 0:
        raise ValueError("throughput world sizes must be positive")
    if active_tokens < 0 or active_tokens > valid_tokens:
        raise RuntimeError("active-token count is outside the valid-token range")
    if valid_tokens > input_tokens:
        raise RuntimeError("supervised-token count exceeds input-token count")
    elapsed_s = elapsed_ms / 1000.0
    unique_rate = valid_tokens / elapsed_s
    per_gpu_rate = unique_rate / float(data_parallel_size)
    input_rate = input_tokens / elapsed_s
    return {
        "step_time_ms": elapsed_ms,
        "steps_per_s": 1.0 / elapsed_s,
        "input_tokens_per_s_global": input_rate,
        "input_tokens_per_s_per_gpu": input_rate / float(world_size),
        "supervised_tokens_per_s_global": unique_rate,
        "supervised_tokens_per_s_per_gpu": unique_rate / float(world_size),
        "tokens_per_s_per_gpu": per_gpu_rate,
        "tokens_per_s_global": per_gpu_rate * float(world_size),
        "unique_tokens_per_s_global": unique_rate,
        "active_tokens_per_s_global": active_tokens / elapsed_s,
        "active_token_fraction": active_tokens / float(valid_tokens),
        "supervised_token_fraction": valid_tokens / float(input_tokens),
    }


def _training_throughput_metrics(
    *,
    rank_metrics: list[dict[str, Any]],
    elapsed_ms: float,
    world_size: int,
    data_parallel_size: int,
) -> dict[str, float]:
    """Summarize the timed window using unique source-rank token counts."""

    measured_steps = {int(item["measured_steps"]) for item in rank_metrics}
    if len(measured_steps) != 1:
        raise RuntimeError("throughput ranks report different measured-step counts")
    steps = next(iter(measured_steps), 0)
    if steps <= 0:
        return {}
    input_tokens = sum(int(item["measured_input_tokens"]) for item in rank_metrics)
    valid_tokens = sum(int(item["measured_valid_tokens"]) for item in rank_metrics)
    active_tokens = sum(int(item["measured_active_tokens"]) for item in rank_metrics)
    metrics = _step_throughput_metrics(
        input_tokens=input_tokens,
        valid_tokens=valid_tokens,
        active_tokens=active_tokens,
        elapsed_ms=float(elapsed_ms) * float(steps),
        world_size=int(world_size),
        data_parallel_size=int(data_parallel_size),
    )
    if metrics:
        metrics["step_time_ms"] = float(elapsed_ms)
        metrics["steps_per_s"] = 1000.0 / float(elapsed_ms)
        metrics["measured_steps"] = float(steps)
        metrics["measured_input_tokens"] = float(input_tokens)
        metrics["measured_valid_tokens"] = float(valid_tokens)
        metrics["measured_active_tokens"] = float(active_tokens)
    return metrics


def _megatron_mfu_metrics(
    *,
    rank_metrics: list[dict[str, Any]],
    model_spec: Any,
    objective_name: str,
    seq_len: int,
    block_size: int,
    vocab_size: int,
    world_size: int,
    elapsed_ms: float,
    dflash_loss_kind: str = "speculators_kl",
    adapter_type: str = "none",
    context_parallel_size: int = 1,
    block_parallel_size: int = 1,
) -> dict[str, Any]:
    """Build one backend-independent Megatron-style MFU summary."""

    family = getattr(model_spec, "family", None)
    if objective_name == "dflash_distillation":
        method = "megatron_dflash_v1"
    elif objective_name == "fast_dllm_v2" and family == "qwen3_8":
        method = "megatron_fast_dllm_v2_qwen3_8_v1"
    elif objective_name == "fast_dllm_v2" and family == "causal_lm":
        method = "megatron_fast_dllm_v2_causal_lm_v1"
    elif objective_name == "diffusiongemma_native_sft" and family == "diffusion_gemma":
        method = "megatron_diffusiongemma_native_all_block_v1"
    elif family == "diffusion_gemma":
        method = "megatron_diffusiongemma_moe_v1"
    elif family == "nemotron_labs_diffusion":
        method = "megatron_block_diffusion_v1"
    else:
        method = "megatron_block_diffusion_unregistered"
    hardware = detect_hardware_preset(allow_default=False)
    peak_flops_per_gpu = peak_flops_for_hardware(hardware)

    def unavailable(reason: str) -> dict[str, Any]:
        return {
            "mfu_pct": None,
            "mfu_hardware": hardware,
            "mfu_peak_flops_per_gpu": peak_flops_per_gpu,
            "mfu_world_size": int(world_size),
            "mfu_method": method,
            "mfu_unavailable_reason": reason,
            "model_flops_per_step": None,
        }

    if elapsed_ms <= 0:
        return unavailable("no positive measured step time")
    if adapter_type != "none":
        return unavailable(
            f"full-training FLOP accounting does not support adapter {adapter_type!r}"
        )
    measured_steps = {int(item["measured_steps"]) for item in rank_metrics}
    if len(measured_steps) != 1 or next(iter(measured_steps)) <= 0:
        raise RuntimeError(
            "MFU requires the same positive measured-step count on every rank"
        )
    step_count = next(iter(measured_steps))

    if objective_name == "dflash_distillation":
        required = (
            getattr(model_spec, "num_layers", None),
            getattr(model_spec, "hidden_size", None),
            getattr(model_spec, "intermediate_size", None),
            getattr(model_spec, "num_attention_heads", None),
            getattr(model_spec, "num_key_value_heads", None),
            getattr(model_spec, "head_dim", None),
            getattr(model_spec, "target_hidden_size", None),
            getattr(model_spec, "target_feature_width", None),
            getattr(model_spec, "draft_vocab_size", None),
            getattr(model_spec, "attention_layer_types", None),
        )
        if getattr(model_spec, "family", None) != "dflash":
            return unavailable(
                "DFlash FLOP accounting requires the dflash model family"
            )
        if any(value is None for value in required):
            return unavailable("DFlash model metadata is incomplete")
        count_names = {
            tuple(item.get("measured_objective_count_names", ()))
            for item in rank_metrics
        }
        if len(count_names) != 1:
            raise RuntimeError("DFlash MFU count schemas differ across ranks")
        names = next(iter(count_names))
        if names != DFLASH_WORK_COUNT_NAMES:
            return unavailable("DFlash useful-work counters are unavailable")
        totals = [0] * len(names)
        for item in rank_metrics:
            values = item.get("measured_objective_counts", ())
            if not values:
                continue
            if len(values) != len(names):
                raise RuntimeError("DFlash MFU count vector has an invalid shape")
            totals = [left + int(right) for left, right in zip(totals, values)]
        counts = {name: total / float(step_count) for name, total in zip(names, totals)}
        if counts["context_token_rows"] <= 0 or counts["valid_anchors"] <= 0:
            return unavailable("no measured DFlash objective work")

        layer_types = tuple(str(kind) for kind in model_spec.attention_layer_types)
        full_layers = sum(kind != "sliding_attention" for kind in layer_types)
        sliding_layers = len(layer_types) - full_layers
        if len(layer_types) != int(model_spec.num_layers):
            raise RuntimeError("DFlash attention-layer metadata is inconsistent")
        valid_anchors = counts["valid_anchors"]
        local_full_pairs = valid_anchors * float(block_size * block_size)
        local_sliding_pairs = (
            local_full_pairs
            if bool(getattr(model_spec, "sliding_window_non_causal", False))
            else valid_anchors * float(block_size * (block_size + 1) // 2)
        )
        attention_pairs = full_layers * (
            counts["full_context_pairs"] + local_full_pairs
        ) + sliding_layers * (counts["sliding_context_pairs"] + local_sliding_pairs)
        context_attention_pairs = (
            full_layers * counts["full_context_pairs"]
            + sliding_layers * counts["sliding_context_pairs"]
        )
        local_attention_pairs = (
            full_layers * local_full_pairs + sliding_layers * local_sliding_pairs
        )
        draft_rows = valid_anchors * float(block_size)
        flops_model = DFlashTransformerFlops(
            num_layers=int(model_spec.num_layers),
            hidden_size=int(model_spec.hidden_size),
            intermediate_size=int(model_spec.intermediate_size),
            num_attention_heads=int(model_spec.num_attention_heads),
            num_key_value_heads=int(model_spec.num_key_value_heads),
            head_dim=int(model_spec.head_dim),
            target_hidden_size=int(model_spec.target_hidden_size),
            target_feature_width=int(model_spec.target_feature_width),
            draft_vocab_size=int(model_spec.draft_vocab_size),
            dflash2_conv_kernel_size=int(
                getattr(model_spec, "dflash2_conv_kernel_size", 0) or 0
            ),
            dflash2_conv_group_size=int(
                getattr(model_spec, "dflash2_conv_group_size", 0) or 0
            ),
            dflash2_selector_rank=int(
                getattr(model_spec, "dflash2_selector_rank", 0) or 0
            ),
            dflash2_selector_top_k=int(
                getattr(model_spec, "dflash2_selector_top_k", 0) or 0
            ),
        )
        breakdown = flops_model.flops_breakdown_per_step(
            context_token_rows=counts["context_token_rows"],
            draft_token_rows=draft_rows,
            attention_pairs=attention_pairs,
            supervised_token_rows=counts["supervised_token_rows"],
            loss_kind=str(dflash_loss_kind),
        )
        model_flops_per_step = sum(breakdown.values())
        mfu_pct = model_flops_utilization_pct(
            model_flops_per_step=model_flops_per_step,
            elapsed_ms=float(elapsed_ms),
            world_size=int(world_size),
            peak_flops_per_gpu=peak_flops_per_gpu,
        )
        executed_breakdown = flops_model.executed_gemm_flops_breakdown_per_step(
            context_token_rows=counts["context_token_rows"],
            draft_token_rows=draft_rows,
            attention_pairs=attention_pairs,
            supervised_token_rows=counts["supervised_token_rows"],
            loss_kind=str(dflash_loss_kind),
            draft_replication=max(
                1.0,
                float(context_parallel_size) / float(max(1, block_parallel_size)),
            ),
            context_attention_pairs=context_attention_pairs,
            local_attention_pairs=local_attention_pairs,
        )
        executed_model_flops_per_step = sum(executed_breakdown.values())
        executed_gemm_mfu_pct = model_flops_utilization_pct(
            model_flops_per_step=executed_model_flops_per_step,
            elapsed_ms=float(elapsed_ms),
            world_size=int(world_size),
            peak_flops_per_gpu=peak_flops_per_gpu,
        )
        return {
            "mfu_pct": mfu_pct,
            "mfu_hardware": hardware,
            "mfu_peak_flops_per_gpu": peak_flops_per_gpu,
            "mfu_world_size": int(world_size),
            "mfu_method": method,
            "mfu_unavailable_reason": None,
            "model_flops_per_step": model_flops_per_step,
            "executed_gemm_mfu_pct": executed_gemm_mfu_pct,
            "executed_gemm_mfu_method": "dflash_executed_gemm_v2",
            "executed_model_flops_per_step": executed_model_flops_per_step,
            "executed_gemm_flops_breakdown": executed_breakdown,
            "mfu_context_token_rows_per_step": counts["context_token_rows"],
            "mfu_draft_token_rows_per_step": draft_rows,
            "mfu_attention_pairs_per_step": attention_pairs,
            "mfu_vocabulary_token_rows_per_step": counts["supervised_token_rows"],
            "mfu_flops_breakdown": breakdown,
        }

    if objective_name == "fast_dllm_v2":
        if family not in {"causal_lm", "qwen3_8"}:
            return unavailable(
                "Fast-dLLM v2 FLOP accounting requires a registered dense "
                "causal-LM conversion family"
            )
        required = (
            getattr(model_spec, "num_layers", None),
            getattr(model_spec, "hidden_size", None),
            getattr(model_spec, "intermediate_size", None),
            getattr(model_spec, "num_attention_heads", None),
            getattr(model_spec, "num_key_value_heads", None),
            getattr(model_spec, "head_dim", None),
        )
        if any(value is None for value in required):
            return unavailable("Fast-dLLM v2 model metadata is incomplete")
        if getattr(model_spec, "num_experts", None) is not None:
            return unavailable(
                "dense Fast-dLLM v2 accounting does not apply to MoE models"
            )

        layer_types_value = getattr(model_spec, "attention_layer_types", None)
        layer_types = (
            tuple(str(value) for value in layer_types_value)
            if layer_types_value is not None
            else ("full_attention",) * int(model_spec.num_layers)
        )
        effective_layers = int(model_spec.num_layers)
        if len(layer_types) < effective_layers:
            return unavailable(
                "Fast-dLLM v2 attention_layer_types does not cover every loaded layer"
            )
        # Smoke recipes may intentionally truncate the checkpoint. The loaded
        # layer count is authoritative for both execution and FLOP accounting.
        layer_types = layer_types[:effective_layers]
        if family == "causal_lm" and "linear_attention" in layer_types:
            return unavailable(
                "generic causal-LM accounting does not support linear-attention layers"
            )

        linear_metadata = (
            getattr(model_spec, "linear_key_head_dim", None),
            getattr(model_spec, "linear_value_head_dim", None),
            getattr(model_spec, "linear_num_key_heads", None),
            getattr(model_spec, "linear_num_value_heads", None),
            getattr(model_spec, "linear_conv_kernel_dim", None),
        )
        if "linear_attention" in layer_types and any(
            value is None for value in linear_metadata
        ):
            return unavailable("Fast-dLLM v2 Gated-DeltaNet metadata is incomplete")

        input_tokens = sum(
            int(item.get("measured_input_tokens", 0)) for item in rank_metrics
        )
        valid_tokens = sum(
            int(item.get("measured_valid_tokens", 0)) for item in rank_metrics
        )
        active_tokens = sum(
            int(item.get("measured_active_tokens", 0)) for item in rank_metrics
        )
        if input_tokens <= 0 or valid_tokens <= 0:
            return unavailable("no measured Fast-dLLM v2 tokens")
        if active_tokens != valid_tokens:
            raise RuntimeError(
                "Fast-dLLM v2 complementary corruption must supervise every valid "
                "shifted token exactly once"
            )

        input_tokens_per_step = input_tokens / float(step_count)
        transformer_token_rows = 2.0 * input_tokens_per_step
        vocabulary_token_rows = active_tokens / float(step_count)
        paired_views_per_step = input_tokens_per_step / float(seq_len)
        full_attention_pairs = paired_views_per_step * block_diffusion_sparse_attention_pairs(
            seq_len=int(seq_len),
            block_size=int(block_size),
        )
        sliding_window = getattr(model_spec, "sliding_window", None)
        if "sliding_attention" in layer_types:
            if sliding_window is None:
                return unavailable(
                    "Fast-dLLM v2 sliding-attention metadata is incomplete"
                )
            sliding_attention_pairs = (
                paired_views_per_step
                * diffusiongemma_block_attention_pairs(
                    seq_len=int(seq_len),
                    block_size=int(block_size),
                    sliding_window=int(sliding_window),
                )
            )
        else:
            sliding_attention_pairs = 0.0

        flops_model = FastDLLMv2TransformerFlops(
            layer_types=layer_types,
            hidden_size=int(model_spec.hidden_size),
            intermediate_size=int(model_spec.intermediate_size),
            num_attention_heads=int(model_spec.num_attention_heads),
            num_key_value_heads=int(model_spec.num_key_value_heads),
            head_dim=int(model_spec.head_dim),
            vocab_size=int(vocab_size),
            attention_output_gate=bool(
                getattr(model_spec, "attention_output_gate", False)
            ),
            linear_key_head_dim=(
                int(linear_metadata[0]) if linear_metadata[0] is not None else None
            ),
            linear_value_head_dim=(
                int(linear_metadata[1]) if linear_metadata[1] is not None else None
            ),
            linear_num_key_heads=(
                int(linear_metadata[2]) if linear_metadata[2] is not None else None
            ),
            linear_num_value_heads=(
                int(linear_metadata[3]) if linear_metadata[3] is not None else None
            ),
            linear_conv_kernel_dim=(
                int(linear_metadata[4]) if linear_metadata[4] is not None else None
            ),
        )
        breakdown = flops_model.flops_breakdown_per_step(
            transformer_token_rows=transformer_token_rows,
            full_attention_pairs=full_attention_pairs,
            sliding_attention_pairs=sliding_attention_pairs,
            vocabulary_token_rows=vocabulary_token_rows,
        )
        model_flops_per_step = sum(breakdown.values())
        mfu_pct = model_flops_utilization_pct(
            model_flops_per_step=model_flops_per_step,
            elapsed_ms=float(elapsed_ms),
            world_size=int(world_size),
            peak_flops_per_gpu=peak_flops_per_gpu,
        )
        return {
            "mfu_pct": mfu_pct,
            "mfu_hardware": hardware,
            "mfu_peak_flops_per_gpu": peak_flops_per_gpu,
            "mfu_world_size": int(world_size),
            "mfu_method": method,
            "mfu_unavailable_reason": None,
            "model_flops_per_step": model_flops_per_step,
            "mfu_transformer_token_rows_per_step": transformer_token_rows,
            "mfu_vocabulary_token_rows_per_step": vocabulary_token_rows,
            "mfu_paired_views_per_step": paired_views_per_step,
            "mfu_full_attention_layers": flops_model.num_full_attention_layers,
            "mfu_sliding_attention_layers": (flops_model.num_sliding_attention_layers),
            "mfu_linear_attention_layers": flops_model.num_linear_attention_layers,
            "mfu_full_attention_pairs_per_layer_per_step": full_attention_pairs,
            "mfu_sliding_attention_pairs_per_layer_per_step": (sliding_attention_pairs),
            "model_flops_breakdown_per_step": breakdown,
        }

    if objective_name == "diffusiongemma_native_sft":
        from dllm_parallel.core.objectives.diffusiongemma import (
            NATIVE_DIFFUSIONGEMMA_WORK_COUNT_NAMES,
        )

        if family != "diffusion_gemma":
            return unavailable(
                "native DiffusionGemma FLOP accounting requires diffusion_gemma"
            )
        layer_types_value = getattr(model_spec, "attention_layer_types", None)
        if layer_types_value is None:
            layer_types_value = getattr(model_spec, "layer_types", None)
        required = (
            getattr(model_spec, "num_layers", None),
            getattr(model_spec, "hidden_size", None),
            getattr(model_spec, "intermediate_size", None),
            getattr(model_spec, "expert_intermediate_size", None),
            getattr(model_spec, "num_attention_heads", None),
            getattr(model_spec, "num_key_value_heads", None),
            getattr(model_spec, "num_global_key_value_heads", None),
            getattr(model_spec, "head_dim", None),
            getattr(model_spec, "global_head_dim", None),
            getattr(model_spec, "num_experts", None),
            getattr(model_spec, "top_k_experts", None),
            getattr(model_spec, "sliding_window", None),
            layer_types_value,
        )
        if any(value is None for value in required):
            return unavailable("DiffusionGemma model metadata is incomplete")
        layer_types = tuple(str(value) for value in layer_types_value)
        if len(layer_types) != int(model_spec.num_layers):
            return unavailable("DiffusionGemma layer types do not match num_layers")
        schemas = {
            tuple(item.get("measured_objective_count_names", ()))
            for item in rank_metrics
        }
        if schemas != {NATIVE_DIFFUSIONGEMMA_WORK_COUNT_NAMES}:
            return unavailable("native DiffusionGemma work counters are unavailable")
        totals = [0] * len(NATIVE_DIFFUSIONGEMMA_WORK_COUNT_NAMES)
        for item in rank_metrics:
            values = item.get("measured_objective_counts", ())
            if values:
                if len(values) != len(totals):
                    raise RuntimeError(
                        "native DiffusionGemma work count vector has an invalid shape"
                    )
                totals = [left + int(right) for left, right in zip(totals, values)]
        counts = {
            name: total / float(step_count)
            for name, total in zip(
                NATIVE_DIFFUSIONGEMMA_WORK_COUNT_NAMES,
                totals,
            )
        }
        if counts["clean_encoder_rows"] <= 0:
            return unavailable("no measured native DiffusionGemma work")
        clean_samples = counts["clean_encoder_rows"] / float(seq_len)
        detached_samples = counts["detached_decoder_rows"] / float(seq_len)
        trained_samples = counts["trained_decoder_rows"] / float(seq_len)
        clean_sliding_per_sample = diffusiongemma_clean_attention_pairs(
            seq_len=int(seq_len),
            sliding_window=int(model_spec.sliding_window),
        )
        clean_full_per_sample = diffusiongemma_clean_attention_pairs(
            seq_len=int(seq_len),
            sliding_window=None,
        )
        total_sliding_per_sample = diffusiongemma_block_attention_pairs(
            seq_len=int(seq_len),
            block_size=int(block_size),
            sliding_window=int(model_spec.sliding_window),
        )
        total_full_per_sample = diffusiongemma_block_attention_pairs(
            seq_len=int(seq_len),
            block_size=int(block_size),
            sliding_window=None,
        )
        decoder_sliding_per_sample = total_sliding_per_sample - clean_sliding_per_sample
        decoder_full_per_sample = total_full_per_sample - clean_full_per_sample
        flops_model = DiffusionGemmaTransformerFlops(
            layer_types=layer_types,
            hidden_size=int(model_spec.hidden_size),
            intermediate_size=int(model_spec.intermediate_size),
            expert_intermediate_size=int(model_spec.expert_intermediate_size),
            num_attention_heads=int(model_spec.num_attention_heads),
            num_key_value_heads=int(model_spec.num_key_value_heads),
            num_global_key_value_heads=int(model_spec.num_global_key_value_heads),
            head_dim=int(model_spec.head_dim),
            global_head_dim=int(model_spec.global_head_dim),
            num_experts=int(model_spec.num_experts),
            top_k_experts=int(model_spec.top_k_experts),
            vocab_size=int(vocab_size),
        )
        breakdown = flops_model.native_flops_breakdown_per_step(
            clean_encoder_rows=counts["clean_encoder_rows"],
            detached_decoder_rows=counts["detached_decoder_rows"],
            trained_decoder_rows=counts["trained_decoder_rows"],
            clean_sliding_attention_pairs=(clean_samples * clean_sliding_per_sample),
            clean_full_attention_pairs=clean_samples * clean_full_per_sample,
            detached_sliding_attention_pairs=(
                detached_samples * decoder_sliding_per_sample
            ),
            detached_full_attention_pairs=(detached_samples * decoder_full_per_sample),
            trained_sliding_attention_pairs=(
                trained_samples * decoder_sliding_per_sample
            ),
            trained_full_attention_pairs=(trained_samples * decoder_full_per_sample),
            self_conditioning_vocabulary_rows=(
                counts["self_conditioning_vocabulary_rows"]
            ),
            decoder_loss_vocabulary_rows=(counts["decoder_loss_vocabulary_rows"]),
            encoder_ar_vocabulary_rows=counts["encoder_ar_vocabulary_rows"],
        )
        model_flops_per_step = sum(breakdown.values())
        mfu_pct = model_flops_utilization_pct(
            model_flops_per_step=model_flops_per_step,
            elapsed_ms=float(elapsed_ms),
            world_size=int(world_size),
            peak_flops_per_gpu=peak_flops_per_gpu,
        )
        return {
            "mfu_pct": mfu_pct,
            "mfu_hardware": hardware,
            "mfu_peak_flops_per_gpu": peak_flops_per_gpu,
            "mfu_world_size": int(world_size),
            "mfu_method": method,
            "mfu_unavailable_reason": None,
            "model_flops_per_step": model_flops_per_step,
            "model_flops_breakdown_per_step": breakdown,
            **{f"mfu_{name}_per_step": value for name, value in counts.items()},
        }

    if objective_name != "standard_block_diffusion":
        return unavailable(f"unsupported objective {objective_name!r}")
    if family not in {"nemotron_labs_diffusion", "diffusion_gemma"}:
        return unavailable(
            f"exact FLOP accounting is not registered for model family {family!r}"
        )
    valid_tokens = sum(int(item["measured_valid_tokens"]) for item in rank_metrics)
    active_tokens = sum(int(item["measured_active_tokens"]) for item in rank_metrics)
    if valid_tokens <= 0:
        return unavailable("no measured standard block-diffusion tokens")
    if active_tokens < 0 or active_tokens > valid_tokens:
        raise RuntimeError(
            "measured active-token count is outside the valid-token range"
        )
    valid_tokens_per_step = valid_tokens / float(step_count)
    active_tokens_per_step = active_tokens / float(step_count)
    samples_per_step = valid_tokens_per_step / float(seq_len)
    transformer_token_rows = 2.0 * valid_tokens_per_step
    family_metrics: dict[str, Any]

    if family == "nemotron_labs_diffusion":
        required = (
            getattr(model_spec, "num_layers", None),
            getattr(model_spec, "hidden_size", None),
            getattr(model_spec, "intermediate_size", None),
            getattr(model_spec, "num_attention_heads", None),
            getattr(model_spec, "num_key_value_heads", None),
            getattr(model_spec, "head_dim", None),
        )
        if getattr(model_spec, "num_experts", None) is not None:
            return unavailable("dense Nemotron accounting does not apply to MoE models")
        if any(value is None for value in required):
            return unavailable("model metadata is incomplete")
        attention_pairs = samples_per_step * block_diffusion_sparse_attention_pairs(
            seq_len=int(seq_len),
            block_size=int(block_size),
        )
        flops_model = MegatronTransformerFlops(
            num_layers=int(model_spec.num_layers),
            hidden_size=int(model_spec.hidden_size),
            intermediate_size=int(model_spec.intermediate_size),
            num_attention_heads=int(model_spec.num_attention_heads),
            num_key_value_heads=int(model_spec.num_key_value_heads),
            head_dim=int(model_spec.head_dim),
            vocab_size=int(vocab_size),
        )
        model_flops_per_step = flops_model.flops_per_step(
            transformer_token_rows=transformer_token_rows,
            attention_pairs=attention_pairs,
            vocabulary_token_rows=active_tokens_per_step,
        )
        family_metrics = {"mfu_attention_pairs_per_step": attention_pairs}
    else:
        layer_types = getattr(model_spec, "attention_layer_types", None)
        if layer_types is None:
            layer_types = getattr(model_spec, "layer_types", None)
        required = (
            getattr(model_spec, "num_layers", None),
            getattr(model_spec, "hidden_size", None),
            getattr(model_spec, "intermediate_size", None),
            getattr(model_spec, "expert_intermediate_size", None),
            getattr(model_spec, "num_attention_heads", None),
            getattr(model_spec, "num_key_value_heads", None),
            getattr(model_spec, "num_global_key_value_heads", None),
            getattr(model_spec, "head_dim", None),
            getattr(model_spec, "global_head_dim", None),
            getattr(model_spec, "num_experts", None),
            getattr(model_spec, "top_k_experts", None),
            getattr(model_spec, "sliding_window", None),
            layer_types,
        )
        if any(value is None for value in required):
            return unavailable("DiffusionGemma model metadata is incomplete")
        layer_types = tuple(str(item) for item in layer_types)
        if len(layer_types) != int(model_spec.num_layers):
            return unavailable("DiffusionGemma layer types do not match num_layers")
        sliding_attention_pairs = (
            samples_per_step
            * diffusiongemma_block_attention_pairs(
                seq_len=int(seq_len),
                block_size=int(block_size),
                sliding_window=int(model_spec.sliding_window),
            )
        )
        full_attention_pairs = samples_per_step * diffusiongemma_block_attention_pairs(
            seq_len=int(seq_len),
            block_size=int(block_size),
            sliding_window=None,
        )
        flops_model = DiffusionGemmaTransformerFlops(
            layer_types=layer_types,
            hidden_size=int(model_spec.hidden_size),
            intermediate_size=int(model_spec.intermediate_size),
            expert_intermediate_size=int(model_spec.expert_intermediate_size),
            num_attention_heads=int(model_spec.num_attention_heads),
            num_key_value_heads=int(model_spec.num_key_value_heads),
            num_global_key_value_heads=int(model_spec.num_global_key_value_heads),
            head_dim=int(model_spec.head_dim),
            global_head_dim=int(model_spec.global_head_dim),
            num_experts=int(model_spec.num_experts),
            top_k_experts=int(model_spec.top_k_experts),
            vocab_size=int(vocab_size),
        )
        breakdown = flops_model.flops_breakdown_per_step(
            transformer_token_rows=transformer_token_rows,
            sliding_attention_pairs=sliding_attention_pairs,
            full_attention_pairs=full_attention_pairs,
            self_conditioning_token_rows=valid_tokens_per_step,
            vocabulary_token_rows=active_tokens_per_step,
        )
        model_flops_per_step = sum(breakdown.values())
        family_metrics = {
            "mfu_sliding_attention_layers": flops_model.num_sliding_layers,
            "mfu_full_attention_layers": flops_model.num_full_layers,
            "mfu_sliding_attention_pairs_per_step": sliding_attention_pairs,
            "mfu_full_attention_pairs_per_step": full_attention_pairs,
            "mfu_self_conditioning_token_rows_per_step": valid_tokens_per_step,
            "mfu_total_experts": int(model_spec.num_experts),
            "mfu_active_experts_per_token": int(model_spec.top_k_experts),
            "model_flops_breakdown_per_step": breakdown,
        }
    mfu_pct = model_flops_utilization_pct(
        model_flops_per_step=model_flops_per_step,
        elapsed_ms=float(elapsed_ms),
        world_size=int(world_size),
        peak_flops_per_gpu=peak_flops_per_gpu,
    )
    return {
        "mfu_pct": mfu_pct,
        "mfu_hardware": hardware,
        "mfu_peak_flops_per_gpu": peak_flops_per_gpu,
        "mfu_world_size": int(world_size),
        "mfu_method": method,
        "mfu_unavailable_reason": None,
        "model_flops_per_step": model_flops_per_step,
        "mfu_transformer_token_rows_per_step": transformer_token_rows,
        "mfu_vocabulary_token_rows_per_step": active_tokens_per_step,
        **family_metrics,
    }


def _init_distributed() -> bool:
    global _CPU_OBJECT_GROUP
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            "nccl",
            device_id=torch.device("cuda", local_rank),
        )
        _CPU_OBJECT_GROUP = dist.new_group(backend="gloo")
    else:
        dist.init_process_group("gloo")
        _CPU_OBJECT_GROUP = dist.group.WORLD
    return True


def _object_gather_group() -> Any:
    return _CPU_OBJECT_GROUP if _CPU_OBJECT_GROUP is not None else dist.group.WORLD


def _trace(enabled: bool, rank: int, step: int, phase: str, **extra: Any) -> None:
    if not bool(enabled):
        return
    payload = {
        "event": "trace",
        "rank": int(rank),
        "step": int(step),
        "phase": phase,
        "time": time.perf_counter(),
        **extra,
    }
    print(json.dumps(payload, sort_keys=True), flush=True)


def _distributed_duration_checkpoint_control(
    *,
    spec: RunSpec,
    elapsed_seconds: float,
    saved_fractions: set[float] | frozenset[float],
    rank: int,
    distributed: bool,
    control: torch.Tensor | None,
) -> tuple[bool, tuple[float, ...]]:
    """Broadcast rank zero's duration/save decision to every training rank.

    Local monotonic clocks and startup timing are not bit-identical.  Branching
    into a collective checkpoint from each rank's local elapsed time can put a
    subset of ranks in the checkpoint barrier while the rest enter the next
    model collective.  Encode both the stop bit and every configured duration
    checkpoint bit in one rank-zero-authored control tensor instead.
    """

    fractions = tuple(float(x) for x in spec.checkpointing.save_duration_fractions)
    if int(rank) == 0:
        duration = spec.training.max_duration_seconds
        reached = duration is not None and float(elapsed_seconds) >= float(duration)
        due = due_duration_checkpoint_fractions(
            spec=spec,
            elapsed_seconds=float(elapsed_seconds),
            saved_fractions=saved_fractions,
        )
    else:
        reached = False
        due = ()
    if not distributed:
        return reached, due
    if control is None or control.numel() != 1 + len(fractions):
        raise RuntimeError("distributed duration control has an invalid shape")
    control.zero_()
    if int(rank) == 0:
        control[0] = int(reached)
        due_set = set(due)
        for index, fraction in enumerate(fractions, 1):
            control[index] = int(fraction in due_set)
    dist.broadcast(control, src=0)
    synchronized_due = tuple(
        fraction
        for index, fraction in enumerate(fractions, 1)
        if bool(control[index].item())
    )
    return bool(control[0].item()), synchronized_due


def _startup_event(rank: int, phase: str, **extra: Any) -> None:
    if int(rank) != 0:
        return
    print(
        json.dumps(
            {
                "event": "startup",
                "phase": phase,
                **extra,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _gradient_accumulation_context(
    module: Any,
    *,
    optimizer_backend: str,
    accumulation_steps: int,
):
    if int(accumulation_steps) <= 1 or optimizer_backend != "deepspeed_zero2":
        return contextlib.nullcontext()
    coalesce_grad_reduction = getattr(module, "coalesce_grad_reduction", None)
    if not callable(coalesce_grad_reduction):
        raise RuntimeError(
            "gradient accumulation with DeepSpeed ZeRO requires DeepSpeed >= 0.19.2"
        )
    return coalesce_grad_reduction()


@contextlib.contextmanager
def _fsdp2_no_sync_context(module: Any):
    module.set_requires_gradient_sync(False)
    try:
        yield
    finally:
        module.set_requires_gradient_sync(True)


def _microbatch_no_sync_context(
    module: Any,
    *,
    optimizer_backend: str,
    enabled: bool,
):
    if not enabled:
        return contextlib.nullcontext()
    if optimizer_backend == "fsdp":
        return module.no_sync()
    if optimizer_backend == "fsdp2":
        return _fsdp2_no_sync_context(module)
    return contextlib.nullcontext()


def _dtensor_local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    to_local = getattr(tensor, "to_local", None)
    if to_local is None:
        return tensor
    return to_local()


def _validate_objective_family(objective_spec: Any, *, family: str) -> None:
    """Validate the objective after ``auto``/``hf`` family resolution."""

    validate_objective_for_family(
        family=str(family),
        objective=str(getattr(objective_spec, "name", "")),
    )


def _resolve_model_family(spec: RunSpec) -> str:
    requested = str(spec.model.family)
    if requested == "hf" or requested in supported_families():
        if spec.objective.name == "fast_dllm_v2" and requested == "hf":
            return "causal_lm"
        return requested
    if spec.objective.name == "fast_dllm_v2":
        model_id = str(spec.model.id).lower()
        if model_id.startswith("qwen/qwen3.8-"):
            return "qwen3_8"
        return "causal_lm"
    return "hf"


def _load_training_model_config(
    spec: RunSpec,
    *,
    family: str,
) -> _TrainingModelConfig:
    model_spec = spec.model
    if family == "dflash":
        executor = executor_for_family("dflash")
        config = executor.load_config_from_run_spec(spec)
        return _TrainingModelConfig(
            config=config,
            spec=executor.metadata(config, model_id=str(model_spec.id)),
            family="dflash",
            hf=False,
        )
    if family != "hf" and family not in supported_families():
        raise ValueError(f"unsupported model family: {family}")
    bundle = load_hf_config(
        model_spec.id,
        trust_remote_code=model_spec.trust_remote_code,
        revision=model_spec.revision,
        family=(None if family == "hf" else family),
    )
    executor = executor_for_family(bundle.spec.family)
    migrate_config = getattr(executor, "migrate_config", None)
    config = (
        migrate_config(bundle.config) if callable(migrate_config) else bundle.config
    )
    model_family_spec = executor.metadata(config, model_id=str(model_spec.id))
    return _TrainingModelConfig(
        config=config,
        spec=model_family_spec,
        family=model_family_spec.family,
        hf=True,
    )


def _runtime_kv_backend(spec: RunSpec, *, family: str) -> str:
    del family
    if int(spec.topology.context_parallel_size) == 1:
        return "replicated"
    return "ring"


def _import_transformers() -> Any:
    try:
        import transformers
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for Hugging Face block-diffusion models"
        ) from exc
    return transformers


def _first_existing_attr(value: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        candidate = getattr(value, name, None)
        if candidate is not None:
            return candidate
    raise TypeError(f"expected one of {', '.join(names)} on {value.__class__.__name__}")


def _optional_first_existing_attr(value: Any, names: tuple[str, ...]) -> Any | None:
    for name in names:
        candidate = getattr(value, name, None)
        if candidate is not None:
            return candidate
    return None


def _config_value(config: Any, name: str, *, nested: str | None = None) -> Any | None:
    if isinstance(config, dict):
        value = config.get(name)
        child = config.get(nested) if nested is not None else None
        if value is not None or child is None:
            return value
        return (
            child.get(name) if isinstance(child, dict) else getattr(child, name, None)
        )
    value = getattr(config, name, None)
    if value is not None or nested is None:
        return value
    child = getattr(config, nested, None)
    if child is None:
        return None
    return getattr(child, name, None)


def _set_config_value(
    config: Any,
    name: str,
    value: Any,
    *,
    nested: str | None = None,
    create: bool = False,
) -> None:
    if isinstance(config, dict):
        if create or name in config:
            config[name] = value
        child = config.get(nested) if nested is not None else None
        if isinstance(child, dict) and (create or name in child):
            child[name] = value
        return
    if create or hasattr(config, name):
        setattr(config, name, value)
    child = getattr(config, nested, None) if nested is not None else None
    if child is not None and (create or hasattr(child, name)):
        setattr(child, name, value)


def _apply_model_config_overrides(config: Any, model_spec: Any) -> None:
    if model_spec.mask_token_id is not None:
        _set_config_value(
            config,
            "mask_token_id",
            int(model_spec.mask_token_id),
            nested="text_config",
            create=True,
        )
    if model_spec.max_position_embeddings is not None:
        _set_config_value(
            config,
            "max_position_embeddings",
            int(model_spec.max_position_embeddings),
            nested="text_config",
        )
    if model_spec.max_layers is not None:
        _set_config_value(
            config,
            "num_hidden_layers",
            int(model_spec.max_layers),
            nested="text_config",
        )


def _apply_loaded_model_config_overrides(model: Any, model_spec: Any) -> None:
    if (
        model_spec.max_layers is None
        and model_spec.max_position_embeddings is None
        and model_spec.mask_token_id is None
    ):
        return
    config = getattr(model, "config", None)
    if config is not None:
        _apply_model_config_overrides(config, model_spec)
    encoder = _optional_first_existing_attr(
        model,
        ("encoder", "model", "transformer", "language_model", "decoder"),
    )
    encoder_config = getattr(encoder, "config", None)
    if encoder_config is not None:
        _apply_model_config_overrides(encoder_config, model_spec)


def _tokenizer_mask_token_id(tokenizer: Any | None) -> int | None:
    if tokenizer is None:
        return None
    value = getattr(tokenizer, "mask_token_id", None)
    if value is not None:
        return int(value)
    mask_token = getattr(tokenizer, "mask_token", None)
    converter = getattr(tokenizer, "convert_tokens_to_ids", None)
    if mask_token is None or not callable(converter):
        return None
    converted = converter(mask_token)
    if converted is None:
        return None
    return int(converted)


def _mask_token_id(config: Any, *, tokenizer: Any | None = None) -> int:
    value = _config_value(config, "mask_token_id", nested="text_config")
    if value is None:
        value = _tokenizer_mask_token_id(tokenizer)
    if value is None:
        raise RuntimeError("diffusion config must expose mask_token_id")
    value = int(value)
    vocab_size = _vocab_size(config, tokenizer=tokenizer, required=False)
    if vocab_size is not None and not (0 <= value < int(vocab_size)):
        raise RuntimeError(
            "diffusion config exposes an invalid mask_token_id: "
            f"{value} for vocab_size={int(vocab_size)}"
        )
    if value < 0:
        raise RuntimeError(
            f"diffusion config exposes an invalid mask_token_id: {value}"
        )
    return value


def _vocab_size(
    config: Any, *, tokenizer: Any | None = None, required: bool = True
) -> int | None:
    value = _config_value(config, "vocab_size", nested="text_config")
    if value is None and tokenizer is not None:
        tokenizer_vocab_size = getattr(tokenizer, "vocab_size", None)
        if tokenizer_vocab_size is not None:
            value = tokenizer_vocab_size
        else:
            try:
                value = len(tokenizer)
            except TypeError:
                value = None
    if value is None:
        if required:
            raise RuntimeError("diffusion config must expose vocab_size")
        return None
    value = int(value)
    if value <= 0:
        raise RuntimeError(f"diffusion config exposes an invalid vocab_size: {value}")
    return value


def _resolve_block_size(cli_block_size: int | None, config: Any) -> int:
    value = cli_block_size
    if value is None:
        value = getattr(config, "block_size", None)
    if value is None:
        value = getattr(config, "canvas_length", None)
    if value is None:
        value = _config_value(config, "block_size", nested="text_config")
    if value is None:
        raise RuntimeError(
            "standard block diffusion training requires an explicit block size; "
            "pass --block-size or provide config.block_size"
        )
    value = int(value)
    if value <= 0:
        raise ValueError("block_size must be positive")
    return value


def _synthetic_random_token_pool(
    model: Any,
    *,
    runtime: Any | None,
    vocab_size: int,
    sample_vocab_size: int | None,
    mask_token_id: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Return legal token ids for synthetic random profiling.

    Real training data/tokenizers already avoid reserved ids. Synthetic random
    profiling must do the same, otherwise checkpoints with nonfinite reserved
    embedding rows can produce NaNs unrelated to the training backend.
    """

    embedding = _input_embedding_module(model)
    weight = getattr(embedding, "weight", None)
    if not torch.is_tensor(weight):
        return None
    local_weight = _dtensor_local_tensor(weight.detach())
    if local_weight.ndim != 2 or int(local_weight.shape[0]) <= 0:
        return None
    limit = (
        vocab_size
        if sample_vocab_size is None
        else min(int(sample_vocab_size), vocab_size)
    )
    limit = int(limit)
    if limit <= 1:
        raise ValueError("sample vocabulary size must be greater than one")
    with torch.no_grad():
        shard_start = getattr(embedding, "vocab_start", None)
        shard_stop = getattr(embedding, "vocab_stop", None)
        if shard_start is None or shard_stop is None:
            if int(local_weight.shape[0]) < limit:
                raise RuntimeError(
                    "synthetic random profiling found a sharded embedding without "
                    "vocab_start/vocab_stop metadata"
                )
            finite_rows = torch.isfinite(local_weight[:limit]).all(dim=1)
        else:
            finite_rows = torch.zeros(
                limit,
                device=local_weight.device,
                dtype=torch.int32,
            )
            shard_start = int(shard_start)
            shard_stop = int(shard_stop)
            local_start = max(0, shard_start)
            local_stop = min(limit, shard_stop)
            if local_stop > local_start:
                offset = local_start - shard_start
                local_finite = torch.isfinite(
                    local_weight[offset : offset + (local_stop - local_start)]
                ).all(dim=1)
                finite_rows[local_start:local_stop] = local_finite.to(torch.int32)
            tp_size = int(getattr(runtime, "tensor_parallel_size", 1) or 1)
            tp_group = getattr(runtime, "tensor_parallel_group", None)
            if tp_size > 1:
                if tp_group is None:
                    raise RuntimeError(
                        "synthetic random profiling requires a tensor-parallel "
                        "group for vocab-parallel embeddings"
                    )
                dist.all_reduce(finite_rows, op=dist.ReduceOp.SUM, group=tp_group)
            finite_rows = finite_rows > 0
        if 0 <= int(mask_token_id) < limit:
            finite_rows[int(mask_token_id)] = False
        if (
            bool(finite_rows.all())
            and sample_vocab_size is None
            and limit == int(vocab_size)
        ):
            return None
        token_pool = torch.nonzero(finite_rows, as_tuple=False).flatten()
    if token_pool.numel() <= 0:
        raise RuntimeError("synthetic random profiling found no finite token ids")
    return token_pool.to(device=device, dtype=torch.long, non_blocking=True)


def _input_embedding_module(model: Any) -> Any:
    return _input_embedding_module_impl(model, seen=set())


def _input_embedding_module_impl(model: Any, *, seen: set[int]) -> Any:
    object_id = id(model)
    if object_id in seen:
        return None
    seen.add(object_id)
    embedding = getattr(model, "embed_tokens", None)
    if embedding is not None:
        return embedding
    for root_name in ("module", "model", "decoder", "encoder", "transformer"):
        root = getattr(model, root_name, None)
        if root is None:
            continue
        embedding = _input_embedding_module_impl(root, seen=seen)
        if embedding is not None:
            return embedding
        embedding = getattr(root, "embed_tokens", None)
        if embedding is not None:
            return embedding
        get_input_embeddings = getattr(root, "get_input_embeddings", None)
        if callable(get_input_embeddings):
            try:
                embedding = get_input_embeddings()
            except AttributeError:
                embedding = None
            if embedding is not None:
                return embedding
    get_input_embeddings = getattr(model, "get_input_embeddings", None)
    if callable(get_input_embeddings):
        try:
            embedding = get_input_embeddings()
        except AttributeError:
            embedding = None
        if embedding is not None:
            return embedding
    return None


def _print_first_nonfinite_gradient(
    model: torch.nn.Module,
    rank: int,
    *,
    prefix: str = "grad",
) -> None:
    missing: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        grad = parameter.grad
        grad_source = "grad"
        if grad is None:
            grad = getattr(parameter, "grad_accum", None)
            grad_source = "grad_accum"
        if grad is None:
            missing.append(name)
            continue
        grad = _dtensor_local_tensor(grad)
        finite = torch.isfinite(grad)
        if not finite.all():
            finite_values = grad[finite]
            max_abs_finite = (
                float(finite_values.detach().abs().max().cpu())
                if finite_values.numel() > 0
                else float("nan")
            )
            flat_bad = torch.nonzero(~finite.flatten(), as_tuple=False).flatten()
            first_bad = int(flat_bad[0].detach().cpu()) if flat_bad.numel() else -1
            print(
                json.dumps(
                    {
                        "event": "nonfinite_gradient",
                        "prefix": prefix,
                        "rank": rank,
                        "name": name,
                        "source": grad_source,
                        "finite": int(finite.sum().item()),
                        "numel": grad.numel(),
                        "nan": int(torch.isnan(grad).sum().item()),
                        "posinf": int(torch.isposinf(grad).sum().item()),
                        "neginf": int(torch.isneginf(grad).sum().item()),
                        "max_abs_finite": max_abs_finite,
                        "first_bad_flat_index": first_bad,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return
    if rank == 0:
        event = (
            "gradient_storage_unavailable"
            if missing
            else "all_gradients_finite"
        )
        print(
            json.dumps(
                {
                    "event": event,
                    "prefix": prefix,
                    "missing_count": len(missing),
                    "first_missing": missing[:8],
                },
                sort_keys=True,
            ),
            flush=True,
        )


def _attach_output_gradient_probe(output: Any, rank: int) -> None:
    tensor = output[0] if isinstance(output, tuple) and output else output
    if not isinstance(tensor, torch.Tensor):
        tensor = getattr(output, "active_hidden", tensor)
    if not isinstance(tensor, torch.Tensor) or not tensor.requires_grad:
        return

    def report(gradient: torch.Tensor) -> torch.Tensor:
        local = _dtensor_local_tensor(gradient)
        finite = torch.isfinite(local)
        if not bool(finite.all()):
            _print_nonfinite_tensor_event(
                event="nonfinite_training_output_gradient",
                rank=rank,
                name="training_output",
                tensor=local,
                finite=finite,
            )
        elif rank == 0:
            print(
                json.dumps(
                    {
                        "event": "training_output_gradient_finite",
                        "max_abs": float(local.detach().abs().max().cpu()),
                        "shape": list(local.shape),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        return gradient

    tensor.register_hook(report)


def _print_first_nonfinite_parameter(model: torch.nn.Module, rank: int) -> None:
    for name, parameter in model.named_parameters():
        local_parameter = _dtensor_local_tensor(parameter)
        if not torch.isfinite(local_parameter).all():
            finite = torch.isfinite(local_parameter)
            print(
                json.dumps(
                    {
                        "event": "nonfinite_parameter",
                        "rank": rank,
                        "name": name,
                        "finite": int(finite.sum().item()),
                        "numel": local_parameter.numel(),
                        "nan": int(torch.isnan(local_parameter).sum().item()),
                        "posinf": int(torch.isposinf(local_parameter).sum().item()),
                        "neginf": int(torch.isneginf(local_parameter).sum().item()),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return
    if rank == 0:
        print(json.dumps({"event": "all_parameters_finite"}), flush=True)


def _print_first_nonfinite_zero_optimizer_state(zero_optimizer: Any, rank: int) -> None:
    for group_idx, partition in enumerate(
        getattr(zero_optimizer, "single_partition_of_fp32_groups", ())
    ):
        tensor = _dtensor_local_tensor(partition)
        finite = torch.isfinite(tensor)
        if not finite.all():
            _print_nonfinite_tensor_event(
                event="nonfinite_zero_fp32_partition",
                rank=rank,
                name=f"group_{group_idx}",
                tensor=tensor,
                finite=finite,
            )
            return
    inner_optimizer = getattr(zero_optimizer, "optimizer", None)
    state = getattr(inner_optimizer, "state", {}) if inner_optimizer is not None else {}
    for param_idx, values in enumerate(state.values()):
        if not isinstance(values, dict):
            continue
        for state_name, value in values.items():
            if not torch.is_tensor(value):
                continue
            tensor = _dtensor_local_tensor(value)
            finite = torch.isfinite(tensor)
            if not finite.all():
                _print_nonfinite_tensor_event(
                    event="nonfinite_zero_optimizer_state",
                    rank=rank,
                    name=f"param_{param_idx}.{state_name}",
                    tensor=tensor,
                    finite=finite,
                )
                return
    if rank == 0:
        print(json.dumps({"event": "all_zero_optimizer_state_finite"}), flush=True)


def _print_first_nonfinite_zero_gradient(zero_optimizer: Any, rank: int) -> None:
    from dllm_parallel.core.parallel.deepspeed import _iter_zero_gradient_tensors

    for grad_idx, grad in enumerate(_iter_zero_gradient_tensors(zero_optimizer)):
        tensor = _dtensor_local_tensor(grad)
        finite = torch.isfinite(tensor)
        if not finite.all():
            _print_nonfinite_tensor_event(
                event="nonfinite_zero_gradient",
                rank=rank,
                name=f"grad_{grad_idx}",
                tensor=tensor,
                finite=finite,
            )
            return
    if rank == 0:
        print(json.dumps({"event": "all_zero_gradients_finite"}), flush=True)


def _print_nonfinite_tensor_event(
    *,
    event: str,
    rank: int,
    name: str,
    tensor: torch.Tensor,
    finite: torch.Tensor,
) -> None:
    finite_values = tensor[finite]
    max_abs_finite = (
        float(finite_values.detach().abs().max().cpu())
        if finite_values.numel() > 0
        else float("nan")
    )
    flat_bad = torch.nonzero(~finite.flatten(), as_tuple=False).flatten()
    first_bad = int(flat_bad[0].detach().cpu()) if flat_bad.numel() else -1
    print(
        json.dumps(
            {
                "event": event,
                "rank": rank,
                "name": name,
                "finite": int(finite.sum().item()),
                "numel": tensor.numel(),
                "nan": int(torch.isnan(tensor).sum().item()),
                "posinf": int(torch.isposinf(tensor).sum().item()),
                "neginf": int(torch.isneginf(tensor).sum().item()),
                "max_abs_finite": max_abs_finite,
                "first_bad_flat_index": first_bad,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
