# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from dllm_parallel.core.parallel.planner import (
    DistributedTrainingPlan,
    build_distributed_training_plan,
)
from dllm_parallel.core.parallel.topology import (
    ParallelPlan,
    RankAssignment,
    build_parallel_plan,
)
from dllm_parallel.core.parallel.hf_tensor_parallel import (
    build_hf_tensor_parallel_device_mesh,
    configure_hf_config_for_tensor_parallel,
    hf_from_pretrained_tensor_parallel_kwargs,
    validate_hf_tensor_parallel_model,
)
from dllm_parallel.core.parallel.preflight import (
    ParallelMeshSizes,
    infer_parallel_mesh_sizes,
    launch_environment_metadata,
    prepare_parallel_environment,
    require_idle_assigned_gpus,
    validate_supported_training_axes,
)
import importlib

__all__ = [
    "DistributedTrainingPlan",
    "DLLMProcessGroupCollection",
    "ParallelPlan",
    "ParallelMeshSizes",
    "RankAssignment",
    "TensorParallelLayerConfig",
    "VocabParallelOutput",
    "all_reduce_model_parallel_gradients",
    "all_reduce_sequence_parallel_replicated_gradients",
    "build_distributed_training_plan",
    "build_hf_tensor_parallel_device_mesh",
    "build_parallel_plan",
    "destroy_parallel_runtime_process_groups",
    "configure_hf_config_for_tensor_parallel",
    "hf_from_pretrained_tensor_parallel_kwargs",
    "infer_parallel_mesh_sizes",
    "launch_environment_metadata",
    "prepare_parallel_environment",
    "require_idle_assigned_gpus",
    "tensor_parallel_layers",
    "validate_hf_tensor_parallel_model",
    "validate_supported_training_axes",
]


def __getattr__(name: str):
    if name in {
        "all_reduce_model_parallel_gradients",
        "all_reduce_sequence_parallel_replicated_gradients",
    }:
        from dllm_parallel.core.parallel import grad_sync

        return getattr(grad_sync, name)
    if name == "DLLMProcessGroupCollection":
        from dllm_parallel.core.parallel.groups import DLLMProcessGroupCollection

        return DLLMProcessGroupCollection
    if name == "destroy_parallel_runtime_process_groups":
        from dllm_parallel.core.parallel.runtime import destroy_parallel_runtime_process_groups

        return destroy_parallel_runtime_process_groups
    if name in {"TensorParallelLayerConfig", "VocabParallelOutput"}:
        tensor_parallel_layers = importlib.import_module(
            "dllm_parallel.core.parallel.tensor_parallel_layers"
        )

        return getattr(tensor_parallel_layers, name)
    if name == "tensor_parallel_layers":
        return importlib.import_module("dllm_parallel.core.parallel.tensor_parallel_layers")
    raise AttributeError(name)
