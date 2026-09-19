# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Hugging Face tensor-parallel loading helpers.

This module is the shared front door for HF/DTensor tensor parallelism.  The
model-family backbone still owns capability validation and packed objective
execution; this module only translates a DLLM runtime into the HF
``tp_plan``/``DeviceMesh`` contract and validates that Transformers actually
attached a TP model.
"""

from __future__ import annotations

from typing import Any


DEFAULT_VOCAB_PARALLEL_OUTPUTS = ("diffusion_head", "lm_head", "embed_out")


def configure_hf_config_for_tensor_parallel(
    config: Any,
    runtime: Any,
    *,
    vocab_parallel_outputs: tuple[str, ...] = DEFAULT_VOCAB_PARALLEL_OUTPUTS,
) -> None:
    """Add DLLM-required TP hints to a mutable HF config object."""

    tp_size = int(getattr(runtime, "tensor_parallel_size", 1) or 1)
    if tp_size <= 1:
        return
    if getattr(config, "model_type", None) == "diffusion_gemma":
        text_config = getattr(config, "text_config", None)
        text_plan = getattr(text_config, "base_model_tp_plan", None)
        if not isinstance(text_plan, dict) or not text_plan:
            raise ValueError(
                "DiffusionGemma tensor parallelism requires a text-model TP plan"
            )
        config.base_model_tp_plan = dict(text_plan)
        if hasattr(config, "use_cache"):
            config.use_cache = False
        return
    tp_plan = dict(getattr(config, "base_model_tp_plan", None) or {})
    for output_name in vocab_parallel_outputs:
        tp_plan.setdefault(output_name, "colwise")
    config.base_model_tp_plan = tp_plan
    if hasattr(config, "use_cache"):
        config.use_cache = False

def build_hf_tensor_parallel_device_mesh(
    runtime: Any,
    *,
    device_type: str,
) -> Any | None:
    """Build a 1-D HF DeviceMesh over the current TP group only.

    CP/BP ranks are independent model-execution replicas that exchange
    activations/KV through DLLM runtime groups. They must not be part of the
    HF tensor-parallel mesh, otherwise Transformers may shard vocab/linear
    weights over the CP/BP axis and break loss ownership and optimizer state.
    """

    tp_size = int(getattr(runtime, "tensor_parallel_size", 1) or 1)
    if tp_size <= 1:
        return None
    ranks = list(getattr(runtime, "tensor_parallel_group_ranks", None) or [])
    if len(ranks) != tp_size:
        raise ValueError("runtime tensor_parallel_group_ranks must match TP size")
    import torch
    from torch.distributed.device_mesh import DeviceMesh

    return DeviceMesh(
        device_type,
        torch.tensor(ranks, dtype=torch.int),
        mesh_dim_names=("tp",),
    )


def hf_from_pretrained_tensor_parallel_kwargs(
    runtime: Any | None,
    *,
    device_type: str,
) -> dict[str, Any]:
    if runtime is None:
        return {}
    mesh = build_hf_tensor_parallel_device_mesh(runtime, device_type=device_type)
    if mesh is None:
        return {}
    return {"tp_plan": "auto", "device_mesh": mesh}


def prepare_hf_tensor_parallel_subgroup(
    distributed_config: Any | None,
    *,
    device_mesh: Any | None = None,
    device_map: Any | None = None,
) -> tuple[Any | None, Any | None, Any | None]:
    """Prepare native HF TP over an explicit DLLM tensor-parallel subgroup.

    Recent Transformers releases require ``tp_size * fsdp_size`` to equal the
    global process count before consulting a caller-provided ``DeviceMesh``.
    DLLM owns data and context parallelism outside Transformers, so that check
    is not applicable: HF must shard only over the supplied TP subgroup.  This
    adapter preserves the native HF initialization and sharding implementation
    while validating the subgroup contract before any model parameters exist.
    """

    if distributed_config is None:
        return None, device_map, device_mesh
    if isinstance(distributed_config, dict):
        from transformers.distributed.configuration_utils import DistributedConfig

        distributed_config = DistributedConfig.from_dict(distributed_config)

    tp_size = int(getattr(distributed_config, "tp_size", 1) or 1)
    fsdp_size = int(getattr(distributed_config, "fsdp_size", 1) or 1)
    if fsdp_size != 1:
        raise ValueError(
            "Hugging Face subgroup tensor parallelism cannot own an FSDP axis; "
            "DLLM data/FSDP parallelism must remain outside the HF TP mesh"
        )
    if tp_size <= 1:
        return distributed_config, device_map, device_mesh
    if device_mesh is None:
        raise ValueError("Hugging Face subgroup tensor parallelism requires a DeviceMesh")

    tp_mesh = (
        device_mesh["tp"]
        if int(getattr(device_mesh, "ndim", 1)) > 1
        else device_mesh
    )
    mesh_size = int(tp_mesh.size())
    if mesh_size != tp_size:
        raise ValueError(
            "Hugging Face TP subgroup size does not match distributed_config: "
            f"mesh={mesh_size}, tp_size={tp_size}"
        )

    tp_plan = getattr(distributed_config, "tp_plan", None)
    if tp_plan is None:
        tp_plan = "auto"
        distributed_config.tp_plan = tp_plan
    from transformers.distributed.utils import initialize_tensor_parallelism

    device_map, prepared_mesh = initialize_tensor_parallelism(
        tp_plan,
        tp_size=tp_size,
        device_mesh=device_mesh,
        device_map=device_map,
    )
    prepared_size = int(prepared_mesh.size())
    if prepared_size != tp_size:
        raise RuntimeError(
            "Transformers initialized an unexpected tensor-parallel mesh: "
            f"expected {tp_size}, got {prepared_size}"
        )
    return distributed_config, device_map, prepared_mesh


def validate_hf_tensor_parallel_model(model: Any, runtime: Any) -> None:
    """Fail closed if HF silently ignored requested TP."""

    expected = int(getattr(runtime, "tensor_parallel_size", 1) or 1)
    if expected <= 1:
        return
    actual = getattr(model, "_tp_size", None)
    if actual is None:
        distributed_config = getattr(
            getattr(model, "config", None),
            "distributed_config",
            None,
        )
        actual = getattr(distributed_config, "tp_size", None)
    if actual is None:
        mesh = getattr(model, "_device_mesh", None)
        if mesh is not None:
            tp_mesh = mesh["tp"] if getattr(mesh, "ndim", 1) > 1 else mesh
            actual = int(tp_mesh.size())
    actual = int(actual or 1)
    if actual != expected:
        raise RuntimeError(
            "requested tensor parallelism, but Hugging Face did not attach "
            f"the expected TP size: expected {expected}, got {actual}"
        )
    if any(_has_sharded_placement(parameter) for parameter in model.parameters()):
        return

    applied_modules = tuple(_native_hf_tp_modules(model))
    if not applied_modules:
        raise RuntimeError(
            "Hugging Face reported tensor parallelism but did not apply its "
            "tensor-parallel plan to any parameter-bearing module"
        )
    for module in applied_modules:
        mesh = getattr(module, "_hf_device_mesh", None)
        if mesh is None or int(mesh.size()) != expected:
            raise RuntimeError(
                "Hugging Face applied a tensor-parallel module on an unexpected "
                f"mesh: expected {expected} ranks"
            )


def _has_sharded_placement(tensor: Any) -> bool:
    placements = getattr(tensor, "placements", None)
    if placements is None:
        placements = getattr(getattr(tensor, "_spec", None), "placements", None)
    return placements is not None and any(
        type(placement).__name__ == "Shard" for placement in placements
    )


def _native_hf_tp_modules(model: Any):
    """Yield modules to which Transformers applied its native TP contract.

    Transformers' native loader stores local weight shards as ordinary
    ``Parameter`` tensors and records the TP style and mesh on the owning
    module.  DTensor placements are therefore not a portable validation signal
    across supported Transformers releases.
    """

    for module in model.modules():
        if getattr(module, "_hf_tp_plan", None) is None:
            continue
        if any(True for _ in module.parameters(recurse=False)):
            yield module
