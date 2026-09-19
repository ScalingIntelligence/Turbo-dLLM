# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Checkpoint format: RNG capture, client-state/metadata assembly, validation.

Depends only on the leaf :mod:`dllm_parallel.core.checkpoint.io` module.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

from dllm_parallel.core.checkpoint.io import (
    CHECKPOINT_FORMAT_VERSION,
    LATEST_TAG,
    METADATA_FILENAME,
    _rank,
    _rank_shard_filename,
    _world_size,
)


def capture_rng_state(device: torch.device | str | None = None) -> dict[str, Any]:
    state: dict[str, Any] = {
        "torch_cpu": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        cuda_device = torch.device(device) if device is not None else torch.device(
            "cuda",
            torch.cuda.current_device(),
        )
        if cuda_device.type == "cuda":
            state["torch_cuda"] = torch.cuda.get_rng_state(cuda_device)
            state["cuda_device_index"] = cuda_device.index
    return state


def restore_rng_state(
    state: dict[str, Any] | None,
    *,
    device: torch.device | str | None = None,
) -> None:
    if not state:
        return
    cpu_state = state.get("torch_cpu")
    if cpu_state is not None:
        torch.random.set_rng_state(cpu_state.cpu())
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        cuda_device = torch.device(device) if device is not None else torch.device(
            "cuda",
            torch.cuda.current_device(),
        )
        torch.cuda.set_rng_state(cuda_state.cpu(), cuda_device)


def _client_state(
    *,
    step: int,
    runtime: Any | None,
    objective_state: dict[str, Any] | None,
    dataloader_state: dict[str, Any] | None,
    config: dict[str, Any] | None,
    scheduler_state: dict[str, Any] | None,
    kernel_metadata: dict[str, Any] | None,
    run_metadata: dict[str, Any] | None,
    profiler_metadata: dict[str, Any] | None,
    backbone_state: dict[str, Any] | None,
    device: torch.device | str | None,
) -> dict[str, Any]:
    return {
        "step": int(step),
        "rank": _rank(),
        "world_size": _world_size(),
        "topology": _runtime_topology(runtime),
        "objective_state": objective_state or {},
        "dataloader_state": dataloader_state or {},
        "scheduler_state": scheduler_state or {},
        "config": config or {},
        "kernel_metadata": kernel_metadata or {},
        "run_metadata": run_metadata or {},
        "profiler_metadata": profiler_metadata or {},
        "backbone_state": backbone_state or {},
        "rng_state": capture_rng_state(device),
    }


def _runtime_topology(runtime: Any | None) -> dict[str, Any]:
    plan = getattr(runtime, "plan", None)
    policy = getattr(runtime, "cp_bp_policy", None)
    process_groups = getattr(runtime, "process_groups", None)
    return {
        "enabled": bool(getattr(runtime, "enabled", False)),
        "layout": getattr(plan, "layout", None),
        "placement": getattr(plan, "placement", None),
        "data_parallel_size": getattr(plan, "data_parallel_size", None),
        "context_parallel_size": getattr(plan, "context_parallel_size", None),
        "block_parallel_size": getattr(plan, "block_parallel_size", None),
        "tensor_parallel_size": getattr(plan, "tensor_parallel_size", None),
        "pipeline_parallel_size": getattr(plan, "pipeline_parallel_size", None),
        "expert_parallel_size": getattr(plan, "expert_parallel_size", None),
        "sequence_parallel": bool(getattr(runtime, "sequence_parallel", False)),
        "cp_bp_policy": policy.to_log_dict() if hasattr(policy, "to_log_dict") else None,
        "process_groups": (
            process_groups.to_log_dict()
            if hasattr(process_groups, "to_log_dict")
            else None
        ),
    }


def inspect_checkpoint(
    checkpoint_dir: str | os.PathLike[str],
    *,
    tag: str = LATEST_TAG,
) -> dict[str, Any]:
    """Return the rank-zero checkpoint manifest without loading model tensors."""

    path = Path(checkpoint_dir)
    metadata = _read_metadata(path)
    _validate_format(metadata)
    resolved_tag = _resolve_tag(path, tag) if tag == LATEST_TAG else str(tag)
    return {
        **metadata,
        "checkpoint_dir": str(path),
        "resolved_tag": resolved_tag,
        "rank_shards": sorted(
            item.name for item in (path / resolved_tag).glob("rank_*_of_*.pt")
        )
        if (path / resolved_tag).exists()
        else [],
    }


def validate_checkpoint_manifest(
    checkpoint_dir: str | os.PathLike[str],
    *,
    tag: str = LATEST_TAG,
    runtime: Any | None = None,
) -> dict[str, Any]:
    """Validate checkpoint manifest format/topology and return it."""

    manifest = inspect_checkpoint(checkpoint_dir, tag=tag)
    if runtime is not None:
        _validate_topology(manifest, runtime=runtime)
    if not manifest.get("format"):
        raise RuntimeError("checkpoint manifest is missing format")
    if not manifest.get("topology"):
        raise RuntimeError("checkpoint manifest is missing topology")
    if not manifest.get("config"):
        raise RuntimeError("checkpoint manifest is missing resolved run spec")
    if manifest.get("checkpoint_backend") == "adapter_model_only":
        if manifest.get("model_state_scope") != "trainable_parameters":
            raise RuntimeError("model-only checkpoint has an invalid tensor scope")
        if not manifest.get("model_only_manifest"):
            raise RuntimeError("model-only checkpoint is missing its content manifest")
    else:
        if not manifest.get("objective_state"):
            raise RuntimeError("checkpoint manifest is missing objective state")
        if not manifest.get("dataloader_state"):
            raise RuntimeError("checkpoint manifest is missing dataloader state")
    return manifest


def _write_metadata(
    path: Path,
    *,
    tag: str,
    step: int,
    runtime: Any | None,
    client_state: dict[str, Any],
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": CHECKPOINT_FORMAT_VERSION,
        "latest_tag": str(tag),
        "step": int(step),
        "world_size": _world_size(),
        "topology": _runtime_topology(runtime),
        "config": client_state.get("config", {}),
        "objective_state": _jsonable_state(client_state.get("objective_state", {})),
        "dataloader_state": _jsonable_state(client_state.get("dataloader_state", {})),
        "scheduler_state": _jsonable_state(client_state.get("scheduler_state", {})),
        "kernel_metadata": _jsonable_state(client_state.get("kernel_metadata", {})),
        "run_metadata": _jsonable_state(client_state.get("run_metadata", {})),
        "profiler_metadata": _jsonable_state(client_state.get("profiler_metadata", {})),
        "backbone_state": _jsonable_state(client_state.get("backbone_state", {})),
        "checkpoint_backend": client_state.get("checkpoint_backend", "rank_local"),
        "model_state_scope": client_state.get("model_state_scope", "full_training_state"),
        "frozen_parameters_excluded": bool(
            client_state.get("frozen_parameters_excluded", False)
        ),
        "model_only_manifest": _jsonable_state(
            client_state.get("model_only_manifest", {})
        ),
        "rank_state_pattern": _rank_shard_filename(0, _world_size()).replace(
            "00000",
            "{rank:05d}",
            1,
        ),
    }
    tmp = path / (METADATA_FILENAME + ".tmp")
    final = path / METADATA_FILENAME
    tmp.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, final)


def _jsonable_state(value: Any) -> Any:
    if torch.is_tensor(value):
        return {
            "tensor": True,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, dict):
        return {str(key): _jsonable_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_state(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _read_metadata(path: Path) -> dict[str, Any]:
    metadata_path = path / METADATA_FILENAME
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _validate_format(metadata: dict[str, Any]) -> None:
    if not metadata:
        return
    saved_format = metadata.get("format")
    if saved_format is not None and str(saved_format) != CHECKPOINT_FORMAT_VERSION:
        raise RuntimeError(
            "checkpoint format version is not supported by this build: "
            f"{saved_format!r} != {CHECKPOINT_FORMAT_VERSION!r}"
        )


def _resolve_tag(path: Path, tag: str) -> str:
    if tag != LATEST_TAG:
        return tag
    metadata = _read_metadata(path)
    latest = metadata.get("latest_tag")
    if not latest:
        raise FileNotFoundError(f"latest checkpoint tag is not recorded in {path}")
    return str(latest)


def _validate_topology(metadata: dict[str, Any], *, runtime: Any | None) -> None:
    if not metadata:
        return
    saved_world_size = metadata.get("world_size")
    if saved_world_size is not None and int(saved_world_size) != _world_size():
        raise RuntimeError(
            "checkpoint world_size does not match current run: "
            f"{saved_world_size} != {_world_size()}"
        )
    saved_topology = metadata.get("topology") or {}
    current_topology = _runtime_topology(runtime)
    checked_keys = (
        "layout",
        "data_parallel_size",
        "context_parallel_size",
        "block_parallel_size",
        "tensor_parallel_size",
        "sequence_parallel",
    )
    for key in checked_keys:
        saved_value = saved_topology.get(key)
        current_value = current_topology.get(key)
        if saved_value != current_value:
            raise RuntimeError(
                "checkpoint topology does not match current run for "
                f"{key}: {saved_value!r} != {current_value!r}. "
                "Resharding checkpoints is not implemented yet."
            )
    for key in ("pipeline_parallel_size", "expert_parallel_size"):
        if key not in saved_topology:
            continue
        saved_value = saved_topology[key]
        current_value = current_topology.get(key)
        if saved_value != current_value:
            raise RuntimeError(
                "checkpoint topology does not match current run for "
                f"{key}: {saved_value!r} != {current_value!r}. "
                "Resharding checkpoints is not implemented yet."
            )
