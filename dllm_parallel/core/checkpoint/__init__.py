# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Rank-aware checkpointing for DLLM distributed training.

The format is intentionally simple and explicit:

- one metadata JSON file written by rank 0,
- one rank-local PyTorch state file per rank for non-DeepSpeed runs,
- DeepSpeed engine checkpoint directories when ZeRO owns optimizer state.

This gives the trainer restartable model/optimizer/RNG/objective state without
adding any work to the hot path unless checkpointing is configured.

The implementation is split across submodules (``io`` primitives, ``format``
metadata/validation, ``sharded`` FSDP/DeepSpeed glue); this package re-exports
the full public surface and keeps the atomic-save primitive + save/load
orchestration here so existing imports — and white-box tests that patch
``checkpoint._atomic_torch_save`` — keep working unchanged.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import re
import shutil
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from dllm_parallel.core.checkpoint.io import (
    CHECKPOINT_FORMAT_VERSION,
    LATEST_TAG,
    METADATA_FILENAME,
    CheckpointResult,
    _AsyncCheckpointHandle,
    _barrier,
    _rank,
    _rank_shard_filename,
    _world_size,
)
from dllm_parallel.core.checkpoint.format import (
    capture_rng_state,
    inspect_checkpoint,
    restore_rng_state,
    validate_checkpoint_manifest,
    _client_state,
    _read_metadata,
    _resolve_tag,
    _validate_format,
    _validate_topology,
    _write_metadata,
)
from dllm_parallel.core.checkpoint.sharded import (
    _checkpoint_backend,
    _checkpoint_state_dicts,
    _load_checkpoint_state_dicts,
    _module,
    _state_dict_module,
)

__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "METADATA_FILENAME",
    "LATEST_TAG",
    "CheckpointResult",
    "capture_rng_state",
    "restore_rng_state",
    "inspect_checkpoint",
    "validate_checkpoint_manifest",
    "save_training_checkpoint",
    "load_training_checkpoint",
    "prune_checkpoints",
    "checkpoint_exists",
    "main",
]


def save_training_checkpoint(
    checkpoint_dir: str | os.PathLike[str],
    *,
    tag: str,
    step: int,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    deepspeed_engine: Any | None = None,
    runtime: Any | None = None,
    objective_state: dict[str, Any] | None = None,
    dataloader_state: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    scheduler_state: dict[str, Any] | None = None,
    kernel_metadata: dict[str, Any] | None = None,
    run_metadata: dict[str, Any] | None = None,
    profiler_metadata: dict[str, Any] | None = None,
    backbone_state: dict[str, Any] | None = None,
    device: torch.device | str | None = None,
    async_save: bool = False,
    model_only: bool = False,
    keep_last_n: int = 0,
    barrier_group: Any | None = None,
) -> CheckpointResult:
    path = Path(checkpoint_dir)
    rank = _rank()
    world_size = _world_size()
    tag = str(tag)
    client_state = _client_state(
        step=int(step),
        runtime=runtime,
        objective_state=objective_state,
        dataloader_state=dataloader_state,
        config=config,
        scheduler_state=scheduler_state,
        kernel_metadata=kernel_metadata,
        run_metadata=run_metadata,
        profiler_metadata=profiler_metadata,
        backbone_state=backbone_state,
        device=device,
    )
    if model_only:
        if async_save:
            raise ValueError("model-only checkpoints do not support async_save")
        checkpoint_model = (
            getattr(deepspeed_engine, "module", None)
            if deepspeed_engine is not None
            else model
        )
        if checkpoint_model is None:
            raise ValueError("model-only checkpoint requires a model or DeepSpeed engine")
        return _save_model_only_checkpoint(
            path,
            tag=tag,
            step=int(step),
            model=checkpoint_model,
            runtime=runtime,
            client_state=client_state,
            keep_last_n=int(keep_last_n),
            barrier_group=barrier_group,
        )
    if deepspeed_engine is not None:
        # DeepSpeed otherwise serializes every frozen base parameter (and its
        # ZeRO-2 frozen fragment) into each checkpoint.  A LoRA run can restore
        # the immutable base from the pinned model revision, so retain only the
        # trainable adapter plus optimizer/client state while remaining fully
        # resumable.
        adapter_state = client_state.get("backbone_state", {}).get("adapter")
        exclude_frozen_parameters = isinstance(adapter_state, Mapping)
        client_state["checkpoint_backend"] = "deepspeed_zero2"
        if exclude_frozen_parameters:
            client_state["model_state_scope"] = "trainable_parameters"
            client_state["frozen_parameters_excluded"] = True
        path.mkdir(parents=True, exist_ok=True)
        _save_deepspeed_checkpoint(
            deepspeed_engine,
            str(path),
            tag=tag,
            client_state=client_state,
            exclude_frozen_parameters=exclude_frozen_parameters,
        )
        _barrier(barrier_group)
        if rank == 0:
            _ensure_replicated_deepspeed_model_aliases(
                path / tag,
                runtime=runtime,
            )
            _write_metadata(
                path,
                tag=tag,
                step=int(step),
                runtime=runtime,
                client_state=client_state,
            )
            prune_checkpoints(path, keep_last_n=int(keep_last_n), latest_tag=tag)
        _barrier(barrier_group)
        return CheckpointResult(path=path, tag=tag, async_save=False)

    if model is None:
        raise ValueError("model is required when deepspeed_engine is not provided")
    rank_dir = path / tag
    rank_dir.mkdir(parents=True, exist_ok=True)
    state_module = _state_dict_module(model)
    client_state["checkpoint_backend"] = _checkpoint_backend(state_module)
    model_state, optimizer_state = _checkpoint_state_dicts(state_module, optimizer)
    state = {
        "client_state": client_state,
        "model": model_state,
        "optimizer": optimizer_state,
        "scheduler": scheduler.state_dict()
        if scheduler is not None and hasattr(scheduler, "state_dict")
        else None,
        "checkpoint_backend": _checkpoint_backend(state_module),
    }
    filename = rank_dir / _rank_shard_filename(rank, world_size)
    if async_save:
        # ``state_dict`` tensors can alias live parameters and optimizer state.
        # Own a CPU snapshot before the optimizer is allowed to mutate them.
        state = _snapshot_checkpoint_state(state)
        error_box: dict[str, BaseException] = {}
        on_complete = lambda error: _finalize_rank_checkpoint(
            path,
            tag=tag,
            step=int(step),
            runtime=runtime,
            client_state=client_state,
            keep_last_n=int(keep_last_n),
            local_error=error,
            barrier_group=barrier_group,
        )
        thread = threading.Thread(
            target=_async_save_target,
            args=(state, filename, error_box),
            name=f"dllm-checkpoint-rank-{rank}",
            daemon=False,
        )
        thread.start()
        return CheckpointResult(
            path=path,
            tag=tag,
            async_save=True,
            async_handle=_AsyncCheckpointHandle(thread, error_box, on_complete),
        )
    _atomic_torch_save(state, filename)
    _finalize_rank_checkpoint(
        path,
        tag=tag,
        step=int(step),
        runtime=runtime,
        client_state=client_state,
        keep_last_n=int(keep_last_n),
        barrier_group=barrier_group,
    )
    return CheckpointResult(path=path, tag=tag, async_save=False)


def _save_deepspeed_checkpoint(
    engine: Any,
    path: str,
    *,
    tag: str,
    client_state: dict[str, Any],
    exclude_frozen_parameters: bool,
) -> None:
    """Save ZeRO state without retaining tied aliases of frozen parameters.

    DeepSpeed's built-in exclusion iterates ``named_parameters()`` with
    duplicate removal. A tied frozen parameter can consequently survive under
    its other state-dict aliases; Qwen's tied embedding/head then adds several
    gigabytes to every LoRA checkpoint. Filter the returned module state by all
    trainable aliases while leaving DeepSpeed's optimizer/client-state format
    untouched and resumable.
    """

    if not exclude_frozen_parameters:
        engine.save_checkpoint(
            path,
            tag=tag,
            client_state=client_state,
            exclude_frozen_parameters=False,
        )
        return
    module = getattr(engine, "module", None)
    original_state_dict = getattr(engine, "module_state_dict", None)
    if module is None or not callable(original_state_dict):
        raise TypeError(
            "compact DeepSpeed checkpoint requires engine.module and "
            "engine.module_state_dict"
        )
    trainable_names = {
        name
        for name, parameter in module.named_parameters(
            recurse=True,
            remove_duplicate=False,
        )
        if parameter.requires_grad
    }
    if not trainable_names:
        raise RuntimeError("compact DeepSpeed checkpoint found no trainable parameters")

    def trainable_state_dict(
        destination: Any | None = None,
        prefix: str = "",
        keep_vars: bool = False,
        exclude_frozen_parameters: bool = False,
    ) -> Any:
        state = original_state_dict(
            destination=destination,
            prefix=prefix,
            keep_vars=keep_vars,
            exclude_frozen_parameters=False,
        )
        if not exclude_frozen_parameters:
            return state
        allowed = {f"{prefix}{name}" for name in trainable_names}
        for name in tuple(state):
            if name not in allowed:
                del state[name]
        return state

    engine.module_state_dict = trainable_state_dict
    try:
        engine.save_checkpoint(
            path,
            tag=tag,
            client_state=client_state,
            exclude_frozen_parameters=True,
        )
    finally:
        engine.module_state_dict = original_state_dict


def load_training_checkpoint(
    checkpoint_dir: str | os.PathLike[str],
    *,
    tag: str = LATEST_TAG,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    deepspeed_engine: Any | None = None,
    runtime: Any | None = None,
    strict: bool = True,
    device: torch.device | str | None = None,
    barrier_group: Any | None = None,
) -> dict[str, Any]:
    path = Path(checkpoint_dir)
    tag = _resolve_tag(path, tag)
    metadata = _read_metadata(path)
    _validate_format(metadata)
    _validate_topology(metadata, runtime=runtime)
    if metadata.get("checkpoint_backend") == "adapter_model_only":
        checkpoint_model = (
            getattr(deepspeed_engine, "module", None)
            if deepspeed_engine is not None
            else model
        )
        if checkpoint_model is None:
            raise ValueError(
                "loading a model-only checkpoint requires a model or DeepSpeed engine"
            )
        client_state = _load_model_only_checkpoint(
            path,
            tag=tag,
            model=checkpoint_model,
            strict=bool(strict),
            device=device,
        )
        _barrier(barrier_group)
        return client_state
    if deepspeed_engine is not None:
        if _rank() == 0:
            _ensure_replicated_deepspeed_model_aliases(
                path / tag,
                runtime=runtime,
            )
        _barrier(barrier_group)
        load_path, client_state = deepspeed_engine.load_checkpoint(
            str(path),
            tag=tag,
            load_module_strict=not (
                metadata.get("model_state_scope") == "trainable_parameters"
                and bool(metadata.get("frozen_parameters_excluded"))
            ),
            load_optimizer_states=True,
            load_lr_scheduler_states=True,
        )
        if load_path is None:
            raise FileNotFoundError(f"DeepSpeed checkpoint not found: {path} tag={tag}")
        if scheduler is not None and hasattr(scheduler, "load_state_dict"):
            scheduler.load_state_dict(client_state.get("scheduler_state"))
        restore_rng_state(client_state.get("rng_state"), device=device)
        return client_state

    if model is None:
        raise ValueError("model is required when deepspeed_engine is not provided")
    rank = _rank()
    world_size = _world_size()
    filename = path / tag / _rank_shard_filename(rank, world_size)
    if not filename.exists():
        raise FileNotFoundError(str(filename))
    state = torch.load(filename, map_location=device or "cpu", weights_only=False)
    state_module = _state_dict_module(model)
    _load_checkpoint_state_dicts(
        state_module,
        optimizer,
        state["model"],
        state.get("optimizer"),
        strict=bool(strict),
    )
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    client_state = dict(state.get("client_state") or {})
    restore_rng_state(client_state.get("rng_state"), device=device)
    _barrier(barrier_group)
    return client_state


def prune_checkpoints(
    checkpoint_dir: str | os.PathLike[str],
    *,
    keep_last_n: int,
    latest_tag: str | None = None,
) -> list[str]:
    """Delete old ``step_*`` checkpoint directories on rank 0.

    ``keep_last_n <= 0`` disables retention. The latest tag is always kept even
    if it does not sort into the newest N entries.
    """

    keep_last_n = int(keep_last_n)
    if keep_last_n <= 0 or _rank() != 0:
        return []
    path = Path(checkpoint_dir)
    if not path.exists():
        return []
    step_dirs = sorted(
        [
            item
            for item in path.iterdir()
            if item.is_dir()
            and (
                item.name.startswith("step_")
                or re.search(r"_step_[0-9]+$", item.name)
            )
        ],
        key=lambda item: (
            int(re.search(r"([0-9]+)$", item.name).group(1)),
            item.name,
        ),
    )
    keep = {item.name for item in step_dirs[-keep_last_n:]}
    if latest_tag:
        keep.add(str(latest_tag))
    removed: list[str] = []
    for item in step_dirs:
        if item.name in keep:
            continue
        shutil.rmtree(item)
        removed.append(item.name)
    return removed


def checkpoint_exists(
    checkpoint_dir: str | os.PathLike[str] | None,
    *,
    tag: str = LATEST_TAG,
) -> bool:
    if checkpoint_dir is None:
        return False
    path = Path(checkpoint_dir)
    if not path.exists():
        return False
    if tag == LATEST_TAG:
        try:
            _resolve_tag(path, tag)
            return True
        except FileNotFoundError:
            return False
    return (path / str(tag)).exists()


def _atomic_torch_save(state: dict[str, Any], filename: Path) -> None:
    tmp = filename.with_suffix(filename.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, filename)


_MODEL_ONLY_FILENAME = "mp_rank_00_model_states.pt"
_MODEL_ONLY_MANIFEST = "model_only_manifest.json"


def _save_model_only_checkpoint(
    path: Path,
    *,
    tag: str,
    step: int,
    model: torch.nn.Module,
    runtime: Any | None,
    client_state: dict[str, Any],
    keep_last_n: int,
    barrier_group: Any | None,
) -> CheckpointResult:
    """Atomically publish one replicated trainable-model checkpoint.

    CP/BP with TP=EP=PP=1 replicates parameters.  Rank zero therefore stores
    one canonical copy of every trainable tensor plus explicit tied aliases;
    frozen base weights, optimizer, scheduler, RNG, and dataloader states are
    intentionally excluded.
    """

    rank = _rank()
    path.mkdir(parents=True, exist_ok=True)
    _barrier(barrier_group)
    rank_zero_error: str | None = None
    if rank == 0:
        final_dir = path / tag
        temporary_dir = path / f".{tag}.model-only.tmp"
        created_final = False
        try:
            if final_dir.exists() or temporary_dir.exists():
                raise FileExistsError(f"checkpoint tag already exists: {final_dir}")
            temporary_dir.mkdir(parents=False)
            module_state, shared_params, inventory = _trainable_model_state(model)
            saved_client_state = {
                "step": int(step),
                "backbone_state": client_state.get("backbone_state", {}),
                "config": client_state.get("config", {}),
                "checkpoint_backend": "adapter_model_only",
                "model_state_scope": "trainable_parameters",
            }
            state = {
                "step": int(step),
                "module": module_state,
                "shared_params": shared_params,
                "client_state": saved_client_state,
                "model_state_scope": "trainable_parameters",
            }
            state_path = temporary_dir / _MODEL_ONLY_FILENAME
            _atomic_torch_save(state, state_path)
            state_sha256 = _file_sha256(state_path)
            manifest = {
                "format": "dllm_parallel.model_only_checkpoint.v1",
                "step": int(step),
                "filename": _MODEL_ONLY_FILENAME,
                "bytes": state_path.stat().st_size,
                "sha256": state_sha256,
                **inventory,
            }
            manifest_path = temporary_dir / _MODEL_ONLY_MANIFEST
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary_dir, final_dir)
            created_final = True
            _validate_model_only_checkpoint(path, tag=tag)
            published_client_state = {
                **client_state,
                "checkpoint_backend": "adapter_model_only",
                "model_state_scope": "trainable_parameters",
                "model_only_manifest": manifest,
            }
            # Do not claim restart state that this checkpoint intentionally omits.
            for field in (
                "objective_state",
                "dataloader_state",
                "scheduler_state",
                "rng_state",
            ):
                published_client_state[field] = {}
            _write_metadata(
                path,
                tag=tag,
                step=int(step),
                runtime=runtime,
                client_state=published_client_state,
            )
            prune_checkpoints(path, keep_last_n=keep_last_n, latest_tag=tag)
        except BaseException as exc:
            if temporary_dir.exists():
                shutil.rmtree(temporary_dir)
            if created_final and final_dir.exists():
                shutil.rmtree(final_dir)
            rank_zero_error = repr(exc)
    rank_zero_error = _broadcast_rank_zero_checkpoint_error(
        rank_zero_error,
        barrier_group=barrier_group,
    )
    if rank_zero_error is not None:
        raise RuntimeError(
            f"model-only checkpoint {tag} was not published: {rank_zero_error}"
        )
    return CheckpointResult(path=path, tag=tag, async_save=False)


def _trainable_model_state(
    model: torch.nn.Module,
) -> tuple[dict[str, torch.Tensor], dict[str, str], dict[str, Any]]:
    module = _module(model)
    by_identity: dict[int, list[tuple[str, torch.nn.Parameter]]] = {}
    for name, parameter in module.named_parameters(
        recurse=True,
        remove_duplicate=False,
    ):
        if parameter.requires_grad:
            by_identity.setdefault(id(parameter), []).append((name, parameter))
    if not by_identity:
        raise RuntimeError("model-only checkpoint found no trainable parameters")
    from dllm_parallel.core.adapters import adapter_metadata

    adapter = adapter_metadata(module)
    if adapter is not None:
        for aliases in by_identity.values():
            parameter = aliases[0][1]
            if not bool(getattr(parameter, "_dllm_lora_parameter", False)):
                raise RuntimeError(
                    f"LoRA checkpoint contains non-adapter trainable tensor: {aliases[0][0]}"
                )

    full_state = module.state_dict()
    module_state: dict[str, torch.Tensor] = {}
    shared_params: dict[str, str] = {}
    canonical_parameter_count = 0
    for aliases in by_identity.values():
        names = [name for name, _ in aliases]
        canonical = min(names, key=_model_only_name_priority)
        parameter = aliases[0][1]
        canonical_parameter_count += int(parameter.numel())
        for name in names:
            value = full_state.get(name)
            if not isinstance(value, torch.Tensor):
                raise RuntimeError(f"state_dict lacks trainable tensor: {name}")
            snapshot = value.detach().to(device="cpu", copy=True).contiguous()
            if not snapshot.is_floating_point() or not bool(torch.isfinite(snapshot).all()):
                raise RuntimeError(f"trainable checkpoint tensor is invalid: {name}")
            module_state[name] = snapshot
            if name != canonical:
                shared_params[name] = canonical
    if adapter is not None and int(adapter.get("trainable_parameters", -1)) != (
        canonical_parameter_count
    ):
        raise RuntimeError(
            "adapter metadata trainable count differs from the saved model state"
        )
    return module_state, shared_params, {
        "canonical_tensors": len(by_identity),
        "stored_tensors": len(module_state),
        "shared_aliases": len(shared_params),
        "canonical_parameters": canonical_parameter_count,
        "tensor_names": sorted(module_state),
    }


def _model_only_name_priority(name: str) -> tuple[int, str]:
    if name.startswith("_te_packed_layers."):
        return (0, name)
    if name.startswith("model.model.decoder.layers."):
        return (0, name)
    if name.startswith("encoder.layers."):
        return (2, name)
    return (1, name)


def _validate_model_only_checkpoint(path: Path, *, tag: str) -> dict[str, Any]:
    tag_dir = path / tag
    manifest_path = tag_dir / _MODEL_ONLY_MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(str(manifest_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "dllm_parallel.model_only_checkpoint.v1":
        raise RuntimeError("model-only checkpoint manifest format drift")
    state_path = tag_dir / str(manifest.get("filename") or "")
    if not state_path.is_file() or state_path.stat().st_size != int(
        manifest.get("bytes", -1)
    ):
        raise RuntimeError("model-only checkpoint file size mismatch")
    if _file_sha256(state_path) != manifest.get("sha256"):
        raise RuntimeError("model-only checkpoint content hash mismatch")
    state = torch.load(state_path, map_location="cpu", mmap=True, weights_only=True)
    if int(state.get("step", -1)) != int(manifest.get("step", -2)):
        raise RuntimeError("model-only checkpoint step differs from manifest")
    if state.get("model_state_scope") != "trainable_parameters":
        raise RuntimeError("model-only checkpoint tensor scope drift")
    module_state = state.get("module")
    shared_params = state.get("shared_params")
    if not isinstance(module_state, dict) or not isinstance(shared_params, dict):
        raise RuntimeError("model-only checkpoint state structure is invalid")
    expected_names = set(manifest.get("tensor_names") or ())
    if set(module_state) != expected_names:
        raise RuntimeError("model-only checkpoint tensor inventory differs from manifest")
    canonical_names = expected_names - set(shared_params)
    if len(canonical_names) != int(manifest.get("canonical_tensors", -1)):
        raise RuntimeError("model-only canonical tensor count mismatch")
    canonical_parameters = 0
    for name in sorted(canonical_names):
        tensor = module_state.get(name)
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(f"model-only checkpoint value is not a tensor: {name}")
        if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"model-only checkpoint tensor is invalid: {name}")
        canonical_parameters += int(tensor.numel())
    if canonical_parameters != int(manifest.get("canonical_parameters", -1)):
        raise RuntimeError("model-only canonical parameter count mismatch")
    for alias, canonical in shared_params.items():
        if alias not in module_state or canonical not in canonical_names:
            raise RuntimeError("model-only shared tensor mapping is invalid")
        if not torch.equal(module_state[alias], module_state[canonical]):
            raise RuntimeError("model-only shared tensor value differs")
    return manifest


def _load_model_only_checkpoint(
    path: Path,
    *,
    tag: str,
    model: torch.nn.Module,
    strict: bool,
    device: torch.device | str | None,
) -> dict[str, Any]:
    _validate_model_only_checkpoint(path, tag=tag)
    state = torch.load(
        path / tag / _MODEL_ONLY_FILENAME,
        map_location=device or "cpu",
        weights_only=True,
    )
    module_state = state.get("module")
    if not isinstance(module_state, dict) or not module_state:
        raise RuntimeError("model-only checkpoint has no trainable model state")
    module = _module(model)
    current_trainable = {
        name
        for name, parameter in module.named_parameters(
            recurse=True,
            remove_duplicate=False,
        )
        if parameter.requires_grad
    }
    observed = set(module_state)
    if current_trainable != observed:
        raise RuntimeError(
            "model-only trainable tensor inventory differs: "
            f"missing={sorted(current_trainable - observed)[:3]} "
            f"extra={sorted(observed - current_trainable)[:3]}"
        )
    incompatible = module.load_state_dict(module_state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"unexpected model-only keys: {incompatible.unexpected_keys[:3]}"
        )
    if strict:
        missing_trainable = current_trainable & set(incompatible.missing_keys)
        if missing_trainable:
            raise RuntimeError(
                f"model-only checkpoint omitted trainable keys: {sorted(missing_trainable)[:3]}"
            )
    return dict(state.get("client_state") or {"step": state.get("step", 0)})


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _broadcast_rank_zero_checkpoint_error(
    error: str | None,
    *,
    barrier_group: Any | None,
) -> str | None:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return error
    payload = [error if _rank() == 0 else None]
    torch.distributed.broadcast_object_list(payload, src=0, group=barrier_group)
    return payload[0]


def _async_save_target(
    state: dict[str, Any],
    filename: Path,
    error_box: dict[str, BaseException],
) -> None:
    try:
        _atomic_torch_save(state, filename)
    except BaseException as exc:
        error_box["error"] = exc


def _finalize_rank_checkpoint(
    path: Path,
    *,
    tag: str,
    step: int,
    runtime: Any | None,
    client_state: dict[str, Any],
    keep_last_n: int,
    local_error: BaseException | None = None,
    barrier_group: Any | None = None,
) -> None:
    """Publish metadata only after every rank-local shard is durable."""

    rank = _rank()
    world_size = _world_size()
    rank_dir = path / tag
    local_error_file = rank_dir / f"rank_{rank:05d}.error"
    if local_error is not None:
        local_error_file.write_text(repr(local_error), encoding="utf-8")
    global_error_file = rank_dir / "checkpoint.error"
    _barrier(barrier_group)
    if _rank() == 0:
        missing = [
            _rank_shard_filename(checkpoint_rank, world_size)
            for checkpoint_rank in range(world_size)
            if not (
                rank_dir / _rank_shard_filename(checkpoint_rank, world_size)
            ).exists()
        ]
        errors = sorted(rank_dir.glob("rank_*.error"))
        if missing or errors:
            details = [f"missing shards: {', '.join(missing)}"] if missing else []
            details.extend(error.read_text(encoding="utf-8") for error in errors)
            global_error_file.write_text("; ".join(details), encoding="utf-8")
        else:
            _write_metadata(
                path,
                tag=tag,
                step=int(step),
                runtime=runtime,
                client_state=client_state,
            )
            prune_checkpoints(path, keep_last_n=int(keep_last_n), latest_tag=tag)
    _barrier(barrier_group)
    if global_error_file.exists():
        raise RuntimeError(
            f"checkpoint {tag} was not published: "
            + global_error_file.read_text(encoding="utf-8")
        )


def _snapshot_checkpoint_state(value: Any) -> Any:
    """Detach checkpoint state from live training storage for async I/O."""

    if torch.is_tensor(value):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, Mapping):
        return type(value)(
            (key, _snapshot_checkpoint_state(item)) for key, item in value.items()
        )
    if isinstance(value, tuple):
        return tuple(_snapshot_checkpoint_state(item) for item in value)
    if isinstance(value, list):
        return [_snapshot_checkpoint_state(item) for item in value]
    return value


_DEEPSPEED_MODEL_RANK_RE = re.compile(r"mp_rank_(\d+)_model_states[.]pt$")


def _ensure_replicated_deepspeed_model_aliases(
    tag_dir: Path,
    *,
    runtime: Any | None,
) -> list[Path]:
    """Publish zero-copy model-state aliases for replicated CP/BP ranks.

    DeepSpeed ZeRO-2 correctly writes one full model state for the optimizer
    data-parallel CP/BP group, but its loader enumerates the execution-model
    world and expects one filename per model rank. For TP1/EP1/PP1 those model
    states are identical. Hard links make that replicated layout explicit
    without multiplying a roughly full-model checkpoint by the CP/BP width.
    """

    plan = getattr(runtime, "plan", None)
    if plan is None:
        return []
    if (
        int(getattr(plan, "tensor_parallel_size", 1) or 1) != 1
        or int(getattr(plan, "expert_parallel_size", 1) or 1) != 1
        or int(getattr(plan, "pipeline_parallel_size", 1) or 1) != 1
    ):
        return []
    expected = int(getattr(plan, "model_parallel_size", 1) or 1)
    if expected <= 1 or not tag_dir.is_dir():
        return []
    candidates = sorted(tag_dir.glob("*model_states.pt"))
    matched = [item for item in candidates if _DEEPSPEED_MODEL_RANK_RE.search(item.name)]
    if not matched:
        return []
    rank_zero = next(
        (
            item
            for item in matched
            if int(_DEEPSPEED_MODEL_RANK_RE.search(item.name).group(1)) == 0
        ),
        None,
    )
    if rank_zero is None:
        raise RuntimeError(
            f"replicated DeepSpeed checkpoint has no rank-zero model state: {tag_dir}"
        )
    for candidate in matched:
        if not os.path.samefile(rank_zero, candidate):
            raise RuntimeError(
                "TP1/EP1/PP1 DeepSpeed checkpoint contains conflicting model states: "
                f"{rank_zero.name}, {candidate.name}"
            )
    created: list[Path] = []
    match = _DEEPSPEED_MODEL_RANK_RE.search(rank_zero.name)
    assert match is not None
    width = len(match.group(1))
    for model_rank in range(expected):
        target_name = _DEEPSPEED_MODEL_RANK_RE.sub(
            f"mp_rank_{model_rank:0{width}d}_model_states.pt",
            rank_zero.name,
        )
        target = tag_dir / target_name
        if target.exists():
            if not os.path.samefile(rank_zero, target):
                raise RuntimeError(
                    f"DeepSpeed model-state alias conflicts with {target}"
                )
            continue
        try:
            os.link(rank_zero, target)
        except OSError as hardlink_error:
            # Some durable object-backed filesystems expose POSIX paths but do
            # not implement hard links. A relative symlink preserves the
            # zero-copy replicated checkpoint layout; copying is the
            # last-resort portable representation.
            if hardlink_error.errno not in {
                errno.EPERM,
                errno.EOPNOTSUPP,
                errno.EXDEV,
            }:
                raise
            try:
                target.symlink_to(rank_zero.name)
            except OSError as symlink_error:
                if symlink_error.errno not in {
                    errno.EPERM,
                    errno.EOPNOTSUPP,
                    errno.ENOSYS,
                    errno.EXDEV,
                }:
                    raise
                shutil.copy2(rank_zero, target)
        created.append(target)
    return created


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Inspect or validate DLLM checkpoints.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("checkpoint_dir")
    inspect_parser.add_argument("--tag", default=LATEST_TAG)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("checkpoint_dir")
    validate_parser.add_argument("--tag", default=LATEST_TAG)
    args = parser.parse_args(argv)
    if args.command == "inspect":
        payload = inspect_checkpoint(args.checkpoint_dir, tag=args.tag)
    elif args.command == "validate":
        payload = validate_checkpoint_manifest(args.checkpoint_dir, tag=args.tag)
    else:  # pragma: no cover - argparse enforces choices.
        raise AssertionError(args.command)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
