# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Production export for packed DiffusionGemma LoRA checkpoints.

Training installs LoRA after Transformer Engine has fused the attention and
dense-MLP projections.  A deployable Hugging Face checkpoint therefore cannot
be produced by PEFT's ``merge_and_unload`` or DeepSpeed's ``zero_to_fp32``:
the selected fused tensors must be translated back to DiffusionGemma's
canonical state dict, with batched three-dimensional merges when grouped expert
adapters are selected.

This module deliberately supports only replicated model weights (TP=EP=1).
Context and block parallel ranks contain replicas and need no reassembly.  A
sharded checkpoint is rejected instead of risking a superficially valid but
incomplete artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch


_EXPORT_FORMAT = "dllm_parallel.diffusiongemma_merged_lora.v1"
_CHECKPOINT_FORMAT = "dllm_parallel.distributed_checkpoint.v1"
_MANIFEST_FILENAME = "dllm_parallel_merge.json"
_INDEX_FILENAME = "model.safetensors.index.json"
_SUPPORTED_ADAPTER_TARGETS = frozenset({"attention", "mlp", "experts"})


@dataclass(frozen=True)
class DiffusionGemmaMergeResult:
    output_dir: Path
    checkpoint_tag: str
    checkpoint_step: int
    modified_tensors: int
    adapter_tensors: int
    adapter_parameters: int
    output_shards: int


@dataclass(frozen=True)
class _MergeTarget:
    adapter_a: str
    adapter_b: str
    checkpoint_base: str
    checkpoint_base_start: int | None = None
    checkpoint_base_stop: int | None = None
    grouped: bool = False


def merge_diffusion_gemma_lora_checkpoint(
    checkpoint_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    base_model: str | os.PathLike[str] | None = None,
    tag: str = "latest",
    local_files_only: bool = False,
    row_chunk_size: int = 4096,
    expert_chunk_size: int = 1,
    verify_base_weights: bool = True,
    hash_checkpoint: bool = True,
    hash_output: bool = True,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> DiffusionGemmaMergeResult:
    """Merge one packed LoRA training checkpoint into its pinned HF model.

    The output directory is published with one atomic rename.  It must not
    already exist.  All matrix products and additions are evaluated in FP32,
    checked for finite values, and cast back to the base tensor dtype.
    """

    row_chunk_size = int(row_chunk_size)
    expert_chunk_size = int(expert_chunk_size)
    if row_chunk_size <= 0:
        raise ValueError("row_chunk_size must be positive")
    if expert_chunk_size <= 0:
        raise ValueError("expert_chunk_size must be positive")

    checkpoint_root = Path(checkpoint_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if not checkpoint_root.is_dir():
        raise FileNotFoundError(str(checkpoint_root))
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite existing export directory: {destination}"
        )
    if destination == checkpoint_root or checkpoint_root in destination.parents:
        raise ValueError("output_dir must not be inside the training checkpoint")

    metadata_path = checkpoint_root / "metadata.json"
    metadata = _read_json(metadata_path)
    if metadata.get("format") != _CHECKPOINT_FORMAT:
        raise RuntimeError(
            f"unsupported or missing checkpoint format: {metadata.get('format')!r}"
        )
    family = (metadata.get("backbone_state") or {}).get("family")
    if family != "diffusion_gemma":
        raise RuntimeError(f"checkpoint backbone is not DiffusionGemma: {family!r}")
    checkpoint_backend = metadata.get("checkpoint_backend")
    if checkpoint_backend not in {None, "deepspeed_zero2", "adapter_model_only"}:
        raise RuntimeError(
            "packed DiffusionGemma export requires a DeepSpeed or adapter-only checkpoint"
        )
    adapter_model_only = checkpoint_backend == "adapter_model_only"
    if adapter_model_only and metadata.get("model_state_scope") != (
        "trainable_parameters"
    ):
        raise RuntimeError("adapter-only checkpoint has an invalid tensor scope")
    compact_deepspeed = (
        checkpoint_backend == "deepspeed_zero2"
        and metadata.get("model_state_scope") == "trainable_parameters"
        and metadata.get("frozen_parameters_excluded") is True
    )
    checkpoint_base_omitted = adapter_model_only or compact_deepspeed
    resolved_tag = _resolve_checkpoint_tag(metadata, tag)
    checkpoint_step = int(metadata.get("step", -1))
    if resolved_tag != str(metadata.get("latest_tag")) and tag == "latest":
        raise RuntimeError("checkpoint latest tag is internally inconsistent")
    tag_dir = checkpoint_root / resolved_tag
    if not tag_dir.is_dir():
        raise FileNotFoundError(str(tag_dir))
    latest_marker = checkpoint_root / "latest"
    if latest_marker.is_file() and tag == "latest":
        marker_tag = latest_marker.read_text(encoding="utf-8").strip()
        if marker_tag != resolved_tag:
            raise RuntimeError("DeepSpeed latest marker and metadata tag differ")

    config = metadata.get("config")
    if not isinstance(config, dict):
        raise RuntimeError("checkpoint metadata is missing its resolved run config")
    _validate_replicated_export_topology(config)
    adapter = _validate_adapter_metadata(metadata, config)
    model_config = config.get("model") or {}
    model_id = model_config.get("id")
    revision = model_config.get("revision")
    if not isinstance(model_id, str) or not model_id:
        raise RuntimeError("checkpoint does not identify its base model")
    if not isinstance(revision, str) or not revision:
        raise RuntimeError("checkpoint does not pin an immutable base-model revision")

    model_state_path = _resolve_deepspeed_model_state(tag_dir)
    if adapter_model_only:
        _validate_model_only_artifact(
            metadata,
            tag_dir=tag_dir,
            model_state_path=model_state_path,
        )
    before_stat = model_state_path.stat()
    _emit(
        progress,
        "checkpoint_loading",
        path=str(model_state_path),
        size_bytes=int(before_stat.st_size),
    )
    checkpoint_state = torch.load(
        model_state_path,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    module_state = checkpoint_state.get("module")
    if not isinstance(module_state, dict) or not module_state:
        raise RuntimeError("DeepSpeed checkpoint has no module state")
    if int(checkpoint_state.get("step", checkpoint_step)) != checkpoint_step:
        raise RuntimeError("checkpoint model state and metadata steps differ")

    base_root = _resolve_base_model(
        base_model if base_model is not None else model_id,
        expected_model_id=model_id,
        revision=revision,
        local_files_only=bool(local_files_only),
    )
    if base_root == checkpoint_root or base_root in destination.parents:
        raise ValueError("output_dir must not replace or contain the base model")
    base_config = _read_json(base_root / "config.json")
    _validate_base_config(base_config)
    weight_map, index_payload = _base_weight_map(base_root)
    tensor_shapes = _safetensor_shapes(base_root, weight_map)
    targets, expected_adapters, allowed_aliases = _build_merge_plan(
        base_config,
        tensor_shapes,
        module_state,
        adapter_targets=frozenset(adapter["targets"]),
        require_checkpoint_base=not checkpoint_base_omitted,
    )
    layer_count = len(base_config["text_config"]["layer_types"])
    adapter_targets = frozenset(adapter["targets"])
    expected_linear_modules = (
        2 * layer_count * sum(role in adapter_targets for role in ("attention", "mlp"))
    )
    if int(adapter["linear_modules"]) != expected_linear_modules:
        raise RuntimeError(
            "checkpoint linear-adapter count differs from the model layers"
        )
    expected_expert_modules = layer_count if "experts" in adapter_targets else 0
    if int(adapter["expert_modules"]) != expected_expert_modules:
        raise RuntimeError(
            "checkpoint expert-adapter count differs from the model layers"
        )
    inventory = _validate_adapter_inventory(
        module_state,
        checkpoint_state.get("shared_params") or {},
        expected_adapters,
        allowed_aliases,
        expected_rank=int(adapter["rank"]),
        expected_trainable_parameters=int(adapter["trainable_parameters"]),
    )
    _emit(
        progress,
        "adapter_validated",
        tensors=inventory["tensor_count"],
        parameters=inventory["parameter_count"],
        zero_b_tensors=inventory["zero_b_tensors"],
    )

    checkpoint_digest = None
    if hash_checkpoint:
        _emit(progress, "checkpoint_hashing", path=str(model_state_path))
        checkpoint_digest = _sha256_file(model_state_path)
    after_stat = model_state_path.stat()
    if (
        before_stat.st_size != after_stat.st_size
        or before_stat.st_mtime_ns != after_stat.st_mtime_ns
    ):
        raise RuntimeError("checkpoint model-state file changed during export")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.merge-",
            dir=str(destination.parent),
        )
    )
    published = False
    try:
        _copy_nonweight_artifacts(base_root, temporary)
        shard_hashes: dict[str, str] = {}
        shard_names = sorted(set(weight_map.values()))
        for shard_index, shard_name in enumerate(shard_names, start=1):
            source_path = base_root / shard_name
            names = [name for name, value in weight_map.items() if value == shard_name]
            _emit(
                progress,
                "shard_merging",
                shard=shard_name,
                index=shard_index,
                total=len(shard_names),
                tensors=len(names),
            )
            from safetensors import safe_open
            from safetensors.torch import save_file

            output_state: dict[str, torch.Tensor] = {}
            with safe_open(str(source_path), framework="pt", device="cpu") as source:
                source_keys = set(source.keys())
                missing = sorted(set(names) - source_keys)
                if missing:
                    raise RuntimeError(
                        f"base shard {shard_name} is missing indexed tensors: {missing[:3]}"
                    )
                for name in names:
                    base_tensor = source.get_tensor(name)
                    target = targets.get(name)
                    if target is None:
                        output_state[name] = base_tensor
                        continue
                    if not checkpoint_base_omitted:
                        checkpoint_base = _checkpoint_base_tensor(module_state, target)
                        if verify_base_weights and not torch.equal(
                            base_tensor,
                            checkpoint_base,
                        ):
                            raise RuntimeError(
                                f"base tensor differs from the frozen training weight: {name}"
                            )
                    output_state[name] = _merge_weight(
                        base_tensor,
                        module_state[target.adapter_a],
                        _checkpoint_adapter_b_tensor(module_state, target),
                        scale=float(adapter["alpha"]) / float(adapter["rank"]),
                        grouped=target.grouped,
                        row_chunk_size=row_chunk_size,
                        expert_chunk_size=expert_chunk_size,
                        name=name,
                    )
            output_path = temporary / shard_name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_shard = output_path.with_name(f".{output_path.name}.tmp")
            save_file(output_state, str(temporary_shard))
            os.replace(temporary_shard, output_path)
            del output_state
            if hash_output:
                shard_hashes[shard_name] = _sha256_file(output_path)

        _atomic_json_write(temporary / _INDEX_FILENAME, index_payload)
        metadata_digest = _sha256_file(metadata_path)
        merge_manifest = {
            "format": _EXPORT_FORMAT,
            "base_model": {"id": model_id, "revision": revision},
            "checkpoint": {
                "path": str(checkpoint_root),
                "tag": resolved_tag,
                "step": checkpoint_step,
                "format": metadata["format"],
                "metadata_sha256": metadata_digest,
                "model_state_filename": model_state_path.name,
                "model_state_size_bytes": int(after_stat.st_size),
                "model_state_sha256": checkpoint_digest,
            },
            "adapter": adapter,
            "merge": {
                "formula": "base_fp32 + (alpha / rank) * (B_fp32 @ A_fp32)",
                "accumulation_dtype": "float32",
                "output_dtype": "base_tensor_dtype",
                "modified_tensors": len(targets),
                "adapter_tensors": inventory["tensor_count"],
                "adapter_parameters": inventory["parameter_count"],
                "zero_b_tensors": inventory["zero_b_tensors"],
                "base_weights_verified": bool(
                    verify_base_weights and not checkpoint_base_omitted
                ),
                "base_verification": (
                    "pinned_huggingface_revision"
                    if checkpoint_base_omitted
                    else "checkpoint_frozen_weights_exact"
                    if verify_base_weights
                    else "disabled"
                ),
                "row_chunk_size": row_chunk_size,
                "expert_chunk_size": expert_chunk_size,
            },
            "output": {
                "weight_index": _INDEX_FILENAME,
                "shards": shard_names,
                "sha256": shard_hashes,
            },
        }
        _atomic_json_write(temporary / _MANIFEST_FILENAME, merge_manifest)
        validation = validate_merged_diffusion_gemma_artifact(
            temporary,
            verify_hashes=bool(hash_output),
        )
        _emit(progress, "artifact_validated", **validation)
        if destination.exists():
            raise FileExistsError(
                "export destination appeared while the merge was running; "
                f"refusing to replace it: {destination}"
            )
        os.replace(temporary, destination)
        published = True
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary)

    return DiffusionGemmaMergeResult(
        output_dir=destination,
        checkpoint_tag=resolved_tag,
        checkpoint_step=checkpoint_step,
        modified_tensors=len(targets),
        adapter_tensors=int(inventory["tensor_count"]),
        adapter_parameters=int(inventory["parameter_count"]),
        output_shards=len(set(weight_map.values())),
    )


def validate_merged_diffusion_gemma_artifact(
    output_dir: str | os.PathLike[str],
    *,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    """Validate tensor coverage, shapes, sizes, and recorded artifact hashes."""

    root = Path(output_dir).expanduser().resolve()
    manifest = _read_json(root / _MANIFEST_FILENAME)
    if manifest.get("format") != _EXPORT_FORMAT:
        raise RuntimeError("merged artifact has an unsupported manifest format")
    config = _read_json(root / "config.json")
    _validate_base_config(config)
    index = _read_json(root / _INDEX_FILENAME)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError("merged artifact has no safetensors weight map")
    from safetensors import safe_open

    observed: set[str] = set()
    total_size = 0
    shard_names = sorted(set(str(value) for value in weight_map.values()))
    for shard_name in shard_names:
        shard_path = root / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(str(shard_path))
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            shard_keys = set(handle.keys())
            expected = {
                name for name, value in weight_map.items() if value == shard_name
            }
            if shard_keys != expected:
                raise RuntimeError(
                    f"merged shard tensor coverage differs for {shard_name}: "
                    f"missing={sorted(expected - shard_keys)[:3]} "
                    f"extra={sorted(shard_keys - expected)[:3]}"
                )
            overlap = observed & shard_keys
            if overlap:
                raise RuntimeError(
                    f"merged tensors occur in multiple shards: {sorted(overlap)[:3]}"
                )
            observed.update(shard_keys)
            for name in shard_keys:
                tensor_slice = handle.get_slice(name)
                shape = tuple(int(value) for value in tensor_slice.get_shape())
                dtype = str(tensor_slice.get_dtype())
                total_size += _dtype_size(dtype) * _numel(shape)
    if observed != set(weight_map):
        raise RuntimeError("merged weight index does not cover the artifact tensors")
    if any("lora" in name.lower() for name in observed):
        raise RuntimeError("merged artifact unexpectedly contains LoRA tensors")
    expected_size = int((index.get("metadata") or {}).get("total_size", total_size))
    if total_size != expected_size:
        raise RuntimeError(
            f"merged tensor bytes differ from index metadata: {total_size} != {expected_size}"
        )
    recorded_hashes = (manifest.get("output") or {}).get("sha256") or {}
    if verify_hashes:
        if set(recorded_hashes) != set(shard_names):
            raise RuntimeError("merged artifact does not record every shard hash")
        for shard_name in shard_names:
            actual = _sha256_file(root / shard_name)
            if actual != recorded_hashes[shard_name]:
                raise RuntimeError(f"merged shard hash differs for {shard_name}")
    return {
        "tensors": len(observed),
        "shards": len(shard_names),
        "total_size_bytes": total_size,
        "hashes_verified": bool(verify_hashes),
    }


def _resolve_checkpoint_tag(metadata: dict[str, Any], tag: str) -> str:
    if tag != "latest":
        return str(tag)
    latest = metadata.get("latest_tag")
    if not isinstance(latest, str) or not latest:
        raise RuntimeError("checkpoint metadata does not publish a latest tag")
    return latest


def _resolve_deepspeed_model_state(tag_dir: Path) -> Path:
    candidates = sorted(tag_dir.glob("*model_states.pt"))
    if not candidates:
        raise RuntimeError("replicated DeepSpeed export found no model-state file")
    canonical = next(
        (
            item
            for item in candidates
            if item.name.endswith("mp_rank_00_model_states.pt")
        ),
        candidates[0],
    )
    if any(not os.path.samefile(canonical, item) for item in candidates):
        raise RuntimeError(
            "replicated DeepSpeed export requires one model state or zero-copy aliases; "
            f"found {[item.name for item in candidates]}"
        )
    return canonical


def _validate_model_only_artifact(
    metadata: dict[str, Any],
    *,
    tag_dir: Path,
    model_state_path: Path,
) -> None:
    manifest_path = tag_dir / "model_only_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("format") != "dllm_parallel.model_only_checkpoint.v1":
        raise RuntimeError("adapter-only checkpoint manifest format drift")
    if manifest.get("filename") != model_state_path.name:
        raise RuntimeError("adapter-only checkpoint filename differs from manifest")
    if int(manifest.get("bytes", -1)) != model_state_path.stat().st_size:
        raise RuntimeError("adapter-only checkpoint size differs from manifest")
    if _sha256_file(model_state_path) != manifest.get("sha256"):
        raise RuntimeError("adapter-only checkpoint hash differs from manifest")
    published = metadata.get("model_only_manifest")
    if published != manifest:
        raise RuntimeError("adapter-only root and tag manifests differ")


def _validate_replicated_export_topology(config: dict[str, Any]) -> None:
    topology = config.get("topology") or {}
    tp = int(topology.get("tensor_parallel_size", 1) or 1)
    ep = int(topology.get("expert_parallel_size", 1) or 1)
    if tp != 1 or ep != 1:
        raise RuntimeError(
            "DiffusionGemma merge currently requires replicated TP=1 and EP=1 "
            f"weights; found TP={tp}, EP={ep}. Shard reassembly is not implemented."
        )


def _validate_adapter_metadata(
    metadata: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    configured = config.get("adapter") or {}
    saved = (metadata.get("backbone_state") or {}).get("adapter") or {}
    if configured.get("type") != "lora":
        raise RuntimeError("checkpoint is not a LoRA run")
    fields = ("rank", "alpha", "dropout", "targets")
    for field in fields:
        left = configured.get(field)
        right = saved.get(field)
        if field == "targets":
            left = tuple(left or ())
            right = tuple(right or ())
        if left != right:
            raise RuntimeError(f"checkpoint adapter metadata differs for {field}")
    rank = int(saved.get("rank", 0))
    alpha = float(saved.get("alpha", 0.0))
    if rank <= 0 or not torch.isfinite(torch.tensor(alpha)):
        raise RuntimeError("checkpoint has an invalid LoRA rank or alpha")
    targets = tuple(str(value) for value in (saved.get("targets") or ()))
    if not targets:
        raise RuntimeError("DiffusionGemma checkpoint has no LoRA targets")
    if len(set(targets)) != len(targets):
        raise RuntimeError("DiffusionGemma checkpoint has duplicate LoRA targets")
    unknown_targets = sorted(set(targets) - _SUPPORTED_ADAPTER_TARGETS)
    if unknown_targets:
        raise RuntimeError(
            "DiffusionGemma checkpoint has unsupported LoRA targets: "
            + ", ".join(unknown_targets)
        )
    trainable = int(saved.get("trainable_parameters", 0))
    if trainable <= 0:
        raise RuntimeError("checkpoint has no recorded trainable adapter parameters")
    return {
        "rank": rank,
        "alpha": alpha,
        "dropout": float(saved.get("dropout", 0.0)),
        "targets": list(targets),
        "linear_modules": int(saved.get("linear_modules", 0)),
        "expert_modules": int(saved.get("expert_modules", 0)),
        "trainable_parameters": trainable,
    }


def _resolve_base_model(
    base_model: str | os.PathLike[str],
    *,
    expected_model_id: str,
    revision: str,
    local_files_only: bool,
) -> Path:
    candidate = Path(base_model).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    if str(base_model) != expected_model_id:
        raise RuntimeError(
            "remote base-model id differs from the checkpoint: "
            f"{base_model!s} != {expected_model_id}"
        )
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=expected_model_id,
            revision=revision,
            local_files_only=bool(local_files_only),
        )
    ).resolve()


def _validate_base_config(config: dict[str, Any]) -> None:
    if config.get("model_type") != "diffusion_gemma":
        raise RuntimeError("base artifact is not a DiffusionGemma model")
    text = config.get("text_config")
    if not isinstance(text, dict):
        raise RuntimeError("DiffusionGemma config has no text_config")
    layer_types = text.get("layer_types")
    if not isinstance(layer_types, list) or not layer_types:
        raise RuntimeError("DiffusionGemma text_config has no layer_types")
    if any(
        value not in {"sliding_attention", "full_attention"} for value in layer_types
    ):
        raise RuntimeError(
            "DiffusionGemma config has unsupported attention layer types"
        )
    declared = int(text.get("num_hidden_layers", len(layer_types)))
    if declared != len(layer_types):
        raise RuntimeError("DiffusionGemma layer_types length is inconsistent")


def _base_weight_map(base_root: Path) -> tuple[dict[str, str], dict[str, Any]]:
    index_path = base_root / _INDEX_FILENAME
    if index_path.is_file():
        payload = _read_json(index_path)
        raw = payload.get("weight_map")
        if not isinstance(raw, dict) or not raw:
            raise RuntimeError("base safetensors index has no weight map")
        weight_map = {str(name): str(filename) for name, filename in raw.items()}
        return weight_map, payload
    single = base_root / "model.safetensors"
    if not single.is_file():
        raise FileNotFoundError(
            f"base model has neither {_INDEX_FILENAME} nor model.safetensors"
        )
    from safetensors import safe_open

    with safe_open(str(single), framework="pt", device="cpu") as handle:
        names = list(handle.keys())
        total_size = sum(
            _dtype_size(str(handle.get_slice(name).get_dtype()))
            * _numel(tuple(int(value) for value in handle.get_slice(name).get_shape()))
            for name in names
        )
    weight_map = {name: single.name for name in names}
    return weight_map, {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }


def _safetensor_shapes(
    base_root: Path, weight_map: dict[str, str]
) -> dict[str, tuple[int, ...]]:
    from safetensors import safe_open

    shapes: dict[str, tuple[int, ...]] = {}
    for filename in sorted(set(weight_map.values())):
        expected = {name for name, value in weight_map.items() if value == filename}
        with safe_open(
            str(base_root / filename), framework="pt", device="cpu"
        ) as handle:
            actual = set(handle.keys())
            if actual != expected:
                raise RuntimeError(
                    f"base shard/index coverage differs for {filename}: "
                    f"missing={sorted(expected - actual)[:3]} "
                    f"extra={sorted(actual - expected)[:3]}"
                )
            for name in expected:
                shapes[name] = tuple(
                    int(value) for value in handle.get_slice(name).get_shape()
                )
    return shapes


def _build_merge_plan(
    base_config: dict[str, Any],
    shapes: dict[str, tuple[int, ...]],
    module_state: dict[str, Any],
    *,
    adapter_targets: frozenset[str],
    require_checkpoint_base: bool = True,
) -> tuple[dict[str, _MergeTarget], set[str], set[str]]:
    layer_types = list(base_config["text_config"]["layer_types"])
    targets: dict[str, _MergeTarget] = {}
    expected_adapters: set[str] = set()
    allowed_aliases: set[str] = set()

    for layer, layer_type in enumerate(layer_types):
        packed = f"_te_packed_layers.{layer}"
        hf = f"model.decoder.layers.{layer}"
        if "attention" in adapter_targets:
            qkv_components = ["q_proj", "k_proj"]
            v_key = f"{hf}.self_attn.v_proj.weight"
            if layer_type == "sliding_attention":
                qkv_components.append("v_proj")
                if v_key not in shapes:
                    raise RuntimeError(f"sliding layer {layer} is missing v_proj")
            elif v_key in shapes:
                raise RuntimeError(
                    f"full-attention layer {layer} unexpectedly has v_proj"
                )
            qkv_a = f"{packed}.qkv.lora_a"
            qkv_b = f"{packed}.qkv.lora_b"
            qkv_base = f"{packed}.qkv.base.weight"
            offset = 0
            for component in qkv_components:
                name = f"{hf}.self_attn.{component}.weight"
                shape = _required_shape(shapes, name, ndim=2)
                targets[name] = _MergeTarget(
                    adapter_a=qkv_a,
                    adapter_b=qkv_b,
                    checkpoint_base=qkv_base,
                    checkpoint_base_start=offset,
                    checkpoint_base_stop=offset + shape[0],
                )
                offset += shape[0]
            _validate_packed_projection(
                module_state,
                qkv_a,
                qkv_b,
                qkv_base,
                out_features=offset,
                in_features=_required_shape(
                    shapes,
                    f"{hf}.self_attn.q_proj.weight",
                    ndim=2,
                )[1],
                require_checkpoint_base=require_checkpoint_base,
            )
            expected_adapters.update((qkv_a, qkv_b))

            _add_direct_linear_target(
                targets,
                expected_adapters,
                shapes,
                module_state,
                output=f"{hf}.self_attn.o_proj.weight",
                adapter=f"{packed}.o_proj",
                require_checkpoint_base=require_checkpoint_base,
            )

        if "mlp" in adapter_targets:
            gate_up_a = f"{packed}.gate_up.lora_a"
            gate_up_b = f"{packed}.gate_up.lora_b"
            gate_up_base = f"{packed}.gate_up.base.weight"
            offset = 0
            for component in ("gate_proj", "up_proj"):
                name = f"{hf}.mlp.{component}.weight"
                shape = _required_shape(shapes, name, ndim=2)
                targets[name] = _MergeTarget(
                    adapter_a=gate_up_a,
                    adapter_b=gate_up_b,
                    checkpoint_base=gate_up_base,
                    checkpoint_base_start=offset,
                    checkpoint_base_stop=offset + shape[0],
                )
                offset += shape[0]
            _validate_packed_projection(
                module_state,
                gate_up_a,
                gate_up_b,
                gate_up_base,
                out_features=offset,
                in_features=_required_shape(
                    shapes,
                    f"{hf}.mlp.gate_proj.weight",
                    ndim=2,
                )[1],
                require_checkpoint_base=require_checkpoint_base,
            )
            expected_adapters.update((gate_up_a, gate_up_b))

            _add_direct_linear_target(
                targets,
                expected_adapters,
                shapes,
                module_state,
                output=f"{hf}.mlp.down_proj.weight",
                adapter=f"{packed}.down_proj",
                require_checkpoint_base=require_checkpoint_base,
            )

        if "experts" in adapter_targets:
            checkpoint_experts = f"model.model.decoder.layers.{layer}.experts"
            alias_experts = f"encoder.layers.{layer}.experts"
            for projection, a_name, b_name in (
                ("gate_up_proj", "gate_up_a", "gate_up_b"),
                ("down_proj", "down_a", "down_b"),
            ):
                output = f"{hf}.experts.{projection}"
                base_name = f"{checkpoint_experts}.{projection}"
                adapter_a = f"{checkpoint_experts}.lora_adapter.{a_name}"
                adapter_b = f"{checkpoint_experts}.lora_adapter.{b_name}"
                shape = _required_shape(shapes, output, ndim=3)
                _validate_grouped_projection(
                    module_state,
                    adapter_a,
                    adapter_b,
                    base_name,
                    shape,
                    require_checkpoint_base=require_checkpoint_base,
                )
                targets[output] = _MergeTarget(
                    adapter_a=adapter_a,
                    adapter_b=adapter_b,
                    checkpoint_base=base_name,
                    grouped=True,
                )
                expected_adapters.update((adapter_a, adapter_b))
                allowed_aliases.update(
                    (
                        f"{alias_experts}.lora_adapter.{a_name}",
                        f"{alias_experts}.lora_adapter.{b_name}",
                    )
                )
    return targets, expected_adapters, allowed_aliases


def _add_direct_linear_target(
    targets: dict[str, _MergeTarget],
    expected_adapters: set[str],
    shapes: dict[str, tuple[int, ...]],
    module_state: dict[str, Any],
    *,
    output: str,
    adapter: str,
    require_checkpoint_base: bool = True,
) -> None:
    shape = _required_shape(shapes, output, ndim=2)
    a_name = f"{adapter}.lora_a"
    b_name = f"{adapter}.lora_b"
    base_name = f"{adapter}.base.weight"
    _validate_packed_projection(
        module_state,
        a_name,
        b_name,
        base_name,
        out_features=shape[0],
        in_features=shape[1],
        require_checkpoint_base=require_checkpoint_base,
    )
    targets[output] = _MergeTarget(
        adapter_a=a_name,
        adapter_b=b_name,
        checkpoint_base=base_name,
    )
    expected_adapters.update((a_name, b_name))


def _validate_packed_projection(
    state: dict[str, Any],
    a_name: str,
    b_name: str,
    base_name: str,
    *,
    out_features: int,
    in_features: int,
    require_checkpoint_base: bool = True,
) -> None:
    a = _required_tensor(state, a_name)
    b = _required_tensor(state, b_name)
    base = _required_tensor(state, base_name) if require_checkpoint_base else None
    if a.ndim != 2 or b.ndim != 2 or (base is not None and base.ndim != 2):
        raise RuntimeError(f"packed LoRA projection has invalid rank: {a_name}")
    rank = int(a.shape[0])
    if (
        tuple(a.shape) != (rank, in_features)
        or tuple(b.shape) != (out_features, rank)
        or (base is not None and tuple(base.shape) != (out_features, in_features))
    ):
        raise RuntimeError(f"packed LoRA projection shape mismatch: {a_name}")


def _validate_grouped_projection(
    state: dict[str, Any],
    a_name: str,
    b_name: str,
    base_name: str,
    output_shape: tuple[int, ...],
    *,
    require_checkpoint_base: bool = True,
) -> None:
    a = _required_tensor(state, a_name)
    b = _required_tensor(state, b_name)
    base = _required_tensor(state, base_name) if require_checkpoint_base else None
    if a.ndim != 3 or b.ndim != 3 or (base is not None and base.ndim != 3):
        raise RuntimeError(f"grouped LoRA projection has invalid rank: {a_name}")
    experts, out_features, in_features = output_shape
    rank = int(a.shape[1])
    if (
        tuple(a.shape) != (experts, rank, in_features)
        or tuple(b.shape) != (experts, out_features, rank)
        or (base is not None and tuple(base.shape) != output_shape)
    ):
        raise RuntimeError(f"grouped LoRA projection shape mismatch: {a_name}")


def _validate_adapter_inventory(
    state: dict[str, Any],
    shared_params: dict[str, Any],
    expected: set[str],
    allowed_aliases: set[str],
    *,
    expected_rank: int,
    expected_trainable_parameters: int,
) -> dict[str, Any]:
    observed = {name for name in state if "lora" in name.lower()}
    missing = expected - observed
    extra = observed - expected - allowed_aliases
    if missing or extra:
        raise RuntimeError(
            "checkpoint LoRA tensor inventory differs: "
            f"missing={sorted(missing)[:3]} extra={sorted(extra)[:3]}"
        )
    for alias in sorted(observed & allowed_aliases):
        canonical = shared_params.get(alias)
        if canonical not in expected:
            raise RuntimeError(f"LoRA alias is not tied to a canonical tensor: {alias}")
        alias_tensor = _required_tensor(state, alias)
        canonical_tensor = _required_tensor(state, str(canonical))
        if not _tensor_alias_or_equal(alias_tensor, canonical_tensor):
            raise RuntimeError(
                f"LoRA tied alias differs from its canonical tensor: {alias}"
            )
    parameter_count = 0
    zero_b: list[str] = []
    for name in sorted(expected):
        tensor = _required_tensor(state, name)
        if not tensor.is_floating_point():
            raise RuntimeError(f"LoRA tensor is not floating point: {name}")
        if not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"LoRA tensor is nonfinite: {name}")
        if name.endswith(("lora_a", "gate_up_a", "down_a")):
            rank_dimension = 0 if tensor.ndim == 2 else 1
        else:
            rank_dimension = tensor.ndim - 1
        if int(tensor.shape[rank_dimension]) != int(expected_rank):
            raise RuntimeError(f"LoRA tensor rank differs from metadata: {name}")
        parameter_count += int(tensor.numel())
        if name.endswith(("lora_b", "gate_up_b", "down_b")) and not bool(
            torch.count_nonzero(tensor)
        ):
            zero_b.append(name)
    if parameter_count != int(expected_trainable_parameters):
        raise RuntimeError(
            "canonical LoRA parameter count differs from checkpoint metadata: "
            f"{parameter_count} != {expected_trainable_parameters}"
        )
    if len(zero_b) == sum(
        name.endswith(("lora_b", "gate_up_b", "down_b")) for name in expected
    ):
        raise RuntimeError("every LoRA B tensor is zero; the adapter was not trained")
    return {
        "tensor_count": len(expected),
        "parameter_count": parameter_count,
        "zero_b_tensors": zero_b,
    }


def _checkpoint_base_tensor(
    state: dict[str, Any],
    target: _MergeTarget,
) -> torch.Tensor:
    tensor = _required_tensor(state, target.checkpoint_base)
    if target.checkpoint_base_start is not None:
        tensor = tensor[
            int(target.checkpoint_base_start) : int(target.checkpoint_base_stop)
        ]
    return tensor


def _checkpoint_adapter_b_tensor(
    state: dict[str, Any],
    target: _MergeTarget,
) -> torch.Tensor:
    tensor = _required_tensor(state, target.adapter_b)
    if target.checkpoint_base_start is not None:
        tensor = tensor[
            int(target.checkpoint_base_start) : int(target.checkpoint_base_stop)
        ]
    return tensor


def _merge_weight(
    base: torch.Tensor,
    adapter_a: torch.Tensor,
    adapter_b: torch.Tensor,
    *,
    scale: float,
    grouped: bool,
    row_chunk_size: int,
    expert_chunk_size: int,
    name: str,
) -> torch.Tensor:
    if not base.is_floating_point() or not bool(torch.isfinite(base).all()):
        raise RuntimeError(f"base tensor is nonfinite or non-floating: {name}")
    output = base.clone()
    if grouped:
        for start in range(0, int(base.shape[0]), expert_chunk_size):
            stop = min(start + expert_chunk_size, int(base.shape[0]))
            delta = torch.bmm(
                adapter_b[start:stop].float(),
                adapter_a[start:stop].float(),
            ).mul_(float(scale))
            merged = base[start:stop].float().add_(delta)
            if not bool(torch.isfinite(merged).all()):
                raise RuntimeError(f"merged tensor is nonfinite: {name}")
            output[start:stop].copy_(merged.to(dtype=base.dtype))
        return output
    a_fp32 = adapter_a.float()
    for start in range(0, int(base.shape[0]), row_chunk_size):
        stop = min(start + row_chunk_size, int(base.shape[0]))
        delta = torch.mm(adapter_b[start:stop].float(), a_fp32).mul_(float(scale))
        merged = base[start:stop].float().add_(delta)
        if not bool(torch.isfinite(merged).all()):
            raise RuntimeError(f"merged tensor is nonfinite: {name}")
        output[start:stop].copy_(merged.to(dtype=base.dtype))
    return output


def _copy_nonweight_artifacts(source: Path, destination: Path) -> None:
    excluded = {
        _INDEX_FILENAME,
        _MANIFEST_FILENAME,
        "EXPORT_COMPLETE.json",
        "pytorch_model.bin.index.json",
    }
    for item in source.iterdir():
        if not item.is_file():
            continue
        if item.name in excluded:
            continue
        if item.suffix in {".safetensors", ".bin", ".pt"}:
            continue
        shutil.copy2(item, destination / item.name, follow_symlinks=True)


def _required_shape(
    shapes: dict[str, tuple[int, ...]],
    name: str,
    *,
    ndim: int,
) -> tuple[int, ...]:
    shape = shapes.get(name)
    if shape is None:
        raise RuntimeError(f"base model is missing required tensor: {name}")
    if len(shape) != ndim:
        raise RuntimeError(f"base tensor has invalid rank: {name}")
    return shape


def _required_tensor(state: dict[str, Any], name: str) -> torch.Tensor:
    value = state.get(name)
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"checkpoint is missing tensor: {name}")
    return value


def _tensor_alias_or_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    if tuple(left.shape) != tuple(right.shape) or left.dtype != right.dtype:
        return False
    try:
        if (
            left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
            and left.storage_offset() == right.storage_offset()
            and left.stride() == right.stride()
        ):
            return True
    except (AttributeError, RuntimeError):
        pass
    return bool(torch.equal(left, right))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON artifact must contain an object: {path}")
    return value


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _numel(shape: tuple[int, ...]) -> int:
    value = 1
    for dimension in shape:
        value *= int(dimension)
    return value


def _dtype_size(dtype: str) -> int:
    sizes = {
        "BOOL": 1,
        "U8": 1,
        "I8": 1,
        "F8_E4M3": 1,
        "F8_E5M2": 1,
        "I16": 2,
        "U16": 2,
        "F16": 2,
        "BF16": 2,
        "I32": 4,
        "U32": 4,
        "F32": 4,
        "I64": 8,
        "U64": 8,
        "F64": 8,
    }
    if dtype not in sizes:
        raise RuntimeError(f"unsupported safetensors dtype in artifact: {dtype}")
    return sizes[dtype]


def _emit(
    callback: Callable[[dict[str, Any]], None] | None,
    event: str,
    **payload: Any,
) -> None:
    if callback is not None:
        callback({"event": event, **payload})


def _print_event(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True), flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Merge and validate packed DiffusionGemma LoRA checkpoints."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("checkpoint_dir")
    merge_parser.add_argument("output_dir")
    merge_parser.add_argument("--base-model")
    merge_parser.add_argument("--tag", default="latest")
    merge_parser.add_argument("--local-files-only", action="store_true")
    merge_parser.add_argument("--row-chunk-size", type=int, default=4096)
    merge_parser.add_argument("--expert-chunk-size", type=int, default=1)
    merge_parser.add_argument("--no-verify-base-weights", action="store_true")
    merge_parser.add_argument("--no-checkpoint-hash", action="store_true")
    merge_parser.add_argument("--no-output-hash", action="store_true")
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("output_dir")
    validate_parser.add_argument("--no-hash", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "merge":
        result = merge_diffusion_gemma_lora_checkpoint(
            args.checkpoint_dir,
            args.output_dir,
            base_model=args.base_model,
            tag=args.tag,
            local_files_only=bool(args.local_files_only),
            row_chunk_size=int(args.row_chunk_size),
            expert_chunk_size=int(args.expert_chunk_size),
            verify_base_weights=not bool(args.no_verify_base_weights),
            hash_checkpoint=not bool(args.no_checkpoint_hash),
            hash_output=not bool(args.no_output_hash),
            progress=_print_event,
        )
        print(json.dumps(asdict(result), default=str, sort_keys=True))
        return
    result = validate_merged_diffusion_gemma_artifact(
        args.output_dir,
        verify_hashes=not bool(args.no_hash),
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
