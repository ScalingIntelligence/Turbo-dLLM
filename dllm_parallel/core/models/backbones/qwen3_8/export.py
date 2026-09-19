# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Merge packed Qwen3.8 LoRA checkpoints into a deployable HF artifact.

Qwen3.8 training replaces the Hugging Face attention/GDN and MLP projections
with packed Transformer Engine linears before LoRA is installed.  The adapter
therefore cannot be merged with PEFT: each packed update must be split back
into the exact tensors from the pinned multimodal Hugging Face checkpoint.

Only TP=EP=1 checkpoints are supported.  CP/BP ranks hold replicated model
weights and DeepSpeed ZeRO-2 shards optimizer state only, so their model-state
files are expected to be hard-linked aliases of one canonical state.  Any
sharded or ambiguous checkpoint is rejected rather than partially exported.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch

# Reuse the already production-tested checkpoint, safetensors, hashing, and
# FP32 merge primitives.  Family-specific validation and tensor mapping remain
# in this module so a Qwen layout change cannot be mistaken for DiffusionGemma.
from dllm_parallel.core.models.backbones.diffusiongemma.export import (
    _CHECKPOINT_FORMAT,
    _INDEX_FILENAME,
    _MergeTarget,
    _add_direct_linear_target,
    _atomic_json_write,
    _base_weight_map,
    _checkpoint_adapter_b_tensor,
    _checkpoint_base_tensor,
    _copy_nonweight_artifacts,
    _dtype_size,
    _merge_weight,
    _numel,
    _read_json,
    _required_shape,
    _resolve_base_model,
    _resolve_checkpoint_tag,
    _resolve_deepspeed_model_state,
    _safetensor_shapes,
    _sha256_file,
    _validate_adapter_inventory,
    _validate_model_only_artifact,
    _validate_packed_projection,
)


_EXPORT_FORMAT = "dllm_parallel.qwen3_8_merged_lora.v1"
_MANIFEST_FILENAME = "dllm_parallel_merge.json"
_SUPPORTED_ADAPTER_TARGETS = frozenset({"attention", "mlp"})
_SUPPORTED_LAYER_TYPES = frozenset({"linear_attention", "full_attention"})
_FAST_DLLM_ARCHITECTURE = "Qwen3_5FastDLLMForCausalLM"
_FAST_DLLM_BLOCK_SIZE = 64
_FAST_DLLM_SMALL_BLOCK_SIZE = 8


@dataclass(frozen=True)
class Qwen38MergeResult:
    output_dir: Path
    checkpoint_tag: str
    checkpoint_step: int
    modified_tensors: int
    adapter_tensors: int
    adapter_parameters: int
    output_shards: int


def merge_qwen38_lora_checkpoint(
    checkpoint_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    base_model: str | os.PathLike[str] | None = None,
    tag: str = "latest",
    local_files_only: bool = False,
    row_chunk_size: int = 4096,
    verify_base_weights: bool = True,
    hash_checkpoint: bool = True,
    hash_output: bool = True,
    fast_dllm_v2_serving: bool = False,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> Qwen38MergeResult:
    """Merge one Qwen3.8 packed-LoRA checkpoint in bounded FP32 chunks."""

    row_chunk_size = int(row_chunk_size)
    if row_chunk_size <= 0:
        raise ValueError("row_chunk_size must be positive")

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
    if family != "qwen3_8":
        raise RuntimeError(f"checkpoint backbone is not Qwen3.8: {family!r}")
    checkpoint_backend = metadata.get("checkpoint_backend")
    if checkpoint_backend not in {None, "deepspeed_zero2", "adapter_model_only"}:
        raise RuntimeError(
            "packed Qwen3.8 export requires a DeepSpeed or adapter-only checkpoint"
        )
    adapter_model_only = checkpoint_backend == "adapter_model_only"
    trainable_model_state = metadata.get("model_state_scope") == (
        "trainable_parameters"
    )
    if adapter_model_only and not trainable_model_state:
        raise RuntimeError("adapter-only checkpoint has an invalid tensor scope")
    serving = (
        _fast_dllm_serving_metadata(metadata)
        if bool(fast_dllm_v2_serving)
        else None
    )

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

    run_config = metadata.get("config")
    if not isinstance(run_config, dict):
        raise RuntimeError("checkpoint metadata is missing its resolved run config")
    _validate_replicated_export_topology(run_config)
    adapter = _validate_adapter_metadata(metadata, run_config)
    model_config = run_config.get("model") or {}
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
    if destination == base_root or base_root in destination.parents:
        raise ValueError("output_dir must not replace or be inside the base model")
    base_config = _read_json(base_root / "config.json")
    _validate_base_config(base_config)
    weight_map, index_payload = _base_weight_map(base_root)
    tensor_shapes = _safetensor_shapes(base_root, weight_map)
    if serving is not None:
        weight_map = {
            name: shard
            for name, shard in weight_map.items()
            if not name.startswith("model.visual.")
        }
        index_payload = json.loads(json.dumps(index_payload))
        index_payload["weight_map"] = dict(weight_map)
        tensor_shapes = {
            name: shape for name, shape in tensor_shapes.items() if name in weight_map
        }
    targets, expected_adapters = _build_merge_plan(
        base_config,
        tensor_shapes,
        module_state,
        adapter_targets=frozenset(adapter["targets"]),
        require_checkpoint_base=not trainable_model_state,
    )
    layer_count = len(base_config["text_config"]["layer_types"])
    expected_linear_modules = 2 * layer_count * len(adapter["targets"])
    if int(adapter["linear_modules"]) != expected_linear_modules:
        raise RuntimeError(
            "checkpoint linear-adapter count differs from the Qwen3.8 layers"
        )
    if int(adapter["expert_modules"]) != 0:
        raise RuntimeError("Qwen3.8 checkpoint unexpectedly records expert adapters")
    inventory = _validate_adapter_inventory(
        module_state,
        checkpoint_state.get("shared_params") or {},
        expected_adapters,
        set(),
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
        if serving is not None:
            serving_config = dict(base_config)
            serving_config.update(serving)
            text_config = dict(serving_config["text_config"])
            text_config["mask_token_id"] = serving["mask_token_id"]
            serving_config["text_config"] = text_config
            _atomic_json_write(temporary / "config.json", serving_config)
        shard_hashes: dict[str, str] = {}
        shard_names = sorted(set(weight_map.values()))
        output_total_size = 0
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
                    if not trainable_model_state:
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
                        grouped=False,
                        row_chunk_size=row_chunk_size,
                        expert_chunk_size=1,
                        name=name,
                    )
            output_total_size += sum(
                tensor.numel() * tensor.element_size()
                for tensor in output_state.values()
            )
            output_path = temporary / shard_name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_shard = output_path.with_name(f".{output_path.name}.tmp")
            save_file(output_state, str(temporary_shard))
            os.replace(temporary_shard, output_path)
            del output_state
            if hash_output:
                shard_hashes[shard_name] = _sha256_file(output_path)

        if serving is not None:
            index_payload.setdefault("metadata", {})["total_size"] = (
                output_total_size
            )
        _atomic_json_write(temporary / _INDEX_FILENAME, index_payload)
        merge_manifest = {
            "format": _EXPORT_FORMAT,
            "base_model": {"id": model_id, "revision": revision},
            "checkpoint": {
                "path": str(checkpoint_root),
                "tag": resolved_tag,
                "step": checkpoint_step,
                "format": metadata["format"],
                "metadata_sha256": _sha256_file(metadata_path),
                "model_state_filename": model_state_path.name,
                "model_state_size_bytes": int(after_stat.st_size),
                "model_state_sha256": checkpoint_digest,
            },
            "adapter": adapter,
            "serving": serving,
            "merge": {
                "formula": "base_fp32 + (alpha / rank) * (B_fp32 @ A_fp32)",
                "accumulation_dtype": "float32",
                "output_dtype": "base_tensor_dtype",
                "modified_tensors": len(targets),
                "adapter_tensors": inventory["tensor_count"],
                "adapter_parameters": inventory["parameter_count"],
                "zero_b_tensors": inventory["zero_b_tensors"],
                "base_weights_verified": bool(
                    verify_base_weights and not trainable_model_state
                ),
                "base_verification": (
                    "pinned_huggingface_revision"
                    if trainable_model_state
                    else "checkpoint_frozen_weights_exact"
                    if verify_base_weights
                    else "disabled"
                ),
                "row_chunk_size": row_chunk_size,
            },
            "output": {
                "weight_index": _INDEX_FILENAME,
                "shards": shard_names,
                "sha256": shard_hashes,
            },
        }
        _atomic_json_write(temporary / _MANIFEST_FILENAME, merge_manifest)
        validation = validate_merged_qwen38_artifact(
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

    return Qwen38MergeResult(
        output_dir=destination,
        checkpoint_tag=resolved_tag,
        checkpoint_step=checkpoint_step,
        modified_tensors=len(targets),
        adapter_tensors=int(inventory["tensor_count"]),
        adapter_parameters=int(inventory["parameter_count"]),
        output_shards=len(set(weight_map.values())),
    )


def validate_merged_qwen38_artifact(
    output_dir: str | os.PathLike[str],
    *,
    verify_hashes: bool = True,
    require_fast_dllm_v2_serving: bool = False,
) -> dict[str, Any]:
    """Validate Qwen3.8 tensor coverage, byte size, and recorded hashes."""

    root = Path(output_dir).expanduser().resolve()
    manifest = _read_json(root / _MANIFEST_FILENAME)
    if manifest.get("format") != _EXPORT_FORMAT:
        raise RuntimeError("merged artifact has an unsupported manifest format")
    completion_path = root / "EXPORT_COMPLETE.json"
    if completion_path.is_file():
        completion = _read_json(completion_path)
        checkpoint = manifest.get("checkpoint") or {}
        checkpoint_path = Path(str(checkpoint.get("path", "")))
        marker_output = Path(str(completion.get("output_dir", ""))).expanduser()
        if (
            completion.get("status") != "complete"
            or marker_output.resolve() != root
            or completion.get("tag") != checkpoint.get("tag")
            or completion.get("run_id") != checkpoint_path.name
        ):
            raise RuntimeError(
                "merged artifact completion marker does not match its merge manifest"
            )
    config = _read_json(root / "config.json")
    _validate_base_config(config)
    serving = manifest.get("serving")
    if require_fast_dllm_v2_serving and serving is None:
        raise RuntimeError("merged artifact is not a Fast-dLLM serving artifact")
    if serving is not None:
        expected = _fast_dllm_serving_config_values(
            mask_token_id=int(serving.get("mask_token_id", -1))
        )
        if serving != expected:
            raise RuntimeError("merged artifact has invalid Fast-dLLM serving metadata")
        for key, value in expected.items():
            if config.get(key) != value:
                raise RuntimeError(
                    f"merged artifact config has invalid Fast-dLLM field: {key}"
                )
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
                total_size += _dtype_size(str(tensor_slice.get_dtype())) * _numel(shape)
    if observed != set(weight_map):
        raise RuntimeError("merged weight index does not cover the artifact tensors")
    if any("lora" in name.lower() for name in observed):
        raise RuntimeError("merged artifact unexpectedly contains LoRA tensors")
    if serving is not None and any(
        name.startswith("model.visual.") for name in observed
    ):
        raise RuntimeError("Fast-dLLM text artifact contains vision tensors")
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
            if _sha256_file(root / shard_name) != recorded_hashes[shard_name]:
                raise RuntimeError(f"merged shard hash differs for {shard_name}")
    return {
        "tensors": len(observed),
        "shards": len(shard_names),
        "total_size_bytes": total_size,
        "hashes_verified": bool(verify_hashes),
    }


def _fast_dllm_serving_config_values(*, mask_token_id: int) -> dict[str, Any]:
    return {
        "architectures": [_FAST_DLLM_ARCHITECTURE],
        "canvas_length": _FAST_DLLM_BLOCK_SIZE,
        "block_size": _FAST_DLLM_BLOCK_SIZE,
        "small_block_size": _FAST_DLLM_SMALL_BLOCK_SIZE,
        "mask_token_id": int(mask_token_id),
    }


def _fast_dllm_serving_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    objective = metadata.get("objective_state") or {}
    backbone = metadata.get("backbone_state") or {}
    if objective.get("kind") != "fast_dllm_v2":
        raise RuntimeError("Fast-dLLM serving export requires fast_dllm_v2 objective")
    if int(objective.get("block_size", -1)) != _FAST_DLLM_BLOCK_SIZE:
        raise RuntimeError("Fast-dLLM serving export requires trained block_size=64")
    # Objective state version 1 is the persisted contract implemented by
    # FastDLLMv2ObjectiveRuntime: complementary absorbing masks and a one-row
    # causal target shift. ``causal_target_shift`` itself is log-only and is
    # intentionally not serialized in checkpoints.
    if int(objective.get("version", -1)) != 1:
        raise RuntimeError("Fast-dLLM serving export requires objective state version=1")
    configured_objective = (metadata.get("config") or {}).get("objective") or {}
    if (
        configured_objective.get("name") != "fast_dllm_v2"
        or int(configured_objective.get("block_size", -1))
        != _FAST_DLLM_BLOCK_SIZE
    ):
        raise RuntimeError("checkpoint config does not match Fast-dLLM-v2 block_size=64")
    mask_token_id = backbone.get("mask_token_id")
    if mask_token_id is None:
        raise RuntimeError("Fast-dLLM serving export requires checkpoint mask_token_id")
    if int(objective.get("mask_token_id", mask_token_id)) != int(mask_token_id):
        raise RuntimeError("checkpoint objective and backbone mask_token_id differ")
    return _fast_dllm_serving_config_values(mask_token_id=int(mask_token_id))


def _validate_replicated_export_topology(config: dict[str, Any]) -> None:
    topology = config.get("topology") or {}
    tp = int(topology.get("tensor_parallel_size", 1) or 1)
    ep = int(topology.get("expert_parallel_size", 1) or 1)
    if tp != 1 or ep != 1:
        raise RuntimeError(
            "Qwen3.8 merge requires replicated TP=1 and EP=1 weights; "
            f"found TP={tp}, EP={ep}. Shard reassembly is not implemented."
        )


def _validate_adapter_metadata(
    metadata: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    configured = config.get("adapter") or {}
    saved = (metadata.get("backbone_state") or {}).get("adapter") or {}
    if configured.get("type") != "lora":
        raise RuntimeError("checkpoint is not a LoRA run")
    for field in ("rank", "alpha", "dropout", "targets"):
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
        raise RuntimeError("Qwen3.8 checkpoint has no LoRA targets")
    if len(set(targets)) != len(targets):
        raise RuntimeError("Qwen3.8 checkpoint has duplicate LoRA targets")
    unknown = sorted(set(targets) - _SUPPORTED_ADAPTER_TARGETS)
    if unknown:
        raise RuntimeError(
            "Qwen3.8 checkpoint has unsupported LoRA targets: " + ", ".join(unknown)
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


def _validate_base_config(config: dict[str, Any]) -> None:
    if config.get("model_type") != "qwen3_5":
        raise RuntimeError("base artifact is not the Qwen3.8/Qwen3.5 architecture")
    text = config.get("text_config")
    if not isinstance(text, dict):
        raise RuntimeError("Qwen3.8 config has no text_config")
    layer_types = text.get("layer_types")
    if not isinstance(layer_types, list) or not layer_types:
        raise RuntimeError("Qwen3.8 text_config has no layer_types")
    unknown = sorted(set(str(value) for value in layer_types) - _SUPPORTED_LAYER_TYPES)
    if unknown:
        raise RuntimeError(
            "Qwen3.8 config has unsupported layer types: " + ", ".join(unknown)
        )
    declared = int(text.get("num_hidden_layers", len(layer_types)))
    if declared != len(layer_types):
        raise RuntimeError("Qwen3.8 layer_types length is inconsistent")


def _build_merge_plan(
    base_config: dict[str, Any],
    shapes: dict[str, tuple[int, ...]],
    module_state: dict[str, Any],
    *,
    adapter_targets: frozenset[str],
    require_checkpoint_base: bool = True,
) -> tuple[dict[str, _MergeTarget], set[str]]:
    layer_types = list(base_config["text_config"]["layer_types"])
    targets: dict[str, _MergeTarget] = {}
    expected_adapters: set[str] = set()

    for layer, layer_type in enumerate(layer_types):
        packed = f"_te_packed_layers.{layer}"
        hf = f"model.language_model.layers.{layer}"
        if "attention" in adapter_targets:
            if layer_type == "full_attention":
                components = tuple(
                    f"{hf}.self_attn.{name}.weight"
                    for name in ("q_proj", "k_proj", "v_proj")
                )
                output = f"{hf}.self_attn.o_proj.weight"
            elif layer_type == "linear_attention":
                components = tuple(
                    f"{hf}.linear_attn.{name}.weight"
                    for name in (
                        "in_proj_qkv",
                        "in_proj_z",
                        "in_proj_b",
                        "in_proj_a",
                    )
                )
                output = f"{hf}.linear_attn.out_proj.weight"
            else:  # guarded by _validate_base_config
                raise RuntimeError(f"unsupported Qwen3.8 layer type: {layer_type}")
            _add_packed_component_targets(
                targets,
                expected_adapters,
                shapes,
                module_state,
                outputs=components,
                adapter=f"{packed}.mixer_input",
                require_checkpoint_base=require_checkpoint_base,
            )
            _add_direct_linear_target(
                targets,
                expected_adapters,
                shapes,
                module_state,
                output=output,
                adapter=f"{packed}.mixer_output",
                require_checkpoint_base=require_checkpoint_base,
            )

        if "mlp" in adapter_targets:
            _add_packed_component_targets(
                targets,
                expected_adapters,
                shapes,
                module_state,
                outputs=tuple(
                    f"{hf}.mlp.{name}.weight" for name in ("gate_proj", "up_proj")
                ),
                adapter=f"{packed}.gate_up",
                require_checkpoint_base=require_checkpoint_base,
            )
            _add_direct_linear_target(
                targets,
                expected_adapters,
                shapes,
                module_state,
                output=f"{hf}.mlp.down_proj.weight",
                adapter=f"{packed}.down_proj",
                require_checkpoint_base=require_checkpoint_base,
            )
    return targets, expected_adapters


def _add_packed_component_targets(
    targets: dict[str, _MergeTarget],
    expected_adapters: set[str],
    shapes: dict[str, tuple[int, ...]],
    module_state: dict[str, Any],
    *,
    outputs: tuple[str, ...],
    adapter: str,
    require_checkpoint_base: bool,
) -> None:
    if not outputs:
        raise RuntimeError("packed Qwen3.8 projection has no components")
    component_shapes = tuple(_required_shape(shapes, name, ndim=2) for name in outputs)
    in_features = component_shapes[0][1]
    if any(shape[1] != in_features for shape in component_shapes):
        raise RuntimeError(f"packed projection input widths differ: {outputs}")
    a_name = f"{adapter}.lora_a"
    b_name = f"{adapter}.lora_b"
    base_name = f"{adapter}.base.weight"
    out_features = sum(shape[0] for shape in component_shapes)
    _validate_packed_projection(
        module_state,
        a_name,
        b_name,
        base_name,
        out_features=out_features,
        in_features=in_features,
        require_checkpoint_base=require_checkpoint_base,
    )
    offset = 0
    for name, shape in zip(outputs, component_shapes, strict=True):
        targets[name] = _MergeTarget(
            adapter_a=a_name,
            adapter_b=b_name,
            checkpoint_base=base_name,
            checkpoint_base_start=offset,
            checkpoint_base_stop=offset + shape[0],
        )
        offset += shape[0]
    expected_adapters.update((a_name, b_name))


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
        description="Merge and validate packed Qwen3.8 LoRA checkpoints."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("checkpoint_dir")
    merge_parser.add_argument("output_dir")
    merge_parser.add_argument("--base-model")
    merge_parser.add_argument("--tag", default="latest")
    merge_parser.add_argument("--local-files-only", action="store_true")
    merge_parser.add_argument("--row-chunk-size", type=int, default=4096)
    merge_parser.add_argument("--no-verify-base-weights", action="store_true")
    merge_parser.add_argument("--no-checkpoint-hash", action="store_true")
    merge_parser.add_argument("--no-output-hash", action="store_true")
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("output_dir")
    validate_parser.add_argument("--no-hash", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "merge":
        result = merge_qwen38_lora_checkpoint(
            args.checkpoint_dir,
            args.output_dir,
            base_model=args.base_model,
            tag=args.tag,
            local_files_only=bool(args.local_files_only),
            row_chunk_size=int(args.row_chunk_size),
            verify_base_weights=not bool(args.no_verify_base_weights),
            hash_checkpoint=not bool(args.no_checkpoint_hash),
            hash_output=not bool(args.no_output_hash),
            progress=_print_event,
        )
        print(json.dumps(asdict(result), default=str, sort_keys=True))
        return
    result = validate_merged_qwen38_artifact(
        args.output_dir,
        verify_hashes=not bool(args.no_hash),
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
