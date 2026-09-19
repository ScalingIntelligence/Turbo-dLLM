# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from dllm_parallel.core.models.backbones.diffusiongemma.export import (
    _merge_weight,
    _resolve_deepspeed_model_state,
    merge_diffusion_gemma_lora_checkpoint,
    validate_merged_diffusion_gemma_artifact,
)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _synthetic_export_inputs(tmp_path: Path) -> dict:
    torch.manual_seed(41)
    base = tmp_path / "base"
    checkpoint = tmp_path / "checkpoint"
    tag_dir = checkpoint / "step_00000007"
    base.mkdir()
    tag_dir.mkdir(parents=True)
    config = {
        "model_type": "diffusion_gemma",
        "architectures": ["DiffusionGemmaForBlockDiffusion"],
        "text_config": {
            "num_hidden_layers": 2,
            "layer_types": ["sliding_attention", "full_attention"],
        },
    }
    _write_json(base / "config.json", config)
    (base / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")

    base_state: dict[str, torch.Tensor] = {
        "model.decoder.embed_tokens.weight": torch.randn(11, 4),
        "model.encoder.embed_vision.projection.weight": torch.randn(3, 3),
    }
    module_state: dict[str, torch.Tensor] = {}
    shared_params: dict[str, str] = {}
    expected: dict[str, torch.Tensor] = {}
    canonical_adapters: list[str] = []
    rank = 2
    scale = 2.0

    def adapter(prefix: str, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.randn(rank, weight.shape[-1]) * 0.1
        b = torch.randn(weight.shape[-2], rank) * 0.1
        module_state[f"{prefix}.lora_a"] = a
        module_state[f"{prefix}.lora_b"] = b
        module_state[f"{prefix}.base.weight"] = weight.clone()
        canonical_adapters.extend((f"{prefix}.lora_a", f"{prefix}.lora_b"))
        return a, b

    for layer, layer_type in enumerate(config["text_config"]["layer_types"]):
        hf = f"model.decoder.layers.{layer}"
        packed = f"_te_packed_layers.{layer}"
        projection_shapes = (
            [("q_proj", 4), ("k_proj", 2), ("v_proj", 2)]
            if layer_type == "sliding_attention"
            else [("q_proj", 6), ("k_proj", 2)]
        )
        qkv_weights = []
        for projection, rows in projection_shapes:
            name = f"{hf}.self_attn.{projection}.weight"
            base_state[name] = torch.randn(rows, 4)
            qkv_weights.append(base_state[name])
        qkv = torch.cat(qkv_weights, dim=0)
        qkv_a, qkv_b = adapter(f"{packed}.qkv", qkv)
        offset = 0
        for projection, rows in projection_shapes:
            name = f"{hf}.self_attn.{projection}.weight"
            expected[name] = base_state[name] + scale * torch.mm(
                qkv_b[offset : offset + rows], qkv_a
            )
            offset += rows

        o_name = f"{hf}.self_attn.o_proj.weight"
        base_state[o_name] = torch.randn(4, projection_shapes[0][1])
        o_a, o_b = adapter(f"{packed}.o_proj", base_state[o_name])
        expected[o_name] = base_state[o_name] + scale * torch.mm(o_b, o_a)

        gate_name = f"{hf}.mlp.gate_proj.weight"
        up_name = f"{hf}.mlp.up_proj.weight"
        base_state[gate_name] = torch.randn(3, 4)
        base_state[up_name] = torch.randn(3, 4)
        gate_up = torch.cat((base_state[gate_name], base_state[up_name]), dim=0)
        gate_a, gate_b = adapter(f"{packed}.gate_up", gate_up)
        expected[gate_name] = base_state[gate_name] + scale * torch.mm(
            gate_b[:3], gate_a
        )
        expected[up_name] = base_state[up_name] + scale * torch.mm(gate_b[3:], gate_a)

        down_name = f"{hf}.mlp.down_proj.weight"
        base_state[down_name] = torch.randn(4, 3)
        down_a, down_b = adapter(f"{packed}.down_proj", base_state[down_name])
        expected[down_name] = base_state[down_name] + scale * torch.mm(down_b, down_a)

        experts = f"model.model.decoder.layers.{layer}.experts"
        expert_shapes = {
            "gate_up_proj": (2, 6, 4, "gate_up_a", "gate_up_b"),
            "down_proj": (2, 4, 3, "down_a", "down_b"),
        }
        for projection, (
            count,
            rows,
            columns,
            a_suffix,
            b_suffix,
        ) in expert_shapes.items():
            output_name = f"{hf}.experts.{projection}"
            weight = torch.randn(count, rows, columns)
            base_state[output_name] = weight
            module_state[f"{experts}.{projection}"] = weight.clone()
            a = torch.randn(count, rank, columns) * 0.1
            b = torch.randn(count, rows, rank) * 0.1
            a_name = f"{experts}.lora_adapter.{a_suffix}"
            b_name = f"{experts}.lora_adapter.{b_suffix}"
            module_state[a_name] = a
            module_state[b_name] = b
            canonical_adapters.extend((a_name, b_name))
            alias_root = f"encoder.layers.{layer}.experts.lora_adapter"
            for canonical, alias in (
                (a_name, f"{alias_root}.{a_suffix}"),
                (b_name, f"{alias_root}.{b_suffix}"),
            ):
                module_state[alias] = module_state[canonical]
                shared_params[alias] = canonical
            expected[output_name] = weight + scale * torch.bmm(b, a)

    weights_path = base / "model.safetensors"
    save_file(base_state, str(weights_path))
    total_size = sum(
        tensor.numel() * tensor.element_size() for tensor in base_state.values()
    )
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": {name: weights_path.name for name in base_state},
    }
    _write_json(base / "model.safetensors.index.json", index)

    trainable = sum(module_state[name].numel() for name in canonical_adapters)
    adapter_metadata = {
        "rank": rank,
        "alpha": 4.0,
        "dropout": 0.0,
        "targets": ["attention", "mlp", "experts"],
        "linear_modules": 8,
        "expert_modules": 2,
        "trainable_parameters": trainable,
        "total_parameters": trainable + sum(t.numel() for t in base_state.values()),
    }
    run_config = {
        "model": {
            "id": "unit/diffusiongemma",
            "revision": "0123456789abcdef",
        },
        "adapter": {
            "type": "lora",
            **{k: adapter_metadata[k] for k in ("rank", "alpha", "dropout", "targets")},
        },
        "topology": {
            "tensor_parallel_size": 1,
            "expert_parallel_size": 1,
            "context_parallel_size": 2,
            "block_parallel_size": 1,
        },
    }
    metadata = {
        "format": "dllm_parallel.distributed_checkpoint.v1",
        "latest_tag": "step_00000007",
        "step": 7,
        "config": run_config,
        "backbone_state": {
            "family": "diffusion_gemma",
            "adapter": adapter_metadata,
        },
    }
    _write_json(checkpoint / "metadata.json", metadata)
    (checkpoint / "latest").write_text("step_00000007\n", encoding="utf-8")
    torch.save(
        {
            "step": 7,
            "module": module_state,
            "shared_params": shared_params,
            "config": run_config,
        },
        tag_dir / "mp_rank_00_model_states.pt",
    )
    return {
        "base": base,
        "checkpoint": checkpoint,
        "tag_dir": tag_dir,
        "base_state": base_state,
        "module_state": module_state,
        "expected": expected,
        "adapter_parameters": trainable,
    }


def test_diffusiongemma_export_merges_every_projection_and_round_trips(
    tmp_path,
) -> None:
    inputs = _synthetic_export_inputs(tmp_path)
    output = tmp_path / "merged"
    events: list[dict] = []

    result = merge_diffusion_gemma_lora_checkpoint(
        inputs["checkpoint"],
        output,
        base_model=inputs["base"],
        row_chunk_size=2,
        expert_chunk_size=1,
        progress=events.append,
    )

    assert result.checkpoint_step == 7
    assert result.modified_tensors == 17
    assert result.adapter_tensors == 24
    assert result.adapter_parameters == inputs["adapter_parameters"]
    assert result.output_shards == 1
    assert output.is_dir()
    assert not any(
        path.name.startswith(".merged.merge-") for path in tmp_path.iterdir()
    )
    assert (output / "tokenizer_config.json").exists()
    exported = load_file(str(output / "model.safetensors"), device="cpu")
    assert set(exported) == set(inputs["base_state"])
    assert not any("lora" in name.lower() for name in exported)
    for name, expected in inputs["expected"].items():
        torch.testing.assert_close(
            exported[name],
            expected,
            atol=1e-6,
            rtol=1e-6,
            msg=name,
        )
    for name, expected in inputs["base_state"].items():
        if name not in inputs["expected"]:
            torch.testing.assert_close(exported[name], expected, atol=0.0, rtol=0.0)
    validation = validate_merged_diffusion_gemma_artifact(output)
    assert validation == {
        "tensors": len(inputs["base_state"]),
        "shards": 1,
        "total_size_bytes": sum(
            tensor.numel() * tensor.element_size()
            for tensor in inputs["base_state"].values()
        ),
        "hashes_verified": True,
    }
    assert events[-1]["event"] == "artifact_validated"


def test_diffusiongemma_export_preserves_runtime_lora_numerics(tmp_path) -> None:
    inputs = _synthetic_export_inputs(tmp_path)
    output = tmp_path / "merged"
    merge_diffusion_gemma_lora_checkpoint(
        inputs["checkpoint"],
        output,
        base_model=inputs["base"],
        hash_checkpoint=False,
    )
    merged = load_file(str(output / "model.safetensors"), device="cpu")
    base = inputs["base_state"]["model.decoder.layers.0.self_attn.o_proj.weight"]
    state = inputs["module_state"]
    prefix = "_te_packed_layers.0.o_proj"
    hidden = torch.randn(5, base.shape[1])
    runtime = F.linear(hidden, base) + 2.0 * F.linear(
        F.linear(hidden, state[f"{prefix}.lora_a"]),
        state[f"{prefix}.lora_b"],
    )
    deployed = F.linear(
        hidden,
        merged["model.decoder.layers.0.self_attn.o_proj.weight"],
    )
    torch.testing.assert_close(deployed, runtime, atol=2e-6, rtol=2e-6)


def test_diffusiongemma_export_merges_model_only_adapter_checkpoint(tmp_path) -> None:
    inputs = _synthetic_export_inputs(tmp_path)
    model_path = inputs["tag_dir"] / "mp_rank_00_model_states.pt"
    state = torch.load(model_path, weights_only=True)
    state["module"] = {
        name: tensor for name, tensor in state["module"].items() if "lora" in name
    }
    state["model_state_scope"] = "trainable_parameters"
    torch.save(state, model_path)
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    model_manifest = {
        "format": "dllm_parallel.model_only_checkpoint.v1",
        "step": 7,
        "filename": model_path.name,
        "bytes": model_path.stat().st_size,
        "sha256": digest,
        "canonical_parameters": inputs["adapter_parameters"],
    }
    _write_json(inputs["tag_dir"] / "model_only_manifest.json", model_manifest)
    metadata_path = inputs["checkpoint"] / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["checkpoint_backend"] = "adapter_model_only"
    metadata["model_state_scope"] = "trainable_parameters"
    metadata["model_only_manifest"] = model_manifest
    _write_json(metadata_path, metadata)

    output = tmp_path / "adapter-only-merged"
    merge_diffusion_gemma_lora_checkpoint(
        inputs["checkpoint"],
        output,
        base_model=inputs["base"],
    )
    merged = load_file(str(output / "model.safetensors"), device="cpu")
    for name, expected in inputs["expected"].items():
        torch.testing.assert_close(merged[name], expected, atol=1e-6, rtol=1e-6)
    merge_manifest = json.loads(
        (output / "dllm_parallel_merge.json").read_text(encoding="utf-8")
    )
    assert merge_manifest["merge"]["base_weights_verified"] is False
    assert merge_manifest["merge"]["base_verification"] == (
        "pinned_huggingface_revision"
    )


def test_diffusiongemma_merge_accumulates_dense_and_expert_updates_in_fp32() -> None:
    torch.manual_seed(59)
    base = torch.randn(7, 5).to(torch.bfloat16)
    a = torch.randn(3, 5).to(torch.bfloat16)
    b = torch.randn(7, 3).to(torch.bfloat16)
    actual = _merge_weight(
        base,
        a,
        b,
        scale=1.75,
        grouped=False,
        row_chunk_size=2,
        expert_chunk_size=1,
        name="dense",
    )
    expected = (base.float() + 1.75 * torch.mm(b.float(), a.float())).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)

    expert_base = torch.randn(4, 6, 5).to(torch.bfloat16)
    expert_a = torch.randn(4, 3, 5).to(torch.bfloat16)
    expert_b = torch.randn(4, 6, 3).to(torch.bfloat16)
    expert_actual = _merge_weight(
        expert_base,
        expert_a,
        expert_b,
        scale=1.75,
        grouped=True,
        row_chunk_size=2,
        expert_chunk_size=2,
        name="experts",
    )
    expert_expected = (
        expert_base.float() + 1.75 * torch.bmm(expert_b.float(), expert_a.float())
    ).to(torch.bfloat16)
    torch.testing.assert_close(expert_actual, expert_expected, atol=0.0, rtol=0.0)


def test_diffusiongemma_export_rejects_wrong_base_and_cleans_temporary_dir(
    tmp_path,
) -> None:
    inputs = _synthetic_export_inputs(tmp_path)
    base_state = load_file(str(inputs["base"] / "model.safetensors"), device="cpu")
    base_state["model.decoder.layers.0.mlp.down_proj.weight"][0, 0] += 1.0
    save_file(base_state, str(inputs["base"] / "model.safetensors"))
    output = tmp_path / "merged"

    with pytest.raises(RuntimeError, match="differs from the frozen training weight"):
        merge_diffusion_gemma_lora_checkpoint(
            inputs["checkpoint"],
            output,
            base_model=inputs["base"],
            hash_checkpoint=False,
        )

    assert not output.exists()
    assert not any(
        path.name.startswith(".merged.merge-") for path in tmp_path.iterdir()
    )


def test_diffusiongemma_export_rejects_nonfinite_adapter(tmp_path) -> None:
    inputs = _synthetic_export_inputs(tmp_path)
    model_path = inputs["tag_dir"] / "mp_rank_00_model_states.pt"
    state = torch.load(model_path, weights_only=True)
    state["module"]["_te_packed_layers.0.qkv.lora_b"][0, 0] = float("nan")
    torch.save(state, model_path)

    with pytest.raises(RuntimeError, match="LoRA tensor is nonfinite"):
        merge_diffusion_gemma_lora_checkpoint(
            inputs["checkpoint"],
            tmp_path / "merged",
            base_model=inputs["base"],
            hash_checkpoint=False,
        )


def test_diffusiongemma_export_rejects_sharded_topology_and_overwrite(tmp_path) -> None:
    inputs = _synthetic_export_inputs(tmp_path)
    metadata_path = inputs["checkpoint"] / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["config"]["topology"]["expert_parallel_size"] = 2
    _write_json(metadata_path, metadata)
    with pytest.raises(RuntimeError, match="TP=1 and EP=1"):
        merge_diffusion_gemma_lora_checkpoint(
            inputs["checkpoint"],
            tmp_path / "merged",
            base_model=inputs["base"],
            hash_checkpoint=False,
        )

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        merge_diffusion_gemma_lora_checkpoint(
            inputs["checkpoint"],
            existing,
            base_model=inputs["base"],
            hash_checkpoint=False,
        )


def test_diffusiongemma_export_accepts_hardlinked_replicated_model_aliases(
    tmp_path,
) -> None:
    tag_dir = tmp_path / "step_00000001"
    tag_dir.mkdir()
    source = tag_dir / "mp_rank_00_model_states.pt"
    source.write_bytes(b"state")
    (tag_dir / "mp_rank_01_model_states.pt").hardlink_to(source)

    assert _resolve_deepspeed_model_state(tag_dir) == source


def test_diffusiongemma_artifact_validator_detects_corruption(tmp_path) -> None:
    inputs = _synthetic_export_inputs(tmp_path)
    output = tmp_path / "merged"
    merge_diffusion_gemma_lora_checkpoint(
        inputs["checkpoint"],
        output,
        base_model=inputs["base"],
        hash_checkpoint=False,
    )
    weights = output / "model.safetensors"
    payload = bytearray(weights.read_bytes())
    payload[-1] ^= 1
    weights.write_bytes(payload)

    with pytest.raises(RuntimeError, match="hash differs"):
        validate_merged_diffusion_gemma_artifact(output)
    with safe_open(str(weights), framework="pt", device="cpu") as handle:
        assert set(handle.keys())
