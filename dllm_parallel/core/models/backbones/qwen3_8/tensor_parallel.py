# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Qwen3.8 tensor-parallel checkpoint layout.

Qwen3.8 stores the Gated DeltaNet Q, K, and V projections in one linear
parameter. A regular column split can cross a segment boundary and therefore
cannot represent local recurrent heads. This module registers one architecture-
specific Transformers TP style that shards each logical segment independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_SEGMENTED_STYLE = "dllm_qwen38_gdn_segmented_colwise"
_REPLICATED_NORM_STYLE = "dllm_qwen38_replicated_norm"


@dataclass(frozen=True)
class Qwen38TensorParallelLayout:
    tensor_parallel_size: int
    tensor_parallel_rank: int
    key_heads: int
    value_heads: int
    key_head_dim: int
    value_head_dim: int

    @property
    def qkv_segments(self) -> tuple[int, int, int]:
        key_width = self.key_heads * self.key_head_dim
        return key_width, key_width, self.value_heads * self.value_head_dim

    @property
    def local_key_heads(self) -> int:
        return self.key_heads // self.tensor_parallel_size

    @property
    def local_value_heads(self) -> int:
        return self.value_heads // self.tensor_parallel_size


def configure_qwen38_tensor_parallel(
    config: Any,
    runtime: Any,
) -> Qwen38TensorParallelLayout | None:
    """Install the exact Qwen3.8 TP load plan before ``from_pretrained``."""

    tp_size = int(getattr(runtime, "tensor_parallel_size", 1) or 1)
    if tp_size <= 1:
        return None
    tp_rank = int(getattr(runtime, "tensor_parallel_rank", 0) or 0)
    layout = Qwen38TensorParallelLayout(
        tensor_parallel_size=tp_size,
        tensor_parallel_rank=tp_rank,
        key_heads=int(config.linear_num_key_heads),
        value_heads=int(config.linear_num_value_heads),
        key_head_dim=int(config.linear_key_head_dim),
        value_head_dim=int(config.linear_value_head_dim),
    )
    for name, heads in (
        ("linear_num_key_heads", layout.key_heads),
        ("linear_num_value_heads", layout.value_heads),
        ("num_attention_heads", int(config.num_attention_heads)),
        ("num_key_value_heads", int(config.num_key_value_heads)),
    ):
        if heads % tp_size:
            raise ValueError(f"Qwen3.8 {name}={heads} must divide evenly by TP={tp_size}")

    _register_segmented_style(layout.qkv_segments)
    _register_replicated_norm_style()
    config.base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.self_attn.q_norm": _REPLICATED_NORM_STYLE,
        "layers.*.self_attn.k_norm": _REPLICATED_NORM_STYLE,
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
        "layers.*.linear_attn.in_proj_qkv": _SEGMENTED_STYLE,
        "layers.*.linear_attn.in_proj_z": "colwise",
        "layers.*.linear_attn.in_proj_b": "colwise",
        "layers.*.linear_attn.in_proj_a": "colwise",
        "layers.*.linear_attn.conv1d": _SEGMENTED_STYLE,
        "layers.*.linear_attn.A_log": "colwise",
        "layers.*.linear_attn.dt_bias": "colwise",
        "layers.*.linear_attn.norm": _REPLICATED_NORM_STYLE,
        "layers.*.linear_attn.out_proj": "rowwise",
    }
    config.use_cache = False
    return layout


def finalize_qwen38_tensor_parallel(
    model: Any,
    layout: Qwen38TensorParallelLayout | None,
) -> None:
    """Bind local recurrent-head metadata after sharded checkpoint loading."""

    if layout is None:
        return
    encoder = getattr(model, "model", None)
    if encoder is None:
        raise TypeError("Qwen3.8 text model must expose model.layers")
    # Install on the final loaded parameters, not the placeholders that HF
    # replaces while loading. Hooks reduce each fresh contribution exactly
    # once, before AccumulateGrad/ZeRO, never the accumulated .grad buffer.
    for module in encoder.modules():
        mesh = getattr(module, "_dllm_norm_tp_mesh", None)
        if mesh is not None:
            install_replicated_norm_gradient_sync(module, mesh.get_group(), int(mesh.size()))
    for layer in encoder.layers:
        mixer = getattr(layer, "linear_attn", None)
        if mixer is not None:
            mixer.num_k_heads = layout.local_key_heads
            mixer.num_v_heads = layout.local_value_heads
            mixer.key_dim = layout.local_key_heads * layout.key_head_dim
            mixer.value_dim = layout.local_value_heads * layout.value_head_dim
            mixer.conv_dim = mixer.key_dim * 2 + mixer.value_dim
            mixer.conv1d.in_channels = mixer.conv_dim
            mixer.conv1d.out_channels = mixer.conv_dim
            mixer.conv1d.groups = mixer.conv_dim
            expected_qkv = mixer.key_dim * 2 + mixer.value_dim
            if int(mixer.in_proj_qkv.weight.shape[0]) != expected_qkv:
                raise RuntimeError(
                    "Qwen3.8 segmented QKV checkpoint shard has an invalid shape: "
                    f"expected {expected_qkv}, got {mixer.in_proj_qkv.weight.shape[0]}"
                )
            if int(mixer.conv1d.weight.shape[0]) != mixer.conv_dim:
                raise RuntimeError("Qwen3.8 convolution shard does not match local GDN heads")
            for module in (
                mixer.in_proj_qkv,
                mixer.in_proj_z,
                mixer.in_proj_b,
                mixer.in_proj_a,
                mixer.conv1d,
                mixer.out_proj,
            ):
                _tag_sharded(module)
            mixer.A_log._dllm_tensor_parallel_sharded = True
            mixer.dt_bias._dllm_tensor_parallel_sharded = True
        attention = getattr(layer, "self_attn", None)
        if attention is not None:
            for module in (
                attention.q_proj,
                attention.k_proj,
                attention.v_proj,
                attention.o_proj,
            ):
                _tag_sharded(module)
        mlp = layer.mlp
        for module in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
            _tag_sharded(module)


def tag_qwen38_sequence_parallel_parameters(model: Any, runtime: Any) -> None:
    """Mark replicated parameters whose gradients span TP token shards."""

    if not bool(getattr(runtime, "sequence_parallel", False)):
        return
    if int(getattr(runtime, "tensor_parallel_size", 1) or 1) <= 1:
        return
    encoder = getattr(model, "encoder", None) or getattr(model, "model", None)
    if encoder is None:
        raise TypeError("Qwen3.8 training model must expose an encoder")
    modules = [encoder.norm]
    for layer in encoder.layers:
        modules.extend((layer.input_layernorm, layer.post_attention_layernorm))
        for output_projection in (
            getattr(getattr(layer, "self_attn", None), "o_proj", None),
            getattr(getattr(layer, "linear_attn", None), "out_proj", None),
            getattr(getattr(layer, "mlp", None), "down_proj", None),
        ):
            bias = getattr(output_projection, "bias", None)
            if bias is not None:
                bias._dllm_sequence_parallel_replicated = True
    for module in modules:
        for parameter in module.parameters():
            parameter._dllm_sequence_parallel_replicated = True


def _register_segmented_style(segment_sizes: tuple[int, ...]) -> None:
    try:
        import torch
        from torch import nn
        from transformers.integrations.tensor_parallel import (
            ALL_PARALLEL_STYLES,
            TensorParallelLayer,
            all_reduce_backward,
            distribute_module,
        )
    except ImportError as exc:
        raise RuntimeError("Qwen3.8 TP requires Transformers tensor parallelism") from exc

    class SegmentedColwiseParallel(TensorParallelLayer):
        def _prepare_input_fn(self, module, inputs, device_mesh):
            del module
            value = inputs[0] if inputs else inputs
            return all_reduce_backward(value, device_mesh)

        def _prepare_output_fn(self, module, outputs, device_mesh):
            del module, device_mesh
            return outputs

        def prepare_module_tp(self, module, device_mesh, **kwargs):
            del kwargs
            if isinstance(module, nn.Linear):
                distribute_module(
                    module,
                    device_mesh,
                    input_fn=self._prepare_input_fn,
                )
            return module

        def shard_tensor(self, param, tensor_idx=None, device=None, dtype=None):
            del tensor_idx
            shape = tuple(
                param.shape if isinstance(param, torch.Tensor) else param.get_shape()
            )
            if not shape or int(shape[0]) != sum(segment_sizes):
                raise ValueError(
                    "Qwen3.8 segmented TP expected leading dimension "
                    f"{sum(segment_sizes)}, got {shape}"
                )
            world_size = int(self.device_mesh.size())
            rank = int(self.rank)
            pieces = []
            offset = 0
            for segment_size in segment_sizes:
                if segment_size % world_size:
                    raise ValueError(
                        f"Qwen3.8 TP segment {segment_size} does not divide TP={world_size}"
                    )
                local_size = segment_size // world_size
                start = offset + rank * local_size
                stop = start + local_size
                slices = (slice(start, stop),) + (slice(None),) * (len(shape) - 1)
                pieces.append(param[slices])
                offset += segment_size
            return torch.cat(pieces, dim=0).to(device=device, dtype=dtype)

        def get_expected_sharded_shape(self, full_shape):
            shape = list(full_shape)
            shape[0] = sum(segment_sizes) // int(self.device_mesh.size())
            return tuple(shape)

        def update_module_attributes(self, module):
            local_width = sum(segment_sizes) // int(self.device_mesh.size())
            if isinstance(module, nn.Linear):
                module.out_features = local_width
            elif isinstance(module, nn.Conv1d):
                module.in_channels = local_width
                module.out_channels = local_width
                module.groups = local_width

    ALL_PARALLEL_STYLES.register(
        _SEGMENTED_STYLE,
        SegmentedColwiseParallel(),
    )
    ALL_PARALLEL_STYLES.register_plan_to_weight_dim(_SEGMENTED_STYLE, None)
    ALL_PARALLEL_STYLES.register_plan_to_bias_dim(_SEGMENTED_STYLE, None)


def install_replicated_norm_gradient_sync(module: Any, group: Any, world_size: int) -> None:
    """Sum fresh head-local norm gradients, safely across checkpoint replays/GAS."""
    if int(world_size) <= 1:
        return
    import torch
    import torch.distributed as dist

    def reduce_gradient(gradient):
        # Incoming gradients may alias other autograd values. Never mutate
        # them (or param.grad); this is a tiny head-dimension-sized buffer.
        reduced = gradient.clone(memory_format=torch.contiguous_format)
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM, group=group)
        return reduced

    for parameter in module.parameters(recurse=False):
        if parameter.requires_grad and not getattr(parameter, "_dllm_norm_tp_hook", False):
            parameter.register_hook(reduce_gradient)
            parameter._dllm_norm_tp_hook = True


def _register_replicated_norm_style() -> None:
    from transformers.integrations.tensor_parallel import (
        ALL_PARALLEL_STYLES,
        ReplicatedWithGradAllReduce,
    )

    class ReplicatedNorm(ReplicatedWithGradAllReduce):
        def prepare_module_tp(self, module, device_mesh, **kwargs):
            # Deliberately do not install HF's module backward hook: it
            # all-reduces the entire accumulated param.grad on every backward.
            module._dllm_norm_tp_mesh = device_mesh

    ALL_PARALLEL_STYLES.register(_REPLICATED_NORM_STYLE, ReplicatedNorm())
    ALL_PARALLEL_STYLES.register_plan_to_weight_dim(_REPLICATED_NORM_STYLE, None)
    ALL_PARALLEL_STYLES.register_plan_to_bias_dim(_REPLICATED_NORM_STYLE, None)


def _tag_sharded(module: Any) -> None:
    for parameter in module.parameters(recurse=False):
        parameter._dllm_tensor_parallel_sharded = True


__all__ = [
    "Qwen38TensorParallelLayout",
    "configure_qwen38_tensor_parallel",
    "finalize_qwen38_tensor_parallel",
    "tag_qwen38_sequence_parallel_parameters",
]
