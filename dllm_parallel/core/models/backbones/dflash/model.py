# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Optimized verifier-conditioned DFlash drafter."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from dllm_parallel.core.attention.dflash_fa4 import DFlashIntervalPlanCache
from dllm_parallel.core.attention.layout import (
    cat_intervals,
    clean_intervals_for_runtime,
)

from dllm_parallel.core.attention.masks import (
    DFlashGlobalContextMask,
    DFlashLocalBlockMask,
)
from dllm_parallel.core.objectives.dflash import DFlashObjectiveBatch
from dllm_parallel.core.kernels.tiled_linear_cross_entropy import (
    tiled_linear_cross_entropy,
)
from dllm_parallel.core.kernels.dflash_kl import frozen_linear_kl
from dllm_parallel.core.parallel.runtime import loss_scale as runtime_loss_scale
from dllm_parallel.core.models.loss import zero_loss_with_module_parameters
from dllm_parallel.core.profiling.operator_trace import operator_scope


DFlashAttentionOp = Callable[..., torch.Tensor]


@dataclass(frozen=True)
class DFlashModelConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    target_hidden_size: int
    target_layer_ids: tuple[int, ...]
    verifier_vocab_size: int
    draft_vocab_size: int
    block_size: int
    mask_token_id: int
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: dict[str, Any] | None
    pad_token_id: int | None
    attention_bias: bool
    mlp_bias: bool
    attention_dropout: float
    hidden_activation: str
    sliding_window: int | None
    layer_types: tuple[str, ...]
    sliding_window_non_causal: bool
    sample_from_anchor: bool
    final_logit_softcapping: float | None = None
    architecture: str = "DFlashDraftModel"
    output_multiplier: float = 1.0
    conv_kernel_size: int | None = None
    conv_group_size: int | None = None
    selector_rank: int | None = None
    selector_top_k: int | None = None
    initializer_range: float = 0.02

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "DFlashModelConfig":
        layer = values.get("transformer_layer_config")
        if layer is None:
            layer = values
        if not isinstance(layer, dict):
            raise ValueError("DFlash transformer configuration must be a mapping")
        rope_parameters = layer.get("rope_parameters", layer.get("rope_scaling"))
        if rope_parameters is not None and not isinstance(rope_parameters, dict):
            raise ValueError("DFlash RoPE parameters must be a mapping")
        hidden_size = _required_int(layer, "hidden_size")
        attention_heads = _required_int(layer, "num_attention_heads")
        head_dim = int(layer.get("head_dim") or hidden_size // attention_heads)
        target_layer_ids = dflash_target_layer_ids(values)
        if not target_layer_ids:
            raise ValueError("DFlash config requires target hidden-state layers")
        num_layers = _required_int(layer, "num_hidden_layers")
        layer_types = tuple(
            layer.get("layer_types") or ("full_attention",) * num_layers
        )
        method = _dflash_section(values)
        architectures = values.get("architectures") or ()
        architecture = str(architectures[0]) if architectures else "DFlashDraftModel"
        config = cls(
            hidden_size=hidden_size,
            intermediate_size=_required_int(layer, "intermediate_size"),
            num_hidden_layers=num_layers,
            num_attention_heads=attention_heads,
            num_key_value_heads=int(
                layer.get("num_key_value_heads") or attention_heads
            ),
            head_dim=head_dim,
            target_hidden_size=int(values.get("target_hidden_size") or hidden_size),
            target_layer_ids=target_layer_ids,
            verifier_vocab_size=_required_int(layer, "vocab_size"),
            draft_vocab_size=int(values.get("draft_vocab_size") or layer["vocab_size"]),
            block_size=dflash_block_size(values),
            mask_token_id=dflash_mask_token_id(values),
            rms_norm_eps=float(layer.get("rms_norm_eps", 1.0e-6)),
            rope_theta=float(
                (rope_parameters or {}).get(
                    "rope_theta",
                    layer.get("rope_theta", 10_000.0),
                )
            ),
            rope_scaling=(
                dict(rope_parameters) if rope_parameters is not None else None
            ),
            pad_token_id=(
                int(layer["pad_token_id"])
                if layer.get("pad_token_id") is not None
                else None
            ),
            attention_bias=bool(layer.get("attention_bias", False)),
            mlp_bias=bool(layer.get("mlp_bias", False)),
            attention_dropout=float(layer.get("attention_dropout", 0.0)),
            hidden_activation=str(layer.get("hidden_act", "silu")),
            sliding_window=(
                int(layer["sliding_window"])
                if layer.get("sliding_window") is not None
                else None
            ),
            layer_types=layer_types,
            sliding_window_non_causal=_sliding_window_non_causal(values, method),
            sample_from_anchor=bool(values.get("sample_from_anchor", False)),
            final_logit_softcapping=(
                float(
                    method.get(
                        "final_logit_softcapping", values.get("final_logit_softcapping")
                    )
                )
                if method.get(
                    "final_logit_softcapping", values.get("final_logit_softcapping")
                )
                is not None
                else None
            ),
            architecture=architecture,
            output_multiplier=float(method.get("output_multiplier", 1.0)),
            conv_kernel_size=_optional_int(method, "conv_kernel_size"),
            conv_group_size=_optional_int(method, "conv_group_size"),
            selector_rank=_optional_int(method, "selector_rank"),
            selector_top_k=_optional_int(method, "selector_top_k"),
            initializer_range=float(layer.get("initializer_range", 0.02)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        positive = {
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "target_hidden_size": self.target_hidden_size,
            "verifier_vocab_size": self.verifier_vocab_size,
            "draft_vocab_size": self.draft_vocab_size,
            "block_size": self.block_size,
        }
        for name, value in positive.items():
            if int(value) <= 0:
                raise ValueError(f"DFlash {name} must be positive")
        if self.block_size < 2:
            raise ValueError("DFlash block_size must be at least two")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("DFlash query heads must divide evenly by K/V heads")
        if self.head_dim % 2:
            raise ValueError("DFlash rotary attention requires an even head_dim")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("DFlash layer_types must match num_hidden_layers")
        if self.sliding_window is None and any(
            kind == "sliding_attention" for kind in self.layer_types
        ):
            raise ValueError("DFlash sliding layers require sliding_window")
        if len(set(self.target_layer_ids)) != len(self.target_layer_ids):
            raise ValueError("DFlash target layer IDs must be unique")
        if self.attention_dropout != 0.0:
            raise ValueError("production DFlash attention requires attention_dropout=0")
        if self.hidden_activation != "silu":
            raise ValueError("production DFlash currently requires hidden_act=silu")
        if (
            self.final_logit_softcapping is not None
            and self.final_logit_softcapping <= 0.0
        ):
            raise ValueError("DFlash final_logit_softcapping must be positive")
        if self.rope_scaling:
            rope_type = self.rope_scaling.get(
                "rope_type", self.rope_scaling.get("type")
            )
            if rope_type not in {None, "default"}:
                raise ValueError(
                    f"DFlash does not support rope scaling type {rope_type!r}"
                )
        if self.architecture not in {
            "DFlashDraftModel",
            "DFlash2DraftModel",
            "DFlashSpeculator",
        }:
            raise ValueError(f"unsupported DFlash architecture {self.architecture!r}")
        if self.output_multiplier <= 0.0:
            raise ValueError("DFlash output_multiplier must be positive")
        if self.initializer_range <= 0.0:
            raise ValueError("DFlash initializer_range must be positive")
        if self.architecture == "DFlash2DraftModel":
            required = {
                "conv_kernel_size": self.conv_kernel_size,
                "conv_group_size": self.conv_group_size,
                "selector_rank": self.selector_rank,
                "selector_top_k": self.selector_top_k,
            }
            for name, value in required.items():
                if value is None or int(value) <= 0:
                    raise ValueError(f"DFlash2 requires positive dflash_config.{name}")
            assert self.conv_kernel_size is not None
            assert self.conv_group_size is not None
            assert self.selector_top_k is not None
            if self.conv_kernel_size > self.block_size:
                raise ValueError("DFlash2 conv_kernel_size must not exceed block_size")
            if self.hidden_size % self.conv_group_size:
                raise ValueError("DFlash2 conv_group_size must divide hidden_size")
            if self.selector_top_k > self.verifier_vocab_size:
                raise ValueError(
                    "DFlash2 selector_top_k must not exceed vocabulary size"
                )
            if self.draft_vocab_size != self.verifier_vocab_size:
                raise ValueError("DFlash2 requires the verifier's full vocabulary")


def _dflash_section(values: dict[str, Any]) -> dict[str, Any]:
    section = values.get("dflash_config")
    if section is None:
        # Speculators checkpoints published before the nested DFlash serving
        # contract keep block/convolution/selector fields at the top level.
        # Treat the root mapping as the method section so those checkpoints
        # remain directly trainable without rewriting their config.json.
        return values
    if not isinstance(section, dict):
        raise ValueError("DFlash dflash_config must be a mapping")
    return section


def dflash_block_size(values: dict[str, Any]) -> int:
    block_size = values.get("block_size")
    if block_size is None:
        block_size = _required_int(_dflash_section(values), "block_size")
    return int(block_size)


def dflash_mask_token_id(values: dict[str, Any]) -> int:
    mask_token_id = values.get("mask_token_id")
    if mask_token_id is None:
        mask_token_id = _required_int(_dflash_section(values), "mask_token_id")
    return int(mask_token_id)


def dflash_target_layer_ids(values: dict[str, Any]) -> tuple[int, ...]:
    layer_ids = values.get("aux_hidden_state_layer_ids")
    if layer_ids:
        return tuple(int(item) for item in layer_ids)
    layer_ids = values.get("target_layer_ids")
    if layer_ids:
        return tuple(int(item) for item in layer_ids)
    # Native z-lab checkpoints index the Transformers hidden-state tuple
    # before its embedding-output entry; canonical feature IDs are +1.
    return tuple(
        int(item) + 1 for item in _dflash_section(values).get("target_layer_ids", ())
    )


@dataclass(frozen=True)
class DFlashForwardOutput:
    hidden_states: torch.Tensor
    teacher_hidden_states: torch.Tensor | None
    target_token_ids: torch.Tensor
    supervised_mask: torch.Tensor
    position_weights: torch.Tensor


@dataclass(frozen=True)
class RotaryFactors:
    cos: torch.Tensor
    sin: torch.Tensor


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = float(eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(
            hidden_states,
            (int(hidden_states.shape[-1]),),
            self.weight,
            self.eps,
        )


class DFlashMLP(nn.Module):
    def __init__(self, config: DFlashModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=config.mlp_bias,
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=config.mlp_bias,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=config.mlp_bias,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        with operator_scope("dflash.mlp"):
            return self.down_proj(
                F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
            )


class DFlashAttention(nn.Module):
    def __init__(
        self,
        config: DFlashModelConfig,
        *,
        layer_idx: int,
        attention_op: DFlashAttentionOp,
    ) -> None:
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = config.head_dim**-0.5
        self.block_size = config.block_size
        self.sliding_window = (
            config.sliding_window
            if config.layer_types[layer_idx] == "sliding_attention"
            else None
        )
        self.local_causal = (
            self.sliding_window is not None and not config.sliding_window_non_causal
        )
        self.attention_op = attention_op
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps)

    def _project_draft_qkv(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.q_proj(hidden_states),
            self.k_proj(hidden_states),
            self.v_proj(hidden_states),
        )

    def _project_context_kv(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _fused_linear_pair(hidden_states, self.k_proj, self.v_proj)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        *,
        objective: DFlashObjectiveBatch,
        target_rotary: RotaryFactors,
        draft_rotary: RotaryFactors,
        runtime: Any | None,
        global_target_length: int,
        interval_plan_cache: DFlashIntervalPlanCache,
        target_intervals: tuple[tuple[int, int], ...] | None = None,
        target_intervals_by_owner: (
            tuple[tuple[tuple[int, int], ...], ...] | None
        ) = None,
    ) -> torch.Tensor:
        batch_size, query_length, _ = hidden_states.shape
        target_length = target_hidden.shape[1]
        with operator_scope("dflash.draft_qkv_projection"):
            query_states, local_key_states, local_value_states = (
                self._project_draft_qkv(hidden_states)
            )
            query = self.q_norm(
                query_states.view(
                    batch_size,
                    query_length,
                    self.num_attention_heads,
                    self.head_dim,
                )
            )
            local_key = self.k_norm(
                local_key_states.view(
                    batch_size,
                    query_length,
                    self.num_key_value_heads,
                    self.head_dim,
                )
            )
            local_value = local_value_states.view(
                batch_size,
                query_length,
                self.num_key_value_heads,
                self.head_dim,
            )
        with operator_scope("dflash.context_kv_projection"):
            context_key_states, context_value_states = self._project_context_kv(
                target_hidden
            )
            context_key = self.k_norm(
                context_key_states.view(
                    batch_size,
                    target_length,
                    self.num_key_value_heads,
                    self.head_dim,
                )
            )
            context_value = context_value_states.view(
                batch_size,
                target_length,
                self.num_key_value_heads,
                self.head_dim,
            )
        with operator_scope("dflash.rotary"):
            query = apply_rotary(query, draft_rotary)
            local_key = apply_rotary(local_key, draft_rotary)
            context_key = apply_rotary(context_key, target_rotary)
        context_starts = objective.document_starts
        global_mask = DFlashGlobalContextMask(
            context_starts=context_starts,
            context_stops=objective.context_stops,
            anchor_valid=objective.anchor_valid,
            block_size=self.block_size,
            global_anchor_count=objective.global_anchor_count,
            sliding_window=self.sliding_window,
            debug_nonfinite_attention=bool(
                getattr(
                    getattr(runtime, "cp_bp_policy", None),
                    "debug_nonfinite_attention",
                    False,
                )
            ),
        )
        local_mask = DFlashLocalBlockMask(
            anchor_valid=objective.anchor_valid,
            block_size=self.block_size,
            causal=self.local_causal,
            debug_nonfinite_attention=global_mask.debug_nonfinite_attention,
        )
        attention_kwargs = {
            "query": query,
            "local_key": local_key,
            "local_value": local_value,
            "global_key": context_key,
            "global_value": context_value,
            "local_attn_mask": local_mask,
            "global_attn_mask": global_mask,
            "interval_plan_cache": interval_plan_cache,
            "scale": self.scaling,
            "runtime": runtime,
            "global_seq_len": int(global_target_length),
        }
        if target_intervals is not None:
            attention_kwargs["global_intervals"] = target_intervals
        if target_intervals_by_owner is not None:
            attention_kwargs["global_intervals_by_owner"] = target_intervals_by_owner
        with operator_scope("dflash.attention"):
            output = self.attention_op(
                **attention_kwargs,
            )
        with operator_scope("dflash.output_projection"):
            return self.o_proj(output.reshape(
                batch_size, query_length, self.num_attention_heads * self.head_dim
            ))


def _fused_linear_pair(
    hidden_states: torch.Tensor,
    first: nn.Linear,
    second: nn.Linear,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run equal-input projections as one wider GEMM without changing keys."""

    if int(first.in_features) != int(second.in_features):
        raise ValueError("fused DFlash projections must share their input width")
    weight = torch.cat((first.weight, second.weight), dim=0)
    if (first.bias is None) != (second.bias is None):
        raise ValueError("fused DFlash projections must share their bias policy")
    bias = (
        None
        if first.bias is None
        else torch.cat((first.bias, second.bias), dim=0)
    )
    projected = F.linear(hidden_states, weight, bias)
    return projected.split((int(first.out_features), int(second.out_features)), dim=-1)


class DFlashDecoderLayer(nn.Module):
    def __init__(
        self,
        config: DFlashModelConfig,
        *,
        layer_idx: int,
        attention_op: DFlashAttentionOp,
    ) -> None:
        super().__init__()
        self.self_attn = DFlashAttention(
            config,
            layer_idx=layer_idx,
            attention_op=attention_op,
        )
        self.mlp = DFlashMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states),
            target_hidden,
            **kwargs,
        )
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            theta
            ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / float(head_dim))
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def prepare(
        self,
        positions: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> RotaryFactors:
        frequencies = torch.einsum(
            "bs,d->bsd",
            positions.to(self.inv_freq.dtype),
            self.inv_freq,
        )
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        return RotaryFactors(
            cos=embedding.cos().unsqueeze(2).to(dtype),
            sin=embedding.sin().unsqueeze(2).to(dtype),
        )


class DFlashModel(nn.Module):
    def __init__(
        self,
        config: DFlashModelConfig,
        *,
        attention_op: DFlashAttentionOp,
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.block_size = config.block_size
        self.embed_tokens = nn.Embedding(
            config.verifier_vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.attention_op = attention_op
        self.layers = nn.ModuleList(
            self._build_decoder_layer(config, index, attention_op)
            for index in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config.head_dim, config.rope_theta)
        self.fc = nn.Linear(
            len(config.target_layer_ids) * config.target_hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.verifier_norm = RMSNorm(config.target_hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(
            config.hidden_size, config.draft_vocab_size, bias=False
        )
        self.verifier_lm_head = nn.Linear(
            config.target_hidden_size,
            config.draft_vocab_size,
            bias=False,
        )
        self._init_draft_head(config)
        self.register_buffer(
            "t2d",
            (
                None
                if config.draft_vocab_size == config.verifier_vocab_size
                else torch.zeros(config.verifier_vocab_size, dtype=torch.bool)
            ),
        )
        self.register_buffer(
            "d2t",
            (
                None
                if config.draft_vocab_size == config.verifier_vocab_size
                else torch.zeros(config.draft_vocab_size, dtype=torch.int64)
            ),
        )
        self._freeze_verifier_weights()
        self.activation_checkpointing = False
        self.runtime = None

    def _build_decoder_layer(
        self,
        config: DFlashModelConfig,
        layer_idx: int,
        attention_op: DFlashAttentionOp,
    ) -> nn.Module:
        return DFlashDecoderLayer(
            config,
            layer_idx=layer_idx,
            attention_op=attention_op,
        )

    def _init_draft_head(self, config: DFlashModelConfig) -> None:
        del config

    def load_vocabulary_mapping(self, *, t2d: torch.Tensor, d2t: torch.Tensor) -> None:
        if self.t2d is None or self.d2t is None:
            raise ValueError("full-vocabulary DFlash does not use a vocabulary mapping")
        t2d = t2d.to(dtype=torch.bool, device=self.t2d.device)
        d2t = d2t.to(dtype=torch.int64, device=self.d2t.device)
        if t2d.shape != self.t2d.shape or d2t.shape != self.d2t.shape:
            raise ValueError("DFlash vocabulary mapping shapes do not match the model")
        if int(t2d.sum()) != self.config.draft_vocab_size:
            raise ValueError("DFlash t2d cardinality does not match draft_vocab_size")
        selected = torch.nonzero(t2d, as_tuple=False).flatten()
        expected_offsets = selected - torch.arange(
            self.config.draft_vocab_size,
            device=selected.device,
            dtype=selected.dtype,
        )
        if not torch.equal(d2t.cpu(), expected_offsets.cpu()):
            raise ValueError("DFlash d2t does not map draft IDs to verifier IDs")
        self.t2d.copy_(t2d)
        self.d2t.copy_(d2t)

    def _freeze_verifier_weights(self) -> None:
        self.embed_tokens.weight.requires_grad_(False)
        self.lm_head.weight.requires_grad_(False)
        self.verifier_lm_head.weight.requires_grad_(False)
        self.verifier_norm.weight.requires_grad_(False)

    def forward(
        self,
        *,
        target_hidden_states: torch.Tensor,
        teacher_hidden_states: torch.Tensor | None,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        target_position_ids: torch.Tensor | None = None,
        objective: DFlashObjectiveBatch,
        runtime: Any | None = None,
        compute_teacher: bool = True,
    ) -> DFlashForwardOutput:
        batch_size, target_length, target_width = target_hidden_states.shape
        expected_width = (
            len(self.config.target_layer_ids) * self.config.target_hidden_size
        )
        if target_width != expected_width:
            raise ValueError(
                f"DFlash target features have width {target_width}, expected {expected_width}"
            )
        if int(input_ids.shape[0]) != batch_size:
            raise ValueError("DFlash input IDs must match target feature batch size")
        if position_ids.shape != input_ids.shape:
            raise ValueError("DFlash position IDs must match input IDs")
        runtime = runtime if runtime is not None else getattr(self, "runtime", None)
        if target_position_ids is None:
            if target_length != int(input_ids.shape[1]):
                raise ValueError(
                    "pre-sharded DFlash target features require target_position_ids"
                )
            target_intervals = _target_intervals(target_length, runtime=runtime)
            target_hidden_shard = cat_intervals(
                target_hidden_states.detach(),
                target_intervals,
            )
            target_position_ids = (
                cat_intervals(
                    position_ids.unsqueeze(-1),
                    target_intervals,
                )
                .squeeze(-1)
                .contiguous()
            )
        else:
            if target_position_ids.shape != target_hidden_states.shape[:2]:
                raise ValueError(
                    "DFlash target_position_ids must match target feature token dimensions"
                )
            target_hidden_shard = target_hidden_states.detach()
        compact_window = _shared_sliding_window(self.layers)
        target_intervals = None
        target_intervals_by_owner = None
        if compact_window is not None and _supports_compact_context(
            objective=objective,
            runtime=runtime,
        ):
            (
                target_hidden_shard,
                target_position_ids,
                target_intervals,
                target_intervals_by_owner,
            ) = compact_dflash_target_context(
                target_hidden_shard,
                target_position_ids,
                objective=objective,
                runtime=runtime,
                global_target_length=int(input_ids.shape[1]),
                sliding_window=int(compact_window),
            )
        with operator_scope("dflash.target_projection"):
            projected_target = self.hidden_norm(self.fc(target_hidden_shard))
        hidden_states = self.embed_tokens(objective.draft_input_ids)
        anchor_position_ids = position_ids.gather(1, objective.anchor_positions)
        draft_position_ids = (
            (
                anchor_position_ids.unsqueeze(-1)
                + torch.arange(
                    self.block_size,
                    device=position_ids.device,
                    dtype=position_ids.dtype,
                ).view(1, 1, -1)
            )
            .reshape(batch_size, -1)
            .contiguous()
        )
        draft_rotary = self.rotary_emb.prepare(
            draft_position_ids,
            dtype=hidden_states.dtype,
        )
        target_rotary = self.rotary_emb.prepare(
            target_position_ids,
            dtype=projected_target.dtype,
        )
        layer_kwargs = {
            "objective": objective,
            "target_rotary": target_rotary,
            "draft_rotary": draft_rotary,
            "runtime": runtime,
            "global_target_length": int(input_ids.shape[1]),
        }
        if target_intervals is not None:
            layer_kwargs["target_intervals"] = target_intervals
        if target_intervals_by_owner is not None:
            layer_kwargs["target_intervals_by_owner"] = target_intervals_by_owner
        interval_plan_caches: dict[int | None, DFlashIntervalPlanCache] = {}
        for layer in self.layers:
            plan_key = layer.self_attn.sliding_window
            interval_plan_cache = interval_plan_caches.setdefault(
                plan_key,
                DFlashIntervalPlanCache(),
            )
            current_layer_kwargs = {
                **layer_kwargs,
                "interval_plan_cache": interval_plan_cache,
            }
            if self.activation_checkpointing and self.training:
                hidden_states = checkpoint(
                    lambda draft, target, module=layer, kwargs=current_layer_kwargs: (
                        module(
                            draft,
                            target,
                            **kwargs,
                        )
                    ),
                    hidden_states,
                    projected_target,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    projected_target,
                    **current_layer_kwargs,
                )
        hidden_states = self.norm(hidden_states)
        output_teacher_hidden_states = None
        if compute_teacher:
            if teacher_hidden_states is None:
                raise ValueError("DFlash KL requires compact teacher hidden states")
            expected_teacher_shape = (
                batch_size,
                objective.teacher_source_positions.numel() // batch_size,
                self.config.target_hidden_size,
            )
            if teacher_hidden_states.shape != expected_teacher_shape:
                raise ValueError(
                    "DFlash compact teacher hidden-state shape does not match "
                    "the owned anchors"
                )
            output_teacher_hidden_states = teacher_hidden_states
        target_token_ids = input_ids.gather(
            1,
            objective.target_token_positions.reshape(batch_size, -1),
        )
        return DFlashForwardOutput(
            hidden_states=hidden_states,
            teacher_hidden_states=output_teacher_hidden_states,
            target_token_ids=target_token_ids,
            supervised_mask=objective.supervised_mask,
            position_weights=objective.position_weights,
        )

    def distributed_dflash_loss(
        self,
        output: DFlashForwardOutput,
        *,
        loss_kind: str,
        normalization_count: torch.Tensor,
        bp_loss_scale: float | None = None,
    ) -> torch.Tensor:
        valid = output.supervised_mask
        valid_indices = torch.nonzero(valid.reshape(-1), as_tuple=False).flatten()
        hidden = output.hidden_states.reshape(
            -1, output.hidden_states.shape[-1]
        ).index_select(
            0,
            valid_indices,
        )
        position_weights = (
            output.position_weights.reshape(-1)
            .index_select(
                0,
                valid_indices,
            )
            .to(torch.float32)
        )
        if valid_indices.numel() == 0:
            token_loss = zero_loss_with_module_parameters(
                output.hidden_states,
                self.lm_head,
            ).reshape(1)
            position_weights = torch.zeros_like(token_loss)
            sample_indices = torch.zeros(
                1,
                dtype=torch.int64,
                device=token_loss.device,
            )
        elif loss_kind == "speculators_kl":
            if output.teacher_hidden_states is None:
                raise RuntimeError("DFlash KL requires teacher logits")
            with torch.no_grad():
                teacher_hidden = output.teacher_hidden_states.reshape(
                    -1,
                    output.teacher_hidden_states.shape[-1],
                ).index_select(0, valid_indices)
                teacher_hidden = self.verifier_norm(teacher_hidden)
            token_loss = frozen_linear_kl(
                hidden,
                self.lm_head.weight,
                teacher_hidden,
                self.verifier_lm_head.weight,
                logit_softcap=self.config.final_logit_softcapping,
            )
        elif loss_kind == "paper_ce":
            token_loss = tiled_linear_cross_entropy(
                hidden,
                self.lm_head.weight,
                output.target_token_ids.reshape(-1).index_select(0, valid_indices),
                reduction="none",
                dtype=hidden.dtype,
                weight_layout="vocab_first",
                logit_softcap=self.config.final_logit_softcapping,
            )
        else:
            raise ValueError(f"unsupported DFlash loss {loss_kind!r}")
        batch_size = int(output.supervised_mask.shape[0])
        tokens_per_sample = int(output.supervised_mask.shape[1])
        if valid_indices.numel() != 0:
            sample_indices = torch.div(
                valid_indices,
                tokens_per_sample,
                rounding_mode="floor",
            )
        numerators = torch.zeros(
            batch_size,
            dtype=torch.float32,
            device=token_loss.device,
        )
        numerators.scatter_add_(
            0,
            sample_indices,
            token_loss.float() * position_weights,
        )
        denominators = normalization_count.to(
            device=token_loss.device,
            dtype=torch.float32,
        )
        if denominators.shape != numerators.shape:
            raise ValueError(
                "DFlash normalization count must contain one value per sample"
            )
        loss = (numerators / (denominators + 1.0e-5)).mean()
        return loss * float(
            bp_loss_scale
            if bp_loss_scale is not None
            else (runtime_loss_scale(self.runtime) if self.runtime is not None else 1.0)
        )


def _rotate_half(states: torch.Tensor) -> torch.Tensor:
    first, second = states.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rotary(states: torch.Tensor, factors: RotaryFactors) -> torch.Tensor:
    return states * factors.cos + _rotate_half(states) * factors.sin


def _shared_sliding_window(layers: nn.ModuleList) -> int | None:
    """Return a safe context window when every draft layer is windowed."""

    windows = [getattr(layer.self_attn, "sliding_window", None) for layer in layers]
    if not windows or any(window is None for window in windows):
        return None
    return max(int(window) for window in windows)


def _supports_compact_context(
    *,
    objective: DFlashObjectiveBatch,
    runtime: Any | None,
) -> bool:
    """Whether ranks can construct consistent compact context ownership.

    Matched fused groups require equally sized anchor packets so their small
    mask metadata can be gathered before every owner builds the same packed
    K/V interval layout. CP-only ranks already carry the complete query set.
    """

    if runtime is None or not bool(
        getattr(runtime, "uses_context_parallel_attention", False)
    ):
        return True
    if getattr(runtime, "active_block_mode", None) != "dual_end":
        return True
    block_parallel_size = int(getattr(runtime, "block_parallel_size", 1) or 1)
    global_anchor_count = int(objective.global_anchor_count)
    if block_parallel_size <= 0 or global_anchor_count % block_parallel_size:
        return False
    return int(objective.anchor_valid.shape[1]) == (
        global_anchor_count // block_parallel_size
    )


def _query_ring_context_metadata(
    objective: DFlashObjectiveBatch,
    *,
    runtime: Any | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect the small anchor mask needed to prune each local K/V shard."""

    starts = objective.document_starts
    stops = objective.context_stops
    valid = objective.anchor_valid
    if (
        runtime is None
        or getattr(runtime, "active_block_mode", None) != "dual_end"
        or not bool(getattr(runtime, "uses_context_parallel_attention", False))
        or len(getattr(runtime, "context_block_parallel_group_ranks", ())) <= 1
        or not dist.is_available()
        or not dist.is_initialized()
    ):
        return starts, stops, valid

    ring_size = len(runtime.context_block_parallel_group_ranks)
    block_parallel_size = int(getattr(runtime, "block_parallel_size", 1) or 1)
    padded_anchors = max(
        int(valid.shape[1]),
        math.ceil(int(objective.global_anchor_count) / block_parallel_size),
    )
    packed = torch.zeros(
        (int(valid.shape[0]), padded_anchors, 3),
        dtype=torch.int64,
        device=valid.device,
    )
    local_anchors = int(valid.shape[1])
    packed[:, :local_anchors, 0].copy_(starts.to(torch.int64))
    packed[:, :local_anchors, 1].copy_(stops.to(torch.int64))
    packed[:, :local_anchors, 2].copy_(valid.to(torch.int64))
    gathered = [torch.empty_like(packed) for _ in range(ring_size)]
    dist.all_gather(
        gathered,
        packed,
        group=runtime.context_block_parallel_group,
    )
    combined = torch.cat(gathered, dim=1)
    return (
        combined[..., 0],
        combined[..., 1],
        combined[..., 2].to(torch.bool),
    )


def _merged_window_intervals(
    starts: torch.Tensor,
    stops: torch.Tensor,
    valid: torch.Tensor,
    *,
    sliding_window: int,
) -> tuple[tuple[int, int], ...]:
    """Build the union of exact token-index windows for a query packet set."""

    if sliding_window <= 0:
        raise ValueError("DFlash compact sliding window must be positive")
    lower = torch.maximum(
        starts.to(torch.int64),
        stops.to(torch.int64) - (int(sliding_window) - 1),
    )
    packed = torch.stack((lower, stops.to(torch.int64)), dim=-1)
    selected = packed[valid.to(torch.bool)].detach().cpu().tolist()
    intervals = sorted(
        (int(start), int(stop)) for start, stop in selected if int(stop) > int(start)
    )
    if not intervals:
        return ()
    merged: list[list[int]] = []
    for start, stop in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, stop])
        else:
            merged[-1][1] = max(merged[-1][1], stop)
    return tuple((start, stop) for start, stop in merged)


def _intersect_intervals(
    left: tuple[tuple[int, int], ...],
    right: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    intersections: list[tuple[int, int]] = []
    for left_start, left_stop in left:
        for right_start, right_stop in right:
            start = max(int(left_start), int(right_start))
            stop = min(int(left_stop), int(right_stop))
            if stop > start:
                intersections.append((start, stop))
    return tuple(intersections)


def compact_dflash_target_context(
    target_hidden_states: torch.Tensor,
    target_position_ids: torch.Tensor,
    *,
    objective: DFlashObjectiveBatch,
    runtime: Any | None,
    global_target_length: int,
    sliding_window: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    tuple[tuple[int, int], ...] | None,
    tuple[tuple[tuple[int, int], ...], ...] | None,
]:
    """Prune context rows that cannot be reached by any sliding query.

    Target projection, normalization, rotary embedding, and K/V projection are
    tokenwise.  Rows outside the union of the active query windows therefore
    have exactly zero influence and zero parameter gradient and may be omitted.
    The returned global intervals retain the original token indices for the
    sparse FA4 mask.
    """

    if target_hidden_states.ndim != 3 or target_position_ids.ndim != 2:
        raise ValueError("DFlash compact context expects [B,S,D] and [B,S]")
    if target_hidden_states.shape[:2] != target_position_ids.shape:
        raise ValueError("DFlash compact context tensor dimensions disagree")
    source_intervals = _target_intervals(
        int(global_target_length),
        runtime=runtime,
    )
    source_intervals_by_owner = (
        clean_intervals_for_runtime(
            int(global_target_length),
            int(runtime.context_attention_size),
            runtime,
        )
        if runtime is not None
        and bool(getattr(runtime, "uses_context_parallel_attention", False))
        else (source_intervals,)
    )
    source_length = sum(stop - start for start, stop in source_intervals)
    if int(target_hidden_states.shape[1]) != int(source_length):
        raise ValueError("DFlash target feature shard does not match CP ownership")
    starts, stops, valid = _query_ring_context_metadata(
        objective,
        runtime=runtime,
    )
    windows = _merged_window_intervals(
        starts,
        stops,
        valid,
        sliding_window=int(sliding_window),
    )
    compact_intervals_by_owner = tuple(
        _intersect_intervals(owner_intervals, windows)
        for owner_intervals in source_intervals_by_owner
    )
    # Empty-key FA4 launches and zero-length ring packets are not portable. A
    # locality group will commonly miss an entire CP owner, so falling back to
    # that owner's full shard defeats compaction. Retain one real but masked
    # source tile instead. A single row gives the wide target projection a
    # pathological M=1 weight-gradient GEMM; one 256-row tensor-core tile is
    # both faster and still tiny. Because the owner's useful intersection is
    # empty, every retained position is outside all query intervals and has
    # exactly zero influence.
    compact_intervals_by_owner = tuple(
        intervals
        if intervals
        else (
            (
                int(owner_intervals[0][0]),
                min(int(owner_intervals[0][1]), int(owner_intervals[0][0]) + 256),
            ),
        )
        for owner_intervals, intervals in zip(
            source_intervals_by_owner,
            compact_intervals_by_owner,
            strict=True,
        )
    )
    local_rank = (
        int(runtime.context_parallel_rank)
        if runtime is not None
        and bool(getattr(runtime, "uses_context_parallel_attention", False))
        else 0
    )
    compact_intervals = compact_intervals_by_owner[local_rank]
    if compact_intervals == source_intervals:
        return (
            target_hidden_states,
            target_position_ids,
            source_intervals,
            compact_intervals_by_owner,
        )

    hidden_chunks: list[torch.Tensor] = []
    position_chunks: list[torch.Tensor] = []
    source_offset = 0
    for source_start, source_stop in source_intervals:
        for compact_start, compact_stop in compact_intervals:
            start = max(int(source_start), int(compact_start))
            stop = min(int(source_stop), int(compact_stop))
            if stop <= start:
                continue
            local_start = source_offset + start - int(source_start)
            local_stop = source_offset + stop - int(source_start)
            hidden_chunks.append(target_hidden_states[:, local_start:local_stop])
            position_chunks.append(target_position_ids[:, local_start:local_stop])
        source_offset += int(source_stop) - int(source_start)
    return (
        torch.cat(hidden_chunks, dim=1).contiguous(),
        torch.cat(position_chunks, dim=1).contiguous(),
        compact_intervals,
        compact_intervals_by_owner,
    )


def _target_intervals(
    sequence_length: int,
    *,
    runtime: Any | None,
) -> tuple[tuple[int, int], ...]:
    if runtime is None or not bool(
        getattr(runtime, "uses_context_parallel_attention", False)
    ):
        return ((0, int(sequence_length)),)
    intervals = clean_intervals_for_runtime(
        int(sequence_length),
        int(runtime.context_attention_size),
        runtime,
    )
    return intervals[int(runtime.context_parallel_rank)]


def _sliding_window_non_causal(
    values: dict[str, Any],
    method: dict[str, Any],
) -> bool:
    """Resolve every supported train/serve causality spelling fail-closed."""

    aliases: list[tuple[str, bool]] = []
    if values.get("is_causal") is not None:
        aliases.append(("is_causal", not bool(values["is_causal"])))
    if values.get("sliding_window_non_causal") is not None:
        aliases.append(
            (
                "sliding_window_non_causal",
                bool(values["sliding_window_non_causal"]),
            )
        )
    if method.get("causal") is not None:
        aliases.append(("dflash_config.causal", not bool(method["causal"])))
    if aliases and len({non_causal for _, non_causal in aliases}) != 1:
        rendered = ", ".join(f"{name}={value!r}" for name, value in aliases)
        raise ValueError(f"DFlash causality fields disagree: {rendered}")
    return aliases[0][1] if aliases else False


def _required_int(values: dict[str, Any], name: str) -> int:
    value = values.get(name)
    if value is None:
        raise ValueError(f"DFlash config requires {name}")
    return int(value)


def _optional_int(values: dict[str, Any], name: str) -> int | None:
    value = values.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"DFlash {name} must be an integer")
    return int(value)


__all__ = [
    "compact_dflash_target_context",
    "DFlashAttentionOp",
    "DFlashForwardOutput",
    "DFlashModel",
    "DFlashModelConfig",
]
