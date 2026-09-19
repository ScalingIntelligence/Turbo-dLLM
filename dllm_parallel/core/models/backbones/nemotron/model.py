# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Nemotron-Labs-Diffusion backbone metadata and packed BP/CP execution."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
import gc
import inspect
import json
from typing import Any, Callable

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from dllm_parallel.core.attention import (
    BlockDenoisingFullMask,
    BlockDenoisingGlobalCleanMask,
    BlockDenoisingLocalActiveMask,
    replicated_block_denoising_attention_bshd,
    fused_block_context_attention_bshd,
    pure_context_persistent_block_denoising_attention_bshd,
)
from dllm_parallel.core.attention.cp_backend import clean_shards_for_rank
from dllm_parallel.core.attention.layout import (
    active_query_indices_for_context_rank,
    clear_sequence_layout_caches,
    runtime_clean_layout,
)
from dllm_parallel.core.parallel.runtime import (
    active_clean_prefix_length,
    active_token_indices,
    begin_clean_replica_tensor,
    loss_scale as runtime_loss_scale,
)
from dllm_parallel.core.parallel.tensor_parallel import (
    VocabParallelEmbedding,
    VocabParallelLinear,
    copy_to_tensor_parallel_region,
    gather_active_from_sequence_parallel_region,
    gather_from_sequence_parallel_region,
    partition_bounds,
    reduce_from_tensor_parallel_region,
    scatter_to_sequence_parallel_region,
)
from dllm_parallel.core.kernels.tiled_linear_cross_entropy import (
    tiled_linear_cross_entropy,
)
from dllm_parallel.core.kernels.streaming_soft_embedding import (
    streaming_soft_embedding,
)
from dllm_parallel.core.models.contracts import (
    BackboneCapabilities,
    BackboneKernelPolicy,
)
from dllm_parallel.core.models.compatibility import validate_objective_for_family
from dllm_parallel.core.models.loss import zero_loss_with_module_parameters
from dllm_parallel.core.models.backbones.config_utils import (
    optional_int,
)
from dllm_parallel.core.models.backbones.moe_checkpoint import (
    moe_route_checkpoint_context_fn,
)
from dllm_parallel.core.models.backbones.nemotron.metadata import (
    build_schedule,
    summarize_config,
    validate_parallel_spec,
)
from dllm_parallel.core.profiling.operator_trace import traced_operator


FAMILY = "nemotron_labs_diffusion"


def _attention_trace_name(
    _model: Any,
    layer_ops: "NemotronDecoderLayerOps",
    *_args: Any,
    **_kwargs: Any,
) -> str:
    kind = "sliding" if layer_ops.layer_type == "sliding_attention" else "full"
    return f"attention.{kind}"


def _feed_forward_trace_name(
    _model: Any,
    layer_ops: "NemotronDecoderLayerOps",
    *_args: Any,
    **_kwargs: Any,
) -> str:
    has_experts = any(
        item is not None
        for item in (layer_ops.router, layer_ops.moe, layer_ops.experts)
    )
    return f"feed_forward.{'moe' if has_experts else 'dense'}"


@dataclass(frozen=True)
class NemotronAttentionOps:
    q_proj: Any
    k_proj: Any
    v_proj: Any
    o_proj: Any
    q_norm: Any | None
    k_norm: Any | None
    v_norm: Any | None
    head_dim: int
    apply_rotary: Callable[..., Any]
    rope_scale: Callable[[Any], Any | None]
    scale: float
    output_gate_proj: Any | None = None
    query_scale_multiplier: float = 1.0
    uses_rope: bool = True
    sliding_window: int | None = None
    value_from_key: bool = False


@dataclass(frozen=True)
class NemotronDecoderLayerOps:
    layer: Any
    input_layernorm: Any
    self_attn: Any
    attention: NemotronAttentionOps
    post_attention_layernorm: Any
    mlp: Any
    layer_style: str = "standard"
    pre_feedforward_layernorm: Any | None = None
    post_feedforward_layernorm: Any | None = None
    router: Any | None = None
    moe: Any | None = None
    experts: Any | None = None
    pre_feedforward_layernorm_2: Any | None = None
    post_feedforward_layernorm_1: Any | None = None
    post_feedforward_layernorm_2: Any | None = None
    layer_scalar: Any | None = None
    layer_type: str | None = None


@dataclass(frozen=True)
class NemotronBlockDiffusionComponents:
    encoder: Any
    embed_tokens: Any
    layers: tuple[NemotronDecoderLayerOps, ...]
    norm: Any
    output_head: Any
    rotary_emb: Callable[..., Any]
    rotary_emb_accepts_layer_type: bool
    num_hidden_layers: int
    embedding_norm: Any | None = None
    embedding_scale: Any | None = None
    soft_embedding_scale: float = 1.0
    final_logit_softcap: float | None = None
    output_multiplier: float | None = None
    self_conditioning: Any | None = None


def _layer_requires_moe_checkpoint_context(
    layer_ops: NemotronDecoderLayerOps,
) -> bool:
    return (
        layer_ops.layer_style == "gemma4"
        or layer_ops.router is not None
        or layer_ops.moe is not None
        or layer_ops.experts is not None
    )


class _TEPackedLayerProjections(nn.Module):
    def __init__(
        self,
        *,
        qkv: nn.Module,
        qkv_local_sizes: tuple[int, ...],
        o_proj: nn.Module,
        gate_up: nn.Module,
        gate_up_local_sizes: tuple[int, int],
        gated_activation: nn.Module | None,
        down_proj: nn.Module,
        activation: Callable[[torch.Tensor], torch.Tensor],
        qkv_value_from_key: bool = False,
        qkv_fuses_input_norm: bool = False,
        gate_up_fuses_pre_ff_norm: bool = False,
    ) -> None:
        super().__init__()
        self.qkv = qkv
        self.qkv_local_sizes = qkv_local_sizes
        self.o_proj = o_proj
        self.gate_up = gate_up
        self.gate_up_local_sizes = gate_up_local_sizes
        self.gated_activation = gated_activation
        self.down_proj = down_proj
        self.activation = activation
        self.qkv_value_from_key = bool(qkv_value_from_key)
        self.qkv_fuses_input_norm = bool(qkv_fuses_input_norm)
        self.gate_up_fuses_pre_ff_norm = bool(gate_up_fuses_pre_ff_norm)


class _ReleasedHFProjection(nn.Module):
    """Sentinel for HF projection modules replaced by packed TP modules."""

    def __init__(self, label: str) -> None:
        super().__init__()
        self.label = str(label)

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        del args, kwargs
        raise RuntimeError(
            f"{self.label} was released after packing into optimized TP modules"
        )


@dataclass(frozen=True)
class _SequenceParallelPackedMeta:
    batch_size: int
    packed_len: int
    hidden_size: int
    total_rows: int
    padded_rows: int
    active_len: int


@dataclass(frozen=True)
class _PackedLayout:
    active_positions: torch.Tensor
    clean_positions: torch.Tensor
    packed_positions: torch.Tensor
    local_attn_mask: BlockDenoisingLocalActiveMask
    global_attn_mask: BlockDenoisingGlobalCleanMask
    active_len: int


@dataclass(frozen=True)
class _DiffusionGemmaCleanStream:
    """Autograd-connected clean states and the per-layer K/V they produced."""

    hidden: torch.Tensor
    positions: torch.Tensor
    layer_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...]

    def detached(self) -> "_DiffusionGemmaCleanStream":
        return _DiffusionGemmaCleanStream(
            hidden=self.hidden.detach(),
            positions=self.positions,
            layer_key_values=tuple(
                (key.detach(), value.detach()) for key, value in self.layer_key_values
            ),
        )

    def index_select_batch(
        self,
        indices: torch.Tensor,
    ) -> "_DiffusionGemmaCleanStream":
        return _DiffusionGemmaCleanStream(
            hidden=self.hidden.index_select(0, indices),
            positions=self.positions,
            layer_key_values=tuple(
                (
                    key.index_select(0, indices),
                    value.index_select(0, indices),
                )
                for key, value in self.layer_key_values
            ),
        )


@dataclass(frozen=True)
class DiffusionGemmaNativeOutput:
    """Loss-bearing output of native all-block DiffusionGemma SFT."""

    loss: torch.Tensor
    decoder_loss: torch.Tensor
    encoder_loss: torch.Tensor
    decoder_hidden: torch.Tensor
    clean_hidden: torch.Tensor
    active_positions: torch.Tensor
    clean_positions: torch.Tensor


def packed_fsdp_modules(model: nn.Module) -> tuple[nn.Module, ...]:
    """Keep packed HF execution in one hook-correct FSDP2 root unit.

    Packed HF layers execute extracted projection modules directly instead of
    calling the original decoder layer. Returning those decoder layers as
    nested FSDP units would bypass their pre-forward parameter all-gathers.
    """

    del model
    return ()


@dataclass(frozen=True)
class _Gemma4PendingMoeDispatch:
    pending: Any
    total_rows: int
    hidden_size: int
    dtype: torch.dtype
    device: torch.device
    input_requires_grad: bool
    nonclean_indices: torch.Tensor
    clean_indices: torch.Tensor
    clean_owner: bool


@dataclass(frozen=True)
class _Gemma4ExpertTokenPlan:
    nonclean_indices: torch.Tensor
    clean_indices: torch.Tensor
    owner_route_indices: torch.Tensor


__all__ = [
    "DiffusionGemmaNativeOutput",
    "NemotronLabsDiffusionBackboneExecutor",
    "NemotronLabsDiffusionPackedBlockDiffusionModel",
    "build_executor",
    "build_packed_block_diffusion_model",
    "build_schedule",
    "summarize_config",
    "validate_parallel_spec",
]


class NemotronLabsDiffusionBackboneExecutor:
    family = FAMILY

    def metadata(self, config: Any, *, model_id: str) -> Any:
        return summarize_config(model_id, _config_to_dict(config))

    def capabilities(self) -> BackboneCapabilities:
        return BackboneCapabilities(
            family=FAMILY,
            packed_block_diffusion=True,
            tensor_parallel=True,
            sequence_parallel=True,
            checkpoint_hooks=True,
            sharded_state_dict=True,
            tokenizer_required_for_text=True,
        )

    def prepare_tokenizer_and_config(
        self,
        spec: Any,
        *,
        config: Any,
        tokenizer: Any | None,
    ) -> None:
        """Nemotron's pinned checkpoint already defines its token IDs."""

        del spec, config, tokenizer

    def validate_run_spec(self, spec: Any, *, config: Any | None = None) -> None:
        del config
        validate_objective_for_family(
            family=self.family,
            objective=str(getattr(getattr(spec, "objective", None), "name", "")),
        )
        if spec.model.family not in {"auto", "hf", FAMILY}:
            raise ValueError(
                "Nemotron-Labs-Diffusion executor requires "
                "model.family=auto, hf, or nemotron_labs_diffusion"
            )
        if spec.kernel.cp_bp_attention_policy != "production":
            raise ValueError("Nemotron CP/BP requires the production attention policy")
        if (
            spec.topology.sequence_parallel
            and int(spec.topology.tensor_parallel_size) <= 1
        ):
            raise ValueError("sequence_parallel requires tensor_parallel_size > 1")
        if int(getattr(spec.topology, "expert_parallel_size", 1) or 1) != 1:
            raise ValueError(
                "Nemotron-Labs-Diffusion is dense and does not support expert_parallel_size > 1"
            )
        if spec.adapter.type == "lora" and "experts" in spec.adapter.targets:
            raise ValueError("Nemotron is dense; remove 'experts' from adapter.targets")

    def build_model(
        self,
        *,
        model_id: str,
        revision: str | None,
        config: Any,
        runtime: Any | None,
        dtype: Any,
        device: Any,
        trust_remote_code: bool,
    ) -> Any:
        from dllm_parallel.core.models.hf_loader import load_hf_model_from_config

        return load_hf_model_from_config(
            model_id,
            config=config,
            trust_remote_code=trust_remote_code,
            revision=revision,
            model_auto_class="model",
            parallel_runtime=runtime,
            torch_dtype=dtype,
            device=device,
            model_kwargs={"low_cpu_mem_usage": True},
        )

    def build_packed_block_diffusion_model(
        self,
        model: Any,
        *,
        runtime: Any,
        seq_len: int,
        block_size: int,
        ring_attention_key_chunk_size: int = 0,
        activation_checkpointing: bool = True,
        activation_checkpointing_scope: str = "full",
        mlp_token_chunk_size: int = 0,
        self_condition_clean_tokens: bool = True,
        encoder_causal_attention: bool = False,
    ) -> Any:
        return build_packed_block_diffusion_model(
            model,
            runtime=runtime,
            seq_len=seq_len,
            block_size=block_size,
            ring_attention_key_chunk_size=ring_attention_key_chunk_size,
            activation_checkpointing=activation_checkpointing,
            activation_checkpointing_scope=activation_checkpointing_scope,
            mlp_token_chunk_size=mlp_token_chunk_size,
            self_condition_clean_tokens=self_condition_clean_tokens,
            encoder_causal_attention=encoder_causal_attention,
        )

    def build_training_model(self, model: Any, *, runtime: Any, spec: Any) -> Any:
        block_size = spec.objective.block_size
        if block_size is None:
            raise ValueError("Nemotron training requires objective.block_size")
        return self.build_packed_block_diffusion_model(
            model,
            runtime=runtime,
            seq_len=int(spec.model.seq_len),
            block_size=int(block_size),
            ring_attention_key_chunk_size=int(
                spec.kernel.ring_attention_key_chunk_size
            ),
            activation_checkpointing=bool(spec.training.activation_checkpointing),
            activation_checkpointing_scope=str(
                spec.training.activation_checkpointing_scope
            ),
            mlp_token_chunk_size=int(spec.kernel.mlp_token_chunk_size),
        )

    def fsdp_modules(self, model: Any) -> tuple[Any, ...]:
        return packed_fsdp_modules(model)

    def build_training_task(self, **kwargs: Any) -> Any:
        from dllm_parallel.core.objectives.training import build_standard_training_task

        return build_standard_training_task(**kwargs)

    def build_data_runtime(self, **kwargs: Any) -> Any:
        from dllm_parallel.core.data import build_standard_data_runtime

        return build_standard_data_runtime(**kwargs)

    def verify_native_kernels(self, spec: Any) -> dict[str, Any]:
        del spec
        from dllm_parallel.core.attention.flex import verify_flex_attention_runtime

        return verify_flex_attention_runtime().to_log_dict()

    def parallel_work_units(self, spec: Any) -> int:
        block_size = spec.objective.block_size
        if block_size is None:
            raise ValueError("Nemotron training requires objective.block_size")
        return int(spec.model.seq_len) // int(block_size)

    def tokenizer_model_id(self, spec: Any) -> str:
        return str(spec.model.id)

    def build_objective_schedule(
        self,
        spec: Any,
        *,
        sequence_length: int | None = None,
    ) -> Any:
        return build_schedule(spec, sequence_length=sequence_length)

    def kernel_policy(self, spec: Any) -> BackboneKernelPolicy:
        del spec
        return BackboneKernelPolicy()

    def validate_tokenizer_data_compatibility(
        self, spec: Any, tokenizer: Any | None
    ) -> None:
        if spec.data.input_mode == "text" and tokenizer is None:
            raise ValueError("text input mode requires a Hugging Face tokenizer")

    def load_checkpoint_hooks(self, checkpoint: Any, model: Any) -> None:
        del model
        state = (checkpoint or {}).get("backbone_state") or {}
        family = state.get("family")
        if family is not None and str(family) != FAMILY:
            raise RuntimeError(
                f"checkpoint backbone family {family!r} cannot be loaded by {FAMILY!r}"
            )

    def save_checkpoint_hooks(self, model: Any) -> dict[str, Any]:
        return {
            "family": FAMILY,
            "model_class": model.__class__.__name__,
            "checkpoint_hooks_version": 1,
        }

    def sharded_state_dict(self, model: Any) -> dict[str, Any]:
        state_dict = getattr(model, "state_dict", None)
        if not callable(state_dict):
            raise TypeError("Nemotron model does not expose state_dict")
        return {
            "format": "dllm_parallel.backbone_state_dict.v1",
            "family": FAMILY,
            "state_dict": state_dict(),
        }

    def migrate_config(self, raw_config: Any) -> Any:
        return raw_config


def build_executor() -> NemotronLabsDiffusionBackboneExecutor:
    return NemotronLabsDiffusionBackboneExecutor()


class NemotronLabsDiffusionPackedBlockDiffusionModel(nn.Module):
    """Nemotron packed active-query + clean-shard BP/CP executor.

    The wrapped Nemotron HF model owns embeddings, decoder layers, final
    norm, and output head. This class owns Nemotron-specific layer access
    and calls the shared optimized block-denoising CP/BP attention backend.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        runtime: Any,
        seq_len: int,
        block_size: int,
        ring_attention_key_chunk_size: int = 0,
        activation_checkpointing: bool = True,
        activation_checkpointing_scope: str = "full",
        mlp_token_chunk_size: int = 0,
        self_condition_clean_tokens: bool = True,
        encoder_causal_attention: bool = False,
    ) -> None:
        super().__init__()
        components = _nemotronlabsdiffusion_components(model)
        self.model = model
        self.encoder = components.encoder
        tied_vocab_weights = (
            getattr(components.output_head, "weight", None) is not None
            and components.output_head.weight is components.embed_tokens.weight
        )
        vocab_size = optional_int(
            getattr(getattr(model, "config", None), "vocab_size", None)
        )
        self.embed_tokens = _maybe_replace_vocab_parallel_embedding(
            components.encoder,
            components.embed_tokens,
            runtime,
            vocab_size=vocab_size,
        )
        self.layers = components.layers
        self._te_packed_layers = nn.ModuleList()
        self._te_packed_by_layer_id: dict[int, _TEPackedLayerProjections] = {}
        tensor_parallel_size = int(getattr(runtime, "tensor_parallel_size", 1) or 1)
        first_layer = self.layers[0] if self.layers else None
        use_te_packed_projections = tensor_parallel_size > 1 or (
            first_layer is not None
            and _first_linear_device(
                first_layer.attention.q_proj,
                first_layer.attention.o_proj,
                first_layer.mlp,
            ).type
            == "cuda"
        )
        if use_te_packed_projections:
            for layer_ops in self.layers:
                packed_layer = self._build_te_packed_layer_projections(
                    layer_ops,
                    runtime,
                )
                self._te_packed_layers.append(packed_layer)
                self._te_packed_by_layer_id[id(layer_ops.layer)] = packed_layer
            self.layers = _release_replaced_hf_tp_modules(
                self.layers,
                self._te_packed_by_layer_id,
            )
            _tag_sequence_parallel_replicated_components(
                self.layers,
                components.norm,
                components.self_conditioning,
                runtime,
            )
        self.norm = components.norm
        self.output_head = _maybe_replace_vocab_parallel_output_head(
            model,
            components.output_head,
            runtime,
            vocab_size=vocab_size,
            shared_weight=self.embed_tokens.weight if tied_vocab_weights else None,
        )
        self.rotary_emb = components.rotary_emb
        self.rotary_emb_accepts_layer_type = components.rotary_emb_accepts_layer_type
        self.embedding_scale = components.embedding_scale
        self.soft_embedding_scale = float(components.soft_embedding_scale)
        self.embedding_norm = components.embedding_norm
        self.final_logit_softcap = components.final_logit_softcap
        self.output_multiplier = components.output_multiplier
        self.self_conditioning = components.self_conditioning
        self.self_condition_clean_tokens = bool(self_condition_clean_tokens)
        self.encoder_causal_attention = bool(encoder_causal_attention)
        if self.encoder_causal_attention:
            for index, layer_ops in enumerate(self.layers):
                if layer_ops.layer_type not in {
                    "sliding_attention",
                    "full_attention",
                }:
                    raise ValueError(
                        f"layer {index} does not expose a supported attention type"
                    )
                if (
                    layer_ops.layer_type == "sliding_attention"
                    and layer_ops.attention.sliding_window is None
                ):
                    raise ValueError(
                        f"sliding-attention layer {index} does not expose its window"
                    )
        self.num_hidden_layers = components.num_hidden_layers
        self.runtime = runtime
        self.max_seq_len = int(seq_len)
        self.seq_len = self.max_seq_len
        self.block_size = int(block_size)
        self.ring_attention_key_chunk_size = int(ring_attention_key_chunk_size or 0)
        self.activation_checkpointing = bool(activation_checkpointing)
        if activation_checkpointing_scope not in {"full", "mlp"}:
            raise ValueError("activation_checkpointing_scope must be 'full' or 'mlp'")
        self.activation_checkpointing_scope = str(activation_checkpointing_scope)
        self._adapter_checkpointing = False
        self.mlp_token_chunk_size = max(0, int(mlp_token_chunk_size or 0))
        self.sequence_parallel = bool(getattr(runtime, "sequence_parallel", False))
        self._clean_expert_token_plan_cache: dict[
            tuple[object, ...],
            _Gemma4ExpertTokenPlan,
        ] = {}
        self._packed_layout_cache: dict[tuple[object, ...], _PackedLayout] = {}
        self._layer_attention_mask_cache: dict[tuple[object, ...], Any] = {}
        self._requires_moe_checkpoint_context = any(
            _layer_requires_moe_checkpoint_context(layer_ops)
            for layer_ops in self.layers
        )
        if self.sequence_parallel:
            if int(getattr(runtime, "tensor_parallel_size", 1) or 1) <= 1:
                raise ValueError("sequence_parallel requires tensor_parallel_size > 1")
            if not self._te_packed_layers:
                raise RuntimeError(
                    "Nemotron sequence_parallel requires TE packed TP projections"
                )
        if self.seq_len <= 0 or self.block_size <= 0:
            raise ValueError("seq_len and block_size must be positive")
        if self.seq_len % self.block_size != 0:
            raise ValueError("seq_len must divide evenly by block_size")

    def _build_te_packed_layer_projections(
        self,
        layer_ops: NemotronDecoderLayerOps,
        runtime: Any,
    ) -> _TEPackedLayerProjections:
        """Build one layer's optimized projections.

        Backbones with non-uniform Q/KV sharding may override this construction
        hook without adding a branch to the per-layer execution path.
        """

        return _build_te_packed_layer_projections(layer_ops, runtime)

    def enable_adapter_checkpointing(self) -> None:
        """Retain parameter graphs when checkpoint inputs originate from a frozen base."""

        self._adapter_checkpointing = True

    def _activate_sequence_length(self, sequence_length: int) -> None:
        sequence_length = int(sequence_length)
        if not 0 < sequence_length <= self.max_seq_len:
            raise ValueError(
                "input sequence length must be positive and no greater than the "
                f"configured maximum ({self.max_seq_len})"
            )
        if sequence_length % self.block_size:
            raise ValueError("input sequence length must divide evenly by block_size")
        if sequence_length == self.seq_len:
            return
        self.seq_len = sequence_length
        self._packed_layout_cache.clear()
        self._layer_attention_mask_cache.clear()
        self._clean_expert_token_plan_cache.clear()
        context_layout_cache = getattr(self, "_context_layout_cache", None)
        if context_layout_cache is not None:
            context_layout_cache.clear()
        clear_sequence_layout_caches()

    def _checkpoint_uses_reentrant_autograd(self) -> bool:
        return not self._adapter_checkpointing

    def forward(
        self,
        *,
        noisy_input_ids: torch.Tensor,
        clean_input_ids: torch.Tensor,
        diffusion_times: torch.Tensor | None = None,
        noise_levels: torch.Tensor | None = None,
        objective_mode: str = "standard_block_diffusion",
        labels: torch.Tensor | None = None,
        scored_mask: torch.Tensor | None = None,
        encoder_valid_mask: torch.Tensor | None = None,
        decoder_position_ids: torch.Tensor | None = None,
        decoder_valid_mask: torch.Tensor | None = None,
        decoder_block_ids: torch.Tensor | None = None,
        response_start: int = 0,
        block_token_counts: torch.Tensor | None = None,
        valid_block_mask: torch.Tensor | None = None,
        self_conditioning_mask: torch.Tensor | None = None,
        self_conditioning_execution_count: int | None = None,
        self_conditioning_execute_all: bool = False,
        encoder_loss_weight: float = 1.0,
        decoder_loss_denominator: float | torch.Tensor | None = None,
        encoder_loss_denominator: float | torch.Tensor | None = None,
        self_conditioning_row_chunk_size: int = 256,
        self_conditioning_vocab_chunk_size: int = 32768,
    ) -> tuple[torch.Tensor, torch.Tensor] | DiffusionGemmaNativeOutput:
        if objective_mode == "diffusiongemma_native_sft":
            required = {
                "labels": labels,
                "scored_mask": scored_mask,
                "decoder_position_ids": decoder_position_ids,
                "decoder_block_ids": decoder_block_ids,
                "block_token_counts": block_token_counts,
                "valid_block_mask": valid_block_mask,
                "self_conditioning_mask": self_conditioning_mask,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError(
                    "native DiffusionGemma forward is missing: " + ", ".join(missing)
                )
            return self.forward_diffusiongemma_native(
                noisy_input_ids=noisy_input_ids,
                clean_input_ids=clean_input_ids,
                labels=labels,
                scored_mask=scored_mask,
                encoder_valid_mask=encoder_valid_mask,
                decoder_position_ids=decoder_position_ids,
                decoder_valid_mask=decoder_valid_mask,
                decoder_block_ids=decoder_block_ids,
                response_start=int(response_start),
                block_token_counts=block_token_counts,
                valid_block_mask=valid_block_mask,
                self_conditioning_mask=self_conditioning_mask,
                self_conditioning_execution_count=(self_conditioning_execution_count),
                self_conditioning_execute_all=bool(self_conditioning_execute_all),
                encoder_loss_weight=float(encoder_loss_weight),
                decoder_loss_denominator=decoder_loss_denominator,
                encoder_loss_denominator=encoder_loss_denominator,
                self_conditioning_row_chunk_size=int(self_conditioning_row_chunk_size),
                self_conditioning_vocab_chunk_size=int(
                    self_conditioning_vocab_chunk_size
                ),
            )
        if objective_mode != "standard_block_diffusion":
            raise ValueError(f"unsupported objective_mode {objective_mode!r}")
        del diffusion_times, noise_levels
        if noisy_input_ids.shape != clean_input_ids.shape:
            raise ValueError(
                "noisy_input_ids and clean_input_ids must have the same shape"
            )
        self._activate_sequence_length(int(noisy_input_ids.shape[1]))
        if not self.runtime.enabled or self.runtime.kv_backend not in {
            "ring",
            "replicated",
        }:
            raise RuntimeError(
                "packed block-diffusion execution requires ring or replicated KV runtime"
            )

        device = noisy_input_ids.device
        layout = self._packed_layout(device)
        active_positions = layout.active_positions
        clean_positions = layout.clean_positions
        packed_positions = layout.packed_positions
        active_len = int(layout.active_len)

        model_dtype = self._model_dtype()
        active_hidden = self._embed_tokens(
            noisy_input_ids.index_select(1, active_positions)
        ).to(dtype=model_dtype)
        clean_hidden = self._embed_tokens(
            clean_input_ids.index_select(1, clean_positions)
        ).to(dtype=model_dtype)
        components_scale = getattr(self, "embedding_scale", None)
        if components_scale is not None:
            scale = torch.as_tensor(
                components_scale,
                device=active_hidden.device,
                dtype=active_hidden.dtype,
            )
            active_hidden = active_hidden * scale
            clean_hidden = clean_hidden * scale
        if self.self_conditioning is not None and self.self_condition_clean_tokens:
            hidden_states = torch.cat((active_hidden, clean_hidden), dim=1)
            hidden_states = self.self_conditioning(
                hidden_states,
                torch.zeros_like(hidden_states),
            )
        else:
            if self.self_conditioning is not None:
                active_hidden = self.self_conditioning(
                    active_hidden,
                    torch.zeros_like(active_hidden),
                )
            hidden_states = torch.cat((active_hidden, clean_hidden), dim=1)

        position_ids = packed_positions.unsqueeze(0).expand(hidden_states.shape[0], -1)
        position_cache: dict[str | None, tuple[torch.Tensor, torch.Tensor]] = {}

        if self.sequence_parallel:
            return self._forward_sequence_parallel(
                hidden_states=hidden_states,
                position_ids=position_ids,
                position_cache=position_cache,
                active_positions=active_positions,
                clean_positions=clean_positions,
                active_len=active_len,
            )

        cache_position = packed_positions
        for layer_index, layer_ops in enumerate(self.layers):
            local_attn_mask, global_attn_mask = self._packed_attention_masks_for_layer(
                layout,
                layer_ops,
            )
            position_embeddings = self._position_embeddings_for_layer(
                layer_ops,
                hidden_states=hidden_states,
                position_ids=position_ids,
                cache=position_cache,
            )
            if self._full_layer_checkpointing_enabled():
                hidden_states = self._checkpoint_decoder_layer(
                    layer_ops,
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    cache_position=cache_position,
                    active_len=active_len,
                    local_attn_mask=local_attn_mask,
                    global_attn_mask=global_attn_mask,
                )
            else:
                hidden_states = self._decoder_layer_forward(
                    layer_ops,
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    cache_position=cache_position,
                    active_len=active_len,
                    local_attn_mask=local_attn_mask,
                    global_attn_mask=global_attn_mask,
                )
            self._debug_check_layer_tensor(hidden_states, layer_index)
        hidden_states = self.norm(hidden_states)
        active_hidden = hidden_states[:, :active_len]
        return active_hidden, active_positions

    def forward_diffusiongemma_native(
        self,
        *,
        noisy_input_ids: torch.Tensor,
        clean_input_ids: torch.Tensor,
        labels: torch.Tensor,
        scored_mask: torch.Tensor,
        encoder_valid_mask: torch.Tensor | None,
        decoder_position_ids: torch.Tensor,
        decoder_valid_mask: torch.Tensor | None,
        decoder_block_ids: torch.Tensor,
        response_start: int,
        block_token_counts: torch.Tensor,
        valid_block_mask: torch.Tensor,
        self_conditioning_mask: torch.Tensor,
        self_conditioning_execution_count: int | None = None,
        self_conditioning_execute_all: bool = False,
        encoder_loss_weight: float,
        decoder_loss_denominator: float | torch.Tensor | None = None,
        encoder_loss_denominator: float | torch.Tensor | None = None,
        self_conditioning_row_chunk_size: int,
        self_conditioning_vocab_chunk_size: int,
    ) -> DiffusionGemmaNativeOutput:
        """Run native uniform-state SFT while reusing one clean K/V stream."""

        if noisy_input_ids.ndim != 2 or clean_input_ids.ndim != 2:
            raise ValueError("native clean and decoder input IDs must be rank two")
        if noisy_input_ids.shape[0] != clean_input_ids.shape[0]:
            raise ValueError("native clean and decoder batch sizes must match")
        if labels.shape != noisy_input_ids.shape or scored_mask.shape != labels.shape:
            raise ValueError(
                "native labels and scored_mask must match decoder input_ids"
            )
        decoder_length = int(noisy_input_ids.shape[1])
        if decoder_length <= 0 or decoder_length % self.block_size:
            raise ValueError("native decoder length must contain complete blocks")
        if decoder_position_ids.shape != (decoder_length,):
            raise ValueError("decoder_position_ids must have shape [decoder_length]")
        if decoder_block_ids.shape != (decoder_length,):
            raise ValueError("decoder_block_ids must have shape [decoder_length]")
        identity_response_layout = (
            int(response_start) == 0
            and decoder_valid_mask is None
            and decoder_length == int(clean_input_ids.shape[1])
        )
        if decoder_valid_mask is not None:
            if decoder_valid_mask.shape != noisy_input_ids.shape:
                raise ValueError("decoder_valid_mask must match decoder input_ids")
            if not bool(decoder_valid_mask.eq(decoder_valid_mask[:1]).all()):
                raise ValueError("native batches require one shared decoder layout")
            expected_valid = (
                decoder_block_ids.ge(0).unsqueeze(0).expand_as(decoder_valid_mask)
            )
            if not torch.equal(decoder_valid_mask, expected_valid):
                raise ValueError("decoder validity and block metadata disagree")
        elif not identity_response_layout and bool(decoder_block_ids.lt(0).any()):
            raise ValueError("negative decoder blocks require decoder_valid_mask")
        if (
            encoder_valid_mask is not None
            and encoder_valid_mask.shape != clean_input_ids.shape
        ):
            raise ValueError("encoder_valid_mask must match clean_input_ids")
        if self_conditioning_mask.shape != (int(clean_input_ids.shape[0]),):
            raise ValueError("self_conditioning_mask must have shape [batch]")
        self._activate_sequence_length(int(clean_input_ids.shape[1]))
        expected_blocks = decoder_length // self.block_size
        expected_block_shape = (int(clean_input_ids.shape[0]), expected_blocks)
        if block_token_counts.shape != expected_block_shape:
            raise ValueError("block_token_counts must have shape [batch, blocks]")
        if valid_block_mask.shape != expected_block_shape:
            raise ValueError("valid_block_mask must have shape [batch, blocks]")
        if not self.encoder_causal_attention:
            raise RuntimeError("native DiffusionGemma requires causal clean attention")
        if self.training and (
            not self.activation_checkpointing
            or self.activation_checkpointing_scope != "full"
        ):
            raise RuntimeError(
                "native DiffusionGemma training requires full activation checkpointing"
            )
        if not self.runtime.enabled or self.runtime.kv_backend not in {
            "ring",
            "replicated",
        }:
            raise RuntimeError(
                "native DiffusionGemma requires ring or replicated KV execution"
            )
        if self.self_conditioning is None:
            raise RuntimeError("native DiffusionGemma requires self-conditioning")

        active_positions, clean_positions, pure_context = self._native_stream_positions(
            clean_input_ids.device,
            decoder_length=decoder_length,
        )
        clean_stream = self._native_encode_clean(
            clean_input_ids,
            clean_positions,
            pure_context=pure_context,
        )
        active_hidden = self._native_embed_active(noisy_input_ids, active_positions)

        conditioned_examples = torch.nonzero(
            self_conditioning_mask,
            as_tuple=False,
        ).flatten()
        conditioned_count = int(conditioned_examples.numel())
        execution_count = (
            conditioned_count
            if self_conditioning_execution_count is None
            else int(self_conditioning_execution_count)
        )
        batch_size = int(active_hidden.shape[0])
        if self_conditioning_execute_all and execution_count != batch_size:
            raise ValueError(
                "full self-conditioning execution requires every batch row"
            )
        if not conditioned_count <= execution_count <= batch_size:
            raise ValueError(
                "self_conditioning_execution_count must be between the local "
                "selected count and batch size"
            )
        if execution_count > 0:
            if self_conditioning_execute_all:
                execution_indices = torch.arange(
                    batch_size,
                    dtype=torch.long,
                    device=active_hidden.device,
                )
            elif conditioned_count == 0:
                execution_indices = torch.zeros(
                    (execution_count,),
                    dtype=torch.long,
                    device=active_hidden.device,
                )
            elif execution_count > conditioned_count:
                padding = conditioned_examples[:1].expand(
                    execution_count - conditioned_count
                )
                execution_indices = torch.cat((conditioned_examples, padding))
            else:
                execution_indices = conditioned_examples
            with torch.no_grad():
                selected_active = active_hidden.index_select(
                    0,
                    execution_indices,
                ).detach()
                selected_active = self.self_conditioning(
                    selected_active,
                    torch.zeros_like(selected_active),
                )
                selected_first_pass = self._native_decode_active(
                    selected_active,
                    active_positions,
                    (
                        clean_stream.detached()
                        if batch_size == 1
                        else clean_stream.index_select_batch(
                            execution_indices
                        ).detached()
                    ),
                    decoder_position_ids=(
                        None if identity_response_layout else decoder_position_ids
                    ),
                    decoder_block_ids=(
                        None if identity_response_layout else decoder_block_ids
                    ),
                    response_start=int(response_start),
                    pure_context=pure_context,
                )
            first_pass_hidden = torch.zeros_like(active_hidden)
            if conditioned_count > 0:
                conditioned_first_pass = (
                    selected_first_pass.index_select(0, conditioned_examples)
                    if self_conditioning_execute_all
                    else selected_first_pass[:conditioned_count]
                )
                first_pass_hidden.index_copy_(
                    0,
                    conditioned_examples,
                    conditioned_first_pass,
                )
        else:
            first_pass_hidden = torch.zeros_like(active_hidden)
        conditioned_hidden = self._native_apply_self_conditioning(
            active_hidden,
            first_pass_hidden,
            self_conditioning_mask,
            row_chunk_size=int(self_conditioning_row_chunk_size),
            vocab_chunk_size=int(self_conditioning_vocab_chunk_size),
        )
        decoder_hidden = self._native_decode_active(
            conditioned_hidden,
            active_positions,
            clean_stream,
            decoder_position_ids=(
                None if identity_response_layout else decoder_position_ids
            ),
            decoder_block_ids=(None if identity_response_layout else decoder_block_ids),
            response_start=int(response_start),
            pure_context=pure_context,
        )
        decoder_loss = self._native_decoder_loss(
            decoder_hidden,
            labels,
            active_positions,
            block_token_counts,
            valid_block_mask,
            loss_denominator=decoder_loss_denominator,
        )
        encoder_loss = self._native_encoder_ar_loss(
            clean_stream.hidden,
            clean_input_ids,
            clean_positions,
            encoder_valid_mask=encoder_valid_mask,
            loss_denominator=encoder_loss_denominator,
        )
        total_loss = decoder_loss + float(encoder_loss_weight) * encoder_loss
        return DiffusionGemmaNativeOutput(
            loss=total_loss,
            decoder_loss=decoder_loss,
            encoder_loss=encoder_loss,
            decoder_hidden=decoder_hidden,
            clean_hidden=clean_stream.hidden,
            active_positions=active_positions,
            clean_positions=clean_positions,
        )

    def _native_stream_positions(
        self,
        device: torch.device,
        *,
        decoder_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        decoder_length = self.seq_len if decoder_length is None else int(decoder_length)
        if decoder_length <= 0 or decoder_length % self.block_size:
            raise ValueError("native decoder length must contain complete blocks")
        layout = self._packed_layout(device) if decoder_length == self.seq_len else None
        clean_positions = (
            layout.clean_positions
            if layout is not None
            else self._local_clean_positions(device)
        )
        if self.runtime.kv_backend == "replicated":
            # The legacy packed path retains only the prefix needed by this
            # rank's decoder blocks. Native SFT also co-trains a full clean AR
            # stream, so CP=1 needs the complete clean sequence.
            clean_positions = torch.arange(
                self.seq_len,
                device=device,
                dtype=torch.long,
            )
        pure_context = (
            self.runtime.kv_backend == "ring"
            and int(getattr(self.runtime, "context_attention_size", 1) or 1) > 1
            and int(getattr(self.runtime, "block_parallel_size", 1) or 1) == 1
        )
        active_positions = None if layout is None else layout.active_positions
        if pure_context:
            active_positions = active_query_indices_for_context_rank(
                active_len=decoder_length,
                block_size=self.block_size,
                context_parallel_size=int(self.runtime.context_attention_size),
                context_parallel_rank=int(self.runtime.context_parallel_rank),
                device=device,
            )
        elif active_positions is None:
            active_positions = active_token_indices(
                seq_len=decoder_length,
                block_size=self.block_size,
                device=device,
                runtime=self.runtime,
            )
            if active_positions is None:
                active_positions = torch.arange(
                    decoder_length,
                    device=device,
                    dtype=torch.long,
                )
            elif active_positions.numel() == 0:
                raise RuntimeError("native BP/CP requires non-empty decoder ownership")
        return active_positions, clean_positions, pure_context

    def _native_embed_positions(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self._embed_tokens(input_ids.index_select(1, positions)).to(
            dtype=self._model_dtype()
        )
        if self.embedding_scale is not None:
            hidden = hidden * torch.as_tensor(
                self.embedding_scale,
                device=hidden.device,
                dtype=hidden.dtype,
            )
        return hidden

    def _native_embed_active(
        self,
        noisy_input_ids: torch.Tensor,
        active_positions: torch.Tensor,
    ) -> torch.Tensor:
        return self._native_embed_positions(noisy_input_ids, active_positions)

    def _native_attention_masks(
        self,
        *,
        query_positions: torch.Tensor,
        clean_positions: torch.Tensor,
        active_key_positions: torch.Tensor,
        query_is_clean: bool,
        layer_ops: NemotronDecoderLayerOps,
        pure_context: bool,
        query_block_ids: torch.Tensor | None = None,
        active_key_block_ids: torch.Tensor | None = None,
        decoder_response_start: int | None = None,
    ) -> tuple[BlockDenoisingLocalActiveMask, BlockDenoisingGlobalCleanMask]:
        cache_key: tuple[object, ...] | None = None
        if (
            not pure_context
            and int(getattr(self.runtime, "block_parallel_size", 1) or 1) > 1
        ):
            cache_key = (
                "native_fused",
                bool(query_is_clean),
                layer_ops.layer_type,
                getattr(layer_ops.attention, "sliding_window", None),
                int(query_positions.data_ptr()),
                int(query_positions.numel()),
                int(clean_positions.data_ptr()),
                int(clean_positions.numel()),
                int(active_key_positions.data_ptr()),
                int(active_key_positions.numel()),
                None if query_block_ids is None else int(query_block_ids.data_ptr()),
                (
                    None
                    if active_key_block_ids is None
                    else int(active_key_block_ids.data_ptr())
                ),
                decoder_response_start,
                query_positions.device.type,
                query_positions.device.index,
            )
            cached = self._layer_attention_mask_cache.get(cache_key)
            if cached is not None:
                return cached
        metadata_positions = query_positions
        if pure_context and not query_is_clean:
            # The persistent pure-CP kernel receives compact local query rows,
            # but selects their metadata from the complete logical canvas.
            metadata_positions = torch.arange(
                self.seq_len,
                device=query_positions.device,
                dtype=torch.long,
            )
        query_blocks = (
            (metadata_positions // self.block_size).to(torch.int32)
            if query_block_ids is None
            else query_block_ids.to(device=query_positions.device, dtype=torch.int32)
        )
        if query_blocks.shape != metadata_positions.shape:
            raise ValueError("query_block_ids must match native query metadata")
        clean_queries = torch.full_like(
            metadata_positions,
            bool(query_is_clean),
            dtype=torch.bool,
        )
        if active_key_block_ids is not None:
            active_blocks = active_key_block_ids.to(
                device=query_positions.device,
                dtype=torch.int32,
            )
        elif pure_context and not query_is_clean:
            active_blocks = (
                torch.arange(
                    self.seq_len,
                    device=query_positions.device,
                    dtype=torch.long,
                )
                // self.block_size
            ).to(torch.int32)
        else:
            active_blocks = (active_key_positions // self.block_size).to(torch.int32)
        bounds = self._clean_attention_bounds(
            query_positions=metadata_positions,
            query_is_clean=clean_queries,
            layer_ops=layer_ops,
            query_blocks=query_blocks if decoder_response_start is not None else None,
            decoder_response_start=decoder_response_start,
        )
        backward_chunk = self._cp_bp_backward_query_chunk_size()
        debug_nonfinite = self._debug_nonfinite_attention_enabled()
        local_mask = BlockDenoisingLocalActiveMask(
            query_blocks=query_blocks,
            query_is_clean=clean_queries,
            active_blocks=active_blocks,
            block_size=self.block_size,
            backward_query_chunk_size=backward_chunk,
            debug_nonfinite_attention=debug_nonfinite,
        )
        global_mask = BlockDenoisingGlobalCleanMask(
            query_blocks=query_blocks,
            query_is_clean=clean_queries,
            block_size=self.block_size,
            clean_context_window=(
                layer_ops.attention.sliding_window
                if layer_ops.layer_type == "sliding_attention"
                else None
            ),
            query_clean_bounds=bounds,
            clean_key_positions=clean_positions.to(torch.int32),
            backward_query_chunk_size=backward_chunk,
            debug_nonfinite_attention=debug_nonfinite,
        )
        masks = (local_mask, global_mask)
        if cache_key is not None:
            self._layer_attention_mask_cache[cache_key] = masks
        return masks

    def _native_attention_from_projected(
        self,
        layer_ops: NemotronDecoderLayerOps,
        *,
        hidden_states: torch.Tensor,
        query: torch.Tensor,
        active_key: torch.Tensor,
        active_value: torch.Tensor,
        clean_key: torch.Tensor,
        clean_value: torch.Tensor,
        output_gate: torch.Tensor | None,
        local_attn_mask: BlockDenoisingLocalActiveMask,
        global_attn_mask: BlockDenoisingGlobalCleanMask,
        pure_context: bool,
        sequence_parallel_meta: _SequenceParallelPackedMeta | None = None,
    ) -> torch.Tensor:
        if pure_context:
            output = pure_context_persistent_block_denoising_attention_bshd(
                query,
                active_key,
                active_value,
                clean_key,
                clean_value,
                global_seq_len=self.seq_len,
                local_attn_mask=local_attn_mask,
                global_attn_mask=global_attn_mask,
                scale=layer_ops.attention.scale,
                runtime=self.runtime,
            )
        elif self.runtime.kv_backend == "ring":
            output = fused_block_context_attention_bshd(
                query,
                active_key,
                active_value,
                clean_key,
                clean_value,
                global_seq_len=self.seq_len,
                local_attn_mask=local_attn_mask,
                global_attn_mask=global_attn_mask,
                scale=layer_ops.attention.scale,
                runtime=self.runtime,
                key_chunk_size=self.ring_attention_key_chunk_size,
            )
        else:
            output = replicated_block_denoising_attention_bshd(
                query,
                active_key,
                active_value,
                clean_key,
                clean_value,
                local_attn_mask=local_attn_mask,
                global_attn_mask=global_attn_mask,
                scale=layer_ops.attention.scale,
            )
        output_shape = (
            (sequence_parallel_meta.total_rows,)
            if sequence_parallel_meta is not None
            else hidden_states.shape[:-1]
        )
        output = output.reshape(*output_shape, -1).contiguous()
        if output_gate is not None:
            output = output * torch.sigmoid(output_gate.reshape(*output_shape, -1))
        if sequence_parallel_meta is not None:
            output = self._pad_rows(output, sequence_parallel_meta.padded_rows)
        return self._attention_output_projection(layer_ops, output)

    def _native_finish_layer(
        self,
        layer_ops: NemotronDecoderLayerOps,
        *,
        residual: torch.Tensor,
        attention_output: torch.Tensor,
        clean_only: bool,
        sequence_parallel_meta: _SequenceParallelPackedMeta | None = None,
    ) -> torch.Tensor:
        if layer_ops.layer_style == "gemma4":
            hidden_states = layer_ops.post_attention_layernorm(attention_output)
            hidden_states = hidden_states + residual
            return self._gemma4_feed_forward_block(
                layer_ops,
                hidden_states,
                active_len=0 if clean_only else None,
                sequence_parallel_meta=sequence_parallel_meta,
                checkpoint_mlp=False,
            )
        hidden_states = residual + attention_output
        return self._mlp_block_forward(
            layer_ops,
            hidden_states,
            checkpoint_mlp=False,
        )

    def _native_clean_layer_forward(
        self,
        layer_ops: NemotronDecoderLayerOps,
        hidden_states: torch.Tensor,
        *,
        position_embeddings: tuple[torch.Tensor | None, torch.Tensor | None],
        clean_positions: torch.Tensor,
        local_attn_mask: BlockDenoisingLocalActiveMask,
        global_attn_mask: BlockDenoisingGlobalCleanMask,
        pure_context: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        normalized = layer_ops.input_layernorm(residual)
        query, key, value, output_gate = self._project_qkv_bshd(
            layer_ops,
            hidden_states=normalized,
            position_embeddings=position_embeddings,
            cache_position=clean_positions,
        )
        empty_key = key[:, :0]
        empty_value = value[:, :0]
        attention_output = self._native_attention_from_projected(
            layer_ops,
            hidden_states=normalized,
            query=query,
            active_key=empty_key,
            active_value=empty_value,
            clean_key=key,
            clean_value=value,
            output_gate=output_gate,
            local_attn_mask=local_attn_mask,
            global_attn_mask=global_attn_mask,
            pure_context=bool(pure_context),
        )
        next_hidden = self._native_finish_layer(
            layer_ops,
            residual=residual,
            attention_output=attention_output,
            clean_only=True,
        )
        return next_hidden, key, value

    def _native_clean_layer_forward_sequence_parallel(
        self,
        layer_ops: NemotronDecoderLayerOps,
        hidden_shard: torch.Tensor,
        *,
        meta: _SequenceParallelPackedMeta,
        position_embeddings: tuple[torch.Tensor | None, torch.Tensor | None],
        clean_positions: torch.Tensor,
        local_attn_mask: BlockDenoisingLocalActiveMask,
        global_attn_mask: BlockDenoisingGlobalCleanMask,
        pure_context: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_shard
        te_layer = self._te_packed_by_layer_id.get(id(layer_ops.layer))
        normalized = (
            residual
            if te_layer is not None and te_layer.qkv_fuses_input_norm
            else layer_ops.input_layernorm(residual)
        )
        query, key, value, output_gate = self._project_qkv_bshd_sequence_parallel(
            layer_ops,
            hidden_shard=normalized,
            meta=meta,
            position_embeddings=position_embeddings,
            cache_position=clean_positions,
        )
        attention_output = self._native_attention_from_projected(
            layer_ops,
            hidden_states=normalized,
            query=query,
            active_key=key[:, :0],
            active_value=value[:, :0],
            clean_key=key,
            clean_value=value,
            output_gate=output_gate,
            local_attn_mask=local_attn_mask,
            global_attn_mask=global_attn_mask,
            pure_context=bool(pure_context),
            sequence_parallel_meta=meta,
        )
        next_hidden = self._native_finish_layer(
            layer_ops,
            residual=residual,
            attention_output=attention_output,
            clean_only=True,
            sequence_parallel_meta=meta,
        )
        return next_hidden, key, value

    def _native_encode_clean(
        self,
        clean_input_ids: torch.Tensor,
        clean_positions: torch.Tensor,
        *,
        pure_context: bool = False,
    ) -> _DiffusionGemmaCleanStream:
        hidden_states = self._native_embed_positions(clean_input_ids, clean_positions)
        sequence_parallel_meta = (
            self._native_sequence_parallel_meta(hidden_states)
            if bool(getattr(self, "sequence_parallel", False))
            else None
        )
        if sequence_parallel_meta is not None:
            hidden_states = self._native_scatter_stream(
                hidden_states,
                sequence_parallel_meta,
            )
        stream_batch_size = (
            sequence_parallel_meta.batch_size
            if sequence_parallel_meta is not None
            else int(hidden_states.shape[0])
        )
        position_ids = clean_positions.unsqueeze(0).expand(stream_batch_size, -1)
        position_cache: dict[
            str | None,
            tuple[torch.Tensor | None, torch.Tensor | None],
        ] = {}
        layer_key_values: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_index, layer_ops in enumerate(self.layers):
            position_embeddings = self._position_embeddings_for_layer(
                layer_ops,
                hidden_states=hidden_states,
                position_ids=position_ids,
                cache=position_cache,
            )
            local_mask, global_mask = self._native_attention_masks(
                query_positions=clean_positions,
                clean_positions=clean_positions,
                active_key_positions=clean_positions[:0],
                query_is_clean=True,
                layer_ops=layer_ops,
                pure_context=bool(pure_context),
            )

            def layer_forward(
                states: torch.Tensor,
                ops: NemotronDecoderLayerOps = layer_ops,
                pe: tuple[
                    torch.Tensor | None,
                    torch.Tensor | None,
                ] = position_embeddings,
                local: BlockDenoisingLocalActiveMask = local_mask,
                global_: BlockDenoisingGlobalCleanMask = global_mask,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                if sequence_parallel_meta is not None:
                    return self._native_clean_layer_forward_sequence_parallel(
                        ops,
                        states,
                        meta=sequence_parallel_meta,
                        position_embeddings=pe,
                        clean_positions=clean_positions,
                        local_attn_mask=local,
                        global_attn_mask=global_,
                        pure_context=bool(pure_context),
                    )
                return self._native_clean_layer_forward(
                    ops,
                    states,
                    position_embeddings=pe,
                    clean_positions=clean_positions,
                    local_attn_mask=local,
                    global_attn_mask=global_,
                    pure_context=bool(pure_context),
                )

            if self._full_layer_checkpointing_enabled() and torch.is_grad_enabled():
                if self._requires_moe_checkpoint_context:
                    hidden_states, key, value = checkpoint(
                        layer_forward,
                        hidden_states,
                        use_reentrant=False,
                        context_fn=moe_route_checkpoint_context_fn,
                    )
                else:
                    hidden_states, key, value = checkpoint(
                        layer_forward,
                        hidden_states,
                        use_reentrant=self._checkpoint_uses_reentrant_autograd(),
                    )
            else:
                hidden_states, key, value = layer_forward(hidden_states)
            layer_key_values.append((key, value))
            self._debug_check_layer_tensor(hidden_states, layer_index)
        hidden_states = self.norm(hidden_states)
        if sequence_parallel_meta is not None:
            # Vocab-parallel consumers already sum their partial input
            # gradients, so this boundary only restores each rank's rows.
            hidden_states = self._native_gather_stream(
                hidden_states,
                sequence_parallel_meta,
                reduce_scatter_grad=False,
            )
        return _DiffusionGemmaCleanStream(
            hidden=hidden_states,
            positions=clean_positions,
            layer_key_values=tuple(layer_key_values),
        )

    def _native_active_layer_forward(
        self,
        layer_ops: NemotronDecoderLayerOps,
        hidden_states: torch.Tensor,
        clean_key: torch.Tensor,
        clean_value: torch.Tensor,
        *,
        position_embeddings: tuple[torch.Tensor | None, torch.Tensor | None],
        active_positions: torch.Tensor,
        local_attn_mask: BlockDenoisingLocalActiveMask,
        global_attn_mask: BlockDenoisingGlobalCleanMask,
        pure_context: bool,
    ) -> torch.Tensor:
        residual = hidden_states
        normalized = layer_ops.input_layernorm(residual)
        query, key, value, output_gate = self._project_qkv_bshd(
            layer_ops,
            hidden_states=normalized,
            position_embeddings=position_embeddings,
            cache_position=active_positions,
        )
        attention_output = self._native_attention_from_projected(
            layer_ops,
            hidden_states=normalized,
            query=query,
            active_key=key,
            active_value=value,
            clean_key=clean_key,
            clean_value=clean_value,
            output_gate=output_gate,
            local_attn_mask=local_attn_mask,
            global_attn_mask=global_attn_mask,
            pure_context=pure_context,
        )
        return self._native_finish_layer(
            layer_ops,
            residual=residual,
            attention_output=attention_output,
            clean_only=False,
        )

    def _native_active_layer_forward_sequence_parallel(
        self,
        layer_ops: NemotronDecoderLayerOps,
        hidden_shard: torch.Tensor,
        clean_key: torch.Tensor,
        clean_value: torch.Tensor,
        *,
        meta: _SequenceParallelPackedMeta,
        position_embeddings: tuple[torch.Tensor | None, torch.Tensor | None],
        active_positions: torch.Tensor,
        local_attn_mask: BlockDenoisingLocalActiveMask,
        global_attn_mask: BlockDenoisingGlobalCleanMask,
        pure_context: bool,
    ) -> torch.Tensor:
        residual = hidden_shard
        te_layer = self._te_packed_by_layer_id.get(id(layer_ops.layer))
        normalized = (
            residual
            if te_layer is not None and te_layer.qkv_fuses_input_norm
            else layer_ops.input_layernorm(residual)
        )
        query, key, value, output_gate = self._project_qkv_bshd_sequence_parallel(
            layer_ops,
            hidden_shard=normalized,
            meta=meta,
            position_embeddings=position_embeddings,
            cache_position=active_positions,
        )
        attention_output = self._native_attention_from_projected(
            layer_ops,
            hidden_states=normalized,
            query=query,
            active_key=key,
            active_value=value,
            clean_key=clean_key,
            clean_value=clean_value,
            output_gate=output_gate,
            local_attn_mask=local_attn_mask,
            global_attn_mask=global_attn_mask,
            pure_context=bool(pure_context),
            sequence_parallel_meta=meta,
        )
        return self._native_finish_layer(
            layer_ops,
            residual=residual,
            attention_output=attention_output,
            clean_only=False,
            sequence_parallel_meta=meta,
        )

    def _native_decode_active(
        self,
        active_hidden: torch.Tensor,
        active_positions: torch.Tensor,
        clean_stream: _DiffusionGemmaCleanStream,
        *,
        decoder_position_ids: torch.Tensor | None = None,
        decoder_block_ids: torch.Tensor | None = None,
        response_start: int = 0,
        pure_context: bool,
    ) -> torch.Tensor:
        if len(clean_stream.layer_key_values) != len(self.layers):
            raise RuntimeError("clean K/V cache does not match the decoder layer count")
        hidden_states = active_hidden
        sequence_parallel_meta = (
            self._native_sequence_parallel_meta(hidden_states)
            if bool(getattr(self, "sequence_parallel", False))
            else None
        )
        if sequence_parallel_meta is not None:
            hidden_states = self._native_scatter_stream(
                hidden_states,
                sequence_parallel_meta,
            )
        rotary_positions = (
            active_positions
            if decoder_position_ids is None
            else decoder_position_ids.index_select(0, active_positions)
        )
        position_ids = rotary_positions.unsqueeze(0).expand(
            (
                sequence_parallel_meta.batch_size
                if sequence_parallel_meta is not None
                else int(hidden_states.shape[0])
            ),
            -1,
        )
        if decoder_block_ids is None:
            query_block_ids = None
            active_key_block_ids = None
        elif pure_context:
            query_block_ids = decoder_block_ids
            active_key_block_ids = decoder_block_ids
        else:
            query_block_ids = decoder_block_ids.index_select(0, active_positions)
            active_key_block_ids = query_block_ids
        position_cache: dict[
            str | None,
            tuple[torch.Tensor | None, torch.Tensor | None],
        ] = {}
        for layer_index, (layer_ops, clean_kv) in enumerate(
            zip(self.layers, clean_stream.layer_key_values, strict=True)
        ):
            clean_key, clean_value = clean_kv
            position_embeddings = self._position_embeddings_for_layer(
                layer_ops,
                hidden_states=hidden_states,
                position_ids=position_ids,
                cache=position_cache,
            )
            local_mask, global_mask = self._native_attention_masks(
                query_positions=active_positions,
                clean_positions=clean_stream.positions,
                active_key_positions=active_positions,
                query_is_clean=False,
                layer_ops=layer_ops,
                pure_context=bool(pure_context),
                query_block_ids=query_block_ids,
                active_key_block_ids=active_key_block_ids,
                decoder_response_start=(
                    int(response_start) if decoder_block_ids is not None else None
                ),
            )

            def layer_forward(
                states: torch.Tensor,
                clean_k: torch.Tensor,
                clean_v: torch.Tensor,
                ops: NemotronDecoderLayerOps = layer_ops,
                pe: tuple[
                    torch.Tensor | None,
                    torch.Tensor | None,
                ] = position_embeddings,
                local: BlockDenoisingLocalActiveMask = local_mask,
                global_: BlockDenoisingGlobalCleanMask = global_mask,
            ) -> torch.Tensor:
                if sequence_parallel_meta is not None:
                    return self._native_active_layer_forward_sequence_parallel(
                        ops,
                        states,
                        clean_k,
                        clean_v,
                        meta=sequence_parallel_meta,
                        position_embeddings=pe,
                        active_positions=rotary_positions,
                        local_attn_mask=local,
                        global_attn_mask=global_,
                        pure_context=bool(pure_context),
                    )
                return self._native_active_layer_forward(
                    ops,
                    states,
                    clean_k,
                    clean_v,
                    position_embeddings=pe,
                    active_positions=rotary_positions,
                    local_attn_mask=local,
                    global_attn_mask=global_,
                    pure_context=bool(pure_context),
                )

            if self._full_layer_checkpointing_enabled() and torch.is_grad_enabled():
                if self._requires_moe_checkpoint_context:
                    hidden_states = checkpoint(
                        layer_forward,
                        hidden_states,
                        clean_key,
                        clean_value,
                        use_reentrant=False,
                        context_fn=moe_route_checkpoint_context_fn,
                    )
                else:
                    hidden_states = checkpoint(
                        layer_forward,
                        hidden_states,
                        clean_key,
                        clean_value,
                        use_reentrant=self._checkpoint_uses_reentrant_autograd(),
                    )
            else:
                hidden_states = layer_forward(hidden_states, clean_key, clean_value)
            self._debug_check_layer_tensor(hidden_states, layer_index)
        hidden_states = self.norm(hidden_states)
        if sequence_parallel_meta is not None:
            # See the clean-stream boundary above: loss-parallel backward
            # supplies an already-reduced full-hidden gradient.
            hidden_states = self._native_gather_stream(
                hidden_states,
                sequence_parallel_meta,
                reduce_scatter_grad=False,
            )
        return hidden_states

    def _native_apply_self_conditioning(
        self,
        active_hidden: torch.Tensor,
        first_pass_hidden: torch.Tensor,
        self_conditioning_mask: torch.Tensor,
        *,
        row_chunk_size: int,
        vocab_chunk_size: int,
    ) -> torch.Tensor:
        projection_hidden = first_pass_hidden
        if self.output_multiplier is not None:
            projection_hidden = projection_hidden * float(self.output_multiplier)
        weight = getattr(self.output_head, "weight", None)
        if not isinstance(weight, torch.Tensor):
            raise RuntimeError(
                "native DiffusionGemma self-conditioning requires a linear output weight"
            )
        embedding_scale = float(getattr(self, "soft_embedding_scale", 1.0))
        selected = torch.nonzero(self_conditioning_mask, as_tuple=False).flatten()
        soft_embeddings = torch.zeros_like(active_hidden)
        if int(selected.numel()) > 0:
            selected_soft = streaming_soft_embedding(
                projection_hidden.index_select(0, selected),
                weight,
                row_chunk_size=int(row_chunk_size),
                vocab_chunk_size=int(vocab_chunk_size),
                logit_softcap=self.final_logit_softcap,
                embedding_scale=embedding_scale,
                tensor_parallel_group=getattr(
                    self.runtime,
                    "tensor_parallel_group",
                    None,
                ),
                tensor_parallel_size=int(
                    getattr(self.runtime, "tensor_parallel_size", 1) or 1
                ),
            )
            soft_embeddings.index_copy_(0, selected, selected_soft)
        return self.self_conditioning(active_hidden, soft_embeddings)

    def _native_token_cross_entropy(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if self.output_multiplier is not None:
            hidden_states = hidden_states * float(self.output_multiplier)
        parallel_ce = getattr(self.output_head, "parallel_cross_entropy", None)
        if callable(parallel_ce):
            losses = parallel_ce(
                hidden_states,
                labels,
                ignore_index=-100,
                exclude_index=None,
                reduction="none",
                logit_softcap=self.final_logit_softcap,
            )
        else:
            weight = getattr(self.output_head, "weight", None)
            bias = getattr(self.output_head, "bias", None)
            if not isinstance(weight, torch.Tensor):
                raise RuntimeError(
                    "native DiffusionGemma requires a linear output head"
                )
            losses = tiled_linear_cross_entropy(
                hidden_states,
                weight,
                labels,
                bias=bias if isinstance(bias, torch.Tensor) else None,
                reduction="none",
                ignore_index=-100,
                exclude_index=None,
                logit_softcap=self.final_logit_softcap,
                dtype=hidden_states.dtype,
                weight_layout="vocab_first",
            )
        return losses.reshape(labels.shape).float()

    def _native_active_loss_scale(self) -> float:
        if (
            int(getattr(self.runtime, "block_parallel_size", 1) or 1) == 1
            and int(getattr(self.runtime, "context_attention_size", 1) or 1) > 1
        ):
            return float(self.runtime.context_attention_size)
        return float(runtime_loss_scale(self.runtime))

    def _native_decoder_loss(
        self,
        active_hidden: torch.Tensor,
        labels: torch.Tensor,
        active_positions: torch.Tensor,
        block_token_counts: torch.Tensor,
        valid_block_mask: torch.Tensor,
        *,
        loss_denominator: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        active_labels = labels.index_select(1, active_positions)
        valid_rows = active_labels != -100
        flat_hidden = active_hidden.reshape(-1, active_hidden.shape[-1])
        flat_labels = active_labels.reshape(-1)
        flat_valid = valid_rows.reshape(-1)
        valid_indices = torch.nonzero(flat_valid, as_tuple=False).flatten()
        if int(valid_indices.numel()) == 0:
            return (
                zero_loss_with_module_parameters(
                    active_hidden,
                    self.output_head,
                )
                * self._native_active_loss_scale()
            )
        token_losses = self._native_token_cross_entropy(
            flat_hidden.index_select(0, valid_indices),
            flat_labels.index_select(0, valid_indices),
        )
        local_counts = block_token_counts.index_select(
            1,
            active_positions // self.block_size,
        ).clamp_min(1)
        valid_blocks_per_example = valid_block_mask.sum(dim=1).clamp_min(1)
        valid_examples = valid_block_mask.any(dim=1).sum().clamp_min(1)
        denominator = (
            valid_examples
            if loss_denominator is None
            else torch.as_tensor(
                loss_denominator,
                device=active_hidden.device,
                dtype=torch.float32,
            )
        )
        coefficients = local_counts.reciprocal().to(token_losses.dtype)
        coefficients = coefficients / valid_blocks_per_example[:, None].to(
            token_losses.dtype
        )
        coefficients = coefficients / denominator.to(token_losses.dtype)
        local_loss = (
            token_losses.reshape(-1)
            * coefficients.reshape(-1).index_select(0, valid_indices)
        ).sum()
        return local_loss * self._native_active_loss_scale()

    def _native_encoder_ar_loss(
        self,
        clean_hidden: torch.Tensor,
        clean_input_ids: torch.Tensor,
        clean_positions: torch.Tensor,
        *,
        encoder_valid_mask: torch.Tensor | None = None,
        loss_denominator: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        source_indices = torch.nonzero(
            clean_positions < (clean_input_ids.shape[1] - 1),
            as_tuple=False,
        ).flatten()
        local_hidden = clean_hidden.index_select(1, source_indices)
        source_positions = clean_positions.index_select(0, source_indices)
        target_positions = source_positions + 1
        local_labels = clean_input_ids.index_select(1, target_positions)
        if encoder_valid_mask is None:
            valid_rows = torch.ones_like(local_labels, dtype=torch.bool)
            target_count = torch.as_tensor(
                clean_input_ids.shape[0] * max(0, clean_input_ids.shape[1] - 1),
                device=clean_hidden.device,
                dtype=torch.int64,
            ).clamp_min(1)
        else:
            if encoder_valid_mask.shape != clean_input_ids.shape:
                raise ValueError("encoder_valid_mask must match clean_input_ids")
            valid_rows = encoder_valid_mask.index_select(
                1, source_positions
            ) & encoder_valid_mask.index_select(1, target_positions)
            target_count = (
                (encoder_valid_mask[:, :-1] & encoder_valid_mask[:, 1:])
                .sum()
                .clamp_min(1)
            )
        denominator = (
            target_count
            if loss_denominator is None
            else torch.as_tensor(
                loss_denominator,
                device=clean_hidden.device,
                dtype=torch.float32,
            )
        )
        context_scale = float(
            int(getattr(self.runtime, "context_attention_size", 1) or 1)
        )
        flat_valid = valid_rows.reshape(-1)
        valid_indices = torch.nonzero(flat_valid, as_tuple=False).flatten()
        if int(valid_indices.numel()) == 0:
            return (
                zero_loss_with_module_parameters(
                    local_hidden,
                    self.output_head,
                )
                * context_scale
            )
        token_losses = self._native_token_cross_entropy(
            local_hidden.reshape(-1, local_hidden.shape[-1]).index_select(
                0,
                valid_indices,
            ),
            local_labels.reshape(-1).index_select(0, valid_indices),
        )
        return token_losses.sum() / denominator.to(token_losses.dtype) * context_scale

    def _embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedding_norm = self.embedding_norm
        if embedding_norm is None:
            return self.embed_tokens(input_ids)
        if getattr(self.embed_tokens, "embed_norm", None) is embedding_norm:
            embedded = torch.nn.functional.embedding(
                input_ids,
                self.embed_tokens.weight,
                padding_idx=getattr(self.embed_tokens, "padding_idx", None),
            )
        else:
            embedded = self.embed_tokens(input_ids)
        return embedding_norm(embedded)

    def distributed_block_diffusion_loss(
        self,
        active_hidden: torch.Tensor,
        labels: torch.Tensor,
        active_positions: torch.Tensor,
        loss_weights: torch.Tensor,
        *,
        vocab_size: int,
        mask_token_id: int,
        seq_len: int,
        valid_token_count: float | torch.Tensor,
        bp_loss_scale: float | None = None,
        exclude_mask_token: bool = True,
    ) -> torch.Tensor:
        del seq_len
        if active_positions.ndim == 2:
            if active_positions.shape[-1] != 2:
                raise ValueError("sequence-parallel active positions must be [N, 2]")
            active_labels = labels[
                active_positions[:, 0].to(torch.long),
                active_positions[:, 1].to(torch.long),
            ]
            active_loss_weights = loss_weights[
                active_positions[:, 0].to(torch.long),
                active_positions[:, 1].to(torch.long) // self.block_size,
            ]
        else:
            active_labels = labels.index_select(1, active_positions)
            active_loss_weights = loss_weights.index_select(
                1,
                active_positions // self.block_size,
            )
        active_hidden, active_labels, active_loss_weights = _compact_valid_loss_rows(
            active_hidden,
            active_labels,
            active_loss_weights,
            ignore_index=-100,
        )
        output_multiplier = getattr(self, "output_multiplier", None)
        if output_multiplier is not None:
            active_hidden = active_hidden * float(output_multiplier)
        if active_labels.numel() == 0:
            loss_sum = zero_loss_with_module_parameters(
                active_hidden,
                self.output_head,
            )
            loss = loss_sum / valid_token_count
            return loss * float(
                bp_loss_scale
                if bp_loss_scale is not None
                else runtime_loss_scale(self.runtime)
            )
        excluded_index = int(mask_token_id) if exclude_mask_token else None
        parallel_ce = getattr(self.output_head, "parallel_cross_entropy", None)
        if callable(parallel_ce):
            token_losses = parallel_ce(
                active_hidden,
                active_labels,
                ignore_index=-100,
                exclude_index=excluded_index,
                reduction="none",
                logit_softcap=self.final_logit_softcap,
            )
            loss_sum = (token_losses.reshape(-1).float() * active_loss_weights).sum()
        else:
            weight = getattr(self.output_head, "weight", None)
            bias = getattr(self.output_head, "bias", None)
            if isinstance(weight, torch.Tensor):
                token_losses = tiled_linear_cross_entropy(
                    active_hidden,
                    weight,
                    active_labels,
                    bias=bias if isinstance(bias, torch.Tensor) else None,
                    reduction="none",
                    ignore_index=-100,
                    exclude_index=excluded_index,
                    logit_softcap=self.final_logit_softcap,
                    dtype=active_hidden.dtype,
                    weight_layout="vocab_first",
                )
                loss_sum = (
                    token_losses.reshape(-1).float() * active_loss_weights
                ).sum()
            else:
                logits = self.output_head(active_hidden)
                if self.final_logit_softcap is not None:
                    cap = float(self.final_logit_softcap)
                    logits = torch.tanh(logits.float() / cap) * cap
                if (
                    excluded_index is not None
                    and 0 <= excluded_index < logits.shape[-1]
                ):
                    logits[..., excluded_index] = -float("inf")
                token_losses = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    active_labels.reshape(-1),
                    ignore_index=-100,
                    reduction="none",
                )
                loss_sum = (token_losses.float() * active_loss_weights).sum()
        loss = loss_sum / valid_token_count
        return loss * float(
            bp_loss_scale
            if bp_loss_scale is not None
            else runtime_loss_scale(self.runtime)
        )

    def _forward_sequence_parallel(
        self,
        *,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        position_cache: dict[str | None, tuple[torch.Tensor, torch.Tensor]],
        active_positions: torch.Tensor,
        clean_positions: torch.Tensor,
        active_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layout = self._packed_layout(hidden_states.device)
        meta = self._sequence_parallel_meta(
            hidden_states,
            active_len=int(active_len),
        )
        hidden_shard = self._scatter_packed_hidden_to_sequence_parallel(
            hidden_states,
            meta,
        )
        cache_position = layout.packed_positions
        for layer_index, layer_ops in enumerate(self.layers):
            local_attn_mask, global_attn_mask = self._packed_attention_masks_for_layer(
                layout,
                layer_ops,
            )
            position_embeddings = self._position_embeddings_for_layer(
                layer_ops,
                hidden_states=hidden_states,
                position_ids=position_ids,
                cache=position_cache,
            )
            if self._full_layer_checkpointing_enabled():
                hidden_shard = self._checkpoint_decoder_layer_sequence_parallel(
                    layer_ops,
                    hidden_shard=hidden_shard,
                    meta=meta,
                    position_embeddings=position_embeddings,
                    cache_position=cache_position,
                    active_len=int(active_len),
                    local_attn_mask=local_attn_mask,
                    global_attn_mask=global_attn_mask,
                )
            else:
                hidden_shard = self._decoder_layer_forward_sequence_parallel(
                    layer_ops,
                    hidden_shard=hidden_shard,
                    meta=meta,
                    position_embeddings=position_embeddings,
                    cache_position=cache_position,
                    active_len=int(active_len),
                    local_attn_mask=local_attn_mask,
                    global_attn_mask=global_attn_mask,
                )
            self._debug_check_layer_tensor(hidden_shard, layer_index)
        hidden_shard = self.norm(hidden_shard)
        active_hidden, active_pairs = gather_active_from_sequence_parallel_region(
            hidden_shard,
            packed_len=meta.packed_len,
            active_len=meta.active_len,
            total_rows=meta.total_rows,
            runtime=self.runtime,
        )
        original_positions = active_positions.index_select(0, active_pairs[:, 1])
        active_position_pairs = torch.stack(
            (active_pairs[:, 0], original_positions),
            dim=-1,
        )
        return active_hidden, active_position_pairs

    def _position_embeddings_for_layer(
        self,
        layer_ops: NemotronDecoderLayerOps,
        *,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        cache: dict[str | None, tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not layer_ops.attention.uses_rope:
            return None, None
        key = layer_ops.layer_type
        cached = cache.get(key)
        if cached is not None:
            return cached
        if key is None or not self.rotary_emb_accepts_layer_type:
            value = self.rotary_emb(hidden_states, position_ids=position_ids)
        else:
            value = self.rotary_emb(hidden_states, position_ids, key)
        if not isinstance(value, tuple) or len(value) != 2:
            raise TypeError("rotary embedding must return (cos, sin)")
        cache[key] = value
        return value

    def _clean_attention_bounds(
        self,
        *,
        query_positions: torch.Tensor,
        query_is_clean: torch.Tensor,
        layer_ops: NemotronDecoderLayerOps,
        query_blocks: torch.Tensor | None = None,
        decoder_response_start: int | None = None,
    ) -> torch.Tensor | None:
        """Return exact encoder-context intervals for a shared encoder/decoder layer."""

        if not self.encoder_causal_attention:
            return None
        positions = query_positions.to(dtype=torch.int32)
        clean_query = query_is_clean.to(dtype=torch.bool)
        if decoder_response_start is None:
            decoder_stop = (positions // int(self.block_size)) * int(self.block_size)
        else:
            if query_blocks is None or query_blocks.shape != query_positions.shape:
                raise ValueError(
                    "response-relative clean bounds require query block metadata"
                )
            valid_decoder = query_blocks.ge(0)
            decoder_stop = int(decoder_response_start) + query_blocks.clamp_min(0).to(
                dtype=torch.int32
            ) * int(self.block_size)
            decoder_stop = torch.where(
                valid_decoder,
                decoder_stop,
                torch.zeros_like(decoder_stop),
            )
        clean_stop = torch.where(clean_query, positions + 1, decoder_stop)
        sliding_window = (
            getattr(layer_ops.attention, "sliding_window", None)
            if layer_ops.layer_type == "sliding_attention"
            else None
        )
        if sliding_window is None:
            clean_start = torch.zeros_like(clean_stop)
        else:
            window = int(sliding_window)
            if window <= 0:
                raise ValueError("sliding attention requires a positive window")
            # Encoder queries include themselves in the window. Decoder canvas
            # queries read the encoder cache, which retains window - 1 states.
            history = torch.where(
                clean_query,
                torch.full_like(clean_stop, window),
                torch.full_like(clean_stop, max(window - 1, 0)),
            )
            clean_start = torch.clamp(clean_stop - history, min=0)
        return torch.stack((clean_start, clean_stop), dim=-1)

    def _packed_attention_masks_for_layer(
        self,
        layout: _PackedLayout,
        layer_ops: NemotronDecoderLayerOps,
    ) -> tuple[BlockDenoisingLocalActiveMask, BlockDenoisingGlobalCleanMask]:
        if not self.encoder_causal_attention:
            return layout.local_attn_mask, layout.global_attn_mask
        cache_key = (
            "packed",
            layer_ops.layer_type,
            getattr(layer_ops.attention, "sliding_window", None),
            layout.packed_positions.device.type,
            layout.packed_positions.device.index,
        )
        cached = self._layer_attention_mask_cache.get(cache_key)
        if cached is not None:
            return cached
        bounds = self._clean_attention_bounds(
            query_positions=layout.packed_positions,
            query_is_clean=layout.global_attn_mask.query_is_clean,
            layer_ops=layer_ops,
        )
        masks = (
            replace(layout.local_attn_mask, flex_cache={}),
            replace(
                layout.global_attn_mask,
                clean_context_window=(
                    getattr(layer_ops.attention, "sliding_window", None)
                    if layer_ops.layer_type == "sliding_attention"
                    else None
                ),
                query_clean_bounds=bounds,
                clean_key_positions=layout.clean_positions.to(dtype=torch.int32),
                flex_cache={},
            ),
        )
        self._layer_attention_mask_cache[cache_key] = masks
        return masks

    def _full_attention_mask_for_layer(
        self,
        mask: BlockDenoisingFullMask,
        *,
        query_positions: torch.Tensor,
        layer_ops: NemotronDecoderLayerOps,
        cache_namespace: str,
    ) -> BlockDenoisingFullMask:
        if not self.encoder_causal_attention:
            return mask
        cache_key = (
            cache_namespace,
            layer_ops.layer_type,
            getattr(layer_ops.attention, "sliding_window", None),
            query_positions.device.type,
            query_positions.device.index,
        )
        cached = self._layer_attention_mask_cache.get(cache_key)
        if cached is not None:
            return cached
        bounds = self._clean_attention_bounds(
            query_positions=query_positions,
            query_is_clean=mask.query_is_clean,
            layer_ops=layer_ops,
        )
        exact = replace(mask, query_clean_bounds=bounds, flex_cache={})
        self._layer_attention_mask_cache[cache_key] = exact
        return exact

    def _full_layer_checkpointing_enabled(self) -> bool:
        return (
            self.activation_checkpointing
            and self.training
            and self.activation_checkpointing_scope == "full"
        )

    def _checkpoint_decoder_layer(
        self,
        layer_ops: NemotronDecoderLayerOps,
        *,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor | None, torch.Tensor | None],
        cache_position: torch.Tensor,
        active_len: int,
        local_attn_mask: BlockDenoisingLocalActiveMask,
        global_attn_mask: BlockDenoisingGlobalCleanMask,
    ) -> torch.Tensor:
        forward = lambda states, ops=layer_ops, pe=position_embeddings: (
            self._decoder_layer_forward(
                ops,
                hidden_states=states,
                position_embeddings=pe,
                cache_position=cache_position,
                active_len=active_len,
                local_attn_mask=local_attn_mask,
                global_attn_mask=global_attn_mask,
                checkpoint_mlp=False,
            )
        )
        if self._requires_moe_checkpoint_context:
            return checkpoint(
                forward,
                hidden_states,
                use_reentrant=False,
                context_fn=moe_route_checkpoint_context_fn,
            )
        return checkpoint(
            forward,
            hidden_states,
            use_reentrant=self._checkpoint_uses_reentrant_autograd(),
        )

    def _checkpoint_decoder_layer_sequence_parallel(
        self,
        layer_ops: NemotronDecoderLayerOps,
        *,
        hidden_shard: torch.Tensor,
        meta: _SequenceParallelPackedMeta,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cache_position: torch.Tensor,
        active_len: int,
        local_attn_mask: BlockDenoisingLocalActiveMask,
        global_attn_mask: BlockDenoisingGlobalCleanMask,
    ) -> torch.Tensor:
        forward = lambda states, ops=layer_ops, pe=position_embeddings: (
            self._decoder_layer_forward_sequence_parallel(
                ops,
                hidden_shard=states,
                meta=meta,
                position_embeddings=pe,
                cache_position=cache_position,
                active_len=int(active_len),
                local_attn_mask=local_attn_mask,
                global_attn_mask=global_attn_mask,
                checkpoint_mlp=False,
            )
        )
        if self._requires_moe_checkpoint_context:
            return checkpoint(
                forward,
                hidden_shard,
                use_reentrant=False,
                context_fn=moe_route_checkpoint_context_fn,
            )
        return checkpoint(
            forward,
            hidden_shard,
            use_reentrant=self._checkpoint_uses_reentrant_autograd(),
        )

    def _decoder_layer_forward(
        self,
        layer_ops: NemotronDecoderLayerOps,
        *,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cache_position: torch.Tensor,
        active_len: int,
        local_attn_mask: BlockDenoisingLocalActiveMask,
        global_attn_mask: BlockDenoisingGlobalCleanMask,
        checkpoint_mlp: bool = True,
    ) -> torch.Tensor:
        if layer_ops.layer_style == "gemma4":
            return self._gemma4_decoder_layer_forward(
                layer_ops,
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                cache_position=cache_position,
                active_len=active_len,
                local_attn_mask=local_attn_mask,
                global_attn_mask=global_attn_mask,
                checkpoint_mlp=checkpoint_mlp,
            )
        residual = hidden_states
        hidden_states = layer_ops.input_layernorm(hidden_states)
        attn_out = self._attention_forward(
            layer_ops,
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            cache_position=cache_position,
            active_len=active_len,
            local_attn_mask=local_attn_mask,
            global_attn_mask=global_attn_mask,
        )
        hidden_states = residual + attn_out
        return self._mlp_block_forward(
            layer_ops,
            hidden_states,
            checkpoint_mlp=checkpoint_mlp,
        )

    def _decoder_layer_forward_sequence_parallel(
        self,
        layer_ops: NemotronDecoderLayerOps,
        *,
        hidden_shard: torch.Tensor,
        meta: _SequenceParallelPackedMeta,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cache_position: torch.Tensor,
        active_len: int,
        local_attn_mask: BlockDenoisingLocalActiveMask,
        global_attn_mask: BlockDenoisingGlobalCleanMask,
        checkpoint_mlp: bool = True,
    ) -> torch.Tensor:
        if layer_ops.layer_style == "gemma4":
            return self._gemma4_decoder_layer_forward_sequence_parallel(
                layer_ops,
                hidden_shard=hidden_shard,
                meta=meta,
                position_embeddings=position_embeddings,
                cache_position=cache_position,
                active_len=active_len,
                local_attn_mask=local_attn_mask,
                global_attn_mask=global_attn_mask,
                checkpoint_mlp=checkpoint_mlp,
            )
        residual = hidden_shard
        hidden_shard = layer_ops.input_layernorm(hidden_shard)
        attn_out = self._attention_forward_sequence_parallel(
            layer_ops,
            hidden_shard=hidden_shard,
            meta=meta,
            position_embeddings=position_embeddings,
            cache_position=cache_position,
            active_len=active_len,
            local_attn_mask=local_attn_mask,
            global_attn_mask=global_attn_mask,
        )
        hidden_shard = residual + attn_out
        return self._mlp_block_forward(
            layer_ops,
            hidden_shard,
            checkpoint_mlp=checkpoint_mlp,
        )

    def _gemma4_decoder_layer_forward(
        self,
        layer_ops: NemotronDecoderLayerOps,
        *,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cache_position: torch.Tensor,
        active_len: int,
        local_attn_mask: BlockDenoisingLocalActiveMask,
        global_attn_mask: BlockDenoisingGlobalCleanMask,
        checkpoint_mlp: bool = True,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = layer_ops.input_layernorm(residual)
        self._debug_check_named_tensor(
            hidden_states,
            layer_ops,
            "attention_normalized_input",
        )
        hidden_states = self._attention_forward(
            layer_ops,
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            cache_position=cache_position,
            active_len=active_len,
            local_attn_mask=local_attn_mask,
            global_attn_mask=global_attn_mask,
        )
        self._debug_check_named_tensor(hidden_states, layer_ops, "attention_output")
        hidden_states = layer_ops.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + residual
        self._debug_check_named_tensor(hidden_states, layer_ops, "feed_forward_input")
        return self._gemma4_feed_forward_block(
            layer_ops,
            hidden_states,
            active_len=active_len,
            checkpoint_mlp=checkpoint_mlp,
        )

    def _gemma4_decoder_layer_forward_sequence_parallel(
        self,
        layer_ops: NemotronDecoderLayerOps,
        *,
        hidden_shard: torch.Tensor,
        meta: _SequenceParallelPackedMeta,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cache_position: torch.Tensor,
        active_len: int,
        local_attn_mask: BlockDenoisingLocalActiveMask,
        global_attn_mask: BlockDenoisingGlobalCleanMask,
        checkpoint_mlp: bool = True,
    ) -> torch.Tensor:
        residual = hidden_shard
        te_layer = self._te_packed_by_layer_id.get(id(layer_ops.layer))
        hidden_shard = (
            residual
            if te_layer is not None and te_layer.qkv_fuses_input_norm
            else layer_ops.input_layernorm(residual)
        )
        hidden_shard = self._attention_forward_sequence_parallel(
            layer_ops,
            hidden_shard=hidden_shard,
            meta=meta,
            position_embeddings=position_embeddings,
            cache_position=cache_position,
            active_len=active_len,
            local_attn_mask=local_attn_mask,
            global_attn_mask=global_attn_mask,
        )
        hidden_shard = layer_ops.post_attention_layernorm(hidden_shard)
        hidden_shard = hidden_shard + residual
        return self._gemma4_feed_forward_block(
            layer_ops,
            hidden_shard,
            active_len=active_len,
            sequence_parallel_meta=meta,
            checkpoint_mlp=checkpoint_mlp,
        )

    @traced_operator(_feed_forward_trace_name)
    def _gemma4_feed_forward_block(
        self,
        layer_ops: NemotronDecoderLayerOps,
        hidden_states: torch.Tensor,
        *,
        active_len: int | None = None,
        sequence_parallel_meta: _SequenceParallelPackedMeta | None = None,
        checkpoint_mlp: bool = True,
    ) -> torch.Tensor:
        residual = hidden_states
        pre_ff = layer_ops.pre_feedforward_layernorm
        post_ff = layer_ops.post_feedforward_layernorm
        if pre_ff is None or post_ff is None:
            raise RuntimeError("Gemma4 packed layer requires feed-forward norms")

        moe_module = layer_ops.moe if layer_ops.moe is not None else layer_ops.experts
        if layer_ops.router is not None or moe_module is not None:
            if (
                layer_ops.router is None
                or moe_module is None
                or layer_ops.pre_feedforward_layernorm_2 is None
                or layer_ops.post_feedforward_layernorm_1 is None
                or layer_ops.post_feedforward_layernorm_2 is None
            ):
                raise RuntimeError("Gemma4 MoE layer is missing router/MoE norms")

            def dense_moe_branches(states: torch.Tensor) -> torch.Tensor:
                flat_states = states.reshape(-1, states.shape[-1])
                token_plan = self._clean_expert_token_plan(
                    device=flat_states.device,
                    total_rows=int(flat_states.shape[0]),
                    states_shape=states.shape,
                    active_len=active_len,
                    sequence_parallel_meta=sequence_parallel_meta,
                )
                pending_expert_dispatch = None
                router_output = None
                moe_input = None
                if (
                    token_plan is not None
                    and hasattr(moe_module, "begin_dispatch")
                    and hasattr(moe_module, "finish_dispatch")
                ):
                    pending_expert_dispatch = self._begin_shared_gemma4_moe_dispatch(
                        moe_module,
                        layer_ops,
                        flat_states,
                        token_plan,
                    )
                else:
                    moe_input = layer_ops.pre_feedforward_layernorm_2(flat_states)
                    self._debug_check_named_tensor(
                        moe_input,
                        layer_ops,
                        "moe_input",
                    )
                    router_output = layer_ops.router(flat_states)
                    if isinstance(router_output, tuple) and router_output:
                        router_probabilities = router_output[0]
                        if isinstance(router_probabilities, torch.Tensor):
                            self._debug_check_named_tensor(
                                router_probabilities,
                                layer_ops,
                                "router_probabilities",
                            )
                    pending_expert_dispatch = self._begin_gemma4_moe_dispatch(
                        moe_module,
                        moe_input,
                        router_output,
                    )

                te_layer = self._te_packed_by_layer_id.get(id(layer_ops.layer))
                dense_input = (
                    states
                    if te_layer is not None and te_layer.gate_up_fuses_pre_ff_norm
                    else pre_ff(states)
                )
                self._debug_check_named_tensor(
                    dense_input,
                    layer_ops,
                    "dense_mlp_input",
                )
                dense_out = self._dense_mlp_forward(layer_ops, dense_input)
                self._debug_check_named_tensor(dense_out, layer_ops, "dense_mlp_output")
                dense_out = layer_ops.post_feedforward_layernorm_1(dense_out)
                self._debug_check_named_tensor(
                    dense_out, layer_ops, "dense_branch_output"
                )

                if pending_expert_dispatch is not None:
                    moe_out = self._finish_gemma4_moe_dispatch(
                        moe_module,
                        pending_expert_dispatch,
                    )
                elif isinstance(router_output, tuple) and len(router_output) >= 3:
                    _, top_k_weights, top_k_index = router_output[:3]
                    if moe_input is None:
                        raise RuntimeError("MoE input was not materialized")
                    moe_out = moe_module(moe_input, top_k_index, top_k_weights)
                else:
                    if moe_input is None:
                        raise RuntimeError("MoE input was not materialized")
                    moe_out = moe_module(moe_input, router_output)
                self._debug_check_named_tensor(moe_out, layer_ops, "moe_output")
                moe_out = layer_ops.post_feedforward_layernorm_2(
                    moe_out.reshape(states.shape)
                )
                self._debug_check_named_tensor(moe_out, layer_ops, "moe_branch_output")
                return dense_out + moe_out

            if checkpoint_mlp and self.activation_checkpointing and self.training:
                hidden_states = checkpoint(
                    dense_moe_branches,
                    hidden_states,
                    use_reentrant=False,
                )
            else:
                hidden_states = dense_moe_branches(hidden_states)
        else:

            def dense_branch(states: torch.Tensor) -> torch.Tensor:
                te_layer = self._te_packed_by_layer_id.get(id(layer_ops.layer))
                dense_input = (
                    states
                    if te_layer is not None and te_layer.gate_up_fuses_pre_ff_norm
                    else pre_ff(states)
                )
                return self._dense_mlp_forward(layer_ops, dense_input)

            if checkpoint_mlp and self.activation_checkpointing and self.training:
                hidden_states = checkpoint(
                    dense_branch, hidden_states, use_reentrant=False
                )
            else:
                hidden_states = dense_branch(hidden_states)

        hidden_states = post_ff(hidden_states)
        self._debug_check_named_tensor(hidden_states, layer_ops, "feed_forward_output")
        hidden_states = hidden_states + residual
        layer_scalar = layer_ops.layer_scalar
        if layer_scalar is not None:
            hidden_states = hidden_states * torch.as_tensor(
                layer_scalar,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        return hidden_states

    def _begin_gemma4_moe_dispatch(
        self,
        moe_module: Any,
        moe_input: torch.Tensor,
        router_output: Any,
    ) -> Any | None:
        if (
            not isinstance(router_output, tuple)
            or len(router_output) < 3
            or not hasattr(moe_module, "begin_dispatch")
            or not hasattr(moe_module, "finish_dispatch")
        ):
            return None
        _, top_k_weights, top_k_index = router_output[:3]
        return moe_module.begin_dispatch(moe_input, top_k_index, top_k_weights)

    def _begin_shared_gemma4_moe_dispatch(
        self,
        moe_module: Any,
        layer_ops: NemotronDecoderLayerOps,
        flat_states: torch.Tensor,
        token_plan: _Gemma4ExpertTokenPlan,
    ) -> _Gemma4PendingMoeDispatch | None:
        hidden_size = int(flat_states.shape[-1])
        clean_owner = int(getattr(self.runtime, "clean_replica_rank", 0) or 0) == 0
        route_indices = (
            token_plan.owner_route_indices
            if clean_owner
            else token_plan.nonclean_indices
        )
        if route_indices.numel() == 0:
            pending = None
        else:
            route_states = flat_states.index_select(0, route_indices)
            route_input = layer_ops.pre_feedforward_layernorm_2(route_states)
            router_output = layer_ops.router(route_states)
            if not isinstance(router_output, tuple) or len(router_output) < 3:
                return None
            _, top_k_weights, top_k_index = router_output[:3]
            top_k = int(top_k_index.shape[-1])
            route_index = top_k_index.reshape(-1, top_k)
            route_weight = top_k_weights.reshape(-1, top_k)
            pending = moe_module.begin_dispatch(route_input, route_index, route_weight)
        return _Gemma4PendingMoeDispatch(
            pending=pending,
            total_rows=int(flat_states.shape[0]),
            hidden_size=hidden_size,
            dtype=flat_states.dtype,
            device=flat_states.device,
            input_requires_grad=bool(flat_states.requires_grad),
            nonclean_indices=token_plan.nonclean_indices,
            clean_indices=token_plan.clean_indices,
            clean_owner=clean_owner,
        )

    def _finish_gemma4_moe_dispatch(
        self,
        moe_module: Any,
        pending: Any,
    ) -> torch.Tensor:
        if not isinstance(pending, _Gemma4PendingMoeDispatch):
            return moe_module.finish_dispatch(pending)
        if pending.pending is None:
            output = torch.empty(
                0,
                pending.hidden_size,
                device=pending.device,
                dtype=pending.dtype,
            )
        else:
            output = moe_module.finish_dispatch(pending.pending)
        nonclean_rows = int(pending.nonclean_indices.numel())
        nonclean_out = output[:nonclean_rows]
        if pending.clean_owner:
            clean_out = output[nonclean_rows:]
        else:
            clean_out = output.new_empty(
                int(pending.clean_indices.numel()),
                pending.hidden_size,
            )
            if pending.input_requires_grad:
                clean_out.requires_grad_(True)
        pending_clean_out = begin_clean_replica_tensor(clean_out, self.runtime)
        full = output.new_empty(pending.total_rows, pending.hidden_size)
        full.index_copy_(0, pending.nonclean_indices, nonclean_out)
        clean_out = pending_clean_out.wait()
        full.index_copy_(0, pending.clean_indices, clean_out)
        return full

    def _clean_expert_token_plan(
        self,
        *,
        device: torch.device,
        total_rows: int,
        states_shape: torch.Size,
        active_len: int | None,
        sequence_parallel_meta: _SequenceParallelPackedMeta | None,
    ) -> _Gemma4ExpertTokenPlan | None:
        if active_len is None:
            return None
        if int(getattr(self.runtime, "clean_replica_size", 1) or 1) <= 1:
            return None
        if getattr(self.runtime, "clean_replica_group", None) is None:
            return None
        device_index = device.index
        if device.type == "cuda" and device_index is None:
            device_index = torch.cuda.current_device()
        meta_key: tuple[int, int, int] | None = None
        if sequence_parallel_meta is not None:
            meta_key = (
                int(sequence_parallel_meta.packed_len),
                int(sequence_parallel_meta.total_rows),
                int(getattr(self.runtime, "tensor_parallel_rank", 0) or 0),
            )
        cache_key = (
            int(total_rows),
            tuple(int(dim) for dim in states_shape),
            int(active_len),
            device.type,
            device_index,
            meta_key,
        )
        cached = self._clean_expert_token_plan_cache.get(cache_key)
        if cached is not None:
            return cached
        if len(states_shape) == 3:
            packed_len = int(states_shape[1])
            row_ids = torch.arange(
                total_rows,
                device=device,
                dtype=torch.long,
            )
            valid = torch.ones(total_rows, device=device, dtype=torch.bool)
        elif len(states_shape) == 2 and sequence_parallel_meta is not None:
            packed_len = int(sequence_parallel_meta.packed_len)
            tp_rank = int(getattr(self.runtime, "tensor_parallel_rank", 0) or 0)
            row_start = tp_rank * total_rows
            row_ids = torch.arange(
                row_start,
                row_start + total_rows,
                device=device,
                dtype=torch.long,
            )
            valid = row_ids < int(sequence_parallel_meta.total_rows)
        else:
            return None
        is_clean = valid & ((row_ids % packed_len) >= int(active_len))
        clean_indices = torch.nonzero(is_clean, as_tuple=False).flatten()
        if clean_indices.numel() == 0:
            return None
        nonclean_indices = torch.nonzero(~is_clean, as_tuple=False).flatten()
        nonclean_indices = nonclean_indices.contiguous()
        clean_indices = clean_indices.contiguous()
        token_plan = _Gemma4ExpertTokenPlan(
            nonclean_indices=nonclean_indices,
            clean_indices=clean_indices,
            owner_route_indices=torch.cat((nonclean_indices, clean_indices), dim=0),
        )
        self._clean_expert_token_plan_cache[cache_key] = token_plan
        return token_plan

    def _dense_mlp_forward(
        self,
        layer_ops: NemotronDecoderLayerOps,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        te_layer = self._te_packed_by_layer_id.get(id(layer_ops.layer))
        if te_layer is not None:
            return _te_gated_mlp_forward(
                te_layer,
                hidden_states,
                token_chunk_size=self.mlp_token_chunk_size,
            )
        token_chunk_size = (
            self.mlp_token_chunk_size
            if not self._full_layer_checkpointing_enabled()
            else 0
        )
        if token_chunk_size > 0:
            return _hf_gated_mlp_forward(
                layer_ops.mlp,
                hidden_states,
                token_chunk_size=token_chunk_size,
            )
        return layer_ops.mlp(hidden_states)

    @traced_operator(_feed_forward_trace_name)
    def _mlp_block_forward(
        self,
        layer_ops: NemotronDecoderLayerOps,
        hidden_states: torch.Tensor,
        *,
        checkpoint_mlp: bool = True,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = layer_ops.post_attention_layernorm(hidden_states)
        te_layer = self._te_packed_by_layer_id.get(id(layer_ops.layer))
        if te_layer is not None:
            if checkpoint_mlp and self.activation_checkpointing and self.training:
                hidden_states = checkpoint(
                    lambda states, packed=te_layer: _te_gated_mlp_forward(
                        packed,
                        states,
                        token_chunk_size=self.mlp_token_chunk_size,
                    ),
                    hidden_states,
                    use_reentrant=False,
                )
            else:
                hidden_states = _te_gated_mlp_forward(
                    te_layer,
                    hidden_states,
                    token_chunk_size=self.mlp_token_chunk_size,
                )
        else:
            self._require_te_packed_tensor_parallel(layer_ops, "MLP")
            hidden_states = self._copy_to_tensor_parallel_input(hidden_states)
            token_chunk_size = (
                self.mlp_token_chunk_size
                if not self._full_layer_checkpointing_enabled()
                else 0
            )
            if checkpoint_mlp and self.activation_checkpointing and self.training:
                if token_chunk_size > 0:
                    hidden_states = checkpoint(
                        lambda states, ops=layer_ops: _hf_gated_mlp_forward(
                            ops.mlp,
                            states,
                            token_chunk_size=token_chunk_size,
                        ),
                        hidden_states,
                        use_reentrant=False,
                    )
                else:
                    hidden_states = checkpoint(
                        lambda states, ops=layer_ops: ops.mlp(states),
                        hidden_states,
                        use_reentrant=False,
                    )
            else:
                if token_chunk_size > 0:
                    hidden_states = _hf_gated_mlp_forward(
                        layer_ops.mlp,
                        hidden_states,
                        token_chunk_size=token_chunk_size,
                    )
                else:
                    hidden_states = layer_ops.mlp(hidden_states)
            hidden_states = self._sum_tensor_parallel_output(hidden_states)
        return residual + hidden_states

    @traced_operator(_attention_trace_name)
    def _attention_forward(
        self,
        layer_ops: NemotronDecoderLayerOps,
        *,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cache_position: torch.Tensor,
        active_len: int,
        local_attn_mask: BlockDenoisingLocalActiveMask,
        global_attn_mask: BlockDenoisingGlobalCleanMask,
    ) -> torch.Tensor:
        attn = layer_ops.attention
        input_shape = hidden_states.shape[:-1]
        query_states, key_states, value_states, output_gate = self._project_qkv_bshd(
            layer_ops,
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            cache_position=cache_position,
        )
        self._debug_check_named_tensor(query_states, layer_ops, "attention_query")
        self._debug_check_named_tensor(key_states, layer_ops, "attention_key")
        self._debug_check_named_tensor(value_states, layer_ops, "attention_value")
        if self.runtime.kv_backend == "ring":
            attn_output = fused_block_context_attention_bshd(
                query=query_states,
                local_key=key_states[:, :active_len],
                local_value=value_states[:, :active_len],
                global_key_shard=key_states[:, active_len:],
                global_value_shard=value_states[:, active_len:],
                global_seq_len=self.seq_len,
                local_attn_mask=local_attn_mask,
                global_attn_mask=global_attn_mask,
                scale=attn.scale,
                runtime=self.runtime,
                key_chunk_size=self.ring_attention_key_chunk_size,
            )
        else:
            attn_output = replicated_block_denoising_attention_bshd(
                query=query_states,
                local_key=key_states[:, :active_len],
                local_value=value_states[:, :active_len],
                global_key=key_states[:, active_len:],
                global_value=value_states[:, active_len:],
                local_attn_mask=local_attn_mask,
                global_attn_mask=global_attn_mask,
                scale=attn.scale,
            )
        self._debug_check_named_tensor(
            attn_output,
            layer_ops,
            "attention_core_output",
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        if output_gate is not None:
            attn_output = attn_output * torch.sigmoid(
                output_gate.reshape(*input_shape, -1)
            )
        self._debug_check_named_tensor(
            attn_output,
            layer_ops,
            "attention_projection_input",
        )
        return self._attention_output_projection(layer_ops, attn_output)

    @traced_operator(_attention_trace_name)
    def _attention_forward_sequence_parallel(
        self,
        layer_ops: NemotronDecoderLayerOps,
        *,
        hidden_shard: torch.Tensor,
        meta: _SequenceParallelPackedMeta,
        position_embeddings: tuple[torch.Tensor | None, torch.Tensor | None],
        cache_position: torch.Tensor,
        active_len: int,
        local_attn_mask: BlockDenoisingLocalActiveMask,
        global_attn_mask: BlockDenoisingGlobalCleanMask,
    ) -> torch.Tensor:
        attn = layer_ops.attention
        query_states, key_states, value_states, output_gate = (
            self._project_qkv_bshd_sequence_parallel(
                layer_ops,
                hidden_shard=hidden_shard,
                meta=meta,
                position_embeddings=position_embeddings,
                cache_position=cache_position,
            )
        )
        if self.runtime.kv_backend == "ring":
            attn_output = fused_block_context_attention_bshd(
                query=query_states,
                local_key=key_states[:, :active_len],
                local_value=value_states[:, :active_len],
                global_key_shard=key_states[:, active_len:],
                global_value_shard=value_states[:, active_len:],
                global_seq_len=self.seq_len,
                local_attn_mask=local_attn_mask,
                global_attn_mask=global_attn_mask,
                scale=attn.scale,
                runtime=self.runtime,
                key_chunk_size=self.ring_attention_key_chunk_size,
            )
        else:
            attn_output = replicated_block_denoising_attention_bshd(
                query=query_states,
                local_key=key_states[:, :active_len],
                local_value=value_states[:, :active_len],
                global_key=key_states[:, active_len:],
                global_value=value_states[:, active_len:],
                local_attn_mask=local_attn_mask,
                global_attn_mask=global_attn_mask,
                scale=attn.scale,
            )
        attn_output = attn_output.reshape(meta.total_rows, -1).contiguous()
        if output_gate is not None:
            attn_output = attn_output * torch.sigmoid(
                output_gate.reshape(meta.total_rows, -1)
            )
        attn_output = self._pad_rows(attn_output, meta.padded_rows)
        return self._attention_output_projection(layer_ops, attn_output)

    def _project_qkv_bshd(
        self,
        layer_ops: NemotronDecoderLayerOps,
        *,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cache_position: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        attn = layer_ops.attention
        input_shape = hidden_states.shape[:-1]
        query_shape = (*input_shape, -1, attn.head_dim)
        te_layer = self._te_packed_by_layer_id.get(id(layer_ops.layer))
        if te_layer is not None:
            packed = te_layer.qkv(hidden_states)
            pieces = packed.split(te_layer.qkv_local_sizes, dim=-1)
            if te_layer.qkv_value_from_key:
                query_states, key_states = pieces[:2]
                value_states = key_states
                output_gate = pieces[2] if len(pieces) == 3 else None
            else:
                query_states, key_states, value_states = pieces[:3]
                output_gate = pieces[3] if len(pieces) == 4 else None
        else:
            self._require_te_packed_tensor_parallel(layer_ops, "QKV projection")
            q_weight, q_bias = _linear_weight_bias(attn.q_proj, "q_proj")
            k_weight, k_bias = _linear_weight_bias(attn.k_proj, "k_proj")
            if attn.value_from_key:
                v_weight, v_bias = k_weight, k_bias
            else:
                v_weight, v_bias = _linear_weight_bias(attn.v_proj, "v_proj")
            from dllm_parallel.core.parallel.tensor_parallel import (
                fused_qkv_column_parallel_linear,
            )

            query_states, key_states, value_states = fused_qkv_column_parallel_linear(
                hidden_states,
                q_weight,
                k_weight,
                v_weight,
                q_bias,
                k_bias,
                v_bias,
                self.runtime,
            )
            output_gate = (
                layer_ops.attention.output_gate_proj(hidden_states)
                if layer_ops.attention.output_gate_proj is not None
                else None
            )
        query_states = query_states.view(query_shape)
        key_states = key_states.view(query_shape)
        value_states = value_states.view(query_shape)
        query_states, key_states, value_states = _apply_qkv_norms(
            attn,
            query_states,
            key_states,
            value_states,
        )
        cos, sin = position_embeddings
        if cos is not None and sin is not None:
            query_states, key_states = attn.apply_rotary(
                query_states,
                key_states,
                cos,
                sin,
                cache_position,
            )
        rope_scale = attn.rope_scale(cache_position)
        if rope_scale is not None:
            query_states = query_states * _rope_scale_bshd(
                rope_scale,
                seq_len=int(query_states.shape[1]),
                dtype=query_states.dtype,
            )
        return (
            query_states.contiguous(),
            key_states.contiguous(),
            value_states.contiguous(),
            None if output_gate is None else output_gate.contiguous(),
        )

    def _project_qkv_bshd_sequence_parallel(
        self,
        layer_ops: NemotronDecoderLayerOps,
        *,
        hidden_shard: torch.Tensor,
        meta: _SequenceParallelPackedMeta,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cache_position: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        attn = layer_ops.attention
        te_layer = self._te_packed_by_layer_id.get(id(layer_ops.layer))
        if te_layer is None:
            self._require_te_packed_tensor_parallel(layer_ops, "QKV projection")
            raise RuntimeError("unreachable")
        packed = te_layer.qkv(hidden_shard)[: meta.total_rows]
        input_shape = (meta.batch_size, meta.packed_len)
        query_shape = (*input_shape, -1, attn.head_dim)
        pieces = packed.split(te_layer.qkv_local_sizes, dim=-1)
        if te_layer.qkv_value_from_key:
            query_states, key_states = pieces[:2]
            value_states = key_states
            output_gate = pieces[2] if len(pieces) == 3 else None
        else:
            query_states, key_states, value_states = pieces[:3]
            output_gate = pieces[3] if len(pieces) == 4 else None
        query_states = query_states.view(query_shape)
        key_states = key_states.view(query_shape)
        value_states = value_states.view(query_shape)
        query_states, key_states, value_states = _apply_qkv_norms(
            attn,
            query_states,
            key_states,
            value_states,
        )
        cos, sin = position_embeddings
        if cos is not None and sin is not None:
            query_states, key_states = attn.apply_rotary(
                query_states,
                key_states,
                cos,
                sin,
                cache_position,
            )
        rope_scale = attn.rope_scale(cache_position)
        if rope_scale is not None:
            query_states = query_states * _rope_scale_bshd(
                rope_scale,
                seq_len=int(query_states.shape[1]),
                dtype=query_states.dtype,
            )
        return (
            query_states.contiguous(),
            key_states.contiguous(),
            value_states.contiguous(),
            None if output_gate is None else output_gate.contiguous(),
        )

    def _attention_output_projection(
        self,
        layer_ops: NemotronDecoderLayerOps,
        value: torch.Tensor,
    ) -> torch.Tensor:
        te_layer = self._te_packed_by_layer_id.get(id(layer_ops.layer))
        if te_layer is not None:
            return te_layer.o_proj(value)
        self._require_te_packed_tensor_parallel(
            layer_ops, "attention output projection"
        )
        return self._sum_tensor_parallel_output(layer_ops.attention.o_proj(value))

    def _require_te_packed_tensor_parallel(
        self,
        layer_ops: NemotronDecoderLayerOps,
        path: str,
    ) -> None:
        del layer_ops
        if int(getattr(self.runtime, "tensor_parallel_size", 1) or 1) > 1:
            raise RuntimeError(
                f"Nemotron TP>1 requires TE packed projections for {path}; "
                "optimized execution does not support Python-autograd TP projections"
            )

    def _sum_tensor_parallel_output(self, value: torch.Tensor) -> torch.Tensor:
        return reduce_from_tensor_parallel_region(
            value,
            self.runtime,
        )

    def _copy_to_tensor_parallel_input(self, value: torch.Tensor) -> torch.Tensor:
        return copy_to_tensor_parallel_region(
            value,
            self.runtime,
        )

    def _sequence_parallel_meta(
        self,
        hidden_states: torch.Tensor,
        *,
        active_len: int,
    ) -> _SequenceParallelPackedMeta:
        if hidden_states.ndim != 3:
            raise RuntimeError("packed hidden states must be [batch, tokens, hidden]")
        batch_size, packed_len, hidden_size = hidden_states.shape
        total_rows = int(batch_size) * int(packed_len)
        tp_size = int(getattr(self.runtime, "tensor_parallel_size", 1) or 1)
        padded_rows = ((total_rows + tp_size - 1) // tp_size) * tp_size
        return _SequenceParallelPackedMeta(
            batch_size=int(batch_size),
            packed_len=int(packed_len),
            hidden_size=int(hidden_size),
            total_rows=int(total_rows),
            padded_rows=int(padded_rows),
            active_len=int(active_len),
        )

    def _scatter_packed_hidden_to_sequence_parallel(
        self,
        hidden_states: torch.Tensor,
        meta: _SequenceParallelPackedMeta,
    ) -> torch.Tensor:
        flat = hidden_states.reshape(meta.total_rows, meta.hidden_size).contiguous()
        flat = self._pad_rows(flat, meta.padded_rows)
        return scatter_to_sequence_parallel_region(flat, self.runtime)

    def _native_sequence_parallel_meta(
        self,
        hidden_states: torch.Tensor,
    ) -> _SequenceParallelPackedMeta:
        """Describe one native stream's independent sequence-row layout."""

        return self._sequence_parallel_meta(
            hidden_states,
            active_len=int(hidden_states.shape[1]),
        )

    def _native_scatter_stream(
        self,
        hidden_states: torch.Tensor,
        meta: _SequenceParallelPackedMeta,
    ) -> torch.Tensor:
        flat = hidden_states.reshape(meta.total_rows, meta.hidden_size).contiguous()
        flat = self._pad_rows(flat, meta.padded_rows)
        return scatter_to_sequence_parallel_region(flat, self.runtime)

    def _native_gather_stream(
        self,
        hidden_shard: torch.Tensor,
        meta: _SequenceParallelPackedMeta,
        *,
        reduce_scatter_grad: bool,
    ) -> torch.Tensor:
        hidden_states = gather_from_sequence_parallel_region(
            hidden_shard,
            self.runtime,
            total_rows=meta.total_rows,
            reduce_scatter_grad=bool(reduce_scatter_grad),
        )
        return hidden_states.reshape(
            meta.batch_size,
            meta.packed_len,
            meta.hidden_size,
        )

    @staticmethod
    def _pad_rows(tensor: torch.Tensor, padded_rows: int) -> torch.Tensor:
        if tensor.shape[0] == int(padded_rows):
            return tensor
        if tensor.shape[0] > int(padded_rows):
            raise ValueError("tensor has more rows than padded_rows")
        pad_shape = (int(padded_rows) - tensor.shape[0], *tensor.shape[1:])
        return torch.cat((tensor, tensor.new_zeros(pad_shape)), dim=0)

    def _packed_layout(self, device: torch.device) -> _PackedLayout:
        device_index = device.index
        if device.type == "cuda" and device_index is None:
            device_index = torch.cuda.current_device()
        cache_key = (
            int(self.seq_len),
            int(self.block_size),
            str(getattr(self.runtime, "kv_backend", "")),
            int(getattr(self.runtime, "context_attention_size", 1) or 1),
            int(getattr(self.runtime, "configured_context_parallel_size", 1) or 1),
            int(getattr(self.runtime, "context_parallel_rank", 0) or 0),
            int(getattr(self.runtime, "block_parallel_size", 1) or 1),
            int(getattr(self.runtime, "block_parallel_rank", 0) or 0),
            bool(self._debug_nonfinite_attention_enabled()),
            device.type,
            device_index,
        )
        cached = self._packed_layout_cache.get(cache_key)
        if cached is not None and cached.packed_positions.device == device:
            return cached
        if self.runtime.kv_backend == "replicated":
            active_positions = active_token_indices(
                seq_len=self.seq_len,
                block_size=self.block_size,
                device=device,
                runtime=self.runtime,
            )
            if active_positions is None:
                active_positions = torch.arange(
                    self.seq_len,
                    device=device,
                    dtype=torch.long,
                )
            elif active_positions.numel() == 0:
                raise RuntimeError("packed BP requires non-empty active ownership")
            clean_positions = self._replicated_clean_positions(device)
        else:
            active_positions = active_token_indices(
                seq_len=self.seq_len,
                block_size=self.block_size,
                device=device,
                runtime=self.runtime,
            )
            if active_positions is None:
                active_positions = torch.arange(
                    self.seq_len,
                    device=device,
                    dtype=torch.long,
                )
            elif active_positions.numel() == 0:
                raise RuntimeError("packed BP/CP requires non-empty active ownership")
            clean_positions = self._local_clean_positions(device)
        packed_positions = torch.cat((active_positions, clean_positions), dim=0)
        local_attn_mask, global_attn_mask = self._packed_masks(
            active_positions=active_positions,
            clean_positions=clean_positions,
        )
        layout = _PackedLayout(
            active_positions=active_positions,
            clean_positions=clean_positions,
            packed_positions=packed_positions,
            local_attn_mask=local_attn_mask,
            global_attn_mask=global_attn_mask,
            active_len=int(active_positions.numel()),
        )
        self._packed_layout_cache[cache_key] = layout
        return layout

    def _replicated_clean_positions(self, device: torch.device) -> torch.Tensor:
        prefix_len = active_clean_prefix_length(
            seq_len=self.seq_len,
            block_size=self.block_size,
            runtime=self.runtime,
        )
        if prefix_len is None:
            prefix_len = max(0, int(self.seq_len) - int(self.block_size))
        return torch.arange(prefix_len, device=device, dtype=torch.long)

    def _local_clean_positions(self, device: torch.device) -> torch.Tensor:
        clean_layout = runtime_clean_layout(self.runtime)
        shards = clean_shards_for_rank(
            seq_len=self.seq_len,
            context_parallel_size=int(self.runtime.context_attention_size),
            rank=int(self.runtime.context_parallel_rank),
            layout=clean_layout,
        )
        if not shards:
            return torch.empty(0, dtype=torch.long, device=device)
        return torch.cat(
            [
                torch.arange(
                    shard.start,
                    shard.stop,
                    device=device,
                    dtype=torch.long,
                )
                for shard in shards
            ],
            dim=0,
        )

    def _packed_masks(
        self,
        *,
        active_positions: torch.Tensor,
        clean_positions: torch.Tensor,
    ) -> tuple[BlockDenoisingLocalActiveMask, BlockDenoisingGlobalCleanMask]:
        query_positions = torch.cat((active_positions, clean_positions), dim=0)
        query_blocks = (query_positions // self.block_size).to(torch.int32)
        query_is_clean = torch.cat(
            (
                torch.zeros_like(active_positions, dtype=torch.bool),
                torch.ones_like(clean_positions, dtype=torch.bool),
            ),
            dim=0,
        )
        active_blocks = (active_positions // self.block_size).to(torch.int32)
        backward_query_chunk_size = self._cp_bp_backward_query_chunk_size()
        debug_nonfinite_attention = self._debug_nonfinite_attention_enabled()
        return (
            BlockDenoisingLocalActiveMask(
                query_blocks=query_blocks,
                query_is_clean=query_is_clean,
                active_blocks=active_blocks,
                block_size=self.block_size,
                backward_query_chunk_size=backward_query_chunk_size,
                debug_nonfinite_attention=debug_nonfinite_attention,
            ),
            BlockDenoisingGlobalCleanMask(
                query_blocks=query_blocks,
                query_is_clean=query_is_clean,
                block_size=self.block_size,
                backward_query_chunk_size=backward_query_chunk_size,
                debug_nonfinite_attention=debug_nonfinite_attention,
            ),
        )

    def _cp_bp_backward_query_chunk_size(self) -> int:
        return 0

    def _debug_nonfinite_attention_enabled(self) -> bool:
        policy = getattr(self.runtime, "cp_bp_policy", None)
        return bool(getattr(policy, "debug_nonfinite_attention", False))

    def _model_dtype(self) -> torch.dtype:
        for layer_ops in self.layers:
            weight = getattr(layer_ops.attention.q_proj, "weight", None)
            if isinstance(weight, torch.Tensor):
                return weight.dtype
        return self.embed_tokens.weight.dtype

    def _debug_check_layer_tensor(
        self,
        hidden_states: torch.Tensor,
        layer_index: int,
    ) -> None:
        policy = getattr(self.runtime, "cp_bp_policy", None)
        if not bool(getattr(policy, "debug_nonfinite_attention", False)):
            return
        if not torch.isfinite(hidden_states).all():
            raise RuntimeError(
                f"nonfinite forward hidden states after layer {layer_index}"
            )

        def check_grad(
            grad: torch.Tensor,
            *,
            index: int = layer_index,
        ) -> torch.Tensor:
            if not torch.isfinite(grad).all():
                finite = torch.isfinite(grad)
                finite_values = grad[finite]
                max_abs = (
                    float(finite_values.detach().abs().max().cpu())
                    if finite_values.numel() > 0
                    else float("nan")
                )
                raise RuntimeError(
                    f"nonfinite backward hidden grad after layer {index}: "
                    f"nan={int(torch.isnan(grad).sum().item())} "
                    f"posinf={int(torch.isposinf(grad).sum().item())} "
                    f"neginf={int(torch.isneginf(grad).sum().item())} "
                    f"max_abs_finite={max_abs} shape={tuple(grad.shape)}"
                )
            return grad

        hidden_states.register_hook(check_grad)

    def _debug_check_named_tensor(
        self,
        tensor: torch.Tensor,
        layer_ops: NemotronDecoderLayerOps,
        name: str,
    ) -> None:
        if not self._debug_nonfinite_attention_enabled():
            return
        layer_index = int(getattr(layer_ops.layer, "layer_idx", -1))
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"nonfinite forward {name} in layer {layer_index}")

        def check_grad(gradient: torch.Tensor) -> torch.Tensor:
            finite = torch.isfinite(gradient)
            if not bool(finite.all()):
                finite_values = gradient[finite]
                max_abs = (
                    float(finite_values.detach().abs().max().cpu())
                    if finite_values.numel() > 0
                    else float("nan")
                )
                raise RuntimeError(
                    f"nonfinite backward {name} grad in layer {layer_index}: "
                    f"nan={int(torch.isnan(gradient).sum().item())} "
                    f"posinf={int(torch.isposinf(gradient).sum().item())} "
                    f"neginf={int(torch.isneginf(gradient).sum().item())} "
                    f"max_abs_finite={max_abs} shape={tuple(gradient.shape)}"
                )
            if layer_index + 1 == self.num_hidden_layers:
                print(
                    json.dumps(
                        {
                            "event": "finite_final_layer_gradient",
                            "layer": layer_index,
                            "name": name,
                            "max_abs": float(gradient.detach().abs().max().cpu()),
                            "shape": list(gradient.shape),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            return gradient

        tensor.register_hook(check_grad)


def build_packed_block_diffusion_model(
    model: nn.Module,
    *,
    runtime: Any,
    seq_len: int,
    block_size: int,
    ring_attention_key_chunk_size: int = 0,
    activation_checkpointing: bool = True,
    activation_checkpointing_scope: str = "full",
    mlp_token_chunk_size: int = 0,
    self_condition_clean_tokens: bool = True,
    encoder_causal_attention: bool = False,
) -> NemotronLabsDiffusionPackedBlockDiffusionModel:
    if (
        str(getattr(runtime, "kv_backend", "")) == "ring"
        and int(getattr(runtime, "context_attention_size", 1) or 1) > 1
        and int(getattr(runtime, "block_parallel_size", 1) or 1) == 1
    ):
        from dllm_parallel.core.models.backbones.nemotron.context_parallel import (
            build_context_parallel_model,
        )

        return build_context_parallel_model(
            model,
            runtime=runtime,
            seq_len=int(seq_len),
            block_size=int(block_size),
            ring_attention_key_chunk_size=int(ring_attention_key_chunk_size),
            activation_checkpointing=bool(activation_checkpointing),
            activation_checkpointing_scope=str(activation_checkpointing_scope),
            mlp_token_chunk_size=int(mlp_token_chunk_size),
            self_condition_clean_tokens=bool(self_condition_clean_tokens),
            encoder_causal_attention=bool(encoder_causal_attention),
        )
    return NemotronLabsDiffusionPackedBlockDiffusionModel(
        model,
        runtime=runtime,
        seq_len=int(seq_len),
        block_size=int(block_size),
        ring_attention_key_chunk_size=int(ring_attention_key_chunk_size),
        activation_checkpointing=bool(activation_checkpointing),
        activation_checkpointing_scope=str(activation_checkpointing_scope),
        mlp_token_chunk_size=int(mlp_token_chunk_size),
        self_condition_clean_tokens=bool(self_condition_clean_tokens),
        encoder_causal_attention=bool(encoder_causal_attention),
    )


def _config_to_dict(config: Any) -> dict[str, Any]:
    if isinstance(config, dict):
        return dict(config)
    if hasattr(config, "to_dict"):
        value = config.to_dict()
        if isinstance(value, dict):
            return value
    raise TypeError("Nemotron config must be a dict or expose to_dict()")


def _dtensor_local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    to_local = getattr(tensor, "to_local", None)
    if to_local is None:
        return tensor
    return to_local()


def _gradient_reduce_group_for_parameter(
    parameter: torch.Tensor,
    *,
    runtime: Any,
    shard_group: Any,
    shard_group_size: int,
    replicated_group: Any,
    replicated_group_size: int,
) -> tuple[Any, int, bool]:
    if bool(getattr(parameter, "_dllm_tensor_parallel_sharded", False)):
        return shard_group, int(shard_group_size), True
    if _dtensor_has_sharded_placement(parameter):
        return shard_group, int(shard_group_size), True
    if int(getattr(runtime, "tensor_parallel_size", 1) or 1) > 1:
        return replicated_group, int(replicated_group_size), False
    return shard_group, int(shard_group_size), True


def _bucket_size_bytes(entries: list[tuple[Any, Any]]) -> int:
    return sum(tensor.numel() * tensor.element_size() for _, tensor in entries)


def _foreach_div_(tensors: list[torch.Tensor], value: float) -> None:
    if not tensors:
        return
    try:
        torch._foreach_div_(tensors, value)
    except RuntimeError:
        for tensor in tensors:
            tensor.div_(value)


def _dtensor_has_sharded_placement(tensor: torch.Tensor) -> bool:
    placements = getattr(tensor, "placements", None)
    if placements is None:
        spec = getattr(tensor, "_spec", None)
        placements = getattr(spec, "placements", None)
    if placements is None:
        return False
    return any(type(placement).__name__ == "Shard" for placement in placements)


def _build_te_packed_layer_projections(
    layer_ops: NemotronDecoderLayerOps,
    runtime: Any,
) -> _TEPackedLayerProjections:
    te_linear = _transformer_engine_linear_cls()
    sequence_parallel = bool(getattr(runtime, "sequence_parallel", False))
    dtype = _first_linear_dtype(
        layer_ops.attention.q_proj,
        layer_ops.attention.o_proj,
        layer_ops.mlp,
    )
    device = _first_linear_device(
        layer_ops.attention.q_proj,
        layer_ops.attention.o_proj,
        layer_ops.mlp,
    )

    qkv_inputs = [
        ("query", layer_ops.attention.q_proj),
        ("key", layer_ops.attention.k_proj),
    ]
    if not layer_ops.attention.value_from_key:
        qkv_inputs.append(("value", layer_ops.attention.v_proj))
    if layer_ops.attention.output_gate_proj is not None:
        qkv_inputs.append(("attention_gate", layer_ops.attention.output_gate_proj))
    fuse_gemma4_norms = sequence_parallel and layer_ops.layer_style == "gemma4"
    if fuse_gemma4_norms:
        qkv = _build_te_rmsnorm_column_linear(
            layer_ops.input_layernorm,
            *qkv_inputs,
            runtime=runtime,
            dtype=dtype,
            device=device,
            name="nemotron_qkv",
        )
    else:
        qkv = _build_te_fused_column_linear(
            te_linear,
            *qkv_inputs,
            runtime=runtime,
            dtype=dtype,
            device=device,
            sequence_parallel=sequence_parallel,
            name="nemotron_qkv",
        )
    o_proj = _build_te_row_linear(
        te_linear,
        layer_ops.attention.o_proj,
        runtime=runtime,
        dtype=dtype,
        device=device,
        sequence_parallel=sequence_parallel,
        name="nemotron_o_proj",
    )

    gate_proj = _first_existing_module(
        layer_ops.mlp,
        ("gate_proj", "gate", "w1"),
        "Nemotron MLP gate projection",
    )
    up_proj = _first_existing_module(
        layer_ops.mlp,
        ("up_proj", "up", "w3"),
        "Nemotron MLP up projection",
    )
    down_proj_src = _first_existing_module(
        layer_ops.mlp,
        ("down_proj", "down", "w2"),
        "Nemotron MLP down projection",
    )
    if fuse_gemma4_norms:
        pre_ff = layer_ops.pre_feedforward_layernorm
        if pre_ff is None:
            raise RuntimeError(
                "Gemma4 TP/SP dense MLP requires pre-feedforward RMSNorm"
            )
        gate_up = _build_te_rmsnorm_column_linear(
            pre_ff,
            ("gate", gate_proj),
            ("up", up_proj),
            runtime=runtime,
            dtype=dtype,
            device=device,
            name="nemotron_gate_up",
        )
    else:
        gate_up = _build_te_fused_column_linear(
            te_linear,
            ("gate", gate_proj),
            ("up", up_proj),
            runtime=runtime,
            dtype=dtype,
            device=device,
            sequence_parallel=sequence_parallel,
            name="nemotron_gate_up",
        )
    down_proj = _build_te_row_linear(
        te_linear,
        down_proj_src,
        runtime=runtime,
        dtype=dtype,
        device=device,
        sequence_parallel=sequence_parallel,
        name="nemotron_down_proj",
    )
    activation = _mlp_activation(layer_ops.mlp)
    return _TEPackedLayerProjections(
        qkv=qkv,
        qkv_local_sizes=_local_split_sizes(qkv, tuple(name for name, _ in qkv_inputs)),
        o_proj=o_proj,
        gate_up=gate_up,
        gate_up_local_sizes=_local_split_sizes(gate_up, ("gate", "up")),
        gated_activation=_build_te_gated_activation(activation),
        down_proj=down_proj,
        activation=activation,
        qkv_value_from_key=layer_ops.attention.value_from_key,
        qkv_fuses_input_norm=fuse_gemma4_norms,
        gate_up_fuses_pre_ff_norm=fuse_gemma4_norms,
    )


def _transformer_engine_linear_cls() -> type[nn.Module]:
    try:
        from transformer_engine.pytorch import Linear as TELinear
    except Exception as exc:  # pragma: no cover - depends on runtime image.
        raise RuntimeError(
            "Nemotron TP packed training requires transformer_engine.pytorch.Linear"
        ) from exc
    return TELinear


def _build_te_rmsnorm_column_linear(
    norm: nn.Module,
    *named_modules: tuple[str, Any],
    runtime: Any,
    dtype: torch.dtype,
    device: torch.device,
    name: str,
) -> nn.Module:
    """Fuse a Gemma4 RMSNorm with its packed TP/SP column projection."""

    try:
        from transformer_engine.pytorch import LayerNormLinear
    except Exception as exc:  # pragma: no cover - depends on production image.
        raise RuntimeError(
            "Gemma4 TP/SP training requires transformer_engine.pytorch.LayerNormLinear"
        ) from exc

    if any(getattr(module, "bias", None) is not None for _, module in named_modules):
        raise RuntimeError("Gemma4 fused TP/SP RMSNorm projections require no bias")
    norm_weight = getattr(norm, "weight", None)
    if not isinstance(norm_weight, torch.Tensor):
        raise RuntimeError("Gemma4 fused TP/SP projection requires RMSNorm.weight")

    tp_size = int(getattr(runtime, "tensor_parallel_size", 1) or 1)
    tp_group = getattr(runtime, "tensor_parallel_group", None)
    local_weights = tuple(
        _dtensor_local_tensor(module.weight) for _, module in named_modules
    )
    hidden_size = int(local_weights[0].shape[1])
    global_split_sizes = tuple(
        _column_linear_out_features(module, tp_size=tp_size)
        for _, module in named_modules
    )
    local_split_sizes = tuple(int(weight.shape[0]) for weight in local_weights)
    eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6)))
    fused = LayerNormLinear(
        hidden_size,
        sum(global_split_sizes),
        eps=eps,
        sequence_parallel=True,
        tp_group=tp_group,
        tp_size=tp_size,
        bias=False,
        normalization="RMSNorm",
        return_bias=False,
        params_dtype=dtype,
        parallel_mode="column",
        device=device,
        name=name,
    )
    _copy_te_parameter(
        fused.layer_norm_weight,
        _dtensor_local_tensor(norm_weight),
        tensor_parallel_sharded=False,
        sequence_parallel_replicated=True,
    )
    _copy_te_parameter(
        fused.weight,
        torch.cat(local_weights, dim=0),
        tensor_parallel_sharded=True,
    )
    fused._dllm_local_split_sizes = local_split_sizes
    return fused


def _build_te_fused_column_linear(
    te_linear: type[nn.Module],
    *named_modules: tuple[str, Any],
    runtime: Any,
    dtype: torch.dtype,
    device: torch.device,
    sequence_parallel: bool,
    name: str,
    ub_overlap_ag: bool = False,
    ub_name: str | None = None,
) -> nn.Module:
    tp_size = int(getattr(runtime, "tensor_parallel_size", 1) or 1)
    tp_group = getattr(runtime, "tensor_parallel_group", None)
    in_features = _column_linear_in_features(named_modules[0][1])
    global_split_sizes = OrderedDict(
        (split_name, _column_linear_out_features(module, tp_size=tp_size))
        for split_name, module in named_modules
    )
    local_split_sizes = OrderedDict(
        (split_name, int(_dtensor_local_tensor(module.weight).shape[0]))
        for split_name, module in named_modules
    )
    use_bias = any(
        getattr(module, "bias", None) is not None for _, module in named_modules
    )
    linear = te_linear(
        in_features,
        sum(global_split_sizes.values()),
        bias=use_bias,
        params_dtype=dtype,
        parallel_mode="column",
        tp_group=tp_group,
        tp_size=tp_size,
        sequence_parallel=sequence_parallel,
        ub_overlap_ag=ub_overlap_ag,
        ub_name=ub_name,
        device=device,
        name=name,
    )
    _copy_te_parameter(
        linear.weight,
        torch.cat(
            [_dtensor_local_tensor(module.weight) for _, module in named_modules],
            dim=0,
        ),
        tensor_parallel_sharded=True,
    )
    linear_bias = getattr(linear, "bias", None)
    if linear_bias is not None and int(linear_bias.numel()) > 0:
        local_biases: list[torch.Tensor] = []
        for _, module in named_modules:
            source_bias = getattr(module, "bias", None)
            if source_bias is None:
                local_rows = int(_dtensor_local_tensor(module.weight).shape[0])
                local_biases.append(torch.zeros(local_rows, dtype=dtype, device=device))
            else:
                local_biases.append(_dtensor_local_tensor(source_bias))
        _copy_te_parameter(
            linear.bias,
            torch.cat(local_biases, dim=0),
            tensor_parallel_sharded=True,
        )
    linear._dllm_local_split_sizes = tuple(
        int(size) for size in local_split_sizes.values()
    )
    return linear


def _build_te_row_linear(
    te_linear: type[nn.Module],
    module: Any,
    *,
    runtime: Any,
    dtype: torch.dtype,
    device: torch.device,
    sequence_parallel: bool,
    name: str,
    ub_overlap_ag: bool = False,
    ub_overlap_rs: bool = False,
    ub_name: str | None = None,
) -> nn.Module:
    tp_size = int(getattr(runtime, "tensor_parallel_size", 1) or 1)
    tp_group = getattr(runtime, "tensor_parallel_group", None)
    linear = te_linear(
        _row_linear_in_features(module, tp_size=tp_size),
        _row_linear_out_features(module),
        bias=getattr(module, "bias", None) is not None,
        params_dtype=dtype,
        parallel_mode="row",
        tp_group=tp_group,
        tp_size=tp_size,
        sequence_parallel=sequence_parallel,
        ub_overlap_ag=ub_overlap_ag,
        ub_overlap_rs=ub_overlap_rs,
        ub_name=ub_name,
        device=device,
        name=name,
    )
    _copy_te_parameter(
        linear.weight,
        _dtensor_local_tensor(module.weight),
        tensor_parallel_sharded=True,
    )
    bias = getattr(module, "bias", None)
    if bias is not None and getattr(linear, "bias", None) is not None:
        _copy_te_parameter(
            linear.bias,
            _dtensor_local_tensor(bias),
            tensor_parallel_sharded=False,
            sequence_parallel_replicated=sequence_parallel,
        )
    return linear


def _copy_te_parameter(
    target: torch.Tensor,
    source: torch.Tensor,
    *,
    tensor_parallel_sharded: bool,
    sequence_parallel_replicated: bool = False,
) -> None:
    source = source.detach().to(device=target.device, dtype=target.dtype).contiguous()
    if tuple(target.shape) != tuple(source.shape):
        raise RuntimeError(
            "TE TP parameter shard shape mismatch: "
            f"target={tuple(target.shape)} source={tuple(source.shape)}"
        )
    with torch.no_grad():
        target.copy_(source)
    target._dllm_tensor_parallel_sharded = bool(tensor_parallel_sharded)
    target._dllm_sequence_parallel_replicated = bool(sequence_parallel_replicated)


def _local_split_sizes(module: nn.Module, names: tuple[str, ...]) -> tuple[int, ...]:
    cached = getattr(module, "_dllm_local_split_sizes", None)
    if cached is not None:
        return tuple(int(size) for size in cached)
    return tuple(int(getattr(module, f"{name}_weight").shape[0]) for name in names)


def _column_linear_in_features(module: Any) -> int:
    weight = _dtensor_local_tensor(module.weight)
    if weight.ndim != 2:
        raise TypeError("linear weight must be 2D")
    return int(weight.shape[1])


def _column_linear_out_features(module: Any, *, tp_size: int) -> int:
    weight = _dtensor_local_tensor(module.weight)
    if weight.ndim != 2:
        raise TypeError("linear weight must be 2D")
    value = getattr(module, "out_features", None)
    if value is not None:
        if int(tp_size) > 1 and int(value) == int(weight.shape[0]):
            return int(value) * int(tp_size)
        return int(value)
    return int(weight.shape[0]) * int(tp_size)


def _row_linear_in_features(module: Any, *, tp_size: int) -> int:
    weight = _dtensor_local_tensor(module.weight)
    if weight.ndim != 2:
        raise TypeError("linear weight must be 2D")
    value = getattr(module, "in_features", None)
    if value is not None:
        if int(tp_size) > 1 and int(value) == int(weight.shape[1]):
            return int(value) * int(tp_size)
        return int(value)
    return int(weight.shape[1]) * int(tp_size)


def _row_linear_out_features(module: Any) -> int:
    weight = _dtensor_local_tensor(module.weight)
    if weight.ndim != 2:
        raise TypeError("linear weight must be 2D")
    return int(weight.shape[0])


def _first_linear_dtype(*modules: Any) -> torch.dtype:
    for module in modules:
        for parameter in getattr(module, "parameters", lambda: ())():
            local = _dtensor_local_tensor(parameter)
            return local.dtype
    return torch.bfloat16


def _first_linear_device(*modules: Any) -> torch.device:
    for module in modules:
        for parameter in getattr(module, "parameters", lambda: ())():
            local = _dtensor_local_tensor(parameter)
            return local.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _first_existing_module(module: Any, names: tuple[str, ...], label: str) -> Any:
    for name in names:
        value = getattr(module, name, None)
        if value is not None:
            return value
    raise TypeError(f"{label} requires one of {', '.join(names)}")


def _mlp_activation(mlp: Any) -> Callable[[torch.Tensor], torch.Tensor]:
    for name in ("act_fn", "activation_fn", "act"):
        value = getattr(mlp, name, None)
        if callable(value):
            return value
    raise TypeError(
        "Nemotron TE TP MLP packing requires the backbone MLP to expose a "
        "callable activation as act_fn, activation_fn, or act"
    )


def _build_te_gated_activation(
    activation: Callable[[torch.Tensor], torch.Tensor],
) -> nn.Module | None:
    """Build the exact TE fused GLU kernel supported by this backbone.

    TE's fused operation applies the nonlinearity to the first half of the
    packed gate/up projection, matching the storage order constructed above.
    Activations without an exact TE gated equivalent retain their backbone
    implementation rather than changing model numerics.
    """

    from transformers.activations import SiLUActivation

    if activation is not torch.nn.functional.silu and not isinstance(
        activation,
        (nn.SiLU, SiLUActivation),
    ):
        return None
    try:
        from transformer_engine.pytorch.ops import Sequential, SwiGLU
    except Exception as exc:  # pragma: no cover - validated in production image.
        raise RuntimeError(
            "Nemotron TP requires Transformer Engine's native SwiGLU operation"
        ) from exc
    return Sequential(SwiGLU())


def _te_gated_mlp_forward(
    packed_layer: _TEPackedLayerProjections,
    hidden_states: torch.Tensor,
    *,
    token_chunk_size: int = 0,
) -> torch.Tensor:
    token_chunk_size = int(token_chunk_size or 0)
    if token_chunk_size > 0 and hidden_states.numel() > 0:
        hidden_shape = tuple(hidden_states.shape)
        hidden_dim = hidden_shape[-1]
        hidden_2d = hidden_states.reshape(-1, hidden_dim)
        if hidden_2d.shape[0] > token_chunk_size:
            outputs = [
                _te_gated_mlp_forward(
                    packed_layer,
                    chunk,
                    token_chunk_size=0,
                )
                for chunk in hidden_2d.split(token_chunk_size, dim=0)
            ]
            return torch.cat(outputs, dim=0).reshape(hidden_shape)
    gate_up = packed_layer.gate_up(hidden_states)
    if packed_layer.gated_activation is not None:
        activated = packed_layer.gated_activation(gate_up)
    else:
        gate, up = gate_up.split(packed_layer.gate_up_local_sizes, dim=-1)
        activated = packed_layer.activation(gate) * up
    return packed_layer.down_proj(activated)


def _hf_gated_mlp_forward(
    mlp: Any,
    hidden_states: torch.Tensor,
    *,
    token_chunk_size: int = 0,
) -> torch.Tensor:
    token_chunk_size = int(token_chunk_size or 0)
    if token_chunk_size > 0 and hidden_states.numel() > 0:
        hidden_shape = tuple(hidden_states.shape)
        hidden_dim = hidden_shape[-1]
        hidden_2d = hidden_states.reshape(-1, hidden_dim)
        if hidden_2d.shape[0] > token_chunk_size:
            outputs = [
                _hf_gated_mlp_forward(
                    mlp,
                    chunk,
                    token_chunk_size=0,
                )
                for chunk in hidden_2d.split(token_chunk_size, dim=0)
            ]
            return torch.cat(outputs, dim=0).reshape(hidden_shape)
    gate_proj = _first_existing_module(
        mlp,
        ("gate_proj", "gate", "w1"),
        "Nemotron MLP gate projection",
    )
    up_proj = _first_existing_module(
        mlp,
        ("up_proj", "up", "w3"),
        "Nemotron MLP up projection",
    )
    down_proj = _first_existing_module(
        mlp,
        ("down_proj", "down", "w2"),
        "Nemotron MLP down projection",
    )
    return down_proj(
        _mlp_activation(mlp)(gate_proj(hidden_states)) * up_proj(hidden_states)
    )


def _compact_valid_loss_rows(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    loss_weights: torch.Tensor,
    *,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden_2d = hidden_states.reshape(-1, hidden_states.shape[-1])
    labels_1d = labels.reshape(-1)
    weights_1d = loss_weights.reshape(-1).float()
    row_indices = torch.nonzero(
        labels_1d != int(ignore_index), as_tuple=False
    ).flatten()
    if row_indices.numel() == labels_1d.numel():
        return hidden_states, labels, weights_1d
    return (
        hidden_2d.index_select(0, row_indices),
        labels_1d.index_select(0, row_indices),
        weights_1d.index_select(0, row_indices),
    )


def _release_replaced_hf_tp_modules(
    layers: tuple[NemotronDecoderLayerOps, ...],
    packed_by_layer_id: dict[int, _TEPackedLayerProjections],
) -> tuple[NemotronDecoderLayerOps, ...]:
    released_layers: list[NemotronDecoderLayerOps] = []
    for layer_ops in layers:
        packed = packed_by_layer_id[id(layer_ops.layer)]
        released_q = _release_child_module(
            layer_ops.self_attn,
            ("q_proj",),
            layer_ops.attention.q_proj,
            "attention q_proj",
        )
        released_k = _release_child_module(
            layer_ops.self_attn,
            ("k_proj",),
            layer_ops.attention.k_proj,
            "attention k_proj",
        )
        released_v = (
            released_k
            if layer_ops.attention.value_from_key
            else _release_child_module(
                layer_ops.self_attn,
                ("v_proj",),
                layer_ops.attention.v_proj,
                "attention v_proj",
            )
        )
        released_o = _release_child_module(
            layer_ops.self_attn,
            ("o_proj",),
            layer_ops.attention.o_proj,
            "attention o_proj",
        )
        released_output_gate = layer_ops.attention.output_gate_proj
        if released_output_gate is not None:
            released_output_gate = _release_child_module(
                layer_ops.self_attn,
                ("gate_proj",),
                released_output_gate,
                "attention output gate projection",
            )
        released_mlp = layer_ops.mlp
        for names, label in (
            (("gate_proj", "gate", "w1"), "MLP gate projection"),
            (("up_proj", "up", "w3"), "MLP up projection"),
            (("down_proj", "down", "w2"), "MLP down projection"),
        ):
            module = _first_existing_module(layer_ops.mlp, names, label)
            _release_child_module(layer_ops.mlp, names, module, label)
        released_attention = replace(
            layer_ops.attention,
            q_proj=released_q,
            k_proj=released_k,
            v_proj=released_v,
            o_proj=released_o,
            output_gate_proj=released_output_gate,
        )
        released_input_norm = layer_ops.input_layernorm
        if packed.qkv_fuses_input_norm:
            released_input_norm = _release_child_module(
                layer_ops.layer,
                ("input_layernorm", "ln_1"),
                layer_ops.input_layernorm,
                "input RMSNorm",
            )
        released_pre_ff_norm = layer_ops.pre_feedforward_layernorm
        if packed.gate_up_fuses_pre_ff_norm:
            if released_pre_ff_norm is None:
                raise RuntimeError(
                    "fused Gemma4 dense MLP is missing pre-feedforward norm"
                )
            released_pre_ff_norm = _release_child_module(
                layer_ops.layer,
                ("pre_feedforward_layernorm",),
                released_pre_ff_norm,
                "pre-feedforward RMSNorm",
            )
        released_layers.append(
            replace(
                layer_ops,
                input_layernorm=released_input_norm,
                attention=released_attention,
                mlp=released_mlp,
                pre_feedforward_layernorm=released_pre_ff_norm,
            )
        )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return tuple(released_layers)


def _release_child_module(
    owner: Any,
    names: tuple[str, ...],
    module: Any,
    label: str,
) -> nn.Module:
    if not isinstance(module, nn.Module):
        raise TypeError(f"{label} must be an nn.Module")
    replacement = _ReleasedHFProjection(label)
    replaced = False
    for name in names:
        if getattr(owner, name, None) is module:
            setattr(owner, name, replacement)
            replaced = True
    if not replaced:
        for name, child in list(getattr(owner, "_modules", {}).items()):
            if child is module:
                setattr(owner, name, replacement)
                replaced = True
    if not replaced:
        raise RuntimeError(f"could not release replaced HF module for {label}")
    return replacement


def _maybe_replace_vocab_parallel_embedding(
    encoder: Any,
    embed_tokens: Any,
    runtime: Any,
    *,
    vocab_size: int | None,
) -> Any:
    tp_size = int(getattr(runtime, "tensor_parallel_size", 1) or 1)
    if tp_size <= 1 or isinstance(embed_tokens, VocabParallelEmbedding):
        return embed_tokens
    weight = getattr(embed_tokens, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        return embed_tokens
    global_vocab_size = int(vocab_size or weight.shape[0])
    tp_rank = int(getattr(runtime, "tensor_parallel_rank", 0) or 0)
    local_weight = _local_vocab_weight_shard(
        weight,
        global_vocab_size=global_vocab_size,
        tp_size=tp_size,
        tp_rank=tp_rank,
    )
    parallel_embed = VocabParallelEmbedding(
        global_vocab_size,
        int(weight.shape[1]),
        tensor_parallel_size=tp_size,
        tensor_parallel_rank=tp_rank,
        embedding_scale=float(getattr(embed_tokens, "scalar_embed_scale", 1.0)),
    ).to(device=local_weight.device, dtype=local_weight.dtype)
    parallel_embed.set_tensor_parallel_runtime(runtime)
    with torch.no_grad():
        parallel_embed.weight.copy_(local_weight)
    setattr(encoder, "embed_tokens", parallel_embed)
    return parallel_embed


def _tag_tensor_parallel_sharded_components(
    layers: tuple[NemotronDecoderLayerOps, ...],
    runtime: Any,
) -> None:
    if int(getattr(runtime, "tensor_parallel_size", 1) or 1) <= 1:
        return
    for layer_ops in layers:
        for module in (
            layer_ops.attention.q_proj,
            layer_ops.attention.k_proj,
            layer_ops.attention.v_proj,
            layer_ops.attention.o_proj,
            layer_ops.mlp,
            layer_ops.router,
            layer_ops.experts,
        ):
            if module is None:
                continue
            for parameter in getattr(module, "parameters", lambda: ())():
                parameter._dllm_tensor_parallel_sharded = True


def _tag_sequence_parallel_replicated_components(
    layers: tuple[NemotronDecoderLayerOps, ...],
    final_norm: Any,
    self_conditioning: Any | None,
    runtime: Any,
) -> None:
    if not bool(getattr(runtime, "sequence_parallel", False)):
        return
    if int(getattr(runtime, "tensor_parallel_size", 1) or 1) <= 1:
        return
    modules = [final_norm, self_conditioning]
    for layer_ops in layers:
        modules.extend(
            (
                layer_ops.input_layernorm,
                layer_ops.post_attention_layernorm,
                layer_ops.attention.q_norm,
                layer_ops.attention.k_norm,
                layer_ops.attention.v_norm,
                layer_ops.pre_feedforward_layernorm,
                layer_ops.post_feedforward_layernorm,
                layer_ops.pre_feedforward_layernorm_2,
                layer_ops.post_feedforward_layernorm_1,
                layer_ops.post_feedforward_layernorm_2,
                layer_ops.router,
                layer_ops.experts,
            )
        )
    for module in modules:
        if module is None:
            continue
        for parameter in getattr(module, "parameters", lambda: ())():
            parameter._dllm_sequence_parallel_replicated = True


def _maybe_replace_vocab_parallel_output_head(
    model: Any,
    output_head: Any,
    runtime: Any,
    *,
    vocab_size: int | None,
    shared_weight: nn.Parameter | None = None,
) -> Any:
    tp_size = int(getattr(runtime, "tensor_parallel_size", 1) or 1)
    if tp_size <= 1 or isinstance(output_head, VocabParallelLinear):
        return output_head
    weight = getattr(output_head, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        return output_head
    bias = getattr(output_head, "bias", None)
    if bias is not None and not isinstance(bias, torch.Tensor):
        return output_head
    global_vocab_size = int(vocab_size or weight.shape[0])
    tp_rank = int(getattr(runtime, "tensor_parallel_rank", 0) or 0)
    local_weight = _local_vocab_weight_shard(
        weight,
        global_vocab_size=global_vocab_size,
        tp_size=tp_size,
        tp_rank=tp_rank,
    )
    local_bias = (
        _local_vocab_weight_shard(
            bias,
            global_vocab_size=global_vocab_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
        )
        if isinstance(bias, torch.Tensor)
        else None
    )
    parallel_head = VocabParallelLinear(
        int(weight.shape[1]),
        global_vocab_size,
        bias=local_bias is not None,
        tensor_parallel_size=tp_size,
        tensor_parallel_rank=tp_rank,
        gather_output=False,
    ).to(device=local_weight.device, dtype=local_weight.dtype)
    parallel_head.set_tensor_parallel_runtime(runtime)
    if shared_weight is not None:
        if shared_weight.shape != parallel_head.weight.shape:
            raise ValueError("tied vocabulary TP shards must have identical shapes")
        parallel_head.weight = shared_weight
    with torch.no_grad():
        if shared_weight is None:
            parallel_head.weight.copy_(local_weight)
        if parallel_head.bias is not None and local_bias is not None:
            parallel_head.bias.copy_(local_bias.to(parallel_head.bias.dtype))
    for owner in (
        model,
        getattr(model, "model", None),
        getattr(model, "encoder", None),
    ):
        if owner is None:
            continue
        for name in ("diffusion_head", "lm_head", "embed_out"):
            if getattr(owner, name, None) is output_head:
                setattr(owner, name, parallel_head)
                return parallel_head
    return parallel_head


def _local_vocab_weight_shard(
    weight: torch.Tensor,
    *,
    global_vocab_size: int,
    tp_size: int,
    tp_rank: int,
) -> torch.Tensor:
    local = _dtensor_local_tensor(weight).detach()
    start, stop = partition_bounds(int(global_vocab_size), int(tp_size), int(tp_rank))
    expected_rows = stop - start
    if local.shape[0] == expected_rows:
        return local.contiguous()
    return local[start:stop].contiguous()


def _linear_weight_bias(
    module: Any, name: str
) -> tuple[torch.Tensor, torch.Tensor | None]:
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise TypeError(
            f"Nemotron packed BP/CP requires {name}.weight to be a 2D tensor"
        )
    bias = getattr(module, "bias", None)
    if bias is not None and not isinstance(bias, torch.Tensor):
        raise TypeError(
            f"Nemotron packed BP/CP requires {name}.bias to be a tensor or None"
        )
    return weight, bias


def _soft_embedding_scale(embed_tokens: Any, embedding_scale: Any | None) -> float:
    """Resolve the scale used for expected token embeddings.

    Released DiffusionGemma embeddings apply ``scalar_embed_scale`` inside
    their own forward, whereas older backbones expose a separate encoder
    normalizer consumed by the packed wrapper.
    """

    return float(
        getattr(
            embed_tokens,
            "scalar_embed_scale",
            1.0 if embedding_scale is None else embedding_scale,
        )
    )


def _nemotronlabsdiffusion_components(model: Any) -> NemotronBlockDiffusionComponents:
    encoder = _language_backbone(model)
    embed_tokens = _first_attr(encoder, ("embed_tokens", "wte"))
    layers = _first_attr(encoder, ("layers", "h"))
    norm = _first_attr(encoder, ("norm", "final_layernorm", "ln_f"))
    output_head = _output_head(model, encoder)
    rotary_emb = getattr(encoder, "rotary_emb", None)
    if not callable(rotary_emb):
        rotary_emb = _null_rotary_embeddings
    config = getattr(encoder, "config", getattr(model, "config", None))
    num_hidden_layers = getattr(config, "num_hidden_layers", None)
    if num_hidden_layers is None:
        num_hidden_layers = len(layers)
    embedding_scale = getattr(encoder, "normalizer", None)
    soft_embedding_scale = _soft_embedding_scale(embed_tokens, embedding_scale)
    embedding_norm = getattr(embed_tokens, "embed_norm", None)
    final_logit_softcap = _final_logit_softcap(model, encoder)
    output_multiplier = _output_multiplier(model, encoder)
    self_conditioning = getattr(encoder, "self_conditioning", None)
    layer_ops = tuple(
        _decoder_layer_ops(layer, index)
        for index, layer in enumerate(layers[: int(num_hidden_layers)])
    )
    return NemotronBlockDiffusionComponents(
        encoder=encoder,
        embed_tokens=embed_tokens,
        layers=layer_ops,
        norm=norm,
        output_head=output_head,
        rotary_emb=rotary_emb,
        rotary_emb_accepts_layer_type=_accepts_layer_type(rotary_emb),
        num_hidden_layers=int(num_hidden_layers),
        embedding_norm=embedding_norm,
        embedding_scale=embedding_scale,
        soft_embedding_scale=soft_embedding_scale,
        final_logit_softcap=final_logit_softcap,
        output_multiplier=output_multiplier,
        self_conditioning=self_conditioning,
    )


def _accepts_layer_type(rotary_emb: Any) -> bool:
    target = rotary_emb.forward if isinstance(rotary_emb, nn.Module) else rotary_emb
    parameters = tuple(inspect.signature(target).parameters)
    return "layer_type" in parameters or len(parameters) >= 3


def _language_backbone(model: Any) -> Any:
    for candidate in _language_backbone_candidates(model):
        if candidate is None:
            continue
        language_model = getattr(candidate, "language_model", None)
        if language_model is not None:
            return language_model
        if (
            getattr(candidate, "layers", None) is not None
            or getattr(candidate, "h", None) is not None
        ):
            return candidate
    raise TypeError(
        "Nemotron packed BP/CP requires a language backbone with decoder layers"
    )


def _language_backbone_candidates(model: Any) -> tuple[Any | None, ...]:
    root = getattr(model, "model", None)
    root_encoder = getattr(root, "encoder", None) if root is not None else None
    top_encoder = getattr(model, "encoder", None)
    return (
        getattr(root, "decoder", None) if root is not None else None,
        getattr(model, "decoder", None),
        getattr(root_encoder, "language_model", None)
        if root_encoder is not None
        else None,
        getattr(top_encoder, "language_model", None)
        if top_encoder is not None
        else None,
        root,
        top_encoder,
        getattr(model, "transformer", None),
    )


def _output_head(model: Any, encoder: Any) -> Any:
    for owner in (model, encoder):
        for name in ("diffusion_head", "lm_head", "embed_out"):
            value = getattr(owner, name, None)
            if value is not None:
                return value
    raise TypeError("Nemotron packed BP/CP requires an LM/diffusion output head")


def _final_logit_softcap(model: Any, encoder: Any) -> float | None:
    for owner in (
        model,
        encoder,
        getattr(model, "config", None),
        getattr(encoder, "config", None),
    ):
        if owner is None:
            continue
        for name in ("final_logit_softcapping", "final_logit_softcap"):
            value = getattr(owner, name, None)
            if value is not None:
                value = float(value)
                return value if value > 0 else None
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        value = getattr(text_config, "final_logit_softcapping", None)
        if value is not None:
            value = float(value)
            return value if value > 0 else None
    return None


def _output_multiplier(model: Any, encoder: Any) -> float | None:
    for owner in (
        model,
        encoder,
        getattr(model, "config", None),
        getattr(encoder, "config", None),
        getattr(getattr(model, "config", None), "text_config", None),
    ):
        if owner is None:
            continue
        value = getattr(owner, "output_multiplier", None)
        if value is not None:
            return float(value)
    return None


def _null_rotary_embeddings(
    hidden_states: torch.Tensor,
    *,
    position_ids: torch.Tensor,
) -> tuple[None, None]:
    del hidden_states, position_ids
    return None, None


def _first_attr(module: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        value = getattr(module, name, None)
        if value is not None:
            return value
    raise TypeError(
        f"Nemotron packed BP/CP requires one of {', '.join(names)} on "
        f"{module.__class__.__name__}"
    )


def _layer_self_attention(layer: Any) -> Any | None:
    for name in ("self_attn", "attention", "attn"):
        value = getattr(layer, name, None)
        if value is not None:
            return value
    return None


def _decoder_layer_ops(layer: Any, index: int) -> NemotronDecoderLayerOps:
    attn = _layer_self_attention(layer)
    if attn is None:
        raise TypeError(f"decoder layer {index} does not expose self attention")
    gemma4_style = (
        getattr(layer, "pre_feedforward_layernorm", None) is not None
        and getattr(layer, "post_feedforward_layernorm", None) is not None
    )
    return NemotronDecoderLayerOps(
        layer=layer,
        input_layernorm=_first_attr(layer, ("input_layernorm", "ln_1")),
        self_attn=attn,
        attention=_attention_ops(attn, layer_index=index),
        post_attention_layernorm=_first_attr(
            layer,
            ("post_attention_layernorm", "ln_2"),
        ),
        mlp=_first_attr(layer, ("mlp", "feed_forward")),
        layer_style="gemma4" if gemma4_style else "standard",
        pre_feedforward_layernorm=getattr(layer, "pre_feedforward_layernorm", None),
        post_feedforward_layernorm=getattr(layer, "post_feedforward_layernorm", None),
        router=getattr(layer, "router", None),
        moe=getattr(layer, "moe", None),
        experts=getattr(layer, "experts", None),
        pre_feedforward_layernorm_2=getattr(layer, "pre_feedforward_layernorm_2", None),
        post_feedforward_layernorm_1=getattr(
            layer, "post_feedforward_layernorm_1", None
        ),
        post_feedforward_layernorm_2=getattr(
            layer, "post_feedforward_layernorm_2", None
        ),
        layer_scalar=getattr(layer, "layer_scalar", None),
        layer_type=_attention_layer_type(attn),
    )


def _attention_layer_type(attn: Any) -> str | None:
    layer_type = getattr(attn, "layer_type", None)
    if layer_type is not None:
        return str(layer_type)
    if hasattr(attn, "is_local_attention"):
        return (
            "sliding_attention" if bool(attn.is_local_attention) else "full_attention"
        )
    return None


def _attention_ops(attn: Any, *, layer_index: int) -> NemotronAttentionOps:
    q_proj, k_proj, v_proj, o_proj, value_from_key = _attention_projections(attn)
    apply_rotary = _attention_rotary_fn(attn)
    rope_scale = _attention_rope_scale_fn(attn)
    shared_qk_norm = getattr(attn, "qk_norm", None)
    q_norm = getattr(attn, "q_norm", None)
    k_norm = getattr(attn, "k_norm", None)
    return NemotronAttentionOps(
        q_proj=q_proj,
        k_proj=k_proj,
        v_proj=v_proj,
        o_proj=o_proj,
        q_norm=shared_qk_norm if q_norm is None else q_norm,
        k_norm=shared_qk_norm if k_norm is None else k_norm,
        v_norm=getattr(attn, "v_norm", None),
        head_dim=_attention_head_dim(attn),
        apply_rotary=apply_rotary,
        rope_scale=rope_scale,
        scale=_attention_scale(attn),
        output_gate_proj=getattr(attn, "gate_proj", None),
        query_scale_multiplier=float(getattr(attn, "qk_scale_factor", 1.0)),
        uses_rope=_attention_uses_rope(attn, layer_index=layer_index),
        sliding_window=optional_int(getattr(attn, "sliding_window", None)),
        value_from_key=value_from_key,
    )


def _attention_projections(attn: Any) -> tuple[Any, Any, Any, Any, bool]:
    from torch import nn

    q_proj = getattr(attn, "q_proj", None)
    k_proj = getattr(attn, "k_proj", None)
    v_proj = getattr(attn, "v_proj", None)
    o_proj = getattr(attn, "o_proj", None)
    value_from_key = False
    if v_proj is None and isinstance(k_proj, nn.Module):
        value_from_key = True
        v_proj = k_proj
    if all(
        isinstance(module, nn.Module) for module in (q_proj, k_proj, v_proj, o_proj)
    ):
        return q_proj, k_proj, v_proj, o_proj, value_from_key
    raise TypeError(
        "Nemotron packed BP/CP requires attention modules with "
        "q_proj/k_proj/v_proj/o_proj"
    )


def _attention_head_dim(attn: Any) -> int:
    value = getattr(attn, "head_dim", None)
    if value is not None:
        return int(value)
    q_proj = getattr(attn, "q_proj", None)
    out_features = getattr(q_proj, "out_features", None)
    num_heads = getattr(attn, "num_heads", None) or getattr(
        getattr(attn, "config", None),
        "num_attention_heads",
        None,
    )
    if out_features is not None and num_heads is not None:
        return int(out_features) // int(num_heads)
    raise TypeError("Nemotron packed BP/CP could not infer attention head_dim")


def _apply_qkv_norms(
    attn: NemotronAttentionOps,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if attn.q_norm is not None:
        query_states = attn.q_norm(query_states)
    if attn.k_norm is not None:
        key_states = attn.k_norm(key_states)
    if attn.v_norm is not None:
        value_states = attn.v_norm(value_states)
    if attn.query_scale_multiplier != 1.0:
        query_states = query_states * float(attn.query_scale_multiplier)
    return query_states, key_states, value_states


def _attention_uses_rope(attn: Any, *, layer_index: int) -> bool:
    config = getattr(attn, "config", None)
    per_layer = getattr(config, "layer_rope_theta", None)
    if per_layer is None:
        return True
    if not 0 <= int(layer_index) < len(per_layer):
        raise ValueError("per-layer RoPE configuration does not cover every layer")
    return bool(float(per_layer[int(layer_index)]))


def _attention_rotary_fn(attn: Any) -> Callable[..., Any]:
    value = getattr(attn, "apply_rotary_pos_emb", None)
    if callable(value):
        return _rotary_bshd_adapter(value)
    value = getattr(attn.forward, "__globals__", {}).get("apply_rotary_pos_emb")
    if callable(value):
        return _rotary_bshd_adapter(value)
    rotary_emb = getattr(attn, "rotary_emb", None)
    if callable(rotary_emb):
        return lambda q, k, cos, sin, cache_position: _apply_module_rotary_bshd(
            rotary_emb,
            q,
            k,
            cache_position,
        )
    raise TypeError("Nemotron packed BP/CP requires a rotary embedding function")


def _rotary_bshd_adapter(
    apply_rotary_pos_emb: Callable[..., Any],
) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    """Resolve a BSHD RoPE call once while constructing the backbone."""

    parameters = tuple(inspect.signature(apply_rotary_pos_emb).parameters)
    if parameters[:3] == ("x", "cos", "sin"):
        if "unsqueeze_dim" not in parameters:
            raise TypeError(
                "single-tensor rotary embedding must expose unsqueeze_dim for BSHD"
            )
        return lambda q, k, cos, sin, cache_position: (
            apply_rotary_pos_emb(q, cos, sin, unsqueeze_dim=2),
            apply_rotary_pos_emb(k, cos, sin, unsqueeze_dim=2),
        )
    if "unsqueeze_dim" in parameters:
        return lambda q, k, cos, sin, cache_position: apply_rotary_pos_emb(
            q,
            k,
            cos,
            sin,
            unsqueeze_dim=2,
        )[:2]
    return lambda q, k, cos, sin, cache_position: _apply_rotary_via_bhsd(
        apply_rotary_pos_emb,
        q,
        k,
        cos,
        sin,
    )


def _apply_rotary_via_bhsd(
    apply_rotary_pos_emb: Callable[..., Any],
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    value = apply_rotary_pos_emb(
        query_states.transpose(1, 2),
        key_states.transpose(1, 2),
        cos,
        sin,
    )
    if isinstance(value, tuple) and len(value) >= 2:
        return value[0].transpose(1, 2), value[1].transpose(1, 2)
    raise TypeError("rotary embedding must return rotated query/key tensors")


def _apply_module_rotary_bshd(
    rotary_emb: Any,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    cache_position: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, seq_len, query_heads, head_dim = query_states.shape
    key_heads = key_states.shape[2]
    positions = cache_position.unsqueeze(0).expand(batch, -1).reshape(-1)
    query_flat = query_states.reshape(batch * seq_len, query_heads * head_dim)
    key_flat = key_states.reshape(batch * seq_len, key_heads * head_dim)
    rotated = rotary_emb(positions, query_flat, key_flat)
    if not isinstance(rotated, tuple) or len(rotated) < 2:
        raise TypeError("rotary_emb must return rotated query/key tensors")
    query_flat, key_flat = rotated[:2]
    query_states = query_flat.reshape(batch, seq_len, query_heads, head_dim)
    key_states = key_flat.reshape(batch, seq_len, key_heads, head_dim)
    return query_states, key_states


def _rope_scale_bshd(
    scale: torch.Tensor,
    *,
    seq_len: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if scale.numel() != int(seq_len):
        raise RuntimeError(
            "position-dependent attention scale must contain one value per token"
        )
    return scale.reshape(1, int(seq_len), 1, 1).to(dtype=dtype)


def _attention_rope_scale_fn(attn: Any) -> Callable[[Any], Any | None]:
    value = getattr(attn.forward, "__globals__", {}).get("_get_llama_4_attn_scale")
    if not callable(value):
        return lambda cache_position: None
    config = getattr(attn, "config", None)
    rope_parameters = getattr(config, "rope_parameters", {}) or {}
    beta = rope_parameters.get("llama_4_scaling_beta")
    original_max_position_embeddings = rope_parameters.get(
        "original_max_position_embeddings"
    )

    def scale(cache_position: Any) -> Any | None:
        return value(
            cache_position,
            beta,
            original_max_position_embeddings,
        )

    return scale


def _attention_scale(attn: Any) -> float:
    value = getattr(attn, "scaling", None)
    if value is not None:
        return float(value)
    head_dim = _attention_head_dim(attn)
    return float(head_dim) ** -0.5
