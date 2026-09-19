# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Text-only Qwen3.8 hybrid Gated-DeltaNet block-diffusion execution."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import json
import math
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

from dllm_parallel.core.adapters import current_lora_token_mask, lora_token_mask

from dllm_parallel.core.attention import (
    fused_block_context_attention_bshd,
    pure_context_block_denoising_attention_bshd,
    pure_context_persistent_block_denoising_attention_bshd,
    replicated_block_denoising_attention_bshd,
)
from dllm_parallel.core.attention.layout import active_query_indices_for_context_rank
from dllm_parallel.core.models.backbones.nemotron.model import (
    NemotronLabsDiffusionPackedBlockDiffusionModel,
    _ReleasedHFProjection,
    _build_te_fused_column_linear,
    _build_te_row_linear,
    _column_linear_out_features,
    _copy_te_parameter,
    _dtensor_local_tensor,
    _first_linear_device,
    _first_linear_dtype,
    _local_split_sizes,
    _maybe_replace_vocab_parallel_embedding,
    _maybe_replace_vocab_parallel_output_head,
    _transformer_engine_linear_cls,
)
from dllm_parallel.core.models.backbones.qwen3_8.token_local import (
    build_pure_cp_token_row_plan,
    compact_pure_cp_token_rows,
    expand_pure_cp_logical_rows,
    reconstruct_pure_cp_token_rows,
    select_pure_cp_logical_rows,
)
from dllm_parallel.core.parallel.runtime import active_blocks_for_runtime


_TEXT_CHECKPOINT_KEY_MAPPING = {
    r"^model\.language_model\.": "model.",
}


def _distributed_checkpoint(
    function: Any,
    *args: torch.Tensor,
    use_reentrant: bool,
) -> torch.Tensor:
    """Checkpoint a collective-bearing region with rank-identical replay."""

    # Non-reentrant checkpointing normally stops replay once the tensors needed
    # by the local autograd graph have been reconstructed. Context shards can
    # need different tensors on different ranks, which would make collective
    # replay diverge. Full replay preserves ordering and still retains the LoRA
    # graph when the frozen input itself does not require gradients.
    with set_checkpoint_early_stop(False):
        return checkpoint(function, *args, use_reentrant=use_reentrant)


def _attach_qwen38_layer_gradient_probe(
    hidden_states: torch.Tensor,
    *,
    layer_index: int,
    rank: int,
    role_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Report a bad activation gradient at its exact transformer boundary."""

    if not hidden_states.requires_grad:
        return hidden_states

    def report(gradient: torch.Tensor) -> torch.Tensor:
        local = _dtensor_local_tensor(gradient)
        finite = torch.isfinite(local)
        role_stats: dict[str, float] = {}
        if isinstance(role_mask, torch.Tensor):
            flat_mask = role_mask.to(device=local.device, dtype=torch.bool).reshape(-1)
            flat_gradient = local.detach().reshape(-1, int(local.shape[-1]))
            if int(flat_mask.numel()) == int(flat_gradient.shape[0]):
                active = flat_gradient[flat_mask]
                teacher = flat_gradient[~flat_mask]
                role_stats = {
                    "active_max_abs": (
                        float(active.abs().amax().item()) if active.numel() else 0.0
                    ),
                    "teacher_max_abs": (
                        float(teacher.abs().amax().item()) if teacher.numel() else 0.0
                    ),
                }
        if not bool(finite.all()):
            print(
                json.dumps(
                    {
                        "event": "nonfinite_layer_output_gradient",
                        "rank": int(rank),
                        "layer_index": int(layer_index),
                        "shape": list(local.shape),
                        "finite": int(finite.sum().item()),
                        "numel": int(local.numel()),
                        "nan": int(torch.isnan(local).sum().item()),
                        "posinf": int(torch.isposinf(local).sum().item()),
                        "neginf": int(torch.isneginf(local).sum().item()),
                        **role_stats,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        elif int(rank) == 0:
            print(
                json.dumps(
                    {
                        "event": "layer_output_gradient_finite",
                        "rank": int(rank),
                        "layer_index": int(layer_index),
                        "shape": list(local.shape),
                        "max_abs": float(local.detach().abs().amax().item()),
                        **role_stats,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        return gradient

    hidden_states.register_hook(report)
    return hidden_states


def _qwen38_context_attention(runtime: Any) -> Any:
    if int(getattr(runtime, "block_parallel_size", 1) or 1) == 1:
        return pure_context_block_denoising_attention_bshd
    return fused_block_context_attention_bshd


def _uses_pure_cp_token_local_mlp(runtime: Any, packed: Any | None) -> bool:
    """Whether Qwen's bias-free fused MLP can shard token rows over pure CP."""

    mlp = getattr(packed, "mlp", None)
    gate_up = getattr(packed, "gate_up", None)
    down_proj = getattr(packed, "down_proj", None)
    if mlp is None and (gate_up is None or down_proj is None):
        return False
    if not dist.is_initialized():
        return False
    if int(getattr(runtime, "block_parallel_size", 1) or 1) != 1:
        return False
    if int(getattr(runtime, "configured_context_parallel_size", 1) or 1) <= 1:
        return False
    if getattr(runtime, "context_block_parallel_group", None) is None:
        return False
    if not bool(getattr(runtime, "sequence_parallel", False)):
        return False
    if int(getattr(runtime, "tensor_parallel_size", 1) or 1) <= 1:
        return False
    if mlp is not None and bool(getattr(mlp, "_dllm_qwen38_bias_free", False)):
        return True
    modules = (mlp,) if mlp is not None else (gate_up, down_proj)
    return all(
        not bool(getattr(module, "use_bias", False))
        and all(
            getattr(module, name, None) is None
            for name in ("bias", "fc1_bias", "fc2_bias")
        )
        for module in modules
    )


def _uses_persistent_pure_cp(
    runtime: Any,
    packed: Any | None,
) -> bool:
    """Whether the complete Qwen layer can retain CP-owned active rows."""

    return bool(
        _uses_pure_cp_token_local_mlp(runtime, packed)
        and getattr(packed, "mixer_input", None) is not None
        and getattr(packed, "mixer_output", None) is not None
    )


def _ordered_gdn_boundary_states(
    nonzero_states: torch.Tensor | None,
    zero_state: torch.Tensor,
    branch_ids: tuple[int, ...],
) -> torch.Tensor:
    """Restore branch order without per-boundary autograd accumulation."""

    expected_nonzero = len(branch_ids) - int(0 in branch_ids)
    if nonzero_states is None:
        if expected_nonzero:
            raise ValueError("nonzero GDN branches require boundary states")
        return zero_state.unsqueeze(1)
    if int(nonzero_states.shape[1]) != expected_nonzero:
        raise ValueError("GDN boundary-state count does not match its branches")
    if 0 not in branch_ids:
        return nonzero_states
    zero_slot = branch_ids.index(0)
    return torch.cat(
        (
            nonzero_states[:, :zero_slot],
            zero_state.unsqueeze(1),
            nonzero_states[:, zero_slot:],
        ),
        dim=1,
    )


def _uses_te_tensor_parallel_overlap(runtime: Any) -> bool:
    """Whether fused Qwen TP/SP projections should use Userbuffers.

    Pure CP keeps its independently optimized compact token-local MLP path and
    does not pay the Userbuffer launch/allocation overhead for the mixer alone.
    """

    return bool(
        getattr(runtime, "tensor_parallel_overlap", False)
        and getattr(runtime, "sequence_parallel", False)
        and int(getattr(runtime, "tensor_parallel_size", 1) or 1) > 1
        and int(getattr(runtime, "block_parallel_size", 1) or 1) > 1
    )


@dataclass(frozen=True)
class _SequenceParallelMeta:
    batch_size: int
    packed_len: int
    hidden_size: int
    total_rows: int
    padded_rows: int
    active_len: int


@dataclass(frozen=True)
class _GDNHeadShard:
    key_start: int
    key_stop: int
    value_start: int
    value_stop: int


class _Qwen38TEProjections(nn.Module):
    """Packed Transformer Engine projections for one Qwen3.8 layer."""

    def __init__(
        self,
        *,
        mixer_input: nn.Module,
        mixer_input_local_sizes: tuple[int, ...],
        mixer_input_fuses_norm: bool,
        mixer_output: nn.Module,
        gate_up: nn.Module | None,
        gate_up_local_sizes: tuple[int, int],
        down_proj: nn.Module | None,
        mlp: nn.Module | None,
    ) -> None:
        super().__init__()
        self.mixer_input = mixer_input
        self.mixer_input_local_sizes = tuple(int(size) for size in mixer_input_local_sizes)
        self.mixer_input_fuses_norm = bool(mixer_input_fuses_norm)
        self.mixer_output = mixer_output
        self.gate_up = gate_up
        self.gate_up_local_sizes = tuple(int(size) for size in gate_up_local_sizes)
        self.down_proj = down_proj
        self.mlp = mlp


def _text_config(config: Any) -> Any:
    return getattr(config, "text_config", None) or config


def _unexpected_text_checkpoint_keys(
    keys: tuple[str, ...],
    *,
    loaded_layers: int,
) -> tuple[str, ...]:
    """Return checkpoint keys that cannot be explained by text-only truncation."""

    unexpected: list[str] = []
    layer_prefix = "model.layers."
    for key in keys:
        if key.startswith("model.visual."):
            continue
        if key.startswith(layer_prefix):
            layer, separator, _ = key[len(layer_prefix) :].partition(".")
            if separator and layer.isdecimal() and int(layer) >= loaded_layers:
                continue
        unexpected.append(key)
    return tuple(unexpected)


def load_text_checkpoint(
    *,
    model_id: str,
    revision: str | None,
    config: Any,
    runtime: Any | None,
    dtype: torch.dtype,
    device: torch.device,
    trust_remote_code: bool,
) -> nn.Module:
    """Load only Qwen3.8 language weights from the multimodal Hub checkpoint."""

    try:
        from transformers import Qwen3_5ForCausalLM
    except ImportError as exc:
        raise RuntimeError("Qwen3.8 training requires transformers>=5.13.0") from exc

    text_config = _text_config(config)
    text_config.use_cache = False
    mask_token_id = getattr(config, "mask_token_id", None)
    if mask_token_id is not None:
        text_config.mask_token_id = int(mask_token_id)

    tp_layout = None
    if runtime is not None and int(getattr(runtime, "tensor_parallel_size", 1)) > 1:
        from dllm_parallel.core.models.backbones.qwen3_8.tensor_parallel import (
            configure_qwen38_tensor_parallel,
        )

        tp_layout = configure_qwen38_tensor_parallel(text_config, runtime)

    kwargs: dict[str, Any] = {
        "config": text_config,
        "revision": revision,
        "trust_remote_code": trust_remote_code,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "key_mapping": _TEXT_CHECKPOINT_KEY_MAPPING,
        "output_loading_info": True,
    }
    if runtime is not None and int(getattr(runtime, "tensor_parallel_size", 1)) > 1:
        from dllm_parallel.core.parallel.hf_tensor_parallel import (
            hf_from_pretrained_tensor_parallel_kwargs,
            validate_hf_tensor_parallel_model,
        )

        kwargs.update(
            hf_from_pretrained_tensor_parallel_kwargs(runtime, device_type=device.type)
        )
    model, loading_info = Qwen3_5ForCausalLM.from_pretrained(model_id, **kwargs)
    missing = tuple(str(key) for key in loading_info.get("missing_keys", ()))
    if missing:
        preview = ", ".join(missing[:8])
        raise RuntimeError(
            "Qwen3.8 text checkpoint did not initialize every language parameter: "
            f"{preview}"
        )
    unexpected = _unexpected_text_checkpoint_keys(
        tuple(str(key) for key in loading_info.get("unexpected_keys", ())),
        loaded_layers=int(text_config.num_hidden_layers),
    )
    if unexpected:
        preview = ", ".join(unexpected[:8])
        raise RuntimeError(f"Qwen3.8 checkpoint contains unexpected text weights: {preview}")
    if runtime is not None and int(getattr(runtime, "tensor_parallel_size", 1)) > 1:
        from dllm_parallel.core.models.backbones.qwen3_8.tensor_parallel import (
            finalize_qwen38_tensor_parallel,
        )

        validate_hf_tensor_parallel_model(model, runtime)
        finalize_qwen38_tensor_parallel(model, tp_layout)
    elif device is not None:
        model.to(device)
    return model


def verify_qwen38_runtime() -> dict[str, Any]:
    """Fail before weight loading unless Qwen3.8 production kernels are installed."""

    try:
        import fla
        import tilelang
        from causal_conv1d import causal_conv1d_fn  # noqa: F401
        from fla.ops.common.backends.tilelang import TileLangBackend
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401
        from transformer_engine.pytorch import Linear as TELinear  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Qwen3.8 production training requires "
            "flash-linear-attention[cuda,conv1d,tilelang]==0.5.2"
        ) from exc
    version = str(getattr(fla, "__version__", ""))
    if version != "0.5.2":
        raise RuntimeError(f"Qwen3.8 requires flash-linear-attention 0.5.2, got {version!r}")
    tilelang_version = str(getattr(tilelang, "__version__", ""))
    if tilelang_version != "0.1.13":
        raise RuntimeError(f"Qwen3.8 requires tilelang 0.1.13, got {tilelang_version!r}")
    if not TileLangBackend.is_available():
        raise RuntimeError("Qwen3.8 requires the FLA TileLang backend on Hopper GPUs")
    from dllm_parallel.core.kernels.gated_delta_boundaries import (
        verify_gated_delta_boundary_runtime,
    )

    return {
        "qwen3_8_text_only": True,
        "gated_delta_net": "flash-linear-attention",
        "flash_linear_attention_version": version,
        "tilelang_version": tilelang_version,
        "causal_conv1d": True,
        "transformer_engine_fused_norm_linears": True,
        **verify_gated_delta_boundary_runtime(),
    }


class Qwen38PackedBlockDiffusionModel(
    NemotronLabsDiffusionPackedBlockDiffusionModel
):
    """Exact packed execution for Qwen3.8's mixed GDN/attention decoder.

    Full-attention layers retain the standard packed CP/BP representation.
    GDN layers evaluate the clean recurrent sequence and each locally owned
    target block from its clean ground-truth prefix. This prevents recurrent
    state from leaking between independently corrupted target blocks.
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
    ) -> None:
        nn.Module.__init__(self)
        encoder = getattr(model, "model", None)
        if encoder is None or not hasattr(encoder, "layers"):
            raise TypeError("Qwen3.8 text checkpoint must expose model.layers")
        config = encoder.config
        layers = tuple(encoder.layers[: int(config.num_hidden_layers)])
        configured_layer_types = tuple(str(value) for value in config.layer_types)
        if len(configured_layer_types) < len(layers):
            raise ValueError("Qwen3.8 layer_types must cover the loaded decoder layers")
        layer_types = configured_layer_types[: len(layers)]

        self.model = model
        self.encoder = encoder
        self.config = config
        self.runtime = runtime
        self.max_seq_len = int(seq_len)
        self.seq_len = int(seq_len)
        self.block_size = int(block_size)
        self.ring_attention_key_chunk_size = int(ring_attention_key_chunk_size)
        self.activation_checkpointing = bool(activation_checkpointing)
        if activation_checkpointing_scope not in {"full", "mlp"}:
            raise ValueError("activation_checkpointing_scope must be 'full' or 'mlp'")
        self.activation_checkpointing_scope = str(activation_checkpointing_scope)
        self.mlp_token_chunk_size = int(mlp_token_chunk_size)
        self.sequence_parallel = bool(getattr(runtime, "sequence_parallel", False))
        self._adapter_checkpointing = False
        self._gradient_boundary_probe_rank: int | None = None
        self.layers = nn.ModuleList(layers)
        self.layer_types = layer_types
        self.rotary_emb = encoder.rotary_emb
        self.norm = encoder.norm
        self.final_logit_softcap = None
        vocab_size = int(config.vocab_size)
        tied_vocab_weights = model.lm_head.weight is encoder.embed_tokens.weight
        self.embed_tokens = _maybe_replace_vocab_parallel_embedding(
            encoder,
            encoder.embed_tokens,
            runtime,
            vocab_size=vocab_size,
        )
        self.output_head = _maybe_replace_vocab_parallel_output_head(
            model,
            model.lm_head,
            runtime,
            vocab_size=vocab_size,
            shared_weight=self.embed_tokens.weight if tied_vocab_weights else None,
        )
        self._packed_layout_cache: dict[tuple[object, ...], Any] = {}
        self._layer_attention_mask_cache: dict[tuple[object, ...], Any] = {}
        self._clean_expert_token_plan_cache: dict[tuple[object, ...], Any] = {}
        self._pure_cp_token_row_plan_cache: dict[tuple[object, ...], Any] = {}

        self._te_packed_layers = nn.ModuleList()
        self._te_packed_by_layer_id: dict[int, _Qwen38TEProjections] = {}
        self._requires_moe_checkpoint_context = False
        self._clean_gather_order_cache: dict[tuple[torch.device, int, int], torch.Tensor] = {}
        self._clean_rank_major_order_cache: dict[
            tuple[torch.device, int, int], torch.Tensor
        ] = {}
        self._gdn_boundary_plan_cache: dict[tuple[object, ...], Any] = {}
        self._active_gather_order_cache: dict[
            tuple[torch.device, int, int, int], torch.Tensor
        ] = {}
        self._active_rank_major_order_cache: dict[
            tuple[torch.device, int, int, int], torch.Tensor
        ] = {}
        self.self_conditioning = None
        self.self_condition_clean_tokens = False
        self.encoder_causal_attention = False
        self.embedding_scale = None
        self.num_hidden_layers = len(layers)

        if self.seq_len <= 0 or self.block_size <= 0 or self.seq_len % self.block_size:
            raise ValueError("seq_len must be positive and divide evenly by block_size")
        from dllm_parallel.core.kernels.gated_delta_boundaries import (
            gated_delta_boundary_chunk_size,
        )

        if self.block_size % gated_delta_boundary_chunk_size():
            raise ValueError(
                "Qwen3.8 block_size must align to the pinned FLA recurrence chunk"
            )
        if self.sequence_parallel and int(getattr(runtime, "tensor_parallel_size", 1)) <= 1:
            raise ValueError("Qwen3.8 sequence parallelism requires tensor parallelism")
        cp = int(getattr(runtime, "configured_context_parallel_size", 1))
        bp = int(getattr(runtime, "block_parallel_size", 1))
        if cp > 1 and bp not in {1, cp}:
            raise ValueError(
                "Qwen3.8 currently requires CP-only or equal-degree fused CP/BP"
            )
        from dllm_parallel.core.models.backbones.qwen3_8.tensor_parallel import (
            tag_qwen38_sequence_parallel_parameters,
        )

        if self.layers and next(self.layers[0].parameters()).device.type == "cuda":
            for layer_index, (layer, layer_type) in enumerate(
                zip(self.layers, self.layer_types, strict=True)
            ):
                packed = _build_qwen38_te_projections(
                    layer,
                    layer_index=layer_index,
                    layer_type=layer_type,
                    runtime=runtime,
                )
                self._te_packed_layers.append(packed)
                self._te_packed_by_layer_id[id(layer)] = packed
                # The packed TE projections replace the HF projections. Release
                # each source layer immediately so conversion never holds both
                # complete 27B copies on one GPU; TP1 otherwise peaks at ~79 GiB
                # before any activations or adapters are allocated.
                _release_qwen38_hf_layer_projections(
                    layer,
                    layer_type,
                    layer_index=layer_index,
                    release_norms=self.sequence_parallel,
                )
                gc.collect()
                torch.cuda.empty_cache()

        tag_qwen38_sequence_parallel_parameters(self, runtime)

    def _checkpoint_uses_reentrant_autograd(self) -> bool:
        # Collective-bearing Qwen layers must replay as one rank-identical
        # region. Reentrant checkpointing avoids non-reentrant replay stalls
        # around the FLA custom backward operators at production scale.
        return True

    def enable_gradient_boundary_probes(self, *, rank: int) -> None:
        self._gradient_boundary_probe_rank = int(rank)

    def _model_dtype(self) -> torch.dtype:
        return self.embed_tokens.weight.dtype

    def _activate_sequence_length(self, sequence_length: int) -> None:
        previous_length = int(self.seq_len)
        super()._activate_sequence_length(sequence_length)
        if int(self.seq_len) != previous_length:
            self._pure_cp_token_row_plan_cache.clear()

    def forward(
        self,
        *,
        noisy_input_ids: torch.Tensor,
        clean_input_ids: torch.Tensor,
        diffusion_times: torch.Tensor,
        noise_levels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del diffusion_times, noise_levels
        if noisy_input_ids.shape != clean_input_ids.shape:
            raise ValueError("noisy_input_ids and clean_input_ids must have equal shape")
        self._activate_sequence_length(int(noisy_input_ids.shape[1]))
        layout = self._packed_layout(noisy_input_ids.device)
        active_positions = layout.active_positions
        clean_positions = layout.clean_positions
        active_len = int(layout.active_len)
        dtype = self._model_dtype()
        active_hidden = self.embed_tokens(
            noisy_input_ids.index_select(1, active_positions)
        ).to(dtype=dtype)
        clean_hidden = self.embed_tokens(
            clean_input_ids.index_select(1, clean_positions)
        ).to(dtype=dtype)
        hidden_states = torch.cat((active_hidden, clean_hidden), dim=1)
        if (
            self.training
            and self.activation_checkpointing
            and self._adapter_checkpointing
            and not hidden_states.requires_grad
        ):
            # Reentrant checkpointing needs a gradient-bearing input to retain
            # parameter gradients when the base embedding is frozen for LoRA.
            hidden_states.requires_grad_(True)

        packed_positions = layout.packed_positions
        position_ids = packed_positions.unsqueeze(0).expand(hidden_states.shape[0], -1)
        if self.sequence_parallel:
            return self._forward_sequence_parallel(
                hidden_states=hidden_states,
                position_ids=position_ids,
                active_positions=active_positions,
                clean_positions=clean_positions,
                active_len=active_len,
                layout=layout,
            )
        for index, (layer, layer_type) in enumerate(zip(self.layers, self.layer_types, strict=True)):
            def layer_forward(
                states: torch.Tensor,
                current_layer: nn.Module = layer,
                current_type: str = layer_type,
            ) -> torch.Tensor:
                if current_type == "linear_attention":
                    return self._gdn_layer_forward(
                        current_layer,
                        states,
                        active_positions=active_positions,
                        clean_positions=clean_positions,
                        active_len=active_len,
                    )
                return self._full_attention_layer_forward(
                    current_layer,
                    states,
                    position_ids=position_ids,
                    layout=layout,
                    active_len=active_len,
                )

            if self._full_layer_checkpointing_enabled():
                hidden_states = _distributed_checkpoint(
                    layer_forward,
                    hidden_states,
                    use_reentrant=self._checkpoint_uses_reentrant_autograd(),
                )
            else:
                hidden_states = layer_forward(hidden_states)
            if self._gradient_boundary_probe_rank is not None:
                hidden_states = _attach_qwen38_layer_gradient_probe(
                    hidden_states,
                    layer_index=index,
                    rank=self._gradient_boundary_probe_rank,
                )
            self._debug_check_layer_tensor(hidden_states, index)
        hidden_states = self.norm(hidden_states)
        return hidden_states[:, :active_len], active_positions

    def _forward_sequence_parallel(
        self,
        *,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        active_positions: torch.Tensor,
        clean_positions: torch.Tensor,
        active_len: int,
        layout: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from dllm_parallel.core.parallel.tensor_parallel import (
            gather_active_from_sequence_parallel_region,
            scatter_to_sequence_parallel_region,
        )

        meta = self._sequence_parallel_meta(hidden_states, active_len=active_len)
        if _uses_te_tensor_parallel_overlap(self.runtime):
            from dllm_parallel.core.parallel.transformer_engine import (
                ensure_transformer_engine_userbuffers,
            )

            initialized = ensure_transformer_engine_userbuffers(
                rows=meta.padded_rows,
                hidden_size=meta.hidden_size,
                tensor_parallel_size=int(self.runtime.tensor_parallel_size),
                dtype=hidden_states.dtype,
            )
            if initialized and int(getattr(self.runtime, "rank", 0) or 0) == 0:
                print(
                    "Transformer Engine Userbuffers initialized: "
                    f"shape=({meta.padded_rows}, {meta.hidden_size}) "
                    f"tp={int(self.runtime.tensor_parallel_size)} "
                    f"dtype={hidden_states.dtype}",
                    flush=True,
                )
        flat = hidden_states.reshape(meta.total_rows, meta.hidden_size).contiguous()
        if meta.padded_rows != meta.total_rows:
            flat = F.pad(flat, (0, 0, 0, meta.padded_rows - meta.total_rows))
        hidden_shard = scatter_to_sequence_parallel_region(flat, self.runtime)
        persistent_plan = None
        first_packed = (
            self._te_packed_by_layer_id.get(id(self.layers[0]))
            if self.layers
            else None
        )
        if _uses_persistent_pure_cp(self.runtime, first_packed):
            persistent_plan = self._pure_cp_token_row_plan(
                meta,
                device=hidden_shard.device,
            )
            if persistent_plan is not None:
                hidden_shard = compact_pure_cp_token_rows(
                    hidden_shard,
                    persistent_plan,
                )
        for index, (layer, layer_type) in enumerate(
            zip(self.layers, self.layer_types, strict=True)
        ):
            def layer_forward(
                states: torch.Tensor,
                current_layer: nn.Module = layer,
                current_type: str = layer_type,
            ) -> torch.Tensor:
                return self._layer_forward_sequence_parallel(
                    current_layer,
                    current_type,
                    states,
                    meta=meta,
                    position_ids=position_ids,
                    layout=layout,
                    active_positions=active_positions,
                    clean_positions=clean_positions,
                    persistent_plan=persistent_plan,
                )

            if self._full_layer_checkpointing_enabled():
                hidden_shard = _distributed_checkpoint(
                    layer_forward,
                    hidden_shard,
                    use_reentrant=self._checkpoint_uses_reentrant_autograd(),
                )
            else:
                hidden_shard = layer_forward(hidden_shard)
            if self._gradient_boundary_probe_rank is not None:
                hidden_shard = _attach_qwen38_layer_gradient_probe(
                    hidden_shard,
                    layer_index=index,
                    rank=self._gradient_boundary_probe_rank,
                )
            self._debug_check_layer_tensor(hidden_shard, index)
        hidden_shard = self.norm(hidden_shard)
        if persistent_plan is not None:
            hidden_shard = reconstruct_pure_cp_token_rows(
                hidden_shard,
                plan=persistent_plan,
                group=self.runtime.context_block_parallel_group,
                context_parallel_size=int(
                    self.runtime.configured_context_parallel_size
                ),
                context_parallel_rank=int(self.runtime.context_parallel_rank),
            )
        active_hidden, active_pairs = gather_active_from_sequence_parallel_region(
            hidden_shard,
            packed_len=meta.packed_len,
            active_len=meta.active_len,
            total_rows=meta.total_rows,
            runtime=self.runtime,
        )
        original_positions = active_positions.index_select(0, active_pairs[:, 1])
        return active_hidden, torch.stack((active_pairs[:, 0], original_positions), dim=-1)

    def _layer_forward_sequence_parallel(
        self,
        layer: nn.Module,
        layer_type: str,
        hidden_shard: torch.Tensor,
        *,
        meta: _SequenceParallelMeta,
        position_ids: torch.Tensor,
        layout: Any,
        active_positions: torch.Tensor,
        clean_positions: torch.Tensor,
        persistent_plan: Any | None = None,
    ) -> torch.Tensor:
        residual = hidden_shard
        packed = self._te_packed_by_layer_id.get(id(layer))
        if packed is None:
            raise RuntimeError(
                "Qwen3.8 sequence parallelism requires packed Transformer Engine projections"
            )
        if persistent_plan is not None:
            return self._persistent_pure_cp_layer_forward_sequence_parallel(
                layer,
                layer_type,
                hidden_shard,
                meta=meta,
                position_ids=position_ids,
                layout=layout,
                clean_positions=clean_positions,
                plan=persistent_plan,
            )
        normalized = (
            hidden_shard
            if packed.mixer_input_fuses_norm
            else layer.input_layernorm(hidden_shard)
        )
        if layer_type == "linear_attention":
            mixed = self._gdn_mixed_features(
                layer,
                normalized,
                active_positions=active_positions,
                clean_positions=clean_positions,
                active_len=meta.active_len,
                total_rows=meta.total_rows,
            )
        else:
            mixed = self._full_attention_mixed_features(
                layer,
                normalized,
                position_ids=position_ids,
                layout=layout,
                active_len=meta.active_len,
                total_rows=meta.total_rows,
            )
        mixed_rows = mixed.reshape(meta.total_rows, -1)
        if meta.padded_rows != meta.total_rows:
            mixed_rows = F.pad(mixed_rows, (0, 0, 0, meta.padded_rows - meta.total_rows))
        mixed_shard = packed.mixer_output(mixed_rows)
        hidden_shard = residual + mixed_shard
        return hidden_shard + self._mlp_forward_sequence_parallel(layer, hidden_shard, meta)

    def _gather_sequence_parallel_hidden(
        self,
        hidden_shard: torch.Tensor,
        meta: _SequenceParallelMeta,
    ) -> torch.Tensor:
        from dllm_parallel.core.parallel.tensor_parallel import (
            gather_from_sequence_parallel_region,
        )

        gathered = gather_from_sequence_parallel_region(
            hidden_shard,
            self.runtime,
            total_rows=meta.total_rows,
            reduce_scatter_grad=True,
        )
        return gathered.view(meta.batch_size, meta.packed_len, meta.hidden_size)

    def _persistent_pure_cp_layer_forward_sequence_parallel(
        self,
        layer: nn.Module,
        layer_type: str,
        hidden_shard: torch.Tensor,
        *,
        meta: _SequenceParallelMeta,
        position_ids: torch.Tensor,
        layout: Any,
        clean_positions: torch.Tensor,
        plan: Any,
    ) -> torch.Tensor:
        """Execute one layer while retaining CP-owned active token rows."""

        packed = self._te_packed_by_layer_id.get(id(layer))
        if packed is None:
            raise RuntimeError("persistent pure CP requires packed TE projections")
        residual = hidden_shard
        normalized = (
            hidden_shard
            if packed.mixer_input_fuses_norm
            else layer.input_layernorm(hidden_shard)
        )
        gathered_projected = packed.mixer_input(normalized)
        logical_projected = select_pure_cp_logical_rows(
            gathered_projected,
            plan,
        ).view(meta.batch_size, plan.logical_packed_len, -1)
        if layer_type == "linear_attention":
            mixed = self._persistent_pure_cp_gdn_mixed_features(
                layer,
                logical_projected,
                clean_positions=clean_positions,
                plan=plan,
                packed=packed,
            )
        else:
            compact_positions = position_ids.gather(
                1,
                plan.logical_token_positions,
            )
            mixed = self._persistent_pure_cp_attention_mixed_features(
                layer,
                logical_projected,
                position_ids=compact_positions,
                layout=layout,
                plan=plan,
                packed=packed,
            )
        expanded_mixed = expand_pure_cp_logical_rows(
            mixed.reshape(plan.logical_rows, -1),
            plan,
        )
        mixed_shard = packed.mixer_output(expanded_mixed)
        hidden_shard = residual + mixed_shard

        def mlp_forward(states: torch.Tensor) -> torch.Tensor:
            if packed.mlp is not None:
                return packed.mlp(states)
            if packed.gate_up is None or packed.down_proj is None:
                raise RuntimeError("persistent pure CP requires packed MLP projections")
            gate_up = packed.gate_up(states)
            gate, up = gate_up.split(packed.gate_up_local_sizes, dim=-1)
            activated = layer.mlp.act_fn(gate) * up
            return packed.down_proj(activated)

        if self._mlp_checkpointing_enabled():
            mlp_output = _distributed_checkpoint(
                mlp_forward,
                hidden_shard,
                use_reentrant=self._checkpoint_uses_reentrant_autograd(),
            )
        else:
            mlp_output = mlp_forward(hidden_shard)
        return hidden_shard + mlp_output

    def _persistent_pure_cp_attention_mixed_features(
        self,
        layer: nn.Module,
        projected: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        layout: Any,
        plan: Any,
        packed: _Qwen38TEProjections,
    ) -> torch.Tensor:
        attn = layer.self_attn
        q_and_gate, key_projection, value_projection = projected.split(
            packed.mixer_input_local_sizes,
            dim=-1,
        )
        batch, tokens, _ = q_and_gate.shape
        q_and_gate = q_and_gate.view(batch, tokens, -1, 2 * attn.head_dim)
        query, gate = q_and_gate.chunk(2, dim=-1)
        query = attn.q_norm(query)
        key = attn.k_norm(
            key_projection.view(batch, tokens, -1, attn.head_dim)
        )
        value = value_projection.view(batch, tokens, -1, attn.head_dim)
        rope_positions = position_ids.unsqueeze(0).expand(3, -1, -1)
        cos, sin = self.rotary_emb(q_and_gate, rope_positions)
        apply_rotary = attn.forward.__globals__.get("apply_rotary_pos_emb")
        if not callable(apply_rotary):
            raise RuntimeError("Qwen3.8 attention does not expose its RoPE operation")
        query, key = apply_rotary(query, key, cos, sin, unsqueeze_dim=2)
        active_len = int(plan.logical_active_len)
        clean_key_shard = key[:, active_len:]
        clean_value_shard = value[:, active_len:]
        output = pure_context_persistent_block_denoising_attention_bshd(
            query=query,
            active_key_shard=key[:, :active_len],
            active_value_shard=value[:, :active_len],
            global_key_shard=clean_key_shard,
            global_value_shard=clean_value_shard,
            global_seq_len=self.seq_len,
            local_attn_mask=layout.local_attn_mask,
            global_attn_mask=layout.global_attn_mask,
            scale=float(attn.scaling),
            runtime=self.runtime,
        )
        output = output.reshape(batch, tokens, -1)
        return output * torch.sigmoid(gate.reshape(batch, tokens, -1))

    def _persistent_pure_cp_gdn_mixed_features(
        self,
        layer: nn.Module,
        projected: torch.Tensor,
        *,
        clean_positions: torch.Tensor,
        plan: Any,
        packed: _Qwen38TEProjections,
    ) -> torch.Tensor:
        from dllm_parallel.core.models.backbones.qwen3_8.context_parallel import (
            head_to_sequence_parallel,
            sequence_to_head_parallel_many,
        )

        mixer = layer.linear_attn
        projected_parts = projected.split(packed.mixer_input_local_sizes, dim=-1)
        active_len = int(plan.logical_active_len)
        active_projected = tuple(value[:, :active_len] for value in projected_parts)
        clean_projected = tuple(value[:, active_len:] for value in projected_parts)
        group = self.runtime.context_block_parallel_group
        world_size = int(self.runtime.context_attention_size)
        rank = int(self.runtime.context_parallel_rank)
        head_shard = self._gdn_head_shard(mixer, world_size=world_size, rank=rank)

        def redistribute(values: tuple[torch.Tensor, ...], order: torch.Tensor):
            qkv, z, beta, gate = values
            q_flat, k_flat, v_flat = qkv.split(
                (mixer.key_dim, mixer.key_dim, mixer.value_dim),
                dim=-1,
            )
            head_values = sequence_to_head_parallel_many(
                (q_flat, k_flat, v_flat, z, beta, gate),
                group=group,
                logical_order=order,
            )
            return (torch.cat(head_values[:3], dim=-1), *head_values[3:])

        active_head = redistribute(
            active_projected,
            self._active_gather_order(
                device=projected.device,
                world_size=world_size,
                active_len=self.seq_len,
            ),
        )
        clean_head = redistribute(
            clean_projected,
            self._clean_gather_order(
                device=projected.device,
                world_size=world_size,
                local_tokens=int(clean_positions.numel()),
            ),
        )
        all_blocks = tuple(range(self.seq_len // self.block_size))
        clean_output, recurrent, convolution = self._gdn_clean_forward(
            mixer,
            clean_head,
            all_blocks,
            head_shard=head_shard,
        )
        active_output = self._gdn_target_forward(
            mixer,
            active_head,
            recurrent_states=recurrent,
            convolution_states=convolution,
            head_shard=head_shard,
        )
        active_local = head_to_sequence_parallel(
            active_output,
            group=group,
            rank_major_order=self._active_rank_major_order(
                device=projected.device,
                world_size=world_size,
                active_len=self.seq_len,
            ),
        )
        clean_local = head_to_sequence_parallel(
            clean_output,
            group=group,
            rank_major_order=self._clean_rank_major_order(
                device=projected.device,
                world_size=world_size,
                local_tokens=int(clean_positions.numel()),
            ),
        )
        return torch.cat((active_local, clean_local), dim=1)

    def _reduce_scatter_hidden(
        self,
        hidden_states: torch.Tensor,
        meta: _SequenceParallelMeta,
    ) -> torch.Tensor:
        from dllm_parallel.core.parallel.tensor_parallel import (
            reduce_scatter_to_sequence_parallel_region,
        )

        flat = hidden_states.reshape(meta.total_rows, meta.hidden_size).contiguous()
        if meta.padded_rows != meta.total_rows:
            flat = F.pad(flat, (0, 0, 0, meta.padded_rows - meta.total_rows))
        return reduce_scatter_to_sequence_parallel_region(flat, self.runtime)

    def _mlp_forward_sequence_parallel(
        self,
        layer: nn.Module,
        hidden_shard: torch.Tensor,
        meta: _SequenceParallelMeta,
    ) -> torch.Tensor:
        packed = self._te_packed_by_layer_id.get(id(layer))
        if packed is None:
            raise RuntimeError(
                "Qwen3.8 sequence parallelism requires packed Transformer Engine projections"
            )
        token_row_plan = None
        context_parallel_size = int(
            getattr(self.runtime, "configured_context_parallel_size", 1) or 1
        )
        context_parallel_rank = int(
            getattr(self.runtime, "context_parallel_rank", 0) or 0
        )
        context_group = getattr(self.runtime, "context_block_parallel_group", None)
        if _uses_pure_cp_token_local_mlp(self.runtime, packed):
            tensor_parallel_size = int(
                getattr(self.runtime, "tensor_parallel_size", 1) or 1
            )
            tensor_parallel_rank = int(
                getattr(self.runtime, "tensor_parallel_rank", 0) or 0
            )
            plan_key = (
                hidden_shard.device,
                int(meta.batch_size),
                int(meta.packed_len),
                int(meta.active_len),
                int(self.block_size),
                tensor_parallel_size,
                tensor_parallel_rank,
                context_parallel_size,
                context_parallel_rank,
            )
            if plan_key not in self._pure_cp_token_row_plan_cache:
                self._pure_cp_token_row_plan_cache[plan_key] = (
                    build_pure_cp_token_row_plan(
                        batch_size=meta.batch_size,
                        packed_len=meta.packed_len,
                        active_len=meta.active_len,
                        block_size=self.block_size,
                        tensor_parallel_size=tensor_parallel_size,
                        tensor_parallel_rank=tensor_parallel_rank,
                        context_parallel_size=context_parallel_size,
                        context_parallel_rank=context_parallel_rank,
                        device=hidden_shard.device,
                    )
                )
            token_row_plan = self._pure_cp_token_row_plan_cache[plan_key]

        def project(states: torch.Tensor) -> torch.Tensor:
            if packed.mlp is not None:
                return packed.mlp(states)
            if packed.gate_up is None or packed.down_proj is None:
                raise RuntimeError("Qwen3.8 packed MLP projections are incomplete")
            mlp_input = (
                states
                if bool(getattr(packed.gate_up, "_dllm_lora_fuses_rmsnorm", False))
                else layer.post_attention_layernorm(states)
            )
            gate_up = packed.gate_up(mlp_input)
            gate, up = gate_up.split(packed.gate_up_local_sizes, dim=-1)
            activated = layer.mlp.act_fn(gate) * up
            return packed.down_proj(activated)

        def forward(states: torch.Tensor) -> torch.Tensor:
            if token_row_plan is not None:
                compact = compact_pure_cp_token_rows(states, token_row_plan)
                role_mask = current_lora_token_mask()
                if isinstance(role_mask, torch.Tensor):
                    flat_role_mask = role_mask.to(
                        device=states.device,
                        dtype=torch.bool,
                    ).reshape(-1)
                    if int(flat_role_mask.numel()) != int(states.shape[0]):
                        raise ValueError(
                            "pure-CP LoRA token mask must match pre-compaction rows: "
                            f"{int(flat_role_mask.numel())} != {int(states.shape[0])}"
                        )
                    compact_role_mask = compact_pure_cp_token_rows(
                        flat_role_mask[:, None],
                        token_row_plan,
                    ).squeeze(-1)
                else:
                    compact_role_mask = role_mask
                with lora_token_mask(compact_role_mask):
                    compact_output = project(compact)
                return reconstruct_pure_cp_token_rows(
                    compact_output,
                    plan=token_row_plan,
                    group=context_group,
                    context_parallel_size=context_parallel_size,
                    context_parallel_rank=context_parallel_rank,
                )
            return project(states)

        if self._mlp_checkpointing_enabled():
            return _distributed_checkpoint(
                forward,
                hidden_shard,
                use_reentrant=self._checkpoint_uses_reentrant_autograd(),
            )
        return forward(hidden_shard)

    def _pure_cp_token_row_plan(
        self,
        meta: _SequenceParallelMeta,
        *,
        device: torch.device,
    ) -> Any | None:
        tensor_parallel_size = int(
            getattr(self.runtime, "tensor_parallel_size", 1) or 1
        )
        tensor_parallel_rank = int(
            getattr(self.runtime, "tensor_parallel_rank", 0) or 0
        )
        context_parallel_size = int(
            getattr(self.runtime, "configured_context_parallel_size", 1) or 1
        )
        context_parallel_rank = int(
            getattr(self.runtime, "context_parallel_rank", 0) or 0
        )
        plan_key = (
            device,
            int(meta.batch_size),
            int(meta.packed_len),
            int(meta.active_len),
            int(self.block_size),
            tensor_parallel_size,
            tensor_parallel_rank,
            context_parallel_size,
            context_parallel_rank,
        )
        if plan_key not in self._pure_cp_token_row_plan_cache:
            self._pure_cp_token_row_plan_cache[plan_key] = build_pure_cp_token_row_plan(
                batch_size=meta.batch_size,
                packed_len=meta.packed_len,
                active_len=meta.active_len,
                block_size=self.block_size,
                tensor_parallel_size=tensor_parallel_size,
                tensor_parallel_rank=tensor_parallel_rank,
                context_parallel_size=context_parallel_size,
                context_parallel_rank=context_parallel_rank,
                device=device,
            )
        return self._pure_cp_token_row_plan_cache[plan_key]

    def _sequence_parallel_meta(
        self,
        hidden_states: torch.Tensor,
        *,
        active_len: int,
    ) -> _SequenceParallelMeta:
        batch_size, packed_len, hidden_size = map(int, hidden_states.shape)
        total_rows = batch_size * packed_len
        tp_size = int(getattr(self.runtime, "tensor_parallel_size", 1) or 1)
        padded_rows = ((total_rows + tp_size - 1) // tp_size) * tp_size
        return _SequenceParallelMeta(
            batch_size=batch_size,
            packed_len=packed_len,
            hidden_size=hidden_size,
            total_rows=total_rows,
            padded_rows=padded_rows,
            active_len=int(active_len),
        )

    def _full_layer_checkpointing_enabled(self) -> bool:
        return bool(
            self.training
            and self.activation_checkpointing
            and self.activation_checkpointing_scope == "full"
        )

    def _mlp_checkpointing_enabled(self) -> bool:
        return bool(
            self.training
            and self.activation_checkpointing
            and self.activation_checkpointing_scope == "mlp"
        )

    def _full_attention_layer_forward(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        layout: Any,
        active_len: int,
    ) -> torch.Tensor:
        residual = hidden_states
        states = layer.input_layernorm(hidden_states)
        output = self._full_attention_mixed_features(
            layer,
            states,
            position_ids=position_ids,
            layout=layout,
            active_len=active_len,
        )
        packed = self._te_packed_by_layer_id.get(id(layer))
        hidden_states = residual + (
            packed.mixer_output(output)
            if packed is not None
            else self._row_parallel_linear(layer.self_attn.o_proj, output)
        )
        return hidden_states + self._mlp_forward(layer, hidden_states)

    def _full_attention_mixed_features(
        self,
        layer: nn.Module,
        states: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        layout: Any,
        active_len: int,
        total_rows: int | None = None,
    ) -> torch.Tensor:
        attn = layer.self_attn
        packed = self._te_packed_by_layer_id.get(id(layer))
        if packed is not None:
            projected = packed.mixer_input(states)
            if states.ndim == 2:
                if total_rows is None:
                    raise ValueError("sequence-parallel attention requires total_rows")
                projected = projected[: int(total_rows)].view(
                    int(position_ids.shape[0]),
                    int(position_ids.shape[1]),
                    -1,
                )
            q_and_gate, key_projection, value_projection = projected.split(
                packed.mixer_input_local_sizes,
                dim=-1,
            )
        else:
            states = self._column_parallel_input(states)
            q_and_gate = F.linear(states, attn.q_proj.weight, attn.q_proj.bias)
            key_projection = F.linear(states, attn.k_proj.weight, attn.k_proj.bias)
            value_projection = F.linear(states, attn.v_proj.weight, attn.v_proj.bias)
        batch, tokens, _ = q_and_gate.shape
        q_and_gate = q_and_gate.view(batch, tokens, -1, 2 * attn.head_dim)
        query, gate = q_and_gate.chunk(2, dim=-1)
        query = attn.q_norm(query)
        key = attn.k_norm(
            key_projection.view(batch, tokens, -1, attn.head_dim)
        )
        value = value_projection.view(batch, tokens, -1, attn.head_dim)
        rope_positions = position_ids.unsqueeze(0).expand(3, -1, -1)
        cos, sin = self.rotary_emb(q_and_gate, rope_positions)
        apply_rotary = attn.forward.__globals__.get("apply_rotary_pos_emb")
        if not callable(apply_rotary):
            raise RuntimeError("Qwen3.8 attention does not expose its RoPE operation")
        query, key = apply_rotary(query, key, cos, sin, unsqueeze_dim=2)
        local_mask = layout.local_attn_mask
        global_mask = layout.global_attn_mask
        if self.runtime.uses_context_parallel_attention:
            output = _qwen38_context_attention(self.runtime)(
                query=query,
                local_key=key[:, :active_len],
                local_value=value[:, :active_len],
                global_key_shard=key[:, active_len:],
                global_value_shard=value[:, active_len:],
                global_seq_len=self.seq_len,
                local_attn_mask=local_mask,
                global_attn_mask=global_mask,
                scale=float(attn.scaling),
                runtime=self.runtime,
            )
        else:
            output = replicated_block_denoising_attention_bshd(
                query=query,
                local_key=key[:, :active_len],
                local_value=value[:, :active_len],
                global_key=key[:, active_len:],
                global_value=value[:, active_len:],
                local_attn_mask=local_mask,
                global_attn_mask=global_mask,
                scale=float(attn.scaling),
            )
        output = output.reshape(batch, tokens, -1)
        return output * torch.sigmoid(gate.reshape(batch, tokens, -1))

    def _mlp_forward(self, layer: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        def forward_chunk(states: torch.Tensor) -> torch.Tensor:
            states = layer.post_attention_layernorm(states)
            packed = self._te_packed_by_layer_id.get(id(layer))
            if packed is not None:
                gate_up = packed.gate_up(states)
                gate, up = gate_up.split(packed.gate_up_local_sizes, dim=-1)
                return packed.down_proj(layer.mlp.act_fn(gate) * up)
            states = self._column_parallel_input(states)
            mlp = layer.mlp
            gate = F.linear(states, mlp.gate_proj.weight, mlp.gate_proj.bias)
            up = F.linear(states, mlp.up_proj.weight, mlp.up_proj.bias)
            partial = F.linear(
                mlp.act_fn(gate) * up,
                mlp.down_proj.weight,
                None,
            )
            return self._row_parallel_output(partial, bias=mlp.down_proj.bias)

        def forward(states: torch.Tensor) -> torch.Tensor:
            token_chunk_size = int(self.mlp_token_chunk_size or 0)
            if token_chunk_size <= 0 or states.numel() == 0:
                return forward_chunk(states)
            original_shape = tuple(states.shape)
            hidden_size = int(original_shape[-1])
            flat_states = states.reshape(-1, hidden_size)
            if int(flat_states.shape[0]) <= token_chunk_size:
                return forward_chunk(states)
            role_mask = current_lora_token_mask()
            flat_role_mask = (
                role_mask.reshape(-1)
                if isinstance(role_mask, torch.Tensor)
                else None
            )
            outputs = []
            offset = 0
            for chunk in flat_states.split(token_chunk_size, dim=0):
                rows = int(chunk.shape[0])
                chunk_mask = (
                    flat_role_mask[offset : offset + rows]
                    if flat_role_mask is not None
                    else role_mask
                )
                with lora_token_mask(chunk_mask):
                    outputs.append(forward_chunk(chunk))
                offset += rows
            return torch.cat(outputs, dim=0).reshape(original_shape)

        if self._mlp_checkpointing_enabled():
            return _distributed_checkpoint(
                forward,
                hidden_states,
                use_reentrant=self._checkpoint_uses_reentrant_autograd(),
            )
        return forward(hidden_states)

    def _gdn_layer_forward(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        *,
        active_positions: torch.Tensor,
        clean_positions: torch.Tensor,
        active_len: int,
    ) -> torch.Tensor:
        residual = hidden_states
        states = layer.input_layernorm(hidden_states)
        mixed = self._gdn_mixed_features(
            layer,
            states,
            active_positions=active_positions,
            clean_positions=clean_positions,
            active_len=active_len,
        )
        packed = self._te_packed_by_layer_id.get(id(layer))
        hidden_states = residual + (
            packed.mixer_output(mixed)
            if packed is not None
            else self._row_parallel_linear(layer.linear_attn.out_proj, mixed)
        )
        return hidden_states + self._mlp_forward(layer, hidden_states)

    def _gdn_mixed_features(
        self,
        layer: nn.Module,
        states: torch.Tensor,
        *,
        active_positions: torch.Tensor,
        clean_positions: torch.Tensor,
        active_len: int,
        total_rows: int | None = None,
    ) -> torch.Tensor:
        del active_positions
        mixer = layer.linear_attn
        packed = self._te_packed_by_layer_id.get(id(layer))
        if states.ndim == 2:
            if packed is None or total_rows is None:
                raise ValueError("sequence-parallel GDN requires packed projections")
            projected_tensor = packed.mixer_input(states)[: int(total_rows)].view(
                -1,
                int(active_len + clean_positions.numel()),
                sum(packed.mixer_input_local_sizes),
            )
            active_projected_tensor = projected_tensor[:, :active_len]
            local_clean_projected_tensor = projected_tensor[:, active_len:]
            local_projected = local_clean_projected_tensor.split(
                packed.mixer_input_local_sizes,
                dim=-1,
            )
            active_projected = active_projected_tensor.split(
                packed.mixer_input_local_sizes,
                dim=-1,
            )
        else:
            active_states = states[:, :active_len]
            local_clean_states = states[:, active_len:]
            active_projected = self._gdn_project(
                mixer,
                active_states,
                packed=packed,
            )
            local_projected = self._gdn_project(
                mixer,
                local_clean_states,
                packed=packed,
            )
        branch_ids = active_blocks_for_runtime(
            seq_len=self.seq_len,
            block_size=self.block_size,
            runtime=self.runtime,
        )
        if branch_ids is None:
            branch_ids = tuple(range(self.seq_len // self.block_size))
        if self.runtime.uses_context_parallel_attention:
            (
                local_clean_output,
                recurrent_states,
                convolution_states,
            ) = self._gdn_context_parallel_clean_forward(
                mixer,
                local_projected,
                clean_positions=clean_positions,
                branch_ids=branch_ids,
            )
        else:
            (
                local_clean_output,
                recurrent_states,
                convolution_states,
            ) = self._gdn_clean_forward(mixer, local_projected, branch_ids)
        active_output = self._gdn_target_forward(
            mixer,
            active_projected,
            recurrent_states=recurrent_states,
            convolution_states=convolution_states,
        )
        return torch.cat((active_output, local_clean_output), dim=1)

    def _gdn_context_parallel_clean_forward(
        self,
        mixer: nn.Module,
        local_projected: tuple[torch.Tensor, ...],
        *,
        clean_positions: torch.Tensor,
        branch_ids: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from dllm_parallel.core.models.backbones.qwen3_8.context_parallel import (
            gather_boundary_heads,
            head_to_sequence_parallel,
            route_boundary_heads,
            sequence_to_head_parallel_many,
        )

        group = self.runtime.context_block_parallel_group
        world_size = int(self.runtime.context_attention_size)
        rank = int(self.runtime.context_parallel_rank)
        if dist.get_world_size(group) != world_size or dist.get_rank(group) != rank:
            raise RuntimeError("Qwen3.8 context group metadata is inconsistent")
        local_tokens = int(local_projected[0].shape[1])
        if local_tokens * world_size != self.seq_len:
            raise ValueError("Qwen3.8 clean shards must evenly cover the clean sequence")
        if clean_positions.numel() != local_tokens:
            raise ValueError("Qwen3.8 clean-position metadata does not match its local shard")
        head_shard = self._gdn_head_shard(mixer, world_size=world_size, rank=rank)
        logical_order = self._clean_gather_order(
            device=local_projected[0].device,
            world_size=world_size,
            local_tokens=local_tokens,
        )
        rank_major_order = self._clean_rank_major_order(
            device=local_projected[0].device,
            world_size=world_size,
            local_tokens=local_tokens,
        )

        qkv, z, beta, gate = local_projected
        q_flat, k_flat, v_flat = qkv.split(
            (mixer.key_dim, mixer.key_dim, mixer.value_dim),
            dim=-1,
        )
        head_projected = sequence_to_head_parallel_many(
            (q_flat, k_flat, v_flat, z, beta, gate),
            group=group,
            logical_order=logical_order,
        )
        clean_projected = (
            torch.cat(head_projected[:3], dim=-1),
            *head_projected[3:],
        )
        all_blocks = tuple(range(self.seq_len // self.block_size))
        clean_output, recurrent, convolution = self._gdn_clean_forward(
            mixer,
            clean_projected,
            all_blocks,
            head_shard=head_shard,
        )
        local_clean_output = head_to_sequence_parallel(
            clean_output,
            group=group,
            rank_major_order=rank_major_order,
        )

        batch_size = int(clean_output.shape[0])
        block_count = len(all_blocks)
        value_heads = head_shard.value_stop - head_shard.value_start
        recurrent = recurrent.view(
            batch_size,
            block_count,
            value_heads,
            mixer.head_k_dim,
            mixer.head_v_dim,
        )
        convolution = convolution.view(
            batch_size,
            block_count,
            convolution.shape[-2],
            convolution.shape[-1],
        )
        key_width = (head_shard.key_stop - head_shard.key_start) * mixer.head_k_dim
        value_width = value_heads * mixer.head_v_dim
        conv_q, conv_k, conv_v = convolution.split(
            (key_width, key_width, value_width),
            dim=2,
        )

        if int(self.runtime.block_parallel_size) > 1:
            blocks_by_rank = self._gdn_blocks_by_rank(world_size)

            def route(value: torch.Tensor) -> torch.Tensor:
                return route_boundary_heads(
                    value,
                    group=group,
                    blocks_by_rank=blocks_by_rank,
                    local_blocks=branch_ids,
                    block_axis=1,
                    head_axis=2,
                )
        else:

            def route(value: torch.Tensor) -> torch.Tensor:
                return gather_boundary_heads(
                    value,
                    group=group,
                    head_axis=2,
                )
        recurrent = route(recurrent)
        convolution = torch.cat((route(conv_q), route(conv_k), route(conv_v)), dim=2)
        return (
            local_clean_output,
            recurrent.flatten(0, 1),
            convolution.flatten(0, 1),
        )

    def _gdn_head_shard(
        self,
        mixer: nn.Module,
        *,
        world_size: int,
        rank: int,
    ) -> _GDNHeadShard:
        key_heads = int(mixer.num_k_heads)
        value_heads = int(mixer.num_v_heads)
        if key_heads % world_size or value_heads % world_size:
            raise ValueError(
                "Qwen3.8 GDN context parallelism requires local recurrent heads "
                "to divide evenly across context ranks"
            )
        key_per_rank = key_heads // world_size
        value_per_rank = value_heads // world_size
        return _GDNHeadShard(
            key_start=rank * key_per_rank,
            key_stop=(rank + 1) * key_per_rank,
            value_start=rank * value_per_rank,
            value_stop=(rank + 1) * value_per_rank,
        )

    def _gdn_blocks_by_rank(
        self,
        world_size: int,
    ) -> tuple[tuple[int, ...], ...]:
        from dllm_parallel.core.schedules import build_block_schedule

        schedule = build_block_schedule(
            num_blocks=self.seq_len // self.block_size,
            block_parallel_size=int(self.runtime.block_parallel_size),
            context_parallel_size=int(self.runtime.configured_context_parallel_size),
        )
        if len(schedule.active_blocks_by_worker) != world_size:
            raise ValueError("Qwen3.8 fused GDN requires equal CP and BP degrees")
        return tuple(
            tuple(sorted(int(block) for block in blocks))
            for blocks in schedule.active_blocks_by_worker
        )

    def _clean_gather_order(
        self,
        *,
        device: torch.device,
        world_size: int,
        local_tokens: int,
    ) -> torch.Tensor:
        key = (device, int(world_size), int(local_tokens))
        cached = self._clean_gather_order_cache.get(key)
        if cached is not None:
            return cached
        from dllm_parallel.core.attention.cp_backend import clean_shards_for_rank
        from dllm_parallel.core.attention.layout import runtime_clean_layout

        gathered_positions: list[int] = []
        for rank in range(int(world_size)):
            shards = clean_shards_for_rank(
                seq_len=self.seq_len,
                context_parallel_size=int(world_size),
                rank=rank,
                layout=runtime_clean_layout(self.runtime),
            )
            for shard in shards:
                gathered_positions.extend(range(int(shard.start), int(shard.stop)))
        if len(gathered_positions) != int(world_size) * int(local_tokens):
            raise ValueError("Qwen3.8 clean layout does not match the gathered shard extent")
        order_list = sorted(range(len(gathered_positions)), key=gathered_positions.__getitem__)
        if [gathered_positions[index] for index in order_list] != list(range(self.seq_len)):
            raise ValueError("Qwen3.8 clean layout must partition every sequence position once")
        order = torch.tensor(order_list, device=device, dtype=torch.long)
        self._clean_gather_order_cache[key] = order
        return order

    def _clean_rank_major_order(
        self,
        *,
        device: torch.device,
        world_size: int,
        local_tokens: int,
    ) -> torch.Tensor:
        key = (device, int(world_size), int(local_tokens))
        cached = self._clean_rank_major_order_cache.get(key)
        if cached is not None:
            return cached
        logical_order = self._clean_gather_order(
            device=device,
            world_size=world_size,
            local_tokens=local_tokens,
        )
        rank_major_order = torch.argsort(logical_order)
        self._clean_rank_major_order_cache[key] = rank_major_order
        return rank_major_order

    def _active_gather_order(
        self,
        *,
        device: torch.device,
        world_size: int,
        active_len: int,
    ) -> torch.Tensor:
        key = (device, int(world_size), int(active_len), int(self.block_size))
        cached = self._active_gather_order_cache.get(key)
        if cached is not None:
            return cached
        owner_major = torch.cat(
            tuple(
                active_query_indices_for_context_rank(
                    active_len=int(active_len),
                    block_size=int(self.block_size),
                    context_parallel_size=int(world_size),
                    context_parallel_rank=rank,
                    device=device,
                )
                for rank in range(int(world_size))
            )
        )
        logical_order = torch.argsort(owner_major)
        self._active_gather_order_cache[key] = logical_order
        return logical_order

    def _active_rank_major_order(
        self,
        *,
        device: torch.device,
        world_size: int,
        active_len: int,
    ) -> torch.Tensor:
        key = (device, int(world_size), int(active_len), int(self.block_size))
        cached = self._active_rank_major_order_cache.get(key)
        if cached is not None:
            return cached
        rank_major_order = torch.argsort(
            self._active_gather_order(
                device=device,
                world_size=world_size,
                active_len=active_len,
            )
        )
        self._active_rank_major_order_cache[key] = rank_major_order
        return rank_major_order

    def _gdn_clean_forward(
        self,
        mixer: nn.Module,
        projected: tuple[torch.Tensor, ...],
        branch_ids: tuple[int, ...],
        *,
        head_shard: _GDNHeadShard | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, clean_length = map(int, projected[0].shape[:2])
        if clean_length % self.block_size:
            raise ValueError("Qwen3.8 clean sequence must contain complete blocks")
        if len(set(branch_ids)) != len(branch_ids):
            raise ValueError("Qwen3.8 target blocks must have unique owners per rank")

        q, k, v, z, beta, gate, _ = self._gdn_convolve(
            mixer,
            projected,
            head_shard=head_shard,
        )
        convolution = _causal_convolution_boundary_states(
            projected[0],
            boundary_tokens=tuple(block_id * self.block_size for block_id in branch_ids),
            state_width=int(mixer.conv1d.weight.shape[-1]),
        )

        block_count = clean_length // self.block_size
        if any(block_id < 0 or block_id > block_count for block_id in branch_ids):
            raise ValueError("Qwen3.8 target block lies outside the clean sequence")
        nonzero_blocks = tuple(block_id for block_id in branch_ids if block_id != 0)
        if nonzero_blocks:
            from dllm_parallel.core.kernels.gated_delta_boundaries import (
                chunk_gated_delta_rule_with_boundaries,
            )

            plan = self._gdn_boundary_plan(
                clean_length=clean_length,
                block_ids=nonzero_blocks,
                device=q.device,
            )
            value_slice = (
                slice(None)
                if head_shard is None
                else slice(head_shard.value_start, head_shard.value_stop)
            )
            clean_output, nonzero_states = chunk_gated_delta_rule_with_boundaries(
                q,
                k,
                v,
                gate,
                beta,
                plan=plan,
                A_log=mixer.A_log[value_slice],
                dt_bias=mixer.dt_bias[value_slice],
                scale=1.0 / math.sqrt(mixer.head_k_dim),
            )
        else:
            clean_output, _ = _chunk_gated_delta_rule(
                q,
                k,
                v,
                gate,
                beta,
                mixer=mixer,
                head_shard=head_shard,
            )
            nonzero_states = None

        zero_state = q.new_zeros(
            batch_size,
            int(v.shape[2]),
            mixer.head_k_dim,
            mixer.head_v_dim,
            dtype=torch.float32,
        )
        # ``nonzero_states`` already follows branch order with the zero
        # boundary omitted. Inserting it as one tensor operation avoids more
        # than a thousand per-slice FP32 autograd adds per GDN layer at 256K.
        recurrent = _ordered_gdn_boundary_states(
            nonzero_states,
            zero_state,
            branch_ids,
        ).reshape(
            batch_size * len(branch_ids),
            *zero_state.shape[1:],
        )
        clean_output = mixer.norm(
            clean_output.reshape(-1, mixer.head_v_dim),
            z.reshape(-1, mixer.head_v_dim),
        ).reshape_as(z).flatten(2)
        return clean_output, recurrent, convolution

    def _gdn_boundary_plan(
        self,
        *,
        clean_length: int,
        block_ids: tuple[int, ...],
        device: torch.device,
    ) -> Any:
        key = (device, int(clean_length), tuple(int(value) for value in block_ids))
        cached = self._gdn_boundary_plan_cache.get(key)
        if cached is not None:
            return cached
        from dllm_parallel.core.kernels.gated_delta_boundaries import (
            build_gated_delta_boundary_plan,
        )

        plan = build_gated_delta_boundary_plan(
            tokens=int(clean_length),
            boundary_tokens=tuple(int(value) * self.block_size for value in block_ids),
            device=device,
        )
        self._gdn_boundary_plan_cache[key] = plan
        return plan

    def _gdn_target_forward(
        self,
        mixer: nn.Module,
        active_projected: tuple[torch.Tensor, ...],
        *,
        recurrent_states: torch.Tensor,
        convolution_states: torch.Tensor,
        head_shard: _GDNHeadShard | None = None,
    ) -> torch.Tensor:
        batch_size, active_length = map(int, active_projected[0].shape[:2])
        if active_length % self.block_size:
            raise ValueError("Qwen3.8 target rows must contain complete blocks")
        branches = active_length // self.block_size
        projected = tuple(
            value.reshape(batch_size * branches, self.block_size, value.shape[-1])
            for value in active_projected
        )
        branch_batch = batch_size * branches
        configured_chunk_size = getattr(self, "_gdn_target_branch_chunk_size", None)
        chunk_size = (
            branch_batch
            if configured_chunk_size is None
            else max(1, int(configured_chunk_size))
        )
        outputs = []
        for start in range(0, branch_batch, chunk_size):
            stop = min(start + chunk_size, branch_batch)
            projected_chunk = tuple(value[start:stop] for value in projected)
            q, k, v, z, beta, gate, _ = self._gdn_convolve(
                mixer,
                projected_chunk,
                head_shard=head_shard,
                initial_state=convolution_states[start:stop],
            )
            output, _ = _chunk_gated_delta_rule(
                q,
                k,
                v,
                gate,
                beta,
                mixer=mixer,
                head_shard=head_shard,
                initial_state=recurrent_states[start:stop],
            )
            outputs.append(
                mixer.norm(
                    output.reshape(-1, mixer.head_v_dim),
                    z.reshape(-1, mixer.head_v_dim),
                ).reshape(stop - start, self.block_size, -1)
            )
        return torch.cat(outputs, dim=0).reshape(
            batch_size,
            branches * self.block_size,
            -1,
        )

    def _gdn_project(
        self,
        mixer: nn.Module,
        states: torch.Tensor,
        *,
        packed: _Qwen38TEProjections | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if packed is not None:
            return packed.mixer_input(states).split(
                packed.mixer_input_local_sizes,
                dim=-1,
            )
        states = self._column_parallel_input(states)
        return (
            F.linear(states, mixer.in_proj_qkv.weight, mixer.in_proj_qkv.bias),
            F.linear(states, mixer.in_proj_z.weight, mixer.in_proj_z.bias),
            F.linear(states, mixer.in_proj_b.weight, mixer.in_proj_b.bias),
            F.linear(states, mixer.in_proj_a.weight, mixer.in_proj_a.bias),
        )

    def _column_parallel_input(self, states: torch.Tensor) -> torch.Tensor:
        if self.sequence_parallel or int(
            getattr(self.runtime, "tensor_parallel_size", 1) or 1
        ) <= 1:
            return states
        from dllm_parallel.core.parallel.tensor_parallel import (
            copy_to_tensor_parallel_region,
        )

        return copy_to_tensor_parallel_region(states, self.runtime)

    def _row_parallel_linear(
        self,
        module: nn.Module,
        states: torch.Tensor,
    ) -> torch.Tensor:
        partial = F.linear(states, module.weight, None)
        return self._row_parallel_output(partial, bias=module.bias)

    def _row_parallel_output(
        self,
        partial: torch.Tensor,
        *,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        if int(getattr(self.runtime, "tensor_parallel_size", 1) or 1) > 1:
            from dllm_parallel.core.parallel.tensor_parallel import (
                reduce_from_tensor_parallel_region,
            )

            partial = reduce_from_tensor_parallel_region(partial, self.runtime)
        return partial if bias is None else partial + bias

    @staticmethod
    def _gdn_convolve(
        mixer: nn.Module,
        projected: tuple[torch.Tensor, ...],
        *,
        head_shard: _GDNHeadShard | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_cpu: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        qkv, z, beta, gate = projected
        try:
            from fla.modules.conv.causal_conv1d import causal_conv1d
        except ImportError as exc:
            raise RuntimeError("Qwen3.8 requires flash-linear-attention") from exc
        conv_weight = mixer.conv1d.weight.squeeze(1)
        conv_bias = mixer.conv1d.bias
        num_k_heads = int(mixer.num_k_heads)
        num_v_heads = int(mixer.num_v_heads)
        key_dim = int(mixer.key_dim)
        value_dim = int(mixer.value_dim)
        if head_shard is not None:
            key_start = head_shard.key_start * int(mixer.head_k_dim)
            key_stop = head_shard.key_stop * int(mixer.head_k_dim)
            value_start = head_shard.value_start * int(mixer.head_v_dim)
            value_stop = head_shard.value_stop * int(mixer.head_v_dim)
            q_weight, k_weight, v_weight = conv_weight.split(
                (key_dim, key_dim, value_dim),
                dim=0,
            )
            conv_weight = torch.cat(
                (
                    q_weight[key_start:key_stop],
                    k_weight[key_start:key_stop],
                    v_weight[value_start:value_stop],
                ),
                dim=0,
            ).contiguous()
            if conv_bias is not None:
                q_bias, k_bias, v_bias = conv_bias.split(
                    (key_dim, key_dim, value_dim),
                    dim=0,
                )
                conv_bias = torch.cat(
                    (
                        q_bias[key_start:key_stop],
                        k_bias[key_start:key_stop],
                        v_bias[value_start:value_stop],
                    ),
                    dim=0,
                ).contiguous()
            num_k_heads = head_shard.key_stop - head_shard.key_start
            num_v_heads = head_shard.value_stop - head_shard.value_start
            key_dim = num_k_heads * int(mixer.head_k_dim)
            value_dim = num_v_heads * int(mixer.head_v_dim)

        qkv, final_state = causal_conv1d(
            x=qkv,
            weight=conv_weight,
            bias=conv_bias,
            activation=mixer.activation,
            backend=(
                "triton"
                if initial_state is not None or output_final_state
                else "cuda"
            ),
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
        )
        q, k, v = qkv.split((key_dim, key_dim, value_dim), dim=-1)
        batch, tokens = q.shape[:2]
        q = q.view(batch, tokens, num_k_heads, mixer.head_k_dim)
        k = k.view(batch, tokens, num_k_heads, mixer.head_k_dim)
        v = v.view(batch, tokens, num_v_heads, mixer.head_v_dim)
        z = z.view(batch, tokens, num_v_heads, mixer.head_v_dim)
        return q, k, v, z, beta, gate, final_state


def _chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    *,
    mixer: nn.Module,
    head_shard: _GDNHeadShard | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    except ImportError as exc:
        raise RuntimeError("Qwen3.8 requires flash-linear-attention") from exc
    value_slice = (
        slice(None)
        if head_shard is None
        else slice(head_shard.value_start, head_shard.value_stop)
    )
    output, final_state = chunk_gated_delta_rule(
        query,
        key,
        value,
        gate,
        beta,
        scale=1.0 / math.sqrt(mixer.head_k_dim),
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        A_log=mixer.A_log[value_slice],
        dt_bias=mixer.dt_bias[value_slice],
        use_beta_sigmoid_in_kernel=True,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
    )
    return output, final_state


def _causal_convolution_boundary_states(
    inputs: torch.Tensor,
    *,
    boundary_tokens: tuple[int, ...],
    state_width: int,
) -> torch.Tensor:
    """Extract exact causal-convolution caches at selected token boundaries."""

    if inputs.ndim != 3:
        raise ValueError("causal-convolution inputs must have shape [batch, tokens, channels]")
    if int(state_width) <= 0:
        raise ValueError("causal-convolution state width must be positive")
    batch_size, sequence_length, channels = map(int, inputs.shape)
    if not boundary_tokens:
        return inputs.new_empty(0, channels, int(state_width))
    if min(boundary_tokens) < 0 or max(boundary_tokens) > sequence_length:
        raise ValueError("causal-convolution boundary lies outside the clean sequence")

    boundaries = torch.tensor(boundary_tokens, device=inputs.device, dtype=torch.long)
    history = torch.arange(-int(state_width), 0, device=inputs.device)
    indices = boundaries[:, None] + history[None, :]
    valid = indices >= 0
    gathered = inputs.index_select(1, indices.clamp_min(0).reshape(-1))
    gathered = gathered.view(batch_size, len(boundary_tokens), int(state_width), channels)
    gathered = gathered * valid[None, :, :, None]
    return gathered.permute(0, 1, 3, 2).reshape(
        batch_size * len(boundary_tokens),
        channels,
        int(state_width),
    )


def _build_qwen38_te_projections(
    layer: nn.Module,
    *,
    layer_index: int,
    layer_type: str,
    runtime: Any,
) -> _Qwen38TEProjections:
    """Pack one Qwen3.8 layer into TE's native TP/SP linear operators."""

    te_linear = _transformer_engine_linear_cls()
    sequence_parallel = bool(getattr(runtime, "sequence_parallel", False))
    tp_overlap = _uses_te_tensor_parallel_overlap(runtime)
    dtype = _first_linear_dtype(layer)
    device = _first_linear_device(layer)
    prefix = f"qwen38_layer_{int(layer_index)}"

    if layer_type == "full_attention":
        mixer = layer.self_attn
        mixer_inputs = (
            ("query_gate", mixer.q_proj),
            ("key", mixer.k_proj),
            ("value", mixer.v_proj),
        )
        mixer_output_source = mixer.o_proj
    elif layer_type == "linear_attention":
        mixer = layer.linear_attn
        mixer_inputs = (
            ("qkv", mixer.in_proj_qkv),
            ("z", mixer.in_proj_z),
            ("beta", mixer.in_proj_b),
            ("gate", mixer.in_proj_a),
        )
        mixer_output_source = mixer.out_proj
    else:
        raise ValueError(f"unsupported Qwen3.8 layer type {layer_type!r}")

    if sequence_parallel:
        mixer_input = _build_qwen38_te_layernorm_column_linear(
            layer.input_layernorm,
            *mixer_inputs,
            runtime=runtime,
            dtype=dtype,
            device=device,
            name=f"{prefix}_mixer_input",
        )
    else:
        mixer_input = _build_te_fused_column_linear(
            te_linear,
            *mixer_inputs,
            runtime=runtime,
            dtype=dtype,
            device=device,
            sequence_parallel=False,
            name=f"{prefix}_mixer_input",
        )
    mixer_output = _build_te_row_linear(
        te_linear,
        mixer_output_source,
        runtime=runtime,
        dtype=dtype,
        device=device,
        sequence_parallel=sequence_parallel,
        name=f"{prefix}_mixer_output",
        ub_overlap_ag=tp_overlap,
        ub_overlap_rs=tp_overlap,
        ub_name="proj" if tp_overlap else None,
    )
    mixer_input._dllm_lora_parallel_mode = "column"
    mixer_output._dllm_lora_parallel_mode = "row"
    if sequence_parallel:
        mlp = _build_qwen38_te_layernorm_mlp(
            layer,
            runtime=runtime,
            dtype=dtype,
            device=device,
            name=f"{prefix}_mlp",
        )
        gate_up = None
        gate_up_local_sizes = (0, 0)
        down_proj = None
    else:
        mlp = None
        gate_up = _build_te_fused_column_linear(
            te_linear,
            ("gate", layer.mlp.gate_proj),
            ("up", layer.mlp.up_proj),
            runtime=runtime,
            dtype=dtype,
            device=device,
            sequence_parallel=False,
            name=f"{prefix}_gate_up",
        )
        gate_up_local_sizes = _local_split_sizes(gate_up, ("gate", "up"))
        down_proj = _build_te_row_linear(
            te_linear,
            layer.mlp.down_proj,
            runtime=runtime,
            dtype=dtype,
            device=device,
            sequence_parallel=False,
            name=f"{prefix}_down_proj",
        )
    if gate_up is not None:
        gate_up._dllm_lora_parallel_mode = "column"
    if down_proj is not None:
        down_proj._dllm_lora_parallel_mode = "row"
    return _Qwen38TEProjections(
        mixer_input=mixer_input,
        mixer_input_local_sizes=_local_split_sizes(
            mixer_input,
            tuple(name for name, _ in mixer_inputs),
        ),
        mixer_input_fuses_norm=sequence_parallel,
        mixer_output=mixer_output,
        gate_up=gate_up,
        gate_up_local_sizes=gate_up_local_sizes,
        down_proj=down_proj,
        mlp=mlp,
    )


def _build_qwen38_te_layernorm_column_linear(
    norm: nn.Module,
    *named_modules: tuple[str, Any],
    runtime: Any,
    dtype: torch.dtype,
    device: torch.device,
    name: str,
) -> nn.Module:
    """Fuse Qwen's input RMSNorm with its packed TP/SP mixer projection."""

    try:
        from transformer_engine.pytorch import LayerNormLinear
    except Exception as exc:  # pragma: no cover - depends on runtime image.
        raise RuntimeError(
            "Qwen3.8 TP/SP training requires transformer_engine.pytorch.LayerNormLinear"
        ) from exc

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
    if any(getattr(module, "bias", None) is not None for _, module in named_modules):
        raise RuntimeError("Qwen3.8 fused TP/SP mixer input requires bias-free projections")
    eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6)))
    tp_overlap = _uses_te_tensor_parallel_overlap(runtime)
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
        ub_overlap_ag=tp_overlap,
        ub_name="qkv" if tp_overlap else None,
        device=device,
        name=name,
    )
    _copy_te_parameter(
        fused.layer_norm_weight,
        _dtensor_local_tensor(norm.weight),
        tensor_parallel_sharded=False,
        sequence_parallel_replicated=True,
    )
    _copy_te_parameter(
        fused.weight,
        torch.cat(local_weights, dim=0),
        tensor_parallel_sharded=True,
    )
    fused._dllm_local_split_sizes = local_split_sizes
    fused._dllm_lora_parallel_mode = "column"
    fused._dllm_lora_fuses_rmsnorm = True
    fused._dllm_lora_norm_eps = eps
    return fused


def _build_qwen38_te_layernorm_mlp(
    layer: nn.Module,
    *,
    runtime: Any,
    dtype: torch.dtype,
    device: torch.device,
    name: str,
) -> nn.Module:
    """Build Qwen's RMSNorm-SwiGLU MLP with TE's fused TP/SP module."""

    try:
        from transformer_engine.pytorch import LayerNormMLP
    except Exception as exc:  # pragma: no cover - depends on runtime image.
        raise RuntimeError(
            "Qwen3.8 TP/SP training requires transformer_engine.pytorch.LayerNormMLP"
        ) from exc

    tp_size = int(getattr(runtime, "tensor_parallel_size", 1) or 1)
    tp_group = getattr(runtime, "tensor_parallel_group", None)
    gate_proj = layer.mlp.gate_proj
    up_proj = layer.mlp.up_proj
    down_proj = layer.mlp.down_proj
    hidden_size = int(_dtensor_local_tensor(gate_proj.weight).shape[1])
    intermediate_size = _column_linear_out_features(gate_proj, tp_size=tp_size)
    eps = float(
        getattr(
            layer.post_attention_layernorm,
            "variance_epsilon",
            getattr(layer.post_attention_layernorm, "eps", 1e-6),
        )
    )
    use_bias = any(
        getattr(module, "bias", None) is not None
        for module in (gate_proj, up_proj, down_proj)
    )
    if use_bias:
        raise RuntimeError("Qwen3.8 fused TP/SP MLP requires bias-free projections")
    tp_overlap = _uses_te_tensor_parallel_overlap(runtime)
    fused = LayerNormMLP(
        hidden_size,
        intermediate_size,
        eps=eps,
        sequence_parallel=True,
        return_bias=False,
        tp_group=tp_group,
        tp_size=tp_size,
        bias=use_bias,
        normalization="RMSNorm",
        activation="swiglu",
        params_dtype=dtype,
        set_parallel_mode=True,
        ub_overlap_ag=tp_overlap,
        ub_overlap_rs=tp_overlap,
        device=device,
        name=name,
    )
    _copy_te_parameter(
        fused.layer_norm_weight,
        _dtensor_local_tensor(layer.post_attention_layernorm.weight),
        tensor_parallel_sharded=False,
        sequence_parallel_replicated=True,
    )
    _copy_te_parameter(
        fused.fc1_weight,
        torch.cat(
            (
                _dtensor_local_tensor(gate_proj.weight),
                _dtensor_local_tensor(up_proj.weight),
            ),
            dim=0,
        ),
        tensor_parallel_sharded=True,
    )
    _copy_te_parameter(
        fused.fc2_weight,
        _dtensor_local_tensor(down_proj.weight),
        tensor_parallel_sharded=True,
    )
    fused._dllm_qwen38_bias_free = True
    return fused


def _release_qwen38_hf_projections(
    layers: nn.ModuleList,
    layer_types: tuple[str, ...],
    *,
    release_norms: bool,
) -> None:
    """Release HF projection copies after their weights are packed into TE."""

    for layer_index, (layer, layer_type) in enumerate(
        zip(layers, layer_types, strict=True)
    ):
        _release_qwen38_hf_layer_projections(
            layer,
            layer_type,
            layer_index=layer_index,
            release_norms=release_norms,
        )
    gc.collect()
    torch.cuda.empty_cache()


def _release_qwen38_hf_layer_projections(
    layer: nn.Module,
    layer_type: str,
    *,
    layer_index: int,
    release_norms: bool,
) -> None:
    """Release one source layer as soon as its packed replacement is complete."""

    if layer_type == "full_attention":
        mixer = layer.self_attn
        projection_names = ("q_proj", "k_proj", "v_proj", "o_proj")
    elif layer_type == "linear_attention":
        mixer = layer.linear_attn
        projection_names = (
            "in_proj_qkv",
            "in_proj_z",
            "in_proj_b",
            "in_proj_a",
            "out_proj",
        )
    else:
        raise ValueError(f"unsupported Qwen3.8 layer type {layer_type!r}")
    for name in projection_names:
        setattr(
            mixer,
            name,
            _ReleasedHFProjection(f"Qwen3.8 layer {layer_index} {name}"),
        )
    for name in ("gate_proj", "up_proj", "down_proj"):
        setattr(
            layer.mlp,
            name,
            _ReleasedHFProjection(f"Qwen3.8 layer {layer_index} MLP {name}"),
        )
    if release_norms:
        layer.input_layernorm = _ReleasedHFProjection(
            f"Qwen3.8 layer {layer_index} input RMSNorm"
        )
        layer.post_attention_layernorm = _ReleasedHFProjection(
            f"Qwen3.8 layer {layer_index} post-attention RMSNorm"
        )
