# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Production sequence-sharded context-parallel Nemotron executor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint

from dllm_parallel.core.attention import (
    BlockDenoisingFullMask,
)
from dllm_parallel.core.attention.full_mask import (
    full_mask_block_denoising_attention_bshd,
)
from dllm_parallel.core.attention.layout import (
    ContextParallelSequenceLayout,
    active_query_indices_for_context_rank,
    context_parallel_sequence_layout,
    gather_context_parallel_sequence,
)
from dllm_parallel.core.models.backbones.moe_checkpoint import (
    moe_route_checkpoint_context_fn,
)
from dllm_parallel.core.models.backbones.nemotron.model import (
    NemotronDecoderLayerOps,
    NemotronLabsDiffusionPackedBlockDiffusionModel,
    _SequenceParallelPackedMeta,
)
from dllm_parallel.core.parallel.tensor_parallel import (
    gather_active_from_sequence_parallel_region,
)
from dllm_parallel.core.profiling.operator_trace import traced_operator


def _attention_trace_name(
    _model: Any,
    layer_ops: NemotronDecoderLayerOps,
    *_args: Any,
    **_kwargs: Any,
) -> str:
    kind = "sliding" if layer_ops.layer_type == "sliding_attention" else "full"
    return f"attention.{kind}"


@dataclass(frozen=True)
class _ContextParallelLayout:
    sequence: ContextParallelSequenceLayout
    attn_mask: BlockDenoisingFullMask
    active_len: int


class NemotronContextParallelModel(NemotronLabsDiffusionPackedBlockDiffusionModel):
    """Full-mask transformer with token-row context-parallel ownership."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._context_layout_cache: dict[
            tuple[object, ...], _ContextParallelLayout
        ] = {}

    def distributed_block_diffusion_loss(
        self,
        active_hidden: torch.Tensor,
        labels: torch.Tensor,
        active_positions: torch.Tensor,
        loss_weights: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Normalize token-row CP shards before model-parallel averaging."""

        if kwargs.get("bp_loss_scale") is None:
            kwargs["bp_loss_scale"] = float(self.runtime.context_attention_size)
        return super().distributed_block_diffusion_loss(
            active_hidden,
            labels,
            active_positions,
            loss_weights,
            **kwargs,
        )

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
    ) -> Any:
        if objective_mode != "standard_block_diffusion":
            return super().forward(
                noisy_input_ids=noisy_input_ids,
                clean_input_ids=clean_input_ids,
                diffusion_times=diffusion_times,
                noise_levels=noise_levels,
                objective_mode=objective_mode,
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
                encoder_loss_weight=encoder_loss_weight,
                decoder_loss_denominator=decoder_loss_denominator,
                encoder_loss_denominator=encoder_loss_denominator,
                self_conditioning_row_chunk_size=(self_conditioning_row_chunk_size),
                self_conditioning_vocab_chunk_size=(self_conditioning_vocab_chunk_size),
            )
        del diffusion_times, noise_levels
        if noisy_input_ids.shape != clean_input_ids.shape:
            raise ValueError(
                "noisy_input_ids and clean_input_ids must have the same shape"
            )
        self._activate_sequence_length(int(noisy_input_ids.shape[1]))
        if self.seq_len % int(self.runtime.context_attention_size):
            raise ValueError(
                "pure context-parallel sequence length must divide evenly by the "
                "context-parallel size"
            )
        if int(getattr(self.runtime, "block_parallel_size", 1) or 1) != 1:
            raise RuntimeError(
                "pure context parallelism requires block_parallel_size=1"
            )
        layout = self._context_layout(noisy_input_ids.device)
        sequence = layout.sequence
        local_ids = gather_context_parallel_sequence(
            noisy_input_ids,
            clean_input_ids,
            sequence,
        )
        hidden_states = self.embed_tokens(local_ids).to(dtype=self._model_dtype())
        if self.embedding_scale is not None:
            hidden_states = hidden_states * torch.as_tensor(
                self.embedding_scale,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        if self.self_conditioning is not None:
            if self.self_condition_clean_tokens:
                hidden_states = self.self_conditioning(
                    hidden_states,
                    torch.zeros_like(hidden_states),
                )
            else:
                noisy_hidden = hidden_states.index_select(
                    1,
                    sequence.noisy_local_indices,
                )
                conditioned = self.self_conditioning(
                    noisy_hidden,
                    torch.zeros_like(noisy_hidden),
                )
                hidden_states = hidden_states.index_copy(
                    1,
                    sequence.noisy_local_indices,
                    conditioned,
                )
        position_ids = sequence.model_positions.unsqueeze(0).expand(
            hidden_states.shape[0], -1
        )
        position_cache: dict[str | None, tuple[torch.Tensor, torch.Tensor]] = {}
        if self.sequence_parallel:
            return self._forward_context_sequence_parallel(
                hidden_states=hidden_states,
                position_ids=position_ids,
                position_cache=position_cache,
                layout=layout,
            )

        cache_position = sequence.model_positions
        for layer_index, layer_ops in enumerate(self.layers):
            attn_mask = self._full_attention_mask_for_layer(
                layout.attn_mask,
                query_positions=sequence.model_positions,
                layer_ops=layer_ops,
                cache_namespace="context_parallel",
            )
            position_embeddings = self._position_embeddings_for_layer(
                layer_ops,
                hidden_states=hidden_states,
                position_ids=position_ids,
                cache=position_cache,
            )
            if self._full_layer_checkpointing_enabled():
                hidden_states = self._checkpoint_context_layer(
                    layer_ops,
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    cache_position=cache_position,
                    attn_mask=attn_mask,
                )
            else:
                hidden_states = self._context_layer_forward(
                    layer_ops,
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    cache_position=cache_position,
                    attn_mask=attn_mask,
                )
            self._debug_check_layer_tensor(hidden_states, layer_index)
        hidden_states = self.norm(hidden_states)
        return (
            hidden_states.index_select(1, sequence.noisy_local_indices),
            sequence.noisy_positions,
        )

    def _native_stream_positions(
        self,
        device: torch.device,
        *,
        decoder_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        context_size = int(self.runtime.context_attention_size)
        if self.block_size % context_size:
            raise ValueError(
                "native pure CP requires block_size divisible by context-parallel size"
            )
        active_positions = active_query_indices_for_context_rank(
            active_len=(self.seq_len if decoder_length is None else int(decoder_length)),
            block_size=self.block_size,
            context_parallel_size=context_size,
            context_parallel_rank=int(self.runtime.context_parallel_rank),
            device=device,
        )
        return active_positions, self._local_clean_positions(device), True

    def _forward_context_sequence_parallel(
        self,
        *,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        position_cache: dict[str | None, tuple[torch.Tensor, torch.Tensor]],
        layout: _ContextParallelLayout,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        meta = self._sequence_parallel_meta(
            hidden_states,
            active_len=layout.active_len,
        )
        hidden_shard = self._scatter_packed_hidden_to_sequence_parallel(
            hidden_states,
            meta,
        )
        cache_position = layout.sequence.model_positions
        for layer_index, layer_ops in enumerate(self.layers):
            attn_mask = self._full_attention_mask_for_layer(
                layout.attn_mask,
                query_positions=layout.sequence.model_positions,
                layer_ops=layer_ops,
                cache_namespace="context_parallel_sequence",
            )
            position_embeddings = self._position_embeddings_for_layer(
                layer_ops,
                hidden_states=hidden_states,
                position_ids=position_ids,
                cache=position_cache,
            )
            if self._full_layer_checkpointing_enabled():
                hidden_shard = self._checkpoint_context_layer_sequence_parallel(
                    layer_ops,
                    hidden_shard=hidden_shard,
                    meta=meta,
                    position_embeddings=position_embeddings,
                    cache_position=cache_position,
                    attn_mask=attn_mask,
                )
            else:
                hidden_shard = self._context_layer_forward_sequence_parallel(
                    layer_ops,
                    hidden_shard=hidden_shard,
                    meta=meta,
                    position_embeddings=position_embeddings,
                    cache_position=cache_position,
                    attn_mask=attn_mask,
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
        original_positions = layout.sequence.noisy_positions.index_select(
            0,
            active_pairs[:, 1],
        )
        return active_hidden, torch.stack(
            (active_pairs[:, 0], original_positions), dim=-1
        )

    def _checkpoint_context_layer(
        self,
        layer_ops: NemotronDecoderLayerOps,
        *,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cache_position: torch.Tensor,
        attn_mask: BlockDenoisingFullMask,
    ) -> torch.Tensor:
        forward = lambda states, ops=layer_ops, pe=position_embeddings: (
            self._context_layer_forward(
                ops,
                hidden_states=states,
                position_embeddings=pe,
                cache_position=cache_position,
                attn_mask=attn_mask,
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

    def _checkpoint_context_layer_sequence_parallel(
        self,
        layer_ops: NemotronDecoderLayerOps,
        *,
        hidden_shard: torch.Tensor,
        meta: _SequenceParallelPackedMeta,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cache_position: torch.Tensor,
        attn_mask: BlockDenoisingFullMask,
    ) -> torch.Tensor:
        def forward(states: torch.Tensor) -> torch.Tensor:
            return self._context_layer_forward_sequence_parallel(
                layer_ops,
                hidden_shard=states,
                meta=meta,
                position_embeddings=position_embeddings,
                cache_position=cache_position,
                attn_mask=attn_mask,
                checkpoint_mlp=False,
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

    def _context_layer_forward(
        self,
        layer_ops: NemotronDecoderLayerOps,
        *,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cache_position: torch.Tensor,
        attn_mask: BlockDenoisingFullMask,
        checkpoint_mlp: bool = True,
    ) -> torch.Tensor:
        if layer_ops.layer_style == "gemma4":
            residual = hidden_states
            hidden_states = layer_ops.input_layernorm(residual)
            self._debug_check_named_tensor(
                hidden_states,
                layer_ops,
                "attention_normalized_input",
            )
            hidden_states = self._context_attention_forward(
                layer_ops,
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                cache_position=cache_position,
                attn_mask=attn_mask,
            )
            self._debug_check_named_tensor(
                hidden_states,
                layer_ops,
                "attention_output",
            )
            hidden_states = layer_ops.post_attention_layernorm(hidden_states)
            hidden_states = hidden_states + residual
            self._debug_check_named_tensor(
                hidden_states,
                layer_ops,
                "feed_forward_input",
            )
            return self._gemma4_feed_forward_block(
                layer_ops,
                hidden_states,
                checkpoint_mlp=checkpoint_mlp,
            )
        residual = hidden_states
        hidden_states = layer_ops.input_layernorm(hidden_states)
        hidden_states = residual + self._context_attention_forward(
            layer_ops,
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            cache_position=cache_position,
            attn_mask=attn_mask,
        )
        return self._mlp_block_forward(
            layer_ops,
            hidden_states,
            checkpoint_mlp=checkpoint_mlp,
        )

    def _context_layer_forward_sequence_parallel(
        self,
        layer_ops: NemotronDecoderLayerOps,
        *,
        hidden_shard: torch.Tensor,
        meta: _SequenceParallelPackedMeta,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cache_position: torch.Tensor,
        attn_mask: BlockDenoisingFullMask,
        checkpoint_mlp: bool = True,
    ) -> torch.Tensor:
        if layer_ops.layer_style == "gemma4":
            residual = hidden_shard
            te_layer = self._te_packed_by_layer_id.get(id(layer_ops.layer))
            hidden_shard = (
                residual
                if te_layer is not None and te_layer.qkv_fuses_input_norm
                else layer_ops.input_layernorm(residual)
            )
            hidden_shard = self._context_attention_forward_sequence_parallel(
                layer_ops,
                hidden_shard=hidden_shard,
                meta=meta,
                position_embeddings=position_embeddings,
                cache_position=cache_position,
                attn_mask=attn_mask,
            )
            hidden_shard = layer_ops.post_attention_layernorm(hidden_shard)
            hidden_shard = hidden_shard + residual
            return self._gemma4_feed_forward_block(
                layer_ops,
                hidden_shard,
                checkpoint_mlp=checkpoint_mlp,
            )
        residual = hidden_shard
        hidden_shard = layer_ops.input_layernorm(hidden_shard)
        hidden_shard = residual + self._context_attention_forward_sequence_parallel(
            layer_ops,
            hidden_shard=hidden_shard,
            meta=meta,
            position_embeddings=position_embeddings,
            cache_position=cache_position,
            attn_mask=attn_mask,
        )
        return self._mlp_block_forward(
            layer_ops,
            hidden_shard,
            checkpoint_mlp=checkpoint_mlp,
        )

    @traced_operator(_attention_trace_name)
    def _context_attention_forward(
        self,
        layer_ops: NemotronDecoderLayerOps,
        *,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cache_position: torch.Tensor,
        attn_mask: BlockDenoisingFullMask,
    ) -> torch.Tensor:
        query, key, value, output_gate = self._project_qkv_bshd(
            layer_ops,
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            cache_position=cache_position,
        )
        self._debug_check_named_tensor(query, layer_ops, "attention_query")
        self._debug_check_named_tensor(key, layer_ops, "attention_key")
        self._debug_check_named_tensor(value, layer_ops, "attention_value")
        output = full_mask_block_denoising_attention_bshd(
            query,
            key,
            value,
            global_seq_len=2 * self.seq_len,
            attn_mask=attn_mask,
            scale=layer_ops.attention.scale,
            runtime=self.runtime,
        )
        self._debug_check_named_tensor(output, layer_ops, "attention_core_output")
        output = output.reshape(*hidden_states.shape[:-1], -1).contiguous()
        if output_gate is not None:
            output = output * torch.sigmoid(
                output_gate.reshape(*hidden_states.shape[:-1], -1)
            )
        self._debug_check_named_tensor(
            output,
            layer_ops,
            "attention_projection_input",
        )
        return self._attention_output_projection(layer_ops, output)

    @traced_operator(_attention_trace_name)
    def _context_attention_forward_sequence_parallel(
        self,
        layer_ops: NemotronDecoderLayerOps,
        *,
        hidden_shard: torch.Tensor,
        meta: _SequenceParallelPackedMeta,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cache_position: torch.Tensor,
        attn_mask: BlockDenoisingFullMask,
    ) -> torch.Tensor:
        query, key, value, output_gate = self._project_qkv_bshd_sequence_parallel(
            layer_ops,
            hidden_shard=hidden_shard,
            meta=meta,
            position_embeddings=position_embeddings,
            cache_position=cache_position,
        )
        output = full_mask_block_denoising_attention_bshd(
            query,
            key,
            value,
            global_seq_len=2 * self.seq_len,
            attn_mask=attn_mask,
            scale=layer_ops.attention.scale,
            runtime=self.runtime,
        )
        output = output.reshape(meta.total_rows, -1).contiguous()
        if output_gate is not None:
            output = output * torch.sigmoid(output_gate.reshape(meta.total_rows, -1))
        return self._attention_output_projection(
            layer_ops,
            self._pad_rows(output, meta.padded_rows),
        )

    def _context_layout(self, device: torch.device) -> _ContextParallelLayout:
        device_index = device.index
        if device.type == "cuda" and device_index is None:
            device_index = torch.cuda.current_device()
        cache_key = (
            self.seq_len,
            self.block_size,
            int(self.runtime.context_attention_size),
            int(self.runtime.context_parallel_rank),
            bool(self._debug_nonfinite_attention_enabled()),
            device.type,
            device_index,
        )
        cached = self._context_layout_cache.get(cache_key)
        if cached is not None:
            return cached
        sequence = context_parallel_sequence_layout(
            seq_len=self.seq_len,
            block_size=self.block_size,
            context_parallel_size=int(self.runtime.context_attention_size),
            rank=int(self.runtime.context_parallel_rank),
            device=device,
        )
        query_is_clean = sequence.logical_positions >= self.seq_len
        layout = _ContextParallelLayout(
            sequence=sequence,
            attn_mask=BlockDenoisingFullMask(
                query_blocks=(sequence.model_positions // self.block_size).to(
                    torch.int32
                ),
                query_is_clean=query_is_clean,
                block_size=self.block_size,
                clean_offset=self.seq_len,
                backward_query_chunk_size=self._cp_bp_backward_query_chunk_size(),
                debug_nonfinite_attention=self._debug_nonfinite_attention_enabled(),
            ),
            active_len=int(sequence.noisy_local_indices.numel()),
        )
        self._context_layout_cache[cache_key] = layout
        return layout


def build_context_parallel_model(
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
) -> NemotronContextParallelModel:
    return NemotronContextParallelModel(
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


__all__ = ["NemotronContextParallelModel", "build_context_parallel_model"]
