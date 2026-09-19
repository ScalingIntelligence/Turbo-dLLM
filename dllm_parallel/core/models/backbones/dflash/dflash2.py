# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""SGLang-compatible DFlash2 architecture and memory-bounded objective."""

from __future__ import annotations

from typing import Any
import weakref

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from dllm_parallel.core.kernels.dflash_linear_topk import frozen_linear_ce_topk
from dllm_parallel.core.models.loss import zero_loss_with_module_parameters
from dllm_parallel.core.parallel.runtime import loss_scale as runtime_loss_scale
from dllm_parallel.core.profiling.operator_trace import operator_scope

from .model import (
    DFlashAttentionOp,
    DFlashAttention,
    DFlashDecoderLayer,
    DFlashForwardOutput,
    DFlashMLP,
    DFlashModel,
    DFlashModelConfig,
)


_DPACE_LOSSES = {
    "dpace",
    "dpace-cumulative-confidence-only",
    "dpace-continuation-value-only",
}


def _grouped_convolve_local(
    blocks: torch.Tensor,
    dynamic: torch.Tensor,
    base_kernel: torch.Tensor,
    taps: int,
) -> torch.Tensor:
    """One static-shape graph for the block-local convolution pipeline."""

    padded = F.pad(blocks, (0, 0, 0, 0, taps - 1, 0))
    sources = padded.unfold(2, taps, 1).permute(0, 1, 2, 5, 3, 4)
    coefficients = base_kernel.reshape(
        1, 1, 1, taps, blocks.shape[-2], blocks.shape[-1]
    ) + dynamic.unsqueeze(-1)
    return (coefficients.flip(3) * sources).sum(dim=3)


def _swiglu_local(gate_up: torch.Tensor, intermediate_size: int) -> torch.Tensor:
    gate, up = gate_up.split((intermediate_size, intermediate_size), dim=-1)
    return F.silu(gate) * up


# These pure, fixed-shape regions are safe CUDA-graph boundaries: they contain
# no NCCL, allocation-sensitive attention planners, or random sampling. Inductor
# creates one graph per shape bucket and reuses it after the first invocation.
_grouped_convolve_graph = torch.compile(
    _grouped_convolve_local,
    fullgraph=True,
    dynamic=False,
    mode="reduce-overhead",
)
_swiglu_graph = torch.compile(
    _swiglu_local,
    fullgraph=True,
    dynamic=False,
    mode="reduce-overhead",
)


class _PackedLinearView(nn.Module):
    """Serving-name view into a permanently packed trainable projection."""

    def __init__(
        self,
        owner: nn.Module,
        start: int,
        stop: int,
        in_features: int,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "_owner_ref", weakref.ref(owner))
        self.start = int(start)
        self.stop = int(stop)
        self.in_features = int(in_features)
        self.out_features = self.stop - self.start

    @property
    def weight(self) -> torch.Tensor:
        owner = object.__getattribute__(self, "_owner_ref")()
        assert owner is not None
        return owner.packed_weight[self.start : self.stop]

    @property
    def bias(self) -> torch.Tensor | None:
        owner = object.__getattribute__(self, "_owner_ref")()
        assert owner is not None
        packed_bias = owner.packed_bias
        return None if packed_bias is None else packed_bias[self.start : self.stop]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states, self.weight, self.bias)


class DFlash2PackedAttention(DFlashAttention):
    """One persistent QKV parameter with legacy SpecForge checkpoint keys."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        projections = (self.q_proj, self.k_proj, self.v_proj)
        sizes = tuple(int(projection.out_features) for projection in projections)
        packed_weight = torch.cat([projection.weight for projection in projections])
        packed_bias = (
            None
            if projections[0].bias is None
            else torch.cat([projection.bias for projection in projections])
        )
        del self.q_proj, self.k_proj, self.v_proj
        self.packed_weight = nn.Parameter(packed_weight)
        self.packed_bias = (
            None if packed_bias is None else nn.Parameter(packed_bias)
        )
        q_stop, k_stop = sizes[0], sizes[0] + sizes[1]
        self.q_proj = _PackedLinearView(self, 0, q_stop, packed_weight.shape[1])
        self.k_proj = _PackedLinearView(
            self, q_stop, k_stop, packed_weight.shape[1]
        )
        self.v_proj = _PackedLinearView(
            self, k_stop, sum(sizes), packed_weight.shape[1]
        )
        self._packed_sizes = sizes
        self.register_state_dict_post_hook(self._legacy_state_dict_hook())
        self.register_load_state_dict_pre_hook(self._legacy_load_hook())

    def _legacy_state_dict_hook(self):
        def hook(module, state_dict, prefix, local_metadata):
            del local_metadata
            weight = state_dict.pop(prefix + "packed_weight")
            bias = state_dict.pop(prefix + "packed_bias", None)
            for name, value in zip(
                ("q_proj", "k_proj", "v_proj"),
                weight.split(module._packed_sizes, dim=0),
                strict=True,
            ):
                state_dict[prefix + name + ".weight"] = value
            if bias is not None:
                for name, value in zip(
                    ("q_proj", "k_proj", "v_proj"),
                    bias.split(module._packed_sizes, dim=0),
                    strict=True,
                ):
                    state_dict[prefix + name + ".bias"] = value

        return hook

    def _legacy_load_hook(self):
        def hook(module, state_dict, prefix, *args):
            del args
            weight_keys = [prefix + name + ".weight" for name in ("q_proj", "k_proj", "v_proj")]
            if all(key in state_dict for key in weight_keys):
                state_dict[prefix + "packed_weight"] = torch.cat(
                    [state_dict.pop(key) for key in weight_keys], dim=0
                )
            bias_keys = [prefix + name + ".bias" for name in ("q_proj", "k_proj", "v_proj")]
            if module.packed_bias is not None and all(key in state_dict for key in bias_keys):
                state_dict[prefix + "packed_bias"] = torch.cat(
                    [state_dict.pop(key) for key in bias_keys], dim=0
                )

        return hook

    def _project_draft_qkv(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        projected = F.linear(hidden_states, self.packed_weight, self.packed_bias)
        return projected.split(self._packed_sizes, dim=-1)

    def _project_context_kv(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_size, k_size, v_size = self._packed_sizes
        weight = self.packed_weight[q_size:]
        bias = None if self.packed_bias is None else self.packed_bias[q_size:]
        projected = F.linear(hidden_states, weight, bias)
        return projected.split((k_size, v_size), dim=-1)


class DFlash2PackedMLP(DFlashMLP):
    """Persistent gate/up packing plus a graph-captured SwiGLU epilogue."""

    def __init__(self, config: DFlashModelConfig) -> None:
        super().__init__(config)
        size = int(self.gate_proj.out_features)
        packed_weight = torch.cat((self.gate_proj.weight, self.up_proj.weight))
        packed_bias = (
            None
            if self.gate_proj.bias is None
            else torch.cat((self.gate_proj.bias, self.up_proj.bias))
        )
        del self.gate_proj, self.up_proj
        self.packed_weight = nn.Parameter(packed_weight)
        self.packed_bias = None if packed_bias is None else nn.Parameter(packed_bias)
        self.gate_proj = _PackedLinearView(self, 0, size, packed_weight.shape[1])
        self.up_proj = _PackedLinearView(self, size, 2 * size, packed_weight.shape[1])
        self.intermediate_size = size
        self.register_state_dict_post_hook(self._legacy_state_dict_hook())
        self.register_load_state_dict_pre_hook(self._legacy_load_hook())

    def _legacy_state_dict_hook(self):
        def hook(module, state_dict, prefix, local_metadata):
            del local_metadata
            gate, up = state_dict.pop(prefix + "packed_weight").chunk(2, dim=0)
            state_dict[prefix + "gate_proj.weight"] = gate
            state_dict[prefix + "up_proj.weight"] = up
            bias = state_dict.pop(prefix + "packed_bias", None)
            if bias is not None:
                gate_bias, up_bias = bias.chunk(2, dim=0)
                state_dict[prefix + "gate_proj.bias"] = gate_bias
                state_dict[prefix + "up_proj.bias"] = up_bias

        return hook

    def _legacy_load_hook(self):
        def hook(module, state_dict, prefix, *args):
            del args
            gate_key, up_key = prefix + "gate_proj.weight", prefix + "up_proj.weight"
            if gate_key in state_dict and up_key in state_dict:
                state_dict[prefix + "packed_weight"] = torch.cat(
                    (state_dict.pop(gate_key), state_dict.pop(up_key)), dim=0
                )
            gate_bias, up_bias = prefix + "gate_proj.bias", prefix + "up_proj.bias"
            if module.packed_bias is not None and gate_bias in state_dict and up_bias in state_dict:
                state_dict[prefix + "packed_bias"] = torch.cat(
                    (state_dict.pop(gate_bias), state_dict.pop(up_bias)), dim=0
                )

        return hook

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        with operator_scope("dflash.mlp"):
            gate_up = F.linear(hidden_states, self.packed_weight, self.packed_bias)
            activated = (
                _swiglu_graph(gate_up, self.intermediate_size)
                if gate_up.is_cuda
                else _swiglu_local(gate_up, self.intermediate_size)
            )
            return self.down_proj(activated)


class DFlashGroupedConv(nn.Module):
    """Grouped dynamic depthwise convolution with per-tap temporary storage."""

    def __init__(
        self,
        hidden_size: int,
        block_size: int,
        taps: int,
        group_size: int,
    ) -> None:
        super().__init__()
        if taps < 1 or taps > block_size:
            raise ValueError("DFlash2 convolution taps must be in [1, block_size]")
        if group_size < 1 or hidden_size % group_size:
            raise ValueError("DFlash2 convolution group_size must divide hidden_size")
        self.block_size = int(block_size)
        self.taps = int(taps)
        self.group_size = int(group_size)
        self.num_groups = int(hidden_size) // int(group_size)
        base_kernel = torch.zeros(2, self.taps, int(hidden_size))
        base_kernel[:, 0] = 1.0
        self.base_kernel = nn.Parameter(base_kernel)
        self.kernel_projection = nn.Linear(
            int(hidden_size),
            2 * self.taps * self.num_groups,
            bias=False,
        )
        nn.init.zeros_(self.kernel_projection.weight)

    def _convolve(
        self,
        hidden_states: torch.Tensor,
        delta: torch.Tensor,
        *,
        side: int,
    ) -> torch.Tensor:
        batch, sequence, hidden_size = hidden_states.shape
        if sequence % self.block_size:
            raise ValueError("DFlash2 convolution input must contain whole blocks")
        blocks = hidden_states.reshape(
            batch,
            sequence // self.block_size,
            self.block_size,
            self.num_groups,
            self.group_size,
        )
        dynamic = delta.reshape(
            batch,
            sequence // self.block_size,
            self.block_size,
            self.taps,
            self.num_groups,
        )
        graph = _grouped_convolve_graph if blocks.is_cuda else _grouped_convolve_local
        output = graph(blocks, dynamic, self.base_kernel[side], self.taps)
        return output.reshape(batch, sequence, hidden_size)

    def prepare(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coefficients = self.kernel_projection(hidden_states).reshape(
            *hidden_states.shape[:-1], 2, self.taps, self.num_groups
        )
        return (
            self._convolve(hidden_states, coefficients[..., 0, :, :], side=0),
            coefficients[..., 1, :, :],
        )

    def finish(
        self,
        hidden_states: torch.Tensor,
        coefficients: torch.Tensor,
    ) -> torch.Tensor:
        return self._convolve(hidden_states, coefficients, side=1)


class DFlash2DecoderLayer(DFlashDecoderLayer):
    """DFlash decoder layer with serving-compatible convolution wrappers."""

    def __init__(
        self,
        config: DFlashModelConfig,
        *,
        layer_idx: int,
        attention_op: DFlashAttentionOp,
    ) -> None:
        super().__init__(config, layer_idx=layer_idx, attention_op=attention_op)
        self.self_attn = DFlash2PackedAttention(
            config,
            layer_idx=layer_idx,
            attention_op=attention_op,
        )
        self.mlp = DFlash2PackedMLP(config)
        assert config.conv_kernel_size is not None
        assert config.conv_group_size is not None
        self.attention_conv = DFlashGroupedConv(
            config.hidden_size,
            config.block_size,
            config.conv_kernel_size,
            config.conv_group_size,
        )
        self.mlp_conv = DFlashGroupedConv(
            config.hidden_size,
            config.block_size,
            config.conv_kernel_size,
            config.conv_group_size,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        residual = hidden_states
        with operator_scope("dflash.convolution"):
            hidden_states, attention_kernel = self.attention_conv.prepare(
                self.input_layernorm(hidden_states)
            )
        hidden_states = self.self_attn(hidden_states, target_hidden, **kwargs)
        with operator_scope("dflash.convolution"):
            hidden_states = residual + self.attention_conv.finish(
                hidden_states, attention_kernel
            )
        residual = hidden_states
        with operator_scope("dflash.convolution"):
            hidden_states, mlp_kernel = self.mlp_conv.prepare(
                self.post_attention_layernorm(hidden_states)
            )
        hidden_states = self.mlp(hidden_states)
        with operator_scope("dflash.convolution"):
            return residual + self.mlp_conv.finish(hidden_states, mlp_kernel)


class CandidateSelector(nn.Module):
    """Low-rank predecessor-conditioned candidate scorer used by SGLang."""

    def __init__(
        self,
        *,
        hidden_size: int,
        vocab_size: int,
        state_rank: int,
        top_k: int,
        initializer_range: float,
    ) -> None:
        super().__init__()
        self.top_k = int(top_k)
        self.predecessor_codebook = nn.Parameter(
            torch.empty(int(vocab_size), int(state_rank))
        )
        self.successor_codebook = nn.Parameter(
            torch.empty(int(vocab_size), int(state_rank))
        )
        self.hidden_projection = nn.Linear(
            int(hidden_size), int(state_rank), bias=False
        )
        nn.init.normal_(self.predecessor_codebook, std=float(initializer_range))
        nn.init.zeros_(self.successor_codebook)

    def score_candidates(
        self,
        *,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        predecessor_ids: torch.Tensor,
    ) -> torch.Tensor:
        predecessor = self.predecessor_codebook[predecessor_ids]
        successor = self.successor_codebook[candidate_ids]
        context = predecessor * self.hidden_projection(hidden_states)
        return unary_logits + torch.einsum("...r,...kr->...k", context, successor)

    def build_lattice(
        self,
        *,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        predecessor_ids = torch.cat(
            (
                anchor_token_ids[:, None, None].expand(-1, 1, self.top_k),
                candidate_ids[:, :-1],
            ),
            dim=1,
        )
        predecessor = self.predecessor_codebook[predecessor_ids]
        successor = self.successor_codebook[candidate_ids]
        context = predecessor * self.hidden_projection(hidden_states)[:, :, None]
        return unary_logits[:, :, None] + torch.einsum(
            "blpr,blcr->blpc", context, successor
        )

    def greedy_path(
        self,
        *,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        predecessor_ids = anchor_token_ids
        path = []
        for position in range(candidate_ids.shape[1]):
            scores = self.score_candidates(
                candidate_ids=candidate_ids[:, position],
                unary_logits=unary_logits[:, position],
                hidden_states=hidden_states[:, position],
                predecessor_ids=predecessor_ids,
            )
            selected = scores.argmax(dim=-1, keepdim=True)
            predecessor_ids = candidate_ids[:, position].gather(1, selected)[:, 0]
            path.append(predecessor_ids)
        return torch.stack(path, dim=1)


class DFlash2Model(DFlashModel):
    """Optimized training model whose trainable state matches DFlash2DraftModel."""

    def forward(self, *args: Any, **kwargs: Any) -> DFlashForwardOutput:
        # Inductor cannot infer iteration boundaries across the separately
        # compiled convolution/SwiGLU regions and their pending backwards.
        # Keep this marker inside the DFlash2 architecture so other model
        # families and the shared trainer retain byte-for-byte runtime behavior.
        torch.compiler.cudagraph_mark_step_begin()
        return super().forward(*args, **kwargs)

    def _build_decoder_layer(
        self,
        config: DFlashModelConfig,
        layer_idx: int,
        attention_op: DFlashAttentionOp,
    ) -> nn.Module:
        return DFlash2DecoderLayer(
            config, layer_idx=layer_idx, attention_op=attention_op
        )

    def _init_draft_head(self, config: DFlashModelConfig) -> None:
        assert config.selector_rank is not None
        assert config.selector_top_k is not None
        self.candidate_selector = CandidateSelector(
            hidden_size=config.hidden_size,
            vocab_size=config.verifier_vocab_size,
            state_rank=config.selector_rank,
            top_k=config.selector_top_k,
            initializer_range=config.initializer_range,
        )

    def transform_unary_logits(self, logits: torch.Tensor) -> torch.Tensor:
        transformed = logits.float() * self.config.output_multiplier
        softcap = self.config.final_logit_softcapping
        if softcap is not None:
            transformed = torch.tanh(transformed / softcap) * softcap
        return transformed

    @staticmethod
    def _dpace_weight(
        probability: torch.Tensor,
        weight_mask: torch.Tensor,
        *,
        alpha: float,
        loss_kind: str,
    ) -> torch.Tensor:
        smooth = (1.0 - alpha) * probability + alpha
        smooth = torch.where(weight_mask > 0, smooth, torch.ones_like(smooth))
        prefix = torch.cumprod(smooth, dim=-1)
        if loss_kind == "dpace-cumulative-confidence-only":
            return prefix
        suffix = torch.flip(
            torch.cumsum(torch.flip(prefix * weight_mask, dims=(-1,)), dim=-1),
            dims=(-1,),
        )
        if loss_kind == "dpace":
            return suffix
        if loss_kind == "dpace-continuation-value-only":
            return suffix / prefix.clamp_min(torch.finfo(prefix.dtype).tiny)
        raise ValueError(f"unsupported D-PACE loss {loss_kind!r}")

    def _global_detached_sum(self, value: torch.Tensor) -> torch.Tensor:
        result = value.detach().clone()
        runtime = self.runtime
        if (
            runtime is not None
            and len(runtime.context_block_parallel_group_ranks) > 1
            and dist.is_available()
            and dist.is_initialized()
        ):
            dist.all_reduce(result, group=runtime.context_block_parallel_group)
            replicated_context_ranks = len(
                runtime.context_block_parallel_group_ranks
            ) // max(1, int(runtime.block_parallel_size))
            if replicated_context_ranks > 1:
                result.div_(float(replicated_context_ranks))
        return result

    def distributed_dflash_loss(
        self,
        output: DFlashForwardOutput,
        *,
        loss_kind: str,
        normalization_count: torch.Tensor,
        bp_loss_scale: float | None = None,
        dpace_alpha: float = 0.5,
        lk_loss_type: str | None = None,
        kl_scale: float = 1.0,
        kl_decay: float = 1.0,
        selector_loss_alpha: float = 1.0,
        selector_stop_gradient: bool = False,
        vocab_block_size: int = 32768,
    ) -> torch.Tensor:
        if loss_kind not in {"dflash", *_DPACE_LOSSES}:
            return super().distributed_dflash_loss(
                output,
                loss_kind=loss_kind,
                normalization_count=normalization_count,
                bp_loss_scale=bp_loss_scale,
            )
        if lk_loss_type not in {None, "alpha", "lambda", "tv"}:
            raise ValueError("lk_loss_type must be None, alpha, lambda, or tv")
        valid = output.supervised_mask.reshape(-1)
        indices = torch.nonzero(valid, as_tuple=False).flatten()
        scale = float(
            bp_loss_scale
            if bp_loss_scale is not None
            else (runtime_loss_scale(self.runtime) if self.runtime is not None else 1.0)
        )
        if indices.numel() == 0:
            global_stats = self._global_detached_sum(
                output.hidden_states.new_zeros((3,), dtype=torch.float32)
            )
            zero = zero_loss_with_module_parameters(output.hidden_states, self)
            return zero * scale / global_stats[0].clamp_min(1.0)

        hidden = output.hidden_states.reshape(
            -1, output.hidden_states.shape[-1]
        ).index_select(0, indices)
        labels = output.target_token_ids.reshape(-1).index_select(0, indices)
        selector_enabled = bool(getattr(self, "selector_objective_enabled", True))
        with operator_scope("dflash.vocabulary_objective"):
            ce, probability, unary_topk, candidate_ids = frozen_linear_ce_topk(
                hidden,
                self.lm_head.weight,
                labels,
                top_k=self.candidate_selector.top_k if selector_enabled else 1,
                vocab_block_size=vocab_block_size,
                output_multiplier=self.config.output_multiplier,
                logit_softcap=self.config.final_logit_softcapping,
            )
        flat_shape = output.supervised_mask.shape
        block_size = self.block_size
        if int(flat_shape[-1]) % block_size:
            raise ValueError("DFlash2 objective tokens must contain whole blocks")
        shape = (
            int(flat_shape[0]),
            int(flat_shape[1]) // block_size,
            block_size,
        )
        full_probability = torch.zeros(shape, device=hidden.device, dtype=torch.float32)
        full_probability.view(-1).index_copy_(0, indices, probability.detach())
        weight_mask = output.supervised_mask.reshape(shape).float()
        if loss_kind == "dflash":
            loss_weights = weight_mask * output.position_weights.reshape(shape).float()
        else:
            with torch.no_grad():
                loss_weights = weight_mask * self._dpace_weight(
                    full_probability,
                    weight_mask,
                    alpha=float(dpace_alpha),
                    loss_kind=loss_kind,
                )
        selected_weights = loss_weights.reshape(-1).index_select(0, indices)
        ce_num = (ce * selected_weights).sum()
        tv_num = ((1.0 - probability) * selected_weights).sum()
        global_stats = self._global_detached_sum(
            torch.stack(
                (
                    selected_weights.sum(),
                    (full_probability * weight_mask).sum(),
                    weight_mask.sum(),
                )
            )
        )
        token_num = ce_num
        if lk_loss_type == "tv":
            token_num = tv_num
        elif lk_loss_type == "lambda":
            acceptance = global_stats[1] / global_stats[2].clamp_min(1.0)
            ce_weight = float(kl_scale) * torch.exp(
                -float(kl_decay) * acceptance.detach()
            )
            token_num = ce_weight * ce_num + (1.0 - ce_weight) * tv_num

        selector_num = token_num.new_zeros(())
        selector_den = token_num.new_zeros(())
        covered = torch.zeros_like(labels, dtype=torch.bool)
        if selector_enabled:
            selector_hidden = hidden.detach() if selector_stop_gradient else hidden
            selector_unary = unary_topk.detach() if selector_stop_gradient else unary_topk
            all_targets = output.target_token_ids.reshape(-1)
            predecessor = (
                torch.cat(
                    (
                        all_targets.reshape(-1, block_size)[:, :1],
                        all_targets.reshape(-1, block_size)[:, :-1],
                    ),
                    dim=-1,
                )
                .reshape(-1)
                .index_select(0, indices)
            )
            matches = candidate_ids.eq(labels[:, None])
            covered = matches.any(dim=-1)
            target_candidate = matches.long().argmax(dim=-1)
            with operator_scope("dflash.selector_objective"):
                selector_logits = self.candidate_selector.score_candidates(
                    candidate_ids=candidate_ids,
                    unary_logits=selector_unary,
                    hidden_states=selector_hidden,
                    predecessor_ids=predecessor,
                )
                selector_ce = F.cross_entropy(
                    selector_logits.float(), target_candidate, reduction="none"
                )
            selector_weights = selected_weights * covered.float()
            selector_num = (selector_ce * selector_weights).sum()
            selector_den = selector_weights.sum().detach()
        local_num = token_num + float(selector_loss_alpha) * selector_num
        global_den = global_stats[0]
        self.last_objective_metrics = {
            "loss_numerator": local_num.detach(),
            "loss_denominator": global_den,
            "target_probability": probability.detach().mean(),
            "selector_coverage": covered.float().mean(),
            "selector_loss": selector_num.detach()
            / selector_den.clamp_min(1.0),
        }
        return local_num / global_den.clamp_min(torch.finfo(torch.float32).tiny) * scale


__all__ = [
    "CandidateSelector",
    "DFlash2DecoderLayer",
    "DFlash2Model",
    "DFlashGroupedConv",
]
