# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Expert-parallel DiffusionGemma MoE execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


_DEEPEP_HIDDEN_ALIGNMENT = 256


@dataclass(slots=True)
class DiffusionGemmaPendingExpertDispatch:
    pending_dispatch: Any
    input_dtype: torch.dtype


class DiffusionGemmaGroupedExperts(nn.Module):
    """Execute local DiffusionGemma experts with jagged grouped GEMMs."""

    def __init__(self, experts: Any) -> None:
        super().__init__()
        self.hidden_dim = int(experts.hidden_dim)
        self.intermediate_dim = int(experts.intermediate_dim)
        self.num_experts = int(experts.num_experts)
        self.act_fn = experts.act_fn
        self.activation_kind = _activation_kind(experts.act_fn)
        self.gate_up_proj = experts.gate_up_proj
        self.down_proj = experts.down_proj
        self.lora_adapter: nn.Module | None = None

    def install_lora(self, *, rank: int, alpha: float, dropout: float) -> None:
        from dllm_parallel.core.adapters import GroupedExpertLoRA

        if self.lora_adapter is not None:
            raise RuntimeError("DiffusionGemma expert LoRA is already installed")
        self.lora_adapter = GroupedExpertLoRA(
            gate_up_weight=self.gate_up_proj,
            down_weight=self.down_proj,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            expert_parallel_sharded=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        return _local_grouped_routed_experts(
            hidden_states,
            top_k_index,
            top_k_weights,
            self.gate_up_proj,
            self.down_proj,
            self.activation_kind,
            lora_adapter=self.lora_adapter,
        )


class DiffusionGemmaExpertParallelExperts(nn.Module):
    """Shard DiffusionGemma routed experts over the expert-parallel group."""

    def __init__(self, experts: Any, runtime: Any) -> None:
        super().__init__()
        self.hidden_dim = int(experts.hidden_dim)
        self.intermediate_dim = int(experts.intermediate_dim)
        self.num_experts = int(experts.num_experts)
        self.ep_size = int(getattr(runtime, "expert_parallel_size", 1) or 1)
        self.ep_rank = int(getattr(runtime, "expert_parallel_rank", 0) or 0)
        self.ep_group = getattr(runtime, "expert_parallel_group", None)
        if self.ep_size <= 1 or self.ep_group is None:
            raise ValueError("DiffusionGemma EP experts require an expert group")
        if self.num_experts % self.ep_size != 0:
            raise ValueError(
                "DiffusionGemma num_experts must be divisible by "
                "expert_parallel_size"
            )
        if self.hidden_dim % _DEEPEP_HIDDEN_ALIGNMENT != 0:
            raise ValueError(
                "DiffusionGemma DeepEP requires hidden size divisible by "
                f"{_DEEPEP_HIDDEN_ALIGNMENT}; found {self.hidden_dim}"
            )
        self.ep_transport = _expert_transport(runtime)
        from dllm_parallel.core.parallel.expert import (
            begin_dispatch_tokens,
            combine_tokens,
            finish_dispatch_tokens,
        )

        self._begin_dispatch_tokens = begin_dispatch_tokens
        self._combine_tokens = combine_tokens
        self._finish_dispatch_tokens = finish_dispatch_tokens
        self.local_num_experts = self.num_experts // self.ep_size
        self.local_start = self.ep_rank * self.local_num_experts
        self.local_stop = self.local_start + self.local_num_experts
        self.act_fn = experts.act_fn
        self.activation_kind = _activation_kind(experts.act_fn)
        self.gate_up_proj = nn.Parameter(
            experts.gate_up_proj.detach()[self.local_start : self.local_stop]
            .clone()
            .contiguous()
        )
        self.down_proj = nn.Parameter(
            experts.down_proj.detach()[self.local_start : self.local_stop]
            .clone()
            .contiguous()
        )
        self.gate_up_proj._dllm_expert_parallel_sharded = True
        self.down_proj._dllm_expert_parallel_sharded = True
        self.lora_adapter: nn.Module | None = None

    def install_lora(self, *, rank: int, alpha: float, dropout: float) -> None:
        from dllm_parallel.core.adapters import GroupedExpertLoRA

        if self.lora_adapter is not None:
            raise RuntimeError("DiffusionGemma expert LoRA is already installed")
        self.lora_adapter = GroupedExpertLoRA(
            gate_up_weight=self.gate_up_proj,
            down_weight=self.down_proj,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            expert_parallel_sharded=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.ndim != 2:
            raise ValueError("DiffusionGemma EP experts expect flattened token states")
        pending = self.begin_dispatch(hidden_states, top_k_index, top_k_weights)
        return self.finish_dispatch(pending)

    def begin_dispatch(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> DiffusionGemmaPendingExpertDispatch:
        """Launch DeepEP dispatch for exact routed expert execution."""

        if hidden_states.ndim != 2:
            raise ValueError("DiffusionGemma EP experts expect flattened token states")
        hidden = int(hidden_states.shape[1])
        if hidden != self.hidden_dim:
            raise ValueError("DiffusionGemma EP hidden size mismatch")
        pending_dispatch = self._begin_dispatch_tokens(
            hidden_states,
            top_k_index,
            top_k_weights,
            num_experts=self.num_experts,
            group=self.ep_group,
            transport=self.ep_transport,
        )
        return DiffusionGemmaPendingExpertDispatch(
            pending_dispatch=pending_dispatch,
            input_dtype=hidden_states.dtype,
        )

    def finish_dispatch(
        self,
        pending: DiffusionGemmaPendingExpertDispatch,
    ) -> torch.Tensor:
        """Run local grouped experts and DeepEP combine for a pending dispatch."""

        routed_hidden, counts, state = self._finish_dispatch_tokens(
            pending.pending_dispatch
        )
        routed_output = self._local_expert_forward(routed_hidden, counts)
        return self._combine_tokens(routed_output, state).to(dtype=pending.input_dtype)

    def _local_expert_forward(
        self,
        hidden_states: torch.Tensor,
        counts: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.numel() == 0:
            return hidden_states
        return _grouped_gated_experts(
            hidden_states,
            self.gate_up_proj,
            self.down_proj,
            counts,
            self.activation_kind,
            lora_adapter=self.lora_adapter,
        )


def _grouped_gated_experts(
    hidden_states: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    counts: torch.Tensor,
    activation_kind: str,
    lora_adapter: nn.Module | None = None,
) -> torch.Tensor:
    _validate_grouped_expert_inputs(hidden_states, gate_up_proj, down_proj, counts)
    grouped_mm = getattr(F, "grouped_mm", None)
    if (
        callable(grouped_mm)
        and hidden_states.is_cuda
        and hidden_states.dtype == torch.bfloat16
    ):
        return _torch_grouped_gated_experts(
            grouped_mm,
            hidden_states,
            gate_up_proj,
            down_proj,
            counts,
            activation_kind,
            lora_adapter=lora_adapter,
        )
    return _reference_grouped_gated_experts(
        hidden_states,
        gate_up_proj,
        down_proj,
        counts,
        activation_kind,
        lora_adapter=lora_adapter,
    )


def _local_grouped_routed_experts(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    activation_kind: str,
    lora_adapter: nn.Module | None = None,
) -> torch.Tensor:
    """Compact local routes, execute each expert once, and restore token order."""

    if hidden_states.ndim != 2:
        raise ValueError("DiffusionGemma experts expect flattened token states")
    if top_k_index.shape != top_k_weights.shape or top_k_index.ndim != 2:
        raise ValueError("DiffusionGemma top-k indices and weights must match")
    if int(top_k_index.shape[0]) != int(hidden_states.shape[0]):
        raise ValueError("DiffusionGemma routes must cover every input token")

    num_experts = int(gate_up_proj.shape[0])
    routes_per_token = int(top_k_index.shape[1])
    flat_experts = top_k_index.reshape(-1).to(dtype=torch.long)
    flat_tokens = torch.arange(
        int(hidden_states.shape[0]),
        device=hidden_states.device,
        dtype=torch.long,
    ).repeat_interleave(routes_per_token)
    order = torch.argsort(flat_experts, stable=True)
    sorted_experts = flat_experts.index_select(0, order)
    sorted_tokens = flat_tokens.index_select(0, order)
    routed_hidden = hidden_states.index_select(0, sorted_tokens)
    counts = torch.bincount(sorted_experts, minlength=num_experts)
    routed_output = _grouped_gated_experts(
        routed_hidden,
        gate_up_proj,
        down_proj,
        counts,
        activation_kind,
        lora_adapter=lora_adapter,
    )
    routed_weights = top_k_weights.reshape(-1).index_select(0, order)
    routed_output = routed_output * routed_weights[:, None].to(routed_output.dtype)
    output = torch.zeros_like(hidden_states)
    output.index_add_(0, sorted_tokens, routed_output)
    return output


def _torch_grouped_gated_experts(
    grouped_mm: Any,
    hidden_states: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    counts: torch.Tensor,
    activation_kind: str,
    lora_adapter: nn.Module | None = None,
) -> torch.Tensor:
    """Execute jagged local experts with PyTorch's differentiable grouped GEMM."""

    offsets = torch.cumsum(
        counts.to(device=hidden_states.device, dtype=torch.int32),
        dim=0,
        dtype=torch.int32,
    )
    gate_up = grouped_mm(
        hidden_states,
        gate_up_proj.transpose(-2, -1),
        offs=offsets,
        out_dtype=hidden_states.dtype,
    )
    if lora_adapter is not None:
        gate_up = gate_up + lora_adapter.gate_up_delta(
            hidden_states,
            counts,
            grouped_mm,
        )
    gate, up = gate_up.chunk(2, dim=-1)
    activated = _activation_forward(gate, activation_kind) * up
    output = grouped_mm(
        activated,
        down_proj.transpose(-2, -1),
        offs=offsets,
        out_dtype=hidden_states.dtype,
    )
    if lora_adapter is not None:
        output = output + lora_adapter.down_delta(activated, counts, grouped_mm)
    return output


def _reference_grouped_gated_experts(
    hidden_states: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    counts: torch.Tensor,
    activation_kind: str,
    lora_adapter: nn.Module | None = None,
) -> torch.Tensor:
    """Exact portable fallback used outside the production BF16 CUDA path."""

    split_sizes = tuple(
        int(count)
        for count in counts.detach().to(device="cpu", dtype=torch.int64).tolist()
    )
    if sum(split_sizes) != int(hidden_states.shape[0]):
        raise ValueError("DiffusionGemma expert counts do not cover routed tokens")
    expert_inputs = hidden_states.split(split_sizes, dim=0)
    outputs: list[torch.Tensor] = []
    gate_up_delta = (
        lora_adapter.gate_up_delta(hidden_states, counts, None)
        if lora_adapter is not None
        else None
    )
    delta_inputs = gate_up_delta.split(split_sizes, dim=0) if gate_up_delta is not None else ()
    activated_outputs: list[torch.Tensor] = []
    for expert_index, expert_input in enumerate(expert_inputs):
        gate_up = F.linear(expert_input, gate_up_proj[expert_index])
        if gate_up_delta is not None:
            gate_up = gate_up + delta_inputs[expert_index]
        gate, up = gate_up.chunk(2, dim=-1)
        activated = _activation_forward(gate, activation_kind) * up
        activated_outputs.append(activated)
        outputs.append(F.linear(activated, down_proj[expert_index]))
    if lora_adapter is not None:
        activated = torch.cat(activated_outputs, dim=0)
        down_delta = lora_adapter.down_delta(activated, counts, None)
        outputs = [
            output + delta
            for output, delta in zip(outputs, down_delta.split(split_sizes, dim=0), strict=True)
        ]
    return torch.cat(outputs, dim=0)


def _validate_grouped_expert_inputs(
    hidden_states: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    counts: torch.Tensor,
) -> None:
    if counts.ndim != 1:
        raise ValueError("DiffusionGemma expert counts must be one-dimensional")
    if gate_up_proj.ndim != 3 or down_proj.ndim != 3:
        raise ValueError("DiffusionGemma expert projections must be three-dimensional")
    if int(counts.numel()) != int(gate_up_proj.shape[0]):
        raise ValueError("DiffusionGemma expert count and projection sizes differ")
    if down_proj.shape[0] != gate_up_proj.shape[0]:
        raise ValueError("DiffusionGemma gate/up and down expert counts differ")
    if gate_up_proj.shape[2] != hidden_states.shape[1]:
        raise ValueError("DiffusionGemma gate/up projection hidden size mismatch")
    if down_proj.shape[1] != hidden_states.shape[1]:
        raise ValueError("DiffusionGemma down projection hidden size mismatch")
    if gate_up_proj.shape[1] != 2 * down_proj.shape[2]:
        raise ValueError("DiffusionGemma gated expert intermediate sizes differ")


def _activation_kind(act_fn: Any) -> str:
    name = getattr(act_fn, "__name__", "")
    qualname = getattr(act_fn, "__qualname__", "")
    module_name = act_fn.__class__.__name__
    text = " ".join(str(part).lower() for part in (name, qualname, module_name, act_fn))
    if "silu" in text or "swish" in text:
        return "silu"
    if "gelu_pytorch_tanh" in text or "gelu_new" in text or "gelu" in text:
        return "gelu_tanh"
    raise RuntimeError(f"unsupported DiffusionGemma expert activation: {act_fn!r}")


def _activation_forward(x: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "silu":
        return torch.nn.functional.silu(x)
    if kind == "gelu_tanh":
        return torch.nn.functional.gelu(x, approximate="tanh")
    raise RuntimeError(f"unsupported gated expert activation kind: {kind}")


def install_diffusion_gemma_expert_parallel(model: Any, runtime: Any | None) -> None:
    ep_size = int(getattr(runtime, "expert_parallel_size", 1) or 1)
    decoder = _decoder(model)
    for layer in getattr(decoder, "layers", ()):
        experts = getattr(layer, "experts", None)
        if experts is None or isinstance(
            experts,
            (DiffusionGemmaGroupedExperts, DiffusionGemmaExpertParallelExperts),
        ):
            continue
        layer.experts = (
            DiffusionGemmaGroupedExperts(experts)
            if ep_size <= 1
            else DiffusionGemmaExpertParallelExperts(experts, runtime)
        )


def _decoder(model: Any) -> Any:
    current = model
    for name in ("model", "decoder"):
        current = getattr(current, name, current)
    if hasattr(current, "layers"):
        return current
    nested = getattr(current, "decoder", None)
    if nested is not None and hasattr(nested, "layers"):
        return nested
    raise TypeError("DiffusionGemma model does not expose decoder layers")


def _expert_transport(runtime: Any) -> str:
    """Select DeepEP transport from the physical expert-group placement."""

    ranks = tuple(getattr(runtime, "expert_parallel_group_ranks", ()) or ())
    node_size = getattr(runtime, "node_size", None)
    if not ranks or node_size is None or int(node_size) <= 0:
        raise ValueError(
            "DiffusionGemma expert parallelism requires resolved physical "
            "expert-group ranks and node size"
        )
    nodes = {int(rank) // int(node_size) for rank in ranks}
    return "nvlink" if len(nodes) == 1 else "elastic"
