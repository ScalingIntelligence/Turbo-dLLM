"""Differentiable intermediate states for the FLA Gated DeltaNet operator."""

from __future__ import annotations

import math
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any

import torch
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

from dllm_parallel.core.kernels.gated_delta_boundary import (
    chunk_gated_delta_rule_bwd_dhu_with_boundaries,
    chunk_gated_delta_rule_fwd_h_with_boundaries,
)

_SUPPORTED_FLA_VERSION = "0.5.2"
_RECURRENCE_CHUNK_SIZE = 64


@dataclass(frozen=True)
class GatedDeltaBoundaryPlan:
    """Device-resident mapping from recurrence chunks to requested states."""

    slots: torch.Tensor
    boundary_tokens: tuple[int, ...]
    final_slot: int | None

    @property
    def num_boundaries(self) -> int:
        return len(self.boundary_tokens)


def gated_delta_boundary_chunk_size() -> int:
    """Return the sequence alignment required by the pinned FLA recurrence."""

    return _RECURRENCE_CHUNK_SIZE


def verify_gated_delta_boundary_runtime() -> dict[str, object]:
    """Verify the exact FLA implementation consumed by this kernel."""

    installed = version("flash-linear-attention")
    if installed != _SUPPORTED_FLA_VERSION:
        raise RuntimeError(
            "Qwen3.8 boundary-state training requires flash-linear-attention "
            f"{_SUPPORTED_FLA_VERSION}, found {installed}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3.8 boundary-state training requires CUDA")
    return {
        "gated_delta_boundary_backend": "fla_triton_differentiable_boundaries",
        "flash_linear_attention_version": installed,
        "gated_delta_boundary_chunk_size": _RECURRENCE_CHUNK_SIZE,
    }


def build_gated_delta_boundary_plan(
    *,
    tokens: int,
    boundary_tokens: tuple[int, ...],
    device: torch.device,
) -> GatedDeltaBoundaryPlan:
    """Build immutable boundary metadata once for repeated layer execution."""

    if tokens <= 0 or tokens % _RECURRENCE_CHUNK_SIZE:
        raise ValueError("Gated DeltaNet token count must divide into FLA chunks")
    if not boundary_tokens:
        raise ValueError("at least one nonzero boundary is required")
    if len(set(boundary_tokens)) != len(boundary_tokens):
        raise ValueError("boundary tokens must be unique")
    chunks = tokens // _RECURRENCE_CHUNK_SIZE
    slot_values = [-1] * chunks
    final_slot = None
    for slot, token in enumerate(boundary_tokens):
        if token <= 0 or token > tokens:
            raise ValueError("boundary tokens must lie within the sequence endpoint")
        if token % _RECURRENCE_CHUNK_SIZE:
            raise ValueError("boundary tokens must align to FLA recurrence chunks")
        if token == tokens:
            final_slot = int(slot)
        else:
            slot_values[token // _RECURRENCE_CHUNK_SIZE] = int(slot)
    slots = torch.tensor(slot_values, dtype=torch.int32, device=device)
    return GatedDeltaBoundaryPlan(
        slots=slots,
        boundary_tokens=tuple(int(token) for token in boundary_tokens),
        final_slot=final_slot,
    )


class _ChunkGatedDeltaRuleBoundaries(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
        boundary_slots: torch.Tensor,
        num_boundaries: int,
        final_boundary_slot: int,
        scale: float,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from fla.modules.l2norm import l2norm_fwd
        from fla.ops.common.chunk_o import chunk_fwd_o
        from fla.ops.common.gate import fused_beta_sigmoid
        from fla.ops.gated_delta_rule.chunk_fwd import (
            chunk_gated_delta_rule_fwd_intra,
        )
        from fla.ops.gated_delta_rule.gate import gdn_gate_chunk_cumsum
        from fla.ops.utils.constant import RCP_LN2

        query, query_rstd = l2norm_fwd(query)
        key, key_rstd = l2norm_fwd(key)
        beta_raw = beta
        beta = fused_beta_sigmoid(beta_raw, scale=1.0)
        gate_input = gate
        gate = gdn_gate_chunk_cumsum(
            g=gate,
            A_log=A_log,
            chunk_size=_RECURRENCE_CHUNK_SIZE,
            scale=RCP_LN2,
            dt_bias=dt_bias,
        )
        weight, updated_value, inverse = chunk_gated_delta_rule_fwd_intra(
            k=key,
            v=value,
            g=gate,
            beta=beta,
            chunk_size=_RECURRENCE_CHUNK_SIZE,
        )
        states, recurrent_value, final_state, boundaries = (
            chunk_gated_delta_rule_fwd_h_with_boundaries(
                k=key,
                w=weight,
                u=updated_value,
                g=gate,
                chunk_size=_RECURRENCE_CHUNK_SIZE,
                boundary_slots=boundary_slots,
                num_boundaries=int(num_boundaries),
                output_final_state=int(final_boundary_slot) >= 0,
            )
        )
        if boundaries is None:
            raise RuntimeError("Gated DeltaNet boundary kernel returned no states")
        if int(final_boundary_slot) >= 0:
            if final_state is None:
                raise RuntimeError("Gated DeltaNet boundary kernel returned no final state")
            boundaries[:, int(final_boundary_slot)].copy_(final_state)
        output = chunk_fwd_o(
            q=query,
            k=key,
            v=recurrent_value,
            h=states,
            g=gate,
            scale=float(scale),
            chunk_size=_RECURRENCE_CHUNK_SIZE,
        )
        ctx.save_for_backward(
            query,
            query_rstd,
            key,
            key_rstd,
            value,
            gate,
            beta_raw,
            beta,
            inverse,
            boundary_slots,
            gate_input,
            A_log,
            dt_bias,
        )
        ctx.scale = float(scale)
        ctx.num_boundaries = int(num_boundaries)
        ctx.final_boundary_slot = int(final_boundary_slot)
        return output.to(dtype=query.dtype), boundaries

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
        grad_boundaries: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, ...]:
        from fla.modules.l2norm import l2norm_bwd
        from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
        from fla.ops.common.chunk_o import chunk_bwd_dqkwg, chunk_bwd_dv_local
        from fla.ops.common.gate import fused_beta_sigmoid_bwd
        from fla.ops.gated_delta_rule.gate import gdn_gate_bwd
        from fla.ops.gated_delta_rule.wy_fast import (
            prepare_wy_repr_bwd,
            recompute_w_u_fwd,
        )
        from fla.ops.utils import chunk_local_cumsum

        (
            query,
            query_rstd,
            key,
            key_rstd,
            value,
            gate,
            beta_raw,
            beta,
            inverse,
            boundary_slots,
            gate_input,
            A_log,
            dt_bias,
        ) = ctx.saved_tensors
        if grad_boundaries is None:
            grad_boundaries = torch.zeros(
                query.shape[0],
                int(ctx.num_boundaries),
                value.shape[2],
                query.shape[-1],
                value.shape[-1],
                device=query.device,
                dtype=torch.float32,
            )
        weight, updated_value = recompute_w_u_fwd(
            k=key,
            v=value,
            beta=beta,
            A=inverse,
            g=gate,
        )
        states, recurrent_value, _ = chunk_gated_delta_rule_fwd_h(
            k=key,
            w=weight,
            u=updated_value,
            g=gate,
            chunk_size=_RECURRENCE_CHUNK_SIZE,
        )
        value_gradient = chunk_bwd_dv_local(
            q=query,
            k=key,
            g=gate,
            do=grad_output,
            scale=ctx.scale,
            chunk_size=_RECURRENCE_CHUNK_SIZE,
        )
        final_state_gradient = (
            grad_boundaries[:, int(ctx.final_boundary_slot)].contiguous()
            if int(ctx.final_boundary_slot) >= 0
            else None
        )
        state_gradient, _, value_gradient = (
            chunk_gated_delta_rule_bwd_dhu_with_boundaries(
                q=query,
                k=key,
                w=weight,
                g=gate,
                do=grad_output,
                dv=value_gradient,
                scale=ctx.scale,
                chunk_size=_RECURRENCE_CHUNK_SIZE,
                boundary_state_grads=grad_boundaries.contiguous(),
                boundary_slots=boundary_slots,
                dht=final_state_gradient,
            )
        )
        query_gradient, key_gradient, weight_gradient, gate_gradient = (
            chunk_bwd_dqkwg(
                q=query,
                k=key,
                v=recurrent_value,
                w=weight,
                g=gate,
                h=states,
                dv=value_gradient,
                do=grad_output,
                dh=state_gradient,
                scale=ctx.scale,
                chunk_size=_RECURRENCE_CHUNK_SIZE,
            )
        )
        key_residual, value_gradient, beta_gradient, gate_residual = (
            prepare_wy_repr_bwd(
                k=key,
                v=value,
                beta=beta,
                g=gate,
                A=inverse,
                dw=weight_gradient,
                du=value_gradient,
            )
        )
        key_gradient.add_(key_residual)
        gate_gradient.add_(gate_residual)
        gate_gradient = chunk_local_cumsum(
            gate_gradient,
            chunk_size=_RECURRENCE_CHUNK_SIZE,
            reverse=True,
        )
        gate_gradient, A_log_gradient, dt_bias_gradient = gdn_gate_bwd(
            g=gate_input,
            A_log=A_log,
            dt_bias=dt_bias,
            dyg=gate_gradient,
        )
        query_gradient = l2norm_bwd(query, query_rstd, query_gradient)
        key_gradient = l2norm_bwd(key, key_rstd, key_gradient)
        beta_gradient = fused_beta_sigmoid_bwd(beta_raw, beta_gradient, scale=1.0)
        return (
            query_gradient.to(query),
            key_gradient.to(key),
            value_gradient.to(value),
            gate_gradient.to(gate_input),
            beta_gradient.to(beta_raw),
            None,
            None,
            None,
            None,
            A_log_gradient,
            dt_bias_gradient,
        )


def chunk_gated_delta_rule_with_boundaries(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    *,
    plan: GatedDeltaBoundaryPlan,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate one GDN sequence and return selected differentiable FP32 states."""

    if plan.slots.device != query.device:
        raise ValueError("boundary plan must be colocated with the recurrence inputs")
    if plan.slots.numel() * _RECURRENCE_CHUNK_SIZE != int(query.shape[1]):
        raise ValueError("boundary plan does not match the recurrence token count")
    return _ChunkGatedDeltaRuleBoundaries.apply(
        query,
        key,
        value,
        gate,
        beta,
        plan.slots,
        plan.num_boundaries,
        int(plan.final_slot if plan.final_slot is not None else -1),
        float(scale if scale is not None else 1.0 / math.sqrt(query.shape[-1])),
        A_log,
        dt_bias,
    )


__all__ = [
    "GatedDeltaBoundaryPlan",
    "build_gated_delta_boundary_plan",
    "chunk_gated_delta_rule_with_boundaries",
    "gated_delta_boundary_chunk_size",
    "verify_gated_delta_boundary_runtime",
]
