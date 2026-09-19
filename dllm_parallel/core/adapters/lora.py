# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""LoRA for the packed linear and grouped-expert modules used in training."""

from __future__ import annotations

import math
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from typing import Any, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F


_LORA_TOKEN_MASK: ContextVar[torch.Tensor | bool | None] = ContextVar(
    "dllm_parallel_lora_token_mask",
    default=None,
)


@contextmanager
def lora_token_mask(mask: torch.Tensor | bool | None) -> Iterator[None]:
    """Apply packed LoRA updates only on the selected token rows.

    Context-local routing makes nested active/teacher projections explicit and
    ensures activation-checkpoint replay observes the mask established by the
    replayed layer body.
    """

    if mask is not None and not isinstance(mask, (torch.Tensor, bool)):
        raise TypeError("LoRA token mask must be a tensor, bool, or None")
    token = _LORA_TOKEN_MASK.set(mask)
    try:
        yield
    finally:
        _LORA_TOKEN_MASK.reset(token)


def current_lora_token_mask() -> torch.Tensor | bool | None:
    """Return the role mask active for the current projection call."""

    return _LORA_TOKEN_MASK.get()


@dataclass(frozen=True)
class LoRAInstallation:
    rank: int
    alpha: float
    dropout: float
    targets: tuple[str, ...]
    linear_modules: int
    expert_modules: int
    trainable_parameters: int
    total_parameters: int


class LoRALinear(nn.Module):
    """Add a low-rank update to one local packed linear projection."""

    def __init__(
        self,
        base: nn.Module,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        weight = _local_weight(base)
        if weight.ndim != 2:
            raise TypeError("LoRA linear base weight must be two-dimensional")
        self.base = base
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0.0 else nn.Identity()
        self.lora_a = nn.Parameter(
            torch.empty(
                self.rank,
                int(weight.shape[1]),
                device=weight.device,
                dtype=weight.dtype,
            )
        )
        self.lora_b = nn.Parameter(
            torch.zeros(
                int(weight.shape[0]),
                self.rank,
                device=weight.device,
                dtype=weight.dtype,
            )
        )
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.lora_a._dllm_lora_parameter = True
        self.lora_b._dllm_lora_parameter = True

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base_output = self.base(hidden_states)
        mask = _LORA_TOKEN_MASK.get()
        if mask is False:
            return base_output
        if isinstance(mask, torch.Tensor):
            expected_shape = tuple(hidden_states.shape[:-1])
            if tuple(mask.shape) != expected_shape:
                raise ValueError(
                    "LoRA token mask shape must match input token rows: "
                    f"{tuple(mask.shape)} != {expected_shape}"
                )
            flat_hidden = hidden_states.reshape(-1, int(hidden_states.shape[-1]))
            selected = mask.to(device=hidden_states.device, dtype=torch.bool).reshape(-1)
            selected_hidden = flat_hidden[selected]
            selected_update = F.linear(
                F.linear(self.dropout(selected_hidden), self.lora_a),
                self.lora_b,
            )
            update = base_output.new_zeros(
                (int(flat_hidden.shape[0]), int(base_output.shape[-1]))
            )
            update[selected] = selected_update
            update = update.reshape_as(base_output)
        else:
            update = F.linear(
                F.linear(self.dropout(hidden_states), self.lora_a),
                self.lora_b,
            )
        return base_output + update * self.scale


class SequenceParallelLoRALinear(nn.Module):
    """LoRA delta matching a TP column/row projection with SP token rows.

    Column projections gather the sequence rows and retain a sharded output
    width. Row projections sum the low-rank input contribution over TP before
    projecting back to hidden size and scattering the output rows. This keeps
    the adapter algebra identical to an unsharded ``B(A(x))`` update.
    """

    def __init__(
        self,
        base: nn.Module,
        *,
        rank: int,
        alpha: float,
        dropout: float,
        runtime: Any,
        parallel_mode: str,
    ) -> None:
        super().__init__()
        if parallel_mode not in {"column", "row"}:
            raise ValueError("TP LoRA parallel_mode must be column or row")
        weight = _local_weight(base)
        if weight.ndim != 2:
            raise TypeError("TP LoRA base weight must be two-dimensional")
        self.base = base
        self.runtime = runtime
        self.parallel_mode = str(parallel_mode)
        self._dllm_lora_parallel_mode = self.parallel_mode
        self._dllm_lora_fuses_rmsnorm = bool(
            getattr(base, "_dllm_lora_fuses_rmsnorm", False)
        )
        self._dllm_lora_norm_eps = float(
            getattr(base, "_dllm_lora_norm_eps", 1.0e-6)
        )
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0.0 else nn.Identity()
        self.lora_a = nn.Parameter(
            torch.empty(
                self.rank,
                int(weight.shape[1]),
                device=weight.device,
                dtype=weight.dtype,
            )
        )
        self.lora_b = nn.Parameter(
            torch.zeros(
                int(weight.shape[0]),
                self.rank,
                device=weight.device,
                dtype=weight.dtype,
            )
        )
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        for parameter in (self.lora_a, self.lora_b):
            parameter._dllm_lora_parameter = True
        if self.parallel_mode == "column":
            self.lora_a._dllm_sequence_parallel_replicated = True
            self.lora_b._dllm_tensor_parallel_sharded = True
        else:
            self.lora_a._dllm_tensor_parallel_sharded = True

    def _gather_rows(self, value: torch.Tensor) -> torch.Tensor:
        from dllm_parallel.core.parallel.tensor_parallel import (
            gather_from_sequence_parallel_region,
        )

        return gather_from_sequence_parallel_region(value, self.runtime)

    def _full_token_mask(
        self,
        mask: torch.Tensor | bool | None,
        *,
        local_rows: int,
        full_rows: int,
        device: torch.device,
    ) -> torch.Tensor | bool | None:
        if not isinstance(mask, torch.Tensor):
            return mask
        flat = mask.to(device=device, dtype=torch.bool).reshape(-1)
        if int(flat.numel()) == int(full_rows):
            return flat
        if int(flat.numel()) != int(local_rows):
            raise ValueError(
                "sequence-parallel LoRA token mask must match local or gathered rows: "
                f"{int(flat.numel())} not in ({int(local_rows)}, {int(full_rows)})"
            )
        return self._gather_rows(flat[:, None]).squeeze(-1).to(dtype=torch.bool)

    def _normalized_column_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gathered = self._gather_rows(hidden_states)
        if not bool(getattr(self.base, "_dllm_lora_fuses_rmsnorm", False)):
            return gathered
        weight = getattr(self.base, "layer_norm_weight", None)
        if not isinstance(weight, torch.Tensor):
            raise RuntimeError("fused TP LoRA projection is missing RMSNorm weight")
        eps = float(getattr(self.base, "_dllm_lora_norm_eps", 1.0e-6))
        variance = gathered.float().square().mean(dim=-1, keepdim=True)
        return (
            gathered.float() * torch.rsqrt(variance + eps)
        ).to(dtype=gathered.dtype) * weight

    @staticmethod
    def _masked_update(
        hidden_states: torch.Tensor,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        dropout: nn.Module,
        mask: torch.Tensor | bool | None,
    ) -> torch.Tensor:
        if mask is False:
            return hidden_states.new_zeros(
                (*hidden_states.shape[:-1], int(lora_b.shape[0]))
            )
        if isinstance(mask, torch.Tensor):
            flat = hidden_states.reshape(-1, int(hidden_states.shape[-1]))
            selected = mask.reshape(-1)
            selected_update = F.linear(
                F.linear(dropout(flat[selected]), lora_a),
                lora_b,
            )
            update = hidden_states.new_zeros((int(flat.shape[0]), int(lora_b.shape[0])))
            update[selected] = selected_update
            return update.view(*hidden_states.shape[:-1], int(lora_b.shape[0]))
        return F.linear(F.linear(dropout(hidden_states), lora_a), lora_b)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base_output = self.base(hidden_states)
        mask = current_lora_token_mask()
        if mask is False:
            return base_output
        if self.parallel_mode == "column":
            gathered = self._normalized_column_input(hidden_states)
            full_mask = self._full_token_mask(
                mask,
                local_rows=int(hidden_states.reshape(-1, hidden_states.shape[-1]).shape[0]),
                full_rows=int(gathered.reshape(-1, gathered.shape[-1]).shape[0]),
                device=hidden_states.device,
            )
            update = self._masked_update(
                gathered,
                self.lora_a,
                self.lora_b,
                self.dropout,
                full_mask,
            )
            if tuple(update.shape) != tuple(base_output.shape):
                raise RuntimeError(
                    "column-parallel LoRA output does not match its base projection: "
                    f"{tuple(update.shape)} != {tuple(base_output.shape)}"
                )
            return base_output + update * self.scale

        from dllm_parallel.core.parallel.tensor_parallel import (
            reduce_from_tensor_parallel_region,
            scatter_to_sequence_parallel_region,
        )

        local_rows = int(base_output.reshape(-1, base_output.shape[-1]).shape[0])
        full_rows = int(hidden_states.reshape(-1, hidden_states.shape[-1]).shape[0])
        full_mask = self._full_token_mask(
            mask,
            local_rows=local_rows,
            full_rows=full_rows,
            device=hidden_states.device,
        )
        if isinstance(full_mask, torch.Tensor):
            masked_hidden = hidden_states * full_mask.view(
                *hidden_states.shape[:-1], 1
            ).to(dtype=hidden_states.dtype)
        elif full_mask is False:
            return base_output
        else:
            masked_hidden = hidden_states
        local_low_rank = F.linear(self.dropout(masked_hidden), self.lora_a)
        low_rank = reduce_from_tensor_parallel_region(local_low_rank, self.runtime)
        full_update = F.linear(low_rank, self.lora_b)
        update = scatter_to_sequence_parallel_region(full_update, self.runtime)
        if tuple(update.shape) != tuple(base_output.shape):
            raise RuntimeError(
                "row-parallel LoRA output does not match its base projection: "
                f"{tuple(update.shape)} != {tuple(base_output.shape)}"
            )
        return base_output + update * self.scale


class GroupedExpertLoRA(nn.Module):
    """Low-rank updates for local sharded MoE gate/up and down projections."""

    def __init__(
        self,
        *,
        gate_up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        rank: int,
        alpha: float,
        dropout: float,
        expert_parallel_sharded: bool,
    ) -> None:
        super().__init__()
        if gate_up_weight.ndim != 3 or down_weight.ndim != 3:
            raise TypeError("grouped expert LoRA requires three-dimensional weights")
        if int(gate_up_weight.shape[0]) != int(down_weight.shape[0]):
            raise ValueError("grouped expert LoRA expert counts differ")
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0.0 else nn.Identity()
        experts = int(gate_up_weight.shape[0])
        self.gate_up_a = nn.Parameter(
            torch.empty(
                experts,
                self.rank,
                int(gate_up_weight.shape[2]),
                device=gate_up_weight.device,
                dtype=gate_up_weight.dtype,
            )
        )
        self.gate_up_b = nn.Parameter(
            torch.zeros(
                experts,
                int(gate_up_weight.shape[1]),
                self.rank,
                device=gate_up_weight.device,
                dtype=gate_up_weight.dtype,
            )
        )
        self.down_a = nn.Parameter(
            torch.empty(
                experts,
                self.rank,
                int(down_weight.shape[2]),
                device=down_weight.device,
                dtype=down_weight.dtype,
            )
        )
        self.down_b = nn.Parameter(
            torch.zeros(
                experts,
                int(down_weight.shape[1]),
                self.rank,
                device=down_weight.device,
                dtype=down_weight.dtype,
            )
        )
        nn.init.uniform_(
            self.gate_up_a,
            -1.0 / math.sqrt(int(gate_up_weight.shape[2])),
            1.0 / math.sqrt(int(gate_up_weight.shape[2])),
        )
        nn.init.uniform_(
            self.down_a,
            -1.0 / math.sqrt(int(down_weight.shape[2])),
            1.0 / math.sqrt(int(down_weight.shape[2])),
        )
        for parameter in self.parameters():
            parameter._dllm_lora_parameter = True
            parameter._dllm_expert_parallel_sharded = bool(expert_parallel_sharded)

    def gate_up_delta(
        self,
        hidden_states: torch.Tensor,
        counts: torch.Tensor,
        grouped_mm: Any | None,
    ) -> torch.Tensor:
        return self._project(
            self.dropout(hidden_states),
            self.gate_up_a,
            self.gate_up_b,
            counts,
            grouped_mm,
        ) * self.scale

    def down_delta(
        self,
        hidden_states: torch.Tensor,
        counts: torch.Tensor,
        grouped_mm: Any | None,
    ) -> torch.Tensor:
        return self._project(
            self.dropout(hidden_states),
            self.down_a,
            self.down_b,
            counts,
            grouped_mm,
        ) * self.scale

    @staticmethod
    def _project(
        hidden_states: torch.Tensor,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        counts: torch.Tensor,
        grouped_mm: Any | None,
    ) -> torch.Tensor:
        if (
            callable(grouped_mm)
            and hidden_states.is_cuda
            and hidden_states.dtype == torch.bfloat16
        ):
            offsets = torch.cumsum(
                counts.to(device=hidden_states.device, dtype=torch.int32),
                dim=0,
                dtype=torch.int32,
            )
            low_rank = grouped_mm(
                hidden_states,
                lora_a.transpose(-2, -1),
                offs=offsets,
                out_dtype=hidden_states.dtype,
            )
            return grouped_mm(
                low_rank,
                lora_b.transpose(-2, -1),
                offs=offsets,
                out_dtype=hidden_states.dtype,
            )
        return _reference_grouped_lora(hidden_states, lora_a, lora_b, counts)


def apply_lora(model: nn.Module, spec: Any, runtime: Any) -> LoRAInstallation | None:
    """Freeze the packed backbone and install role-based trainable adapters."""

    if str(spec.type) == "none":
        return None
    if str(spec.type) != "lora":
        raise ValueError(f"unsupported adapter type: {spec.type!r}")
    tp_size = int(getattr(runtime, "tensor_parallel_size", 1) or 1)
    if tp_size != 1 and not bool(getattr(runtime, "sequence_parallel", False)):
        raise ValueError(
            "tensor-parallel packed LoRA requires sequence_parallel=true"
        )

    targets = tuple(str(target) for target in spec.targets)
    packed_layers = tuple(getattr(model, "_te_packed_layers", ()) or ())
    linear_slots = _linear_adapter_slots(packed_layers, targets=targets)
    expert_installers = []
    if "experts" in targets:
        for module in model.modules():
            installer = getattr(module, "install_lora", None)
            if callable(installer):
                expert_installers.append(installer)

    requested_linear_roles = set(targets) & {"attention", "mlp"}
    observed_linear_roles = {role for role, _, _ in linear_slots}
    missing_roles = sorted(requested_linear_roles - observed_linear_roles)
    if missing_roles:
        raise RuntimeError(
            "LoRA found no supported packed projections for requested roles: "
            + ", ".join(missing_roles)
        )
    if "experts" in targets and not expert_installers:
        raise RuntimeError("LoRA found no supported expert projections")
    if not linear_slots and not expert_installers:
        raise RuntimeError(
            "LoRA found no supported packed projections; build the production packed "
            "backbone before installing adapters"
        )

    # Validate the complete installation plan before freezing or replacing any
    # modules.  This prevents a family/layout mismatch from leaving a partially
    # adapted model behind.
    for _, parent, attribute in linear_slots:
        module = getattr(parent, attribute)
        if isinstance(module, (LoRALinear, SequenceParallelLoRALinear)):
            raise RuntimeError("LoRA adapters are already installed")
        weight = _local_weight(module)
        if weight.ndim != 2:
            raise TypeError("LoRA linear base weight must be two-dimensional")

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    for _, parent, attribute in linear_slots:
        setattr(
            parent,
            attribute,
            _wrap_linear(
                getattr(parent, attribute),
                spec,
                runtime=runtime,
            ),
        )

    for installer in expert_installers:
        installer(
            rank=int(spec.rank),
            alpha=float(spec.alpha),
            dropout=float(spec.dropout),
        )
    linear_modules = len(linear_slots)
    expert_modules = len(expert_installers)
    trainable, total = trainable_parameter_summary(model)
    if trainable <= 0:
        raise RuntimeError("LoRA installation produced no trainable parameters")
    installation = LoRAInstallation(
        rank=int(spec.rank),
        alpha=float(spec.alpha),
        dropout=float(spec.dropout),
        targets=targets,
        linear_modules=linear_modules,
        expert_modules=expert_modules,
        trainable_parameters=trainable,
        total_parameters=total,
    )
    configure_checkpointing = getattr(model, "enable_adapter_checkpointing", None)
    if callable(configure_checkpointing):
        configure_checkpointing()
    model._dllm_adapter_metadata = asdict(installation)
    return installation


def _linear_adapter_slots(
    packed_layers: tuple[nn.Module, ...],
    *,
    targets: tuple[str, ...],
) -> list[tuple[str, nn.Module, str]]:
    """Resolve family-specific packed projections into stable adapter roles.

    Nemotron and DiffusionGemma expose ``qkv``/``o_proj`` while Qwen3.8's
    hybrid full-attention/GDN executor exposes ``mixer_input``/``mixer_output``.
    Both layouts use the same fused ``gate_up``/``down_proj`` MLP pair at TP1.
    LoRA intentionally rejects ambiguous or partial layouts instead of silently
    training only a subset of the requested projections.
    """

    slots: list[tuple[str, nn.Module, str]] = []
    for layer_index, layer in enumerate(packed_layers):
        if "attention" in targets:
            layouts = []
            for names in (("qkv", "o_proj"), ("mixer_input", "mixer_output")):
                present = tuple(getattr(layer, name, None) is not None for name in names)
                if any(present) and not all(present):
                    raise RuntimeError(
                        "LoRA packed attention layout is incomplete at layer "
                        f"{layer_index}: {names}"
                    )
                if all(present):
                    layouts.append(names)
            if len(layouts) != 1:
                raise RuntimeError(
                    "LoRA requires exactly one supported packed attention layout at "
                    f"layer {layer_index}; found {len(layouts)}"
                )
            slots.extend(("attention", layer, name) for name in layouts[0])

        if "mlp" in targets:
            names = ("gate_up", "down_proj")
            present = tuple(getattr(layer, name, None) is not None for name in names)
            if not all(present):
                raise RuntimeError(
                    "LoRA packed MLP layout is incomplete at layer "
                    f"{layer_index}: {names}"
                )
            slots.extend(("mlp", layer, name) for name in names)
    return slots


def adapter_metadata(model: nn.Module) -> dict[str, Any] | None:
    for module in model.modules():
        value = getattr(module, "_dllm_adapter_metadata", None)
        if value is not None:
            return dict(value)
    return None


def trainable_parameter_summary(model: nn.Module) -> tuple[int, int]:
    total = sum(int(parameter.numel()) for parameter in model.parameters())
    trainable = sum(
        int(parameter.numel())
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return trainable, total


def validate_adapter_checkpoint(checkpoint: Any, model: nn.Module) -> None:
    saved = ((checkpoint or {}).get("backbone_state") or {}).get("adapter")
    current = adapter_metadata(model)
    if saved is None and current is None:
        return
    if saved is None or current is None:
        raise RuntimeError("checkpoint and current run use different adapter modes")
    fields = ("rank", "alpha", "dropout", "targets")
    mismatches: dict[str, tuple[Any, Any]] = {}
    for field in fields:
        saved_value = saved.get(field)
        current_value = current.get(field)
        if field == "targets":
            saved_value = tuple(saved_value or ())
            current_value = tuple(current_value or ())
        if saved_value != current_value:
            mismatches[field] = (saved_value, current_value)
    if mismatches:
        raise RuntimeError(f"checkpoint LoRA configuration mismatch: {mismatches}")


def _wrap_linear(
    module: nn.Module,
    spec: Any,
    *,
    runtime: Any,
) -> LoRALinear | SequenceParallelLoRALinear:
    if isinstance(module, (LoRALinear, SequenceParallelLoRALinear)):
        raise RuntimeError("LoRA adapters are already installed")
    tp_size = int(getattr(runtime, "tensor_parallel_size", 1) or 1)
    if tp_size > 1:
        parallel_mode = getattr(module, "_dllm_lora_parallel_mode", None)
        if parallel_mode not in {"column", "row"}:
            raise RuntimeError(
                "TP LoRA projection is missing column/row parallel metadata"
            )
        return SequenceParallelLoRALinear(
            module,
            rank=int(spec.rank),
            alpha=float(spec.alpha),
            dropout=float(spec.dropout),
            runtime=runtime,
            parallel_mode=str(parallel_mode),
        )
    return LoRALinear(
        module,
        rank=int(spec.rank),
        alpha=float(spec.alpha),
        dropout=float(spec.dropout),
    )


def _local_weight(module: nn.Module) -> torch.Tensor:
    weight = getattr(module, "weight", None)
    if weight is None:
        raise TypeError(f"{type(module).__name__} does not expose a weight")
    to_local = getattr(weight, "to_local", None)
    return to_local() if callable(to_local) else weight


def _reference_grouped_lora(
    hidden_states: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    split_sizes = tuple(
        int(value) for value in counts.detach().to(device="cpu", dtype=torch.int64).tolist()
    )
    if sum(split_sizes) != int(hidden_states.shape[0]):
        raise ValueError("expert counts do not cover grouped LoRA inputs")
    outputs = [
        F.linear(F.linear(expert_input, lora_a[index]), lora_b[index])
        for index, expert_input in enumerate(hidden_states.split(split_sizes, dim=0))
    ]
    return torch.cat(outputs, dim=0)
