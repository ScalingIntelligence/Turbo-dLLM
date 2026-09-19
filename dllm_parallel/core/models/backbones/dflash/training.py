# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""DFlash objective preparation and distributed anchor ownership."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch
from dllm_parallel.core.objectives.dflash import (
    DFlashObjectiveBatch,
    DFlashObjectiveRuntime,
)
from dllm_parallel.core.profiling.perf import DFLASH_WORK_COUNT_NAMES


@dataclass(frozen=True)
class DFlashPreparedBatch:
    data: Any
    objective: DFlashObjectiveBatch
    normalization_count: torch.Tensor
    teacher_hidden_states: torch.Tensor | None
    performance_counts: torch.Tensor


class DFlashTrainingTask:
    def __init__(
        self,
        *,
        block_size: int,
        max_anchors: int,
        anchor_sampling: str = "uniform",
        anchor_group_size: int = 0,
        mask_token_id: int,
        decay_gamma: float,
        sample_from_anchor: bool,
        loss_kind: str,
        device: torch.device,
        seed: int,
        bp_loss_scale: float | None,
        runtime: Any | None,
        sequence_length: int,
        sliding_window: int | None,
        dpace_alpha: float = 0.5,
        lk_loss_type: str | None = None,
        kl_scale: float = 1.0,
        kl_decay: float = 1.0,
        selector_loss_alpha: float = 1.0,
        selector_warmup_ratio: float = 0.0,
        selector_ramp_ratio: float = 0.0,
        selector_stop_gradient: bool = False,
        vocab_block_size: int = 32768,
    ) -> None:
        self.loss_kind = str(loss_kind)
        self.bp_loss_scale = bp_loss_scale
        self.runtime = runtime
        self.sequence_length = int(sequence_length)
        self.sliding_window = (
            int(sliding_window) if sliding_window is not None else None
        )
        self.performance_count_names = DFLASH_WORK_COUNT_NAMES
        self.dpace_alpha = float(dpace_alpha)
        self.lk_loss_type = lk_loss_type
        self.kl_scale = float(kl_scale)
        self.kl_decay = float(kl_decay)
        self.selector_loss_alpha = float(selector_loss_alpha)
        self.selector_warmup_ratio = float(selector_warmup_ratio)
        self.selector_ramp_ratio = float(selector_ramp_ratio)
        self.selector_stop_gradient = bool(selector_stop_gradient)
        self.vocab_block_size = int(vocab_block_size)
        self.effective_selector_loss_alpha = self.selector_loss_alpha
        self.objective_runtime = DFlashObjectiveRuntime(
            block_size=int(block_size),
            max_anchors=int(max_anchors),
            anchor_sampling=str(anchor_sampling),
            anchor_group_size=int(anchor_group_size),
            mask_token_id=int(mask_token_id),
            decay_gamma=float(decay_gamma),
            sample_from_anchor=bool(sample_from_anchor),
            device=device,
            seed=int(seed),
        )

    def accumulation_loss_denominator(self, batches: list[Any]) -> None:
        del batches
        return None

    def set_optimizer_step(self, step: int, total_steps: int) -> None:
        """Update the SpecForge-compatible selector warmup/ramp schedule."""

        warmup_steps = int(max(1, total_steps) * self.selector_warmup_ratio)
        if int(step) < warmup_steps:
            factor = 0.0
        else:
            ramp_steps = int(max(1, total_steps) * self.selector_ramp_ratio)
            factor = (
                1.0
                if ramp_steps <= 0
                else min(max((int(step) - warmup_steps + 1) / ramp_steps, 0.0), 1.0)
            )
        self.effective_selector_loss_alpha = self.selector_loss_alpha * factor

    def prepare(
        self,
        batch: Any,
        *,
        loss_denominator: float | torch.Tensor | None = None,
    ) -> DFlashPreparedBatch:
        if loss_denominator is not None:
            raise RuntimeError("DFlash does not accept an external loss denominator")
        if batch.loss_mask is None or batch.position_ids is None:
            raise RuntimeError("DFlash data must provide loss_mask and position_ids")
        if batch.target_hidden_states is None:
            raise RuntimeError("DFlash data must provide target_hidden_states")
        if self.loss_kind == "speculators_kl" and batch.verifier_last_hidden_states is None:
            raise RuntimeError("DFlash KL data must provide verifier_last_hidden_states")
        full_objective = self.objective_runtime.prepare(
            batch.input_ids,
            batch.loss_mask,
            batch.document_ids,
        )
        performance_counts = dflash_performance_counts(
            full_objective,
            sequence_length=self.sequence_length,
            sliding_window=self.sliding_window,
        )
        anchor_indices, global_anchor_count = _owned_anchor_selection(
            full_objective,
            runtime=self.runtime,
            device=batch.input_ids.device,
        )
        objective = full_objective.select_anchors(
            anchor_indices,
            global_anchor_count=global_anchor_count,
        )
        teacher_hidden_states = None
        if self.loss_kind == "speculators_kl":
            verifier_features = batch.verifier_last_hidden_states
            if verifier_features is None:
                raise RuntimeError("DFlash KL data must provide verifier hidden states")
            teacher_hidden_states = verifier_features.gather(
                objective.teacher_source_positions,
                device=batch.input_ids.device,
            )
            batch = replace(batch, verifier_last_hidden_states=None)
        return DFlashPreparedBatch(
            data=batch,
            objective=objective,
            normalization_count=full_objective.valid_supervised_tokens,
            teacher_hidden_states=teacher_hidden_states,
            performance_counts=performance_counts,
        )

    def forward(self, model: Any, prepared: DFlashPreparedBatch) -> Any:
        data = prepared.data
        return model(
            target_hidden_states=data.target_hidden_states,
            teacher_hidden_states=prepared.teacher_hidden_states,
            input_ids=data.input_ids,
            position_ids=data.position_ids,
            target_position_ids=data.target_position_ids,
            objective=prepared.objective,
            compute_teacher=self.loss_kind == "speculators_kl",
        )

    def loss(self, model: Any, prepared: DFlashPreparedBatch, output: Any) -> torch.Tensor:
        unwrapped = getattr(model, "module", model)
        loss_fn = getattr(unwrapped, "distributed_dflash_loss", None)
        if not callable(loss_fn):
            raise RuntimeError("DFlash models must implement distributed_dflash_loss")
        kwargs = {
            "output": output,
            "loss_kind": self.loss_kind,
            "normalization_count": prepared.normalization_count,
            "bp_loss_scale": self.bp_loss_scale,
        }
        if hasattr(unwrapped, "candidate_selector"):
            kwargs.update(
                dpace_alpha=self.dpace_alpha,
                lk_loss_type=self.lk_loss_type,
                kl_scale=self.kl_scale,
                kl_decay=self.kl_decay,
                selector_loss_alpha=self.effective_selector_loss_alpha,
                selector_stop_gradient=self.selector_stop_gradient,
                vocab_block_size=self.vocab_block_size,
            )
        return loss_fn(**kwargs)

    def output_shape(self, output: Any) -> tuple[int, ...]:
        return tuple(output.hidden_states.shape)

    def skip(self, data_runtime: Any, count: int) -> None:
        for _ in range(max(0, int(count))):
            self.prepare(data_runtime.next_batch())


def build_training_task(
    *,
    spec: Any,
    runtime: Any,
    device: torch.device,
    seed: int,
    data_parallel_seed: int,
    mask_token_id: int,
    vocab_size: int,
    token_accounting: Any,
    block_size: int,
    model_metadata: Any,
) -> DFlashTrainingTask:
    del seed, vocab_size, token_accounting
    objective = spec.objective
    if objective.max_anchors is None:
        raise ValueError("DFlash requires max_anchors")
    if objective.decay_gamma is None:
        raise ValueError("DFlash requires decay_gamma")
    return DFlashTrainingTask(
        block_size=int(block_size),
        max_anchors=int(objective.max_anchors),
        anchor_sampling=str(objective.anchor_sampling),
        anchor_group_size=int(objective.anchor_group_size),
        mask_token_id=int(mask_token_id),
        decay_gamma=float(objective.decay_gamma),
        sample_from_anchor=bool(objective.sample_from_anchor),
        loss_kind=str(objective.dflash_loss),
        device=device,
        seed=int(data_parallel_seed),
        bp_loss_scale=objective.bp_loss_scale,
        runtime=runtime,
        sequence_length=int(spec.model.seq_len),
        sliding_window=getattr(model_metadata, "sliding_window", None),
        dpace_alpha=float(objective.dpace_alpha),
        lk_loss_type=objective.lk_loss_type,
        kl_scale=float(objective.kl_scale),
        kl_decay=float(objective.kl_decay),
        selector_loss_alpha=float(objective.selector_loss_alpha),
        selector_warmup_ratio=float(objective.selector_warmup_ratio),
        selector_ramp_ratio=float(objective.selector_ramp_ratio),
        selector_stop_gradient=bool(objective.selector_stop_gradient),
        vocab_block_size=int(objective.dflash_vocab_block_size),
    )


def dflash_performance_counts(
    objective: DFlashObjectiveBatch,
    *,
    sequence_length: int,
    sliding_window: int | None,
) -> torch.Tensor:
    """Return topology-independent useful-work counts for DFlash MFU."""

    if sequence_length <= 0:
        raise ValueError("DFlash sequence length must be positive")
    if sliding_window is not None and sliding_window <= 0:
        raise ValueError("DFlash sliding window must be positive")
    valid = objective.anchor_valid.to(torch.int64)
    context_widths = (
        objective.context_stops.to(torch.int64)
        - objective.document_starts.to(torch.int64)
    ).clamp_min(0) * valid
    block_size = int(objective.block_positions.shape[-1])
    if sliding_window is None:
        sliding_context_pairs = context_widths.sum() * block_size
    else:
        per_slot_width = (
            int(sliding_window)
            - 1
            - torch.arange(
                block_size,
                dtype=torch.int64,
                device=objective.anchor_valid.device,
            )
        ).clamp_min(0)
        sliding_context_pairs = torch.minimum(
            context_widths.unsqueeze(-1),
            per_slot_width,
        ).sum()
    context_rows = (
        torch.full(
            (),
            int(objective.anchor_valid.shape[0]) * int(sequence_length),
            dtype=torch.int64,
            device=objective.anchor_valid.device,
        )
        if sliding_window is None
        else _sliding_context_union_rows(
            objective,
            sequence_length=int(sequence_length),
            sliding_window=int(sliding_window),
        )
    )
    return torch.stack(
        (
            context_rows,
            valid.sum(),
            context_widths.sum() * block_size,
            sliding_context_pairs,
            objective.supervised_mask.sum(dtype=torch.int64),
        )
    )


def _sliding_context_union_rows(
    objective: DFlashObjectiveBatch,
    *,
    sequence_length: int,
    sliding_window: int,
) -> torch.Tensor:
    """Count context rows whose tokenwise projections affect any query."""

    batch_size = int(objective.anchor_valid.shape[0])
    starts = torch.maximum(
        objective.document_starts.to(torch.int64),
        objective.context_stops.to(torch.int64) - (int(sliding_window) - 1),
    ).clamp(0, int(sequence_length))
    stops = objective.context_stops.to(torch.int64).clamp(0, int(sequence_length))
    valid = objective.anchor_valid & stops.gt(starts)
    delta = torch.zeros(
        (batch_size, int(sequence_length) + 1),
        dtype=torch.int32,
        device=objective.anchor_valid.device,
    )
    signed = valid.to(torch.int32)
    delta.scatter_add_(1, starts, signed)
    delta.scatter_add_(1, stops, -signed)
    return delta[:, :-1].cumsum(dim=1).gt(0).sum(dtype=torch.int64)


def owned_anchor_indices(
    objective: DFlashObjectiveBatch,
    *,
    runtime: Any | None,
    device: torch.device,
) -> torch.Tensor:
    return _owned_anchor_selection(
        objective,
        runtime=runtime,
        device=device,
    )[0]


def _owned_anchor_selection(
    objective: DFlashObjectiveBatch,
    *,
    runtime: Any | None,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    num_anchors = int(objective.anchor_positions.shape[1])
    candidates = torch.nonzero(
        objective.anchor_valid.any(dim=0),
        as_tuple=False,
    ).flatten()
    if candidates.numel() == 0:
        # Keep invalid placeholder anchors, including on CP-only ranks. Their
        # loss is zero, but every owner still traverses the collective graph.
        candidates = torch.arange(num_anchors, device=device, dtype=torch.int64)
    if runtime is None:
        if candidates.numel() == 0:
            return (
                torch.arange(num_anchors, device=device, dtype=torch.int64),
                num_anchors,
            )
        return candidates, int(candidates.numel())
    if getattr(runtime, "active_block_mode", None) != "dual_end":
        return candidates, int(candidates.numel())
    owner_count = int(getattr(runtime, "block_parallel_size", 1))
    if owner_count <= 1:
        raise RuntimeError(
            "DFlash target-anchor ownership requires block_parallel_size > 1"
        )
    if int(candidates.numel()) < owner_count:
        candidates = torch.arange(num_anchors, device=device, dtype=torch.int64)
    global_anchor_count = int(candidates.numel())
    size = int(runtime.block_parallel_size)
    rank = int(runtime.block_parallel_rank)
    if size <= 1:
        return candidates, global_anchor_count
    costs = (
        (objective.context_stops - objective.document_starts)
        .clamp_min(0)
        .to(torch.int64)
        .mul(objective.anchor_valid.to(torch.int64))
        .sum(dim=0)
    ).index_select(0, candidates)
    sorted_by_cost = torch.argsort(costs, descending=False, stable=True)
    slots = torch.arange(candidates.numel(), device=device, dtype=torch.int64)
    pair = torch.div(slots, 2, rounding_mode="floor")
    paired_indices = torch.where(
        slots.remainder(2).eq(0),
        sorted_by_cost.numel() - 1 - pair,
        pair,
    )
    balanced_order = candidates.index_select(
        0,
        sorted_by_cost.index_select(0, paired_indices),
    )
    # When there are fewer pairs than owners, distribute individual anchors.
    # Preserve the established paired schedule for all sufficiently large jobs.
    owner_slots = slots if int(candidates.numel()) < 2 * size else pair
    owned = balanced_order[owner_slots.remainder(size).eq(rank)]

    # Real trajectory supervision is commonly much shorter than max_anchors.
    # Passing exact valid-anchor counts through static compiled DFlash2 regions
    # would create one graph per trajectory length. Pad fused-BP packets with
    # globally invalid slots into bounded power-of-two buckets instead. These
    # slots have zero supervision and therefore zero loss and gradient.
    if 0 < int(candidates.numel()) < num_anchors and num_anchors % size == 0:
        owner_counts = [
            _balanced_owner_slot_count(int(candidates.numel()), size, owner_rank)
            for owner_rank in range(size)
        ]
        required = max(owner_counts)
        max_per_owner = num_anchors // size
        minimum_bucket = min(128, max_per_owner)
        power_of_two_bucket = 1 << max(0, required - 1).bit_length()
        bucket = min(max_per_owner, max(minimum_bucket, power_of_two_bucket))
        candidate_mask = torch.zeros(num_anchors, dtype=torch.bool, device=device)
        candidate_mask[candidates] = True
        padding_candidates = torch.nonzero(~candidate_mask, as_tuple=False).flatten()
        deficits = [bucket - count for count in owner_counts]
        padding_start = sum(deficits[:rank])
        padding_stop = padding_start + deficits[rank]
        owned = torch.cat(
            (owned, padding_candidates[padding_start:padding_stop]),
            dim=0,
        )
        global_anchor_count = bucket * size
    return owned.sort().values, global_anchor_count


def _balanced_owner_slot_count(count: int, size: int, rank: int) -> int:
    """Return the exact count produced by the paired owner-slot schedule."""

    if count < 2 * size:
        return int(rank < count)
    pairs, singleton = divmod(count, 2)
    per_owner_pairs, extra_pairs = divmod(pairs, size)
    owned = 2 * (per_owner_pairs + int(rank < extra_pairs))
    if singleton and rank == pairs % size:
        owned += 1
    return owned


__all__ = [
    "build_training_task",
    "DFlashPreparedBatch",
    "DFlashTrainingTask",
    "dflash_performance_counts",
    "owned_anchor_indices",
]
