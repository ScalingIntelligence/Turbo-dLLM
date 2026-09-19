# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Exact anchor-block objective state for DFlash training."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Any

import torch
import torch.nn.functional as F


DFLASH_OBJECTIVE = "dflash_distillation"
DFLASH_OBJECTIVE_STATE_VERSION = 1


@dataclass(frozen=True)
class DFlashObjectiveBatch:
    anchor_positions: torch.Tensor
    anchor_valid: torch.Tensor
    block_positions: torch.Tensor
    teacher_source_positions: torch.Tensor
    target_token_positions: torch.Tensor
    draft_input_ids: torch.Tensor
    supervised_mask: torch.Tensor
    position_weights: torch.Tensor
    document_starts: torch.Tensor
    context_stops: torch.Tensor
    global_anchor_count: int = 0

    @property
    def valid_supervised_tokens(self) -> torch.Tensor:
        return self.supervised_mask.sum(dim=1, dtype=torch.float32)

    def select_anchors(
        self,
        indices: torch.Tensor,
        *,
        global_anchor_count: int | None = None,
    ) -> "DFlashObjectiveBatch":
        batch_size, anchors = self.anchor_positions.shape
        block_size = self.block_positions.shape[-1]

        def select(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.index_select(1, indices)

        return DFlashObjectiveBatch(
            anchor_positions=select(self.anchor_positions),
            anchor_valid=select(self.anchor_valid),
            block_positions=select(self.block_positions),
            teacher_source_positions=select(self.teacher_source_positions),
            target_token_positions=select(self.target_token_positions),
            draft_input_ids=select(
                self.draft_input_ids.view(batch_size, anchors, block_size)
            )
            .reshape(batch_size, -1)
            .contiguous(),
            supervised_mask=select(
                self.supervised_mask.view(batch_size, anchors, block_size)
            )
            .reshape(batch_size, -1)
            .contiguous(),
            position_weights=select(
                self.position_weights.view(batch_size, anchors, block_size)
            )
            .reshape(batch_size, -1)
            .contiguous(),
            document_starts=select(self.document_starts),
            context_stops=select(self.context_stops),
            global_anchor_count=(
                int(global_anchor_count)
                if global_anchor_count is not None
                else int(self.global_anchor_count)
            ),
        )


class DFlashObjectiveRuntime:
    """Samples DFlash anchors and owns reproducible objective state."""

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
        device: torch.device,
        seed: int,
    ) -> None:
        if block_size < 2:
            raise ValueError("DFlash block_size must be at least two")
        if max_anchors <= 0:
            raise ValueError("DFlash max_anchors must be positive")
        if anchor_sampling not in {"uniform", "locality"}:
            raise ValueError("DFlash anchor_sampling must be uniform or locality")
        if anchor_group_size < 0 or anchor_group_size > max_anchors:
            raise ValueError("DFlash anchor_group_size must be in [0, max_anchors]")
        if decay_gamma <= 0:
            raise ValueError("DFlash decay_gamma must be positive")
        self.block_size = int(block_size)
        self.max_anchors = int(max_anchors)
        self.anchor_sampling = str(anchor_sampling)
        self.anchor_group_size = int(anchor_group_size)
        self.mask_token_id = int(mask_token_id)
        self.decay_gamma = float(decay_gamma)
        self.sample_from_anchor = bool(sample_from_anchor)
        self.device = device
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(int(seed))
        self.batches_sampled = 0
        self.anchors_sampled = 0
        self.supervised_tokens_seen = torch.zeros((), device=device, dtype=torch.int64)

    def prepare(
        self,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        document_ids: torch.Tensor | None = None,
    ) -> DFlashObjectiveBatch:
        _validate_inputs(input_ids, loss_mask, document_ids)
        batch_size, sequence_length = input_ids.shape
        if document_ids is None:
            document_ids = torch.zeros_like(input_ids)
        document_ids = document_ids.to(device=input_ids.device, dtype=torch.int64)
        valid_anchor = _valid_anchor_mask(
            loss_mask=loss_mask,
            document_ids=document_ids,
        )
        anchor_positions, anchor_valid = _sample_anchors(
            valid_anchor,
            max_anchors=self.max_anchors,
            sampling=self.anchor_sampling,
            group_size=self.anchor_group_size,
            group_stride=self.block_size,
            generator=self.generator,
        )
        offsets = torch.arange(
            self.block_size,
            device=input_ids.device,
            dtype=torch.int64,
        )
        block_positions = anchor_positions.unsqueeze(-1) + offsets
        if self.sample_from_anchor:
            target_token_positions = block_positions + 1
        else:
            target_token_positions = block_positions

        safe_block_positions = block_positions.clamp_max(sequence_length - 1)
        safe_target_positions = target_token_positions.clamp_max(sequence_length - 1)
        teacher_source_positions = (
            safe_block_positions
            if self.sample_from_anchor
            else (safe_target_positions - 1).clamp_min(0)
        )
        draft_input_ids = torch.full(
            (batch_size, self.max_anchors, self.block_size),
            self.mask_token_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        anchor_tokens = input_ids.gather(1, anchor_positions)
        draft_input_ids[:, :, 0] = anchor_tokens
        supervised_mask = (
            loss_mask.to(torch.bool)
            .gather(
                1,
                safe_target_positions.reshape(batch_size, -1),
            )
            .reshape(batch_size, self.max_anchors, self.block_size)
        )
        target_in_bounds = target_token_positions.lt(sequence_length)
        document_starts = _document_starts(document_ids)
        anchor_document_starts = document_starts.gather(1, anchor_positions)
        # IDs may be reused in disjoint segments. Segment starts, unlike raw
        # IDs, cannot re-enable supervision after crossing another document.
        target_documents = document_starts.gather(
            1,
            safe_target_positions.reshape(batch_size, -1),
        ).reshape(batch_size, self.max_anchors, self.block_size)
        anchor_documents = anchor_document_starts.unsqueeze(-1)
        supervised_mask &= target_in_bounds
        supervised_mask &= target_documents.eq(anchor_documents)
        supervised_mask &= anchor_valid.unsqueeze(-1)
        if not self.sample_from_anchor:
            supervised_mask[:, :, 0] = False

        position_weights = _position_weights(
            block_size=self.block_size,
            gamma=self.decay_gamma,
            sample_from_anchor=self.sample_from_anchor,
            device=input_ids.device,
        ).view(1, 1, self.block_size)
        position_weights = position_weights.expand_as(supervised_mask).to(torch.float32)

        invalid_context = torch.full_like(anchor_positions, sequence_length)
        anchor_document_starts = torch.where(
            anchor_valid,
            anchor_document_starts,
            invalid_context,
        )
        context_stops = torch.where(
            anchor_valid,
            anchor_positions,
            invalid_context,
        )

        self.batches_sampled += 1
        self.anchors_sampled += batch_size * self.max_anchors
        self.supervised_tokens_seen += supervised_mask.sum(dtype=torch.int64)
        return DFlashObjectiveBatch(
            anchor_positions=anchor_positions,
            anchor_valid=anchor_valid,
            block_positions=safe_block_positions,
            teacher_source_positions=teacher_source_positions,
            target_token_positions=safe_target_positions,
            draft_input_ids=draft_input_ids.reshape(batch_size, -1).contiguous(),
            supervised_mask=supervised_mask.reshape(batch_size, -1).contiguous(),
            position_weights=position_weights.reshape(batch_size, -1).contiguous(),
            document_starts=anchor_document_starts.to(torch.int32),
            context_stops=context_stops.to(torch.int32),
            global_anchor_count=self.max_anchors,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": DFLASH_OBJECTIVE,
            "version": DFLASH_OBJECTIVE_STATE_VERSION,
            "block_size": self.block_size,
            "max_anchors": self.max_anchors,
            "anchor_sampling": self.anchor_sampling,
            "anchor_group_size": self.anchor_group_size,
            "mask_token_id": self.mask_token_id,
            "decay_gamma": self.decay_gamma,
            "sample_from_anchor": self.sample_from_anchor,
            "generator_state": self.generator.get_state(),
            "batches_sampled": self.batches_sampled,
            "anchors_sampled": self.anchors_sampled,
            "supervised_tokens_seen": self.supervised_tokens_seen.detach().cpu(),
        }

    def load_state_dict(self, state: dict[str, Any] | None) -> None:
        if not state:
            return
        if state.get("kind") != DFLASH_OBJECTIVE:
            raise RuntimeError("checkpoint does not contain DFlash objective state")
        if state.get("version") != DFLASH_OBJECTIVE_STATE_VERSION:
            raise RuntimeError(
                "checkpoint DFlash objective state version is incompatible"
            )
        expected = {
            "block_size": self.block_size,
            "max_anchors": self.max_anchors,
            "mask_token_id": self.mask_token_id,
            "decay_gamma": self.decay_gamma,
            "sample_from_anchor": self.sample_from_anchor,
        }
        checkpoint_sampling = str(state.get("anchor_sampling", "uniform"))
        checkpoint_group_size = int(state.get("anchor_group_size", 0))
        if checkpoint_sampling != self.anchor_sampling:
            raise RuntimeError(
                "checkpoint DFlash anchor_sampling does not match current run: "
                f"{checkpoint_sampling!r} != {self.anchor_sampling!r}"
            )
        if checkpoint_group_size != self.anchor_group_size:
            raise RuntimeError(
                "checkpoint DFlash anchor_group_size does not match current run: "
                f"{checkpoint_group_size!r} != {self.anchor_group_size!r}"
            )
        for name, value in expected.items():
            if state.get(name) != value:
                raise RuntimeError(
                    f"checkpoint DFlash {name} does not match current run: "
                    f"{state.get(name)!r} != {value!r}"
                )
        generator_state = state.get("generator_state")
        if generator_state is not None:
            self.generator.set_state(generator_state.cpu())
        self.batches_sampled = int(state.get("batches_sampled", 0))
        self.anchors_sampled = int(state.get("anchors_sampled", 0))
        self.supervised_tokens_seen = torch.as_tensor(
            state.get("supervised_tokens_seen", 0),
            device=self.device,
            dtype=torch.int64,
        )

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "kind": DFLASH_OBJECTIVE,
            "version": DFLASH_OBJECTIVE_STATE_VERSION,
            "block_size": self.block_size,
            "max_anchors": self.max_anchors,
            "anchor_sampling": self.anchor_sampling,
            "anchor_group_size": self.anchor_group_size,
            "mask_token_id": self.mask_token_id,
            "decay_gamma": self.decay_gamma,
            "sample_from_anchor": self.sample_from_anchor,
            "batches_sampled": self.batches_sampled,
            "anchors_sampled": self.anchors_sampled,
            "supervised_tokens_seen": int(self.supervised_tokens_seen.detach().cpu()),
        }


def reference_dflash_loss(
    draft_logits: torch.Tensor,
    *,
    supervised_mask: torch.Tensor,
    position_weights: torch.Tensor,
    teacher_logits: torch.Tensor | None = None,
    target_token_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference DFlash CE or forward-KL reduction for equivalence tests."""

    if draft_logits.ndim != 3:
        raise ValueError("DFlash logits must have shape [batch, tokens, vocabulary]")
    if supervised_mask.shape != draft_logits.shape[:2]:
        raise ValueError("DFlash supervised mask must match logit token dimensions")
    if position_weights.shape != supervised_mask.shape:
        raise ValueError("DFlash position weights must match the supervised mask")
    if (teacher_logits is None) == (target_token_ids is None):
        raise ValueError("provide exactly one of teacher_logits or target_token_ids")
    if teacher_logits is not None:
        if teacher_logits.shape != draft_logits.shape:
            raise ValueError("DFlash teacher logits must match draft logits")
        draft_log_prob = F.log_softmax(draft_logits, dim=-1, dtype=torch.float32)
        teacher_prob = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
        token_loss = F.kl_div(
            draft_log_prob,
            teacher_prob,
            reduction="none",
        ).sum(dim=-1)
    else:
        if target_token_ids is None or target_token_ids.shape != supervised_mask.shape:
            raise ValueError("DFlash target token IDs must match the supervised mask")
        token_loss = F.cross_entropy(
            draft_logits.float().reshape(-1, draft_logits.shape[-1]),
            target_token_ids.reshape(-1),
            reduction="none",
        ).reshape_as(supervised_mask)
    mask = supervised_mask.to(torch.float32)
    numerator = (token_loss * position_weights.to(torch.float32) * mask).sum(dim=1)
    denominator = mask.sum(dim=1) + 1.0e-5
    return (numerator / denominator).mean()


def _validate_inputs(
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    document_ids: torch.Tensor | None,
) -> None:
    if input_ids.ndim != 2:
        raise ValueError("DFlash input_ids must have shape [batch, sequence]")
    if loss_mask.shape != input_ids.shape:
        raise ValueError("DFlash loss_mask must match input_ids")
    if document_ids is not None and document_ids.shape != input_ids.shape:
        raise ValueError("DFlash document_ids must match input_ids")
    if input_ids.shape[1] == 0:
        raise ValueError("DFlash sequences must be nonempty")


def _valid_anchor_mask(
    *,
    loss_mask: torch.Tensor,
    document_ids: torch.Tensor,
) -> torch.Tensor:
    valid = loss_mask.to(torch.bool).clone()
    sequence_length = int(valid.shape[1])
    positions = torch.arange(
        sequence_length,
        device=valid.device,
        dtype=torch.int64,
    ).view(1, -1)
    next_positions = positions + 1
    in_bounds = next_positions.lt(sequence_length)
    safe_next = next_positions.clamp_max(sequence_length - 1).expand_as(document_ids)
    segment_ids = torch.zeros_like(document_ids, dtype=torch.int64)
    if sequence_length > 1:
        segment_ids[:, 1:] = document_ids[:, 1:].ne(document_ids[:, :-1])
        segment_ids = segment_ids.cumsum(dim=1)
    valid &= in_bounds
    valid &= document_ids.ne(-1)
    valid &= loss_mask.to(torch.bool).gather(1, safe_next)
    valid &= segment_ids.eq(segment_ids.gather(1, safe_next))
    return valid


def _sample_anchors(
    valid_anchor: torch.Tensor,
    *,
    max_anchors: int,
    sampling: str = "uniform",
    group_size: int = 0,
    group_stride: int = 1,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    if sampling == "locality":
        return _sample_locality_anchors(
            valid_anchor,
            max_anchors=max_anchors,
            group_size=group_size,
            group_stride=group_stride,
            generator=generator,
        )
    batch_size, sequence_length = valid_anchor.shape
    sample_count = min(max_anchors, sequence_length)
    scores = torch.rand(
        valid_anchor.shape,
        dtype=torch.float32,
        device=valid_anchor.device,
        generator=generator,
    )
    scores.masked_fill_(~valid_anchor, 2.0)
    selected_scores, selected = torch.topk(
        scores,
        k=sample_count,
        dim=1,
        largest=False,
        sorted=True,
    )
    selected_valid = selected_scores.lt(2.0)
    selected = torch.where(
        selected_valid,
        selected,
        torch.full_like(selected, sequence_length),
    )
    selected, order = selected.sort(dim=1)
    selected_valid = selected_valid.gather(1, order)
    anchors = torch.zeros(
        (batch_size, max_anchors),
        dtype=torch.int64,
        device=valid_anchor.device,
    )
    anchor_valid = torch.zeros(
        (batch_size, max_anchors),
        dtype=torch.bool,
        device=valid_anchor.device,
    )
    anchors[:, :sample_count] = torch.where(selected_valid, selected, 0)
    anchor_valid[:, :sample_count] = selected_valid
    return anchors, anchor_valid


def _sample_locality_anchors(
    valid_anchor: torch.Tensor,
    *,
    max_anchors: int,
    group_size: int,
    group_stride: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample locally correlated anchors with uniform per-slot marginals.

    A uniformly random cyclic start plus fixed offsets is uniform for every
    output slot. Correlating the slots changes estimator variance, not its
    expectation, and lets sliding context projections be reused across many
    draft blocks. The block-sized stride also avoids overlapping supervised
    blocks for the normal dense-loss-mask case.
    """

    batch_size, _ = valid_anchor.shape
    anchors = torch.zeros(
        (batch_size, max_anchors),
        dtype=torch.int64,
        device=valid_anchor.device,
    )
    anchor_valid = torch.zeros(
        (batch_size, max_anchors),
        dtype=torch.bool,
        device=valid_anchor.device,
    )
    stride = max(1, int(group_stride))
    for batch_index in range(batch_size):
        candidates = torch.nonzero(valid_anchor[batch_index], as_tuple=False).flatten()
        sample_count = min(int(max_anchors), int(candidates.numel()))
        if sample_count == 0:
            continue
        # A raw modular stride repeats after n/gcd(n, stride) positions. That
        # silently duplicates anchors for common dense response lengths (for
        # example, n divisible by block_size). Move to the nearest coprime
        # stride so this remains a single uniformly rotated permutation: every
        # slot has a uniform marginal and every selected anchor is unique.
        effective_stride = stride
        while gcd(effective_stride, int(candidates.numel())) != 1:
            effective_stride += 1
        start = torch.randint(
            int(candidates.numel()),
            (1,),
            device=valid_anchor.device,
            generator=generator,
        )
        offsets = torch.arange(
            sample_count,
            dtype=torch.int64,
            device=valid_anchor.device,
        )
        candidate_ranks = (start + offsets * effective_stride).remainder(
            int(candidates.numel())
        )
        selected = candidates.index_select(0, candidate_ranks)
        selected = selected.sort().values
        anchors[batch_index, :sample_count] = selected
        anchor_valid[batch_index, :sample_count] = True
    return anchors, anchor_valid


def _position_weights(
    *,
    block_size: int,
    gamma: float,
    sample_from_anchor: bool,
    device: torch.device,
) -> torch.Tensor:
    positions = torch.arange(block_size, dtype=torch.float32, device=device)
    if sample_from_anchor:
        return torch.exp(-positions / gamma)
    weights = torch.exp(-(positions - 1).clamp_min(0) / gamma)
    weights[0] = 0
    return weights


def _document_starts(document_ids: torch.Tensor) -> torch.Tensor:
    batch_size, sequence_length = document_ids.shape
    positions = (
        torch.arange(
            sequence_length,
            device=document_ids.device,
            dtype=torch.int64,
        )
        .view(1, -1)
        .expand(batch_size, -1)
    )
    starts = torch.zeros_like(document_ids)
    if sequence_length > 1:
        changed = document_ids[:, 1:].ne(document_ids[:, :-1])
        starts[:, 1:] = torch.where(changed, positions[:, 1:], 0)
    return torch.cummax(starts, dim=1).values


__all__ = [
    "DFLASH_OBJECTIVE",
    "DFlashObjectiveBatch",
    "DFlashObjectiveRuntime",
    "reference_dflash_loss",
]
