# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Objective runtime state for standard block diffusion training."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch


LABEL_IGNORE_INDEX = -100
OBJECTIVE_STATE_VERSION = 3


@dataclass(frozen=True)
class CorruptedBatch:
    clean_input_ids: torch.Tensor
    noisy_input_ids: torch.Tensor
    labels: torch.Tensor
    diffusion_times: torch.Tensor
    noise_levels: torch.Tensor
    loss_weights: torch.Tensor
    active_tokens: torch.Tensor
    valid_tokens: int
    loss_denominator: float | torch.Tensor


@dataclass(frozen=True)
class TokenAccountingPolicy:
    global_batch_size: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    data_parallel_size: int
    context_parallel_size: int
    block_parallel_size: int
    tensor_parallel_size: int
    sequence_parallel: bool

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "global_batch_size": int(self.global_batch_size),
            "micro_batch_size": int(self.micro_batch_size),
            "gradient_accumulation_steps": int(self.gradient_accumulation_steps),
            "data_parallel_size": int(self.data_parallel_size),
            "context_parallel_size": int(self.context_parallel_size),
            "block_parallel_size": int(self.block_parallel_size),
            "tensor_parallel_size": int(self.tensor_parallel_size),
            "sequence_parallel": bool(self.sequence_parallel),
        }


class StandardBlockDiffusionObjectiveRuntime:
    """Owns corruption RNG, loss-token accounting, and objective state."""

    def __init__(
        self,
        *,
        mask_token_id: int,
        block_size: int,
        seq_len: int,
        device: torch.device,
        seed: int,
        token_accounting: TokenAccountingPolicy,
        noise_schedule: str = "loglinear",
        loss_weighting: str = "inverse_move_chance",
        noise_schedule_epsilon: float = 1.0e-3,
        sampling_epsilon_min: float = 1.0e-3,
        sampling_epsilon_max: float = 1.0,
        antithetic_sampling: bool = True,
    ) -> None:
        if seq_len <= 0 or block_size <= 0:
            raise ValueError("seq_len and block_size must be positive")
        if seq_len % block_size != 0:
            raise ValueError("seq_len must divide evenly by block_size")
        if noise_schedule != "loglinear":
            raise ValueError(
                "standard block diffusion currently requires noise_schedule=loglinear"
            )
        if loss_weighting not in {"inverse_move_chance", "unit"}:
            raise ValueError(
                "standard block diffusion loss_weighting must be "
                "'inverse_move_chance' or 'unit'"
            )
        if not (0.0 < float(noise_schedule_epsilon) < 1.0):
            raise ValueError("noise_schedule_epsilon must be in (0, 1)")
        if not (
            0.0 < float(sampling_epsilon_min) <= float(sampling_epsilon_max) <= 1.0
        ):
            raise ValueError("sampling epsilon bounds must satisfy 0 < min <= max <= 1")
        self.mask_token_id = int(mask_token_id)
        self.block_size = int(block_size)
        self.max_seq_len = int(seq_len)
        self.device = device
        self.noise_schedule = str(noise_schedule)
        self.loss_weighting = str(loss_weighting)
        self.noise_schedule_epsilon = float(noise_schedule_epsilon)
        self.sampling_epsilon_min = float(sampling_epsilon_min)
        self.sampling_epsilon_max = float(sampling_epsilon_max)
        self.antithetic_sampling = bool(antithetic_sampling)
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(int(seed))
        self.token_accounting = token_accounting
        self.batches_corrupted = 0
        self._loss_tokens_seen = torch.zeros((), dtype=torch.int64, device=device)
        self.valid_tokens_seen = 0

    def corrupt(
        self,
        x0: torch.Tensor,
        *,
        supervision_mask: torch.Tensor | None = None,
        supervision_count: int | None = None,
    ) -> CorruptedBatch:
        if x0.ndim != 2:
            raise ValueError(
                f"expected rank-2 input_ids, got {tuple(x0.shape)}"
            )
        batch_size = int(x0.shape[0])
        sequence_length = int(x0.shape[1])
        if not 0 < sequence_length <= self.max_seq_len:
            raise ValueError(
                "input sequence length must be positive and no greater than the "
                f"configured maximum ({self.max_seq_len})"
            )
        if sequence_length % self.block_size:
            raise ValueError("input sequence length must divide evenly by block_size")
        num_blocks = sequence_length // self.block_size
        if supervision_mask is None:
            if supervision_count is not None:
                raise ValueError(
                    "supervision_count requires an explicit supervision_mask"
                )
            eligible = None
            valid_tokens = int(x0.numel())
        else:
            if supervision_mask.shape != x0.shape:
                raise ValueError("supervision_mask must match input_ids")
            eligible = supervision_mask.to(device=x0.device, dtype=torch.bool)
            valid_tokens = (
                int(eligible.sum())
                if supervision_count is None
                else int(supervision_count)
            )
            if not 0 <= valid_tokens <= int(x0.numel()):
                raise ValueError("supervision_count is outside the batch token range")
        if valid_tokens <= 0:
            raise ValueError("each training batch must contain supervised tokens")
        diffusion_times = torch.rand(
            (1,) if self.antithetic_sampling else (batch_size, num_blocks),
            dtype=torch.float32,
            device=x0.device,
            generator=self.generator,
        )
        if self.antithetic_sampling:
            sample_count = batch_size * num_blocks
            offsets = torch.arange(
                sample_count,
                dtype=torch.float32,
                device=x0.device,
            ).view(batch_size, num_blocks)
            # Randomly rotate the strata: every fixed block must have a
            # uniform marginal over the entire noise schedule. Assigning
            # (rand[i] + i) / N permanently ties noise to sequence position.
            diffusion_times = (
                diffusion_times + offsets / float(sample_count)
            ).remainder_(1.0)
        diffusion_times = (
            diffusion_times * (self.sampling_epsilon_max - self.sampling_epsilon_min)
            + self.sampling_epsilon_min
        )
        move_chance = diffusion_times
        loss_weights = (
            torch.ones_like(diffusion_times)
            if self.loss_weighting == "unit"
            else diffusion_times.reciprocal()
        )
        noise_levels = -torch.log1p(-diffusion_times)
        noise_levels.clamp_(max=-math.log(self.noise_schedule_epsilon))
        token_move_chance = move_chance.repeat_interleave(self.block_size, dim=1)
        mask = (
            torch.rand(
                x0.shape,
                dtype=torch.float32,
                device=x0.device,
                generator=self.generator,
            )
            <= token_move_chance
        )
        if eligible is not None:
            mask.logical_and_(eligible)
        xt = torch.where(mask, torch.full_like(x0, self.mask_token_id), x0)
        labels = torch.where(mask, x0, torch.full_like(x0, LABEL_IGNORE_INDEX))
        active_tokens = mask.sum(dtype=torch.int64)
        self.batches_corrupted += 1
        self._loss_tokens_seen.add_(active_tokens)
        self.valid_tokens_seen += valid_tokens
        return CorruptedBatch(
            clean_input_ids=x0,
            noisy_input_ids=xt,
            labels=labels,
            diffusion_times=diffusion_times,
            noise_levels=noise_levels,
            loss_weights=loss_weights,
            active_tokens=active_tokens,
            valid_tokens=valid_tokens,
            loss_denominator=float(valid_tokens),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "standard_block_diffusion",
            "version": OBJECTIVE_STATE_VERSION,
            "mask_token_id": int(self.mask_token_id),
            "block_size": int(self.block_size),
            "seq_len": int(self.max_seq_len),
            "noise_schedule": self.noise_schedule,
            "loss_weighting": self.loss_weighting,
            "noise_schedule_epsilon": self.noise_schedule_epsilon,
            "sampling_epsilon_min": self.sampling_epsilon_min,
            "sampling_epsilon_max": self.sampling_epsilon_max,
            "antithetic_sampling": self.antithetic_sampling,
            "generator_state": self.generator.get_state(),
            "batches_corrupted": int(self.batches_corrupted),
            "loss_tokens_seen": self._loss_tokens_seen.detach().cpu(),
            "valid_tokens_seen": int(self.valid_tokens_seen),
            "token_accounting": self.token_accounting.to_log_dict(),
        }

    def load_state_dict(self, state: dict[str, Any] | None) -> None:
        if not state:
            return
        if state.get("kind") != "standard_block_diffusion":
            raise RuntimeError(
                "checkpoint does not contain standard block-diffusion state"
            )
        if state.get("version") != OBJECTIVE_STATE_VERSION:
            raise RuntimeError(
                "checkpoint objective state version is incompatible with the exact "
                "blockwise schedule runtime"
            )
        for key, expected in (
            ("mask_token_id", self.mask_token_id),
            ("block_size", self.block_size),
            ("seq_len", self.max_seq_len),
        ):
            observed = state.get(key)
            if observed is not None and int(observed) != int(expected):
                raise RuntimeError(
                    f"checkpoint objective {key} does not match current run: "
                    f"{observed} != {expected}"
                )
        for key, expected in (
            ("noise_schedule", self.noise_schedule),
            ("loss_weighting", self.loss_weighting),
            ("noise_schedule_epsilon", self.noise_schedule_epsilon),
            ("sampling_epsilon_min", self.sampling_epsilon_min),
            ("sampling_epsilon_max", self.sampling_epsilon_max),
            ("antithetic_sampling", self.antithetic_sampling),
        ):
            observed = state.get(key)
            if observed is not None and observed != expected:
                raise RuntimeError(
                    f"checkpoint objective {key} does not match current run: "
                    f"{observed} != {expected}"
                )
        generator_state = state.get("generator_state")
        if generator_state is not None:
            self.generator.set_state(generator_state.cpu())
        self.batches_corrupted = int(state.get("batches_corrupted", 0))
        self._loss_tokens_seen.copy_(
            torch.as_tensor(
                state.get("loss_tokens_seen", 0),
                dtype=torch.int64,
                device=self.device,
            )
        )
        self.valid_tokens_seen = int(state.get("valid_tokens_seen", 0))

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "kind": "standard_block_diffusion",
            "version": OBJECTIVE_STATE_VERSION,
            "block_size": int(self.block_size),
            "seq_len": int(self.max_seq_len),
            "mask_token_id": int(self.mask_token_id),
            "noise_schedule": self.noise_schedule,
            "loss_weighting": self.loss_weighting,
            "noise_schedule_epsilon": self.noise_schedule_epsilon,
            "sampling_epsilon_min": self.sampling_epsilon_min,
            "sampling_epsilon_max": self.sampling_epsilon_max,
            "antithetic_sampling": self.antithetic_sampling,
            "batches_corrupted": int(self.batches_corrupted),
            "loss_tokens_seen": int(self.loss_tokens_seen),
            "valid_tokens_seen": int(self.valid_tokens_seen),
            "token_accounting": self.token_accounting.to_log_dict(),
        }

    @property
    def loss_tokens_seen(self) -> int:
        return int(self._loss_tokens_seen)
