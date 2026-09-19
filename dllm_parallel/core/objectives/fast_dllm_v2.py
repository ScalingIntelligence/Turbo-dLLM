# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Fast-dLLM v2 complementary block-diffusion objective."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from dllm_parallel.core.objectives.block_diffusion import (
    standard_block_diffusion_schedule,
)
from dllm_parallel.core.objectives.runtime import (
    CorruptedBatch,
    LABEL_IGNORE_INDEX,
    TokenAccountingPolicy,
)


FAST_DLLM_V2_OBJECTIVE = "fast_dllm_v2"
FAST_DLLM_V2_STATE_VERSION = 1


class FastDLLMv2ObjectiveRuntime:
    """Generate the exact paired corruption used by Fast-dLLM v2 training.

    A single blockwise Bernoulli draw creates complementary views. Every
    supervised target except the unshiftable first sequence token is masked in
    exactly one view. Labels are moved to the preceding query row to match the
    causal-LM next-token loss used by the reference implementation.
    """

    def __init__(
        self,
        *,
        mask_token_id: int,
        block_size: int,
        seq_len: int,
        device: torch.device,
        seed: int,
        token_accounting: TokenAccountingPolicy,
        noise_schedule_epsilon: float = 1.0e-3,
    ) -> None:
        if seq_len <= 1:
            raise ValueError("Fast-dLLM v2 requires seq_len greater than one")
        if block_size <= 0 or seq_len % block_size:
            raise ValueError("seq_len must divide evenly by block_size")
        if not 0.0 < float(noise_schedule_epsilon) < 1.0:
            raise ValueError("noise_schedule_epsilon must be in (0, 1)")
        self.mask_token_id = int(mask_token_id)
        self.block_size = int(block_size)
        self.max_seq_len = int(seq_len)
        self.noise_schedule_epsilon = float(noise_schedule_epsilon)
        self.device = device
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
            raise ValueError(f"expected rank-2 input_ids, got {tuple(x0.shape)}")
        batch_size, sequence_length = map(int, x0.shape)
        if not 1 < sequence_length <= self.max_seq_len:
            raise ValueError(
                "input sequence length must be greater than one and no greater "
                f"than the configured maximum ({self.max_seq_len})"
            )
        if sequence_length % self.block_size:
            raise ValueError("input sequence length must divide evenly by block_size")

        if supervision_mask is None:
            if supervision_count is not None:
                raise ValueError(
                    "supervision_count requires an explicit supervision_mask"
                )
            eligible = torch.ones_like(x0, dtype=torch.bool)
        else:
            if supervision_mask.shape != x0.shape:
                raise ValueError("supervision_mask must match input_ids")
            eligible = supervision_mask.to(device=x0.device, dtype=torch.bool)

        # Position zero has no preceding query under the reference shifted loss.
        if supervision_mask is None:
            valid_tokens = batch_size * (sequence_length - 1)
        elif supervision_count is None:
            valid_tokens = int(eligible[:, 1:].sum())
        else:
            valid_tokens = int(supervision_count) - int(eligible[:, 0].sum())
        if valid_tokens <= 0:
            raise ValueError(
                "each training batch must contain shifted supervised tokens"
            )

        num_blocks = sequence_length // self.block_size
        diffusion_times = torch.rand(
            (batch_size, num_blocks),
            dtype=torch.float32,
            device=x0.device,
            generator=self.generator,
        )
        move_chance = (
            1.0 - self.noise_schedule_epsilon
        ) * diffusion_times + self.noise_schedule_epsilon
        token_move_chance = move_chance.repeat_interleave(self.block_size, dim=1)
        sampled_mask = (
            torch.rand(
                x0.shape,
                dtype=torch.float32,
                device=x0.device,
                generator=self.generator,
            )
            < token_move_chance
        )
        base_mask = sampled_mask & eligible
        complement_mask = (~sampled_mask) & eligible

        base_noisy = torch.where(
            base_mask,
            torch.full_like(x0, self.mask_token_id),
            x0,
        )
        complement_noisy = torch.where(
            complement_mask,
            torch.full_like(x0, self.mask_token_id),
            x0,
        )
        base_targets = torch.where(
            base_mask,
            x0,
            torch.full_like(x0, LABEL_IGNORE_INDEX),
        )
        complement_targets = torch.where(
            complement_mask,
            x0,
            torch.full_like(x0, LABEL_IGNORE_INDEX),
        )

        labels = torch.full(
            (2 * batch_size, sequence_length),
            LABEL_IGNORE_INDEX,
            dtype=x0.dtype,
            device=x0.device,
        )
        labels[:batch_size, :-1] = base_targets[:, 1:]
        labels[batch_size:, :-1] = complement_targets[:, 1:]

        active_tokens = (labels != LABEL_IGNORE_INDEX).sum(dtype=torch.int64)

        self.batches_corrupted += 1
        self._loss_tokens_seen.add_(active_tokens)
        self.valid_tokens_seen += valid_tokens
        return CorruptedBatch(
            clean_input_ids=torch.cat((x0, x0), dim=0),
            noisy_input_ids=torch.cat((base_noisy, complement_noisy), dim=0),
            labels=labels,
            diffusion_times=torch.cat((diffusion_times, diffusion_times), dim=0),
            noise_levels=torch.cat((move_chance, move_chance), dim=0),
            loss_weights=torch.ones(
                (2 * batch_size, num_blocks),
                dtype=torch.float32,
                device=x0.device,
            ),
            active_tokens=active_tokens,
            valid_tokens=valid_tokens,
            loss_denominator=float(valid_tokens),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": FAST_DLLM_V2_OBJECTIVE,
            "version": FAST_DLLM_V2_STATE_VERSION,
            "mask_token_id": self.mask_token_id,
            "block_size": self.block_size,
            "seq_len": self.max_seq_len,
            "noise_schedule_epsilon": self.noise_schedule_epsilon,
            "generator_state": self.generator.get_state(),
            "batches_corrupted": self.batches_corrupted,
            "loss_tokens_seen": self._loss_tokens_seen.detach().cpu(),
            "valid_tokens_seen": self.valid_tokens_seen,
            "token_accounting": self.token_accounting.to_log_dict(),
        }

    def load_state_dict(self, state: dict[str, Any] | None) -> None:
        if not state:
            return
        if state.get("kind") != FAST_DLLM_V2_OBJECTIVE:
            raise RuntimeError("checkpoint does not contain Fast-dLLM v2 state")
        if state.get("version") != FAST_DLLM_V2_STATE_VERSION:
            raise RuntimeError("checkpoint Fast-dLLM v2 state version is incompatible")
        for key, expected in (
            ("mask_token_id", self.mask_token_id),
            ("block_size", self.block_size),
            ("seq_len", self.max_seq_len),
        ):
            if int(state.get(key, expected)) != int(expected):
                raise RuntimeError(
                    f"checkpoint objective {key} does not match current run"
                )
        observed_epsilon = float(
            state.get("noise_schedule_epsilon", self.noise_schedule_epsilon)
        )
        if observed_epsilon != self.noise_schedule_epsilon:
            raise RuntimeError(
                "checkpoint objective noise_schedule_epsilon does not match current run"
            )
        generator_state = state.get("generator_state")
        if generator_state is None:
            raise RuntimeError(
                "checkpoint Fast-dLLM v2 state is missing generator_state"
            )
        self.generator.set_state(generator_state.to(device="cpu"))
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
            "kind": FAST_DLLM_V2_OBJECTIVE,
            "version": FAST_DLLM_V2_STATE_VERSION,
            "block_size": self.block_size,
            "seq_len": self.max_seq_len,
            "mask_token_id": self.mask_token_id,
            "noise_schedule_epsilon": self.noise_schedule_epsilon,
            "paired_complementary_corruption": True,
            "causal_target_shift": 1,
            "batches_corrupted": self.batches_corrupted,
            "loss_tokens_seen": self.loss_tokens_seen,
            "valid_tokens_seen": self.valid_tokens_seen,
            "token_accounting": self.token_accounting.to_log_dict(),
        }

    @property
    def loss_tokens_seen(self) -> int:
        return int(self._loss_tokens_seen)


def fast_dllm_v2_schedule(
    *,
    sequence_length: int,
    block_size: int,
    mask_token_id: int | None,
):
    """Return the shared block-causal schedule used by Fast-dLLM v2."""

    schedule = standard_block_diffusion_schedule(
        sequence_length=sequence_length,
        block_size=block_size,
        mask_token_id=mask_token_id,
        region_prefix="fast_dllm_v2_block",
    )
    return replace(
        schedule,
        objective=replace(schedule.objective, name=FAST_DLLM_V2_OBJECTIVE),
    )


__all__ = [
    "FAST_DLLM_V2_OBJECTIVE",
    "FastDLLMv2ObjectiveRuntime",
    "fast_dllm_v2_schedule",
]
