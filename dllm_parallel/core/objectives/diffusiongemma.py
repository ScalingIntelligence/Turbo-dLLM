# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Native all-block DiffusionGemma SFT corruption and persistent state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from dllm_parallel.core.objectives.runtime import (
    LABEL_IGNORE_INDEX,
    TokenAccountingPolicy,
)


NATIVE_DIFFUSIONGEMMA_STATE_VERSION = 3
NATIVE_DIFFUSIONGEMMA_WORK_COUNT_NAMES = (
    "clean_encoder_rows",
    "detached_decoder_rows",
    "trained_decoder_rows",
    "self_conditioning_vocabulary_rows",
    "decoder_loss_vocabulary_rows",
    "encoder_ar_vocabulary_rows",
    "valid_blocks",
    "self_conditioned_examples",
)


@dataclass(frozen=True)
class DiffusionGemmaNativeBatch:
    """One all-block uniform-state corruption draw.

    ``labels`` and ``scored_mask`` cover every supervised token.  The
    ``replacement_mask`` is diagnostic only and never gates the loss.
    """

    clean_input_ids: torch.Tensor
    noisy_input_ids: torch.Tensor
    labels: torch.Tensor
    scored_mask: torch.Tensor
    encoder_valid_mask: torch.Tensor | None
    decoder_position_ids: torch.Tensor
    decoder_valid_mask: torch.Tensor | None
    decoder_block_ids: torch.Tensor
    response_start: int
    replacement_mask: torch.Tensor
    diffusion_times: torch.Tensor
    noise_levels: torch.Tensor
    loss_weights: torch.Tensor
    block_token_counts: torch.Tensor
    valid_block_mask: torch.Tensor
    self_conditioning_mask: torch.Tensor
    self_conditioning_execution_count: int
    active_tokens: torch.Tensor
    replaced_tokens: torch.Tensor
    valid_tokens: int
    loss_denominator: float | torch.Tensor
    performance_counts: torch.Tensor


class DiffusionGemmaNativeObjectiveRuntime:
    """Own native DiffusionGemma all-block corruption RNG and accounting."""

    def __init__(
        self,
        *,
        block_size: int,
        seq_len: int,
        vocab_size: int,
        device: torch.device,
        seed: int,
        token_accounting: TokenAccountingPolicy,
        sampling_epsilon_min: float = 1.0e-3,
        sampling_epsilon_max: float = 1.0,
        self_conditioning_probability: float = 0.5,
        antithetic_sampling: bool = True,
        minimum_decoder_blocks: int = 1,
    ) -> None:
        if int(seq_len) <= 0 or int(block_size) <= 0:
            raise ValueError("seq_len and block_size must be positive")
        if int(seq_len) % int(block_size):
            raise ValueError("seq_len must divide evenly by block_size")
        if int(vocab_size) <= 1:
            raise ValueError("vocab_size must be greater than one")
        if not (
            0.0 < float(sampling_epsilon_min) <= float(sampling_epsilon_max) <= 1.0
        ):
            raise ValueError("sampling epsilon bounds must satisfy 0 < min <= max <= 1")
        if not 0.0 <= float(self_conditioning_probability) <= 1.0:
            raise ValueError("self_conditioning_probability must be in [0, 1]")
        self.block_size = int(block_size)
        self.max_seq_len = int(seq_len)
        self.vocab_size = int(vocab_size)
        self.device = device
        self.sampling_epsilon_min = float(sampling_epsilon_min)
        self.sampling_epsilon_max = float(sampling_epsilon_max)
        self.self_conditioning_probability = float(self_conditioning_probability)
        self.antithetic_sampling = bool(antithetic_sampling)
        self.minimum_decoder_blocks = int(minimum_decoder_blocks)
        self.token_accounting = token_accounting
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(int(seed))
        self.batches_corrupted = 0
        self._scored_tokens_seen = torch.zeros((), dtype=torch.int64, device=device)
        self._replaced_tokens_seen = torch.zeros((), dtype=torch.int64, device=device)
        self.valid_tokens_seen = 0

    def corrupt(
        self,
        x0: torch.Tensor,
        *,
        supervision_mask: torch.Tensor | None = None,
        supervision_count: int | None = None,
        valid_mask: torch.Tensor | None = None,
        response_starts: torch.Tensor | None = None,
    ) -> DiffusionGemmaNativeBatch:
        if x0.ndim != 2:
            raise ValueError(f"expected rank-2 input_ids, got {tuple(x0.shape)}")
        batch_size, sequence_length = (int(x0.shape[0]), int(x0.shape[1]))
        if not 0 < sequence_length <= self.max_seq_len:
            raise ValueError(
                "input sequence length must be positive and no greater than the "
                f"configured maximum ({self.max_seq_len})"
            )
        if sequence_length % self.block_size:
            raise ValueError("input sequence length must divide evenly by block_size")
        positions = torch.arange(sequence_length, device=x0.device)
        zero_prefix_fast_path = (
            supervision_mask is None
            and supervision_count is None
            and valid_mask is None
            and response_starts is None
        )
        if zero_prefix_fast_path:
            valid = None
            encoder_valid_mask = None
            response_start = 0
            response_length = sequence_length
            decoder_length = sequence_length
            logical_decoder_length = decoder_length
            decoder_offsets = positions
            decoder_position_ids = positions
            decoder_block_ids = (positions // self.block_size).to(torch.int32)
            decoder_clean = x0
            scored_mask = torch.ones_like(x0, dtype=torch.bool)
            decoder_valid_mask = None
            valid_tokens = int(x0.numel())
        else:
            encoder_valid_mask: torch.Tensor | None = None
            if valid_mask is None:
                valid = torch.ones_like(x0, dtype=torch.bool)
                valid_lengths = torch.full(
                    (batch_size,),
                    sequence_length,
                    dtype=torch.long,
                    device=x0.device,
                )
            else:
                if valid_mask.shape != x0.shape:
                    raise ValueError("valid_mask must match input_ids")
                valid = valid_mask.to(device=x0.device, dtype=torch.bool)
                valid_lengths = valid.sum(dim=1, dtype=torch.long)
                if bool((valid_lengths <= 0).any()):
                    raise ValueError(
                        "each native DiffusionGemma row must contain valid tokens"
                    )
                expected_valid = positions.unsqueeze(0) < valid_lengths.unsqueeze(1)
                if not torch.equal(valid, expected_valid):
                    raise ValueError(
                        "valid_mask must be one contiguous prefix per example"
                    )
                if not bool(valid.all()):
                    encoder_valid_mask = valid

            if supervision_mask is None:
                if supervision_count is not None:
                    raise ValueError(
                        "supervision_count requires an explicit supervision_mask"
                    )
                scored_full = valid
            else:
                if supervision_mask.shape != x0.shape:
                    raise ValueError("supervision_mask must match input_ids")
                scored_full = supervision_mask.to(device=x0.device, dtype=torch.bool)
                if bool((scored_full & ~valid).any()):
                    raise ValueError("supervision_mask cannot score invalid padding")

            row_counts = scored_full.sum(dim=1, dtype=torch.long)
            if bool((row_counts <= 0).any()):
                raise ValueError("each training row must contain a supervised response")
            if response_starts is None:
                starts = scored_full.to(dtype=torch.int64).argmax(dim=1)
            else:
                if response_starts.shape != (batch_size,):
                    raise ValueError("response_starts must have shape [batch]")
                starts = response_starts.to(device=x0.device, dtype=torch.long)
            if bool((starts < 0).any()) or bool((starts >= valid_lengths).any()):
                raise ValueError("response_starts must lie inside the valid sequence")
            expected_scored = (positions.unsqueeze(0) >= starts.unsqueeze(1)) & valid
            if not torch.equal(scored_full, expected_scored):
                if response_starts is None:
                    raise ValueError("supervision_mask must be one contiguous suffix")
                raise ValueError(
                    "response_starts must identify the contiguous supervised suffix"
                )
            if not bool(starts.eq(starts[0]).all()):
                raise ValueError("native batches require one shared response start")
            if not bool(valid_lengths.eq(valid_lengths[0]).all()):
                raise ValueError(
                    "native batches require one shared valid sequence length"
                )

            observed = int(scored_full.sum())
            valid_tokens = (
                observed if supervision_count is None else int(supervision_count)
            )
            if valid_tokens != observed:
                raise ValueError(
                    "supervision_count must equal the explicit supervision_mask count"
                )

            response_start = int(starts[0].item())
            response_length = int(valid_lengths[0].item()) - response_start
            logical_decoder_length = (
                (response_length + self.block_size - 1) // self.block_size
            ) * self.block_size
            decoder_length = max(
                logical_decoder_length,
                self.minimum_decoder_blocks * self.block_size,
            )
            decoder_offsets = torch.arange(decoder_length, device=x0.device)
            decoder_valid_1d = decoder_offsets < response_length
            decoder_position_ids = (decoder_offsets + response_start).clamp(
                max=int(valid_lengths[0].item()) - 1
            )
            decoder_block_ids = (decoder_offsets // self.block_size).to(torch.int32)
            decoder_block_ids = torch.where(
                decoder_valid_1d,
                decoder_block_ids,
                torch.full_like(decoder_block_ids, -1),
            )
            decoder_clean = x0.index_select(1, decoder_position_ids)
            decoder_valid_mask = decoder_valid_1d.unsqueeze(0).expand(
                batch_size, -1
            )
            scored_mask = decoder_valid_mask

        num_blocks = decoder_length // self.block_size
        logical_num_blocks = logical_decoder_length // self.block_size
        logical_diffusion_times = self._sample_diffusion_times(
            batch_size, logical_num_blocks, x0.device
        )
        token_move_chance = logical_diffusion_times.repeat_interleave(
            self.block_size, dim=1
        )
        logical_clean = decoder_clean[:, :logical_decoder_length]
        logical_replacement_mask = (
            torch.rand(
                logical_clean.shape,
                dtype=torch.float32,
                device=x0.device,
                generator=self.generator,
            )
            < token_move_chance
        )
        logical_replacement_mask.logical_and_(
            scored_mask[:, :logical_decoder_length]
        )
        random_tokens = torch.randint(
            0,
            self.vocab_size,
            logical_clean.shape,
            dtype=logical_clean.dtype,
            device=x0.device,
            generator=self.generator,
        )
        logical_noisy = torch.where(
            logical_replacement_mask,
            random_tokens,
            logical_clean,
        )
        if decoder_length == logical_decoder_length:
            diffusion_times = logical_diffusion_times
            replacement_mask = logical_replacement_mask
            noisy = logical_noisy
        else:
            padding_blocks = num_blocks - logical_num_blocks
            diffusion_times = torch.cat(
                (
                    logical_diffusion_times,
                    torch.full(
                        (batch_size, padding_blocks),
                        self.sampling_epsilon_min,
                        dtype=torch.float32,
                        device=x0.device,
                    ),
                ),
                dim=1,
            )
            replacement_mask = torch.cat(
                (
                    logical_replacement_mask,
                    torch.zeros(
                        (batch_size, decoder_length - logical_decoder_length),
                        dtype=torch.bool,
                        device=x0.device,
                    ),
                ),
                dim=1,
            )
            noisy = torch.cat(
                (logical_noisy, decoder_clean[:, logical_decoder_length:]),
                dim=1,
            )
        labels = torch.where(
            scored_mask,
            decoder_clean,
            torch.full_like(decoder_clean, LABEL_IGNORE_INDEX),
        )
        block_token_counts = scored_mask.reshape(
            batch_size,
            num_blocks,
            self.block_size,
        ).sum(dim=-1, dtype=torch.int64)
        valid_block_mask = block_token_counts > 0
        self_conditioning_mask = (
            torch.rand(
                (batch_size,),
                dtype=torch.float32,
                device=x0.device,
                generator=self.generator,
            )
            < self.self_conditioning_probability
        )
        scored_tokens = scored_mask.sum(dtype=torch.int64)
        replaced_tokens = replacement_mask.sum(dtype=torch.int64)
        self_conditioned_examples = self_conditioning_mask.sum(dtype=torch.int64)
        sequence_rows = batch_size * sequence_length
        decoder_rows = batch_size * decoder_length
        selected_rows = self_conditioned_examples * decoder_length
        encoder_ar_rows = (
            torch.as_tensor(
                batch_size * max(0, sequence_length - 1),
                device=x0.device,
                dtype=torch.int64,
            )
            if valid is None
            else (valid[:, :-1] & valid[:, 1:]).sum(dtype=torch.int64)
        )
        performance_counts = torch.stack(
            (
                torch.as_tensor(sequence_rows, device=x0.device, dtype=torch.int64),
                selected_rows,
                torch.as_tensor(decoder_rows, device=x0.device, dtype=torch.int64),
                selected_rows,
                scored_tokens,
                encoder_ar_rows,
                valid_block_mask.sum(dtype=torch.int64),
                self_conditioned_examples,
            )
        )
        self.batches_corrupted += 1
        self._scored_tokens_seen.add_(scored_tokens)
        self._replaced_tokens_seen.add_(replaced_tokens)
        self.valid_tokens_seen += valid_tokens
        return DiffusionGemmaNativeBatch(
            clean_input_ids=x0,
            noisy_input_ids=noisy,
            labels=labels,
            scored_mask=scored_mask,
            encoder_valid_mask=encoder_valid_mask,
            decoder_position_ids=decoder_position_ids,
            decoder_valid_mask=decoder_valid_mask,
            decoder_block_ids=decoder_block_ids,
            response_start=response_start,
            replacement_mask=replacement_mask,
            diffusion_times=diffusion_times,
            noise_levels=-torch.log1p(-diffusion_times.clamp(max=1.0 - 1.0e-7)),
            loss_weights=torch.ones_like(diffusion_times),
            block_token_counts=block_token_counts,
            valid_block_mask=valid_block_mask,
            self_conditioning_mask=self_conditioning_mask,
            self_conditioning_execution_count=int(self_conditioned_examples),
            active_tokens=scored_tokens,
            replaced_tokens=replaced_tokens,
            valid_tokens=valid_tokens,
            loss_denominator=float(valid_tokens),
            performance_counts=performance_counts,
        )

    def _sample_diffusion_times(
        self,
        batch_size: int,
        num_blocks: int,
        device: torch.device,
    ) -> torch.Tensor:
        sample_count = int(batch_size) * int(num_blocks)
        if self.antithetic_sampling:
            base = torch.rand(
                (1,), dtype=torch.float32, device=device, generator=self.generator
            )
            offsets = torch.arange(
                sample_count, dtype=torch.float32, device=device
            ).reshape(batch_size, num_blocks)
            values = (base + offsets / float(sample_count)).remainder(1.0)
        else:
            values = torch.rand(
                (batch_size, num_blocks),
                dtype=torch.float32,
                device=device,
                generator=self.generator,
            )
        return (
            values * (self.sampling_epsilon_max - self.sampling_epsilon_min)
            + self.sampling_epsilon_min
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "diffusiongemma_native_sft",
            "version": NATIVE_DIFFUSIONGEMMA_STATE_VERSION,
            "block_size": self.block_size,
            "seq_len": self.max_seq_len,
            "vocab_size": self.vocab_size,
            "sampling_epsilon_min": self.sampling_epsilon_min,
            "sampling_epsilon_max": self.sampling_epsilon_max,
            "self_conditioning_probability": self.self_conditioning_probability,
            "antithetic_sampling": self.antithetic_sampling,
            "minimum_decoder_blocks": self.minimum_decoder_blocks,
            "generator_state": self.generator.get_state(),
            "batches_corrupted": self.batches_corrupted,
            "scored_tokens_seen": self._scored_tokens_seen.detach().cpu(),
            "replaced_tokens_seen": self._replaced_tokens_seen.detach().cpu(),
            "valid_tokens_seen": self.valid_tokens_seen,
            "token_accounting": self.token_accounting.to_log_dict(),
        }

    def load_state_dict(self, state: dict[str, Any] | None) -> None:
        if not state:
            return
        if state.get("kind") != "diffusiongemma_native_sft":
            raise RuntimeError(
                "checkpoint does not contain native DiffusionGemma SFT state"
            )
        if state.get("version") not in {
            1,
            2,
            NATIVE_DIFFUSIONGEMMA_STATE_VERSION,
        }:
            raise RuntimeError(
                "checkpoint native DiffusionGemma objective version is incompatible"
            )
        for key, expected in (
            ("block_size", self.block_size),
            ("seq_len", self.max_seq_len),
            ("vocab_size", self.vocab_size),
            ("sampling_epsilon_min", self.sampling_epsilon_min),
            ("sampling_epsilon_max", self.sampling_epsilon_max),
            ("self_conditioning_probability", self.self_conditioning_probability),
            ("antithetic_sampling", self.antithetic_sampling),
            ("minimum_decoder_blocks", self.minimum_decoder_blocks),
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
        self._scored_tokens_seen.copy_(
            torch.as_tensor(state.get("scored_tokens_seen", 0), device=self.device)
        )
        self._replaced_tokens_seen.copy_(
            torch.as_tensor(state.get("replaced_tokens_seen", 0), device=self.device)
        )
        self.valid_tokens_seen = int(state.get("valid_tokens_seen", 0))

    @property
    def loss_tokens_seen(self) -> int:
        return int(self._scored_tokens_seen.item())

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "kind": "diffusiongemma_native_sft",
            "block_size": self.block_size,
            "seq_len": self.max_seq_len,
            "vocab_size": self.vocab_size,
            "sampling_epsilon_min": self.sampling_epsilon_min,
            "sampling_epsilon_max": self.sampling_epsilon_max,
            "self_conditioning_probability": self.self_conditioning_probability,
            "antithetic_sampling": self.antithetic_sampling,
            "minimum_decoder_blocks": self.minimum_decoder_blocks,
            "semantic_layout_version": NATIVE_DIFFUSIONGEMMA_STATE_VERSION,
            "batches_corrupted": self.batches_corrupted,
            "scored_tokens_seen": int(self._scored_tokens_seen.item()),
            "replaced_tokens_seen": int(self._replaced_tokens_seen.item()),
            "valid_tokens_seen": self.valid_tokens_seen,
            "token_accounting": self.token_accounting.to_log_dict(),
        }


__all__ = [
    "DiffusionGemmaNativeBatch",
    "DiffusionGemmaNativeObjectiveRuntime",
    "NATIVE_DIFFUSIONGEMMA_STATE_VERSION",
    "NATIVE_DIFFUSIONGEMMA_WORK_COUNT_NAMES",
]
