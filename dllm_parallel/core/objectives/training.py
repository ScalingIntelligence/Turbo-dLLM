# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Objective-bound microbatch execution for the canonical trainer."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Protocol

import torch
import torch.distributed as dist

from dllm_parallel.core.objectives.runtime import (
    CorruptedBatch,
    StandardBlockDiffusionObjectiveRuntime,
    TokenAccountingPolicy,
)
from dllm_parallel.core.objectives.fast_dllm_v2 import (
    FastDLLMv2ObjectiveRuntime,
)
from dllm_parallel.core.objectives.diffusiongemma import (
    DiffusionGemmaNativeBatch,
    DiffusionGemmaNativeObjectiveRuntime,
    NATIVE_DIFFUSIONGEMMA_WORK_COUNT_NAMES,
)
from dllm_parallel.core.data import DataBatch, DataRuntime


def _broadcast_model_parallel_batch(*tensors: torch.Tensor, runtime: Any) -> None:
    if (
        runtime is None
        or not getattr(runtime, "enabled", False)
        or getattr(runtime, "model_input_group", None) is None
        or len(getattr(runtime, "model_input_group_ranks", ()) or ()) <= 1
    ):
        return
    for tensor in tensors:
        dist.broadcast(
            tensor,
            src=int(runtime.model_input_src_rank),
            group=runtime.model_input_group,
        )


class TrainingTask(Protocol):
    objective_runtime: Any

    def accumulation_loss_denominator(
        self,
        batches: list[DataBatch],
    ) -> float | torch.Tensor | None: ...

    def prepare(
        self,
        batch: DataBatch,
        *,
        loss_denominator: float | torch.Tensor | None = None,
    ) -> Any: ...

    def forward(self, model: Any, prepared: Any) -> Any: ...

    def loss(self, model: Any, prepared: Any, output: Any) -> torch.Tensor: ...

    def output_shape(self, output: Any) -> tuple[int, ...]: ...

    def skip(self, data_runtime: DataRuntime, count: int) -> None: ...


@dataclass(frozen=True)
class StandardBlockDiffusionPreparedBatch:
    corrupted: CorruptedBatch


class StandardBlockDiffusionTrainingTask:
    def __init__(
        self,
        *,
        mask_token_id: int,
        block_size: int,
        seq_len: int,
        vocab_size: int,
        device: torch.device,
        seed: int,
        runtime: Any,
        token_accounting: TokenAccountingPolicy,
        noise_schedule: str,
        loss_weighting: str,
        noise_schedule_epsilon: float,
        sampling_epsilon_min: float,
        sampling_epsilon_max: float,
        antithetic_sampling: bool,
        bp_loss_scale: float | None,
    ) -> None:
        self.mask_token_id = int(mask_token_id)
        self.vocab_size = int(vocab_size)
        self.max_seq_len = int(seq_len)
        self.runtime = runtime
        self.bp_loss_scale = bp_loss_scale
        self.exclude_mask_token = True
        self.objective_runtime = StandardBlockDiffusionObjectiveRuntime(
            mask_token_id=int(mask_token_id),
            block_size=int(block_size),
            seq_len=int(seq_len),
            device=device,
            seed=int(seed),
            token_accounting=token_accounting,
            noise_schedule=str(noise_schedule),
            loss_weighting=str(loss_weighting),
            noise_schedule_epsilon=float(noise_schedule_epsilon),
            sampling_epsilon_min=float(sampling_epsilon_min),
            sampling_epsilon_max=float(sampling_epsilon_max),
            antithetic_sampling=bool(antithetic_sampling),
        )

    def accumulation_loss_denominator(
        self,
        batches: list[DataBatch],
    ) -> float | torch.Tensor | None:
        return _accumulation_loss_denominator(
            batches,
            runtime=self.runtime,
            shifted_targets=False,
        )

    def prepare(
        self,
        batch: DataBatch,
        *,
        loss_denominator: float | torch.Tensor | None = None,
    ) -> StandardBlockDiffusionPreparedBatch:
        corrupted = self.objective_runtime.corrupt(
            batch.input_ids,
            supervision_mask=batch.loss_mask,
            supervision_count=batch.supervised_token_count,
        )
        if batch.loss_mask is not None:
            if loss_denominator is None:
                loss_denominator = self.accumulation_loss_denominator([batch])
            corrupted = replace(corrupted, loss_denominator=loss_denominator)
        _broadcast_model_parallel_batch(
            corrupted.clean_input_ids,
            corrupted.noisy_input_ids,
            corrupted.labels,
            corrupted.diffusion_times,
            corrupted.noise_levels,
            corrupted.loss_weights,
            runtime=self.runtime,
        )
        return StandardBlockDiffusionPreparedBatch(corrupted=corrupted)

    def forward(
        self,
        model: Any,
        prepared: StandardBlockDiffusionPreparedBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = prepared.corrupted
        return model(
            noisy_input_ids=batch.noisy_input_ids,
            clean_input_ids=batch.clean_input_ids,
            diffusion_times=batch.diffusion_times,
            noise_levels=batch.noise_levels,
        )

    def loss(
        self,
        model: Any,
        prepared: StandardBlockDiffusionPreparedBatch,
        output: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden, active_positions = output
        batch = prepared.corrupted
        loss_fn = getattr(
            getattr(model, "module", model), "distributed_block_diffusion_loss", None
        )
        if not callable(loss_fn):
            raise RuntimeError(
                "distributed DLLM models must implement distributed_block_diffusion_loss"
            )
        loss_options = {
            "vocab_size": self.vocab_size,
            "mask_token_id": self.mask_token_id,
            "seq_len": int(batch.labels.shape[1]),
            "valid_token_count": batch.loss_denominator,
            "bp_loss_scale": self.bp_loss_scale,
        }
        if not self.exclude_mask_token:
            loss_options["exclude_mask_token"] = False
        return loss_fn(
            hidden,
            batch.labels,
            active_positions,
            batch.loss_weights,
            **loss_options,
        )

    def output_shape(
        self, output: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[int, ...]:
        return tuple(output[0].shape)

    def skip(self, data_runtime: DataRuntime, count: int) -> None:
        for _ in range(max(0, int(count))):
            batch = data_runtime.next_batch()
            self.objective_runtime.corrupt(
                batch.input_ids,
                supervision_mask=batch.loss_mask,
                supervision_count=batch.supervised_token_count,
            )


class FastDLLMv2TrainingTask(StandardBlockDiffusionTrainingTask):
    """Canonical trainer task for Fast-dLLM v2 converted causal LMs."""

    def __init__(
        self,
        *,
        mask_token_id: int,
        block_size: int,
        seq_len: int,
        vocab_size: int,
        device: torch.device,
        seed: int,
        runtime: Any,
        token_accounting: TokenAccountingPolicy,
        noise_schedule_epsilon: float,
        bp_loss_scale: float | None,
    ) -> None:
        self.mask_token_id = int(mask_token_id)
        self.vocab_size = int(vocab_size)
        self.max_seq_len = int(seq_len)
        self.runtime = runtime
        self.bp_loss_scale = bp_loss_scale
        self.exclude_mask_token = False
        self.objective_runtime = FastDLLMv2ObjectiveRuntime(
            mask_token_id=int(mask_token_id),
            block_size=int(block_size),
            seq_len=int(seq_len),
            device=device,
            seed=int(seed),
            token_accounting=token_accounting,
            noise_schedule_epsilon=float(noise_schedule_epsilon),
        )

    def accumulation_loss_denominator(
        self,
        batches: list[DataBatch],
    ) -> float | torch.Tensor | None:
        return _accumulation_loss_denominator(
            batches,
            runtime=self.runtime,
            shifted_targets=True,
        )


@dataclass(frozen=True)
class DiffusionGemmaNativePreparedBatch:
    corrupted: DiffusionGemmaNativeBatch

    @property
    def performance_counts(self) -> torch.Tensor:
        return self.corrupted.performance_counts


class DiffusionGemmaNativeTrainingTask:
    """All-block uniform-state SFT task for the DiffusionGemma backbone."""

    def __init__(
        self,
        *,
        block_size: int,
        seq_len: int,
        vocab_size: int,
        device: torch.device,
        seed: int,
        runtime: Any,
        token_accounting: TokenAccountingPolicy,
        sampling_epsilon_min: float,
        sampling_epsilon_max: float,
        self_conditioning_probability: float,
        antithetic_sampling: bool,
        encoder_loss_weight: float,
        self_conditioning_row_chunk_size: int,
        self_conditioning_vocab_chunk_size: int,
        self_conditioning_execute_all: bool,
    ) -> None:
        self.runtime = runtime
        self.vocab_size = int(vocab_size)
        self.block_size = int(block_size)
        self.encoder_loss_weight = float(encoder_loss_weight)
        self.self_conditioning_row_chunk_size = int(self_conditioning_row_chunk_size)
        self.self_conditioning_vocab_chunk_size = int(
            self_conditioning_vocab_chunk_size
        )
        self.self_conditioning_execute_all = bool(self_conditioning_execute_all)
        self.performance_count_names = NATIVE_DIFFUSIONGEMMA_WORK_COUNT_NAMES
        self.objective_runtime = DiffusionGemmaNativeObjectiveRuntime(
            block_size=int(block_size),
            seq_len=int(seq_len),
            vocab_size=int(vocab_size),
            device=device,
            seed=int(seed),
            token_accounting=token_accounting,
            sampling_epsilon_min=float(sampling_epsilon_min),
            sampling_epsilon_max=float(sampling_epsilon_max),
            self_conditioning_probability=float(self_conditioning_probability),
            antithetic_sampling=bool(antithetic_sampling),
            minimum_decoder_blocks=int(
                getattr(runtime, "block_parallel_size", 1) or 1
            ),
        )

    def accumulation_loss_denominator(
        self,
        batches: list[DataBatch],
    ) -> torch.Tensor:
        return _native_accumulation_loss_denominators(
            batches,
            runtime=self.runtime,
        )

    def prepare(
        self,
        batch: DataBatch,
        *,
        loss_denominator: float | torch.Tensor | None = None,
    ) -> DiffusionGemmaNativePreparedBatch:
        corrupted = self.objective_runtime.corrupt(
            batch.input_ids,
            supervision_mask=batch.loss_mask,
            supervision_count=batch.supervised_token_count,
            valid_mask=batch.valid_mask,
            response_starts=batch.response_starts,
        )
        if loss_denominator is None:
            loss_denominator = self.accumulation_loss_denominator([batch])
        denominators = torch.as_tensor(
            loss_denominator,
            device=batch.input_ids.device,
            dtype=torch.float64,
        ).reshape(-1)
        if int(denominators.numel()) != 2:
            raise ValueError(
                "native DiffusionGemma loss denominator must contain "
                "[valid_examples, encoder_ar_pairs]"
            )
        corrupted = replace(corrupted, loss_denominator=denominators)
        broadcast_tensors = [
            corrupted.clean_input_ids,
            corrupted.noisy_input_ids,
            corrupted.labels,
            corrupted.scored_mask,
            corrupted.replacement_mask,
            corrupted.diffusion_times,
            corrupted.noise_levels,
            corrupted.loss_weights,
            corrupted.block_token_counts,
            corrupted.valid_block_mask,
            corrupted.self_conditioning_mask,
        ]
        if corrupted.encoder_valid_mask is not None:
            broadcast_tensors.append(corrupted.encoder_valid_mask)
        if corrupted.decoder_valid_mask is not None:
            broadcast_tensors.append(corrupted.decoder_valid_mask)
        _broadcast_model_parallel_batch(*broadcast_tensors, runtime=self.runtime)
        execution_count = (
            int(corrupted.clean_input_ids.shape[0])
            if self.self_conditioning_execute_all
            else _synchronize_self_conditioning_execution_count(
                corrupted.self_conditioning_mask,
                runtime=self.runtime,
            )
        )
        performance_counts = corrupted.performance_counts.clone()
        performance_counts[1] = execution_count * int(
            corrupted.noisy_input_ids.shape[1]
        )
        corrupted = replace(
            corrupted,
            self_conditioning_execution_count=execution_count,
            performance_counts=performance_counts,
        )
        return DiffusionGemmaNativePreparedBatch(corrupted=corrupted)

    def forward(self, model: Any, prepared: DiffusionGemmaNativePreparedBatch) -> Any:
        batch = prepared.corrupted
        denominators = torch.as_tensor(
            batch.loss_denominator,
            device=batch.clean_input_ids.device,
            dtype=torch.float64,
        ).reshape(-1)
        return model(
            objective_mode="diffusiongemma_native_sft",
            noisy_input_ids=batch.noisy_input_ids,
            clean_input_ids=batch.clean_input_ids,
            labels=batch.labels,
            scored_mask=batch.scored_mask,
            encoder_valid_mask=batch.encoder_valid_mask,
            decoder_position_ids=batch.decoder_position_ids,
            decoder_valid_mask=batch.decoder_valid_mask,
            decoder_block_ids=batch.decoder_block_ids,
            response_start=batch.response_start,
            block_token_counts=batch.block_token_counts,
            valid_block_mask=batch.valid_block_mask,
            self_conditioning_mask=batch.self_conditioning_mask,
            self_conditioning_execution_count=(batch.self_conditioning_execution_count),
            self_conditioning_execute_all=self.self_conditioning_execute_all,
            encoder_loss_weight=self.encoder_loss_weight,
            decoder_loss_denominator=denominators[0],
            encoder_loss_denominator=denominators[1],
            self_conditioning_row_chunk_size=self.self_conditioning_row_chunk_size,
            self_conditioning_vocab_chunk_size=self.self_conditioning_vocab_chunk_size,
        )

    def loss(
        self,
        model: Any,
        prepared: DiffusionGemmaNativePreparedBatch,
        output: Any,
    ) -> torch.Tensor:
        del model, prepared
        loss = getattr(output, "loss", None)
        if not isinstance(loss, torch.Tensor):
            raise RuntimeError("DiffusionGemma native output must expose tensor loss")
        return loss

    def output_shape(self, output: Any) -> tuple[int, ...]:
        decoder_hidden = getattr(output, "decoder_hidden", None)
        return (
            tuple(decoder_hidden.shape)
            if isinstance(decoder_hidden, torch.Tensor)
            else ()
        )

    def skip(self, data_runtime: DataRuntime, count: int) -> None:
        for _ in range(max(0, int(count))):
            batch = data_runtime.next_batch()
            self.objective_runtime.corrupt(
                batch.input_ids,
                supervision_mask=batch.loss_mask,
                supervision_count=batch.supervised_token_count,
                valid_mask=batch.valid_mask,
                response_starts=batch.response_starts,
            )


def _synchronize_self_conditioning_execution_count(
    self_conditioning_mask: torch.Tensor,
    *,
    runtime: Any,
) -> int:
    """Choose an EP-consistent detached-pass row count without changing masks."""

    local_count = self_conditioning_mask.sum(dtype=torch.int64)
    expert_parallel_size = int(getattr(runtime, "expert_parallel_size", 1) or 1)
    if expert_parallel_size > 1:
        expert_parallel_group = getattr(runtime, "expert_parallel_group", None)
        if expert_parallel_group is None:
            raise RuntimeError(
                "native self-conditioning requires an expert-parallel group"
            )
        dist.all_reduce(
            local_count,
            op=dist.ReduceOp.MAX,
            group=expert_parallel_group,
        )
    return int(local_count.item())


def _native_accumulation_loss_denominators(
    batches: list[DataBatch],
    *,
    runtime: Any,
) -> torch.Tensor:
    """Return per-microbatch global means for native decoder and AR losses."""

    if not batches:
        raise RuntimeError("an optimizer step must contain training batches")
    valid_examples = 0
    encoder_ar_pairs = 0
    for batch in batches:
        if batch.loss_mask is None:
            valid_examples += int(batch.input_ids.shape[0])
        else:
            mask = batch.loss_mask.to(dtype=torch.bool)
            valid_examples += int(mask.any(dim=1).sum())
        if batch.valid_mask is None:
            encoder_ar_pairs += int(batch.input_ids.shape[0]) * max(
                0,
                int(batch.input_ids.shape[1]) - 1,
            )
        else:
            valid = batch.valid_mask.to(dtype=torch.bool)
            encoder_ar_pairs += int((valid[:, :-1] & valid[:, 1:]).sum())
    if valid_examples <= 0:
        raise RuntimeError("an optimizer step must contain supervised examples")

    counts = torch.tensor(
        (valid_examples, encoder_ar_pairs),
        dtype=torch.int64,
        device=batches[0].input_ids.device,
    )
    data_parallel_size = int(getattr(runtime, "data_parallel_size", 1) or 1)
    expert_parallel_size = int(getattr(runtime, "expert_parallel_size", 1) or 1)
    if data_parallel_size > 1:
        data_parallel_group = getattr(runtime, "data_parallel_group", None)
        if data_parallel_group is None:
            raise RuntimeError(
                "distributed native SFT normalization requires a data-parallel group"
            )
        dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=data_parallel_group)
    if expert_parallel_size > 1:
        expert_parallel_group = getattr(runtime, "expert_parallel_group", None)
        if expert_parallel_group is None:
            raise RuntimeError(
                "distributed native SFT normalization requires an expert-parallel group"
            )
        dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=expert_parallel_group)
    # An optimizer step can legitimately have no shifted supervised pair (for
    # example, only position zero is scored). Its AR numerator is also zero.
    counts[1].clamp_min_(1)
    return counts.to(dtype=torch.float64) / float(
        data_parallel_size * expert_parallel_size * len(batches)
    )


def _accumulation_loss_denominator(
    batches: list[DataBatch],
    *,
    runtime: Any,
    shifted_targets: bool,
) -> float | torch.Tensor | None:
    masks = [batch.loss_mask for batch in batches]
    if all(mask is None for mask in masks):
        return None
    if any(mask is None for mask in masks):
        raise RuntimeError(
            "an optimizer step cannot mix supervised and unsupervised batches"
        )
    local_tokens = 0
    for batch in batches:
        if batch.loss_mask is None:
            continue
        if shifted_targets:
            local_tokens += int(batch.loss_mask[:, 1:].sum())
        else:
            local_tokens += (
                int(batch.supervised_token_count)
                if batch.supervised_token_count is not None
                else int(batch.loss_mask.sum())
            )
    if local_tokens <= 0:
        raise RuntimeError("an optimizer step must contain supervised tokens")
    data_parallel_size = int(getattr(runtime, "data_parallel_size", 1) or 1)
    expert_parallel_size = int(getattr(runtime, "expert_parallel_size", 1) or 1)
    if data_parallel_size <= 1 and expert_parallel_size <= 1:
        return float(local_tokens) / float(len(batches))
    device = batches[0].input_ids.device
    count = torch.tensor(local_tokens, dtype=torch.int64, device=device)
    if data_parallel_size > 1:
        data_parallel_group = getattr(runtime, "data_parallel_group", None)
        if data_parallel_group is None:
            raise RuntimeError(
                "distributed SFT token normalization requires a data-parallel group"
            )
        dist.all_reduce(count, op=dist.ReduceOp.SUM, group=data_parallel_group)
    if expert_parallel_size > 1:
        expert_parallel_group = getattr(runtime, "expert_parallel_group", None)
        if expert_parallel_group is None:
            raise RuntimeError(
                "distributed SFT token normalization requires an expert-parallel group"
            )
        dist.all_reduce(count, op=dist.ReduceOp.SUM, group=expert_parallel_group)
    return count.to(dtype=torch.float64) / float(
        data_parallel_size * expert_parallel_size * len(batches)
    )


__all__ = [
    "build_diffusiongemma_native_training_task",
    "build_fast_dllm_v2_training_task",
    "build_standard_training_task",
    "FastDLLMv2TrainingTask",
    "StandardBlockDiffusionTrainingTask",
    "TrainingTask",
]


def build_diffusiongemma_native_training_task(
    *,
    spec: Any,
    runtime: Any,
    device: torch.device,
    seed: int,
    data_parallel_seed: int,
    mask_token_id: int,
    vocab_size: int,
    token_accounting: TokenAccountingPolicy,
    block_size: int,
    model_metadata: Any = None,
) -> DiffusionGemmaNativeTrainingTask:
    del seed, mask_token_id, model_metadata
    objective = spec.objective
    return DiffusionGemmaNativeTrainingTask(
        block_size=int(block_size),
        seq_len=int(spec.model.seq_len),
        vocab_size=int(vocab_size),
        device=device,
        seed=int(data_parallel_seed),
        runtime=runtime,
        token_accounting=token_accounting,
        sampling_epsilon_min=float(objective.sampling_epsilon_min),
        sampling_epsilon_max=float(objective.sampling_epsilon_max),
        self_conditioning_probability=float(objective.self_conditioning_probability),
        antithetic_sampling=bool(objective.antithetic_sampling),
        encoder_loss_weight=float(objective.encoder_loss_weight),
        self_conditioning_row_chunk_size=int(
            objective.self_conditioning_row_chunk_size
        ),
        self_conditioning_vocab_chunk_size=int(
            objective.self_conditioning_vocab_chunk_size
        ),
        self_conditioning_execute_all=float(spec.adapter.dropout) > 0.0,
    )


def build_standard_training_task(
    *,
    spec: Any,
    runtime: Any,
    device: torch.device,
    seed: int,
    data_parallel_seed: int,
    mask_token_id: int,
    vocab_size: int,
    token_accounting: TokenAccountingPolicy,
    block_size: int,
    model_metadata: Any = None,
) -> StandardBlockDiffusionTrainingTask:
    del seed, model_metadata
    objective = spec.objective
    return StandardBlockDiffusionTrainingTask(
        mask_token_id=int(mask_token_id),
        block_size=int(block_size),
        seq_len=int(spec.model.seq_len),
        vocab_size=int(vocab_size),
        device=device,
        seed=int(data_parallel_seed),
        runtime=runtime,
        token_accounting=token_accounting,
        noise_schedule=str(objective.noise_schedule),
        loss_weighting=str(objective.loss_weighting),
        noise_schedule_epsilon=float(objective.noise_schedule_epsilon),
        sampling_epsilon_min=float(objective.sampling_epsilon_min),
        sampling_epsilon_max=float(objective.sampling_epsilon_max),
        antithetic_sampling=bool(objective.antithetic_sampling),
        bp_loss_scale=objective.bp_loss_scale,
    )


def build_fast_dllm_v2_training_task(
    *,
    spec: Any,
    runtime: Any,
    device: torch.device,
    seed: int,
    data_parallel_seed: int,
    mask_token_id: int,
    vocab_size: int,
    token_accounting: TokenAccountingPolicy,
    block_size: int,
    model_metadata: Any = None,
) -> FastDLLMv2TrainingTask:
    del seed, model_metadata
    return FastDLLMv2TrainingTask(
        mask_token_id=int(mask_token_id),
        block_size=int(block_size),
        seq_len=int(spec.model.seq_len),
        vocab_size=int(vocab_size),
        device=device,
        seed=int(data_parallel_seed),
        runtime=runtime,
        token_accounting=token_accounting,
        noise_schedule_epsilon=float(spec.objective.noise_schedule_epsilon),
        bp_loss_scale=spec.objective.bp_loss_scale,
    )
