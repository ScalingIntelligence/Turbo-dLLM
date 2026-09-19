# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Restartable data runtimes shared by production training executors."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from dllm_parallel.core.parallel.runtime import data_parallel_coordinates

_SMOKE_TEXT = (
    "Block diffusion language models learn to reconstruct masked spans "
    "from clean context while preserving the denoising objective. "
)

INDEXED_SUPERVISED_FORMAT = "dllm_parallel.indexed_supervised_tokens"
INDEXED_SUPERVISED_VERSION = 1
PACKED_TOKEN_FORMAT = "dllm_parallel.packed_tokens"
PACKED_TOKEN_VERSION = 1
INDEXED_SUPERVISED_COLUMNS = (
    "token_offset",
    "sequence_length",
    "supervision_start",
    "sample_id",
)


@dataclass(frozen=True)
class DataBatch:
    input_ids: torch.Tensor
    loss_mask: torch.Tensor | None = None
    sample_ids: torch.Tensor | None = None
    supervised_token_count: int | None = None
    valid_mask: torch.Tensor | None = None
    response_starts: torch.Tensor | None = None


class DataRuntime(ABC):
    """Restartable source of rank-local training batches."""

    @abstractmethod
    def next_batch(self) -> DataBatch:
        """Return the next rank-local batch."""

    def next_batches(self, count: int) -> list[DataBatch]:
        """Return all microbatches for one optimizer step."""
        if int(count) <= 0:
            raise ValueError("batch count must be positive")
        return [self.next_batch() for _ in range(int(count))]

    def next_optimizer_step_batches(self, count: int) -> list[DataBatch]:
        """Return the complete, ordered microbatch set for one optimizer update.

        Most sources use the statically configured gradient-accumulation count.
        Group-aware supervised sources may override this to return a variable
        number of microbatches, such as every assistant turn from one trajectory.
        """

        return self.next_batches(count)

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Return restart state for the next unread batch."""

    @abstractmethod
    def load_state_dict(self, state: dict[str, Any] | None) -> None:
        """Restore restart state produced by :meth:`state_dict`."""

    @abstractmethod
    def to_log_dict(self) -> dict[str, Any]:
        """Return stable, JSON-compatible source metadata."""


def build_standard_data_runtime(
    *,
    spec: Any,
    tokenizer: Any | None,
    vocab_size: int,
    mask_token_id: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    token_pool: torch.Tensor | None,
    runtime: Any | None = None,
    rank: int = 0,
    world_size: int = 1,
) -> DataRuntime:
    """Build the standard token data source for a resolved training spec."""

    del dtype
    data_parallel_rank, data_parallel_size = data_parallel_coordinates(
        runtime,
        rank=rank,
        world_size=world_size,
        distributed_data_parallel=True,
    )
    if spec.data.input_mode == "random":
        return RandomTokenDataRuntime(
            batch_size=int(spec.training.batch_size),
            seq_len=int(spec.model.seq_len),
            vocab_size=int(vocab_size),
            sample_vocab_size=spec.data.vocab_sample_size,
            mask_token_id=int(mask_token_id),
            device=device,
            seed=int(seed),
            token_pool=token_pool,
        )
    if spec.data.input_mode == "text":
        if tokenizer is None:
            raise RuntimeError("text input mode requires tokenizer")
        return PackedTokenDataRuntime.from_repeated_text(
            tokenizer=tokenizer,
            batch_size=int(spec.training.batch_size),
            seq_len=int(spec.model.seq_len),
            device=device,
            data_parallel_rank=data_parallel_rank,
            data_parallel_size=data_parallel_size,
        )
    if spec.data.input_mode == "dataset":
        if spec.data.dataset_path is None:
            raise ValueError("dataset input mode requires dataset_path")
        return PackedTokenDataRuntime.from_dataset_path(
            dataset_path=str(spec.data.dataset_path),
            tokenizer=tokenizer,
            mask_token_id=int(mask_token_id),
            batch_size=int(spec.training.batch_size),
            seq_len=int(spec.model.seq_len),
            device=device,
            data_parallel_rank=data_parallel_rank,
            data_parallel_size=data_parallel_size,
            blend_seed=int(spec.data.blend_seed),
            shuffle=bool(spec.data.shuffle),
            optimizer_step_unit=str(spec.data.optimizer_step_unit),
            minimum_sequence_length=spec.data.minimum_sequence_length,
            block_size=int(spec.objective.block_size or 1),
            context_parallel_size=int(spec.topology.context_parallel_size),
            block_parallel_size=int(spec.topology.block_parallel_size),
            group_by_supervision_start=(
                str(spec.objective.name) == "diffusiongemma_native_sft"
            ),
        )
    raise ValueError(f"unsupported input_mode: {spec.data.input_mode}")


class RandomTokenDataRuntime(DataRuntime):
    """Synthetic token runtime for smoke/profile recipes only."""

    def __init__(
        self,
        *,
        batch_size: int,
        seq_len: int,
        vocab_size: int,
        sample_vocab_size: int | None,
        mask_token_id: int,
        device: torch.device,
        seed: int,
        token_pool: torch.Tensor | None = None,
    ) -> None:
        self.batch_size = int(batch_size)
        self.seq_len = int(seq_len)
        self.vocab_size = int(vocab_size)
        self.sample_vocab_size = (
            None if sample_vocab_size is None else int(sample_vocab_size)
        )
        self.mask_token_id = int(mask_token_id)
        self.device = device
        self.token_pool = token_pool
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(int(seed))
        self.samples_consumed = 0

    def next_batch(self) -> DataBatch:
        if self.token_pool is not None:
            if self.token_pool.numel() <= 0:
                raise ValueError("synthetic random token pool is empty")
            offsets = torch.randint(
                low=0,
                high=int(self.token_pool.numel()),
                size=(self.batch_size, self.seq_len),
                device=self.device,
                dtype=torch.long,
                generator=self.generator,
            )
            ids = self.token_pool.index_select(0, offsets.reshape(-1)).reshape(
                self.batch_size,
                self.seq_len,
            )
        else:
            high = (
                self.vocab_size
                if self.sample_vocab_size is None
                else min(self.sample_vocab_size, self.vocab_size)
            )
            if high <= 1:
                raise ValueError("sample vocabulary size must be greater than one")
            ids = torch.randint(
                low=0,
                high=high,
                size=(self.batch_size, self.seq_len),
                device=self.device,
                dtype=torch.long,
                generator=self.generator,
            )
            ids = torch.where(
                ids == self.mask_token_id,
                (ids + 1) % self.vocab_size,
                ids,
            )
        self.samples_consumed += self.batch_size
        return DataBatch(input_ids=ids.contiguous())

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "random",
            "samples_consumed": int(self.samples_consumed),
            "generator_state": self.generator.get_state(),
        }

    def load_state_dict(self, state: dict[str, Any] | None) -> None:
        if not state:
            return
        generator_state = state.get("generator_state")
        if generator_state is not None:
            self.generator.set_state(generator_state.cpu())
        self.samples_consumed = int(state.get("samples_consumed", 0))

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "kind": "random",
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "sample_vocab_size": self.sample_vocab_size,
            "samples_consumed": int(self.samples_consumed),
        }


class PackedTokenDataRuntime(DataRuntime):
    """Restartable packed token stream for long-context training."""

    def __init__(
        self,
        *,
        token_stream: torch.Tensor,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        source: str,
        kind: str,
        dataset_fingerprint: str | None = None,
        data_parallel_rank: int = 0,
        data_parallel_size: int = 1,
    ) -> None:
        if token_stream.ndim != 1:
            token_stream = token_stream.reshape(-1)
        if token_stream.numel() <= 0:
            raise ValueError("packed token stream is empty")
        if token_stream.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise TypeError("packed token streams must use an integer dtype")
        self.tokens = token_stream.detach().to(device="cpu").contiguous()
        self.batch_size = int(batch_size)
        self.seq_len = int(seq_len)
        self.device = device
        self.source = source
        self.kind = kind
        self.dataset_fingerprint = dataset_fingerprint
        self.data_parallel_rank = int(data_parallel_rank)
        self.data_parallel_size = int(data_parallel_size)
        if self.data_parallel_rank < 0:
            raise ValueError("data_parallel_rank must be non-negative")
        if self.data_parallel_size <= 0:
            raise ValueError("data_parallel_size must be positive")
        if self.data_parallel_rank >= self.data_parallel_size:
            raise ValueError("data_parallel_rank must be less than data_parallel_size")
        self.cursor = 0
        self.samples_consumed = 0
        self._transfer_stream: torch.cuda.Stream | None = None
        self._host_buffers: list[torch.Tensor] = []
        self._device_buffers: list[torch.Tensor] = []
        self._ready_events: list[torch.cuda.Event] = []
        self._consumed_events: list[torch.cuda.Event] = []
        self._slot_initialized: list[bool] = []
        self._next_slot = 0
        self._returned_slot: int | None = None
        self._prefetch_initialized = False
        if self.device.type == "cuda":
            needed = self.batch_size * self.seq_len
            self._transfer_stream = torch.cuda.Stream(device=self.device)
            self._host_buffers = [
                torch.empty(needed, dtype=torch.long, pin_memory=True) for _ in range(2)
            ]
            self._device_buffers = [
                torch.empty(needed, dtype=torch.long, device=self.device)
                for _ in range(2)
            ]
            self._ready_events = [torch.cuda.Event() for _ in range(2)]
            self._consumed_events = [torch.cuda.Event() for _ in range(2)]
            self._slot_initialized = [False, False]

    @classmethod
    def from_dataset_path(
        cls,
        *,
        dataset_path: str,
        tokenizer: Any | None,
        mask_token_id: int | None = None,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        data_parallel_rank: int = 0,
        data_parallel_size: int = 1,
        blend_seed: int = 0,
        shuffle: bool = True,
        optimizer_step_unit: str = "microbatch",
        minimum_sequence_length: int | None = None,
        block_size: int = 1,
        context_parallel_size: int = 1,
        block_parallel_size: int = 1,
        group_by_supervision_start: bool = False,
    ) -> DataRuntime:
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(str(path))
        if path.is_dir():
            manifest_path = path / "manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(
                    f"prepared dataset manifest not found: {manifest_path}"
                )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("format") == PACKED_TOKEN_FORMAT:
                if int(manifest.get("version", -1)) != PACKED_TOKEN_VERSION:
                    raise RuntimeError("unsupported packed-token artifact version")
                files = manifest.get("files")
                token_metadata = (
                    files.get("tokens.i32") if isinstance(files, dict) else None
                )
                token_path = path / "tokens.i32"
                token_count = int(manifest.get("token_count", -1))
                if (
                    not isinstance(token_metadata, dict)
                    or token_metadata.get("dtype") != "int32"
                    or int(token_metadata.get("elements", -1)) != token_count
                    or int(token_metadata.get("bytes", -1)) != token_count * 4
                    or token_count <= 0
                    or not token_path.is_file()
                    or token_path.stat().st_size != token_count * 4
                ):
                    raise RuntimeError("prepared packed-token artifact is inconsistent")
                metadata = manifest.get("metadata")
                if not isinstance(metadata, dict):
                    raise RuntimeError(
                        "prepared packed-token artifact is missing metadata"
                    )
                fingerprint = metadata.get("dataset_fingerprint")
                if not isinstance(fingerprint, str) or len(fingerprint) != 64:
                    raise RuntimeError(
                        "prepared packed-token artifact is missing its dataset "
                        "fingerprint"
                    )
                if "tokenizer_vocabulary_sha256" in metadata:
                    _validate_tokenizer_identity(
                        metadata,
                        tokenizer,
                        artifact_label="prepared packed-token",
                    )
                return cls(
                    token_stream=torch.from_file(
                        str(token_path),
                        dtype=torch.int32,
                        size=token_count,
                    ),
                    batch_size=batch_size,
                    seq_len=seq_len,
                    device=device,
                    source=str(path),
                    kind="prepared_packed_tokens",
                    dataset_fingerprint=fingerprint,
                    data_parallel_rank=data_parallel_rank,
                    data_parallel_size=data_parallel_size,
                )
            return IndexedSupervisedTokenDataRuntime.from_directory(
                path,
                tokenizer=tokenizer,
                mask_token_id=mask_token_id,
                batch_size=batch_size,
                max_seq_len=seq_len,
                block_size=block_size,
                context_parallel_size=context_parallel_size,
                block_parallel_size=block_parallel_size,
                device=device,
                data_parallel_rank=data_parallel_rank,
                data_parallel_size=data_parallel_size,
                seed=blend_seed,
                shuffle=shuffle,
                optimizer_step_unit=optimizer_step_unit,
                minimum_sequence_length=minimum_sequence_length,
                group_by_supervision_start=group_by_supervision_start,
            )
        if path.suffix == ".json":
            return BlendedPackedTokenDataRuntime.from_manifest_path(
                manifest_path=path,
                tokenizer=tokenizer,
                batch_size=batch_size,
                seq_len=seq_len,
                device=device,
                data_parallel_rank=data_parallel_rank,
                data_parallel_size=data_parallel_size,
                seed=blend_seed,
            )
        if path.suffix == ".pt":
            payload = _load_torch_payload(path)
            tokens, kind = _token_stream_from_torch_payload(payload)
        else:
            tokens, kind = _load_token_stream(path, tokenizer=tokenizer)
        return cls(
            token_stream=tokens,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            source=str(path),
            kind=kind,
            data_parallel_rank=data_parallel_rank,
            data_parallel_size=data_parallel_size,
        )

    @classmethod
    def from_repeated_text(
        cls,
        *,
        tokenizer: Any,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        data_parallel_rank: int = 0,
        data_parallel_size: int = 1,
    ) -> "PackedTokenDataRuntime":
        ids = tokenizer.encode(_SMOKE_TEXT, add_special_tokens=False)
        if not ids:
            raise RuntimeError("tokenizer produced no tokens for text input mode")
        repeats = max(1, (seq_len * max(1, batch_size) + len(ids) - 1) // len(ids))
        tokens = torch.tensor(ids * repeats, dtype=torch.long)
        return cls(
            token_stream=tokens,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            source="smoke_repeated_text",
            kind="text",
            data_parallel_rank=data_parallel_rank,
            data_parallel_size=data_parallel_size,
        )

    def next_batch(self) -> DataBatch:
        needed = self.batch_size * self.seq_len
        if needed <= 0:
            raise ValueError("batch_size * seq_len must be positive")
        stream_len = int(self.tokens.numel())
        if self.device.type == "cuda":
            ids = self._next_cuda_batch(needed)
        else:
            flat = torch.empty(needed, dtype=torch.long)
            self._copy_batch_to(flat, cursor=self.cursor)
            ids = flat.reshape(self.batch_size, self.seq_len)
        self.cursor = int((self.cursor + needed) % stream_len)
        self.samples_consumed += self.batch_size
        return DataBatch(input_ids=ids)

    def _next_cuda_batch(self, needed: int) -> torch.Tensor:
        assert self._transfer_stream is not None
        compute_stream = torch.cuda.current_stream(self.device)
        if not self._prefetch_initialized:
            self._launch_prefetch(self._next_slot, cursor=self.cursor)
            self._prefetch_initialized = True
        if self._returned_slot is not None:
            consumed = self._consumed_events[self._returned_slot]
            consumed.record(compute_stream)
        else:
            consumed = None
        slot = self._next_slot
        compute_stream.wait_event(self._ready_events[slot])
        ids = self._device_buffers[slot].reshape(self.batch_size, self.seq_len)
        next_cursor = int((self.cursor + needed) % int(self.tokens.numel()))
        next_slot = 1 - slot
        self._launch_prefetch(next_slot, cursor=next_cursor, wait_event=consumed)
        self._returned_slot = slot
        self._next_slot = next_slot
        return ids

    def _launch_prefetch(
        self,
        slot: int,
        *,
        cursor: int,
        wait_event: torch.cuda.Event | None = None,
    ) -> None:
        assert self._transfer_stream is not None
        host = self._host_buffers[slot]
        if self._slot_initialized[slot]:
            # The pinned staging buffer cannot be rewritten until its previous
            # asynchronous H2D transfer has released it. Device-buffer reuse is
            # protected separately by ``wait_event`` below.
            self._ready_events[slot].synchronize()
        self._copy_batch_to(host, cursor=cursor)
        with torch.cuda.stream(self._transfer_stream):
            if wait_event is not None:
                self._transfer_stream.wait_event(wait_event)
            self._device_buffers[slot].copy_(host, non_blocking=True)
            self._ready_events[slot].record(self._transfer_stream)
        self._slot_initialized[slot] = True

    def _copy_batch_to(self, destination: torch.Tensor, *, cursor: int) -> None:
        needed = int(destination.numel())
        stream_len = int(self.tokens.numel())
        offset = (
            int(cursor) * int(self.data_parallel_size)
            + int(self.data_parallel_rank) * needed
        ) % stream_len
        first = min(needed, stream_len - offset)
        destination[:first].copy_(self.tokens.narrow(0, offset, first))
        copied = first
        while copied < needed:
            count = min(needed - copied, stream_len)
            destination[copied : copied + count].copy_(self.tokens.narrow(0, 0, count))
            copied += count

    def state_dict(self) -> dict[str, Any]:
        state = {
            "kind": self.kind,
            "source": self.source,
            "cursor": int(self.cursor),
            "samples_consumed": int(self.samples_consumed),
            "token_count": int(self.tokens.numel()),
            "data_parallel_rank": int(self.data_parallel_rank),
            "data_parallel_size": int(self.data_parallel_size),
        }
        if self.dataset_fingerprint is not None:
            state["dataset_fingerprint"] = self.dataset_fingerprint
        return state

    def load_state_dict(self, state: dict[str, Any] | None) -> None:
        if not state:
            return
        if self.dataset_fingerprint is not None:
            saved_fingerprint = state.get("dataset_fingerprint")
            if saved_fingerprint != self.dataset_fingerprint:
                raise RuntimeError(
                    "checkpoint dataset fingerprint does not match the current "
                    "prepared dataset"
                )
        token_count = int(state.get("token_count", self.tokens.numel()))
        if token_count != int(self.tokens.numel()):
            raise RuntimeError(
                "checkpoint data token_count does not match current dataset: "
                f"{token_count} != {int(self.tokens.numel())}"
            )
        saved_dp_size = state.get("data_parallel_size")
        saved_dp_rank = state.get("data_parallel_rank")
        if saved_dp_size is not None and int(saved_dp_size) != self.data_parallel_size:
            raise RuntimeError(
                "checkpoint data_parallel_size does not match current run: "
                f"{int(saved_dp_size)} != {self.data_parallel_size}"
            )
        if saved_dp_rank is not None and int(saved_dp_rank) != self.data_parallel_rank:
            raise RuntimeError(
                "checkpoint data_parallel_rank does not match current run: "
                f"{int(saved_dp_rank)} != {self.data_parallel_rank}"
            )
        self.cursor = int(state.get("cursor", 0)) % max(1, int(self.tokens.numel()))
        self.samples_consumed = int(state.get("samples_consumed", 0))
        self._next_slot = 0
        self._returned_slot = None
        self._prefetch_initialized = False
        if self._slot_initialized:
            self._slot_initialized = [False for _ in self._slot_initialized]

    def to_log_dict(self) -> dict[str, Any]:
        details = {
            "kind": self.kind,
            "source": self.source,
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "cursor": int(self.cursor),
            "samples_consumed": int(self.samples_consumed),
            "token_count": int(self.tokens.numel()),
            "data_parallel_rank": int(self.data_parallel_rank),
            "data_parallel_size": int(self.data_parallel_size),
        }
        if self.dataset_fingerprint is not None:
            details["dataset_fingerprint"] = self.dataset_fingerprint
        return details


class IndexedSupervisedTokenDataRuntime(DataRuntime):
    """Memory-mapped variable-length SFT data with deterministic DP batches.

    Samples are grouped by their exact block-aligned sequence length. A complete
    global batch is formed before slicing it across data-parallel ranks, so every
    rank executes the same tensor shape without padding or cross-sample attention.
    """

    def __init__(
        self,
        *,
        tokens: torch.Tensor,
        index: torch.Tensor,
        group_ids: torch.Tensor | None = None,
        loss_masks: torch.Tensor | None = None,
        batch_size: int,
        max_seq_len: int,
        block_size: int,
        context_parallel_size: int,
        block_parallel_size: int,
        device: torch.device,
        source: str,
        data_parallel_rank: int,
        data_parallel_size: int,
        seed: int,
        shuffle: bool,
        minimum_sequence_length: int | None,
        dataset_fingerprint: str,
        optimizer_step_unit: str = "microbatch",
        group_by_supervision_start: bool = False,
    ) -> None:
        if tokens.ndim != 1 or tokens.dtype != torch.int32:
            raise TypeError("indexed supervised tokens must be a flat int32 tensor")
        if index.ndim != 2 or tuple(index.shape[1:]) != (
            len(INDEXED_SUPERVISED_COLUMNS),
        ):
            raise ValueError("indexed supervised records must have shape [N, 4]")
        if index.dtype != torch.int64 or int(index.shape[0]) <= 0:
            raise TypeError("indexed supervised records must be non-empty int64")
        if group_ids is not None and (
            group_ids.dtype != torch.int64
            or group_ids.ndim != 1
            or int(group_ids.numel()) != int(index.shape[0])
        ):
            raise TypeError("indexed supervised group IDs must be one int64 per sample")
        if loss_masks is not None and (
            loss_masks.dtype != torch.uint8
            or loss_masks.ndim != 1
            or int(loss_masks.numel()) != int(tokens.numel())
        ):
            raise TypeError(
                "indexed arbitrary loss masks must be one uint8 value per token"
            )
        self.batch_size = int(batch_size)
        self.max_seq_len = int(max_seq_len)
        self.block_size = int(block_size)
        self.context_parallel_size = int(context_parallel_size)
        self.block_parallel_size = int(block_parallel_size)
        self.group_by_supervision_start = bool(group_by_supervision_start)
        self.minimum_sequence_length = (
            None if minimum_sequence_length is None else int(minimum_sequence_length)
        )
        self.data_parallel_rank = int(data_parallel_rank)
        self.data_parallel_size = int(data_parallel_size)
        if self.batch_size <= 0 or self.max_seq_len <= 0 or self.block_size <= 0:
            raise ValueError("batch size and sequence dimensions must be positive")
        if self.max_seq_len % self.block_size:
            raise ValueError("maximum sequence length must divide evenly by block_size")
        if self.context_parallel_size <= 0 or self.block_parallel_size <= 0:
            raise ValueError("context and block parallel sizes must be positive")
        if self.minimum_sequence_length is not None:
            if not 0 < self.minimum_sequence_length <= self.max_seq_len:
                raise ValueError(
                    "minimum sequence length must be positive and no greater than "
                    "the configured maximum"
                )
            if self.minimum_sequence_length % self.block_size:
                raise ValueError(
                    "minimum sequence length must divide evenly by block_size"
                )
        if not 0 <= self.data_parallel_rank < self.data_parallel_size:
            raise ValueError("invalid data-parallel coordinates")

        offsets = index[:, 0]
        lengths = index[:, 1]
        supervision_starts = index[:, 2]
        sample_ids = index[:, 3]
        if bool((offsets < 0).any()) or bool((lengths <= 0).any()):
            raise RuntimeError("indexed supervised offsets and lengths are invalid")
        if bool((offsets + lengths > int(tokens.numel())).any()):
            raise RuntimeError("indexed supervised records exceed the token file")
        if bool((supervision_starts < 0).any()) or bool(
            (supervision_starts >= lengths).any()
        ):
            raise RuntimeError("indexed supervised supervision ranges are invalid")
        if int(sample_ids.unique().numel()) != int(sample_ids.numel()):
            raise RuntimeError("indexed supervised sample IDs must be unique")

        # BP owns complete blocks, not indivisible dual-end pairs. The short
        # schedule splits pairs as needed, so one block per BP worker is enough.
        minimum_blocks = self.block_parallel_size
        minimum_length = minimum_blocks * self.block_size
        if self.minimum_sequence_length is not None:
            minimum_length = max(minimum_length, self.minimum_sequence_length)
        eligible = lengths.le(self.max_seq_len)
        eligible.logical_and_(lengths.remainder(self.block_size).eq(0))
        eligible.logical_and_(lengths.ge(minimum_length))
        # Native DiffusionGemma batches group identical response starts, but a
        # short logical response remains eligible: its objective runtime pads
        # only the physical decoder canvas so every BP worker owns one block.
        if self.context_parallel_size > 1 and self.block_parallel_size == 1:
            eligible.logical_and_(lengths.remainder(self.context_parallel_size).eq(0))
        eligible_indices = torch.nonzero(eligible, as_tuple=False).flatten()
        if int(eligible_indices.numel()) <= 0:
            raise RuntimeError(
                "indexed supervised dataset has no samples compatible with the "
                "configured maximum length and parallel topology"
            )

        self.tokens = tokens
        self.index = index
        self.group_ids = group_ids
        self.loss_masks = loss_masks
        self.eligible_indices = eligible_indices
        self.device = device
        self.source = str(source)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.optimizer_step_unit = str(optimizer_step_unit)
        if self.optimizer_step_unit not in {"microbatch", "trajectory"}:
            raise ValueError(
                "indexed optimizer step unit must be 'microbatch' or 'trajectory'"
            )
        if self.optimizer_step_unit == "trajectory":
            if group_ids is None:
                raise RuntimeError(
                    "trajectory optimizer steps require indexed trajectory group IDs"
                )
            if self.batch_size != 1 or self.data_parallel_size != 1:
                raise RuntimeError(
                    "trajectory optimizer steps currently require batch_size=1 and "
                    "data_parallel_size=1"
                )
        self.dataset_fingerprint = str(dataset_fingerprint)
        if not self.dataset_fingerprint:
            raise ValueError("indexed supervised dataset fingerprint must not be empty")
        self.epoch = 0
        self.batch_cursor = 0
        self.optimizer_step_cursor = 0
        self.samples_consumed = 0
        (
            self._batches,
            self._dropped_samples,
            self._optimizer_step_sizes,
        ) = self._build_epoch_layout()

    @classmethod
    def from_directory(
        cls,
        root: Path,
        *,
        tokenizer: Any | None,
        mask_token_id: int | None,
        batch_size: int,
        max_seq_len: int,
        block_size: int,
        context_parallel_size: int,
        block_parallel_size: int,
        device: torch.device,
        data_parallel_rank: int,
        data_parallel_size: int,
        seed: int,
        shuffle: bool,
        minimum_sequence_length: int | None,
        optimizer_step_unit: str = "microbatch",
        group_by_supervision_start: bool = False,
    ) -> "IndexedSupervisedTokenDataRuntime":
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"indexed dataset manifest not found: {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != INDEXED_SUPERVISED_FORMAT:
            raise RuntimeError(f"unsupported indexed dataset format: {root}")
        if int(manifest.get("version", -1)) != INDEXED_SUPERVISED_VERSION:
            raise RuntimeError("unsupported indexed supervised dataset version")
        metadata = manifest.get("metadata")
        _validate_supervised_tokenizer(metadata, tokenizer)
        if metadata.get("semantics") not in {
            "assistant_only_sft",
            "completion_only_sft",
            "full_sequence",
            "supervised_tokens",
        }:
            raise RuntimeError("indexed dataset uses unsupported supervision semantics")
        if (
            group_by_supervision_start
            and metadata.get("supervision_shape") == "arbitrary"
        ):
            raise RuntimeError(
                "native response-boundary batching requires full-sequence or "
                "contiguous-suffix supervision"
            )
        if int(metadata.get("block_size", -1)) != int(block_size):
            raise RuntimeError(
                "indexed dataset block size does not match the training objective"
            )
        if (
            mask_token_id is not None
            and metadata.get("contains_mask_token") is not False
        ):
            raise RuntimeError(
                "indexed source data may contain the reserved mask token"
            )

        files = manifest.get("files")
        if not isinstance(files, dict):
            raise RuntimeError("indexed dataset manifest is missing file metadata")
        token_path = _validated_indexed_file(root, files, "tokens.i32", 4)
        index_path = _validated_indexed_file(root, files, "index.i64", 8)
        group_path = None
        if "groups.i64" in files:
            group_path = _validated_indexed_file(root, files, "groups.i64", 8)
        loss_mask_path = None
        if "loss_mask.u8" in files:
            loss_mask_path = _validated_indexed_file(root, files, "loss_mask.u8", 1)
        if files["tokens.i32"].get("dtype") != "int32":
            raise RuntimeError("indexed token artifact must use int32 storage")
        if (
            files["index.i64"].get("dtype") != "int64"
            or tuple(files["index.i64"].get("columns", ()))
            != INDEXED_SUPERVISED_COLUMNS
        ):
            raise RuntimeError("indexed record artifact has an incompatible schema")
        token_count = int(files["tokens.i32"]["elements"])
        if int(manifest.get("token_count", -1)) != token_count:
            raise RuntimeError("indexed supervised token count is inconsistent")
        record_count = int(manifest.get("sample_count", -1))
        index_elements = int(files["index.i64"]["elements"])
        if record_count <= 0 or index_elements != record_count * len(
            INDEXED_SUPERVISED_COLUMNS
        ):
            raise RuntimeError("indexed supervised record count is inconsistent")
        tokens = torch.from_file(
            str(token_path),
            dtype=torch.int32,
            size=token_count,
        )
        index = torch.from_file(
            str(index_path),
            dtype=torch.int64,
            size=index_elements,
        ).reshape(record_count, len(INDEXED_SUPERVISED_COLUMNS))
        group_ids = None
        if group_path is not None:
            group_metadata = files["groups.i64"]
            if (
                group_metadata.get("dtype") != "int64"
                or int(group_metadata.get("elements", -1)) != record_count
            ):
                raise RuntimeError("indexed trajectory group artifact is inconsistent")
            group_ids = torch.from_file(
                str(group_path),
                dtype=torch.int64,
                size=record_count,
            )
        loss_masks = None
        if loss_mask_path is not None:
            loss_mask_metadata = files["loss_mask.u8"]
            if (
                loss_mask_metadata.get("dtype") != "uint8"
                or int(loss_mask_metadata.get("elements", -1)) != token_count
                or metadata.get("loss_mask_encoding") != "uint8_per_token_v1"
            ):
                raise RuntimeError(
                    "indexed arbitrary loss-mask artifact is inconsistent"
                )
            loss_masks = torch.from_file(
                str(loss_mask_path),
                dtype=torch.uint8,
                size=token_count,
            )
        return cls(
            tokens=tokens,
            index=index,
            group_ids=group_ids,
            loss_masks=loss_masks,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            block_size=block_size,
            context_parallel_size=context_parallel_size,
            block_parallel_size=block_parallel_size,
            device=device,
            source=str(root),
            data_parallel_rank=data_parallel_rank,
            data_parallel_size=data_parallel_size,
            seed=seed,
            shuffle=shuffle,
            optimizer_step_unit=optimizer_step_unit,
            group_by_supervision_start=group_by_supervision_start,
            minimum_sequence_length=minimum_sequence_length,
            dataset_fingerprint=str(metadata.get("dataset_fingerprint", "")),
        )

    def _build_epoch_layout(
        self,
    ) -> tuple[list[torch.Tensor], int, list[int] | None]:
        if self.optimizer_step_unit == "trajectory":
            batches, step_sizes = self._build_trajectory_optimizer_steps()
            return batches, 0, step_sizes
        batches, dropped = self._build_epoch_batches()
        return batches, dropped, None

    def _build_epoch_batches(self) -> tuple[list[torch.Tensor], int]:
        global_batch_size = self.batch_size * self.data_parallel_size
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self.epoch)
        if self.shuffle and self.group_ids is not None:
            return self._build_trajectory_balanced_batches(
                global_batch_size=global_batch_size,
                generator=generator,
            )
        by_key: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for index_value in self.eligible_indices.tolist():
            length = int(self.index[index_value, 1])
            key = (
                (length, int(self.index[index_value, 2]))
                if self.group_by_supervision_start
                else (length,)
            )
            by_key[key].append(int(index_value))
        keys = sorted(by_key)
        batches: list[torch.Tensor] = []
        dropped = 0
        for key in keys:
            indices = torch.tensor(by_key[key], dtype=torch.long)
            if self.shuffle and int(indices.numel()) > 1:
                permutation = torch.randperm(
                    int(indices.numel()),
                    generator=generator,
                )
                indices = indices.index_select(0, permutation)
            complete = (int(indices.numel()) // global_batch_size) * global_batch_size
            dropped += int(indices.numel()) - complete
            for start in range(0, complete, global_batch_size):
                batches.append(indices.narrow(0, start, global_batch_size))
        if self.shuffle and len(batches) > 1:
            order = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[position] for position in order]
        if not batches:
            raise RuntimeError(
                "indexed supervised dataset has no complete same-length global batch"
            )
        return batches, dropped

    def _build_trajectory_optimizer_steps(
        self,
    ) -> tuple[list[torch.Tensor], list[int]]:
        if self.group_ids is None:
            raise RuntimeError("trajectory optimizer steps require group IDs")
        eligible = set(int(value) for value in self.eligible_indices.tolist())
        by_group: dict[int, list[int]] = defaultdict(list)
        all_by_group: dict[int, list[int]] = defaultdict(list)
        for index_value, raw_group_id in enumerate(self.group_ids.tolist()):
            group_id = int(raw_group_id)
            all_by_group[group_id].append(index_value)
            if index_value in eligible:
                by_group[group_id].append(index_value)
        partial_groups = sorted(
            group_id
            for group_id, indices in all_by_group.items()
            if len(by_group.get(group_id, ())) != len(indices)
        )
        if partial_groups:
            raise RuntimeError(
                "trajectory optimizer steps cannot silently omit ineligible turns; "
                f"partially eligible group count={len(partial_groups)}"
            )
        group_ids = sorted(by_group)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self.epoch)
        if self.shuffle and len(group_ids) > 1:
            order = torch.randperm(len(group_ids), generator=generator).tolist()
            group_ids = [group_ids[position] for position in order]
        batches: list[torch.Tensor] = []
        step_sizes: list[int] = []
        for group_id in group_ids:
            indices = sorted(by_group[group_id])
            if not indices:
                raise RuntimeError("trajectory optimizer step must not be empty")
            step_sizes.append(len(indices))
            batches.extend(torch.tensor([index], dtype=torch.long) for index in indices)
        if not batches:
            raise RuntimeError("indexed supervised dataset has no trajectory steps")
        return batches, step_sizes

    def _build_trajectory_balanced_batches(
        self,
        *,
        global_batch_size: int,
        generator: torch.Generator,
    ) -> tuple[list[torch.Tensor], int]:
        if self.group_ids is None:
            raise RuntimeError("trajectory-balanced sampling requires group IDs")
        by_group: dict[int, list[int]] = defaultdict(list)
        for index_value in self.eligible_indices.tolist():
            group_id = int(self.group_ids[index_value])
            by_group[group_id].append(int(index_value))
        group_ids = sorted(by_group)
        if len(group_ids) > 1:
            order = torch.randperm(len(group_ids), generator=generator).tolist()
            group_ids = [group_ids[position] for position in order]
        for group_id in group_ids:
            indices = by_group[group_id]
            if len(indices) > 1:
                order = torch.randperm(len(indices), generator=generator).tolist()
                by_group[group_id] = [indices[position] for position in order]

        ordered: list[int] = []
        maximum_turns = max(len(indices) for indices in by_group.values())
        for round_index in range(maximum_turns):
            active = [
                group_id
                for group_id in group_ids
                if round_index < len(by_group[group_id])
            ]
            if len(active) > 1:
                order = torch.randperm(len(active), generator=generator).tolist()
                active = [active[position] for position in order]
            ordered.extend(by_group[group_id][round_index] for group_id in active)

        pending: dict[tuple[int, ...], list[int]] = defaultdict(list)
        batches: list[torch.Tensor] = []
        for index_value in ordered:
            length = int(self.index[index_value, 1])
            key = (
                (length, int(self.index[index_value, 2]))
                if self.group_by_supervision_start
                else (length,)
            )
            bucket = pending[key]
            bucket.append(index_value)
            if len(bucket) == global_batch_size:
                batches.append(torch.tensor(bucket, dtype=torch.long))
                pending[key] = []
        dropped = sum(len(indices) for indices in pending.values())
        if not batches:
            raise RuntimeError(
                "indexed supervised dataset has no complete same-length global batch"
            )
        return batches, dropped

    def _advance_epoch(self) -> None:
        self.epoch += 1
        self.batch_cursor = 0
        self.optimizer_step_cursor = 0
        (
            self._batches,
            self._dropped_samples,
            self._optimizer_step_sizes,
        ) = self._build_epoch_layout()

    def _next_batch_from_current_epoch(self) -> DataBatch:
        global_indices = self._batches[self.batch_cursor]
        self.batch_cursor += 1
        start = self.data_parallel_rank * self.batch_size
        local_indices = global_indices.narrow(0, start, self.batch_size)
        sequence_length = int(self.index[int(local_indices[0]), 1])
        pin_memory = self.device.type == "cuda"
        input_ids = torch.empty(
            (self.batch_size, sequence_length),
            dtype=torch.long,
            pin_memory=pin_memory,
        )
        loss_mask = torch.zeros(
            (self.batch_size, sequence_length),
            dtype=torch.bool,
            pin_memory=pin_memory,
        )
        sample_ids = torch.empty(self.batch_size, dtype=torch.int64)
        response_starts = torch.empty(self.batch_size, dtype=torch.int64)
        supervised_tokens = 0
        for row, sample_index in enumerate(local_indices.tolist()):
            offset, length, supervision_start, sample_id = (
                int(value) for value in self.index[sample_index].tolist()
            )
            if length != sequence_length:
                raise RuntimeError(
                    "indexed global batch contains mixed sequence lengths"
                )
            input_ids[row].copy_(self.tokens.narrow(0, offset, length))
            if self.loss_masks is None:
                loss_mask[row, supervision_start:] = True
                sample_supervised_tokens = length - supervision_start
            else:
                stored_mask = self.loss_masks.narrow(0, offset, length)
                if bool(stored_mask.gt(1).any()):
                    raise RuntimeError(
                        "indexed loss mask contains a value other than 0 or 1"
                    )
                loss_mask[row].copy_(stored_mask.to(torch.bool))
                sample_supervised_tokens = int(stored_mask.sum())
                if sample_supervised_tokens <= 0:
                    raise RuntimeError("indexed sample contains no supervised tokens")
            sample_ids[row] = sample_id
            response_starts[row] = supervision_start
            supervised_tokens += sample_supervised_tokens
        self.samples_consumed += self.batch_size
        return DataBatch(
            input_ids=input_ids.to(
                device=self.device,
                non_blocking=pin_memory,
            ),
            loss_mask=loss_mask.to(
                device=self.device,
                non_blocking=pin_memory,
            ),
            sample_ids=sample_ids,
            supervised_token_count=supervised_tokens,
            response_starts=response_starts.to(
                device=self.device,
                non_blocking=pin_memory,
            ),
        )

    def next_batch(self) -> DataBatch:
        return self.next_batches(1)[0]

    def next_batches(self, count: int) -> list[DataBatch]:
        count = int(count)
        if count <= 0:
            raise ValueError("batch count must be positive")
        if count > len(self._batches):
            raise ValueError(
                "indexed supervised dataset must contain one complete accumulated "
                "optimizer step"
            )
        if self.batch_cursor + count > len(self._batches):
            self._advance_epoch()
        return [self._next_batch_from_current_epoch() for _ in range(count)]

    def next_optimizer_step_batches(self, count: int) -> list[DataBatch]:
        if self.optimizer_step_unit != "trajectory":
            return self.next_batches(count)
        if int(count) != 1:
            raise RuntimeError(
                "trajectory optimizer steps require configured "
                "gradient_accumulation_steps=1; group size is determined by data"
            )
        if self._optimizer_step_sizes is None:
            raise RuntimeError("trajectory optimizer-step layout is missing")
        if self.optimizer_step_cursor >= len(self._optimizer_step_sizes):
            self._advance_epoch()
        microbatch_count = self._optimizer_step_sizes[self.optimizer_step_cursor]
        self.optimizer_step_cursor += 1
        return [
            self._next_batch_from_current_epoch() for _ in range(int(microbatch_count))
        ]

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "indexed_supervised_tokens",
            "source": self.source,
            "sample_count": int(self.index.shape[0]),
            "eligible_sample_count": int(self.eligible_indices.numel()),
            "batch_size": self.batch_size,
            "maximum_sequence_length": self.max_seq_len,
            "minimum_sequence_length": self.minimum_sequence_length,
            "block_size": self.block_size,
            "context_parallel_size": self.context_parallel_size,
            "block_parallel_size": self.block_parallel_size,
            "epoch": self.epoch,
            "batch_cursor": self.batch_cursor,
            "samples_consumed": self.samples_consumed,
            "data_parallel_rank": self.data_parallel_rank,
            "data_parallel_size": self.data_parallel_size,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "optimizer_step_unit": self.optimizer_step_unit,
            "group_by_supervision_start": self.group_by_supervision_start,
            "optimizer_step_cursor": self.optimizer_step_cursor,
            "dataset_fingerprint": self.dataset_fingerprint,
        }

    def load_state_dict(self, state: dict[str, Any] | None) -> None:
        if not state:
            return
        expected = {
            "kind": "indexed_supervised_tokens",
            "sample_count": int(self.index.shape[0]),
            "eligible_sample_count": int(self.eligible_indices.numel()),
            "batch_size": self.batch_size,
            "maximum_sequence_length": self.max_seq_len,
            "minimum_sequence_length": self.minimum_sequence_length,
            "block_size": self.block_size,
            "context_parallel_size": self.context_parallel_size,
            "block_parallel_size": self.block_parallel_size,
            "data_parallel_rank": self.data_parallel_rank,
            "data_parallel_size": self.data_parallel_size,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "optimizer_step_unit": self.optimizer_step_unit,
            "group_by_supervision_start": self.group_by_supervision_start,
            "dataset_fingerprint": self.dataset_fingerprint,
        }
        for name, value in expected.items():
            if state.get(name) != value:
                raise RuntimeError(
                    f"checkpoint indexed supervised data {name} does not match: "
                    f"{state.get(name)!r} != {value!r}"
                )
        self.epoch = int(state.get("epoch", 0))
        self.batch_cursor = int(state.get("batch_cursor", 0))
        self.optimizer_step_cursor = int(state.get("optimizer_step_cursor", 0))
        self.samples_consumed = int(state.get("samples_consumed", 0))
        (
            self._batches,
            self._dropped_samples,
            self._optimizer_step_sizes,
        ) = self._build_epoch_layout()
        if not 0 <= self.batch_cursor <= len(self._batches):
            raise RuntimeError("checkpoint indexed data batch cursor is out of range")
        if self._optimizer_step_sizes is not None:
            if not 0 <= self.optimizer_step_cursor <= len(self._optimizer_step_sizes):
                raise RuntimeError(
                    "checkpoint indexed optimizer-step cursor is out of range"
                )
            expected_batch_cursor = sum(
                self._optimizer_step_sizes[: self.optimizer_step_cursor]
            )
            if self.batch_cursor != expected_batch_cursor:
                raise RuntimeError(
                    "checkpoint indexed batch and trajectory-step cursors diverge"
                )

    def to_log_dict(self) -> dict[str, Any]:
        lengths = self.index.index_select(0, self.eligible_indices)[:, 1]
        return {
            "kind": "indexed_supervised_tokens",
            "source": self.source,
            "sample_count": int(self.index.shape[0]),
            "eligible_sample_count": int(self.eligible_indices.numel()),
            "minimum_sequence_length": int(lengths.min()),
            "maximum_sequence_length": int(lengths.max()),
            "configured_maximum_sequence_length": self.max_seq_len,
            "length_bucket_count": int(lengths.unique().numel()),
            "dropped_incomplete_bucket_samples": self._dropped_samples,
            "epoch_batches": len(self._batches),
            "epoch": self.epoch,
            "batch_cursor": self.batch_cursor,
            "samples_consumed": self.samples_consumed,
            "data_parallel_rank": self.data_parallel_rank,
            "data_parallel_size": self.data_parallel_size,
            "shuffle": self.shuffle,
            "optimizer_step_unit": self.optimizer_step_unit,
            "group_by_supervision_start": self.group_by_supervision_start,
            "epoch_optimizer_steps": (
                len(self._optimizer_step_sizes)
                if self._optimizer_step_sizes is not None
                else len(self._batches)
            ),
            "optimizer_step_cursor": self.optimizer_step_cursor,
            "dataset_fingerprint": self.dataset_fingerprint,
        }


class BlendedPackedTokenDataRuntime(DataRuntime):
    """Deterministic weighted blend of multiple packed token streams."""

    def __init__(
        self,
        *,
        datasets: list[PackedTokenDataRuntime],
        weights: torch.Tensor,
        seed: int,
        source: str,
    ) -> None:
        if not datasets:
            raise ValueError("blended dataset requires at least one source")
        if weights.ndim != 1 or int(weights.numel()) != len(datasets):
            raise ValueError("blend weights must match dataset count")
        if not bool(torch.isfinite(weights).all()) or float(weights.sum()) <= 0.0:
            raise ValueError("blend weights must be finite and positive")
        self.datasets = list(datasets)
        self.weights = (weights.float() / weights.float().sum()).cpu()
        self.source = str(source)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))
        self.samples_consumed = 0

    @classmethod
    def from_manifest_path(
        cls,
        *,
        manifest_path: Path,
        tokenizer: Any | None,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        data_parallel_rank: int,
        data_parallel_size: int,
        seed: int,
    ) -> "BlendedPackedTokenDataRuntime":
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = manifest.get("datasets") if isinstance(manifest, dict) else manifest
        if not isinstance(entries, list) or not entries:
            raise ValueError("dataset manifest must contain a non-empty datasets list")
        datasets: list[PackedTokenDataRuntime] = []
        weights: list[float] = []
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("dataset manifest entries must be mappings")
            entry_path = Path(str(entry.get("path", "")))
            if not entry_path.is_absolute():
                entry_path = manifest_path.parent / entry_path
            tokens, kind = _load_token_stream(entry_path, tokenizer=tokenizer)
            datasets.append(
                PackedTokenDataRuntime(
                    token_stream=tokens,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    device=device,
                    source=str(entry_path),
                    kind=kind,
                    data_parallel_rank=data_parallel_rank,
                    data_parallel_size=data_parallel_size,
                )
            )
            weights.append(float(entry.get("weight", 1.0)))
        return cls(
            datasets=datasets,
            weights=torch.tensor(weights, dtype=torch.float32),
            seed=int(seed),
            source=str(manifest_path),
        )

    def next_batch(self) -> DataBatch:
        index = int(
            torch.multinomial(
                self.weights,
                num_samples=1,
                replacement=True,
                generator=self.generator,
            ).item()
        )
        batch = self.datasets[index].next_batch()
        self.samples_consumed += int(batch.input_ids.shape[0])
        return batch

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "blend",
            "source": self.source,
            "samples_consumed": int(self.samples_consumed),
            "weights": [float(value) for value in self.weights.tolist()],
            "generator_state": self.generator.get_state(),
            "datasets": [dataset.state_dict() for dataset in self.datasets],
        }

    def load_state_dict(self, state: dict[str, Any] | None) -> None:
        if not state:
            return
        if str(state.get("kind", "")) != "blend":
            raise RuntimeError("checkpoint data kind does not match blended runtime")
        saved_datasets = state.get("datasets")
        if not isinstance(saved_datasets, list) or len(saved_datasets) != len(
            self.datasets
        ):
            raise RuntimeError("checkpoint blended dataset count does not match")
        for dataset, dataset_state in zip(self.datasets, saved_datasets):
            dataset.load_state_dict(dataset_state)
        generator_state = state.get("generator_state")
        if generator_state is not None:
            self.generator.set_state(generator_state.cpu())
        self.samples_consumed = int(state.get("samples_consumed", 0))

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "kind": "blend",
            "source": self.source,
            "samples_consumed": int(self.samples_consumed),
            "weights": [float(value) for value in self.weights.tolist()],
            "datasets": [dataset.to_log_dict() for dataset in self.datasets],
        }


def _load_token_stream(
    path: Path,
    *,
    tokenizer: Any | None,
) -> tuple[torch.Tensor, str]:
    if path.suffix == ".pt":
        return _token_stream_from_torch_payload(_load_torch_payload(path))
    if path.suffix in {".bin", ".tokens", ".i64", ".int64"}:
        element_size = torch.empty((), dtype=torch.int64).element_size()
        if path.stat().st_size % element_size != 0:
            raise ValueError(f"binary int64 token file has invalid byte size: {path}")
        return (
            torch.from_file(
                str(path),
                dtype=torch.int64,
                size=path.stat().st_size // element_size,
            ).reshape(-1),
            "binary_i64",
        )
    if path.suffix in {".i32", ".int32"}:
        element_size = torch.empty((), dtype=torch.int32).element_size()
        if path.stat().st_size % element_size != 0:
            raise ValueError(f"binary int32 token file has invalid byte size: {path}")
        return (
            torch.from_file(
                str(path),
                dtype=torch.int32,
                size=path.stat().st_size // element_size,
            ).reshape(-1),
            "binary_i32",
        )
    if tokenizer is None:
        raise RuntimeError("text dataset files require a tokenizer-backed model family")
    text = path.read_text(encoding="utf-8")
    ids = tokenizer.encode(text, add_special_tokens=False)
    return torch.tensor(ids, dtype=torch.long), "text"


def _load_torch_payload(path: Path) -> Any:
    return torch.load(
        path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )


def _token_stream_from_torch_payload(payload: Any) -> tuple[torch.Tensor, str]:
    if isinstance(payload, dict):
        for key in ("input_ids", "tokens", "data"):
            if key in payload:
                payload = payload[key]
                break
    return torch.as_tensor(payload).reshape(-1), "torch"


def _validate_supervised_tokenizer(metadata: Any, tokenizer: Any | None) -> None:
    _validate_tokenizer_identity(
        metadata,
        tokenizer,
        artifact_label="supervised-token",
    )


def _validate_tokenizer_identity(
    metadata: Any,
    tokenizer: Any | None,
    *,
    artifact_label: str,
) -> None:
    if not isinstance(metadata, dict):
        raise RuntimeError(f"{artifact_label} dataset is missing tokenizer metadata")
    if tokenizer is None:
        raise RuntimeError(f"{artifact_label} datasets require the recorded tokenizer")
    observed_size = len(tokenizer)
    expected_size = int(metadata.get("tokenizer_size", -1))
    if observed_size != expected_size:
        raise RuntimeError(
            f"{artifact_label} tokenizer size does not match the dataset: "
            f"{observed_size} != {expected_size}"
        )
    observed_vocab_hash = tokenizer_vocabulary_sha256(tokenizer)
    if observed_vocab_hash != metadata.get("tokenizer_vocabulary_sha256"):
        raise RuntimeError(
            f"{artifact_label} tokenizer vocabulary does not match the dataset"
        )
    observed_mask = getattr(tokenizer, "mask_token_id", None)
    expected_mask = metadata.get("mask_token_id")
    if observed_mask != expected_mask:
        raise RuntimeError(
            f"{artifact_label} mask token does not match the dataset: "
            f"{observed_mask!r} != {expected_mask!r}"
        )
    template = str(getattr(tokenizer, "chat_template", "") or "")
    observed_hash = hashlib.sha256(template.encode("utf-8")).hexdigest()
    if observed_hash != metadata.get("chat_template_sha256"):
        raise RuntimeError(f"{artifact_label} chat template does not match the dataset")


def _validated_indexed_file(
    root: Path,
    files: dict[str, Any],
    name: str,
    element_size: int,
) -> Path:
    metadata = files.get(name)
    path = root / name
    if not isinstance(metadata, dict) or not path.is_file():
        raise RuntimeError(f"indexed supervised artifact is missing: {name}")
    elements = int(metadata.get("elements", -1))
    expected_bytes = elements * int(element_size)
    if elements < 0 or int(metadata.get("bytes", -1)) != expected_bytes:
        raise RuntimeError(f"indexed supervised metadata is invalid: {name}")
    if path.stat().st_size != expected_bytes:
        raise RuntimeError(f"indexed supervised artifact size does not match: {name}")
    return path


def tokenizer_vocabulary_sha256(tokenizer: Any) -> str:
    """Hash the complete token-to-ID mapping used by a prepared dataset."""

    getter = getattr(tokenizer, "get_vocab", None)
    if not callable(getter):
        raise RuntimeError("supervised-token preparation requires tokenizer.get_vocab")
    vocabulary = getter()
    if not isinstance(vocabulary, dict):
        raise RuntimeError("tokenizer.get_vocab must return a token-to-ID mapping")
    digest = hashlib.sha256()
    for token, token_id in sorted(vocabulary.items(), key=lambda item: item[0]):
        encoded = str(token).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="little"))
        digest.update(encoded)
        digest.update(int(token_id).to_bytes(8, byteorder="little", signed=True))
    return digest.hexdigest()
