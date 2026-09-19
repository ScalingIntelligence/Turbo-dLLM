# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Restartable verifier-feature data for DFlash training."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import threading
from typing import Any

import torch

from dllm_parallel.core.attention.layout import clean_intervals_for_runtime

FEATURE_FORMAT = "dllm_parallel.dflash_features.v1"
FEATURE_SHARDED_FORMAT = "dllm_parallel.dflash_features.sharded.v1"
FEATURE_COMPACT_SHARDED_FORMAT = (
    "dllm_parallel.dflash_features.compact_sharded.v1"
)


@dataclass(frozen=True)
class DFlashFeatureBatch:
    input_ids: torch.Tensor
    loss_mask: torch.Tensor
    document_ids: torch.Tensor
    position_ids: torch.Tensor
    target_hidden_states: torch.Tensor
    target_position_ids: torch.Tensor
    verifier_last_hidden_states: "DFlashVerifierFeatureBatch | None"


@dataclass(frozen=True)
class DFlashVerifierFeatureBatch:
    """Lazy CPU view that materializes only sampled verifier rows."""

    source: torch.Tensor
    sample_indices: torch.Tensor
    compute_dtype: torch.dtype

    @property
    def device(self) -> torch.device:
        return self.source.device

    @property
    def dtype(self) -> torch.dtype:
        return self.compute_dtype

    @property
    def shape(self) -> torch.Size:
        return torch.Size(
            (int(self.sample_indices.numel()), *self.source.shape[1:])
        )

    def gather(
        self,
        positions: torch.Tensor,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if self.source.device.type != "cpu":
            raise ValueError("DFlash offline verifier features must remain on CPU")
        indices = positions.reshape(positions.shape[0], -1).to(
            device="cpu",
            dtype=torch.int64,
        )
        if int(indices.shape[0]) != int(self.sample_indices.numel()):
            raise ValueError("DFlash teacher positions do not match the feature batch")
        selected = torch.empty(
            (
                int(indices.shape[0]),
                int(indices.shape[1]),
                *self.source.shape[2:],
            ),
            dtype=self.source.dtype,
            device="cpu",
            pin_memory=device.type == "cuda",
        )
        for batch_index, sample_index in enumerate(self.sample_indices.tolist()):
            torch.index_select(
                self.source[int(sample_index)],
                0,
                indices[batch_index],
                out=selected[batch_index],
            )
        return selected.to(
            device=device,
            dtype=self.compute_dtype,
            non_blocking=True,
        )


@dataclass(frozen=True)
class DFlashSyntheticVerifierFeatureBatch:
    """Profile-only verifier rows generated without a dense host feature file."""

    hidden_size: int
    batch_size: int
    dtype: torch.dtype

    def gather(
        self,
        positions: torch.Tensor,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        rows = positions.reshape(int(self.batch_size), -1)
        return torch.zeros(
            (int(rows.shape[0]), int(rows.shape[1]), int(self.hidden_size)),
            dtype=self.dtype,
            device=device,
        )


@dataclass(frozen=True)
class _DFlashHostFeatureBatch:
    cursor: int
    sample_indices: torch.Tensor
    selected: dict[str, torch.Tensor]
    target_hidden_states: torch.Tensor


@dataclass(frozen=True)
class _DFlashDeviceFeatureBatch:
    cursor: int
    sample_indices: torch.Tensor
    selected: dict[str, torch.Tensor]
    target_hidden_states: torch.Tensor
    ready: torch.cuda.Event


@dataclass(frozen=True)
class DFlashFeatureManifest:
    verifier_id: str
    verifier_revision: str | None
    target_layer_ids: tuple[int, ...]
    sequence_length: int
    sample_count: int
    target_feature_width: int
    verifier_hidden_size: int | None
    storage_format: str = FEATURE_FORMAT
    records: tuple[dict[str, Any], ...] = ()

    @classmethod
    def load(cls, feature_path: Path) -> "DFlashFeatureManifest":
        path = (
            feature_path / "manifest.json"
            if feature_path.is_dir()
            else feature_path.with_name(feature_path.name + ".manifest.json")
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"DFlash feature manifest is missing: {path}"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("format") not in {
            FEATURE_FORMAT,
            FEATURE_SHARDED_FORMAT,
            FEATURE_COMPACT_SHARDED_FORMAT,
        }:
            raise ValueError(f"invalid DFlash feature manifest: {path}")
        storage_format = str(payload["format"])
        raw_records = payload.get("records", ())
        if storage_format in {
            FEATURE_SHARDED_FORMAT,
            FEATURE_COMPACT_SHARDED_FORMAT,
        }:
            if not isinstance(raw_records, list) or not raw_records:
                raise ValueError("sharded DFlash manifest requires records")
            if not all(isinstance(record, dict) for record in raw_records):
                raise ValueError("sharded DFlash records must be objects")
        elif raw_records:
            raise ValueError("monolithic DFlash manifest cannot declare records")
        return cls(
            verifier_id=str(payload["verifier_id"]),
            verifier_revision=(
                str(payload["verifier_revision"])
                if payload.get("verifier_revision") is not None
                else None
            ),
            target_layer_ids=tuple(int(value) for value in payload["target_layer_ids"]),
            sequence_length=int(payload["sequence_length"]),
            sample_count=int(payload["sample_count"]),
            target_feature_width=int(payload["target_feature_width"]),
            verifier_hidden_size=(
                int(payload["verifier_hidden_size"])
                if payload.get("verifier_hidden_size") is not None
                else None
            ),
            storage_format=storage_format,
            records=tuple(dict(record) for record in raw_records),
        )

    def validate(
        self,
        *,
        verifier_id: str,
        verifier_revision: str | None,
        target_layer_ids: tuple[int, ...],
        sequence_length: int,
        require_teacher_features: bool,
    ) -> None:
        expected = {
            "verifier_id": (self.verifier_id, str(verifier_id)),
            "target_layer_ids": (
                self.target_layer_ids,
                tuple(int(value) for value in target_layer_ids),
            ),
            "sequence_length": (self.sequence_length, int(sequence_length)),
        }
        if verifier_revision is not None:
            expected["verifier_revision"] = (
                self.verifier_revision,
                str(verifier_revision),
            )
        for name, (observed, required) in expected.items():
            if observed != required:
                raise ValueError(
                    f"DFlash feature manifest {name} mismatch: "
                    f"{observed!r} != {required!r}"
                )
        if self.sample_count <= 0 or self.target_feature_width <= 0:
            raise ValueError("DFlash feature manifest dimensions must be positive")
        if self.storage_format in {
            FEATURE_SHARDED_FORMAT,
            FEATURE_COMPACT_SHARDED_FORMAT,
        } and (
            len(self.records) != self.sample_count
        ):
            raise ValueError(
                "DFlash sharded record count does not match its sample_count"
            )
        if require_teacher_features and (
            self.verifier_hidden_size is None or self.verifier_hidden_size <= 0
        ):
            raise ValueError(
                "DFlash KL feature manifest requires verifier_hidden_size"
            )


class DFlashFeatureDataRuntime:
    _REQUIRED_TENSORS = ("input_ids", "loss_mask", "target_hidden_states")

    def __init__(
        self,
        *,
        tensors: dict[str, torch.Tensor],
        manifest: DFlashFeatureManifest,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        source: str,
        require_teacher_features: bool,
        target_intervals: tuple[tuple[int, int], ...],
        data_parallel_rank: int,
        data_parallel_size: int,
        tensors_are_normalized: bool = False,
    ) -> None:
        missing = [name for name in self._REQUIRED_TENSORS if name not in tensors]
        if require_teacher_features and "verifier_last_hidden_states" not in tensors:
            missing.append("verifier_last_hidden_states")
        if missing:
            raise ValueError(
                "DFlash feature dataset is missing tensors: " + ", ".join(missing)
            )
        self.seq_len = int(manifest.sequence_length)
        normalized = (
            dict(tensors)
            if tensors_are_normalized
            else {
                name: _normalize_feature_tensor(name, tensor, seq_len=self.seq_len)
                for name, tensor in tensors.items()
                if torch.is_tensor(tensor)
            }
        )
        sample_count = int(normalized["input_ids"].shape[0])
        if sample_count != manifest.sample_count:
            raise ValueError(
                "DFlash feature sample count does not match its manifest"
            )
        for name, tensor in normalized.items():
            if int(tensor.shape[0]) != sample_count:
                raise ValueError(
                    f"DFlash feature tensor {name!r} has a mismatched sample count"
                )
        if int(normalized["target_hidden_states"].shape[-1]) != (
            manifest.target_feature_width
        ):
            raise ValueError(
                "DFlash target feature width does not match its manifest"
            )
        verifier_hidden = normalized.get("verifier_last_hidden_states")
        if verifier_hidden is not None and int(verifier_hidden.shape[-1]) != (
            manifest.verifier_hidden_size
        ):
            raise ValueError(
                "DFlash verifier hidden width does not match its manifest"
            )
        self.tensors = normalized
        self.manifest = manifest
        self.batch_size = int(batch_size)
        self.device = device
        self.dtype = dtype
        self.source = str(source)
        self.require_teacher_features = bool(require_teacher_features)
        self.target_intervals = _validate_token_intervals(
            target_intervals,
            sequence_length=self.seq_len,
        )
        self.data_parallel_rank = int(data_parallel_rank)
        self.data_parallel_size = int(data_parallel_size)
        if self.batch_size <= 0:
            raise ValueError("DFlash feature batch_size must be positive")
        if not 0 <= self.data_parallel_rank < self.data_parallel_size:
            raise ValueError("invalid DFlash feature data-parallel coordinate")
        self.cursor = 0
        self.samples_consumed = 0
        self._prefetch_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="dflash-features")
            if self.device.type == "cuda"
            else None
        )
        self._transfer_stream = (
            torch.cuda.Stream(device=self.device)
            if self.device.type == "cuda"
            else None
        )
        self._device_batch: (
            Future[_DFlashDeviceFeatureBatch] | _DFlashDeviceFeatureBatch | None
        ) = None
        self._next_host_batch: Future[_DFlashHostFeatureBatch] | None = None
        self._start_prefetch()

    @classmethod
    def from_path(
        cls,
        *,
        path: str,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        require_teacher_features: bool,
        verifier_id: str,
        verifier_revision: str | None,
        target_layer_ids: tuple[int, ...],
        runtime: Any | None,
        data_parallel_rank: int,
        data_parallel_size: int,
    ) -> "DFlashFeatureDataRuntime":
        feature_path = Path(path).expanduser()
        if not feature_path.exists() or not (
            feature_path.is_file() or feature_path.is_dir()
        ):
            raise FileNotFoundError(feature_path)
        manifest = DFlashFeatureManifest.load(feature_path)
        manifest.validate(
            verifier_id=str(verifier_id),
            verifier_revision=verifier_revision,
            target_layer_ids=target_layer_ids,
            sequence_length=int(seq_len),
            require_teacher_features=bool(require_teacher_features),
        )
        tensors = _load_feature_tensors(feature_path, manifest=manifest)
        return cls(
            tensors=tensors,
            manifest=manifest,
            batch_size=int(batch_size),
            device=device,
            dtype=dtype,
            source=str(feature_path),
            require_teacher_features=bool(require_teacher_features),
            target_intervals=_target_intervals(int(seq_len), runtime),
            data_parallel_rank=int(data_parallel_rank),
            data_parallel_size=int(data_parallel_size),
            tensors_are_normalized=(
                manifest.storage_format
                in {FEATURE_SHARDED_FORMAT, FEATURE_COMPACT_SHARDED_FORMAT}
            ),
        )

    def next_batch(self) -> DFlashFeatureBatch:
        if self.device.type == "cuda":
            device_batch = self._take_device_batch()
            if device_batch.cursor != self.cursor:
                raise RuntimeError("DFlash device prefetch cursor is out of sequence")
            consumer_stream = torch.cuda.current_stream(self.device)
            consumer_stream.wait_event(device_batch.ready)
            selected = device_batch.selected
            target_hidden_states = device_batch.target_hidden_states
            for tensor in selected.values():
                tensor.record_stream(consumer_stream)
            target_hidden_states.record_stream(consumer_stream)
            sample_indices = device_batch.sample_indices
        else:
            host_batch = self._prepare_host_batch(int(self.cursor))
            selected = host_batch.selected
            target_hidden_states = host_batch.target_hidden_states.to(dtype=self.dtype)
            sample_indices = host_batch.sample_indices
        next_cursor = _next_cursor(
            self.cursor,
            batch_size=self.batch_size,
            sample_count=self.manifest.sample_count,
        )
        self.cursor = next_cursor
        self.samples_consumed += self.batch_size
        self._advance_prefetch()
        input_ids = selected["input_ids"].to(torch.long).contiguous()
        loss_mask = selected["loss_mask"].to(torch.bool).contiguous()
        document_ids = selected.get("document_ids")
        if document_ids is None:
            document_ids = torch.zeros_like(input_ids)
        else:
            document_ids = document_ids.to(torch.long).contiguous()
        position_ids = selected.get("position_ids")
        if position_ids is None:
            position_ids = torch.arange(
                self.seq_len,
                device=self.device,
                dtype=torch.long,
            ).expand(self.batch_size, -1)
        else:
            position_ids = position_ids.to(torch.long).contiguous()
        target_position_ids = _cat_token_intervals(
            position_ids,
            self.target_intervals,
        ).contiguous()
        return DFlashFeatureBatch(
            input_ids=input_ids,
            loss_mask=loss_mask,
            document_ids=document_ids,
            position_ids=position_ids,
            target_hidden_states=target_hidden_states.contiguous(),
            target_position_ids=target_position_ids,
            verifier_last_hidden_states=(
                DFlashVerifierFeatureBatch(
                    source=self.tensors["verifier_last_hidden_states"],
                    sample_indices=sample_indices,
                    compute_dtype=self.dtype,
                )
                if self.require_teacher_features
                else None
            ),
        )

    def next_batches(self, count: int) -> list[DFlashFeatureBatch]:
        """Return the ordered microbatches for one optimizer step."""

        count = int(count)
        if count <= 0:
            raise ValueError("batch count must be positive")
        return [self.next_batch() for _ in range(count)]

    def next_optimizer_step_batches(self, count: int) -> list[DFlashFeatureBatch]:
        """Implement the shared trainer's optimizer-step data contract."""

        return self.next_batches(count)

    def _prepare_host_batch(self, cursor: int) -> _DFlashHostFeatureBatch:
        indices = _sample_indices(
            cursor=int(cursor),
            batch_size=self.batch_size,
            sample_count=self.manifest.sample_count,
            data_parallel_rank=self.data_parallel_rank,
            data_parallel_size=self.data_parallel_size,
        )
        pin_memory = self.device.type == "cuda"
        selected = {
            name: _copy_samples_to_host(
                tensor,
                indices,
                pin_memory=pin_memory,
            )
            for name, tensor in self.tensors.items()
            if name not in {
                "target_hidden_states",
                "verifier_last_hidden_states",
            }
        }
        target_hidden_states = _copy_samples_and_token_intervals_to_host(
            self.tensors["target_hidden_states"],
            indices,
            self.target_intervals,
            pin_memory=pin_memory,
        )
        return _DFlashHostFeatureBatch(
            cursor=int(cursor),
            sample_indices=indices,
            selected=selected,
            target_hidden_states=target_hidden_states,
        )

    def _prepare_device_batch(self, cursor: int) -> _DFlashDeviceFeatureBatch:
        return self._copy_host_batch_to_device(
            self._prepare_host_batch(int(cursor))
        )

    def _copy_host_batch_to_device(
        self,
        host_batch: _DFlashHostFeatureBatch,
    ) -> _DFlashDeviceFeatureBatch:
        if self._transfer_stream is None:
            raise RuntimeError("DFlash device prefetch requires a CUDA stream")
        with torch.cuda.device(self.device), torch.cuda.stream(self._transfer_stream):
            selected = {
                name: tensor.to(device=self.device, non_blocking=True)
                for name, tensor in host_batch.selected.items()
            }
            target_hidden_states = host_batch.target_hidden_states.to(
                device=self.device,
                dtype=self.dtype,
                non_blocking=True,
            )
            ready = torch.cuda.Event()
            ready.record(self._transfer_stream)
        return _DFlashDeviceFeatureBatch(
            cursor=host_batch.cursor,
            sample_indices=host_batch.sample_indices,
            selected=selected,
            target_hidden_states=target_hidden_states,
            ready=ready,
        )

    def _start_prefetch(self) -> None:
        if self._prefetch_executor is None:
            return
        if self._device_batch is not None or self._next_host_batch is not None:
            raise RuntimeError("DFlash prefetch pipeline is already active")
        self._device_batch = self._prefetch_executor.submit(
            self._prepare_device_batch,
            int(self.cursor),
        )
        self._next_host_batch = self._prefetch_executor.submit(
            self._prepare_host_batch,
            _next_cursor(
                self.cursor,
                batch_size=self.batch_size,
                sample_count=self.manifest.sample_count,
            ),
        )

    def _advance_prefetch(self) -> None:
        if self._prefetch_executor is None:
            return
        if self._device_batch is not None:
            raise RuntimeError("DFlash current device batch was not consumed")
        if self._next_host_batch is None:
            raise RuntimeError("DFlash next host batch was not prepared")
        host_batch = self._next_host_batch.result()
        self._next_host_batch = None
        if host_batch.cursor != self.cursor:
            raise RuntimeError("DFlash host prefetch cursor is out of sequence")
        self._device_batch = self._copy_host_batch_to_device(host_batch)
        self._next_host_batch = self._prefetch_executor.submit(
            self._prepare_host_batch,
            _next_cursor(
                self.cursor,
                batch_size=self.batch_size,
                sample_count=self.manifest.sample_count,
            ),
        )

    def _take_device_batch(self) -> _DFlashDeviceFeatureBatch:
        if self._device_batch is None:
            return self._prepare_device_batch(int(self.cursor))
        device_batch = (
            self._device_batch.result()
            if isinstance(self._device_batch, Future)
            else self._device_batch
        )
        self._device_batch = None
        return device_batch

    def _reset_prefetch(self) -> None:
        if self._device_batch is not None:
            device_batch = (
                self._device_batch.result()
                if isinstance(self._device_batch, Future)
                else self._device_batch
            )
            device_batch.ready.synchronize()
            self._device_batch = None
        if self._next_host_batch is not None:
            self._next_host_batch.result()
            self._next_host_batch = None
        self._start_prefetch()

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "dflash_features",
            "source": self.source,
            "cursor": int(self.cursor),
            "samples_consumed": int(self.samples_consumed),
            "sample_count": self.manifest.sample_count,
            "data_parallel_rank": self.data_parallel_rank,
            "data_parallel_size": self.data_parallel_size,
            "target_intervals": [list(interval) for interval in self.target_intervals],
        }

    def load_state_dict(self, state: dict[str, Any] | None) -> None:
        if not state:
            return
        if state.get("kind") != "dflash_features":
            raise RuntimeError("checkpoint data kind does not match DFlash features")
        expected = {
            "source": self.source,
            "sample_count": self.manifest.sample_count,
            "data_parallel_rank": self.data_parallel_rank,
            "data_parallel_size": self.data_parallel_size,
            "target_intervals": [list(interval) for interval in self.target_intervals],
        }
        for name, value in expected.items():
            if state.get(name) != value:
                raise RuntimeError(
                    f"checkpoint DFlash data {name} does not match current run"
                )
        self.cursor = int(state.get("cursor", 0)) % self.manifest.sample_count
        self.samples_consumed = int(state.get("samples_consumed", 0))
        self._reset_prefetch()

    def to_log_dict(self) -> dict[str, Any]:
        return self.state_dict()


class DFlashSyntheticFeatureDataRuntime:
    """Deterministic, rank-local DFlash data for performance profiles.

    The target tensor is allocated once on its owning accelerator. This keeps
    the profiled model work identical in shape to offline training without
    materializing a sequence-wide verifier-feature file on every host process.
    """

    def __init__(
        self,
        *,
        batch_size: int,
        seq_len: int,
        target_feature_width: int,
        verifier_hidden_size: int,
        vocab_size: int,
        device: torch.device,
        dtype: torch.dtype,
        seed: int,
        runtime: Any | None,
        data_parallel_rank: int,
        data_parallel_size: int,
        require_teacher_features: bool,
    ) -> None:
        self.batch_size = int(batch_size)
        self.seq_len = int(seq_len)
        self.target_feature_width = int(target_feature_width)
        self.verifier_hidden_size = int(verifier_hidden_size)
        self.data_parallel_rank = int(data_parallel_rank)
        self.data_parallel_size = int(data_parallel_size)
        self.target_intervals = _target_intervals(self.seq_len, runtime)
        self.require_teacher_features = bool(require_teacher_features)
        self.cursor = 0
        self.samples_consumed = 0
        if min(
            self.batch_size,
            self.seq_len,
            self.target_feature_width,
            self.verifier_hidden_size,
            int(vocab_size),
        ) <= 0:
            raise ValueError("synthetic DFlash dimensions must be positive")
        generator = torch.Generator(device=device).manual_seed(int(seed))
        self.input_ids = torch.randint(
            0,
            int(vocab_size),
            (self.batch_size, self.seq_len),
            generator=generator,
            dtype=torch.int64,
            device=device,
        )
        self.loss_mask = torch.ones_like(self.input_ids, dtype=torch.bool)
        self.document_ids = torch.zeros_like(self.input_ids)
        self.position_ids = torch.arange(
            self.seq_len,
            dtype=torch.int64,
            device=device,
        ).expand(self.batch_size, -1)
        self.target_position_ids = _cat_token_intervals(
            self.position_ids,
            self.target_intervals,
        ).contiguous()
        local_tokens = int(self.target_position_ids.shape[1])
        self.target_hidden_states = torch.zeros(
            (self.batch_size, local_tokens, self.target_feature_width),
            dtype=dtype,
            device=device,
        )

    def next_batch(self) -> DFlashFeatureBatch:
        self.cursor += 1
        self.samples_consumed += self.batch_size
        return DFlashFeatureBatch(
            input_ids=self.input_ids,
            loss_mask=self.loss_mask,
            document_ids=self.document_ids,
            position_ids=self.position_ids,
            target_hidden_states=self.target_hidden_states,
            target_position_ids=self.target_position_ids,
            verifier_last_hidden_states=(
                DFlashSyntheticVerifierFeatureBatch(
                    hidden_size=self.verifier_hidden_size,
                    batch_size=self.batch_size,
                    dtype=self.target_hidden_states.dtype,
                )
                if self.require_teacher_features
                else None
            ),
        )

    def next_batches(self, count: int) -> list[DFlashFeatureBatch]:
        if int(count) <= 0:
            raise ValueError("batch count must be positive")
        return [self.next_batch() for _ in range(int(count))]

    def next_optimizer_step_batches(self, count: int) -> list[DFlashFeatureBatch]:
        """Implement the shared trainer's optimizer-step data contract."""

        return self.next_batches(count)

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "dflash_synthetic_features",
            "cursor": int(self.cursor),
            "samples_consumed": int(self.samples_consumed),
            "data_parallel_rank": self.data_parallel_rank,
            "data_parallel_size": self.data_parallel_size,
            "target_intervals": [list(value) for value in self.target_intervals],
        }

    def load_state_dict(self, state: dict[str, Any] | None) -> None:
        if not state:
            return
        expected = self.state_dict()
        if state.get("kind") != expected["kind"]:
            raise RuntimeError("checkpoint data kind does not match synthetic DFlash")
        for name in (
            "data_parallel_rank",
            "data_parallel_size",
            "target_intervals",
        ):
            if state.get(name) != expected[name]:
                raise RuntimeError(
                    f"checkpoint synthetic DFlash {name} does not match current run"
                )
        self.cursor = int(state.get("cursor", 0))
        self.samples_consumed = int(state.get("samples_consumed", 0))

    def to_log_dict(self) -> dict[str, Any]:
        return self.state_dict()


class _DFlashShardedRecordStore:
    """Memory-map at most two immutable feature records per rank."""

    def __init__(
        self,
        root: Path,
        manifest: DFlashFeatureManifest,
    ) -> None:
        self.root = root.resolve()
        self.manifest = manifest
        self.paths: list[Path] = []
        self.expected_bytes: list[int] = []
        self.expected_sha256: list[str] = []
        for record in manifest.records:
            relative = record.get("path")
            size = record.get("bytes")
            sha256 = record.get("sha256")
            if not isinstance(relative, str) or not relative:
                raise ValueError("DFlash sharded record path is missing")
            path = (self.root / relative).resolve()
            if self.root not in path.parents:
                raise ValueError("DFlash sharded record escapes its dataset root")
            if not path.is_file():
                raise FileNotFoundError(path)
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise ValueError("DFlash sharded record bytes must be positive")
            if path.stat().st_size != size:
                raise ValueError(f"DFlash sharded record size mismatch: {path}")
            if (
                not isinstance(sha256, str)
                or len(sha256) != 64
                or any(character not in "0123456789abcdef" for character in sha256)
            ):
                raise ValueError("DFlash sharded record sha256 is invalid")
            self.paths.append(path)
            self.expected_bytes.append(size)
            self.expected_sha256.append(sha256)
        self._cache: OrderedDict[int, dict[str, torch.Tensor]] = OrderedDict()
        self._verified: set[int] = set()
        self._lock = threading.Lock()

    def load(self, sample_index: int) -> dict[str, torch.Tensor]:
        sample_index = int(sample_index)
        if not 0 <= sample_index < len(self.paths):
            raise IndexError("DFlash sharded sample index is out of range")
        with self._lock:
            cached = self._cache.pop(sample_index, None)
            if cached is not None:
                self._cache[sample_index] = cached
                return cached
            path = self.paths[sample_index]
            if sample_index not in self._verified:
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest() != self.expected_sha256[sample_index]:
                    raise ValueError(f"DFlash sharded record hash mismatch: {path}")
                self._verified.add(sample_index)
            payload = torch.load(
                path,
                map_location="cpu",
                mmap=True,
                weights_only=True,
            )
            normalized = _normalize_sharded_feature_record(
                payload,
                manifest=self.manifest,
                path=path,
            )
            self._cache[sample_index] = normalized
            while len(self._cache) > 2:
                self._cache.popitem(last=False)
            return normalized


class _DFlashShardedTensorSource:
    """Tensor-shaped view over one key in a sharded record store."""

    def __init__(
        self,
        store: _DFlashShardedRecordStore,
        key: str,
        sample: torch.Tensor,
    ) -> None:
        self.store = store
        self.key = str(key)
        sample_shape = sample.shape
        if (
            store.manifest.storage_format == FEATURE_COMPACT_SHARDED_FORMAT
            and self.key in {"target_hidden_states", "verifier_last_hidden_states"}
        ):
            sample_shape = torch.Size(
                (store.manifest.sequence_length, int(sample.shape[-1]))
            )
        self.shape = torch.Size((store.manifest.sample_count, *sample_shape))
        self.dtype = sample.dtype
        self.device = torch.device("cpu")

    def __getitem__(self, index: Any) -> torch.Tensor:
        if isinstance(index, tuple):
            sample_index, *remainder = index
        else:
            sample_index, remainder = index, []
        if not isinstance(sample_index, int):
            raise TypeError("DFlash sharded tensors require scalar sample indexing")
        tensor = self.store.load(sample_index)[self.key]
        if (
            self.store.manifest.storage_format == FEATURE_COMPACT_SHARDED_FORMAT
            and self.key in {"target_hidden_states", "verifier_last_hidden_states"}
        ):
            sequence_length = int(self.store.manifest.sequence_length)
            if not remainder:
                output = torch.zeros(
                    (sequence_length, int(tensor.shape[-1])), dtype=tensor.dtype
                )
                output[: int(tensor.shape[0])].copy_(tensor)
                return output
            if len(remainder) != 1 or not isinstance(remainder[0], slice):
                raise TypeError(
                    "compact DFlash feature tensors support contiguous token slices"
                )
            start, stop, step = remainder[0].indices(sequence_length)
            if step != 1:
                raise ValueError(
                    "compact DFlash feature tensors require unit-stride token slices"
                )
            output = torch.zeros(
                (max(0, stop - start), int(tensor.shape[-1])), dtype=tensor.dtype
            )
            copied_stop = min(stop, int(tensor.shape[0]))
            if start < copied_stop:
                output[: copied_stop - start].copy_(tensor[start:copied_stop])
            return output
        return tensor[tuple(remainder)] if remainder else tensor


def _normalize_sharded_feature_record(
    payload: Any,
    *,
    manifest: DFlashFeatureManifest,
    path: Path,
) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise ValueError(f"DFlash sharded record is not a tensor mapping: {path}")
    required = {"input_ids", "loss_mask", "target_hidden_states"}
    missing = required - set(payload)
    if missing:
        raise ValueError(
            f"DFlash sharded record {path} is missing tensors: {sorted(missing)}"
        )
    normalized: dict[str, torch.Tensor] = {}
    for name, raw in payload.items():
        if not torch.is_tensor(raw):
            continue
        value = raw.detach().cpu()
        if name in {"input_ids", "loss_mask", "document_ids", "position_ids"}:
            if value.ndim == 2 and int(value.shape[0]) == 1:
                value = value.squeeze(0)
            if value.ndim != 1 or int(value.shape[0]) != manifest.sequence_length:
                raise ValueError(
                    f"DFlash sharded tensor {name!r} in {path} must have shape "
                    f"[{manifest.sequence_length}]"
                )
        elif name in {"target_hidden_states", "verifier_last_hidden_states"}:
            if value.ndim == 3 and int(value.shape[0]) == 1:
                value = value.squeeze(0)
            compact = manifest.storage_format == FEATURE_COMPACT_SHARDED_FORMAT
            valid_length = (
                value.ndim == 2
                and 0 < int(value.shape[0]) <= manifest.sequence_length
                if compact
                else value.ndim == 2
                and int(value.shape[0]) == manifest.sequence_length
            )
            if not valid_length:
                raise ValueError(
                    f"DFlash sharded tensor {name!r} in {path} must have shape "
                    + (
                        f"[1..{manifest.sequence_length}, hidden]"
                        if compact
                        else f"[{manifest.sequence_length}, hidden]"
                    )
                )
            if not value.dtype.is_floating_point:
                raise TypeError(
                    f"DFlash sharded tensor {name!r} in {path} must be floating point"
                )
        normalized[str(name)] = value
    if int(normalized["target_hidden_states"].shape[-1]) != (
        manifest.target_feature_width
    ):
        raise ValueError(f"DFlash sharded target feature width mismatch: {path}")
    if manifest.storage_format == FEATURE_COMPACT_SHARDED_FORMAT:
        documents = normalized.get("document_ids")
        if documents is None:
            raise ValueError(
                f"compact DFlash shard requires document_ids to recover real tokens: {path}"
            )
        valid_documents = documents.ge(0)
        real_token_count = int(valid_documents.sum().item())
        if (
            real_token_count <= 0
            or not bool(valid_documents[:real_token_count].all())
            or bool(valid_documents[real_token_count:].any())
        ):
            raise ValueError(
                f"compact DFlash shard real tokens must form one non-empty prefix: {path}"
            )
        for name in {"target_hidden_states", "verifier_last_hidden_states"}:
            value = normalized.get(name)
            if value is not None and int(value.shape[0]) != real_token_count:
                raise ValueError(
                    f"compact DFlash tensor {name!r} in {path} does not match "
                    f"the real token count {real_token_count}"
                )
    verifier = normalized.get("verifier_last_hidden_states")
    if verifier is not None and int(verifier.shape[-1]) != manifest.verifier_hidden_size:
        raise ValueError(f"DFlash sharded verifier hidden width mismatch: {path}")
    return normalized


def _load_sharded_feature_tensors(
    path: Path,
    *,
    manifest: DFlashFeatureManifest,
) -> dict[str, Any]:
    store = _DFlashShardedRecordStore(path, manifest)
    first = store.load(0)
    return {
        name: _DFlashShardedTensorSource(store, name, tensor)
        for name, tensor in first.items()
    }


def _load_feature_tensors(
    path: Path,
    *,
    manifest: DFlashFeatureManifest,
) -> dict[str, Any]:
    if path.is_dir():
        if manifest.storage_format not in {
            FEATURE_SHARDED_FORMAT,
            FEATURE_COMPACT_SHARDED_FORMAT,
        }:
            raise ValueError("DFlash feature directories require the sharded format")
        return _load_sharded_feature_tensors(path, manifest=manifest)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return dict(load_file(str(path), device="cpu"))
    if path.suffix == ".pt":
        payload = torch.load(
            path,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
        if not isinstance(payload, dict):
            raise ValueError("DFlash .pt feature dataset must contain a tensor mapping")
        return {
            str(name): value
            for name, value in payload.items()
            if torch.is_tensor(value)
        }
    raise ValueError("DFlash feature datasets must use .pt or .safetensors")


def _target_intervals(
    sequence_length: int,
    runtime: Any | None,
) -> tuple[tuple[int, int], ...]:
    if runtime is None or not bool(
        getattr(runtime, "uses_context_parallel_attention", False)
    ):
        return ((0, int(sequence_length)),)
    rank_intervals = clean_intervals_for_runtime(
        int(sequence_length),
        int(runtime.context_attention_size),
        runtime,
    )
    return tuple(
        (int(start), int(stop))
        for start, stop in rank_intervals[int(runtime.context_parallel_rank)]
    )


def _validate_token_intervals(
    intervals: tuple[tuple[int, int], ...],
    *,
    sequence_length: int,
) -> tuple[tuple[int, int], ...]:
    normalized = tuple((int(start), int(stop)) for start, stop in intervals)
    if not normalized:
        raise ValueError("DFlash target feature ownership cannot be empty")
    occupied: list[tuple[int, int]] = []
    for start, stop in normalized:
        if not 0 <= start < stop <= int(sequence_length):
            raise ValueError(
                f"invalid DFlash target feature interval {(start, stop)} for "
                f"sequence length {sequence_length}"
            )
        if any(
            start < prior_stop and prior_start < stop
            for prior_start, prior_stop in occupied
        ):
            raise ValueError("DFlash target feature ownership intervals overlap")
        occupied.append((start, stop))
    return normalized


def _cat_token_intervals(
    tensor: torch.Tensor,
    intervals: tuple[tuple[int, int], ...],
) -> torch.Tensor:
    shards = [tensor[:, start:stop] for start, stop in intervals]
    return shards[0] if len(shards) == 1 else torch.cat(shards, dim=1)


def _sample_indices(
    *,
    cursor: int,
    batch_size: int,
    sample_count: int,
    data_parallel_rank: int,
    data_parallel_size: int,
) -> torch.Tensor:
    start = (
        int(cursor) * int(data_parallel_size)
        + int(data_parallel_rank) * int(batch_size)
    )
    return (
        torch.arange(int(batch_size), dtype=torch.long) + start
    ) % int(sample_count)


def _next_cursor(cursor: int, *, batch_size: int, sample_count: int) -> int:
    return (int(cursor) + int(batch_size)) % int(sample_count)


def _copy_samples_to_host(
    tensor: torch.Tensor,
    sample_indices: torch.Tensor,
    *,
    pin_memory: bool,
) -> torch.Tensor:
    output = torch.empty(
        (int(sample_indices.numel()), *tensor.shape[1:]),
        dtype=tensor.dtype,
        device="cpu",
        pin_memory=bool(pin_memory),
    )
    for output_index, sample_index in enumerate(sample_indices.tolist()):
        output[output_index].copy_(tensor[int(sample_index)])
    return output


def _copy_samples_and_token_intervals_to_host(
    tensor: torch.Tensor,
    sample_indices: torch.Tensor,
    intervals: tuple[tuple[int, int], ...],
    *,
    pin_memory: bool,
) -> torch.Tensor:
    token_count = sum(int(stop) - int(start) for start, stop in intervals)
    output = torch.empty(
        (int(sample_indices.numel()), token_count, *tensor.shape[2:]),
        dtype=tensor.dtype,
        device="cpu",
        pin_memory=bool(pin_memory),
    )
    for output_index, sample_index in enumerate(sample_indices.tolist()):
        output_offset = 0
        for start, stop in intervals:
            interval_length = int(stop) - int(start)
            output[
                output_index,
                output_offset : output_offset + interval_length,
            ].copy_(tensor[int(sample_index), int(start) : int(stop)])
            output_offset += interval_length
    return output


def _normalize_feature_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    seq_len: int,
) -> torch.Tensor:
    value = tensor.detach().cpu()
    if name in {"input_ids", "loss_mask", "document_ids", "position_ids"}:
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.ndim != 2 or int(value.shape[1]) != int(seq_len):
            raise ValueError(
                f"DFlash feature tensor {name!r} must have shape "
                f"[samples, {seq_len}]"
            )
    elif name in {"target_hidden_states", "verifier_last_hidden_states"}:
        if value.ndim == 2:
            value = value.unsqueeze(0)
        if value.ndim != 3 or int(value.shape[1]) != int(seq_len):
            raise ValueError(
                f"DFlash feature tensor {name!r} must have shape "
                f"[samples, {seq_len}, hidden]"
            )
        if not value.dtype.is_floating_point:
            raise TypeError(f"DFlash feature tensor {name!r} must be floating point")
    return value.contiguous()


__all__ = [
    "DFlashFeatureBatch",
    "DFlashFeatureDataRuntime",
    "DFlashFeatureManifest",
    "DFlashSyntheticFeatureDataRuntime",
    "DFlashSyntheticVerifierFeatureBatch",
    "DFlashVerifierFeatureBatch",
    "FEATURE_COMPACT_SHARDED_FORMAT",
    "FEATURE_FORMAT",
    "FEATURE_SHARDED_FORMAT",
]
