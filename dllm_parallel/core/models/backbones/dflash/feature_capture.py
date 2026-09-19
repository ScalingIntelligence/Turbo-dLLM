# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Validated conversion from indexed SFT tokens to sharded DFlash features."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Protocol, Sequence

import torch

from dllm_parallel.core.data import (
    INDEXED_SUPERVISED_COLUMNS,
    INDEXED_SUPERVISED_FORMAT,
    INDEXED_SUPERVISED_VERSION,
)
from dllm_parallel.core.models.backbones.dflash.data import (
    FEATURE_COMPACT_SHARDED_FORMAT,
    FEATURE_SHARDED_FORMAT,
)
from dllm_parallel.data.indexed import SUPPORTED_INDEXED_SEMANTICS


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _validated_source_file(
    root: Path,
    files: Mapping[str, Any],
    name: str,
    *,
    dtype: str,
    element_size: int,
) -> tuple[Path, int, dict[str, Any]]:
    raw = files.get(name)
    path = root / name
    if not isinstance(raw, dict) or not path.is_file():
        raise ValueError(f"indexed capture source is missing {name}")
    if raw.get("dtype") != dtype:
        raise ValueError(f"indexed capture source {name} has the wrong dtype")
    elements = raw.get("elements")
    byte_count = raw.get("bytes")
    if (
        isinstance(elements, bool)
        or not isinstance(elements, int)
        or elements < 0
        or isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count != elements * int(element_size)
        or path.stat().st_size != byte_count
    ):
        raise ValueError(f"indexed capture source {name} has invalid dimensions")
    expected_digest = _require_sha256(raw.get("sha256"), field=f"{name}.sha256")
    if _sha256_file(path) != expected_digest:
        raise ValueError(f"indexed capture source {name} failed SHA-256 validation")
    return path, elements, dict(raw)


@dataclass(frozen=True)
class IndexedDFlashCaptureSource:
    """Immutable, memory-mapped full-trajectory input to verifier capture."""

    root: Path
    manifest: dict[str, Any]
    tokens: torch.Tensor
    loss_masks: torch.Tensor
    index: torch.Tensor
    dataset_fingerprint: str
    pad_token_id: int

    @property
    def sample_count(self) -> int:
        return int(self.index.shape[0])

    @classmethod
    def open(cls, root: str | Path) -> "IndexedDFlashCaptureSource":
        source_root = Path(root).expanduser().resolve()
        manifest_path = source_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("indexed capture manifest must be an object")
        if payload.get("format") != INDEXED_SUPERVISED_FORMAT:
            raise ValueError("unsupported indexed capture source format")
        if int(payload.get("version", -1)) != INDEXED_SUPERVISED_VERSION:
            raise ValueError("unsupported indexed capture source version")

        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("indexed capture manifest is missing metadata")
        if metadata.get("semantics") not in SUPPORTED_INDEXED_SEMANTICS:
            raise ValueError(
                "indexed capture source has unsupported supervision semantics"
            )
        if metadata.get("loss_mask_encoding") != "uint8_per_token_v1":
            raise ValueError("indexed capture source requires per-token loss masks")
        if metadata.get("sequence_layout") != "record_aligned":
            raise ValueError("indexed capture source must preserve record boundaries")
        pad_token_id = metadata.get("pad_token_id")
        if pad_token_id is None:
            pad_token_id = metadata.get("eos_token_id")
        if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int):
            raise ValueError(
                "indexed capture source requires an integer tokenizer pad or EOS token ID"
            )
        dataset_fingerprint = metadata.get("dataset_fingerprint")
        if not isinstance(dataset_fingerprint, str) or not dataset_fingerprint:
            raise ValueError("indexed capture source requires a dataset fingerprint")

        files = payload.get("files")
        if not isinstance(files, dict):
            raise ValueError("indexed capture manifest is missing file metadata")
        token_path, token_count, _ = _validated_source_file(
            source_root,
            files,
            "tokens.i32",
            dtype="int32",
            element_size=4,
        )
        mask_path, mask_count, _ = _validated_source_file(
            source_root,
            files,
            "loss_mask.u8",
            dtype="uint8",
            element_size=1,
        )
        index_path, index_elements, index_metadata = _validated_source_file(
            source_root,
            files,
            "index.i64",
            dtype="int64",
            element_size=8,
        )
        if tuple(index_metadata.get("columns", ())) != INDEXED_SUPERVISED_COLUMNS:
            raise ValueError("indexed capture record schema is incompatible")
        if (
            mask_count != token_count
            or int(payload.get("token_count", -1)) != token_count
        ):
            raise ValueError("indexed capture token and loss-mask counts disagree")
        sample_count = payload.get("sample_count")
        if (
            isinstance(sample_count, bool)
            or not isinstance(sample_count, int)
            or sample_count <= 0
            or index_elements != sample_count * len(INDEXED_SUPERVISED_COLUMNS)
        ):
            raise ValueError("indexed capture sample count is inconsistent")

        tokens = torch.from_file(str(token_path), dtype=torch.int32, size=token_count)
        loss_masks = torch.from_file(str(mask_path), dtype=torch.uint8, size=mask_count)
        index = torch.from_file(
            str(index_path), dtype=torch.int64, size=index_elements
        ).reshape(sample_count, len(INDEXED_SUPERVISED_COLUMNS))
        seen_sample_ids: set[int] = set()
        for row_number, row in enumerate(index.tolist()):
            offset, length, supervision_start, sample_id = map(int, row)
            if offset < 0 or length <= 0 or offset + length > token_count:
                raise ValueError(
                    f"indexed capture record {row_number} is out of bounds"
                )
            if not 0 <= supervision_start <= length:
                raise ValueError(
                    f"indexed capture record {row_number} has an invalid supervision start"
                )
            if sample_id in seen_sample_ids:
                raise ValueError("indexed capture sample IDs must be unique")
            seen_sample_ids.add(sample_id)
            record_mask = loss_masks.narrow(0, offset, length)
            if not bool(record_mask.to(torch.bool).any()):
                raise ValueError(
                    f"indexed capture record {row_number} has no supervised tokens"
                )
        return cls(
            root=source_root,
            manifest=payload,
            tokens=tokens,
            loss_masks=loss_masks,
            index=index,
            dataset_fingerprint=dataset_fingerprint,
            pad_token_id=pad_token_id,
        )

    def record(self, sample_index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        position = int(sample_index)
        if not 0 <= position < self.sample_count:
            raise IndexError("indexed capture sample index is out of range")
        offset, length, _, sample_id = map(int, self.index[position].tolist())
        input_ids = self.tokens.narrow(0, offset, length).to(torch.int64).clone()
        loss_mask = self.loss_masks.narrow(0, offset, length).to(torch.bool).clone()
        return input_ids.contiguous(), loss_mask.contiguous(), sample_id


@dataclass(frozen=True)
class DFlashCaptureContract:
    """Layer and block metadata derived from a published DFlash2 draft."""

    block_size: int
    capture_layer_ids: tuple[int, ...]
    target_layer_ids: tuple[int, ...]
    model_type: str


def resolve_dflash_capture_contract(
    draft_model: str | Path,
    *,
    revision: str | None = None,
) -> DFlashCaptureContract:
    """Resolve capture layers from a local or Hugging Face DFlash2 config."""

    candidate = Path(draft_model).expanduser()
    if candidate.is_dir():
        config_path = candidate / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    elif candidate.is_file():
        raw = json.loads(candidate.read_text(encoding="utf-8"))
    else:
        from transformers import AutoConfig

        raw = AutoConfig.from_pretrained(
            str(draft_model),
            revision=revision,
            trust_remote_code=False,
        ).to_dict()
    if not isinstance(raw, Mapping):
        raise ValueError("DFlash2 draft config must be a JSON object")
    method = raw.get("dflash_config")
    if not isinstance(method, Mapping):
        raise ValueError("DFlash2 draft config is missing dflash_config")
    block_size = method.get("block_size", raw.get("block_size"))
    layers = method.get("target_layer_ids")
    if (
        isinstance(block_size, bool)
        or not isinstance(block_size, int)
        or block_size <= 0
    ):
        raise ValueError("DFlash2 draft config has an invalid block_size")
    if (
        not isinstance(layers, Sequence)
        or isinstance(layers, (str, bytes))
        or not layers
        or any(
            isinstance(value, bool) or not isinstance(value, int) for value in layers
        )
    ):
        raise ValueError("DFlash2 draft config has invalid target_layer_ids")
    capture_layers = tuple(int(value) for value in layers)
    if len(set(capture_layers)) != len(capture_layers) or any(
        value < 0 for value in capture_layers
    ):
        raise ValueError("DFlash2 capture layer IDs must be unique and non-negative")
    return DFlashCaptureContract(
        block_size=int(block_size),
        capture_layer_ids=capture_layers,
        target_layer_ids=tuple(value + 1 for value in capture_layers),
        model_type=str(raw.get("model_type", "")),
    )


def build_padded_dflash_feature_record(
    *,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    target_hidden_states: torch.Tensor,
    sequence_length: int,
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    """Build one fixed-length native DFlash shard without changing token alignment."""

    if input_ids.ndim != 1 or loss_mask.ndim != 1:
        raise ValueError("DFlash capture token inputs must be one-dimensional")
    captured_length = int(input_ids.numel())
    if captured_length <= 0 or int(loss_mask.numel()) != captured_length:
        raise ValueError("DFlash capture token and loss-mask lengths disagree")
    fixed_length = int(sequence_length)
    if captured_length > fixed_length:
        raise ValueError("DFlash capture record exceeds the configured sequence length")
    if target_hidden_states.ndim != 2:
        raise ValueError(
            "captured DFlash target features must have shape [tokens, width]"
        )
    if int(target_hidden_states.shape[0]) != captured_length:
        raise ValueError("captured DFlash features do not align with the input tokens")
    if int(target_hidden_states.shape[1]) <= 0:
        raise ValueError("captured DFlash feature width must be positive")
    if not target_hidden_states.dtype.is_floating_point:
        raise TypeError("captured DFlash target features must be floating point")

    ids = torch.full((fixed_length,), int(pad_token_id), dtype=torch.int64)
    masks = torch.zeros((fixed_length,), dtype=torch.bool)
    documents = torch.full((fixed_length,), -1, dtype=torch.int64)
    positions = torch.arange(fixed_length, dtype=torch.int64)
    features = torch.zeros(
        (fixed_length, int(target_hidden_states.shape[1])),
        dtype=target_hidden_states.dtype,
    )
    ids[:captured_length].copy_(input_ids.detach().to(device="cpu", dtype=torch.int64))
    masks[:captured_length].copy_(loss_mask.detach().to(device="cpu", dtype=torch.bool))
    documents[:captured_length] = 0
    features[:captured_length].copy_(target_hidden_states.detach().to(device="cpu"))
    return {
        "input_ids": ids.contiguous(),
        "loss_mask": masks.contiguous(),
        "document_ids": documents.contiguous(),
        "position_ids": positions.contiguous(),
        "target_hidden_states": features.contiguous(),
    }


def build_compact_dflash_feature_record(
    *,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    target_hidden_states: torch.Tensor,
    sequence_length: int,
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    """Pad inexpensive token metadata but store verifier features only once."""

    if input_ids.ndim != 1 or loss_mask.ndim != 1:
        raise ValueError("DFlash capture token inputs must be one-dimensional")
    captured_length = int(input_ids.numel())
    if captured_length <= 0 or int(loss_mask.numel()) != captured_length:
        raise ValueError("DFlash capture token and loss-mask lengths disagree")
    fixed_length = int(sequence_length)
    if captured_length > fixed_length:
        raise ValueError("DFlash capture record exceeds the configured sequence length")
    if target_hidden_states.ndim != 2:
        raise ValueError(
            "captured DFlash target features must have shape [tokens, width]"
        )
    if int(target_hidden_states.shape[0]) != captured_length:
        raise ValueError("captured DFlash features do not align with the input tokens")
    if int(target_hidden_states.shape[1]) <= 0:
        raise ValueError("captured DFlash feature width must be positive")
    if not target_hidden_states.dtype.is_floating_point:
        raise TypeError("captured DFlash target features must be floating point")

    ids = torch.full((fixed_length,), int(pad_token_id), dtype=torch.int64)
    masks = torch.zeros((fixed_length,), dtype=torch.bool)
    documents = torch.full((fixed_length,), -1, dtype=torch.int64)
    positions = torch.arange(fixed_length, dtype=torch.int64)
    ids[:captured_length].copy_(input_ids.detach().to(device="cpu", dtype=torch.int64))
    masks[:captured_length].copy_(loss_mask.detach().to(device="cpu", dtype=torch.bool))
    documents[:captured_length] = 0
    features = target_hidden_states.detach().to(device="cpu").contiguous()
    return {
        "input_ids": ids.contiguous(),
        "loss_mask": masks.contiguous(),
        "document_ids": documents.contiguous(),
        "position_ids": positions.contiguous(),
        "target_hidden_states": features,
    }


def write_dflash_feature_record(
    output_dir: str | Path,
    *,
    sample_index: int,
    sample_id: int,
    payload: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Atomically write one immutable feature record and return its manifest entry."""

    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    records_dir = root / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    position = int(sample_index)
    if position < 0:
        raise ValueError("DFlash feature sample index must be non-negative")
    if not payload or not all(torch.is_tensor(value) for value in payload.values()):
        raise TypeError("DFlash feature record must contain only tensors")
    relative_path = Path("records") / f"record_{position:05d}.pt"
    destination = root / relative_path
    if destination.exists():
        raise FileExistsError(destination)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=records_dir
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "sample_index": position,
        "sample_id": int(sample_id),
        "path": relative_path.as_posix(),
        "bytes": destination.stat().st_size,
        "sha256": _sha256_file(destination),
    }


def finalize_dflash_feature_manifest(
    output_dir: str | Path,
    *,
    verifier_id: str,
    verifier_revision: str | None,
    target_layer_ids: Sequence[int],
    sequence_length: int,
    target_feature_width: int,
    verifier_hidden_size: int | None,
    source: IndexedDFlashCaptureSource,
    records: Sequence[Mapping[str, Any]],
    compact_target_features: bool = False,
) -> dict[str, Any]:
    """Bind complete ordered shards to their exact indexed training source."""

    root = Path(output_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    if len(records) != source.sample_count:
        raise ValueError("DFlash feature record count does not match the source")
    normalized_records: list[dict[str, Any]] = []
    observed_ids: set[int] = set()
    for expected_index, raw in enumerate(records):
        record = dict(raw)
        if int(record.get("sample_index", -1)) != expected_index:
            raise ValueError("DFlash feature records are not in source order")
        expected_id = int(source.index[expected_index, 3].item())
        if int(record.get("sample_id", -1)) != expected_id:
            raise ValueError("DFlash feature record sample ID does not match source")
        if expected_id in observed_ids:
            raise ValueError("DFlash feature record sample IDs must be unique")
        observed_ids.add(expected_id)
        relative = record.get("path")
        if not isinstance(relative, str) or not relative:
            raise ValueError("DFlash feature record path is missing")
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError("DFlash feature record path is invalid")
        if path.stat().st_size != int(record.get("bytes", -1)):
            raise ValueError("DFlash feature record size changed before finalization")
        expected_digest = _require_sha256(record.get("sha256"), field="record.sha256")
        if _sha256_file(path) != expected_digest:
            raise ValueError("DFlash feature record changed before finalization")
        normalized_records.append(record)

    layers = tuple(int(value) for value in target_layer_ids)
    if (
        not layers
        or len(set(layers)) != len(layers)
        or any(value < 0 for value in layers)
    ):
        raise ValueError("DFlash target layer IDs must be unique non-negative values")
    if int(sequence_length) <= 0 or int(target_feature_width) <= 0:
        raise ValueError("DFlash feature dimensions must be positive")
    if not str(verifier_id):
        raise ValueError("DFlash verifier ID must not be empty")
    if verifier_hidden_size is not None and int(verifier_hidden_size) <= 0:
        raise ValueError("DFlash verifier hidden size must be positive when present")

    source_manifest_path = source.root / "manifest.json"
    payload = {
        "format": (
            FEATURE_COMPACT_SHARDED_FORMAT
            if compact_target_features
            else FEATURE_SHARDED_FORMAT
        ),
        "verifier_id": str(verifier_id),
        "verifier_revision": (
            str(verifier_revision) if verifier_revision is not None else None
        ),
        "target_layer_ids": list(layers),
        "sequence_length": int(sequence_length),
        "sample_count": source.sample_count,
        "target_feature_width": int(target_feature_width),
        "verifier_hidden_size": (
            int(verifier_hidden_size) if verifier_hidden_size is not None else None
        ),
        "source_dataset_fingerprint": source.dataset_fingerprint,
        "source_manifest_sha256": _sha256_file(source_manifest_path),
        "source_files": {
            name: {
                "bytes": int(metadata["bytes"]),
                "sha256": str(metadata["sha256"]),
            }
            for name, metadata in sorted(source.manifest["files"].items())
            if name in {"tokens.i32", "loss_mask.u8", "index.i64"}
        },
        "records": normalized_records,
    }
    root.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=".manifest.", suffix=".tmp", dir=root
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, manifest_path)
    finally:
        temporary.unlink(missing_ok=True)
    return payload


class DFlashCaptureBackend(Protocol):
    """Minimal verifier interface used by the provider-independent frontend."""

    def set_capture_layers(
        self, layer_ids: Sequence[int], *, capture_method: str
    ) -> None: ...

    def capture_rows(
        self, input_ids: list[list[int]]
    ) -> tuple[Sequence[torch.Tensor], Sequence[torch.Tensor]]: ...


def _capture_target_rows(
    backend: DFlashCaptureBackend,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    captured_rows, _last_rows = backend.capture_rows([input_ids.tolist()])
    if len(captured_rows) != 1:
        raise RuntimeError("DFlash verifier capture must return exactly one row")
    captured = captured_rows[0]
    if not torch.is_tensor(captured) or captured.ndim != 2:
        raise RuntimeError("DFlash verifier capture must have shape [tokens, width]")
    if int(captured.shape[0]) != int(input_ids.numel()) or int(captured.shape[1]) <= 0:
        raise RuntimeError("DFlash verifier features do not align with source tokens")
    if not captured.dtype.is_floating_point:
        raise TypeError("DFlash verifier features must be floating point")
    for start in range(0, int(captured.shape[0]), 4096):
        if not bool(torch.isfinite(captured[start : start + 4096]).all()):
            raise RuntimeError("DFlash verifier features contain non-finite values")
    return captured


def capture_dflash_features(
    *,
    source: str | Path,
    output: str | Path,
    capture_backend: DFlashCaptureBackend,
    capture_layer_ids: Sequence[int],
    target_layer_ids: Sequence[int],
    verifier_id: str,
    verifier_revision: str | None,
    sequence_length: int,
) -> dict[str, Any]:
    """Capture one complete feature artifact with any compatible backend.

    This function is intentionally independent of SGLang, SpecForge, cloud
    providers, and the training runtime. Distributed frontends may shard the
    same record-building primitives while preserving this artifact contract.
    """

    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if int(sequence_length) <= 0:
        raise ValueError("DFlash capture sequence_length must be positive")
    published_layers = tuple(int(value) for value in capture_layer_ids)
    canonical_layers = tuple(int(value) for value in target_layer_ids)
    if (
        not published_layers
        or len(published_layers) != len(canonical_layers)
        or any(
            target != source_layer + 1
            for source_layer, target in zip(published_layers, canonical_layers)
        )
    ):
        raise ValueError(
            "DFlash target layer IDs must be the canonical +1 capture layer IDs"
        )
    capture_source = IndexedDFlashCaptureSource.open(source)
    if int(capture_source.index[:, 1].max().item()) > int(sequence_length):
        raise ValueError("DFlash source contains a record longer than sequence_length")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_parent = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.capture-", dir=destination.parent)
    )
    staging = staging_parent / "artifact"
    staging.mkdir()
    capture_backend.set_capture_layers(list(published_layers), capture_method="dflash")
    records: list[dict[str, Any]] = []
    feature_width: int | None = None
    try:
        for sample_index in range(capture_source.sample_count):
            input_ids, loss_mask, sample_id = capture_source.record(sample_index)
            captured = _capture_target_rows(capture_backend, input_ids)
            observed_width = int(captured.shape[1])
            if feature_width is None:
                feature_width = observed_width
            elif observed_width != feature_width:
                raise RuntimeError(
                    "DFlash verifier feature width changed between records"
                )
            payload = build_compact_dflash_feature_record(
                input_ids=input_ids,
                loss_mask=loss_mask,
                target_hidden_states=captured,
                sequence_length=int(sequence_length),
                pad_token_id=capture_source.pad_token_id,
            )
            records.append(
                write_dflash_feature_record(
                    staging,
                    sample_index=sample_index,
                    sample_id=sample_id,
                    payload=payload,
                )
            )
        if feature_width is None:
            raise RuntimeError("DFlash capture source contains no records")
        manifest = finalize_dflash_feature_manifest(
            staging,
            verifier_id=verifier_id,
            verifier_revision=verifier_revision,
            target_layer_ids=canonical_layers,
            sequence_length=int(sequence_length),
            target_feature_width=feature_width,
            verifier_hidden_size=None,
            source=capture_source,
            records=records,
            compact_target_features=True,
        )
        os.replace(staging, destination)
        return manifest
    finally:
        shutil.rmtree(staging_parent, ignore_errors=True)


def capture_dflash_features_with_specforge(
    *,
    source: str | Path,
    output: str | Path,
    draft_model: str | Path,
    draft_revision: str | None,
    verifier_model: str,
    verifier_revision: str,
    sequence_length: int,
    tensor_parallel_size: int = 1,
    attention_backend: str = "flashinfer",
    gpu_memory_utilization: float = 0.82,
    distributed_timeout_minutes: int = 120,
    trust_remote_code: bool = False,
) -> dict[str, Any] | None:
    """Capture a sharded artifact with SpecForge's local SGLang backend.

    Run this entrypoint under ``torchrun``. Tensor-parallel ranks cooperate on
    one verifier replica; orthogonal data-parallel groups capture disjoint
    records. Only rank zero returns the completed manifest.
    """

    try:
        from specforge.distributed import (
            destroy_distributed,
            get_dp_group,
            get_tp_group,
            init_distributed,
        )
        from specforge.offline_capture import load_offline_capture
    except ImportError as error:
        raise RuntimeError(
            "DFlash2 feature capture requires SpecForge in this separate capture "
            "environment on Linux with Python 3.11-3.13; install it with "
            '`python -m pip install "turbo-dllm[capture]"`'
        ) from error
    import torch.distributed as dist

    if int(sequence_length) <= 0 or int(tensor_parallel_size) <= 0:
        raise ValueError("DFlash2 capture sizes must be positive")
    if not 0.0 < float(gpu_memory_utilization) < 1.0:
        raise ValueError("gpu_memory_utilization must be in (0, 1)")
    contract = resolve_dflash_capture_contract(draft_model, revision=draft_revision)
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    init_distributed(
        timeout=int(distributed_timeout_minutes),
        tp_size=int(tensor_parallel_size),
    )
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    staging_parent: Path | None = None
    try:
        if world_size % int(tensor_parallel_size):
            raise ValueError("world size must be divisible by tensor_parallel_size")
        dp_group = get_dp_group()
        tp_group = get_tp_group()
        dp_rank = dist.get_rank(dp_group)
        dp_size = dist.get_world_size(dp_group)
        tp_rank = dist.get_rank(tp_group)
        capture_source = IndexedDFlashCaptureSource.open(source)
        if capture_source.sample_count < dp_size:
            raise ValueError("capture data-parallel size exceeds the sample count")
        if int(capture_source.index[:, 1].max().item()) > int(sequence_length):
            raise ValueError(
                "DFlash source contains a record longer than sequence_length"
            )

        staging_payload: list[str | None] = [None]
        if global_rank == 0:
            destination.parent.mkdir(parents=True, exist_ok=True)
            staging_parent = Path(
                tempfile.mkdtemp(
                    prefix=f".{destination.name}.capture-", dir=destination.parent
                )
            )
            staging = staging_parent / "artifact"
            staging.mkdir()
            staging_payload[0] = str(staging)
        dist.broadcast_object_list(staging_payload, src=0)
        if staging_payload[0] is None:
            raise RuntimeError("rank zero did not publish a capture staging path")
        staging = Path(staging_payload[0])

        target = load_offline_capture(
            str(verifier_model),
            revision=str(verifier_revision),
            torch_dtype=torch.bfloat16,
            trust_remote_code=bool(trust_remote_code),
            attention_backend=str(attention_backend),
            mem_fraction_static=float(gpu_memory_utilization),
            context_length=int(sequence_length),
            max_running_requests=1,
            max_total_tokens=int(sequence_length),
            disable_radix_cache=True,
        )
        target.set_capture_layers(
            list(contract.capture_layer_ids), capture_method="dflash"
        )

        local_records: list[dict[str, Any]] = []
        widths: set[int] = set()

        def capture_one(sample_index: int) -> None:
            input_ids, loss_mask, sample_id = capture_source.record(sample_index)
            captured = _capture_target_rows(target, input_ids)
            widths.add(int(captured.shape[1]))
            if tp_rank == 0:
                payload = build_compact_dflash_feature_record(
                    input_ids=input_ids,
                    loss_mask=loss_mask,
                    target_hidden_states=captured,
                    sequence_length=int(sequence_length),
                    pad_token_id=capture_source.pad_token_id,
                )
                local_records.append(
                    write_dflash_feature_record(
                        staging,
                        sample_index=sample_index,
                        sample_id=sample_id,
                        payload=payload,
                    )
                )
            del captured

        longest = int(torch.argmax(capture_source.index[:, 1]).item())
        if dp_rank == longest % dp_size:
            capture_one(longest)
        dist.barrier()
        for sample_index in range(dp_rank, capture_source.sample_count, dp_size):
            if sample_index != longest:
                capture_one(sample_index)
        dist.barrier()

        gathered: list[dict[str, Any] | None] = [None] * world_size
        dist.all_gather_object(
            gathered,
            {"records": local_records, "widths": sorted(widths)},
        )
        manifest = None
        if global_rank == 0:
            all_records = [
                record
                for rank_payload in gathered
                if rank_payload is not None
                for record in rank_payload["records"]
            ]
            all_records.sort(key=lambda record: int(record["sample_index"]))
            all_widths = {
                int(width)
                for rank_payload in gathered
                if rank_payload is not None
                for width in rank_payload["widths"]
            }
            if len(all_widths) != 1:
                raise RuntimeError(
                    "DFlash verifier feature width differs across capture ranks"
                )
            manifest = finalize_dflash_feature_manifest(
                staging,
                verifier_id=str(verifier_model),
                verifier_revision=str(verifier_revision),
                target_layer_ids=contract.target_layer_ids,
                sequence_length=int(sequence_length),
                target_feature_width=all_widths.pop(),
                verifier_hidden_size=None,
                source=capture_source,
                records=all_records,
                compact_target_features=True,
            )
            os.replace(staging, destination)
        dist.barrier()
        return manifest
    finally:
        if global_rank == 0 and staging_parent is not None:
            shutil.rmtree(staging_parent, ignore_errors=True)
        destroy_distributed()


__all__ = [
    "DFlashCaptureBackend",
    "DFlashCaptureContract",
    "IndexedDFlashCaptureSource",
    "build_compact_dflash_feature_record",
    "build_padded_dflash_feature_record",
    "capture_dflash_features",
    "capture_dflash_features_with_specforge",
    "finalize_dflash_feature_manifest",
    "resolve_dflash_capture_contract",
    "write_dflash_feature_record",
]
