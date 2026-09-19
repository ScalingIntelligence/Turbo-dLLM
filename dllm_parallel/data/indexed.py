# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Versioned prepared-data artifact writers, readers, and validators."""

from __future__ import annotations

import array
import copy
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from dllm_parallel.core.data import (
    INDEXED_SUPERVISED_COLUMNS,
    INDEXED_SUPERVISED_FORMAT,
    INDEXED_SUPERVISED_VERSION,
    PACKED_TOKEN_FORMAT,
    PACKED_TOKEN_VERSION,
)
from dllm_parallel.data.packing import (
    PackedTokenAccounting,
    iter_packed_segments,
    prepare_indexed_record,
)
from dllm_parallel.data.provenance import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from dllm_parallel.data.schemas import PackingSpec
from dllm_parallel.data.tokenization import TokenizedRecord

SUPPORTED_INDEXED_SEMANTICS = frozenset(
    {
        "assistant_only_sft",
        "completion_only_sft",
        "full_sequence",
        "supervised_tokens",
    }
)


def write_packed_artifact(
    root: str | Path,
    records: Iterable[TokenizedRecord],
    *,
    packing: PackingSpec,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Stream full-sequence records into a packed int32 token artifact."""

    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    token_path = root / "tokens.i32"
    accounting = PackedTokenAccounting()
    with token_path.open("wb") as handle:
        for segment in iter_packed_segments(
            records,
            packing,
            accounting=accounting,
        ):
            _write_i32(handle, segment)
    if accounting.record_count <= 0 or accounting.stored_tokens <= 0:
        raise ValueError("prepared packed dataset contains no tokens")
    files = {
        "tokens.i32": _file_metadata(
            token_path,
            "int32",
            accounting.stored_tokens,
        )
    }
    stats = {
        "records": accounting.record_count,
        "logical_chunks": accounting.logical_chunks,
        "separator_tokens": accounting.separator_tokens,
        "split_records": accounting.split_records,
        "source_tokens": accounting.source_tokens,
        "stored_tokens": accounting.stored_tokens,
        "truncated_tokens": accounting.truncated_tokens,
    }
    manifest = _manifest(
        format_name=PACKED_TOKEN_FORMAT,
        version=PACKED_TOKEN_VERSION,
        metadata=dict(metadata),
        files=files,
        stats=stats,
        sample_count=accounting.record_count,
        token_count=accounting.stored_tokens,
    )
    _write_manifest(root, manifest)
    return manifest


def write_indexed_artifact(
    root: str | Path,
    records: Iterable[TokenizedRecord],
    *,
    packing: PackingSpec,
    metadata: Mapping[str, Any],
    include_groups: bool,
) -> dict[str, Any]:
    """Stream record-level tokens and masks into the indexed runtime format."""

    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    token_path = root / "tokens.i32"
    index_path = root / "index.i64"
    mask_path = root / "loss_mask.u8"
    group_path = root / "groups.i64"
    sample_count = 0
    token_count = 0
    source_tokens = 0
    supervised_tokens = 0
    truncated_tokens = 0
    padding_tokens = 0
    lengths: Counter[int] = Counter()
    token_ids_seen: set[int] = set()
    all_full_supervision = True
    all_contiguous_suffix = True
    with (
        token_path.open("wb") as token_handle,
        index_path.open("wb") as index_handle,
        mask_path.open("wb") as mask_handle,
    ):
        group_handle = group_path.open("wb") if include_groups else None
        try:
            for position, record in enumerate(records):
                source_tokens += len(record.tokens)
                packed = prepare_indexed_record(record, packing)
                prepared = packed.record
                assert prepared.loss_mask is not None
                sample_id = _stable_int_id(
                    position if prepared.sample_id is None else prepared.sample_id
                )
                if include_groups:
                    if prepared.group_id is None:
                        raise ValueError(
                            "records.group_id_field requires a group ID on every record"
                        )
                    assert group_handle is not None
                    _write_i64(group_handle, [_stable_int_id(prepared.group_id)])
                supervision_start = prepared.loss_mask.index(True)
                all_full_supervision = all_full_supervision and all(prepared.loss_mask)
                all_contiguous_suffix = (
                    all_contiguous_suffix
                    and all(prepared.loss_mask[supervision_start:])
                    and not any(prepared.loss_mask[:supervision_start])
                )
                _write_i32(token_handle, prepared.tokens)
                _write_i64(
                    index_handle,
                    [token_count, len(prepared.tokens), supervision_start, sample_id],
                )
                mask_handle.write(bytes(int(value) for value in prepared.loss_mask))
                token_count += len(prepared.tokens)
                supervised_tokens += sum(prepared.loss_mask)
                truncated_tokens += packed.truncated_tokens
                padding_tokens += packed.padding_tokens
                lengths[len(prepared.tokens)] += 1
                token_ids_seen.update(prepared.tokens)
                sample_count += 1
        finally:
            if group_handle is not None:
                group_handle.close()
    if sample_count <= 0:
        raise ValueError("prepared indexed dataset contains no samples")
    files = {
        "tokens.i32": _file_metadata(token_path, "int32", token_count),
        "index.i64": {
            **_file_metadata(
                index_path,
                "int64",
                sample_count * len(INDEXED_SUPERVISED_COLUMNS),
            ),
            "columns": list(INDEXED_SUPERVISED_COLUMNS),
        },
        "loss_mask.u8": {
            **_file_metadata(mask_path, "uint8", token_count),
            "semantics": "supervised_token_mask",
        },
    }
    if include_groups:
        files["groups.i64"] = {
            **_file_metadata(group_path, "int64", sample_count),
            "semantics": "group_id",
        }
    stats = {
        "distinct_sequence_lengths": len(lengths),
        "length_histogram": {str(key): lengths[key] for key in sorted(lengths)},
        "maximum_sequence_length": max(lengths),
        "minimum_sequence_length": min(lengths),
        "padding_tokens": padding_tokens,
        "samples": sample_count,
        "source_tokens": source_tokens,
        "stored_tokens": token_count,
        "supervised_tokens": supervised_tokens,
        "truncated_tokens": truncated_tokens,
    }
    artifact_metadata = dict(metadata)
    supervision_shape = (
        "full_sequence"
        if all_full_supervision
        else "contiguous_suffix"
        if all_contiguous_suffix
        else "arbitrary"
    )
    artifact_metadata.update(
        {
            "loss_mask_encoding": "uint8_per_token_v1",
            "minimum_token_id": min(token_ids_seen),
            "maximum_token_id": max(token_ids_seen),
            "padding": "token_padding" if padding_tokens else "none",
            "sequence_layout": "record_aligned",
            "supervision_shape": supervision_shape,
        }
    )
    manifest = _manifest(
        format_name=INDEXED_SUPERVISED_FORMAT,
        version=INDEXED_SUPERVISED_VERSION,
        metadata=artifact_metadata,
        files=files,
        stats=stats,
        sample_count=sample_count,
        token_count=token_count,
    )
    _write_manifest(root, manifest)
    return manifest


def inspect_artifact(path: str | Path) -> dict[str, Any]:
    """Read manifest metadata without hashing potentially large payloads."""

    root, manifest = _load_manifest(path)
    return {
        "status": "present",
        "path": str(root),
        "format": manifest.get("format"),
        "version": manifest.get("version"),
        "dataset_fingerprint": (manifest.get("metadata") or {}).get(
            "dataset_fingerprint"
        ),
        "sample_count": manifest.get("sample_count"),
        "token_count": manifest.get("token_count"),
        "metadata": manifest.get("metadata"),
        "stats": manifest.get("stats"),
        "files": manifest.get("files"),
    }


def validate_artifact(
    path: str | Path,
    *,
    verify_checksums: bool = True,
) -> dict[str, Any]:
    """Validate artifact schema, sizes, checksums, and indexed bounds."""

    root, manifest = _load_manifest(path)
    format_name, version, expected_files = _artifact_contract(manifest)
    files = _artifact_files(manifest, expected_files)
    _validate_artifact_payloads(
        root,
        files,
        expected_files,
        verify_checksums=verify_checksums,
    )
    token_count, sample_count = _artifact_counts(manifest, files)
    if format_name == INDEXED_SUPERVISED_FORMAT:
        _validate_indexed_contents(root, manifest)
    _validate_fingerprint(
        manifest,
        files,
        format_name=format_name,
        version=version,
        sample_count=sample_count,
        token_count=token_count,
    )
    return manifest


def _artifact_contract(
    manifest: Mapping[str, Any],
) -> tuple[str, int, dict[str, tuple[str, int]]]:
    format_name = manifest.get("format")
    version = int(manifest.get("version", -1))
    metadata = manifest.get("metadata")
    if not isinstance(metadata, Mapping):
        raise RuntimeError("artifact metadata is invalid")
    if format_name == PACKED_TOKEN_FORMAT:
        if version != PACKED_TOKEN_VERSION:
            raise RuntimeError("unsupported packed-token artifact version")
        expected_files = {"tokens.i32": ("int32", 4)}
    elif format_name == INDEXED_SUPERVISED_FORMAT:
        if version != INDEXED_SUPERVISED_VERSION:
            raise RuntimeError("unsupported indexed-token artifact version")
        expected_files = {
            "tokens.i32": ("int32", 4),
            "index.i64": ("int64", 8),
            "loss_mask.u8": ("uint8", 1),
        }
        if "groups.i64" in (manifest.get("files") or {}):
            expected_files["groups.i64"] = ("int64", 8)
        semantics = metadata.get("semantics")
        if semantics not in SUPPORTED_INDEXED_SEMANTICS:
            raise RuntimeError(f"unsupported indexed semantics: {semantics!r}")
    else:
        raise RuntimeError(f"unsupported prepared dataset format: {format_name!r}")
    return str(format_name), version, expected_files


def _artifact_files(
    manifest: Mapping[str, Any],
    expected_files: Mapping[str, tuple[str, int]],
) -> Mapping[str, Any]:
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise RuntimeError("artifact manifest is missing file metadata")
    if set(files) != set(expected_files):
        raise RuntimeError("artifact file set does not match its format")
    return files


def _validate_artifact_payloads(
    root: Path,
    files: Mapping[str, Any],
    expected_files: Mapping[str, tuple[str, int]],
    *,
    verify_checksums: bool,
) -> None:
    for name, (dtype, item_size) in expected_files.items():
        _validate_payload(
            root,
            name,
            files[name],
            dtype=dtype,
            item_size=item_size,
            verify_checksum=verify_checksums,
        )


def _artifact_counts(
    manifest: Mapping[str, Any],
    files: Mapping[str, Any],
) -> tuple[int, int]:
    token_count = int(manifest.get("token_count", -1))
    sample_count = int(manifest.get("sample_count", -1))
    if token_count <= 0 or sample_count <= 0:
        raise RuntimeError("artifact counts must be positive")
    if int(files["tokens.i32"].get("elements", -1)) != token_count:
        raise RuntimeError("artifact token count is inconsistent")
    return token_count, sample_count


def _validate_fingerprint(
    manifest: Mapping[str, Any],
    files: Mapping[str, Any],
    *,
    format_name: str,
    version: int,
    sample_count: int,
    token_count: int,
) -> None:
    metadata = manifest.get("metadata")
    if not isinstance(metadata, Mapping):
        raise RuntimeError("artifact metadata is invalid")
    fingerprint = metadata.get("dataset_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise RuntimeError("artifact dataset fingerprint is invalid")
    observed_fingerprint = _dataset_fingerprint(
        format_name=str(format_name),
        version=version,
        metadata=dict(metadata),
        files=dict(files),
        sample_count=sample_count,
        token_count=token_count,
    )
    if fingerprint != observed_fingerprint:
        raise RuntimeError("artifact dataset fingerprint does not match its manifest")


def validate_artifact_for_run(
    path: str | Path,
    run_spec: Any,
    *,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate artifact shape requirements against a resolved training spec."""

    if manifest is None:
        manifest = validate_artifact(path)
    if getattr(run_spec.data, "input_mode", None) != "dataset":
        raise ValueError("prepared artifacts require data.input_mode=dataset")
    if manifest["format"] == INDEXED_SUPERVISED_FORMAT:
        metadata = manifest["metadata"]
        block_size = int(getattr(run_spec.objective, "block_size", 1) or 1)
        if int(metadata.get("block_size", -1)) != block_size:
            raise ValueError("prepared artifact block size does not match the run")
        maximum = int((manifest.get("stats") or {}).get("maximum_sequence_length", -1))
        if maximum > int(run_spec.model.seq_len):
            raise ValueError(
                "prepared artifact contains sequences longer than model.seq_len"
            )
        objective_name = str(getattr(run_spec.objective, "name", ""))
        supervision_shape = metadata.get("supervision_shape")
        if supervision_shape is None:
            supervision_shape = {
                "completion_only_sft": "contiguous_suffix",
                "full_sequence": "full_sequence",
            }.get(metadata.get("semantics"), "arbitrary")
        if objective_name == "diffusiongemma_native_sft" and supervision_shape not in {
            "full_sequence",
            "contiguous_suffix",
        }:
            raise ValueError(
                "diffusiongemma_native_sft requires full-sequence or contiguous "
                "suffix supervision"
            )
    return {
        "compatible": True,
        "format": manifest["format"],
        "dataset_fingerprint": manifest["metadata"]["dataset_fingerprint"],
    }


def _manifest(
    *,
    format_name: str,
    version: int,
    metadata: dict[str, Any],
    files: dict[str, Any],
    stats: dict[str, Any],
    sample_count: int,
    token_count: int,
) -> dict[str, Any]:
    metadata["dataset_fingerprint"] = _dataset_fingerprint(
        format_name=format_name,
        version=version,
        metadata=metadata,
        files=files,
        sample_count=sample_count,
        token_count=token_count,
    )
    return {
        "format": format_name,
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata,
        "sample_count": sample_count,
        "token_count": token_count,
        "files": files,
        "stats": stats,
    }


def _dataset_fingerprint(
    *,
    format_name: str,
    version: int,
    metadata: dict[str, Any],
    files: dict[str, Any],
    sample_count: int,
    token_count: int,
) -> str:
    stable_metadata = copy.deepcopy(metadata)
    stable_metadata.pop("dataset_fingerprint", None)
    source = stable_metadata.get("source")
    if isinstance(source, dict) and isinstance(source.get("files"), list):
        for item in source["files"]:
            if isinstance(item, dict):
                item.pop("path", None)
    preparation = stable_metadata.get("preparation")
    if isinstance(preparation, dict):
        preparation_source = preparation.get("source")
        if isinstance(preparation_source, dict):
            preparation_source.pop("path", None)
            preparation_source.pop("paths", None)
    fingerprint_payload = {
        "format": format_name,
        "version": version,
        "metadata": stable_metadata,
        "files": {
            name: {
                key: value
                for key, value in file_metadata.items()
                if key in {"columns", "dtype", "elements", "sha256"}
            }
            for name, file_metadata in files.items()
        },
        "sample_count": sample_count,
        "token_count": token_count,
    }
    return sha256_bytes(canonical_json_bytes(fingerprint_payload))


def _write_manifest(root: Path, manifest: Mapping[str, Any]) -> None:
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _file_metadata(path: Path, dtype: str, elements: int) -> dict[str, Any]:
    return {
        "dtype": dtype,
        "elements": int(elements),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _load_manifest(path: str | Path) -> tuple[Path, dict[str, Any]]:
    candidate = Path(path).resolve()
    manifest_path = (
        candidate if candidate.name == "manifest.json" else candidate / "manifest.json"
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(f"prepared dataset manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("prepared dataset manifest must be a JSON object")
    return manifest_path.parent, payload


def _validate_payload(
    root: Path,
    name: str,
    metadata: Any,
    *,
    dtype: str,
    item_size: int,
    verify_checksum: bool,
) -> None:
    if not isinstance(metadata, Mapping):
        raise RuntimeError(f"artifact metadata is invalid for {name}")
    if metadata.get("dtype") != dtype:
        raise RuntimeError(f"artifact dtype is invalid for {name}")
    elements = int(metadata.get("elements", -1))
    expected_bytes = elements * item_size
    path = root / name
    if elements < 0 or int(metadata.get("bytes", -1)) != expected_bytes:
        raise RuntimeError(f"artifact byte metadata is invalid for {name}")
    if not path.is_file() or path.stat().st_size != expected_bytes:
        raise RuntimeError(f"artifact payload size is invalid for {name}")
    if verify_checksum and sha256_file(path) != metadata.get("sha256"):
        raise RuntimeError(f"artifact checksum does not match for {name}")


def _validate_indexed_contents(root: Path, manifest: Mapping[str, Any]) -> None:
    sample_count = int(manifest["sample_count"])
    token_count = int(manifest["token_count"])
    files = manifest["files"]
    expected_index_elements = sample_count * len(INDEXED_SUPERVISED_COLUMNS)
    if int(files["index.i64"].get("elements", -1)) != expected_index_elements:
        raise RuntimeError("indexed artifact record count is inconsistent")
    if tuple(files["index.i64"].get("columns", ())) != INDEXED_SUPERVISED_COLUMNS:
        raise RuntimeError("indexed artifact columns are incompatible")
    if int(files["loss_mask.u8"].get("elements", -1)) != token_count:
        raise RuntimeError("indexed artifact loss mask size is inconsistent")
    index = torch.from_file(
        str(root / "index.i64"),
        dtype=torch.int64,
        size=expected_index_elements,
    ).reshape(sample_count, len(INDEXED_SUPERVISED_COLUMNS))
    masks = torch.from_file(
        str(root / "loss_mask.u8"), dtype=torch.uint8, size=token_count
    )
    previous_end = 0
    for offset, length, supervision_start, _sample_id in index.tolist():
        if offset != previous_end or length <= 0 or offset + length > token_count:
            raise RuntimeError("indexed artifact token offsets are invalid")
        if not 0 <= supervision_start < length:
            raise RuntimeError("indexed artifact supervision start is invalid")
        record_mask = masks.narrow(0, offset, length)
        if bool(record_mask.gt(1).any()) or int(record_mask.sum()) <= 0:
            raise RuntimeError("indexed artifact supervision mask is invalid")
        first_supervised = int(record_mask.nonzero(as_tuple=False)[0].item())
        if supervision_start != first_supervised:
            raise RuntimeError("indexed artifact supervision start is inconsistent")
        previous_end = offset + length
    if previous_end != token_count:
        raise RuntimeError("indexed artifact does not cover its token payload")


def _write_i32(handle: Any, values: Iterable[int]) -> None:
    payload = array.array("i", (int(value) for value in values))
    if payload.itemsize != 4:  # pragma: no cover - platform guard
        raise RuntimeError("platform int storage is not 32 bits")
    if payload and (min(payload) < 0 or max(payload) >= 2**31):
        raise ValueError("token IDs must fit nonnegative int32 storage")
    payload.tofile(handle)


def _write_i64(handle: Any, values: Iterable[int]) -> None:
    payload = array.array("q", (int(value) for value in values))
    if payload.itemsize != 8:  # pragma: no cover - platform guard
        raise RuntimeError("platform long-long storage is not 64 bits")
    payload.tofile(handle)


def _stable_int_id(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("sample and group IDs must not be booleans")
    if isinstance(value, int):
        result = int(value)
        if not -(2**63) <= result < 2**63:
            raise ValueError("sample and group IDs must fit int64 storage")
        return result
    digest = sha256_bytes(canonical_json_bytes(value))
    return int(digest[:16], 16) & ((1 << 63) - 1)


__all__ = (
    "PACKED_TOKEN_FORMAT",
    "PACKED_TOKEN_VERSION",
    "SUPPORTED_INDEXED_SEMANTICS",
    "inspect_artifact",
    "validate_artifact",
    "validate_artifact_for_run",
    "write_indexed_artifact",
    "write_packed_artifact",
)
