# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""End-to-end orchestration for generic offline dataset preparation."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dllm_parallel.core.data import (
    INDEXED_SUPERVISED_FORMAT,
    tokenizer_vocabulary_sha256,
)
from dllm_parallel.data.formatting import format_record
from dllm_parallel.data.indexed import (
    PACKED_TOKEN_FORMAT,
    write_indexed_artifact,
    write_packed_artifact,
)
from dllm_parallel.data.schemas import PreparationSpec
from dllm_parallel.data.sources import (
    iter_source_records,
    resolve_source_paths,
    source_provenance,
)
from dllm_parallel.data.tokenization import (
    TokenizedRecord,
    load_tokenizer,
    tokenize_record,
)


@dataclass(frozen=True)
class PreparationResult:
    artifact_path: Path
    training_path: Path
    format: str
    sample_count: int
    token_count: int
    dataset_fingerprint: str
    manifest: dict[str, Any]


def prepare_dataset(
    spec: PreparationSpec | Mapping[str, Any] | str | Path,
    *,
    tokenizer: Any | None = None,
) -> PreparationResult:
    """Compile a common dataset source into an optimized training artifact."""

    resolved = _spec(spec)
    format_name = _output_format(resolved)
    destination = Path(resolved.output.path).expanduser().resolve()
    _validate_destination(destination)
    if destination.exists() and not resolved.output.overwrite:
        raise FileExistsError(str(destination))
    if tokenizer is None:
        tokenizer = load_tokenizer(resolved.tokenizer)
    if format_name == INDEXED_SUPERVISED_FORMAT and tokenizer is None:
        raise ValueError(
            "indexed artifacts require tokenizer.model so runtime identity can be "
            "verified"
        )
    source_paths = resolve_source_paths(resolved.source)
    records = _tokenized_records(resolved, tokenizer, source_paths=source_paths)
    source = source_provenance(resolved.source, resolved_paths=source_paths)
    metadata = {
        "contains_mask_token": False,
        "preparation": _sanitized_preparation(resolved),
        "source": source,
    }
    if tokenizer is not None:
        metadata.update(_tokenizer_metadata(resolved, tokenizer))
    if format_name == INDEXED_SUPERVISED_FORMAT:
        assert tokenizer is not None
        metadata.update(
            {
                "block_size": int(resolved.packing.alignment),
                "maximum_sequence_length": int(resolved.packing.maximum_length),
                "semantics": _semantics(resolved.supervision.policy),
                "sampling_policy": "same_length_global_batches_v1",
                "sequence_layout": "record_aligned",
            }
        )
    else:
        metadata.update(
            {
                "maximum_record_length": int(resolved.packing.maximum_length),
                "padding": "none",
                "sampling_policy": "sequential_cyclic_rank_strided_v1",
                "semantics": "full_sequence",
                "sequence_layout": "continuous_stream",
                "supervision_shape": "full_sequence",
            }
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_parent = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    staging = staging_parent / "artifact"
    try:
        if format_name == PACKED_TOKEN_FORMAT:
            manifest = write_packed_artifact(
                staging,
                records,
                packing=resolved.packing,
                metadata=metadata,
            )
        else:
            manifest = write_indexed_artifact(
                staging,
                records,
                packing=resolved.packing,
                metadata=metadata,
                include_groups=resolved.records.group_id_field is not None,
            )
        _install_staged_artifact(
            staging,
            destination,
            overwrite=bool(resolved.output.overwrite),
        )
    finally:
        shutil.rmtree(staging_parent, ignore_errors=True)
    return PreparationResult(
        artifact_path=destination,
        training_path=destination,
        format=str(manifest["format"]),
        sample_count=int(manifest["sample_count"]),
        token_count=int(manifest["token_count"]),
        dataset_fingerprint=str(manifest["metadata"]["dataset_fingerprint"]),
        manifest=manifest,
    )


def _tokenized_records(
    spec: PreparationSpec,
    tokenizer: Any | None,
    *,
    source_paths: tuple[Path, ...],
) -> Iterator[TokenizedRecord]:
    for raw_record in iter_source_records(
        spec.source,
        resolved_paths=source_paths,
    ):
        formatted = format_record(raw_record, spec.records)
        yield tokenize_record(
            formatted,
            tokenizer=tokenizer,
            tokenizer_spec=spec.tokenizer,
            supervision=spec.supervision,
        )


def _spec(
    value: PreparationSpec | Mapping[str, Any] | str | Path,
) -> PreparationSpec:
    if isinstance(value, PreparationSpec):
        value.validate()
        return value
    if isinstance(value, Mapping):
        return PreparationSpec.from_mapping(value)
    return PreparationSpec.from_path(value)


def _output_format(spec: PreparationSpec) -> str:
    requested = spec.output.format
    if requested == "auto":
        return (
            INDEXED_SUPERVISED_FORMAT
            if spec.supervision.policy != "full"
            else PACKED_TOKEN_FORMAT
        )
    if requested == "packed":
        return PACKED_TOKEN_FORMAT
    return INDEXED_SUPERVISED_FORMAT


def _semantics(policy: str) -> str:
    return {
        "assistant_only": "assistant_only_sft",
        "completion_only": "completion_only_sft",
        "full": "full_sequence",
        "provided": "supervised_tokens",
    }[policy]


def _tokenizer_metadata(spec: PreparationSpec, tokenizer: Any) -> dict[str, Any]:
    template = (
        spec.tokenizer.chat_template
        if spec.tokenizer.chat_template is not None
        else str(getattr(tokenizer, "chat_template", "") or "")
    )
    return {
        "tokenizer_model": spec.tokenizer.model,
        "tokenizer_revision": spec.tokenizer.revision,
        "tokenizer_size": len(tokenizer),
        "tokenizer_vocabulary_sha256": tokenizer_vocabulary_sha256(tokenizer),
        "chat_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
        "mask_token_id": getattr(tokenizer, "mask_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
    }


def _sanitized_preparation(spec: PreparationSpec) -> dict[str, Any]:
    mapping = spec.to_mapping()
    mapping["output"] = {
        "format": spec.output.format,
        "path": "<artifact>",
        "overwrite": False,
    }
    mapping["source"]["loader_kwargs"] = _redacted(
        mapping["source"].get("loader_kwargs", {})
    )
    mapping["tokenizer"]["kwargs"] = _redacted(mapping["tokenizer"].get("kwargs", {}))
    return mapping


def _redacted(values: Mapping[str, Any]) -> dict[str, Any]:
    secret_fragments = ("credential", "key", "password", "secret", "token")
    return {
        str(key): (
            "<redacted>"
            if any(fragment in str(key).lower() for fragment in secret_fragments)
            else value
        )
        for key, value in values.items()
    }


def _validate_destination(path: Path) -> None:
    if path == path.parent or path.name in {"", ".", ".."}:
        raise ValueError("output.path must name a specific artifact directory")


def _install_staged_artifact(
    staging: Path,
    destination: Path,
    *,
    overwrite: bool,
) -> None:
    if not destination.exists():
        os.replace(staging, destination)
        return
    if not overwrite:
        raise FileExistsError(str(destination))
    backup = destination.parent / f".{destination.name}.backup-{os.getpid()}"
    if backup.exists():
        raise FileExistsError(str(backup))
    os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except BaseException:
        os.replace(backup, destination)
        raise
    if backup.is_dir() and not backup.is_symlink():
        shutil.rmtree(backup)
    else:
        backup.unlink()


__all__ = ("PreparationResult", "prepare_dataset")
