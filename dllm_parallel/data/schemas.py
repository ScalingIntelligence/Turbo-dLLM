# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Strict configuration schemas for offline dataset preparation."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from dllm_parallel.data.registry import (
    is_formatter_registered,
    is_source_registered,
)

PREPARATION_SPEC_FORMAT = "dllm.data.prepare.v1"
BUILTIN_SOURCE_TYPES = frozenset(
    {"huggingface", "jsonl", "parquet", "pretokenized", "text"}
)
BUILTIN_RECORD_TYPES = frozenset(
    {"messages", "pretokenized", "prompt_completion", "text"}
)


@dataclass(frozen=True)
class SourceSpec:
    type: str
    path: str | None = None
    paths: tuple[str, ...] = ()
    id: str | None = None
    name: str | None = None
    split: str = "train"
    revision: str | None = None
    streaming: bool = False
    text_mode: str = "document"
    encoding: str = "utf-8"
    loader_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecordSpec:
    type: str
    text_field: str = "text"
    messages_field: str = "messages"
    role_field: str = "role"
    content_field: str = "content"
    prompt_field: str = "prompt"
    completion_field: str = "completion"
    input_ids_field: str = "input_ids"
    labels_field: str = "labels"
    loss_mask_field: str = "loss_mask"
    assistant_mask_field: str = "assistant_masks"
    completion_mask_field: str = "completion_mask"
    sample_id_field: str | None = None
    group_id_field: str | None = None


@dataclass(frozen=True)
class TokenizerSpec:
    model: str | None = None
    revision: str | None = None
    trust_remote_code: bool = False
    use_fast: bool = True
    chat_template: str | None = None
    add_eos: bool = False
    kwargs: dict[str, Any] = field(default_factory=dict)
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SupervisionSpec:
    policy: str = "full"


@dataclass(frozen=True)
class PackingSpec:
    maximum_length: int
    alignment: int = 1
    overflow: str = "reject"
    alignment_policy: str = "truncate_right"
    pad_token_id: int | None = None
    separator_token_id: int | None = None
    minimum_length: int | None = None


@dataclass(frozen=True)
class OutputSpec:
    path: str
    format: str = "auto"
    overwrite: bool = False


@dataclass(frozen=True)
class PreparationSpec:
    source: SourceSpec
    records: RecordSpec
    packing: PackingSpec
    output: OutputSpec
    tokenizer: TokenizerSpec = field(default_factory=TokenizerSpec)
    supervision: SupervisionSpec = field(default_factory=SupervisionSpec)
    format: str = PREPARATION_SPEC_FORMAT

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any],
        *,
        base_dir: str | Path | None = None,
    ) -> PreparationSpec:
        if not isinstance(mapping, Mapping):
            raise ValueError("preparation configuration must be a mapping")
        _unknown(
            mapping,
            {
                "format",
                "source",
                "records",
                "tokenizer",
                "supervision",
                "packing",
                "output",
            },
            "config",
        )
        format_name = str(mapping.get("format", PREPARATION_SPEC_FORMAT))
        if format_name != PREPARATION_SPEC_FORMAT:
            raise ValueError(
                f"format must be {PREPARATION_SPEC_FORMAT!r}, got {format_name!r}"
            )
        root = None if base_dir is None else Path(base_dir).resolve()
        source_values = _section(mapping, "source", required=True)
        output_values = _section(mapping, "output", required=True)
        source_values = dict(source_values)
        output_values = dict(output_values)
        if root is not None:
            if source_values.get("path") is not None:
                source_values["path"] = _resolve_path(source_values["path"], root)
            if source_values.get("paths") is not None:
                source_values["paths"] = tuple(
                    _resolve_path(value, root) for value in source_values["paths"]
                )
            if output_values.get("path") is not None:
                output_values["path"] = _resolve_path(output_values["path"], root)
        source = _construct(SourceSpec, source_values, "source")
        records = _construct(
            RecordSpec,
            _section(mapping, "records", required=True),
            "records",
        )
        tokenizer = _construct(
            TokenizerSpec,
            _section(mapping, "tokenizer"),
            "tokenizer",
        )
        supervision = _construct(
            SupervisionSpec,
            _section(mapping, "supervision"),
            "supervision",
        )
        packing = _construct(
            PackingSpec,
            _section(mapping, "packing", required=True),
            "packing",
        )
        output = _construct(OutputSpec, output_values, "output")
        spec = cls(
            format=format_name,
            source=source,
            records=records,
            tokenizer=tokenizer,
            supervision=supervision,
            packing=packing,
            output=output,
        )
        spec.validate()
        return spec

    @classmethod
    def from_path(cls, path: str | Path) -> PreparationSpec:
        config_path = Path(path).resolve()
        if not config_path.is_file():
            raise FileNotFoundError(str(config_path))
        text = config_path.read_text(encoding="utf-8")
        if config_path.suffix.lower() == ".json":
            payload = json.loads(text)
        else:
            payload = yaml.safe_load(text)
        return cls.from_mapping(payload, base_dir=config_path.parent)

    def validate(self) -> None:
        _validate_field_types(self)
        source_type = _normalized_name(self.source.type)
        record_type = _normalized_name(self.records.type).replace("-", "_")
        _validate_source(self.source, source_type)
        _validate_supervision(self.supervision, record_type)
        _validate_packing(self.packing)
        _validate_output(self)

    def to_mapping(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _validate_field_types(spec: PreparationSpec) -> None:
    _require_bool("source.streaming", spec.source.streaming)
    _require_mapping("source.loader_kwargs", spec.source.loader_kwargs)
    for name in ("trust_remote_code", "use_fast", "add_eos"):
        _require_bool(f"tokenizer.{name}", getattr(spec.tokenizer, name))
    _require_mapping("tokenizer.kwargs", spec.tokenizer.kwargs)
    _require_mapping(
        "tokenizer.chat_template_kwargs",
        spec.tokenizer.chat_template_kwargs,
    )
    _require_int("packing.maximum_length", spec.packing.maximum_length)
    _require_int("packing.alignment", spec.packing.alignment)
    for name in ("pad_token_id", "separator_token_id", "minimum_length"):
        value = getattr(spec.packing, name)
        if value is not None:
            _require_int(f"packing.{name}", value)
    _require_bool("output.overwrite", spec.output.overwrite)


def _validate_source(spec: SourceSpec, source_type: str) -> None:
    if source_type not in BUILTIN_SOURCE_TYPES and not is_source_registered(
        source_type
    ):
        raise ValueError(f"unsupported source.type: {spec.type!r}")
    if source_type == "huggingface" and not spec.id:
        raise ValueError("huggingface sources require source.id")
    if (
        source_type in {"jsonl", "parquet", "pretokenized", "text"}
        and not spec.path
        and not spec.paths
    ):
        raise ValueError(f"{source_type} sources require source.path or source.paths")
    if spec.path and spec.paths:
        raise ValueError("source.path and source.paths are mutually exclusive")
    _choice("source.text_mode", spec.text_mode, {"document", "line"})


def _validate_supervision(spec: SupervisionSpec, record_type: str) -> None:
    if record_type not in BUILTIN_RECORD_TYPES and not is_formatter_registered(
        record_type
    ):
        raise ValueError(f"unsupported records.type: {record_type!r}")
    _choice(
        "supervision.policy",
        spec.policy,
        {"assistant_only", "completion_only", "full", "provided"},
    )
    allowed_records = {
        "assistant_only": {"messages", "pretokenized"},
        "completion_only": {"prompt_completion"},
        "provided": {"pretokenized"},
    }
    allowed = allowed_records.get(spec.policy)
    if allowed is not None and record_type not in allowed:
        expected = " or ".join(sorted(allowed))
        raise ValueError(
            f"supervision.policy={spec.policy} requires {expected} records"
        )


def _validate_packing(spec: PackingSpec) -> None:
    if int(spec.maximum_length) <= 0:
        raise ValueError("packing.maximum_length must be positive")
    if int(spec.alignment) <= 0:
        raise ValueError("packing.alignment must be positive")
    if spec.maximum_length % spec.alignment:
        raise ValueError(
            "packing.maximum_length must be divisible by packing.alignment"
        )
    _choice(
        "packing.overflow",
        spec.overflow,
        {"reject", "split", "truncate_left", "truncate_right"},
    )
    _choice(
        "packing.alignment_policy",
        spec.alignment_policy,
        {"pad_left", "pad_right", "reject", "truncate_left", "truncate_right"},
    )
    if spec.alignment_policy in {"pad_left", "pad_right"}:
        raise ValueError(
            "preparation-time padding requires a valid-token mask artifact, "
            "which data format v1 does not provide"
        )
    if spec.minimum_length is not None:
        if spec.minimum_length <= 0:
            raise ValueError("packing.minimum_length must be positive")
        if spec.minimum_length > spec.maximum_length:
            raise ValueError(
                "packing.minimum_length cannot exceed packing.maximum_length"
            )


def _validate_output(spec: PreparationSpec) -> None:
    _choice("output.format", spec.output.format, {"auto", "indexed", "packed"})
    if not str(spec.output.path).strip():
        raise ValueError("output.path must not be empty")
    supervised = spec.supervision.policy != "full"
    if supervised and spec.output.format == "packed":
        raise ValueError("supervised records require output.format=indexed")
    if spec.packing.overflow == "split":
        if supervised:
            raise ValueError("packing.overflow=split requires full supervision")
        if spec.output.format == "indexed":
            raise ValueError("packing.overflow=split requires packed output")


def _section(
    mapping: Mapping[str, Any],
    name: str,
    *,
    required: bool = False,
) -> Mapping[str, Any]:
    value = mapping.get(name)
    if value is None:
        if required:
            raise ValueError(f"configuration requires {name}")
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _construct(cls: type[Any], values: Mapping[str, Any], section: str) -> Any:
    allowed = {item.name for item in dataclasses.fields(cls)}
    _unknown(values, allowed, section)
    normalized = dict(values)
    if cls is SourceSpec and "paths" in normalized:
        paths = normalized["paths"]
        if isinstance(paths, (str, bytes)) or not isinstance(paths, (list, tuple)):
            raise ValueError("source.paths must be a list")
        normalized["paths"] = tuple(str(value) for value in paths)
    try:
        return cls(**normalized)
    except TypeError as exc:
        raise ValueError(f"invalid {section} configuration: {exc}") from exc


def _unknown(values: Mapping[str, Any], allowed: set[str], section: str) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown {section} keys: {', '.join(unknown)}")


def _choice(name: str, value: str, choices: set[str]) -> None:
    if value not in choices:
        raise ValueError(f"{name} must be one of {sorted(choices)}, got {value!r}")


def _require_bool(name: str, value: Any) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")


def _require_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")


def _require_mapping(name: str, value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")


def _resolve_path(value: Any, root: Path) -> str:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = root / path
    return str(path.resolve())


def _normalized_name(value: str) -> str:
    return str(value).strip().lower().replace("_", "-")


__all__ = (
    "BUILTIN_RECORD_TYPES",
    "BUILTIN_SOURCE_TYPES",
    "PREPARATION_SPEC_FORMAT",
    "OutputSpec",
    "PackingSpec",
    "PreparationSpec",
    "RecordSpec",
    "SourceSpec",
    "SupervisionSpec",
    "TokenizerSpec",
)
