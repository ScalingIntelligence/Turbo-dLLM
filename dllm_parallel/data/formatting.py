# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Dataset-agnostic raw-record normalization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
import torch

from dllm_parallel.data.registry import get_formatter
from dllm_parallel.data.schemas import BUILTIN_RECORD_TYPES, RecordSpec

IntegerSequence = Sequence[int] | np.ndarray | torch.Tensor


@dataclass(frozen=True)
class FormattedRecord:
    kind: str
    text: str | None = None
    messages: tuple[dict[str, Any], ...] | None = None
    prompt: str | None = None
    completion: str | None = None
    prompt_messages: tuple[dict[str, Any], ...] | None = None
    completion_messages: tuple[dict[str, Any], ...] | None = None
    input_ids: IntegerSequence | None = None
    labels: tuple[int, ...] | None = None
    loss_mask: tuple[bool, ...] | None = None
    assistant_mask: tuple[bool, ...] | None = None
    completion_mask: tuple[bool, ...] | None = None
    sample_id: Any | None = None
    group_id: Any | None = None


def format_record(record: Any, spec: RecordSpec) -> FormattedRecord:
    """Normalize one source record into a typed, tokenizer-ready record."""

    record_type = _record_type(spec.type)
    if record_type not in BUILTIN_RECORD_TYPES:
        return _format_custom(record, spec, record_type)
    if record_type == "text":
        return _format_text(record, spec)
    if not isinstance(record, Mapping):
        raise ValueError(f"{record_type} records must be mappings")
    common: dict[str, Any] = {
        "sample_id": _optional_field(record, spec.sample_id_field),
        "group_id": _optional_field(record, spec.group_id_field),
    }
    if record_type == "messages":
        raw_messages = _field(record, spec.messages_field)
        messages = _normalize_messages(raw_messages, spec, label="messages")
        return FormattedRecord(kind="messages", messages=messages, **common)
    if record_type == "prompt_completion":
        return _format_prompt_completion(record, spec, common)
    return _format_pretokenized(record, spec, common)


def _format_custom(record: Any, spec: RecordSpec, record_type: str) -> FormattedRecord:
    result = get_formatter(record_type)(record, spec)
    if not isinstance(result, FormattedRecord):
        raise TypeError("custom record formatter must return FormattedRecord")
    return result


def _format_text(record: Any, spec: RecordSpec) -> FormattedRecord:
    if isinstance(record, str):
        text = record
    else:
        try:
            text = _field(record, spec.text_field)
        except ValueError as exc:
            raise ValueError(f"text field {spec.text_field!r} is missing") from exc
    if not isinstance(text, str):
        raise ValueError(f"text field {spec.text_field!r} must be a string")
    return FormattedRecord(
        kind="text",
        text=text,
        sample_id=_optional_field(record, spec.sample_id_field),
        group_id=_optional_field(record, spec.group_id_field),
    )


def _format_prompt_completion(
    record: Mapping[str, Any],
    spec: RecordSpec,
    common: Mapping[str, Any],
) -> FormattedRecord:
    prompt = _field(record, spec.prompt_field)
    completion = _field(record, spec.completion_field)
    if isinstance(prompt, str) and isinstance(completion, str):
        return FormattedRecord(
            kind="prompt_completion",
            prompt=prompt,
            completion=completion,
            **common,
        )
    if _is_message_sequence(prompt) and _is_message_sequence(completion):
        return FormattedRecord(
            kind="prompt_completion",
            prompt_messages=_normalize_messages(prompt, spec, label="prompt"),
            completion_messages=_normalize_messages(
                completion,
                spec,
                label="completion",
            ),
            **common,
        )
    raise ValueError(
        "prompt and completion fields must be both strings or both message sequences"
    )


def _format_pretokenized(
    record: Mapping[str, Any],
    spec: RecordSpec,
    common: Mapping[str, Any],
) -> FormattedRecord:
    input_ids = _integer_sequence(_field(record, spec.input_ids_field), "input_ids")
    return FormattedRecord(
        kind="pretokenized",
        input_ids=input_ids,
        labels=_optional_integer_tuple(record, spec.labels_field),
        loss_mask=_optional_bool_tuple(record, spec.loss_mask_field),
        assistant_mask=_optional_bool_tuple(record, spec.assistant_mask_field),
        completion_mask=_optional_bool_tuple(record, spec.completion_mask_field),
        **common,
    )


def _field(record: Any, path: str) -> Any:
    current = record
    for component in str(path).split("."):
        if not isinstance(current, Mapping) or component not in current:
            raise ValueError(f"record is missing field {path!r}")
        current = current[component]
    return current


def _is_message_sequence(value: Any) -> bool:
    return not isinstance(value, (str, bytes)) and isinstance(value, Sequence)


def _normalize_messages(
    raw_messages: Any,
    spec: RecordSpec,
    *,
    label: str,
) -> tuple[dict[str, Any], ...]:
    if not _is_message_sequence(raw_messages):
        raise ValueError(f"{label} field must be a sequence")
    messages: list[dict[str, Any]] = []
    for index, message in enumerate(raw_messages):
        if not isinstance(message, Mapping):
            raise ValueError(f"{label}[{index}] must be a mapping")
        try:
            role = message[spec.role_field]
            content = message[spec.content_field]
        except KeyError as exc:
            raise ValueError(
                f"{label}[{index}] requires role and content fields"
            ) from exc
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError(f"{label}[{index}] role and content must be strings")
        normalized = dict(message)
        normalized.pop(spec.role_field, None)
        normalized.pop(spec.content_field, None)
        normalized.update({"role": role, "content": content})
        messages.append(normalized)
    if not messages:
        raise ValueError(f"{label} record must not be empty")
    return tuple(messages)


def _optional_field(record: Any, path: str | None) -> Any | None:
    if path is None:
        return None
    return _field(record, path)


def _optional_integer_tuple(
    record: Mapping[str, Any], path: str
) -> tuple[int, ...] | None:
    try:
        value = _field(record, path)
    except ValueError:
        return None
    return _integer_tuple(value, path)


def _optional_bool_tuple(
    record: Mapping[str, Any], path: str
) -> tuple[bool, ...] | None:
    try:
        value = _field(record, path)
    except ValueError:
        return None
    value = _plain_sequence(value)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{path} must be a sequence")
    result = tuple(bool(item) for item in value)
    if any(item not in (0, 1, False, True) for item in value):
        raise ValueError(f"{path} values must be boolean or 0/1")
    return result


def _integer_tuple(value: Any, name: str) -> tuple[int, ...]:
    value = _plain_sequence(value)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, Integral):
            raise ValueError(f"{name} values must be integers")
        result.append(int(item))
    if not result:
        raise ValueError(f"{name} must not be empty")
    return tuple(result)


def _integer_sequence(value: Any, name: str) -> IntegerSequence:
    if isinstance(value, np.ndarray):
        if value.ndim != 1 or not np.issubdtype(value.dtype, np.integer):
            raise ValueError(f"{name} must be a one-dimensional integer sequence")
        if value.size == 0:
            raise ValueError(f"{name} must not be empty")
        return value
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        integer_dtypes = {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
        if value.ndim != 1 or value.dtype not in integer_dtypes:
            raise ValueError(f"{name} must be a one-dimensional integer sequence")
        if value.numel() == 0:
            raise ValueError(f"{name} must not be empty")
        return value
    return _integer_tuple(value, name)


def _plain_sequence(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _record_type(value: str) -> str:
    return str(value).strip().lower().replace("-", "_")


__all__ = ("FormattedRecord", "format_record")
