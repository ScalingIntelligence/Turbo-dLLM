# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Shared config helpers for backbone modules."""

from __future__ import annotations

from typing import Any

from dllm_parallel.core.models.contracts import ModelFamilySpec
from dllm_parallel.core.specs import ParallelSpec


class ConfigNamespace:
    """Recursive attribute namespace for backbone-local config objects."""

    def __init__(self, values: dict[str, Any]) -> None:
        for key, value in values.items():
            setattr(self, str(key), namespace_from_mapping(value))


def namespace_from_mapping(value: Any) -> Any:
    if isinstance(value, ConfigNamespace):
        return value
    if isinstance(value, dict):
        return ConfigNamespace(value)
    if isinstance(value, list):
        return [namespace_from_mapping(item) for item in value]
    if isinstance(value, tuple):
        return tuple(namespace_from_mapping(item) for item in value)
    return value


def namespace_to_dict(value: Any) -> Any:
    if isinstance(value, ConfigNamespace):
        return {
            key: namespace_to_dict(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    if isinstance(value, dict):
        return {str(key): namespace_to_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [namespace_to_dict(item) for item in value]
    if isinstance(value, tuple):
        return [namespace_to_dict(item) for item in value]
    return value


def node(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    return value if isinstance(value, dict) else {}


def first(value: Any) -> str | None:
    if isinstance(value, list) and value:
        return str(value[0])
    if value is None:
        return None
    return str(value)


def string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def string_tuple(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise TypeError("value must be a list or tuple of strings")
    return tuple(str(item) for item in value)


def bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return bool(value)


def optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def required_int(value: Any, name: str) -> int:
    if value is None:
        raise ValueError(f"{name} is required")
    return int(value)


def validate_positive_parallel(parallel: ParallelSpec) -> None:
    for field in (
        "data_parallel_size",
        "context_parallel_size",
        "block_parallel_size",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "expert_parallel_size",
    ):
        if int(getattr(parallel, field)) <= 0:
            raise ValueError(f"{field} must be positive")
    if bool(getattr(parallel, "sequence_parallel", False)) and int(
        getattr(parallel, "tensor_parallel_size", 1)
    ) <= 1:
        raise ValueError("sequence_parallel requires tensor_parallel_size > 1")


def validate_tensor_parallel_heads(
    spec: ModelFamilySpec,
    parallel: ParallelSpec,
) -> None:
    tp = int(parallel.tensor_parallel_size)
    if tp <= 1:
        return
    if spec.num_attention_heads and spec.num_attention_heads % tp != 0:
        raise ValueError("num_attention_heads must divide evenly by tensor_parallel_size")
    if spec.num_key_value_heads and spec.num_key_value_heads % tp != 0:
        raise ValueError("num_key_value_heads must divide evenly by tensor_parallel_size")
