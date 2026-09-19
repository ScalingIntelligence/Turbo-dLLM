# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Extension registries for generic data preparation frontends."""

from __future__ import annotations

from importlib import metadata
from collections.abc import Callable
from typing import Any

SourceFactory = Callable[[Any], Any]
RecordFormatter = Callable[[Any, Any], Any]

_SOURCES: dict[str, SourceFactory] = {}
_FORMATTERS: dict[str, RecordFormatter] = {}

SOURCE_ENTRY_POINT_GROUP = "dllm_parallel.data_sources.v1"
FORMATTER_ENTRY_POINT_GROUP = "dllm_parallel.data_formatters.v1"
_DISCOVERED_GROUPS: set[str] = set()


def _name(value: str) -> str:
    normalized = str(value).strip().lower().replace("_", "-")
    if not normalized:
        raise ValueError("registry name must not be empty")
    return normalized


def _register(
    registry: dict[str, Callable[..., Any]],
    kind: str,
    name: str,
    handler: Callable[..., Any],
    *,
    replace: bool,
) -> None:
    normalized = _name(name)
    if not callable(handler):
        raise TypeError(f"{kind} handler must be callable")
    if normalized in registry and not replace:
        raise ValueError(f"{kind} {normalized!r} is already registered")
    registry[normalized] = handler


def register_source(
    name: str,
    handler: SourceFactory,
    *,
    replace: bool = False,
) -> None:
    """Register a raw-record source factory."""

    _register(_SOURCES, "data source", name, handler, replace=replace)


def register_formatter(
    name: str,
    handler: RecordFormatter,
    *,
    replace: bool = False,
) -> None:
    """Register a raw-record formatter."""

    _register(_FORMATTERS, "record formatter", name, handler, replace=replace)


def get_source(name: str) -> SourceFactory:
    """Return a registered source or raise an actionable error."""

    normalized = _name(name)
    if normalized not in _SOURCES:
        _discover_entry_points(SOURCE_ENTRY_POINT_GROUP, _SOURCES, "data source")
    try:
        return _SOURCES[normalized]
    except KeyError as exc:
        raise KeyError(f"unknown data source: {normalized}") from exc


def get_formatter(name: str) -> RecordFormatter:
    """Return a registered formatter or raise an actionable error."""

    normalized = _name(name)
    if normalized not in _FORMATTERS:
        _discover_entry_points(
            FORMATTER_ENTRY_POINT_GROUP,
            _FORMATTERS,
            "record formatter",
        )
    try:
        return _FORMATTERS[normalized]
    except KeyError as exc:
        raise KeyError(f"unknown record formatter: {normalized}") from exc


def is_source_registered(name: str) -> bool:
    normalized = _name(name)
    if normalized not in _SOURCES:
        _discover_entry_points(SOURCE_ENTRY_POINT_GROUP, _SOURCES, "data source")
    return normalized in _SOURCES


def is_formatter_registered(name: str) -> bool:
    normalized = _name(name)
    if normalized not in _FORMATTERS:
        _discover_entry_points(
            FORMATTER_ENTRY_POINT_GROUP,
            _FORMATTERS,
            "record formatter",
        )
    return normalized in _FORMATTERS


def _discover_entry_points(
    group: str,
    registry: dict[str, Callable[..., Any]],
    kind: str,
) -> None:
    if group in _DISCOVERED_GROUPS:
        return
    discovered = metadata.entry_points()
    if hasattr(discovered, "select"):
        entries = discovered.select(group=group)
    else:  # pragma: no cover - compatibility with older importlib-metadata
        entries = discovered.get(group, ())
    loaded: list[tuple[str, Callable[..., Any]]] = []
    for entry in entries:
        handler = entry.load()
        if not callable(handler):
            raise TypeError(
                f"{kind} entry point {entry.name!r} from {group!r} must be callable"
            )
        loaded.append((_name(entry.name), handler))
    for name, handler in loaded:
        _register(registry, kind, name, handler, replace=False)
    _DISCOVERED_GROUPS.add(group)


__all__ = (
    "RecordFormatter",
    "FORMATTER_ENTRY_POINT_GROUP",
    "SOURCE_ENTRY_POINT_GROUP",
    "SourceFactory",
    "get_formatter",
    "get_source",
    "is_formatter_registered",
    "is_source_registered",
    "register_formatter",
    "register_source",
)
