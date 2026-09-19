# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""MoE routing state for checkpoint-stable dynamic expert dispatch."""

from __future__ import annotations

from collections import defaultdict, deque
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

import torch

_RouteCache = dict[int, deque[torch.Tensor]]

_ROUTE_MODE: ContextVar[str | None] = ContextVar("dllm_moe_route_mode", default=None)
_ROUTE_CACHE: ContextVar[_RouteCache | None] = ContextVar("dllm_moe_route_cache", default=None)


def moe_route_checkpoint_context_fn() -> tuple[Any, Any]:
    route_cache: _RouteCache = defaultdict(deque)
    return (
        _moe_route_checkpoint_context("record", route_cache),
        _moe_route_checkpoint_context("replay", route_cache),
    )


@contextmanager
def _moe_route_checkpoint_context(mode: str, cache: _RouteCache) -> Iterator[None]:
    mode_token = _ROUTE_MODE.set(mode)
    cache_token = _ROUTE_CACHE.set(cache)
    try:
        yield
    finally:
        _ROUTE_CACHE.reset(cache_token)
        _ROUTE_MODE.reset(mode_token)


def record_moe_route_indices(router: Any, indices: torch.Tensor) -> None:
    if _ROUTE_MODE.get() != "record":
        return
    cache = _ROUTE_CACHE.get()
    if cache is None:
        return
    cache[id(router)].append(indices.detach().contiguous())


def replay_moe_route_indices(router: Any) -> torch.Tensor | None:
    if _ROUTE_MODE.get() != "replay":
        return None
    cache = _ROUTE_CACHE.get()
    if cache is None:
        return None
    entries = cache.get(id(router))
    if not entries:
        raise RuntimeError("MoE route checkpoint replay missing recorded router indices")
    return entries.popleft()
