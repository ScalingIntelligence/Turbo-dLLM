# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Profiler-only model-operator ranges.

The ranges are disabled outside a bounded system trace so production profiles
do not emit per-layer annotations on every optimizer step.
"""

from __future__ import annotations

import contextlib
import os
import re
from contextlib import AbstractContextManager
from functools import wraps
from typing import Any, Callable, TypeVar


_TRACE_BACKEND_ENV = "DLLM_INTERNAL_OPERATOR_TRACE_BACKEND"
_NULL_RANGE = contextlib.nullcontext()
_F = TypeVar("_F", bound=Callable[..., Any])
OperatorName = str | Callable[..., str]
_COMMUNICATION_FIELD_RE = re.compile(r"^[a-z0-9_]+$")


def enable_operator_tracing(backend: str) -> str | None:
    """Enable annotations for the trainer and its autograd worker threads."""

    if backend not in {"kineto", "nsys"}:
        raise ValueError("operator trace backend must be kineto or nsys")
    previous = os.environ.get(_TRACE_BACKEND_ENV)
    os.environ[_TRACE_BACKEND_ENV] = backend
    return previous


def reset_operator_tracing(previous: str | None) -> None:
    """Restore the process-wide operator-tracing state."""

    if previous is None:
        os.environ.pop(_TRACE_BACKEND_ENV, None)
    else:
        os.environ[_TRACE_BACKEND_ENV] = previous


def _active_trace_backend() -> str | None:
    """Read trace state shared by every loaded copy of this module."""

    backend = os.environ.get(_TRACE_BACKEND_ENV)
    if backend in {"kineto", "nsys"}:
        return backend
    # Editable installs and mounted profiler images can load the trainer and
    # attention code from different copies of the package. Kineto itself is
    # process-wide, so use its active state as the authoritative fallback.
    import torch

    profiler_enabled = getattr(torch.autograd, "_profiler_enabled", None)
    if profiler_enabled is not None and profiler_enabled():
        return "kineto"
    return None


def operator_scope(name: str) -> AbstractContextManager[Any]:
    """Return a profiler range for one model operator when tracing is active."""

    backend = _active_trace_backend()
    if backend is None:
        return _NULL_RANGE
    label = f"dllm.operator.{name}"
    if backend == "kineto":
        from torch.profiler import record_function

        return record_function(label)
    import torch

    return torch.cuda.nvtx.range(label)


def communication_scope(
    *,
    domain: str,
    phase: str,
    collective: str,
    input_bytes: int,
    logical_bytes: int,
) -> AbstractContextManager[Any]:
    """Annotate one distributed operation with its runtime tensor volume.

    ``input_bytes`` is the size of the tensor supplied to the collective on the
    current rank. ``logical_bytes`` counts the subset sent to other ranks and
    excludes the local contribution. The annotation exists only inside a
    bounded system-trace window, leaving untraced training unchanged.
    """

    backend = _active_trace_backend()
    if backend is None:
        return _NULL_RANGE
    fields = {
        "domain": domain,
        "phase": phase,
        "collective": collective,
    }
    for field, value in fields.items():
        if not _COMMUNICATION_FIELD_RE.fullmatch(value):
            raise ValueError(
                f"communication {field} must contain lowercase letters, digits, "
                "and underscores"
            )
    if input_bytes < 0 or logical_bytes < 0:
        raise ValueError("communication byte counts must be non-negative")
    label = (
        "dllm.communication."
        f"domain={domain};phase={phase};collective={collective};"
        f"input_bytes={int(input_bytes)};logical_bytes={int(logical_bytes)}"
    )
    if backend == "kineto":
        from torch.profiler import record_function

        return record_function(label)
    import torch

    return torch.cuda.nvtx.range(label)


def traced_operator(name: OperatorName) -> Callable[[_F], _F]:
    """Annotate a model method only while a bounded system trace is active."""

    def decorate(function: _F) -> _F:
        @wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            resolved = name(*args, **kwargs) if callable(name) else name
            with operator_scope(resolved):
                return function(*args, **kwargs)

        return wrapped  # type: ignore[return-value]

    return decorate
