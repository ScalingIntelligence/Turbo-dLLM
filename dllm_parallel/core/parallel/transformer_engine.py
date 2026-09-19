# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Lifecycle helpers for Transformer Engine tensor-parallel Userbuffers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class _UserbufferState:
    rows: int
    hidden_size: int
    tensor_parallel_size: int
    dtype: torch.dtype


_USERBUFFER_STATE: _UserbufferState | None = None


def ensure_transformer_engine_userbuffers(
    *,
    rows: int,
    hidden_size: int,
    tensor_parallel_size: int,
    dtype: torch.dtype,
) -> bool:
    """Initialize TE's process-global TP overlap buffers once.

    ``rows`` is the exact global token-row count expected by every overlapped
    projection. Transformer Engine's Userbuffer kernels require the local
    input numel to match the initialized buffer view exactly.

    Returns ``True`` only for the call that performs initialization.
    """

    if min(int(rows), int(hidden_size), int(tensor_parallel_size)) <= 0:
        raise ValueError("Transformer Engine Userbuffer dimensions must be positive")
    requested = _UserbufferState(
        rows=int(rows),
        hidden_size=int(hidden_size),
        tensor_parallel_size=int(tensor_parallel_size),
        dtype=dtype,
    )
    global _USERBUFFER_STATE
    if _USERBUFFER_STATE is not None:
        compatible = (
            requested.rows == _USERBUFFER_STATE.rows
            and requested.hidden_size == _USERBUFFER_STATE.hidden_size
            and requested.tensor_parallel_size
            == _USERBUFFER_STATE.tensor_parallel_size
            and requested.dtype == _USERBUFFER_STATE.dtype
        )
        if not compatible:
            raise RuntimeError(
                "Transformer Engine Userbuffers are already initialized with an "
                f"incompatible shape or TP domain: current={_USERBUFFER_STATE}, "
                f"requested={requested}"
            )
        return False

    try:
        from transformer_engine.pytorch import (
            UserBufferQuantizationMode,
            initialize_ub,
        )
    except Exception as exc:  # pragma: no cover - depends on production CUDA image.
        raise RuntimeError(
            "tensor_parallel_overlap requires Transformer Engine Userbuffers"
        ) from exc

    initialize_ub(
        shape=[requested.rows, requested.hidden_size],
        tp_size=requested.tensor_parallel_size,
        quantization_modes=[UserBufferQuantizationMode.NONE],
        dtype=requested.dtype,
        bootstrap_backend="nccl",
    )
    _USERBUFFER_STATE = requested
    return True


def destroy_transformer_engine_userbuffers() -> None:
    """Release TE's process-global overlap buffers before process-group teardown."""

    global _USERBUFFER_STATE
    if _USERBUFFER_STATE is None:
        return
    try:
        from transformer_engine.pytorch import destroy_ub

        destroy_ub()
    finally:
        _USERBUFFER_STATE = None


def transformer_engine_userbuffer_state() -> dict[str, Any] | None:
    """Return serializable diagnostics for the active Userbuffer allocation."""

    if _USERBUFFER_STATE is None:
        return None
    return {
        "rows": _USERBUFFER_STATE.rows,
        "hidden_size": _USERBUFFER_STATE.hidden_size,
        "tensor_parallel_size": _USERBUFFER_STATE.tensor_parallel_size,
        "dtype": str(_USERBUFFER_STATE.dtype),
    }


__all__ = [
    "destroy_transformer_engine_userbuffers",
    "ensure_transformer_engine_userbuffers",
    "transformer_engine_userbuffer_state",
]
