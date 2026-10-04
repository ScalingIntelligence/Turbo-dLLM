# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Transformer Engine loading and tensor-parallel Userbuffer lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import importlib
from importlib import metadata
from pathlib import Path, PurePosixPath
import sys
from threading import RLock
from types import ModuleType
from typing import Any

import torch


_CUDNN_HANDLES: dict[Path, Any] = {}
_CUDNN_LOCK = RLock()


def load_transformer_engine(module: str = "transformer_engine.pytorch") -> ModuleType:
    """Import TE with the cuDNN libraries installed for PyTorch's CUDA major.

    TE's native library needs cuDNN components that PyTorch does not itself
    load. Python wheel directories are absent from the system loader's search
    path, so preload the installed distribution's libraries by absolute path.
    Keep their handles alive and retry dependency ordering without altering
    process search paths. CPU and system-cuDNN installations import normally.
    """
    if sys.platform == "linux" and torch.version.cuda:
        try:
            distribution = metadata.distribution(
                f"nvidia-cudnn-cu{torch.version.cuda.split('.')[0]}"
            )
        except metadata.PackageNotFoundError:
            distribution = None
        if distribution is not None:
            paths = sorted(
                {
                    Path(distribution.locate_file(file)).resolve()
                    for file in distribution.files or ()
                    if PurePosixPath(str(file)).parent
                    == PurePosixPath("nvidia/cudnn/lib")
                    and PurePosixPath(str(file)).match("libcudnn*.so.9")
                    and Path(distribution.locate_file(file)).is_file()
                }
            )
            with _CUDNN_LOCK:
                pending = [path for path in paths if path not in _CUDNN_HANDLES]
                while pending:
                    failures: dict[Path, OSError] = {}
                    for path in pending:
                        try:
                            _CUDNN_HANDLES[path] = ctypes.CDLL(
                                str(path), mode=ctypes.RTLD_GLOBAL
                            )
                        except OSError as error:
                            failures[path] = error
                    if len(failures) == len(pending):
                        names = ", ".join(path.name for path in failures)
                        raise RuntimeError(
                            f"Cannot load packaged cuDNN libraries for Transformer Engine: {names}"
                        ) from next(iter(failures.values()))
                    pending = list(failures)
    return importlib.import_module(module)


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
            and requested.tensor_parallel_size == _USERBUFFER_STATE.tensor_parallel_size
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
        te = load_transformer_engine()
    except Exception as exc:  # pragma: no cover - depends on production CUDA image.
        raise RuntimeError(
            "tensor_parallel_overlap requires Transformer Engine Userbuffers"
        ) from exc

    te.initialize_ub(
        shape=[requested.rows, requested.hidden_size],
        tp_size=requested.tensor_parallel_size,
        quantization_modes=[te.UserBufferQuantizationMode.NONE],
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
        load_transformer_engine().destroy_ub()
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
    "load_transformer_engine",
    "destroy_transformer_engine_userbuffers",
    "ensure_transformer_engine_userbuffers",
    "transformer_engine_userbuffer_state",
]
