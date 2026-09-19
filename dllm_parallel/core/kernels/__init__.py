# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Native kernels and fused objective helpers.

Package-level function exports are lightweight proxies so public namespace
imports do not initialize torch or native-extension state.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "chunked_linear_ce_native",
    "cp_fusion",
    "configure_kernel_runtime",
    "frozen_linear_ce_topk",
    "kernel_runtime_policy",
    "tiled_linear_cross_entropy",
]


def frozen_linear_ce_topk(*args: Any, **kwargs: Any) -> Any:
    from dllm_parallel.core.kernels.dflash_linear_topk import (
        frozen_linear_ce_topk as impl,
    )

    return impl(*args, **kwargs)


def tiled_linear_cross_entropy(*args: Any, **kwargs: Any) -> Any:
    from dllm_parallel.core.kernels.tiled_linear_cross_entropy import (
        tiled_linear_cross_entropy as impl,
    )

    return impl(*args, **kwargs)


def configure_kernel_runtime(*args: Any, **kwargs: Any) -> Any:
    from dllm_parallel.core.kernels.runtime import configure_kernel_runtime as impl

    return impl(*args, **kwargs)


def kernel_runtime_policy(*args: Any, **kwargs: Any) -> Any:
    from dllm_parallel.core.kernels.runtime import kernel_runtime_policy as impl

    return impl(*args, **kwargs)


def __getattr__(name: str) -> Any:
    if name == "chunked_linear_ce_native":
        return importlib.import_module("dllm_parallel.core.kernels.chunked_linear_ce_native")
    if name == "cp_fusion":
        return importlib.import_module("dllm_parallel.core.kernels.cp_fusion")
    raise AttributeError(name)
