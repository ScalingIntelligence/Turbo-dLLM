# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Production-facing diffusion-LM parallel training library."""

from __future__ import annotations

from importlib import metadata
from typing import TYPE_CHECKING, Any

from dllm_parallel.core.specs import (
    AttentionPlan,
    BlockDiffusionObjective,
    CPBPPolicy,
    DiffusionSchedule,
    LossRegion,
    ParallelSpec,
    SequenceRegion,
)

if TYPE_CHECKING:
    from dllm_parallel.training import RunSpec

try:
    __version__ = metadata.version("turbo-dllm")
except metadata.PackageNotFoundError:
    __version__ = "0.1.1"

__all__ = [
    "AttentionPlan",
    "BlockDiffusionObjective",
    "CPBPPolicy",
    "DiffusionSchedule",
    "LossRegion",
    "ParallelSpec",
    "SequenceRegion",
    "RunSpec",
    "load_run_spec",
    "train",
    "__version__",
]


def __getattr__(name: str) -> Any:
    if name in {"RunSpec", "load_run_spec", "train"}:
        from dllm_parallel import training

        return getattr(training, name)
    raise AttributeError(name)
