# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Execution contract consumed by the canonical training engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class DistributedExecution(Protocol):
    """Select distributed model execution without forking the training loop."""

    name: str

    def active_block_mode(self, *, block_parallel_size: int) -> str:
        """Return the runtime ownership mode for the resolved topology."""

    def build_distributed_model(
        self,
        executor: Any,
        model: Any,
        *,
        runtime: Any,
        spec: Any,
    ) -> Any:
        """Build the model used by the configured distributed topology."""


@dataclass(frozen=True)
class ProductionDistributedExecution:
    """The only execution policy exposed by the production package."""

    name: str = "production"

    def active_block_mode(self, *, block_parallel_size: int) -> str:
        return "dual_end" if int(block_parallel_size) > 1 else "all_blocks"

    def build_distributed_model(
        self,
        executor: Any,
        model: Any,
        *,
        runtime: Any,
        spec: Any,
    ) -> Any:
        return executor.build_training_model(model, runtime=runtime, spec=spec)


PRODUCTION_EXECUTION = ProductionDistributedExecution()


__all__ = [
    "DistributedExecution",
    "PRODUCTION_EXECUTION",
    "ProductionDistributedExecution",
]
