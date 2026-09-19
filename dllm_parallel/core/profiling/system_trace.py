# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Bounded GPU-system tracing for the canonical trainer.

Kineto provides portable CUPTI traces from every rank. Nsight Systems remains
available for native clusters where the external profiler can observe device
kernels. Both backends are inert unless explicitly enabled.
"""

from __future__ import annotations

import contextlib
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from dllm_parallel.core.profiling.operator_trace import (
    enable_operator_tracing,
    reset_operator_tracing,
)

TRACE_WINDOW_NAME = "dllm.system_trace"
_NULL_RANGE = contextlib.nullcontext()


class SystemTrace:
    """Capture one bounded optimizer-step window on every distributed rank."""

    def __init__(
        self,
        *,
        enabled: bool,
        backend: str,
        output_dir: str | None,
        start_step: int,
        steps: int,
        device: torch.device,
        rank: int,
    ) -> None:
        if backend not in {"kineto", "nsys"}:
            raise ValueError("system trace backend must be kineto or nsys")
        self._enabled = bool(enabled)
        self._backend = backend
        self._output_dir = Path(output_dir) if output_dir else None
        self._start_step = int(start_step)
        self._stop_step = self._start_step + int(steps)
        self._device = device
        self._rank = int(rank)
        self._step_range: AbstractContextManager[Any] | None = None
        self._window_range: AbstractContextManager[Any] | None = None
        self._profiler: Any | None = None
        self._step_open = False
        self._operator_trace_previous: str | None = None
        self._operator_trace_active = False

    @property
    def active(self) -> bool:
        return self._step_open

    def begin_step(self, step: int) -> None:
        if not self._enabled or not self._start_step <= step < self._stop_step:
            return
        if step == self._start_step:
            self._barrier()
            torch.cuda.synchronize(self._device)
            if self._backend == "kineto":
                self._start_kineto()
                self._window_range = self._record_function(TRACE_WINDOW_NAME)
                self._window_range.__enter__()
            else:
                torch.cuda.cudart().cudaProfilerStart()
                torch.cuda.nvtx.range_push(TRACE_WINDOW_NAME)
        if self._backend == "kineto":
            self._step_range = self._record_function(f"dllm.step.{step}")
            self._step_range.__enter__()
        else:
            torch.cuda.nvtx.range_push(f"dllm.step.{step}")
        self._operator_trace_previous = enable_operator_tracing(self._backend)
        self._operator_trace_active = True
        self._step_open = True

    def phase(self, name: str) -> AbstractContextManager[Any]:
        if not self._step_open:
            return _NULL_RANGE
        label = f"dllm.phase.{name}"
        if self._backend == "kineto":
            return self._record_function(label)
        return torch.cuda.nvtx.range(label)

    def end_step(self, step: int) -> None:
        if not self._step_open:
            return
        if self._backend == "kineto":
            assert self._step_range is not None
            self._step_range.__exit__(None, None, None)
            self._step_range = None
        else:
            torch.cuda.nvtx.range_pop()
        assert self._operator_trace_active
        reset_operator_tracing(self._operator_trace_previous)
        self._operator_trace_previous = None
        self._operator_trace_active = False
        self._step_open = False
        if step + 1 != self._stop_step:
            return

        if self._backend == "kineto":
            assert self._window_range is not None
            self._window_range.__exit__(None, None, None)
            self._window_range = None
            torch.cuda.synchronize(self._device)
            self._stop_kineto()
        else:
            torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize(self._device)
            torch.cuda.cudart().cudaProfilerStop()
        self._barrier()

    def _start_kineto(self) -> None:
        if self._output_dir is None:
            raise ValueError("Kineto system tracing requires profiler.system_trace_dir")
        from torch.profiler import ProfilerActivity, profile

        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_flops=False,
            acc_events=True,
        )
        self._profiler.start()

    def _stop_kineto(self) -> None:
        assert self._profiler is not None
        profiler = self._profiler
        self._profiler = None
        profiler.stop()
        assert self._output_dir is not None
        prefix = self._output_dir / f"rank_{self._rank}"
        profiler.export_chrome_trace(str(prefix.with_suffix(".trace.json.gz")))

    @staticmethod
    def _record_function(name: str) -> AbstractContextManager[Any]:
        from torch.profiler import record_function

        return record_function(name)

    @staticmethod
    def _barrier() -> None:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
