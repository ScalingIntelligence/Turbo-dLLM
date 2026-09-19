from __future__ import annotations

from types import SimpleNamespace

import torch

from dllm_parallel.core.profiling.system_trace import SystemTrace


class _Range:
    def __init__(self, calls: list[str], name: str) -> None:
        self.calls = calls
        self.name = name

    def __enter__(self) -> None:
        self.calls.append(f"phase_push:{self.name}")

    def __exit__(self, *_args: object) -> None:
        self.calls.append(f"phase_pop:{self.name}")


def _patch_cuda(monkeypatch) -> list[str]:
    calls: list[str] = []
    cudart = SimpleNamespace(
        cudaProfilerStart=lambda: calls.append("profiler_start"),
        cudaProfilerStop=lambda: calls.append("profiler_stop"),
    )
    nvtx = SimpleNamespace(
        range_push=lambda name: calls.append(f"push:{name}"),
        range_pop=lambda: calls.append("pop"),
        range=lambda name: _Range(calls, name),
    )
    monkeypatch.setattr(torch.cuda, "cudart", lambda: cudart)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: calls.append("sync"))
    monkeypatch.setattr(torch.cuda, "nvtx", nvtx)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: False)
    return calls


def test_disabled_system_trace_does_not_touch_cuda(monkeypatch) -> None:
    calls = _patch_cuda(monkeypatch)
    trace = SystemTrace(
        enabled=False,
        backend="nsys",
        output_dir=None,
        start_step=1,
        steps=1,
        device=torch.device("cuda", 0),
        rank=0,
    )
    trace.begin_step(1)
    with trace.phase("forward"):
        pass
    trace.end_step(1)
    assert calls == []


def test_system_trace_bounds_window_and_phases(monkeypatch) -> None:
    calls = _patch_cuda(monkeypatch)
    trace = SystemTrace(
        enabled=True,
        backend="nsys",
        output_dir=None,
        start_step=2,
        steps=2,
        device=torch.device("cuda", 0),
        rank=0,
    )
    trace.begin_step(1)
    trace.end_step(1)
    trace.begin_step(2)
    with trace.phase("forward"):
        pass
    trace.end_step(2)
    trace.begin_step(3)
    trace.end_step(3)

    assert calls == [
        "sync",
        "profiler_start",
        "push:dllm.system_trace",
        "push:dllm.step.2",
        "phase_push:dllm.phase.forward",
        "phase_pop:dllm.phase.forward",
        "pop",
        "push:dllm.step.3",
        "pop",
        "pop",
        "sync",
        "profiler_stop",
    ]
