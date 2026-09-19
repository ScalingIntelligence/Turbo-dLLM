from __future__ import annotations

import importlib.util
import threading

import torch

import dllm_parallel.core.profiling.operator_trace as operator_trace_module

from dllm_parallel.core.profiling.operator_trace import (
    communication_scope,
    enable_operator_tracing,
    operator_scope,
    reset_operator_tracing,
)


class _Range:
    def __init__(self, calls: list[str], name: str) -> None:
        self.calls = calls
        self.name = name

    def __enter__(self) -> None:
        self.calls.append(f"enter:{self.name}")

    def __exit__(self, *_args: object) -> None:
        self.calls.append(f"exit:{self.name}")


def test_operator_trace_is_visible_to_autograd_worker_threads(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        torch.profiler,
        "record_function",
        lambda name: _Range(calls, name),
    )

    def run_operator() -> None:
        with operator_scope("attention.full"):
            pass

    previous = enable_operator_tracing("kineto")
    try:
        worker = threading.Thread(target=run_operator)
        worker.start()
        worker.join()
    finally:
        reset_operator_tracing(previous)

    assert calls == [
        "enter:dllm.operator.attention.full",
        "exit:dllm.operator.attention.full",
    ]


def test_communication_scope_records_runtime_byte_metadata(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        torch.profiler,
        "record_function",
        lambda name: _Range(calls, name),
    )

    previous = enable_operator_tracing("kineto")
    try:
        with communication_scope(
            domain="attention",
            phase="forward",
            collective="all_gather_into_tensor",
            input_bytes=1024,
            logical_bytes=3072,
        ):
            pass
    finally:
        reset_operator_tracing(previous)

    label = (
        "dllm.communication.domain=attention;phase=forward;"
        "collective=all_gather_into_tensor;input_bytes=1024;"
        "logical_bytes=3072"
    )
    assert calls == [f"enter:{label}", f"exit:{label}"]


def test_trace_state_is_shared_across_duplicate_module_loads(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        torch.profiler,
        "record_function",
        lambda name: _Range(calls, name),
    )
    spec = importlib.util.spec_from_file_location(
        "_dllm_operator_trace_duplicate",
        operator_trace_module.__file__,
    )
    assert spec is not None and spec.loader is not None
    duplicate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(duplicate)

    previous = enable_operator_tracing("kineto")
    try:
        with duplicate.communication_scope(
            domain="attention",
            phase="backward",
            collective="reduce_scatter_tensor",
            input_bytes=2048,
            logical_bytes=1024,
        ):
            pass
    finally:
        reset_operator_tracing(previous)

    assert calls == [
        "enter:dllm.communication.domain=attention;phase=backward;"
        "collective=reduce_scatter_tensor;input_bytes=2048;logical_bytes=1024",
        "exit:dllm.communication.domain=attention;phase=backward;"
        "collective=reduce_scatter_tensor;input_bytes=2048;logical_bytes=1024",
    ]


def test_active_kineto_profiler_is_used_as_process_wide_fallback(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.delenv("DLLM_INTERNAL_OPERATOR_TRACE_BACKEND", raising=False)
    monkeypatch.setattr(torch.autograd, "_profiler_enabled", lambda: True)
    monkeypatch.setattr(
        torch.profiler,
        "record_function",
        lambda name: _Range(calls, name),
    )

    with communication_scope(
        domain="attention",
        phase="forward",
        collective="all_gather_into_tensor",
        input_bytes=1024,
        logical_bytes=512,
    ):
        pass

    assert calls == [
        "enter:dllm.communication.domain=attention;phase=forward;"
        "collective=all_gather_into_tensor;input_bytes=1024;logical_bytes=512",
        "exit:dllm.communication.domain=attention;phase=forward;"
        "collective=all_gather_into_tensor;input_bytes=1024;logical_bytes=512",
    ]
