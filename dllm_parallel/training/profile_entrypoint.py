# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Profiling entrypoint with bounded TorchInductor worker cleanup."""

from __future__ import annotations

import atexit
import json
import os
import threading
from collections.abc import Callable, Sequence


def _configure_dynamo_recompile_limits() -> None:
    """Apply opt-in cache limits for workloads with many valid input shapes."""

    limit_raw = os.environ.get("DLLM_TORCHDYNAMO_RECOMPILE_LIMIT")
    accumulated_raw = os.environ.get("DLLM_TORCHDYNAMO_ACCUMULATED_RECOMPILE_LIMIT")
    if limit_raw is None and accumulated_raw is None:
        return
    import torch._dynamo.config as dynamo_config

    if limit_raw is not None:
        limit = int(limit_raw)
        if limit < 1:
            raise ValueError("DLLM_TORCHDYNAMO_RECOMPILE_LIMIT must be positive")
        dynamo_config.recompile_limit = limit
    if accumulated_raw is not None:
        accumulated = int(accumulated_raw)
        if accumulated < 1:
            raise ValueError(
                "DLLM_TORCHDYNAMO_ACCUMULATED_RECOMPILE_LIMIT must be positive"
            )
        dynamo_config.accumulated_recompile_limit = accumulated


def _shutdown_inductor_workers(
    *,
    timeout_seconds: float = 10.0,
    shutdown: Callable[[], None] | None = None,
) -> bool:
    """Return whether Inductor compile workers stopped within the bound."""

    if shutdown is None:
        try:
            from torch._inductor.async_compile import shutdown_compile_workers
        except (ImportError, AttributeError):
            return True
        # This profile entrypoint takes ownership of the same cleanup below.
        # Removing PyTorch's unbounded atexit retry is what makes the timeout
        # effective when the pinned 2.10 process pool fails to terminate.
        atexit.unregister(shutdown_compile_workers)
        shutdown = shutdown_compile_workers

    error: list[BaseException] = []

    def run_shutdown() -> None:
        try:
            shutdown()
        except BaseException as exc:  # pragma: no cover - dependency guard
            error.append(exc)

    worker = threading.Thread(
        target=run_shutdown,
        name="dllm-inductor-shutdown",
        daemon=True,
    )
    worker.start()
    worker.join(timeout_seconds)
    completed = not worker.is_alive()
    print(
        json.dumps(
            {
                "event": "profile_inductor_shutdown",
                "completed": completed,
                "timeout_seconds": timeout_seconds,
                "error": None if not error else repr(error[0]),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return completed and not error


def main(argv: Sequence[str] | None = None) -> None:
    """Run training, then prevent a known PyTorch 2.10 exit-time stall."""

    _configure_dynamo_recompile_limits()
    from dllm_parallel.training.entrypoint import main as training_main

    try:
        training_main(argv)
    except BaseException:
        # Optimizer, process-group, and artifact cleanup has completed before
        # training_main returns or re-raises. The official Inductor shutdown is
        # started here so its 300-second internal wait cannot block profile exit.
        clean_shutdown = _shutdown_inductor_workers()
        if not clean_shutdown:
            os._exit(1)
        raise
    clean_shutdown = _shutdown_inductor_workers()
    if not clean_shutdown:
        # The compiler subprocesses watch their parent and terminate after it
        # exits. Avoid running Python's remaining finalizers after the known
        # cleanup timeout; all training and distributed cleanup is already done.
        os._exit(0)


if __name__ == "__main__":
    main()
