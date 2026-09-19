# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import threading
import time

from dllm_parallel.training.profile_entrypoint import _shutdown_inductor_workers


def test_inductor_worker_shutdown_reports_completion() -> None:
    assert _shutdown_inductor_workers(timeout_seconds=0.1, shutdown=lambda: None)


def test_inductor_worker_shutdown_is_bounded() -> None:
    release = threading.Event()

    def block() -> None:
        release.wait()

    started = time.monotonic()
    try:
        assert not _shutdown_inductor_workers(
            timeout_seconds=0.01,
            shutdown=block,
        )
        assert time.monotonic() - started < 0.5
    finally:
        release.set()
