# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Fault-tolerance helpers for production launches."""

from __future__ import annotations

import faulthandler
import json
import os
import signal
from pathlib import Path
from typing import Any


def install_fault_handlers() -> None:
    """Install low-overhead signal handlers for stack diagnostics."""

    for signum in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        try:
            faulthandler.register(signum, all_threads=True, chain=True)
        except Exception:
            pass


def write_failure_marker(
    *,
    run_dir: str | os.PathLike[str] | None,
    payload: dict[str, Any],
) -> None:
    """Write a rank-local fatal marker for launchers and postmortem tooling."""

    if not run_dir:
        return
    try:
        path = Path(run_dir)
        path.mkdir(parents=True, exist_ok=True)
        rank = int(payload.get("rank", os.environ.get("RANK", 0)))
        marker = path / f"fatal_rank_{rank:05d}.json"
        tmp = marker.with_suffix(marker.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, marker)
    except Exception:
        pass
