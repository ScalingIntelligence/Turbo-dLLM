# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Checkpoint I/O primitives: rank/world helpers, shard filenames, async handle.

Leaf module for the checkpoint package — no dependency on the other checkpoint
submodules, so format/sharded/__init__ can all import from here without cycles.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import torch.distributed as dist


CHECKPOINT_FORMAT_VERSION = "dllm_parallel.distributed_checkpoint.v1"
METADATA_FILENAME = "metadata.json"
LATEST_TAG = "latest"


def _rank_shard_filename(rank: int, world_size: int) -> str:
    return f"rank_{int(rank):05d}_of_{int(world_size):05d}.pt"


class _AsyncCheckpointHandle:
    def __init__(
        self,
        thread: threading.Thread,
        error_box: dict[str, BaseException],
        on_complete: object | None = None,
    ):
        self._thread = thread
        self._error_box = error_box
        self._on_complete = on_complete
        self._waited = False

    @property
    def done(self) -> bool:
        return not self._thread.is_alive()

    def wait(self) -> None:
        if self._waited:
            return
        self._thread.join()
        self._waited = True
        if callable(self._on_complete):
            self._on_complete(self._error_box.get("error"))
        error = self._error_box.get("error")
        if error is not None:
            raise error


@dataclass(frozen=True)
class CheckpointResult:
    path: Path
    tag: str
    async_save: bool
    async_handle: _AsyncCheckpointHandle | None = None

    def wait(self) -> "CheckpointResult":
        if self.async_handle is not None:
            self.async_handle.wait()
        return self


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return 0


def _world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_world_size())
    return 1


def _barrier(group: object | None = None) -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier(group=group)
