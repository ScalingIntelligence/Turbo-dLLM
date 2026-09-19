# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Run-provenance metadata for reproducible DLLM training jobs."""

from __future__ import annotations

import os
import platform
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

from dllm_parallel.core.parallel.preflight import launch_environment_metadata


_PACKAGE_VERSION_NAMES = (
    "turbo-dllm",
    "torch",
    "transformers",
    "deepspeed",
    "triton",
    "flash-attn",
)


def run_metadata() -> dict[str, Any]:
    git_dirty = _git_dirty()
    if git_dirty is None:
        git_dirty = _environment_bool("DLLM_SOURCE_GIT_DIRTY")
    return {
        "git_sha": _git_output(("rev-parse", "HEAD"))
        or os.environ.get("DLLM_SOURCE_GIT_SHA"),
        "git_branch": _git_output(("rev-parse", "--abbrev-ref", "HEAD"))
        or os.environ.get("DLLM_SOURCE_GIT_BRANCH"),
        "git_dirty": git_dirty,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(),
        "device": device_metadata(),
        "launch_environment": launch_environment_metadata(),
    }


@dataclass
class RunContext:
    """Run-wide metadata shared by trainer, checkpoints, profiling, and logs."""

    run_id: str
    created_unix: float
    resolved_spec: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=run_metadata)
    kernel_metadata: dict[str, Any] = field(default_factory=dict)
    process_group_plan: dict[str, Any] | None = None
    optimizer_policy: dict[str, Any] = field(default_factory=dict)
    checkpoint_policy: dict[str, Any] = field(default_factory=dict)
    profiler_policy: dict[str, Any] = field(default_factory=dict)
    logging_policy: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, *, resolved_spec: dict[str, Any]) -> "RunContext":
        return cls(
            run_id=str(uuid.uuid4()),
            created_unix=time.time(),
            resolved_spec=resolved_spec,
        )

    def with_runtime(
        self,
        *,
        runtime: Any | None,
        kernel_metadata: dict[str, Any] | None = None,
        optimizer_policy: dict[str, Any] | None = None,
        checkpoint_policy: dict[str, Any] | None = None,
        profiler_policy: dict[str, Any] | None = None,
        logging_policy: dict[str, Any] | None = None,
    ) -> "RunContext":
        process_groups = getattr(runtime, "process_groups", None)
        self.process_group_plan = (
            process_groups.to_log_dict()
            if hasattr(process_groups, "to_log_dict")
            else None
        )
        if kernel_metadata is not None:
            self.kernel_metadata = dict(kernel_metadata)
        if optimizer_policy is not None:
            self.optimizer_policy = dict(optimizer_policy)
        if checkpoint_policy is not None:
            self.checkpoint_policy = dict(checkpoint_policy)
        if profiler_policy is not None:
            self.profiler_policy = dict(profiler_policy)
        if logging_policy is not None:
            self.logging_policy = dict(logging_policy)
        return self

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_unix": self.created_unix,
            "resolved_spec": self.resolved_spec,
            "metadata": self.metadata,
            "kernel_metadata": self.kernel_metadata,
            "process_group_plan": self.process_group_plan,
            "optimizer_policy": self.optimizer_policy,
            "checkpoint_policy": self.checkpoint_policy,
            "profiler_policy": self.profiler_policy,
            "logging_policy": self.logging_policy,
        }


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in _PACKAGE_VERSION_NAMES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def device_metadata() -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda_available": False}
        major, minor = torch.cuda.get_device_capability()
        return {
            "cuda_available": True,
            "device_count": int(torch.cuda.device_count()),
            "device_name": torch.cuda.get_device_name(),
            "compute_capability": f"{major}.{minor}",
            "torch_cuda": torch.version.cuda,
        }
    except Exception:
        return {"cuda_available": False}


def _git_output(args: tuple[str, ...]) -> str | None:
    try:
        result = subprocess.run(
            ("git", "-C", str(Path(__file__).resolve().parent), *args),
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    output = result.stdout.strip()
    return output or None


def _git_dirty() -> bool | None:
    status = _git_output(("status", "--porcelain"))
    if status is None:
        return None
    return bool(status)


def _environment_bool(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    return None


__all__ = ["RunContext", "run_metadata", "package_versions", "device_metadata"]
