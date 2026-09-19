# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Trainer checkpoint policy helpers.

The checkpoint package owns serialization. This module owns when the trainer
loads/saves checkpoints from the typed ``RunSpec``.
"""

from __future__ import annotations

from typing import Any

from dllm_parallel.core.checkpoint import checkpoint_exists
from dllm_parallel.training.run_spec import RunSpec


def resolve_load_checkpoint(
    spec: RunSpec,
) -> tuple[str | None, str, str]:
    """Return ``(path, tag, reason)`` for the checkpoint to load, if any."""

    checkpoint = spec.checkpointing
    if checkpoint.load_checkpoint_dir:
        return (
            str(checkpoint.load_checkpoint_dir),
            str(checkpoint.checkpoint_tag),
            "explicit",
        )
    if (
        checkpoint.auto_resume
        and checkpoint.save_checkpoint_dir
        and checkpoint_exists(
            checkpoint.save_checkpoint_dir,
            tag=str(checkpoint.checkpoint_tag),
        )
    ):
        return (
            str(checkpoint.save_checkpoint_dir),
            str(checkpoint.checkpoint_tag),
            "auto_resume",
        )
    return None, str(checkpoint.checkpoint_tag), "none"


def should_save_checkpoint(
    *,
    spec: RunSpec,
    global_step: int,
    target_total_steps: int | None = None,
    training_complete: bool = False,
) -> bool:
    checkpoint = spec.checkpointing
    if checkpoint.save_checkpoint_dir is None:
        return False
    global_step = int(global_step)
    if bool(checkpoint.save_first_step) and global_step == 1:
        return True
    interval = int(checkpoint.save_checkpoint_interval)
    if interval > 0 and global_step % interval == 0:
        return True
    if bool(checkpoint.save_final) and (
        bool(training_complete)
        or (target_total_steps is not None and global_step >= int(target_total_steps))
    ):
        return True
    return False


def due_duration_checkpoint_fractions(
    *,
    spec: RunSpec,
    elapsed_seconds: float,
    saved_fractions: set[float] | frozenset[float],
) -> tuple[float, ...]:
    """Return elapsed-time checkpoints that have become due, in order."""

    duration = spec.training.max_duration_seconds
    if duration is None or float(duration) <= 0.0:
        return ()
    elapsed_fraction = max(0.0, float(elapsed_seconds)) / float(duration)
    return tuple(
        float(fraction)
        for fraction in spec.checkpointing.save_duration_fractions
        if float(fraction) not in saved_fractions
        and elapsed_fraction >= float(fraction)
    )


def checkpoint_log_payload(
    *,
    event: str,
    path: str,
    tag: str,
    step: int,
    **extra: Any,
) -> dict[str, Any]:
    payload = {
        "event": str(event),
        "path": str(path),
        "tag": str(tag),
        "step": int(step),
    }
    payload.update(extra)
    return payload
