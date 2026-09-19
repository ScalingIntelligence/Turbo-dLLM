# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""FSDP / DeepSpeed sharded state-dict glue for checkpointing.

Wraps the FSDP helpers so save/load can treat FSDP-wrapped models and plain
modules uniformly. Topology validation / resharding rejection lives in
:mod:`dllm_parallel.core.checkpoint.format`.
"""

from __future__ import annotations

import contextlib
from typing import Any

import torch

from dllm_parallel.core.optim.fsdp import (
    fsdp2_state_dicts,
    fsdp_optimizer_state_dict,
    fsdp_sharded_state_dict_context,
    is_fsdp2_wrapped_model,
    is_fsdp_wrapped_model,
    load_fsdp2_state_dicts,
    load_fsdp_optimizer_state_dict,
)


def _checkpoint_state_dicts(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if is_fsdp2_wrapped_model(model):
        return fsdp2_state_dicts(model, optimizer)
    with _checkpoint_state_dict_context(model, optimizer):
        return model.state_dict(), _optimizer_state_dict(model, optimizer)


def _load_checkpoint_state_dicts(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    model_state: dict[str, Any],
    optimizer_state: dict[str, Any] | None,
    *,
    strict: bool,
) -> None:
    if is_fsdp2_wrapped_model(model):
        load_fsdp2_state_dicts(
            model,
            optimizer,
            model_state,
            optimizer_state,
            strict=bool(strict),
        )
        return
    with _checkpoint_state_dict_context(model, optimizer):
        model.load_state_dict(model_state, strict=bool(strict))
        _load_optimizer_state_dict(model, optimizer, optimizer_state)


def _module(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "module", model)


def _state_dict_module(model: torch.nn.Module) -> torch.nn.Module:
    if is_fsdp_wrapped_model(model):
        return model
    return _module(model)


def _checkpoint_backend(model: torch.nn.Module) -> str:
    if is_fsdp_wrapped_model(model):
        return "fsdp_sharded_rank_local"
    return "rank_local"


def _checkpoint_state_dict_context(model: torch.nn.Module, optimizer: Any | None):
    if is_fsdp_wrapped_model(model):
        return fsdp_sharded_state_dict_context(model, optimizer)
    return contextlib.nullcontext()


def _optimizer_state_dict(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, Any] | None:
    if optimizer is None:
        return None
    if is_fsdp_wrapped_model(model):
        return fsdp_optimizer_state_dict(model, optimizer)
    return optimizer.state_dict()


def _load_optimizer_state_dict(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    state: dict[str, Any] | None,
) -> None:
    if optimizer is None or state is None:
        return
    if is_fsdp_wrapped_model(model):
        load_fsdp_optimizer_state_dict(model, optimizer, state)
        return
    optimizer.load_state_dict(state)
