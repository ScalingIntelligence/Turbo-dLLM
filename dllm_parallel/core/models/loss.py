# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Loss-graph utilities shared by distributed model backbones."""

from __future__ import annotations

import torch
from torch import nn


def zero_loss_with_module_parameters(
    hidden_states: torch.Tensor,
    module: nn.Module,
) -> torch.Tensor:
    """Return exact zero while retaining the rank's full parameter graph.

    Block ownership can leave a rank with no supervised tokens. Distributed
    optimizers still require every rank to mark the same parameters ready, so
    the empty rank retains scalar dependencies without evaluating logits.
    """

    if hidden_states.numel() == 0:
        loss = hidden_states.sum(dtype=torch.float32) * 0.0
    else:
        loss = hidden_states.reshape(-1)[0].float() * 0.0
    for parameter in module.parameters():
        if parameter.requires_grad and parameter.numel() > 0:
            loss = loss + parameter.reshape(-1)[0].float() * 0.0
    return loss
