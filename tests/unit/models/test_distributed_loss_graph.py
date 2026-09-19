# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import torch
from torch import nn

from dllm_parallel.core.models.loss import zero_loss_with_module_parameters


def test_zero_loss_retains_hidden_and_parameter_gradients() -> None:
    source = torch.randn(2, 3, requires_grad=True)
    hidden = source[:0]
    head = nn.Linear(3, 5)

    loss = zero_loss_with_module_parameters(hidden, head)
    loss.backward()

    assert loss.item() == 0.0
    assert source.grad is not None
    assert torch.count_nonzero(source.grad) == 0
    for parameter in head.parameters():
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad) == 0
