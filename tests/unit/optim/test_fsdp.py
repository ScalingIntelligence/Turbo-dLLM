# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import torch

from dllm_parallel.core.optim.fsdp import FSDPTrainingModule, _fsdp2_forward_methods


class _Task:
    def forward(self, model: torch.nn.Module, prepared: torch.Tensor) -> torch.Tensor:
        return model(prepared)

    def loss(
        self,
        model: torch.nn.Module,
        prepared: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        del model, prepared
        return output.square().sum()


def test_fsdp_training_module_keeps_forward_and_loss_in_one_call() -> None:
    model = torch.nn.Linear(3, 2, bias=False)
    wrapped = FSDPTrainingModule(model, _Task())
    inputs = torch.randn(4, 3)

    loss = wrapped(inputs)
    loss.backward()

    expected = model(inputs).square().sum()
    assert torch.allclose(loss.detach(), expected.detach())
    assert model.weight.grad is not None


class _CustomForwardModule(torch.nn.Module):
    __fsdp_forward_methods__ = ("packed_forward",)

    def packed_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


def test_fsdp2_custom_forward_methods_are_declared_by_the_module() -> None:
    assert _fsdp2_forward_methods(_CustomForwardModule()) == ("packed_forward",)


def test_fsdp2_custom_forward_methods_reject_missing_methods() -> None:
    module = _CustomForwardModule()
    module.__fsdp_forward_methods__ = ("missing",)
    try:
        _fsdp2_forward_methods(module)
    except ValueError as exc:
        assert "not callable" in str(exc)
    else:
        raise AssertionError("missing FSDP2 forward method was accepted")
