from __future__ import annotations

import importlib
import sys
import types

import torch


def _deepep_module():
    if "deep_ep" not in sys.modules:
        deep_ep = types.ModuleType("deep_ep")
        deep_ep.Buffer = type("Buffer", (), {})
        deep_ep.ElasticBuffer = type("ElasticBuffer", (), {})
        sys.modules["deep_ep"] = deep_ep
    return importlib.import_module("dllm_parallel.core.parallel.expert.deepep")


def test_collapse_nvlink_routes_matches_index_add_and_gradient() -> None:
    deepep = _deepep_module()
    expert_output = torch.randn(7, 5, dtype=torch.float64, requires_grad=True)
    recv_token_index = torch.tensor([0, 2, 1, 0, 3, 2, 3])
    restore_order = torch.argsort(recv_token_index, stable=True)
    route_lengths = torch.bincount(recv_token_index, minlength=4)

    actual = deepep._collapse_nvlink_routes(
        expert_output,
        restore_order,
        route_lengths,
    )
    expected = torch.zeros(4, 5, dtype=expert_output.dtype)
    expected.index_add_(0, recv_token_index, expert_output)
    torch.testing.assert_close(actual, expected)

    grad = torch.randn_like(actual)
    (actual * grad).sum().backward()
    torch.testing.assert_close(
        expert_output.grad,
        grad.index_select(0, recv_token_index),
    )
