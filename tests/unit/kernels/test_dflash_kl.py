from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from dllm_parallel.core.kernels.dflash_kl import frozen_linear_kl
from dllm_parallel.core.kernels.tiled_linear_cross_entropy import (
    tiled_linear_cross_entropy,
)


def test_frozen_linear_kl_matches_dense_reference() -> None:
    torch.manual_seed(17)
    draft_hidden = torch.randn(7, 5, requires_grad=True)
    teacher_hidden = torch.randn(7, 6)
    draft_weight = torch.randn(11, 5)
    teacher_weight = torch.randn(11, 6)
    scale = torch.randn(7)

    observed = frozen_linear_kl(
        draft_hidden,
        draft_weight,
        teacher_hidden,
        teacher_weight,
    )
    observed_gradient = torch.autograd.grad((observed * scale).sum(), draft_hidden)[0]

    reference_hidden = draft_hidden.detach().requires_grad_(True)
    draft_logits = F.linear(reference_hidden, draft_weight)
    teacher_logits = F.linear(teacher_hidden, teacher_weight)
    expected = F.kl_div(
        F.log_softmax(draft_logits, dim=-1, dtype=torch.float32),
        F.softmax(teacher_logits, dim=-1, dtype=torch.float32),
        reduction="none",
    ).sum(dim=-1)
    expected_gradient = torch.autograd.grad((expected * scale).sum(), reference_hidden)[0]

    assert torch.allclose(observed, expected)
    assert torch.allclose(observed_gradient, expected_gradient)


def test_frozen_linear_kl_softcap_matches_dense_reference() -> None:
    torch.manual_seed(23)
    softcap = 2.5
    draft_hidden = torch.randn(5, 7, requires_grad=True)
    teacher_hidden = torch.randn(5, 9)
    draft_weight = torch.randn(13, 7)
    teacher_weight = torch.randn(13, 9)
    scale = torch.randn(5)

    observed = frozen_linear_kl(
        draft_hidden,
        draft_weight,
        teacher_hidden,
        teacher_weight,
        logit_softcap=softcap,
    )
    observed_gradient = torch.autograd.grad((observed * scale).sum(), draft_hidden)[0]

    reference_hidden = draft_hidden.detach().requires_grad_(True)
    draft_logits = F.linear(reference_hidden, draft_weight)
    teacher_logits = F.linear(teacher_hidden, teacher_weight)
    draft_logits = softcap * torch.tanh(draft_logits / softcap)
    teacher_logits = softcap * torch.tanh(teacher_logits / softcap)
    expected = F.kl_div(
        F.log_softmax(draft_logits, dim=-1, dtype=torch.float32),
        F.softmax(teacher_logits, dim=-1, dtype=torch.float32),
        reduction="none",
    ).sum(dim=-1)
    expected_gradient = torch.autograd.grad((expected * scale).sum(), reference_hidden)[0]

    torch.testing.assert_close(observed, expected)
    torch.testing.assert_close(observed_gradient, expected_gradient)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_frozen_linear_kl_cuda_matches_dense_forward_and_backward() -> None:
    torch.manual_seed(31)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    draft_hidden = torch.randn(
        7, 32, device=device, dtype=dtype, requires_grad=True
    )
    teacher_hidden = torch.randn(7, 48, device=device, dtype=dtype)
    draft_weight = torch.randn(257, 32, device=device, dtype=dtype)
    teacher_weight = torch.randn(257, 48, device=device, dtype=dtype)
    scale = torch.randn(7, device=device, dtype=torch.float32)

    observed = frozen_linear_kl(
        draft_hidden,
        draft_weight,
        teacher_hidden,
        teacher_weight,
    )
    observed_gradient = torch.autograd.grad(
        (observed * scale).sum(), draft_hidden
    )[0]

    reference_hidden = draft_hidden.detach().clone().requires_grad_(True)
    draft_logits = F.linear(reference_hidden, draft_weight)
    teacher_logits = F.linear(teacher_hidden, teacher_weight)
    expected = F.kl_div(
        F.log_softmax(draft_logits, dim=-1, dtype=torch.float32),
        F.softmax(teacher_logits, dim=-1, dtype=torch.float32),
        reduction="none",
    ).sum(dim=-1)
    expected_gradient = torch.autograd.grad(
        (expected * scale).sum(), reference_hidden
    )[0]

    torch.testing.assert_close(observed, expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(
        observed_gradient.float(),
        expected_gradient.float(),
        atol=4e-2,
        rtol=4e-2,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_frozen_linear_kl_cuda_softcap_matches_dense_forward_and_backward() -> None:
    torch.manual_seed(37)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    softcap = 3.0
    draft_hidden = torch.randn(7, 32, device=device, dtype=dtype, requires_grad=True)
    teacher_hidden = torch.randn(7, 48, device=device, dtype=dtype)
    draft_weight = torch.randn(257, 32, device=device, dtype=dtype)
    teacher_weight = torch.randn(257, 48, device=device, dtype=dtype)
    scale = torch.randn(7, device=device, dtype=torch.float32)

    observed = frozen_linear_kl(
        draft_hidden,
        draft_weight,
        teacher_hidden,
        teacher_weight,
        logit_softcap=softcap,
    )
    observed_gradient = torch.autograd.grad((observed * scale).sum(), draft_hidden)[0]

    reference_hidden = draft_hidden.detach().clone().requires_grad_(True)
    draft_logits = F.linear(reference_hidden, draft_weight)
    teacher_logits = F.linear(teacher_hidden, teacher_weight)
    draft_logits = softcap * torch.tanh(draft_logits / softcap)
    teacher_logits = softcap * torch.tanh(teacher_logits / softcap)
    expected = F.kl_div(
        F.log_softmax(draft_logits, dim=-1, dtype=torch.float32),
        F.softmax(teacher_logits, dim=-1, dtype=torch.float32),
        reduction="none",
    ).sum(dim=-1)
    expected_gradient = torch.autograd.grad((expected * scale).sum(), reference_hidden)[0]

    torch.testing.assert_close(observed, expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(
        observed_gradient.float(),
        expected_gradient.float(),
        atol=4e-2,
        rtol=4e-2,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_frozen_linear_ce_cuda_matches_dense_forward_and_backward() -> None:
    torch.manual_seed(47)
    device = torch.device("cuda")
    hidden = torch.randn(
        13,
        64,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    weight = torch.randn(521, 64, device=device, dtype=torch.bfloat16)
    labels = torch.randint(0, weight.shape[0], (hidden.shape[0],), device=device)

    observed = tiled_linear_cross_entropy(
        hidden,
        weight,
        labels,
        reduction="mean",
        dtype=hidden.dtype,
        weight_layout="vocab_first",
    )
    observed_gradient = torch.autograd.grad(observed, hidden)[0]

    reference_hidden = hidden.detach().clone().requires_grad_(True)
    expected = F.cross_entropy(
        F.linear(reference_hidden.float(), weight.float()),
        labels,
    )
    expected_gradient = torch.autograd.grad(expected, reference_hidden)[0]

    torch.testing.assert_close(observed, expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(
        observed_gradient.float(),
        expected_gradient.float(),
        atol=4e-2,
        rtol=4e-2,
    )
