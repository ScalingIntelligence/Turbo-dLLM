# Copyright 2026 The bdlm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import pytest
import subprocess
import sys
import torch
import torch.nn.functional as F

from dllm_parallel.core.kernels import configure_kernel_runtime
from dllm_parallel.core.kernels.tiled_linear_cross_entropy import tiled_linear_cross_entropy


def test_package_tiled_linear_cross_entropy_export_is_lazy_callable() -> None:
  result = subprocess.run(
    [
      sys.executable,
      "-c",
      (
        "import sys; "
        "from dllm_parallel.core.kernels import tiled_linear_cross_entropy; "
        "print(callable(tiled_linear_cross_entropy), 'torch' in sys.modules)"
      ),
    ],
    check=True,
    capture_output=True,
    text=True,
  )
  assert result.stdout.strip() == "True False"


def _reference_ce(
    hidden,
    weight,
    labels,
    *,
    bias=None,
    reduction="mean",
    ignore_index=-100,
    exclude_index=None,
    weight_dim_first=False,
):
  hidden_flat = hidden.reshape(-1, hidden.shape[-1])
  labels_flat = labels.reshape(-1)
  logits = hidden_flat @ weight if weight_dim_first else hidden_flat @ weight.t()
  if bias is not None:
    logits = logits + bias
  if exclude_index is not None:
    logits[:, exclude_index] = -float("inf")
  loss = F.cross_entropy(
    logits,
    labels_flat,
    reduction=reduction,
    ignore_index=ignore_index)
  if reduction == "none":
    loss = loss.reshape(labels.shape)
  return loss


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_tiled_linear_cross_entropy_matches_torch_loss_and_grads(reduction):
  torch.manual_seed(0)
  hidden = torch.randn(2, 5, 7, requires_grad=True)
  weight = torch.randn(13, 7, requires_grad=True)
  bias = torch.randn(13, requires_grad=True)
  labels = torch.randint(0, 13, (2, 5))
  labels[0, 1] = -100

  ref_hidden = hidden.detach().clone().requires_grad_(True)
  ref_weight = weight.detach().clone().requires_grad_(True)
  ref_bias = bias.detach().clone().requires_grad_(True)

  loss = tiled_linear_cross_entropy(
    hidden,
    weight,
    labels,
    bias=bias,
    vocab_block_size=4,
    reduction=reduction)
  ref_loss = _reference_ce(
    ref_hidden,
    ref_weight,
    labels,
    bias=ref_bias,
    reduction=reduction)

  torch.testing.assert_close(loss, ref_loss, atol=2e-5, rtol=2e-5)

  grad = torch.randn_like(loss) if reduction == "none" else None
  loss.backward(grad)
  ref_loss.backward(grad)

  torch.testing.assert_close(hidden.grad, ref_hidden.grad, atol=3e-5, rtol=3e-5)
  torch.testing.assert_close(weight.grad, ref_weight.grad, atol=3e-5, rtol=3e-5)
  torch.testing.assert_close(bias.grad, ref_bias.grad, atol=3e-5, rtol=3e-5)


def test_tiled_linear_cross_entropy_supports_dim_first_weight():
  torch.manual_seed(1)
  hidden = torch.randn(3, 4)
  weight = torch.randn(4, 9)
  labels = torch.tensor([0, 5, 8])

  loss = tiled_linear_cross_entropy(
    hidden,
    weight,
    labels,
    vocab_block_size=3,
    reduction="none",
    weight_layout="dim_first")
  ref_loss = _reference_ce(
    hidden,
    weight,
    labels,
    reduction="none",
    weight_dim_first=True)
  torch.testing.assert_close(loss, ref_loss, atol=2e-5, rtol=2e-5)


def test_tiled_linear_cross_entropy_excludes_mask_token():
  torch.manual_seed(2)
  hidden = torch.randn(2, 3, 5, requires_grad=True)
  weight = torch.randn(11, 5, requires_grad=True)
  labels = torch.randint(0, 10, (2, 3))
  mask_index = 10

  ref_hidden = hidden.detach().clone().requires_grad_(True)
  ref_weight = weight.detach().clone().requires_grad_(True)

  loss = tiled_linear_cross_entropy(
    hidden,
    weight,
    labels,
    vocab_block_size=4,
    reduction="mean",
    exclude_index=mask_index)
  ref_loss = _reference_ce(
    ref_hidden,
    ref_weight,
    labels,
    reduction="mean",
    exclude_index=mask_index)

  torch.testing.assert_close(loss, ref_loss, atol=2e-5, rtol=2e-5)
  loss.backward()
  ref_loss.backward()
  torch.testing.assert_close(hidden.grad, ref_hidden.grad, atol=3e-5, rtol=3e-5)
  torch.testing.assert_close(weight.grad, ref_weight.grad, atol=3e-5, rtol=3e-5)
  assert torch.count_nonzero(weight.grad[mask_index]) == 0


def test_tiled_linear_cross_entropy_non_contiguous_inputs():
  torch.manual_seed(3)
  hidden_base = torch.randn(2, 8, 6)
  labels_base = torch.randint(0, 17, (2, 8))
  hidden = hidden_base[:, ::2, :].requires_grad_(True)
  labels = labels_base[:, ::2]
  weight = torch.randn(17, 6, requires_grad=True)

  assert not hidden.is_contiguous()
  assert not labels.is_contiguous()
  loss = tiled_linear_cross_entropy(
    hidden,
    weight,
    labels,
    vocab_block_size=5)
  ref_loss = _reference_ce(hidden, weight, labels)
  torch.testing.assert_close(loss, ref_loss, atol=2e-5, rtol=2e-5)


def test_tiled_linear_cross_entropy_vocab_tile_sizes_are_equivalent():
  torch.manual_seed(31)
  base_hidden = torch.randn(3, 5, 9)
  base_weight = torch.randn(23, 9)
  base_bias = torch.randn(23)
  labels = torch.randint(0, 22, (3, 5))
  labels[0, 0] = -100
  labels[2, 4] = -100
  grad_out = torch.randn_like(labels, dtype=torch.float32)

  ref_hidden = base_hidden.detach().clone().requires_grad_(True)
  ref_weight = base_weight.detach().clone().requires_grad_(True)
  ref_bias = base_bias.detach().clone().requires_grad_(True)
  ref_loss = _reference_ce(
    ref_hidden,
    ref_weight,
    labels,
    bias=ref_bias,
    reduction="none",
    exclude_index=22)
  ref_loss.backward(grad_out)

  for vocab_block_size in [
      1,
      5,
      7,
      64,
  ]:
    hidden = base_hidden.detach().clone().requires_grad_(True)
    weight = base_weight.detach().clone().requires_grad_(True)
    bias = base_bias.detach().clone().requires_grad_(True)
    loss = tiled_linear_cross_entropy(
      hidden,
      weight,
      labels,
      bias=bias,
      vocab_block_size=vocab_block_size,
      reduction="none",
      exclude_index=22)

    torch.testing.assert_close(loss, ref_loss, atol=2e-5, rtol=2e-5)
    loss.backward(grad_out)
    torch.testing.assert_close(
      hidden.grad, ref_hidden.grad, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(
      weight.grad, ref_weight.grad, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(
      bias.grad, ref_bias.grad, atol=3e-5, rtol=3e-5)


def test_tiled_linear_cross_entropy_bfloat16_forward():
  torch.manual_seed(4)
  hidden = torch.randn(2, 4, 8, dtype=torch.bfloat16)
  weight = torch.randn(19, 8, dtype=torch.bfloat16)
  labels = torch.randint(0, 19, (2, 4))

  loss = tiled_linear_cross_entropy(
    hidden,
    weight,
    labels,
    vocab_block_size=7,
    dtype=torch.float32)
  ref_loss = _reference_ce(
    hidden.float(),
    weight.float(),
    labels)
  torch.testing.assert_close(loss, ref_loss, atol=3e-2, rtol=3e-2)


def test_tiled_linear_cross_entropy_large_vocab_smoke_backward():
  torch.manual_seed(5)
  hidden = torch.randn(2, 16, 16, requires_grad=True)
  weight = torch.randn(32768, 16, requires_grad=True)
  labels = torch.randint(0, 32768, (2, 16))

  loss = tiled_linear_cross_entropy(
    hidden,
    weight,
    labels,
    vocab_block_size=1024)
  assert torch.isfinite(loss)
  loss.backward()
  assert hidden.grad is not None
  assert weight.grad is not None
  assert hidden.grad.shape == hidden.shape
  assert weight.grad.shape == weight.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_tiled_linear_cross_entropy_cuda_matches_dense_forward_and_backward():
  configure_kernel_runtime(allow_runtime_jit=True)
  try:
    torch.manual_seed(51)
    device = torch.device("cuda")
    hidden = torch.randn(3, 7, 33, device=device, requires_grad=True)
    weight = torch.randn(4097, 33, device=device, requires_grad=True)
    bias = torch.randn(4097, device=device, requires_grad=True)
    labels = torch.randint(0, 4096, (3, 7), device=device)
    labels[0, 3] = -100
    labels[2, 5] = -100

    ref_hidden = hidden.detach().clone().requires_grad_(True)
    ref_weight = weight.detach().clone().requires_grad_(True)
    ref_bias = bias.detach().clone().requires_grad_(True)

    loss = tiled_linear_cross_entropy(
      hidden,
      weight,
      labels,
      bias=bias,
      vocab_block_size=257,
      reduction="mean",
      exclude_index=4096)
    ref_loss = _reference_ce(
      ref_hidden,
      ref_weight,
      labels,
      bias=ref_bias,
      reduction="mean",
      exclude_index=4096)

    torch.testing.assert_close(loss, ref_loss, atol=2e-5, rtol=2e-5)
    loss.backward()
    ref_loss.backward()
    torch.testing.assert_close(hidden.grad, ref_hidden.grad, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(weight.grad, ref_weight.grad, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(bias.grad, ref_bias.grad, atol=3e-5, rtol=3e-5)
  finally:
    configure_kernel_runtime(allow_runtime_jit=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_tiled_linear_cross_entropy_cuda_bfloat16_optimized_path():
  configure_kernel_runtime(allow_runtime_jit=True)
  try:
    torch.manual_seed(52)
    device = torch.device("cuda")
    hidden = (
      torch.randn(31, 96, device=device, dtype=torch.bfloat16) * 0.1
    ).requires_grad_(True)
    weight = (
      torch.randn(1025, 96, device=device, dtype=torch.bfloat16) * 0.1
    ).requires_grad_(True)
    bias = (
      torch.randn(1025, device=device, dtype=torch.bfloat16) * 0.1
    ).requires_grad_(True)
    labels = torch.randint(0, 1024, (31,), device=device)
    labels[::5] = -100
    mask_index = 1024

    ref_hidden = hidden.detach().clone().requires_grad_(True)
    ref_weight = weight.detach().clone().requires_grad_(True)
    ref_bias = bias.detach().clone().requires_grad_(True)

    loss = tiled_linear_cross_entropy(
      hidden,
      weight,
      labels,
      bias=bias,
      vocab_block_size=257,
      reduction="mean",
      exclude_index=mask_index,
      dtype=torch.bfloat16,
      weight_layout="vocab_first")
    ref_logits = ref_hidden.float() @ ref_weight.float().t() + ref_bias.float()
    ref_logits[:, mask_index] = -float("inf")
    ref_loss = F.cross_entropy(
      ref_logits,
      labels,
      reduction="mean",
      ignore_index=-100)

    torch.testing.assert_close(loss, ref_loss, atol=2e-5, rtol=2e-5)
    loss.backward()
    ref_loss.backward()
    torch.testing.assert_close(
      hidden.grad.float(), ref_hidden.grad.float(), atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(
      weight.grad.float(), ref_weight.grad.float(), atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(
      bias.grad.float(), ref_bias.grad.float(), atol=2e-3, rtol=2e-2)
  finally:
    configure_kernel_runtime(allow_runtime_jit=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_tiled_linear_cross_entropy_diffusiongemma_shape_hidden_gradient():
  """Cover the production DiffusionGemma vocabulary projection dimensions."""
  configure_kernel_runtime(allow_runtime_jit=True)
  try:
    torch.manual_seed(53)
    device = torch.device("cuda")
    rows = 4
    hidden_size = 2816
    vocab_size = 262144
    hidden = (
      torch.randn(rows, hidden_size, device=device, dtype=torch.bfloat16)
    ).requires_grad_(True)
    weight = (
      torch.randn(vocab_size, hidden_size, device=device, dtype=torch.bfloat16)
      * 0.02
    )
    labels = torch.randint(0, vocab_size, (rows,), device=device)
    token_weights = torch.tensor(
      [0.25, 0.75, 1.25, 2.0], device=device, dtype=torch.float32)

    ref_hidden = hidden.detach().clone().requires_grad_(True)
    token_losses = tiled_linear_cross_entropy(
      hidden,
      weight,
      labels,
      reduction="none",
      exclude_index=4,
      logit_softcap=30.0,
      dtype=torch.bfloat16,
      weight_layout="vocab_first")
    loss = (token_losses.float() * token_weights).sum() / rows

    ref_logits = ref_hidden.float() @ weight.float().t()
    ref_logits = torch.tanh(ref_logits / 30.0) * 30.0
    ref_logits[:, 4] = -float("inf")
    ref_losses = F.cross_entropy(ref_logits, labels, reduction="none")
    ref_loss = (ref_losses * token_weights).sum() / rows

    torch.testing.assert_close(loss, ref_loss, atol=2e-4, rtol=2e-4)
    loss.backward()
    ref_loss.backward()
    assert hidden.grad is not None
    assert ref_hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()
    torch.testing.assert_close(
      hidden.grad.float(), ref_hidden.grad.float(), atol=8e-3, rtol=4e-2)
  finally:
    configure_kernel_runtime(allow_runtime_jit=False)


def test_tiled_linear_cross_entropy_matches_bdlm_subs_masked_token_loss():
  torch.manual_seed(6)
  batch, seq_len, hidden_dim, vocab_size = 2, 6, 7, 23
  mask_index = vocab_size - 1
  hidden = torch.randn(batch, seq_len, hidden_dim, requires_grad=True)
  weight = torch.randn(vocab_size, hidden_dim, requires_grad=True)
  bias = torch.randn(vocab_size, requires_grad=True)
  x0 = torch.randint(0, vocab_size - 1, (batch, seq_len))
  xt = x0.clone()
  xt[0, 1] = mask_index
  xt[0, 4] = mask_index
  xt[1, 2] = mask_index
  loss_scale = -torch.rand(batch, seq_len)

  ref_hidden = hidden.detach().clone().requires_grad_(True)
  ref_weight = weight.detach().clone().requires_grad_(True)
  ref_bias = bias.detach().clone().requires_grad_(True)

  logits = ref_hidden @ ref_weight.t() + ref_bias
  logits[:, :, mask_index] = -float("inf")
  log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
  unmasked = xt != mask_index
  log_probs[unmasked] = -float("inf")
  log_probs[unmasked, xt[unmasked]] = 0
  ref_loss = loss_scale * torch.gather(
    log_probs,
    dim=-1,
    index=x0[:, :, None]).squeeze(-1)

  labels = torch.where(xt == mask_index, x0, torch.full_like(x0, -100))
  ce = tiled_linear_cross_entropy(
    hidden,
    weight,
    labels,
    bias=bias,
    vocab_block_size=5,
    reduction="none",
    exclude_index=mask_index)
  loss = -loss_scale * ce

  torch.testing.assert_close(loss, ref_loss, atol=2e-5, rtol=2e-5)
  loss.sum().backward()
  ref_loss.sum().backward()
  torch.testing.assert_close(hidden.grad, ref_hidden.grad, atol=3e-5, rtol=3e-5)
  torch.testing.assert_close(weight.grad, ref_weight.grad, atol=3e-5, rtol=3e-5)
  torch.testing.assert_close(bias.grad, ref_bias.grad, atol=3e-5, rtol=3e-5)
