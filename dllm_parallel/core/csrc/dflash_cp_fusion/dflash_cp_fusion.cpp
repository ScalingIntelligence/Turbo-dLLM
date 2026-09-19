// Copyright 2026 The dllm_parallel Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");

#include <torch/extension.h>

void dflash_merge_bshd_cuda(
    torch::Tensor numerator,
    torch::Tensor m,
    torch::Tensor l,
    torch::Tensor output,
    torch::Tensor lse);

void dflash_merge_state_cuda(
    torch::Tensor numerator,
    torch::Tensor m,
    torch::Tensor l,
    torch::Tensor incoming_numerator,
    torch::Tensor incoming_m,
    torch::Tensor incoming_l);

void merge_bshd_(
    torch::Tensor numerator,
    torch::Tensor m,
    torch::Tensor l,
    torch::Tensor output,
    torch::Tensor lse) {
  TORCH_CHECK(numerator.is_cuda(), "merge_bshd_ expects CUDA tensors");
  TORCH_CHECK(numerator.dtype() == torch::kFloat32, "numerator must be float32");
  TORCH_CHECK(
      output.dtype() == torch::kFloat32 || output.dtype() == torch::kFloat16 ||
          output.dtype() == torch::kBFloat16,
      "output must be float32, float16, or bfloat16");
  TORCH_CHECK(
      m.dtype() == torch::kFloat32 && l.dtype() == torch::kFloat32 &&
          lse.dtype() == torch::kFloat32,
      "softmax statistics must be float32");
  TORCH_CHECK(
      numerator.is_contiguous() && m.is_contiguous() && l.is_contiguous() &&
          output.is_contiguous() && lse.is_contiguous(),
      "merge_bshd_ tensors must be contiguous");
  TORCH_CHECK(
      numerator.dim() == 4 && output.dim() == 4,
      "numerator/output must be rank four");
  TORCH_CHECK(
      m.dim() == 3 && l.dim() == 3 && lse.dim() == 3,
      "softmax statistics must be rank three");
  TORCH_CHECK(
      numerator.size(0) == output.size(0) &&
          numerator.size(1) == output.size(2) &&
          numerator.size(2) == output.size(1) &&
          numerator.size(3) == output.size(3),
      "output must use BSHD layout");
  TORCH_CHECK(
      m.sizes() == l.sizes() && m.sizes() == lse.sizes(),
      "softmax statistic shapes must match");
  dflash_merge_bshd_cuda(numerator, m, l, output, lse);
}

void merge_state_(
    torch::Tensor numerator,
    torch::Tensor m,
    torch::Tensor l,
    torch::Tensor incoming_numerator,
    torch::Tensor incoming_m,
    torch::Tensor incoming_l) {
  TORCH_CHECK(numerator.is_cuda(), "merge_state_ expects CUDA tensors");
  TORCH_CHECK(
      numerator.dtype() == torch::kFloat32 && m.dtype() == torch::kFloat32 &&
          l.dtype() == torch::kFloat32 &&
          incoming_numerator.dtype() == torch::kFloat32 &&
          incoming_m.dtype() == torch::kFloat32 &&
          incoming_l.dtype() == torch::kFloat32,
      "merge_state_ tensors must be float32");
  TORCH_CHECK(
      numerator.is_contiguous() && m.is_contiguous() && l.is_contiguous() &&
          incoming_numerator.is_contiguous() && incoming_m.is_contiguous() &&
          incoming_l.is_contiguous(),
      "merge_state_ tensors must be contiguous");
  TORCH_CHECK(
      numerator.sizes() == incoming_numerator.sizes() &&
          m.sizes() == incoming_m.sizes() && l.sizes() == incoming_l.sizes() &&
          numerator.dim() == 4 && m.dim() == 3 && l.dim() == 3 &&
          numerator.size(0) == m.size(0) && numerator.size(1) == m.size(1) &&
          numerator.size(2) == m.size(2),
      "merge_state_ state shapes must match");
  dflash_merge_state_cuda(
      numerator, m, l, incoming_numerator, incoming_m, incoming_l);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("merge_bshd_", &merge_bshd_, "DFlash BSHD online stats merge");
  m.def("merge_state_", &merge_state_, "DFlash online state merge");
}
