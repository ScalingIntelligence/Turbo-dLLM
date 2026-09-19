// Copyright 2026 The dllm_parallel Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
//
// Host bindings for the BDLM CP/BP online-softmax stats merge kernels.
// CUDA kernel definitions live in cp_fusion_cuda.cu.

#include <torch/extension.h>

void bdlm_merge_full_cuda(
    torch::Tensor old_num,
    torch::Tensor old_m,
    torch::Tensor old_l,
    torch::Tensor new_num,
    torch::Tensor new_m,
    torch::Tensor new_l);

void bdlm_merge_compact_cuda(
    torch::Tensor numerator,
    torch::Tensor m,
    torch::Tensor l,
    torch::Tensor query_indices,
    torch::Tensor compact_output,
    torch::Tensor compact_lse);

void bdlm_merge_backward_cuda(
    torch::Tensor shard_output,
    torch::Tensor shard_lse,
    torch::Tensor final_output,
    torch::Tensor final_lse,
    torch::Tensor grad_output,
    torch::Tensor shard_grad_output,
    torch::Tensor grad_lse);

void bdlm_finalize_bshd_cuda(
    torch::Tensor numerator,
    torch::Tensor m,
    torch::Tensor l,
    torch::Tensor output,
    torch::Tensor final_lse);


void merge_full_(
    torch::Tensor old_num,
    torch::Tensor old_m,
    torch::Tensor old_l,
    torch::Tensor new_num,
    torch::Tensor new_m,
    torch::Tensor new_l) {
  TORCH_CHECK(old_num.is_cuda(), "merge_full_ expects CUDA tensors");
  TORCH_CHECK(old_num.dtype() == torch::kFloat32, "old_num must be float32");
  TORCH_CHECK(new_num.dtype() == torch::kFloat32 || new_num.dtype() == torch::kFloat16 ||
              new_num.dtype() == torch::kBFloat16,
              "new_num must be float32, float16, or bfloat16");
  TORCH_CHECK(old_m.dtype() == torch::kFloat32, "old_m must be float32");
  TORCH_CHECK(old_l.dtype() == torch::kFloat32, "old_l must be float32");
  TORCH_CHECK(new_m.dtype() == torch::kFloat32, "new_m must be float32");
  TORCH_CHECK(new_l.dtype() == torch::kFloat32, "new_l must be float32");
  TORCH_CHECK(old_num.is_contiguous(), "old_num must be contiguous");
  TORCH_CHECK(old_m.is_contiguous(), "old_m must be contiguous");
  TORCH_CHECK(old_l.is_contiguous(), "old_l must be contiguous");
  TORCH_CHECK(new_m.is_contiguous(), "new_m must be contiguous");
  TORCH_CHECK(new_l.is_contiguous(), "new_l must be contiguous");
  TORCH_CHECK(old_num.sizes() == new_num.sizes(), "numerator shapes must match");
  TORCH_CHECK(old_m.sizes() == new_m.sizes(), "m shapes must match");
  TORCH_CHECK(old_l.sizes() == new_l.sizes(), "l shapes must match");
  TORCH_CHECK(old_num.dim() == old_m.dim() + 1, "old_num must have one extra head_dim");
  TORCH_CHECK(old_num.size(old_num.dim() - 2) == old_m.size(old_m.dim() - 1),
              "query dimensions must match");
  bdlm_merge_full_cuda(old_num, old_m, old_l, new_num, new_m, new_l);
}

void merge_compact_(
    torch::Tensor numerator,
    torch::Tensor m,
    torch::Tensor l,
    torch::Tensor query_indices,
    torch::Tensor compact_output,
    torch::Tensor compact_lse) {
  TORCH_CHECK(numerator.is_cuda(), "merge_compact_ expects CUDA tensors");
  TORCH_CHECK(numerator.dtype() == torch::kFloat32, "numerator must be float32");
  TORCH_CHECK(compact_output.dtype() == torch::kFloat32 ||
              compact_output.dtype() == torch::kFloat16 ||
              compact_output.dtype() == torch::kBFloat16,
              "compact_output must be float32, float16, or bfloat16");
  TORCH_CHECK(m.dtype() == torch::kFloat32, "m must be float32");
  TORCH_CHECK(l.dtype() == torch::kFloat32, "l must be float32");
  TORCH_CHECK(compact_lse.dtype() == torch::kFloat32, "compact_lse must be float32");
  TORCH_CHECK(query_indices.dtype() == torch::kInt64, "query_indices must be int64");
  TORCH_CHECK(numerator.is_contiguous(), "numerator must be contiguous");
  TORCH_CHECK(m.is_contiguous(), "m must be contiguous");
  TORCH_CHECK(l.is_contiguous(), "l must be contiguous");
  TORCH_CHECK(query_indices.is_contiguous(), "query_indices must be contiguous");
  TORCH_CHECK(compact_lse.is_contiguous(), "compact_lse must be contiguous");
  TORCH_CHECK(numerator.dim() == 4, "numerator must be [B,H,Q,D]");
  TORCH_CHECK(m.dim() == 3 && l.dim() == 3, "m/l must be [B,H,Q]");
  TORCH_CHECK(compact_output.dim() == 4, "compact_output must be [B,H,N,D]");
  TORCH_CHECK(compact_lse.dim() == 3, "compact_lse must be [B,H,N]");
  TORCH_CHECK(query_indices.dim() == 1, "query_indices must be [N]");
  TORCH_CHECK(numerator.size(0) == compact_output.size(0), "batch mismatch");
  TORCH_CHECK(numerator.size(1) == compact_output.size(1), "head mismatch");
  TORCH_CHECK(numerator.size(3) == compact_output.size(3), "head_dim mismatch");
  TORCH_CHECK(query_indices.size(0) == compact_output.size(2), "compact query mismatch");
  TORCH_CHECK(compact_lse.size(0) == compact_output.size(0), "lse batch mismatch");
  TORCH_CHECK(compact_lse.size(1) == compact_output.size(1), "lse head mismatch");
  TORCH_CHECK(compact_lse.size(2) == compact_output.size(2), "lse query mismatch");
  bdlm_merge_compact_cuda(numerator, m, l, query_indices, compact_output, compact_lse);
}

void merge_backward_(
    torch::Tensor shard_output,
    torch::Tensor shard_lse,
    torch::Tensor final_output,
    torch::Tensor final_lse,
    torch::Tensor grad_output,
    torch::Tensor shard_grad_output,
    torch::Tensor grad_lse) {
  TORCH_CHECK(shard_output.is_cuda(), "merge_backward_ expects CUDA tensors");
  TORCH_CHECK(shard_output.dtype() == torch::kFloat32 ||
              shard_output.dtype() == torch::kFloat16 ||
              shard_output.dtype() == torch::kBFloat16,
              "shard_output must be float32, float16, or bfloat16");
  TORCH_CHECK(final_output.dtype() == shard_output.dtype(), "output dtype mismatch");
  TORCH_CHECK(grad_output.dtype() == shard_output.dtype(), "grad_output dtype mismatch");
  TORCH_CHECK(shard_grad_output.dtype() == shard_output.dtype(), "shard_grad_output dtype mismatch");
  TORCH_CHECK(shard_lse.dtype() == torch::kFloat32, "shard_lse must be float32");
  TORCH_CHECK(final_lse.dtype() == torch::kFloat32, "final_lse must be float32");
  TORCH_CHECK(grad_lse.dtype() == torch::kFloat32, "grad_lse must be float32");
  TORCH_CHECK(shard_lse.is_contiguous(), "shard_lse must be contiguous");
  TORCH_CHECK(final_lse.is_contiguous(), "final_lse must be contiguous");
  TORCH_CHECK(grad_lse.is_contiguous(), "grad_lse must be contiguous");
  TORCH_CHECK(shard_output.sizes() == final_output.sizes(), "output shapes must match");
  TORCH_CHECK(shard_output.sizes() == grad_output.sizes(), "grad output shape mismatch");
  TORCH_CHECK(shard_output.sizes() == shard_grad_output.sizes(), "shard grad output shape mismatch");
  TORCH_CHECK(shard_lse.sizes() == final_lse.sizes(), "lse shapes must match");
  TORCH_CHECK(shard_lse.sizes() == grad_lse.sizes(), "grad lse shape mismatch");
  TORCH_CHECK(shard_output.dim() == 4, "outputs must be [B,H,Q,D]");
  TORCH_CHECK(shard_lse.dim() == 3, "lse tensors must be [B,H,Q]");
  TORCH_CHECK(shard_output.size(0) == shard_lse.size(0), "batch mismatch");
  TORCH_CHECK(shard_output.size(1) == shard_lse.size(1), "head mismatch");
  TORCH_CHECK(shard_output.size(2) == shard_lse.size(2), "query mismatch");
  bdlm_merge_backward_cuda(
      shard_output,
      shard_lse,
      final_output,
      final_lse,
      grad_output,
      shard_grad_output,
      grad_lse);
}

void finalize_bshd_(
    torch::Tensor numerator,
    torch::Tensor m,
    torch::Tensor l,
    torch::Tensor output,
    torch::Tensor final_lse) {
  TORCH_CHECK(numerator.is_cuda(), "finalize_bshd_ expects CUDA tensors");
  TORCH_CHECK(numerator.dtype() == torch::kFloat32, "numerator must be float32");
  TORCH_CHECK(m.dtype() == torch::kFloat32 && l.dtype() == torch::kFloat32,
              "m/l must be float32");
  TORCH_CHECK(output.dtype() == torch::kFloat16 || output.dtype() == torch::kBFloat16 ||
              output.dtype() == torch::kFloat32,
              "output must be float32, float16, or bfloat16");
  TORCH_CHECK(final_lse.dtype() == torch::kFloat32, "final_lse must be float32");
  TORCH_CHECK(numerator.is_contiguous() && m.is_contiguous() && l.is_contiguous(),
              "numerator and online-softmax stats must be contiguous");
  TORCH_CHECK(output.is_contiguous() && final_lse.is_contiguous(),
              "output and final_lse must be contiguous");
  TORCH_CHECK(numerator.dim() == 4, "numerator must be [B,H,Q,D]");
  TORCH_CHECK(m.dim() == 3 && l.dim() == 3, "m/l must be [B,H,Q]");
  TORCH_CHECK(output.dim() == 4, "output must be [B,Q,H,D]");
  TORCH_CHECK(final_lse.dim() == 3, "final_lse must be [B,H,Q]");
  TORCH_CHECK(numerator.size(0) == output.size(0) &&
              numerator.size(1) == output.size(2) &&
              numerator.size(2) == output.size(1) &&
              numerator.size(3) == output.size(3),
              "output shape must be the BSHD view of numerator");
  TORCH_CHECK(m.sizes() == l.sizes() && m.sizes() == final_lse.sizes(),
              "online-softmax stat shapes must match");
  TORCH_CHECK(m.size(0) == numerator.size(0) &&
              m.size(1) == numerator.size(1) &&
              m.size(2) == numerator.size(2),
              "online-softmax stats must match numerator");
  bdlm_finalize_bshd_cuda(numerator, m, l, output, final_lse);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("merge_full_", &merge_full_, "BDLM CP/BP online stats merge");
  m.def("merge_compact_", &merge_compact_, "BDLM CP/BP compact online stats merge");
  m.def("merge_backward_", &merge_backward_, "BDLM CP/BP backward stats merge");
  m.def("finalize_bshd_", &finalize_bshd_, "BDLM CP BSHD online stats finalize");
}
