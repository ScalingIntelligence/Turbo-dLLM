// Copyright 2026 The dllm_parallel Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
//
// Host bindings for the native fused-scheduled LM-head cross entropy. CUDA and
// cuBLAS kernel definitions live in fused_linear_ce_cuda.cu.

#include <torch/extension.h>

#include <string>
#include <vector>

std::vector<torch::Tensor> fused_linear_ce_forward_cuda(
    torch::Tensor hidden,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor labels,
    int64_t row_block_size,
    int64_t ignore_index,
    int64_t exclude_index,
    double logit_softcap,
    bool weight_dim_first,
    bool has_bias);

std::vector<torch::Tensor> fused_linear_ce_backward_cuda(
    torch::Tensor hidden,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor labels,
    torch::Tensor lse,
    torch::Tensor valid_count,
    torch::Tensor grad_output,
    int64_t row_block_size,
    int64_t reduction_code,
    int64_t vocab_start,
    int64_t ignore_index,
    int64_t exclude_index,
    double logit_softcap,
    bool weight_dim_first,
    bool has_bias,
    bool compute_weight_grad);

std::vector<torch::Tensor> fused_vocab_parallel_ce_forward_cuda(
    torch::Tensor hidden,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor labels,
    int64_t vocab_start,
    int64_t ignore_index,
    int64_t exclude_index,
    double logit_softcap,
    bool has_bias);

std::vector<torch::Tensor> dflash_frozen_kl_forward_cuda(
    torch::Tensor draft_hidden,
    torch::Tensor draft_weight,
    torch::Tensor teacher_hidden,
    torch::Tensor teacher_weight,
    double logit_softcap);

torch::Tensor dflash_frozen_kl_backward_cuda(
    torch::Tensor draft_hidden,
    torch::Tensor draft_weight,
    torch::Tensor teacher_hidden,
    torch::Tensor teacher_weight,
    torch::Tensor draft_lse,
    torch::Tensor teacher_lse,
    torch::Tensor grad_output,
    double logit_softcap);

std::vector<torch::Tensor> dflash_frozen_ce_topk_forward_cuda(
    torch::Tensor hidden,
    torch::Tensor weight,
    torch::Tensor labels,
    int64_t top_k,
    int64_t row_block_size,
    double output_multiplier,
    double logit_softcap);

torch::Tensor dflash_frozen_ce_topk_backward_cuda(
    torch::Tensor hidden,
    torch::Tensor weight,
    torch::Tensor labels,
    torch::Tensor lse,
    torch::Tensor probabilities,
    torch::Tensor top_ids,
    torch::Tensor grad_loss,
    torch::Tensor grad_probability,
    torch::Tensor grad_top_values,
    int64_t row_block_size,
    double output_multiplier,
    double logit_softcap);

std::vector<torch::Tensor> chunked_linear_ce_forward(
    torch::Tensor hidden,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor labels,
    int64_t row_block_size,
    int64_t ignore_index,
    int64_t exclude_index,
    double logit_softcap,
    std::string compute_dtype_name,
    bool weight_dim_first,
    bool has_bias) {
  (void)compute_dtype_name;
  TORCH_CHECK(hidden.is_cuda(), "native linear CE expects CUDA hidden");
  TORCH_CHECK(weight.is_cuda(), "native linear CE expects CUDA weight");
  TORCH_CHECK(labels.is_cuda(), "native linear CE expects CUDA labels");
  TORCH_CHECK(hidden.dim() == 2, "hidden must be flattened [N, D]");
  TORCH_CHECK(weight.dim() == 2, "weight must be a matrix");
  return fused_linear_ce_forward_cuda(
      hidden,
      weight,
      bias,
      labels,
      row_block_size,
      ignore_index,
      exclude_index,
      logit_softcap,
      weight_dim_first,
      has_bias);
}

std::vector<torch::Tensor> chunked_linear_ce_backward(
    torch::Tensor hidden,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor labels,
    torch::Tensor lse,
    torch::Tensor valid_count,
    torch::Tensor grad_output,
    int64_t row_block_size,
    int64_t reduction_code,
    int64_t ignore_index,
    int64_t exclude_index,
    double logit_softcap,
    std::string compute_dtype_name,
    bool weight_dim_first,
    bool has_bias) {
  (void)compute_dtype_name;
  TORCH_CHECK(hidden.is_cuda(), "native linear CE backward expects CUDA hidden");
  TORCH_CHECK(weight.is_cuda(), "native linear CE backward expects CUDA weight");
  TORCH_CHECK(labels.is_cuda(), "native linear CE backward expects CUDA labels");
  return fused_linear_ce_backward_cuda(
      hidden,
      weight,
      bias,
      labels,
      lse,
      valid_count,
      grad_output,
      row_block_size,
      reduction_code,
      0,
      ignore_index,
      exclude_index,
      logit_softcap,
      weight_dim_first,
      has_bias,
      true);
}

std::vector<torch::Tensor> chunked_linear_ce_backward_frozen_weight(
    torch::Tensor hidden,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor labels,
    torch::Tensor lse,
    torch::Tensor valid_count,
    torch::Tensor grad_output,
    int64_t row_block_size,
    int64_t reduction_code,
    int64_t ignore_index,
    int64_t exclude_index,
    double logit_softcap,
    std::string compute_dtype_name,
    bool weight_dim_first,
    bool has_bias) {
  (void)compute_dtype_name;
  TORCH_CHECK(
      hidden.is_cuda(),
      "native frozen linear CE backward expects CUDA hidden");
  TORCH_CHECK(
      weight.is_cuda(),
      "native frozen linear CE backward expects CUDA weight");
  TORCH_CHECK(
      labels.is_cuda(),
      "native frozen linear CE backward expects CUDA labels");
  return fused_linear_ce_backward_cuda(
      hidden,
      weight,
      bias,
      labels,
      lse,
      valid_count,
      grad_output,
      row_block_size,
      reduction_code,
      0,
      ignore_index,
      exclude_index,
      logit_softcap,
      weight_dim_first,
      has_bias,
      false);
}

std::vector<torch::Tensor> vocab_parallel_ce_forward_local(
    torch::Tensor hidden,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor labels,
    int64_t vocab_start,
    int64_t ignore_index,
    int64_t exclude_index,
    double logit_softcap,
    bool has_bias) {
  TORCH_CHECK(hidden.is_cuda(), "vocab-parallel CE expects CUDA hidden states");
  TORCH_CHECK(weight.is_cuda(), "vocab-parallel CE expects a CUDA weight shard");
  TORCH_CHECK(labels.is_cuda(), "vocab-parallel CE expects CUDA labels");
  return fused_vocab_parallel_ce_forward_cuda(
      hidden,
      weight,
      bias,
      labels,
      vocab_start,
      ignore_index,
      exclude_index,
      logit_softcap,
      has_bias);
}

std::vector<torch::Tensor> vocab_parallel_ce_backward_local(
    torch::Tensor hidden,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor labels,
    torch::Tensor global_lse,
    torch::Tensor valid_count,
    torch::Tensor grad_output,
    int64_t reduction_code,
    int64_t vocab_start,
    int64_t ignore_index,
    int64_t exclude_index,
    double logit_softcap,
    bool has_bias) {
  return fused_linear_ce_backward_cuda(
      hidden,
      weight,
      bias,
      labels,
      global_lse,
      valid_count,
      grad_output,
      0,
      reduction_code,
      vocab_start,
      ignore_index,
      exclude_index,
      logit_softcap,
      false,
      has_bias,
      true);
}

std::vector<torch::Tensor> dflash_frozen_kl_forward(
    torch::Tensor draft_hidden,
    torch::Tensor draft_weight,
    torch::Tensor teacher_hidden,
    torch::Tensor teacher_weight,
    double logit_softcap) {
  TORCH_CHECK(draft_hidden.is_cuda(), "DFlash KL expects CUDA draft hidden states");
  TORCH_CHECK(draft_weight.is_cuda(), "DFlash KL expects a CUDA draft head");
  TORCH_CHECK(teacher_hidden.is_cuda(), "DFlash KL expects CUDA teacher hidden states");
  TORCH_CHECK(teacher_weight.is_cuda(), "DFlash KL expects a CUDA teacher head");
  return dflash_frozen_kl_forward_cuda(
      draft_hidden,
      draft_weight,
      teacher_hidden,
      teacher_weight,
      logit_softcap);
}

torch::Tensor dflash_frozen_kl_backward(
    torch::Tensor draft_hidden,
    torch::Tensor draft_weight,
    torch::Tensor teacher_hidden,
    torch::Tensor teacher_weight,
    torch::Tensor draft_lse,
    torch::Tensor teacher_lse,
    torch::Tensor grad_output,
    double logit_softcap) {
  TORCH_CHECK(draft_hidden.is_cuda(), "DFlash KL backward expects CUDA draft hidden states");
  return dflash_frozen_kl_backward_cuda(
      draft_hidden,
      draft_weight,
      teacher_hidden,
      teacher_weight,
      draft_lse,
      teacher_lse,
      grad_output,
      logit_softcap);
}

std::vector<torch::Tensor> dflash_frozen_ce_topk_forward(
    torch::Tensor hidden,
    torch::Tensor weight,
    torch::Tensor labels,
    int64_t top_k,
    int64_t row_block_size,
    double output_multiplier,
    double logit_softcap) {
  TORCH_CHECK(hidden.is_cuda(), "DFlash CE/top-k expects CUDA hidden states");
  TORCH_CHECK(weight.is_cuda(), "DFlash CE/top-k expects a CUDA frozen head");
  TORCH_CHECK(labels.is_cuda(), "DFlash CE/top-k expects CUDA labels");
  return dflash_frozen_ce_topk_forward_cuda(
      hidden,
      weight,
      labels,
      top_k,
      row_block_size,
      output_multiplier,
      logit_softcap);
}

torch::Tensor dflash_frozen_ce_topk_backward(
    torch::Tensor hidden,
    torch::Tensor weight,
    torch::Tensor labels,
    torch::Tensor lse,
    torch::Tensor probabilities,
    torch::Tensor top_ids,
    torch::Tensor grad_loss,
    torch::Tensor grad_probability,
    torch::Tensor grad_top_values,
    int64_t row_block_size,
    double output_multiplier,
    double logit_softcap) {
  TORCH_CHECK(hidden.is_cuda(), "DFlash CE/top-k backward expects CUDA hidden states");
  return dflash_frozen_ce_topk_backward_cuda(
      hidden,
      weight,
      labels,
      lse,
      probabilities,
      top_ids,
      grad_loss,
      grad_probability,
      grad_top_values,
      row_block_size,
      output_multiplier,
      logit_softcap);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("chunked_linear_ce_forward", &chunked_linear_ce_forward,
        "Native fused-scheduled linear cross entropy forward");
  m.def("chunked_linear_ce_backward", &chunked_linear_ce_backward,
        "Native fused-scheduled linear cross entropy backward");
  m.def(
      "chunked_linear_ce_backward_frozen_weight",
      &chunked_linear_ce_backward_frozen_weight,
      "Native fused-scheduled frozen-weight linear cross entropy backward");
  m.def(
      "vocab_parallel_ce_forward_local",
      &vocab_parallel_ce_forward_local,
      "Native local vocabulary-shard CE statistics");
  m.def(
      "vocab_parallel_ce_backward_local",
      &vocab_parallel_ce_backward_local,
      "Native local vocabulary-shard CE backward");
  m.def(
      "dflash_frozen_kl_forward",
      &dflash_frozen_kl_forward,
      "Native streaming DFlash frozen-head KL forward");
  m.def(
      "dflash_frozen_kl_backward",
      &dflash_frozen_kl_backward,
      "Native streaming DFlash frozen-head KL backward");
  m.def(
      "dflash_frozen_ce_topk_forward",
      &dflash_frozen_ce_topk_forward,
      "Native streaming DFlash frozen-head CE/top-k forward");
  m.def(
      "dflash_frozen_ce_topk_backward",
      &dflash_frozen_ce_topk_backward,
      "Native streaming DFlash frozen-head CE/top-k backward");
}
