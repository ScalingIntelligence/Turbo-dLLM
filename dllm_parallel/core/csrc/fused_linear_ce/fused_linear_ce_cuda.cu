// Copyright 2026 The dllm_parallel Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
//
// CUDA + cuBLAS kernels for the native fused-scheduled LM-head cross entropy.
// The extension keeps the scheduler below Python: it projects native-scheduled
// token tiles with cuBLAS tensor-core GEMMs and runs custom CUDA kernels for the
// cross-entropy stats, loss, grad-logits, and dtype casts, avoiding full
// sequence logits materialization while preserving tensor-core GEMM throughput.
//
// The numerics and native scheduling here are intentionally unchanged from the
// reference implementation; do not alter the GEMM layout or reduction math
// without re-validating against the tiled reference and throughput targets.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <limits>
#include <sstream>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kCastThreads = 256;
constexpr int kMaxCastGridBlocks = 4096;

void check_cublas(cublasStatus_t status, const char* expr) {
  if (status == CUBLAS_STATUS_SUCCESS) {
    return;
  }
  std::ostringstream oss;
  oss << "cuBLAS call failed (" << expr << ") with status " << int(status);
  TORCH_CHECK(false, oss.str());
}

#define CHECK_CUBLAS(EXPR) check_cublas((EXPR), #EXPR)

cudaDataType_t cuda_type(at::ScalarType scalar_type) {
  switch (scalar_type) {
    case at::kFloat:
      return CUDA_R_32F;
    case at::kHalf:
      return CUDA_R_16F;
    case at::kBFloat16:
      return CUDA_R_16BF;
    default:
      TORCH_CHECK(false, "native linear CE supports float32, float16, and bfloat16");
  }
}

cublasComputeType_t compute_type(at::ScalarType scalar_type) {
  (void)scalar_type;
  return CUBLAS_COMPUTE_32F;
}

__device__ inline float warp_reduce_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__device__ inline float warp_reduce_max(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
  }
  return value;
}

__device__ inline float block_reduce_sum(float value) {
  __shared__ float shared[32];
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  value = warp_reduce_sum(value);
  if (lane == 0) {
    shared[warp] = value;
  }
  __syncthreads();
  value = (threadIdx.x < (blockDim.x + 31) / 32) ? shared[lane] : 0.0f;
  if (warp == 0) {
    value = warp_reduce_sum(value);
  }
  return value;
}

__device__ inline float block_reduce_max(float value) {
  __shared__ float shared[32];
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  value = warp_reduce_max(value);
  if (lane == 0) {
    shared[warp] = value;
  }
  __syncthreads();
  value = (threadIdx.x < (blockDim.x + 31) / 32) ? shared[lane] : -INFINITY;
  if (warp == 0) {
    value = warp_reduce_max(value);
  }
  return value;
}

template <bool kApplySoftcap>
__device__ inline float maybe_apply_logit_softcap(float value, float cap) {
  if constexpr (kApplySoftcap) {
    return tanhf(value / cap) * cap;
  }
  return value;
}

template <bool kApplySoftcap>
__device__ inline float maybe_logit_softcap_grad(float value, float cap) {
  if constexpr (kApplySoftcap) {
    float t = tanhf(value / cap);
    return 1.0f - t * t;
  }
  return 1.0f;
}

template <typename scalar_t>
__device__ inline scalar_t from_float_device(float value) {
  return static_cast<scalar_t>(value);
}

template <>
__device__ inline c10::Half from_float_device<c10::Half>(float value) {
  return c10::Half(value);
}

template <>
__device__ inline c10::BFloat16 from_float_device<c10::BFloat16>(float value) {
  return c10::BFloat16(value);
}

template <typename scalar_t>
__global__ void cast_float_to_scalar_kernel(
    const float* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t numel) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; idx < numel; idx += stride) {
    output[idx] = from_float_device<scalar_t>(input[idx]);
  }
}

template <bool kApplySoftcap>
__global__ void ce_forward_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ bias,
    const int64_t* __restrict__ labels,
    float* __restrict__ losses,
    float* __restrict__ lse,
    float* __restrict__ valid_count,
    int64_t token_start,
    int64_t block_tokens,
    int64_t vocab_size,
    int64_t ignore_index,
    int64_t exclude_index,
    float logit_softcap,
    bool has_bias) {
  int64_t local_token = static_cast<int64_t>(blockIdx.x);
  if (local_token >= block_tokens) {
    return;
  }
  int64_t token = token_start + local_token;
  int64_t label = labels[token];
  bool valid = label != ignore_index;

  float local_max = -INFINITY;
  for (int64_t vocab = threadIdx.x; vocab < vocab_size; vocab += blockDim.x) {
    if (vocab == exclude_index) {
      continue;
    }
    float value = logits[local_token * vocab_size + vocab];
    if (has_bias) {
      value += bias[vocab];
    }
    value = maybe_apply_logit_softcap<kApplySoftcap>(value, logit_softcap);
    local_max = fmaxf(local_max, value);
  }
  float max_value = block_reduce_max(local_max);
  __shared__ float shared_max;
  if (threadIdx.x == 0) {
    shared_max = max_value;
  }
  __syncthreads();

  float local_sum = 0.0f;
  float local_true = 0.0f;
  for (int64_t vocab = threadIdx.x; vocab < vocab_size; vocab += blockDim.x) {
    if (vocab == exclude_index) {
      continue;
    }
    float value = logits[local_token * vocab_size + vocab];
    if (has_bias) {
      value += bias[vocab];
    }
    value = maybe_apply_logit_softcap<kApplySoftcap>(value, logit_softcap);
    local_sum += expf(value - shared_max);
    if (valid && vocab == label) {
      local_true = value;
    }
  }
  float sum_value = block_reduce_sum(local_sum);
  float true_logit = block_reduce_sum(local_true);
  if (threadIdx.x == 0) {
    float token_lse = shared_max + logf(sum_value);
    lse[token] = token_lse;
    losses[token] = valid ? token_lse - true_logit : 0.0f;
    if (valid) {
      atomicAdd(valid_count, 1.0f);
    }
  }
}

__device__ inline float token_scale_value(
    const float* __restrict__ grad_output,
    const float* __restrict__ valid_count,
    int64_t token,
    int64_t reduction_code) {
  if (reduction_code == 0) {
    return grad_output[token];
  }
  if (reduction_code == 1) {
    return grad_output[0];
  }
  return grad_output[0] / valid_count[0];
}

template <bool kApplySoftcap>
__global__ void ce_grad_logits_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ bias,
    const int64_t* __restrict__ labels,
    const float* __restrict__ lse,
    const float* __restrict__ valid_count,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_logits,
    float* __restrict__ grad_bias,
    int64_t token_start,
    int64_t block_tokens,
    int64_t vocab_size,
    int64_t reduction_code,
    int64_t vocab_start,
    int64_t ignore_index,
    int64_t exclude_index,
    float logit_softcap,
    bool has_bias) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = block_tokens * vocab_size;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; idx < total; idx += stride) {
    int64_t local_token = idx / vocab_size;
    int64_t vocab = idx - local_token * vocab_size;
    int64_t token = token_start + local_token;
    int64_t label = labels[token];
    bool valid = label != ignore_index;
    float grad = 0.0f;
    int64_t global_vocab = vocab_start + vocab;
    if (valid && global_vocab != exclude_index) {
      float value = logits[idx];
      if (has_bias) {
        value += bias[vocab];
      }
      float capped_value = maybe_apply_logit_softcap<kApplySoftcap>(
          value,
          logit_softcap);
      grad = expf(capped_value - lse[token]);
      if (global_vocab == label) {
        grad -= 1.0f;
      }
      grad *= token_scale_value(grad_output, valid_count, token, reduction_code);
      grad *= maybe_logit_softcap_grad<kApplySoftcap>(value, logit_softcap);
    }
    grad_logits[idx] = grad;
    if (has_bias && grad != 0.0f) {
      atomicAdd(grad_bias + vocab, grad);
    }
  }
}

template <bool kApplySoftcap>
__global__ void vocab_parallel_ce_stats_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ bias,
    const int64_t* __restrict__ labels,
    float* __restrict__ local_max_out,
    float* __restrict__ local_sum_out,
    float* __restrict__ target_logits,
    float* __restrict__ valid_count,
    int64_t token_start,
    int64_t block_tokens,
    int64_t vocab_size,
    int64_t vocab_start,
    int64_t ignore_index,
    int64_t exclude_index,
    float logit_softcap,
    bool has_bias) {
  int64_t local_token = static_cast<int64_t>(blockIdx.x);
  if (local_token >= block_tokens) {
    return;
  }
  int64_t token = token_start + local_token;
  int64_t label = labels[token];
  bool valid = label != ignore_index;

  float local_max = -INFINITY;
  for (int64_t vocab = threadIdx.x; vocab < vocab_size; vocab += blockDim.x) {
    if (vocab_start + vocab == exclude_index) {
      continue;
    }
    float value = logits[local_token * vocab_size + vocab];
    if (has_bias) {
      value += bias[vocab];
    }
    value = maybe_apply_logit_softcap<kApplySoftcap>(value, logit_softcap);
    local_max = fmaxf(local_max, value);
  }
  float max_value = block_reduce_max(local_max);
  __shared__ float shared_max;
  if (threadIdx.x == 0) {
    shared_max = max_value;
  }
  __syncthreads();

  float local_sum = 0.0f;
  float local_true = 0.0f;
  for (int64_t vocab = threadIdx.x; vocab < vocab_size; vocab += blockDim.x) {
    int64_t global_vocab = vocab_start + vocab;
    if (global_vocab == exclude_index) {
      continue;
    }
    float value = logits[local_token * vocab_size + vocab];
    if (has_bias) {
      value += bias[vocab];
    }
    value = maybe_apply_logit_softcap<kApplySoftcap>(value, logit_softcap);
    local_sum += expf(value - shared_max);
    if (valid && global_vocab == label) {
      local_true = value;
    }
  }
  float sum_value = block_reduce_sum(local_sum);
  float true_logit = block_reduce_sum(local_true);
  if (threadIdx.x == 0) {
    local_max_out[token] = shared_max;
    local_sum_out[token] = sum_value;
    bool excluded_target_is_local = valid && label == exclude_index &&
        label >= vocab_start && label < vocab_start + vocab_size;
    target_logits[token] = excluded_target_is_local ? -INFINITY : true_logit;
    if (valid) {
      atomicAdd(valid_count, 1.0f);
    }
  }
}

template <bool kApplySoftcap>
__global__ void dflash_kl_forward_kernel(
    const float* __restrict__ draft_logits,
    const float* __restrict__ teacher_logits,
    float* __restrict__ losses,
    float* __restrict__ draft_lse,
    float* __restrict__ teacher_lse,
    int64_t token_start,
    int64_t block_tokens,
    int64_t vocab_size,
    float logit_softcap) {
  int64_t local_token = static_cast<int64_t>(blockIdx.x);
  if (local_token >= block_tokens) {
    return;
  }
  int64_t row = local_token * vocab_size;
  float draft_max = -INFINITY;
  float teacher_max = -INFINITY;
  for (int64_t vocab = threadIdx.x; vocab < vocab_size; vocab += blockDim.x) {
    draft_max = fmaxf(
        draft_max,
        maybe_apply_logit_softcap<kApplySoftcap>(
            draft_logits[row + vocab], logit_softcap));
    teacher_max = fmaxf(
        teacher_max,
        maybe_apply_logit_softcap<kApplySoftcap>(
            teacher_logits[row + vocab], logit_softcap));
  }
  draft_max = block_reduce_max(draft_max);
  __syncthreads();
  teacher_max = block_reduce_max(teacher_max);
  __shared__ float shared_draft_max;
  __shared__ float shared_teacher_max;
  if (threadIdx.x == 0) {
    shared_draft_max = draft_max;
    shared_teacher_max = teacher_max;
  }
  __syncthreads();

  float draft_sum = 0.0f;
  float teacher_sum = 0.0f;
  for (int64_t vocab = threadIdx.x; vocab < vocab_size; vocab += blockDim.x) {
    draft_sum += expf(
        maybe_apply_logit_softcap<kApplySoftcap>(
            draft_logits[row + vocab], logit_softcap)
        - shared_draft_max);
    teacher_sum += expf(
        maybe_apply_logit_softcap<kApplySoftcap>(
            teacher_logits[row + vocab], logit_softcap)
        - shared_teacher_max);
  }
  draft_sum = block_reduce_sum(draft_sum);
  __syncthreads();
  teacher_sum = block_reduce_sum(teacher_sum);
  __shared__ float shared_draft_lse;
  __shared__ float shared_teacher_lse;
  if (threadIdx.x == 0) {
    shared_draft_lse = shared_draft_max + logf(draft_sum);
    shared_teacher_lse = shared_teacher_max + logf(teacher_sum);
  }
  __syncthreads();

  float local_loss = 0.0f;
  for (int64_t vocab = threadIdx.x; vocab < vocab_size; vocab += blockDim.x) {
    float teacher_log_prob =
        maybe_apply_logit_softcap<kApplySoftcap>(
            teacher_logits[row + vocab], logit_softcap)
        - shared_teacher_lse;
    float draft_log_prob =
        maybe_apply_logit_softcap<kApplySoftcap>(
            draft_logits[row + vocab], logit_softcap)
        - shared_draft_lse;
    local_loss += expf(teacher_log_prob) * (teacher_log_prob - draft_log_prob);
  }
  local_loss = block_reduce_sum(local_loss);
  if (threadIdx.x == 0) {
    int64_t token = token_start + local_token;
    losses[token] = local_loss;
    draft_lse[token] = shared_draft_lse;
    teacher_lse[token] = shared_teacher_lse;
  }
}

template <bool kApplySoftcap>
__global__ void dflash_kl_grad_logits_kernel(
    const float* __restrict__ draft_logits,
    const float* __restrict__ teacher_logits,
    const float* __restrict__ draft_lse,
    const float* __restrict__ teacher_lse,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_logits,
    int64_t token_start,
    int64_t block_tokens,
    int64_t vocab_size,
    float logit_softcap) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = block_tokens * vocab_size;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; idx < total; idx += stride) {
    int64_t local_token = idx / vocab_size;
    int64_t token = token_start + local_token;
    float draft_logit = draft_logits[idx];
    float capped_draft = maybe_apply_logit_softcap<kApplySoftcap>(
        draft_logit, logit_softcap);
    float capped_teacher = maybe_apply_logit_softcap<kApplySoftcap>(
        teacher_logits[idx], logit_softcap);
    grad_logits[idx] = (
        expf(capped_draft - draft_lse[token])
        - expf(capped_teacher - teacher_lse[token]))
        * grad_output[token]
        * maybe_logit_softcap_grad<kApplySoftcap>(
            draft_logit, logit_softcap);
  }
}

template <bool kApplySoftcap>
__global__ void dflash_ce_topk_stats_kernel(
    const float* __restrict__ raw_logits,
    const int64_t* __restrict__ labels,
    float* __restrict__ losses,
    float* __restrict__ probabilities,
    float* __restrict__ lse,
    int64_t token_start,
    int64_t block_tokens,
    int64_t vocab_size,
    float output_multiplier,
    float logit_softcap) {
  int64_t local_token = static_cast<int64_t>(blockIdx.x);
  if (local_token >= block_tokens) {
    return;
  }
  int64_t row = local_token * vocab_size;
  int64_t label = labels[token_start + local_token];
  float local_max = -INFINITY;
  for (int64_t vocab = threadIdx.x; vocab < vocab_size; vocab += blockDim.x) {
    float value = maybe_apply_logit_softcap<kApplySoftcap>(
        raw_logits[row + vocab] * output_multiplier,
        logit_softcap);
    local_max = fmaxf(local_max, value);
  }
  float max_value = block_reduce_max(local_max);
  __shared__ float shared_max;
  if (threadIdx.x == 0) {
    shared_max = max_value;
  }
  __syncthreads();

  float local_sum = 0.0f;
  float local_target = 0.0f;
  for (int64_t vocab = threadIdx.x; vocab < vocab_size; vocab += blockDim.x) {
    float value = maybe_apply_logit_softcap<kApplySoftcap>(
        raw_logits[row + vocab] * output_multiplier,
        logit_softcap);
    local_sum += expf(value - shared_max);
    if (vocab == label) {
      local_target = value;
    }
  }
  float sum_value = block_reduce_sum(local_sum);
  float target_value = block_reduce_sum(local_target);
  if (threadIdx.x == 0) {
    int64_t token = token_start + local_token;
    float token_lse = shared_max + logf(sum_value);
    float loss = token_lse - target_value;
    lse[token] = token_lse;
    losses[token] = loss;
    probabilities[token] = expf(-loss);
  }
}

template <typename scalar_t, bool kApplySoftcap>
__global__ void dflash_ce_topk_grad_logits_kernel(
    const float* __restrict__ raw_logits,
    const int64_t* __restrict__ labels,
    const float* __restrict__ lse,
    const float* __restrict__ probabilities,
    const float* __restrict__ grad_loss,
    const float* __restrict__ grad_probability,
    scalar_t* __restrict__ grad_logits,
    int64_t token_start,
    int64_t block_tokens,
    int64_t vocab_size,
    float output_multiplier,
    float logit_softcap) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = block_tokens * vocab_size;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; idx < total; idx += stride) {
    int64_t local_token = idx / vocab_size;
    int64_t vocab = idx - local_token * vocab_size;
    int64_t token = token_start + local_token;
    float raw = raw_logits[idx] * output_multiplier;
    float capped = maybe_apply_logit_softcap<kApplySoftcap>(raw, logit_softcap);
    float ce_scale = grad_loss[token] - grad_probability[token] * probabilities[token];
    float grad = expf(capped - lse[token]);
    if (vocab == labels[token]) {
      grad -= 1.0f;
    }
    grad *= ce_scale * output_multiplier;
    grad *= maybe_logit_softcap_grad<kApplySoftcap>(raw, logit_softcap);
    grad_logits[idx] = from_float_device<scalar_t>(grad);
  }
}

template <typename scalar_t, bool kApplySoftcap>
__global__ void dflash_ce_topk_scatter_kernel(
    const float* __restrict__ raw_logits,
    const int64_t* __restrict__ top_ids,
    const float* __restrict__ grad_top_values,
    scalar_t* __restrict__ grad_logits,
    int64_t token_start,
    int64_t block_tokens,
    int64_t vocab_size,
    int64_t top_k,
    float output_multiplier,
    float logit_softcap) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = block_tokens * top_k;
  if (idx >= total) {
    return;
  }
  int64_t local_token = idx / top_k;
  int64_t slot = idx - local_token * top_k;
  int64_t token = token_start + local_token;
  int64_t vocab = top_ids[token * top_k + slot];
  int64_t logit_idx = local_token * vocab_size + vocab;
  float raw = raw_logits[logit_idx] * output_multiplier;
  float derivative = output_multiplier
      * maybe_logit_softcap_grad<kApplySoftcap>(raw, logit_softcap);
  float value = static_cast<float>(grad_logits[logit_idx]);
  value += grad_top_values[token * top_k + slot] * derivative;
  grad_logits[logit_idx] = from_float_device<scalar_t>(value);
}

template <typename scalar_t>
void launch_cast(
    const torch::Tensor& input,
    const torch::Tensor& output,
    cudaStream_t stream) {
  int64_t numel = input.numel();
  if (numel == 0) {
    return;
  }
  int blocks = static_cast<int>(
      std::min<int64_t>(
          (numel + kCastThreads - 1) / kCastThreads,
          kMaxCastGridBlocks));
  cast_float_to_scalar_kernel<scalar_t><<<blocks, kCastThreads, 0, stream>>>(
      input.data_ptr<float>(), output.data_ptr<scalar_t>(), numel);
}

void project_logits(
    cublasHandle_t handle,
    at::ScalarType scalar_type,
    const void* hidden,
    const void* weight,
    float* logits,
    int64_t block_tokens,
    int64_t hidden_dim,
    int64_t vocab_size,
    bool weight_dim_first) {
  float alpha = 1.0f;
  float beta = 0.0f;
  cudaDataType_t input_type = cuda_type(scalar_type);
  if (weight_dim_first) {
    CHECK_CUBLAS(cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        static_cast<int>(vocab_size),
        static_cast<int>(block_tokens),
        static_cast<int>(hidden_dim),
        &alpha,
        weight,
        input_type,
        static_cast<int>(vocab_size),
        hidden,
        input_type,
        static_cast<int>(hidden_dim),
        &beta,
        logits,
        CUDA_R_32F,
        static_cast<int>(vocab_size),
        compute_type(scalar_type),
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  } else {
    CHECK_CUBLAS(cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        static_cast<int>(vocab_size),
        static_cast<int>(block_tokens),
        static_cast<int>(hidden_dim),
        &alpha,
        weight,
        input_type,
        static_cast<int>(hidden_dim),
        hidden,
        input_type,
        static_cast<int>(hidden_dim),
        &beta,
        logits,
        CUDA_R_32F,
        static_cast<int>(vocab_size),
        compute_type(scalar_type),
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
}

void project_grad_hidden(
    cublasHandle_t handle,
    at::ScalarType scalar_type,
    const void* grad_logits,
    const void* weight,
    float* grad_hidden,
    int64_t block_tokens,
    int64_t hidden_dim,
    int64_t vocab_size,
    bool weight_dim_first) {
  float alpha = 1.0f;
  float beta = 0.0f;
  cudaDataType_t input_type = cuda_type(scalar_type);
  if (weight_dim_first) {
    CHECK_CUBLAS(cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        static_cast<int>(hidden_dim),
        static_cast<int>(block_tokens),
        static_cast<int>(vocab_size),
        &alpha,
        weight,
        input_type,
        static_cast<int>(vocab_size),
        grad_logits,
        input_type,
        static_cast<int>(vocab_size),
        &beta,
        grad_hidden,
        CUDA_R_32F,
        static_cast<int>(hidden_dim),
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  } else {
    CHECK_CUBLAS(cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        static_cast<int>(hidden_dim),
        static_cast<int>(block_tokens),
        static_cast<int>(vocab_size),
        &alpha,
        weight,
        input_type,
        static_cast<int>(hidden_dim),
        grad_logits,
        input_type,
        static_cast<int>(vocab_size),
        &beta,
        grad_hidden,
        CUDA_R_32F,
        static_cast<int>(hidden_dim),
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
}

void accumulate_grad_weight(
    cublasHandle_t handle,
    at::ScalarType scalar_type,
    const void* hidden,
    const void* grad_logits,
    float* grad_weight,
    int64_t block_tokens,
    int64_t hidden_dim,
    int64_t vocab_size,
    bool weight_dim_first,
    bool first_chunk) {
  float alpha = 1.0f;
  float beta = first_chunk ? 0.0f : 1.0f;
  cudaDataType_t input_type = cuda_type(scalar_type);
  if (weight_dim_first) {
    CHECK_CUBLAS(cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        static_cast<int>(vocab_size),
        static_cast<int>(hidden_dim),
        static_cast<int>(block_tokens),
        &alpha,
        grad_logits,
        input_type,
        static_cast<int>(vocab_size),
        hidden,
        input_type,
        static_cast<int>(hidden_dim),
        &beta,
        grad_weight,
        CUDA_R_32F,
        static_cast<int>(vocab_size),
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  } else {
    CHECK_CUBLAS(cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        static_cast<int>(hidden_dim),
        static_cast<int>(vocab_size),
        static_cast<int>(block_tokens),
        &alpha,
        hidden,
        input_type,
        static_cast<int>(hidden_dim),
        grad_logits,
        input_type,
        static_cast<int>(vocab_size),
        &beta,
        grad_weight,
        CUDA_R_32F,
        static_cast<int>(hidden_dim),
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
}

int64_t scalar_type_bytes(at::ScalarType scalar_type) {
  switch (scalar_type) {
    case at::kFloat:
      return 4;
    case at::kHalf:
    case at::kBFloat16:
      return 2;
    default:
      return 4;
  }
}

int64_t native_ce_block_size(
    int64_t requested_block_size,
    int64_t num_tokens,
    int64_t vocab_size,
    at::ScalarType compute_scalar_type) {
  if (num_tokens <= 0) {
    return 1;
  }
  if (requested_block_size > 0) {
    return std::max<int64_t>(1, std::min(requested_block_size, num_tokens));
  }
  cudaDeviceProp props;
  C10_CUDA_CHECK(cudaGetDeviceProperties(&props, at::cuda::current_device()));
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  C10_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  const int64_t fp32_bytes = 4;
  const int64_t compute_bytes = scalar_type_bytes(compute_scalar_type);
  const int64_t bytes_per_token =
      std::max<int64_t>(1, vocab_size) * (2 * fp32_bytes + compute_bytes);
  (void)total_bytes;
  const int64_t resident_blocks_per_sm =
      std::max<int64_t>(1, props.maxThreadsPerMultiProcessor / kThreads);
  const int64_t resident_rows =
      std::max<int64_t>(1, static_cast<int64_t>(props.multiProcessorCount) *
                               resident_blocks_per_sm);
  const int64_t memory_rows =
      std::max<int64_t>(1, static_cast<int64_t>(free_bytes) / bytes_per_token);
  int64_t target_rows = std::min<int64_t>(resident_rows, memory_rows);
  const int64_t gemm_row_granularity = std::max<int64_t>(1, kThreads / 2);
  if (target_rows >= gemm_row_granularity) {
    target_rows = (target_rows / gemm_row_granularity) * gemm_row_granularity;
  }
  return std::max<int64_t>(
      1,
      std::min<int64_t>(num_tokens, target_rows));
}

int64_t normalized_block_size(
    int64_t row_block_size,
    int64_t num_tokens,
    int64_t vocab_size,
    at::ScalarType compute_scalar_type) {
  if (row_block_size <= 0) {
    return native_ce_block_size(
        row_block_size,
        num_tokens,
        vocab_size,
        compute_scalar_type);
  }
  if (row_block_size > num_tokens) {
    return num_tokens;
  }
  return row_block_size;
}

int64_t dflash_kl_block_size(
    int64_t requested_block_size,
    int64_t num_tokens,
    int64_t vocab_size,
    at::ScalarType scalar_type,
    int64_t fp32_workspace_count) {
  if (num_tokens <= 0) {
    return 1;
  }
  if (requested_block_size > 0) {
    return std::max<int64_t>(1, std::min(requested_block_size, num_tokens));
  }
  cudaDeviceProp props;
  C10_CUDA_CHECK(cudaGetDeviceProperties(&props, at::cuda::current_device()));
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  C10_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  (void)total_bytes;
  int64_t bytes_per_token = std::max<int64_t>(1, vocab_size) * (
      fp32_workspace_count * static_cast<int64_t>(sizeof(float))
      + scalar_type_bytes(scalar_type));
  int64_t resident_blocks_per_sm = std::max<int64_t>(
      1,
      props.maxThreadsPerMultiProcessor / kThreads);
  int64_t resident_rows = std::max<int64_t>(
      1,
      static_cast<int64_t>(props.multiProcessorCount) * resident_blocks_per_sm);
  int64_t memory_rows = std::max<int64_t>(
      1,
      static_cast<int64_t>(free_bytes) / bytes_per_token);
  return std::max<int64_t>(
      1,
      std::min<int64_t>(num_tokens, std::min(resident_rows, memory_rows)));
}

}  // namespace

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
    bool has_bias) {
  const c10::cuda::OptionalCUDAGuard device_guard(hidden.device());
  auto hidden_c = hidden.contiguous();
  auto weight_c = weight.contiguous();
  auto labels_c = labels.reshape({-1}).to(torch::kLong).contiguous();
  if (has_bias) {
    TORCH_CHECK(bias.is_cuda(), "native linear CE expects CUDA bias");
    bias = bias.to(torch::kFloat32).contiguous();
  }
  int64_t num_tokens = hidden_c.size(0);
  int64_t hidden_dim = hidden_c.size(1);
  int64_t vocab_size = weight_dim_first ? weight_c.size(1) : weight_c.size(0);
  TORCH_CHECK(vocab_size <= std::numeric_limits<int>::max(),
              "native linear CE vocab_size exceeds cuBLAS int range");
  TORCH_CHECK(hidden_dim <= std::numeric_limits<int>::max(),
              "native linear CE hidden_dim exceeds cuBLAS int range");

  auto float_options = hidden_c.options().dtype(torch::kFloat32);
  auto losses = torch::zeros({num_tokens}, float_options);
  auto lse = torch::empty({num_tokens}, float_options);
  auto valid_count = torch::zeros({1}, float_options);
  int64_t block_size = normalized_block_size(
      row_block_size,
      num_tokens,
      vocab_size,
      hidden_c.scalar_type());
  float softcap = static_cast<float>(logit_softcap);
  auto logits = torch::empty({block_size, vocab_size}, float_options);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  CHECK_CUBLAS(cublasSetStream(handle, stream));

  for (int64_t token_start = 0; token_start < num_tokens; token_start += block_size) {
    int64_t block_tokens = std::min(block_size, num_tokens - token_start);
    auto hidden_block = hidden_c.narrow(0, token_start, block_tokens);
    auto logits_block = logits.narrow(0, 0, block_tokens);
    project_logits(
        handle,
        hidden_c.scalar_type(),
        hidden_block.data_ptr(),
        weight_c.data_ptr(),
        logits_block.data_ptr<float>(),
        block_tokens,
        hidden_dim,
        vocab_size,
        weight_dim_first);
    if (softcap > 0.0f) {
      ce_forward_kernel<true><<<block_tokens, kThreads, 0, stream>>>(
          logits_block.data_ptr<float>(),
          has_bias ? bias.data_ptr<float>() : nullptr,
          labels_c.data_ptr<int64_t>(),
          losses.data_ptr<float>(),
          lse.data_ptr<float>(),
          valid_count.data_ptr<float>(),
          token_start,
          block_tokens,
          vocab_size,
          ignore_index,
          exclude_index,
          softcap,
          has_bias);
    } else {
      ce_forward_kernel<false><<<block_tokens, kThreads, 0, stream>>>(
          logits_block.data_ptr<float>(),
          has_bias ? bias.data_ptr<float>() : nullptr,
          labels_c.data_ptr<int64_t>(),
          losses.data_ptr<float>(),
          lse.data_ptr<float>(),
          valid_count.data_ptr<float>(),
          token_start,
          block_tokens,
          vocab_size,
          ignore_index,
          exclude_index,
          softcap,
          has_bias);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {losses, lse, valid_count};
}

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
    bool compute_weight_grad) {
  const c10::cuda::OptionalCUDAGuard device_guard(hidden.device());
  auto hidden_c = hidden.contiguous();
  auto weight_c = weight.contiguous();
  auto labels_c = labels.reshape({-1}).to(torch::kLong).contiguous();
  auto lse_c = lse.contiguous();
  auto valid_count_c = valid_count.contiguous();
  auto grad_output_c = grad_output.reshape({-1}).to(torch::kFloat32).contiguous();
  auto bias_input = bias;
  if (has_bias) {
    TORCH_CHECK(bias.is_cuda(), "native linear CE backward expects CUDA bias");
    bias = bias.to(torch::kFloat32).contiguous();
  }

  int64_t num_tokens = hidden_c.size(0);
  int64_t hidden_dim = hidden_c.size(1);
  int64_t vocab_size = weight_dim_first ? weight_c.size(1) : weight_c.size(0);
  auto float_options = hidden_c.options().dtype(torch::kFloat32);
  auto grad_hidden_f = torch::empty({num_tokens, hidden_dim}, float_options);
  auto grad_weight_f = compute_weight_grad
      ? torch::empty_like(weight_c, float_options)
      : torch::empty({0}, float_options);
  auto grad_bias_f = has_bias
      ? torch::zeros({vocab_size}, float_options)
      : torch::empty({0}, float_options);
  int64_t block_size = normalized_block_size(
      row_block_size,
      num_tokens,
      vocab_size,
      hidden_c.scalar_type());
  float softcap = static_cast<float>(logit_softcap);
  auto logits = torch::empty({block_size, vocab_size}, float_options);
  auto grad_logits = torch::empty({block_size, vocab_size}, float_options);
  auto grad_logits_gemm = hidden_c.scalar_type() == at::kFloat
      ? grad_logits
      : torch::empty({block_size, vocab_size}, hidden_c.options());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  CHECK_CUBLAS(cublasSetStream(handle, stream));

  bool first_chunk = true;
  for (int64_t token_start = 0; token_start < num_tokens; token_start += block_size) {
    int64_t block_tokens = std::min(block_size, num_tokens - token_start);
    auto hidden_block = hidden_c.narrow(0, token_start, block_tokens);
    auto grad_hidden_block = grad_hidden_f.narrow(0, token_start, block_tokens);
    auto logits_block = logits.narrow(0, 0, block_tokens);
    auto grad_logits_block = grad_logits.narrow(0, 0, block_tokens);
    auto grad_logits_gemm_block = grad_logits_gemm.narrow(0, 0, block_tokens);

    project_logits(
        handle,
        hidden_c.scalar_type(),
        hidden_block.data_ptr(),
        weight_c.data_ptr(),
        logits_block.data_ptr<float>(),
        block_tokens,
        hidden_dim,
        vocab_size,
        weight_dim_first);
    int blocks = static_cast<int>(
        std::min<int64_t>((block_tokens * vocab_size + 255) / 256, 65535));
    if (softcap > 0.0f) {
      ce_grad_logits_kernel<true><<<blocks, 256, 0, stream>>>(
          logits_block.data_ptr<float>(),
          has_bias ? bias.data_ptr<float>() : nullptr,
          labels_c.data_ptr<int64_t>(),
          lse_c.data_ptr<float>(),
          valid_count_c.data_ptr<float>(),
          grad_output_c.data_ptr<float>(),
          grad_logits_block.data_ptr<float>(),
          has_bias ? grad_bias_f.data_ptr<float>() : nullptr,
          token_start,
          block_tokens,
          vocab_size,
          reduction_code,
          vocab_start,
          ignore_index,
          exclude_index,
          softcap,
          has_bias);
    } else {
      ce_grad_logits_kernel<false><<<blocks, 256, 0, stream>>>(
          logits_block.data_ptr<float>(),
          has_bias ? bias.data_ptr<float>() : nullptr,
          labels_c.data_ptr<int64_t>(),
          lse_c.data_ptr<float>(),
          valid_count_c.data_ptr<float>(),
          grad_output_c.data_ptr<float>(),
          grad_logits_block.data_ptr<float>(),
          has_bias ? grad_bias_f.data_ptr<float>() : nullptr,
          token_start,
          block_tokens,
          vocab_size,
          reduction_code,
          vocab_start,
          ignore_index,
          exclude_index,
          softcap,
          has_bias);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if (hidden_c.scalar_type() == at::kFloat) {
      grad_logits_gemm_block = grad_logits_block;
    } else {
      AT_DISPATCH_FLOATING_TYPES_AND2(
          at::kHalf,
          at::kBFloat16,
          hidden_c.scalar_type(),
          "native_linear_ce_grad_logits_cast_cuda",
          [&] {
            launch_cast<scalar_t>(
                grad_logits_block, grad_logits_gemm_block, stream);
          });
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    project_grad_hidden(
        handle,
        hidden_c.scalar_type(),
        grad_logits_gemm_block.data_ptr(),
        weight_c.data_ptr(),
        grad_hidden_block.data_ptr<float>(),
        block_tokens,
        hidden_dim,
        vocab_size,
        weight_dim_first);
    if (compute_weight_grad) {
      accumulate_grad_weight(
          handle,
          hidden_c.scalar_type(),
          hidden_block.data_ptr(),
          grad_logits_gemm_block.data_ptr(),
          grad_weight_f.data_ptr<float>(),
          block_tokens,
          hidden_dim,
          vocab_size,
          weight_dim_first,
          first_chunk);
      first_chunk = false;
    }
  }

  auto grad_hidden = torch::empty_like(hidden_c);
  auto grad_weight = compute_weight_grad
      ? torch::empty_like(weight_c)
      : torch::empty({0}, weight_c.options());
  torch::Tensor grad_bias = has_bias ? torch::empty_like(bias_input) : grad_bias_f;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf,
      at::kBFloat16,
      hidden_c.scalar_type(),
      "native_linear_ce_cast_cuda",
      [&] {
        launch_cast<scalar_t>(grad_hidden_f, grad_hidden, stream);
        if (compute_weight_grad) {
          launch_cast<scalar_t>(grad_weight_f, grad_weight, stream);
        }
        if (has_bias) {
          launch_cast<scalar_t>(grad_bias_f, grad_bias, stream);
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_hidden, grad_weight, grad_bias};
}

std::vector<torch::Tensor> fused_vocab_parallel_ce_forward_cuda(
    torch::Tensor hidden,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor labels,
    int64_t vocab_start,
    int64_t ignore_index,
    int64_t exclude_index,
    double logit_softcap,
    bool has_bias) {
  const c10::cuda::OptionalCUDAGuard device_guard(hidden.device());
  auto hidden_c = hidden.contiguous();
  auto weight_c = weight.contiguous();
  auto labels_c = labels.reshape({-1}).to(torch::kLong).contiguous();
  if (has_bias) {
    TORCH_CHECK(bias.is_cuda(), "vocab-parallel CE expects CUDA bias");
    bias = bias.to(torch::kFloat32).contiguous();
  }
  TORCH_CHECK(hidden_c.dim() == 2, "vocab-parallel hidden must be [N, D]");
  TORCH_CHECK(weight_c.dim() == 2, "vocab-parallel weight must be [V_local, D]");
  TORCH_CHECK(weight_c.size(1) == hidden_c.size(1), "vocab-parallel head width mismatch");
  TORCH_CHECK(labels_c.numel() == hidden_c.size(0), "vocab-parallel label count mismatch");

  int64_t num_tokens = hidden_c.size(0);
  int64_t hidden_dim = hidden_c.size(1);
  int64_t vocab_size = weight_c.size(0);
  auto float_options = hidden_c.options().dtype(torch::kFloat32);
  auto local_max = torch::empty({num_tokens}, float_options);
  auto local_sum = torch::empty({num_tokens}, float_options);
  auto target_logits = torch::zeros({num_tokens}, float_options);
  auto valid_count = torch::zeros({1}, float_options);
  int64_t block_size = normalized_block_size(
      0,
      num_tokens,
      vocab_size,
      hidden_c.scalar_type());
  auto logits = torch::empty({block_size, vocab_size}, float_options);
  float softcap = static_cast<float>(logit_softcap);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  CHECK_CUBLAS(cublasSetStream(handle, stream));
  for (int64_t token_start = 0; token_start < num_tokens; token_start += block_size) {
    int64_t block_tokens = std::min(block_size, num_tokens - token_start);
    auto hidden_block = hidden_c.narrow(0, token_start, block_tokens);
    auto logits_block = logits.narrow(0, 0, block_tokens);
    project_logits(
        handle,
        hidden_c.scalar_type(),
        hidden_block.data_ptr(),
        weight_c.data_ptr(),
        logits_block.data_ptr<float>(),
        block_tokens,
        hidden_dim,
        vocab_size,
        false);
    if (softcap > 0.0f) {
      vocab_parallel_ce_stats_kernel<true><<<block_tokens, kThreads, 0, stream>>>(
          logits_block.data_ptr<float>(),
          has_bias ? bias.data_ptr<float>() : nullptr,
          labels_c.data_ptr<int64_t>(),
          local_max.data_ptr<float>(),
          local_sum.data_ptr<float>(),
          target_logits.data_ptr<float>(),
          valid_count.data_ptr<float>(),
          token_start,
          block_tokens,
          vocab_size,
          vocab_start,
          ignore_index,
          exclude_index,
          softcap,
          has_bias);
    } else {
      vocab_parallel_ce_stats_kernel<false><<<block_tokens, kThreads, 0, stream>>>(
          logits_block.data_ptr<float>(),
          has_bias ? bias.data_ptr<float>() : nullptr,
          labels_c.data_ptr<int64_t>(),
          local_max.data_ptr<float>(),
          local_sum.data_ptr<float>(),
          target_logits.data_ptr<float>(),
          valid_count.data_ptr<float>(),
          token_start,
          block_tokens,
          vocab_size,
          vocab_start,
          ignore_index,
          exclude_index,
          softcap,
          has_bias);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {local_max, local_sum, target_logits, valid_count};
}

std::vector<torch::Tensor> dflash_frozen_kl_forward_cuda(
    torch::Tensor draft_hidden,
    torch::Tensor draft_weight,
    torch::Tensor teacher_hidden,
    torch::Tensor teacher_weight,
    double logit_softcap) {
  const c10::cuda::OptionalCUDAGuard device_guard(draft_hidden.device());
  auto draft_hidden_c = draft_hidden.contiguous();
  auto draft_weight_c = draft_weight.contiguous();
  auto teacher_hidden_c = teacher_hidden.contiguous();
  auto teacher_weight_c = teacher_weight.contiguous();
  TORCH_CHECK(draft_hidden_c.dim() == 2 && teacher_hidden_c.dim() == 2,
              "DFlash KL hidden states must be [N, D]");
  TORCH_CHECK(draft_weight_c.dim() == 2 && teacher_weight_c.dim() == 2,
              "DFlash KL heads must be [V, D]");
  TORCH_CHECK(draft_hidden_c.size(0) == teacher_hidden_c.size(0),
              "DFlash KL draft and teacher token counts differ");
  TORCH_CHECK(draft_weight_c.size(0) == teacher_weight_c.size(0),
              "DFlash KL draft and teacher vocabulary sizes differ");
  TORCH_CHECK(draft_weight_c.size(1) == draft_hidden_c.size(1),
              "DFlash KL draft head width is incompatible");
  TORCH_CHECK(teacher_weight_c.size(1) == teacher_hidden_c.size(1),
              "DFlash KL teacher head width is incompatible");
  TORCH_CHECK(draft_hidden_c.scalar_type() == draft_weight_c.scalar_type() &&
              draft_hidden_c.scalar_type() == teacher_hidden_c.scalar_type() &&
              draft_hidden_c.scalar_type() == teacher_weight_c.scalar_type(),
              "DFlash KL tensors must have one compute dtype");
  TORCH_CHECK(logit_softcap >= 0.0,
              "DFlash KL logit softcap must be non-negative");

  int64_t num_tokens = draft_hidden_c.size(0);
  int64_t draft_dim = draft_hidden_c.size(1);
  int64_t teacher_dim = teacher_hidden_c.size(1);
  int64_t vocab_size = draft_weight_c.size(0);
  auto float_options = draft_hidden_c.options().dtype(torch::kFloat32);
  auto losses = torch::empty({num_tokens}, float_options);
  auto draft_lse = torch::empty({num_tokens}, float_options);
  auto teacher_lse = torch::empty({num_tokens}, float_options);
  int64_t block_size = dflash_kl_block_size(
      0,
      num_tokens,
      vocab_size,
      draft_hidden_c.scalar_type(),
      2);
  auto draft_logits = torch::empty({block_size, vocab_size}, float_options);
  auto teacher_logits = torch::empty({block_size, vocab_size}, float_options);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  CHECK_CUBLAS(cublasSetStream(handle, stream));
  for (int64_t token_start = 0; token_start < num_tokens; token_start += block_size) {
    int64_t block_tokens = std::min(block_size, num_tokens - token_start);
    auto draft_hidden_block = draft_hidden_c.narrow(0, token_start, block_tokens);
    auto teacher_hidden_block = teacher_hidden_c.narrow(0, token_start, block_tokens);
    auto draft_logits_block = draft_logits.narrow(0, 0, block_tokens);
    auto teacher_logits_block = teacher_logits.narrow(0, 0, block_tokens);
    project_logits(
        handle,
        draft_hidden_c.scalar_type(),
        draft_hidden_block.data_ptr(),
        draft_weight_c.data_ptr(),
        draft_logits_block.data_ptr<float>(),
        block_tokens,
        draft_dim,
        vocab_size,
        false);
    project_logits(
        handle,
        teacher_hidden_c.scalar_type(),
        teacher_hidden_block.data_ptr(),
        teacher_weight_c.data_ptr(),
        teacher_logits_block.data_ptr<float>(),
        block_tokens,
        teacher_dim,
        vocab_size,
        false);
    if (logit_softcap > 0.0) {
      dflash_kl_forward_kernel<true><<<block_tokens, kThreads, 0, stream>>>(
          draft_logits_block.data_ptr<float>(),
          teacher_logits_block.data_ptr<float>(),
          losses.data_ptr<float>(),
          draft_lse.data_ptr<float>(),
          teacher_lse.data_ptr<float>(),
          token_start,
          block_tokens,
          vocab_size,
          static_cast<float>(logit_softcap));
    } else {
      dflash_kl_forward_kernel<false><<<block_tokens, kThreads, 0, stream>>>(
          draft_logits_block.data_ptr<float>(),
          teacher_logits_block.data_ptr<float>(),
          losses.data_ptr<float>(),
          draft_lse.data_ptr<float>(),
          teacher_lse.data_ptr<float>(),
          token_start,
          block_tokens,
          vocab_size,
          0.0f);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {losses, draft_lse, teacher_lse};
}

torch::Tensor dflash_frozen_kl_backward_cuda(
    torch::Tensor draft_hidden,
    torch::Tensor draft_weight,
    torch::Tensor teacher_hidden,
    torch::Tensor teacher_weight,
    torch::Tensor draft_lse,
    torch::Tensor teacher_lse,
    torch::Tensor grad_output,
    double logit_softcap) {
  const c10::cuda::OptionalCUDAGuard device_guard(draft_hidden.device());
  auto draft_hidden_c = draft_hidden.contiguous();
  auto draft_weight_c = draft_weight.contiguous();
  auto teacher_hidden_c = teacher_hidden.contiguous();
  auto teacher_weight_c = teacher_weight.contiguous();
  auto draft_lse_c = draft_lse.contiguous();
  auto teacher_lse_c = teacher_lse.contiguous();
  auto grad_output_c = grad_output.reshape({-1}).to(torch::kFloat32).contiguous();
  int64_t num_tokens = draft_hidden_c.size(0);
  int64_t draft_dim = draft_hidden_c.size(1);
  int64_t teacher_dim = teacher_hidden_c.size(1);
  int64_t vocab_size = draft_weight_c.size(0);
  TORCH_CHECK(grad_output_c.numel() == num_tokens,
              "DFlash KL gradient must contain one value per token");
  auto float_options = draft_hidden_c.options().dtype(torch::kFloat32);
  auto grad_hidden_f = torch::empty({num_tokens, draft_dim}, float_options);
  int64_t block_size = dflash_kl_block_size(
      0,
      num_tokens,
      vocab_size,
      draft_hidden_c.scalar_type(),
      3);
  auto draft_logits = torch::empty({block_size, vocab_size}, float_options);
  auto teacher_logits = torch::empty({block_size, vocab_size}, float_options);
  auto grad_logits = torch::empty({block_size, vocab_size}, float_options);
  auto grad_logits_gemm = draft_hidden_c.scalar_type() == at::kFloat
      ? grad_logits
      : torch::empty({block_size, vocab_size}, draft_hidden_c.options());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  CHECK_CUBLAS(cublasSetStream(handle, stream));
  for (int64_t token_start = 0; token_start < num_tokens; token_start += block_size) {
    int64_t block_tokens = std::min(block_size, num_tokens - token_start);
    auto draft_hidden_block = draft_hidden_c.narrow(0, token_start, block_tokens);
    auto teacher_hidden_block = teacher_hidden_c.narrow(0, token_start, block_tokens);
    auto draft_logits_block = draft_logits.narrow(0, 0, block_tokens);
    auto teacher_logits_block = teacher_logits.narrow(0, 0, block_tokens);
    auto grad_logits_block = grad_logits.narrow(0, 0, block_tokens);
    auto grad_logits_gemm_block = grad_logits_gemm.narrow(0, 0, block_tokens);
    auto grad_hidden_block = grad_hidden_f.narrow(0, token_start, block_tokens);
    project_logits(
        handle,
        draft_hidden_c.scalar_type(),
        draft_hidden_block.data_ptr(),
        draft_weight_c.data_ptr(),
        draft_logits_block.data_ptr<float>(),
        block_tokens,
        draft_dim,
        vocab_size,
        false);
    project_logits(
        handle,
        teacher_hidden_c.scalar_type(),
        teacher_hidden_block.data_ptr(),
        teacher_weight_c.data_ptr(),
        teacher_logits_block.data_ptr<float>(),
        block_tokens,
        teacher_dim,
        vocab_size,
        false);
    int blocks = static_cast<int>(std::min<int64_t>(
        (block_tokens * vocab_size + kThreads - 1) / kThreads,
        65535));
    if (logit_softcap > 0.0) {
      dflash_kl_grad_logits_kernel<true><<<blocks, kThreads, 0, stream>>>(
          draft_logits_block.data_ptr<float>(),
          teacher_logits_block.data_ptr<float>(),
          draft_lse_c.data_ptr<float>(),
          teacher_lse_c.data_ptr<float>(),
          grad_output_c.data_ptr<float>(),
          grad_logits_block.data_ptr<float>(),
          token_start,
          block_tokens,
          vocab_size,
          static_cast<float>(logit_softcap));
    } else {
      dflash_kl_grad_logits_kernel<false><<<blocks, kThreads, 0, stream>>>(
          draft_logits_block.data_ptr<float>(),
          teacher_logits_block.data_ptr<float>(),
          draft_lse_c.data_ptr<float>(),
          teacher_lse_c.data_ptr<float>(),
          grad_output_c.data_ptr<float>(),
          grad_logits_block.data_ptr<float>(),
          token_start,
          block_tokens,
          vocab_size,
          0.0f);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (draft_hidden_c.scalar_type() != at::kFloat) {
      AT_DISPATCH_FLOATING_TYPES_AND2(
          at::kHalf,
          at::kBFloat16,
          draft_hidden_c.scalar_type(),
          "dflash_kl_grad_logits_cast_cuda",
          [&] {
            launch_cast<scalar_t>(
                grad_logits_block,
                grad_logits_gemm_block,
                stream);
          });
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    project_grad_hidden(
        handle,
        draft_hidden_c.scalar_type(),
        grad_logits_gemm_block.data_ptr(),
        draft_weight_c.data_ptr(),
        grad_hidden_block.data_ptr<float>(),
        block_tokens,
        draft_dim,
        vocab_size,
        false);
  }
  auto grad_hidden = torch::empty_like(draft_hidden_c);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf,
      at::kBFloat16,
      draft_hidden_c.scalar_type(),
      "dflash_kl_grad_hidden_cast_cuda",
      [&] {
        launch_cast<scalar_t>(grad_hidden_f, grad_hidden, stream);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_hidden;
}

std::vector<torch::Tensor> dflash_frozen_ce_topk_forward_cuda(
    torch::Tensor hidden,
    torch::Tensor weight,
    torch::Tensor labels,
    int64_t top_k,
    int64_t row_block_size,
    double output_multiplier,
    double logit_softcap) {
  const c10::cuda::OptionalCUDAGuard device_guard(hidden.device());
  auto hidden_c = hidden.contiguous();
  auto weight_c = weight.contiguous();
  auto labels_c = labels.reshape({-1}).to(torch::kLong).contiguous();
  TORCH_CHECK(hidden_c.dim() == 2, "DFlash CE/top-k hidden must be [N, D]");
  TORCH_CHECK(weight_c.dim() == 2, "DFlash CE/top-k weight must be [V, D]");
  TORCH_CHECK(weight_c.size(1) == hidden_c.size(1), "DFlash CE/top-k head width mismatch");
  TORCH_CHECK(labels_c.numel() == hidden_c.size(0), "DFlash CE/top-k label count mismatch");
  TORCH_CHECK(top_k > 0 && top_k <= weight_c.size(0), "DFlash CE/top-k invalid top_k");
  TORCH_CHECK(hidden_c.scalar_type() == weight_c.scalar_type(), "DFlash CE/top-k dtype mismatch");

  int64_t num_tokens = hidden_c.size(0);
  int64_t vocab_size = weight_c.size(0);
  auto float_options = hidden_c.options().dtype(torch::kFloat32);
  int64_t block_size = dflash_kl_block_size(
      row_block_size,
      num_tokens,
      vocab_size,
      hidden_c.scalar_type(),
      4);
  auto losses = torch::empty({num_tokens}, float_options);
  auto probabilities = torch::empty({num_tokens}, float_options);
  auto top_values = torch::empty({num_tokens, top_k}, float_options);
  auto top_ids = torch::empty({num_tokens, top_k}, labels_c.options());
  auto lse = torch::empty({num_tokens}, float_options);
  auto raw_logits = torch::empty({block_size, vocab_size}, float_options);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  CHECK_CUBLAS(cublasSetStream(handle, stream));

  for (int64_t token_start = 0; token_start < num_tokens; token_start += block_size) {
    int64_t block_tokens = std::min(block_size, num_tokens - token_start);
    auto hidden_block = hidden_c.narrow(0, token_start, block_tokens);
    auto raw_logits_block = raw_logits.narrow(0, 0, block_tokens);
    project_logits(
        handle,
        hidden_c.scalar_type(),
        hidden_block.data_ptr(),
        weight_c.data_ptr(),
        raw_logits_block.data_ptr<float>(),
        block_tokens,
        hidden_c.size(1),
        vocab_size,
        false);
    if (logit_softcap > 0.0) {
      dflash_ce_topk_stats_kernel<true><<<block_tokens, kThreads, 0, stream>>>(
          raw_logits_block.data_ptr<float>(),
          labels_c.data_ptr<int64_t>(),
          losses.data_ptr<float>(),
          probabilities.data_ptr<float>(),
          lse.data_ptr<float>(),
          token_start,
          block_tokens,
          vocab_size,
          static_cast<float>(output_multiplier),
          static_cast<float>(logit_softcap));
    } else {
      dflash_ce_topk_stats_kernel<false><<<block_tokens, kThreads, 0, stream>>>(
          raw_logits_block.data_ptr<float>(),
          labels_c.data_ptr<int64_t>(),
          losses.data_ptr<float>(),
          probabilities.data_ptr<float>(),
          lse.data_ptr<float>(),
          token_start,
          block_tokens,
          vocab_size,
          static_cast<float>(output_multiplier),
          0.0f);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Positive scaling preserves candidate order. For the common uncapped
    // DFlash2 head, only the tiny [rows, K] output is transformed. Preserve
    // PyTorch's exact tie behavior for optional saturating softcaps by ranking
    // the transformed values in that uncommon branch.
    auto top_source = raw_logits_block;
    if (logit_softcap > 0.0) {
      top_source = at::tanh(
          raw_logits_block * static_cast<float>(output_multiplier)
          / static_cast<float>(logit_softcap))
          * static_cast<float>(logit_softcap);
    }
    auto top = at::topk(top_source, top_k, -1, true, true);
    auto values = logit_softcap > 0.0
        ? std::get<0>(top)
        : std::get<0>(top) * static_cast<float>(output_multiplier);
    top_values.narrow(0, token_start, block_tokens).copy_(values);
    top_ids.narrow(0, token_start, block_tokens).copy_(std::get<1>(top));
  }
  return {losses, probabilities, top_values, top_ids, lse};
}

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
    double logit_softcap) {
  const c10::cuda::OptionalCUDAGuard device_guard(hidden.device());
  auto hidden_c = hidden.contiguous();
  auto weight_c = weight.contiguous();
  auto labels_c = labels.reshape({-1}).to(torch::kLong).contiguous();
  auto lse_c = lse.to(torch::kFloat32).contiguous();
  auto probability_c = probabilities.to(torch::kFloat32).contiguous();
  auto grad_loss_c = grad_loss.to(torch::kFloat32).contiguous();
  auto grad_probability_c = grad_probability.to(torch::kFloat32).contiguous();
  auto grad_top_c = grad_top_values.to(torch::kFloat32).contiguous();
  auto top_ids_c = top_ids.to(torch::kLong).contiguous();
  int64_t num_tokens = hidden_c.size(0);
  int64_t hidden_dim = hidden_c.size(1);
  int64_t vocab_size = weight_c.size(0);
  auto float_options = hidden_c.options().dtype(torch::kFloat32);
  int64_t block_size = dflash_kl_block_size(
      row_block_size,
      num_tokens,
      vocab_size,
      hidden_c.scalar_type(),
      5);
  auto raw_logits = torch::empty({block_size, vocab_size}, float_options);
  auto grad_logits = torch::empty({block_size, vocab_size}, float_options);
  auto grad_logits_gemm = hidden_c.scalar_type() == at::kFloat
      ? grad_logits
      : torch::empty({block_size, vocab_size}, hidden_c.options());
  auto grad_hidden_f = torch::empty({num_tokens, hidden_dim}, float_options);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  CHECK_CUBLAS(cublasSetStream(handle, stream));

  for (int64_t token_start = 0; token_start < num_tokens; token_start += block_size) {
    int64_t block_tokens = std::min(block_size, num_tokens - token_start);
    auto hidden_block = hidden_c.narrow(0, token_start, block_tokens);
    auto raw_logits_block = raw_logits.narrow(0, 0, block_tokens);
    auto grad_logits_block = grad_logits.narrow(0, 0, block_tokens);
    auto grad_logits_gemm_block = grad_logits_gemm.narrow(0, 0, block_tokens);
    auto grad_hidden_block = grad_hidden_f.narrow(0, token_start, block_tokens);
    project_logits(
        handle,
        hidden_c.scalar_type(),
        hidden_block.data_ptr(),
        weight_c.data_ptr(),
        raw_logits_block.data_ptr<float>(),
        block_tokens,
        hidden_dim,
        vocab_size,
        false);
    int blocks = static_cast<int>(std::min<int64_t>(
        (block_tokens * vocab_size + kThreads - 1) / kThreads,
        65535));
    if (logit_softcap > 0.0) {
      dflash_ce_topk_grad_logits_kernel<float, true>
          <<<blocks, kThreads, 0, stream>>>(
              raw_logits_block.data_ptr<float>(),
              labels_c.data_ptr<int64_t>(),
              lse_c.data_ptr<float>(),
              probability_c.data_ptr<float>(),
              grad_loss_c.data_ptr<float>(),
              grad_probability_c.data_ptr<float>(),
              grad_logits_block.data_ptr<float>(),
              token_start,
              block_tokens,
              vocab_size,
              static_cast<float>(output_multiplier),
              static_cast<float>(logit_softcap));
    } else {
      dflash_ce_topk_grad_logits_kernel<float, false>
          <<<blocks, kThreads, 0, stream>>>(
              raw_logits_block.data_ptr<float>(),
              labels_c.data_ptr<int64_t>(),
              lse_c.data_ptr<float>(),
              probability_c.data_ptr<float>(),
              grad_loss_c.data_ptr<float>(),
              grad_probability_c.data_ptr<float>(),
              grad_logits_block.data_ptr<float>(),
              token_start,
              block_tokens,
              vocab_size,
              static_cast<float>(output_multiplier),
              0.0f);
    }
    int scatter_blocks = static_cast<int>(
        (block_tokens * top_ids_c.size(1) + kThreads - 1) / kThreads);
    if (logit_softcap > 0.0) {
      dflash_ce_topk_scatter_kernel<float, true>
          <<<scatter_blocks, kThreads, 0, stream>>>(
              raw_logits_block.data_ptr<float>(),
              top_ids_c.data_ptr<int64_t>(),
              grad_top_c.data_ptr<float>(),
              grad_logits_block.data_ptr<float>(),
              token_start,
              block_tokens,
              vocab_size,
              top_ids_c.size(1),
              static_cast<float>(output_multiplier),
              static_cast<float>(logit_softcap));
    } else {
      dflash_ce_topk_scatter_kernel<float, false>
          <<<scatter_blocks, kThreads, 0, stream>>>(
              raw_logits_block.data_ptr<float>(),
              top_ids_c.data_ptr<int64_t>(),
              grad_top_c.data_ptr<float>(),
              grad_logits_block.data_ptr<float>(),
              token_start,
              block_tokens,
              vocab_size,
              top_ids_c.size(1),
              static_cast<float>(output_multiplier),
              0.0f);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (hidden_c.scalar_type() != at::kFloat) {
      AT_DISPATCH_FLOATING_TYPES_AND2(
          at::kHalf,
          at::kBFloat16,
          hidden_c.scalar_type(),
          "dflash_ce_topk_grad_logits_cast_cuda",
          [&] {
            launch_cast<scalar_t>(
                grad_logits_block,
                grad_logits_gemm_block,
                stream);
          });
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    project_grad_hidden(
        handle,
        hidden_c.scalar_type(),
        grad_logits_gemm_block.data_ptr(),
        weight_c.data_ptr(),
        grad_hidden_block.data_ptr<float>(),
        block_tokens,
        hidden_dim,
        vocab_size,
        false);
  }
  auto grad_hidden = torch::empty_like(hidden_c);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf,
      at::kBFloat16,
      hidden_c.scalar_type(),
      "dflash_ce_topk_grad_hidden_cast_cuda",
      [&] {
        launch_cast<scalar_t>(grad_hidden_f, grad_hidden, stream);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_hidden;
}
