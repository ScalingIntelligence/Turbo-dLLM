// Copyright 2026 The dllm_parallel Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");

#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <algorithm>

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = kThreadsPerBlock / kWarpSize;
constexpr int kMaxGridBlocks = 4096;

__device__ __forceinline__ float finite_weight(float value, float maximum) {
  return isfinite(value) ? expf(value - maximum) : 0.0f;
}

template <typename scalar_t>
__global__ void merge_bshd_kernel(
    float* __restrict__ numerator,
    float* __restrict__ m,
    float* __restrict__ l,
    const scalar_t* __restrict__ output,
    const float* __restrict__ lse,
    int64_t heads,
    int64_t query_len,
    int64_t head_dim,
    int64_t rows) {
  const int warp_id = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  for (int64_t row = int64_t(blockIdx.x) * kWarpsPerBlock + warp_id;
       row < rows;
       row += int64_t(gridDim.x) * kWarpsPerBlock) {
    const int64_t q = row % query_len;
    const int64_t h = (row / query_len) % heads;
    const int64_t b = row / (query_len * heads);
    float old_m = lane == 0 ? m[row] : 0.0f;
    float new_m = lane == 0 ? lse[row] : 0.0f;
    float old_l = lane == 0 ? l[row] : 0.0f;
    old_m = __shfl_sync(0xffffffff, old_m, 0);
    new_m = __shfl_sync(0xffffffff, new_m, 0);
    const float next_m = fmaxf(old_m, new_m);
    const float safe_m = isfinite(next_m) ? next_m : 0.0f;
    const float old_weight = finite_weight(old_m, safe_m);
    const float new_weight = finite_weight(new_m, safe_m);
    const int64_t numerator_row = row * head_dim;
    const int64_t output_row =
        ((b * query_len + q) * heads + h) * head_dim;
    for (int64_t d = lane; d < head_dim; d += kWarpSize) {
      numerator[numerator_row + d] =
          old_weight * numerator[numerator_row + d] +
          new_weight * static_cast<float>(output[output_row + d]);
    }
    if (lane == 0) {
      l[row] =
          old_weight * old_l +
          new_weight * static_cast<float>(isfinite(new_m));
      m[row] = next_m;
    }
  }
}

__global__ void merge_state_kernel(
    float* __restrict__ numerator,
    float* __restrict__ m,
    float* __restrict__ l,
    const float* __restrict__ incoming_numerator,
    const float* __restrict__ incoming_m,
    const float* __restrict__ incoming_l,
    int64_t head_dim,
    int64_t rows) {
  const int warp_id = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  for (int64_t row = int64_t(blockIdx.x) * kWarpsPerBlock + warp_id;
       row < rows;
       row += int64_t(gridDim.x) * kWarpsPerBlock) {
    float old_m = lane == 0 ? m[row] : 0.0f;
    float new_m = lane == 0 ? incoming_m[row] : 0.0f;
    float old_l = lane == 0 ? l[row] : 0.0f;
    float new_l = lane == 0 ? incoming_l[row] : 0.0f;
    old_m = __shfl_sync(0xffffffff, old_m, 0);
    new_m = __shfl_sync(0xffffffff, new_m, 0);
    const float next_m = fmaxf(old_m, new_m);
    const float safe_m = isfinite(next_m) ? next_m : 0.0f;
    const float old_weight = finite_weight(old_m, safe_m);
    const float new_weight = finite_weight(new_m, safe_m);
    const int64_t offset = row * head_dim;
    for (int64_t d = lane; d < head_dim; d += kWarpSize) {
      numerator[offset + d] =
          old_weight * numerator[offset + d] +
          new_weight * incoming_numerator[offset + d];
    }
    if (lane == 0) {
      l[row] = old_weight * old_l + new_weight * new_l;
      m[row] = next_m;
    }
  }
}

int blocks_for(int64_t work) {
  const int64_t blocks = (work + kThreadsPerBlock - 1) / kThreadsPerBlock;
  return static_cast<int>(
      std::max<int64_t>(1, std::min<int64_t>(blocks, kMaxGridBlocks)));
}

}  // namespace

void dflash_merge_bshd_cuda(
    torch::Tensor numerator,
    torch::Tensor m,
    torch::Tensor l,
    torch::Tensor output,
    torch::Tensor lse) {
  const int64_t heads = numerator.size(1);
  const int64_t query_len = numerator.size(2);
  const int64_t head_dim = numerator.size(3);
  const int64_t rows = m.numel();
  if (rows == 0 || head_dim == 0) {
    return;
  }
  const int blocks = blocks_for(rows * kWarpSize);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      output.scalar_type(),
      "dflash_merge_bshd_cuda",
      [&] {
        merge_bshd_kernel<scalar_t>
            <<<blocks,
               kThreadsPerBlock,
               0,
               c10::cuda::getCurrentCUDAStream()>>>(
                numerator.data_ptr<float>(),
                m.data_ptr<float>(),
                l.data_ptr<float>(),
                output.data_ptr<scalar_t>(),
                lse.data_ptr<float>(),
                heads,
                query_len,
                head_dim,
                rows);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dflash_merge_state_cuda(
    torch::Tensor numerator,
    torch::Tensor m,
    torch::Tensor l,
    torch::Tensor incoming_numerator,
    torch::Tensor incoming_m,
    torch::Tensor incoming_l) {
  const int64_t rows = m.numel();
  const int64_t head_dim = numerator.size(3);
  if (rows == 0 || head_dim == 0) {
    return;
  }
  merge_state_kernel<<<
      blocks_for(rows * kWarpSize),
      kThreadsPerBlock,
      0,
      c10::cuda::getCurrentCUDAStream()>>>(
      numerator.data_ptr<float>(),
      m.data_ptr<float>(),
      l.data_ptr<float>(),
      incoming_numerator.data_ptr<float>(),
      incoming_m.data_ptr<float>(),
      incoming_l.data_ptr<float>(),
      head_dim,
      rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
