// Copyright 2026 The dllm_parallel Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
//
// CUDA kernels for the BDLM CP/BP online-softmax stats merge. The numerics here
// are intentionally unchanged from the reference implementation; do not alter
// the reduction math without re-validating against merge_attention_stats.

#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <algorithm>

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kMaxGridBlocks = 4096;

__device__ __forceinline__ float finite_or_zero_weight(float value, float next_m) {
  return isfinite(value) ? expf(value - next_m) : 0.0f;
}

template <typename scalar_t>
__global__ void merge_full_kernel(
    float* __restrict__ old_num,
    float* __restrict__ old_m,
    float* __restrict__ old_l,
    const scalar_t* __restrict__ new_num,
    const float* __restrict__ new_m,
    const float* __restrict__ new_l,
    int64_t heads,
    int64_t query_len,
    int64_t rows,
    int64_t head_dim,
    int64_t new_stride_b,
    int64_t new_stride_h,
    int64_t new_stride_q,
    int64_t new_stride_d) {
  const int64_t total = rows * head_dim;
  for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
       linear < total;
       linear += int64_t(blockDim.x) * gridDim.x) {
    const int64_t row = linear / head_dim;
    const int64_t d = linear - row * head_dim;
    const int64_t q = row % query_len;
    const int64_t h = (row / query_len) % heads;
    const int64_t b = row / query_len / heads;
    const int64_t new_linear =
        b * new_stride_b + h * new_stride_h + q * new_stride_q + d * new_stride_d;
    const float om = old_m[row];
    const float nm = new_m[row];
    const float next_m = fmaxf(om, nm);
    const float safe_m = isfinite(next_m) ? next_m : 0.0f;
    const float old_w = finite_or_zero_weight(om, safe_m);
    const float new_w = finite_or_zero_weight(nm, safe_m);
    old_num[linear] =
        old_w * old_num[linear] + new_w * static_cast<float>(new_num[new_linear]);
    if (d == 0) {
      old_l[row] = old_w * old_l[row] + new_w * new_l[row];
      old_m[row] = next_m;
    }
  }
}

template <typename scalar_t>
__global__ void merge_compact_kernel(
    float* __restrict__ numerator,
    float* __restrict__ m,
    float* __restrict__ l,
    const int64_t* __restrict__ query_indices,
    const scalar_t* __restrict__ compact_output,
    const float* __restrict__ compact_lse,
    int64_t batch_heads,
    int64_t heads,
    int64_t query_len,
    int64_t compact_len,
    int64_t head_dim,
    int64_t compact_stride_b,
    int64_t compact_stride_h,
    int64_t compact_stride_q,
    int64_t compact_stride_d) {
  const int64_t total = batch_heads * compact_len * head_dim;
  for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
       linear < total;
       linear += int64_t(blockDim.x) * gridDim.x) {
    const int64_t d = linear % head_dim;
    const int64_t compact_row = linear / head_dim;
    const int64_t cq = compact_row % compact_len;
    const int64_t bh = compact_row / compact_len;
    const int64_t h = bh % heads;
    const int64_t b = bh / heads;
    const int64_t q = query_indices[cq];
    const int64_t full_row = bh * query_len + q;
    const int64_t full_linear = full_row * head_dim + d;
    const int64_t compact_linear =
        b * compact_stride_b + h * compact_stride_h +
        cq * compact_stride_q + d * compact_stride_d;

    const float om = m[full_row];
    const float nm = compact_lse[compact_row];
    const float next_m = fmaxf(om, nm);
    const float safe_m = isfinite(next_m) ? next_m : 0.0f;
    const float old_w = finite_or_zero_weight(om, safe_m);
    const float new_w = finite_or_zero_weight(nm, safe_m);
    numerator[full_linear] =
        old_w * numerator[full_linear] +
        new_w * static_cast<float>(compact_output[compact_linear]);
    if (d == 0) {
      l[full_row] = old_w * l[full_row] + new_w * (isfinite(nm) ? 1.0f : 0.0f);
      m[full_row] = next_m;
    }
  }
}

template <typename scalar_t>
__global__ void merge_backward_kernel(
    const scalar_t* __restrict__ shard_output,
    const float* __restrict__ shard_lse,
    const scalar_t* __restrict__ final_output,
    const float* __restrict__ final_lse,
    const scalar_t* __restrict__ grad_output,
    scalar_t* __restrict__ shard_grad_output,
    float* __restrict__ grad_lse,
    int64_t rows,
    int64_t head_dim,
    int64_t heads,
    int64_t query_len,
    int64_t shard_stride_b,
    int64_t shard_stride_h,
    int64_t shard_stride_q,
    int64_t shard_stride_d,
    int64_t final_stride_b,
    int64_t final_stride_h,
    int64_t final_stride_q,
    int64_t final_stride_d,
    int64_t grad_stride_b,
    int64_t grad_stride_h,
    int64_t grad_stride_q,
    int64_t grad_stride_d,
    int64_t shard_grad_stride_b,
    int64_t shard_grad_stride_h,
    int64_t shard_grad_stride_q,
    int64_t shard_grad_stride_d) {
  constexpr int warps_per_block = 8;
  constexpr int warp_size = 32;
  const int warp_id = threadIdx.x / warp_size;
  const int lane = threadIdx.x - warp_id * warp_size;
  const int64_t row = int64_t(blockIdx.x) * warps_per_block + warp_id;
  if (row >= rows) {
    return;
  }

  const float sm = shard_lse[row];
  const float fm = final_lse[row];
  const bool finite = isfinite(sm) && isfinite(fm);
  const float weight = finite ? expf(sm - fm) : 0.0f;
  const int64_t q = row % query_len;
  const int64_t h = (row / query_len) % heads;
  const int64_t b = row / query_len / heads;
  if (!finite) {
    for (int64_t d = lane; d < head_dim; d += warp_size) {
      const int64_t shard_grad_index =
          b * shard_grad_stride_b + h * shard_grad_stride_h +
          q * shard_grad_stride_q + d * shard_grad_stride_d;
      shard_grad_output[shard_grad_index] = static_cast<scalar_t>(0.0f);
    }
    if (lane == 0) {
      grad_lse[row] = 0.0f;
    }
    return;
  }
  float dot = 0.0f;
  for (int64_t d = lane; d < head_dim; d += warp_size) {
    const int64_t shard_index =
        b * shard_stride_b + h * shard_stride_h +
        q * shard_stride_q + d * shard_stride_d;
    const int64_t final_index =
        b * final_stride_b + h * final_stride_h +
        q * final_stride_q + d * final_stride_d;
    const int64_t grad_index =
        b * grad_stride_b + h * grad_stride_h +
        q * grad_stride_q + d * grad_stride_d;
    const int64_t shard_grad_index =
        b * shard_grad_stride_b + h * shard_grad_stride_h +
        q * shard_grad_stride_q + d * shard_grad_stride_d;
    const float go = static_cast<float>(grad_output[grad_index]);
    shard_grad_output[shard_grad_index] = static_cast<scalar_t>(go * weight);
    dot += go * (
        static_cast<float>(shard_output[shard_index]) -
        static_cast<float>(final_output[final_index]));
  }
  #pragma unroll
  for (int offset = warp_size / 2; offset > 0; offset /= 2) {
    dot += __shfl_down_sync(0xffffffff, dot, offset);
  }
  if (lane == 0) {
    grad_lse[row] = weight * dot;
  }
}

template <typename scalar_t>
__global__ void finalize_bshd_kernel(
    const float* __restrict__ numerator,
    const float* __restrict__ m,
    const float* __restrict__ l,
    scalar_t* __restrict__ output,
    float* __restrict__ final_lse,
    int64_t batch,
    int64_t heads,
    int64_t query_len,
    int64_t head_dim) {
  const int64_t rows = batch * heads * query_len;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const int64_t q = row % query_len;
    const int64_t h = (row / query_len) % heads;
    const int64_t b = row / query_len / heads;
    const float row_l = l[row];
    const bool valid = row_l > 0.0f;
    const float inverse_l = valid ? 1.0f / row_l : 0.0f;
    for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
      const int64_t bhqd = row * head_dim + d;
      const int64_t bqhd = ((b * query_len + q) * heads + h) * head_dim + d;
      output[bqhd] = static_cast<scalar_t>(numerator[bhqd] * inverse_l);
    }
    if (threadIdx.x == 0) {
      final_lse[row] = valid ? m[row] + logf(row_l) : -INFINITY;
    }
  }
}

int blocks_for(int64_t work) {
  const int64_t blocks = (work + kThreadsPerBlock - 1) / kThreadsPerBlock;
  return static_cast<int>(
      std::max<int64_t>(1, std::min<int64_t>(blocks, kMaxGridBlocks)));
}

}  // namespace

void bdlm_merge_full_cuda(
    torch::Tensor old_num,
    torch::Tensor old_m,
    torch::Tensor old_l,
    torch::Tensor new_num,
    torch::Tensor new_m,
    torch::Tensor new_l) {
  const int64_t head_dim = old_num.size(old_num.dim() - 1);
  const int64_t heads = old_num.size(1);
  const int64_t query_len = old_num.size(2);
  const int64_t rows = old_m.numel();
  const int64_t work = rows * head_dim;
  if (work == 0) {
    return;
  }
  constexpr int threads = 256;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      new_num.scalar_type(),
      "bdlm_merge_full_cuda",
      [&] {
        merge_full_kernel<scalar_t>
            <<<blocks_for(work), threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
                old_num.data_ptr<float>(),
                old_m.data_ptr<float>(),
                old_l.data_ptr<float>(),
                new_num.data_ptr<scalar_t>(),
                new_m.data_ptr<float>(),
                new_l.data_ptr<float>(),
                heads,
                query_len,
                rows,
                head_dim,
                new_num.stride(0),
                new_num.stride(1),
                new_num.stride(2),
                new_num.stride(3));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void bdlm_merge_compact_cuda(
    torch::Tensor numerator,
    torch::Tensor m,
    torch::Tensor l,
    torch::Tensor query_indices,
    torch::Tensor compact_output,
    torch::Tensor compact_lse) {
  const int64_t batch_heads = numerator.size(0) * numerator.size(1);
  const int64_t heads = numerator.size(1);
  const int64_t query_len = numerator.size(2);
  const int64_t compact_len = compact_output.size(2);
  const int64_t head_dim = numerator.size(3);
  const int64_t work = batch_heads * compact_len * head_dim;
  if (work == 0) {
    return;
  }
  constexpr int threads = 256;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      compact_output.scalar_type(),
      "bdlm_merge_compact_cuda",
      [&] {
        merge_compact_kernel<scalar_t>
            <<<blocks_for(work), threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
                numerator.data_ptr<float>(),
                m.data_ptr<float>(),
                l.data_ptr<float>(),
                query_indices.data_ptr<int64_t>(),
                compact_output.data_ptr<scalar_t>(),
                compact_lse.data_ptr<float>(),
                batch_heads,
                heads,
                query_len,
                compact_len,
                head_dim,
                compact_output.stride(0),
                compact_output.stride(1),
                compact_output.stride(2),
                compact_output.stride(3));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void bdlm_merge_backward_cuda(
    torch::Tensor shard_output,
    torch::Tensor shard_lse,
    torch::Tensor final_output,
    torch::Tensor final_lse,
    torch::Tensor grad_output,
    torch::Tensor shard_grad_output,
    torch::Tensor grad_lse) {
  const int64_t rows = shard_lse.numel();
  const int64_t head_dim = shard_output.size(3);
  const int64_t heads = shard_output.size(1);
  const int64_t query_len = shard_output.size(2);
  if (rows == 0 || head_dim == 0) {
    return;
  }
  constexpr int threads = 256;
  constexpr int warps_per_block = 8;
  const int blocks = static_cast<int>((rows + warps_per_block - 1) / warps_per_block);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      shard_output.scalar_type(),
      "bdlm_merge_backward_cuda",
      [&] {
        merge_backward_kernel<scalar_t>
            <<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
                shard_output.data_ptr<scalar_t>(),
                shard_lse.data_ptr<float>(),
                final_output.data_ptr<scalar_t>(),
                final_lse.data_ptr<float>(),
                grad_output.data_ptr<scalar_t>(),
                shard_grad_output.data_ptr<scalar_t>(),
                grad_lse.data_ptr<float>(),
                rows,
                head_dim,
                heads,
                query_len,
                shard_output.stride(0),
                shard_output.stride(1),
                shard_output.stride(2),
                shard_output.stride(3),
                final_output.stride(0),
                final_output.stride(1),
                final_output.stride(2),
                final_output.stride(3),
                grad_output.stride(0),
                grad_output.stride(1),
                grad_output.stride(2),
                grad_output.stride(3),
                shard_grad_output.stride(0),
                shard_grad_output.stride(1),
                shard_grad_output.stride(2),
                shard_grad_output.stride(3));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void bdlm_finalize_bshd_cuda(
    torch::Tensor numerator,
    torch::Tensor m,
    torch::Tensor l,
    torch::Tensor output,
    torch::Tensor final_lse) {
  const int64_t batch = numerator.size(0);
  const int64_t heads = numerator.size(1);
  const int64_t query_len = numerator.size(2);
  const int64_t head_dim = numerator.size(3);
  const int64_t rows = batch * heads * query_len;
  if (rows == 0 || head_dim == 0) {
    return;
  }
  constexpr int threads = 256;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      output.scalar_type(),
      "bdlm_finalize_bshd_cuda",
      [&] {
        finalize_bshd_kernel<scalar_t>
            <<<std::min<int64_t>(rows, kMaxGridBlocks), threads, 0,
               c10::cuda::getCurrentCUDAStream()>>>(
                numerator.data_ptr<float>(),
                m.data_ptr<float>(),
                l.data_ptr<float>(),
                output.data_ptr<scalar_t>(),
                final_lse.data_ptr<float>(),
                batch,
                heads,
                query_len,
                head_dim);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
