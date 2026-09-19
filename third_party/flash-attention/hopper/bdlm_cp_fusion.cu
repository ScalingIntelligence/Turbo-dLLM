/******************************************************************************
 * Copyright (c) 2026, The bdlm_parallel Authors.
 ******************************************************************************/

#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace {

__device__ __forceinline__ float finite_weight(float value, float next_m) {
    return isfinite(value) ? expf(value - next_m) : 0.0f;
}

__device__ __forceinline__ bool intervals_overlap(
    int lhs_begin,
    int lhs_end,
    int rhs_begin,
    int rhs_end) {
    return lhs_begin < lhs_end
        && rhs_begin < rhs_end
        && lhs_begin < rhs_end
        && rhs_begin < lhs_end;
}

__device__ __forceinline__ bool bdlm_row_has_valid_key(
    int query_block,
    bool query_is_clean,
    int block_size,
    int key_start,
    int clean_offset,
    int key_len) {
    const bool full_mask = key_start < 0;
    const int logical_begin = full_mask ? -key_start - 1 : key_start;
    const int logical_end = logical_begin + key_len;
    if (!full_mask) {
        const int prefix_end =
            (query_block + (query_is_clean ? 1 : 0)) * block_size;
        return logical_begin < prefix_end && logical_end > 0;
    }
    const int clean_prefix_end = clean_offset
        + (query_block + (query_is_clean ? 1 : 0)) * block_size;
    if (query_is_clean) {
        return intervals_overlap(
            logical_begin, logical_end, clean_offset, clean_prefix_end);
    }
    const int noisy_block_begin = query_block * block_size;
    const int noisy_block_end = noisy_block_begin + block_size;
    return intervals_overlap(
               logical_begin, logical_end, noisy_block_begin, noisy_block_end)
        || intervals_overlap(
               logical_begin, logical_end, clean_offset, clean_prefix_end);
}

template <typename scalar_t>
__device__ __forceinline__ float load_as_float(scalar_t value);

template <>
__device__ __forceinline__ float load_as_float<float>(float value) {
    return value;
}

template <>
__device__ __forceinline__ float load_as_float<__half>(__half value) {
    return __half2float(value);
}

template <>
__device__ __forceinline__ float load_as_float<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename scalar_t, bool Initial>
__global__ void bdlm_accum_full_from_fa_kernel(
    float* __restrict__ numerator,       // [B, H, Q, D]
    float* __restrict__ m,               // [B, H, Q]
    float* __restrict__ l,               // [B, H, Q]
    const scalar_t* __restrict__ output, // [B, Q, H, D]
    const float* __restrict__ lse,       // [B, H, Q]
    const int* __restrict__ query_blocks,
    const bool* __restrict__ query_is_clean,
    int64_t query_blocks_batch_stride,
    int64_t query_is_clean_batch_stride,
    int block_size,
    int key_start,
    int clean_offset,
    int key_len,
    int64_t batch,
    int64_t heads,
    int64_t query_len,
    int64_t head_dim) {
    const int64_t rows = batch * heads * query_len;
    for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
        const int64_t q = row % query_len;
        const int64_t h = (row / query_len) % heads;
        const int64_t b = row / query_len / heads;
        const int64_t query_blocks_offset = b * query_blocks_batch_stride + q;
        const int64_t query_is_clean_offset = b * query_is_clean_batch_stride + q;
        const bool row_has_valid_key = bdlm_row_has_valid_key(
            query_blocks[query_blocks_offset],
            query_is_clean[query_is_clean_offset],
            block_size,
            key_start,
            clean_offset,
            key_len);
        const float om = Initial ? -INFINITY : m[row];
        const float old_l = Initial ? 0.0f : l[row];
        const float nm = row_has_valid_key ? lse[row] : -INFINITY;
        __syncthreads();

        const float next_m = Initial ? nm : fmaxf(om, nm);
        const float safe_m = isfinite(next_m) ? next_m : 0.0f;
        const float old_w = Initial ? 0.0f : finite_weight(om, safe_m);
        const float new_w = Initial ? 1.0f : finite_weight(nm, safe_m);
        const bool finite_nm = isfinite(nm);
        for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            const int64_t acc_linear = row * head_dim + d;
            const int64_t out_linear = ((b * query_len + q) * heads + h) * head_dim + d;
            const float out_value = finite_nm ? load_as_float(output[out_linear]) : 0.0f;
            numerator[acc_linear] = Initial
                ? out_value
                : old_w * numerator[acc_linear] + new_w * out_value;
        }
        if (threadIdx.x == 0) {
            m[row] = next_m;
            l[row] = Initial
                ? (finite_nm ? 1.0f : 0.0f)
                : old_w * old_l + new_w * (finite_nm ? 1.0f : 0.0f);
        }
    }
}

template <typename scalar_t>
__global__ void bdlm_accum_compact_from_fa_kernel(
    float* __restrict__ numerator,       // [B, H, Q, D]
    float* __restrict__ m,               // [B, H, Q]
    float* __restrict__ l,               // [B, H, Q]
    const int64_t* __restrict__ query_indices,
    const scalar_t* __restrict__ output, // [B, N, H, D]
    const float* __restrict__ lse,       // [B, H, N]
    int64_t batch,
    int64_t heads,
    int64_t query_len,
    int64_t compact_len,
    int64_t head_dim) {
    const int64_t compact_rows = batch * heads * compact_len;
    for (int64_t compact_row = blockIdx.x; compact_row < compact_rows; compact_row += gridDim.x) {
        const int64_t cq = compact_row % compact_len;
        const int64_t h = (compact_row / compact_len) % heads;
        const int64_t b = compact_row / compact_len / heads;
        const int64_t q = query_indices[cq];
        const int64_t row = (b * heads + h) * query_len + q;
        const int64_t lse_row = (b * heads + h) * compact_len + cq;

        const float om = m[row];
        const float old_l = l[row];
        const float nm = lse[lse_row];
        __syncthreads();

        const float next_m = fmaxf(om, nm);
        const float safe_m = isfinite(next_m) ? next_m : 0.0f;
        const float old_w = finite_weight(om, safe_m);
        const float new_w = finite_weight(nm, safe_m);
        const bool finite_nm = isfinite(nm);
        for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            const int64_t acc_linear = row * head_dim + d;
            const int64_t out_linear = ((b * compact_len + cq) * heads + h) * head_dim + d;
            numerator[acc_linear] =
                old_w * numerator[acc_linear]
                + new_w * (finite_nm ? load_as_float(output[out_linear]) : 0.0f);
        }
        if (threadIdx.x == 0) {
            l[row] = old_w * old_l + new_w * (finite_nm ? 1.0f : 0.0f);
            m[row] = next_m;
        }
    }
}

int blocks_for_rows(int64_t rows) {
    const int64_t blocks = rows;
    return static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(blocks, 4096)));
}

} // namespace

void bdlm_accum_full_from_fa_cuda(
    void* numerator,
    void* m,
    void* l,
    void const* output,
    void const* lse,
    void const* query_blocks,
    void const* query_is_clean,
    int64_t query_blocks_batch_stride,
    int64_t query_is_clean_batch_stride,
    int block_size,
    int key_start,
    int clean_offset,
    int key_len,
    int output_dtype,
    int64_t batch,
    int64_t heads,
    int64_t query_len,
    int64_t head_dim,
    bool initial) {
    const int64_t rows = batch * heads * query_len;
    if (rows == 0 || head_dim == 0) {
        return;
    }
    constexpr int threads = 256;
    auto launch = [&](auto scalar_ptr) {
        using scalar_t = std::remove_pointer_t<decltype(scalar_ptr)>;
        if (initial) {
            bdlm_accum_full_from_fa_kernel<scalar_t, true>
                <<<blocks_for_rows(rows), threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
                    static_cast<float*>(numerator),
                    static_cast<float*>(m),
                    static_cast<float*>(l),
                    static_cast<scalar_t const*>(output),
                    static_cast<float const*>(lse),
                    static_cast<int const*>(query_blocks),
                    static_cast<bool const*>(query_is_clean),
                    query_blocks_batch_stride,
                    query_is_clean_batch_stride,
                    block_size,
                    key_start,
                    clean_offset,
                    key_len,
                    batch,
                    heads,
                    query_len,
                    head_dim);
        } else {
            bdlm_accum_full_from_fa_kernel<scalar_t, false>
                <<<blocks_for_rows(rows), threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
                    static_cast<float*>(numerator),
                    static_cast<float*>(m),
                    static_cast<float*>(l),
                    static_cast<scalar_t const*>(output),
                    static_cast<float const*>(lse),
                    static_cast<int const*>(query_blocks),
                    static_cast<bool const*>(query_is_clean),
                    query_blocks_batch_stride,
                    query_is_clean_batch_stride,
                    block_size,
                    key_start,
                    clean_offset,
                    key_len,
                    batch,
                    heads,
                    query_len,
                    head_dim);
        }
    };
    if (output_dtype == 0) {
        launch(static_cast<__half const*>(nullptr));
    } else if (output_dtype == 1) {
        launch(static_cast<__nv_bfloat16 const*>(nullptr));
    } else if (output_dtype == 2) {
        launch(static_cast<float const*>(nullptr));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void bdlm_accum_compact_from_fa_cuda(
    void* numerator,
    void* m,
    void* l,
    void const* query_indices,
    void const* output,
    void const* lse,
    int output_dtype,
    int64_t batch,
    int64_t heads,
    int64_t query_len,
    int64_t compact_len,
    int64_t head_dim) {
    const int64_t compact_rows = batch * heads * compact_len;
    if (compact_rows == 0 || head_dim == 0) {
        return;
    }
    constexpr int threads = 256;
    auto launch = [&](auto scalar_ptr) {
        using scalar_t = std::remove_pointer_t<decltype(scalar_ptr)>;
        bdlm_accum_compact_from_fa_kernel<scalar_t>
            <<<blocks_for_rows(compact_rows), threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
                static_cast<float*>(numerator),
                static_cast<float*>(m),
                static_cast<float*>(l),
                static_cast<int64_t const*>(query_indices),
                static_cast<scalar_t const*>(output),
                static_cast<float const*>(lse),
                batch,
                heads,
                query_len,
                compact_len,
                head_dim);
    };
    if (output_dtype == 0) {
        launch(static_cast<__half const*>(nullptr));
    } else if (output_dtype == 1) {
        launch(static_cast<__nv_bfloat16 const*>(nullptr));
    } else if (output_dtype == 2) {
        launch(static_cast<float const*>(nullptr));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
