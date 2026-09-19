/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cstdint>

#include <cute/tensor.hpp>

#include "cutlass/fast_math.h"  // For cutlass::FastDivmod

#include "utils.h"

namespace flash {

using namespace cute;

template <int kBlockM, int kBlockN, bool PackGQA, typename TiledMma,
          bool SwapAB=false, bool Is_dflash=false>
struct Mask {

    static_assert(!(PackGQA && SwapAB), "Cannot be both PackGQA and SwapAB");

    int const thread_idx;
    int const seqlen_q, seqlen_k;
    int const window_size_left, window_size_right, sink_token_length;
    cutlass::FastDivmod const attention_chunk_divmod;
    cutlass::FastDivmod const qhead_per_khead_divmod;
    int const* const bdlm_query_blocks;
    bool const* const bdlm_query_is_clean;
    int64_t const bdlm_query_blocks_batch_stride;
    int64_t const bdlm_query_is_clean_batch_stride;
    int const bidb;
    int const bdlm_block_size;
    int const bdlm_key_start;
    int const bdlm_clean_offset;
    int const* const dflash_context_starts;
    int const* const dflash_context_stops;
    bool const* const dflash_anchor_valid;
    int64_t const dflash_context_starts_batch_stride;
    int64_t const dflash_context_stops_batch_stride;
    int64_t const dflash_anchor_valid_batch_stride;
    int const dflash_block_size;
    int const dflash_key_start;

    CUTLASS_DEVICE
    Mask(const int thread_idx, const int seqlen_q, const int seqlen_k,
         const int window_size_left, const int window_size_right, const int sink_token_length,
         cutlass::FastDivmod const &attention_chunk_divmod,
         cutlass::FastDivmod const &qhead_per_khead_divmod,
         int const* const bdlm_query_blocks=nullptr,
         bool const* const bdlm_query_is_clean=nullptr,
         int64_t const bdlm_query_blocks_batch_stride=0,
         int64_t const bdlm_query_is_clean_batch_stride=0,
         int const bidb=0,
         int const bdlm_block_size=0,
         int const bdlm_key_start=0,
         int const bdlm_clean_offset=0,
         int const* const dflash_context_starts=nullptr,
         int const* const dflash_context_stops=nullptr,
         bool const* const dflash_anchor_valid=nullptr,
         int64_t const dflash_context_starts_batch_stride=0,
         int64_t const dflash_context_stops_batch_stride=0,
         int64_t const dflash_anchor_valid_batch_stride=0,
         int const dflash_block_size=0,
         int const dflash_key_start=0)
        : thread_idx(thread_idx)
        , seqlen_q(seqlen_q)
        , seqlen_k(seqlen_k)
        , window_size_left(window_size_left)
        , window_size_right(window_size_right)
        , sink_token_length(sink_token_length)
        , attention_chunk_divmod(attention_chunk_divmod)
        , qhead_per_khead_divmod(qhead_per_khead_divmod)
        , bdlm_query_blocks(bdlm_query_blocks)
        , bdlm_query_is_clean(bdlm_query_is_clean)
        , bdlm_query_blocks_batch_stride(bdlm_query_blocks_batch_stride)
        , bdlm_query_is_clean_batch_stride(bdlm_query_is_clean_batch_stride)
        , bidb(bidb)
        , bdlm_block_size(bdlm_block_size)
        , bdlm_key_start(bdlm_key_start)
        , bdlm_clean_offset(bdlm_clean_offset)
        , dflash_context_starts(dflash_context_starts)
        , dflash_context_stops(dflash_context_stops)
        , dflash_anchor_valid(dflash_anchor_valid)
        , dflash_context_starts_batch_stride(dflash_context_starts_batch_stride)
        , dflash_context_stops_batch_stride(dflash_context_stops_batch_stride)
        , dflash_anchor_valid_batch_stride(dflash_anchor_valid_batch_stride)
        , dflash_block_size(dflash_block_size)
        , dflash_key_start(dflash_key_start)
    {
    };

    template <bool Seqlenk_mask=false, bool Causal_mask=false, bool Local_mask=false,
        typename Engine, typename Layout>
    CUTLASS_DEVICE
    void apply(Tensor<Engine, Layout> &tSrS, const int m_block, const int n_block) const {
        static_assert(!(Causal_mask && Local_mask), "Cannot be both causal and local");
        static_assert(Layout::rank == 3, "Only support 3D Tensor");
        // Prefix-only BDLM shards are fully valid below their scheduler-pruned
        // boundary, but the full [noisy; clean] mask can contain non-prefix
        // structure inside any tile. Keep the fast interior-tile return for
        // prefix attention and evaluate every full-mask tile exactly.
        if constexpr (!Is_dflash) {
            if (!Seqlenk_mask && !Causal_mask && !Local_mask
                && !(bdlm_query_blocks != nullptr && bdlm_key_start < 0)) { return; }
        }

        auto thread_mma = TiledMma{}.get_thread_slice(thread_idx);
        auto thread0_mma = TiledMma{}.get_thread_slice(_0{});

        static constexpr int Row = !SwapAB ? 0 : 1, Col = !SwapAB ? 1 : 0;

        Tensor cS = cute::make_identity_tensor(Shape<Int<!SwapAB ? kBlockM : kBlockN>, Int<!SwapAB ? kBlockN : kBlockM>>{});
        Tensor tScS = thread_mma.partition_C(cS);
        Tensor tSrS_rowcol = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tSrS.layout()));
        Tensor tScS_rowcol = make_tensor(tScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tScS.layout()));
        Tensor t0ScS = thread0_mma.partition_C(cS);
        Tensor t0ScS_rowcol = make_tensor(t0ScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(t0ScS.layout()));
        if constexpr (Is_dflash) {
            int const* const starts = dflash_context_starts
                + bidb * dflash_context_starts_batch_stride;
            int const* const stops = dflash_context_stops
                + bidb * dflash_context_stops_batch_stride;
            bool const* const valid = dflash_anchor_valid
                + bidb * dflash_anchor_valid_batch_stride;
            int const thread_row_offset = get<Row>(tScS_rowcol(_0{}, _0{}));
            int const thread_col_offset = get<Col>(tScS_rowcol(_0{}, _0{}));
            static constexpr int kMmaThreadsPerRow = size<0, 0>(typename TiledMma::AtomLayoutC_TV{});
            int mma_m_idx = 0;
            if constexpr (PackGQA && !SwapAB) {
                mma_m_idx = qhead_per_khead_divmod.divide(
                    m_block * kBlockM + get<Row>(tScS_rowcol(thread_idx % kMmaThreadsPerRow, _0{})));
            }
            #pragma unroll
            for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                int const row_idx = [&] {
                    if constexpr (PackGQA && !SwapAB) {
                        return __shfl_sync(0xffffffff, mma_m_idx, m % kMmaThreadsPerRow, kMmaThreadsPerRow);
                    } else if constexpr (SwapAB) {
                        return m_block * kBlockM + thread_row_offset
                            + int(get<Row>(t0ScS_rowcol(m, _0{})));
                    } else {
                        return int(get<Row>(tScS_rowcol(m, _0{}))) + m_block * kBlockM;
                    }
                }();
                #pragma unroll
                for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                    int const col_idx = n_block * kBlockN + thread_col_offset
                        + int(get<Col>(t0ScS_rowcol(_0{}, n)));
                    bool masked = row_idx >= seqlen_q || col_idx >= seqlen_k;
                    if (!masked) {
                        int const anchor = row_idx / dflash_block_size;
                        int const global_col = dflash_key_start + col_idx;
                        masked = !valid[anchor]
                            || global_col < starts[anchor]
                            || global_col >= stops[anchor];
                    }
                    if (masked) { tSrS_rowcol(m, n) = -INFINITY; }
                }
            }
            return;
        }
        if (bdlm_query_blocks != nullptr) {
            int const* const q_blocks = bdlm_query_blocks + bidb * bdlm_query_blocks_batch_stride;
            bool const* const q_clean = bdlm_query_is_clean + bidb * bdlm_query_is_clean_batch_stride;
            bool const bdlm_full_mask = bdlm_key_start < 0;
            int const logical_key_start = bdlm_full_mask ? (-bdlm_key_start - 1) : bdlm_key_start;
            int const clean_offset = bdlm_full_mask
                ? bdlm_clean_offset
                : seqlen_q / 2;
            int const thread_row_offset = get<Row>(tScS_rowcol(_0{}, _0{}));
            int const thread_col_offset = get<Col>(tScS_rowcol(_0{}, _0{}));
            static constexpr int kMmaThreadsPerRow = size<0, 0>(typename TiledMma::AtomLayoutC_TV{});
            int mma_m_idx = 0;
            if constexpr (PackGQA && !SwapAB) {
                mma_m_idx = qhead_per_khead_divmod.divide(m_block * kBlockM + get<Row>(tScS_rowcol(thread_idx % kMmaThreadsPerRow, _0{})));
            }
            #pragma unroll
            for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                int const row_idx = [&] {
                    if constexpr (PackGQA && !SwapAB) {
                        return __shfl_sync(0xffffffff, mma_m_idx, m % kMmaThreadsPerRow, kMmaThreadsPerRow);
                    } else if constexpr (SwapAB) {
                        return m_block * kBlockM + thread_row_offset + int(get<Row>(t0ScS_rowcol(m, _0{})));
                    } else {
                        return int(get<Row>(tScS_rowcol(m, _0{}))) + m_block * kBlockM;
                    }
                }();
                #pragma unroll
                for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                    int const col_idx = n_block * kBlockN + thread_col_offset + int(get<Col>(t0ScS_rowcol(_0{}, n)));
                    bool masked = row_idx >= seqlen_q || col_idx >= seqlen_k;
                    if (!masked && bdlm_full_mask) {
                        int const global_col = logical_key_start + col_idx;
                        bool const kv_clean = global_col >= clean_offset;
                        int const kv_pos = kv_clean ? global_col - clean_offset : global_col;
                        int const kv_block = kv_pos / bdlm_block_size;
                        int const q_block = q_blocks[row_idx];
                        bool const query_clean = q_clean[row_idx];
                        bool const allowed = query_clean
                            ? (kv_clean && q_block >= kv_block)
                            : ((!kv_clean && q_block == kv_block) || (kv_clean && q_block > kv_block));
                        masked = !allowed;
                    } else if (!masked) {
                        int const key_stop = (
                            q_blocks[row_idx] + (q_clean[row_idx] ? 1 : 0)
                        ) * bdlm_block_size - logical_key_start;
                        masked = col_idx >= key_stop;
                    }
                    if (masked) {
                        tSrS_rowcol(m, n) = -INFINITY;
                    }
                }
            }
            return;
        }
        // We want to use the col indices of thread0 to compare, since that is known at compile time.
        // So we subtract the limit by the first col index of this thread (get<Col>(tScS_rowcol(_0{}, _0{})))
        int const thread_col_offset = get<Col>(tScS_rowcol(_0{}, _0{}));
        int const seqlenk_col_limit = seqlen_k - n_block * kBlockN - thread_col_offset;
        if constexpr (!Causal_mask && !Local_mask) {
            if constexpr (Seqlenk_mask) {  // Just masking based on col
                #pragma unroll
                for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                    if (int(get<Col>(t0ScS_rowcol(_0{}, n))) >= seqlenk_col_limit) {
                        #pragma unroll
                        for (int m = 0; m < size<0>(tSrS_rowcol); ++m) { tSrS_rowcol(m, n) = -INFINITY; }
                    }
                }
            }
        } else {  // mask based on both row and col
            if constexpr (!SwapAB) {
                // If PackGQA, we split the work of compute divmod among threads in the same row
                static constexpr int kMmaThreadsPerRow = size<0, 0>(typename TiledMma::AtomLayoutC_TV{});
                static_assert(cutlass::NumThreadsPerWarp % kMmaThreadsPerRow == 0);
                static_assert(!PackGQA || CUTE_STATIC_V(size<0>(tSrS_rowcol)) <= kMmaThreadsPerRow);
                int mma_m_idx;
                // Might get OOB but it's ok since we'll check it later
                if constexpr (PackGQA) {
                    mma_m_idx = qhead_per_khead_divmod.divide(m_block * kBlockM + get<Row>(tScS_rowcol(thread_idx % kMmaThreadsPerRow, _0{})));
                }
                int const causal_row_offset = 1 + seqlen_k - n_block * kBlockN - seqlen_q - thread_col_offset;
                if constexpr (Causal_mask) {
                    #pragma unroll
                    for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                        int const row_idx = !PackGQA
                            ? get<Row>(tScS_rowcol(m, _0{})) + m_block * kBlockM
                            :  __shfl_sync(0xffffffff, mma_m_idx, m % kMmaThreadsPerRow, kMmaThreadsPerRow);
                        int const col_limit_right = !Seqlenk_mask
                            ? row_idx + causal_row_offset
                            : __viaddmin_s32(row_idx, causal_row_offset, seqlenk_col_limit);
                        #pragma unroll
                        for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                            if (int(get<Col>(t0ScS_rowcol(_0{}, n))) >= col_limit_right) { tSrS_rowcol(m, n) = -INFINITY; }
                        }
                    }
                } else {
                    int const local_row_offset_right = causal_row_offset + window_size_right;
                    int const local_row_offset_left = causal_row_offset - 1 - window_size_left;
                    int const col_limit_sink = sink_token_length - n_block * kBlockN;  // TODO: subtract thread_col_offset?
                    #pragma unroll
                    for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                        int const row_idx = !PackGQA
                            ? get<Row>(tScS_rowcol(m, _0{})) + m_block * kBlockM
                            :  __shfl_sync(0xffffffff, mma_m_idx, m % kMmaThreadsPerRow, kMmaThreadsPerRow);
                        int col_limit_right = !Seqlenk_mask
                            ? row_idx + local_row_offset_right
                            : __viaddmin_s32(row_idx, local_row_offset_right, seqlenk_col_limit);
                        int col_limit_left = row_idx + local_row_offset_left;
                        if (attention_chunk_divmod.divisor > 0) {
                            int col_limit_left_chunk = flash::round_down(attention_chunk_divmod, row_idx + seqlen_k - seqlen_q) - n_block * kBlockN - thread_col_offset;
                            col_limit_left = std::max(col_limit_left, col_limit_left_chunk);
                            col_limit_right = std::min(col_limit_right, col_limit_left_chunk + attention_chunk_divmod.divisor);
                        }
                        #pragma unroll
                        for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                            int const col_idx = int(get<Col>(t0ScS_rowcol(m, n)));
                            if (col_idx >= col_limit_right || (col_idx < col_limit_left && col_idx >= col_limit_sink)) { tSrS_rowcol(m, n) = -INFINITY; }
                        }
                    }
                }
            } else {
                // TODO: backward does not support attention_chunk yet
                int const thread_row_offset = get<Row>(tScS_rowcol(_0{}, _0{}));
                int const causal_row_offset = seqlenk_col_limit - seqlen_q + m_block * kBlockM + thread_row_offset;
                if constexpr (Causal_mask) {
                    #pragma unroll
                    for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                        int const col0 = int(get<Col>(t0ScS_rowcol(_0{}, n)));
                        // If col0 is beyond the column limit, we want to mask out the entire column, by setting
                        // row limit to be kBlockM.
                        int const row_limit_top = col0 >= seqlenk_col_limit ? kBlockM : col0 - causal_row_offset;
                        #pragma unroll
                        for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                            if (int(get<Row>(t0ScS_rowcol(m, _0{}))) < row_limit_top) { tSrS_rowcol(m, n) = -INFINITY; }
                        }
                    }
                } else {
                    int const col_limit_sink = sink_token_length - n_block * kBlockN - thread_col_offset;
                    #pragma unroll
                    for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                        int const col0 = int(get<Col>(t0ScS_rowcol(_0{}, n)));
                        // If col0 is beyond the column limit, we want to mask out the entire column, by setting
                        // row limit to be kBlockM.
                        int const row_limit_top = col0 >= seqlenk_col_limit ? kBlockM : col0 - causal_row_offset - window_size_right;
                        int const row_limit_bot = col0 < col_limit_sink ? kBlockM : col0 - causal_row_offset + window_size_left;
                        #pragma unroll
                        for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                            int const row_idx = int(get<Row>(t0ScS_rowcol(m, _0{})));
                            if (row_idx < row_limit_top || row_idx > row_limit_bot) { tSrS_rowcol(m, n) = -INFINITY; }
                        }
                    }
                }
            }
        }
    };

};

} // namespace flash
