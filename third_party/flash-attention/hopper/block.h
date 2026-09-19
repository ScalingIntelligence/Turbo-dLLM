/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cstdint>

namespace flash {

template <class SeqlenInfo_t, int kBlockM, int kBlockN, bool Is_causal, bool Is_local,
          bool PackGQA=false, bool Split=false, bool Is_dflash=false>
struct BlockMN {

    static
    CUTLASS_DEVICE
    cute::tuple<int, int> get_n_block_min_max(
            SeqlenInfo_t const& seqlen_info,
            int const m_block, int const bidb, int const split_idx, int const num_splits,
            int const window_size_left, int const window_size_right,
            cutlass::FastDivmod const& attention_chunk_divmod,
            cutlass::FastDivmod const& qhead_per_khead_divmod,
            int const* const bdlm_query_blocks = nullptr,
            bool const* const bdlm_query_is_clean = nullptr,
            int64_t const bdlm_query_blocks_batch_stride = 0,
            int64_t const bdlm_query_is_clean_batch_stride = 0,
            int const bdlm_block_size = 0,
            int const bdlm_key_start = 0,
            int const bdlm_clean_offset = 0,
            int const* const dflash_context_starts = nullptr,
            int const* const dflash_context_stops = nullptr,
            bool const* const dflash_anchor_valid = nullptr,
            int64_t const dflash_context_starts_batch_stride = 0,
            int64_t const dflash_context_stops_batch_stride = 0,
            int64_t const dflash_anchor_valid_batch_stride = 0,
            int const dflash_block_size = 0,
            int const dflash_key_start = 0) {

        int const seqlen_k = seqlen_info.seqlen_k;
        int const seqlen_q = seqlen_info.seqlen_q;
        int n_block_max = cute::ceil_div(seqlen_k, kBlockN);
        int bdlm_n_block_min = 0;
        if constexpr (Is_dflash) {
            int const packed_row_begin = m_block * kBlockM;
            int const packed_row_end = (m_block + 1) * kBlockM;
            int const row_begin = !PackGQA
                ? packed_row_begin
                : qhead_per_khead_divmod.divide(packed_row_begin);
            int const row_end = std::min(
                !PackGQA
                    ? packed_row_end
                    : qhead_per_khead_divmod.divide(packed_row_end - 1) + 1,
                seqlen_q);
            int const* const starts = dflash_context_starts
                + bidb * dflash_context_starts_batch_stride;
            int const* const stops = dflash_context_stops
                + bidb * dflash_context_stops_batch_stride;
            bool const* const valid = dflash_anchor_valid
                + bidb * dflash_anchor_valid_batch_stride;
            int min_key_begin = dflash_key_start + seqlen_k;
            int max_key_stop = dflash_key_start;
            CUTLASS_PRAGMA_UNROLL
            for (int row = row_begin; row < row_end; ++row) {
                int const anchor = row / dflash_block_size;
                if (!valid[anchor]) { continue; }
                min_key_begin = std::min(min_key_begin, starts[anchor]);
                max_key_stop = std::max(max_key_stop, stops[anchor]);
            }
            int const relative_begin = std::max(min_key_begin - dflash_key_start, 0);
            int const relative_stop = std::min(max_key_stop - dflash_key_start, seqlen_k);
            if (relative_stop <= relative_begin) {
                n_block_max = 0;
            } else {
                bdlm_n_block_min = relative_begin / kBlockN;
                n_block_max = std::min(n_block_max, cute::ceil_div(relative_stop, kBlockN));
            }
        } else if (bdlm_query_blocks != nullptr) {
            int const packed_row_begin = m_block * kBlockM;
            int const packed_row_end = (m_block + 1) * kBlockM;
            int const row_begin = !PackGQA
                ? packed_row_begin
                : qhead_per_khead_divmod.divide(packed_row_begin);
            int const row_end = std::min(
                !PackGQA
                    ? packed_row_end
                    : qhead_per_khead_divmod.divide(packed_row_end - 1) + 1,
                seqlen_q);
            int const* const q_blocks = bdlm_query_blocks + bidb * bdlm_query_blocks_batch_stride;
            bool const* const q_clean = bdlm_query_is_clean + bidb * bdlm_query_is_clean_batch_stride;
            bool const bdlm_full_mask = bdlm_key_start < 0;
            int const logical_key_start = bdlm_full_mask ? (-bdlm_key_start - 1) : bdlm_key_start;
            int min_key_begin = logical_key_start + seqlen_k;
            int max_key_stop = logical_key_start;
            int const full_clean_start = bdlm_full_mask
                ? bdlm_clean_offset
                : seqlen_q / 2;
            bool const noisy_only_shard = bdlm_full_mask
                && logical_key_start + seqlen_k <= full_clean_start;
            bool const clean_only_shard = bdlm_full_mask
                && logical_key_start >= full_clean_start;
            CUTLASS_PRAGMA_UNROLL
            for (int row = row_begin; row < row_end; ++row) {
                int const query_clean_offset = q_clean[row] ? 1 : 0;
                int key_begin = logical_key_start;
                int key_stop = logical_key_start;
                if (noisy_only_shard) {
                    if (q_clean[row]) { continue; }
                    key_begin = q_blocks[row] * bdlm_block_size;
                    key_stop = key_begin + bdlm_block_size;
                } else if (clean_only_shard) {
                    key_begin = full_clean_start;
                    key_stop = full_clean_start
                        + (q_blocks[row] + query_clean_offset) * bdlm_block_size;
                } else if (bdlm_full_mask) {
                    key_begin = 0;
                    key_stop = full_clean_start
                        + (q_blocks[row] + query_clean_offset) * bdlm_block_size;
                } else {
                    key_begin = 0;
                    key_stop = (q_blocks[row] + query_clean_offset) * bdlm_block_size;
                }
                min_key_begin = std::min(min_key_begin, key_begin);
                max_key_stop = std::max(max_key_stop, key_stop);
            }
            int const relative_begin = std::max(min_key_begin - logical_key_start, 0);
            int const relative_stop = std::min(max_key_stop - logical_key_start, seqlen_k);
            if (relative_stop <= relative_begin) {
                n_block_max = 0;
            } else {
                bdlm_n_block_min = relative_begin / kBlockN;
                n_block_max = std::min(
                    n_block_max,
                    cute::ceil_div(relative_stop, kBlockN));
            }
        }
        if constexpr (Is_causal || Is_local) {
            int m_idx_max = (m_block + 1) * kBlockM;
            // TODO: check off-by-1 error
            if (PackGQA) { m_idx_max = qhead_per_khead_divmod.divide(m_idx_max - 1) + 1 ; }
            int const n_idx = m_idx_max + seqlen_info.seqlen_k - seqlen_info.seqlen_q;
            int n_idx_right = !Is_local ? n_idx : n_idx + window_size_right;
            if (Is_local && attention_chunk_divmod.divisor > 0) {
                n_idx_right = std::min(n_idx_right, flash::round_up(attention_chunk_divmod, n_idx));
            }
            n_block_max = std::min(n_block_max, cute::ceil_div(n_idx_right, kBlockN));
        }
        int n_block_min = bdlm_n_block_min;
        if constexpr (Is_local) {
            int m_idx_min = m_block * kBlockM;
            if (PackGQA) { m_idx_min = qhead_per_khead_divmod.divide(m_idx_min); }
            int const n_idx = m_idx_min + seqlen_k - seqlen_q;
            int n_idx_left = n_idx - window_size_left;
            if (attention_chunk_divmod.divisor > 0) {
                n_idx_left = std::max(n_idx_left, flash::round_down(attention_chunk_divmod, n_idx));
            }
            n_block_min = std::max(int(0), n_idx_left / kBlockN);
        }
        // if (threadIdx.x == 128) { printf("Inside, bid.x = %d, bid.y = %d, bid.z = %d, split_idx = %d, n_block_min: %d, n_block_max: %d\n", blockIdx.x, blockIdx.y, blockIdx.z, split_idx, n_block_min, n_block_max); }
        if constexpr (Split) {
            uint32_t num_splits_dynamic_u = reinterpret_cast<uint32_t const&>(split_idx) >> 16; // first 16 bits are for num_splits
            int num_splits_dynamic = reinterpret_cast<int&>(num_splits_dynamic_u);
            int split_idx_actual = split_idx & 0x0000FFFF;
            int num_splits_actual = num_splits_dynamic > 0 ? num_splits_dynamic : num_splits;
            int num_n_blocks_per_split = n_block_max <= n_block_min ? 0 : cute::ceil_div(n_block_max - n_block_min, num_splits_actual);
            n_block_min = n_block_min + split_idx_actual * num_n_blocks_per_split;
            n_block_max = std::min(n_block_min + num_n_blocks_per_split, n_block_max);
            // if (threadIdx.x == 128) { printf("Inside, bid.x = %d, bid.y = %d, bid.z = %d, split_idx = %d, num_splits_dynamic = %d, num_splits_actual = %d, num_n_blocks_per_split = %d, n_block_min: %d, n_block_max: %d\n", blockIdx.x, blockIdx.y, blockIdx.z, split_idx, num_splits_dynamic, num_splits_actual, num_n_blocks_per_split, n_block_min, n_block_max); }
        }
        // if (threadIdx.x == 128) { printf("After split, inside, bid.y = %d, bid.z = %d, split_idx = %d, n_block_min: %d, n_block_max: %d\n", blockIdx.y, blockIdx.z, split_idx, n_block_min, n_block_max); }
        return {n_block_min, n_block_max};
    }

    static
    CUTLASS_DEVICE
    cute::tuple<int, int> get_n_block_k_new_min_max(
            SeqlenInfo_t const& seqlen_info,
            int const m_block, int const bidb, int const split_idx, int const num_splits,
            int const window_size_left, int const window_size_right,
            cutlass::FastDivmod const& attention_chunk_divmod,
            cutlass::FastDivmod const& qhead_per_khead_divmod) {

        auto [n_block_min, n_block_max] = get_n_block_min_max(
            seqlen_info, m_block, bidb, split_idx, num_splits,
            window_size_left, window_size_right, attention_chunk_divmod, qhead_per_khead_divmod);
        int const idx_k_new_min = std::max(n_block_min * kBlockN - seqlen_info.seqlen_k_og, 0);
        int const idx_k_new_max = std::min(n_block_max * kBlockN - seqlen_info.seqlen_k_og, seqlen_info.seqlen_k_new);
        int const n_block_new_min = idx_k_new_min / kBlockN;
        int const n_block_new_max = idx_k_new_max > idx_k_new_min ? cute::ceil_div(idx_k_new_max, kBlockN) : n_block_new_min;
        // if (threadIdx.x == 128 && m_block == 0) { printf("bidb = %d, seqlen_k_new = %d, seqlen_k_og = %d, n_block_min = %d, n_block_max = %d, idx_k_new_min = %d, idx_k_new_max = %d, n_block_new_min = %d, n_block_new_max = %d\n", bidb, seqlen_k_new, seqlen_k_og, n_block_min, n_block_max, idx_k_new_min, idx_k_new_max, n_block_new_min, n_block_new_max);}
        return {n_block_new_min, n_block_new_max};
    }

    static
    CUTLASS_DEVICE
    cute::tuple<int, int> get_m_block_min_max(
            SeqlenInfo_t const& seqlen_info,
            int const n_block, int const bidb,
            int const window_size_left, int const window_size_right, int const sink_token_length,
            int const* const bdlm_query_blocks = nullptr,
            bool const* const bdlm_query_is_clean = nullptr,
            int64_t const bdlm_query_blocks_batch_stride = 0,
            int64_t const bdlm_query_is_clean_batch_stride = 0,
            int const bdlm_block_size = 0,
            int const bdlm_key_start = 0,
            int const bdlm_clean_offset = 0,
            int const* const dflash_context_starts = nullptr,
            int const* const dflash_context_stops = nullptr,
            bool const* const dflash_anchor_valid = nullptr,
            int64_t const dflash_context_starts_batch_stride = 0,
            int64_t const dflash_context_stops_batch_stride = 0,
            int64_t const dflash_anchor_valid_batch_stride = 0,
            int const dflash_block_size = 0,
            int const dflash_key_start = 0) {
        // TODO: support attention_chunk
        int const seqlen_q = seqlen_info.seqlen_q;
        int const seqlen_k = seqlen_info.seqlen_k;
        int m_block_max = cute::ceil_div(seqlen_q, kBlockM);
        int m_block_min = 0;
        if constexpr (Is_dflash) {
            int const tile_key_start = dflash_key_start + n_block * kBlockN;
            int const tile_key_stop = std::min(
                tile_key_start + kBlockN,
                dflash_key_start + seqlen_k);
            int const* const starts = dflash_context_starts
                + bidb * dflash_context_starts_batch_stride;
            int const* const stops = dflash_context_stops
                + bidb * dflash_context_stops_batch_stride;
            int const num_anchors = cute::ceil_div(seqlen_q, dflash_block_size);
            int lo = 0;
            int hi = num_anchors;
            while (lo < hi) {
                int const mid = lo + (hi - lo) / 2;
                if (stops[mid] <= tile_key_start) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            int const first = lo;
            lo = first;
            hi = num_anchors;
            while (lo < hi) {
                int const mid = lo + (hi - lo) / 2;
                if (starts[mid] < tile_key_stop) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            int const last_exclusive = lo;
            if (last_exclusive <= first) {
                m_block_max = 0;
            } else {
                m_block_min = (first * dflash_block_size) / kBlockM;
                m_block_max = std::min(
                    m_block_max,
                    cute::ceil_div(last_exclusive * dflash_block_size, kBlockM));
            }
        } else if (bdlm_query_blocks != nullptr) {
            bool const bdlm_full_mask = bdlm_key_start < 0;
            int const logical_key_start = bdlm_full_mask ? (-bdlm_key_start - 1) : bdlm_key_start;
            int const tile_key_start = logical_key_start + n_block * kBlockN;
            int const tile_key_stop = std::min(
                tile_key_start + kBlockN,
                logical_key_start + seqlen_k);
            int const* const q_blocks = bdlm_query_blocks + bidb * bdlm_query_blocks_batch_stride;
            bool const* const q_clean = bdlm_query_is_clean + bidb * bdlm_query_is_clean_batch_stride;
            if (bdlm_full_mask) {
                bool const noisy_tile = tile_key_stop <= bdlm_clean_offset;
                bool const clean_tile = tile_key_start >= bdlm_clean_offset;
                if (noisy_tile || clean_tile) {
                    int clean_begin = 0;
                    int clean_end = seqlen_q;
                    while (clean_begin < clean_end) {
                        int const midpoint = clean_begin + (clean_end - clean_begin) / 2;
                        if (q_clean[midpoint]) {
                            clean_end = midpoint;
                        } else {
                            clean_begin = midpoint + 1;
                        }
                    }
                    auto lower_bound_block = [&](int begin, int end, int target) {
                        while (begin < end) {
                            int const midpoint = begin + (end - begin) / 2;
                            if (q_blocks[midpoint] < target) {
                                begin = midpoint + 1;
                            } else {
                                end = midpoint;
                            }
                        }
                        return begin;
                    };
                    int first_valid_row = seqlen_q;
                    int last_valid_row = 0;
                    if (noisy_tile) {
                        // GQA backward uses an ordered dQ semaphore whose
                        // predecessor is the previous local K tile. A noisy
                        // query attends an interior same-block K interval, so
                        // pruning earlier K tiles would leave a missing
                        // predecessor and deadlock. Keep noisy-owner traversal
                        // dense here; the element mask still makes all
                        // disallowed contributions exactly zero. Clean-prefix
                        // tiles below remain safely prunable because their
                        // valid K tiles form a prefix.
                        first_valid_row = 0;
                        last_valid_row = seqlen_q;
                    } else {
                        int const first_key_block =
                            (tile_key_start - bdlm_clean_offset) / bdlm_block_size;
                        int const noisy_begin = lower_bound_block(
                            0, clean_begin, first_key_block + 1);
                        int const clean_row_begin = lower_bound_block(
                            clean_begin, seqlen_q, first_key_block);
                        if (noisy_begin < clean_begin) {
                            first_valid_row = noisy_begin;
                            last_valid_row = clean_begin;
                        }
                        if (clean_row_begin < seqlen_q) {
                            first_valid_row = std::min(first_valid_row, clean_row_begin);
                            last_valid_row = seqlen_q;
                        }
                    }
                    if (last_valid_row == 0) {
                        m_block_min = 0;
                        m_block_max = 0;
                    } else {
                        m_block_min = first_valid_row / kBlockM;
                        m_block_max = cute::ceil_div(last_valid_row, kBlockM);
                    }
                } else {
                    // Full-mask CP splits noisy and clean intervals before launch.
                    // Keep mixed tiles conservative for direct full-sequence callers.
                    m_block_min = 0;
                }
            } else {
                CUTLASS_PRAGMA_NO_UNROLL
                for (; m_block_min < m_block_max; ++m_block_min) {
                    int max_key_stop = 0;
                    int const row_begin = m_block_min * kBlockM;
                    int const row_end = std::min(row_begin + kBlockM, seqlen_q);
                    CUTLASS_PRAGMA_UNROLL
                    for (int row = row_begin; row < row_end; ++row) {
                        int const query_clean_offset = q_clean[row] ? 1 : 0;
                        int const key_stop =
                            (q_blocks[row] + query_clean_offset) * bdlm_block_size;
                        max_key_stop = std::max(max_key_stop, key_stop);
                    }
                    if (max_key_stop > tile_key_start) { break; }
                }
            }
        }
        if constexpr (Is_local) {
            if (n_block >= cute::ceil_div(sink_token_length, kBlockN)) {
                m_block_max = std::min(m_block_max, cute::ceil_div((n_block + 1) * kBlockN + seqlen_q - seqlen_k + window_size_left, kBlockM));
            }
        }
        if constexpr (Is_causal || Is_local) {
            m_block_min = std::max(m_block_min, (n_block * kBlockN + seqlen_q - seqlen_k - window_size_right) / kBlockM);
        }
        return {m_block_min, m_block_max};
    }

    // If we have separate iterations with causal or local masking at the start, where do we stop
    static
    CUTLASS_DEVICE
    int get_n_block_min_causal_local_mask(
            SeqlenInfo_t const& seqlen_info,
            int const m_block, int const n_block_min, int const window_size_right,
            cutlass::FastDivmod const& attention_chunk_divmod,
            cutlass::FastDivmod const& qhead_per_khead_divmod) {
        int const m_idx_min = !PackGQA ? m_block * kBlockM : qhead_per_khead_divmod.divide(m_block * kBlockM);
        int const n_idx = m_idx_min + seqlen_info.seqlen_k - seqlen_info.seqlen_q;
        int n_idx_right = !Is_local ? n_idx : n_idx + window_size_right;
        if (Is_local && attention_chunk_divmod.divisor > 0) {
            n_idx_right = std::min(n_idx_right, flash::round_up(attention_chunk_divmod, n_idx));
        }
        return std::max(n_block_min, n_idx_right / kBlockN);
    }

    // If we have separate iterations with local masking at the end, where do we stop the non-masked iterations
    static
    CUTLASS_DEVICE
    int get_n_block_min_before_local_mask(
            SeqlenInfo_t const& seqlen_info,
            int const m_block, int const n_block_min, int const window_size_left,
            cutlass::FastDivmod const& attention_chunk_divmod,
            cutlass::FastDivmod const& qhead_per_khead_divmod) {
        int const m_idx_max = !PackGQA ? (m_block + 1) * kBlockM : qhead_per_khead_divmod.divide((m_block + 1) * kBlockM - 1) + 1;
        int const n_idx = m_idx_max + seqlen_info.seqlen_k - seqlen_info.seqlen_q;
        int n_idx_left = !Is_local ? n_idx : n_idx - window_size_left;
        if (Is_local && attention_chunk_divmod.divisor > 0) {
            n_idx_left = std::max(n_idx_left, flash::round_down(attention_chunk_divmod, n_idx));
        }
        return !Is_local ? n_block_min : std::max(n_block_min, cute::ceil_div(n_idx_left, kBlockN));
    }

};

} // namespace flash
