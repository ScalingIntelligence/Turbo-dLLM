import pytest
import torch
from torch.nn.attention.flex_attention import BlockMask

from dllm_parallel.core.attention.context_parallel_attention import (
    _normalize_flex_block_mask_storage,
)


@pytest.mark.parametrize("single", [True, False])
def test_normalization_reconstructs_reverse_counts_from_forward_metadata(single):
    partial = torch.tensor(
        [[[[1]]]] if single else [[[[1, 1], [0, 1], [0, 0]]]], dtype=torch.bool
    )
    full = (
        torch.zeros_like(partial)
        if single
        else torch.tensor([[[[0, 0], [1, 0], [0, 1]]]], dtype=torch.bool)
    )
    rows, cols = partial.shape[-2:]

    def ordered(dense):
        return dense.sum(-1, dtype=torch.int32), dense.to(torch.int32).argsort(
            dim=-1, descending=True, stable=True
        ).to(torch.int32)

    pc, pi = ordered(partial)
    fc, fi = ordered(full)
    mask = BlockMask(
        seq_lengths=(64 if single else rows * 128, 64 if single else cols * 128),
        kv_num_blocks=pc,
        kv_indices=pi,
        full_kv_num_blocks=fc,
        full_kv_indices=fi,
        q_num_blocks=torch.full((1, 1, cols), 16843009, dtype=torch.int32),
        q_indices=torch.zeros((1, 1, cols, rows), dtype=torch.int32),
        full_q_num_blocks=torch.full((1, 1, cols), 513, dtype=torch.int32),
        full_q_indices=torch.zeros((1, 1, cols, rows), dtype=torch.int32),
        BLOCK_SIZE=(128, 128),
        mask_mod=lambda b, h, q, k: q >= k,
    )
    normalized = _normalize_flex_block_mask_storage(mask)
    torch.testing.assert_close(
        normalized.q_num_blocks, partial.sum(-2, dtype=torch.int32)
    )
    torch.testing.assert_close(
        normalized.full_q_num_blocks, full.sum(-2, dtype=torch.int32)
    )
    assert normalized.q_indices.shape[-1] == rows
    assert normalized.full_q_indices.shape[-1] == rows
    assert normalized.seq_lengths == mask.seq_lengths

    def dense_from_lists(counts, indices):
        occupied = torch.zeros((*indices.shape[:-1], rows), dtype=torch.bool)
        for column in range(cols):
            for position in range(int(counts[0, 0, column])):
                occupied[0, 0, column, int(indices[0, 0, column, position])] = True
        return occupied

    torch.testing.assert_close(
        dense_from_lists(normalized.q_num_blocks, normalized.q_indices),
        partial.transpose(-2, -1),
    )
    torch.testing.assert_close(
        dense_from_lists(normalized.full_q_num_blocks, normalized.full_q_indices),
        full.transpose(-2, -1),
    )
