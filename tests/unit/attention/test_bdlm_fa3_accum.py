import math

import pytest
import torch


def _dense_full(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_blocks: torch.Tensor,
    query_is_clean: torch.Tensor,
    *,
    block_size: int,
    clean_offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    repeats = query.shape[2] // key.shape[2]
    expanded_key = key.repeat_interleave(repeats, dim=2)
    expanded_value = value.repeat_interleave(repeats, dim=2)
    scale = 1.0 / math.sqrt(query.shape[-1])
    scores = torch.einsum("bqhd,bkhd->bhqk", query.float(), expanded_key.float()) * scale
    columns = torch.arange(key.shape[1], device=query.device)
    key_is_clean = columns >= clean_offset
    key_positions = torch.where(key_is_clean, columns - clean_offset, columns)
    key_blocks = key_positions // block_size
    allowed = torch.where(
        query_is_clean[:, None],
        key_is_clean[None, :] & (query_blocks[:, None] >= key_blocks[None, :]),
        ((~key_is_clean[None, :]) & (query_blocks[:, None] == key_blocks[None, :]))
        | (key_is_clean[None, :] & (query_blocks[:, None] > key_blocks[None, :])),
    )
    scores.masked_fill_(~allowed[None, None], -torch.inf)
    probabilities = torch.softmax(scores, dim=-1)
    return (
        torch.einsum("bhqk,bkhd->bqhd", probabilities, expanded_value.float()),
        torch.logsumexp(scores, dim=-1),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_bdlm_gqa_full_and_sharded_accum_match_dense() -> None:
    import flash_attn_3._C  # noqa: F401

    torch.manual_seed(710)
    device = torch.device("cuda")
    batch, seq_len, query_heads, kv_heads, head_dim = 2, 64, 4, 2, 64
    block_size = 16
    logical_len = 2 * seq_len
    query_positions = torch.cat(
        (
            torch.arange(0, 32, device=device),
            torch.arange(96, 128, device=device),
        )
    )
    query_is_clean = (query_positions >= seq_len).contiguous()
    model_positions = torch.where(query_is_clean, query_positions - seq_len, query_positions)
    query_blocks = (model_positions // block_size).to(torch.int32).contiguous()
    query = torch.randn(
        batch,
        query_positions.numel(),
        query_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    key = torch.randn(
        batch,
        logical_len,
        kv_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    value = torch.randn_like(key)
    scale = 1.0 / math.sqrt(head_dim)
    reference, reference_lse = _dense_full(
        query,
        key,
        value,
        query_blocks,
        query_is_clean,
        block_size=block_size,
        clean_offset=seq_len,
    )

    full_output, full_lse, _, _ = torch.ops.flash_attn_3.fwd(
        query,
        key,
        value,
        softmax_scale=scale,
        num_splits=1,
        bdlm_query_blocks=query_blocks,
        bdlm_query_is_clean=query_is_clean,
        bdlm_block_size=block_size,
        bdlm_key_start=-1,
        bdlm_clean_offset=seq_len,
    )
    torch.testing.assert_close(full_output.float(), reference, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(full_lse, reference_lse, atol=2e-3, rtol=2e-3)

    numerator = torch.empty(
        batch,
        query_heads,
        query.shape[1],
        head_dim,
        device=device,
        dtype=torch.float32,
    )
    maximum = torch.empty(
        batch,
        query_heads,
        query.shape[1],
        device=device,
        dtype=torch.float32,
    )
    denominator = torch.empty_like(maximum)
    intervals = ((0, 32), (96, 128), (32, 64), (64, 96))
    for index, (start, stop) in enumerate(intervals):
        torch.ops.flash_attn_3.bdlm_fwd_accum(
            query,
            key[:, start:stop],
            value[:, start:stop],
            numerator,
            maximum,
            denominator,
            None,
            query_blocks,
            query_is_clean,
            block_size,
            -start - 1,
            seq_len,
            scale,
            index == 0,
        )
    output = (numerator / denominator.unsqueeze(-1)).permute(0, 2, 1, 3)
    lse = maximum + torch.log(denominator)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, reference, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse, reference_lse, atol=2e-3, rtol=2e-3)
