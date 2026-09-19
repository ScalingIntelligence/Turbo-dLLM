import math

import pytest
import torch
import flash_attn_3._C  # noqa: F401
from flash_attn_interface import bdlm_flash_attn_func


def dense_bdlm(q, k, v, query_blocks, query_is_clean, block_size, key_start, scale):
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) * scale
    cols = torch.arange(k.shape[1], device=q.device)
    kv_blocks = (key_start + cols) // block_size
    allowed = torch.where(
        query_is_clean[:, None],
        query_blocks[:, None] >= kv_blocks[None, :],
        query_blocks[:, None] > kv_blocks[None, :],
    )
    scores = scores.masked_fill(~allowed[None, None, :, :], -float("inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhqk,bkhd->bqhd", probs, v.float())
    lse = torch.logsumexp(scores, dim=-1)
    return out, lse


def dense_bdlm_full(
    q,
    k,
    v,
    query_blocks,
    query_is_clean,
    block_size,
    key_start,
    clean_offset,
    scale,
):
    q_heads = q.shape[2]
    kv_heads = k.shape[2]
    assert q_heads % kv_heads == 0
    repeats = q_heads // kv_heads
    k_expanded = k.repeat_interleave(repeats, dim=2)
    v_expanded = v.repeat_interleave(repeats, dim=2)
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k_expanded.float()) * scale
    logical_key_start = -key_start - 1
    global_cols = logical_key_start + torch.arange(k.shape[1], device=q.device)
    kv_is_clean = global_cols >= clean_offset
    kv_positions = torch.where(kv_is_clean, global_cols - clean_offset, global_cols)
    kv_blocks = kv_positions // block_size
    allowed = torch.where(
        query_is_clean[:, None],
        kv_is_clean[None, :] & (query_blocks[:, None] >= kv_blocks[None, :]),
        (~kv_is_clean[None, :] & (query_blocks[:, None] == kv_blocks[None, :]))
        | (kv_is_clean[None, :] & (query_blocks[:, None] > kv_blocks[None, :])),
    )
    scores = scores.masked_fill(~allowed[None, None, :, :], -float("inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhqk,bkhd->bqhd", probs, v_expanded.float())
    lse = torch.logsumexp(scores, dim=-1)
    return out, lse


def run_case(batch, seqlen, heads, headdim, block_size, key_start):
    torch.manual_seed(batch * 1000 + seqlen + heads)
    q = torch.randn(batch, seqlen, heads, headdim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, seqlen, heads, headdim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, seqlen, heads, headdim, device="cuda", dtype=torch.bfloat16)
    query_blocks = ((torch.arange(seqlen, device="cuda", dtype=torch.int32) + key_start) // block_size).contiguous()
    query_is_clean = ((torch.arange(seqlen, device="cuda") % (2 * block_size)) < block_size).contiguous()
    scale = 1.0 / math.sqrt(headdim)

    out, lse, _, _ = torch.ops.flash_attn_3.fwd(
        q,
        k,
        v,
        softmax_scale=scale,
        num_splits=1,
        bdlm_query_blocks=query_blocks,
        bdlm_query_is_clean=query_is_clean,
        bdlm_block_size=block_size,
        bdlm_key_start=key_start,
    )
    dout = torch.randn_like(out)
    dlse = torch.randn_like(lse)
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    torch.ops.flash_attn_3.bwd(
        dout,
        q,
        k,
        v,
        out,
        lse,
        dq=dq,
        dk=dk,
        dv=dv,
        softmax_scale=scale,
        bdlm_query_blocks=query_blocks,
        bdlm_query_is_clean=query_is_clean,
        bdlm_block_size=block_size,
        bdlm_key_start=key_start,
        softmax_lse_grad=dlse,
    )

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    out_ref, lse_ref = dense_bdlm(
        q_ref,
        k_ref,
        v_ref,
        query_blocks,
        query_is_clean,
        block_size,
        key_start,
        scale,
    )
    (out_ref * dout.float()).sum().backward(retain_graph=True)
    q_grad_out = q_ref.grad.detach().clone()
    k_grad_out = k_ref.grad.detach().clone()
    v_grad_out = v_ref.grad.detach().clone()
    q_ref.grad.zero_()
    k_ref.grad.zero_()
    v_ref.grad.zero_()
    ((out_ref * dout.float()).sum() + (lse_ref * dlse).sum()).backward()
    torch.cuda.synchronize()

    assert (out.float() - out_ref).abs().max().item() < 1e-2
    assert (lse.float() - lse_ref).abs().max().item() < 1e-5
    assert (q_ref.grad.float() - q_grad_out.float()).abs().max().item() > 1e-4
    assert (k_ref.grad.float() - k_grad_out.float()).abs().max().item() > 1e-4
    assert v_ref.grad is not None
    assert (v_ref.grad.float() - v_grad_out.float()).abs().max().item() == 0.0
    assert (dq.float() - q_ref.grad.float()).abs().max().item() < 2e-2
    assert (dk.float() - k_ref.grad.float()).abs().max().item() < 2e-2
    assert (dv.float() - v_ref.grad.float()).abs().max().item() < 2e-2

    q_autograd = q.detach().clone().requires_grad_(True)
    k_autograd = k.detach().clone().requires_grad_(True)
    v_autograd = v.detach().clone().requires_grad_(True)
    out_autograd, lse_autograd = bdlm_flash_attn_func(
        q_autograd,
        k_autograd,
        v_autograd,
        query_blocks,
        query_is_clean,
        block_size,
        key_start,
        scale,
        return_softmax=True,
    )
    ((out_autograd * dout).sum() + (lse_autograd * dlse).sum()).backward()
    assert (q_autograd.grad.float() - q_ref.grad.float()).abs().max().item() < 2e-2
    assert (k_autograd.grad.float() - k_ref.grad.float()).abs().max().item() < 2e-2
    assert (v_autograd.grad.float() - v_ref.grad.float()).abs().max().item() < 2e-2


def test_bdlm_flash_attention_forward_backward():
    assert torch.cuda.get_device_capability()[0] >= 8
    run_case(1, 128, 2, 64, 32, 0)
    run_case(2, 96, 3, 64, 16, 8)
    run_case(1, 160, 1, 64, 32, 16)


def test_bdlm_full_mask_interior_high_key_tile_forward():
    major, _ = torch.cuda.get_device_capability()
    assert major >= 8
    torch.manual_seed(41)
    batch, query_len, key_len, heads, head_dim = 1, 192, 2048, 2, 64
    block_size = 32
    clean_offset = key_len // 2
    scale = 1.0 / math.sqrt(head_dim)

    query_blocks = torch.full(
        (query_len,),
        clean_offset // block_size - 1,
        device="cuda",
        dtype=torch.int32,
    )
    query_is_clean = torch.ones(query_len, device="cuda", dtype=torch.bool)
    query_blocks[0] = 0
    query_is_clean[0] = False

    # The forward kernels visit key tiles from high to low. A long key sequence
    # guarantees multiple fully masked interior tiles for row 0 on every
    # supported kernel shape, while row 0 still has valid keys at the start.
    global_cols = torch.arange(key_len, device="cuda")
    kv_is_clean = global_cols >= clean_offset
    kv_positions = torch.where(kv_is_clean, global_cols - clean_offset, global_cols)
    kv_blocks = kv_positions // block_size
    row0_allowed = (~kv_is_clean) & (kv_blocks == query_blocks[0])
    assert row0_allowed.any()
    assert not row0_allowed[block_size:].any()

    q = torch.randn(
        batch, query_len, heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    k = torch.randn(
        batch, key_len, heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    v = torch.randn_like(k)
    out, lse, _, _ = torch.ops.flash_attn_3.fwd(
        q,
        k,
        v,
        softmax_scale=scale,
        num_splits=1,
        bdlm_query_blocks=query_blocks,
        bdlm_query_is_clean=query_is_clean,
        bdlm_block_size=block_size,
        bdlm_key_start=-1,
        bdlm_clean_offset=clean_offset,
    )
    out_ref, lse_ref = dense_bdlm_full(
        q,
        k,
        v,
        query_blocks,
        query_is_clean,
        block_size,
        -1,
        clean_offset,
        scale,
    )
    torch.cuda.synchronize()

    assert torch.isfinite(out).all()
    assert torch.isfinite(lse).all()
    torch.testing.assert_close(out.float(), out_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse.float(), lse_ref, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize(("q_heads", "kv_heads"), ((4, 2), (4, 1)))
@pytest.mark.parametrize("head_dim", (64, 128, 256))
def test_bdlm_full_mask_matches_dense_gqa_forward_backward(
    q_heads: int,
    kv_heads: int,
    head_dim: int,
):
    assert torch.cuda.get_device_capability()[0] >= 8
    torch.manual_seed(29 + head_dim + kv_heads)
    batch, seq_len = 1, 192
    logical_len = 2 * seq_len
    block_size = 32
    scale = 1.0 / math.sqrt(head_dim)
    query_logical = torch.cat(
        (
            torch.arange(0, 96, device="cuda"),
            torch.arange(logical_len - 96, logical_len, device="cuda"),
        )
    )
    query_is_clean = (query_logical >= seq_len).contiguous()
    query_positions = torch.where(query_is_clean, query_logical - seq_len, query_logical)
    query_blocks = (query_positions // block_size).to(torch.int32).contiguous()
    q = torch.randn(
        batch,
        query_logical.numel(),
        q_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    k = torch.randn(
        batch,
        logical_len,
        kv_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    v = torch.randn_like(k, requires_grad=True)
    out, lse = bdlm_flash_attn_func(
        q,
        k,
        v,
        query_blocks,
        query_is_clean,
        block_size,
        -1,
        scale,
        return_softmax=True,
        clean_offset=seq_len,
    )
    dout = torch.randn_like(out)
    (out * dout).sum().backward()

    q_ref = q.detach().float().requires_grad_(True)
    k_ref = k.detach().float().requires_grad_(True)
    v_ref = v.detach().float().requires_grad_(True)
    out_ref, lse_ref = dense_bdlm_full(
        q_ref,
        k_ref,
        v_ref,
        query_blocks,
        query_is_clean,
        block_size,
        -1,
        seq_len,
        scale,
    )
    (out_ref * dout.float()).sum().backward()
    torch.cuda.synchronize()

    torch.testing.assert_close(out.float(), out_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse.float(), lse_ref, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(q.grad.float(), q_ref.grad, atol=4e-2, rtol=4e-2)
    torch.testing.assert_close(k.grad.float(), k_ref.grad, atol=4e-2, rtol=4e-2)
    torch.testing.assert_close(v.grad.float(), v_ref.grad, atol=4e-2, rtol=4e-2)


def test_bdlm_ring_backward_reuses_global_output_and_lse():
    """Owner-shard backward must equal one full block-masked backward."""

    assert torch.cuda.get_device_capability()[0] >= 9
    torch.manual_seed(17)
    batch, seq_len, query_heads, kv_heads, head_dim = 1, 512, 4, 2, 64
    logical_len = 2 * seq_len
    block_size = 64
    scale = 1.0 / math.sqrt(head_dim)
    query_logical = torch.cat(
        (
            torch.arange(seq_len // 2, seq_len, device="cuda"),
            torch.arange(seq_len, seq_len + seq_len // 2, device="cuda"),
        )
    )
    query_is_clean = (query_logical >= seq_len).contiguous()
    query_positions = torch.where(
        query_is_clean,
        query_logical - seq_len,
        query_logical,
    )
    query_blocks = (query_positions // block_size).to(torch.int32).contiguous()
    q = torch.randn(
        batch,
        query_logical.numel(),
        query_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k = torch.randn(
        batch, logical_len, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    v = torch.randn_like(k)
    out, lse, _, _ = torch.ops.flash_attn_3.fwd(
        q,
        k,
        v,
        softmax_scale=scale,
        num_splits=1,
        bdlm_query_blocks=query_blocks,
        bdlm_query_is_clean=query_is_clean,
        bdlm_block_size=block_size,
        bdlm_key_start=-1,
        bdlm_clean_offset=seq_len,
    )
    dout = torch.randn_like(out)

    dq_full = torch.empty_like(q)
    dk_full = torch.empty_like(k)
    dv_full = torch.empty_like(v)
    torch.ops.flash_attn_3.bwd(
        dout,
        q,
        k,
        v,
        out,
        lse,
        dq=dq_full,
        dk=dk_full,
        dv=dv_full,
        softmax_scale=scale,
        bdlm_query_blocks=query_blocks,
        bdlm_query_is_clean=query_is_clean,
        bdlm_block_size=block_size,
        bdlm_key_start=-1,
        bdlm_clean_offset=seq_len,
    )

    dq_sharded = torch.zeros_like(q, dtype=torch.float32)
    dk_sharded = torch.empty_like(k)
    dv_sharded = torch.empty_like(v)
    owner_shard_len = seq_len // 2
    for start in range(0, logical_len, owner_shard_len):
        stop = start + owner_shard_len
        dq_part = torch.empty_like(q)
        dk_part = torch.empty_like(k[:, start:stop])
        dv_part = torch.empty_like(v[:, start:stop])
        torch.ops.flash_attn_3.bwd(
            dout,
            q,
            k[:, start:stop],
            v[:, start:stop],
            out,
            lse,
            dq=dq_part,
            dk=dk_part,
            dv=dv_part,
            softmax_scale=scale,
            bdlm_query_blocks=query_blocks,
            bdlm_query_is_clean=query_is_clean,
            bdlm_block_size=block_size,
            bdlm_key_start=-start - 1,
            bdlm_clean_offset=seq_len,
        )
        dq_sharded.add_(dq_part.float())
        dk_sharded[:, start:stop].copy_(dk_part)
        dv_sharded[:, start:stop].copy_(dv_part)

    torch.cuda.synchronize()
    torch.testing.assert_close(dq_sharded, dq_full.float(), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(dk_sharded.float(), dk_full.float(), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(dv_sharded.float(), dv_full.float(), atol=3e-2, rtol=3e-2)


def test_partial_kv_backward_reuses_merged_output_and_lse():
    """Local noisy and clean-prefix DKV shards sum to the full backward."""

    assert torch.cuda.get_device_capability()[0] >= 8
    torch.manual_seed(53)
    batch, query_len, local_len, clean_len = 1, 64, 64, 128
    query_heads, kv_heads, head_dim = 4, 2, 128
    scale = 1.0 / math.sqrt(head_dim)
    q = torch.randn(
        batch,
        query_len,
        query_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    local_k = torch.randn(
        batch,
        local_len,
        kv_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    local_v = torch.randn_like(local_k)
    clean_k = torch.randn(
        batch,
        clean_len,
        kv_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    clean_v = torch.randn_like(clean_k)
    full_k = torch.cat((local_k, clean_k), dim=1)
    full_v = torch.cat((local_v, clean_v), dim=1)
    out, lse, _, _ = torch.ops.flash_attn_3.fwd(
        q,
        full_k,
        full_v,
        softmax_scale=scale,
        num_splits=1,
    )
    dout = torch.randn_like(out)

    dq_full = torch.empty_like(q)
    dk_full = torch.empty_like(full_k)
    dv_full = torch.empty_like(full_v)
    torch.ops.flash_attn_3.bwd(
        dout,
        q,
        full_k,
        full_v,
        out,
        lse,
        dq=dq_full,
        dk=dk_full,
        dv=dv_full,
        softmax_scale=scale,
    )

    partial_dq = torch.zeros_like(q, dtype=torch.float32)
    partial_dk = torch.empty_like(full_k)
    partial_dv = torch.empty_like(full_v)
    for start, stop, key, value in (
        (0, local_len, local_k, local_v),
        (local_len, local_len + clean_len, clean_k, clean_v),
    ):
        dq = torch.empty_like(q)
        dk = torch.empty_like(key)
        dv = torch.empty_like(value)
        torch.ops.flash_attn_3.bwd(
            dout,
            q,
            key,
            value,
            out,
            lse,
            dq=dq,
            dk=dk,
            dv=dv,
            softmax_scale=scale,
        )
        partial_dq.add_(dq.float())
        partial_dk[:, start:stop].copy_(dk)
        partial_dv[:, start:stop].copy_(dv)

    torch.cuda.synchronize()
    torch.testing.assert_close(partial_dq, dq_full.float(), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(partial_dk.float(), dk_full.float(), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(partial_dv.float(), dv_full.float(), atol=3e-2, rtol=3e-2)
