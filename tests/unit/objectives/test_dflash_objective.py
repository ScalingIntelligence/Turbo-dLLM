from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import torch

from dllm_parallel.core.objectives.dflash import (
    DFlashObjectiveBatch,
    DFlashObjectiveRuntime,
    reference_dflash_loss,
)
from dllm_parallel.core.models.backbones.dflash.training import owned_anchor_indices


def _runtime(*, sample_from_anchor: bool = False, seed: int = 7):
    return DFlashObjectiveRuntime(
        block_size=4,
        max_anchors=3,
        mask_token_id=99,
        decay_gamma=4.0,
        sample_from_anchor=sample_from_anchor,
        device=torch.device("cpu"),
        seed=seed,
    )


def test_standard_dflash_alignment_and_weights() -> None:
    runtime = _runtime()
    input_ids = torch.arange(12).view(1, -1)
    loss_mask = torch.ones_like(input_ids, dtype=torch.bool)
    batch = runtime.prepare(input_ids, loss_mask)

    assert batch.anchor_valid.all()
    assert torch.equal(batch.context_stops, batch.anchor_positions)
    assert torch.equal(batch.target_token_positions, batch.block_positions)
    expected_sources = (batch.block_positions - 1).clamp_min(0)
    assert torch.equal(batch.teacher_source_positions, expected_sources)
    assert not batch.supervised_mask.view(1, 3, 4)[:, :, 0].any()
    weights = batch.position_weights.view(1, 3, 4)[0, 0]
    assert torch.equal(weights[:2], torch.tensor([0.0, 1.0]))
    assert torch.allclose(weights[2:], torch.exp(-torch.tensor([1.0, 2.0]) / 4.0))
    blocks = batch.draft_input_ids.view(1, 3, 4)
    assert torch.equal(blocks[:, :, 0], input_ids.gather(1, batch.anchor_positions))
    assert blocks[:, :, 1:].eq(99).all()


def test_sample_from_anchor_alignment() -> None:
    runtime = _runtime(sample_from_anchor=True)
    input_ids = torch.arange(16).view(1, -1)
    batch = runtime.prepare(input_ids, torch.ones_like(input_ids, dtype=torch.bool))

    assert torch.equal(batch.teacher_source_positions, batch.block_positions)
    assert torch.equal(
        batch.target_token_positions,
        (batch.block_positions + 1).clamp_max(input_ids.shape[1] - 1),
    )
    expected_supervision = (
        batch.anchor_positions.unsqueeze(-1)
        + torch.arange(runtime.block_size).view(1, 1, -1)
        + 1
    ).lt(input_ids.shape[1])
    assert torch.equal(
        batch.supervised_mask,
        expected_supervision.reshape_as(batch.supervised_mask),
    )
    weights = batch.position_weights.view(1, 3, 4)[0, 0]
    assert torch.allclose(weights, torch.exp(-torch.arange(4) / 4.0))


def test_anchor_sampling_matches_dflash_loss_mask_semantics() -> None:
    runtime = _runtime(seed=11)
    input_ids = torch.arange(16).view(1, -1)
    loss_mask = torch.ones_like(input_ids, dtype=torch.bool)
    document_ids = torch.tensor([[0] * 6 + [1] * 6 + [-1] * 4])
    batch = runtime.prepare(input_ids, loss_mask, document_ids)

    valid_anchors = batch.anchor_positions[batch.anchor_valid]
    assert (((0 <= valid_anchors) & (valid_anchors < 5)) | ((6 <= valid_anchors) & (valid_anchors < 11))).all()
    starts = batch.document_starts[batch.anchor_valid]
    assert torch.equal(starts, torch.where(valid_anchors < 6, 0, 6))


def test_invalid_anchor_intervals_remain_sorted_for_native_scheduling() -> None:
    runtime = DFlashObjectiveRuntime(
        block_size=4,
        max_anchors=6,
        mask_token_id=99,
        decay_gamma=4.0,
        sample_from_anchor=False,
        device=torch.device("cpu"),
        seed=13,
    )
    input_ids = torch.arange(12).view(1, -1)
    loss_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    loss_mask[:, :2] = True
    batch = runtime.prepare(input_ids, loss_mask)

    assert torch.all(batch.context_stops[:, 1:] >= batch.context_stops[:, :-1])
    assert torch.all(batch.document_starts[:, 1:] >= batch.document_starts[:, :-1])
    assert batch.context_stops[~batch.anchor_valid].eq(input_ids.shape[1]).all()
    assert batch.document_starts[~batch.anchor_valid].eq(input_ids.shape[1]).all()


def test_checkpoint_restores_anchor_rng_exactly() -> None:
    input_ids = torch.arange(24).view(1, -1)
    loss_mask = torch.ones_like(input_ids, dtype=torch.bool)
    original = _runtime(seed=23)
    original.prepare(input_ids, loss_mask)
    state = original.state_dict()
    expected = original.prepare(input_ids, loss_mask)

    resumed = _runtime(seed=999)
    resumed.load_state_dict(state)
    observed = resumed.prepare(input_ids, loss_mask)
    assert torch.equal(observed.anchor_positions, expected.anchor_positions)
    assert torch.equal(observed.anchor_valid, expected.anchor_valid)


def test_reference_forward_kl_matches_manual_reduction() -> None:
    draft = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]]])
    teacher = torch.tensor([[[0.5, 0.5], [1.0, -1.0], [0.0, 0.0]]])
    mask = torch.tensor([[False, True, True]])
    weights = torch.tensor([[0.0, 1.0, 0.5]])
    loss = reference_dflash_loss(
        draft,
        teacher_logits=teacher,
        supervised_mask=mask,
        position_weights=weights,
    )
    log_q = torch.log_softmax(draft.float(), dim=-1)
    p = torch.softmax(teacher.float(), dim=-1)
    token_loss = (p * (p.log() - log_q)).sum(dim=-1)
    expected = (token_loss * weights * mask).sum() / mask.sum()
    assert torch.allclose(loss, expected)


def test_reference_ce_uses_unweighted_valid_token_denominator() -> None:
    logits = torch.tensor([[[3.0, 0.0], [0.0, 3.0], [1.0, 1.0]]])
    targets = torch.tensor([[0, 1, 0]])
    mask = torch.tensor([[False, True, True]])
    weights = torch.tensor([[0.0, 1.0, 0.25]])
    loss = reference_dflash_loss(
        logits,
        target_token_ids=targets,
        supervised_mask=mask,
        position_weights=weights,
    )
    token_loss = torch.nn.functional.cross_entropy(
        logits.view(-1, 2), targets.view(-1), reduction="none"
    ).view_as(mask)
    expected = (token_loss * weights * mask).sum() / mask.sum()
    assert torch.allclose(loss, expected)


def test_reference_loss_normalizes_each_sample_independently() -> None:
    logits = torch.tensor(
        [
            [[2.0, 0.0], [0.0, 2.0], [1.0, 0.0]],
            [[0.0, 2.0], [2.0, 0.0], [0.0, 1.0]],
        ]
    )
    targets = torch.tensor([[0, 1, 0], [1, 0, 1]])
    mask = torch.tensor([[True, False, False], [True, True, True]])
    weights = torch.ones_like(mask, dtype=torch.float32)
    loss = reference_dflash_loss(
        logits,
        target_token_ids=targets,
        supervised_mask=mask,
        position_weights=weights,
    )
    token_loss = torch.nn.functional.cross_entropy(
        logits.view(-1, 2),
        targets.view(-1),
        reduction="none",
    ).view_as(mask)
    expected = (
        (token_loss * mask).sum(dim=1)
        / (mask.sum(dim=1, dtype=torch.float32) + 1.0e-5)
    ).mean()
    assert torch.allclose(loss, expected)


class _Runtime:
    active_block_mode = "dual_end"
    block_parallel_size = 4

    def __init__(self, rank: int) -> None:
        self.block_parallel_rank = rank


class _ContextRuntime:
    active_block_mode = "all_blocks"
    uses_context_parallel_attention = True
    block_parallel_size = 1
    context_attention_size = 4

    def __init__(self, rank: int) -> None:
        self.context_parallel_rank = rank


def _ownership_objective(costs: list[int]) -> DFlashObjectiveBatch:
    anchors = len(costs)
    return DFlashObjectiveBatch(
        anchor_positions=torch.arange(anchors).view(1, anchors),
        anchor_valid=torch.ones(1, anchors, dtype=torch.bool),
        block_positions=torch.zeros(1, anchors, 1, dtype=torch.int64),
        teacher_source_positions=torch.zeros(1, anchors, 1, dtype=torch.int64),
        target_token_positions=torch.zeros(1, anchors, 1, dtype=torch.int64),
        draft_input_ids=torch.zeros(1, anchors, dtype=torch.int64),
        supervised_mask=torch.ones(1, anchors, dtype=torch.bool),
        position_weights=torch.ones(1, anchors),
        document_starts=torch.zeros(1, anchors, dtype=torch.int32),
        context_stops=torch.tensor(costs, dtype=torch.int32).view(1, anchors),
    )


def test_dual_end_anchor_ownership_is_disjoint_and_complete() -> None:
    objective = _ownership_objective([0, 1, 4, 9, 16, 25, 36, 49, 64])
    owned = [
        owned_anchor_indices(
            objective,
            runtime=_Runtime(rank),
            device=torch.device("cpu"),
        )
        for rank in range(_Runtime.block_parallel_size)
    ]
    flattened = torch.cat(owned)
    assert torch.equal(flattened.sort().values, torch.arange(9))
    assert sum(indices.numel() for indices in owned) == 9
    for left in range(len(owned)):
        for right in range(left + 1, len(owned)):
            assert not torch.isin(owned[left], owned[right]).any()


def test_dual_end_assigns_complete_cost_pairs_to_each_rank() -> None:
    costs = list(range(16))
    objective = _ownership_objective(costs)
    owned = [
        owned_anchor_indices(
            objective,
            runtime=_Runtime(rank),
            device=torch.device("cpu"),
        )
        for rank in range(_Runtime.block_parallel_size)
    ]
    rank_costs = [sum(costs[index] for index in indices.tolist()) for indices in owned]
    assert rank_costs == [30, 30, 30, 30]


def test_non_bp_execution_retains_every_anchor() -> None:
    objective = _ownership_objective([0, 1, 2, 3, 4])
    indices = owned_anchor_indices(
        objective,
        runtime=None,
        device=torch.device("cpu"),
    )
    assert torch.equal(indices, torch.arange(5))


def test_anchor_ownership_drops_globally_invalid_padding() -> None:
    objective = replace(
        _ownership_objective([0, 1, 2, 3]),
        anchor_valid=torch.tensor([[True, True, False, False]]),
    )
    indices = owned_anchor_indices(
        objective,
        runtime=None,
        device=torch.device("cpu"),
    )
    assert torch.equal(indices, torch.tensor([0, 1]))


def test_pure_cp_replicates_every_anchor_on_every_rank() -> None:
    objective = _ownership_objective(list(range(11)))
    owned = [
        owned_anchor_indices(
            objective,
            runtime=_ContextRuntime(rank),
            device=torch.device("cpu"),
        )
        for rank in range(_ContextRuntime.context_attention_size)
    ]
    assert all(torch.equal(indices, torch.arange(11)) for indices in owned)


def test_sparse_dual_end_ownership_uses_compile_stable_anchor_buckets() -> None:
    anchor_count = 2048
    observed: set[int] = set()
    for valid_count in (97, 193, 385, 769, 1537, 2047):
        objective = replace(
            _ownership_objective(list(range(anchor_count))),
            anchor_valid=(torch.arange(anchor_count).view(1, -1) < valid_count),
        )
        owned = [
            owned_anchor_indices(
                objective,
                runtime=SimpleNamespace(
                    active_block_mode="dual_end",
                    block_parallel_size=2,
                    block_parallel_rank=rank,
                ),
                device=torch.device("cpu"),
            )
            for rank in range(2)
        ]

        assert owned[0].numel() == owned[1].numel()
        assert owned[0].numel() in {128, 256, 512, 1024}
        observed.add(owned[0].numel())
        flattened = torch.cat(owned)
        assert torch.unique(flattened).numel() == flattened.numel()
        assert torch.isin(torch.arange(valid_count), flattened).all()
        assert flattened.numel() - valid_count == flattened.ge(valid_count).sum()

    assert observed == {128, 256, 512, 1024}
