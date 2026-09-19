from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from dllm_parallel.core.data import DataBatch
from dllm_parallel.core.models.backbones.nemotron.model import (
    NemotronLabsDiffusionPackedBlockDiffusionModel,
)
from dllm_parallel.core.objectives.fast_dllm_v2 import FastDLLMv2ObjectiveRuntime
from dllm_parallel.core.objectives.runtime import (
    LABEL_IGNORE_INDEX,
    TokenAccountingPolicy,
)
from dllm_parallel.core.objectives.training import FastDLLMv2TrainingTask


def _accounting() -> TokenAccountingPolicy:
    return TokenAccountingPolicy(
        global_batch_size=1,
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        data_parallel_size=1,
        context_parallel_size=1,
        block_parallel_size=1,
        tensor_parallel_size=1,
        sequence_parallel=False,
    )


def _runtime(seed: int = 7) -> FastDLLMv2ObjectiveRuntime:
    return FastDLLMv2ObjectiveRuntime(
        mask_token_id=31,
        block_size=4,
        seq_len=8,
        device=torch.device("cpu"),
        seed=seed,
        token_accounting=_accounting(),
    )


def test_complementary_views_partition_every_shifted_target_once() -> None:
    tokens = torch.arange(8, dtype=torch.long).unsqueeze(0)
    batch = _runtime().corrupt(tokens)

    assert batch.clean_input_ids.shape == (2, 8)
    assert batch.noisy_input_ids.shape == (2, 8)
    assert batch.labels.shape == (2, 8)
    torch.testing.assert_close(batch.clean_input_ids[0], tokens[0])
    torch.testing.assert_close(batch.clean_input_ids[1], tokens[0])

    for target_position in range(1, 8):
        query_position = target_position - 1
        supervised_views = torch.nonzero(
            batch.labels[:, query_position] != LABEL_IGNORE_INDEX,
            as_tuple=False,
        ).flatten()
        assert supervised_views.numel() == 1
        view = int(supervised_views.item())
        assert int(batch.labels[view, query_position]) == int(
            tokens[0, target_position]
        )
        assert int(batch.noisy_input_ids[view, target_position]) == 31
        assert int(batch.noisy_input_ids[1 - view, target_position]) == int(
            tokens[0, target_position]
        )

    assert torch.all(batch.labels[:, -1] == LABEL_IGNORE_INDEX)
    assert int(batch.active_tokens) == 7
    assert batch.valid_tokens == 7
    assert batch.loss_denominator == 7.0


def test_complementary_views_respect_sft_supervision_mask() -> None:
    tokens = torch.arange(8, dtype=torch.long).unsqueeze(0)
    supervision = torch.tensor(
        [[False, False, False, True, True, False, True, True]],
        dtype=torch.bool,
    )
    batch = _runtime().corrupt(
        tokens,
        supervision_mask=supervision,
        supervision_count=int(supervision.sum()),
    )

    assert int(batch.active_tokens) == 4
    assert batch.valid_tokens == 4
    for target_position in range(1, 8):
        count = int((batch.labels[:, target_position - 1] != LABEL_IGNORE_INDEX).sum())
        assert count == int(supervision[0, target_position])
        if not supervision[0, target_position]:
            torch.testing.assert_close(
                batch.noisy_input_ids[:, target_position],
                tokens[:, target_position].expand(2),
            )


def test_objective_state_restores_complementary_rng() -> None:
    tokens = torch.arange(8, dtype=torch.long).unsqueeze(0)
    runtime = _runtime(seed=11)
    runtime.corrupt(tokens)
    state = runtime.state_dict()
    expected = runtime.corrupt(tokens)

    restored = _runtime(seed=99)
    restored.load_state_dict(state)
    actual = restored.corrupt(tokens)

    torch.testing.assert_close(actual.noisy_input_ids, expected.noisy_input_ids)
    torch.testing.assert_close(actual.labels, expected.labels)


def test_fast_task_normalizes_shifted_sft_targets() -> None:
    task = FastDLLMv2TrainingTask(
        mask_token_id=31,
        block_size=4,
        seq_len=8,
        vocab_size=32,
        device=torch.device("cpu"),
        seed=7,
        runtime=SimpleNamespace(data_parallel_size=1, expert_parallel_size=1),
        token_accounting=_accounting(),
        noise_schedule_epsilon=1.0e-3,
        bp_loss_scale=None,
    )
    loss_mask = torch.tensor(
        [[True, True, False, True, False, True, False, True]],
        dtype=torch.bool,
    )
    batch = DataBatch(
        input_ids=torch.arange(8, dtype=torch.long).unsqueeze(0),
        loss_mask=loss_mask,
        supervised_token_count=int(loss_mask.sum()),
    )

    assert task.accumulation_loss_denominator([batch]) == 4.0
    assert task.prepare(batch).corrupted.loss_denominator == 4.0


def test_fast_loss_keeps_mask_token_in_softmax_denominator() -> None:
    torch.manual_seed(3)
    head = nn.Linear(4, 8, bias=False)
    owner = SimpleNamespace(
        output_head=head,
        final_logit_softcap=None,
        block_size=2,
        runtime=SimpleNamespace(),
    )
    hidden = torch.randn(1, 2, 4)
    labels = torch.tensor([[1, 2]], dtype=torch.long)
    positions = torch.tensor([0, 1], dtype=torch.long)
    weights = torch.ones(1, 1)

    actual = (
        NemotronLabsDiffusionPackedBlockDiffusionModel.distributed_block_diffusion_loss(
            owner,
            hidden,
            labels,
            positions,
            weights,
            vocab_size=8,
            mask_token_id=7,
            seq_len=2,
            valid_token_count=2.0,
            bp_loss_scale=1.0,
            exclude_mask_token=False,
        )
    )
    expected = F.cross_entropy(head(hidden).reshape(-1, 8), labels.reshape(-1))

    torch.testing.assert_close(actual, expected)
