from __future__ import annotations

import pytest
import torch

from dllm_parallel.core.checkpoint import (
    load_training_checkpoint,
    save_training_checkpoint,
)


pytestmark = pytest.mark.checkpoint_resume


def test_checkpoint_resume_matches_uninterrupted_training(tmp_path) -> None:
    def build() -> tuple[torch.nn.Module, torch.optim.Optimizer]:
        torch.manual_seed(5)
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        return model, optimizer

    def step(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: torch.Tensor,
    ) -> None:
        optimizer.zero_grad(set_to_none=True)
        model(batch).square().mean().backward()
        optimizer.step()

    batches = (
        torch.tensor([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]]),
        torch.tensor([[3.0, 2.0, 1.0], [2.5, 1.5, 0.5]]),
    )
    uninterrupted_model, uninterrupted_optimizer = build()
    for batch in batches:
        step(uninterrupted_model, uninterrupted_optimizer, batch)

    resumed_model, resumed_optimizer = build()
    step(resumed_model, resumed_optimizer, batches[0])
    save_training_checkpoint(
        tmp_path,
        tag="step_00000001",
        step=1,
        model=resumed_model,
        optimizer=resumed_optimizer,
        objective_state={"batches_corrupted": 1},
        dataloader_state={"cursor": 1, "token_count": 2},
        config={"integration_test": True},
    )

    reloaded_model, reloaded_optimizer = build()
    load_training_checkpoint(
        tmp_path,
        tag="latest",
        model=reloaded_model,
        optimizer=reloaded_optimizer,
    )
    step(reloaded_model, reloaded_optimizer, batches[1])

    for expected, observed in zip(
        uninterrupted_model.parameters(),
        reloaded_model.parameters(),
    ):
        torch.testing.assert_close(observed, expected)
