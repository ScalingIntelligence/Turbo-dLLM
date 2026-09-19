from __future__ import annotations

import pytest

from dllm_parallel.training.block_diffusion_trainer import (
    _optimizer_step_loss_plan,
)


def test_token_mean_uses_one_shared_denominator_and_uniform_weights() -> None:
    batches = [object(), object(), object()]
    calls: list[list[object]] = []

    def denominator(items: list[object]) -> float:
        calls.append(items)
        return 37.0

    denominators, weights = _optimizer_step_loss_plan(
        batches,
        accumulation_loss_denominator=denominator,
        trajectory_loss_reduction="token_mean",
        trajectory_terminal_turn_weight=9.0,
    )

    assert calls == [batches]
    assert denominators == (37.0, 37.0, 37.0)
    assert weights == (1.0, 1.0, 1.0)


def test_turn_mean_normalizes_each_turn_and_upweights_terminal_turn() -> None:
    batches = [object(), object(), object()]
    token_counts = {id(batch): count for batch, count in zip(batches, (8, 80, 2))}

    def denominator(items: list[object]) -> float:
        assert len(items) == 1
        return float(token_counts[id(items[0])])

    denominators, weights = _optimizer_step_loss_plan(
        batches,
        accumulation_loss_denominator=denominator,
        trajectory_loss_reduction="turn_mean",
        trajectory_terminal_turn_weight=4.0,
    )

    assert denominators == (8.0, 80.0, 2.0)
    assert weights == (1.0, 1.0, 4.0)
    assert weights[-1] / sum(weights) == pytest.approx(2.0 / 3.0)


@pytest.mark.parametrize("weight", [0.0, -1.0, float("inf"), float("nan")])
def test_turn_mean_rejects_invalid_terminal_weight(weight: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        _optimizer_step_loss_plan(
            [object()],
            accumulation_loss_denominator=lambda _: 1.0,
            trajectory_loss_reduction="turn_mean",
            trajectory_terminal_turn_weight=weight,
        )
