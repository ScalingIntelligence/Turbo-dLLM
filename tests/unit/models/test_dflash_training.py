# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import torch

from dllm_parallel.core.models.backbones.dflash.training import (
    dflash_performance_counts,
)
from dllm_parallel.core.objectives.dflash import DFlashObjectiveBatch
from dllm_parallel.core.profiling.perf import DFLASH_WORK_COUNT_NAMES


def test_dflash_performance_counts_cover_global_objective_before_bp_ownership() -> None:
    objective = DFlashObjectiveBatch(
        anchor_positions=torch.tensor([[2, 7], [3, 8]]),
        anchor_valid=torch.tensor([[True, True], [True, False]]),
        block_positions=torch.zeros((2, 2, 4), dtype=torch.int64),
        teacher_source_positions=torch.zeros((2, 2, 4), dtype=torch.int64),
        target_token_positions=torch.zeros((2, 2, 4), dtype=torch.int64),
        draft_input_ids=torch.zeros((2, 8), dtype=torch.int64),
        supervised_mask=torch.tensor(
            [
                [True, True, True, True, True, True, False, False],
                [True, True, True, False, False, False, False, False],
            ]
        ),
        position_weights=torch.ones((2, 8)),
        document_starts=torch.tensor([[0, 1], [1, 16]], dtype=torch.int32),
        context_stops=torch.tensor([[2, 7], [3, 16]], dtype=torch.int32),
        global_anchor_count=2,
    )

    counts = dflash_performance_counts(
        objective,
        sequence_length=16,
        sliding_window=4,
    )

    assert dict(zip(DFLASH_WORK_COUNT_NAMES, counts.tolist())) == {
        "context_token_rows": 7,
        "valid_anchors": 3,
        "full_context_pairs": (2 + 6 + 2) * 4,
        "sliding_context_pairs": 16,
        "supervised_token_rows": 9,
    }
