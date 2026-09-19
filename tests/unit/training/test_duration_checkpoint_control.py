# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

from pathlib import Path

import torch

from dllm_parallel.training.block_diffusion_trainer import (
    _distributed_duration_checkpoint_control,
)
from dllm_parallel.training.run_spec import RunSpec


def _duration_control_worker(
    rank: int,
    world_size: int,
    init_file: str,
    result_dir: str,
) -> None:
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        spec = RunSpec.from_mapping(
            {
                "training": {"max_duration_seconds": 100.0},
                "checkpointing": {
                    "save_checkpoint_dir": "/tmp/unit-checkpoints",
                    "model_only": True,
                    "save_duration_fractions": [0.25, 0.5, 0.75, 1.0],
                },
            }
        ).validate()
        control = torch.zeros(5, dtype=torch.uint8)

        # Rank one's local clock has crossed 25%, but rank zero has not.  Both
        # ranks must follow rank zero and continue without checkpointing.
        first = _distributed_duration_checkpoint_control(
            spec=spec,
            elapsed_seconds=24.9 if rank == 0 else 25.1,
            saved_fractions=set(),
            rank=rank,
            distributed=True,
            control=control,
        )
        # Now only rank zero has crossed.  Both ranks must enter the same 25%
        # checkpoint branch despite the deliberately skewed follower clock.
        second = _distributed_duration_checkpoint_control(
            spec=spec,
            elapsed_seconds=25.1 if rank == 0 else 24.9,
            saved_fractions=set(),
            rank=rank,
            distributed=True,
            control=control,
        )
        # The stop decision is rank-zero authoritative too.
        third = _distributed_duration_checkpoint_control(
            spec=spec,
            elapsed_seconds=100.1 if rank == 0 else 99.9,
            saved_fractions={0.25, 0.5, 0.75},
            rank=rank,
            distributed=True,
            control=control,
        )
        torch.save(
            {"first": first, "second": second, "third": third},
            Path(result_dir) / f"rank-{rank}.pt",
        )
    finally:
        torch.distributed.destroy_process_group()


def test_duration_checkpoint_control_uses_rank_zero_clock(tmp_path) -> None:
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    torch.multiprocessing.spawn(
        _duration_control_worker,
        args=(2, str(tmp_path / "init"), str(result_dir)),
        nprocs=2,
        join=True,
    )
    results = [
        torch.load(result_dir / f"rank-{rank}.pt", weights_only=True)
        for rank in range(2)
    ]
    assert results[0] == results[1]
    assert results[0] == {
        "first": (False, ()),
        "second": (False, (0.25,)),
        "third": (True, (1.0,)),
    }
