# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dllm_parallel.core.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    _rank_shard_filename,
    _validate_format,
    inspect_checkpoint,
    load_training_checkpoint,
    prune_checkpoints,
    save_training_checkpoint,
    validate_checkpoint_manifest,
)
import dllm_parallel.core.checkpoint as checkpointing
from dllm_parallel.training.checkpointing import (
    due_duration_checkpoint_fractions,
    resolve_load_checkpoint,
    should_save_checkpoint,
)
from dllm_parallel.training.run_spec import RunSpec


class _AdapterCheckpointModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = torch.nn.Linear(3, 4, bias=False)
        self.base.weight.requires_grad_(False)
        self.lora_a = torch.nn.Parameter(torch.randn(2, 3))
        self.lora_b = torch.nn.Parameter(torch.randn(4, 2))
        self.lora_a._dllm_lora_parameter = True
        self.lora_b._dllm_lora_parameter = True
        self.tied_lora_a = self.lora_a
        trainable = self.lora_a.numel() + self.lora_b.numel()
        self._dllm_adapter_metadata = {
            "rank": 2,
            "alpha": 4.0,
            "dropout": 0.0,
            "targets": ("attention",),
            "linear_modules": 1,
            "expert_modules": 0,
            "trainable_parameters": trainable,
            "total_parameters": trainable + self.base.weight.numel(),
        }

def _distributed_model_only_save_worker(
    rank: int,
    world_size: int,
    init_file: str,
    checkpoint_dir: str,
) -> None:
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(919)
        model = _AdapterCheckpointModel()
        save_training_checkpoint(
            checkpoint_dir,
            tag="duration_100pct_step_00000013",
            step=13,
            model=model,
            config={"model": {"id": "unit/base", "revision": "abc"}},
            backbone_state={
                "family": "diffusion_gemma",
                "adapter": model._dllm_adapter_metadata,
            },
            model_only=True,
        )
    finally:
        torch.distributed.destroy_process_group()

def test_model_only_checkpoint_excludes_frozen_and_optimizer_state(tmp_path) -> None:
    torch.manual_seed(101)
    model = _AdapterCheckpointModel()
    expected_a = model.lora_a.detach().clone()
    expected_b = model.lora_b.detach().clone()
    optimizer = torch.optim.AdamW((model.lora_a, model.lora_b), lr=1e-3)
    (model.lora_a.square().sum() + model.lora_b.square().sum()).backward()
    optimizer.step()

    result = save_training_checkpoint(
        tmp_path,
        tag="duration_025pct_step_00000007",
        step=7,
        model=model,
        optimizer=optimizer,
        objective_state={"should_not_persist": 1},
        dataloader_state={"should_not_persist": 2},
        scheduler_state={"should_not_persist": 3},
        config={"model": {"id": "unit/base", "revision": "abc"}},
        backbone_state={
            "family": "diffusion_gemma",
            "adapter": model._dllm_adapter_metadata,
        },
        model_only=True,
    )
    assert result.tag == "duration_025pct_step_00000007"
    manifest = validate_checkpoint_manifest(tmp_path)
    assert manifest["checkpoint_backend"] == "adapter_model_only"
    assert manifest["model_state_scope"] == "trainable_parameters"
    assert manifest["objective_state"] == {}
    assert manifest["dataloader_state"] == {}
    state = torch.load(
        tmp_path / result.tag / "mp_rank_00_model_states.pt",
        weights_only=True,
    )
    assert "optimizer" not in state
    assert "scheduler" not in state
    assert "rng_state" not in state
    assert "base.weight" not in state["module"]
    assert set(state["module"]) == {"lora_a", "lora_b", "tied_lora_a"}
    assert state["shared_params"] == {"tied_lora_a": "lora_a"}

    with torch.no_grad():
        model.lora_a.zero_()
        model.lora_b.zero_()
    loaded = load_training_checkpoint(tmp_path, model=model)
    assert loaded["step"] == 7
    # Expected values are captured before the optimizer update above.
    assert not torch.equal(model.lora_a, expected_a)
    assert not torch.equal(model.lora_b, expected_b)
    torch.testing.assert_close(model.lora_a, state["module"]["lora_a"])
    torch.testing.assert_close(model.lora_b, state["module"]["lora_b"])

def test_model_only_checkpoint_detects_content_corruption(tmp_path) -> None:
    model = _AdapterCheckpointModel()
    result = save_training_checkpoint(
        tmp_path,
        tag="duration_050pct_step_00000009",
        step=9,
        model=model,
        config={"model": {"id": "unit/base", "revision": "abc"}},
        backbone_state={
            "family": "diffusion_gemma",
            "adapter": model._dllm_adapter_metadata,
        },
        model_only=True,
    )
    path = tmp_path / result.tag / "mp_rank_00_model_states.pt"
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)
    with pytest.raises(RuntimeError, match="content hash mismatch"):
        load_training_checkpoint(tmp_path, model=model)

def test_model_only_checkpoint_uses_deepspeed_module_without_engine_state(
    tmp_path,
) -> None:
    model = _AdapterCheckpointModel()

    class Engine:
        module = model

        def save_checkpoint(self, *_args, **_kwargs):
            raise AssertionError("DeepSpeed training-state save must not run")

    save_training_checkpoint(
        tmp_path,
        tag="duration_075pct_step_00000011",
        step=11,
        deepspeed_engine=Engine(),
        config={"model": {"id": "unit/base", "revision": "abc"}},
        backbone_state={
            "family": "diffusion_gemma",
            "adapter": model._dllm_adapter_metadata,
        },
        model_only=True,
    )
    assert inspect_checkpoint(tmp_path)["checkpoint_backend"] == "adapter_model_only"

def test_model_only_checkpoint_distributed_rank_zero_publication(tmp_path) -> None:
    checkpoint_dir = tmp_path / "distributed-model-only"
    init_file = tmp_path / "gloo-init"
    torch.multiprocessing.spawn(
        _distributed_model_only_save_worker,
        args=(2, str(init_file), str(checkpoint_dir)),
        nprocs=2,
        join=True,
    )
    manifest = validate_checkpoint_manifest(checkpoint_dir)
    assert manifest["step"] == 13
    tag_dir = checkpoint_dir / manifest["latest_tag"]
    assert [path.name for path in tag_dir.glob("*model_states.pt")] == [
        "mp_rank_00_model_states.pt"
    ]

def test_rank_local_checkpoint_round_trip(tmp_path) -> None:
    torch.manual_seed(123)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    x = torch.randn(4, 3)
    model(x).sum().backward()
    optimizer.step()
    saved_weight = model.weight.detach().clone()
    saved_rng = torch.random.get_rng_state().clone()

    result = save_training_checkpoint(
        tmp_path,
        tag="step_00000001",
        step=1,
        model=model,
        optimizer=optimizer,
        objective_state={"noise_schedule": "loglinear"},
        dataloader_state={"cursor": 12, "token_count": 128},
        config={"model_id": "test"},
        kernel_metadata={"flash_attention": {"source_hash": "abc"}},
        profiler_metadata={"gate": "unit"},
    )

    assert result.tag == "step_00000001"
    assert (tmp_path / "metadata.json").exists()

    with torch.no_grad():
        model.weight.add_(10.0)
    torch.manual_seed(999)

    state = load_training_checkpoint(
        tmp_path,
        tag="latest",
        model=model,
        optimizer=optimizer,
    )

    assert state["step"] == 1
    assert state["objective_state"]["noise_schedule"] == "loglinear"
    manifest = inspect_checkpoint(tmp_path)
    assert manifest["format"] == CHECKPOINT_FORMAT_VERSION
    assert manifest["objective_state"]["noise_schedule"] == "loglinear"
    assert manifest["dataloader_state"]["cursor"] == 12
    assert manifest["kernel_metadata"]["flash_attention"]["source_hash"] == "abc"
    assert validate_checkpoint_manifest(tmp_path)["resolved_tag"] == "step_00000001"
    assert torch.allclose(model.weight, saved_weight)
    assert torch.equal(torch.random.get_rng_state(), saved_rng)

def test_rank_shard_filename_is_zero_padded() -> None:
    assert _rank_shard_filename(2, 4) == "rank_00002_of_00004.pt"

def test_replicated_deepspeed_checkpoint_publishes_zero_copy_rank_aliases(
    tmp_path,
) -> None:
    tag_dir = tmp_path / "step_00000001"
    tag_dir.mkdir()
    source = tag_dir / "mp_rank_00_model_states.pt"
    source.write_bytes(b"replicated model state")
    runtime = SimpleNamespace(
        plan=SimpleNamespace(
            tensor_parallel_size=1,
            expert_parallel_size=1,
            pipeline_parallel_size=1,
            model_parallel_size=4,
        )
    )

    created = checkpointing._ensure_replicated_deepspeed_model_aliases(
        tag_dir,
        runtime=runtime,
    )

    assert [item.name for item in created] == [
        "mp_rank_01_model_states.pt",
        "mp_rank_02_model_states.pt",
        "mp_rank_03_model_states.pt",
    ]
    assert all(source.samefile(item) for item in created)
    assert source.stat().st_nlink == 4
    assert (
        checkpointing._ensure_replicated_deepspeed_model_aliases(
            tag_dir,
            runtime=runtime,
        )
        == []
    )

def test_deepspeed_model_aliases_are_disabled_for_tensor_parallel_state(
    tmp_path,
) -> None:
    tag_dir = tmp_path / "step_00000001"
    tag_dir.mkdir()
    (tag_dir / "mp_rank_00_model_states.pt").write_bytes(b"shard zero")
    runtime = SimpleNamespace(
        plan=SimpleNamespace(
            tensor_parallel_size=2,
            expert_parallel_size=1,
            pipeline_parallel_size=1,
            model_parallel_size=2,
        )
    )

    assert (
        checkpointing._ensure_replicated_deepspeed_model_aliases(
            tag_dir,
            runtime=runtime,
        )
        == []
    )
    assert len(list(tag_dir.glob("*model_states.pt"))) == 1

def test_deepspeed_checkpoint_uses_supplied_control_barrier(
    tmp_path,
    monkeypatch,
) -> None:
    barriers = []
    control_group = object()

    class FakeEngine:
        def save_checkpoint(
            self,
            path,
            *,
            tag,
            client_state,
            exclude_frozen_parameters,
        ):
            del client_state
            assert exclude_frozen_parameters is False
            tag_dir = Path(path) / tag
            tag_dir.mkdir(parents=True)
            (tag_dir / "mp_rank_00_model_states.pt").write_bytes(b"state")

    runtime = SimpleNamespace(
        plan=SimpleNamespace(
            tensor_parallel_size=1,
            expert_parallel_size=1,
            pipeline_parallel_size=1,
            model_parallel_size=4,
        )
    )
    monkeypatch.setattr(checkpointing, "_rank", lambda: 0)
    monkeypatch.setattr(checkpointing, "_world_size", lambda: 4)
    monkeypatch.setattr(checkpointing, "_barrier", barriers.append)

    save_training_checkpoint(
        tmp_path,
        tag="step_00000001",
        step=1,
        deepspeed_engine=FakeEngine(),
        runtime=runtime,
        barrier_group=control_group,
    )

    assert barriers == [control_group, control_group]
    assert len(list((tmp_path / "step_00000001").glob("*model_states.pt"))) == 4

def test_validate_format_accepts_current_and_empty() -> None:
    _validate_format({})
    _validate_format({"format": CHECKPOINT_FORMAT_VERSION})

def test_validate_format_rejects_unknown_version() -> None:
    with pytest.raises(RuntimeError, match="format version"):
        _validate_format({"format": "dllm_parallel.distributed_checkpoint.v0"})

def test_async_checkpoint_save_round_trip(tmp_path) -> None:
    model = torch.nn.Linear(3, 2)
    result = save_training_checkpoint(
        tmp_path,
        tag="step_00000001",
        step=1,
        model=model,
        async_save=True,
    )

    assert result.async_save
    result.wait()
    assert (tmp_path / "step_00000001" / "rank_00000_of_00001.pt").exists()

    state = load_training_checkpoint(tmp_path, tag="latest", model=model)
    assert state["step"] == 1

def test_async_checkpoint_snapshots_live_model_state(tmp_path, monkeypatch) -> None:
    release = __import__("threading").Event()
    entered = __import__("threading").Event()
    real_save = checkpointing._atomic_torch_save

    def delayed_save(state, filename):
        entered.set()
        assert release.wait(timeout=5.0)
        real_save(state, filename)

    monkeypatch.setattr(checkpointing, "_atomic_torch_save", delayed_save)
    model = torch.nn.Linear(3, 2)
    saved_weight = model.weight.detach().clone()
    result = save_training_checkpoint(
        tmp_path,
        tag="step_00000001",
        step=1,
        model=model,
        async_save=True,
    )
    assert entered.wait(timeout=5.0)
    with torch.no_grad():
        model.weight.add_(100.0)
    assert not (tmp_path / "metadata.json").exists()
    release.set()
    result.wait()

    restored = torch.nn.Linear(3, 2)
    load_training_checkpoint(tmp_path, tag="latest", model=restored)
    torch.testing.assert_close(restored.weight, saved_weight)

def test_async_checkpoint_save_propagates_error(tmp_path, monkeypatch) -> None:
    def boom(state, filename):
        raise RuntimeError("simulated disk failure")

    monkeypatch.setattr(checkpointing, "_atomic_torch_save", boom)
    model = torch.nn.Linear(3, 2)
    result = save_training_checkpoint(
        tmp_path,
        tag="step_00000001",
        step=1,
        model=model,
        async_save=True,
    )

    with pytest.raises(RuntimeError, match="simulated disk failure"):
        result.wait()

def test_checkpoint_retention_prunes_old_step_dirs(tmp_path) -> None:
    model = torch.nn.Linear(3, 2)
    for step in range(1, 5):
        save_training_checkpoint(
            tmp_path,
            tag=f"step_{step:08d}",
            step=step,
            model=model,
            objective_state={"step": step},
            dataloader_state={"cursor": step},
            config={"unit": True},
            keep_last_n=2,
        )

    assert not (tmp_path / "step_00000001").exists()
    assert not (tmp_path / "step_00000002").exists()
    assert (tmp_path / "step_00000003").exists()
    assert (tmp_path / "step_00000004").exists()
    assert inspect_checkpoint(tmp_path)["resolved_tag"] == "step_00000004"

def test_prune_checkpoints_disabled_when_keep_last_zero(tmp_path) -> None:
    for step in range(1, 3):
        (tmp_path / f"step_{step:08d}").mkdir()

    assert prune_checkpoints(tmp_path, keep_last_n=0) == []
    assert (tmp_path / "step_00000001").exists()

def test_checkpoint_policy_selects_explicit_load_before_auto_resume(tmp_path) -> None:
    explicit = tmp_path / "explicit"
    auto = tmp_path / "auto"
    model = torch.nn.Linear(3, 2)
    save_training_checkpoint(
        auto,
        tag="step_00000003",
        step=3,
        model=model,
    )
    save_training_checkpoint(
        explicit,
        tag="step_00000002",
        step=2,
        model=model,
    )
    spec = RunSpec.from_mapping(
        {
            "checkpointing": {
                "save_checkpoint_dir": str(auto),
                "load_checkpoint_dir": str(explicit),
                "checkpoint_tag": "latest",
                "auto_resume": True,
            }
        }
    )

    path, tag, reason = resolve_load_checkpoint(spec)

    assert path == str(explicit)
    assert tag == "latest"
    assert reason == "explicit"

def test_checkpoint_policy_auto_resume_uses_save_dir_when_present(tmp_path) -> None:
    model = torch.nn.Linear(3, 2)
    save_training_checkpoint(
        tmp_path,
        tag="step_00000005",
        step=5,
        model=model,
    )
    spec = RunSpec.from_mapping(
        {
            "checkpointing": {
                "save_checkpoint_dir": str(tmp_path),
                "auto_resume": True,
            }
        }
    )

    path, tag, reason = resolve_load_checkpoint(spec)

    assert path == str(tmp_path)
    assert tag == "latest"
    assert reason == "auto_resume"

def test_checkpoint_policy_save_first_interval_and_final(tmp_path) -> None:
    spec = RunSpec.from_mapping(
        {
            "checkpointing": {
                "save_checkpoint_dir": str(tmp_path),
                "save_checkpoint_interval": 4,
                "save_first_step": True,
                "save_final": True,
            }
        }
    )

    assert should_save_checkpoint(spec=spec, global_step=1, target_total_steps=9)
    assert not should_save_checkpoint(spec=spec, global_step=2, target_total_steps=9)
    assert should_save_checkpoint(spec=spec, global_step=4, target_total_steps=9)
    assert should_save_checkpoint(spec=spec, global_step=9, target_total_steps=9)
    assert should_save_checkpoint(
        spec=spec,
        global_step=3,
        target_total_steps=9,
        training_complete=True,
    )

def test_duration_checkpoint_policy_emits_each_fraction_once(tmp_path) -> None:
    spec = RunSpec.from_mapping(
        {
            "training": {"max_duration_seconds": 100.0},
            "checkpointing": {
                "save_checkpoint_dir": str(tmp_path),
                "model_only": True,
                "save_duration_fractions": [0.25, 0.5, 0.75, 1.0],
            },
        }
    ).validate()
    saved: set[float] = set()
    assert (
        due_duration_checkpoint_fractions(
            spec=spec, elapsed_seconds=24.99, saved_fractions=saved
        )
        == ()
    )
    due = due_duration_checkpoint_fractions(
        spec=spec, elapsed_seconds=25.0, saved_fractions=saved
    )
    assert due == (0.25,)
    saved.update(due)
    assert due_duration_checkpoint_fractions(
        spec=spec, elapsed_seconds=74.99, saved_fractions=saved
    ) == (0.5,)
    saved.add(0.5)
    assert due_duration_checkpoint_fractions(
        spec=spec, elapsed_seconds=101.0, saved_fractions=saved
    ) == (0.75, 1.0)


@pytest.mark.parametrize(
    "checkpointing,match",
    [
        (
            {"save_duration_fractions": [0.25]},
            "requires save_checkpoint_dir",
        ),
        (
            {
                "save_checkpoint_dir": "/tmp/checkpoints",
                "save_duration_fractions": [0.5, 0.25],
            },
            "strictly increasing",
        ),
        (
            {
                "save_checkpoint_dir": "/tmp/checkpoints",
                "model_only": True,
                "async_checkpoint_save": True,
            },
            "does not support async",
        ),
    ],
)
def test_duration_model_only_checkpoint_policy_rejects_invalid_config(
    checkpointing, match
) -> None:
    with pytest.raises(ValueError, match=match):
        RunSpec.from_mapping(
            {
                "training": {"max_duration_seconds": 100.0},
                "checkpointing": checkpointing,
            }
        ).validate()
