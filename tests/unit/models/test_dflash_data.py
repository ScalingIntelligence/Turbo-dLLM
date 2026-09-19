from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from dllm_parallel.core.models.backbones.dflash.data import (
    DFlashFeatureDataRuntime,
    FEATURE_FORMAT,
)


def _write_features(tmp_path, *, sequence_length: int = 8):
    path = tmp_path / "features.pt"
    tensors = {
        "input_ids": torch.arange(2 * sequence_length).view(2, sequence_length),
        "loss_mask": torch.ones(2, sequence_length, dtype=torch.bool),
        "target_hidden_states": torch.arange(
            2 * sequence_length * 6,
            dtype=torch.float32,
        ).view(2, sequence_length, 6),
        "verifier_last_hidden_states": torch.arange(
            2 * sequence_length * 3,
            dtype=torch.float32,
        ).view(2, sequence_length, 3),
    }
    torch.save(tensors, path)
    manifest = {
        "format": FEATURE_FORMAT,
        "verifier_id": "unit/verifier",
        "verifier_revision": "revision",
        "target_layer_ids": [1, 3],
        "sequence_length": sequence_length,
        "sample_count": 2,
        "target_feature_width": 6,
        "verifier_hidden_size": 3,
    }
    (tmp_path / "features.pt.manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return path, tensors


def test_dflash_feature_manifest_and_cp_local_transfer(tmp_path) -> None:
    path, tensors = _write_features(tmp_path)
    runtime = SimpleNamespace(
        uses_context_parallel_attention=True,
        context_attention_size=2,
        context_parallel_rank=1,
        block_parallel_size=1,
    )
    data = DFlashFeatureDataRuntime.from_path(
        path=str(path),
        batch_size=1,
        seq_len=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
        require_teacher_features=True,
        verifier_id="unit/verifier",
        verifier_revision="revision",
        target_layer_ids=(1, 3),
        runtime=runtime,
        data_parallel_rank=0,
        data_parallel_size=1,
    )

    batch = data.next_batch()

    assert batch.target_hidden_states.shape == (1, 4, 6)
    assert torch.equal(
        batch.target_hidden_states,
        tensors["target_hidden_states"][:1, 4:],
    )
    assert torch.equal(batch.target_position_ids, torch.arange(4, 8).view(1, -1))
    assert batch.verifier_last_hidden_states.device.type == "cpu"
    assert batch.verifier_last_hidden_states.shape == (1, 8, 3)
    gathered = batch.verifier_last_hidden_states.gather(
        torch.tensor([[1, 6]]),
        device=torch.device("cpu"),
    )
    assert torch.equal(
        gathered,
        tensors["verifier_last_hidden_states"][:1, [1, 6]],
    )


def test_dflash_feature_manifest_rejects_verifier_mismatch(tmp_path) -> None:
    path, _ = _write_features(tmp_path)

    with pytest.raises(ValueError, match="verifier_id mismatch"):
        DFlashFeatureDataRuntime.from_path(
            path=str(path),
            batch_size=1,
            seq_len=8,
            device=torch.device("cpu"),
            dtype=torch.float32,
            require_teacher_features=True,
            verifier_id="wrong/verifier",
            verifier_revision="revision",
            target_layer_ids=(1, 3),
            runtime=None,
            data_parallel_rank=0,
            data_parallel_size=1,
        )


def test_dflash_feature_transfer_uses_model_compute_dtype(tmp_path) -> None:
    path, _ = _write_features(tmp_path)
    data = DFlashFeatureDataRuntime.from_path(
        path=str(path),
        batch_size=1,
        seq_len=8,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        require_teacher_features=True,
        verifier_id="unit/verifier",
        verifier_revision="revision",
        target_layer_ids=(1, 3),
        runtime=None,
        data_parallel_rank=0,
        data_parallel_size=1,
    )

    batch = data.next_batch()

    assert batch.target_hidden_states.dtype == torch.bfloat16
    assert batch.verifier_last_hidden_states.dtype == torch.bfloat16


def test_dflash_feature_runtime_returns_ordered_microbatches(tmp_path) -> None:
    path, tensors = _write_features(tmp_path)
    data = DFlashFeatureDataRuntime.from_path(
        path=str(path),
        batch_size=1,
        seq_len=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
        require_teacher_features=True,
        verifier_id="unit/verifier",
        verifier_revision="revision",
        target_layer_ids=(1, 3),
        runtime=None,
        data_parallel_rank=0,
        data_parallel_size=1,
    )

    batches = data.next_batches(2)

    assert len(batches) == 2
    assert torch.equal(batches[0].input_ids, tensors["input_ids"][:1])
    assert torch.equal(batches[1].input_ids, tensors["input_ids"][1:2])
    assert data.samples_consumed == 2
    with pytest.raises(ValueError, match="batch count must be positive"):
        data.next_batches(0)


def test_dflash_feature_runtime_restores_the_next_sample(tmp_path) -> None:
    path, tensors = _write_features(tmp_path)
    data = DFlashFeatureDataRuntime.from_path(
        path=str(path), batch_size=1, seq_len=8, device=torch.device("cpu"),
        dtype=torch.float32, require_teacher_features=True,
        verifier_id="unit/verifier", verifier_revision="revision",
        target_layer_ids=(1, 3), runtime=None,
        data_parallel_rank=0, data_parallel_size=1,
    )
    data.next_batch()
    checkpoint = data.state_dict()

    restored = DFlashFeatureDataRuntime.from_path(
        path=str(path), batch_size=1, seq_len=8, device=torch.device("cpu"),
        dtype=torch.float32, require_teacher_features=True,
        verifier_id="unit/verifier", verifier_revision="revision",
        target_layer_ids=(1, 3), runtime=None,
        data_parallel_rank=0, data_parallel_size=1,
    )
    restored.load_state_dict(checkpoint)
    resumed = restored.next_batch()

    assert torch.equal(resumed.input_ids, tensors["input_ids"][1:2])
    assert restored.samples_consumed == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA prefetch")
def test_dflash_cuda_prefetch_preserves_order_and_restore(tmp_path) -> None:
    path, tensors = _write_features(tmp_path)

    def build_runtime() -> DFlashFeatureDataRuntime:
        return DFlashFeatureDataRuntime.from_path(
            path=str(path),
            batch_size=1,
            seq_len=8,
            device=torch.device("cuda"),
            dtype=torch.float32,
            require_teacher_features=True,
            verifier_id="unit/verifier",
            verifier_revision="revision",
            target_layer_ids=(1, 3),
            runtime=None,
            data_parallel_rank=0,
            data_parallel_size=1,
        )

    data = build_runtime()
    first = data.next_batch()
    torch.cuda.current_stream().synchronize()
    assert torch.equal(first.input_ids.cpu(), tensors["input_ids"][:1])
    checkpoint = data.state_dict()

    restored = build_runtime()
    restored.load_state_dict(checkpoint)
    resumed = restored.next_batch()
    teacher = resumed.verifier_last_hidden_states.gather(
        torch.tensor([[1, 6]], device="cuda"),
        device=torch.device("cuda"),
    )
    torch.cuda.current_stream().synchronize()

    assert torch.equal(resumed.input_ids.cpu(), tensors["input_ids"][1:2])
    assert torch.equal(
        teacher.cpu(),
        tensors["verifier_last_hidden_states"][1:2, [1, 6]],
    )
