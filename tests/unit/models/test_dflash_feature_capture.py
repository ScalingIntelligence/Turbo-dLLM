from __future__ import annotations

import json

import torch

from dllm_parallel.core.models.backbones.dflash.data import (
    DFlashFeatureDataRuntime,
    FEATURE_COMPACT_SHARDED_FORMAT,
)
from dllm_parallel.core.models.backbones.dflash.feature_capture import (
    IndexedDFlashCaptureSource,
    capture_dflash_features,
    resolve_dflash_capture_contract,
)
from dllm_parallel.data.indexed import write_indexed_artifact
from dllm_parallel.data.schemas import PackingSpec
from dllm_parallel.data.tokenization import TokenizedRecord


def _write_source(tmp_path):
    root = tmp_path / "prepared"
    write_indexed_artifact(
        root,
        [
            TokenizedRecord(
                tokens=(3, 4, 5),
                loss_mask=(False, True, True),
                sample_id=11,
            ),
            TokenizedRecord(
                tokens=(6, 7),
                loss_mask=(False, True),
                sample_id=12,
            ),
        ],
        packing=PackingSpec(maximum_length=8, alignment=1),
        metadata={
            "semantics": "assistant_only_sft",
            "pad_token_id": 1,
            "tokenizer_model": "unit/tokenizer",
            "tokenizer_revision": "revision",
        },
        include_groups=False,
    )
    return root


class _CaptureBackend:
    def __init__(self):
        self.layers = None

    def set_capture_layers(self, layers, *, capture_method):
        self.layers = (tuple(layers), capture_method)

    def capture_rows(self, rows):
        captured = []
        final = []
        for row in rows:
            values = torch.tensor(row, dtype=torch.float32)
            captured.append(torch.stack((values, values + 1), dim=-1))
            final.append(values[:, None])
        return captured, final


def test_indexed_capture_source_accepts_generic_supervised_artifact(tmp_path) -> None:
    source = IndexedDFlashCaptureSource.open(_write_source(tmp_path))

    input_ids, loss_mask, sample_id = source.record(1)

    assert input_ids.tolist() == [6, 7]
    assert loss_mask.tolist() == [False, True]
    assert sample_id == 12
    assert source.pad_token_id == 1


def test_capture_contract_uses_published_and_canonical_layer_ids(tmp_path) -> None:
    draft = tmp_path / "draft"
    draft.mkdir()
    (draft / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "dflash_config": {
                    "block_size": 16,
                    "target_layer_ids": [5, 19, 33],
                },
            }
        ),
        encoding="utf-8",
    )

    contract = resolve_dflash_capture_contract(draft)

    assert contract.block_size == 16
    assert contract.capture_layer_ids == (5, 19, 33)
    assert contract.target_layer_ids == (6, 20, 34)


def test_generic_capture_writes_artifact_consumed_by_training_runtime(tmp_path) -> None:
    source = _write_source(tmp_path)
    output = tmp_path / "features"
    backend = _CaptureBackend()

    manifest = capture_dflash_features(
        source=source,
        output=output,
        capture_backend=backend,
        capture_layer_ids=(5,),
        target_layer_ids=(6,),
        verifier_id="unit/verifier",
        verifier_revision="revision",
        sequence_length=8,
    )

    assert backend.layers == ((5,), "dflash")
    assert manifest["format"] == FEATURE_COMPACT_SHARDED_FORMAT
    assert manifest["sample_count"] == 2
    assert manifest["target_feature_width"] == 2
    runtime = DFlashFeatureDataRuntime.from_path(
        path=str(output),
        batch_size=1,
        seq_len=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
        require_teacher_features=False,
        verifier_id="unit/verifier",
        verifier_revision="revision",
        target_layer_ids=(6,),
        runtime=None,
        data_parallel_rank=0,
        data_parallel_size=1,
    )
    batch = runtime.next_batch()
    assert batch.input_ids.tolist() == [[3, 4, 5, 1, 1, 1, 1, 1]]
    assert batch.loss_mask.tolist() == [
        [False, True, True, False, False, False, False, False]
    ]
    assert batch.target_hidden_states.shape == (1, 8, 2)
    assert torch.count_nonzero(batch.target_hidden_states[:, 3:]).item() == 0
