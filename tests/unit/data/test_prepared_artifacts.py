from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from dllm_parallel.data import indexed
from dllm_parallel.core.data import (
    IndexedSupervisedTokenDataRuntime,
    PackedTokenDataRuntime,
)
from dllm_parallel.data.indexed import (
    PACKED_TOKEN_FORMAT,
    inspect_artifact,
    validate_artifact,
    validate_artifact_for_run,
    write_packed_artifact,
)
from dllm_parallel.data.prepare import PreparationResult, prepare_dataset
from dllm_parallel.data.schemas import PreparationSpec
from dllm_parallel.data.schemas import PackingSpec
from dllm_parallel.data.tokenization import TokenizedRecord


class ArtifactTokenizer:
    eos_token_id = 9
    pad_token_id = 0
    mask_token_id = 8
    chat_template = "artifact-template"

    def __len__(self) -> int:
        return 10

    def get_vocab(self) -> dict[str, int]:
        return {str(index): index for index in range(10)}

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [ord(character) % 7 + 1 for character in text]


def _packed_spec(
    source: Path, output: Path, *, overwrite: bool = False
) -> PreparationSpec:
    return PreparationSpec.from_mapping(
        {
            "source": {"type": "jsonl", "path": str(source)},
            "records": {"type": "pretokenized"},
            "tokenizer": {"add_eos": False},
            "supervision": {"policy": "full"},
            "packing": {
                "maximum_length": 4,
                "separator_token_id": 7,
                "overflow": "truncate_right",
            },
            "output": {
                "path": str(output),
                "format": "packed",
                "overwrite": overwrite,
            },
        }
    )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_prepare_packed_artifact_is_versioned_checksummed_and_deterministic(
    tmp_path: Path,
) -> None:
    source = tmp_path / "records.jsonl"
    _write_jsonl(source, [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5, 6, 1, 2]}])

    first = prepare_dataset(_packed_spec(source, tmp_path / "first"))
    second = prepare_dataset(_packed_spec(source, tmp_path / "second"))

    assert isinstance(first, PreparationResult)
    assert first.format == PACKED_TOKEN_FORMAT
    assert first.training_path == first.artifact_path
    assert first.token_count == 8
    assert first.sample_count == 2
    assert first.dataset_fingerprint == second.dataset_fingerprint
    assert (first.artifact_path / "tokens.i32").is_file()
    manifest = validate_artifact(first.artifact_path)
    assert manifest["format"] == PACKED_TOKEN_FORMAT
    assert manifest["version"] == 1
    assert manifest["metadata"]["sequence_layout"] == "continuous_stream"
    assert manifest["metadata"]["supervision_shape"] == "full_sequence"
    assert manifest["metadata"]["padding"] == "none"
    assert (
        manifest["metadata"]["sampling_policy"] == "sequential_cyclic_rank_strided_v1"
    )
    assert manifest["stats"] == {
        "logical_chunks": 2,
        "records": 2,
        "separator_tokens": 1,
        "split_records": 0,
        "source_tokens": 8,
        "stored_tokens": 8,
        "truncated_tokens": 1,
    }
    assert inspect_artifact(first.artifact_path)["status"] == "present"


def test_prepare_tokenized_packed_artifact_records_tokenizer_identity(
    tmp_path: Path,
) -> None:
    source = tmp_path / "records.jsonl"
    _write_jsonl(source, [{"text": "generic training text"}])
    spec = PreparationSpec.from_mapping(
        {
            "source": {"type": "jsonl", "path": str(source)},
            "records": {"type": "text"},
            "tokenizer": {"model": "example/tokenizer", "revision": "abc123"},
            "supervision": {"policy": "full"},
            "packing": {"maximum_length": 64},
            "output": {"path": str(tmp_path / "prepared"), "format": "packed"},
        }
    )

    manifest = prepare_dataset(spec, tokenizer=ArtifactTokenizer()).manifest
    metadata = manifest["metadata"]

    assert metadata["tokenizer_model"] == "example/tokenizer"
    assert metadata["tokenizer_revision"] == "abc123"
    assert metadata["tokenizer_size"] == 10
    assert len(metadata["tokenizer_vocabulary_sha256"]) == 64
    assert len(metadata["chat_template_sha256"]) == 64

    runtime = PackedTokenDataRuntime.from_dataset_path(
        dataset_path=str(tmp_path / "prepared"),
        tokenizer=ArtifactTokenizer(),
        batch_size=1,
        seq_len=4,
        device=torch.device("cpu"),
    )
    assert runtime.dataset_fingerprint == manifest["metadata"]["dataset_fingerprint"]

    class WrongTokenizer(ArtifactTokenizer):
        def __len__(self) -> int:
            return 11

    with pytest.raises(RuntimeError, match="tokenizer size"):
        PackedTokenDataRuntime.from_dataset_path(
            dataset_path=str(tmp_path / "prepared"),
            tokenizer=WrongTokenizer(),
            batch_size=1,
            seq_len=4,
            device=torch.device("cpu"),
        )


def test_prepare_packed_split_is_lossless_and_does_not_add_chunk_separators(
    tmp_path: Path,
) -> None:
    source = tmp_path / "records.jsonl"
    _write_jsonl(source, [{"input_ids": [1, 2, 3, 4, 5]}, {"input_ids": [6]}])
    mapping = _packed_spec(source, tmp_path / "prepared").to_mapping()
    mapping["packing"]["maximum_length"] = 2
    mapping["packing"]["overflow"] = "split"

    result = prepare_dataset(PreparationSpec.from_mapping(mapping))
    tokens = torch.from_file(
        str(result.artifact_path / "tokens.i32"),
        dtype=torch.int32,
        size=result.token_count,
    )

    assert tokens.tolist() == [1, 2, 3, 4, 5, 7, 6]
    assert result.manifest["stats"]["source_tokens"] == 6
    assert result.manifest["stats"]["truncated_tokens"] == 0
    assert result.manifest["stats"]["split_records"] == 1
    assert result.manifest["stats"]["logical_chunks"] == 4


def test_packed_split_writes_bounded_segments(tmp_path: Path) -> None:
    class BoundedSequence:
        def __init__(self, values: tuple[int, ...], maximum_slice: int) -> None:
            self.values = values
            self.maximum_slice = maximum_slice

        def __len__(self) -> int:
            return len(self.values)

        def __iter__(self):
            raise AssertionError("writer must not materialize the complete record")

        def __getitem__(self, item):
            if isinstance(item, slice):
                start, stop, step = item.indices(len(self.values))
                assert step == 1
                assert stop - start <= self.maximum_slice
                return self.values[item]
            return self.values[item]

    tokens = BoundedSequence((1, 2, 3, 4, 5), maximum_slice=2)
    manifest = write_packed_artifact(
        tmp_path / "prepared",
        [TokenizedRecord(tokens=tokens)],  # type: ignore[arg-type]
        packing=PackingSpec(maximum_length=2, overflow="split"),
        metadata={"semantics": "full_sequence"},
    )

    assert manifest["token_count"] == 5
    assert (tmp_path / "prepared" / "tokens.i32").read_bytes() == (
        torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32).numpy().tobytes()
    )


def test_validate_artifact_detects_payload_corruption(tmp_path: Path) -> None:
    source = tmp_path / "records.jsonl"
    _write_jsonl(source, [{"input_ids": [1, 2, 3]}])
    result = prepare_dataset(_packed_spec(source, tmp_path / "prepared"))
    with (result.artifact_path / "tokens.i32").open("r+b") as handle:
        handle.write(b"\xff\xff\xff\xff")

    with pytest.raises(RuntimeError, match="checksum"):
        validate_artifact(result.artifact_path)


def test_fingerprint_is_content_based_and_manifest_tampering_is_detected(
    tmp_path: Path,
) -> None:
    first_source = tmp_path / "one" / "records.jsonl"
    second_source = tmp_path / "two" / "records.jsonl"
    first_source.parent.mkdir()
    second_source.parent.mkdir()
    _write_jsonl(first_source, [{"input_ids": [1, 2, 3]}])
    _write_jsonl(second_source, [{"input_ids": [1, 2, 3]}])

    first = prepare_dataset(_packed_spec(first_source, tmp_path / "first"))
    second = prepare_dataset(_packed_spec(second_source, tmp_path / "second"))

    assert first.dataset_fingerprint == second.dataset_fingerprint
    manifest_path = first.artifact_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"]["maximum_record_length"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="fingerprint"):
        validate_artifact(first.artifact_path)


def test_preparation_provenance_redacts_loader_credentials(tmp_path: Path) -> None:
    source = tmp_path / "records.jsonl"
    _write_jsonl(source, [{"input_ids": [1, 2]}])
    mapping = _packed_spec(source, tmp_path / "prepared").to_mapping()
    mapping["source"]["loader_kwargs"] = {
        "use_auth_token": "sensitive",
        "verification_mode": "all_checks",
    }
    result = prepare_dataset(PreparationSpec.from_mapping(mapping))
    recorded = result.manifest["metadata"]["preparation"]["source"]["loader_kwargs"]

    assert recorded["use_auth_token"] == "<redacted>"
    assert recorded["verification_mode"] == "all_checks"


def test_prepare_rejects_existing_destination_until_overwrite_is_explicit(
    tmp_path: Path,
) -> None:
    source = tmp_path / "records.jsonl"
    _write_jsonl(source, [{"input_ids": [1, 2]}])
    destination = tmp_path / "prepared"
    destination.mkdir()
    (destination / "keep.txt").write_text("old", encoding="utf-8")

    with pytest.raises(FileExistsError):
        prepare_dataset(_packed_spec(source, destination))

    result = prepare_dataset(_packed_spec(source, destination, overwrite=True))
    assert result.artifact_path == destination
    assert not (destination / "keep.txt").exists()
    assert validate_artifact(destination)["format"] == PACKED_TOKEN_FORMAT


def test_existing_destination_is_rejected_before_reading_source(tmp_path: Path) -> None:
    source = tmp_path / "records.jsonl"
    source.write_text("not-json\n", encoding="utf-8")
    destination = tmp_path / "prepared"
    destination.mkdir()

    with pytest.raises(FileExistsError):
        prepare_dataset(_packed_spec(source, destination))


def test_prepare_indexed_artifact_round_trips_arbitrary_supervision(
    tmp_path: Path,
) -> None:
    source = tmp_path / "records.jsonl"
    _write_jsonl(
        source,
        [
            {"input_ids": [1, 2, 3, 4], "loss_mask": [0, 1, 0, 1], "id": 17},
            {"input_ids": [4, 3, 2, 1], "loss_mask": [0, 0, 1, 1], "id": 18},
        ],
    )
    spec = PreparationSpec.from_mapping(
        {
            "source": {"type": "jsonl", "path": str(source)},
            "records": {
                "type": "pretokenized",
                "sample_id_field": "id",
            },
            "tokenizer": {"model": "unused", "add_eos": False},
            "supervision": {"policy": "provided"},
            "packing": {"maximum_length": 4, "alignment": 2},
            "output": {"path": str(tmp_path / "indexed"), "format": "indexed"},
        }
    )

    result = prepare_dataset(spec, tokenizer=ArtifactTokenizer())
    manifest = validate_artifact(result.artifact_path)
    runtime = IndexedSupervisedTokenDataRuntime.from_directory(
        result.artifact_path,
        tokenizer=ArtifactTokenizer(),
        mask_token_id=8,
        batch_size=2,
        max_seq_len=4,
        block_size=2,
        context_parallel_size=1,
        block_parallel_size=1,
        device=torch.device("cpu"),
        data_parallel_rank=0,
        data_parallel_size=1,
        seed=0,
        shuffle=False,
        minimum_sequence_length=None,
    )
    batch = runtime.next_batch()

    assert manifest["metadata"]["semantics"] == "supervised_tokens"
    assert manifest["metadata"]["sequence_layout"] == "record_aligned"
    assert manifest["metadata"]["supervision_shape"] == "arbitrary"
    assert manifest["metadata"]["padding"] == "none"
    assert manifest["files"]["loss_mask.u8"]["sha256"]
    assert batch.input_ids.tolist() == [[1, 2, 3, 4], [4, 3, 2, 1]]
    assert batch.loss_mask.tolist() == [
        [False, True, False, True],
        [False, False, True, True],
    ]
    assert batch.sample_ids.tolist() == [17, 18]

    with pytest.raises(RuntimeError, match="contiguous-suffix"):
        IndexedSupervisedTokenDataRuntime.from_directory(
            result.artifact_path,
            tokenizer=ArtifactTokenizer(),
            mask_token_id=8,
            batch_size=2,
            max_seq_len=4,
            block_size=2,
            context_parallel_size=1,
            block_parallel_size=1,
            device=torch.device("cpu"),
            data_parallel_rank=0,
            data_parallel_size=1,
            seed=0,
            shuffle=False,
            minimum_sequence_length=None,
            group_by_supervision_start=True,
        )


@pytest.mark.parametrize(
    ("policy", "loss_mask", "expected_shape"),
    [
        ("full", None, "full_sequence"),
        ("provided", [0, 0, 1, 1], "contiguous_suffix"),
    ],
)
def test_prepare_indexed_artifact_derives_supervision_shape(
    tmp_path: Path,
    policy: str,
    loss_mask: list[int] | None,
    expected_shape: str,
) -> None:
    source = tmp_path / f"{policy}.jsonl"
    record: dict[str, Any] = {"input_ids": [1, 2, 3, 4]}
    if loss_mask is not None:
        record["loss_mask"] = loss_mask
    _write_jsonl(source, [record])
    result = prepare_dataset(
        PreparationSpec.from_mapping(
            {
                "source": {"type": "jsonl", "path": str(source)},
                "records": {"type": "pretokenized"},
                "tokenizer": {"model": "unused"},
                "supervision": {"policy": policy},
                "packing": {"maximum_length": 4},
                "output": {
                    "path": str(tmp_path / f"prepared-{policy}"),
                    "format": "indexed",
                },
            }
        ),
        tokenizer=ArtifactTokenizer(),
    )

    assert result.manifest["metadata"]["supervision_shape"] == expected_shape


def test_validate_artifact_for_run_checks_length_alignment_and_input_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "records.jsonl"
    _write_jsonl(
        source,
        [{"input_ids": [1, 2, 3, 4], "loss_mask": [0, 1, 0, 1]}],
    )
    spec = PreparationSpec.from_mapping(
        {
            "source": {"type": "jsonl", "path": str(source)},
            "records": {"type": "pretokenized"},
            "tokenizer": {"model": "unused", "add_eos": False},
            "supervision": {"policy": "provided"},
            "packing": {"maximum_length": 4, "alignment": 2},
            "output": {"path": str(tmp_path / "indexed")},
        }
    )
    result = prepare_dataset(spec, tokenizer=ArtifactTokenizer())
    compatible = SimpleNamespace(
        data=SimpleNamespace(input_mode="dataset"),
        model=SimpleNamespace(seq_len=4),
        objective=SimpleNamespace(name="standard_block_diffusion", block_size=2),
    )
    assert validate_artifact_for_run(result.artifact_path, compatible)["compatible"]
    manifest = validate_artifact(result.artifact_path)

    def unexpected_validation(path: str | Path) -> dict[str, Any]:
        raise AssertionError("a supplied validated manifest must be reused")

    with monkeypatch.context() as context:
        context.setattr(indexed, "validate_artifact", unexpected_validation)
        assert validate_artifact_for_run(
            result.artifact_path,
            compatible,
            manifest=manifest,
        )["compatible"]

    with pytest.raises(ValueError, match="block size"):
        validate_artifact_for_run(
            result.artifact_path,
            SimpleNamespace(
                data=SimpleNamespace(input_mode="dataset"),
                model=SimpleNamespace(seq_len=4),
                objective=SimpleNamespace(
                    name="standard_block_diffusion", block_size=4
                ),
            ),
        )
    with pytest.raises(ValueError, match="input_mode=dataset"):
        validate_artifact_for_run(
            result.artifact_path,
            SimpleNamespace(
                data=SimpleNamespace(input_mode="random"),
                model=SimpleNamespace(seq_len=4),
                objective=SimpleNamespace(
                    name="standard_block_diffusion", block_size=2
                ),
            ),
        )

    with pytest.raises(ValueError, match="contiguous suffix"):
        validate_artifact_for_run(
            result.artifact_path,
            SimpleNamespace(
                data=SimpleNamespace(input_mode="dataset"),
                model=SimpleNamespace(seq_len=4),
                objective=SimpleNamespace(
                    name="diffusiongemma_native_sft", block_size=2
                ),
            ),
        )


def test_validate_indexed_artifact_checks_exact_supervision_start(
    tmp_path: Path,
) -> None:
    source = tmp_path / "records.jsonl"
    _write_jsonl(source, [{"input_ids": [1, 2], "loss_mask": [0, 1]}])
    spec = PreparationSpec.from_mapping(
        {
            "source": {"type": "jsonl", "path": str(source)},
            "records": {"type": "pretokenized"},
            "tokenizer": {"model": "unused"},
            "supervision": {"policy": "provided"},
            "packing": {"maximum_length": 2, "alignment": 1},
            "output": {"path": str(tmp_path / "indexed")},
        }
    )
    result = prepare_dataset(spec, tokenizer=ArtifactTokenizer())
    index = torch.from_file(
        str(result.artifact_path / "index.i64"),
        shared=True,
        dtype=torch.int64,
        size=4,
    )
    index[2] = 0

    with pytest.raises(RuntimeError, match="supervision start"):
        validate_artifact(result.artifact_path, verify_checksums=False)
