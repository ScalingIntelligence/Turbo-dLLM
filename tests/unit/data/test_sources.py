from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dllm_parallel.data import sources
from dllm_parallel.data.registry import register_source
from dllm_parallel.data.schemas import SourceSpec
from dllm_parallel.data.sources import iter_source_records, source_provenance


def test_text_source_reads_sorted_documents_and_nonempty_lines(tmp_path: Path) -> None:
    (tmp_path / "b.txt").write_text("beta\n\nsecond\n", encoding="utf-8")
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")

    documents = list(
        iter_source_records(SourceSpec(type="text", path=str(tmp_path / "*.txt")))
    )
    lines = list(
        iter_source_records(
            SourceSpec(
                type="text",
                path=str(tmp_path / "*.txt"),
                text_mode="line",
            )
        )
    )

    assert documents == ["alpha\n", "beta\n\nsecond\n"]
    assert lines == ["alpha", "beta", "second"]


def test_jsonl_source_reads_plain_and_gzip_records(tmp_path: Path) -> None:
    plain = tmp_path / "a.jsonl"
    plain.write_text('{"text":"one"}\n\n{"text":"two"}\n', encoding="utf-8")
    compressed = tmp_path / "b.jsonl.gz"
    with gzip.open(compressed, "wt", encoding="utf-8") as handle:
        handle.write('{"text":"three"}\n')

    records = list(
        iter_source_records(
            SourceSpec(type="jsonl", paths=(str(compressed), str(plain)))
        )
    )

    assert records == [{"text": "one"}, {"text": "two"}, {"text": "three"}]


def test_jsonl_source_reports_file_and_line_for_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"ok":1}\nnot-json\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"bad\.jsonl:2"):
        list(iter_source_records(SourceSpec(type="jsonl", path=str(path))))


def test_local_source_reports_an_unmatched_glob(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="matched no files"):
        list(
            iter_source_records(
                SourceSpec(type="jsonl", path=str(tmp_path / "missing-*.jsonl"))
            )
        )


@pytest.mark.parametrize(("suffix", "dtype"), [("i32", np.int32), ("i64", np.int64)])
def test_pretokenized_source_reads_binary_tokens(
    tmp_path: Path,
    suffix: str,
    dtype: type[np.generic],
) -> None:
    path = tmp_path / f"tokens.{suffix}"
    np.asarray([1, 2, 3], dtype=dtype).tofile(path)

    records = list(iter_source_records(SourceSpec(type="pretokenized", path=str(path))))

    assert len(records) == 1
    assert np.array_equal(records[0]["input_ids"], [1, 2, 3])


def test_pretokenized_binary_source_is_memory_mapped(tmp_path: Path) -> None:
    path = tmp_path / "tokens.i32"
    np.arange(1024, dtype=np.int32).tofile(path)

    record = next(iter_source_records(SourceSpec(type="pretokenized", path=str(path))))

    assert isinstance(record["input_ids"], np.memmap)
    assert record["input_ids"].dtype == np.int32
    assert Path(record["input_ids"].filename) == path.resolve()


def test_pretokenized_source_reads_numpy_and_torch_payloads(tmp_path: Path) -> None:
    numpy_path = tmp_path / "records.npy"
    np.save(numpy_path, np.asarray([[1, 2], [3, 4]], dtype=np.int64))
    torch_path = tmp_path / "records.pt"
    torch.save({"input_ids": torch.tensor([[5, 6], [7, 8]])}, torch_path)

    numpy_records = list(
        iter_source_records(SourceSpec(type="pretokenized", path=str(numpy_path)))
    )
    torch_records = list(
        iter_source_records(SourceSpec(type="pretokenized", path=str(torch_path)))
    )

    assert [record["input_ids"].tolist() for record in numpy_records] == [
        [1, 2],
        [3, 4],
    ]
    assert [record["input_ids"].tolist() for record in torch_records] == [
        [5, 6],
        [7, 8],
    ]


def test_pretokenized_source_preserves_columnar_masks_and_json_token_vectors(
    tmp_path: Path,
) -> None:
    torch_path = tmp_path / "supervised.pt"
    torch.save(
        {
            "input_ids": torch.tensor([[1, 2], [3, 4]]),
            "loss_mask": torch.tensor([[0, 1], [1, 1]]),
            "sample_id": torch.tensor([10, 11]),
        },
        torch_path,
    )
    json_path = tmp_path / "tokens.json"
    json_path.write_text("[5, 6, 7]", encoding="utf-8")

    records = list(
        iter_source_records(SourceSpec(type="pretokenized", path=str(torch_path)))
    )
    assert [record["input_ids"].tolist() for record in records] == [[1, 2], [3, 4]]
    assert [record["loss_mask"] for record in records] == [[0, 1], [1, 1]]
    assert [record["sample_id"] for record in records] == [10, 11]
    assert list(
        iter_source_records(SourceSpec(type="pretokenized", path=str(json_path)))
    ) == [{"input_ids": [5, 6, 7]}]


def test_huggingface_and_parquet_sources_forward_explicit_loader_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def load_dataset(*args: object, **kwargs: object) -> list[dict[str, object]]:
        calls.append((args, kwargs))
        return [{"text": "loaded"}]

    monkeypatch.setitem(
        sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset)
    )
    parquet = tmp_path / "records.parquet"
    parquet.write_bytes(b"parquet-placeholder")

    assert list(
        iter_source_records(
            SourceSpec(
                type="huggingface",
                id="org/data",
                name="subset",
                split="validation",
                revision="abc123",
                streaming=True,
                loader_kwargs={"token": False},
            )
        )
    ) == [{"text": "loaded"}]
    assert list(
        iter_source_records(
            SourceSpec(type="parquet", path=str(parquet), split="train")
        )
    ) == [{"text": "loaded"}]

    assert calls[0] == (
        ("org/data", "subset"),
        {
            "revision": "abc123",
            "split": "validation",
            "streaming": True,
            "token": False,
        },
    )
    assert calls[1] == (
        ("parquet",),
        {"data_files": [str(parquet.resolve())], "split": "train"},
    )


def test_optional_dataset_dependency_error_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "datasets", None)

    with pytest.raises(RuntimeError, match=r"turbo-dllm\[data\]"):
        list(
            iter_source_records(
                SourceSpec(type="huggingface", id="org/data", split="train")
            )
        )


def test_custom_source_and_provenance_hashes(tmp_path: Path) -> None:
    register_source(
        "source-test-custom",
        lambda spec: iter(({"text": spec.id},)),
        replace=True,
    )
    custom = list(
        iter_source_records(SourceSpec(type="source-test-custom", id="value"))
    )
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps({"text": "hello"}) + "\n", encoding="utf-8")
    provenance = source_provenance(SourceSpec(type="jsonl", path=str(path)))

    assert custom == [{"text": "value"}]
    assert provenance["type"] == "jsonl"
    assert provenance["files"][0]["path"] == str(path.resolve())
    assert provenance["files"][0]["bytes"] == path.stat().st_size
    assert len(provenance["files"][0]["sha256"]) == 64


def test_resolved_source_paths_are_reused_for_iteration_and_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "tokens.i32"
    np.asarray([1, 2, 3], dtype=np.int32).tofile(path)
    spec = SourceSpec(type="pretokenized", path=str(path))
    resolved = (path.resolve(),)

    def unexpected_resolution(spec: SourceSpec) -> tuple[Path, ...]:
        raise AssertionError("already-resolved paths must be reused")

    monkeypatch.setattr(sources, "resolve_source_paths", unexpected_resolution)

    record = next(iter_source_records(spec, resolved_paths=resolved))
    provenance = source_provenance(spec, resolved_paths=resolved)

    assert record["input_ids"].tolist() == [1, 2, 3]
    assert provenance["files"][0]["path"] == str(path.resolve())
