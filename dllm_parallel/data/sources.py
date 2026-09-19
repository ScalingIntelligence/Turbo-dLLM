# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Raw-record sources for generic offline data preparation."""

from __future__ import annotations

import glob
import gzip
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dllm_parallel.data.provenance import sha256_file
from dllm_parallel.data.registry import get_source
from dllm_parallel.data.schemas import BUILTIN_SOURCE_TYPES, SourceSpec


def iter_source_records(
    spec: SourceSpec,
    *,
    resolved_paths: Sequence[Path] | None = None,
) -> Iterator[Any]:
    """Yield raw records from one configured source."""

    source_type = _normalized(spec.type)
    if source_type == "text":
        yield from _text_records(spec, resolved_paths)
    elif source_type == "jsonl":
        yield from _jsonl_records(spec, resolved_paths)
    elif source_type == "pretokenized":
        yield from _pretokenized_records(spec, resolved_paths)
    elif source_type == "huggingface":
        yield from _huggingface_records(spec)
    elif source_type == "parquet":
        yield from _parquet_records(spec, resolved_paths)
    else:
        if source_type not in BUILTIN_SOURCE_TYPES:
            yield from get_source(source_type)(spec)
        else:  # pragma: no cover - exhaustive guard
            raise ValueError(f"unsupported source type: {source_type}")


def source_provenance(
    spec: SourceSpec,
    *,
    resolved_paths: Sequence[Path] | None = None,
) -> dict[str, Any]:
    """Return stable source identity without reading records into memory."""

    source_type = _normalized(spec.type)
    result: dict[str, Any] = {
        "type": source_type,
        "id": spec.id,
        "name": spec.name,
        "split": spec.split,
        "revision": spec.revision,
        "streaming": bool(spec.streaming),
    }
    paths = resolve_source_paths(spec) if resolved_paths is None else resolved_paths
    if paths:
        result["files"] = [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in paths
        ]
    return result


def resolve_source_paths(spec: SourceSpec) -> tuple[Path, ...]:
    """Resolve local source inputs once for provenance and record iteration."""

    raw_values = list(spec.paths)
    if spec.path is not None:
        raw_values.append(spec.path)
    paths: set[Path] = set()
    for raw_value in raw_values:
        expanded = Path(raw_value).expanduser()
        matches = [Path(value) for value in glob.glob(str(expanded), recursive=True)]
        if not matches and not glob.has_magic(str(expanded)):
            matches = [expanded]
        for match in matches:
            if match.is_dir():
                paths.update(
                    path.resolve() for path in match.rglob("*") if path.is_file()
                )
            elif match.is_file():
                paths.add(match.resolve())
            else:
                raise FileNotFoundError(str(match))
    if raw_values and not paths:
        joined = ", ".join(str(value) for value in raw_values)
        raise FileNotFoundError(f"source path or pattern matched no files: {joined}")
    return tuple(sorted(paths, key=lambda path: str(path)))


def _text_records(
    spec: SourceSpec,
    paths: Sequence[Path] | None,
) -> Iterator[str]:
    for path in resolve_source_paths(spec) if paths is None else paths:
        text = path.read_text(encoding=spec.encoding)
        if spec.text_mode == "document":
            if text:
                yield text
        else:
            for line in text.splitlines():
                if line.strip():
                    yield line


def _jsonl_records(
    spec: SourceSpec,
    paths: Sequence[Path] | None = None,
) -> Iterator[Mapping[str, Any]]:
    for path in resolve_source_paths(spec) if paths is None else paths:
        opener = gzip.open if path.name.endswith(".gz") else open
        with opener(path, "rt", encoding=spec.encoding) as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSON at {path}:{line_number}: {exc.msg}"
                    ) from exc
                if not isinstance(payload, Mapping):
                    raise ValueError(
                        f"JSONL record at {path}:{line_number} must be an object"
                    )
                yield payload


def _pretokenized_records(
    spec: SourceSpec,
    paths: Sequence[Path] | None,
) -> Iterator[Mapping[str, Any]]:
    for path in resolve_source_paths(spec) if paths is None else paths:
        yield from _pretokenized_path(path, encoding=spec.encoding)


def _pretokenized_path(path: Path, *, encoding: str) -> Iterator[Mapping[str, Any]]:
    suffix = path.suffix.lower()
    binary_dtype = _binary_token_dtype(suffix)
    if binary_dtype is not None:
        yield {"input_ids": np.memmap(path, dtype=binary_dtype, mode="r")}
        return
    if suffix == ".npy":
        yield from _array_records(np.load(path, allow_pickle=False, mmap_mode="r"))
        return
    if suffix == ".pt":
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        yield from _array_records(payload)
        return
    if suffix == ".json":
        yield from _pretokenized_json(path, encoding=encoding)
        return
    if path.name.endswith((".jsonl", ".jsonl.gz")):
        yield from _jsonl_records(
            SourceSpec(type="jsonl", path=str(path), encoding=encoding),
            (path,),
        )
        return
    raise ValueError(f"unsupported pretokenized source file: {path}")


def _binary_token_dtype(suffix: str) -> type[np.int32] | type[np.int64] | None:
    if suffix in {".i32", ".int32"}:
        return np.int32
    if suffix in {".i64", ".int64", ".bin", ".tokens"}:
        return np.int64
    return None


def _pretokenized_json(
    path: Path,
    *,
    encoding: str,
) -> Iterator[Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding=encoding))
    if isinstance(payload, Mapping):
        yield payload
        return
    if not isinstance(payload, list):
        raise ValueError(f"pretokenized JSON must be an object or list: {path}")
    if not payload or all(isinstance(item, int) for item in payload):
        yield {"input_ids": payload}
        return
    for item in payload:
        yield item if isinstance(item, Mapping) else {"input_ids": item}


def _array_records(value: Any) -> Iterator[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield from _mapping_array_records(value)
        return
    array = _cpu_array(value)
    if array.ndim == 1:
        yield {"input_ids": array}
    elif array.ndim == 2:
        for row in array:
            yield {"input_ids": row}
    else:
        raise ValueError("pretokenized tensors must have one or two dimensions")


def _mapping_array_records(value: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    input_ids = value.get("input_ids")
    if input_ids is None:
        yield {key: _plain(column) for key, column in value.items()}
        return
    input_array = _cpu_array(input_ids)
    if input_array.ndim == 1:
        yield {
            str(key): input_array if key == "input_ids" else _plain(column)
            for key, column in value.items()
        }
        return
    if input_array.ndim != 2:
        raise ValueError("pretokenized input_ids must have one or two dimensions")
    row_count = int(input_array.shape[0])
    for row in range(row_count):
        yield _mapping_array_row(value, row=row, row_count=row_count)


def _mapping_array_row(
    value: Mapping[str, Any],
    *,
    row: int,
    row_count: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, column in value.items():
        if _is_row_column(column, row_count):
            row_value = _cpu_array(column)[row]
            output[str(key)] = row_value if key == "input_ids" else _plain(row_value)
        else:
            output[str(key)] = _plain(column)
    return output


def _plain(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    elif isinstance(value, np.ndarray):
        value = value.tolist()
    return value


def _cpu_array(value: Any) -> np.ndarray | torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, np.ndarray):
        return value
    return torch.as_tensor(value).cpu()


def _is_row_column(value: Any, row_count: int) -> bool:
    if isinstance(value, (str, bytes)) or not hasattr(value, "__len__"):
        return False
    try:
        return len(value) == row_count
    except TypeError:
        return False


def _datasets_loader() -> Any:
    try:
        from datasets import load_dataset
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "Hugging Face and Parquet sources require `pip install turbo-dllm[data]`"
        ) from exc
    return load_dataset


def _huggingface_records(spec: SourceSpec) -> Iterator[Mapping[str, Any]]:
    if not spec.id:
        raise ValueError("huggingface sources require source.id")
    positional = (spec.id,) if spec.name is None else (spec.id, spec.name)
    kwargs = dict(spec.loader_kwargs)
    kwargs.update({"split": spec.split, "streaming": bool(spec.streaming)})
    if spec.revision is not None:
        kwargs["revision"] = spec.revision
    dataset = _datasets_loader()(*positional, **kwargs)
    yield from _mapping_records(dataset, "Hugging Face")


def _parquet_records(
    spec: SourceSpec,
    resolved_paths: Sequence[Path] | None,
) -> Iterator[Mapping[str, Any]]:
    local_paths = (
        resolve_source_paths(spec) if resolved_paths is None else resolved_paths
    )
    paths = [str(path) for path in local_paths]
    if not paths:
        raise ValueError("parquet source contains no files")
    kwargs = dict(spec.loader_kwargs)
    kwargs.update({"data_files": paths, "split": spec.split})
    dataset = _datasets_loader()("parquet", **kwargs)
    yield from _mapping_records(dataset, "Parquet")


def _mapping_records(
    records: Iterable[Any],
    source_name: str,
) -> Iterator[Mapping[str, Any]]:
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"{source_name} record {index} must be a mapping")
        yield record


def _normalized(value: str) -> str:
    return str(value).strip().lower().replace("_", "-")


__all__ = ("iter_source_records", "source_provenance")
