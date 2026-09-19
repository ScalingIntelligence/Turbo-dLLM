from __future__ import annotations

from dllm_parallel import data
from dllm_parallel.core import data as core_data


def test_public_data_api_reexports_generic_runtime_contracts() -> None:
    assert data.DataBatch is core_data.DataBatch
    assert data.DataRuntime is core_data.DataRuntime
    assert data.IndexedDataset is core_data.IndexedSupervisedTokenDataRuntime
    assert data.PackedTokenDataset is core_data.PackedTokenDataRuntime
    assert data.build_data_runtime is core_data.build_standard_data_runtime


def test_public_data_api_is_explicit_and_dataset_agnostic() -> None:
    assert data.__all__ == (
        "DataBatch",
        "DataRuntime",
        "IndexedDataset",
        "PackedTokenDataset",
        "PreparationResult",
        "PreparationSpec",
        "build_data_runtime",
        "inspect_artifact",
        "prepare_dataset",
        "register_formatter",
        "register_source",
        "validate_artifact",
        "validate_artifact_for_run",
    )


def test_public_data_api_exports_preparation_frontend() -> None:
    spec = data.PreparationSpec.from_mapping(
        {
            "source": {"type": "pretokenized", "path": "tokens.i32"},
            "records": {"type": "pretokenized"},
            "packing": {"maximum_length": 8},
            "output": {"path": "prepared"},
        }
    )
    assert spec.source.type == "pretokenized"
    assert callable(data.prepare_dataset)
    assert callable(data.validate_artifact)
    assert callable(data.register_source)
