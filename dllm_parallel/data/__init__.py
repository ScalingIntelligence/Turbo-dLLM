# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Stable, dataset-agnostic data interfaces.

The implementations remain in :mod:`dllm_parallel.core.data` so the release
facade does not disturb the established training execution path.
"""

from dllm_parallel.core.data import (
    DataBatch,
    DataRuntime,
    IndexedSupervisedTokenDataRuntime as IndexedDataset,
    PackedTokenDataRuntime as PackedTokenDataset,
    build_standard_data_runtime as build_data_runtime,
)
from dllm_parallel.data.indexed import (
    inspect_artifact,
    validate_artifact,
    validate_artifact_for_run,
)
from dllm_parallel.data.prepare import PreparationResult, prepare_dataset
from dllm_parallel.data.registry import register_formatter, register_source
from dllm_parallel.data.schemas import PreparationSpec

__all__ = (
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
