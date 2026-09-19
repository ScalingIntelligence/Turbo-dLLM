# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

from types import ModuleType, SimpleNamespace
import sys

import pytest
import torch

from dllm_parallel.core.models.backbones.qwen3_8.model import (
    _uses_te_tensor_parallel_overlap,
)
from dllm_parallel.core.parallel import transformer_engine as overlap


def test_qwen_te_overlap_requires_explicit_tp_sequence_parallel_policy() -> None:
    assert _uses_te_tensor_parallel_overlap(
        SimpleNamespace(
            tensor_parallel_overlap=True,
            sequence_parallel=True,
            tensor_parallel_size=2,
            block_parallel_size=4,
        )
    )
    assert not _uses_te_tensor_parallel_overlap(
        SimpleNamespace(
            tensor_parallel_overlap=False,
            sequence_parallel=True,
            tensor_parallel_size=2,
            block_parallel_size=4,
        )
    )
    assert not _uses_te_tensor_parallel_overlap(
        SimpleNamespace(
            sequence_parallel=True,
            tensor_parallel_size=2,
            block_parallel_size=4,
        )
    )


def test_qwen_te_overlap_excludes_pure_cp() -> None:
    common = {
        "tensor_parallel_overlap": True,
        "sequence_parallel": True,
        "tensor_parallel_size": 2,
        "configured_context_parallel_size": 4,
    }
    assert not _uses_te_tensor_parallel_overlap(
        SimpleNamespace(block_parallel_size=1, **common)
    )
    assert _uses_te_tensor_parallel_overlap(
        SimpleNamespace(block_parallel_size=4, **common)
    )


def test_te_userbuffers_initialize_once_and_validate_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    class _Mode:
        NONE = object()

    pytorch = ModuleType("transformer_engine.pytorch")
    pytorch.UserBufferQuantizationMode = _Mode
    pytorch.initialize_ub = lambda **kwargs: calls.append(kwargs)
    pytorch.destroy_ub = lambda: calls.append({"destroy": True})
    package = ModuleType("transformer_engine")
    package.pytorch = pytorch
    monkeypatch.setitem(sys.modules, "transformer_engine", package)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch", pytorch)
    monkeypatch.setattr(overlap, "_USERBUFFER_STATE", None)

    assert overlap.ensure_transformer_engine_userbuffers(
        rows=40960,
        hidden_size=5120,
        tensor_parallel_size=2,
        dtype=torch.bfloat16,
    )
    assert calls == [
        {
            "shape": [40960, 5120],
            "tp_size": 2,
            "quantization_modes": [_Mode.NONE],
            "dtype": torch.bfloat16,
            "bootstrap_backend": "nccl",
        }
    ]

    with pytest.raises(RuntimeError, match="incompatible shape or TP domain"):
        overlap.ensure_transformer_engine_userbuffers(
            rows=22528,
            hidden_size=5120,
            tensor_parallel_size=2,
            dtype=torch.bfloat16,
        )
    with pytest.raises(RuntimeError, match="incompatible shape or TP domain"):
        overlap.ensure_transformer_engine_userbuffers(
            rows=40961,
            hidden_size=5120,
            tensor_parallel_size=2,
            dtype=torch.bfloat16,
        )

    overlap.destroy_transformer_engine_userbuffers()
    assert calls[-1] == {"destroy": True}
    assert overlap.transformer_engine_userbuffer_state() is None
