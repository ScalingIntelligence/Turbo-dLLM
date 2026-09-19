# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Text-only Qwen3.8 block-diffusion backbone."""

from dllm_parallel.core.models.backbones.qwen3_8.executor import (
    Qwen38BackboneExecutor,
    build_executor,
    summarize_config,
)

_ARTIFACT_EXPORTS = {
    "Qwen38MergeResult",
    "merge_qwen38_lora_checkpoint",
    "validate_merged_qwen38_artifact",
}

__all__ = [
    "Qwen38BackboneExecutor",
    "build_executor",
    "summarize_config",
    "Qwen38MergeResult",
    "merge_qwen38_lora_checkpoint",
    "validate_merged_qwen38_artifact",
]


def __getattr__(name: str):
    if name in _ARTIFACT_EXPORTS:
        from dllm_parallel.core.models.backbones.qwen3_8 import export as module

        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(name)
